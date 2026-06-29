/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.spark.rapids

import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong

import scala.collection.mutable
import scala.util.Try
import scala.util.control.NonFatal

import org.apache.spark.SparkContext
import org.apache.spark.internal.Logging
import org.apache.spark.scheduler.{SparkListener, SparkListenerJobStart, SparkListenerStageCompleted,
  SparkListenerStageSubmitted}
import org.apache.spark.sql.execution.SQLExecution
import org.apache.spark.sql.rapids.execution.TrampolineUtil

/**
 * Driver-side owner of the current per-stage hint map for the graph autotuner.
 *
 * Process-wide singleton (one driver plugin per JVM). It assigns monotonically increasing hint
 * versions, answers executor hint requests ([[handleHintRequest]]), and re-posts applied-hint
 * reports as [[SparkRapidsAutotuneHintAppliedEvent]] eventlog records ([[handleHintApplied]]).
 * When autotune is disabled it holds no hints and reports nothing. Tests must call [[init]] /
 * [[shutdown]] to manage the shared state.
 */
object RapidsAutotuneDriverEndpoint extends Logging {
  private val hints = new ConcurrentHashMap[AutotuneStageKey, StageRuntimeHint]()
  // Cumulative per-stage aggregate -- feeds the eventlog observation record and `observationFor`.
  // Monotonic by design (lifetime totals/high-water), so it is not the optimizer's window input.
  private val observations = new ConcurrentHashMap[AutotuneStageKey, StageObservationAgg]()
  private val nextHintVersion = new AtomicLong(1L)

  @volatile private var sparkContext: SparkContext = _
  @volatile private var enabled = false
  // The graph optimizer only runs in GRAPH/OPTIMIZE modes; OBSERVE/LOCAL never republish.
  @volatile private var optimizerEnabled = false
  @volatile private var optimizer: GraphWideAutotuneOptimizer = _
  // Injectable monotonic clock so optimizer update cadence is deterministically testable.
  @volatile private var nanoSource: () => Long = () => System.nanoTime()

  def init(sc: SparkContext, conf: RapidsConf): Unit = synchronized {
    enabled = conf.autotuneGraphEnabled
    optimizerEnabled = conf.isAutotuneClosedLoopMode
    sparkContext = if (enabled) sc else null
    optimizer = new AnalyticalGraphWideAutotuneOptimizer(
      GraphOptimizerConstraints.fromConf(conf))
    nanoSource = () => System.nanoTime()
    hints.clear()
    observations.clear()
    nextHintVersion.set(1L)
    if (enabled) {
      logInfo(s"Initialized RAPIDS graph autotune driver endpoint in " +
        s"${conf.autotuneGraphMode} mode")
    }
  }

  def shutdown(): Unit = synchronized {
    enabled = false
    optimizerEnabled = false
    sparkContext = null
    optimizer = null
    nanoSource = () => System.nanoTime()
    hints.clear()
    observations.clear()
    nextHintVersion.set(1L)
  }

  /** Test-only: inject a deterministic monotonic clock for optimizer update cadence. */
  private[rapids] def setNanoSourceForTest(source: () => Long): Unit = synchronized {
    nanoSource = source
  }

  def publishDefaultNoopHint(key: AutotuneStageKey): StageRuntimeHint = synchronized {
    publishStageHint(key, ScanRuntimeHint.empty, GpuRuntimeHint.empty)
  }

  /** Register a graph node and publish the optimizer's complete initial joint hint. */
  def publishOptimizerHint(
      key: AutotuneStageKey,
      descriptor: AutotuneStageDescriptor): StageRuntimeHint = synchronized {
    if (!enabled || optimizer == null) {
      StageRuntimeHint.empty(key)
    } else {
      val content = optimizer.initialHint(key, descriptor)
      publishStageHint(key, content.scan, content.gpu, content.shuffle, content.batch)
    }
  }

  def stageSubmitted(key: AutotuneStageKey): Unit = {
    if (enabled && optimizer != null) {
      optimizer.stageSubmitted(key)
    }
  }

  def stageCompleted(key: AutotuneStageKey): Unit = {
    if (enabled && optimizer != null) {
      optimizer.stageCompleted(key)
    }
  }

  /**
   * Publish (once) the hint for a stage key. The first publication for a key wins and fixes its
   * version; later calls for the same un-expired key return the existing hint so executors see a
   * stable version for the stage attempt.
   */
  def publishStageHint(
      key: AutotuneStageKey,
      scanHint: ScanRuntimeHint,
      gpuHint: GpuRuntimeHint = GpuRuntimeHint.empty,
      shuffleHint: ShuffleRuntimeHint = ShuffleRuntimeHint.empty,
      batchHint: BatchRuntimeHint = BatchRuntimeHint.empty): StageRuntimeHint = synchronized {
    if (!enabled) {
      StageRuntimeHint.empty(key)
    } else {
      Option(hints.get(key))
        .filterNot(_.isExpired(System.nanoTime()))
        .getOrElse {
          val hint = StageRuntimeHint(
            executionId = key.executionId,
            stageId = key.stageId,
            stageAttemptId = key.stageAttemptId,
            version = nextHintVersion.getAndIncrement(),
            scan = scanHint,
            gpu = gpuHint,
            shuffle = shuffleHint,
            batch = batchHint,
            expiresAtNanos = Long.MaxValue)
          hints.put(key, hint)
          hint
        }
    }
  }

  /**
   * Republish an adjusted hint for a key, bumping its version so executors that re-fetch (after
   * their cache TTL) see the newer hint. Unlike [[publishStageHint]] (first-publication-wins), this
   * overwrites the existing entry -- it is the graph optimizer's update path. The supplied
   * `content` carries the new knob values; this stamps a fresh version and keeps the stage-attempt
   * expiry unchanged (`Long.MaxValue`), so the wire-level expiry semantics are untouched and the
   * executor fetch-TTL is what drives pickup. Stays within the applicable static (GRAPH) or
   * explicit safety-envelope (OPTIMIZE) caps by construction.
   */
  def republishStageHint(key: AutotuneStageKey, content: StageRuntimeHint): StageRuntimeHint =
    synchronized {
      if (!enabled) {
        StageRuntimeHint.empty(key)
      } else {
        val hint = content.copy(
          executionId = key.executionId,
          stageId = key.stageId,
          stageAttemptId = key.stageAttemptId,
          version = nextHintVersion.getAndIncrement(),
          expiresAtNanos = Long.MaxValue)
        hints.put(key, hint)
        hint
      }
    }

  def handleHintRequest(msg: RapidsAutotuneHintRequestMsg): RapidsAutotuneHintResponseMsg = {
    val hint = if (enabled) {
      Option(hints.get(msg.key)).filterNot(_.isExpired(System.nanoTime()))
    } else {
      None
    }
    RapidsAutotuneHintResponseMsg(msg.key, hint)
  }

  def handleHintApplied(msg: RapidsAutotuneHintAppliedMsg): Unit = {
    if (!enabled || sparkContext == null) {
      return
    }
    TrampolineUtil.postEvent(sparkContext, toAppliedEvent(msg))
  }

  /**
   * Flatten an applied-hint report into its eventlog record. Pure (no SparkContext / endpoint
   * state required) so the per-field mapping can be unit-tested directly.
   */
  private[rapids] def toAppliedEvent(
      msg: RapidsAutotuneHintAppliedMsg): SparkRapidsAutotuneHintAppliedEvent =
    SparkRapidsAutotuneHintAppliedEvent(
      executorId = msg.executorId,
      executionId = msg.key.executionId,
      stageId = msg.key.stageId,
      stageAttemptId = msg.key.stageAttemptId,
      taskAttemptId = msg.taskAttemptId,
      partitionId = msg.partitionId,
      hintVersion = msg.hintVersion,
      hasHint = msg.hasHint,
      scanEagerPrefetch = msg.scan.eagerPrefetch,
      scanMinReadWindow = msg.scan.minReadWindow,
      scanMaxReadWindow = msg.scan.maxReadWindow,
      scanMaxReadyBytes = msg.scan.maxReadyBytes,
      gpuMaxConcurrentTasks = msg.gpu.maxConcurrentTasks,
      gpuAppliedMaxConcurrentTasks = msg.gpuAppliedMaxConcurrentTasks,
      shufflePrefetchWindow = msg.shuffle.prefetchWindow,
      shuffleMaxReadyBytes = msg.shuffle.maxReadyBytes,
      shuffleCoalesceTargetBytes = msg.shuffle.coalesceTargetBytes,
      batchTargetBatchBytes = msg.batch.targetBatchBytes,
      batchMaxBatchBytes = msg.batch.maxBatchBytes,
      batchSplitUntilSize = msg.batch.splitUntilSize)

  /**
   * Ingest a runtime observation: update the per-stage aggregate, post an eventlog record, and (in
   * a closed-loop mode) run the graph optimizer, possibly republishing a complete joint hint.
   */
  def handleObservation(msg: RapidsAutotuneObservationMsg): Unit = {
    if (!enabled) {
      return
    }
    val agg = observations.compute(msg.key, (_, prev) =>
      (if (prev == null) StageObservationAgg.empty else prev).merge(msg))
    if (sparkContext != null) {
      TrampolineUtil.postEvent(sparkContext, toObservationEvent(msg, agg))
    }
    maybeOptimize(msg)
  }

  /**
   * Closed-loop optimizer trigger. The optimizer owns its observation windows, update cadence,
   * stage graph and joint decision. This endpoint owns only wire publication/versioning.
   *
   * Fully guarded: an optimizer exception is swallowed (advisory path) so it can never break
   * observation handling for other stages or the driver RPC dispatcher.
   */
  private def maybeOptimize(msg: RapidsAutotuneObservationMsg): Unit = synchronized {
    if (!optimizerEnabled || optimizer == null) {
      return
    }
    try {
      val current = hints.get(msg.key)
      if (current == null) {
        return
      }
      optimizer.observe(msg, current, nanoSource()).foreach { decision =>
        val published = republishStageHint(msg.key, decision.hint)
        logDebug(s"RAPIDS graph optimizer re-hinted stage ${msg.key.stageId}." +
          s"${msg.key.stageAttemptId} to version ${published.version}; predicted task work " +
          s"${decision.predictedCurrentNanos} -> ${decision.predictedSelectedNanos} ns; " +
          s"scan ${published.scan}, GPU ${published.gpu}, shuffle ${published.shuffle}, " +
          s"batch ${published.batch}")
      }
    } catch {
      case NonFatal(e) =>
        logWarning("RAPIDS graph optimizer evaluation failed; leaving hint unchanged", e)
    }
  }

  /**
   * Cumulative per-stage observation aggregate for a key (eventlog/inspection view). NOTE: this is
   * the lifetime aggregate, not the recent window the graph optimizer consumes.
   */
  def observationFor(key: AutotuneStageKey): Option[StageObservationAgg] =
    Option(observations.get(key))

  private[rapids] def toObservationEvent(
      msg: RapidsAutotuneObservationMsg,
      agg: StageObservationAgg): SparkRapidsAutotuneObservationEvent =
    SparkRapidsAutotuneObservationEvent(
      executorId = msg.executorId,
      executionId = msg.key.executionId,
      stageId = msg.key.stageId,
      stageAttemptId = msg.key.stageAttemptId,
      taskAttemptId = msg.taskAttemptId,
      partitionId = msg.partitionId,
      hintVersion = msg.hintVersion,
      gpuSemaphoreWaitNanos = msg.gpuSemaphoreWaitNanos,
      gpuHoldingNanos = msg.gpuHoldingNanos,
      hostMemoryBytes = msg.hostMemoryBytes,
      stageTaskCount = agg.taskCount,
      stageMaxHostMemoryBytes = agg.maxHostMemoryBytes,
      spillBytes = msg.spillBytes,
      stageTotalSpillBytes = agg.totalSpillBytes,
      shuffleReadLimiterAcquireCount = msg.shuffleReadLimiterAcquireCount,
      shuffleReadLimiterAcquireFailCount = msg.shuffleReadLimiterAcquireFailCount,
      stageShuffleReadLimiterAcquireCount = agg.totalShuffleReadLimiterAcquireCount,
      stageShuffleReadLimiterAcquireFailCount = agg.totalShuffleReadLimiterAcquireFailCount,
      taskDurationNanos = msg.taskDurationNanos,
      inputBytes = msg.inputBytes,
      outputBytes = msg.outputBytes,
      shuffleReadBytes = msg.shuffleReadBytes,
      shuffleWriteBytes = msg.shuffleWriteBytes,
      inputRows = msg.inputRows,
      outputRows = msg.outputRows,
      pinnedMemoryBytes = msg.pinnedMemoryBytes,
      deviceMemoryBytes = msg.deviceMemoryBytes,
      retryOrLostTimeNanos = msg.retryOrLostTimeNanos,
      stageTotalTaskDurationNanos = agg.totalTaskDurationNanos,
      stageTotalInputBytes = agg.totalInputBytes,
      stageTotalOutputBytes = agg.totalOutputBytes,
      stageTotalShuffleReadBytes = agg.totalShuffleReadBytes,
      stageTotalShuffleWriteBytes = agg.totalShuffleWriteBytes,
      stageMaxPinnedMemoryBytes = agg.maxPinnedMemoryBytes,
      stageMaxDeviceMemoryBytes = agg.maxDeviceMemoryBytes,
      stageTotalRetryOrLostTimeNanos = agg.totalRetryOrLostTimeNanos)
}

/**
 * Driver-side running aggregate of executor observations for one stage attempt. Pure value type;
 * the graph optimizer reads it to derive measured resource work.
 */
case class StageObservationAgg(
    taskCount: Long,
    totalGpuSemaphoreWaitNanos: Long,
    totalGpuHoldingNanos: Long,
    maxHostMemoryBytes: Long,
    totalSpillBytes: Long,
    totalShuffleReadLimiterAcquireCount: Long = 0L,
    totalShuffleReadLimiterAcquireFailCount: Long = 0L,
    totalTaskDurationNanos: Long = 0L,
    totalInputBytes: Long = 0L,
    totalOutputBytes: Long = 0L,
    totalShuffleReadBytes: Long = 0L,
    totalShuffleWriteBytes: Long = 0L,
    totalInputRows: Long = 0L,
    totalOutputRows: Long = 0L,
    maxPinnedMemoryBytes: Long = 0L,
    maxDeviceMemoryBytes: Long = 0L,
    totalRetryOrLostTimeNanos: Long = 0L) {
  def merge(msg: RapidsAutotuneObservationMsg): StageObservationAgg = StageObservationAgg(
    taskCount = taskCount + 1L,
    totalGpuSemaphoreWaitNanos = totalGpuSemaphoreWaitNanos + msg.gpuSemaphoreWaitNanos,
    totalGpuHoldingNanos = totalGpuHoldingNanos + msg.gpuHoldingNanos,
    maxHostMemoryBytes = math.max(maxHostMemoryBytes, msg.hostMemoryBytes),
    totalSpillBytes = totalSpillBytes + msg.spillBytes,
    totalShuffleReadLimiterAcquireCount = totalShuffleReadLimiterAcquireCount +
      msg.shuffleReadLimiterAcquireCount,
    totalShuffleReadLimiterAcquireFailCount = totalShuffleReadLimiterAcquireFailCount +
      msg.shuffleReadLimiterAcquireFailCount,
    totalTaskDurationNanos = totalTaskDurationNanos + msg.taskDurationNanos,
    totalInputBytes = totalInputBytes + msg.inputBytes,
    totalOutputBytes = totalOutputBytes + msg.outputBytes,
    totalShuffleReadBytes = totalShuffleReadBytes + msg.shuffleReadBytes,
    totalShuffleWriteBytes = totalShuffleWriteBytes + msg.shuffleWriteBytes,
    totalInputRows = totalInputRows + msg.inputRows,
    totalOutputRows = totalOutputRows + msg.outputRows,
    maxPinnedMemoryBytes = math.max(maxPinnedMemoryBytes, msg.pinnedMemoryBytes),
    maxDeviceMemoryBytes = math.max(maxDeviceMemoryBytes, msg.deviceMemoryBytes),
    totalRetryOrLostTimeNanos = totalRetryOrLostTimeNanos + msg.retryOrLostTimeNanos)

  /**
   * GPU wait ratio = sum(semaphore wait) / sum(holding) across the stage's reported tasks -- a
   * time-weighted stage ratio (not a mean of per-task ratios), retained for eventlog diagnostics.
   */
  def gpuWaitRatio: Double =
    if (totalGpuHoldingNanos > 0L) {
      totalGpuSemaphoreWaitNanos.toDouble / totalGpuHoldingNanos
    } else {
      0.0
    }

  /** Fraction of multithreaded shuffle-reader limiter attempts deferred by the current bound. */
  def shuffleReadLimiterFailureRatio: Double =
    if (totalShuffleReadLimiterAcquireCount > 0L) {
      totalShuffleReadLimiterAcquireFailCount.toDouble /
        totalShuffleReadLimiterAcquireCount.toDouble
    } else {
      0.0
    }
}

object StageObservationAgg {
  val empty: StageObservationAgg = StageObservationAgg(0L, 0L, 0L, 0L, 0L)
}

/**
 * Driver `SparkListener` that publishes a [[StageRuntimeHint]] for each stage of a SQL execution
 * when its job starts, using the configured scan/GPU hint policies. Stages outside a SQL execution
 * (no execution id) are skipped.
 */
class RapidsAutotuneStageHintListener(conf: RapidsConf) extends SparkListener with Logging {
  private val stageKeys = mutable.HashMap.empty[(Int, Int), AutotuneStageKey]

  override def onJobStart(jobStart: SparkListenerJobStart): Unit = {
    val executionId = executionIdFrom(jobStart.properties)

    executionId.foreach { execId =>
      jobStart.stageInfos.foreach { stageInfo =>
        publishHintForStage(
          execId,
          stageInfo.stageId,
          stageInfo.attemptNumber(),
          AutotuneStageShape.fromStageInfo(stageInfo),
          stageInfo.parentIds)
      }
    }
  }

  override def onStageSubmitted(stageSubmitted: SparkListenerStageSubmitted): Unit = synchronized {
    val info = stageSubmitted.stageInfo
    val lookup = (info.stageId, info.attemptNumber())
    val key = stageKeys.get(lookup).orElse {
      executionIdFrom(stageSubmitted.properties).map { executionId =>
        publishHintForStage(
          executionId,
          info.stageId,
          info.attemptNumber(),
          AutotuneStageShape.fromStageInfo(info),
          info.parentIds).key
      }
    }
    key.foreach(RapidsAutotuneDriverEndpoint.stageSubmitted)
  }

  override def onStageCompleted(stageCompleted: SparkListenerStageCompleted): Unit = synchronized {
    val info = stageCompleted.stageInfo
    stageKeys.get((info.stageId, info.attemptNumber())).foreach(
      RapidsAutotuneDriverEndpoint.stageCompleted)
  }

  private[rapids] def publishDefaultHint(
      executionId: Long,
      stageId: Int,
      stageAttemptId: Int): StageRuntimeHint = {
    publishHintForStage(
      executionId,
      stageId,
      stageAttemptId,
      AutotuneStageShape(hasGpuScan = false, hasGpuPrefetchConsumer = false, numTasks = 0),
      Seq.empty)
  }

  private[rapids] def publishHintForStage(
      executionId: Long,
      stageId: Int,
      stageAttemptId: Int,
      stageShape: AutotuneStageShape,
      parentStageIds: Seq[Int] = Seq.empty): StageRuntimeHint = synchronized {
    val key = AutotuneStageKey(
      executionId = executionId,
      stageId = stageId,
      stageAttemptId = stageAttemptId)
    stageKeys.put((stageId, stageAttemptId), key)
    val hint = RapidsAutotuneDriverEndpoint.publishOptimizerHint(
      key, AutotuneStageDescriptor(stageShape, parentStageIds))
    if (hint.version > 0) {
      logDebug(s"Published RAPIDS graph autotune hint version ${hint.version} " +
        s"for execution $executionId, stage ${key.stageId}.${key.stageAttemptId}, " +
        s"scan hint ${hint.scan}, GPU hint ${hint.gpu}, shuffle hint ${hint.shuffle}, " +
        s"batch hint ${hint.batch}")
    }
    hint
  }

  private def executionIdFrom(properties: java.util.Properties): Option[Long] =
    Option(properties)
      .flatMap(p => Option(p.getProperty(SQLExecution.EXECUTION_ID_KEY)))
      .flatMap(v => Try(v.toLong).toOption)
}

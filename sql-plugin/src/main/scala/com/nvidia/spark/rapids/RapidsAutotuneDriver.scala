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

import scala.util.Try

import org.apache.spark.SparkContext
import org.apache.spark.internal.Logging
import org.apache.spark.scheduler.{SparkListener, SparkListenerJobStart}
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
  private val nextHintVersion = new AtomicLong(1L)

  @volatile private var sparkContext: SparkContext = _
  @volatile private var enabled = false

  def init(sc: SparkContext, conf: RapidsConf): Unit = synchronized {
    enabled = conf.autotuneGraphEnabled
    sparkContext = if (enabled) sc else null
    hints.clear()
    nextHintVersion.set(1L)
    if (enabled) {
      logInfo(s"Initialized RAPIDS graph autotune driver endpoint in " +
        s"${conf.autotuneGraphMode} mode")
    }
  }

  def shutdown(): Unit = synchronized {
    enabled = false
    sparkContext = null
    hints.clear()
    nextHintVersion.set(1L)
  }

  def publishDefaultNoopHint(key: AutotuneStageKey): StageRuntimeHint = synchronized {
    publishStageHint(key, ScanRuntimeHint.empty, GpuRuntimeHint.empty)
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
}

/**
 * Driver `SparkListener` that publishes a [[StageRuntimeHint]] for each stage of a SQL execution
 * when its job starts, using the configured scan/GPU hint policies. Stages outside a SQL execution
 * (no execution id) are skipped.
 */
class RapidsAutotuneStageHintListener(conf: RapidsConf) extends SparkListener with Logging {
  private val scanHintPolicy = GraphScanHintPolicy.fromConf(conf)
  private val gpuHintPolicy = GraphGpuHintPolicy.fromConf(conf)
  private val shuffleHintPolicy = GraphShuffleHintPolicy.fromConf(conf)
  private val batchHintPolicy = GraphBatchHintPolicy.fromConf(conf)

  override def onJobStart(jobStart: SparkListenerJobStart): Unit = {
    val executionId = Option(jobStart.properties)
      .flatMap(p => Option(p.getProperty(SQLExecution.EXECUTION_ID_KEY)))
      .flatMap(v => Try(v.toLong).toOption)

    executionId.foreach { execId =>
      jobStart.stageInfos.foreach { stageInfo =>
        publishHintForStage(
          execId,
          stageInfo.stageId,
          stageInfo.attemptNumber(),
          AutotuneStageShape.fromStageInfo(stageInfo))
      }
    }
  }

  private[rapids] def publishDefaultHint(
      executionId: Long,
      stageId: Int,
      stageAttemptId: Int): StageRuntimeHint = {
    publishHintForStage(
      executionId,
      stageId,
      stageAttemptId,
      AutotuneStageShape(hasGpuScan = false, hasGpuPrefetchConsumer = false, numTasks = 0))
  }

  private[rapids] def publishHintForStage(
      executionId: Long,
      stageId: Int,
      stageAttemptId: Int,
      stageShape: AutotuneStageShape): StageRuntimeHint = {
    val key = AutotuneStageKey(
      executionId = executionId,
      stageId = stageId,
      stageAttemptId = stageAttemptId)
    val hint = RapidsAutotuneDriverEndpoint.publishStageHint(
      key,
      scanHintPolicy.scanHintFor(stageShape),
      gpuHintPolicy.gpuHintFor(stageShape),
      shuffleHintPolicy.shuffleHintFor(stageShape),
      batchHintPolicy.batchHintFor(stageShape))
    if (hint.version > 0) {
      logDebug(s"Published RAPIDS graph autotune hint version ${hint.version} " +
        s"for execution $executionId, stage ${key.stageId}.${key.stageAttemptId}, " +
        s"scan hint ${hint.scan}, GPU hint ${hint.gpu}, shuffle hint ${hint.shuffle}, " +
        s"batch hint ${hint.batch}")
    }
    hint
  }
}

/**
 * Driver policy that emits a bounded scan-prefetch hint for stages classified as scan-prefetch
 * candidates, only in GRAPH mode. The hinted window never exceeds the static read-window cap.
 */
case class GraphScanHintPolicy(
    enabled: Boolean,
    maxReadWindow: Int,
    maxReadyBytes: Long) {
  def scanHintFor(stageShape: AutotuneStageShape): ScanRuntimeHint = {
    if (enabled && stageShape.isScanPrefetchCandidate && maxReadWindow > 0) {
      ScanRuntimeHint(
        eagerPrefetch = true,
        minReadWindow = ScanReadWindowSettings.MIN_WINDOW,
        maxReadWindow = maxReadWindow,
        maxReadyBytes = maxReadyBytes)
    } else {
      ScanRuntimeHint.empty
    }
  }
}

object GraphScanHintPolicy {
  def fromConf(conf: RapidsConf): GraphScanHintPolicy = {
    GraphScanHintPolicy(
      enabled = conf.autotuneGraphEnabled && conf.autotuneGraphMode == AutotuneGraphMode.GRAPH,
      maxReadWindow = conf.autotuneScanReadWindowCap,
      maxReadyBytes = conf.autotuneScanMaxReadyBytes)
  }
}

/**
 * Driver policy that emits a GPU admission hint (a per-stage max-concurrent-tasks cap), only in
 * GRAPH mode and only when a positive cap is configured. The executor admission controller can
 * only tighten the static GPU concurrency limit with it.
 */
case class GraphGpuHintPolicy(
    enabled: Boolean,
    maxConcurrentTasks: Int) {
  def gpuHintFor(stageShape: AutotuneStageShape): GpuRuntimeHint = {
    if (enabled && maxConcurrentTasks > 0 && stageShape.numTasks > 0) {
      GpuRuntimeHint(maxConcurrentTasks = maxConcurrentTasks)
    } else {
      GpuRuntimeHint.empty
    }
  }
}

object GraphGpuHintPolicy {
  def fromConf(conf: RapidsConf): GraphGpuHintPolicy = {
    GraphGpuHintPolicy(
      enabled = conf.autotuneGraphEnabled && conf.autotuneGraphMode == AutotuneGraphMode.GRAPH,
      maxConcurrentTasks = conf.autotuneGpuMaxConcurrentTasks)
  }
}

/**
 * Driver policy that emits a shuffle read/coalesce hint for shuffle-bearing stages, only in GRAPH
 * mode and only when explicitly enabled. The hint is currently observe-only: it is published and
 * recorded in the eventlog but no shuffle reader/coalesce actuator consumes it yet, so it never
 * changes execution. Default-off (the enable flag) keeps the published hint empty.
 */
case class GraphShuffleHintPolicy(
    enabled: Boolean,
    maxPrefetchWindow: Int,
    maxReadyBytes: Long,
    coalesceTargetBytes: Long) {
  def shuffleHintFor(stageShape: AutotuneStageShape): ShuffleRuntimeHint = {
    if (enabled && stageShape.isShuffleStage) {
      ShuffleRuntimeHint(
        prefetchWindow = maxPrefetchWindow,
        maxReadyBytes = maxReadyBytes,
        coalesceTargetBytes = coalesceTargetBytes)
    } else {
      ShuffleRuntimeHint.empty
    }
  }
}

object GraphShuffleHintPolicy {
  def fromConf(conf: RapidsConf): GraphShuffleHintPolicy = {
    GraphShuffleHintPolicy(
      enabled = conf.autotuneGraphEnabled && conf.autotuneGraphMode == AutotuneGraphMode.GRAPH &&
        conf.autotuneShuffleEnabled,
      maxPrefetchWindow = conf.autotuneShuffleMaxPrefetchWindow,
      maxReadyBytes = conf.autotuneShuffleMaxReadyBytes,
      coalesceTargetBytes = conf.autotuneShuffleCoalesceTargetBytes)
  }
}

/**
 * Driver policy that emits a batch-sizing hint for stages doing GPU work, only in GRAPH mode and
 * only when explicitly enabled. Observe-only like [[GraphShuffleHintPolicy]]: no coalesce/batch
 * actuator consumes it yet, so it never changes execution. Default-off keeps the hint empty.
 */
case class GraphBatchHintPolicy(
    enabled: Boolean,
    targetBatchBytes: Long,
    maxBatchBytes: Long,
    splitUntilSize: Long) {
  def batchHintFor(stageShape: AutotuneStageShape): BatchRuntimeHint = {
    if (enabled && stageShape.hasGpuWork) {
      BatchRuntimeHint(
        targetBatchBytes = targetBatchBytes,
        maxBatchBytes = maxBatchBytes,
        splitUntilSize = splitUntilSize)
    } else {
      BatchRuntimeHint.empty
    }
  }
}

object GraphBatchHintPolicy {
  def fromConf(conf: RapidsConf): GraphBatchHintPolicy = {
    GraphBatchHintPolicy(
      enabled = conf.autotuneGraphEnabled && conf.autotuneGraphMode == AutotuneGraphMode.GRAPH &&
        conf.autotuneBatchEnabled,
      targetBatchBytes = conf.autotuneBatchTargetBytes,
      maxBatchBytes = conf.autotuneBatchMaxBytes,
      splitUntilSize = conf.autotuneBatchSplitUntilSize)
  }
}

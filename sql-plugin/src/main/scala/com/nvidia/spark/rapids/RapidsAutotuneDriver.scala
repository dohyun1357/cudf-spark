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
import scala.util.control.NonFatal

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
  // Cumulative per-stage aggregate -- feeds the eventlog observation record and `observationFor`.
  // Monotonic by design (lifetime totals/high-water), so it is NOT the model's input.
  private val observations = new ConcurrentHashMap[AutotuneStageKey, StageObservationAgg]()
  // Recent-window aggregate the closed-loop model actually consumes: it accumulates observations
  // since the model last evaluated this key and is reset on consumption. A windowed view (rather
  // than the cumulative one) is required so transient pressure does not latch -- otherwise a single
  // spilled byte or host-memory high-water mark would pin the pressure signals forever and the AIMD
  // restore could never fire.
  private val modelWindows = new ConcurrentHashMap[AutotuneStageKey, StageObservationAgg]()
  // Per-key debounce/cooldown timestamps for the closed-loop model (Slice 2). Separate from the
  // hint map so the model's stateful oscillation controls do not leak into the wire hint.
  private val modelTiming = new ConcurrentHashMap[AutotuneStageKey, ModelTiming]()
  private val nextHintVersion = new AtomicLong(1L)

  @volatile private var sparkContext: SparkContext = _
  @volatile private var enabled = false
  // The closed-loop model only runs in GRAPH mode; OBSERVE/LOCAL never republish.
  @volatile private var modelEnabled = false
  @volatile private var caps: AutotuneModelCaps = AutotuneModelCaps(0, 0L, 0, Long.MaxValue)
  @volatile private var updateIntervalNanos = 0L
  @volatile private var cooldownNanos = 0L
  // Injectable monotonic clock so the debounce/cooldown logic is deterministically testable.
  @volatile private var nanoSource: () => Long = () => System.nanoTime()

  /**
   * Per-key model timing: when the model last evaluated the key (debounce) and when it last
   * *lowered* a knob (cooldown). `lastDecreaseNanos == Long.MinValue` means "no decrease yet" and
   * is checked before any subtraction so the cooldown comparison is safe even when the monotonic
   * clock is negative.
   */
  private case class ModelTiming(lastEvalNanos: Long, lastDecreaseNanos: Long)

  private val NoDecrease: Long = Long.MinValue

  def init(sc: SparkContext, conf: RapidsConf): Unit = synchronized {
    enabled = conf.autotuneGraphEnabled
    modelEnabled = conf.isAutotuneClosedLoopMode
    sparkContext = if (enabled) sc else null
    // GPU ceiling the model may raise toward. GRAPH bounds it at the static autotune cap; OPTIMIZE
    // raises it to the (higher) OPTIMIZE ceiling, so above-static increase is gated on a config the
    // operator set. If the OPTIMIZE ceiling is unset (0) it degrades to the static cap.
    val gpuCeiling = if (conf.isAutotuneOptimizeMode) {
      math.max(conf.autotuneGpuMaxConcurrentTasks, conf.autotuneOptimizeGpuMaxConcurrentTasks)
    } else {
      conf.autotuneGpuMaxConcurrentTasks
    }
    caps = AutotuneModelCaps(
      scanMaxReadWindow = conf.autotuneEffectiveScanReadWindowCap,
      scanMaxReadyBytes = conf.autotuneScanMaxReadyBytes,
      gpuMaxConcurrentTasks = gpuCeiling,
      minSampleTasks = conf.autotuneGraphMinSampleTasks.toLong,
      optimizeGpu = conf.isAutotuneOptimizeMode)
    updateIntervalNanos = conf.autotuneGraphUpdateIntervalMs.toLong * 1000000L
    // Cooldown after a decrease is two debounce intervals, so a knob is not raised the very next
    // interval after it was lowered (anti-flap hysteresis).
    cooldownNanos = updateIntervalNanos * 2L
    nanoSource = () => System.nanoTime()
    hints.clear()
    observations.clear()
    modelWindows.clear()
    modelTiming.clear()
    nextHintVersion.set(1L)
    if (enabled) {
      logInfo(s"Initialized RAPIDS graph autotune driver endpoint in " +
        s"${conf.autotuneGraphMode} mode")
    }
  }

  def shutdown(): Unit = synchronized {
    enabled = false
    modelEnabled = false
    sparkContext = null
    nanoSource = () => System.nanoTime()
    hints.clear()
    observations.clear()
    modelWindows.clear()
    modelTiming.clear()
    nextHintVersion.set(1L)
  }

  /** Test-only: inject a deterministic monotonic clock for the debounce/cooldown logic. */
  private[rapids] def setNanoSourceForTest(source: () => Long): Unit = synchronized {
    nanoSource = source
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

  /**
   * Republish an adjusted hint for a key, bumping its version so executors that re-fetch (after
   * their cache TTL) see the newer hint. Unlike [[publishStageHint]] (first-publication-wins), this
   * overwrites the existing entry -- it is the closed-loop model's update path. The supplied
   * `content` carries the new knob values; this stamps a fresh version and keeps the stage-attempt
   * expiry unchanged (`Long.MaxValue`), so the wire-level expiry semantics are untouched and the
   * executor fetch-TTL is what drives pickup. Stays within the static caps by construction (the
   * model never produces a value above them).
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
   * GRAPH mode) run the closed-loop model, possibly republishing an adjusted hint for the stage.
   */
  def handleObservation(msg: RapidsAutotuneObservationMsg): Unit = {
    if (!enabled) {
      return
    }
    val agg = observations.compute(msg.key, (_, prev) =>
      (if (prev == null) StageObservationAgg.empty else prev).merge(msg))
    if (modelEnabled) {
      modelWindows.compute(msg.key, (_, prev) =>
        (if (prev == null) StageObservationAgg.empty else prev).merge(msg))
    }
    if (sparkContext != null) {
      TrampolineUtil.postEvent(sparkContext, toObservationEvent(msg, agg))
    }
    maybeRetune(msg.key)
  }

  /**
   * Closed-loop model trigger (Slice 2). Runs on the observation path -- no extra thread -- and is
   * the only place that calls [[republishStageHint]]. Reads the RECENT-WINDOW aggregate (reset on
   * consumption) so transient pressure does not latch. Stateful oscillation controls live here:
   * a min-sample gate on the window, debounce (at most one evaluation per `updateIntervalNanos` per
   * key), and cooldown after a decrease (an increase is suppressed for `cooldownNanos` after the
   * last knob reduction). The model ([[AutotuneGraphModel.decide]]) contributes the stateless
   * controls (per-step clamp, prefer-reduce) and only ever returns hints within the static caps.
   *
   * Fully guarded: a model exception is swallowed (advisory path) so it can never break observation
   * handling for other stages or the driver RPC dispatcher.
   */
  private def maybeRetune(key: AutotuneStageKey): Unit = synchronized {
    if (!modelEnabled) {
      return
    }
    try {
      val current = hints.get(key)
      val window = modelWindows.get(key)
      // No published hint to adjust, or not enough fresh samples in the window yet.
      if (current == null || window == null || window.taskCount < caps.minSampleTasks) {
        return
      }
      val now = nanoSource()
      val timing = modelTiming.get(key)
      // Debounce: at most one evaluation per interval per key.
      if (timing != null && (now - timing.lastEvalNanos) < updateIntervalNanos) {
        return
      }
      // Consume the window: reset it so the next evaluation sees only newer observations.
      modelWindows.put(key, StageObservationAgg.empty)
      val priorDecrease = if (timing != null) timing.lastDecreaseNanos else NoDecrease
      var lastDecrease = priorDecrease
      AutotuneGraphModel.decide(window, current, caps).foreach { decision =>
        val inCooldown =
          priorDecrease != NoDecrease && (now - priorDecrease) < cooldownNanos
        // Decreases always allowed (subject to debounce); an increase waits out the cooldown.
        if (decision.isDecrease || !inCooldown) {
          val published = republishStageHint(key, decision.hint)
          if (decision.isDecrease) {
            lastDecrease = now
          }
          logDebug(s"RAPIDS graph autotune re-hinted stage ${key.stageId}.${key.stageAttemptId} " +
            s"to version ${published.version} (decrease=${decision.isDecrease}), " +
            s"scan ${published.scan}, GPU ${published.gpu}")
        }
      }
      modelTiming.put(key, ModelTiming(now, lastDecrease))
    } catch {
      case NonFatal(e) =>
        logWarning("RAPIDS graph autotune model evaluation failed; leaving hint unchanged", e)
    }
  }

  /**
   * Cumulative per-stage observation aggregate for a key (eventlog/inspection view). NOTE: this is
   * the lifetime aggregate, not the recent window the closed-loop model consumes.
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
      stageTotalSpillBytes = agg.totalSpillBytes)
}

/**
 * Driver-side running aggregate of executor observations for one stage attempt. Pure value type;
 * the closed-loop model reads it to derive per-stage pressure signals.
 */
case class StageObservationAgg(
    taskCount: Long,
    totalGpuSemaphoreWaitNanos: Long,
    totalGpuHoldingNanos: Long,
    maxHostMemoryBytes: Long,
    totalSpillBytes: Long) {
  def merge(msg: RapidsAutotuneObservationMsg): StageObservationAgg = StageObservationAgg(
    taskCount = taskCount + 1L,
    totalGpuSemaphoreWaitNanos = totalGpuSemaphoreWaitNanos + msg.gpuSemaphoreWaitNanos,
    totalGpuHoldingNanos = totalGpuHoldingNanos + msg.gpuHoldingNanos,
    maxHostMemoryBytes = math.max(maxHostMemoryBytes, msg.hostMemoryBytes),
    totalSpillBytes = totalSpillBytes + msg.spillBytes)

  /**
   * GPU wait ratio = sum(semaphore wait) / sum(holding) across the stage's reported tasks -- a
   * time-weighted stage ratio (not a mean of per-task ratios); a key pressure signal for the model.
   */
  def gpuWaitRatio: Double =
    if (totalGpuHoldingNanos > 0L) {
      totalGpuSemaphoreWaitNanos.toDouble / totalGpuHoldingNanos
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
 * candidates, in the closed-loop modes (GRAPH or OPTIMIZE). GRAPH bounds the window at the static
 * read-window cap; OPTIMIZE raises it to the effective (above-static) cap. The executor still
 * bounds the actual window by the per-task file count, so it never exceeds stock reader concurrency.
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
      enabled = conf.isAutotuneClosedLoopMode,
      // OPTIMIZE raises this above the static cap; the executor still bounds it by files-per-task.
      maxReadWindow = conf.autotuneEffectiveScanReadWindowCap,
      maxReadyBytes = conf.autotuneScanMaxReadyBytes)
  }
}

/**
 * Driver policy that emits a GPU admission hint (a per-stage max-concurrent-tasks cap), in the
 * closed-loop modes (GRAPH or OPTIMIZE) and only when a positive cap is configured. In GRAPH the
 * executor admission controller can only tighten the static GPU concurrency limit; in OPTIMIZE the
 * model may raise it above static (the semaphore permit pool remains the hard memory bound).
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
      enabled = conf.isAutotuneClosedLoopMode,
      maxConcurrentTasks = conf.autotuneGpuMaxConcurrentTasks)
  }
}

/**
 * Driver policy that emits a shuffle read/coalesce hint for shuffle-bearing stages, in the
 * closed-loop modes (GRAPH or OPTIMIZE) and only when explicitly enabled. Default-off (the enable
 * flag) keeps the published hint empty.
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
      enabled = conf.isAutotuneClosedLoopMode && conf.autotuneShuffleEnabled,
      maxPrefetchWindow = conf.autotuneShuffleMaxPrefetchWindow,
      maxReadyBytes = conf.autotuneShuffleMaxReadyBytes,
      coalesceTargetBytes = conf.autotuneShuffleCoalesceTargetBytes)
  }
}

/**
 * Driver policy that emits a batch-sizing hint for stages doing GPU work, in the closed-loop modes
 * (GRAPH or OPTIMIZE) and only when explicitly enabled. Observe-only: no coalesce/batch actuator
 * consumes it yet, so it never changes execution. Default-off keeps the hint empty.
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
      enabled = conf.isAutotuneClosedLoopMode && conf.autotuneBatchEnabled,
      targetBatchBytes = conf.autotuneBatchTargetBytes,
      maxBatchBytes = conf.autotuneBatchMaxBytes,
      splitUntilSize = conf.autotuneBatchSplitUntilSize)
  }
}

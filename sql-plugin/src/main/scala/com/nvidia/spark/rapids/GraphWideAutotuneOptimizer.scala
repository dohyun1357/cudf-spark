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

import scala.collection.mutable

/** Hard envelopes for the four actuator families controlled by the graph optimizer. */
case class GraphOptimizerConstraints(
    minSampleTasks: Long,
    updateIntervalNanos: Long,
    scan: ScanOptimizerBounds,
    gpu: GpuOptimizerBounds,
    shuffle: ShuffleOptimizerBounds,
    batch: BatchOptimizerBounds)

case class ScanOptimizerBounds(
    enabled: Boolean,
    initialReadWindow: Int,
    maxReadWindow: Int,
    maxReadyBytes: Long)

case class GpuOptimizerBounds(
    enabled: Boolean,
    initialConcurrentTasks: Int,
    maxConcurrentTasks: Int)

case class ShuffleOptimizerBounds(
    enabled: Boolean,
    initialPrefetchWindow: Int,
    maxPrefetchWindow: Int,
    initialReadyBytes: Long,
    maxReadyBytes: Long,
    initialCoalesceBytes: Long)

case class BatchOptimizerBounds(
    enabled: Boolean,
    initialTargetBytes: Long,
    maxBatchBytes: Long,
    initialSplitUntilSize: Long)

object GraphOptimizerConstraints {
  def fromConf(conf: RapidsConf): GraphOptimizerConstraints = {
    val gpuMaximum = if (conf.isAutotuneOptimizeMode) {
      math.max(conf.autotuneGpuMaxConcurrentTasks,
        conf.autotuneOptimizeGpuMaxConcurrentTasks)
    } else {
      conf.autotuneGpuMaxConcurrentTasks
    }
    GraphOptimizerConstraints(
      minSampleTasks = conf.autotuneGraphMinSampleTasks.toLong,
      updateIntervalNanos = conf.autotuneGraphUpdateIntervalMs.toLong * 1000000L,
      scan = ScanOptimizerBounds(
        enabled = conf.isAutotuneClosedLoopMode,
        // Start at the static envelope. OPTIMIZE makes the larger value part of the feasible
        // domain; it is the optimizer, rather than an initial per-knob policy, that selects it.
        initialReadWindow = conf.autotuneScanReadWindowCap,
        maxReadWindow = conf.autotuneEffectiveScanReadWindowCap,
        maxReadyBytes = conf.autotuneScanMaxReadyBytes),
      gpu = GpuOptimizerBounds(
        enabled = conf.isAutotuneClosedLoopMode && conf.autotuneGpuMaxConcurrentTasks > 0,
        initialConcurrentTasks = conf.autotuneGpuMaxConcurrentTasks,
        maxConcurrentTasks = gpuMaximum),
      shuffle = ShuffleOptimizerBounds(
        enabled = conf.isAutotuneClosedLoopMode && conf.autotuneShuffleEnabled,
        initialPrefetchWindow = 1,
        maxPrefetchWindow = conf.autotuneShuffleMaxPrefetchWindow,
        initialReadyBytes = conf.autotuneInitialShuffleMaxReadyBytes,
        maxReadyBytes = conf.autotuneEffectiveShuffleMaxBytesInFlight,
        initialCoalesceBytes = conf.autotuneShuffleCoalesceTargetBytes),
      batch = BatchOptimizerBounds(
        enabled = conf.isAutotuneClosedLoopMode && conf.autotuneBatchEnabled,
        initialTargetBytes = conf.autotuneBatchTargetBytes,
        maxBatchBytes = conf.autotuneBatchMaxBytes,
        initialSplitUntilSize = conf.autotuneBatchSplitUntilSize))
  }
}

case class AutotuneStageDescriptor(
    shape: AutotuneStageShape,
    parentStageIds: Seq[Int] = Seq.empty)

case class GraphOptimizerDecision(
    hint: StageRuntimeHint,
    predictedCurrentNanos: Double,
    predictedSelectedNanos: Double)

/**
 * Single owner of initial and updated joint [[StageRuntimeHint]] values.
 *
 * Implementations receive graph lifecycle and raw task observations. They return complete hints;
 * scan, GPU, shuffle and batch decisions are never delegated to independent knob policies.
 */
trait GraphWideAutotuneOptimizer {
  def initialHint(key: AutotuneStageKey, descriptor: AutotuneStageDescriptor): StageRuntimeHint

  def observe(
      msg: RapidsAutotuneObservationMsg,
      current: StageRuntimeHint,
      nowNanos: Long): Option[GraphOptimizerDecision]

  def stageSubmitted(key: AutotuneStageKey): Unit

  def stageCompleted(key: AutotuneStageKey): Unit
}

/**
 * Driver graph optimizer backed by a constrained analytical stage-cost model.
 *
 * The optimizer uses discrete coordinate descent over one joint objective. Small integer domains
 * are enumerated exactly; large integer and byte domains use a logarithmic representation plus
 * every configured/current endpoint. The objective is predicted critical-path service time,
 * calibrated from measured task elapsed time and resource work. Spill and retry are costs in that
 * same objective, not fixed pressure thresholds. User configuration and executor clamps define
 * feasibility, not policy.
 */
class AnalyticalGraphWideAutotuneOptimizer(
    constraints: GraphOptimizerConstraints) extends GraphWideAutotuneOptimizer {

  private case class StageState(
      descriptor: AutotuneStageDescriptor,
      var active: Boolean,
      var window: StageObservationAgg,
      var windowHintVersion: Long,
      var lastEvaluationNanos: Long)

  private val stages = mutable.HashMap.empty[AutotuneStageKey, StageState]

  override def initialHint(
      key: AutotuneStageKey,
      descriptor: AutotuneStageDescriptor): StageRuntimeHint = synchronized {
    stages.put(key, StageState(descriptor, active = false, StageObservationAgg.empty,
      0L, Long.MinValue))
    initialHintContent(key, descriptor.shape)
  }

  override def observe(
      msg: RapidsAutotuneObservationMsg,
      current: StageRuntimeHint,
      nowNanos: Long): Option[GraphOptimizerDecision] = synchronized {
    stages.get(msg.key).flatMap { state =>
      // A cost calibration is valid only for the complete joint hint that produced its samples.
      // Tasks from an older version can finish after a re-hint; mixing those observations with the
      // new candidate would attribute elapsed/resource work to settings the task never used.
      if (msg.hintVersion != current.version) {
        return None
      }
      if (state.windowHintVersion != msg.hintVersion) {
        state.window = StageObservationAgg.empty
        state.windowHintVersion = msg.hintVersion
      }
      state.window = state.window.merge(msg)
      val enoughSamples = state.window.taskCount >= constraints.minSampleTasks
      val intervalElapsed = state.lastEvaluationNanos == Long.MinValue ||
        nowNanos - state.lastEvaluationNanos >= constraints.updateIntervalNanos
      if (!enoughSamples || !intervalElapsed) {
        None
      } else {
        val observation = state.window
        state.window = StageObservationAgg.empty
        state.lastEvaluationNanos = nowNanos
        AnalyticalStageCostModel.optimize(
          observation, current, state.descriptor.shape, constraints)
      }
    }
  }

  override def stageSubmitted(key: AutotuneStageKey): Unit = synchronized {
    stages.get(key).foreach(_.active = true)
  }

  override def stageCompleted(key: AutotuneStageKey): Unit = synchronized {
    stages.get(key).foreach(_.active = false)
  }

  private def initialHintContent(
      key: AutotuneStageKey,
      shape: AutotuneStageShape): StageRuntimeHint = {
    val scanHint = if (constraints.scan.enabled && shape.isScanPrefetchCandidate &&
        constraints.scan.initialReadWindow > 0) {
      ScanRuntimeHint(
        eagerPrefetch = true,
        minReadWindow = ScanReadWindowSettings.MIN_WINDOW,
        maxReadWindow = constraints.scan.initialReadWindow,
        maxReadyBytes = constraints.scan.maxReadyBytes)
    } else {
      ScanRuntimeHint.empty
    }
    val gpuHint = if (constraints.gpu.enabled && shape.numTasks > 0) {
      GpuRuntimeHint(constraints.gpu.initialConcurrentTasks)
    } else {
      GpuRuntimeHint.empty
    }
    val shuffleHint = if (constraints.shuffle.enabled && shape.isShuffleStage) {
      ShuffleRuntimeHint(
        prefetchWindow = constraints.shuffle.initialPrefetchWindow,
        maxReadyBytes = constraints.shuffle.initialReadyBytes,
        coalesceTargetBytes = constraints.shuffle.initialCoalesceBytes)
    } else {
      ShuffleRuntimeHint.empty
    }
    val batchHint = if (constraints.batch.enabled && shape.hasGpuWork) {
      BatchRuntimeHint(
        targetBatchBytes = constraints.batch.initialTargetBytes,
        maxBatchBytes = constraints.batch.maxBatchBytes,
        splitUntilSize = constraints.batch.initialSplitUntilSize)
    } else {
      BatchRuntimeHint.empty
    }
    StageRuntimeHint.empty(key).copy(
      scan = scanHint,
      gpu = gpuHint,
      shuffle = shuffleHint,
      batch = batchHint)
  }
}

/** Pure constrained optimizer used by [[AnalyticalGraphWideAutotuneOptimizer]]. */
object AnalyticalStageCostModel {
  private case class Candidate(
      scanWindow: Int,
      gpuTasks: Int,
      shuffleWindow: Int,
      shuffleBytes: Long,
      batchBytes: Long)

  private case class CalibratedWork(
      scanNanosAtCurrent: Double,
      gpuHoldingNanos: Double,
      gpuWaitNanos: Double,
      shuffleNanosAtCurrent: Double,
      batchOverheadNanosAtCurrent: Double,
      fixedNanos: Double,
      retryNanos: Double,
      spillNanosAtCurrent: Double,
      currentScanWindow: Int,
      currentGpuTasks: Int,
      currentShuffleWindow: Int,
      currentShuffleBytes: Long,
      currentBatchBytes: Long,
      processedBytes: Long)

  def optimize(
      observation: StageObservationAgg,
      current: StageRuntimeHint,
      shape: AutotuneStageShape,
      constraints: GraphOptimizerConstraints): Option[GraphOptimizerDecision] = {
    if (observation.taskCount < constraints.minSampleTasks ||
        observation.totalTaskDurationNanos <= 0L) {
      return None
    }

    val work = calibrate(observation, current, shape)
    val currentCandidate = Candidate(
      activeScanWindow(current), activeGpuTasks(current), activeShuffleWindow(current),
      activeShuffleBytes(current), activeBatchBytes(current))
    val scanValues = intDomain(
      currentCandidate.scanWindow,
      if (current.scan.maxReadWindow > 0 && work.scanNanosAtCurrent > 0.0) {
        constraints.scan.maxReadWindow
      } else 0)
    val gpuValues = intDomain(
      currentCandidate.gpuTasks,
      if (current.gpu.maxConcurrentTasks > 0 &&
          work.gpuHoldingNanos + work.gpuWaitNanos > 0.0) {
        constraints.gpu.maxConcurrentTasks
      } else 0)
    val shuffleValues = byteDomain(
      constraints.shuffle.initialReadyBytes,
      currentCandidate.shuffleBytes,
      if (isActiveShuffle(current) && work.shuffleNanosAtCurrent > 0.0) {
        constraints.shuffle.maxReadyBytes
      } else 0L)
    val shuffleWindowValues = intDomain(
      currentCandidate.shuffleWindow,
      if (isActiveShuffle(current) && work.shuffleNanosAtCurrent > 0.0) {
        constraints.shuffle.maxPrefetchWindow
      } else 0)
    val batchValues = byteDomain(
      constraints.batch.initialTargetBytes,
      currentCandidate.batchBytes,
      if (current.batch.targetBatchBytes > 0L && work.batchOverheadNanosAtCurrent > 0.0) {
        constraints.batch.maxBatchBytes
      } else 0L)

    var selected = currentCandidate
    var selectedCost = predictedNanos(work, currentCandidate)
    def descend(preferMoreOnTie: Boolean): Unit = {
      var changed = true
      while (changed) {
        changed = false
        def choose(candidates: Seq[Candidate]): Unit = candidates.foreach { candidate =>
          val cost = predictedNanos(work, candidate)
          val preferredTie = cost == selectedCost &&
            (if (preferMoreOnTie) lessResource(selected, candidate)
            else lessResource(candidate, selected))
          if (cost < selectedCost || preferredTie) {
            selected = candidate
            selectedCost = cost
            changed = true
          }
        }
        choose(scanValues.map(value => selected.copy(scanWindow = value)))
        choose(gpuValues.map(value => selected.copy(gpuTasks = value)))
        choose(shuffleWindowValues.map(value => selected.copy(shuffleWindow = value)))
        choose(shuffleValues.map(value => selected.copy(shuffleBytes = value)))
        // These two controls form one shuffle-service capacity, so optimize their pair exactly.
        choose(for {
          window <- shuffleWindowValues
          bytes <- shuffleValues
        } yield selected.copy(shuffleWindow = window, shuffleBytes = bytes))
        choose(batchValues.map(value => selected.copy(batchBytes = value)))
      }
    }
    // First cross equal-cost max-resource plateaus to find the minimum modeled stage time; then
    // remove any capacity that does not contribute to that optimum.
    descend(preferMoreOnTie = true)
    descend(preferMoreOnTie = false)

    if (selected == currentCandidate) {
      None
    } else {
      val selectedHint = current.copy(
        scan = if (current.scan.maxReadWindow > 0) {
          current.scan.copy(maxReadWindow = selected.scanWindow)
        } else current.scan,
        gpu = if (current.gpu.maxConcurrentTasks > 0) {
          current.gpu.copy(maxConcurrentTasks = selected.gpuTasks)
        } else current.gpu,
        shuffle = if (isActiveShuffle(current)) {
          current.shuffle.copy(
            prefetchWindow = selected.shuffleWindow,
            maxReadyBytes = selected.shuffleBytes,
            coalesceTargetBytes = selectedCoalesceBytes(current, selected))
        } else current.shuffle,
        batch = if (current.batch.targetBatchBytes > 0L &&
            selected.batchBytes != currentCandidate.batchBytes) {
          current.batch.copy(
            targetBatchBytes = selected.batchBytes,
            splitUntilSize = selectedSplitSize(current, selected.batchBytes))
        } else current.batch)
      Some(GraphOptimizerDecision(
        selectedHint,
        predictedCurrentNanos = predictedNanos(work, currentCandidate),
        predictedSelectedNanos = selectedCost))
    }
  }

  private def calibrate(
      obs: StageObservationAgg,
      current: StageRuntimeHint,
      shape: AutotuneStageShape): CalibratedWork = {
    val elapsed = math.max(obs.totalTaskDurationNanos.toDouble,
      (obs.totalGpuHoldingNanos + obs.totalGpuSemaphoreWaitNanos +
        obs.totalRetryOrLostTimeNanos).toDouble)
    val retry = obs.totalRetryOrLostTimeNanos.toDouble
    val availableNonGpu = math.max(0.0,
      elapsed - obs.totalGpuHoldingNanos - obs.totalGpuSemaphoreWaitNanos - retry)
    val scanBytes = if (shape.hasGpuScan) {
      math.max(0L, obs.totalInputBytes - obs.totalShuffleReadBytes)
    } else {
      0L
    }
    val shuffleBytes = if (shape.hasShuffle) {
      obs.totalShuffleReadBytes + obs.totalShuffleWriteBytes
    } else {
      0L
    }
    val classifiedBytes = scanBytes + shuffleBytes
    val scanShare = if (classifiedBytes > 0L) scanBytes.toDouble / classifiedBytes else 0.0
    val shuffleShare = if (classifiedBytes > 0L) {
      shuffleBytes.toDouble / classifiedBytes
    } else {
      0.0
    }
    val classifiedNanos = if (classifiedBytes > 0L) availableNonGpu else 0.0
    val scanNanos = classifiedNanos * scanShare
    val shuffleNanos = classifiedNanos * shuffleShare
    val unclassifiedNanos = math.max(0.0, availableNonGpu - scanNanos - shuffleNanos)
    val processedBytes = math.max(obs.totalInputBytes, obs.totalOutputBytes)
    val currentBatch = activeBatchBytes(current)
    // If batch sizing is active, the unclassified per-task overhead is modeled by the number of
    // target-sized batches. This is identifiable from measured elapsed work and bytes; no fixed
    // "good batch size" or tuning threshold is embedded in the optimizer.
    val batchOverhead = if (currentBatch > 0L && processedBytes > 0L) {
      unclassifiedNanos
    } else {
      0.0
    }
    CalibratedWork(
      scanNanosAtCurrent = scanNanos,
      gpuHoldingNanos = obs.totalGpuHoldingNanos.toDouble,
      gpuWaitNanos = obs.totalGpuSemaphoreWaitNanos.toDouble,
      shuffleNanosAtCurrent = shuffleNanos,
      batchOverheadNanosAtCurrent = batchOverhead,
      fixedNanos = math.max(0.0, unclassifiedNanos - batchOverhead),
      retryNanos = retry,
      spillNanosAtCurrent = spillTimeEquivalent(obs, availableNonGpu),
      currentScanWindow = activeScanWindow(current),
      currentGpuTasks = activeGpuTasks(current),
      currentShuffleWindow = activeShuffleWindow(current),
      currentShuffleBytes = activeShuffleBytes(current),
      currentBatchBytes = currentBatch,
      processedBytes = processedBytes)
  }

  private def predictedNanos(work: CalibratedWork, candidate: Candidate): Double = {
    val scan = scaleInverse(
      work.scanNanosAtCurrent, work.currentScanWindow, candidate.scanWindow)
    val gpuHolding = scaleInverse(
      work.gpuHoldingNanos, work.currentGpuTasks, candidate.gpuTasks)
    // Semaphore wait is admission queueing caused by the current task-count cap. Increasing the
    // cap drains that queue; device-memory permits remain the independent hard safety constraint.
    // Any real cost of additional load is represented by measured spill/retry below rather than by
    // treating queueing itself as contention in the opposite direction.
    val gpuWait = scaleInverse(
      work.gpuWaitNanos, work.currentGpuTasks, candidate.gpuTasks)
    val shuffle = scaleInverse(
      scaleInverse(work.shuffleNanosAtCurrent,
        work.currentShuffleBytes, candidate.shuffleBytes),
      work.currentShuffleWindow, candidate.shuffleWindow)
    val batch = batchOverhead(work, candidate.batchBytes)
    val loadScale = resourceLoadScale(work, candidate)
    val pressure = work.retryNanos + work.spillNanosAtCurrent * loadScale
    math.max(math.max(scan, gpuHolding + gpuWait), shuffle) + batch + work.fixedNanos + pressure
  }

  private def spillTimeEquivalent(obs: StageObservationAgg, nonGpuNanos: Double): Double = {
    val ioBytes = obs.totalInputBytes + obs.totalOutputBytes +
      obs.totalShuffleReadBytes + obs.totalShuffleWriteBytes
    if (obs.totalSpillBytes > 0L && ioBytes > 0L && nonGpuNanos > 0.0) {
      nonGpuNanos * obs.totalSpillBytes.toDouble / ioBytes.toDouble
    } else if (obs.totalSpillBytes > 0L) {
      obs.totalTaskDurationNanos.toDouble
    } else {
      0.0
    }
  }

  private def batchOverhead(work: CalibratedWork, candidateBatchBytes: Long): Double = {
    if (work.batchOverheadNanosAtCurrent <= 0.0 || work.currentBatchBytes <= 0L ||
        candidateBatchBytes <= 0L || work.processedBytes <= 0L) {
      work.batchOverheadNanosAtCurrent
    } else {
      val currentBatches = ceilDiv(work.processedBytes, work.currentBatchBytes)
      val candidateBatches = ceilDiv(work.processedBytes, candidateBatchBytes)
      work.batchOverheadNanosAtCurrent * candidateBatches.toDouble / currentBatches.toDouble
    }
  }

  private def ceilDiv(value: Long, divisor: Long): Long = {
    if (value <= 0L) 1L else 1L + (value - 1L) / math.max(1L, divisor)
  }

  private def intDomain(current: Int, maximum: Int): Seq[Int] = {
    if (current <= 0 || maximum <= 0) {
      Seq(current)
    } else if (maximum <= 64) {
      (1 to maximum).toSeq
    } else {
      val values = mutable.TreeSet.empty[Int]
      values += 1
      values += math.min(current, maximum)
      values += maximum
      var value = 1
      while (value < maximum && value <= Integer.MAX_VALUE / 2) {
        value = math.min(maximum, value * 2)
        values += value
      }
      values.toSeq
    }
  }

  private def byteDomain(minimum: Long, current: Long, maximum: Long): Seq[Long] = {
    if (minimum <= 0L || current <= 0L || maximum <= 0L || maximum == Long.MaxValue) {
      Seq(current)
    } else {
      val lower = math.min(minimum, maximum)
      val values = mutable.TreeSet.empty[Long]
      values += lower
      values += math.min(current, maximum)
      values += maximum
      var value = lower
      while (value < maximum && value <= Long.MaxValue / 2L) {
        value = math.min(maximum, value * 2L)
        values += value
      }
      values.toSeq
    }
  }

  private def scaleInverse(value: Double, current: Long, candidate: Long): Double = {
    if (value <= 0.0 || current <= 0L || candidate <= 0L) value
    else value * current.toDouble / candidate.toDouble
  }

  private def ratio(candidate: Long, current: Long): Double = {
    if (candidate <= 0L || current <= 0L) 1.0 else candidate.toDouble / current.toDouble
  }

  private def resourceLoadScale(work: CalibratedWork, candidate: Candidate): Double = {
    val ratios = Seq(
      if (work.currentScanWindow > 0) {
        Some(ratio(candidate.scanWindow, work.currentScanWindow))
      } else None,
      if (work.currentGpuTasks > 0) {
        Some(ratio(candidate.gpuTasks, work.currentGpuTasks))
      } else None,
      if (work.currentShuffleWindow > 0) {
        Some(ratio(candidate.shuffleWindow, work.currentShuffleWindow))
      } else None,
      if (work.currentShuffleBytes > 0L) {
        Some(ratio(candidate.shuffleBytes, work.currentShuffleBytes))
      } else None,
      if (work.currentBatchBytes > 0L) {
        Some(ratio(candidate.batchBytes, work.currentBatchBytes))
      } else None).flatten
    if (ratios.isEmpty) 1.0 else ratios.sum / ratios.size.toDouble
  }

  private def lessResource(left: Candidate, right: Candidate): Boolean = {
    val leftTotal = left.scanWindow.toDouble + left.gpuTasks.toDouble +
      left.shuffleWindow.toDouble + left.shuffleBytes.toDouble + left.batchBytes.toDouble
    val rightTotal = right.scanWindow.toDouble + right.gpuTasks.toDouble +
      right.shuffleWindow.toDouble + right.shuffleBytes.toDouble + right.batchBytes.toDouble
    leftTotal < rightTotal
  }

  private def activeScanWindow(hint: StageRuntimeHint): Int =
    if (hint.scan.maxReadWindow > 0) hint.scan.maxReadWindow else 0

  private def activeGpuTasks(hint: StageRuntimeHint): Int =
    if (hint.gpu.maxConcurrentTasks > 0) hint.gpu.maxConcurrentTasks else 0

  private def isActiveShuffle(hint: StageRuntimeHint): Boolean =
    hint.shuffle.maxReadyBytes > 0L && hint.shuffle.maxReadyBytes != Long.MaxValue

  private def activeShuffleWindow(hint: StageRuntimeHint): Int =
    if (isActiveShuffle(hint)) math.max(1, hint.shuffle.prefetchWindow) else 0

  private def activeShuffleBytes(hint: StageRuntimeHint): Long =
    if (isActiveShuffle(hint)) hint.shuffle.maxReadyBytes else 0L

  private def activeBatchBytes(hint: StageRuntimeHint): Long =
    if (hint.batch.targetBatchBytes > 0L) hint.batch.targetBatchBytes else 0L

  private def selectedCoalesceBytes(current: StageRuntimeHint, selected: Candidate): Long = {
    if (current.shuffle.coalesceTargetBytes > 0L && selected.batchBytes > 0L) {
      selected.batchBytes
    } else {
      current.shuffle.coalesceTargetBytes
    }
  }

  private def selectedSplitSize(current: StageRuntimeHint, selectedBatchBytes: Long): Long = {
    if (current.batch.splitUntilSize > 0L) {
      math.min(current.batch.maxBatchBytes, selectedBatchBytes)
    } else {
      current.batch.splitUntilSize
    }
  }
}

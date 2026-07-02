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

/** Hard envelopes for the runtime actuator families controlled by the graph optimizer. */
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
    initialSplitUntilSize: Long,
    minimumTargetBytes: Long = 0L)

object GraphOptimizerConstraints {
  def fromConf(
      conf: RapidsConf,
      nativeGpuTaskSlots: Int = 0): GraphOptimizerConstraints = {
    val configuredGpuMaximum = if (conf.isAutotuneOptimizeMode) {
      math.max(conf.autotuneGpuMaxConcurrentTasks,
        conf.autotuneOptimizeGpuMaxConcurrentTasks)
    } else {
      conf.autotuneGpuMaxConcurrentTasks
    }
    // concurrentGpuTasks seeds RAPIDS' dynamic memory-permit estimator; it is not a hard task
    // limit. A cold graph hint must therefore allow every Spark task slot that stock RAPIDS could
    // admit. The existing permit pool remains the authoritative memory-safety boundary.
    val nativeGpuMaximum = if (nativeGpuTaskSlots > 0) {
      nativeGpuTaskSlots
    } else {
      conf.autotuneGpuMaxConcurrentTasks
    }
    val gpuMaximum = math.max(nativeGpuMaximum, configuredGpuMaximum)
    val nativeBatchTarget = conf.gpuTargetBatchSizeBytes
    val configuredMinimumTargets = Seq(
      conf.autotuneBatchTargetBytes,
      conf.autotuneShuffleCoalesceTargetBytes).filter(_ > 0L)
    val minimumBatchTarget = (nativeBatchTarget +: configuredMinimumTargets).min
    // The native target is already a deployed-safe point and must never be excluded by an
    // autotune envelope. Long.MaxValue is the unset sentinel, not a useful search endpoint.
    val configuredBatchMaximum = conf.autotuneBatchMaxBytes
    val batchMaximum = if (configuredBatchMaximum == Long.MaxValue) {
      nativeBatchTarget
    } else {
      math.max(nativeBatchTarget, configuredBatchMaximum)
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
        initialConcurrentTasks = nativeGpuMaximum,
        maxConcurrentTasks = gpuMaximum),
      shuffle = ShuffleOptimizerBounds(
        enabled = conf.isAutotuneClosedLoopMode && conf.autotuneShuffleEnabled,
        // A cold stage begins at the configured feasible envelope. Window 1 was an unevidenced
        // policy choice and could serialize a short stage before the first feedback epoch.
        initialPrefetchWindow = conf.autotuneShuffleMaxPrefetchWindow,
        maxPrefetchWindow = conf.autotuneShuffleMaxPrefetchWindow,
        initialReadyBytes = conf.autotuneInitialShuffleMaxReadyBytes,
        maxReadyBytes = conf.autotuneEffectiveShuffleMaxBytesInFlight,
        // Shuffle coalescing targets the same native GPU batch size. A smaller configured value
        // is a model candidate, never an unconditional cold-stage override.
        initialCoalesceBytes = nativeBatchTarget),
      batch = BatchOptimizerBounds(
        enabled = conf.isAutotuneClosedLoopMode && conf.autotuneBatchEnabled,
        initialTargetBytes = nativeBatchTarget,
        maxBatchBytes = batchMaximum,
        initialSplitUntilSize = conf.autotuneBatchSplitUntilSize,
        minimumTargetBytes = minimumBatchTarget))
  }
}

case class AutotuneStageDescriptor(
    shape: AutotuneStageShape,
    parentStageIds: Seq[Int] = Seq.empty)

case class GraphOptimizerDecision(
    hint: StageRuntimeHint,
    predictedCurrentNanos: Double,
    predictedSelectedNanos: Double)

private[rapids] case class GraphControlFreezeReasons(
    scanWindow: String = "",
    gpuTasks: String = "",
    shuffleWindow: String = "",
    shuffleBytes: String = "",
    batchBytes: String = "")

private[rapids] case class GraphStageAnalyticalState(
    currentControl: GpuFlowControl,
    currentGradient: GpuFlowGradient,
    bounds: GpuFlowControlBounds,
    freezeReasons: GraphControlFreezeReasons)

private[rapids] case class GpuFlowCalibrationSample(
    scanUnitNanos: Double,
    scanBytes: Long,
    gpuUnitNanos: Double,
    gpuBytes: Long,
    shuffleUnitNanos: Double,
    shuffleBytes: Long,
    broadcastNanos: Double,
    broadcastBytes: Long,
    batchUnitNanos: Double,
    batchBytes: Long,
    nonGpuNanos: Double,
    ioBytes: Long,
    taskCount: Long)

private[rapids] case class GpuFlowAqeCalibration(
    scanUnitNanosPerByte: Option[Double],
    gpuUnitNanosPerByte: Option[Double],
    shuffleUnitNanosPerByte: Option[Double],
    broadcastNanosPerByte: Option[Double],
    batchUnitNanosPerByte: Option[Double],
    fixedNanosPerTask: Option[Double],
    referenceControl: GpuFlowControl,
    bounds: GpuFlowControlBounds,
    sampleWindows: Long,
    fixedNanosPerTaskStandardError: Option[Double] = None,
    fixedTaskCostSampleWindows: Long = 0L,
    fixedTaskCostReason: String = "missing-fixed-task-cost",
    fixedTaskCostSource: String = "",
    // Cluster-wide CPU task slots from registered executors, stamped by the driver endpoint when
    // the snapshot is served. Zero means no executor census exists yet; consumers that need wave
    // arithmetic must freeze rather than substitute a per-executor quantity.
    clusterTaskSlots: Int = 0)

private[rapids] object GpuFlowAqeCalibration {
  def empty(constraints: GraphOptimizerConstraints): GpuFlowAqeCalibration = {
    val reference = GpuFlowControl(
      scanWindow = if (constraints.scan.enabled) constraints.scan.initialReadWindow else 0.0,
      gpuTasks = if (constraints.gpu.enabled) constraints.gpu.initialConcurrentTasks else 0.0,
      shuffleWindow = if (constraints.shuffle.enabled) {
        constraints.shuffle.initialPrefetchWindow
      } else 0.0,
      shuffleBytes = if (constraints.shuffle.enabled) {
        constraints.shuffle.initialReadyBytes
      } else 0.0,
      batchBytes = if (constraints.batch.enabled) constraints.batch.initialTargetBytes else 0.0)
    // A rate measured at one actuator setting identifies service demand at that operating point,
    // not the counterfactual response to a different scan window, GPU quota, shuffle envelope, or
    // batch size. AQE may use these rates to compare plan demand, but it must price every candidate
    // at the deployed reference control until multi-setting response evidence exists.
    GpuFlowAqeCalibration(None, None, None, None, None, None, reference,
      GpuFlowControlBounds(reference, reference), sampleWindows = 0L)
  }
}

private[rapids] class GpuFlowAqeCalibrationAccumulator(
    constraints: GraphOptimizerConstraints) {
  private case class FixedTaskCostFit(
      estimate: Option[Double],
      standardError: Option[Double],
      sampleWindows: Long,
      reason: String,
      source: String = "fitted-task-duration-intercept")

  // A 95% normal confidence boundary is an evidence requirement, not a tuning preference. The
  // reducer rule may only use the fitted intercept when measurement uncertainty excludes zero.
  private val fixedTaskCostConfidenceZ = 1.96
  private var scanUnitNanos = 0.0
  private var scanBytes = 0L
  private var gpuUnitNanos = 0.0
  private var gpuBytes = 0L
  private var shuffleUnitNanos = 0.0
  private var shuffleBytes = 0L
  private var broadcastNanos = 0.0
  private var broadcastBytes = 0L
  private var batchUnitNanos = 0.0
  private var batchBytes = 0L
  private var regressionWindows = 0L
  private var meanIoBytesPerTask = 0.0
  private var meanNonGpuNanosPerTask = 0.0
  private var ioVariance = 0.0
  private var ioNonGpuCovariance = 0.0
  private var nonGpuVariance = 0.0
  private var directFixedNanosPerTaskLowerBound = Double.PositiveInfinity
  private var directFixedSampleWindows = 0L
  private var windows = 0L

  def add(sample: GpuFlowCalibrationSample): Unit = {
    if (sample.scanUnitNanos > 0.0 && sample.scanBytes > 0L) {
      scanUnitNanos += sample.scanUnitNanos
      scanBytes += sample.scanBytes
    }
    if (sample.gpuUnitNanos > 0.0 && sample.gpuBytes > 0L) {
      gpuUnitNanos += sample.gpuUnitNanos
      gpuBytes += sample.gpuBytes
    }
    if (sample.shuffleUnitNanos > 0.0 && sample.shuffleBytes > 0L) {
      shuffleUnitNanos += sample.shuffleUnitNanos
      shuffleBytes += sample.shuffleBytes
    }
    if (sample.broadcastNanos > 0.0 && sample.broadcastBytes > 0L) {
      broadcastNanos += sample.broadcastNanos
      broadcastBytes += sample.broadcastBytes
    }
    if (sample.batchUnitNanos > 0.0 && sample.batchBytes > 0L) {
      batchUnitNanos += sample.batchUnitNanos
      batchBytes += sample.batchBytes
    }
    if (sample.nonGpuNanos >= 0.0 && sample.ioBytes >= 0L && sample.taskCount > 0L) {
      val ioPerTask = sample.ioBytes.toDouble / sample.taskCount.toDouble
      val nanosPerTask = sample.nonGpuNanos / sample.taskCount.toDouble
      regressionWindows += 1L
      val ioDelta = ioPerTask - meanIoBytesPerTask
      val nanosDelta = nanosPerTask - meanNonGpuNanosPerTask
      meanIoBytesPerTask += ioDelta / regressionWindows.toDouble
      meanNonGpuNanosPerTask += nanosDelta / regressionWindows.toDouble
      ioVariance += ioDelta * (ioPerTask - meanIoBytesPerTask)
      ioNonGpuCovariance += ioDelta * (nanosPerTask - meanNonGpuNanosPerTask)
      nonGpuVariance += nanosDelta * (nanosPerTask - meanNonGpuNanosPerTask)
    }
    windows += 1L
  }

  def addTaskSetupNanos(setupNanos: Long): Unit = {
    if (setupNanos >= 0L) {
      directFixedNanosPerTaskLowerBound = math.min(
        directFixedNanosPerTaskLowerBound, setupNanos.toDouble)
      directFixedSampleWindows += 1L
    }
  }

  def snapshot: GpuFlowAqeCalibration = {
    val empty = GpuFlowAqeCalibration.empty(constraints)
    val fixedTaskCost = fittedFixedTaskCost
    empty.copy(
      scanUnitNanosPerByte = rate(scanUnitNanos, scanBytes),
      gpuUnitNanosPerByte = rate(gpuUnitNanos, gpuBytes),
      shuffleUnitNanosPerByte = rate(shuffleUnitNanos, shuffleBytes),
      broadcastNanosPerByte = rate(broadcastNanos, broadcastBytes),
      batchUnitNanosPerByte = rate(batchUnitNanos, batchBytes),
      fixedNanosPerTask = fixedTaskCost.estimate,
      sampleWindows = windows,
      fixedNanosPerTaskStandardError = fixedTaskCost.standardError,
      fixedTaskCostSampleWindows = fixedTaskCost.sampleWindows,
      fixedTaskCostReason = fixedTaskCost.reason,
      fixedTaskCostSource = fixedTaskCost.source)
  }

  private def rate(unitNanos: Double, bytes: Long): Option[Double] =
    if (unitNanos > 0.0 && bytes > 0L) Some(unitNanos / bytes.toDouble) else None

  /**
   * Fit non-GPU time/task = bytes/task * rate + fixed task cost.
   *
   * A non-negative least-squares boundary solution is not identification: when the best fit puts
   * the intercept at zero, byte service and fixed task overhead have not been separated. Using
   * that boundary as a measured zero caused the reducer rule to add tasks for microseconds of
   * modeled byte-service gain. Require an interior ordinary-least-squares solution and enough
   * residual evidence for the intercept's 95% lower confidence bound to remain above zero.
   */
  private def fittedFixedTaskCost: FixedTaskCostFit = {
    if (directFixedSampleWindows > 0L && directFixedNanosPerTaskLowerBound > 0.0 &&
        java.lang.Double.isFinite(directFixedNanosPerTaskLowerBound)) {
      return FixedTaskCostFit(Some(directFixedNanosPerTaskLowerBound), None,
        directFixedSampleWindows, "", "executor-deserialize-lower-bound")
    }
    if (regressionWindows < 3L) {
      return FixedTaskCostFit(None, None, regressionWindows,
        "insufficient-fixed-task-cost-samples")
    }

    val count = regressionWindows.toDouble
    if (ioVariance <= 0.0) {
      if (meanIoBytesPerTask != 0.0) {
        return FixedTaskCostFit(None, None, regressionWindows,
          "collinear-fixed-task-cost-samples")
      }
      val estimate = meanNonGpuNanosPerTask
      val residualVariance = math.max(0.0, nonGpuVariance) / (count - 1.0)
      val standardError = math.sqrt(residualVariance / count)
      return identifiedFixedTaskCost(estimate, standardError)
    }

    val slope = ioNonGpuCovariance / ioVariance
    val estimate = meanNonGpuNanosPerTask - slope * meanIoBytesPerTask
    if (slope < 0.0 || estimate <= 0.0) {
      return FixedTaskCostFit(None, None, regressionWindows,
        "boundary-fixed-task-cost-fit")
    }
    val residualSumSquares = math.max(0.0,
      nonGpuVariance - ioNonGpuCovariance * ioNonGpuCovariance / ioVariance)
    val residualVariance = residualSumSquares / (count - 2.0)
    val standardError = math.sqrt(residualVariance *
      (1.0 / count + meanIoBytesPerTask * meanIoBytesPerTask / ioVariance))
    identifiedFixedTaskCost(estimate, standardError)
  }

  private def identifiedFixedTaskCost(
      estimate: Double,
      standardError: Double): FixedTaskCostFit = {
    if (!java.lang.Double.isFinite(estimate) || !java.lang.Double.isFinite(standardError) ||
        estimate <= 0.0 ||
        estimate - fixedTaskCostConfidenceZ * standardError <= 0.0) {
      FixedTaskCostFit(None, Some(standardError), regressionWindows,
        "uncertain-fixed-task-cost-fit")
    } else {
      FixedTaskCostFit(Some(estimate), Some(standardError), regressionWindows, "")
    }
  }
}

/** Replay-complete record for one stage at one graph decision epoch. */
private[rapids] case class GraphStageDecisionRecord(
    epochId: Long,
    trigger: String,
    key: AutotuneStageKey,
    active: Boolean,
    completed: Boolean,
    parentStageIds: Seq[Int],
    observedTasks: Long,
    sampleTasks: Long,
    currentHintVersion: Long,
    graphCurrentObjectiveNanos: Double,
    graphSelectedObjectiveNanos: Double,
    predictedCurrentNanos: Double,
    predictedSelectedNanos: Double,
    durationAdjoint: Double,
    currentControl: GpuFlowControl,
    selectedControl: GpuFlowControl,
    endToEndGradient: GpuFlowGradient,
    freezeReasons: GraphControlFreezeReasons)

/** Best locally feasible complete hint for one fixed GPU-stage quota. */
private[rapids] case class GraphStageCostAlternative(
    gpuTasks: Int,
    hint: StageRuntimeHint,
    predictedNanos: Double,
    control: Option[GpuFlowControl] = None,
    localGradient: GpuFlowGradient = GpuFlowGradient())

/** Model-calibrated stage cost curve consumed by the graph-wide shared allocator. */
private[rapids] case class GraphStageCostCurve(
    predictedCurrentNanos: Double,
    sampleTasks: Long,
    alternatives: Seq[GraphStageCostAlternative],
    analyticalState: Option[GraphStageAnalyticalState] = None,
    calibrationSample: Option[GpuFlowCalibrationSample] = None)

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
      nowNanos: Long): Seq[GraphOptimizerDecision]

  def stageSubmitted(key: AutotuneStageKey): Seq[GraphOptimizerDecision]

  def stageCompleted(key: AutotuneStageKey): Seq[GraphOptimizerDecision]

  /** Feed back the version-stamped hint published by the driver endpoint. */
  def hintPublished(hint: StageRuntimeHint): Unit

  /** Release graph state after a SQL execution ends. */
  def executionCompleted(executionId: Long): Unit

  /** Drain replay-complete records produced by graph decision epochs. */
  def drainDecisionRecords(): Seq[GraphStageDecisionRecord] = Seq.empty

  /** Record one finalized Spark task setup cost for execution-local AQE calibration. */
  def observeTaskSetup(executionId: Long, setupNanos: Long): Unit = {}

  /** Calibrated resource-demand rates for one SQL execution's complete AQE plan evaluation. */
  def aqeCalibrationSnapshot(executionId: Long): Option[GpuFlowAqeCalibration] = None
}

/**
 * Driver graph optimizer backed by a constrained analytical stage-cost model.
 *
 * The optimizer uses a projected-gradient inner solve for continuous controls and an exact
 * discrete outer solve for shared GPU quotas. The objective is predicted end-to-end completion of
 * the remaining max-plus Spark DAG, calibrated from measured task elapsed time and resource work.
 * A reverse pass supplies stage criticality and control gradients. User configuration and executor
 * clamps define feasibility, not policy; unidentifiable counterfactual controls remain ineligible.
 */
class AnalyticalGraphWideAutotuneOptimizer(
    constraints: GraphOptimizerConstraints) extends GraphWideAutotuneOptimizer {

  private case class StageState(
      descriptor: AutotuneStageDescriptor,
      var active: Boolean,
      var completed: Boolean,
      var window: StageObservationAgg,
      var windowHintVersion: Long,
      var lastEvaluationNanos: Long,
      var observedTasks: Long,
      var currentHint: StageRuntimeHint,
      var costCurve: Option[GraphStageCostCurve])

  private val stages = mutable.HashMap.empty[AutotuneStageKey, StageState]
  private val pendingDecisionRecords = mutable.ArrayBuffer.empty[GraphStageDecisionRecord]
  // Rates and fixed task cost are execution-local evidence. Mixing unrelated queries made the
  // fixed-cost intercept depend on workload order: a large exchange could calibrate tiny later
  // exchanges, while tiny exchanges could collapse a useful large-exchange fit to its boundary.
  private val aqeCalibration = mutable.HashMap.empty[Long, GpuFlowAqeCalibrationAccumulator]
  private var nextEpochId = 1L

  override def initialHint(
      key: AutotuneStageKey,
      descriptor: AutotuneStageDescriptor): StageRuntimeHint = synchronized {
    val content = initialHintContent(key, descriptor.shape)
    stages.put(key, StageState(descriptor, active = false, completed = false,
      StageObservationAgg.empty,
      0L, Long.MinValue, 0L, content, None))
    content
  }

  override def observe(
      msg: RapidsAutotuneObservationMsg,
      current: StageRuntimeHint,
      nowNanos: Long): Seq[GraphOptimizerDecision] = synchronized {
    stages.get(msg.key).map { state =>
      state.currentHint = current
      // Receiving task work is itself authoritative evidence that the stage is active. This also
      // keeps direct unit/integration users correct if a Spark stage-submitted event is delayed.
      if (!state.completed) {
        state.active = true
      }
      // A cost calibration is valid only for the complete joint hint that produced its samples.
      // Tasks from an older version can finish after a re-hint; mixing those observations with the
      // new candidate would attribute elapsed/resource work to settings the task never used.
      if (msg.hintVersion != current.version) {
        return Seq.empty
      }
      if (state.windowHintVersion != msg.hintVersion) {
        state.window = StageObservationAgg.empty
        state.windowHintVersion = msg.hintVersion
      }
      state.window = state.window.merge(msg)
      state.observedTasks += 1L
      val enoughSamples = state.window.taskCount >= constraints.minSampleTasks
      val intervalElapsed = state.lastEvaluationNanos == Long.MinValue ||
        nowNanos - state.lastEvaluationNanos >= constraints.updateIntervalNanos
      if (!enoughSamples || !intervalElapsed) {
        Seq.empty
      } else {
        val observation = state.window
        state.window = StageObservationAgg.empty
        state.lastEvaluationNanos = nowNanos
        state.costCurve = AnalyticalStageCostModel.costCurve(
          observation, current, state.descriptor.shape, constraints)
        state.costCurve.flatMap(_.calibrationSample).foreach { sample =>
          aqeCalibration.getOrElseUpdate(msg.key.executionId,
            new GpuFlowAqeCalibrationAccumulator(constraints)).add(sample)
        }
        recomputeAllocations("feedback")
      }
    }.getOrElse(Seq.empty)
  }

  override def stageSubmitted(key: AutotuneStageKey): Seq[GraphOptimizerDecision] = synchronized {
    stages.get(key).foreach { state =>
      state.active = true
      state.completed = false
    }
    recomputeAllocations("stage-submitted")
  }

  override def stageCompleted(key: AutotuneStageKey): Seq[GraphOptimizerDecision] = synchronized {
    stages.get(key).foreach { state =>
      state.active = false
      state.completed = true
    }
    recomputeAllocations("stage-completed")
  }

  override def hintPublished(hint: StageRuntimeHint): Unit = synchronized {
    stages.get(hint.key).foreach(_.currentHint = hint)
  }

  override def executionCompleted(executionId: Long): Unit = synchronized {
    stages.retain { case (key, _) => key.executionId != executionId }
    aqeCalibration.remove(executionId)
  }

  override def observeTaskSetup(executionId: Long, setupNanos: Long): Unit = synchronized {
    aqeCalibration.getOrElseUpdate(executionId,
      new GpuFlowAqeCalibrationAccumulator(constraints)).addTaskSetupNanos(setupNanos)
  }

  override def drainDecisionRecords(): Seq[GraphStageDecisionRecord] = synchronized {
    val drained = pendingDecisionRecords.toVector
    pendingDecisionRecords.clear()
    drained
  }

  override def aqeCalibrationSnapshot(
      executionId: Long): Option[GpuFlowAqeCalibration] = synchronized {
    aqeCalibration.get(executionId).map(_.snapshot).filter(_.sampleWindows > 0L)
  }

  private def recomputeAllocations(trigger: String): Seq[GraphOptimizerDecision] = {
    val inputs = stages.iterator.map { case (key, state) =>
      GraphStageAllocationInput(
        key = key,
        descriptor = state.descriptor,
        active = state.active,
        completed = state.completed,
        observedTasks = state.observedTasks,
        currentHint = state.currentHint,
        costCurve = state.costCurve)
    }.toSeq
    val solution = GraphCriticalPathAllocator.solve(inputs, constraints.gpu.maxConcurrentTasks)
    recordDecisionEpoch(trigger, inputs, solution)
    solution.toSeq.flatMap(_.allocations).flatMap {
      allocation =>
        stages.get(allocation.key).flatMap { state =>
          if (sameHintContent(state.currentHint, allocation.hint)) {
            None
          } else {
            state.currentHint = allocation.hint
            Some(GraphOptimizerDecision(
              hint = allocation.hint,
              predictedCurrentNanos = allocation.predictedCurrentNanos,
              predictedSelectedNanos = allocation.predictedSelectedNanos))
          }
        }
    }
  }

  private def recordDecisionEpoch(
      trigger: String,
      inputs: Seq[GraphStageAllocationInput],
      solution: Option[GraphAllocationSolution]): Unit = {
    val epochId = nextEpochId
    nextEpochId += 1L
    val allocationByKey = solution.toSeq.flatMap(_.allocations).map(a => a.key -> a).toMap
    val selectedAdjoints = solution.map(_.selectedFlow.durationAdjoints).getOrElse(Map.empty)
    inputs.sortBy(input => (input.key.executionId, input.key.stageId, input.key.stageAttemptId))
      .foreach { input =>
        val curve = input.costCurve
        val analytical = curve.flatMap(_.analyticalState)
        val allocation = allocationByKey.get(input.key)
        val currentControl = analytical.map(_.currentControl)
          .getOrElse(AnalyticalStageCostModel.controlFor(input.currentHint))
        val selectedControl = allocation.flatMap(_.control).getOrElse(currentControl)
        val localGradient = allocation.map(_.localGradient)
          .orElse(analytical.map(_.currentGradient)).getOrElse(GpuFlowGradient())
        val adjoint = selectedAdjoints.getOrElse(input.key, 0.0)
        val reasons = analytical.map(_.freezeReasons).getOrElse {
          val reason = if (input.completed) "completed-work-is-sunk"
          else if (!input.active) "stage-not-active"
          else "no-current-version-calibration"
          GraphControlFreezeReasons(reason, reason, reason, reason, reason)
        }
        pendingDecisionRecords += GraphStageDecisionRecord(
          epochId = epochId,
          trigger = trigger,
          key = input.key,
          active = input.active,
          completed = input.completed,
          parentStageIds = input.descriptor.parentStageIds,
          observedTasks = input.observedTasks,
          sampleTasks = curve.map(_.sampleTasks).getOrElse(0L),
          currentHintVersion = input.currentHint.version,
          graphCurrentObjectiveNanos = solution.map(_.currentFlow.objectiveNanos).getOrElse(0.0),
          graphSelectedObjectiveNanos = solution.map(_.selectedFlow.objectiveNanos).getOrElse(0.0),
          predictedCurrentNanos = curve.map(_.predictedCurrentNanos).getOrElse(0.0),
          predictedSelectedNanos = allocation.map(_.predictedSelectedNanos)
            .getOrElse(curve.map(_.predictedCurrentNanos).getOrElse(0.0)),
          durationAdjoint = adjoint,
          currentControl = currentControl,
          selectedControl = selectedControl,
          endToEndGradient = localGradient.scale(adjoint),
          freezeReasons = reasons)
      }
  }

  private def sameHintContent(left: StageRuntimeHint, right: StageRuntimeHint): Boolean =
    left.scan == right.scan && left.gpu == right.gpu &&
      left.shuffle == right.shuffle && left.batch == right.batch

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
      GpuRuntimeHint(
        maxConcurrentTasks = constraints.gpu.initialConcurrentTasks,
        sharedMaxConcurrentTasks = constraints.gpu.maxConcurrentTasks)
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
        // Preserve the executor's native, pool-derived split threshold until the model selects a
        // different batch point from observations.
        splitUntilSize = 0L)
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

private[rapids] case class GraphStageAllocationInput(
    key: AutotuneStageKey,
    descriptor: AutotuneStageDescriptor,
    active: Boolean,
    completed: Boolean = false,
    observedTasks: Long,
    currentHint: StageRuntimeHint,
    costCurve: Option[GraphStageCostCurve])

private[rapids] case class GraphStageAllocation(
    key: AutotuneStageKey,
    hint: StageRuntimeHint,
    predictedCurrentNanos: Double,
    predictedSelectedNanos: Double,
    control: Option[GpuFlowControl] = None,
    localGradient: GpuFlowGradient = GpuFlowGradient())

private[rapids] case class GraphAllocationSolution(
    allocations: Seq[GraphStageAllocation],
    currentFlow: GpuFlowGraphEvaluation,
    selectedFlow: GpuFlowGraphEvaluation)

/**
 * Pure shared-resource solver. It enumerates legal stage GPU quotas under one shared budget and
 * minimizes the sum of each SQL execution's remaining DAG critical-path length. No stage class,
 * fixed priority, or pressure threshold participates in allocation: priorities are the modeled
 * path-through lengths of the selected graph solution.
 */
private[rapids] object GraphCriticalPathAllocator {
  def allocate(
      graph: Seq[GraphStageAllocationInput],
      sharedGpuBudget: Int): Seq[GraphStageAllocation] =
    solve(graph, sharedGpuBudget).map(_.allocations).getOrElse(Seq.empty)

  def solve(
      graph: Seq[GraphStageAllocationInput],
      sharedGpuBudget: Int): Option[GraphAllocationSolution] = {
    val hasCalibratedStage = graph.exists(input =>
      input.active && input.costCurve.exists(_.alternatives.nonEmpty))
    if (!hasCalibratedStage) {
      return None
    }
    val tunable = graph.filter(_.active).flatMap { input =>
      input.costCurve.filter(_.alternatives.nonEmpty)
        .orElse(uncalibratedGpuCurve(input))
        .map(input -> _)
    }.sortBy { case (input, _) =>
      (input.key.executionId, input.key.stageId, input.key.stageAttemptId)
    }
    if (tunable.isEmpty) {
      return None
    }
    val tunableByKey = tunable.map { case (input, curve) =>
      (input.key, (input, curve))
    }.toMap

    val gpuStageCount = tunable.count { case (_, curve) =>
      curve.alternatives.exists(_.gpuTasks > 0)
    }
    // A positive quota keeps a stage runnable while the lower shared semaphore cap and
    // model-derived priority decide who runs. If the minimum eligible quotas cannot fit this
    // enumeration budget, the solve defers instead of inventing an unsupported lower quota.
    val quotaBudget = math.max(math.max(0, sharedGpuBudget), gpuStageCount)
    val choices = tunable.map { case (input, curve) =>
      val byGpuTasks = curve.alternatives
        .filter(alt => alt.gpuTasks <= quotaBudget || alt.gpuTasks <= 0)
        .groupBy(_.gpuTasks)
        .map { case (_, alternatives) => alternatives.minBy(_.predictedNanos) }
        .toSeq
        .sortBy(_.gpuTasks)
      (input, (curve, byGpuTasks))
    }

    var best = Map.empty[AutotuneStageKey, GraphStageCostAlternative]
    var bestObjective = Double.PositiveInfinity
    var bestGpuTasks = Int.MaxValue

    def search(
        index: Int,
        usedGpuTasks: Int,
        selected: Map[AutotuneStageKey, GraphStageCostAlternative]): Unit = {
      if (index >= choices.length) {
        val costs = selected.map { case (key, alternative) =>
          val (input, curve) = tunableByKey(key)
          key -> remainingNanos(input, alternative.predictedNanos,
            curve.sampleTasks)
        }
        val objective = graphObjective(graph, costs)
        if (objective < bestObjective ||
            (objective == bestObjective && usedGpuTasks < bestGpuTasks)) {
          best = selected
          bestObjective = objective
          bestGpuTasks = usedGpuTasks
        }
      } else {
        val (input, (_, alternatives)) = choices(index)
        alternatives.foreach { alternative =>
          val consumed = math.max(0, alternative.gpuTasks)
          if (usedGpuTasks + consumed <= quotaBudget) {
            search(index + 1, usedGpuTasks + consumed,
              selected + (input.key -> alternative))
          }
        }
      }
    }

    search(0, 0, Map.empty)
    if (best.isEmpty) {
      return None
    }

    val currentCosts = tunable.map { case (input, curve) =>
      input.key -> remainingNanos(input, curve.predictedCurrentNanos, curve.sampleTasks)
    }.toMap
    val selectedCosts = best.map { case (key, alternative) =>
      val (input, curve) = tunableByKey(key)
      key -> remainingNanos(input, alternative.predictedNanos, curve.sampleTasks)
    }
    val currentFlow = evaluateGraph(graph, currentCosts)
    val selectedFlow = evaluateGraph(graph, selectedCosts)
    val allocations = tunable.flatMap { case (input, curve) =>
      best.get(input.key).map { alternative =>
        // Reverse-mode criticality is the marginal change in the end-to-end objective caused by
        // one additional nanosecond in this stage. Multiplying by its remaining duration gives
        // the amount of query completion time currently exposed through this node.
        val priority = toPriority(
          selectedFlow.durationAdjoints.getOrElse(input.key, 0.0) *
            selectedCosts.getOrElse(input.key, 0.0))
        val allocatedGpu = if (alternative.hint.gpu.maxConcurrentTasks > 0) {
          alternative.hint.gpu.copy(
            maxConcurrentTasks = alternative.gpuTasks,
            sharedMaxConcurrentTasks = math.max(0, sharedGpuBudget),
            schedulingPriority = priority)
        } else {
          alternative.hint.gpu
        }
        GraphStageAllocation(
          key = input.key,
          hint = alternative.hint.copy(gpu = allocatedGpu),
          predictedCurrentNanos = curve.predictedCurrentNanos,
          predictedSelectedNanos = alternative.predictedNanos,
          control = alternative.control,
          localGradient = alternative.localGradient)
      }
    }
    Some(GraphAllocationSolution(allocations, currentFlow, selectedFlow))
  }

  /** Keep an uncalibrated active GPU stage at its native allocation. */
  private def uncalibratedGpuCurve(
      input: GraphStageAllocationInput): Option[GraphStageCostCurve] = {
    val currentGpuTasks = input.currentHint.gpu.maxConcurrentTasks
    if (input.costCurve.isEmpty && currentGpuTasks > 0) {
      Some(GraphStageCostCurve(
        predictedCurrentNanos = 0.0,
        sampleTasks = 1L,
        // Unknown cost is not evidence for tightening. Keeping only the native allocation makes
        // the global solve defer if calibrated siblings cannot coexist with this cold stage.
        alternatives = Seq(GraphStageCostAlternative(
          currentGpuTasks,
          input.currentHint,
          predictedNanos = 0.0))))
    } else {
      None
    }
  }

  private def remainingNanos(
      input: GraphStageAllocationInput,
      sampledNanos: Double,
      sampleTasks: Long): Double = {
    val totalTasks = math.max(1L, input.descriptor.shape.numTasks.toLong)
    val remainingTasks = math.max(1L, totalTasks - math.min(totalTasks, input.observedTasks))
    sampledNanos / math.max(1L, sampleTasks).toDouble * remainingTasks.toDouble
  }

  private def graphObjective(
      graph: Seq[GraphStageAllocationInput],
      costs: Map[AutotuneStageKey, Double]): Double = {
    evaluateGraph(graph, costs).objectiveNanos
  }

  private def evaluateGraph(
      graph: Seq[GraphStageAllocationInput],
      costs: Map[AutotuneStageKey, Double]): GpuFlowGraphEvaluation = {
    val nodes = graph.groupBy(_.key.executionId).values.flatMap { execution =>
      val parents = parentKeys(execution)
      execution.map { node =>
        GpuFlowGraphNode(
          key = node.key,
          parents = parents.getOrElse(node.key, Seq.empty),
          evaluation = GpuFlowStageEvaluation(
            predictedNanos = costs.getOrElse(node.key, 0.0),
            gradient = GpuFlowGradient()))
      }
    }.toSeq
    GpuFlowModel.evaluate(nodes)
  }

  private def parentKeys(
      execution: Seq[GraphStageAllocationInput]): Map[AutotuneStageKey, Seq[AutotuneStageKey]] = {
    val latestByStageId = execution.groupBy(_.key.stageId).map { case (stageId, attempts) =>
      stageId -> attempts.maxBy(_.key.stageAttemptId).key
    }
    execution.map { node =>
      node.key -> node.descriptor.parentStageIds.flatMap(latestByStageId.get).distinct
    }.toMap
  }

  private def toPriority(nanos: Double): Long = {
    if (nanos.isNaN || nanos <= 0.0) 0L
    else if (nanos >= Long.MaxValue.toDouble) Long.MaxValue
    else math.ceil(nanos).toLong
  }
}

/** Pure constrained optimizer used by [[AnalyticalGraphWideAutotuneOptimizer]]. */
object AnalyticalStageCostModel {
  def optimize(
      observation: StageObservationAgg,
      current: StageRuntimeHint,
      shape: AutotuneStageShape,
      constraints: GraphOptimizerConstraints): Option[GraphOptimizerDecision] = {
    costCurve(observation, current, shape, constraints).flatMap { curve =>
      val selected = curve.alternatives.reduceLeft { (left, right) =>
        if (right.predictedNanos < left.predictedNanos ||
            (right.predictedNanos == left.predictedNanos &&
              hintResource(right.hint) < hintResource(left.hint))) {
          right
        } else {
          left
        }
      }
      if (sameHintContent(selected.hint, current)) {
        None
      } else {
        Some(GraphOptimizerDecision(
          selected.hint,
          predictedCurrentNanos = curve.predictedCurrentNanos,
          predictedSelectedNanos = selected.predictedNanos))
      }
    }
  }

  /** Build the locally optimized stage-cost curve for every legal fixed GPU quota. */
  def costCurve(
      observation: StageObservationAgg,
      current: StageRuntimeHint,
      shape: AutotuneStageShape,
      constraints: GraphOptimizerConstraints): Option[GraphStageCostCurve] = {
    if (observation.taskCount < constraints.minSampleTasks ||
        observation.totalTaskDurationNanos <= 0L) {
      return None
    }

    val work = calibrate(observation, current, shape)
    val currentControl = controlFor(current)
    // A single feedback window supplies one operating point. It can calibrate the work already
    // performed, but cannot identify any actuator's counterfactual response. Keep every continuous
    // control at that measured point. Structural AQE parallelism is optimized separately from
    // measured map sizes and task-wave cost, so it remains eligible without fabricating a response
    // curve for executor controls.
    val gpuValues = Seq(currentControl.gpuTasks.toInt)
    val bounds = GpuFlowControlBounds(currentControl, currentControl)
    val currentEvaluation = GpuFlowStageModel.evaluate(work, currentControl)

    val alternatives = gpuValues.map { gpuTasks =>
      val fixedGpu = gpuTasks.toDouble
      val fixedBounds = bounds.copy(
        minimum = bounds.minimum.copy(gpuTasks = fixedGpu),
        maximum = bounds.maximum.copy(gpuTasks = fixedGpu))
      val selected = GpuFlowProjectedOptimizer.optimize(
        work, currentControl.copy(gpuTasks = fixedGpu), fixedBounds)
      val selectedEvaluation = GpuFlowStageModel.evaluate(work, selected)
      GraphStageCostAlternative(
        gpuTasks = gpuTasks,
        hint = hintForCandidate(
          current, currentControl, selected, constraints.batch.initialSplitUntilSize),
        predictedNanos = selectedEvaluation.predictedNanos,
        control = Some(selected),
        localGradient = selectedEvaluation.gradient)
    }
    Some(GraphStageCostCurve(
      predictedCurrentNanos = currentEvaluation.predictedNanos,
      sampleTasks = observation.taskCount,
      alternatives = alternatives,
      analyticalState = Some(GraphStageAnalyticalState(
        currentControl = currentControl,
        currentGradient = currentEvaluation.gradient,
        bounds = bounds,
        freezeReasons = freezeReasons(work, currentControl, bounds))),
      calibrationSample = Some(calibrationSample(observation, shape, work))))
  }

  private def hintForCandidate(
      current: StageRuntimeHint,
      currentControl: GpuFlowControl,
      selected: GpuFlowControl,
      configuredSplitUntilSize: Long): StageRuntimeHint = current.copy(
    scan = if (current.scan.maxReadWindow > 0) {
      current.scan.copy(maxReadWindow = selected.scanWindow.toInt)
    } else current.scan,
    gpu = if (current.gpu.maxConcurrentTasks > 0) {
      current.gpu.copy(maxConcurrentTasks = selected.gpuTasks.toInt)
    } else current.gpu,
    shuffle = if (isActiveShuffle(current)) {
      current.shuffle.copy(
        prefetchWindow = selected.shuffleWindow.toInt,
        maxReadyBytes = selected.shuffleBytes.toLong,
        coalesceTargetBytes = selectedCoalesceBytes(current, selected))
    } else current.shuffle,
    batch = if (current.batch.targetBatchBytes > 0L &&
        selected.batchBytes != currentControl.batchBytes) {
      current.batch.copy(
        targetBatchBytes = selected.batchBytes.toLong,
        splitUntilSize = selectedSplitSize(
          current, selected.batchBytes.toLong, configuredSplitUntilSize))
    } else current.batch)

  private def sameHintContent(left: StageRuntimeHint, right: StageRuntimeHint): Boolean =
    left.scan == right.scan && left.gpu == right.gpu &&
      left.shuffle == right.shuffle && left.batch == right.batch

  private def hintResource(hint: StageRuntimeHint): Double =
    math.max(0, hint.scan.maxReadWindow).toDouble +
      math.max(0, hint.gpu.maxConcurrentTasks).toDouble +
      math.max(0, hint.shuffle.prefetchWindow).toDouble +
      math.max(0L, if (hint.shuffle.maxReadyBytes == Long.MaxValue) {
        0L
      } else hint.shuffle.maxReadyBytes).toDouble +
      math.max(0L, hint.batch.targetBatchBytes).toDouble

  private def calibrate(
      obs: StageObservationAgg,
      current: StageRuntimeHint,
      shape: AutotuneStageShape): GpuFlowStageWork = {
    val classified = classifyNonGpuWork(obs, shape)
    val retry = obs.totalRetryOrLostTimeNanos.toDouble
    // One observation at one batch target cannot distinguish fixed task cost from per-batch
    // setup. Keep batch response frozen until history contains multiple batch controls; treating
    // the entire residual as batch work fabricates a derivative and biases AQE operator counts.
    val batchOverhead = 0.0
    GpuFlowStageWork(
      scanNanos = classified.scanNanos,
      gpuNanos = obs.totalGpuHoldingNanos.toDouble +
        obs.totalGpuSemaphoreWaitNanos.toDouble,
      shuffleNanos = classified.shuffleNanos,
      batchNanos = batchOverhead,
      fixedNanos = classified.broadcastNanos +
        classified.unclassifiedNanos,
      retryNanos = retry,
      baseline = controlFor(current))
  }

  private case class NonGpuClassification(
      scanNanos: Double,
      scanBytes: Long,
      shuffleNanos: Double,
      shuffleBytes: Long,
      broadcastNanos: Double,
      broadcastBytes: Long,
      unclassifiedNanos: Double)

  private def classifyNonGpuWork(
      observation: StageObservationAgg,
      shape: AutotuneStageShape): NonGpuClassification = {
    val elapsed = math.max(observation.totalTaskDurationNanos.toDouble,
      (observation.totalGpuHoldingNanos + observation.totalGpuSemaphoreWaitNanos +
        observation.totalRetryOrLostTimeNanos).toDouble)
    val available = math.max(0.0,
      elapsed - observation.totalGpuHoldingNanos - observation.totalGpuSemaphoreWaitNanos -
        observation.totalRetryOrLostTimeNanos)
    val scanBytes = if (shape.hasGpuScan) {
      math.max(0L, observation.totalInputBytes - observation.totalShuffleReadBytes)
    } else 0L
    val shuffleBytes = if (shape.hasShuffle) {
      observation.totalShuffleReadBytes + observation.totalShuffleWriteBytes
    } else 0L
    // Broadcast exchange build/distribution runs on the driver and is not measured by this task
    // observation clock. Do not relabel task input/output time as broadcast service time.
    val broadcastBytes = 0L
    val classifiedBytes = scanBytes + shuffleBytes + broadcastBytes
    def nanos(bytes: Long): Double = {
      if (classifiedBytes > 0L) available * bytes.toDouble / classifiedBytes.toDouble else 0.0
    }
    val scanNanos = nanos(scanBytes)
    val shuffleNanos = nanos(shuffleBytes)
    val broadcastNanos = nanos(broadcastBytes)
    NonGpuClassification(
      scanNanos,
      scanBytes,
      shuffleNanos,
      shuffleBytes,
      broadcastNanos,
      broadcastBytes,
      math.max(0.0, available - scanNanos - shuffleNanos - broadcastNanos))
  }

  private[rapids] def controlFor(hint: StageRuntimeHint): GpuFlowControl = GpuFlowControl(
    scanWindow = activeScanWindow(hint).toDouble,
    gpuTasks = activeGpuTasks(hint).toDouble,
    shuffleWindow = activeShuffleWindow(hint).toDouble,
    shuffleBytes = activeShuffleBytes(hint).toDouble,
    batchBytes = activeBatchBytes(hint).toDouble)

  private def freezeReasons(
      work: GpuFlowStageWork,
      current: GpuFlowControl,
      bounds: GpuFlowControlBounds): GraphControlFreezeReasons = {
    def reason(value: Double, measuredWork: Double, minimum: Double, maximum: Double): String = {
      if (value <= 0.0) "actuator-not-present"
      else if (measuredWork <= 0.0) "no-measured-work"
      else if (maximum <= minimum) "single-operating-point-response-unidentified"
      else ""
    }
    GraphControlFreezeReasons(
      scanWindow = reason(current.scanWindow, work.scanNanos,
        bounds.minimum.scanWindow, bounds.maximum.scanWindow),
      gpuTasks = reason(current.gpuTasks, work.gpuNanos,
        bounds.minimum.gpuTasks, bounds.maximum.gpuTasks),
      shuffleWindow = reason(current.shuffleWindow, work.shuffleNanos,
        bounds.minimum.shuffleWindow, bounds.maximum.shuffleWindow),
      shuffleBytes = reason(current.shuffleBytes, work.shuffleNanos,
        bounds.minimum.shuffleBytes, bounds.maximum.shuffleBytes),
      batchBytes = reason(current.batchBytes, work.batchNanos,
        bounds.minimum.batchBytes, bounds.maximum.batchBytes))
  }

  private def calibrationSample(
      observation: StageObservationAgg,
      shape: AutotuneStageShape,
      work: GpuFlowStageWork): GpuFlowCalibrationSample = {
    val classified = classifyNonGpuWork(observation, shape)
    val processedBytes = Seq(
      observation.totalInputBytes,
      observation.totalOutputBytes,
      classified.shuffleBytes).max
    GpuFlowCalibrationSample(
      scanUnitNanos = work.scanNanos * work.baseline.scanWindow,
      scanBytes = classified.scanBytes,
      gpuUnitNanos = work.gpuNanos * work.baseline.gpuTasks,
      gpuBytes = processedBytes,
      shuffleUnitNanos = work.shuffleNanos * work.baseline.shuffleWindow *
        work.baseline.shuffleBytes,
      shuffleBytes = classified.shuffleBytes,
      broadcastNanos = classified.broadcastNanos,
      broadcastBytes = classified.broadcastBytes,
      batchUnitNanos = work.batchNanos * work.baseline.batchBytes,
      batchBytes = processedBytes,
      nonGpuNanos = classified.scanNanos + classified.shuffleNanos +
        classified.broadcastNanos + classified.unclassifiedNanos,
      ioBytes = classified.scanBytes + classified.shuffleBytes + classified.broadcastBytes,
      taskCount = observation.taskCount)
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

  private def selectedCoalesceBytes(
      current: StageRuntimeHint,
      selected: GpuFlowControl): Long = {
    if (current.shuffle.coalesceTargetBytes > 0L && selected.batchBytes > 0L) {
      selected.batchBytes.toLong
    } else {
      current.shuffle.coalesceTargetBytes
    }
  }

  private def selectedSplitSize(
      current: StageRuntimeHint,
      selectedBatchBytes: Long,
      configuredSplitUntilSize: Long): Long = {
    if (configuredSplitUntilSize > 0L) {
      math.min(current.batch.maxBatchBytes, selectedBatchBytes)
    } else {
      current.batch.splitUntilSize
    }
  }
}

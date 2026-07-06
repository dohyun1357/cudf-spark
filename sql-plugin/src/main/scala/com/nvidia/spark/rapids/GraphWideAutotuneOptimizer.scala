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
    shuffle: ShuffleOptimizerBounds,
    batch: BatchOptimizerBounds)

case class ScanOptimizerBounds(
    enabled: Boolean,
    initialReadWindow: Int,
    maxReadyBytes: Long)

case class ShuffleOptimizerBounds(
    enabled: Boolean,
    initialPrefetchWindow: Int,
    initialReadyBytes: Long,
    initialCoalesceBytes: Long)

case class BatchOptimizerBounds(
    enabled: Boolean,
    initialTargetBytes: Long,
    maxBatchBytes: Long)

object GraphOptimizerConstraints {
  def fromConf(conf: RapidsConf): GraphOptimizerConstraints = {
    val nativeBatchTarget = conf.gpuTargetBatchSizeBytes
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
        enabled = conf.isAutotuneClosedLoopMode && conf.autotuneScanEnabled,
        // Start at the static envelope; the optimizer, rather than an initial per-knob policy,
        // selects any different feasible point.
        initialReadWindow = conf.autotuneScanReadWindowCap,
        maxReadyBytes = conf.autotuneScanMaxReadyBytes),
      shuffle = ShuffleOptimizerBounds(
        enabled = conf.isAutotuneClosedLoopMode && conf.autotuneShuffleEnabled,
        // A cold stage begins at the configured feasible envelope. Window 1 was an unevidenced
        // policy choice and could serialize a short stage before the first feedback epoch.
        initialPrefetchWindow = conf.autotuneShuffleMaxPrefetchWindow,
        initialReadyBytes = conf.autotuneShuffleMaxReadyBytes,
        // Shuffle coalescing targets the native GPU batch size; a cold stage never overrides it.
        initialCoalesceBytes = nativeBatchTarget),
      batch = BatchOptimizerBounds(
        enabled = conf.isAutotuneClosedLoopMode && conf.autotuneBatchEnabled,
        initialTargetBytes = nativeBatchTarget,
        maxBatchBytes = batchMaximum))
  }
}

case class AutotuneStageDescriptor(
    shape: AutotuneStageShape,
    parentStageIds: Seq[Int] = Seq.empty)

private[rapids] case class GraphControlFreezeReasons(
    scanWindow: String = "",
    gpuTasks: String = "",
    shuffleWindow: String = "",
    shuffleBytes: String = "",
    batchBytes: String = "")

private[rapids] case class GpuFlowCalibrationSample(
    shuffleUnitNanos: Double,
    shuffleBytes: Long,
    taskShuffleReadBytes: Long,
    nonGpuNanos: Double,
    ioBytes: Long,
    taskCount: Long)

private[rapids] case class GpuFlowAqeCalibration(
    shuffleUnitNanosPerByte: Option[Double],
    fixedNanosPerTask: Option[Double],
    referenceControl: GpuFlowControl,
    sampleWindows: Long,
    fixedNanosPerTaskStandardError: Option[Double] = None,
    fixedTaskCostSampleWindows: Long = 0L,
    fixedTaskCostReason: String = "missing-fixed-task-cost",
    fixedTaskCostSource: String = "",
    // Largest mean per-task shuffle-read byte load observed across this execution's calibration
    // windows. The shuffle rate identifies byte service only at task sizes it was measured at;
    // a reducer range built beyond this region extrapolates into a possibly superlinear operator
    // regime (a q21 join stage more than doubled per-task time for +25% bytes) and must freeze.
    maxCalibratedTaskShuffleReadBytes: Long = 0L,
    // Cluster-wide CPU task slots from registered executors, stamped by the driver endpoint when
    // the snapshot is served. Zero means no executor census exists yet; consumers that need wave
    // arithmetic must freeze rather than substitute a per-executor quantity.
    clusterTaskSlots: Int = 0)

private[rapids] object GpuFlowAqeCalibration {
  def empty(constraints: GraphOptimizerConstraints): GpuFlowAqeCalibration = {
    val reference = GpuFlowControl(
      scanWindow = if (constraints.scan.enabled) constraints.scan.initialReadWindow else 0.0,
      // The GPU admission hint surface was removed; the gpu dimension stays in the model as the
      // disabled (zero) actuator, matching the configuration validated by ablation.
      gpuTasks = 0.0,
      shuffleWindow = if (constraints.shuffle.enabled) {
        constraints.shuffle.initialPrefetchWindow
      } else 0.0,
      shuffleBytes = if (constraints.shuffle.enabled) {
        constraints.shuffle.initialReadyBytes
      } else 0.0,
      batchBytes = if (constraints.batch.enabled) constraints.batch.initialTargetBytes else 0.0)
    // A rate measured at one actuator setting identifies service demand at that operating point,
    // not the counterfactual response to a different setting. Consumers may use these rates to
    // compare plan demand, but must price every candidate at the deployed reference control until
    // multi-setting response evidence exists.
    GpuFlowAqeCalibration(None, None, reference, sampleWindows = 0L)
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
  private var shuffleUnitNanos = 0.0
  private var shuffleBytes = 0L
  private var maxTaskShuffleReadBytes = 0L
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
    if (sample.shuffleUnitNanos > 0.0 && sample.shuffleBytes > 0L) {
      shuffleUnitNanos += sample.shuffleUnitNanos
      shuffleBytes += sample.shuffleBytes
    }
    if (sample.taskShuffleReadBytes > maxTaskShuffleReadBytes) {
      maxTaskShuffleReadBytes = sample.taskShuffleReadBytes
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
    val fixedTaskCost = fittedFixedTaskCost
    GpuFlowAqeCalibration.empty(constraints).copy(
      shuffleUnitNanosPerByte =
        if (shuffleUnitNanos > 0.0 && shuffleBytes > 0L) {
          Some(shuffleUnitNanos / shuffleBytes.toDouble)
        } else None,
      fixedNanosPerTask = fixedTaskCost.estimate,
      sampleWindows = windows,
      fixedNanosPerTaskStandardError = fixedTaskCost.standardError,
      fixedTaskCostSampleWindows = fixedTaskCost.sampleWindows,
      fixedTaskCostReason = fixedTaskCost.reason,
      fixedTaskCostSource = fixedTaskCost.source,
      maxCalibratedTaskShuffleReadBytes = maxTaskShuffleReadBytes)
  }

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
    predictedCurrentNanos: Double,
    durationAdjoint: Double,
    currentControl: GpuFlowControl,
    endToEndGradient: GpuFlowGradient,
    freezeReasons: GraphControlFreezeReasons)

/** Model-calibrated stage cost at the measured operating point, one graph epoch input. */
private[rapids] case class GraphStageCostCurve(
    predictedCurrentNanos: Double,
    sampleTasks: Long,
    currentControl: GpuFlowControl,
    currentGradient: GpuFlowGradient,
    freezeReasons: GraphControlFreezeReasons,
    calibrationSample: GpuFlowCalibrationSample)

/**
 * Driver graph optimizer backed by a constrained analytical stage-cost model. Single owner of
 * the complete initial joint [[StageRuntimeHint]]; scan, shuffle and batch decisions are never
 * delegated to independent knob policies.
 *
 * The objective is predicted end-to-end completion of the remaining max-plus Spark DAG,
 * calibrated from measured task elapsed time and resource work. A reverse pass supplies stage
 * criticality and control gradients for eventlog reporting. User configuration and executor
 * clamps define feasibility, not policy; unidentifiable counterfactual controls remain
 * ineligible, so every continuous control is held at its measured operating point and a decision
 * epoch is reporting-only.
 */
class AnalyticalGraphWideAutotuneOptimizer(constraints: GraphOptimizerConstraints) {

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

  /** Register a graph node and return its complete initial joint hint content. */
  def initialHint(
      key: AutotuneStageKey,
      descriptor: AutotuneStageDescriptor): StageRuntimeHint = synchronized {
    val content = initialHintContent(key, descriptor.shape)
    stages.put(key, StageState(descriptor, active = false, completed = false,
      StageObservationAgg.empty,
      0L, Long.MinValue, 0L, content, None))
    content
  }

  /** Ingest one task observation; a completed window records a reporting-only decision epoch. */
  def observe(
      msg: RapidsAutotuneObservationMsg,
      current: StageRuntimeHint,
      nowNanos: Long): Unit = synchronized {
    stages.get(msg.key).foreach { state =>
      state.currentHint = current
      // Receiving task work is itself authoritative evidence that the stage is active. This also
      // keeps direct unit/integration users correct if a Spark stage-submitted event is delayed.
      if (!state.completed) {
        state.active = true
      }
      // A cost calibration is valid only for the complete joint hint that produced its samples.
      // Tasks from an older version can finish after a re-hint; mixing those observations with
      // the new candidate would attribute elapsed/resource work to settings the task never used.
      if (msg.hintVersion == current.version) {
        if (state.windowHintVersion != msg.hintVersion) {
          state.window = StageObservationAgg.empty
          state.windowHintVersion = msg.hintVersion
        }
        state.window = state.window.merge(msg)
        state.observedTasks += 1L
        val enoughSamples = state.window.taskCount >= constraints.minSampleTasks
        val intervalElapsed = state.lastEvaluationNanos == Long.MinValue ||
          nowNanos - state.lastEvaluationNanos >= constraints.updateIntervalNanos
        if (enoughSamples && intervalElapsed) {
          val observation = state.window
          state.window = StageObservationAgg.empty
          state.lastEvaluationNanos = nowNanos
          state.costCurve = AnalyticalStageCostModel.costCurve(
            observation, current, state.descriptor.shape, constraints)
          state.costCurve.foreach { curve =>
            aqeCalibration.getOrElseUpdate(msg.key.executionId,
              new GpuFlowAqeCalibrationAccumulator(constraints)).add(curve.calibrationSample)
          }
          recordDecisionEpoch("feedback")
        }
      }
    }
  }

  def stageSubmitted(key: AutotuneStageKey): Unit = synchronized {
    stages.get(key).foreach { state =>
      state.active = true
      state.completed = false
    }
    recordDecisionEpoch("stage-submitted")
  }

  def stageCompleted(key: AutotuneStageKey): Unit = synchronized {
    stages.get(key).foreach { state =>
      state.active = false
      state.completed = true
    }
    recordDecisionEpoch("stage-completed")
  }

  /** Feed back the version-stamped hint published by the driver endpoint. */
  def hintPublished(hint: StageRuntimeHint): Unit = synchronized {
    stages.get(hint.key).foreach(_.currentHint = hint)
  }

  /** Release graph state after a SQL execution ends. */
  def executionCompleted(executionId: Long): Unit = synchronized {
    stages.retain { case (key, _) => key.executionId != executionId }
    aqeCalibration.remove(executionId)
  }

  /** Record one finalized Spark task setup cost for execution-local AQE calibration. */
  def observeTaskSetup(executionId: Long, setupNanos: Long): Unit = synchronized {
    aqeCalibration.getOrElseUpdate(executionId,
      new GpuFlowAqeCalibrationAccumulator(constraints)).addTaskSetupNanos(setupNanos)
  }

  /** Drain replay-complete records produced by graph decision epochs. */
  def drainDecisionRecords(): Seq[GraphStageDecisionRecord] = synchronized {
    val drained = pendingDecisionRecords.toVector
    pendingDecisionRecords.clear()
    drained
  }

  /** Calibrated resource-demand rates for one SQL execution's AQE plan decisions. */
  def aqeCalibrationSnapshot(executionId: Long): Option[GpuFlowAqeCalibration] = synchronized {
    aqeCalibration.get(executionId).map(_.snapshot).filter(_.sampleWindows > 0L)
  }

  /**
   * Record one graph-wide decision epoch. Every continuous control is held at its measured
   * operating point, so an epoch is reporting-only: it evaluates the remaining max-plus graph at
   * the current controls (objective, reverse-mode criticality, freeze reasons) and never
   * republishes a hint.
   */
  private def recordDecisionEpoch(trigger: String): Unit = {
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
    val epochId = nextEpochId
    nextEpochId += 1L
    val calibrated = inputs.filter(input => input.active && input.costCurve.isDefined)
    val flow = if (calibrated.isEmpty) None else {
      val costs = calibrated.flatMap { input =>
        input.costCurve.map { curve =>
          input.key -> remainingNanos(input, curve.predictedCurrentNanos, curve.sampleTasks)
        }
      }.toMap
      Some(evaluateGraph(inputs, costs))
    }
    val adjoints = flow.map(_.durationAdjoints).getOrElse(Map.empty)
    inputs.sortBy(input => (input.key.executionId, input.key.stageId, input.key.stageAttemptId))
      .foreach { input =>
        val curve = input.costCurve
        val currentControl = curve.map(_.currentControl)
          .getOrElse(AnalyticalStageCostModel.controlFor(input.currentHint))
        val localGradient = curve.map(_.currentGradient).getOrElse(GpuFlowGradient())
        val adjoint = adjoints.getOrElse(input.key, 0.0)
        val reasons = curve.map(_.freezeReasons).getOrElse {
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
          graphCurrentObjectiveNanos = flow.map(_.objectiveNanos).getOrElse(0.0),
          predictedCurrentNanos = curve.map(_.predictedCurrentNanos).getOrElse(0.0),
          durationAdjoint = adjoint,
          currentControl = currentControl,
          endToEndGradient = localGradient.scale(adjoint),
          freezeReasons = reasons)
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

  private def evaluateGraph(
      graph: Seq[GraphStageAllocationInput],
      costs: Map[AutotuneStageKey, Double]): GpuFlowGraphEvaluation = {
    val nodes = graph.groupBy(_.key.executionId).values.flatMap { execution =>
      val parents = parentKeys(execution)
      execution.map { node =>
        GpuFlowGraphNode(
          key = node.key,
          parents = parents.getOrElse(node.key, Seq.empty),
          durationNanos = costs.getOrElse(node.key, 0.0))
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
        maxBatchBytes = constraints.batch.maxBatchBytes)
    } else {
      BatchRuntimeHint.empty
    }
    StageRuntimeHint.empty(key).copy(
      scan = scanHint,
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

/** Pure stage-cost calibration used by [[AnalyticalGraphWideAutotuneOptimizer]]. */
object AnalyticalStageCostModel {
  /** Calibrate the stage cost at the measured operating point of one observation window. */
  def costCurve(
      observation: StageObservationAgg,
      current: StageRuntimeHint,
      shape: AutotuneStageShape,
      constraints: GraphOptimizerConstraints): Option[GraphStageCostCurve] = {
    if (observation.taskCount < constraints.minSampleTasks ||
        observation.totalTaskDurationNanos <= 0L) {
      return None
    }

    // A single feedback window supplies one operating point. It can calibrate the work already
    // performed, but cannot identify any actuator's counterfactual response, so every continuous
    // control stays at that measured point; guessing a descent direction from one point is what
    // caused the historical q17 scan/shuffle regressions. Structural AQE parallelism is
    // optimized separately from measured map sizes and task-wave cost.
    val work = calibrate(observation, current, shape)
    val evaluation = GpuFlowStageModel.evaluate(work)
    Some(GraphStageCostCurve(
      predictedCurrentNanos = evaluation.predictedNanos,
      sampleTasks = observation.taskCount,
      currentControl = work.baseline,
      currentGradient = evaluation.gradient,
      freezeReasons = freezeReasons(work, work.baseline),
      calibrationSample = calibrationSample(observation, shape, work)))
  }

  private def calibrate(
      obs: StageObservationAgg,
      current: StageRuntimeHint,
      shape: AutotuneStageShape): GpuFlowStageWork = {
    val classified = classifyNonGpuWork(obs, shape)
    GpuFlowStageWork(
      scanNanos = classified.scanNanos,
      gpuNanos = obs.totalGpuHoldingNanos.toDouble +
        obs.totalGpuSemaphoreWaitNanos.toDouble,
      shuffleNanos = classified.shuffleNanos,
      fixedNanos = classified.unclassifiedNanos,
      retryNanos = obs.totalRetryOrLostTimeNanos.toDouble,
      baseline = controlFor(current))
  }

  private case class NonGpuClassification(
      scanNanos: Double,
      scanBytes: Long,
      shuffleNanos: Double,
      shuffleBytes: Long,
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
    // observation clock, so no broadcast lane exists here. Driver broadcast service remains
    // unmeasured until it has its own clock; do not relabel task input/output time as broadcast.
    val classifiedBytes = scanBytes + shuffleBytes
    def nanos(bytes: Long): Double = {
      if (classifiedBytes > 0L) available * bytes.toDouble / classifiedBytes.toDouble else 0.0
    }
    val scanNanos = nanos(scanBytes)
    val shuffleNanos = nanos(shuffleBytes)
    NonGpuClassification(
      scanNanos,
      scanBytes,
      shuffleNanos,
      shuffleBytes,
      math.max(0.0, available - scanNanos - shuffleNanos))
  }

  private[rapids] def controlFor(hint: StageRuntimeHint): GpuFlowControl = GpuFlowControl(
    scanWindow = activeScanWindow(hint).toDouble,
    // The GPU admission hint surface was removed; report the gpu control dimension as the
    // disabled (zero) actuator.
    gpuTasks = 0.0,
    shuffleWindow = activeShuffleWindow(hint).toDouble,
    shuffleBytes = activeShuffleBytes(hint).toDouble,
    batchBytes = activeBatchBytes(hint).toDouble)

  private def freezeReasons(
      work: GpuFlowStageWork,
      current: GpuFlowControl): GraphControlFreezeReasons = {
    // A rate measured at one actuator setting identifies work at that operating point only, so a
    // present actuator with measured work is always frozen as response-unidentified.
    def reason(value: Double, measuredWork: Double): String = {
      if (value <= 0.0) "actuator-not-present"
      else if (measuredWork <= 0.0) "no-measured-work"
      else "single-operating-point-response-unidentified"
    }
    GraphControlFreezeReasons(
      scanWindow = reason(current.scanWindow, work.scanNanos),
      gpuTasks = reason(current.gpuTasks, work.gpuNanos),
      shuffleWindow = reason(current.shuffleWindow, work.shuffleNanos),
      shuffleBytes = reason(current.shuffleBytes, work.shuffleNanos),
      // One observation at one batch target cannot separate per-batch setup from fixed task
      // cost, so batch work is never measured separately from the residual.
      batchBytes = reason(current.batchBytes, 0.0))
  }

  private def calibrationSample(
      observation: StageObservationAgg,
      shape: AutotuneStageShape,
      work: GpuFlowStageWork): GpuFlowCalibrationSample = {
    val classified = classifyNonGpuWork(observation, shape)
    GpuFlowCalibrationSample(
      shuffleUnitNanos = work.shuffleNanos * work.baseline.shuffleWindow *
        work.baseline.shuffleBytes,
      shuffleBytes = classified.shuffleBytes,
      taskShuffleReadBytes =
        observation.totalShuffleReadBytes / math.max(1L, observation.taskCount),
      nonGpuNanos = classified.scanNanos + classified.shuffleNanos +
        classified.unclassifiedNanos,
      ioBytes = classified.scanBytes + classified.shuffleBytes,
      taskCount = observation.taskCount)
  }

  private def activeScanWindow(hint: StageRuntimeHint): Int =
    if (hint.scan.maxReadWindow > 0) hint.scan.maxReadWindow else 0

  private def isActiveShuffle(hint: StageRuntimeHint): Boolean =
    hint.shuffle.maxReadyBytes > 0L && hint.shuffle.maxReadyBytes != Long.MaxValue

  private def activeShuffleWindow(hint: StageRuntimeHint): Int =
    if (isActiveShuffle(hint)) math.max(1, hint.shuffle.prefetchWindow) else 0

  private def activeShuffleBytes(hint: StageRuntimeHint): Long =
    if (isActiveShuffle(hint)) hint.shuffle.maxReadyBytes else 0L

  private def activeBatchBytes(hint: StageRuntimeHint): Long =
    if (hint.batch.targetBatchBytes > 0L) hint.batch.targetBatchBytes else 0L
}

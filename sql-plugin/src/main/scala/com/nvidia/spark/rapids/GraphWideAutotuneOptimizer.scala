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
    val staticGpuLimit = conf.maxConcurrentGpuTasks.intValue()
    val nativeGpuMaximum = if (nativeGpuTaskSlots > 0) {
      if (staticGpuLimit > 0) math.min(nativeGpuTaskSlots, staticGpuLimit)
      else nativeGpuTaskSlots
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

/** Best locally feasible complete hint for one fixed GPU-stage quota. */
case class GraphStageCostAlternative(
    gpuTasks: Int,
    hint: StageRuntimeHint,
    predictedNanos: Double)

/** Model-calibrated stage cost curve consumed by the graph-wide shared allocator. */
case class GraphStageCostCurve(
    predictedCurrentNanos: Double,
    sampleTasks: Long,
    alternatives: Seq[GraphStageCostAlternative])

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
        recomputeAllocations()
      }
    }.getOrElse(Seq.empty)
  }

  override def stageSubmitted(key: AutotuneStageKey): Seq[GraphOptimizerDecision] = synchronized {
    stages.get(key).foreach { state =>
      state.active = true
      state.completed = false
    }
    recomputeAllocations()
  }

  override def stageCompleted(key: AutotuneStageKey): Seq[GraphOptimizerDecision] = synchronized {
    stages.get(key).foreach { state =>
      state.active = false
      state.completed = true
    }
    recomputeAllocations()
  }

  override def hintPublished(hint: StageRuntimeHint): Unit = synchronized {
    stages.get(hint.key).foreach(_.currentHint = hint)
  }

  override def executionCompleted(executionId: Long): Unit = synchronized {
    stages.retain { case (key, _) => key.executionId != executionId }
  }

  private def recomputeAllocations(): Seq[GraphOptimizerDecision] = {
    val inputs = stages.iterator.map { case (key, state) =>
      GraphStageAllocationInput(
        key = key,
        descriptor = state.descriptor,
        active = state.active,
        observedTasks = state.observedTasks,
        currentHint = state.currentHint,
        costCurve = state.costCurve)
    }.toSeq
    GraphCriticalPathAllocator.allocate(inputs, constraints.gpu.maxConcurrentTasks).flatMap {
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
    observedTasks: Long,
    currentHint: StageRuntimeHint,
    costCurve: Option[GraphStageCostCurve])

private[rapids] case class GraphStageAllocation(
    key: AutotuneStageKey,
    hint: StageRuntimeHint,
    predictedCurrentNanos: Double,
    predictedSelectedNanos: Double)

/**
 * Pure shared-resource solver. It enumerates legal stage GPU quotas under one shared budget and
 * minimizes the sum of each SQL execution's remaining DAG critical-path length. No stage class,
 * fixed priority, or pressure threshold participates in allocation: priorities are the modeled
 * path-through lengths of the selected graph solution.
 */
private[rapids] object GraphCriticalPathAllocator {
  def allocate(
      graph: Seq[GraphStageAllocationInput],
      sharedGpuBudget: Int): Seq[GraphStageAllocation] = {
    val hasCalibratedStage = graph.exists(input =>
      input.active && input.costCurve.exists(_.alternatives.nonEmpty))
    if (!hasCalibratedStage) {
      return Seq.empty
    }
    val tunable = graph.filter(_.active).flatMap { input =>
      input.costCurve.filter(_.alternatives.nonEmpty)
        .orElse(uncalibratedGpuCurve(input))
        .map(input -> _)
    }.sortBy { case (input, _) =>
      (input.key.executionId, input.key.stageId, input.key.stageAttemptId)
    }
    if (tunable.isEmpty) {
      return Seq.empty
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
      return Seq.empty
    }

    val selectedCosts = best.map { case (key, alternative) =>
      val (input, curve) = tunableByKey(key)
      key -> remainingNanos(input, alternative.predictedNanos, curve.sampleTasks)
    }
    val flow = evaluateGraph(graph, selectedCosts)
    tunable.flatMap { case (input, curve) =>
      best.get(input.key).map { alternative =>
        // Reverse-mode criticality is the marginal change in the end-to-end objective caused by
        // one additional nanosecond in this stage. Multiplying by its remaining duration gives
        // the amount of query completion time currently exposed through this node.
        val priority = toPriority(
          flow.durationAdjoints.getOrElse(input.key, 0.0) *
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
          predictedSelectedNanos = alternative.predictedNanos)
      }
    }
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
    val gpuValues = intDomain(
      currentControl.gpuTasks.toInt,
      if (current.gpu.maxConcurrentTasks > 0 &&
          work.gpuNanos > 0.0) {
        constraints.gpu.maxConcurrentTasks
      } else 0)
    val bounds = controlBounds(work, currentControl, constraints)

    val alternatives = gpuValues.map { gpuTasks =>
      val fixedGpu = gpuTasks.toDouble
      val fixedBounds = bounds.copy(
        minimum = bounds.minimum.copy(gpuTasks = fixedGpu),
        maximum = bounds.maximum.copy(gpuTasks = fixedGpu))
      val selected = GpuFlowProjectedOptimizer.optimize(
        work, currentControl.copy(gpuTasks = fixedGpu), fixedBounds)
      GraphStageCostAlternative(
        gpuTasks = gpuTasks,
        hint = hintForCandidate(
          current, currentControl, selected, constraints.batch.initialSplitUntilSize),
        predictedNanos = GpuFlowStageModel.evaluate(work, selected).predictedNanos)
    }
    Some(GraphStageCostCurve(
      predictedCurrentNanos = GpuFlowStageModel.evaluate(work, currentControl).predictedNanos,
      sampleTasks = observation.taskCount,
      alternatives = alternatives))
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
    GpuFlowStageWork(
      scanNanos = scanNanos,
      gpuNanos = obs.totalGpuHoldingNanos.toDouble +
        obs.totalGpuSemaphoreWaitNanos.toDouble,
      shuffleNanos = shuffleNanos,
      batchNanos = batchOverhead,
      fixedNanos = math.max(0.0, unclassifiedNanos - batchOverhead),
      retryNanos = retry,
      baseline = controlFor(current))
  }

  private def controlFor(hint: StageRuntimeHint): GpuFlowControl = GpuFlowControl(
    scanWindow = activeScanWindow(hint).toDouble,
    gpuTasks = activeGpuTasks(hint).toDouble,
    shuffleWindow = activeShuffleWindow(hint).toDouble,
    shuffleBytes = activeShuffleBytes(hint).toDouble,
    batchBytes = activeBatchBytes(hint).toDouble)

  private def controlBounds(
      work: GpuFlowStageWork,
      current: GpuFlowControl,
      constraints: GraphOptimizerConstraints): GpuFlowControlBounds = {
    def range(
        value: Double,
        hasWork: Boolean,
        minimum: Double,
        maximum: Double): (Double, Double) = {
      if (value <= 0.0 || !hasWork || maximum <= 0.0) (value, value)
      else (math.min(minimum, maximum), math.max(value, maximum))
    }
    val scan = range(current.scanWindow, work.scanNanos > 0.0,
      1.0, constraints.scan.maxReadWindow.toDouble)
    val gpu = range(current.gpuTasks, work.gpuNanos > 0.0,
      1.0, constraints.gpu.maxConcurrentTasks.toDouble)
    val shuffleWindow = range(current.shuffleWindow, work.shuffleNanos > 0.0,
      1.0, constraints.shuffle.maxPrefetchWindow.toDouble)
    val shuffleBytes = range(current.shuffleBytes, work.shuffleNanos > 0.0,
      constraints.shuffle.initialReadyBytes.toDouble,
      constraints.shuffle.maxReadyBytes.toDouble)
    val minimumBatch = if (constraints.batch.minimumTargetBytes > 0L) {
      constraints.batch.minimumTargetBytes
    } else {
      constraints.batch.initialTargetBytes
    }
    val batch = range(current.batchBytes, work.batchNanos > 0.0,
      minimumBatch.toDouble, constraints.batch.maxBatchBytes.toDouble)
    GpuFlowControlBounds(
      minimum = GpuFlowControl(scan._1, gpu._1, shuffleWindow._1,
        shuffleBytes._1, batch._1),
      maximum = GpuFlowControl(scan._2, gpu._2, shuffleWindow._2,
        shuffleBytes._2, batch._2))
  }

  private def intDomain(current: Int, maximum: Int): Seq[Int] = {
    if (current <= 0 || maximum <= 0) {
      Seq(current)
    } else if (maximum <= 64) {
      // A GPU-admission quota gates the complete task, not only the measured GPU-holding lane.
      // One observation at `current` cannot identify the counterfactual response below it. Keep
      // lower quotas ineligible until replay/history supplies that response; online queueing can
      // still support retaining or raising the quota within the hard envelope.
      (math.min(current, maximum) to maximum).toSeq
    } else {
      val values = mutable.TreeSet.empty[Int]
      values += math.min(current, maximum)
      values += maximum
      var value = math.max(1, current)
      while (value < maximum && value <= Integer.MAX_VALUE / 2) {
        value = math.min(maximum, value * 2)
        values += value
      }
      values.toSeq
    }
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

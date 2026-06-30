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
        initialPrefetchWindow = 1,
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
    // When more stages are active than there are shared slots, every stage keeps a quota of one
    // while the lower shared cap and model-derived queue priority decide who runs. This avoids a
    // zero quota that could strand already-launched tasks; it does not raise the shared cap.
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
    val priorities = pathThroughNanos(graph, selectedCosts)
    tunable.flatMap { case (input, curve) =>
      best.get(input.key).map { alternative =>
        val priority = toPriority(priorities.getOrElse(input.key, 0.0))
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
    graph.groupBy(_.key.executionId).values.map { execution =>
      val parents = parentKeys(execution)
      val memo = mutable.HashMap.empty[AutotuneStageKey, Double]
      def longestTo(key: AutotuneStageKey, visiting: Set[AutotuneStageKey]): Double = {
        if (visiting.contains(key)) {
          costs.getOrElse(key, 0.0)
        } else {
          memo.getOrElseUpdate(key, {
            val upstream = parents.getOrElse(key, Seq.empty)
              .map(parent => longestTo(parent, visiting + key))
            costs.getOrElse(key, 0.0) + (if (upstream.isEmpty) 0.0 else upstream.max)
          })
        }
      }
      execution.map(node => longestTo(node.key, Set.empty)).foldLeft(0.0)(math.max)
    }.sum
  }

  private def pathThroughNanos(
      graph: Seq[GraphStageAllocationInput],
      costs: Map[AutotuneStageKey, Double]): Map[AutotuneStageKey, Double] = {
    graph.groupBy(_.key.executionId).values.flatMap { execution =>
      val parents = parentKeys(execution)
      val children = mutable.HashMap.empty[AutotuneStageKey, Seq[AutotuneStageKey]]
        .withDefaultValue(Seq.empty)
      parents.foreach { case (child, stageParents) =>
        stageParents.foreach(parent => children.update(parent, children(parent) :+ child))
      }
      val toMemo = mutable.HashMap.empty[AutotuneStageKey, Double]
      val fromMemo = mutable.HashMap.empty[AutotuneStageKey, Double]
      def longestTo(key: AutotuneStageKey, visiting: Set[AutotuneStageKey]): Double = {
        if (visiting.contains(key)) costs.getOrElse(key, 0.0) else {
          toMemo.getOrElseUpdate(key, {
            val upstream = parents.getOrElse(key, Seq.empty)
              .map(parent => longestTo(parent, visiting + key))
            costs.getOrElse(key, 0.0) + (if (upstream.isEmpty) 0.0 else upstream.max)
          })
        }
      }
      def longestFrom(key: AutotuneStageKey, visiting: Set[AutotuneStageKey]): Double = {
        if (visiting.contains(key)) costs.getOrElse(key, 0.0) else {
          fromMemo.getOrElseUpdate(key, {
            val downstream = children(key).map(child => longestFrom(child, visiting + key))
            costs.getOrElse(key, 0.0) + (if (downstream.isEmpty) 0.0 else downstream.max)
          })
        }
      }
      execution.map { node =>
        val own = costs.getOrElse(node.key, 0.0)
        node.key -> math.max(0.0,
          longestTo(node.key, Set.empty) + longestFrom(node.key, Set.empty) - own)
      }
    }.toMap
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
      if (constraints.batch.minimumTargetBytes > 0L) {
        constraints.batch.minimumTargetBytes
      } else {
        constraints.batch.initialTargetBytes
      },
      currentCandidate.batchBytes,
      if (current.batch.targetBatchBytes > 0L && work.batchOverheadNanosAtCurrent > 0.0) {
        constraints.batch.maxBatchBytes
      } else 0L)

    val alternatives = gpuValues.map { gpuTasks =>
      val selected = optimizeForGpu(
        work,
        currentCandidate.copy(gpuTasks = gpuTasks),
        scanValues,
        shuffleWindowValues,
        shuffleValues,
        batchValues)
      GraphStageCostAlternative(
        gpuTasks = gpuTasks,
        hint = hintForCandidate(
          current, currentCandidate, selected, constraints.batch.initialSplitUntilSize),
        predictedNanos = predictedNanos(work, selected))
    }
    Some(GraphStageCostCurve(
      predictedCurrentNanos = predictedNanos(work, currentCandidate),
      sampleTasks = observation.taskCount,
      alternatives = alternatives))
  }

  private def optimizeForGpu(
      work: CalibratedWork,
      initial: Candidate,
      scanValues: Seq[Int],
      shuffleWindowValues: Seq[Int],
      shuffleValues: Seq[Long],
      batchValues: Seq[Long]): Candidate = {
    var selected = initial
    var selectedCost = predictedNanos(work, initial)
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
        choose(shuffleWindowValues.map(value => selected.copy(shuffleWindow = value)))
        choose(shuffleValues.map(value => selected.copy(shuffleBytes = value)))
        choose(for {
          window <- shuffleWindowValues
          bytes <- shuffleValues
        } yield selected.copy(shuffleWindow = window, shuffleBytes = bytes))
        choose(batchValues.map(value => selected.copy(batchBytes = value)))
      }
    }
    descend(preferMoreOnTie = true)
    descend(preferMoreOnTie = false)
    selected
  }

  private def hintForCandidate(
      current: StageRuntimeHint,
      currentCandidate: Candidate,
      selected: Candidate,
      configuredSplitUntilSize: Long): StageRuntimeHint = current.copy(
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
        splitUntilSize = selectedSplitSize(
          current, selected.batchBytes, configuredSplitUntilSize))
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

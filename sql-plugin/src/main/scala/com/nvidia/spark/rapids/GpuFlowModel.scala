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

/**
 * Continuous controls owned jointly by the graph-wide optimizer.
 *
 * A zero value means that the corresponding actuator is absent from this stage. Positive values
 * are expressed in the actuator's native unit. The analytical model differentiates with respect
 * to log(control), which makes gradients comparable across counts and byte-valued controls.
 */
private[rapids] case class GpuFlowControl(
    scanWindow: Double,
    gpuTasks: Double,
    shuffleWindow: Double,
    shuffleBytes: Double,
    batchBytes: Double) {
  def value(index: Int): Double = index match {
    case GpuFlowControl.ScanWindow => scanWindow
    case GpuFlowControl.GpuTasks => gpuTasks
    case GpuFlowControl.ShuffleWindow => shuffleWindow
    case GpuFlowControl.ShuffleBytes => shuffleBytes
    case GpuFlowControl.BatchBytes => batchBytes
  }

  def updated(index: Int, value: Double): GpuFlowControl = index match {
    case GpuFlowControl.ScanWindow => copy(scanWindow = value)
    case GpuFlowControl.GpuTasks => copy(gpuTasks = value)
    case GpuFlowControl.ShuffleWindow => copy(shuffleWindow = value)
    case GpuFlowControl.ShuffleBytes => copy(shuffleBytes = value)
    case GpuFlowControl.BatchBytes => copy(batchBytes = value)
  }
}

private[rapids] object GpuFlowControl {
  val ScanWindow = 0
  val GpuTasks = 1
  val ShuffleWindow = 2
  val ShuffleBytes = 3
  val BatchBytes = 4
  val Indices: Seq[Int] = 0 until 5
}

private[rapids] case class GpuFlowControlBounds(
    minimum: GpuFlowControl,
    maximum: GpuFlowControl) {
  require(GpuFlowControl.Indices.forall { index =>
    val low = minimum.value(index)
    val high = maximum.value(index)
    low >= 0.0 && high >= low && finite(low) && finite(high)
  }, "flow-control bounds must be finite, non-negative, and ordered")

  def project(control: GpuFlowControl): GpuFlowControl = {
    GpuFlowControl.Indices.foldLeft(control) { case (projected, index) =>
      projected.updated(index,
        math.max(minimum.value(index), math.min(maximum.value(index), control.value(index))))
    }
  }

  private def finite(value: Double): Boolean = !value.isNaN && !value.isInfinity
}

/** Calibrated work at the controls that produced an observation window. */
private[rapids] case class GpuFlowStageWork(
    scanNanos: Double,
    gpuNanos: Double,
    shuffleNanos: Double,
    batchNanos: Double,
    fixedNanos: Double,
    retryNanos: Double,
    baseline: GpuFlowControl)

/** Derivative of stage latency with respect to log(control). */
private[rapids] case class GpuFlowGradient(
    scanWindow: Double = 0.0,
    gpuTasks: Double = 0.0,
    shuffleWindow: Double = 0.0,
    shuffleBytes: Double = 0.0,
    batchBytes: Double = 0.0) {
  def value(index: Int): Double = index match {
    case GpuFlowControl.ScanWindow => scanWindow
    case GpuFlowControl.GpuTasks => gpuTasks
    case GpuFlowControl.ShuffleWindow => shuffleWindow
    case GpuFlowControl.ShuffleBytes => shuffleBytes
    case GpuFlowControl.BatchBytes => batchBytes
  }

  def scale(factor: Double): GpuFlowGradient = GpuFlowGradient(
    scanWindow * factor,
    gpuTasks * factor,
    shuffleWindow * factor,
    shuffleBytes * factor,
    batchBytes * factor)
}

private[rapids] case class GpuFlowStageEvaluation(
    predictedNanos: Double,
    gradient: GpuFlowGradient)

/**
 * Differentiable local flow model for one Spark stage.
 *
 * Scan, GPU, and shuffle are concurrent producer/consumer lanes, so stage service time contains
 * their maximum. Batch setup is serial work. Measured retry time is retained in the objective.
 * Spill bytes do not have time units or an identifiable derivative from one observation window,
 * so this model does not invent a resource-pressure penalty from them.
 */
private[rapids] object GpuFlowStageModel {
  def evaluate(work: GpuFlowStageWork, control: GpuFlowControl): GpuFlowStageEvaluation = {
    val scanScale = scale(control.scanWindow, work.baseline.scanWindow)
    val gpuScale = scale(control.gpuTasks, work.baseline.gpuTasks)
    val shuffleWindowScale = scale(control.shuffleWindow, work.baseline.shuffleWindow)
    val shuffleBytesScale = scale(control.shuffleBytes, work.baseline.shuffleBytes)
    val batchScale = scale(control.batchBytes, work.baseline.batchBytes)

    val scan = inverse(work.scanNanos, scanScale)
    val gpu = inverse(work.gpuNanos, gpuScale)
    val shuffle = inverse(work.shuffleNanos, shuffleWindowScale * shuffleBytesScale)
    val batch = inverse(work.batchNanos, batchScale)

    val lanes = Array(scan, gpu, shuffle)
    val bottleneck = lanes.max
    val tieTolerance = math.max(1.0, math.abs(bottleneck)) * 1e-12
    val criticalLaneCount = lanes.count(value => math.abs(value - bottleneck) <= tieTolerance)
    val laneShare = if (criticalLaneCount > 0) 1.0 / criticalLaneCount.toDouble else 0.0

    val gradient = GpuFlowGradient(
      scanWindow = (if (math.abs(scan - bottleneck) <= tieTolerance) {
        -scan * laneShare
      } else 0.0),
      gpuTasks = (if (math.abs(gpu - bottleneck) <= tieTolerance) {
        -gpu * laneShare
      } else 0.0),
      shuffleWindow = (if (math.abs(shuffle - bottleneck) <= tieTolerance) {
        -shuffle * laneShare
      } else 0.0),
      shuffleBytes = (if (math.abs(shuffle - bottleneck) <= tieTolerance) {
        -shuffle * laneShare
      } else 0.0),
      batchBytes = -batch)

    GpuFlowStageEvaluation(
      predictedNanos = bottleneck + batch + work.fixedNanos + work.retryNanos,
      gradient = gradient)
  }

  private def scale(candidate: Double, baseline: Double): Double = {
    if (candidate > 0.0 && baseline > 0.0) candidate / baseline else 1.0
  }

  private def inverse(value: Double, divisor: Double): Double = {
    if (value <= 0.0 || divisor <= 0.0) math.max(0.0, value) else value / divisor
  }
}

/** One node in the remaining Spark execution graph. */
private[rapids] case class GpuFlowGraphNode(
    key: AutotuneStageKey,
    parents: Seq[AutotuneStageKey],
    evaluation: GpuFlowStageEvaluation)

private[rapids] case class GpuFlowGraphEvaluation(
    objectiveNanos: Double,
    completionNanos: Map[AutotuneStageKey, Double],
    durationAdjoints: Map[AutotuneStageKey, Double],
    controlGradients: Map[AutotuneStageKey, GpuFlowGradient])

/**
 * Max-plus Spark DAG objective and its reverse-mode derivative.
 *
 * Forward: C(node) = duration(node) + max(C(parent)). The query objective is the maximum
 * completion in each SQL execution, summed across executions. Reverse: adjoints flow from every
 * execution output through the parents that attained the max. Exact ties share the adjoint, which
 * is a valid subgradient and avoids stage-id-dependent decisions.
 */
private[rapids] object GpuFlowModel {
  def evaluate(nodes: Seq[GpuFlowGraphNode]): GpuFlowGraphEvaluation = {
    if (nodes.isEmpty) {
      return GpuFlowGraphEvaluation(0.0, Map.empty, Map.empty, Map.empty)
    }
    val byKey = nodes.map(node => node.key -> node).toMap
    require(byKey.size == nodes.size, "flow graph stage keys must be unique")
    val ordered = topologicalOrder(nodes, byKey)
    val completion = mutable.HashMap.empty[AutotuneStageKey, Double]
    val maximizingParents = mutable.HashMap.empty[AutotuneStageKey, Seq[AutotuneStageKey]]

    ordered.foreach { node =>
      val knownParents = node.parents.filter(byKey.contains)
      val maximum = if (knownParents.isEmpty) 0.0 else knownParents.map(completion).max
      val tolerance = math.max(1.0, math.abs(maximum)) * 1e-12
      maximizingParents.put(node.key,
        knownParents.filter(parent => math.abs(completion(parent) - maximum) <= tolerance))
      completion.put(node.key, maximum + math.max(0.0, node.evaluation.predictedNanos))
    }

    val adjoints = mutable.HashMap.empty[AutotuneStageKey, Double].withDefaultValue(0.0)
    var objective = 0.0
    ordered.groupBy(_.key.executionId).values.foreach { executionNodes =>
      val maximum = executionNodes.map(node => completion(node.key)).max
      objective += maximum
      val tolerance = math.max(1.0, math.abs(maximum)) * 1e-12
      val outputs = executionNodes.filter { node =>
        math.abs(completion(node.key) - maximum) <= tolerance
      }
      val share = 1.0 / outputs.size.toDouble
      outputs.foreach(node => adjoints.update(node.key, adjoints(node.key) + share))
    }

    ordered.reverse.foreach { node =>
      val parents = maximizingParents(node.key)
      if (parents.nonEmpty) {
        val share = adjoints(node.key) / parents.size.toDouble
        parents.foreach(parent => adjoints.update(parent, adjoints(parent) + share))
      }
    }
    val durationAdjoints = ordered.map(node => node.key -> adjoints(node.key)).toMap
    val gradients = ordered.map { node =>
      node.key -> node.evaluation.gradient.scale(adjoints(node.key))
    }.toMap
    GpuFlowGraphEvaluation(objective, completion.toMap, durationAdjoints, gradients)
  }

  private def topologicalOrder(
      nodes: Seq[GpuFlowGraphNode],
      byKey: Map[AutotuneStageKey, GpuFlowGraphNode]): Seq[GpuFlowGraphNode] = {
    val visiting = mutable.HashSet.empty[AutotuneStageKey]
    val visited = mutable.HashSet.empty[AutotuneStageKey]
    val ordered = mutable.ArrayBuffer.empty[GpuFlowGraphNode]
    def visit(node: GpuFlowGraphNode): Unit = {
      require(!visiting.contains(node.key), s"cycle in flow graph at ${node.key}")
      if (!visited.contains(node.key)) {
        visiting += node.key
        node.parents.flatMap(byKey.get).foreach(visit)
        visiting -= node.key
        visited += node.key
        ordered += node
      }
    }
    nodes.sortBy(node => (node.key.executionId, node.key.stageId, node.key.stageAttemptId))
      .foreach(visit)
    ordered.toSeq
  }
}

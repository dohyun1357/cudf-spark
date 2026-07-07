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
 * are expressed in the actuator's native unit.
 */
private[rapids] case class GpuFlowControl(
    scanWindow: Double,
    shuffleWindow: Double,
    shuffleBytes: Double)

/** Calibrated work at the controls that produced an observation window. */
private[rapids] case class GpuFlowStageWork(
    scanNanos: Double,
    gpuNanos: Double,
    shuffleNanos: Double,
    fixedNanos: Double,
    retryNanos: Double,
    baseline: GpuFlowControl)

/**
 * Derivative of stage latency with respect to log(load) of each concurrent service lane. Batch
 * work is never measured separately from the residual, so there is no batch lane; both shuffle
 * controls act on the single shuffle lane.
 */
private[rapids] case class GpuFlowGradient(
    scan: Double = 0.0,
    gpu: Double = 0.0,
    shuffle: Double = 0.0) {
  def scale(factor: Double): GpuFlowGradient = GpuFlowGradient(
    scan * factor,
    gpu * factor,
    shuffle * factor)
}

private[rapids] case class GpuFlowStageEvaluation(
    predictedNanos: Double,
    gradient: GpuFlowGradient)

/**
 * Local flow model for one Spark stage at its measured operating point.
 *
 * Scan, GPU, and shuffle are concurrent producer/consumer lanes, so stage service time contains
 * their maximum. Measured retry time is retained in the objective. Spill bytes do not have time
 * units or an identifiable derivative from one observation window, so this model does not invent
 * a resource-pressure penalty from them. The gradient assumes inverse scaling in log(control) at
 * the measured point and is reported as eventlog evidence only; no continuous control moves until
 * multi-setting response evidence exists.
 */
private[rapids] object GpuFlowStageModel {
  def evaluate(work: GpuFlowStageWork): GpuFlowStageEvaluation = {
    val scan = math.max(0.0, work.scanNanos)
    val gpu = math.max(0.0, work.gpuNanos)
    val shuffle = math.max(0.0, work.shuffleNanos)

    val lanes = Array(scan, gpu, shuffle)
    val bottleneck = lanes.max
    val tieTolerance = math.max(1.0, math.abs(bottleneck)) * 1e-12
    val criticalLaneCount = lanes.count(value => math.abs(value - bottleneck) <= tieTolerance)
    val laneShare = if (criticalLaneCount > 0) 1.0 / criticalLaneCount.toDouble else 0.0
    def laneGradient(lane: Double): Double =
      if (math.abs(lane - bottleneck) <= tieTolerance) -lane * laneShare else 0.0

    GpuFlowStageEvaluation(
      predictedNanos = bottleneck + work.fixedNanos + work.retryNanos,
      gradient = GpuFlowGradient(
        scan = laneGradient(scan),
        gpu = laneGradient(gpu),
        shuffle = laneGradient(shuffle)))
  }
}

/** One node in the remaining Spark execution graph. */
private[rapids] case class GpuFlowGraphNode(
    key: AutotuneStageKey,
    parents: Seq[AutotuneStageKey],
    durationNanos: Double)

private[rapids] case class GpuFlowGraphEvaluation(
    objectiveNanos: Double,
    durationAdjoints: Map[AutotuneStageKey, Double])

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
      return GpuFlowGraphEvaluation(0.0, Map.empty)
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
      completion.put(node.key, maximum + math.max(0.0, node.durationNanos))
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
    GpuFlowGraphEvaluation(objective, ordered.map(node => node.key -> adjoints(node.key)).toMap)
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

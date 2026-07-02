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

import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.SparkConf
import org.apache.spark.sql.execution.adaptive.Cost

class GpuFlowAqeCostEvaluatorSuite extends AnyFunSuite {
  private case class RankedCost(value: Int) extends Cost {
    override def compare(other: Cost): Int = other match {
      case that: RankedCost => Integer.compare(value, that.value)
      case _ => 0
    }
  }

  private def cost(
      evaluation: GpuFlowAqePlanEvaluation,
      fallback: Int = 0): GpuFlowAqeCost =
    GpuFlowAqeCost(evaluation, RankedCost(fallback))

  private val fixedControl = GpuFlowControl(1.0, 1.0, 1.0, 1.0, 1.0)
  private val fixedBounds = GpuFlowControlBounds(fixedControl, fixedControl)

  private def calibration(
      scan: Option[Double] = Some(1.0),
      gpu: Option[Double] = Some(1.0),
      shuffle: Option[Double] = Some(1.0),
      broadcast: Option[Double] = Some(1.0),
      batch: Option[Double] = Some(1.0),
      fixed: Option[Double] = Some(1.0)): GpuFlowAqeCalibration =
    GpuFlowAqeCalibration(
      scanUnitNanosPerByte = scan,
      gpuUnitNanosPerByte = gpu,
      shuffleUnitNanosPerByte = shuffle,
      broadcastNanosPerByte = broadcast,
      batchUnitNanosPerByte = batch,
      fixedNanosPerTask = fixed,
      referenceControl = fixedControl,
      bounds = fixedBounds,
      sampleWindows = 4L)

  test("AQE flow cost chooses a lower complete-plan objective") {
    val input = AutotuneStageKey(0L, 1, 0)
    val join = AutotuneStageKey(0L, 2, 0)
    val shuffleNodes = Seq(
      GpuFlowAqeNode(input, Seq.empty, GpuFlowAqeDemand(shuffleBytes = 1000.0)),
      GpuFlowAqeNode(join, Seq(input), GpuFlowAqeDemand(gpuBytes = 1000.0,
        batchBytes = 1000.0)))
    val broadcastNodes = Seq(
      GpuFlowAqeNode(input, Seq.empty, GpuFlowAqeDemand(broadcastBytes = 1000.0)),
      GpuFlowAqeNode(join, Seq(input), GpuFlowAqeDemand(gpuBytes = 1000.0,
        batchBytes = 1000.0)))
    val measured = calibration(shuffle = Some(2.0), broadcast = Some(0.25))

    val shufflePlan = GpuFlowAqePlanModel.evaluateNodes(
      shuffleNodes, measured, "join", "shuffle")
    val broadcastPlan = GpuFlowAqePlanModel.evaluateNodes(
      broadcastNodes, measured, "join", "broadcast")

    assert(shufflePlan.identifiable && broadcastPlan.identifiable)
    assert(broadcastPlan.objectiveNanos < shufflePlan.objectiveNanos)
    assert(cost(broadcastPlan) < cost(shufflePlan))
  }

  test("identical physical structure delegates statistics refinement to Spark") {
    val key = AutotuneStageKey(0L, 1, 0)
    val stale = GpuFlowAqePlanModel.evaluateNodes(
      Seq(GpuFlowAqeNode(key, Seq.empty, GpuFlowAqeDemand(shuffleBytes = 1000.0))),
      calibration(), "join", "shuffle")
    val refreshed = GpuFlowAqePlanModel.evaluateNodes(
      Seq(GpuFlowAqeNode(key, Seq.empty, GpuFlowAqeDemand(shuffleBytes = 1005.0))),
      calibration(), "join", "shuffle")

    assert(stale.identifiable && refreshed.identifiable)
    assert(stale.objectiveNanos < refreshed.objectiveNanos)
    // Same operators and same topology: the objective delta is refreshed runtime statistics,
    // not a physical alternative, so Spark's native contract owns the comparison and its equal
    // native cost adopts the refreshed plan.
    assert(cost(stale, fallback = 1).compare(cost(refreshed, fallback = 1)) == 0)
    assert(cost(stale, fallback = 1) == cost(refreshed, fallback = 1))
    assert(cost(refreshed, fallback = 2) > cost(stale, fallback = 1))
  }

  test("AQE flow cost ties when either complete plan is not identifiable") {
    val key = AutotuneStageKey(0L, 1, 0)
    val nodes = Seq(GpuFlowAqeNode(
      key, Seq.empty, GpuFlowAqeDemand(broadcastBytes = 1000.0)))
    // Structurally distinct candidates (different topology): an unidentified rate must freeze
    // the current plan. A same-structure pair is a statistics refresh and delegates to Spark
    // regardless of identifiability.
    val known = GpuFlowAqePlanModel.evaluateNodes(
      nodes, calibration(), "join", "broadcast")
    val unknown = GpuFlowAqePlanModel.evaluateNodes(
      nodes, calibration(broadcast = None), "join", "broadcast-reordered")

    assert(known.identifiable)
    assert(!unknown.identifiable)
    assert(unknown.reason == "missing-broadcast-rate")
    assert(cost(known).compare(cost(unknown)) == 0)
    assert(cost(unknown).compare(cost(known)) == 0)
    assert(cost(known) != cost(unknown))
  }

  test("AQE preserves Spark authority until different operator responses are measured") {
    val key = AutotuneStageKey(0L, 1, 0)
    val slow = GpuFlowAqePlanModel.evaluateNodes(
      Seq(GpuFlowAqeNode(key, Seq.empty, GpuFlowAqeDemand(shuffleBytes = 1000.0))),
      calibration(), "sort|join", "shuffle-sort-merge")
    val fast = GpuFlowAqePlanModel.evaluateNodes(
      Seq(GpuFlowAqeNode(key, Seq.empty, GpuFlowAqeDemand(broadcastBytes = 100.0))),
      calibration(), "join", "broadcast-hash")

    assert(fast.objectiveNanos < slow.objectiveNanos)
    assert(cost(fast, fallback = 2) > cost(slow, fallback = 1))
    assert(cost(fast, fallback = 1) == cost(slow, fallback = 1))
  }

  test("AQE uses the flow objective for the same measured operator basis") {
    val key = AutotuneStageKey(0L, 1, 0)
    val slow = GpuFlowAqePlanModel.evaluateNodes(
      Seq(GpuFlowAqeNode(key, Seq.empty, GpuFlowAqeDemand(shuffleBytes = 1000.0))),
      calibration(), "join", "shuffle-left")
    val fast = GpuFlowAqePlanModel.evaluateNodes(
      Seq(GpuFlowAqeNode(key, Seq.empty, GpuFlowAqeDemand(shuffleBytes = 100.0))),
      calibration(), "join", "shuffle-right")

    assert(cost(fast, fallback = 2) < cost(slow, fallback = 1))
  }

  test("AQE preserves Spark ordering when distinct plans collapse to one flow objective") {
    val key = AutotuneStageKey(0L, 1, 0)
    val tied = GpuFlowAqePlanModel.evaluateNodes(
      Seq(GpuFlowAqeNode(key, Seq.empty, GpuFlowAqeDemand(shuffleBytes = 100.0))),
      calibration(), "join", "same-modeled-topology")

    assert(cost(tied, fallback = 2) > cost(tied, fallback = 1))
    assert(cost(tied, fallback = 1) == cost(tied, fallback = 1))
  }

  test("AQE flow plan uses max-plus branch completion instead of summing branches") {
    val left = AutotuneStageKey(0L, 1, 0)
    val right = AutotuneStageKey(0L, 2, 0)
    val join = AutotuneStageKey(0L, 3, 0)
    val nodes = Seq(
      GpuFlowAqeNode(left, Seq.empty, GpuFlowAqeDemand(gpuBytes = 100.0)),
      GpuFlowAqeNode(right, Seq.empty, GpuFlowAqeDemand(gpuBytes = 50.0)),
      GpuFlowAqeNode(join, Seq(left, right), GpuFlowAqeDemand(gpuBytes = 10.0)))

    val evaluation = GpuFlowAqePlanModel.evaluateNodes(
      nodes, calibration(batch = None), "join", "two-branch")
    assert(evaluation.identifiable)
    assert(evaluation.objectiveNanos == 110.0)
  }

  test("AQE cost event preserves model evidence") {
    val key = AutotuneStageKey(0L, 1, 0)
    val evaluation = GpuFlowAqePlanModel.evaluateNodes(
      Seq(GpuFlowAqeNode(key, Seq.empty, GpuFlowAqeDemand(shuffleBytes = 100.0))),
      calibration(), "aggregate", "shuffle")
    val event = RapidsAutotuneDriverEndpoint.toAqeCostEvent(7L, 11L, evaluation)

    assert(event.evaluationId == 7L && event.executionId == 11L)
    assert(event.identifiable)
    assert(event.objectiveNanos == evaluation.objectiveNanos)
    assert(event.sparkFallbackCost == 0L)
    assert(event.shuffleBytes == 100.0)
    assert(event.calibrationSampleWindows == 4L)
  }

  test("OPTIMIZE installs AQE authority without overwriting a user evaluator") {
    val optimize = new SparkConf(false)
      .set(RapidsConf.AUTOTUNE_GRAPH_ENABLED.key, "true")
      .set(RapidsConf.AUTOTUNE_GRAPH_MODE.key, AutotuneGraphMode.OPTIMIZE.toString)
    assert(GpuFlowAqeCostEvaluator.configureIfNeeded(optimize).isEmpty)
    assert(optimize.get(GpuFlowAqeCostEvaluator.SparkCostEvaluatorKey) ==
      GpuFlowAqeCostEvaluator.ClassName)

    val graphOnly = new SparkConf(false)
      .set(RapidsConf.AUTOTUNE_GRAPH_ENABLED.key, "true")
      .set(RapidsConf.AUTOTUNE_GRAPH_MODE.key, AutotuneGraphMode.GRAPH.toString)
    assert(GpuFlowAqeCostEvaluator.configureIfNeeded(graphOnly).isEmpty)
    assert(!graphOnly.contains(GpuFlowAqeCostEvaluator.SparkCostEvaluatorKey))

    val custom = optimize.clone().set(
      GpuFlowAqeCostEvaluator.SparkCostEvaluatorKey, "example.UserCostEvaluator")
    assert(GpuFlowAqeCostEvaluator.configureIfNeeded(custom).contains(
      "example.UserCostEvaluator"))
    assert(custom.get(GpuFlowAqeCostEvaluator.SparkCostEvaluatorKey) ==
      "example.UserCostEvaluator")
  }
}

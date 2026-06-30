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

class GpuFlowAqeCostEvaluatorSuite extends AnyFunSuite {
  private val fixedControl = GpuFlowControl(1.0, 1.0, 1.0, 1.0, 1.0)
  private val fixedBounds = GpuFlowControlBounds(fixedControl, fixedControl)

  private def calibration(
      scan: Option[Double] = Some(1.0),
      gpu: Option[Double] = Some(1.0),
      shuffle: Option[Double] = Some(1.0),
      broadcast: Option[Double] = Some(1.0),
      batch: Option[Double] = Some(1.0)): GpuFlowAqeCalibration =
    GpuFlowAqeCalibration(
      scanUnitNanosPerByte = scan,
      gpuUnitNanosPerByte = gpu,
      shuffleUnitNanosPerByte = shuffle,
      broadcastNanosPerByte = broadcast,
      batchUnitNanosPerByte = batch,
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
    assert(GpuFlowAqeCost(broadcastPlan) < GpuFlowAqeCost(shufflePlan))
  }

  test("AQE flow cost ties when either complete plan is not identifiable") {
    val key = AutotuneStageKey(0L, 1, 0)
    val nodes = Seq(GpuFlowAqeNode(
      key, Seq.empty, GpuFlowAqeDemand(broadcastBytes = 1000.0)))
    val known = GpuFlowAqePlanModel.evaluateNodes(
      nodes, calibration(), "join", "broadcast")
    val unknown = GpuFlowAqePlanModel.evaluateNodes(
      nodes, calibration(broadcast = None), "join", "broadcast")

    assert(known.identifiable)
    assert(!unknown.identifiable)
    assert(unknown.reason == "missing-broadcast-rate")
    assert(GpuFlowAqeCost(known).compare(GpuFlowAqeCost(unknown)) == 0)
    assert(GpuFlowAqeCost(unknown).compare(GpuFlowAqeCost(known)) == 0)
  }

  test("AQE compares identifiable physical alternatives with different operator fingerprints") {
    val key = AutotuneStageKey(0L, 1, 0)
    val slow = GpuFlowAqePlanModel.evaluateNodes(
      Seq(GpuFlowAqeNode(key, Seq.empty, GpuFlowAqeDemand(shuffleBytes = 1000.0))),
      calibration(), "sort|join", "shuffle-sort-merge")
    val fast = GpuFlowAqePlanModel.evaluateNodes(
      Seq(GpuFlowAqeNode(key, Seq.empty, GpuFlowAqeDemand(broadcastBytes = 100.0))),
      calibration(), "join", "broadcast-hash")

    assert(fast.objectiveNanos < slow.objectiveNanos)
    assert(GpuFlowAqeCost(fast) < GpuFlowAqeCost(slow))
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

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

class GpuFlowModelSuite extends AnyFunSuite {
  private val key = AutotuneStageKey(executionId = 11L, stageId = 3, stageAttemptId = 1)

  test("flow-stage analytic gradients match finite differences") {
    val baseline = GpuFlowControl(
      scanWindow = 2.0,
      gpuTasks = 2.0,
      shuffleWindow = 2.0,
      shuffleBytes = 128.0,
      batchBytes = 256.0)
    val work = GpuFlowStageWork(
      scanNanos = 1000.0,
      gpuNanos = 600.0,
      shuffleNanos = 400.0,
      batchNanos = 200.0,
      fixedNanos = 100.0,
      retryNanos = 20.0,
      baseline = baseline)
    val analytic = GpuFlowStageModel.evaluate(work, baseline)
    val epsilon = 1e-6

    GpuFlowControl.Indices.foreach { index =>
      val plus = baseline.updated(index, baseline.value(index) * math.exp(epsilon))
      val minus = baseline.updated(index, baseline.value(index) * math.exp(-epsilon))
      val finiteDifference = (
        GpuFlowStageModel.evaluate(work, plus).predictedNanos -
          GpuFlowStageModel.evaluate(work, minus).predictedNanos) / (2.0 * epsilon)
      assert(math.abs(analytic.gradient.value(index) - finiteDifference) < 1e-4,
        s"control $index: analytic=${analytic.gradient.value(index)} " +
          s"finiteDifference=$finiteDifference")
    }
  }

  test("flow-graph reverse pass assigns end-to-end criticality") {
    val left = key.copy(stageId = 20)
    val right = key.copy(stageId = 21)
    val join = key.copy(stageId = 22)
    def node(stageKey: AutotuneStageKey, parents: Seq[AutotuneStageKey], nanos: Double) =
      GpuFlowGraphNode(stageKey, parents,
        GpuFlowStageEvaluation(nanos, GpuFlowGradient(gpuTasks = -nanos)))

    val evaluation = GpuFlowModel.evaluate(Seq(
      node(left, Seq.empty, 100.0),
      node(right, Seq.empty, 60.0),
      node(join, Seq(left, right), 20.0)))

    assert(evaluation.objectiveNanos == 120.0)
    assert(evaluation.durationAdjoints(left) == 1.0)
    assert(evaluation.durationAdjoints(right) == 0.0)
    assert(evaluation.durationAdjoints(join) == 1.0)
    assert(evaluation.controlGradients(left).gpuTasks == -100.0)
    assert(evaluation.controlGradients(right).gpuTasks == 0.0)
  }

  test("flow-graph reverse pass shares exact critical-path ties") {
    val left = key.copy(stageId = 30)
    val right = key.copy(stageId = 31)
    val join = key.copy(stageId = 32)
    def node(stageKey: AutotuneStageKey, parents: Seq[AutotuneStageKey], nanos: Double) =
      GpuFlowGraphNode(stageKey, parents,
        GpuFlowStageEvaluation(nanos, GpuFlowGradient()))

    val evaluation = GpuFlowModel.evaluate(Seq(
      node(left, Seq.empty, 100.0),
      node(right, Seq.empty, 100.0),
      node(join, Seq(left, right), 20.0)))

    assert(evaluation.durationAdjoints(left) == 0.5)
    assert(evaluation.durationAdjoints(right) == 0.5)
    assert(evaluation.durationAdjoints(join) == 1.0)
  }

  test("projected flow optimizer decreases the modeled objective within bounds") {
    val baseline = GpuFlowControl(2.0, 2.0, 2.0, 128.0, 256.0)
    val bounds = GpuFlowControlBounds(
      GpuFlowControl(1.0, 2.0, 1.0, 64.0, 128.0),
      GpuFlowControl(8.0, 2.0, 8.0, 1024.0, 2048.0))
    val work = GpuFlowStageWork(
      scanNanos = 1000.0,
      gpuNanos = 500.0,
      shuffleNanos = 800.0,
      batchNanos = 200.0,
      fixedNanos = 100.0,
      retryNanos = 0.0,
      baseline = baseline)

    val selected = GpuFlowProjectedOptimizer.optimize(work, baseline, bounds)
    assert(selected.gpuTasks == 2.0)
    assert(GpuFlowStageModel.evaluate(work, selected).predictedNanos <
      GpuFlowStageModel.evaluate(work, baseline).predictedNanos)
    GpuFlowControl.Indices.foreach { index =>
      assert(selected.value(index) >= bounds.minimum.value(index))
      assert(selected.value(index) <= bounds.maximum.value(index))
    }
  }

}

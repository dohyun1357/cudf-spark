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
  private val baseline = GpuFlowControl(
    scanWindow = 2.0,
    gpuTasks = 0.0,
    shuffleWindow = 2.0,
    shuffleBytes = 128.0,
    batchBytes = 256.0)

  test("flow-stage evaluation sums the bottleneck lane with fixed and retry work") {
    val work = GpuFlowStageWork(
      scanNanos = 1000.0,
      gpuNanos = 600.0,
      shuffleNanos = 400.0,
      fixedNanos = 100.0,
      retryNanos = 20.0,
      baseline = baseline)
    val evaluation = GpuFlowStageModel.evaluate(work)

    assert(evaluation.predictedNanos == 1120.0)
    // Only the critical scan lane carries sensitivity at the measured point.
    assert(evaluation.gradient.scanWindow == -1000.0)
    assert(evaluation.gradient.gpuTasks == 0.0)
    assert(evaluation.gradient.shuffleWindow == 0.0)
    assert(evaluation.gradient.shuffleBytes == 0.0)
    assert(evaluation.gradient.batchBytes == 0.0)
  }

  test("flow-stage evaluation shares exact lane ties") {
    val work = GpuFlowStageWork(
      scanNanos = 500.0,
      gpuNanos = 500.0,
      shuffleNanos = 100.0,
      fixedNanos = 0.0,
      retryNanos = 0.0,
      baseline = baseline)
    val evaluation = GpuFlowStageModel.evaluate(work)

    assert(evaluation.predictedNanos == 500.0)
    assert(evaluation.gradient.scanWindow == -250.0)
    assert(evaluation.gradient.gpuTasks == -250.0)
    assert(evaluation.gradient.shuffleWindow == 0.0)
  }

  test("flow-graph reverse pass assigns end-to-end criticality") {
    val left = key.copy(stageId = 20)
    val right = key.copy(stageId = 21)
    val join = key.copy(stageId = 22)

    val evaluation = GpuFlowModel.evaluate(Seq(
      GpuFlowGraphNode(left, Seq.empty, 100.0),
      GpuFlowGraphNode(right, Seq.empty, 60.0),
      GpuFlowGraphNode(join, Seq(left, right), 20.0)))

    assert(evaluation.objectiveNanos == 120.0)
    assert(evaluation.durationAdjoints(left) == 1.0)
    assert(evaluation.durationAdjoints(right) == 0.0)
    assert(evaluation.durationAdjoints(join) == 1.0)
  }

  test("flow-graph reverse pass shares exact critical-path ties") {
    val left = key.copy(stageId = 30)
    val right = key.copy(stageId = 31)
    val join = key.copy(stageId = 32)

    val evaluation = GpuFlowModel.evaluate(Seq(
      GpuFlowGraphNode(left, Seq.empty, 100.0),
      GpuFlowGraphNode(right, Seq.empty, 100.0),
      GpuFlowGraphNode(join, Seq(left, right), 20.0)))

    assert(evaluation.durationAdjoints(left) == 0.5)
    assert(evaluation.durationAdjoints(right) == 0.5)
    assert(evaluation.durationAdjoints(join) == 1.0)
  }
}

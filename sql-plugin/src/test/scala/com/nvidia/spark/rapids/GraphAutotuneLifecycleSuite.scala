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

class GraphAutotuneLifecycleSuite extends AnyFunSuite {
  private val key = AutotuneStageKey(executionId = 11L, stageId = 3, stageAttemptId = 1)

  test("native GPU task slots respect both CPU and Spark GPU resources") {
    assert(GpuTaskSlotEstimator.estimate(8, Some(0.25), Some(1.0)) == 4)
    assert(GpuTaskSlotEstimator.estimate(2, Some(0.25), Some(2.0)) == 2)
    assert(GpuTaskSlotEstimator.estimate(8, None, None) == 8)
  }

  test("SQL execution cleanup removes only that execution's runtime state") {
    val conf = new RapidsConf(Map(RapidsConf.AUTOTUNE_GRAPH_ENABLED.key -> "true"))
    val otherKey = key.copy(executionId = key.executionId + 1, stageId = 4)
    RapidsAutotuneDriverEndpoint.init(null, conf)
    try {
      val first = RapidsAutotuneDriverEndpoint.publishDefaultNoopHint(key)
      val other = RapidsAutotuneDriverEndpoint.publishDefaultNoopHint(otherKey)
      RapidsAutotuneDriverEndpoint.handleObservation(
        RapidsAutotuneObservationMsg(
          executorId = "exec-1",
          key = key,
          taskAttemptId = 1L,
          partitionId = 0,
          hintVersion = first.version,
          gpuSemaphoreWaitNanos = 1L,
          gpuHoldingNanos = 2L,
          hostMemoryBytes = 3L))

      assert(RapidsAutotuneDriverEndpoint.observationFor(key).nonEmpty)
      RapidsAutotuneDriverEndpoint.executionCompleted(key.executionId)

      assert(RapidsAutotuneDriverEndpoint.observationFor(key).isEmpty)
      assert(RapidsAutotuneDriverEndpoint.handleHintRequest(
        RapidsAutotuneHintRequestMsg("exec-1", key)).hint.isEmpty)
      assert(RapidsAutotuneDriverEndpoint.handleHintRequest(
        RapidsAutotuneHintRequestMsg("exec-1", otherKey)).hint.contains(other))
    } finally {
      RapidsAutotuneDriverEndpoint.shutdown()
    }
  }
}

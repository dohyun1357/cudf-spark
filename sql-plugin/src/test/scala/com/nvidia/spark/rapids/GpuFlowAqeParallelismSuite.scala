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

class GpuFlowAqeParallelismSuite extends AnyFunSuite {
  private val constraints = GraphOptimizerConstraints(
    minSampleTasks = 1L,
    updateIntervalNanos = 0L,
    scan = ScanOptimizerBounds(false, 0, 0, 0L),
    gpu = GpuOptimizerBounds(true, 4, 4),
    shuffle = ShuffleOptimizerBounds(false, 0, 0, 0L, 0L, 0L),
    batch = BatchOptimizerBounds(false, 0L, 0L, 0L))

  private def calibrationSample(
      nonGpuNanos: Double,
      ioBytes: Long,
      tasks: Long): GpuFlowCalibrationSample = GpuFlowCalibrationSample(
    scanUnitNanos = 0.0,
    scanBytes = 0L,
    gpuUnitNanos = 0.0,
    gpuBytes = 0L,
    shuffleUnitNanos = 0.0,
    shuffleBytes = 0L,
    broadcastNanos = 0.0,
    broadcastBytes = 0L,
    batchUnitNanos = 0.0,
    batchBytes = 0L,
    nonGpuNanos = nonGpuNanos,
    ioBytes = ioBytes,
    taskCount = tasks)

  test("balanced reducer ranges are contiguous complete and exact-count") {
    val ranges = GpuFlowPartitionOptimizer.balancedRanges(
      Seq(9L, 1L, 8L, 2L, 7L, 3L), targetCount = 3)

    assert(ranges.size == 3)
    assert(ranges.head._1 == 0)
    assert(ranges.last._2 == 6)
    assert(ranges.sliding(2).forall(pair => pair.head._2 == pair.last._1))
    assert(ranges.forall { case (start, end) => end > start })
  }

  test("parallelism solve balances service without adding a fixed-cost wave") {
    val bytes = Seq.fill(8)(10L)
    val current = Seq((0, 4), (4, 8))
    val selected = GpuFlowPartitionOptimizer.optimize(
      bytes,
      current,
      taskSlots = 4,
      variableNanosPerByte = 1.0,
      fixedNanosPerTask = 100.0).get

    assert(selected.ranges.size == 4)
    assert(selected.objectiveNanos == 120.0)
  }

  test("parallelism solve retains an exact current-layout tie") {
    val bytes = Seq.fill(4)(10L)
    val current = Seq((0, 1), (1, 2), (2, 3), (3, 4))
    val selected = GpuFlowPartitionOptimizer.optimize(
      bytes,
      current,
      taskSlots = 4,
      variableNanosPerByte = 1.0,
      fixedNanosPerTask = 100.0).get

    assert(selected.ranges.map(range =>
      (range.startReducerIndex, range.endReducerIndex)) == current)
  }

  test("fixed task cost is identified from distinct measured bytes per task") {
    val accumulator = new GpuFlowAqeCalibrationAccumulator(constraints)
    // y/task = 2 * bytes/task + 50
    accumulator.add(calibrationSample(nonGpuNanos = 2500.0, ioBytes = 1000L, tasks = 10L))
    accumulator.add(calibrationSample(nonGpuNanos = 4500.0, ioBytes = 2000L, tasks = 10L))

    assert(math.abs(accumulator.snapshot.fixedNanosPerTask.get - 50.0) < 1e-9)
  }
}

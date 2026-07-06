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
    scan = ScanOptimizerBounds(false, 0, 0L),
    shuffle = ShuffleOptimizerBounds(false, 0, 0L, 0L),
    batch = BatchOptimizerBounds(false, 0L, 0L))

  private def calibrationSample(
      nonGpuNanos: Double,
      ioBytes: Long,
      tasks: Long,
      taskShuffleReadBytes: Long = 0L): GpuFlowCalibrationSample = GpuFlowCalibrationSample(
    shuffleUnitNanos = 0.0,
    shuffleBytes = 0L,
    taskShuffleReadBytes = taskShuffleReadBytes,
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

  test("ranges beyond the identified byte region are never selected") {
    val bytes = Seq.fill(8)(10L)
    val current = bytes.indices.map(index => (index, index + 1))
    // Balanced 4-range layouts carry 20 bytes per range; a 10-byte region excludes them, so the
    // wave-cheaper coalesce is rejected and the current identity layout is retained.
    val capped = GpuFlowPartitionOptimizer.optimize(
      bytes, current, taskSlots = 4, variableNanosPerByte = 1.0,
      fixedNanosPerTask = 100.0, maxPartitions = 8, maxRangeBytes = 10L).get
    val uncapped = GpuFlowPartitionOptimizer.optimize(
      bytes, current, taskSlots = 4, variableNanosPerByte = 1.0,
      fixedNanosPerTask = 100.0, maxPartitions = 8).get

    assert(uncapped.ranges.size == 4)
    assert(capped.ranges.size == 8)
    assert(capped.ranges.map(range =>
      (range.startReducerIndex, range.endReducerIndex)) == current)
  }

  test("calibration snapshot carries the largest observed per-task shuffle-read load") {
    val accumulator = new GpuFlowAqeCalibrationAccumulator(constraints)
    accumulator.add(calibrationSample(1000.0, 100L, 10L, taskShuffleReadBytes = 64L))
    accumulator.add(calibrationSample(1000.0, 100L, 10L, taskShuffleReadBytes = 256L))
    accumulator.add(calibrationSample(1000.0, 100L, 10L, taskShuffleReadBytes = 128L))

    assert(accumulator.snapshot.maxCalibratedTaskShuffleReadBytes == 256L)
    assert(new GpuFlowAqeCalibrationAccumulator(constraints)
      .snapshot.maxCalibratedTaskShuffleReadBytes == 0L)
  }

  test("parallelism solve can freeze candidates above Spark's current layout") {
    val bytes = Seq.fill(8)(10L)
    val current = Seq((0, 8))
    val unconstrained = GpuFlowPartitionOptimizer.optimize(
      bytes, current, taskSlots = 4, variableNanosPerByte = 1.0,
      fixedNanosPerTask = 100.0).get
    val bounded = GpuFlowPartitionOptimizer.optimize(
      bytes, current, taskSlots = 4, variableNanosPerByte = 1.0,
      fixedNanosPerTask = 100.0, maxPartitions = current.size).get

    assert(unconstrained.ranges.size == 4)
    assert(bounded.ranges.size == 1)
  }

  test("cluster-width task slots keep a single-wave layout a per-executor count collapses") {
    // 100 partitions of 10 MiB on a 128-slot cluster already run in one wave, so the layout must
    // be retained. Pricing the same layout with one executor's 16 slots wrongly charges seven
    // fixed-cost waves and coalesces real cluster parallelism down to 16 tasks.
    val bytes = Seq.fill(100)(10L * 1024 * 1024)
    val current = bytes.indices.map(index => (index, index + 1))
    val clusterWide = GpuFlowPartitionOptimizer.optimize(
      bytes, current, taskSlots = 128, variableNanosPerByte = 1.0,
      fixedNanosPerTask = 1.0e9).get
    val perExecutor = GpuFlowPartitionOptimizer.optimize(
      bytes, current, taskSlots = 16, variableNanosPerByte = 1.0,
      fixedNanosPerTask = 1.0e9).get

    assert(clusterWide.ranges.size == 100)
    assert(perExecutor.ranges.size == 16)
  }

  test("calibration snapshot does not fabricate cluster task slots") {
    val calibration = new GpuFlowAqeCalibrationAccumulator(constraints).snapshot
    assert(calibration.clusterTaskSlots == 0)
  }

  test("fixed task cost is identified from distinct measured bytes per task") {
    val accumulator = new GpuFlowAqeCalibrationAccumulator(constraints)
    // y/task = 2 * bytes/task + 50
    accumulator.add(calibrationSample(nonGpuNanos = 2500.0, ioBytes = 1000L, tasks = 10L))
    accumulator.add(calibrationSample(nonGpuNanos = 4500.0, ioBytes = 2000L, tasks = 10L))
    accumulator.add(calibrationSample(nonGpuNanos = 6500.0, ioBytes = 3000L, tasks = 10L))

    val calibration = accumulator.snapshot
    assert(math.abs(calibration.fixedNanosPerTask.get - 50.0) < 1e-9)
    assert(calibration.fixedNanosPerTaskStandardError.contains(0.0))
    assert(calibration.fixedTaskCostSampleWindows == 3L)
    assert(calibration.fixedTaskCostReason.isEmpty)
  }

  test("zero-intercept boundary fit does not identify fixed task cost") {
    val accumulator = new GpuFlowAqeCalibrationAccumulator(constraints)
    // y/task = 2 * bytes/task. A non-negative fit can return fixed=0, but zero is not evidence
    // that task launch and setup are free.
    accumulator.add(calibrationSample(nonGpuNanos = 2000.0, ioBytes = 1000L, tasks = 10L))
    accumulator.add(calibrationSample(nonGpuNanos = 4000.0, ioBytes = 2000L, tasks = 10L))
    accumulator.add(calibrationSample(nonGpuNanos = 6000.0, ioBytes = 3000L, tasks = 10L))

    val calibration = accumulator.snapshot
    assert(calibration.fixedNanosPerTask.isEmpty)
    assert(calibration.fixedTaskCostReason == "boundary-fixed-task-cost-fit")
  }

  test("uncertain intercept does not identify fixed task cost") {
    val accumulator = new GpuFlowAqeCalibrationAccumulator(constraints)
    // The point estimate has a positive intercept, but its confidence interval crosses zero.
    accumulator.add(calibrationSample(nonGpuNanos = 2500.0, ioBytes = 1000L, tasks = 10L))
    accumulator.add(calibrationSample(nonGpuNanos = 5500.0, ioBytes = 2000L, tasks = 10L))
    accumulator.add(calibrationSample(nonGpuNanos = 5500.0, ioBytes = 3000L, tasks = 10L))
    accumulator.add(calibrationSample(nonGpuNanos = 9500.0, ioBytes = 4000L, tasks = 10L))

    val calibration = accumulator.snapshot
    assert(calibration.fixedNanosPerTask.isEmpty)
    assert(calibration.fixedNanosPerTaskStandardError.exists(_ > 0.0))
    assert(calibration.fixedTaskCostReason == "uncertain-fixed-task-cost-fit")
  }

  test("direct task setup metric supplies a conservative fixed-cost lower bound") {
    val accumulator = new GpuFlowAqeCalibrationAccumulator(constraints)
    // The duration/byte regression is a zero-intercept boundary fit, but executor deserialize is
    // directly paid once per task. Use the smallest observed task value as the lower bound.
    accumulator.add(calibrationSample(2000.0, 1000L, 10L))
    accumulator.add(calibrationSample(4000.0, 2000L, 10L))
    accumulator.add(calibrationSample(6000.0, 3000L, 10L))
    accumulator.addTaskSetupNanos(30L)
    accumulator.addTaskSetupNanos(20L)
    accumulator.addTaskSetupNanos(25L)

    val calibration = accumulator.snapshot
    assert(calibration.fixedNanosPerTask.contains(20.0))
    assert(calibration.fixedNanosPerTaskStandardError.isEmpty)
    assert(calibration.fixedTaskCostSampleWindows == 3L)
    assert(calibration.fixedTaskCostReason.isEmpty)
    assert(calibration.fixedTaskCostSource == "executor-deserialize-lower-bound")
  }
}

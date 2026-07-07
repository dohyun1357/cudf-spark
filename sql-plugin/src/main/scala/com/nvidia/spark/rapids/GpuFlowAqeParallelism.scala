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

import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.{CoalescedPartitionSpec, ShufflePartitionSpec, SparkPlan,
  UnaryExecNode, UnionExec}
import org.apache.spark.sql.execution.adaptive.{AQEShuffleReadExec, AQEShuffleReadRule,
  ExchangeQueryStageExec, ShuffleQueryStageExec}
import org.apache.spark.sql.execution.exchange.{ENSURE_REQUIREMENTS, REBALANCE_PARTITIONS_BY_COL,
  REBALANCE_PARTITIONS_BY_NONE, REPARTITION_BY_COL, ShuffleOrigin}
import org.apache.spark.sql.execution.joins.{BroadcastHashJoinExec, BroadcastNestedLoopJoinExec,
  CartesianProductExec}

private[rapids] case class GpuFlowPartitionRange(
    startReducerIndex: Int,
    endReducerIndex: Int,
    bytes: Long)

private[rapids] case class GpuFlowPartitionLayout(
    ranges: Seq[GpuFlowPartitionRange],
    objectiveNanos: Double)

private[rapids] case class GpuFlowParallelismDecision(
    stageIds: Seq[Int],
    originalPartitions: Int,
    currentPartitions: Int,
    selectedPartitions: Int,
    totalBytes: Long,
    taskSlots: Int,
    currentObjectiveNanos: Double,
    selectedObjectiveNanos: Double,
    variableNanosPerByte: Double,
    fixedNanosPerTask: Double,
    reason: String,
    fixedNanosPerTaskStandardError: Double = 0.0,
    fixedTaskCostSampleWindows: Long = 0L,
    fixedTaskCostSource: String = "",
    serialLaunchNanosPerTask: Double = 0.0,
    serialLaunchSampleStages: Long = 0L)

/** Pure, threshold-free optimization of contiguous reducer ranges. */
private[rapids] object GpuFlowPartitionOptimizer {
  def optimize(
      partitionBytes: Seq[Long],
      currentRanges: Seq[(Int, Int)],
      taskSlots: Int,
      variableNanosPerByte: Double,
      fixedNanosPerTask: Double,
      launchNanosPerTask: Double = 0.0,
      maxPartitions: Int = Int.MaxValue,
      maxRangeBytes: Long = Long.MaxValue,
      minSplitRangeBytes: Long = 0L): Option[GpuFlowPartitionLayout] = {
    if (partitionBytes.isEmpty || taskSlots <= 0 || variableNanosPerByte < 0.0 ||
        fixedNanosPerTask < 0.0 || launchNanosPerTask < 0.0 || currentRanges.isEmpty) {
      return None
    }
    val originalPartitions = partitionBytes.size
    val current = layoutForRanges(partitionBytes, currentRanges, taskSlots,
      variableNanosPerByte, fixedNanosPerTask, launchNanosPerTask)
    val waveEndpoints = taskSlots.to(originalPartitions, taskSlots)
    val candidateCounts = (waveEndpoints :+ originalPartitions :+ currentRanges.size)
      .filter(count => count > 0 && count <= originalPartitions && count <= maxPartitions)
      .distinct.sorted
    // The current (deployed) layout is always priced. A candidate layout is eligible only when
    // every range stays within `maxRangeBytes`, the byte region where the calibrated linear rate
    // was actually measured: ranges beyond it have an unidentified, possibly superlinear response.
    // A candidate finer than the current layout (a re-split) must additionally keep every range
    // at or above `minSplitRangeBytes`; below one native GPU batch per task the response leaves
    // the region where task cost is byte service plus the measured fixed and launch costs.
    Some(candidateCounts.foldLeft(current) { (best, count) =>
      val candidate = layoutForRanges(partitionBytes, balancedRanges(partitionBytes, count),
        taskSlots, variableNanosPerByte, fixedNanosPerTask, launchNanosPerTask)
      // balancedRanges may materialize fewer ranges than requested; the re-split floor applies
      // to layouts that are actually finer than the current one.
      val withinSplitFloor = candidate.ranges.size <= currentRanges.size ||
        candidate.ranges.forall(_.bytes >= minSplitRangeBytes)
      if (candidate.ranges.forall(_.bytes <= maxRangeBytes) && withinSplitFloor &&
          candidate.objectiveNanos < best.objectiveNanos) {
        candidate
      } else {
        best
      }
    })
  }

  private[rapids] def balancedRanges(
      partitionBytes: Seq[Long],
      targetCount: Int): Seq[(Int, Int)] = {
    val count = math.max(1, math.min(targetCount, partitionBytes.size))
    if (count == partitionBytes.size) {
      return partitionBytes.indices.map(index => (index, index + 1))
    }
    var low = partitionBytes.max
    var high = saturatingSum(partitionBytes)
    while (low < high) {
      val midpoint = low + (high - low) / 2L
      if (minimumGroupCount(partitionBytes, midpoint) <= count) high = midpoint
      else low = midpoint + 1L
    }

    val ranges = mutable.ArrayBuffer.empty[(Int, Int)]
    var start = 0
    var bytes = 0L
    var index = 0
    while (index < partitionBytes.size) {
      val remainingItems = partitionBytes.size - index
      val remainingGroups = count - ranges.size
      val value = math.max(0L, partitionBytes(index))
      // If closing the current range would leave exactly one item for every remaining range,
      // it must close now. The current open range itself is included in `remainingGroups`.
      val mustStartSingletons = index > start && remainingItems == remainingGroups - 1
      val exceeds = index > start && bytes > low - value
      if (mustStartSingletons || exceeds) {
        ranges += ((start, index))
        start = index
        bytes = 0L
      }
      bytes = saturatingAdd(bytes, value)
      index += 1
    }
    ranges += ((start, partitionBytes.size))
    ranges.toSeq
  }

  private def minimumGroupCount(partitionBytes: Seq[Long], maximumBytes: Long): Int = {
    var groups = 1
    var bytes = 0L
    partitionBytes.foreach { raw =>
      val value = math.max(0L, raw)
      if (bytes > maximumBytes - value) {
        groups += 1
        bytes = value
      } else {
        bytes += value
      }
    }
    groups
  }

  private[rapids] def layoutForRanges(
      partitionBytes: Seq[Long],
      ranges: Seq[(Int, Int)],
      taskSlots: Int,
      variableNanosPerByte: Double,
      fixedNanosPerTask: Double,
      launchNanosPerTask: Double = 0.0): GpuFlowPartitionLayout = {
    val modeled = ranges.map { case (start, end) =>
      GpuFlowPartitionRange(start, end, saturatingSum(partitionBytes.slice(start, end)))
    }
    // Spark schedules the layout in waves of `taskSlots` tasks; each wave lasts as long as its
    // largest task and every task pays the measured fixed cost inside its wave. The former
    // max(total/slots, largest) packing bound priced 200 balanced tasks on 160 slots as 1.25
    // waves of byte service when the measured makespan is two full waves, hiding the dominant
    // cost of wave-misaligned layouts. Serial driver dispatch is paid once per task launch, so
    // it scales with the range count, not the wave count.
    val variableDesc = modeled.map(_.bytes.toDouble * variableNanosPerByte).sortBy(v => -v)
    val waveNanos = variableDesc.grouped(taskSlots).map(_.head).sum
    val waves = (modeled.size + taskSlots - 1) / taskSlots
    GpuFlowPartitionLayout(modeled,
      waveNanos + waves.toDouble * fixedNanosPerTask +
        modeled.size.toDouble * launchNanosPerTask)
  }

  private def saturatingSum(values: Seq[Long]): Long =
    values.foldLeft(0L)((sum, value) => saturatingAdd(sum, math.max(0L, value)))

  private def saturatingAdd(left: Long, right: Long): Long =
    if (right > Long.MaxValue - left) Long.MaxValue else left + right
}

/**
 * AQE query-stage optimizer rule that replaces Spark's coalesced reducer layout with the layout
 * minimizing the calibrated flow makespan. It runs after Spark's coalescing/local-read rules and
 * only rewrites plain coalesced reads; skew and locality partition specs retain Spark ownership.
 */
case class GpuFlowAqeParallelismRule()
    extends Rule[SparkPlan] with AQEShuffleReadRule {
  override val supportedShuffleOrigins: Seq[ShuffleOrigin] = Seq(
    ENSURE_REQUIREMENTS,
    REPARTITION_BY_COL,
    REBALANCE_PARTITIONS_BY_NONE,
    REBALANCE_PARTITIONS_BY_COL)

  private case class StageInfo(
      stage: ShuffleQueryStageExec,
      specs: Seq[CoalescedPartitionSpec])

  override def apply(plan: SparkPlan): SparkPlan = {
    val rapidsConf = new RapidsConf(plan.conf)
    if (!rapidsConf.autotuneGraphEnabled || !rapidsConf.isAutotuneOptimizeMode) {
      return plan
    }
    val calibration = RapidsAutotuneDriverEndpoint.aqeCalibrationSnapshot
    val nativeBatchBytes = rapidsConf.gpuTargetBatchSizeBytes
    val replacements = mutable.HashMap.empty[Int, Seq[ShufflePartitionSpec]]
    collectGroups(plan).foreach { group =>
      val decision = optimizeGroup(group, calibration, nativeBatchBytes)
      RapidsAutotuneDriverEndpoint.recordParallelismDecision(decision._1)
      decision._2.foreach(replacements ++= _)
    }
    if (replacements.isEmpty) plan else {
      plan.transformUp {
        case AQEShuffleReadExec(stage: ShuffleQueryStageExec, _)
            if replacements.contains(stage.id) =>
          AQEShuffleReadExec(stage, replacements(stage.id))
      }
    }
  }

  private def optimizeGroup(
      group: Seq[StageInfo],
      calibration: Option[GpuFlowAqeCalibration],
      nativeBatchBytes: Long)
      : (GpuFlowParallelismDecision, Option[Map[Int, Seq[ShufflePartitionSpec]]]) = {
    val stageIds = group.map(_.stage.id)
    val originalPartitions = group.head.stage.mapStats.map(_.bytesByPartitionId.length).getOrElse(0)
    val currentPartitions = group.head.specs.size
    val emptyDecision = GpuFlowParallelismDecision(stageIds, originalPartitions,
      currentPartitions, currentPartitions, 0L, 0, 0.0, 0.0, 0.0, 0.0, "")
    if (originalPartitions <= 0 || group.exists { info =>
      info.stage.mapStats.forall(_.bytesByPartitionId.length != originalPartitions)
    }) {
      return (emptyDecision.copy(reason = "missing-or-incompatible-map-statistics"), None)
    }
    if (group.map(_.specs.map(spec =>
        (spec.startReducerIndex, spec.endReducerIndex))).distinct.size != 1) {
      return (emptyDecision.copy(reason = "incompatible-current-partition-boundaries"), None)
    }
    calibration match {
      case None => (emptyDecision.copy(reason = "no-runtime-calibration"), None)
      case Some(calibrated) =>
        // Wave arithmetic needs the cluster-wide CPU task-slot width the Spark scheduler admits
        // tasks against, measured from executor registration. The per-executor GPU admission
        // quota is a different resource and undercounts a multi-executor cluster.
        val taskSlots = calibrated.clusterTaskSlots
        if (taskSlots <= 0) {
          return (emptyDecision.copy(reason = "cluster-task-slots-unmeasured"), None)
        }
        val variableRate = divideRate(calibrated.shuffleUnitNanosPerByte,
          calibrated.referenceControl.shuffleWindow * calibrated.referenceControl.shuffleBytes)
        if (variableRate.isEmpty) {
          return (emptyDecision.copy(taskSlots = taskSlots,
            reason = "missing-reducer-service-rate"), None)
        }
        val combined = Array.fill[Long](originalPartitions)(0L)
        group.foreach { info =>
          info.stage.mapStats.get.bytesByPartitionId.zipWithIndex.foreach {
            case (bytes, index) =>
              combined(index) = if (bytes > Long.MaxValue - combined(index)) {
                Long.MaxValue
              } else combined(index) + math.max(0L, bytes)
          }
        }
        val totalBytes = combined.foldLeft(0L) { (sum, bytes) =>
          if (bytes > Long.MaxValue - sum) Long.MaxValue else sum + bytes
        }
        val rate = variableRate.get
        if (calibrated.fixedNanosPerTask.isEmpty) {
          return (emptyDecision.copy(taskSlots = taskSlots,
            totalBytes = totalBytes,
            variableNanosPerByte = rate,
            fixedNanosPerTaskStandardError =
              calibrated.fixedNanosPerTaskStandardError.getOrElse(0.0),
            fixedTaskCostSampleWindows = calibrated.fixedTaskCostSampleWindows,
            fixedTaskCostSource = calibrated.fixedTaskCostSource,
            serialLaunchNanosPerTask = calibrated.serialLaunchNanosPerTask.getOrElse(0.0),
            serialLaunchSampleStages = calibrated.serialLaunchSampleStages,
            reason = calibrated.fixedTaskCostReason), None)
        }
        val currentRanges = group.head.specs.map(spec =>
          (spec.startReducerIndex, spec.endReducerIndex))
        val fixed = calibrated.fixedNanosPerTask.get
        val serialLaunch = calibrated.serialLaunchNanosPerTask
        val launchNanos = serialLaunch.getOrElse(0.0)
        // The calibrated byte rate is linear evidence only inside the region it was measured in:
        // up to the largest per-task shuffle-read load observed in this execution, floored by the
        // native GPU batch target (below one batch, operators stay in their single-batch regime).
        // Ranges beyond that region cross unmeasured operator thresholds where per-task cost can
        // turn superlinear (an SF3K q21 join stage doubled per-task time for +25% bytes).
        val identifiedRangeBytes = math.max(
          nativeBatchBytes, calibrated.maxCalibratedTaskShuffleReadBytes)
        // Spark's current AQE layout is the identified baseline. Coalescing below it has exact
        // map statistics plus measured task-wave cost. A re-split above it (never above the
        // map-side partition count) additionally spends serial driver dispatch work, so it is
        // eligible only once launch response has been measured, and every re-split range must
        // hold at least one native GPU batch so per-task cost stays in the identified byte
        // region. Pricing smaller-than-measured tasks at the calibrated rate is conservative
        // for re-splits: per-byte service does not improve with task size below the measured
        // point once fixed and launch costs are charged separately, so the modeled wave-
        // alignment benefit understates the measured one (SF3K 160/200/320/1280 evidence).
        val splitCap = if (serialLaunch.isDefined) originalPartitions else currentRanges.size
        val unconstrained = GpuFlowPartitionOptimizer.optimize(
          combined.toSeq, currentRanges, taskSlots, rate, fixed, launchNanos).get
        val uncapped = GpuFlowPartitionOptimizer.optimize(
          combined.toSeq, currentRanges, taskSlots, rate, fixed, launchNanos,
          maxPartitions = splitCap,
          minSplitRangeBytes = nativeBatchBytes).get
        val unfloored = GpuFlowPartitionOptimizer.optimize(
          combined.toSeq, currentRanges, taskSlots, rate, fixed, launchNanos,
          maxPartitions = splitCap,
          maxRangeBytes = identifiedRangeBytes).get
        val selected = GpuFlowPartitionOptimizer.optimize(
          combined.toSeq, currentRanges, taskSlots, rate, fixed, launchNanos,
          maxPartitions = splitCap,
          maxRangeBytes = identifiedRangeBytes,
          minSplitRangeBytes = nativeBatchBytes).get
        val currentObjective = GpuFlowPartitionOptimizer.layoutForRanges(
          combined.toSeq, currentRanges, taskSlots, rate, fixed, launchNanos).objectiveNanos
        val changed = selected.ranges.map(range =>
          (range.startReducerIndex, range.endReducerIndex)) != currentRanges
        val decision = emptyDecision.copy(
          selectedPartitions = selected.ranges.size,
          totalBytes = totalBytes,
          taskSlots = taskSlots,
          currentObjectiveNanos = currentObjective,
          selectedObjectiveNanos = selected.objectiveNanos,
          variableNanosPerByte = rate,
          fixedNanosPerTask = fixed,
          reason = if (changed) {
            ""
          } else if (serialLaunch.isEmpty &&
              unconstrained.ranges.size > currentRanges.size) {
            "higher-parallelism-response-unidentified"
          } else if (unfloored.objectiveNanos < selected.objectiveNanos) {
            "sub-batch-range-response-unidentified"
          } else if (uncapped.objectiveNanos < selected.objectiveNanos) {
            "larger-range-response-unidentified"
          } else {
            "current-layout-optimal"
          },
          fixedNanosPerTaskStandardError =
            calibrated.fixedNanosPerTaskStandardError.getOrElse(0.0),
          fixedTaskCostSampleWindows = calibrated.fixedTaskCostSampleWindows,
          fixedTaskCostSource = calibrated.fixedTaskCostSource,
          serialLaunchNanosPerTask = launchNanos,
          serialLaunchSampleStages = calibrated.serialLaunchSampleStages)
        val replacement = if (!changed || selected.objectiveNanos >= currentObjective) None else {
          Some(group.map { info =>
            val stageBytes = info.stage.mapStats.get.bytesByPartitionId
            val specs: Seq[ShufflePartitionSpec] = selected.ranges.map { range =>
              val bytes = stageBytes.slice(
                range.startReducerIndex, range.endReducerIndex).foldLeft(0L) { (sum, raw) =>
                val value = math.max(0L, raw)
                if (value > Long.MaxValue - sum) Long.MaxValue else sum + value
              }
              CoalescedPartitionSpec(
                range.startReducerIndex, range.endReducerIndex, math.max(0L, bytes))
            }
            info.stage.id -> specs
          }.toMap)
        }
        (decision, replacement)
    }
  }

  private def collectGroups(plan: SparkPlan): Seq[Seq[StageInfo]] = plan match {
    case read: AQEShuffleReadExec =>
      stageInfo(read).map(info => Seq(Seq(info))).getOrElse(Seq.empty)
    case unary: UnaryExecNode => collectGroups(unary.child)
    case node if !childrenNeedCompatiblePartitioning(node) =>
      node.children.flatMap(collectGroups)
    case node if node.collectLeaves().forall(_.isInstanceOf[ExchangeQueryStageExec]) =>
      val infos = node.collect { case read: AQEShuffleReadExec => stageInfo(read) }.flatten
      if (infos.nonEmpty) Seq(infos) else Seq.empty
    case _ => Seq.empty
  }

  private def stageInfo(read: AQEShuffleReadExec): Option[StageInfo] = read match {
    case AQEShuffleReadExec(stage: ShuffleQueryStageExec, specs)
        if !read.isLocalRead && !read.hasSkewedPartition && isSupported(stage.shuffle) &&
          specs.nonEmpty && specs.forall(_.isInstanceOf[CoalescedPartitionSpec]) =>
      Some(StageInfo(stage, specs.map(_.asInstanceOf[CoalescedPartitionSpec])))
    case _ => None
  }

  private def childrenNeedCompatiblePartitioning(plan: SparkPlan): Boolean = plan match {
    case _: UnionExec => false
    case _: CartesianProductExec => false
    case _: BroadcastHashJoinExec => false
    case _: BroadcastNestedLoopJoinExec => false
    case _ => true
  }

  private def divideRate(rate: Option[Double], capacity: Double): Option[Double] =
    rate.filter(_ >= 0.0).map(value => if (capacity > 0.0) value / capacity else value)
}

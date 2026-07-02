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
import scala.util.control.NonFatal

import org.apache.spark.SparkConf
import org.apache.spark.internal.Logging
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.adaptive.{Cost, CostEvaluator, QueryStageExec, SimpleCost,
  SimpleCostEvaluator}
import org.apache.spark.sql.execution.exchange.{BroadcastExchangeLike, ShuffleExchangeLike}

private[rapids] case class GpuFlowAqeDemand(
    scanBytes: Double = 0.0,
    gpuBytes: Double = 0.0,
    shuffleBytes: Double = 0.0,
    broadcastBytes: Double = 0.0,
    batchBytes: Double = 0.0)

private[rapids] case class GpuFlowAqeNode(
    key: AutotuneStageKey,
    parents: Seq[AutotuneStageKey],
    demand: GpuFlowAqeDemand)

private[rapids] case class GpuFlowAqePlanEvaluation(
    objectiveNanos: Double,
    identifiable: Boolean,
    reason: String,
    operatorFingerprint: String,
    topologyFingerprint: String,
    scanBytes: Double,
    gpuBytes: Double,
    shuffleBytes: Double,
    broadcastBytes: Double,
    batchBytes: Double,
    selectedControl: GpuFlowControl,
    sampleWindows: Long,
    sparkFallbackCost: Long = 0L)

/**
 * A Spark AQE cost with conservative comparison semantics.
 *
 * Spark compares the current and proposed complete physical plans after evaluating both. If
 * either plan is not identifiable from measured resource rates, comparison returns an exact tie
 * and Spark retains the current plan. Spark's AQE rule supplies semantics-preserving current and
 * replacement candidates; physical operator differences are therefore costs to model, not a
 * reason to reject the comparison.
 */
private[rapids] case class GpuFlowAqeCost(
    evaluation: GpuFlowAqePlanEvaluation,
    sparkFallback: Cost) extends Cost {
  override def compare(other: Cost): Int = other match {
    // A generic compute-byte rate does not identify the relative response of different physical
    // operators (for example SortMergeJoin versus BroadcastHashJoin). Until operator-specific
    // runtime calibration exists, preserve Spark's native AQE decision for that discrete basis.
    case that: GpuFlowAqeCost if usesSparkFallback(that) =>
      sparkFallback.compare(that.sparkFallback)
    case that: GpuFlowAqeCost if evaluation.identifiable && that.evaluation.identifiable =>
      java.lang.Double.compare(evaluation.objectiveNanos, that.evaluation.objectiveNanos)
    case _: GpuFlowAqeCost => 0
    case _ => 0
  }

  /**
   * Spark accepts a distinct proposed plan on an equal cost only when the Cost values are equal.
   * Keep equality consistent with the selected comparison basis, while leaving an unidentified
   * tie between structurally distinct candidates unequal so Spark freezes the current plan.
   * Identical-structure pairs always take the Spark fallback basis, so a statistics refresh is
   * adopted under Spark's native contract even when rates are unidentified.
   */
  override def equals(other: Any): Boolean = other match {
    case that: GpuFlowAqeCost if usesSparkFallback(that) =>
      sparkFallback == that.sparkFallback
    case that: GpuFlowAqeCost => this eq that
    case _ => false
  }

  // Equality is comparison-context dependent, so no stable field subset can provide a narrower
  // hash while preserving the equals contract.
  override def hashCode(): Int = 0

  private def usesSparkFallback(that: GpuFlowAqeCost): Boolean = {
    val differentOperatorBasis =
      evaluation.operatorFingerprint != that.evaluation.operatorFingerprint
    // Identical operator and topology fingerprints mean the candidates share one physical
    // structure and differ only in the runtime statistics Spark refreshed between plannings.
    // There is no physical counterfactual for the flow model to price; ranking the pair by the
    // statistics delta vetoed every AQE re-optimization and pinned stale plan statistics (every
    // strict flow decision in the SF3000 stream was such a pair). Spark's native contract owns
    // the choice and adopts the refreshed plan.
    val identicalStructure = !differentOperatorBasis &&
      evaluation.topologyFingerprint == that.evaluation.topologyFingerprint
    val equalIdentifiedObjective = evaluation.identifiable && that.evaluation.identifiable &&
      java.lang.Double.compare(
        evaluation.objectiveNanos, that.evaluation.objectiveNanos) == 0
    // Distinct candidates collapsing to the same objective expose an unmodeled physical
    // difference. Preserve Spark's native ordering until measured features distinguish them.
    differentOperatorBasis || identicalStructure || equalIdentifiedObjective
  }
}

/** Builds and analytically optimizes the remaining max-plus flow graph of one AQE candidate. */
private[rapids] object GpuFlowAqePlanModel {
  private case class Built(
      boundaryBytes: Seq[Double],
      roots: Seq[AutotuneStageKey],
      semantic: Seq[String],
      topology: Seq[String])

  private val WrapperTokens = Seq(
    "AdaptiveSparkPlan",
    "AQEShuffleRead",
    "CustomShuffleReader",
    "InputAdapter",
    "WholeStageCodegen",
    "ColumnarToRow",
    "RowToColumnar",
    "ReusedExchange")

  private val ComputeCategories = Seq(
    "Join" -> "join",
    "Aggregate" -> "aggregate",
    "Sort" -> "sort",
    "Window" -> "window",
    "Filter" -> "filter",
    "Project" -> "project",
    "Generate" -> "generate",
    "Expand" -> "expand",
    "Limit" -> "limit",
    "TakeOrdered" -> "limit")

  def evaluate(
      plan: SparkPlan,
      calibration: Option[GpuFlowAqeCalibration]): GpuFlowAqePlanEvaluation = {
    calibration match {
      case None => unknown("no-runtime-calibration", "", plan.nodeName, 0L)
      case Some(calibrated) =>
        try {
          val nodes = mutable.ArrayBuffer.empty[GpuFlowAqeNode]
          val unknownReasons = mutable.ArrayBuffer.empty[String]
          var nextNodeId = 0

          def addNode(
              parents: Seq[AutotuneStageKey],
              demand: GpuFlowAqeDemand): AutotuneStageKey = {
            val key = AutotuneStageKey(0L, nextNodeId, 0)
            nextNodeId += 1
            nodes += GpuFlowAqeNode(key, parents.distinct, demand)
            key
          }

          def build(node: SparkPlan): Built = node match {
            case stage: QueryStageExec =>
              val bytes = statBytes(stage.getRuntimeStatistics)
              if (bytes.isEmpty) {
                unknownReasons += s"missing-size:${stage.nodeName}"
              }
              Built(bytes.toSeq, Seq.empty, Seq.empty,
                Seq(s"materialized:${stage.nodeName}"))
            case _ =>
              val children = node.children.map(build)
              val childRoots = children.flatMap(_.roots)
              val childSemantic = children.flatMap(_.semantic)
              val childTopology = children.flatMap(_.topology)
              val childBoundary = children.flatMap(_.boundaryBytes)
              val childBytes = if (childBoundary.nonEmpty) Some(childBoundary.sum) else None
              // Intermediate physical operators often have no logicalLink and therefore expose
              // no output statistics. Keep their nearest measured input boundaries instead of
              // inventing selectivity. Semantically equivalent AQE candidates consequently use
              // the same byte basis for their compute operators. An unmaterialized exchange also
              // preserves that measured input boundary: its `runtimeStatistics` still contain
              // logical estimates, sometimes orders of magnitude above observed query-stage
              // bytes. QueryStageExec above is the only source of measured AQE runtime size.

              node match {
                case _: ShuffleExchangeLike =>
                  childBytes match {
                    case Some(bytes) =>
                      val key = addNode(childRoots, GpuFlowAqeDemand(shuffleBytes = bytes))
                      Built(Seq(bytes), Seq(key), childSemantic,
                        childTopology :+ "shuffle-exchange")
                    case None =>
                      unknownReasons += s"missing-size:${node.nodeName}"
                      Built(Seq.empty, childRoots, childSemantic,
                        childTopology :+ "shuffle-exchange")
                  }
                case _: BroadcastExchangeLike =>
                  childBytes match {
                    case Some(bytes) =>
                      val key = addNode(childRoots, GpuFlowAqeDemand(broadcastBytes = bytes))
                      Built(Seq(bytes), Seq(key), childSemantic,
                        childTopology :+ "broadcast-exchange")
                    case None =>
                      unknownReasons += s"missing-size:${node.nodeName}"
                      Built(Seq.empty, childRoots, childSemantic,
                        childTopology :+ "broadcast-exchange")
                  }
                case _ if isWrapper(node.nodeName) =>
                  Built(childBoundary, childRoots, childSemantic,
                    childTopology :+ s"wrapper:${normalize(node.nodeName)}")
                case _ if isScan(node.nodeName) =>
                  planBytes(node) match {
                    case Some(bytes) =>
                      val key = addNode(childRoots, GpuFlowAqeDemand(scanBytes = bytes))
                      Built(Seq(bytes), Seq(key), childSemantic :+ "scan",
                        childTopology :+ s"scan:${normalize(node.nodeName)}")
                    case None =>
                      unknownReasons += s"missing-size:${node.nodeName}"
                      Built(Seq.empty, childRoots, childSemantic :+ "scan", childTopology)
                  }
                case _ =>
                  computeCategory(node.nodeName) match {
                    case Some(category) =>
                      childBytes match {
                        case Some(processed) =>
                          val key = addNode(childRoots,
                            GpuFlowAqeDemand(
                              gpuBytes = processed,
                              // Batch response is optional and independently identifiable. An
                              // absent rate freezes that control at its reference value without
                              // preventing the measured GPU/exchange topology comparison.
                              batchBytes = if (calibrated.batchUnitNanosPerByte.isDefined) {
                                processed
                              } else 0.0))
                          Built(childBoundary, Seq(key), childSemantic :+ category,
                            childTopology :+ s"$category:${normalize(node.nodeName)}")
                        case None =>
                          unknownReasons += s"missing-input-size:${node.nodeName}"
                          Built(Seq.empty, childRoots, childSemantic :+ category, childTopology)
                      }
                    case None if node.children.size == 1 =>
                      unknownReasons += s"unsupported-operator:${node.nodeName}"
                      Built(childBoundary, childRoots,
                        childSemantic :+ s"unsupported:${normalize(node.nodeName)}", childTopology)
                    case None if node.children.isEmpty =>
                      unknownReasons += s"unsupported-leaf:${node.nodeName}"
                      Built(Seq.empty, childRoots,
                        childSemantic :+ s"unsupported:${normalize(node.nodeName)}", childTopology)
                    case None =>
                      unknownReasons += s"unsupported-operator:${node.nodeName}"
                      Built(childBoundary, childRoots,
                        childSemantic :+ s"unsupported:${normalize(node.nodeName)}", childTopology)
                  }
              }
          }

          val built = build(plan)
          val operatorFingerprint = built.semantic.sorted.mkString("|")
          val topologyFingerprint = built.topology.mkString("|")
          if (unknownReasons.nonEmpty) {
            unknown(unknownReasons.distinct.sorted.mkString(","), operatorFingerprint,
              topologyFingerprint, calibrated.sampleWindows)
          } else {
            evaluateNodes(nodes.toSeq, calibrated, operatorFingerprint, topologyFingerprint)
          }
        } catch {
          case NonFatal(e) =>
            unknown(s"model-error:${e.getClass.getSimpleName}", "", plan.nodeName,
              calibrated.sampleWindows)
        }
    }
  }

  private[rapids] def evaluateNodes(
      nodes: Seq[GpuFlowAqeNode],
      calibration: GpuFlowAqeCalibration,
      operatorFingerprint: String,
      topologyFingerprint: String): GpuFlowAqePlanEvaluation = {
    val total = nodes.foldLeft(GpuFlowAqeDemand()) { (sum, node) =>
      GpuFlowAqeDemand(
        scanBytes = sum.scanBytes + node.demand.scanBytes,
        gpuBytes = sum.gpuBytes + node.demand.gpuBytes,
        shuffleBytes = sum.shuffleBytes + node.demand.shuffleBytes,
        broadcastBytes = sum.broadcastBytes + node.demand.broadcastBytes,
        batchBytes = sum.batchBytes + node.demand.batchBytes)
    }
    val missing = Seq(
      requiredRate(total.scanBytes, calibration.scanUnitNanosPerByte, "scan"),
      requiredRate(total.gpuBytes, calibration.gpuUnitNanosPerByte, "gpu"),
      requiredRate(total.shuffleBytes, calibration.shuffleUnitNanosPerByte, "shuffle"),
      requiredRate(total.broadcastBytes, calibration.broadcastNanosPerByte, "broadcast"),
      requiredRate(total.batchBytes, calibration.batchUnitNanosPerByte, "batch"))
      .flatten
    if (missing.nonEmpty) {
      return unknown(missing.mkString(","), operatorFingerprint, topologyFingerprint,
        calibration.sampleWindows, total)
    }

    val evaluatedNodes = nodes.map { node =>
      val work = workFor(node.demand, calibration)
      val selected = GpuFlowProjectedOptimizer.optimize(
        work, calibration.referenceControl, calibration.bounds)
      val evaluation = GpuFlowStageModel.evaluate(work, selected)
      GpuFlowGraphNode(node.key, node.parents, evaluation)
    }
    val flow = GpuFlowModel.evaluate(evaluatedNodes)
    val selectedControl = if (nodes.isEmpty) calibration.referenceControl else {
      // Controls are stage-local. The event carries their demand-weighted mean as a compact
      // diagnostic; the objective above is evaluated with each node's own selected control.
      weightedSelectedControl(nodes, calibration)
    }
    GpuFlowAqePlanEvaluation(
      objectiveNanos = flow.objectiveNanos,
      identifiable = true,
      reason = "",
      operatorFingerprint = operatorFingerprint,
      topologyFingerprint = topologyFingerprint,
      scanBytes = total.scanBytes,
      gpuBytes = total.gpuBytes,
      shuffleBytes = total.shuffleBytes,
      broadcastBytes = total.broadcastBytes,
      batchBytes = total.batchBytes,
      selectedControl = selectedControl,
      sampleWindows = calibration.sampleWindows)
  }

  private def workFor(
      demand: GpuFlowAqeDemand,
      calibration: GpuFlowAqeCalibration): GpuFlowStageWork = {
    val reference = calibration.referenceControl
    GpuFlowStageWork(
      scanNanos = normalizedWork(
        calibration.scanUnitNanosPerByte, demand.scanBytes, reference.scanWindow),
      gpuNanos = normalizedWork(
        calibration.gpuUnitNanosPerByte, demand.gpuBytes, reference.gpuTasks),
      shuffleNanos = normalizedWork(
        calibration.shuffleUnitNanosPerByte,
        demand.shuffleBytes,
        reference.shuffleWindow * reference.shuffleBytes),
      batchNanos = normalizedWork(
        calibration.batchUnitNanosPerByte, demand.batchBytes, reference.batchBytes),
      fixedNanos = calibration.broadcastNanosPerByte.getOrElse(0.0) * demand.broadcastBytes,
      retryNanos = 0.0,
      baseline = reference)
  }

  private def weightedSelectedControl(
      nodes: Seq[GpuFlowAqeNode],
      calibration: GpuFlowAqeCalibration): GpuFlowControl = {
    val weighted = nodes.map { node =>
      val weight = Seq(node.demand.scanBytes, node.demand.gpuBytes, node.demand.shuffleBytes,
        node.demand.broadcastBytes, node.demand.batchBytes).sum
      val work = workFor(node.demand, calibration)
      val selected = GpuFlowProjectedOptimizer.optimize(
        work, calibration.referenceControl, calibration.bounds)
      (math.max(1.0, weight), selected)
    }
    val denominator = weighted.map(_._1).sum
    GpuFlowControl(
      scanWindow = weighted.map { case (weight, control) => weight * control.scanWindow }.sum /
        denominator,
      gpuTasks = weighted.map { case (weight, control) => weight * control.gpuTasks }.sum /
        denominator,
      shuffleWindow = weighted.map { case (weight, control) =>
        weight * control.shuffleWindow
      }.sum / denominator,
      shuffleBytes = weighted.map { case (weight, control) =>
        weight * control.shuffleBytes
      }.sum / denominator,
      batchBytes = weighted.map { case (weight, control) => weight * control.batchBytes }.sum /
        denominator)
  }

  private def normalizedWork(
      rate: Option[Double],
      bytes: Double,
      referenceCapacity: Double): Double = {
    if (bytes <= 0.0) 0.0 else rate.getOrElse(0.0) * bytes / referenceCapacity
  }

  private def requiredRate(
      bytes: Double,
      rate: Option[Double],
      lane: String): Option[String] = {
    if (bytes > 0.0 && rate.isEmpty) Some(s"missing-$lane-rate") else None
  }

  private def planBytes(plan: SparkPlan): Option[Double] = {
    plan.logicalLink.flatMap(link => statBytes(link.stats))
  }

  private def statBytes(stats: org.apache.spark.sql.catalyst.plans.logical.Statistics)
      : Option[Double] = {
    val bytes = stats.sizeInBytes
    if (bytes >= 0 && bytes.isValidLong && bytes.longValue != Long.MaxValue) {
      Some(bytes.doubleValue)
    } else {
      None
    }
  }

  private def isWrapper(nodeName: String): Boolean = WrapperTokens.exists(nodeName.contains)

  private def isScan(nodeName: String): Boolean = nodeName.contains("Scan")

  private def computeCategory(nodeName: String): Option[String] =
    ComputeCategories.collectFirst {
      case (token, category) if nodeName.contains(token) => category
    }

  private def normalize(nodeName: String): String =
    nodeName.replace("Gpu", "").replace("Exec", "").replaceAll("\\s+", "")

  private def unknown(
      reason: String,
      operatorFingerprint: String,
      topologyFingerprint: String,
      sampleWindows: Long,
      demand: GpuFlowAqeDemand = GpuFlowAqeDemand()): GpuFlowAqePlanEvaluation =
    GpuFlowAqePlanEvaluation(
      objectiveNanos = 0.0,
      identifiable = false,
      reason = reason,
      operatorFingerprint = operatorFingerprint,
      topologyFingerprint = topologyFingerprint,
      scanBytes = demand.scanBytes,
      gpuBytes = demand.gpuBytes,
      shuffleBytes = demand.shuffleBytes,
      broadcastBytes = demand.broadcastBytes,
      batchBytes = demand.batchBytes,
      selectedControl = GpuFlowControl(0.0, 0.0, 0.0, 0.0, 0.0),
      sampleWindows = sampleWindows)
}

/** Spark's pre-materialization AQE authority backed by the shared analytical flow model. */
class GpuFlowAqeCostEvaluator(conf: SparkConf) extends CostEvaluator with Logging {
  def this() = this(new SparkConf(false))

  private val sparkFallback = new SimpleCostEvaluator(
    conf.getBoolean("spark.sql.adaptive.forceOptimizeSkewedJoin", false))

  override def evaluateCost(plan: SparkPlan): Cost = {
    val nativeCost = sparkFallback.evaluateCost(plan)
    val nativeValue = nativeCost match {
      case cost: SimpleCost => cost.value
      case _ => 0L
    }
    val evaluation = GpuFlowAqePlanModel.evaluate(
      plan, RapidsAutotuneDriverEndpoint.aqeCalibrationSnapshot)
      .copy(sparkFallbackCost = nativeValue)
    RapidsAutotuneDriverEndpoint.recordAqeCostEvaluation(evaluation)
    GpuFlowAqeCost(evaluation, nativeCost)
  }
}

private[rapids] object GpuFlowAqeCostEvaluator {
  val SparkCostEvaluatorKey = "spark.sql.adaptive.customCostEvaluatorClass"
  val ClassName: String = classOf[GpuFlowAqeCostEvaluator].getName

  /**
   * Install AQE authority only in OPTIMIZE mode. A user-provided evaluator is never overwritten;
   * its class name is returned so the caller can report why graph topology remains external.
   */
  def configureIfNeeded(conf: SparkConf): Option[String] = {
    val rapidsConf = new RapidsConf(conf)
    if (!rapidsConf.autotuneGraphEnabled || !rapidsConf.isAutotuneOptimizeMode) {
      None
    } else {
      conf.getOption(SparkCostEvaluatorKey) match {
        case None =>
          conf.set(SparkCostEvaluatorKey, ClassName)
          None
        case Some(ClassName) => None
        case conflict => conflict
      }
    }
  }
}

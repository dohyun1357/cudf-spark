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

/**
 * One complete, semantics-preserving AQE topology supplied by Spark-side candidate construction.
 * Its nodes contain only remaining work; already materialized query stages are sunk cost.
 * Continuous controls must be optimized for this fixed topology before it is submitted here.
 */
private[rapids] case class GraphAqeCandidate(
    id: String,
    nodes: Seq[GpuFlowGraphNode],
    current: Boolean = false)

private[rapids] case class GraphAqeCandidateScore(
    candidate: GraphAqeCandidate,
    flow: GpuFlowGraphEvaluation)

/**
 * Discrete outer optimizer for AQE alternatives.
 *
 * The evaluator does not invent partition sizes, join thresholds, or rewrite rules. Spark-side
 * code is responsible for enumerating legal plans. Every complete candidate is scored by the same
 * end-to-end flow objective used for runtime controls. The current plan wins an exact tie so a
 * topology rewrite requires a strictly better modeled query completion time.
 */
private[rapids] object GraphAqeCandidateEvaluator {
  def evaluate(candidates: Seq[GraphAqeCandidate]): Seq[GraphAqeCandidateScore] = {
    require(candidates.nonEmpty, "at least one AQE candidate is required")
    require(candidates.map(_.id).distinct.size == candidates.size,
      "AQE candidate ids must be unique")
    candidates.map(candidate => GraphAqeCandidateScore(
      candidate, GpuFlowModel.evaluate(candidate.nodes)))
  }

  def select(candidates: Seq[GraphAqeCandidate]): GraphAqeCandidateScore = {
    evaluate(candidates).reduceLeft { (left, right) =>
      if (right.flow.objectiveNanos < left.flow.objectiveNanos ||
          (right.flow.objectiveNanos == left.flow.objectiveNanos &&
            right.candidate.current && !left.candidate.current)) {
        right
      } else {
        left
      }
    }
  }
}

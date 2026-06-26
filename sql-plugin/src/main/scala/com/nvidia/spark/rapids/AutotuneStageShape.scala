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

import org.apache.spark.scheduler.StageInfo
import org.apache.spark.storage.RDDInfo

/**
 * Driver-side classification of a Spark stage's resource shape, used to decide which stages should
 * receive scan-prefetch hints. Derived from the operation scopes of the stage's RDDs.
 */
case class AutotuneStageShape(
    hasGpuScan: Boolean,
    hasGpuPrefetchConsumer: Boolean,
    numTasks: Int) {
  def isScanPrefetchCandidate: Boolean =
    hasGpuScan && hasGpuPrefetchConsumer && numTasks > 0
}

object AutotuneStageShape {
  private val GpuScanPrefix = "GpuScan"

  // "GPU prefetch consumer" = a GPU op downstream of a scan whose presence makes eager scan
  // prefetch worthwhile. This is the driver-side, RDD-scope-name view of that concept; the
  // columnar-plan-rule view is BucketJoinTwoSidesPrefetch.isGpuPrefetchConsumer (class matching).
  // They are intentionally the same concept in two forms and should be kept in sync. Known
  // divergence: this scope-prefix view matches shuffled joins via the broad "GpuShuffled" prefix
  // but does not reliably catch broadcast hash joins, which the class-matching view catches via the
  // GpuHashJoin trait.
  private val GpuConsumerPrefixes = Seq(
    "GpuHashAggregate",
    "GpuSort",
    "GpuShuffled",
    "GpuHashJoin")

  def fromStageInfo(stageInfo: StageInfo): AutotuneStageShape = {
    fromRddScopeNames(stageInfo.rddInfos.flatMap(scopeName), stageInfo.numTasks)
  }

  def fromRddScopeNames(scopeNames: Seq[String], numTasks: Int): AutotuneStageShape = {
    AutotuneStageShape(
      hasGpuScan = scopeNames.exists(_.startsWith(GpuScanPrefix)),
      hasGpuPrefetchConsumer = scopeNames.exists { name =>
        GpuConsumerPrefixes.exists(name.startsWith)
      },
      numTasks = numTasks)
  }

  private def scopeName(info: RDDInfo): Option[String] =
    info.scope.map(_.name)
}

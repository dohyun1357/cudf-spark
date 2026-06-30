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

private case class ActiveGpuAdmissionHint(
    var maxConcurrentTasks: Int,
    var sharedMaxConcurrentTasks: Int,
    var schedulingPriority: Long,
    var hintVersion: Long,
    activeTasks: mutable.Set[Long])

/** Stable semaphore admission-group id available from both the wire key and Spark TaskContext. */
private[rapids] object GpuAdmissionStageGroup {
  def apply(key: AutotuneStageKey): Long = apply(key.stageId, key.stageAttemptId)

  def apply(stageId: Int, stageAttemptId: Int): Long =
    (stageId.toLong << 32) | (stageAttemptId.toLong & 0xffffffffL)
}

/**
 * Executor-local GPU admission controller driven by graph autotune hints.
 *
 * It tracks the graph optimizer's per-stage quotas, shared executor-wide cap, and critical-path
 * priorities, then installs the complete active allocation atomically in `GpuSemaphore`. The
 * semaphore's memory-permit pool remains the hard safety layer in every mode.
 *
 * The cap tracks the highest-version optimizer hint seen for each key, in both GRAPH and OPTIMIZE.
 * GRAPH is still bounded by its static envelope; OPTIMIZE may use its explicit larger envelope.
 * Versioning (not arrival order) decides which complete optimizer decision wins, so out-of-order
 * task starts cannot regress the cap to stale content. The permit pool gates every admission.
 *
 * Active membership is tracked by `taskAttemptId`, not a bare counter, so a completion only ever
 * releases a task that actually registered. This matters once hints change mid-stage (the Slice-2
 * fetch TTL): a task that started under no hint (cap 0, never registered) must NOT release an entry
 * created by a sibling task that started under a positive cap.
 */
private[rapids] class RapidsAutotuneGpuAdmissionController(
    applyLimit: Int => Int,
    applyStageAllocation: Option[(Int, Map[Long, Int], Map[Long, Long]) => Int] = None,
    initialMaxSharedConcurrentTasks: Int = 0) {
  private val activeHints = mutable.HashMap.empty[AutotuneStageKey, ActiveGpuAdmissionHint]
  private var maxSharedConcurrentTasks = math.max(0, initialMaxSharedConcurrentTasks)
  private var appliedLimit = 0

  def configureMaxSharedConcurrentTasks(maxTasks: Int): Unit = synchronized {
    maxSharedConcurrentTasks = math.max(0, maxTasks)
    if (activeHints.nonEmpty) {
      applyCurrentAllocation()
    }
  }

  def taskStarted(
      key: AutotuneStageKey,
      taskAttemptId: Long,
      cachedHint: AutotuneCachedHint): Int = synchronized {
    val maxConcurrentTasks =
      if (cachedHint.hasHint) math.max(0, cachedHint.hint.gpu.maxConcurrentTasks) else 0
    if (maxConcurrentTasks > 0) {
      val version = cachedHint.version
      val sharedMaxConcurrentTasks = math.max(0,
        cachedHint.hint.gpu.sharedMaxConcurrentTasks)
      val schedulingPriority = cachedHint.hint.gpu.schedulingPriority
      activeHints.get(key) match {
        case Some(active) =>
          if (version >= active.hintVersion) {
            active.maxConcurrentTasks = maxConcurrentTasks
            active.sharedMaxConcurrentTasks = sharedMaxConcurrentTasks
            active.schedulingPriority = schedulingPriority
            active.hintVersion = version
          }
          active.activeTasks += taskAttemptId
        case None =>
          activeHints.put(key,
            ActiveGpuAdmissionHint(maxConcurrentTasks, sharedMaxConcurrentTasks,
              schedulingPriority, version, mutable.Set(taskAttemptId)))
      }
      applyCurrentAllocation()
    } else {
      appliedLimit
    }
  }

  def taskCompleted(key: AutotuneStageKey, taskAttemptId: Long): Int = synchronized {
    activeHints.get(key) match {
      // Only release a task that actually registered under a positive cap for this key.
      case Some(active) if active.activeTasks.remove(taskAttemptId) =>
        if (active.activeTasks.isEmpty) {
          activeHints.remove(key)
        }
        applyCurrentAllocation()
      case _ =>
        appliedLimit
    }
  }

  def reset(): Int = synchronized {
    activeHints.clear()
    appliedLimit = applyStageAllocation
      .map(_(0, Map.empty, Map.empty))
      .getOrElse(applyLimit(0))
    appliedLimit
  }

  private def applyCurrentAllocation(): Int = {
    if (activeHints.isEmpty) {
      return reset()
    }

    val stageLimits = activeHints.iterator.map { case (key, hint) =>
      GpuAdmissionStageGroup(key) -> hint.maxConcurrentTasks
    }.toMap
    val stagePriorities = activeHints.iterator.map { case (key, hint) =>
      GpuAdmissionStageGroup(key) -> hint.schedulingPriority
    }.toMap
    val requestedShared = activeHints.valuesIterator
      .map(_.sharedMaxConcurrentTasks)
      .filter(_ > 0)
      .toSeq
    // New hints carry one shared graph budget. Legacy one-field hints retain the old minimum-cap
    // behavior until every active task has refreshed to a graph allocation.
    val requestedLimit = if (requestedShared.nonEmpty) {
      requestedShared.min
    } else {
      activeHints.valuesIterator.map(_.maxConcurrentTasks).min
    }
    val sharedLimit = if (maxSharedConcurrentTasks > 0) {
      math.min(requestedLimit, maxSharedConcurrentTasks)
    } else {
      requestedLimit
    }
    appliedLimit = applyStageAllocation
      .map(_(sharedLimit, stageLimits, stagePriorities))
      .getOrElse(applyLimit(sharedLimit))
    appliedLimit
  }
}

/**
 * Process-wide singleton wiring [[RapidsAutotuneGpuAdmissionController]] to the executor's
 * `GpuSemaphore`. There is one GPU semaphore per executor, so a single controller owns the runtime
 * cap. `reset()` is called on plugin shutdown to drop the runtime cap back to the static limit.
 */
object RapidsAutotuneGpuAdmission {
  // OPTIMIZE mode lets the runtime GPU cap EXCEED the static `concurrentGpuTasks` limit; this is
  // memory-safe because the GpuSemaphore permit pool still gates every admission (see
  // PrioritySemaphore.canAcquire). Set per-executor from the autotune mode and read when the
  // controller pushes the limit into the semaphore. Other modes leave it false (tighten-only).
  @volatile private var allowAboveStatic: Boolean = false

  private val controller = new RapidsAutotuneGpuAdmissionController(
    applyLimit = limit =>
      GpuSemaphore.setRuntimeMaxConcurrentGpuTasksLimit(limit, allowAboveStatic),
    applyStageAllocation = Some((sharedLimit, stageLimits, stagePriorities) =>
      GpuSemaphore.setRuntimeGpuTaskAllocation(
        sharedLimit, stageLimits, stagePriorities, allowAboveStatic)))

  /** Configure the executor-side hard envelope for graph allocations. */
  def configure(allowAboveStatic: Boolean, maxSharedConcurrentTasks: Int): Unit = {
    this.allowAboveStatic = allowAboveStatic
    controller.configureMaxSharedConcurrentTasks(maxSharedConcurrentTasks)
  }

  /** Backward-compatible test hook; production uses [[configure]]. */
  def setAllowAboveStatic(allow: Boolean): Unit = {
    allowAboveStatic = allow
  }

  def taskStarted(key: AutotuneStageKey, taskAttemptId: Long, cachedHint: AutotuneCachedHint): Int =
    controller.taskStarted(key, taskAttemptId, cachedHint)

  def taskCompleted(key: AutotuneStageKey, taskAttemptId: Long): Int =
    controller.taskCompleted(key, taskAttemptId)

  def reset(): Int = {
    allowAboveStatic = false
    controller.reset()
  }
}

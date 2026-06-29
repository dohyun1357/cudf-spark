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
    var hintVersion: Long,
    activeTasks: mutable.Set[Long])

/**
 * Executor-local GPU admission controller driven by graph autotune hints.
 *
 * It tracks, per active stage key, the GPU-concurrency cap requested for that stage and pushes the
 * executor-wide effective cap (the minimum across active stages) into the `GpuSemaphore` via
 * `applyLimit`. The semaphore's memory-permit pool remains the hard safety layer in every mode.
 *
 * Per-key cap policy depends on the mode (`optimizeRaise`):
 *  - GRAPH / default (`optimizeRaise = false`): the cap only ratchets DOWN for the life of a key's
 *    active window. A newer hint that *raises* the cap is ignored while tasks admitted under the
 *    stricter cap are still running, so admission never loosens out from under them (tighten-only).
 *  - OPTIMIZE (`optimizeRaise = true`): the cap tracks the highest-VERSION hint seen for the key,
 *    so it follows the model's latest decision UP as well as down. Raising out from under active
 *    tasks is safe here: the permit pool still gates every admission (see PrioritySemaphore), so
 *    above-static concurrency cannot OOM. Versioning (not arrival order) decides which hint wins,
 *    so out-of-order task starts cannot regress the cap to a stale value.
 *
 * Active membership is tracked by `taskAttemptId`, not a bare counter, so a completion only ever
 * releases a task that actually registered. This matters once hints change mid-stage (the Slice-2
 * fetch TTL): a task that started under no hint (cap 0, never registered) must NOT release an entry
 * created by a sibling task that started under a positive cap.
 */
private[rapids] class RapidsAutotuneGpuAdmissionController(
    applyLimit: Int => Int,
    optimizeRaise: () => Boolean = () => false) {
  private val activeHints = mutable.HashMap.empty[AutotuneStageKey, ActiveGpuAdmissionHint]

  def taskStarted(
      key: AutotuneStageKey,
      taskAttemptId: Long,
      cachedHint: AutotuneCachedHint): Int = synchronized {
    val maxConcurrentTasks =
      if (cachedHint.hasHint) math.max(0, cachedHint.hint.gpu.maxConcurrentTasks) else 0
    if (maxConcurrentTasks > 0) {
      val version = cachedHint.version
      activeHints.get(key) match {
        case Some(active) =>
          if (optimizeRaise()) {
            // Track the latest model decision (by hint version) -- may raise or lower the cap.
            if (version >= active.hintVersion) {
              active.maxConcurrentTasks = maxConcurrentTasks
              active.hintVersion = version
            }
          } else {
            // Ratchet to the tightest cap seen during this key's active window (see class doc).
            active.maxConcurrentTasks = math.min(active.maxConcurrentTasks, maxConcurrentTasks)
          }
          active.activeTasks += taskAttemptId
        case None =>
          activeHints.put(key,
            ActiveGpuAdmissionHint(maxConcurrentTasks, version, mutable.Set(taskAttemptId)))
      }
      applyCurrentLimit()
    } else {
      currentLimit
    }
  }

  def taskCompleted(key: AutotuneStageKey, taskAttemptId: Long): Int = synchronized {
    activeHints.get(key) match {
      // Only release a task that actually registered under a positive cap for this key.
      case Some(active) if active.activeTasks.remove(taskAttemptId) =>
        if (active.activeTasks.isEmpty) {
          activeHints.remove(key)
        }
        applyCurrentLimit()
      case _ =>
        currentLimit
    }
  }

  def reset(): Int = synchronized {
    activeHints.clear()
    applyLimit(0)
  }

  private def currentLimit: Int = {
    if (activeHints.isEmpty) {
      0
    } else {
      activeHints.values.map(_.maxConcurrentTasks).min
    }
  }

  private def applyCurrentLimit(): Int = applyLimit(currentLimit)
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
    limit => GpuSemaphore.setRuntimeMaxConcurrentGpuTasksLimit(limit, allowAboveStatic),
    () => allowAboveStatic)

  /** Set whether the runtime cap may exceed the static cap (true only in OPTIMIZE mode). */
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

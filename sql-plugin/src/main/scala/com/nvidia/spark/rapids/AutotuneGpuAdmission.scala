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
    var activeTasks: Int)

/**
 * Executor-local GPU admission controller driven by graph autotune hints.
 *
 * It tracks, per active stage key, the tightest GPU-concurrency cap requested by tasks of that
 * stage and pushes the executor-wide effective cap (the minimum across active stages) into the
 * `GpuSemaphore` via `applyLimit`. The runtime cap can only ever *tighten* the static
 * `spark.rapids.sql.concurrentGpuTasks` limit; the semaphore remains the memory-safety layer.
 *
 * Conservatism by design: a key's cap only ratchets down for the life of its active window (until
 * its in-flight tasks drain). A newer hint that *raises* the cap is ignored while tasks admitted
 * under the stricter cap are still running, so admission never loosens out from under them.
 */
private[rapids] class RapidsAutotuneGpuAdmissionController(applyLimit: Int => Int) {
  private val activeHints = mutable.HashMap.empty[AutotuneStageKey, ActiveGpuAdmissionHint]

  def taskStarted(key: AutotuneStageKey, cachedHint: AutotuneCachedHint): Int = synchronized {
    val maxConcurrentTasks =
      if (cachedHint.hasHint) math.max(0, cachedHint.hint.gpu.maxConcurrentTasks) else 0
    if (maxConcurrentTasks > 0) {
      activeHints.get(key) match {
        case Some(active) =>
          // Ratchet to the tightest cap seen during this key's active window (see class doc).
          active.maxConcurrentTasks = math.min(active.maxConcurrentTasks, maxConcurrentTasks)
          active.activeTasks += 1
        case None =>
          activeHints.put(key, ActiveGpuAdmissionHint(maxConcurrentTasks, activeTasks = 1))
      }
      applyCurrentLimit()
    } else {
      currentLimit
    }
  }

  def taskCompleted(key: AutotuneStageKey): Int = synchronized {
    activeHints.get(key) match {
      case Some(active) =>
        active.activeTasks -= 1
        if (active.activeTasks <= 0) {
          activeHints.remove(key)
        }
        applyCurrentLimit()
      case None =>
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
  private val controller =
    new RapidsAutotuneGpuAdmissionController(GpuSemaphore.setRuntimeMaxConcurrentGpuTasksLimit)

  def taskStarted(key: AutotuneStageKey, cachedHint: AutotuneCachedHint): Int =
    controller.taskStarted(key, cachedHint)

  def taskCompleted(key: AutotuneStageKey): Int = controller.taskCompleted(key)

  def reset(): Int = controller.reset()
}

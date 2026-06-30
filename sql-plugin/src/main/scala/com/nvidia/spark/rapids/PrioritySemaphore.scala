/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.
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

import java.util.PriorityQueue
import java.util.concurrent.locks.{Condition, ReentrantLock}

import scala.collection.JavaConverters.asScalaIteratorConverter
import scala.collection.mutable

import org.apache.spark.sql.rapids.GpuTaskMetrics

object PrioritySemaphore {
  val NoAdmissionGroup: Long = Long.MinValue
}

class PrioritySemaphore[T](val maxPermits: Long, val maxConcurrentGpuTasksLimit: Int)
  (implicit ordering: Ordering[T]) {
  // This lock is used to generate condition variables, which affords us the flexibility to notify
  // specific threads at a time. If we use the regular synchronized pattern, we have to either
  // notify randomly, or if we try creating condition variables not tied to a shared lock, they
  // won't work together properly, and we see things like deadlocks.
  private val lock = new ReentrantLock()
  private var occupiedSlots: Long = 0
  private var currentConcurrentGpuTasksNum: Long = 0
  private var runtimeMaxConcurrentGpuTasksLimit: Int = 0
  // Optional graph-autotuner allocation. The global runtime limit remains authoritative; these
  // maps additionally cap each active stage and order eligible waiters by modeled critical-path
  // value. Missing groups retain the historical behavior.
  private var runtimeGroupTaskLimits = Map.empty[Long, Int]
  private var runtimeGroupPriorities = Map.empty[Long, Long]
  private val currentConcurrentGpuTasksByGroup = mutable.HashMap.empty[Long, Long]
  // When true, the runtime cap may EXCEED the static `maxConcurrentGpuTasksLimit` (autotune
  // OPTIMIZE mode). This is memory-safe by construction: `canAcquire` still gates every admission
  // on the permit pool (`occupiedSlots + numPermits <= maxPermits`), so a higher task-count cap
  // only lets more tasks attempt -- permits still throttle and cannot be exceeded. Default false
  // keeps the historical tighten-only `min(static, runtime)` behavior.
  private var runtimeMaxCanExceedStatic: Boolean = false

  private case class ThreadInfo(priority: T,
                                condition: Condition,
                                computeNumPermits: () => Long,
                                wasOnGpuBefore: () => Boolean,
                                taskId: Long,
                                admissionGroup: Long) {
    var signaled: Boolean = false
    var permitsUsed: Long = 0
  }

  // Higher graph critical-path priority wins first, then the existing RAPIDS task priority, with
  // task id as a deterministic tie breaker. Group priorities may change when the graph optimizer
  // reallocates; `setRuntimeGpuTaskAllocation` re-heaps queued waiters under the new ordering.
  private val priorityComp = new Ordering[ThreadInfo] {
    override def compare(left: ThreadInfo, right: ThreadInfo): Int = {
      val groupCmp = java.lang.Long.compare(
        groupPriority(right.admissionGroup), groupPriority(left.admissionGroup))
      if (groupCmp != 0) {
        groupCmp
      } else {
        val taskPriorityCmp = ordering.compare(right.priority, left.priority)
        if (taskPriorityCmp != 0) taskPriorityCmp else left.taskId.compareTo(right.taskId)
      }
    }
  }

  // We expect a relatively small number of threads to be contending for this lock at any given
  // time, therefore we are not concerned with the insertion/removal time complexity.
  private val waitingQueue: PriorityQueue[ThreadInfo] =
    new PriorityQueue[ThreadInfo](priorityComp)

  def tryAcquire(numPermits: Long,
                 priority: T,
                 wasOnGpuBefore: () => Boolean,
                 taskAttemptId: Long,
                 admissionGroup: Long = PrioritySemaphore.NoAdmissionGroup): Boolean = {
    lock.lock()
    try {
      if (waitingQueue.size() > 0 &&
        priorityComp.compare(
          waitingQueue.peek(),
          ThreadInfo(priority, null, () => numPermits, wasOnGpuBefore, taskAttemptId,
            admissionGroup)
        ) < 0) {
        false
      } else if (!canAcquire(numPermits, admissionGroup)) {
        false
      } else {
        commitAcquire(numPermits, admissionGroup)
        true
      }
    } finally {
      lock.unlock()
    }
  }

  def acquire(computePermits: () => Long, wasOnGpuBefore: () => Boolean,
              priority: T, taskAttemptId: Long,
              admissionGroup: Long = PrioritySemaphore.NoAdmissionGroup): Long = {
    lock.lock()
    try {
      val numPermitsNow = computePermits()
      if (!tryAcquire(numPermitsNow, priority, wasOnGpuBefore, taskAttemptId, admissionGroup)) {
        val condition = lock.newCondition()
        val info = ThreadInfo(priority, condition, computePermits, wasOnGpuBefore, taskAttemptId,
          admissionGroup)
        try {
          waitingQueue.add(info)
          // only count tasks that had held semaphore before,
          // so they're very likely to have remaining data on GPU
          GpuTaskMetrics.get.recordOnGpuTasksWaitingNumber(
            waitingQueue.iterator().asScala.count(_.wasOnGpuBefore()))

          while (!info.signaled) {
            info.condition.await()
          }
          info.permitsUsed
        } catch {
          case e: Exception =>
            waitingQueue.remove(info)
            if (info.signaled) {
              release(info.permitsUsed, info.admissionGroup)
            }
            throw e
        }
      } else {
        numPermitsNow
      }
    } finally {
      lock.unlock()
    }
  }

  private def commitAcquire(numPermits: Long, admissionGroup: Long): Unit = {
    occupiedSlots += numPermits
    currentConcurrentGpuTasksNum += 1
    if (admissionGroup != PrioritySemaphore.NoAdmissionGroup) {
      currentConcurrentGpuTasksByGroup.update(admissionGroup,
        currentConcurrentGpuTasksByGroup.getOrElse(admissionGroup, 0L) + 1L)
    }
    // Report current concurrent tasks to GpuTaskMetrics, let it handle max tracking
    GpuTaskMetrics.get.recordConcurrentGpuTasks(currentConcurrentGpuTasksNum)
  }

  def release(
      numPermits: Long,
      admissionGroup: Long = PrioritySemaphore.NoAdmissionGroup): Unit = {
    lock.lock()
    try {
      occupiedSlots -= numPermits
      currentConcurrentGpuTasksNum -= 1
      if (admissionGroup != PrioritySemaphore.NoAdmissionGroup) {
        val remaining = currentConcurrentGpuTasksByGroup.getOrElse(admissionGroup, 0L) - 1L
        if (remaining <= 0L) {
          currentConcurrentGpuTasksByGroup.remove(admissionGroup)
        } else {
          currentConcurrentGpuTasksByGroup.update(admissionGroup, remaining)
        }
      }
      wakeEligibleWaiters()
    } finally {
      lock.unlock()
    }
  }

  def setRuntimeMaxConcurrentGpuTasksLimit(
      maxTasks: Int,
      allowAboveStatic: Boolean = false): Int = {
    lock.lock()
    try {
      runtimeMaxConcurrentGpuTasksLimit = math.max(0, maxTasks)
      runtimeMaxCanExceedStatic = allowAboveStatic
      runtimeGroupTaskLimits = Map.empty
      runtimeGroupPriorities = Map.empty
      reheapWaitingQueue()
      wakeEligibleWaiters()
      effectiveMaxConcurrentGpuTasksLimit
    } finally {
      lock.unlock()
    }
  }

  /**
   * Install one graph-wide GPU allocation atomically. `maxTasks` is the shared executor-wide cap;
   * `groupLimits` are stage quotas inside that cap; and `groupPriorities` order eligible waiters.
   * Memory permits are still checked independently for every acquisition.
   */
  def setRuntimeGpuTaskAllocation(
      maxTasks: Int,
      groupLimits: Map[Long, Int],
      groupPriorities: Map[Long, Long],
      allowAboveStatic: Boolean = false): Int = {
    lock.lock()
    try {
      runtimeMaxConcurrentGpuTasksLimit = math.max(0, maxTasks)
      runtimeMaxCanExceedStatic = allowAboveStatic
      runtimeGroupTaskLimits = groupLimits.map { case (group, limit) =>
        group -> math.max(0, limit)
      }
      runtimeGroupPriorities = groupPriorities
      reheapWaitingQueue()
      wakeEligibleWaiters()
      effectiveMaxConcurrentGpuTasksLimit
    } finally {
      lock.unlock()
    }
  }

  def getEffectiveMaxConcurrentGpuTasksLimit: Int = {
    lock.lock()
    try {
      effectiveMaxConcurrentGpuTasksLimit
    } finally {
      lock.unlock()
    }
  }

  private[rapids] def getNumWaitingTasks: Int = {
    lock.lock()
    try {
      waitingQueue.size()
    } finally {
      lock.unlock()
    }
  }

  private def wakeEligibleWaiters(): Unit = {
    // acquire and wakeup for all threads that now have enough permits
    var done = false
    while (!done && waitingQueue.size() > 0) {
      val nextThread = waitingQueue.peek()
      val threadPermits = nextThread.computeNumPermits()
      if (canAcquire(threadPermits, nextThread.admissionGroup)) {
        val popped = waitingQueue.poll()
        assert(popped eq nextThread)
        commitAcquire(threadPermits, nextThread.admissionGroup)
        nextThread.signaled = true
        nextThread.permitsUsed = threadPermits
        nextThread.condition.signal()
      } else {
        done = true
      }
    }
  }

  private def effectiveMaxConcurrentGpuTasksLimit: Int = {
    if (runtimeMaxConcurrentGpuTasksLimit <= 0) {
      maxConcurrentGpuTasksLimit
    } else if (maxConcurrentGpuTasksLimit <= 0) {
      runtimeMaxConcurrentGpuTasksLimit
    } else if (runtimeMaxCanExceedStatic) {
      // OPTIMIZE: the runtime cap fully governs (may exceed static); the permit pool still gates.
      runtimeMaxConcurrentGpuTasksLimit
    } else {
      math.min(maxConcurrentGpuTasksLimit, runtimeMaxConcurrentGpuTasksLimit)
    }
  }

  private def canAcquire(numPermits: Long, admissionGroup: Long): Boolean = {
    val hasPermits = occupiedSlots + numPermits <= maxPermits
    val effectiveTaskLimit = effectiveMaxConcurrentGpuTasksLimit
    val withinTaskLimit = effectiveTaskLimit <= 0 ||
      currentConcurrentGpuTasksNum < effectiveTaskLimit
    val withinGroupLimit = runtimeGroupTaskLimits.get(admissionGroup).forall { limit =>
      limit > 0 && currentConcurrentGpuTasksByGroup.getOrElse(admissionGroup, 0L) < limit
    }
    hasPermits && withinTaskLimit && withinGroupLimit
  }

  private def groupPriority(admissionGroup: Long): Long =
    runtimeGroupPriorities.getOrElse(admissionGroup, 0L)

  private def reheapWaitingQueue(): Unit = {
    if (!waitingQueue.isEmpty) {
      val queued = new java.util.ArrayList[ThreadInfo](waitingQueue)
      waitingQueue.clear()
      waitingQueue.addAll(queued)
    }
  }
}

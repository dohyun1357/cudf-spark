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

import java.util.concurrent.ConcurrentLinkedQueue

import org.scalatest.funsuite.AnyFunSuite

class PrioritySemaphoreRuntimeLimitSuite extends AnyFunSuite {
  type TestPrioritySemaphore = PrioritySemaphore[Long]

  test("runtime maxConcurrentGpuTasks limit tightens disabled static limit") {
    val semaphore = new TestPrioritySemaphore(100, 0)

    assert(semaphore.setRuntimeMaxConcurrentGpuTasksLimit(1) == 1)
    assert(semaphore.getEffectiveMaxConcurrentGpuTasksLimit == 1)

    assert(semaphore.tryAcquire(10, 0, () => false, 0))
    assert(!semaphore.tryAcquire(10, 0, () => false, 1))

    assert(semaphore.setRuntimeMaxConcurrentGpuTasksLimit(0) == 0)
    assert(semaphore.tryAcquire(10, 0, () => false, 1))

    semaphore.release(10)
    semaphore.release(10)
  }

  test("runtime maxConcurrentGpuTasks limit cannot loosen static hard limit") {
    val semaphore = new TestPrioritySemaphore(100, 2)

    assert(semaphore.setRuntimeMaxConcurrentGpuTasksLimit(4) == 2)
    assert(semaphore.getEffectiveMaxConcurrentGpuTasksLimit == 2)

    assert(semaphore.tryAcquire(10, 0, () => false, 0))
    assert(semaphore.tryAcquire(10, 0, () => false, 1))
    assert(!semaphore.tryAcquire(10, 0, () => false, 2))

    semaphore.release(10)
    semaphore.release(10)
  }

  test("runtime maxConcurrentGpuTasks limit can tighten static hard limit") {
    val semaphore = new TestPrioritySemaphore(100, 4)

    assert(semaphore.setRuntimeMaxConcurrentGpuTasksLimit(2) == 2)
    assert(semaphore.tryAcquire(10, 0, () => false, 0))
    assert(semaphore.tryAcquire(10, 0, () => false, 1))
    assert(!semaphore.tryAcquire(10, 0, () => false, 2))

    semaphore.release(10)
    semaphore.release(10)
  }

  test("runtime limit may exceed the static limit when allowAboveStatic (OPTIMIZE)") {
    val semaphore = new TestPrioritySemaphore(100, 2) // static cap 2, ample permits

    assert(semaphore.setRuntimeMaxConcurrentGpuTasksLimit(4, allowAboveStatic = true) == 4)
    assert(semaphore.getEffectiveMaxConcurrentGpuTasksLimit == 4)

    assert(semaphore.tryAcquire(10, 0, () => false, 0))
    assert(semaphore.tryAcquire(10, 0, () => false, 1))
    assert(semaphore.tryAcquire(10, 0, () => false, 2)) // above the static cap of 2 -> allowed
    assert(semaphore.tryAcquire(10, 0, () => false, 3))
    assert(!semaphore.tryAcquire(10, 0, () => false, 4)) // 5th blocked by the runtime task limit 4

    (0 until 4).foreach(_ => semaphore.release(10))
  }

  test("permit pool still gates an above-static runtime limit (cannot OOM)") {
    // Task-count limit raised well above static, but only 25 permits in the pool.
    val semaphore = new TestPrioritySemaphore(25, 2)

    assert(semaphore.setRuntimeMaxConcurrentGpuTasksLimit(8, allowAboveStatic = true) == 8)
    assert(semaphore.tryAcquire(10, 0, () => false, 0)) // 10/25 permits used
    assert(semaphore.tryAcquire(10, 0, () => false, 1)) // 20/25 permits used
    // Task limit is 8, but a 3rd 10-permit task needs 30 > 25 -> blocked by PERMITS, not the cap.
    assert(!semaphore.tryAcquire(10, 0, () => false, 2))

    semaphore.release(10)
    semaphore.release(10)
  }

  test("graph allocation enforces shared cap and per-stage quotas") {
    val semaphore = new TestPrioritySemaphore(100, 0)
    val left = 10L
    val right = 20L

    assert(semaphore.setRuntimeGpuTaskAllocation(
      maxTasks = 3,
      groupLimits = Map(left -> 2, right -> 1),
      groupPriorities = Map(left -> 100L, right -> 40L)) == 3)

    assert(semaphore.tryAcquire(10, 0L, () => false, 1L, left))
    assert(semaphore.tryAcquire(10, 0L, () => false, 2L, left))
    assert(!semaphore.tryAcquire(10, 0L, () => false, 3L, left)) // left quota exhausted
    assert(semaphore.tryAcquire(10, 0L, () => false, 4L, right))
    assert(!semaphore.tryAcquire(10, 0L, () => false, 5L)) // shared cap exhausted

    semaphore.release(10, left)
    assert(semaphore.tryAcquire(10, 0L, () => false, 5L, left))

    semaphore.release(10, left)
    semaphore.release(10, left)
    semaphore.release(10, right)
  }

  test("graph critical-path priority orders eligible stage waiters") {
    val semaphore = new TestPrioritySemaphore(100, 0)
    val low = 10L
    val high = 20L
    semaphore.setRuntimeGpuTaskAllocation(
      maxTasks = 1,
      groupLimits = Map(low -> 1, high -> 1),
      groupPriorities = Map(low -> 10L, high -> 100L))
    assert(semaphore.tryAcquire(10, 0L, () => false, 0L))

    val acquisitionOrder = new ConcurrentLinkedQueue[Long]()
    def waiter(group: Long, taskId: Long): Thread = new Thread(new Runnable {
      override def run(): Unit = {
        val permits = semaphore.acquire(() => 10L, () => false, 0L, taskId, group)
        acquisitionOrder.add(group)
        semaphore.release(permits, group)
      }
    })
    def awaitWaiters(expected: Int): Unit = {
      val deadline = System.nanoTime() + 5000000000L
      while (semaphore.getNumWaitingTasks < expected && System.nanoTime() < deadline) {
        Thread.`yield`()
      }
      assert(semaphore.getNumWaitingTasks == expected)
    }

    val lowThread = waiter(low, 1L)
    lowThread.start()
    awaitWaiters(1)
    val highThread = waiter(high, 2L)
    highThread.start()
    awaitWaiters(2)

    semaphore.release(10)
    highThread.join(5000L)
    lowThread.join(5000L)
    assert(!highThread.isAlive && !lowThread.isAlive)
    assert(acquisitionOrder.toArray.toSeq == Seq(high, low))
  }
}

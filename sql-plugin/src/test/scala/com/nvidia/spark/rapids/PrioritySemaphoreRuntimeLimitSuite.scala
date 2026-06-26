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
}

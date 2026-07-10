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

package com.nvidia.spark.rapids.parquet

import java.util.concurrent.{CountDownLatch, Executors, TimeUnit}
import java.util.concurrent.atomic.AtomicInteger

import ai.rapids.cudf.HostMemoryBuffer
import com.nvidia.spark.rapids.Arm.withResource
import org.scalatest.BeforeAndAfterEach
import org.scalatest.funsuite.AnyFunSuite

class ParquetFooterCacheSuite extends AnyFunSuite with BeforeAndAfterEach {

  override def beforeEach(): Unit = {
    ParquetFooterCache.clearForTesting()
    ParquetFooterCache.clearMetaForTesting()
  }

  override def afterEach(): Unit = {
    ParquetFooterCache.clearForTesting()
    ParquetFooterCache.clearMetaForTesting()
  }

  private def key(path: String, size: Long = 100, mtime: Long = 1): ParquetFooterCache.Key =
    ParquetFooterCache.Key(path, size, mtime)

  private def buf(len: Int, fill: Byte): HostMemoryBuffer = {
    val b = HostMemoryBuffer.allocate(len, false)
    (0 until len).foreach(i => b.setByte(i, fill))
    b
  }

  test("second lookup of the same key does not invoke the loader") {
    val loads = new AtomicInteger(0)
    def load(): HostMemoryBuffer = { loads.incrementAndGet(); buf(16, 1) }
    withResource(ParquetFooterCache.getOrLoad(key("a"), 1024 * 1024)(load())) { b1 =>
      assert(b1.getLength == 16)
      assert(b1.getByte(3) == 1)
    }
    withResource(ParquetFooterCache.getOrLoad(key("a"), 1024 * 1024)(load())) { b2 =>
      assert(b2.getLength == 16)
      assert(b2.getByte(7) == 1)
    }
    assert(loads.get() == 1)
  }

  test("distinct sizes or mtimes for the same path are distinct entries") {
    val loads = new AtomicInteger(0)
    def load(fill: Byte): HostMemoryBuffer = { loads.incrementAndGet(); buf(8, fill) }
    withResource(ParquetFooterCache.getOrLoad(key("a", size = 1), 1024 * 1024)(load(1))) { b =>
      assert(b.getByte(0) == 1)
    }
    withResource(ParquetFooterCache.getOrLoad(key("a", size = 2), 1024 * 1024)(load(2))) { b =>
      assert(b.getByte(0) == 2)
    }
    withResource(ParquetFooterCache.getOrLoad(key("a", size = 1, mtime = 9),
        1024 * 1024)(load(3))) { b =>
      assert(b.getByte(0) == 3)
    }
    assert(loads.get() == 3)
  }

  test("cached bytes survive after the first caller closes its slice") {
    withResource(ParquetFooterCache.getOrLoad(key("a"), 1024 * 1024)(buf(32, 5))) { b =>
      assert(b.getByte(31) == 5)
    }
    // First slice closed above; a hit must still return live, correct bytes.
    withResource(ParquetFooterCache.getOrLoad(key("a"), 1024 * 1024)(
        throw new IllegalStateException("loader must not run on a hit"))) { b =>
      assert(b.getLength == 32)
      assert(b.getByte(0) == 5)
    }
  }

  test("eviction respects the byte budget in LRU order") {
    val loads = new AtomicInteger(0)
    def load(fill: Byte): HostMemoryBuffer = { loads.incrementAndGet(); buf(64, fill) }
    // Budget of 128 holds exactly two 64-byte entries.
    withResource(ParquetFooterCache.getOrLoad(key("a"), 128)(load(1)))(_ => ())
    withResource(ParquetFooterCache.getOrLoad(key("b"), 128)(load(2)))(_ => ())
    // Touch "a" so "b" becomes LRU, then insert "c" which must evict "b".
    withResource(ParquetFooterCache.getOrLoad(key("a"), 128)(load(1)))(_ => ())
    withResource(ParquetFooterCache.getOrLoad(key("c"), 128)(load(3)))(_ => ())
    assert(loads.get() == 3)
    withResource(ParquetFooterCache.getOrLoad(key("a"), 128)(load(1)))(_ => ())
    assert(loads.get() == 3, "a must still be cached")
    withResource(ParquetFooterCache.getOrLoad(key("b"), 128)(load(2)))(_ => ())
    assert(loads.get() == 4, "b must have been evicted")
  }

  test("a slice taken before eviction stays valid after eviction") {
    val outer = ParquetFooterCache.getOrLoad(key("a"), 128)(buf(64, 7))
    try {
      // Evict "a" by filling the budget with new keys.
      withResource(ParquetFooterCache.getOrLoad(key("b"), 128)(buf(64, 8)))(_ => ())
      withResource(ParquetFooterCache.getOrLoad(key("c"), 128)(buf(64, 9)))(_ => ())
      // The cache reference is gone but our slice must still be readable.
      assert(outer.getByte(0) == 7)
      assert(outer.getByte(63) == 7)
    } finally {
      outer.close()
    }
  }

  test("failed loads are not cached and later attempts retry") {
    val attempts = new AtomicInteger(0)
    def failing(): HostMemoryBuffer = {
      attempts.incrementAndGet()
      throw new java.io.FileNotFoundException("nope")
    }
    intercept[java.io.FileNotFoundException] {
      ParquetFooterCache.getOrLoad(key("a"), 1024)(failing())
    }
    intercept[java.io.FileNotFoundException] {
      ParquetFooterCache.getOrLoad(key("a"), 1024)(failing())
    }
    assert(attempts.get() == 2)
    // And a successful load afterwards is cached normally.
    withResource(ParquetFooterCache.getOrLoad(key("a"), 1024)(buf(8, 4))) { b =>
      assert(b.getByte(0) == 4)
    }
  }

  test("budget of zero disables caching entirely") {
    val loads = new AtomicInteger(0)
    def load(): HostMemoryBuffer = { loads.incrementAndGet(); buf(8, 1) }
    withResource(ParquetFooterCache.getOrLoad(key("a"), 0)(load()))(_ => ())
    withResource(ParquetFooterCache.getOrLoad(key("a"), 0)(load()))(_ => ())
    assert(loads.get() == 2)
  }

  private def metaKey(path: String, filters: String = "f",
      schema: String = "s"): ParquetFooterCache.MetaKey =
    ParquetFooterCache.MetaKey(path, 100, 1, 0, 100, schema, filters, "h")

  test("metadata section caches per key and returns copies") {
    val loads = new AtomicInteger(0)
    case class Meta(var tag: Int)
    def load(): Meta = { loads.incrementAndGet(); Meta(1) }
    val first = ParquetFooterCache.getOrLoadMeta(metaKey("a"), 100)(load())(m => m.copy())
    assert(first.tag == 1)
    first.tag = 99 // mutating the returned copy must not corrupt the cache
    val second = ParquetFooterCache.getOrLoadMeta(metaKey("a"), 100)(load())(m => m.copy())
    assert(second.tag == 1)
    assert(loads.get() == 1)
    // Different filters or schema are different entries.
    ParquetFooterCache.getOrLoadMeta(metaKey("a", filters = "g"), 100)(load())(m => m.copy())
    ParquetFooterCache.getOrLoadMeta(metaKey("a", schema = "t"), 100)(load())(m => m.copy())
    assert(loads.get() == 3)
  }

  test("metadata section count eviction and disable") {
    val loads = new AtomicInteger(0)
    val id = (s: String) => s
    def load(): String = { loads.incrementAndGet(); "v" }
    ParquetFooterCache.getOrLoadMeta(metaKey("a"), 2)(load())(id)
    ParquetFooterCache.getOrLoadMeta(metaKey("b"), 2)(load())(id)
    ParquetFooterCache.getOrLoadMeta(metaKey("a"), 2)(load())(id) // touch a
    ParquetFooterCache.getOrLoadMeta(metaKey("c"), 2)(load())(id) // evicts b
    assert(loads.get() == 3)
    ParquetFooterCache.getOrLoadMeta(metaKey("a"), 2)(load())(id)
    assert(loads.get() == 3)
    ParquetFooterCache.getOrLoadMeta(metaKey("b"), 2)(load())(id)
    assert(loads.get() == 4)
    // disabled
    ParquetFooterCache.getOrLoadMeta(metaKey("z"), 0)(load())(id)
    ParquetFooterCache.getOrLoadMeta(metaKey("z"), 0)(load())(id)
    assert(loads.get() == 6)
  }

  test("metadata section failed loads retry") {
    val attempts = new AtomicInteger(0)
    val id = (s: String) => s
    def failing(): String = {
      attempts.incrementAndGet()
      throw new java.io.FileNotFoundException("nope")
    }
    intercept[java.io.FileNotFoundException] {
      ParquetFooterCache.getOrLoadMeta[String](metaKey("a"), 10)(failing())(id)
    }
    val v = ParquetFooterCache.getOrLoadMeta[String](metaKey("a"), 10) {
      attempts.incrementAndGet(); "ok"
    }(id)
    assert(v == "ok")
    assert(attempts.get() == 2)
  }

  test("concurrent lookups of one key load once and all see the same bytes") {
    val loads = new AtomicInteger(0)
    val entered = new CountDownLatch(1)
    val release = new CountDownLatch(1)
    val pool = Executors.newFixedThreadPool(8)
    try {
      def slowLoad(): HostMemoryBuffer = {
        loads.incrementAndGet()
        entered.countDown()
        release.await(30, TimeUnit.SECONDS)
        buf(16, 42)
      }
      val futures = (0 until 8).map { _ =>
        pool.submit(new java.util.concurrent.Callable[Int] {
          override def call(): Int = {
            withResource(ParquetFooterCache.getOrLoad(key("hot"), 1024 * 1024)(slowLoad())) { b =>
              b.getByte(5).toInt
            }
          }
        })
      }
      // Wait until one loader is inside, then release it; every waiter shares its result.
      assert(entered.await(30, TimeUnit.SECONDS))
      release.countDown()
      futures.foreach(f => assert(f.get(30, TimeUnit.SECONDS) == 42))
      assert(loads.get() == 1)
    } finally {
      pool.shutdownNow()
    }
  }
}

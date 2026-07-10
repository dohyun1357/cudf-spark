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

import java.util.concurrent.CompletableFuture
import java.util.concurrent.atomic.LongAdder

import scala.util.control.NonFatal

import ai.rapids.cudf.HostMemoryBuffer
import com.nvidia.spark.rapids.Arm.closeOnExcept

import org.apache.spark.internal.Logging

/**
 * Executor-wide in-memory cache of raw (framed) Parquet footer bytes.
 *
 * Every task reads and parses the footer of every file (or file split) it scans. On tables laid
 * out as many small files — hive/iceberg/delta style date-partitioned tables are the common case
 * — the same footer is fetched over and over: once per task per file, thousands of times per
 * query for a table with ~2k files, with each fetch paying several filesystem round trips
 * (file status, open + trailer read, open + footer body read). Those round trips, not the footer
 * parse, dominate scan time for such layouts.
 *
 * The cache stores the framed footer buffer (`MAGIC + footer + footerLen + MAGIC`, the exact
 * bytes `ParquetFooterUtils` produces) keyed by (path, file size, modification time), all of
 * which come from Spark's `PartitionedFile` without extra filesystem calls. Loading is
 * single-flight: concurrent tasks that miss on the same key wait for one load instead of issuing
 * duplicate reads. Entries are evicted in LRU order once the configured byte budget is exceeded.
 *
 * Ownership: the cache holds one reference to each buffer; callers always receive a slice
 * (which increments the underlying reference count) and must close it. Eviction closes the
 * cache's reference, so the memory is freed when the last outstanding slice is closed.
 *
 * Predicate filtering is deliberately NOT cached: filtering depends on the query, and parsing
 * cached bytes is microseconds. Only the immutable bytes are shared.
 */
object ParquetFooterCache extends Logging {

  case class Key(path: String, fileSize: Long, modificationTime: Long)

  private class Entry {
    val future = new CompletableFuture[HostMemoryBuffer]()
    var sizeBytes: Long = 0L
    var evicted: Boolean = false
  }

  private[this] val lock = new Object
  // Access-ordered for LRU; guarded by `lock`.
  private[this] val entries = new java.util.LinkedHashMap[Key, Entry](64, 0.75f, true)
  private[this] var totalBytes = 0L
  private[this] var maxBytesOrUnset = -1L
  private[this] val hitCount = new LongAdder
  private[this] val missCount = new LongAdder

  def hits: Long = hitCount.sum()
  def misses: Long = missCount.sum()

  /** Visible for tests. Drops every cached entry. */
  def clearForTesting(): Unit = lock.synchronized {
    val it = entries.values().iterator()
    while (it.hasNext) {
      val e = it.next()
      if (e.future.isDone && !e.evicted) {
        e.evicted = true
        totalBytes -= e.sizeBytes
        e.future.join().close()
      }
      it.remove()
    }
    totalBytes = 0L
    maxBytesOrUnset = -1L
  }

  private def maxBytes(confMaxSize: Long): Long = {
    if (maxBytesOrUnset < 0) {
      lock.synchronized {
        if (maxBytesOrUnset < 0) {
          maxBytesOrUnset = confMaxSize
          if (maxBytesOrUnset > 0) {
            logInfo(s"Parquet footer cache enabled with a budget of $maxBytesOrUnset bytes")
          }
        }
      }
    }
    maxBytesOrUnset
  }

  /**
   * Return the framed footer buffer for `key`, loading it with `loader` on a miss. The returned
   * buffer is owned by the caller and must be closed; it is always a slice of the cached buffer
   * (or the loader's buffer directly when caching is disabled or raced with eviction).
   */
  def getOrLoad(key: Key, confMaxSize: Long)(loader: => HostMemoryBuffer): HostMemoryBuffer = {
    val budget = maxBytes(confMaxSize)
    if (budget <= 0) {
      return loader
    }
    var owner = false
    val entry = lock.synchronized {
      val existing = entries.get(key)
      if (existing != null && !existing.evicted) {
        existing
      } else {
        val e = new Entry
        entries.put(key, e)
        owner = true
        e
      }
    }
    if (owner) {
      val loaded = try {
        loader
      } catch {
        case t: Throwable =>
          lock.synchronized {
            // Only remove our own failed entry; a different entry may have replaced it.
            if (entries.get(key) eq entry) {
              entries.remove(key)
            }
          }
          entry.future.completeExceptionally(t)
          throw t
      }
      closeOnExcept(loaded) { _ =>
        lock.synchronized {
          entry.sizeBytes = loaded.getLength
          totalBytes += entry.sizeBytes
          evictAsNeeded(budget, keep = entry)
        }
      }
      missCount.increment()
      // Hand the cache its reference first, then take the caller's slice.
      entry.future.complete(loaded)
      sliceOrReload(key, entry, loaded)(loader)
    } else {
      val buf = try {
        entry.future.join()
      } catch {
        case NonFatal(_) =>
          // The owning load failed; fail or succeed on our own terms so each task gets its
          // own exception type and stack, and retries are not poisoned by a stale failure.
          missCount.increment()
          return loader
      }
      hitCount.increment()
      maybeLogStats()
      sliceOrReload(key, entry, buf)(loader)
    }
  }

  /**
   * Take a caller-owned slice of the cached buffer, falling back to a direct load if the entry
   * was evicted before the slice could be taken (the buffer may already be closed then).
   */
  private def sliceOrReload(key: Key, entry: Entry, buf: HostMemoryBuffer)
      (loader: => HostMemoryBuffer): HostMemoryBuffer = {
    val sliced = lock.synchronized {
      if (!entry.evicted) {
        buf.slice(0, buf.getLength)
      } else {
        null
      }
    }
    if (sliced != null) {
      sliced
    } else {
      loader
    }
  }

  /** Must be called while holding `lock`. */
  private def evictAsNeeded(budget: Long, keep: Entry): Unit = {
    if (totalBytes <= budget) {
      return
    }
    val it = entries.entrySet().iterator()
    while (totalBytes > budget && it.hasNext) {
      val e = it.next().getValue
      // Never evict in-flight loads (unknown size, waiters expect a live buffer right after
      // completion) or the entry we just inserted.
      if ((e ne keep) && e.future.isDone && !e.future.isCompletedExceptionally && !e.evicted) {
        e.evicted = true
        totalBytes -= e.sizeBytes
        e.future.join().close()
        it.remove()
      }
    }
  }

  private def maybeLogStats(): Unit = {
    val h = hitCount.sum()
    if ((h & 0x3FFF) == 0 && log.isDebugEnabled) {
      logDebug(s"Parquet footer cache: hits=$h misses=${missCount.sum()} bytes=" +
        lock.synchronized(totalBytes))
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Metadata section: caches the fully parsed and filtered per-split scan metadata
  // (ParquetFileInfoWithBlockMeta) so the footer parse, schema clipping, rebase-mode detection
  // and row-group filtering (including dictionary-page reads for pushed predicates) run once per
  // (file, split, read schema, predicate) instead of once per stage x file. Values may hold
  // mutable parquet-mr objects, so every lookup returns a defensive copy produced by the
  // caller-supplied `copyOut` and the cached instance never escapes.
  // ---------------------------------------------------------------------------------------------

  case class MetaKey(
      path: String,
      fileSize: Long,
      modificationTime: Long,
      splitStart: Long,
      splitLength: Long,
      readSchemaFingerprint: String,
      filtersFingerprint: String,
      handlerFingerprint: String)

  private class MetaEntry {
    val future = new CompletableFuture[AnyRef]()
  }

  private[this] val metaLock = new Object
  private[this] val metaEntries = new java.util.LinkedHashMap[MetaKey, MetaEntry](256, 0.75f, true)
  private[this] val metaHitCount = new LongAdder
  private[this] val metaMissCount = new LongAdder

  def metaHits: Long = metaHitCount.sum()
  def metaMisses: Long = metaMissCount.sum()

  /** Visible for tests. Drops every cached metadata entry. */
  def clearMetaForTesting(): Unit = metaLock.synchronized {
    metaEntries.clear()
  }

  /**
   * Return scan metadata for `key`, loading it with `loader` on a miss. Both hits and the
   * loading call return `copyOut(cached)` so callers can never mutate the cached instance.
   * `maxEntries <= 0` disables the section and just evaluates the loader.
   */
  def getOrLoadMeta[V <: AnyRef](key: MetaKey, maxEntries: Int)
      (loader: => V)(copyOut: V => V): V = {
    if (maxEntries <= 0) {
      return loader
    }
    var owner = false
    val entry = metaLock.synchronized {
      val existing = metaEntries.get(key)
      if (existing != null) {
        existing
      } else {
        val e = new MetaEntry
        metaEntries.put(key, e)
        // Count-based LRU eviction; incomplete loads are skipped.
        if (metaEntries.size() > maxEntries) {
          val it = metaEntries.values().iterator()
          var removed = false
          while (!removed && it.hasNext) {
            val cand = it.next()
            if ((cand ne e) && cand.future.isDone) {
              it.remove()
              removed = true
            }
          }
        }
        owner = true
        e
      }
    }
    if (owner) {
      val loaded = try {
        loader
      } catch {
        case t: Throwable =>
          metaLock.synchronized {
            if (metaEntries.get(key) eq entry) {
              metaEntries.remove(key)
            }
          }
          entry.future.completeExceptionally(t)
          throw t
      }
      metaMissCount.increment()
      entry.future.complete(loaded)
      copyOut(loaded)
    } else {
      val value = try {
        entry.future.join()
      } catch {
        case NonFatal(_) =>
          // The owning load failed; load on our own terms so each task sees its own
          // exception type and stack.
          metaMissCount.increment()
          return loader
      }
      metaHitCount.increment()
      copyOut(value.asInstanceOf[V])
    }
  }
}

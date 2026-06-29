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

import java.util.concurrent.ConcurrentHashMap

import scala.util.control.NonFatal

import org.apache.spark.api.plugin.PluginContext
import org.apache.spark.internal.Logging

/**
 * Executor-local cache of per-stage hints.
 *
 * `get` memoizes the driver response per stage key and refetches it once the cached entry is stale;
 * `fetchHint` is the (RPC) loader. `put` is the driver-push primitive (pre-populate/refresh a hint
 * without a round-trip) for future graph-mode use.
 *
 * An entry is stale when either the hint itself has expired (stage-attempt expiry, carried on the
 * wire as `expiresAtNanos`) OR the entry is older than `fetchTtlNanos`. The fetch TTL is a SEPARATE
 * mechanism from wire expiry: in GRAPH mode the driver republishes higher-version hints mid-query
 * (the Phase 6 closed loop), and the TTL is what makes a long stage's later tasks (and downstream
 * tasks) re-fetch and pick up the newer version. It does not touch `expiresAtNanos`, so the
 * stage-attempt expiry semantics are unchanged. `fetchTtlNanos = 0` disables the TTL (refresh on
 * expiry only),
 * preserving pre-Slice-2 behavior for OBSERVE/LOCAL modes. `nowNanos` is injectable for testing.
 */
class AutotuneHintCache(
    fetchHint: AutotuneStageKey => AutotuneCachedHint,
    fetchTtlNanos: Long = 0L,
    nowNanos: () => Long = () => System.nanoTime()) {

  private case class Entry(hint: AutotuneCachedHint, fetchedAtNanos: Long) {
    def isStale(now: Long): Boolean =
      hint.isExpired(now) || (fetchTtlNanos > 0L && (now - fetchedAtNanos) >= fetchTtlNanos)
  }

  private val hints = new ConcurrentHashMap[AutotuneStageKey, Entry]()

  def get(key: AutotuneStageKey): AutotuneCachedHint = {
    val now = nowNanos()
    val existing = hints.get(key)
    if (existing != null && !existing.isStale(now)) {
      existing.hint
    } else {
      val fetched = Entry(fetchHint(key), now)
      val previous = if (existing == null) {
        hints.putIfAbsent(key, fetched)
      } else if (hints.replace(key, existing, fetched)) {
        null
      } else {
        hints.get(key)
      }
      if (previous == null) {
        fetched.hint
      } else {
        previous.hint
      }
    }
  }

  def put(hint: StageRuntimeHint): Unit = {
    hints.put(hint.key, Entry(AutotuneCachedHint(hint, hasHint = true), nowNanos()))
  }

  def clear(): Unit = hints.clear()
}

/**
 * Executor-side autotune endpoint.
 *
 * Fetches and caches per-stage hints from the driver, applies the GPU admission hint through
 * [[RapidsAutotuneGpuAdmission]] on task start/completion, and reports applied hints back to the
 * driver for eventlog evidence. All driver interaction is fail-open: when `autotune.failOpen` is
 * true, RPC failures fall back to "no hint" / current RAPIDS behavior instead of failing the task.
 */
class RapidsAutotuneExecutorEndpoint(
    pluginContext: PluginContext,
    conf: RapidsConf) extends Logging {
  private val executorId = pluginContext.executorID()
  private val failOpen = conf.autotuneFailOpen
  // Only GRAPH mode republishes hints mid-query, so only there does the cache need a refresh TTL;
  // OBSERVE/LOCAL keep the pre-Slice-2 "refresh on expiry only" behavior (TTL disabled).
  private val fetchTtlNanos =
    if (conf.isAutotuneGraphMode) conf.autotuneGraphUpdateIntervalMs.toLong * 1000000L else 0L
  private val cache = new AutotuneHintCache(fetchHintFromDriver, fetchTtlNanos)

  def hintFor(key: AutotuneStageKey): AutotuneCachedHint = cache.get(key)

  def recordAppliedHint(
      key: AutotuneStageKey,
      taskAttemptId: Long,
      partitionId: Int,
      cachedHint: AutotuneCachedHint,
      gpuAppliedMaxConcurrentTasks: Int): Unit = {
    logDebug(s"Applied RAPIDS graph autotune hint version ${cachedHint.version} " +
      s"for execution ${key.executionId}, stage ${key.stageId}.${key.stageAttemptId}, " +
      s"task $taskAttemptId")
    try {
      pluginContext.send(RapidsAutotuneHintAppliedMsg(
        executorId = executorId,
        key = key,
        taskAttemptId = taskAttemptId,
        partitionId = partitionId,
        hintVersion = cachedHint.version,
        hasHint = cachedHint.hasHint,
        scan = cachedHint.hint.scan,
        gpu = cachedHint.hint.gpu,
        shuffle = cachedHint.hint.shuffle,
        batch = cachedHint.hint.batch,
        gpuAppliedMaxConcurrentTasks = gpuAppliedMaxConcurrentTasks))
    } catch {
      case NonFatal(e) if failOpen =>
        logWarning("Failed to report RAPIDS graph autotune applied hint; continuing", e)
    }
  }

  /**
   * Report a runtime observation for a completed task back to the driver (fire-and-forget). Feeds
   * the driver closed-loop model; advisory, so a dropped report never affects correctness.
   */
  def reportObservation(
      key: AutotuneStageKey,
      taskAttemptId: Long,
      partitionId: Int,
      hintVersion: Long,
      gpuSemaphoreWaitNanos: Long,
      gpuHoldingNanos: Long,
      hostMemoryBytes: Long,
      spillBytes: Long): Unit = {
    try {
      pluginContext.send(RapidsAutotuneObservationMsg(
        executorId = executorId,
        key = key,
        taskAttemptId = taskAttemptId,
        partitionId = partitionId,
        hintVersion = hintVersion,
        gpuSemaphoreWaitNanos = gpuSemaphoreWaitNanos,
        gpuHoldingNanos = gpuHoldingNanos,
        hostMemoryBytes = hostMemoryBytes,
        spillBytes = spillBytes))
    } catch {
      case NonFatal(e) if failOpen =>
        logWarning("Failed to report RAPIDS graph autotune observation; continuing", e)
    }
  }

  def taskStarted(
      key: AutotuneStageKey,
      taskAttemptId: Long,
      cachedHint: AutotuneCachedHint): Int = {
    try {
      RapidsAutotuneGpuAdmission.taskStarted(key, taskAttemptId, cachedHint)
    } catch {
      case NonFatal(e) if failOpen =>
        logWarning("Failed to apply RAPIDS graph autotune GPU hint; continuing", e)
        0
    }
  }

  def taskCompleted(key: AutotuneStageKey, taskAttemptId: Long): Unit = {
    try {
      RapidsAutotuneGpuAdmission.taskCompleted(key, taskAttemptId)
    } catch {
      case NonFatal(e) if failOpen =>
        logWarning("Failed to clear RAPIDS graph autotune GPU hint; continuing", e)
    }
  }

  def shutdown(): Unit = cache.clear()

  private def fetchHintFromDriver(key: AutotuneStageKey): AutotuneCachedHint = {
    try {
      pluginContext.ask(RapidsAutotuneHintRequestMsg(executorId, key)) match {
        case RapidsAutotuneHintResponseMsg(_, Some(hint)) =>
          AutotuneCachedHint(hint, hasHint = true)
        case RapidsAutotuneHintResponseMsg(_, None) =>
          AutotuneCachedHint.empty(key)
        case other =>
          val msg = s"Unexpected RAPIDS graph autotune hint response: $other"
          if (failOpen) {
            logWarning(msg)
            AutotuneCachedHint.empty(key)
          } else {
            throw new IllegalStateException(msg)
          }
      }
    } catch {
      case NonFatal(e) if failOpen =>
        logWarning("Failed to fetch RAPIDS graph autotune hint; continuing with no hint", e)
        AutotuneCachedHint.empty(key)
    }
  }
}

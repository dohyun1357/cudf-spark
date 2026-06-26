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
 * `get` memoizes the driver response per stage key and refreshes it once the cached hint expires;
 * `fetchHint` is the (RPC) loader. `put` is the driver-push primitive (pre-populate/refresh a hint
 * without a round-trip) for future graph-mode use.
 */
class AutotuneHintCache(fetchHint: AutotuneStageKey => AutotuneCachedHint) {
  private val hints = new ConcurrentHashMap[AutotuneStageKey, AutotuneCachedHint]()

  def get(key: AutotuneStageKey): AutotuneCachedHint = {
    val nowNanos = System.nanoTime()
    val existing = hints.get(key)
    if (existing != null && !existing.isExpired(nowNanos)) {
      existing
    } else {
      val fetched = fetchHint(key)
      val previous = if (existing == null) {
        hints.putIfAbsent(key, fetched)
      } else if (hints.replace(key, existing, fetched)) {
        null
      } else {
        hints.get(key)
      }
      if (previous == null) {
        fetched
      } else {
        previous
      }
    }
  }

  def put(hint: StageRuntimeHint): Unit = {
    hints.put(hint.key, AutotuneCachedHint(hint, hasHint = true))
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
  private val cache = new AutotuneHintCache(fetchHintFromDriver)

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
      hostMemoryBytes: Long): Unit = {
    try {
      pluginContext.send(RapidsAutotuneObservationMsg(
        executorId = executorId,
        key = key,
        taskAttemptId = taskAttemptId,
        partitionId = partitionId,
        hintVersion = hintVersion,
        gpuSemaphoreWaitNanos = gpuSemaphoreWaitNanos,
        gpuHoldingNanos = gpuHoldingNanos,
        hostMemoryBytes = hostMemoryBytes))
    } catch {
      case NonFatal(e) if failOpen =>
        logWarning("Failed to report RAPIDS graph autotune observation; continuing", e)
    }
  }

  def taskStarted(key: AutotuneStageKey, cachedHint: AutotuneCachedHint): Int = {
    try {
      RapidsAutotuneGpuAdmission.taskStarted(key, cachedHint)
    } catch {
      case NonFatal(e) if failOpen =>
        logWarning("Failed to apply RAPIDS graph autotune GPU hint; continuing", e)
        0
    }
  }

  def taskCompleted(key: AutotuneStageKey): Unit = {
    try {
      RapidsAutotuneGpuAdmission.taskCompleted(key)
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

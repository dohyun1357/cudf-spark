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

import org.apache.spark.SparkContext
import org.apache.spark.api.plugin.PluginContext
import org.apache.spark.internal.Logging
import org.apache.spark.sql.rapids.execution.TrampolineUtil

case class AutotuneStageKey(
    executionId: Long,
    stageId: Int,
    stageAttemptId: Int)

case class ScanRuntimeHint(
    eagerPrefetch: Boolean,
    minReadWindow: Int,
    maxReadWindow: Int,
    maxReadyBytes: Long)

object ScanRuntimeHint {
  val empty: ScanRuntimeHint = ScanRuntimeHint(
    eagerPrefetch = false,
    minReadWindow = 0,
    maxReadWindow = 0,
    maxReadyBytes = Long.MaxValue)
}

case class StageRuntimeHint(
    executionId: Long,
    stageId: Int,
    stageAttemptId: Int,
    version: Long,
    scan: ScanRuntimeHint,
    expiresAtNanos: Long) {
  def key: AutotuneStageKey = AutotuneStageKey(executionId, stageId, stageAttemptId)

  def isExpired(nowNanos: Long): Boolean = expiresAtNanos <= nowNanos
}

object StageRuntimeHint {
  def empty(key: AutotuneStageKey): StageRuntimeHint = StageRuntimeHint(
    executionId = key.executionId,
    stageId = key.stageId,
    stageAttemptId = key.stageAttemptId,
    version = 0L,
    scan = ScanRuntimeHint.empty,
    expiresAtNanos = Long.MaxValue)
}

case class AutotuneCachedHint(hint: StageRuntimeHint, hasHint: Boolean) {
  def version: Long = hint.version

  def isExpired(nowNanos: Long): Boolean = hasHint && hint.isExpired(nowNanos)
}

object AutotuneCachedHint {
  def empty(key: AutotuneStageKey): AutotuneCachedHint =
    AutotuneCachedHint(StageRuntimeHint.empty(key), hasHint = false)
}

case class RapidsAutotuneHintRequestMsg(
    executorId: String,
    key: AutotuneStageKey)

case class RapidsAutotuneHintResponseMsg(
    key: AutotuneStageKey,
    hint: Option[StageRuntimeHint])

case class RapidsAutotuneHintAppliedMsg(
    executorId: String,
    key: AutotuneStageKey,
    taskAttemptId: Long,
    partitionId: Int,
    hintVersion: Long,
    hasHint: Boolean)

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

object RapidsAutotuneDriverEndpoint extends Logging {
  private val hints = new ConcurrentHashMap[AutotuneStageKey, StageRuntimeHint]()

  @volatile private var sparkContext: SparkContext = _
  @volatile private var enabled = false

  def init(sc: SparkContext, conf: RapidsConf): Unit = synchronized {
    enabled = conf.autotuneGraphEnabled
    sparkContext = if (enabled) sc else null
    hints.clear()
    if (enabled) {
      logInfo(s"Initialized RAPIDS graph autotune driver endpoint in " +
        s"${conf.autotuneGraphMode} mode")
    }
  }

  def shutdown(): Unit = synchronized {
    enabled = false
    sparkContext = null
    hints.clear()
  }

  def publishHint(hint: StageRuntimeHint): Unit = {
    if (enabled) {
      hints.put(hint.key, hint)
    }
  }

  def handleHintRequest(msg: RapidsAutotuneHintRequestMsg): RapidsAutotuneHintResponseMsg = {
    val hint = if (enabled) {
      Option(hints.get(msg.key)).filterNot(_.isExpired(System.nanoTime()))
    } else {
      None
    }
    RapidsAutotuneHintResponseMsg(msg.key, hint)
  }

  def handleHintApplied(msg: RapidsAutotuneHintAppliedMsg): Unit = {
    if (!enabled || sparkContext == null) {
      return
    }
    TrampolineUtil.postEvent(sparkContext, SparkRapidsAutotuneHintAppliedEvent(
      executorId = msg.executorId,
      executionId = msg.key.executionId,
      stageId = msg.key.stageId,
      stageAttemptId = msg.key.stageAttemptId,
      taskAttemptId = msg.taskAttemptId,
      partitionId = msg.partitionId,
      hintVersion = msg.hintVersion,
      hasHint = msg.hasHint))
  }
}

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
      cachedHint: AutotuneCachedHint): Unit = {
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
        hasHint = cachedHint.hasHint))
    } catch {
      case NonFatal(e) if failOpen =>
        logWarning("Failed to report RAPIDS graph autotune applied hint; continuing", e)
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

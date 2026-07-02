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

// The serialized autotune wire types -- AutotuneStageKey, ScanRuntimeHint,
// StageRuntimeHint and the RapidsAutotuneHint*Msg messages -- live in the sql-plugin-api module
// (see GraphAutotuneMessages) so they are packaged at the jar root / base classloader and Spark's
// RpcEnv can deserialize them. The types in this package use them but are not themselves sent over
// the wire.

/**
 * A [[StageRuntimeHint]] paired with whether the driver actually had a hint for the stage.
 *
 * `hasHint = false` is the "no hint published / fail-open" state: callers must treat it as "use
 * current RAPIDS behavior". An empty-but-present hint (`hasHint = true`, no-op fields) is distinct:
 * it means the driver explicitly published a no-op (e.g. OBSERVE mode).
 */
case class AutotuneCachedHint(hint: StageRuntimeHint, hasHint: Boolean) {
  def version: Long = hint.version

  def isExpired(nowNanos: Long): Boolean = hasHint && hint.isExpired(nowNanos)
}

object AutotuneCachedHint {
  def empty(key: AutotuneStageKey): AutotuneCachedHint =
    AutotuneCachedHint(StageRuntimeHint.empty(key), hasHint = false)
}

/**
 * Executor task-local store for the hint that applies to the currently running task.
 *
 * The executor plugin sets this on task start and clears it on completion (see
 * `RapidsExecutorPlugin.onTaskStart`), so reader/operator code on the task thread can read the
 * applicable hint without threading it through every call site. The `current*Hint` accessors only
 * expose a hint when one was actually published (`hasHint`), so consumers fall back to default
 * behavior otherwise.
 */
object RapidsAutotuneTaskHints {
  private val currentHint = new ThreadLocal[AutotuneCachedHint]()

  def setCurrentHint(hint: AutotuneCachedHint): Unit = currentHint.set(hint)

  def clearCurrentHint(): Unit = currentHint.remove()

  def currentScanHint: Option[ScanRuntimeHint] =
    Option(currentHint.get()).filter(_.hasHint).map(_.hint.scan)

  def currentShuffleHint: Option[ShuffleRuntimeHint] =
    Option(currentHint.get()).filter(_.hasHint).map(_.hint.shuffle)

  def currentBatchHint: Option[BatchRuntimeHint] =
    Option(currentHint.get()).filter(_.hasHint).map(_.hint.batch)
}

/** Task-local resolution of optimizer-owned batch/coalesce targets. */
object BatchRuntimeHints {
  def effectiveTargetBatchBytes(
      staticTargetBytes: Long,
      hint: Option[BatchRuntimeHint]): Long = {
    hint.filter(h => h.targetBatchBytes > 0L && h.maxBatchBytes > 0L)
      .map(h => math.max(1L, math.min(h.targetBatchBytes, h.maxBatchBytes)))
      .getOrElse(staticTargetBytes)
  }

  def effectiveShuffleCoalesceTargetBytes(
      staticTargetBytes: Long,
      shuffleHint: Option[ShuffleRuntimeHint],
      batchHint: Option[BatchRuntimeHint]): Long = {
    val requested = shuffleHint.map(_.coalesceTargetBytes).filter(_ > 0L)
      .orElse(batchHint.map(_.targetBatchBytes).filter(_ > 0L))
    requested.map { target =>
      val ceiling = batchHint.map(_.maxBatchBytes).filter(_ > 0L).getOrElse(Long.MaxValue)
      math.max(1L, math.min(target, ceiling))
    }.getOrElse(staticTargetBytes)
  }

  def effectiveSplitUntilSize(
      staticSplitUntilSize: Long,
      hint: Option[BatchRuntimeHint]): Long = {
    hint.filter(h => h.splitUntilSize > 0L && h.maxBatchBytes > 0L)
      .map(h => math.max(GpuDeviceManager.MIN_SPLIT_UNTIL_SIZE,
        math.min(h.splitUntilSize, h.maxBatchBytes)))
      .getOrElse(staticSplitUntilSize)
  }
}

/**
 * Executor-side resolution of the graph autotune shuffle-read hint.
 *
 * The multithreaded shuffle reader sizes its per-reader `BytesInFlightLimiter` at construction, on
 * the task thread where [[RapidsAutotuneTaskHints]] is live. This turns the task-local
 * [[ShuffleRuntimeHint]] into an effective limiter bound. A hint can only tighten the static
 * `spark.rapids.shuffle.multiThreaded.maxBytesInFlight` cap. Fail-open: when disabled, with no
 * hint, or with the empty-hint sentinel (`Long.MaxValue`), the static cap is returned unchanged. A
 * non-positive `maxReadyBytes` is also ignored. The executor applies the cap even if a bad or
 * stale driver hint asks for more, so the static host-memory envelope is a hard backstop.
 */
object ShuffleReadHints {
  def effectivePrefetchWindow(hint: Option[ShuffleRuntimeHint]): Int =
    hint.map(_.prefetchWindow).filter(_ > 0).getOrElse(Integer.MAX_VALUE)

  def blocksToDrain(prefetchWindow: Int, outstandingBlocks: Long, readyBlocks: Int): Int = {
    val available = math.max(1L,
      math.max(1, prefetchWindow).toLong - math.max(0L, outstandingBlocks))
      .min(Integer.MAX_VALUE.toLong).toInt
    math.min(math.max(readyBlocks, 1), available)
  }

  def effectiveMaxBytesInFlight(
      staticMaxBytesInFlight: Long,
      hint: Option[ShuffleRuntimeHint],
      enabled: Boolean): Long = {
    if (!enabled) {
      staticMaxBytesInFlight
    } else {
      hint.map(_.maxReadyBytes).filter(b => b > 0L && b != Long.MaxValue) match {
        case Some(hintBound) =>
          math.max(1L, math.min(staticMaxBytesInFlight, hintBound))
        case None => staticMaxBytesInFlight
      }
    }
  }
}

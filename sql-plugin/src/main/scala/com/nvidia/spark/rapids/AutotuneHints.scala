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

// The serialized autotune wire types -- AutotuneStageKey, ScanRuntimeHint, GpuRuntimeHint,
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

  def currentGpuHint: Option[GpuRuntimeHint] =
    Option(currentHint.get()).filter(_.hasHint).map(_.hint.gpu)

  def currentShuffleHint: Option[ShuffleRuntimeHint] =
    Option(currentHint.get()).filter(_.hasHint).map(_.hint.shuffle)
}

/**
 * Executor-side resolution of the graph autotune shuffle-read hint.
 *
 * The multithreaded shuffle reader sizes its per-reader `BytesInFlightLimiter` at construction, on
 * the task thread where [[RapidsAutotuneTaskHints]] is live. This turns the task-local
 * [[ShuffleRuntimeHint]] into an effective limiter bound that can only ever *tighten* the static
 * `spark.rapids.shuffle.multiThreaded.maxBytesInFlight` cap, never loosen it. Fail-open: when
 * disabled or with no hint the static cap is returned unchanged. A non-positive `maxReadyBytes`
 * is ignored (static cap); the empty-hint sentinel (`Long.MaxValue`) is kept but `min` with the
 * static cap leaves it unchanged -- so by default behavior matches current RAPIDS. Tightening only
 * happens when a hint carries a positive `maxReadyBytes` below the static cap.
 */
object ShuffleReadHints {
  def effectiveMaxBytesInFlight(
      staticMaxBytesInFlight: Long,
      hint: Option[ShuffleRuntimeHint],
      enabled: Boolean): Long = {
    if (!enabled) {
      staticMaxBytesInFlight
    } else {
      hint.map(_.maxReadyBytes).filter(_ > 0L) match {
        case Some(hintBound) => math.max(1L, math.min(staticMaxBytesInFlight, hintBound))
        case None => staticMaxBytesInFlight
      }
    }
  }
}

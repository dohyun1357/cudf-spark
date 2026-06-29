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

/**
 * Wire types for the graph-wide runtime autotuner.
 *
 * These classes are exchanged between the driver and executor autotune endpoints over Spark's
 * `RpcEnv` (`RapidsAutotuneExecutorEndpoint` <-> the driver plugin handler). Spark's
 * `NettyRpcEnv` deserializes RPC payloads with the base application classloader, NOT the RAPIDS
 * shim/parallel-world classloader. They must therefore live in `sql-plugin-api`, which is
 * packaged at the jar root (base classloader) rather than under a `spark-shared/` parallel
 * world -- exactly like the shuffle heartbeat messages in `RapidsShuffleHeartbeatHandler`.
 *
 * Do NOT move these (or any type reachable from their fields) back into `sql-plugin`: that
 * lands them in the parallel world, and the receiving endpoint fails to resolve them with
 * `java.lang.ClassNotFoundException`, which kills the executor on every autotune hint exchange.
 *
 * The autotune endpoints, policies, caches and admission controller that *use* these messages
 * stay in `sql-plugin` (`GraphAutotuneRuntime`); only the serialized wire closure lives here.
 */

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

case class GpuRuntimeHint(maxConcurrentTasks: Int)

object GpuRuntimeHint {
  val empty: GpuRuntimeHint = GpuRuntimeHint(maxConcurrentTasks = 0)
}

case class ShuffleRuntimeHint(
    prefetchWindow: Int,
    maxReadyBytes: Long,
    coalesceTargetBytes: Long)

object ShuffleRuntimeHint {
  val empty: ShuffleRuntimeHint = ShuffleRuntimeHint(
    prefetchWindow = 0,
    maxReadyBytes = Long.MaxValue,
    coalesceTargetBytes = 0L)
}

case class BatchRuntimeHint(
    targetBatchBytes: Long,
    maxBatchBytes: Long,
    splitUntilSize: Long)

object BatchRuntimeHint {
  val empty: BatchRuntimeHint = BatchRuntimeHint(
    targetBatchBytes = 0L,
    maxBatchBytes = Long.MaxValue,
    splitUntilSize = 0L)
}

case class StageRuntimeHint(
    executionId: Long,
    stageId: Int,
    stageAttemptId: Int,
    version: Long,
    scan: ScanRuntimeHint,
    gpu: GpuRuntimeHint,
    shuffle: ShuffleRuntimeHint = ShuffleRuntimeHint.empty,
    batch: BatchRuntimeHint = BatchRuntimeHint.empty,
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
    gpu = GpuRuntimeHint.empty,
    shuffle = ShuffleRuntimeHint.empty,
    batch = BatchRuntimeHint.empty,
    expiresAtNanos = Long.MaxValue)
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
    hasHint: Boolean,
    scan: ScanRuntimeHint,
    gpu: GpuRuntimeHint,
    shuffle: ShuffleRuntimeHint,
    batch: BatchRuntimeHint,
    gpuAppliedMaxConcurrentTasks: Int)

/**
 * Executor -> driver runtime observation reported (fire-and-forget) at task completion. Feeds the
 * driver-side closed-loop model. Advisory only: dropped/lost messages never affect correctness.
 * Carries the pressure signals available at task end; more fields can be added as the model needs
 * them (this is the Phase 6 "Observation and Update" channel).
 *
 * The metric values are best-effort snapshots read from `GpuTaskMetrics` at completion. Task
 * completion listeners have no guaranteed ordering, so a value may occasionally be approximate
 * (e.g. a final semaphore-hold interval not yet folded in). That is acceptable: observations are
 * advisory model input, never used for correctness.
 */
case class RapidsAutotuneObservationMsg(
    executorId: String,
    key: AutotuneStageKey,
    taskAttemptId: Long,
    partitionId: Int,
    hintVersion: Long,
    gpuSemaphoreWaitNanos: Long,
    gpuHoldingNanos: Long,
    hostMemoryBytes: Long,
    // Total bytes spilled (host + disk) by the task. A spill pressure signal for the closed-loop
    // model: when positive, the model reduces a knob (read window / GPU concurrency) to relieve
    // memory pressure. Defaulted so older positional constructions keep compiling.
    spillBytes: Long = 0L)

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

import java.io.IOException
import java.util.{Collections, Map => JMap}

import scala.collection.mutable

import com.codahale.metrics.MetricRegistry
import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.SparkConf
import org.apache.spark.api.plugin.PluginContext
import org.apache.spark.resource.ResourceInformation

class GraphAutotuneRuntimeSuite extends AnyFunSuite {
  private val key = AutotuneStageKey(executionId = 11L, stageId = 3, stageAttemptId = 1)

  private class FailingPluginContext extends PluginContext {
    override def metricRegistry(): MetricRegistry = new MetricRegistry()

    override def conf(): SparkConf = new SparkConf(false)

    override def executorID(): String = "exec-1"

    override def hostname(): String = "localhost"

    override def resources(): JMap[String, ResourceInformation] = Collections.emptyMap()

    override def send(message: Any): Unit = throw new IOException("send failed")

    override def ask(message: Any): AnyRef = throw new IOException("ask failed")
  }

  private class CapturingPluginContext extends PluginContext {
    var lastSent: Any = _

    override def metricRegistry(): MetricRegistry = new MetricRegistry()

    override def conf(): SparkConf = new SparkConf(false)

    override def executorID(): String = "exec-7"

    override def hostname(): String = "localhost"

    override def resources(): JMap[String, ResourceInformation] = Collections.emptyMap()

    override def send(message: Any): Unit = lastSent = message

    override def ask(message: Any): AnyRef = RapidsAutotuneHintResponseMsg(key, None)
  }

  test("stage runtime hint preserves cache key and expiration") {
    val hint = StageRuntimeHint(
      executionId = key.executionId,
      stageId = key.stageId,
      stageAttemptId = key.stageAttemptId,
      version = 7L,
      scan = ScanRuntimeHint(eagerPrefetch = true, minReadWindow = 1,
        maxReadWindow = 4, maxReadyBytes = 1024L),
      gpu = GpuRuntimeHint(maxConcurrentTasks = 2),
      expiresAtNanos = 10L)

    assert(hint.key == key)
    assert(!hint.isExpired(9L))
    assert(hint.isExpired(10L))
  }

  test("shuffle and batch runtime hints default to empty no-op") {
    assert(ShuffleRuntimeHint.empty.prefetchWindow == 0)
    assert(ShuffleRuntimeHint.empty.maxReadyBytes == Long.MaxValue)
    assert(ShuffleRuntimeHint.empty.coalesceTargetBytes == 0L)
    assert(BatchRuntimeHint.empty.targetBatchBytes == 0L)
    assert(BatchRuntimeHint.empty.maxBatchBytes == Long.MaxValue)
    assert(BatchRuntimeHint.empty.splitUntilSize == 0L)

    val empty = StageRuntimeHint.empty(key)
    assert(empty.shuffle == ShuffleRuntimeHint.empty)
    assert(empty.batch == BatchRuntimeHint.empty)
  }

  test("hint cache memoizes no-hint responses by stage key") {
    var fetches = 0
    val cache = new AutotuneHintCache(stageKey => {
      assert(stageKey == key)
      fetches += 1
      AutotuneCachedHint.empty(stageKey)
    })

    val first = cache.get(key)
    val second = cache.get(key)

    assert(!first.hasHint)
    assert(first.version == 0L)
    assert(first eq second)
    assert(fetches == 1)
  }

  test("hint cache refreshes expired hints") {
    var version = 0L
    val cache = new AutotuneHintCache(stageKey => {
      version += 1
      AutotuneCachedHint(StageRuntimeHint(
        executionId = stageKey.executionId,
        stageId = stageKey.stageId,
        stageAttemptId = stageKey.stageAttemptId,
        version = version,
        scan = ScanRuntimeHint.empty,
        gpu = GpuRuntimeHint.empty,
        expiresAtNanos = 0L), hasHint = true)
    })

    assert(cache.get(key).version == 1L)
    assert(cache.get(key).version == 2L)
  }

  test("executor endpoint fails open when driver hint RPCs fail") {
    val conf = new RapidsConf(Map(RapidsConf.AUTOTUNE_FAIL_OPEN.key -> "true"))
    val endpoint = new RapidsAutotuneExecutorEndpoint(new FailingPluginContext, conf)

    val hint = endpoint.hintFor(key)
    assert(!hint.hasHint)
    assert(hint.version == 0L)
    endpoint.recordAppliedHint(
      key,
      taskAttemptId = 22L,
      partitionId = 1,
      hint,
      gpuAppliedMaxConcurrentTasks = 0)
  }

  test("task hint state exposes only applied hints") {
    val scanHint = ScanRuntimeHint(
      eagerPrefetch = true,
      minReadWindow = 1,
      maxReadWindow = 4,
      maxReadyBytes = 1024L)
    val gpuHint = GpuRuntimeHint(maxConcurrentTasks = 2)
    val shuffleHint = ShuffleRuntimeHint(
      prefetchWindow = 2, maxReadyBytes = 4096L, coalesceTargetBytes = 1024L)
    val hint = AutotuneCachedHint(StageRuntimeHint(
      executionId = key.executionId,
      stageId = key.stageId,
      stageAttemptId = key.stageAttemptId,
      version = 1L,
      scan = scanHint,
      gpu = gpuHint,
      shuffle = shuffleHint,
      expiresAtNanos = Long.MaxValue), hasHint = true)

    try {
      RapidsAutotuneTaskHints.clearCurrentHint()
      assert(RapidsAutotuneTaskHints.currentScanHint.isEmpty)
      assert(RapidsAutotuneTaskHints.currentGpuHint.isEmpty)
      assert(RapidsAutotuneTaskHints.currentShuffleHint.isEmpty)
      RapidsAutotuneTaskHints.setCurrentHint(AutotuneCachedHint.empty(key))
      assert(RapidsAutotuneTaskHints.currentScanHint.isEmpty)
      assert(RapidsAutotuneTaskHints.currentGpuHint.isEmpty)
      assert(RapidsAutotuneTaskHints.currentShuffleHint.isEmpty)
      RapidsAutotuneTaskHints.setCurrentHint(hint)
      assert(RapidsAutotuneTaskHints.currentScanHint.contains(scanHint))
      assert(RapidsAutotuneTaskHints.currentGpuHint.contains(gpuHint))
      assert(RapidsAutotuneTaskHints.currentShuffleHint.contains(shuffleHint))
    } finally {
      RapidsAutotuneTaskHints.clearCurrentHint()
    }
    assert(RapidsAutotuneTaskHints.currentScanHint.isEmpty)
    assert(RapidsAutotuneTaskHints.currentGpuHint.isEmpty)
    assert(RapidsAutotuneTaskHints.currentShuffleHint.isEmpty)
  }

  test("shuffle read actuator only tightens the static bytes-in-flight cap") {
    val staticCap = 128L * 1024 * 1024
    def hint(b: Long): Option[ShuffleRuntimeHint] =
      Some(ShuffleRuntimeHint(prefetchWindow = 0, maxReadyBytes = b, coalesceTargetBytes = 0L))

    // Disabled -> static unchanged, even with a tightening hint present.
    assert(ShuffleReadHints.effectiveMaxBytesInFlight(staticCap, hint(64L), enabled = false) ==
      staticCap)
    // Enabled, no hint -> static (fail-open).
    assert(ShuffleReadHints.effectiveMaxBytesInFlight(staticCap, None, enabled = true) == staticCap)
    // Enabled, hint below static -> tightened to the hint bound.
    assert(ShuffleReadHints.effectiveMaxBytesInFlight(
      staticCap, hint(64L * 1024 * 1024), enabled = true) == 64L * 1024 * 1024)
    // Enabled, hint above static -> never loosens past the static cap.
    assert(ShuffleReadHints.effectiveMaxBytesInFlight(
      staticCap, hint(staticCap * 4), enabled = true) == staticCap)
    // Enabled, empty/no-op hint (Long.MaxValue sentinel) -> static.
    assert(ShuffleReadHints.effectiveMaxBytesInFlight(
      staticCap, Some(ShuffleRuntimeHint.empty), enabled = true) == staticCap)
    // Enabled, zero/negative hint -> static (never a 0 or negative cap).
    assert(ShuffleReadHints.effectiveMaxBytesInFlight(staticCap, hint(0L), enabled = true) ==
      staticCap)
    assert(ShuffleReadHints.effectiveMaxBytesInFlight(staticCap, hint(-5L), enabled = true) ==
      staticCap)
  }

  test("stage shape detects gpu scan prefetch candidates from RDD scopes") {
    val candidate = AutotuneStageShape.fromRddScopeNames(
      Seq("GpuScan parquet ", "GpuFilter", "GpuHashAggregate", "GpuColumnarExchange"),
      numTasks = 50)
    val scanOnly = AutotuneStageShape.fromRddScopeNames(
      Seq("GpuScan parquet ", "GpuFilter"),
      numTasks = 50)
    val consumerOnly = AutotuneStageShape.fromRddScopeNames(
      Seq("GpuHashAggregate", "GpuColumnarExchange"),
      numTasks = 50)

    assert(candidate.isScanPrefetchCandidate)
    assert(!scanOnly.isScanPrefetchCandidate)
    assert(!consumerOnly.isScanPrefetchCandidate)
  }

  test("stage shape detects shuffle-bearing stages from RDD scopes") {
    val exchange = AutotuneStageShape.fromRddScopeNames(
      Seq("GpuScan parquet ", "GpuFilter", "GpuColumnarExchange"), numTasks = 50)
    val coalesce = AutotuneStageShape.fromRddScopeNames(
      Seq("GpuShuffleCoalesce", "GpuHashAggregate"), numTasks = 50)
    val scanOnly = AutotuneStageShape.fromRddScopeNames(
      Seq("GpuScan parquet ", "GpuFilter"), numTasks = 50)
    val noTasks = AutotuneStageShape.fromRddScopeNames(
      Seq("GpuColumnarExchange"), numTasks = 0)

    assert(exchange.hasShuffle && exchange.isShuffleStage && exchange.hasGpuWork)
    assert(coalesce.hasShuffle && coalesce.isShuffleStage)
    assert(!scanOnly.hasShuffle && !scanOnly.isShuffleStage)
    assert(scanOnly.hasGpuWork)
    assert(noTasks.hasShuffle && !noTasks.isShuffleStage && !noTasks.hasGpuWork)
  }

  test("graph scan hint policy emits bounded scan hints only in graph mode") {
    val graphConf = new RapidsConf(Map(
      RapidsConf.AUTOTUNE_GRAPH_ENABLED.key -> "true",
      RapidsConf.AUTOTUNE_GRAPH_MODE.key -> AutotuneGraphMode.GRAPH.toString,
      RapidsConf.AUTOTUNE_SCAN_MAX_READ_WINDOW.key -> "8",
      RapidsConf.SCAN_PREFETCH_MAX_PARALLELISM.key -> "4",
      RapidsConf.AUTOTUNE_SCAN_MAX_READY_BYTES.key -> "1024"))
    val localConf = new RapidsConf(Map(
      RapidsConf.AUTOTUNE_GRAPH_ENABLED.key -> "true",
      RapidsConf.AUTOTUNE_GRAPH_MODE.key -> AutotuneGraphMode.LOCAL.toString,
      RapidsConf.AUTOTUNE_SCAN_MAX_READ_WINDOW.key -> "8",
      RapidsConf.SCAN_PREFETCH_MAX_PARALLELISM.key -> "4",
      RapidsConf.AUTOTUNE_SCAN_MAX_READY_BYTES.key -> "1024"))
    val candidate =
      AutotuneStageShape(hasGpuScan = true, hasGpuPrefetchConsumer = true, numTasks = 50)

    val graphHint = GraphScanHintPolicy.fromConf(graphConf).scanHintFor(candidate)
    assert(graphHint.eagerPrefetch)
    assert(graphHint.minReadWindow == 1)
    assert(graphHint.maxReadWindow == 4)
    assert(graphHint.maxReadyBytes == 1024L)
    assert(GraphScanHintPolicy.fromConf(localConf).scanHintFor(candidate) == ScanRuntimeHint.empty)
  }

  test("graph GPU hint policy emits positive caps only in graph mode") {
    val graphConf = new RapidsConf(Map(
      RapidsConf.AUTOTUNE_GRAPH_ENABLED.key -> "true",
      RapidsConf.AUTOTUNE_GRAPH_MODE.key -> AutotuneGraphMode.GRAPH.toString,
      RapidsConf.AUTOTUNE_GPU_MAX_CONCURRENT_TASKS.key -> "4"))
    val localConf = new RapidsConf(Map(
      RapidsConf.AUTOTUNE_GRAPH_ENABLED.key -> "true",
      RapidsConf.AUTOTUNE_GRAPH_MODE.key -> AutotuneGraphMode.LOCAL.toString,
      RapidsConf.AUTOTUNE_GPU_MAX_CONCURRENT_TASKS.key -> "4"))
    val defaultConf = new RapidsConf(Map(
      RapidsConf.AUTOTUNE_GRAPH_ENABLED.key -> "true",
      RapidsConf.AUTOTUNE_GRAPH_MODE.key -> AutotuneGraphMode.GRAPH.toString))
    val stageShape =
      AutotuneStageShape(hasGpuScan = false, hasGpuPrefetchConsumer = false, numTasks = 50)

    assert(GraphGpuHintPolicy.fromConf(graphConf).gpuHintFor(stageShape) ==
      GpuRuntimeHint(maxConcurrentTasks = 4))
    assert(GraphGpuHintPolicy.fromConf(localConf).gpuHintFor(stageShape) == GpuRuntimeHint.empty)
    assert(GraphGpuHintPolicy.fromConf(defaultConf).gpuHintFor(stageShape) == GpuRuntimeHint.empty)
    assert(GraphGpuHintPolicy.fromConf(graphConf).gpuHintFor(stageShape.copy(numTasks = 0)) ==
      GpuRuntimeHint.empty)
  }

  test("graph shuffle hint policy emits bounded shuffle hints only when enabled in graph mode") {
    val graphConf = new RapidsConf(Map(
      RapidsConf.AUTOTUNE_GRAPH_ENABLED.key -> "true",
      RapidsConf.AUTOTUNE_GRAPH_MODE.key -> AutotuneGraphMode.GRAPH.toString,
      RapidsConf.AUTOTUNE_SHUFFLE_ENABLED.key -> "true",
      RapidsConf.AUTOTUNE_SHUFFLE_MAX_PREFETCH_WINDOW.key -> "6",
      RapidsConf.AUTOTUNE_SHUFFLE_MAX_READY_BYTES.key -> "2048",
      RapidsConf.AUTOTUNE_SHUFFLE_COALESCE_TARGET_BYTES.key -> "1024"))
    val disabledConf = new RapidsConf(Map(
      RapidsConf.AUTOTUNE_GRAPH_ENABLED.key -> "true",
      RapidsConf.AUTOTUNE_GRAPH_MODE.key -> AutotuneGraphMode.GRAPH.toString,
      RapidsConf.AUTOTUNE_SHUFFLE_MAX_PREFETCH_WINDOW.key -> "6"))
    val localConf = new RapidsConf(Map(
      RapidsConf.AUTOTUNE_GRAPH_ENABLED.key -> "true",
      RapidsConf.AUTOTUNE_GRAPH_MODE.key -> AutotuneGraphMode.LOCAL.toString,
      RapidsConf.AUTOTUNE_SHUFFLE_ENABLED.key -> "true"))
    val shuffleStage = AutotuneStageShape(
      hasGpuScan = false, hasGpuPrefetchConsumer = false, numTasks = 50, hasShuffle = true)
    val nonShuffleStage =
      AutotuneStageShape(hasGpuScan = true, hasGpuPrefetchConsumer = true, numTasks = 50)

    assert(GraphShuffleHintPolicy.fromConf(graphConf).shuffleHintFor(shuffleStage) ==
      ShuffleRuntimeHint(prefetchWindow = 6, maxReadyBytes = 2048L, coalesceTargetBytes = 1024L))
    // A non-shuffle stage gets an empty hint even when the policy is enabled.
    assert(GraphShuffleHintPolicy.fromConf(graphConf).shuffleHintFor(nonShuffleStage) ==
      ShuffleRuntimeHint.empty)
    // Default-off enable flag and non-GRAPH mode both keep the hint empty.
    assert(GraphShuffleHintPolicy.fromConf(disabledConf).shuffleHintFor(shuffleStage) ==
      ShuffleRuntimeHint.empty)
    assert(GraphShuffleHintPolicy.fromConf(localConf).shuffleHintFor(shuffleStage) ==
      ShuffleRuntimeHint.empty)
  }

  test("graph batch hint policy emits bounded batch hints only when enabled in graph mode") {
    val graphConf = new RapidsConf(Map(
      RapidsConf.AUTOTUNE_GRAPH_ENABLED.key -> "true",
      RapidsConf.AUTOTUNE_GRAPH_MODE.key -> AutotuneGraphMode.GRAPH.toString,
      RapidsConf.AUTOTUNE_BATCH_ENABLED.key -> "true",
      RapidsConf.AUTOTUNE_BATCH_TARGET_BYTES.key -> "4096",
      RapidsConf.AUTOTUNE_BATCH_MAX_BYTES.key -> "8192",
      RapidsConf.AUTOTUNE_BATCH_SPLIT_UNTIL_SIZE.key -> "512"))
    val disabledConf = new RapidsConf(Map(
      RapidsConf.AUTOTUNE_GRAPH_ENABLED.key -> "true",
      RapidsConf.AUTOTUNE_GRAPH_MODE.key -> AutotuneGraphMode.GRAPH.toString,
      RapidsConf.AUTOTUNE_BATCH_TARGET_BYTES.key -> "4096"))
    val gpuStage =
      AutotuneStageShape(hasGpuScan = true, hasGpuPrefetchConsumer = false, numTasks = 50)
    val noGpuStage =
      AutotuneStageShape(hasGpuScan = false, hasGpuPrefetchConsumer = false, numTasks = 50)

    assert(GraphBatchHintPolicy.fromConf(graphConf).batchHintFor(gpuStage) ==
      BatchRuntimeHint(targetBatchBytes = 4096L, maxBatchBytes = 8192L, splitUntilSize = 512L))
    // A stage doing no GPU work gets an empty hint even when enabled.
    assert(GraphBatchHintPolicy.fromConf(graphConf).batchHintFor(noGpuStage) ==
      BatchRuntimeHint.empty)
    assert(GraphBatchHintPolicy.fromConf(disabledConf).batchHintFor(gpuStage) ==
      BatchRuntimeHint.empty)
  }

  test("GPU admission controller applies the minimum active positive stage cap") {
    val appliedLimits = mutable.ArrayBuffer.empty[Int]
    val controller = new RapidsAutotuneGpuAdmissionController(limit => {
      appliedLimits += limit
      limit
    })
    val secondKey = key.copy(stageId = 4, stageAttemptId = 0)
    val hint4 = AutotuneCachedHint(StageRuntimeHint.empty(key).copy(
      version = 1L,
      gpu = GpuRuntimeHint(maxConcurrentTasks = 4)), hasHint = true)
    val hint2 = AutotuneCachedHint(StageRuntimeHint.empty(secondKey).copy(
      version = 2L,
      gpu = GpuRuntimeHint(maxConcurrentTasks = 2)), hasHint = true)

    assert(controller.taskStarted(key, 100L, hint4) == 4)        // key active {100}
    assert(controller.taskStarted(key, 101L, hint4) == 4)        // key active {100,101}
    assert(controller.taskStarted(secondKey, 200L, hint2) == 2)  // min(4,2)
    assert(controller.taskCompleted(secondKey, 200L) == 4)       // secondKey drained -> 4
    assert(controller.taskCompleted(key, 100L) == 4)             // key still has {101}
    assert(controller.taskCompleted(key, 101L) == 0)             // key drained -> 0
    assert(controller.reset() == 0)

    assert(appliedLimits == Seq(4, 4, 2, 4, 4, 0, 0))
  }

  test("driver endpoint publishes stable positive default no-op hints") {
    val conf = new RapidsConf(Map(RapidsConf.AUTOTUNE_GRAPH_ENABLED.key -> "true"))
    RapidsAutotuneDriverEndpoint.init(null, conf)
    try {
      val first = RapidsAutotuneDriverEndpoint.publishDefaultNoopHint(key)
      val again = RapidsAutotuneDriverEndpoint.publishDefaultNoopHint(key)
      val secondKey = key.copy(stageId = 4)
      val second = RapidsAutotuneDriverEndpoint.publishDefaultNoopHint(secondKey)

      assert(first.version == 1L)
      assert(first == again)
      assert(second.version == 2L)
      assert(!first.scan.eagerPrefetch)
      assert(first.scan.maxReadWindow == 0)
      assert(first.gpu == GpuRuntimeHint.empty)
      assert(first.shuffle == ShuffleRuntimeHint.empty)
      assert(first.batch == BatchRuntimeHint.empty)

      val response = RapidsAutotuneDriverEndpoint.handleHintRequest(
        RapidsAutotuneHintRequestMsg("exec-1", key))
      assert(response.hint.contains(first))
    } finally {
      RapidsAutotuneDriverEndpoint.shutdown()
    }
  }

  test("stage hint listener publishes default hints for stage keys") {
    val conf = new RapidsConf(Map(RapidsConf.AUTOTUNE_GRAPH_ENABLED.key -> "true"))
    RapidsAutotuneDriverEndpoint.init(null, conf)
    try {
      val listener = new RapidsAutotuneStageHintListener(conf)
      listener.publishDefaultHint(key.executionId, key.stageId, key.stageAttemptId)
      listener.publishDefaultHint(key.executionId, 4, 0)

      val first = RapidsAutotuneDriverEndpoint.handleHintRequest(
        RapidsAutotuneHintRequestMsg("exec-1", key)).hint
      val second = RapidsAutotuneDriverEndpoint.handleHintRequest(
        RapidsAutotuneHintRequestMsg("exec-1", key.copy(stageId = 4, stageAttemptId = 0))).hint

      assert(first.exists(_.version == 1L))
      assert(second.exists(_.version == 2L))
      assert(first.exists(_.scan == ScanRuntimeHint.empty))
      assert(second.exists(_.scan == ScanRuntimeHint.empty))
      assert(first.exists(_.gpu == GpuRuntimeHint.empty))
      assert(second.exists(_.gpu == GpuRuntimeHint.empty))
      assert(first.exists(_.shuffle == ShuffleRuntimeHint.empty))
      assert(second.exists(_.shuffle == ShuffleRuntimeHint.empty))
      assert(first.exists(_.batch == BatchRuntimeHint.empty))
      assert(second.exists(_.batch == BatchRuntimeHint.empty))
    } finally {
      RapidsAutotuneDriverEndpoint.shutdown()
    }
  }

  test("stage hint listener publishes graph scan hints for candidate stages") {
    val conf = new RapidsConf(Map(
      RapidsConf.AUTOTUNE_GRAPH_ENABLED.key -> "true",
      RapidsConf.AUTOTUNE_GRAPH_MODE.key -> AutotuneGraphMode.GRAPH.toString,
      RapidsConf.AUTOTUNE_SCAN_MAX_READ_WINDOW.key -> "8",
      RapidsConf.SCAN_PREFETCH_MAX_PARALLELISM.key -> "4",
      RapidsConf.AUTOTUNE_SCAN_MAX_READY_BYTES.key -> "1024",
      RapidsConf.AUTOTUNE_GPU_MAX_CONCURRENT_TASKS.key -> "4"))
    RapidsAutotuneDriverEndpoint.init(null, conf)
    try {
      val listener = new RapidsAutotuneStageHintListener(conf)
      val candidate =
        AutotuneStageShape(hasGpuScan = true, hasGpuPrefetchConsumer = true, numTasks = 50)
      val hint = listener.publishHintForStage(
        key.executionId, key.stageId, key.stageAttemptId, candidate)

      assert(hint.version == 1L)
      assert(hint.scan == ScanRuntimeHint(
        eagerPrefetch = true,
        minReadWindow = 1,
        maxReadWindow = 4,
        maxReadyBytes = 1024L))
      assert(hint.gpu == GpuRuntimeHint(maxConcurrentTasks = 4))

      val response = RapidsAutotuneDriverEndpoint.handleHintRequest(
        RapidsAutotuneHintRequestMsg("exec-1", key))
      assert(response.hint.contains(hint))
    } finally {
      RapidsAutotuneDriverEndpoint.shutdown()
    }
  }

  test("GPU admission controller no-ops on unknown-key completion and clears on reset") {
    val appliedLimits = mutable.ArrayBuffer.empty[Int]
    val controller = new RapidsAutotuneGpuAdmissionController(limit => {
      appliedLimits += limit
      limit
    })

    // Completing a key that never started is a no-op: returns the current limit, applies nothing.
    assert(controller.taskCompleted(key, 1L) == 0)
    // A task with no positive GPU cap does not register and applies nothing.
    val noCap = AutotuneCachedHint(StageRuntimeHint.empty(key).copy(version = 1L), hasHint = true)
    assert(controller.taskStarted(key, 1L, noCap) == 0)
    assert(appliedLimits.isEmpty)

    // Reset while a positive-cap task is active drops the runtime cap back to 0, and the now
    // forgotten task's completion is a no-op.
    val capped = AutotuneCachedHint(StageRuntimeHint.empty(key).copy(
      version = 2L, gpu = GpuRuntimeHint(maxConcurrentTasks = 3)), hasHint = true)
    assert(controller.taskStarted(key, 2L, capped) == 3)
    assert(controller.reset() == 0)
    assert(controller.taskCompleted(key, 2L) == 0)

    assert(appliedLimits == Seq(3, 0))
  }

  test("GPU admission: a no-hint task's completion does not release a capped sibling's entry") {
    // Regression for the fetch-TTL asymmetry: task A starts under no hint (cap 0, unregistered),
    // task B starts under cap 3 (registers). A completing must NOT remove B's still-active entry.
    val appliedLimits = mutable.ArrayBuffer.empty[Int]
    val controller = new RapidsAutotuneGpuAdmissionController(limit => {
      appliedLimits += limit
      limit
    })
    val noHint = AutotuneCachedHint.empty(key)
    val capped = AutotuneCachedHint(StageRuntimeHint.empty(key).copy(
      version = 2L, gpu = GpuRuntimeHint(maxConcurrentTasks = 3)), hasHint = true)

    assert(controller.taskStarted(key, 100L, noHint) == 0) // A: unregistered, applies nothing
    assert(controller.taskStarted(key, 200L, capped) == 3) // B: registers, cap 3
    assert(controller.taskCompleted(key, 100L) == 3)       // A completes -> B's entry intact
    assert(controller.taskCompleted(key, 200L) == 0)       // B completes -> entry drained

    // A's unregistered completion does not re-apply a limit; only B's start (3) and drain (0) do.
    assert(appliedLimits == Seq(3, 0))
  }

  test("hint cache put overrides fetch and refreshes once the pushed hint expires") {
    var fetches = 0
    val cache = new AutotuneHintCache(stageKey => {
      fetches += 1
      AutotuneCachedHint.empty(stageKey)
    })

    cache.put(StageRuntimeHint.empty(key).copy(version = 5L, expiresAtNanos = Long.MaxValue))
    val got = cache.get(key)
    assert(got.hasHint)
    assert(got.version == 5L)
    assert(fetches == 0)

    // An already-expired pushed hint forces a refetch on the next get.
    cache.put(StageRuntimeHint.empty(key).copy(version = 6L, expiresAtNanos = 0L))
    val refreshed = cache.get(key)
    assert(fetches == 1)
    assert(!refreshed.hasHint)
  }

  test("executor endpoint reports the applied-hint payload to the driver") {
    val ctx = new CapturingPluginContext
    val conf = new RapidsConf(Map(RapidsConf.AUTOTUNE_FAIL_OPEN.key -> "true"))
    val endpoint = new RapidsAutotuneExecutorEndpoint(ctx, conf)
    val scanHint = ScanRuntimeHint(
      eagerPrefetch = true, minReadWindow = 1, maxReadWindow = 4, maxReadyBytes = 2048L)
    val gpuHint = GpuRuntimeHint(maxConcurrentTasks = 3)
    val shuffleHint = ShuffleRuntimeHint(
      prefetchWindow = 6, maxReadyBytes = 4096L, coalesceTargetBytes = 1024L)
    val batchHint = BatchRuntimeHint(
      targetBatchBytes = 4096L, maxBatchBytes = 8192L, splitUntilSize = 512L)
    val cached = AutotuneCachedHint(StageRuntimeHint.empty(key).copy(
      version = 9L, scan = scanHint, gpu = gpuHint, shuffle = shuffleHint, batch = batchHint),
      hasHint = true)

    endpoint.recordAppliedHint(
      key, taskAttemptId = 42L, partitionId = 5, cached, gpuAppliedMaxConcurrentTasks = 2)

    ctx.lastSent match {
      case msg: RapidsAutotuneHintAppliedMsg =>
        assert(msg.executorId == "exec-7")
        assert(msg.key == key)
        assert(msg.taskAttemptId == 42L)
        assert(msg.partitionId == 5)
        assert(msg.hintVersion == 9L)
        assert(msg.hasHint)
        assert(msg.scan == scanHint)
        assert(msg.gpu == gpuHint)
        assert(msg.shuffle == shuffleHint)
        assert(msg.batch == batchHint)
        assert(msg.gpuAppliedMaxConcurrentTasks == 2)
      case other => fail(s"unexpected message sent to driver: $other")
    }
  }

  test("driver flattens an applied-hint report into the eventlog record") {
    // Distinct, non-symmetric values per field so a swapped/omitted mapping cannot pass.
    val msg = RapidsAutotuneHintAppliedMsg(
      executorId = "exec-9",
      key = key,
      taskAttemptId = 42L,
      partitionId = 5,
      hintVersion = 9L,
      hasHint = true,
      scan = ScanRuntimeHint(
        eagerPrefetch = true, minReadWindow = 1, maxReadWindow = 4, maxReadyBytes = 2048L),
      gpu = GpuRuntimeHint(maxConcurrentTasks = 3),
      shuffle = ShuffleRuntimeHint(
        prefetchWindow = 6, maxReadyBytes = 4096L, coalesceTargetBytes = 1024L),
      batch = BatchRuntimeHint(
        targetBatchBytes = 7000L, maxBatchBytes = 8192L, splitUntilSize = 512L),
      gpuAppliedMaxConcurrentTasks = 2)

    val event = RapidsAutotuneDriverEndpoint.toAppliedEvent(msg)

    assert(event.executorId == "exec-9")
    assert(event.executionId == key.executionId)
    assert(event.stageId == key.stageId)
    assert(event.stageAttemptId == key.stageAttemptId)
    assert(event.taskAttemptId == 42L)
    assert(event.partitionId == 5)
    assert(event.hintVersion == 9L)
    assert(event.hasHint)
    assert(event.scanEagerPrefetch)
    assert(event.scanMinReadWindow == 1)
    assert(event.scanMaxReadWindow == 4)
    assert(event.scanMaxReadyBytes == 2048L)
    assert(event.gpuMaxConcurrentTasks == 3)
    assert(event.gpuAppliedMaxConcurrentTasks == 2)
    assert(event.shufflePrefetchWindow == 6)
    assert(event.shuffleMaxReadyBytes == 4096L)
    assert(event.shuffleCoalesceTargetBytes == 1024L)
    assert(event.batchTargetBatchBytes == 7000L)
    assert(event.batchMaxBatchBytes == 8192L)
    assert(event.batchSplitUntilSize == 512L)
  }

  test("stage hint listener publishes bounded shuffle and batch hints for shuffle stages") {
    val conf = new RapidsConf(Map(
      RapidsConf.AUTOTUNE_GRAPH_ENABLED.key -> "true",
      RapidsConf.AUTOTUNE_GRAPH_MODE.key -> AutotuneGraphMode.GRAPH.toString,
      RapidsConf.AUTOTUNE_SHUFFLE_ENABLED.key -> "true",
      RapidsConf.AUTOTUNE_SHUFFLE_MAX_PREFETCH_WINDOW.key -> "6",
      RapidsConf.AUTOTUNE_SHUFFLE_MAX_READY_BYTES.key -> "2048",
      RapidsConf.AUTOTUNE_SHUFFLE_COALESCE_TARGET_BYTES.key -> "1024",
      RapidsConf.AUTOTUNE_BATCH_ENABLED.key -> "true",
      RapidsConf.AUTOTUNE_BATCH_TARGET_BYTES.key -> "4096",
      RapidsConf.AUTOTUNE_BATCH_MAX_BYTES.key -> "8192",
      RapidsConf.AUTOTUNE_BATCH_SPLIT_UNTIL_SIZE.key -> "512"))
    RapidsAutotuneDriverEndpoint.init(null, conf)
    try {
      val listener = new RapidsAutotuneStageHintListener(conf)
      val shuffleStage = AutotuneStageShape(
        hasGpuScan = false, hasGpuPrefetchConsumer = false, numTasks = 50, hasShuffle = true)
      val hint = listener.publishHintForStage(
        key.executionId, key.stageId, key.stageAttemptId, shuffleStage)

      assert(hint.version == 1L)
      assert(hint.shuffle ==
        ShuffleRuntimeHint(prefetchWindow = 6, maxReadyBytes = 2048L, coalesceTargetBytes = 1024L))
      assert(hint.batch ==
        BatchRuntimeHint(targetBatchBytes = 4096L, maxBatchBytes = 8192L, splitUntilSize = 512L))
      // A shuffle-only stage is not a scan-prefetch candidate, so the scan slot stays empty.
      assert(hint.scan == ScanRuntimeHint.empty)

      val response = RapidsAutotuneDriverEndpoint.handleHintRequest(
        RapidsAutotuneHintRequestMsg("exec-1", key))
      assert(response.hint.contains(hint))
    } finally {
      RapidsAutotuneDriverEndpoint.shutdown()
    }
  }

  test("stage observation aggregate merges per-task pressure signals") {
    def obs(wait: Long, hold: Long, host: Long) = RapidsAutotuneObservationMsg(
      executorId = "exec-1", key = key, taskAttemptId = 1L, partitionId = 0,
      hintVersion = 1L, gpuSemaphoreWaitNanos = wait, gpuHoldingNanos = hold,
      hostMemoryBytes = host)
    val agg = StageObservationAgg.empty.merge(obs(10L, 100L, 500L)).merge(obs(30L, 100L, 800L))
    assert(agg.taskCount == 2L)
    assert(agg.totalGpuSemaphoreWaitNanos == 40L)
    assert(agg.totalGpuHoldingNanos == 200L)
    assert(agg.maxHostMemoryBytes == 800L)
    assert(math.abs(agg.gpuWaitRatio - 0.2) < 1e-9)
    assert(StageObservationAgg.empty.gpuWaitRatio == 0.0)  // no divide-by-zero
  }

  test("driver ingests observations into the per-stage aggregate") {
    val conf = new RapidsConf(Map(RapidsConf.AUTOTUNE_GRAPH_ENABLED.key -> "true"))
    RapidsAutotuneDriverEndpoint.init(null, conf)
    try {
      val m1 = RapidsAutotuneObservationMsg("exec-1", key, 1L, 0, 1L, 10L, 100L, 500L)
      val m2 = RapidsAutotuneObservationMsg("exec-1", key, 2L, 1, 1L, 30L, 100L, 800L)
      RapidsAutotuneDriverEndpoint.handleObservation(m1)
      RapidsAutotuneDriverEndpoint.handleObservation(m2)
      val agg = RapidsAutotuneDriverEndpoint.observationFor(key)
      assert(agg.exists(_.taskCount == 2L))
      assert(agg.exists(_.totalGpuSemaphoreWaitNanos == 40L))
      assert(agg.exists(_.maxHostMemoryBytes == 800L))
      val ev = RapidsAutotuneDriverEndpoint.toObservationEvent(m2, agg.get)
      assert(ev.executionId == key.executionId && ev.stageId == key.stageId)
      assert(ev.taskAttemptId == 2L && ev.partitionId == 1 && ev.gpuSemaphoreWaitNanos == 30L)
      assert(ev.hostMemoryBytes == 800L && ev.stageTaskCount == 2L)
      assert(ev.stageMaxHostMemoryBytes == 800L)
    } finally {
      RapidsAutotuneDriverEndpoint.shutdown()
    }
  }

  test("executor endpoint reports a runtime observation to the driver") {
    val ctx = new CapturingPluginContext
    val conf = new RapidsConf(Map(RapidsConf.AUTOTUNE_FAIL_OPEN.key -> "true"))
    val endpoint = new RapidsAutotuneExecutorEndpoint(ctx, conf)
    endpoint.reportObservation(key, taskAttemptId = 7L, partitionId = 3, hintVersion = 5L,
      gpuSemaphoreWaitNanos = 11L, gpuHoldingNanos = 22L, hostMemoryBytes = 33L, spillBytes = 44L)
    ctx.lastSent match {
      case msg: RapidsAutotuneObservationMsg =>
        assert(msg.executorId == "exec-7")
        assert(msg.key == key && msg.taskAttemptId == 7L && msg.partitionId == 3)
        assert(msg.hintVersion == 5L && msg.gpuSemaphoreWaitNanos == 11L)
        assert(msg.gpuHoldingNanos == 22L && msg.hostMemoryBytes == 33L && msg.spillBytes == 44L)
      case other => fail(s"unexpected message: $other")
    }
  }

  test("executor endpoint fails open when reporting an observation fails") {
    val conf = new RapidsConf(Map(RapidsConf.AUTOTUNE_FAIL_OPEN.key -> "true"))
    val endpoint = new RapidsAutotuneExecutorEndpoint(new FailingPluginContext, conf)
    // FailingPluginContext.send throws; reportObservation must swallow it (fail-open), not throw.
    endpoint.reportObservation(key, taskAttemptId = 1L, partitionId = 0, hintVersion = 1L,
      gpuSemaphoreWaitNanos = 5L, gpuHoldingNanos = 10L, hostMemoryBytes = 20L, spillBytes = 30L)
  }

  // ---------------------------------------------------------------------------
  // Slice 2: closed-loop model + mid-query re-hint
  // ---------------------------------------------------------------------------

  private val slice2Caps = AutotuneModelCaps(
    scanMaxReadWindow = 8, scanMaxReadyBytes = 1000L, gpuMaxConcurrentTasks = 4,
    minSampleTasks = 2L)

  private def stageHint(scanWindow: Int, gpuCap: Int): StageRuntimeHint =
    StageRuntimeHint.empty(key).copy(
      version = 1L,
      scan = if (scanWindow > 0) {
        ScanRuntimeHint(eagerPrefetch = true, minReadWindow = 1,
          maxReadWindow = scanWindow, maxReadyBytes = 1000L)
      } else ScanRuntimeHint.empty,
      gpu = if (gpuCap > 0) GpuRuntimeHint(gpuCap) else GpuRuntimeHint.empty)

  private def agg(
      tasks: Long, wait: Long, hold: Long, host: Long, spill: Long): StageObservationAgg =
    StageObservationAgg(taskCount = tasks, totalGpuSemaphoreWaitNanos = wait,
      totalGpuHoldingNanos = hold, maxHostMemoryBytes = host, totalSpillBytes = spill)

  test("model does not act before the minimum sample count") {
    // Heavy pressure, but only one sampled task (< minSampleTasks=2): no decision.
    val obs = agg(tasks = 1L, wait = 1000L, hold = 1L, host = 999L, spill = 5L)
    assert(AutotuneGraphModel.decide(obs, stageHint(8, 4), slice2Caps).isEmpty)
  }

  test("model leaves inactive (un-hinted) knobs untouched") {
    // No scan window, no GPU cap in the current hint -> nothing to tune even under pressure.
    val obs = agg(tasks = 50L, wait = 1000L, hold = 1L, host = 999L, spill = 100L)
    assert(AutotuneGraphModel.decide(obs, stageHint(0, 0), slice2Caps).isEmpty)
  }

  test("model halves the scan window under host-memory pressure, never below the floor") {
    val obs = agg(tasks = 50L, wait = 0L, hold = 100L, host = 950L, spill = 0L) // 950 >= 0.9*1000
    val d = AutotuneGraphModel.decide(obs, stageHint(8, 0), slice2Caps).get
    assert(d.isDecrease)
    assert(d.hint.scan.maxReadWindow == 4)
    // From a window of 1 (== floor) there is nowhere to go down -> no decision.
    assert(AutotuneGraphModel.decide(obs, stageHint(1, 0), slice2Caps).isEmpty)
  }

  test("model lowers the GPU cap under high semaphore-wait pressure") {
    val obs = agg(tasks = 50L, wait = 1000L, hold = 100L, host = 0L, spill = 0L) // ratio 10 >= 1.0
    val d = AutotuneGraphModel.decide(obs, stageHint(0, 4), slice2Caps).get
    assert(d.isDecrease && d.hint.gpu.maxConcurrentTasks == 2)
    assert(AutotuneGraphModel.decide(obs, stageHint(0, 1), slice2Caps).isEmpty) // already at floor
  }

  test("model reduces both knobs under spill pressure (prefer-reduce over restore)") {
    // Spill present plus otherwise-relaxed signals (low wait, low host): reduce wins, no restore.
    val obs = agg(tasks = 50L, wait = 0L, hold = 100L, host = 10L, spill = 1L)
    val d = AutotuneGraphModel.decide(obs, stageHint(8, 4), slice2Caps).get
    assert(d.isDecrease)
    assert(d.hint.scan.maxReadWindow == 4)
    assert(d.hint.gpu.maxConcurrentTasks == 2)
  }

  test("model restores knobs toward the static cap once pressure clears, never beyond it") {
    // Relaxed: low wait ratio, no spill, host well under budget. Knobs sit below their caps.
    val relaxed = agg(tasks = 50L, wait = 1L, hold = 100L, host = 100L, spill = 0L)
    val d = AutotuneGraphModel.decide(relaxed, stageHint(4, 2), slice2Caps).get
    assert(!d.isDecrease)
    assert(d.hint.scan.maxReadWindow == 5)            // +1 toward cap 8
    assert(d.hint.gpu.maxConcurrentTasks == 3)        // +1 toward cap 4
    // At the caps already -> nothing to restore.
    assert(AutotuneGraphModel.decide(relaxed, stageHint(8, 4), slice2Caps).isEmpty)
  }

  test("republishStageHint overwrites with a bumped version and is inert when disabled") {
    val conf = new RapidsConf(Map(
      RapidsConf.AUTOTUNE_GRAPH_ENABLED.key -> "true",
      RapidsConf.AUTOTUNE_GRAPH_MODE.key -> AutotuneGraphMode.GRAPH.toString))
    RapidsAutotuneDriverEndpoint.init(null, conf)
    try {
      val first = RapidsAutotuneDriverEndpoint.publishDefaultNoopHint(key)
      assert(first.version == 1L)
      val updated = RapidsAutotuneDriverEndpoint.republishStageHint(
        key, first.copy(gpu = GpuRuntimeHint(2)))
      assert(updated.version == 2L)
      assert(updated.gpu == GpuRuntimeHint(2))
      assert(updated.expiresAtNanos == Long.MaxValue)
      // The endpoint now serves the republished version.
      val served = RapidsAutotuneDriverEndpoint.handleHintRequest(
        RapidsAutotuneHintRequestMsg("exec-1", key)).hint
      assert(served.contains(updated))
    } finally {
      RapidsAutotuneDriverEndpoint.shutdown()
    }
    // Disabled endpoint republishes nothing.
    assert(RapidsAutotuneDriverEndpoint.republishStageHint(key, stageHint(8, 4)).version == 0L)
  }

  test("driver closes the loop: observed GPU pressure bumps the hint version and lowers the cap") {
    val conf = new RapidsConf(Map(
      RapidsConf.AUTOTUNE_GRAPH_ENABLED.key -> "true",
      RapidsConf.AUTOTUNE_GRAPH_MODE.key -> AutotuneGraphMode.GRAPH.toString,
      RapidsConf.AUTOTUNE_GPU_MAX_CONCURRENT_TASKS.key -> "4",
      RapidsConf.AUTOTUNE_GRAPH_MIN_SAMPLE_TASKS.key -> "2",
      RapidsConf.AUTOTUNE_GRAPH_UPDATE_INTERVAL_MS.key -> "0")) // no debounce/cooldown for the test
    RapidsAutotuneDriverEndpoint.init(null, conf)
    try {
      val listener = new RapidsAutotuneStageHintListener(conf)
      val shape =
        AutotuneStageShape(hasGpuScan = false, hasGpuPrefetchConsumer = false, numTasks = 50)
      val initial = listener.publishHintForStage(
        key.executionId, key.stageId, key.stageAttemptId, shape)
      assert(initial.version == 1L && initial.gpu.maxConcurrentTasks == 4)

      // High semaphore-wait observations (wait >> holding) -> GPU contention pressure. The model
      // consumes a per-decision window, so each decision needs minSampleTasks (=2) fresh samples.
      def obs() = RapidsAutotuneObservationMsg("exec-1", key, taskAttemptId = 1L, partitionId = 0,
        hintVersion = 1L, gpuSemaphoreWaitNanos = 500L, gpuHoldingNanos = 50L,
        hostMemoryBytes = 0L, spillBytes = 0L)
      def feed(n: Int): Unit =
        (1 to n).foreach(_ => RapidsAutotuneDriverEndpoint.handleObservation(obs()))
      def served = RapidsAutotuneDriverEndpoint.handleHintRequest(
        RapidsAutotuneHintRequestMsg("exec-1", key)).hint.get

      feed(1) // window 1 < 2 -> no decision yet
      assert(served.version == 1L && served.gpu.maxConcurrentTasks == 4)
      feed(1) // window hits 2 -> reduce 4 -> 2, v2
      assert(served.version == 2L && served.gpu.maxConcurrentTasks == 2)
      feed(2) // next window -> reduce 2 -> 1, v3
      assert(served.version == 3L && served.gpu.maxConcurrentTasks == 1)
      feed(2) // at floor -> no change, version held
      assert(served.version == 3L && served.gpu.maxConcurrentTasks == 1)
    } finally {
      RapidsAutotuneDriverEndpoint.shutdown()
    }
  }

  test("driver does not run the model outside GRAPH mode") {
    val conf = new RapidsConf(Map(
      RapidsConf.AUTOTUNE_GRAPH_ENABLED.key -> "true", // OBSERVE mode (default)
      RapidsConf.AUTOTUNE_GRAPH_MIN_SAMPLE_TASKS.key -> "1"))
    RapidsAutotuneDriverEndpoint.init(null, conf)
    try {
      RapidsAutotuneDriverEndpoint.publishDefaultNoopHint(key)
      RapidsAutotuneDriverEndpoint.handleObservation(RapidsAutotuneObservationMsg(
        "exec-1", key, 1L, 0, 1L, 5000L, 1L, 999L, 1000L))
      // OBSERVE never republishes: still version 1.
      assert(RapidsAutotuneDriverEndpoint.handleHintRequest(
        RapidsAutotuneHintRequestMsg("exec-1", key)).hint.exists(_.version == 1L))
    } finally {
      RapidsAutotuneDriverEndpoint.shutdown()
    }
  }

  test("stage observation aggregate sums spill bytes") {
    def obs(spill: Long) = RapidsAutotuneObservationMsg(
      executorId = "exec-1", key = key, taskAttemptId = 1L, partitionId = 0,
      hintVersion = 1L, gpuSemaphoreWaitNanos = 0L, gpuHoldingNanos = 0L,
      hostMemoryBytes = 0L, spillBytes = spill)
    val a = StageObservationAgg.empty.merge(obs(100L)).merge(obs(50L))
    assert(a.totalSpillBytes == 150L)
  }

  test("hint cache refetches after the fetch TTL elapses, memoizes within it") {
    var version = 0L
    var nowNanos = 1000L
    val cache = new AutotuneHintCache(
      stageKey => {
        version += 1
        AutotuneCachedHint(StageRuntimeHint.empty(stageKey).copy(version = version), hasHint = true)
      },
      fetchTtlNanos = 100L,
      nowNanos = () => nowNanos)

    assert(cache.get(key).version == 1L) // initial fetch at t=1000
    nowNanos = 1050L
    assert(cache.get(key).version == 1L) // within TTL -> memoized
    nowNanos = 1100L                     // 1100 - 1000 == 100 >= TTL -> stale
    assert(cache.get(key).version == 2L) // refetch
    assert(version == 2L)
  }

  test("model RESTORE is reachable on the live path once a fresh window clears (windowing)") {
    // Regression for the cumulative-aggregate latch: drive the GPU cap down under pressure, then
    // feed a clean window and confirm the cap is restored toward the static cap. With a lifetime
    // cumulative aggregate this would be impossible (pressure would latch forever).
    val conf = new RapidsConf(Map(
      RapidsConf.AUTOTUNE_GRAPH_ENABLED.key -> "true",
      RapidsConf.AUTOTUNE_GRAPH_MODE.key -> AutotuneGraphMode.GRAPH.toString,
      RapidsConf.AUTOTUNE_GPU_MAX_CONCURRENT_TASKS.key -> "4",
      RapidsConf.AUTOTUNE_GRAPH_MIN_SAMPLE_TASKS.key -> "2",
      RapidsConf.AUTOTUNE_GRAPH_UPDATE_INTERVAL_MS.key -> "0")) // no debounce/cooldown for the test
    RapidsAutotuneDriverEndpoint.init(null, conf)
    try {
      val listener = new RapidsAutotuneStageHintListener(conf)
      val shape =
        AutotuneStageShape(hasGpuScan = false, hasGpuPrefetchConsumer = false, numTasks = 50)
      listener.publishHintForStage(key.executionId, key.stageId, key.stageAttemptId, shape)
      def feed(n: Int, wait: Long, hold: Long): Unit = (1 to n).foreach(_ =>
        RapidsAutotuneDriverEndpoint.handleObservation(RapidsAutotuneObservationMsg(
          "exec-1", key, 1L, 0, 1L, wait, hold, hostMemoryBytes = 0L, spillBytes = 0L)))
      def served = RapidsAutotuneDriverEndpoint.handleHintRequest(
        RapidsAutotuneHintRequestMsg("exec-1", key)).hint.get

      feed(2, wait = 500L, hold = 50L)  // pressure window -> reduce 4 -> 2
      assert(served.gpu.maxConcurrentTasks == 2)
      feed(2, wait = 1L, hold = 100L)   // clean window (ratio 0.01) -> restore 2 -> 3
      assert(served.gpu.maxConcurrentTasks == 3)
      feed(2, wait = 1L, hold = 100L)   // clean window -> restore 3 -> 4 (the static cap)
      assert(served.gpu.maxConcurrentTasks == 4)
      val atCap = served.version
      feed(2, wait = 1L, hold = 100L)   // at the cap -> no change
      assert(served.gpu.maxConcurrentTasks == 4 && served.version == atCap)
    } finally {
      RapidsAutotuneDriverEndpoint.shutdown()
    }
  }

  test("driver debounces model evaluations within the update interval") {
    val conf = new RapidsConf(Map(
      RapidsConf.AUTOTUNE_GRAPH_ENABLED.key -> "true",
      RapidsConf.AUTOTUNE_GRAPH_MODE.key -> AutotuneGraphMode.GRAPH.toString,
      RapidsConf.AUTOTUNE_GPU_MAX_CONCURRENT_TASKS.key -> "4",
      RapidsConf.AUTOTUNE_GRAPH_MIN_SAMPLE_TASKS.key -> "1",
      RapidsConf.AUTOTUNE_GRAPH_UPDATE_INTERVAL_MS.key -> "1")) // 1ms = 1_000_000ns debounce
    RapidsAutotuneDriverEndpoint.init(null, conf)
    var clock = 10_000_000L
    RapidsAutotuneDriverEndpoint.setNanoSourceForTest(() => clock)
    try {
      val listener = new RapidsAutotuneStageHintListener(conf)
      val shape =
        AutotuneStageShape(hasGpuScan = false, hasGpuPrefetchConsumer = false, numTasks = 50)
      listener.publishHintForStage(key.executionId, key.stageId, key.stageAttemptId, shape)
      def pressure(): Unit = RapidsAutotuneDriverEndpoint.handleObservation(
        RapidsAutotuneObservationMsg("exec-1", key, 1L, 0, 1L, 500L, 50L, 0L, 0L))
      def served = RapidsAutotuneDriverEndpoint.handleHintRequest(
        RapidsAutotuneHintRequestMsg("exec-1", key)).hint.get

      pressure()                       // first eval -> reduce 4 -> 2, v2
      assert(served.version == 2L && served.gpu.maxConcurrentTasks == 2)
      clock += 500_000L                // 0.5ms < 1ms -> debounced
      pressure()
      assert(served.version == 2L && served.gpu.maxConcurrentTasks == 2)
      clock += 600_000L                // total 1.1ms since last eval -> allowed
      pressure()
      assert(served.version == 3L && served.gpu.maxConcurrentTasks == 1)
    } finally {
      RapidsAutotuneDriverEndpoint.shutdown()
    }
  }

  test("driver allows decreases during cooldown but suppresses increases until it elapses") {
    val conf = new RapidsConf(Map(
      RapidsConf.AUTOTUNE_GRAPH_ENABLED.key -> "true",
      RapidsConf.AUTOTUNE_GRAPH_MODE.key -> AutotuneGraphMode.GRAPH.toString,
      RapidsConf.AUTOTUNE_GPU_MAX_CONCURRENT_TASKS.key -> "4",
      RapidsConf.AUTOTUNE_GRAPH_MIN_SAMPLE_TASKS.key -> "1",
      RapidsConf.AUTOTUNE_GRAPH_UPDATE_INTERVAL_MS.key -> "1")) // interval 1ms, cooldown 2ms
    RapidsAutotuneDriverEndpoint.init(null, conf)
    var clock = 10_000_000L
    RapidsAutotuneDriverEndpoint.setNanoSourceForTest(() => clock)
    try {
      val listener = new RapidsAutotuneStageHintListener(conf)
      val shape =
        AutotuneStageShape(hasGpuScan = false, hasGpuPrefetchConsumer = false, numTasks = 50)
      listener.publishHintForStage(key.executionId, key.stageId, key.stageAttemptId, shape)
      def obs(wait: Long, hold: Long): Unit = RapidsAutotuneDriverEndpoint.handleObservation(
        RapidsAutotuneObservationMsg("exec-1", key, 1L, 0, 1L, wait, hold, 0L, 0L))
      def served = RapidsAutotuneDriverEndpoint.handleHintRequest(
        RapidsAutotuneHintRequestMsg("exec-1", key)).hint.get

      obs(500L, 50L)        // reduce 4 -> 2 (decrease at t0)
      assert(served.gpu.maxConcurrentTasks == 2)
      clock += 1_000_000L   // past debounce, still within 2ms cooldown
      obs(500L, 50L)        // a DECREASE is still allowed during cooldown -> 2 -> 1
      assert(served.gpu.maxConcurrentTasks == 1)
      clock += 1_000_000L   // past debounce; cooldown measured from the t1 decrease -> still inside
      obs(1L, 100L)         // relaxed -> would restore, but suppressed by cooldown
      assert(served.gpu.maxConcurrentTasks == 1)
      val held = served.version
      clock += 1_500_000L   // now well past the cooldown from the last decrease
      obs(1L, 100L)         // relaxed -> restore allowed 1 -> 2
      assert(served.gpu.maxConcurrentTasks == 2 && served.version > held)
    } finally {
      RapidsAutotuneDriverEndpoint.shutdown()
    }
  }

  test("model does not restore a knob with headroom while another knob is under pressure") {
    // Global 'relaxed' gate: GPU pressure must block a scan-window restore even though the scan
    // knob has headroom and no host/spill pressure of its own. ratio 1000/100 = 10 -> gpu pressure.
    val obs = agg(tasks = 50L, wait = 1000L, hold = 100L, host = 0L, spill = 0L)
    val d = AutotuneGraphModel.decide(obs, stageHint(4, 2), slice2Caps).get
    assert(d.isDecrease)
    assert(d.hint.scan.maxReadWindow == 4)        // scan NOT restored (gpu pressure blocks relaxed)
    assert(d.hint.gpu.maxConcurrentTasks == 1)    // gpu reduced
  }

  test("driver model no-ops when observations arrive before any hint is published") {
    val conf = new RapidsConf(Map(
      RapidsConf.AUTOTUNE_GRAPH_ENABLED.key -> "true",
      RapidsConf.AUTOTUNE_GRAPH_MODE.key -> AutotuneGraphMode.GRAPH.toString,
      RapidsConf.AUTOTUNE_GRAPH_MIN_SAMPLE_TASKS.key -> "1"))
    RapidsAutotuneDriverEndpoint.init(null, conf)
    try {
      // No publishStageHint for this key. Observations must still be aggregated (cumulative view)
      // but the model must NOT synthesize/serve a hint for an un-published stage, and the call must
      // not propagate an exception to the RPC handler.
      RapidsAutotuneDriverEndpoint.handleObservation(
        RapidsAutotuneObservationMsg("exec-1", key, 1L, 0, 0L, 500L, 50L, 0L, 0L))
      RapidsAutotuneDriverEndpoint.handleObservation(
        RapidsAutotuneObservationMsg("exec-1", key, 2L, 1, 0L, 500L, 50L, 0L, 0L))
      // Observations were ingested into the cumulative aggregate...
      assert(RapidsAutotuneDriverEndpoint.observationFor(key).exists(_.taskCount == 2L))
      // ...but no hint was ever published or synthesized for the key.
      assert(RapidsAutotuneDriverEndpoint.handleHintRequest(
        RapidsAutotuneHintRequestMsg("exec-1", key)).hint.isEmpty)
    } finally {
      RapidsAutotuneDriverEndpoint.shutdown()
    }
  }
}

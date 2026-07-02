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

import com.codahale.metrics.MetricRegistry
import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.SparkConf
import org.apache.spark.api.plugin.PluginContext
import org.apache.spark.resource.ResourceInformation
import org.apache.spark.scheduler.{SparkListenerExecutorAdded, SparkListenerExecutorRemoved}
import org.apache.spark.scheduler.cluster.ExecutorInfo
import org.apache.spark.sql.rapids.GpuTaskMetrics

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

  test("shuffle read actuator raises only in OPTIMIZE and clamps to the host ceiling") {
    val staticCap = 128L
    val optimizeCeiling = 512L
    def hint(b: Long): Option[ShuffleRuntimeHint] =
      Some(ShuffleRuntimeHint(prefetchWindow = 0, maxReadyBytes = b, coalesceTargetBytes = 0L))

    // GRAPH semantics remain tighten-only even when an above-static ceiling is supplied.
    assert(ShuffleReadHints.effectiveMaxBytesInFlight(
      staticCap, hint(256L), enabled = true, allowAboveStatic = false,
      optimizeMaxBytesInFlight = optimizeCeiling) == staticCap)
    // OPTIMIZE accepts an above-static hint but never exceeds the separate hard ceiling.
    assert(ShuffleReadHints.effectiveMaxBytesInFlight(
      staticCap, hint(256L), enabled = true, allowAboveStatic = true,
      optimizeMaxBytesInFlight = optimizeCeiling) == 256L)
    assert(ShuffleReadHints.effectiveMaxBytesInFlight(
      staticCap, hint(1024L), enabled = true, allowAboveStatic = true,
      optimizeMaxBytesInFlight = optimizeCeiling) == optimizeCeiling)
    // No configured increase and the empty sentinel both preserve the static cap.
    assert(ShuffleReadHints.effectiveMaxBytesInFlight(
      staticCap, hint(256L), enabled = true, allowAboveStatic = true,
      optimizeMaxBytesInFlight = 0L) == staticCap)
    assert(ShuffleReadHints.effectiveMaxBytesInFlight(
      staticCap, Some(ShuffleRuntimeHint.empty), enabled = true, allowAboveStatic = true,
      optimizeMaxBytesInFlight = optimizeCeiling) == staticCap)
  }

  test("shuffle prefetch actuator enforces the optimizer-selected outstanding-work window") {
    val hint = Some(ShuffleRuntimeHint(
      prefetchWindow = 4, maxReadyBytes = 128L, coalesceTargetBytes = 0L))
    assert(ShuffleReadHints.effectivePrefetchWindow(hint) == 4)
    assert(ShuffleReadHints.effectivePrefetchWindow(None) == Integer.MAX_VALUE)
    assert(ShuffleReadHints.blocksToDrain(4, outstandingBlocks = 1L, readyBlocks = 10) == 3)
    // Always permit one blocking fetch so a tight/stale window cannot deadlock the reader.
    assert(ShuffleReadHints.blocksToDrain(4, outstandingBlocks = 4L, readyBlocks = 10) == 1)
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

  test("stage shape identifies broadcast work for AQE calibration") {
    val broadcast = AutotuneStageShape.fromRddScopeNames(
      Seq("GpuBroadcastExchange", "GpuHashJoin"), numTasks = 8)
    val shuffle = AutotuneStageShape.fromRddScopeNames(
      Seq("GpuColumnarExchange", "GpuHashJoin"), numTasks = 8)

    assert(broadcast.hasBroadcast && broadcast.hasGpuWork)
    assert(!broadcast.hasShuffle)
    assert(shuffle.hasShuffle && !shuffle.hasBroadcast)
  }

  test("one graph optimizer owns the complete initial joint hint") {
    val conf = new RapidsConf(Map(
      RapidsConf.AUTOTUNE_GRAPH_ENABLED.key -> "true",
      RapidsConf.AUTOTUNE_GRAPH_MODE.key -> AutotuneGraphMode.OPTIMIZE.toString,
      RapidsConf.AUTOTUNE_SCAN_MAX_READ_WINDOW.key -> "8",
      RapidsConf.SCAN_PREFETCH_MAX_PARALLELISM.key -> "8",
      RapidsConf.AUTOTUNE_OPTIMIZE_SCAN_MAX_READ_WINDOW.key -> "32",
      RapidsConf.AUTOTUNE_SCAN_MAX_READY_BYTES.key -> "1024",
      RapidsConf.AUTOTUNE_GPU_MAX_CONCURRENT_TASKS.key -> "4",
      RapidsConf.AUTOTUNE_OPTIMIZE_GPU_MAX_CONCURRENT_TASKS.key -> "8",
      RapidsConf.AUTOTUNE_SHUFFLE_ENABLED.key -> "true",
      RapidsConf.SHUFFLE_MULTITHREADED_MAX_BYTES_IN_FLIGHT.key -> "2048",
      RapidsConf.AUTOTUNE_OPTIMIZE_SHUFFLE_MAX_BYTES_IN_FLIGHT.key -> "8192",
      RapidsConf.AUTOTUNE_SHUFFLE_MAX_PREFETCH_WINDOW.key -> "6",
      RapidsConf.AUTOTUNE_SHUFFLE_COALESCE_TARGET_BYTES.key -> "4096",
      RapidsConf.AUTOTUNE_BATCH_ENABLED.key -> "true",
      RapidsConf.GPU_BATCH_SIZE_BYTES.key -> "8192",
      RapidsConf.AUTOTUNE_BATCH_TARGET_BYTES.key -> "4096",
      RapidsConf.AUTOTUNE_BATCH_MAX_BYTES.key -> "16384",
      RapidsConf.AUTOTUNE_BATCH_SPLIT_UNTIL_SIZE.key -> "512"))
    val optimizer = new AnalyticalGraphWideAutotuneOptimizer(
      GraphOptimizerConstraints.fromConf(conf, nativeGpuTaskSlots = 16))
    val shape = AutotuneStageShape(
      hasGpuScan = true, hasGpuPrefetchConsumer = true, numTasks = 50, hasShuffle = true)
    val hint = optimizer.initialHint(key, AutotuneStageDescriptor(shape, Seq(1, 2)))

    // Cold hints begin at deployed/native settings or the configured feasible envelope, not at
    // an unevidenced low-resource point.
    assert(hint.scan == ScanRuntimeHint(true, 1, 8, 1024L))
    assert(hint.gpu == GpuRuntimeHint(
      maxConcurrentTasks = 16, sharedMaxConcurrentTasks = 16))
    assert(hint.shuffle == ShuffleRuntimeHint(6, 2048L, 8192L))
    assert(hint.batch == BatchRuntimeHint(8192L, 16384L, 0L))
  }

  test("native GPU slots and batch size remain in the optimizer domain") {
    val oneGiB = 1024L * 1024L * 1024L
    val conf = new RapidsConf(Map(
      RapidsConf.AUTOTUNE_GRAPH_ENABLED.key -> "true",
      RapidsConf.AUTOTUNE_GRAPH_MODE.key -> AutotuneGraphMode.OPTIMIZE.toString,
      RapidsConf.CONCURRENT_GPU_TASKS.key -> "4",
      RapidsConf.AUTOTUNE_GPU_MAX_CONCURRENT_TASKS.key -> "4",
      RapidsConf.AUTOTUNE_OPTIMIZE_GPU_MAX_CONCURRENT_TASKS.key -> "8",
      RapidsConf.AUTOTUNE_SHUFFLE_ENABLED.key -> "true",
      RapidsConf.AUTOTUNE_SHUFFLE_COALESCE_TARGET_BYTES.key -> "128m",
      RapidsConf.AUTOTUNE_BATCH_ENABLED.key -> "true",
      RapidsConf.GPU_BATCH_SIZE_BYTES.key -> "1g",
      RapidsConf.AUTOTUNE_BATCH_TARGET_BYTES.key -> "128m",
      RapidsConf.AUTOTUNE_BATCH_MAX_BYTES.key -> "512m",
      RapidsConf.AUTOTUNE_BATCH_SPLIT_UNTIL_SIZE.key -> "128m"))

    val constraints = GraphOptimizerConstraints.fromConf(conf, nativeGpuTaskSlots = 16)
    // concurrentGpuTasks seeds dynamic memory permits; it is not a hard scheduling envelope.
    assert(constraints.gpu.initialConcurrentTasks == 16)
    assert(constraints.gpu.maxConcurrentTasks == 16)
    assert(constraints.batch.initialTargetBytes == oneGiB)
    assert(constraints.batch.minimumTargetBytes == 128L * 1024L * 1024L)
    assert(constraints.batch.maxBatchBytes == oneGiB)

    val optimizer = new AnalyticalGraphWideAutotuneOptimizer(constraints)
    val shape = AutotuneStageShape(
      hasGpuScan = false, hasGpuPrefetchConsumer = true, numTasks = 200, hasShuffle = true)
    val hint = optimizer.initialHint(key, AutotuneStageDescriptor(shape))
    assert(hint.gpu == GpuRuntimeHint(16, sharedMaxConcurrentTasks = 16))
    assert(hint.shuffle.coalesceTargetBytes == oneGiB)
    assert(hint.batch == BatchRuntimeHint(oneGiB, oneGiB, splitUntilSize = 0L))
  }

  test("each autotune subsystem gates independently for mechanism ablation") {
    val conf = new RapidsConf(Map(
      RapidsConf.AUTOTUNE_GRAPH_ENABLED.key -> "true",
      RapidsConf.AUTOTUNE_GRAPH_MODE.key -> AutotuneGraphMode.OPTIMIZE.toString,
      RapidsConf.AUTOTUNE_SCAN_ENABLED.key -> "false",
      RapidsConf.AUTOTUNE_GPU_MAX_CONCURRENT_TASKS.key -> "4",
      RapidsConf.AUTOTUNE_SHUFFLE_ENABLED.key -> "false",
      RapidsConf.AUTOTUNE_BATCH_ENABLED.key -> "false"))

    val constraints = GraphOptimizerConstraints.fromConf(conf, nativeGpuTaskSlots = 16)
    assert(!constraints.scan.enabled)
    assert(!constraints.shuffle.enabled)
    assert(!constraints.batch.enabled)
    assert(constraints.gpu.enabled)

    // A disabled subsystem publishes only the empty no-op hint for its actuator.
    val optimizer = new AnalyticalGraphWideAutotuneOptimizer(constraints)
    val shape = AutotuneStageShape(
      hasGpuScan = true, hasGpuPrefetchConsumer = true, numTasks = 200, hasShuffle = true)
    val hint = optimizer.initialHint(key, AutotuneStageDescriptor(shape))
    assert(hint.scan == ScanRuntimeHint.empty)
    assert(hint.shuffle == ShuffleRuntimeHint.empty)
    assert(hint.batch == BatchRuntimeHint.empty)
    assert(hint.gpu.maxConcurrentTasks > 0)
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

  test("driver endpoint sums cluster task slots from registered executors") {
    val conf = new RapidsConf(Map(RapidsConf.AUTOTUNE_GRAPH_ENABLED.key -> "true"))
    RapidsAutotuneDriverEndpoint.init(null, conf)
    try {
      assert(RapidsAutotuneDriverEndpoint.clusterTaskSlots == 0)
      RapidsAutotuneDriverEndpoint.executorAdded("1", 16)
      RapidsAutotuneDriverEndpoint.executorAdded("2", 16)
      assert(RapidsAutotuneDriverEndpoint.clusterTaskSlots == 32)
      // Re-registration replaces the census entry; it never double-counts an executor.
      RapidsAutotuneDriverEndpoint.executorAdded("1", 8)
      assert(RapidsAutotuneDriverEndpoint.clusterTaskSlots == 24)
      RapidsAutotuneDriverEndpoint.executorRemoved("2")
      assert(RapidsAutotuneDriverEndpoint.clusterTaskSlots == 8)
      RapidsAutotuneDriverEndpoint.executorRemoved("unknown")
      RapidsAutotuneDriverEndpoint.executorAdded("3", 0)
      assert(RapidsAutotuneDriverEndpoint.clusterTaskSlots == 8)
      RapidsAutotuneDriverEndpoint.setTaskCpusForTest(2)
      assert(RapidsAutotuneDriverEndpoint.clusterTaskSlots == 4)
    } finally {
      RapidsAutotuneDriverEndpoint.shutdown()
    }
  }

  test("stage hint listener feeds executor registration into the slot census") {
    val conf = new RapidsConf(Map(RapidsConf.AUTOTUNE_GRAPH_ENABLED.key -> "true"))
    RapidsAutotuneDriverEndpoint.init(null, conf)
    try {
      val listener = new RapidsAutotuneStageHintListener(conf)
      listener.onExecutorAdded(SparkListenerExecutorAdded(
        0L, "1", new ExecutorInfo("host-1", 16, Map.empty)))
      listener.onExecutorAdded(SparkListenerExecutorAdded(
        0L, "2", new ExecutorInfo("host-2", 16, Map.empty)))
      assert(RapidsAutotuneDriverEndpoint.clusterTaskSlots == 32)
      listener.onExecutorRemoved(SparkListenerExecutorRemoved(1L, "1", "lost"))
      assert(RapidsAutotuneDriverEndpoint.clusterTaskSlots == 16)
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
      assert(hint.gpu == GpuRuntimeHint(
        maxConcurrentTasks = 4, sharedMaxConcurrentTasks = 4))

      val response = RapidsAutotuneDriverEndpoint.handleHintRequest(
        RapidsAutotuneHintRequestMsg("exec-1", key))
      assert(response.hint.contains(hint))
    } finally {
      RapidsAutotuneDriverEndpoint.shutdown()
    }
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
    val gpuHint = GpuRuntimeHint(
      maxConcurrentTasks = 3,
      sharedMaxConcurrentTasks = 5,
      schedulingPriority = 700L)
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
      gpu = GpuRuntimeHint(
        maxConcurrentTasks = 3,
        sharedMaxConcurrentTasks = 5,
        schedulingPriority = 700L),
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
    assert(event.gpuSharedMaxConcurrentTasks == 5)
    assert(event.gpuSchedulingPriority == 700L)
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
      RapidsConf.GPU_BATCH_SIZE_BYTES.key -> "4096",
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
        ShuffleRuntimeHint(prefetchWindow = 6, maxReadyBytes = 2048L,
          coalesceTargetBytes = 4096L))
      assert(hint.batch ==
        BatchRuntimeHint(targetBatchBytes = 4096L, maxBatchBytes = 8192L, splitUntilSize = 0L))
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
    def obs(wait: Long, hold: Long, host: Long, shuffleAcquires: Long, shuffleFails: Long) =
      RapidsAutotuneObservationMsg(
      executorId = "exec-1", key = key, taskAttemptId = 1L, partitionId = 0,
      hintVersion = 1L, gpuSemaphoreWaitNanos = wait, gpuHoldingNanos = hold,
      hostMemoryBytes = host, shuffleReadLimiterAcquireCount = shuffleAcquires,
      shuffleReadLimiterAcquireFailCount = shuffleFails)
    val agg = StageObservationAgg.empty
      .merge(obs(10L, 100L, 500L, 40L, 4L))
      .merge(obs(30L, 100L, 800L, 60L, 16L))
    assert(agg.taskCount == 2L)
    assert(agg.totalGpuSemaphoreWaitNanos == 40L)
    assert(agg.totalGpuHoldingNanos == 200L)
    assert(agg.maxHostMemoryBytes == 800L)
    assert(agg.totalShuffleReadLimiterAcquireCount == 100L)
    assert(agg.totalShuffleReadLimiterAcquireFailCount == 20L)
    assert(math.abs(agg.gpuWaitRatio - 0.2) < 1e-9)
    assert(math.abs(agg.shuffleReadLimiterFailureRatio - 0.2) < 1e-9)
    assert(StageObservationAgg.empty.gpuWaitRatio == 0.0)  // no divide-by-zero
    assert(StageObservationAgg.empty.shuffleReadLimiterFailureRatio == 0.0)
  }

  test("task metrics record multithreaded shuffle limiter pressure") {
    val metrics = new GpuTaskMetrics
    metrics.recordShuffleReadLimiterAcquire(didFit = true)
    metrics.recordShuffleReadLimiterAcquire(didFit = false)
    assert(metrics.getShuffleReadLimiterAcquireCount == 2L)
    assert(metrics.getShuffleReadLimiterAcquireFailCount == 1L)
  }

  test("driver ingests observations into the per-stage aggregate") {
    val conf = new RapidsConf(Map(RapidsConf.AUTOTUNE_GRAPH_ENABLED.key -> "true"))
    RapidsAutotuneDriverEndpoint.init(null, conf)
    try {
      val m1 = RapidsAutotuneObservationMsg("exec-1", key, 1L, 0, 1L, 10L, 100L, 500L,
        shuffleReadLimiterAcquireCount = 40L, shuffleReadLimiterAcquireFailCount = 4L)
      val m2 = RapidsAutotuneObservationMsg("exec-1", key, 2L, 1, 1L, 30L, 100L, 800L,
        shuffleReadLimiterAcquireCount = 60L, shuffleReadLimiterAcquireFailCount = 16L)
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
      assert(ev.shuffleReadLimiterAcquireCount == 60L)
      assert(ev.shuffleReadLimiterAcquireFailCount == 16L)
      assert(ev.stageShuffleReadLimiterAcquireCount == 100L)
      assert(ev.stageShuffleReadLimiterAcquireFailCount == 20L)
    } finally {
      RapidsAutotuneDriverEndpoint.shutdown()
    }
  }

  test("executor endpoint reports a runtime observation to the driver") {
    val ctx = new CapturingPluginContext
    val conf = new RapidsConf(Map(RapidsConf.AUTOTUNE_FAIL_OPEN.key -> "true"))
    val endpoint = new RapidsAutotuneExecutorEndpoint(ctx, conf)
    endpoint.reportObservation(key, taskAttemptId = 7L, partitionId = 3, hintVersion = 5L,
      gpuSemaphoreWaitNanos = 11L, gpuHoldingNanos = 22L, hostMemoryBytes = 33L, spillBytes = 44L,
      shuffleReadLimiterAcquireCount = 55L, shuffleReadLimiterAcquireFailCount = 6L)
    ctx.lastSent match {
      case msg: RapidsAutotuneObservationMsg =>
        assert(msg.executorId == "exec-7")
        assert(msg.key == key && msg.taskAttemptId == 7L && msg.partitionId == 3)
        assert(msg.hintVersion == 5L && msg.gpuSemaphoreWaitNanos == 11L)
        assert(msg.gpuHoldingNanos == 22L && msg.hostMemoryBytes == 33L && msg.spillBytes == 44L)
        assert(msg.shuffleReadLimiterAcquireCount == 55L)
        assert(msg.shuffleReadLimiterAcquireFailCount == 6L)
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
  // Graph-wide analytical optimizer and closed-loop re-hint
  // ---------------------------------------------------------------------------

  private def optimizerConstraints(
      minSamples: Long = 2L,
      intervalNanos: Long = 0L): GraphOptimizerConstraints = GraphOptimizerConstraints(
    minSampleTasks = minSamples,
    updateIntervalNanos = intervalNanos,
    scan = ScanOptimizerBounds(enabled = true, initialReadWindow = 2,
      maxReadWindow = 8, maxReadyBytes = 4096L),
    gpu = GpuOptimizerBounds(enabled = true, initialConcurrentTasks = 2,
      maxConcurrentTasks = 8),
    shuffle = ShuffleOptimizerBounds(enabled = true, initialPrefetchWindow = 1,
      maxPrefetchWindow = 16,
      initialReadyBytes = 128L, maxReadyBytes = 512L, initialCoalesceBytes = 256L),
    batch = BatchOptimizerBounds(enabled = true, initialTargetBytes = 256L,
      maxBatchBytes = 1024L, initialSplitUntilSize = 128L))

  private def optimizerObservation(
      stageKey: AutotuneStageKey,
      taskAttemptId: Long,
      wait: Long,
      holding: Long,
      duration: Long,
      input: Long,
      output: Long,
      shuffleRead: Long = 0L,
      shuffleWrite: Long = 0L,
      spill: Long = 0L,
      retry: Long = 0L): RapidsAutotuneObservationMsg = RapidsAutotuneObservationMsg(
    executorId = "exec-1",
    key = stageKey,
    taskAttemptId = taskAttemptId,
    partitionId = taskAttemptId.toInt,
    hintVersion = 1L,
    gpuSemaphoreWaitNanos = wait,
    gpuHoldingNanos = holding,
    hostMemoryBytes = 0L,
    spillBytes = spill,
    taskDurationNanos = duration,
    inputBytes = input,
    outputBytes = output,
    shuffleReadBytes = shuffleRead,
    shuffleWriteBytes = shuffleWrite,
    retryOrLostTimeNanos = retry)

  test("critical-path allocator spends a shared GPU budget on the longest active branch") {
    val leftKey = key.copy(stageId = 10)
    val rightKey = key.copy(stageId = 11)
    val joinKey = key.copy(stageId = 12)
    val shape = AutotuneStageShape(
      hasGpuScan = false, hasGpuPrefetchConsumer = true, numTasks = 1)
    def hint(stageKey: AutotuneStageKey, gpuTasks: Int): StageRuntimeHint =
      StageRuntimeHint.empty(stageKey).copy(gpu = GpuRuntimeHint(gpuTasks))
    def curve(
        stageKey: AutotuneStageKey,
        costs: Seq[Double]): GraphStageCostCurve = GraphStageCostCurve(
      predictedCurrentNanos = costs.head,
      sampleTasks = 1L,
      alternatives = costs.zipWithIndex.map { case (cost, index) =>
        val gpuTasks = index + 1
        GraphStageCostAlternative(gpuTasks, hint(stageKey, gpuTasks), cost)
      })
    val graph = Seq(
      GraphStageAllocationInput(leftKey, AutotuneStageDescriptor(shape), active = true,
        observedTasks = 0L, currentHint = hint(leftKey, 1),
        costCurve = Some(curve(leftKey, Seq(100.0, 50.0, 34.0)))),
      GraphStageAllocationInput(rightKey, AutotuneStageDescriptor(shape), active = true,
        observedTasks = 0L, currentHint = hint(rightKey, 1),
        costCurve = Some(curve(rightKey, Seq(40.0, 25.0, 20.0)))),
      GraphStageAllocationInput(joinKey, AutotuneStageDescriptor(shape, Seq(10, 11)),
        active = false, observedTasks = 0L, currentHint = hint(joinKey, 1), costCurve = None))

    val allocations = GraphCriticalPathAllocator.allocate(graph, sharedGpuBudget = 3)
      .map(allocation => allocation.key -> allocation.hint.gpu).toMap

    assert(allocations(leftKey).maxConcurrentTasks == 2)
    assert(allocations(rightKey).maxConcurrentTasks == 1)
    assert(allocations.values.map(_.maxConcurrentTasks).sum == 3)
    assert(allocations.values.forall(_.sharedMaxConcurrentTasks == 3))
    assert(allocations(leftKey).schedulingPriority > allocations(rightKey).schedulingPriority)
  }

  test("oversubscribed graph keeps every active stage runnable under the lower shared cap") {
    val shape = AutotuneStageShape(
      hasGpuScan = false, hasGpuPrefetchConsumer = true, numTasks = 1)
    val graph = (1 to 3).map { stageId =>
      val stageKey = key.copy(stageId = stageId)
      val stageHint = StageRuntimeHint.empty(stageKey).copy(gpu = GpuRuntimeHint(1))
      GraphStageAllocationInput(
        stageKey,
        AutotuneStageDescriptor(shape),
        active = true,
        observedTasks = 0L,
        currentHint = stageHint,
        costCurve = Some(GraphStageCostCurve(
          predictedCurrentNanos = stageId.toDouble,
          sampleTasks = 1L,
          alternatives = Seq(GraphStageCostAlternative(1, stageHint, stageId.toDouble)))))
    }

    val allocations = GraphCriticalPathAllocator.allocate(graph, sharedGpuBudget = 2)
    assert(allocations.size == 3)
    assert(allocations.forall(_.hint.gpu.maxConcurrentTasks == 1))
    assert(allocations.forall(_.hint.gpu.sharedMaxConcurrentTasks == 2))
  }

  test("shared allocator preserves the native quota for an uncalibrated sibling") {
    val calibratedKey = key.copy(stageId = 20)
    val coldKey = key.copy(stageId = 21)
    val shape = AutotuneStageShape(
      hasGpuScan = false, hasGpuPrefetchConsumer = true, numTasks = 10)
    def hint(stageKey: AutotuneStageKey, gpuTasks: Int): StageRuntimeHint =
      StageRuntimeHint.empty(stageKey).copy(gpu = GpuRuntimeHint(gpuTasks))
    val calibratedHint = hint(calibratedKey, 1)
    val graph = Seq(
      GraphStageAllocationInput(
        calibratedKey,
        AutotuneStageDescriptor(shape),
        active = true,
        observedTasks = 2L,
        currentHint = calibratedHint,
        costCurve = Some(GraphStageCostCurve(
          predictedCurrentNanos = 100.0,
          sampleTasks = 2L,
          alternatives = Seq(
            GraphStageCostAlternative(1, calibratedHint, 100.0),
            GraphStageCostAlternative(2, hint(calibratedKey, 2), 50.0),
            GraphStageCostAlternative(3, hint(calibratedKey, 3), 34.0))))),
      GraphStageAllocationInput(
        coldKey,
        AutotuneStageDescriptor(shape),
        active = true,
        observedTasks = 0L,
        currentHint = hint(coldKey, 2),
        costCurve = None))

    val allocations = GraphCriticalPathAllocator.allocate(graph, sharedGpuBudget = 3)
      .map(allocation => allocation.key -> allocation.hint.gpu).toMap
    assert(allocations(calibratedKey).maxConcurrentTasks == 1)
    assert(allocations(coldKey).maxConcurrentTasks == 2)
    assert(allocations.values.map(_.maxConcurrentTasks).sum == 3)
  }

  test("single-setting feedback freezes joint executor controls at the measured point") {
    val constraints = optimizerConstraints()
    val shape = AutotuneStageShape(
      hasGpuScan = true, hasGpuPrefetchConsumer = true, numTasks = 50, hasShuffle = true)
    val current = StageRuntimeHint.empty(key).copy(
      scan = ScanRuntimeHint(true, 1, 2, 4096L),
      gpu = GpuRuntimeHint(2),
      shuffle = ShuffleRuntimeHint(1, 128L, 0L))
    val observation = StageObservationAgg.empty
      .merge(optimizerObservation(key, 1L, wait = 0L, holding = 2000L,
        duration = 10000L, input = 1000L, output = 500L, shuffleRead = 500L))
      .merge(optimizerObservation(key, 2L, wait = 0L, holding = 2000L,
        duration = 10000L, input = 1000L, output = 500L, shuffleRead = 500L))

    val curve = AnalyticalStageCostModel.costCurve(
      observation, current, shape, constraints).get
    assert(curve.alternatives.size == 1)
    assert(curve.alternatives.head.hint.scan == current.scan)
    assert(curve.alternatives.head.hint.gpu == current.gpu)
    assert(curve.alternatives.head.hint.shuffle == current.shuffle)
    assert(curve.alternatives.head.predictedNanos == curve.predictedCurrentNanos)
    assert(AnalyticalStageCostModel.optimize(
      observation, current, shape, constraints).isEmpty)
    val reasons = curve.analyticalState.get.freezeReasons
    assert(reasons.scanWindow == "single-operating-point-response-unidentified")
    assert(reasons.gpuTasks == "single-operating-point-response-unidentified")
    assert(reasons.shuffleWindow == "single-operating-point-response-unidentified")
    assert(reasons.shuffleBytes == "single-operating-point-response-unidentified")
  }

  test("semaphore wait alone does not identify a different GPU admission quota") {
    val constraints = optimizerConstraints()
    val shape = AutotuneStageShape(
      hasGpuScan = false, hasGpuPrefetchConsumer = true, numTasks = 50)
    val current = StageRuntimeHint.empty(key).copy(gpu = GpuRuntimeHint(4))
    val observation = StageObservationAgg.empty
      .merge(optimizerObservation(key, 1L, wait = 1000L, holding = 100L,
        duration = 1100L, input = 0L, output = 0L))
      .merge(optimizerObservation(key, 2L, wait = 1000L, holding = 100L,
        duration = 1100L, input = 0L, output = 0L))

    val curve = AnalyticalStageCostModel.costCurve(
      observation, current, shape, constraints).get
    assert(curve.alternatives.map(_.gpuTasks) == Seq(4))
    assert(curve.analyticalState.get.currentGradient.gpuTasks < 0.0)
    assert(curve.analyticalState.get.freezeReasons.gpuTasks ==
      "single-operating-point-response-unidentified")
  }

  test("online flow solve does not invent an unobserved lower GPU quota") {
    val constraints = optimizerConstraints()
    val shape = AutotuneStageShape(
      hasGpuScan = true, hasGpuPrefetchConsumer = true, numTasks = 50)
    val current = StageRuntimeHint.empty(key).copy(
      scan = ScanRuntimeHint(true, 1, 2, 4096L),
      gpu = GpuRuntimeHint(4))
    val observation = StageObservationAgg.empty
      .merge(optimizerObservation(key, 1L, wait = 0L, holding = 100L,
        duration = 10000L, input = 1000L, output = 500L))
      .merge(optimizerObservation(key, 2L, wait = 0L, holding = 100L,
        duration = 10000L, input = 1000L, output = 500L))

    val curve = AnalyticalStageCostModel.costCurve(
      observation, current, shape, constraints).get
    assert(curve.alternatives.map(_.gpuTasks) == Seq(current.gpu.maxConcurrentTasks))
  }

  test("single-setting observations do not invent a batch-size response") {
    val constraints = optimizerConstraints()
    val shape = AutotuneStageShape(
      hasGpuScan = false, hasGpuPrefetchConsumer = true, numTasks = 50)
    val current = StageRuntimeHint.empty(key).copy(
      gpu = GpuRuntimeHint(2),
      batch = BatchRuntimeHint(256L, 1024L, 128L))
    val observation = StageObservationAgg.empty
      .merge(optimizerObservation(key, 1L, wait = 0L, holding = 1000L,
        duration = 5000L, input = 4096L, output = 4096L))
      .merge(optimizerObservation(key, 2L, wait = 0L, holding = 1000L,
        duration = 5000L, input = 4096L, output = 4096L))

    val curve = AnalyticalStageCostModel.costCurve(
      observation, current, shape, constraints).get
    assert(curve.alternatives.head.hint.batch.targetBatchBytes == 256L)
    assert(curve.alternatives.head.hint.batch.splitUntilSize == 128L)
    assert(curve.analyticalState.get.freezeReasons.batchBytes == "no-measured-work")
  }

  test("task observations do not invent driver-side broadcast service time") {
    val constraints = optimizerConstraints()
    val shape = AutotuneStageShape(
      hasGpuScan = false,
      hasGpuPrefetchConsumer = true,
      numTasks = 50,
      hasBroadcast = true)
    val current = StageRuntimeHint.empty(key).copy(gpu = GpuRuntimeHint(2))
    val observation = StageObservationAgg.empty
      .merge(optimizerObservation(key, 1L, wait = 0L, holding = 1000L,
        duration = 5000L, input = 4096L, output = 1024L))
      .merge(optimizerObservation(key, 2L, wait = 0L, holding = 1000L,
        duration = 5000L, input = 4096L, output = 1024L))

    val sample = AnalyticalStageCostModel.costCurve(
      observation, current, shape, constraints).get.calibrationSample.get
    assert(sample.broadcastNanos == 0.0)
    assert(sample.broadcastBytes == 0L)
  }

  test("optimizer state owns sample windows and update cadence") {
    val optimizer = new AnalyticalGraphWideAutotuneOptimizer(
      optimizerConstraints(minSamples = 2L, intervalNanos = 100L))
    val shape = AutotuneStageShape(
      hasGpuScan = false, hasGpuPrefetchConsumer = true, numTasks = 50)
    val initial = optimizer.initialHint(key, AutotuneStageDescriptor(shape)).copy(version = 1L)
    val first = optimizerObservation(key, 1L, wait = 0L, holding = 1000L,
      duration = 1000L, input = 0L, output = 0L)
    val second = optimizerObservation(key, 2L, wait = 0L, holding = 1000L,
      duration = 1000L, input = 0L, output = 0L)

    assert(optimizer.observe(first, initial, nowNanos = 1000L).isEmpty)
    val changed = optimizer.observe(second, initial, nowNanos = 1000L)
    assert(changed.exists(_.hint.gpu.maxConcurrentTasks == 2))
    // A fresh two-sample window inside the interval remains queued, not evaluated.
    val updated = changed.head.hint
    assert(optimizer.observe(first, updated, nowNanos = 1050L).isEmpty)
    assert(optimizer.observe(second, updated, nowNanos = 1050L).isEmpty)
  }

  test("graph decision epochs expose replay-complete sensitivities and freeze reasons") {
    val optimizer = new AnalyticalGraphWideAutotuneOptimizer(
      optimizerConstraints(minSamples = 2L, intervalNanos = 0L))
    val shape = AutotuneStageShape(
      hasGpuScan = false, hasGpuPrefetchConsumer = true, numTasks = 50)
    val initial = optimizer.initialHint(
      key, AutotuneStageDescriptor(shape, parentStageIds = Seq(1))).copy(version = 1L)
    optimizer.hintPublished(initial)
    optimizer.stageSubmitted(key)

    val cold = optimizer.drainDecisionRecords()
    assert(cold.size == 1)
    assert(cold.head.trigger == "stage-submitted")
    assert(cold.head.freezeReasons.gpuTasks == "no-current-version-calibration")

    val first = optimizerObservation(key, 1L, wait = 1000L, holding = 100L,
      duration = 1100L, input = 0L, output = 0L)
    val second = first.copy(taskAttemptId = 2L)
    assert(optimizer.observe(first, initial, nowNanos = 1000L).isEmpty)
    assert(optimizer.observe(second, initial, nowNanos = 1000L).nonEmpty)

    val calibrated = optimizer.drainDecisionRecords()
    assert(calibrated.size == 1)
    val record = calibrated.head
    assert(record.trigger == "feedback")
    assert(record.graphSelectedObjectiveNanos == record.graphCurrentObjectiveNanos)
    assert(record.durationAdjoint == 1.0)
    assert(record.selectedControl.gpuTasks == record.currentControl.gpuTasks)
    assert(record.endToEndGradient.gpuTasks < 0.0)
    assert(record.freezeReasons.gpuTasks ==
      "single-operating-point-response-unidentified")

    val event = RapidsAutotuneDriverEndpoint.toDecisionEvent(record)
    assert(event.epochId == record.epochId)
    assert(event.parentStageIds == Seq(1))
    assert(event.selectedGpuTasks == record.selectedControl.gpuTasks)
    assert(event.gpuTasksGradient == record.endToEndGradient.gpuTasks)
    assert(event.gpuTasksFreezeReason == "single-operating-point-response-unidentified")
  }

  test("optimizer calibrates only observations from the current complete hint version") {
    val optimizer = new AnalyticalGraphWideAutotuneOptimizer(
      optimizerConstraints(minSamples = 2L, intervalNanos = 0L))
    val shape = AutotuneStageShape(
      hasGpuScan = false, hasGpuPrefetchConsumer = true, numTasks = 50)
    val initial = optimizer.initialHint(key, AutotuneStageDescriptor(shape)).copy(version = 4L)
    val stale = optimizerObservation(key, 1L, wait = 0L, holding = 1000L,
      duration = 1000L, input = 0L, output = 0L).copy(hintVersion = 3L)
    val current1 = stale.copy(taskAttemptId = 2L, hintVersion = 4L)
    val current2 = stale.copy(taskAttemptId = 3L, hintVersion = 4L)

    assert(optimizer.observe(stale, initial, nowNanos = 1000L).isEmpty)
    assert(optimizer.observe(current1, initial, nowNanos = 1000L).isEmpty)
    assert(optimizer.observe(current2, initial, nowNanos = 1000L).nonEmpty)
  }

  test("AQE calibration is isolated by SQL execution and released on completion") {
    val optimizer = new AnalyticalGraphWideAutotuneOptimizer(
      optimizerConstraints(minSamples = 1L, intervalNanos = 0L))
    val identifiedKey = key.copy(executionId = 21L)
    val boundaryKey = key.copy(executionId = 22L)
    val shape = AutotuneStageShape(
      hasGpuScan = true, hasGpuPrefetchConsumer = true, numTasks = 3)
    val identifiedHint = optimizer.initialHint(
      identifiedKey, AutotuneStageDescriptor(shape)).copy(version = 1L)
    val boundaryHint = optimizer.initialHint(
      boundaryKey, AutotuneStageDescriptor(shape)).copy(version = 1L)

    Seq((1000L, 2500L), (2000L, 4500L), (3000L, 6500L)).zipWithIndex.foreach {
      case ((bytes, duration), index) =>
        optimizer.observe(optimizerObservation(identifiedKey, index + 1L,
          wait = 0L, holding = 0L, duration = duration, input = bytes, output = bytes),
          identifiedHint, nowNanos = index.toLong)
    }
    Seq((1000L, 2000L), (2000L, 4000L), (3000L, 6000L)).zipWithIndex.foreach {
      case ((bytes, duration), index) =>
        optimizer.observe(optimizerObservation(boundaryKey, index + 1L,
          wait = 0L, holding = 0L, duration = duration, input = bytes, output = bytes),
          boundaryHint, nowNanos = index.toLong)
    }

    val identified = optimizer.aqeCalibrationSnapshot(identifiedKey.executionId).get
    val boundary = optimizer.aqeCalibrationSnapshot(boundaryKey.executionId).get
    assert(math.abs(identified.fixedNanosPerTask.get - 500.0) < 1e-9)
    assert(identified.fixedTaskCostReason.isEmpty)
    assert(boundary.fixedNanosPerTask.isEmpty)
    assert(boundary.fixedTaskCostReason == "boundary-fixed-task-cost-fit")

    optimizer.executionCompleted(identifiedKey.executionId)
    assert(optimizer.aqeCalibrationSnapshot(identifiedKey.executionId).isEmpty)
    assert(optimizer.aqeCalibrationSnapshot(boundaryKey.executionId).nonEmpty)
  }

  test("late observations do not reactivate a completed stage") {
    val optimizer = new AnalyticalGraphWideAutotuneOptimizer(
      optimizerConstraints(minSamples = 2L, intervalNanos = 0L))
    val shape = AutotuneStageShape(
      hasGpuScan = false, hasGpuPrefetchConsumer = true, numTasks = 50)
    val initial = optimizer.initialHint(key, AutotuneStageDescriptor(shape)).copy(version = 1L)
    optimizer.hintPublished(initial)
    optimizer.stageSubmitted(key)
    optimizer.stageCompleted(key)
    val first = optimizerObservation(key, 1L, wait = 0L, holding = 1000L,
      duration = 1000L, input = 0L, output = 0L)
    val second = first.copy(taskAttemptId = 2L)

    assert(optimizer.observe(first, initial, nowNanos = 1000L).isEmpty)
    assert(optimizer.observe(second, initial, nowNanos = 1000L).isEmpty)
  }

  test("optimizer retains unidentifiable batch control instead of minimizing it to one byte") {
    val constraints = optimizerConstraints()
    val shape = AutotuneStageShape(
      hasGpuScan = true, hasGpuPrefetchConsumer = true, numTasks = 50, hasShuffle = true)
    val current = StageRuntimeHint.empty(key).copy(
      scan = ScanRuntimeHint(true, 1, 2, 4096L),
      gpu = GpuRuntimeHint(2),
      shuffle = ShuffleRuntimeHint(1, 128L, 256L),
      batch = BatchRuntimeHint(256L, 1024L, 128L))
    val observation = StageObservationAgg.empty
      .merge(optimizerObservation(key, 1L, wait = 0L, holding = 1000L,
        duration = 5000L, input = 4096L, output = 4096L, shuffleWrite = 4096L))
      .merge(optimizerObservation(key, 2L, wait = 0L, holding = 1000L,
        duration = 5000L, input = 4096L, output = 4096L, shuffleWrite = 4096L))

    val curve = AnalyticalStageCostModel.costCurve(
      observation, current, shape, constraints).get
    assert(curve.alternatives.head.hint.batch.targetBatchBytes == 256L)
    assert(curve.alternatives.head.hint.batch.splitUntilSize == 128L)
  }

  test("driver publishes and republishes optimizer-owned complete hints") {
    val conf = new RapidsConf(Map(
      RapidsConf.AUTOTUNE_GRAPH_ENABLED.key -> "true",
      RapidsConf.AUTOTUNE_GRAPH_MODE.key -> AutotuneGraphMode.OPTIMIZE.toString,
      RapidsConf.AUTOTUNE_GPU_MAX_CONCURRENT_TASKS.key -> "2",
      RapidsConf.AUTOTUNE_OPTIMIZE_GPU_MAX_CONCURRENT_TASKS.key -> "8",
      RapidsConf.AUTOTUNE_GRAPH_MIN_SAMPLE_TASKS.key -> "2",
      RapidsConf.AUTOTUNE_GRAPH_UPDATE_INTERVAL_MS.key -> "0"))
    RapidsAutotuneDriverEndpoint.init(null, conf)
    try {
      val listener = new RapidsAutotuneStageHintListener(conf)
      val shape = AutotuneStageShape(
        hasGpuScan = false, hasGpuPrefetchConsumer = true, numTasks = 50)
      val initial = listener.publishHintForStage(
        key.executionId, key.stageId, key.stageAttemptId, shape, Seq(1))
      assert(initial.version == 1L && initial.gpu.maxConcurrentTasks == 2)
      RapidsAutotuneDriverEndpoint.handleObservation(
        optimizerObservation(key, 1L, wait = 0L, holding = 1000L,
          duration = 1000L, input = 0L, output = 0L))
      RapidsAutotuneDriverEndpoint.handleObservation(
        optimizerObservation(key, 2L, wait = 0L, holding = 1000L,
          duration = 1000L, input = 0L, output = 0L))
      val served = RapidsAutotuneDriverEndpoint.handleHintRequest(
        RapidsAutotuneHintRequestMsg("exec-1", key)).hint.get
      assert(served.version == 2L)
      assert(served.gpu.maxConcurrentTasks == 2)
    } finally {
      RapidsAutotuneDriverEndpoint.shutdown()
    }
  }

  test("batch runtime hint helpers clamp and fail open") {
    val batch = BatchRuntimeHint(targetBatchBytes = 1024L, maxBatchBytes = 768L,
      splitUntilSize = 900L)
    val shuffle = ShuffleRuntimeHint(prefetchWindow = 0, maxReadyBytes = 1L,
      coalesceTargetBytes = 2048L)
    assert(BatchRuntimeHints.effectiveTargetBatchBytes(256L, Some(batch)) == 768L)
    assert(BatchRuntimeHints.effectiveShuffleCoalesceTargetBytes(
      256L, Some(shuffle), Some(batch)) == 768L)
    assert(BatchRuntimeHints.effectiveSplitUntilSize(256L, Some(batch)) ==
      GpuDeviceManager.MIN_SPLIT_UNTIL_SIZE)
    assert(BatchRuntimeHints.effectiveTargetBatchBytes(256L, None) == 256L)
    assert(BatchRuntimeHints.effectiveSplitUntilSize(256L, None) == 256L)
  }
}

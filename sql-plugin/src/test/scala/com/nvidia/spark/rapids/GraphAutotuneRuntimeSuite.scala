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

  test("stage runtime hint exposes its stage key") {
    val hint = StageRuntimeHint(
      executionId = key.executionId,
      stageId = key.stageId,
      stageAttemptId = key.stageAttemptId,
      version = 7L,
      scan = ScanRuntimeHint(eagerPrefetch = true, maxReadWindow = 4, maxReadyBytes = 1024L))

    assert(hint.key == key)
  }

  test("shuffle runtime hints default to empty no-op") {
    assert(ShuffleRuntimeHint.empty.prefetchWindow == 0)
    assert(ShuffleRuntimeHint.empty.maxReadyBytes == Long.MaxValue)

    val empty = StageRuntimeHint.empty(key)
    assert(empty.shuffle == ShuffleRuntimeHint.empty)
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
      hint)
  }

  test("task hint state exposes only applied hints") {
    val scanHint = ScanRuntimeHint(
      eagerPrefetch = true,
      maxReadWindow = 4,
      maxReadyBytes = 1024L)
    val shuffleHint = ShuffleRuntimeHint(prefetchWindow = 2, maxReadyBytes = 4096L)
    val hint = AutotuneCachedHint(StageRuntimeHint(
      executionId = key.executionId,
      stageId = key.stageId,
      stageAttemptId = key.stageAttemptId,
      version = 1L,
      scan = scanHint,
      shuffle = shuffleHint), hasHint = true)

    try {
      RapidsAutotuneTaskHints.clearCurrentHint()
      assert(RapidsAutotuneTaskHints.currentScanHint.isEmpty)
      assert(RapidsAutotuneTaskHints.currentShuffleHint.isEmpty)
      RapidsAutotuneTaskHints.setCurrentHint(AutotuneCachedHint.empty(key))
      assert(RapidsAutotuneTaskHints.currentScanHint.isEmpty)
      assert(RapidsAutotuneTaskHints.currentShuffleHint.isEmpty)
      RapidsAutotuneTaskHints.setCurrentHint(hint)
      assert(RapidsAutotuneTaskHints.currentScanHint.contains(scanHint))
      assert(RapidsAutotuneTaskHints.currentShuffleHint.contains(shuffleHint))
    } finally {
      RapidsAutotuneTaskHints.clearCurrentHint()
    }
    assert(RapidsAutotuneTaskHints.currentScanHint.isEmpty)
    assert(RapidsAutotuneTaskHints.currentShuffleHint.isEmpty)
  }

  test("shuffle read actuator only tightens the static bytes-in-flight cap") {
    val staticCap = 128L * 1024 * 1024
    def hint(b: Long): Option[ShuffleRuntimeHint] =
      Some(ShuffleRuntimeHint(prefetchWindow = 0, maxReadyBytes = b))

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

  test("shuffle prefetch actuator enforces the optimizer-selected outstanding-work window") {
    val hint = Some(ShuffleRuntimeHint(prefetchWindow = 4, maxReadyBytes = 128L))
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

  test("stage shape identifies broadcast work") {
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
      RapidsConf.AUTOTUNE_SCAN_MAX_READY_BYTES.key -> "1024",
      RapidsConf.AUTOTUNE_SHUFFLE_ENABLED.key -> "true",
      RapidsConf.SHUFFLE_MULTITHREADED_MAX_BYTES_IN_FLIGHT.key -> "2048",
      RapidsConf.AUTOTUNE_SHUFFLE_MAX_READY_BYTES.key -> "2048",
      RapidsConf.AUTOTUNE_SHUFFLE_MAX_PREFETCH_WINDOW.key -> "6"))
    val optimizer = new AnalyticalGraphWideAutotuneOptimizer(
      GraphOptimizerConstraints.fromConf(conf))
    val shape = AutotuneStageShape(
      hasGpuScan = true, hasGpuPrefetchConsumer = true, numTasks = 50, hasShuffle = true)
    val hint = optimizer.initialHint(key, AutotuneStageDescriptor(shape, Seq(1, 2)))

    // Cold hints begin at deployed/native settings or the configured feasible envelope, not at
    // an unevidenced low-resource point.
    assert(hint.scan == ScanRuntimeHint(true, 8, 1024L))
    assert(hint.shuffle == ShuffleRuntimeHint(6, 2048L))
  }

  test("each autotune subsystem gates independently for mechanism ablation") {
    val conf = new RapidsConf(Map(
      RapidsConf.AUTOTUNE_GRAPH_ENABLED.key -> "true",
      RapidsConf.AUTOTUNE_GRAPH_MODE.key -> AutotuneGraphMode.OPTIMIZE.toString,
      RapidsConf.AUTOTUNE_SCAN_ENABLED.key -> "false",
      RapidsConf.AUTOTUNE_SHUFFLE_ENABLED.key -> "true"))

    val constraints = GraphOptimizerConstraints.fromConf(conf)
    assert(!constraints.scan.enabled)
    assert(constraints.shuffle.enabled)

    // A disabled subsystem publishes only the empty no-op hint for its actuator, while an
    // enabled sibling subsystem still publishes its own hint.
    val optimizer = new AnalyticalGraphWideAutotuneOptimizer(constraints)
    val shape = AutotuneStageShape(
      hasGpuScan = true, hasGpuPrefetchConsumer = true, numTasks = 200, hasShuffle = true)
    val hint = optimizer.initialHint(key, AutotuneStageDescriptor(shape))
    assert(hint.scan == ScanRuntimeHint.empty)
    assert(hint.shuffle.maxReadyBytes > 0L)
  }

  test("driver endpoint publishes stable positive no-op hints") {
    val conf = new RapidsConf(Map(RapidsConf.AUTOTUNE_GRAPH_ENABLED.key -> "true"))
    RapidsAutotuneDriverEndpoint.init(null, conf)
    try {
      val first = RapidsAutotuneDriverEndpoint.publishStageHint(key, ScanRuntimeHint.empty)
      val again = RapidsAutotuneDriverEndpoint.publishStageHint(key, ScanRuntimeHint.empty)
      val secondKey = key.copy(stageId = 4)
      val second = RapidsAutotuneDriverEndpoint.publishStageHint(secondKey, ScanRuntimeHint.empty)

      assert(first.version == 1L)
      assert(first == again)
      assert(second.version == 2L)
      assert(!first.scan.eagerPrefetch)
      assert(first.scan.maxReadWindow == 0)
      assert(first.shuffle == ShuffleRuntimeHint.empty)

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

  test("stage hint listener publishes no-op hints for non-candidate stages") {
    val conf = new RapidsConf(Map(RapidsConf.AUTOTUNE_GRAPH_ENABLED.key -> "true"))
    RapidsAutotuneDriverEndpoint.init(null, conf)
    try {
      val listener = new RapidsAutotuneStageHintListener(conf)
      val noopShape =
        AutotuneStageShape(hasGpuScan = false, hasGpuPrefetchConsumer = false, numTasks = 0)
      listener.publishHintForStage(key.executionId, key.stageId, key.stageAttemptId, noopShape)
      listener.publishHintForStage(key.executionId, 4, 0, noopShape)

      val first = RapidsAutotuneDriverEndpoint.handleHintRequest(
        RapidsAutotuneHintRequestMsg("exec-1", key)).hint
      val second = RapidsAutotuneDriverEndpoint.handleHintRequest(
        RapidsAutotuneHintRequestMsg("exec-1", key.copy(stageId = 4, stageAttemptId = 0))).hint

      assert(first.exists(_.version == 1L))
      assert(second.exists(_.version == 2L))
      assert(first.exists(_.scan == ScanRuntimeHint.empty))
      assert(second.exists(_.scan == ScanRuntimeHint.empty))
      assert(first.exists(_.shuffle == ShuffleRuntimeHint.empty))
      assert(second.exists(_.shuffle == ShuffleRuntimeHint.empty))
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
      RapidsConf.AUTOTUNE_SCAN_MAX_READY_BYTES.key -> "1024"))
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
        maxReadWindow = 4,
        maxReadyBytes = 1024L))

      val response = RapidsAutotuneDriverEndpoint.handleHintRequest(
        RapidsAutotuneHintRequestMsg("exec-1", key))
      assert(response.hint.contains(hint))
    } finally {
      RapidsAutotuneDriverEndpoint.shutdown()
    }
  }

  test("executor endpoint reports the applied-hint payload to the driver") {
    val ctx = new CapturingPluginContext
    val conf = new RapidsConf(Map(RapidsConf.AUTOTUNE_FAIL_OPEN.key -> "true"))
    val endpoint = new RapidsAutotuneExecutorEndpoint(ctx, conf)
    val scanHint = ScanRuntimeHint(
      eagerPrefetch = true, maxReadWindow = 4, maxReadyBytes = 2048L)
    val shuffleHint = ShuffleRuntimeHint(prefetchWindow = 6, maxReadyBytes = 4096L)
    val cached = AutotuneCachedHint(StageRuntimeHint.empty(key).copy(
      version = 9L, scan = scanHint, shuffle = shuffleHint),
      hasHint = true)

    endpoint.recordAppliedHint(key, taskAttemptId = 42L, partitionId = 5, cached)

    ctx.lastSent match {
      case msg: RapidsAutotuneHintAppliedMsg =>
        assert(msg.executorId == "exec-7")
        assert(msg.key == key)
        assert(msg.taskAttemptId == 42L)
        assert(msg.partitionId == 5)
        assert(msg.hintVersion == 9L)
        assert(msg.hasHint)
        assert(msg.scan == scanHint)
        assert(msg.shuffle == shuffleHint)
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
        eagerPrefetch = true, maxReadWindow = 4, maxReadyBytes = 2048L),
      shuffle = ShuffleRuntimeHint(prefetchWindow = 6, maxReadyBytes = 4096L))

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
    assert(event.scanMaxReadWindow == 4)
    assert(event.scanMaxReadyBytes == 2048L)
    assert(event.shufflePrefetchWindow == 6)
    assert(event.shuffleMaxReadyBytes == 4096L)
  }

  test("stage hint listener publishes bounded shuffle hints for shuffle stages") {
    val conf = new RapidsConf(Map(
      RapidsConf.AUTOTUNE_GRAPH_ENABLED.key -> "true",
      RapidsConf.AUTOTUNE_GRAPH_MODE.key -> AutotuneGraphMode.GRAPH.toString,
      RapidsConf.AUTOTUNE_SHUFFLE_ENABLED.key -> "true",
      RapidsConf.AUTOTUNE_SHUFFLE_MAX_PREFETCH_WINDOW.key -> "6",
      RapidsConf.AUTOTUNE_SHUFFLE_MAX_READY_BYTES.key -> "2048"))
    RapidsAutotuneDriverEndpoint.init(null, conf)
    try {
      val listener = new RapidsAutotuneStageHintListener(conf)
      val shuffleStage = AutotuneStageShape(
        hasGpuScan = false, hasGpuPrefetchConsumer = false, numTasks = 50, hasShuffle = true)
      val hint = listener.publishHintForStage(
        key.executionId, key.stageId, key.stageAttemptId, shuffleStage)

      assert(hint.version == 1L)
      assert(hint.shuffle == ShuffleRuntimeHint(prefetchWindow = 6, maxReadyBytes = 2048L))
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
  }

  test("task metrics record multithreaded shuffle limiter pressure") {
    val metrics = new GpuTaskMetrics
    metrics.recordShuffleReadLimiterAcquire(didFit = true)
    metrics.recordShuffleReadLimiterAcquire(didFit = false)
    assert(metrics.getShuffleReadLimiterAcquireCount == 2L)
    assert(metrics.getShuffleReadLimiterAcquireFailCount == 1L)
  }

  test("driver flattens an observation into the eventlog record") {
    val m1 = RapidsAutotuneObservationMsg("exec-1", key, 1L, 0, 1L, 10L, 100L, 500L,
      shuffleReadLimiterAcquireCount = 40L, shuffleReadLimiterAcquireFailCount = 4L)
    val m2 = RapidsAutotuneObservationMsg("exec-1", key, 2L, 1, 1L, 30L, 100L, 800L,
      shuffleReadLimiterAcquireCount = 60L, shuffleReadLimiterAcquireFailCount = 16L)
    val agg = StageObservationAgg.empty.merge(m1).merge(m2)

    val ev = RapidsAutotuneDriverEndpoint.toObservationEvent(m2, agg)
    assert(ev.executionId == key.executionId && ev.stageId == key.stageId)
    assert(ev.taskAttemptId == 2L && ev.partitionId == 1 && ev.gpuSemaphoreWaitNanos == 30L)
    assert(ev.hostMemoryBytes == 800L && ev.stageTaskCount == 2L)
    assert(ev.stageMaxHostMemoryBytes == 800L)
    assert(ev.shuffleReadLimiterAcquireCount == 60L)
    assert(ev.shuffleReadLimiterAcquireFailCount == 16L)
    assert(ev.stageShuffleReadLimiterAcquireCount == 100L)
    assert(ev.stageShuffleReadLimiterAcquireFailCount == 20L)
  }

  test("executor endpoint reports a runtime observation to the driver") {
    val ctx = new CapturingPluginContext
    val conf = new RapidsConf(Map(RapidsConf.AUTOTUNE_FAIL_OPEN.key -> "true"))
    val endpoint = new RapidsAutotuneExecutorEndpoint(ctx, conf)
    endpoint.reportObservation(RapidsAutotuneObservationMsg(
      executorId = endpoint.executorId, key = key, taskAttemptId = 7L, partitionId = 3,
      hintVersion = 5L, gpuSemaphoreWaitNanos = 11L, gpuHoldingNanos = 22L,
      hostMemoryBytes = 33L, spillBytes = 44L,
      shuffleReadLimiterAcquireCount = 55L, shuffleReadLimiterAcquireFailCount = 6L))
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
    endpoint.reportObservation(RapidsAutotuneObservationMsg(
      executorId = endpoint.executorId, key = key, taskAttemptId = 1L, partitionId = 0,
      hintVersion = 1L, gpuSemaphoreWaitNanos = 5L, gpuHoldingNanos = 10L,
      hostMemoryBytes = 20L, spillBytes = 30L))
  }

  // ---------------------------------------------------------------------------
  // Graph-wide analytical optimizer decision epochs
  // ---------------------------------------------------------------------------

  private def optimizerConstraints(
      minSamples: Long = 2L,
      intervalNanos: Long = 0L): GraphOptimizerConstraints = GraphOptimizerConstraints(
    minSampleTasks = minSamples,
    updateIntervalNanos = intervalNanos,
    scan = ScanOptimizerBounds(enabled = true, initialReadWindow = 2, maxReadyBytes = 4096L),
    shuffle = ShuffleOptimizerBounds(enabled = true, initialPrefetchWindow = 1,
      initialReadyBytes = 128L))

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

  test("single-setting feedback freezes joint executor controls at the measured point") {
    val constraints = optimizerConstraints()
    val shape = AutotuneStageShape(
      hasGpuScan = true, hasGpuPrefetchConsumer = true, numTasks = 50, hasShuffle = true)
    val current = StageRuntimeHint.empty(key).copy(
      scan = ScanRuntimeHint(true, 2, 4096L),
      shuffle = ShuffleRuntimeHint(1, 128L))
    val observation = StageObservationAgg.empty
      .merge(optimizerObservation(key, 1L, wait = 0L, holding = 2000L,
        duration = 10000L, input = 1000L, output = 500L, shuffleRead = 500L))
      .merge(optimizerObservation(key, 2L, wait = 0L, holding = 2000L,
        duration = 10000L, input = 1000L, output = 500L, shuffleRead = 500L))

    val curve = AnalyticalStageCostModel.costCurve(
      observation, current, shape, constraints).get
    assert(curve.predictedCurrentNanos > 0.0)
    // The calibrated control is the measured operating point of the current hint.
    assert(curve.currentControl.scanWindow == 2.0)
    assert(curve.currentControl.shuffleWindow == 1.0)
    assert(curve.currentControl.shuffleBytes == 128.0)
    val reasons = curve.freezeReasons
    assert(reasons.scanWindow == "single-operating-point-response-unidentified")
    assert(reasons.shuffleWindow == "single-operating-point-response-unidentified")
    assert(reasons.shuffleBytes == "single-operating-point-response-unidentified")
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

    // An under-sampled window is queued without an evaluation epoch.
    optimizer.observe(first, initial, nowNanos = 1000L)
    assert(optimizer.drainDecisionRecords().isEmpty)
    // The second sample completes the window and triggers a feedback decision epoch.
    optimizer.observe(second, initial, nowNanos = 1000L)
    val evaluated = optimizer.drainDecisionRecords()
    assert(evaluated.nonEmpty)
    assert(evaluated.forall(_.trigger == "feedback"))
    // A fresh two-sample window inside the interval remains queued, not evaluated.
    optimizer.observe(first, initial, nowNanos = 1050L)
    optimizer.observe(second, initial, nowNanos = 1050L)
    assert(optimizer.drainDecisionRecords().isEmpty)
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
    assert(cold.head.freezeReasons.scanWindow == "no-current-version-calibration")

    val first = optimizerObservation(key, 1L, wait = 1000L, holding = 100L,
      duration = 1100L, input = 0L, output = 0L)
    val second = first.copy(taskAttemptId = 2L)
    optimizer.observe(first, initial, nowNanos = 1000L)
    optimizer.observe(second, initial, nowNanos = 1000L)

    val calibrated = optimizer.drainDecisionRecords()
    assert(calibrated.size == 1)
    val record = calibrated.head
    assert(record.trigger == "feedback")
    assert(record.graphCurrentObjectiveNanos > 0.0)
    assert(record.durationAdjoint == 1.0)
    // GPU service time is still measured and reported through its lane gradient even though
    // there is no gpu actuator control.
    assert(record.endToEndGradient.gpu < 0.0)

    val event = RapidsAutotuneDriverEndpoint.toDecisionEvent(record)
    assert(event.epochId == record.epochId)
    assert(event.parentStageIds == Seq(1))
    assert(event.gpuGradient == record.endToEndGradient.gpu)
    assert(event.scanGradient == record.endToEndGradient.scan)
    assert(event.shuffleGradient == record.endToEndGradient.shuffle)
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

    // The stale-version observation never counts toward the current window's sample budget.
    optimizer.observe(stale, initial, nowNanos = 1000L)
    optimizer.observe(current1, initial, nowNanos = 1000L)
    assert(optimizer.drainDecisionRecords().isEmpty)
    optimizer.observe(current2, initial, nowNanos = 1000L)
    assert(optimizer.drainDecisionRecords().nonEmpty)
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

  test("AQE calibration tracks the largest observed per-task shuffle-read load") {
    val optimizer = new AnalyticalGraphWideAutotuneOptimizer(
      optimizerConstraints(minSamples = 1L, intervalNanos = 0L))
    val shape = AutotuneStageShape(
      hasGpuScan = false, hasGpuPrefetchConsumer = true, numTasks = 4, hasShuffle = true)
    val hint = optimizer.initialHint(key, AutotuneStageDescriptor(shape)).copy(version = 1L)
    optimizer.observe(optimizerObservation(key, 1L, wait = 0L, holding = 0L,
      duration = 2000L, input = 512L, output = 0L, shuffleRead = 512L), hint, nowNanos = 1L)
    optimizer.observe(optimizerObservation(key, 2L, wait = 0L, holding = 0L,
      duration = 2000L, input = 2048L, output = 0L, shuffleRead = 2048L), hint, nowNanos = 2L)

    val calibration = optimizer.aqeCalibrationSnapshot(key.executionId).get
    assert(calibration.maxCalibratedTaskShuffleReadBytes == 2048L)
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
    optimizer.drainDecisionRecords()
    val first = optimizerObservation(key, 1L, wait = 0L, holding = 1000L,
      duration = 1000L, input = 0L, output = 0L)
    val second = first.copy(taskAttemptId = 2L)

    optimizer.observe(first, initial, nowNanos = 1000L)
    optimizer.observe(second, initial, nowNanos = 1000L)
    val late = optimizer.drainDecisionRecords()
    assert(late.nonEmpty)
    assert(late.forall(record => record.completed && !record.active))
  }

}

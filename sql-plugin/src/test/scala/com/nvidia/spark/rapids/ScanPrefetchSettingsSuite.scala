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

import java.util.concurrent.TimeUnit

import org.apache.hadoop.conf.Configuration
import org.scalatest.funsuite.AnyFunSuite

class ScanPrefetchSettingsSuite extends AnyFunSuite {

  test("initial fanout defaults to the static reader cap") {
    val conf = new Configuration(false)
    assert(ScanPrefetchSettings.initialFanout(conf, maxNumFileProcessed = 4,
      inputFileCount = 10) == 4)
    assert(ScanPrefetchSettings.initialFanout(conf, maxNumFileProcessed = 4,
      inputFileCount = 3) == 3)
  }

  test("initial fanout honors a lower prefetch window") {
    val conf = new Configuration(false)
    conf.setInt(ScanPrefetchSettings.WINDOW_KEY, 2)
    assert(ScanPrefetchSettings.initialFanout(conf, maxNumFileProcessed = 4,
      inputFileCount = 10) == 2)
  }

  test("initial fanout never exceeds the static reader cap") {
    val conf = new Configuration(false)
    conf.setInt(ScanPrefetchSettings.WINDOW_KEY, 8)
    assert(ScanPrefetchSettings.initialFanout(conf, maxNumFileProcessed = 4,
      inputFileCount = 10) == 4)
  }

  test("invalid Hadoop prefetch window falls back to the static reader cap") {
    val conf = new Configuration(false)
    conf.setInt(ScanPrefetchSettings.WINDOW_KEY, 0)
    assert(ScanPrefetchSettings.initialFanout(conf, maxNumFileProcessed = 4,
      inputFileCount = 10) == 4)
  }

  test("public scan prefetch configs parse and validate") {
    val conf = new RapidsConf(Map(
      RapidsConf.SCAN_PREFETCH_ENABLED.key -> "all",
      RapidsConf.SCAN_PREFETCH_MAX_PARALLELISM.key -> "3",
      RapidsConf.SCAN_PREFETCH_MIN_SCAN_FILES.key -> "2",
      RapidsConf.AUTOTUNE_GRAPH_ENABLED.key -> "true",
      RapidsConf.AUTOTUNE_GRAPH_MODE.key -> "local",
      RapidsConf.AUTOTUNE_SCAN_MAX_READ_WINDOW.key -> "4",
      RapidsConf.AUTOTUNE_SCAN_MAX_READY_BYTES.key -> "16m",
      RapidsConf.AUTOTUNE_FAIL_OPEN.key -> "true"))
    assert(conf.scanPrefetchMode == ScanPrefetchMode.ALL)
    assert(conf.scanPrefetchMaxParallelism == 3)
    assert(conf.scanPrefetchMinScanFiles == 2)
    assert(conf.autotuneGraphEnabled)
    assert(conf.autotuneGraphMode == AutotuneGraphMode.LOCAL)
    assert(conf.isAutotuneLocalMode)
    assert(conf.autotuneScanMaxReadWindow == 4)
    assert(conf.autotuneScanMaxReadyBytes == 16L * 1024L * 1024L)
    assert(conf.autotuneFailOpen)

    assertThrows[IllegalArgumentException] {
      new RapidsConf(Map(RapidsConf.SCAN_PREFETCH_ENABLED.key -> "sometimes"))
        .scanPrefetchMode
    }
    assertThrows[IllegalArgumentException] {
      new RapidsConf(Map(RapidsConf.SCAN_PREFETCH_MAX_PARALLELISM.key -> "0"))
        .scanPrefetchMaxParallelism
    }
    assertThrows[IllegalArgumentException] {
      new RapidsConf(Map(RapidsConf.AUTOTUNE_GRAPH_MODE.key -> "driver"))
        .autotuneGraphMode
    }
    assertThrows[IllegalArgumentException] {
      new RapidsConf(Map(RapidsConf.AUTOTUNE_SCAN_MAX_READ_WINDOW.key -> "0"))
        .autotuneScanMaxReadWindow
    }
  }

  test("scan read window settings preserve default fanout when disabled") {
    val conf = new Configuration(false)
    conf.setInt(ScanPrefetchSettings.WINDOW_KEY, 2)
    val settings = ScanReadWindowSettings.fromConf(conf, maxNumFileProcessed = 4,
      inputFileCount = 10)
    assert(!settings.enabled)
    assert(settings.initialWindow == 2)
    assert(settings.maxWindow == 2)
  }

  test("scan read window settings honor initial and max caps when enabled") {
    val conf = new Configuration(false)
    conf.setBoolean(ScanReadWindowSettings.ENABLED_KEY, true)
    conf.setInt(ScanReadWindowSettings.INITIAL_WINDOW_KEY, 2)
    conf.setInt(ScanReadWindowSettings.MAX_WINDOW_KEY, 8)
    conf.setLong(ScanReadWindowSettings.MAX_READY_BYTES_KEY, 1024L)
    val settings = ScanReadWindowSettings.fromConf(conf, maxNumFileProcessed = 4,
      inputFileCount = 3)
    assert(settings.enabled)
    assert(settings.initialWindow == 2)
    assert(settings.maxWindow == 3)
    assert(settings.maxReadyBytes == 1024L)
  }

  test("scan read window settings honor graph scan hint caps") {
    val hint = ScanRuntimeHint(
      eagerPrefetch = true,
      minReadWindow = 2,
      maxReadWindow = 8,
      maxReadyBytes = 1024L)
    val settings = ScanReadWindowSettings.fromHint(
      hint, maxReadWindowCap = 4, inputFileCount = 3)

    assert(settings.contains(ScanReadWindowSettings(
      enabled = true,
      initialWindow = 2,
      maxWindow = 3,
      maxReadyBytes = 1024L)))

    assert(ScanReadWindowSettings.fromHint(
      hint.copy(maxReadWindow = 0), maxReadWindowCap = 4, inputFileCount = 3).isEmpty)
    assert(ScanReadWindowSettings.fromHint(
      hint, maxReadWindowCap = 0, inputFileCount = 3).isEmpty)
    assert(ScanReadWindowSettings.fromHint(
      hint, maxReadWindowCap = 4, inputFileCount = 0).isEmpty)
  }

  test("scan read window controller increases on consumer read wait") {
    val settings = ScanReadWindowSettings(
      enabled = true,
      initialWindow = 1,
      maxWindow = 3,
      maxReadyBytes = Long.MaxValue)
    val controller = new ScanReadWindowController(settings)
    controller.observeReadWait(TimeUnit.MILLISECONDS.toNanos(2), 0L, 0L, 0L)
    assert(controller.currentReadWindow == 2)
    controller.observeReadWait(TimeUnit.MILLISECONDS.toNanos(2), 0L, 0L, 0L)
    assert(controller.currentReadWindow == 3)
    controller.observeReadWait(TimeUnit.MILLISECONDS.toNanos(2), 0L, 0L, 0L)
    assert(controller.currentReadWindow == 3)
  }

  test("scan read window controller decreases on memory or schedule pressure") {
    val settings = ScanReadWindowSettings(
      enabled = true,
      initialWindow = 4,
      maxWindow = 8,
      maxReadyBytes = 100L)
    val controller = new ScanReadWindowController(settings)
    controller.observeReadWait(0L, 0L, 0L, 101L)
    assert(controller.currentReadWindow == 2)
    controller.observeReadWait(1L, 0L, 2L, 0L)
    assert(controller.currentReadWindow == 1)
  }

  test("scan read window controller records decision metrics") {
    val metricKeys = Seq(
      GpuMetric.SCAN_READ_WINDOW_INITIAL,
      GpuMetric.SCAN_READ_WINDOW_CURRENT,
      GpuMetric.SCAN_READ_WINDOW_MAX,
      GpuMetric.SCAN_READ_IN_FLIGHT_MAX,
      GpuMetric.SCAN_READ_READY_MAX,
      GpuMetric.SCAN_READ_BACKLOG_MAX,
      GpuMetric.SCAN_READ_WINDOW_INCREASES,
      GpuMetric.SCAN_READ_WINDOW_DECREASES,
      GpuMetric.SCAN_READ_WINDOW_MEMORY_DECREASES,
      GpuMetric.SCAN_READ_WINDOW_SCHEDULE_DECREASES)
    val metrics = metricKeys.map(_ -> new LocalGpuMetric()).toMap
    val settings = ScanReadWindowSettings(
      enabled = true,
      initialWindow = 1,
      maxWindow = 4,
      maxReadyBytes = 100L)
    val controller = new ScanReadWindowController(settings, metrics)

    assert(metrics(GpuMetric.SCAN_READ_WINDOW_INITIAL).value == 1)
    assert(metrics(GpuMetric.SCAN_READ_WINDOW_CURRENT).value == 1)
    assert(metrics(GpuMetric.SCAN_READ_WINDOW_MAX).value == 1)

    controller.observeReadQueue(inFlightReadTasks = 2, readyReadTasks = 1,
      backlogReadTasks = 5)
    controller.observeReadQueue(inFlightReadTasks = 1, readyReadTasks = 3,
      backlogReadTasks = 2)
    assert(metrics(GpuMetric.SCAN_READ_IN_FLIGHT_MAX).value == 2)
    assert(metrics(GpuMetric.SCAN_READ_READY_MAX).value == 3)
    assert(metrics(GpuMetric.SCAN_READ_BACKLOG_MAX).value == 5)

    controller.observeReadWait(TimeUnit.MILLISECONDS.toNanos(2), 0L, 0L, 0L)
    assert(controller.currentReadWindow == 2)
    assert(metrics(GpuMetric.SCAN_READ_WINDOW_CURRENT).value == 2)
    assert(metrics(GpuMetric.SCAN_READ_WINDOW_MAX).value == 2)
    assert(metrics(GpuMetric.SCAN_READ_WINDOW_INCREASES).value == 1)

    controller.observeReadWait(0L, 0L, 1L, 101L)
    assert(controller.currentReadWindow == 1)
    assert(metrics(GpuMetric.SCAN_READ_WINDOW_CURRENT).value == 1)
    assert(metrics(GpuMetric.SCAN_READ_WINDOW_DECREASES).value == 1)
    assert(metrics(GpuMetric.SCAN_READ_WINDOW_MEMORY_DECREASES).value == 1)
    assert(metrics(GpuMetric.SCAN_READ_WINDOW_SCHEDULE_DECREASES).value == 1)
  }

  test("disabled scan read window controller does not record decision metrics") {
    val metricKeys = Seq(
      GpuMetric.SCAN_READ_WINDOW_INITIAL,
      GpuMetric.SCAN_READ_WINDOW_CURRENT,
      GpuMetric.SCAN_READ_WINDOW_MAX,
      GpuMetric.SCAN_READ_IN_FLIGHT_MAX,
      GpuMetric.SCAN_READ_READY_MAX,
      GpuMetric.SCAN_READ_BACKLOG_MAX)
    val metrics = metricKeys.map(_ -> new LocalGpuMetric()).toMap
    val settings = ScanReadWindowSettings(
      enabled = false,
      initialWindow = 4,
      maxWindow = 4,
      maxReadyBytes = Long.MaxValue)
    val controller = new ScanReadWindowController(settings, metrics)

    controller.observeReadQueue(inFlightReadTasks = 2, readyReadTasks = 1,
      backlogReadTasks = 5)

    metricKeys.foreach { key =>
      assert(metrics(key).value == 0)
    }
  }

  test("disabled read window controller reports the legacy static fanout as its window") {
    val conf = new Configuration(false)
    // Feature off and no prefetch window set: the controller must reproduce the pre-autotune
    // initial fanout of min(maxNumFileProcessed, inputFileCount) and stay inert.
    val settings = ScanReadWindowSettings.fromConf(
      conf, maxNumFileProcessed = 4, inputFileCount = 10)
    assert(!settings.enabled)
    val controller = new ScanReadWindowController(settings)
    assert(!controller.enabled)
    assert(controller.initialReadWindow == 4)
    assert(controller.currentReadWindow == 4)

    val fewerFiles = ScanReadWindowSettings.fromConf(
      conf, maxNumFileProcessed = 4, inputFileCount = 3)
    assert(new ScanReadWindowController(fewerFiles).initialReadWindow == 3)
  }

  test("scan read window controller cools down after a decrease before re-increasing") {
    val settings = ScanReadWindowSettings(
      enabled = true,
      initialWindow = 4,
      maxWindow = 8,
      maxReadyBytes = 100L)
    val controller = new ScanReadWindowController(settings)

    // Memory pressure halves the window and arms a one-read cooldown.
    controller.observeReadWait(0L, 0L, 0L, 101L)
    assert(controller.currentReadWindow == 2)
    // The next observation is an increase signal but is swallowed by the cooldown.
    controller.observeReadWait(TimeUnit.MILLISECONDS.toNanos(2), 0L, 0L, 0L)
    assert(controller.currentReadWindow == 2)
    // Cooldown elapsed: the window can grow again.
    controller.observeReadWait(TimeUnit.MILLISECONDS.toNanos(2), 0L, 0L, 0L)
    assert(controller.currentReadWindow == 3)
  }

  test("scan read window controller increases on GPU-idle wait without buffer wait") {
    val settings = ScanReadWindowSettings(
      enabled = true,
      initialWindow = 1,
      maxWindow = 3,
      maxReadyBytes = Long.MaxValue)
    val controller = new ScanReadWindowController(settings)

    controller.observeReadWait(bufferWaitNs = 0L, bufferGpuIdleNs = 5L, scheduleWaitNs = 0L,
      hostBytesAllocated = 0L)
    assert(controller.currentReadWindow == 2)
  }
}

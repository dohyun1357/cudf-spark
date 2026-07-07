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
      RapidsConf.AUTOTUNE_GRAPH_MODE.key -> "graph",
      RapidsConf.AUTOTUNE_SCAN_MAX_READ_WINDOW.key -> "4",
      RapidsConf.AUTOTUNE_SCAN_MAX_READY_BYTES.key -> "16m",
      RapidsConf.AUTOTUNE_FAIL_OPEN.key -> "true"))
    assert(conf.scanPrefetchMode == ScanPrefetchMode.ALL)
    assert(conf.scanPrefetchMaxParallelism == 3)
    assert(conf.scanPrefetchMinScanFiles == 2)
    assert(conf.autotuneGraphEnabled)
    assert(conf.autotuneGraphMode == AutotuneGraphMode.GRAPH)
    assert(conf.isAutotuneGraphMode && conf.isAutotuneClosedLoopMode)
    assert(conf.autotuneScanMaxReadWindow == 4)
    // The effective cap is the tighter of the autotune and general prefetch caps.
    assert(conf.autotuneScanReadWindowCap == 3)
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
    assert(settings.readWindow == 2)
  }

  test("scan read window settings honor graph scan hint caps") {
    val hint = ScanRuntimeHint(
      eagerPrefetch = true,
      maxReadWindow = 8,
      maxReadyBytes = 1024L)
    val settings = ScanReadWindowSettings.fromHint(
      hint, maxReadWindowCap = 4, inputFileCount = 3)

    assert(settings.contains(ScanReadWindowSettings(enabled = true, readWindow = 3)))

    assert(ScanReadWindowSettings.fromHint(
      hint.copy(maxReadWindow = 0), maxReadWindowCap = 4, inputFileCount = 3).isEmpty)
    assert(ScanReadWindowSettings.fromHint(
      hint, maxReadWindowCap = 0, inputFileCount = 3).isEmpty)
    assert(ScanReadWindowSettings.fromHint(
      hint, maxReadWindowCap = 4, inputFileCount = 0).isEmpty)
  }

  test("graph scan hints apply a fixed optimizer-selected target") {
    val settings = ScanReadWindowSettings.fromHint(
      ScanRuntimeHint(true, maxReadWindow = 8, maxReadyBytes = 1024L),
      maxReadWindowCap = 8,
      inputFileCount = 6).get
    val controller = new ScanReadWindowController(settings)
    assert(controller.enabled)
    assert(controller.readWindow == 6)
  }

  test("scan read window controller records queue-depth evidence metrics") {
    val metricKeys = Seq(
      GpuMetric.SCAN_READ_WINDOW,
      GpuMetric.SCAN_READ_IN_FLIGHT_MAX,
      GpuMetric.SCAN_READ_READY_MAX,
      GpuMetric.SCAN_READ_BACKLOG_MAX)
    val metrics = metricKeys.map(_ -> new LocalGpuMetric()).toMap
    val settings = ScanReadWindowSettings(enabled = true, readWindow = 4)
    val controller = new ScanReadWindowController(settings, metrics)

    assert(metrics(GpuMetric.SCAN_READ_WINDOW).value == 4)

    controller.observeReadQueue(inFlightReadTasks = 2, readyReadTasks = 1,
      backlogReadTasks = 5)
    controller.observeReadQueue(inFlightReadTasks = 1, readyReadTasks = 3,
      backlogReadTasks = 2)
    assert(metrics(GpuMetric.SCAN_READ_IN_FLIGHT_MAX).value == 2)
    assert(metrics(GpuMetric.SCAN_READ_READY_MAX).value == 3)
    assert(metrics(GpuMetric.SCAN_READ_BACKLOG_MAX).value == 5)
  }

  test("disabled scan read window controller does not record metrics") {
    val metricKeys = Seq(
      GpuMetric.SCAN_READ_WINDOW,
      GpuMetric.SCAN_READ_IN_FLIGHT_MAX,
      GpuMetric.SCAN_READ_READY_MAX,
      GpuMetric.SCAN_READ_BACKLOG_MAX)
    val metrics = metricKeys.map(_ -> new LocalGpuMetric()).toMap
    val settings = ScanReadWindowSettings(enabled = false, readWindow = 4)
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
    assert(controller.readWindow == 4)

    val fewerFiles = ScanReadWindowSettings.fromConf(
      conf, maxNumFileProcessed = 4, inputFileCount = 3)
    assert(new ScanReadWindowController(fewerFiles).readWindow == 3)
  }
}

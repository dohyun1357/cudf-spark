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
}

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
      RapidsConf.SCAN_PREFETCH_MIN_SCAN_FILES.key -> "2"))
    assert(conf.scanPrefetchMode == ScanPrefetchMode.ALL)
    assert(conf.scanPrefetchMaxParallelism == 3)
    assert(conf.scanPrefetchMinScanFiles == 2)

    assertThrows[IllegalArgumentException] {
      new RapidsConf(Map(RapidsConf.SCAN_PREFETCH_ENABLED.key -> "sometimes"))
        .scanPrefetchMode
    }
    assertThrows[IllegalArgumentException] {
      new RapidsConf(Map(RapidsConf.SCAN_PREFETCH_MAX_PARALLELISM.key -> "0"))
        .scanPrefetchMaxParallelism
    }
  }
}

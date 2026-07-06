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

import com.nvidia.spark.rapids.GpuMetric._
import org.apache.hadoop.conf.Configuration

/**
 * Executor-local scan read-window actuator for the graph-wide autotuner.
 *
 * These types are the seam between the autotune hint/config layer and the multi-file readers in
 * `GpuMultiFileReader.scala`. They are kept out of that (very large) file so the autotune surface
 * area is easy to find and evolve. Three pieces:
 *
 *   - [[ScanPrefetchSettings]]: the eager-prefetch Hadoop-conf keys and the initial-fanout
 *     computation shared by the Parquet/ORC/Avro cloud readers.
 *   - [[ScanReadWindowSettings]]: the resolved read window a reader runs under, either the
 *     pre-existing static fanout ([[ScanReadWindowSettings.fromConf]]) or a driver
 *     [[ScanRuntimeHint]] ([[ScanReadWindowSettings.fromHint]], GRAPH/OPTIMIZE modes).
 *   - [[ScanReadWindowController]]: applies the fixed selected window and records queue-depth
 *     evidence metrics. The driver optimizer owns tuning; no executor-local tuner exists.
 *
 * The whole feature is default-off: without a hint the settings collapse to the pre-existing
 * static fanout and the controller is inert, so reader behavior is unchanged.
 */
object ScanPrefetchSettings {
  val ENABLED_KEY = "rapids.sql.scan.prefetch"
  val WINDOW_KEY = "rapids.sql.scan.prefetch.window"

  /**
   * The number of async read tasks a multi-file cloud reader submits up front. This is the
   * pre-existing `min(maxNumFileProcessed, inputFileCount)` static fanout, optionally lowered by a
   * prefetch window set in the Hadoop conf ([[WINDOW_KEY]]). A non-positive configured window
   * falls back to `maxNumFileProcessed`, so the result never exceeds the static reader cap.
   */
  def initialFanout(conf: Configuration, maxNumFileProcessed: Int, inputFileCount: Int): Int = {
    val requestedWindow = conf.getInt(WINDOW_KEY, maxNumFileProcessed)
    val prefetchWindow =
      if (requestedWindow > 0) requestedWindow else maxNumFileProcessed
    Seq(maxNumFileProcessed, inputFileCount, prefetchWindow).min
  }
}

/**
 * Resolved read window for a single multi-file reader. When `enabled` is false the controller is
 * inert and `readWindow` is the pre-existing static initial fanout.
 */
case class ScanReadWindowSettings(
    enabled: Boolean,
    readWindow: Int)

object ScanReadWindowSettings {
  val MIN_WINDOW = 1

  /** Inert all-zero settings for readers that consult the window only when enabled. */
  val disabled: ScanReadWindowSettings = ScanReadWindowSettings(enabled = false, readWindow = 0)

  /** Inert settings preserving the reader's static initial fanout as the window. */
  def fromConf(
      conf: Configuration,
      maxNumFileProcessed: Int,
      inputFileCount: Int): ScanReadWindowSettings = {
    ScanReadWindowSettings(
      enabled = false,
      readWindow = ScanPrefetchSettings.initialFanout(conf, maxNumFileProcessed, inputFileCount))
  }

  /**
   * The optimizer-selected fixed read window for this task, bounded by the static cap and the
   * input file count. In graph modes `maxReadWindow` is the optimizer-selected target, not an
   * envelope for a second executor-local tuner; it applies immediately.
   */
  def fromHint(
      hint: ScanRuntimeHint,
      maxReadWindowCap: Int,
      inputFileCount: Int): Option[ScanReadWindowSettings] = {
    if (hint.maxReadWindow <= 0 || maxReadWindowCap <= 0 || inputFileCount <= 0) {
      None
    } else {
      Some(ScanReadWindowSettings(
        enabled = true,
        readWindow = Seq(maxReadWindowCap, inputFileCount, hint.maxReadWindow).min))
    }
  }
}

/**
 * Per-reader application of the scan read window plus queue-depth evidence.
 *
 * Threading: a controller instance is owned by one reader and only touched from that reader's
 * task thread (the async read runners never call back in), so its mutable `var`s need no locking.
 *
 * Metric note: the gauge-style metrics (window, in-flight/ready/backlog max) are written as
 * per-task absolute values into Spark SUM accumulators via [[updateMetricValue]]. Spark sums each
 * task's final value across the tasks of a scan node, so the per-node value shown in the
 * UI/eventlog is the sum of per-task values (sum-of-per-task-maxima for the *_MAX family), not a
 * global gauge/max. Offline tooling (`performance/autotuner`) re-aggregates these per scan.
 */
class ScanReadWindowController(
    settings: ScanReadWindowSettings,
    metrics: Map[String, GpuMetric] = Map.empty) {
  private var maxObservedInFlightReads = 0
  private var maxObservedReadyReads = 0
  private var maxObservedBacklogReads = 0

  private val inFlightReadMaxMetric =
    metrics.getOrElse(SCAN_READ_IN_FLIGHT_MAX, NoopMetric)
  private val readyReadMaxMetric =
    metrics.getOrElse(SCAN_READ_READY_MAX, NoopMetric)
  private val backlogReadMaxMetric =
    metrics.getOrElse(SCAN_READ_BACKLOG_MAX, NoopMetric)

  if (settings.enabled) {
    updateMetricValue(metrics.getOrElse(SCAN_READ_WINDOW, NoopMetric), settings.readWindow)
  }

  def enabled: Boolean = settings.enabled

  def readWindow: Int = settings.readWindow

  def observeReadQueue(
      inFlightReadTasks: Int,
      readyReadTasks: Int,
      backlogReadTasks: Int): Unit = {
    if (!settings.enabled) {
      return
    }
    maxObservedInFlightReads = setMaxMetric(
      inFlightReadMaxMetric, maxObservedInFlightReads, inFlightReadTasks)
    maxObservedReadyReads = setMaxMetric(
      readyReadMaxMetric, maxObservedReadyReads, readyReadTasks)
    maxObservedBacklogReads = setMaxMetric(
      backlogReadMaxMetric, maxObservedBacklogReads, backlogReadTasks)
  }

  private def setMaxMetric(metric: GpuMetric, currentMax: Int, observed: Int): Int = {
    val sanitized = math.max(observed, 0)
    if (sanitized > currentMax) {
      updateMetricValue(metric, sanitized)
      sanitized
    } else {
      currentMax
    }
  }

  // Drive a SUM accumulator to hold an absolute (gauge) value for this task by posting the signed
  // delta from its current value. Correct within a single task thread; see the class metric note
  // for how these aggregate across tasks at the driver.
  private def updateMetricValue(metric: GpuMetric, value: Long): Unit = {
    val delta = value - metric.value
    if (delta != 0) {
      metric += delta
    }
  }
}

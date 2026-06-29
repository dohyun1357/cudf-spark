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

import com.nvidia.spark.rapids.GpuMetric._
import org.apache.hadoop.conf.Configuration

import org.apache.spark.internal.Logging

/**
 * Executor-local scan read-window actuator for the graph-wide autotuner.
 *
 * These types are the seam between the autotune hint/config layer and the multi-file readers in
 * `GpuMultiFileReader.scala`. They are kept out of that (very large) file so the autotune surface
 * area is easy to find and evolve. Three pieces:
 *
 *   - [[ScanPrefetchSettings]]: the eager-prefetch Hadoop-conf keys and the initial-fanout
 *     computation shared by the Parquet/ORC/Avro cloud readers (Phase 1.5 general prefetch).
 *   - [[ScanReadWindowSettings]]: the resolved, bounded read-window envelope a reader runs under,
 *     built either from Hadoop conf ([[ScanReadWindowSettings.fromConf]], LOCAL mode) or from a
 *     driver [[ScanRuntimeHint]] ([[ScanReadWindowSettings.fromHint]], GRAPH mode).
 *   - [[ScanReadWindowController]]: the per-reader actuator. LOCAL mode retains its AIMD behavior;
 *     GRAPH/OPTIMIZE tasks apply the optimizer's selected window directly and only record metrics.
 *
 * The whole feature is default-off: when disabled the settings collapse to the pre-existing static
 * fanout and the controller is inert, so reader behavior is unchanged.
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
 * Resolved read-window envelope for a single multi-file reader.
 *
 * When `enabled` is false the controller is inert and only `initialWindow` is consulted (as the
 * static initial fanout); `maxWindow`/`maxReadyBytes` are ignored. See
 * [[ScanReadWindowSettings.disabled]] for the all-zero inert instance and
 * [[ScanReadWindowSettings.fromConf]] for the fanout-preserving disabled instance used by the
 * cloud reader.
 */
case class ScanReadWindowSettings(
    enabled: Boolean,
    initialWindow: Int,
    maxWindow: Int,
    maxReadyBytes: Long,
    adaptive: Boolean = true)

object ScanReadWindowSettings {
  val ENABLED_KEY = "rapids.sql.autotune.scan.readWindow.enabled"
  val INITIAL_WINDOW_KEY = "rapids.sql.autotune.scan.readWindow.initial"
  val MAX_WINDOW_KEY = "rapids.sql.autotune.scan.readWindow.max"
  val MAX_READY_BYTES_KEY = "rapids.sql.autotune.scan.maxReadyBytes"

  val MIN_WINDOW = 1

  /**
   * Inert all-zero settings. Used as the no-hint fallback for the coalescing reader, which never
   * reads the window values while disabled. The cloud reader instead uses [[fromConf]], which
   * preserves the static initial fanout it consults unconditionally.
   */
  val disabled: ScanReadWindowSettings =
    ScanReadWindowSettings(enabled = false, initialWindow = 0, maxWindow = 0,
      maxReadyBytes = Long.MaxValue)

  private def positiveOrElse(value: Int, fallback: Int): Int =
    if (value > 0) value else fallback

  private def positiveLongOrElse(value: Long, fallback: Long): Long =
    if (value > 0) value else fallback

  def fromConf(
      conf: Configuration,
      maxNumFileProcessed: Int,
      inputFileCount: Int): ScanReadWindowSettings = {
    val existingInitialFanout =
      ScanPrefetchSettings.initialFanout(conf, maxNumFileProcessed, inputFileCount)
    if (!conf.getBoolean(ENABLED_KEY, false)) {
      ScanReadWindowSettings(
        enabled = false,
        initialWindow = existingInitialFanout,
        maxWindow = existingInitialFanout,
        maxReadyBytes = Long.MaxValue)
    } else {
      val configuredMaxWindow =
        positiveOrElse(conf.getInt(MAX_WINDOW_KEY, maxNumFileProcessed), maxNumFileProcessed)
      val maxWindow = Seq(maxNumFileProcessed, inputFileCount, configuredMaxWindow).min
      val configuredInitial =
        positiveOrElse(conf.getInt(INITIAL_WINDOW_KEY, MIN_WINDOW), MIN_WINDOW)
      val initialWindow =
        if (maxWindow > 0) math.min(math.max(configuredInitial, MIN_WINDOW), maxWindow) else 0
      val maxReadyBytes =
        positiveLongOrElse(conf.getLong(MAX_READY_BYTES_KEY, Long.MaxValue), Long.MaxValue)
      ScanReadWindowSettings(
        enabled = true,
        initialWindow = initialWindow,
        maxWindow = maxWindow,
        maxReadyBytes = maxReadyBytes)
    }
  }

  def fromHint(
      hint: ScanRuntimeHint,
      maxReadWindowCap: Int,
      inputFileCount: Int): Option[ScanReadWindowSettings] = {
    if (hint.maxReadWindow <= 0 || maxReadWindowCap <= 0 || inputFileCount <= 0) {
      None
    } else {
      val maxWindow = Seq(maxReadWindowCap, inputFileCount, hint.maxReadWindow).min
      val maxReadyBytes =
        positiveLongOrElse(hint.maxReadyBytes, Long.MaxValue)
      Some(ScanReadWindowSettings(
        enabled = true,
        // In graph modes maxReadWindow is the optimizer-selected target, not an envelope for a
        // second executor-local tuner. Apply it immediately for this task.
        initialWindow = maxWindow,
        maxWindow = maxWindow,
        maxReadyBytes = maxReadyBytes,
        adaptive = false))
    }
  }
}

/**
 * Per-reader actuator for the scan read window.
 *
 * LOCAL mode feeds it consumer-side signals ([[observeReadWait]]) and uses its historical AIMD
 * behavior. GRAPH/OPTIMIZE settings have `adaptive=false`: the driver optimizer owns tuning, and
 * this class applies the selected fixed target while continuing to record queue-depth evidence.
 *
 * Threading: a controller instance is owned by one reader and only touched from that reader's
 * task thread (the async read runners never call back in), so its mutable `var`s need no locking.
 *
 * Metric note: the gauge-style metrics (initial/current/max, in-flight/ready/backlog max) are
 * written as per-task absolute values into Spark SUM accumulators via [[updateMetricValue]]. Spark
 * sums each task's final value across the tasks of a scan node, so the per-node value shown in the
 * UI/eventlog is the sum of per-task values (sum-of-per-task-maxima for the *_MAX family), not a
 * global gauge/max. The increase/decrease metrics are true counters and sum correctly. Offline
 * tooling (`performance/autotuner`) re-aggregates these per scan.
 */
class ScanReadWindowController(
    settings: ScanReadWindowSettings,
    metrics: Map[String, GpuMetric] = Map.empty) extends Logging {
  private val MIN_WAIT_NS_FOR_INCREASE = TimeUnit.MILLISECONDS.toNanos(1)

  private var readWindow = settings.initialWindow
  private var increaseCooldownReads = 0
  private var maxObservedReadWindow = 0
  private var maxObservedInFlightReads = 0
  private var maxObservedReadyReads = 0
  private var maxObservedBacklogReads = 0

  private val readWindowInitialMetric =
    metrics.getOrElse(SCAN_READ_WINDOW_INITIAL, NoopMetric)
  private val readWindowCurrentMetric =
    metrics.getOrElse(SCAN_READ_WINDOW_CURRENT, NoopMetric)
  private val readWindowMaxMetric =
    metrics.getOrElse(SCAN_READ_WINDOW_MAX, NoopMetric)
  private val inFlightReadMaxMetric =
    metrics.getOrElse(SCAN_READ_IN_FLIGHT_MAX, NoopMetric)
  private val readyReadMaxMetric =
    metrics.getOrElse(SCAN_READ_READY_MAX, NoopMetric)
  private val backlogReadMaxMetric =
    metrics.getOrElse(SCAN_READ_BACKLOG_MAX, NoopMetric)
  private val readWindowIncreaseMetric =
    metrics.getOrElse(SCAN_READ_WINDOW_INCREASES, NoopMetric)
  private val readWindowDecreaseMetric =
    metrics.getOrElse(SCAN_READ_WINDOW_DECREASES, NoopMetric)
  private val readWindowMemoryDecreaseMetric =
    metrics.getOrElse(SCAN_READ_WINDOW_MEMORY_DECREASES, NoopMetric)
  private val readWindowScheduleDecreaseMetric =
    metrics.getOrElse(SCAN_READ_WINDOW_SCHEDULE_DECREASES, NoopMetric)

  if (settings.enabled) {
    updateMetricValue(readWindowInitialMetric, readWindow)
    recordCurrentReadWindow()
  }

  def enabled: Boolean = settings.enabled

  def initialReadWindow: Int = readWindow

  def currentReadWindow: Int = readWindow

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

  def observeReadWait(
      bufferWaitNs: Long,
      bufferGpuIdleNs: Long,
      scheduleWaitNs: Long,
      hostBytesAllocated: Long): Unit = {
    if (!settings.enabled || !settings.adaptive || settings.maxWindow <= 0) {
      return
    }

    val memoryPressureHigh = hostBytesAllocated > settings.maxReadyBytes
    val scheduleWaitHigh = scheduleWaitNs > bufferWaitNs && scheduleWaitNs > 0
    if (memoryPressureHigh || scheduleWaitHigh) {
      // Multiplicative decrease. At the floor (readWindow == MIN_WINDOW) the window cannot shrink
      // further, so the decrease block (and its post-decrease cooldown) is intentionally skipped;
      // the controller may re-grow on the next non-pressure observation.
      val nextWindow = math.max(ScanReadWindowSettings.MIN_WINDOW, readWindow / 2)
      if (nextWindow < readWindow) {
        logDebug(s"decrease scan read window from $readWindow to $nextWindow " +
          s"(hostBytes=$hostBytesAllocated, maxReadyBytes=${settings.maxReadyBytes}, " +
          s"bufferWaitNs=$bufferWaitNs, scheduleWaitNs=$scheduleWaitNs)")
        readWindow = nextWindow
        readWindowDecreaseMetric += 1
        if (memoryPressureHigh) {
          readWindowMemoryDecreaseMetric += 1
        }
        if (scheduleWaitHigh) {
          readWindowScheduleDecreaseMetric += 1
        }
        recordCurrentReadWindow()
        increaseCooldownReads = 1
      }
    } else if (increaseCooldownReads > 0) {
      increaseCooldownReads -= 1
    } else if (readWindow < settings.maxWindow &&
        (bufferWaitNs >= MIN_WAIT_NS_FOR_INCREASE || bufferGpuIdleNs > 0)) {
      val nextWindow = readWindow + 1
      logDebug(s"increase scan read window from $readWindow to $nextWindow " +
        s"(bufferWaitNs=$bufferWaitNs, bufferGpuIdleNs=$bufferGpuIdleNs)")
      readWindow = nextWindow
      readWindowIncreaseMetric += 1
      recordCurrentReadWindow()
    }
  }

  private def recordCurrentReadWindow(): Unit = {
    updateMetricValue(readWindowCurrentMetric, readWindow)
    maxObservedReadWindow = setMaxMetric(
      readWindowMaxMetric, maxObservedReadWindow, readWindow)
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

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
 * Static safety bounds and gating thresholds the [[AutotuneGraphModel]] decision function operates
 * within. These come from the user-configured caps (see `RapidsConf`), so every decision the model
 * makes is bounded by what the operator already allows.
 *
 * @param scanMaxReadWindow      hard ceiling for the scan read-window knob (the value the initial
 *                               graph scan hint was published at); restores never exceed it.
 * @param scanMaxReadyBytes      host-memory budget used as the scan host-pressure reference.
 * @param gpuMaxConcurrentTasks  hard ceiling for the GPU admission knob (the value the initial GPU
 *                               hint was published at); restores never exceed it.
 * @param minSampleTasks         minimum reported-task observations for a stage before the model
 *                               acts (hysteresis: do not retune on a handful of noisy samples).
 * @param optimizeGpu            OPTIMIZE mode: the GPU cap may be raised ABOVE static (up to
 *                               `gpuMaxConcurrentTasks`, which is then the higher OPTIMIZE ceiling)
 *                               using an aggressive raise gate decoupled from host-memory pressure.
 *                               When false (GRAPH) the GPU restore stays bounded by the static cap
 *                               and coupled to the global "no pressure" gate.
 */
case class AutotuneModelCaps(
    scanMaxReadWindow: Int,
    scanMaxReadyBytes: Long,
    gpuMaxConcurrentTasks: Int,
    minSampleTasks: Long,
    optimizeGpu: Boolean = false)

/**
 * A model decision: the updated stage hint content plus whether it lowered any knob. `isDecrease`
 * lets the driver enforce a cooldown after decreases (suppress an immediate re-increase) without
 * the model itself needing to be stateful.
 */
case class AutotuneDecision(hint: StageRuntimeHint, isDecrease: Boolean)

/**
 * The Phase 6 closed-loop decision function: given a stage's aggregated runtime observations, its
 * currently published hint, and the static caps, decide whether to republish an adjusted hint.
 *
 * Slice 2 policy -- conservative, analytical, and ALWAYS bounded by the static caps (it can only
 * range a knob within `[floor, initialHintValue]`, never above what the driver first published,
 * which itself equals the user's static cap). This is AIMD: multiplicative-decrease a knob under
 * observed pressure, additive-restore it toward the cap once pressure clears. Above-static increase
 * is deliberately NOT part of this slice -- that is Slice 3 (`OPTIMIZE` mode).
 *
 * The function is PURE (no SparkContext, no clock, no mutable state) so it is exhaustively
 * unit-testable. All oscillation control that needs state -- debounce, cooldown-after-decrease --
 * is enforced by the caller ([[RapidsAutotuneDriverEndpoint]]); the model contributes the stateless
 * controls: the min-sample gate, per-step clamping, and "prefer reducing pressure" (a pressure
 * signal always wins over a restore, because a restore requires the absence of every pressure).
 *
 * The model reads only the current hint, so it never needs the stage shape: a knob that the stage
 * was not hinted for is inactive (window / cap <= 0) and is left untouched.
 */
object AutotuneGraphModel {
  /** semaphore wait >= holding: GPU admission is contended -> relieve by lowering concurrency. */
  val GpuWaitRatioHigh: Double = 1.0
  /** little semaphore waiting: headroom to restore the GPU knob toward the static cap. */
  val GpuWaitRatioLow: Double = 0.25
  /** host allocation >= this fraction of the ready-bytes budget counts as host-memory pressure. */
  val HostHighFraction: Double = 0.9
  /** host allocation <= this fraction of the ready-bytes budget counts as host-memory headroom. */
  val HostLowFraction: Double = 0.5

  def decide(
      obs: StageObservationAgg,
      current: StageRuntimeHint,
      caps: AutotuneModelCaps): Option[AutotuneDecision] = {
    if (obs.taskCount < caps.minSampleTasks) {
      // Min-sample hysteresis: not enough evidence to act.
      None
    } else {
      val hostPressure = caps.scanMaxReadyBytes > 0L &&
        obs.maxHostMemoryBytes.toDouble >= caps.scanMaxReadyBytes.toDouble * HostHighFraction
      val spillPressure = obs.totalSpillBytes > 0L
      val gpuPressure = obs.gpuWaitRatio >= GpuWaitRatioHigh
      // Restore only when EVERY pressure signal is clear (prefer reducing pressure): low GPU wait,
      // no spill, and host allocation comfortably below the budget.
      val relaxed = !spillPressure && obs.gpuWaitRatio <= GpuWaitRatioLow &&
        (caps.scanMaxReadyBytes <= 0L ||
          obs.maxHostMemoryBytes.toDouble <= caps.scanMaxReadyBytes.toDouble * HostLowFraction)

      // GPU raise gate. GRAPH: coupled to the global relaxed gate, bounded by the static cap.
      // OPTIMIZE: aggressive AIMD-up decoupled from host-memory pressure (host memory bounds the
      // scan window, not GPU concurrency) -- raise toward the OPTIMIZE ceiling whenever the GPU is
      // not contended and there is no spill. The permit pool remains the hard memory backstop.
      val gpuRaiseGate =
        if (caps.optimizeGpu) !spillPressure && obs.gpuWaitRatio <= GpuWaitRatioLow else relaxed

      val (newScan, scanDown) =
        decideScan(current.scan, caps, reduce = hostPressure || spillPressure, restore = relaxed)
      val (newGpu, gpuDown) =
        decideGpu(current.gpu, caps, reduce = gpuPressure || spillPressure, restore = gpuRaiseGate)

      if (newScan == current.scan && newGpu == current.gpu) {
        None
      } else {
        Some(AutotuneDecision(
          current.copy(scan = newScan, gpu = newGpu),
          isDecrease = scanDown || gpuDown))
      }
    }
  }

  /** Returns (adjusted scan hint, decreased?). Only adjusts an active scan-prefetch hint. */
  private def decideScan(
      scan: ScanRuntimeHint,
      caps: AutotuneModelCaps,
      reduce: Boolean,
      restore: Boolean): (ScanRuntimeHint, Boolean) = {
    if (scan.maxReadWindow <= 0) {
      (scan, false) // inactive knob (stage not hinted for scan prefetch)
    } else if (reduce) {
      val floor = math.max(1, scan.minReadWindow)
      val reduced = math.max(floor, scan.maxReadWindow / 2)
      if (reduced < scan.maxReadWindow) {
        (scan.copy(maxReadWindow = reduced), true)
      } else {
        (scan, false)
      }
    } else if (restore && scan.maxReadWindow < caps.scanMaxReadWindow) {
      val raised = math.min(caps.scanMaxReadWindow, scan.maxReadWindow + 1)
      (scan.copy(maxReadWindow = raised), false)
    } else {
      (scan, false)
    }
  }

  /** Returns (adjusted GPU hint, decreased?). Only adjusts an active GPU admission hint. */
  private def decideGpu(
      gpu: GpuRuntimeHint,
      caps: AutotuneModelCaps,
      reduce: Boolean,
      restore: Boolean): (GpuRuntimeHint, Boolean) = {
    if (gpu.maxConcurrentTasks <= 0) {
      (gpu, false) // inactive knob (no GPU admission hint configured)
    } else if (reduce) {
      val reduced = math.max(1, gpu.maxConcurrentTasks / 2)
      if (reduced < gpu.maxConcurrentTasks) {
        (gpu.copy(maxConcurrentTasks = reduced), true)
      } else {
        (gpu, false)
      }
    } else if (restore && gpu.maxConcurrentTasks < caps.gpuMaxConcurrentTasks) {
      val raised = math.min(caps.gpuMaxConcurrentTasks, gpu.maxConcurrentTasks + 1)
      (gpu.copy(maxConcurrentTasks = raised), false)
    } else {
      (gpu, false)
    }
  }
}

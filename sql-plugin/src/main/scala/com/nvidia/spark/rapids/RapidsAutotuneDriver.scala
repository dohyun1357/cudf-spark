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

import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.util.Try
import scala.util.control.NonFatal

import org.apache.spark.SparkContext
import org.apache.spark.internal.Logging
import org.apache.spark.scheduler.{SparkListener, SparkListenerEvent,
  SparkListenerExecutorAdded, SparkListenerExecutorRemoved, SparkListenerJobStart,
  SparkListenerStageCompleted, SparkListenerStageSubmitted, SparkListenerTaskEnd}
import org.apache.spark.sql.execution.SQLExecution
import org.apache.spark.sql.execution.ui.SparkListenerSQLExecutionEnd
import org.apache.spark.sql.rapids.execution.TrampolineUtil

/**
 * Driver-side owner of the current per-stage hint map for the graph autotuner.
 *
 * Process-wide singleton (one driver plugin per JVM). It assigns monotonically increasing hint
 * versions, answers executor hint requests ([[handleHintRequest]]), and re-posts applied-hint
 * reports as [[SparkRapidsAutotuneHintAppliedEvent]] eventlog records ([[handleHintApplied]]).
 * When autotune is disabled it holds no hints and reports nothing. Tests must call [[init]] /
 * [[shutdown]] to manage the shared state.
 */
object RapidsAutotuneDriverEndpoint extends Logging {
  private val hints = new ConcurrentHashMap[AutotuneStageKey, StageRuntimeHint]()
  // Cumulative per-stage aggregate -- feeds the eventlog observation record. Monotonic by design
  // (lifetime totals/high-water), so it is not the optimizer's window input.
  private val observations = new ConcurrentHashMap[AutotuneStageKey, StageObservationAgg]()
  private val nextHintVersion = new AtomicLong(1L)
  private val nextParallelismDecisionId = new AtomicLong(1L)
  // Live executor-core census from scheduler registration events. Cluster task slots are the sum
  // of floor(cores / spark.task.cpus) over registered executors -- the width Spark's scheduler
  // actually admits tasks against. The per-executor GPU admission quota is a different resource
  // and undercounts a multi-executor cluster.
  private val executorCores = new ConcurrentHashMap[String, Int]()
  // Recent per-stage initial-burst launch spacing measurements (nanos per task launch). Their
  // median estimates the serial driver dispatch work one extra reducer task adds to a stage.
  // Stage overlap can only stretch a burst (busy slots delay launches), so a contaminated
  // sample overestimates dispatch and errs against re-splitting.
  private val launchSpacingSamples = mutable.ArrayBuffer.empty[Double]
  private val maxLaunchSpacingSamples = 64
  private var launchSpacingStages = 0L
  @volatile private var taskCpus = 1
  @volatile private var sparkContext: SparkContext = _
  @volatile private var enabled = false
  // The graph optimizer only evaluates decision epochs in the GRAPH/OPTIMIZE modes.
  @volatile private var optimizerEnabled = false
  @volatile private var optimizer: AnalyticalGraphWideAutotuneOptimizer = _
  // Injectable monotonic clock so optimizer update cadence is deterministically testable.
  @volatile private var nanoSource: () => Long = () => System.nanoTime()

  def init(sc: SparkContext, conf: RapidsConf): Unit = synchronized {
    enabled = conf.autotuneGraphEnabled
    optimizerEnabled = conf.isAutotuneClosedLoopMode
    sparkContext = if (enabled) sc else null
    optimizer = new AnalyticalGraphWideAutotuneOptimizer(
      GraphOptimizerConstraints.fromConf(conf))
    nanoSource = () => System.nanoTime()
    hints.clear()
    observations.clear()
    executorCores.clear()
    launchSpacingSamples.clear()
    launchSpacingStages = 0L
    taskCpus = if (sc == null) 1 else math.max(1, sc.getConf.getInt("spark.task.cpus", 1))
    nextHintVersion.set(1L)
    nextParallelismDecisionId.set(1L)
    if (enabled) {
      logInfo(s"Initialized RAPIDS graph autotune driver endpoint in " +
        s"${conf.autotuneGraphMode} mode")
    }
  }

  def shutdown(): Unit = synchronized {
    enabled = false
    optimizerEnabled = false
    sparkContext = null
    optimizer = null
    nanoSource = () => System.nanoTime()
    hints.clear()
    observations.clear()
    executorCores.clear()
    launchSpacingSamples.clear()
    launchSpacingStages = 0L
    taskCpus = 1
    nextHintVersion.set(1L)
    nextParallelismDecisionId.set(1L)
  }

  def executorAdded(executorId: String, totalCores: Int): Unit = {
    if (totalCores > 0) {
      executorCores.put(executorId, totalCores)
    }
  }

  def executorRemoved(executorId: String): Unit = {
    executorCores.remove(executorId)
  }

  /** Measured cluster CPU task slots; zero until an executor has registered. */
  private[rapids] def clusterTaskSlots: Int =
    executorCores.values().asScala.map(_ / taskCpus).sum

  /**
   * Initial-burst dispatch spacing of one completed stage. The first min(tasks, slots) launches
   * are dispatched back-to-back when the stage starts, so their mean gap measures the serial
   * driver scheduler/launch response; predecessor overlap only stretches a burst, so a
   * contaminated sample overestimates. Fewer than eight gaps cannot separate dispatch spacing
   * from a single scheduling hiccup and yield no measurement.
   */
  private[rapids] def initialBurstLaunchSpacingNanos(
      launchTimesMs: Seq[Long],
      taskSlots: Int): Option[Double] = {
    if (taskSlots <= 0) {
      return None
    }
    val burst = launchTimesMs.sorted.take(math.min(launchTimesMs.size, taskSlots))
    val gaps = burst.size - 1
    if (gaps < 8) None else {
      Some(math.max(0L, burst.last - burst.head).toDouble * 1e6 / gaps.toDouble)
    }
  }

  /** Record one completed stage's measured initial-burst launch spacing. */
  private[rapids] def recordStageLaunchSpacing(spacingNanos: Double): Unit = synchronized {
    if (spacingNanos >= 0.0 && java.lang.Double.isFinite(spacingNanos)) {
      launchSpacingSamples += spacingNanos
      launchSpacingStages += 1L
      if (launchSpacingSamples.size > maxLaunchSpacingSamples) {
        launchSpacingSamples.remove(0)
      }
    }
  }

  /** Median measured launch spacing; None until a completed stage has been measured. */
  private[rapids] def serialLaunchNanosPerTask: Option[Double] = synchronized {
    if (launchSpacingSamples.isEmpty) None else {
      val sorted = launchSpacingSamples.sorted
      Some(sorted(sorted.size / 2))
    }
  }

  /** Test-only: inject a deterministic monotonic clock for optimizer update cadence. */
  private[rapids] def setNanoSourceForTest(source: () => Long): Unit = synchronized {
    nanoSource = source
  }

  /** Test-only: override the task-CPU divisor used by the cluster slot census. */
  private[rapids] def setTaskCpusForTest(cpus: Int): Unit = synchronized {
    taskCpus = math.max(1, cpus)
  }

  /** Register a graph node and publish the optimizer's complete initial joint hint. */
  def publishOptimizerHint(
      key: AutotuneStageKey,
      descriptor: AutotuneStageDescriptor): StageRuntimeHint = synchronized {
    if (!enabled || optimizer == null) {
      StageRuntimeHint.empty(key)
    } else {
      val content = optimizer.initialHint(key, descriptor)
      val published = publishStageHint(
        key, content.scan, content.shuffle, content.batch)
      optimizer.hintPublished(published)
      published
    }
  }

  def stageSubmitted(key: AutotuneStageKey): Unit = {
    if (enabled && optimizer != null) {
      try {
        optimizer.stageSubmitted(key)
        publishDecisionRecords()
      } catch {
        case NonFatal(e) =>
          logWarning("RAPIDS graph optimizer stage-submission update failed", e)
      }
    }
  }

  def stageCompleted(key: AutotuneStageKey): Unit = {
    if (enabled && optimizer != null) {
      try {
        optimizer.stageCompleted(key)
        publishDecisionRecords()
      } catch {
        case NonFatal(e) =>
          logWarning("RAPIDS graph optimizer stage-completion update failed", e)
      }
    }
  }

  def observeTaskSetup(executionId: Long, setupNanos: Long): Unit = {
    if (enabled && optimizer != null) {
      try {
        optimizer.observeTaskSetup(executionId, setupNanos)
      } catch {
        case NonFatal(e) =>
          logWarning("RAPIDS graph optimizer task-setup calibration failed", e)
      }
    }
  }

  def executionCompleted(executionId: Long): Unit = synchronized {
    hints.keySet().asScala.filter(_.executionId == executionId).foreach(hints.remove)
    observations.keySet().asScala.filter(_.executionId == executionId).foreach(observations.remove)
    if (optimizer != null) {
      optimizer.executionCompleted(executionId)
    }
  }

  /**
   * Publish (once) the hint for a stage key. The first publication for a key wins and fixes its
   * version; later calls for the same key return the existing hint so executors see a stable
   * version for the stage attempt.
   */
  def publishStageHint(
      key: AutotuneStageKey,
      scanHint: ScanRuntimeHint,
      shuffleHint: ShuffleRuntimeHint = ShuffleRuntimeHint.empty,
      batchHint: BatchRuntimeHint = BatchRuntimeHint.empty): StageRuntimeHint = synchronized {
    if (!enabled) {
      StageRuntimeHint.empty(key)
    } else {
      Option(hints.get(key)).getOrElse {
        val hint = StageRuntimeHint(
          executionId = key.executionId,
          stageId = key.stageId,
          stageAttemptId = key.stageAttemptId,
          version = nextHintVersion.getAndIncrement(),
          scan = scanHint,
          shuffle = shuffleHint,
          batch = batchHint)
        hints.put(key, hint)
        hint
      }
    }
  }

  def handleHintRequest(msg: RapidsAutotuneHintRequestMsg): RapidsAutotuneHintResponseMsg = {
    val hint = if (enabled) Option(hints.get(msg.key)) else None
    RapidsAutotuneHintResponseMsg(msg.key, hint)
  }

  def handleHintApplied(msg: RapidsAutotuneHintAppliedMsg): Unit = {
    if (!enabled || sparkContext == null) {
      return
    }
    TrampolineUtil.postEvent(sparkContext, toAppliedEvent(msg))
  }

  /**
   * Flatten an applied-hint report into its eventlog record. Pure (no SparkContext / endpoint
   * state required) so the per-field mapping can be unit-tested directly.
   */
  private[rapids] def toAppliedEvent(
      msg: RapidsAutotuneHintAppliedMsg): SparkRapidsAutotuneHintAppliedEvent =
    SparkRapidsAutotuneHintAppliedEvent(
      executorId = msg.executorId,
      executionId = msg.key.executionId,
      stageId = msg.key.stageId,
      stageAttemptId = msg.key.stageAttemptId,
      taskAttemptId = msg.taskAttemptId,
      partitionId = msg.partitionId,
      hintVersion = msg.hintVersion,
      hasHint = msg.hasHint,
      scanEagerPrefetch = msg.scan.eagerPrefetch,
      scanMinReadWindow = msg.scan.minReadWindow,
      scanMaxReadWindow = msg.scan.maxReadWindow,
      scanMaxReadyBytes = msg.scan.maxReadyBytes,
      shufflePrefetchWindow = msg.shuffle.prefetchWindow,
      shuffleMaxReadyBytes = msg.shuffle.maxReadyBytes,
      shuffleCoalesceTargetBytes = msg.shuffle.coalesceTargetBytes,
      batchTargetBatchBytes = msg.batch.targetBatchBytes,
      batchMaxBatchBytes = msg.batch.maxBatchBytes)

  /**
   * Ingest a runtime observation: update the per-stage aggregate, post an eventlog record, and (in
   * a closed-loop mode) run the graph optimizer, possibly republishing a complete joint hint.
   */
  def handleObservation(msg: RapidsAutotuneObservationMsg): Unit = {
    if (!enabled) {
      return
    }
    val agg = observations.compute(msg.key, (_, prev) =>
      (if (prev == null) StageObservationAgg.empty else prev).merge(msg))
    if (sparkContext != null) {
      TrampolineUtil.postEvent(sparkContext, toObservationEvent(msg, agg))
    }
    maybeOptimize(msg)
  }

  /**
   * Closed-loop optimizer trigger. The optimizer owns its observation windows, update cadence and
   * stage graph. This endpoint owns only wire publication/versioning.
   *
   * Fully guarded: an optimizer exception is swallowed (advisory path) so it can never break
   * observation handling for other stages or the driver RPC dispatcher.
   */
  private def maybeOptimize(msg: RapidsAutotuneObservationMsg): Unit = synchronized {
    if (!optimizerEnabled || optimizer == null) {
      return
    }
    try {
      val current = hints.get(msg.key)
      if (current == null) {
        return
      }
      optimizer.observe(msg, current, nanoSource())
      publishDecisionRecords()
    } catch {
      case NonFatal(e) =>
        logWarning("RAPIDS graph optimizer evaluation failed; leaving hint unchanged", e)
    }
  }

  private def publishDecisionRecords(): Unit = {
    if (optimizer != null) {
      val records = optimizer.drainDecisionRecords()
      if (sparkContext != null) {
        records.foreach(record => TrampolineUtil.postEvent(sparkContext, toDecisionEvent(record)))
      }
    }
  }

  private[rapids] def toDecisionEvent(
      record: GraphStageDecisionRecord): SparkRapidsAutotuneGraphDecisionEvent =
    SparkRapidsAutotuneGraphDecisionEvent(
      epochId = record.epochId,
      trigger = record.trigger,
      executionId = record.key.executionId,
      stageId = record.key.stageId,
      stageAttemptId = record.key.stageAttemptId,
      active = record.active,
      completed = record.completed,
      parentStageIds = record.parentStageIds,
      observedTasks = record.observedTasks,
      sampleTasks = record.sampleTasks,
      currentHintVersion = record.currentHintVersion,
      graphCurrentObjectiveNanos = record.graphCurrentObjectiveNanos,
      predictedCurrentNanos = record.predictedCurrentNanos,
      durationAdjoint = record.durationAdjoint,
      currentScanWindow = record.currentControl.scanWindow,
      currentGpuTasks = record.currentControl.gpuTasks,
      currentShuffleWindow = record.currentControl.shuffleWindow,
      currentShuffleBytes = record.currentControl.shuffleBytes,
      currentBatchBytes = record.currentControl.batchBytes,
      scanWindowGradient = record.endToEndGradient.scanWindow,
      gpuTasksGradient = record.endToEndGradient.gpuTasks,
      shuffleWindowGradient = record.endToEndGradient.shuffleWindow,
      shuffleBytesGradient = record.endToEndGradient.shuffleBytes,
      batchBytesGradient = record.endToEndGradient.batchBytes,
      scanWindowFreezeReason = record.freezeReasons.scanWindow,
      gpuTasksFreezeReason = record.freezeReasons.gpuTasks,
      shuffleWindowFreezeReason = record.freezeReasons.shuffleWindow,
      shuffleBytesFreezeReason = record.freezeReasons.shuffleBytes,
      batchBytesFreezeReason = record.freezeReasons.batchBytes)

  private[rapids] def aqeCalibrationSnapshot: Option[GpuFlowAqeCalibration] = synchronized {
    val executionId = Option(sparkContext)
      .flatMap(sc => Option(sc.getLocalProperty(SQLExecution.EXECUTION_ID_KEY)))
      .flatMap(value => Try(value.toLong).toOption)
    executionId.flatMap(id => Option(optimizer).flatMap(_.aqeCalibrationSnapshot(id)))
      .map(_.copy(
        clusterTaskSlots = clusterTaskSlots,
        serialLaunchNanosPerTask = serialLaunchNanosPerTask,
        serialLaunchSampleStages = launchSpacingStages))
  }

  private[rapids] def recordParallelismDecision(
      decision: GpuFlowParallelismDecision): Unit = {
    val sc = sparkContext
    if (enabled && sc != null) {
      val executionId = Option(sc.getLocalProperty(SQLExecution.EXECUTION_ID_KEY))
        .flatMap(value => Try(value.toLong).toOption).getOrElse(-1L)
      TrampolineUtil.postEvent(sc, SparkRapidsAutotuneParallelismEvent(
        decisionId = nextParallelismDecisionId.getAndIncrement(),
        executionId = executionId,
        stageIds = decision.stageIds,
        originalPartitions = decision.originalPartitions,
        currentPartitions = decision.currentPartitions,
        selectedPartitions = decision.selectedPartitions,
        totalBytes = decision.totalBytes,
        taskSlots = decision.taskSlots,
        currentObjectiveNanos = decision.currentObjectiveNanos,
        selectedObjectiveNanos = decision.selectedObjectiveNanos,
        variableNanosPerByte = decision.variableNanosPerByte,
        fixedNanosPerTask = decision.fixedNanosPerTask,
        fixedNanosPerTaskStandardError = decision.fixedNanosPerTaskStandardError,
        fixedTaskCostSampleWindows = decision.fixedTaskCostSampleWindows,
        fixedTaskCostSource = decision.fixedTaskCostSource,
        serialLaunchNanosPerTask = decision.serialLaunchNanosPerTask,
        serialLaunchSampleStages = decision.serialLaunchSampleStages,
        reason = decision.reason))
    }
  }

  private[rapids] def toObservationEvent(
      msg: RapidsAutotuneObservationMsg,
      agg: StageObservationAgg): SparkRapidsAutotuneObservationEvent =
    SparkRapidsAutotuneObservationEvent(
      executorId = msg.executorId,
      executionId = msg.key.executionId,
      stageId = msg.key.stageId,
      stageAttemptId = msg.key.stageAttemptId,
      taskAttemptId = msg.taskAttemptId,
      partitionId = msg.partitionId,
      hintVersion = msg.hintVersion,
      gpuSemaphoreWaitNanos = msg.gpuSemaphoreWaitNanos,
      gpuHoldingNanos = msg.gpuHoldingNanos,
      hostMemoryBytes = msg.hostMemoryBytes,
      stageTaskCount = agg.taskCount,
      stageMaxHostMemoryBytes = agg.maxHostMemoryBytes,
      spillBytes = msg.spillBytes,
      stageTotalSpillBytes = agg.totalSpillBytes,
      shuffleReadLimiterAcquireCount = msg.shuffleReadLimiterAcquireCount,
      shuffleReadLimiterAcquireFailCount = msg.shuffleReadLimiterAcquireFailCount,
      stageShuffleReadLimiterAcquireCount = agg.totalShuffleReadLimiterAcquireCount,
      stageShuffleReadLimiterAcquireFailCount = agg.totalShuffleReadLimiterAcquireFailCount,
      taskDurationNanos = msg.taskDurationNanos,
      inputBytes = msg.inputBytes,
      outputBytes = msg.outputBytes,
      shuffleReadBytes = msg.shuffleReadBytes,
      shuffleWriteBytes = msg.shuffleWriteBytes,
      inputRows = msg.inputRows,
      outputRows = msg.outputRows,
      pinnedMemoryBytes = msg.pinnedMemoryBytes,
      deviceMemoryBytes = msg.deviceMemoryBytes,
      retryOrLostTimeNanos = msg.retryOrLostTimeNanos,
      stageTotalTaskDurationNanos = agg.totalTaskDurationNanos,
      stageTotalInputBytes = agg.totalInputBytes,
      stageTotalOutputBytes = agg.totalOutputBytes,
      stageTotalShuffleReadBytes = agg.totalShuffleReadBytes,
      stageTotalShuffleWriteBytes = agg.totalShuffleWriteBytes,
      stageMaxPinnedMemoryBytes = agg.maxPinnedMemoryBytes,
      stageMaxDeviceMemoryBytes = agg.maxDeviceMemoryBytes,
      stageTotalRetryOrLostTimeNanos = agg.totalRetryOrLostTimeNanos)
}

/**
 * Driver-side running aggregate of executor observations for one stage attempt. Pure value type;
 * the graph optimizer reads it to derive measured resource work.
 */
case class StageObservationAgg(
    taskCount: Long,
    totalGpuSemaphoreWaitNanos: Long,
    totalGpuHoldingNanos: Long,
    maxHostMemoryBytes: Long,
    totalSpillBytes: Long,
    totalShuffleReadLimiterAcquireCount: Long = 0L,
    totalShuffleReadLimiterAcquireFailCount: Long = 0L,
    totalTaskDurationNanos: Long = 0L,
    totalInputBytes: Long = 0L,
    totalOutputBytes: Long = 0L,
    totalShuffleReadBytes: Long = 0L,
    totalShuffleWriteBytes: Long = 0L,
    maxPinnedMemoryBytes: Long = 0L,
    maxDeviceMemoryBytes: Long = 0L,
    totalRetryOrLostTimeNanos: Long = 0L) {
  def merge(msg: RapidsAutotuneObservationMsg): StageObservationAgg = StageObservationAgg(
    taskCount = taskCount + 1L,
    totalGpuSemaphoreWaitNanos = totalGpuSemaphoreWaitNanos + msg.gpuSemaphoreWaitNanos,
    totalGpuHoldingNanos = totalGpuHoldingNanos + msg.gpuHoldingNanos,
    maxHostMemoryBytes = math.max(maxHostMemoryBytes, msg.hostMemoryBytes),
    totalSpillBytes = totalSpillBytes + msg.spillBytes,
    totalShuffleReadLimiterAcquireCount = totalShuffleReadLimiterAcquireCount +
      msg.shuffleReadLimiterAcquireCount,
    totalShuffleReadLimiterAcquireFailCount = totalShuffleReadLimiterAcquireFailCount +
      msg.shuffleReadLimiterAcquireFailCount,
    totalTaskDurationNanos = totalTaskDurationNanos + msg.taskDurationNanos,
    totalInputBytes = totalInputBytes + msg.inputBytes,
    totalOutputBytes = totalOutputBytes + msg.outputBytes,
    totalShuffleReadBytes = totalShuffleReadBytes + msg.shuffleReadBytes,
    totalShuffleWriteBytes = totalShuffleWriteBytes + msg.shuffleWriteBytes,
    maxPinnedMemoryBytes = math.max(maxPinnedMemoryBytes, msg.pinnedMemoryBytes),
    maxDeviceMemoryBytes = math.max(maxDeviceMemoryBytes, msg.deviceMemoryBytes),
    totalRetryOrLostTimeNanos = totalRetryOrLostTimeNanos + msg.retryOrLostTimeNanos)
}

object StageObservationAgg {
  val empty: StageObservationAgg = StageObservationAgg(0L, 0L, 0L, 0L, 0L)
}

/**
 * Driver `SparkListener` that publishes a [[StageRuntimeHint]] for each stage of a SQL execution
 * when its job starts, using the configured scan/GPU hint policies. Stages outside a SQL execution
 * (no execution id) are skipped.
 */
class RapidsAutotuneStageHintListener(conf: RapidsConf) extends SparkListener with Logging {
  private val stageKeys = mutable.HashMap.empty[(Int, Int), AutotuneStageKey]
  // Successful-task launch times per running stage attempt, for the initial-burst dispatch
  // spacing measurement at stage completion. Dispatch is a driver property, so every stage
  // measures it, not only SQL stages with hints. Insertion-ordered so the bound evicts the
  // stalest stage first.
  private val stageLaunchTimes =
    mutable.LinkedHashMap.empty[(Int, Int), mutable.ArrayBuffer[Long]]
  private val maxTrackedLaunchStages = 256
  private val maxTrackedLaunchesPerStage = 65536

  override def onJobStart(jobStart: SparkListenerJobStart): Unit = {
    val executionId = executionIdFrom(jobStart.properties)

    executionId.foreach { execId =>
      jobStart.stageInfos.foreach { stageInfo =>
        publishHintForStage(
          execId,
          stageInfo.stageId,
          stageInfo.attemptNumber(),
          AutotuneStageShape.fromStageInfo(stageInfo),
          stageInfo.parentIds)
      }
    }
  }

  override def onStageSubmitted(stageSubmitted: SparkListenerStageSubmitted): Unit = synchronized {
    val info = stageSubmitted.stageInfo
    val lookup = (info.stageId, info.attemptNumber())
    val key = stageKeys.get(lookup).orElse {
      executionIdFrom(stageSubmitted.properties).map { executionId =>
        publishHintForStage(
          executionId,
          info.stageId,
          info.attemptNumber(),
          AutotuneStageShape.fromStageInfo(info),
          info.parentIds).key
      }
    }
    key.foreach(RapidsAutotuneDriverEndpoint.stageSubmitted)
  }

  override def onStageCompleted(stageCompleted: SparkListenerStageCompleted): Unit = synchronized {
    val info = stageCompleted.stageInfo
    stageLaunchTimes.remove((info.stageId, info.attemptNumber())).foreach { launches =>
      RapidsAutotuneDriverEndpoint.initialBurstLaunchSpacingNanos(
        launches.toSeq, RapidsAutotuneDriverEndpoint.clusterTaskSlots)
        .foreach(RapidsAutotuneDriverEndpoint.recordStageLaunchSpacing)
    }
    stageKeys.get((info.stageId, info.attemptNumber())).foreach(
      RapidsAutotuneDriverEndpoint.stageCompleted)
  }

  override def onExecutorAdded(executorAdded: SparkListenerExecutorAdded): Unit = {
    RapidsAutotuneDriverEndpoint.executorAdded(
      executorAdded.executorId, executorAdded.executorInfo.totalCores)
  }

  override def onExecutorRemoved(executorRemoved: SparkListenerExecutorRemoved): Unit = {
    RapidsAutotuneDriverEndpoint.executorRemoved(executorRemoved.executorId)
  }

  override def onTaskEnd(taskEnd: SparkListenerTaskEnd): Unit = synchronized {
    val metrics = taskEnd.taskMetrics
    val info = taskEnd.taskInfo
    if (info != null && info.successful) {
      val launches = stageLaunchTimes.getOrElseUpdate(
        (taskEnd.stageId, taskEnd.stageAttemptId), {
          while (stageLaunchTimes.size >= maxTrackedLaunchStages) {
            stageLaunchTimes.remove(stageLaunchTimes.head._1)
          }
          mutable.ArrayBuffer.empty[Long]
        })
      if (launches.size < maxTrackedLaunchesPerStage) {
        launches += info.launchTime
      }
    }
    if (metrics != null && info != null && info.successful) {
      stageKeys.get((taskEnd.stageId, taskEnd.stageAttemptId)).foreach { key =>
        val millis = math.max(0L, metrics.executorDeserializeTime)
        val nanos = if (millis > Long.MaxValue / 1000000L) {
          Long.MaxValue
        } else {
          millis * 1000000L
        }
        RapidsAutotuneDriverEndpoint.observeTaskSetup(key.executionId, nanos)
      }
    }
  }

  override def onOtherEvent(event: SparkListenerEvent): Unit = {
    event match {
      case sqlEnd: SparkListenerSQLExecutionEnd => synchronized {
        stageKeys.retain { case (_, key) => key.executionId != sqlEnd.executionId }
        RapidsAutotuneDriverEndpoint.executionCompleted(sqlEnd.executionId)
      }
      case _ =>
    }
  }

  private[rapids] def publishHintForStage(
      executionId: Long,
      stageId: Int,
      stageAttemptId: Int,
      stageShape: AutotuneStageShape,
      parentStageIds: Seq[Int] = Seq.empty): StageRuntimeHint = synchronized {
    val key = AutotuneStageKey(
      executionId = executionId,
      stageId = stageId,
      stageAttemptId = stageAttemptId)
    stageKeys.put((stageId, stageAttemptId), key)
    val hint = RapidsAutotuneDriverEndpoint.publishOptimizerHint(
      key, AutotuneStageDescriptor(stageShape, parentStageIds))
    if (hint.version > 0) {
      logDebug(s"Published RAPIDS graph autotune hint version ${hint.version} " +
        s"for execution $executionId, stage ${key.stageId}.${key.stageAttemptId}, " +
        s"scan hint ${hint.scan}, shuffle hint ${hint.shuffle}, " +
        s"batch hint ${hint.batch}")
    }
    hint
  }

  private def executionIdFrom(properties: java.util.Properties): Option[Long] =
    Option(properties)
      .flatMap(p => Option(p.getProperty(SQLExecution.EXECUTION_ID_KEY)))
      .flatMap(v => Try(v.toLong).toOption)
}

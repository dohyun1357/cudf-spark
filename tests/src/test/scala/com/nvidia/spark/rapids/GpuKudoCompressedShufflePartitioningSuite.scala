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

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}

import scala.collection.mutable

import ai.rapids.cudf.Table
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.RapidsConf.ShuffleKudoMode
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import com.nvidia.spark.rapids.jni.RmmSpark
import com.nvidia.spark.rapids.jni.kudo.DumpOption
import com.nvidia.spark.rapids.shims.GpuHashPartitioning
import org.scalatest.BeforeAndAfterEach
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.prop.TableDrivenPropertyChecks

import org.apache.spark.SparkConf
import org.apache.spark.sql.catalyst.expressions.{AttributeReference, ExprId}
import org.apache.spark.sql.execution.metric.SQLMetric
import org.apache.spark.sql.rapids.GpuShuffleEnv
import org.apache.spark.sql.rapids.execution.{GpuShuffleExchangeExecBase, TrampolineUtil}
import org.apache.spark.sql.types.{DataType, IntegerType, StringType}
import org.apache.spark.sql.vectorized.ColumnarBatch

object GpuKudoCompressedShufflePartitioningSuite {
  val dataTypes: Array[DataType] = Array(IntegerType, StringType)

  /**
   * Builds a compressible batch: low-cardinality ints (some nulls) and repetitive
   * strings so device-side zstd is guaranteed to shrink the serialized payload.
   */
  def buildCompressibleBatch(numRows: Int): ColumnarBatch = {
    val ints = new Array[java.lang.Integer](numRows)
    val strs = new Array[String](numRows)
    (0 until numRows).foreach { i =>
      ints(i) = if (i % 7 == 0) null else Int.box(i % 100)
      strs(i) = if (i % 11 == 0) null else s"repeated_payload_${i % 13}"
    }
    withResource(new Table.TestBuilder()
      .column(ints: _*)
      .column(strs: _*)
      .build()) { table =>
      GpuColumnVector.from(table, dataTypes)
    }
  }

  /** The multiset of (int, string) pairs expected from one compressible batch. */
  def expectedCounts(numRows: Int): Map[(Option[Int], Option[String]), Int] = {
    (0 until numRows).map { i =>
      val iv = if (i % 7 == 0) None else Some(i % 100)
      val sv = if (i % 11 == 0) None else Some(s"repeated_payload_${i % 13}")
      (iv, sv)
    }.groupBy(identity).map { case (k, v) => (k, v.size) }
  }

  /** Extract the (int, string) multiset from deserialized batches. */
  def extractCounts(batches: Seq[ColumnarBatch]): Map[(Option[Int], Option[String]), Int] = {
    val rows = mutable.ArrayBuffer[(Option[Int], Option[String])]()
    batches.foreach { batch =>
      val gpuVecs = GpuColumnVector.extractBases(batch)
      withResource(gpuVecs.safeMap(_.copyToHost())) { hostVecs =>
        val intCol = hostVecs(0)
        val strCol = hostVecs(1)
        (0 until batch.numRows()).foreach { i =>
          val iv = if (intCol.isNull(i)) None else Some(intCol.getInt(i))
          val sv = if (strCol.isNull(i)) None else Some(strCol.getJavaString(i))
          rows.append((iv, sv))
        }
      }
    }
    rows.groupBy(identity).map { case (k, v) => (k, v.size) }
  }

  /**
   * Returns true if the stream contains a compressed record (KUDZ or KUDL magic).
   * Walks record boundaries exactly like the shuffle reader: raw record payloads can
   * contain arbitrary bytes (including stale magics from recycled host buffers), so a
   * byte scan would false-positive.
   */
  def containsCompressedFrame(bytes: Array[Byte]): Boolean = {
    val in = new java.io.DataInputStream(new ByteArrayInputStream(bytes))
    while (true) {
      in.mark(4)
      val first = in.read()
      if (first < 0) {
        return false
      }
      val magic = (first << 24) | (in.read() << 16) | (in.read() << 8) | in.read()
      if (magic == 0x4B55445A || magic == 0x4B55444C) {
        return true
      }
      in.reset()
      val header = com.nvidia.spark.rapids.jni.kudo.KudoTableHeader.readFrom(in).orElse(null)
      if (header == null) {
        return false
      }
      var toSkip = header.getTotalDataLen.toLong
      while (toSkip > 0) {
        val skipped = in.skip(toSkip)
        if (skipped <= 0) {
          return false
        }
        toSkip -= skipped
      }
    }
    false
  }
}

class GpuKudoCompressedShufflePartitioningSuite extends AnyFunSuite with BeforeAndAfterEach
    with TableDrivenPropertyChecks {
  import GpuKudoCompressedShufflePartitioningSuite._

  override def beforeEach(): Unit = {
    super.beforeEach()
    TrampolineUtil.cleanupAnyExistingSession()
  }

  override def afterEach(): Unit = {
    super.afterEach()
    TrampolineUtil.cleanupAnyExistingSession()
  }

  private def createConf(codec: String = "zstd"): SparkConf = {
    // KUDZ only engages under the MULTITHREADED RapidsShuffleManager, so register the
    // shim's manager for the local session.
    val shimVersion = ShimLoader.getShimVersion
    val shuffleManagerClass = s"com.nvidia.spark.rapids.spark" +
      s"${shimVersion.toString.replace(".", "")}." +
      "RapidsShuffleManager"
    new SparkConf()
      .set("spark.shuffle.manager", shuffleManagerClass)
      .set(RapidsConf.SHUFFLE_MANAGER_MODE.key, "MULTITHREADED")
      .set(RapidsConf.SHUFFLE_KUDO_SERIALIZER_ENABLED.key, "true")
      .set(RapidsConf.SHUFFLE_KUDO_COMPRESSION_CODEC.key, codec)
      // Test batches are far below the default gate.
      .set(RapidsConf.SHUFFLE_KUDO_COMPRESSION_MIN_BATCH_SIZE.key, "0")
      // Adaptive mode; beforeEach forces the demand latch on for deterministic
      // engagement, and the adaptive/scan tests override it.
      .set(RapidsConf.SHUFFLE_KUDO_COMPRESSION_MODE.key, "adaptive")
      // The compression path must engage regardless of the configured write mode.
      .set(RapidsConf.SHUFFLE_KUDO_WRITE_MODE.key, "CPU")
      .set(RapidsConf.SHUFFLE_KUDO_READ_MODE.key, "GPU")
  }

  private def createSerializer(): GpuColumnarBatchSerializer = {
    val serializerMetrics = Map[String, GpuMetric]().withDefaultValue(NoopMetric)
    new GpuColumnarBatchSerializer(serializerMetrics, dataTypes,
      ShuffleKudoMode.GPU, useKudo = true, kudoMeasureBufferCopy = false)
  }

  private def setupShuffleDependency(
      inputRDD: org.apache.spark.rdd.RDD[ColumnarBatch],
      serializer: GpuColumnarBatchSerializer,
      numPartitions: Int,
      scanForced: Boolean = true):
      org.apache.spark.ShuffleDependency[Int, ColumnarBatch, ColumnarBatch] = {
    val gpuPartitioning = GpuHashPartitioning(
      Seq(GpuBoundReference(0, IntegerType, nullable = true)(ExprId(0), "key")),
      numPartitions)
    val outputAttributes = Seq(
      AttributeReference("id", IntegerType, nullable = true)(ExprId(0)),
      AttributeReference("name", StringType, nullable = true)(ExprId(1)))
    val writeMetrics = Map[String, SQLMetric]()
    val metrics = Map[String, GpuMetric]().withDefaultValue(NoopMetric)
    // Pass the scan-size decision through the same path the exchange uses; it is
    // applied to the bound partitioner inside prepareBatchShuffleDependency.
    val dependency = GpuShuffleExchangeExecBase.prepareBatchShuffleDependency(
      inputRDD,
      outputAttributes,
      gpuPartitioning,
      dataTypes,
      serializer,
      useGPUShuffle = false,
      useMultiThreadedShuffle = true,
      metrics,
      writeMetrics,
      Map.empty,
      None,
      Seq.empty,
      enableOpTimeTrackingRdd = false,
      kudoScanForced = scanForced)
    dependency
  }

  private def collectBatches(
      dependency: org.apache.spark.ShuffleDependency[Int, ColumnarBatch, ColumnarBatch])
      : mutable.ArrayBuffer[(Int, ColumnarBatch)] = {
    val out = mutable.ArrayBuffer[(Int, ColumnarBatch)]()
    dependency.rdd.partitions.foreach { partition =>
      val it = dependency.rdd.iterator(partition, org.apache.spark.TaskContext.get())
      while (it.hasNext) {
        // The shuffle writer path emits reused MutablePair records, so read the
        // Product2 fields instead of destructuring a Tuple2.
        val record = it.next()
        val partitionId = record._1
        val batch = record._2
        if (batch.numCols() > 0) {
          batch.column(0) match {
            case _: SlicedSerializedColumnVector =>
              SlicedSerializedColumnVector.incRefCount(batch)
            case _: SlicedGpuColumnVector =>
              // Batches the compression gate routed through the CPU slicing path.
              SlicedGpuColumnVector.incRefCount(batch)
            case _ =>
              GpuColumnVector.incRefCounts(batch)
          }
        } else {
          GpuColumnVector.incRefCounts(batch)
        }
        out.append((partitionId, batch))
      }
    }
    out
  }

  /**
   * Serializes the sliced batches through the shuffle serializer, then reads them back
   * through the deserializer and GPU coalesce iterator. Returns the raw stream bytes and
   * the multiset of deserialized rows.
   */
  private def roundTrip(slicedBatches: Seq[ColumnarBatch],
      serializer: GpuColumnarBatchSerializer):
      (Array[Byte], Map[(Option[Int], Option[String]), Int]) = {
    val byteOutputStream = new ByteArrayOutputStream()
    withResource(serializer.newInstance().serializeStream(byteOutputStream)) { ser =>
      slicedBatches.foreach { batch =>
        ser.writeKey(0)
        ser.writeValue(batch)
      }
    }
    val bytes = byteOutputStream.toByteArray
    val deserializationStream =
      serializer.newInstance().deserializeStream(new ByteArrayInputStream(bytes))
    val kudoBatchesIter = deserializationStream.asKeyValueIterator.map(_._2)
      .asInstanceOf[Iterator[ColumnarBatch]]
    val readOption = CoalesceReadOption(
      kudoEnabled = true,
      kudoMode = ShuffleKudoMode.GPU,
      kudoDebugMode = DumpOption.Never,
      kudoDebugDumpPrefix = None,
      useAsync = false)
    val metricsMap = Map[String, GpuMetric]().withDefaultValue(NoopMetric)
    val coalesced = GpuShuffleCoalesceUtils.getGpuShuffleCoalesceIterator(
      kudoBatchesIter, Long.MaxValue, dataTypes, readOption, metricsMap).toSeq
    try {
      (bytes, extractCounts(coalesced))
    } finally {
      coalesced.foreach(_.close())
    }
  }

  private val partitionCounts = Table("numPartitions", 1, 2, 4, 20)

  test("GPU-compressed kudo shuffle round trip preserves data") {
    forAll(partitionCounts) { numPartitions =>
      TrampolineUtil.cleanupAnyExistingSession()
      val conf = createConf()
      TestUtils.withGpuSparkSession(conf) { spark =>
        GpuShuffleEnv.init(new RapidsConf(conf))
        val serializer = createSerializer()
        val numRows = 50000
        val inputRDD = spark.sparkContext.parallelize(Seq(0), numSlices = 1)
          .mapPartitions { _ =>
            Iterator(GpuKudoCompressedShufflePartitioningSuite
              .buildCompressibleBatch(50000))
          }
        val dependency = setupShuffleDependency(inputRDD, serializer, numPartitions)
        val batches = collectBatches(dependency)
        try {
          assert(batches.map(_._2.numRows()).sum == numRows)
          val (bytes, counts) = roundTrip(batches.map(_._2).toSeq, serializer)
          assert(containsCompressedFrame(bytes),
            "expected at least one KUDZ compressed record in the shuffle stream")
          assert(counts == expectedCounts(numRows))
        } finally {
          batches.foreach(_._2.close())
        }
      }
    }
  }

  test("tiny batches fall back to plain kudo records and round trip") {
    TrampolineUtil.cleanupAnyExistingSession()
    val conf = createConf()
    TestUtils.withGpuSparkSession(conf) { spark =>
      GpuShuffleEnv.init(new RapidsConf(conf))
      val serializer = createSerializer()
      val numRows = 10
      val inputRDD = spark.sparkContext.parallelize(Seq(0), numSlices = 1)
        .mapPartitions { _ =>
          Iterator(GpuKudoCompressedShufflePartitioningSuite.buildCompressibleBatch(10))
        }
      val dependency = setupShuffleDependency(inputRDD, serializer, 4)
      val batches = collectBatches(dependency)
      try {
        assert(batches.map(_._2.numRows()).sum == numRows)
        val (_, counts) = roundTrip(batches.map(_._2).toSeq, serializer)
        assert(counts == expectedCounts(numRows))
      } finally {
        batches.foreach(_._2.close())
      }
    }
  }

  /** Builds one compressible batch, shuffles it, and round trips the stream. */
  private def shuffleRoundTrip(spark: org.apache.spark.sql.SparkSession,
      serializer: GpuColumnarBatchSerializer,
      numRows: Int): (Array[Byte], Map[(Option[Int], Option[String]), Int]) = {
    val inputRDD = spark.sparkContext.parallelize(Seq(0), numSlices = 1)
      .mapPartitions { _ =>
        Iterator(GpuKudoCompressedShufflePartitioningSuite.buildCompressibleBatch(numRows))
      }
    val dependency = setupShuffleDependency(inputRDD, serializer, 4)
    val batches = collectBatches(dependency)
    try {
      assert(batches.map(_._2.numRows()).sum == numRows)
      roundTrip(batches.map(_._2).toSeq, serializer)
    } finally {
      batches.foreach(_._2.close())
    }
  }

  test("adaptive scan-size prior engages a large exchange and skips a small one") {
    TrampolineUtil.cleanupAnyExistingSession()
    val conf = createConf()
      .set(RapidsConf.SHUFFLE_KUDO_COMPRESSION_MODE.key, "adaptive")
    TestUtils.withGpuSparkSession(conf) { spark =>
      GpuShuffleEnv.init(new RapidsConf(conf))
      val serializer = createSerializer()
      val numRows = 50000
      def roundTripWithScan(scanForced: Boolean): Array[Byte] = {
        val inputRDD = spark.sparkContext.parallelize(Seq(0), numSlices = 1)
          .mapPartitions { _ =>
            Iterator(GpuKudoCompressedShufflePartitioningSuite.buildCompressibleBatch(numRows))
          }
        val dependency = setupShuffleDependency(inputRDD, serializer, 4, scanForced)
        val batches = collectBatches(dependency)
        try {
          val (bytes, counts) = roundTrip(batches.map(_._2).toSeq, serializer)
          assert(counts == expectedCounts(numRows))
          bytes
        } finally {
          batches.foreach(_._2.close())
        }
      }
      // A large-scan exchange (prior forced) compresses; a small-scan one does not.
      assert(containsCompressedFrame(roundTripWithScan(scanForced = true)),
        "scan-size prior should compress a large-scan exchange")
      assert(!containsCompressedFrame(roundTripWithScan(scanForced = false)),
        "a small-scan exchange should stay uncompressed")
    }
  }

  test("lz4 codec round trips compressed and incompressible data") {
    TrampolineUtil.cleanupAnyExistingSession()
    val conf = createConf(codec = "lz4")
    TestUtils.withGpuSparkSession(conf) { spark =>
      GpuShuffleEnv.init(new RapidsConf(conf))
      val serializer = createSerializer()
      val numRows = 50000
      val (bytes, counts) = shuffleRoundTrip(spark, serializer, numRows)
      assert(containsCompressedFrame(bytes),
        "expected compressed records in the lz4 shuffle stream")
      assert(counts == expectedCounts(numRows))
    }
  }

  test("min batch size gate keeps small batches uncompressed") {
    TrampolineUtil.cleanupAnyExistingSession()
    val conf = createConf()
      // 50k compressible rows serialize to well under this, so the gate must hold
      // compression off while the stream still round trips as plain kudo records.
      .set(RapidsConf.SHUFFLE_KUDO_COMPRESSION_MIN_BATCH_SIZE.key, "1g")
    TestUtils.withGpuSparkSession(conf) { spark =>
      GpuShuffleEnv.init(new RapidsConf(conf))
      val serializer = createSerializer()
      val numRows = 50000
      val inputRDD = spark.sparkContext.parallelize(Seq(0), numSlices = 1)
        .mapPartitions { _ =>
          Iterator(GpuKudoCompressedShufflePartitioningSuite
            .buildCompressibleBatch(50000))
        }
      val dependency = setupShuffleDependency(inputRDD, serializer, 4)
      val batches = collectBatches(dependency)
      try {
        assert(batches.map(_._2.numRows()).sum == numRows)
        val (bytes, counts) = roundTrip(batches.map(_._2).toSeq, serializer)
        assert(!containsCompressedFrame(bytes),
          "expected no KUDZ records below the min batch size gate")
        assert(counts == expectedCounts(numRows))
      } finally {
        batches.foreach(_._2.close())
      }
    }
  }

  test("GPU-compressed kudo shuffle with split retry preserves data") {
    TrampolineUtil.cleanupAnyExistingSession()
    val conf = createConf()
    TestUtils.withGpuSparkSession(conf) { _ =>
      GpuShuffleEnv.init(new RapidsConf(conf))
      val serializer = createSerializer()
      val numRows = 50000
      val gpuPartitioning = GpuHashPartitioning(
        Seq(GpuBoundReference(0, IntegerType, nullable = true)(ExprId(0), "key")), 4)
      // Build the input before arming the injection so the forced OOM lands in the
      // partition/serialize/compress work, then run the same retry composition as the
      // shuffle write path in GpuShuffleExchangeExecBase.
      val spillable = SpillableColumnarBatch(
        GpuKudoCompressedShufflePartitioningSuite.buildCompressibleBatch(numRows),
        SpillPriorities.ACTIVE_ON_DECK_PRIORITY)
      RmmSpark.currentThreadIsDedicatedToTask(1)
      val (batches, retryCount) = try {
        RmmSpark.forceSplitAndRetryOOM(RmmSpark.getCurrentThreadId, 1,
          RmmSpark.OomInjectionType.GPU.ordinal, 0)
        val partitioned = RmmRapidsRetryIterator.withRetry(spillable,
            RmmRapidsRetryIterator.splitSpillableInHalfByRows) { sp =>
          gpuPartitioning.columnarEvalAny(sp.getColumnarBatch())
            .asInstanceOf[Array[(ColumnarBatch, Int)]]
        }.toArray.flatten
        (partitioned, RmmSpark.getAndResetNumSplitRetryThrow(1))
      } finally {
        RmmSpark.removeAllCurrentThreadAssociation()
      }
      try {
        assert(retryCount > 0, s"expected at least one split retry, saw $retryCount")
        assert(batches.map(_._1.numRows()).sum == numRows)
        val (_, counts) = roundTrip(batches.map(_._1).toSeq, serializer)
        assert(counts == expectedCounts(numRows))
      } finally {
        batches.foreach(_._1.close())
      }
    }
  }
}

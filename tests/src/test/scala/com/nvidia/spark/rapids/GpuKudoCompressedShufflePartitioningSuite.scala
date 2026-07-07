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

  /** Returns true if the serialized stream bytes contain the KUDZ frame magic. */
  def containsCompressedFrame(bytes: Array[Byte]): Boolean = {
    val magic = Array[Byte](0x4B, 0x55, 0x44, 0x5A)
    bytes.sliding(4).exists(_.sameElements(magic))
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

  private def createConf(): SparkConf = {
    new SparkConf()
      .set(RapidsConf.SHUFFLE_KUDO_SERIALIZER_ENABLED.key, "true")
      .set(RapidsConf.SHUFFLE_KUDO_COMPRESSION_CODEC.key, "zstd")
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
      numPartitions: Int):
      org.apache.spark.ShuffleDependency[Int, ColumnarBatch, ColumnarBatch] = {
    val gpuPartitioning = GpuHashPartitioning(
      Seq(GpuBoundReference(0, IntegerType, nullable = true)(ExprId(0), "key")),
      numPartitions)
    val outputAttributes = Seq(
      AttributeReference("id", IntegerType, nullable = true)(ExprId(0)),
      AttributeReference("name", StringType, nullable = true)(ExprId(1)))
    val writeMetrics = Map[String, SQLMetric]()
    val metrics = Map[String, GpuMetric]().withDefaultValue(NoopMetric)
    GpuShuffleExchangeExecBase.prepareBatchShuffleDependency(
      inputRDD,
      outputAttributes,
      gpuPartitioning,
      dataTypes,
      serializer,
      useGPUShuffle = false,
      useMultiThreadedShuffle = false,
      metrics,
      writeMetrics,
      Map.empty,
      None,
      Seq.empty,
      enableOpTimeTrackingRdd = false)
  }

  private def collectBatches(
      dependency: org.apache.spark.ShuffleDependency[Int, ColumnarBatch, ColumnarBatch],
      injectOOM: Boolean = false): mutable.ArrayBuffer[(Int, ColumnarBatch)] = {
    val out = mutable.ArrayBuffer[(Int, ColumnarBatch)]()
    var firstIteration = true
    if (injectOOM) {
      RmmSpark.currentThreadIsDedicatedToTask(1)
    }
    try {
      dependency.rdd.partitions.foreach { partition =>
        val it = dependency.rdd.iterator(partition, org.apache.spark.TaskContext.get())
        if (injectOOM && firstIteration && it.hasNext) {
          val threadId = RmmSpark.getCurrentThreadId
          RmmSpark.forceSplitAndRetryOOM(threadId, 1,
            RmmSpark.OomInjectionType.GPU.ordinal, 0)
          firstIteration = false
        }
        while (it.hasNext) {
          val (partitionId, batch) = it.next()
          if (batch.numCols() > 0 &&
              batch.column(0).isInstanceOf[SlicedSerializedColumnVector]) {
            SlicedSerializedColumnVector.incRefCount(batch)
          } else {
            GpuColumnVector.incRefCounts(batch)
          }
          out.append((partitionId, batch))
        }
      }
    } finally {
      if (injectOOM) {
        RmmSpark.removeAllCurrentThreadAssociation()
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

  test("GPU-compressed kudo shuffle with split retry preserves data") {
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
      val dependency = setupShuffleDependency(inputRDD, serializer, 4)
      val batches = collectBatches(dependency, injectOOM = true)
      try {
        val retryCount = RmmSpark.getAndResetNumSplitRetryThrow(1)
        assert(retryCount > 0, s"expected at least one split retry, saw $retryCount")
        assert(batches.map(_._2.numRows()).sum == numRows)
        val (_, counts) = roundTrip(batches.map(_._2).toSeq, serializer)
        assert(counts == expectedCounts(numRows))
      } finally {
        batches.foreach(_._2.close())
      }
    }
  }
}

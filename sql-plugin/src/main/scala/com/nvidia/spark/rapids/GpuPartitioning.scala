/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

import scala.collection.mutable.ArrayBuffer

import ai.rapids.cudf.{BaseDeviceMemoryBuffer, ContiguousTable, Cuda, DeviceMemoryBuffer,
  HostMemoryBuffer, NvtxColor, NvtxRange, Table}
import ai.rapids.cudf.nvcomp.BatchedZstdCompressor
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import com.nvidia.spark.rapids.RmmRapidsRetryIterator.withRetryNoSplit
import com.nvidia.spark.rapids.jni.kudo.KudoGpuSerializer

import org.apache.spark.TaskContext
import org.apache.spark.sql.catalyst.plans.physical.Partitioning
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.rapids.GpuShuffleEnv
import org.apache.spark.sql.vectorized.ColumnarBatch

trait GpuPartitioning extends Partitioning {
  private[this] val (
    maxCpuBatchSize, maxCompressionBatchSize, _useGPUShuffle,
        _useKudoGPUSlicing, _useMultiThreadedShuffle, _kudoCompressionCodec,
        _kudoCompressionChunkSize) = {
    val rapidsConf = new RapidsConf(SQLConf.get)
    val useGpuShuffle = GpuShuffleEnv.useGPUShuffle(rapidsConf)
    val useMtShuffle = GpuShuffleEnv.useMultiThreadedShuffle(rapidsConf)
    // Device-side payload compression is only wired into the built-in Spark sort shuffle
    // path; the RapidsShuffleManager paths manage compression on their own.
    val kudoCompression = if (rapidsConf.shuffleKudoCompressionRequested &&
        !useGpuShuffle && !useMtShuffle) {
      rapidsConf.shuffleKudoCompressionCodec
    } else {
      "none"
    }
    (rapidsConf.shuffleParitioningMaxCpuBatchSize,
      rapidsConf.shuffleCompressionMaxBatchMemory,
      useGpuShuffle,
      rapidsConf.shuffleKudoGpuSerializerEnabled,
      useMtShuffle,
      kudoCompression,
      rapidsConf.shuffleCompressionZstdChunkSize)
  }

  final def columnarEval(batch: ColumnarBatch): GpuColumnVector = {
    throw new IllegalStateException(
      "Partitioners do not support columnarEval, only columnarEvalAny")
  }

  def usesGPUShuffle: Boolean = _useGPUShuffle

  def usesKudoGPUSlicing: Boolean = _useKudoGPUSlicing

  def usesKudoCompression: Boolean = _kudoCompressionCodec != "none"

  def usesMultiThreadedShuffle: Boolean = _useMultiThreadedShuffle

  def sliceBatch(vectors: Array[RapidsHostColumnVector], start: Int, end: Int): ColumnarBatch = {
    var ret: ColumnarBatch = null
    val count = end - start
    if (count > 0) {
      ret = new ColumnarBatch(vectors.map(vec => new SlicedGpuColumnVector(vec, start, end)))
      ret.setNumRows(count)
    }
    ret
  }

  def sliceInternalOnGpuAndClose(numRows: Int, partitionIndexes: Array[Int],
      partitionColumns: Array[GpuColumnVector]): Array[ColumnarBatch] = {
    // The first index will always be 0, so we need to skip it.
    val batches = if (numRows > 0) {
      val parts = partitionIndexes.slice(1, partitionIndexes.length)
      closeOnExcept(new ArrayBuffer[ColumnarBatch](numPartitions)) { splits =>
        val contiguousTables = withResource(partitionColumns) { _ =>
          withResource(new Table(partitionColumns.map(_.getBase).toArray: _*)) { table =>
            table.contiguousSplit(parts: _*)
          }
        }
        GpuShuffleEnv.rapidsShuffleCodec match {
          case Some(codec) =>
            compressSplits(splits, codec, contiguousTables)
          case None =>
            // GpuPackedTableColumn takes ownership of the contiguous tables
            closeOnExcept(contiguousTables) { cts =>
              cts.foreach { ct => splits.append(GpuPackedTableColumn.from(ct)) }
            }
        }
        // synchronize our stream to ensure we have caught up with contiguous split
        // as downstream consumers (RapidsShuffleManager) will add hundreds of buffers
        // to the spill framework, this makes it so here we synchronize once.
        Cuda.DEFAULT_STREAM.sync()
        splits.toArray
      }
    } else {
      Array[ColumnarBatch]()
    }

    GpuSemaphore.releaseIfNecessary(TaskContext.get())
    batches
  }

  private def reslice(batch: ColumnarBatch, numSlices: Int): Seq[ColumnarBatch] = {
    if (batch.numCols() > 0) {
      withResource(batch) { _ =>
        val totalRows = batch.numRows()
        val rowsPerBatch = math.ceil(totalRows.toDouble / numSlices).toInt
        val first = batch.column(0).asInstanceOf[SlicedGpuColumnVector]
        val startOffset = first.getStart
        val endOffset = first.getEnd
        val hostColumns = (0 until batch.numCols()).map { index =>
          batch.column(index).asInstanceOf[SlicedGpuColumnVector].getWrap
        }.toArray

        startOffset.until(endOffset, rowsPerBatch).map { startIndex =>
          val end = math.min(startIndex + rowsPerBatch, endOffset)
          sliceBatch(hostColumns, startIndex, end)
        }.toList
      }
    } else {
      // This should never happen, but...
      Seq(batch)
    }
  }

  def sliceInternalOnCpuAndClose(numRows: Int, partitionIndexes: Array[Int],
      partitionColumns: Array[GpuColumnVector]): Array[(ColumnarBatch, Int)] = {
    // We need to make sure that we have a null count calculated ahead of time.
    // This should be a temp work around.
    partitionColumns.foreach(_.getBase.getNullCount)
    val totalInputSize = GpuColumnVector.getTotalDeviceMemoryUsed(partitionColumns)
    val mightNeedToSplit = totalInputSize > maxCpuBatchSize

    // We have to wrap the NvtxWithMetrics over both copyToHostAsync and corresponding CudaSync,
    // because the copyToHostAsync calls above are not guaranteed to be asynchronous (e.g.: when
    // the copy is from pageable memory, and we're not guaranteed to be using pinned memory).
    val hostPartColumns = NvtxIdWithMetrics(NvtxRegistry.PARTITION_D2H, memCopyTime) {
      val hostColumns = withResource(partitionColumns) { _ =>
        withRetryNoSplit {
          partitionColumns.safeMap(_.copyToHostAsync(Cuda.DEFAULT_STREAM))
        }
      }
      closeOnExcept(hostColumns) { _ =>
        Cuda.DEFAULT_STREAM.sync()
      }
      hostColumns
    }

    withResource(hostPartColumns) { _ =>
      // Leaving the GPU for a while
      GpuSemaphore.releaseIfNecessary(TaskContext.get())

      val origParts = new Array[ColumnarBatch](numPartitions)
      var start = 0
      for (i <- 1 until Math.min(numPartitions, partitionIndexes.length)) {
        val idx = partitionIndexes(i)
        origParts(i - 1) = sliceBatch(hostPartColumns, start, idx)
        start = idx
      }
      origParts(numPartitions - 1) = sliceBatch(hostPartColumns, start, numRows)
      val tmp = origParts.zipWithIndex.filter(_._1 != null)
      // Spark CPU shuffle in some cases has limits on the size of the data a single
      //  row can have. It is a little complicated because the limit is on the compressed
      //  and encrypted buffer, but for now we are just going to assume it is about the same
      // size.
      if (mightNeedToSplit) {
        tmp.flatMap {
          case (batch, part) =>
            val totalSize = if (batch.numCols() > 0) {
              batch.column(0) match {
                case _: SlicedGpuColumnVector =>
                  SlicedGpuColumnVector.getTotalHostMemoryUsed(batch)
                case _: SlicedSerializedColumnVector =>
                  SlicedSerializedColumnVector.getTotalHostMemoryUsed(batch)
                case _ =>
                  0L
              }
            } else {
              0L
            }
            val numOutputBatches =
              math.ceil(totalSize.toDouble / maxCpuBatchSize).toInt
            if (numOutputBatches > 1) {
              // For now we are going to slice it on number of rows instead of looking
              // at each row to try and decide. If we get in trouble we can probably
              // make this recursive and keep splitting more until it is small enough.
              reslice(batch, numOutputBatches).map { subBatch =>
                (subBatch, part)
              }
            } else {
              Seq((batch, part))
            }
        }
      } else {
        tmp
      }
    }
  }

  private def gpuSplitAndSerialize(table: Table, slices: Int*): Array[DeviceMemoryBuffer] = {
    NvtxRegistry.GPU_KUDO_SERIALIZE {
      withRetryNoSplit {
        KudoGpuSerializer.splitAndSerializeToDevice(table, slices: _*)
      }
    }
  }

  private def sliceAndSerializeOnGpu(numRows: Int, partitionIndexes: Array[Int],
      partitionColumns: Array[GpuColumnVector]): Array[(ColumnarBatch, Int)] = {
    partitionColumns.foreach(_.getBase.getNullCount)
    val (dataHost, offsetsHost) = withResource(partitionColumns) { _ =>
      withResource(new Table(partitionColumns.map(_.getBase).toArray: _*)) { table =>
        withResource(gpuSplitAndSerialize(table,
          partitionIndexes.tail: _*)) { dmbs =>
          val data = dmbs(0)
          val offsets = dmbs(1)
          closeOnExcept(Seq(HostMemoryBuffer.allocate(data.getLength),
            HostMemoryBuffer.allocate(offsets.getLength))) { seq =>
            val dataHost = seq(0)
            val offsetsHost = seq(1)
            NvtxRegistry.GPU_KUDO_COPY_TO_HOST {
              dataHost.copyFromDeviceBufferAsync(data, Cuda.DEFAULT_STREAM)
              offsetsHost.copyFromDeviceBufferAsync(offsets, Cuda.DEFAULT_STREAM)
              Cuda.DEFAULT_STREAM.sync()
            }
            (dataHost, offsetsHost)
          }
        }
      }
    }
    GpuSemaphore.releaseIfNecessary(TaskContext.get())

    NvtxRegistry.GPU_KUDO_SLICE_BUFFERS {
      withResource(Seq(dataHost, offsetsHost)) { _ =>
        val numSlices = numPartitions + 1
        val elemSize = offsetsHost.getLength / numSlices

        val res = new Array[ColumnarBatch](numPartitions)
        var start = 0
        var prevIndex: Int = 0
        for (i <- 1 until numPartitions) {
          val idx = offsetsHost.getLong((i) * elemSize).toInt
          val partNumRows = partitionIndexes(i) - prevIndex
          if (partNumRows > 0) {
            res(i - 1) = new ColumnarBatch(Array(
              new SlicedSerializedColumnVector(dataHost, start, idx)))
            res(i - 1).setNumRows(partNumRows)
          }
          prevIndex = partitionIndexes(i)
          start = idx
        }
        val partNumRows = numRows - prevIndex
        if (partNumRows > 0) {
          res(numPartitions - 1) = new ColumnarBatch(Array(
            new SlicedSerializedColumnVector(dataHost, start, dataHost.getLength.toInt)))
          res(numPartitions - 1).setNumRows(partNumRows)
        }

        res.zipWithIndex.filter(_._1 != null)
      }
    }
  }

  /**
   * Writes the KudoCompressedFrame header for one compressed record. All fields are
   * big-endian to match the DataInput framing used on the read side.
   */
  private def writeCompressedFrameHeader(buf: HostMemoryBuffer, offset: Long, numRows: Int,
      chunkSize: Int, uncompressedLen: Long, compressedLen: Int): Unit = {
    buf.setInt(offset, Integer.reverseBytes(KudoCompressedFrame.MAGIC))
    buf.setInt(offset + 4, Integer.reverseBytes(numRows))
    buf.setInt(offset + 8, Integer.reverseBytes(chunkSize))
    buf.setLong(offset + 12, java.lang.Long.reverseBytes(uncompressedLen))
    buf.setInt(offset + 20, Integer.reverseBytes(compressedLen))
  }

  /**
   * Serializes the partitions of the given batch into kudo format on the device, then
   * compresses each partition's serialized bytes on the device with batched nvcomp ZSTD
   * before a single copy to the host. Partitions whose compressed form is not smaller are
   * written as plain kudo records, so the resulting shuffle stream can mix "KUD0" and
   * "KUDZ" records. The GPU is idle during shuffle write on the CPU path while the CPU
   * pays for serialization and compression; moving both to the device reduces host CPU
   * and shrinks the bytes that reach the shuffle file.
   */
  private def sliceSerializeCompressOnGpu(numRows: Int, partitionIndexes: Array[Int],
      partitionColumns: Array[GpuColumnVector]): Array[(ColumnarBatch, Int)] = {
    val frameBytes = KudoCompressedFrame.FRAME_HEADER_BYTES
    require(_kudoCompressionChunkSize > 0 && _kudoCompressionChunkSize <= Int.MaxValue,
      s"invalid kudo compression chunk size ${_kudoCompressionChunkSize}")
    val chunkSize = _kudoCompressionChunkSize.toInt
    partitionColumns.foreach(_.getBase.getNullCount)
    val (frameHost, entries) = withResource(partitionColumns) { _ =>
      withResource(new Table(partitionColumns.map(_.getBase).toArray: _*)) { table =>
        withResource(gpuSplitAndSerialize(table, partitionIndexes.tail: _*)) { dmbs =>
          val dataDev = dmbs(0)
          val offsetsDev = dmbs(1)
          // The number of serialized regions equals the number of partition start indexes
          // (this can be fewer than numPartitions, e.g. a range partition with no bounds).
          val numRegions = partitionIndexes.length
          // Pull the numRegions + 1 size_t region offsets to the host.
          val offsets = withResource(HostMemoryBuffer.allocate(offsetsDev.getLength)) { oh =>
            oh.copyFromDeviceBuffer(offsetsDev)
            val elemSize = offsetsDev.getLength / (numRegions + 1)
            (0 to numRegions).map(i => oh.getLong(i * elemSize)).toArray
          }
          val rowsPerPartition = Array.tabulate(numRegions) { i =>
            if (i < numRegions - 1) {
              partitionIndexes(i + 1) - partitionIndexes(i)
            } else {
              numRows - partitionIndexes(numRegions - 1)
            }
          }
          val included = (0 until numRegions).filter(rowsPerPartition(_) > 0).toArray
          if (included.isEmpty) {
            (null, Array.empty[(Int, Long, Long, Int)])
          } else {
            // Compress all non-empty partition regions in one batched device call. The
            // compressor closes the input slices, which only drops their extra
            // references on the underlying serialized buffer.
            val compressed = withResource(new NvtxRange("gpuKudoCompress", NvtxColor.ORANGE)) {
              _ =>
              val inputs: Array[BaseDeviceMemoryBuffer] = included.map { i =>
                dataDev.slice(offsets(i), offsets(i + 1) - offsets(i))
                  .asInstanceOf[BaseDeviceMemoryBuffer]
              }
              val compressor = new BatchedZstdCompressor(chunkSize, maxCompressionBatchSize)
              compressor.compress(inputs, Cuda.DEFAULT_STREAM)
            }
            withResource(compressed.toSeq) { _ =>
              // Frame layout and per-partition compressed-vs-raw decision.
              val frameLens = included.indices.map { k =>
                val rawLen = offsets(included(k) + 1) - offsets(included(k))
                val compLen = compressed(k).getLength
                if (compLen + frameBytes < rawLen) frameBytes + compLen else rawLen
              }
              val total = frameLens.sum
              require(total <= Int.MaxValue,
                s"Compressed shuffle batch is too large: $total bytes")
              val hostBuf = withRetryNoSplit[HostMemoryBuffer] {
                HostMemoryBuffer.allocate(total)
              }
              closeOnExcept(hostBuf) { _ =>
                withResource(new NvtxRange("gpuKudoCompressD2H", NvtxColor.PURPLE)) { _ =>
                  var cursor = 0L
                  val entries = included.indices.map { k =>
                    val i = included(k)
                    val rawLen = offsets(i + 1) - offsets(i)
                    val compLen = compressed(k).getLength
                    val start = cursor
                    if (compLen + frameBytes < rawLen) {
                      writeCompressedFrameHeader(hostBuf, cursor, rowsPerPartition(i),
                        chunkSize, rawLen, compLen.toInt)
                      hostBuf.copyFromMemoryBufferAsync(cursor + frameBytes, compressed(k),
                        0, compLen, Cuda.DEFAULT_STREAM)
                      cursor += frameBytes + compLen
                    } else {
                      // Not worth compressing; emit the plain kudo record bytes.
                      hostBuf.copyFromMemoryBufferAsync(cursor, dataDev, offsets(i),
                        rawLen, Cuda.DEFAULT_STREAM)
                      cursor += rawLen
                    }
                    (i, start, cursor, rowsPerPartition(i))
                  }.toArray
                  Cuda.DEFAULT_STREAM.sync()
                  (hostBuf, entries)
                }
              }
            }
          }
        }
      }
    }
    GpuSemaphore.releaseIfNecessary(TaskContext.get())

    if (frameHost == null) {
      Array.empty[(ColumnarBatch, Int)]
    } else {
      NvtxRegistry.GPU_KUDO_SLICE_BUFFERS {
        withResource(frameHost) { _ =>
          entries.map { case (partId, start, end, partRows) =>
            val cb = new ColumnarBatch(Array(
              new SlicedSerializedColumnVector(frameHost, start.toInt, end.toInt)))
            cb.setNumRows(partRows)
            (cb, partId)
          }
        }
      }
    }
  }

  def sliceInternalGpuOrCpuAndClose(numRows: Int, partitionIndexes: Array[Int],
      partitionColumns: Array[GpuColumnVector]): Array[(ColumnarBatch, Int)] = {
    if (usesKudoCompression && numRows > 0 && partitionColumns.nonEmpty) {
      sliceSerializeCompressOnGpu(numRows, partitionIndexes, partitionColumns)
    } else if (usesKudoGPUSlicing) {
      sliceAndSerializeOnGpu(numRows, partitionIndexes, partitionColumns)
    } else {
      val sliceOnGpu = usesGPUShuffle
      val nvtxId = if (sliceOnGpu) {
        NvtxRegistry.SLICE_INTERNAL_GPU
      } else {
        NvtxRegistry.SLICE_INTERNAL_CPU
      }
      // If we are not using the Rapids shuffle we fall back to CPU splits way to avoid the hit
      // for large number of small splits.
      nvtxId {
        if (sliceOnGpu) {
          val tmp = sliceInternalOnGpuAndClose(numRows, partitionIndexes, partitionColumns)
          tmp.zipWithIndex.filter(_._1 != null)
        } else {
          sliceInternalOnCpuAndClose(numRows, partitionIndexes, partitionColumns)
        }
      }
    }
  }

  /**
   * Compress contiguous tables representing the splits into compressed columnar batches.
   * Contiguous tables corresponding to splits with no data will not be compressed.
   * @param outputBatches where to collect the corresponding columnar batches for the splits
   * @param codec compression codec to use
   * @param contiguousTables contiguous tables to compress
   */
  def compressSplits(
      outputBatches: ArrayBuffer[ColumnarBatch],
      codec: TableCompressionCodec,
      contiguousTables: Array[ContiguousTable]): Unit = {
    withResource(codec.createBatchCompressor(maxCompressionBatchSize,
        Cuda.DEFAULT_STREAM)) { compressor =>
      // tracks batches with no data and the corresponding output index for the batch
      val emptyBatches = new ArrayBuffer[(ColumnarBatch, Int)]

      // add each table either to the batch to be compressed or to the empty batch tracker
      contiguousTables.zipWithIndex.foreach { case (ct, i) =>
        if (ct.getRowCount == 0) {
          emptyBatches.append((GpuPackedTableColumn.from(ct), i))
        } else {
          compressor.addTableToCompress(ct)
        }
      }

      withResource(compressor.finish()) { compressedTables =>
        var compressedTableIndex = 0
        var outputIndex = 0
        emptyBatches.foreach { case (emptyBatch, emptyOutputIndex) =>
          require(emptyOutputIndex >= outputIndex)
          // add any compressed batches that need to appear before the next empty batch
          val numCompressedToAdd = emptyOutputIndex - outputIndex
          (0 until numCompressedToAdd).foreach { _ =>
            val compressedTable = compressedTables(compressedTableIndex)
            outputBatches.append(GpuCompressedColumnVector.from(compressedTable))
            compressedTableIndex += 1
          }
          outputBatches.append(emptyBatch)
          outputIndex = emptyOutputIndex + 1
        }

        // add any compressed batches that remain after the last empty batch
        (compressedTableIndex until compressedTables.length).foreach { i =>
          val ct = compressedTables(i)
          outputBatches.append(GpuCompressedColumnVector.from(ct))
        }
      }
    }
  }

  private var memCopyTime: GpuMetric = NoopMetric

  /**
   * Setup sub-metrics for the performance debugging of GpuPartition. This method is expected to
   * be called at the query planning stage. Therefore, this method is NOT thread safe.
   */
  def setupDebugMetrics(metrics: Map[String, GpuMetric]): Unit = {
    metrics.get(GpuMetric.COPY_TO_HOST_TIME).foreach(memCopyTime = _)
  }
}

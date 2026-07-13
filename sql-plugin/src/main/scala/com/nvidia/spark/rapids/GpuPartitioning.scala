/*
 * Copyright (c) 2020-2026, NVIDIA CORPORATION.
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
import ai.rapids.cudf.nvcomp.{BatchedCompressor, BatchedLZ4Compressor, BatchedZstdCompressor}
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import com.nvidia.spark.rapids.RmmRapidsRetryIterator.withRetryNoSplit
import com.nvidia.spark.rapids.jni.kudo.KudoGpuSerializer

import org.apache.spark.TaskContext
import org.apache.spark.sql.catalyst.plans.physical.Partitioning
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.rapids.GpuShuffleEnv
import org.apache.spark.sql.vectorized.ColumnarBatch

object GpuPartitioning {
  /**
   * KUDZ wraps records moved by the RapidsShuffleManager MULTITHREADED writer, the
   * standard GPU deployment. The built-in Spark sort shuffle stays uncompressed, and
   * UCX/CACHE_ONLY use the GPU shuffle transport and its native compression format.
   */
  private[rapids] def supportsKudoCompression(
      useGpuShuffle: Boolean,
      useMultiThreadedShuffle: Boolean): Boolean = {
    require(!(useGpuShuffle && useMultiThreadedShuffle),
      "GPU shuffle and MULTITHREADED shuffle cannot both be active")
    useMultiThreadedShuffle
  }
}

trait GpuPartitioning extends Partitioning {
  private[this] val (
    maxCpuBatchSize, maxCompressionBatchSize, _useGPUShuffle,
        _useKudoGPUSlicing, _useMultiThreadedShuffle, _kudoCompressionCodec,
        _kudoCompressionChunkSize, _kudoCompressionMinBatchSize) = {
    val rapidsConf = new RapidsConf(SQLConf.get)
    val useGpuShuffle = GpuShuffleEnv.useGPUShuffle(rapidsConf)
    val useMtShuffle = GpuShuffleEnv.useMultiThreadedShuffle(rapidsConf)
    // Device-side payload compression is wired into the RapidsShuffleManager
    // MULTITHREADED writer: records are opaque bytes to its limiter/spill/merge
    // pipeline, and the read side dispatches per record on the KUD0/KUDZ magic.
    // The UCX/CACHE_ONLY GPU-transport paths manage compression on their own, and the
    // built-in Spark sort shuffle stays uncompressed.
    val kudoCompression = if (rapidsConf.shuffleKudoCompressionRequested &&
        rapidsConf.shuffleKudoCompressionMode != "never" &&
        GpuPartitioning.supportsKudoCompression(useGpuShuffle, useMtShuffle)) {
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
      rapidsConf.shuffleCompressionZstdChunkSize,
      rapidsConf.shuffleKudoCompressionMinBatchSize)
  }

  // Lift once GPU shuffle supports long (64-bit) serialized-slice offsets.
  // protected[rapids] so tests can override it to exercise the guard below.
  protected[rapids] def maxGpuSerializedSliceBytes: Long = Int.MaxValue

  // Set on the driver by the exchange when this exchange's estimated map-side scan
  // input reaches the scan-size threshold. A large scan predicts a large shuffle,
  // which is where device-side compression pays; small exchanges stay uncompressed
  // so compute-bound workloads are never taxed. A plain var so it serializes to
  // executors like the debug metrics; only set under adaptive mode by the exchange.
  private var _kudoCompressionScanForced: Boolean = false

  def setKudoCompressionScanForced(enabled: Boolean): Unit = {
    _kudoCompressionScanForced = enabled
  }

  private def kudoCompressionEngaged: Boolean = _kudoCompressionScanForced

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

  private type SerializedEntry = (Int, Long, Long, Int)

  private def readSerializedOffsets(offsetsHost: HostMemoryBuffer,
      numRegions: Int): Array[Long] = {
    val numOffsets = numRegions + 1
    require(offsetsHost.getLength % numOffsets == 0,
      s"Invalid kudo offsets buffer length ${offsetsHost.getLength} for $numRegions regions")
    val elemSize = offsetsHost.getLength / numOffsets
    require(elemSize == java.lang.Long.BYTES,
      s"Unsupported kudo offset width: $elemSize bytes")
    Array.tabulate(numOffsets)(i => offsetsHost.getLong(i * elemSize))
  }

  private def serializedEntries(offsets: Array[Long],
      rowsPerPartition: Array[Int]): Array[SerializedEntry] = {
    rowsPerPartition.indices.collect {
      case i if rowsPerPartition(i) > 0 =>
        (i, offsets(i), offsets(i + 1), rowsPerPartition(i))
    }.toArray
  }

  private def copySerializedPartitionsToHost(
      dataDev: DeviceMemoryBuffer,
      offsetsDev: DeviceMemoryBuffer,
      rowsPerPartition: Array[Int]): (HostMemoryBuffer, Array[SerializedEntry]) = {
    // This bound keeps the SlicedSerializedColumnVector Long-to-Int narrowing lossless.
    require(dataDev.getLength <= maxGpuSerializedSliceBytes,
      s"GPU-serialized shuffle batch is ${dataDev.getLength} bytes, exceeding the " +
        s"$maxGpuSerializedSliceBytes-byte (2GB) limit addressable by the Int " +
        s"serialized-slice offsets; reduce spark.rapids.sql.batchSizeBytes")
    val dataHost = HostMemoryBuffer.allocate(dataDev.getLength)
    closeOnExcept(dataHost) { _ =>
      withResource(HostMemoryBuffer.allocate(offsetsDev.getLength)) { offsetsHost =>
        NvtxRegistry.GPU_KUDO_COPY_TO_HOST {
          dataHost.copyFromDeviceBufferAsync(dataDev, Cuda.DEFAULT_STREAM)
          offsetsHost.copyFromDeviceBufferAsync(offsetsDev, Cuda.DEFAULT_STREAM)
          Cuda.DEFAULT_STREAM.sync()
        }
        val offsets = readSerializedOffsets(offsetsHost, rowsPerPartition.length)
        (dataHost, serializedEntries(offsets, rowsPerPartition))
      }
    }
  }

  /**
   * Writes the KudoCompressedFrame header for one compressed record. All fields are
   * big-endian to match the DataInput framing used on the read side.
   */
  private def writeCompressedFrameHeader(buf: HostMemoryBuffer, offset: Long, numRows: Int,
      chunkSize: Int, uncompressedLen: Long, compressedLen: Int): Unit = {
    buf.setInt(offset, Integer.reverseBytes(KudoCompressedFrame.magicFor(_kudoCompressionCodec)))
    buf.setInt(offset + 4, Integer.reverseBytes(numRows))
    buf.setInt(offset + 8, Integer.reverseBytes(chunkSize))
    buf.setLong(offset + 12, java.lang.Long.reverseBytes(uncompressedLen))
    buf.setInt(offset + 20, Integer.reverseBytes(compressedLen))
  }

  private def compressSerializedPartitionsToHost(
      dataDev: DeviceMemoryBuffer,
      offsetsDev: DeviceMemoryBuffer,
      rowsPerPartition: Array[Int]): (HostMemoryBuffer, Array[SerializedEntry]) = {
    val frameBytes = KudoCompressedFrame.FRAME_HEADER_BYTES
    require(_kudoCompressionChunkSize > 0 && _kudoCompressionChunkSize <= Int.MaxValue,
      s"invalid kudo compression chunk size ${_kudoCompressionChunkSize}")
    val chunkSize = _kudoCompressionChunkSize.toInt
    val offsets = withResource(HostMemoryBuffer.allocate(offsetsDev.getLength)) { offsetsHost =>
      offsetsHost.copyFromDeviceBuffer(offsetsDev)
      readSerializedOffsets(offsetsHost, rowsPerPartition.length)
    }
    val included = rowsPerPartition.indices.filter(rowsPerPartition(_) > 0).toArray
    if (included.isEmpty) {
      return (null, Array.empty)
    }

    // Compress all non-empty regions in one call. The compressor closes the input slices,
    // which only drops their extra references on the underlying serialized buffer.
    val compressed = withResource(new NvtxRange("gpuKudoCompress", NvtxColor.ORANGE)) { _ =>
      val inputs: Array[BaseDeviceMemoryBuffer] = included.map { i =>
        dataDev.slice(offsets(i), offsets(i + 1) - offsets(i))
          .asInstanceOf[BaseDeviceMemoryBuffer]
      }
      val compressor: BatchedCompressor = _kudoCompressionCodec match {
        case "lz4" => new BatchedLZ4Compressor(chunkSize, maxCompressionBatchSize)
        case _ => new BatchedZstdCompressor(chunkSize, maxCompressionBatchSize)
      }
      compressor
        .compress(inputs, Cuda.DEFAULT_STREAM)
    }
    withResource(compressed.toSeq) { _ =>
      val outputLengths = included.indices.map { k =>
        val rawLength = offsets(included(k) + 1) - offsets(included(k))
        val compressedLength = compressed(k).getLength
        if (compressedLength + frameBytes < rawLength) {
          frameBytes + compressedLength
        } else {
          rawLength
        }
      }.toArray
      val totalLength = outputLengths.foldLeft(0L) { (total, length) =>
        Math.addExact(total, length)
      }
      require(totalLength <= Int.MaxValue,
        s"Compressed shuffle batch is too large: $totalLength bytes")
      val hostBuffer = withRetryNoSplit[HostMemoryBuffer] {
        HostMemoryBuffer.allocate(totalLength)
      }
      closeOnExcept(hostBuffer) { _ =>
        withResource(new NvtxRange("gpuKudoCompressD2H", NvtxColor.PURPLE)) { _ =>
          var outputOffset = 0L
          val entries = included.indices.map { k =>
            val partitionId = included(k)
            val rawLength = offsets(partitionId + 1) - offsets(partitionId)
            val compressedLength = compressed(k).getLength
            val start = outputOffset
            if (compressedLength + frameBytes < rawLength) {
              writeCompressedFrameHeader(hostBuffer, outputOffset,
                rowsPerPartition(partitionId), chunkSize, rawLength, compressedLength.toInt)
              hostBuffer.copyFromMemoryBufferAsync(outputOffset + frameBytes, compressed(k),
                0, compressedLength, Cuda.DEFAULT_STREAM)
            } else {
              hostBuffer.copyFromMemoryBufferAsync(outputOffset, dataDev,
                offsets(partitionId), rawLength, Cuda.DEFAULT_STREAM)
            }
            outputOffset += outputLengths(k)
            (partitionId, start, outputOffset, rowsPerPartition(partitionId))
          }.toArray
          Cuda.DEFAULT_STREAM.sync()
          (hostBuffer, entries)
        }
      }
    }
  }

  private def sliceSerializedHostBuffer(hostBuffer: HostMemoryBuffer,
      entries: Array[SerializedEntry]): Array[(ColumnarBatch, Int)] = {
    if (hostBuffer == null) {
      Array.empty
    } else {
      NvtxRegistry.GPU_KUDO_SLICE_BUFFERS {
        withResource(hostBuffer) { _ =>
          entries.map { case (partitionId, start, end, numRows) =>
            val batch = new ColumnarBatch(Array(
              new SlicedSerializedColumnVector(hostBuffer, start.toInt, end.toInt)))
            batch.setNumRows(numRows)
            (batch, partitionId)
          }
        }
      }
    }
  }

  /**
   * Serializes each non-empty partition to Kudo on the GPU. KUDZ optionally compresses
   * those same serialized regions before their single copy to host; all setup and slicing
   * is shared with the existing raw KUD0 path. Batches whose serialized bytes fall below
   * the minimum-batch-size gate skip compression entirely: the per-batch cost of the
   * compression pass (kernel launch, device-to-host staging, stream sync, and a longer
   * GPU-semaphore hold) is fixed, so on small batches it is pure overhead while the byte
   * savings are too small to relieve any downstream bottleneck.
   */
  private def sliceAndSerializeOnGpu(numRows: Int, partitionIndexes: Array[Int],
      partitionColumns: Array[GpuColumnVector],
      compressBatch: Boolean): Array[(ColumnarBatch, Int)] = {
    partitionColumns.foreach(_.getBase.getNullCount)
    val numRegions = partitionIndexes.length
    val rowsPerPartition = Array.tabulate(numRegions) { i =>
      if (i + 1 < numRegions) {
        partitionIndexes(i + 1) - partitionIndexes(i)
      } else {
        numRows - partitionIndexes(i)
      }
    }
    val (hostBuffer, entries) = withResource(partitionColumns) { _ =>
      withResource(new Table(partitionColumns.map(_.getBase).toArray: _*)) { table =>
        withResource(gpuSplitAndSerialize(table, partitionIndexes.tail: _*)) { buffers =>
          if (compressBatch && buffers(0).getLength >= _kudoCompressionMinBatchSize) {
            compressSerializedPartitionsToHost(buffers(0), buffers(1), rowsPerPartition)
          } else {
            copySerializedPartitionsToHost(buffers(0), buffers(1), rowsPerPartition)
          }
        }
      }
    }
    GpuSemaphore.releaseIfNecessary(TaskContext.get())
    sliceSerializedHostBuffer(hostBuffer, entries)
  }

  def sliceInternalGpuOrCpuAndClose(numRows: Int, partitionIndexes: Array[Int],
      partitionColumns: Array[GpuColumnVector]): Array[(ColumnarBatch, Int)] = {
    // Decide compression BEFORE committing to device-side serialization: when the
    // gate declines (adaptive pressure low, or the batch is under the size floor),
    // the batch must take the configured write path. Routing a batch onto the GPU
    // serialize path without compressing it pays GPU time and semaphore holds for
    // nothing -- measured at +20% on NDS SF3K when every batch did so.
    val wantCompress = usesKudoCompression && kudoCompressionEngaged &&
      numRows > 0 && partitionColumns.nonEmpty && {
        val estimatedBytes = partitionColumns.map(_.getBase.getDeviceMemorySize).sum
        estimatedBytes >= _kudoCompressionMinBatchSize
      }
    if (usesKudoCompression && numRows > 0 && partitionColumns.nonEmpty) {
      (if (wantCompress) kudoCompressedBatches else kudoUncompressedBatches) += 1
    }
    if (wantCompress || usesKudoGPUSlicing) {
      sliceAndSerializeOnGpu(numRows, partitionIndexes, partitionColumns, wantCompress)
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
  private var kudoCompressedBatches: GpuMetric = NoopMetric
  private var kudoUncompressedBatches: GpuMetric = NoopMetric

  /**
   * Setup sub-metrics for the performance debugging of GpuPartition. This method is expected to
   * be called at the query planning stage. Therefore, this method is NOT thread safe.
   */
  def setupDebugMetrics(metrics: Map[String, GpuMetric]): Unit = {
    metrics.get(GpuMetric.COPY_TO_HOST_TIME).foreach(memCopyTime = _)
    metrics.get(GpuMetric.KUDO_COMPRESSED_BATCHES).foreach(kudoCompressedBatches = _)
    metrics.get(GpuMetric.KUDO_UNCOMPRESSED_BATCHES).foreach(kudoUncompressedBatches = _)
  }
}

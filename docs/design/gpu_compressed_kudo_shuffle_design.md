---
layout: page
title: GPU-Compressed Kudo Shuffle Payload
nav_order: 17
parent: Developer Overview
---
# GPU-Compressed Kudo Shuffle Payload (nvcomp ZSTD)

## 1. Background & Motivation

### The Measured Problem

With the built-in Spark sort shuffle and the kudo serializer (write mode
CPU, read mode GPU — the defaults), every shuffled byte makes this journey:

```
GPU partition -> D2H raw columns -> CPU slice -> CPU kudo serialize
  -> CPU lz4 (~1.85x) -> bypass-merge writer (bytes written TWICE)
  -> fetch -> CPU lz4 decode -> host concat -> H2D -> GPU assemble
```

On a measured NDS-H SF100 cluster (2 executors, 4 cores each, one shared
disk) this path dominated the benchmark twice over:

1. **CPU**: shuffle-heavy reduce stages ran at 74-84% CPU (lz4 plus kudo
   serialization) on 4-core hosts, while the GPUs were under 2% busy.
2. **Disk + page cache**: one query iteration wrote ~16GB per node of
   shuffle traffic (bypass-merge writes bytes twice), evicting the input
   page cache on the shared disk. Scan-stage blocked time exploded across
   iterations (12s -> 45s wall with constant CPU) as inputs were re-read
   cold. Shuffle churn and scan time were coupled through the cache.

Two knob-only diagnostics established that no existing configuration fixes
this: CPU zstd (`spark.io.compression.codec=zstd`) cut shuffle bytes 31%
but doubled CPU (net zero); the kudo GPU write mode moved serialization to
the GPU but *increased* on-disk bytes 13% (lz4 compresses the padded GPU
layout worse).

### The Key Insight

The kudo GPU serializer (`KudoGpuSerializer.splitAndSerializeToDevice`)
already produces, **on the device**, one contiguous buffer of per-partition
kudo records plus an offsets vector. That is exactly the shape batched
nvcomp compression wants, and the GPU is the one idle resource. Compressing
there gets the byte reduction of zstd *and* removes the serialize+compress
work from the CPU — the combination no knob can reach.

## 2. Design

### Write Path (`GpuPartitioning.sliceSerializeCompressOnGpu`)

1. `splitAndSerializeToDevice` produces the serialized partitions and
   offsets on the device (region count is keyed off
   `partitionIndexes.length`, which also fixes a latent mismatch for range
   partitions without bounds).
2. All non-empty partition regions are compressed in a single
   `BatchedZstdCompressor` call (per-partition device views; 64KB chunks;
   the existing `spark.rapids.shuffle.compression.zstd.chunkSize`).
3. A single framed host buffer is assembled with one async D2H per
   partition payload and one stream sync. Per partition, whichever is
   smaller wins:
   - compressed: a `KUDZ` frame (below), or
   - raw: the plain kudo record bytes (`KUD0...`), so incompressible
     partitions cost nothing.
4. The host buffer is sliced into the usual per-partition
   `SlicedSerializedColumnVector` records; the GPU-mode serializer instance
   writes them to the shuffle stream verbatim (zero serializer changes).

### Stream Framing

```
int  magic = KUDZ          (0x4B55445A; plain records keep kudo's KUD0)
int  numRows
int  chunkSize             (uncompressed chunk size used by the compressor)
long uncompressedLength    (length of the raw kudo record: header+payload)
int  compressedLength
byte[compressedLength]     (nvcomp batched-zstd blob: chunk metadata + chunks)
```

All fields are big-endian to match the kudo header framing. Streams may
freely mix `KUD0` and `KUDZ` records; the deserializer dispatches on the
4-byte magic (mark/reset on the buffered stream). Because the wire format
supports per-record choice, future *adaptive* policies (e.g. skip
compression when the GPU is contended) need no format change.

### Read Path

`KudoSerializedBatchIterator` parses `KUDZ` frames into
`KudoSerializedTableColumn`s carrying a `KudoCompressedMeta`
(numRows, uncompressedLength, chunkSize) and a spillable compressed host
buffer; batch-size accounting (`getDataLen`) reports uncompressed bytes so
coalesce targeting and join sizing see GPU-relevant sizes.

`KudoGpuTableOperator.concat` (the GPU coalesce operator) handles a batch
containing compressed records by:

1. Host-concatenating the compressed blobs (one H2D copy — now ~2.6x
   smaller than before).
2. Batch-decompressing all records in one `BatchedZstdDecompressor` call,
   with per-record output views placed **directly at their slots in the
   raw record layout** that `KudoGpuSerializer.assembleFromDeviceRaw`
   expects (compressing whole records — header included — makes
   decompression output assemble-ready with no device-side stitching).
   Plain records in the same batch are copied straight into their slots.
3. Assembling as before.

The sized-join spillable path (`KudoCompressedSpillableHostConcatResult`)
preserves the metadata across spill round trips. The CPU merge operator and
debug dump reject compressed records loudly; they are unreachable because
compression only engages when the kudo GPU read mode is active.

### Guards

A new internal conf `spark.rapids.shuffle.kudo.serializer.compression.codec`
(`zstd` | `none`) gates the path. It engages only with: kudo enabled, kudo
GPU read mode, and the built-in Spark sort shuffle (the RapidsShuffleManager
paths manage compression themselves).

## 3. Correctness

- Exact saved-output comparison (schema equality plus `exceptAll` both
  directions) passed for the full 22-query NDS-H suite against the
  unmodified release baseline.
- A synthetic GPU-vs-CPU test covering nulls, empty partitions, extreme
  skew, unicode/empty strings, decimal64/decimal128, arrays, structs, and
  array<struct> passes.
- `GpuKudoCompressedShufflePartitioningSuite` round-trips the write and
  read paths, asserts compression actually engaged (KUDZ magic present),
  exercises the tiny-batch raw fallback, and passes under OOM-injected
  split-and-retry.
- No spills, task failures, or new CPU fallbacks in any validation run.

## 4. Performance Results (2-worker H200 cluster, NDS-H SF100)

Interleaved unprofiled A/B against the release baseline (B,C,C,B; warmup 1,
3 measured iterations each):

- Full-stream power test: 1257s/1421s -> 620s/625s (**2.15x**); sum of
  per-query medians 308.2s -> 141.0s (**2.19x**). Candidate run-to-run
  spread fell from 13% to 0.8% (the cache-eviction feedback loop is gone).
- Biggest per-query gains: q9 5.6x, q8 3.9x, q5 3.3x, q4 2.2x. Scan-only
  queries (q6, q15) are unchanged, as expected.
- Mechanism metrics (same input bytes): shuffle bytes on disk -47..50%,
  executor CPU -56..63%. Candidate profile shows the zstd compression /
  decompression kernels on the previously idle GPU; single-query solo A/B
  on q9 gave 3.79x with non-overlapping distributions.

## 5. Regime Analysis: When the GPU Is Not Idle

This optimization spends GPU cycles (roughly 1s of zstd kernel time per
worker per heavy query on H200) to save CPU and disk. On clusters where the
GPU is the bottleneck the trade can invert. Mitigations:

1. The codec conf is a static opt-out.
2. The mixed `KUD0`/`KUDZ` framing makes per-batch adaptive policies
   wire-compatible: a future writer can skip compression under GPU
   contention (e.g. driven by semaphore wait times) with no format change.
3. Even GPU-bound, compression still relieves shuffle network and disk,
   which are often the binding constraint at scale — this is a policy
   point, not a hardcoded decision.

## 6. Files Changed Summary

| File | Change |
|------|--------|
| `RapidsConf.scala` | New internal codec conf |
| `GpuPartitioning.scala` | `sliceSerializeCompressOnGpu` write path + dispatch |
| `GpuColumnarBatchSerializer.scala` | `KUDZ` framing, `KudoCompressedMeta`, deserializer magic dispatch |
| `GpuShuffleCoalesceExec.scala` | GPU concat-with-decompress operator path |
| `GpuShuffledSizedHashJoinExec.scala` | Compressed spillable host concat result |
| `SpillableKudoTable.scala` | Raw host buffer accessor |
| `GpuShuffleExchangeExecBase.scala` | Serializer instance follows the compression decision |
| `DumpUtils.scala` | Loud rejection for compressed records in debug dump |
| `tests/.../GpuKudoCompressedShufflePartitioningSuite.scala` | Round-trip, fallback, and OOM-retry tests |

---
layout: page
title: Parallel Sub-Range Reads in the Parquet Coalescing Reader
nav_order: 17
parent: Developer Overview
---
# Parallel Sub-Range Reads in the Parquet Coalescing Reader

## 1. Background & Motivation

### The Key Insight: the Coalesced Buffer Layout Is Fully Precomputed

The multi-file coalescing parquet reader plans the entire destination host
buffer before reading a single byte: `copyBlocksData` walks the clipped
block metadata and assigns every column-chunk `CopyRange` an exact
`(inputOffset, length, outputOffset)` triple, and `coalesceReads` merges
adjacent ranges. Because every range's destination is known up front and
ranges are disjoint, **range reads commute** — they can execute in any
order, or concurrently, and produce a byte-identical buffer.

### The Problem: One Thread, One Stream, Per File

`RapidsInputFile.readVectored`'s default implementation (used by
`HadoopInputFile`) reads all ranges sequentially:

```
try (SeekableInputStream input = open()) {
  for (CopyRange copyRange : copyRanges) {
    input.seek(copyRange.getInputOffset());
    output.copyFromStream(copyRange.getOutputOffset(), input, copyRange.getLength());
  }
}
```

and the coalescing reader submits **one copy runner per file**, so a task
reading a single ~120MB file split performs all of its range reads on one
thread through one HDFS stream. The multithreaded-reader pool sits idle
during the buffer phase of each task.

On a measured NDS-H SF100 cluster this made GpuScan "buffer time" the
single largest executor-time component of the top queries (e.g. one
scan-bound query spent 206s of 255s of executor time in buffer time), with
scan stages 60-90% blocked and per-thread read throughput of roughly
125MB/s.

## 2. Design

`GpuParquetScan.readRangesToHostMemory` gains a parallel path:

1. Engagement: only for `HadoopInputFile` (filesystems with their own
   optimized `readVectored`, such as the S3 implementation, keep using it)
   and only when the coalesced ranges total at least 8MB. Small reads stay
   on the cheaper sequential path.
2. The coalesced ranges are split into sub-ranges of at most 16MB.
3. Each sub-range is submitted to a dedicated cached daemon pool
   ("parquet parallel range reader", 16 threads, core-timeout enabled).
   Every sub-read opens its own stream via `inputFile.open()`, seeks, and
   copies directly into its disjoint slice of the destination
   `HostMemoryBuffer`.
4. Error and interrupt handling: the submitting thread **drains every
   future** (successfully or not) before returning, so no sub-read can
   touch the destination buffer after the call completes — even on failure
   or task interruption. Interrupts are remembered and re-asserted after
   the drain; the first real error is rethrown.

The nested-pool deadlock hazard is avoided by construction: sub-reads are
leaf tasks that never submit further work, and they run on their own pool
rather than the multithreaded-reader pool whose threads invoke this path.

### Metrics Correctness

Spark's task input-bytes accounting is derived from Hadoop `FileSystem`
**thread** statistics, which only observe reads performed on the calling
thread. Bytes read by the helper threads are therefore recorded in a
monotonic per-thread counter (`MultiFileReaderFunctions.addOffThreadBytesRead`)
that `fileSystemBytesRead()` adds to the Hadoop thread statistics. Both
terms are monotonically increasing, so all existing delta computations
remain correct. Without this, `input_bytes` for parallel-read stages drops
to near zero (verified before/after: 0.0GB -> 10.2GB per lineitem scan
stage).

## 3. Correctness

- The destination buffer layout is deterministic and precomputed; sub-reads
  write disjoint slices; the result is byte-identical to the sequential
  path.
- Exact saved-output comparison (schema equality plus `exceptAll` in both
  directions) passed for the scan-bound NDS-H queries and later for the
  full 22-query suite.
- Drain-all-futures semantics make destination-buffer use-after-free
  impossible on error paths.

## 4. Performance Results (2-worker H200 cluster, NDS-H SF100)

- q7 (most scan-bound): median 7948ms -> 7420ms (-6.6%), every sample
  rank-below the reference run.
- Full stream: 620/625s reference -> 591/603s (-4.9% power test time),
  with gains concentrated in cold-read iterations where the deeper IO
  queue helps a throughput-limited disk; warm iterations improve less
  because the scan-stage floor moves to downstream work.
- Lineitem scan stage wall: 4.4s -> 3.6s per iteration.

## 5. Regime Notes

- No GPU cost; the change is purely host-side IO scheduling.
- On storage with no concurrency headroom the sub-reads degrade to roughly
  sequential behavior; the 8MB engagement floor and per-file runner reuse
  keep overhead negligible.
- Constants (16MB sub-range, 16 threads, 8MB floor) are conservative;
  they can be made configurable if tuning across storage types proves
  worthwhile.

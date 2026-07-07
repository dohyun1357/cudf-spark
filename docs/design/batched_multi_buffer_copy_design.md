---
layout: page
title: Batched Device Memcpy for multiBufferCopyAsync
nav_order: 18
parent: Developer Overview
---
# Batched Device Memcpy Kernel for `Cuda.multiBufferCopyAsync`

## 1. Background & Motivation

### The Measured Problem

`ai.rapids.cudf.Cuda.multiBufferCopyAsync` is the utility the shuffle
compression path uses to compact many small buffers into one contiguous
buffer. Its original Java implementation was literally a per-buffer loop of
JNI `asyncMemcpy` calls, self-described in the code as a "temporary sub-par
stand-in for a multi-buffer copy CUDA kernel".

The dominant caller is nvcomp batched compression:
`BatchedCompressor.stitchOutput` compacts each ~64KB compressed chunk of a
shuffle batch into the output blob. On a measured NDS-H SF100 cluster, one
TPC-H q21 iteration issued **268,405 memcpys per worker, of which 221,901
were device-to-device averaging 80KB** (17.9GB total). Each one is a JNI
crossing plus a `cudaMemcpyAsync` launch, executed while the task holds the
GPU semaphore — roughly 1-2 seconds of semaphore-held CPU per worker per
iteration, plus stream serialization from the launch cadence.

### The Key Insight

All of the copies in one `multiBufferCopyAsync` call are independent
(disjoint destinations, known sizes, same stream). That is exactly the
shape `cub::DeviceMemcpy::Batched` (exposed through
`cudf::detail::batched_memcpy_async`) is built for: ship one address/size
table to the device and let a single kernel perform every copy.

## 2. Design

- `Cuda.multiBufferCopyAsync` keeps its Java signature. The JNI
  implementation packs the source/destination/size arrays into a single
  host-side table, performs **one** H2D copy of the table, and launches
  **one** batched-memcpy kernel on the target stream.
- The kernel call lives in `java/src/main/native/src/multi_buffer_copy.cu`
  because cub headers require CUDA compilation.
- Tiny batches keep the plain per-buffer loop: below a small cutoff the
  kernel launch plus table upload costs more than a handful of direct
  `cudaMemcpyAsync` calls.
- Stream semantics are unchanged: all copies are enqueued on the caller's
  stream, so completion ordering and downstream stream-ordered frees are
  identical to the loop implementation.

## 3. Measured Results (NDS-H SF100, 2x H200, 4-core hosts)

- Mechanism: device-to-device memcpy count on q21 collapsed
  **221,901 -> 2,375 (-99%)**, replaced by 3,984 batched-copy kernels;
  H2D/D2H counts unchanged.
- End-to-end: adjacent interleaved full streams improved **-3.5% power
  time** (662s vs 686s), sum of per-query medians **-5.2%**; q9 solo
  median 14.1s vs 17.7s with every sample ranking below the control.
- Exact multiset correctness on q21 saved output vs the frozen baseline
  through the new copy path.

## 4. Edge Cases & Risks

- Empty batch and single-buffer calls short-circuit.
- Mixed memory kinds (host pinned sources) remain supported; the batched
  kernel path engages only for device-to-device copies.
- The address/size table upload is stream-ordered and freed
  stream-ordered; no synchronization is added.
- Regime note: the change is strictly less work (fewer launches, fewer JNI
  crossings, same bytes moved), so it is safe on GPU-bound clusters too.

## 5. Source

cudf: [`dohyun1357/cudf` branch `opt/batched-multi-buffer-copy`](https://github.com/dohyun1357/cudf/tree/opt/batched-multi-buffer-copy)
(commits `546dac24e0` + `88adad676f`); JNI gitlink:
[`dohyun1357/cudf-spark-jni` branch `opt/batched-multi-buffer-copy`](https://github.com/dohyun1357/cudf-spark-jni/tree/opt/batched-multi-buffer-copy)
(commit `3da47aa5`). Experiment record: OPTIMIZATION_LOG.md
EXP-20260707-004.

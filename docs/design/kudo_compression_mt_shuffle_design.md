---
layout: page
title: GPU Kudo Compression under the MULTITHREADED Shuffle Manager (upstream candidate)
nav_order: 21
parent: Developer Overview
---
# GPU-Compressed Kudo Payloads under the RapidsShuffleManager MULTITHREADED Writer

Status: **rejected on the measurement cluster, strong upstream candidate.**
Implementation preserved in history: cudf-spark `c2967a888` (reverted by
`84e2ff9a3`); validated jar kept as
`rapids-4-spark-dev-20260707-130515.jar`. Experiment record:
OPTIMIZATION_LOG.md EXP-20260707-010.

## 1. The Gap

Device-side zstd compression of kudo shuffle payloads (see
`gpu_compressed_kudo_shuffle_design.md`) is guarded to the built-in Spark
sort shuffle: `GpuPartitioning` disables it when
`GpuShuffleEnv.useMultiThreadedShuffle` is set. The MULTITHREADED
RapidsShuffleManager therefore ships **raw** kudo payloads wrapped only by
the CPU codec — and the GPU-serialized kudo layout is padded, so the CPU
codec compresses it poorly.

Measured on NDS-H SF100 (2x H200, 4-core hosts, one shared disk):
enabling the MULTITHREADED manager on the compression-capable line
regressed TPC-H q9 from ~13.4s to **[52.7s, 85.5s, 97.8s, 83.6s]** —
5-7x — while shuffle bytes went from 12.4GB to **30.5GB per iteration**
(94% blocked time; the disk and page cache absorb 2.5x the bytes). Any
deployment that pairs the MULTITHREADED manager with the kudo GPU
serializer pays this today.

## 2. The Change

One-line guard relaxation plus comments: engage kudo compression whenever
the payload moves verbatim — the built-in sort shuffle **and** the
MULTITHREADED manager. Records are opaque bytes to the threaded writer's
limiter/spill/merge pipeline, and the read side dispatches per record on
the KUD0/KUDZ magic independent of the manager (`CoalesceReadOption`
derives from the kudo configs, and the threaded reader deserializes
through the same `KudoSerializedBatchIterator`). The UCX/CACHE_ONLY
GPU-transport paths keep their own compression handling and stay excluded.

## 3. Validation (all on the composed path, mt manager active)

- Synthetic kudo shuffle suite (nulls, skew, empty partitions, unicode,
  decimal64/128, arrays, structs, array<struct>): exact GPU-vs-CPU PASS.
- Compression engages: q9 12.4GB/iter and stream-wide 325GB — identical
  to the sort-shuffle compression line; write-path executor CPU roughly
  halves (threaded pipeline plus no bypass-merge concatenation).
- q9 solo warm interleaved vs the sort-shuffle line: parity (13369 vs
  13249ms pooled medians), slightly tighter dispersion.
- Full-stream matched pairs in both window orders on the measurement
  cluster: 685/680s vs 628s power (+8-9%) — the threaded manager's read
  path costs more there than the single-pass write saves, because the
  62GB page cache absorbs the bypass-merge double write. Hence rejected
  *for that cluster's vanilla-manager baseline*, not on the merits of the
  composition.

## 4. Upstream Argument

For deployments where the MULTITHREADED RapidsShuffleManager is already
the operating choice (its threaded reader/writer and single-pass write
path exist for good reasons on other storage/network profiles), this
change is the difference between losing device-side compression entirely
(5-7x on shuffle-heavy queries above) and full compression performance.
The composition is mechanically sound, exactly correct on the edge-case
suite, and strictly reduces bytes and CPU on the write path.

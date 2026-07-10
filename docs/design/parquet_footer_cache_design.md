---
layout: page
title: Executor-Wide Parquet Footer and Scan-Metadata Cache
nav_order: 18
parent: Developer Overview
---
# Executor-Wide Parquet Footer and Scan-Metadata Cache

## 1. The Measured Problem

Every Spark task reads and parses the footer of every Parquet file (or file
split) it scans, on the task's critical path, before any data is buffered or
decoded. On common lakehouse layouts this work is massively repeated:

- **Many small files** (hive/iceberg/delta date-partitioned tables): NDS
  (TPC-DS) SF100 `store_sales` is 1,825 directories of ~7MB files. One q28
  run performs ~11,000 footer fetch+parse+filter passes over ~1,825 distinct
  files (once per file per stage, x6 subquery stages, x iterations), ~18ms
  each.
- **Few big files** (TPC-H style): NDS-H `lineitem` is read as ~200 x 128MB
  splits; every split's task re-fetches and re-parses the same large footer.

Measured on a 2-worker H200 cluster at SF100 (26.06.0 baseline, DEBUG
metrics): GpuScan `op time` is 65% of ALL NDS task time and 53% of NDS-H
task time, while `GPU decode time` is under 3% of it. For a single q28
iteration, `filter time` (the footer path) is 202.5s of 249.3s scan op time,
with per-task scan cost ~260ms independent of input size
(corr(scan time, input bytes) = -0.45). The JNI footer parse itself is
~120us; the cost is (a) filesystem round trips — file status, open + 8-byte
trailer read, open + footer body read, and dictionary-page reads when
predicates are pushed (`new ParquetFileReader(...).getRowGroups`) — and
(b) the per-call Java thrift parse, schema clip, and rebase-mode detection.
GPU semaphore wait was measured at 1.7s of that 249.3s — an initial
semaphore-convoy hypothesis was refuted by measurement.

## 2. Design

`ParquetFooterCache` is an executor-wide, two-level cache used by
`GpuParquetFileFilterHandler` (all reader types: COALESCING, MULTITHREADED,
PERFILE; both NATIVE and JAVA footer paths).

### Level 1: framed footer bytes

- Key: `(path, fileSize, modificationTime)` — all taken from Spark's
  `PartitionedFile`, so a hit costs zero filesystem calls; a change to the
  file invalidates the key naturally.
- Value: the framed `MAGIC + footer + footerLen + MAGIC` HostMemoryBuffer
  exactly as `ParquetFooterUtils` produces (composes with the disk
  FileCache: the loader still consults it on a miss).
- Byte-budget LRU: `spark.rapids.sql.format.parquet.footerCache.size`
  (internal, startup-only, default 128MiB, 0 disables). The cache owns one
  reference; every caller receives a slice (refcounted), so eviction can
  never free memory under a reader and outstanding slices survive eviction.

### Level 2: parsed-and-filtered scan metadata

- Key: `(path, fileSize, modificationTime, splitStart, splitLength,
  readSchema.json, pushed-filter fingerprint, session-settings
  fingerprint)`. The settings fingerprint covers every session conf that
  changes `filterBlocks` output (case sensitivity, pushdown flags and
  thresholds, rebase modes, field-id handling), because the cache is
  executor-global and outlives sessions.
- Value: the complete `ParquetFileInfoWithBlockMeta` produced by
  `filterBlocks` — footer parse, schema clip, rebase detection, and
  row-group filtering including dictionary-page I/O.
- Every lookup (including the loading call) returns a defensive copy:
  blocks are rebuilt via the existing `GpuParquetUtilsShims.newBlockMeta`
  (preserving row count, row-index offset, and column chunk refs, which are
  treated as immutable), and `partValues` is taken from the caller's own
  `PartitionedFile`. Cached parquet-mr objects never escape.
- Count-bounded LRU: `...footerCache.metadataEntries` (default 32768, 0
  disables).

Both sections load **single-flight**: concurrent tasks missing on one key
wait on one load instead of stampeding the filesystem (this alone removes
the q28-style 6-concurrent-stage duplicate fetch storm). Failed loads are
never cached; each waiter retries on its own terms so exception types
(e.g. `FileNotFoundException` under `ignoreMissingFiles`) and stacks stay
per-task.

## 3. Cold vs. warm behavior (honest accounting)

- A truly cold, single scan of a table gains only Level 1 (q28: -13%).
- Repeated touches — multiple subquery stages over one table, repeated
  or similar queries (dashboards), iterated runs — also hit Level 2, which
  removes the parse/clip/dictionary work: q28 31.3->18.5s (1.69x), q88
  21.3->12.1s (1.76x) at w1 i3.

## 4. Results (2-worker H200, SF100, w1 i3 legs, medians)

- **NDS (TPC-DS) full stream: sum of per-query medians 420.1s -> 298.2s =
  1.409x (-29.0%)** — baseline pooled from 4 clean legs across two days
  (drift < 0.5%), candidate from 2 clean legs; 45 of 103 queries >=1.25x
  (q44 3.57x, q84 2.34x, q90 2.13x, q49 2.09x, q85 1.98x, q80 1.92x, q88
  1.82x, q28 1.70x); **zero regressions >5%**.
- **NDS-H (TPC-H) full stream: 318.7s -> 283.7s = 1.123x (-11.0%)**
  (baseline pooled from 3 clean legs across two days, candidate from 2
  clean legs; same-era comparison against the freshest baseline leg is
  1.230x). Shuffle-heavy majors improve consistently (same-era: q15_p2
  1.87x, q14 1.66x, q8 1.62x, q5 1.54x, q9 1.35x); remaining per-query
  deltas are within this cluster's disk-noise envelope (verified at
  sample level, e.g. q16's shift appears identically in the same-era
  baseline leg, ratio 1.00). A separate shuffle-light subset A/B
  (q1,q6,q13,q14,q19,q22) measured **2.11x** (1.23x-3.38x, all improved).
- **Aggregate: 738.8s -> 581.9s = 1.270x (-21.2%)** — past the >=10%
  cross-benchmark goal under every baseline pooling choice (worst case
  1.255x).
- Correctness: full-stream exact saved-output comparison — NDS-H 22/22
  PASS; NDS 100/103 PASS with the three known baseline-nondeterministic
  queries (q39 AVG/STDDEV-over-double at 4.4e-16 relative, q79 LIMIT-100
  tie swap) reproduced between two baseline-only runs.
- Unit suite: 11 tests (identity, LRU, slice lifetime across eviction,
  failure retry, disabled modes, 8-thread single-flight, copy isolation).

## 5. Generality and limits

The mechanism keys off file immutability (path+size+mtime), the same
invariant Spark's own file index and the disk FileCache rely on. It is
filesystem-agnostic (HDFS, S3, ABFS all pay more per round trip, not
less) and layout-agnostic (helps both many-small-files and
big-file-many-splits shapes). Memory cost is bounded (128MiB bytes budget;
metadata entries are heap objects, typically KBs each). Workloads that
scan each file exactly once with a unique schema/predicate and never
repeat see only Level 1 benefits and a negligible lookup overhead.

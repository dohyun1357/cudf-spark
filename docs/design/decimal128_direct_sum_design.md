---
layout: page
title: Direct DECIMAL128 Sums (upstream candidate)
nav_order: 20
parent: Developer Overview
---
# Direct DECIMAL128 Sums: Wait-Free Two-Limb Atomic + De-Chunked Lowering

Status: **not shipped here; upstream candidate.** Rejected locally only
because the target cluster is storage-bound and a verified -14.3% on the
dominant aggregation kernel produced no end-to-end change there. The work
is regime-safe and strictly less work per row, so it should win on
GPU-/compute-bound clusters. Preserved in history: cudf `8eca175954`
(reverted `154bffe2f5`), plugin `6518a88f4` (reverted `d86bfa618`).
Experiment record: OPTIMIZATION_LOG.md EXP-20260707-008.

## 1. Two Stale Assumptions

1. **cudf**: `cudf::detail::atomic_add(__int128_t)` is a CAS retry loop —
   a native 128-bit CAS on sm_90 (including a plain, tearable 128-bit load
   of `expected`), two chained 64-bit CAS loops elsewhere. Under few-group
   aggregation contention every colliding thread spins.
2. **plugin**: `GpuDecimal128Sum`/`GpuDecimal128Average` decompose every
   decimal sum with a >18-digit result into four 32-bit chunk sums plus
   extract/assemble kernels, documented as needed because "DECIMAL128 sums
   are only implemented for sort-based aggregations" in cudf. Current cudf
   explicitly supports decimal128 SUM in hash groupby
   (`can_use_hash_groupby_fn`), routed through the atomic above. The
   chunked machinery's second purpose — overflow detection — is only
   needed where a 128-bit accumulator could wrap undetected.

A consequence worth knowing: **under Spark RAPIDS, `atomic_add(__int128_t)`
is completely dormant** — no plan ever issues a cudf sum on a decimal128
column. Kernel work there is invisible until the lowering changes.

## 2. Design

### cudf: two-limb wait-free `atomic_add(__int128_t)`

A 128-bit addition decomposes into a 64-bit `atomicAdd` of the low limbs
plus a 64-bit `atomicAdd` of the high limbs and the low-limb carry. Each
add detects its own carry exactly once from its returned old value;
addition and carries commute; the result is exact mod 2^128 with the same
wrap semantics as the CAS version. No retry loop, no 128-bit load, works
identically on global and shared memory and on all supported
architectures. The returned "old value" keeps the previous per-limb
(non-snapshot) guarantee; all aggregation call sites discard it.

### plugin: direct lowering for wrap-safe precisions

For input precision <= 18 digits (`Decimal.MAX_LONG_DIGITS`), a 128-bit
accumulator provably cannot wrap: an update batch is bounded by 2^31 rows
x 10^18 < 2.2e27 and a merge batch by 2^31 values already bounds-checked
to the result type (<= 10^28), both far below 1.7e38. Route those sums
(and the sum half of averages) to the direct
`GpuBasicDecimalSum`/`GpuBasicDecimalAverage` path: one DECIMAL128
aggregation target instead of four chunk sums, no chunk extract/assemble
kernel passes, half the per-sum shared-memory accumulator footprint. All
existing per-batch/merge/final bounds checks are retained unchanged.
Guarded by `spark.rapids.sql.decimal128.directSum.enabled`.

cudf support verified for every consuming context: hash groupby, sort
groupby, reduction, rolling windows, scan, and groupby-scan all accept
fixed-point SUM.

## 3. Evidence (NDS-H SF100, 2x H200)

- Correctness: 20M-row directed stress — 4-group maximum-contention sums,
  decimal(18,2) group sums crossing the 2^64 limb boundary (carry-heavy),
  decimal(27,0) values ~2^89 unscaled on the still-chunked tier, mixed
  signs, nulls, an exact-zero cancellation invariant, sums and averages,
  500k-group digests, reductions: all byte-exact vs CPU.
- Plan verified: TPC-H q1 lowers `sum(l_quantity)`/`sum(l_extendedprice)`
  to `gpubasicdecimalsum` while the (34,4)/(38,6) computed-column sums
  stay chunked. (q1's averages are `avg(UnscaledValue, DoubleType)` —
  double-based — so average rerouting is a no-op for TPC-H.)
- Mechanism: q1's `single_pass_shmem_aggs_kernel` improved
  **8.378 -> 7.183 ms/instance (-14.3%)** at identical batch geometry.
- End-to-end on the storage-bound test cluster: no change (two
  interleaved 10-sample q1 A/Bs) — the aggregation kernel is not that
  cluster's binding constraint. Expected to convert on compute-bound
  clusters.

## 4. Upstreaming Notes

- The cudf atomic change and the plugin lowering are independently
  mergeable; the lowering is what makes the atomic hot.
- The precision-38 chunked path is untouched (it still owns overflow
  detection where wrap is reachable — and is the subject of a separate
  known correctness defect; see diagnostics/d128-null-defect/).
- Extending the direct tier past 18-digit inputs requires the high-digits
  merge check (strategy 3 in `GpuDecimalSumOverflow`) or an equivalent
  guarantee; 18 is the provably-safe cutoff with no new machinery.

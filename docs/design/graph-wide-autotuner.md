---
layout: page
title: Graph-Wide Runtime Autotuning
nav_order: 15
parent: Developer Overview
---
# Graph-Wide Runtime Autotuning

This document describes the design of the graph-wide runtime autotuner: a driver-side
optimizer that observes measured task behavior across a SQL execution's stage graph and, where
the evidence identifies a better choice, adjusts runtime decisions that the static
configuration otherwise fixes. It currently owns one decision surface — the AQE reducer
partition layout — plus the sensing, calibration, and observability infrastructure that any
future decision surface needs. Everything is off by default and gated behind
`spark.rapids.sql.autotune.graph.enabled`.

## Motivation

Cluster-wide static configuration cannot be right for every stage of every query. Two measured
examples from NDS-H at SF3000 on a 160-slot cluster motivate the design:

- **Wave quantization.** With `spark.sql.shuffle.partitions=200`, the heaviest stages run 200
  balanced tasks on 160 slots: one full wave plus a 25%-occupied second wave that still lasts
  a full task duration. The stage pays two waves of latency for 1.25 waves of work. A layout
  of 160 or 320 partitions is 20-40% faster for the same bytes, but the best count differs per
  stage.
- **Per-task size cliffs.** One join-heavy stage processes bytes at 2,000 ms/GiB with
  1.4 GiB tasks but at 11,000-19,000 ms/GiB beyond 2.2 GiB per task (join sub-batching turns
  per-task cost superlinear). A +25% increase in per-task bytes more than doubled per-task
  time. Other stages in the same workload show flat or even mildly decreasing per-byte cost
  with size, so no single response curve can be assumed.

Both effects are invisible to a static configuration and to cost models that price bytes at a
constant rate with a fractional-wave (packing lower bound) makespan.

## Design principles

1. **One model, one objective.** Decisions minimize the predicted remaining completion time of
   the execution's stage graph; there are no independent per-knob policies or preferred sizes.
2. **Configuration defines feasibility, never preference.** Config values bound what the
   optimizer may do; they are not target values.
3. **Unknown response is frozen or delegated, never guessed.** A measurement at one operating
   point identifies work at that point, not the counterfactual at another. Every decision that
   would require an unmeasured response is frozen (current behavior retained) or delegated to
   Spark's native decision, and the freeze reason is reported in the event log.
4. **Spark owns semantic legality; the model owns measured cost.** The rule rewrites only
   choices Spark exposes as legal alternatives (e.g., coalesced shuffle-read layouts), and
   user-supplied mechanisms (custom AQE cost evaluators) are never overridden.
5. **Completed and materialized work is sunk.** Only remaining work is optimized.
6. **Fail open.** Every runtime path is guarded; an autotuner failure logs and leaves stock
   behavior unchanged (`spark.rapids.sql.autotune.failOpen`).
7. **Replay-complete evidence.** Every decision, including every freeze, is posted as a Spark
   listener event with the inputs that produced it, so acceptance checks can be done entirely
   from event logs.

## Architecture

```
 driver                                          executors
 ┌──────────────────────────────────────────┐    ┌──────────────────────────────┐
 │ RapidsAutotuneDriverEndpoint (singleton) │    │ RapidsAutotuneExecutor       │
 │  · per-stage hint publication/versioning │◄───┤  · hint fetch (once/stage)   │
 │  · per-task observation ingest           │    │  · per-task observation msg  │
 │  · executor task-slot census             │    │    (durations, bytes, GPU    │
 │  · TaskEnd deserialize clock             │    │     semaphore/holding clocks)│
 │  · initial-burst launch spacing          │    └──────────────────────────────┘
 │ AnalyticalGraphWideAutotuneOptimizer     │
 │  · stage graph + max-plus evaluation     │
 │  · execution-local calibration           │
 │ GpuFlowAqeParallelismRule (AQE rule)     │
 │  · reducer layout decisions              │
 └──────────────────────────────────────────┘
```

- **Wire types** live in `sql-plugin-api` (`GraphAutotuneMessages.scala`) so RPC classes load
  from the jar root rather than a shim parallel world; executor-side deserialization must not
  depend on the shim classloader.
- **`RapidsAutotuneStageHintListener`** (driver `SparkListener`) registers stages as they are
  submitted, publishes one versioned hint per stage attempt, feeds the finalized `TaskEnd`
  deserialize clock and the launch-spacing measurement, and releases all state when the SQL
  execution ends.
- **`AnalyticalGraphWideAutotuneOptimizer`** owns per-stage observation windows and the
  execution's calibration. The stage graph is evaluated with max-plus composition
  (`completion(stage) = local(stage) + max(completion(parents))`); a reverse pass assigns
  criticality (adjoints) to stages on the current critical path. At present all continuous
  executor controls are held at their measured operating point (see identifiability, below),
  so graph evaluation is a reporting/evidence layer: it is how acceptance checks verify the
  freezes, not an actuator.
- **`GpuFlowAqeParallelismRule`** is the one active decision surface, installed via
  `injectQueryStageOptimizerRule` so it runs at AQE's query-stage optimization boundary.

## Measured inputs

All inputs are measurements of the running application; none are configured preferences.

| input | source | used for |
|---|---|---|
| per-task duration, bytes, GPU clocks | executor observation messages, aggregated per stage window | service-rate calibration |
| cluster task slots | `SparkListenerExecutorAdded/Removed` census: `Σ floor(totalCores / spark.task.cpus)` | wave arithmetic |
| fixed per-task cost | finalized driver `TaskEnd` executor-deserialize time (minimum over successful tasks), or the duration/bytes regression intercept when its 95% lower confidence bound is positive | wave pricing |
| serial dispatch cost | initial-burst launch spacing: for each completed stage, the mean gap over the first `min(tasks, slots)` launch times (at least eight gaps); the endpoint serves the median over recent stages | re-split pricing |
| shuffle service rate | execution-local classified non-GPU task time per shuffle byte | layout pricing |
| per-task size envelope | largest mean per-task shuffle-read load observed in this execution | identified byte region |

Calibration is **execution-local**: rates and fixed-cost fits are scoped to one SQL execution
and released at execution end. A driver-global fit proved workload-order-dependent (a large
exchange could calibrate unrelated micro-exchanges and vice versa). The slot census and launch
spacing are application-level, like the scheduler properties they measure.

## Identifiability and freezes

The model refuses to price what it has not measured:

- **Byte region.** The calibrated linear rate is evidence only up to
  `max(gpuTargetBatchSizeBytes, largest measured per-task shuffle read)`. Candidate layouts
  with a range beyond that region are ineligible — this is the guard that fences superlinear
  operator cliffs, which on measured evidence can sit within +25% of the current point.
- **Fixed-cost boundary fits.** A non-negative least-squares intercept at zero is not a
  measured zero: using it as one made micro-exchanges look free to split. Zero-boundary and
  zero-crossing-confidence fits freeze the decision.
- **Dispatch response.** Until a launch-spacing measurement exists, candidates above Spark's
  current layout are frozen (`higher-parallelism-response-unidentified`).
- **Sub-batch region.** Re-split candidates must keep their mean range size at or above one
  native GPU batch. Fixed and launch costs are already charged for every range, so the mean
  identifies the candidate's typical per-task regime without rejecting a balanced layout only
  because its remainder range is smaller. A mean below one batch per task leaves the response
  unidentified (`sub-batch-range-response-unidentified`) and makes micro-exchange splits
  structurally impossible. The byte-region guard remains a per-range limit because an oversized
  remainder can cross an operator cost cliff on its own.
- **Continuous executor controls** (scan read-ahead window, shuffle prefetch/ready bytes,
  batch size) are all held at the deployed operating point and report
  `single-operating-point-response-unidentified`: a single-point observation cannot identify
  the response to a different setting, and historical single-point extrapolation produced
  reproducible regressions. Unlocking any of these requires multi-setting response evidence,
  which is future work.

Every freeze is a first-class result with a stable reason string in the decision events.

## The reducer layout rule

At AQE's query-stage optimization boundary the rule sees, for each plain coalesced shuffle
read (skew and local readers keep Spark ownership; join groups that must stay co-partitioned
are handled as one group with combined per-partition bytes), Spark's exact map-output
statistics. For each candidate partition count — task-slot wave endpoints, the original
(map-side) partition count, and the current count — it builds contiguous balanced ranges
(min-max binary search) and prices the layout with a wave-quantized makespan:

```
objective(layout) = Σ over scheduling waves of (largest range bytes in wave × rate)
                  + ceil(count / slots) × fixedTaskCost
                  + count × serialLaunchCost
```

Waves are formed by descending greedy assignment of `slots` ranges at a time. This replaces
the earlier `max(totalBytes/slots, largestRange) × rate` packing bound, which prices 200
balanced tasks on 160 slots as 1.25 waves when the schedule really takes two full waves — the
dominant mispricing for wave-misaligned heavy stages.

Authority is asymmetric, matching the measured risk:

- **Same-count rebalances and coalesces below Spark's layout** need only the byte-region guard:
  they have exact map statistics and measured wave costs.
- **Re-splits above Spark's layout** (never above the map-side partition count — contiguous
  ranges cannot subdivide map partitions) additionally require the measured dispatch response
  and a mean range size of at least one native GPU batch. Pricing smaller-than-measured tasks at
  the calibrated rate is conservative for this direction: on matched-stage evidence across four
  operating points, per-byte service does not improve as tasks shrink toward the batch-size floor
  once fixed and launch costs are charged separately, so the modeled wave-alignment benefit
  understates the measured one.
- **Larger-than-measured ranges** stay frozen behind the byte-region envelope regardless of
  how attractive the wave arithmetic looks; the measured superlinear cliffs live there.

Exact objective ties retain the current layout. A rewrite is applied only when its priced
objective is strictly lower than the identically priced current layout.

## AQE plan comparisons

The flow model does not currently own any AQE plan choice; measured delegation rules decide
who compares plans:

- Identical operator and topology fingerprints (a statistics refresh of one physical
  structure) always delegate to Spark's native contract. Ranking such pairs by modeled demand
  turned statistics-refinement noise into a systematic veto that pinned stale plans.
- Different operator bases (e.g., join strategy changes) delegate to Spark: a generic byte
  rate cannot identify sort-merge versus broadcast response, and driver-side broadcast service
  is not on any task clock we measure.
- User-supplied cost evaluators are never replaced.

## Observability

Two replay-complete event types cover the active surfaces:

- `SparkRapidsAutotuneGraphDecisionEvent`: per stage per decision epoch — graph objective,
  criticality adjoint, measured control point, per-control gradients, and freeze reasons.
- `SparkRapidsAutotuneParallelismEvent`: per reducer decision — original/current/selected
  partitions, bytes, task slots, calibrated rate, fixed-cost estimate with its standard error,
  sample support and source, serial-launch estimate with its stage support, both objectives,
  and the decision or freeze reason.

Per-task observation and hint-application events exist for deeper debugging. Acceptance
checking is designed to be done from event logs alone: baselines must emit zero autotuner
events; treatments must show the expected decisions and freezes.

## Validation methodology

- Performance claims use two independently built artifacts: a clean baseline from the exact
  upstream merge base and the candidate, with pinned revisions and externally verified
  SHA-256 hashes, identical non-autotuner configuration, and both arm orders (some queries
  have discrete fast/slow AQE modes; a single arm order is not attributable).
- Mechanism claims are checked from event logs (decision counts, freeze taxonomy, slot census
  values, calibration support), not from wall clock.
- Code removals were ablation-gated: mechanisms were cut only after a config-only A/B showed
  the remaining mechanisms retain the measured win (for example, GPU admission priorities
  regressed +6% without the IO actuators and were dropped entirely, while the deployed-point
  actuators alone held -4.4%).

## Failed experiments that shaped the boundaries

These are retained because they define the model's validity boundary; each has a guard in the
code where it bit:

- Overriding Spark's AQE join selection with a generic byte-rate model retained a shuffled
  join and slowed the query 2x. Spark owns unmeasured operator changes.
- Treating task input/output time as broadcast service made a 165 KiB broadcast look
  expensive. Driver broadcast work is not on the task clock; no broadcast lane exists.
- Single-point inverse scaling of scan/shuffle windows produced reproducible 2x stage
  regressions. All continuous controls are frozen at the measured point.
- A zero-boundary regression intercept treated as a measured zero expanded micro-exchanges.
  Boundary and uncertain fits freeze.
- A driver-global fixed-cost fit was workload-order dependent. Calibration is execution-local.
- The executor-side completion hook reports deserialize time as zero; the finalized driver
  `TaskEnd` event is the authoritative clock.
- Using the per-executor GPU admission quota as cluster width undercounted an 8-executor
  cluster 8x and squeezed real exchanges to 1/8 width. Task slots come from the executor
  census; a missing census freezes rather than falling back to any quota.
- Ranking same-structure AQE pairs by modeled demand vetoed 100% of Spark's statistics
  refreshes on one workload (demand deltas under 1.3 ppm) and pinned stale plans. Identical
  fingerprints always delegate.

## Future work

- Multi-setting response identification for the frozen continuous controls (controlled,
  version-stamped operating points with uncertainty), rather than unlocking on gradient sign.
- Response history keyed by stable operator/stage signatures: per-stage size response cannot
  be pooled within an execution (measured shapes range from superlinear cliffs to mildly
  decreasing), so upsizing beyond the measured envelope needs same-signature evidence across
  executions.
- A measured driver broadcast service clock, prerequisite to owning any join-strategy
  comparison.
- Cross-executor placement/skew awareness.

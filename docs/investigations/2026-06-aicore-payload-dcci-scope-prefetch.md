# AICore dispatch-payload: scope the per-dispatch invalidate + prefetch the head line

**Date**: 2026-06-25
**Verdict**: dropped — no end-to-end signal on any example (incl. the heaviest);
the per-dispatch payload dcci window is already ~0 µs warm, matching
[2026-06-aicore-cold-start-warmup.md](2026-06-aicore-cold-start-warmup.md)

## Question

The AICore hot loop in
`src/a2a3/runtime/tensormap_and_ringbuffer/aicore/aicore_executor.cpp`
invalidates the payload every dispatch with
`dcci(exec_payload, ENTIRE_DATA_CACHE)`. Two intuitions said this is wasteful:

1. **Scope (A2).** `ENTIRE_DATA_CACHE` clean-invalidates the *whole* L1, which
   also evicts (and, for dirty lines, writes back) the AICore working set the
   next kernel reuses. `build_payload()` writes `args[]` compactly
   (`args[0..tc+sc-1]`, tensor then scalar); the trailing slots are never
   written nor read. So AICore should be able to invalidate only the populated
   prefix + the always-read context tail.
2. **Prefetch (B).** dcci only invalidates; it does not prefetch. Issuing a
   `__builtin_prefetch` of the head line (`function_bin_addr`) right after the
   invalidate could overlap its GM reload with the ACK write + timestamp,
   instead of stalling `execute_task`'s first read.

## What was tried

Branch `opt/aicore-payload-dcci` off `8a32aa2b`, two commits:

- **A2** (`56e9f7d8`): added `PTO2DispatchPayload::valid_arg_count` (reused
  reserved pad, `sizeof` unchanged at 512), published `n = tc + sc` from
  `SchedulerContext::build_payload`, and replaced the whole-cache dcci with a
  cache-line-granular `invalidate_gm_range` covering only the context tail
  (`args[48..]` → end, always read) + the `function + valid_arg_count` args
  prefix. Small tasks drop from 8 invalidated lines to ~3.
- **B** (`413a0cbc`): `__builtin_prefetch(exec_payload, 0, 3)` after the
  invalidate. The CCE/dav backend accepts the builtin with a `__gm__`-qualified
  pointer (a non-`__gm__` `reinterpret_cast` is rejected at compile time).

Measurement: `/benchmark`-style compare, baseline (merge-base `8a32aa2b`) in a
git-worktree venv vs the branch, both on the **same locked device** via
`task-submit --device auto`, `tools/benchmark_rounds.sh -n 100`
(tensormap_and_ringbuffer, a2a3 onboard). Headline metric = `Total` trimmed avg
(drops 10 low + 10 high of 100 rounds). Correctness via
`pytest examples/a2a3/tensormap_and_ringbuffer/...` (golden compare).

## Result

`Total` µs, baseline → branch (Δ%):

| Example                              | A2 vs main | A2+B vs main |
| ------------------------------------ | ---------: | -----------: |
| alternating_matmul_add (Case1)       |     −1.11% |       −0.81% |
| benchmark_bgemm (Case0)              |     −0.91% |       −0.66% |
| paged_attention_unroll (Case1)       |     +0.30% |       +0.17% |
| paged_attention_unroll (Case2)       |     −0.76% |       +0.59% |
| pau_manual_scope (Case1)             |     −0.03% |       −0.13% |
| pau_manual_scope (Case2)             |     +0.14% |       +0.17% |
| batch_paged_attention (Case1)        |     +0.28% |       +1.79% |

Every example lands inside the ±2% run-to-run noise band, with mixed signs.
`Device` / `Sched` / `Orch` move the same way (within noise). The heaviest
workload (batch_paged_attention, ~3160 µs Total) — the most likely to carry a
large dirty L1 working set, which was the one angle the cold-start-warmup
investigation had not isolated — shows no benefit (A2 +0.28%, A2+B +1.79%, i.e.
slightly worse, dominated by noise).

Correctness: A2 passed golden on vector_example / scalar_data_test /
paged_attention (3/3). A2+B passed 4 of 5 (those three plus
paged_attention_unroll_manual_scope); the lone failure,
`paged_attention_manual_scope`, is a **pre-existing borderline-tolerance
flaky** — re-running it 5× on the unmodified merge-base failed 2/5 (golden
`max_diff ≈ 0.0037` vs `rtol/atol = 0.001` on the fp attention output). The
small magnitude (not garbage / not a hang) rules out a stale-payload read from
the narrowed invalidate. So A2/B introduce no correctness regression.

(`spmd_paged_attention` Case1/Case2 also fail in the benchmark harness on
**both** baseline and branch — pre-existing, unrelated.)

## Why not (now)

- **The cost being optimized is already ~0.** Per
  [2026-06-aicore-cold-start-warmup.md](2026-06-aicore-cold-start-warmup.md),
  `local_setup` (= the per-dispatch dcci + ack window, exactly what A2/B target)
  is 0–0.02 µs on warm cache; the only non-trivial head-OH is the cold first
  task, which is NoC/FFTS routing latency, not AICore-side cache work and not
  software-resolvable. Scoping or prefetching a ~0 µs window cannot move
  end-to-end time.
- **The "dirty-L1 writeback" hypothesis did not materialize.** The one angle
  not previously measured — that `ENTIRE_DATA_CACHE` is expensive because it
  writes back a large dirty working set — would show up most on
  batch_paged_attention. It didn't.
- **Cost without benefit.** A2 adds ABI surface (`valid_arg_count`), an AICPU
  write each dispatch, and a two-stage invalidate; B adds a hot-path
  instruction on every dispatch (including the non-profiled path). Trading that
  for a within-noise result is a poor deal.

## When to reconsider

- A workload appears whose AICore kernels leave a **large dirty L1 working set
  at dispatch boundaries** (so `ENTIRE_DATA_CACHE`'s writeback is real) AND
  whose `local_setup` is shown > ~1 µs in a level-3 L2-swimlane capture. Then
  A2 (scope) is the lever; re-measure `local_setup`, not just `Total`.
- AICore moves to a generation where the cold-start head-OH tail collapses (so
  per-dispatch cache work becomes the dominant head cost). Re-run the level-3
  swimlane and check whether `local_setup` becomes a meaningful fraction.

## References

- Branch `opt/aicore-payload-dcci`: commits `56e9f7d8` (A2), `413a0cbc` (B) —
  kept on the branch, not merged (no signal; retained only for the
  reconsider-conditions above).
- [2026-06-aicore-cold-start-warmup.md](2026-06-aicore-cold-start-warmup.md) —
  the prior measurement of `local_setup` / head-OH that predicted this.
- `.claude/rules/discipline.md` §4 (check investigations before optimizing).

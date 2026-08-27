---
name: benchmark
description: "Benchmark iterative A5 performance with a three-way Main/Before/After comparison, change-scope-aware case selection, a one-round regression screen, precise runs, chip swimlanes, and timestamped reports. Use when asked to benchmark, measure latency, compare performance, run an ablation, or produce A5 swimlanes; example-only changes retest only affected cases, while shared algorithm/runtime changes retest the full suite."
---

# A5 Performance Update Workflow

Use this workflow for every performance-related code update unless the user explicitly overrides it.

## Required comparison

Every update compares exactly three datasets:

1. **Main** — matching historical data for the recorded Main commit.
2. **Before** — matching historical data for the workspace immediately before the current experiment.
3. **After** — newly measured data for selected affected cases; for an example-only update, unaffected
   cases may reuse matching Before data and must be labeled as reused.

Never replace this with a two-way comparison. Show all three absolute values and these deltas:

- Before versus Main;
- After versus Main;
- After versus Before — the primary effect of the current update.

Positive latency change means slower; negative means faster.

## Historical-data rule

Do not rerun Main or Before. Read both from existing timestamped reports, `summary.csv` files, and
raw timing summaries under `outputs/` or another user-specified historical result directory.

Resolve provenance before editing code:

- record Main's exact commit SHA;
- record Before's `HEAD`, changed source files, and local diff identity;
- identify the historical source path and timestamp for both datasets;
- verify device, CANN, PTO-ISA pin, runtime, case, round count, and aggregation method.

Before normally equals the preceding update report's After dataset. Do not use a similarly named
result unless its code identity and measurement method match. If matching historical data is absent
or ambiguous, report the missing provenance and ask the user which history to use; do not silently
rerun Main/Before or fabricate values.

Historical data may use a different time window. Record that limitation. Use the same NPU device as
the historical datasets for After whenever possible; otherwise call out device-to-device variance.

## Select benchmark scope from the code diff

Classify the update before running hardware:

- **Example-only change** — changes are confined to one or more workload-specific example/test
  implementations and do not modify shared runtime, scheduler, allocator, graph construction,
  codegen, library, or common kernel logic. Retest only the directly affected case or cases.
- **Algorithm/shared change** — changes touch a shared algorithm, runtime, scheduler, orchestrator,
  allocator, ring/queue logic, common library, codegen, shared kernel, or another component that can
  affect multiple workloads. Retest the full default suite.
- **Mixed or ambiguous change** — if example-specific and shared code both change, or the impact
  boundary cannot be established from the diff, classify it as algorithm/shared and retest the full
  suite.

For an example-only update, keep the full three-way document coherent by reusing Before as After
for unaffected cases and marking those rows `reused`, not measured. Copy their matching historical
swimlane artifacts into the new artifact directory when a complete eight-case artifact set is useful;
capture new swimlanes only for affected cases. Never present a reused `0.00%` row as a new measurement.

Record the classification, changed files, selected cases, and why those cases cover the diff in the
timestamped report.

## Defaults

- Platform: `a5`.
- Runtime: `tensormap_and_ringbuffer`.
- Device: the same card used by the selected historical datasets; card 1 when their established
  baseline is card 1.
- Coarse screen: one performance round per selected case, without swimlane instrumentation.
- Precise non-Qwen: 100 rounds per selected case, arithmetic average.
- Precise Qwen3: five rounds. Sort by Device latency, drop the fastest and slowest rounds, then
  average the other three rounds and their corresponding Host/Effective/Orch/Sched values.
- Chip swimlane: after the coarse gate passes, one separate level-4 round per selected After case.
- Performance runs and swimlane runs are separate; swimlane instrumentation must not affect the
  performance table.

Default A5 TMR cases:

| Case | Test file | Rounds |
| --- | --- | ---: |
| alternating_matmul_add Case1 | `tests/st/a5/tensormap_and_ringbuffer/alternating_matmul_add/test_alternating_matmul_add.py` | 100 |
| benchmark_bgemm Case0 | `examples/a5/tensormap_and_ringbuffer/benchmark_bgemm/test_benchmark_bgemm.py` | 100 |
| paged_attention_unroll Case1 | `tests/st/a5/tensormap_and_ringbuffer/paged_attention_unroll/test_paged_attention_unroll.py` | 100 |
| paged_attention_unroll Case2 | same as above | 100 |
| paged_attention_unroll_manual_scope Case1 | `examples/a5/tensormap_and_ringbuffer/paged_attention_unroll_manual_scope/test_paged_attention_unroll_manual_scope.py` | 100 |
| paged_attention_unroll_manual_scope Case2 | same as above | 100 |
| batch_paged_attention Case1 | `tests/st/a5/tensormap_and_ringbuffer/batch_paged_attention/test_batch_paged_attention.py` | 100 |
| qwen3_14b_decode StressBatch16Seq3500 | `examples/a5/tensormap_and_ringbuffer/qwen3_14b_decode/test_qwen3_14b_decode.py` | 5 |

The default case list defines the full suite for algorithm/shared changes. Example-only changes use
only the affected rows from this list. If several example directories change, select their union.
Do not run a selected list with one uniform `-n 100`: Qwen precise measurement remains five rounds.

## Workflow

### 1. Define the update

Before changing code:

- state the single mechanism or hypothesis being tested;
- capture `git rev-parse HEAD`, `git status --short`, `git diff --stat`, and `pto_isa.pin`;
- classify the diff as example-only or algorithm/shared and record the selected cases;
- locate and validate the Main and Before historical datasets;
- choose a timestamp and short slug for the update.

Use `YYYYMMDD_HHMMSS` in Asia/Shanghai time. Keep this timestamp stable for all artifacts from the
same update.

### 2. Implement and validate locally

Make the requested code change without disturbing unrelated workspace edits. Run formatting,
`git diff --check`, and tests proportional to the changed path. The coarse one-round pass for each
selected case should be golden-enabled when output correctness is meaningful, so it acts as both a
correctness check and the first performance screen.

Use a workspace-local venv:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --no-build-isolation -e .
```

Verify `build/lib/pto_isa_build.json` points into the current workspace and matches `pto_isa.pin`.

### 3. Run a one-round coarse screen

Use one `task-submit` allocation for all selected cases. Never run two benchmark processes on the
same device concurrently. Run each selected After case exactly once without swimlane
instrumentation. Do not run Main or Before.

Compare the coarse Device value with the matching historical Before value. Prefer a matching raw
Before round when available; otherwise use the recorded Before aggregate and state that the screen
compares a single After sample with an aggregate baseline.

```bash
task-submit --timeout 7200 --max-time 7200 --device <historical-device> \
  --run "bash '<absolute-coarse-payload-script>'"
```

For each selected case, invoke its test file with:

```bash
--platform a5 --device "$DEVICE_ID" --case <case> --manual include \
--rounds 1
```

Save the raw output and render its timing table. Apply this gate to the primary Device metric:

- if any selected case is at least `+5.00%` slower than Before, stop before precise measurement and
  swimlane capture;
- if every selected case is below `+5.00%` regression, proceed to precise measurement;
- if correctness fails or timing markers are missing, stop that case and treat the gate as failed.

The coarse screen is a regression guard, not the final performance result. Do not combine its value
with precise rounds or use it in the final precise average.

### 4. Run precise After performance after the gate passes

Use one `task-submit` allocation for the complete selected-case precise sequence. For each selected
case, invoke its test file with:

```bash
--platform a5 --device "$DEVICE_ID" --case <case> --manual include \
--rounds <100-or-5> --skip-golden
```

Use 100 rounds for every selected non-Qwen case and five rounds for selected Qwen. Save the raw
combined output, then render its timing table:

```bash
python -m simpler_setup.tools.strace_timing <raw.log> --rounds-table
```

Do not rebuild or execute Main/Before during this step.

### 5. Capture selected After swimlanes once

After the coarse gate passes, run one separate level-4 capture for each selected After case. Do not
capture Main or Before again. For example-only changes, copy matching unaffected historical
artifacts if the update directory should expose all eight cases; do not recapture them.

```bash
--platform a5 --device "$DEVICE_ID" --case <case> --manual include \
--rounds 1 --skip-golden --enable-chip-swimlane 4
```

Preserve, per case:

- `chip_swimlane_records.json`;
- `merged_swimlane.json` for direct Perfetto loading;
- the kernel name map when generated;
- the capture log.

Validate every newly captured raw file reports `chip_swimlane_level == 4`, contains task records,
and converts to a non-empty Perfetto trace. Validate copied artifacts as well before linking them.

### 6. Calculate the three-way result

For every metric available to the runtime, report:

| Case | Main | Before | After | Before/Main | After/Main | After/Before |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |

TMR metrics are Host, Device, Effective, Orch, and Sched. Use Device as the primary end-to-end
latency headline and retain the other columns for mechanism analysis. HBG has only Host and Device;
do not invent missing fields.

Calculate percentage change as `(new / reference - 1) * 100`. For multi-case summaries, calculate
the geometric mean of per-case ratios, not an arithmetic mean of percentages.

For Qwen, list all five Device samples, identify the dropped fastest/slowest rounds, and state the
three retained indices or values. Apply those same retained rounds to every other Qwen metric.

For an example-only change, include the selected cases as newly measured After data. If the report
also shows the full suite, set unaffected After values equal to the matching Before history, label
their provenance as `reused`, and exclude them from claims about measured change. For an
algorithm/shared change, all eight After rows must come from the new precise run.

Interpretation guide:

| After/Before | Assessment |
| ---: | --- |
| below -2% | notable improvement |
| -2% to +2% | likely within noise unless consistently supported by related metrics/traces |
| above +2% | potential regression requiring explicit discussion |

Highlight every regression at or above 5%. Do not hide mixed results behind the geometric mean.

### 7. Write exactly one timestamped update report

Create one new Markdown document for each update:

```text
outputs/perf-updates/<YYYYMMDD_HHMMSS>_<slug>.md
```

Never overwrite or append the previous update's report. Do not create an additional README for the
same update unless the user requests one. CSV files, raw logs, and traces are artifacts rather than
additional update documents and may live under:

```text
outputs/perf-updates/<YYYYMMDD_HHMMSS>_<slug>/
```

The report must contain:

1. **Update purpose** — what changed, the hypothesis, and the exact code scope.
2. **Code identities** — Main SHA, Before identity, After identity/local diff, and PTO-ISA pin.
3. **Historical provenance** — source path and timestamp for Main and Before; explicitly state they
   were reused rather than rerun.
4. **Method** — device, CANN, runtime, cases, round counts, aggregation, ordering, and whether runs
   were interleaved; include change classification, selected cases, coarse results, and gate decision.
5. **Three-way results** — absolute Main/Before/After data and all three delta views.
6. **Effect analysis** — which cases improved/regressed, likely mechanism, noise limitations, and a
   clear retain/revert/further-test conclusion.
7. **Validation** — local tests and golden results.
8. **Swimlanes** — one clickable `merged_swimlane.json` link per selected After case plus the
   raw-artifact directory; identify copied historical artifacts separately.

If the coarse gate fails, still create the one timestamped report. Mark it as a coarse-only stopped
experiment, show the one-round values and regression, link the raw logs, and state that precise runs
and swimlanes were intentionally not performed.

State clearly when the After workspace is uncommitted. Use relative paths inside repository docs;
do not embed usernames or user-specific absolute paths.

## Failure handling

| Failure | Action |
| --- | --- |
| Main or Before history is missing/ambiguous | Stop the comparison and request the correct historical source; do not rerun it automatically |
| `task-submit` cannot allocate the device | Report the allocation/queue error; do not run unlocked |
| Coarse Device regression is at least 5% | Stop before precise runs and swimlanes; preserve logs and write a coarse-only timestamped report |
| Coarse correctness/timing fails | Treat the gate as failed; preserve logs and do not start precise measurement for that case |
| One After benchmark fails | Preserve its log, continue safe independent cases, and mark the three-way table incomplete |
| Timing markers are absent | Preserve the log and report that the case has no usable performance data |
| Swimlane conversion fails | Preserve the raw JSON and capture log; report the affected case without claiming 8/8 coverage |
| Build metadata points outside the workspace | Rebuild in the local venv before running hardware |

## Completion checklist

- [ ] Main historical dataset identified and not rerun.
- [ ] Before historical dataset identified before editing and not rerun.
- [ ] After workspace identity and update purpose recorded.
- [ ] Diff classified as example-only or algorithm/shared; selected cases and rationale recorded.
- [ ] One coarse round completed for every selected case and the 5% Device gate evaluated.
- [ ] If the gate passed, selected non-Qwen cases use 100 precise rounds and selected Qwen uses five.
- [ ] Qwen middle-three aggregation uses Device ordering and matching rows for all metrics.
- [ ] One level-4 chip swimlane captured and validated for every selected After case after gate pass.
- [ ] Unaffected reused rows/artifacts are clearly labeled and were not rerun.
- [ ] Main/Before/After absolute values and all three delta views are present.
- [ ] One new timestamped Markdown report describes the update and its effect.
- [ ] Raw logs, timing summaries, CSV, and swimlane artifacts are linked from that report.

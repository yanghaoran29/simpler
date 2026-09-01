---
name: benchmark
description: "Benchmark iterative A5 performance with a three-way Main/Before/After comparison, change-scope-aware case selection, a coarse screen with chip swimlanes, precise runs, and timestamped reports. Use when asked to benchmark, measure latency, compare performance, run an ablation, or produce A5 swimlanes; example-only changes retest only affected cases, while shared algorithm/runtime changes retest the full suite."
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
- Coarse screen: one golden-enabled performance round, one separate level-4 chip-swimlane round,
  and one separate dependency-graph round per selected case. Run them serially, never combine
  `--enable-chip-swimlane` and `--enable-dep-gen` in one process, and never overlap those two
  processes. Gate on the swimlane's full AICore task window versus the matching historical Main
  swimlane; the dependency run is artifact-only and does not contribute timing.
- Precise default: 100 rounds per selected case, arithmetic average.
- Precise Qwen3 / DeepSeek-V4 Pro attention: five rounds. For Qwen3, sort by Device latency, drop
  the fastest and slowest rounds, then average the other three rounds and their corresponding
  Host/Effective/Orch/Sched values. DeepSeek-V4 Pro attention uses the arithmetic mean of its five
  rounds.
- Performance runs and swimlane runs are separate; swimlane instrumentation must not affect the
  performance table. Do not recapture swimlanes after the gate passes; the coarse-stage capture is
  the update's one required After swimlane.

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
| deepseek_v4_pro_attention DecodeSWA | `examples/a5/tensormap_and_ringbuffer/deepseek_v4_pro_attention/test_deepseek_v4_pro_attention.py` | 5 |
| deepseek_v4_pro_attention DecodeCSA | same as above | 5 |
| deepseek_v4_pro_attention DecodeHCA | same as above | 5 |
| deepseek_v4_pro_attention PrefillSWA | same as above | 5 |
| deepseek_v4_pro_attention PrefillCSA | same as above | 5 |
| deepseek_v4_pro_attention PrefillHCA | same as above | 5 |

The default case list defines the full suite for algorithm/shared changes. Example-only changes use
only the affected rows from this list. If several example directories change, select their union.
Do not run a selected list with one uniform `-n 100`: Qwen3 and DeepSeek-V4 Pro attention precise
measurement remain five rounds.

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

### 3. Run the coarse screen and capture swimlanes

Use one `task-submit` allocation for all selected cases. Never run two benchmark processes on the
same device concurrently. Do not run Main or Before. On known local A5 hosts where
`onboard-arch-precheck` cannot read `npu-smi` board info, set
`SIMPLER_SKIP_ARCH_PRECHECK=1` and proceed with `--platform a5` (see the personal
`a5-onboard-host` skill).

For every selected After case, run these as three separate processes/runs in this order:

1. one golden-enabled performance round without swimlane instrumentation, for correctness and a
   supporting one-round performance signal;
2. one `--skip-golden --enable-chip-swimlane 4` round, saved as the update's After swimlane;
3. after the swimlane process has fully exited, one `--skip-golden --enable-dep-gen` round using the
   same case, inputs, runtime, build, and round count, saved as that swimlane's `deps.json`.

The swimlane and dependency graph must be captured serially. Do not place
`--enable-chip-swimlane 4` and `--enable-dep-gen` on the same command line, do not launch their
processes in the background, and do not start dep-gen while the swimlane process is still running.
Dep-gen perturbs device execution and is not valid swimlane timing. After both captures finish,
place `deps.json` beside `chip_swimlane_records.json` in the final per-case artifact directory, then
run the swimlane converter so `merged_swimlane.json` includes dependency identities and arrows.

Compare the After swimlane's full AICore task window with the matching historical **Main
swimlane**. Define this window as:

```text
max(aicore_tasks.end_time) - min(aicore_tasks.start_time)
```

It spans the first AIC/AIV task start through the final AIC/AIV task end. Do not use the capture
log's Device wall time as the gate metric; report it only as supporting data. Main swimlane history
must use the same case, accelerator model, runtime and instrumentation level. Prefer the same card;
if only another card of the same model is available, use it only with an explicit cross-card
variance caveat. Do not substitute Main's precise aggregate or Before's swimlane for this gate. If
matching Main swimlane history is absent or ambiguous, stop and request the correct history rather
than rerunning Main automatically.

```bash
task-submit --timeout 7200 --max-time 7200 --device <historical-device> \
  --run "bash '<absolute-coarse-payload-script>'"
```

For the golden-enabled coarse performance round, invoke:

```bash
--platform a5 --device "$DEVICE_ID" --case <case> --manual include \
--rounds 1
```

For the separate coarse-stage swimlane round, invoke:

```bash
--platform a5 --device "$DEVICE_ID" --case <case> --manual include \
--rounds 1 --skip-golden --enable-chip-swimlane 4
```

Only after that command exits, invoke the separate dependency-graph round:

```bash
--platform a5 --device "$DEVICE_ID" --case <case> --manual include \
--rounds 1 --skip-golden --enable-dep-gen
```

Save all three raw outputs, render the performance timing table, and preserve the swimlane and
dependency artifacts listed in step 5. Apply this gate to the full AICore task-window metric:

- if any selected After swimlane's full AICore task window is at least `+5.00%` longer than its
  matching historical Main swimlane, stop before precise measurement;
- if every selected case is below `+5.00%` regression, proceed to precise measurement;
- if correctness fails, dependency capture fails, swimlane conversion fails, or either timing
  source is missing, stop that case and treat the gate as failed.

The uninstrumented coarse time and the instrumented swimlane time are both diagnostic only. Do not
combine either with precise rounds or use them in the final precise average. Record the supporting
coarse After/Before comparison, but do not use it as the 5% stop gate unless the user explicitly
requests that additional gate.

### 4. Run precise After performance after the gate passes

Use one `task-submit` allocation for the complete selected-case precise sequence. For each selected
case, invoke its test file with:

```bash
--platform a5 --device "$DEVICE_ID" --case <case> --manual include \
--rounds <100-or-5> --skip-golden
```

Use 100 rounds for every selected default case and five rounds for selected Qwen3 or
DeepSeek-V4 Pro attention cases. Save the raw
combined output, then render its timing table:

```bash
python -m simpler_setup.tools.strace_timing <raw.log> --rounds-table
```

Do not rebuild or execute Main/Before during this step.

### 5. Validate the coarse-stage After swimlanes

Use the level-4 captures already produced during step 3. Do not capture selected cases a second time
after the gate. Do not capture Main or Before again. For example-only changes, copy matching
unaffected historical artifacts if the update directory should expose all eight cases; do not
recapture them.

Preserve, per case:

- `chip_swimlane_records.json`;
- `deps.json`, captured by a separate serial `--enable-dep-gen` run with the same topology;
- `merged_swimlane.json` for direct Perfetto loading;
- the kernel name map when generated;
- the swimlane capture log and the dep-gen capture log.

Validate every newly captured raw file reports `chip_swimlane_level == 4`, contains task records,
and converts with its separately captured `deps.json` to a non-empty Perfetto trace. Validate that
`deps.json` is non-empty, parseable, and contains the expected task topology. Validate copied
artifacts as well before linking them.

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

Highlight every Device, Effective, Orchestrator, or Scheduler regression at or above 5%. Do not
hide mixed results behind the geometric mean. Use Effective only for the deprecated-name acceptance
rule below so one workload is counted once rather than once per correlated metric.

## Deprecated experiment naming

Use like-for-like **After versus Before Effective latency** as the precise-run acceptance metric.
Do not count Main deltas or the correlated Device, Orchestrator, and Scheduler columns again when
choosing the filename. Continue to show and discuss all of those metrics in the report.

Classify each selected case by its precise round count:

- **Five-round cases** — currently Qwen3 StressBatch16Seq3500 and all six
  `deepseek_v4_pro_attention` cases (DecodeSWA/CSA/HCA, PrefillSWA/CSA/HCA).
- **Other cases** — the remaining default suite rows (100-round cases).

Prefix the experiment with `deprecated_` when **any** of the following hold over the selected
cases:

1. **Five-round single-case hard limit** — any selected five-round case has Effective regression
   of at least `+5.00%` versus matching Before.
2. **Five-round geometric-mean optimization** — over all selected five-round cases, compute
   `ratio_i = After Effective / Before Effective` and the geometric-mean optimization
   `(1 - exp(mean(log(ratio_i)))) * 100%`. This value **must be strictly positive** (overall
   faster). Zero or negative → deprecated. Skip this clause (and clause 1) when no five-round
   case is selected.
3. **Other-case allowance** — among selected non-five-round cases, at most **two** may show
   Effective regression of at least `+5.00%`. Three or more → deprecated.

Do **not** use the older Qwen-only `+2.00%` rule or the older non-Qwen geometric-mean tier table
that permitted 0/1/2 regressions from aggregate gain brackets. For an example-only experiment,
apply the three clauses only over newly measured selected cases; exclude reused rows.

Never use Host time to decide whether an experiment is deprecated. Host remains a diagnostic field
in the comparison table. Device, Orchestrator, Scheduler, After/Main, and swimlane deltas remain
mandatory diagnostics and may still fail their own coarse gate, but they do not independently add
to the precise-run deprecated count unless the user explicitly designates another acceptance
metric for that update.

Do not trigger this rule by comparing an uninstrumented one-round coarse result with a historical
100/5-round aggregate, by comparing different cards without an accepted cross-card baseline, or by
using a diagnostic metric that the report has established as non-comparable. Main baseline and Main
rerun documents are datasets rather than experiments and are not deprecated because two Main
captures differ by 5%.

Use the exact sibling names:

```text
outputs/perf-updates/deprecated_<YYYYMMDD_HHMMSS>_<slug>.md
outputs/perf-updates/deprecated_<YYYYMMDD_HHMMSS>_<slug>/
```

Historical reports under `outputs/perf-updates/202608/` keep their existing names; do not bulk
rename them when the acceptance rule changes. Rename both the Markdown report and its artifact
directory for **new** experiments after the precise acceptance calculation is complete. Update
every relative link in that report and every cross-report reference to either old name. Preserve
all raw results; `deprecated_` marks the conclusion and must not delete data.

### 7. Write exactly one timestamped update report

Create one new Markdown document for each update; use the deprecated form immediately when the
experiment has met the rule above:

```text
outputs/perf-updates/<YYYYMMDD_HHMMSS>_<slug>.md
outputs/perf-updates/deprecated_<YYYYMMDD_HHMMSS>_<slug>.md
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
8. **Swimlanes** — one clickable `merged_swimlane.json` link and one `deps.json` link per selected
   After case plus the raw-artifact directory; state that swimlane and dep-gen were captured in
   separate serial runs, and identify copied historical artifacts separately.

If the coarse gate fails, still create the one timestamped report. Mark it as a coarse-and-swimlane
stopped experiment, show the uninstrumented coarse values and the Main/After swimlane gate, link all
raw logs and traces, and state that precise runs were intentionally not performed.

State clearly when the After workspace is uncommitted. Use relative paths inside repository docs;
do not embed usernames or user-specific absolute paths.

## Failure handling

| Failure | Action |
| --- | --- |
| Main or Before history is missing/ambiguous | Stop the comparison and request the correct historical source; do not rerun it automatically |
| `task-submit` cannot allocate the device | Report the allocation/queue error; do not run unlocked |
| After full AICore swimlane window versus matching Main swimlane is at least 5% longer | Stop before precise runs; preserve coarse logs and swimlanes and write a timestamped stopped-experiment report; identify the coarse-gate failure, but do not apply the precise-run aggregate naming rule without precise data |
| Coarse correctness/timing fails | Treat the gate as failed; preserve logs and do not start precise measurement for that case |
| One After benchmark fails | Preserve its log, continue safe independent cases, and mark the three-way table incomplete |
| Timing markers are absent | Preserve the log and report that the case has no usable performance data |
| Dependency-graph capture fails or `deps.json` is invalid | Preserve the independent swimlane and both capture logs; treat the case's coarse artifact gate as failed and do not start its precise measurement |
| Swimlane conversion fails | Preserve the raw JSON and capture log; report the affected case without claiming 8/8 coverage |
| Build metadata points outside the workspace | Rebuild in the local venv before running hardware |

## Completion checklist

- [ ] Main historical dataset identified and not rerun.
- [ ] Before historical dataset identified before editing and not rerun.
- [ ] After workspace identity and update purpose recorded.
- [ ] Diff classified as example-only or algorithm/shared; selected cases and rationale recorded.
- [ ] One golden-enabled coarse round, one level-4 swimlane round, and one separate dep-gen round completed for every selected case.
- [ ] Swimlane and dep-gen ran serially as separate processes; neither command enabled both features.
- [ ] Matching historical Main swimlane provenance identified and the 5% full-AICore-window gate evaluated.
- [ ] If the gate passed, selected default cases use 100 precise rounds; selected Qwen3 and
      DeepSeek-V4 Pro attention use five.
- [ ] Qwen middle-three aggregation uses Device ordering and matching rows for all metrics.
- [ ] One coarse-stage level-4 chip swimlane captured and validated for every selected After case.
- [ ] One matching, separately captured `deps.json` is beside every selected After swimlane and was used during conversion.
- [ ] Unaffected reused rows/artifacts are clearly labeled and were not rerun.
- [ ] Main/Before/After absolute values and all three delta views are present.
- [ ] Qwen Effective is below +2%, and the non-Qwen Effective geometric-mean optimization and permitted/actual >=5% regression counts are recorded.
- [ ] If Qwen fails or the non-Qwen actual count exceeds its permitted count, `deprecated_` is on both the report and artifact directory, with links updated.
- [ ] One new timestamped Markdown report describes the update and its effect.
- [ ] Raw logs, timing summaries, CSV, swimlane artifacts, and `deps.json` files are linked from that report.

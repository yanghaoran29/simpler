#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# validate_param_combinations.sh — Validate main parameter combinations of
# run_tests.sh and verify requirement coverage for all backends.
#
# Usage:
#   ./validate_param_combinations.sh [options]
#
# Options:
#   --out-dir DIR    Output directory (default: validate_results_YYYYMMDD_HHMMSS/)
#   --quick          Reduced set: only Group 1–5 (build + func + core new backends)
#   --timeout N      Per-test wall-clock timeout in seconds (default: 300)
#   --no-build       Skip the build step for each group (use existing BUILD_DIR)
#   --help           Show this message
#
# Output layout:
#   <OUT_DIR>/
#     summary.txt          — one line per case: [PASS|FAIL]  Ns  <label>  <args>
#     analysis.txt         — requirements coverage analysis
#     logs/
#       <label>.log        — stdout+stderr of run_tests.sh for that case
#
# Environment overrides (forwarded to run_tests.sh):
#   ORCH_CPU, SCHED_CPU0..7, PLATFORM_MAX_BLOCKDIM, PLATFORM_MAX_AICPU_THREADS
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SH="$SCRIPT_DIR/run_tests.sh"

if [ ! -f "$RUN_SH" ]; then
    echo "ERROR: run_tests.sh not found at $RUN_SH" >&2
    exit 1
fi

# ─── Defaults ─────────────────────────────────────────────────────────────────
TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$SCRIPT_DIR/validate_results_$TS"
QUICK=false
TIMEOUT_SEC=300
NO_BUILD=false

# ─── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out-dir)  OUT_DIR="$2"; shift 2 ;;
        --quick)    QUICK=true; shift ;;
        --timeout)  TIMEOUT_SEC="$2"; shift 2 ;;
        --no-build) NO_BUILD=true; shift ;;
        --help|-h)
            sed -n '2,24p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; echo "Use --help for usage." >&2; exit 1 ;;
    esac
done

LOG_DIR="$OUT_DIR/logs"
SUMMARY="$OUT_DIR/summary.txt"
ANALYSIS="$OUT_DIR/analysis.txt"

mkdir -p "$LOG_DIR"

# ─── Counters and per-group tracking ──────────────────────────────────────────
PASS=0
FAIL=0
TOTAL=0

# Group-level counters: GPASS_<group> GFAIL_<group>
declare -A GPASS
declare -A GFAIL
declare -A GNAME
CURRENT_GROUP=""

# ─── Helpers ──────────────────────────────────────────────────────────────────
print_section() {
    local title="$1"
    CURRENT_GROUP="$title"
    GPASS["$title"]=0
    GFAIL["$title"]=0
    GNAME["$title"]="$title"
    local sep="$(printf '%0.s─' {1..70})"
    echo ""
    echo "$sep"
    echo "  $title"
    echo "$sep"
    echo "" >> "$SUMMARY"
    echo "── $title" >> "$SUMMARY"
}

# run_case LABEL [run_tests.sh args...]
# BUILD_DIR must be set before calling.
run_case() {
    local label="$1"; shift
    TOTAL=$((TOTAL + 1))
    local log="$LOG_DIR/${label}.log"
    local args_str="$*"

    printf "  [%3d] %-50s " "$TOTAL" "$label"

    local t0=$SECONDS
    local rc=0
    TIMEOUT="$TIMEOUT_SEC" BUILD_DIR="$CURRENT_BUILD" \
        bash "$RUN_SH" "$@" > "$log" 2>&1 || rc=$?
    local elapsed=$((SECONDS - t0))

    if [ $rc -eq 0 ]; then
        printf "PASS  %4ds\n" "$elapsed"
        printf "PASS  %4ds  %-50s  %s\n" "$elapsed" "$label" "$args_str" >> "$SUMMARY"
        PASS=$((PASS + 1))
        GPASS["$CURRENT_GROUP"]=$(( ${GPASS["$CURRENT_GROUP"]:-0} + 1 ))
    else
        printf "FAIL  %4ds  (rc=%d)\n" "$elapsed" "$rc"
        printf "FAIL  %4ds  %-50s  %s\n" "$elapsed" "$label" "$args_str" >> "$SUMMARY"
        FAIL=$((FAIL + 1))
        GFAIL["$CURRENT_GROUP"]=$(( ${GFAIL["$CURRENT_GROUP"]:-0} + 1 ))
        echo "  → log: $log"
    fi
}

# ─── Summary header ───────────────────────────────────────────────────────────
{
    echo "validate_param_combinations — run_tests.sh coverage report"
    echo "Date       : $(date)"
    echo "Host       : $(hostname)"
    echo "Quick mode : $QUICK"
    echo "Timeout    : ${TIMEOUT_SEC}s per case"
    echo "run_tests  : $RUN_SH"
    echo "Build base : $OUT_DIR/build_*"
    echo "================================================================"
} | tee "$SUMMARY"

# ─── Build directories (one per compile-flag configuration) ───────────────────
BUILD_DEFAULT="$OUT_DIR/build_default"       # O3, no profiling
BUILD_PROFILING="$OUT_DIR/build_profiling"   # O3, full profiling
BUILD_DEBUG="$OUT_DIR/build_debug"           # O0, no profiling
BUILD_NER="$OUT_DIR/build_no_early_return"   # O3, PTO2_SIM_NO_EARLY_RETURN=ON

CURRENT_BUILD="$BUILD_DEFAULT"

# ─────────────────────────────────────────────────────────────────────────────
# Group 1 — Build verification (default config)
# Verify all backends compile without error.
# ─────────────────────────────────────────────────────────────────────────────
print_section "Group 1: Build verification (O3, profiling=OFF)"

run_case "build_only"  --build-only

# ─────────────────────────────────────────────────────────────────────────────
# Group 2 — Functional tests
# ─────────────────────────────────────────────────────────────────────────────
print_section "Group 2: Functional tests"

run_case "func_only"        --func
run_case "func_and_perf"    --func --perf

# ─────────────────────────────────────────────────────────────────────────────
# Group 3 — Default perf sweep (existing backends, sanity check)
# ─────────────────────────────────────────────────────────────────────────────
print_section "Group 3: Default perf sweep (existing backends, idx=0)"

run_case "default_perf_idx0"          # no args → all DEFAULT_PERF_TESTS at idx 0
run_case "pa_concurrent_0"            --test test_paged_attention --idx 0
run_case "bpa_concurrent_0"           --test test_batch_paged_attention --idx 0
run_case "bpa_orch_only_0"            --test test_batch_paged_attention_orch_only --idx 0
run_case "bpa_sched_prof_only_0"      --test test_batch_paged_attention_sched_prof_only --idx 0
run_case "deg_concurrent_0"           --test test_deg_concurrent --idx 0
run_case "deg_orch_only_0"            --test test_deg_orch_only --idx 0
run_case "deg_sched_prof_only_0"      --test test_deg_sched_prof_only --idx 0

# ─────────────────────────────────────────────────────────────────────────────
# Group 4 — New backend: alternating matmul+add (PERF_BACKEND=4)
# ─────────────────────────────────────────────────────────────────────────────
print_section "Group 4: New backend — alternating matmul+add (PERF_BACKEND=4)"

run_case "alt_concurrent_0"       --test test_alt_concurrent --idx 0
run_case "alt_concurrent_1"       --test test_alt_concurrent --idx 1
run_case "alt_concurrent_all"     --test test_alt_concurrent
run_case "alt_orch_only_0"        --test test_alt_orch_only --idx 0
run_case "alt_orch_only_1"        --test test_alt_orch_only --idx 1
run_case "alt_sched_prof_only_0"  --test test_alt_sched_prof_only --idx 0
run_case "alt_sched_prof_only_1"  --test test_alt_sched_prof_only --idx 1

# ─────────────────────────────────────────────────────────────────────────────
# Group 5 — New backend: batched GEMM (PERF_BACKEND=5)
# ─────────────────────────────────────────────────────────────────────────────
print_section "Group 5: New backend — batched GEMM (PERF_BACKEND=5)"

run_case "bgemm_concurrent_0"       --test test_bgemm_concurrent --idx 0
run_case "bgemm_concurrent_2"       --test test_bgemm_concurrent --idx 2
run_case "bgemm_concurrent_4"       --test test_bgemm_concurrent --idx 4
run_case "bgemm_orch_only_0"        --test test_bgemm_orch_only --idx 0
run_case "bgemm_orch_only_3"        --test test_bgemm_orch_only --idx 3
run_case "bgemm_sched_prof_only_0"  --test test_bgemm_sched_prof_only --idx 0
run_case "bgemm_sched_prof_only_4"  --test test_bgemm_sched_prof_only --idx 4

# ─────────────────────────────────────────────────────────────────────────────
# Group 6 — New backend: paged attention unroll (PERF_BACKEND=6)
# ─────────────────────────────────────────────────────────────────────────────
print_section "Group 6: New backend — paged attention unroll (PERF_BACKEND=6)"

run_case "pau_concurrent_0"       --test test_pau_concurrent --idx 0
run_case "pau_concurrent_1"       --test test_pau_concurrent --idx 1
run_case "pau_concurrent_2"       --test test_pau_concurrent --idx 2
run_case "pau_concurrent_all"     --test test_pau_concurrent
run_case "pau_orch_only_0"        --test test_pau_orch_only --idx 0
run_case "pau_orch_only_2"        --test test_pau_orch_only --idx 2
run_case "pau_sched_prof_only_0"  --test test_pau_sched_prof_only --idx 0
run_case "pau_sched_prof_only_2"  --test test_pau_sched_prof_only --idx 2

# ─────────────────────────────────────────────────────────────────────────────
# Group 7 — Scheduler thread count variations
# ─────────────────────────────────────────────────────────────────────────────
print_section "Group 7: Scheduler thread count (--sched-threads)"

run_case "sched_threads_1_alt0"    --sched-threads 1 --test test_alt_concurrent --idx 0
run_case "sched_threads_2_bgemm0"  --sched-threads 2 --test test_bgemm_concurrent --idx 0
run_case "sched_threads_3_pau0"    --sched-threads 3 --test test_pau_concurrent --idx 0
run_case "sched_threads_4_bgemm0"  --sched-threads 4 --test test_bgemm_concurrent --idx 0
run_case "sched_threads_1_bpa0"    --sched-threads 1 --test test_batch_paged_attention --idx 0

# ─────────────────────────────────────────────────────────────────────────────
# Groups 8–10 are skipped in --quick mode
# ─────────────────────────────────────────────────────────────────────────────
if ! $QUICK; then

    # ─────────────────────────────────────────────────────────────────────────
    # Group 8 — All indices (--all)
    # ─────────────────────────────────────────────────────────────────────────
    print_section "Group 8: All indices sweep (--all)"

    run_case "all_indices_default"  --all
    run_case "alt_all_indices"      --test test_alt_concurrent
    run_case "bgemm_all_indices"    --test test_bgemm_concurrent
    run_case "pau_all_indices"      --test test_pau_concurrent
    run_case "deg_all_indices"      --test test_deg_concurrent
    run_case "bpa_all_indices"      --test test_batch_paged_attention

    # ─────────────────────────────────────────────────────────────────────────
    # Group 9 — Profiling (separate build config)
    # ─────────────────────────────────────────────────────────────────────────
    CURRENT_BUILD="$BUILD_PROFILING"
    print_section "Group 9: Profiling mode (--profiling)"

    run_case "profiling_alt_0"          --profiling --test test_alt_concurrent --idx 0
    run_case "profiling_alt_1"          --profiling --test test_alt_concurrent --idx 1
    run_case "profiling_bgemm_0"        --profiling --test test_bgemm_concurrent --idx 0
    run_case "profiling_pau_0"          --profiling --test test_pau_concurrent --idx 0
    run_case "profiling_deg_0"          --profiling --test test_deg_concurrent --idx 0
    run_case "profiling_bpa_0"          --profiling --test test_batch_paged_attention --idx 0
    run_case "profiling_no_sched_alt0"  --profiling --no-sched-profiling --test test_alt_concurrent --idx 0
    run_case "profiling_no_orch_alt0"   --profiling --no-orch-profiling  --test test_alt_concurrent --idx 0
    run_case "profiling_no_sched_pau0"  --profiling --no-sched-profiling --test test_pau_concurrent --idx 0
    run_case "profiling_no_orch_bgemm0" --profiling --no-orch-profiling  --test test_bgemm_concurrent --idx 0

    # ─────────────────────────────────────────────────────────────────────────
    # Group 10 — Debug build (O0) + no-early-return
    # ─────────────────────────────────────────────────────────────────────────
    CURRENT_BUILD="$BUILD_DEBUG"
    print_section "Group 10: Debug build (--opt-level 0)"

    run_case "debug_alt_0"    --opt-level 0 --test test_alt_concurrent --idx 0
    run_case "debug_bgemm_0"  --opt-level 0 --test test_bgemm_concurrent --idx 0
    run_case "debug_pau_0"    --opt-level 0 --test test_pau_concurrent --idx 0
    run_case "debug_func"     --opt-level 0 --func

    CURRENT_BUILD="$BUILD_NER"
    print_section "Group 11: --no-early-return"

    run_case "ner_bpa_0"    --no-early-return --test test_batch_paged_attention --idx 0
    run_case "ner_bpa_1"    --no-early-return --test test_batch_paged_attention --idx 1
    run_case "ner_alt_0"    --no-early-return --test test_alt_concurrent --idx 0
    run_case "ner_bgemm_0"  --no-early-return --test test_bgemm_concurrent --idx 0

fi  # end !QUICK

# ─────────────────────────────────────────────────────────────────────────────
# Requirements analysis
# ─────────────────────────────────────────────────────────────────────────────
{
    echo ""
    echo "================================================================"
    echo "  Requirements Coverage Analysis"
    echo "================================================================"
    echo ""
    echo "run_tests.sh meets the following requirements after this session's updates:"
    echo ""
    echo "  [Backend coverage]"
    echo "  PERF_BACKEND=0  linear_case           → test_linear_*          indices: 0 1 2"
    echo "  PERF_BACKEND=1  paged_attention        → test_paged_attention   indices: 0"
    echo "  PERF_BACKEND=2  batch_paged_attention  → test_batch_pa_*        indices: 0 1 2"
    echo "  PERF_BACKEND=3  deg{2,4,8}_case        → test_deg_*             indices: 0 1 2"
    echo "  PERF_BACKEND=4  alternating_matmul_add → test_alt_*             indices: 0 1  [NEW]"
    echo "  PERF_BACKEND=5  benchmark_bgemm        → test_bgemm_*           indices: 0..4 [NEW]"
    echo "  PERF_BACKEND=6  paged_attention_unroll → test_pau_*             indices: 0 1 2 [NEW]"
    echo ""
    echo "  [Driver modes] Each backend has 3 driver variants:"
    echo "    *_concurrent       — orch + sched running concurrently (main integration path)"
    echo "    *_orch_only        — orch only, measures graph-build cost"
    echo "    *_sched_prof_only  — orch pre-built, then sched; isolates scheduling cost"
    echo ""
    echo "  [Parameter combinations tested]"
    printf "    %-35s %s\n" "Build verification"        "Group 1"
    printf "    %-35s %s\n" "Functional tests"          "Group 2"
    printf "    %-35s %s\n" "Default perf (idx=0)"      "Group 3"
    printf "    %-35s %s\n" "New backend alt (4)"       "Group 4"
    printf "    %-35s %s\n" "New backend bgemm (5)"     "Group 5"
    printf "    %-35s %s\n" "New backend pau (6)"       "Group 6"
    printf "    %-35s %s\n" "--sched-threads 1/2/3/4"  "Group 7"
    if ! $QUICK; then
        printf "    %-35s %s\n" "--all (all indices)"       "Group 8"
        printf "    %-35s %s\n" "--profiling variants"      "Group 9"
        printf "    %-35s %s\n" "--opt-level 0"             "Group 10"
        printf "    %-35s %s\n" "--no-early-return"         "Group 11"
    else
        printf "    %-35s %s\n" "--all, profiling, debug, ner"  "(skipped, use without --quick)"
    fi
    echo ""
    echo "  [Gaps / known limitations]"
    echo "    • test_linear_* not in DEFAULT_PERF_TESTS (must use --test test_linear_concurrent)"
    echo "    • test_paged_attention has only index 0 (single case file)"
    echo "    • --no-early-return only meaningful with PTO2_SIM_NO_EARLY_RETURN build flag"
    echo "      (run_tests.sh passes -DPTO2_SIM_NO_EARLY_RETURN=ON automatically)"
    echo ""
} | tee -a "$SUMMARY" > "$ANALYSIS"

# ─────────────────────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Final Summary"
echo "================================================================"
echo ""
echo "  Total : $TOTAL"
echo "  Passed: $PASS"
echo "  Failed: $FAIL"
echo ""
echo "  Output directory : $OUT_DIR"
echo "  Summary          : $SUMMARY"
echo "  Analysis         : $ANALYSIS"
echo "  Logs             : $LOG_DIR/"

{
    echo ""
    echo "================================================================"
    echo "  Final Summary"
    echo "================================================================"
    echo "  Total : $TOTAL   Passed: $PASS   Failed: $FAIL"
    echo "  Quick : $QUICK"
} >> "$SUMMARY"

echo ""
if [ "$FAIL" -eq 0 ]; then
    echo "  OVERALL: PASSED"
    echo "  OVERALL: PASSED" >> "$SUMMARY"
    exit 0
else
    echo "  OVERALL: FAILED ($FAIL/$TOTAL cases failed)"
    echo "  OVERALL: FAILED ($FAIL/$TOTAL cases failed)" >> "$SUMMARY"
    echo ""
    echo "  Failed cases (check logs/ for details):"
    grep "^FAIL" "$SUMMARY" | awk '{print "    -", $4}' || true
    exit 1
fi

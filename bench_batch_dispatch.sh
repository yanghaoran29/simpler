#!/usr/bin/env bash
# bench_batch_dispatch.sh
#
# Run batch-dispatch profiling benchmark for batch_paged_attention,
# comparing BATCH_DISPATCH=0 vs BATCH_DISPATCH=1 (BATCH_ENQUEUE=0 fixed).
#
# Profiling data is collected from run_example.py stdout/stderr output.
# Results are saved to a timestamped Markdown report.
#
# Usage:
#   ./bench_batch_dispatch.sh [-d <device_id>] [-r <runs>] [-o <output_file>]
#
# Defaults: device=10, runs=10

set -euo pipefail

# ─── Argument parsing ────────────────────────────────────────────────────────
PLATFORM="a2a3"
RUNS=10
DEVICE_ID="10"
OUTPUT_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--device)  DEVICE_ID="$2";   shift 2 ;;
        -r|--runs)    RUNS="$2";        shift 2 ;;
        -o|--output)  OUTPUT_FILE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIMPLER_DIR="${SCRIPT_DIR}/simpler"
cd "$SIMPLER_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [[ -z "$OUTPUT_FILE" ]]; then
    OUTPUT_FILE="${SCRIPT_DIR}/bench_batch_dispatch_report_${TIMESTAMP}.md"
fi

DATA_DIR="${SCRIPT_DIR}/bench_data_${TIMESTAMP}"
mkdir -p "$DATA_DIR"
OUT_DIR="$DATA_DIR"

# ─── Test case ───────────────────────────────────────────────────────────────
KERNELS_DIR="tests/device_tests/tensormap_and_ringbuffer/batch_paged_attention/kernels"
GOLDEN="tests/device_tests/tensormap_and_ringbuffer/batch_paged_attention/golden.py"
TEST_NAME="batch_paged_attention"

# ─── 2 configurations: BATCH_ENQUEUE=0 fixed, BATCH_DISPATCH 0|1 ─────────────
CFG_KEYS=("D0E0" "D1E0")
CFG_LABELS=(
    "DISPATCH=0 ENQUEUE=0"
    "DISPATCH=1 ENQUEUE=0"
)
CFG_FLAGS=(
    "-DPTO2_PROFILING=1 -DPTO2_SCHED_PROFILING=1 -DPTO2_BATCH_DISPATCH=0 -DPTO2_BATCH_ENQUEUE=0"
    "-DPTO2_PROFILING=1 -DPTO2_SCHED_PROFILING=1 -DPTO2_BATCH_DISPATCH=1 -DPTO2_BATCH_ENQUEUE=0"
)
NUM_CFGS=${#CFG_KEYS[@]}

echo "============================================================"
echo "  Batch-Dispatch Benchmark  (a2a3 hardware)"
echo "  Device   : $DEVICE_ID"
echo "  Runs     : $RUNS per configuration"
echo "  Test     : $TEST_NAME"
echo "  Configs  : ${NUM_CFGS} (BATCH_DISPATCH=0 vs 1, ENQUEUE=0 fixed)"
echo "  Report   : $OUTPUT_FILE"
echo "  Data dir : $DATA_DIR"
echo "============================================================"
echo ""

# ─── Run one configuration, collect profiling lines from output ──────────────
# Args: kernels_dir  golden  cfg_idx  out_file
run_variant() {
    local kernels_dir="$1"
    local golden="$2"
    local cfg_idx="$3"
    local out_file="$4"
    local cxxflags="${CFG_FLAGS[$cfg_idx]}"
    local run_cmd="python examples/scripts/run_example.py -k ${kernels_dir} -g ${golden} -p ${PLATFORM} -d ${DEVICE_ID} --enable-profiling"

    > "${out_file}.raw"

    local i rc
    for i in $(seq 1 "$RUNS"); do
        echo "    run ${i}/${RUNS} ..."
        local run_log="${OUT_DIR}/run_${DEVICE_ID}_${CFG_KEYS[$cfg_idx]}_${i}.log"
        CXXFLAGS="$cxxflags" $run_cmd > "$run_log" 2>&1 && rc=0 || rc=$?

        if [[ $rc -ne 0 ]]; then
            echo "    ERROR: run_example.py failed (exit $rc)" >&2
            echo "    --- last 20 lines ---" >&2
            tail -20 "$run_log" >&2
            echo "    ---------------------" >&2
            continue
        fi

        # Extract profiling lines directly from run output
        grep '\[ALWAYS\]' "$run_log" 2>/dev/null \
            | grep -E 'Scheduler (Phase Breakdown|summary)|complete *:|dispatch *:|poll *:|otc_|pop *:|setup *:|scan *:|idle *:|avg/complete' \
            | sed 's/.*\[ALWAYS\][^"]*"\(.*\)"/\1/' \
            >> "${out_file}.raw" || true
    done

    sort "${out_file}.raw" > "${out_file}" 2>/dev/null || cp "${out_file}.raw" "${out_file}"
}

# ─── Metric extraction helpers ───────────────────────────────────────────────
# All functions return distribution stats as "min/p50/p90/max" across all
# scheduler-thread lines (threads × runs) collected in the raw data file.

extract_us_stats() {
    local file="$1" pattern="$2"
    grep "$pattern" "$file" 2>/dev/null \
        | grep -oE '[0-9]+\.[0-9]+us' | grep -oE '[0-9]+\.[0-9]+' \
        | sort -n \
        | awk 'BEGIN{n=0} {v[n++]=$1}
               END {
                   if (n==0) { print "N/A"; exit }
                   p50=v[int(n*0.50)]; p90=v[int(n*0.90 < n ? n*0.90 : n-1)]
                   printf "%.2f/%.2f/%.2f/%.2f", v[0], p50, p90, v[n-1]
               }' \
        || echo "N/A"
}

extract_atomics_stats() {
    local file="$1" pattern="$2" key="${3:-atomics}"
    grep "$pattern" "$file" 2>/dev/null \
        | grep -oE "${key}=[0-9]+" | grep -oE '[0-9]+' \
        | sort -n \
        | awk 'BEGIN{n=0} {v[n++]=$1}
               END {
                   if (n==0) { print "N/A"; exit }
                   p50=v[int(n*0.50)]; p90=v[int(n*0.90 < n ? n*0.90 : n-1)]
                   printf "%d/%d/%d/%d", v[0], p50, p90, v[n-1]
               }' \
        || echo "N/A"
}

extract_sum() {
    local file="$1" pattern="$2" key="$3"
    grep "$pattern" "$file" 2>/dev/null \
        | grep -oE "${key}=[0-9]+" | grep -oE '[0-9]+' \
        | awk '{s+=$1} END {if(NR>0) print s; else print "N/A"}' \
        || echo "N/A"
}

sum_us_stats_p50() {
    local file="$1"; shift
    local total=0 has=0 v p50
    for pat in "$@"; do
        v=$(extract_us_stats "$file" "$pat")
        if [[ "$v" != "N/A" ]]; then
            p50=$(echo "$v" | cut -d/ -f2)
            total=$(awk -v t="$total" -v p="$p50" 'BEGIN{printf "%.2f", t+p}')
            has=1
        fi
    done
    [[ $has -eq 1 ]] && echo "$total" || echo "N/A"
}

stats_p50() { echo "$1" | cut -d/ -f2; }

delta_vs_baseline() {
    local base="$1" val="$2"
    if [[ "$base" == "N/A" || "$val" == "N/A" ]]; then echo "N/A"; return; fi
    local b v
    b=$(stats_p50 "$base" | grep -oE '[0-9]+\.?[0-9]*' | head -1)
    v=$(stats_p50 "$val"  | grep -oE '[0-9]+\.?[0-9]*' | head -1)
    awk "BEGIN {
        b=${b:-0}; v=${v:-0}
        if (b > 0) printf \"%+.1f%%\", (v - b) * 100.0 / b
        else print \"N/A\"
    }"
}

# ─── Report helpers ───────────────────────────────────────────────────────────
PASS_COUNT=0
FAIL_COUNT=0

report_metric_table_2() {
    local f0="$1" f1="$2"

    printf '| %-44s | %20s | %25s |\n' \
        "Metric (min/p50/p90/max)" \
        "D=0 E=0 (base)" "D=1 E=0 (Δp50)" \
        >> "$OUTPUT_FILE"
    printf '| %-44s | %-20s | %-25s |\n' \
        "$(printf -- '-%.0s' {1..44})" \
        "$(printf -- '-%.0s' {1..20})" \
        "$(printf -- '-%.0s' {1..25})" \
        >> "$OUTPUT_FILE"

    mdrow2() {
        local label="$1" v0="$2" v1="$3"
        printf '| %-44s | %20s | %25s |\n' \
            "$label" \
            "$v0" \
            "${v1}  $(delta_vs_baseline "$v0" "$v1")" \
            >> "$OUTPUT_FILE"
    }

    {
        mdrow2 "dispatch phase (us)" \
            "$(extract_us_stats "$f0" "dispatch *:")" \
            "$(extract_us_stats "$f1" "dispatch *:")"
        mdrow2 "  dispatch.pop (us)" \
            "$(extract_us_stats "$f0" "pop *:")" \
            "$(extract_us_stats "$f1" "pop *:")"
        mdrow2 "  dispatch.poll (us)" \
            "$(extract_us_stats "$f0" "poll *:")" \
            "$(extract_us_stats "$f1" "poll *:")"
        mdrow2 "  dispatch.setup (us)" \
            "$(extract_us_stats "$f0" "setup *:")" \
            "$(extract_us_stats "$f1" "setup *:")"
        mdrow2 "pop() atomics" \
            "$(extract_atomics_stats "$f0" "pop *:" "atomics")" \
            "$(extract_atomics_stats "$f1" "pop *:" "atomics")"
        mdrow2 "pop() wait_cycle" \
            "$(extract_atomics_stats "$f0" "pop *:" "wait")" \
            "$(extract_atomics_stats "$f1" "pop *:" "wait")"
        mdrow2 "complete phase (us)" \
            "$(extract_us_stats "$f0" "complete *:")" \
            "$(extract_us_stats "$f1" "complete *:")"
        mdrow2 "  otc_lock atomics" \
            "$(extract_atomics_stats "$f0" "otc_lock" "atomics")" \
            "$(extract_atomics_stats "$f1" "otc_lock" "atomics")"
        mdrow2 "  otc_fanout atomics" \
            "$(extract_atomics_stats "$f0" "otc_fanout" "atomics")" \
            "$(extract_atomics_stats "$f1" "otc_fanout" "atomics")"
        mdrow2 "  otc_fanin atomics" \
            "$(extract_atomics_stats "$f0" "otc_fanin" "atomics")" \
            "$(extract_atomics_stats "$f1" "otc_fanin" "atomics")"
        mdrow2 "scan phase (us)" \
            "$(extract_us_stats "$f0" "scan *:")" \
            "$(extract_us_stats "$f1" "scan *:")"
        mdrow2 "idle phase (us)" \
            "$(extract_us_stats "$f0" "idle *:")" \
            "$(extract_us_stats "$f1" "idle *:")"
        mdrow2 "avg/complete (us/task)" \
            "$(extract_us_stats "$f0" "avg/complete")" \
            "$(extract_us_stats "$f1" "avg/complete")"
        mdrow2 "total tasks_scheduled" \
            "$(extract_sum "$f0" "Scheduler summary" "tasks_scheduled")" \
            "$(extract_sum "$f1" "Scheduler summary" "tasks_scheduled")"
        mdrow2 "total scheduler loops" \
            "$(extract_sum "$f0" "Scheduler summary" "loops")" \
            "$(extract_sum "$f1" "Scheduler summary" "loops")"
        local t0 t1
        t0=$(sum_us_stats_p50 "$f0" "dispatch *:" "complete *:" "scan *:" "idle *:")
        t1=$(sum_us_stats_p50 "$f1" "dispatch *:" "complete *:" "scan *:" "idle *:")
        mdrow2 "total sched time (us p50/thread)" "$t0" "$t1"
    }
}

# ─── Initialise report document ───────────────────────────────────────────────
cat > "$OUTPUT_FILE" << EOF
# Batch-Dispatch Benchmark Report

| Field | Value |
|---|---|
| Platform | ${PLATFORM} |
| Device | ${DEVICE_ID} |
| Runs per configuration | ${RUNS} |
| Generated | $(date) |
| Test | ${TEST_NAME} |

## Configurations

| Key | BATCH_DISPATCH | BATCH_ENQUEUE | Description |
|---|---|---|---|
| D=0 E=0 | 0 | 0 | baseline: one-by-one pop |
| D=1 E=0 | 1 | 0 | batch pop (under test) |

Distribution columns show **min/p50/p90/max** across all scheduler threads × runs.
Δp50 shows % change in median vs D=0 E=0 baseline (negative = faster).
Raw data saved in: ${DATA_DIR}

---
EOF

# ─── Run benchmark ────────────────────────────────────────────────────────────
echo "------------------------------------------------------------"
echo "  $TEST_NAME"
echo "------------------------------------------------------------"
printf '\n## %s\n\n' "$TEST_NAME" >> "$OUTPUT_FILE"

declare -a OUT_FILES=()
for ci in $(seq 0 $((NUM_CFGS - 1))); do
    out_f="${OUT_DIR}/cfg${ci}_${TEST_NAME}.txt"
    OUT_FILES+=("$out_f")
    echo "  [${CFG_LABELS[$ci]}] ..."
    run_variant "$KERNELS_DIR" "$GOLDEN" "$ci" "$out_f"
done

any_data=0
for f in "${OUT_FILES[@]}"; do
    lines=$(wc -l < "$f" 2>/dev/null || echo 0)
    [[ "$lines" -gt 0 ]] && any_data=1
done

if [[ $any_data -eq 0 ]]; then
    printf '**Status**: FAIL — no profiling lines in output (check --enable-profiling and [ALWAYS] log output)\n\n' \
        >> "$OUTPUT_FILE"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    echo "[a2a3:dev${DEVICE_ID}] FAIL: $TEST_NAME"
else
    printf '**Status**: PASS\n\n' >> "$OUTPUT_FILE"
    report_metric_table_2 "${OUT_FILES[0]}" "${OUT_FILES[1]}"
    printf '\n' >> "$OUTPUT_FILE"
    PASS_COUNT=$((PASS_COUNT + 1))
    echo "[a2a3:dev${DEVICE_ID}] PASS: $TEST_NAME"
fi

# ─── Summary section ─────────────────────────────────────────────────────────
cat >> "$OUTPUT_FILE" << EOF

---

## Summary

| Result | Count |
|---|---|
| Total  | $((PASS_COUNT + FAIL_COUNT)) |
| Passed | ${PASS_COUNT} |
| Failed | ${FAIL_COUNT} |

> **Primary metric**: dispatch.pop (batch pop gain, D=1 vs D=0).
> Distribution columns: min/p50/p90/max across all scheduler threads × runs.
> Raw data directory: ${DATA_DIR}
EOF

echo ""
echo "============================================================"
echo "  Report written to: $OUTPUT_FILE"
printf "  Total: %d   Passed: %d   Failed: %d\n" \
    "$((PASS_COUNT + FAIL_COUNT))" "$PASS_COUNT" "$FAIL_COUNT"
echo "============================================================"

[[ $FAIL_COUNT -eq 0 ]]

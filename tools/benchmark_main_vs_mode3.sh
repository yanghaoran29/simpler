#!/usr/bin/env bash
# Main (mode 2) vs RTT (mode 3) using tools/benchmark_rounds.sh.
# Each example: compile once, then ROUNDS on-device runs. Skip qwen3 by default.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ROUNDS="${ROUNDS:-20}"
DEVICE_ID="${TASK_DEVICE:-0}"
export BENCHMARK_SKIP_EXAMPLES="${BENCHMARK_SKIP_EXAMPLES:-qwen3_14b_decode}"

# Force this repo's simpler_setup (avoid stale PYTHONPATH from other checkouts).
unset PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT"

SUMMARY_FILE="/home/pyptouser/yanghaoran/tmp/benchmark_main_vs_mode3_$(date +%Y%m%d_%H%M%S).txt"
exec > >(tee "$SUMMARY_FILE") 2>&1

echo "=== benchmark_rounds.sh: main (mode 2) vs RTT (mode 3) ==="
echo "ROUNDS=$ROUNDS skip=$BENCHMARK_SKIP_EXAMPLES device=$DEVICE_ID"
echo "Each example compiles once per mode pass, then runs $ROUNDS rounds on device."
echo "Trimmed avg: drop floor(n/4) low + high (20 rounds -> drop 5+5)."
echo "Summary: $SUMMARY_FILE"
echo

run_pass() {
    local mode="$1" label="$2"
    echo
    echo "################################################################"
    echo "# $label  (SIMPLER_SCHED_AICORE_ASSIGNMENT_OVERRIDE=$mode)"
    echo "################################################################"
    SIMPLER_SCHED_AICORE_ASSIGNMENT_OVERRIDE="$mode" \
        "$PROJECT_ROOT/tools/benchmark_rounds.sh" \
            -p a5 -d "$DEVICE_ID" -n "$ROUNDS" -r tensormap_and_ringbuffer
}

run_pass 2 "main round-robin baseline"
run_pass 3 "RTT die preflight"

echo
echo "=== Done. See Trimmed Avg lines per example above. Summary: $SUMMARY_FILE ==="

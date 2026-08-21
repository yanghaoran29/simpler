#!/usr/bin/env bash
# Compare die-aware scheduler placement vs mainline round-robin baseline on A5 hardware.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ROUNDS="${ROUNDS:-30}"
DEVICE_ID="${DEVICE_ID:-0}"
EXAMPLE_DIR="$PROJECT_ROOT/examples/a5/tensormap_and_ringbuffer/benchmark_bgemm"
TEST_FILE="$EXAMPLE_DIR/test_benchmark_bgemm.py"

usage() {
    cat <<'EOF'
benchmark_die_aware_vs_baseline.sh — A/B A5 scheduler assignment benchmark

Runs benchmark_bgemm Case0 twice:
  die-aware (override=1): die-aware host order + contiguous AICore blocks
  baseline   (override=2): FG host order + round-robin cluster assignment

Environment:
  ROUNDS=30          Number of timed rounds per mode
  DEVICE_ID=0        NPU id when not using task-submit
  SKIP_INSTALL=1     Skip pip install (use already-installed package)

Wrap on shared boxes:
  task-submit --device auto --device-num 1 --timeout 3600 --max-time 3600 \
      --run /path/to/simpler/tools/benchmark_die_aware_vs_baseline.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ -n "${TASK_DEVICE:-}" ]]; then
    DEVICE_ID="$TASK_DEVICE"
fi

if [[ ! -f "$TEST_FILE" ]]; then
    echo "ERROR: missing $TEST_FILE" >&2
    exit 1
fi

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
    echo "=== building a5 runtime (local build/lib) ==="
    mkdir -p "$PROJECT_ROOT/build/lib" "$PROJECT_ROOT/build/cache"
    if [[ ! -f "$PROJECT_ROOT/build/lib/libsimpler_log.so" ]]; then
        for candidate in \
            "$HOME/Desktop/simpler/build/lib/libsimpler_log.so" \
            "$HOME/.local/lib/python3.11/site-packages/simpler_setup/_assets/build/lib/libsimpler_log.so"; do
            if [[ -f "$candidate" ]]; then
                cp -f "$candidate" "$PROJECT_ROOT/build/lib/libsimpler_log.so"
                break
            fi
        done
    fi
    if [[ ! -f "$PROJECT_ROOT/build/lib/libsimpler_log.so" ]]; then
        echo "ERROR: libsimpler_log.so missing; run pip install once or set SKIP_INSTALL=1" >&2
        exit 1
    fi
    unset PYTHONPATH
    PYTHONPATH="$PROJECT_ROOT" python3 "$PROJECT_ROOT/simpler_setup/build_runtimes.py" \
        --platforms a5 \
        --lib-dir "$PROJECT_ROOT/build/lib" \
        --cache-dir "$PROJECT_ROOT/build/cache"
fi

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

run_mode() {
    local mode="$1"
    local label="$2"
    local log
    log="$(mktemp)"
    echo
    echo "========================================"
    echo " Mode: $label (SIMPLER_SCHED_AICORE_ASSIGNMENT_OVERRIDE=$mode)"
    echo "========================================"
    SIMPLER_SCHED_AICORE_ASSIGNMENT_OVERRIDE="$mode" \
        python3 "$TEST_FILE" \
            --platform a5 --device "$DEVICE_ID" \
            --rounds "$ROUNDS" --skip-golden --case Case0 \
            >"$log" 2>&1 || {
            echo "FAILED: see $log" >&2
            tail -40 "$log" >&2
            return 1
        }
    python3 -m simpler_setup.tools.strace_timing "$log" --rounds-table || true
    rm -f "$log"
}

run_mode 1 "die-aware optimized"
run_mode 2 "mainline round-robin baseline"

echo
echo "Done. Lower Effective/Sched us = better."

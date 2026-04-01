#!/usr/bin/env bash
# Post-run analyzers extracted from run_tests.sh.
# Inputs are passed via environment variables by run_tests.sh.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-}"
if [[ -z "$PROJECT_ROOT" ]]; then
    echo "[post_run_analysis] PROJECT_ROOT is required" >&2
    exit 1
fi
AICPU_UT_SCRIPT_DIR="${AICPU_UT_SCRIPT_DIR:-}"
if [[ -z "$AICPU_UT_SCRIPT_DIR" ]]; then
    echo "[post_run_analysis] AICPU_UT_SCRIPT_DIR is required" >&2
    exit 1
fi

# Quiet mode or non-perf runs: skip all post analysis.
if [[ -n "${AICPU_UT_QUIET:-}" || "${RUN_PERF:-false}" != "true" ]]; then
    exit 0
fi

# Final scheduler overhead summary (unless already emitted per-index).
if [[ "${SKIP_FINAL_ANALYSIS:-false}" != "true" ]]; then
    SCHED_SCRIPT="${AICPU_UT_SCRIPT_DIR}/tools/profiling/sched_overhead_analysis.py"
    if [[ -f "$SCHED_SCRIPT" ]]; then
        SIM_LOG_ARGS=()
        if [[ -n "${AICPU_UT_PHASE_LOG:-}" && -s "${AICPU_UT_PHASE_LOG:-}" ]]; then
            SIM_LOG_ARGS=("--sim-log" "$AICPU_UT_PHASE_LOG" "--no-sources")
            if [[ "${PROFILING_MODE:-0}" == "1" ]]; then
                grep -E "aicpu_orchestration_entry returned|Scheduler summary:" "$AICPU_UT_PHASE_LOG" 2>/dev/null || true
            fi
        elif [[ -n "${SIM_LOG:-}" && -s "${SIM_LOG:-}" ]]; then
            SIM_LOG_ARGS=("--sim-log" "$SIM_LOG" "--no-sources")
            if [[ "${PROFILING_MODE:-0}" == "1" ]]; then
                grep -E "aicpu_orchestration_entry returned|Scheduler summary:" "$SIM_LOG" 2>/dev/null || true
            fi
        fi

        if python3 "$SCHED_SCRIPT" "${SIM_LOG_ARGS[@]}" ${AICPU_UT_DEVICE_ID:+-d "$AICPU_UT_DEVICE_ID"} 2>/dev/null; then
            :
        else
            echo "  (Skip: no perf_swimlane_*.json in outputs/ and no --sim-log; or device log resolve failed.)"
        fi
    fi
fi

# Optional swimlane diagram generation.
if [[ "${GEN_SWIMLANE:-false}" == "true" && -n "${AICPU_UT_SWIMLANE_DIR:-}" ]]; then
    SWIMLANE_SCRIPT="${AICPU_UT_SCRIPT_DIR}/tools/profiling/swimlane_converter.py"
    MERMAID_SCRIPT="${AICPU_UT_SCRIPT_DIR}/tools/profiling/perf_to_mermaid.py"
    LATEST_SWIMLANE=$(ls -t "${AICPU_UT_SWIMLANE_DIR}"/perf_swimlane_*.json 2>/dev/null | head -1 || true)
    if [[ -n "$LATEST_SWIMLANE" ]]; then
        echo ""
        echo "============================================================"
        echo "  Generating swimlane diagrams"
        echo "============================================================"
        echo "  Source: $LATEST_SWIMLANE"
        if [[ -f "$SWIMLANE_SCRIPT" ]]; then
            python3 "$SWIMLANE_SCRIPT" "$LATEST_SWIMLANE" 2>/dev/null || \
                echo "  (swimlane_converter.py failed or no output)"
        fi
        if [[ -f "$MERMAID_SCRIPT" ]]; then
            python3 "$MERMAID_SCRIPT" "$LATEST_SWIMLANE" 2>/dev/null || \
                echo "  (perf_to_mermaid.py failed or no output)"
        fi
    fi
fi

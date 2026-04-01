#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_tests.sh — Build and run the benchmark/ orchestration samples
#
# Usage:
#   ./run_tests.sh --test <name>                   # build + run one benchmark (concurrent)
#   ./run_tests.sh --test <name> --orch            # orch-only variant
#   ./run_tests.sh --test <name> --sched           # sched-profiling-only variant
#   ./run_tests.sh --test <name> --build-only      # build without running
#   ./run_tests.sh --test <name> --opt-level 0     # compile with -O0
#   ./run_tests.sh --test <name> --profiling 0     # profiling off (default)
#   ./run_tests.sh --test <name> --profiling 1     # PTO2_PROFILING only
#   ./run_tests.sh --test <name> --profiling 2     # full profiling (Sched+Orch+Tensormap)
#   ./run_tests.sh --test <name> --profiling 2 --enable-l2-swimlane  # + real l2_perf_records.json → Perfetto trace (outputs/merged_swimlane_*.json)
#   ./run_tests.sh --list                          # list available benchmarks
#
# Available benchmarks (all single-case, idx 0):
#   test_paged_consumer       (benchmark/paged_consumer_block_table.cpp)
#   test_qwen3_tensormap      (benchmark/qwen3_decode_tensormap.cpp)
#   test_qwen3_manual_scope   (benchmark/qwen3_decode_manual_scope.cpp)
#
# Optional environment overrides:
#   TIMEOUT=300 ./run_tests.sh --test test_qwen3_manual_scope
#   BUILD_DIR=/tmp/my_build ./run_tests.sh --test test_qwen3_manual_scope
#   ORCH_CPU=4 SCHED_CPU0=8 ./run_tests.sh --test test_qwen3_manual_scope
#   PLATFORM_MAX_BLOCKDIM=32 ./run_tests.sh --test test_qwen3_manual_scope
# ─────────────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─── Test Registry ────────────────────────────────────────────────────────────
declare -A TEST_INDICES
TEST_INDICES["test_paged_consumer"]="0"
TEST_INDICES["test_qwen3_tensormap"]="0"
TEST_INDICES["test_qwen3_manual_scope"]="0"

ALL_TESTS=(test_paged_consumer test_qwen3_tensormap test_qwen3_manual_scope)

# ─── Defaults ─────────────────────────────────────────────────────────────────
TIMEOUT=${TIMEOUT:-600}
BUILD_DIR=${BUILD_DIR:-"${SCRIPT_DIR}/build"}

ORCH_CPU=${ORCH_CPU:-4}
SCHED_CPU0=${SCHED_CPU0:-8}
SCHED_CPU1=${SCHED_CPU1:-9}
SCHED_CPU2=${SCHED_CPU2:-10}
SCHED_CPU3=${SCHED_CPU3:-11}
SCHED_CPU4=${SCHED_CPU4:-12}
SCHED_CPU5=${SCHED_CPU5:-13}
SCHED_CPU6=${SCHED_CPU6:-14}
SCHED_CPU7=${SCHED_CPU7:-15}

PLATFORM_MAX_BLOCKDIM=${PLATFORM_MAX_BLOCKDIM:-24}
PLATFORM_AIC_CORES_PER_BLOCKDIM=${PLATFORM_AIC_CORES_PER_BLOCKDIM:-1}
PLATFORM_AIV_CORES_PER_BLOCKDIM=${PLATFORM_AIV_CORES_PER_BLOCKDIM:-2}
PLATFORM_MAX_AICPU_THREADS=${PLATFORM_MAX_AICPU_THREADS:-4}

# Profiling: --profiling <0|1|2>
#   0 = off (default)
#   1 = PTO2_PROFILING only (base)
#   2 = full profiling (PTO2_SCHED + PTO2_ORCH + PTO2_TENSORMAP)
PROFILING_MODE=${PROFILING_MODE:-0}

BUILD_ONLY=false
OPT_LEVEL=${OPT_LEVEL:-3}
AICPU_UT_PROFILE_CALLSTACK=${AICPU_UT_PROFILE_CALLSTACK:-ON}
FILTER_TEST=""
FILTER_IDX=""
THREAD_MODE="concurrent"        # set by --orch (orch-only) or --sched (sched profiling only)
ENABLE_L2_SWIMLANE=false        # set by --enable-l2-swimlane: real l2_perf_records.json + Perfetto trace

# ─── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --orch)         THREAD_MODE="orch"; shift ;;
        --sched)        THREAD_MODE="sched"; shift ;;
        --build-only)   BUILD_ONLY=true; shift ;;
        --enable-l2-swimlane) ENABLE_L2_SWIMLANE=true; shift ;;
        --opt-level)
            if [[ -z "${2:-}" ]]; then
                echo "--opt-level requires a numeric argument (0/1/2/3)." >&2; exit 1
            fi
            OPT_LEVEL="$2"; shift 2 ;;
        --profiling)
            if [[ -n "${2:-}" && "$2" =~ ^[012]$ ]]; then
                PROFILING_MODE="$2"; shift 2
            else
                PROFILING_MODE=2; shift
            fi ;;
        --test)
            if [[ -z "${2:-}" ]]; then
                echo "--test requires a test name argument." >&2; exit 1
            fi
            FILTER_TEST="$2"; shift 2 ;;
        --idx)
            if [[ -z "${2:-}" ]]; then
                echo "--idx requires a numeric argument." >&2; exit 1
            fi
            FILTER_IDX="$2"; shift 2 ;;
        --list)
            echo "Available benchmarks (use --orch or --sched to select thread mode):"
            for name in "${ALL_TESTS[@]}"; do
                printf "  %-28s (indices: %s)\n" "$name" "${TEST_INDICES[$name]}"
            done
            exit 0 ;;
        --help|-h)
            sed -n '2,28p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Use --help for usage." >&2
            exit 1 ;;
    esac
done

# ─── Validate --test / --idx ──────────────────────────────────────────────────
if [ -z "$FILTER_TEST" ]; then
    echo "No test selected. Use --test <name> (see --list)." >&2
    exit 1
fi
if [ -z "${TEST_INDICES[$FILTER_TEST]+x}" ]; then
    echo "Unknown test: '$FILTER_TEST'" >&2
    echo "Use --list to see available benchmarks." >&2
    exit 1
fi
if [ -n "$FILTER_IDX" ] && ! echo " ${TEST_INDICES[$FILTER_TEST]} " | grep -qw "$FILTER_IDX"; then
    echo "Index '$FILTER_IDX' is not valid for test '$FILTER_TEST'." >&2
    echo "  Valid indices: ${TEST_INDICES[$FILTER_TEST]}" >&2
    exit 1
fi

# ─── Map PROFILING_MODE → PTO2_* build flags ──────────────────────────────────
case "$PROFILING_MODE" in
    0) PTO2_PROFILING=OFF; PTO2_SCHED_PROFILING=OFF; PTO2_ORCH_PROFILING=OFF; PTO2_TENSORMAP_PROFILING=OFF ;;
    1) PTO2_PROFILING=ON;  PTO2_SCHED_PROFILING=OFF; PTO2_ORCH_PROFILING=OFF; PTO2_TENSORMAP_PROFILING=OFF ;;
    2) PTO2_PROFILING=ON;  PTO2_SCHED_PROFILING=ON;  PTO2_ORCH_PROFILING=ON;  PTO2_TENSORMAP_PROFILING=ON  ;;
    *) echo "Invalid PROFILING_MODE: $PROFILING_MODE (must be 0, 1, or 2)." >&2; exit 1 ;;
esac
if [ "$PTO2_PROFILING" = "ON" ]; then
    AICPU_UT_QUIET=
else
    AICPU_UT_QUIET=1   # profiling off: only output pass/fail summary
fi

# ─── Helpers ──────────────────────────────────────────────────────────────────
quiet_echo() { [ -z "${AICPU_UT_QUIET:-}" ] && echo "$@" || true; }
print_separator() { quiet_echo "============================================================"; }

# Map script test name + THREAD_MODE to actual binary.
get_binary_path() {
    local name="$1"
    local idx="$2"
    local suffix
    case "$THREAD_MODE" in
        orch)  suffix="orch_only" ;;
        sched) suffix="sched_prof_only" ;;
        *)     suffix="concurrent" ;;
    esac
    echo "${BIN_DIR}/${name}_${suffix}_${idx}"
}

PASS_COUNT=0
FAIL_COUNT=0
FAILED_TESTS=()

run_binary() {
    local bin="$1"
    local label="$2"
    shift 2
    if [ -n "${AICPU_UT_QUIET:-}" ]; then
        if timeout "$TIMEOUT" "$bin" "$@" >/dev/null 2>&1; then
            return 0
        else
            return $?
        fi
    fi
    quiet_echo ""
    print_separator
    quiet_echo "  Running: $label"
    print_separator
    if [ -n "${SIM_LOG:-}" ]; then
        local tmp_log rc
        tmp_log=$(mktemp)
        if timeout "$TIMEOUT" "$bin" "$@" >"$tmp_log" 2>&1; then
            cat "$tmp_log"; cat "$tmp_log" >> "$SIM_LOG"; rm -f "$tmp_log"; return 0
        else
            rc=$?
            cat "$tmp_log"; cat "$tmp_log" >> "$SIM_LOG"; rm -f "$tmp_log"
            echo ""; echo "  FAILED (exit $rc)" >&2; return $rc
        fi
    else
        timeout "$TIMEOUT" "$bin" "$@" || { local rc=$?; echo ""; echo "  FAILED (exit $rc)" >&2; return $rc; }
    fi
}

run_test() {
    local bin="$1"
    local label="$2"
    shift 2
    if run_binary "$bin" "$label" "$@"; then
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_TESTS+=("$label")
    fi
}

# ─── Step 1: CMake configure ──────────────────────────────────────────────────
quiet_echo ""
print_separator
quiet_echo "  Configuring with CMake"
print_separator
quiet_echo "  Build dir : $BUILD_DIR"
quiet_echo "  Opt level : -O${OPT_LEVEL}"
quiet_echo "  Profiling : mode=$PROFILING_MODE (0=off 1=base 2=full)"

mkdir -p "$BUILD_DIR"

cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" \
    -DORCH_CPU="$ORCH_CPU" \
    -DSCHED_CPU0="$SCHED_CPU0" \
    -DSCHED_CPU1="$SCHED_CPU1" \
    -DSCHED_CPU2="$SCHED_CPU2" \
    -DSCHED_CPU3="$SCHED_CPU3" \
    -DSCHED_CPU4="$SCHED_CPU4" \
    -DSCHED_CPU5="$SCHED_CPU5" \
    -DSCHED_CPU6="$SCHED_CPU6" \
    -DSCHED_CPU7="$SCHED_CPU7" \
    -DPLATFORM_MAX_BLOCKDIM="$PLATFORM_MAX_BLOCKDIM" \
    -DPLATFORM_AIC_CORES_PER_BLOCKDIM="$PLATFORM_AIC_CORES_PER_BLOCKDIM" \
    -DPLATFORM_AIV_CORES_PER_BLOCKDIM="$PLATFORM_AIV_CORES_PER_BLOCKDIM" \
    -DPLATFORM_MAX_AICPU_THREADS="$PLATFORM_MAX_AICPU_THREADS" \
    -DOPT_LEVEL="$OPT_LEVEL" \
    -DAICPU_UT_PROFILE_CALLSTACK="$AICPU_UT_PROFILE_CALLSTACK" \
    -DPTO2_PROFILING="$PTO2_PROFILING" \
    -DPTO2_SCHED_PROFILING="$PTO2_SCHED_PROFILING" \
    -DPTO2_ORCH_PROFILING="$PTO2_ORCH_PROFILING" \
    -DPTO2_TENSORMAP_PROFILING="$PTO2_TENSORMAP_PROFILING"

# ─── Step 2: Build ────────────────────────────────────────────────────────────
BIN_DIR="${BUILD_DIR}/bin"
NPROC=$(nproc 2>/dev/null || echo 4)

IDX="${FILTER_IDX:-0}"
TARGET="$(basename "$(get_binary_path "$FILTER_TEST" "$IDX")")"

quiet_echo ""
print_separator
quiet_echo "  Building: $TARGET"
print_separator
cmake --build "$BUILD_DIR" --target "$TARGET" --parallel "$NPROC"
quiet_echo "  Built: ${BIN_DIR}/${TARGET}"

if $BUILD_ONLY; then
    quiet_echo ""
    quiet_echo "Build complete (--build-only, skipping run)."
    exit 0
fi

# ─── Step 3: Run ──────────────────────────────────────────────────────────────
PROJECT_ROOT="${SCRIPT_DIR}/../.."
SIM_LOG=""
if [ -z "${AICPU_UT_QUIET:-}" ]; then
    mkdir -p "$PROJECT_ROOT/outputs"
    SIM_LOG="$PROJECT_ROOT/outputs/aicpu_ut_sim_run.log"
    : > "$SIM_LOG"
fi
if $ENABLE_L2_SWIMLANE; then
    # Activates SimL2Swimlane in the binary (drives the real host L2PerfCollector
    # → outputs/l2_perf_records.json), then converted to a Perfetto trace below.
    export AICPU_UT_L2_SWIMLANE=1
    mkdir -p "$PROJECT_ROOT/outputs"
    export AICPU_UT_SWIMLANE_DIR="$PROJECT_ROOT/outputs"
fi

quiet_echo ""
print_separator
quiet_echo "  Running tests"
print_separator

mode_suffix=""
[ "$THREAD_MODE" = "orch" ]  && mode_suffix=" [orch]"
[ "$THREAD_MODE" = "sched" ] && mode_suffix=" [sched]"
run_test "${BIN_DIR}/${TARGET}" "${FILTER_TEST}[${IDX}]${mode_suffix}"

# ─── Post-run swimlane conversion (mainline tooling) ────────────────────────
# --enable-l2-swimlane makes the binary emit outputs/l2_perf_records.json; convert
# it to a Perfetto trace via the mainline simpler_setup.tools.swimlane_converter.
if $ENABLE_L2_SWIMLANE && [ -z "${AICPU_UT_QUIET:-}" ]; then
    L2_JSON="$PROJECT_ROOT/outputs/l2_perf_records.json"
    if [ -f "$L2_JSON" ]; then
        TS=$(date +%Y%m%d_%H%M%S)
        MERGED="$PROJECT_ROOT/outputs/merged_swimlane_${TS}.json"
        quiet_echo ""
        print_separator
        quiet_echo "  Generating swimlane (Perfetto) via mainline converter"
        print_separator
        ( cd "$PROJECT_ROOT" && PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
            python3 -m simpler_setup.tools.swimlane_converter "$L2_JSON" -o "$MERGED" ) || \
            echo "  (swimlane_converter failed)"
        quiet_echo "  Perfetto trace: $MERGED  (drag into https://ui.perfetto.dev/)"
    else
        echo "  [l2-swimlane] no $L2_JSON produced (enable only supported on the concurrent perf driver)."
    fi
fi

# ─── Summary ──────────────────────────────────────────────────────────────────
print_separator
quiet_echo "  Test Run Summary"
print_separator
quiet_echo "  Passed: $PASS_COUNT"
quiet_echo "  Failed: $FAIL_COUNT"

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    quiet_echo ""
    quiet_echo "  Failed tests:"
    for t in "${FAILED_TESTS[@]}"; do
        quiet_echo "    - $t"
    done
    quiet_echo ""
    quiet_echo "  OVERALL: FAILED"
    exit 1
else
    quiet_echo ""
    quiet_echo "  OVERALL: PASSED"
    exit 0
fi

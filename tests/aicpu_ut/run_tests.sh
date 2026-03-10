#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_tests.sh — Build and run orchestration unit tests
#
# Usage:
#   ./run_tests.sh                                 # run all tests
#   ./run_tests.sh --func                          # functional tests only
#   ./run_tests.sh --perf                          # performance tests only
#   ./run_tests.sh --test <name>                   # run one test (all parameter sets)
#   ./run_tests.sh --test <name> --idx <n>         # run one specific parameter set
#   ./run_tests.sh --build-only                    # build without running
#   ./run_tests.sh --no-profiling                  # disable all profiling (PTO2_*_PROFILING=OFF)
#   ./run_tests.sh --no-sched-profiling            # disable only PTO2_SCHED_PROFILING
#   ./run_tests.sh --no-orch-profiling             # disable only PTO2_ORCH_PROFILING
#   ./run_tests.sh --list                          # list all available tests
#
# Available tests:
#   test_cpu_affinity               (functional)
#   test_platform_config            (functional)
#   test_paged_attention            (perf, indices: 0)
#   test_batch_paged_attention      (perf, indices: 0 1 2)
#
# Optional environment overrides:
#   TIMEOUT=300 ./run_tests.sh
#   ORCH_CPU=4 SCHED_CPU0=5 ./run_tests.sh
#   PLATFORM_MAX_BLOCKDIM=32 ./run_tests.sh
#   BUILD_DIR=/tmp/my_build ./run_tests.sh
#   PTO2_PROFILING=OFF ./run_tests.sh
#   PTO2_SCHED_PROFILING=OFF PTO2_ORCH_PROFILING=ON ./run_tests.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─── Test Registry ────────────────────────────────────────────────────────────
# TEST_TYPE:    "func" or "perf"
# TEST_INDICES: "-" for single-binary tests; space-separated indices for multi-case tests
declare -A TEST_TYPE
declare -A TEST_INDICES

TEST_TYPE["test_cpu_affinity"]="func"            ; TEST_INDICES["test_cpu_affinity"]="-"
TEST_TYPE["test_platform_config"]="func"         ; TEST_INDICES["test_platform_config"]="-"
TEST_TYPE["test_paged_attention"]="perf"         ; TEST_INDICES["test_paged_attention"]="0"
TEST_TYPE["test_batch_paged_attention"]="perf"   ; TEST_INDICES["test_batch_paged_attention"]="0 1 2"

ALL_TESTS=(test_cpu_affinity test_platform_config test_paged_attention test_batch_paged_attention)

# ─── Defaults ─────────────────────────────────────────────────────────────────
TIMEOUT=${TIMEOUT:-600}
BUILD_DIR=${BUILD_DIR:-"${SCRIPT_DIR}/build"}

ORCH_CPU=${ORCH_CPU:-0}
SCHED_CPU0=${SCHED_CPU0:-1}
SCHED_CPU1=${SCHED_CPU1:-2}
SCHED_CPU2=${SCHED_CPU2:-3}
SCHED_CPU3=${SCHED_CPU3:-4}
SCHED_CPU4=${SCHED_CPU4:-5}
SCHED_CPU5=${SCHED_CPU5:-6}
SCHED_CPU6=${SCHED_CPU6:-7}
SCHED_CPU7=${SCHED_CPU7:-8}

PLATFORM_MAX_BLOCKDIM=${PLATFORM_MAX_BLOCKDIM:-24}
PLATFORM_AIC_CORES_PER_BLOCKDIM=${PLATFORM_AIC_CORES_PER_BLOCKDIM:-1}
PLATFORM_AIV_CORES_PER_BLOCKDIM=${PLATFORM_AIV_CORES_PER_BLOCKDIM:-2}
PLATFORM_MAX_AICPU_THREADS=${PLATFORM_MAX_AICPU_THREADS:-4}

# Profiling: default all ON. --no-profiling sets all OFF. PTO2_PROFILING is OFF only when both SCHED and ORCH are OFF.
PTO2_PROFILING=${PTO2_PROFILING:-ON}
PTO2_SCHED_PROFILING=${PTO2_SCHED_PROFILING:-ON}
PTO2_ORCH_PROFILING=${PTO2_ORCH_PROFILING:-ON}

RUN_FUNC=true
RUN_PERF=true
BUILD_ONLY=false
FILTER_TEST=""
FILTER_IDX=""

# ─── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --func)         RUN_PERF=false; shift ;;
        --perf)         RUN_FUNC=false; shift ;;
        --build-only)   BUILD_ONLY=true; shift ;;
        --no-profiling)       PTO2_PROFILING=OFF; PTO2_SCHED_PROFILING=OFF; PTO2_ORCH_PROFILING=OFF; shift ;;
        --no-sched-profiling) PTO2_SCHED_PROFILING=OFF; shift ;;
        --no-orch-profiling)  PTO2_ORCH_PROFILING=OFF; shift ;;
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
            echo "Available tests:"
            for name in "${ALL_TESTS[@]}"; do
                t="${TEST_TYPE[$name]}"
                idx="${TEST_INDICES[$name]}"
                if [ "$idx" = "-" ]; then
                    printf "  %-36s (%s)\n" "$name" "$t"
                else
                    printf "  %-36s (%s, indices: %s)\n" "$name" "$t" "$idx"
                fi
            done
            exit 0 ;;
        --help|-h)
            sed -n '2,26p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Use --help for usage." >&2
            exit 1 ;;
    esac
done

# ─── Validate --test / --idx ──────────────────────────────────────────────────
if [ -n "$FILTER_TEST" ]; then
    if [ -z "${TEST_TYPE[$FILTER_TEST]+x}" ]; then
        echo "Unknown test: '$FILTER_TEST'" >&2
        echo "Use --list to see available tests." >&2
        exit 1
    fi
    if [ -n "$FILTER_IDX" ]; then
        if [ "${TEST_INDICES[$FILTER_TEST]}" = "-" ]; then
            echo "Test '$FILTER_TEST' is a single-binary test; --idx is not applicable." >&2
            exit 1
        fi
        if ! echo " ${TEST_INDICES[$FILTER_TEST]} " | grep -qw "$FILTER_IDX"; then
            echo "Index '$FILTER_IDX' is not valid for test '$FILTER_TEST'." >&2
            echo "  Valid indices: ${TEST_INDICES[$FILTER_TEST]}" >&2
            exit 1
        fi
    fi
fi

# PTO2_PROFILING is OFF only when both PTO2_SCHED_PROFILING and PTO2_ORCH_PROFILING are OFF
if [ "$PTO2_SCHED_PROFILING" = "OFF" ] && [ "$PTO2_ORCH_PROFILING" = "OFF" ]; then
    PTO2_PROFILING=OFF
else
    PTO2_PROFILING=ON
fi

# ─── Helpers ──────────────────────────────────────────────────────────────────
print_separator() {
    printf '%0.s─' $(seq 1 70); echo
}

run_binary() {
    local bin="$1"
    local label="$2"
    shift 2
    echo ""
    print_separator
    echo "  Running: $label"
    print_separator
    if timeout "$TIMEOUT" "$bin" "$@"; then
        return 0
    else
        local rc=$?
        if [ $rc -eq 124 ]; then
            echo ""
            echo "  TIMEOUT: exceeded ${TIMEOUT}s" >&2
        else
            echo ""
            echo "  FAILED (exit $rc)" >&2
        fi
        return $rc
    fi
}

PASS_COUNT=0
FAIL_COUNT=0
FAILED_TESTS=()

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

# Run all indices (or a specific one) for a registered test.
run_test_entry() {
    local name="$1"
    local idx="$2"   # empty = run all indices
    local type="${TEST_TYPE[$name]}"
    local indices="${TEST_INDICES[$name]}"

    if [ "$indices" = "-" ]; then
        run_test "${BIN_DIR}/${name}" "${name} (${type})"
    elif [ -n "$idx" ]; then
        run_test "${BIN_DIR}/${name}_${idx}" "${name}[${idx}] (${type})"
    else
        for i in $indices; do
            run_test "${BIN_DIR}/${name}_${i}" "${name}[${i}] (${type})"
        done
    fi
}

# ─── Step 1: CMake configure ──────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Configuring with CMake"
echo "============================================================"
echo "  Build dir : $BUILD_DIR"
echo "  Source dir: $SCRIPT_DIR"
echo "  Profiling : PTO2_PROFILING=$PTO2_PROFILING PTO2_SCHED_PROFILING=$PTO2_SCHED_PROFILING PTO2_ORCH_PROFILING=$PTO2_ORCH_PROFILING"

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
    -DPTO2_PROFILING="$PTO2_PROFILING" \
    -DPTO2_SCHED_PROFILING="$PTO2_SCHED_PROFILING" \
    -DPTO2_ORCH_PROFILING="$PTO2_ORCH_PROFILING"

# ─── Step 2: Build ────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Building all test binaries"
echo "============================================================"

NPROC=$(nproc 2>/dev/null || echo 4)
cmake --build "$BUILD_DIR" --parallel "$NPROC"

BIN_DIR="${BUILD_DIR}/bin"
echo ""
echo "  Built binaries:"
for b in "${BIN_DIR}"/test_*; do
    echo "    $b"
done

if $BUILD_ONLY; then
    echo ""
    echo "Build complete (--build-only, skipping tests)."
    exit 0
fi

# ─── Step 3: Run tests ────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Running tests"
echo "============================================================"

if [ -n "$FILTER_TEST" ]; then
    run_test_entry "$FILTER_TEST" "$FILTER_IDX"
else
    for name in "${ALL_TESTS[@]}"; do
        type="${TEST_TYPE[$name]}"
        if { [ "$type" = "func" ] && $RUN_FUNC; } || \
           { [ "$type" = "perf" ] && $RUN_PERF; }; then
            run_test_entry "$name" ""
        fi
    done
fi

# ─── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Test Run Summary"
echo "============================================================"
echo "  Passed: $PASS_COUNT"
echo "  Failed: $FAIL_COUNT"

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    echo ""
    echo "  Failed tests:"
    for t in "${FAILED_TESTS[@]}"; do
        echo "    - $t"
    done
    echo ""
    echo "  OVERALL: FAILED"
    exit 1
else
    echo ""
    echo "  OVERALL: PASSED"
    exit 0
fi

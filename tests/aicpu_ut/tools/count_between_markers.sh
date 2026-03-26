#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_TESTS_SH="${UT_DIR}/run_tests.sh"
PLUGIN_SO="${UT_DIR}/plugins/libinsn_count.so"
QEMU_BIN="${QEMU_BIN:-/data/y00955915/.local/bin/qemu-aarch64}"
LOG_DIR="${UT_DIR}/outputs/log"

TEST_NAME="${TEST_NAME:-test_batch_paged_attention}"
TEST_IDX="${TEST_IDX:-0}"
THREAD_MODE="${THREAD_MODE:-concurrent}"  # concurrent|orch|sched

# Marker encodings for:
#   start: orr x3, x3, x3  -> 0xaa030063
#   end  : orr x4, x4, x4  -> 0xaa040084
MARKER_START="${MARKER_START:-0xaa030063}"
MARKER_END="${MARKER_END:-0xaa040084}"

usage() {
    cat <<'EOF'
Usage:
  ./tools/count_between_markers.sh [options]

Options:
  --test <name>            Test name (default: test_batch_paged_attention)
  --idx <n>                Test index (default: 0)
  --thread-mode <mode>     concurrent|orch|sched (default: concurrent)
  --marker-start <hex>     Start marker encoding (default: 0xaa030063)
  --marker-end <hex>       End marker encoding (default: 0xaa040084)
  --qemu-bin <path>        qemu-aarch64 path
  --help                   Show this help

Environment overrides:
  TEST_NAME, TEST_IDX, THREAD_MODE, MARKER_START, MARKER_END, QEMU_BIN
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test) TEST_NAME="$2"; shift 2 ;;
        --idx) TEST_IDX="$2"; shift 2 ;;
        --thread-mode) THREAD_MODE="$2"; shift 2 ;;
        --marker-start) MARKER_START="$2"; shift 2 ;;
        --marker-end) MARKER_END="$2"; shift 2 ;;
        --qemu-bin) QEMU_BIN="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

case "${THREAD_MODE}" in
    concurrent|orch|sched) ;;
    *) echo "Invalid --thread-mode: ${THREAD_MODE}" >&2; exit 1 ;;
esac

if [[ ! -x "${RUN_TESTS_SH}" ]]; then
    echo "Missing run_tests.sh: ${RUN_TESTS_SH}" >&2
    exit 1
fi
if [[ ! -f "${PLUGIN_SO}" ]]; then
    echo "Missing plugin: ${PLUGIN_SO}" >&2
    echo "Build plugin first: make -C \"${UT_DIR}/plugins\" QEMU_BUILD_DIR=\$QEMU_BUILD_DIR" >&2
    exit 1
fi
if [[ ! -x "${QEMU_BIN}" ]]; then
    echo "qemu-aarch64 not found/executable: ${QEMU_BIN}" >&2
    exit 1
fi

echo "[marker-count] build binary via run_tests.sh ..."
build_args=(--test "${TEST_NAME}" --idx "${TEST_IDX}" --build-only)
if [[ "${THREAD_MODE}" == "orch" ]]; then
    build_args+=(--orch)
elif [[ "${THREAD_MODE}" == "sched" ]]; then
    build_args+=(--sched)
fi
"${RUN_TESTS_SH}" "${build_args[@]}"

BIN_DIR="${UT_DIR}/build/bin"
case "${TEST_NAME}" in
    test_batch_paged_attention)
        case "${THREAD_MODE}" in
            concurrent) BIN="${BIN_DIR}/test_concurrent_${TEST_IDX}" ;;
            orch) BIN="${BIN_DIR}/test_orch_only_${TEST_IDX}" ;;
            sched) BIN="${BIN_DIR}/test_sched_prof_only_${TEST_IDX}" ;;
        esac
        ;;
    test_alt)
        case "${THREAD_MODE}" in
            concurrent) BIN="${BIN_DIR}/test_alt_concurrent_${TEST_IDX}" ;;
            orch) BIN="${BIN_DIR}/test_alt_orch_only_${TEST_IDX}" ;;
            sched) BIN="${BIN_DIR}/test_alt_sched_prof_only_${TEST_IDX}" ;;
        esac
        ;;
    test_bgemm)
        case "${THREAD_MODE}" in
            concurrent) BIN="${BIN_DIR}/test_bgemm_concurrent_${TEST_IDX}" ;;
            orch) BIN="${BIN_DIR}/test_bgemm_orch_only_${TEST_IDX}" ;;
            sched) BIN="${BIN_DIR}/test_bgemm_sched_prof_only_${TEST_IDX}" ;;
        esac
        ;;
    test_pau)
        case "${THREAD_MODE}" in
            concurrent) BIN="${BIN_DIR}/test_pau_concurrent_${TEST_IDX}" ;;
            orch) BIN="${BIN_DIR}/test_pau_orch_only_${TEST_IDX}" ;;
            sched) BIN="${BIN_DIR}/test_pau_sched_prof_only_${TEST_IDX}" ;;
        esac
        ;;
    test_throughput)
        case "${THREAD_MODE}" in
            concurrent) BIN="${BIN_DIR}/test_throughput_concurrent_${TEST_IDX}" ;;
            orch) BIN="${BIN_DIR}/test_throughput_orch_only_${TEST_IDX}" ;;
            sched) BIN="${BIN_DIR}/test_throughput_sched_prof_only_${TEST_IDX}" ;;
        esac
        ;;
    test_latency)
        case "${THREAD_MODE}" in
            concurrent) BIN="${BIN_DIR}/test_latency_concurrent_${TEST_IDX}" ;;
            orch) BIN="${BIN_DIR}/test_latency_orch_only_${TEST_IDX}" ;;
            sched) BIN="${BIN_DIR}/test_latency_sched_prof_only_${TEST_IDX}" ;;
        esac
        ;;
    test_paged_attention)
        BIN="${BIN_DIR}/test_pa_concurrent_${TEST_IDX}"
        ;;
    *)
        echo "Unsupported test for auto binary mapping: ${TEST_NAME}" >&2
        echo "Please extend script mapping if needed." >&2
        exit 1
        ;;
esac

if [[ ! -x "${BIN}" ]]; then
    echo "Built binary not found: ${BIN}" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
OUTFILE="${LOG_DIR}/${TEST_NAME}_${TEST_IDX}_between_markers_${TS}.txt"
RUNLOG="${LOG_DIR}/${TEST_NAME}_${TEST_IDX}_submit_args_${TS}.log"

echo "[marker-count] run pass#1 (insn_count only): ${BIN}"
"${QEMU_BIN}" -plugin "file=${PLUGIN_SO},outfile=${OUTFILE},markers=1,marker_start=${MARKER_START},marker_end=${MARKER_END}" "${BIN}" >/dev/null 2>&1 || true

# Pass#2: rebuild with submit-args trace macro and run once without plugin
# to collect pto2_submit_mixed_task argument lines.
echo "[marker-count] run pass#2 (submit args trace): rebuild with PTO2_TRACE_SUBMIT_ARGS_ENABLE=1"
trace_build_args=(--test "${TEST_NAME}" --idx "${TEST_IDX}" --build-only)
if [[ "${THREAD_MODE}" == "orch" ]]; then
    trace_build_args+=(--orch)
elif [[ "${THREAD_MODE}" == "sched" ]]; then
    trace_build_args+=(--sched)
fi
PTO2_TRACE_SUBMIT_ARGS_ENABLE=1 "${RUN_TESTS_SH}" "${trace_build_args[@]}"

echo "[marker-count] run pass#2 binary: ${BIN}"
"${BIN}" >"${RUNLOG}" 2>&1 || true

if rg -q "^\\[submit-args\\]" "${RUNLOG}"; then
    {
        echo ""
        echo "# pto2_submit_mixed_task args"
        rg "^\\[submit-args\\]" "${RUNLOG}"
    } >> "${OUTFILE}"
fi

echo "[marker-count] result file: ${OUTFILE}"
sed -n '1,12p' "${OUTFILE}"

#!/usr/bin/env bash
# Instruction-type + memory histogram inside resolve_and_dispatch_pto2 scheduler loop (orr x23/x24).
# Requires PTO2_INSTR_COUNT_SCHEDULER_LOOP_ENABLE; plugin single marker pair (no marker_phases).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_TESTS_SH="${UT_DIR}/run_tests.sh"
PLUGIN_SO="${UT_DIR}/plugins/libinsn_count.so"
QEMU_BIN="${QEMU_BIN:-"$("${UT_DIR}/tools/resolve_plugin_qemu.sh" "${UT_DIR}")"}"
LOG_DIR="${UT_DIR}/outputs/log"

TEST_NAME="${TEST_NAME:-test_batch_paged_attention}"
TEST_IDX="${TEST_IDX:-0}"
THREAD_MODE="${THREAD_MODE:-concurrent}"
MARKER_LOOP_START="${MARKER_LOOP_START:-0xaa1702f7}"
MARKER_LOOP_END="${MARKER_LOOP_END:-0xaa180318}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test) TEST_NAME="$2"; shift 2 ;;
        --idx) TEST_IDX="$2"; shift 2 ;;
        --thread-mode) THREAD_MODE="$2"; shift 2 ;;
        --marker-start) MARKER_LOOP_START="$2"; shift 2 ;;
        --marker-end) MARKER_LOOP_END="$2"; shift 2 ;;
        --qemu-bin) QEMU_BIN="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

case "${THREAD_MODE}" in
    concurrent|orch|sched) ;;
    *) echo "Invalid --thread-mode: ${THREAD_MODE}" >&2; exit 1 ;;
esac

if [[ ! -f "${PLUGIN_SO}" ]]; then
    echo "Missing plugin: ${PLUGIN_SO}" >&2
    exit 1
fi
if [[ ! -x "${QEMU_BIN}" ]]; then
    echo "qemu-aarch64 not found: ${QEMU_BIN}" >&2
    exit 1
fi

BIN_DIR="${UT_DIR}/build/bin"
case "${TEST_NAME}" in
    test_batch_paged_attention)
        case "${THREAD_MODE}" in
            concurrent) BIN="${BIN_DIR}/test_concurrent_${TEST_IDX}" ;;
            orch) BIN="${BIN_DIR}/test_orch_only_${TEST_IDX}" ;;
            sched) BIN="${BIN_DIR}/test_sched_prof_only_${TEST_IDX}" ;;
        esac ;;
    test_pau|test_paged_attention_unroll)
        case "${THREAD_MODE}" in
            concurrent) BIN="${BIN_DIR}/test_pau_concurrent_${TEST_IDX}" ;;
            orch) BIN="${BIN_DIR}/test_pau_orch_only_${TEST_IDX}" ;;
            sched) BIN="${BIN_DIR}/test_pau_sched_prof_only_${TEST_IDX}" ;;
        esac ;;
    *)
        echo "Unsupported test: ${TEST_NAME}" >&2
        exit 1
        ;;
esac

echo "[sched-insn-types] build (PTO2_INSTR_COUNT_SCHEDULER_LOOP_ENABLE=1, trace off) ..."
build_args=(--test "${TEST_NAME}" --idx "${TEST_IDX}" --build-only)
[[ "${THREAD_MODE}" == "orch" ]]  && build_args+=(--orch)
[[ "${THREAD_MODE}" == "sched" ]] && build_args+=(--sched)
PTO2_INSTR_COUNT_SCHEDULER_LOOP_ENABLE=1 PTO2_TRACE_SUBMIT_ARGS_ENABLE=0 "${RUN_TESTS_SH}" "${build_args[@]}"

mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
OUTFILE="${LOG_DIR}/${TEST_NAME}_${TEST_IDX}_scheduler_insn_types_${TS}.txt"

echo "[sched-insn-types] QEMU+plugin (markers x23/x24, insn_types+mem, no marker_phases): ${BIN}"
set +e
_qlog="$(mktemp)"
"${QEMU_BIN}" -plugin "file=${PLUGIN_SO},outfile=${OUTFILE},markers=1,insn_types=1,insn_mem=1,marker_start=${MARKER_LOOP_START},marker_end=${MARKER_LOOP_END}" "${BIN}" >"${_qlog}" 2>&1
_rc=$?
set -e
if [[ ${_rc} -ne 0 ]] || [[ ! -s "${OUTFILE}" ]]; then
    echo "[sched-insn-types] WARN: QEMU rc=${_rc}; outfile missing or empty." >&2
    [[ -s "${_qlog}" ]] && tail -n 15 "${_qlog}" >&2 || true
fi
rm -f "${_qlog}"

echo "[sched-insn-types] result: ${OUTFILE}"
head -25 "${OUTFILE}"

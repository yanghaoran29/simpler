#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_TESTS_SH="${UT_DIR}/run_tests.sh"
PLUGIN_SO="${UT_DIR}/plugins/libinsn_count.so"
# Default: qemu-aarch64 from plugins/Makefile QEMU_BUILD_DIR (must match libinsn_count.so; ~/.local/bin often differs and causes SIGTRAP/double-free).
QEMU_BIN="${QEMU_BIN:-"$("${UT_DIR}/tools/resolve_plugin_qemu.sh" "${UT_DIR}")"}"
LOG_DIR="${UT_DIR}/outputs/log"

TEST_NAME="${TEST_NAME:-test_batch_paged_attention}"
TEST_IDX="${TEST_IDX:-0}"
THREAD_MODE="${THREAD_MODE:-concurrent}"  # concurrent|orch|sched

# Marker encodings for pto2_submit_mixed_task outer pair + phase nesting:
#   start: orr x3, x3, x3  -> 0xaa030063
#   end  : orr x4, x4, x4  -> 0xaa040084
MARKER_START="${MARKER_START:-0xaa030063}"
MARKER_END="${MARKER_END:-0xaa040084}"
# build_graph only in test_orchestrator_scheduler.cpp (concurrent), when PTO2_INSTR_COUNT_BUILD_GRAPH_ENABLE=ON:
#   start: orr x17, x17, x17 -> 0xaa110231
#   end  : orr x18, x18, x18 -> 0xaa120252
MARKER_BUILD_GRAPH_START="${MARKER_BUILD_GRAPH_START:-0xaa110231}"
MARKER_BUILD_GRAPH_END="${MARKER_BUILD_GRAPH_END:-0xaa120252}"

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
  TEST_NAME, TEST_IDX, THREAD_MODE, MARKER_START, MARKER_END,
  MARKER_BUILD_GRAPH_START, MARKER_BUILD_GRAPH_END, QEMU_BIN
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

echo "[marker-count] build binary via run_tests.sh (PTO2_INSTR_COUNT_BUILD_GRAPH_ENABLE=1 for build_graph markers) ..."
build_args=(--test "${TEST_NAME}" --idx "${TEST_IDX}" --build-only)
if [[ "${THREAD_MODE}" == "orch" ]]; then
    build_args+=(--orch)
elif [[ "${THREAD_MODE}" == "sched" ]]; then
    build_args+=(--sched)
fi
PTO2_INSTR_COUNT_BUILD_GRAPH_ENABLE=1 "${RUN_TESTS_SH}" "${build_args[@]}"

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
    test_pau|test_paged_attention_unroll)
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
BG_OUT="${LOG_DIR}/${TEST_NAME}_${TEST_IDX}_build_graph_${TS}.txt"
PHASE_OUT="${LOG_DIR}/${TEST_NAME}_${TEST_IDX}_submit_phases_${TS}.txt"
RUNLOG="${LOG_DIR}/${TEST_NAME}_${TEST_IDX}_submit_args_${TS}.log"

# Pass#0: build_graph only — orr x17/x18 (concurrent driver); other thread modes have no such markers.
if [[ "${THREAD_MODE}" == "concurrent" ]]; then
    echo "[marker-count] run pass#0 (build_graph insn_count, markers x17/x18): ${BIN}"
    if [[ "$(uname -m 2>/dev/null)" == "aarch64" ]]; then
        echo "[marker-count] note: host is aarch64; qemu-aarch64+TCG plugin sometimes hits SIGTRAP (Trace/breakpoint trap)." >&2
        echo "[marker-count]       if pass#0 fails, try another QEMU build (QEMU_BIN=...) or run this script on x86_64 with user-mode aarch64." >&2
    fi
    set +e
    _qemu_log="$(mktemp)"
    "${QEMU_BIN}" -plugin "file=${PLUGIN_SO},outfile=${BG_OUT},markers=1,marker_start=${MARKER_BUILD_GRAPH_START},marker_end=${MARKER_BUILD_GRAPH_END}" "${BIN}" >"${_qemu_log}" 2>&1
    _pass0_rc=$?
    set -e
    if [[ ${_pass0_rc} -ne 0 ]] || [[ ! -s "${BG_OUT}" ]]; then
        _p0_sig=""
        if [[ ${_pass0_rc} -gt 128 && ${_pass0_rc} -lt 256 ]]; then
            _s=$((_pass0_rc - 128))
            _p0_sig=" (128+${_s}"
            [[ ${_s} -eq 5 ]] && _p0_sig+=", SIGTRAP"
            _p0_sig+=")"
        fi
        echo "[marker-count] WARN: pass#0 QEMU rc=${_pass0_rc}${_p0_sig}; plugin outfile missing or empty. build_graph insn skipped." >&2
        echo "[marker-count]       if log shows 'double free or corruption', rebuild plugins with QEMU_BUILD_DIR matching QEMU_BIN (see ${UT_DIR}/plugins/Makefile)." >&2
        if [[ -s "${_qemu_log}" ]]; then
            echo "[marker-count] --- QEMU log tail ---" >&2
            tail -n 20 "${_qemu_log}" | sed 's/^/[qemu] /' >&2 || true
        fi
        if [[ ! -f "${BG_OUT}" ]] || [[ ! -s "${BG_OUT}" ]]; then
            {
                echo "# pass#0 failed: QEMU exited ${_pass0_rc}${_p0_sig}"
                echo "# common: SIGTRAP / heap corruption when plugin and QEMU_BIN are from different QEMU builds."
                echo "# raw tail from QEMU run:"
                tail -n 30 "${_qemu_log}" 2>/dev/null || true
            } >"${BG_OUT}"
        fi
    fi
    rm -f "${_qemu_log}"
else
    echo "[marker-count] skip pass#0 build_graph (markers exist only in concurrent test_orchestrator_scheduler.cpp)"
    echo "# (skipped: use --thread-mode concurrent for build_graph insn markers)" >"${BG_OUT}"
fi

echo "[marker-count] run pass#1 (pto2_submit_mixed_task phase insn_count): ${BIN}"
set +e
_qemu_log1="$(mktemp)"
"${QEMU_BIN}" -plugin "file=${PLUGIN_SO},outfile=${PHASE_OUT},markers=1,marker_phases=1,marker_start=${MARKER_START},marker_end=${MARKER_END}" "${BIN}" >"${_qemu_log1}" 2>&1
_pass1_rc=$?
set -e
if [[ ${_pass1_rc} -ne 0 ]] || [[ ! -s "${PHASE_OUT}" ]]; then
    _p1_sig=""
    if [[ ${_pass1_rc} -gt 128 && ${_pass1_rc} -lt 256 ]]; then
        _s=$((_pass1_rc - 128))
        _p1_sig=" (128+${_s}"
        [[ ${_s} -eq 5 ]] && _p1_sig+=", SIGTRAP"
        _p1_sig+=")"
    fi
    echo "[marker-count] WARN: pass#1 QEMU rc=${_pass1_rc}${_p1_sig}; phase outfile missing or empty." >&2
    if [[ -s "${_qemu_log1}" ]]; then
        echo "[marker-count] --- QEMU log tail ---" >&2
        tail -n 20 "${_qemu_log1}" | sed 's/^/[qemu] /' >&2 || true
    fi
    if [[ ! -f "${PHASE_OUT}" ]] || [[ ! -s "${PHASE_OUT}" ]]; then
        echo "# pass#1 failed: QEMU exited ${_pass1_rc}" >"${PHASE_OUT}"
        tail -n 30 "${_qemu_log1}" >>"${PHASE_OUT}" 2>/dev/null || true
    fi
fi
rm -f "${_qemu_log1}"

# Pass#2: rebuild without build_graph markers but with submit-args trace; run once without plugin.
echo "[marker-count] run pass#2 (submit args trace): rebuild PTO2_TRACE_SUBMIT_ARGS_ENABLE=1, PTO2_INSTR_COUNT_BUILD_GRAPH_ENABLE=0"
trace_build_args=(--test "${TEST_NAME}" --idx "${TEST_IDX}" --build-only)
if [[ "${THREAD_MODE}" == "orch" ]]; then
    trace_build_args+=(--orch)
elif [[ "${THREAD_MODE}" == "sched" ]]; then
    trace_build_args+=(--sched)
fi
PTO2_TRACE_SUBMIT_ARGS_ENABLE=1 PTO2_INSTR_COUNT_BUILD_GRAPH_ENABLE=0 "${RUN_TESTS_SH}" "${trace_build_args[@]}"

echo "[marker-count] run pass#2 binary (no plugin): ${BIN}"
"${BIN}" >"${RUNLOG}" 2>&1 || true

if rg -q "^\\[submit-args\\]" "${RUNLOG}"; then
    {
        echo ""
        echo "# pto2_submit_mixed_task args"
        rg "^\\[submit-args\\]" "${RUNLOG}"
    } >> "${PHASE_OUT}"
fi

# Reformat output into per-submit breakdown:
# [submit-args] ...
# submit_total: N
#   alloc: N
#   ...
#   others: submit_total-(alloc+sync+lookup+insert+params+fanin)
if rg -q "^session_id,phase_id,cpu_id,insn_count$" "${PHASE_OUT}" && rg -q "^\\[submit-args\\]" "${RUNLOG}"; then
    python3 - "${PHASE_OUT}" "${RUNLOG}" <<'PY'
import re
import sys
from collections import defaultdict

outfile, runlog = sys.argv[1], sys.argv[2]
phase_name = {
    0: "submit_total",
    1: "alloc",
    2: "sync",
    3: "lookup",
    4: "insert",
    5: "params",
    6: "fanin",
}
phase_order = [1, 2, 3, 4, 5, 6]

with open(outfile, "r", encoding="utf-8", errors="ignore") as f:
    raw_lines = [ln.rstrip("\n") for ln in f]
with open(runlog, "r", encoding="utf-8", errors="ignore") as f:
    submit_args = [ln.rstrip("\n") for ln in f if ln.startswith("[submit-args]")]

phase_rows = []
csv_started = False
for ln in raw_lines:
    if ln.strip() == "session_id,phase_id,cpu_id,insn_count":
        csv_started = True
        continue
    if not csv_started:
        continue
    m = re.match(r"^\s*(\d+),(\d+),(\d+),(\d+)\s*$", ln)
    if m:
        _sid, pid, _cpu, insn = m.groups()
        phase_rows.append((int(pid), int(insn)))

phase_values = defaultdict(list)
for pid, insn in phase_rows:
    if pid in phase_name:
        phase_values[pid].append(insn)

n = min(len(submit_args), len(phase_values[0]))
for pid in phase_order:
    n = min(n, len(phase_values[pid]))

formatted = []
grouped = defaultdict(list)
for i in range(n):
    total = phase_values[0][i]
    vals = {pid: phase_values[pid][i] for pid in phase_order}
    others = total - sum(vals.values())
    tag = submit_args[i]
    formatted.append(tag)
    formatted.append(f"submit_total: {total}")
    for pid in phase_order:
        formatted.append(f"    {phase_name[pid]}: {vals[pid]}")
    formatted.append(f"    others: {others}")
    grouped[tag].append({
        "submit_total": total,
        "alloc": vals[1],
        "sync": vals[2],
        "lookup": vals[3],
        "insert": vals[4],
        "params": vals[5],
        "fanin": vals[6],
        "others": others,
    })

def stat_triplet(values):
    if not values:
        return (0, 0, 0.0)
    mx = max(values)
    mn = min(values)
    avg = sum(values) / len(values)
    return (mx, mn, avg)

if grouped:
    formatted.append("")
    formatted.append("# grouped stats by submit-args")
    metric_order = ["submit_total", "alloc", "sync", "lookup", "insert", "params", "fanin", "others"]
    for tag in sorted(grouped.keys()):
        rows = grouped[tag]
        formatted.append(tag)
        for m in metric_order:
            vals = [r[m] for r in rows]
            mx, mn, avg = stat_triplet(vals)
            formatted.append(f"    {m}: max={mx} min={mn} avg={avg:.2f}")

summary_lines = []
if n > 0 and phase_values[0]:
    totals = [phase_values[0][i] for i in range(n)]
    sum_t = sum(totals)
    summary_lines = [
        "# pto2_submit_mixed_task 指令统计（submit_total 外层区间，每项为一次提交）",
        f"pto2_submit_mixed_task 指令总数: {sum_t}",
        f"任务数（提交次数）: {n}",
        f"平均每任务指令数: {sum_t / n:.2f}",
        f"各任务指令数最小值: {min(totals)}",
        f"各任务指令数最大值: {max(totals)}",
        "",
    ]
formatted = summary_lines + formatted

with open(outfile, "w", encoding="utf-8") as f:
    f.write("\n".join(formatted))
    if formatted:
        f.write("\n")
PY
fi

# build_graph 指令数行放在文件最前；其后为 Python 写入的 pto2_submit_mixed_task 汇总与逐条明细
BG_INSN=""
if [[ -f "${BG_OUT}" ]]; then
    BG_INSN="$(sed -n 's/^QEMU_TCG between_markers_insns: \([0-9][0-9]*\).*/\1/p' "${BG_OUT}" | head -1)"
fi
if [[ -n "${BG_INSN}" ]]; then
    _ph_tmp="$(mktemp)"
    {
        echo "build_graph 指令数: ${BG_INSN}"
        echo ""
        cat "${PHASE_OUT}"
    } > "${_ph_tmp}"
    mv -f "${_ph_tmp}" "${PHASE_OUT}"
fi

{
    echo "# === build_graph (orr x17 start / orr x18 end; needs -DPTO2_INSTR_COUNT_BUILD_GRAPH_ENABLE=ON at build) ==="
    if [[ -f "${BG_OUT}" ]]; then
        cat "${BG_OUT}"
    else
        echo "# (missing ${BG_OUT}: pass#0 did not produce outfile)"
    fi
    echo ""
    echo "# === pto2_submit_mixed_task / phase markers ==="
    if [[ -f "${PHASE_OUT}" ]]; then
        cat "${PHASE_OUT}"
    else
        echo "# (missing ${PHASE_OUT})"
    fi
} > "${OUTFILE}"

echo "[marker-count] result file: ${OUTFILE}"
sed -n '1,24p' "${OUTFILE}"

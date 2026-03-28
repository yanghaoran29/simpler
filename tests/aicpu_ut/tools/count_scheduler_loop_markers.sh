#!/usr/bin/env bash
# Count dynamic guest instructions per Scheduler loop iteration (resolve_and_dispatch_pto2 main while)
# using QEMU TCG plugin + orr x23/x23/x23 → orr x24/x24/x24 (see aicpu_executor.cpp when
# -DPTO2_INSTR_COUNT_SCHEDULER_LOOP_ENABLE=ON).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_TESTS_SH="${UT_DIR}/run_tests.sh"
PLUGIN_SO="${UT_DIR}/plugins/libinsn_count.so"
QEMU_BIN="${QEMU_BIN:-"$("${UT_DIR}/tools/resolve_plugin_qemu.sh" "${UT_DIR}")"}"
LOG_DIR="${UT_DIR}/outputs/log"

TEST_NAME="${TEST_NAME:-test_batch_paged_attention}"
TEST_IDX="${TEST_IDX:-0}"
THREAD_MODE="${THREAD_MODE:-concurrent}"

# orr x23, x23, x23  -> 0xaa1702f7 ; orr x24, x24, x24 -> 0xaa180318 (aarch64-linux-gnu as)
MARKER_LOOP_START="${MARKER_LOOP_START:-0xaa1702f7}"
MARKER_LOOP_END="${MARKER_LOOP_END:-0xaa180318}"

usage() {
    cat <<'EOF'
Usage:
  ./tools/count_scheduler_loop_markers.sh [options]

Options:
  --test <name>            Test name (default: test_batch_paged_attention)
  --idx <n>                Test index (default: 0)
  --thread-mode <mode>     concurrent|orch|sched (default: concurrent)
  --marker-start <hex>     Loop start marker encoding (default: 0xaa1702f7)
  --marker-end <hex>       Loop end marker encoding (default: 0xaa180318)
  --qemu-bin <path>        qemu-aarch64 path
  --help                   Show this help

Environment:
  TEST_NAME, TEST_IDX, THREAD_MODE, MARKER_LOOP_START, MARKER_LOOP_END, QEMU_BIN
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test) TEST_NAME="$2"; shift 2 ;;
        --idx) TEST_IDX="$2"; shift 2 ;;
        --thread-mode) THREAD_MODE="$2"; shift 2 ;;
        --marker-start) MARKER_LOOP_START="$2"; shift 2 ;;
        --marker-end) MARKER_LOOP_END="$2"; shift 2 ;;
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
    echo "Build: make -C \"${UT_DIR}/plugins\"" >&2
    exit 1
fi
if [[ ! -x "${QEMU_BIN}" ]]; then
    echo "qemu-aarch64 not found/executable: ${QEMU_BIN}" >&2
    exit 1
fi

echo "[sched-loop-marker] build binary (PTO2_INSTR_COUNT_SCHEDULER_LOOP_ENABLE=1) ..."
build_args=(--test "${TEST_NAME}" --idx "${TEST_IDX}" --build-only)
if [[ "${THREAD_MODE}" == "orch" ]]; then
    build_args+=(--orch)
elif [[ "${THREAD_MODE}" == "sched" ]]; then
    build_args+=(--sched)
fi
PTO2_INSTR_COUNT_SCHEDULER_LOOP_ENABLE=1 "${RUN_TESTS_SH}" "${build_args[@]}"

BIN_DIR="${UT_DIR}/build/bin"
case "${TEST_NAME}" in
    test_batch_paged_attention)
        case "${THREAD_MODE}" in
            concurrent) BIN="${BIN_DIR}/test_concurrent_${TEST_IDX}" ;;
            orch) BIN="${BIN_DIR}/test_orch_only_${TEST_IDX}" ;;
            sched) BIN="${BIN_DIR}/test_sched_prof_only_${TEST_IDX}" ;;
        esac ;;
    test_alt)
        case "${THREAD_MODE}" in
            concurrent) BIN="${BIN_DIR}/test_alt_concurrent_${TEST_IDX}" ;;
            orch) BIN="${BIN_DIR}/test_alt_orch_only_${TEST_IDX}" ;;
            sched) BIN="${BIN_DIR}/test_alt_sched_prof_only_${TEST_IDX}" ;;
        esac ;;
    test_bgemm)
        case "${THREAD_MODE}" in
            concurrent) BIN="${BIN_DIR}/test_bgemm_concurrent_${TEST_IDX}" ;;
            orch) BIN="${BIN_DIR}/test_bgemm_orch_only_${TEST_IDX}" ;;
            sched) BIN="${BIN_DIR}/test_bgemm_sched_prof_only_${TEST_IDX}" ;;
        esac ;;
    test_pau|test_paged_attention_unroll)
        case "${THREAD_MODE}" in
            concurrent) BIN="${BIN_DIR}/test_pau_concurrent_${TEST_IDX}" ;;
            orch) BIN="${BIN_DIR}/test_pau_orch_only_${TEST_IDX}" ;;
            sched) BIN="${BIN_DIR}/test_pau_sched_prof_only_${TEST_IDX}" ;;
        esac ;;
    test_throughput)
        case "${THREAD_MODE}" in
            concurrent) BIN="${BIN_DIR}/test_throughput_concurrent_${TEST_IDX}" ;;
            orch) BIN="${BIN_DIR}/test_throughput_orch_only_${TEST_IDX}" ;;
            sched) BIN="${BIN_DIR}/test_throughput_sched_prof_only_${TEST_IDX}" ;;
        esac ;;
    test_latency)
        case "${THREAD_MODE}" in
            concurrent) BIN="${BIN_DIR}/test_latency_concurrent_${TEST_IDX}" ;;
            orch) BIN="${BIN_DIR}/test_latency_orch_only_${TEST_IDX}" ;;
            sched) BIN="${BIN_DIR}/test_latency_sched_prof_only_${TEST_IDX}" ;;
        esac ;;
    test_paged_attention)
        BIN="${BIN_DIR}/test_pa_concurrent_${TEST_IDX}"
        ;;
    *)
        echo "Unsupported test for auto binary mapping: ${TEST_NAME}" >&2
        exit 1
        ;;
esac

if [[ ! -x "${BIN}" ]]; then
    echo "Built binary not found: ${BIN}" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOOP_RAW="${LOG_DIR}/${TEST_NAME}_${TEST_IDX}_scheduler_loop_markers_${TS}.txt"
OUTFILE="${LOG_DIR}/${TEST_NAME}_${TEST_IDX}_scheduler_loop_summary_${TS}.txt"
GUEST_LOG="${LOG_DIR}/${TEST_NAME}_${TEST_IDX}_scheduler_guest_${TS}.log"

echo "[sched-loop-marker] QEMU pass (markers ${MARKER_LOOP_START} → ${MARKER_LOOP_END}): ${BIN}"
if [[ "$(uname -m 2>/dev/null)" == "aarch64" ]]; then
    echo "[sched-loop-marker] note: host is aarch64; qemu-aarch64+TCG plugin may SIGTRAP; try matching QEMU_BIN to plugin build or x86_64 user-mode." >&2
fi
set +e
"${QEMU_BIN}" -plugin "file=${PLUGIN_SO},outfile=${LOOP_RAW},markers=1,marker_start=${MARKER_LOOP_START},marker_end=${MARKER_LOOP_END}" "${BIN}" >"${GUEST_LOG}" 2>&1
_qemu_rc=$?
set -e
if [[ ${_qemu_rc} -ne 0 ]] || [[ ! -s "${LOOP_RAW}" ]]; then
    _sig_note=""
    if [[ ${_qemu_rc} -gt 128 && ${_qemu_rc} -lt 256 ]]; then
        _s=$((_qemu_rc - 128))
        _sig_note=" (status 128+${_s}"
        [[ ${_s} -eq 5 ]] && _sig_note+=", SIGTRAP"
        _sig_note+=")"
    fi
    echo "[sched-loop-marker] WARN: QEMU rc=${_qemu_rc}${_sig_note}; plugin outfile missing or empty." >&2
    echo "[sched-loop-marker]       'double free or corruption' in the log = QEMU/plugin heap bug — almost always libinsn_count.so built for a *different* QEMU tree than QEMU_BIN." >&2
    echo "[sched-loop-marker]       fix: make -C \"${UT_DIR}/plugins\" QEMU_BUILD_DIR=<dir of the qemu that built ${QEMU_BIN}>; then point QEMU_BIN to that qemu-aarch64." >&2
    if [[ -s "${GUEST_LOG}" ]]; then
        echo "[sched-loop-marker] --- captured QEMU stderr/stdout (tail) ---" >&2
        tail -n 25 "${GUEST_LOG}" | sed 's/^/[qemu] /' >&2 || true
    fi
    if [[ ! -f "${LOOP_RAW}" ]] || [[ ! -s "${LOOP_RAW}" ]]; then
        {
            echo "# QEMU pass failed: rc=${_qemu_rc}${_sig_note}"
            echo "# If log shows 'double free or corruption', rebuild plugin against the same QEMU sources as QEMU_BIN (see tests/aicpu_ut/plugins/Makefile, QEMU_BUILD_DIR)."
            echo "# --- QEMU stderr/stdout tail ---"
            tail -n 40 "${GUEST_LOG}" 2>/dev/null || true
        } >"${LOOP_RAW}"
    fi
fi

python3 - "${LOOP_RAW}" "${OUTFILE}" "${TEST_NAME}" "${TEST_IDX}" "${THREAD_MODE}" "${GUEST_LOG}" <<'PY'
import os
import re
import sys
from collections import defaultdict

raw_path, out_path, test_name, test_idx, thread_mode, guest_log_path = sys.argv[1:7]


def parse_rdp2_invoke_guest_log(path):
    if not path or not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "[rdp2-invoke-stats]" not in line:
                continue
            m = re.search(r"total=(\d+)", line)
            if not m:
                return None
            total = int(m.group(1))
            per = {}
            for mm in re.finditer(r"\bt(\d+)=(\d+)", line):
                per[int(mm.group(1))] = int(mm.group(2))
            return total, per
    return None

rows = []
with open(raw_path, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.strip()
        if re.match(r"^\d+,\d+,\d+,\d+$", line):
            parts = line.split(",")
            rows.append(
                {
                    "session_id": int(parts[0]),
                    "phase_id": int(parts[1]),
                    "cpu_id": int(parts[2]),
                    "insn_count": int(parts[3]),
                }
            )

summary_line = ""
with open(raw_path, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        if line.startswith("QEMU_TCG between_markers_insns:"):
            summary_line = line.strip()
            break

by_cpu = defaultdict(list)
for r in rows:
    by_cpu[r["cpu_id"]].append(r["insn_count"])

counts = [r["insn_count"] for r in rows]
n = len(counts)
total = sum(counts)

lines = [
    f"# Scheduler 主循环指令统计 (orr x23 迭代开始 / orr x24 迭代结束)",
    f"# test={test_name} idx={test_idx} thread_mode={thread_mode}",
    "",
]
inv = parse_rdp2_invoke_guest_log(guest_log_path)
if inv:
    tot, per = inv
    lines += [
        "# resolve_and_dispatch_pto2 调用统计（函数入口次数；thread_idx 为 AicpuExecutor 调度线程索引）",
        f"resolve_and_dispatch_pto2 函数调用总次数: {tot}",
        "resolve_and_dispatch_pto2 各 thread_idx 调用次数:",
    ]
    for tid in sorted(per.keys()):
        lines.append(f"  thread_idx={tid}: {per[tid]}")
    lines.append("")
else:
    lines += [
        "# resolve_and_dispatch_pto2 调用统计: 未在 guest 日志中解析到 [rdp2-invoke-stats]（见 scheduler_guest_*.log；需本选项构建二进制）",
        "",
    ]

if summary_line:
    lines.append(summary_line)
    lines.append("")

lines += [
    f"scheduler_loop 迭代次数（会话数）: {n}",
    f"scheduler_loop 动态指令总数（所有 vCPU 合计）: {total}",
]

if n > 0:
    lines += [
        f"每迭代指令数 最小值: {min(counts)}",
        f"每迭代指令数 最大值: {max(counts)}",
        f"每迭代指令数 平均值: {total / n:.2f}",
        "",
        "# 按 cpu_id（多 Scheduler 线程时各线程独立会话）",
    ]
    for cid in sorted(by_cpu.keys()):
        cc = by_cpu[cid]
        s = sum(cc)
        lines.append(
            f"  cpu_id={cid}: 迭代数={len(cc)} 合计指令={s} "
            f"min={min(cc)} max={max(cc)} avg={s / len(cc):.2f}"
        )
else:
    lines += ["", "(无 session 行：请确认已用 -DPTO2_INSTR_COUNT_SCHEDULER_LOOP_ENABLE=ON 构建，且二进制在 QEMU 下执行了 resolve_and_dispatch_pto2 调度循环；SIMULATE 用例见 run_tests 说明)"]

lines += ["", "# --- 原始插件输出（含逐条 session） ---", ""]
with open(raw_path, "r", encoding="utf-8", errors="replace") as f:
    lines.append(f.read().rstrip("\n"))

text = "\n".join(lines) + "\n"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(text)

print(text[:2000])
if len(text) > 2000:
    print("... [truncated in stdout; see full file]")
PY

echo "[sched-loop-marker] summary: ${OUTFILE}"
echo "[sched-loop-marker] raw plugin log: ${LOOP_RAW}"
echo "[sched-loop-marker] guest stdout/stderr: ${GUEST_LOG}"

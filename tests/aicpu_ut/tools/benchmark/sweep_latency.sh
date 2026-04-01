#!/usr/bin/env bash
# sweep_latency.sh — 参数扫描脚本：对 test_latency idx=1 (aic/aiv alternate) 跑完全部参数组合
#
# 全部参数组合（均会测到，每组合 RUNS 次）：
#   Group X : (chain-num, chain-length) = (1, 128)                           — 1 组
#   Group Y : (1, 128), (1, 256), (1, 512), (1, 1024), (1, 2048), (1, 4096), (1, 8192), (1, 16384) — 8 组
#   合计 9 组参数，每组 RUNS 次（默认 10）。
#
# 用法：
#   bash tools/sweep_latency.sh
# 输出：outputs/sweep_latency[(_p1|_p2)[_orch|_sched]]/latency_<label>_run<n>.log（每样例 RUNS 次，原始 log 全部保留）
# 任意一次测试失败立即退出。每样例至少 10 次取平均：RUNS=10（默认），可由环境变量 RUNS 覆盖。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEARCH_DIR="$SCRIPT_DIR"
while [[ "$SEARCH_DIR" != "/" && ! -f "$SEARCH_DIR/run_tests.sh" ]]; do
    SEARCH_DIR="$(dirname "$SEARCH_DIR")"
done
if [[ ! -f "$SEARCH_DIR/run_tests.sh" ]]; then
    echo "Error: cannot locate run_tests.sh from $SCRIPT_DIR" >&2
    exit 1
fi
AICPU_UT_DIR="$SEARCH_DIR"
if [[ "${1:-}" = "--profiling" && -n "${2:-}" ]]; then
    PROFILING_MODE="$2"; shift 2
else
    PROFILING_MODE=${PROFILING_MODE:-2}
fi
THREAD_MODE=${THREAD_MODE:-concurrent}
SUFFIX="_p${PROFILING_MODE}"
case "$THREAD_MODE" in orch) SUFFIX="${SUFFIX}_orch" ;; sched) SUFFIX="${SUFFIX}_sched" ;; esac
LOG_DIR="${AICPU_UT_DIR}/outputs/sweep_latency${SUFFIX}"
mkdir -p "$LOG_DIR"

RUNS=${RUNS:-10}

# ─── 默认参数 ─────────────────────────────────────────────────────────────────
DEF_X=1     # --chain-num
DEF_Y=128   # --chain-length

# ─── 单次运行 ─────────────────────────────────────────────────────────────────
run_one() {
    local X=$1 Y=$2 label=$3 run_idx=$4
    local log_file="${LOG_DIR}/latency_${label}_run${run_idx}.log"
    printf "  [%d/%d] → %s\n" "$run_idx" "$RUNS" "$(basename "$log_file")"
    bash "${AICPU_UT_DIR}/run_tests.sh" \
        --test test_latency \
        --chain-num    "$X" \
        --chain-length "$Y" \
        --profiling "$PROFILING_MODE" \
        $( [ "$THREAD_MODE" = "orch" ] && echo --orch; [ "$THREAD_MODE" = "sched" ] && echo --sched ) \
        --idx 1 \
        > "$log_file" 2>&1 || true
    if ! grep -q "OVERALL: PASSED" "$log_file"; then
        echo "  FAILED — see: $log_file" >&2
        exit 1
    fi
}

# ─── 一组参数跑 RUNS 次 ───────────────────────────────────────────────────────
run_group() {
    local X=$1 Y=$2 label=$3
    echo ""
    echo "  X=$X  Y=$Y"
    for run_idx in $(seq 1 "$RUNS"); do
        run_one "$X" "$Y" "$label" "$run_idx"
    done
}

# ─── Group X ──────────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━  Group X  (vary chain-num, Y=${DEF_Y})  ━━━━━━━━━━━━━━"
for X in 1; do
    run_group "$X" "$DEF_Y" "X${X}_Y${DEF_Y}"
done

# ─── Group Y ──────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━  Group Y  (vary chain-length, X=${DEF_X})  ━━━━━━━━━━━━━━"
for Y in 128 256 512 1024 2048 4096 8192 16384; do
    run_group "$DEF_X" "$Y" "X${DEF_X}_Y${Y}"
done

# ─── 汇总 ─────────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
total=$(( (1 + 8) * RUNS ))
echo "  All done — ${total} runs passed."
echo "  原始 log 已保留: $LOG_DIR (每样例 ${RUNS} 次)"

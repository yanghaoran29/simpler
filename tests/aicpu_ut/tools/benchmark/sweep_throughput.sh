#!/usr/bin/env bash
# sweep_throughput.sh — 参数扫描脚本：对 test_throughput idx=0 跑完全部参数组合
#
# 全部参数组合（均会测到，每组合 RUNS 次）：
#   Group YZ : (layer-num, dependency, overlap, layer0-task-num) = (2,2,0,1024),(2,2,1,1024),(2,4,0,1024),(2,4,2,1024),(2,6,0,1024),(2,6,2,1024),(2,6,4,1024),(2,8,4,1024) — 8 组
#   Group W  : (2,6,2,1024),(2,6,2,2048),(2,6,2,3072),(2,6,2,4096),(2,6,2,5120),(2,6,2,6144) — 6 组
#   合计 14 组参数，每组 RUNS 次（默认 10）。
#
# 用法：
#   bash tools/sweep_throughput.sh
#   bash tools/sweep_throughput.sh --profiling 1
#   THREAD_MODE=orch bash tools/sweep_throughput.sh --profiling 1
#   THREAD_MODE=sched bash tools/sweep_throughput.sh --profiling 1
# 输出：outputs/sweep_throughput[_p1|_p2][_orch|_sched]/<label>_run<n>.log（每个样例 RUNS 次，原始 log 全部保留）
# 任意一次测试失败立即退出。
# RUNS=10（默认，至少 10 次取平均），可由环境变量覆盖。

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
SUFFIX=""
case "$PROFILING_MODE" in 0) SUFFIX="_p0" ;; 1) SUFFIX="_p1" ;; 2) SUFFIX="_p2" ;; *) SUFFIX="_p${PROFILING_MODE}" ;; esac
case "$THREAD_MODE" in orch) SUFFIX="${SUFFIX}_orch" ;; sched) SUFFIX="${SUFFIX}_sched" ;; esac
LOG_DIR="${AICPU_UT_DIR}/outputs/sweep_throughput${SUFFIX}"
mkdir -p "$LOG_DIR"

RUNS=${RUNS:-10}

# ─── 默认参数 ─────────────────────────────────────────────────────────────────
DEF_X=2    # --layer-num (固定)
DEF_Y=6    # --dependency (Group W 用)
DEF_Z=2    # --overlap (Group W 用)
DEF_W=1024 # --layer0-task-num (Group YZ 用)

# ─── 单次运行 ─────────────────────────────────────────────────────────────────
run_one() {
    local X=$1 Y=$2 Z=$3 W=$4 label=$5 run_idx=$6
    local log_file="${LOG_DIR}/${label}_run${run_idx}.log"
    printf "  [%d/%d] → %s\n" "$run_idx" "$RUNS" "$(basename "$log_file")"
    bash "${AICPU_UT_DIR}/run_tests.sh" \
        --test test_throughput \
        --layer-num    "$X" \
        --dependency   "$Y" \
        --overlap      "$Z" \
        --layer0-task-num "$W" \
        --profiling "$PROFILING_MODE" \
        --idx 0 \
        $( [ "$THREAD_MODE" = "orch" ] && echo --orch; [ "$THREAD_MODE" = "sched" ] && echo --sched ) \
        > "$log_file" 2>&1 || true
    if ! grep -q "OVERALL: PASSED" "$log_file"; then
        echo "  FAILED — see: $log_file" >&2
        exit 1
    fi
}

# ─── 一组参数跑 RUNS 次 ───────────────────────────────────────────────────────
run_group() {
    local X=$1 Y=$2 Z=$3 W=$4 label=$5
    echo ""
    echo "  X=$X  Y=$Y  Z=$Z  W=$W"
    for run_idx in $(seq 1 "$RUNS"); do
        run_one "$X" "$Y" "$Z" "$W" "$label" "$run_idx"
    done
}

# ─── Group YZ ─────────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━  Group YZ  (vary deps/overlap, X=${DEF_X} W=${DEF_W})  ━━━━━━━━━━━━━━"
for pair in "2 0" "2 1" "4 0" "4 2" "6 0" "6 2" "6 4" "8 4"; do
    Y=${pair% *}
    Z=${pair#* }
    run_group "$DEF_X" "$Y" "$Z" "$DEF_W" \
        "grpYZ_n${DEF_X}_D${Y}_O${Z}_W${DEF_W}"
done

# ─── Group W ──────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━  Group W  (vary layer0, X=${DEF_X} D=${DEF_Y} O=${DEF_Z})  ━━━━━━━━━━━━━━"
for W in 1024 2048 3072 4096 5120 6144; do
    run_group "$DEF_X" "$DEF_Y" "$DEF_Z" "$W" \
        "grpW_n${DEF_X}_D${DEF_Y}_O${DEF_Z}_W${W}"
done

# ─── 汇总 ─────────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
total=$(( (8 + 6) * RUNS ))
echo "  All done — ${total} runs passed."
echo "  原始 log 已保留: $LOG_DIR (每样例 ${RUNS} 次)"

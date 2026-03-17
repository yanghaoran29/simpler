#!/usr/bin/env bash
# sweep_throughput.sh — 参数扫描脚本：对 test_throughput idx=0 分两组变参各跑 5 次
#
# 固定 X=2(layers)
# 分组：
#   Group YZ : (Y,Z) ∈ {(2,0),(2,1),(4,0),(4,2),(6,0),(6,2),(6,4),(8,4)}, X=2  W=1024
#   Group W  : W ∈ {1024,2048,3072,4096,5120,6144},  X=2  Y=6  Z=2
#
# 用法：
#   bash tools/sweep_throughput.sh
#   PROFILING_MODE=1 bash tools/sweep_throughput.sh   # --profiling 1 → outputs/sweep_throughput_p1/
#   PROFILING_MODE=2 bash tools/sweep_throughput.sh   # --profiling 2 → outputs/sweep_throughput_p2/
# 输出：outputs/sweep_throughput[_p1|_p2]/<label>_run<n>.log
# 任意一次测试失败立即退出。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILING_MODE=${PROFILING_MODE:-2}
SUFFIX=""
case "$PROFILING_MODE" in 0) SUFFIX="_p0" ;; 1) SUFFIX="_p1" ;; 2) SUFFIX="_p2" ;; *) SUFFIX="_p${PROFILING_MODE}" ;; esac
LOG_DIR="${SCRIPT_DIR}/../outputs/sweep_throughput${SUFFIX}"
mkdir -p "$LOG_DIR"

RUNS=5

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
    bash "${SCRIPT_DIR}/../run_tests.sh" \
        --test test_throughput \
        --layer-num    "$X" \
        --dependency   "$Y" \
        --overlap      "$Z" \
        --layer0-task-num "$W" \
        --profiling "$PROFILING_MODE" \
        --idx 0 \
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
echo "  Logs: $LOG_DIR"

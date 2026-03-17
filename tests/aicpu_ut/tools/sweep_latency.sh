#!/usr/bin/env bash
# sweep_latency.sh — 参数扫描脚本：对 test_latency idx=1 (aic/aiv alternate) 分两组变参各跑 5 次
#
# 默认值：X=64(chain-num), Y=64(chain-length)
# 分组：
#   Group X : X ∈ {16,32,48,64,96,128,192},  Y=64
#   Group Y : Y ∈ {4,8,16,32,64,128},         X=64
#
# 用法：
#   bash tools/sweep_latency.sh              # 默认 --profiling 2，输出 sweep_latency/
#   PROFILING_MODE=1 bash tools/sweep_latency.sh   # --profiling 1，输出 sweep_latency_p1/
#   PROFILING_MODE=2 bash tools/sweep_latency.sh   # --profiling 2，输出 sweep_latency_p2/
# 输出：outputs/sweep_latency[_p1|_p2]/latency_<label>_run<n>.log
# 任意一次测试失败立即退出。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILING_MODE=${PROFILING_MODE:-2}
SUFFIX=""
case "$PROFILING_MODE" in 0) SUFFIX="_p0" ;; 1) SUFFIX="_p1" ;; 2) SUFFIX="_p2" ;; *) SUFFIX="_p${PROFILING_MODE}" ;; esac
LOG_DIR="${SCRIPT_DIR}/../outputs/sweep_latency${SUFFIX}"
mkdir -p "$LOG_DIR"

RUNS=5

# ─── 默认参数 ─────────────────────────────────────────────────────────────────
DEF_X=64   # --chain-num
DEF_Y=64   # --chain-length

# ─── 单次运行 ─────────────────────────────────────────────────────────────────
run_one() {
    local X=$1 Y=$2 label=$3 run_idx=$4
    local log_file="${LOG_DIR}/latency_${label}_run${run_idx}.log"
    printf "  [%d/%d] → %s\n" "$run_idx" "$RUNS" "$(basename "$log_file")"
    bash "${SCRIPT_DIR}/../run_tests.sh" \
        --test test_latency \
        --chain-num    "$X" \
        --chain-length "$Y" \
        --profiling "$PROFILING_MODE" \
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
for X in 16 32 48 64 96 128 192; do
    run_group "$X" "$DEF_Y" "X${X}_Y${DEF_Y}"
done

# ─── Group Y ──────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━  Group Y  (vary chain-length, X=${DEF_X})  ━━━━━━━━━━━━━━"
for Y in 4 8 16 32 64 128; do
    run_group "$DEF_X" "$Y" "X${DEF_X}_Y${Y}"
done

# ─── 汇总 ─────────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
total=$(( (7 + 6) * RUNS ))
echo "  All done — ${total} runs passed."
echo "  Logs: $LOG_DIR"

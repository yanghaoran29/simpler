#!/usr/bin/env bash
# compare_profiling.sh — 用少量用例对比 --profiling 1 与 --profiling 2 的性能
#
# 用法：
#   bash tools/compare_profiling.sh
# 各跑 2 次 test_latency（X=64,Y=64）和 2 次 test_throughput（n=2,D=6,O=2,W=1024），
# 解析 Scheduler Total (us) 与 Orchestrator 时间，输出 profiling 1 vs 2 的差距。
#
# 依赖：run_tests.sh 支持 --profiling 1 / --profiling 2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AICPU_UT_DIR="${SCRIPT_DIR}/.."
OUT_DIR="${SCRIPT_DIR}/../outputs/compare_profiling"
RUNS=2

mkdir -p "$OUT_DIR"

# 从单次运行 log 中解析：Scheduler SUM 行最后一列 (Total us)；Orchestrator 总时间 (us)
parse_log() {
    local log="$1"
    local sched_us="" orch_us=""
    while IFS= read -r line; do
        # SUM 行最后一列为 Total (us)，如 "  SUM  1316  4096  3.112  3339.0"
        if [[ "$line" =~ SUM ]]; then
            sched_us=$(echo "$line" | awk '{print $NF}')
            [[ "$sched_us" =~ ^[0-9.]+$ ]] || sched_us=""
        fi
        # Orchestrator run time: 123.456us  (profiling 1)
        if [[ "$line" =~ Orchestrator[[:space:]]run[[:space:]]time:[[:space:]]+([0-9.]+)us ]]; then
            orch_us="${BASH_REMATCH[1]}"
        fi
        # Orchestrator Profiling: N tasks, total=123.456us  (profiling 2)
        if [[ "$line" =~ Orchestrator[[:space:]]Profiling:.*total=([0-9.]+)us ]]; then
            orch_us="${BASH_REMATCH[1]}"
        fi
    done < "$log"
    echo "${sched_us:-nan} ${orch_us:-nan}"
}

echo "=============================================="
echo "  Profiling 1 vs 2 性能对比（各 ${RUNS} 次）"
echo "=============================================="
echo ""

# --- Latency: X=64 Y=64 ---
echo "  [1/4] test_latency (chain-num=64, chain-length=64) — profiling 1"
for i in $(seq 1 "$RUNS"); do
    bash "${AICPU_UT_DIR}/run_tests.sh" --test test_latency --chain-num 64 --chain-length 64 --profiling 1 --idx 1 \
        > "${OUT_DIR}/latency_p1_run${i}.log" 2>&1 || true
done
echo "  [2/4] test_latency (chain-num=64, chain-length=64) — profiling 2"
for i in $(seq 1 "$RUNS"); do
    bash "${AICPU_UT_DIR}/run_tests.sh" --test test_latency --chain-num 64 --chain-length 64 --profiling 2 --idx 1 \
        > "${OUT_DIR}/latency_p2_run${i}.log" 2>&1 || true
done

# --- Throughput: n=2 D=6 O=2 W=1024 ---
echo "  [3/4] test_throughput (n=2,D=6,O=2,W=1024) — profiling 1"
for i in $(seq 1 "$RUNS"); do
    bash "${AICPU_UT_DIR}/run_tests.sh" --test test_throughput --layer-num 2 --dependency 6 --overlap 2 --layer0-task-num 1024 --profiling 1 --idx 0 \
        > "${OUT_DIR}/throughput_p1_run${i}.log" 2>&1 || true
done
echo "  [4/4] test_throughput (n=2,D=6,O=2,W=1024) — profiling 2"
for i in $(seq 1 "$RUNS"); do
    bash "${AICPU_UT_DIR}/run_tests.sh" --test test_throughput --layer-num 2 --dependency 6 --overlap 2 --layer0-task-num 1024 --profiling 2 --idx 0 \
        > "${OUT_DIR}/throughput_p2_run${i}.log" 2>&1 || true
done

echo ""
echo "=============================================="
echo "  解析结果（Scheduler Total us / Orchestrator us）"
echo "=============================================="

collect_avg() {
    local prefix="$1"
    local p="$2"
    local sched_sum=0 sched_n=0 orch_sum=0 orch_n=0
    for f in "${OUT_DIR}/${prefix}_p${p}_run"*.log; do
        [ -f "$f" ] || continue
        read -r s u <<< "$(parse_log "$f")"
        if [[ "$s" != "nan" && "$s" != "" ]]; then
            sched_sum=$(echo "$sched_sum + $s" | bc)
            sched_n=$((sched_n + 1))
        fi
        if [[ "$u" != "nan" && "$u" != "" ]]; then
            orch_sum=$(echo "$orch_sum + $u" | bc)
            orch_n=$((orch_n + 1))
        fi
    done
    local savg="N/A" oavg="N/A"
    [ "$sched_n" -gt 0 ] && savg=$(echo "scale=2; $sched_sum / $sched_n" | bc)
    [ "$orch_n" -gt 0 ] && oavg=$(echo "scale=2; $orch_sum / $orch_n" | bc)
    echo "$savg $oavg"
}

# 输出表格
printf "\n  %-24s %12s %12s %12s %12s\n" "Case" "P1 Sched(us)" "P2 Sched(us)" "P1 Orch(us)" "P2 Orch(us)"
printf "  %s\n" "$(printf '%.0s-' {1..76})"

for label in "latency" "throughput"; do
    read -r s1 o1 <<< "$(collect_avg "$label" 1)"
    read -r s2 o2 <<< "$(collect_avg "$label" 2)"
    printf "  %-24s %12s %12s %12s %12s\n" "$label" "$s1" "$s2" "$o1" "$o2"
done

# 计算并打印性能差距（P2 相对 P1 的百分比，P2 因插桩更多通常更慢）
echo ""
echo "  性能差距（Profiling 2 相对 Profiling 1）："
echo "  - Scheduler: P2 比 P1 多出的时间比例 = (P2-P1)/P1 * 100%"
echo "  - Orchestrator: 同上"
echo ""

for label in "latency" "throughput"; do
    read -r s1 o1 <<< "$(collect_avg "$label" 1)"
    read -r s2 o2 <<< "$(collect_avg "$label" 2)"
    gap_s="N/A" gap_o="N/A"
    if [[ "$s1" != "N/A" && "$s2" != "N/A" && "$s1" != "0" ]]; then
        gap_s=$(echo "scale=2; ($s2 - $s1) / $s1 * 100" | bc 2>/dev/null || echo "N/A")
    fi
    if [[ "$o1" != "N/A" && "$o2" != "N/A" && "$o1" != "0" ]]; then
        gap_o=$(echo "scale=2; ($o2 - $o1) / $o1 * 100" | bc 2>/dev/null || echo "N/A")
    fi
    echo "  $label:  Scheduler 差距 ${gap_s}%   Orchestrator 差距 ${gap_o}%"
done

echo ""
echo "  Logs: $OUT_DIR"
echo ""

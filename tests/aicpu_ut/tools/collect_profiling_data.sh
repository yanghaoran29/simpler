#!/usr/bin/env bash
# collect_profiling_data.sh — 为 PROFILING_REPORT.md 收集环境信息、运行对比与 sweep 全量，并输出原始数据
# 用法：bash tools/collect_profiling_data.sh
# 会跑满 sweep_latency.sh / sweep_throughput.sh 中的全部参数组合（各组合 RUNS 次），并保留所有 log。
# 输出：env.txt, compare_p1_p2.txt, raw_data.txt；sweep log 在 outputs/sweep_latency_p1/, outputs/sweep_throughput_p1/；线程模式 log 在 outputs/profiling_report/。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AICPU_UT_DIR="${SCRIPT_DIR}/.."
OUT="${AICPU_UT_DIR}/outputs/profiling_report"
SWEEP_LAT_DIR="${AICPU_UT_DIR}/outputs/sweep_latency_p1"
SWEEP_THR_DIR="${AICPU_UT_DIR}/outputs/sweep_throughput_p1"
RUNS=${RUNS:-10}
mkdir -p "$OUT"

echo "=== 0. 环境信息 ==="
env_info="${OUT}/env.txt"
{
    echo "--- lscpu ---"
    lscpu 2>/dev/null || true
    echo ""
    echo "--- /proc/cpuinfo (model name) ---"
    grep "model name" /proc/cpuinfo 2>/dev/null | head -1 || true
    echo ""
    echo "--- 内存 (free -h) ---"
    free -h
    echo ""
    echo "--- 绑核说明 (run_tests.sh 默认) ---"
    echo "ORCH_CPU=4 (orchestrator 绑核 4)"
    echo "SCHED_CPU0=8, SCHED_CPU1=9, SCHED_CPU2=10, SCHED_CPU3=11 (scheduler 线程 0~3 绑核 8~11)"
    echo "可通过环境变量覆盖: ORCH_CPU=4 SCHED_CPU0=8 ... ./run_tests.sh"
} | tee "$env_info"

echo ""
echo "=== 1. Profiling 1 vs 2 对比 (各 ${RUNS} 次，log 保留在 compare_profiling/) ==="
RUNS="$RUNS" bash "${SCRIPT_DIR}/compare_profiling.sh" 2>&1 | tee "${OUT}/compare_p1_p2.txt"

echo ""
echo "=== 2. sweep_latency --profiling 1（全部 9 组参数，每组 ${RUNS} 次，log → ${SWEEP_LAT_DIR}）==="
RUNS="$RUNS" bash "${SCRIPT_DIR}/sweep_latency.sh" --profiling 1

echo ""
echo "=== 3. sweep_throughput --profiling 1（全部 14 组参数，每组 ${RUNS} 次，log → ${SWEEP_THR_DIR}）==="
RUNS="$RUNS" bash "${SCRIPT_DIR}/sweep_throughput.sh" --profiling 1

echo ""
echo "=== 4. 三种线程模式对比（代表点: latency X=1 Y=128, throughput n=2 D=6 O=2 W=1024，各 ${RUNS} 次）==="
for mode in "" "--orch" "--sched"; do
    name="concurrent"; [ -n "$mode" ] && name="${mode#--}"
    for r in $(seq 1 "$RUNS"); do
        printf "  [%s] run %d/%d\n" "$name" "$r" "$RUNS"
        bash "${AICPU_UT_DIR}/run_tests.sh" --test test_latency --chain-num 1 --chain-length 128 --profiling 1 --idx 1 $mode \
            > "${OUT}/latency_p1_${name}_run${r}.log" 2>&1 || true
        bash "${AICPU_UT_DIR}/run_tests.sh" --test test_throughput --layer-num 2 --dependency 6 --overlap 2 --layer0-task-num 1024 --profiling 1 --idx 0 $mode \
            > "${OUT}/throughput_p1_${name}_run${r}.log" 2>&1 || true
    done
done

echo ""
echo "=== 5. 原始数据提取 ==="
raw="${OUT}/raw_data.txt"
echo "Profiling report raw data — $(date -Iseconds)" > "$raw"
echo "" >> "$raw"

parse_one() {
    local log="$1"
    local sched_us="" orch_us="" tasks=""
    while IFS= read -r line; do
        if [[ "$line" =~ SUM ]]; then
            sched_us=$(echo "$line" | awk '{print $NF}')
            [[ "$sched_us" =~ ^[0-9.]+$ ]] || sched_us=""
        fi
        if [[ "$line" =~ Orchestrator[[:space:]]run[[:space:]]time:[[:space:]]+([0-9.]+)us ]]; then
            orch_us="${BASH_REMATCH[1]}"
        fi
        if [[ "$line" =~ Orchestrator[[:space:]]Profiling:.*total=([0-9.]+)us ]]; then
            orch_us="${BASH_REMATCH[1]}"
        fi
        if [[ "$line" =~ Total[[:space:]]tasks[[:space:]]submitted:[[:space:]]+([0-9]+) ]]; then
            tasks="${BASH_REMATCH[1]}"
        fi
    done < "$log"
    echo "${sched_us:-nan} ${orch_us:-nan} ${tasks:-nan}"
}

echo "--- compare_p1_p2 (from compare_profiling output) ---" >> "$raw"
grep -E "latency|throughput|差距|Sched|Orch" "${OUT}/compare_p1_p2.txt" >> "$raw" 2>/dev/null || true
echo "" >> "$raw"

echo "--- sweep_latency_p1 全部参数组合 (sched_us orch_us tasks)，log 目录: ${SWEEP_LAT_DIR} ---" >> "$raw"
for f in "${SWEEP_LAT_DIR}"/latency_*_run*.log; do
    [ -f "$f" ] || continue
    echo "$(basename "$f"): $(parse_one "$f")" >> "$raw"
done
echo "" >> "$raw"

echo "--- sweep_throughput_p1 全部参数组合，log 目录: ${SWEEP_THR_DIR} ---" >> "$raw"
for f in "${SWEEP_THR_DIR}"/*_run*.log; do
    [ -f "$f" ] || continue
    echo "$(basename "$f"): $(parse_one "$f")" >> "$raw"
done
echo "" >> "$raw"

echo "--- thread_mode latency (concurrent/orch/sched 各 ${RUNS} 次) ---" >> "$raw"
for name in concurrent orch sched; do
    for f in "${OUT}"/latency_p1_${name}_run*.log; do
        [ -f "$f" ] || continue
        echo "$(basename "$f"): $(parse_one "$f")" >> "$raw"
    done
done
echo "" >> "$raw"

echo "--- thread_mode throughput ---" >> "$raw"
for name in concurrent orch sched; do
    for f in "${OUT}"/throughput_p1_${name}_run*.log; do
        [ -f "$f" ] || continue
        echo "$(basename "$f"): $(parse_one "$f")" >> "$raw"
    done
done

echo ""
echo "=== 6. 统计汇总（均值、标准差、最小值、最大值，用于报告体现数据波动）==="
stats_file="${OUT}/summary_stats.txt"
echo "# section|sample_id|tasks|sched_mean|sched_std|sched_min|sched_max|orch_mean|orch_std|orch_min|orch_max" > "$stats_file"
echo "# 由 collect_profiling_data.sh 根据各样例 RUNS 次独立 log 计算" >> "$stats_file"

# 对一行数字（空格分隔）计算 mean,std,min,max，输出 mean std min max
calc_stats() {
    echo "$1" | tr ' ' '\n' | awk '
        NF { n++; sum+=$1; a[n]=$1 }
        END {
            if (n<1) exit;
            mean = sum/n;
            var = 0;
            for (i=1;i<=n;i++) var += (a[i]-mean)^2;
            std = (n>1) ? sqrt(var/(n-1)) : 0;
            min = max = a[1];
            for (i=2;i<=n;i++) { if (a[i]<min) min=a[i]; if (a[i]>max) max=a[i]; }
            printf "%.2f %.2f %.2f %.2f", mean, std, min, max
        }'
}
# 从一组 log 文件计算 sched/orch 的 mean,std,min,max 并追加一行到 stats_file
stats_from_logs() {
    local file_list="$1"
    local section="$2"
    local sample_id="$3"
    local tasks="" sched_list="" orch_list=""
    for f in $file_list; do
        [[ -f "$f" ]] || continue
        read -r s o t <<< "$(parse_one "$f")"
        [[ "$s" =~ ^[0-9.]+$ ]] && sched_list="${sched_list:+$sched_list }}$s"
        [[ "$o" =~ ^[0-9.]+$ ]] && orch_list="${orch_list:+$orch_list }}$o"
        [[ "$t" =~ ^[0-9]+$ ]] && tasks="$t"
    done
    [[ -z "$sched_list" && -z "$orch_list" ]] && return
    local sched_mean sched_std sched_min sched_max orch_mean orch_std orch_min orch_max
    if [[ -n "$sched_list" ]]; then
        read -r sched_mean sched_std sched_min sched_max <<< "$(calc_stats "$sched_list")"
    else
        sched_mean=""; sched_std=""; sched_min=""; sched_max=""
    fi
    if [[ -n "$orch_list" ]]; then
        read -r orch_mean orch_std orch_min orch_max <<< "$(calc_stats "$orch_list")"
    else
        orch_mean=""; orch_std=""; orch_min=""; orch_max=""
    fi
    echo "${section}|${sample_id}|${tasks}|${sched_mean}|${sched_std}|${sched_min}|${sched_max}|${orch_mean}|${orch_std}|${orch_min}|${orch_max}" >> "$stats_file"
}

# sweep_latency：按 label 分组（latency_X1_Y128_run1.log 等）
for base in latency_X1_Y128 latency_X1_Y256 latency_X1_Y512 latency_X1_Y1024 latency_X1_Y2048 latency_X1_Y4096 latency_X1_Y8192 latency_X1_Y16383; do
    files=""
    for f in "${SWEEP_LAT_DIR}/${base}_run"*.log; do [ -f "$f" ] && files="$files $f"; done
    [[ -z "$files" ]] && continue
    sample_id="${base#latency_}"
    stats_from_logs "$files" "sweep_latency" "$sample_id"
done

# sweep_throughput：按 prefix 分组
for base in grpW_n2_D6_O2_W1024 grpW_n2_D6_O2_W2048 grpW_n2_D6_O2_W3072 grpYZ_n2_D2_O0_W1024 grpYZ_n2_D2_O1_W1024 grpYZ_n2_D4_O0_W1024 grpYZ_n2_D4_O2_W1024 grpYZ_n2_D6_O0_W1024 grpYZ_n2_D6_O2_W1024 grpYZ_n2_D6_O4_W1024 grpYZ_n2_D8_O4_W1024; do
    files=""
    for f in "${SWEEP_THR_DIR}/${base}_run"*.log; do [ -f "$f" ] && files="$files $f"; done
    [[ -z "$files" ]] && continue
    stats_from_logs "$files" "sweep_throughput" "$base"
done

# thread_mode
for name in concurrent orch sched; do
    files=""
    for f in "${OUT}"/latency_p1_${name}_run*.log; do [ -f "$f" ] && files="$files $f"; done
    [[ -n "$files" ]] && stats_from_logs "$files" "thread_latency" "$name"
    files=""
    for f in "${OUT}"/throughput_p1_${name}_run*.log; do [ -f "$f" ] && files="$files $f"; done
    [[ -n "$files" ]] && stats_from_logs "$files" "thread_throughput" "$name"
done

echo "  已写入 ${stats_file}（均值、标准差、最小、最大）"

echo "Done. 原始 log 已保留："
echo "  sweep 全部参数: $SWEEP_LAT_DIR (9 组×${RUNS} 次), $SWEEP_THR_DIR (14 组×${RUNS} 次)"
echo "  线程模式与汇总: $OUT (env.txt, compare_p1_p2.txt, raw_data.txt, latency/throughput_p1_*_run<n>.log)"

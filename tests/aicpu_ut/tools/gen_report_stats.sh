#!/usr/bin/env bash
# 从 summary_stats.txt 生成报告中 2.1 / 3.1 / 4.1 的汇总表（平均值 ± 百分比（最小值–最大值），百分比=变异系数）
# 用法：bash tools/gen_report_stats.sh
# 依赖：outputs/profiling_report/summary_stats.txt（由 collect_profiling_data.sh RUNS=10 后生成）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATS="${SCRIPT_DIR}/../outputs/profiling_report/summary_stats.txt"

if [[ ! -f "$STATS" ]]; then
    echo "Error: $STATS not found. Run: RUNS=10 bash tools/collect_profiling_data.sh" >&2
    exit 1
fi

# 输出：平均值 ± 百分比（最小值–最大值），百分比 = 标准差/平均值×100（变异系数）
fmt_cell() {
    local mean="$1" std="$2" min="$3" max="$4"
    if [[ -z "$mean" || "$mean" == "nan" ]]; then echo "—"; return; fi
    local pct="0"
    if [[ -n "$std" && "$std" == *[0-9]* && "$mean" == *[0-9]* ]] && ! echo "$mean" | grep -qE '^0+\.?0*$'; then
        pct=$(echo "scale=1; $std * 100 / $mean" | bc 2>/dev/null || echo "0")
    fi
    if [[ -z "$min" || -z "$max" ]]; then echo "${mean} ± ${pct}%"; return; fi
    echo "${mean} ± ${pct}% (${min}–${max})"
}

echo "=== 2.1 sweep_latency 汇总表（平均值 ± 百分比（最小值–最大值））==="
echo "| 链长 | 任务数 | Scheduler Total(us) | Orchestrator(us) | Sched us/task | Orch us/task |"
echo "|------|--------|---------------------|------------------|---------------|---------------|"
while IFS='|' read -r section sample_id tasks s_m s_std s_min s_max o_m o_std o_min o_max; do
    [[ "$section" != "sweep_latency" ]] && continue
    # sample_id 如 X1_Y128 → 链长取最后一个数字或整段
    chain_len="${sample_id##*_Y}"
    sched_cell=$(fmt_cell "$s_m" "$s_std" "$s_min" "$s_max")
    orch_cell=$(fmt_cell "$o_m" "$o_std" "$o_min" "$o_max")
    if [[ -n "$tasks" && "$tasks" =~ ^[0-9]+$ ]]; then
        s_ut=""; o_ut=""
        [[ -n "$s_m" && "$s_m" =~ ^[0-9.]+$ ]] && s_ut=$(echo "scale=2; $s_m / $tasks" | bc 2>/dev/null || echo "")
        [[ -n "$o_m" && "$o_m" =~ ^[0-9.]+$ ]] && o_ut=$(echo "scale=2; $o_m / $tasks" | bc 2>/dev/null || echo "")
        sched_ut_cell=$(fmt_cell "$s_ut" "" "" ""); orch_ut_cell=$(fmt_cell "$o_ut" "" "" "")
    else
        sched_ut_cell="—"; orch_ut_cell="—"
    fi
    echo "| $chain_len | $tasks | $sched_cell | $orch_cell | $sched_ut_cell | $orch_ut_cell |"
done < <(grep -v '^#' "$STATS" | grep -v '^$')

echo ""
echo "=== 3.1 sweep_throughput 汇总表 ==="
echo "| 参数/标签 | 任务数 | Scheduler Total(us) | Orchestrator(us) | Sched us/task | Orch us/task |"
echo "|-----------|--------|---------------------|------------------|---------------|---------------|"
while IFS='|' read -r section sample_id tasks s_m s_std s_min s_max o_m o_std o_min o_max; do
    [[ "$section" != "sweep_throughput" ]] && continue
    sched_cell=$(fmt_cell "$s_m" "$s_std" "$s_min" "$s_max")
    orch_cell=$(fmt_cell "$o_m" "$o_std" "$o_min" "$o_max")
    if [[ -n "$tasks" && "$tasks" =~ ^[0-9]+$ ]]; then
        s_ut=""; o_ut=""
        [[ -n "$s_m" && "$s_m" =~ ^[0-9.]+$ ]] && s_ut=$(echo "scale=3; $s_m / $tasks" | bc 2>/dev/null || echo "")
        [[ -n "$o_m" && "$o_m" =~ ^[0-9.]+$ ]] && o_ut=$(echo "scale=3; $o_m / $tasks" | bc 2>/dev/null || echo "")
        sched_ut_cell=$(fmt_cell "$s_ut" "" "" ""); orch_ut_cell=$(fmt_cell "$o_ut" "" "" "")
    else
        sched_ut_cell="—"; orch_ut_cell="—"
    fi
    echo "| $sample_id | $tasks | $sched_cell | $orch_cell | $sched_ut_cell | $orch_ut_cell |"
done < <(grep -v '^#' "$STATS" | grep -v '^$')

echo ""
echo "=== 4.1 三种线程模式汇总表 ==="
echo "| 用例 | 模式 | 任务数 | Scheduler Total(us) | Orchestrator(us) | Sched us/task | Orch us/task |"
echo "|------|------|--------|---------------------|------------------|---------------|---------------|"
for sec in thread_latency thread_throughput; do
    case "$sec" in thread_latency) case_name="latency (1,128)" ;; thread_throughput) case_name="throughput (n=2,D=6,O=2,W=1024)" ;; esac
    while IFS='|' read -r section sample_id tasks s_m s_std s_min s_max o_m o_std o_min o_max; do
        [[ "$section" != "$sec" ]] && continue
        sched_cell=$(fmt_cell "$s_m" "$s_std" "$s_min" "$s_max")
        orch_cell=$(fmt_cell "$o_m" "$o_std" "$o_min" "$o_max")
        if [[ -n "$tasks" && "$tasks" =~ ^[0-9]+$ ]]; then
            s_ut=""; o_ut=""
            [[ -n "$s_m" && "$s_m" =~ ^[0-9.]+$ ]] && s_ut=$(echo "scale=3; $s_m / $tasks" | bc 2>/dev/null || echo "")
            [[ -n "$o_m" && "$o_m" =~ ^[0-9.]+$ ]] && o_ut=$(echo "scale=3; $o_m / $tasks" | bc 2>/dev/null || echo "")
            sched_ut_cell=$(fmt_cell "$s_ut" "" "" ""); orch_ut_cell=$(fmt_cell "$o_ut" "" "" "")
        else
            sched_ut_cell="—"; orch_ut_cell="—"
        fi
        echo "| $case_name | $sample_id | $tasks | $sched_cell | $orch_cell | $sched_ut_cell | $orch_ut_cell |"
    done < <(grep -v '^#' "$STATS" | grep -v '^$')
done

echo ""
echo "以上表格可复制到 PROFILING_REPORT.md 对应小节。"

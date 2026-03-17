#!/usr/bin/env bash
# all_in_one.sh — 一键生成 test_latency / test_throughput 等性能分析报告文档
#
# 流程：收集 profiling 数据（compare + sweep_latency + sweep_throughput + 线程模式）→ summary_stats.txt → 报告表格并写入文档
#
# 用法：
#   bash tools/all_in_one.sh                    # 完整流程，RUNS=10，报告写入 outputs/profiling_report/PROFILING_REPORT.md
#   RUNS=5 bash tools/all_in_one.sh
#   bash tools/all_in_one.sh --skip-collect     # 仅从已有 summary_stats.txt 重新生成报告
#   bash tools/all_in_one.sh --output my_report.md
#
# 依赖：在 simpler/tests/aicpu_ut 下执行；需 tools/compare_profiling.sh、sweep_latency.sh、sweep_throughput.sh；run_tests.sh 可用且已编译。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AICPU_UT_DIR="${SCRIPT_DIR}/.."
OUT_DIR="${AICPU_UT_DIR}/outputs/profiling_report"
SWEEP_LAT_DIR="${AICPU_UT_DIR}/outputs/sweep_latency_p1"
SWEEP_THR_DIR="${AICPU_UT_DIR}/outputs/sweep_throughput_p1"
REPORT_FILE="${OUT_DIR}/PROFILING_REPORT.md"
STATS_FILE="${OUT_DIR}/summary_stats.txt"
SKIP_COLLECT=false
CUSTOM_OUTPUT=""
RUNS="${RUNS:-10}"

# 解析参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-collect) SKIP_COLLECT=true ;;
        --output)       CUSTOM_OUTPUT="$2"; shift ;;
        --runs)         RUNS="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [--skip-collect] [--output FILE] [--runs N]"
            echo "  --skip-collect   Only regenerate report from existing summary_stats.txt"
            echo "  --output FILE    Write report to FILE (default: outputs/profiling_report/PROFILING_REPORT.md)"
            echo "  --runs N         Run each config N times (default: \$RUNS or 10)"
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

[[ -n "$CUSTOM_OUTPUT" ]] && REPORT_FILE="$CUSTOM_OUTPUT"
mkdir -p "$(dirname "$REPORT_FILE")"
mkdir -p "$OUT_DIR"
cd "$AICPU_UT_DIR"

echo "=============================================="
echo "  AICPU UT 性能分析报告 — 一键生成"
echo "=============================================="
echo "  AICPU_UT_DIR: $AICPU_UT_DIR"
echo "  报告输出:     $REPORT_FILE"
echo "  RUNS:         $RUNS"
echo "=============================================="
echo ""

# -----------------------------------------------------------------------------
# Step 1: 收集数据（可选跳过）
# -----------------------------------------------------------------------------
if [[ "$SKIP_COLLECT" == true ]]; then
    if [[ ! -f "$STATS_FILE" ]]; then
        echo "Error: --skip-collect 但 $STATS_FILE 不存在。请先不带 --skip-collect 运行一次。" >&2
        exit 1
    fi
    echo "[1/2] 跳过数据收集，使用已有 summary_stats.txt"
else
    echo "[1/2] 收集 profiling 数据（compare + sweep_latency + sweep_throughput + 线程模式，每组 ${RUNS} 次）..."

    echo "=== 0. 环境信息 ==="
    env_info="${OUT_DIR}/env.txt"
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
    echo "=== 1. Profiling 1 vs 2 对比 (各 ${RUNS} 次) ==="
    RUNS="$RUNS" bash "${SCRIPT_DIR}/compare_profiling.sh" 2>&1 | tee "${OUT_DIR}/compare_p1_p2.txt"

    echo ""
    echo "=== 2. sweep_latency --profiling 1（9 组参数，每组 ${RUNS} 次）==="
    RUNS="$RUNS" bash "${SCRIPT_DIR}/sweep_latency.sh" --profiling 1

    echo ""
    echo "=== 3. sweep_throughput --profiling 1（14 组参数，每组 ${RUNS} 次）==="
    RUNS="$RUNS" bash "${SCRIPT_DIR}/sweep_throughput.sh" --profiling 1

    echo ""
    echo "=== 4. 三种线程模式对比（代表点各 ${RUNS} 次）==="
    for mode in "" "--orch" "--sched"; do
        name="concurrent"; [ -n "$mode" ] && name="${mode#--}"
        for r in $(seq 1 "$RUNS"); do
            printf "  [%s] run %d/%d\n" "$name" "$r" "$RUNS"
            bash "${AICPU_UT_DIR}/run_tests.sh" --test test_latency --chain-num 1 --chain-length 8192 --profiling 1 --idx 1 $mode \
                > "${OUT_DIR}/latency_p1_${name}_run${r}.log" 2>&1 || true
            bash "${AICPU_UT_DIR}/run_tests.sh" --test test_throughput --layer-num 2 --dependency 6 --overlap 2 --layer0-task-num 1024 --profiling 1 --idx 0 $mode \
                > "${OUT_DIR}/throughput_p1_${name}_run${r}.log" 2>&1 || true
        done
    done

    echo ""
    echo "=== 5. 原始数据提取 ==="
    raw="${OUT_DIR}/raw_data.txt"
    echo "Profiling report raw data — $(date -Iseconds)" > "$raw"
    echo "" >> "$raw"

    # 使用 grep/awk 提取，避免在子 shell 中 BASH_REMATCH 不可靠导致 sched_us/orch_us 为空
    parse_one() {
        local log="$1"
        local sched_us orch_us tasks
        sched_us=$(grep -E '^[[:space:]]+SUM[[:space:]]+' "$log" 2>/dev/null | tail -1 | awk '{print $NF}')
        [[ -z "$sched_us" || ! "$sched_us" =~ ^[0-9.]+$ ]] && sched_us=""
        orch_us=$(grep -E 'Orchestrator run time: [0-9.]+us' "$log" 2>/dev/null | sed -n 's/.*run time: \([0-9.]*\)us.*/\1/p' | head -1)
        [[ -z "$orch_us" ]] && orch_us=$(grep -oE 'Orchestrator Profiling:.*total=[0-9.]+us' "$log" 2>/dev/null | grep -oE '[0-9.]+' | tail -1)
        tasks=$(grep -oE 'Total tasks submitted: [0-9]+' "$log" 2>/dev/null | grep -oE '[0-9]+$' | head -1)
        echo "${sched_us:-nan} ${orch_us:-nan} ${tasks:-nan}"
    }

    echo "--- compare_p1_p2 ---" >> "$raw"
    grep -E "latency|throughput|差距|Sched|Orch" "${OUT_DIR}/compare_p1_p2.txt" >> "$raw" 2>/dev/null || true
    echo "" >> "$raw"
    echo "--- sweep_latency_p1 ---" >> "$raw"
    for f in "${SWEEP_LAT_DIR}"/latency_*_run*.log; do [ -f "$f" ] || continue; echo "$(basename "$f"): $(parse_one "$f")" >> "$raw"; done
    echo "" >> "$raw"
    echo "--- sweep_throughput_p1 ---" >> "$raw"
    for f in "${SWEEP_THR_DIR}"/*_run*.log; do [ -f "$f" ] || continue; echo "$(basename "$f"): $(parse_one "$f")" >> "$raw"; done
    echo "" >> "$raw"
    echo "--- thread_mode latency ---" >> "$raw"
    for name in concurrent orch sched; do
        for f in "${OUT_DIR}"/latency_p1_${name}_run*.log; do [ -f "$f" ] || continue; echo "$(basename "$f"): $(parse_one "$f")" >> "$raw"; done
    done
    echo "" >> "$raw"
    echo "--- thread_mode throughput ---" >> "$raw"
    for name in concurrent orch sched; do
        for f in "${OUT_DIR}"/throughput_p1_${name}_run*.log; do [ -f "$f" ] || continue; echo "$(basename "$f"): $(parse_one "$f")" >> "$raw"; done
    done

    echo "=== 6. 统计汇总（summary_stats.txt）==="
    echo "# section|sample_id|tasks|sched_mean|sched_std|sched_min|sched_max|orch_mean|orch_std|orch_min|orch_max" > "$STATS_FILE"
    echo "# 由 all_in_one.sh 根据各样例 RUNS 次独立 log 计算" >> "$STATS_FILE"

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
    stats_from_logs() {
        local file_list="$1" section="$2" sample_id="$3"
        local tasks="" sched_list="" orch_list=""
        for f in $file_list; do
            [[ -f "$f" ]] || continue
            read -r s o t <<< "$(parse_one "$f")"
            [[ "$s" =~ ^[0-9.]+$ ]] && sched_list="${sched_list:+$sched_list }$s"
            [[ "$o" =~ ^[0-9.]+$ ]] && orch_list="${orch_list:+$orch_list }$o"
            [[ "$t" =~ ^[0-9]+$ ]] && tasks="$t"
        done
        [[ -z "$sched_list" && -z "$orch_list" ]] && return
        local sched_mean sched_std sched_min sched_max orch_mean orch_std orch_min orch_max
        if [[ -n "$sched_list" ]]; then read -r sched_mean sched_std sched_min sched_max <<< "$(calc_stats "$sched_list")"
        else sched_mean=""; sched_std=""; sched_min=""; sched_max=""; fi
        if [[ -n "$orch_list" ]]; then read -r orch_mean orch_std orch_min orch_max <<< "$(calc_stats "$orch_list")"
        else orch_mean=""; orch_std=""; orch_min=""; orch_max=""; fi
        echo "${section}|${sample_id}|${tasks}|${sched_mean}|${sched_std}|${sched_min}|${sched_max}|${orch_mean}|${orch_std}|${orch_min}|${orch_max}" >> "$STATS_FILE"
    }

    for base in latency_X1_Y128 latency_X1_Y256 latency_X1_Y512 latency_X1_Y1024 latency_X1_Y2048 latency_X1_Y4096 latency_X1_Y8192 latency_X1_Y16383; do
        files=""
        for f in "${SWEEP_LAT_DIR}/${base}_run"*.log; do [ -f "$f" ] && files="$files $f"; done
        [[ -z "$files" ]] && continue
        stats_from_logs "$files" "sweep_latency" "${base#latency_}"
    done
    for base in grpW_n2_D6_O2_W1024 grpW_n2_D6_O2_W2048 grpW_n2_D6_O2_W3072 grpYZ_n2_D2_O0_W1024 grpYZ_n2_D2_O1_W1024 grpYZ_n2_D4_O0_W1024 grpYZ_n2_D4_O2_W1024 grpYZ_n2_D6_O0_W1024 grpYZ_n2_D6_O2_W1024 grpYZ_n2_D6_O4_W1024 grpYZ_n2_D8_O4_W1024; do
        files=""
        for f in "${SWEEP_THR_DIR}/${base}_run"*.log; do [ -f "$f" ] && files="$files $f"; done
        [[ -z "$files" ]] && continue
        stats_from_logs "$files" "sweep_throughput" "$base"
    done
    for name in concurrent orch sched; do
        files=""; for f in "${OUT_DIR}"/latency_p1_${name}_run*.log; do [ -f "$f" ] && files="$files $f"; done
        [[ -n "$files" ]] && stats_from_logs "$files" "thread_latency" "$name"
        files=""; for f in "${OUT_DIR}"/throughput_p1_${name}_run*.log; do [ -f "$f" ] && files="$files $f"; done
        [[ -n "$files" ]] && stats_from_logs "$files" "thread_throughput" "$name"
    done
    echo "  已写入 $STATS_FILE"
    echo ""
fi

# -----------------------------------------------------------------------------
# Step 2: 从 summary_stats.txt 生成报告表格并写入文档
# -----------------------------------------------------------------------------
echo "[2/2] 生成报告表格并写入文档..."
if [[ ! -f "$STATS_FILE" ]]; then
    echo "Error: $STATS_FILE not found." >&2
    exit 1
fi

fmt_cell() {
    local mean="$1" std="$2" min="$3" max="$4"
    if [[ -z "$mean" || "$mean" == "nan" ]]; then echo "—"; return; fi
    local is_zero
    is_zero=$(awk "BEGIN{print ($mean+0 == 0) ? 1 : 0}" 2>/dev/null)
    [[ "$is_zero" == "1" ]] && { echo "/"; return; }
    local pct="0"
    if [[ -n "$std" && "$std" == *[0-9]* && "$mean" == *[0-9]* ]] && ! echo "$mean" | grep -qE '^0+\.?0*$'; then
        pct=$(echo "scale=1; $std * 100 / $mean" | bc 2>/dev/null | sed 's/^\./0./' || echo "0")
    fi
    if [[ -z "$min" || -z "$max" ]]; then echo "${mean} ± ${pct}%"; return; fi
    echo "${mean} ± ${pct}% (${min}–${max})"
}

gen_tables() {
    echo "=== 2.1 sweep_latency 汇总表（平均值 ± 百分比（最小值–最大值））==="
    echo "| 链长 | 任务数 | Scheduler Total(us) | Orchestrator(us) | Sched us/task | Orch us/task |"
    echo "|------|--------|---------------------|------------------|---------------|---------------|"
    while IFS='|' read -r section sample_id tasks s_m s_std s_min s_max o_m o_std o_min o_max; do
        [[ "$section" != "sweep_latency" ]] && continue
        chain_len="${sample_id##*_Y}"
        sched_cell=$(fmt_cell "$s_m" "$s_std" "$s_min" "$s_max")
        orch_cell=$(fmt_cell "$o_m" "$o_std" "$o_min" "$o_max")
        if [[ -n "$tasks" && "$tasks" =~ ^[0-9]+$ ]]; then
            s_ut=""; o_ut=""
            [[ -n "$s_m" && "$s_m" =~ ^[0-9.]+$ ]] && s_ut=$(echo "scale=2; $s_m / $tasks" | bc 2>/dev/null | sed 's/^\./0./' || echo "")
            [[ -n "$o_m" && "$o_m" =~ ^[0-9.]+$ ]] && o_ut=$(echo "scale=2; $o_m / $tasks" | bc 2>/dev/null | sed 's/^\./0./' || echo "")
            sched_ut_cell=$(fmt_cell "$s_ut" "" "" ""); orch_ut_cell=$(fmt_cell "$o_ut" "" "" "")
        else
            sched_ut_cell="—"; orch_ut_cell="—"
        fi
        echo "| $chain_len | $tasks | $sched_cell | $orch_cell | $sched_ut_cell | $orch_ut_cell |"
    done < <(grep -v '^#' "$STATS_FILE" | grep -v '^$')

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
            [[ -n "$s_m" && "$s_m" =~ ^[0-9.]+$ ]] && s_ut=$(echo "scale=3; $s_m / $tasks" | bc 2>/dev/null | sed 's/^\./0./' || echo "")
            [[ -n "$o_m" && "$o_m" =~ ^[0-9.]+$ ]] && o_ut=$(echo "scale=3; $o_m / $tasks" | bc 2>/dev/null | sed 's/^\./0./' || echo "")
            sched_ut_cell=$(fmt_cell "$s_ut" "" "" ""); orch_ut_cell=$(fmt_cell "$o_ut" "" "" "")
        else
            sched_ut_cell="—"; orch_ut_cell="—"
        fi
        echo "| $sample_id | $tasks | $sched_cell | $orch_cell | $sched_ut_cell | $orch_ut_cell |"
    done < <(grep -v '^#' "$STATS_FILE" | grep -v '^$')

    echo ""
    echo "=== 4.1 三种线程模式汇总表 ==="
    echo "| 用例 | 模式 | 任务数 | Scheduler Total(us) | Orchestrator(us) | Sched us/task | Orch us/task |"
    echo "|------|------|--------|---------------------|------------------|---------------|---------------|"
    for sec in thread_latency thread_throughput; do
        case "$sec" in thread_latency) case_name="latency (1,8192)" ;; thread_throughput) case_name="throughput (n=2,D=6,O=2,W=1024)" ;; esac
        while IFS='|' read -r section sample_id tasks s_m s_std s_min s_max o_m o_std o_min o_max; do
            [[ "$section" != "$sec" ]] && continue
            sched_cell=$(fmt_cell "$s_m" "$s_std" "$s_min" "$s_max")
            orch_cell=$(fmt_cell "$o_m" "$o_std" "$o_min" "$o_max")
            if [[ -n "$tasks" && "$tasks" =~ ^[0-9]+$ ]]; then
                s_ut=""; o_ut=""
                [[ -n "$s_m" && "$s_m" =~ ^[0-9.]+$ ]] && s_ut=$(echo "scale=3; $s_m / $tasks" | bc 2>/dev/null | sed 's/^\./0./' || echo "")
                [[ -n "$o_m" && "$o_m" =~ ^[0-9.]+$ ]] && o_ut=$(echo "scale=3; $o_m / $tasks" | bc 2>/dev/null | sed 's/^\./0./' || echo "")
                sched_ut_cell=$(fmt_cell "$s_ut" "" "" ""); orch_ut_cell=$(fmt_cell "$o_ut" "" "" "")
            else
                sched_ut_cell="—"; orch_ut_cell="—"
            fi
            echo "| $case_name | $sample_id | $tasks | $sched_cell | $orch_cell | $sched_ut_cell | $orch_ut_cell |"
        done < <(grep -v '^#' "$STATS_FILE" | grep -v '^$')
    done
    echo ""
    echo "报告表格已写入本文件。"
}

{
    echo "# AICPU UT 性能分析报告"
    echo ""
    echo "生成时间: $(date -Iseconds)"
    echo ""
    echo "本报告由 \`tools/all_in_one.sh\` 一键生成，包含 sweep_latency、sweep_throughput 及三种线程模式（concurrent / orch / sched）的汇总表。"
    echo ""
    echo "环境信息见: \`outputs/profiling_report/env.txt\`"
    echo ""
    echo "---"
    echo ""
    gen_tables
} > "$REPORT_FILE"

echo ""
echo "Done. 报告已写入: $REPORT_FILE"
echo "  原始数据与 log: $OUT_DIR (env.txt, raw_data.txt, summary_stats.txt, sweep log 等)"

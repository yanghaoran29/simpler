#!/usr/bin/env bash
# all_in_one.sh — 一键生成 test_latency / test_throughput 等性能分析报告文档
#
# 流程：收集 profiling 数据（compare + Linear latency + Depend1~8 throughput + 线程模式）→ summary_stats.txt → 报告表格（含 Linear + Depend1~8 对比）并写入文档
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
SWEEP_LAT_DIR_P2="${AICPU_UT_DIR}/outputs/sweep_latency_p2"
SWEEP_THR_DIR_P2="${AICPU_UT_DIR}/outputs/sweep_throughput_p2"
# run_tests.sh writes phase breakdown log under PROJECT_ROOT=.../simpler/tests/outputs/
PHASE_LOG_SRC="${AICPU_UT_DIR}/../outputs/aicpu_ut_phase_breakdown.log"
REPORT_FILE="${OUT_DIR}/PROFILING_REPORT.md"
STATS_FILE="${OUT_DIR}/summary_stats.txt"
STATS_FILE_P2="${OUT_DIR}/summary_stats_p2.txt"
PHASE_FILE_P2="${OUT_DIR}/phase_stats_p2.txt"
SKIP_COLLECT=false
CUSTOM_OUTPUT=""
RUNS="${RUNS:-10}"
COLLECT_P1=true
COLLECT_P2=true
ONLY_STEP5=false

# 解析参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-collect) SKIP_COLLECT=true ;;
        --profiling1|--p1) COLLECT_P1=true ;;
        --profiling2|--p2) COLLECT_P2=true ;;
        --no-profiling2|--no-p2) COLLECT_P2=false ;;
        --only-step5|--step5) ONLY_STEP5=true ;;
        --output)       CUSTOM_OUTPUT="$2"; shift ;;
        --runs)         RUNS="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [--skip-collect] [--profiling1] [--profiling2] [--output FILE] [--runs N]"
            echo "  --skip-collect   Only regenerate report from existing summary_stats.txt"
            echo "  --profiling1     Collect profiling=1 sweep data (default: on)"
            echo "  --profiling2     Collect profiling=2 sweep data + phase breakdown analysis (default: on)"
            echo "  --no-profiling2  Disable profiling=2 collection/analysis"
            echo "  --only-step5     Debug only: run minimal P2 cases and print step5 table"
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
AICPU_UT_DIR="$(pwd)"
OUT_DIR="${AICPU_UT_DIR}/outputs/profiling_report"
[[ -z "$CUSTOM_OUTPUT" ]] && REPORT_FILE="${OUT_DIR}/PROFILING_REPORT.md"
STATS_FILE="${OUT_DIR}/summary_stats.txt"
STATS_FILE_P2="${OUT_DIR}/summary_stats_p2.txt"
PHASE_FILE_P2="${OUT_DIR}/phase_stats_p2.txt"
SWEEP_THR_DIR_P2="${AICPU_UT_DIR}/outputs/sweep_throughput_p2"
SWEEP_LAT_DIR_P2="${AICPU_UT_DIR}/outputs/sweep_latency_p2"

echo "=============================================="
echo "  AICPU UT 性能分析报告 — 一键生成"
echo "=============================================="
echo "  AICPU_UT_DIR: $AICPU_UT_DIR"
echo "  报告输出:     $REPORT_FILE"
echo "  RUNS:         $RUNS"
echo "=============================================="
echo ""

if [[ "$ONLY_STEP5" == true ]]; then
    # Debug only: avoid running full pipeline. Generate phase logs for two minimal cases (profiling=2),
    # then parse "Scheduler Phase Breakdown" from the phase log and print a small table.
    mkdir -p "$OUT_DIR"
    echo "[step5] 仅运行 profiling=2 的最小用例并解析 phase breakdown..."

    bash "${AICPU_UT_DIR}/run_tests.sh" --test test_latency --chain-num 128 --chain-length 128 --profiling 2 --idx 1 \
        > "${OUT_DIR}/latency_p2_concurrent_run1.log" 2>&1 || true

    bash "${AICPU_UT_DIR}/run_tests.sh" --test test_throughput --layer-num 128 --layer0-task-num 128 \
        --fix-tail --dependency 8 --overlap 7 --profiling 2 --idx 0 \
        > "${OUT_DIR}/throughput_p2_concurrent_run1.log" 2>&1 || true

    parse_one_phase() {
        local log="$1"
        awk '
            BEGIN{in_breakdown=0; total=0; lockwait=0; fanoutwait=0}
            /=== Scheduler Phase Breakdown: total=/{
                in_breakdown=1;
                if (match($0, /total=([0-9.]+)us/, m)) total=m[1]+0;
                next
            }
            in_breakdown==1{
                if ($0 ~ /^$/) { in_breakdown=0; next }
                if (match($0, /^[[:space:]]*otc_lock[[:space:]]*:[[:space:]]*[0-9.]+us.*wait=([0-9.]+)us/, m)) lockwait=m[1]+0;
                if (match($0, /^[[:space:]]*otc_fanout[[:space:]]*:[[:space:]]*[0-9.]+us.*wait=([0-9.]+)us/, m)) fanoutwait=m[1]+0;
            }
            END{
                if (total<=0) exit 1;
                excl = total - (lockwait + fanoutwait);
                if (excl < 0) excl = 0;
                printf "%.3f %.3f %.3f %.3f\n", total, lockwait, fanoutwait, excl;
            }
        ' "$log"
    }

    echo ""
    echo "=== 5. Profiling 2 分段时间分析（Scheduler Phase Breakdown）==="
    echo ""
    echo "| 用例(Profiling2) | total(us) | lock_wait(us) | fanout_wait(us) | 排除等待后的 total(us) |"
    echo "|------------------|----------:|--------------:|----------------:|-----------------------:|"
    for label in latency_concurrent throughput_concurrent; do
        case "$label" in
            latency_concurrent) f="${OUT_DIR}/latency_p2_concurrent_run1.log" ;;
            throughput_concurrent) f="${OUT_DIR}/throughput_p2_concurrent_run1.log" ;;
        esac
        if [[ -f "$f" ]]; then
            if row=$(parse_one_phase "$f" 2>/dev/null); then
                read -r total lockw fanw excl <<< "$row"
                echo "| ${label} | ${total} | ${lockw} | ${fanw} | ${excl} |"
            else
                echo "| ${label} | — | — | — | — |"
            fi
        else
            echo "| ${label} | — | — | — | — |"
        fi
    done
    echo ""
    echo "备注：此处“排除等待”仅扣除 otc_lock/otc_fanout 中显式标注的 wait 时间；poll 是否计入等待需结合场景另行判断。"
    exit 0
fi

# -----------------------------------------------------------------------------
# Step 1: 收集数据（可选跳过）
# -----------------------------------------------------------------------------
if [[ "$SKIP_COLLECT" == true ]]; then
    if [[ ! -f "$STATS_FILE" ]]; then
        echo "Error: --skip-collect 但 $STATS_FILE 不存在。请先不带 --skip-collect 运行一次。" >&2
        exit 1
    fi
    if [[ "$COLLECT_P2" == true && ! -f "$STATS_FILE_P2" ]]; then
        echo "Warn: --skip-collect 但 $STATS_FILE_P2 不存在，将自动仅用 profiling=1 的统计生成报告（如需 P2 对比，请先完整跑一次）。" >&2
        COLLECT_P2=false
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
    # 确保 P2 采集使用的 binary 是以 PTO2_ORCH_PROFILING=1 编译的，否则 log 中不会有 Orchestrator 子项
    if [[ "$COLLECT_P2" == true ]]; then
        echo "=== 1.5 以 profiling=2 构建（保证 PTO2_ORCH_PROFILING=1，便于输出 Orchestrator 子项）==="
        bash "${AICPU_UT_DIR}/run_tests.sh" --profiling 2 --build-only > "${OUT_DIR}/build_p2.log" 2>&1 || true
        echo ""
    fi
    echo "=== 2. Latency Linear（X128Y128，每组 ${RUNS} 次）==="
    if [[ "$COLLECT_P1" == true ]]; then
        mkdir -p "$SWEEP_LAT_DIR"
        for r in $(seq 1 "$RUNS"); do
            printf "  [P1][Linear] run %d/%d\n" "$r" "$RUNS"
            bash "${AICPU_UT_DIR}/run_tests.sh" --test test_latency --chain-num 128 --chain-length 128 --profiling 1 --idx 1 \
                > "${SWEEP_LAT_DIR}/latency_Linear_run${r}.log" 2>&1 || true
        done
    fi
    if [[ "$COLLECT_P2" == true ]]; then
        mkdir -p "$SWEEP_LAT_DIR_P2"
        mkdir -p "${OUT_DIR}/phase_p2"
        for r in $(seq 1 "$RUNS"); do
            printf "  [P2][Linear] run %d/%d\n" "$r" "$RUNS"
            bash "${AICPU_UT_DIR}/run_tests.sh" --test test_latency --chain-num 128 --chain-length 128 --profiling 2 --idx 1 \
                > "${SWEEP_LAT_DIR_P2}/latency_Linear_run${r}.log" 2>&1 || true
            if [[ -s "$PHASE_LOG_SRC" ]]; then
                cp "$PHASE_LOG_SRC" "${OUT_DIR}/phase_p2/latency_Linear_run${r}.phase.log" 2>/dev/null || true
            fi
        done
    fi

    echo ""
    echo "=== 3. Throughput Depend1~8（n=128, W=128, fix-tail, D=1..8, O=D-1，每组 ${RUNS} 次）==="
    if [[ "$COLLECT_P1" == true ]]; then
        mkdir -p "$SWEEP_THR_DIR"
        for D in 1 2 3 4 5 6 7 8; do
            O=$((D - 1))
            for r in $(seq 1 "$RUNS"); do
                printf "  [P1][Depend%d] run %d/%d (D=%d,O=%d)\n" "$D" "$r" "$RUNS" "$D" "$O"
                bash "${AICPU_UT_DIR}/run_tests.sh" --test test_throughput --layer-num 128 --layer0-task-num 128 \
                    --fix-tail --dependency "$D" --overlap "$O" --profiling 1 --idx 0 \
                    > "${SWEEP_THR_DIR}/throughput_Depend${D}_run${r}.log" 2>&1 || true
            done
        done
    fi
    if [[ "$COLLECT_P2" == true ]]; then
        mkdir -p "$SWEEP_THR_DIR_P2"
        mkdir -p "${OUT_DIR}/phase_p2"
        for D in 1 2 3 4 5 6 7 8; do
            O=$((D - 1))
            for r in $(seq 1 "$RUNS"); do
                printf "  [P2][Depend%d] run %d/%d (D=%d,O=%d)\n" "$D" "$r" "$RUNS" "$D" "$O"
                bash "${AICPU_UT_DIR}/run_tests.sh" --test test_throughput --layer-num 128 --layer0-task-num 128 \
                    --fix-tail --dependency "$D" --overlap "$O" --profiling 2 --idx 0 \
                    > "${SWEEP_THR_DIR_P2}/throughput_Depend${D}_run${r}.log" 2>&1 || true
                if [[ -s "$PHASE_LOG_SRC" ]]; then
                    cp "$PHASE_LOG_SRC" "${OUT_DIR}/phase_p2/throughput_Depend${D}_run${r}.phase.log" 2>/dev/null || true
                fi
            done
        done
    fi

    echo ""
    echo "=== 4. 三种线程模式对比（Linear / Depend8 各 ${RUNS} 次）==="
    if [[ "$COLLECT_P1" == true ]]; then
        for mode in "" "--orch" "--sched"; do
            name="concurrent"; [ -n "$mode" ] && name="${mode#--}"
            for r in $(seq 1 "$RUNS"); do
                printf "  [P1][%s] Linear run %d/%d\n" "$name" "$r" "$RUNS"
                bash "${AICPU_UT_DIR}/run_tests.sh" --test test_latency --chain-num 128 --chain-length 128 --profiling 1 --idx 1 $mode \
                    > "${OUT_DIR}/latency_p1_${name}_run${r}.log" 2>&1 || true
                printf "  [P1][%s] Depend8 run %d/%d\n" "$name" "$r" "$RUNS"
                bash "${AICPU_UT_DIR}/run_tests.sh" --test test_throughput --layer-num 128 --layer0-task-num 128 \
                    --fix-tail --dependency 8 --overlap 7 --profiling 1 --idx 0 $mode \
                    > "${OUT_DIR}/throughput_p1_${name}_run${r}.log" 2>&1 || true
            done
        done
    fi
    if [[ "$COLLECT_P2" == true ]]; then
        for mode in "" "--orch" "--sched"; do
            name="concurrent"; [ -n "$mode" ] && name="${mode#--}"
            for r in $(seq 1 "$RUNS"); do
                printf "  [P2][%s] Linear run %d/%d\n" "$name" "$r" "$RUNS"
                bash "${AICPU_UT_DIR}/run_tests.sh" --test test_latency --chain-num 128 --chain-length 128 --profiling 2 --idx 1 $mode \
                    > "${OUT_DIR}/latency_p2_${name}_run${r}.log" 2>&1 || true
                if [[ -s "$PHASE_LOG_SRC" ]]; then
                    cp "$PHASE_LOG_SRC" "${OUT_DIR}/phase_p2/latency_${name}_run${r}.phase.log" 2>/dev/null || true
                fi
                printf "  [P2][%s] Depend8 run %d/%d\n" "$name" "$r" "$RUNS"
                bash "${AICPU_UT_DIR}/run_tests.sh" --test test_throughput --layer-num 128 --layer0-task-num 128 \
                    --fix-tail --dependency 8 --overlap 7 --profiling 2 --idx 0 $mode \
                    > "${OUT_DIR}/throughput_p2_${name}_run${r}.log" 2>&1 || true
                if [[ -s "$PHASE_LOG_SRC" ]]; then
                    cp "$PHASE_LOG_SRC" "${OUT_DIR}/phase_p2/throughput_${name}_run${r}.phase.log" 2>/dev/null || true
                fi
            done
        done
    fi

    echo ""
    echo "=== 5. 原始数据提取 ==="
    raw="${OUT_DIR}/raw_data.txt"
    echo "Profiling report raw data — $(date -Iseconds)" > "$raw"
    echo "" >> "$raw"

    # 使用 grep/awk 提取，避免在子 shell 中 BASH_REMATCH 不可靠导致 sched_us/orch_us 为空
    parse_one() {
        local log="$1"
        local sched_us orch_us tasks
        # parse_one 需在 set -euo pipefail 下也稳定：
        # grep 在没命中时会返回 1，导致整个脚本提前退出，所以这里统一用 || true 吞掉非 0。
        sched_us=$(grep -E '^[[:space:]]+SUM[[:space:]]+' "$log" 2>/dev/null | tail -1 | awk '{print $NF}' || true)
        [[ -z "$sched_us" || ! "$sched_us" =~ ^[0-9.]+$ ]] && sched_us=""
        # 优先匹配原始 cost 行（即使格式化汇总行异常，也可得到单次真实值）
        orch_us=$(grep -oE 'aicpu_orchestration_entry returned, cost [0-9.]+us' "$log" 2>/dev/null \
            | grep -oE '[0-9.]+' | head -1 || true)
        # 次优匹配格式化后行（P1 旧日志）
        [[ -z "$orch_us" ]] && orch_us=$(grep -E 'Orchestrator[[:space:]]+thread[[:space:]]+[0-9]+:[[:space:]]+[0-9.]+[[:space:]]*us' "$log" 2>/dev/null \
            | sed -n 's/.*:[[:space:]]*\([0-9][0-9.]*\)[[:space:]]*us.*/\1/p' | head -1 || true)
        [[ -z "$orch_us" ]] && orch_us=$(grep -E 'Orchestrator run time: [0-9.]+us' "$log" 2>/dev/null \
            | sed -n 's/.*run time: \([0-9.]*\)us.*/\1/p' | head -1 || true)
        [[ -z "$orch_us" ]] && orch_us=$(grep -oE 'Orchestrator Profiling:.*total=[0-9.]+us' "$log" 2>/dev/null \
            | grep -oE '[0-9.]+' | tail -1 || true)
        # 兼容极端格式差异
        tasks=$(grep -oE 'Total tasks submitted: [0-9]+' "$log" 2>/dev/null \
            | grep -oE '[0-9]+$' | head -1 || true)
        echo "${sched_us:-nan} ${orch_us:-nan} ${tasks:-nan}"
    }

    echo "--- compare_p1_p2 ---" >> "$raw"
    grep -E "latency|throughput|差距|Sched|Orch" "${OUT_DIR}/compare_p1_p2.txt" >> "$raw" 2>/dev/null || true
    echo "" >> "$raw"
    echo "--- Linear (latency X128Y128) ---" >> "$raw"
    for f in "${SWEEP_LAT_DIR}"/latency_Linear_run*.log; do [ -f "$f" ] || continue; echo "P1/$(basename "$f"): $(parse_one "$f")" >> "$raw"; done
    for f in "${SWEEP_LAT_DIR_P2}"/latency_Linear_run*.log; do [ -f "$f" ] || continue; echo "P2/$(basename "$f"): $(parse_one "$f")" >> "$raw"; done
    echo "" >> "$raw"
    echo "--- Depend1~8 (throughput n=128 W=128 fix-tail D=1..8 O=D-1) ---" >> "$raw"
    for d in 1 2 3 4 5 6 7 8; do
        for f in "${SWEEP_THR_DIR}"/throughput_Depend${d}_run*.log; do [ -f "$f" ] || continue; echo "P1/$(basename "$f"): $(parse_one "$f")" >> "$raw"; done
        for f in "${SWEEP_THR_DIR_P2}"/throughput_Depend${d}_run*.log; do [ -f "$f" ] || continue; echo "P2/$(basename "$f"): $(parse_one "$f")" >> "$raw"; done
    done
    echo "" >> "$raw"
    echo "--- thread_mode latency ---" >> "$raw"
    for name in concurrent orch sched; do
        for f in "${OUT_DIR}"/latency_p1_${name}_run*.log; do [ -f "$f" ] || continue; echo "P1/$(basename "$f"): $(parse_one "$f")" >> "$raw"; done
        for f in "${OUT_DIR}"/latency_p2_${name}_run*.log; do [ -f "$f" ] || continue; echo "P2/$(basename "$f"): $(parse_one "$f")" >> "$raw"; done
    done
    echo "" >> "$raw"
    echo "--- thread_mode throughput ---" >> "$raw"
    for name in concurrent orch sched; do
        for f in "${OUT_DIR}"/throughput_p1_${name}_run*.log; do [ -f "$f" ] || continue; echo "P1/$(basename "$f"): $(parse_one "$f")" >> "$raw"; done
        for f in "${OUT_DIR}"/throughput_p2_${name}_run*.log; do [ -f "$f" ] || continue; echo "P2/$(basename "$f"): $(parse_one "$f")" >> "$raw"; done
    done

    echo "=== 6. 统计汇总（summary_stats.txt）==="
    echo "# section|sample_id|tasks|sched_mean|sched_std|sched_min|sched_max|orch_mean|orch_std|orch_min|orch_max" > "$STATS_FILE"
    echo "# 由 all_in_one.sh 根据各样例 RUNS 次独立 log 计算" >> "$STATS_FILE"
    if [[ "$COLLECT_P2" == true ]]; then
        echo "# section|sample_id|tasks|sched_mean|sched_std|sched_min|sched_max|orch_mean|orch_std|orch_min|orch_max" > "$STATS_FILE_P2"
        echo "# 由 all_in_one.sh 根据各样例 RUNS 次独立 log 计算（profiling=2）" >> "$STATS_FILE_P2"
    fi

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
    # 去掉一个最高一个最低后计算均值/标准差/最小/最大；最后两列为排除的最低值、最高值（n>=3 时有效）
    calc_stats_trimmed() {
        echo "$1" | tr ' ' '\n' | awk '
            NF { n++; a[n]=$1+0 }
            END {
                if (n<1) exit;
                if (n<3) {
                    sum=0; for(i=1;i<=n;i++) sum+=a[i]; mean=sum/n;
                    var=0; for(i=1;i<=n;i++) var+=(a[i]-mean)^2;
                    std=(n>1)?sqrt(var/(n-1)):0;
                    min=max=a[1]; for(i=2;i<=n;i++){if(a[i]<min)min=a[i];if(a[i]>max)max=a[i];}
                    printf "%.2f %.2f %.2f %.2f — —", mean, std, min, max
                    exit
                }
                asort(a);
                ex_lo=a[1]; ex_hi=a[n];
                sum=0; for(i=2;i<=n-1;i++) sum+=a[i];
                mean=sum/(n-2);
                var=0; for(i=2;i<=n-1;i++) var+=(a[i]-mean)^2;
                std=(n>3)?sqrt(var/(n-3)):0;
                min=a[2]; max=a[n-1];
                printf "%.2f %.2f %.2f %.2f %.2f %.2f", mean, std, min, max, ex_lo, ex_hi
            }'
    }
    stats_from_logs() {
        local file_list="$1" section="$2" sample_id="$3" use_trim="${4:-}" out_file="${5:-$STATS_FILE}"
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
        local sched_ex_lo sched_ex_hi orch_ex_lo orch_ex_hi
        if [[ "$use_trim" == "trim" ]]; then
            if [[ -n "$sched_list" ]]; then
                read -r sched_mean sched_std sched_min sched_max sched_ex_lo sched_ex_hi <<< "$(calc_stats_trimmed "$sched_list")"
            else sched_mean=""; sched_std=""; sched_min=""; sched_max=""; sched_ex_lo=""; sched_ex_hi=""; fi
            if [[ -n "$orch_list" ]]; then
                read -r orch_mean orch_std orch_min orch_max orch_ex_lo orch_ex_hi <<< "$(calc_stats_trimmed "$orch_list")"
            else orch_mean=""; orch_std=""; orch_min=""; orch_max=""; orch_ex_lo=""; orch_ex_hi=""; fi
            echo "${section}|${sample_id}|${tasks}|${sched_mean}|${sched_std}|${sched_min}|${sched_max}|${orch_mean}|${orch_std}|${orch_min}|${orch_max}" >> "$out_file"
            # 仅 Linear + Depend1~8 的异常值写入 outliers_depend.txt；线程模式也去极值但不单独落表
            if [[ "$section" == "depend" && -n "$sched_ex_lo" && "$sched_ex_lo" != "—" ]]; then
                echo "${sample_id}|Scheduler Total(us)|${sched_ex_lo}|${sched_ex_hi}" >> "${OUT_DIR}/outliers_depend.txt"
            fi
            if [[ "$section" == "depend" && -n "$orch_ex_lo" && "$orch_ex_lo" != "—" ]]; then
                echo "${sample_id}|Orchestrator(us)|${orch_ex_lo}|${orch_ex_hi}" >> "${OUT_DIR}/outliers_depend.txt"
            fi
        else
            if [[ -n "$sched_list" ]]; then read -r sched_mean sched_std sched_min sched_max <<< "$(calc_stats "$sched_list")"
            else sched_mean=""; sched_std=""; sched_min=""; sched_max=""; fi
            if [[ -n "$orch_list" ]]; then read -r orch_mean orch_std orch_min orch_max <<< "$(calc_stats "$orch_list")"
            else orch_mean=""; orch_std=""; orch_min=""; orch_max=""; fi
            echo "${section}|${sample_id}|${tasks}|${sched_mean}|${sched_std}|${sched_min}|${sched_max}|${orch_mean}|${orch_std}|${orch_min}|${orch_max}" >> "$out_file"
        fi
    }

    # Linear + Depend1~8：去一最高一最低后统计，异常值写入 outliers_depend.txt
    : > "${OUT_DIR}/outliers_depend.txt"
    if [[ "$COLLECT_P1" == true ]]; then
        # Linear: latency X128Y128
        files=""
        for f in "${SWEEP_LAT_DIR}"/latency_Linear_run*.log; do [ -f "$f" ] && files="$files $f"; done
        [[ -n "$files" ]] && stats_from_logs "$files" "depend" "Linear" "trim" "$STATS_FILE"
        # Depend1~8: throughput n=128 W=128 fix-tail D=1..8 O=D-1
        for d in 1 2 3 4 5 6 7 8; do
            files=""
            for f in "${SWEEP_THR_DIR}"/throughput_Depend${d}_run*.log; do [ -f "$f" ] && files="$files $f"; done
            [[ -z "$files" ]] && continue
            stats_from_logs "$files" "depend" "Depend${d}" "trim" "$STATS_FILE"
        done
        # 三种线程模式：同样去一最高一最低后统计
        for name in concurrent orch sched; do
            files=""; for f in "${OUT_DIR}"/latency_p1_${name}_run*.log; do [ -f "$f" ] && files="$files $f"; done
            [[ -n "$files" ]] && stats_from_logs "$files" "thread_latency" "$name" "trim" "$STATS_FILE"
            files=""; for f in "${OUT_DIR}"/throughput_p1_${name}_run*.log; do [ -f "$f" ] && files="$files $f"; done
            [[ -n "$files" ]] && stats_from_logs "$files" "thread_throughput" "$name" "trim" "$STATS_FILE"
        done
    fi
    if [[ "$COLLECT_P2" == true ]]; then
        # Linear: latency X128Y128
        files=""
        for f in "${SWEEP_LAT_DIR_P2}"/latency_Linear_run*.log; do [ -f "$f" ] && files="$files $f"; done
        [[ -n "$files" ]] && stats_from_logs "$files" "depend" "Linear" "trim" "$STATS_FILE_P2"
        # Depend1~8: throughput n=128 W=128 fix-tail D=1..8 O=D-1
        for d in 1 2 3 4 5 6 7 8; do
            files=""
            for f in "${SWEEP_THR_DIR_P2}"/throughput_Depend${d}_run*.log; do [ -f "$f" ] && files="$files $f"; done
            [[ -z "$files" ]] && continue
            stats_from_logs "$files" "depend" "Depend${d}" "trim" "$STATS_FILE_P2"
        done
        # 三种线程模式：同样去一最高一最低后统计
        for name in concurrent orch sched; do
            files=""; for f in "${OUT_DIR}"/latency_p2_${name}_run*.log; do [ -f "$f" ] && files="$files $f"; done
            [[ -n "$files" ]] && stats_from_logs "$files" "thread_latency" "$name" "trim" "$STATS_FILE_P2"
            files=""; for f in "${OUT_DIR}"/throughput_p2_${name}_run*.log; do [ -f "$f" ] && files="$files $f"; done
            [[ -n "$files" ]] && stats_from_logs "$files" "thread_throughput" "$name" "trim" "$STATS_FILE_P2"
        done
    fi
    echo "  已写入 $STATS_FILE"
    [[ "$COLLECT_P2" == true ]] && echo "  已写入 $STATS_FILE_P2"
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
if [[ "$COLLECT_P2" == true && ! -f "$STATS_FILE_P2" ]]; then
    echo "Error: $STATS_FILE_P2 not found (profiling2 enabled)." >&2
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
    # 不输出 ±0%
    if [[ -z "$min" || -z "$max" ]]; then
        if [[ "$pct" == "0" || "$pct" == "0.0" ]]; then echo "${mean}"; else echo "${mean} ± ${pct}%"; fi
        return
    fi
    if [[ "$pct" == "0" || "$pct" == "0.0" ]]; then echo "${mean} (${min}–${max})"; else echo "${mean} ± ${pct}% (${min}–${max})"; fi
}

gen_tables() {
    echo "=== 2.1 Linear + Depend1~8 对比（去一最高一最低后：平均值 ± 百分比（最小值–最大值））==="
    echo "| 样例 | 任务数 | Scheduler Total(us) | Orchestrator(us) | Sched us/task | Orch us/task |"
    echo "|------|--------|---------------------|------------------|---------------|---------------|"
    for sample_id in Linear Depend1 Depend2 Depend3 Depend4 Depend5 Depend6 Depend7 Depend8; do
        line=$(grep -v '^#' "$STATS_FILE" | grep -v '^$' | grep "^depend|${sample_id}|" | head -1) || true
        [[ -z "$line" ]] && continue
        IFS='|' read -r section sample_id tasks s_m s_std s_min s_max o_m o_std o_min o_max <<< "$line"
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
    done

    echo ""
    echo "=== 3.1 三种线程模式汇总表（Linear / Depend8，去一最高一最低后）==="
    echo "| 用例 | 平均依赖数 | 模式 | 任务数 | Scheduler Total(us) | Orchestrator(us) | Sched us/task | Orch us/task |"
    echo "|------|------------|------|--------|---------------------|------------------|---------------|---------------|"
    for sec in thread_latency thread_throughput; do
        case "$sec" in thread_latency) case_name="Linear"; avg_dep="—" ;; thread_throughput) case_name="Depend8"; avg_dep="8" ;; esac
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
            echo "| $case_name | $avg_dep | $sample_id | $tasks | $sched_cell | $orch_cell | $sched_ut_cell | $orch_ut_cell |"
        done < <(grep -v '^#' "$STATS_FILE" | grep -v '^$')
    done
    echo ""
    echo "报告表格已写入本文件。"
}

lookup_stat() {
    local stats_file="$1" section="$2" sample="$3"
    grep -v '^#' "$stats_file" | grep -v '^$' | grep "^${section}|${sample}|" | head -1
}

fmt_gap_pct() {
    local p1="$1" p2="$2"
    if [[ -z "$p1" || -z "$p2" || "$p1" == "nan" || "$p2" == "nan" || "$p1" == "0" ]]; then
        echo "—"
        return
    fi
    echo "$(echo "scale=1; ($p2 - $p1) * 100 / $p1" | bc 2>/dev/null | sed 's/^\./0./' || echo "—")%"
}

gen_p1_p2_compare_tables() {
    [[ "$COLLECT_P2" != true ]] && return
    echo ""
    echo "=== 4. Profiling 1 vs 2 对比（同用例：P2 相对 P1 的变化）==="
    echo "| 用例 | 指标 | P1 mean(us) | P2 mean(us) | P2-P1 | 变化比例 |"
    echo "|------|------|------------:|------------:|------:|---------:|"

    for sample_id in Linear Depend1 Depend2 Depend3 Depend4 Depend5 Depend6 Depend7 Depend8; do
        l1=$(lookup_stat "$STATS_FILE" "depend" "$sample_id" || true)
        l2=$(lookup_stat "$STATS_FILE_P2" "depend" "$sample_id" || true)
        [[ -z "$l1" || -z "$l2" ]] && continue
        IFS='|' read -r _ _ _ s1 _ _ _ o1 _ _ _ <<< "$l1"
        IFS='|' read -r _ _ _ s2 _ _ _ o2 _ _ _ <<< "$l2"
        gap_s="—"; gap_o="—"; pct_s="—"; pct_o="—"
        if [[ "$s1" =~ ^[0-9.]+$ && "$s2" =~ ^[0-9.]+$ ]]; then
            gap_s=$(echo "scale=2; $s2 - $s1" | bc 2>/dev/null | sed 's/^\./0./' || echo "—")
            pct_s=$(fmt_gap_pct "$s1" "$s2")
        fi
        if [[ "$o1" =~ ^[0-9.]+$ && "$o2" =~ ^[0-9.]+$ ]]; then
            gap_o=$(echo "scale=2; $o2 - $o1" | bc 2>/dev/null | sed 's/^\./0./' || echo "—")
            pct_o=$(fmt_gap_pct "$o1" "$o2")
        fi
        echo "| $sample_id | Scheduler Total | $s1 | $s2 | $gap_s | $pct_s |"
        echo "| $sample_id | Orchestrator | $o1 | $o2 | $gap_o | $pct_o |"
    done

    echo ""
    echo "注：P2 插桩更完整，通常会引入额外开销；此表用于量化插桩成本。"
}

parse_p2_phase_one() {
    # 输出：total_us complete_us poll_us lock_work_us lock_wait_us fanout_work_us fanout_wait_us fanin_us self_us dispatch_us excl_wait_us
    # 若不存在 phase breakdown，则输出空。
    local log="$1"
    awk '
        BEGIN{
            in_breakdown=0;
            total=complete=poll=fanin=self=dispatch=avg_complete=0;
            lockw=lockwait=fanoutw=fanoutwait=0;
        }
        /=== Scheduler Phase Breakdown: total=/{
            in_breakdown=1;
            if (match($0, /total=([0-9.]+)us/, m)) total=m[1]+0;
            next
        }
        in_breakdown==1{
            if ($0 ~ /^$/) { in_breakdown=0; next }
            if (match($0, /^[[:space:]]*complete[[:space:]]*:[[:space:]]*([0-9.]+)us/, m)) complete=m[1]+0;
            if (match($0, /^[[:space:]]*poll[[:space:]]*:[[:space:]]*([0-9.]+)us/, m)) poll=m[1]+0;
            if (match($0, /^[[:space:]]*otc_lock[[:space:]]*:[[:space:]]*([0-9.]+)us.*work=([0-9.]+)us[[:space:]]+wait=([0-9.]+)us/, m)) { lockw=m[2]+0; lockwait=m[3]+0; }
            if (match($0, /^[[:space:]]*otc_fanout[[:space:]]*:[[:space:]]*([0-9.]+)us.*work=([0-9.]+)us[[:space:]]+wait=([0-9.]+)us/, m)) { fanoutw=m[2]+0; fanoutwait=m[3]+0; }
            if (match($0, /^[[:space:]]*otc_fanin[[:space:]]*:[[:space:]]*([0-9.]+)us/, m)) fanin=m[1]+0;
            if (match($0, /^[[:space:]]*otc_self[[:space:]]*:[[:space:]]*([0-9.]+)us/, m)) self=m[1]+0;
            if (match($0, /^[[:space:]]*dispatch[[:space:]]*:[[:space:]]*([0-9.]+)us/, m)) dispatch=m[1]+0;
            if (match($0, /^[[:space:]]*avg\/complete[[:space:]]*:[[:space:]]*([0-9.]+)us/, m)) avg_complete=m[1]+0;
        }
        END{
            if (total<=0) exit 0;
            excl = total - (lockwait + fanoutwait);
            if (excl < 0) excl = 0;
            printf "%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n",
                   total, complete, poll, lockw, lockwait, fanoutw, fanoutwait, fanin, self, dispatch, avg_complete, excl;
        }
    ' "$log"
}

collect_phase_trimmed() {
    # 输入：glob（空格分隔的文件列表字符串），输出每列去一高一低后的均值
    local file_list="$1"
    local rows=""
    for f in $file_list; do
        [[ -f "$f" ]] || continue
        one=$(parse_p2_phase_one "$f" || true)
        [[ -n "$one" ]] && rows="${rows}${one}"$'\n'
    done
    [[ -z "$rows" ]] && return 1
    # 对每一列分别去一高一低后求均值
    echo "$rows" | awk '
        NF>=12{
            n++
            for(i=1;i<=12;i++){ a[i,n]=$i+0 }
        }
        function mean_trim(col,   i, arr, m, sum){
            if (n<1) return "";
            # copy
            for(i=1;i<=n;i++) arr[i]=a[col,i]
            # sort
            asort(arr)
            if (n<=2){
                sum=0; for(i=1;i<=n;i++) sum+=arr[i];
                return sum/n
            }
            sum=0; for(i=2;i<=n-1;i++) sum+=arr[i];
            return sum/(n-2)
        }
        END{
            if(n<1) exit 1;
            for(c=1;c<=12;c++){
                v=mean_trim(c);
                printf "%.3f%s", v, (c==12?ORS:OFS)
            }
        }
    ' OFS=' '
}

gen_p2_phase_analysis() {
    [[ "$COLLECT_P2" != true ]] && return
    echo ""
    echo "=== 5. Profiling 2 细粒度分段时间对比分析（Depend + 模式）==="
    echo ""
    echo "说明："
    echo "- **Scheduler Phase Breakdown**：取 profiling=2 日志里的 total/complete/poll/otc_lock(work & wait)/otc_fanout(work & wait)/otc_fanin/otc_self/dispatch/avg/complete"
    echo "- **sched_overhead_analysis Phase 表**：取 profiling=2 日志里的 Phase(Complete/Scan/Dispatch/Idle) Total(us) 与 Pop(hit/miss/hit_rate)、Fanout/Fanin edges 统计"
    echo "- **排除等待**：total_excl_wait_us = total - (otc_lock.wait + otc_fanout.wait)"
    echo "- **Orchestrator**：取 Orchestrator thread 的总耗时；子项（sync_tensormap / task_ring_alloc / param_copy / lookup+dep / heap_alloc / tensormap_ins / fanin+ready / scope_end）仅在 **P2 构建**（PTO2_ORCH_PROFILING=1）下由 C++ 输出，本节从 P2 日志中解析并做对比；若表中为「无数据」请确认以 \`--profiling 2\` 构建并重新采集。"
    echo ""
    echo "**为何 aicpu_ut 的 P2 log 里没有 Orchestrator 子项 / 格式与设备 log 不同？**"
    echo "- **格式**：你看到的 \`[ERROR] AICPU(pid,thread):... [device_log.cpp:71][ALWAYS] ... run \"Thread 3: === Orchestrator Profiling: ...\"\` 来自 **onboard（设备侧）** 的 \`device_log.cpp\`，通过 \`dlog_error\` 输出。aicpu_ut 跑的是 **sim + test_log_stubs**，stub 里 \`dev_log_always\` 只做 \`printf(\"%s\\\\n\", buf)\`，不会加 [ALWAYS]/device_log/进程号等前缀，因此即时有子项也只会是纯行 \`Thread 3: === Orchestrator Profiling: ...\`。"
    echo "- **内容缺失**：若 log 里在 \`Thread N: aicpu_orchestration_entry returned, cost Xus\` 之后完全没有 \`Thread N: === Orchestrator Profiling:\` 和 sync_tensormap 等行，说明当前可执行文件在编译时 **未打开 PTO2_ORCH_PROFILING**（例如曾用 \`--profiling 1\` 构建后未再以 \`--profiling 2\` 重新配置编译就跑了 P2 采集）。解决：在采集 P2 前用 \`bash run_tests.sh --profiling 2\` 做一次完整配置与编译（建议先 \`rm -rf build\` 再跑），再重跑 all_in_one 采集，则新生成的 P2 log 中会出现上述纯文本形式的 Orchestrator 子项行。"
    echo ""

    fmt_cell() { [[ -n "$1" ]] && echo "$1" || echo "无数据"; }

    # 从单条 P2 日志中解析 Orchestrator Profiling 子项（Thread N:   sync_tensormap : Xus ... 等）
    # 输出：sync_us task_ring_alloc_us param_copy_us lookup_dep_us heap_alloc_us tensormap_ins_us fanin_ready_us scope_end_us avg_task_us（空格分隔，缺则空）
    parse_orch_phase_one_log() {
        local log="$1"
        [[ -f "$log" ]] || return 1
        awk '
            BEGIN{ got=0 }
            /sync_tensormap[[:space:]]*:[[:space:]]*[0-9.]+us/{ n=$0; sub(/.*sync_tensormap[[:space:]]*:[[:space:]]*/, "", n); sub(/us.*/, "", n); sync=n+0; got=1; next }
            /task_ring_alloc[[:space:]]*:[[:space:]]*[0-9.]+us/{ n=$0; sub(/.*task_ring_alloc[[:space:]]*:[[:space:]]*/, "", n); sub(/us.*/, "", n); task_ring=n+0; got=1; next }
            /param_copy[[:space:]]*:[[:space:]]*[0-9.]+us/{ n=$0; sub(/.*param_copy[[:space:]]*:[[:space:]]*/, "", n); sub(/us.*/, "", n); param=n+0; got=1; next }
            /lookup[+]dep[[:space:]]*:[[:space:]]*[0-9.]+us/{ n=$0; sub(/.*lookup[+]dep[[:space:]]*:[[:space:]]*/, "", n); sub(/us.*/, "", n); lookup=n+0; got=1; next }
            /heap_alloc[[:space:]]*:[[:space:]]*[0-9.]+us/{ n=$0; sub(/.*heap_alloc[[:space:]]*:[[:space:]]*/, "", n); sub(/us.*/, "", n); heap=n+0; got=1; next }
            /tensormap_ins[[:space:]]*:[[:space:]]*[0-9.]+us/{ n=$0; sub(/.*tensormap_ins[[:space:]]*:[[:space:]]*/, "", n); sub(/us.*/, "", n); tensormap_ins=n+0; got=1; next }
            /fanin[+]ready[[:space:]]*:[[:space:]]*[0-9.]+us/{ n=$0; sub(/.*fanin[+]ready[[:space:]]*:[[:space:]]*/, "", n); sub(/us.*/, "", n); fanin_ready=n+0; got=1; next }
            /scope_end[[:space:]]*:[[:space:]]*[0-9.]+us/{ n=$0; sub(/.*scope_end[[:space:]]*:[[:space:]]*/, "", n); sub(/us.*/, "", n); scope_end=n+0; got=1; next }
            /avg[/]task[[:space:]]*:[[:space:]]*[0-9.]+us/{ n=$0; sub(/.*avg[/]task[[:space:]]*:[[:space:]]*/, "", n); sub(/us.*/, "", n); avg_task=n+0; got=1; next }
            END{
                if (!got) exit 1;
                printf "%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n", sync+0, task_ring+0, param+0, lookup+0, heap+0, tensormap_ins+0, fanin_ready+0, scope_end+0, avg_task+0;
            }
        ' "$log" 2>/dev/null || return 1
    }

    collect_orch_phase_trimmed() {
        local file_list="$1"
        local rows="" f one
        for f in $file_list; do
            [[ -f "$f" ]] || continue
            one=$(parse_orch_phase_one_log "$f" 2>/dev/null || true)
            [[ -n "$one" ]] && rows="${rows}${one}"$'\n'
        done
        [[ -z "$rows" ]] && return 1
        echo "$rows" | awk '
            NF>=9{ n++; for(i=1;i<=9;i++) a[i,n]=$i+0 }
            function mean_trim(col,   i, arr, sum){
                if (n<1) return "";
                for(i=1;i<=n;i++) arr[i]=a[col,i];
                asort(arr);
                if (n<=2){ sum=0; for(i=1;i<=n;i++) sum+=arr[i]; return sum/n; }
                sum=0; for(i=2;i<=n-1;i++) sum+=arr[i]; return sum/(n-2);
            }
            END{
                if (n<1) exit 1;
                for(c=1;c<=9;c++) printf "%.3f%s", mean_trim(c), (c==9?ORS:OFS);
            }
        ' OFS=' '
    }

    # 从单条日志提取 Orchestrator 总耗时：支持 P1 格式 "Orchestrator thread N: X us (total)" 与 P2 格式 "total=Xus" / "cost Xus"
    extract_orch_total_one_log() {
        local log="$1"
        [[ -f "$log" ]] || return 1
        awk '
            {
                if (match($0, /Orchestrator[[:space:]]+thread[[:space:]]+[0-9]+:[[:space:]]*([0-9.]+)[[:space:]]+us[[:space:]]*\(total\)/, m))
                    sum += m[1];
                if (match($0, /Orchestrator Profiling:.*total=([0-9.]+)us/, m))
                    p2_total = m[1] + 0;
                if (match($0, /aicpu_orchestration_entry returned,[[:space:]]*cost[[:space:]]+([0-9.]+)us/, m))
                    p2_cost = m[1] + 0;
            }
            END{
                if (p2_total > 0) { printf "%.3f", p2_total; exit 0 }
                if (p2_cost > 0)  { printf "%.3f", p2_cost;  exit 0 }
                if (sum > 0)      { printf "%.3f", sum;     exit 0 }
            }' "$log"
    }

    calc_trimmed_mean_from_vals() {
        # stdin: one value per line
        awk '
            NF { a[++n]=$1 }
            END{
                if (n<1) exit 1;
                if (n<=2){
                    sum=0; for(i=1;i<=n;i++) sum+=a[i];
                    printf "%.3f", sum/n;
                    exit 0;
                }
                asort(a);
                sum=0;
                for(i=2;i<=n-1;i++) sum += a[i];
                printf "%.3f", sum/(n-2);
            }'
    }

    collect_orch_total_trimmed() {
        local file_list="$1"
        local vals=""
        local f v
        for f in $file_list; do
            [[ -f "$f" ]] || continue
            v="$(extract_orch_total_one_log "$f" 2>/dev/null || true)"
            [[ "$v" =~ ^[0-9.]+$ ]] || continue
            vals="${vals}${vals:+ }$v"
        done
        [[ -z "$vals" ]] && return 1
        echo "$vals" | tr ' ' '\n' | calc_trimmed_mean_from_vals
    }

    parse_p2_sched_overhead_one() {
        # 输出：
        # phase_complete_us phase_scan_us phase_dispatch_us phase_idle_us
        # pop_hit pop_miss pop_hit_rate
        # fanout_edges fanout_max_degree fanout_avg_degree
        # fanin_edges fanin_max_degree fanin_avg_degree
        local log="$1"
        awk '
            BEGIN{
                in_phase=0;
                have=0;
                c=s=d=i=0;
                ph=pm=0; phr="";
                fo_e=fo_max=fo_avg="";
                fi_e=fi_max=fi_avg="";
            }
            # Example header:
            # "  Phase                                               Total (us) % of total  Avg/task (us)"
            /^[[:space:]]*Phase[[:space:]]+.*Total[[:space:]]*\(us\)/{ in_phase=1; have=1; next }
            in_phase==1{
                if ($0 ~ /^[[:space:]]*-{5,}/) next;
                if ($0 ~ /^$/) { in_phase=0; next }
                if ($0 ~ /^[[:space:]]*Complete[[:space:]]*\(/) { if (match($0, /[[:space:]]([0-9.]+)[[:space:]]+[-0-9.]+%/, m)) c=m[1]+0; next }
                if ($0 ~ /^[[:space:]]*Scan[[:space:]]*\(/)     { if (match($0, /[[:space:]]([0-9.]+)[[:space:]]+[-0-9.]+%/, m)) s=m[1]+0; next }
                if ($0 ~ /^[[:space:]]*Dispatch[[:space:]]*\(/)  { if (match($0, /[[:space:]]([0-9.]+)[[:space:]]+[-0-9.]+%/, m)) d=m[1]+0; next }
                if ($0 ~ /^[[:space:]]*Idle[[:space:]]*\(/)      { if (match($0, /[[:space:]]([0-9.]+)[[:space:]]+[-0-9.]+%/, m)) i=m[1]+0; next }
            }
            /Pop:[[:space:]]*hit=/{
                if (match($0, /hit=([0-9]+),[[:space:]]*miss=([0-9]+),[[:space:]]*hit_rate=([0-9.]+)%/, m)) { ph=m[1]+0; pm=m[2]+0; phr=m[3]; }
                have=1;
            }
            /Fanout[[:space:]]*\(notify consumers\):/{
                if (match($0, /total edges=([0-9]+),[[:space:]]*max_degree=([0-9]+),[[:space:]]*avg_degree=([0-9.]+)/, m)) { fo_e=m[1]; fo_max=m[2]; fo_avg=m[3]; }
                have=1;
            }
            /Fanin[[:space:]]*\(release producers\):/{
                if (match($0, /total edges=([0-9]+),[[:space:]]*max_degree=([0-9]+),[[:space:]]*avg_degree=([0-9.]+)/, m)) { fi_e=m[1]; fi_max=m[2]; fi_avg=m[3]; }
                have=1;
            }
            END{
                if (!have) exit 0;
                printf "%.3f %.3f %.3f %.3f %d %d %s %s %s %s %s %s %s\n",
                       c,s,d,i, ph,pm, (phr==""?"—":phr),
                       (fo_e==""?"—":fo_e), (fo_max==""?"—":fo_max), (fo_avg==""?"—":fo_avg),
                       (fi_e==""?"—":fi_e), (fi_max==""?"—":fi_max), (fi_avg==""?"—":fi_avg);
            }
        ' "$log"
    }

    collect_sched_overhead_mean_files() {
        # 暂不去极值：对每列做简单均值；缺失值输出 —；hit_rate 取最后一个非 —。
        local file_list="$*"
        local rows=""
        local f one
        for f in $file_list; do
            [[ -f "$f" ]] || continue
            one=$(parse_p2_sched_overhead_one "$f" 2>/dev/null || true)
            [[ -n "$one" ]] && rows="${rows}${one}"$'\n'
        done
        [[ -z "$rows" ]] && return 1
        echo "$rows" | awk '
            NF>=13{
                n++;
                for(i=1;i<=6;i++) sum[i]+=$i;
                if($7!="—") hitr=$7;
                for(i=8;i<=13;i++) if($i!="—"){ sum[i]+=$i; cnt[i]++; }
            }
            END{
                if(n<1) exit 1;
                for(i=1;i<=6;i++) printf "%.3f ", sum[i]/n;
                printf "%s ", (hitr==""?"—":hitr);
                for(i=8;i<=13;i++){
                    if(cnt[i]<1) printf "—%s", (i==13?ORS:OFS);
                    else printf "%.3f%s", sum[i]/cnt[i], (i==13?ORS:OFS);
                }
            }'
    }

    : > "$PHASE_FILE_P2"
    echo "# D|total|complete|poll|otc_lock_work|otc_lock_wait|otc_fanout_work|otc_fanout_wait|otc_fanin|otc_self|avg_complete|total_excl_wait" > "$PHASE_FILE_P2"

    # 1) 对比：orch+sche (concurrent) 在 Depend1~8 的各分段变化（Depend1~8 均为 throughput，D=1..8）
    declare -A v_total v_complete v_poll v_lock_work v_lock_wait v_fanout_work v_fanout_wait v_fanin v_self v_dispatch v_avg_complete v_excl_wait v_orch_total
    declare -A v_orch_sync v_orch_task_ring v_orch_param v_orch_lookup v_orch_heap v_orch_tensormap_ins v_orch_fanin_ready v_orch_scope_end v_orch_avg_task
    for D in 1 2 3 4 5 6 7 8; do
        files=""
        for f in "${SWEEP_THR_DIR_P2}/throughput_Depend${D}_run"*.log; do [ -f "$f" ] && files="$files $f"; done
        # 去掉 files 前导空格，避免解析时第一个 f 为空
        files="${files# }"
        [[ -z "$files" ]] && continue
        if row=$(collect_phase_trimmed "$files" 2>/dev/null); then
            IFS=' ' read -r v_total[$D] v_complete[$D] v_poll[$D] v_lock_work[$D] v_lock_wait[$D] v_fanout_work[$D] v_fanout_wait[$D] v_fanin[$D] v_self[$D] v_dispatch[$D] v_avg_complete[$D] v_excl_wait[$D] <<< "$row"
            orch="$(collect_orch_total_trimmed "$files" 2>/dev/null || true)"
            [[ "$orch" =~ ^[0-9.]+$ ]] && v_orch_total[$D]="$orch"
            echo "$D|${row// /|}" >> "$PHASE_FILE_P2"
        fi
        orch_phase=$(collect_orch_phase_trimmed "$files" 2>/dev/null || true)
        if [[ -n "$orch_phase" ]]; then
            IFS=' ' read -r v_orch_sync[$D] v_orch_task_ring[$D] v_orch_param[$D] v_orch_lookup[$D] v_orch_heap[$D] v_orch_tensormap_ins[$D] v_orch_fanin_ready[$D] v_orch_scope_end[$D] v_orch_avg_task[$D] <<< "$orch_phase"
        fi
    done

    echo "（1）orch+sche(concurrent) 在不同 Depend 下的 Scheduler Phase Breakdown（去一最高一最低后均值）"
    echo "| Metric | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 |"
    echo "|--------|---:|---:|---:|---:|---:|---:|---:|---:|"
    echo "| total | $(fmt_cell "${v_total[1]:-}") | $(fmt_cell "${v_total[2]:-}") | $(fmt_cell "${v_total[3]:-}") | $(fmt_cell "${v_total[4]:-}") | $(fmt_cell "${v_total[5]:-}") | $(fmt_cell "${v_total[6]:-}") | $(fmt_cell "${v_total[7]:-}") | $(fmt_cell "${v_total[8]:-}") |"
    echo "| complete | $(fmt_cell "${v_complete[1]:-}") | $(fmt_cell "${v_complete[2]:-}") | $(fmt_cell "${v_complete[3]:-}") | $(fmt_cell "${v_complete[4]:-}") | $(fmt_cell "${v_complete[5]:-}") | $(fmt_cell "${v_complete[6]:-}") | $(fmt_cell "${v_complete[7]:-}") | $(fmt_cell "${v_complete[8]:-}") |"
    echo "| poll | $(fmt_cell "${v_poll[1]:-}") | $(fmt_cell "${v_poll[2]:-}") | $(fmt_cell "${v_poll[3]:-}") | $(fmt_cell "${v_poll[4]:-}") | $(fmt_cell "${v_poll[5]:-}") | $(fmt_cell "${v_poll[6]:-}") | $(fmt_cell "${v_poll[7]:-}") | $(fmt_cell "${v_poll[8]:-}") |"
    echo "| otc_lock_work | $(fmt_cell "${v_lock_work[1]:-}") | $(fmt_cell "${v_lock_work[2]:-}") | $(fmt_cell "${v_lock_work[3]:-}") | $(fmt_cell "${v_lock_work[4]:-}") | $(fmt_cell "${v_lock_work[5]:-}") | $(fmt_cell "${v_lock_work[6]:-}") | $(fmt_cell "${v_lock_work[7]:-}") | $(fmt_cell "${v_lock_work[8]:-}") |"
    echo "| otc_lock_wait | $(fmt_cell "${v_lock_wait[1]:-}") | $(fmt_cell "${v_lock_wait[2]:-}") | $(fmt_cell "${v_lock_wait[3]:-}") | $(fmt_cell "${v_lock_wait[4]:-}") | $(fmt_cell "${v_lock_wait[5]:-}") | $(fmt_cell "${v_lock_wait[6]:-}") | $(fmt_cell "${v_lock_wait[7]:-}") | $(fmt_cell "${v_lock_wait[8]:-}") |"
    echo "| otc_fanout_work | $(fmt_cell "${v_fanout_work[1]:-}") | $(fmt_cell "${v_fanout_work[2]:-}") | $(fmt_cell "${v_fanout_work[3]:-}") | $(fmt_cell "${v_fanout_work[4]:-}") | $(fmt_cell "${v_fanout_work[5]:-}") | $(fmt_cell "${v_fanout_work[6]:-}") | $(fmt_cell "${v_fanout_work[7]:-}") | $(fmt_cell "${v_fanout_work[8]:-}") |"
    echo "| otc_fanout_wait | $(fmt_cell "${v_fanout_wait[1]:-}") | $(fmt_cell "${v_fanout_wait[2]:-}") | $(fmt_cell "${v_fanout_wait[3]:-}") | $(fmt_cell "${v_fanout_wait[4]:-}") | $(fmt_cell "${v_fanout_wait[5]:-}") | $(fmt_cell "${v_fanout_wait[6]:-}") | $(fmt_cell "${v_fanout_wait[7]:-}") | $(fmt_cell "${v_fanout_wait[8]:-}") |"
    echo "| otc_fanin | $(fmt_cell "${v_fanin[1]:-}") | $(fmt_cell "${v_fanin[2]:-}") | $(fmt_cell "${v_fanin[3]:-}") | $(fmt_cell "${v_fanin[4]:-}") | $(fmt_cell "${v_fanin[5]:-}") | $(fmt_cell "${v_fanin[6]:-}") | $(fmt_cell "${v_fanin[7]:-}") | $(fmt_cell "${v_fanin[8]:-}") |"
    echo "| otc_self | $(fmt_cell "${v_self[1]:-}") | $(fmt_cell "${v_self[2]:-}") | $(fmt_cell "${v_self[3]:-}") | $(fmt_cell "${v_self[4]:-}") | $(fmt_cell "${v_self[5]:-}") | $(fmt_cell "${v_self[6]:-}") | $(fmt_cell "${v_self[7]:-}") | $(fmt_cell "${v_self[8]:-}") |"
    echo "| dispatch | $(fmt_cell "${v_dispatch[1]:-}") | $(fmt_cell "${v_dispatch[2]:-}") | $(fmt_cell "${v_dispatch[3]:-}") | $(fmt_cell "${v_dispatch[4]:-}") | $(fmt_cell "${v_dispatch[5]:-}") | $(fmt_cell "${v_dispatch[6]:-}") | $(fmt_cell "${v_dispatch[7]:-}") | $(fmt_cell "${v_dispatch[8]:-}") |"
    echo "| avg/complete | $(fmt_cell "${v_avg_complete[1]:-}") | $(fmt_cell "${v_avg_complete[2]:-}") | $(fmt_cell "${v_avg_complete[3]:-}") | $(fmt_cell "${v_avg_complete[4]:-}") | $(fmt_cell "${v_avg_complete[5]:-}") | $(fmt_cell "${v_avg_complete[6]:-}") | $(fmt_cell "${v_avg_complete[7]:-}") | $(fmt_cell "${v_avg_complete[8]:-}") |"
    echo "| total_excl_wait_us | $(fmt_cell "${v_excl_wait[1]:-}") | $(fmt_cell "${v_excl_wait[2]:-}") | $(fmt_cell "${v_excl_wait[3]:-}") | $(fmt_cell "${v_excl_wait[4]:-}") | $(fmt_cell "${v_excl_wait[5]:-}") | $(fmt_cell "${v_excl_wait[6]:-}") | $(fmt_cell "${v_excl_wait[7]:-}") | $(fmt_cell "${v_excl_wait[8]:-}") |"

    echo ""
    echo "（1b）orch+sche(concurrent) 的 Orchestrator thread 总耗时（去一最高一最低后均值）"
    echo "| Orchestrator(us) | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 |"
    echo "|------------------|---:|---:|---:|---:|---:|---:|---:|---:|"
    echo "| Orchestrator total | $(fmt_cell "${v_orch_total[1]:-}") | $(fmt_cell "${v_orch_total[2]:-}") | $(fmt_cell "${v_orch_total[3]:-}") | $(fmt_cell "${v_orch_total[4]:-}") | $(fmt_cell "${v_orch_total[5]:-}") | $(fmt_cell "${v_orch_total[6]:-}") | $(fmt_cell "${v_orch_total[7]:-}") | $(fmt_cell "${v_orch_total[8]:-}") |"

    echo ""
    echo "（1d）Orchestrator 子项耗时（us，去一最高一最低后均值；仅 P2 构建且日志含 \`Thread N: === Orchestrator Profiling:\` 及 sync_tensormap 等行时有效）"
    echo "| Phase | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 |"
    echo "|-------|---:|---:|---:|---:|---:|---:|---:|---:|"
    echo "| sync_tensormap | $(fmt_cell "${v_orch_sync[1]:-}") | $(fmt_cell "${v_orch_sync[2]:-}") | $(fmt_cell "${v_orch_sync[3]:-}") | $(fmt_cell "${v_orch_sync[4]:-}") | $(fmt_cell "${v_orch_sync[5]:-}") | $(fmt_cell "${v_orch_sync[6]:-}") | $(fmt_cell "${v_orch_sync[7]:-}") | $(fmt_cell "${v_orch_sync[8]:-}") |"
    echo "| task_ring_alloc | $(fmt_cell "${v_orch_task_ring[1]:-}") | $(fmt_cell "${v_orch_task_ring[2]:-}") | $(fmt_cell "${v_orch_task_ring[3]:-}") | $(fmt_cell "${v_orch_task_ring[4]:-}") | $(fmt_cell "${v_orch_task_ring[5]:-}") | $(fmt_cell "${v_orch_task_ring[6]:-}") | $(fmt_cell "${v_orch_task_ring[7]:-}") | $(fmt_cell "${v_orch_task_ring[8]:-}") |"
    echo "| param_copy | $(fmt_cell "${v_orch_param[1]:-}") | $(fmt_cell "${v_orch_param[2]:-}") | $(fmt_cell "${v_orch_param[3]:-}") | $(fmt_cell "${v_orch_param[4]:-}") | $(fmt_cell "${v_orch_param[5]:-}") | $(fmt_cell "${v_orch_param[6]:-}") | $(fmt_cell "${v_orch_param[7]:-}") | $(fmt_cell "${v_orch_param[8]:-}") |"
    echo "| lookup+dep | $(fmt_cell "${v_orch_lookup[1]:-}") | $(fmt_cell "${v_orch_lookup[2]:-}") | $(fmt_cell "${v_orch_lookup[3]:-}") | $(fmt_cell "${v_orch_lookup[4]:-}") | $(fmt_cell "${v_orch_lookup[5]:-}") | $(fmt_cell "${v_orch_lookup[6]:-}") | $(fmt_cell "${v_orch_lookup[7]:-}") | $(fmt_cell "${v_orch_lookup[8]:-}") |"
    echo "| heap_alloc | $(fmt_cell "${v_orch_heap[1]:-}") | $(fmt_cell "${v_orch_heap[2]:-}") | $(fmt_cell "${v_orch_heap[3]:-}") | $(fmt_cell "${v_orch_heap[4]:-}") | $(fmt_cell "${v_orch_heap[5]:-}") | $(fmt_cell "${v_orch_heap[6]:-}") | $(fmt_cell "${v_orch_heap[7]:-}") | $(fmt_cell "${v_orch_heap[8]:-}") |"
    echo "| tensormap_ins | $(fmt_cell "${v_orch_tensormap_ins[1]:-}") | $(fmt_cell "${v_orch_tensormap_ins[2]:-}") | $(fmt_cell "${v_orch_tensormap_ins[3]:-}") | $(fmt_cell "${v_orch_tensormap_ins[4]:-}") | $(fmt_cell "${v_orch_tensormap_ins[5]:-}") | $(fmt_cell "${v_orch_tensormap_ins[6]:-}") | $(fmt_cell "${v_orch_tensormap_ins[7]:-}") | $(fmt_cell "${v_orch_tensormap_ins[8]:-}") |"
    echo "| fanin+ready | $(fmt_cell "${v_orch_fanin_ready[1]:-}") | $(fmt_cell "${v_orch_fanin_ready[2]:-}") | $(fmt_cell "${v_orch_fanin_ready[3]:-}") | $(fmt_cell "${v_orch_fanin_ready[4]:-}") | $(fmt_cell "${v_orch_fanin_ready[5]:-}") | $(fmt_cell "${v_orch_fanin_ready[6]:-}") | $(fmt_cell "${v_orch_fanin_ready[7]:-}") | $(fmt_cell "${v_orch_fanin_ready[8]:-}") |"
    echo "| scope_end | $(fmt_cell "${v_orch_scope_end[1]:-}") | $(fmt_cell "${v_orch_scope_end[2]:-}") | $(fmt_cell "${v_orch_scope_end[3]:-}") | $(fmt_cell "${v_orch_scope_end[4]:-}") | $(fmt_cell "${v_orch_scope_end[5]:-}") | $(fmt_cell "${v_orch_scope_end[6]:-}") | $(fmt_cell "${v_orch_scope_end[7]:-}") | $(fmt_cell "${v_orch_scope_end[8]:-}") |"
    echo "| avg/task | $(fmt_cell "${v_orch_avg_task[1]:-}") | $(fmt_cell "${v_orch_avg_task[2]:-}") | $(fmt_cell "${v_orch_avg_task[3]:-}") | $(fmt_cell "${v_orch_avg_task[4]:-}") | $(fmt_cell "${v_orch_avg_task[5]:-}") | $(fmt_cell "${v_orch_avg_task[6]:-}") | $(fmt_cell "${v_orch_avg_task[7]:-}") | $(fmt_cell "${v_orch_avg_task[8]:-}") |"

    echo ""
    echo "（1c）D1 -> D8：各分段项的变化速览（Delta / Delta%）"
    echo "| Metric | D1 | D8 | Delta(us) | Delta% |"
    echo "|--------|---:|---:|----------:|---------:|"
    calc_delta() { awk -v a="$2" -v b="$1" 'BEGIN{printf "%.3f", a-b}'; }
    calc_deltapct() { awk -v a="$2" -v b="$1" 'BEGIN{ if (b==0) {print "—"} else printf "%.1f", (a-b)*100/b}'; }
    for m in total complete poll otc_lock_work otc_lock_wait otc_fanout_work otc_fanout_wait otc_fanin otc_self dispatch avg_complete total_excl_wait_us; do
        case "$m" in
            total) a="${v_total[1]:-}"; b="${v_total[8]:-}";;
            complete) a="${v_complete[1]:-}"; b="${v_complete[8]:-}";;
            poll) a="${v_poll[1]:-}"; b="${v_poll[8]:-}";;
            otc_lock_work) a="${v_lock_work[1]:-}"; b="${v_lock_work[8]:-}";;
            otc_lock_wait) a="${v_lock_wait[1]:-}"; b="${v_lock_wait[8]:-}";;
            otc_fanout_work) a="${v_fanout_work[1]:-}"; b="${v_fanout_work[8]:-}";;
            otc_fanout_wait) a="${v_fanout_wait[1]:-}"; b="${v_fanout_wait[8]:-}";;
            otc_fanin) a="${v_fanin[1]:-}"; b="${v_fanin[8]:-}";;
            otc_self) a="${v_self[1]:-}"; b="${v_self[8]:-}";;
            dispatch) a="${v_dispatch[1]:-}"; b="${v_dispatch[8]:-}";;
            avg_complete) a="${v_avg_complete[1]:-}"; b="${v_avg_complete[8]:-}";;
            total_excl_wait_us) a="${v_excl_wait[1]:-}"; b="${v_excl_wait[8]:-}";;
        esac
        if [[ -z "$a" || -z "$b" ]]; then
            echo "| $m | — | — | — | — |"
        else
            d="$(calc_delta "$a" "$b")"
            p="$(calc_deltapct "$a" "$b")"
            echo "| $m | $(fmt_cell "$a") | $(fmt_cell "$b") | $d | $p% |"
        fi
    done

    echo ""
    echo "（1d）orch+sche(concurrent) 在不同 Depend 下的 Phase(Complete/Scan/Dispatch/Idle) + Pop/Fanout/Fanin（暂用 run1 单次样本；缺项标注无数据）"
    echo "| Metric | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 |"
    echo "|--------|---:|---:|---:|---:|---:|---:|---:|---:|"
    for metric in phase_complete phase_scan phase_dispatch phase_idle pop_hit pop_miss pop_hit_rate fanout_edges fanout_max fanout_avg fanin_edges fanin_max fanin_avg; do
        row="| $metric "
        for D in 1 2 3 4 5 6 7 8; do
            f=""
            [[ -f "${SWEEP_THR_DIR_P2}/throughput_Depend${D}_run1.log" ]] && f="${SWEEP_THR_DIR_P2}/throughput_Depend${D}_run1.log"
            if [[ -z "$f" ]]; then row="$row | 无数据"; continue; fi
            ov="$(parse_p2_sched_overhead_one "$f" 2>/dev/null || true)"
            if [[ -z "$ov" ]]; then row="$row | 无数据"; continue; fi
            set -- $ov
            case "$metric" in
                phase_complete) val="$1" ;;
                phase_scan) val="$2" ;;
                phase_dispatch) val="$3" ;;
                phase_idle) val="$4" ;;
                pop_hit) val="$5" ;;
                pop_miss) val="$6" ;;
                pop_hit_rate) val="$7" ;;
                fanout_edges) val="$8" ;;
                fanout_max) val="$9" ;;
                fanout_avg) val="${10}" ;;
                fanin_edges) val="${11}" ;;
                fanin_max) val="${12}" ;;
                fanin_avg) val="${13}" ;;
            esac
            [[ -z "$val" ]] && val="无数据"
            row="$row | $val"
        done
        echo "${row} |"
    done

    # 2) 对比：同一 Depend 内 concurrent vs orch-only vs sched-only
    echo ""
    echo "（2）同一用例同一 Depend：orch+sche(concurrent) vs orch-only vs sched-only 的差距（缺项也输出，标注无数据）"

    for DEP in 1 8; do
        if [[ "$DEP" == "1" ]]; then
            files_c="${OUT_DIR}/latency_p2_concurrent_run"*.log
            files_o="${OUT_DIR}/latency_p2_orch_run"*.log
            files_s="${OUT_DIR}/latency_p2_sched_run"*.log
            name="Linear"
        else
            files_c="${OUT_DIR}/throughput_p2_concurrent_run"*.log
            files_o="${OUT_DIR}/throughput_p2_orch_run"*.log
            files_s="${OUT_DIR}/throughput_p2_sched_run"*.log
            name="Depend8"
        fi

        # Convert globs to space-separated lists
        fc=""; fo=""; fs=""
        for f in $files_c; do [ -f "$f" ] && fc="$fc $f"; done
        for f in $files_o; do [ -f "$f" ] && fo="$fo $f"; done
        for f in $files_s; do [ -f "$f" ] && fs="$fs $f"; done

        row_c=""; row_s=""
        [[ -n "$fc" ]] && row_c=$(collect_phase_trimmed "$fc" 2>/dev/null || true)
        [[ -n "$fs" ]] && row_s=$(collect_phase_trimmed "$fs" 2>/dev/null || true)

        orch_c=""; orch_o=""
        [[ -n "$fc" ]] && orch_c=$(collect_orch_total_trimmed "$fc" 2>/dev/null || true)
        [[ -n "$fo" ]] && orch_o=$(collect_orch_total_trimmed "$fo" 2>/dev/null || true)

        echo ""
        echo "### ${name}：Orchestrator total 对比（去一最高一最低后均值）"
        echo "| Mode | Orchestrator thread total (us) |"
        echo "|------|-------------------------------:|"
        echo "| concurrent (orch+sche) | $(fmt_cell "$orch_c") |"
        echo "| orch-only | $(fmt_cell "$orch_o") |"
        echo "| sched-only | 无数据 |"

        echo ""
        echo "### ${name}：Scheduler Phase Breakdown 对比（去一最高一最低后均值）"
        echo "| Metric | concurrent (orch+sche) | sched-only | Delta(us) | Delta% |"
        echo "|--------|-------------------------:|------------:|----------:|-------:|"
        get_metric_from_row() {
            local row="$1"; local idx="$2"
            [[ -z "$row" ]] && echo "—" && return
            set -- $row
            echo "${!idx}"
        }

        # Row列顺序：
        # 1 total, 2 complete, 3 poll, 4 otc_lock_work, 5 otc_lock_wait,
        # 6 otc_fanout_work, 7 otc_fanout_wait, 8 otc_fanin, 9 otc_self,
        # 10 dispatch, 11 avg_complete, 12 total_excl_wait
        # 注：此处直接从已解析的 row_c/row_s 取 idx（用 set 逐项展开）。
        for m in total complete poll otc_lock_work otc_lock_wait otc_fanout_work otc_fanout_wait otc_fanin otc_self dispatch avg_complete total_excl_wait; do
            case "$m" in
                total) idx=1 ;;
                complete) idx=2 ;;
                poll) idx=3 ;;
                otc_lock_work) idx=4 ;;
                otc_lock_wait) idx=5 ;;
                otc_fanout_work) idx=6 ;;
                otc_fanout_wait) idx=7 ;;
                otc_fanin) idx=8 ;;
                otc_self) idx=9 ;;
                dispatch) idx=10 ;;
                avg_complete) idx=11 ;;
                total_excl_wait) idx=12 ;;
            esac
            # Extract numeric by re-expanding row to positional params.
            tc=""; ts=""
            if [[ -n "$row_c" ]]; then
                set -- $row_c
                # idx -> $1..$12
                eval tc=\${$idx}
            fi
            if [[ -n "$row_s" ]]; then
                set -- $row_s
                eval ts=\${$idx}
            fi
            if [[ "$tc" =~ ^[0-9.]+$ && "$ts" =~ ^[0-9.]+$ ]]; then
                dd=$(echo "scale=3; $ts - $tc" | bc 2>/dev/null || echo "")
                pp=$(echo "scale=1; if ($tc==0) 0 else ($ts-$tc)*100/$tc" | bc 2>/dev/null || echo "")
                echo "| $m | $(fmt_cell "$tc") | $(fmt_cell "$ts") | $dd | ${pp}% |"
            else
                echo "| $m | $(fmt_cell "$tc") | $(fmt_cell "$ts") | 无数据 | 无数据 |"
            fi
        done

        echo ""
        echo "### ${name}：Phase(Complete/Scan/Dispatch/Idle) + Pop/Fanout/Fanin 对比（暂用 run1 单次样本；缺项标注无数据）"
        echo "| Metric | concurrent | sched-only |"
        echo "|--------|-----------:|----------:|"
        # pick run1 files
        if [[ "$DEP" == "1" ]]; then
            f_c="${OUT_DIR}/latency_p2_concurrent_run1.log"
            f_s="${OUT_DIR}/latency_p2_sched_run1.log"
        else
            f_c="${OUT_DIR}/throughput_p2_concurrent_run1.log"
            f_s="${OUT_DIR}/throughput_p2_sched_run1.log"
        fi
        ov_c=""; ov_s=""
        [[ -f "$f_c" ]] && ov_c=$(parse_p2_sched_overhead_one "$f_c" 2>/dev/null || true)
        [[ -f "$f_s" ]] && ov_s=$(parse_p2_sched_overhead_one "$f_s" 2>/dev/null || true)
        for metric in phase_complete phase_scan phase_dispatch phase_idle pop_hit pop_miss pop_hit_rate fanout_edges fanout_max fanout_avg fanin_edges fanin_max fanin_avg; do
            vc="无数据"; vs="无数据"
            if [[ -n "$ov_c" ]]; then
                set -- $ov_c
                case "$metric" in
                    phase_complete) vc="$1" ;;
                    phase_scan) vc="$2" ;;
                    phase_dispatch) vc="$3" ;;
                    phase_idle) vc="$4" ;;
                    pop_hit) vc="$5" ;;
                    pop_miss) vc="$6" ;;
                    pop_hit_rate) vc="$7" ;;
                    fanout_edges) vc="$8" ;;
                    fanout_max) vc="$9" ;;
                    fanout_avg) vc="${10}" ;;
                    fanin_edges) vc="${11}" ;;
                    fanin_max) vc="${12}" ;;
                    fanin_avg) vc="${13}" ;;
                esac
            fi
            if [[ -n "$ov_s" ]]; then
                set -- $ov_s
                case "$metric" in
                    phase_complete) vs="$1" ;;
                    phase_scan) vs="$2" ;;
                    phase_dispatch) vs="$3" ;;
                    phase_idle) vs="$4" ;;
                    pop_hit) vs="$5" ;;
                    pop_miss) vs="$6" ;;
                    pop_hit_rate) vs="$7" ;;
                    fanout_edges) vs="$8" ;;
                    fanout_max) vs="$9" ;;
                    fanout_avg) vs="${10}" ;;
                    fanin_edges) vs="${11}" ;;
                    fanin_max) vs="${12}" ;;
                    fanin_avg) vs="${13}" ;;
                esac
            fi
            echo "| $metric | $vc | $vs |"
        done
    done
}

{
    echo "# AICPU UT 性能分析报告"
    echo ""
    echo "生成时间: $(date -Iseconds)"
    echo ""
    echo "本报告由 \`tools/all_in_one.sh\` 一键生成，包含 Linear（latency X128Y128）+ Depend1~8（throughput n=128 W=128 fix-tail D=1..8 O=D-1）对比及三种线程模式（concurrent / orch / sched）的汇总表。"
    echo "如启用 \`--profiling2\`，将额外生成 Profiling1 vs Profiling2 对比表与 Profiling2 分段时间分析。"
    echo ""
    echo "环境信息见: \`outputs/profiling_report/env.txt\`"
    echo ""
    echo "---"
    echo ""
    gen_tables
    gen_p1_p2_compare_tables
    gen_p2_phase_analysis
} > "$REPORT_FILE"

# 若有异常值（Linear + Depend1~8 去一最高一最低时排除的值），在文档末尾标注
if [[ -s "${OUT_DIR}/outliers_depend.txt" ]]; then
    {
        echo ""
        echo "---"
        echo ""
        echo "## 异常值标注"
        echo ""
        echo "Linear + Depend1~8 对比表中各样例已**去掉一个最高值、一个最低值**后计算均值与区间；下表为被排除的异常值。"
        echo ""
        echo "| 样例 | 指标 | 排除最低 | 排除最高 |"
        echo "|------|------|----------|----------|"
        while IFS='|' read -r sample metric ex_lo ex_hi; do
            echo "| $sample | $metric | $ex_lo | $ex_hi |"
        done < "${OUT_DIR}/outliers_depend.txt"
        echo ""
    } >> "$REPORT_FILE"
fi

echo ""
echo "Done. 报告已写入: $REPORT_FILE"
echo "  原始数据与 log: $OUT_DIR (env.txt, raw_data.txt, summary_stats.txt, sweep log 等)"

#!/usr/bin/env bash
# tools/gen_sweep_flamegraphs.sh
# 生成 Depend1~Depend8 的逐线程 flamegraph（O2/O0）+ Profiling1 开销分析
# 运行时自动下载 FlameGraph 到 tools/_deps/FlameGraph（该目录已在 .gitignore 中忽略）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEARCH_DIR="$SCRIPT_DIR"
while [[ "$SEARCH_DIR" != "/" && ! -f "$SEARCH_DIR/run_tests.sh" ]]; do
  SEARCH_DIR="$(dirname "$SEARCH_DIR")"
done
AICPU_UT_DIR="$SEARCH_DIR"
ROOT_DIR="$(cd "${AICPU_UT_DIR}/../.." && pwd)"
DEPS_DIR="${SCRIPT_DIR}/_deps"
FLAMEGRAPH_DIR="${DEPS_DIR}/FlameGraph"
RUN_TESTS="${AICPU_UT_DIR}/run_tests.sh"
BIN_DIR="${AICPU_UT_DIR}/build/bin"
OUT_ROOT="${ROOT_DIR}/sweep_flamegraph/outputs"

FLAMEGRAPH_REPO_URL="${FLAMEGRAPH_REPO_URL:-https://github.com/brendangregg/FlameGraph.git}"
FLAMEGRAPH_REF="${FLAMEGRAPH_REF:-master}"

PERF_SAMPLE_FREQ=${PERF_SAMPLE_FREQ:-19999}
PERF_CALLGRAPH_MODE=${PERF_CALLGRAPH_MODE:-dwarf}
PERF_CALLGRAPH_STACK=${PERF_CALLGRAPH_STACK:-16384}
SKIP_FLAMEGRAPH=${SKIP_FLAMEGRAPH:-0}
PROFILING1_REPEAT=${PROFILING1_REPEAT:-3}

COLLAPSE="${FLAMEGRAPH_DIR}/stackcollapse-perf.pl"
FLAMEGRAPH="${FLAMEGRAPH_DIR}/flamegraph.pl"

ensure_flamegraph() {
  mkdir -p "$DEPS_DIR"
  if [[ ! -f "$COLLAPSE" || ! -f "$FLAMEGRAPH" ]]; then
    echo "[deps] FlameGraph not found, cloning..."
    rm -rf "$FLAMEGRAPH_DIR"
    git clone --depth 1 --branch "$FLAMEGRAPH_REF" "$FLAMEGRAPH_REPO_URL" "$FLAMEGRAPH_DIR"
  else
    echo "[deps] FlameGraph already exists: $FLAMEGRAPH_DIR"
  fi
  [[ -f "$COLLAPSE" ]] || { echo "Error: missing $COLLAPSE" >&2; exit 1; }
  [[ -f "$FLAMEGRAPH" ]] || { echo "Error: missing $FLAMEGRAPH" >&2; exit 1; }
}

collapse_and_flamegraph() {
  local perf_file="$1"
  local out_prefix="$2"
  local title="$3"
  "$COLLAPSE" "$perf_file" > "${out_prefix}.folded" 2>/dev/null || true
  [[ -s "${out_prefix}.folded" ]] || return 0
  "$FLAMEGRAPH" --title "$title" "${out_prefix}.folded" > "${out_prefix}.svg" 2>/dev/null || true
}

split_perf_by_tid() {
  local perf_file="$1"
  local out_prefix="$2"
  awk -v prefix="$out_prefix" '
    function open_file(tid) { return prefix ".tid" tid ".perf" }
    /^[^[:space:]]+[[:space:]][0-9]+[[:space:]]/ {
      tid = $2; current_file = open_file(tid); print $0 >> current_file; next
    }
    current_file != "" { print $0 >> current_file }
  ' "$perf_file"
}

classify_thread_role() {
  local thread_perf="$1"
  if grep -qE 'memset|__memset|__memset_aarch64' "$thread_perf"; then
    echo "main"
    return
  fi
  if grep -qE 'pto2_submit_mixed_task|pto2_scope_end|build_graph' "$thread_perf"; then
    echo "Orchestrator"
    return
  fi
  if grep -qE 'aicpu_executor_sim_run_resolve_and_dispatch_pto2|PTO2SchedulerState::|dispatch_ready|on_mixed_task_complete|on_task_release' "$thread_perf"; then
    echo "Scheduler"
    return
  fi
  echo "main"
}

do_perf_and_flamegraph() {
  local bin="$1"
  local out_dir="$2"
  local title_prefix="$3"

  mkdir -p "$out_dir"
  [[ -x "$bin" ]] || { echo "  Skip (bin not executable): $bin" >&2; return 1; }

  local perf_data="${out_dir}/perf.data"
  local perf_script="${out_dir}/perf.script"

  local -a perf_args=(perf record -F "$PERF_SAMPLE_FREQ")
  case "$PERF_CALLGRAPH_MODE" in
    dwarf) perf_args+=(--call-graph "dwarf,${PERF_CALLGRAPH_STACK}") ;;
    fp)    perf_args+=(-g) ;;
    none)  ;;
    *) echo "Error: invalid PERF_CALLGRAPH_MODE=$PERF_CALLGRAPH_MODE (dwarf|fp|none)" >&2; return 1 ;;
  esac

  echo "  perf record..."
  "${perf_args[@]}" -o "$perf_data" -- "$bin" 2>&1 || true

  echo "  perf script..."
  perf script -i "$perf_data" > "$perf_script" 2>/dev/null || true
  [[ -s "$perf_script" ]] || { echo "  Warn: empty perf script: $perf_script" >&2; return 0; }

  rm -f "${out_dir}"/thread.tid*.perf "${out_dir}"/thread.tid*.folded "${out_dir}"/thread.tid*.svg 2>/dev/null || true
  split_perf_by_tid "$perf_script" "${out_dir}/thread"

  shopt -s nullglob
  local thread_perfs=("${out_dir}/thread".tid*.perf)
  shopt -u nullglob
  [[ ${#thread_perfs[@]} -gt 0 ]] || { echo "  Warn: no per-thread blocks found" >&2; return 0; }

  declare -A sched_idx=()
  local next_sched=0

  for thread_perf in "${thread_perfs[@]}"; do
    local base tid role role_name out_prefix
    base="${thread_perf##*/}"
    tid="${base#thread.tid}"
    tid="${tid%.perf}"
    role="$(classify_thread_role "$thread_perf")"
    role_name="$role"
    if [[ "$role" == "Scheduler" ]]; then
      if [[ -z "${sched_idx[$tid]:-}" ]]; then
        sched_idx[$tid]="$next_sched"
        next_sched=$((next_sched + 1))
      fi
      role_name="Scheduler${sched_idx[$tid]}"
    fi
    out_prefix="${out_dir}/${role_name}_tid${tid}"
    collapse_and_flamegraph "$thread_perf" "$out_prefix" "${title_prefix} - ${role_name} (tid ${tid})"
  done
}

build_bin() {
  local opt_level="$1"; shift
  bash "$RUN_TESTS" --opt-level "$opt_level" "$@" --build-only >/dev/null 2>&1 || true
}

extract_timing() {
  local log="$1"
  local orch_vals
  orch_vals=$(grep -oE 'cost [0-9]+\.[0-9]+us' "$log" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -20)
  local orch
  if [[ -z "$orch_vals" ]]; then
    orch="N/A"
  else
    orch=$(echo "$orch_vals" | awk '{s+=$1; n++} END { if(n>0) printf "%.1f", s/n; else print "N/A" }')
  fi
  local sched_vals
  sched_vals=$(grep -oE 'total_time=[0-9]+\.[0-9]+us' "$log" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -60)
  local sched
  if [[ -z "$sched_vals" ]]; then
    sched="N/A"
  else
    sched=$(echo "$sched_vals" | awk '{s+=$1; n++} END { if(n>0) printf "%.1f", s/n; else print "N/A" }')
  fi
  echo "$orch $sched"
}

run_profiling1_overhead() {
  local out_dir="${OUT_ROOT}/Profiling1"
  mkdir -p "$out_dir"

  echo ""
  echo "======================================================="
  echo "  Profiling1: profiling 打点开销分析（Depend8 / test_throughput O2）"
  echo "======================================================="

  echo "[build] test_throughput O2 --profiling 1 (D=8 O=7 fix-tail idx=${THROUGHPUT_IDX}) ..."
  bash "$RUN_TESTS" --opt-level 2 --test test_throughput \
    --layer-num 128 --layer0-task-num 128 --dependency 8 --overlap 7 --fix-tail --idx "$THROUGHPUT_IDX" \
    --profiling 1 --build-only 2>&1 | grep -E '^\s*(Built|Error|CMake)' || true

  local bin="${BIN_DIR}/test_throughput_concurrent_${THROUGHPUT_IDX}"
  if [[ ! -x "$bin" ]]; then
    echo "  Skip: binary not found: $bin" >&2
    return 0
  fi

  local -a freqs=(0 99 999 9999 19999)
  local -a freq_labels=("no-perf" "99Hz" "999Hz" "9999Hz" "19999Hz")
  declare -A orch_results=()
  declare -A sched_results=()

  local i
  for i in "${!freqs[@]}"; do
    local freq="${freqs[$i]}"
    local label="${freq_labels[$i]}"
    local run_log="${out_dir}/run_${label}.log"
    : > "$run_log"

    echo "  [${label}] running ${PROFILING1_REPEAT} times ..."
    local r
    for r in $(seq 1 "$PROFILING1_REPEAT"); do
      if [[ "$freq" -eq 0 ]]; then
        "$bin" >> "$run_log" 2>&1 || true
      else
        local perf_data="${out_dir}/perf_${label}_r${r}.data"
        local -a perf_args=(perf record -F "$freq")
        case "$PERF_CALLGRAPH_MODE" in
          dwarf) perf_args+=(--call-graph "dwarf,${PERF_CALLGRAPH_STACK}") ;;
          fp)    perf_args+=(-g) ;;
          none)  ;;
        esac
        perf_args+=(-o "$perf_data" -- "$bin")
        "${perf_args[@]}" >> "$run_log" 2>&1 || true
        rm -f "$perf_data" 2>/dev/null || true
      fi
    done

    local timing
    timing=$(extract_timing "$run_log")
    orch_results[$label]="${timing%% *}"
    sched_results[$label]="${timing##* }"
    echo "    orch=${orch_results[$label]} us  sched=${sched_results[$label]} us"
  done

  local report="${out_dir}/profiling_overhead_report.md"
  {
    echo "# Profiling1 打点开销分析报告"
    echo ""
    echo "测试对象：\`test_latency\`（chains=128, len=128, idx=${LATENCY_IDX}），O2 编译，\`--profiling 1\`"
    echo ""
    echo "每种条件重复 ${PROFILING1_REPEAT} 次取均值。Scheduler 列为所有 scheduler 线程 total_time 的均值。"
    echo ""
    echo "| 采样条件 | Orchestrator (us) | Scheduler avg (us) |"
    echo "|----------|------------------:|-------------------:|"
    for i in "${!freqs[@]}"; do
      local lbl="${freq_labels[$i]}"
      printf "| %-8s | %17s | %18s |\n" "$lbl" "${orch_results[$lbl]:-N/A}" "${sched_results[$lbl]:-N/A}"
    done
    echo ""
    echo "说明："
    echo "- \`no-perf\`：不挂 perf，纯 profiling 打点开销基线"
    echo "- \`99Hz\` ~ \`19999Hz\`：挂 perf record 采样，观察采样频率对 profiling 打点时间的影响"
    echo "- Orchestrator：来自 \`aicpu_orchestration_entry returned, cost NNN.NNNus\`"
    echo "- Scheduler：来自 \`Scheduler summary: total_time=NNN.NNNus\`（多线程取均值）"
    echo "- 测试样例：Depend8（test_throughput, D=8, O=7, fix-tail, n=128, W=128, idx=${THROUGHPUT_IDX}）"
    echo ""
    echo "生成时间：$(date '+%Y-%m-%d %H:%M:%S')"
  } > "$report"

  echo ""
  echo "  Report: $report"
}

run_sample() {
  local opt_level="$1"
  local opt_label="$2"
  local sample="$3"
  local test_name="$4"
  shift 4
  local -a build_args=("$@")

  local sample_dir="${OUT_ROOT}/${opt_label}/${sample}"
  mkdir -p "$sample_dir"

  echo ""
  echo "======================================================="
  echo "  Build + Perf: ${opt_label} / ${sample} / ${test_name}"
  echo "======================================================="

  echo "[build] concurrent/orch/sched (opt-level=${opt_level})..."
  build_bin "$opt_level" --test "$test_name" "${build_args[@]}"
  build_bin "$opt_level" --test "$test_name" "${build_args[@]}" --orch
  build_bin "$opt_level" --test "$test_name" "${build_args[@]}" --sched

  local idx="0"
  local mode bin
  for mode in concurrent orch sched; do
    case "$test_name" in
      test_latency)
        idx="${LATENCY_IDX}"
        case "$mode" in
          concurrent) bin="${BIN_DIR}/test_latency_concurrent_${idx}" ;;
          orch)       bin="${BIN_DIR}/test_latency_orch_only_${idx}" ;;
          sched)      bin="${BIN_DIR}/test_latency_sched_prof_only_${idx}" ;;
        esac
        ;;
      test_throughput)
        idx="${THROUGHPUT_IDX}"
        case "$mode" in
          concurrent) bin="${BIN_DIR}/test_throughput_concurrent_${idx}" ;;
          orch)       bin="${BIN_DIR}/test_throughput_orch_only_${idx}" ;;
          sched)      bin="${BIN_DIR}/test_throughput_sched_prof_only_${idx}" ;;
        esac
        ;;
      *)
        echo "Error: unknown test_name=$test_name" >&2
        return 1
        ;;
    esac

    local out_dir="${sample_dir}/${mode}"
    echo "[perf] ${opt_label}/${sample}/${mode} ..."
    do_perf_and_flamegraph "$bin" "$out_dir" "CPU Flame Graph: ${opt_label} ${sample} (${mode})"
  done
}

LATENCY_IDX=1
THROUGHPUT_IDX=0

ensure_flamegraph
[[ -x "$RUN_TESTS" ]] || { echo "Error: run_tests.sh not executable: $RUN_TESTS" >&2; exit 1; }
mkdir -p "$OUT_ROOT"

echo "Outputs: ${OUT_ROOT}/O2 and ${OUT_ROOT}/O0"
echo "Perf: F=${PERF_SAMPLE_FREQ}, callgraph=${PERF_CALLGRAPH_MODE}"

if [[ "$SKIP_FLAMEGRAPH" != "1" ]]; then
  for opt_level in 2 0; do
    opt_label="O${opt_level}"
    run_sample "$opt_level" "$opt_label" "Depend1" "test_latency" \
      --chain-num 128 --chain-length 128 --idx "$LATENCY_IDX"

    for d in 2 3 4 5 6 7 8; do
      o=$((d - 1))
      run_sample "$opt_level" "$opt_label" "Depend${d}" "test_throughput" \
        --layer-num 128 --layer0-task-num 128 --dependency "$d" --overlap "$o" --fix-tail --idx "$THROUGHPUT_IDX"
    done
  done
fi

run_profiling1_overhead
echo ""
echo "Done. Outputs under: ${OUT_ROOT}/"

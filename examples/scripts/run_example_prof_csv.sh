#!/usr/bin/env bash
# 使用 run_example.py 运行样例（默认开启 --enable-profiling），随后在 outputs/ 中查找
# 最新的 perf_swimlane_*.json，并生成任务依赖 CSV（perf_swimlane_to_task_csv.py）。
#
# 用法：
#   ./examples/scripts/run_example_prof_csv.sh -k <kernels> -g <golden.py> -p a2a3 -d 0
#   OUTPUT_CSV=/tmp/tasks.csv ./examples/scripts/run_example_prof_csv.sh ...
#
# 环境变量：
#   OUTPUT_CSV   若设置，将 CSV 写到该路径；否则与 JSON 同目录，名为 perf_swimlane_*_tasks.csv
#   SKIP_RUN       若设为 1，跳过 run_example，仅根据现有最新 JSON 生成 CSV

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

RUN_ARGS=()
HAS_PROF=0
for a in "$@"; do
  if [[ "$a" == "--enable-profiling" ]]; then
    HAS_PROF=1
  fi
  RUN_ARGS+=("$a")
done
if [[ "$HAS_PROF" -eq 0 ]]; then
  RUN_ARGS+=(--enable-profiling)
fi

if [[ "${SKIP_RUN:-0}" != "1" ]]; then
  python3 "$ROOT/examples/scripts/run_example.py" "${RUN_ARGS[@]}"
fi

OUT_CSV_ARG=()
if [[ -n "${OUTPUT_CSV:-}" ]]; then
  OUT_CSV_ARG=(-o "$OUTPUT_CSV")
fi

python3 "$ROOT/examples/scripts/perf_swimlane_to_task_csv.py" "${OUT_CSV_ARG[@]}"

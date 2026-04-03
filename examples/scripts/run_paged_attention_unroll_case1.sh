#!/usr/bin/env bash
# 运行 tensormap ST：paged_attention_unroll 的 Case1（完整约 1280 task）。
#
# 用法：
#   ./examples/scripts/run_paged_attention_unroll_case1.sh
#   ENABLE_PROFILING=1 ./examples/scripts/run_paged_attention_unroll_case1.sh
#
# 开启 ENABLE_PROFILING=1 时默认会再生成与 perf_swimlane_*_tasks.csv 相同列格式的 CSV
#（task_id, fanout_*, fanin_count, fanin_refcount 等），见 perf_swimlane_to_task_csv.py。
#
# 环境变量：
#   ENABLE_PROFILING=1     打开 profiling，写出 outputs/perf_swimlane_*.json
#   EXPORT_TASK_CSV=1      跑完后从最新 JSON 生成 CSV（默认在 ENABLE_PROFILING=1 时开启；设为 0 可跳过）
#   OUTPUT_CSV=/path/x.csv  指定 CSV 路径；不设置则写入 outputs/perf_swimlane_<时间戳>_tasks.csv
#   PERF_JSON=/path/x.json  指定用于生成 CSV 的 JSON（不设置则用 outputs/ 下最新的 perf_swimlane_*.json）
#   MIN_PERF_TASKS=N       若 JSON 中 tasks 条数 < N 则失败（默认 1000；Case1 完整约 1280；调试用 0 关闭）
#   DEVICE / PLATFORM      同前
#
# 在仓库根目录 simpler/ 下执行（脚本会自动 cd 到该目录）。

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

DEVICE="${DEVICE:-0}"
PLATFORM="${PLATFORM:-a2a3}"

KERNELS="${ROOT}/tests/st/a2a3/tensormap_and_ringbuffer/paged_attention_unroll/kernels"
GOLDEN="${ROOT}/tests/st/a2a3/tensormap_and_ringbuffer/paged_attention_unroll/golden.py"

EXTRA=()
if [[ "${ENABLE_PROFILING:-0}" == "1" ]]; then
  EXTRA+=(--enable-profiling)
fi

python3 "${ROOT}/examples/scripts/run_example.py" \
  -k "${KERNELS}" \
  -g "${GOLDEN}" \
  -p "${PLATFORM}" \
  -d "${DEVICE}" \
  --case Case1 \
  "${EXTRA[@]}" \
  "$@"

# 与 perf_swimlane_*_tasks.csv 相同格式：由 perf_swimlane_to_task_csv.py 生成
if [[ "${ENABLE_PROFILING:-0}" == "1" ]] && [[ "${EXPORT_TASK_CSV:-1}" == "1" ]]; then
  CSV_ARGS=()
  if [[ -n "${OUTPUT_CSV:-}" ]]; then
    CSV_ARGS+=(-o "${OUTPUT_CSV}")
  fi
  MIN_PERF_TASKS="${MIN_PERF_TASKS:-1000}"
  if [[ "${MIN_PERF_TASKS}" != "0" ]]; then
    CSV_ARGS+=(--min-tasks "${MIN_PERF_TASKS}")
  fi
  if [[ -n "${PERF_JSON:-}" ]]; then
    CSV_ARGS+=("${PERF_JSON}")
  fi
  python3 "${ROOT}/examples/scripts/perf_swimlane_to_task_csv.py" "${CSV_ARGS[@]}"
fi

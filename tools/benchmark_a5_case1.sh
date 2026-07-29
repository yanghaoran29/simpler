#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# Ablation entry for PR #906 × main: a5 paged_attention_unroll Case1 only.
# Thin wrapper around tools/benchmark_rounds.sh with a5 sample map + BENCH_ONLY_EXAMPLE.
#
# Usage:
#   ./tools/benchmark_a5_case1.sh [-d <device>] [-n <rounds>] [-v]
#   task-submit --device auto --run './tools/benchmark_a5_case1.sh -d $TASK_DEVICE -n 100'

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

DEVICE_ID="${TASK_DEVICE:-0}"
ROUNDS=100
VERBOSE_FLAG=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--device) DEVICE_ID="$2"; shift 2 ;;
        -n|--rounds) ROUNDS="$2"; shift 2 ;;
        -v|--verbose) VERBOSE_FLAG=(-v); shift ;;
        -h|--help)
            sed -n '2,12p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown arg: $1" >&2
            exit 1
            ;;
    esac
done

export BENCH_ONLY_EXAMPLE=paged_attention_unroll
exec "$SCRIPT_DIR/benchmark_rounds.sh" \
    -p a5 -d "$DEVICE_ID" -n "$ROUNDS" -r tensormap_and_ringbuffer \
    "${VERBOSE_FLAG[@]}"

#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# perf_sched.sh — 对 build/bin 中任意二进制做 perf record
#
# 原理：
#   test_batch_paged_attention* 系列二进制均支持 PERF_WAIT_AFTER_INIT=1：
#     init 完成后自我暂停（SIGSTOP），脚本 attach perf record -p，再 SIGCONT 恢复；
#     采样窗口精确覆盖工作阶段，排除 init 噪音。
#   其他二进制直接启动，perf record 全程跟踪。
#
# 用法：
#   ./perf_sched.sh --bin test_batch_paged_attention_sched_prof_only_0
#   ./perf_sched.sh --bin test_batch_paged_attention_0 --build          # 先构建再采样
#   ./perf_sched.sh --bin test_batch_paged_attention_sched_prof_only_1 --sched-threads 5
#   ./perf_sched.sh --bin test_batch_paged_attention_0 --output my.data
#   ./perf_sched.sh --bin test_batch_paged_attention_0 --report         # 采样后立即打开 perf report
#   ./perf_sched.sh --bin test_batch_paged_attention_0 --profiling      # 开启 profiling 插桩（默认关闭）
#   ./perf_sched.sh --bin test_batch_paged_attention_0 --call-graph fp  # 使用帧指针展开（默认 dwarf）
#   ./perf_sched.sh --bin test_batch_paged_attention_0 --call-graph lbr # 使用 LBR 展开（仅 x86）
#   ./perf_sched.sh --bin test_batch_paged_attention_0 --dwarf-size 131072  # 增大 dwarf 栈快照
# ─────────────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEARCH_DIR="$SCRIPT_DIR"
while [[ "$SEARCH_DIR" != "/" && ! -f "$SEARCH_DIR/run_tests.sh" ]]; do
    SEARCH_DIR="$(dirname "$SEARCH_DIR")"
done
AICPU_UT_DIR="$SEARCH_DIR"

# ─── 参数默认值 ───────────────────────────────────────────────────────────────
BIN_NAME=""            # 必填，通过 --bin 指定
SCHED_THREADS=3
OUTPUT=""
DO_BUILD=false
DO_REPORT=false
PROFILING=false        # 默认关闭 profiling，避免插桩影响 perf 测量
CALL_GRAPH=dwarf       # 默认 dwarf 展开，避免 [unknown]（fp 需各库均保留帧指针）
DWARF_SIZE=65528       # dwarf 栈快照大小（字节），perf 最大值为 65528
PERF_ATTACH_WAIT=0.3   # 等待 perf attach 完成的秒数

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bin)
            BIN_NAME="$2"; shift 2 ;;
        --sched-threads)
            SCHED_THREADS="$2"; shift 2 ;;
        --output)
            OUTPUT="$2"; shift 2 ;;
        --build)
            DO_BUILD=true; shift ;;
        --report)
            DO_REPORT=true; shift ;;
        --profiling)
            PROFILING=true; shift ;;
        --call-graph)
            CALL_GRAPH="$2"; shift 2 ;;
        --dwarf-size)
            DWARF_SIZE="$2"; shift 2 ;;
        --help|-h)
            sed -n '2,20p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Use --help for usage." >&2
            exit 1 ;;
    esac
done

BUILD_DIR="${AICPU_UT_DIR}/build"

if [[ -z "$BIN_NAME" ]]; then
    echo "Error: --bin <binary_name> is required." >&2
    echo "Use --help for usage." >&2
    exit 1
fi
BIN="${BUILD_DIR}/bin/${BIN_NAME}"
OUTPUT="${OUTPUT:-${BUILD_DIR}/perf.${BIN_NAME}.data}"

# test_batch_paged_attention* 支持 PERF_WAIT_AFTER_INIT，其他二进制全程采样
if [[ "$BIN_NAME" == test_batch_paged_attention* ]]; then
    USE_WAIT_AFTER_INIT=true
else
    USE_WAIT_AFTER_INIT=false
fi

# ─── 构建 ─────────────────────────────────────────────────────────────────────
if $DO_BUILD; then
    echo ""
    echo "============================================================"
    echo "  Building all binaries"
    echo "============================================================"
    PROFILING_MODE=0
    if $PROFILING; then
        PROFILING_MODE=2
    fi
    bash "${AICPU_UT_DIR}/run_tests.sh" \
        --build-only \
        --sched-threads "$SCHED_THREADS" \
        --profiling "$PROFILING_MODE" \
        2>&1 | awk 'NF'
fi

if [[ ! -x "$BIN" ]]; then
    echo "Binary not found: $BIN" >&2
    echo "Available binaries:" >&2
    ls "${BUILD_DIR}/bin/" 2>/dev/null | sed 's/^/  /' >&2
    exit 1
fi

# ─── 运行 ─────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Perf profiling: ${BIN_NAME}"
echo "  sched_threads=${SCHED_THREADS}  wait_after_init=${USE_WAIT_AFTER_INIT}"
echo "  Output: ${OUTPUT}"
echo "============================================================"
echo ""

CALL_GRAPH_ARG="$CALL_GRAPH"
[[ "$CALL_GRAPH" == "dwarf" ]] && CALL_GRAPH_ARG="dwarf,${DWARF_SIZE}"

if $USE_WAIT_AFTER_INIT; then
    # ── SIGSTOP/SIGCONT：init 后暂停，精确覆盖工作阶段 ──────────────────────
    PERF_WAIT_AFTER_INIT=1 AICPU_UT_NUM_SCHED_THREADS="$SCHED_THREADS" \
        "$BIN" &
    BIN_PID=$!

    # 等待进程到达 SIGSTOP（状态变为 T）
    for i in $(seq 1 100); do
        sleep 0.1
        state=$(awk '{print $3}' /proc/${BIN_PID}/stat 2>/dev/null) || {
            echo "Process exited before reaching SIGSTOP." >&2; exit 1
        }
        [[ "$state" == "T" ]] && break
        if [[ $i -eq 100 ]]; then
            echo "Timeout waiting for process to reach SIGSTOP." >&2
            kill "$BIN_PID" 2>/dev/null || true
            exit 1
        fi
    done

    echo "  Process paused (PID=$BIN_PID). Attaching perf record..."
    perf record --call-graph "$CALL_GRAPH_ARG" -p "$BIN_PID" -o "$OUTPUT" &
    PERF_PID=$!

    sleep "$PERF_ATTACH_WAIT"

    echo "  Sending SIGCONT — Scheduler starting..."
    kill -CONT "$BIN_PID"

    wait "$BIN_PID" || true
    kill -INT "$PERF_PID" 2>/dev/null || true
    wait "$PERF_PID" 2>/dev/null || true
else
    # ── 其他二进制：全程采样 ──────────────────────────────────────────────────
    perf record --call-graph "$CALL_GRAPH_ARG" -o "$OUTPUT" \
        -- env AICPU_UT_NUM_SCHED_THREADS="$SCHED_THREADS" "$BIN"
fi

echo ""
echo "  perf data saved to: ${OUTPUT}"
echo ""

# ─── 可选：直接打开报告 ───────────────────────────────────────────────────────
if $DO_REPORT; then
    perf report -i "$OUTPUT"
fi

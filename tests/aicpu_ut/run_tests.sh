#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_tests.sh — Build and run orchestration unit tests
#
# Usage:
#   ./run_tests.sh                                 # run batch_paged_attention perf tests only (idx 0), skip test_paged_attention
#   ./run_tests.sh --all                           # run all batch perf tests (all indices)
#   ./run_tests.sh --func                          # functional tests only (explicit)
#   ./run_tests.sh --func --perf                   # run all (functional + perf)
#   ./run_tests.sh --test <name>                   # run one test (all parameter sets)
#   ./run_tests.sh --test <name> --idx <n>         # run one specific parameter set
#   ./run_tests.sh --test <name> --idx <n> --sched-threads 4   # 4 scheduler threads (perf tests)
#   ./run_tests.sh --test test_throughput --layer-num 10 --dependency 6 --overlap 5 --layer0-task-num 320  # 分层 DAG，默认 n=10 layer0=320 D=6 O=5；运行全部三个样例
#   ./run_tests.sh --test test_throughput --dependency 4 --overlap 3 --fix-tail  # D-O=1 时每层任务数固定为 layer0_size
#   ./run_tests.sh --test test_throughput --idx 0       # 样例 0：所有任务均为 AIV
#   ./run_tests.sh --test test_throughput --idx 1       # 样例 1：奇数编号任务为 AIC，偶数为 AIV
#   ./run_tests.sh --test test_throughput --idx 2       # 样例 2：每层前半为 AIC，后半为 AIV
#   ./run_tests.sh --test test_latency [--idx 0] --chain-num N --chain-length L  # 编译期写入 (N≤128, L≤256)
#   ./run_tests.sh --orch                          # run orch-only variant of all perf tests
#   ./run_tests.sh --sched                         # run sched-only variant of all perf tests
#   ./run_tests.sh --test <name> --orch            # run orch-only variant of one test
#   ./run_tests.sh --test <name> --sched           # run sched-only variant of one test
#   ./run_tests.sh --no-early-return               # drain before break/return so completed==consumed, fanin==fanout
#   ./run_tests.sh --build-only                    # build without running
#   ./run_tests.sh --opt-level 0                   # compile with -O0 (debug, no optimization)
#   AICPU_UT_PROFILE_CALLSTACK=ON ./run_tests.sh   # keep sibling-call opts off for better perf stacks
#   ./run_tests.sh --profiling 0                    # 不开启 profiling（默认）
#   ./run_tests.sh --profiling 1                    # 仅开启 PTO2_PROFILING（基础 profiling，不含 Sched/Orch 明细）
#   ./run_tests.sh --profiling 2                    # 完整 profiling（Sched+Orch 明细）；stdout 经 format_profiling_output 整理
#   ./run_tests.sh --profiling 2 --swimlane        # 完整 profiling + 将 swimlane JSON 转为 Perfetto + Mermaid
#   ./run_tests.sh --no-check                        # skip P1/P2 invariant checks (AICPU_UT_NO_CHECK=1)
#   ./run_tests.sh --count-submit-task-instructions --test test_batch_paged_attention --idx 0
#     # 统计 pto2_submit_mixed_task 标记区间动态指令数（调用 tools/count_between_markers.sh）
#   ./run_tests.sh --list                          # list all available tests
#
# Available tests:
#   test_cpu_affinity               (functional)
#   test_platform_config            (functional)
#   test_paged_attention            (perf, indices: 0)
#   test_batch_paged_attention      (perf, indices: 0 1 2; --orch: orch-only, --sched: sched profiling only)
#   test_alt                        (perf, alternating matmul+add, indices: 0 1; --orch: orch-only, --sched: sched profiling only)
#   test_bgemm                      (perf, batched GEMM, indices: 0 1 2 3 4; --orch: orch-only, --sched: sched profiling only)
#   test_pau                        (perf, paged attention unroll, indices: 0 1 2; --orch: orch-only, --sched: sched profiling only)
#   test_throughput                 (perf, 分层 DAG 吞吐, indices: 0 1 2; idx=0 all-AIV, idx=1 odd-AIC/even-AIV, idx=2 half-AIC/half-AIV per layer; --layer-num n --layer0-task-num W --dependency D --overlap O, 默认 n=10 layer0=320 D=6 O=5; --orch/--sched)
#   test_latency                    (perf, 极限延迟; indices: 0 1; idx=0 all-AIV, idx=1 aic/aiv alternate; 默认 chains=64 len=64; --chain-num/--chain-length/--latency-nelems 在编译期写入二进制; --orch/--sched)
#
# Optional environment overrides:
#   TIMEOUT=300 ./run_tests.sh
#   ORCH_CPU=4 SCHED_CPU0=5 ./run_tests.sh
#   PLATFORM_MAX_BLOCKDIM=32 ./run_tests.sh
#   BUILD_DIR=/tmp/my_build ./run_tests.sh
#   ./run_tests.sh --profiling 2   # 或通过环境变量覆盖：PTO2_PROFILING=ON 等
#   AICPU_UT_DEVICE_ID=8 ./run_tests.sh   # device id for sched_overhead_analysis (resolve device log)
#   ./run_tests.sh --test test_batch_paged_attention --idx 0 --sched-threads 4   # 4 scheduler threads
#   AICPU_UT_THROUGHPUT_LAYERS=5 AICPU_UT_THROUGHPUT_DEPS=4 AICPU_UT_THROUGHPUT_OVERLAP=2 ./run_tests.sh --test test_throughput  # 分层 DAG 环境变量覆盖
#   ./run_tests.sh --test test_latency --chain-num 4 --chain-length 8  # 极限延迟：命令行指定 chains/chain_length
# ─────────────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORIG_ARGS=("$@")

# ─── Test Registry ────────────────────────────────────────────────────────────
# TEST_TYPE:    "func" or "perf"
# TEST_INDICES: "-" for single-binary tests; space-separated indices for multi-case tests
declare -A TEST_TYPE
declare -A TEST_INDICES

TEST_TYPE["test_cpu_affinity"]="func"            ; TEST_INDICES["test_cpu_affinity"]="-"
TEST_TYPE["test_platform_config"]="func"         ; TEST_INDICES["test_platform_config"]="-"
TEST_TYPE["test_paged_attention"]="perf"         ; TEST_INDICES["test_paged_attention"]="0"
TEST_TYPE["test_batch_paged_attention"]="perf"   ; TEST_INDICES["test_batch_paged_attention"]="0 1 2"
TEST_TYPE["test_alt"]="perf"                     ; TEST_INDICES["test_alt"]="0 1"
TEST_TYPE["test_bgemm"]="perf"                   ; TEST_INDICES["test_bgemm"]="0 1 2 3 4"
TEST_TYPE["test_pau"]="perf"                     ; TEST_INDICES["test_pau"]="0 1 2"
TEST_TYPE["test_throughput"]="perf"              ; TEST_INDICES["test_throughput"]="0 1 2"
TEST_TYPE["test_latency"]="perf"                ; TEST_INDICES["test_latency"]="0 1"

ALL_TESTS=(test_cpu_affinity test_platform_config test_paged_attention test_batch_paged_attention test_alt test_bgemm test_pau test_throughput test_latency)

# ─── Defaults ─────────────────────────────────────────────────────────────────
TIMEOUT=${TIMEOUT:-600}
BUILD_DIR=${BUILD_DIR:-"${SCRIPT_DIR}/build"}

ORCH_CPU=${ORCH_CPU:-4}
SCHED_CPU0=${SCHED_CPU0:-8}
SCHED_CPU1=${SCHED_CPU1:-9}
SCHED_CPU2=${SCHED_CPU2:-10}
SCHED_CPU3=${SCHED_CPU3:-11}
SCHED_CPU4=${SCHED_CPU4:-12}
SCHED_CPU5=${SCHED_CPU5:-13}
SCHED_CPU6=${SCHED_CPU6:-14}
SCHED_CPU7=${SCHED_CPU7:-15}

PLATFORM_MAX_BLOCKDIM=${PLATFORM_MAX_BLOCKDIM:-24}
PLATFORM_AIC_CORES_PER_BLOCKDIM=${PLATFORM_AIC_CORES_PER_BLOCKDIM:-1}
PLATFORM_AIV_CORES_PER_BLOCKDIM=${PLATFORM_AIV_CORES_PER_BLOCKDIM:-2}
PLATFORM_MAX_AICPU_THREADS=${PLATFORM_MAX_AICPU_THREADS:-4}

# Profiling: --profiling <0|1|2>
#   0 = 不开启
#   1 = 仅开启 PTO2_PROFILING（基础 profiling，不含 Sched/Orch 明细）
#   2 = 完整 profiling（PTO2_SCHED_PROFILING + PTO2_ORCH_PROFILING + PTO2_TENSORMAP_PROFILING）
# 未传 --profiling 时默认 0。传 --profiling 但未写数字时默认 2（兼容旧用法）。
PROFILING_MODE=${PROFILING_MODE:-0}

# Default: run perf only = batch_paged_attention* (idx 0). Use --all for all indices; use --test test_paged_attention to run single paged_attention.
RUN_FUNC=false
RUN_PERF=true
RUN_ALL_INDICES=false
# Perf tests run by default (excludes test_paged_attention)
# 默认 perf 集合中暂时跳过 test_bgemm；如需运行可显式 --test test_bgemm
DEFAULT_PERF_TESTS=(test_batch_paged_attention test_alt test_pau test_throughput test_latency)
BUILD_ONLY=false
SKIP_FINAL_ANALYSIS=false
OPT_LEVEL=${OPT_LEVEL:-3}
AICPU_UT_PROFILE_CALLSTACK=${AICPU_UT_PROFILE_CALLSTACK:-ON}
FILTER_TEST=""
FILTER_IDX=""
THREAD_MODE="concurrent"        # set by --orch (orch-only) or --sched (sched profiling only); default: both threads
AICPU_UT_NUM_SCHED_THREADS=""   # set by --sched-threads N (default: test binary uses 3)
AICPU_UT_THROUGHPUT_LAYERS=""       # set by --layer-num n (层数, default in case: 10)
AICPU_UT_THROUGHPUT_LAYER0_SIZE=""  # set by --layer0-task-num W (第一层任务数, default in case: 320)
AICPU_UT_THROUGHPUT_DEPS=""         # set by --dependency D (依赖数, default: 6)
AICPU_UT_THROUGHPUT_OVERLAP=""     # set by --overlap O (重叠数, default: 5)
AICPU_UT_THROUGHPUT_FIX_TAIL=""   # set by --fix-tail: 仅当 D-O=1 时生效，每层任务数固定为 layer0_size
AICPU_UT_LATENCY_NUM_CHAINS=""       # set by --chain-num N (override num_chains for test_latency, max 128)
AICPU_UT_LATENCY_CHAIN_LENGTH=""     # set by --chain-length L (override chain_length for test_latency, max 256)
AICPU_UT_NO_EARLY_RETURN=""     # set by --no-early-return: drain before break/return so completed==consumed
AICPU_UT_NO_CHECK=""            # set by --no-check: skip P1/P2 invariant checks in test binaries
GEN_SWIMLANE=false              # set by --swimlane: convert perf_swimlane_*.json after run
COUNT_SUBMIT_TASK_INSNS=0       # set by --count-submit-task-instructions

# ─── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --func)         RUN_PERF=false; shift ;;
        --perf)         RUN_FUNC=false; shift ;;
        --all)          RUN_ALL_INDICES=true; shift ;;
        --orch)         THREAD_MODE="orch"; shift ;;
        --sched)        THREAD_MODE="sched"; shift ;;
        --build-only)   BUILD_ONLY=true; shift ;;
        --opt-level)
            if [[ -z "${2:-}" ]]; then
                echo "--opt-level requires a numeric argument (0/1/2/3)." >&2; exit 1
            fi
            OPT_LEVEL="$2"; shift 2 ;;
        --profiling)
            if [[ -n "${2:-}" && "$2" =~ ^[012]$ ]]; then
                PROFILING_MODE="$2"; shift 2
            else
                # 未写数字时默认 2（兼容旧用法 --profiling）
                PROFILING_MODE=2; shift
            fi ;;
        --test)
            if [[ -z "${2:-}" ]]; then
                echo "--test requires a test name argument." >&2; exit 1
            fi
            FILTER_TEST="$2"; shift 2 ;;
        --idx)
            if [[ -z "${2:-}" ]]; then
                echo "--idx requires a numeric argument." >&2; exit 1
            fi
            FILTER_IDX="$2"; shift 2 ;;
        --sched-threads)
            if [[ -z "${2:-}" ]]; then
                echo "--sched-threads requires a numeric argument (e.g. 3 or 4)." >&2; exit 1
            fi
            AICPU_UT_NUM_SCHED_THREADS="$2"; shift 2 ;;
        --layer-num)
            if [[ -z "${2:-}" ]]; then
                echo "--layer-num requires a numeric argument (e.g. 3, layer count)." >&2; exit 1
            fi
            AICPU_UT_THROUGHPUT_LAYERS="$2"; shift 2 ;;
        --layer0-task-num)
            if [[ -z "${2:-}" ]]; then
                echo "--layer0-task-num requires a numeric argument (e.g. 10, first layer size)." >&2; exit 1
            fi
            AICPU_UT_THROUGHPUT_LAYER0_SIZE="$2"; shift 2 ;;
        --dependency)
            if [[ -z "${2:-}" ]]; then
                echo "--dependency requires a numeric argument (e.g. 4, deps per task)." >&2; exit 1
            fi
            AICPU_UT_THROUGHPUT_DEPS="$2"; shift 2 ;;
        --overlap)
            if [[ -z "${2:-}" ]]; then
                echo "--overlap requires a numeric argument (e.g. 2, overlap count)." >&2; exit 1
            fi
            AICPU_UT_THROUGHPUT_OVERLAP="$2"; shift 2 ;;
        --fix-tail)
            AICPU_UT_THROUGHPUT_FIX_TAIL=1; shift ;;
        --throughput-nelems)
            echo "--throughput-nelems is no longer supported; tensor_nelems is fixed to 1." >&2; exit 1 ;;
        --chain-num)
            if [[ -z "${2:-}" ]]; then
                echo "--chain-num requires a numeric argument (e.g. 4, max 128)." >&2; exit 1
            fi
            AICPU_UT_LATENCY_NUM_CHAINS="$2"; shift 2 ;;
        --chain-length)
            if [[ -z "${2:-}" ]]; then
                echo "--chain-length requires a numeric argument (e.g. 8, max 256)." >&2; exit 1
            fi
            AICPU_UT_LATENCY_CHAIN_LENGTH="$2"; shift 2 ;;
        --latency-nelems)
            echo "--latency-nelems is no longer supported; tensor_nelems is fixed to 1." >&2; exit 1 ;;
        --no-early-return)
            AICPU_UT_NO_EARLY_RETURN=1; shift ;;
        --no-check)
            AICPU_UT_NO_CHECK=1; shift ;;
        --swimlane)
            GEN_SWIMLANE=true; shift ;;
        --count-submit-task-instructions)
            COUNT_SUBMIT_TASK_INSNS=1; shift ;;
        --list)
            echo "Available tests (use --orch or --sched to select thread mode for perf tests):"
            for name in "${ALL_TESTS[@]}"; do
                t="${TEST_TYPE[$name]}"
                idx="${TEST_INDICES[$name]}"
                if [ "$idx" = "-" ]; then
                    printf "  %-36s (%s)\n" "$name" "$t"
                else
                    printf "  %-36s (%s, indices: %s)\n" "$name" "$t" "$idx"
                fi
            done
            exit 0 ;;
        --help|-h)
            sed -n '2,50p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Use --help for usage." >&2
            exit 1 ;;
    esac
done

if [ "${COUNT_SUBMIT_TASK_INSNS:-0}" = 1 ]; then
    tool_script="${SCRIPT_DIR}/tools/count_between_markers.sh"
    plugin_so="${SCRIPT_DIR}/plugins/libinsn_count.so"
    plugin_dir="${SCRIPT_DIR}/plugins"
    if [ ! -x "$tool_script" ]; then
        echo "[run_tests] missing executable tool: $tool_script" >&2
        echo "[run_tests] run: chmod +x ${SCRIPT_DIR}/tools/count_between_markers.sh" >&2
        exit 1
    fi
    # Auto-build qemu plugin when missing, so users don't need a manual make step.
    if [ ! -f "$plugin_so" ]; then
        if [ ! -d "$plugin_dir" ]; then
            echo "[run_tests] missing plugin directory: $plugin_dir" >&2
            exit 1
        fi
        echo "[run_tests] plugin missing, auto-building: $plugin_so"
        if [ -n "${QEMU_BUILD_DIR:-}" ]; then
            make -C "$plugin_dir" QEMU_BUILD_DIR="$QEMU_BUILD_DIR"
        else
            make -C "$plugin_dir"
        fi
        if [ ! -f "$plugin_so" ]; then
            echo "[run_tests] plugin build failed: $plugin_so not found" >&2
            exit 1
        fi
    fi
    mode_arg="concurrent"
    case "${THREAD_MODE}" in
        orch) mode_arg="orch" ;;
        sched) mode_arg="sched" ;;
        *) mode_arg="concurrent" ;;
    esac
    test_arg="${FILTER_TEST:-test_batch_paged_attention}"
    idx_arg="${FILTER_IDX:-0}"
    "$tool_script" --test "$test_arg" --idx "$idx_arg" --thread-mode "$mode_arg"
    exit $?
fi

# When --sched-threads N is used, ensure PLATFORM_MAX_AICPU_THREADS >= N so the build supports N threads
if [ -n "${AICPU_UT_NUM_SCHED_THREADS:-}" ] && [ "$AICPU_UT_NUM_SCHED_THREADS" -gt "${PLATFORM_MAX_AICPU_THREADS:-0}" ] 2>/dev/null; then
    PLATFORM_MAX_AICPU_THREADS=$AICPU_UT_NUM_SCHED_THREADS
fi

# ─── Validate --test / --idx ──────────────────────────────────────────────────
if [ -n "$FILTER_TEST" ]; then
    if [ -z "${TEST_TYPE[$FILTER_TEST]+x}" ]; then
        echo "Unknown test: '$FILTER_TEST'" >&2
        echo "Use --list to see available tests." >&2
        exit 1
    fi
    if [ -n "$FILTER_IDX" ]; then
        if [ "${TEST_INDICES[$FILTER_TEST]}" = "-" ]; then
            echo "Test '$FILTER_TEST' is a single-binary test; --idx is not applicable." >&2
            exit 1
        fi
        if ! echo " ${TEST_INDICES[$FILTER_TEST]} " | grep -qw "$FILTER_IDX"; then
            echo "Index '$FILTER_IDX' is not valid for test '$FILTER_TEST'." >&2
            echo "  Valid indices: ${TEST_INDICES[$FILTER_TEST]}" >&2
            exit 1
        fi
    fi
fi

# 根据 PROFILING_MODE 设置 PTO2_*：0=不开启，1=仅 PTO2_PROFILING（基础），2=完整 profiling
case "$PROFILING_MODE" in
    0) PTO2_PROFILING=OFF; PTO2_SCHED_PROFILING=OFF; PTO2_ORCH_PROFILING=OFF; PTO2_TENSORMAP_PROFILING=OFF ;;
    1) PTO2_PROFILING=ON;  PTO2_SCHED_PROFILING=OFF; PTO2_ORCH_PROFILING=OFF; PTO2_TENSORMAP_PROFILING=OFF ;;
    2) PTO2_PROFILING=ON;  PTO2_SCHED_PROFILING=ON;  PTO2_ORCH_PROFILING=ON;  PTO2_TENSORMAP_PROFILING=ON  ;;
    *) echo "Invalid PROFILING_MODE: $PROFILING_MODE (must be 0, 1, or 2)." >&2; exit 1 ;;
esac
if [ "$PTO2_PROFILING" = "ON" ]; then
    AICPU_UT_QUIET=
else
    AICPU_UT_QUIET=1   # profiling off: only output pass/fail summary
fi

# profiling 1: re-exec through format_profiling_output.py for unified formatting.
# profiling 2: raw output, no reformatting.
if [ "$PROFILING_MODE" = "1" ] && [ -z "${AICPU_UT_FORMATTED:-}" ]; then
    export AICPU_UT_FORMATTED=1
    exec bash "$0" "${ORIG_ARGS[@]}" 2>&1 | python3 "$SCRIPT_DIR/tools/format_profiling_output.py"
    exit 0
fi

# ─── Helpers ──────────────────────────────────────────────────────────────────
# When AICPU_UT_QUIET=1 (profiling off), only pass/fail summary is printed; skip this echo.
# Must return 0 when skipping so set -e does not exit: use || true.
quiet_echo() { [ -z "${AICPU_UT_QUIET:-}" ] && echo "$@" || true; }

# 60-char separator to match test binary output
print_separator() {
    quiet_echo "============================================================"
}

run_binary() {
    local bin="$1"
    local label="$2"
    shift 2
    if [ -n "${AICPU_UT_QUIET:-}" ]; then
        # profiling off: run silently, only exit code matters for pass/fail count
        if timeout "$TIMEOUT" "$bin" "$@" >/dev/null 2>&1; then
            return 0
        else
            return $?
        fi
    fi
    quiet_echo ""
    print_separator
    quiet_echo "  Running: $label"
    print_separator
    if [ -n "${SIM_LOG:-}" ]; then
        # Capture to temp file then cat to terminal and append to SIM_LOG, so profiling output is always visible
        # (piping directly to tee can leave terminal empty due to buffering in some environments)
        local tmp_log sanitized timestamp log_file
        tmp_log=$(mktemp)
        if [ -n "${LOG_DIR:-}" ]; then
            sanitized=$(echo "$label" | sed 's/\[/ /g; s/\]/ /g; s/(perf)//; s/(func)//; s/  */_/g; s/^_//; s/_$//')
            timestamp=$(date +%Y%m%d_%H%M%S)
            if [ -n "${AICPU_UT_PHASE_LOG:-}" ]; then
                export AICPU_UT_PHASE_LOG="${LOG_DIR}/${sanitized}_phase_${timestamp}.log"
            fi
        fi
        if timeout "$TIMEOUT" "$bin" "$@" >"$tmp_log" 2>&1; then
            cat "$tmp_log"
            cat "$tmp_log" >> "$SIM_LOG"
            if [ -n "${LOG_DIR:-}" ]; then
                log_file="${LOG_DIR}/${sanitized}_${timestamp}.log"
                cp "$tmp_log" "$log_file"
                quiet_echo "  Log: $log_file"
            fi
            rm -f "$tmp_log"
            return 0
        else
            local rc=$?
            cat "$tmp_log"
            cat "$tmp_log" >> "$SIM_LOG"
            if [ -n "${LOG_DIR:-}" ]; then
                log_file="${LOG_DIR}/${sanitized}_${timestamp}.log"
                cp "$tmp_log" "$log_file"
                quiet_echo "  Log: $log_file"
            fi
            rm -f "$tmp_log"
        fi
    else
        if timeout "$TIMEOUT" "$bin" "$@"; then
            return 0
        else
            local rc=$?
        fi
    fi
    if [ -z "${rc:-}" ]; then return 1; fi
    if [ $rc -eq 124 ]; then
        echo ""
        echo "  TIMEOUT: exceeded ${TIMEOUT}s" >&2
    else
        echo ""
        echo "  FAILED (exit $rc)" >&2
    fi
    return $rc
}

PASS_COUNT=0
FAIL_COUNT=0
FAILED_TESTS=()

run_test() {
    local bin="$1"
    local label="$2"
    shift 2
    if run_binary "$bin" "$label" "$@"; then
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_TESTS+=("$label")
    fi
}

# Map script test name + THREAD_MODE to actual binary.
# --orch selects the orch-only variant; --sched selects the sched-profiling-only variant;
# default (concurrent) runs both threads.
get_binary_path() {
    local name="$1"
    local idx="$2"
    case "$name" in
        test_batch_paged_attention)
            case "$THREAD_MODE" in
                orch)  echo "${BIN_DIR}/test_orch_only_${idx}" ;;
                sched) echo "${BIN_DIR}/test_sched_prof_only_${idx}" ;;
                *)     echo "${BIN_DIR}/test_concurrent_${idx}" ;;
            esac ;;
        test_alt)
            case "$THREAD_MODE" in
                orch)  echo "${BIN_DIR}/test_alt_orch_only_${idx}" ;;
                sched) echo "${BIN_DIR}/test_alt_sched_prof_only_${idx}" ;;
                *)     echo "${BIN_DIR}/test_alt_concurrent_${idx}" ;;
            esac ;;
        test_bgemm)
            case "$THREAD_MODE" in
                orch)  echo "${BIN_DIR}/test_bgemm_orch_only_${idx}" ;;
                sched) echo "${BIN_DIR}/test_bgemm_sched_prof_only_${idx}" ;;
                *)     echo "${BIN_DIR}/test_bgemm_concurrent_${idx}" ;;
            esac ;;
        test_pau)
            case "$THREAD_MODE" in
                orch)  echo "${BIN_DIR}/test_pau_orch_only_${idx}" ;;
                sched) echo "${BIN_DIR}/test_pau_sched_prof_only_${idx}" ;;
                *)     echo "${BIN_DIR}/test_pau_concurrent_${idx}" ;;
            esac ;;
        test_throughput)
            case "$THREAD_MODE" in
                orch)  echo "${BIN_DIR}/test_throughput_orch_only_${idx}" ;;
                sched) echo "${BIN_DIR}/test_throughput_sched_prof_only_${idx}" ;;
                *)     echo "${BIN_DIR}/test_throughput_concurrent_${idx}" ;;
            esac ;;
        test_latency)
            case "$THREAD_MODE" in
                orch)  echo "${BIN_DIR}/test_latency_orch_only_${idx}" ;;
                sched) echo "${BIN_DIR}/test_latency_sched_prof_only_${idx}" ;;
                *)     echo "${BIN_DIR}/test_latency_concurrent_${idx}" ;;
            esac ;;
        test_paged_attention) echo "${BIN_DIR}/test_pa_concurrent_${idx}" ;;
        *) echo "${BIN_DIR}/${name}_${idx}" ;;
    esac
}

# Collect CMake target names that need to be built for the current run.
# Prints one target per line.
# When FILTER_TEST is set, only the targets for that specific test are printed.
# When no filter, prints nothing (caller should build all).
collect_build_targets() {
    local name idx
    if [ -n "$FILTER_TEST" ]; then
        name="$FILTER_TEST"
        local indices="${TEST_INDICES[$name]}"
        if [ "$indices" = "-" ]; then
            echo "$name"
        elif [ -n "$FILTER_IDX" ]; then
            basename "$(get_binary_path "$name" "$FILTER_IDX")"
        else
            for idx in $indices; do
                basename "$(get_binary_path "$name" "$idx")"
            done
        fi
    fi
}

# Run all indices (or a specific one) for a registered test.
# After each individual binary, call sched_overhead_analysis.py with the current
# AICPU_UT_PHASE_LOG so each idx gets its own breakdown, then clear the log for
# the next run (prevents later runs from overwriting earlier data in the parser).
run_idx_analysis() {
    [ -z "${AICPU_UT_QUIET:-}" ] || return 0
    [ -n "${AICPU_UT_PHASE_LOG:-}" ] || return 0
    [ -s "$AICPU_UT_PHASE_LOG" ] || return 0
    local sched_script="${PROJECT_ROOT}/tools/sched_overhead_analysis.py"
    [ -f "$sched_script" ] || return 0
    # profiling 1: timing lines were suppressed by format_profiling_output.py pipe; re-emit them here.
    # profiling 2: dev_log_always already outputs to stdout directly; no need to re-emit.
    if [ "$PROFILING_MODE" = "1" ]; then
        grep -E "aicpu_orchestration_entry returned|Scheduler summary:" "$AICPU_UT_PHASE_LOG" 2>/dev/null || true
    fi
    (cd "$PROJECT_ROOT" && python3 tools/sched_overhead_analysis.py --sim-log "$AICPU_UT_PHASE_LOG" --no-sources 2>/dev/null) | awk 'NF{while(n--)print ""; n=0; print; next} {n++}' || true
    # (phase log kept per-test, not cleared)
    SKIP_FINAL_ANALYSIS=true
}

run_test_entry() {
    local name="$1"
    local idx="$2"   # empty = run all indices
    local type="${TEST_TYPE[$name]}"
    local indices="${TEST_INDICES[$name]}"
    local mode_suffix=""
    [ "$THREAD_MODE" = "orch" ]  && mode_suffix=" [orch]"
    [ "$THREAD_MODE" = "sched" ] && mode_suffix=" [sched]"

    if [ "$indices" = "-" ]; then
        run_test "${BIN_DIR}/${name}" "${name} (${type})"
    elif [ -n "$idx" ]; then
        local bin
        bin=$(get_binary_path "$name" "$idx")
        run_test "$bin" "${name}[${idx}]${mode_suffix} (${type})"
        run_idx_analysis
    else
        for i in $indices; do
            bin=$(get_binary_path "$name" "$i")
            run_test "$bin" "${name}[${i}]${mode_suffix} (${type})"
            run_idx_analysis
        done
    fi
}

# ─── Step 1: CMake configure ──────────────────────────────────────────────────
quiet_echo ""
quiet_echo "============================================================"
quiet_echo "  Configuring with CMake"
quiet_echo "============================================================"
quiet_echo "  Build dir : $BUILD_DIR"
quiet_echo "  Source dir: $SCRIPT_DIR"
quiet_echo "  Opt level : -O${OPT_LEVEL}"
quiet_echo "  Callstack : AICPU_UT_PROFILE_CALLSTACK=$AICPU_UT_PROFILE_CALLSTACK"
quiet_echo "  Profiling : mode=$PROFILING_MODE (0=off 1=base 2=full) PTO2_PROFILING=$PTO2_PROFILING"

mkdir -p "$BUILD_DIR"

cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" \
    -DORCH_CPU="$ORCH_CPU" \
    -DSCHED_CPU0="$SCHED_CPU0" \
    -DSCHED_CPU1="$SCHED_CPU1" \
    -DSCHED_CPU2="$SCHED_CPU2" \
    -DSCHED_CPU3="$SCHED_CPU3" \
    -DSCHED_CPU4="$SCHED_CPU4" \
    -DSCHED_CPU5="$SCHED_CPU5" \
    -DSCHED_CPU6="$SCHED_CPU6" \
    -DSCHED_CPU7="$SCHED_CPU7" \
    -DPLATFORM_MAX_BLOCKDIM="$PLATFORM_MAX_BLOCKDIM" \
    -DPLATFORM_AIC_CORES_PER_BLOCKDIM="$PLATFORM_AIC_CORES_PER_BLOCKDIM" \
    -DPLATFORM_AIV_CORES_PER_BLOCKDIM="$PLATFORM_AIV_CORES_PER_BLOCKDIM" \
    -DPLATFORM_MAX_AICPU_THREADS="$PLATFORM_MAX_AICPU_THREADS" \
    -DOPT_LEVEL="$OPT_LEVEL" \
    -DAICPU_UT_PROFILE_CALLSTACK="$AICPU_UT_PROFILE_CALLSTACK" \
    -DPTO2_PROFILING="$PTO2_PROFILING" \
    -DPTO2_SCHED_PROFILING="$PTO2_SCHED_PROFILING" \
    -DPTO2_ORCH_PROFILING="$PTO2_ORCH_PROFILING" \
    -DPTO2_TENSORMAP_PROFILING="$PTO2_TENSORMAP_PROFILING" \
    -DAICPU_UT_LATENCY_NUM_CHAINS="${AICPU_UT_LATENCY_NUM_CHAINS:-}" \
    -DAICPU_UT_LATENCY_CHAIN_LENGTH="${AICPU_UT_LATENCY_CHAIN_LENGTH:-}" \
    $( [ -n "${AICPU_UT_THROUGHPUT_LAYERS:-}" ]       && echo "-DAICPU_UT_THROUGHPUT_LAYERS=$AICPU_UT_THROUGHPUT_LAYERS" ) \
    $( [ -n "${AICPU_UT_THROUGHPUT_LAYER0_SIZE:-}" ]  && echo "-DAICPU_UT_THROUGHPUT_LAYER0_SIZE=$AICPU_UT_THROUGHPUT_LAYER0_SIZE" ) \
    $( [ -n "${AICPU_UT_THROUGHPUT_DEPS:-}" ]         && echo "-DAICPU_UT_THROUGHPUT_DEPS=$AICPU_UT_THROUGHPUT_DEPS" ) \
    $( [ -n "${AICPU_UT_THROUGHPUT_OVERLAP:-}" ]      && echo "-DAICPU_UT_THROUGHPUT_OVERLAP=$AICPU_UT_THROUGHPUT_OVERLAP" ) \
    $( [ -n "${AICPU_UT_THROUGHPUT_FIX_TAIL:-}" ]     && echo "-DAICPU_UT_THROUGHPUT_FIX_TAIL=$AICPU_UT_THROUGHPUT_FIX_TAIL" ) \
    $( [ -n "${AICPU_UT_NUM_SCHED_THREADS:-}" ]       && echo "-DAICPU_UT_NUM_SCHED_THREADS=$AICPU_UT_NUM_SCHED_THREADS" )

# ─── Step 2: Build ────────────────────────────────────────────────────────────
BIN_DIR="${BUILD_DIR}/bin"

# Batch tests that default to idx 0 only unless --all
BATCH_IDX0_ONLY_TESTS=(test_batch_paged_attention test_alt test_bgemm test_pau test_latency)

NPROC=$(nproc 2>/dev/null || echo 4)

# Collect targets; empty array means build all.
BUILD_TARGETS=()
while IFS= read -r t; do
    BUILD_TARGETS+=("$t")
done < <(collect_build_targets)

if [ ${#BUILD_TARGETS[@]} -gt 0 ]; then
    quiet_echo ""
    quiet_echo "============================================================"
    quiet_echo "  Building: ${BUILD_TARGETS[*]}"
    quiet_echo "============================================================"
    cmake --build "$BUILD_DIR" --target "${BUILD_TARGETS[@]}" --parallel "$NPROC"
else
    quiet_echo ""
    quiet_echo "============================================================"
    quiet_echo "  Building all test binaries"
    quiet_echo "============================================================"
    cmake --build "$BUILD_DIR" --parallel "$NPROC"
fi
quiet_echo ""
quiet_echo "  Built binaries:"
if [ ${#BUILD_TARGETS[@]} -gt 0 ]; then
    for t in "${BUILD_TARGETS[@]}"; do
        quiet_echo "    ${BIN_DIR}/${t}"
    done
else
    for b in "${BIN_DIR}"/test_*; do
        quiet_echo "    $b"
    done
fi

if $BUILD_ONLY; then
    quiet_echo ""
    quiet_echo "Build complete (--build-only, skipping tests)."
    exit 0
fi

# ─── Step 3: Run tests ────────────────────────────────────────────────────────
PROJECT_ROOT="${SCRIPT_DIR}/../.."
SIM_LOG=""
if [ -n "${AICPU_UT_NUM_SCHED_THREADS:-}" ]; then
    export AICPU_UT_NUM_SCHED_THREADS
fi
if [ -n "${AICPU_UT_THROUGHPUT_LAYERS:-}" ]; then
    export AICPU_UT_THROUGHPUT_LAYERS
fi
if [ -n "${AICPU_UT_THROUGHPUT_LAYER0_SIZE:-}" ]; then
    export AICPU_UT_THROUGHPUT_LAYER0_SIZE
fi
if [ -n "${AICPU_UT_THROUGHPUT_DEPS:-}" ]; then
    export AICPU_UT_THROUGHPUT_DEPS
fi
if [ -n "${AICPU_UT_THROUGHPUT_OVERLAP:-}" ]; then
    export AICPU_UT_THROUGHPUT_OVERLAP
fi
if [ -n "${AICPU_UT_THROUGHPUT_FIX_TAIL:-}" ]; then
    export AICPU_UT_THROUGHPUT_FIX_TAIL
fi
if [ -n "${AICPU_UT_LATENCY_NUM_CHAINS:-}" ]; then
    export AICPU_UT_LATENCY_NUM_CHAINS
fi
if [ -n "${AICPU_UT_LATENCY_CHAIN_LENGTH:-}" ]; then
    export AICPU_UT_LATENCY_CHAIN_LENGTH
fi
if [ -n "${AICPU_UT_NO_CHECK:-}" ]; then
    export AICPU_UT_NO_CHECK
fi
if $RUN_PERF && [ -z "${AICPU_UT_QUIET:-}" ]; then
    mkdir -p "$PROJECT_ROOT/outputs"
    SIM_LOG="$PROJECT_ROOT/outputs/aicpu_ut_sim_run.log"
    : > "$SIM_LOG"
    # Per-case log dir: each sample output to one file (sample_name + params + timestamp)
    LOG_DIR="${SCRIPT_DIR}/outputs/log"
    mkdir -p "$LOG_DIR"
    # DEV_ALWAYS output (Thread release, PTO2 progress, Scheduler Phase Breakdown) is also written here
    export AICPU_UT_PHASE_LOG="$PROJECT_ROOT/outputs/aicpu_ut_phase_breakdown.log"
    : > "$AICPU_UT_PHASE_LOG"
    export AICPU_UT_SWIMLANE_DIR="$PROJECT_ROOT/outputs"
fi

quiet_echo ""
quiet_echo "============================================================"
quiet_echo "  Running tests"
quiet_echo "============================================================"

if [ -n "$FILTER_TEST" ]; then
    run_test_entry "$FILTER_TEST" "$FILTER_IDX"
else
    if $RUN_FUNC; then
        for name in "${ALL_TESTS[@]}"; do
            [ "${TEST_TYPE[$name]}" = "func" ] || continue
            run_test_entry "$name" ""
        done
    fi
    if $RUN_PERF; then
        for name in "${DEFAULT_PERF_TESTS[@]}"; do
            idx_arg=""
            if ! $RUN_ALL_INDICES; then
                for b in "${BATCH_IDX0_ONLY_TESTS[@]}"; do
                    if [ "$name" = "$b" ]; then idx_arg="0"; break; fi
                done
            fi
            run_test_entry "$name" "$idx_arg"
        done
    fi
fi

# ─── Scheduler overhead analysis ─────────────────────────────────────────────
# 若传入 --sim-log，Part 2 使用本次 aicpu_ut(sim) 的终端输出，可得到各 Phase 数据。
# 否则用 device log（需之前设备跑数）；无 perf_swimlane 时仅输出 Part 2（sim log）。
# 不向控制台打印 Phase breakdown log 路径及 Scheduler overhead analysis 标题。
# 若 run_idx_analysis 已逐 idx 运行分析，跳过此处汇总（避免多次 run 数据被覆盖）。
if $RUN_PERF && [ -z "${AICPU_UT_QUIET:-}" ] && ! $SKIP_FINAL_ANALYSIS; then
    SCHED_SCRIPT="${PROJECT_ROOT}/tools/sched_overhead_analysis.py"
    if [ -f "$SCHED_SCRIPT" ]; then
        # Use phase breakdown log for Part 2 when DEV_ALWAYS was redirected there (no Phase Breakdown in SIM_LOG)
        SIM_LOG_ARGS=()
        if [ -n "${AICPU_UT_PHASE_LOG:-}" ] && [ -s "$AICPU_UT_PHASE_LOG" ]; then
            SIM_LOG_ARGS=("--sim-log" "$AICPU_UT_PHASE_LOG" "--no-sources")
            if [ "$PROFILING_MODE" = "1" ]; then
                grep -E "aicpu_orchestration_entry returned|Scheduler summary:" "$AICPU_UT_PHASE_LOG" 2>/dev/null || true
            fi
        elif [ -n "${SIM_LOG:-}" ] && [ -s "$SIM_LOG" ]; then
            SIM_LOG_ARGS=("--sim-log" "$SIM_LOG" "--no-sources")
            if [ "$PROFILING_MODE" = "1" ]; then
                grep -E "aicpu_orchestration_entry returned|Scheduler summary:" "$SIM_LOG" 2>/dev/null || true
            fi
        fi
        if (cd "$PROJECT_ROOT" && python3 tools/sched_overhead_analysis.py "${SIM_LOG_ARGS[@]}" ${AICPU_UT_DEVICE_ID:+-d "$AICPU_UT_DEVICE_ID"} 2>/dev/null); then
            :
        else
            quiet_echo "  (Skip: no perf_swimlane_*.json in outputs/ and no --sim-log; or device log resolve failed.)"
        fi
    fi
fi

# ─── Swimlane diagram generation ─────────────────────────────────────────────
# Convert perf_swimlane_*.json written by export_sim_swimlane() to Perfetto
# Chrome Trace JSON (swimlane_converter.py) and Mermaid flowchart (perf_to_mermaid.py).
# Only runs when profiling is enabled (AICPU_UT_SWIMLANE_DIR is set).
if $RUN_PERF && [ -z "${AICPU_UT_QUIET:-}" ] && $GEN_SWIMLANE && [ -n "${AICPU_UT_SWIMLANE_DIR:-}" ]; then
    SWIMLANE_SCRIPT="${PROJECT_ROOT}/tools/swimlane_converter.py"
    MERMAID_SCRIPT="${PROJECT_ROOT}/tools/perf_to_mermaid.py"
    # Find the latest swimlane JSON produced by this run.
    LATEST_SWIMLANE=$(ls -t "${AICPU_UT_SWIMLANE_DIR}"/perf_swimlane_*.json 2>/dev/null | head -1)
    if [ -n "$LATEST_SWIMLANE" ]; then
        quiet_echo ""
        quiet_echo "============================================================"
        quiet_echo "  Generating swimlane diagrams"
        quiet_echo "============================================================"
        quiet_echo "  Source: $LATEST_SWIMLANE"
        if [ -f "$SWIMLANE_SCRIPT" ]; then
            (cd "$PROJECT_ROOT" && python3 tools/swimlane_converter.py "$LATEST_SWIMLANE" 2>/dev/null) || \
                quiet_echo "  (swimlane_converter.py failed or no output)"
        fi
        if [ -f "$MERMAID_SCRIPT" ]; then
            (cd "$PROJECT_ROOT" && python3 tools/perf_to_mermaid.py "$LATEST_SWIMLANE" 2>/dev/null) || \
                quiet_echo "  (perf_to_mermaid.py failed or no output)"
        fi
    fi
fi

# ─── Summary ──────────────────────────────────────────────────────────────────
quiet_echo "============================================================"
quiet_echo "  Test Run Summary"
quiet_echo "============================================================"
quiet_echo "  Passed: $PASS_COUNT"
quiet_echo "  Failed: $FAIL_COUNT"

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    quiet_echo ""
    quiet_echo "  Failed tests:"
    for t in "${FAILED_TESTS[@]}"; do
        quiet_echo "    - $t"
    done
    quiet_echo ""
    quiet_echo "  OVERALL: FAILED"
    exit 1
else
    quiet_echo ""
    quiet_echo "  OVERALL: PASSED"
    exit 0
fi

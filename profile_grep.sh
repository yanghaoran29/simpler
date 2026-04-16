#!/usr/bin/env bash
# 运行 tensormap_and_ringbuffer 单测并过滤 profiling / CSV / 真机诊断相关日志。
# 用法: ./profile_grep.sh <py文件名> -p <platform> [-d <卡号>] [--case <case名>]
# 示例: ./profile_grep.sh test_l3_dependency.py -p a2a3sim
#       ./profile_grep.sh test_l3_dependency.py -p a2a3
#       ./profile_grep.sh test_paged_attention_unroll.py -p a2a3 --case Case2
#       ./profile_grep.sh test_paged_attention_unroll.py -p a2a3 -d 1 --case Case1
# 本脚本位于 simpler/ 根目录；可从任意目录调用（脚本会 cd 到 simpler 根）。
# 真机 a2a3：DEV_ALWAYS 由 onboard kernel 在 enable_profiling 时镜像到 stderr（设备无主机 getenv）；
# 需重新编译部署 aicpu_kernel 后本脚本 grep 的 [ALWAYS]/CSV 行才会出现。
# 脚本另匹配 launch/寄存器等 host 行；无 grep 命中时回退末段输出（PROFILE_GREP_FALLBACK_LINES）。

usage() {
    echo "用法: $0 <测试py文件名> -p <platform> [-d <卡号>] [--case <case名>]" >&2
    echo "示例: $0 test_l3_dependency.py -p a2a3sim" >&2
    echo "      $0 test_l3_dependency.py -p a2a3" >&2
    echo "      $0 test_paged_attention_unroll.py -p a2a3 --case Case2" >&2
    echo "      $0 test_paged_attention_unroll.py -p a2a3 -d 1 --case Case1" >&2
    echo "  <测试py文件名>  仅写文件名，固定目录 tests/st/a2a3/tensormap_and_ringbuffer/" >&2
    echo "  -d <卡号>       设备卡号（默认 0）" >&2
    echo "  --case <name>   指定用例名；对含 manual:True 用例默认为 Case1" >&2
    echo "  环境变量 PROFILE_GREP_FALLBACK_LINES=40  无 grep 命中时打印末尾行数（默认 24）" >&2
    exit 1
}

if [[ $# -lt 3 ]]; then
    usage
fi

TEST_BASENAME="$1"
if [[ "$2" != "-p" ]]; then
    echo "错误: 第二个参数必须是 -p" >&2
    usage
fi
PLATFORM="$3"
shift 3

CASE_NAME=""
DEVICE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --case)
            CASE_NAME="$2"
            shift 2
            ;;
        -d)
            DEVICE="$2"
            shift 2
            ;;
        *)
            echo "错误: 未知参数 $1" >&2
            usage
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIMPLER_ROOT="${SCRIPT_DIR}"
TEST_BASE_DIR="tests/st/a2a3/tensormap_and_ringbuffer"

# Try flat path first, then search recursively in subdirectories
TEST_REL="${TEST_BASE_DIR}/${TEST_BASENAME}"
if [[ ! -f "${SIMPLER_ROOT}/${TEST_REL}" ]]; then
    FOUND="$(find "${SIMPLER_ROOT}/${TEST_BASE_DIR}" -name "${TEST_BASENAME}" -type f 2>/dev/null | head -1)"
    if [[ -z "${FOUND}" ]]; then
        echo "错误: 找不到 ${TEST_BASENAME} (在 ${SIMPLER_ROOT}/${TEST_BASE_DIR} 及其子目录下)" >&2
        exit 1
    fi
    TEST_REL="${FOUND#${SIMPLER_ROOT}/}"
fi

cd "${SIMPLER_ROOT}" || exit 1

# 若测试文件含 manual:True 用例且未指定 --case，默认跑 Case1
if [[ -z "${CASE_NAME}" ]] && grep -q '"manual".*True\|manual.*=.*True' "${TEST_REL}" 2>/dev/null; then
    CASE_NAME="Case1"
    echo "--- 检测到 manual 用例，默认使用 --case Case1（可用 --case <name> 覆盖）---" >&2
fi

CASE_ARGS=()
if [[ -n "${CASE_NAME}" ]]; then
    CASE_ARGS=(--case "${CASE_NAME}")
fi

DEVICE_ARGS=()
if [[ -n "${DEVICE}" ]]; then
    DEVICE_ARGS=(-d "${DEVICE}")
fi

# sim：与 test_l3_dependency.py 注释中 profiling grep 一致。
# 真机 a2a3：补充 host/启动侧常见行（profiling 若未回传 stdout 仍有可核对信息）。
GREP_PATTERN='CSV注释变量|CSV变量说明|CSV按task种类|CSVglossary|-> P\(|编译常量|Orchestrator CSV|Scheduler CSV|--- |r=|PASSED|FAILED|无bucket'
GREP_PATTERN+='|launch_aicpu_kernel|DynTileFwk|get_aicore_reg_info|handshake_all_cores|Register base:|physical_id='
GREP_PATTERN+='|sm_ptr=|arg_count=|Orchestrator|Profiling|resolve_and_dispatch_pto2|[[]ALWAYS[]]'

FALLBACK_LINES="${PROFILE_GREP_FALLBACK_LINES:-24}"
TMP_OUT="$(mktemp "${TMPDIR:-/tmp}/profile_grep.XXXXXX")"
cleanup() { rm -f "${TMP_OUT}"; }
trap cleanup EXIT

python "${TEST_REL}" -p "${PLATFORM}" --enable-profiling "${CASE_ARGS[@]}" "${DEVICE_ARGS[@]}" 2>&1 | tee "${TMP_OUT}" | grep -E "${GREP_PATTERN}"
grep_rc="${PIPESTATUS[2]}"

if [[ "${grep_rc}" -ne 0 ]]; then
    echo "--- profile_grep: 无行匹配 GREP_PATTERN（真机 a2a3 上 profiling 常不在本机 stdout）；回退末 ${FALLBACK_LINES} 行 ---" >&2
    tail -n "${FALLBACK_LINES}" "${TMP_OUT}"
fi

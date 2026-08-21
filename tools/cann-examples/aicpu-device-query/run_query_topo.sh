#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

if [[ -d /usr/local/Ascend/ascend-toolkit/latest ]]; then
    export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
elif [[ -d /usr/local/Ascend/cann-9.2.0 ]]; then
    export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.0
else
    echo "ASCEND_HOME_PATH not found" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "${ASCEND_HOME_PATH}/set_env.sh"
echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"

export SIMPLER_DISPATCHER_SO="${SIMPLER_DISPATCHER_SO:-/home/pyptouser/yanghaoran/.local/lib/python3.11/site-packages/simpler_setup/_assets/build/lib/a5/dispatcher/libsimpler_aicpu_dispatcher.so}"

DISPATCHER="${SIMPLER_DISPATCHER_SO}"
if [[ ! -f "${DISPATCHER}" ]]; then
    echo "Building a5 runtime artifacts (includes dispatcher)..."
    PYTHONPATH="${REPO}:${PYTHONPATH:-}" python3 "${REPO}/simpler_setup/build_runtimes.py" \
        --lib-dir "${REPO}/build/lib" \
        --cache-dir "${REPO}/build/cache" \
        --platforms a5
fi

QUERY_SO="${REPO}/tools/cann-examples/aicpu-device-query/device/build/libaicpu_query.so"
if [[ ! -f "${QUERY_SO}" ]]; then
    QUERY_SO="/home/pyptouser/yanghaoran/Desktop/simpler-08051/simpler/tools/cann-examples/aicpu-device-query/device/build/libaicpu_query.so"
fi
export SIMPLER_AICPU_QUERY_SO="${QUERY_SO}"

HOST_BIN="${REPO}/tools/cann-examples/aicpu-device-query/host/build/query_device_hal"
if [[ ! -f "${HOST_BIN}" ]]; then
    cmake -S "${REPO}/tools/cann-examples/aicpu-device-query/host" \
        -B "${REPO}/tools/cann-examples/aicpu-device-query/host/build"
    cmake --build "${REPO}/tools/cann-examples/aicpu-device-query/host/build" -j"$(nproc)"
fi

export SIMPLER_DISPATCHER_SO="${DISPATCHER}"
export SIMPLER_AICPU_QUERY_SO="${QUERY_SO}"

DEVICE_ID="${TASK_DEVICE:-0}"
if [[ "${1:-}" == "--device" && -n "${2:-}" ]]; then
    DEVICE_ID="$2"
elif [[ -n "${1:-}" && "${1}" != "--device" ]]; then
    DEVICE_ID="$1"
fi
echo "=== query_device_hal --json on device ${DEVICE_ID} ===" >&2
exec "${HOST_BIN}" "${DEVICE_ID}" --json

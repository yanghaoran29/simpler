#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.0
# shellcheck disable=SC1091
source "${ASCEND_HOME_PATH}/set_env.sh"

export SIMPLER_DISPATCHER_SO="${SIMPLER_DISPATCHER_SO:-/home/pyptouser/yanghaoran/.local/lib/python3.11/site-packages/simpler_setup/_assets/build/lib/a5/dispatcher/libsimpler_aicpu_dispatcher.so}"

QUERY_SO="${REPO}/tools/cann-examples/aicpu-device-query/device/build/libaicpu_query.so"
if [[ ! -f "${QUERY_SO}" ]]; then
    QUERY_SO="/home/pyptouser/yanghaoran/Desktop/simpler-08051/simpler/tools/cann-examples/aicpu-device-query/device/build/libaicpu_query.so"
fi
export SIMPLER_AICPU_QUERY_SO="${QUERY_SO}"

HOST_BIN="${REPO}/tools/cann-examples/aicpu-device-query/host/build/query_device_hal"
if [[ ! -x "${HOST_BIN}" ]]; then
    cmake -S "${REPO}/tools/cann-examples/aicpu-device-query/host" \
        -B "${REPO}/tools/cann-examples/aicpu-device-query/host/build"
    cmake --build "${REPO}/tools/cann-examples/aicpu-device-query/host/build" -j"$(nproc)"
fi

DEVICE_ID="${TASK_DEVICE:-0}"
if [[ "${1:-}" == "--device" && -n "${2:-}" ]]; then
    DEVICE_ID="$2"
elif [[ -n "${1:-}" && "${1}" != "--device" ]]; then
    DEVICE_ID="$1"
fi
echo "dispatcher=${SIMPLER_DISPATCHER_SO}" >&2
echo "query_so=${SIMPLER_AICPU_QUERY_SO}" >&2
echo "host=${HOST_BIN}" >&2
echo "device=${DEVICE_ID}" >&2
exec "${HOST_BIN}" "${DEVICE_ID}" --json

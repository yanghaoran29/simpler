#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set -euo pipefail

# Run the topology query and the expensive RTT calibration as an explicit
# maintenance workflow. The production runtime never invokes this script.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TOOL_ROOT="${ROOT}/tools/cann-examples/a5-aicore-rtt"
QUERY_BIN="${ROOT}/tools/cann-examples/aicpu-device-query/host/build/query_device_hal"
RTT_BIN="${TOOL_ROOT}/host/build/launch_a5_aicore_rtt"
GENERATOR="${TOOL_ROOT}/host/generate_runtime_assignment.py"

DEVICE_ID="${TASK_DEVICE:-${1:-}}"
OUTPUT_DIR="${2:-${PWD}/a5-rtt-calibration}"
if [[ -z "${DEVICE_ID}" ]]; then
    echo "usage: $0 <device_id> [output_dir]" >&2
    exit 2
fi
if [[ ! -x "${QUERY_BIN}" || ! -x "${RTT_BIN}" ]]; then
    echo "error: build aicpu-device-query and a5-aicore-rtt host binaries first" >&2
    exit 2
fi

mkdir -p "${OUTPUT_DIR}"
TOPOLOGY_JSON="$(mktemp "${OUTPUT_DIR}/topology.XXXXXX.json")"
trap 'rm -f "${TOPOLOGY_JSON}"' EXIT

# Reuse the runtime's normal topology selection to obtain [S0,S1,S2,S3,O].
"${QUERY_BIN}" "${DEVICE_ID}" --json >"${TOPOLOGY_JSON}"
ALLOWED_CPUS="$(python3 - "${TOPOLOGY_JSON}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    topology = json.load(stream)
allowed = topology.get("launch_plan", {}).get("allowed_cpus", [])
if len(allowed) != 5 or len(set(allowed)) != 5:
    raise SystemExit("topology JSON does not contain five unique allowed_cpus")
print(",".join(str(cpu) for cpu in allowed))
PY
)"

RAW_JSON="${OUTPUT_DIR}/a5-aicore-rtt-device${DEVICE_ID}.json"
FRAGMENT_JSON="${OUTPUT_DIR}/a5-runtime-assignment-device${DEVICE_ID}.json"
"${RTT_BIN}" "${DEVICE_ID}" --samples 50 --warmup 10 --allowed-cpus "${ALLOWED_CPUS}" --json "${RAW_JSON}"
python3 "${GENERATOR}" "${RAW_JSON}" --topology "${TOPOLOGY_JSON}" --output "${FRAGMENT_JSON}"

echo "raw RTT JSON: ${RAW_JSON}"
echo "packaged-config fragment: ${FRAGMENT_JSON}"

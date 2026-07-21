#!/bin/bash
set -euo pipefail
cd /home/pyptouser/yanghaoran/Desktop/simpler2
source .venv/bin/activate
DEV="${ASCEND_RT_VISIBLE_DEVICES:-${DEVICE:-}}"
# Prefer lock-assigned device: task-submit often sets ASCEND_RT_VISIBLE_DEVICES
if [[ -z "${DEV}" ]]; then
  echo "No device env; failing"
  env | sort | head -40
  exit 2
fi
# If ASCEND is set to a single id by task-submit, use it
echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-unset} DEVICE=${DEVICE:-unset}"
# Use first id for --device
DID="${ASCEND_RT_VISIBLE_DEVICES%%,*}"
DID="${DID:-$DEV}"
echo "Using --device $DID"
TEST="${1:?test path}"
NAME="${2:?label}"
python -m pytest "$TEST" --platform a5 --device "$DID" -v --tb=short \
  --enable-l2-swimlane 4 --enable-dep-gen \
  2>&1 | tee "_early_st_out/${NAME}_board.log"

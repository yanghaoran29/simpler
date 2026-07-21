#!/bin/bash
set -euo pipefail
cd /home/pyptouser/yanghaoran/Desktop/simpler2
source .venv/bin/activate
# task-submit --device N may not export ASCEND; hardcode from argv
DEV="${1:?device}"
SHIFT=1
shift
python -m pytest "$@" --platform a5 --device "$DEV" -v --tb=line

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Cross-platform l2_swimlane validation helpers shared by a2a3 and a5.

The per-platform ``_swimlane_validate.py`` files live in sibling leaf
packages (``.../a2a3/.../l2_swimlane`` and ``.../a5/.../l2_swimlane``) that
have no shared package parent under ``--import-mode=importlib``. Helpers that
encode a contract both platforms must keep byte-identical live here and are
loaded by each validator via a ``__file__``-relative module load, so the
contract has a single source of truth instead of being hand-synced across
two copies.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def verify_dependency_view_parity(perf: dict, deps_json: Path, out_dir: Path) -> None:
    """Assert the Scheduler View mirrors every Worker View dependency arrow.

    Runs the converter with ``--deps-json`` (so flow events are emitted) and
    counts ``dependency`` / ``hb_violation`` flow starts on each view: pid=4
    Worker View (AICore timestamps) and pid=3 Scheduler View (AICPU). With
    AICPU timing present (level>=2) the two counts must match — the Scheduler
    mirror is required to anchor on the first subtask it can bind to (SPMD
    sibling fallback) and to land an inbound arrow on a successor's dispatch
    even when that successor's own completion wasn't captured, so no arrow the
    Worker View draws may go missing. Skipped at level 1 (no Scheduler View).

    Args:
        perf: parsed ``l2_swimlane_records.json`` dict.
        deps_json: sibling ``deps.json`` co-captured with the perf artifact.
        out_dir: per-case output directory for the converter's JSON.
    """
    tasks = perf.get("tasks", [])
    has_aicpu = any(t.get("dispatch_time_us", 0) >= 0 and t.get("finish_time_us", 0) > 0 for t in tasks)
    if not has_aicpu:
        return  # level 1: AICore-only, no Scheduler View to mirror.

    trace_path = out_dir / "_smoke_swimlane_deps.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "simpler_setup.tools.swimlane_converter",
            str(deps_json.parent / "l2_swimlane_records.json"),
            "-o",
            str(trace_path),
            "--deps-json",
            str(deps_json),
        ],
        check=True,
        timeout=60,
    )

    with trace_path.open() as f:
        events = json.load(f)["traceEvents"]

    def _dep_flow_starts(pid: int) -> int:
        return sum(
            1
            for e in events
            if e.get("cat") == "flow"
            and e.get("ph") == "s"
            and e.get("pid") == pid
            and e.get("name") in ("dependency", "hb_violation")
        )

    worker = _dep_flow_starts(4)
    scheduler = _dep_flow_starts(3)
    if worker == 0:
        return  # deps.json resolved no edges onto captured tasks — nothing to mirror.
    assert scheduler == worker, (
        f"Scheduler View dropped dependency arrows the Worker View drew "
        f"(Worker={worker}, Scheduler={scheduler}). The Scheduler mirror must "
        f"anchor on the first bindable subtask and not require a captured "
        f"finish on the successor — see swimlane_converter._scheduler_anchor_row."
    )

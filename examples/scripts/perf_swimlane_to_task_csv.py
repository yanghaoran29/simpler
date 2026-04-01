#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
"""从 Host 导出的 perf_swimlane_*.json 的 tasks[] 提取 task 依赖信息为 CSV。

列：task_id, ring_id, func_id, core_id, core_type, fanout_task_ids, fanout_count,
    fanin_refcount

fanout_task_ids 为分号分隔的十进制 task_id（与 JSON 中 fanout 数组一致）。

旧版 JSON（无 fanin_* 字段）对应列留空或 0。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def _project_root() -> Path:
    # examples/scripts/<this>.py -> simpler/
    return Path(__file__).resolve().parent.parent.parent


def _latest_swimlane_json(outputs_dir: Path) -> Path | None:
    files = sorted(outputs_dir.glob("perf_swimlane_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def main() -> int:
    ap = argparse.ArgumentParser(description="perf_swimlane JSON → task dependency CSV")
    ap.add_argument(
        "json_path",
        nargs="?",
        help="perf_swimlane_*.json；省略则在 outputs/ 下取最新",
    )
    ap.add_argument("-o", "--output", help="输出 CSV 路径（默认与 JSON 同目录，后缀 _tasks.csv）")
    ap.add_argument(
        "--outputs-dir",
        default=None,
        help="搜索最新 perf_swimlane 的目录（默认 <simpler>/outputs）",
    )
    ap.add_argument(
        "--min-tasks",
        type=int,
        default=0,
        metavar="N",
        help="若 tasks 条数 < N 则失败退出（用于发现 perf 收集不完整）",
    )
    args = ap.parse_args()

    root = _project_root()
    out_dir = Path(args.outputs_dir) if args.outputs_dir else (root / "outputs")
    if args.json_path:
        jp = Path(args.json_path)
        if not jp.is_file():
            print(f"Error: file not found: {jp}", file=sys.stderr)
            return 1
    else:
        jp = _latest_swimlane_json(out_dir)
        if jp is None:
            print(f"Error: no perf_swimlane_*.json under {out_dir}", file=sys.stderr)
            return 1
        print(f"[perf_swimlane_to_task_csv] using {jp}", file=sys.stderr)

    data = json.loads(jp.read_text(encoding="utf-8"))
    tasks = data.get("tasks")
    if not isinstance(tasks, list):
        print("Error: JSON missing 'tasks' array", file=sys.stderr)
        return 1

    ntasks = len(tasks)
    if args.min_tasks > 0 and ntasks < args.min_tasks:
        print(
            f"Error: JSON has only {ntasks} tasks (require --min-tasks {args.min_tasks}). "
            f"File: {jp}\n"
            "常见原因：Host 侧 perf 在 execution_complete 时尚未收齐缓冲（见日志 "
            '"Execution complete signal received" / "Total records collected" / '
            '"export_swimlane_json: Records:"）；或本次运行并非完整 Case1 大图。\n'
            "可改用明确的一次成功导出的 JSON：\n"
            f"  python3 {Path(__file__).name} path/to/perf_swimlane_*.json",
            file=sys.stderr,
        )
        return 1

    if args.output:
        outp = Path(args.output)
    else:
        outp = jp.with_name(jp.stem + "_tasks.csv")

    fieldnames = [
        "task_id",
        "ring_id",
        "func_id",
        "core_id",
        "core_type",
        "fanout_task_ids",
        "fanout_count",
        "fanin_refcount",
    ]

    with outp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for t in tasks:
            if not isinstance(t, dict):
                continue
            fanout = t.get("fanout")
            if isinstance(fanout, list):
                fo_str = ";".join(str(int(x)) for x in fanout)
            else:
                fo_str = ""
            tid = t.get("task_id", 0)
            try:
                tid_i = int(tid)
            except (TypeError, ValueError):
                tid_i = 0
            row = {
                "task_id": tid_i,
                "ring_id": t.get("ring_id", (tid_i >> 32) if tid_i else ""),
                "func_id": t.get("func_id", ""),
                "core_id": t.get("core_id", ""),
                "core_type": t.get("core_type", ""),
                "fanout_task_ids": fo_str,
                "fanout_count": t.get("fanout_count", ""),
                "fanin_refcount": t.get("fanin_refcount", ""),
            }
            w.writerow(row)

    print(f"[perf_swimlane_to_task_csv] wrote {outp}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

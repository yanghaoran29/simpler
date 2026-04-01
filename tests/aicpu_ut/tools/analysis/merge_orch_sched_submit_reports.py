#!/usr/bin/env python3
"""Merge Scheduler submit report + Orchestrator submit report into one markdown document."""
from __future__ import annotations

import argparse
from pathlib import Path


def demote_headings(text: str, levels: int = 1) -> str:
    """Prefix each markdown heading line with extra '#' so it nests under a parent section."""
    if levels <= 0:
        return text
    prefix = "#" * levels
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            # Preserve original indentation before hashes
            indent = line[: len(line) - len(stripped)]
            out.append(indent + prefix + stripped)
        else:
            out.append(line)
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Merge *_scheduler_submit_report_*.md and *_submit_report_*.md into one file."
    )
    ap.add_argument("--scheduler-md", required=True, type=Path)
    ap.add_argument("--orche-md", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--test-name", default="test")
    ap.add_argument("--report-idx", default="0")
    args = ap.parse_args()

    sched = args.scheduler_md.read_text(encoding="utf-8", errors="replace")
    orch = args.orche_md.read_text(encoding="utf-8", errors="replace")

    # Nest original H1.. under ## sections (+1 level each)
    sched_body = demote_headings(sched, 1)
    orch_body = demote_headings(orch, 1)

    out = args.output
    with out.open("w", encoding="utf-8") as f:
        f.write(
            f"# {args.test_name} idx={args.report_idx} "
            "Scheduler 与 Orchestrator 指令统计（合并报告）\n\n"
        )
        f.write("> 由 `./run_tests.sh --count-orch-and-sched-submit-instructions` 一次流程生成，"
        "合并下列两份报告的全部内容：\n")
        f.write(f"> - Scheduler：`{args.scheduler_md.name}`\n")
        f.write(f"> - Orchestrator（`pto2_submit_mixed_task`）：`{args.orche_md.name}`\n\n")
        f.write("---\n\n")

        f.write("## 第一部分：Scheduler（`resolve_and_dispatch_pto2` 主循环）\n\n")
        f.write(sched_body)
        f.write("\n\n---\n\n")

        f.write("## 第二部分：Orchestrator（`pto2_submit_mixed_task` / build_graph）\n\n")
        f.write(orch_body)
        f.write("\n")

    print(str(out))


if __name__ == "__main__":
    main()

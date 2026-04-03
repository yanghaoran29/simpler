#!/usr/bin/env python3
"""Merge Scheduler submit report + Orchestrator submit report into one markdown document.

Layout (for --count-orch-and-sched-submit-instructions):
  1) Orchestrator 摘要（汇总表、阶段均值、任务类型分组、指令/访存汇总表等）
  2) Scheduler 摘要（调用次数、总汇总、per-step、不含会话级大表与原始附录）
  3) 详细数据（Orchestrator 逐任务明细 + Scheduler 从「每迭代」起的全部内容）
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def demote_headings(text: str, levels: int = 1) -> str:
    """Prefix each markdown heading line with extra '#' so it nests under a parent section."""
    if levels <= 0 or not text:
        return text
    prefix = "#" * levels
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            indent = line[: len(line) - len(stripped)]
            out.append(indent + prefix + stripped)
        else:
            out.append(line)
    return "\n".join(out)


def _split_at_heading_line(text: str, predicate) -> tuple[str, str]:
    """Split into (before_first_matching_line, from_first_matching_line_inclusive)."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if predicate(line):
            summary = "\n".join(lines[:i]).rstrip()
            detail = "\n".join(lines[i:]).rstrip()
            return summary, detail
    return text.rstrip(), ""


def split_orche_report(text: str) -> tuple[str, str]:
    """Orchestrator: per-task blocks start with '## 任务编号 '."""
    return _split_at_heading_line(text, lambda ln: ln.startswith("## 任务编号 "))


def split_scheduler_report(text: str) -> tuple[str, str]:
    """Scheduler: session-level / insn / mem / raw appendix from '## 3) 每迭代动态指令数'."""
    return _split_at_heading_line(
        text, lambda ln: ln.startswith("## 3) 每迭代动态指令数")
    )


def _first_int(text: str, pattern: str) -> int:
    m = re.search(pattern, text, re.M)
    return int(m.group(1)) if m else 0


def _best_probe(text: str, name: str):
    ms = list(
        re.finditer(
            rf"^\s*{re.escape(name)}\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\b",
            text,
            re.M,
        )
    )
    if not ms:
        return None
    # Prefer non-zero sessions when duplicated (e.g. preamble phase line + corrected table line).
    best = max(ms, key=lambda m: int(m.group(1)))
    return best


def _best_probe_table(text: str, label: str):
    ms = list(
        re.finditer(
            rf"\|\s*{re.escape(label)}\s*\|\s*`?(\d+)`?\s*\|\s*`?(\d+)`?\s*\|\s*`?(\d+)`?\s*\|\s*`?(\d+)`?\s*\|\s*`?(\d+)`?\s*\|",
            text,
            re.M,
        )
    )
    if not ms:
        return None
    return max(ms, key=lambda m: int(m.group(1)))


def _parse_sched_compact_block(sched_sum: str) -> str:
    total = _first_int(sched_sum, r"QEMU_TCG between_markers_insns:\s*(\d+)")
    cpu_rows: dict[int, tuple[int, int]] = {}
    for m in re.finditer(r"cpu_id=(\d+):\s*迭代数=(\d+)\s*合计指令=(\d+)", sched_sum):
        cid = int(m.group(1))
        iters = int(m.group(2))
        insn = int(m.group(3))
        if cid not in cpu_rows:
            cpu_rows[cid] = (iters, insn)

    phase_rows = []
    for m in re.finditer(
        r"^\s*(sched_loop|complete|dispatch|idle)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\b",
        sched_sum,
        re.M,
    ):
        phase_rows.append(
            (m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)))
        )

    td = _best_probe(sched_sum, "task_dispatch")
    tc = _best_probe(sched_sum, "task_complete")
    st = _best_probe(sched_sum, "subtask_complete")
    td_tbl = _best_probe_table(sched_sum, "task_dispatch (x21/x22)")
    st_tbl = _best_probe_table(sched_sum, "subtask_complete (x33/x34)")
    tc_tbl = _best_probe_table(sched_sum, "task_complete (x17/x18)")
    if td_tbl and (td is None or int(td_tbl.group(1)) > int(td.group(1))):
        td = td_tbl
    if st_tbl and (st is None or int(st_tbl.group(1)) > int(st.group(1))):
        st = st_tbl
    if tc_tbl and (tc is None or int(tc_tbl.group(1)) > int(tc.group(1))):
        tc = tc_tbl
    db = re.search(
        r"##\s*x31/x32.*?\n\s*sessions=(\d+)\s+total_insns=(\d+)\s+avg=(\d+)\s+max=(\d+)\s+min=(\d+)",
        sched_sum,
        re.S,
    )
    db_tbl = _best_probe_table(sched_sum, "dispatch_block (x31/x32)")
    if db_tbl and (db is None or int(db_tbl.group(1)) > int(db.group(1))):
        db = db_tbl

    lines: list[str] = []
    lines.append("### 1) Scheduler 主循环指令统计")
    lines.append("")
    if cpu_rows:
        parts = [f"cpu_id={cid}指令数{cpu_rows[cid][1]}" for cid in sorted(cpu_rows.keys())]
        lines.append(f"总指令数：{total}，其中" + "，".join(parts) + "。")
    else:
        lines.append(f"总指令数：{total}。")
    lines.append("")

    lines.append("### 2) Scheduler 主要步骤指令统计")
    lines.append("任务下发/回收打点（sessions = 触发次数 = 任务数）")
    lines.append("  打点               sessions   total_insns       avg       max       min  说明")
    # User-facing compact line uses x31/x32 (dispatch_block) as the dispatch baseline.
    td_show = db if db else td
    if td_show:
        lines.append(
            f"  task_dispatch    {int(td_show.group(1)):>8}  {int(td_show.group(2)):>12}  {int(td_show.group(3)):>8}  {int(td_show.group(4)):>8}  {int(td_show.group(5)):>8}  下发任务（x31/x32）"
        )
    if st:
        lines.append(
            f"  subtask_complete {int(st.group(1)):>8}  {int(st.group(2)):>12}  {int(st.group(3)):>8}  {int(st.group(4)):>8}  {int(st.group(5)):>8}  子任务完成（x33/x34）"
        )
    if tc:
        lines.append(
            f"  task_complete    {int(tc.group(1)):>8}  {int(tc.group(2)):>12}  {int(tc.group(3)):>8}  {int(tc.group(4)):>8}  {int(tc.group(5)):>8}  任务完成（x17/x18）"
        )
    lines.append("")

    lines.append("### 3) 各步骤指令统计")
    lines.append("总体：")
    lines.append("  步骤              calls   avg_insns   max_insns   min_insns  说明")
    desc = {
        "sched_loop": "主循环迭代（外层 x23/x24）",
        "complete": "完成检查（x26 向前匹配最近 x25）",
        "dispatch": "任务分发（x28 向前匹配最近 x25）",
        "idle": "空闲等待（x30 向前匹配最近 x25）",
    }
    for name, calls, avg, mx, mn in phase_rows:
        lines.append(f"  {name:<12} {calls:>8}  {avg:>10}  {mx:>10}  {mn:>10}  {desc.get(name, '')}")
    lines.append("")
    lines.append("各线程：")
    for cid in sorted(cpu_rows.keys()):
        iters, insn = cpu_rows[cid]
        avg = insn / iters if iters else 0.0
        lines.append(f"  cpu_id={cid}: 迭代数={iters} 合计指令={insn} avg={avg:.2f}")
    lines.append("")
    return "\n".join(lines).rstrip()


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

    orch_sum, orch_det = split_orche_report(orch)
    sched_sum, sched_det = split_scheduler_report(sched)

    orch_sum_body = demote_headings(orch_sum, 1)
    sched_sum_body = _parse_sched_compact_block(sched_sum)
    orch_det_body = demote_headings(orch_det, 2) if orch_det else ""
    sched_det_body = demote_headings(sched_det, 2) if sched_det else ""

    out = args.output
    with out.open("w", encoding="utf-8") as f:
        f.write(
            f"# {args.test_name} idx={args.report_idx} "
            "Orchestrator 与 Scheduler 指令统计（合并报告）\n\n"
        )
        f.write("---\n\n")

        f.write("## 第一部分：Orchestrator（`pto2_submit_mixed_task` / build_graph）\n\n")
        f.write(orch_sum_body)
        f.write("\n\n---\n\n")

        f.write("## 第二部分：Scheduler（`resolve_and_dispatch_pto2` 主循环）\n\n")
        f.write(sched_sum_body)
        f.write("\n\n---\n\n")

        f.write("## 第三部分：详细数据\n\n")
        f.write("以下为 Orchestrator 逐任务明细与 Scheduler 会话级/聚合明细及原始附录，便于前半部分专注阅读汇总。\n\n")

        f.write("### A) Orchestrator：任务级明细\n\n")
        if orch_det_body:
            f.write(orch_det_body)
            f.write("\n\n")
        else:
            f.write("- （无 `## 任务编号` 逐任务段：可能为任务数 0、降级报告或旧版报告格式。）\n\n")

        f.write("### B) Scheduler：每迭代统计、指令/访存明细与原始附录\n\n")
        if sched_det_body:
            f.write(sched_det_body)
            f.write("\n")
        else:
            f.write(
                "- （未识别到 `## 3) 每迭代动态指令数` 起章节：可能为旧版或裁剪过的 Scheduler 报告，"
                "请直接打开单独的 scheduler 报告文件。）\n"
            )

    print(str(out))


if __name__ == "__main__":
    main()

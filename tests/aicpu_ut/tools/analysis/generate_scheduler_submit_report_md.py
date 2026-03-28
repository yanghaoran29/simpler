#!/usr/bin/env python3
"""Markdown report for --count-scheduler-submit-task-instructions (scheduler loop insn + optional insn_types)."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Optional


def parse_summary(path: Path) -> tuple[str, str, list[tuple[int, int, int, int]]]:
    """Return (preamble before raw section), raw tail, csv rows as (sid, phase, cpu, insn)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    sep = "# --- 原始插件输出（含逐条 session） ---"
    if sep in text:
        pre, post = text.split(sep, 1)
    else:
        pre, post = text, ""

    rows: list[tuple[int, int, int, int]] = []
    in_csv = False
    for line in post.splitlines():
        s = line.strip()
        if s == "session_id,phase_id,cpu_id,insn_count":
            in_csv = True
            continue
        if in_csv:
            m = re.match(r"^(\d+),(\d+),(\d+),(\d+)$", s)
            if m:
                rows.append(tuple(int(x) for x in m.groups()))
            elif s and not s.startswith("#"):
                in_csv = False
    return pre.strip(), post.strip(), rows


def parse_insn_types(path: Path) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = re.match(r"([A-Za-z0-9._]+)\s+(\d+)$", s)
        if m:
            out.append((m.group(1), int(m.group(2))))
    return out


def parse_mem_top(path: Path, topn: int = 20):
    summary: dict = {}
    rows: list[dict] = []
    in_section = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("# memory access summary"):
            in_section = True
            continue
        if in_section and s.startswith("# total="):
            m = re.search(
                r"total=(\d+)\s+load=(\d+)\s+store=(\d+)\s+unique_addr=(\d+)", s
            )
            if m:
                summary = {
                    "total": int(m.group(1)),
                    "load": int(m.group(2)),
                    "store": int(m.group(3)),
                    "unique_addr": int(m.group(4)),
                }
            continue
        if not in_section or s.startswith("#"):
            continue
        m = re.match(r"0x([0-9a-fA-F]+)\s+(\d+)\s+(\d+)\s+(\d+)$", s)
        if m:
            rows.append(
                {
                    "vaddr": "0x" + m.group(1).lower(),
                    "total": int(m.group(2)),
                    "load": int(m.group(3)),
                    "store": int(m.group(4)),
                }
            )
    return summary, rows[:topn]


def parse_rdp2_invoke_from_preamble(pre: str) -> Optional[dict[str, Any]]:
    """Parse human lines emitted by count_scheduler_loop_markers.sh from guest [rdp2-invoke-stats]."""
    total: int | None = None
    per: dict[int, int] = {}
    for line in pre.splitlines():
        s = line.strip()
        m = re.match(r"resolve_and_dispatch_pto2 函数调用总次数:\s*(\d+)\s*$", s)
        if m:
            total = int(m.group(1))
            continue
        m = re.match(r"thread_idx=(\d+):\s*(\d+)\s*$", s)
        if m:
            per[int(m.group(1))] = int(m.group(2))
    if total is None and not per:
        return None
    if total is None:
        total = sum(per.values())
    return {"total": total, "per_thread": per}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheduler-summary", required=True, type=Path)
    ap.add_argument("--insn-types", type=Path, default=None)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    pre, raw_tail, csv_rows = parse_summary(args.scheduler_summary)
    insn_types: list[tuple[str, int]] = []
    mem_summary: dict = {}
    mem_top: list[dict] = []
    if args.insn_types and args.insn_types.is_file():
        insn_types = parse_insn_types(args.insn_types)
        mem_summary, mem_top = parse_mem_top(args.insn_types, topn=20)

    insn_total = sum(c for _, c in insn_types)
    counts = [r[3] for r in csv_rows]
    n = len(counts)
    total_insns = sum(counts) if counts else 0

    title_m = re.search(r"test=(\S+)\s+idx=(\S+)", pre)
    test_guess = title_m.group(1) if title_m else "test"
    idx_guess = title_m.group(2) if title_m else "?"

    rdp2_inv = parse_rdp2_invoke_from_preamble(pre)

    out = args.output
    with out.open("w", encoding="utf-8") as f:
        f.write(f"# {test_guess} idx={idx_guess} Scheduler 主循环指令报告\n\n")
        f.write("## 说明\n\n")
        f.write(
            "- 统计区间：`resolve_and_dispatch_pto2` 主循环每次迭代（`orr x23` → `orr x24`）。\n"
        )
        f.write(
            f"- 摘要来源：`{args.scheduler_summary.name}`；"
            "指令类型/访存来自 scheduler 标记窗口内的 QEMU 插件聚合。\n\n"
        )

        f.write("## 0) resolve_and_dispatch_pto2 调用次数\n\n")
        if rdp2_inv:
            f.write(
                "以下为函数 **入口** 计数（`thread_idx` 与 `AicpuExecutor::resolve_and_dispatch_pto2` "
                "形参一致）。**不等于** 上文中主循环迭代次数（`orr x23/x24` 会话数）；"
                "在典型 SIM 并发用例中，每个调度线程通常 **进入该函数一次**。\n\n"
            )
            f.write("| 统计项 | 值 |\n")
            f.write("| --- | ---: |\n")
            f.write(f"| 函数调用总次数 | `{rdp2_inv['total']}` |\n\n")
            if rdp2_inv["per_thread"]:
                f.write("| thread_idx（调度线程索引） | 调用次数 |\n")
                f.write("| ---: | ---: |\n")
                for tid in sorted(rdp2_inv["per_thread"]):
                    f.write(f"| {tid} | `{rdp2_inv['per_thread'][tid]}` |\n")
                f.write("\n")
        else:
            f.write(
                "- 未在摘要中解析到 `resolve_and_dispatch_pto2 函数调用总次数` / `thread_idx=` 行；"
                "请确认已用 `PTO2_INSTR_COUNT_SCHEDULER_LOOP_ENABLE` 构建，并检查同时间戳的 "
                "`scheduler_guest_*.log` 是否含 `[rdp2-invoke-stats]`。\n\n"
            )

        f.write("## 1) 总汇总（来自 scheduler_loop 摘要）\n\n")
        f.write("```\n")
        f.write(pre)
        f.write("\n```\n\n")

        if n > 0:
            f.write("## 2) 每迭代动态指令数（session 级）\n\n")
            f.write(f"- 会话数: `{n}`\n")
            f.write(f"- 合计指令（各 session insn_count 之和）: `{total_insns}`\n")
            f.write(f"- 最小 / 最大 / 平均: `{min(counts)}` / `{max(counts)}` / `{total_insns / n:.2f}`\n\n")

            by_cpu: dict[int, list[int]] = {}
            for _sid, _ph, cpu, insn in csv_rows:
                by_cpu.setdefault(cpu, []).append(insn)
            if len(by_cpu) > 1 or (len(by_cpu) == 1 and 0 not in by_cpu):
                f.write("### 2.1) 按 cpu_id\n\n")
                f.write("| cpu_id | 迭代数 | 合计指令 | 最小 | 最大 | 平均 |\n")
                f.write("|---:|---:|---:|---:|---:|---:|\n")
                for cid in sorted(by_cpu.keys()):
                    cc = by_cpu[cid]
                    s = sum(cc)
                    f.write(
                        f"| {cid} | {len(cc)} | {s} | {min(cc)} | {max(cc)} | {s / len(cc):.2f} |\n"
                    )
                f.write("\n")

            f.write("<details><summary>逐条 session（CSV 节选前 200 行）</summary>\n\n")
            f.write("```\n")
            f.write("session_id,phase_id,cpu_id,insn_count\n")
            for row in csv_rows[:200]:
                f.write(f"{row[0]},{row[1]},{row[2]},{row[3]}\n")
            if len(csv_rows) > 200:
                f.write(f"... ({len(csv_rows) - 200} more rows)\n")
            f.write("```\n\n</details>\n\n")
        else:
            f.write("## 2) 每迭代动态指令数\n\n")
            f.write("- 未解析到 session CSV 行；请检查 QEMU 插件是否成功写出 `session_id,...` 段。\n\n")

        if insn_types:
            base = insn_total if insn_total > 0 else max(total_insns, 1)
            f.write("## 3) 主要汇编指令分布（scheduler 标记窗口内聚合）\n\n")
            f.write("| 指令 | 次数 | 占比(相对窗口内指令类型合计) |\n")
            f.write("|---|---:|---:|\n")
            for op, cnt in insn_types[:40]:
                f.write(f"| {op} | {cnt} | {cnt / base * 100:.2f}% |\n")
            f.write("\n")
        else:
            f.write("## 3) 主要汇编指令分布\n\n")
            f.write("- 未提供或未生成 `scheduler_insn_types` 文件，跳过。\n\n")

        if mem_summary:
            f.write("## 4) 访存地址 Top N（scheduler 标记窗口）\n\n")
            f.write(
                f"- 访存总次数: `{mem_summary.get('total', 0)}`，"
                f"load: `{mem_summary.get('load', 0)}`，"
                f"store: `{mem_summary.get('store', 0)}`，"
                f"唯一地址数: `{mem_summary.get('unique_addr', 0)}`\n\n"
            )
            f.write("| 地址(vaddr) | total | load | store |\n")
            f.write("|---|---:|---:|---:|\n")
            for r in mem_top:
                f.write(
                    f"| `{r['vaddr']}` | {r['total']} | {r['load']} | {r['store']} |\n"
                )
            f.write("\n")
        else:
            f.write("## 4) 访存统计\n\n")
            f.write("- 无访存摘要数据。\n\n")

        if raw_tail:
            f.write("## 5) 原始插件输出附录\n\n")
            f.write("<details><summary>展开</summary>\n\n```\n")
            f.write(raw_tail[:120000])
            if len(raw_tail) > 120000:
                f.write("\n... [truncated]\n")
            f.write("\n```\n\n</details>\n")

    print(str(out))


if __name__ == "__main__":
    main()

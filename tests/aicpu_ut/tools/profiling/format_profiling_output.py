#!/usr/bin/env python3
"""
整理 run_tests.sh --profiling 的终端输出为统一格式。

不修改 main 分支自带的 run_tests.sh，仅对脚本输出做后处理：
- 去掉源码位置前缀 [file:line]
- 统一分隔线长度与样式
- 按参考格式组织各段（Config / Orchestrator / CPU affinity / Scheduler / 汇总）
- --profiling 1：从 DEV_ALWAYS 日志中提取 orchestrator/scheduler 时间，格式化为 Timing 汇总块
- 从 Part2 loop breakdown 表格提取每线程 loops/completed/total_us，追加到 Timing 汇总块

用法（不修改 run_tests.sh，仅整理其输出）:
  # 直接管道：运行 profiling 测试并整理输出
  bash run_tests.sh --profiling 2>&1 | python3 tools/format_profiling_output.py

  # 从已保存的完整输出整理
  python3 tools/format_profiling_output.py < saved_output.txt

  # 从单次测试的 log 文件整理（仅该测试内容）
  python3 tools/format_profiling_output.py outputs/log/test_latency_0_20260316_105051.log

输出格式与参考一致：
  - 去掉 [file:line] 前缀，保留两空格缩进
  - 主分隔线统一为 100 个等号
  - Config / Orchestrator / CPU affinity / Scheduler 小节分隔线统一
  - Timing 汇总块（来自 DEV_ALWAYS）：每测试结束前输出 orchestrator 和 scheduler 总时间
  - Loop breakdown 行（来自 Part2 表格）：追加到 Timing 汇总块
"""

import re
import sys
from pathlib import Path

# 分隔线：与参考格式一致
SEP_MAIN = "=" * 100
SEP_RUN  = "=" * 100
SEP_SUB  = "-" * 46   # Config / Orchestrator / CPU / Scheduler 等小节
SEP_65   = "=" * 65   # sched_overhead_analysis 用的短分隔线

# 匹配行首的 [filename:line] 及其后空白，替换为两空格以保持参考格式缩进
LOG_PREFIX = re.compile(r"^\[[^\]]+\]\s*")

# 替换后内容行统一缩进（与参考格式一致）
CONTENT_INDENT = "  "

AICPU_HEADER = "  -------------------------- AICPU scheduler loop breakdown (from log) -------------------------"

# ── Timing extraction patterns ────────────────────────────────────────────────
# Match: "Thread N: aicpu_orchestration_entry returned, cost X.XXXus (orch_idx=M)"
ORCH_TIME_RE = re.compile(
    r'Thread\s*(\d+)[:\s].*aicpu_orchestration_entry returned,\s*cost\s*([\d.]+)us.*orch_idx=(\d+)'
)
# Match: "Thread N: Scheduler summary: total_time=X.XXXus, loops=N, tasks_scheduled=M"
SCHED_TIME_RE = re.compile(
    r'Thread\s*(\d+)[:\s].*Scheduler summary:\s*total_time=([\d.]+)us,\s*loops=(\d+),\s*tasks_scheduled=(\d+)'
)
# Suppress: "Thread=N orch_start=NNN" (raw cycle count, noise)
ORCH_START_RE = re.compile(r'Thread[=\s]\d+\s+orch_start=\d+')
# Match: loop breakdown table row from Part2, e.g. "  T0  5121  1869  0.365  3463.4"
LOOP_ROW_RE = re.compile(
    r'^\s*(T\d+|SUM)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s*$'
)


def strip_log_prefix(line: str) -> str:
    """去掉行首的 [file:line] 前缀，保留两空格缩进。"""
    if LOG_PREFIX.match(line):
        return CONTENT_INDENT + LOG_PREFIX.sub("", line).lstrip()
    return line.rstrip()


def is_running_line(line: str) -> bool:
    return "  Running:" in line and ("test_" in line or "Test Run Summary" in line)


def flush_timing(out_list: list, orch: list, sched: list, loop: list) -> None:
    """输出当前测试的 orchestrator / scheduler / loop breakdown 时间汇总，然后清空列表。"""
    if not orch and not sched and not loop:
        return
    out_list.append("")
    out_list.append("  " + "-" * 56)
    # 按线程归并 orchestrator 各次调用耗时，输出总和
    orch_total: dict[int, float] = {}
    for (tid, cost, _oidx) in orch:
        orch_total[tid] = orch_total.get(tid, 0.0) + float(cost)
    for tid in sorted(orch_total):
        out_list.append(f"  Orchestrator  thread {tid}: {orch_total[tid]:>9.3f} us  (total)")
    for (tid, total, loops, tasks) in sorted(sched, key=lambda x: x[0]):
        out_list.append(
            f"  Scheduler     thread {tid}: {float(total):>9.3f} us"
            f"  (loops={loops}, tasks_scheduled={tasks})"
        )
    for (name, loops, completed, total_us) in loop:
        out_list.append(
            f"  Loop      {name:<6s}: {float(total_us):>9.1f} us"
            f"  (loops={loops}, completed={completed})"
        )
    orch.clear()
    sched.clear()
    loop.clear()


def main() -> None:
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if path.is_file():
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        else:
            sys.exit(f"Not a file: {path}")
    else:
        lines = [line.rstrip("\n") for line in sys.stdin]

    out: list[str] = []
    # 参考格式中只保留 Part2（AICPU scheduler loop breakdown），不输出 Part1/Part3 及 Perf data/Log/Selection
    skip_until_part2 = False   # 跳过 Perf data / Log / Selection 及 Part1 整块
    in_part2 = False           # 正在输出 Part2 内容
    skip_part3 = False         # 跳过 Part3 整块
    part2_just_header = False  # 下一行若是 65 等号则改为空行（参考格式中 Part2 标题与 Thread 表之间是空行）

    # 当前测试的 timing 收集列表（flush 在下一个 Running: 行或输出末尾）
    orch_timings: list = []   # [(thread_id, cost_us_str, orch_idx), ...]
    sched_timings: list = []  # [(thread_id, total_us_str, loops_str, tasks_str), ...]
    loop_rows: list = []      # [(name, loops_str, completed_str, total_us_str), ...]

    for raw in lines:
        line = strip_log_prefix(raw) if raw.strip() else raw.rstrip()
        stripped = line.strip()

        # 空行
        if not raw.strip():
            if in_part2 and not part2_just_header:
                out.append("")
            elif not skip_until_part2 and not skip_part3:
                out.append("")
            continue

        # ── Timing：抑制噪声行 orch_start= ──────────────────────────────────
        if ORCH_START_RE.search(raw) or ORCH_START_RE.search(stripped):
            continue

        # ── Timing：收集 orchestrator 时间行，并保留原始行到 log ──────────────────
        m = ORCH_TIME_RE.search(raw) or ORCH_TIME_RE.search(stripped)
        if m:
            rec = (int(m.group(1)), m.group(2), int(m.group(3)))
            # run_tests.sh 在 profiling=1 下可能会 re-emit 同一行，避免重复累计导致 total 翻倍
            if rec not in orch_timings:
                orch_timings.append(rec)
                out.append(line if line else raw.rstrip())  # 把原始行也打印到 log
            continue

        # ── Timing：收集 scheduler 时间行并抑制原始输出 ─────────────────────
        m = SCHED_TIME_RE.search(raw) or SCHED_TIME_RE.search(stripped)
        if m:
            rec = (int(m.group(1)), m.group(2), m.group(3), m.group(4))
            # profiling=1 可能重复出现同一条 summary；去重避免汇总重复展示
            if rec not in sched_timings:
                sched_timings.append(rec)
            continue

        # 主分隔线（100 等号）统一
        if stripped == "=" * 100 or stripped == "============================================================":
            if "Test Run Summary" in raw or (out and "Test Run Summary" in out[-1]):
                pass  # 下一行会是 Summary，不重复输出分隔线
            if skip_part3:
                skip_part3 = False
            out.append(SEP_MAIN)
            continue

        # 中等长度等号分隔线（sched_overhead_analysis 用 90，也有 65）：Part2 内改为空行；Part1/Part3 内忽略
        is_mid_sep = 61 <= len(stripped) <= 99 and set(stripped) <= {"="}
        if is_mid_sep:
            if part2_just_header:
                out.append("")
                part2_just_header = False
                continue
            if in_part2:
                out.append("")
                continue
            if skip_until_part2 or skip_part3:
                continue
            out.append(SEP_MAIN)  # 其他 65 等号统一成主分隔
            continue

        # 跳过 Part3 整块
        if skip_part3:
            continue
        if "Part 3:" in line or "Part 3:" in raw:
            skip_part3 = True
            in_part2 = False
            while out and out[-1] == "":
                out.pop()
            continue

        # 跳过 Perf data / Log (phase_breakdown) / Selection，并进入"跳过直到 Part2"
        if (
            stripped.startswith("Perf data:")
            or (stripped.startswith("Log:") and "phase_breakdown" in raw)
            or stripped.startswith("Selection:")
        ):
            skip_until_part2 = True
            continue

        # Part 1 整块跳过
        if "Part 1:" in line or "Part 1:" in raw:
            skip_until_part2 = True
            continue
        if skip_until_part2:
            if "Part 2:" in line or "Part 2:" in raw or "AICPU scheduler loop breakdown" in line:
                skip_until_part2 = False
                in_part2 = True
                out.append("")
                out.append(AICPU_HEADER)
                part2_just_header = True
                # 若本行是 "Part 2: ..." 则已处理；若是 "  -------------------------- AICPU ..." 则已输出
                if "3 scheduler threads" in line or "3 scheduler threads" in raw:
                    out.append("  3 scheduler threads")
                    part2_just_header = False
            continue

        # Part 2 标题行：统一为 AICPU scheduler loop breakdown
        if ("Part 2:" in line or "Part 2:" in raw) and "AICPU" in line:
            in_part2 = True
            out.append("")
            out.append(AICPU_HEADER)
            part2_just_header = True
            continue
        if "scheduler loop breakdown" in line.lower() and "Part" not in line and not in_part2:
            in_part2 = True
            out.append("")
            out.append(AICPU_HEADER)
            part2_just_header = True
            continue

        # Part 2 内的 "3 scheduler threads" 单独一行
        if in_part2 and ("3 scheduler threads" in line or "3 scheduler threads" in raw):
            out.append("  3 scheduler threads")
            part2_just_header = True
            continue

        # Test Run Summary：flush timing before printing summary header
        if "Test Run Summary" in stripped:
            flush_timing(out, orch_timings, sched_timings, loop_rows)

        # Running / Summary：在输出新 Running 行之前 flush 上一个测试的 timing
        if is_running_line(raw):
            flush_timing(out, orch_timings, sched_timings, loop_rows)
            in_part2 = False
            part2_just_header = False
            out.append(line)
            continue

        # 小节标题
        if "---------------------------------------------- Config" in line or "Config ----------------------------------------------" in line:
            out.append("---------------------------------------------- Config ----------------------------------------------")
            continue
        if "Orchestrator Profiling" in line and "------" in line:
            out.append("-------------------------------------- Orchestrator Profiling --------------------------------------")
            continue
        if "CPU affinity" in line and "------" in line:
            out.append("------------------------------------------- CPU affinity -------------------------------------------")
            continue
        if "Scheduler Profiling" in line and "------" in line:
            out.append("--------------------------------------- Scheduler Profiling ----------------------------------------")
            continue

        # ── Loop breakdown：在 Part2 内提取 T0/T1/T2/SUM 行数据，同时保留原始显示 ──
        if in_part2 and stripped:
            m = LOOP_ROW_RE.match(stripped)
            if m:
                loop_rows.append((m.group(1), m.group(2), m.group(3), m.group(5)))

        out.append(line if line else raw.rstrip())

    # 末尾 flush（最后一个测试的 timing）
    flush_timing(out, orch_timings, sched_timings, loop_rows)

    # 输出
    for x in out:
        print(x)


if __name__ == "__main__":
    main()

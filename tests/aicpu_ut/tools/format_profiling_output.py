#!/usr/bin/env python3
"""
整理 run_tests.sh --profiling 的终端输出为统一格式。

不修改 main 分支自带的 run_tests.sh，仅对脚本输出做后处理：
- 去掉源码位置前缀 [file:line]
- 统一分隔线长度与样式
- 按参考格式组织各段（Config / Orchestrator / CPU affinity / Scheduler / 汇总）

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


def strip_log_prefix(line: str) -> str:
    """去掉行首的 [file:line] 前缀，保留两空格缩进。"""
    if LOG_PREFIX.match(line):
        return CONTENT_INDENT + LOG_PREFIX.sub("", line).lstrip()
    return line.rstrip()


def is_running_line(line: str) -> bool:
    return "  Running:" in line and ("test_" in line or "Test Run Summary" in line)


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

        # 跳过 Perf data / Log (phase_breakdown) / Selection，并进入“跳过直到 Part2”
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

        # Running / Summary
        if is_running_line(raw):
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

        out.append(line if line else raw.rstrip())

    # 输出
    for x in out:
        print(x)


if __name__ == "__main__":
    main()

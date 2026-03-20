#!/usr/bin/env python3
"""
从 perf_swimlane_*.json 读取性能数据，打印「Task Statistics by Function」表格。

数据来源：带 --enable-profiling 的 device 跑完后，由 PerformanceCollector 导出到
outputs/perf_swimlane_*.json。每条 task 包含 func_id, start_time_us, end_time_us,
duration_us, dispatch_time_us, finish_time_us 等，本脚本按 func_id 聚合后输出与
run_example.py + swimlane_converter 一致的表格。

Usage:
    # 使用 outputs/ 下最新的 perf_swimlane_*.json
    python tools/print_task_stats.py

    # 指定 JSON 文件
    python tools/print_task_stats.py outputs/perf_swimlane_20260310_185122.json

    # 指定 kernel_config.py 以显示 QK/SF/PV/UP 等名称
    python tools/print_task_stats.py outputs/perf_swimlane_20260310_185122.json -k path/to/kernel_config.py

    # 同时指定 device log，会多打印 Sched CPU (from device log)
    python tools/print_task_stats.py perf_swimlane_xxx.json -k kernel_config.py --device-log /path/to/device-8_xxx.log
"""

import argparse
import sys
from pathlib import Path

# 支持从项目根或 tools 目录运行
if __name__ == "__main__":
    _tools_dir = Path(__file__).resolve().parent
    if _tools_dir.name == "tools" and str(_tools_dir.parent) not in sys.path:
        sys.path.insert(0, str(_tools_dir.parent))
    if str(_tools_dir) not in sys.path:
        sys.path.insert(0, str(_tools_dir))

from swimlane_converter import (
    read_perf_data,
    load_kernel_config,
    print_task_statistics,
    parse_sched_cpu_from_device_log,
)


def _latest_perf_json():
    """返回 outputs/ 下最新的 perf_swimlane_*.json 路径。"""
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "outputs"
    if not out_dir.exists():
        return None
    files = sorted(out_dir.glob("perf_swimlane_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def main():
    parser = argparse.ArgumentParser(
        description="从 perf_swimlane_*.json 读取数据并打印 Task Statistics by Function 表格",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "perf_json",
        nargs="?",
        default=None,
        help="perf_swimlane_*.json 路径；不填则使用 outputs/ 下最新文件",
    )
    parser.add_argument(
        "-k", "--kernel-config",
        help="kernel_config.py 路径，用于 func_id -> 名称（如 QK/SF/PV/UP）",
    )
    parser.add_argument(
        "--device-log",
        help="Device log 路径，用于解析 Sched CPU (from device log)",
    )
    args = parser.parse_args()

    path = args.perf_json
    if path is None:
        path = _latest_perf_json()
        if path is None:
            print("Error: No perf_swimlane_*.json found in outputs/. Run with --enable-profiling first or pass a JSON path.", file=sys.stderr)
            return 1
        print(f"Using latest: {path}", file=sys.stderr)
    else:
        path = Path(path)
    if not path.exists():
        print(f"Error: File not found: {path}", file=sys.stderr)
        return 1

    data = read_perf_data(path)
    tasks = data["tasks"]
    if not tasks:
        print("Error: No tasks in JSON.", file=sys.stderr)
        return 1

    func_id_to_name = None
    if args.kernel_config:
        kpath = Path(args.kernel_config)
        if not kpath.exists():
            print(f"Warning: kernel_config not found: {kpath}, using Func_N names.", file=sys.stderr)
        else:
            try:
                func_id_to_name = load_kernel_config(kpath)
            except Exception as e:
                print(f"Warning: Failed to load kernel_config: {e}", file=sys.stderr)

    sched_info = None
    if args.device_log:
        log_path = Path(args.device_log)
        if log_path.exists():
            sched_info = parse_sched_cpu_from_device_log(log_path, len(tasks))
        else:
            print(f"Warning: Device log not found: {log_path}", file=sys.stderr)

    print_task_statistics(tasks, func_id_to_name=func_id_to_name, sched_info=sched_info)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
organize_and_label_flamegraphs.py

用途（轻量版“找回脚本”）：
- 将 gen_sweep_flamegraphs.sh 生成的 outputs/ 目录按 throughput 的 dependency D 分组，方便对比：
  outputs/Depend8/...  ← 来自 throughput_*_D8_*
  outputs/Depend2/...  ← 来自 throughput_*_D2_*
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


THROUGHPUT_RE = re.compile(r"^throughput_.*_D(?P<D>\d+)_O(?P<O>\d+)_.*$")


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", required=True, help="gen_sweep_flamegraphs.sh 输出目录")
    parser.add_argument("--out", required=True, help="归档输出目录（可与 --outputs 相同）")
    args = parser.parse_args()

    outputs = Path(args.outputs).resolve()
    out = Path(args.out).resolve()

    if not outputs.is_dir():
        raise SystemExit(f"--outputs is not a directory: {outputs}")
    out.mkdir(parents=True, exist_ok=True)

    moved = 0
    kept = 0
    for child in sorted(outputs.iterdir()):
        if not child.is_dir():
            continue
        m = THROUGHPUT_RE.match(child.name)
        if not m:
            kept += 1
            continue

        depend_dir = out / f"Depend{m.group('D')}"
        depend_dir.mkdir(parents=True, exist_ok=True)
        copy_tree(child, depend_dir / child.name)
        moved += 1

    print(f"[organize] outputs={outputs}")
    print(f"[organize] out={out}")
    print(f"[organize] throughput grouped dirs: {moved}")
    print(f"[organize] untouched dirs (latency/others): {kept}")
    print("[organize] Note: this script copies directories; original outputs are kept.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

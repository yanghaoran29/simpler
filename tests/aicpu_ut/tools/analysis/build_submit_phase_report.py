#!/usr/bin/env python3
import argparse
import re
from collections import defaultdict


def parse_submit_args(path):
    pat = re.compile(
        r"\[submit-args\]\s+mixed=\(aic=(-?\d+),aiv0=(-?\d+),aiv1=(-?\d+)\)\s+"
        r"tensors=(\d+)\s+scalars=(\d+)\s+input=(\d+)\s+output=(\d+)\s+inout=(\d+)"
    )
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            rows.append(
                {
                    "aic": int(m.group(1)),
                    "aiv0": int(m.group(2)),
                    "aiv1": int(m.group(3)),
                    "tensors": int(m.group(4)),
                    "scalars": int(m.group(5)),
                    "input": int(m.group(6)),
                    "output": int(m.group(7)),
                    "inout": int(m.group(8)),
                }
            )
    return rows


def parse_trace_sessions(path):
    vals = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.search(r"^session\[(\d+)\]:\s*(\d+)\s*$", line.strip())
            if m:
                vals.append(int(m.group(2)))
    return vals


def grouped_stats(rows, phase_keys):
    g = defaultdict(list)
    for r in rows:
        key = (
            r["aic"],
            r["aiv0"],
            r["aiv1"],
            r["tensors"],
            r["scalars"],
            r["input"],
            r["output"],
            r["inout"],
        )
        g[key].append(r)
    out = []
    for k, vals in g.items():
        out.append((k, vals))
    return sorted(out, key=lambda x: x[0])


def fmt_phase_stats(name, vals):
    mn = min(vals)
    mx = max(vals)
    avg = sum(vals) / len(vals) if vals else 0.0
    return f"    {name}: min={mn}, max={mx}, avg={avg:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit-args-log", required=True)
    ap.add_argument("--submit-total-trace", required=True)
    ap.add_argument("--alloc-trace", required=True)
    ap.add_argument("--sync-trace", required=True)
    ap.add_argument("--lookup-trace", required=True)
    ap.add_argument("--insert-trace", required=True)
    ap.add_argument("--params-trace", required=True)
    ap.add_argument("--fanin-trace", required=True)
    ap.add_argument("--others-trace", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    submit_args = parse_submit_args(args.submit_args_log)
    phases = {
        "submit_total": parse_trace_sessions(args.submit_total_trace),
        "alloc": parse_trace_sessions(args.alloc_trace),
        "sync": parse_trace_sessions(args.sync_trace),
        "lookup": parse_trace_sessions(args.lookup_trace),
        "insert": parse_trace_sessions(args.insert_trace),
        "params": parse_trace_sessions(args.params_trace),
        "fanin": parse_trace_sessions(args.fanin_trace),
        "others": parse_trace_sessions(args.others_trace),
    }

    n = len(submit_args)
    for k, v in phases.items():
        if len(v) < n:
            raise RuntimeError(f"{k} sessions {len(v)} < submit-args {n}")
        if len(v) > n:
            phases[k] = v[:n]

    rows = []
    for i in range(n):
        r = dict(submit_args[i])
        for k in phases:
            r[k] = phases[k][i]
        rows.append(r)

    with open(args.output, "w", encoding="utf-8") as f:
        submit_total_sum = sum(r["submit_total"] for r in rows)
        f.write("# pto2_submit_mixed_task 指令统计（submit_total 外层区间，每项为一次提交）\n")
        f.write(f"pto2_submit_mixed_task 指令总数: {submit_total_sum}\n")
        f.write(f"任务数（提交次数）: {n}\n")
        f.write(f"平均每任务指令数: {submit_total_sum / n:.2f}\n")
        f.write(f"各任务指令数最小值: {min(r['submit_total'] for r in rows)}\n")
        f.write(f"各任务指令数最大值: {max(r['submit_total'] for r in rows)}\n\n")

        for r in rows:
            f.write(
                f"[submit-args] mixed=(aic={r['aic']},aiv0={r['aiv0']},aiv1={r['aiv1']}) "
                f"tensors={r['tensors']} scalars={r['scalars']} input={r['input']} output={r['output']} inout={r['inout']}\n"
            )
            f.write(f"submit_total: {r['submit_total']}\n")
            f.write(f"    alloc: {r['alloc']}\n")
            f.write(f"    sync: {r['sync']}\n")
            f.write(f"    lookup: {r['lookup']}\n")
            f.write(f"    insert: {r['insert']}\n")
            f.write(f"    params: {r['params']}\n")
            f.write(f"    fanin: {r['fanin']}\n")
            f.write(f"    others: {r['others']}\n")

        f.write("\n# 按 submit-args 分组统计（各 phase 的 min/max/avg）\n")
        phase_keys = ["submit_total", "alloc", "sync", "lookup", "insert", "params", "fanin", "others"]
        for key, vals in grouped_stats(rows, phase_keys):
            aic, aiv0, aiv1, tensors, scalars, inp, out, io = key
            f.write(
                f"\n[submit-args-group] mixed=(aic={aic},aiv0={aiv0},aiv1={aiv1}) "
                f"tensors={tensors} scalars={scalars} input={inp} output={out} inout={io} count={len(vals)}\n"
            )
            for ph in phase_keys:
                f.write(fmt_phase_stats(ph, [x[ph] for x in vals]) + "\n")


if __name__ == "__main__":
    main()


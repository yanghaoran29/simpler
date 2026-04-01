#!/usr/bin/env python3
"""
Parse submit_phases log: phases vs input/output/inout/scalars and optional kernel ids.
Usage:
  python3 analyze_submit_phases_vs_tensors.py <submit_phases_log.txt>
"""
import re
import sys
from collections import defaultdict

import numpy as np


def parse_log(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    blocks = []
    # One line submit-args
    pat = re.compile(
        r"\[submit-args\]\s+mixed=\(aic=(-?\d+),aiv0=(-?\d+),aiv1=(-?\d+)\)\s+"
        r"tensors=(\d+)\s+scalars=(\d+)\s+input=(\d+)\s+output=(\d+)\s+inout=(\d+)\s*\n"
        r"submit_total:\s*(\d+)\s*\n"
        r"\s*alloc:\s*(\d+)\s*\n"
        r"\s*sync:\s*(\d+)\s*\n"
        r"\s*lookup:\s*(\d+)\s*\n"
        r"\s*insert:\s*(\d+)\s*\n"
        r"\s*params:\s*(\d+)\s*\n"
        r"\s*fanin:\s*(\d+)\s*\n"
        r"\s*others:\s*(\d+)\s*",
        re.MULTILINE,
    )
    for m in pat.finditer(text):
        aic, aiv0, aiv1 = int(m.group(1)), int(m.group(2)), int(m.group(3))
        tensors = int(m.group(4))
        scalars = int(m.group(5))
        inp, out, inout = int(m.group(6)), int(m.group(7)), int(m.group(8))
        total = int(m.group(9))
        alloc = int(m.group(10))
        sync = int(m.group(11))
        lookup = int(m.group(12))
        insert = int(m.group(13))
        params = int(m.group(14))
        fanin = int(m.group(15))
        others = int(m.group(16))
        blocks.append(
            {
                "aic": aic,
                "aiv0": aiv0,
                "aiv1": aiv1,
                "tensors": tensors,
                "scalars": scalars,
                "input": inp,
                "output": out,
                "inout": inout,
                "Ieff": inp + inout,
                "Oeff": out + inout,
                "submit_total": total,
                "alloc": alloc,
                "sync": sync,
                "lookup": lookup,
                "insert": insert,
                "params": params,
                "fanin": fanin,
                "others": others,
            }
        )
    return blocks


def ols(y, X, names):
    """y: (n,), X: (n, k) design including intercept column if desired."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    resid = y - pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    mae = float(np.mean(np.abs(resid)))
    return coef, r2, mae, pred


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("usage: analyze_submit_phases_vs_tensors.py <log.txt>", file=sys.stderr)
        sys.exit(1)
    rows = parse_log(path)
    if not rows:
        print("no blocks parsed", file=sys.stderr)
        sys.exit(1)

    n = len(rows)
    print(f"parsed_records={n}  file={path}\n")

    # Feature matrix variants
    def as_matrix(keys, intercept=True):
        cols = []
        if intercept:
            cols.append(np.ones(n))
        for k in keys:
            cols.append(np.array([r[k] for r in rows], dtype=np.float64))
        return np.column_stack(cols), (["1"] if intercept else []) + list(keys)

    phases = ["alloc", "sync", "lookup", "insert", "params", "fanin", "others"]

    # Correlations (Pearson) between features and phases
    feat_keys = ["input", "output", "inout", "scalars", "tensors", "Ieff", "Oeff"]
    Xraw = np.array([[r[k] for k in feat_keys] for r in rows], dtype=np.float64)
    print("--- Pearson correlation (features vs phases) ---")
    header = "phase".ljust(10) + "".join(f"{k:>10}" for k in feat_keys)
    print(header)
    for ph in phases:
        yv = np.array([r[ph] for r in rows], dtype=np.float64)
        line = ph.ljust(10)
        for j, fk in enumerate(feat_keys):
            x = Xraw[:, j]
            if np.std(x) < 1e-9 or np.std(yv) < 1e-9:
                line += f"{0.0:10.3f}"
            else:
                line += f"{np.corrcoef(x, yv)[0, 1]:10.3f}"
        print(line)

    print("\n--- OLS: y ~ 1 + input + output + inout + scalars ---")
    X, xnames = as_matrix(["input", "output", "inout", "scalars"])
    for ph in phases:
        yv = np.array([r[ph] for r in rows], dtype=np.float64)
        coef, r2, mae, _ = ols(yv, X, xnames)
        print(f"  {ph:8s}  R2={r2:.4f}  MAE={mae:.2f}  " + " ".join(f"{a}*{n}={c:.3f}" for a, n, c in zip(["+", "+", "+", "+", "+"], xnames, coef)))

    print("\n--- OLS: y ~ 1 + Ieff + Oeff + scalars   (Ieff=input+inout, Oeff=output+inout) ---")
    X2, xnames2 = as_matrix(["Ieff", "Oeff", "scalars"])
    for ph in phases:
        yv = np.array([r[ph] for r in rows], dtype=np.float64)
        coef, r2, mae, _ = ols(yv, X2, xnames2)
        print(f"  {ph:8s}  R2={r2:.4f}  MAE={mae:.2f}  " + " ".join(f"{n}={c:.3f}" for n, c in zip(xnames2, coef)))

    # Grouped means by signature (input, output, inout) — unique configs in test
    print("\n--- Grouped mean (by input,output,inout) — counts per group ---")
    groups = defaultdict(list)
    for r in rows:
        key = (r["input"], r["output"], r["inout"])
        groups[key].append(r)
    for key in sorted(groups.keys()):
        g = groups[key]
        print(
            f"  I,O,IO={key}  n={len(g)}  "
            + " ".join(
                f"{ph}={sum(x[ph] for x in g)/len(g):.1f}" for ph in phases
            )
        )

    # Kernel shape: count active slots
    def kernel_tag(r):
        aic, a0, a1 = r["aic"], r["aiv0"], r["aiv1"]
        slots = sum(1 for x in (aic, a0, a1) if x >= 0)
        return slots

    print("\n--- Mean submit_total by kernel slot count (aic/aiv0/aiv1 valid) ---")
    kg = defaultdict(list)
    for r in rows:
        kg[kernel_tag(r)].append(r["submit_total"])
    for k in sorted(kg.keys()):
        v = kg[k]
        print(f"  active_slots={k}  n={len(v)}  mean_total={sum(v)/len(v):.1f}")

    print("\nNote: 本日志若未单独列出 heap，则 orchestrator 中 heap 阶段可能并入 alloc/others；")
    print("      七项 alloc,sync,lookup,insert,params,fanin,others 之和 = submit_total。")


if __name__ == "__main__":
    main()

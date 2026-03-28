#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from collections import defaultdict

TASK_TYPE_TO_SUBMIT = {
    # batch_paged_attention orchestration/paged_attention_orch.cpp
    # params_hub
    (0, 3, 0, 0): "AIV_HUB (FUNC_AIV_HUB)",
    # params_qk
    (2, 1, 0, 7): "QK_MATMUL (FUNC_QK_MATMUL)",
    # params_sf
    (1, 3, 0, 5): "SOFTMAX_PREPARE (FUNC_SOFTMAX_PREPARE)",
    # params_pv
    (2, 1, 0, 5): "PV_MATMUL (FUNC_PV_MATMUL)",
    # params_up (log-observed tuple in this build)
    (3, 2, 2, 6): "ONLINE_UPDATE (FUNC_ONLINE_UPDATE)",
}


def strip_submit_name(task_name):
    if not task_name:
        return "UNKNOWN"
    return task_name.split("(", 1)[0].strip()


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
            aic, aiv0, aiv1, tensors, scalars, inp, out, inout = map(int, m.groups())
            rows.append(
                {
                    "aic": aic,
                    "aiv0": aiv0,
                    "aiv1": aiv1,
                    "tensors": tensors,
                    "scalars": scalars,
                    "input": inp,
                    "output": out,
                    "inout": inout,
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


def parse_insn_types(path):
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            m = re.match(r"([A-Za-z0-9._]+)\s+(\d+)$", s)
            if m:
                out.append((m.group(1), int(m.group(2))))
    return out


def parse_mem_top(path, topn=20):
    summary = {}
    rows = []
    in_section = False
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("# memory access summary"):
                in_section = True
                continue
            if in_section and s.startswith("# total="):
                m = re.search(r"total=(\d+)\s+load=(\d+)\s+store=(\d+)\s+unique_addr=(\d+)", s)
                if m:
                    summary = {
                        "total": int(m.group(1)),
                        "load": int(m.group(2)),
                        "store": int(m.group(3)),
                        "unique_addr": int(m.group(4)),
                    }
                continue
            if not in_section:
                continue
            if s.startswith("#"):
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
    ap.add_argument("--insn-types", required=True)
    ap.add_argument("--build-graph-insns", type=int, default=-1)
    ap.add_argument(
        "--test-name",
        default="test_batch_paged_attention",
        help="Report title (default: test_batch_paged_attention)",
    )
    ap.add_argument("--report-idx", default="0", help="Index shown in report title")
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
    insn_types = parse_insn_types(args.insn_types)
    insn_total = sum(cnt for _, cnt in insn_types)
    mem_summary, mem_top = parse_mem_top(args.insn_types, topn=20)

    n = len(submit_args)
    for k, v in phases.items():
        if len(v) < n:
            raise RuntimeError(f"{k} sessions {len(v)} < submit-args {n}")
        if len(v) > n:
            phases[k] = v[:n]

    rows = []
    for i in range(n):
        r = {"task_id": i + 1}
        r.update(submit_args[i])
        for k in phases:
            r[k] = phases[k][i]
        rows.append(r)

    total = sum(x["submit_total"] for x in rows)
    total_for_dist = insn_total if insn_total > 0 else total
    phase_keys = ["submit_total", "alloc", "sync", "lookup", "insert", "params", "fanin", "others"]

    # Task type grouped by (input, output, inout, scalars)
    groups = defaultdict(list)
    for r in rows:
        key = (r["input"], r["output"], r["inout"], r["scalars"])
        groups[key].append(r)
    out = Path(args.output)
    with out.open("w", encoding="utf-8") as f:
        f.write(f"# {args.test_name} idx={args.report_idx} 提交任务分阶段指令报告\n\n")
        if args.build_graph_insns >= 0 and n > 0:
            other_insns = args.build_graph_insns - total
            avg_task_insns = args.build_graph_insns / n
            avg_submit_insns = total / n
            f.write("## 1) 总汇总表\n\n")
            f.write(f"- build_graph 指令数: `{args.build_graph_insns}`\n")
            f.write(f"- pto2_submit_mixed_task 总指令数: `{total}`\n")
            f.write(f"- 其他部分指令数: `{other_insns}`\n\n")
            f.write(f"- 任务总数: `{n}`\n")
            f.write(f"- 平均每任务指令数: `{avg_task_insns:.2f}`\n")
            f.write(f"- 平均每次pto2_submit_mixed_task指令数: `{avg_submit_insns:.2f}`\n\n")
            f.write(f"- phase统计submit_total总指令数: `{total}`\n\n")
        else:
            if args.build_graph_insns >= 0:
                f.write(f"- build_graph 指令数: `{args.build_graph_insns}`\n")
            f.write(f"- 任务总数: `{n}`\n")
            f.write(f"- pto2_submit_mixed_task 总指令数: `{total_for_dist}`\n")
            f.write(f"- phase统计submit_total总指令数: `{total}`\n")
            other_insns = args.build_graph_insns if args.build_graph_insns >= 0 else 0
            f.write(f"- 其他部分指令数: `{other_insns}`\n\n")
        if n == 0:
            f.write("## 数据说明\n\n")
            f.write("- 未捕获到 submit-args/phase trace，无法生成任务级和任务类型分组统计。\n")
            f.write("- 请确认运行链路包含 submit-args 打点和 phase trace 输出。\n\n")
            if insn_types:
                f.write("## 主要汇编指令分布\n\n")
                f.write("| 指令 | 次数 |\n")
                f.write("|---|---:|\n")
                for op, cnt in insn_types[:20]:
                    f.write(f"| {op} | {cnt} |\n")
            return

        f.write("### 2) 平均每阶段的指令数\n\n")
        f.write("| 阶段 | 平均指令数 | 最小值 | 最大值 |\n")
        f.write("|---|---:|---:|---:|\n")
        for k in ["alloc", "sync", "lookup", "insert", "params", "fanin", "others", "submit_total"]:
            vals = [x[k] for x in rows]
            f.write(f"| {k} | {sum(vals)/n:.3f} | {min(vals)} | {max(vals)} |\n")

        f.write("\n### 2.1) 按任务类型分组统计（input/output/inout/scalar）\n\n")
        f.write("- 统计项为每个任务类型在总体与各阶段上的平均/最小/最大指令数。\n\n")
        f.write("| submit任务名 | 任务类型(input,output,inout,scalar) | 样本数 |\n")
        f.write("|---|---|---:|\n")
        for key in sorted(groups.keys()):
            task_name = strip_submit_name(TASK_TYPE_TO_SUBMIT.get(key, "UNKNOWN"))
            f.write(f"| `{task_name}` | `{key}` | {len(groups[key])} |\n")
        f.write("\n")
        for key in sorted(groups.keys()):
            inp, outp, inout, scalars = key
            items = groups[key]
            task_name = TASK_TYPE_TO_SUBMIT.get(key, "UNKNOWN")
            task_name_short = strip_submit_name(task_name)
            f.write(
                f"#### {task_name_short} (样本数={len(items)})\n\n"
            )
            f.write(f"- 对应 submit 任务名: `{task_name}`\n")
            f.write(f"- 参数属性: `input={inp}, output={outp}, inout={inout}, scalar={scalars}`\n\n")
            f.write("| 指标 | 平均 | 最小 | 最大 |\n")
            f.write("|---|---:|---:|---:|\n")
            for ph in phase_keys:
                vals = [x[ph] for x in items]
                f.write(f"| {ph} | {sum(vals)/len(vals):.3f} | {min(vals)} | {max(vals)} |\n")
            f.write("\n")

        f.write("\n### 3) 主要汇编指令分布\n\n")
        f.write("| 指令 | 次数 | 占比(相对submit_total总指令) |\n")
        f.write("|---|---:|---:|\n")
        for op, cnt in insn_types[:20]:
            f.write(f"| {op} | {cnt} | {cnt / total_for_dist * 100:.2f}% |\n")

        if mem_summary:
            f.write("\n### 4) 访存地址 Top N（来自 insn_types 采集）\n\n")
            f.write(
                f"- 访存总次数: `{mem_summary.get('total', 0)}`，"
                f"load: `{mem_summary.get('load', 0)}`，"
                f"store: `{mem_summary.get('store', 0)}`，"
                f"唯一地址数: `{mem_summary.get('unique_addr', 0)}`\n\n"
            )
            f.write("| 地址(vaddr) | total | load | store |\n")
            f.write("|---|---:|---:|---:|\n")
            for r in mem_top:
                f.write(f"| `{r['vaddr']}` | {r['total']} | {r['load']} | {r['store']} |\n")

        f.write("\n")
        for r in rows:
            f.write(f"## 任务编号 {r['task_id']}\n")
            f.write(f"- input数量: `{r['input']}`\n")
            f.write(f"- output数量: `{r['output']}`\n")
            f.write(f"- inout数量: `{r['inout']}`\n")
            f.write(f"- scalar数量: `{r['scalars']}`\n")
            f.write(f"- alloc指令数: `{r['alloc']}`\n")
            f.write(f"- sync指令数: `{r['sync']}`\n")
            f.write(f"- lookup指令数: `{r['lookup']}`\n")
            f.write(f"- insert指令数: `{r['insert']}`\n")
            f.write(f"- params指令数: `{r['params']}`\n")
            f.write(f"- fanin指令数: `{r['fanin']}`\n")
            f.write(f"- others指令数: `{r['others']}`\n")
            f.write(f"- submit_total指令数: `{r['submit_total']}`\n\n")

    print(str(out))


if __name__ == "__main__":
    main()


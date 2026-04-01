#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from collections import defaultdict

TASK_NAME_BY_AIV0_KERNEL_ID = {
    # Optional human-readable aliases for known kernels.
    # Example:
    # 3: "AIV_HUB (FUNC_AIV_HUB)",
}

KERNEL_DESCRIPTION_BY_ID = {
    0: "RMS 统计 + 逆均方根准备（prefill 前置）",
    1: "RMSNorm（输入）",
    2: "Q 投影 GEMM（AIC）",
    3: "Q 投影结果整理/写回（AIV）",
    4: "K/V 投影 GEMM（AIC）",
    5: "K/V 投影结果整理/写回（AIV）",
    6: "初始化注意力输出 tile",
    7: "提取当前 token 的 K group",
    8: "RoPE + 写 KV Cache",
    9: "初始化单行注意力累加器",
    10: "提取当前 token 的 Q group",
    11: "Q 旋转编码（RoPE）",
    12: "初始化在线 softmax 状态（li/mi/oi）",
    13: "QK MatMul（分块上下文）",
    14: "Softmax 准备（max/sum/exp）",
    15: "PV MatMul（分块上下文）",
    16: "在线 softmax 状态更新（融合）",
    17: "回写注意力行（按头组）",
    18: "取 hidden 残差分支切片",
    19: "WO 投影 GEMM（AIC）",
    20: "WO 后处理 + residual add（AIV）",
    21: "RMS 统计 + 逆均方根准备（post-attn）",
    22: "Post-RMSNorm + 初始化 down_proj",
    23: "FFN gate/up GEMM（AIC）",
    24: "FFN 激活与 chunk 融合（AIV）",
    25: "FFN down GEMM（AIC）",
    26: "FFN down 累加回写（AIV）",
    27: "最终输出回写（residual + down_proj）",
}


def strip_submit_name(task_name):
    if not task_name:
        return "UNKNOWN"
    return task_name.split("(", 1)[0].strip()


def submit_task_name_from_row(r):
    """
    使用 pto2_rt_submit_aiv_task(kernel_id, ...) 的第一个参数（aiv0_kernel_id）作为 submit 任务名主键。
    若当前提交不是 AIV-only（aiv0<0），再回退到 mixed/aic 表示。
    """
    aiv0 = int(r.get("aiv0", -1))
    aiv1 = int(r.get("aiv1", -1))
    aic = int(r.get("aic", -1))
    if aiv0 >= 0:
        alias = TASK_NAME_BY_AIV0_KERNEL_ID.get(aiv0)
        return alias if alias else f"AIV_KERNEL_{aiv0}"
    if aiv1 >= 0:
        return f"AIV1_KERNEL_{aiv1}"
    if aic >= 0:
        return f"AIC_KERNEL_{aic}"
    return "UNKNOWN"


def submit_task_desc_from_row(r):
    aiv0 = int(r.get("aiv0", -1))
    aiv1 = int(r.get("aiv1", -1))
    aic = int(r.get("aic", -1))
    if aiv0 >= 0:
        return KERNEL_DESCRIPTION_BY_ID.get(aiv0, "AIV 提交任务（未标注）")
    if aiv1 >= 0:
        return KERNEL_DESCRIPTION_BY_ID.get(aiv1, "AIV1 提交任务（未标注）")
    if aic >= 0:
        return KERNEL_DESCRIPTION_BY_ID.get(aic, "AIC 提交任务（未标注）")
    return "未知任务"


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
            m = re.search(r"^session\[(\d+)\]:\s*(-?\d+)\s*$", line.strip())
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

    # Group by submit task name (primary key: aiv0 kernel_id from pto2_rt_submit_aiv_task).
    groups = defaultdict(list)
    for r in rows:
        groups[submit_task_name_from_row(r)].append(r)
    out = Path(args.output)
    detail_out = out.with_name(f"{out.stem}_detail.md")
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

        f.write("\n### 2.1) 按 submit 任务名分组统计（主键: pto2_rt_submit_aiv_task 第一个参数）\n\n")
        f.write("- submit任务名优先使用 `aiv0_kernel_id`（即 `pto2_rt_submit_aiv_task(kernel_id, ...)` 的 kernel_id）。\n")
        f.write("- 非 AIV-only 提交会回退显示为 AIC/MIXED 内核标识。\n\n")
        f.write("| submit任务名 | 任务描述 | 样本数 | 参数类型覆盖(input,output,inout,scalar) |\n")
        f.write("|---|---|---:|---|\n")
        for task_name in sorted(groups.keys()):
            items = groups[task_name]
            desc = submit_task_desc_from_row(items[0]) if items else "未知任务"
            type_keys = sorted({(x["input"], x["output"], x["inout"], x["scalars"]) for x in items})
            type_keys_str = ", ".join(str(k) for k in type_keys[:3])
            if len(type_keys) > 3:
                type_keys_str += f", ...(+{len(type_keys)-3})"
            f.write(f"| `{strip_submit_name(task_name)}` | {desc} | {len(items)} | `{type_keys_str}` |\n")
        f.write("\n")
        for task_name in sorted(groups.keys()):
            items = groups[task_name]
            task_name_short = strip_submit_name(task_name)
            aiv0_vals = sorted({x["aiv0"] for x in items})
            aic_vals = sorted({x["aic"] for x in items})
            aiv1_vals = sorted({x["aiv1"] for x in items})
            f.write(
                f"#### {task_name_short} (样本数={len(items)})\n\n"
            )
            f.write(f"- 对应 submit 任务名: `{task_name}`\n")
            f.write(f"- kernel_id集合: `aiv0={aiv0_vals}, aiv1={aiv1_vals}, aic={aic_vals}`\n")
            io_type_keys = sorted({(x['input'], x['output'], x['inout'], x['scalars']) for x in items})
            f.write(f"- 参数类型覆盖(input,output,inout,scalar): `{io_type_keys}`\n\n")
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

        f.write("\n### 5) 任务级明细\n\n")
        f.write(f"- 任务级明细已拆分到：`{detail_out.name}`\n\n")

    with detail_out.open("w", encoding="utf-8") as df:
        df.write(f"# {args.test_name} idx={args.report_idx} 提交任务分阶段指令明细\n\n")
        df.write(f"- 由主报告 `{out.name}` 拆分生成。\n\n")
        for r in rows:
            df.write(f"## 任务编号 {r['task_id']}\n")
            df.write(f"- input数量: `{r['input']}`\n")
            df.write(f"- output数量: `{r['output']}`\n")
            df.write(f"- inout数量: `{r['inout']}`\n")
            df.write(f"- scalar数量: `{r['scalars']}`\n")
            df.write(f"- alloc指令数: `{r['alloc']}`\n")
            df.write(f"- sync指令数: `{r['sync']}`\n")
            df.write(f"- lookup指令数: `{r['lookup']}`\n")
            df.write(f"- insert指令数: `{r['insert']}`\n")
            df.write(f"- params指令数: `{r['params']}`\n")
            df.write(f"- fanin指令数: `{r['fanin']}`\n")
            df.write(f"- others指令数: `{r['others']}`\n")
            df.write(f"- submit_total指令数: `{r['submit_total']}`\n\n")

    print(str(out))


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Generate tensor I/O CSVs for tensormap_and_ringbuffer ST samples (per orchestration code).

依据各子目录 kernels/orchestration/*.cpp 中 pto2_rt_submit_* 前的
add_input / add_inout / add_output 次数（不含 add_scalar）。

输出：
  tensor_io_per_task.csv — 每行一种 task_name，task_count_in_case 为该 Case 一次运行中的提交次数
  tensor_io_weighted_avg_per_case.csv — 各 Case 内按任务次数加权的平均 input/inout/output

重新生成：python3 generate_tensor_io_csvs.py
"""

from __future__ import annotations

import csv
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent

N_UNROLL = 64
IN_CORE_BATCH = 16

# Paged-attention family: fixed per submit (add_input / add_inout / add_output counts)
PA_ROWS = [
    ("AIV_HUB", 0, 0, 3),
    ("QK_MATMUL", 2, 0, 1),
    ("SOFTMAX_PREPARE", 1, 0, 3),
    ("PV_MATMUL", 2, 0, 1),
    ("ONLINE_UPDATE", 3, 4, 0),
]


def q_loop_from_heads(num_heads: int) -> int:
    q_tile = min(num_heads, 128)
    return (num_heads + q_tile - 1) // q_tile


def bn_from_ctx(context_len: int, block_size: int) -> int:
    return (context_len + block_size - 1) // block_size


def append_pa_family(
    rows_detail: list,
    rows_avg: list,
    sample: str,
    case_name: str,
    batch: int,
    num_heads: int,
    block_size: int,
    context_len: int,
    mode: str,
) -> None:
    """mode: paged_attention | paged_attention_unroll | batch_paged_attention"""
    ql = q_loop_from_heads(num_heads)
    bn = bn_from_ctx(context_len, block_size)

    if mode == "paged_attention":
        c_hub = batch * ql
        g = bn
        c_stage = batch * ql * g  # per-stage count for QK/SF/PV/UP
    elif mode == "paged_attention_unroll":
        c_hub = batch * ql
        g = (bn + N_UNROLL - 1) // N_UNROLL
        c_stage = batch * ql * g
    elif mode == "batch_paged_attention":
        num_chunks = (batch + IN_CORE_BATCH - 1) // IN_CORE_BATCH
        c_hub = ql * num_chunks
        c_stage = ql * num_chunks * bn
    else:
        raise ValueError(mode)

    total_tasks = c_hub + 4 * c_stage
    for name, inn, ino, out in PA_ROWS:
        cnt = c_hub if name == "AIV_HUB" else c_stage
        rows_detail.append([sample, case_name, name, inn, ino, out, cnt])

    tot_in = 8 * c_stage
    tot_ino = 4 * c_stage
    tot_out = 3 * c_hub + 5 * c_stage
    rows_avg.append(
        [
            sample,
            case_name,
            total_tasks,
            tot_in / total_tasks,
            tot_ino / total_tasks,
            tot_out / total_tasks,
        ]
    )


def main() -> None:
    rows_detail: list[list] = []
    rows_avg: list[list] = []

    # --- paged_attention / unroll / batch_paged: same Case1-3 params ---
    cases_common = {
        "Case1": dict(batch=256, num_heads=16, block_size=128, context_len=8192),
        "Case2": dict(batch=64, num_heads=64, block_size=64, context_len=8192),
        "Case3": dict(batch=64, num_heads=64, block_size=64, context_len=8192),
    }
    # Order: 按样例分组，再按 Case1→Case3
    pa_variants: list[tuple[str, str]] = [
        ("paged_attention", "paged_attention"),
        ("paged_attention_unroll", "paged_attention_unroll"),
        ("batch_paged_attention", "batch_paged_attention"),
    ]
    for sample, mode in pa_variants:
        for case_name, p in cases_common.items():
            append_pa_family(
                rows_detail,
                rows_avg,
                sample,
                case_name,
                p["batch"],
                p["num_heads"],
                p["block_size"],
                p["context_len"],
                mode,
            )

    # paged_attention_ringbuffer: Case1 only (from golden.py)
    append_pa_family(
        rows_detail,
        rows_avg,
        "paged_attention_ringbuffer",
        "Case1",
        32,
        16,
        128,
        4096,
        "paged_attention",
    )

    # --- scalar_data_test: single graph, 8 submits ---
    scalar_tasks = [
        ("kernel_add_c=a+b", 2, 0, 1),
        ("noop_runtime_scalar_init", 0, 0, 1),
        ("noop_scalar_inout", 0, 1, 0),
        ("noop_alloc_d", 0, 0, 1),
        ("kernel_add_d_plus_a", 2, 0, 1),
        ("kernel_add_WAW_consumer", 2, 0, 1),
        ("noop_ext_b_output", 0, 0, 1),
        ("kernel_add_result", 2, 0, 1),
    ]
    st_total = 8
    st_in = sum(x[1] for x in scalar_tasks)
    st_ino = sum(x[2] for x in scalar_tasks)
    st_out = sum(x[3] for x in scalar_tasks)
    for name, inn, ino, out in scalar_tasks:
        rows_detail.append(["scalar_data_test", "default", name, inn, ino, out, 1])
    rows_avg.append(
        [
            "scalar_data_test",
            "default",
            st_total,
            st_in / st_total,
            st_ino / st_total,
            st_out / st_total,
        ]
    )

    # --- alternating_matmul_add ---
    alt_cases = {
        "Case1": dict(batch=500, M=4, N=4, matmul_batch=4, add_batch=4),
        "Case2": dict(batch=512, M=2, N=5, matmul_batch=4, add_batch=5),
    }
    for cname, p in alt_cases.items():
        tmm = p["batch"] * p["M"]
        tad = p["batch"] * p["N"]
        nmg = tmm // p["matmul_batch"]
        nag = tad // p["add_batch"]
        cnt_mm = nmg
        cnt_ad = nag
        total = cnt_mm + cnt_ad
        # detail rows
        rows_detail.append(["alternating_matmul_add", cname, "AIC_MATMUL", 2, 0, 1, cnt_mm])
        rows_detail.append(["alternating_matmul_add", cname, "AIV_ADD", 2, 0, 1, cnt_ad])
        tot_in = 2 * total
        tot_out = 1 * total
        rows_avg.append(
            [
                "alternating_matmul_add",
                cname,
                total,
                tot_in / total,
                0.0,
                tot_out / total,
            ]
        )

    # --- benchmark_bgemm: per (group,k): GEMM then TILE_ADD ---
    bgemm_cases = {
        "Case0": (500, 2),
        "Case1": (64, 2),
        "Case2": (256, 2),
        "Case3": (64, 2),
        "Case4": (64, 4),
    }
    for cname, (matmul_add_task_num, grid_k) in bgemm_cases.items():
        num_groups = matmul_add_task_num // grid_k
        cnt_gemm = num_groups * grid_k
        cnt_add = num_groups * grid_k
        total = cnt_gemm + cnt_add
        rows_detail.append(["benchmark_bgemm", cname, "AIC_GEMM_TILE", 3, 0, 1, cnt_gemm])
        rows_detail.append(["benchmark_bgemm", cname, "AIV_TILE_ADD", 2, 1, 0, cnt_add])
        tot_in = cnt_gemm * 3 + cnt_add * 2
        tot_ino = cnt_add * 1
        tot_out = cnt_gemm * 1
        rows_avg.append(
            [
                "benchmark_bgemm",
                cname,
                total,
                tot_in / total,
                tot_ino / total,
                tot_out / total,
            ]
        )

    csv1 = OUT_DIR / "tensor_io_per_task.csv"
    csv2 = OUT_DIR / "tensor_io_weighted_avg_per_case.csv"

    with csv1.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "sample",
                "case",
                "task_name",
                "input_count",
                "inout_count",
                "output_count",
                "task_count_in_case",
            ]
        )
        w.writerows(rows_detail)

    with csv2.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "sample",
                "case",
                "total_tasks",
                "weighted_avg_input",
                "weighted_avg_inout",
                "weighted_avg_output",
            ]
        )
        for row in rows_avg:
            w.writerow(
                [
                    row[0],
                    row[1],
                    int(row[2]),
                    f"{row[3]:.12g}",
                    f"{row[4]:.12g}",
                    f"{row[5]:.12g}",
                ]
            )

    print(f"Wrote {csv1}")
    print(f"Wrote {csv2}")


if __name__ == "__main__":
    main()

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""由 ``block_table`` 驱动的分页投影 + 分页 RMSNorm 样例。

精简为 "cube 生产者 + vec 消费者" 一对, 用来展示 ``block_table`` 间接寻址下
正确的任务切片方式。Qwen3 / DeepSeek paged-KV-cache decode 里同型用法见
``models/qwen3/14b/decode_layer.py`` (``q_proj`` / ``qk_norm`` / ``fa_qks``)
与 ``models/deepseek/v4/decode_attention_swa.py``。

Stage 1 —— ``paged_proj`` (AIC, 纯 cube):
    每个页号 ob 上, 在 orchestration 层先切 ``x_page`` ([PAGE_M, HIDDEN])
    视图, 再进入 ``pl.at`` 做 K 维顺序累加 (首拍 ``pl.matmul``, 其余
    ``pl.matmul_acc``), FP32 结果 ``pl.assemble`` 写入 ``paged_y``。pl.at
    内只有 cube + MTE 算子。不引用 ``block_table``。

Stage 2 —— ``paged_rmsnorm`` (AIV, 纯 vec):
    每个输出页号 ob 上, 在 orchestration 层用 ``pl.tensor.read(block_table,
    [ob])`` 解出物理源页号, 切 ``y_src = paged_y[page_id*PAGE_M : ...]`` 视图,
    再进入 ``pl.at`` 做标准 RMSNorm。pl.at 内只有 vec + MTE 算子。

把 per-page 的 ``pl.slice`` / ``pl.tensor.read`` 放在 ``pl.at`` *外* 是关键:
生成的 orchestrator 会以小块视图作为每个任务的输入/输出依赖, 而不是把整张
``x`` / ``paged_y`` 加一个 scalar ``ob`` 传进 kernel —— 后者会让任务图退化成
"stage 2 全等 stage 1 完成" 的粗粒度依赖, ``block_table`` 间接寻址就形同虚设。
"""

import pypto.language as pl

# ---------------------------------------------------------------------------
# Problem shape — small but representative of an LLM projection layer.
# Tile shape ([PAGE_M, N{1,2}] = [16, 256], K_CHUNK = 128) matches the
# validated cube/vec budget of examples/advanced/multi_proj.py so both
# stages fit within Mat (524288 B) and Vec (196608 B) buffer limits.
# ---------------------------------------------------------------------------
NUM_PAGES = 16
PAGE_M    = 16                          # rows per page (cube-friendly M)
BATCH     = NUM_PAGES * PAGE_M          # 256

HIDDEN    = 2048                        # stage-1 K
N1        = 256                         # stage-1 N (= stage-2 RMSNorm width)

K1_CHUNK  = 128                         # stage-1 K tile  -> 16 K-iterations

EPS       = 1.0e-6                      # RMSNorm epsilon


@pl.jit
def paged_consumer_block_table(
    x:           pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    w1:          pl.Tensor[[HIDDEN, N1],    pl.BF16],
    gamma:       pl.Tensor[[1, N1],         pl.FP32],
    block_table: pl.Tensor[[NUM_PAGES],     pl.INT32],
    out:         pl.Out[pl.Tensor[[BATCH, N1], pl.FP32]],
):
    k1_blocks = HIDDEN // K1_CHUNK

    # Intermediate filled by stage 1 (cube) and gathered by stage 2 (vec).
    # FP32 keeps stage 1 a pure cube region (no cast epilogue).
    paged_y = pl.create_tensor([NUM_PAGES * PAGE_M, N1], dtype=pl.FP32)

    # Stage 1: pure-cube projection (AIC). x_page is sliced outside pl.at
    # so each task's data dep is the [PAGE_M, HIDDEN] strip, not all of x.
    for ob in pl.parallel(0, NUM_PAGES):
        m0 = ob * PAGE_M
        x_page = pl.slice(x, [PAGE_M, HIDDEN], [m0, 0])
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="paged_proj"):
            acc1 = pl.create_tensor([PAGE_M, N1], dtype=pl.FP32)
            tile_x0  = pl.slice(x_page, [PAGE_M,   K1_CHUNK], [0, 0])
            tile_w10 = pl.slice(w1,     [K1_CHUNK, N1],       [0, 0])
            acc1 = pl.matmul(tile_x0, tile_w10, out_dtype=pl.FP32)
            for kb in pl.range(1, k1_blocks):
                k0 = kb * K1_CHUNK
                tile_x  = pl.slice(x_page, [PAGE_M,   K1_CHUNK], [0, k0])
                tile_w1 = pl.slice(w1,     [K1_CHUNK, N1],       [k0, 0])
                acc1 = pl.matmul_acc(acc1, tile_x, tile_w1)
            paged_y = pl.assemble(paged_y, acc1, [m0, 0])

    # Stage 2: pure-vec RMSNorm (AIV). block_table read + y_src slice are
    # hoisted out of pl.at so the task graph carries the per-page edge
    # stage2[ob] -> stage1[block_table[ob]] explicitly.
    for ob in pl.parallel(0, NUM_PAGES):
        page_id = pl.cast(pl.tensor.read(block_table, [ob]), pl.INDEX)
        src_row = page_id * PAGE_M
        out_m0  = ob * PAGE_M
        y_src   = pl.slice(paged_y, [PAGE_M, N1], [src_row, 0])
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="paged_rmsnorm"):
            gamma_tile = pl.slice(gamma, [1, N1], [0, 0])

            sq_sum  = pl.row_sum(pl.mul(y_src, y_src))   # [PAGE_M, 1]
            inv_rms = pl.rsqrt(pl.add(pl.mul(sq_sum, 1.0 / N1), EPS))
            normed  = pl.col_expand_mul(
                pl.row_expand_mul(y_src, inv_rms),
                gamma_tile,
            )
            out = pl.assemble(out, normed, [out_m0, 0])

    return out


def build_tensor_specs():
    import torch

    from golden import TensorSpec

    scale_h = HIDDEN ** 0.5

    def init_x():
        return torch.rand(BATCH, HIDDEN) - 0.5

    def init_w1():
        return (torch.rand(HIDDEN, N1) - 0.5) / scale_h

    def init_gamma():
        # Per-column RMSNorm scale; small jitter around 1.0, mirroring the
        # q_norm / k_norm weight shape (~1.0) in Qwen3.
        return 1.0 + 0.1 * (torch.rand(1, N1) - 0.5)

    def init_block_table():
        # Random permutation of [0, NUM_PAGES) drives the page shuffle.
        return torch.randperm(NUM_PAGES).to(torch.int32)

    return [
        TensorSpec("x",           [BATCH, HIDDEN], torch.bfloat16, init_value=init_x),
        TensorSpec("w1",          [HIDDEN, N1],    torch.bfloat16, init_value=init_w1),
        TensorSpec("gamma",       [1, N1],         torch.float32,  init_value=init_gamma),
        TensorSpec("block_table", [NUM_PAGES],     torch.int32,    init_value=init_block_table),
        TensorSpec("out",         [BATCH, N1],     torch.float32,  is_output=True),
    ]


def golden_paged_consumer_block_table(tensors):
    import torch

    x_f32     = tensors["x"].float()
    w1_f32    = tensors["w1"].float()
    gamma_f32 = tensors["gamma"].float()                # [1, N1]
    btab      = tensors["block_table"]

    y_f32 = x_f32 @ w1_f32                              # [BATCH, N1]

    # Per-page block_table gather + RMSNorm.
    out_f32 = torch.zeros(BATCH, N1, dtype=torch.float32)
    for ob in range(NUM_PAGES):
        page_id = int(btab[ob].item())
        src     = y_f32[page_id * PAGE_M : (page_id + 1) * PAGE_M, :]   # [PAGE_M, N1]
        mean_sq = (src * src).mean(dim=-1, keepdim=True)                # [PAGE_M, 1]
        inv_rms = torch.rsqrt(mean_sq + EPS)
        out_f32[ob * PAGE_M : (ob + 1) * PAGE_M, :] = src * inv_rms * gamma_f32
    tensors["out"][:] = out_f32


if __name__ == "__main__":
    import argparse

    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=paged_consumer_block_table,
        specs=build_tensor_specs(),
        golden_fn=golden_paged_consumer_block_table,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=4e-3,
        atol=4e-3,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)

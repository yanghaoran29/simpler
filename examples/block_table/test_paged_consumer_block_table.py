#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
"""Paged consumer block_table — runtime test for the AIC+AIV mixed example.

Drives the pre-compiled orchestration in ``orchestration/`` and the two
incore kernels in ``kernels/{aic,aiv}/`` through the ``tensormap_and_ringbuffer``
runtime. Mirrors the DSL spec in ``paged_consumer_block_table_pypto_syntax.py``.

Pipeline:
  Stage 1 (AIC paged_proj):  y = x @ w1            -> paged_y (FP32, [256, 256])
  Stage 2 (AIV paged_rmsnorm): out[ob] = RMSNorm(paged_y[block_table[ob]]) * gamma

Orchestration entry takes 5 tensors: x, w1, gamma, block_table, out.

The per-page slices (``x_page``, ``paged_y[m0:m0+PAGE_M]``,
``paged_y[page_id*PAGE_M:...]`` and ``out[ob*PAGE_M:...]``) are taken in the
orchestrator via ``ext_*.view(...)`` / ``get_tensor_data<int32_t>(block_table,
...)``, so each AIC / AIV task only carries its own sub-tile as a data
dependency. ``block_table`` itself is read by the orchestrator and is NOT
passed into the AIV kernel; the AIV kernel only receives ``gamma`` and the
gathered ``y_src`` view (plus an ``out_m0`` scalar that the windowed pass
no longer uses for offset).
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test


NUM_PAGES = 16
PAGE_M = 16
BATCH = NUM_PAGES * PAGE_M           # 256
HIDDEN = 2048
N1 = 256
EPS = 1.0e-6


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestPagedConsumerBlockTable(SceneTestCase):
    """Block-table driven paged projection + paged RMSNorm."""

    RTOL = 4e-3
    ATOL = 4e-3

    CALLABLE = {
        "orchestration": {
            "source": "orchestration/paged_consumer_block_table.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.IN, D.IN, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "paged_proj",
                "source": "kernels/aic/paged_proj.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "name": "paged_rmsnorm",
                "source": "kernels/aiv/paged_rmsnorm.cpp",
                "core_type": "aiv",
                # Orchestrator submits (gamma, y_src, out_view) + scalar
                # out_m0 — block_table is consumed in orchestration, not
                # by the AIV kernel.
                "signature": [D.IN, D.IN, D.OUT],
            },
        ],
    }

    CASES = [
        {
            "name": "Case1",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 24},
            "params": {"dtype": "bfloat16"},
        },
    ]

    def generate_args(self, params):
        scale_h = HIDDEN ** 0.5

        x = (torch.rand(BATCH, HIDDEN, dtype=torch.float32) - 0.5).to(torch.bfloat16)
        w1 = ((torch.rand(HIDDEN, N1, dtype=torch.float32) - 0.5) / scale_h).to(torch.bfloat16)
        gamma = 1.0 + 0.1 * (torch.rand(1, N1, dtype=torch.float32) - 0.5)
        block_table = torch.randperm(NUM_PAGES).to(torch.int32)
        out = torch.zeros(BATCH, N1, dtype=torch.float32)

        return TaskArgsBuilder(
            Tensor("x", x),
            Tensor("w1", w1),
            Tensor("gamma", gamma),
            Tensor("block_table", block_table),
            Tensor("out", out),
        )

    def compute_golden(self, args, params):
        x_f32 = args.x.float()
        w1_f32 = args.w1.float()
        gamma_f32 = args.gamma.float()
        btab = args.block_table

        # Stage 1: contiguous FP32 projection — matches the FP32 paged_y the
        # AIC kernel assembles.
        y_f32 = x_f32 @ w1_f32  # [BATCH, N1]

        # Stage 2: per-output-page block_table gather, then RMSNorm.
        out_f32 = torch.zeros(BATCH, N1, dtype=torch.float32)
        for ob in range(NUM_PAGES):
            page_id = int(btab[ob].item())
            src = y_f32[page_id * PAGE_M : (page_id + 1) * PAGE_M, :]
            mean_sq = (src * src).mean(dim=-1, keepdim=True)
            inv_rms = torch.rsqrt(mean_sq + EPS)
            out_f32[ob * PAGE_M : (ob + 1) * PAGE_M, :] = src * inv_rms * gamma_f32

        args.out[:] = out_f32


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)

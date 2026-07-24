#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A flagged producer feeds a MIX sync_start early-dispatch consumer (a5).

Sized from available_block_dim (=N). Cases EarlyOn / EarlyOff toggle producer
``allow_early_resolve`` via orch scalar so swimlanes can be compared.
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, Tensor, available_block_dim, scene_test

FLOATS_PER_CACHE_LINE = 16
SLOTS_PER_BLOCK = 3


def _layout(platform: str):
    n = available_block_dim(platform)
    return n, n, n + n * SLOTS_PER_BLOCK


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestSpmdSyncStartEarlyDispatch(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/spmd_sync_start_early_dispatch_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "SPMD_WRITE_AIC",
                "source": "kernels/aiv/kernel_spmd_write_slow.cpp",
                "core_type": "aic",
                "signature": [D.INOUT],
            },
            {
                "func_id": 1,
                "name": "SPMD_MIX_AIC",
                "source": "../spmd_multiblock_mix/kernels/aic/kernel_spmd_mix.cpp",
                "core_type": "aic",
                "signature": [D.INOUT],
            },
            {
                "func_id": 2,
                "name": "SPMD_MIX_AIV0",
                "source": "../spmd_multiblock_mix/kernels/aiv/kernel_spmd_mix.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
            {
                "func_id": 3,
                "name": "SPMD_MIX_AIV1",
                "source": "../spmd_multiblock_mix/kernels/aiv/kernel_spmd_mix.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
        ],
    }

    CASES = [
        {
            "name": "EarlyOn",
            "platforms": ["a5sim", "a5"],
            "config": {},
            "params": {"early_on": 1},
        },
        {
            "name": "EarlyOff",
            "platforms": ["a5sim", "a5"],
            "config": {},
            "params": {"early_on": 0},
        },
    ]

    def generate_args(self, params):
        platform = getattr(self, "_st_platform", "a5")
        _, _, total_cl = _layout(platform)
        return TaskArgsBuilder(
            Tensor("output", torch.zeros(total_cl * FLOATS_PER_CACHE_LINE, dtype=torch.float32)),
            Scalar("early_on", int(params.get("early_on", 1))),
        )

    def compute_golden(self, args, params):
        platform = getattr(self, "_st_platform", "a5")
        producer_blocks, sync_blocks, _ = _layout(platform)
        out = args.output
        for block_idx in range(producer_blocks):
            out[block_idx * FLOATS_PER_CACHE_LINE] = float(block_idx)
        for block_idx in range(sync_blocks):
            for slot in range(SLOTS_PER_BLOCK):
                out[(producer_blocks + block_idx * SLOTS_PER_BLOCK + slot) * FLOATS_PER_CACHE_LINE] = float(block_idx)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)

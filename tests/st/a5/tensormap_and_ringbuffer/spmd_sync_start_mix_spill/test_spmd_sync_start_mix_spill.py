#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""sync_start MIX per-core pending-spill on a5 (36 AIC + 72 AIV).

A flagged AIV producer occupies all 72 AIV cores and spins; the require_sync_start
MIX consumer (24 clusters) pre-stages with AIC on idle running slots and AIVs on
busy pending slots. EarlyOn / EarlyOff toggle producer ``allow_early_resolve``.
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, Tensor, scene_test

FLOATS_PER_CACHE_LINE = 16
SLOTS_PER_BLOCK = 3
PRODUCER_BLOCKS = 72  # fill all a5 AIV cores
PRODUCER_BASE_CL = 0
CONSUMER_BLOCKS = 24
CONSUMER_BASE_CL = PRODUCER_BLOCKS
TOTAL_CL = PRODUCER_BLOCKS + CONSUMER_BLOCKS * SLOTS_PER_BLOCK  # 144


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestSpmdSyncStartMixSpill(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/spmd_sync_start_mix_spill_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "SPMD_MIX_AIC",
                "source": "kernels/aic/kernel_spmd_mix_slow.cpp",
                "core_type": "aic",
                "signature": [D.INOUT],
            },
            {
                "func_id": 1,
                "name": "SPMD_MIX_AIV0",
                "source": "kernels/aiv/kernel_spmd_mix_slow.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
            {
                "func_id": 2,
                "name": "SPMD_MIX_AIV1",
                "source": "kernels/aiv/kernel_spmd_mix_slow.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
            {
                "func_id": 3,
                "name": "SPMD_WRITE_AIV",
                "source": "kernels/aiv/kernel_spmd_write_slow.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
        ],
    }

    CASES = [
        {
            "name": "EarlyOn",
            "platforms": ["a5sim", "a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 24},
            "params": {"early_on": 1},
        },
        {
            "name": "EarlyOff",
            "platforms": ["a5sim", "a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 24},
            "params": {"early_on": 0},
        },
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            Tensor("output", torch.zeros(TOTAL_CL * FLOATS_PER_CACHE_LINE, dtype=torch.float32)),
            Scalar("early_on", int(params.get("early_on", 1))),
        )

    def compute_golden(self, args, params):
        out = args.output
        for block_idx in range(PRODUCER_BLOCKS):
            out[(PRODUCER_BASE_CL + block_idx) * FLOATS_PER_CACHE_LINE] = float(block_idx)
        for block_idx in range(CONSUMER_BLOCKS):
            for slot in range(SLOTS_PER_BLOCK):
                out[(CONSUMER_BASE_CL + block_idx * SLOTS_PER_BLOCK + slot) * FLOATS_PER_CACHE_LINE] = float(block_idx)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)

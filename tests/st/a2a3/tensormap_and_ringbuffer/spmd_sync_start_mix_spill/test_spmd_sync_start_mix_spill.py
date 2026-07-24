#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""sync_start MIX per-core pending-spill (sized from available_block_dim).

A flagged AIV producer occupies all AIV cores and spins; the require_sync_start
MIX consumer pre-stages with AIC on idle running slots and AIVs on busy pending
slots.
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, available_block_dim, scene_test

FLOATS_PER_CACHE_LINE = 16
SLOTS_PER_BLOCK = 3


def _layout(platform: str):
    n_cluster = available_block_dim(platform)
    n_aiv = n_cluster * 2
    return n_cluster, n_aiv, n_aiv + n_cluster * SLOTS_PER_BLOCK


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
            "name": "Case1",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {},
            "params": {},
        }
    ]

    def generate_args(self, params):
        platform = getattr(self, "_st_platform", "a2a3")
        _, _, total_cl = _layout(platform)
        return TaskArgsBuilder(Tensor("output", torch.zeros(total_cl * FLOATS_PER_CACHE_LINE, dtype=torch.float32)))

    def compute_golden(self, args, params):
        platform = getattr(self, "_st_platform", "a2a3")
        n_cluster, n_aiv, _ = _layout(platform)
        out = args.output
        for block_idx in range(n_aiv):
            out[block_idx * FLOATS_PER_CACHE_LINE] = float(block_idx)
        for block_idx in range(n_cluster):
            for slot in range(SLOTS_PER_BLOCK):
                out[(n_aiv + block_idx * SLOTS_PER_BLOCK + slot) * FLOATS_PER_CACHE_LINE] = float(block_idx)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)

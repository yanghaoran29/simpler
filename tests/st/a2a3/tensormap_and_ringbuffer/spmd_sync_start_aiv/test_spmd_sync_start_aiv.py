#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""SPMD sync_start AIV: 4 AIV tasks testing fast path and drain (T3 uses N clusters)."""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, available_block_dim, scene_test

FLOATS_PER_CACHE_LINE = 16


def _tasks(platform: str):
    n = available_block_dim(platform)
    bns = [4, 16, 4, n]
    tasks = []
    base = 0
    for bn in bns:
        tasks.append((bn, base))
        base += bn
    return tasks


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestSpmdSyncStartAiv(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/spmd_sync_start_aiv_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "SPMD_WRITE_AIV",
                "source": '../spmd_multiblock_aiv/kernels/aiv/kernel_spmd_write.cpp',
                "core_type": "aiv",
            },
        ],
    }

    CASES = [
        {
            "name": "Case1",
            "platforms": ['a2a3sim', 'a2a3'],
            "config": {},
            "params": {},
        }
    ]

    def generate_args(self, params):
        platform = getattr(self, "_st_platform", 'a2a3')
        tasks = _tasks(platform)
        total_cl = sum(bn for bn, _ in tasks)
        return TaskArgsBuilder(Tensor("output", torch.zeros(total_cl * FLOATS_PER_CACHE_LINE, dtype=torch.float32)))

    def compute_golden(self, args, params):
        platform = getattr(self, "_st_platform", 'a2a3')
        out = args.output
        for block_num, base_cl in _tasks(platform):
            for block_idx in range(block_num):
                out[(base_cl + block_idx) * FLOATS_PER_CACHE_LINE] = float(block_idx)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)

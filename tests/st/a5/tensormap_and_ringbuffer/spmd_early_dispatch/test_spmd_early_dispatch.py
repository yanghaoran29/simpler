#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Plain allow_early_resolve ST (a5) — NO require_sync_start.

Producer leaves spare AIC cores so the consumer can stage via
``early_dispatch_queues`` while the producer is still spinning. EarlyOn /
EarlyOff toggle the producer flag for swimlane comparison.
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, Tensor, scene_test

FLOATS_PER_CACHE_LINE = 16
PRODUCER_BLOCKS = 8
CONSUMER_BLOCKS = 8
CONSUMER_BASE_CL = PRODUCER_BLOCKS
TOTAL_CL = CONSUMER_BASE_CL + CONSUMER_BLOCKS


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestSpmdEarlyDispatch(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/spmd_early_dispatch_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "SPMD_WRITE_AIC",
                "source": "../spmd_sync_start_early_dispatch/kernels/aiv/kernel_spmd_write_slow.cpp",
                "core_type": "aic",
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
        # orch stamps early_on into out[1] as a host-side probe (kernels leave it alone).
        out[1] = float(int(params.get("early_on", 1)))
        for block_idx in range(PRODUCER_BLOCKS):
            out[block_idx * FLOATS_PER_CACHE_LINE] = float(block_idx)
        for block_idx in range(CONSUMER_BLOCKS):
            out[(CONSUMER_BASE_CL + block_idx) * FLOATS_PER_CACHE_LINE] = float(block_idx)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)

#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CONTROL: full Acc C2V (Valid≡Rows) — should PASS with or without the ISA fix.

Same matmul shapes as the strip repro; only the Acc→Vec transfer differs
(one full TPUSH vs ValidShape strip windows).
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test

M, N, K = 128, 256, 32


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestAccC2VFullControl(SceneTestCase):
    RTOL = 5e-2
    ATOL = 0.5

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/acc_c2v_strip_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "ACC_C2V_FULL_AIC",
                "source": "kernels/mix/kernel_acc_c2v_full.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "name": "ACC_C2V_FULL_AIV",
                "source": "kernels/mix/kernel_acc_c2v_full.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
        ],
    }

    CASES = [
        {
            "name": "default",
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 1},
            "params": {},
        }
    ]

    def generate_args(self, params):
        torch.manual_seed(0)
        A = torch.randn(M, K, dtype=torch.float32) * 1.0
        B = torch.randn(K, N, dtype=torch.float32) * 1.0
        C = torch.zeros(M, N, dtype=torch.float32)
        return TaskArgsBuilder(
            Tensor("A", A.reshape(-1)),
            Tensor("B", B.reshape(-1)),
            Tensor("C", C.reshape(-1)),
        )

    def compute_golden(self, args, params):
        A = args.A.reshape(M, K)
        B = args.B.reshape(K, N)
        args.C[:] = torch.matmul(A, B).reshape(-1)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)

#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A5 Acc ValidShape strip C2V repro (pto-isa TMovCcToUb srcStride).

Shapes: Acc[M=128,N=256] drained as 8× ValidShape(H=16,N) TPUSH with
addr = row * 64. Golden is float32 C = A @ B (K=64).

Expect:
  - unfixed pto-isa (srcStride=align(validRow)): FAIL (~90%+ mistmatch after strip0)
  - fixed pto-isa   (srcStride=align(Rows)):     PASS

Run (from simpler repo root, with installed simpler package)::

  # Fixed ISA (e.g. pypto/build/pto-isa @ 0ebbd03d):
  PTO_ISA_ROOT=/path/to/fixed/pto-isa \\
    python examples/a5/tensormap_and_ringbuffer/acc_c2v_strip_validshape/test_acc_c2v_strip_validshape.py -p a5 -d 0

  # Unfixed ISA (checkout before srcStride fix):
  PTO_ISA_ROOT=/path/to/unfixed/pto-isa \\
    python .../test_acc_c2v_strip_validshape.py -p a5 -d 0
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test

M, N, K = 128, 256, 32
H = 16


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestAccC2VStripValidShape(SceneTestCase):
    """Strip Acc C2V vs matmul golden — fails before ISA srcStride=Rows fix."""

    # Match issue / board thresholds used for Acc C2V strip diagnosis.
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
                "name": "ACC_C2V_STRIP_AIC",
                "source": "kernels/mix/kernel_acc_c2v_strip.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "name": "ACC_C2V_STRIP_AIV",
                "source": "kernels/mix/kernel_acc_c2v_strip.cpp",
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
        # Scale down so Acc values stay moderate for atol/rtol.
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

#!/usr/bin/env python3
"""CONTROL: AIC-only Acc matmul + TSTORE (no C2V)."""
import torch
from simpler.task_interface import ArgDirection as D
from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test

M, N, K = 128, 256, 32


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestAccMatmulTstore(SceneTestCase):
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
                "name": "MATMUL_TSTORE",
                "source": "kernels/aic/kernel_matmul_tstore.cpp",
                "core_type": "aic",
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
        return TaskArgsBuilder(Tensor("A", A.reshape(-1)), Tensor("B", B.reshape(-1)), Tensor("C", C.reshape(-1)))

    def compute_golden(self, args, params):
        args.C[:] = torch.matmul(args.A.reshape(M, K), args.B.reshape(K, N)).reshape(-1)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)

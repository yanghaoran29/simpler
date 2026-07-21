#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
"""A5 Acc ValidShape strip C2V repro — device C_strip vs device C_full.

Drains the same Acc twice (full Valid≡Rows vs ValidShape strip windows).
After the run we require C_strip ≈ C_full.

  unfixed ISA (srcStride=validRow): FAIL (often strip0 OK, strip1+ bad)
  fixed   ISA (srcStride=Rows):     PASS
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test
from simpler_setup.scene_test import _build_chip_task_args, _temporary_env

M, N, K = 32, 128, 32
H = 16


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestAccC2VStripVsFull(SceneTestCase):
    RTOL = 1e-4
    ATOL = 1e-4

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/acc_c2v_compare_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.OUT, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "ACC_C2V_COMPARE_AIC",
                "source": "kernels/mix/kernel_acc_c2v_compare.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.OUT, D.OUT],
            },
            {
                "func_id": 1,
                "name": "ACC_C2V_COMPARE_AIV",
                "source": "kernels/mix/kernel_acc_c2v_compare.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT, D.OUT],
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
        A = torch.randn(M, K, dtype=torch.float32)
        B = torch.randn(K, N, dtype=torch.float32)
        return TaskArgsBuilder(
            Tensor("A", A.reshape(-1)),
            Tensor("B", B.reshape(-1)),
            Tensor("C_full", torch.zeros(M * N, dtype=torch.float32)),
            Tensor("C_strip", torch.zeros(M * N, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        pass

    def _run_and_validate_l2(  # noqa: PLR0913
        self,
        worker,
        callable_obj,
        case,
        rounds=1,
        skip_golden=False,
        enable_l2_swimlane=0,
        enable_dump_args=False,
        enable_pmu=0,
        enable_dep_gen=False,
        enable_scope_stats=False,
        output_prefix="",
    ):
        del rounds, skip_golden  # single-shot self-compare
        params = case.get("params", {})
        config_dict = case.get("config", {})
        orch_sig = self.CALLABLE.get("orchestration", {}).get("signature", [])
        handle = getattr(type(self), "_st_l2_handle", None)
        if handle is None:
            handle = worker.register(callable_obj)
            type(self)._st_l2_handle = handle

        test_args = self.generate_args(params)
        chip_args, _ = _build_chip_task_args(test_args, orch_sig)
        config = self._build_config(
            config_dict,
            enable_l2_swimlane=enable_l2_swimlane,
            enable_dump_args=enable_dump_args,
            enable_pmu=enable_pmu,
            enable_dep_gen=enable_dep_gen,
            enable_scope_stats=enable_scope_stats,
            output_prefix=output_prefix,
        )
        with _temporary_env(self._resolve_env()):
            worker.run(handle, chip_args, config=config)

        full = test_args.C_full.reshape(M, N)
        strip = test_args.C_strip.reshape(M, N)
        if not torch.allclose(strip, full, rtol=self.RTOL, atol=self.ATOL):
            diff = (strip - full).abs().max().item()
            s0 = (strip[:H] - full[:H]).abs().max().item()
            s1 = (strip[H : 2 * H] - full[H : 2 * H]).abs().max().item()
            raise AssertionError(
                f"C_strip vs C_full mismatch: max_diff={diff}, "
                f"strip0_max={s0}, strip1_max={s1}, rtol={self.RTOL}, atol={self.ATOL}"
            )


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)

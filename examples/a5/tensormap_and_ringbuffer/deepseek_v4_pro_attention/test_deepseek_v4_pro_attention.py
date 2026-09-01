#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 Pro attention benchmark over harvested PyPTO codegen."""

import ctypes
import importlib.util
import json
from functools import cache
from pathlib import Path

import torch

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

_ARTIFACT_ROOT = Path(__file__).parent / "kernels" / "vendor"
_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
}
_SCALARS = {
    "float32": ctypes.c_float,
    "int32": ctypes.c_int32,
    "int64": ctypes.c_int64,
}


@cache
def _fixture_meta(artifact: str) -> dict:
    with (_ARTIFACT_ROOT / artifact / "fixture_meta.json").open(encoding="utf-8") as file:
        return json.load(file)


def _load_callable(artifact: str) -> dict:
    config_path = _ARTIFACT_ROOT / artifact / "kernel_config.py"
    spec = importlib.util.spec_from_file_location(f"_dsv4_pro_{artifact}_kernel_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load harvested kernel config: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    callable_spec = {"orchestration": module.ORCHESTRATION, "incores": module.KERNELS}
    tensor_params = [param for param in _fixture_meta(artifact)["params"] if param["kind"] == "tensor"]
    signature = callable_spec["orchestration"]["signature"]
    if len(tensor_params) != len(signature):
        raise RuntimeError(
            f"{artifact}: fixture has {len(tensor_params)} tensors, orchestration expects {len(signature)}"
        )
    return callable_spec


def _initialize_tensor(name: str, tensor: torch.Tensor, start_pos: int) -> None:
    """Fill only data that controls bounds or avoids degenerate scale math."""
    if name == "position_ids":
        tensor.copy_(torch.arange(start_pos, start_pos + tensor.numel(), dtype=tensor.dtype).reshape(tensor.shape))
    elif name.endswith("slot_mapping"):
        tensor.copy_(torch.arange(tensor.numel(), dtype=tensor.dtype).reshape(tensor.shape))
    elif name.endswith("_lens"):
        tensor.fill_(128)
    elif name == "kv_seq_lens":
        tensor.fill_(start_pos + 2)
    elif name == "freqs_cos":
        tensor.fill_(1)
    elif name.endswith("_scale") or "norm_w" in name or name.startswith("gamma_"):
        tensor.fill_(1)


class _DeepSeekV4ProAttentionBase(SceneTestCase):
    """Common synthetic fixture for orchestration/scheduler benchmarking."""

    SKIP_GOLDEN = True
    ARTIFACT = ""

    def generate_args(self, params):
        meta = _fixture_meta(self.ARTIFACT)
        specs = []
        for param in meta["params"]:
            dtype_name = param["dtype"]
            if param["kind"] == "scalar":
                specs.append(Scalar(param["name"], _SCALARS[dtype_name](param["value"])))
                continue

            tensor = torch.zeros(param["shape"], dtype=_DTYPES[dtype_name])
            _initialize_tensor(param["name"], tensor, meta["start_pos"])
            specs.append(TensorArg(param["name"], tensor))
        return TaskArgsBuilder(*specs)

    def compute_golden(self, args, params):
        raise AssertionError(
            "This benchmark is intentionally execution-only; use the pypto-lib source for golden checks"
        )


# Match pypto-lib prefill_attention_* ring heap (4 GiB/ring) and image defaults:
# decode start_pos=8192, prefill start_pos=0.
_RUNTIME_ENV = {
    "ring_task_window": 16384,
    "ring_heap": 4 << 30,
    "ring_dep_pool": 16384,
}


def _benchmark_case(name: str) -> list[dict]:
    return [
        {
            "name": name,
            "platforms": ["a5"],
            "manual": True,
            "config": {"runtime_env": dict(_RUNTIME_ENV)},
        }
    ]


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestDecodeSWA(_DeepSeekV4ProAttentionBase):
    ARTIFACT = "decode_swa"
    CALLABLE = _load_callable(ARTIFACT)
    CASES = _benchmark_case("DecodeSWA")


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestDecodeCSA(_DeepSeekV4ProAttentionBase):
    ARTIFACT = "decode_csa"
    CALLABLE = _load_callable(ARTIFACT)
    CASES = _benchmark_case("DecodeCSA")


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestDecodeHCA(_DeepSeekV4ProAttentionBase):
    ARTIFACT = "decode_hca"
    CALLABLE = _load_callable(ARTIFACT)
    CASES = _benchmark_case("DecodeHCA")


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestPrefillSWA(_DeepSeekV4ProAttentionBase):
    ARTIFACT = "prefill_swa"
    CALLABLE = _load_callable(ARTIFACT)
    CASES = _benchmark_case("PrefillSWA")


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestPrefillCSA(_DeepSeekV4ProAttentionBase):
    ARTIFACT = "prefill_csa"
    CALLABLE = _load_callable(ARTIFACT)
    CASES = _benchmark_case("PrefillCSA")


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestPrefillHCA(_DeepSeekV4ProAttentionBase):
    ARTIFACT = "prefill_hca"
    CALLABLE = _load_callable(ARTIFACT)
    CASES = _benchmark_case("PrefillHCA")


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)

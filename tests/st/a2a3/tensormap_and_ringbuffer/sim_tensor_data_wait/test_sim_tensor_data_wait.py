#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""a2a3sim tensor-data wait budget vs onboard (issue #2278).

One finite AIV producer sleeps, then writes 7; orchestration reads via
get_tensor_data and copies to a host OUT tensor. Exercises the same
wait_for_tensor_ready path that falsely reaped slow sim producers under the
shared 15 s limit.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from simpler.task_interface import (
    ArgDirection,
    CallConfig,
    ChipCallable,
    CoreCallable,
    DataType,
    TaskArgs,
    TensorArgType,
)
from simpler.worker import Worker

from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.log_config import configure_logging
from simpler_setup.pto_isa import ensure_pto_isa_root

RUNTIME = "tensormap_and_ringbuffer"

ORCH = r"""
#include <cstdint>
#include "orchestration_api.h"
extern "C" __attribute__((visibility("default")))
OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 1};
}
extern "C" __attribute__((visibility("default")))
void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    uint32_t shape[1] = {1}, index[1] = {0};
    CoreTaskArgs task;
    TensorCreateInfo info(shape, 1, DataType::INT32);
    task.add_output(info);
    auto outputs = rt_submit_aiv_task(0, task);
    int32_t value = get_tensor_data<int32_t>(outputs.get_ref(0), 1, index);
    if (rt_is_fatal()) return;
    set_tensor_data(args.tensor(0).ref(), 1, index, value);
}
"""

CORE = r"""
#include <cstdint>
#include <chrono>
#include <thread>
#include <pto/pto-inst.hpp>
#include "tensor.h"
extern "C" void kernel_entry(int64_t *args) {
    auto *tensor = reinterpret_cast<Tensor *>(args[0]);
    std::this_thread::sleep_for(std::chrono::milliseconds(DELAY_MS));
    auto *out = reinterpret_cast<int32_t *>(tensor->buffer.addr);
    out[tensor->start_offset] = 7;
}
"""


def _build_chip(platform: str, delay_ms: int) -> ChipCallable:
    compiler = KernelCompiler(platform=platform)
    with tempfile.TemporaryDirectory(prefix="sim_tensor_wait_") as tmp:
        root = Path(tmp)
        (root / "orch.cpp").write_text(ORCH)
        (root / "core.cpp").write_text(CORE.replace("DELAY_MS", str(delay_ms)))
        core = compiler.compile_incore(
            str(root / "core.cpp"),
            core_type="aiv",
            pto_isa_root=ensure_pto_isa_root(),
            extra_include_dirs=compiler.get_orchestration_include_dirs(RUNTIME),
        )
        orch = compiler.compile_orchestration(RUNTIME, str(root / "orch.cpp"))
    return ChipCallable.build(
        signature=[ArgDirection.OUT],
        func_name="aicpu_orchestration_entry",
        binary=orch,
        children=[(0, CoreCallable.build(signature=[ArgDirection.OUT], binary=core))],
    )


def _run_scalar_read(platform: str, delay_ms: int, device_id: int = 0) -> int:
    chip = _build_chip(platform, delay_ms)
    worker = Worker(level=2, platform=platform, runtime=RUNTIME, device_id=device_id)
    handle = worker.register(chip)
    worker.init()
    try:
        output = torch.zeros(1, dtype=torch.int32)
        buffer = worker.malloc(4)
        worker.copy_to(buffer, output)
        call_args = TaskArgs()
        call_args.add_tensor(
            buffer.tensor(shapes=(1,), dtype=DataType.INT32),
            TensorArgType.OUTPUT_EXISTING,
        )
        config = CallConfig()
        config.aicpu_thread_num = 2
        worker.run(handle, call_args, config)
        worker.copy_from(output, buffer)
        worker.free(buffer)
        return int(output.item())
    finally:
        worker.close()


@pytest.mark.platforms(["a2a3sim"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(RUNTIME)
def test_sim_tensor_wait_short_producer(st_platform, st_device_ids):
    """2 s producer finishes under both the old 15 s and new 120 s sim budgets."""
    configure_logging("error")
    assert _run_scalar_read(st_platform, 2000, int(st_device_ids[0])) == 7


@pytest.mark.platforms(["a2a3sim"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(RUNTIME)
def test_sim_tensor_wait_slow_finite_producer(st_platform, st_device_ids):
    """18 s exceeds the old shared 15 s limit but is within the 120 s sim budget."""
    configure_logging("error")
    assert _run_scalar_read(st_platform, 18000, int(st_device_ids[0])) == 7


@pytest.mark.platforms(["a2a3sim"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(RUNTIME)
@pytest.mark.manual(["a2a3sim"])
def test_sim_tensor_wait_over_budget_times_out(st_platform, st_device_ids, monkeypatch):
    """Producer longer than the 120 s sim budget still latches TENSOR_WAIT_TIMEOUT (8).

    Raise the scheduler no-progress watchdog so the tensor-data wait wins.
    Manual: wall time is ~120 s plus cleanup.
    """
    configure_logging("error")
    monkeypatch.setenv("SIMPLER_SCHEDULER_TIMEOUT_MS", "180000")
    chip = _build_chip(st_platform, 130000)
    worker = Worker(level=2, platform=st_platform, runtime=RUNTIME, device_id=int(st_device_ids[0]))
    handle = worker.register(chip)
    worker.init()
    try:
        output = torch.zeros(1, dtype=torch.int32)
        buffer = worker.malloc(4)
        worker.copy_to(buffer, output)
        call_args = TaskArgs()
        call_args.add_tensor(
            buffer.tensor(shapes=(1,), dtype=DataType.INT32),
            TensorArgType.OUTPUT_EXISTING,
        )
        config = CallConfig()
        config.aicpu_thread_num = 2
        with pytest.raises(RuntimeError, match=r"(run_runtime|run) failed with code -8\b"):
            worker.run(handle, call_args, config)
        worker.free(buffer)
    finally:
        worker.close()

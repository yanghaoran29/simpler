#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Onboard microbenchmark: pageable vs aclrtHostRegisterV2 pinned host memory for 1 MiB sync H2D.

See docs/investigations/2026-08-hostregister-v2-1m-h2d-microbench.md for setup and recorded results.
"""

from __future__ import annotations

import ctypes
import json
import math
import mmap
import os
import statistics
import time

SIZE = 1 << 20
TRIALS = 5
SAMPLES = 2000
WARMUPS = 100
HOST_TO_DEVICE = 1
MALLOC_HUGE_FIRST = 0
HOST_REGISTER_PINNED = 0x10000000


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def summarize(values: list[float]) -> dict[str, float | int]:
    mean = statistics.mean(values)
    return {
        "n": len(values),
        "mean_us": mean / 1000,
        "p50_us": statistics.median(values) / 1000,
        "p95_us": percentile(values, 0.95) / 1000,
        "p99_us": percentile(values, 0.99) / 1000,
        "std_us": statistics.stdev(values) / 1000,
        "cv_pct": statistics.stdev(values) / mean * 100,
        "min_us": min(values) / 1000,
        "max_us": max(values) / 1000,
    }


def checked(name: str, rc: int) -> None:
    if rc != 0:
        raise RuntimeError(f"{name} failed: {rc}")


def main() -> None:
    acl = ctypes.CDLL("libascendcl.so")
    acl.aclInit.argtypes = [ctypes.c_char_p]
    acl.aclInit.restype = ctypes.c_int
    acl.aclFinalize.argtypes = []
    acl.aclFinalize.restype = ctypes.c_int
    acl.aclrtSetDevice.argtypes = [ctypes.c_int]
    acl.aclrtSetDevice.restype = ctypes.c_int
    acl.aclrtResetDevice.argtypes = [ctypes.c_int]
    acl.aclrtResetDevice.restype = ctypes.c_int
    acl.aclrtMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int]
    acl.aclrtMalloc.restype = ctypes.c_int
    acl.aclrtFree.argtypes = [ctypes.c_void_p]
    acl.aclrtFree.restype = ctypes.c_int
    acl.aclrtMemcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    acl.aclrtMemcpy.restype = ctypes.c_int
    acl.aclrtHostRegisterV2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32]
    acl.aclrtHostRegisterV2.restype = ctypes.c_int
    acl.aclrtHostUnregister.argtypes = [ctypes.c_void_p]
    acl.aclrtHostUnregister.restype = ctypes.c_int

    device_text = os.environ.get("NPU_LOCKED_DEVICE", os.environ.get("TASK_DEVICE", "0"))
    device = int(device_text.split(",")[0])
    checked("aclInit", acl.aclInit(None))
    checked("aclrtSetDevice", acl.aclrtSetDevice(device))

    dev_ptr = ctypes.c_void_p()
    checked("aclrtMalloc", acl.aclrtMalloc(ctypes.byref(dev_ptr), SIZE, MALLOC_HUGE_FIRST))
    host = mmap.mmap(-1, SIZE, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    host_ptr = ctypes.addressof(ctypes.c_char.from_buffer(host))
    ctypes.memset(host_ptr, 0x5A, SIZE)

    all_samples: dict[str, list[float]] = {"pageable": [], "registered": []}
    trial_rows: list[dict] = []
    register_ns: list[float] = []
    unregister_ns: list[float] = []

    def copy_once() -> None:
        checked(
            "aclrtMemcpy",
            acl.aclrtMemcpy(dev_ptr, SIZE, ctypes.c_void_p(host_ptr), SIZE, HOST_TO_DEVICE),
        )

    def measure() -> list[float]:
        for _ in range(WARMUPS):
            copy_once()
        values: list[float] = []
        for _ in range(SAMPLES):
            begin = time.perf_counter_ns()
            copy_once()
            values.append(time.perf_counter_ns() - begin)
        return values

    def register_host() -> None:
        begin = time.perf_counter_ns()
        checked(
            "aclrtHostRegisterV2",
            acl.aclrtHostRegisterV2(ctypes.c_void_p(host_ptr), SIZE, HOST_REGISTER_PINNED),
        )
        register_ns.append(time.perf_counter_ns() - begin)

    def unregister_host() -> None:
        begin = time.perf_counter_ns()
        checked("aclrtHostUnregister", acl.aclrtHostUnregister(ctypes.c_void_p(host_ptr)))
        unregister_ns.append(time.perf_counter_ns() - begin)

    try:
        for trial in range(TRIALS):
            results: dict[str, list[float]] = {}
            if trial % 2 == 0:
                results["pageable"] = measure()
                register_host()
                results["registered"] = measure()
                unregister_host()
            else:
                register_host()
                results["registered"] = measure()
                unregister_host()
                results["pageable"] = measure()

            row: dict = {"trial": trial + 1}
            for mode in ("pageable", "registered"):
                all_samples[mode].extend(results[mode])
                row[mode] = summarize(results[mode])
            trial_rows.append(row)

        pageable = summarize(all_samples["pageable"])
        registered = summarize(all_samples["registered"])
        reg = summarize(register_ns)
        unreg = summarize(unregister_ns)
        saved_ns = pageable["mean_us"] * 1000 - registered["mean_us"] * 1000
        break_even = (
            None
            if saved_ns <= 0
            else (statistics.mean(register_ns) + statistics.mean(unregister_ns)) / saved_ns
        )
        output = {
            "device": device,
            "size_bytes": SIZE,
            "trials": TRIALS,
            "samples_per_trial": SAMPLES,
            "trial_rows": trial_rows,
            "pageable": pageable,
            "registered": registered,
            "register": reg,
            "unregister": unreg,
            "mean_latency_change_pct": (registered["mean_us"] / pageable["mean_us"] - 1) * 100,
            "p50_latency_change_pct": (registered["p50_us"] / pageable["p50_us"] - 1) * 100,
            "break_even_copies": break_even,
        }
        print(json.dumps(output, sort_keys=True))
    finally:
        host.close()
        checked("aclrtFree", acl.aclrtFree(dev_ptr))
        checked("aclrtResetDevice", acl.aclrtResetDevice(device))
        checked("aclFinalize", acl.aclFinalize())


if __name__ == "__main__":
    main()

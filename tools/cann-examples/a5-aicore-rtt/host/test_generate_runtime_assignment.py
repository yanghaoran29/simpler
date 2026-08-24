#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for the offline RTT-to-runtime assignment converter."""

import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("generate_runtime_assignment.py")
SPEC = importlib.util.spec_from_file_location("generate_runtime_assignment", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load module from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_raw(partitions, ambiguous_boundary=False):
    """Build a minimal five-round raw result with controlled die affinity."""
    records = []
    scheduler_cpus = [1, 2, 3, 4]
    for round_index, die0_cpus in enumerate(partitions):
        for scheduler, cpu in enumerate(scheduler_cpus):
            if ambiguous_boundary and cpu in (2, 3):
                die0_ticks, die1_ticks = (150, 151) if cpu in die0_cpus else (151, 150)
            else:
                die0_ticks, die1_ticks = (100, 200) if cpu in die0_cpus else (200, 100)
            for die, ticks in enumerate((die0_ticks, die1_ticks)):
                records.append(
                    {
                        "round": round_index,
                        "scheduler_index": scheduler,
                        "die": die,
                        "samples_ticks": [ticks, ticks],
                    }
                )
    return {
        "soc_name": "Ascend950PR_test",
        "counter_frequency_hz": 1_000_000_000 if ambiguous_boundary else 1_000_000,
        "round_count": 5,
        "samples_requested": 2,
        "warmup_requested": 1,
        "allowed_cpus": scheduler_cpus + [5],
        "records": records,
    }


class AssignmentTest(unittest.TestCase):
    def test_accepts_four_of_five_matching_partitions(self):
        raw = make_raw([{1, 3}, {1, 3}, {1, 3}, {2, 4}, {1, 3}])
        order, metadata = MODULE.derive_assignment(raw)
        self.assertEqual(order, [1, 3, 2, 4])
        self.assertEqual(metadata["partition_agreement_rounds"], 4)

    def test_rejects_unstable_partitions(self):
        raw = make_raw([{1, 2}, {1, 3}, {1, 4}, {2, 3}, {2, 4}])
        with self.assertRaisesRegex(ValueError, "unstable"):
            MODULE.derive_assignment(raw)

    def test_tie_breaks_an_ambiguous_middle_pair_by_cpu_id(self):
        raw = make_raw([{1, 3}, {1, 2}, {1, 3}, {1, 2}, {1, 3}], ambiguous_boundary=True)
        order, metadata = MODULE.derive_assignment(raw)
        self.assertEqual(order, [1, 2, 3, 4])
        self.assertEqual(metadata["deterministic_tie_break"]["boundary_scheduler_cpus"], [2, 3])


if __name__ == "__main__":
    unittest.main()

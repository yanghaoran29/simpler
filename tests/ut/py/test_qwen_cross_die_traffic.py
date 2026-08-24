# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from simpler_setup.tools.qwen_cross_die_traffic import calculate_mlp_traffic, data_aware_placement


def _synthetic_index():
    next_task = 1
    index = {}
    for layer in range(40):
        for role in ("gate", "up"):
            for shard in range(17):
                for split in range(5):
                    if shard < 6:
                        key = (role, layer, shard, split)
                        if key not in index:
                            index[key] = next_task
                            next_task += 1
                    else:
                        index[(role, layer, shard, split)] = next_task
                        next_task += 1
        for shard in range(17):
            index[("silu", layer, shard)] = next_task
            next_task += 1
        for output_group in range(5):
            for input_shard in range(17):
                index[("down", layer, output_group, input_shard)] = next_task
                next_task += 1
        dcr_task = next_task
        next_task += 1
        for output_group in range(5):
            index[("dcr", layer, output_group)] = dcr_task
    return index


def test_data_aware_policy_keeps_accumulator_contributions_local():
    index = _synthetic_index()
    report = calculate_mlp_traffic(index, data_aware_placement(index))

    assert report["categories"].get("gate_atomic_to_silu_home", 0) == 0
    assert report["categories"].get("up_atomic_to_silu_home", 0) == 0
    assert report["categories"].get("down_atomic_to_dcr_home", 0) == 0
    assert report["categories"]["silu_read_by_down"] == 43 * 40 * 16 * 1024 * 2
    assert report["directions"]["die0_to_die1"] == report["directions"]["die1_to_die0"]

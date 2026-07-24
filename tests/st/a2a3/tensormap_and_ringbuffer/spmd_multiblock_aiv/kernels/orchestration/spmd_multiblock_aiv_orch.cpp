/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

/**
 * SPMD Multi-Block AIV Orchestration
 *
 * Submits five AIV tasks with increasing block_num. Full-pool sizes use
 * rt_available_cluster_count() (=N) / AIV pool (=2N):
 *   T0: block_num=4   — fits within a single sched thread
 *   T1: block_num=16  — saturates one sched thread (typical 8 clusters x 2 AIV)
 *   T2: block_num=N   — forces cross-thread re-push via ready_queue
 *   T3: block_num=2N  — occupy all AIV cores across all sched threads
 *   T4: block_num=4N  — two full rounds of all AIV cores
 *
 * Args layout: [output]
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"

#define FUNC_SPMD_WRITE_AIV 0

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return PTO2OrchestrationConfig{
        .expected_arg_count = 1,
    };
}

static void submit_spmd_aiv(int32_t kernel_id, const Tensor &out, int16_t block_num, int64_t base_cl) {
    L0TaskArgs args;
    args.add_inout(out);
    args.add_scalar(base_cl);
    args.launch_spec.set_block_num(block_num);
    rt_submit_aiv_task(kernel_id, args);
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const L2TaskArgs &orch_args) {
    const Tensor &ext_output = orch_args.tensor(0).ref();
    const int32_t n = rt_available_cluster_count();
    const int16_t bn0 = 4;
    const int16_t bn1 = 16;
    const int16_t bn2 = static_cast<int16_t>(n);
    const int16_t bn3 = static_cast<int16_t>(2 * n);
    const int16_t bn4 = static_cast<int16_t>(4 * n);
    const int64_t base0 = 0;
    const int64_t base1 = base0 + bn0;
    const int64_t base2 = base1 + bn1;
    const int64_t base3 = base2 + bn2;
    const int64_t base4 = base3 + bn3;

    submit_spmd_aiv(FUNC_SPMD_WRITE_AIV, ext_output, bn0, base0);
    submit_spmd_aiv(FUNC_SPMD_WRITE_AIV, ext_output, bn1, base1);
    submit_spmd_aiv(FUNC_SPMD_WRITE_AIV, ext_output, bn2, base2);
    submit_spmd_aiv(FUNC_SPMD_WRITE_AIV, ext_output, bn3, base3);
    submit_spmd_aiv(FUNC_SPMD_WRITE_AIV, ext_output, bn4, base4);

    LOG_INFO_V9(
        "[spmd_multiblock_aiv] Submitted 5 AIV tasks: block_num=4,16,%d,%d,%d", n, 2 * n, 4 * n
    );
}

}  // extern "C"

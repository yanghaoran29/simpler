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
 * SPMD Multi-Block MIX Orchestration
 *
 * Submits five MIX tasks with increasing block_num. Full-pool sizes use
 * rt_available_cluster_count() (=N), not a hardcoded ceiling:
 *   T0: block_num=2   — basic multi-block MIX
 *   T1: block_num=8   — saturates one sched thread (typical 8 clusters)
 *   T2: block_num=12  — forces cross-thread re-push via ready_queue
 *   T3: block_num=N   — occupy all clusters across all sched threads
 *   T4: block_num=2N  — two full rounds of all clusters
 *
 * Args layout: [output]
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"

#define FUNC_SPMD_MIX_AIC 0
#define FUNC_SPMD_MIX_AIV0 1
#define FUNC_SPMD_MIX_AIV1 2

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return PTO2OrchestrationConfig{
        .expected_arg_count = 1,
    };
}

static void submit_spmd_mix(
    int32_t aic_id, int32_t aiv0_id, int32_t aiv1_id, const Tensor &out, int16_t block_num, int64_t base_cl
) {
    MixedKernels mk;
    mk.aic_kernel_id = aic_id;
    mk.aiv0_kernel_id = aiv0_id;
    mk.aiv1_kernel_id = aiv1_id;

    L0TaskArgs args;
    args.add_inout(out);
    args.add_scalar(base_cl);
    args.launch_spec.set_block_num(block_num);
    rt_submit_task(mk, args);
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const L2TaskArgs &orch_args) {
    const Tensor &ext_output = orch_args.tensor(0).ref();
    const int32_t n = rt_available_cluster_count();
    const int16_t bn0 = 2;
    const int16_t bn1 = 8;
    const int16_t bn2 = 12;
    const int16_t bn3 = static_cast<int16_t>(n);
    const int16_t bn4 = static_cast<int16_t>(2 * n);
    const int64_t base0 = 0;
    const int64_t base1 = base0 + bn0 * 3;
    const int64_t base2 = base1 + bn1 * 3;
    const int64_t base3 = base2 + bn2 * 3;
    const int64_t base4 = base3 + bn3 * 3;

    submit_spmd_mix(FUNC_SPMD_MIX_AIC, FUNC_SPMD_MIX_AIV0, FUNC_SPMD_MIX_AIV1, ext_output, bn0, base0);
    submit_spmd_mix(FUNC_SPMD_MIX_AIC, FUNC_SPMD_MIX_AIV0, FUNC_SPMD_MIX_AIV1, ext_output, bn1, base1);
    submit_spmd_mix(FUNC_SPMD_MIX_AIC, FUNC_SPMD_MIX_AIV0, FUNC_SPMD_MIX_AIV1, ext_output, bn2, base2);
    submit_spmd_mix(FUNC_SPMD_MIX_AIC, FUNC_SPMD_MIX_AIV0, FUNC_SPMD_MIX_AIV1, ext_output, bn3, base3);
    submit_spmd_mix(FUNC_SPMD_MIX_AIC, FUNC_SPMD_MIX_AIV0, FUNC_SPMD_MIX_AIV1, ext_output, bn4, base4);

    LOG_INFO_V9(
        "[spmd_multiblock_mix] Submitted 5 MIX tasks: block_num=2,8,12,%d,%d", n, 2 * n
    );
}

}  // extern "C"

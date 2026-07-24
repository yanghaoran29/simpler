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
 * SPMD Sync-Start AIV Orchestration
 *
 * Submits AIV-only tasks with require_sync_start=true to exercise:
 *   - AIV fast path: count_idle_aiv_cores() >= block_num (small block_num)
 *   - AIV drain path: block_num exceeds local AIV cores (cross-thread drain)
 *
 * Tasks (N = rt_available_cluster_count()):
 *   T0: block_num=4,  require_sync_start=true   (fast path)
 *   T1: block_num=16, require_sync_start=true   (saturate one thread: 8 clusters x 2 AIV)
 *   T2: block_num=4,  require_sync_start=false  (baseline)
 *   T3: block_num=N,  require_sync_start=true   (cross-thread drain)
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

static void submit_aiv(const Tensor &out, int16_t block_num, int64_t base_cl, bool sync_start) {
    L0TaskArgs args;
    args.add_inout(out);
    args.add_scalar(base_cl);
    args.launch_spec.set_block_num(block_num);
    args.launch_spec.set_require_sync_start(sync_start);
    rt_submit_aiv_task(FUNC_SPMD_WRITE_AIV, args);
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const L2TaskArgs &orch_args) {
    const Tensor &ext_output = orch_args.tensor(0).ref();
    const int32_t n = rt_available_cluster_count();
    const int16_t bn0 = 4;
    const int16_t bn1 = 16;
    const int16_t bn2 = 4;
    const int16_t bn3 = static_cast<int16_t>(n);
    const int64_t base0 = 0;
    const int64_t base1 = base0 + bn0;
    const int64_t base2 = base1 + bn1;
    const int64_t base3 = base2 + bn2;

    submit_aiv(ext_output, bn0, base0, true);
    submit_aiv(ext_output, bn1, base1, true);
    submit_aiv(ext_output, bn2, base2, false);
    submit_aiv(ext_output, bn3, base3, true);

    LOG_INFO_V9(
        "[spmd_sync_start_aiv] Submitted 4 AIV tasks (3 sync_start + 1 baseline), T3 block_num=%d", n
    );
}

}  // extern "C"

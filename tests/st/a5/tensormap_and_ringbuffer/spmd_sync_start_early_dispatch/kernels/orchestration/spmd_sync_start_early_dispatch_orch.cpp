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
 * SPMD Sync-Start Early-Dispatch Orchestration (a5)
 *
 * Port of a2a3 spmd_sync_start_early_dispatch. Host scalar[0] selects whether
 * the producer is flagged allow_early_resolve (1) or not (0), so the same scene
 * can compare early vs ready-only swimlanes.
 *
 *   P: AIC core_num=50, base_cl=0, allow_early_resolve = scalar early_on
 *   C: MIX core_num=24, base_cl=50, require_sync_start=true, dep=[P]
 *
 * Args layout: [output], scalar: early_on
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)
#include "pto_arg_with_deps.h"      // NOLINT(build/include_subdir)

#define FUNC_SPMD_WRITE_AIC 0
#define FUNC_SPMD_MIX_AIC 1
#define FUNC_SPMD_MIX_AIV0 2
#define FUNC_SPMD_MIX_AIV1 3

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return PTO2OrchestrationConfig{
        .expected_arg_count = 1,
    };
}

// Match a2a3 ST (1e7): board is fast enough that 2e6 often finishes before the
// scheduler can stage the sync_start consumer on the early path.
static constexpr int64_t PRODUCER_SPIN_ITERS = 10000000;

static PTO2TaskId submit_producer(const Tensor &out, int16_t core_num, int64_t base_cl, bool early_on) {
    L0TaskArgs args;
    args.add_inout(out);
    args.add_scalar(base_cl);
    args.add_scalar(PRODUCER_SPIN_ITERS);
    args.launch_spec.set_core_num(core_num);
    args.set_allow_early_resolve(early_on);
    return rt_submit_aic_task(FUNC_SPMD_WRITE_AIC, args).task_id();
}

static void submit_sync_consumer(const Tensor &out, int16_t core_num, int64_t base_cl, PTO2TaskId dep) {
    MixedKernels kernels;
    kernels.aic_kernel_id = FUNC_SPMD_MIX_AIC;
    kernels.aiv0_kernel_id = FUNC_SPMD_MIX_AIV0;
    kernels.aiv1_kernel_id = FUNC_SPMD_MIX_AIV1;
    L0TaskArgsWithDeps<4> args;
    args.add_inout(out);
    args.add_scalar(base_cl);
    args.launch_spec.set_core_num(core_num);
    args.launch_spec.set_require_sync_start(true);
    args.add_dep(dep);
    rt_submit_task(kernels, args);
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const L2TaskArgs &orch_args) {
    const Tensor &ext_output = orch_args.tensor(0).ref();
    const bool early_on = orch_args.scalar(0) != 0;

    rt_scope_begin(PTO2ScopeMode::MANUAL);
    PTO2TaskId prod = submit_producer(ext_output, 50, 0, early_on);
    submit_sync_consumer(ext_output, 24, 50, prod);
    rt_scope_end();

    LOG_INFO_V9(
        "[spmd_sync_start_early_dispatch] early_on=%d wide AIC producer + MIX sync_start consumer submitted",
        early_on ? 1 : 0
    );
}

}  // extern "C"

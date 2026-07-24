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
 * SPMD sync_start MIX pending-spill Orchestration (a5)
 *
 * Sized from rt_available_* (not hardcoded 24/36/72):
 *   P: AIV core_num=available_aiv, base_cl=0,  allow_early_resolve = scalar early_on
 *   C: MIX core_num=available_cluster, base_cl=available_aiv, require_sync_start=true, dep=[P]
 *
 * Args layout: [output], scalar: early_on
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)
#include "pto_arg_with_deps.h"      // NOLINT(build/include_subdir)

#define FUNC_SPMD_MIX_AIC 0
#define FUNC_SPMD_MIX_AIV0 1
#define FUNC_SPMD_MIX_AIV1 2
#define FUNC_SPMD_WRITE_AIV 3

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return PTO2OrchestrationConfig{
        .expected_arg_count = 1,
    };
}

static constexpr int64_t PRODUCER_SPIN_ITERS = 2000000;

static MixedKernels mix_kernels() {
    MixedKernels mk;
    mk.aic_kernel_id = FUNC_SPMD_MIX_AIC;
    mk.aiv0_kernel_id = FUNC_SPMD_MIX_AIV0;
    mk.aiv1_kernel_id = FUNC_SPMD_MIX_AIV1;
    return mk;
}

static PTO2TaskId submit_aiv_producer(const Tensor &out, int16_t core_num, int64_t base_cl, bool early_on) {
    L0TaskArgs args;
    args.add_inout(out);
    args.add_scalar(base_cl);
    args.add_scalar(PRODUCER_SPIN_ITERS);
    args.launch_spec.set_core_num(core_num);
    args.set_allow_early_resolve(early_on);
    return rt_submit_aiv_task(FUNC_SPMD_WRITE_AIV, args).task_id();
}

static void submit_mix_sync_consumer(const Tensor &out, int16_t core_num, int64_t base_cl, PTO2TaskId dep) {
    L0TaskArgsWithDeps<4> args;
    args.add_inout(out);
    args.add_scalar(base_cl);
    args.add_scalar(0);  // consumer does not spin
    args.launch_spec.set_core_num(core_num);
    args.launch_spec.set_require_sync_start(true);
    args.add_dep(dep);
    rt_submit_task(mix_kernels(), args);
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const L2TaskArgs &orch_args) {
    const Tensor &ext_output = orch_args.tensor(0).ref();
    const bool early_on = orch_args.scalar(0) != 0;

    const PTO2SyncStartCapacity cap = rt_sync_start_capacity();
    const int32_t n_cluster = cap.mix;
    const int32_t n_aiv = cap.aiv;

    PTO2TaskId prod = submit_aiv_producer(ext_output, static_cast<int16_t>(n_aiv), 0, early_on);
    submit_mix_sync_consumer(ext_output, static_cast<int16_t>(n_cluster), n_aiv, prod);

    LOG_INFO_V9(
        "[spmd_sync_start_mix_spill] early_on=%d sync_cap aic=%d aiv=%d mix=%d; AIV producer(%d) + MIX sync_start consumer(%d) submitted",
        early_on ? 1 : 0, cap.aic, cap.aiv, cap.mix, n_aiv, n_cluster
    );
}

}  // extern "C"

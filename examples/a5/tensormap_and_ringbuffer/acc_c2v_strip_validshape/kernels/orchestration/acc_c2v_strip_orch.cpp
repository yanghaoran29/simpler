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
 * Orchestration for Acc ValidShape strip C2V repro.
 * One MixedKernels task: C = A @ B via Acc strip TPUSH/TPOP.
 * Arg layout: [A, B, C]
 */

#include <cstdint>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_AIC 0
#define FUNC_AIV 1

static constexpr int M = 128;
static constexpr int N = 256;
static constexpr int K = 32;

static constexpr uint32_t A_ELEMS = static_cast<uint32_t>(M * K);
static constexpr uint32_t B_ELEMS = static_cast<uint32_t>(K * N);
static constexpr uint32_t C_ELEMS = static_cast<uint32_t>(M * N);

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 3,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const L2TaskArgs &orch_args) {
    const Tensor &ext_A = orch_args.tensor(0).ref();
    const Tensor &ext_B = orch_args.tensor(1).ref();
    const Tensor &ext_C = orch_args.tensor(2).ref();

    LOG_INFO_V0("[acc_c2v_strip] M=%d N=%d K=%d  Acc ValidShape strip C2V repro", M, N, K);

    uint32_t a_shapes[1] = {A_ELEMS};
    uint32_t b_shapes[1] = {B_ELEMS};
    uint32_t c_shapes[1] = {C_ELEMS};
    uint32_t zero[1] = {0};

    Tensor A_view = ext_A.view(a_shapes, zero);
    Tensor B_view = ext_B.view(b_shapes, zero);
    Tensor C_view = ext_C.view(c_shapes, zero);

    L0TaskArgs args;
    args.add_input(A_view);
    args.add_input(B_view);
    args.add_output(C_view);

    MixedKernels mk;
    mk.aic_kernel_id = FUNC_AIC;
    mk.aiv0_kernel_id = FUNC_AIV;
    mk.aiv1_kernel_id = FUNC_AIV;
    rt_submit_task(mk, args);

    LOG_INFO_V0("[acc_c2v_strip] submitted 1 MixedKernels (AIC+AIV strip C2V)");
}

}  // extern "C"

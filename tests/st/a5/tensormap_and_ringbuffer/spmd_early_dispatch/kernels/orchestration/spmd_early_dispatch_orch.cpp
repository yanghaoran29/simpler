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
 * SPMD Early-Dispatch Orchestration (a5) — plain early_resolve, NO sync_start
 *
 * Exercises the shape-queued early path (`early_dispatch_queues[]`), not the
 * sync_start cohort lane (`early_sync_start_queue`).
 *
 * Topology is sized so the producer leaves spare AIC cores: while P spins on
 * 8 AICs, C can gated-stage onto the remaining idle AICs.
 *
 *   P: AIC core_num=8, base_cl=0, allow_early_resolve = scalar early_on, spin
 *   C: AIC core_num=8, base_cl=8, dep=[P], require_sync_start=false
 *
 * Args layout: [output], scalar: early_on
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)
#include "pto_arg_with_deps.h"      // NOLINT(build/include_subdir)

#define FUNC_SPMD_WRITE_AIC 0

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return PTO2OrchestrationConfig{
        .expected_arg_count = 1,
    };
}

// Match a2a3 #1079-era spin: long enough that C can stage while P is still on-core.
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

static void submit_consumer(const Tensor &out, int16_t core_num, int64_t base_cl, PTO2TaskId dep) {
    L0TaskArgsWithDeps<4> args;
    args.add_inout(out);
    args.add_scalar(base_cl);
    args.add_scalar(0);  // no spin
    args.launch_spec.set_core_num(core_num);
    // deliberately NOT require_sync_start — plain early_dispatch_queues path
    args.add_dep(dep);
    rt_submit_aic_task(FUNC_SPMD_WRITE_AIC, args);
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const L2TaskArgs &orch_args) {
    const Tensor &ext_output = orch_args.tensor(0).ref();
    // Host Scalar("early_on") is orch scalar(0). Also stamp a sentinel into the
    // output so board runs can prove the scalar reached the orch (out[1] = early_on).
    const bool early_on = orch_args.scalar(0) != 0;
    {
        // out layout: CL0..7 producer, CL8..15 consumer; float index 1 is unused by kernels.
        float *out = reinterpret_cast<float *>(ext_output.buffer.addr) + ext_output.start_offset;
        out[1] = early_on ? 1.0f : 0.0f;
    }

    rt_scope_begin(PTO2ScopeMode::MANUAL);
    PTO2TaskId prod = submit_producer(ext_output, 8, 0, early_on);
    submit_consumer(ext_output, 8, 8, prod);
    rt_scope_end();

    LOG_INFO_V9("[spmd_early_dispatch] early_on=%d AIC producer + AIC consumer (no sync_start)", early_on ? 1 : 0);
}

}  // extern "C"

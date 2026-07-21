/*
 * Orchestration: A, B, C_full, C_strip — one MixedKernels.
 */
#include <cstdint>
#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_AIC 0
#define FUNC_AIV 1

static constexpr int M = 32;
static constexpr int N = 128;
static constexpr int K = 32;
static constexpr uint32_t A_ELEMS = static_cast<uint32_t>(M * K);
static constexpr uint32_t B_ELEMS = static_cast<uint32_t>(K * N);
static constexpr uint32_t C_ELEMS = static_cast<uint32_t>(M * N);

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{.expected_arg_count = 4};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const L2TaskArgs &orch_args) {
    const Tensor &ext_A = orch_args.tensor(0).ref();
    const Tensor &ext_B = orch_args.tensor(1).ref();
    const Tensor &ext_Cf = orch_args.tensor(2).ref();
    const Tensor &ext_Cs = orch_args.tensor(3).ref();

    uint32_t a_shapes[1] = {A_ELEMS};
    uint32_t b_shapes[1] = {B_ELEMS};
    uint32_t c_shapes[1] = {C_ELEMS};
    uint32_t zero[1] = {0};

    Tensor A_view = ext_A.view(a_shapes, zero);
    Tensor B_view = ext_B.view(b_shapes, zero);
    Tensor Cf_view = ext_Cf.view(c_shapes, zero);
    Tensor Cs_view = ext_Cs.view(c_shapes, zero);

    L0TaskArgs args;
    args.add_input(A_view);
    args.add_input(B_view);
    args.add_output(Cf_view);
    args.add_output(Cs_view);

    MixedKernels mk;
    mk.aic_kernel_id = FUNC_AIC;
    mk.aiv0_kernel_id = FUNC_AIV;
    mk.aiv1_kernel_id = FUNC_AIV;
    rt_submit_task(mk, args);
}

}  // extern "C"

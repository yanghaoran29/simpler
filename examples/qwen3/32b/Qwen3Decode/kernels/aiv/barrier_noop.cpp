// Kernel Function: barrier_noop
//
// Synthetic AIV kernel used purely as a dependency-reduction barrier.
// The kernel does no compute and touches no memory — its only purpose is
// to give the task scheduler a target task_id that fans-in to many upstream
// producers, so a downstream task can depend on the single barrier id
// instead of the full producer list. This keeps per-task explicit deps
// below PTO2_MAX_EXPLICIT_DEPS without raising the cap.
//
// Signature: [] (no tensor args, no scalars). The orchestration passes
// only add_dep(...) on the upstream producers.

#include <cstdint>

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#if defined(__CPU_SIM)
#define __aicore__
#else
#define __aicore__ [aicore]
#endif
#endif

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args)
{
    (void)args;
    // No-op: completion signal alone drives downstream readiness.
    return;
}

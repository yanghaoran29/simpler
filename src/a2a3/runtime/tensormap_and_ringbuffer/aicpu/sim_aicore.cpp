/**
 * @file sim_aicore.cpp
 * @brief Simulated AICore COND state and API (PTO2_SIM_AICORE_UT).
 *
 * Simulates the AICore side of the register handshake: scheduler writes
 * DATA_MAIN_BASE = task_id+1 -> sim "receives" task -> we set COND = FIN(task_id)
 * so the scheduler sees completion on the next read_reg(COND) without running a kernel.
 */

#if defined(PTO2_SIM_AICORE_UT)

#include "sim_aicore.h"
#include "runtime.h"                   // RUNTIME_MAX_WORKER
#include "common/platform_config.h"   // AICORE_IDLE_VALUE, MAKE_FIN_VALUE
#include <atomic>
#include <cstring>

#ifndef AICPU_SIM_PROF_WORKER_TYPES
#define AICPU_SIM_PROF_WORKER_TYPES 4
#endif

uint32_t s_sim_core_cond_value[RUNTIME_MAX_WORKER] = {};

// Run profiling: accumulated from all scheduler threads during aicpu_sim_run_pto2
static std::atomic<int64_t> s_sim_tasks_dispatched[AICPU_SIM_PROF_WORKER_TYPES];
static std::atomic<int64_t> s_sim_fanout_edges_total{0};
static std::atomic<int32_t> s_sim_fanout_max_degree{0};
static std::atomic<int64_t> s_sim_tasks_enqueued_by_completion{0};
static std::atomic<int64_t> s_sim_fanin_edges_total{0};
static std::atomic<int32_t> s_sim_fanin_max_degree{0};
static std::atomic<int64_t> s_sim_rounds_total{0};
static std::atomic<int64_t> s_sim_rounds_with_progress{0};
static std::atomic<uint64_t> s_sim_complete_cycle{0};
static std::atomic<uint64_t> s_sim_dispatch_cycle{0};

extern "C" uint64_t pto2_sim_read_cond_reg(int32_t core_id) {
    if (core_id < 0 || core_id >= RUNTIME_MAX_WORKER)
        return static_cast<uint64_t>(AICORE_IDLE_VALUE);
    return static_cast<uint64_t>(s_sim_core_cond_value[core_id]);
}

extern "C" void pto2_sim_aicore_on_task_received(int32_t core_id, int32_t task_id) {
    if (core_id < 0 || core_id >= RUNTIME_MAX_WORKER)
        return;
    s_sim_core_cond_value[core_id] = static_cast<uint32_t>(MAKE_FIN_VALUE(task_id));
}

extern "C" void pto2_sim_aicore_set_idle(int32_t core_id) {
    if (core_id < 0 || core_id >= RUNTIME_MAX_WORKER)
        return;
    s_sim_core_cond_value[core_id] = static_cast<uint32_t>(AICORE_IDLE_VALUE);
}

extern "C" void pto2_sim_aicore_init_all_idle(void) {
    for (int32_t i = 0; i < RUNTIME_MAX_WORKER; i++)
        s_sim_core_cond_value[i] = static_cast<uint32_t>(AICORE_IDLE_VALUE);
}

void pto2_sim_reset_run_prof(void) {
    for (int i = 0; i < AICPU_SIM_PROF_WORKER_TYPES; i++)
        s_sim_tasks_dispatched[i].store(0, std::memory_order_relaxed);
    s_sim_fanout_edges_total.store(0, std::memory_order_relaxed);
    s_sim_fanout_max_degree.store(0, std::memory_order_relaxed);
    s_sim_tasks_enqueued_by_completion.store(0, std::memory_order_relaxed);
    s_sim_fanin_edges_total.store(0, std::memory_order_relaxed);
    s_sim_fanin_max_degree.store(0, std::memory_order_relaxed);
    s_sim_rounds_total.store(0, std::memory_order_relaxed);
    s_sim_rounds_with_progress.store(0, std::memory_order_relaxed);
    s_sim_complete_cycle.store(0, std::memory_order_relaxed);
    s_sim_dispatch_cycle.store(0, std::memory_order_relaxed);
}

void pto2_sim_accumulate_rounds(int64_t total_inc, int64_t with_progress_inc) {
    if (total_inc > 0) s_sim_rounds_total.fetch_add(total_inc, std::memory_order_relaxed);
    if (with_progress_inc > 0) s_sim_rounds_with_progress.fetch_add(with_progress_inc, std::memory_order_relaxed);
}

void pto2_sim_accumulate_fanin(int32_t fe) {
    if (fe <= 0) return;
    s_sim_fanin_edges_total.fetch_add(static_cast<int64_t>(fe), std::memory_order_relaxed);
    int32_t cur = s_sim_fanin_max_degree.load(std::memory_order_relaxed);
    while (fe > cur && !s_sim_fanin_max_degree.compare_exchange_weak(cur, fe, std::memory_order_relaxed))
        ;
}

void pto2_sim_accumulate_fanout(int64_t edges, int64_t enqueued, int32_t max_degree) {
    if (edges > 0) s_sim_fanout_edges_total.fetch_add(edges, std::memory_order_relaxed);
    if (enqueued > 0) s_sim_tasks_enqueued_by_completion.fetch_add(enqueued, std::memory_order_relaxed);
    if (max_degree <= 0) return;
    int32_t cur = s_sim_fanout_max_degree.load(std::memory_order_relaxed);
    while (max_degree > cur && !s_sim_fanout_max_degree.compare_exchange_weak(cur, max_degree, std::memory_order_relaxed))
        ;
}

void pto2_sim_accumulate_dispatch(int32_t worker_type) {
    if (worker_type >= 0 && worker_type < AICPU_SIM_PROF_WORKER_TYPES)
        s_sim_tasks_dispatched[worker_type].fetch_add(1, std::memory_order_relaxed);
}

void pto2_sim_accumulate_cycles(uint64_t complete_cycle, uint64_t dispatch_cycle) {
    if (complete_cycle > 0) s_sim_complete_cycle.fetch_add(complete_cycle, std::memory_order_relaxed);
    if (dispatch_cycle > 0) s_sim_dispatch_cycle.fetch_add(dispatch_cycle, std::memory_order_relaxed);
}

extern "C" void aicpu_sim_get_run_prof(AicpuSimRunProf* out) {
    if (!out) return;
    std::memset(out, 0, sizeof(AicpuSimRunProf));
    for (int i = 0; i < AICPU_SIM_PROF_WORKER_TYPES; i++)
        out->tasks_dispatched[i] = s_sim_tasks_dispatched[i].load(std::memory_order_relaxed);
    out->fanout_edges_total = s_sim_fanout_edges_total.load(std::memory_order_relaxed);
    out->fanout_max_degree = s_sim_fanout_max_degree.load(std::memory_order_relaxed);
    out->tasks_enqueued_by_completion = s_sim_tasks_enqueued_by_completion.load(std::memory_order_relaxed);
    out->fanin_edges_total = s_sim_fanin_edges_total.load(std::memory_order_relaxed);
    out->fanin_max_degree = s_sim_fanin_max_degree.load(std::memory_order_relaxed);
    out->rounds_total = s_sim_rounds_total.load(std::memory_order_relaxed);
    out->rounds_with_progress = s_sim_rounds_with_progress.load(std::memory_order_relaxed);
    out->complete_cycle = s_sim_complete_cycle.load(std::memory_order_relaxed);
    out->dispatch_cycle = s_sim_dispatch_cycle.load(std::memory_order_relaxed);
}

#endif  // PTO2_SIM_AICORE_UT

/**
 * PTO Runtime2 - Scheduler Implementation
 *
 * Implements scheduler state management, ready queues, and task lifecycle.
 *
 * Based on: docs/runtime_buffer_manager_methods.md
 */

#include "pto_scheduler.h"
#include <inttypes.h>
#include <new>
#include <stdlib.h>
#include <utility>
#include "common/unified_log.h"

// =============================================================================
// Scheduler Profiling Counters
// =============================================================================

#if PTO2_SCHED_PROFILING || PTO2_ORCH_PROFILING || PTO2_PROFILING
#include "common/platform_config.h"
#endif
#if PTO2_ORCH_PROFILING
#include "pto_orchestrator.h"
#endif

#if PTO2_SCHED_PROFILING

uint64_t g_sched_lock_cycle[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_fanout_cycle[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_fanin_cycle[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_self_consumed_cycle[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_lock_wait_cycle[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_push_wait_cycle[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_pop_wait_cycle[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_lock_atomic_count[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_fanout_atomic_count[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_fanin_atomic_count[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_self_atomic_count[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_pop_atomic_count[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_complete_count[PLATFORM_MAX_AICPU_THREADS] = {};

PTO2SchedProfilingData pto2_scheduler_get_profiling(int thread_idx) {
    PTO2SchedProfilingData d;
    d.lock_cycle = std::exchange(g_sched_lock_cycle[thread_idx], 0);
    d.fanout_cycle = std::exchange(g_sched_fanout_cycle[thread_idx], 0);
    d.fanin_cycle = std::exchange(g_sched_fanin_cycle[thread_idx], 0);
    d.self_consumed_cycle = std::exchange(g_sched_self_consumed_cycle[thread_idx], 0);
    d.lock_wait_cycle = std::exchange(g_sched_lock_wait_cycle[thread_idx], 0);
    d.push_wait_cycle = std::exchange(g_sched_push_wait_cycle[thread_idx], 0);
    d.pop_wait_cycle = std::exchange(g_sched_pop_wait_cycle[thread_idx], 0);
    d.lock_atomic_count = std::exchange(g_sched_lock_atomic_count[thread_idx], 0);
    d.fanout_atomic_count = std::exchange(g_sched_fanout_atomic_count[thread_idx], 0);
    d.fanin_atomic_count = std::exchange(g_sched_fanin_atomic_count[thread_idx], 0);
    d.self_atomic_count = std::exchange(g_sched_self_atomic_count[thread_idx], 0);
    d.pop_atomic_count = std::exchange(g_sched_pop_atomic_count[thread_idx], 0);
    d.complete_count = std::exchange(g_sched_complete_count[thread_idx], 0);
    return d;
}
#endif

// =============================================================================
// Task State Names
// =============================================================================

const char* pto2_task_state_name(PTO2TaskState state) {
    switch (state) {
        case PTO2_TASK_PENDING:   return "PENDING";
        case PTO2_TASK_READY:     return "READY";
        case PTO2_TASK_RUNNING:   return "RUNNING";
        case PTO2_TASK_COMPLETED: return "COMPLETED";
        case PTO2_TASK_CONSUMED:  return "CONSUMED";
        default:                  return "UNKNOWN";
    }
}

// =============================================================================
// Ready Queue Implementation
// =============================================================================

bool pto2_ready_queue_init(PTO2ReadyQueue* queue, uint64_t capacity) {
    queue->slots = (PTO2ReadyQueueSlot*)malloc(capacity * sizeof(PTO2ReadyQueueSlot));
    if (!queue->slots) {
        return false;
    }

    queue->capacity = capacity;
    queue->mask = capacity - 1;
    queue->enqueue_pos.store(0, std::memory_order_relaxed);
    queue->dequeue_pos.store(0, std::memory_order_relaxed);

    for (uint64_t i = 0; i < capacity; i++) {
        queue->slots[i].sequence.store((int64_t)i, std::memory_order_relaxed);
        queue->slots[i].task_id = -1;
    }

    return true;
}

void pto2_ready_queue_destroy(PTO2ReadyQueue* queue) {
    if (queue->slots) {
        free(queue->slots);
        queue->slots = NULL;
    }
}

// =============================================================================
// Scheduler Initialization
// =============================================================================

bool pto2_scheduler_init(PTO2SchedulerState* sched,
                          PTO2SharedMemoryHandle* sm_handle,
                          void* heap_base) {
    sched->sm_handle = sm_handle;
    sched->task_descriptors = sm_handle->task_descriptors;
    sched->heap_base = heap_base;
    sched->slot_states = nullptr;
#if PTO2_SCHED_PROFILING
    sched->tasks_completed.store(0, std::memory_order_relaxed);
    sched->tasks_consumed.store(0, std::memory_order_relaxed);
#endif
    sched->ring_advance_lock.store(0, std::memory_order_relaxed);

    // Get runtime task_window_size from shared memory header
    uint64_t window_size = sm_handle->header->task_window_size;
    sched->task_window_size = window_size;
    sched->task_window_mask = window_size - 1;  // For fast modulo (window_size must be power of 2)

    // Initialize local copies of ring pointers
    sched->last_task_alive = 0;
    sched->last_heap_consumed = 0;
    sched->heap_tail = 0;

    // Allocate per-task slot state array (dynamically sized based on runtime window_size)
    sched->slot_states = new (std::nothrow) PTO2TaskSlotState[window_size];
    if (!sched->slot_states) {
        return false;
    }

    // Zero-initialize all per-task slot state fields.
    // new[] default-initializes std::atomic<T> which leaves values indeterminate.
    // Scheduler logic (e.g. fanin_refcount fetch_add in release_fanin_and_check_ready)
    // assumes slots start at zero before init_task writes them.
    for (uint64_t i = 0; i < window_size; i++) {
        sched->slot_states[i].fanout_lock.store(0, std::memory_order_relaxed);
        sched->slot_states[i].fanout_count = 0;
        sched->slot_states[i].fanout_head = nullptr;
        sched->slot_states[i].task_state.store(static_cast<PTO2TaskState>(0), std::memory_order_relaxed);
        sched->slot_states[i].fanin_refcount.store(0, std::memory_order_relaxed);
        sched->slot_states[i].fanin_count = 0;
        sched->slot_states[i].fanout_refcount.store(0, std::memory_order_relaxed);
    }

    // Initialize ready queues (one per resource shape)
    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        if (!pto2_ready_queue_init(&sched->ready_queues[i], PTO2_READY_QUEUE_SIZE)) {
            // Cleanup on failure
            for (int j = 0; j < i; j++) {
                pto2_ready_queue_destroy(&sched->ready_queues[j]);
            }
            delete[] sched->slot_states;
            sched->slot_states = nullptr;
            return false;
        }
    }

    return true;
}

void pto2_scheduler_destroy(PTO2SchedulerState* sched) {
    if (sched->slot_states) {
        delete[] sched->slot_states;
        sched->slot_states = nullptr;
    }

    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        pto2_ready_queue_destroy(&sched->ready_queues[i]);
    }
}

// =============================================================================
// Debug Utilities
// =============================================================================

void pto2_scheduler_print_stats(PTO2SchedulerState* sched) {
    LOG_INFO("=== Scheduler Statistics ===");
    LOG_INFO("last_task_alive:   %d", sched->last_task_alive);
    LOG_INFO("heap_tail:         %" PRIu64, sched->heap_tail);
#if PTO2_SCHED_PROFILING
    LOG_INFO("tasks_completed:   %lld", (long long)sched->tasks_completed.load(std::memory_order_relaxed));
    LOG_INFO("tasks_consumed:    %lld", (long long)sched->tasks_consumed.load(std::memory_order_relaxed));
#endif
    LOG_INFO("============================");
}

void pto2_scheduler_print_queues(PTO2SchedulerState* sched) {
    LOG_INFO("=== Ready Queues ===");

    const char* shape_names[] = {"AIC_ONLY", "AIV_X1", "AIV_X2", "AIC_AIV_X1", "AIC_AIV_X2"};

    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        LOG_INFO("  %s: count=%" PRIu64, shape_names[i],
                 sched->ready_queues[i].size());
    }

    LOG_INFO("====================");
}

// =============================================================================
// Profiling Print Functions
// =============================================================================

#if PTO2_SCHED_PROFILING
void pto2_print_sched_profiling(int thread_idx) {
    PTO2SchedProfilingData sp = pto2_scheduler_get_profiling(thread_idx);
    uint64_t total = sp.lock_cycle + sp.fanout_cycle + sp.fanin_cycle + sp.self_consumed_cycle;
    if (total == 0) total = 1;

    LOG_ALWAYS("Thread %d: === Scheduler Profiling: %lld tasks, total=%.3fus ===",
        thread_idx, (long long)sp.complete_count, cycles_to_us(total));
    LOG_ALWAYS("Thread %d:   lock+state   : %.3fus  work=%.3fus wait=%.3fus  atomics=%llu",
        thread_idx, cycles_to_us(sp.lock_cycle),
        cycles_to_us(sp.lock_cycle - sp.lock_wait_cycle), cycles_to_us(sp.lock_wait_cycle),
        (unsigned long long)sp.lock_atomic_count);
    LOG_ALWAYS("Thread %d:   fanout       : %.3fus  work=%.3fus wait=%.3fus  atomics=%llu",
        thread_idx, cycles_to_us(sp.fanout_cycle),
        cycles_to_us(sp.fanout_cycle - sp.push_wait_cycle), cycles_to_us(sp.push_wait_cycle),
        (unsigned long long)sp.fanout_atomic_count);
    LOG_ALWAYS("Thread %d:   fanin        : %.3fus  atomics=%llu",
        thread_idx, cycles_to_us(sp.fanin_cycle),
        (unsigned long long)sp.fanin_atomic_count);
    LOG_ALWAYS("Thread %d:   self_consumed: %.3fus  atomics=%llu",
        thread_idx, cycles_to_us(sp.self_consumed_cycle),
        (unsigned long long)sp.self_atomic_count);
    LOG_ALWAYS("Thread %d:   pop_wait     : %.3fus  atomics=%llu",
        thread_idx, cycles_to_us(sp.pop_wait_cycle),
        (unsigned long long)sp.pop_atomic_count);
    if (sp.complete_count > 0) {
        LOG_ALWAYS("Thread %d:   avg/task     : %.3fus",
            thread_idx, cycles_to_us(total) / sp.complete_count);
    }
}
#endif

#if PTO2_ORCH_PROFILING

void pto2_print_orch_profiling() {
    PTO2OrchProfilingData p = pto2_orchestrator_get_profiling();
    uint64_t total = p.sync_cycle + p.alloc_cycle + p.params_cycle +
                     p.lookup_cycle + p.heap_cycle + p.insert_cycle +
                     p.fanin_cycle;
    if (total == 0) total = 1;

    LOG_ALWAYS("  Orchestrator Profiling: %lld tasks, total=%.3fus",
        (long long)p.submit_count, cycles_to_us(total));
    LOG_ALWAYS("  sync_tensormap : %.3fus (%.1f%%)",
        cycles_to_us(p.sync_cycle), p.sync_cycle * 100.0 / total);
    LOG_ALWAYS("  task_ring_alloc: %.3fus  work=%.3fus wait=%.3fus  atomics=%llu",
        cycles_to_us(p.alloc_cycle),
        cycles_to_us(p.alloc_cycle - p.alloc_wait_cycle), cycles_to_us(p.alloc_wait_cycle),
        (unsigned long long)p.alloc_atomic_count);
    LOG_ALWAYS("  param_copy     : %.3fus  atomics=%llu",
        cycles_to_us(p.params_cycle), (unsigned long long)p.params_atomic_count);
    LOG_ALWAYS("  lookup+dep     : %.3fus", cycles_to_us(p.lookup_cycle));
    LOG_ALWAYS("  heap_alloc     : %.3fus  work=%.3fus wait=%.3fus  atomics=%llu",
        cycles_to_us(p.heap_cycle),
        cycles_to_us(p.heap_cycle - p.heap_wait_cycle), cycles_to_us(p.heap_wait_cycle),
        (unsigned long long)p.heap_atomic_count);
    LOG_ALWAYS("  tensormap_ins  : %.3fus", cycles_to_us(p.insert_cycle));
    LOG_ALWAYS("  fanin+ready    : %.3fus  work=%.3fus wait=%.3fus  atomics=%llu",
        cycles_to_us(p.fanin_cycle),
        cycles_to_us(p.fanin_cycle - p.fanin_wait_cycle), cycles_to_us(p.fanin_wait_cycle),
        (unsigned long long)p.fanin_atomic_count);
    LOG_ALWAYS("  finalize+SM    : wait=%.3fus  atomics=%llu",
        cycles_to_us(p.finalize_wait_cycle),
        (unsigned long long)p.finalize_atomic_count);
    LOG_ALWAYS("  scope_end      : %.3fus  atomics=%llu",
        cycles_to_us(p.scope_end_cycle), (unsigned long long)p.scope_end_atomic_count);
    if (p.submit_count > 0) {
        LOG_ALWAYS("  avg/task       : %.3fus", cycles_to_us(total) / p.submit_count);
    }
}
#endif

#if PTO2_PROFILING
void pto2_print_sim_sched_summary(const PTO2SimSchedSummary* s, int64_t tasks_completed, int64_t tasks_consumed) {
    if (s == nullptr) return;
    static const char* wt_names[] = {"CUBE", "VECTOR", "AI_CPU", "ACCELERATOR"};
    int64_t total = 0;
    for (int i = 0; i < PTO2_NUM_WORKER_TYPES; i++)
        total += s->tasks_dispatched[i];

    double sched_cpu_us = 0.0;
    if (total > 0) {
        uint64_t sc = s->dispatch_cycle + s->complete_cycle;
        sched_cpu_us = cycles_to_us(sc) / (double)total;
    }
    LOG_ALWAYS("");
    LOG_ALWAYS("  ---------------------------- Task execution vs Scheduler overhead ----------------------------");
#if defined(PTO2_SIM_AICORE_UT)
    LOG_ALWAYS("    Per-task (all):  Avg Latency (dispatch→finish) = N/A (sim)");
#else
    LOG_ALWAYS("    Per-task (all):  Avg Latency (dispatch→finish) = 0.00 us");
#endif
    LOG_ALWAYS("    Sched CPU (sim): %.2f us/task", sched_cpu_us);
    LOG_ALWAYS("    (Sched CPU = scheduler thread CPU per task)");
    LOG_ALWAYS("    Scheduler Profiling (%lld tasks)", (long long)total);
    for (int i = 0; i < PTO2_NUM_WORKER_TYPES; i++) {
        int64_t n = s->tasks_dispatched[i];
        if (n == 0) continue;
        LOG_ALWAYS("      %-12s %6lld tasks  (%4.1f%%)",
            wt_names[i], (long long)n, total > 0 ? n * 100.0 / total : 0.0);
    }
    LOG_ALWAYS("    fanout:          %lld edges, max_degree=%d, enqueued=%lld",
        (long long)s->fanout_edges_total, s->fanout_max_degree, (long long)s->tasks_enqueued_by_completion);
    LOG_ALWAYS("    fanin:           %lld edges, max_degree=%d",
        (long long)s->fanin_edges_total, s->fanin_max_degree);
    LOG_ALWAYS("    sim_rounds:      %lld total, %lld with_progress",
        (long long)s->rounds_total, (long long)s->rounds_with_progress);
    LOG_ALWAYS("    tasks_completed: %lld  tasks_consumed: %lld",
        (long long)tasks_completed, (long long)tasks_consumed);
}
#endif

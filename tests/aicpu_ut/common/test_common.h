/**
 * test_common.h
 *
 * Common definitions for orchestration unit tests
 */

#ifndef TEST_COMMON_H
#define TEST_COMMON_H

#include <cstdio>
#include <cstdint>

// Forward declaration
struct PTO2Runtime;

// Global test counters
extern int g_pass;
extern int g_fail;

// Test framework macros
#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            fprintf(stderr, "  FAIL [%s:%d]  %s\n",                         \
                    __FILE__, __LINE__, #cond);                              \
            g_fail++;                                                        \
        } else {                                                             \
            g_pass++;                                                        \
        }                                                                    \
    } while (0)

#define TEST_BEGIN(name) printf("\n=== %s ===\n", (name))
#define TEST_END() printf("  PASS: %d, FAIL: %d\n", g_pass, g_fail)

// Common helper functions
uint64_t float_to_u64(float f);
uint64_t perf_now_us();          // Monotonic clock, returns microseconds, for performance measurement
PTO2Runtime* make_runtime();
int sim_drain_one_pass(PTO2Runtime* rt);
int sim_run_all(PTO2Runtime* rt, int max_rounds = 1000);
void print_orch_profiling();

// Scheduler profiling data collected during simulation.
// Array size 4 matches PTO2_NUM_WORKER_TYPES (CUBE/VECTOR/AI_CPU/ACCELERATOR).
struct SchedProfilingData {
    int64_t tasks_dispatched[4];          // per worker type
    int64_t fanout_edges_total;           // total fanout edges traversed on completion
    int32_t fanout_max_degree;            // max fanout edges for a single task
    int64_t tasks_enqueued_by_completion; // tasks that became READY via on_task_complete
    int64_t fanin_edges_total;            // total fanin edges released on completion
    int32_t fanin_max_degree;             // max fanin edges for a single task
    int64_t rounds_total;                 // total simulation drain rounds
    int64_t rounds_with_progress;         // rounds that dispatched at least one task
    // Scheduler phase cycle counts (analogous to AicpuPhaseId SCHED_* phases)
    uint64_t dispatch_cycle;              // SCHED_DISPATCH: get_ready_task + mark_running
    uint64_t complete_cycle;             // SCHED_COMPLETE: on_task_complete
};

void print_sched_profiling(PTO2Runtime* rt);

#if PTO2_PROFILING
void orch_timing_begin();
void orch_timing_end();
#endif

#endif  // TEST_COMMON_H

/**
 * cpu_affinity.h
 *
 * CPU affinity utility functions and thread-based affinity test helpers.
 *
 * Utility functions:
 *   - Query number of online CPU cores
 *   - Get current thread's running CPU core number
 *   - Bind current thread to specified CPU core
 *   - Read current thread's CPU affinity mask (returns first bound core)
 *   - Print separator line helper function
 *
 * Thread affinity test helpers (for functional and perf tests):
 *   - ThreadReport struct to record single thread's binding result
 *   - run_threads() to create/join a group of AICPU-model threads
 *   - Three reusable affinity test cases callable from any test suite
 */

#ifndef CPU_AFFINITY_H
#define CPU_AFFINITY_H

#include <cstdio>

// ─────────────────────────────────────────────────────────────────────────────
// AICPU thread model: Thread 0-2 are schedulers, Thread 3 is orchestrator
// Corresponds to the LAUNCH_AICPU_NUM == 4 scenario in device_runner.cpp
// ─────────────────────────────────────────────────────────────────────────────

static const int LAUNCH_AICPU_NUM = 4;

// ─────────────────────────────────────────────────────────────────────────────
// Test report for each thread
// ─────────────────────────────────────────────────────────────────────────────

struct ThreadReport {
    int  thread_idx;
    int  target_cpu;   // Expected bound core (-1 = not set)
    int  bound_cpu;    // Core read back from pthread_getaffinity_np (-1 = not specifically bound)
    int  actual_cpu;   // Actual running core from sched_getcpu()
    bool bind_ok;      // Whether bind_to_cpu() succeeded
};

// ─────────────────────────────────────────────────────────────────────────────
// Utility functions
// ─────────────────────────────────────────────────────────────────────────────

// Print n '-' characters as indented separator line
void print_separator(int n);

// Return number of online CPU cores, returns 1 on non-Linux
int num_cpus_online();

// Return current thread's running CPU core number, returns -1 on failure or non-Linux
int current_cpu();

// Bind current thread to specified CPU core, returns 0 on success, non-zero on failure, -1 on non-Linux
int bind_to_cpu(int cpu_core);

// Clear current thread's CPU affinity restriction, allowing scheduling on all online cores
// Returns 0 on success, non-zero on failure, -1 on non-Linux
int unbind_from_cpu();

// Read which cores current thread is bound to, returns first bound core
// If binding set covers all online cores (considered "not specifically bound") or non-Linux, returns -1
int get_bound_cpu();

// Verify current thread is running on specified CPU core, print result and return whether verification passed
bool verify_cpu_binding(int target_cpu, const char* context = "");

// ─────────────────────────────────────────────────────────────────────────────
// Thread affinity test helpers
// ─────────────────────────────────────────────────────────────────────────────

// Run n threads using pthread_create/join, calling fn with each ThreadReport*
// Each reports[i].thread_idx is set to i before launch
void run_threads(void* (*fn)(void*), ThreadReport* reports, int n);

// Reusable affinity test cases — callable from any functional or perf test suite
void test_cpu_affinity_without_binding();
void test_cpu_affinity_with_binding_default();
void test_cpu_affinity_comparison();

#endif  // CPU_AFFINITY_H

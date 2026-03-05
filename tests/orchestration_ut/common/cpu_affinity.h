/**
 * cpu_affinity.h
 *
 * CPU affinity utility functions for all test files.
 *
 * Features:
 *   - Query number of online CPU cores
 *   - Get current thread's running CPU core number
 *   - Bind current thread to specified CPU core
 *   - Read current thread's CPU affinity mask (returns first bound core)
 *   - Print separator line helper function
 *   - ThreadReport struct to record single thread's binding result
 */

#ifndef CPU_AFFINITY_H
#define CPU_AFFINITY_H

#include <cstdio>

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

#endif  // CPU_AFFINITY_H

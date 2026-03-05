/**
 * test_cpu_affinity.cpp
 *
 * CPU Affinity Unit Tests
 *
 * Tests run directly on host CPU using pthread interface to verify:
 *   1. Which cores threads run on without binding (observe OS scheduling)
 *   2. After binding, verify each thread runs on the specified core
 *   3. Compare thread distribution with and without binding
 *
 * Corresponds to binding logic in device_runner.cpp:
 *   - When sche_cpu_num == 4: Thread 0-2 are schedulers, Thread 3 is orchestrator
 *   - When cpu_affinity_enabled = true: bind according to configuration
 *   - When orch_cpu_core / sched_cpu_cores[i] == -1: use defaults (orch→core0, sched[i]→core i+1)
 */

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

#include <cstdio>
#include <cstdint>

#include "test_common.h"
#include "cpu_affinity.h"

// ─────────────────────────────────────────────────────────────────────────────
// Thread model parameters (corresponds to LAUNCH_AICPU_NUM == 4 scenario in device_runner.cpp)
// ─────────────────────────────────────────────────────────────────────────────

static const int LAUNCH_AICPU_NUM = 4;  // Thread 0-2: scheduler, Thread 3: orchestrator

// ─────────────────────────────────────────────────────────────────────────────
// pthread entry function — without binding
// ─────────────────────────────────────────────────────────────────────────────

static void* thread_fn_no_bind(void* arg) {
    ThreadReport* r = (ThreadReport*)arg;
    r->target_cpu = -1;
    r->bind_ok    = false;

    // Clear inherited affinity mask to ensure truly "unbound" state
    unbind_from_cpu();

    // Busy-wait for a while to give OS a chance to schedule thread to different cores
    volatile uint64_t dummy = 0;
    for (int k = 0; k < 2000000; k++) dummy += (uint64_t)k;
    (void)dummy;

    r->bound_cpu  = get_bound_cpu();   // Should be -1 (unbound)
    r->actual_cpu = current_cpu();
    return nullptr;
}

// ─────────────────────────────────────────────────────────────────────────────
// pthread entry function — with binding (target_cpu written to r->target_cpu before launch)
// ─────────────────────────────────────────────────────────────────────────────

static void* thread_fn_with_bind(void* arg) {
    ThreadReport* r = (ThreadReport*)arg;
    int rc = bind_to_cpu(r->target_cpu);
    r->bind_ok = (rc == 0);

    volatile uint64_t dummy = 0;
    for (int k = 0; k < 2000000; k++) dummy += (uint64_t)k;
    (void)dummy;

    r->bound_cpu  = get_bound_cpu();
    r->actual_cpu = current_cpu();
    return nullptr;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: run a group of threads using pthread_create / pthread_join
// ─────────────────────────────────────────────────────────────────────────────

static void run_threads(void* (*fn)(void*), ThreadReport* reports, int n) {
    pthread_t tids[LAUNCH_AICPU_NUM];
    for (int i = 0; i < n; i++) {
        reports[i].thread_idx = i;
        pthread_create(&tids[i], nullptr, fn, &reports[i]);
    }
    for (int i = 0; i < n; i++) {
        pthread_join(tids[i], nullptr);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: Without binding — observe OS free scheduling
// ─────────────────────────────────────────────────────────────────────────────

void test_cpu_affinity_without_binding() {
    TEST_BEGIN("CPU Affinity - Without Binding (Observe OS Free Scheduling)");

    int ncpus = num_cpus_online();
    printf("  System CPU cores: %d\n", ncpus);
    printf("  Launching %d AICPU threads (Thread 0-2: scheduler, Thread 3: orchestrator)\n",
           LAUNCH_AICPU_NUM);

    ThreadReport reports[LAUNCH_AICPU_NUM];

#ifdef __linux__
    run_threads(thread_fn_no_bind, reports, LAUNCH_AICPU_NUM);

    printf("\n  %-12s %-15s %-14s %-14s\n", "Thread ID", "Role", "Bound Core", "Actual Core");
    print_separator(58);
    for (int i = 0; i < LAUNCH_AICPU_NUM; i++) {
        const ThreadReport& r = reports[i];
        const char* role = (i == 3) ? "orchestrator" : "scheduler";
        printf("  Thread %-5d %-15s %-14d %-14d\n",
               i, role, r.bound_cpu, r.actual_cpu);
    }

    printf("\n  Note: Without binding, threads may run on any core (OS decides)\n");
    printf("        bound_cpu should be -1 (no specific binding)\n");

    // No CHECK here, just observe
    CHECK(true);  // Always pass, this is observational
    TEST_END();
#else
    printf("  Skipped: Not Linux platform\n");
    TEST_END();
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: With binding (default) — orch→core0, sched[i]→core i+1
// ─────────────────────────────────────────────────────────────────────────────

void test_cpu_affinity_with_binding_default() {
    TEST_BEGIN("CPU Affinity - With Binding (Default: orch→0, sched[i]→i+1)");

    int ncpus = num_cpus_online();
    printf("  System CPU cores: %d\n", ncpus);
    printf("  Launching %d AICPU threads (Thread 0-2: scheduler, Thread 3: orchestrator)\n",
           LAUNCH_AICPU_NUM);

    // Default binding: orchestrator→core0, scheduler[i]→core i+1
    int target_cores[LAUNCH_AICPU_NUM] = {1, 2, 3, 0};  // sched0→1, sched1→2, sched2→3, orch→0

    ThreadReport reports[LAUNCH_AICPU_NUM];

#ifdef __linux__
    for (int i = 0; i < LAUNCH_AICPU_NUM; i++) {
        reports[i].target_cpu = target_cores[i];
    }

    run_threads(thread_fn_with_bind, reports, LAUNCH_AICPU_NUM);

    bool enough_cores = (ncpus >= 4);
    if (!enough_cores) {
        printf("\n  Warning: System has only %d cores, need at least 4 for this test\n", ncpus);
        printf("           Some threads may fail to bind to non-existent cores\n");
    }

    printf("\n  %-12s %-15s %-12s %-14s %-14s %-8s\n",
           "Thread ID", "Role", "Target Core", "Bound Core", "Actual Core", "Result");
    print_separator(82);

    int pass_count = 0;
    for (int i = 0; i < LAUNCH_AICPU_NUM; i++) {
        const ThreadReport& r = reports[i];
        const char* role = (i == 3) ? "orchestrator" : "scheduler";
        bool correct = r.bind_ok
                    && (r.bound_cpu  == r.target_cpu)
                    && (r.actual_cpu == r.target_cpu);
        printf("  Thread %-5d %-15s %-12d %-14d %-14d %s\n",
               i, role, r.target_cpu, r.bound_cpu, r.actual_cpu,
               correct ? "OK" : "FAIL");
        if (correct) pass_count++;
        if (enough_cores) CHECK(correct);
    }

    printf("\n  Summary: %d/%d threads successfully bound\n",
           pass_count, enough_cores ? LAUNCH_AICPU_NUM : pass_count);

    TEST_END();
#else
    printf("  Skipped: Not Linux platform\n");
    TEST_END();
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: With binding (custom) — user-specified cores
// ─────────────────────────────────────────────────────────────────────────────

void test_cpu_affinity_with_binding_custom() {
    TEST_BEGIN("CPU Affinity - With Binding (Custom: user-specified cores)");

    int ncpus = num_cpus_online();
    printf("  System CPU cores: %d\n", ncpus);
    printf("  Launching %d AICPU threads (Thread 0-2: scheduler, Thread 3: orchestrator)\n",
           LAUNCH_AICPU_NUM);

    // Custom binding: all threads to core 0 (stress test)
    int target_cores[LAUNCH_AICPU_NUM] = {0, 0, 0, 0};

    ThreadReport reports[LAUNCH_AICPU_NUM];

#ifdef __linux__
    for (int i = 0; i < LAUNCH_AICPU_NUM; i++) {
        reports[i].target_cpu = target_cores[i];
    }

    run_threads(thread_fn_with_bind, reports, LAUNCH_AICPU_NUM);

    printf("\n  %-12s %-15s %-12s %-14s %-14s %-8s\n",
           "Thread ID", "Role", "Target Core", "Bound Core", "Actual Core", "Result");
    print_separator(82);

    int pass_count = 0;
    for (int i = 0; i < LAUNCH_AICPU_NUM; i++) {
        const ThreadReport& r = reports[i];
        const char* role = (i == 3) ? "orchestrator" : "scheduler";
        bool correct = r.bind_ok
                    && (r.bound_cpu  == r.target_cpu)
                    && (r.actual_cpu == r.target_cpu);
        printf("  Thread %-5d %-15s %-12d %-14d %-14d %s\n",
               i, role, r.target_cpu, r.bound_cpu, r.actual_cpu,
               correct ? "OK" : "FAIL");
        if (correct) pass_count++;
        CHECK(correct);
    }

    printf("\n  Summary: %d/%d threads successfully bound\n", pass_count, LAUNCH_AICPU_NUM);

    TEST_END();
#else
    printf("  Skipped: Not Linux platform\n");
    TEST_END();
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: Comparison — with vs without binding
// ─────────────────────────────────────────────────────────────────────────────

void test_cpu_affinity_comparison() {
    TEST_BEGIN("CPU Affinity - Comparison (With vs Without Binding)");

    int ncpus = num_cpus_online();
    printf("  System CPU cores: %d\n", ncpus);
    printf("  Launching %d AICPU threads (Thread 0-2: scheduler, Thread 3: orchestrator)\n",
           LAUNCH_AICPU_NUM);

#ifdef __linux__
    // ── Scenario A: Without binding ──────────────────────────────────────────
    printf("\n  --- Scenario A: Without Binding ---\n");
    ThreadReport free_reports[LAUNCH_AICPU_NUM];
    run_threads(thread_fn_no_bind, free_reports, LAUNCH_AICPU_NUM);

    printf("  %-12s %-15s %-14s\n", "Thread ID", "Role", "Actual Core");
    print_separator(44);
    for (int i = 0; i < LAUNCH_AICPU_NUM; i++) {
        const ThreadReport& r = free_reports[i];
        const char* role = (i == 3) ? "orchestrator" : "scheduler";
        printf("  Thread %-5d %-15s %-14d\n", i, role, r.actual_cpu);
    }

    // ── Scenario B: With binding (default) ───────────────────────────────────
    printf("\n  --- Scenario B: With Binding (Default) ---\n");
    int target_cores[LAUNCH_AICPU_NUM] = {1, 2, 3, 0};  // sched0→1, sched1→2, sched2→3, orch→0
    ThreadReport bound_reports[LAUNCH_AICPU_NUM];
    for (int i = 0; i < LAUNCH_AICPU_NUM; i++) {
        bound_reports[i].target_cpu = target_cores[i];
    }
    run_threads(thread_fn_with_bind, bound_reports, LAUNCH_AICPU_NUM);

    bool enough_cores = (ncpus >= 4);
    if (!enough_cores) {
        printf("\n  Warning: System has only %d cores, need at least 4 for this test\n", ncpus);
        printf("           Some threads may fail to bind to non-existent cores\n");
    }

    printf("  %-12s %-15s %-12s %-14s %-14s %-8s\n",
           "Thread ID", "Role", "Target Core", "Bound Core", "Actual Core", "Result");
    print_separator(82);

    int pass_count = 0;
    for (int i = 0; i < LAUNCH_AICPU_NUM; i++) {
        const ThreadReport& r = bound_reports[i];
        const char* role = (i == 3) ? "orchestrator" : "scheduler";
        bool correct = r.bind_ok
                    && (r.bound_cpu  == r.target_cpu)
                    && (r.actual_cpu == r.target_cpu);
        printf("  Thread %-5d %-15s %-12d %-14d %-14d %s\n",
               i, role, r.target_cpu, r.bound_cpu, r.actual_cpu,
               correct ? "OK" : "FAIL");
        if (correct) pass_count++;
        if (enough_cores) CHECK(correct);
    }

    // ── Summary Comparison ───────────────────────────────────────────────────
    printf("\n  --- Summary Comparison ---\n");
    printf("  %-12s %-15s %-22s %-22s\n",
           "Thread ID", "Role", "Scenario A (Actual)", "Scenario B (Actual)");
    print_separator(74);
    for (int i = 0; i < LAUNCH_AICPU_NUM; i++) {
        int a = free_reports[i].actual_cpu;
        int b = bound_reports[i].actual_cpu;
        printf("  Thread %-5d %-15s Core %-18d Core %-18d %s\n",
               i, (i == 3) ? "orchestrator" : "scheduler",
               a, b,
               (a != b) ? "<- Binding effective" : "(Same core, may be coincidence)");
    }

    printf("\n  Conclusion: %d/%d threads successfully bound\n",
           pass_count, enough_cores ? LAUNCH_AICPU_NUM : pass_count);

    TEST_END();
#endif
}

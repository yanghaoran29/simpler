/**
 * cpu_affinity.cpp
 *
 * CPU affinity utility function implementation and thread affinity test helpers.
 */

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#endif

#include <cstdio>
#include <cstdint>
#include "cpu_affinity.h"
#include "test_common.h"

void print_separator(int n) {
    printf("  ");
    for (int i = 0; i < n; i++) putchar('-');
    putchar('\n');
}

int num_cpus_online() {
#ifdef __linux__
    return (int)sysconf(_SC_NPROCESSORS_ONLN);
#else
    return 1;
#endif
}

int current_cpu() {
#ifdef __linux__
    return sched_getcpu();
#else
    return -1;
#endif
}

int bind_to_cpu(int cpu_core) {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_core, &cpuset);
    return pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#else
    (void)cpu_core;
    return -1;
#endif
}

int unbind_from_cpu() {
#ifdef __linux__
    int ncpus = num_cpus_online();
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int i = 0; i < ncpus; i++) CPU_SET(i, &cpuset);
    return pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#else
    return -1;
#endif
}

int get_bound_cpu() {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    if (pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
        return -1;
    }
    int count = 0;
    int first = -1;
    for (int i = 0; i < CPU_SETSIZE; i++) {
        if (CPU_ISSET(i, &cpuset)) {
            if (first < 0) first = i;
            count++;
        }
    }
    // If bound to all online cores, consider as "not specifically bound"
    if (count >= num_cpus_online()) return -1;
    return first;
#else
    return -1;
#endif
}

bool verify_cpu_binding(int target_cpu, const char* context) {
#ifdef __linux__
    int bound = get_bound_cpu();
    int actual = current_cpu();
    bool ok = (bound == target_cpu) && (actual == target_cpu);
    if (context && context[0] != '\0') {
        printf("  [%s] Binding verification: expected=%d, bound(read)=%d, actual(measured)=%d => %s\n",
               context, target_cpu, bound, actual, ok ? "OK" : "FAIL");
    } else {
        printf("  Binding verification: expected=%d, bound(read)=%d, actual(measured)=%d => %s\n",
               target_cpu, bound, actual, ok ? "OK" : "FAIL");
    }
    return ok;
#else
    (void)target_cpu;
    (void)context;
    printf("  Binding verification: Non-Linux platform, skipped\n");
    return true;
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// Thread affinity test helpers
// ─────────────────────────────────────────────────────────────────────────────

#ifdef __linux__
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
#endif  // __linux__

void run_threads(void* (*fn)(void*), ThreadReport* reports, int n) {
#ifdef __linux__
    pthread_t tids[LAUNCH_AICPU_NUM];
    for (int i = 0; i < n; i++) {
        reports[i].thread_idx = i;
        pthread_create(&tids[i], nullptr, fn, &reports[i]);
    }
    for (int i = 0; i < n; i++) {
        pthread_join(tids[i], nullptr);
    }
#else
    (void)fn; (void)reports; (void)n;
#endif
}

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

    CHECK(true);  // Observational only
    TEST_END();
#else
    (void)reports;
    printf("  Skipped: Not Linux platform\n");
    TEST_END();
#endif
}

void test_cpu_affinity_with_binding_default() {
    TEST_BEGIN("CPU Affinity - With Binding (Configured via ORCH_CPU / SCHED_CPUn)");

    int ncpus = num_cpus_online();
    printf("  System CPU cores: %d\n", ncpus);
    printf("  Launching %d AICPU threads (Thread 0-2: scheduler, Thread 3: orchestrator)\n",
           LAUNCH_AICPU_NUM);

    // Binding targets read from compile-time macros (cmake -DORCH_CPU=N -DSCHED_CPUn=N)
    int target_cores[LAUNCH_AICPU_NUM] = {SCHED_CPU0, SCHED_CPU1, SCHED_CPU2, ORCH_CPU};
    printf("  Configured: orch→core%d, sched[0]→core%d, sched[1]→core%d, sched[2]→core%d\n",
           ORCH_CPU, SCHED_CPU0, SCHED_CPU1, SCHED_CPU2);

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
    (void)target_cores; (void)reports;
    printf("  Skipped: Not Linux platform\n");
    TEST_END();
#endif
}

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

    // ── Scenario B: With binding (configured) ────────────────────────────────
    printf("\n  --- Scenario B: With Binding (Configured: orch→core%d, sched[0-2]→core%d,%d,%d) ---\n",
           ORCH_CPU, SCHED_CPU0, SCHED_CPU1, SCHED_CPU2);
    int target_cores[LAUNCH_AICPU_NUM] = {SCHED_CPU0, SCHED_CPU1, SCHED_CPU2, ORCH_CPU};
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

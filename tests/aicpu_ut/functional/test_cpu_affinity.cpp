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
 *
 * Test cases are implemented in common/cpu_affinity.cpp and can be reused
 * by any functional or performance test suite via #include "cpu_affinity.h".
 */

#include <cstdio>
#include "test_common.h"
#include "cpu_affinity.h"

int g_pass = 0;
int g_fail = 0;

int main() {
    printf("========================================\n");
    printf("CPU Affinity Functional Tests\n");
    printf("========================================\n");

    printf("\n[1/3] "); test_cpu_affinity_without_binding();
    printf("\n[2/3] "); test_cpu_affinity_with_binding_default();
    printf("\n[3/3] "); test_cpu_affinity_comparison();

    printf("\n========================================\n");
    printf("Summary: PASS=%d, FAIL=%d\n", g_pass, g_fail);
    printf("========================================\n");

    return (g_fail == 0) ? 0 : 1;
}

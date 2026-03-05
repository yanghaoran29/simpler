/**
 * main.cpp
 *
 * Orchestration Unit Test Entry
 *
 * Usage:
 *   ./test_orchestration           # run both functional and performance tests
 *   ./test_orchestration --func    # run functional tests only
 *   ./test_orchestration --perf    # run performance tests only
 *
 * Divided into two phases:
 *   1. Functional tests — verify CPU affinity behavior is correct, report PASS/FAIL
 *   2. Performance tests — measure Paged Attention graph build and simulation execution throughput
 */

#include <cstdio>
#include <cstring>

#include "test_common.h"

// Global counters (only for functional tests)
int g_pass = 0;
int g_fail = 0;

// ── Functional tests (tests/functional/test_cpu_affinity.cpp) ────────────────
extern void test_cpu_affinity_without_binding();
extern void test_cpu_affinity_with_binding_default();
extern void test_cpu_affinity_with_binding_custom();
extern void test_cpu_affinity_comparison();

// ── Performance tests (tests/perf/) ──────────────────────────────────────────
extern void test_paged_attention_basic();
extern void test_batch_paged_attention_basic();
extern void test_batch_paged_attention_chunked();
extern void test_batch_paged_attention_large_block_num_16();
extern void test_batch_paged_attention_large_block_num_128();
extern void test_batch_paged_attention_large_block_num_256();

static void run_functional_tests() {
    printf("\n----------------------------------------\n");
    printf("Functional Tests\n");
    printf("----------------------------------------\n");

    g_pass = 0;
    g_fail = 0;

    printf("\n[1/4] ");
    test_cpu_affinity_without_binding();

    printf("\n[2/4] ");
    test_cpu_affinity_with_binding_default();

    printf("\n[3/4] ");
    test_cpu_affinity_with_binding_custom();

    printf("\n[4/4] ");
    test_cpu_affinity_comparison();

    printf("\n========================================\n");
    printf("Functional Tests Summary: PASS=%d, FAIL=%d\n", g_pass, g_fail);
    printf("========================================\n");
}

static void run_performance_tests() {
    printf("\n----------------------------------------\n");
    printf("Performance Tests\n");
    printf("----------------------------------------\n");

    printf("\n[1/6] ");
    test_paged_attention_basic();

    printf("\n[2/6] ");
    test_batch_paged_attention_basic();

    printf("\n[3/6] ");
    test_batch_paged_attention_chunked();

    printf("\n[4/6] ");
    test_batch_paged_attention_large_block_num_16();

    printf("\n[5/6] ");
    test_batch_paged_attention_large_block_num_128();

    printf("\n[6/6] ");
    test_batch_paged_attention_large_block_num_256();

    printf("\n========================================\n");
    printf("Performance Tests Complete\n");
    printf("========================================\n");
}

int main(int argc, char* argv[]) {
    bool run_func = true;
    bool run_perf = true;

    if (argc >= 2) {
        if (strcmp(argv[1], "--func") == 0) {
            run_perf = false;
        } else if (strcmp(argv[1], "--perf") == 0) {
            run_func = false;
        }
    }

    printf("========================================\n");
    printf("Orchestration Unit Tests\n");
    printf("========================================\n");

    if (run_func) run_functional_tests();
    if (run_perf) run_performance_tests();

    return (g_fail == 0) ? 0 : 1;
}

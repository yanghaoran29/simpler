/**
 * test_main.cpp
 *
 * Unified test runner for orchestration unit tests
 */

#include "test_common.h"

// Define global test counters here
int g_pass = 0;
int g_fail = 0;

// Test functions from test_paged_attention.cpp
extern void test_paged_attention_basic();

// Test functions from test_batch_paged_attention.cpp
extern void test_batch_paged_attention_basic();
extern void test_batch_paged_attention_chunked();

int main() {
    printf("========================================\n");
    printf("Orchestration Unit Tests\n");
    printf("========================================\n");

    // Reset counters
    g_pass = 0;
    g_fail = 0;

    // Run paged attention tests
    printf("\n[1/3] Running Paged Attention Tests\n");
    test_paged_attention_basic();

    // Run batch paged attention tests
    printf("\n[2/3] Running Batch Paged Attention (Basic)\n");
    test_batch_paged_attention_basic();

    printf("\n[3/3] Running Batch Paged Attention (Chunked)\n");
    test_batch_paged_attention_chunked();

    // Final summary
    printf("\n========================================\n");
    printf("All Tests Summary: PASS=%d, FAIL=%d\n", g_pass, g_fail);
    printf("========================================\n");

    return (g_fail == 0) ? 0 : 1;
}

/**
 * test_platform_config.cpp
 *
 * Platform Configuration Validation Tests
 *
 * Verifies that platform configuration parameters are correctly set
 * through compile-time macros and that derived values are calculated properly.
 *
 * Tests validate:
 *   1. Base platform parameters (BLOCKDIM, cores per blockdim, AICPU threads)
 *   2. Derived limits (cores per thread, total cores)
 *   3. Consistency between base and derived values
 */

#include <cstdio>
#include "test_common.h"
#include "common/platform_config.h"

// Test 1: Verify base platform configuration parameters
void test_platform_base_config() {
    TEST_BEGIN("Platform Configuration - Base Parameters");

    printf("  PLATFORM_MAX_BLOCKDIM = %d\n", PLATFORM_MAX_BLOCKDIM);
    printf("  PLATFORM_AIC_CORES_PER_BLOCKDIM = %d\n", PLATFORM_AIC_CORES_PER_BLOCKDIM);
    printf("  PLATFORM_AIV_CORES_PER_BLOCKDIM = %d\n", PLATFORM_AIV_CORES_PER_BLOCKDIM);
    printf("  PLATFORM_MAX_AICPU_THREADS = %d\n", PLATFORM_MAX_AICPU_THREADS);

    // Validate that base parameters are positive
    CHECK(PLATFORM_MAX_BLOCKDIM > 0);
    CHECK(PLATFORM_AIC_CORES_PER_BLOCKDIM > 0);
    CHECK(PLATFORM_AIV_CORES_PER_BLOCKDIM > 0);
    CHECK(PLATFORM_MAX_AICPU_THREADS > 0);

    // Validate reasonable ranges
    CHECK(PLATFORM_MAX_BLOCKDIM <= 128);
    CHECK(PLATFORM_MAX_AICPU_THREADS <= 64);

    TEST_END();
}

// Test 2: Verify derived platform limits
void test_platform_derived_limits() {
    TEST_BEGIN("Platform Configuration - Derived Limits");

    printf("  PLATFORM_MAX_AIC_PER_THREAD = %d\n", PLATFORM_MAX_AIC_PER_THREAD);
    printf("  PLATFORM_MAX_AIV_PER_THREAD = %d\n", PLATFORM_MAX_AIV_PER_THREAD);
    printf("  PLATFORM_MAX_CORES_PER_THREAD = %d\n", PLATFORM_MAX_CORES_PER_THREAD);
    printf("  PLATFORM_MAX_CORES = %d\n", PLATFORM_MAX_CORES);

    // Verify derived calculations
    int expected_aic_per_thread = PLATFORM_MAX_BLOCKDIM * PLATFORM_AIC_CORES_PER_BLOCKDIM;
    int expected_aiv_per_thread = PLATFORM_MAX_BLOCKDIM * PLATFORM_AIV_CORES_PER_BLOCKDIM;
    int expected_cores_per_thread = expected_aic_per_thread + expected_aiv_per_thread;
    int expected_total_cores = PLATFORM_MAX_BLOCKDIM * PLATFORM_CORES_PER_BLOCKDIM;

    CHECK(PLATFORM_MAX_AIC_PER_THREAD == expected_aic_per_thread);
    CHECK(PLATFORM_MAX_AIV_PER_THREAD == expected_aiv_per_thread);
    CHECK(PLATFORM_MAX_CORES_PER_THREAD == expected_cores_per_thread);
    CHECK(PLATFORM_MAX_CORES == expected_total_cores);

    TEST_END();
}

// Test 3: Verify configuration consistency
void test_platform_config_consistency() {
    TEST_BEGIN("Platform Configuration - Consistency Check");

    // CORES_PER_BLOCKDIM should equal sum of AIC and AIV cores
    int expected_cores_per_blockdim = PLATFORM_AIC_CORES_PER_BLOCKDIM +
                                      PLATFORM_AIV_CORES_PER_BLOCKDIM;

    printf("  PLATFORM_CORES_PER_BLOCKDIM = %d (expected: %d)\n",
           PLATFORM_CORES_PER_BLOCKDIM, expected_cores_per_blockdim);

    CHECK(PLATFORM_CORES_PER_BLOCKDIM == expected_cores_per_blockdim);

    // MAX_CORES should be divisible by CORES_PER_BLOCKDIM
    CHECK(PLATFORM_MAX_CORES % PLATFORM_CORES_PER_BLOCKDIM == 0);

    // Verify relationship between per-thread and total cores
    int cores_from_per_thread = PLATFORM_MAX_AIC_PER_THREAD + PLATFORM_MAX_AIV_PER_THREAD;
    CHECK(cores_from_per_thread == PLATFORM_MAX_CORES);

    TEST_END();
}

// Test 4: Display complete configuration summary
void test_platform_config_summary() {
    TEST_BEGIN("Platform Configuration - Summary");

    printf("\n");
    printf("  ┌─────────────────────────────────────────────────────┐\n");
    printf("  │         Platform Configuration Summary             │\n");
    printf("  ├─────────────────────────────────────────────────────┤\n");
    printf("  │ Base Configuration:                                 │\n");
    printf("  │   MAX_BLOCKDIM              : %-4d                 │\n", PLATFORM_MAX_BLOCKDIM);
    printf("  │   AIC_CORES_PER_BLOCKDIM    : %-4d                 │\n", PLATFORM_AIC_CORES_PER_BLOCKDIM);
    printf("  │   AIV_CORES_PER_BLOCKDIM    : %-4d                 │\n", PLATFORM_AIV_CORES_PER_BLOCKDIM);
    printf("  │   CORES_PER_BLOCKDIM        : %-4d                 │\n", PLATFORM_CORES_PER_BLOCKDIM);
    printf("  │   MAX_AICPU_THREADS         : %-4d                 │\n", PLATFORM_MAX_AICPU_THREADS);
    printf("  ├─────────────────────────────────────────────────────┤\n");
    printf("  │ Derived Limits:                                     │\n");
    printf("  │   MAX_AIC_PER_THREAD        : %-4d                 │\n", PLATFORM_MAX_AIC_PER_THREAD);
    printf("  │   MAX_AIV_PER_THREAD        : %-4d                 │\n", PLATFORM_MAX_AIV_PER_THREAD);
    printf("  │   MAX_CORES_PER_THREAD      : %-4d                 │\n", PLATFORM_MAX_CORES_PER_THREAD);
    printf("  │   MAX_CORES                 : %-4d                 │\n", PLATFORM_MAX_CORES);
    printf("  ├─────────────────────────────────────────────────────┤\n");
    printf("  │ Calculated Ratios:                                  │\n");
    printf("  │   AIC/AIV ratio per block   : %d:%d                 │\n",
           PLATFORM_AIC_CORES_PER_BLOCKDIM, PLATFORM_AIV_CORES_PER_BLOCKDIM);
    printf("  │   Total AIC cores           : %-4d                 │\n",
           PLATFORM_MAX_BLOCKDIM * PLATFORM_AIC_CORES_PER_BLOCKDIM);
    printf("  │   Total AIV cores           : %-4d                 │\n",
           PLATFORM_MAX_BLOCKDIM * PLATFORM_AIV_CORES_PER_BLOCKDIM);
    printf("  └─────────────────────────────────────────────────────┘\n");
    printf("\n");

    TEST_END();
}

// Register all platform config tests
void register_platform_config_tests() {
    test_platform_base_config();
    test_platform_derived_limits();
    test_platform_config_consistency();
    test_platform_config_summary();
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

int g_pass = 0;
int g_fail = 0;

int main() {
    printf("========================================\n");
    printf("Platform Configuration Functional Tests\n");
    printf("========================================\n");

    register_platform_config_tests();

    printf("\n========================================\n");
    printf("Summary: PASS=%d, FAIL=%d\n", g_pass, g_fail);
    printf("========================================\n");

    return (g_fail == 0) ? 0 : 1;
}

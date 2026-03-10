/**
 * test_log_stubs.cpp
 *
 * Unified log function stubs for orchestration unit tests.
 * When PTO2_SIM_AICORE_UT: also provides device_log (dev_log_*) stubs for aicpu_executor.
 */

#include <cstdarg>
#include <cstdio>

#if defined(PTO2_SIM_AICORE_UT)
#include "aicpu/device_log.h"
bool g_is_log_enable_debug = false;
bool g_is_log_enable_info = false;
bool g_is_log_enable_warn = false;
bool g_is_log_enable_error = true;
const char* TILE_FWK_DEVICE_MACHINE = "ut";

void dev_log_debug(const char* /* func */, const char* /* fmt */, ...) {}
void dev_log_info(const char* /* func */, const char* /* fmt */, ...) {}
void dev_log_warn(const char* func, const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    fprintf(stderr, "[WARN] %s: ", func); vfprintf(stderr, fmt, ap); fputc('\n', stderr);
    va_end(ap);
}
void dev_log_error(const char* func, const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    fprintf(stderr, "[ERR]  %s: ", func); vfprintf(stderr, fmt, ap); fputc('\n', stderr);
    va_end(ap);
}
void dev_log_always(const char* /* func */, const char* /* fmt */, ...) {}
#endif

extern "C" {

void unified_log_error(const char* func, const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    fprintf(stderr, "[ERR]  %s: ", func); vfprintf(stderr, fmt, ap); fputc('\n', stderr);
    va_end(ap);
}

void unified_log_warn(const char* func, const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    fprintf(stderr, "[WARN] %s: ", func); vfprintf(stderr, fmt, ap); fputc('\n', stderr);
    va_end(ap);
}

void unified_log_info([[maybe_unused]] const char* func,
                      [[maybe_unused]] const char* fmt, ...) {
    // Silence INFO during tests
}

void unified_log_debug([[maybe_unused]] const char* func,
                       [[maybe_unused]] const char* fmt, ...) {}

void unified_log_always(const char* func, const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    printf("[ALWY] %s: ", func); vprintf(fmt, ap); putchar('\n');
    va_end(ap);
}

}  // extern "C"

// Stub for AICPU orchestrator phase profiling (no-op in unit test context,
// which has no shared memory profiling buffer).
#include "common/perf_profiling.h"
void perf_aicpu_record_orch_phase(AicpuPhaseId, uint64_t, uint64_t, uint32_t) {}
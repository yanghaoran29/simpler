/**
 * test_log_stubs.cpp
 *
 * Unified log function stubs for orchestration unit tests.
 * When PTO2_SIM_AICORE_UT: also provides device_log (dev_log_*) stubs for aicpu_executor.
 * When AICPU_UT_PHASE_LOG is set, dev_log_always also appends each line to that file
 * (Thread release, PTO2 progress, Scheduler Phase Breakdown, etc.).
 */

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#if defined(PTO2_SIM_AICORE_UT)
#include "aicpu/device_log.h"
bool g_is_log_enable_debug = false;
bool g_is_log_enable_info = false;
bool g_is_log_enable_warn = false;
bool g_is_log_enable_error = true;
const char* TILE_FWK_DEVICE_MACHINE = "ut";

static std::mutex g_dev_log_always_mutex;

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
void dev_log_always(const char* /* func */, const char* fmt, ...) {
#if !PTO2_PROFILING
    (void)fmt;
    return;
#endif
    va_list ap;
    va_start(ap, fmt);
    std::lock_guard<std::mutex> lock(g_dev_log_always_mutex);
    char buf[2048];
    int n = std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n <= 0) return;
    printf("%s\n", buf);
    fflush(stdout);
    const char* phase_log = std::getenv("AICPU_UT_PHASE_LOG");
    if (phase_log && *phase_log) {
        FILE* f = std::fopen(phase_log, "a");
        if (f) {
            std::fprintf(f, "%s\n", buf);
            std::fclose(f);
        }
    }
}
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

void unified_log_always(const char* /* func */, const char* fmt, ...) {
#if !PTO2_PROFILING
    (void)fmt;
    return;
#endif
    va_list ap; va_start(ap, fmt);
    vprintf(fmt, ap); putchar('\n');
    va_end(ap);
}

}  // extern "C"

// Stub for AICPU orchestrator phase profiling (no-op in unit test context,
// which has no shared memory profiling buffer).
#include "common/perf_profiling.h"
void perf_aicpu_record_orch_phase(AicpuPhaseId, uint64_t, uint64_t, uint32_t) {}
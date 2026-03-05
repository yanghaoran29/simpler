/**
 * test_log_stubs.cpp
 *
 * Unified log function stubs for orchestration unit tests
 */

#include <cstdarg>
#include <cstdio>

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
    // 测试期间静默 INFO
}

void unified_log_debug([[maybe_unused]] const char* func,
                       [[maybe_unused]] const char* fmt, ...) {}

void unified_log_always(const char* func, const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    printf("[ALWY] %s: ", func); vprintf(fmt, ap); putchar('\n');
    va_end(ap);
}

}  // extern "C"

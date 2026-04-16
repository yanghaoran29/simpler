/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
/**
 * Device logging implementation for AICPU kernel
 */

#include "aicpu/device_log.h"
#include "dlog_pub.h"  // CANN dlog API
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

bool g_is_log_enable_debug = false;
bool g_is_log_enable_info = false;
bool g_is_log_enable_warn = false;
bool g_is_log_enable_error = false;

const char *TILE_FWK_DEVICE_MACHINE = "AI_CPU";

static bool g_dev_always_mirror_to_stderr = false;

void dev_log_set_always_mirror_to_stderr(bool enable) { g_dev_always_mirror_to_stderr = enable; }

void init_log_switch() {
    g_is_log_enable_debug = CheckLogLevel(AICPU, DLOG_DEBUG);
    g_is_log_enable_info = CheckLogLevel(AICPU, DLOG_INFO);
    g_is_log_enable_warn = CheckLogLevel(AICPU, DLOG_WARN);
    g_is_log_enable_error = CheckLogLevel(AICPU, DLOG_ERROR);
}

// =============================================================================
// Platform-Specific Logging Functions (Real Hardware: use CANN dlog API)
// =============================================================================

void dev_log_debug(const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char buffer[2048];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    // Add quotes around message to match original behavior (#fmt stringification)
    dlog_debug(AICPU, "%lu %s\n\"%s\"", GET_TID(), func, buffer);
}

void dev_log_info(const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char buffer[2048];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    dlog_info(AICPU, "%lu %s\n\"%s\"", GET_TID(), func, buffer);
}

void dev_log_warn(const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char buffer[2048];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    dlog_warn(AICPU, "%lu %s\n\"%s\"", GET_TID(), func, buffer);
}

void dev_log_error(const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char buffer[2048];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    dlog_error(AICPU, "%lu %s\n\"%s\"", GET_TID(), func, buffer);
}

void dev_log_always(const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char buffer[2048];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    dlog_error(AICPU, "[ALWAYS] %lu %s\n\"%s\"", GET_TID(), func, buffer);
    // 设备进程无主机环境变量：由 kernel 入口 dev_log_set_always_mirror_to_stderr(runtime->enable_profiling) 打开。
    const char *mir = std::getenv("PTO_AICPU_ALWAYS_STDERR");
    const bool mirror_env =
        mir != nullptr && mir[0] != '\0' && std::strcmp(mir, "0") != 0;
    if (g_dev_always_mirror_to_stderr || mirror_env) {
        std::fputs("[ALWAYS] ", stderr);
        std::fputs(func, stderr);
        std::fputs(": ", stderr);
        std::fputs(buffer, stderr);
        std::fputc('\n', stderr);
        std::fflush(stderr);
    }
}

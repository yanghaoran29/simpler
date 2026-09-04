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

#pragma once

#include <cstdio>
#include <cstring>

#include "runtime_c_api.h"

inline bool native_run_is_prewarm_dry_run(uint32_t flags) {
    return (flags & PTO_NATIVE_RUN_FLAG_PREWARM_DRY_RUN) != 0;
}

inline bool native_run_is_prewarm_dry_run(const NativeRunDescriptor &descriptor) {
    return native_run_is_prewarm_dry_run(descriptor.flags);
}

/** Map chip.run / chip.run.* onto chip.prewarm.run / chip.prewarm.* for dry-run. */
inline const char *native_run_span_name(bool prewarm, const char *name) {
    if (!prewarm || name == nullptr) return name;
    static constexpr char kRun[] = "chip.run";
    static constexpr size_t kRunLen = sizeof(kRun) - 1;
    if (std::strncmp(name, kRun, kRunLen) != 0) return name;
    // Nested STRACE scopes keep the pointer, so one buffer would rename the
    // outer span when a child calls this helper.
    thread_local char bufs[8][160];
    thread_local unsigned idx = 0;
    char *buf = bufs[idx++ & 7u];
    if (name[kRunLen] == '\0') {
        std::snprintf(buf, 160, "chip.prewarm.run");
        return buf;
    }
    if (name[kRunLen] == '.') {
        std::snprintf(buf, 160, "chip.prewarm%s", name + kRunLen);
        return buf;
    }
    return name;
}

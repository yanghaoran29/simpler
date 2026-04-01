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

/**
 * QEMU marker NOPs: `orr xR, xR, xR` (distinct R per phase) as recognizable signatures.
 *
 * PTO2_SPECIAL_INSTRUCTION_PLAIN(reg, flag)  — emit orr xR,xR,xR  when flag is non-zero.
 * PTO2_SPECIAL_INSTRUCTION_MEMORY(reg, flag) — same with "memory" clobber (compiler fence).
 *
 * flag is evaluated at compile time; zero → entire call is a no-op via DCE.
 * The #if defined(__aarch64__) guard is here so call sites need no arch checks.
 */
#ifndef PTO2_INSTR_COUNT_BUILD_GRAPH_ENABLE
#define PTO2_INSTR_COUNT_BUILD_GRAPH_ENABLE 0
#endif

/* Default ON: matches legacy `PTO2_SPECIAL_INSTRUCTION_PLAIN(..., 1)` in orchestrator so
 * QEMU insn_count / count_between_markers.sh (orr x3/x4 around pto2_submit_mixed_task) keeps working.
 * To strip markers, compile with -DPTO2_INSTR_COUNT_ORCHESTRATOR_ENABLE=0. */
#ifndef PTO2_INSTR_COUNT_ORCHESTRATOR_ENABLE
#define PTO2_INSTR_COUNT_ORCHESTRATOR_ENABLE 1
#endif

#ifndef PTO2_INSTR_COUNT_SCHEDULER_ENABLE
#if defined(PTO2_INSTR_COUNT_SCHEDULER_LOOP_ENABLE)
#define PTO2_INSTR_COUNT_SCHEDULER_ENABLE PTO2_INSTR_COUNT_SCHEDULER_LOOP_ENABLE
#else
#define PTO2_INSTR_COUNT_SCHEDULER_ENABLE 0
#endif
#endif

#ifndef PTO2_INSTR_COUNT_SCHEDULER_LOOP_ENABLE
#define PTO2_INSTR_COUNT_SCHEDULER_LOOP_ENABLE PTO2_INSTR_COUNT_SCHEDULER_ENABLE
#endif

#if defined(__aarch64__)
#define _PTO2_MARKER_ORR_SELF_ASM(n) "orr x" #n ", x" #n ", x" #n
#define PTO2_SPECIAL_INSTRUCTION_PLAIN(n, flag)                 \
    do {                                                        \
        if constexpr (flag) {                                   \
            __asm__ __volatile__(_PTO2_MARKER_ORR_SELF_ASM(n)); \
        }                                                       \
    } while (0)
#define PTO2_SPECIAL_INSTRUCTION_MEMORY(n, flag)                             \
    do {                                                                     \
        if constexpr (flag) {                                                \
            __asm__ __volatile__(_PTO2_MARKER_ORR_SELF_ASM(n)::: "memory"); \
        }                                                                    \
    } while (0)
#else
#define PTO2_SPECIAL_INSTRUCTION_PLAIN(n, flag)  ((void)(flag))
#define PTO2_SPECIAL_INSTRUCTION_MEMORY(n, flag) ((void)(flag))
#endif

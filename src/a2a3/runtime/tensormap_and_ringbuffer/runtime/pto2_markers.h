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
 * QEMU marker signatures (distinct encodings by numeric id).
 *
 * Numeric mapping for PTO2_SPECIAL_INSTRUCTION_* (n):
 * - 1..30  -> emit `orr xN, xN, xN`
 * - 31..60 -> emit `and x(N-30), x(N-30), x(N-30)` (e.g. 31 -> and x1,x1,x1)
 *
 * PTO2_SPECIAL_INSTRUCTION_PLAIN(n, flag)  — emit marker when flag is non-zero.
 * PTO2_SPECIAL_INSTRUCTION_MEMORY(n, flag) — same with "memory" clobber.
 *
 * flag is evaluated at compile time; zero -> no-op via DCE.
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
#define _PTO2_MARKER_AND_SELF_ASM(n) "and x" #n ", x" #n ", x" #n
#define _PTO2_MARKER_ORR_CASE_PLAIN(n) case n: __asm__ __volatile__(_PTO2_MARKER_ORR_SELF_ASM(n)); break
#define _PTO2_MARKER_ORR_CASE_MEM(n) case n: __asm__ __volatile__(_PTO2_MARKER_ORR_SELF_ASM(n) ::: "memory"); break
#define _PTO2_MARKER_AND_CASE_PLAIN(i, r) case i: __asm__ __volatile__(_PTO2_MARKER_AND_SELF_ASM(r)); break
#define _PTO2_MARKER_AND_CASE_MEM(i, r) case i: __asm__ __volatile__(_PTO2_MARKER_AND_SELF_ASM(r) ::: "memory"); break
#define _PTO2_EMIT_MARKER_PLAIN(n)                       \
    do {                                                 \
        switch (n) {                                     \
            _PTO2_MARKER_ORR_CASE_PLAIN(1);             \
            _PTO2_MARKER_ORR_CASE_PLAIN(2);             \
            _PTO2_MARKER_ORR_CASE_PLAIN(3);             \
            _PTO2_MARKER_ORR_CASE_PLAIN(4);             \
            _PTO2_MARKER_ORR_CASE_PLAIN(5);             \
            _PTO2_MARKER_ORR_CASE_PLAIN(6);             \
            _PTO2_MARKER_ORR_CASE_PLAIN(7);             \
            _PTO2_MARKER_ORR_CASE_PLAIN(8);             \
            _PTO2_MARKER_ORR_CASE_PLAIN(9);             \
            _PTO2_MARKER_ORR_CASE_PLAIN(10);            \
            _PTO2_MARKER_ORR_CASE_PLAIN(11);            \
            _PTO2_MARKER_ORR_CASE_PLAIN(12);            \
            _PTO2_MARKER_ORR_CASE_PLAIN(13);            \
            _PTO2_MARKER_ORR_CASE_PLAIN(14);            \
            _PTO2_MARKER_ORR_CASE_PLAIN(15);            \
            _PTO2_MARKER_ORR_CASE_PLAIN(16);            \
            _PTO2_MARKER_ORR_CASE_PLAIN(17);            \
            _PTO2_MARKER_ORR_CASE_PLAIN(18);            \
            _PTO2_MARKER_ORR_CASE_PLAIN(19);            \
            _PTO2_MARKER_ORR_CASE_PLAIN(20);            \
            _PTO2_MARKER_ORR_CASE_PLAIN(21);            \
            _PTO2_MARKER_ORR_CASE_PLAIN(22);            \
            _PTO2_MARKER_ORR_CASE_PLAIN(23);            \
            _PTO2_MARKER_ORR_CASE_PLAIN(24);            \
            _PTO2_MARKER_ORR_CASE_PLAIN(25);            \
            _PTO2_MARKER_ORR_CASE_PLAIN(26);            \
            _PTO2_MARKER_ORR_CASE_PLAIN(27);            \
            _PTO2_MARKER_ORR_CASE_PLAIN(28);            \
            _PTO2_MARKER_ORR_CASE_PLAIN(29);            \
            _PTO2_MARKER_ORR_CASE_PLAIN(30);            \
            _PTO2_MARKER_AND_CASE_PLAIN(31, 1);         \
            _PTO2_MARKER_AND_CASE_PLAIN(32, 2);         \
            _PTO2_MARKER_AND_CASE_PLAIN(33, 3);         \
            _PTO2_MARKER_AND_CASE_PLAIN(34, 4);         \
            _PTO2_MARKER_AND_CASE_PLAIN(35, 5);         \
            _PTO2_MARKER_AND_CASE_PLAIN(36, 6);         \
            _PTO2_MARKER_AND_CASE_PLAIN(37, 7);         \
            _PTO2_MARKER_AND_CASE_PLAIN(38, 8);         \
            _PTO2_MARKER_AND_CASE_PLAIN(39, 9);         \
            _PTO2_MARKER_AND_CASE_PLAIN(40, 10);        \
            _PTO2_MARKER_AND_CASE_PLAIN(41, 11);        \
            _PTO2_MARKER_AND_CASE_PLAIN(42, 12);        \
            _PTO2_MARKER_AND_CASE_PLAIN(43, 13);        \
            _PTO2_MARKER_AND_CASE_PLAIN(44, 14);        \
            _PTO2_MARKER_AND_CASE_PLAIN(45, 15);        \
            _PTO2_MARKER_AND_CASE_PLAIN(46, 16);        \
            _PTO2_MARKER_AND_CASE_PLAIN(47, 17);        \
            _PTO2_MARKER_AND_CASE_PLAIN(48, 18);        \
            _PTO2_MARKER_AND_CASE_PLAIN(49, 19);        \
            _PTO2_MARKER_AND_CASE_PLAIN(50, 20);        \
            _PTO2_MARKER_AND_CASE_PLAIN(51, 21);        \
            _PTO2_MARKER_AND_CASE_PLAIN(52, 22);        \
            _PTO2_MARKER_AND_CASE_PLAIN(53, 23);        \
            _PTO2_MARKER_AND_CASE_PLAIN(54, 24);        \
            _PTO2_MARKER_AND_CASE_PLAIN(55, 25);        \
            _PTO2_MARKER_AND_CASE_PLAIN(56, 26);        \
            _PTO2_MARKER_AND_CASE_PLAIN(57, 27);        \
            _PTO2_MARKER_AND_CASE_PLAIN(58, 28);        \
            _PTO2_MARKER_AND_CASE_PLAIN(59, 29);        \
            _PTO2_MARKER_AND_CASE_PLAIN(60, 30);        \
            default: break;                              \
        }                                                \
    } while (0)
#define _PTO2_EMIT_MARKER_MEMORY(n)                      \
    do {                                                 \
        switch (n) {                                     \
            _PTO2_MARKER_ORR_CASE_MEM(1);               \
            _PTO2_MARKER_ORR_CASE_MEM(2);               \
            _PTO2_MARKER_ORR_CASE_MEM(3);               \
            _PTO2_MARKER_ORR_CASE_MEM(4);               \
            _PTO2_MARKER_ORR_CASE_MEM(5);               \
            _PTO2_MARKER_ORR_CASE_MEM(6);               \
            _PTO2_MARKER_ORR_CASE_MEM(7);               \
            _PTO2_MARKER_ORR_CASE_MEM(8);               \
            _PTO2_MARKER_ORR_CASE_MEM(9);               \
            _PTO2_MARKER_ORR_CASE_MEM(10);              \
            _PTO2_MARKER_ORR_CASE_MEM(11);              \
            _PTO2_MARKER_ORR_CASE_MEM(12);              \
            _PTO2_MARKER_ORR_CASE_MEM(13);              \
            _PTO2_MARKER_ORR_CASE_MEM(14);              \
            _PTO2_MARKER_ORR_CASE_MEM(15);              \
            _PTO2_MARKER_ORR_CASE_MEM(16);              \
            _PTO2_MARKER_ORR_CASE_MEM(17);              \
            _PTO2_MARKER_ORR_CASE_MEM(18);              \
            _PTO2_MARKER_ORR_CASE_MEM(19);              \
            _PTO2_MARKER_ORR_CASE_MEM(20);              \
            _PTO2_MARKER_ORR_CASE_MEM(21);              \
            _PTO2_MARKER_ORR_CASE_MEM(22);              \
            _PTO2_MARKER_ORR_CASE_MEM(23);              \
            _PTO2_MARKER_ORR_CASE_MEM(24);              \
            _PTO2_MARKER_ORR_CASE_MEM(25);              \
            _PTO2_MARKER_ORR_CASE_MEM(26);              \
            _PTO2_MARKER_ORR_CASE_MEM(27);              \
            _PTO2_MARKER_ORR_CASE_MEM(28);              \
            _PTO2_MARKER_ORR_CASE_MEM(29);              \
            _PTO2_MARKER_ORR_CASE_MEM(30);              \
            _PTO2_MARKER_AND_CASE_MEM(31, 1);           \
            _PTO2_MARKER_AND_CASE_MEM(32, 2);           \
            _PTO2_MARKER_AND_CASE_MEM(33, 3);           \
            _PTO2_MARKER_AND_CASE_MEM(34, 4);           \
            _PTO2_MARKER_AND_CASE_MEM(35, 5);           \
            _PTO2_MARKER_AND_CASE_MEM(36, 6);           \
            _PTO2_MARKER_AND_CASE_MEM(37, 7);           \
            _PTO2_MARKER_AND_CASE_MEM(38, 8);           \
            _PTO2_MARKER_AND_CASE_MEM(39, 9);           \
            _PTO2_MARKER_AND_CASE_MEM(40, 10);          \
            _PTO2_MARKER_AND_CASE_MEM(41, 11);          \
            _PTO2_MARKER_AND_CASE_MEM(42, 12);          \
            _PTO2_MARKER_AND_CASE_MEM(43, 13);          \
            _PTO2_MARKER_AND_CASE_MEM(44, 14);          \
            _PTO2_MARKER_AND_CASE_MEM(45, 15);          \
            _PTO2_MARKER_AND_CASE_MEM(46, 16);          \
            _PTO2_MARKER_AND_CASE_MEM(47, 17);          \
            _PTO2_MARKER_AND_CASE_MEM(48, 18);          \
            _PTO2_MARKER_AND_CASE_MEM(49, 19);          \
            _PTO2_MARKER_AND_CASE_MEM(50, 20);          \
            _PTO2_MARKER_AND_CASE_MEM(51, 21);          \
            _PTO2_MARKER_AND_CASE_MEM(52, 22);          \
            _PTO2_MARKER_AND_CASE_MEM(53, 23);          \
            _PTO2_MARKER_AND_CASE_MEM(54, 24);          \
            _PTO2_MARKER_AND_CASE_MEM(55, 25);          \
            _PTO2_MARKER_AND_CASE_MEM(56, 26);          \
            _PTO2_MARKER_AND_CASE_MEM(57, 27);          \
            _PTO2_MARKER_AND_CASE_MEM(58, 28);          \
            _PTO2_MARKER_AND_CASE_MEM(59, 29);          \
            _PTO2_MARKER_AND_CASE_MEM(60, 30);          \
            default: break;                              \
        }                                                \
    } while (0)
#define PTO2_SPECIAL_INSTRUCTION_PLAIN(n, flag) \
    do { if constexpr (flag) { _PTO2_EMIT_MARKER_PLAIN(n); } } while (0)
#define PTO2_SPECIAL_INSTRUCTION_MEMORY(n, flag) \
    do { if constexpr (flag) { _PTO2_EMIT_MARKER_MEMORY(n); } } while (0)
#else
#define PTO2_SPECIAL_INSTRUCTION_PLAIN(n, flag)  ((void)(flag))
#define PTO2_SPECIAL_INSTRUCTION_MEMORY(n, flag) ((void)(flag))
#endif

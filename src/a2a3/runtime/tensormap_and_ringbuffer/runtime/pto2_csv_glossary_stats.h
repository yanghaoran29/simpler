/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

/**
 * module-struct-access.csv 第 1–9 行注释中的符号（P/C/S/N、N_in、N_out、N_scope、tensor_count、scalar_count）
 * 在编排侧按「一次成功 submit」采样，按键合并计数，供 flush 日志与 CSV 五列对照。
 */

#pragma once

#include <stdint.h>

#ifndef PTO2_CSV_GLOSSARY_BUCKET_MAX
#define PTO2_CSV_GLOSSARY_BUCKET_MAX 32
#endif

/** 与 CSV 注释对齐的一条「任务形状」键（memcmp 判等；packed 避免隐式填充） */
struct __attribute__((packed)) PTO2CsvGlossaryTaskKindKey {
    int32_t kernel_aic;
    int32_t kernel_aiv0;
    int32_t kernel_aiv1;
    uint8_t active_mask; /**< 子任务位 + require_sync_start 等（见 MixedKernels） */
    uint8_t ring_id;
    int8_t kind_tag; /**< 0 = pto2_submit_mixed_task；1 = pto2_alloc_tensors（无 InCore） */
    int16_t scope_depth; /**< submit 时 scope 栈深度+1，作 N_scope 的近似（非严格「scope_end 配对次数」） */
    int32_t P_fanin_producers; /**< CSV P：本 consumer 在编排侧挂接的 producer 条数（fanin 条数） */
    int32_t C_fanout_minus_scope; /**< CSV C 近似：fanout_count-1（编排刚结束时多为 0，直至其它任务依赖本输出） */
    int32_t S_subtasks; /**< CSV S：total_required_subtasks */
    int32_t N_ring_acquire_proxy; /**< CSV N：单任务路径未拆分时置 0；全 run 见 RingFlowControl acquire 汇总于 CSV ①③ r= */
    int16_t N_in; /**< CSV N_in：INPUT + INOUT 计输入侧 */
    int16_t N_out; /**< CSV N_out：OUTPUT/OUTPUT_EXISTING + INOUT 计输出侧 */
    int16_t tensor_count; /**< CSV tensor_count */
    int16_t scalar_count; /**< CSV scalar_count */
};

struct PTO2CsvGlossaryTaskKindBucket {
    PTO2CsvGlossaryTaskKindKey k;
    uint32_t submit_count; /**< 该键上成功 submit 次数 */
};

struct PTO2CsvGlossaryStats {
    uint32_t bucket_count;
    PTO2CsvGlossaryTaskKindBucket buckets[PTO2_CSV_GLOSSARY_BUCKET_MAX];
};

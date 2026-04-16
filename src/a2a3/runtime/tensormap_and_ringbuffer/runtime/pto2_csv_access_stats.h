/*
 * Copyright (c) PyPTO Contributors.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/**
 * 与 module-struct-access.csv 中计数口径对齐的一组计数器。
 * 基础列为「读次数、写次数、atomic操作次数、锁次数、CAS次数」；另扩展 atomic 读/写拆分列。
 * 各字段表示运行期观测到的事件累计次数（或文档约定的近似）；不含 cacheline(B)，后者由离线用 CSV「每次读/写」系数换算。
 */
#pragma once

#include <cstdint>

struct PTO2CsvAccessCounters {
    uint64_t read_events;   // CSV「读次数」：本行数据结构维度上发生的读类事件次数（含 load/遍历一次算一次等口径见汇总处注释）
    uint64_t write_events;  // CSV「写次数」：写类事件次数
    uint64_t atomic_ops;    // CSV「atomic操作次数」：fetch_add / RMW / 队列原子序列等按实现约定累计
    uint64_t atomic_read_ops;   // atomic读次数：load/轮询等原子读事件（代码语义拆分）
    uint64_t atomic_write_ops;  // atomic写次数：store/fetch_add/CAS成功写等原子写事件（代码语义拆分）
    uint64_t lock_ops;      // CSV「锁次数」：显式锁 acquire/release 或等价互斥次数
    uint64_t cas_ops;       // CSV「CAS次数」：compare_exchange 成功或失败尝试次数（以汇总处注释为准）
};

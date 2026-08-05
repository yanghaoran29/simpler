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

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#if __has_include(<driver/ascend_hal_base.h>)
#include <driver/ascend_hal_base.h>
#else
// Pure host unit tests also build on machines without CANN headers. Keep the
// compatibility names identical to the public driver enum names; production
// builds take the definitions above directly from ascend_hal_base.h.
enum : int32_t {
    MODULE_TYPE_SYSTEM = 0,
    MODULE_TYPE_AICPU = 1,
    INFO_TYPE_OS_SCHED = 5,
    INFO_TYPE_OCCUPY = 8,
    INFO_TYPE_PF_OCCUPY = 21,
};
#endif

#if __has_include(<driver/dsmi_common_interface.h>)
#include <driver/dsmi_common_interface.h>
#else
constexpr unsigned int DSMI_MAIN_CMD_SOC_INFO = 14;
#endif

namespace pto::a5::driver_abi {

// CPU_TOPO is an A5 driver extension not declared by currently installed
// public CANN headers. Values match the query reference tool and CANN 9.1.
constexpr int32_t kHalInfoCpuTopology = 59;
constexpr unsigned int kDsmiCpuTopologySubcommand = 2;

// CPU ids are encoded in the same uint64_t bitmap returned by OCCUPY.
constexpr size_t kAicpuCpuMaskBits = std::numeric_limits<uint64_t>::digits;

// Natural-alignment wire layout expected by HAL/DSMI. Packing this structure
// changes its size and causes the driver to reject the query buffer.
struct DsmiSingleCpu {
    uint64_t cpu_mask;
    uint8_t cpu_id;
    uint8_t is_share;
    uint8_t phy_cpu_id;
    uint8_t hyperthread_id;
};

struct DsmiCpuTopology {
    uint32_t total_nums;
    DsmiSingleCpu cpus[kAicpuCpuMaskBits];
};

static_assert(
    std::is_trivially_copyable_v<DsmiSingleCpu> && std::is_standard_layout_v<DsmiSingleCpu>,
    "DsmiSingleCpu must remain a memcpy-safe driver wire type"
);
static_assert(sizeof(DsmiSingleCpu) == 16, "DsmiSingleCpu ABI size drift");
static_assert(offsetof(DsmiSingleCpu, cpu_id) == 8, "DsmiSingleCpu::cpu_id offset drift");
static_assert(offsetof(DsmiSingleCpu, hyperthread_id) == 11, "DsmiSingleCpu::hyperthread_id offset drift");
static_assert(
    std::is_trivially_copyable_v<DsmiCpuTopology> && std::is_standard_layout_v<DsmiCpuTopology>,
    "DsmiCpuTopology must remain a memcpy-safe driver wire type"
);
static_assert(offsetof(DsmiCpuTopology, cpus) == 8, "DsmiCpuTopology::cpus offset drift");
static_assert(
    sizeof(DsmiCpuTopology) == 8 + kAicpuCpuMaskBits * sizeof(DsmiSingleCpu), "DsmiCpuTopology ABI size drift"
);

}  // namespace pto::a5::driver_abi

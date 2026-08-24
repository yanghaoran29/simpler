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

#include <acl/acl.h>
#include <driver/ascend_hal.h>
#include <driver/ascend_hal_define.h>
#include <runtime/rt.h>
#include <runtime/runtime/rts/rts_kernel.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include "../shared/rtt_types.h"

namespace fingerprint {

// Extract a stable GNU build ID when the input is a supported ELF image.
bool ElfBuildId(const char *data, size_t len, uint64_t *output) {
    if (len < 64 ||
        std::memcmp(
            data,
            "\x7f"
            "ELF",
            4
        ) != 0 ||
        data[4] != 2) {
        return false;
    }
    uint64_t section_offset = 0;
    uint16_t section_entry_size = 0;
    uint16_t section_count = 0;
    uint16_t string_index = 0;
    std::memcpy(&section_offset, data + 40, 8);
    std::memcpy(&section_entry_size, data + 58, 2);
    std::memcpy(&section_count, data + 60, 2);
    std::memcpy(&string_index, data + 62, 2);
    if (section_entry_size != 64 || section_offset > len ||
        static_cast<uint64_t>(section_entry_size) * section_count > len - section_offset ||
        string_index >= section_count) {
        return false;
    }

    const char *string_table = nullptr;
    uint64_t string_table_size = 0;
    {
        const char *section = data + section_offset + static_cast<uint64_t>(section_entry_size) * string_index;
        uint64_t offset = 0;
        std::memcpy(&offset, section + 24, 8);
        std::memcpy(&string_table_size, section + 32, 8);
        if (offset > len || string_table_size > len - offset) {
            return false;
        }
        string_table = data + offset;
    }

    for (uint16_t i = 0; i < section_count; ++i) {
        const char *section = data + section_offset + static_cast<uint64_t>(section_entry_size) * i;
        uint32_t name_offset = 0;
        uint32_t section_type = 0;
        uint64_t offset = 0;
        uint64_t size = 0;
        std::memcpy(&name_offset, section, 4);
        std::memcpy(&section_type, section + 4, 4);
        std::memcpy(&offset, section + 24, 8);
        std::memcpy(&size, section + 32, 8);
        if (section_type != 7 || name_offset >= string_table_size) {
            continue;
        }
        const char *name = string_table + name_offset;
        const size_t maximum_name_size = string_table_size - name_offset;
        const void *name_end = std::memchr(name, '\0', maximum_name_size);
        if (name_end == nullptr || std::string_view(name) != ".note.gnu.build-id" || size < 16 || offset > len ||
            size > len - offset) {
            continue;
        }
        const char *note = data + offset;
        uint32_t name_size = 0;
        uint32_t descriptor_size = 0;
        uint32_t type = 0;
        std::memcpy(&name_size, note, 4);
        std::memcpy(&descriptor_size, note + 4, 4);
        std::memcpy(&type, note + 8, 4);
        const size_t aligned_name_size = (name_size + 3u) & ~3u;
        if (type != 3 || descriptor_size < 8 || aligned_name_size > size - 12 || size - 12 - aligned_name_size < 8) {
            return false;
        }
        std::memcpy(output, note + 12 + aligned_name_size, 8);
        return true;
    }
    return false;
}

// Compute a stable binary fingerprint, falling back to FNV-1a for non-ELF input.
uint64_t Compute(const char *data, size_t size) {
    uint64_t output = 0;
    if (ElfBuildId(data, size, &output)) {
        return output;
    }
    output = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < size; ++i) {
        output ^= static_cast<unsigned char>(data[i]);
        output *= 0x100000001b3ULL;
    }
    return output;
}

}  // namespace fingerprint

namespace {

#define ACL_CHECK(call, description)                                                    \
    do {                                                                                \
        const aclError check_rc = (call);                                               \
        if (check_rc != ACL_SUCCESS) {                                                  \
            std::fprintf(stderr, "%s failed: %d (%s)\n", #call, check_rc, description); \
            return 1;                                                                   \
        }                                                                               \
    } while (0)

#define RT_CHECK(call, description)                                                     \
    do {                                                                                \
        const rtError_t check_rc = (call);                                              \
        if (check_rc != RT_ERROR_NONE) {                                                \
            std::fprintf(stderr, "%s failed: %d (%s)\n", #call, check_rc, description); \
            return 1;                                                                   \
        }                                                                               \
    } while (0)

struct Options {
    int device_id{-1};
    uint32_t samples{a5_rtt::kDefaultSamples};
    uint32_t warmup{a5_rtt::kDefaultWarmup};
    std::array<int32_t, a5_rtt::kAllowedCpuCount> allowed_cpus{1, 2, 3, 4, 5};
    std::string json_path;
    std::string plot_path;
};

struct DeviceBuffer {
    void *ptr{nullptr};

    // Allocate one device-side buffer using the runtime's huge-page preference.
    aclError Allocate(size_t bytes) {
        const aclError rc = aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (rc != ACL_SUCCESS) {
            ptr = nullptr;
        }
        return rc;
    }

    // Release an owned device allocation during normal stack unwinding.
    ~DeviceBuffer() {
        if (ptr != nullptr) {
            aclrtFree(ptr);
        }
    }

    // Drop ownership after a forced device reset invalidates the allocation.
    void Abandon() { ptr = nullptr; }
};

struct AclScope {
    bool initialized{false};

    // Initialize ACL for the lifetime of the launcher process.
    AclScope() :
        initialized(aclInit(nullptr) == ACL_SUCCESS) {}

    // Finalize ACL only when initialization completed successfully.
    ~AclScope() {
        if (initialized) {
            aclFinalize();
        }
    }
};

struct DeviceContext {
    int device_id{-1};
    aclrtStream aicore_stream{nullptr};
    aclrtStream aicpu_stream{nullptr};
    // Select the requested device and create separate AICore and AICPU streams.
    explicit DeviceContext(int requested_device) {
        if (aclrtSetDevice(requested_device) != ACL_SUCCESS) {
            return;
        }
        device_id = requested_device;
        if (aclrtCreateStream(&aicore_stream) != ACL_SUCCESS) {
            aicore_stream = nullptr;
        }
        if (aclrtCreateStream(&aicpu_stream) != ACL_SUCCESS) {
            aicpu_stream = nullptr;
        }
    }

    // Destroy streams and reset the selected device during normal cleanup.
    ~DeviceContext() {
        if (aicpu_stream != nullptr) {
            aclrtDestroyStream(aicpu_stream);
        }
        if (aicore_stream != nullptr) {
            aclrtDestroyStream(aicore_stream);
        }
        if (device_id >= 0) {
            aclrtResetDevice(device_id);
        }
    }

    // Report whether every runtime resource required by the benchmark exists.
    bool Valid() const { return device_id >= 0 && aicore_stream != nullptr && aicpu_stream != nullptr; }

    // Reset immediately after a stream failure and prevent duplicate cleanup.
    void ResetNow() {
        if (device_id >= 0) {
            aclrtResetDevice(device_id);
        }
        device_id = -1;
        aicore_stream = nullptr;
        aicpu_stream = nullptr;
    }
};

struct TempFile {
    std::string path;

    // Remove the transient AICPU operator descriptor on every exit path.
    ~TempFile() {
        if (!path.empty()) {
            std::remove(path.c_str());
        }
    }
};

struct Statistics {
    uint64_t minimum{0};
    uint64_t p50{0};
    uint64_t p90{0};
    uint64_t p99{0};
    uint64_t maximum{0};
    double average{0};
};

struct MeanAccumulator {
    long double tick_sum{0};
    uint64_t sample_count{0};
};

enum class CoreGroup : uint32_t { kAic, kAiv, kAll };

// Parse a non-negative command-line integer without accepting trailing text.
bool ParseUnsigned(std::string_view text, uint32_t *output) {
    uint32_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc() || result.ptr != text.data() + text.size()) {
        return false;
    }
    *output = value;
    return true;
}

// Parse the production-shaped [S0,S1,S2,S3,O] AICPU affinity list.
bool ParseAllowedCpus(std::string_view text, std::array<int32_t, a5_rtt::kAllowedCpuCount> *output) {
    size_t begin = 0;
    for (size_t index = 0; index < output->size(); ++index) {
        const size_t end = text.find(',', begin);
        const std::string_view field = text.substr(begin, end == std::string_view::npos ? end : end - begin);
        uint32_t value = 0;
        if (!ParseUnsigned(field, &value) || value > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
            return false;
        }
        (*output)[index] = static_cast<int32_t>(value);
        if (index + 1 == output->size()) {
            if (end != std::string_view::npos) return false;
        } else {
            if (end == std::string_view::npos) return false;
            begin = end + 1;
        }
    }
    std::array<int32_t, a5_rtt::kAllowedCpuCount> sorted = *output;
    std::sort(sorted.begin(), sorted.end());
    return std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end();
}

// Parse the fixed five-round benchmark options.
bool ParseOptions(int argc, char **argv, Options *options) {
    if (argc < 2) {
        return false;
    }
    uint32_t device = 0;
    if (!ParseUnsigned(argv[1], &device) || device > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
        return false;
    }
    options->device_id = static_cast<int>(device);
    options->json_path = "a5-aicore-rtt-device" + std::to_string(device) + ".json";
    for (int i = 2; i < argc; ++i) {
        const std::string_view option(argv[i]);
        if ((option == "--samples" || option == "--warmup" || option == "--allowed-cpus" || option == "--json" ||
             option == "--plot") &&
            i + 1 >= argc) {
            return false;
        }
        if (option == "--samples") {
            if (!ParseUnsigned(argv[++i], &options->samples)) {
                return false;
            }
        } else if (option == "--warmup") {
            if (!ParseUnsigned(argv[++i], &options->warmup)) {
                return false;
            }
        } else if (option == "--allowed-cpus") {
            if (!ParseAllowedCpus(argv[++i], &options->allowed_cpus)) {
                return false;
            }
        } else if (option == "--json") {
            options->json_path = argv[++i];
        } else if (option == "--plot") {
            options->plot_path = argv[++i];
        } else {
            return false;
        }
    }
    return options->samples > 0 && options->samples <= a5_rtt::kMaxSamples && options->warmup <= a5_rtt::kMaxWarmup;
}

// Print the launcher syntax and its required binary locations.
void PrintUsage(const char *program) {
    std::fprintf(
        stderr,
        "usage: %s <device_id> [--samples 50] [--warmup 10] [--allowed-cpus S0,S1,S2,S3,O] "
        "[--json output.json] [--plot output.png]\n",
        program
    );
    std::fprintf(stderr, "required env: SIMPLER_DISPATCHER_SO\n");
    std::fprintf(stderr, "optional env: A5_RTT_CONSUMER_SO, A5_RTT_PRODUCER_O\n");
}

// Read a binary artifact into host memory for device upload.
bool ReadFile(const std::string &path, std::vector<char> *output) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream.is_open()) {
        std::fprintf(stderr, "open %s failed: %s\n", path.c_str(), std::strerror(errno));
        return false;
    }
    stream.seekg(0, std::ios::end);
    const std::streamoff size = stream.tellg();
    if (size <= 0) {
        return false;
    }
    output->resize(static_cast<size_t>(size));
    stream.seekg(0);
    stream.read(output->data(), static_cast<std::streamsize>(size));
    return stream.gcount() == static_cast<std::streamsize>(size);
}

// Translate a runtime-visible device index to the HAL physical index.
int HalDeviceId(int logical_id) {
    const char *visible = std::getenv("ASCEND_RT_VISIBLE_DEVICES");
    if (visible == nullptr || *visible == '\0') {
        return logical_id;
    }
    const std::string_view list(visible);
    size_t position = 0;
    int index = 0;
    while (position < list.size()) {
        while (position < list.size() && (list[position] == ' ' || list[position] == ',')) {
            ++position;
        }
        if (position == list.size()) {
            break;
        }
        int value = -1;
        const auto parsed = std::from_chars(list.data() + position, list.data() + list.size(), value);
        const bool terminated = parsed.ptr == list.data() + list.size() || *parsed.ptr == ' ' || *parsed.ptr == ',';
        if (parsed.ec != std::errc() || parsed.ptr == list.data() + position || value < 0 || !terminated) {
            std::fprintf(stderr, "malformed ASCEND_RT_VISIBLE_DEVICES=%s\n", visible);
            return -1;
        }
        if (index == logical_id) {
            return value;
        }
        ++index;
        position = static_cast<size_t>(parsed.ptr - list.data());
    }
    std::fprintf(stderr, "logical device %d is absent from ASCEND_RT_VISIBLE_DEVICES=%s\n", logical_id, visible);
    return -1;
}

// Map every possible A5 physical core register window into the process.
bool MapRegisterAddresses(int logical_device_id, a5_rtt::SharedState *state) {
    using HalResMap = int (*)(uint32_t, struct res_map_info *, uint64_t *, uint32_t *);
    auto hal_res_map = reinterpret_cast<HalResMap>(dlsym(nullptr, "halResMap"));
    if (hal_res_map == nullptr) {
        std::fprintf(stderr, "halResMap was not found; link and load libascend_hal\n");
        return false;
    }
    const int physical_device_id = HalDeviceId(logical_device_id);
    if (physical_device_id < 0) {
        return false;
    }

    for (uint32_t physical_cluster = 0; physical_cluster < a5_rtt::kPhysicalClusterCount; ++physical_cluster) {
        struct res_map_info map_info{};
        map_info.target_proc_type = PROCESS_CP1;
        map_info.res_type = RES_AICORE;
        map_info.res_id = physical_cluster;
        uint64_t map_address = 0;
        uint32_t map_length = a5_rtt::kPhysicalClusterMapSize;
        const int rc = hal_res_map(static_cast<uint32_t>(physical_device_id), &map_info, &map_address, &map_length);
        if (rc != 0 || map_address == 0 || map_length < a5_rtt::kPhysicalClusterMapSize) {
            std::fprintf(
                stderr, "halResMap(device=%d, physical_cluster=%u) failed: rc=%d address=0x%lx length=%u\n",
                physical_device_id, physical_cluster, rc, static_cast<unsigned long>(map_address), map_length
            );
            return false;
        }

        state->register_addrs[a5_rtt::PhysicalAicRegisterSlot(physical_cluster)] = map_address;
        state->register_addrs[a5_rtt::PhysicalAiv0RegisterSlot(physical_cluster)] =
            map_address + a5_rtt::kSubcoreRegisterStride;
        state->register_addrs[a5_rtt::PhysicalAiv1RegisterSlot(physical_cluster)] =
            map_address + 2 * a5_rtt::kSubcoreRegisterStride;
    }
    return true;
}

// Write one 64-bit bootstrap field without depending on an undocumented struct.
void WriteU64(char *buffer, size_t offset, uint64_t value) { std::memcpy(buffer + offset, &value, sizeof(value)); }

// Build the temporary descriptor used to resolve the uploaded AICPU consumer.
std::string MakeJsonDescriptor(uint64_t fingerprint, const std::string &so_basename) {
    // Render one exported consumer function as an AICPU operator entry.
    auto make_entry = [&](const char *exported_name) {
        char op_name[128];
        std::snprintf(op_name, sizeof(op_name), "%s_%016lx", exported_name, fingerprint);
        std::ostringstream output;
        output << "  \"" << op_name << "\": {\n"
               << "    \"opInfo\": {\n"
               << "      \"functionName\": \"" << exported_name << "\",\n"
               << "      \"kernelSo\": \"" << so_basename << "\",\n"
               << "      \"opKernelLib\": \"AICPUKernel\",\n"
               << "      \"computeCost\": \"100\",\n"
               << "      \"engine\": \"DNN_VM_AICPU\",\n"
               << "      \"flagAsync\": \"False\",\n"
               << "      \"flagPartial\": \"False\",\n"
               << "      \"userDefined\": \"False\"\n"
               << "    }\n  }";
        return output.str();
    };
    return "{\n" + make_entry("simpler_aicpu_init") + ",\n" + make_entry("simpler_aicpu_run") + "\n}\n";
}

// Compute distribution statistics for one raw scheduler/round/core record.
Statistics ComputeStatistics(const a5_rtt::CoreRttResult &record) {
    Statistics statistics;
    if (record.completed_samples == 0) {
        return statistics;
    }
    const uint32_t sample_count = std::min(record.completed_samples, a5_rtt::kMaxSamples);
    std::vector<uint64_t> samples(record.sample_ticks, record.sample_ticks + sample_count);
    std::sort(samples.begin(), samples.end());
    long double sum = 0;
    for (const uint64_t sample : samples) {
        sum += sample;
    }
    // Select the nearest lower-rank sample for stable integer percentiles.
    auto percentile = [&](uint32_t percentage) {
        const size_t index = (samples.size() - 1) * percentage / 100;
        return samples[index];
    };
    statistics.minimum = samples.front();
    statistics.p50 = percentile(50);
    statistics.p90 = percentile(90);
    statistics.p99 = percentile(99);
    statistics.maximum = samples.back();
    statistics.average = static_cast<double>(sum / samples.size());
    return statistics;
}

// Convert virtual-counter ticks to microseconds.
double TicksToMicroseconds(double ticks, uint64_t frequency) {
    return frequency == 0 ? 0 : ticks * 1.0e6 / static_cast<double>(frequency);
}

// Render a logical-core kind in tables and JSON.
const char *KindName(uint32_t core_kind) {
    return core_kind == static_cast<uint32_t>(a5_rtt::CoreKind::kAic) ? "AIC" : "AIV";
}

// Add every completed raw sample in one record to an arithmetic-mean accumulator.
void AddRecord(const a5_rtt::CoreRttResult &record, MeanAccumulator *accumulator) {
    const uint32_t sample_count = std::min(record.completed_samples, a5_rtt::kMaxSamples);
    for (uint32_t sample = 0; sample < sample_count; ++sample) {
        accumulator->tick_sum += record.sample_ticks[sample];
    }
    accumulator->sample_count += sample_count;
}

// Convert an accumulated raw-sample mean to microseconds.
double MeanMicroseconds(const MeanAccumulator &accumulator, uint64_t frequency) {
    return accumulator.sample_count == 0 ?
               0.0 :
               TicksToMicroseconds(static_cast<double>(accumulator.tick_sum / accumulator.sample_count), frequency);
}

// Add all five records for one scheduler and logical core.
void AddLogicalCore(
    const a5_rtt::RttOutput &output, uint32_t scheduler, uint32_t logical_core, MeanAccumulator *accumulator
) {
    for (uint32_t round = 0; round < output.round_count; ++round) {
        AddRecord(
            output.records[a5_rtt::ResultIndex(scheduler, round, logical_core, output.logical_core_count)], accumulator
        );
    }
}

// Aggregate AIC, AIV, or all logical cores in one cluster.
MeanAccumulator ClusterMean(const a5_rtt::RttOutput &output, uint32_t scheduler, uint32_t cluster, CoreGroup group) {
    MeanAccumulator accumulator;
    if (group == CoreGroup::kAic || group == CoreGroup::kAll) {
        AddLogicalCore(output, scheduler, a5_rtt::AicLogicalId(cluster), &accumulator);
    }
    if (group == CoreGroup::kAiv || group == CoreGroup::kAll) {
        AddLogicalCore(output, scheduler, a5_rtt::Aiv0LogicalId(cluster, output.cluster_count), &accumulator);
        AddLogicalCore(output, scheduler, a5_rtt::Aiv1LogicalId(cluster, output.cluster_count), &accumulator);
    }
    return accumulator;
}

// Aggregate one scheduler's logical cores that physically reside on one die.
MeanAccumulator DieMean(const a5_rtt::RttOutput &output, uint32_t scheduler, uint32_t die, CoreGroup group) {
    MeanAccumulator accumulator;
    for (uint32_t round = 0; round < output.round_count; ++round) {
        for (uint32_t logical = 0; logical < output.logical_core_count; ++logical) {
            const auto &record =
                output.records[a5_rtt::ResultIndex(scheduler, round, logical, output.logical_core_count)];
            const bool kind_matches =
                group == CoreGroup::kAll ||
                (group == CoreGroup::kAic && record.core_kind == static_cast<uint32_t>(a5_rtt::CoreKind::kAic)) ||
                (group == CoreGroup::kAiv && record.core_kind == static_cast<uint32_t>(a5_rtt::CoreKind::kAiv));
            if (kind_matches && a5_rtt::PhysicalCoreDie(record.physical_core_id) == die) {
                AddRecord(record, &accumulator);
            }
        }
    }
    return accumulator;
}

using OwnerFunction = uint32_t (*)(uint32_t, uint32_t);

// Aggregate the clusters assigned to one scheduler by a hypothetical policy.
MeanAccumulator
AssignmentMean(const a5_rtt::RttOutput &output, uint32_t scheduler, CoreGroup group, OwnerFunction owner) {
    MeanAccumulator accumulator;
    for (uint32_t cluster = 0; cluster < output.cluster_count; ++cluster) {
        if (owner(cluster, output.cluster_count) != scheduler) {
            continue;
        }
        const MeanAccumulator cluster_mean = ClusterMean(output, scheduler, cluster, group);
        accumulator.tick_sum += cluster_mean.tick_sum;
        accumulator.sample_count += cluster_mean.sample_count;
    }
    return accumulator;
}

// Format an ownership policy's cluster IDs for the human-readable tables.
std::string AssignedClusters(uint32_t scheduler, uint32_t cluster_count, OwnerFunction owner) {
    std::ostringstream stream;
    bool first = true;
    for (uint32_t cluster = 0; cluster < cluster_count; ++cluster) {
        if (owner(cluster, cluster_count) == scheduler) {
            stream << (first ? "" : ",") << cluster;
            first = false;
        }
    }
    return stream.str();
}

// Print per-cluster AIC and two-AIV means for all four schedulers.
void PrintClusterTable(const a5_rtt::RttOutput &output) {
    std::printf("\n1. Mean RTT by scheduler, cluster, and core kind (us)\n");
    std::printf("cluster");
    for (uint32_t scheduler = 0; scheduler < a5_rtt::kSchedulerCount; ++scheduler) {
        std::printf(" | S%u AIC | S%u AIV", scheduler, scheduler);
    }
    std::printf("\n");
    for (uint32_t cluster = 0; cluster < output.cluster_count; ++cluster) {
        std::printf("%7u", cluster);
        for (uint32_t scheduler = 0; scheduler < a5_rtt::kSchedulerCount; ++scheduler) {
            const double aic =
                MeanMicroseconds(ClusterMean(output, scheduler, cluster, CoreGroup::kAic), output.counter_frequency_hz);
            const double aiv =
                MeanMicroseconds(ClusterMean(output, scheduler, cluster, CoreGroup::kAiv), output.counter_frequency_hz);
            std::printf(" | %6.4f | %6.4f", aic, aiv);
        }
        std::printf("\n");
    }
}

// Print each scheduler's AIC, AIV, and combined mean for both physical dies.
void PrintDieTable(const a5_rtt::RttOutput &output) {
    std::printf("\n2. Mean RTT by scheduler and physical die (us)\n");
    std::printf("scheduler | die | AIC | AIV | all cores\n");
    for (uint32_t scheduler = 0; scheduler < a5_rtt::kSchedulerCount; ++scheduler) {
        for (uint32_t die = 0; die < a5_rtt::kDieCount; ++die) {
            std::printf(
                "S%u        | %u   | %.4f | %.4f | %.4f\n", scheduler, die,
                MeanMicroseconds(DieMean(output, scheduler, die, CoreGroup::kAic), output.counter_frequency_hz),
                MeanMicroseconds(DieMean(output, scheduler, die, CoreGroup::kAiv), output.counter_frequency_hz),
                MeanMicroseconds(DieMean(output, scheduler, die, CoreGroup::kAll), output.counter_frequency_hz)
            );
        }
    }
}

// Print one derived ownership policy without running a second device experiment.
void PrintAssignmentTable(const a5_rtt::RttOutput &output, const char *title, OwnerFunction owner) {
    std::printf("\n%s\n", title);
    std::printf("scheduler | clusters | AIC mean | AIV mean | cluster mean (us)\n");
    for (uint32_t scheduler = 0; scheduler < a5_rtt::kSchedulerCount; ++scheduler) {
        std::printf(
            "S%u        | %-24s | %.4f | %.4f | %.4f\n", scheduler,
            AssignedClusters(scheduler, output.cluster_count, owner).c_str(),
            MeanMicroseconds(AssignmentMean(output, scheduler, CoreGroup::kAic, owner), output.counter_frequency_hz),
            MeanMicroseconds(AssignmentMean(output, scheduler, CoreGroup::kAiv, owner), output.counter_frequency_hz),
            MeanMicroseconds(AssignmentMean(output, scheduler, CoreGroup::kAll, owner), output.counter_frequency_hz)
        );
    }
}

// Print launch diagnostics followed by all four requested aggregate tables.
void PrintSummary(const a5_rtt::RttOutput &output) {
    std::printf(
        "\nA5 AICore RTT: rc=%d complete=%u failed=%u clusters=%u logical=%u rounds=%u samples=%u warmup=%u "
        "counter=%lu Hz\n",
        output.consumer_rc, output.completed_records, output.failed_records, output.cluster_count,
        output.logical_core_count, output.round_count, output.samples_requested, output.warmup_requested,
        static_cast<unsigned long>(output.counter_frequency_hz)
    );
    std::printf(
        "entry_count=%u claims=0x%x arrivals=%u observed_cpus=[%d,%d,%d,%d,%d] "
        "ready_counts=[%u,%u,%u,%u] turn=%u first_error=(scheduler=%d, logical=%d)\n",
        output.entry_count, output.cpu_claim_mask, output.launch_arrivals, output.observed_cpus[0],
        output.observed_cpus[1], output.observed_cpus[2], output.observed_cpus[3], output.observed_cpus[4],
        output.scheduler_ready_counts[0], output.scheduler_ready_counts[1], output.scheduler_ready_counts[2],
        output.scheduler_ready_counts[3], output.measurement_turn, output.first_error_scheduler,
        output.first_error_logical_core
    );
    if (output.counter_frequency_hz == 0 || output.cluster_count == 0 || output.logical_core_count == 0) {
        return;
    }
    PrintClusterTable(output);
    PrintDieTable(output);
    PrintAssignmentTable(output, "3. Derived modulo-four assignment", a5_rtt::ModuloOwner);
    PrintAssignmentTable(output, "4. Derived contiguous assignment", a5_rtt::ContiguousOwner);
}

// Write one policy's derived scheduler means into the JSON summary.
void WriteAssignmentJson(
    std::ofstream &stream, const a5_rtt::RttOutput &output, const char *name, OwnerFunction owner, bool trailing_comma
) {
    stream << "    \"" << name << "\": [\n";
    for (uint32_t scheduler = 0; scheduler < a5_rtt::kSchedulerCount; ++scheduler) {
        stream << "      {\"scheduler\": " << scheduler << ", \"clusters\": [";
        bool first = true;
        for (uint32_t cluster = 0; cluster < output.cluster_count; ++cluster) {
            if (owner(cluster, output.cluster_count) == scheduler) {
                stream << (first ? "" : ", ") << cluster;
                first = false;
            }
        }
        stream
            << "], \"aic_mean_us\": "
            << MeanMicroseconds(AssignmentMean(output, scheduler, CoreGroup::kAic, owner), output.counter_frequency_hz)
            << ", \"aiv_mean_us\": "
            << MeanMicroseconds(AssignmentMean(output, scheduler, CoreGroup::kAiv, owner), output.counter_frequency_hz)
            << ", \"cluster_mean_us\": "
            << MeanMicroseconds(AssignmentMean(output, scheduler, CoreGroup::kAll, owner), output.counter_frequency_hz)
            << "}" << (scheduler + 1 == a5_rtt::kSchedulerCount ? "\n" : ",\n");
    }
    stream << "    ]" << (trailing_comma ? ",\n" : "\n");
}

// Persist metadata, derived summaries, per-record statistics, and every raw RTT sample.
bool WriteJson(const std::string &path, const std::string &soc_name, const a5_rtt::RttOutput &output) {
    std::ofstream stream(path);
    if (!stream.is_open()) {
        std::fprintf(stderr, "open JSON output %s failed\n", path.c_str());
        return false;
    }
    stream << std::setprecision(10);
    stream << "{\n  \"schema\": \"simpler.a5-aicore-rtt.v2\",\n";
    stream << "  \"soc_name\": \"" << soc_name << "\",\n";
    stream << "  \"counter_frequency_hz\": " << output.counter_frequency_hz << ",\n";
    stream << "  \"cluster_count\": " << output.cluster_count << ",\n";
    stream << "  \"logical_core_count\": " << output.logical_core_count << ",\n";
    stream << "  \"round_count\": " << output.round_count << ",\n";
    stream << "  \"samples_requested\": " << output.samples_requested << ",\n";
    stream << "  \"warmup_requested\": " << output.warmup_requested << ",\n";
    stream << "  \"consumer_rc\": " << output.consumer_rc << ",\n";
    stream << "  \"completed_records\": " << output.completed_records << ",\n";
    stream << "  \"failed_records\": " << output.failed_records << ",\n";
    stream << "  \"entry_count\": " << output.entry_count << ",\n";
    stream << "  \"cpu_claim_mask\": " << output.cpu_claim_mask << ",\n";
    stream << "  \"launch_arrivals\": " << output.launch_arrivals << ",\n";
    stream << "  \"measurement_turn\": " << output.measurement_turn << ",\n";
    stream << "  \"first_error_scheduler\": " << output.first_error_scheduler << ",\n";
    stream << "  \"first_error_logical_core\": " << output.first_error_logical_core << ",\n";
    stream << "  \"allowed_cpus\": [";
    for (uint32_t i = 0; i < a5_rtt::kAllowedCpuCount; ++i) {
        stream << (i == 0 ? "" : ", ") << output.allowed_cpus[i];
    }
    stream << "],\n  \"observed_cpus\": [";
    for (uint32_t i = 0; i < a5_rtt::kAllowedCpuCount; ++i) {
        stream << (i == 0 ? "" : ", ") << output.observed_cpus[i];
    }
    stream << "],\n  \"scheduler_ready_counts\": [";
    for (uint32_t i = 0; i < a5_rtt::kSchedulerCount; ++i) {
        stream << (i == 0 ? "" : ", ") << output.scheduler_ready_counts[i];
    }
    stream << "],\n  \"cluster_averages_us\": [\n";
    for (uint32_t scheduler = 0; scheduler < a5_rtt::kSchedulerCount; ++scheduler) {
        for (uint32_t cluster = 0; cluster < output.cluster_count; ++cluster) {
            stream << "    {\"scheduler\": " << scheduler << ", \"cluster\": " << cluster << ", \"aic_mean_us\": "
                   << MeanMicroseconds(
                          ClusterMean(output, scheduler, cluster, CoreGroup::kAic), output.counter_frequency_hz
                      )
                   << ", \"aiv_mean_us\": "
                   << MeanMicroseconds(
                          ClusterMean(output, scheduler, cluster, CoreGroup::kAiv), output.counter_frequency_hz
                      )
                   << ", \"cluster_mean_us\": "
                   << MeanMicroseconds(
                          ClusterMean(output, scheduler, cluster, CoreGroup::kAll), output.counter_frequency_hz
                      )
                   << "}"
                   << (scheduler + 1 == a5_rtt::kSchedulerCount && cluster + 1 == output.cluster_count ? "\n" : ",\n");
        }
    }
    stream << "  ],\n  \"die_averages_us\": [\n";
    for (uint32_t scheduler = 0; scheduler < a5_rtt::kSchedulerCount; ++scheduler) {
        for (uint32_t die = 0; die < a5_rtt::kDieCount; ++die) {
            stream << "    {\"scheduler\": " << scheduler << ", \"die\": " << die << ", \"aic_mean_us\": "
                   << MeanMicroseconds(DieMean(output, scheduler, die, CoreGroup::kAic), output.counter_frequency_hz)
                   << ", \"aiv_mean_us\": "
                   << MeanMicroseconds(DieMean(output, scheduler, die, CoreGroup::kAiv), output.counter_frequency_hz)
                   << ", \"all_mean_us\": "
                   << MeanMicroseconds(DieMean(output, scheduler, die, CoreGroup::kAll), output.counter_frequency_hz)
                   << "}" << (scheduler + 1 == a5_rtt::kSchedulerCount && die + 1 == a5_rtt::kDieCount ? "\n" : ",\n");
        }
    }
    stream << "  ],\n  \"assignment_estimates_us\": {\n";
    WriteAssignmentJson(stream, output, "modulo", a5_rtt::ModuloOwner, true);
    WriteAssignmentJson(stream, output, "contiguous", a5_rtt::ContiguousOwner, false);
    stream << "  },\n  \"records\": [\n";
    const uint32_t record_count = output.scheduler_count * output.round_count * output.logical_core_count;
    for (uint32_t i = 0; i < record_count; ++i) {
        const auto &record = output.records[i];
        const Statistics statistics = ComputeStatistics(record);
        stream << "    {\"round\": " << record.round_index << ", \"visit_position\": " << record.visit_position
               << ", \"scheduler_index\": " << record.scheduler_index << ", \"aicpu_cpu_id\": " << record.aicpu_cpu_id
               << ", \"logical_core_id\": " << record.logical_core_id
               << ", \"physical_core_id\": " << record.physical_core_id
               << ", \"die\": " << a5_rtt::PhysicalCoreDie(record.physical_core_id) << ", \"kind\": \""
               << KindName(record.core_kind) << "\", \"cluster\": " << record.cluster_id
               << ", \"lane\": " << record.lane << ", \"completed_samples\": " << record.completed_samples
               << ", \"error_code\": " << record.error_code << ", \"window_ticks\": " << record.window_ticks
               << ", \"window_us\": " << TicksToMicroseconds(record.window_ticks, output.counter_frequency_hz)
               << ", \"summary_ticks\": {\"min\": " << statistics.minimum << ", \"avg\": " << statistics.average
               << ", \"p50\": " << statistics.p50 << ", \"p90\": " << statistics.p90 << ", \"p99\": " << statistics.p99
               << ", \"max\": " << statistics.maximum << "}, \"samples_ticks\": [";
        const uint32_t sample_count = std::min(record.completed_samples, a5_rtt::kMaxSamples);
        for (uint32_t sample = 0; sample < sample_count; ++sample) {
            stream << (sample == 0 ? "" : ", ") << record.sample_ticks[sample];
        }
        stream << "]}" << (i + 1 == record_count ? "\n" : ",\n");
    }
    stream << "  ]\n}\n";
    return stream.good();
}

// Invoke the optional Matplotlib renderer without exposing paths to a shell.
void RenderPlot(const std::string &json_path, const std::string &plot_path) {
    if (plot_path.empty()) {
        return;
    }
#ifdef A5_RTT_PLOT_SCRIPT
    const char *plot_script = A5_RTT_PLOT_SCRIPT;
#else
    const char *plot_script = "plot_results.py";
#endif
    const pid_t child = fork();
    if (child < 0) {
        std::fprintf(stderr, "warning: cannot start plot renderer: %s\n", std::strerror(errno));
        return;
    }
    if (child == 0) {
        execlp("python3", "python3", plot_script, json_path.c_str(), "--output", plot_path.c_str(), nullptr);
        _exit(127);
    }
    int status = 0;
    if (waitpid(child, &status, 0) < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        std::fprintf(stderr, "warning: plot rendering failed; raw JSON remains available at %s\n", json_path.c_str());
    }
}

}  // namespace

// Discover the active topology, launch the benchmark, and emit tables plus raw JSON.
int main(int argc, char **argv) {
    // Parse only measurement controls; the active AIC/AIV counts come from the runtime.
    Options options;
    if (!ParseOptions(argc, argv, &options)) {
        PrintUsage(argv[0]);
        return 1;
    }

    // Resolve and load the three binaries used by the host, AICPU, and AICore sides.
    const char *dispatcher_path = std::getenv("SIMPLER_DISPATCHER_SO");
    if (dispatcher_path == nullptr) {
        std::fprintf(stderr, "SIMPLER_DISPATCHER_SO is required\n");
        return 1;
    }
    const char *consumer_env = std::getenv("A5_RTT_CONSUMER_SO");
    const char *producer_env = std::getenv("A5_RTT_PRODUCER_O");
    const std::string consumer_path =
        consumer_env != nullptr ? consumer_env : "../device-aicpu/build/liba5_rtt_consumer.so";
    const std::string producer_path =
        producer_env != nullptr ? producer_env : "../device-aicore/build/a5_rtt_producer.o";

    std::vector<char> dispatcher_bytes;
    std::vector<char> consumer_bytes;
    std::vector<char> producer_bytes;
    if (!ReadFile(dispatcher_path, &dispatcher_bytes) || !ReadFile(consumer_path, &consumer_bytes) ||
        !ReadFile(producer_path, &producer_bytes)) {
        return 1;
    }

    // Initialize ACL and create independent streams for the persistent producer and consumer.
    AclScope acl;
    if (!acl.initialized) {
        std::fprintf(stderr, "aclInit failed\n");
        return 1;
    }
    DeviceContext device(options.device_id);
    if (!device.Valid()) {
        std::fprintf(stderr, "failed to initialize device %d and two streams\n", options.device_id);
        return 1;
    }

    // Query the stream resources exactly as Simpler does instead of assuming 28 clusters.
    const char *soc = aclrtGetSocName();
    const std::string soc_name = soc != nullptr ? soc : "";
    uint32_t cube_cores = 0;
    uint32_t vector_cores = 0;
    if (aclrtGetStreamResLimit(device.aicore_stream, ACL_RT_DEV_RES_CUBE_CORE, &cube_cores) != ACL_SUCCESS ||
        aclrtGetStreamResLimit(device.aicore_stream, ACL_RT_DEV_RES_VECTOR_CORE, &vector_cores) != ACL_SUCCESS) {
        std::fprintf(stderr, "failed to query stream AIC/AIV resources\n");
        return 1;
    }
    if (cube_cores == 0 || vector_cores < 2) {
        std::fprintf(stderr, "unsupported stream topology: cube=%u vector=%u\n", cube_cores, vector_cores);
        return 1;
    }
    const uint32_t cluster_count = std::min({cube_cores, vector_cores / 2, a5_rtt::kMaxClusterCount});
    const uint32_t logical_core_count = cluster_count * a5_rtt::kSubcoresPerPhysicalCluster;
    std::printf(
        "runtime topology: soc=%s reported_AIC=%u reported_AIV=%u selected_clusters=%u logical_cores=%u\n",
        soc_name.c_str(), cube_cores, vector_cores, cluster_count, logical_core_count
    );

    // Map the complete A5 register-slot capacity; only runtime-discovered cores are measured.
    a5_rtt::SharedState host_state{};
    if (!MapRegisterAddresses(options.device_id, &host_state)) {
        return 1;
    }

    // Allocate and initialize all device buffers before bootstrapping the dispatcher.
    DeviceBuffer device_dispatcher;
    DeviceBuffer device_consumer;
    DeviceBuffer device_args;
    DeviceBuffer device_state;
    DeviceBuffer device_output;
    ACL_CHECK(device_dispatcher.Allocate(dispatcher_bytes.size()), "dispatcher bytes");
    ACL_CHECK(device_consumer.Allocate(consumer_bytes.size()), "consumer bytes");
    ACL_CHECK(device_args.Allocate(sizeof(a5_rtt::RttDeviceArgs)), "device args");
    ACL_CHECK(device_state.Allocate(sizeof(a5_rtt::SharedState)), "shared state");
    ACL_CHECK(device_output.Allocate(sizeof(a5_rtt::RttOutput)), "output");
    ACL_CHECK(
        aclrtMemcpy(
            device_dispatcher.ptr, dispatcher_bytes.size(), dispatcher_bytes.data(), dispatcher_bytes.size(),
            ACL_MEMCPY_HOST_TO_DEVICE
        ),
        "copy dispatcher"
    );
    ACL_CHECK(
        aclrtMemcpy(
            device_consumer.ptr, consumer_bytes.size(), consumer_bytes.data(), consumer_bytes.size(),
            ACL_MEMCPY_HOST_TO_DEVICE
        ),
        "copy consumer"
    );
    ACL_CHECK(
        aclrtMemcpy(device_state.ptr, sizeof(host_state), &host_state, sizeof(host_state), ACL_MEMCPY_HOST_TO_DEVICE),
        "copy shared state"
    );
    a5_rtt::RttOutput initial_output{};
    std::fill(std::begin(initial_output.observed_cpus), std::end(initial_output.observed_cpus), -1);
    initial_output.first_error_scheduler = -1;
    initial_output.first_error_logical_core = -1;
    initial_output.scheduler_count = a5_rtt::kSchedulerCount;
    initial_output.cluster_count = cluster_count;
    initial_output.logical_core_count = logical_core_count;
    initial_output.samples_requested = options.samples;
    initial_output.warmup_requested = options.warmup;
    initial_output.round_count = a5_rtt::kRoundCount;
    ACL_CHECK(
        aclrtMemcpy(
            device_output.ptr, sizeof(initial_output), &initial_output, sizeof(initial_output),
            ACL_MEMCPY_HOST_TO_DEVICE
        ),
        "initialize output"
    );
    // Invalidate allocations before a forced reset because their runtime handles cease to exist.
    auto reset_after_stream_failure = [&]() {
        device_dispatcher.Abandon();
        device_consumer.Abandon();
        device_args.Abandon();
        device_state.Abandon();
        device_output.Abandon();
        device.ResetNow();
    };

    // Populate the dispatcher's undocumented bootstrap ABI at its established byte offsets.
    constexpr size_t kBootstrapDispatcherAddress = 96;
    constexpr size_t kBootstrapDispatcherLength = 104;
    constexpr size_t kBootstrapDeviceId = 112;
    constexpr size_t kBootstrapInnerAddress = 120;
    constexpr size_t kBootstrapInnerLength = 128;
    std::array<char, sizeof(a5_rtt::RttDeviceArgs)> bootstrap_args{};
    WriteU64(bootstrap_args.data(), kBootstrapDispatcherAddress, reinterpret_cast<uint64_t>(device_dispatcher.ptr));
    WriteU64(bootstrap_args.data(), kBootstrapDispatcherLength, dispatcher_bytes.size());
    WriteU64(bootstrap_args.data(), kBootstrapDeviceId, static_cast<uint64_t>(options.device_id));
    WriteU64(bootstrap_args.data(), kBootstrapInnerAddress, reinterpret_cast<uint64_t>(device_consumer.ptr));
    WriteU64(bootstrap_args.data(), kBootstrapInnerLength, consumer_bytes.size());
    ACL_CHECK(
        aclrtMemcpy(
            device_args.ptr, bootstrap_args.size(), bootstrap_args.data(), bootstrap_args.size(),
            ACL_MEMCPY_HOST_TO_DEVICE
        ),
        "copy bootstrap args"
    );

    // Launch and synchronize the one-time AICPU dispatcher bootstrap.
    {
        struct BootstrapArgs {
            struct {
                uint64_t reserved[5]{};
                uint64_t device_args_ptr{0};
                uint64_t padding[20]{};
            } kernel_args;
            char kernel_name[32]{};
            char so_name[32]{};
            char op_name[32]{};
        } args;
        args.kernel_args.device_args_ptr = reinterpret_cast<uint64_t>(device_args.ptr);
        std::strncpy(args.kernel_name, "DynTileFwkKernelServerInit", sizeof(args.kernel_name) - 1);
        std::strncpy(args.so_name, "libaicpu_extend_kernels.so", sizeof(args.so_name) - 1);
        rtAicpuArgsEx_t runtime_args{};
        runtime_args.args = &args;
        runtime_args.argsSize = sizeof(args);
        runtime_args.kernelNameAddrOffset = offsetof(BootstrapArgs, kernel_name);
        runtime_args.soNameAddrOffset = offsetof(BootstrapArgs, so_name);
        RT_CHECK(
            rtAicpuKernelLaunchExWithArgs(
                rtKernelType_t::KERNEL_TYPE_AICPU_KFC, "AST_DYN_AICPU", 1, &runtime_args, nullptr, device.aicpu_stream,
                0
            ),
            "bootstrap AICPU dispatcher"
        );
    }
    ACL_CHECK(aclrtSynchronizeStream(device.aicpu_stream), "wait for dispatcher bootstrap");

    // Generate a unique operator identity so stale AICPU registrations cannot be reused.
    const uint64_t consumer_fingerprint = fingerprint::Compute(consumer_bytes.data(), consumer_bytes.size());
    char consumer_basename_buffer[96];
    std::snprintf(
        consumer_basename_buffer, sizeof(consumer_basename_buffer), "simpler_inner_%016lx_%d.so", consumer_fingerprint,
        options.device_id
    );
    const std::string consumer_basename = consumer_basename_buffer;
    TempFile descriptor;
    {
        char path[] = "/tmp/a5_aicore_rtt_XXXXXX.json";
        const int fd = mkstemps(path, 5);
        if (fd < 0) {
            std::fprintf(stderr, "mkstemps failed: %s\n", std::strerror(errno));
            return 1;
        }
        descriptor.path = path;
        const std::string json = MakeJsonDescriptor(consumer_fingerprint, consumer_basename);
        const ssize_t written = write(fd, json.data(), json.size());
        close(fd);
        if (written != static_cast<ssize_t>(json.size())) {
            std::fprintf(stderr, "failed to write temporary AICPU descriptor\n");
            return 1;
        }
    }

    // Load the generated descriptor and resolve the consumer's init and run functions.
    rtLoadBinaryOption_t load_option{};
    load_option.optionId = RT_LOAD_BINARY_OPT_CPU_KERNEL_MODE;
    load_option.value.cpuKernelMode = 0;
    rtLoadBinaryConfig_t load_config{};
    load_config.options = &load_option;
    load_config.numOpt = 1;
    void *aicpu_binary_handle = nullptr;
    RT_CHECK(
        rtsBinaryLoadFromFile(descriptor.path.c_str(), &load_config, &aicpu_binary_handle), "load AICPU descriptor"
    );

    rtFuncHandle init_handle = nullptr;
    rtFuncHandle run_handle = nullptr;
    {
        char init_name[128];
        char run_name[128];
        std::snprintf(init_name, sizeof(init_name), "simpler_aicpu_init_%016lx", consumer_fingerprint);
        std::snprintf(run_name, sizeof(run_name), "simpler_aicpu_run_%016lx", consumer_fingerprint);
        RT_CHECK(rtsFuncGetByName(aicpu_binary_handle, init_name, &init_handle), "resolve AICPU init");
        RT_CHECK(rtsFuncGetByName(aicpu_binary_handle, run_name, &run_handle), "resolve AICPU run");
    }

    // Register the mixed AIC/AIV persistent producer object.
    rtDevBinary_t producer_binary{};
    producer_binary.magic = RT_DEV_BINARY_MAGIC_ELF;
    producer_binary.version = 0;
    producer_binary.data = producer_bytes.data();
    producer_binary.length = producer_bytes.size();
    void *producer_handle = nullptr;
    RT_CHECK(rtRegisterAllKernel(&producer_binary, &producer_handle), "register mixed AICore producer");

    // Pass the discovered topology and fixed five-round protocol to every AICPU entry.
    a5_rtt::RttDeviceArgs run_args{};
    run_args.output_addr = reinterpret_cast<uint64_t>(device_output.ptr);
    run_args.shared_state_addr = reinterpret_cast<uint64_t>(device_state.ptr);
    run_args.samples = options.samples;
    run_args.warmup = options.warmup;
    run_args.round_count = a5_rtt::kRoundCount;
    run_args.cluster_count = cluster_count;
    run_args.logical_core_count = logical_core_count;
    run_args.allowed_cpu_count = a5_rtt::kAllowedCpuCount;
    run_args.scheduler_count = a5_rtt::kSchedulerCount;
    std::copy(options.allowed_cpus.begin(), options.allowed_cpus.end(), run_args.allowed_cpus);
    ACL_CHECK(
        aclrtMemcpy(device_args.ptr, sizeof(run_args), &run_args, sizeof(run_args), ACL_MEMCPY_HOST_TO_DEVICE),
        "copy measurement args"
    );

    // Initialize AICPU process globals before any of the five concurrent entries run.
    struct CpuLaunchArgs {
        uint64_t reserved[5]{};
        uint64_t device_args_ptr{0};
        uint64_t padding[20]{};
    } cpu_launch_args;
    cpu_launch_args.device_args_ptr = reinterpret_cast<uint64_t>(device_args.ptr);
    rtCpuKernelArgs_t cpu_args{};
    cpu_args.baseArgs.args = &cpu_launch_args;
    cpu_args.baseArgs.argsSize = sizeof(cpu_launch_args);
    rtLaunchKernelAttr_t launch_attribute{};
    rtKernelLaunchCfg_t cpu_launch_config{&launch_attribute, 0};
    RT_CHECK(
        rtsLaunchCpuKernel(init_handle, 1, device.aicpu_stream, &cpu_launch_config, &cpu_args), "initialize consumer"
    );
    ACL_CHECK(aclrtSynchronizeStream(device.aicpu_stream), "wait for consumer init");

    // Launch exactly one mixed producer block per runtime-discovered cluster.
    a5_rtt::AicoreKernelArgs aicore_args{reinterpret_cast<uint64_t>(device_state.ptr)};
    rtArgsEx_t aicore_runtime_args{};
    aicore_runtime_args.args = &aicore_args;
    aicore_runtime_args.argsSize = sizeof(aicore_args);
    rtTaskCfgInfo_t aicore_launch_config{};
    aicore_launch_config.schemMode = RT_SCHEM_MODE_BATCH;
    RT_CHECK(
        rtKernelLaunchWithHandleV2(
            producer_handle, 0, cluster_count, &aicore_runtime_args, nullptr, device.aicore_stream,
            &aicore_launch_config
        ),
        "launch runtime-sized mixed producer"
    );
    // Run four schedulers plus the production-shaped fifth orchestrator entry.
    RT_CHECK(
        rtsLaunchCpuKernel(run_handle, a5_rtt::kAllowedCpuCount, device.aicpu_stream, &cpu_launch_config, &cpu_args),
        "launch four schedulers plus orchestrator"
    );

    // Wait for all five ticketed rounds and recover safely if a device stream stalls.
    constexpr int32_t kStreamTimeoutMilliseconds = 15000;
    aclError sync_rc = aclrtSynchronizeStreamWithTimeout(device.aicpu_stream, kStreamTimeoutMilliseconds);
    if (sync_rc != ACL_SUCCESS) {
        std::fprintf(stderr, "AICPU measurement stream failed or timed out: %d\n", sync_rc);
        reset_after_stream_failure();
        return 1;
    }

    // Copy both the benchmark output and readiness records back for diagnostics.
    a5_rtt::RttOutput output{};
    ACL_CHECK(
        aclrtMemcpy(&output, sizeof(output), device_output.ptr, sizeof(output), ACL_MEMCPY_DEVICE_TO_HOST),
        "copy RTT output"
    );
    a5_rtt::SharedState observed_state{};
    ACL_CHECK(
        aclrtMemcpy(
            &observed_state, sizeof(observed_state), device_state.ptr, sizeof(observed_state), ACL_MEMCPY_DEVICE_TO_HOST
        ),
        "copy ready state"
    );
    uint32_t ready_magic_count = 0;
    uint32_t nonzero_ready_count = 0;
    for (uint32_t logical = 0; logical < logical_core_count; ++logical) {
        const auto &ready = observed_state.ready[logical];
        ready_magic_count += ready.ready_magic == a5_rtt::kReadyMagic ? 1u : 0u;
        nonzero_ready_count += (ready.ready_magic | ready.logical_core_id | ready.physical_core_id | ready.core_kind |
                                ready.cluster_id | ready.lane) != 0 ?
                                   1u :
                                   0u;
    }
    std::printf("GM ready diagnostics: magic_records=%u nonzero_records=%u\n", ready_magic_count, nonzero_ready_count);
    PrintSummary(output);
    if (!WriteJson(options.json_path, soc_name, output)) {
        return 1;
    }
    std::printf("full samples: %s\n", options.json_path.c_str());
    // Render the optional four-panel plot after the complete raw JSON is durable.
    RenderPlot(options.json_path, options.plot_path);

    // Confirm every persistent producer consumed its exit token before normal cleanup.
    sync_rc = aclrtSynchronizeStreamWithTimeout(device.aicore_stream, kStreamTimeoutMilliseconds);
    if (sync_rc != ACL_SUCCESS) {
        std::fprintf(stderr, "AICore producer stream failed or timed out: %d\n", sync_rc);
        reset_after_stream_failure();
        return 1;
    }

    // Treat incomplete metadata, records, or turn progression as a benchmark failure.
    const uint32_t expected_records = a5_rtt::kSchedulerCount * a5_rtt::kRoundCount * logical_core_count;
    const bool valid = output.magic == a5_rtt::kOutputMagic && output.consumer_rc == a5_rtt::kOk &&
                       output.scheduler_count == a5_rtt::kSchedulerCount && output.cluster_count == cluster_count &&
                       output.logical_core_count == logical_core_count && output.round_count == a5_rtt::kRoundCount &&
                       output.measurement_turn == a5_rtt::kSchedulerCount * a5_rtt::kRoundCount &&
                       output.completed_records == expected_records && output.failed_records == 0;
    return valid ? 0 : 2;
}

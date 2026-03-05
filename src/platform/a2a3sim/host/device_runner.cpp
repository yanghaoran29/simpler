/**
 * Device Runner Implementation - Thread-Based Simulation (Static Link Mode)
 *
 * This file implements the simulated device execution using host threads.
 * AICPU and AICore executors are statically linked into this library.
 * Kernel functions are resolved via get_kernel_func_addr() from the
 * generated kernel_dispatch.cpp (or kernel_dispatch_stub.cpp for stub builds).
 *
 * Cross-platform notes:
 * - Linux: Uses MAP_ANONYMOUS for anonymous memory mapping
 */

#include "device_runner.h"

// Statically linked executor functions (from libaicpu_kernel.a and libaicore_kernel.a)
extern "C" int aicpu_execute(Runtime* runtime);
extern "C" void aicore_execute_wrapper(Runtime* runtime, int block_idx, CoreType core_type,
                                       uint32_t physical_core_id, uint64_t regs);
extern "C" void set_platform_regs(uint64_t regs);

// Kernel dispatch function (from generated kernel_dispatch.cpp or kernel_dispatch_stub.cpp)
extern "C" uint64_t get_kernel_func_addr(int func_id);

// =============================================================================
// DeviceRunner Implementation
// =============================================================================

DeviceRunner& DeviceRunner::get() {
    static DeviceRunner runner;
    return runner;
}

DeviceRunner::~DeviceRunner() {
    finalize();
}

void* DeviceRunner::allocate_tensor(size_t bytes) {
    return mem_alloc_.alloc(bytes);
}

void DeviceRunner::free_tensor(void* dev_ptr) {
    if (dev_ptr != nullptr) {
        mem_alloc_.free(dev_ptr);
    }
}

int DeviceRunner::copy_to_device(void* dev_ptr, const void* host_ptr, size_t bytes) {
    // In simulation, this is just a memcpy
    std::memcpy(dev_ptr, host_ptr, bytes);
    return 0;
}

int DeviceRunner::copy_from_device(void* host_ptr, const void* dev_ptr, size_t bytes) {
    // In simulation, this is just a memcpy
    std::memcpy(host_ptr, dev_ptr, bytes);
    return 0;
}

int DeviceRunner::run(Runtime& runtime,
                      int block_dim,
                      int device_id,
                      const std::vector<uint8_t>& aicpu_so_binary,
                      const std::vector<uint8_t>& aicore_kernel_binary,
                      int launch_aicpu_num) {

    // Suppress unused parameter warnings (binaries are ignored in static mode)
    (void)aicpu_so_binary;
    (void)aicore_kernel_binary;

    device_id_ = device_id;

    // Validate launch_aicpu_num
    if (launch_aicpu_num < 1 || launch_aicpu_num > PLATFORM_MAX_AICPU_THREADS) {
        LOG_ERROR("launch_aicpu_num (%d) must be in range [1, %d]",
                       launch_aicpu_num, PLATFORM_MAX_AICPU_THREADS);
        return -1;
    }

    // Validate block_dim
    if (block_dim < 1 || block_dim > PLATFORM_MAX_BLOCKDIM) {
        LOG_ERROR("block_dim (%d) must be in range [1, %d]",
                       block_dim, PLATFORM_MAX_BLOCKDIM);
        return -1;
    }

    // Validate even distribution: block_dim must be divisible by scheduler thread count
    // When launch_aicpu_num == 4: 3 schedulers + 1 orchestrator (thread 3 has 0 cores)
    int scheduler_thread_num = (launch_aicpu_num == 4) ? 3 : launch_aicpu_num;
    if (block_dim % scheduler_thread_num != 0) {
        LOG_ERROR("block_dim (%d) must be evenly divisible by scheduler_thread_num (%d)",
                       block_dim, scheduler_thread_num);
        return -1;
    }

    // Calculate execution parameters
    block_dim_ = block_dim;
    int num_aicore = block_dim * cores_per_blockdim_;

    if (num_aicore > RUNTIME_MAX_WORKER) {
        LOG_ERROR("num_aicore (%d) exceeds RUNTIME_MAX_WORKER (%d)",
                       num_aicore, RUNTIME_MAX_WORKER);
        return -1;
    }

    // Initialize handshake buffers
    runtime.worker_count = num_aicore;
    worker_count_ = num_aicore;
    runtime.sche_cpu_num = launch_aicpu_num;

    // Calculate number of AIC cores
    int num_aic = block_dim;

    for (int i = 0; i < num_aicore; i++) {
        runtime.workers[i].aicpu_ready = 0;
        runtime.workers[i].aicore_done = 0;
        runtime.workers[i].control = 0;
        runtime.workers[i].task = 0;
        runtime.workers[i].task_status = 0;
        // First 1/3 are AIC, remaining 2/3 are AIV
        runtime.workers[i].core_type = (i < num_aic) ? CoreType::AIC : CoreType::AIV;
    }

    // Set function_bin_addr for each task from Runtime's func_id_to_addr_[] array
    // (addresses were stored there during init_runtime via upload_kernel_binary)
    LOG_DEBUG("Setting function_bin_addr for Tasks (Simulation)");
    for (int i = 0; i < runtime.get_task_count(); i++) {
        Task* task = runtime.get_task(i);
        if (task != nullptr) {
            uint64_t addr = runtime.get_function_bin_addr(task->func_id);
            task->function_bin_addr = addr;
            LOG_DEBUG("Task %d (func_id=%d) -> function_bin_addr=0x%lx",
                          i, task->func_id, addr);
        }
    }

    // Store runtime pointer for print_handshake_results
    last_runtime_ = &runtime;

    // Initialize performance profiling if enabled
    int rc = 0;
    if (runtime.enable_profiling) {
        rc = init_performance_profiling(runtime, num_aicore, device_id);
        if (rc != 0) {
            LOG_ERROR("init_performance_profiling failed: %d", rc);
            return rc;
        }
    }

    // Allocate simulated register blocks for all AICore cores
    size_t total_reg_size = num_aicore * SIM_REG_BLOCK_SIZE;
    void* reg_blocks = mem_alloc_.alloc(total_reg_size);
    if (reg_blocks == nullptr) {
        LOG_ERROR("Failed to allocate simulated register memory (%zu bytes)", total_reg_size);
        return -1;
    }
    std::memset(reg_blocks, 0, total_reg_size);

    // Build array of per-core register base addresses
    size_t regs_array_size = num_aicore * sizeof(uint64_t);
    uint64_t* regs_array = reinterpret_cast<uint64_t*>(mem_alloc_.alloc(regs_array_size));
    if (regs_array == nullptr) {
        LOG_ERROR("Failed to allocate register address array");
        return -1;
    }
    for (int i = 0; i < num_aicore; i++) {
        regs_array[i] = reinterpret_cast<uint64_t>(
            static_cast<uint8_t*>(reg_blocks) + i * SIM_REG_BLOCK_SIZE);
    }
    kernel_args_.regs = reinterpret_cast<uint64_t>(regs_array);

    LOG_INFO("Allocated simulated registers: %d cores x 0x%x bytes", num_aicore, SIM_REG_BLOCK_SIZE);

    // Set platform regs before launching threads (statically linked function)
    set_platform_regs(kernel_args_.regs);

    // Launch AICPU threads (statically linked aicpu_execute)
    LOG_INFO("Launching %d AICPU thread(s)", launch_aicpu_num);
    std::vector<std::thread> aicpu_threads;
    for (int i = 0; i < launch_aicpu_num; i++) {
        aicpu_threads.emplace_back([&runtime]() {
            aicpu_execute(&runtime);
        });
    }

    // Capture regs value for AICore threads
    uint64_t regs_val = kernel_args_.regs;

    // Launch AICore threads (statically linked aicore_execute_wrapper)
    LOG_INFO("Launching %d AICore thread(s)", num_aicore);
    std::vector<std::thread> aicore_threads;
    for (int i = 0; i < num_aicore; i++) {
        CoreType core_type = runtime.workers[i].core_type;
        uint32_t physical_core_id = static_cast<uint32_t>(i);
        aicore_threads.emplace_back([&runtime, i, core_type, physical_core_id, regs_val]() {
            aicore_execute_wrapper(&runtime, i, core_type, physical_core_id, regs_val);
        });
    }

    // Poll and collect performance data during execution (if enabled)
    std::thread collector_thread;
    if (runtime.enable_profiling) {
        collector_thread = std::thread([this, &runtime]() {
            poll_and_collect_performance_data(runtime.get_task_count());
        });
    }

    // Wait for all threads to complete
    LOG_INFO("Waiting for threads to complete");
    for (auto& t : aicpu_threads) {
        t.join();
    }
    for (auto& t : aicore_threads) {
        t.join();
    }

    // Wait for collector thread if it was launched
    if (runtime.enable_profiling && collector_thread.joinable()) {
        collector_thread.join();
    }

    LOG_INFO("All threads completed");

    // Collect AICPU phase data and print performance data after execution completes
    if (runtime.enable_profiling) {
        perf_collector_.collect_phase_data();
        export_swimlane_json();
    }

    return 0;
}

void DeviceRunner::print_handshake_results() {
    if (worker_count_ == 0 || last_runtime_ == nullptr) {
        return;
    }

    LOG_DEBUG("Handshake results for %d cores:", worker_count_);
    for (int i = 0; i < worker_count_; i++) {
        LOG_DEBUG("  Core %d: aicore_done=%d aicpu_ready=%d control=%d task=%d",
                      i,
                      last_runtime_->workers[i].aicore_done,
                      last_runtime_->workers[i].aicpu_ready,
                      last_runtime_->workers[i].control,
                      last_runtime_->workers[i].task);
    }
}

int DeviceRunner::clean_cache() {
    // Skip if not initialized
    if (device_id_ == -1) {
        return 0;
    }

    // Clear kernel address cache (next call will re-resolve via get_kernel_func_addr)
    func_id_to_addr_.clear();

    LOG_INFO("DeviceRunner(sim): cache cleaned");
    return 0;
}

int DeviceRunner::finalize() {
    // Skip if already finalized
    if (device_id_ == -1) {
        return 0;
    }

    // Print handshake results before cleanup
    print_handshake_results();

    // Cleanup performance profiling
    if (perf_collector_.is_initialized()) {
        auto free_cb = [](void* dev_ptr, void* user_data) -> int {
            (void)user_data;
            free(dev_ptr);
            return 0;
        };

        perf_collector_.finalize(nullptr, free_cb, nullptr);
    }

    // Clear kernel address cache
    func_id_to_addr_.clear();

    // Free all remaining allocations
    mem_alloc_.finalize();

    device_id_ = -1;
    worker_count_ = 0;
    last_runtime_ = nullptr;

    LOG_INFO("DeviceRunner(sim) finalized");
    return 0;
}

// =============================================================================
// Kernel Binary Upload (static-link mode: resolves via get_kernel_func_addr)
// =============================================================================

uint64_t DeviceRunner::upload_kernel_binary(int func_id, const uint8_t* bin_data, size_t bin_size) {
    // Suppress unused parameter warnings (binary is ignored in static mode)
    (void)bin_data;
    (void)bin_size;

    // Return cached address if already resolved
    auto it = func_id_to_addr_.find(func_id);
    if (it != func_id_to_addr_.end()) {
        LOG_INFO("Kernel func_id=%d already resolved, returning cached address", func_id);
        return it->second.func_addr;
    }

    // Resolve via statically-linked dispatch table
    uint64_t addr = get_kernel_func_addr(func_id);
    if (addr == 0) {
        LOG_ERROR("get_kernel_func_addr returned 0 for func_id=%d", func_id);
        return 0;
    }

    MappedKernel kernel;
    kernel.func_addr = addr;
    func_id_to_addr_[func_id] = kernel;

    LOG_DEBUG("Resolved kernel (static): func_id=%d -> addr=0x%lx", func_id, addr);

    return addr;
}

// =============================================================================
// Performance Profiling Implementation
// =============================================================================

int DeviceRunner::init_performance_profiling(Runtime& runtime, int num_aicore, int device_id) {
    // Define allocation callback (a2a3sim: use malloc)
    auto alloc_cb = [](size_t size, void* user_data) -> void* {
        (void)user_data;  // Not needed for malloc
        return malloc(size);
    };

    // Simulation: no registration needed (pass nullptr)
    return perf_collector_.initialize(runtime, num_aicore, device_id,
                                       alloc_cb, nullptr, nullptr);
}

void DeviceRunner::poll_and_collect_performance_data(int expected_tasks) {
    perf_collector_.poll_and_collect(expected_tasks);
}

int DeviceRunner::export_swimlane_json(const std::string& output_path) {
    return perf_collector_.export_swimlane_json(output_path);
}

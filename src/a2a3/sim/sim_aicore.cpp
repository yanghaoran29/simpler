#include "sim_aicore.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include <pthread.h>
#include <unistd.h>

#include "common/platform_config.h"

namespace {
constexpr int kMaxSimCores = 4096;
std::array<std::atomic<uint64_t>, kMaxSimCores> g_cond_regs{};

thread_local int32_t g_current_core_id = -1;
thread_local bool g_current_is_sim = false;

// Delayed completion (duration_ns > 0): ACK on receive, FIN after steady_clock elapsed >= duration_ns.
std::mutex g_sim_mu;
std::array<int32_t, kMaxSimCores> g_pending_task_id = [] {
    std::array<int32_t, kMaxSimCores> a{};
    a.fill(AICPU_TASK_INVALID);
    return a;
}();
std::array<int64_t, kMaxSimCores> g_task_start_ns{};

std::atomic<uint64_t> g_task_duration_ns{0};
std::atomic<bool> g_manual_duration_set{false};

std::atomic<bool> g_poller_started{false};

inline bool core_valid(int32_t core_id) {
    return core_id >= 0 && core_id < kMaxSimCores;
}

inline int64_t steady_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

uint64_t parse_env_duration_ns() {
    const char* s = std::getenv("AICPU_UT_SIM_AICORE_TASK_DURATION_NS");
    if (!s || !*s) {
        return 0;
    }
    char* end = nullptr;
    unsigned long long v = std::strtoull(s, &end, 10);
    if (end == s) {
        return 0;
    }
    return static_cast<uint64_t>(v);
}

uint64_t resolved_task_duration_ns() {
    if (g_manual_duration_set.load(std::memory_order_acquire)) {
        return g_task_duration_ns.load(std::memory_order_relaxed);
    }
    static std::once_flag env_once;
    static uint64_t env_duration_ns = 0;
    std::call_once(env_once, [] { env_duration_ns = parse_env_duration_ns(); });
    return env_duration_ns;
}

void sim_aicore_poller_thread_main() {
    for (;;) {
        // Avoid std::this_thread::sleep_for here: some fully-static UT link modes throw std::system_error.
        (void)usleep(10);
        uint64_t dur = resolved_task_duration_ns();
        if (dur == 0) {
            continue;
        }
        const int64_t now = steady_now_ns();
        const int64_t need = static_cast<int64_t>(dur);
        std::lock_guard<std::mutex> lock(g_sim_mu);
        for (int i = 0; i < kMaxSimCores; ++i) {
            const int32_t pending = g_pending_task_id[static_cast<size_t>(i)];
            if (pending == AICPU_TASK_INVALID) {
                continue;
            }
            if (now - g_task_start_ns[static_cast<size_t>(i)] < need) {
                continue;
            }
            const uint64_t reg = g_cond_regs[static_cast<size_t>(i)].load(std::memory_order_acquire);
            if (EXTRACT_TASK_ID(reg) != pending || EXTRACT_TASK_STATE(reg) != TASK_ACK_STATE) {
                continue;
            }
            g_cond_regs[static_cast<size_t>(i)].store(MAKE_FIN_VALUE(pending), std::memory_order_release);
            g_pending_task_id[static_cast<size_t>(i)] = AICPU_TASK_INVALID;
        }
    }
}

void* sim_aicore_poller_pthread_entry(void* /*arg*/) {
    sim_aicore_poller_thread_main();
    return nullptr;
}

void ensure_poller_started() {
    bool expected = false;
    if (!g_poller_started.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        return;
    }
    pthread_t thr{};
    const int err = pthread_create(&thr, nullptr, sim_aicore_poller_pthread_entry, nullptr);
    if (err != 0) {
        g_poller_started.store(false, std::memory_order_release);
        std::fprintf(stderr, "sim_aicore: pthread_create poller failed: %s\n", std::strerror(err));
        std::abort();
    }
    (void)pthread_detach(thr);
}
}  // namespace

extern "C" uint64_t pto2_sim_read_cond_reg(int32_t core_id) {
    if (!core_valid(core_id)) {
        return static_cast<uint64_t>(AICORE_IDLE_VALUE);
    }
    return g_cond_regs[static_cast<size_t>(core_id)].load(std::memory_order_acquire);
}

extern "C" void pto2_sim_aicore_set_task_duration_ns(uint64_t ns) {
    g_task_duration_ns.store(ns, std::memory_order_relaxed);
    g_manual_duration_set.store(true, std::memory_order_release);
}

extern "C" uint64_t pto2_sim_aicore_get_task_duration_ns(void) {
    return resolved_task_duration_ns();
}

extern "C" void pto2_sim_aicore_on_task_received(int32_t core_id, int32_t task_id) {
    if (!core_valid(core_id)) {
        return;
    }
    const uint64_t dur = resolved_task_duration_ns();
    const size_t idx = static_cast<size_t>(core_id);
    if (dur == 0) {
        std::lock_guard<std::mutex> lock(g_sim_mu);
        g_pending_task_id[idx] = AICPU_TASK_INVALID;
        g_cond_regs[idx].store(MAKE_FIN_VALUE(task_id), std::memory_order_release);
        return;
    }
    ensure_poller_started();
    std::lock_guard<std::mutex> lock(g_sim_mu);
    g_cond_regs[idx].store(MAKE_ACK_VALUE(task_id), std::memory_order_release);
    g_pending_task_id[idx] = task_id;
    g_task_start_ns[idx] = steady_now_ns();
}

extern "C" void pto2_sim_aicore_set_idle(int32_t core_id) {
    if (!core_valid(core_id)) {
        return;
    }
    const size_t idx = static_cast<size_t>(core_id);
    std::lock_guard<std::mutex> lock(g_sim_mu);
    g_pending_task_id[idx] = AICPU_TASK_INVALID;
    g_cond_regs[idx].store(static_cast<uint64_t>(AICORE_IDLE_VALUE), std::memory_order_release);
}

extern "C" void pto2_sim_set_current_core(int32_t core_id, bool is_sim) {
    g_current_core_id = core_id;
    g_current_is_sim = is_sim;
}

extern "C" void pto2_sim_clear_current_core() {
    g_current_core_id = -1;
    g_current_is_sim = false;
}

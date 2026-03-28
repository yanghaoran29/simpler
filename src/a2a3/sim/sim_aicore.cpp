#include "sim_aicore.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include "common/platform_config.h"

namespace {
constexpr int kMaxSimCores = 4096;
std::array<std::atomic<uint64_t>, kMaxSimCores> g_cond_regs{};

thread_local int32_t g_current_core_id = -1;
thread_local bool g_current_is_sim = false;

// Delayed completion (duration_ns > 0): ACK on receive, FIN after steady_clock elapsed >= duration_ns.
std::array<int32_t, kMaxSimCores> g_poller_ack_task_id = [] {
    std::array<int32_t, kMaxSimCores> a{};
    a.fill(AICPU_TASK_INVALID);
    return a;
}();
constexpr int64_t kTaskStartUnsetNs = -1;
std::array<int64_t, kMaxSimCores> g_task_start_ns = [] {
    std::array<int64_t, kMaxSimCores> a{};
    a.fill(kTaskStartUnsetNs);
    return a;
}();

std::atomic<uint64_t> g_task_duration_ns{0};
std::atomic<bool> g_manual_duration_set{false};

std::atomic<bool> g_poller_started{false};
constexpr int kDefaultPollerCpu = 16;

// Thread: Scheduler线程或模拟AICore轮询线程（公共辅助函数）。
inline bool core_valid(int32_t core_id) {
    return core_id >= 0 && core_id < kMaxSimCores;
}

// Thread: Scheduler线程或模拟AICore轮询线程（公共辅助函数）。
inline int64_t steady_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// Thread: Scheduler线程或模拟AICore轮询线程（公共辅助函数）。
inline uint64_t compose_cond_value(int32_t task_id, int32_t task_state) {
    const uint64_t id_part = static_cast<uint64_t>(static_cast<uint32_t>(task_id) & TASK_ID_MASK);
    const uint64_t state_part = (task_state == TASK_FIN_STATE) ? TASK_STATE_MASK : 0ULL;
    return id_part | state_part;
}

// Thread: Scheduler线程或模拟AICore轮询线程（经 resolved_task_duration_ns 间接调用）。
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

// Thread: Scheduler线程或模拟AICore轮询线程（两侧都会读取配置）。
uint64_t resolved_task_duration_ns() {
    if (g_manual_duration_set.load(std::memory_order_acquire)) {
        return g_task_duration_ns.load(std::memory_order_relaxed);
    }
    static std::once_flag env_once;
    static uint64_t env_duration_ns = 0;
    std::call_once(env_once, [] { env_duration_ns = parse_env_duration_ns(); });
    return env_duration_ns;
}

int resolve_poller_cpu() {
    const char* s = std::getenv("AICPU_UT_SIM_AICORE_POLLER_CPU");
    if (!s || !*s) {
        return kDefaultPollerCpu;
    }
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (end == s || v < 0) {
        return kDefaultPollerCpu;
    }
    return static_cast<int>(v);
}

// Thread: 模拟AICore轮询线程。
void sim_aicore_poller_thread_main() {
    for (;;) {
        uint64_t dur = resolved_task_duration_ns();
        if (dur == 0) {
            continue;
        }
        const int64_t now = steady_now_ns();
        const int64_t need = static_cast<int64_t>(dur);
        for (int i = 0; i < kMaxSimCores; ++i) {
            const size_t idx = static_cast<size_t>(i);
            const uint64_t reg = g_cond_regs[idx].load(std::memory_order_acquire);
            const int32_t reg_task_id = EXTRACT_TASK_ID(reg);
            const int32_t reg_state = EXTRACT_TASK_STATE(reg);
            if (reg_state != TASK_ACK_STATE) {
                g_poller_ack_task_id[idx] = AICPU_TASK_INVALID;
                g_task_start_ns[idx] = kTaskStartUnsetNs;
                continue;
            }
            if (g_poller_ack_task_id[idx] != reg_task_id) {
                g_poller_ack_task_id[idx] = reg_task_id;
                g_task_start_ns[idx] = now;
                continue;
            }
            int64_t& start_ns = g_task_start_ns[idx];
            if (start_ns == kTaskStartUnsetNs) {
                start_ns = now;
                continue;
            }
            if (now - start_ns < need) {
                continue;
            }
            uint64_t expected = compose_cond_value(reg_task_id, TASK_ACK_STATE);
            if (g_cond_regs[idx].compare_exchange_strong(
                    expected, compose_cond_value(reg_task_id, TASK_FIN_STATE),
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                g_poller_ack_task_id[idx] = AICPU_TASK_INVALID;
                start_ns = kTaskStartUnsetNs;
                continue;
            }
            g_poller_ack_task_id[idx] = AICPU_TASK_INVALID;
            start_ns = kTaskStartUnsetNs;
        }
    }
}

// Thread: 模拟AICore轮询线程（pthread 入口）。
void* sim_aicore_poller_pthread_entry(void* /*arg*/) {
    sim_aicore_poller_thread_main();
    return nullptr;
}

// Thread: Scheduler线程（首次发任务时拉起模拟AICore轮询线程）。
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
    const int cpu = resolve_poller_cpu();
    if (cpu >= 0) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu, &cpuset);
        const int aff_err = pthread_setaffinity_np(thr, sizeof(cpuset), &cpuset);
        if (aff_err != 0) {
            std::fprintf(stderr, "sim_aicore: setaffinity cpu=%d failed: %s\n", cpu, std::strerror(aff_err));
        }
    }
    (void)pthread_detach(thr);
}
}  // namespace

// Thread: Scheduler线程（轮询读取AICore COND结果）。
extern "C" uint64_t pto2_sim_read_cond_reg(int32_t core_id) {
    if (!core_valid(core_id)) {
        return static_cast<uint64_t>(AICORE_IDLE_VALUE);
    }
    return g_cond_regs[static_cast<size_t>(core_id)].load(std::memory_order_acquire);
}

// Thread: Scheduler线程（测试/控制侧通过该接口设置时延）。
extern "C" void pto2_sim_aicore_set_task_duration_ns(uint64_t ns) {
    g_task_duration_ns.store(ns, std::memory_order_relaxed);
    g_manual_duration_set.store(true, std::memory_order_release);
}

// Thread: Scheduler线程或模拟AICore轮询线程（读取有效时延）。
extern "C" uint64_t pto2_sim_aicore_get_task_duration_ns(void) {
    return resolved_task_duration_ns();
}

extern "C" void pto2_sim_aicore_start_poller(void) {
    ensure_poller_started();
}

// Thread: Scheduler线程（向AICore发放任务时触发）。
extern "C" void pto2_sim_aicore_on_task_received(int32_t core_id, int32_t task_id) {
    if (!core_valid(core_id)) {
        return;
    }
    const uint64_t dur = resolved_task_duration_ns();
    const size_t idx = static_cast<size_t>(core_id);
    if (dur == 0) {
        g_cond_regs[idx].store(compose_cond_value(task_id, TASK_FIN_STATE), std::memory_order_release);
        return;
    }
    ensure_poller_started();
    g_cond_regs[idx].store(compose_cond_value(task_id, TASK_ACK_STATE), std::memory_order_release);
}

// Thread: Scheduler线程（写 idle/exit 时触发）。
extern "C" void pto2_sim_aicore_set_idle(int32_t core_id) {
    if (!core_valid(core_id)) {
        return;
    }
    const size_t idx = static_cast<size_t>(core_id);
    g_cond_regs[idx].store(static_cast<uint64_t>(AICORE_IDLE_VALUE), std::memory_order_release);
}

// Thread: Scheduler线程（设置当前线程上下文）。
extern "C" void pto2_sim_set_current_core(int32_t core_id, bool is_sim) {
    g_current_core_id = core_id;
    g_current_is_sim = is_sim;
}

// Thread: Scheduler线程（清理当前线程上下文）。
extern "C" void pto2_sim_clear_current_core() {
    g_current_core_id = -1;
    g_current_is_sim = false;
}

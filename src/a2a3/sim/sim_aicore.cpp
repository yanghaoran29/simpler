#include "sim_aicore.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include "common/platform_config.h"
#include "common/l2_perf_profiling.h"
#include "aicpu/device_time.h"

namespace {
constexpr int kMaxSimCores = 4096;
std::array<std::atomic<uint64_t>, kMaxSimCores> g_cond_regs{};

// Host-allocated per-core L2Perf staging-ring table (uint64_t[num_cores], each
// entry an L2PerfAicoreRing*). nullptr when L2 swimlane is off.
std::atomic<uint64_t*> g_l2_perf_ring_table{nullptr};
std::atomic<int32_t> g_l2_perf_ring_cores{0};

// Resolve the per-core L2Perf staging-ring slot for a task, or nullptr when the
// swimlane table is not installed / core out of range.
inline L2PerfRecord* l2_perf_ring_slot(int32_t core_id, int32_t task_id) {
    uint64_t* table = g_l2_perf_ring_table.load(std::memory_order_acquire);
    if (table == nullptr || core_id >= g_l2_perf_ring_cores.load(std::memory_order_relaxed)) {
        return nullptr;
    }
    auto* ring = reinterpret_cast<L2PerfAicoreRing*>(table[core_id]);
    if (ring == nullptr) {
        return nullptr;
    }
    return &ring->dual_issue_slots[static_cast<uint32_t>(task_id) % PLATFORM_L2_AICORE_RING_SIZE];
}

// Open the AICore execution window: stamp start (= receive time) and a
// provisional end (= start, so the instant-FIN path already has a valid slot).
// The FIN path overwrites end with the real finish time. Timestamps use
// get_sys_cnt_aicpu so they share the AICPU dispatch/finish swimlane timeline.
inline void l2_perf_ring_stamp_start(int32_t core_id, int32_t task_id, uint64_t start_ts) {
    L2PerfRecord* slot = l2_perf_ring_slot(core_id, task_id);
    if (slot == nullptr) {
        return;
    }
    slot->start_time = start_ts;
    slot->end_time = start_ts;
    slot->duration = 0;
    slot->task_id = static_cast<uint64_t>(static_cast<uint32_t>(task_id));
    std::atomic_thread_fence(std::memory_order_release);
}

// Close the AICore execution window: stamp end (= finish time) + duration. Must
// be published before the FIN signal so the scheduler's complete_record (which
// reads the slot after observing FIN) sees the final window. Skipped if the
// slot was already reused by a later task on this core.
inline void l2_perf_ring_stamp_end(int32_t core_id, int32_t task_id, uint64_t end_ts) {
    L2PerfRecord* slot = l2_perf_ring_slot(core_id, task_id);
    if (slot == nullptr || static_cast<uint32_t>(slot->task_id) != static_cast<uint32_t>(task_id)) {
        return;
    }
    const uint64_t start = slot->start_time;
    slot->end_time = end_ts;
    slot->duration = (end_ts > start) ? (end_ts - start) : 0;
    std::atomic_thread_fence(std::memory_order_release);
}

thread_local int32_t g_current_core_id = -1;
thread_local bool g_current_is_sim = false;

// Per-core serial-execution model (duration_ns > 0). Mirrors the onboard AICore
// executor (runtime/.../aicore/aicore_executor.cpp): a core runs its tasks one at
// a time, and the start timestamp is taken when execution *begins* (right after
// the previous task's FIN), not at dispatch. on_task_received only ENQUEUES; the
// poller is the per-core serial executor (start → wait dur → end → FIN → next).
// This keeps a dual-issue pending task — pre-staged while its predecessor is still
// running — from getting a window that overlaps the predecessor's.
//
// SPSC per core: the single scheduler thread that owns the core is the producer
// (on_task_received), the poller is the consumer. Capacity exceeds the dual-issue
// in-flight depth (2) and must be a power of two.
constexpr int32_t kCoreQueueCap = 8;
constexpr uint32_t kCoreQueueMask = static_cast<uint32_t>(kCoreQueueCap) - 1u;
static_assert((kCoreQueueCap & (kCoreQueueCap - 1)) == 0, "kCoreQueueCap must be a power of two");

std::array<std::array<int32_t, kCoreQueueCap>, kMaxSimCores> g_core_queue{};
// Per-queue-entry duration (ns), published alongside the task_id under the same
// tail release-store so the poller sees a task's duration when it dequeues it.
std::array<std::array<uint64_t, kCoreQueueCap>, kMaxSimCores> g_core_queue_dur{};
std::array<std::atomic<uint32_t>, kMaxSimCores> g_core_queue_head{};  // consumer (poller)
std::array<std::atomic<uint32_t>, kMaxSimCores> g_core_queue_tail{};  // producer (scheduler)

std::array<int32_t, kMaxSimCores> g_exec_task_id = [] {
    std::array<int32_t, kMaxSimCores> a{};
    a.fill(AICPU_TASK_INVALID);
    return a;
}();
constexpr int64_t kTaskStartUnsetNs = -1;
std::array<int64_t, kMaxSimCores> g_exec_steady_start = [] {
    std::array<int64_t, kMaxSimCores> a{};
    a.fill(kTaskStartUnsetNs);
    return a;
}();
// Duration (ns) of the task currently running on each core (poller-private).
std::array<int64_t, kMaxSimCores> g_exec_need_ns{};

// Per-func_id simulated AICore execution time (ns). The table is owned and
// defined by the test/benchmark case (durations differ per sample) and installed
// via pto2_sim_aicore_set_func_duration_table before the scheduler runs; the sim
// only borrows the pointer. nullptr ⇒ every task completes instantly (no timing).
std::atomic<const int*> g_func_duration_table{nullptr};
std::atomic<int32_t> g_func_duration_count{0};
std::atomic<int32_t> g_func_duration_correction_ns{0};

// Map a func_id to its effective (correction-adjusted) duration in ns. Returns 0
// when no table is installed, for out-of-range func_ids, or when correction >=
// duration — all of which the dispatch path treats as instant completion.
inline uint64_t func_id_to_duration_ns(int32_t func_id) {
    const int* table = g_func_duration_table.load(std::memory_order_acquire);
    if (table == nullptr || func_id < 0 || func_id >= g_func_duration_count.load(std::memory_order_relaxed)) {
        return 0;
    }
    const int adjusted = table[func_id] - g_func_duration_correction_ns.load(std::memory_order_relaxed);
    return adjusted > 0 ? static_cast<uint64_t>(adjusted) : 0;
}

// Per-core func_id of the next dispatch, set by the scheduler thread that owns
// the core (pto2_sim_aicore_set_task_func_id) immediately before the
// DATA_MAIN_BASE write that drives on_task_received on the same thread, and
// consumed synchronously there. Single-thread-per-core ⇒ no cross-thread race.
std::array<int32_t, kMaxSimCores> g_task_func_id = [] {
    std::array<int32_t, kMaxSimCores> a{};
    a.fill(-1);
    return a;
}();

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

// Thread: 模拟AICore轮询线程（每核串行执行器，对齐 onboard aicore_executor）。
void sim_aicore_poller_thread_main() {
    for (;;) {
        const int64_t now = steady_now_ns();
        for (int i = 0; i < kMaxSimCores; ++i) {
            const size_t idx = static_cast<size_t>(i);
            if (g_exec_task_id[idx] == AICPU_TASK_INVALID) {
                // Core idle: start the next queued task, if any. The window opens
                // now — i.e. only after the previous task's FIN — exactly like the
                // onboard executor reading start_time right before execute_task, so
                // a pending task pre-staged during its predecessor never overlaps.
                const uint32_t head = g_core_queue_head[idx].load(std::memory_order_relaxed);
                const uint32_t tail = g_core_queue_tail[idx].load(std::memory_order_acquire);
                if (head == tail) {
                    continue;  // queue empty
                }
                const int32_t task_id = g_core_queue[idx][head & kCoreQueueMask];
                g_exec_need_ns[idx] = static_cast<int64_t>(g_core_queue_dur[idx][head & kCoreQueueMask]);
                g_core_queue_head[idx].store(head + 1u, std::memory_order_release);
                l2_perf_ring_stamp_start(i, task_id, get_sys_cnt_aicpu());
                g_exec_task_id[idx] = task_id;
                g_exec_steady_start[idx] = now;
                g_cond_regs[idx].store(compose_cond_value(task_id, TASK_ACK_STATE), std::memory_order_release);
                continue;
            }
            // Core busy: finish once this task's nominal duration has elapsed.
            if (now - g_exec_steady_start[idx] < g_exec_need_ns[idx]) {
                continue;
            }
            const int32_t task_id = g_exec_task_id[idx];
            // Close the AICore window before signalling FIN so the scheduler reads
            // the final [start,end] when it observes completion. The poller is the
            // sole COND writer during the run, so a plain store (no CAS) suffices.
            l2_perf_ring_stamp_end(i, task_id, get_sys_cnt_aicpu());
            g_exec_task_id[idx] = AICPU_TASK_INVALID;
            g_exec_steady_start[idx] = kTaskStartUnsetNs;
            g_cond_regs[idx].store(compose_cond_value(task_id, TASK_FIN_STATE), std::memory_order_release);
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

// Thread: 样例初始化（build_graph，调度运行前安装 per-func_id 时长表）。
extern "C" void pto2_sim_aicore_set_func_duration_table(
    const int* durations_ns, int32_t count, int32_t correction_ns
) {
    g_func_duration_count.store(durations_ns != nullptr ? count : 0, std::memory_order_relaxed);
    g_func_duration_correction_ns.store(correction_ns, std::memory_order_relaxed);
    g_func_duration_table.store(durations_ns, std::memory_order_release);
}

// Thread: Scheduler线程（发任务前记录本次 dispatch 的 func_id）。
extern "C" void pto2_sim_aicore_set_task_func_id(int32_t core_id, int32_t func_id) {
    if (!core_valid(core_id)) {
        return;
    }
    g_task_func_id[static_cast<size_t>(core_id)] = func_id;
}

extern "C" void pto2_sim_aicore_start_poller(void) {
    ensure_poller_started();
}

// Thread: Scheduler线程（向AICore发放任务时触发）。
extern "C" void pto2_sim_aicore_on_task_received(int32_t core_id, int32_t task_id) {
    if (!core_valid(core_id)) {
        return;
    }
    const size_t idx = static_cast<size_t>(core_id);
    // Per-func_id timing: resolve this task's duration from the func_id stashed by
    // the owning scheduler thread just before the DATA_MAIN_BASE write.
    const uint64_t dur = func_id_to_duration_ns(g_task_func_id[idx]);

    if (dur == 0) {
        // No timing model for this func_id: zero-width window, FIN immediately.
        l2_perf_ring_stamp_start(core_id, task_id, get_sys_cnt_aicpu());
        g_cond_regs[idx].store(compose_cond_value(task_id, TASK_FIN_STATE), std::memory_order_release);
        return;
    }

    // Serial-execution model: only enqueue here. The per-core poller "executor"
    // opens the window (start timestamp) and signals ACK when it actually begins
    // running the task — after the previous task on this core has finished — then
    // stamps end + FIN once dur elapses. Opening the window at dispatch time would
    // give a dual-issue pending task (pre-staged while its predecessor still runs)
    // an overlapping window and strand the predecessor's end at start (duration 0).
    ensure_poller_started();
    const uint32_t tail = g_core_queue_tail[idx].load(std::memory_order_relaxed);
    g_core_queue[idx][tail & kCoreQueueMask] = task_id;
    g_core_queue_dur[idx][tail & kCoreQueueMask] = dur;
    g_core_queue_tail[idx].store(tail + 1u, std::memory_order_release);
}

// Thread: Scheduler线程（写 idle/exit 时触发）。
extern "C" void pto2_sim_aicore_set_idle(int32_t core_id) {
    if (!core_valid(core_id)) {
        return;
    }
    const size_t idx = static_cast<size_t>(core_id);
    // Reset the per-core serial-executor state. Called at init (before any
    // dispatch) and at shutdown (after drain), so the queue is quiescent here.
    g_core_queue_head[idx].store(0, std::memory_order_relaxed);
    g_core_queue_tail[idx].store(0, std::memory_order_relaxed);
    g_exec_task_id[idx] = AICPU_TASK_INVALID;
    g_exec_steady_start[idx] = kTaskStartUnsetNs;
    g_exec_need_ns[idx] = 0;
    g_cond_regs[idx].store(static_cast<uint64_t>(AICORE_IDLE_VALUE), std::memory_order_release);
}

// Thread: Scheduler线程（运行前由 UT harness 注入 L2Perf staging-ring 表）。
extern "C" void pto2_sim_aicore_set_l2_perf_ring_table(uint64_t* ring_table, int32_t num_cores) {
    g_l2_perf_ring_cores.store(num_cores, std::memory_order_relaxed);
    g_l2_perf_ring_table.store(ring_table, std::memory_order_release);
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

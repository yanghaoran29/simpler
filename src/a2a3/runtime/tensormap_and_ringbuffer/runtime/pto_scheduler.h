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

/**
 * PTO Runtime2 - Scheduler Interface
 *
 * The Scheduler is responsible for:
 * 1. Maintaining per-resource-shape ready queues
 * 2. Tracking task state (PENDING -> READY -> RUNNING -> COMPLETED -> CONSUMED)
 * 3. Managing fanin/fanout refcounts for dependency resolution
 * 4. Advancing last_task_alive for heap reclamation
 * 5. Two-stage mixed-task completion (subtask done bits → mixed-task complete)
 *
 * The Scheduler runs on Device AI_CPU and processes:
 * - Task state transitions based on fanin_refcount
 * - Buffer lifecycle based on fanout_refcount
 * - Ring pointer advancement for flow control
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#pragma once

#include <atomic>

#include "common/core_type.h"
#include "pto2_csv_access_stats.h"
#include "pto_ring_buffer.h"
#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"

#if PTO2_ORCH_PROFILING
/** CSV ①「PTO2ReadyQueue」写次数：wiring_queue.push 成功一次计一次（见 pto_orchestrator.cpp 内计数器） */
void pto2_orch_profile_csv_m1_wiring_queue_push_done();
#endif

#if PTO2_SCHED_PROFILING
#include "aicpu/device_time.h"
#define PTO2_SCHED_CYCLE_START() uint64_t _st0 = get_sys_cnt_aicpu(), _st1
#define PTO2_SCHED_CYCLE_LAP(acc)   \
    do {                            \
        _st1 = get_sys_cnt_aicpu(); \
        acc += (_st1 - _st0);       \
        _st0 = _st1;                \
    } while (0)
#endif

// =============================================================================
// Ready Queue (Lock-free bounded MPMC — Vyukov design)
// =============================================================================

/**
 * Per-slot entry: sequence counter for ABA safety + task payload
 */
struct PTO2ReadyQueueSlot {
    std::atomic<int64_t> sequence;
    PTO2TaskSlotState *slot_state;
};

/**
 * Thread-local ready buffer for local-first dispatch optimization.
 *
 * Two buffers per scheduling thread, one per CoreType (AIC=0, AIV=1).
 * Initialized once before the scheduling loop; must be empty at
 * the start of each iteration (verified by always_assert).
 *
 * Phase 1 fills per-CoreType buffers via on_task_complete().
 * dispatch_ready_tasks_to_idle_cores drains them: local-first via
 * get_ready_task_batch, then remaining tasks pushed to global readyQ.
 */
// Number of CoreType values eligible for local dispatch (AIC=0, AIV=1)
static constexpr int PTO2_LOCAL_DISPATCH_TYPE_NUM = 2;

struct PTO2LocalReadyBuffer {
    PTO2TaskSlotState **slot_states = nullptr;
    int count = 0;
    int capacity = 0;

    void reset(PTO2TaskSlotState **buf, int cap) {
        slot_states = buf;
        count = 0;
        capacity = cap;
    }

    bool try_push(PTO2TaskSlotState *s) {
        if (slot_states && count < capacity) {
            slot_states[count++] = s;
            return true;
        }
        return false;
    }

    PTO2TaskSlotState *pop() { return (count > 0) ? slot_states[--count] : nullptr; }
};

/**
 * Lock-free bounded MPMC queue (Dmitry Vyukov design)
 *
 * Key properties:
 * - enqueue_pos and dequeue_pos on separate cache lines (no false sharing)
 * - Per-slot sequence counter prevents ABA problem
 * - Empty queue pop returns immediately (single atomic load, no lock)
 * - CAS contention is split: producers only touch enqueue_pos,
 *   consumers only touch dequeue_pos
 */
struct alignas(64) PTO2ReadyQueue {
    PTO2ReadyQueueSlot *slots;
    uint64_t capacity;
    uint64_t mask;        // capacity - 1
    char _pad0[64 - 24];  // Pad to own cache line

    std::atomic<uint64_t> enqueue_pos;
    char _pad1[64 - sizeof(std::atomic<uint64_t>)];  // Own cache line

    std::atomic<uint64_t> dequeue_pos;
    char _pad2[64 - sizeof(std::atomic<uint64_t>)];  // Own cache line

    uint64_t size() {
        uint64_t e = enqueue_pos.load(std::memory_order_relaxed);
        uint64_t d = dequeue_pos.load(std::memory_order_relaxed);
        return (e >= d) ? (e - d) : 0;
    }

    bool push(PTO2TaskSlotState *slot_state) {
        uint64_t pos;
        PTO2ReadyQueueSlot *slot;
#if PTO2_SCHED_PROFILING
        extern uint64_t g_sched_m2_ready_queue_atomic_read_raw[], g_sched_m2_ready_queue_atomic_write_raw[];
        extern uint64_t g_sched_m2_ready_queue_spin_retry_raw[];
#endif
        while (true) {
            pos = enqueue_pos.load(std::memory_order_relaxed);
            slot = &slots[pos & mask];
            int64_t seq = slot->sequence.load(std::memory_order_acquire);
#if PTO2_SCHED_PROFILING
            // wiring push（CSV②）底层原子读：enqueue_pos.load + sequence.load
            g_sched_m2_ready_queue_atomic_read_raw[0] += 2;
#endif
            int64_t diff = seq - static_cast<int64_t>(pos);
            if (diff == 0) {
                if (enqueue_pos.compare_exchange_weak(
                        pos, pos + 1, std::memory_order_relaxed, std::memory_order_relaxed
                    )) {
#if PTO2_SCHED_PROFILING
                    // CAS 成功计原子写
                    g_sched_m2_ready_queue_atomic_write_raw[0] += 1;
#endif
                    break;
                }
#if PTO2_SCHED_PROFILING
                // CAS 失败重试仍记原子写尝试
                g_sched_m2_ready_queue_atomic_write_raw[0] += 1;
                g_sched_m2_ready_queue_spin_retry_raw[0] += 1;
#endif
            } else if (diff < 0) {
                return false;  // Queue full
            } else {
#if PTO2_SCHED_PROFILING
                // slot 尚未 release，push 路径轮询重试
                g_sched_m2_ready_queue_spin_retry_raw[0] += 1;
#endif
            }
        }

        slot->slot_state = slot_state;
        slot->sequence.store(static_cast<int64_t>(pos + 1), std::memory_order_release);
#if PTO2_SCHED_PROFILING
        // 发布 sequence.store 计原子写
        g_sched_m2_ready_queue_atomic_write_raw[0] += 1;
#endif
#if PTO2_ORCH_PROFILING
        // CSV ① PTO2ReadyQueue「写次数」+1：per-ring wiring_queue 一次成功 push（Payload 构建阶段）
        pto2_orch_profile_csv_m1_wiring_queue_push_done();
#endif
        return true;
    }

    // Batch push: reserve count slots with a single CAS after confirming
    // every target slot is available under the usual Vyukov sequence check.
    void push_batch(PTO2TaskSlotState **items, int count) {
        if (count == 0) return;

        uint64_t pos;
        while (true) {
            pos = enqueue_pos.load(std::memory_order_relaxed);
            bool ready = true;
            for (int i = 0; i < count; i++) {
                PTO2ReadyQueueSlot *slot = &slots[(pos + i) & mask];
                int64_t seq = slot->sequence.load(std::memory_order_acquire);
                int64_t diff = seq - static_cast<int64_t>(pos + i);
                if (diff != 0) {
                    ready = false;
                    break;
                }
            }
            if (!ready) {
                continue;
            }
            if (enqueue_pos.compare_exchange_weak(
                    pos, pos + count, std::memory_order_relaxed, std::memory_order_relaxed
                )) {
                break;
            }
        }

        for (int i = 0; i < count; i++) {
            PTO2ReadyQueueSlot *slot = &slots[(pos + i) & mask];
            slot->slot_state = items[i];
            slot->sequence.store(static_cast<int64_t>(pos + i + 1), std::memory_order_release);
        }
    }

#if PTO2_ORCH_PROFILING || PTO2_SCHED_PROFILING
    bool push(
        PTO2TaskSlotState *slot_state, uint64_t &atomic_count, uint64_t &wait_cycle, uint64_t &atomic_read_count,
        uint64_t &atomic_write_count
    ) {
        uint64_t pos;
        PTO2ReadyQueueSlot *slot;
        uint64_t t0 = get_sys_cnt_aicpu();
        bool contended = false;
        uint32_t atomic_ops = 0;
        while (true) {
            pos = enqueue_pos.load(std::memory_order_relaxed);
            slot = &slots[pos & mask];
            int64_t seq = slot->sequence.load(std::memory_order_acquire);
            int64_t diff = seq - static_cast<int64_t>(pos);
            atomic_ops += 2;  // enqueue_pos.load + sequence.load
            atomic_read_count += 2;
            if (diff == 0) {
                if (enqueue_pos.compare_exchange_weak(
                        pos, pos + 1, std::memory_order_relaxed, std::memory_order_relaxed
                    )) {
                    atomic_ops++;  // successful CAS
                    atomic_write_count++;
                    break;
                }
                contended = true;
                atomic_ops++;  // failed CAS
                atomic_write_count++;
            } else if (diff < 0) {
                return false;  // Queue full
            } else {
                contended = true;  // diff > 0: slot not yet released, spin
            }
        }
        atomic_ops++;  // final sequence.store
        atomic_write_count++;
        atomic_count += atomic_ops;
        if (contended) {
            wait_cycle += (get_sys_cnt_aicpu() - t0);
        }

        slot->slot_state = slot_state;
        slot->sequence.store(static_cast<int64_t>(pos + 1), std::memory_order_release);
        return true;
    }
#endif

    PTO2TaskSlotState *pop() {
        // Fast-path: skip slot load when queue is clearly empty
        uint64_t d = dequeue_pos.load(std::memory_order_relaxed);
        uint64_t e = enqueue_pos.load(std::memory_order_relaxed);
        if (d >= e) {
            return nullptr;
        }

        uint64_t pos;
        PTO2ReadyQueueSlot *slot;
        while (true) {
            pos = dequeue_pos.load(std::memory_order_relaxed);
            slot = &slots[pos & mask];
            int64_t seq = slot->sequence.load(std::memory_order_acquire);
            int64_t diff = seq - static_cast<int64_t>(pos + 1);
            if (diff == 0) {
                if (dequeue_pos.compare_exchange_weak(
                        pos, pos + 1, std::memory_order_relaxed, std::memory_order_relaxed
                    ))
                    break;
            } else if (diff < 0) {
                return nullptr;  // Queue empty
            }
        }

        PTO2TaskSlotState *result = slot->slot_state;
        slot->sequence.store(static_cast<int64_t>(pos + mask + 1), std::memory_order_release);
        return result;
    }

#if PTO2_SCHED_PROFILING
    PTO2TaskSlotState *pop(
        uint64_t &atomic_count, uint64_t &wait_cycle, uint64_t &atomic_read_count, uint64_t &atomic_write_count
    ) {
        // Fast-path: skip slot load when queue is clearly empty
        uint64_t d = dequeue_pos.load(std::memory_order_relaxed);
        uint64_t e = enqueue_pos.load(std::memory_order_relaxed);
        atomic_count += 2;  // dequeue_pos.load + enqueue_pos.load
        atomic_read_count += 2;
        if (d >= e) {
            return nullptr;
        }

        uint64_t pos;
        PTO2ReadyQueueSlot *slot;
        uint64_t t0 = get_sys_cnt_aicpu();
        bool contended = false;
        uint32_t atomic_ops = 0;
        while (true) {
            pos = dequeue_pos.load(std::memory_order_relaxed);
            slot = &slots[pos & mask];
            int64_t seq = slot->sequence.load(std::memory_order_acquire);
            int64_t diff = seq - static_cast<int64_t>(pos + 1);
            atomic_ops += 2;  // dequeue_pos.load + sequence.load
            atomic_read_count += 2;
            if (diff == 0) {
                if (dequeue_pos.compare_exchange_weak(
                        pos, pos + 1, std::memory_order_relaxed, std::memory_order_relaxed
                    )) {
                    atomic_ops++;  // successful CAS
                    atomic_write_count++;
                    break;
                }
                contended = true;
                atomic_ops++;  // failed CAS
                atomic_write_count++;
            } else if (diff < 0) {
                atomic_count += atomic_ops;
                return nullptr;  // Queue empty
            } else {
                contended = true;
            }
        }
        atomic_ops++;  // final sequence.store
        atomic_write_count++;
        atomic_count += atomic_ops;
        if (contended) {
            wait_cycle += (get_sys_cnt_aicpu() - t0);
        }

        PTO2TaskSlotState *result = slot->slot_state;
        slot->sequence.store(static_cast<int64_t>(pos + mask + 1), std::memory_order_release);
        return result;
    }
#endif

    // Batch pop: reserve a contiguous run of ready slots with a single CAS.
    // Returns actual number of items popped (may be less than max_count).
    int pop_batch(PTO2TaskSlotState **out, int max_count) {
        uint64_t pos;
        int count;
        while (true) {
            pos = dequeue_pos.load(std::memory_order_relaxed);
            count = 0;
            while (count < max_count) {
                PTO2ReadyQueueSlot *slot = &slots[(pos + count) & mask];
                int64_t seq = slot->sequence.load(std::memory_order_acquire);
                int64_t diff = seq - static_cast<int64_t>(pos + count + 1);
                if (diff == 0) {
                    count++;
                    continue;
                }
                if (diff < 0) {
                    break;
                }
                count = -1;
                break;
            }
            if (count == 0) return 0;
            if (count < 0) continue;
            if (dequeue_pos.compare_exchange_weak(
                    pos, pos + count, std::memory_order_relaxed, std::memory_order_relaxed
                )) {
                break;
            }
        }

        for (int i = 0; i < count; i++) {
            PTO2ReadyQueueSlot *slot = &slots[(pos + i) & mask];
            out[i] = slot->slot_state;
            slot->sequence.store(static_cast<int64_t>(pos + i + mask + 1), std::memory_order_release);
        }
        return count;
    }

#if PTO2_SCHED_PROFILING
    int pop_batch(
        PTO2TaskSlotState **out, int max_count, uint64_t &atomic_count, uint64_t &wait_cycle,
        uint64_t &atomic_read_count, uint64_t &atomic_write_count, uint64_t &spin_retry_count,
        uint64_t &empty_poll_count
    ) {
        uint64_t pos;
        int count;
        uint64_t t0 = get_sys_cnt_aicpu();
        bool contended = false;
        uint32_t atomic_ops = 0;
        while (true) {
            pos = dequeue_pos.load(std::memory_order_relaxed);
            atomic_ops++;  // dequeue_pos.load
            atomic_read_count++;
            count = 0;
            while (count < max_count) {
                PTO2ReadyQueueSlot *slot = &slots[(pos + count) & mask];
                int64_t seq = slot->sequence.load(std::memory_order_acquire);
                int64_t diff = seq - static_cast<int64_t>(pos + count + 1);
                atomic_ops++;  // sequence.load
                atomic_read_count++;
                if (diff == 0) {
                    count++;
                    continue;
                }
                if (diff < 0) {
                    break;
                }
                contended = true;
                count = -1;
                break;
            }
            if (count == 0) {
                empty_poll_count++;
                atomic_count += atomic_ops;
                return 0;
            }
            if (count < 0) {
                spin_retry_count++;
                continue;
            }
            if (dequeue_pos.compare_exchange_weak(
                    pos, pos + count, std::memory_order_relaxed, std::memory_order_relaxed
                )) {
                atomic_ops++;  // successful CAS
                    atomic_write_count++;
                break;
            }
            contended = true;
            spin_retry_count++;
            atomic_ops++;  // failed CAS
                atomic_write_count++;
        }

        for (int i = 0; i < count; i++) {
            PTO2ReadyQueueSlot *slot = &slots[(pos + i) & mask];
            out[i] = slot->slot_state;
            slot->sequence.store(static_cast<int64_t>(pos + i + mask + 1), std::memory_order_release);
            atomic_ops++;  // sequence.store
            atomic_write_count++;
        }
        atomic_count += atomic_ops;
        if (contended) {
            wait_cycle += (get_sys_cnt_aicpu() - t0);
        }
        return count;
    }
#endif
};

// Cold-path ready queue operations (defined in pto_scheduler.cpp)
bool pto2_ready_queue_init(PTO2ReadyQueue *queue, uint64_t capacity);
void pto2_ready_queue_destroy(PTO2ReadyQueue *queue);

// =============================================================================
// SPSC Queue (Single-Producer Single-Consumer, wait-free)
// =============================================================================
//
// Bounded ring buffer optimized for the wiring queue use case:
//   - Producer: orchestrator thread (push)
//   - Consumer: scheduler thread 0 (pop_batch)
//
// Design based on Rigtorp's cached-index technique: each side caches
// the other's index locally, avoiding cross-core cache line bouncing
// on the hot path. Only when the local cache says "full" or "empty"
// does the thread issue an acquire load on the remote index.
//
// Memory layout: 4 cache-line-aligned fields ensure zero false sharing.

struct alignas(64) PTO2SpscQueue {
    // --- Producer cache lines (orchestrator thread) ---
    alignas(64) std::atomic<uint64_t> head_{0};
    alignas(64) uint64_t tail_cached_{0};

    // --- Consumer cache lines (scheduler thread 0) ---
    alignas(64) std::atomic<uint64_t> tail_{0};
    alignas(64) uint64_t head_cached_{0};

    // --- Shared (immutable after init) ---
    PTO2TaskSlotState **buffer_{nullptr};
    uint64_t mask_{0};

    bool init(uint64_t capacity) {
        if (capacity == 0 || (capacity & (capacity - 1)) != 0) return false;
        buffer_ = static_cast<PTO2TaskSlotState **>(calloc(capacity, sizeof(PTO2TaskSlotState *)));
        if (!buffer_) return false;
        mask_ = capacity - 1;
        head_.store(0, std::memory_order_relaxed);
        tail_.store(0, std::memory_order_relaxed);
        tail_cached_ = 0;
        head_cached_ = 0;
        return true;
    }

    void destroy() {
        if (buffer_) {
            free(buffer_);
            buffer_ = nullptr;
        }
    }

    // Push one item (producer only). Returns false if queue is full.
    // Full condition: next_h - tail > mask_ (i.e. > capacity-1), so the
    // effective usable capacity is capacity-1 (one slot is wasted as a
    // sentinel to distinguish full from empty). uint64_t wrapping is safe
    // since head and tail are monotonically increasing and subtraction
    // wraps correctly.
    bool push(PTO2TaskSlotState *item) {
        uint64_t h = head_.load(std::memory_order_relaxed);
        uint64_t next_h = h + 1;
        if (next_h - tail_cached_ > mask_) {
            tail_cached_ = tail_.load(std::memory_order_acquire);
            if (next_h - tail_cached_ > mask_) {
                return false;
            }
        }
        buffer_[h & mask_] = item;
        head_.store(next_h, std::memory_order_release);
        return true;
    }

    // Pop up to max_count items (consumer only). Returns actual count.
    int pop_batch(PTO2TaskSlotState **out, int max_count) {
        uint64_t t = tail_.load(std::memory_order_relaxed);
        uint64_t avail = head_cached_ - t;
        if (avail == 0) {
            head_cached_ = head_.load(std::memory_order_acquire);
            avail = head_cached_ - t;
            if (avail == 0) return 0;
        }
        int count = (avail < static_cast<uint64_t>(max_count)) ? static_cast<int>(avail) : max_count;
        for (int i = 0; i < count; i++) {
            out[i] = buffer_[(t + i) & mask_];
        }
        tail_.store(t + count, std::memory_order_release);
        return count;
    }

    // Approximate size (used for backoff decisions, not exact).
    uint64_t size() const {
        uint64_t h = head_.load(std::memory_order_acquire);
        uint64_t t = tail_.load(std::memory_order_acquire);
        return h - t;
    }
};

static_assert(sizeof(PTO2SpscQueue) == 256, "PTO2SpscQueue must be exactly 4 cache lines (256B)");
// =============================================================================

/**
 * Statistics returned by mixed-task completion processing
 */
struct PTO2CompletionStats {
    int32_t fanout_edges;       // Number of fanout edges traversed (notify consumers)
    int32_t tasks_enqueued;     // Number of consumers that became READY
    int32_t fanin_edges;        // Number of fanin edges traversed (release producers)
    bool mixed_task_completed;  // True only when this callback completed a mixed task
};

/**
 * Scheduler state structure
 *
 * Contains dynamic state updated during task execution.
 * Separated from shared memory for cache efficiency.
 * Hot-path methods are defined inline (implicitly inline as member functions).
 */
struct PTO2SchedulerState {
    // Shared memory access
    PTO2SharedMemoryHeader *sm_header;

    // Per-ring state
    struct alignas(64) RingSchedState {
        // --- Cache Line 0: ring pointer (read-only) + hot path (read-write) ---
        PTO2SharedMemoryRingHeader *ring;
        int32_t last_task_alive;
        std::atomic<int32_t> advance_lock;  // multi-thread CAS

        // --- Cache Line 1+: Thread 0 only (wiring dep_pool) ---
        alignas(64) PTO2DepListPool dep_pool;

        bool init(PTO2SharedMemoryHeader *sm_header, int32_t ring_id);
        void destroy();

        void sync_to_sm() { ring->fc.last_task_alive.store(last_task_alive, std::memory_order_release); }

        void advance_ring_pointers() {
            int32_t current_task_index = ring->fc.current_task_index.load(std::memory_order_acquire);
            int32_t old_last_task_alive = last_task_alive;

            while (last_task_alive < current_task_index) {
                PTO2TaskSlotState &slot_state = ring->get_slot_state_by_task_id(last_task_alive);
                if (slot_state.task_state.load(std::memory_order_acquire) != PTO2_TASK_CONSUMED) {
                    break;
                }
                last_task_alive++;
            }

            // Eager reset: prepare reclaimed slots for reuse while still hot in cache.
            // Safe because last_task_alive has advanced past these slots but
            // sync_to_sm has not yet published — the orchestrator cannot reuse
            // them until the release store below.
            // Skips payload, task, ring_id — immutable after RingSchedState::init().
            for (int32_t id = old_last_task_alive; id < last_task_alive; id++) {
                ring->get_slot_state_by_task_id(id).reset_for_reuse();
            }

            sync_to_sm();
        }
    } ring_sched_states[PTO2_MAX_RING_DEPTH];

    // Ready queues remain global (scheduling is ring-agnostic)
    PTO2ReadyQueue ready_queues[PTO2_NUM_RESOURCE_SHAPES];

    // Wiring subsystem — groups all wiring-related state for cache-line isolation.
    //
    // Three cache-line regions by writer:
    //   1. batch_*  / backoff — thread 0 exclusive (local batch buffer)
    //   2. queue    — SPSC: orchestrator push, thread 0 pop
    //   3. orch_needs_drain — orchestrator write, thread 0 read
    struct alignas(64) WiringState {
        static constexpr uint64_t BATCH_SIZE = 30;
        static constexpr int BACKOFF_LIMIT = 32;

        // --- Thread 0 exclusive: local batch buffer + backoff ---
        int batch_count = 0;
        int batch_index = 0;
        int backoff_counter = 0;
        PTO2TaskSlotState *batch[BATCH_SIZE];

        // --- SPSC queue: orchestrator (push) ↔ thread 0 (pop) ---
        PTO2SpscQueue queue;

        // --- Orchestrator write, thread 0 read ---
        alignas(64) std::atomic<bool> orch_needs_drain{false};
    } wiring;

    static_assert(
        offsetof(WiringState, queue) == 256, "WiringState: batch region must be exactly 4 cache lines before queue"
    );
    static_assert(sizeof(WiringState) == 576, "WiringState must be exactly 9 cache lines (576B)");

    // Statistics (cold path, isolated from hot-path fields)
#if PTO2_SCHED_PROFILING
    alignas(64) std::atomic<int64_t> tasks_completed;
    std::atomic<int64_t> tasks_consumed;
#endif
    // =========================================================================
    // Inline hot-path methods
    // =========================================================================

    /**
     * Drain wiring queue: pop submitted tasks and wire their fanout edges.
     * Called by scheduler thread 0 each loop iteration. Sets fanin_count,
     * acquires fanout_lock per producer, allocates dep_pool entries, and
     * pushes ready tasks to the appropriate ready queue.
     *
     * @return Number of tasks wired this call.
     */

    int drain_wiring_queue(bool force_drain = false) {
        int wired = 0;

        // Refill local batch buffer when exhausted.
        if (wiring.batch_index >= wiring.batch_count) {
            // Backoff: defer pop when queue holds fewer than a full batch,
            // unless force_drain, orch_needs_drain, or backoff limit reached.
            if (!force_drain && wiring.queue.size() < WiringState::BATCH_SIZE) {
                if (!wiring.orch_needs_drain.load(std::memory_order_acquire) &&
                    wiring.backoff_counter < WiringState::BACKOFF_LIMIT) {
                    wiring.backoff_counter++;
                    return 0;
                }
            }
            wiring.backoff_counter = 0;
            wiring.batch_count = wiring.queue.pop_batch(wiring.batch, WiringState::BATCH_SIZE);
            wiring.batch_index = 0;
            if (wiring.batch_count == 0) return 0;
        }

        // Process tasks from local buffer in strict FIFO order.
        while (wiring.batch_index < wiring.batch_count) {
            PTO2TaskSlotState *ws = wiring.batch[wiring.batch_index];
            int ring_id = ws->ring_id;
            auto &rss = ring_sched_states[ring_id];
            int32_t wfanin = ws->payload->fanin_actual_count;

            if (wfanin > 0 && rss.dep_pool.available() < wfanin) {
                rss.dep_pool.reclaim(*rss.ring, rss.last_task_alive);
                if (wfanin > 0 && rss.dep_pool.available() < wfanin) {
                    break;  // not enough dep_pool space — keep remainder for next call
                }
            }

            wiring.batch_index++;
            wire_task(rss, ws, wfanin);
            wired++;
        }

        return wired;
    }

    /**
     * Wire fanout edges for a single task. Sets fanin_count, acquires each
     * producer's fanout_lock, allocates dep_pool entries for live producers,
     * pushes the task to the ready queue once its fanin refcount is satisfied.
     */
    void wire_task(RingSchedState &rss, PTO2TaskSlotState *ws, int32_t wfanin) {
        PTO2TaskPayload *wp = ws->payload;
        ws->fanin_count = wfanin + 1;
#if PTO2_SCHED_PROFILING
        extern uint64_t g_sched_wire_task_count[], g_sched_wire_fanin_count[], g_sched_wire_dep_entry_count[];
        // CSV ②「读/写」原始量：wire_task 次数 wt；本行对 fanin 边数 wf 累加（∑P，每 producer 一条边）
        g_sched_wire_task_count[0]++;
        g_sched_wire_fanin_count[0] += static_cast<uint64_t>(wfanin);
#endif

        if (wfanin != 0) {
            int32_t early_finished = 0;
            pto2_for_each_fanin_slot_state(*wp, [&](PTO2TaskSlotState *producer) {
                pto2_fanout_lock(*producer);
                int32_t pstate = producer->task_state.load(std::memory_order_acquire);
                if (pstate >= PTO2_TASK_COMPLETED) {
                    early_finished++;
                } else {
                    producer->fanout_head = rss.dep_pool.prepend(producer->fanout_head, ws);
#if PTO2_SCHED_PROFILING
                    // CSV ②「PTO2DepListEntry」写次数：每存活 producer 分配并挂接一条 dep 节点
                    g_sched_wire_dep_entry_count[0]++;
#endif
                }
                pto2_fanout_unlock(*producer);
            });

            int32_t init_rc = early_finished + 1;
            int32_t new_rc = ws->fanin_refcount.fetch_add(init_rc, std::memory_order_acq_rel) + init_rc;
            if (new_rc >= ws->fanin_count) {
                ready_queues[static_cast<int32_t>(pto2_active_mask_to_shape(ws->active_mask))].push(ws);
            }
        } else {
            ws->fanin_refcount.fetch_add(1, std::memory_order_acq_rel);
            ready_queues[static_cast<int32_t>(pto2_active_mask_to_shape(ws->active_mask))].push(ws);
        }

        ws->dep_pool_mark = rss.dep_pool.top;
    }

    void check_and_handle_consumed(PTO2TaskSlotState &slot_state) {
        if (slot_state.fanout_refcount.load(std::memory_order_acquire) != slot_state.fanout_count) return;

        PTO2TaskState expected = PTO2_TASK_COMPLETED;
        if (!slot_state.task_state.compare_exchange_strong(
                expected, PTO2_TASK_CONSUMED, std::memory_order_acq_rel, std::memory_order_acquire
            )) {
            return;
        }

#if PTO2_SCHED_PROFILING
        tasks_consumed.fetch_add(1, std::memory_order_relaxed);
#endif

        int32_t ring_id = slot_state.ring_id;
        // Try-lock — if another thread is advancing this ring, it will scan our CONSUMED task
        int32_t expected_lock = 0;
#if PTO2_SCHED_PROFILING
        extern uint64_t g_sched_advance_lock_cas_count[];
        // CSV ⑦ advance_lock「CAS次数」：每次尝试 compare_exchange（线程索引 0 表示非 profiling 重载路径）
        g_sched_advance_lock_cas_count[0]++;
#endif
        if (ring_sched_states[ring_id].advance_lock.compare_exchange_strong(
                expected_lock, 1, std::memory_order_acquire, std::memory_order_relaxed
            )) {
            ring_sched_states[ring_id].advance_ring_pointers();
            ring_sched_states[ring_id].advance_lock.store(0, std::memory_order_release);
        }
    }

#if PTO2_ORCH_PROFILING || PTO2_SCHED_PROFILING
    void check_and_handle_consumed(
        PTO2TaskSlotState &slot_state, uint64_t &atomic_count, uint64_t &atomic_read_count,
        uint64_t &atomic_write_count, int32_t thread_idx = 0
    ) {
        int32_t fc = slot_state.fanout_count;
        int32_t rc = slot_state.fanout_refcount.load(std::memory_order_acquire);

        atomic_count += 2;  // fanout_count.load + fanout_refcount.load
        atomic_read_count += 2;

        if (rc != fc) return;

        PTO2TaskState expected = PTO2_TASK_COMPLETED;
        if (!slot_state.task_state.compare_exchange_strong(
                expected, PTO2_TASK_CONSUMED, std::memory_order_acq_rel, std::memory_order_acquire
            )) {
            atomic_count += 1;  // failed CAS
            atomic_write_count += 1;
            return;
        }

        atomic_count += 1;  // successful CAS
        atomic_write_count += 1;

#if PTO2_SCHED_PROFILING
        tasks_consumed.fetch_add(1, std::memory_order_relaxed);
#endif

        int32_t ring_id = slot_state.ring_id;
        // Try-lock — if another thread is advancing this ring, it will scan our CONSUMED task
        int32_t expected_lock = 0;
#if PTO2_SCHED_PROFILING
        extern uint64_t g_sched_advance_lock_cas_count[];
        // CSV ⑦ advance_lock「CAS次数」：本调度线程 thread_idx 上每次 try-lock 计 1
        g_sched_advance_lock_cas_count[thread_idx]++;
#endif
        if (ring_sched_states[ring_id].advance_lock.compare_exchange_strong(
                expected_lock, 1, std::memory_order_acquire, std::memory_order_relaxed
            )) {
            ring_sched_states[ring_id].advance_ring_pointers();
            ring_sched_states[ring_id].advance_lock.store(0, std::memory_order_release);
            atomic_count += 2;  // advance_lock CAS 成功 + advance_lock.store(0)
            atomic_write_count += 2;
            extern uint64_t g_sched_advance_count[];
            // CSV ⑦「PTO2RingFlowControl」写侧：成功推进环指针一次（advance_ring_pointers）
            g_sched_advance_count[thread_idx]++;
        } else {
            atomic_count += 1;  // failed try-lock CAS
            atomic_write_count += 1;
        }
    }
#endif

    void release_producer(PTO2TaskSlotState &slot_state) {
        slot_state.fanout_refcount.fetch_add(1, std::memory_order_acq_rel);
        check_and_handle_consumed(slot_state);
    }

#if PTO2_ORCH_PROFILING || PTO2_SCHED_PROFILING
    void release_producer(
        PTO2TaskSlotState &slot_state, uint64_t &atomic_count, uint64_t &atomic_read_count,
        uint64_t &atomic_write_count, int32_t thread_idx = 0
    ) {
        slot_state.fanout_refcount.fetch_add(1, std::memory_order_acq_rel);
        atomic_count += 1;  // fanout_refcount.fetch_add
        atomic_write_count += 1;
        check_and_handle_consumed(slot_state, atomic_count, atomic_read_count, atomic_write_count, thread_idx);
    }
#endif

    bool release_fanin_and_check_ready(PTO2TaskSlotState &slot_state, PTO2LocalReadyBuffer *local_bufs = nullptr) {
        // Atomically increment fanin_refcount and check if all producers are done
        // ACQ_REL on fanin_refcount already synchronizes with the orchestrator's
        // init release, making fanin_count visible — plain load suffices.
        int32_t new_refcount = slot_state.fanin_refcount.fetch_add(1, std::memory_order_acq_rel) + 1;

        if (new_refcount == slot_state.fanin_count) {
            // Local-first: try per-CoreType thread-local buffer before global queue
            // Route by active_mask: AIC-containing tasks → buf[0], AIV-only → buf[1]
            PTO2ResourceShape shape = pto2_active_mask_to_shape(slot_state.active_mask);
            if (!local_bufs || !local_bufs[static_cast<int32_t>(shape)].try_push(&slot_state)) {
                ready_queues[static_cast<int32_t>(shape)].push(&slot_state);
            }
            return true;
        }
        return false;
    }

#if PTO2_ORCH_PROFILING || PTO2_SCHED_PROFILING
    bool release_fanin_and_check_ready(
        PTO2TaskSlotState &slot_state, uint64_t &atomic_count, uint64_t &atomic_read_count,
        uint64_t &atomic_write_count, uint64_t &push_wait, uint64_t &queue_atomic_read_count,
        uint64_t &queue_atomic_write_count,
        PTO2LocalReadyBuffer *local_bufs = nullptr
    ) {
        int32_t new_refcount = slot_state.fanin_refcount.fetch_add(1, std::memory_order_acq_rel) + 1;
        atomic_count += 1;  // fanin_refcount.fetch_add
        atomic_write_count += 1;

        if (new_refcount == slot_state.fanin_count) {
            PTO2TaskState expected = PTO2_TASK_PENDING;
            if (slot_state.task_state.compare_exchange_strong(
                    expected, PTO2_TASK_READY, std::memory_order_acq_rel, std::memory_order_acquire
                )) {
                atomic_count += 1;  // CAS(task_state PENDING→READY)
                atomic_write_count += 1;
                // Local-first: try per-CoreType thread-local buffer before global queue
                PTO2ResourceShape shape = pto2_active_mask_to_shape(slot_state.active_mask);
                if (!local_bufs || !local_bufs[static_cast<int32_t>(shape)].try_push(&slot_state)) {
                    uint64_t q_read_before = atomic_read_count;
                    uint64_t q_write_before = atomic_write_count;
                    ready_queues[static_cast<int32_t>(shape)].push(
                        &slot_state, atomic_count, push_wait, atomic_read_count, atomic_write_count
                    );
                    queue_atomic_read_count += (atomic_read_count - q_read_before);
                    queue_atomic_write_count += (atomic_write_count - q_write_before);
                }
                return true;
            }
        }
        return false;
    }
#endif

    int get_ready_tasks_batch(
        PTO2ResourceShape shape, PTO2LocalReadyBuffer &local_buf, PTO2TaskSlotState **out, int max_count
    ) {
        int count = 0;
        while (count < max_count && local_buf.count > 0) {
            out[count++] = local_buf.slot_states[--local_buf.count];
        }
        int remaining = max_count - count;
        if (remaining > 0) {
            count += ready_queues[static_cast<int32_t>(shape)].pop_batch(out + count, remaining);
        }
        return count;
    }

#if PTO2_SCHED_PROFILING
    int get_ready_tasks_batch(
        PTO2ResourceShape shape, PTO2LocalReadyBuffer &local_buf, PTO2TaskSlotState **out, int max_count,
        uint64_t &atomic_count, uint64_t &atomic_read_count, uint64_t &atomic_write_count, uint64_t &wait_cycle,
        uint64_t &local_dispatch_count, uint64_t &spin_retry_count, uint64_t &empty_poll_count
    ) {
        int count = 0;
        while (count < max_count && local_buf.count > 0) {
            local_dispatch_count++;
            out[count++] = local_buf.slot_states[--local_buf.count];
        }
        int remaining = max_count - count;
        if (remaining > 0) {
            count += ready_queues[static_cast<int32_t>(shape)].pop_batch(
                out + count, remaining, atomic_count, wait_cycle, atomic_read_count, atomic_write_count,
                spin_retry_count, empty_poll_count
            );
        }
        return count;
    }
#endif

    void on_scope_end(PTO2TaskSlotState **task_slot_states, int32_t count) {
#if PTO2_ORCH_PROFILING
        // release_producer 内原子累加到 g_orch_scope_end_atomic_count，供 CSV ① PTO2TaskSlotState.atomic_ops 汇总
        extern uint64_t g_orch_scope_end_atomic_count, g_orch_scope_end_atomic_read_count,
            g_orch_scope_end_atomic_write_count;
        if (count > 0) __builtin_prefetch(task_slot_states[0], 1, 0);
        for (int32_t i = 0; i < count; i++) {
            if (i + 1 < count) __builtin_prefetch(task_slot_states[i + 1], 1, 0);
            release_producer(
                *task_slot_states[i], g_orch_scope_end_atomic_count, g_orch_scope_end_atomic_read_count,
                g_orch_scope_end_atomic_write_count
            );
        }
#else
        if (count > 0) __builtin_prefetch(task_slot_states[0], 1, 0);
        for (int32_t i = 0; i < count; i++) {
            if (i + 1 < count) __builtin_prefetch(task_slot_states[i + 1], 1, 0);
            release_producer(*task_slot_states[i]);
        }
#endif
    }

    /**
     * Subtask completion: atomic counter model.
     * Called when a single subtask (AIC, AIV0, or AIV1) finishes on any block.
     * Atomically increments completed_subtasks and checks whether all subtasks
     * across all blocks are done.
     *
     * @return true if this was the last subtask, completing the entire task.
     */
    bool on_subtask_complete(PTO2TaskSlotState &slot_state) {
        int16_t prev = slot_state.completed_subtasks.fetch_add(1, std::memory_order_acq_rel);
        return (prev + 1) == slot_state.total_required_subtasks;
    }

    /**
     * Two-stage completion: second stage.
     * Called exactly once when all subtasks of a mixed task are done
     * (i.e., on_subtask_complete returned true).
     * Handles fanout notification, fanin release, and self-consumption check.
     */
#if PTO2_SCHED_PROFILING
    PTO2CompletionStats
#else
    void
#endif
    on_mixed_task_complete(
        PTO2TaskSlotState &slot_state,
#if PTO2_SCHED_PROFILING
        int thread_idx,
#endif

        PTO2LocalReadyBuffer *local_bufs = nullptr
    ) {
#if PTO2_SCHED_PROFILING
        PTO2CompletionStats stats = {0, 0, 0, true};
#endif
#if PTO2_SCHED_PROFILING
        extern uint64_t g_sched_lock_cycle[], g_sched_fanout_cycle[];
        extern uint64_t g_sched_lock_atomic_count[], g_sched_lock_wait_cycle[];
        extern uint64_t g_sched_fanout_atomic_count[], g_sched_push_wait_cycle[], g_sched_lock_atomic_read_count[],
            g_sched_lock_atomic_write_count[], g_sched_fanout_atomic_read_count[], g_sched_fanout_atomic_write_count[],
            g_sched_m6_ready_queue_atomic_read_count[], g_sched_m6_ready_queue_atomic_write_count[];
        uint64_t lock_atomics = 0, lock_wait = 0;
        PTO2_SCHED_CYCLE_START();
#endif

#if PTO2_SCHED_PROFILING
        pto2_fanout_lock(slot_state, lock_atomics, lock_wait);
#else
        pto2_fanout_lock(slot_state);
#endif
        slot_state.task_state.store(PTO2_TASK_COMPLETED, std::memory_order_release);
        PTO2DepListEntry *current = slot_state.fanout_head;  // Protected by fanout_lock
        pto2_fanout_unlock(slot_state);

#if PTO2_SCHED_PROFILING
        lock_atomics += 2;  // task_state.store(COMPLETED) + fanout_unlock.store
        // ⑥ SlotState 等：fanout_lock 段原子/RWM 累计，用于周期分解（非直接 CSV 列，但与 CSV「锁次数」相关路径）
        g_sched_lock_atomic_count[thread_idx] += lock_atomics;
        g_sched_lock_atomic_read_count[thread_idx] += 0;
        g_sched_lock_atomic_write_count[thread_idx] += lock_atomics;
        g_sched_lock_wait_cycle[thread_idx] += lock_wait;
        PTO2_SCHED_CYCLE_LAP(g_sched_lock_cycle[thread_idx]);
#endif

        // Fanout: notify consumers
#if PTO2_SCHED_PROFILING
        uint64_t fanout_atomics = 0, fanout_atomic_reads = 0, fanout_atomic_writes = 0;
        uint64_t push_wait = 0, m6_readyq_atomic_reads = 0, m6_readyq_atomic_writes = 0;
#endif
        while (current != nullptr) {
            PTO2TaskSlotState &consumer_slot = *current->slot_state;
#if PTO2_SCHED_PROFILING
            stats.fanout_edges++;
            if (release_fanin_and_check_ready(
                    consumer_slot, fanout_atomics, fanout_atomic_reads, fanout_atomic_writes, push_wait,
                    m6_readyq_atomic_reads, m6_readyq_atomic_writes, local_bufs
                )) {
                stats.tasks_enqueued++;
            }
#else
            release_fanin_and_check_ready(consumer_slot, local_bufs);
#endif
            current = current->next;
        }

#if PTO2_SCHED_PROFILING
        // fanout 通知 consumer 时队列原子/等待（对应 CSV ⑥ ReadyQueue 原子近似的数据源之一）
        g_sched_fanout_atomic_count[thread_idx] += fanout_atomics;
        g_sched_fanout_atomic_read_count[thread_idx] += fanout_atomic_reads;
        g_sched_fanout_atomic_write_count[thread_idx] += fanout_atomic_writes;
        g_sched_m6_ready_queue_atomic_read_count[thread_idx] += m6_readyq_atomic_reads;
        g_sched_m6_ready_queue_atomic_write_count[thread_idx] += m6_readyq_atomic_writes;
        g_sched_push_wait_cycle[thread_idx] += push_wait;
        PTO2_SCHED_CYCLE_LAP(g_sched_fanout_cycle[thread_idx]);
        extern uint64_t g_sched_fanout_edge_count[], g_sched_tasks_enqueued_count[], g_sched_dep_list_traverse_count[];
        // CSV ⑥「C」：沿 fanout 边通知 consumer 的次数（每条边一次）
        g_sched_fanout_edge_count[thread_idx] += static_cast<uint64_t>(stats.fanout_edges);
        // CSV ⑥：因 fanin 就绪而 push 进全局/本地就绪队列的次数
        g_sched_tasks_enqueued_count[thread_idx] += static_cast<uint64_t>(stats.tasks_enqueued);
        // CSV ⑥「PTO2DepListEntry」读：每遍历一条 fanout 边读 dep 节点一次（与 fanout_edges 同计数）
        g_sched_dep_list_traverse_count[thread_idx] += static_cast<uint64_t>(stats.fanout_edges);
        return stats;
#endif
    }

    /**
     * Cold path: release producers (fanin traversal) + check self for CONSUMED.
     * Returns fanin edge count for profiling.
     */

#if PTO2_SCHED_PROFILING
    int32_t on_task_release(PTO2TaskSlotState &slot_state, int32_t thread_idx) {
        PTO2_SCHED_CYCLE_START();
        extern uint64_t g_sched_fanin_cycle[], g_sched_fanin_atomic_count[], g_sched_fanin_atomic_read_count[],
            g_sched_fanin_atomic_write_count[];
        extern uint64_t g_sched_self_atomic_count[], g_sched_self_atomic_read_count[], g_sched_self_atomic_write_count[];
        extern uint64_t g_sched_self_consumed_cycle[];
        extern uint64_t g_sched_complete_count[];
        extern uint64_t g_sched_fanin_spill_read_count[];
        uint64_t fanin_atomics = 0, fanin_atomic_reads = 0, fanin_atomic_writes = 0;
#else
    int32_t on_task_release(PTO2TaskSlotState &slot_state) {
#endif
        PTO2TaskPayload *payload = slot_state.payload;
#if PTO2_SCHED_PROFILING
        // Count spill entries accessed (entries beyond PTO2_FANIN_INLINE_CAP)
        int32_t spill_count = payload->fanin_actual_count - std::min(payload->fanin_actual_count, PTO2_FANIN_INLINE_CAP);
        if (spill_count > 0) {
            // CSV ⑦「PTO2FaninSpillEntry」读次数：超过内联上限后每条 spill 槽计一次读
            g_sched_fanin_spill_read_count[thread_idx] += static_cast<uint64_t>(spill_count);
        }
#endif
        pto2_for_each_fanin_slot_state(*payload, [&](PTO2TaskSlotState *producer_slot_state) {
#if PTO2_SCHED_PROFILING
            release_producer(
                *producer_slot_state, fanin_atomics, fanin_atomic_reads, fanin_atomic_writes, thread_idx
            );
#else
            release_producer(*producer_slot_state);
#endif
        });
#if PTO2_SCHED_PROFILING
        // ⑦ producer 侧 fanout_refcount.fetch_add 等原子累计（见 release_producer 内 fanin_atomics）
        g_sched_fanin_atomic_count[thread_idx] += fanin_atomics;
        g_sched_fanin_atomic_read_count[thread_idx] += fanin_atomic_reads;
        g_sched_fanin_atomic_write_count[thread_idx] += fanin_atomic_writes;
        PTO2_SCHED_CYCLE_LAP(g_sched_fanin_cycle[thread_idx]);
#endif

        // Self consumed check
#if PTO2_SCHED_PROFILING
        uint64_t self_atomics = 0, self_atomic_reads = 0, self_atomic_writes = 0;
        check_and_handle_consumed(slot_state, self_atomics, self_atomic_reads, self_atomic_writes, thread_idx);
        // ⑦ 自身 CONSUMED 判定路径上的原子累计（含 advance_lock 尝试等，见 check_and_handle_consumed）
        g_sched_self_atomic_count[thread_idx] += self_atomics;
        g_sched_self_atomic_read_count[thread_idx] += self_atomic_reads;
        g_sched_self_atomic_write_count[thread_idx] += self_atomic_writes;
        PTO2_SCHED_CYCLE_LAP(g_sched_self_consumed_cycle[thread_idx]);
        // 非 CSV 五列：本线程完成一次「释放生产者 + 自检 CONSUMED」的调用次数窗口累计
        g_sched_complete_count[thread_idx]++;
        extern uint64_t g_sched_task_release_count[], g_sched_fanin_release_count[];
        // CSV ⑦「on_task_release」调用次数（每释放一个已完成任务 payload 计一次）
        g_sched_task_release_count[thread_idx]++;
        // CSV ⑦「∑P」：本任务 fanin 边数，每边 release 上做一次 fanout_refcount.fetch_add
        g_sched_fanin_release_count[thread_idx] += static_cast<uint64_t>(payload->fanin_actual_count);
#else
        check_and_handle_consumed(slot_state);
#endif
        return payload->fanin_actual_count;
    }
};

// =============================================================================
// Scheduler API (cold path, defined in pto_scheduler.cpp)
// =============================================================================

bool pto2_scheduler_init(
    PTO2SchedulerState *sched, PTO2SharedMemoryHeader *sm_header, int32_t dep_pool_capacity = PTO2_DEP_LIST_POOL_SIZE
);
void pto2_scheduler_destroy(PTO2SchedulerState *sched);

// =============================================================================
// Debug Utilities (cold path, defined in pto_scheduler.cpp)
// =============================================================================

void pto2_scheduler_print_stats(PTO2SchedulerState *sched);
void pto2_scheduler_print_queues(PTO2SchedulerState *sched);
const char *pto2_task_state_name(PTO2TaskState state);

// =============================================================================
// Scheduler Profiling Data
// =============================================================================

#if PTO2_SCHED_PROFILING
struct PTO2SchedProfilingData {
    // 周期量：on_mixed_task_complete 内各子阶段（非 CSV 五列，单位 CPU cycle）
    uint64_t lock_cycle;           // fanout_lock 持锁 + 写 task_state + 解锁
    uint64_t fanout_cycle;         // 遍历 fanout 链表通知 consumer
    uint64_t fanin_cycle;          // fanin 侧 release / 入队路径
    uint64_t self_consumed_cycle;   // 自身 CONSUMED 检查 check_and_handle_consumed

    uint64_t lock_wait_cycle;   // fanout_lock 自旋等待周期
    uint64_t push_wait_cycle;   // 就绪队列 push CAS 争用等待周期
    uint64_t pop_wait_cycle;    // 就绪队列 pop CAS 争用等待周期

    uint64_t lock_atomic_count;    // lock 子阶段内原子/RWM 等累计次数（实现内部口径）
    uint64_t fanout_atomic_count;  // fanout 子阶段原子操作累计
    uint64_t fanin_atomic_count;   // fanin 子阶段原子操作累计
    uint64_t self_atomic_count;    // self_consumed 子阶段原子操作累计
    uint64_t pop_atomic_count;     // get_ready_tasks_batch / pop_batch 路径原子累计

    /** 窗口内 mixed 任务完成次数 = 对应 g_sched_complete_count[thread] 清零前的累计 */
    int64_t complete_count;
    /** CSV ② 行「PTO2TaskSlotState」五列（依赖构建 / wire） */
    PTO2CsvAccessCounters csv_m2_pto2_task_slot_state;
    /** CSV ② 行「PTO2TaskPayload」五列（读 fanin 元数据等） */
    PTO2CsvAccessCounters csv_m2_pto2_task_payload;
    /** CSV ② 行「PTO2DepListEntry」五列 */
    PTO2CsvAccessCounters csv_m2_pto2_dep_list_entry;
    /** CSV ② 行「PTO2ReadyQueue」五列（wiring pop / 入全局队列等近似） */
    PTO2CsvAccessCounters csv_m2_pto2_ready_queue;
    /** CSV ② ReadyQueue wiring push 的显式自旋/重试次数（CAS 失败 + slot 未就绪轮询） */
    uint64_t csv_m2_pto2_ready_queue_spin_retry_ops;
    /** CSV ④ ReadyQueue(pop) 显式自旋/重试次数（count<0 或 CAS 失败） */
    uint64_t csv_m4_pto2_ready_queue_spin_retry_ops;
    /** CSV ④ ReadyQueue(pop) 空轮询次数（count==0 返回） */
    uint64_t csv_m4_pto2_ready_queue_empty_poll_ops;
    /** CSV ③ 行「PTO2RingFlowControl」五列（调度器侧未单独埋点时为 0） */
    PTO2CsvAccessCounters csv_m3_pto2_ring_flow_control;
    /** CSV ③ 行「PTO2SharedMemoryHeader」五列（当前未埋点，占位） */
    PTO2CsvAccessCounters csv_m3_pto2_shared_memory_header;
    /** CSV ④ 行「PTO2TaskSlotState」五列（dispatch 子任务 RMW 等） */
    PTO2CsvAccessCounters csv_m4_pto2_task_slot_state;
    /** CSV ④ 行「PTO2TaskPayload」CL0~2 元数据读五列 */
    PTO2CsvAccessCounters csv_m4_pto2_task_payload_meta;
    /** CSV ④ 行「PTO2TaskPayload」CL3~34 tensors[] 参与 build_payload 的读次数五列 */
    PTO2CsvAccessCounters csv_m4_pto2_task_payload_tensors;
    /** CSV ④ 行「PTO2TaskPayload」CL35~50 scalars[] 参与 build_payload 的读次数五列 */
    PTO2CsvAccessCounters csv_m4_pto2_task_payload_scalars;
    /** CSV ④ 行「PTO2TaskDescriptor」五列 */
    PTO2CsvAccessCounters csv_m4_pto2_task_descriptor;
    /** CSV ④ 行「PTO2DispatchPayload」五列（按子任务写满 dispatch 缓冲的写事件近似） */
    PTO2CsvAccessCounters csv_m4_pto2_dispatch_payload;
    /** CSV ④ 行「PTO2ReadyQueue」全局 pop 五列 */
    PTO2CsvAccessCounters csv_m4_pto2_ready_queue;
    /** CSV ④ 行「PTO2ReadyQueue(pop命中)」五列（按每次 pop 调用 count>0 拆分） */
    PTO2CsvAccessCounters csv_m4_pto2_ready_queue_pop_hit;
    /** CSV ④ 行「PTO2ReadyQueue(pop空转)」五列（按每次 pop 调用 count==0 拆分） */
    PTO2CsvAccessCounters csv_m4_pto2_ready_queue_pop_miss;
    /** CSV ⑤ 行「PTO2TaskSlotState」五列（completed_subtasks 等） */
    PTO2CsvAccessCounters csv_m5_pto2_task_slot_state;
    /** CSV ⑤ 行「PTO2DispatchPayload」读 args 五列 */
    PTO2CsvAccessCounters csv_m5_pto2_dispatch_payload;
    /** CSV ⑤ 行「Tensor」五列（与 dispatch 张量遍历统计关联的近似） */
    PTO2CsvAccessCounters csv_m5_tensor;
    /** CSV ⑥ 行「PTO2TaskSlotState」五列 */
    PTO2CsvAccessCounters csv_m6_pto2_task_slot_state;
    /** CSV ⑥ 行「PTO2DepListEntry」五列 */
    PTO2CsvAccessCounters csv_m6_pto2_dep_list_entry;
    /** CSV ⑥ 行「PTO2ReadyQueue」五列 */
    PTO2CsvAccessCounters csv_m6_pto2_ready_queue;
    /** CSV ⑦ 行「PTO2TaskSlotState」五列 */
    PTO2CsvAccessCounters csv_m7_pto2_task_slot_state;
    /** CSV ⑦ 行「PTO2TaskPayload」五列 */
    PTO2CsvAccessCounters csv_m7_pto2_task_payload;
    /** CSV ⑦ 行「PTO2RingFlowControl」五列（advance_ring_pointers） */
    PTO2CsvAccessCounters csv_m7_pto2_ring_flow_control;
    /** CSV ⑦ 行「PTO2FaninSpillEntry」五列 */
    PTO2CsvAccessCounters csv_m7_pto2_fanin_spill_entry;
    /** CSV ⑦ 行「RingSchedState.advance_lock」五列 */
    PTO2CsvAccessCounters csv_m7_ring_sched_state_advance_lock;
};

/**
 * Get and reset scheduler profiling data for a specific thread.
 * Returns accumulated profiling data and resets counters.
 */
PTO2SchedProfilingData pto2_scheduler_get_profiling(int thread_idx);
#endif

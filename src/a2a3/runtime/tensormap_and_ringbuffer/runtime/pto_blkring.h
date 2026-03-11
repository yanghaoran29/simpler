/**
 * PTO Runtime2 - Block Ring Buffer (BLKRING)
 *
 * High-performance batch-oriented MPMC ready queue.
 *
 * Core idea: claim a fixed-size block of slots per CAS, amortising the
 * per-element CAS cost of the Vyukov MPMC design by BLOCK_SIZE.
 *
 * Layout
 * ------
 *   PTO2BlkRingSlot     — one 64-byte cache line; holds up to BLOCK_SIZE task_ids
 *   PTO2BlkReadyQueue   — block-granularity MPMC (enqueue_pos / dequeue_pos on
 *                         separate cache lines, same ABA-safe sequence scheme as Vyukov)
 *   PTO2BlkPendingBuffer — per-producer thread-local accumulator; tasks are
 *                         staged here until a full block is ready to commit, or
 *                         until the producer explicitly calls flush_all().
 *
 * Always enabled (BLKRING is the only ready-queue path).
 */

#ifndef PTO_BLKRING_H
#define PTO_BLKRING_H

#include "pto_runtime2_types.h"   // PTO2_BLKRING_BLOCK_SIZE, PTO2_NUM_WORKER_TYPES

#include <atomic>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static_assert(8 + 4 * PTO2_BLKRING_BLOCK_SIZE <= 64,
              "PTO2_BLKRING_BLOCK_SIZE too large to fit in a 64-byte cache line");

// =============================================================================
// Block Ring Slot
// =============================================================================

/**
 * One ring slot holds a fixed-size batch of task_ids.
 *
 * Size breakdown (BLOCK_SIZE = 8):
 *   sequence  : 4 bytes  (atomic<int32_t>)
 *   count     : 4 bytes  (int32_t)
 *   task_ids  : 32 bytes (int32_t[8])
 *   _pad      : 24 bytes
 *   Total     : 64 bytes — exactly one cache line
 *
 * The sequence field plays the same ABA-prevention role as in the Vyukov
 * MPMC design, but at block granularity: one CAS covers BLOCK_SIZE tasks.
 */
struct alignas(64) PTO2BlkRingSlot {
    std::atomic<int32_t> sequence;
    int32_t              count;
    int32_t              task_ids[PTO2_BLKRING_BLOCK_SIZE];
    char _pad[64 - 8 - 4 * PTO2_BLKRING_BLOCK_SIZE];
};

static_assert(sizeof(PTO2BlkRingSlot) == 64, "PTO2BlkRingSlot must be exactly 64 bytes");

// =============================================================================
// Block Ring Queue
// =============================================================================

/**
 * Block-granularity MPMC ring queue.
 *
 * push_block(ids, count) — commit one block; one CAS on enqueue_pos per call
 * pop_block(out_ids)     — consume one block; one CAS on dequeue_pos per call
 * approx_size()          — non-atomic estimate of enqueued blocks (for idle check)
 *
 * Thread safety: MPMC — any number of producers and consumers may call
 * push_block / pop_block concurrently without external locking.
 *
 * Memory model: identical to Vyukov MPMC —
 *   sequence.load(acquire) + CAS(relaxed) + sequence.store(release)
 */
struct alignas(64) PTO2BlkReadyQueue {
    PTO2BlkRingSlot* slots;
    uint32_t         block_capacity;   // number of block slots (power of 2)
    uint32_t         mask;             // block_capacity - 1
    char _pad0[64 - 16];

    std::atomic<uint64_t> enqueue_pos;
    char _pad1[64 - sizeof(std::atomic<uint64_t>)];

    std::atomic<uint64_t> dequeue_pos;
    char _pad2[64 - sizeof(std::atomic<uint64_t>)];

    /**
     * Push count task_ids as one block.  count must be in [1, BLOCK_SIZE].
     * Returns true on success, false when the queue is full.
     * One CAS on enqueue_pos regardless of count.
     */
    bool push_block(const int32_t* task_ids, int32_t count) {
        uint64_t pos;
        PTO2BlkRingSlot* slot;
        while (true) {
            pos  = enqueue_pos.load(std::memory_order_relaxed);
            slot = &slots[pos & mask];
            int32_t seq  = slot->sequence.load(std::memory_order_acquire);
            int32_t diff = seq - (int32_t)(uint32_t)pos;
            if (diff == 0) {
                if (enqueue_pos.compare_exchange_weak(pos, pos + 1,
                        std::memory_order_relaxed, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                return false;   // queue full
            }
        }
        slot->count = count;
        for (int32_t i = 0; i < count; i++) {
            slot->task_ids[i] = task_ids[i];
        }
        slot->sequence.store((int32_t)(uint32_t)(pos + 1), std::memory_order_release);
        return true;
    }

    /**
     * Pop one block into out_task_ids[0..return_value-1].
     * Returns 0 when the queue is empty, otherwise the number of tasks in the block.
     * out_task_ids must point to at least PTO2_BLKRING_BLOCK_SIZE elements.
     * One CAS on dequeue_pos per call.
     */
    int32_t pop_block(int32_t* out_task_ids) {
        // Fast-path empty check (two relaxed loads, no CAS)
        uint64_t d = dequeue_pos.load(std::memory_order_relaxed);
        uint64_t e = enqueue_pos.load(std::memory_order_relaxed);
        if (d >= e) {
            return 0;
        }

        uint64_t pos;
        PTO2BlkRingSlot* slot;
        while (true) {
            pos  = dequeue_pos.load(std::memory_order_relaxed);
            slot = &slots[pos & mask];
            int32_t seq  = slot->sequence.load(std::memory_order_acquire);
            int32_t diff = seq - (int32_t)(uint32_t)(pos + 1);
            if (diff == 0) {
                if (dequeue_pos.compare_exchange_weak(pos, pos + 1,
                        std::memory_order_relaxed, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                return 0;   // queue empty
            }
        }

        int32_t count = slot->count;
        for (int32_t i = 0; i < count; i++) {
            out_task_ids[i] = slot->task_ids[i];
        }
        // Release slot for reuse: seq = pos + mask + 1  (same as Vyukov)
        slot->sequence.store((int32_t)(uint32_t)(pos + (uint64_t)mask + 1),
                             std::memory_order_release);
        return count;
    }

    /** Non-atomic approximation of enqueued blocks (for idle checks). */
    uint64_t approx_size() const {
        uint64_t e = enqueue_pos.load(std::memory_order_relaxed);
        uint64_t d = dequeue_pos.load(std::memory_order_relaxed);
        return (e >= d) ? (e - d) : 0;
    }
};

// =============================================================================
// Per-Producer Pending Buffer
// =============================================================================

/**
 * Thread-local accumulator for batch push into PTO2BlkReadyQueue.
 *
 * Each producer thread holds one PTO2BlkPendingBuffer.  Tasks are added via
 * add(); when a per-worker-type sub-buffer reaches BLOCK_SIZE, it is flushed
 * automatically (flush_worker).  The producer must call flush_all() at the
 * end of each scheduling iteration to commit any partial block.
 *
 * There is no locking: the buffer is owned by a single thread.
 */
struct PTO2BlkPendingBuffer {
    int32_t task_ids[PTO2_NUM_WORKER_TYPES][PTO2_BLKRING_BLOCK_SIZE];
    int32_t counts[PTO2_NUM_WORKER_TYPES];

    void reset() {
        for (int i = 0; i < PTO2_NUM_WORKER_TYPES; i++) {
            counts[i] = 0;
        }
    }

    /**
     * Stage one task.  If the per-worker sub-buffer is already full, flush it first;
     * then append the new task.  When a block becomes full after the append, it is
     * flushed immediately.
     * Returns true if the task was added (and any full block flushed), false if
     * worker_type is out of range, blk_queues is null, or push_block failed (queue
     * full).  On push_block failure the buffer is left unchanged so no tasks are lost.
     */
    bool add(int32_t task_id, int32_t worker_type, PTO2BlkReadyQueue* blk_queues) {
        if (worker_type < 0 || worker_type >= PTO2_NUM_WORKER_TYPES || blk_queues == nullptr) {
            return false;
        }
        if (counts[worker_type] >= PTO2_BLKRING_BLOCK_SIZE) {
            bool ok = blk_queues[worker_type].push_block(task_ids[worker_type],
                                                          counts[worker_type]);
            if (!ok) {
                return false;  // Buffer full and queue full; caller must retry later
            }
            counts[worker_type] = 0;
        }
        int32_t idx = counts[worker_type]++;
        task_ids[worker_type][idx] = task_id;
        if (counts[worker_type] >= PTO2_BLKRING_BLOCK_SIZE) {
            bool ok = blk_queues[worker_type].push_block(task_ids[worker_type],
                                                          counts[worker_type]);
            if (ok) {
                counts[worker_type] = 0;
            }
            return ok;
        }
        return true;
    }

    /** Flush one worker type's pending tasks (may be a partial block). */
    bool flush_worker(int32_t worker_type, PTO2BlkReadyQueue* blk_queues) {
        if (worker_type < 0 || worker_type >= PTO2_NUM_WORKER_TYPES || blk_queues == nullptr) {
            return false;
        }
        if (counts[worker_type] == 0) {
            return true;
        }
        bool ok = blk_queues[worker_type].push_block(task_ids[worker_type],
                                                      counts[worker_type]);
        if (ok) {
            counts[worker_type] = 0;
        }
        return ok;
    }

    /** Flush all pending tasks across all worker types. */
    bool flush_all(PTO2BlkReadyQueue* blk_queues) {
        bool ok = true;
        for (int i = 0; i < PTO2_NUM_WORKER_TYPES; i++) {
            if (!flush_worker(i, blk_queues)) {
                ok = false;
            }
        }
        return ok;
    }

    bool any_pending() const {
        for (int i = 0; i < PTO2_NUM_WORKER_TYPES; i++) {
            if (counts[i] > 0) return true;
        }
        return false;
    }
};

// =============================================================================
// Cold-path declarations (implemented in pto_scheduler.cpp)
// =============================================================================

bool pto2_blkring_init(PTO2BlkReadyQueue* q, uint32_t block_capacity);
void pto2_blkring_destroy(PTO2BlkReadyQueue* q);

#endif  // PTO_BLKRING_H

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
 * tm_tensormap.h — minimal, dependency-free producer-lookup map.
 *
 * A standalone reimplementation of the PTO2TensorMap engine (producer lookup +
 * sub-region overlap detection + lazy invalidation + pooled entries) with no
 * dependency on any project struct/class. It uses only <cstdint>/<cstring>/
 * <cassert>.
 *
 * Memory: the map never allocates. The caller hands it one raw buffer (sized
 * via bytes_required) at the single dependency point init()/attach(). All state
 * — header, hash buckets, entry pool, free list, per-ring task heads — lives
 * inside that buffer, addressed by offsets, with intrusive links stored as pool
 * indices (not pointers). The image is therefore position-independent: build it
 * on the host, memcpy it elsewhere, and attach(base) with no pointer fix-up.
 *
 * Producer identity is an opaque uint64_t whose ring/local encoding is owned by
 * this module (make_id / ring_of / local_of); validity follows a per-ring
 * "last alive" watermark (sync / cleanup_retired).
 */

#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>

namespace tmap {

constexpr uint32_t TM_MAX_DIMS = 5;   // per-region dimensionality cap
constexpr uint32_t TM_MAX_RINGS = 8;  // task-id ring layers cap

// Configuration POD. num_buckets and every task_window[r] must be powers of two.
struct TmConfig {
    uint32_t num_buckets;
    uint32_t pool_size;
    uint32_t num_rings;
    uint32_t task_window[TM_MAX_RINGS];
};

// Region descriptor — the only input type. Replaces the project Tensor.
//   extent_elem    : element count spanned by this view (L1 range = [start_offset, start_offset+extent_elem))
//   storage_numel  : total elements in the backing buffer (used for L2 reference-shape derivation)
//   elem_size      : bytes per element (stands in for dtype)
// Strides are element-granular and strictly > 0; layout matches PTO2 Tensor semantics.
struct TmRegion {
    uint64_t base_addr;
    uint64_t start_offset;
    uint64_t extent_elem;
    uint64_t storage_numel;
    uint32_t elem_size;
    uint32_t ndims;
    int32_t version;
    uint8_t is_contiguous;
    uint32_t shapes[TM_MAX_DIMS];
    uint32_t strides[TM_MAX_DIMS];
};

enum class TmOverlap { None, Covered, Other };

// Pool entry. Links are pool indices (-1 = none) so the buffer is relocatable.
struct TmEntry {
    uint64_t base_addr;
    uint64_t producer_id;
    uint64_t start_offset;
    uint64_t extent_elem;
    int32_t version;
    uint32_t ndims;
    uint32_t elem_size;
    uint8_t is_contiguous;
    uint32_t shapes[TM_MAX_DIMS];
    uint32_t strides[TM_MAX_DIMS];
    int32_t next_in_bucket;
    int32_t prev_in_bucket;
    int32_t next_in_task;
    int32_t prev_in_task;
    int32_t bucket_index;  // -1 when unlinked
};

// In-buffer header: config echo + cursors + sub-region offsets.
struct TmHeader {
    TmConfig cfg;
    int32_t next_entry_idx;
    int32_t free_num;
    int32_t last_alive[TM_MAX_RINGS];
    int32_t last_cleanup[TM_MAX_RINGS];
    uint64_t off_buckets;
    uint64_t off_pool;
    uint64_t off_free;
    uint64_t off_task_heads[TM_MAX_RINGS];
};

namespace detail {
inline uint64_t align_up(uint64_t x, uint64_t a) { return (x + a - 1) & ~(a - 1); }
constexpr uint64_t kRegionAlign = 64;

// Single source of truth for region placement, shared by bytes_required / init.
inline uint64_t layout(const TmConfig &cfg, TmHeader *out /* may be null */) {
    uint64_t cur = align_up(sizeof(TmHeader), kRegionAlign);
    const uint64_t off_buckets = cur;
    cur = align_up(cur + uint64_t(cfg.num_buckets) * sizeof(int32_t), kRegionAlign);
    const uint64_t off_pool = cur;
    cur = align_up(cur + uint64_t(cfg.pool_size) * sizeof(TmEntry), kRegionAlign);
    const uint64_t off_free = cur;
    cur = align_up(cur + uint64_t(cfg.pool_size) * sizeof(int32_t), kRegionAlign);
    uint64_t off_task[TM_MAX_RINGS] = {};
    for (uint32_t r = 0; r < cfg.num_rings; r++) {
        off_task[r] = cur;
        cur = align_up(cur + uint64_t(cfg.task_window[r]) * sizeof(int32_t), kRegionAlign);
    }
    if (out != nullptr) {
        out->off_buckets = off_buckets;
        out->off_pool = off_pool;
        out->off_free = off_free;
        for (uint32_t r = 0; r < cfg.num_rings; r++) out->off_task_heads[r] = off_task[r];
    }
    return cur;
}
}  // namespace detail

// Three-level overlap cascade (L1 byte-range / L2 hyper-rectangle / L3 conservative OTHER).
// `in` is the probe (consumer); `e` is a stored producer entry sharing the same base buffer.
inline TmOverlap tm_overlap(const TmRegion &in, const TmEntry &e) {
    // A newer storage generation always depends on the older producer (whole-buffer mutation).
    if (in.version > e.version) return TmOverlap::Other;

    // L1 — O(1) byte-range intersection.
    const uint64_t in_begin = in.start_offset, in_end = in.start_offset + in.extent_elem;
    const uint64_t e_begin = e.start_offset, e_end = e.start_offset + e.extent_elem;
    if (!(in_end > e_begin && e_end > in_begin)) return TmOverlap::None;

    // L2 prerequisites — same canonical row-major axis layout.
    if (in.elem_size != e.elem_size || in.ndims != e.ndims || e.ndims == 0) return TmOverlap::Other;
    for (uint32_t i = 0; i < e.ndims; i++) {
        if (in.strides[i] != e.strides[i]) return TmOverlap::Other;
    }
    if (e.strides[e.ndims - 1] != 1) return TmOverlap::Other;
    for (uint32_t i = 1; i < e.ndims; i++) {
        if (e.strides[i - 1] % e.strides[i] != 0) return TmOverlap::Other;
    }

    // Derive reference shape A from stride: A[i] = strides[i-1]/strides[i]; A[0] from storage size.
    uint32_t ref[TM_MAX_DIMS] = {};
    for (uint32_t i = 1; i < e.ndims; i++) ref[i] = e.strides[i - 1] / e.strides[i];
    const uint32_t stride0 = e.strides[0];
    if (stride0 == 0 || in.storage_numel % stride0 != 0) return TmOverlap::Other;
    ref[0] = static_cast<uint32_t>(in.storage_numel / stride0);

    // Decompose start offsets into per-axis coords (row-major: divide by stride[i]).
    uint32_t in_off[TM_MAX_DIMS] = {}, e_off[TM_MAX_DIMS] = {};
    uint64_t in_rem = in.start_offset, e_rem = e.start_offset;
    for (uint32_t i = 0; i < e.ndims; i++) {
        const uint32_t s = e.strides[i];
        in_off[i] = static_cast<uint32_t>(in_rem / s);
        e_off[i] = static_cast<uint32_t>(e_rem / s);
        in_rem %= s;
        e_rem %= s;
    }
    if (in_rem != 0 || e_rem != 0) return TmOverlap::Other;
    for (uint32_t i = 0; i < e.ndims; i++) {
        if (uint64_t(in_off[i]) + in.shapes[i] > ref[i]) return TmOverlap::Other;
        if (uint64_t(e_off[i]) + e.shapes[i] > ref[i]) return TmOverlap::Other;
    }

    // L2 core — per-dim segment intersection; COVERED iff probe contains entry on every axis.
    bool covered = true;
    for (uint32_t i = 0; i < e.ndims; i++) {
        const uint64_t a0 = in_off[i], a1 = a0 + in.shapes[i];
        const uint64_t b0 = e_off[i], b1 = b0 + e.shapes[i];
        if (!(a1 > b0 && b1 > a0)) return TmOverlap::None;
        if (!(a0 <= b0 && b1 <= a1)) covered = false;
    }
    return covered ? TmOverlap::Covered : TmOverlap::Other;
}

class TensorMap {
public:
    // Bytes the caller must provide to init() for this config.
    static uint64_t bytes_required(const TmConfig &cfg) { return detail::layout(cfg, nullptr); }

    // Single memory dependency point: lay out and zero the map inside `base`.
    // `base` must be at least bytes_required(cfg) bytes and 64-byte aligned.
    void init(void *base, const TmConfig &cfg) {
        base_ = static_cast<uint8_t *>(base);
        TmHeader *h = hdr();
        h->cfg = cfg;
        h->next_entry_idx = 0;
        h->free_num = 0;
        for (uint32_t r = 0; r < TM_MAX_RINGS; r++) {
            h->last_alive[r] = 0;
            h->last_cleanup[r] = 0;
            h->off_task_heads[r] = 0;
        }
        detail::layout(cfg, h);

        int32_t *bk = buckets();
        for (uint32_t i = 0; i < cfg.num_buckets; i++) bk[i] = -1;
        TmEntry *pl = pool();
        std::memset(pl, 0, uint64_t(cfg.pool_size) * sizeof(TmEntry));
        for (uint32_t i = 0; i < cfg.pool_size; i++) {
            pl[i].bucket_index = -1;
            pl[i].next_in_bucket = pl[i].prev_in_bucket = -1;
            pl[i].next_in_task = pl[i].prev_in_task = -1;
        }
        for (uint32_t r = 0; r < cfg.num_rings; r++) {
            int32_t *th = task_heads(r);
            for (uint32_t i = 0; i < cfg.task_window[r]; i++) th[i] = -1;
        }
    }

    // Bind to a buffer that already holds an initialized image (no fix-up needed).
    void attach(void *base) { base_ = static_cast<uint8_t *>(base); }

    // Register a region produced by `producer_id`.
    void insert(const TmRegion &r, uint64_t producer_id) {
        const int32_t idx = new_entry();
        TmEntry &e = pool()[idx];
        e.base_addr = r.base_addr;
        e.start_offset = r.start_offset;
        e.extent_elem = r.extent_elem;
        e.version = r.version;
        e.ndims = r.ndims;
        e.elem_size = r.elem_size;
        e.is_contiguous = r.is_contiguous;
        for (uint32_t i = 0; i < r.ndims; i++) {
            e.shapes[i] = r.shapes[i];
            e.strides[i] = r.strides[i];
        }
        link_entry(idx, r.base_addr, producer_id);
    }

    // Invoke on_match(TmEntry&, TmOverlap) for each valid overlapping entry.
    // Return true to continue, false to stop early. The callback may call
    // remove(entry) on the current entry; the next link is latched beforehand.
    template <typename Fn>
    void lookup(const TmRegion &r, Fn &&on_match) {
        const uint32_t b = hash(r.base_addr);
        int32_t cur = buckets()[b];
        TmEntry *pl = pool();
        while (cur != -1) {
            const int32_t next = pl[cur].next_in_bucket;
            TmEntry &e = pl[cur];
            if (entry_valid(e) && e.base_addr == r.base_addr) {
                const TmOverlap st = tm_overlap(r, e);
                if (st != TmOverlap::None) {
                    if (!on_match(e, st)) return;
                }
            }
            cur = next;
        }
    }

    // Unlink one entry from both chains and return it to the pool.
    void remove(TmEntry &e) {
        const int32_t idx = static_cast<int32_t>(&e - pool());
        remove_from_task(idx);
        free_entry(idx);
    }

    // Advance the per-ring validity watermark.
    void sync(uint32_t ring, int32_t last_alive) { hdr()->last_alive[ring] = last_alive; }

    // Convenience for the submit hot path: advance the watermark and reclaim any
    // entries that just retired. Correctness-first (cleans up to the watermark on
    // every advance); no periodic-gating optimization.
    void sync_tensormap(uint32_t ring, int32_t last_alive) {
        sync(ring, last_alive);
        const int32_t old = hdr()->last_cleanup[ring];
        if (last_alive > old) cleanup_retired(ring, old, last_alive);
    }

    // Reclaim entries of tasks in [old_alive, new_alive) on `ring`.
    void cleanup_retired(uint32_t ring, int32_t old_alive, int32_t new_alive) {
        const uint32_t mask = hdr()->cfg.task_window[ring] - 1;
        int32_t *th = task_heads(ring);
        TmEntry *pl = pool();
        for (int32_t local = old_alive; local < new_alive; local++) {
            const uint32_t slot = static_cast<uint32_t>(local) & mask;
            int32_t cur = th[slot];
            while (cur != -1) {
                const int32_t next = pl[cur].next_in_task;
                free_entry(cur);
                cur = next;
            }
            th[slot] = -1;
        }
        hdr()->last_cleanup[ring] = new_alive;
    }

    // Producer-id encoding (owned by this module).
    static uint64_t make_id(uint32_t ring, uint32_t local) {
        return (static_cast<uint64_t>(ring) << 32) | local;
    }
    static uint32_t ring_of(uint64_t id) { return static_cast<uint32_t>(id >> 32); }
    static uint32_t local_of(uint64_t id) { return static_cast<uint32_t>(id & 0xFFFFFFFFu); }

    // Number of currently-valid (non-retired) entries — debug/testing helper.
    int32_t valid_count() const {
        const TmHeader *h = hdr();
        const TmEntry *pl = pool();
        int32_t n = 0;
        for (int32_t i = 0; i < h->next_entry_idx; i++) {
            if (pl[i].bucket_index != -1 && entry_valid(pl[i])) n++;
        }
        return n;
    }

private:
    uint8_t *base_ = nullptr;

    TmHeader *hdr() const { return reinterpret_cast<TmHeader *>(base_); }
    int32_t *buckets() const { return reinterpret_cast<int32_t *>(base_ + hdr()->off_buckets); }
    TmEntry *pool() const { return reinterpret_cast<TmEntry *>(base_ + hdr()->off_pool); }
    int32_t *free_list() const { return reinterpret_cast<int32_t *>(base_ + hdr()->off_free); }
    int32_t *task_heads(uint32_t ring) const {
        return reinterpret_cast<int32_t *>(base_ + hdr()->off_task_heads[ring]);
    }

    uint32_t hash(uint64_t key) const {
        key *= 0x9E3779B97F4A7C15ULL;
        return static_cast<uint32_t>(key >> (64 - __builtin_ctz(hdr()->cfg.num_buckets)));
    }

    bool entry_valid(const TmEntry &e) const {
        return static_cast<int32_t>(local_of(e.producer_id)) >= hdr()->last_alive[ring_of(e.producer_id)];
    }

    int32_t new_entry() {
        TmHeader *h = hdr();
        if (h->free_num > 0) return free_list()[--h->free_num];
        assert(h->next_entry_idx < static_cast<int32_t>(h->cfg.pool_size));
        return h->next_entry_idx++;
    }

    void link_entry(int32_t idx, uint64_t addr, uint64_t producer_id) {
        TmEntry *pl = pool();
        int32_t *bk = buckets();
        TmEntry &e = pl[idx];
        e.producer_id = producer_id;

        const uint32_t b = hash(addr);
        e.bucket_index = static_cast<int32_t>(b);
        e.prev_in_bucket = -1;
        e.next_in_bucket = bk[b];
        if (bk[b] != -1) pl[bk[b]].prev_in_bucket = idx;
        bk[b] = idx;

        const uint32_t ring = ring_of(producer_id);
        const uint32_t slot = local_of(producer_id) & (hdr()->cfg.task_window[ring] - 1);
        int32_t *th = task_heads(ring);
        e.prev_in_task = -1;
        e.next_in_task = th[slot];
        if (th[slot] != -1) pl[th[slot]].prev_in_task = idx;
        th[slot] = idx;
    }

    void remove_from_task(int32_t idx) {
        TmEntry *pl = pool();
        TmEntry &e = pl[idx];
        if (e.prev_in_task == -1) {
            const uint32_t ring = ring_of(e.producer_id);
            const uint32_t slot = local_of(e.producer_id) & (hdr()->cfg.task_window[ring] - 1);
            task_heads(ring)[slot] = e.next_in_task;
        } else {
            pl[e.prev_in_task].next_in_task = e.next_in_task;
        }
        if (e.next_in_task != -1) pl[e.next_in_task].prev_in_task = e.prev_in_task;
        e.next_in_task = e.prev_in_task = -1;
    }

    void free_entry(int32_t idx) {
        TmEntry *pl = pool();
        int32_t *bk = buckets();
        TmEntry &e = pl[idx];
        if (e.prev_in_bucket == -1) {
            bk[e.bucket_index] = e.next_in_bucket;
        } else {
            pl[e.prev_in_bucket].next_in_bucket = e.next_in_bucket;
        }
        if (e.next_in_bucket != -1) pl[e.next_in_bucket].prev_in_bucket = e.prev_in_bucket;

        free_list()[hdr()->free_num++] = idx;
        e.bucket_index = -1;
        e.next_in_bucket = e.prev_in_bucket = -1;
        e.next_in_task = e.prev_in_task = -1;
    }
};

}  // namespace tmap

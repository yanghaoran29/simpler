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

#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include "tensormap/tm_tensormap_c.h"

namespace {

TmConfig make_config() {
    TmConfig cfg{};
    cfg.num_buckets = 16;
    cfg.pool_size = 64;
    cfg.num_rings = 2;
    cfg.task_window[0] = 16;
    cfg.task_window[1] = 16;
    return cfg;
}

// Contiguous 1D view of `len` elements starting at element `off` inside a
// buffer of `storage` elements at base address `base`.
TmRegion region_1d(uint64_t base, uint64_t off, uint64_t len, uint64_t storage) {
    TmRegion r{};
    r.base_addr = base;
    r.start_offset = off;
    r.extent_elem = len;
    r.storage_numel = storage;
    r.elem_size = 4;
    r.ndims = 1;
    r.version = 0;
    r.is_contiguous = 1;
    r.shapes[0] = static_cast<uint32_t>(len);
    r.strides[0] = 1;
    return r;
}

// insert() takes a TmRegion by pointer; materialize the temporary first.
void insert(TmTensorMap &map, const TmRegion &r, uint64_t producer_id) {
    TmRegion tmp = r;
    tm_insert(&map, &tmp, producer_id);
}

struct Hit {
    uint64_t producer;
    TmOverlap status;
};

bool collect_cb(TmEntry *e, TmOverlap st, void *ctx) {
    static_cast<std::vector<Hit> *>(ctx)->push_back({e->producer_id, st});
    return true;
}

std::vector<Hit> collect(TmTensorMap &map, const TmRegion &probe) {
    std::vector<Hit> hits;
    TmRegion p = probe;
    tm_lookup(&map, &p, collect_cb, &hits);
    return hits;
}

bool remove_cb(TmEntry *e, TmOverlap, void *ctx) {
    tm_remove(static_cast<TmTensorMap *>(ctx), e);
    return true;
}

}  // namespace

class TmTensorMapTest : public ::testing::Test {
protected:
    void SetUp() override {
        cfg_ = make_config();
        buf_.resize(tm_bytes_required(&cfg_));
        tm_init(&map_, buf_.data(), &cfg_);
    }
    TmConfig cfg_{};
    std::vector<uint8_t> buf_;
    TmTensorMap map_{};
};

TEST_F(TmTensorMapTest, ExactRegionIsCovered) {
    const uint64_t base = 0x1000;
    insert(map_, region_1d(base, 0, 128, 128), tm_make_id(0, 0));
    auto hits = collect(map_, region_1d(base, 0, 128, 128));
    ASSERT_EQ(hits.size(), 1u);
    EXPECT_EQ(hits[0].producer, tm_make_id(0, 0));
    EXPECT_EQ(hits[0].status, TM_OVERLAP_COVERED);
}

TEST_F(TmTensorMapTest, DisjointRegionsDoNotMatch) {
    const uint64_t base = 0x2000;
    insert(map_, region_1d(base, 0, 128, 256), tm_make_id(0, 1));
    auto hits = collect(map_, region_1d(base, 128, 128, 256));  // [128,256) vs [0,128)
    EXPECT_TRUE(hits.empty());
}

TEST_F(TmTensorMapTest, PartialOverlapIsOther) {
    const uint64_t base = 0x3000;
    insert(map_, region_1d(base, 0, 128, 256), tm_make_id(0, 2));
    auto hits = collect(map_, region_1d(base, 64, 128, 256));  // [64,192) partially covers [0,128)
    ASSERT_EQ(hits.size(), 1u);
    EXPECT_EQ(hits[0].status, TM_OVERLAP_OTHER);
}

TEST_F(TmTensorMapTest, DifferentBaseAddressIsolated) {
    insert(map_, region_1d(0x4000, 0, 128, 128), tm_make_id(0, 3));
    auto hits = collect(map_, region_1d(0x5000, 0, 128, 128));
    EXPECT_TRUE(hits.empty());
}

TEST_F(TmTensorMapTest, LazyInvalidationSkipsRetiredProducer) {
    const uint64_t base = 0x6000;
    insert(map_, region_1d(base, 0, 128, 128), tm_make_id(/*ring=*/0, /*local=*/5));
    EXPECT_EQ(tm_valid_count(&map_), 1);

    tm_sync(&map_, /*ring=*/0, /*last_alive=*/6);  // local 5 < 6 → retired
    EXPECT_EQ(tm_valid_count(&map_), 0);
    EXPECT_TRUE(collect(map_, region_1d(base, 0, 128, 128)).empty());
}

TEST_F(TmTensorMapTest, CleanupRetiredReclaimsPool) {
    const uint64_t base = 0x7000;
    for (uint32_t local = 0; local < 8; local++) {
        insert(map_, region_1d(base, 0, 16, 128), tm_make_id(0, local));
    }
    tm_sync(&map_, 0, 8);
    tm_cleanup_retired(&map_, 0, 0, 8);
    EXPECT_EQ(tm_valid_count(&map_), 0);

    // Pool slots are reusable: a fresh insert is found again.
    tm_sync(&map_, 0, 8);
    insert(map_, region_1d(base, 0, 16, 128), tm_make_id(0, 8));
    auto hits = collect(map_, region_1d(base, 0, 16, 128));
    ASSERT_EQ(hits.size(), 1u);
    EXPECT_EQ(hits[0].producer, tm_make_id(0, 8));
}

TEST_F(TmTensorMapTest, RemoveFromCallback) {
    const uint64_t base = 0x8000;
    insert(map_, region_1d(base, 0, 64, 64), tm_make_id(0, 0));
    TmRegion probe = region_1d(base, 0, 64, 64);
    tm_lookup(&map_, &probe, remove_cb, &map_);
    EXPECT_TRUE(collect(map_, region_1d(base, 0, 64, 64)).empty());
}

TEST_F(TmTensorMapTest, AttachToRelocatedImage) {
    const uint64_t base = 0x9000;
    insert(map_, region_1d(base, 0, 128, 128), tm_make_id(1, 7));

    // memcpy the whole image into a second buffer; indices/offsets are
    // position-independent so attach() alone rebinds it.
    std::vector<uint8_t> buf2(buf_.size());
    std::memcpy(buf2.data(), buf_.data(), buf_.size());
    TmTensorMap map2{};
    tm_attach(&map2, buf2.data());

    auto hits = collect(map2, region_1d(base, 0, 128, 128));
    ASSERT_EQ(hits.size(), 1u);
    EXPECT_EQ(hits[0].producer, tm_make_id(1, 7));
    EXPECT_EQ(hits[0].status, TM_OVERLAP_COVERED);
}

TEST_F(TmTensorMapTest, MultiProducerSameBucketAllReported) {
    const uint64_t base = 0xA000;
    insert(map_, region_1d(base, 0, 64, 256), tm_make_id(0, 0));
    insert(map_, region_1d(base, 0, 64, 256), tm_make_id(0, 1));
    auto hits = collect(map_, region_1d(base, 0, 64, 256));
    EXPECT_EQ(hits.size(), 2u);
}

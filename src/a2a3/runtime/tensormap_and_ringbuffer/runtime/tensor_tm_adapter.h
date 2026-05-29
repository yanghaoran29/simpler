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
 * Bridges the runtime Tensor onto the standalone tm_tensormap's TmRegion POD.
 * The map itself stays dependency-free; this adapter is the runtime-side glue.
 */

#pragma once

#include <type_traits>

#include "tensor.h"
#include "tm_tensormap_c.h"

inline TmRegion to_tm_region(const Tensor &t) {
    TmRegion r{};
    r.base_addr = t.buffer.addr;
    r.start_offset = t.start_offset;
    r.extent_elem = t.extent_elem();
    const uint32_t es = get_element_size(t.dtype);
    r.storage_numel = (es != 0) ? (t.buffer.size / es) : 0;
    r.elem_size = es;
    r.ndims = t.ndims;
    r.version = t.version;
    r.is_contiguous = t.is_contiguous;
    for (uint32_t i = 0; i < t.ndims && i < TM_MAX_DIMS; i++) {
        r.shapes[i] = t.shapes[i];
        r.strides[i] = t.strides[i];
    }
    return r;
}

// C++ ergonomic wrapper over the C tm_lookup: forwards a callable
// (TmEntry&, TmOverlap) -> bool to the function-pointer + context callback.
// The callable may call tm_remove(&map, &entry) on the current entry.
template <typename Fn>
inline void tm_lookup_each(TmTensorMap &map, const TmRegion &region, Fn &&fn) {
    TmRegion r = region;  // stable address for the C API
    TmMatchFn cb = [](TmEntry *e, TmOverlap st, void *ctx) -> bool {
        return (*static_cast<std::remove_reference_t<Fn> *>(ctx))(*e, st);
    };
    tm_lookup(&map, &r, cb, static_cast<void *>(&fn));
}

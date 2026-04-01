#pragma once

#include "tensor.h"

/**
 * Create a Tensor for pre-allocated external memory.
 */
inline Tensor make_tensor_external(void* addr,
    const uint32_t shapes[],
    uint32_t ndims,
    DataType dtype = DataType::FLOAT32,
    bool manual_dep = false,
    int32_t version = 0) {
    static uint32_t zero_offsets[RUNTIME_MAX_TENSOR_DIMS] = {};
    uint64_t total = 1;
    for (uint32_t i = 0; i < ndims; i++) {
        total *= shapes[i];
    }
    return Tensor(addr, total * get_element_size(dtype), shapes, shapes, zero_offsets, ndims, dtype, version,
                  /*is_all_offset_zero=*/true, /*is_raw_eq_shapes=*/true, manual_dep);
}

/**
 * Create an unallocated Tensor (buffer addr = nullptr) with given shape/dtype.
 */
inline Tensor make_tensor(const uint32_t* shapes, uint32_t ndims, DataType dtype) {
    return make_tensor_external(nullptr, shapes, ndims, dtype);
}


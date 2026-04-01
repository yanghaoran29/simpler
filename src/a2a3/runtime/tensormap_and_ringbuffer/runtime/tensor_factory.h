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
    uint64_t total = 1;
    for (uint32_t i = 0; i < ndims; i++) {
        total *= shapes[i];
    }
    Tensor t;
    t.init_external(addr, total * get_element_size(dtype), shapes, ndims, dtype, version, manual_dep);
    return t;
}

/**
 * Create an unallocated Tensor (buffer addr = nullptr) with given shape/dtype.
 */
inline Tensor make_tensor(const uint32_t* shapes, uint32_t ndims, DataType dtype) {
    return make_tensor_external(nullptr, shapes, ndims, dtype);
}


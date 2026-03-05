/**
 * Kernel dispatch stub for builds without statically-linked kernels.
 *
 * This stub is used when building host_runtime.so without user kernels
 * (e.g., during pytest integration tests). In production, the generated
 * kernel_dispatch.cpp replaces this stub via the KERNEL_DISPATCH_SOURCE
 * CMake parameter.
 */

#include <stdint.h>

extern "C" uint64_t get_kernel_func_addr(int func_id) {
    (void)func_id;
    return 0;
}

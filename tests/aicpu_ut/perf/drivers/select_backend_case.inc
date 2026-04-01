#if PERF_BACKEND == 1
#include "test_paged_attention.cpp"
#elif PERF_BACKEND == 2
#include "test_batch_paged_attention.cpp"
#elif PERF_BACKEND == 4
#include "test_alternating_matmul_add.cpp"
#elif PERF_BACKEND == 5
#include "test_benchmark_bgemm.cpp"
#elif PERF_BACKEND == 6
#include "test_paged_attention_unroll.cpp"
#elif PERF_BACKEND == 7
#include "test_throughput.cpp"
#elif PERF_BACKEND == 8
#include "test_latency.cpp"
#elif PERF_BACKEND == 9
#include "test_spmd_basic.cpp"
#elif PERF_BACKEND == 10
#include "test_spmd_multiblock_aiv.cpp"
#elif PERF_BACKEND == 11
#include "test_spmd_multiblock_mix.cpp"
#else
#error "Unsupported PERF_BACKEND"
#endif

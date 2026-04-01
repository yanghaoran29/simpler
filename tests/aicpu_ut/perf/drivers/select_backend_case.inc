#if PERF_BACKEND == 12
#include "paged_consumer_block_table.cpp"
#elif PERF_BACKEND == 13
#include "qwen3_decode_tensormap.cpp"
#elif PERF_BACKEND == 14
#include "qwen3_decode_manual_scope.cpp"
#else
#error "Unsupported PERF_BACKEND"
#endif

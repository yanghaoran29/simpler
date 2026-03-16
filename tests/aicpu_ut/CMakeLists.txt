# ─────────────────────────────────────────────────────────────────────────────
# CMakeLists.txt — PTO Runtime2 Orchestrator-Scheduler simulation tests
#
# Usage:
#   mkdir build && cd build
#   cmake ..                         # configure (discovers *.a from system)
#   make                             # build all test binaries
#   make test                        # run all tests via ctest
#
# Optional overrides:
#   cmake .. -DORCH_CPU=4 -DSCHED_CPU0=5 ...
#   cmake .. -DPLATFORM_MAX_BLOCKDIM=32 -DPLATFORM_AIV_CORES_PER_BLOCKDIM=3
#   cmake .. -DOPT_LEVEL=0           # debug (no optimization)
#   cmake .. -DPTO2_PROFILING=OFF
# ─────────────────────────────────────────────────────────────────────────────

cmake_minimum_required(VERSION 3.15)
project(orchestration_ut CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Place all test binaries under build/bin/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ─── CPU affinity parameters ──────────────────────────────────────────────────
set(ORCH_CPU   "4"  CACHE STRING "Orchestrator CPU core")
set(SCHED_CPU0 "8"  CACHE STRING "Scheduler CPU core 0")
set(SCHED_CPU1 "9"  CACHE STRING "Scheduler CPU core 1")
set(SCHED_CPU2 "10" CACHE STRING "Scheduler CPU core 2")
set(SCHED_CPU3 "11" CACHE STRING "Scheduler CPU core 3")
set(SCHED_CPU4 "12" CACHE STRING "Scheduler CPU core 4")
set(SCHED_CPU5 "13" CACHE STRING "Scheduler CPU core 5")
set(SCHED_CPU6 "14" CACHE STRING "Scheduler CPU core 6")
set(SCHED_CPU7 "15" CACHE STRING "Scheduler CPU core 7")

# ─── Platform configuration parameters ────────────────────────────────────────
set(PLATFORM_MAX_BLOCKDIM            "24" CACHE STRING "Max block dim")
set(PLATFORM_AIC_CORES_PER_BLOCKDIM  "1"  CACHE STRING "AIC cores per blockdim")
set(PLATFORM_AIV_CORES_PER_BLOCKDIM  "2"  CACHE STRING "AIV cores per blockdim")
set(PLATFORM_MAX_AICPU_THREADS       "4"  CACHE STRING "Max AICPU threads")

# ─── Profiling control ────────────────────────────────────────────────────────
# PTO2_PROFILING: master switch (typically ON when either SCHED or ORCH is ON).
# PTO2_SCHED_PROFILING / PTO2_ORCH_PROFILING: can be toggled independently via run_tests.sh.
option(PTO2_PROFILING "Enable PTO2 profiling instrumentation" ON)
option(PTO2_SCHED_PROFILING "Enable scheduler profiling" ON)
option(PTO2_ORCH_PROFILING "Enable orchestrator profiling" ON)

# ─── Sim AICore / 0-core paths (for aicpu_ut perf) ─────────────────────────────
# When ON: enables cores_total_num_==0, sim_aicore_mode, core_num==0 drain loop in executor.
# Define PTO2_SIM_AICORE_UT=1 so that executor/Runtime sim branches are compiled in.
# Default ON for aicpu_ut so that perf tests get current behavior when executor is linked.
option(PTO2_SIM_AICORE_UT "Enable sim AICore and 0-core paths (aicpu_ut perf)" ON)

# ─── Source directories ────────────────────────────────────────────────────────
set(ROOT_DIR    ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(RUNTIME_DIR ${ROOT_DIR}/src/a2a3/runtime/tensormap_and_ringbuffer/runtime)
set(AICPU_DIR   ${ROOT_DIR}/src/a2a3/runtime/tensormap_and_ringbuffer/aicpu)
set(ORCH_DIR    ${ROOT_DIR}/src/a2a3/runtime/tensormap_and_ringbuffer/orchestration)
set(COMMON_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/common)
set(FUNC_DIR    ${CMAKE_CURRENT_SOURCE_DIR}/functional)
set(PERF_DIR    ${CMAKE_CURRENT_SOURCE_DIR}/perf)
set(PLATFORM_AICPU_DIR ${ROOT_DIR}/src/a2a3/platform/src/aicpu)

# ─── Common include directories ───────────────────────────────────────────────
set(COMMON_INCLUDES
    ${RUNTIME_DIR}
    ${ROOT_DIR}/src/a2a3/platform/include
    ${ORCH_DIR}
    ${COMMON_DIR}
)
if(PTO2_SIM_AICORE_UT)
    list(APPEND COMMON_INCLUDES ${AICPU_DIR} ${ROOT_DIR}/src/a2a3/platform/sim/aicpu)
endif()

# ─── Common compile definitions ───────────────────────────────────────────────
set(COMMON_COMPILE_DEFS
    ORCH_CPU=${ORCH_CPU}
    SCHED_CPU0=${SCHED_CPU0}
    SCHED_CPU1=${SCHED_CPU1}
    SCHED_CPU2=${SCHED_CPU2}
    SCHED_CPU3=${SCHED_CPU3}
    SCHED_CPU4=${SCHED_CPU4}
    SCHED_CPU5=${SCHED_CPU5}
    SCHED_CPU6=${SCHED_CPU6}
    SCHED_CPU7=${SCHED_CPU7}
    PLATFORM_MAX_BLOCKDIM=${PLATFORM_MAX_BLOCKDIM}
    PLATFORM_AIC_CORES_PER_BLOCKDIM=${PLATFORM_AIC_CORES_PER_BLOCKDIM}
    PLATFORM_AIV_CORES_PER_BLOCKDIM=${PLATFORM_AIV_CORES_PER_BLOCKDIM}
    PLATFORM_MAX_AICPU_THREADS=${PLATFORM_MAX_AICPU_THREADS}
)
if(PTO2_PROFILING)
    list(APPEND COMMON_COMPILE_DEFS PTO2_PROFILING=1)
else()
    list(APPEND COMMON_COMPILE_DEFS PTO2_PROFILING=0)
endif()
if(PTO2_SCHED_PROFILING)
    list(APPEND COMMON_COMPILE_DEFS PTO2_SCHED_PROFILING=1)
endif()
if(PTO2_ORCH_PROFILING)
    list(APPEND COMMON_COMPILE_DEFS PTO2_ORCH_PROFILING=1)
endif()
if(PTO2_SIM_AICORE_UT)
    list(APPEND COMMON_COMPILE_DEFS PTO2_SIM_AICORE_UT=1)
endif()

# ─── Optimization level ───────────────────────────────────────────────────────
set(OPT_LEVEL "3" CACHE STRING "Compiler optimization level (0/1/2/3)")

# ─── Latency case compile-time overrides (run_tests.sh --chain-num/--chain-length/--latency-nelems) ───
# When set, these are compiled into the latency test binaries; PERF_CASES + getenv are ignored for that param.
set(AICPU_UT_LATENCY_NUM_CHAINS    "" CACHE STRING "Latency: num_chains (1-128) at compile time")
set(AICPU_UT_LATENCY_CHAIN_LENGTH   "" CACHE STRING "Latency: chain_length (1-256) at compile time")
set(LATENCY_EXTRA_DEFS "")
if(NOT AICPU_UT_LATENCY_NUM_CHAINS STREQUAL "")
    list(APPEND LATENCY_EXTRA_DEFS "AICPU_UT_LATENCY_NUM_CHAINS=${AICPU_UT_LATENCY_NUM_CHAINS}")
endif()
if(NOT AICPU_UT_LATENCY_CHAIN_LENGTH STREQUAL "")
    list(APPEND LATENCY_EXTRA_DEFS "AICPU_UT_LATENCY_CHAIN_LENGTH=${AICPU_UT_LATENCY_CHAIN_LENGTH}")
endif()

# ─── Throughput case compile-time overrides (run_tests.sh --layer-num/--layer0-task-num/etc.) ───
set(AICPU_UT_THROUGHPUT_LAYERS        "" CACHE STRING "Throughput: num_layers at compile time")
set(AICPU_UT_THROUGHPUT_LAYER0_SIZE   "" CACHE STRING "Throughput: layer0_size at compile time")
set(AICPU_UT_THROUGHPUT_DEPS          "" CACHE STRING "Throughput: deps_per_task at compile time")
set(AICPU_UT_THROUGHPUT_OVERLAP       "" CACHE STRING "Throughput: overlap at compile time")
set(AICPU_UT_NUM_SCHED_THREADS        "" CACHE STRING "Number of scheduler threads at compile time")
set(THROUGHPUT_EXTRA_DEFS "")
if(NOT AICPU_UT_THROUGHPUT_LAYERS STREQUAL "")
    list(APPEND THROUGHPUT_EXTRA_DEFS "AICPU_UT_THROUGHPUT_LAYERS=${AICPU_UT_THROUGHPUT_LAYERS}")
endif()
if(NOT AICPU_UT_THROUGHPUT_LAYER0_SIZE STREQUAL "")
    list(APPEND THROUGHPUT_EXTRA_DEFS "AICPU_UT_THROUGHPUT_LAYER0_SIZE=${AICPU_UT_THROUGHPUT_LAYER0_SIZE}")
endif()
if(NOT AICPU_UT_THROUGHPUT_DEPS STREQUAL "")
    list(APPEND THROUGHPUT_EXTRA_DEFS "AICPU_UT_THROUGHPUT_DEPS=${AICPU_UT_THROUGHPUT_DEPS}")
endif()
if(NOT AICPU_UT_THROUGHPUT_OVERLAP STREQUAL "")
    list(APPEND THROUGHPUT_EXTRA_DEFS "AICPU_UT_THROUGHPUT_OVERLAP=${AICPU_UT_THROUGHPUT_OVERLAP}")
endif()
if(NOT AICPU_UT_NUM_SCHED_THREADS STREQUAL "")
    list(APPEND THROUGHPUT_EXTRA_DEFS "AICPU_UT_NUM_SCHED_THREADS=${AICPU_UT_NUM_SCHED_THREADS}")
endif()


set(COMMON_COMPILE_OPTIONS
    -O${OPT_LEVEL}
    -ggdb3
    -fno-omit-frame-pointer
    -Wall
    -Wextra
    -Wno-unused-parameter
    -Wno-class-memaccess
)

# ─── Runtime source files ─────────────────────────────────────────────────────
set(RUNTIME_SRCS
    ${RUNTIME_DIR}/pto_runtime2.cpp
    ${RUNTIME_DIR}/pto_orchestrator.cpp
    ${RUNTIME_DIR}/pto_scheduler.cpp
    ${RUNTIME_DIR}/pto_ring_buffer.cpp
    ${RUNTIME_DIR}/pto_shared_memory.cpp
    ${RUNTIME_DIR}/pto_tensormap.cpp
    ${ORCH_DIR}/common.cpp
    ${ROOT_DIR}/src/a2a3/platform/sim/aicpu/device_time.cpp
)
if(PTO2_SIM_AICORE_UT)
    list(APPEND RUNTIME_SRCS
        ${RUNTIME_DIR}/runtime.cpp
        ${AICPU_DIR}/aicpu_executor.cpp
        ${AICPU_DIR}/sim_aicore.cpp
        ${PLATFORM_AICPU_DIR}/platform_regs.cpp
        ${PLATFORM_AICPU_DIR}/performance_collector_aicpu.cpp
        ${ROOT_DIR}/src/a2a3/platform/sim/aicpu/cache_ops.cpp
    )
endif()

# ─── Common test source files ─────────────────────────────────────────────────
set(COMMON_TEST_SRCS
    ${COMMON_DIR}/test_common.cpp
    ${COMMON_DIR}/test_log_stubs.cpp
    ${COMMON_DIR}/cpu_affinity.cpp
)

# ─── Discover static libraries from the current system ────────────────────────

# GCC internal libraries: query from compiler
execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libgcc.a
    OUTPUT_VARIABLE STATIC_LIBGCC
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libgcc_eh.a
    OUTPUT_VARIABLE STATIC_LIBGCC_EH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# libstdc++.a: prefer CANN toolkit version (aarch64 cross-build), fall back to system
file(GLOB LIBSTDCXX_CANDIDATES
    "$ENV{HOME}/Ascend/cann-*/tools/hcc/aarch64-target-linux-gnu/lib64/libstdc++.a"
    "$ENV{HOME}/Ascend/cann-*/tools/hcc/lib64/libstdc++.a"
    "/opt/Ascend/cann-*/tools/hcc/aarch64-target-linux-gnu/lib64/libstdc++.a"
)
if(LIBSTDCXX_CANDIDATES)
    list(SORT LIBSTDCXX_CANDIDATES ORDER DESCENDING)
    list(GET LIBSTDCXX_CANDIDATES 0 STATIC_LIBSTDCXX)
else()
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libstdc++.a
        OUTPUT_VARIABLE STATIC_LIBSTDCXX
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    # Compiler returns bare "libstdc++.a" when it doesn't know the path — fall
    # back to a global filesystem search across common GCC install trees.
    if(NOT IS_ABSOLUTE "${STATIC_LIBSTDCXX}")
        file(GLOB_RECURSE LIBSTDCXX_GLOBAL
            "/usr/lib/gcc/*/libstdc++.a"
            "/usr/lib64/gcc/*/libstdc++.a"
            "/usr/local/lib/gcc/*/libstdc++.a"
            "/usr/lib/libstdc++.a"
            "/usr/lib64/libstdc++.a"
        )
        if(LIBSTDCXX_GLOBAL)
            list(SORT LIBSTDCXX_GLOBAL ORDER DESCENDING)
            list(GET LIBSTDCXX_GLOBAL 0 STATIC_LIBSTDCXX)
        else()
            message(WARNING "libstdc++.a not found anywhere — linking may fail")
            set(STATIC_LIBSTDCXX "")
        endif()
    endif()
endif()

# System static libraries: search standard locations, prefer .a over .so
set(_saved_suffixes ${CMAKE_FIND_LIBRARY_SUFFIXES})
set(CMAKE_FIND_LIBRARY_SUFFIXES .a)

set(SYSTEM_LIB_PATHS /usr/lib64 /usr/lib /lib64 /lib)
find_library(STATIC_LIBM      m       PATHS ${SYSTEM_LIB_PATHS} NO_DEFAULT_PATH)
find_library(STATIC_LIBC      c       PATHS ${SYSTEM_LIB_PATHS} NO_DEFAULT_PATH)
find_library(STATIC_LIBPTHREAD pthread PATHS ${SYSTEM_LIB_PATHS} NO_DEFAULT_PATH)
find_library(STATIC_LIBDL     dl      PATHS ${SYSTEM_LIB_PATHS} NO_DEFAULT_PATH)

set(CMAKE_FIND_LIBRARY_SUFFIXES ${_saved_suffixes})

# ─── Report discovered libraries ──────────────────────────────────────────────
message(STATUS "Static libraries discovered:")
message(STATUS "  libstdc++.a : ${STATIC_LIBSTDCXX}")
message(STATUS "  libm.a      : ${STATIC_LIBM}")
message(STATUS "  libc.a      : ${STATIC_LIBC}")
message(STATUS "  libpthread.a: ${STATIC_LIBPTHREAD}")
message(STATUS "  libdl.a     : ${STATIC_LIBDL}")
message(STATUS "  libgcc.a    : ${STATIC_LIBGCC}")
message(STATUS "  libgcc_eh.a : ${STATIC_LIBGCC_EH}")

# Warn if any required library is missing
foreach(_lib STATIC_LIBM STATIC_LIBC STATIC_LIBPTHREAD STATIC_LIBDL)
    if(NOT ${_lib})
        message(WARNING "${_lib} not found — linking may fail")
    endif()
endforeach()

# ─── Shared object library: runtime + common test code ────────────────────────
# OBJECT library avoids re-compiling these sources for every test binary.
add_library(runtime_common OBJECT
    ${RUNTIME_SRCS}
    ${COMMON_TEST_SRCS}
)
target_include_directories(runtime_common PUBLIC ${COMMON_INCLUDES})
target_compile_definitions(runtime_common PUBLIC ${COMMON_COMPILE_DEFS})
target_compile_options(runtime_common PRIVATE ${COMMON_COMPILE_OPTIONS})

# ─── Helper: configure a test executable ──────────────────────────────────────
# Each test binary links against the shared OBJECT library and the static system
# libraries with nodefaultlibs + group linking to resolve circular references.
function(configure_test_target target_name)
    target_include_directories(${target_name} PRIVATE ${COMMON_INCLUDES})
    target_compile_definitions(${target_name} PRIVATE ${COMMON_COMPILE_DEFS})
    target_compile_options(${target_name} PRIVATE ${COMMON_COMPILE_OPTIONS})
    # All link specs go into target_link_libraries so they appear AFTER object
    # files in the linker command.  target_link_options puts flags BEFORE object
    # files, which causes static libs to be scanned before any undefined
    # references exist — making all symbols unavailable.
    target_link_libraries(${target_name} PRIVATE
        runtime_common
        -nodefaultlibs
        -Wl,--start-group
        ${STATIC_LIBSTDCXX}
        ${STATIC_LIBM}
        ${STATIC_LIBPTHREAD}
        ${STATIC_LIBDL}
        ${STATIC_LIBC}
        ${STATIC_LIBGCC}
        ${STATIC_LIBGCC_EH}
        -Wl,--end-group
    )
endfunction()

# ─── Test executable 1: CPU affinity functional tests ─────────────────────────
add_executable(test_cpu_affinity
    ${FUNC_DIR}/test_cpu_affinity.cpp
)
configure_test_target(test_cpu_affinity)

# ─── Test executable 2: Platform configuration functional tests ───────────────
add_executable(test_platform_config
    ${FUNC_DIR}/test_platform_config.cpp
)
configure_test_target(test_platform_config)

# ─── Test executables: perf tests (one binary per mode per backend per case; PERF_BACKEND at compile time) ─
set(PERF_DRIVERS ${PERF_DIR}/drivers)
set(PERF_CASES  ${PERF_DIR}/cases)

set(PA_CASE_INDICES 0)
set(BPA_CASE_INDICES 0 1 2)
set(LINEAR_CASE_INDICES 0 1 2)
set(DEG_CASE_INDICES 0 1 2)
set(ALT_CASE_INDICES 0 1)
set(BGEMM_CASE_INDICES 0 1 2 3 4)
set(PAU_CASE_INDICES 0 1 2)
set(THROUGHPUT_CASE_INDICES 0 1 2)
set(LATENCY_CASE_INDICES 0 1)

# Perf tests: one driver .cpp per target; it #includes the backend .cpp (cases/test_*.cpp, 以 test 开头、去掉结尾 case、其余不变), so do NOT add cases/test_*.cpp as source.
foreach(i IN LISTS LINEAR_CASE_INDICES)
    set(target test_linear_sched_prof_only_${i})
    add_executable(${target} ${PERF_DRIVERS}/test_sched_prof_only.cpp)
    configure_test_target(${target})
    target_include_directories(${target} PRIVATE ${PERF_DRIVERS} ${PERF_CASES})
    target_compile_definitions(${target} PRIVATE PERF_BACKEND=0 PERF_CASE_IDX=${i})
endforeach()
foreach(i IN LISTS PA_CASE_INDICES)
    set(target test_pa_sched_prof_only_${i})
    add_executable(${target} ${PERF_DRIVERS}/test_sched_prof_only.cpp)
    configure_test_target(${target})
    target_include_directories(${target} PRIVATE ${PERF_DRIVERS} ${PERF_CASES})
    target_compile_definitions(${target} PRIVATE PERF_BACKEND=1 PERF_CASE_IDX=${i})
endforeach()
foreach(i IN LISTS BPA_CASE_INDICES)
    set(target test_sched_prof_only_${i})
    add_executable(${target} ${PERF_DRIVERS}/test_sched_prof_only.cpp)
    configure_test_target(${target})
    target_include_directories(${target} PRIVATE ${PERF_DRIVERS} ${PERF_CASES})
    target_compile_definitions(${target} PRIVATE PERF_BACKEND=2 PERF_CASE_IDX=${i})
endforeach()
foreach(i IN LISTS DEG_CASE_INDICES)
    set(target test_deg_sched_prof_only_${i})
    add_executable(${target} ${PERF_DRIVERS}/test_sched_prof_only.cpp)
    configure_test_target(${target})
    target_include_directories(${target} PRIVATE ${PERF_DRIVERS} ${PERF_CASES})
    target_compile_definitions(${target} PRIVATE PERF_BACKEND=3 PERF_CASE_IDX=${i})
endforeach()
foreach(i IN LISTS ALT_CASE_INDICES)
    set(target test_alt_sched_prof_only_${i})
    add_executable(${target} ${PERF_DRIVERS}/test_sched_prof_only.cpp)
    configure_test_target(${target})
    target_include_directories(${target} PRIVATE ${PERF_DRIVERS} ${PERF_CASES})
    target_compile_definitions(${target} PRIVATE PERF_BACKEND=4 PERF_CASE_IDX=${i})
endforeach()
foreach(i IN LISTS BGEMM_CASE_INDICES)
    set(target test_bgemm_sched_prof_only_${i})
    add_executable(${target} ${PERF_DRIVERS}/test_sched_prof_only.cpp)
    configure_test_target(${target})
    target_include_directories(${target} PRIVATE ${PERF_DRIVERS} ${PERF_CASES})
    target_compile_definitions(${target} PRIVATE PERF_BACKEND=5 PERF_CASE_IDX=${i})
endforeach()
foreach(i IN LISTS PAU_CASE_INDICES)
    set(target test_pau_sched_prof_only_${i})
    add_executable(${target} ${PERF_DRIVERS}/test_sched_prof_only.cpp)
    configure_test_target(${target})
    target_include_directories(${target} PRIVATE ${PERF_DRIVERS} ${PERF_CASES})
    target_compile_definitions(${target} PRIVATE PERF_BACKEND=6 PERF_CASE_IDX=${i})
endforeach()
foreach(i IN LISTS THROUGHPUT_CASE_INDICES)
    set(target test_throughput_sched_prof_only_${i})
    add_executable(${target} ${PERF_DRIVERS}/test_sched_prof_only.cpp)
    configure_test_target(${target})
    target_include_directories(${target} PRIVATE ${PERF_DRIVERS} ${PERF_CASES})
    target_compile_definitions(${target} PRIVATE PERF_BACKEND=7 PERF_CASE_IDX=${i} ${THROUGHPUT_EXTRA_DEFS})
endforeach()
foreach(i IN LISTS LATENCY_CASE_INDICES)
    set(target test_latency_sched_prof_only_${i})
    add_executable(${target} ${PERF_DRIVERS}/test_sched_prof_only.cpp)
    configure_test_target(${target})
    target_include_directories(${target} PRIVATE ${PERF_DRIVERS} ${PERF_CASES})
    target_compile_definitions(${target} PRIVATE PERF_BACKEND=8 PERF_CASE_IDX=${i} ${LATENCY_EXTRA_DEFS})
endforeach()

foreach(i IN LISTS LINEAR_CASE_INDICES)
    set(target test_linear_orch_only_${i})
    add_executable(${target} ${PERF_DRIVERS}/test_orch_only.cpp)
    configure_test_target(${target})
    target_include_directories(${target} PRIVATE ${PERF_DRIVERS} ${PERF_CASES})
    target_compile_definitions(${target} PRIVATE PERF_BACKEND=0 PERF_CASE_IDX=${i})
endforeach()
foreach(i IN LISTS PA_CASE_INDICES)
    set(target test_pa_orch_only_${i})
    add_executable(${target} ${PERF_DRIVERS}/test_orch_only.cpp)
    configure_test_target(${target})
    target_include_directories(${target} PRIVATE ${PERF_DRIVERS} ${PERF_CASES})
    target_compile_definitions(${target} PRIVATE PERF_BACKEND=1 PERF_CASE_IDX=${i})
endforeach()
foreach(i IN LISTS BPA_CASE_INDICES)
    set(target test_orch_only_${i})
    add_executable(${target} ${PERF_DRIVERS}/test_orch_only.cpp)
    configure_test_target(${target})
    target_include_directories(${target} PRIVATE ${PERF_DRIVERS} ${PERF_CASES})
    target_compile_definitions(${target} PRIVATE PERF_BACKEND=2 PERF_CASE_IDX=${i})
endforeach()
foreach(i IN LISTS DEG_CASE_INDICES)
    set(target test_deg_orch_only_${i})
    add_executable(${target} ${PERF_DRIVERS}/test_orch_only.cpp)
    configure_test_target(${target})
    target_include_directories(${target} PRIVATE ${PERF_DRIVERS} ${PERF_CASES})
    target_compile_definitions(${target} PRIVATE PERF_BACKEND=3 PERF_CASE_IDX=${i})
endforeach()
foreach(i IN LISTS ALT_CASE_INDICES)
    set(target test_alt_orch_only_${i})
    add_executable(${target} ${PERF_DRIVERS}/test_orch_only.cpp)
    configure_test_target(${target})
    target_include_directories(${target} PRIVATE ${PERF_DRIVERS} ${PERF_CASES})
    target_compile_definitions(${target} PRIVATE PERF_BACKEND=4 PERF_CASE_IDX=${i})
endforeach()
foreach(i IN LISTS BGEMM_CASE_INDICES)
    set(target test_bgemm_orch_only_${i})
    add_executable(${target} ${PERF_DRIVERS}/test_orch_only.cpp)
    configure_test_target(${target})
    target_include_directories(${target} PRIVATE ${PERF_DRIVERS} ${PERF_CASES})
    target_compile_definitions(${target} PRIVATE PERF_BACKEND=5 PERF_CASE_IDX=${i})
endforeach()
foreach(i IN LISTS PAU_CASE_INDICES)
    set(target test_pau_orch_only_${i})
    add_executable(${target} ${PERF_DRIVERS}/test_orch_only.cpp)
    configure_test_target(${target})
    target_include_directories(${target} PRIVATE ${PERF_DRIVERS} ${PERF_CASES})
    target_compile_definitions(${target} PRIVATE PERF_BACKEND=6 PERF_CASE_IDX=${i})
endforeach()
foreach(i IN LISTS THROUGHPUT_CASE_INDICES)
    set(target test_throughput_orch_only_${i})
    add_executable(${target} ${PERF_DRIVERS}/test_orch_only.cpp)
    configure_test_target(${target})
    target_include_directories(${target} PRIVATE ${PERF_DRIVERS} ${PERF_CASES})
    target_compile_definitions(${target} PRIVATE PERF_BACKEND=7 PERF_CASE_IDX=${i} ${THROUGHPUT_EXTRA_DEFS})
endforeach()
foreach(i IN LISTS LATENCY_CASE_INDICES)
    set(target test_latency_orch_only_${i})
    add_executable(${target} ${PERF_DRIVERS}/test_orch_only.cpp)
    configure_test_target(${target})
    target_include_directories(${target} PRIVATE ${PERF_DRIVERS} ${PERF_CASES})
    target_compile_definitions(${target} PRIVATE PERF_BACKEND=8 PERF_CASE_IDX=${i} ${LATENCY_EXTRA_DEFS})
endforeach()

foreach(i IN LISTS LINEAR_CASE_INDICES)
    set(target test_linear_concurrent_${i})
    add_executable(${target} ${PERF_DRIVERS}/test_concurrent.cpp)
    configure_test_target(${target})
    target_include_directories(${target} PRIVATE ${PERF_DRIVERS} ${PERF_CASES})
    target_compile_definitions(${target} PRIVATE PERF_BACKEND=0 PERF_CASE_IDX=${i})
endforeach()
foreach(i IN LISTS PA_CASE_INDICES)
    set(target test_pa_concurrent_${i})
    add_executable(${target} ${PERF_DRIVERS}/test_concurrent.cpp)
    configure_test_target(${target})
    target_include_directories(${target} PRIVATE ${PERF_DRIVERS} ${PERF_CASES})
    target_compile_definitions(${target} PRIVATE PERF_BACKEND=1 PERF_CASE_IDX=${i})
endforeach()
foreach(i IN LISTS BPA_CASE_INDICES)
    set(target test_concurrent_${i})
    add_executable(${target} ${PERF_DRIVERS}/test_concurrent.cpp)
    configure_test_target(${target})
    target_include_directories(${target} PRIVATE ${PERF_DRIVERS} ${PERF_CASES})
    target_compile_definitions(${target} PRIVATE PERF_BACKEND=2 PERF_CASE_IDX=${i})
endforeach()
foreach(i IN LISTS DEG_CASE_INDICES)
    set(target test_deg_concurrent_${i})
    add_executable(${target} ${PERF_DRIVERS}/test_concurrent.cpp)
    configure_test_target(${target})
    target_include_directories(${target} PRIVATE ${PERF_DRIVERS} ${PERF_CASES})
    target_compile_definitions(${target} PRIVATE PERF_BACKEND=3 PERF_CASE_IDX=${i})
endforeach()
foreach(i IN LISTS ALT_CASE_INDICES)
    set(target test_alt_concurrent_${i})
    add_executable(${target} ${PERF_DRIVERS}/test_concurrent.cpp)
    configure_test_target(${target})
    target_include_directories(${target} PRIVATE ${PERF_DRIVERS} ${PERF_CASES})
    target_compile_definitions(${target} PRIVATE PERF_BACKEND=4 PERF_CASE_IDX=${i})
endforeach()
foreach(i IN LISTS BGEMM_CASE_INDICES)
    set(target test_bgemm_concurrent_${i})
    add_executable(${target} ${PERF_DRIVERS}/test_concurrent.cpp)
    configure_test_target(${target})
    target_include_directories(${target} PRIVATE ${PERF_DRIVERS} ${PERF_CASES})
    target_compile_definitions(${target} PRIVATE PERF_BACKEND=5 PERF_CASE_IDX=${i})
endforeach()
foreach(i IN LISTS PAU_CASE_INDICES)
    set(target test_pau_concurrent_${i})
    add_executable(${target} ${PERF_DRIVERS}/test_concurrent.cpp)
    configure_test_target(${target})
    target_include_directories(${target} PRIVATE ${PERF_DRIVERS} ${PERF_CASES})
    target_compile_definitions(${target} PRIVATE PERF_BACKEND=6 PERF_CASE_IDX=${i})
endforeach()
foreach(i IN LISTS THROUGHPUT_CASE_INDICES)
    set(target test_throughput_concurrent_${i})
    add_executable(${target} ${PERF_DRIVERS}/test_concurrent.cpp)
    configure_test_target(${target})
    target_include_directories(${target} PRIVATE ${PERF_DRIVERS} ${PERF_CASES})
    target_compile_definitions(${target} PRIVATE PERF_BACKEND=7 PERF_CASE_IDX=${i} ${THROUGHPUT_EXTRA_DEFS})
endforeach()
foreach(i IN LISTS LATENCY_CASE_INDICES)
    set(target test_latency_concurrent_${i})
    add_executable(${target} ${PERF_DRIVERS}/test_concurrent.cpp)
    configure_test_target(${target})
    target_include_directories(${target} PRIVATE ${PERF_DRIVERS} ${PERF_CASES})
    target_compile_definitions(${target} PRIVATE PERF_BACKEND=8 PERF_CASE_IDX=${i} ${LATENCY_EXTRA_DEFS})
endforeach()

# ─── CTest integration ────────────────────────────────────────────────────────
enable_testing()

add_test(NAME cpu_affinity    COMMAND test_cpu_affinity)
add_test(NAME platform_config COMMAND test_platform_config)
set_tests_properties(cpu_affinity platform_config
    PROPERTIES TIMEOUT 600 LABELS "functional")

foreach(i IN LISTS PA_CASE_INDICES)
    add_test(NAME pa_concurrent_${i} COMMAND test_pa_concurrent_${i})
    set_tests_properties(pa_concurrent_${i} PROPERTIES TIMEOUT 600 LABELS "perf")
endforeach()
foreach(i IN LISTS PA_CASE_INDICES)
    add_test(NAME pa_orch_only_${i} COMMAND test_pa_orch_only_${i})
    set_tests_properties(pa_orch_only_${i} PROPERTIES TIMEOUT 600 LABELS "perf;orch_only")
endforeach()
foreach(i IN LISTS PA_CASE_INDICES)
    add_test(NAME pa_sched_prof_only_${i} COMMAND test_pa_sched_prof_only_${i})
    set_tests_properties(pa_sched_prof_only_${i} PROPERTIES TIMEOUT 600 LABELS "perf;sched_prof_only")
endforeach()

foreach(i IN LISTS BPA_CASE_INDICES)
    add_test(NAME concurrent_${i} COMMAND test_concurrent_${i})
    set_tests_properties(concurrent_${i} PROPERTIES TIMEOUT 600 LABELS "perf")
endforeach()
foreach(i IN LISTS BPA_CASE_INDICES)
    add_test(NAME orch_only_${i} COMMAND test_orch_only_${i})
    set_tests_properties(orch_only_${i} PROPERTIES TIMEOUT 600 LABELS "perf;orch_only")
endforeach()
foreach(i IN LISTS BPA_CASE_INDICES)
    add_test(NAME sched_prof_only_${i} COMMAND test_sched_prof_only_${i})
    set_tests_properties(sched_prof_only_${i} PROPERTIES TIMEOUT 600 LABELS "perf;sched_prof_only")
endforeach()

foreach(i IN LISTS LINEAR_CASE_INDICES)
    add_test(NAME linear_concurrent_${i} COMMAND test_linear_concurrent_${i})
    set_tests_properties(linear_concurrent_${i} PROPERTIES TIMEOUT 600 LABELS "perf")
endforeach()
foreach(i IN LISTS LINEAR_CASE_INDICES)
    add_test(NAME linear_orch_only_${i} COMMAND test_linear_orch_only_${i})
    set_tests_properties(linear_orch_only_${i} PROPERTIES TIMEOUT 600 LABELS "perf;orch_only")
endforeach()
foreach(i IN LISTS LINEAR_CASE_INDICES)
    add_test(NAME linear_sched_prof_only_${i} COMMAND test_linear_sched_prof_only_${i})
    set_tests_properties(linear_sched_prof_only_${i} PROPERTIES TIMEOUT 600 LABELS "perf;sched_prof_only")
endforeach()

foreach(i IN LISTS DEG_CASE_INDICES)
    add_test(NAME deg_concurrent_${i} COMMAND test_deg_concurrent_${i})
    set_tests_properties(deg_concurrent_${i} PROPERTIES TIMEOUT 600 LABELS "perf")
endforeach()
foreach(i IN LISTS DEG_CASE_INDICES)
    add_test(NAME deg_orch_only_${i} COMMAND test_deg_orch_only_${i})
    set_tests_properties(deg_orch_only_${i} PROPERTIES TIMEOUT 600 LABELS "perf;orch_only")
endforeach()
foreach(i IN LISTS DEG_CASE_INDICES)
    add_test(NAME deg_sched_prof_only_${i} COMMAND test_deg_sched_prof_only_${i})
    set_tests_properties(deg_sched_prof_only_${i} PROPERTIES TIMEOUT 600 LABELS "perf;sched_prof_only")
endforeach()

foreach(i IN LISTS ALT_CASE_INDICES)
    add_test(NAME alt_concurrent_${i} COMMAND test_alt_concurrent_${i})
    set_tests_properties(alt_concurrent_${i} PROPERTIES TIMEOUT 600 LABELS "perf")
endforeach()
foreach(i IN LISTS ALT_CASE_INDICES)
    add_test(NAME alt_orch_only_${i} COMMAND test_alt_orch_only_${i})
    set_tests_properties(alt_orch_only_${i} PROPERTIES TIMEOUT 600 LABELS "perf;orch_only")
endforeach()
foreach(i IN LISTS ALT_CASE_INDICES)
    add_test(NAME alt_sched_prof_only_${i} COMMAND test_alt_sched_prof_only_${i})
    set_tests_properties(alt_sched_prof_only_${i} PROPERTIES TIMEOUT 600 LABELS "perf;sched_prof_only")
endforeach()

foreach(i IN LISTS BGEMM_CASE_INDICES)
    add_test(NAME bgemm_concurrent_${i} COMMAND test_bgemm_concurrent_${i})
    set_tests_properties(bgemm_concurrent_${i} PROPERTIES TIMEOUT 600 LABELS "perf")
endforeach()
foreach(i IN LISTS BGEMM_CASE_INDICES)
    add_test(NAME bgemm_orch_only_${i} COMMAND test_bgemm_orch_only_${i})
    set_tests_properties(bgemm_orch_only_${i} PROPERTIES TIMEOUT 600 LABELS "perf;orch_only")
endforeach()
foreach(i IN LISTS BGEMM_CASE_INDICES)
    add_test(NAME bgemm_sched_prof_only_${i} COMMAND test_bgemm_sched_prof_only_${i})
    set_tests_properties(bgemm_sched_prof_only_${i} PROPERTIES TIMEOUT 600 LABELS "perf;sched_prof_only")
endforeach()

foreach(i IN LISTS PAU_CASE_INDICES)
    add_test(NAME pau_concurrent_${i} COMMAND test_pau_concurrent_${i})
    set_tests_properties(pau_concurrent_${i} PROPERTIES TIMEOUT 600 LABELS "perf")
endforeach()
foreach(i IN LISTS PAU_CASE_INDICES)
    add_test(NAME pau_orch_only_${i} COMMAND test_pau_orch_only_${i})
    set_tests_properties(pau_orch_only_${i} PROPERTIES TIMEOUT 600 LABELS "perf;orch_only")
endforeach()
foreach(i IN LISTS PAU_CASE_INDICES)
    add_test(NAME pau_sched_prof_only_${i} COMMAND test_pau_sched_prof_only_${i})
    set_tests_properties(pau_sched_prof_only_${i} PROPERTIES TIMEOUT 600 LABELS "perf;sched_prof_only")
endforeach()

foreach(i IN LISTS THROUGHPUT_CASE_INDICES)
    add_test(NAME throughput_concurrent_${i} COMMAND test_throughput_concurrent_${i})
    set_tests_properties(throughput_concurrent_${i} PROPERTIES TIMEOUT 600 LABELS "perf")
endforeach()
foreach(i IN LISTS THROUGHPUT_CASE_INDICES)
    add_test(NAME throughput_orch_only_${i} COMMAND test_throughput_orch_only_${i})
    set_tests_properties(throughput_orch_only_${i} PROPERTIES TIMEOUT 600 LABELS "perf;orch_only")
endforeach()
foreach(i IN LISTS THROUGHPUT_CASE_INDICES)
    add_test(NAME throughput_sched_prof_only_${i} COMMAND test_throughput_sched_prof_only_${i})
    set_tests_properties(throughput_sched_prof_only_${i} PROPERTIES TIMEOUT 600 LABELS "perf;sched_prof_only")
endforeach()

foreach(i IN LISTS LATENCY_CASE_INDICES)
    add_test(NAME latency_concurrent_${i} COMMAND test_latency_concurrent_${i})
    set_tests_properties(latency_concurrent_${i} PROPERTIES TIMEOUT 600 LABELS "perf")
endforeach()
foreach(i IN LISTS LATENCY_CASE_INDICES)
    add_test(NAME latency_orch_only_${i} COMMAND test_latency_orch_only_${i})
    set_tests_properties(latency_orch_only_${i} PROPERTIES TIMEOUT 600 LABELS "perf;orch_only")
endforeach()
foreach(i IN LISTS LATENCY_CASE_INDICES)
    add_test(NAME latency_sched_prof_only_${i} COMMAND test_latency_sched_prof_only_${i})
    set_tests_properties(latency_sched_prof_only_${i} PROPERTIES TIMEOUT 600 LABELS "perf;sched_prof_only")
endforeach()


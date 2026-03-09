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
#   cmake .. -DPTO2_PROFILING=OFF
# ─────────────────────────────────────────────────────────────────────────────

cmake_minimum_required(VERSION 3.15)
project(orchestration_ut CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Place all test binaries under build/bin/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ─── CPU affinity parameters ──────────────────────────────────────────────────
set(ORCH_CPU   "0" CACHE STRING "Orchestrator CPU core")
set(SCHED_CPU0 "1" CACHE STRING "Scheduler CPU core 0")
set(SCHED_CPU1 "2" CACHE STRING "Scheduler CPU core 1")
set(SCHED_CPU2 "3" CACHE STRING "Scheduler CPU core 2")
set(SCHED_CPU3 "4" CACHE STRING "Scheduler CPU core 3")
set(SCHED_CPU4 "5" CACHE STRING "Scheduler CPU core 4")
set(SCHED_CPU5 "6" CACHE STRING "Scheduler CPU core 5")
set(SCHED_CPU6 "7" CACHE STRING "Scheduler CPU core 6")
set(SCHED_CPU7 "8" CACHE STRING "Scheduler CPU core 7")

# ─── Platform configuration parameters ────────────────────────────────────────
set(PLATFORM_MAX_BLOCKDIM            "24" CACHE STRING "Max block dim")
set(PLATFORM_AIC_CORES_PER_BLOCKDIM  "1"  CACHE STRING "AIC cores per blockdim")
set(PLATFORM_AIV_CORES_PER_BLOCKDIM  "2"  CACHE STRING "AIV cores per blockdim")
set(PLATFORM_MAX_AICPU_THREADS       "4"  CACHE STRING "Max AICPU threads")

# ─── Profiling control ────────────────────────────────────────────────────────
option(PTO2_PROFILING "Enable PTO2 profiling instrumentation" ON)

# ─── Source directories ────────────────────────────────────────────────────────
set(ROOT_DIR    ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(RUNTIME_DIR ${ROOT_DIR}/src/a2a3/runtime/tensormap_and_ringbuffer/runtime)
set(ORCH_DIR    ${ROOT_DIR}/src/a2a3/runtime/tensormap_and_ringbuffer/orchestration)
set(COMMON_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/common)
set(FUNC_DIR    ${CMAKE_CURRENT_SOURCE_DIR}/tests/functional)
set(PERF_DIR    ${CMAKE_CURRENT_SOURCE_DIR}/tests/perf)

# ─── Common include directories ───────────────────────────────────────────────
set(COMMON_INCLUDES
    ${RUNTIME_DIR}
    ${ROOT_DIR}/src/a2a3/platform/include
    ${ORCH_DIR}
    ${COMMON_DIR}
)

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
    list(APPEND COMMON_COMPILE_DEFS PTO2_ORCH_PROFILING=1)
    list(APPEND COMMON_COMPILE_DEFS PTO2_SCHED_PROFILING=1)
endif()

# ─── Common compile options ───────────────────────────────────────────────────
set(COMMON_COMPILE_OPTIONS
    -O0
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

# ─── Test executables: Paged attention performance tests ──────────────────────
# One binary per case; PERF_CASE_IDX selects the case at compile time.
set(PA_CASE_INDICES 0)

foreach(i IN LISTS PA_CASE_INDICES)
    set(target test_paged_attention_${i})
    add_executable(${target} ${PERF_DIR}/test_paged_attention.cpp)
    configure_test_target(${target})
    target_compile_definitions(${target} PRIVATE PERF_CASE_IDX=${i})
endforeach()

# ─── Test executables: Batch paged attention performance tests ────────────────
# One binary per case; PERF_CASE_IDX selects the case at compile time.
set(BPA_CASE_INDICES 0 1 2 3 4)

foreach(i IN LISTS BPA_CASE_INDICES)
    set(target test_batch_paged_attention_${i})
    add_executable(${target} ${PERF_DIR}/test_batch_paged_attention.cpp)
    configure_test_target(${target})
    target_compile_definitions(${target} PRIVATE PERF_CASE_IDX=${i})
endforeach()

# ─── CTest integration ────────────────────────────────────────────────────────
enable_testing()

add_test(NAME cpu_affinity    COMMAND test_cpu_affinity)
add_test(NAME platform_config COMMAND test_platform_config)
set_tests_properties(cpu_affinity platform_config
    PROPERTIES TIMEOUT 600 LABELS "functional")

foreach(i IN LISTS PA_CASE_INDICES)
    add_test(NAME paged_attention_${i} COMMAND test_paged_attention_${i})
    set_tests_properties(paged_attention_${i} PROPERTIES TIMEOUT 600 LABELS "perf")
endforeach()

foreach(i IN LISTS BPA_CASE_INDICES)
    add_test(NAME batch_paged_attention_${i} COMMAND test_batch_paged_attention_${i})
    set_tests_properties(batch_paged_attention_${i} PROPERTIES TIMEOUT 600 LABELS "perf")
endforeach()

# Build Host runtime for a2a3sim (Simulation)
#
# This builds the simulated host runtime that uses threads instead of
# actual device execution. AICPU and AICore executors are linked statically.
#
# CMAKE_C_COMPILER and CMAKE_CXX_COMPILER are set when calling cmake
# CUSTOM_INCLUDE_DIRS and CUSTOM_SOURCE_DIRS are set when calling cmake
# Optional static linking parameters:
#   AICPU_STATIC_LIB    - path to libaicpu_kernel.a
#   AICORE_STATIC_LIB   - path to libaicore_kernel.a
#   EXTRA_OBJECTS       - semicolon-separated list of extra .o files to link
#   KERNEL_DISPATCH_SOURCE - path to generated kernel_dispatch.cpp

cmake_minimum_required(VERSION 3.16.3)

project(host_runtime_sim LANGUAGES C CXX)

# Build complete include list
set(CMAKE_CUSTOM_INCLUDE_DIRS "")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../include")

if(DEFINED CUSTOM_INCLUDE_DIRS)
    foreach(INC_DIR ${CUSTOM_INCLUDE_DIRS})
        list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${INC_DIR}")
    endforeach()
else()
    message(FATAL_ERROR "MUST set CUSTOM_INCLUDE_DIRS to build Host runtime (simulation)")
endif()

# Build complete source list
set(HOST_RUNTIME_SOURCES "")
list(APPEND HOST_RUNTIME_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/device_runner.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/memory_allocator.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/pto_runtime_c_api.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/platform_compile_info.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/host/host_log.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/host/unified_log_host.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/host/performance_collector.cpp"
)

if(DEFINED CUSTOM_SOURCE_DIRS)
    foreach(SRC_DIR ${CUSTOM_SOURCE_DIRS})
        file(GLOB DIR_SOURCES "${SRC_DIR}/*.cpp" "${SRC_DIR}/*.c")
        list(APPEND HOST_RUNTIME_SOURCES ${DIR_SOURCES})
    endforeach()
else()
    message(STATUS "CUSTOM_SOURCE_DIRS not set, using only core sources")
endif()

# Add kernel dispatch: use generated source if provided, otherwise use stub
if(DEFINED KERNEL_DISPATCH_SOURCE)
    list(APPEND HOST_RUNTIME_SOURCES "${KERNEL_DISPATCH_SOURCE}")
else()
    list(APPEND HOST_RUNTIME_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/kernel_dispatch_stub.cpp")
endif()

# Create shared library (still .so so Python ctypes can load it)
add_library(host_runtime SHARED ${HOST_RUNTIME_SOURCES})

# C++ standard (applied only to C++ files)
set_target_properties(host_runtime PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Compile options
target_compile_options(host_runtime
    PRIVATE
        -Wall
        -Wextra
        -fPIC
        -O3
        -g
)

# Define STATIC_ORCH_LINK: replaces dlopen for aicpu/aicore with direct calls
target_compile_definitions(host_runtime PRIVATE STATIC_ORCH_LINK)

# Include directories
target_include_directories(host_runtime
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${CMAKE_CUSTOM_INCLUDE_DIRS}
)

# Link against pthread (for std::thread)
# Allow multiple definitions for common.cpp (compiled in both host runtime and orchestration static objects)
target_link_libraries(host_runtime
    PRIVATE
        pthread
        dl
        -Wl,--allow-multiple-definition
)

# Link static aicpu and aicore libraries if provided
if(DEFINED AICPU_STATIC_LIB AND DEFINED AICORE_STATIC_LIB)
    target_link_libraries(host_runtime PRIVATE
        "${AICPU_STATIC_LIB}" "${AICORE_STATIC_LIB}"
    )
endif()

# Add extra object files (orch.o, kernel*.o) if provided
if(DEFINED EXTRA_OBJECTS)
    foreach(OBJ_PATH ${EXTRA_OBJECTS})
        target_link_libraries(host_runtime PRIVATE "${OBJ_PATH}")
    endforeach()
endif()

set_target_properties(host_runtime PROPERTIES
    OUTPUT_NAME "host_runtime"
    # Force .so suffix on all platforms (macOS defaults to .dylib)
    # This matches the hardcoded paths in Python/C++ code
    SUFFIX ".so"
)


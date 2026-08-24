# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
project(a5_aicore_rtt_host LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(launch_a5_aicore_rtt launch.cpp)
add_executable(a5_aicore_rtt_layout_test layout_test.cpp)
target_compile_definitions(launch_a5_aicore_rtt PRIVATE
    A5_RTT_PLOT_SCRIPT="${CMAKE_CURRENT_SOURCE_DIR}/plot_results.py"
)
target_compile_options(launch_a5_aicore_rtt PRIVATE -Wall -Wextra -Werror -Wno-deprecated-declarations)
target_compile_options(a5_aicore_rtt_layout_test PRIVATE -Wall -Wextra -Werror)

if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
    target_include_directories(launch_a5_aicore_rtt PRIVATE
        "${ASCEND_HOME_PATH}/include"
        "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc"
        "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/runtime"
        "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/runtime/runtime"
        "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/profiling"
        "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/driver"
    )
    target_link_directories(launch_a5_aicore_rtt PRIVATE
        "${ASCEND_HOME_PATH}/lib64"
        "${ASCEND_HOME_PATH}/runtime/lib64"
        "${ASCEND_HOME_PATH}/driver/lib64"
        "${ASCEND_HOME_PATH}/driver/lib64/driver"
        "${ASCEND_HOME_PATH}/../driver/lib64"
        "${ASCEND_HOME_PATH}/../driver/lib64/driver"
        "${ASCEND_HOME_PATH}/../../driver/lib64"
        "${ASCEND_HOME_PATH}/../../driver/lib64/driver"
    )
    target_link_libraries(launch_a5_aicore_rtt PRIVATE ascendcl runtime ascend_hal dl)
else()
    set_target_properties(launch_a5_aicore_rtt PROPERTIES EXCLUDE_FROM_ALL TRUE EXCLUDE_FROM_DEFAULT_BUILD TRUE)
    message(STATUS "ASCEND_HOME_PATH is unset; only the layout test is available")
endif()

enable_testing()
add_test(NAME a5_aicore_rtt_layout COMMAND a5_aicore_rtt_layout_test)
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_test(
        NAME a5_aicore_rtt_assignment_generator
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_generate_runtime_assignment.py"
    )
endif()

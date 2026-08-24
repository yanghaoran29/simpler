# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
project(a5_rtt_producer LANGUAGES NONE)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH must be set")
endif()
set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
set(BISHENG_CC "${ASCEND_HOME_PATH}/bin/ccec")
if(NOT EXISTS "${BISHENG_CC}")
    set(BISHENG_CC "${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin/ccec")
endif()
set(BISHENG_LD "${ASCEND_HOME_PATH}/compiler/ccec_compiler/bin/ld.lld")
if(NOT EXISTS "${BISHENG_LD}")
    set(BISHENG_LD "${ASCEND_HOME_PATH}/bin/ld.lld")
endif()
if(NOT EXISTS "${BISHENG_LD}")
    set(BISHENG_LD "${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin/ld.lld")
endif()
if(NOT EXISTS "${BISHENG_CC}" OR NOT EXISTS "${BISHENG_LD}")
    message(FATAL_ERROR "A5 ccec/ld.lld toolchain was not found under ASCEND_HOME_PATH")
endif()

set(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/producer.cce)
set(AIC_OBJECT ${CMAKE_CURRENT_BINARY_DIR}/producer_aic.o)
set(AIV_OBJECT ${CMAKE_CURRENT_BINARY_DIR}/producer_aiv.o)
set(AIC_COMBINED_OBJECT ${CMAKE_CURRENT_BINARY_DIR}/producer_aic_combined.o)
set(AIV_COMBINED_OBJECT ${CMAKE_CURRENT_BINARY_DIR}/producer_aiv_combined.o)
set(OUTPUT_OBJECT ${CMAKE_CURRENT_BINARY_DIR}/a5_rtt_producer.o)
set(INCLUDE_FLAGS
    -I${ASCEND_HOME_PATH}/include
    -I${ASCEND_HOME_PATH}/include/external
    -I${ASCEND_HOME_PATH}/include/c_api
    -I${ASCEND_HOME_PATH}/include/c_api/internal
    -I${ASCEND_HOME_PATH}/${CMAKE_HOST_SYSTEM_PROCESSOR}-linux/include
    -I${ASCEND_HOME_PATH}/${CMAKE_HOST_SYSTEM_PROCESSOR}-linux/asc
    -I${ASCEND_HOME_PATH}/${CMAKE_HOST_SYSTEM_PROCESSOR}-linux/asc/include
    -I${ASCEND_HOME_PATH}/${CMAKE_HOST_SYSTEM_PROCESSOR}-linux/asc/include/basic_api
    -I${ASCEND_HOME_PATH}/${CMAKE_HOST_SYSTEM_PROCESSOR}-linux/asc/include/interface
    -I${ASCEND_HOME_PATH}/${CMAKE_HOST_SYSTEM_PROCESSOR}-linux/ascendc/include
    -I${CMAKE_CURRENT_SOURCE_DIR}/../../../../src/a5/platform/onboard/aicore
)
set(COMPILE_FLAGS
    -c -O3 -g -x cce -Wall -std=c++17 --cce-aicore-only
    -mllvm -cce-aicore-stack-size=0x8000
    -mllvm -cce-aicore-function-stack-size=0x8000
    -mllvm -cce-aicore-record-overflow=false
    -mllvm -cce-aicore-addr-transform
    -mllvm -cce-aicore-dcci-insert-for-scalar=false
)

add_custom_command(
    OUTPUT ${AIC_OBJECT}
    COMMAND ${BISHENG_CC} ${COMPILE_FLAGS} ${INCLUDE_FLAGS} --cce-aicore-arch=dav-c310-cube
            -o ${AIC_OBJECT} ${SOURCE}
    DEPENDS ${SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/../shared/rtt_types.h
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../../src/a5/platform/onboard/aicore/simt_anchor.h
    VERBATIM
)
add_custom_command(
    OUTPUT ${AIV_OBJECT}
    COMMAND ${BISHENG_CC} ${COMPILE_FLAGS} ${INCLUDE_FLAGS} --cce-aicore-arch=dav-c310-vec
            -o ${AIV_OBJECT} ${SOURCE}
    DEPENDS ${SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/../shared/rtt_types.h
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../../src/a5/platform/onboard/aicore/simt_anchor.h
    VERBATIM
)
add_custom_command(
    OUTPUT ${AIC_COMBINED_OBJECT}
    COMMAND ${BISHENG_LD} -r -o ${AIC_COMBINED_OBJECT} ${AIC_OBJECT}
    DEPENDS ${AIC_OBJECT}
    VERBATIM
)
add_custom_command(
    OUTPUT ${AIV_COMBINED_OBJECT}
    COMMAND ${BISHENG_LD} -r -o ${AIV_COMBINED_OBJECT} ${AIV_OBJECT}
    DEPENDS ${AIV_OBJECT}
    VERBATIM
)
add_custom_command(
    OUTPUT ${OUTPUT_OBJECT}
    COMMAND ${BISHENG_LD} -m aicorelinux -Ttext=0 -static -n -o ${OUTPUT_OBJECT}
            ${AIC_COMBINED_OBJECT} ${AIV_COMBINED_OBJECT}
    DEPENDS ${AIC_COMBINED_OBJECT} ${AIV_COMBINED_OBJECT}
    VERBATIM
)
add_custom_target(a5_rtt_producer ALL DEPENDS ${OUTPUT_OBJECT})

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.16)
project(tilelang_st)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CMake 3.27+ may ask the linker to emit dependency files via
# `--dependency-file`. bisheng/cce-ld does not support that flag, so disable
# linker-generated link dependencies for this standalone ST build.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.27)
    set(CMAKE_LINK_DEPENDS_USE_LINKER FALSE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# --------------------------------------------------------------------------
# PTOAS binary — passed by run_st.py via -DPTOAS_BIN=...
# --------------------------------------------------------------------------
if(NOT DEFINED PTOAS_BIN)
    message(FATAL_ERROR "PTOAS_BIN is not set. Pass -DPTOAS_BIN=/path/to/ptoas to cmake.")
endif()

# --------------------------------------------------------------------------
# ASCEND environment
# --------------------------------------------------------------------------
if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please run set_env.sh.")
else()
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

set(PTO_ISA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../../../../../pto-isa" CACHE PATH "Path to pto-isa repo")
if(NOT EXISTS "${PTO_ISA_ROOT}/include/pto/pto-inst.hpp")
    message(FATAL_ERROR "Cannot find ${PTO_ISA_ROOT}/include/pto/pto-inst.hpp")
endif()
set(PTO_TILELANG_ST_COMMON_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../common")
set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)

set(CMAKE_COMPILER bisheng)
set(CMAKE_C_COMPILER ${CMAKE_COMPILER})
set(CMAKE_CXX_COMPILER ${CMAKE_COMPILER})

add_compile_options(
    -D_FORTIFY_SOURCE=2
    -O2 -std=c++17
    -Wno-macro-redefined -Wno-ignored-attributes -Wno-unknown-attributes
    -fstack-protector-strong
    -fPIC
)
add_link_options(
    -s
    -Wl,-z,relro
    -Wl,-z,now
)

set(CMAKE_CCE_COMPILE_OPTIONS
    -xcce
    -fPIC
    -Xhost-start -Xhost-end
    "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-record-overflow=true"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false"
)

set(CMAKE_CPP_COMPILE_OPTIONS
    -xc++
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h"
)

include_directories(
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_DRIVER_PATH}/kernel/inc
)
include_directories(BEFORE
    ${PTO_ISA_ROOT}/include
)

add_subdirectory(testcase)

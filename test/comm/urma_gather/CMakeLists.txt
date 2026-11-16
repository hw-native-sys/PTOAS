# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


cmake_minimum_required(VERSION 3.16)
project(ptoas_urma_gather LANGUAGES CXX)
set(CMAKE_LINK_DEPENDS_USE_LINKER FALSE)

set(PTOAS_BIN "" CACHE FILEPATH "Built PTOAS CLI")
set(PTO_ISA_ROOT "$ENV{PTO_ISA_ROOT}" CACHE PATH "PTO-ISA checkout with A5 URMA Gather")
set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "CANN toolkit root")
set(PTO_URMA_GATHER_PATH "auto" CACHE STRING "Device fill path (auto, scalar or simt)")
set_property(CACHE PTO_URMA_GATHER_PATH PROPERTY STRINGS auto scalar simt)
if(NOT EXISTS "${PTOAS_BIN}" OR NOT EXISTS "${PTO_ISA_ROOT}/include/pto/comm/async/urma/urma_gather_async.hpp")
  message(FATAL_ERROR "Set PTOAS_BIN and PTO_ISA_ROOT to a Gather-capable build and checkout")
endif()
if(NOT IS_DIRECTORY "${ASCEND_HOME_PATH}/include")
  message(FATAL_ERROR "Set ASCEND_HOME_PATH by sourcing the CANN environment")
endif()

set(generated "${CMAKE_CURRENT_BINARY_DIR}/generated_urma_gather.hpp")
set(ir "${CMAKE_CURRENT_SOURCE_DIR}/kernels.pto")
add_custom_command(
  OUTPUT "${generated}"
  COMMAND "${PTOAS_BIN}" --pto-arch=a5 "${ir}" -o "${generated}"
  DEPENDS "${PTOAS_BIN}" "${ir}"
  VERBATIM
)
add_custom_target(urma_gather_emitc DEPENDS "${generated}")
add_executable(urma_gather urma_gather.cpp "${generated}")
target_compile_features(urma_gather PRIVATE cxx_std_17)
target_compile_definitions(urma_gather PRIVATE REGISTER_BASE)
if(PTO_URMA_GATHER_PATH STREQUAL "scalar")
  target_compile_definitions(urma_gather PRIVATE PTO_URMA_GATHER_FORCE_SCALAR)
elseif(PTO_URMA_GATHER_PATH STREQUAL "simt")
  target_compile_definitions(urma_gather PRIVATE PTO_URMA_GATHER_FORCE_SIMT)
elseif(NOT PTO_URMA_GATHER_PATH STREQUAL "auto")
  message(FATAL_ERROR "PTO_URMA_GATHER_PATH must be auto, scalar or simt")
endif()

set(toolkit "${ASCEND_HOME_PATH}/${CMAKE_HOST_SYSTEM_PROCESSOR}-linux")
target_include_directories(urma_gather PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}"
  "${PTO_ISA_ROOT}/include"
  "${PTO_ISA_ROOT}/tests/common"
  "${PTO_ISA_ROOT}/tests/npu/a5/comm/st/testcase"
  "${ASCEND_HOME_PATH}/include"
  "${ASCEND_HOME_PATH}/include/hccl"
  "${ASCEND_HOME_PATH}/pkg_inc"
  "${ASCEND_HOME_PATH}/pkg_inc/profiling"
  "${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime"
  "${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel/include"
  "${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel/include"
  "${toolkit}/asc"
  "${toolkit}/asc/include"
  "${toolkit}/asc/include/interface"
  "${toolkit}/asc/include/utils"
  "${toolkit}/asc/impl/basic_api"
  "${toolkit}/ascendc/include"
  "${toolkit}/ascendc/include/basic_api"
  "${toolkit}/ascendc/include/basic_api/impl"
  "${toolkit}/include/ascendc"
  "${toolkit}/include/ascendc/basic_api"
)
target_compile_options(urma_gather PRIVATE
  -O2 -xcce --cce-aicore-arch=dav-c310-vec -Xhost-start -Xhost-end
  "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
  "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
  "SHELL:-mllvm -cce-aicore-addr-transform"
  "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false"
)
target_link_options(urma_gather PRIVATE --cce-fatobj-link)
target_link_directories(urma_gather PRIVATE "${ASCEND_HOME_PATH}/lib64")
target_link_libraries(urma_gather PRIVATE ascendcl hcomm runtime tiling_api platform c_sec dl nnopbase pthread stdc++ m)

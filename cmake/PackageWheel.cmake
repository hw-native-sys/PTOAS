# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Package-only entrypoint. build.sh installs the wheel build's PTOAS_CMake
# component first; this configuration only stages it alongside the wheel.
if(NOT EXISTS "${PTOAS_NATIVE_BUILD_DIR}/CMakeCache.txt")
  message(FATAL_ERROR "PTOAS_NATIVE_BUILD_DIR must point to the completed wheel build")
endif()
load_cache("${PTOAS_NATIVE_BUILD_DIR}" READ_WITH_PREFIX _ptoas_native_
  PTOAS_CMAKE_INSTALL_DIR CMAKE_BUILD_TYPE)
if(NOT _ptoas_native_PTOAS_CMAKE_INSTALL_DIR)
  message(FATAL_ERROR "The wheel build does not provide PTOAS CMake metadata")
endif()
set(PTOAS_CMAKE_INSTALL_DIR "${_ptoas_native_PTOAS_CMAKE_INSTALL_DIR}")
set(CMAKE_BUILD_TYPE "${_ptoas_native_CMAKE_BUILD_TYPE}")
include("${CMAKE_CURRENT_LIST_DIR}/package.cmake")
include("${PROJECT_SOURCE_DIR}/version.cmake")
check_cann_pkg_build_deps("pto_as")
add_cann_version_info_targets()
pack_built_in()

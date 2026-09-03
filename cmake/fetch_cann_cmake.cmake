# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(NOT PROJECT_SOURCE_DIR)
    # Temporary test pin for cann/cmake MR !277. Revert this URL/ref pair
    # after validation and switch to the released cann/cmake tag once MR !277
    # is merged.
    if(CANN_3RD_LIB_PATH AND IS_DIRECTORY "${CANN_3RD_LIB_PATH}/cann-cmake")
        include("${CANN_3RD_LIB_PATH}/cann-cmake/function/prepare.cmake")
    else()
        set(CANN_CMAKE_GIT_URL "$ENV{CANN_CMAKE_GIT_URL}" CACHE STRING "CANN cmake repository URL")
        set(CANN_CMAKE_GIT_REF "$ENV{CANN_CMAKE_GIT_REF}" CACHE STRING "CANN cmake repository ref")
        if(NOT CANN_CMAKE_GIT_URL)
            set(CANN_CMAKE_GIT_URL "https://gitcode.com/zoujiangjiang/cmake.git")
        endif()
        if(NOT CANN_CMAKE_GIT_REF)
            set(CANN_CMAKE_GIT_REF "fe/llvm-vpto-patch")
        endif()
        if(CMAKE_SCRIPT_MODE_FILE)
            if(NOT CANN_3RD_LIB_PATH)
                message(FATAL_ERROR "CANN_3RD_LIB_PATH is required to fetch CANN cmake")
            endif()

            file(REMOVE_RECURSE "${CANN_3RD_LIB_PATH}/cann-cmake")
            execute_process(
                COMMAND git -c http.version=HTTP/1.1 clone
                        --depth 1
                        --single-branch
                        --branch ${CANN_CMAKE_GIT_REF}
                        ${CANN_CMAKE_GIT_URL}
                        "${CANN_3RD_LIB_PATH}/cann-cmake"
                RESULT_VARIABLE CANN_CMAKE_CLONE_RESULT
            )
            if(NOT CANN_CMAKE_CLONE_RESULT EQUAL 0)
                message(FATAL_ERROR "failed to fetch CANN cmake from ${CANN_CMAKE_GIT_URL} ${CANN_CMAKE_GIT_REF}")
            endif()
            return()
        endif()

        include(FetchContent)

        set(CANN_CMAKE_TAG "pr277")
        if(CANN_3RD_LIB_PATH AND EXISTS "${CANN_3RD_LIB_PATH}/cmake-${CANN_CMAKE_TAG}.tar.gz")
            FetchContent_Declare(
                cann-cmake
                URL "${CANN_3RD_LIB_PATH}/cmake-${CANN_CMAKE_TAG}.tar.gz"
                SOURCE_DIR "${CANN_3RD_LIB_PATH}/cann-cmake"
            )
        else()
            FetchContent_Declare(
                cann-cmake
                GIT_REPOSITORY ${CANN_CMAKE_GIT_URL}
                GIT_TAG        ${CANN_CMAKE_GIT_REF}
                GIT_SHALLOW    TRUE
                SOURCE_DIR "${CANN_3RD_LIB_PATH}/cann-cmake"
            )
        endif()
        FetchContent_GetProperties(cann-cmake)
        if(NOT cann-cmake_POPULATED)
            FetchContent_Populate(cann-cmake)
        endif()
        include("${cann-cmake_SOURCE_DIR}/function/prepare.cmake")
    endif()
endif()

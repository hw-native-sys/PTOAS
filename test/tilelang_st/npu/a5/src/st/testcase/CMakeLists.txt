# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# --------------------------------------------------------------------------
# pto_tilelang_st(NAME)
#
# CMake macro for TileLang ST test cases.  Unlike pto-isa's pto_vec_st()
# which compiles a hand-written kernel.cpp with -xcce, this macro:
#   1. Runs ptoas to compile <NAME>.pto → kernel.fatobj.o
#   2. Links the fatobj object with launch.cpp → shared library
#   3. Builds host executable from main.cpp (no GTest — comparison via compare.py)
# --------------------------------------------------------------------------
set(PTO_TILELANG_ST_TESTCASE_DIR ${CMAKE_CURRENT_LIST_DIR})
if(DEFINED ENV{TILELANG_ST_SIM_LIB_DIR})
    set(PTO_TILELANG_ST_SIM_LIB_DIR $ENV{TILELANG_ST_SIM_LIB_DIR})
else()
    set(PTO_TILELANG_ST_SIM_LIB_DIR ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib)
endif()

function(pto_tilelang_st NAME)
    set(options DISABLE_INSERT_SYNC)
    set(oneValueArgs PTO_LEVEL AICORE_ARCH)
    cmake_parse_arguments(PTO_TILELANG_ST "${options}" "${oneValueArgs}" "" ${ARGN})

    set(PTOAS_ENABLE_INSERT_SYNC ON)
    if(PTO_TILELANG_ST_DISABLE_INSERT_SYNC)
        set(PTOAS_ENABLE_INSERT_SYNC OFF)
    endif()

    set(PTOAS_PTO_LEVEL "")
    if(DEFINED PTO_TILELANG_ST_PTO_LEVEL)
        set(PTOAS_PTO_LEVEL "${PTO_TILELANG_ST_PTO_LEVEL}")
    endif()

    set(AICORE_ARCH "dav-c310-vec")
    if(DEFINED PTO_TILELANG_ST_AICORE_ARCH)
        set(AICORE_ARCH "${PTO_TILELANG_ST_AICORE_ARCH}")
    endif()

    # Step 1: ptoas  .pto → kernel fatobj object
    set(PTO_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${NAME}.pto)
    set(KERNEL_FATOBJ ${CMAKE_CURRENT_BINARY_DIR}/${NAME}_kernel.o)
    set(PTOAS_CAPTURE_SCRIPT
        ${PTO_TILELANG_ST_TESTCASE_DIR}/run_ptoas_to_file.cmake)
    add_custom_command(
        OUTPUT ${KERNEL_FATOBJ}
        COMMAND ${CMAKE_COMMAND}
            -DPTOAS_BIN=${PTOAS_BIN}
            -DPTO_SRC=${PTO_SRC}
            -DKERNEL_FATOBJ=${KERNEL_FATOBJ}
            -DPTOAS_ENABLE_INSERT_SYNC=${PTOAS_ENABLE_INSERT_SYNC}
            -DPTOAS_PTO_LEVEL=${PTOAS_PTO_LEVEL}
            -P ${PTOAS_CAPTURE_SCRIPT}
        DEPENDS ${PTO_SRC} ${PTOAS_CAPTURE_SCRIPT}
        COMMENT "ptoas: ${NAME}.pto -> ${NAME}_kernel.o"
        VERBATIM
    )

    # Step 2: link the fatobj object with launch.cpp.
    add_library(${NAME}_kernel SHARED launch.cpp ${KERNEL_FATOBJ})
    set_source_files_properties(${KERNEL_FATOBJ}
        PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
    target_compile_options(${NAME}_kernel PRIVATE
        ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=${AICORE_ARCH} -std=c++17)
    target_include_directories(${NAME}_kernel PRIVATE
        ${ASCEND_HOME_PATH}/pkg_inc/
        ${ASCEND_HOME_PATH}/pkg_inc/profiling/
        ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
    )
    target_link_options(${NAME}_kernel PRIVATE --cce-fatobj-link)

    # Step 3: main.cpp → host executable
    add_executable(${NAME} main.cpp)
    target_compile_options(${NAME} PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
    target_include_directories(${NAME} PRIVATE
        ${PTO_TILELANG_ST_COMMON_DIR}
    )

    target_link_directories(${NAME} PUBLIC
        ${ASCEND_HOME_PATH}/lib64
        ${PTO_TILELANG_ST_SIM_LIB_DIR}
    )

    target_link_libraries(${NAME} PRIVATE
        ${NAME}_kernel
        $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
        $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
        stdc++ ascendcl m tiling_api platform c_sec dl nnopbase pthread
    )
endfunction()

function(pto_tilelang_vec_st NAME)
    pto_tilelang_st(
        ${NAME}
        AICORE_ARCH dav-c310-vec
        ${ARGN}
    )
endfunction()

function(pto_tilelang_cube_st NAME)
    pto_tilelang_st(
        ${NAME}
        AICORE_ARCH dav-c310-cube
        ${ARGN}
    )
endfunction()

# --------------------------------------------------------------------------
# Test case registry — add new ops here.
# --------------------------------------------------------------------------
set(ALL_TESTCASES
    tadd
    tinsert
    tinsert_acc2vec
    tinsert_vec
    tsub
    tmul
    tdiv
    tmax
    tmin
    tmov
    tmrgsort
    tshl
    tshr
    tand
    tor
    txor
    tcmp
    tfmod
    trem
    tcvt
    tload
    tlrelu
    trelu
    tsel
    tsels
    tcolmax
    tcolmin
    tcolsum
    tcolprod
    tcolargmax
    tcolargmin
    tcolexpand
    tcolexpandadd
    tcolexpandsub
    tcolexpandmul
    tcolexpanddiv
    tcolexpandmax
    tcolexpandmin
    tcolexpandexpdif
    softmax
    tabs
    texp
    tlog
    tneg
    tnot
    tpartmax
    tpartmin
    tpartadd
    tpartmul
    tprelu
    trandom
    trecip
    trowargmax
    trowargmin
    trowsum
    trowmax
    trowmin
    trowprod
    trsqrt
    tsort32
    tsqrt
    trowexpand
    trowexpandadd
    trowexpanddiv
    trowexpandexpdif
    trowexpandmax
    trowexpandmin
    trowexpandmul
    trowexpandsub
    texpands
    tfillpad
    tfillpad_inplace
    tfillpad_expand
    tadds
    tands
    tdivs
    tmaxs
    tmins
    tmuls
    tors
    tshls
    tshrs
    tsubs
    txors
    trems
    tfmods
    tcmps
    tmatmul
    tmatmul_acc
    tmatmul_bias
    tmatmul_bias_mx
    tmatmul_mx
    tgemv
    tgemv_mx
    textract
    textract_fp
    textract_v2v
    tmov2left
    tmov2right
    tmov2bias
    tmov2scale
    tmov2vec
    tmov_fp
    tstore_acc2gm
    tload_mat
)

if((TEST_CASE IN_LIST ALL_TESTCASES) OR (TEST_CASE STREQUAL "all"))
    message(STATUS "run: ${TEST_CASE}")
else()
    message(FATAL_ERROR "not found TEST_CASE: ${TEST_CASE}, supported: ${ALL_TESTCASES}")
endif()

foreach(TESTCASE ${ALL_TESTCASES})
    if((DEFINED TEST_CASE AND TEST_CASE STREQUAL TESTCASE) OR (NOT DEFINED TEST_CASE) OR (TEST_CASE STREQUAL "all"))
        add_subdirectory(${TESTCASE})
    endif()
endforeach()

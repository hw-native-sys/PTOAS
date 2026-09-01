# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.20.0)

if(NOT PTOAS_SOURCE_DIR)
    message(FATAL_ERROR "PTOAS_SOURCE_DIR is required")
endif()

if(NOT CANN_3RD_LIB_PATH)
    set(CANN_3RD_LIB_PATH ${PTOAS_SOURCE_DIR}/third_party)
endif()

include("${PTOAS_SOURCE_DIR}/cmake/fetch_cann_cmake.cmake")

project(ptoas_superbuild NONE)

init_cann_project()
include(ExternalProject)

if(NOT CANN_CMAKE_SOURCE_DIR)
    set(CANN_CMAKE_SOURCE_DIR ${CANN_3RD_LIB_PATH}/cann-cmake)
endif()

if(NOT EXISTS "${CANN_CMAKE_SOURCE_DIR}/third_party/llvm.cmake")
    message(FATAL_ERROR "missing llvm cmake cache helper: ${CANN_CMAKE_SOURCE_DIR}/third_party/llvm.cmake")
endif()
set(LLVM_CMAKE_MODULE "${CANN_CMAKE_SOURCE_DIR}/third_party/llvm.cmake")

if(NOT PTOAS_BUILD_DIR)
    set(PTOAS_BUILD_DIR ${PTOAS_SOURCE_DIR}/build)
endif()

if(NOT PTOAS_INSTALL_DIR)
    set(PTOAS_INSTALL_DIR ${PTOAS_SOURCE_DIR}/install)
endif()

if(NOT LLVM_SOURCE_DIR)
    set(LLVM_SOURCE_DIR ${CANN_3RD_LIB_PATH}/llvm-19)
endif()

if(NOT LLVM_NATIVE_BUILD_DIR)
    set(LLVM_NATIVE_BUILD_DIR ${CANN_3RD_LIB_PATH}/lib_cache/llvm_19.1.7/build-native-tools)
endif()

if(NOT LLVM_BUILD_DIR)
    set(LLVM_BUILD_DIR ${CANN_3RD_LIB_PATH}/lib_cache/llvm_19.1.7/build-shared)
endif()

if(NOT PYBIND11_CMAKE_DIR)
    message(FATAL_ERROR "PYBIND11_CMAKE_DIR is required")
endif()

if(NOT HARDENING_CACHE_FILE)
    set(HARDENING_CACHE_FILE ${PTOAS_SOURCE_DIR}/cmake/LinuxHardeningCache.cmake)
endif()

if(NOT PTOAS_FORTIFY_MARKER_OBJECT)
    message(FATAL_ERROR "PTOAS_FORTIFY_MARKER_OBJECT is required")
endif()

# Only default to the CentOS7 devtoolset-7 sysroot + gcc-toolchain when the
# toolchain is actually present. Use NOT DEFINED so an explicit empty value
# passed from build.sh is respected instead of being treated as "unset".
if(NOT DEFINED DEVTOOLSET_TOOLCHAIN_FLAGS)
    if(EXISTS "/opt/rh/devtoolset-7/root")
        set(DEVTOOLSET_TOOLCHAIN_FLAGS "--sysroot=/opt/rh/devtoolset-7/root --gcc-toolchain=/opt/rh/devtoolset-7/root/usr")
    else()
        set(DEVTOOLSET_TOOLCHAIN_FLAGS "")
    endif()
endif()

set(PTOAS_RUNTIME_C_FLAGS "${DEVTOOLSET_TOOLCHAIN_FLAGS} -fstack-protector-all")
set(PTOAS_RUNTIME_CXX_FLAGS "${DEVTOOLSET_TOOLCHAIN_FLAGS} -fstack-protector-all")

# Let llvm.cmake decide LLVM_REQUIRE_SOURCE based on cache detection.
# When the LLVM/MLIR install cache is found, llvm.cmake returns early and
# skips the ExternalProject_Add for third_party_llvm source download.
include("${LLVM_CMAKE_MODULE}")

function(reset_llvm_build_tree_if_source_changed build_dir expected_source_dir)
    set(cache_file "${build_dir}/CMakeCache.txt")
    if(NOT EXISTS "${cache_file}")
        return()
    endif()

    file(STRINGS "${cache_file}" cached_home_dir_line REGEX "^CMAKE_HOME_DIRECTORY:INTERNAL=")
    if(NOT cached_home_dir_line)
        message(STATUS "Reset LLVM build tree without CMAKE_HOME_DIRECTORY: ${build_dir}")
        file(REMOVE_RECURSE "${build_dir}")
        return()
    endif()

    string(REPLACE "CMAKE_HOME_DIRECTORY:INTERNAL=" "" cached_home_dir "${cached_home_dir_line}")
    get_filename_component(cached_home_dir "${cached_home_dir}" ABSOLUTE)
    get_filename_component(expected_source_dir "${expected_source_dir}" ABSOLUTE)
    if(NOT cached_home_dir STREQUAL expected_source_dir)
        # The cached source dir may differ from the current one when the same
        # LLVM source tree is mounted at a different base path across CI images
        # (e.g. /home/jenkins/opensource vs /home/opensource). If the tail
        # components match (llvm-19/llvm), create a symlink from the cached
        # base path to the current base path so that all hardcoded paths in
        # CMakeCache.txt and build.ninja remain valid.
        get_filename_component(cached_tail "${cached_home_dir}" NAME)
        get_filename_component(expected_tail "${expected_source_dir}" NAME)
        get_filename_component(cached_parent "${cached_home_dir}" DIRECTORY)
        get_filename_component(cached_parent_tail "${cached_parent}" NAME)
        get_filename_component(expected_parent "${expected_source_dir}" DIRECTORY)
        get_filename_component(expected_parent_tail "${expected_parent}" NAME)
        if(cached_tail STREQUAL expected_tail AND cached_parent_tail STREQUAL expected_parent_tail
           AND EXISTS "${expected_source_dir}/CMakeLists.txt")
            # Extract the differing base path prefix from the cached source dir.
            # e.g. cached=/home/jenkins/opensource/llvm-19/llvm, expected=/home/opensource/llvm-19/llvm
            #      -> cached_base=/home/jenkins/opensource, expected_base=/home/opensource
            get_filename_component(cached_base "${cached_parent}" DIRECTORY)
            get_filename_component(expected_base "${expected_parent}" DIRECTORY)
            if(NOT EXISTS "${cached_base}" AND EXISTS "${expected_base}")
                # Create parent directories and symlink cached_base -> expected_base
                get_filename_component(cached_base_parent "${cached_base}" PATH)
                file(MAKE_DIRECTORY "${cached_base_parent}")
                file(CREATE_LINK "${expected_base}" "${cached_base}" SYMBOLIC)
                message(STATUS "Created symlink: ${cached_base} -> ${expected_base}")
            endif()
        else()
            message(STATUS "Reset LLVM build tree because source dir changed: ${build_dir}")
            message(STATUS "  cached source: ${cached_home_dir}")
            message(STATUS "  current source: ${expected_source_dir}")
            file(REMOVE_RECURSE "${build_dir}")
        endif()
    endif()
endfunction()

reset_llvm_build_tree_if_source_changed("${LLVM_NATIVE_BUILD_DIR}" "${LLVM_SOURCE_DIR}/llvm")
reset_llvm_build_tree_if_source_changed("${LLVM_BUILD_DIR}" "${LLVM_SOURCE_DIR}/llvm")

set(LLVM_SOURCE_DEPENDS)
if(TARGET third_party_llvm)
    list(APPEND LLVM_SOURCE_DEPENDS third_party_llvm)
endif()

# Detect archiver tools in the current environment so that cached build trees
# created on a different CI image (where CMAKE_AR may point at a tool under a
# different base path, e.g. /home/jenkins/Ascend/.../llvm-ar) get the correct
# tool paths passed explicitly via -D, overriding the stale CMakeCache.txt value.
find_program(PTOAS_CMAKE_AR NAMES llvm-ar ar DOC "Path to the archiver")
find_program(PTOAS_CMAKE_RANLIB NAMES llvm-ranlib ranlib DOC "Path to the index generator")
find_program(PTOAS_CMAKE_STRIP NAMES llvm-strip strip DOC "Path to the symbol stripping tool")
mark_as_advanced(PTOAS_CMAKE_AR PTOAS_CMAKE_RANLIB PTOAS_CMAKE_STRIP)

set(LLVM_COMMON_OPTIONS
    -DLLVM_ENABLE_PROJECTS=mlir
    -DLLVM_ENABLE_ZSTD=OFF
    -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
    -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
    -DCMAKE_C_COMPILER=clang
    -DCMAKE_CXX_COMPILER=clang++
    -DCMAKE_AR=${PTOAS_CMAKE_AR}
    -DCMAKE_RANLIB=${PTOAS_CMAKE_RANLIB}
    -DCMAKE_STRIP=${PTOAS_CMAKE_STRIP}
    -DLLVM_USE_LINKER=lld
    -DPython3_EXECUTABLE=${Python3_EXECUTABLE}
    -DCMAKE_BUILD_TYPE=Release
    -DLLVM_TARGETS_TO_BUILD=host
    -DLLVM_INCLUDE_TESTS=OFF
    -DLLVM_INCLUDE_BENCHMARKS=OFF
    -DLLVM_INCLUDE_EXAMPLES=OFF
)

# When a fully-built LLVM/MLIR runtime already exists in LLVM_BUILD_DIR (the
# prebuilt cache shipped under lib_cache/llvm_19.1.7), skip rebuilding both the
# native tblgen tools and the shared LLVM/MLIR libraries. ptoas_project only
# needs LLVM_BUILD_DIR/{lib,bin,lib/cmake} to be in place, so we replace the two
# ExternalProject steps with no-op stubs that still satisfy the DEPENDS chain.
#
# The cached MLIR Python bindings (mlir/_mlir_libs/_mlir*.so) are compiled for a
# specific CPython ABI tag (e.g. cpython-310) and cannot be loaded by a
# different interpreter minor version. Besides the C++ libraries, also require
# a _mlir extension matching the current Python3_EXECUTABLE; otherwise fall back
# to a full LLVM/MLIR rebuild so the bindings are regenerated for this Python.
set(LLVM_RUNTIME_PREBUILT FALSE)
set(_ptoas_mlir_pyext_suffix "")
if(Python3_EXECUTABLE)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import sysconfig;import sys;sys.stdout.write(sysconfig.get_config_var('EXT_SUFFIX') or '')"
        OUTPUT_VARIABLE _ptoas_mlir_pyext_suffix
        OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
set(_ptoas_mlir_pyext_path
    "${LLVM_BUILD_DIR}/tools/mlir/python_packages/mlir_core/mlir/_mlir_libs/_mlir${_ptoas_mlir_pyext_suffix}")
if(EXISTS "${LLVM_BUILD_DIR}/lib/libMLIRIR.so.19.1"
   AND EXISTS "${LLVM_BUILD_DIR}/lib/libLLVMSupport.so.19.1"
   AND EXISTS "${LLVM_BUILD_DIR}/lib/cmake/llvm/LLVMConfig.cmake"
   AND _ptoas_mlir_pyext_suffix
   AND EXISTS "${_ptoas_mlir_pyext_path}")
    set(LLVM_RUNTIME_PREBUILT TRUE)
    message(STATUS "LLVM runtime prebuilt artifacts detected at ${LLVM_BUILD_DIR} (Python ext suffix: ${_ptoas_mlir_pyext_suffix}), skip LLVM build")
else()
    message(STATUS "LLVM runtime prebuilt cache not usable"
        " (libMLIRIR=${LLVM_BUILD_DIR}/lib/libMLIRIR.so.19.1"
        ", mlir_pyext=${_ptoas_mlir_pyext_path}), will build LLVM/MLIR")
endif()

if(LLVM_RUNTIME_PREBUILT)
    # No-op stubs: keep the same target names so ptoas_project's DEPENDS still
    # resolve, but do not reconfigure/rebuild the cached LLVM tree.
    add_custom_target(llvm_native_tools)
    add_custom_target(llvm_runtime DEPENDS llvm_native_tools)
else()
ExternalProject_Add(llvm_native_tools
    DEPENDS ${LLVM_SOURCE_DEPENDS}
    SOURCE_DIR ${LLVM_SOURCE_DIR}/llvm
    BINARY_DIR ${LLVM_NATIVE_BUILD_DIR}
    DOWNLOAD_COMMAND ""
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ${CMAKE_COMMAND} -G Ninja
        -S ${LLVM_SOURCE_DIR}/llvm
        -B ${LLVM_NATIVE_BUILD_DIR}
        ${LLVM_COMMON_OPTIONS}
        -DBUILD_SHARED_LIBS=OFF
        -DMLIR_ENABLE_BINDINGS_PYTHON=OFF
    BUILD_COMMAND ${CMAKE_COMMAND} --build ${LLVM_NATIVE_BUILD_DIR}
        --target llvm-min-tblgen llvm-tblgen mlir-tblgen
    INSTALL_COMMAND ""
    EXCLUDE_FROM_ALL TRUE
)

ExternalProject_Add(llvm_runtime
    DEPENDS llvm_native_tools
    SOURCE_DIR ${LLVM_SOURCE_DIR}/llvm
    BINARY_DIR ${LLVM_BUILD_DIR}
    DOWNLOAD_COMMAND ""
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ${CMAKE_COMMAND} -C ${HARDENING_CACHE_FILE} -G Ninja
        -S ${LLVM_SOURCE_DIR}/llvm
        -B ${LLVM_BUILD_DIR}
        ${LLVM_COMMON_OPTIONS}
        -DBUILD_SHARED_LIBS=ON
        -DHAVE_LIBRT=ON
        -DLLVM_NATIVE_TOOL_DIR=${LLVM_NATIVE_BUILD_DIR}/bin
        -DLLVM_TABLEGEN=${LLVM_NATIVE_BUILD_DIR}/bin/llvm-tblgen
        -DMLIR_TABLEGEN_EXE=${LLVM_NATIVE_BUILD_DIR}/bin/mlir-tblgen
        -DMLIR_ENABLE_BINDINGS_PYTHON=ON
        -DCMAKE_C_FLAGS=${PTOAS_RUNTIME_C_FLAGS}
        -DCMAKE_CXX_FLAGS=${PTOAS_RUNTIME_CXX_FLAGS}
        -DPTOAS_FORTIFY_MARKER_OBJECT=${PTOAS_FORTIFY_MARKER_OBJECT}
    BUILD_COMMAND ${CMAKE_COMMAND} --build ${LLVM_BUILD_DIR}
    INSTALL_COMMAND ""
    EXCLUDE_FROM_ALL TRUE
)
endif()

ExternalProject_Add(ptoas_project
    DEPENDS llvm_runtime
    SOURCE_DIR ${PTOAS_SOURCE_DIR}
    BINARY_DIR ${PTOAS_BUILD_DIR}
    CONFIGURE_COMMAND ${CMAKE_COMMAND} -C ${HARDENING_CACHE_FILE} -G Ninja
        -S ${PTOAS_SOURCE_DIR}
        -B ${PTOAS_BUILD_DIR}
        -DLLVM_DIR=${LLVM_BUILD_DIR}/lib/cmake/llvm
        -DMLIR_DIR=${LLVM_BUILD_DIR}/lib/cmake/mlir
        -DPython3_EXECUTABLE=${Python3_EXECUTABLE}
        -DPython3_FIND_STRATEGY=LOCATION
        -Dpybind11_DIR=${PYBIND11_CMAKE_DIR}
        -DMLIR_ENABLE_BINDINGS_PYTHON=ON
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_C_COMPILER=clang
        -DCMAKE_CXX_COMPILER=clang++
        -DCMAKE_C_FLAGS=${PTOAS_RUNTIME_C_FLAGS}
        -DCMAKE_CXX_FLAGS=${PTOAS_RUNTIME_CXX_FLAGS}
        -DLLVM_USE_LINKER=lld
        -DMLIR_PYTHON_PACKAGE_DIR=${LLVM_BUILD_DIR}/tools/mlir/python_packages/mlir_core
        -DCMAKE_INSTALL_PREFIX=${PTOAS_INSTALL_DIR}
        -DCANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH}
        -DENABLE_PACKAGE=TRUE
        -DPACKAGE_TYPE=${PACKAGE_TYPE}
        -DPTOAS_FORTIFY_MARKER_OBJECT=${PTOAS_FORTIFY_MARKER_OBJECT}
    BUILD_COMMAND ${CMAKE_COMMAND} --build ${PTOAS_BUILD_DIR}
    INSTALL_COMMAND ""
    STEP_TARGETS build
    BUILD_ALWAYS TRUE
    EXCLUDE_FROM_ALL TRUE
)

add_custom_target(ptoas_install
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "${PTOAS_BUILD_DIR}/package_runtime/tools/ptoas/bin"
        "${PTOAS_BUILD_DIR}/package_runtime/tools/ptoas/lib"
    COMMAND ${CMAKE_COMMAND} -E env
        "LLVM_RUNTIME_LIB_DIR=${LLVM_BUILD_DIR}/lib"
        "LLVM_STRIP_BIN=${LLVM_BUILD_DIR}/bin/llvm-strip"
        bash
        "${PTOAS_SOURCE_DIR}/scripts/package/collect_ptoas_runtime_deps.sh"
        "${PTOAS_BUILD_DIR}"
        "${PTOAS_BUILD_DIR}/tools/ptoas/ptoas"
        "${PTOAS_BUILD_DIR}/package_runtime/tools/ptoas/bin/ptoas"
        "${PTOAS_BUILD_DIR}/package_runtime/tools/ptoas/lib"
    COMMAND ${CMAKE_COMMAND} --build "${PTOAS_BUILD_DIR}" --target install
    DEPENDS ptoas_project-build
    VERBATIM
)

add_custom_target(ptoas_package
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "${PTOAS_BUILD_DIR}/package_runtime/tools/ptoas/bin"
        "${PTOAS_BUILD_DIR}/package_runtime/tools/ptoas/lib"
    COMMAND ${CMAKE_COMMAND} -E env
        "LLVM_RUNTIME_LIB_DIR=${LLVM_BUILD_DIR}/lib"
        "LLVM_STRIP_BIN=${LLVM_BUILD_DIR}/bin/llvm-strip"
        bash
        "${PTOAS_SOURCE_DIR}/scripts/package/collect_ptoas_runtime_deps.sh"
        "${PTOAS_BUILD_DIR}"
        "${PTOAS_BUILD_DIR}/tools/ptoas/ptoas"
        "${PTOAS_BUILD_DIR}/package_runtime/tools/ptoas/bin/ptoas"
        "${PTOAS_BUILD_DIR}/package_runtime/tools/ptoas/lib"
    COMMAND ${CMAKE_COMMAND} --build "${PTOAS_BUILD_DIR}" --target install
    COMMAND ${CMAKE_COMMAND} --build "${PTOAS_BUILD_DIR}" --target package
    DEPENDS ptoas_project-build
    VERBATIM
)

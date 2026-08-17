# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.20.0)

# CANN/Ascend CMake packages used by downstream PTOAS builds may still use the
# pre-CMake-3.30 FetchContent_Populate(<declared-name>) pattern. Keep that
# compatibility path enabled so CMake 4.x developer warnings do not become
# configure errors when users build with strict warning settings. The default
# covers nested third-party directories that set their own cmake_minimum_required().
if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
  set(CMAKE_POLICY_DEFAULT_CMP0169 OLD)
endif()

# LLVM 19's AddMLIRPython.cmake still passes PRE_BUILD to the OUTPUT form of
# add_custom_command(). CMake 4.1 diagnoses that legacy combination under the
# NEW behavior of CMP0175, so retain the compatibility behavior while PTOAS is
# built against LLVM 19's installed CMake package.
if(POLICY CMP0175)
  cmake_policy(SET CMP0175 OLD)
  set(CMAKE_POLICY_DEFAULT_CMP0175 OLD)
endif()

# Standard Python build backends pass environment variables through to CMake
# but do not synthesize project-specific cache arguments. Accept the existing
# LLVM_BUILD_DIR convention directly so `pip wheel .` and `pip install .` can
# use the same configured LLVM tree as developer CMake builds.
if((NOT DEFINED LLVM_DIR OR LLVM_DIR STREQUAL "")
   AND DEFINED ENV{LLVM_BUILD_DIR}
   AND NOT "$ENV{LLVM_BUILD_DIR}" STREQUAL "")
  set(LLVM_DIR "$ENV{LLVM_BUILD_DIR}/lib/cmake/llvm" CACHE PATH
      "LLVM package directory derived from LLVM_BUILD_DIR")
endif()
if((NOT DEFINED MLIR_DIR OR MLIR_DIR STREQUAL "")
   AND DEFINED ENV{LLVM_BUILD_DIR}
   AND NOT "$ENV{LLVM_BUILD_DIR}" STREQUAL "")
  set(MLIR_DIR "$ENV{LLVM_BUILD_DIR}/lib/cmake/mlir" CACHE PATH
      "MLIR package directory derived from LLVM_BUILD_DIR")
endif()

project(ptoas VERSION 0.59)

# Wheel builds carry the same Linux hardening policy as release CMake builds.
# Including the cache settings here keeps the policy backend-independent.
if(SKBUILD AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/LinuxHardeningCache.cmake")
endif()

set(PTOAS_RELEASE_VERSION_OVERRIDE ""
    CACHE STRING "Override the version printed by ptoas --version")
# scikit-build-core owns wheel metadata and supplies its resolved PEP 621
# version here. The optional label distinguishes distributions that install the
# same CLI without making CMake a second source of package-version truth.
set(PTOAS_CLI_VERSION_LABEL ""
    CACHE STRING "Optional label prepended to the packaged ptoas version")

# Editable installs intentionally link against the developer's external LLVM
# build instead of bundling it. Apply that contract explicitly to every native
# Python target; generated and project-owned extensions live in different
# CMake directories and must use the same install RPATH policy.
function(ptoas_append_editable_llvm_rpath target_name)
  if(NOT SKBUILD_STATE STREQUAL "editable")
    return()
  endif()
  get_filename_component(_ptoas_editable_llvm_library_dir
    "${LLVM_BUILD_LIBRARY_DIR}" REALPATH)
  set_property(TARGET "${target_name}" APPEND PROPERTY
    INSTALL_RPATH "${_ptoas_editable_llvm_library_dir}")
endfunction()

set(PTOAS_EXACT_GIT_TAG_VERSION "")
if(EXISTS "${PROJECT_SOURCE_DIR}/.git")
  find_package(Git QUIET)
  if(GIT_FOUND)
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" -C "${PROJECT_SOURCE_DIR}"
              describe --tags --exact-match --match "v[0-9]*.[0-9]*"
      RESULT_VARIABLE PTOAS_GIT_DESCRIBE_RESULT
      OUTPUT_VARIABLE PTOAS_EXACT_GIT_TAG
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
    if(PTOAS_GIT_DESCRIBE_RESULT EQUAL 0
       AND PTOAS_EXACT_GIT_TAG MATCHES "^v([0-9]+\\.[0-9]+)$")
      set(PTOAS_EXACT_GIT_TAG_VERSION "${CMAKE_MATCH_1}")
    endif()
  endif()
endif()

set(PTOAS_CLI_VERSION "${PROJECT_VERSION}")
if(PTOAS_EXACT_GIT_TAG_VERSION)
  set(PTOAS_CLI_VERSION "${PTOAS_EXACT_GIT_TAG_VERSION}")
endif()
if(SKBUILD AND SKBUILD_PROJECT_VERSION_FULL)
  set(PTOAS_CLI_VERSION "${SKBUILD_PROJECT_VERSION_FULL}")
endif()
if(PTOAS_CLI_VERSION_LABEL)
  set(PTOAS_CLI_VERSION
      "${PTOAS_CLI_VERSION_LABEL} ${PTOAS_CLI_VERSION}")
endif()
if(PTOAS_RELEASE_VERSION_OVERRIDE)
  set(PTOAS_CLI_VERSION "${PTOAS_RELEASE_VERSION_OVERRIDE}")
endif()
message(STATUS "PTOAS CLI version: ${PTOAS_CLI_VERSION}")

# =========================================================
# [新增] 强制设置 C++17 标准 (LLVM/MLIR 必需)
# =========================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =========================================================
# 1. 显式寻找 LLVM (这是解决 tablegen 报错的关键)
# =========================================================
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "LLVM CMake Dir: ${LLVM_CMAKE_DIR}")
message(STATUS "LLVM Include Dir: ${LLVM_INCLUDE_DIRS}")
if(NOT DEFINED LLVM_VERSION_MAJOR)
  string(REGEX MATCH "^[0-9]+" LLVM_VERSION_MAJOR "${LLVM_PACKAGE_VERSION}")
endif()
if(NOT LLVM_VERSION_MAJOR STREQUAL "19")
  message(FATAL_ERROR
    "PTOAS requires LLVM 19, but found "
    "LLVM ${LLVM_PACKAGE_VERSION}. Please point LLVM_DIR/MLIR_DIR to an "
    "LLVM 19 build, for example vpto-dev/llvm-project:feature-vpto.")
endif()
# 将 LLVM 模块路径加入 CMake 搜索路径
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

# 关键：加载 LLVM 基础宏 (这里定义了 tablegen, add_llvm_pass 等)
include(AddLLVM)
include(TableGen) # 双重保险，确保 tablegen 宏可用
include(HandleLLVMOptions) # 处理 RTTI, EH 等编译选项

# =========================================================
# 2. 寻找 MLIR
# =========================================================
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Found MLIR ${MLIR_PACKAGE_VERSION}")
message(STATUS "MLIR CMake Dir: ${MLIR_CMAKE_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddMLIR)

# =========================================================
# 3. 配置构建环境
# =========================================================
# Keep warning policy in CMake so CI, wheel, and external build entrypoints
# all enforce the same diagnostics contract. Warnings are errors by default;
# directories that compile only third-party sources may opt out narrowly.
option(PTOAS_ENABLE_WERROR "Treat PTOAS warnings as errors" ON)
if(PTOAS_ENABLE_WERROR)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Werror>")
  elseif(MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/WX>")
  endif()
endif()

# macOS: 移除已废弃的 -undefined suppress，避免 ld 警告（保留 -undefined dynamic_lookup 即可）
if(APPLE)
  foreach(_var CMAKE_SHARED_MODULE_CREATE_CXX_FLAGS CMAKE_MODULE_LINKER_FLAGS)
    string(REPLACE "-Wl,-undefined -Wl,suppress" "" ${_var} "${${_var}}")
    string(REPLACE "-undefined suppress" "" ${_var} "${${_var}}")
  endforeach()
endif()

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# =========================================================
# 3.1 Testing option setup
# =========================================================
include(CTest)

# 开启 Python 绑定选项
option(PTO_ENABLE_PYTHON_BINDING "Enable Python bindings" ON)

include(cmake/VfSimulator.cmake)

if(PTO_ENABLE_PYTHON_BINDING)
  include(MLIRDetectPythonEnv)
  include(AddMLIRPython)
  mlir_configure_python_dev_packages()

  add_subdirectory(ptodsl)
endif()

# =========================================================
# 4. 添加子目录
# =========================================================
if(BUILD_TESTING AND NOT TARGET ptoas_runtime_deps)
  # Define this aggregate target before tools/ptoas runs so it can attach the
  # staged ptoas._core package as a dependency during configure.
  add_custom_target(ptoas_runtime_deps)
endif()
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

# =========================================================
# 4.1 Tests (ctest)
# =========================================================
if(BUILD_TESTING)
  enable_testing()
  if(PTO_ENABLE_PYTHON_BINDING)
    get_filename_component(_pto_python_bin_dir "${Python3_EXECUTABLE}" DIRECTORY)
    set(_pto_python_test_pythonpath
      "${CMAKE_BINARY_DIR}/python:${CMAKE_BINARY_DIR}/python/ptoas/_runtime/share/ptoas:${CMAKE_SOURCE_DIR}/lib:$ENV{PYTHONPATH}"
    )
    set(_pto_python_test_env
      "PYTHONPATH=${_pto_python_test_pythonpath}"
      "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:${LLVM_BUILD_LIBRARY_DIR}:${CMAKE_INSTALL_PREFIX}/lib:$ENV{LD_LIBRARY_PATH}"
      "PATH=${CMAKE_BINARY_DIR}/tools/ptoas:${CMAKE_INSTALL_PREFIX}/bin:${_pto_python_bin_dir}:$ENV{PATH}"
      "PTOAS_BIN=${CMAKE_BINARY_DIR}/tools/ptoas/ptoas"
    )

    add_subdirectory(ptodsl/tests)
    add_test(
      NAME pto_memory_consistency_bindings
      COMMAND "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/test/python/memory_consistency_bindings.py"
    )
    set_tests_properties(pto_memory_consistency_bindings PROPERTIES
      LABELS "PTO"
      ENVIRONMENT "${_pto_python_test_env}"
    )
    add_test(
      NAME pto_low_precision_types
      COMMAND "${Python3_EXECUTABLE}" -S
              "${CMAKE_CURRENT_SOURCE_DIR}/test/python/low_precision_types.py"
    )
    set_tests_properties(pto_low_precision_types PROPERTIES
      LABELS "PTO"
      ENVIRONMENT "${_pto_python_test_env}"
    )
    add_test(
      NAME pto_mx_grp_axis_bindings
      COMMAND "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/test/python/mx_grp_axis_bindings.py"
    )
    set_tests_properties(pto_mx_grp_axis_bindings PROPERTIES
      LABELS "PTO"
      ENVIRONMENT "${_pto_python_test_env}"
    )
    add_test(
      NAME pto_install_nightly_wheel
      COMMAND "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/test/python/test_install_nightly_wheel.py"
    )
    set_tests_properties(pto_install_nightly_wheel PROPERTIES
      LABELS "PTO"
      ENVIRONMENT "${_pto_python_test_env}"
    )
  endif()
  add_subdirectory(tools/ptobc/tests)
endif()

# =========================================================
# 4.2 Lit tests
# =========================================================
if(BUILD_TESTING)
  add_subdirectory(test/lit)

  set(_pto_check_ctest_depends)
  foreach(_target IN ITEMS
      PTOAS_Python
      PTODSLPackage
      ptoas_runtime_deps
      ptobc)
    if(TARGET ${_target})
      list(APPEND _pto_check_ctest_depends ${_target})
    endif()
  endforeach()

  if(NOT TARGET check-ctest)
    add_custom_target(check-ctest
      COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
      DEPENDS ${_pto_check_ctest_depends}
      USES_TERMINAL
      COMMENT "Running the CTest regression tests"
    )
    set_target_properties(check-ctest PROPERTIES FOLDER "Tests")
  endif()

  if(NOT TARGET check)
    add_custom_target(check)
  endif()
  if(TARGET check-pto)
    add_dependencies(check check-pto)
  endif()
  if(TARGET check-ctest)
    add_dependencies(check check-ctest)
  endif()
endif()

# =========================================================
# 5. 安装配置 (生成PTOASConfig.cmake)
# 将 include/PTO目录下的头文件安装到 <prefix>/include/PTO
install(DIRECTORY include/PTO
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT PTOAS_Development
        FILES_MATCHING PATTERN "*.h" PATTERN "*.td"
)
# 安装构建目录下的 .inc 生成文件
install(DIRECTORY ${PROJECT_BINARY_DIR}/include/PTO
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT PTOAS_Development
        FILES_MATCHING PATTERN "*.inc"
)

# 6. 定义 CMake 配置文件安装位置 (例如 lib/cmake/PTOAS)
set(PTOAS_CMAKE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/PTOAS")

# 7. 导出 Targets (PTOASTargets.cmake) 供 find_package 使用
install(EXPORT PTOASTargets
        FILE PTOASTargets.cmake
        NAMESPACE PTOAS::
        DESTINATION ${PTOAS_CMAKE_INSTALL_DIR}
)

# 8. 生成 Config 文件 (PTOASConfig.cmake)
include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/PTOASConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/PTOASConfig.cmake"
    INSTALL_DESTINATION ${PTOAS_CMAKE_INSTALL_DIR}
    PATH_VARS CMAKE_INSTALL_INCLUDEDIR
)

# 9. 安装 Config 文件
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/PTOASConfig.cmake"
    DESTINATION ${PTOAS_CMAKE_INSTALL_DIR}
)

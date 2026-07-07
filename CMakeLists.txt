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

function(ptoas_get_cmake_cache_value cache_file variable out_var)
  if(NOT EXISTS "${cache_file}")
    set(${out_var} "" PARENT_SCOPE)
    return()
  endif()

  file(STRINGS "${cache_file}" cache_lines
       REGEX "^${variable}:[^=]*=")
  if(cache_lines)
    list(GET cache_lines 0 cache_line)
    string(REGEX REPLACE "^[^=]*=" "" cache_value "${cache_line}")
    set(${out_var} "${cache_value}" PARENT_SCOPE)
  else()
    set(${out_var} "" PARENT_SCOPE)
  endif()
endfunction()

function(ptoas_get_llvm_build_dir_from_llvm_dir out_var)
  if(NOT DEFINED LLVM_DIR OR LLVM_DIR STREQUAL "")
    set(${out_var} "" PARENT_SCOPE)
    return()
  endif()

  get_filename_component(llvm_cmake_dir "${LLVM_DIR}" REALPATH)
  get_filename_component(llvm_build_dir "${llvm_cmake_dir}/../../.." REALPATH)
  if(EXISTS "${llvm_build_dir}/CMakeCache.txt")
    set(${out_var} "${llvm_build_dir}" PARENT_SCOPE)
  else()
    set(${out_var} "" PARENT_SCOPE)
  endif()
endfunction()

function(ptoas_preseed_compilers_from_llvm_cache)
  ptoas_get_llvm_build_dir_from_llvm_dir(llvm_build_dir)
  if(llvm_build_dir STREQUAL "")
    return()
  endif()

  set(llvm_cache "${llvm_build_dir}/CMakeCache.txt")
  if(NOT DEFINED CMAKE_C_COMPILER AND "$ENV{CC}" STREQUAL "")
    ptoas_get_cmake_cache_value("${llvm_cache}" "CMAKE_C_COMPILER"
                                llvm_c_compiler)
    if(llvm_c_compiler AND EXISTS "${llvm_c_compiler}")
      set(CMAKE_C_COMPILER "${llvm_c_compiler}" CACHE FILEPATH
          "C compiler inherited from the configured LLVM build" FORCE)
    endif()
  endif()

  if(NOT DEFINED CMAKE_CXX_COMPILER AND "$ENV{CXX}" STREQUAL "")
    ptoas_get_cmake_cache_value("${llvm_cache}" "CMAKE_CXX_COMPILER"
                                llvm_cxx_compiler)
    if(llvm_cxx_compiler AND EXISTS "${llvm_cxx_compiler}")
      set(CMAKE_CXX_COMPILER "${llvm_cxx_compiler}" CACHE FILEPATH
          "C++ compiler inherited from the configured LLVM build" FORCE)
    endif()
  endif()
endfunction()

ptoas_preseed_compilers_from_llvm_cache()

project(ptoas VERSION 0.49)

option(PTOAS_VALIDATE_CMAKE4_FETCHCONTENT_COMPAT
       "Validate CMake 4 compatibility for legacy FetchContent users" OFF)
if(PTOAS_VALIDATE_CMAKE4_FETCHCONTENT_COMPAT)
  set(_ptoas_cmp0169_check_dir "${CMAKE_BINARY_DIR}/cmake-cmp0169-compat")
  file(MAKE_DIRECTORY "${_ptoas_cmp0169_check_dir}/nested/cann-cmake")
  file(WRITE "${_ptoas_cmp0169_check_dir}/nested/cann-cmake/CMakeLists.txt"
       "cmake_minimum_required(VERSION 3.20)\n"
       "project(ptoas_cmp0169_cann_cmake_fixture NONE)\n")
  file(WRITE "${_ptoas_cmp0169_check_dir}/nested/CMakeLists.txt" [=[
cmake_minimum_required(VERSION 3.20)
project(ptoas_cmp0169_nested_fixture NONE)
include(FetchContent)
FetchContent_Declare(cann-cmake SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/cann-cmake")
FetchContent_GetProperties(cann-cmake)
if(NOT cann-cmake_POPULATED)
  FetchContent_Populate(cann-cmake)
endif()
]=])
  add_subdirectory("${_ptoas_cmp0169_check_dir}/nested"
                   "${_ptoas_cmp0169_check_dir}/nested-build")
endif()

set(PTOAS_RELEASE_VERSION_OVERRIDE ""
    CACHE STRING "Override the version printed by ptoas --version")
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
if(NOT LLVM_VERSION_MAJOR STREQUAL "21")
  message(FATAL_ERROR
    "PTOAS requires LLVM 21 after the LLVM21 upgrade, but found "
    "LLVM ${LLVM_PACKAGE_VERSION}. Please point LLVM_DIR/MLIR_DIR to an "
    "LLVM 21 build, for example vpto-dev/llvm-project:feature-vpto-llvm21.")
endif()
get_filename_component(PTO_LLVM_BUILD_LIBRARY_DIR
  "${LLVM_BUILD_LIBRARY_DIR}" REALPATH)
ptoas_get_llvm_build_dir_from_llvm_dir(PTO_LLVM_BUILD_DIR)
if(PTO_LLVM_BUILD_DIR)
  ptoas_get_cmake_cache_value("${PTO_LLVM_BUILD_DIR}/CMakeCache.txt"
                              "CMAKE_CXX_COMPILER"
                              PTO_LLVM_CXX_COMPILER)
endif()
if(PTO_LLVM_CXX_COMPILER AND EXISTS "${PTO_LLVM_CXX_COMPILER}")
  execute_process(
    COMMAND "${PTO_LLVM_CXX_COMPILER}" --version
    OUTPUT_VARIABLE PTO_LLVM_CXX_VERSION_TEXT
    ERROR_VARIABLE PTO_LLVM_CXX_VERSION_TEXT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE)
  if(PTO_LLVM_CXX_VERSION_TEXT MATCHES "[Cc]lang"
     AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(FATAL_ERROR
      "PTOAS is being configured with GNU C++ (${CMAKE_CXX_COMPILER}), "
      "but the selected LLVM build was compiled with a Clang-compatible "
      "compiler (${PTO_LLVM_CXX_COMPILER}). MLIR TypeID fallback strings are "
      "compiler-family sensitive across shared libraries; use "
      "-DCMAKE_CXX_COMPILER=${PTO_LLVM_CXX_COMPILER} or leave the compiler "
      "unset so PTOAS can inherit it from LLVM_DIR.")
  endif()
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
# all enforce the same diagnostics contract.
option(PTOAS_ENABLE_WERROR "Treat PTOAS warnings as errors" ON)
if(PTOAS_ENABLE_WERROR)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Werror>)
  elseif(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/WX>)
  endif()
endif()

option(PTOAS_ENABLE_PATCHED_LLVM_EXTENSIONS
       "Build code paths that depend on PTOAS LLVM patch-only APIs" ON)
if(PTOAS_ENABLE_PATCHED_LLVM_EXTENSIONS)
  add_compile_definitions(PTOAS_ENABLE_PATCHED_LLVM_EXTENSIONS=1)
else()
  add_compile_definitions(PTOAS_ENABLE_PATCHED_LLVM_EXTENSIONS=0)
endif()

# macOS: 移除已废弃的 -undefined suppress，避免 ld 警告（保留 -undefined dynamic_lookup 即可）
if(APPLE)
  foreach(_var CMAKE_SHARED_MODULE_CREATE_CXX_FLAGS CMAKE_MODULE_LINKER_FLAGS)
    string(REPLACE "-Wl,-undefined -Wl,suppress" "" ${_var} "${${_var}}")
    string(REPLACE "-undefined suppress" "" ${_var} "${${_var}}")
  endforeach()
endif()

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

link_directories(${PTO_LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# =========================================================
# 3.1 Testing option setup
# =========================================================
include(CTest)

# 开启 Python 绑定选项
option(PTO_ENABLE_PYTHON_BINDING "Enable Python bindings" ON)

if(PTO_ENABLE_PYTHON_BINDING)
  find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)

  # 设置 Python 扩展的输出目录，方便调试
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python/pto)

  add_subdirectory(python)
  add_subdirectory(ptodsl)
  add_subdirectory(tilelang-dsl)
endif()

# =========================================================
# 4. 添加子目录
# =========================================================
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

# =========================================================
# 4.1 Tests (ctest)
# =========================================================
if(BUILD_TESTING)
  enable_testing()
  if(PTO_ENABLE_PYTHON_BINDING)
    get_filename_component(_llvm_root "${LLVM_DIR}/../../.." ABSOLUTE)
    set(_pto_python_test_pythonpath
      "${CMAKE_BINARY_DIR}/python:${MLIR_PYTHON_PACKAGE_DIR}:${CMAKE_INSTALL_PREFIX}:$ENV{PYTHONPATH}"
    )
    if(DEFINED ENV{PTO_INSTALL_DIR} AND NOT "$ENV{PTO_INSTALL_DIR}" STREQUAL "")
      set(_pto_python_test_pythonpath "$ENV{PTO_INSTALL_DIR}:${_pto_python_test_pythonpath}")
    endif()
    set(_pto_python_test_env
      "PYTHONPATH=${_pto_python_test_pythonpath}"
      "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:${_llvm_root}/lib:${CMAKE_INSTALL_PREFIX}/lib:$ENV{LD_LIBRARY_PATH}"
      "PATH=${CMAKE_BINARY_DIR}/tools/ptoas:${CMAKE_INSTALL_PREFIX}/bin:$ENV{PATH}"
    )

    add_subdirectory(ptodsl/tests)
    add_test(
      NAME tilelang_dsl_import
      COMMAND "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/tilelang-dsl/tests/import_tilelang_dsl.py"
    )
    set_tests_properties(tilelang_dsl_import PROPERTIES
      LABELS "PTODSL"
      ENVIRONMENT "${_pto_python_test_env}"
    )
    add_test(
      NAME tilelang_dsl_unittest
      COMMAND "${Python3_EXECUTABLE}" -m unittest discover
              -s "${CMAKE_CURRENT_SOURCE_DIR}/tilelang-dsl/tests"
              -p "test_*.py"
    )
    set_tests_properties(tilelang_dsl_unittest PROPERTIES
      LABELS "PTODSL"
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
  foreach(_target IN ITEMS PTOPythonModules pto-opt ptoas ptobc)
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

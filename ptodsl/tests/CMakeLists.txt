# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

if(NOT DEFINED _pto_python_test_env)
  get_filename_component(_pto_python_bin_dir "${Python3_EXECUTABLE}" DIRECTORY)
  set(_ptodsl_test_pythonpath
    "${CMAKE_BINARY_DIR}/python:${CMAKE_BINARY_DIR}/python/ptoas/_runtime/share/ptoas:${CMAKE_SOURCE_DIR}/lib:$ENV{PYTHONPATH}"
  )
  set(_pto_python_test_env
    "PYTHONPATH=${_ptodsl_test_pythonpath}"
    "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:${LLVM_BUILD_LIBRARY_DIR}:${CMAKE_INSTALL_PREFIX}/lib:$ENV{LD_LIBRARY_PATH}"
    "PATH=${CMAKE_BINARY_DIR}/tools/ptoas:${CMAKE_INSTALL_PREFIX}/bin:${_pto_python_bin_dir}:$ENV{PATH}"
    "PTOAS_BIN=${CMAKE_BINARY_DIR}/tools/ptoas/ptoas"
  )
endif()

function(add_ptodsl_python_test test_name script_path)
  add_test(
    NAME ${test_name}
    COMMAND "${Python3_EXECUTABLE}" ${ARGN} "${CMAKE_SOURCE_DIR}/${script_path}"
  )
  set_tests_properties(${test_name} PROPERTIES
    LABELS "PTODSL"
    ENVIRONMENT "${_pto_python_test_env}"
  )
endfunction()

# Shell-level regressions live next to the Python ones and follow the same
# naming convention, so one ctest label covers both.
function(add_ptodsl_shell_test test_name script_path)
  add_test(
    NAME ${test_name}
    COMMAND bash "${CMAKE_SOURCE_DIR}/${script_path}"
  )
  set_tests_properties(${test_name} PROPERTIES
    LABELS "PTODSL"
    ENVIRONMENT "${_pto_python_test_env}"
  )
endfunction()

# Auto-discover PTODSL Python regressions by filename convention:
#   - ptodsl/tests/test_*.py
file(
  GLOB _ptodsl_test_files
  CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/test_*.py"
)

list(SORT _ptodsl_test_files)

# These modules use pytest fixtures and parameterization.
set(_ptodsl_pytest_tests ptodsl_expand_dtype_regressions ptodsl_tile_div_dispatch)

foreach(test_file IN LISTS _ptodsl_test_files)
  file(RELATIVE_PATH rel_path "${CMAKE_SOURCE_DIR}" "${test_file}")
  get_filename_component(stem "${test_file}" NAME_WE)

  if(stem MATCHES "^test_")
    string(REGEX REPLACE "^test_" "" stem_suffix "${stem}")
    set(test_name "ptodsl_${stem_suffix}")
  else()
    set(test_name "ptodsl_${stem}")
  endif()

  if(test_name IN_LIST _ptodsl_pytest_tests)
    add_ptodsl_python_test("${test_name}" "${rel_path}" -m pytest -q --import-mode=importlib)
  else()
    add_ptodsl_python_test("${test_name}" "${rel_path}")
  endif()
endforeach()

# Shell regressions under the same convention: ptodsl/tests/test_*.sh
file(
  GLOB _ptodsl_shell_test_files
  CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/test_*.sh"
)

list(SORT _ptodsl_shell_test_files)

foreach(test_file IN LISTS _ptodsl_shell_test_files)
  file(RELATIVE_PATH rel_path "${CMAKE_SOURCE_DIR}" "${test_file}")
  get_filename_component(stem "${test_file}" NAME_WE)
  string(REGEX REPLACE "^test_" "" stem_suffix "${stem}")
  add_ptodsl_shell_test("ptodsl_${stem_suffix}" "${rel_path}")
endforeach()

if(NOT TARGET check-dsl)
  set(_ptodsl_check_depends)
  foreach(_target IN ITEMS PTODSLPackage PTOAS_Python ptoas_runtime_deps)
    if(TARGET ${_target})
      list(APPEND _ptodsl_check_depends ${_target})
    endif()
  endforeach()

  add_custom_target(check-dsl
    COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure -L PTODSL
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    DEPENDS ${_ptodsl_check_depends}
    USES_TERMINAL
    COMMENT "Running the PTODSL regression tests"
  )
  set_target_properties(check-dsl PROPERTIES FOLDER "Tests")
endif()

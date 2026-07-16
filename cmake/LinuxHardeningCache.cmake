# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Shared Linux hardening flags for LLVM/PTOAS package builds.
# This cache file is intended to be passed via `cmake -C ...` so release and
# delivery builds inherit the same compiler options that codecheck expects.
# Keep the Linux wheel/toolchain target on the x86-64 baseline only for x86
# hosts so auditwheel can repair the shipped ptoas binary into a portable
# manylinux wheel without breaking aarch64 builds.

execute_process(
  COMMAND uname -m
  OUTPUT_VARIABLE _ptoas_linux_machine
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)
string(TOLOWER "${_ptoas_linux_machine}" _ptoas_linux_machine)
set(_ptoas_linux_x86_64_baseline_flags "")
if(_ptoas_linux_machine MATCHES "^(x86_64|amd64|i[3-6]86)$")
  set(_ptoas_linux_x86_64_baseline_flags -march=x86-64 -mtune=generic)
endif()

set(_ptoas_linux_compile_flag_vars
    CMAKE_C_FLAGS
    CMAKE_CXX_FLAGS
    CMAKE_C_FLAGS_DEBUG
    CMAKE_C_FLAGS_MINSIZEREL
    CMAKE_C_FLAGS_RELEASE
    CMAKE_C_FLAGS_RELWITHDEBINFO
    CMAKE_CXX_FLAGS_DEBUG
    CMAKE_CXX_FLAGS_MINSIZEREL
    CMAKE_CXX_FLAGS_RELEASE
    CMAKE_CXX_FLAGS_RELWITHDEBINFO)

foreach(_flag_var IN LISTS _ptoas_linux_compile_flag_vars)
  set(_flag_value "${${_flag_var}}")
  foreach(_hardening_flag
      -D_FORTIFY_SOURCE=2
      -fstack-protector-strong
      -ftrapv
      ${_ptoas_linux_x86_64_baseline_flags})
    if(NOT " ${_flag_value} " MATCHES "(^| )${_hardening_flag}( |$)")
      string(APPEND _flag_value " ${_hardening_flag}")
    endif()
  endforeach()
  string(STRIP "${_flag_value}" _flag_value)
  set(${_flag_var} "${_flag_value}" CACHE STRING "Linux hardening flags" FORCE)
endforeach()

foreach(_flag_var
    CMAKE_EXE_LINKER_FLAGS
    CMAKE_SHARED_LINKER_FLAGS
    CMAKE_MODULE_LINKER_FLAGS)
  set(_flag_value "${${_flag_var}}")
  foreach(_hardening_flag
      -Wl,-z,relro
      -Wl,-z,now)
    if(NOT " ${_flag_value} " MATCHES "(^| )${_hardening_flag}( |$)")
      string(APPEND _flag_value " ${_hardening_flag}")
    endif()
  endforeach()
  string(STRIP "${_flag_value}" _flag_value)
  set(${_flag_var} "${_flag_value}" CACHE STRING "Linux hardening linker flags" FORCE)
endforeach()

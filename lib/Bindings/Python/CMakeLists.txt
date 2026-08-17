# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

include(AddMLIRPython)

# Keep the complete PTOAS-owned MLIR Python runtime below the project namespace
# so it can coexist with independently packaged MLIR distributions. This is a
# package-prefix contract as well as a Python import path: MLIR's generated
# capsules and pybind11 registration code use the same value.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=ptoas.mlir.")

# Do not carry toolchain-injected paths, such as a Conda compiler RPATH, into
# installed Python modules. Wheel builds retain AddMLIRPython's package-relative
# RPATHs for repair; editable builds append only the external LLVM directory.
set(CMAKE_INSTALL_REMOVE_ENVIRONMENT_RPATH ON)

# This directory materializes C++ sources exported by upstream MLIR. PTOAS
# keeps -Werror enabled globally, but upstream sources are not part of the
# project warning contract. Scope the exception to this directory; the
# project-owned ptoas._core extension is built in tools/ptoas and therefore
# remains under -Werror.
if(PTOAS_ENABLE_WERROR)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wno-error>")
  elseif(MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/WX->")
  endif()
endif()

# Build the `ptoas.mlir` Python package from MLIR's exported source declarations and
# add PTOAS's dialect to the same package. This is the composition mechanism
# supported by AddMLIRPython and avoids copying or overlaying another build's
# package tree.
set(PTOAS_MLIR_PYTHON_PACKAGE_DIR "${CMAKE_BINARY_DIR}/python/ptoas/mlir")

declare_mlir_python_sources(PTOASPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT PTOASPythonSources
  ROOT_DIR "${CMAKE_SOURCE_DIR}/python/pto"
  TD_FILE dialects/PTOOps.td
  SOURCES dialects/pto.py
  DIALECT_NAME pto
)

# Use MLIR's standard site-initializer hook to seed each Python Context with
# the selected upstream dialects shipped by PTOAS. Declaring the extension in
# the Python source graph lets AddMLIRPython own its build, installation, and
# common-CAPI dependency collection.
set(PTOAS_PYTHON_REGISTRATION_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/PythonRegistration.cpp")
declare_mlir_python_extension(PTOASPythonSources.Registration
  MODULE_NAME _site_initialize_0
  ADD_TO_PARENT PTOASPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES PythonRegistration.cpp
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIArith
    MLIRCAPIFunc
    MLIRCAPILLVM
    MLIRCAPIMath
    MLIRCAPIMemRef
    MLIRCAPISCF
)

# This directory otherwise materializes only upstream MLIR/pybind11 C++
# sources under -Wno-error. Keep the project-owned registration source under
# PTOAS's default strict warning policy while retaining one declarative Python
# package graph.
if(PTOAS_ENABLE_WERROR)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    set_property(SOURCE "${PTOAS_PYTHON_REGISTRATION_SOURCE}" APPEND PROPERTY
      COMPILE_OPTIONS -Werror)
  elseif(MSVC)
    set_property(SOURCE "${PTOAS_PYTHON_REGISTRATION_SOURCE}" APPEND PROPERTY
      COMPILE_OPTIONS /WX)
  endif()
endif()

# Package only the upstream Python APIs used by PTOAS and PTODSL. In
# particular, do not instantiate MLIRPythonSources or RegisterEverything:
# downstream MLIR packages are expected to select their public dialect set
# explicitly rather than linking every upstream dialect, translation and pass.
set(PTOAS_MLIR_PYTHON_SOURCE_SETS
  MLIRPythonSources.Core
  MLIRPythonSources.Dialects.arith
  MLIRPythonSources.Dialects.builtin
  MLIRPythonSources.Dialects.func
  MLIRPythonSources.Dialects.llvm
  MLIRPythonSources.Dialects.math
  MLIRPythonSources.Dialects.memref
  MLIRPythonSources.Dialects.scf
)

# Keep the MLIR Python CAPI and PTOAS compiler implementation in one package-
# owned DSO. Generated extensions and the thin ptoas._core entry point all
# resolve this target, so Python-created and compiler-created IR share one
# native MLIR runtime.
add_mlir_python_common_capi_library(PTOASCompiler
  INSTALL_COMPONENT PTOAS_Python
  INSTALL_DESTINATION "ptoas/mlir/_mlir_libs"
  OUTPUT_DIRECTORY "${PTOAS_MLIR_PYTHON_PACKAGE_DIR}/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../"
  DECLARED_HEADERS
    MLIRPythonCAPI.HeaderSources
  DECLARED_SOURCES
    ${PTOAS_MLIR_PYTHON_SOURCE_SETS}
    PTOASPythonSources
  # MLIRPythonExtension.Core currently obtains its rewrite CAPI through
  # RegisterEverything. PTOAS deliberately omits RegisterEverything, so retain
  # the missing direct dependency until upstream Core declares it itself.
  EMBED_LIBS
    MLIRCAPITransforms
    PTOCAPI
    PTOIR
    PTOTransforms
    ptobc_lib
    PTOASCompilerImplementation
)
# Imported MLIR aggregate objects do not expose a source language from which
# CMake can infer the linker driver. The common CAPI is C++, matching MLIR's
# own in-tree target.
set_property(TARGET PTOASCompiler PROPERTY LINKER_LANGUAGE CXX)
# This is a package-private same-build boundary, not an LLVM-style SDK ABI.
# Avoid installing both a versioned file and its symlink: wheel archives
# dereference the symlink and would otherwise store the DSO twice.
set_property(TARGET PTOASCompiler PROPERTY VERSION)
set_property(TARGET PTOASCompiler PROPERTY SOVERSION)

add_mlir_python_modules(PTOAS_Python
  ROOT_PREFIX "${PTOAS_MLIR_PYTHON_PACKAGE_DIR}"
  INSTALL_PREFIX ptoas/mlir
  DECLARED_SOURCES
    ${PTOAS_MLIR_PYTHON_SOURCE_SETS}
    PTOASPythonSources
  COMMON_CAPI_LINK_LIBS
    PTOASCompiler
)

# AddMLIRPython deliberately resets each generated target's install RPATH to
# its package-relative layout. Append the external LLVM directory afterwards
# for editable installs, where those dependencies are not bundled into the
# Python environment. Discover the native targets by CMake type so additional
# generated MLIR extensions inherit the same contract automatically.
if(SKBUILD_STATE STREQUAL "editable")
  get_property(_ptoas_python_directory_targets
    DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
  foreach(_ptoas_python_target IN LISTS _ptoas_python_directory_targets)
    get_target_property(_ptoas_python_target_type
      "${_ptoas_python_target}" TYPE)
    if(_ptoas_python_target_type MATCHES "^(MODULE|SHARED)_LIBRARY$")
      ptoas_append_editable_llvm_rpath("${_ptoas_python_target}")
    endif()
  endforeach()
endif()

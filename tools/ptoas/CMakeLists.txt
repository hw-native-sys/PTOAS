# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Do not carry toolchain-injected paths, such as a Conda compiler RPATH, into
# installed Python modules. Distribution builds keep only package-relative
# RPATHs; editable builds add the external LLVM directory explicitly below.
set(CMAKE_INSTALL_REMOVE_ENVIRONMENT_RPATH ON)

set(PTOAS_RUNTIME_SOURCES
  ptoas.cpp
  ptoas_name_hints.cpp
  ptoas_cpp_rewrite.cpp
  ptoas_pipeline.cpp
  driver.cpp
  VPTOHostStubEmission.cpp
  ObjectEmission.cpp
)

add_library(PTOASVFSIMTSizePatcher STATIC
  VFSIMTSizePatcher.cpp
)
llvm_update_compile_flags(PTOASVFSIMTSizePatcher)
target_link_libraries(PTOASVFSIMTSizePatcher PUBLIC
  LLVMCore
  LLVMObject
  LLVMSupport
  MLIRSupport
)

set(PTOAS_WRAPPER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/ptoas_wrapper.py")
set(PTOAS_WRAPPER_BINARY "${CMAKE_CURRENT_BINARY_DIR}/ptoas")
set(PTOAS_INSTALL_WRAPPER_BINARY "${CMAKE_CURRENT_BINARY_DIR}/ptoas-install")

set(PTOAS_WRAPPER_PYTHON_ROOT_MODE "absolute")
set(PTOAS_WRAPPER_PYTHON_ROOT "${CMAKE_BINARY_DIR}/python")
configure_file(
  "${PTOAS_WRAPPER_SOURCE}"
  "${PTOAS_WRAPPER_BINARY}"
  @ONLY
)

# CMake install prefixes may be overridden at install time. Resolve the
# installed package relative to bin/ptoas instead of baking in the configure
# prefix.
set(PTOAS_WRAPPER_PYTHON_ROOT_MODE "wrapper-relative")
set(PTOAS_WRAPPER_PYTHON_ROOT "..")
configure_file(
  "${PTOAS_WRAPPER_SOURCE}"
  "${PTOAS_INSTALL_WRAPPER_BINARY}"
  @ONLY
)

foreach(_ptoas_wrapper IN ITEMS
    "${PTOAS_WRAPPER_BINARY}"
    "${PTOAS_INSTALL_WRAPPER_BINARY}")
  file(CHMOD "${_ptoas_wrapper}"
    PERMISSIONS
      OWNER_READ OWNER_WRITE OWNER_EXECUTE
      GROUP_READ GROUP_EXECUTE
      WORLD_READ WORLD_EXECUTE
  )
endforeach()

function(ptoas_configure_runtime_compile_target target_name)
  target_compile_definitions(${target_name} PRIVATE
    PTOAS_RELEASE_VERSION="${PTOAS_CLI_VERSION}"
    ${ARGN}
  )
  add_dependencies(${target_name}
    PTOOpsIncGen
  )
endfunction()

set(PTOAS_RUNTIME_LINK_LIBS
  PTOIR
  PTOTransforms
  PTOASVFSIMTSizePatcher
  ptobc_lib
  MLIRMlirOptMain
  MLIRBufferizationTransforms
  MLIRArithTransforms
  MLIRTensorTransforms
  MLIRFuncTransforms
  MLIRIR
  MLIRParser
  MLIRPass
  MLIREmitCDialect
  MLIRSCFDialect
  MLIRGPUDialect
  MLIRTransforms
  MLIRTargetCpp
  MLIRAffineDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRControlFlowDialect
  MLIRBufferizationDialect
  MLIRFuncDialect
  MLIRFuncInlinerExtension
  MLIRLLVMIRTransforms
  MLIRSupport
  LLVMOption
  MLIRTransformUtils
)

# This aggregation root owns all compiler and PTO Python binding objects. The
# final shared library is declared earlier as PTOASCompiler; AddMLIR resolves
# this forward target relationship during generation.
add_mlir_library(PTOASCompilerImplementation
  OBJECT
  ENABLE_AGGREGATION
  EXCLUDE_FROM_LIBMLIR
  DISABLE_INSTALL
  PARTIAL_SOURCES_INTENDED

  ${PTOAS_RUNTIME_SOURCES}

  LINK_LIBS PUBLIC
  ${PTOAS_RUNTIME_LINK_LIBS}
)
ptoas_configure_runtime_compile_target(obj.PTOASCompilerImplementation)
target_compile_definitions(obj.PTOASCompilerImplementation PRIVATE
  "MLIR_PYTHON_PACKAGE_PREFIX=ptoas.mlir."
  PTOAS_COMPILER_BUILDING_LIBRARY=1
)
target_include_directories(PTOASCompilerImplementation PRIVATE
  "${CMAKE_SOURCE_DIR}/lib/Bindings/Python"
)

set_target_properties(PTOASCompiler PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
)

# Build the single PTOAS-owned Python extension as a thin entry point. Compiler,
# PTO dialect, and MLIR CAPI implementation all live in PTOASCompiler.
set(PTOAS_BUILD_PYTHON_PACKAGE_DIR "${CMAKE_BINARY_DIR}/python/ptoas")
add_mlir_python_extension(PTOASPythonCore _core
  INSTALL_COMPONENT PTOAS_Python
  INSTALL_DIR ptoas
  OUTPUT_DIRECTORY "${PTOAS_BUILD_PYTHON_PACKAGE_DIR}"
  SOURCES
    NativeModule.cpp
    "${CMAKE_SOURCE_DIR}/lib/Bindings/Python/PTOModule.cpp"
  LINK_LIBS
    PTOASCompiler
)
target_compile_definitions(PTOASPythonCore PRIVATE
  "MLIR_PYTHON_PACKAGE_PREFIX=ptoas.mlir."
)
# PTOModule.cpp now defines populatePTODialectBindings inside _core itself
# rather than importing it from the DSO, so mark the defining TU as the
# library-building side (dllexport on Windows; default visibility elsewhere).
set_property(SOURCE "${CMAKE_SOURCE_DIR}/lib/Bindings/Python/PTOModule.cpp"
  APPEND PROPERTY COMPILE_DEFINITIONS PTOAS_COMPILER_BUILDING_LIBRARY=1)
target_include_directories(PTOASPythonCore PRIVATE
  "${CMAKE_SOURCE_DIR}/lib/Bindings/Python"
)
mlir_python_setup_extension_rpath(PTOASPythonCore)

if(APPLE)
  set_property(TARGET PTOASPythonCore PROPERTY INSTALL_RPATH
    "@loader_path;@loader_path/mlir/_mlir_libs")
elseif(UNIX)
  set_property(TARGET PTOASPythonCore PROPERTY INSTALL_RPATH
    "$ORIGIN;$ORIGIN/mlir/_mlir_libs")
endif()
ptoas_append_editable_llvm_rpath(PTOASPythonCore)

# declare_mlir_python_sources' SOURCES_GLOB resolves to a bare file(GLOB_RECURSE)
# without CONFIGURE_DEPENDS, so the .py file list is captured only at configure
# time. On an incrementally reused build tree (e.g. a persistent CI worker), a
# newly added source is never re-globbed and its staging symlink is silently
# missing, surfacing as a ModuleNotFoundError at runtime. Collect the sources
# ourselves with CONFIGURE_DEPENDS and pass them via SOURCES so adding/removing
# a .py under these roots forces a reconfigure.
file(GLOB_RECURSE _ptoas_runtime_py_sources CONFIGURE_DEPENDS
  RELATIVE "${CMAKE_SOURCE_DIR}/ptodsl/ptoas"
  "${CMAKE_SOURCE_DIR}/ptodsl/ptoas/*.py"
)
declare_mlir_python_sources(PTOASRuntimePythonSources
  ROOT_DIR "${CMAKE_SOURCE_DIR}/ptodsl/ptoas"
  SOURCES ${_ptoas_runtime_py_sources}
)
add_mlir_python_sources_target(PTOASRuntimePythonSourcesStage
  OUTPUT_DIRECTORY "${PTOAS_BUILD_PYTHON_PACKAGE_DIR}"
  SOURCES_TARGETS PTOASRuntimePythonSources
)

add_dependencies(PTOASRuntimePythonSourcesStage PTODSLPackageStage)

declare_mlir_python_sources(PTOCostModelSources
  ROOT_DIR "${CMAKE_SOURCE_DIR}/ptodsl/pto_costmodel"
  SOURCES_GLOB "*.py" "*.json"
)
add_mlir_python_sources_target(PTOCostModelSourcesStage
  OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/pto_costmodel"
  SOURCES_TARGETS PTOCostModelSources
)
add_dependencies(PTOASRuntimePythonSourcesStage PTOCostModelSourcesStage)

file(GLOB_RECURSE _ptoas_tileops_py_sources CONFIGURE_DEPENDS
  RELATIVE "${CMAKE_SOURCE_DIR}/lib/TileOps"
  "${CMAKE_SOURCE_DIR}/lib/TileOps/*.py"
)
declare_mlir_python_sources(PTOASTileOpsPythonSources
  ROOT_DIR "${CMAKE_SOURCE_DIR}/lib/TileOps"
  SOURCES ${_ptoas_tileops_py_sources}
)
add_mlir_python_sources_target(PTOASTileOpsPythonSourcesStage
  OUTPUT_DIRECTORY
  "${PTOAS_BUILD_PYTHON_PACKAGE_DIR}/_runtime/share/ptoas/TileOps"
  SOURCES_TARGETS PTOASTileOpsPythonSources
)

file(GLOB_RECURSE _ptoas_softops_py_sources CONFIGURE_DEPENDS
  RELATIVE "${CMAKE_SOURCE_DIR}/lib/SoftOps"
  "${CMAKE_SOURCE_DIR}/lib/SoftOps/*.py"
)
declare_mlir_python_sources(PTOASSoftOpsPythonSources
  ROOT_DIR "${CMAKE_SOURCE_DIR}/lib/SoftOps"
  SOURCES ${_ptoas_softops_py_sources}
)
add_mlir_python_sources_target(PTOASSoftOpsPythonSourcesStage
  OUTPUT_DIRECTORY
  "${PTOAS_BUILD_PYTHON_PACKAGE_DIR}/_runtime/share/ptoas/SoftOps"
  SOURCES_TARGETS PTOASSoftOpsPythonSources
)

add_custom_target(PTOASPythonPackage ALL)
add_dependencies(PTOASPythonPackage
  PTOASPythonCore
  PTOAS_Python
  PTOASRuntimePythonSourcesStage
  PTOASTileOpsPythonSourcesStage
  PTOASSoftOpsPythonSourcesStage
)
if(TARGET PTODSLPackage)
  add_dependencies(PTOASPythonPackage PTODSLPackage)
endif()

if(TARGET ptoas_runtime_deps)
  add_dependencies(ptoas_runtime_deps PTOASPythonPackage)
endif()

install(PROGRAMS "${PTOAS_INSTALL_WRAPPER_BINARY}"
  DESTINATION bin
  RENAME ptoas
  COMPONENT PTOAS_CompilerArchive
)

set(PTOAS_ARCHIVE_PYTHON_VERSION_FILE
    "${CMAKE_CURRENT_BINARY_DIR}/.ptoas-python-version")
file(WRITE "${PTOAS_ARCHIVE_PYTHON_VERSION_FILE}"
  "${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}\n")
install(FILES "${PTOAS_ARCHIVE_PYTHON_VERSION_FILE}"
  DESTINATION "."
  COMPONENT PTOAS_CompilerArchive
)

# The standalone compiler archive is staged by installing PTOAS_Python first
# and PTOAS_CompilerArchive second into the same prefix. Generate the deploy
# script so known native entry points come from CMake targets; the script scans
# only AddMLIRPython's generated extension directory for the remaining modules.
configure_file(
  "${CMAKE_SOURCE_DIR}/cmake/RelocatePTOASCompilerArchive.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/RelocatePTOASCompilerArchive.cmake.in"
  @ONLY
)
file(GENERATE
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/RelocatePTOASCompilerArchive.cmake"
  INPUT "${CMAKE_CURRENT_BINARY_DIR}/RelocatePTOASCompilerArchive.cmake.in"
)
install(SCRIPT
  "${CMAKE_CURRENT_BINARY_DIR}/RelocatePTOASCompilerArchive.cmake"
  COMPONENT PTOAS_CompilerArchive
)

install(
  DIRECTORY "${CMAKE_SOURCE_DIR}/lib/TileOps"
  DESTINATION "ptoas/_runtime/share/ptoas"
  COMPONENT PTOAS_Python
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

install(
  DIRECTORY "${CMAKE_SOURCE_DIR}/lib/SoftOps"
  DESTINATION "ptoas/_runtime/share/ptoas"
  COMPONENT PTOAS_Python
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

# ---------------------------------------------------------------------------
# Online-compilation fallback: ship the version-sensitive `_core` as source +
# a self-contained header closure + a standalone CMakeLists, so a matching
# extension can be compiled at first import against the shipped, Python-
# independent libPTOASCompiler DSO. The prebuilt `_core` above stays the fast
# path; this is only the fallback for non-matching interpreters. The online
# `_core` is full-featured (main / TileLib / SoftLib), NOT dialect-only: it
# forward-declares runPTOAS and links the shipped DSO, sharing the same real
# LLVM/MLIR C++ header closure (mlir_include/) as the pybind11 family below.
# ---------------------------------------------------------------------------
option(PTOAS_ENABLE_ONLINE_CORE_COMPILE
  "Ship ptoas._core sources + header closure for runtime online compilation" OFF)

if(PTOAS_ENABLE_ONLINE_CORE_COMPILE)
  set(_ptoas_online_dst "ptoas/_online")

  # 1. The PTO dialect binding TU, the full-featured module entry (NativeModule.cpp,
  #    compiled with PTOAS_ONLINE_BUILD so it forward-declares runPTOAS instead of
  #    pulling the heavy ptoas.h driver closure), and the shared PTOModule.h header.
  #    All .cpp files `#include "PTOModule.h"`, so it is staged next to them
  #    (quote-include resolves within src/).
  install(FILES
      "${CMAKE_SOURCE_DIR}/lib/Bindings/Python/PTOModule.cpp"
      "${CMAKE_SOURCE_DIR}/lib/Bindings/Python/PTOModule.h"
      "${CMAKE_CURRENT_SOURCE_DIR}/NativeModule.cpp"
    DESTINATION "${_ptoas_online_dst}/src"
    COMPONENT PTOAS_Python)

  # 2. Standalone runtime build recipe (staged as ptoas/_online/CMakeLists.txt).
  install(FILES
      "${CMAKE_CURRENT_SOURCE_DIR}/online_template/CMakeLists.txt"
    DESTINATION "${_ptoas_online_dst}"
    COMPONENT PTOAS_Python)

  # 3. Repo-owned project headers included by the _core TUs (PTOModule.cpp pulls
  #    pto-c/Dialect/*; NativeModule.cpp pulls PTO/Transforms/*Service.h and
  #    PTO/Support/CodeConstants.h). Staged under repo_include/ preserving their
  #    include subpath. _core reuses the real LLVM/MLIR C++ closure staged under
  #    mlir_include/ (item 9), the SAME tree the pybind11 family uses -- there is
  #    no separate patched PybindAdaptors.h / minimal mlir-c list for _core.
  install(FILES
      "${CMAKE_SOURCE_DIR}/include/pto-c/Dialect/PTO.h"
      "${CMAKE_SOURCE_DIR}/include/pto-c/Dialect/PTOEnums.h"
    DESTINATION "${_ptoas_online_dst}/repo_include/pto-c/Dialect"
    COMPONENT PTOAS_Python)
  install(FILES
      "${CMAKE_SOURCE_DIR}/include/PTO/Transforms/TileLibService.h"
      "${CMAKE_SOURCE_DIR}/include/PTO/Transforms/SoftLibService.h"
    DESTINATION "${_ptoas_online_dst}/repo_include/PTO/Transforms"
    COMPONENT PTOAS_Python)
  install(FILES
      "${CMAKE_SOURCE_DIR}/include/PTO/Support/CodeConstants.h"
    DESTINATION "${_ptoas_online_dst}/repo_include/PTO/Support"
    COMPONENT PTOAS_Python)

  # ---------------------------------------------------------------------------
  # Family extensions: _mlir (core), _mlirDialectsLLVM, _site_initialize_0.
  # These are the genuine upstream pybind11 wrappers; like the full-featured
  # `_core` they need the real PybindAdaptors.h and the full LLVM/MLIR C++ header
  # closure. Their private sibling headers are staged under `src/mlir`; the
  # shared real C++ header closure is staged under `mlir_include` (item 9) and is
  # consumed by BOTH the family and `_core`.
  # ---------------------------------------------------------------------------

  # 6. Locate the MLIR C++ source tree that ships the python-binding TUs. Derive
  #    it from the include dir that holds the C++ PybindAdaptors.h (the source
  #    include root), then strip `/include` -> `<mlir_src>` -> lib/Bindings/Python.
  set(_ptoas_mlir_src_root "")
  foreach(_inc IN LISTS MLIR_INCLUDE_DIRS)
    if(EXISTS "${_inc}/mlir/Bindings/Python/PybindAdaptors.h")
      get_filename_component(_ptoas_mlir_src_root "${_inc}" DIRECTORY)
      break()
    endif()
  endforeach()
  if(NOT _ptoas_mlir_src_root)
    message(FATAL_ERROR
      "PTOAS_ENABLE_ONLINE_CORE_COMPILE: cannot locate the MLIR source include "
      "holding mlir/Bindings/Python/PybindAdaptors.h in "
      "MLIR_INCLUDE_DIRS='${MLIR_INCLUDE_DIRS}'")
  endif()
  set(_ptoas_mlir_py_src "${_ptoas_mlir_src_root}/lib/Bindings/Python")

  # 7. Stage the _mlir core TUs (9 .cpp) + their private sibling headers (5 .h,
  #    quote-included so they must sit next to the .cpp) + the DialectLLVM TU.
  set(_ptoas_mlir_family_srcs
      MainModule.cpp
      IRAffine.cpp
      IRAttributes.cpp
      IRCore.cpp
      IRInterfaces.cpp
      IRModule.cpp
      IRTypes.cpp
      Pass.cpp
      Rewrite.cpp
      Globals.h
      IRModule.h
      Pass.h
      PybindUtils.h
      Rewrite.h
      DialectLLVM.cpp)
  set(_ptoas_mlir_family_paths "")
  foreach(_f IN LISTS _ptoas_mlir_family_srcs)
    if(NOT EXISTS "${_ptoas_mlir_py_src}/${_f}")
      message(FATAL_ERROR
        "PTOAS_ENABLE_ONLINE_CORE_COMPILE: missing MLIR python source "
        "'${_ptoas_mlir_py_src}/${_f}'")
    endif()
    list(APPEND _ptoas_mlir_family_paths "${_ptoas_mlir_py_src}/${_f}")
  endforeach()
  install(FILES ${_ptoas_mlir_family_paths}
    DESTINATION "${_ptoas_online_dst}/src/mlir"
    COMPONENT PTOAS_Python)

  # 8. Our site-initializer TU (registers the selected upstream dialects). Lives
  #    in the PTOAS repo, staged alongside the upstream family sources.
  install(FILES
      "${CMAKE_SOURCE_DIR}/lib/Bindings/Python/PythonRegistration.cpp"
    DESTINATION "${_ptoas_online_dst}/src/mlir"
    COMPONENT PTOAS_Python)

  # 9. Full LLVM/MLIR C++ header closure, driven by the frozen manifest under
  #    online_manifest/. Regenerate the manifest on an LLVM upgrade with
  #    online_manifest/gen_mlir_family_headers.sh. Blank/`#` lines are skipped.
  set(_ptoas_family_manifest
      "${CMAKE_CURRENT_SOURCE_DIR}/online_manifest/mlir_family_headers.manifest")
  if(NOT EXISTS "${_ptoas_family_manifest}")
    message(FATAL_ERROR
      "PTOAS_ENABLE_ONLINE_CORE_COMPILE: missing header manifest "
      "'${_ptoas_family_manifest}'")
  endif()
  file(STRINGS "${_ptoas_family_manifest}" _ptoas_family_headers)
  foreach(_hdr IN LISTS _ptoas_family_headers)
    string(STRIP "${_hdr}" _hdr)
    if(_hdr STREQUAL "" OR _hdr MATCHES "^#")
      continue()
    endif()
    set(_ptoas_found "")
    foreach(_inc IN LISTS MLIR_INCLUDE_DIRS LLVM_INCLUDE_DIRS)
      if(EXISTS "${_inc}/${_hdr}")
        set(_ptoas_found "${_inc}/${_hdr}")
        break()
      endif()
    endforeach()
    if(NOT _ptoas_found)
      message(FATAL_ERROR
        "PTOAS_ENABLE_ONLINE_CORE_COMPILE: manifest header '${_hdr}' not found "
        "in MLIR_INCLUDE_DIRS/LLVM_INCLUDE_DIRS")
    endif()
    get_filename_component(_ptoas_hdr_subdir "${_hdr}" DIRECTORY)
    install(FILES "${_ptoas_found}"
      DESTINATION "${_ptoas_online_dst}/mlir_include/${_ptoas_hdr_subdir}"
      COMPONENT PTOAS_Python)
  endforeach()

  # 10. Ship the shared LLVMSupport (+ its Demangle dependency) that the family
  #     extensions dynamically link. The prebuilt libPTOASCompiler DSO exports
  #     only C-API symbols, so these C++ symbols must travel as their own DSOs
  #     next to it in ptoas/mlir/_mlir_libs. Requires an LLVM built with shared
  #     libraries (BUILD_SHARED_LIBS / LLVM_BUILD_LLVM_DYLIB).
  set(_ptoas_llvm_shared_libs LLVMSupport LLVMDemangle)
  foreach(_lib IN LISTS _ptoas_llvm_shared_libs)
    set(_ptoas_lib_link
        "${LLVM_LIBRARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}${_lib}${CMAKE_SHARED_LIBRARY_SUFFIX}")
    if(NOT EXISTS "${_ptoas_lib_link}")
      message(FATAL_ERROR
        "PTOAS_ENABLE_ONLINE_CORE_COMPILE: shared '${_lib}' not found at "
        "'${_ptoas_lib_link}'. Build LLVM with shared libraries so the online "
        "mlir family can link it.")
    endif()
    # `${_ptoas_lib_link}` is the unversioned dev symlink (e.g.
    # libLLVMSupport.so -> libLLVMSupport.so.19.1). `install(FILES)` on it
    # dereferences the symlink but keeps the *unversioned* name, so the shipped
    # file is `libLLVMSupport.so` while its embedded SONAME stays
    # `libLLVMSupport.so.19.1`. A dependent linked by full path records
    # DT_NEEDED = SONAME, so at load time the dynamic loader searches its rpath
    # for `libLLVMSupport.so.19.1` -- a name that does not exist on disk -- and
    # the load only succeeds because the Python loader happens to have preloaded
    # the file with RTLD_GLOBAL first (a mere warning on failure). Ship the file
    # under its real SONAME'd name instead so rpath resolution works on its own,
    # and recreate the unversioned symlink for name-based `-lLLVMSupport`.
    get_filename_component(_ptoas_lib_real "${_ptoas_lib_link}" REALPATH)
    get_filename_component(_ptoas_lib_soname "${_ptoas_lib_real}" NAME)
    get_filename_component(_ptoas_lib_linkname "${_ptoas_lib_link}" NAME)
    install(FILES "${_ptoas_lib_real}"
      DESTINATION "ptoas/mlir/_mlir_libs"
      COMPONENT PTOAS_Python)
    if(NOT _ptoas_lib_soname STREQUAL _ptoas_lib_linkname)
      install(CODE
        "execute_process(COMMAND \"\${CMAKE_COMMAND}\" -E create_symlink \
          \"${_ptoas_lib_soname}\" \
          \"\${CMAKE_INSTALL_PREFIX}/ptoas/mlir/_mlir_libs/${_ptoas_lib_linkname}\")"
        COMPONENT PTOAS_Python)
    endif()
  endforeach()
endif()

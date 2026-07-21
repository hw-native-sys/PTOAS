# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set(PTOAS_RUNTIME_SOURCES
  ptoas.cpp
  driver.cpp
  VPTOHostStubEmission.cpp
  ObjectEmission.cpp
  TilelangDaemon.cpp
)

set(PTOAS_WRAPPER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/ptoas_wrapper.py")
set(PTOAS_WRAPPER_BINARY "${CMAKE_CURRENT_BINARY_DIR}/ptoas")

configure_file(
  "${PTOAS_WRAPPER_SOURCE}"
  "${PTOAS_WRAPPER_BINARY}"
  @ONLY
)
file(CHMOD "${PTOAS_WRAPPER_BINARY}"
  PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
)

function(ptoas_configure_runtime_target target_name)
  target_compile_definitions(${target_name} PRIVATE
    PTOAS_RELEASE_VERSION="${PTOAS_CLI_VERSION}"
    # Source-tree defaults for TileLib expansion. These let ptoas run directly
    # from the build tree without passing --tilelang-path /
    # --tilelang-pkg-path / --ptodsl-pkg-path. Installed layouts are expected
    # to launch through the wrapper/launcher flow, which injects explicit
    # runtime paths instead of relying on executable-path probing.
    PTOAS_DEFAULT_TILELANG_PATH="${CMAKE_SOURCE_DIR}/lib/TileOps"
    PTOAS_DEFAULT_TILELANG_PKG_PATH="${CMAKE_SOURCE_DIR}/tilelang-dsl/python"
    PTOAS_DEFAULT_PTODSL_PKG_PATH="${CMAKE_SOURCE_DIR}/ptodsl"
    ${ARGN}
  )
  target_link_libraries(${target_name} PRIVATE
    # === 你的本地库 (请确认 lib/PTO/*/CMakeLists.txt 里的定义名) ===
    PTOIR            # 替代 MLIRPTODialect
    PTOTransforms    # 替代 MLIRPTOTransforms
    ptobc_lib

    # === MLIR 基础设施 ===
    MLIRMlirOptMain  # [重要] 包含了 main 函数入口逻辑，通常必须链接

    # === MLIR 标准库 ===
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
  add_dependencies(${target_name}
    PTOOpsIncGen
  )
endfunction()

add_library(ptoas_runtime SHARED ${PTOAS_RUNTIME_SOURCES})
# Keep the shared runtime's compile mode aligned with LLVM/MLIR so RTTI,
# exceptions, and visibility expectations match the libraries it links to.
llvm_update_compile_flags(ptoas_runtime)
set_target_properties(ptoas_runtime PROPERTIES
  OUTPUT_NAME "ptoas"
  PREFIX ""
)
ptoas_configure_runtime_target(ptoas_runtime
  PTOAS_BUILD_SHARED_LIBRARY=1
)

set(PTOAS_BUILD_RUNTIME_STAGING_DIR "${CMAKE_BINARY_DIR}/runtime-staging")
add_custom_target(ptoas_runtime_staging ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory
          "${PTOAS_BUILD_RUNTIME_STAGING_DIR}/lib"
          "${PTOAS_BUILD_RUNTIME_STAGING_DIR}/share/ptoas"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "$<TARGET_FILE:ptoas_runtime>"
          "${PTOAS_BUILD_RUNTIME_STAGING_DIR}/lib/ptoas.so"
  COMMAND ${CMAKE_COMMAND} -E remove_directory
          "${PTOAS_BUILD_RUNTIME_STAGING_DIR}/share/ptoas/TileOps"
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${CMAKE_SOURCE_DIR}/lib/TileOps"
          "${PTOAS_BUILD_RUNTIME_STAGING_DIR}/share/ptoas/TileOps"
  DEPENDS ptoas_runtime
  COMMENT "Staging ptoas runtime into build/runtime-staging"
  VERBATIM
)

if(TARGET PTODSLPackage)
  # The build-tree launcher imports ptoas._runtime_entry from build/python.
  # Keep that package staging coupled to every runtime staging entrypoint,
  # including partial builds such as tools/ptoas/all.
  add_dependencies(ptoas_runtime_staging PTODSLPackage)
endif()

if(TARGET ptoas_runtime_deps)
  add_dependencies(ptoas_runtime_deps ptoas_runtime_staging)
endif()

install(PROGRAMS "${PTOAS_WRAPPER_BINARY}"
  DESTINATION bin
  COMPONENT PTOAS_Runtime
)

install(TARGETS ptoas_runtime
  LIBRARY DESTINATION lib
  COMPONENT PTOAS_Runtime
)

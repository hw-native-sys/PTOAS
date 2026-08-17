# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# =========================================================
# PTODSL package wiring
# =========================================================

set(PTODSL_PACKAGE_SRC_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/ptodsl")
set(PTODSL_BUILD_PACKAGE_DIR
    "${CMAKE_BINARY_DIR}/python/ptodsl")

declare_mlir_python_sources(PTODSLPythonSources
  ROOT_DIR "${PTODSL_PACKAGE_SRC_DIR}"
  SOURCES_GLOB "*.py"
)
add_mlir_python_sources_target(PTODSLPackageStage
  OUTPUT_DIRECTORY "${PTODSL_BUILD_PACKAGE_DIR}"
  SOURCES_TARGETS PTODSLPythonSources
)
add_custom_target(PTODSLPackage ALL)
add_dependencies(PTODSLPackage PTODSLPackageStage)

install(
  DIRECTORY "${PTODSL_PACKAGE_SRC_DIR}"
  DESTINATION "."
  COMPONENT PTOAS_CompilerArchive
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ptoas"
  DESTINATION "."
  COMPONENT PTOAS_CompilerArchive
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

install(
  DIRECTORY "${TILEOPS_PACKAGE_SRC_DIR}"
  DESTINATION "share/ptoas"
  COMPONENT PTOAS_Runtime
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

install(
  DIRECTORY "${CMAKE_SOURCE_DIR}/lib/SoftOps"
  DESTINATION "share/ptoas"
  COMPONENT PTOAS_Runtime
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

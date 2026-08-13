# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# ============================================================
# 1. Generate dialect-wide declarations and family operation declarations.
# ============================================================
set(LLVM_TARGET_DEFINITIONS PTOOpCommon.td)

# 生成 Dialect 定义
mlir_tablegen(PTODialect.h.inc -gen-dialect-decls -dialect=pto)
mlir_tablegen(PTODialect.cpp.inc -gen-dialect-defs -dialect=pto)

# Generate public operation declarations per semantic family. Keep the
# aggregate definition output until registration and implementations move in a
# later refactoring step.
set(LLVM_TARGET_DEFINITIONS PTOBaseOps.td)
mlir_tablegen(PTOBaseOps.h.inc -gen-op-decls)

set(LLVM_TARGET_DEFINITIONS PTOTileOps.td)
mlir_tablegen(PTOTileOps.h.inc -gen-op-decls)

foreach(tile_family
    PTOTileMemoryOps PTOTileCubeOps PTOTilePipelineOps
    PTOTileVectorDataOps PTOTileVectorElementwiseOps PTOTileVectorReductionOps)
  set(LLVM_TARGET_DEFINITIONS ${tile_family}.td)
  mlir_tablegen(${tile_family}.h.inc -gen-op-decls)
endforeach()

set(LLVM_TARGET_DEFINITIONS VMIOps.td)
mlir_tablegen(VMIOps.h.inc -gen-op-decls)

set(LLVM_TARGET_DEFINITIONS VPTOOps.td)
mlir_tablegen(VPTOOps.h.inc -gen-op-decls)

foreach(vpto_family VPTOControlOps VPTOSimtOps VPTOMteOps VPTOCubeOps VPTOVectorOps)
  set(LLVM_TARGET_DEFINITIONS ${vpto_family}.td)
  mlir_tablegen(${vpto_family}.h.inc -gen-op-decls)
endforeach()

set(LLVM_TARGET_DEFINITIONS VPTOUbOps.td)
mlir_tablegen(VPTOUbOps.h.inc -gen-op-decls)

set(LLVM_TARGET_DEFINITIONS PTOOps.td)
mlir_tablegen(PTOOps.h.inc -gen-op-decls)
mlir_tablegen(PTOOps.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS PTOOpCommon.td)

# 生成 Types 定义
mlir_tablegen(PTOTypeDefs.h.inc -gen-typedef-decls -typedefs-dialect=pto)
mlir_tablegen(PTOTypeDefs.cpp.inc -gen-typedef-defs -typedefs-dialect=pto)

# 生成 Attributes 定义
mlir_tablegen(PTOAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=pto)
mlir_tablegen(PTOAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=pto)

# [修复] 生成 Enums 定义 (不再注释)
mlir_tablegen(PTOEnums.h.inc -gen-enum-decls)
mlir_tablegen(PTOEnums.cpp.inc -gen-enum-defs)

# ============================================================
# 2. 处理 Interfaces (防止下个报错)
# ============================================================
# 如果你的 Interfaces 定义在单独的 PTOInterfaces.td 文件里：
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/PTOInterfaces.td)
  set(LLVM_TARGET_DEFINITIONS PTOInterfaces.td)
  mlir_tablegen(PTOInterfaces.h.inc -gen-op-interface-decls)
  mlir_tablegen(PTOInterfaces.cpp.inc -gen-op-interface-defs)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/VPTOInterfaces.td)
  set(LLVM_TARGET_DEFINITIONS VPTOInterfaces.td)
  mlir_tablegen(VPTOInterfaces.h.inc -gen-op-interface-decls)
  mlir_tablegen(VPTOInterfaces.cpp.inc -gen-op-interface-defs)
endif()

# ============================================================
# 3. 注册聚合目标
# ============================================================
add_public_tablegen_target(PTOOpsIncGen)

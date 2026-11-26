# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set(LLVM_TARGET_DEFINITIONS LinxOps.td)
mlir_tablegen(LinxDialect.h.inc -gen-dialect-decls -dialect=pto)
mlir_tablegen(LinxDialect.cpp.inc -gen-dialect-defs -dialect=pto)
mlir_tablegen(LinxOps.h.inc -gen-op-decls)
mlir_tablegen(LinxOps.cpp.inc -gen-op-defs)
mlir_tablegen(LinxTypes.h.inc -gen-typedef-decls -typedefs-dialect=pto)
mlir_tablegen(LinxTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=pto)
mlir_tablegen(LinxEnums.h.inc -gen-enum-decls)
mlir_tablegen(LinxEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(LinxPTOOpsIncGen)

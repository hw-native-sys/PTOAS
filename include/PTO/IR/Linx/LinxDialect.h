// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Independent Linx implementation of the textual pto dialect.
#ifndef PTO_IR_LINX_DIALECT_H
#define PTO_IR_LINX_DIALECT_H
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "PTO/IR/Linx/LinxDialect.h.inc"
namespace mlir {
class ModuleOp;
}
namespace mlir::pto::linx {
LogicalResult verifyModule(ModuleOp module);
} // namespace mlir::pto::linx
#include "PTO/IR/Linx/LinxEnums.h.inc"
#define GET_TYPEDEF_CLASSES
#include "PTO/IR/Linx/LinxTypes.h.inc"
#define GET_OP_CLASSES
#include "PTO/IR/Linx/LinxOps.h.inc"
#endif

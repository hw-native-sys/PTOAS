// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Transport layout completion after a costed plan has been committed.
#ifndef PTO_TRANSFORMS_VMI_VMILAYOUTSTRUCTURALASSIGNMENT_H
#define PTO_TRANSFORMS_VMI_VMILAYOUTSTRUCTURALASSIGNMENT_H
#include "mlir/IR/BuiltinOps.h"
namespace mlir::pto {
class VMILayoutPropagator;
LogicalResult completeVMILayoutTransportAssignments(ModuleOp module, VMILayoutPropagator& propagator);
} // namespace mlir::pto
#endif // PTO_TRANSFORMS_VMI_VMILAYOUTSTRUCTURALASSIGNMENT_H

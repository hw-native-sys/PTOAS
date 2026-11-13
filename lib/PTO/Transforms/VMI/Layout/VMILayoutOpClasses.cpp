// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//
//===- VMILayoutOpClasses.cpp - VMI layout op classification --------------===//
//
// Operation-class predicates the costed layout solver needs.  They live in
// their own unit because the cost model calls isVMILayoutCastOp and therefore
// drags the definition into any full link: keeping it next to the propagator's
// isVMISameLayoutOp keeps the two classifications side by side, and the planner
// that consumes them must not define them again.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/VMILayoutPlanner.h"

using namespace mlir;

bool mlir::pto::isVMILayoutCastOp(Operation* op)
{
    return isa<VMIExtFOp, VMIExtSIOp, VMIExtUIOp, VMITruncFOp, VMITruncIOp, VMIFPToSIOp, VMIFPToUIOp, VMISIToFPOp>(op);
}

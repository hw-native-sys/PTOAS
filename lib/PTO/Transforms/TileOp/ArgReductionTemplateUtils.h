// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#pragma once

#include "PTO/IR/PTO.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace pto {
namespace detail {

template <typename OpTy>
SmallVector<Value, 4> getArgReductionTemplateOperands(OpTy op)
{
    Value src = op.getSrc();
    Value tmp = op.getTmp();
    Value dstVal = op.getDstVal();
    // Keep the tmp slot stable for candidate matching when the IR omits tmp.
    SmallVector<Value, 4> operands{src, tmp ? tmp : src, op.getDstIdx()};
    if (dstVal) {
        operands.push_back(dstVal);
    }
    return operands;
}

inline SmallVector<Value, 4> getArgReductionTemplateOperands(Operation *op)
{
    return llvm::TypeSwitch<Operation *, SmallVector<Value, 4>>(op)
        .Case<TRowArgMaxOp>([](auto typedOp) {
            return getArgReductionTemplateOperands(typedOp);
        })
        .Case<TRowArgMinOp>([](auto typedOp) {
            return getArgReductionTemplateOperands(typedOp);
        })
        .Case<TColArgMaxOp>([](auto typedOp) {
            return getArgReductionTemplateOperands(typedOp);
        })
        .Case<TColArgMinOp>([](auto typedOp) {
            return getArgReductionTemplateOperands(typedOp);
        })
        .Default([](Operation *genericOp) {
            return SmallVector<Value, 4>(genericOp->getOperands().begin(),
                                         genericOp->getOperands().end());
        });
}

} // namespace detail
} // namespace pto
} // namespace mlir

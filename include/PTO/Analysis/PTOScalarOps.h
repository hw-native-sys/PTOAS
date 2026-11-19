// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOScalarOps.h - Scalar arithmetic spelling compatibility ---------===//
//===----------------------------------------------------------------------===//
//
// Frontend PTO IR spells scalar integer arithmetic with the PTO dialect
// (`pto.addi`, `pto.muli`, `pto.index_cast`, ...) and the lowering restates it
// with `arith`. Analyses and combines that reason about scalar expressions have
// to accept both spellings, so the predicates below are the single place that
// knows about the two, instead of every caller testing `arith` ops only.
//
//===----------------------------------------------------------------------===//

#ifndef PTO_ANALYSIS_PTOSCALAROPS_H
#define PTO_ANALYSIS_PTOSCALAROPS_H

#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir {
namespace pto {

inline bool isScalarAddOp(Operation* op) { return isa_and_nonnull<AddIOp, arith::AddIOp>(op); }

inline bool isScalarSubOp(Operation* op) { return isa_and_nonnull<SubIOp, arith::SubIOp>(op); }

inline bool isScalarMulOp(Operation* op) { return isa_and_nonnull<MulIOp, arith::MulIOp>(op); }

inline bool isScalarIndexCastOp(Operation* op)
{
    return isa_and_nonnull<IndexCastOp, arith::IndexCastOp, arith::IndexCastUIOp>(op);
}

inline bool isUnsignedScalarIndexCastOp(Operation* op)
{
    if (isa_and_nonnull<arith::IndexCastUIOp>(op)) {
        return true;
    }
    auto cast = dyn_cast_or_null<IndexCastOp>(op);
    return cast && cast.getSignedness().getValue() == Signedness::Unsigned;
}

inline bool isScalarTruncOp(Operation* op) { return isa_and_nonnull<TruncIOp, arith::TruncIOp>(op); }

inline bool isScalarExtOp(Operation* op) { return isa_and_nonnull<ExtIOp, arith::ExtSIOp, arith::ExtUIOp>(op); }

inline bool isUnsignedScalarExtOp(Operation* op)
{
    if (isa_and_nonnull<arith::ExtUIOp>(op)) {
        return true;
    }
    auto ext = dyn_cast_or_null<ExtIOp>(op);
    return ext && ext.getSignedness().getValue() == Signedness::Unsigned;
}

/// Any integer cast the value-evolution analysis can see through.
inline bool isScalarCastOp(Operation* op)
{
    return isScalarIndexCastOp(op) || isScalarTruncOp(op) || isScalarExtOp(op);
}

inline Value getScalarLhs(Operation* op) { return op->getOperand(0); }

inline Value getScalarRhs(Operation* op) { return op->getOperand(1); }

} // namespace pto
} // namespace mlir

#endif // PTO_ANALYSIS_PTOSCALAROPS_H

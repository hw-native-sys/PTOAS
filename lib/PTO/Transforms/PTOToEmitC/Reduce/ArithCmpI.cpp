// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- ArithCmpI.cpp - Reduce ArithCmpI op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "ReduceInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

class ArithCmpIToEmitC : public OpConversionPattern<arith::CmpIOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  // Map arith.cmpi's predicate onto the EmitC spelling; unsigned variants
  // reuse the signed EmitC predicates after operand re-interpretation.
  static LogicalResult resolveCmpIPredicate(arith::CmpIOp op,
                                            emitc::CmpPredicate &pred,
                                            bool &isUnsignedPred) {
    isUnsignedPred = op.getPredicate() == arith::CmpIPredicate::ult ||
                     op.getPredicate() == arith::CmpIPredicate::ule ||
                     op.getPredicate() == arith::CmpIPredicate::ugt ||
                     op.getPredicate() == arith::CmpIPredicate::uge;
    switch (op.getPredicate()) {
      case arith::CmpIPredicate::eq:  pred = emitc::CmpPredicate::eq; break;
      case arith::CmpIPredicate::ne:  pred = emitc::CmpPredicate::ne; break;
      case arith::CmpIPredicate::slt: pred = emitc::CmpPredicate::lt; break;
      case arith::CmpIPredicate::sle: pred = emitc::CmpPredicate::le; break;
      case arith::CmpIPredicate::sgt: pred = emitc::CmpPredicate::gt; break;
      case arith::CmpIPredicate::sge: pred = emitc::CmpPredicate::ge; break;
      case arith::CmpIPredicate::ult: pred = emitc::CmpPredicate::lt; break;
      case arith::CmpIPredicate::ule: pred = emitc::CmpPredicate::le; break;
      case arith::CmpIPredicate::ugt: pred = emitc::CmpPredicate::gt; break;
      case arith::CmpIPredicate::uge: pred = emitc::CmpPredicate::ge; break;
    }
    return success();
  }

  // Reinterpret unsigned-comparison operands in the unsigned C++ type of the
  // same width (i1 keeps its operands as-is).
  static FailureOr<std::pair<Value, Value>>
  adaptCmpIOperands(arith::CmpIOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter, Location loc,
                    bool isUnsignedPred) {
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    if (!isUnsignedPred)
      return std::make_pair(lhs, rhs);

    Type opTy = op.getLhs().getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(
          op, "expected scalar integer or index operands");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);
    if (bitWidth != 1) {
      lhs = castSignlessIntToUnsignedSameWidth(rewriter, loc, lhs, bitWidth);
      rhs = castSignlessIntToUnsignedSameWidth(rewriter, loc, rhs, bitWidth);
    }
    return std::make_pair(lhs, rhs);
  }

  LogicalResult matchAndRewrite(arith::CmpIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    emitc::CmpPredicate emitcPred = emitc::CmpPredicate::eq;
    bool isUnsignedPred = false;
    if (failed(resolveCmpIPredicate(op, emitcPred, isUnsignedPred)))
      return failure();

    Type resTy = getTypeConverter()->convertType(op.getType());
    if (!resTy)
      return failure();

    auto operands = adaptCmpIOperands(op, adaptor, rewriter, loc,
                                      isUnsignedPred);
    if (failed(operands))
      return failure();
    Value lhs = operands->first;
    Value rhs = operands->second;

    rewriter.replaceOpWithNewOp<emitc::CmpOp>(
        op, 
        /*resultType=*/resTy, // i1 -> bool/i1
        emitcPred,
        lhs,
        rhs
    );
    return success();
  }
};

void populateReduceArithCmpIPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<ArithCmpIToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir

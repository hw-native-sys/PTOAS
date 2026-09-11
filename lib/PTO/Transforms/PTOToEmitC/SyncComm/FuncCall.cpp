// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- FuncCall.cpp - SyncComm FuncCall op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "SyncCommInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct ReturnToEmitC : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (auto emitcFunc = op->getParentOfType<emitc::FuncOp>()) {
      if (auto modeAttr =
              emitcFunc->getAttrOfType<StringAttr>(kAutoSyncTailPendingModeAttr)) {
        auto *ctx = rewriter.getContext();
        rewriter.setInsertionPoint(op);
        auto args = rewriter.getArrayAttr(
            {emitc::OpaqueAttr::get(ctx, modeAttr.getValue())});
        rewriter.create<emitc::CallOpaqueOp>(
            op.getLoc(), TypeRange{}, "ptoas_auto_sync_tail",
            args, ArrayAttr{}, ValueRange{});
      }
    }

    auto vals = adaptor.getOperands();
    if (vals.empty()) {
      rewriter.replaceOpWithNewOp<emitc::ReturnOp>(op, Value{});
      return success();
    }
    if (vals.size() == 1) {
      rewriter.replaceOpWithNewOp<emitc::ReturnOp>(op, vals[0]);
      return success();
    }
    return rewriter.notifyMatchFailure(op, "EmitC cannot return multiple values");
  }
};

struct CallToEmitC : public OpConversionPattern<func::CallOp> {
  using OpConversionPattern<func::CallOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() > 1)
      return rewriter.notifyMatchFailure(
          op, "EmitC cannot lower calls with multiple results");

    SmallVector<Type> resultTypes;
    if (failed(
            getTypeConverter()->convertTypes(op.getResultTypes(), resultTypes)))
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert call result types");

    SmallVector<Value> operands;
    operands.reserve(adaptor.getOperands().size());
    auto calleeType = op.getCalleeType();
    unsigned originalArgCount = calleeType.getNumInputs();
    if (originalArgCount != adaptor.getOperands().size())
      return rewriter.notifyMatchFailure(
          op, "call operand count mismatch after type conversion");

    for (auto [index, loweredOperand] : llvm::enumerate(adaptor.getOperands())) {
      FailureOr<Value> adapted = adaptCallOperandForEmitC(
          getTypeConverter(), rewriter, op.getLoc(), calleeType.getInput(index),
          op.getOperand(index),
          loweredOperand);
      if (failed(adapted))
        return rewriter.notifyMatchFailure(op,
                                           "failed to adapt call operand for EmitC ABI");
      operands.push_back(*adapted);
    }

    rewriter.replaceOpWithNewOp<emitc::CallOp>(op, op.getCalleeAttr(),
                                               resultTypes, operands);
    return success();
  }
};

void populateSyncCommFuncCallPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<CallToEmitC, ReturnToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir

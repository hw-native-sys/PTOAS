// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- Struct.cpp - ScalarMisc Struct op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "ScalarMiscInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct PTODeclareStructToEmitC
    : public OpConversionPattern<mlir::pto::DeclareStructOp> {
  using OpConversionPattern<mlir::pto::DeclareStructOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::DeclareStructOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    const bool hasUnexpectedResultCount = op->getNumResults() != 1;
    if (hasUnexpectedResultCount) {
      return rewriter.notifyMatchFailure(op, "expected one !pto.struct result");
    }
    Type structTy =
        getTypeConverter()->convertType(op->getResult(0).getType());
    if (!structTy) {
      return rewriter.notifyMatchFailure(op, "failed to map !pto.struct type");
    }

    // The struct converts to a pointer, so declare the storage as a local
    // variable and hand out its address. buildStructMemberChain recognises the
    // address-of and walks that variable directly, so a struct that never
    // leaves the function still prints as `s.f0` rather than `p->f0`.
    auto ptrTy = dyn_cast<emitc::PointerType>(structTy);
    if (!ptrTy)
      return rewriter.notifyMatchFailure(op,
                                         "!pto.struct did not map to a pointer");

    Value storage = rewriter
                        .create<emitc::VariableOp>(
                            op.getLoc(), ptrTy.getPointee(),
                            emitc::OpaqueAttr::get(rewriter.getContext(), ""))
                        .getResult();
    rewriter.replaceOpWithNewOp<emitc::ApplyOp>(op, ptrTy, "&", storage);
    return success();
  }
};

struct PTOStructGetToEmitC
    : public OpConversionPattern<mlir::pto::StructGetOp> {
  using OpConversionPattern<mlir::pto::StructGetOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::StructGetOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultTy = getTypeConverter()->convertType(op.getValue().getType());
    if (!resultTy) {
      return rewriter.notifyMatchFailure(op, "failed to map struct field type");
    }

    FailureOr<Value> member = resolveStructMember(
        op.getOperation(), adaptor.getOperands(),
        op->getOperand(0).getType(), op.getPath(), rewriter,
        getTypeConverter());
    if (failed(member)) {
      return rewriter.notifyMatchFailure(op, "failed to map struct field type");
    }

    // Materialize the read into its own C++ variable so the SSA result keeps
    // its value even if a later pto.struct_set writes the same field.
    auto snapshot =
        rewriter
            .create<emitc::VariableOp>(
                op.getLoc(), resultTy,
                emitc::OpaqueAttr::get(rewriter.getContext(), ""))
            .getResult();
    rewriter.create<emitc::AssignOp>(op.getLoc(), snapshot, *member);
    rewriter.replaceOp(op, snapshot);
    return success();
  }
};

struct PTOStructSetToEmitC
    : public OpConversionPattern<mlir::pto::StructSetOp> {
  using OpConversionPattern<mlir::pto::StructSetOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::StructSetOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> member = resolveStructMember(
        op.getOperation(), adaptor.getOperands(),
        op->getOperand(0).getType(), op.getPath(), rewriter,
        getTypeConverter());
    if (failed(member)) {
      return rewriter.notifyMatchFailure(op, "failed to map struct field type");
    }

    rewriter.create<emitc::AssignOp>(op.getLoc(), *member, adaptor.getValue());
    rewriter.eraseOp(op);
    return success();
  }
};

void populateScalarMiscStructPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTODeclareStructToEmitC>(typeConverter, ctx);
  patterns.add<PTOStructGetToEmitC>(typeConverter, ctx);
  patterns.add<PTOStructSetToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir

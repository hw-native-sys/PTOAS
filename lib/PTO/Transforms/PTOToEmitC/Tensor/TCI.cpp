// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- TCI.cpp - Tensor TCI op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "TensorInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct PTOTCIToEmitC : public OpConversionPattern<pto::TCIOp> {
  using OpConversionPattern<pto::TCIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TCIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value dst = adaptor.getDst();
    Value S = adaptor.getOperands()[0];
    Value tmp = op.getTmp() ? adaptor.getTmp() : Value();

    // The TCI scalar template parameter should follow the original PTO IR
    // scalar type, not the converted EmitC value type.
    std::string scalarTok = "int32_t";
    if (auto it = dyn_cast<IntegerType>(op->getOperand(0).getType())) {
      bool isUnsigned = it.isUnsigned();
      if (it.getWidth() == 16) {
        scalarTok = isUnsigned ? "uint16_t" : "int16_t";
      } else {
        scalarTok = isUnsigned ? "uint32_t" : "int32_t";
      }
    }

    // descending -> "0"/"1"
    std::string descTok = op.getDescending() ? "1" : "0";

    ArrayAttr targs;
    if (auto ot = mlir::dyn_cast<emitc::OpaqueType>(dst.getType())) {
      SmallVector<Attribute, 4> templateArgVec;
      templateArgVec.push_back(
          emitc::OpaqueAttr::get(ctx, ot.getValue().str()));
      if (tmp) {
        auto tmpOt = mlir::dyn_cast<emitc::OpaqueType>(tmp.getType());
        if (!tmpOt)
          return rewriter.notifyMatchFailure(
              op, "expected tmp tile to lower to emitc::OpaqueType");
        templateArgVec.push_back(
            emitc::OpaqueAttr::get(ctx, tmpOt.getValue().str()));
      }
      templateArgVec.push_back(emitc::OpaqueAttr::get(ctx, scalarTok));
      templateArgVec.push_back(emitc::OpaqueAttr::get(ctx, descTok));
      targs = rewriter.getArrayAttr(templateArgVec);
    } else {
      targs = rewriter.getArrayAttr({});
    }

    SmallVector<Value, 3> operands{dst, S};
    if (tmp)
      operands.push_back(tmp);

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCI",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/targs,
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

void populateTensorTCIPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTOTCIToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir

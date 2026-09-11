// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- TPrintTrap.cpp - Reduce TPrintTrap op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "ReduceInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct PTOPrintToTPRINT : public OpConversionPattern<pto::TPrintOp> {
  using OpConversionPattern<pto::TPrintOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPrintOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto printFormatTok = [&](pto::PrintFormat format) -> StringRef {
      switch (format) {
      case pto::PrintFormat::Width8_Precision4:
        return "pto::PrintFormat::Width8_Precision4";
      case pto::PrintFormat::Width8_Precision2:
        return "pto::PrintFormat::Width8_Precision2";
      case pto::PrintFormat::Width10_Precision6:
        return "pto::PrintFormat::Width10_Precision6";
      }
      llvm_unreachable("unknown PrintFormat");
    };

    Value src = adaptor.getSrc();
    if (isa<MemRefType>(op.getSrc().getType()) ||
        isa<mlir::pto::PartitionTensorViewType>(op.getSrc().getType())) {
      src = maybeWrapGlobalMemrefAsGlobalTensor(
          rewriter, loc, src, op.getSrc().getType(), op.getOperation());
    }

    SmallVector<Value, 4> operands{src};
    if (Value tmp = op->getNumOperands() > 1 ? op->getOperand(1) : Value()) {
      Value tmpValue = adaptor.getOperands().size() > 1 ? adaptor.getOperands()[1]
                                                        : Value();
      tmpValue = peelUnrealized(tmpValue);
      if (isa<MemRefType>(tmp.getType()) ||
          isa<mlir::pto::PartitionTensorViewType>(tmp.getType())) {
        tmpValue = maybeWrapGlobalMemrefAsGlobalTensor(
            rewriter, loc, tmpValue, tmp.getType(), op.getOperation());
      }
      operands.push_back(tmpValue);
    }

    SmallVector<Attribute, 1> templateArgVec;
    if (auto formatAttr =
            dyn_cast_or_null<pto::PrintFormatAttr>(
                op.getProperties().printFormat)) {
      templateArgVec.push_back(emitc::OpaqueAttr::get(
          ctx, printFormatTok(formatAttr.getValue())));
    }
    ArrayAttr templateArgs =
        templateArgVec.empty() ? ArrayAttr{} : rewriter.getArrayAttr(templateArgVec);
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TPRINT",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOPrintOpToEmitC : public OpConversionPattern<pto::PrintOp> {
  using OpConversionPattern<pto::PrintOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::PrintOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    std::string fmt = op.getFormat().str();
    if (fmt.empty())
      fmt = "%f";
    std::string quoted = "\"";
    for (char c : fmt) {
      if (c == '"' || c == '\\') {
        quoted += '\\';
      } else if (c == '\n') {
        quoted += "\\n";
      } else if (c == '\t') {
        quoted += "\\t";
      } else {
        quoted += c;
      }
    }
    quoted += "\"";

    Value scalar = adaptor.getScalar();
    auto argsAttr = rewriter.getArrayAttr(
        {emitc::OpaqueAttr::get(ctx, quoted),
         IntegerAttr::get(IndexType::get(ctx), 0)});
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "cce::printf",
        /*args=*/argsAttr,
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{scalar});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOTrapOpToEmitC : public OpConversionPattern<pto::TrapOp> {
  using OpConversionPattern<pto::TrapOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TrapOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "trap",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{});

    rewriter.eraseOp(op);
    return success();
  }
};

void populateReduceTPrintTrapPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTOPrintToTPRINT>(typeConverter, ctx);
  patterns.add<PTOPrintOpToEmitC>(typeConverter, ctx);
  patterns.add<PTOTrapOpToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir

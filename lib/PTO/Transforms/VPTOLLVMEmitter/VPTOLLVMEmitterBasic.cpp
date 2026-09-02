// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// the CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. This software is provided on an "AS IS" BASIS.

#include "VPTOLLVMEmitterInternal.h"

#include "PTO/IR/PTO.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::pto::detail {

class LowerTrapOpPattern final : public OpConversionPattern<pto::TrapOp> {
public:
  explicit LowerTrapOpPattern(TypeConverter &typeConverter,
                              MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::TrapOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::TrapOp op, pto::TrapOp::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    constexpr StringLiteral calleeName = "llvm.hivm.TRAP";
    auto funcType = rewriter.getFunctionType({}, {});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{},
                                   ValueRange{});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

void populateVPTOBasicPatterns(TypeConverter &typeConverter,
                                RewritePatternSet &patterns,
                                LoweringState &state) {
  patterns.add<LowerTrapOpPattern>(typeConverter, patterns.getContext(),
                                   state);
}

} // namespace mlir::pto::detail

// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- Section.cpp - Tile Section op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "TileInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

template <typename SectionOpTy>
struct SectionToEmitC : public OpConversionPattern<SectionOpTy> {
  using OpConversionPattern<SectionOpTy>::OpConversionPattern;

  std::string getMacroName() const {
    if (std::is_same<SectionOpTy, pto::SectionCubeOp>::value)
      return "__DAV_CUBE__";
    if (std::is_same<SectionOpTy, pto::SectionVectorOp>::value)
      return "__DAV_VEC__";
    return "UNKNOWN_MACRO";
  }

  LogicalResult
  matchAndRewrite(SectionOpTy op, typename SectionOpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    bool needsNoSplitGuard = needsA5NoSplitVectorGuard(op.getOperation());

    std::string startMacro = "\n#if defined(" + getMacroName() + ")";
    rewriter.create<emitc::VerbatimOp>(loc, startMacro);

    if constexpr (std::is_same_v<SectionOpTy, pto::SectionVectorOp>) {
      // Vector mask is a global HW state and may be modified by previous kernels
      // (or earlier sections). Reset it to a well-defined state for deterministic
      // execution of VEC ops.
      rewriter.create<emitc::VerbatimOp>(loc, "set_mask_norm();");
      rewriter.create<emitc::VerbatimOp>(loc, "set_vector_mask(-1, -1);");
    }

    if (needsNoSplitGuard) {
      rewriter.create<emitc::VerbatimOp>(
          loc, "if (get_subblockid() == 0) {");
    }

    Block &innerBlock = op.getBody().front();
    if (!innerBlock.empty()) {
      rewriter.inlineBlockBefore(&innerBlock, op.getOperation(), ValueRange{});
    }

    if (needsNoSplitGuard)
      rewriter.create<emitc::VerbatimOp>(loc, "}");

    std::string endMacro = "#endif // " + getMacroName() + "\n";
    rewriter.create<emitc::VerbatimOp>(loc, endMacro);

    rewriter.eraseOp(op);

    return success();
  }
};

void populateTileSectionPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<SectionToEmitC<pto::SectionCubeOp>>(typeConverter, ctx);
  patterns.add<SectionToEmitC<pto::SectionVectorOp>>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir

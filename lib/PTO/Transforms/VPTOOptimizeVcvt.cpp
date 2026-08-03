// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VPTOOPTIMIZEVCVT
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

static bool isOddPart(StringRef part) {
  return part == "ODD" || part == "PART_ODD";
}

static bool isAllTrueMask(Value mask) {
  if (auto op = mask.getDefiningOp<PsetB8Op>())
    return op.getPattern() == "PAT_ALL";
  if (auto op = mask.getDefiningOp<PsetB16Op>())
    return op.getPattern() == "PAT_ALL";
  if (auto op = mask.getDefiningOp<PsetB32Op>())
    return op.getPattern() == "PAT_ALL";
  return false;
}

static bool isPairEquivalentLoadDist(StringRef dist) {
  return dist == "BRC_B8" || dist == "BRC_B16" || dist == "BRC_B32" ||
         dist == "US_B8" || dist == "US_B16" || dist == "E2B_B16" ||
         dist == "E2B_B32";
}

static bool hasEvenOddEquivalentLanes(Value value) {
  if (value.getDefiningOp<VbrOp>())
    return true;

  auto load = value.getDefiningOp<VldsOp>();
  if (!load || value != load.getResult())
    return false;

  std::optional<StringRef> dist = load.getDist();
  return dist && isPairEquivalentLoadDist(*dist);
}

static bool isNarrowToWideVcvt(VcvtOp op) {
  auto inputType = dyn_cast<VRegType>(op.getInput().getType());
  auto resultType = dyn_cast<VRegType>(op.getResult().getType());
  if (!inputType || !resultType)
    return false;

  unsigned inputBits = getPTOStorageElemBitWidth(inputType.getElementType());
  unsigned resultBits = getPTOStorageElemBitWidth(resultType.getElementType());
  return inputBits != 0 && resultBits != 0 && inputBits < resultBits;
}

static Value stripVbitcasts(Value value) {
  while (auto bitcast = value.getDefiningOp<VbitcastOp>())
    value = bitcast.getInput();
  return value;
}

struct AlignedUnsignedWidening {
  unsigned payloadBits;
  unsigned carrierBits;
};

static bool isZeroGapLoad(Value value, AlignedUnsignedWidening widening) {
  auto load = stripVbitcasts(value).getDefiningOp<VldsOp>();
  if (!load)
    return false;

  auto loadType = dyn_cast<VRegType>(load.getResult().getType());
  std::optional<StringRef> dist = load.getDist();
  if (!loadType || !dist ||
      getPTOStorageElemBitWidth(loadType.getElementType()) !=
          widening.payloadBits)
    return false;

  if (widening.payloadBits == 8 && widening.carrierBits == 16)
    return *dist == "UNPK_B8";
  if (widening.payloadBits == 16 && widening.carrierBits == 32)
    return *dist == "UNPK_B16";
  if (widening.payloadBits == 8 && widening.carrierBits == 32)
    return *dist == "UNPK4";
  return false;
}

static bool isZeroGapNarrowingVcvt(Value value,
                                   AlignedUnsignedWidening widening) {
  auto cvt = value.getDefiningOp<VcvtOp>();
  if (!cvt || !isAllTrueMask(cvt.getMask()))
    return false;

  auto inputType = dyn_cast<VRegType>(cvt.getInput().getType());
  auto resultType = dyn_cast<VRegType>(cvt.getResult().getType());
  if (!inputType || !resultType)
    return false;

  unsigned inputBits = getPTOStorageElemBitWidth(inputType.getElementType());
  unsigned resultBits = getPTOStorageElemBitWidth(resultType.getElementType());
  if (inputBits != widening.carrierBits ||
      resultBits != widening.payloadBits ||
      inputBits <= resultBits ||
      inputType.getElementCount() * inputBits !=
          resultType.getElementCount() * resultBits)
    return false;

  std::optional<StringRef> part = cvt.getPart();
  if (!part)
    return false;
  return (inputBits == resultBits * 2 && *part == "EVEN") ||
         (inputBits == resultBits * 4 && *part == "P0");
}

// Deliberately admit only instructions that construct a carrier by zero
// filling its gaps. Ordinary computation is not treated as provenance even
// when a particular operand combination could preserve zero gaps.
static bool isCanonicalZeroGapCarrier(Value value,
                                      AlignedUnsignedWidening widening) {
  value = stripVbitcasts(value);
  if (isZeroGapLoad(value, widening))
    return true;
  if (isZeroGapNarrowingVcvt(value, widening))
    return true;

  auto unpack = value.getDefiningOp<VzunpackOp>();
  if (!unpack)
    return false;
  auto sourceType = dyn_cast<VRegType>(unpack.getSrc().getType());
  auto resultType = dyn_cast<VRegType>(unpack.getResult().getType());
  if (!sourceType || !resultType)
    return false;

  unsigned sourceBits =
      getPTOStorageElemBitWidth(sourceType.getElementType());
  unsigned resultBits =
      getPTOStorageElemBitWidth(resultType.getElementType());
  if (sourceBits == 0 || resultBits != widening.carrierBits ||
      resultBits != sourceBits * 2 || widening.payloadBits > sourceBits)
    return false;

  if (widening.payloadBits == sourceBits)
    return true;
  return isCanonicalZeroGapCarrier(
      unpack.getSrc(), {widening.payloadBits, sourceBits});
}

static std::optional<AlignedUnsignedWidening>
matchAlignedUnsignedWidening(VcvtOp op) {
  auto inputType = dyn_cast<VRegType>(op.getInput().getType());
  auto resultType = dyn_cast<VRegType>(op.getResult().getType());
  if (!inputType || !resultType || op.getRndAttr() || op.getSatAttr() ||
      !isAllTrueMask(op.getMask()))
    return std::nullopt;

  auto inputElementType = dyn_cast<IntegerType>(inputType.getElementType());
  auto resultElementType = dyn_cast<IntegerType>(resultType.getElementType());
  if (!inputElementType || !resultElementType ||
      !inputElementType.isUnsigned() || !resultElementType.isUnsigned())
    return std::nullopt;

  unsigned inputBits = inputElementType.getWidth();
  unsigned resultBits = resultElementType.getWidth();
  if (inputBits >= resultBits ||
      inputType.getElementCount() * inputBits !=
          resultType.getElementCount() * resultBits)
    return std::nullopt;

  std::optional<StringRef> part = op.getPart();
  if (!part)
    return std::nullopt;
  if ((resultBits == inputBits * 2 && *part != "EVEN") ||
      (resultBits == inputBits * 4 && *part != "P0") ||
      (resultBits != inputBits * 2 && resultBits != inputBits * 4))
    return std::nullopt;

  return AlignedUnsignedWidening{inputBits, resultBits};
}

struct CanonicalizeEquivalentPartPattern : public OpRewritePattern<VcvtOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(VcvtOp op,
                                PatternRewriter &rewriter) const override {
    std::optional<StringRef> part = op.getPart();
    if (!part || !isOddPart(*part) || !isNarrowToWideVcvt(op) ||
        !isAllTrueMask(op.getMask()) ||
        !hasEvenOddEquivalentLanes(op.getInput()))
      return failure();

    rewriter.modifyOpInPlace(
        op, [&] { op.setPartAttr(rewriter.getStringAttr("EVEN")); });
    return success();
  }
};

struct FoldZeroGapExtensionPattern : public OpRewritePattern<VcvtOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(VcvtOp op,
                                PatternRewriter &rewriter) const override {
    std::optional<AlignedUnsignedWidening> widening =
        matchAlignedUnsignedWidening(op);
    if (!widening || !isCanonicalZeroGapCarrier(op.getInput(), *widening))
      return failure();

    Value carrier = stripVbitcasts(op.getInput());
    Value result = carrier.getType() == op.getResult().getType()
                       ? carrier
                       : rewriter
                             .create<VbitcastOp>(op.getLoc(),
                                                 op.getResult().getType(),
                                                 op.getInput())
                             .getResult();
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct VPTOOptimizeVcvtPass
    : public pto::impl::VPTOOptimizeVcvtBase<VPTOOptimizeVcvtPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<CanonicalizeEquivalentPartPattern,
                 FoldZeroGapExtensionPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVPTOOptimizeVcvtPass() {
  return std::make_unique<VPTOOptimizeVcvtPass>();
}

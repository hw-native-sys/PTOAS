// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOExpandL1L0a3d.cpp - A5 3D im2col load expansion helpers --------===//
//===----------------------------------------------------------------------===//

#include "PTO/Support/CodeConstants.h"
#include "VPTOExpandL1L0a3d.h"

#include "PTO/IR/PTO.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr uint64_t kU16Mask = 0xffffULL;
constexpr uint64_t kU8Mask = 0xffULL;

static bool foldI64Value(Value value, int64_t &out) {
  llvm::APInt constantValue;
  if (!matchPattern(value, m_ConstantInt(&constantValue))) {
    return false;
  }
  out = constantValue.getSExtValue();
  return true;
}

static uint64_t ceilToMultiple(uint64_t value, uint64_t multiple) {
  if (multiple == 0) {
    return value;
  }
  return ((value + multiple - 1) / multiple) * multiple;
}

static uint64_t packL3dFmatrixStatic(
    uint64_t fmapH, uint64_t fmapW, uint64_t padTop, uint64_t padBottom, uint64_t padLeft, uint64_t padRight)
{
    return (fmapW & kU16Mask) | ((fmapH & kU16Mask) << mlir::pto::kValue16) |
           ((padLeft & kU8Mask) << mlir::pto::kValue32) | ((padRight & kU8Mask) << mlir::pto::kValue40) |
           ((padTop & kU8Mask) << mlir::pto::kValue48) | ((padBottom & kU8Mask) << mlir::pto::kValue56);
}

static uint64_t packL3dRepeatStatic(uint64_t repeatSourceStride,
                                    uint64_t repeatTime, uint64_t repeatMode,
                                    uint64_t dstStride,
                                    uint64_t dstMPosition) {
  return (repeatSourceStride & kU16Mask) |
        ((repeatTime & kU8Mask) << mlir::pto::kValue16) |
        ((repeatMode & kU8Mask) << mlir::pto::kValue24) |
        ((dstStride & kU16Mask) << mlir::pto::kValue32) |
        ((dstMPosition & kU16Mask) << mlir::pto::kValue48);
}

// Masks, shifts, and ORs one control field into an accumulated 64-bit
// control word.
static Value packControlField(
    PatternRewriter& rewriter, Location loc, Value result, Value value, uint64_t mask, uint64_t shift)
{
    auto constant = [&rewriter, &loc](uint64_t v) { return rewriter.create<arith::ConstantIntOp>(loc, v, kValue64); };
    Value field = rewriter.create<arith::AndIOp>(loc, value, constant(mask));
    if (shift != 0) {
        field = rewriter.create<arith::ShLIOp>(loc, field, constant(shift));
    }
    return rewriter.create<arith::OrIOp>(loc, result, field);
}

static Value constantI64(PatternRewriter& rewriter, Location loc, uint64_t value)
{
    return rewriter.create<arith::ConstantIntOp>(loc, value, kValue64);
}

// Derived controls for the all-constant geometry: the arith chain would
// otherwise be a long constant chain, so fold them to i64 constants here.
static Derived3dLoadControls deriveStatic3dControlsImpl(
    PatternRewriter& rewriter, Location loc, MteL1L0a3dOp op, uint64_t c0Size, int64_t fmapH, int64_t fmapW,
    int64_t filterH, int64_t filterW, int64_t channelSize, int64_t strideH, int64_t strideW, int64_t dilationH,
    int64_t dilationW, int64_t padTop, int64_t padBottom, int64_t padLeft, int64_t padRight, int64_t repeatTime,
    int64_t repeatMode)
{
    Derived3dLoadControls controls;
    // The verifier rejects constant-zero stride/repeat_time; guard the
    // divisions anyway so the static analysis sees no zero divisor.
    uint64_t outputH = 0;
    uint64_t outputW = 0;
    uint64_t mStepStatic = 0;
    uint64_t kStepStatic = 0;
    if (strideH != 0 && strideW != 0 && repeatTime != 0 && c0Size != 0) {
        const uint64_t effectiveH = (filterH - 1) * dilationH + 1;
        const uint64_t effectiveW = (filterW - 1) * dilationW + 1;
        outputH = (fmapH + padTop + padBottom - effectiveH) / strideH + 1;
        outputW = (fmapW + padLeft + padRight - effectiveW) / strideW + 1;
        const uint64_t totalM = outputH * outputW;
        mStepStatic = (totalM + repeatTime - 1) / repeatTime;
        const uint64_t k = channelSize * filterH * filterW;
        kStepStatic = ceilToMultiple(k, c0Size);
        if (op.getDstStride()) {
            controls.dstStride = op.getDstStride();
            controls.repeatSourceStride = op.getRepeatSourceStride();
        } else {
            controls.dstStride = constantI64(rewriter, loc, (totalM + mlir::pto::kValue15) / mlir::pto::kValue16);
            controls.repeatSourceStride =
                constantI64(rewriter, loc, repeatMode == 0 ? (mStepStatic + 15) / 16 : kStepStatic / c0Size);
        }
    }
    controls.mStep = constantI64(rewriter, loc, mStepStatic);
    controls.kStep = constantI64(rewriter, loc, kStepStatic);
    return controls;
}

// Stride pair derivation for the dynamic path: dst_stride defaults to
// ceil(M/16) fractal blocks and repeat_source_stride to 16-row blocks in
// M direction or c0 blocks in K direction, selected by repeat_mode.
static void deriveDynamicStridePair(PatternRewriter &rewriter, Location loc,
                                    MteL1L0a3dOp op, Value totalM, Value mStep,
                                    Value kStep, Value c0SizeValue,
                                    Derived3dLoadControls &controls) {
  if (op.getDstStride()) {
    controls.dstStride = op.getDstStride();
    controls.repeatSourceStride = op.getRepeatSourceStride();
    return;
  }
  controls.dstStride = rewriter.create<arith::DivUIOp>(
      loc, rewriter.create<arith::AddIOp>(
        loc, totalM, constantI64(rewriter, loc, mlir::pto::kValue15)),
      constantI64(rewriter, loc, 16));
  Value mBlock = rewriter.create<arith::DivUIOp>(
      loc, rewriter.create<arith::AddIOp>(
               loc, mStep, constantI64(rewriter, loc, 15)),
      constantI64(rewriter, loc, 16));
  Value kBlock =
      rewriter.create<arith::DivUIOp>(loc, kStep, c0SizeValue);
  Value isM = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, op.getRepeatMode(),
      constantI64(rewriter, loc, 0));
  controls.repeatSourceStride =
      rewriter.create<arith::SelectOp>(loc, isM, mBlock, kBlock);
}

} // namespace

namespace mlir {
namespace pto {

bool fold3dGeometryControls(MteL1L0a3dOp op, int64_t &fmapH, int64_t &fmapW,
                            int64_t &filterH, int64_t &filterW,
                            int64_t &channelSize, int64_t &strideH,
                            int64_t &strideW, int64_t &dilationH,
                            int64_t &dilationW, int64_t &padTop,
                            int64_t &padBottom, int64_t &padLeft,
                            int64_t &padRight, int64_t &repeatTime,
                            int64_t &repeatMode) {
  return foldI64Value(op.getFmapH(), fmapH) &&
         foldI64Value(op.getFmapW(), fmapW) &&
         foldI64Value(op.getFilterH(), filterH) &&
         foldI64Value(op.getFilterW(), filterW) &&
         foldI64Value(op.getChannelSize(), channelSize) &&
         foldI64Value(op.getStrideH(), strideH) &&
         foldI64Value(op.getStrideW(), strideW) &&
         foldI64Value(op.getDilationH(), dilationH) &&
         foldI64Value(op.getDilationW(), dilationW) &&
         foldI64Value(op.getPadTop(), padTop) &&
         foldI64Value(op.getPadBottom(), padBottom) &&
         foldI64Value(op.getPadLeft(), padLeft) &&
         foldI64Value(op.getPadRight(), padRight) &&
         foldI64Value(op.getRepeatTime(), repeatTime) &&
         foldI64Value(op.getRepeatMode(), repeatMode);
}

Derived3dLoadControls deriveStatic3dControls(
    PatternRewriter &rewriter, Location loc, MteL1L0a3dOp op, uint64_t c0Size,
    int64_t fmapH, int64_t fmapW, int64_t filterH, int64_t filterW,
    int64_t channelSize, int64_t strideH, int64_t strideW, int64_t dilationH,
    int64_t dilationW, int64_t padTop, int64_t padBottom, int64_t padLeft,
    int64_t padRight, int64_t repeatTime, int64_t repeatMode) {
  return deriveStatic3dControlsImpl(rewriter, loc, op, c0Size, fmapH, fmapW,
                                    filterH, filterW, channelSize, strideH,
                                    strideW, dilationH, dilationW, padTop,
                                    padBottom, padLeft, padRight, repeatTime,
                                    repeatMode);
}

Derived3dLoadControls deriveDynamic3dControls(PatternRewriter& rewriter, Location loc, MteL1L0a3dOp op, uint64_t c0Size)
{
    Derived3dLoadControls controls;
    auto mul = [&rewriter, &loc](Value a, Value b) { return rewriter.create<arith::MulIOp>(loc, a, b); };
    auto add = [&rewriter, &loc](Value a, Value b) { return rewriter.create<arith::AddIOp>(loc, a, b); };
    auto sub = [&rewriter, &loc](Value a, Value b) { return rewriter.create<arith::SubIOp>(loc, a, b); };
    auto div = [&rewriter, &loc](Value a, Value b) {
        // Dynamic values cannot be range-checked here; clamp the divisor to >= 1
        // so the LLVM udiv never sees a zero divisor (poison in LLVM lowering).
        Value safe = rewriter.create<arith::MaxUIOp>(loc, b, constantI64(rewriter, loc, 1));
        return rewriter.create<arith::DivUIOp>(loc, a, safe);
    };
    // Dynamic dilation/repeat_time are clamped to >= 1 (constant-zero is
    // rejected by the verifier; the hardware treats dilation 0 as 1). The
    // padded-vs-effective difference is clamped to 0 so an undersized padded
    // feature map cannot wrap around to a huge derived M.
    Value dilationHSafe = rewriter.create<arith::MaxUIOp>(loc, op.getDilationH(), constantI64(rewriter, loc, 1));
    Value dilationWSafe = rewriter.create<arith::MaxUIOp>(loc, op.getDilationW(), constantI64(rewriter, loc, 1));
    // effectiveH = (filterH - 1) * dilationH + 1; paddedH = fmapH + padT + padB
    Value effectiveH =
        add(mul(sub(op.getFilterH(), constantI64(rewriter, loc, 1)), dilationHSafe), constantI64(rewriter, loc, 1));
    Value effectiveW =
        add(mul(sub(op.getFilterW(), constantI64(rewriter, loc, 1)), dilationWSafe), constantI64(rewriter, loc, 1));
    Value paddedH = add(add(op.getFmapH(), op.getPadTop()), op.getPadBottom());
    Value paddedW = add(add(op.getFmapW(), op.getPadLeft()), op.getPadRight());
    // The padded-vs-effective difference is interpreted as signed so an
    // undersized padded feature map (padded < effective, which wraps in
    // unsigned) clamps to 0 instead of to a huge value.
    Value diffH = rewriter.create<arith::MaxSIOp>(loc, sub(paddedH, effectiveH), constantI64(rewriter, loc, 0));
    Value diffW = rewriter.create<arith::MaxSIOp>(loc, sub(paddedW, effectiveW), constantI64(rewriter, loc, 0));
    Value outputH = add(div(diffH, op.getStrideH()), constantI64(rewriter, loc, 1));
    Value outputW = add(div(diffW, op.getStrideW()), constantI64(rewriter, loc, 1));
    Value totalM = mul(outputH, outputW);
    // Same zero-guard as div(): repeat_time=0 must not reach the udiv.
    Value repeatTimeSafe = rewriter.create<arith::MaxUIOp>(loc, op.getRepeatTime(), constantI64(rewriter, loc, 1));
    // mStep = ceil(totalM / repeatTime) = (totalM + repeatTime - 1) / repeatTime
    controls.mStep = rewriter.create<arith::DivUIOp>(
        loc, add(totalM, sub(repeatTimeSafe, constantI64(rewriter, loc, 1))), repeatTimeSafe);
    Value k = mul(mul(op.getChannelSize(), op.getFilterH()), op.getFilterW());
    Value c0SizeValue = constantI64(rewriter, loc, c0Size);
    controls.kStep = mul(div(add(k, constantI64(rewriter, loc, c0Size - 1)), c0SizeValue), c0SizeValue);
    deriveDynamicStridePair(rewriter, loc, op, totalM, controls.mStep, controls.kStep, c0SizeValue, controls);
    return controls;
}

Value packL3dFmatrix(
    PatternRewriter& rewriter, Location loc, Value fmapW, Value fmapH, Value padLeft, Value padRight, Value padTop,
    Value padBottom)
{
    int64_t w;
    int64_t h;
    int64_t l;
    int64_t r;
    int64_t t;
    int64_t b;
    const bool hasStaticFmatrix = foldI64Value(fmapW, w) && foldI64Value(fmapH, h) && foldI64Value(padLeft, l) &&
                                  foldI64Value(padRight, r) && foldI64Value(padTop, t) &&
                                  foldI64Value(padBottom, b);
    if (hasStaticFmatrix) {
        return constantI64(rewriter, loc, packL3dFmatrixStatic(h, w, t, b, l, r));
    }
    Value result = constantI64(rewriter, loc, 0);
    result = packControlField(rewriter, loc, result, fmapW, kU16Mask, 0);
    result = packControlField(rewriter, loc, result, fmapH, kU16Mask, mlir::pto::kValue16);
    result = packControlField(rewriter, loc, result, padLeft, kU8Mask, mlir::pto::kValue32);
    result = packControlField(rewriter, loc, result, padRight, kU8Mask, mlir::pto::kValue40);
    result = packControlField(rewriter, loc, result, padTop, kU8Mask, mlir::pto::kValue48);
    result = packControlField(rewriter, loc, result, padBottom, kU8Mask, mlir::pto::kValue56);
    return result;
}

Value packL3dRepeat(
    PatternRewriter& rewriter, Location loc, Value repeatSourceStride, Value repeatTime, Value repeatMode,
    Value dstStride, Value dstMPosition)
{
    int64_t rss;
    int64_t rt;
    int64_t rm;
    int64_t ds;
    int64_t dmp;
    const bool hasStaticRepeat = foldI64Value(repeatSourceStride, rss) && foldI64Value(repeatTime, rt) &&
                                 foldI64Value(repeatMode, rm) && foldI64Value(dstStride, ds) &&
                                 foldI64Value(dstMPosition, dmp);
    if (hasStaticRepeat) {
        return constantI64(rewriter, loc, packL3dRepeatStatic(rss, rt, rm, ds, dmp));
    }
    Value result = constantI64(rewriter, loc, 0);
    result = packControlField(rewriter, loc, result, repeatSourceStride, kU16Mask, 0);
    result = packControlField(rewriter, loc, result, repeatTime, kU8Mask, mlir::pto::kValue16);
    result = packControlField(rewriter, loc, result, repeatMode, kU8Mask, mlir::pto::kValue24);
    result = packControlField(rewriter, loc, result, dstStride, kU16Mask, mlir::pto::kValue32);
    result = packControlField(rewriter, loc, result, dstMPosition, kU16Mask, mlir::pto::kValue48);
    return result;
}

} // namespace pto
} // namespace mlir

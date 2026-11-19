// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOMteL1L0a.cpp - pto.MteL1L0a methods ----------------------------===//
//===----------------------------------------------------------------------===//

#include <limits>

#include "PTO/Support/CodeConstants.h"
#include "VPTOMteInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::mte_detail;

LogicalResult MteL1L0aOp::verify()
{
    if (failed(verifyCubeBridgeLoadLikeOp(*this, AddressSpace::LEFT, "LEFT"))) {
        return failure();
    }
    return verifyMteL1L0LoadOperands(
        getOperation(), {getM(), getK(), getStartRow(), getStartCol()}, {"m", "k", "start_row", "start_col"},
        {getMStart(), getKStart(), getMStep(), getKStep(), getSrcStride(), getDstStride()});
}

ParseResult MteL1L0aOp::parse(OpAsmParser& parser, OperationState& result)
{
    static constexpr StringRef kShapeNames[] = {"m", "k", "start_row", "start_col"};
    static constexpr StringRef kFullNames[] = {"m_start", "k_start", "m_step", "k_step", "src_stride", "dst_stride"};
    return parseMteL1L0OptionalOperandsOp<MteL1L0aOp>(parser, result, kShapeNames, kFullNames);
}

void MteL1L0aOp::print(OpAsmPrinter& p)
{
    static constexpr StringRef kShapeNames[] = {"m", "k", "start_row", "start_col"};
    static constexpr StringRef kFullNames[] = {"m_start", "k_start", "m_step", "k_step", "src_stride", "dst_stride"};
    printMteL1L0OptionalOperandsOp(
        p, getOperation(), getSource(), getDestination(), {getM(), getK(), getStartRow(), getStartCol()}, kShapeNames,
        {getMStart(), getKStart(), getMStep(), getKStep(), getSrcStride(), getDstStride()}, kFullNames);
}

void MteL1L0aOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>& effects)
{
    effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
    effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

//===--- A5 3D im2col load family (pto.mte_l1_l0a_3d, design v0.7) -------===//
// The wrapper is checked before expansion and the concrete load is checked
// again after expansion, when its controls are SSA operands rather than
// attributes.
static LogicalResult verifyL1L0a3dElementTypes(
    Operation* op, Type sourceType, Type destinationType, std::optional<Type> elementType)
{
    Type sourceElem = getBufferElementType(sourceType);
    Type destinationElem = getBufferElementType(destinationType);
    if (!sourceElem || !destinationElem) {
        return op->emitOpError("requires source and destination element types with known type");
    }
    if (sourceElem != destinationElem) {
        return op->emitOpError("requires source and destination element types to match");
    }
    if (elementType && *elementType != sourceElem) {
        return op->emitOpError("requires element_type to match source and destination element type");
    }
    if (!sourceElem.isF16() && !sourceElem.isBF16() && !sourceElem.isF32()) {
        auto intType = dyn_cast<IntegerType>(sourceElem);
        if (!intType || intType.getWidth() != mlir::pto::kValue8) {
            return op->emitOpError("requires f16, bf16, f32, or 8-bit integer element type");
        }
    }
    return success();
}

static LogicalResult verifyL1L0a3dDerivedControls(
    Operation* op, Type elementType, uint64_t fmapH, uint64_t fmapW, uint64_t filterH, uint64_t filterW,
    uint64_t strideH, uint64_t strideW, uint64_t dilationH, uint64_t dilationW, uint64_t padTop, uint64_t padBottom,
    uint64_t padLeft, uint64_t padRight, uint64_t channelSize)
{
    uint64_t elementBytes = getPTOStorageElemByteSize(elementType);
    if (elementBytes == 0 || elementBytes > mlir::pto::kValue32 || mlir::pto::kValue32 % elementBytes != 0) {
        return op->emitOpError("requires an element type with a byte width dividing the 32-byte C0");
    }

    uint64_t effectiveH = (filterH - 1) * dilationH + 1;
    uint64_t effectiveW = (filterW - 1) * dilationW + 1;
    uint64_t paddedH = fmapH + padTop + padBottom;
    uint64_t paddedW = fmapW + padLeft + padRight;
    if (paddedH < effectiveH || paddedW < effectiveW) {
        return success();
    }

    if (strideH == 0 || strideW == 0) {
        return op->emitOpError("stride must be non-zero");
    }
    uint64_t outputH = (paddedH - effectiveH) / strideH + 1;
    uint64_t outputW = (paddedW - effectiveW) / strideW + 1;
    if (outputH != 0 && outputW > std::numeric_limits<uint64_t>::max() / outputH) {
        return op->emitOpError("derived M extent overflows 64 bits");
    }
    uint64_t totalM = outputH * outputW;
    uint64_t c0Size = 32 / elementBytes;
    if (channelSize != 0 && filterH != 0 && filterW > std::numeric_limits<uint64_t>::max() / (channelSize * filterH)) {
        return op->emitOpError("derived K extent overflows 64 bits");
    }
    uint64_t k = channelSize * filterH * filterW;
    if (k > std::numeric_limits<uint64_t>::max() - (c0Size - 1)) {
        return op->emitOpError("derived K extent overflows 64 bits");
    }
    uint64_t kStep = ((k + c0Size - 1) / c0Size) * c0Size;

    if (totalM == 0 || totalM > mlir::pto::kValue65535) {
        return op->emitOpError() << "derived M extent out of range [1, 65535], got " << totalM;
    }
    if (kStep == 0 || kStep > mlir::pto::kValue65535) {
        return op->emitOpError() << "derived k_step out of range [1, 65535], got " << kStep;
    }
    return success();
}

static LogicalResult verifyL1L0a3dGeometryRanges(MteL1L0a3dOp op)
{
    const bool hasInvalidGeometry =
        failed(verifyStaticControlRange(op.getOperation(), op.getFmapH(), "fmap_h", 1, mlir::pto::kValue32767)) ||
        failed(verifyStaticControlRange(op.getOperation(), op.getFmapW(), "fmap_w", 1, mlir::pto::kValue32767)) ||
        failed(verifyStaticControlRange(op.getOperation(), op.getFilterH(), "filter_h", 1, mlir::pto::kValue511)) ||
        failed(verifyStaticControlRange(op.getOperation(), op.getFilterW(), "filter_w", 1, mlir::pto::kValue511)) ||
        failed(verifyStaticControlRange(
            op.getOperation(), op.getChannelSize(), "channel_size", 1, mlir::pto::kValue65535)) ||
        failed(verifyStaticControlRange(op.getOperation(), op.getStrideH(), "stride_h", 1, mlir::pto::kValue63)) ||
        failed(verifyStaticControlRange(op.getOperation(), op.getStrideW(), "stride_w", 1, mlir::pto::kValue63)) ||
        failed(verifyStaticControlRange(op.getOperation(), op.getDilationH(), "dilation_h", 1, mlir::pto::kValue255)) ||
        failed(verifyStaticControlRange(op.getOperation(), op.getDilationW(), "dilation_w", 1, mlir::pto::kValue255)) ||
        failed(verifyStaticControlRange(op.getOperation(), op.getPadTop(), "pad_top", 0, mlir::pto::kValue255)) ||
        failed(verifyStaticControlRange(op.getOperation(), op.getPadBottom(), "pad_bottom", 0, mlir::pto::kValue255)) ||
        failed(verifyStaticControlRange(op.getOperation(), op.getPadLeft(), "pad_left", 0, mlir::pto::kValue255)) ||
        failed(verifyStaticControlRange(op.getOperation(), op.getPadRight(), "pad_right", 0, mlir::pto::kValue255));
    if (hasInvalidGeometry) {
        return failure();
    }
    return success();
}

static LogicalResult verifyL1L0a3dControlRanges(MteL1L0a3dOp op)
{
    Operation* operation = op.getOperation();
    auto hasInvalidOptionalControl = [operation](Value control, StringRef name, int64_t min, int64_t max) {
        return control && failed(verifyStaticControlRange(operation, control, name, min, max));
    };
    const bool hasInvalidControls =
        failed(verifyStaticControlRange(op.getOperation(), op.getMStart(), "m_start", 0, mlir::pto::kValue32767)) ||
        failed(verifyStaticControlRange(op.getOperation(), op.getKStart(), "k_start", 0, mlir::pto::kValue65535)) ||
        failed(
            verifyStaticControlRange(op.getOperation(), op.getRepeatTime(), "repeat_time", 1, mlir::pto::kValue255)) ||
        failed(verifyStaticControlRange(op.getOperation(), op.getRepeatMode(), "repeat_mode", 0, 1)) ||
        failed(verifyStaticControlRange(
            op.getOperation(), op.getDstMPosition(), "dst_m_position", 0, mlir::pto::kValue65535)) ||
        hasInvalidOptionalControl(op.getRepeatSourceStride(), "repeat_source_stride", 0, mlir::pto::kValue65535) ||
        hasInvalidOptionalControl(op.getDstStride(), "dst_stride", 1, mlir::pto::kValue65535);
    if (hasInvalidControls) {
        return failure();
    }
    // (repeat_source_stride, dst_stride) must travel as a pair: programmatic
    // construction (C++/DSL builders) can bypass the textual IR format, so
    // defend here to keep the expansion from crashing on a half pair.
    if (static_cast<bool>(op.getRepeatSourceStride()) != static_cast<bool>(op.getDstStride())) {
        return op.emitOpError("repeat_source_stride and dst_stride must be provided together");
    }
    return success();
}

// Derived-geometry checks (padded fmap vs kernel, M, capacity, split/repeat
// start contract): only trigger when every relevant operand folds to a
// constant; dynamic inputs skip them and are handled by the expansion.
// Folds every geometry/control operand of the wrapper; also resolves the
// explicit stride pair. Returns true when all are compile-time constants.
static bool foldGeometryConstants(
    MteL1L0a3dOp op, int64_t& fmapH, int64_t& fmapW, int64_t& filterH, int64_t& filterW, int64_t& channelSize,
    int64_t& strideH, int64_t& strideW, int64_t& dilationH, int64_t& dilationW, int64_t& padTop, int64_t& padBottom,
    int64_t& padLeft, int64_t& padRight)
{
    auto fold = [](Value value, int64_t& out) {
        llvm::APInt constantValue;
        if (!matchPattern(value, m_ConstantInt(&constantValue))) {
            return false;
        }
        out = constantValue.getSExtValue();
        return true;
    };
    return fold(op.getFmapH(), fmapH) && fold(op.getFmapW(), fmapW) && fold(op.getFilterH(), filterH) &&
           fold(op.getFilterW(), filterW) && fold(op.getChannelSize(), channelSize) && fold(op.getStrideH(), strideH) &&
           fold(op.getStrideW(), strideW) && fold(op.getDilationH(), dilationH) && fold(op.getDilationW(), dilationW) &&
           fold(op.getPadTop(), padTop) && fold(op.getPadBottom(), padBottom) && fold(op.getPadLeft(), padLeft) &&
           fold(op.getPadRight(), padRight);
}

static bool foldL1L0a3dConstants(
    MteL1L0a3dOp op, int64_t& fmapH, int64_t& fmapW, int64_t& filterH, int64_t& filterW, int64_t& channelSize,
    int64_t& strideH, int64_t& strideW, int64_t& dilationH, int64_t& dilationW, int64_t& padTop, int64_t& padBottom,
    int64_t& padLeft, int64_t& padRight, int64_t& mStart, int64_t& kStart, int64_t& repeatTime, int64_t& repeatMode,
    int64_t& dstMPosition, int64_t& dstStride, int64_t& repeatSourceStride, bool& hasStrides)
{
    auto fold = [](Value value, int64_t& out) {
        llvm::APInt constantValue;
        if (!matchPattern(value, m_ConstantInt(&constantValue))) {
            return false;
        }
        out = constantValue.getSExtValue();
        return true;
    };
    const bool foldedGeometry = foldGeometryConstants(
        op, fmapH, fmapW, filterH, filterW, channelSize, strideH, strideW, dilationH, dilationW, padTop, padBottom,
        padLeft, padRight);
    const bool hasUnfoldedControl = !foldedGeometry || !fold(op.getMStart(), mStart) || !fold(op.getKStart(), kStart) ||
                                    !fold(op.getRepeatTime(), repeatTime) || !fold(op.getRepeatMode(), repeatMode) ||
                                    !fold(op.getDstMPosition(), dstMPosition);
    if (hasUnfoldedControl) {
        return false;
    }
    // Explicit stride pair: present but dynamic values also skip the derived checks.
    hasStrides = op.getDstStride() && op.getRepeatSourceStride();
    if (hasStrides && (!fold(op.getDstStride(), dstStride) || !fold(op.getRepeatSourceStride(), repeatSourceStride))) {
        return false;
    }
    return true;
}

// Derived M extent, capacity rules, and the split/repeat start contract;
// only reachable when every operand folded to a constant.
// Capacity rules (design spec): L1 source image within 512 KB, kernel
// footprint below 2^16 elements, and output row count below 2^15.
static LogicalResult verifyL1L0a3dCapacity(
    MteL1L0a3dOp op, int64_t fmapH, int64_t fmapW, int64_t filterH, int64_t filterW, int64_t channelSize,
    int64_t totalM)
{
    const int64_t elementBytes = getPTOStorageElemByteSize(getBufferElementType(op.getSource().getType()));
    if (fmapW > 0 && fmapH > 0 && channelSize > 0 &&
        fmapW * fmapH * channelSize * elementBytes > mlir::pto::kValue512 * mlir::pto::kValue1024) {
        return op.emitOpError() << "source feature map exceeds the L1 capacity: "
                                << fmapW * fmapH * channelSize * elementBytes << " bytes > "
                                << mlir::pto::kValue512 * mlir::pto::kValue1024;
    }
    if (filterW * filterH >= mlir::pto::kValue65536) {
        return op.emitOpError() << "filter footprint must be < 2^16 elements, got " << filterW * filterH;
    }
    if (totalM >= mlir::pto::kValue32768) {
        return op.emitOpError() << "derived M extent must be < 2^15, got " << totalM;
    }
    return success();
}

static LogicalResult verifyL1L0a3dMAndCapacity(
    MteL1L0a3dOp op, int64_t fmapH, int64_t fmapW, int64_t filterH, int64_t filterW, int64_t channelSize,
    int64_t strideH, int64_t strideW, int64_t dilationH, int64_t dilationW, int64_t padTop, int64_t padBottom,
    int64_t padLeft, int64_t padRight, int64_t mStart, int64_t kStart, int64_t repeatTime, int64_t* totalM,
    int64_t* mStep)
{
    const int64_t hKernel = (filterH - 1) * dilationH + 1;
    const int64_t wKernel = (filterW - 1) * dilationW + 1;
    const int64_t paddedH = fmapH + padTop + padBottom;
    const int64_t paddedW = fmapW + padLeft + padRight;
    if (paddedH < hKernel || paddedW < wKernel) {
        return op.emitOpError() << "padded feature map smaller than dilated kernel " << hKernel << "x" << wKernel;
    }
    if (strideH == 0 || strideW == 0) {
        return op.emitOpError("stride must be non-zero");
    }
    const int64_t outputH = (paddedH - hKernel) / strideH + 1;
    const int64_t outputW = (paddedW - wKernel) / strideW + 1;
    *totalM = outputH * outputW;
    if (*totalM <= 0) {
        return op.emitOpError() << "derived M extent must be positive, got " << *totalM;
    }
    if (failed(verifyL1L0a3dCapacity(op, fmapH, fmapW, filterH, filterW, channelSize, *totalM))) {
        return failure();
    }
    // Start + step must stay within the derived extent (split/repeat contract).
    if (repeatTime == 0) {
        return op.emitOpError("repeat_time must be non-zero");
    }
    *mStep = (*totalM + repeatTime - 1) / repeatTime;
    if (mStart + *mStep > *totalM) {
        return op.emitOpError() << "m_start + m_step exceeds the derived M extent: " << mStart << " + " << *mStep
                                << " > " << *totalM;
    }
    if (kStart != 0) {
        return op.emitOpError() << "k_start must be zero: split-K is not supported by the 3D load "
                                   "lowering, got k_start="
                                << kStart;
    }
    LogicalResult derivedControlsResult = verifyL1L0a3dDerivedControls(
        op.getOperation(), getBufferElementType(op.getSource().getType()), fmapH, fmapW, filterH, filterW, strideH,
        strideW, dilationH, dilationW, padTop, padBottom, padLeft, padRight, channelSize);
    if (failed(derivedControlsResult)) {
        return failure();
    }
    return success();
}

static LogicalResult verifyL1L0a3dDerivedGeometry(
    MteL1L0a3dOp op, int64_t* totalM, int64_t* mStep, int64_t* repeatTime, int64_t* repeatMode, int64_t* mStart,
    int64_t* repeatSourceStride, bool* hasStrides)
{
    int64_t fmapH;
    int64_t fmapW;
    int64_t filterH;
    int64_t filterW;
    int64_t channelSize;
    int64_t strideH;
    int64_t strideW;
    int64_t dilationH;
    int64_t dilationW;
    int64_t padTop;
    int64_t padBottom;
    int64_t padLeft;
    int64_t padRight;
    int64_t kStart;
    int64_t dstMPosition;
    int64_t dstStride = 0;
    const bool foldedConstants = foldL1L0a3dConstants(
        op, fmapH, fmapW, filterH, filterW, channelSize, strideH, strideW, dilationH, dilationW, padTop, padBottom,
        padLeft, padRight, *mStart, kStart, *repeatTime, *repeatMode, dstMPosition, dstStride, *repeatSourceStride,
        *hasStrides);
    if (!foldedConstants) {
        return success();
    }
    return verifyL1L0a3dMAndCapacity(
        op, fmapH, fmapW, filterH, filterW, channelSize, strideH, strideW, dilationH, dilationW, padTop, padBottom,
        padLeft, padRight, *mStart, kStart, *repeatTime, totalM, mStep);
}

static LogicalResult verifyL1L0a3dRepeatControls(
    MteL1L0a3dOp op, int64_t totalM, int64_t mStep, int64_t repeatTime, int64_t repeatMode, int64_t mStart,
    int64_t repeatSourceStride, bool hasStrides)
{
    if (repeatTime == 0) {
        return success();
    }
    if (repeatTime == 1) {
        return success();
    }
    if (repeatMode != 0) {
        return op.emitOpError() << "repeat_time>1 is only supported with repeat_mode=0 (M "
                                   "direction) by the current 3D load lowering, got repeat_mode="
                                << repeatMode;
    }
    if (totalM % repeatTime != 0) {
        return op.emitOpError() << "repeat_time=" << repeatTime << " must evenly divide the derived M extent "
                                << totalM;
    }
    const int64_t repeatChunk = totalM / repeatTime;
    if (repeatChunk % mlir::pto::kValue16 != 0) {
        return op.emitOpError() << "repeat chunk (M extent / repeat_time) = " << repeatChunk
                                << " must be a multiple of 16";
    }
    // The repeat window must stay inside the derived M extent. With an
    // explicit stride pair the user-provided repeat_source_stride is used;
    // otherwise the default derived rss (ceil(m_step / 16) 16-row blocks,
    // matching the expansion) applies, which makes m_start>0 invalid for a
    // full-load repeat since the window then exceeds the extent.
    const int64_t defaultRss = (mStep + 15) / 16;
    const int64_t rss = hasStrides ? repeatSourceStride : defaultRss;
    const int64_t repeatWindowEnd = mStart + rss * 16 * (repeatTime - 1) + repeatChunk;
    if (repeatWindowEnd > totalM) {
        return op.emitOpError() << "repeat window exceeds the derived M extent: m_start=" << mStart
                                << ", repeat_source_stride=" << rss << (hasStrides ? "" : " (derived 16-row blocks)")
                                << ", repeat_time=" << repeatTime << ", chunk=" << repeatChunk
                                << ", M extent=" << totalM;
    }
    return success();
}

LogicalResult MteL1L0a3dOp::verify()
{
    if (failed(verifyCubeBridgeLoadLikeOp(*this, AddressSpace::LEFT, "LEFT"))) {
        return failure();
    }
    LogicalResult elementTypesResult =
        verifyL1L0a3dElementTypes(getOperation(), getSource().getType(), getDestination().getType(), std::nullopt);
    if (failed(elementTypesResult)) {
        return failure();
    }
    // Static range checks via the shared cube-bridge helper (same as the 2D
    // family): only apply when the operand folds to a compile-time constant;
    // dynamic values are constrained by the expansion/hardware.
    if (failed(verifyL1L0a3dGeometryRanges(*this)) || failed(verifyL1L0a3dControlRanges(*this))) {
        return failure();
    }
    int64_t totalM = 0;
    int64_t mStep = 0;
    int64_t repeatTime = 0;
    int64_t repeatMode = 0;
    int64_t mStart = 0;
    int64_t repeatSourceStride = 0;
    bool hasStrides = false;
    LogicalResult derivedGeometryResult = verifyL1L0a3dDerivedGeometry(
        *this, &totalM, &mStep, &repeatTime, &repeatMode, &mStart, &repeatSourceStride, &hasStrides);
    if (failed(derivedGeometryResult)) {
        return failure();
    }
    return verifyL1L0a3dRepeatControls(
        *this, totalM, mStep, repeatTime, repeatMode, mStart, repeatSourceStride, hasStrides);
}

static LogicalResult verifyLoadL1L0a3dSpaces(LoadL1L0a3dOp op)
{
    if (isBufferLike(op.getSource().getType()) && isBufferLike(op.getDestination().getType())) {
        if (failed(verifyCubeBridgeLoadLikeOp(op, AddressSpace::LEFT, "LEFT"))) {
            return failure();
        }
        return verifyL1L0a3dElementTypes(
            op.getOperation(), op.getSource().getType(), op.getDestination().getType(), op.getElementType());
    }
    if (!op.getElementType()) {
        return op.emitOpError("requires element_type metadata after pointer normalization");
    }
    auto sourcePtr = dyn_cast<LLVM::LLVMPointerType>(op.getSource().getType());
    auto destinationPtr = dyn_cast<LLVM::LLVMPointerType>(op.getDestination().getType());
    const bool hasInvalidAddressSpaces = !sourcePtr || !destinationPtr ||
                                         sourcePtr.getAddressSpace() != static_cast<unsigned>(AddressSpace::MAT) ||
                                         destinationPtr.getAddressSpace() != static_cast<unsigned>(AddressSpace::LEFT);
    if (hasInvalidAddressSpaces) {
        return op.emitOpError("requires MAT source and LEFT destination buffers or normalized LLVM pointers");
    }
    return success();
}

LogicalResult LoadL1L0a3dOp::verify()
{
    if (failed(verifyLoadL1L0a3dSpaces(*this))) {
        return failure();
    }
    const bool hasInvalidControls =
        failed(verifyStaticControlRange(getOperation(), getMStart(), "m_start", 0, mlir::pto::kValue32767)) ||
        failed(verifyStaticControlRange(getOperation(), getKStart(), "k_start", 0, mlir::pto::kValue65535)) ||
        failed(verifyStaticControlRange(getOperation(), getMStep(), "m_step", 1, mlir::pto::kValue65535)) ||
        failed(verifyStaticControlRange(getOperation(), getKStep(), "k_step", 1, mlir::pto::kValue65535)) ||
        failed(verifyStaticControlRange(getOperation(), getStrideW(), "stride_w", 1, mlir::pto::kValue63)) ||
        failed(verifyStaticControlRange(getOperation(), getStrideH(), "stride_h", 1, mlir::pto::kValue63)) ||
        failed(verifyStaticControlRange(getOperation(), getFilterW(), "filter_w", 1, mlir::pto::kValue511)) ||
        failed(verifyStaticControlRange(getOperation(), getFilterH(), "filter_h", 1, mlir::pto::kValue511)) ||
        failed(verifyStaticControlRange(getOperation(), getDilationW(), "dilation_w", 1, mlir::pto::kValue255)) ||
        failed(verifyStaticControlRange(getOperation(), getDilationH(), "dilation_h", 1, mlir::pto::kValue255)) ||
        failed(verifyStaticControlRange(getOperation(), getChannelSize(), "channel_size", 1, mlir::pto::kValue65535));
    if (hasInvalidControls) {
        return failure();
    }
    return success();
}

void MteL1L0a3dOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>& effects)
{
    effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
    effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

void LoadL1L0a3dOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>& effects)
{
    effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
    effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

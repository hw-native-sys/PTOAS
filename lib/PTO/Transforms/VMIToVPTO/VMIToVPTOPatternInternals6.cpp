// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#pragma once
//===- VMIToVPTOPatternInternals6.inc - VMIToVPTO internals -*- C++ -*-===//
//===----------------------------------------------------------------------===//

/// Report a masked cast sub-path that has no conversion instruction to carry
/// the predicate (issue #1671).  The one-to-N driver runs the patterns with
/// `applyPatternsAndFoldGreedily`, so a failing pattern is retried after every
/// other rewrite; remember the rejected operation so the diagnostic stays
/// single-shot instead of repeating once per retry.
static LogicalResult emitMaskedCastPathError(Operation* op, const Twine& message)
{
    constexpr StringLiteral kReportedAttr = "pto.vmi.masked_cast_rejected";
    if (op->hasAttr(kReportedAttr)) {
        return failure();
    }
    op->setAttr(kReportedAttr, UnitAttr::get(op->getContext()));
    return op->emitError() << kVMIDiagUnsupportedPrefix << message;
}

template <typename OpT>
struct OneToNVMIExtIOpPattern : OneToNOpConversionPattern<OpT> {
    using OneToNOpConversionPattern<OpT>::OneToNOpConversionPattern;

private:
    // vmi.extui is a zero-extension and vmi.extsi is a sign-extension, while the
    // 8-bit widening pto.vcvt pairs are signedness-specific on A5 (u8->u32
    // zero-extends and s8->s32 sign-extends; no 8->32 cross pair exists).  The
    // extension kind therefore fixes both vcvt element signednesses, independent
    // of the signless spelling a VMI operand may carry.
    Type getVcvtElementType(Type elementType) const
    {
        auto integerType = dyn_cast<IntegerType>(elementType);
        if (!integerType) {
            return elementType;
        }
        auto signedness = std::is_same_v<OpT, VMIExtUIOp> ? IntegerType::SignednessSemantics::Unsigned :
                                                            IntegerType::SignednessSemantics::Signed;
        return IntegerType::get(integerType.getContext(), integerType.getWidth(), signedness);
    }

    FailureOr<Value> getVcvtExtensionSource(OpT op, Value sourcePart, OneToNPatternRewriter& rewriter) const
    {
        auto sourceType = dyn_cast<VRegType>(sourcePart.getType());
        if (!sourceType) {
            return rewriter.notifyMatchFailure(op, "expected physical integer extension source");
        }
        Type vcvtElementType = getVcvtElementType(sourceType.getElementType());
        if (vcvtElementType == sourceType.getElementType()) {
            return sourcePart;
        }
        auto vcvtSourceType = VRegType::get(sourceType.getContext(), sourceType.getElementCount(), vcvtElementType);
        return bitcastVReg(op.getLoc(), sourcePart, vcvtSourceType, rewriter);
    }

    // Builds one integer-extension vcvt on the extension kind's element
    // signedness and reinterprets the result back to the signless or unsigned
    // type the converted users expect.
    FailureOr<Value> createExtensionVcvt(
        OpT op, VRegType expectedResultType, Value sourcePart, Value mask, StringRef part,
        OneToNPatternRewriter& rewriter) const
    {
        FailureOr<Value> vcvtSource = getVcvtExtensionSource(op, sourcePart, rewriter);
        if (failed(vcvtSource)) {
            return failure();
        }
        Type vcvtElementType = getVcvtElementType(expectedResultType.getElementType());
        auto vcvtResultType =
            VRegType::get(expectedResultType.getContext(), expectedResultType.getElementCount(), vcvtElementType);
        Value converted =
            rewriter
                .create<VcvtOp>(
                    op.getLoc(), vcvtResultType, *vcvtSource, mask, nullptr, nullptr, rewriter.getStringAttr(part))
                .getResult();
        bool alreadyConverted = converted.getType() == expectedResultType;
        if (alreadyConverted) {
            return converted;
        }
        return bitcastVReg(op.getLoc(), converted, expectedResultType, rewriter);
    }

    FailureOr<Value> buildFactorExtensionResult(
        OpT op, Value sourcePart, VRegType resultType, Value mask, StringRef part,
        OneToNPatternRewriter& rewriter) const
    {
        return createExtensionVcvt(op, resultType, sourcePart, mask, part, rewriter);
    }

    LogicalResult emitFactorExtension(
        OpT op, ValueRange sourceParts, ArrayRef<VRegType> resultVRegTypes, ArrayRef<StringRef> parts,
        int64_t liveParts, int64_t partStride, Value mask, ValueRange masks, OneToNPatternRewriter& rewriter) const
    {
        SmallVector<Value> results;
        results.reserve(resultVRegTypes.size());
        for (int64_t partIndex = 0; partIndex < liveParts; ++partIndex) {
            for (auto [chunkIndex, sourcePart] : llvm::enumerate(sourceParts)) {
                VRegType resultType = resultVRegTypes[partIndex * sourceParts.size() + chunkIndex];
                // Every conversion part of a chunk reads that chunk's authored
                // predicate; the unmasked path keeps its shared all-true seed.
                FailureOr<Value> partMask = getConversionMaskForChunk(masks, chunkIndex, mask);
                if (failed(partMask)) {
                    return rewriter.notifyMatchFailure(op, "failed to build integer extension predicate");
                }
                FailureOr<Value> result = buildFactorExtensionResult(
                    op, sourcePart, resultType, *partMask, parts[partIndex * partStride], rewriter);
                if (failed(result)) {
                    return failure();
                }
                results.push_back(*result);
            }
        }
        replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
        return success();
    }

    FailureOr<Value> buildDenseLaneExtensionResult(
        OpT op, Value sourcePart, VRegType resultType, Value mask, StringRef part,
        OneToNPatternRewriter& rewriter) const
    {
        return createExtensionVcvt(op, resultType, sourcePart, mask, part, rewriter);
    }

    LogicalResult emitDenseLaneExtension(
        OpT op, ValueRange sourceParts, ArrayRef<VRegType> resultTypes, VRegType sourceType, StringRef part,
        ValueRange masks, OneToNPatternRewriter& rewriter) const
    {
        Value unmaskedSeed;
        if (masks.empty()) {
            FailureOr<Value> mask = createAllTrueMaskForVReg(op.getLoc(), sourceType, rewriter);
            if (failed(mask)) {
                return rewriter.notifyMatchFailure(op, "failed to build integer extension seed mask");
            }
            unmaskedSeed = *mask;
        }
        SmallVector<Value> results;
        results.reserve(resultTypes.size());
        for (auto [index, sourceAndResult] : llvm::enumerate(llvm::zip_equal(sourceParts, resultTypes))) {
            auto [sourcePart, resultType] = sourceAndResult;
            FailureOr<Value> mask = getConversionMaskForChunk(masks, index, unmaskedSeed);
            if (failed(mask)) {
                return rewriter.notifyMatchFailure(op, "failed to build integer extension predicate");
            }
            FailureOr<Value> result = buildDenseLaneExtensionResult(op, sourcePart, resultType, *mask, part, rewriter);
            if (failed(result)) {
                return failure();
            }
            results.push_back(*result);
        }
        replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
        return success();
    }

    FailureOr<std::pair<ArrayRef<StringRef>, int64_t>> getExtensionPartPlan(
        OpT op, ValueRange sourceParts, ArrayRef<Type> resultTypes, unsigned sourceBits, unsigned resultBits,
        int64_t sourceStride, OneToNPatternRewriter& rewriter) const
    {
        constexpr int64_t kEvenOddFactor = 2;
        constexpr int64_t kPacked4Factor = 4;
        ArrayRef<StringRef> parts;
        int64_t factor = 0;
        if (resultBits == sourceBits * kEvenOddFactor) {
            static constexpr StringRef kEvenOddParts[] = {"EVEN", "ODD"};
            parts = kEvenOddParts;
            factor = kEvenOddFactor;
        } else if (resultBits == sourceBits * kPacked4Factor) {
            static constexpr StringRef kPacked4Parts[] = {"P0", "P1", "P2", "P3"};
            parts = kPacked4Parts;
            factor = kPacked4Factor;
        } else {
            return rewriter.notifyMatchFailure(
                op, "unsupported physical integer extension source/result width "
                    "relation");
        }
        if (sourceStride <= 0) {
            sourceStride = 1;
        }
        // The source carries one semantic element every `sourceStride` lanes, so
        // only every `sourceStride`-th conversion part is live; the rest are holes
        // that the result's deinterleave factor already names.  A unit source
        // stride keeps the full part set.
        if (factor % sourceStride != 0) {
            return rewriter.notifyMatchFailure(op, "unsupported physical integer extension source lane stride");
        }
        int64_t liveParts = factor / sourceStride;
        bool arityMismatch = resultTypes.size() != static_cast<size_t>(liveParts) * sourceParts.size();
        if (arityMismatch) {
            return rewriter.notifyMatchFailure(
                op, "unsupported physical integer extension source/result arity "
                    "relation");
        }
        return std::make_pair(parts, liveParts);
    }

    FailureOr<VRegType> validateDenseGroupSlotExtensionResult(
        OpT op, Type resultType, VMIVRegType sourceVMIType, VMIVRegType resultVMIType, unsigned sourceBits,
        unsigned resultBits, OneToNPatternRewriter& rewriter) const
    {
        if (sourceBits == 0 || resultBits == 0) {
            return rewriter.notifyMatchFailure(op, "group-slot integer extension requires positive bit widths");
        }
        FailureOr<int64_t> sourceLanes = getDataLanesPerPart(sourceVMIType.getElementType());
        FailureOr<int64_t> resultLanes = getDataLanesPerPart(resultVMIType.getElementType());
        bool carrierShapeMismatch = failed(sourceLanes) || failed(resultLanes) ||
                                    *sourceLanes != *resultLanes * static_cast<int64_t>(resultBits / sourceBits);
        auto physicalResultType = dyn_cast<VRegType>(resultType);
        bool invalidResultType = !physicalResultType || failed(resultLanes) ||
                                 physicalResultType.getElementCount() != *resultLanes ||
                                 pto::getPTOStorageElemBitWidth(physicalResultType.getElementType()) != resultBits;
        if (carrierShapeMismatch || invalidResultType) {
            return rewriter.notifyMatchFailure(
                op, carrierShapeMismatch ? "unsupported dense group-slot integer extension carrier shape" :
                                           "unsupported dense group-slot integer extension result type");
        }
        return physicalResultType;
    }

    FailureOr<Value> buildDenseGroupSlotExtensionResult(
        OpT op, Value sourcePart, Type resultType, VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
        IntegerType resultIntegerType, unsigned sourceBits, unsigned resultBits, OneToNPatternRewriter& rewriter) const
    {
        FailureOr<VRegType> physicalResultType = validateDenseGroupSlotExtensionResult(
            op, resultType, sourceVMIType, resultVMIType, sourceBits, resultBits, rewriter);
        if (failed(physicalResultType)) {
            return failure();
        }

        Value current = sourcePart;
        unsigned currentBits = sourceBits;
        while (currentBits < resultBits) {
            FailureOr<Value> next = extendDenseGroupSlotCarrier(op, current, currentBits, resultIntegerType, rewriter);
            if (failed(next)) {
                return failure();
            }
            current = *next;
            currentBits *= mlir::pto::kValue2;
        }
        FailureOr<Value> result = bitcastVReg(op.getLoc(), current, *physicalResultType, rewriter);
        if (failed(result)) {
            return rewriter.notifyMatchFailure(op, "failed to materialize dense group-slot unpack result");
        }
        return *result;
    }

    FailureOr<Value> extendDenseGroupSlotCarrier(
        OpT op, Value current, unsigned currentBits, IntegerType resultIntegerType,
        OneToNPatternRewriter& rewriter) const
    {
        unsigned nextBits = currentBits * 2;
        auto nextElementType = IntegerType::get(rewriter.getContext(), nextBits, resultIntegerType.getSignedness());
        FailureOr<int64_t> nextLanes = getDataLanesPerPart(nextElementType);
        if (failed(nextLanes)) {
            return rewriter.notifyMatchFailure(op, "failed to derive dense group-slot unpack result lanes");
        }
        auto nextType = VRegType::get(rewriter.getContext(), *nextLanes, nextElementType);
        auto currentType = dyn_cast<VRegType>(current.getType());
        bool unpackLaneMismatch = !currentType || currentType.getElementCount() != *nextLanes * 2;
        if (unpackLaneMismatch) {
            return rewriter.notifyMatchFailure(op, "dense group-slot unpack source/result lane mismatch");
        }
        Value part = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
        if constexpr (std::is_same_v<OpT, VMIExtSIOp>) {
            return rewriter.create<VsunpackOp>(op.getLoc(), nextType, current, part).getResult();
        }
        return rewriter.create<VzunpackOp>(op.getLoc(), nextType, current, part).getResult();
    }

    LogicalResult lowerDenseGroupSlotExtension(
        OpT op, ValueRange sourceParts, ArrayRef<Type> resultTypes, VMIVRegType sourceVMIType,
        VMIVRegType resultVMIType, IntegerType resultIntegerType, unsigned sourceBits, unsigned resultBits,
        ValueRange masks, OneToNPatternRewriter& rewriter) const
    {
        // This path widens with vsunpack/vzunpack only: there is no
        // zero-inactive conversion instruction to carry the predicate, so a
        // masked cast must fail loudly instead of dropping it (issue #1671).
        if (!masks.empty()) {
            return emitMaskedCastPathError(
                op, Twine("masked ") + OpT::getOperationName() +
                        " cannot carry its predicate on the dense group-slot unpack path: the widening is "
                        "vsunpack/vzunpack with no conversion instruction (issue #1671)");
        }
        SmallVector<Value> results;
        results.reserve(resultTypes.size());
        for (auto [sourcePart, resultType] : llvm::zip_equal(sourceParts, resultTypes)) {
            FailureOr<Value> result = buildDenseGroupSlotExtensionResult(
                op, sourcePart, resultType, sourceVMIType, resultVMIType, resultIntegerType, sourceBits, resultBits,
                rewriter);
            if (failed(result)) {
                return failure();
            }
            results.push_back(*result);
        }
        replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
        return success();
    }

    FailureOr<Value> buildLegacyGroupSlotExtensionResult(
        OpT op, Value sourcePart, Type resultType, VRegType conversionSourceType, Value slotMask, StringAttr part,
        unsigned resultBits, OneToNPatternRewriter& rewriter) const
    {
        auto resultVRegType = dyn_cast<VRegType>(resultType);
        bool invalidResultType =
            !resultVRegType || pto::getPTOStorageElemBitWidth(resultVRegType.getElementType()) != resultBits;
        if (invalidResultType) {
            return rewriter.notifyMatchFailure(op, "unsupported group-slot integer extension result type");
        }
        FailureOr<Value> conversionSource = bitcastVReg(op.getLoc(), sourcePart, conversionSourceType, rewriter);
        if (failed(conversionSource)) {
            return rewriter.notifyMatchFailure(op, "failed to expose group-slot extension source elements");
        }
        return createExtensionVcvt(op, resultVRegType, *conversionSource, slotMask, part.getValue(), rewriter);
    }

    FailureOr<std::tuple<int64_t, VRegType, Value>> prepareLegacyGroupSlotExtension(
        OpT op, ValueRange sourceParts, ArrayRef<Type> resultTypes, VMIVRegType sourceVMIType,
        VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout, unsigned sourceBits, unsigned resultBits,
        OneToNPatternRewriter& rewriter) const
    {
        bool invalidShape = sourceLayout.getNumGroups() != resultLayout.getNumGroups() ||
                            sourceLayout.getSlots() != resultLayout.getSlots() ||
                            (sourceLayout.getSlots() != 1 && sourceLayout.getSlots() != 8) || sourceBits == 0 ||
                            sourceBits >= resultBits || resultBits % sourceBits != 0 ||
                            (resultBits / sourceBits != 2 && resultBits / sourceBits != 4) ||
                            (sourceLayout.getSlots() == 8 && sourceLayout.getLaneStride() != resultBits / sourceBits) ||
                            resultLayout.getLaneStride() != 1 || sourceParts.size() != resultTypes.size();
        if (invalidShape) {
            return rewriter.notifyMatchFailure(op, "unsupported group-slot integer extension shape");
        }
        FailureOr<int64_t> sourceLanes = getDataLanesPerPart(sourceVMIType.getElementType());
        if (failed(sourceLanes)) {
            return rewriter.notifyMatchFailure(op, "failed to derive group-slot integer extension source lanes");
        }
        auto conversionSourceType =
            VRegType::get(rewriter.getContext(), *sourceLanes, getVcvtElementType(sourceVMIType.getElementType()));
        FailureOr<MaskType> maskType = getMaskTypeForVReg(conversionSourceType, rewriter.getContext());
        if (failed(maskType)) {
            return rewriter.notifyMatchFailure(op, "failed to create group-slot integer extension mask type");
        }
        FailureOr<Value> slotMask = createPrefixMaskForActiveLanes(
            op.getLoc(), *maskType, sourceLayout.getSlots() * sourceLayout.getLaneStride(), rewriter);
        if (failed(slotMask)) {
            return rewriter.notifyMatchFailure(op, "failed to build group-slot integer extension mask");
        }
        return std::make_tuple(static_cast<int64_t>(resultBits / sourceBits), conversionSourceType, *slotMask);
    }

    LogicalResult lowerLegacyGroupSlotExtension(
        OpT op, ValueRange sourceParts, ArrayRef<Type> resultTypes, VMIVRegType sourceVMIType,
        VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout, unsigned sourceBits, unsigned resultBits,
        ValueRange masks, OneToNPatternRewriter& rewriter) const
    {
        FailureOr<std::tuple<int64_t, VRegType, Value>> preparation = prepareLegacyGroupSlotExtension(
            op, sourceParts, resultTypes, sourceVMIType, sourceLayout, resultLayout, sourceBits, resultBits, rewriter);
        if (failed(preparation)) {
            return failure();
        }
        int64_t widenFactor = std::get<0>(*preparation);
        VRegType conversionSourceType = std::get<1>(*preparation);
        Value slotMask = std::get<2>(*preparation);
        StringAttr part = rewriter.getStringAttr(widenFactor == 2 ? "EVEN" : "P0");
        // A group-slot part carries padding plus the active slots, and the
        // authored predicate describes only the active slots.  Combine the two
        // so the conversion still never converts a padding slot while the
        // inactive active slots are zeroed by the conversion contract.
        SmallVector<Value> partMasks;
        if (!masks.empty()) {
            auto slotMaskType = dyn_cast<MaskType>(slotMask.getType());
            if (!slotMaskType) {
                return rewriter.notifyMatchFailure(op, "unsupported group-slot integer extension mask type");
            }
            partMasks.reserve(sourceParts.size());
            for (auto [index, sourcePart] : llvm::enumerate(sourceParts)) {
                FailureOr<Value> userMask = getConversionMaskForChunk(masks, index, Value());
                if (failed(userMask)) {
                    return rewriter.notifyMatchFailure(op, "failed to build a group-slot integer extension predicate");
                }
                FailureOr<Value> allTrue = createAllTrueMaskForVReg(op.getLoc(), conversionSourceType, rewriter);
                if (failed(allTrue)) {
                    return rewriter.notifyMatchFailure(
                        op, "failed to build the group-slot integer extension all-true mask");
                }
                partMasks.push_back(
                    rewriter.create<PandOp>(op.getLoc(), slotMaskType, *userMask, slotMask, *allTrue).getResult());
            }
        }
        SmallVector<Value> results;
        results.reserve(resultTypes.size());
        for (auto [index, sourceAndResult] : llvm::enumerate(llvm::zip_equal(sourceParts, resultTypes))) {
            auto [sourcePart, resultType] = sourceAndResult;
            Value conversionMask = partMasks.empty() ? slotMask : partMasks[index];
            FailureOr<Value> result = buildLegacyGroupSlotExtensionResult(
                op, sourcePart, resultType, conversionSourceType, conversionMask, part, resultBits, rewriter);
            if (failed(result)) {
                return failure();
            }
            results.push_back(*result);
        }
        replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
        return success();
    }

    LogicalResult lowerPhysicalExtension(
        OpT op, ValueRange sourceParts, ArrayRef<VRegType> resultVRegTypes, ArrayRef<Type> resultTypes,
        VRegType sourceType, unsigned sourceBits, unsigned resultBits, VMILayoutAttr sourceLayout,
        VMILayoutAttr resultLayout, ValueRange masks, OneToNPatternRewriter& rewriter) const
    {
        // The carrier view of each declared layout drives the plan, exactly as in
        // pto.vmi.extf: a single-carrier group-slot packet whose stride is the
        // widening factor, or a packet that keeps one group at lane 0 of every
        // part, selects the same source lanes as the dense lane-strided form and
        // forwards the physical part unchanged.
        VMILayoutAttr sourceView = getVMICastDenseCarrierView(sourceLayout, sourceType.getElementType());
        VMILayoutAttr resultView =
            resultVRegTypes.empty() ?
                VMILayoutAttr() :
                getVMICastDenseCarrierView(resultLayout, resultVRegTypes.front().getElementType());
        unsigned pairWidth = static_cast<unsigned>(kPairWidth);
        unsigned quadWidth = static_cast<unsigned>(kQuadWidth);
        bool evenRadix = resultBits == sourceBits * pairWidth && sourceView.getLaneStride() == kPairWidth;
        bool quadRadix = resultBits == sourceBits * quadWidth && sourceView.getLaneStride() == kQuadWidth;
        bool sourceSelectsPartLanes =
            sourceView && (evenRadix || quadRadix || isVMISingleGroupPerPartPacket(sourceLayout));
        bool denseLaneExtension = sourceView && resultView && sourceView.isContiguous() && resultView.isContiguous() &&
                                  resultView.getLaneStride() == 1 && sourceSelectsPartLanes &&
                                  resultTypes.size() == sourceParts.size();
        if (denseLaneExtension) {
            StringRef part = resultBits == sourceBits * 2 ? StringRef("EVEN") : StringRef("P0");
            return emitDenseLaneExtension(op, sourceParts, resultVRegTypes, sourceType, part, masks, rewriter);
        }

        int64_t sourceStride = sourceView && sourceView.isContiguous() ? sourceView.getLaneStride() : 1;
        FailureOr<std::pair<ArrayRef<StringRef>, int64_t>> partPlan =
            getExtensionPartPlan(op, sourceParts, resultTypes, sourceBits, resultBits, sourceStride, rewriter);
        if (failed(partPlan)) {
            return failure();
        }

        Value unmaskedSeed;
        if (masks.empty()) {
            FailureOr<Value> mask = createAllTrueMaskForVReg(op.getLoc(), sourceType, rewriter);
            if (failed(mask)) {
                return rewriter.notifyMatchFailure(op, "failed to build integer extension seed mask");
            }
            unmaskedSeed = *mask;
        }

        return emitFactorExtension(
            op, sourceParts, resultVRegTypes, partPlan->first, partPlan->second, sourceStride, unmaskedSeed, masks,
            rewriter);
    }

    FailureOr<SmallVector<VRegType>> collectExtensionResultTypes(
        OpT op, ArrayRef<Type> resultTypes, OneToNPatternRewriter& rewriter) const
    {
        SmallVector<VRegType> resultVRegTypes;
        resultVRegTypes.reserve(resultTypes.size());
        for (Type resultType : resultTypes) {
            auto resultVRegType = dyn_cast<VRegType>(resultType);
            bool invalidType = !resultVRegType || !isa<IntegerType>(resultVRegType.getElementType()) ||
                               (!resultVRegTypes.empty() && resultVRegType != resultVRegTypes.front());
            if (invalidType) {
                return rewriter.notifyMatchFailure(op, "unsupported physical integer extension result type");
            }
            resultVRegTypes.push_back(resultVRegType);
        }
        return resultVRegTypes;
    }

    FailureOr<VRegType> getUniformExtensionSourceType(
        OpT op, ValueRange sourceParts, OneToNPatternRewriter& rewriter) const
    {
        if (sourceParts.empty()) {
            return rewriter.notifyMatchFailure(op, "integer extension requires at least one physical source chunk");
        }
        auto sourceType = dyn_cast<VRegType>(sourceParts.front().getType());
        if (!sourceType) {
            return rewriter.notifyMatchFailure(op, "expected physical integer extension source");
        }
        for (Value sourcePart : sourceParts) {
            auto currentSourceType = dyn_cast<VRegType>(sourcePart.getType());
            if (!currentSourceType || currentSourceType != sourceType) {
                return rewriter.notifyMatchFailure(
                    op, "integer extension source physical parts must have matching "
                        "type");
            }
        }
        return sourceType;
    }

    struct ExtensionLoweringInput {
        VMIVRegType sourceVMIType;
        VMIVRegType resultVMIType;
        ValueRange sourceParts;
        SmallVector<Type> resultTypes;
        VRegType sourceType;
        VMILayoutAttr sourceLayout;
        VMILayoutAttr resultLayout;
    };

    FailureOr<ExtensionLoweringInput> getLoweringInput(
        OpT op, typename OneToNOpConversionPattern<OpT>::OpAdaptor adaptor, OneToNPatternRewriter& rewriter) const
    {
        auto sourceVMIType = cast<VMIVRegType>(op.getSource().getType());
        auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
        ValueRange sourceParts = adaptor.getSource();
        FailureOr<SmallVector<Type>> resultTypes = getConvertedResultTypes(op, 0, *this->getTypeConverter());
        if (failed(resultTypes)) {
            return failure();
        }
        FailureOr<VRegType> sourceType = getUniformExtensionSourceType(op, sourceParts, rewriter);
        if (failed(sourceType)) {
            return failure();
        }
        return ExtensionLoweringInput{
            sourceVMIType,
            resultVMIType,
            sourceParts,
            std::move(*resultTypes),
            *sourceType,
            sourceVMIType.getLayoutAttr(),
            resultVMIType.getLayoutAttr()};
    }

    LogicalResult lowerGroupSlotByLayout(
        OpT op, const ExtensionLoweringInput& input, ValueRange masks, OneToNPatternRewriter& rewriter) const
    {
        unsigned sourceBits = pto::getPTOStorageElemBitWidth(input.sourceVMIType.getElementType());
        unsigned resultBits = pto::getPTOStorageElemBitWidth(input.resultVMIType.getElementType());
        auto sourceIntegerType = dyn_cast<IntegerType>(input.sourceVMIType.getElementType());
        auto resultIntegerType = dyn_cast<IntegerType>(input.resultVMIType.getElementType());
        int64_t slots = input.sourceLayout.getSlots();
        bool denseGroupSlotExtension =
            sourceIntegerType && resultIntegerType &&
            input.sourceLayout.getNumGroups() == input.resultLayout.getNumGroups() &&
            input.sourceLayout.getLaneStride() == 1 && input.resultLayout.getLaneStride() == 1 &&
            input.sourceLayout.getSlots() == input.resultLayout.getSlots() &&
            (slots == 2 || slots == 4 || slots == 8) && sourceBits > 0 && resultBits > sourceBits &&
            resultBits % sourceBits == 0 && (resultBits / sourceBits == 2 || resultBits / sourceBits == 4) &&
            input.sourceParts.size() == input.resultTypes.size();
        if (denseGroupSlotExtension) {
            return lowerDenseGroupSlotExtension(
                op, input.sourceParts, input.resultTypes, input.sourceVMIType, input.resultVMIType, resultIntegerType,
                sourceBits, resultBits, masks, rewriter);
        }
        return lowerLegacyGroupSlotExtension(
            op, input.sourceParts, input.resultTypes, input.sourceVMIType, input.sourceLayout, input.resultLayout,
            sourceBits, resultBits, masks, rewriter);
    }

    LogicalResult lowerNonGroupSlotByLayout(
        OpT op, const ExtensionLoweringInput& input, ValueRange masks, OneToNPatternRewriter& rewriter) const
    {
        FailureOr<SmallVector<VRegType>> resultVRegTypes = collectExtensionResultTypes(op, input.resultTypes, rewriter);
        if (failed(resultVRegTypes)) {
            return failure();
        }
        unsigned sourceBits = pto::getPTOStorageElemBitWidth(input.sourceType.getElementType());
        unsigned resultBits = pto::getPTOStorageElemBitWidth(resultVRegTypes->front().getElementType());
        return lowerPhysicalExtension(
            op, input.sourceParts, *resultVRegTypes, input.resultTypes, input.sourceType, sourceBits, resultBits,
            input.sourceLayout, input.resultLayout, masks, rewriter);
    }

public:
    LogicalResult matchAndRewrite(
        OpT op, typename OneToNOpConversionPattern<OpT>::OpAdaptor adaptor,
        OneToNPatternRewriter& rewriter) const override
    {
        FailureOr<ExtensionLoweringInput> input = getLoweringInput(op, adaptor, rewriter);
        if (failed(input)) {
            return failure();
        }
        unsigned sourceElementBits = pto::getPTOStorageElemBitWidth(input->sourceVMIType.getElementType());
        FailureOr<SmallVector<Value>> masks =
            applyConversionPredicate(op, input->sourceParts, adaptor.getMask(), sourceElementBits, rewriter);
        if (failed(masks)) {
            return failure();
        }
        bool groupSlotLayouts = input->sourceLayout && input->resultLayout && input->sourceLayout.isGroupSlots() &&
                                input->resultLayout.isGroupSlots();
        if (groupSlotLayouts) {
            return lowerGroupSlotByLayout(op, *input, *masks, rewriter);
        }
        return lowerNonGroupSlotByLayout(op, *input, *masks, rewriter);
    }
};

// TruncI lowering support matrix
//
// Keep this comment aligned with both:
//   1. verifySupportedVMIToVPTOOps() diagnostics below, and
//   2. the actual OneToN lowering implemented in this pattern.
//
// Dense logical layouts
//   - deinterleaved factor 2/4 -> contiguous
//     Example: 32 -> 16 or 32 -> 8.
//     Lowering shape: emit vcvt parts EVEN/ODD or P0/P1/P2/P3, then merge
//     physical results when multiple source chunks contribute to one result.
//     A result lane stride scales the required source arity: a result whose
//     logical elements sit `laneStride` apart inside a wider carrier already
//     consumes `laneStride` of the converted sub-lanes of one source lane, so
//     each result chunk is filled by `factor / laneStride` source chunks and
//     the k-th converted sub-lane is selected by part `parts[k * laneStride]`.
//     Example: 32 -> 8 with result lane_stride = 2 takes two source chunks and
//     selects parts P0 + P2 before the merge.
//   - deinterleaved factor 4 -> deinterleaved factor 2
//     Example: 32 -> 16.
//     Lowering shape: emit vcvt EVEN/ODD per source chunk pair.
//   - contiguous lane_stride = 1 -> contiguous lane_stride = 2/4
//     Example: 16 -> 8 lane_stride=2, 32 -> 8 lane_stride=4.
//     Lowering shape: NOSAT keeps/bitcasts the source carrier; SAT emits vcvt
//     into the logical-element vector whose live results occupy the requested
//     strided lanes.
//
// Group-slots logical layouts
//   - slots = 1 preserves the layout for 2x/4x narrowing.
//   - slots = 8 multiplies the source lane stride by the narrowing factor.
//   - 2x narrowing lowers with part = EVEN; 4x narrowing uses part = P0.
//   - 32-bit integer -> 8-bit integer, slots = 8, result lane_stride = 4
//     Lowering shape: no vcvt; keep/bitcast the 32-bit carrier and let the
//     later store consume it as PK4_B32.
struct OneToNVMITruncIOpPattern : OneToNOpConversionPattern<VMITruncIOp> {
    using OneToNOpConversionPattern<VMITruncIOp>::OneToNOpConversionPattern;

private:
    FailureOr<std::pair<VRegType, VRegType>> getUniformTruncTypes(
        VMITruncIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes, OneToNPatternRewriter& rewriter) const
    {
        bool emptyPhysicalParts = sourceParts.empty() || resultTypes.empty();
        if (emptyPhysicalParts) {
            return rewriter.notifyMatchFailure(op, "trunci requires non-empty physical source and result parts");
        }
        auto sourceType = dyn_cast<VRegType>(sourceParts.front().getType());
        auto resultType = dyn_cast<VRegType>(resultTypes.front());
        bool invalidTypes = !sourceType || !isa<IntegerType>(sourceType.getElementType()) || !resultType ||
                            !isa<IntegerType>(resultType.getElementType());
        if (invalidTypes) {
            return rewriter.notifyMatchFailure(op, "unsupported physical trunci source/result type");
        }
        for (Value sourcePart : sourceParts) {
            auto currentType = dyn_cast<VRegType>(sourcePart.getType());
            if (!currentType || currentType != sourceType) {
                return rewriter.notifyMatchFailure(op, "trunci source physical parts must have matching integer type");
            }
        }
        for (Type physicalResultType : resultTypes) {
            auto currentType = dyn_cast<VRegType>(physicalResultType);
            if (!currentType || currentType != resultType) {
                return rewriter.notifyMatchFailure(op, "trunci result physical parts must have matching integer type");
            }
        }
        return std::make_pair(sourceType, resultType);
    }

    void finalizeResults(
        VMITruncIOp op, SmallVectorImpl<Value>& results, bool requiresResultBitcast, ArrayRef<Type> originalResultTypes,
        OneToNPatternRewriter& rewriter) const
    {
        if (requiresResultBitcast) {
            for (auto&& [index, result] : llvm::enumerate(results)) {
                result = rewriter.create<VbitcastOp>(op.getLoc(), originalResultTypes[index], result).getResult();
            }
        }
        replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
    }

    FailureOr<Value> lowerGroupSlotTruncPart(
        VMITruncIOp op, Value sourcePart, VRegType resultType, VMIVRegType resultVMIType, VMILayoutAttr resultLayout,
        unsigned sourceLogicalBits, unsigned resultLogicalBits, Value activeSlotMask, bool masked, StringAttr sat,
        OneToNPatternRewriter& rewriter) const
    {
        unsigned physicalResultBits = pto::getPTOStorageElemBitWidth(resultType.getElementType());
        bool directCarrier = resultLayout.hasLaneStride() && resultLayout.getLaneStride() == 4 &&
                             resultLogicalBits == 8 && physicalResultBits == 32;
        if (directCarrier) {
            // The 8-bit group-slot result is forwarded as a 32-bit carrier and
            // narrowed by the consuming store: no conversion instruction exists
            // here to carry the predicate, so a masked cast must fail loudly
            // instead of dropping it (issue #1671).
            if (masked) {
                return emitMaskedCastPathError(
                    op, "masked pto.vmi.trunci cannot carry its predicate on the group-slot direct-carrier "
                        "path: the 8-bit result is forwarded as a 32-bit carrier with no conversion "
                        "instruction (issue #1671)");
            }
            return lowerGroupSlotDirectCarrier(op, sourcePart, resultType, rewriter);
        }
        bool wideCarrier = resultLayout.hasLaneStride() && resultLayout.getLaneStride() == 2 &&
                           resultLogicalBits == 16 && physicalResultBits == 32;
        if (wideCarrier) {
            return lowerGroupSlotWideCarrier(op, sourcePart, resultType, resultVMIType, activeSlotMask, sat, rewriter);
        }
        bool validNarrowResult = physicalResultBits == 16 || physicalResultBits == 8;
        if (!validNarrowResult) {
            return rewriter.notifyMatchFailure(op, "unsupported group-slot trunci physical type");
        }
        StringAttr part = rewriter.getStringAttr(sourceLogicalBits == 2 * resultLogicalBits ? "EVEN" : "P0");
        return rewriter.create<VcvtOp>(op.getLoc(), resultType, sourcePart, activeSlotMask, nullptr, sat, part)
            .getResult();
    }

    FailureOr<Value> lowerGroupSlotDirectCarrier(
        VMITruncIOp op, Value sourcePart, VRegType resultType, OneToNPatternRewriter& rewriter) const
    {
        return sourcePart.getType() == resultType ?
                   sourcePart :
                   rewriter.create<VbitcastOp>(op.getLoc(), resultType, sourcePart).getResult();
    }

    FailureOr<Value> lowerGroupSlotWideCarrier(
        VMITruncIOp op, Value sourcePart, VRegType resultType, VMIVRegType resultVMIType, Value activeSlotMask,
        StringAttr sat, OneToNPatternRewriter& rewriter) const
    {
        FailureOr<int64_t> lanes = getDataLanesPerPart(resultVMIType.getElementType());
        if (failed(lanes)) {
            return rewriter.notifyMatchFailure(op, "failed to derive group-slot trunci conversion lanes");
        }
        auto conversionType = VRegType::get(rewriter.getContext(), *lanes, resultVMIType.getElementType());
        Value converted = rewriter
                              .create<VcvtOp>(
                                  op.getLoc(), conversionType, sourcePart, activeSlotMask, nullptr, sat,
                                  rewriter.getStringAttr("EVEN"))
                              .getResult();
        FailureOr<Value> carrier = bitcastVReg(op.getLoc(), converted, resultType, rewriter);
        if (failed(carrier)) {
            return rewriter.notifyMatchFailure(op, "failed to expose group-slot trunci result carrier");
        }
        return *carrier;
    }

    FailureOr<std::pair<bool, bool>> getGroupSlotTruncModes(
        VMITruncIOp op, VMIVRegType sourceVMIType, VMIVRegType resultVMIType, VMILayoutAttr sourceLayout,
        VMILayoutAttr resultLayout, ValueRange sourceParts, ArrayRef<Type> resultTypes,
        OneToNPatternRewriter& rewriter) const
    {
        unsigned sourceBits = pto::getPTOStorageElemBitWidth(sourceVMIType.getElementType());
        unsigned resultBits = pto::getPTOStorageElemBitWidth(resultVMIType.getElementType());
        bool supportsDirect = (sourceBits == 32 && (resultBits == 16 || resultBits == 8)) ||
                              (sourceBits == 16 && resultBits == 8 && sourceLayout.getSlots() == 1);
        bool supportsPacked = sourceBits == 16 && resultBits == 8 && sourceLayout.getSlots() == 8 &&
                              resultLayout.getSlots() == 8 && resultLayout.hasLaneStride() &&
                              resultLayout.getLaneStride() == sourceLayout.getLaneStride() * kPairWidth;
        bool invalidShape = sourceLayout.getNumGroups() != resultLayout.getNumGroups() ||
                            sourceLayout.getSlots() != resultLayout.getSlots() ||
                            (sourceLayout.getSlots() != 1 && sourceLayout.getSlots() != 8) ||
                            (!supportsDirect && !supportsPacked) || sourceParts.size() != resultTypes.size();
        if (invalidShape) {
            (void)rewriter.notifyMatchFailure(op, "unsupported group-slot trunci shape");
            return failure();
        }
        return std::make_pair(supportsDirect, supportsPacked);
    }

    FailureOr<SmallVector<Value>> lowerGroupSlotTruncParts(
        VMITruncIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes, VMIVRegType resultVMIType,
        VMILayoutAttr resultLayout, unsigned sourceBits, unsigned resultBits, bool supportsPacked, Value activeSlotMask,
        ValueRange partMasks, StringAttr sat, OneToNPatternRewriter& rewriter) const
    {
        SmallVector<Value> results;
        results.reserve(resultTypes.size());
        for (auto [index, sourceAndResult] : llvm::enumerate(llvm::zip_equal(sourceParts, resultTypes))) {
            auto [sourcePart, physicalResultType] = sourceAndResult;
            auto sourceType = dyn_cast<VRegType>(sourcePart.getType());
            auto resultType = dyn_cast<VRegType>(physicalResultType);
            bool validPhysicalTypes =
                sourceType && pto::getPTOStorageElemBitWidth(sourceType.getElementType()) == sourceBits && resultType;
            if (!validPhysicalTypes) {
                (void)rewriter.notifyMatchFailure(op, "unsupported group-slot trunci physical type");
                return failure();
            }
            Value conversionMask = partMasks.empty() ? activeSlotMask : partMasks[index];
            if (supportsPacked) {
                results.push_back(rewriter
                                      .create<VcvtOp>(
                                          op.getLoc(), resultType, sourcePart, conversionMask, nullptr, sat,
                                          rewriter.getStringAttr("EVEN"))
                                      .getResult());
                continue;
            }
            FailureOr<Value> lowered = lowerGroupSlotTruncPart(
                op, sourcePart, resultType, resultVMIType, resultLayout, sourceBits, resultBits, conversionMask,
                !partMasks.empty(), sat, rewriter);
            if (failed(lowered)) {
                return failure();
            }
            results.push_back(*lowered);
        }
        return results;
    }

    LogicalResult lowerGroupSlotTrunc(
        VMITruncIOp op, OpAdaptor adaptor, OneToNPatternRewriter& rewriter, VMIVRegType sourceVMIType,
        VMIVRegType resultVMIType, VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout, ArrayRef<Type> resultTypes,
        ValueRange masks) const
    {
        ValueRange sourceParts = adaptor.getSource();
        FailureOr<std::pair<bool, bool>> modes = getGroupSlotTruncModes(
            op, sourceVMIType, resultVMIType, sourceLayout, resultLayout, sourceParts, resultTypes, rewriter);
        if (failed(modes)) {
            return failure();
        }
        unsigned sourceLogicalBits = pto::getPTOStorageElemBitWidth(sourceVMIType.getElementType());
        unsigned resultLogicalBits = pto::getPTOStorageElemBitWidth(resultVMIType.getElementType());
        bool supportsPacked = modes->second;

        StringAttr sat = op->getAttrOfType<StringAttr>("saturate");
        bool forwarding = sourceLogicalBits == 16 && resultLogicalBits == 8 && sat && sat.getValue() == "NOSAT";
        if (forwarding) {
            return lowerGroupSlotTruncForwarding(op, sourceParts, resultTypes, masks, rewriter);
        }
        // Native 16-bit sums occupy the even halfword lanes. Include the whole
        // strided packet so all eight low halves reach the narrowing conversion.
        int64_t activeLanes = sourceLayout.getSlots() * sourceLayout.getLaneStride();
        StringRef activeSlotGranularity = sourceLogicalBits == 16 ? "b16" : "b32";
        FailureOr<Value> activeSlotMask = createPrefixMaskForActiveLanes(
            op.getLoc(), MaskType::get(rewriter.getContext(), activeSlotGranularity), activeLanes, rewriter);
        if (failed(activeSlotMask)) {
            return rewriter.notifyMatchFailure(op, "failed to build group-slot trunci active slot mask");
        }
        FailureOr<SmallVector<Value>> partMasks =
            buildGroupSlotTruncPartMasks(op, sourceParts, masks, *activeSlotMask, rewriter);
        if (failed(partMasks)) {
            return failure();
        }
        FailureOr<SmallVector<Value>> results = lowerGroupSlotTruncParts(
            op, sourceParts, resultTypes, resultVMIType, resultLayout, sourceLogicalBits, resultLogicalBits,
            supportsPacked, *activeSlotMask, *partMasks, sat, rewriter);
        if (failed(results)) {
            return failure();
        }
        finalizeResults(op, *results, false, resultTypes, rewriter);
        return success();
    }

    /// The saturate = "NOSAT" 16-bit to 8-bit group-slot view forwards the bytes:
    /// slots = 8 doubles the source lane stride for low bytes and slots = 1 has
    /// only lane zero live, so neither view needs a conversion instruction.  In
    /// particular A5 has no signed 16-to-signed-8 vcvt instruction, so a masked
    /// cast has no instruction to carry its predicate (issue #1671).
    LogicalResult lowerGroupSlotTruncForwarding(
        VMITruncIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes, ValueRange masks,
        OneToNPatternRewriter& rewriter) const
    {
        if (!masks.empty()) {
            return emitMaskedCastPathError(
                op, "masked pto.vmi.trunci cannot carry its predicate on the saturate = \"NOSAT\" "
                    "16-bit to 8-bit group-slot path: the bytes are forwarded with a bitcast and no "
                    "conversion instruction (issue #1671)");
        }
        SmallVector<Value> results;
        for (auto [source, type] : llvm::zip_equal(sourceParts, resultTypes)) {
            auto result = bitcastVReg(op.getLoc(), source, cast<VRegType>(type), rewriter);
            if (failed(result)) {
                return failure();
            }
            results.push_back(*result);
        }
        finalizeResults(op, results, false, resultTypes, rewriter);
        return success();
    }

    /// Combine the authored conversion predicate of every physical source part
    /// with the active-slot prefix mask.  A group-slot part carries padding plus
    /// the active slots, and the authored predicate describes only the active
    /// slots, so the two are ANDed: the conversion still never converts a padding
    /// slot while the conversion contract zeroes the inactive active slots.
    FailureOr<SmallVector<Value>> buildGroupSlotTruncPartMasks(
        VMITruncIOp op, ValueRange sourceParts, ValueRange masks, Value activeSlotMask,
        OneToNPatternRewriter& rewriter) const
    {
        SmallVector<Value> partMasks;
        if (masks.empty()) {
            return partMasks;
        }
        auto activeSlotMaskType = dyn_cast<MaskType>(activeSlotMask.getType());
        if (!activeSlotMaskType) {
            return rewriter.notifyMatchFailure(op, "unsupported group-slot trunci active slot mask type");
        }
        partMasks.reserve(sourceParts.size());
        for (auto [index, sourcePart] : llvm::enumerate(sourceParts)) {
            auto sourceType = dyn_cast<VRegType>(sourcePart.getType());
            if (!sourceType) {
                return rewriter.notifyMatchFailure(op, "unsupported group-slot trunci physical type");
            }
            FailureOr<Value> userMask = getConversionMaskForChunk(masks, index, Value());
            if (failed(userMask)) {
                return rewriter.notifyMatchFailure(op, "failed to build a group-slot trunci predicate");
            }
            FailureOr<Value> allTrue = createAllTrueMaskForVReg(op.getLoc(), sourceType, rewriter);
            if (failed(allTrue)) {
                return rewriter.notifyMatchFailure(op, "failed to build the group-slot trunci all-true mask");
            }
            partMasks.push_back(
                rewriter.create<PandOp>(op.getLoc(), activeSlotMaskType, *userMask, activeSlotMask, *allTrue)
                    .getResult());
        }
        return partMasks;
    }

    LogicalResult lowerDenseLaneStrideTrunc(
        VMITruncIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes, int64_t factor, StringAttr sat,
        bool requiresResultBitcast, ArrayRef<Type> originalResultTypes, ValueRange masks,
        OneToNPatternRewriter& rewriter) const
    {
        StringAttr part = rewriter.getStringAttr(factor == 2 ? "EVEN" : "P0");
        auto sourceType = dyn_cast<VRegType>(sourceParts.front().getType());
        if (!sourceType) {
            return rewriter.notifyMatchFailure(op, "unsupported dense trunci source type");
        }
        Value unmaskedSeed;
        if (masks.empty()) {
            FailureOr<Value> sourceMask = createAllTrueMaskForVReg(op.getLoc(), sourceType, rewriter);
            if (failed(sourceMask)) {
                return rewriter.notifyMatchFailure(op, "failed to build trunci masks");
            }
            unmaskedSeed = *sourceMask;
        }
        SmallVector<Value> results;
        results.reserve(resultTypes.size());
        for (auto [index, sourceAndResult] : llvm::enumerate(llvm::zip_equal(sourceParts, resultTypes))) {
            auto [sourcePart, resultType] = sourceAndResult;
            auto resultVRegType = dyn_cast<VRegType>(resultType);
            if (!resultVRegType) {
                return rewriter.notifyMatchFailure(op, "unsupported dense trunci result type");
            }
            FailureOr<Value> mask = getConversionMaskForChunk(masks, index, unmaskedSeed);
            if (failed(mask)) {
                return rewriter.notifyMatchFailure(op, "failed to build trunci predicate");
            }
            results.push_back(
                rewriter.create<VcvtOp>(op.getLoc(), resultVRegType, sourcePart, *mask, nullptr, sat, part)
                    .getResult());
        }
        finalizeResults(op, results, requiresResultBitcast, originalResultTypes, rewriter);
        return success();
    }

    LogicalResult lowerNoSatDenseCarrier(
        VMITruncIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes, ValueRange masks,
        OneToNPatternRewriter& rewriter) const
    {
        // NOSAT dense narrowing forwards the wide carrier and lets the consumer
        // read the low bytes: no conversion instruction exists here to carry the
        // predicate, so a masked cast must fail loudly instead of dropping it
        // (issue #1671).
        if (!masks.empty()) {
            return emitMaskedCastPathError(
                op, "masked pto.vmi.trunci cannot carry its predicate on the saturate = \"NOSAT\" dense "
                    "lane-stride path: the carrier is forwarded with a bitcast and no conversion "
                    "instruction (issue #1671)");
        }
        SmallVector<Value> results;
        results.reserve(resultTypes.size());
        for (auto [sourcePart, resultType] : llvm::zip_equal(sourceParts, resultTypes)) {
            FailureOr<Value> result = bitcastVReg(op.getLoc(), sourcePart, resultType, rewriter);
            if (failed(result)) {
                return rewriter.notifyMatchFailure(op, "failed to forward NOSAT trunci carrier");
            }
            results.push_back(*result);
        }
        replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
        return success();
    }

    // Physical plan for the dense factor route of an integer narrowing.
    //
    // `factor` is the element-width ratio; `resultLaneStride` is the lane stride
    // of the declared result layout, taken from its carrier view.  A result whose
    // elements sit `resultLaneStride` apart in a wider carrier consumes
    // `resultLaneStride` converted sub-lanes of one source lane per result lane,
    // so one result chunk is filled by `factor / resultLaneStride` source chunks
    // rather than by `factor` of them, and the part that carries the k-th
    // converted sub-lane is `parts[k * resultLaneStride]`.
    //
    // This is the same shape the fp-to-int narrowing already uses; before this
    // existed the required source arity was `factor` chunks per result chunk,
    // which only ever matched layouts whose result carried no lane stride.
    struct TruncINarrowingPlan {
        ArrayRef<StringRef> parts;
        int64_t factor = 0;
        int64_t sourceFactor = 0;
        int64_t resultLaneStride = 0;
        int64_t resultPartCount = 0;
        bool interleavedSourceParts = false;
    };

    FailureOr<Value> buildFactorTruncResult(
        VMITruncIOp op, ValueRange sourceParts, Type resultType, int64_t resultIndex, const TruncINarrowingPlan& plan,
        StringAttr sat, Value sourceMask, ValueRange masks, Value resultMask, OneToNPatternRewriter& rewriter) const
    {
        bool invalidPlan = plan.factor <= 0 || plan.sourceFactor <= 0 || plan.resultLaneStride <= 0 ||
                           static_cast<size_t>(plan.factor) > plan.parts.size();
        if (invalidPlan) {
            return rewriter.notifyMatchFailure(op, "unsupported physical trunci conversion factor");
        }
        auto resultVRegType = dyn_cast<VRegType>(resultType);
        if (!resultVRegType) {
            return rewriter.notifyMatchFailure(op, "unsupported physical trunci result type");
        }

        // Dense-split sources and lane-strided results both group source chunks
        // by conversion part across result chunks.
        SmallVector<Value> partials;
        partials.reserve(static_cast<size_t>(plan.sourceFactor));
        for (int64_t partIndex = 0; partIndex < plan.sourceFactor; ++partIndex) {
            int64_t sourceIndex = plan.interleavedSourceParts || plan.resultLaneStride > 1 ?
                                      partIndex * plan.resultPartCount + resultIndex :
                                      resultIndex * plan.sourceFactor + partIndex;
            if (sourceIndex < 0 || sourceIndex >= static_cast<int64_t>(sourceParts.size())) {
                return rewriter.notifyMatchFailure(op, "trunci source part index exceeds physical arity");
            }
            int64_t partNameIndex = partIndex * plan.resultLaneStride;
            if (partNameIndex < 0 || partNameIndex >= static_cast<int64_t>(plan.parts.size())) {
                return rewriter.notifyMatchFailure(op, "trunci part index exceeds the conversion factor");
            }
            // Every partial converts its own source chunk, so a masked cast
            // carries that chunk's predicate; the vor merge below is a pure
            // lane combine over already-zeroed inactive lanes.
            FailureOr<Value> mask = getConversionMaskForChunk(masks, sourceIndex, sourceMask);
            if (failed(mask)) {
                return rewriter.notifyMatchFailure(op, "failed to build a trunci predicate");
            }
            partials.push_back(rewriter
                                   .create<VcvtOp>(
                                       op.getLoc(), resultVRegType, sourceParts[sourceIndex], *mask, nullptr, sat,
                                       rewriter.getStringAttr(plan.parts[partNameIndex]))
                                   .getResult());
        }

        Value merged = partials.front();
        for (Value partial : llvm::drop_begin(partials)) {
            merged = rewriter.create<VorOp>(op.getLoc(), resultVRegType, merged, partial, resultMask).getResult();
        }
        return merged;
    }

    LogicalResult lowerFactorTrunc(
        VMITruncIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes, const TruncINarrowingPlan& plan,
        StringAttr sat, bool requiresResultBitcast, ArrayRef<Type> originalResultTypes, ValueRange masks,
        OneToNPatternRewriter& rewriter) const
    {
        auto sourceType = dyn_cast<VRegType>(sourceParts.front().getType());
        auto resultType = dyn_cast<VRegType>(resultTypes.front());
        if (!sourceType || !resultType) {
            return rewriter.notifyMatchFailure(op, "unsupported physical trunci source/result type");
        }
        Value sourceMask;
        if (masks.empty()) {
            FailureOr<Value> seed = createAllTrueMaskForVReg(op.getLoc(), sourceType, rewriter);
            if (failed(seed)) {
                return rewriter.notifyMatchFailure(op, "failed to build trunci masks");
            }
            sourceMask = *seed;
        }
        FailureOr<Value> resultMask = createAllTrueMaskForVReg(op.getLoc(), resultType, rewriter);
        if (failed(resultMask)) {
            return rewriter.notifyMatchFailure(op, "failed to build trunci masks");
        }
        SmallVector<Value> results;
        results.reserve(resultTypes.size());
        for (int64_t resultIndex = 0; resultIndex < static_cast<int64_t>(resultTypes.size()); ++resultIndex) {
            FailureOr<Value> result = buildFactorTruncResult(
                op, sourceParts, resultTypes[resultIndex], resultIndex, plan, sat, sourceMask, masks, *resultMask,
                rewriter);
            if (failed(result)) {
                return failure();
            }
            results.push_back(*result);
        }
        finalizeResults(op, results, requiresResultBitcast, originalResultTypes, rewriter);
        return success();
    }

    struct TruncIPhysicalPlan {
        VRegType sourceType;
        VRegType resultType;
        int64_t factor;
        bool denseLaneStride;
        StringAttr saturate;
    };

    FailureOr<TruncIPhysicalPlan> buildPhysicalPlan(
        VMITruncIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes, VMILayoutAttr sourceLayout,
        VMILayoutAttr resultLayout, OneToNPatternRewriter& rewriter) const
    {
        FailureOr<std::pair<VRegType, VRegType>> uniformTypes =
            getUniformTruncTypes(op, sourceParts, resultTypes, rewriter);
        if (failed(uniformTypes)) {
            return failure();
        }
        unsigned sourceBits = pto::getPTOStorageElemBitWidth(uniformTypes->first.getElementType());
        unsigned resultBits = pto::getPTOStorageElemBitWidth(uniformTypes->second.getElementType());
        bool invalidWidth = sourceBits == 0 || resultBits == 0 || sourceBits % resultBits != 0;
        if (invalidWidth) {
            return rewriter.notifyMatchFailure(op, "unsupported physical trunci source/result width relation");
        }
        int64_t factor = sourceBits / resultBits;
        bool denseLaneStride = sourceLayout && resultLayout && sourceLayout.isContiguous() &&
                               sourceLayout.getLaneStride() == 1 && resultLayout.isContiguous() &&
                               resultLayout.getLaneStride() == factor && sourceParts.size() == resultTypes.size();
        bool unsupportedDenseFactor = denseLaneStride && factor != 2 && factor != 4;
        if (unsupportedDenseFactor) {
            return rewriter.notifyMatchFailure(op, "unsupported dense lane_stride trunci result layout");
        }
        return TruncIPhysicalPlan{
            uniformTypes->first, uniformTypes->second, factor, denseLaneStride,
            op->getAttrOfType<StringAttr>("saturate")};
    }

    struct TruncIAliasPlan {
        SmallVector<Value> sourceParts;
        SmallVector<Type> resultTypes;
        SmallVector<Type> originalResultTypes;
        bool requiresResultBitcast = false;
    };

    TruncIAliasPlan materializeSignedNarrowingAlias(
        VMITruncIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes, const TruncIPhysicalPlan& physicalPlan,
        OneToNPatternRewriter& rewriter) const
    {
        TruncIAliasPlan aliasPlan;
        auto sourceIntegerType = dyn_cast<IntegerType>(physicalPlan.sourceType.getElementType());
        auto resultIntegerType = dyn_cast<IntegerType>(physicalPlan.resultType.getElementType());
        bool signedNarrowing = sourceIntegerType && resultIntegerType && sourceIntegerType.isSigned() &&
                               resultIntegerType.isSigned() &&
                               sourceIntegerType.getWidth() > resultIntegerType.getWidth();
        // A5 has no signed vcvt to an 8-bit destination (S16/S32 -> S8); the wider
        // signed destinations such as S32 -> S16 are native.  For the non-native
        // pairs a non-saturating truncation is bit-identical to the unsigned
        // conversion, which is the only form the hardware offers.
        bool noSat = !physicalPlan.saturate || physicalPlan.saturate.getValue() == "NOSAT";
        bool noNativeSignedNarrow = resultIntegerType.getWidth() == mlir::pto::kValue8;
        bool requiresAlias = signedNarrowing && noSat && noNativeSignedNarrow;
        if (!requiresAlias) {
            aliasPlan.sourceParts.assign(sourceParts.begin(), sourceParts.end());
            aliasPlan.resultTypes.assign(resultTypes.begin(), resultTypes.end());
            return aliasPlan;
        }
        // Both carrier aliases use unsigned integer element types.
        auto uSourceElemTy = rewriter.getIntegerType(sourceIntegerType.getWidth(), false);
        auto uResultElemTy = rewriter.getIntegerType(resultIntegerType.getWidth(), false);
        aliasPlan.originalResultTypes.assign(resultTypes.begin(), resultTypes.end());
        aliasPlan.sourceParts.reserve(sourceParts.size());
        for (Value sourcePart : sourceParts) {
            auto sourcePartType = cast<VRegType>(sourcePart.getType());
            aliasPlan.sourceParts.push_back(
                rewriter
                    .create<VbitcastOp>(
                        op.getLoc(),
                        VRegType::get(sourcePartType.getContext(), sourcePartType.getElementCount(), uSourceElemTy),
                        sourcePart)
                    .getResult());
        }
        aliasPlan.resultTypes.reserve(resultTypes.size());
        for (Type resultType : resultTypes) {
            auto resultVRegType = cast<VRegType>(resultType);
            aliasPlan.resultTypes.push_back(
                VRegType::get(resultVRegType.getContext(), resultVRegType.getElementCount(), uResultElemTy));
        }
        aliasPlan.requiresResultBitcast = true;
        return aliasPlan;
    }

    FailureOr<TruncINarrowingPlan> buildTruncINarrowingPlan(
        VMITruncIOp op, const TruncIPhysicalPlan& physicalPlan, VMIVRegType resultVMIType, VMILayoutAttr sourceLayout,
        size_t sourcePartCount, size_t resultPartCount, OneToNPatternRewriter& rewriter) const
    {
        int64_t factor = physicalPlan.factor;
        bool unsupportedFactor = factor != mlir::pto::kValue2 && factor != mlir::pto::kValue4;
        if (unsupportedFactor) {
            return rewriter.notifyMatchFailure(op, "unsupported physical trunci conversion factor");
        }
        VMILayoutAttr resultView =
            getVMICastDenseCarrierView(resultVMIType.getLayoutAttr(), resultVMIType.getElementType());
        int64_t resultLaneStride = resultView && resultView.isContiguous() ? resultView.getLaneStride() : 1;
        bool unsupportedLaneStride = resultLaneStride <= 0 || factor % resultLaneStride != 0;
        if (unsupportedLaneStride) {
            return rewriter.notifyMatchFailure(op, "unsupported physical trunci result lane stride");
        }
        int64_t sourceFactor = factor / resultLaneStride;
        bool sourceArityMismatch = sourcePartCount != static_cast<size_t>(sourceFactor) * resultPartCount;
        if (sourceArityMismatch) {
            return rewriter.notifyMatchFailure(op, "unsupported physical trunci source/result arity relation");
        }
        static constexpr StringRef kEvenOddParts[] = {"EVEN", "ODD"};
        static constexpr StringRef kPacked4Parts[] = {"P0", "P1", "P2", "P3"};
        ArrayRef<StringRef> parts =
            factor == mlir::pto::kValue2 ? ArrayRef<StringRef>(kEvenOddParts) : ArrayRef<StringRef>(kPacked4Parts);
        bool interleavedSourceParts =
            sourceLayout && sourceLayout.isDenseSplit() && resultView && resultView.isContiguous();
        return TruncINarrowingPlan{parts,
                                   factor,
                                   sourceFactor,
                                   resultLaneStride,
                                   static_cast<int64_t>(resultPartCount),
                                   interleavedSourceParts};
    }

    LogicalResult lowerNonGroupSlotTrunc(
        VMITruncIOp op, ValueRange sourceParts, SmallVector<Type> resultTypes, VMILayoutAttr sourceLayout,
        VMILayoutAttr resultLayout, ValueRange masks, OneToNPatternRewriter& rewriter) const
    {
        FailureOr<TruncIPhysicalPlan> physicalPlan =
            buildPhysicalPlan(op, sourceParts, resultTypes, sourceLayout, resultLayout, rewriter);
        if (failed(physicalPlan)) {
            return failure();
        }
        bool useNoSatCarrier =
            physicalPlan->denseLaneStride && physicalPlan->saturate && physicalPlan->saturate.getValue() == "NOSAT";
        if (useNoSatCarrier) {
            return lowerNoSatDenseCarrier(op, sourceParts, resultTypes, masks, rewriter);
        }
        TruncIAliasPlan aliasPlan =
            materializeSignedNarrowingAlias(op, sourceParts, resultTypes, *physicalPlan, rewriter);
        if (physicalPlan->denseLaneStride) {
            return lowerDenseLaneStrideTrunc(
                op, aliasPlan.sourceParts, aliasPlan.resultTypes, physicalPlan->factor, physicalPlan->saturate,
                aliasPlan.requiresResultBitcast, aliasPlan.originalResultTypes, masks, rewriter);
        }
        FailureOr<TruncINarrowingPlan> narrowingPlan = buildTruncINarrowingPlan(
            op, *physicalPlan, cast<VMIVRegType>(op.getResult().getType()), sourceLayout, aliasPlan.sourceParts.size(),
            aliasPlan.resultTypes.size(), rewriter);
        if (failed(narrowingPlan)) {
            return failure();
        }
        return lowerFactorTrunc(
            op, aliasPlan.sourceParts, aliasPlan.resultTypes, *narrowingPlan, physicalPlan->saturate,
            aliasPlan.requiresResultBitcast, aliasPlan.originalResultTypes, masks, rewriter);
    }

public:
    LogicalResult matchAndRewrite(VMITruncIOp op, OpAdaptor adaptor, OneToNPatternRewriter& rewriter) const override
    {
        FailureOr<VMIPhysicalConversionInput> input =
            getVMIPhysicalConversionInput(op, adaptor, *this->getTypeConverter());
        if (failed(input)) {
            return failure();
        }
        VMILayoutAttr sourceLayout = input->sourceVMIType.getLayoutAttr();
        VMILayoutAttr resultLayout = input->resultVMIType.getLayoutAttr();
        unsigned sourceElementBits = pto::getPTOStorageElemBitWidth(input->sourceVMIType.getElementType());
        FailureOr<SmallVector<Value>> masks =
            applyConversionPredicate(op, input->sourceParts, adaptor.getMask(), sourceElementBits, rewriter);
        if (failed(masks)) {
            return failure();
        }
        bool groupSlotLayouts =
            sourceLayout && resultLayout && sourceLayout.isGroupSlots() && resultLayout.isGroupSlots();
        if (groupSlotLayouts) {
            return lowerGroupSlotTrunc(
                op, adaptor, rewriter, input->sourceVMIType, input->resultVMIType, sourceLayout, resultLayout,
                input->resultTypes, *masks);
        }

        return lowerNonGroupSlotTrunc(
            op, input->sourceParts, std::move(input->resultTypes), sourceLayout, resultLayout, *masks, rewriter);
    }
};

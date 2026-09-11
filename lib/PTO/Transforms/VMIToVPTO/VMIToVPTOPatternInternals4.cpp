// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#pragma once
//===- VMIToVPTOPatternInternals4.inc - VMIToVPTO internals -*- C++ -*-===//
//===----------------------------------------------------------------------===//

struct OneToNVMIStrideLoadOpPattern
    : OneToNOpConversionPattern<VMIStrideLoadOp> {
  using OneToNOpConversionPattern<VMIStrideLoadOp>::OneToNOpConversionPattern;

private:
  LogicalResult lowerStrideLoad(
      VMIStrideLoadOp op, OneToNPatternRewriter &rewriter, Value source,
      Value offset, Value blockStride, Value repeatStride, ValueRange maskParts,
      ArrayRef<Type> resultTypes) const {
    bool invalidPhysicalArity = resultTypes.size() != 1 || maskParts.size() != 1;
    if (invalidPhysicalArity) {
      return rewriter.notifyMatchFailure(
          op, "stride_load supports one physical result/mask chunk");
    }
    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    if (!resultType || !isa<MaskType>(maskParts.front().getType())) {
      return rewriter.notifyMatchFailure(
          op, "stride_load requires physical vreg/mask parts");
    }
    Value base = rewriter
                     .create<AddPtrOp>(op.getLoc(), source.getType(), source,
                                       offset)
                     .getResult();
    Value loaded = rewriter
                       .create<VsldbOp>(op.getLoc(), resultType,
                                        /*updated_base=*/Type{}, base,
                                        blockStride, repeatStride,
                                        maskParts.front())
                       .getResult();
    replaceOpWithFlatConvertedValues(rewriter, op, SmallVector<Value>{loaded},
                                     *this->getTypeConverter());
    return success();
  }

public:

  LogicalResult
  matchAndRewrite(VMIStrideLoadOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    FailureOr<Value> source = getSingleValue(
        op, adaptor.getSource(), "stride_load source must convert to one value",
        rewriter);
    FailureOr<Value> offset = getSingleValue(
        op, adaptor.getOffset(), "stride_load offset must convert to one value",
        rewriter);
    FailureOr<Value> blockStride = getSingleValue(
        op, adaptor.getBlockStride(),
        "stride_load block_stride must convert to one value", rewriter);
    FailureOr<Value> repeatStride = Value(
        rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0, 16));
    bool invalidOperands = failed(source) || failed(offset) ||
                           failed(blockStride) || failed(repeatStride);
    if (invalidOperands) {
      return failure();
    }

    ValueRange maskParts = adaptor.getMask();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypesOrFailure(op, *this->getTypeConverter());
    if (failed(maybe_resultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    return lowerStrideLoad(op, rewriter, *source, *offset, *blockStride,
                           *repeatStride, maskParts, resultTypes);
  }
};

struct OneToNVMIStrideStoreOpPattern
    : OneToNOpConversionPattern<VMIStrideStoreOp> {
  using OneToNOpConversionPattern<VMIStrideStoreOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIStrideStoreOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    FailureOr<Value> destination = getSingleValue(
        op, adaptor.getDestination(),
        "stride_store destination must convert to one value", rewriter);
    FailureOr<Value> offset = getSingleValue(
        op, adaptor.getOffset(),
        "stride_store offset must convert to one value", rewriter);
    FailureOr<Value> blockStride = getSingleValue(
        op, adaptor.getBlockStride(),
        "stride_store block_stride must convert to one value", rewriter);
    FailureOr<Value> repeatStride = Value(
        rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0, 16));
    bool failedOperands = failed(destination) || failed(offset) ||
                          failed(blockStride) || failed(repeatStride);
    if (failedOperands) {
      return failure();
    }

    ValueRange valueParts = adaptor.getValue();
    ValueRange maskParts = adaptor.getMask();
    bool invalidArity = valueParts.size() != 1 || maskParts.size() != 1;
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, "stride_store supports one physical value/mask chunk");
    }
    bool invalidTypes = !isa<VRegType>(valueParts.front().getType()) ||
                        !isa<MaskType>(maskParts.front().getType());
    if (invalidTypes) {
      return rewriter.notifyMatchFailure(
          op, "stride_store requires physical vreg/mask parts");
    }

    Value base = rewriter
                     .create<AddPtrOp>(op.getLoc(), (*destination).getType(),
                                       *destination, *offset)
                     .getResult();
    rewriter.create<VsstbOp>(op.getLoc(), /*updated_base=*/Type{},
                             valueParts.front(), base, *blockStride,
                             *repeatStride, maskParts.front());
    rewriter.eraseOp(op);
    return success();
  }
};

struct OneToNVMIScatterOpPattern : OneToNOpConversionPattern<VMIScatterOp> {
  using OneToNOpConversionPattern<VMIScatterOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIScatterOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    FailureOr<Value> destination = getSingleValue(
        op, adaptor.getDestination(),
        "scatter destination must convert to one value", rewriter);
    if (failed(destination)) {
      return failure();
    }

    ValueRange valueParts = adaptor.getValue();
    ValueRange indicesParts = adaptor.getIndices();
    ValueRange maskParts = adaptor.getMask();
    bool invalidArity = valueParts.size() != indicesParts.size() ||
                        valueParts.size() != maskParts.size();
    if (invalidArity) {
      return rewriter.notifyMatchFailure(op, "scatter physical arity mismatch");
    }

    for (auto [value, indices, mask] :
         llvm::zip_equal(valueParts, indicesParts, maskParts)) {
      bool invalidTypes = !isa<VRegType>(value.getType()) ||
                          !isa<VRegType>(indices.getType()) ||
                          !isa<MaskType>(mask.getType());
      if (invalidTypes) {
        return rewriter.notifyMatchFailure(
            op, "scatter physical part type mismatch");
      }
      rewriter.create<VscatterOp>(op.getLoc(), value, *destination, indices,
                                  mask);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

template <typename SourceOp, typename TargetOp, bool IsMaskResult = false>
struct OneToNVMIBinaryOpPattern : OneToNOpConversionPattern<SourceOp> {
  using OneToNOpConversionPattern<SourceOp>::OneToNOpConversionPattern;

private:
  FailureOr<Value> lowerBinaryPart(
      SourceOp op, Value lhs, Value rhs, Type resultType,
      OneToNPatternRewriter &rewriter) const {
    if constexpr (IsMaskResult) {
      auto maskType = dyn_cast<MaskType>(resultType);
      bool invalidTypes =
          !maskType || lhs.getType() != resultType || rhs.getType() != resultType;
      if (invalidTypes) {
        return rewriter.notifyMatchFailure(
            op, "physical mask binary part type mismatch");
      }
      FailureOr<Value> seedMask =
          createAllTrueMask(op.getLoc(), maskType, rewriter);
      if (failed(seedMask)) {
        return rewriter.notifyMatchFailure(
            op, "unsupported mask type for all-true mask binary seed");
      }
      return rewriter
          .create<TargetOp>(op.getLoc(), resultType, lhs, rhs, *seedMask)
          .getResult();
    }
    auto vregType = dyn_cast<VRegType>(resultType);
    bool invalidTypes =
        !vregType || lhs.getType() != resultType || rhs.getType() != resultType;
    if (invalidTypes) {
      return rewriter.notifyMatchFailure(
          op, "physical binary part type mismatch");
    }
    FailureOr<Value> mask =
        createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter);
    if (failed(mask)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported element type for all-true binary mask");
    }
    return rewriter
        .create<TargetOp>(op.getLoc(), resultType, lhs, rhs, *mask)
        .getResult();
  }

  LogicalResult lowerBinaryParts(
      SourceOp op, ValueRange lhsParts, ValueRange rhsParts,
      ArrayRef<Type> resultTypes, OneToNPatternRewriter &rewriter) const {
    bool invalidArity = lhsParts.size() != rhsParts.size() ||
                        lhsParts.size() != resultTypes.size();
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, IsMaskResult ? "physical mask binary arity mismatch"
                           : "physical binary arity mismatch");
    }
    return lowerPointwisePhysicalParts(
        op, resultTypes,
        IsMaskResult ? "physical mask binary arity mismatch"
                     : "physical binary arity mismatch",
        rewriter,
        [&](int64_t index, Type resultType) -> FailureOr<Value> {
          return lowerBinaryPart(op, lhsParts[index], rhsParts[index],
                                 resultType, rewriter);
        },
        *this->getTypeConverter());
  }

public:
  LogicalResult matchAndRewrite(
      SourceOp op,
      typename OneToNOpConversionPattern<SourceOp>::OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter) const override {
    ValueRange lhsParts = adaptor.getLhs();
    ValueRange rhsParts = adaptor.getRhs();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypesOrFailure(op, *this->getTypeConverter());
    if (failed(maybe_resultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    return lowerBinaryParts(op, lhsParts, rhsParts, resultTypes, rewriter);
  }
};

// VPTO vector shifts require a signed shift-count carrier regardless of the
// signedness of the value being shifted. Preserve the count bits with a
// bitcast before creating the physical shift operation.
template <typename SourceOp, typename TargetOp>
struct OneToNVMIShiftOpPattern : OneToNOpConversionPattern<SourceOp> {
  using OneToNOpConversionPattern<SourceOp>::OneToNOpConversionPattern;

private:
  FailureOr<Value> lowerShiftPart(
      SourceOp op, Value lhs, Value rhs, Type resultType,
      OneToNPatternRewriter &rewriter) const {
    auto resultVRegType = dyn_cast<VRegType>(resultType);
    auto rhsVRegType = dyn_cast<VRegType>(rhs.getType());
    auto rhsElementType =
        rhsVRegType ? dyn_cast<IntegerType>(rhsVRegType.getElementType())
                    : IntegerType();
    bool invalidPhysicalPart =
        !resultVRegType || lhs.getType() != resultType || !rhsElementType;
    if (invalidPhysicalPart) {
      return rewriter.notifyMatchFailure(op, "physical shift part type mismatch");
    }
    auto signedElementType = IntegerType::get(
        rewriter.getContext(), rhsElementType.getWidth(),
        IntegerType::SignednessSemantics::Signed);
    auto signedRhsType = VRegType::get(
        rewriter.getContext(), rhsVRegType.getElementCount(), signedElementType);
    FailureOr<Value> signedRhs =
        bitcastVReg(op.getLoc(), rhs, signedRhsType, rewriter);
    if (failed(signedRhs)) {
      return rewriter.notifyMatchFailure(
          op, "unable to normalize physical shift-count type");
    }
    FailureOr<Value> mask =
        createAllTrueMaskForVReg(op.getLoc(), resultVRegType, rewriter);
    if (failed(mask)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported element type for all-true shift mask");
    }
    return rewriter
        .create<TargetOp>(op.getLoc(), resultType, lhs, *signedRhs, *mask)
        .getResult();
  }

public:

  LogicalResult matchAndRewrite(
      SourceOp op,
      typename OneToNOpConversionPattern<SourceOp>::OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter) const override {
    ValueRange lhsParts = adaptor.getLhs();
    ValueRange rhsParts = adaptor.getRhs();
    FailureOr<SmallVector<Type>> maybeResultTypes =
        getConvertedResultTypesOrFailure(op, *this->getTypeConverter());
    if (failed(maybeResultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybeResultTypes);
    const bool hasMismatchedPhysicalArity =
        lhsParts.size() != rhsParts.size() ||
        lhsParts.size() != resultTypes.size();
    if (hasMismatchedPhysicalArity) {
      return rewriter.notifyMatchFailure(op, "physical shift arity mismatch");
    }

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [lhs, rhs, resultType] :
         llvm::zip_equal(lhsParts, rhsParts, resultTypes)) {
      FailureOr<Value> shifted =
          lowerShiftPart(op, lhs, rhs, resultType, rewriter);
      if (failed(shifted)) {
        return failure();
      }
      results.push_back(*shifted);
    }

    return replacePhysicalResults(rewriter, op, results,
                                  *this->getTypeConverter());
  }
};

template <typename SourceOp, typename TargetOp>
struct OneToNVMIVecScalarOpPattern : OneToNOpConversionPattern<SourceOp> {
  using OneToNOpConversionPattern<SourceOp>::OneToNOpConversionPattern;

private:
  LogicalResult lowerVectorScalarParts(
      SourceOp op, ValueRange sourceParts, Value scalar,
      ValueRange maskParts, ArrayRef<Type> resultTypes,
      OneToNPatternRewriter &rewriter) const {
    const bool invalidArity =
        sourceParts.empty() || sourceParts.size() != maskParts.size() ||
        sourceParts.size() != resultTypes.size();
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, "physical vector-scalar arity mismatch");
    }

    return lowerPointwisePhysicalParts(
        op, resultTypes, "physical vector-scalar arity mismatch", rewriter,
        [&](int64_t index, Type resultType) -> FailureOr<Value> {
          auto vregType = dyn_cast<VRegType>(resultType);
          auto maskType = dyn_cast<MaskType>(maskParts[index].getType());
          const bool hasMismatchedPartType =
              !vregType || !maskType ||
              sourceParts[index].getType() != resultType;
          if (hasMismatchedPartType) {
            return rewriter.notifyMatchFailure(
                op, "physical vector-scalar part type mismatch");
          }
          return rewriter
              .create<TargetOp>(op.getLoc(), resultType, sourceParts[index],
                                scalar, maskParts[index])
              .getResult();
        },
        *this->getTypeConverter());
  }

public:
  LogicalResult matchAndRewrite(
      SourceOp op,
      typename OneToNOpConversionPattern<SourceOp>::OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter) const override {
    const bool requiresPassthru =
        op.getPmode().has_value() && *op.getPmode() == "merge";
    if (requiresPassthru) {
      return rewriter.notifyMatchFailure(
          op, "merge predicate mode requires an explicit passthru lowering");
    }

    ValueRange sourceParts = adaptor.getSrc();
    FailureOr<Value> scalar =
        getSingleValue(op, adaptor.getScalar(),
                       "vector-scalar scalar must convert to one value",
                       rewriter);
    ValueRange maskParts = adaptor.getMask();
    FailureOr<SmallVector<Type>> maybeResultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    const bool conversionFailed = failed(scalar) || failed(maybeResultTypes);
    if (conversionFailed) {
      return failure();
    }
    Value scalarValue = *scalar;
    SmallVector<Type> resultTypes = std::move(*maybeResultTypes);
    return lowerVectorScalarParts(op, sourceParts, scalarValue, maskParts,
                                  resultTypes, rewriter);
  }
};

struct OneToNVMIVaddcOpPattern : OneToNOpConversionPattern<VMIVaddcOp> {
  using OneToNOpConversionPattern<VMIVaddcOp>::OneToNOpConversionPattern;

private:
  LogicalResult lowerParts(VMIVaddcOp op, ValueRange lhsParts,
                           ValueRange rhsParts, ValueRange maskParts,
                           ArrayRef<Type> resultTypes,
                           ArrayRef<Type> carryTypes,
                           SmallVectorImpl<Value> &results,
                           SmallVectorImpl<Value> &carries,
                           OneToNPatternRewriter &rewriter) const {
    const bool invalidArity =
        lhsParts.empty() || rhsParts.size() != lhsParts.size() ||
        maskParts.size() != lhsParts.size() ||
        resultTypes.size() != lhsParts.size() ||
        carryTypes.size() != lhsParts.size();
    if (invalidArity) {
      return rewriter.notifyMatchFailure(op, "vaddc physical arity mismatch");
    }
    for (auto [lhs, rhs, mask, resultType, carryType] :
         llvm::zip_equal(lhsParts, rhsParts, maskParts, resultTypes,
                         carryTypes)) {
      auto dataType = dyn_cast<VRegType>(resultType);
      auto integerType = dataType
                             ? dyn_cast<IntegerType>(dataType.getElementType())
                             : IntegerType();
      const bool invalidPart =
          !dataType || !integerType || integerType.getWidth() != 32 ||
          !isa<MaskType>(mask.getType()) || !isa<MaskType>(carryType) ||
          !cast<MaskType>(carryType).isB32() || lhs.getType() != resultType ||
          rhs.getType() != resultType;
      if (invalidPart) {
        return rewriter.notifyMatchFailure(
            op, "vaddc requires matching 32-bit data and b32 mask parts");
      }
      auto addc = rewriter.create<VaddcOp>(op.getLoc(), resultType, carryType,
                                           lhs, rhs, mask);
      results.push_back(addc.getResult());
      carries.push_back(addc.getCarry());
    }
    return success();
  }

public:
  LogicalResult matchAndRewrite(VMIVaddcOp op, OpAdaptor adaptor,
                                OneToNPatternRewriter &rewriter) const override {
    ValueRange lhsParts = adaptor.getLhs();
    ValueRange rhsParts = adaptor.getRhs();
    ValueRange maskParts = adaptor.getMask();
    return lowerCarryResultParts(
        op, rewriter, *this->getTypeConverter(),
        [&](ArrayRef<Type> resultTypes, ArrayRef<Type> carryTypes,
            SmallVectorImpl<Value> &results,
            SmallVectorImpl<Value> &carries) {
          return lowerParts(op, lhsParts, rhsParts, maskParts, resultTypes,
                            carryTypes, results, carries, rewriter);
        });
  }
};

struct OneToNVMIVaddcsOpPattern : OneToNOpConversionPattern<VMIVaddcsOp> {
  using OneToNOpConversionPattern<VMIVaddcsOp>::OneToNOpConversionPattern;

private:
  LogicalResult lowerParts(VMIVaddcsOp op, ValueRange lhsParts,
                           ValueRange rhsParts, ValueRange carryInParts,
                           ValueRange maskParts, ArrayRef<Type> resultTypes,
                           ArrayRef<Type> carryTypes,
                           SmallVectorImpl<Value> &results,
                           SmallVectorImpl<Value> &carries,
                           OneToNPatternRewriter &rewriter) const {
    const bool invalidArity =
        lhsParts.empty() || rhsParts.size() != lhsParts.size() ||
        carryInParts.size() != lhsParts.size() ||
        maskParts.size() != lhsParts.size() ||
        resultTypes.size() != lhsParts.size() ||
        carryTypes.size() != lhsParts.size();
    if (invalidArity) {
      return rewriter.notifyMatchFailure(op,
                                         "vaddcs physical arity mismatch");
    }
    for (auto [lhs, rhs, carryIn, mask, resultType, carryType] :
         llvm::zip_equal(lhsParts, rhsParts, carryInParts, maskParts,
                         resultTypes, carryTypes)) {
      auto dataType = dyn_cast<VRegType>(resultType);
      auto integerType = dataType
                             ? dyn_cast<IntegerType>(dataType.getElementType())
                             : IntegerType();
      const bool invalidPart =
          !dataType || !integerType || integerType.getWidth() != 32 ||
          !isa<MaskType>(carryIn.getType()) || !isa<MaskType>(mask.getType()) ||
          !isa<MaskType>(carryType) ||
          !cast<MaskType>(carryIn.getType()).isB32() ||
          !cast<MaskType>(mask.getType()).isB32() ||
          !cast<MaskType>(carryType).isB32() || lhs.getType() != resultType ||
          rhs.getType() != resultType;
      if (invalidPart) {
        return rewriter.notifyMatchFailure(
            op, "vaddcs requires matching 32-bit data and b32 mask parts");
      }
      auto addcs = rewriter.create<VaddcsOp>(
          op.getLoc(), resultType, carryType, lhs, rhs, carryIn, mask);
      results.push_back(addcs.getResult());
      carries.push_back(addcs.getCarry());
    }
    return success();
  }

public:
  LogicalResult matchAndRewrite(VMIVaddcsOp op, OpAdaptor adaptor,
                                OneToNPatternRewriter &rewriter) const override {
    ValueRange lhsParts = adaptor.getLhs();
    ValueRange rhsParts = adaptor.getRhs();
    ValueRange carryInParts = adaptor.getCarryIn();
    ValueRange maskParts = adaptor.getMask();
    return lowerCarryResultParts(
        op, rewriter, *this->getTypeConverter(),
        [&](ArrayRef<Type> resultTypes, ArrayRef<Type> carryTypes,
            SmallVectorImpl<Value> &results,
            SmallVectorImpl<Value> &carries) {
          return lowerParts(op, lhsParts, rhsParts, carryInParts, maskParts,
                            resultTypes, carryTypes, results, carries,
                            rewriter);
        });
  }
};

struct OneToNVMIVmullOpPattern : OneToNOpConversionPattern<VMIVmullOp> {
  using OneToNOpConversionPattern<VMIVmullOp>::OneToNOpConversionPattern;

private:
  LogicalResult lowerPart(VMIVmullOp op, Value lhs, Value rhs, Value mask,
                          Type lowType, Type highType,
                          SmallVectorImpl<Value> &lows,
                          SmallVectorImpl<Value> &highs,
                          OneToNPatternRewriter &rewriter) const {
    auto dataType = dyn_cast<VRegType>(lowType);
    auto maskType = dyn_cast<MaskType>(mask.getType());
    const bool invalidShape =
        !dataType || dataType.getElementCount() != 64 || lowType != highType ||
        lhs.getType() != lowType || rhs.getType() != lowType;
    if (invalidShape) {
      return rewriter.notifyMatchFailure(
          op, "vmull requires matching 64-lane physical data part types");
    }
    auto elementType = dyn_cast<IntegerType>(dataType.getElementType());
    const bool invalidElementType =
        !elementType || elementType.getWidth() != 32 ||
        (!elementType.isSignless() && !elementType.isUnsigned());
    if (invalidElementType) {
      return rewriter.notifyMatchFailure(
          op, "vmull requires physical i32 or ui32 data parts");
    }
    if (!maskType || !maskType.isB32()) {
      return rewriter.notifyMatchFailure(
          op, "vmull requires a corresponding b32 mask part");
    }
    auto vmull = rewriter.create<VmullOp>(op.getLoc(), lowType, highType, lhs,
                                          rhs, mask);
    lows.push_back(vmull.getLow());
    highs.push_back(vmull.getHigh());
    return success();
  }

public:
  LogicalResult
  matchAndRewrite(VMIVmullOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange aParts = adaptor.getA();
    ValueRange bParts = adaptor.getB();
    ValueRange maskParts = adaptor.getMask();
    FailureOr<SmallVector<Type>> maybeLowTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    FailureOr<SmallVector<Type>> maybeHighTypes =
        getConvertedResultTypes(op, 1, *this->getTypeConverter());
    const bool conversionFailed =
        failed(maybeLowTypes) || failed(maybeHighTypes);
    if (conversionFailed) {
      return failure();
    }
    SmallVector<Type> lowTypes = std::move(*maybeLowTypes);
    SmallVector<Type> highTypes = std::move(*maybeHighTypes);

    size_t arity = aParts.size();
    const bool invalidArity =
        arity == 0 || bParts.size() != arity || maskParts.size() != arity ||
        lowTypes.size() != arity || highTypes.size() != arity;
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, "physical vmull arity mismatch across a, b, mask, low, and high");
    }

    SmallVector<Value> lows;
    SmallVector<Value> highs;
    lows.reserve(arity);
    highs.reserve(arity);
    for (size_t index = 0; index < arity; ++index) {
      if (failed(lowerPart(op, aParts[index], bParts[index], maskParts[index],
                           lowTypes[index], highTypes[index], lows, highs,
                           rewriter))) {
        return failure();
      }
    }

    SmallVector<Value> results;
    results.reserve(lows.size() + highs.size());
    results.append(lows);
    results.append(highs);
    return replacePhysicalResults(rewriter, op, results,
                                  *this->getTypeConverter());
  }
};

template <typename SourceOp, typename TargetOp>
struct OneToNVMIInterleaveOpPattern : OneToNOpConversionPattern<SourceOp> {
  using OneToNOpConversionPattern<SourceOp>::OneToNOpConversionPattern;

private:
  FailureOr<std::pair<Value, Value>> materializeLaneStrideInterleavePair(
      SourceOp op, OneToNPatternRewriter &rewriter, Value lhs, Value rhs,
      Type lowType, Type highType, int64_t carrierBits) const {
    FailureOr<VRegType> carrierType = getUnsignedCarrierVRegType(
        rewriter.getContext(), static_cast<unsigned>(carrierBits));
    if (failed(carrierType)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported lane-stride interleave carrier width");
    }
    FailureOr<Value> carrierLhs =
        bitcastVReg(op.getLoc(), lhs, *carrierType, rewriter);
    FailureOr<Value> carrierRhs =
        bitcastVReg(op.getLoc(), rhs, *carrierType, rewriter);
    bool failedInputs = failed(carrierLhs) || failed(carrierRhs);
    if (failedInputs) {
      return rewriter.notifyMatchFailure(
          op, "failed to bitcast lane-stride interleave inputs");
    }
    auto interleave = rewriter.create<TargetOp>(
        op.getLoc(), *carrierType, *carrierType, *carrierLhs, *carrierRhs);
    FailureOr<Value> low =
        bitcastVReg(op.getLoc(), interleave.getLow(), lowType, rewriter);
    FailureOr<Value> high =
        bitcastVReg(op.getLoc(), interleave.getHigh(), highType, rewriter);
    bool failedResults = failed(low) || failed(high);
    if (failedResults) {
      return rewriter.notifyMatchFailure(
          op, "failed to bitcast lane-stride interleave results");
    }
    return std::make_pair(*low, *high);
  }

  FailureOr<std::pair<SmallVector<Type>, SmallVector<Type>>>
  getInterleaveResultTypes(
      SourceOp op, ValueRange lhsParts, ValueRange rhsParts,
      ValueRange maskParts, OneToNPatternRewriter &rewriter) const {
    FailureOr<SmallVector<Type>> lowTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    FailureOr<SmallVector<Type>> highTypes =
        getConvertedResultTypes(op, 1, *this->getTypeConverter());
    bool invalidArity =
        failed(lowTypes) || failed(highTypes) || lhsParts.size() != rhsParts.size() ||
        lhsParts.size() != lowTypes->size() || lhsParts.size() != highTypes->size() ||
        (!maskParts.empty() && maskParts.size() != lhsParts.size());
    if (invalidArity) {
      return rewriter.notifyMatchFailure(op, "physical interleave arity mismatch");
    }
    return std::make_pair(std::move(*lowTypes), std::move(*highTypes));
  }

  FailureOr<VMIInterleaveLayoutFact> getInterleaveLayoutFact(
      SourceOp op, VMIVRegType lhsType, VMIVRegType rhsType,
      VMIMaskType maskType, VMIVRegType lowType, VMIVRegType highType,
      OneToNPatternRewriter &rewriter) const {
    VMILayoutSupport supports;
    FailureOr<VMIInterleaveLayoutFact> fact;
    if constexpr (std::is_same_v<SourceOp, VMIVintlvOp>) {
      fact = supports.getVintlvLayoutFactForLayouts(lhsType, rhsType, maskType,
                                                    lowType, highType);
    } else {
      fact = supports.getVdintlvLayoutFactForLayouts(lhsType, rhsType, maskType,
                                                     lowType, highType);
    }
    if (failed(fact)) {
      (void)rewriter.notifyMatchFailure(op, "unsupported interleave layout relation");
    }
    return fact;
  }

  enum class InterleaveLoweringKind { LaneStride, Contiguous, ZeroCopy };
  struct InterleaveLoweringPlan {
    InterleaveLoweringKind kind;
    int64_t inputFactor = 0;
    int64_t outputFactor = 0;
    bool zeroCopyVintlv = false;
  };

  static bool hasLaneStrideInterleaveLayout(
      const VMIInterleaveLayoutFact &fact) {
    return fact.lhsLayout == fact.rhsLayout &&
           fact.lhsLayout == fact.maskLayout &&
           fact.lhsLayout == fact.lowLayout &&
           fact.lhsLayout == fact.highLayout &&
           fact.lhsLayout.isContiguous() &&
           fact.lhsLayout.getLaneStride() > 1;
  }

  static bool hasUnitStrideContiguousInterleaveLayout(
      const VMIInterleaveLayoutFact &fact) {
    auto contiguous = [](VMILayoutAttr layout) {
      return layout && layout.isContiguous() && layout.getLaneStride() == 1;
    };
    return contiguous(fact.lhsLayout) && contiguous(fact.rhsLayout) &&
           contiguous(fact.maskLayout) && contiguous(fact.lowLayout) &&
           contiguous(fact.highLayout);
  }

  std::optional<InterleaveLoweringPlan> getZeroCopyInterleavePlan(
      const VMIInterleaveLayoutFact &fact) const {
    int64_t inputFactor = getElementDeinterleaveFactor(fact.lhsLayout);
    int64_t outputFactor = getElementDeinterleaveFactor(fact.lowLayout);
    bool matchingInputs = fact.rhsLayout == fact.lhsLayout &&
                          fact.maskLayout == fact.lhsLayout;
    bool matchingOutputs = fact.highLayout == fact.lowLayout;
    bool vintlv = std::is_same_v<SourceOp, VMIVintlvOp> && inputFactor > 0 &&
                  matchingInputs && matchingOutputs &&
                  outputFactor == 2 * inputFactor;
    bool vdintlv = std::is_same_v<SourceOp, VMIVdintlvOp> && inputFactor > 0 &&
                   matchingInputs && matchingOutputs &&
                   inputFactor == 2 * outputFactor;
    if (!vintlv && !vdintlv) {
      return std::nullopt;
    }
    return InterleaveLoweringPlan{InterleaveLoweringKind::ZeroCopy,
                                  inputFactor, outputFactor, vintlv};
  }

  FailureOr<InterleaveLoweringPlan> classifyInterleaveLowering(
      SourceOp op, const VMIInterleaveLayoutFact &fact,
      OneToNPatternRewriter &rewriter) const {
    if (hasLaneStrideInterleaveLayout(fact)) {
      return InterleaveLoweringPlan{InterleaveLoweringKind::LaneStride};
    }
    if (hasUnitStrideContiguousInterleaveLayout(fact)) {
      return InterleaveLoweringPlan{InterleaveLoweringKind::Contiguous};
    }
    std::optional<InterleaveLoweringPlan> zeroCopyPlan =
        getZeroCopyInterleavePlan(fact);
    if (!zeroCopyPlan) {
      return rewriter.notifyMatchFailure(op, "unsupported interleave physical layout relation");
    }
    return *zeroCopyPlan;
  }

  LogicalResult lowerInterleaveByLayout(
      SourceOp op, OneToNPatternRewriter &rewriter, ValueRange lhsParts,
      ValueRange rhsParts, ValueRange maskParts, ArrayRef<Type> lowTypes,
      ArrayRef<Type> highTypes, Type elementType,
      const VMIInterleaveLayoutFact &fact) const {
    FailureOr<InterleaveLoweringPlan> plan =
        classifyInterleaveLowering(op, fact, rewriter);
    if (failed(plan)) {
      return failure();
    }
    if (plan->kind == InterleaveLoweringKind::LaneStride) {
      return lowerLaneStrideInterleave(op, rewriter, lhsParts, rhsParts,
                                       lowTypes, highTypes, elementType, fact);
    }
    if (plan->kind == InterleaveLoweringKind::Contiguous) {
      return lowerContiguous(op, rewriter, lhsParts, rhsParts, maskParts,
                             lowTypes, highTypes);
    }
    FailureOr<SmallVector<Value>> results = materializeZeroCopyResults(
        op, lhsParts, rhsParts, lowTypes, highTypes, plan->inputFactor,
        plan->outputFactor, plan->zeroCopyVintlv, rewriter);
    if (failed(results)) {
      return failure();
    }
    replaceOpWithFlatConvertedValues(rewriter, op, *results,
                                     *this->getTypeConverter());
    return success();
  }

  LogicalResult lowerLaneStrideInterleave(
      SourceOp op, OneToNPatternRewriter &rewriter, ValueRange lhsParts,
      ValueRange rhsParts, TypeRange lowTypes, TypeRange highTypes,
      Type elementType, const VMIInterleaveLayoutFact &fact) const {
    bool invalidArity = lhsParts.size() != 1 || rhsParts.size() != 1 ||
                        lowTypes.size() != 1 || highTypes.size() != 1;
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, "lane-stride interleave expects one physical carrier part");
    }
    unsigned elementBits =
        pto::getPTOStorageElemBitWidth(elementType);
    int64_t laneStride = fact.lhsLayout.getLaneStride();
    int64_t carrierBits = static_cast<int64_t>(elementBits) * laneStride;
    bool invalidCarrier = elementBits == 0 || laneStride <= 1 ||
                          carrierBits <= 0 || carrierBits > 32;
    if (invalidCarrier) {
      return rewriter.notifyMatchFailure(
          op, "invalid lane-stride interleave carrier width");
    }
    FailureOr<std::pair<Value, Value>> pair =
        materializeLaneStrideInterleavePair(
            op, rewriter, lhsParts.front(), rhsParts.front(), lowTypes.front(),
            highTypes.front(), carrierBits);
    if (failed(pair)) {
      return failure();
    }
    SmallVector<Value, 2> results = {pair->first, pair->second};
    return lowerBinaryPhysicalResults(op, results, rewriter,
                                      *this->getTypeConverter());
  }

  void appendVintlvZeroCopyResults(SmallVectorImpl<Value> &results,
                                   ValueRange lhsParts, ValueRange rhsParts,
                                   int64_t inputFactor) const {
    int64_t safeInputFactor = inputFactor > 0 ? inputFactor : 1;
    size_t groupChunks = lhsParts.size() / safeInputFactor;
    size_t halfGroupChunks = groupChunks / 2;
    for (int64_t group = 0; group < inputFactor; ++group) {
      size_t offset = group * groupChunks;
      llvm::append_range(results, lhsParts.slice(offset, halfGroupChunks));
      llvm::append_range(results, rhsParts.slice(offset, halfGroupChunks));
    }
    for (int64_t group = 0; group < inputFactor; ++group) {
      size_t offset = group * groupChunks + halfGroupChunks;
      llvm::append_range(results, lhsParts.slice(offset, halfGroupChunks));
      llvm::append_range(results, rhsParts.slice(offset, halfGroupChunks));
    }
  }

  void appendVdintlvZeroCopyResults(SmallVectorImpl<Value> &results,
                                    ValueRange lhsParts, ValueRange rhsParts,
                                    int64_t inputFactor,
                                    int64_t outputFactor) const {
    int64_t safeInputFactor = inputFactor > 0 ? inputFactor : 1;
    size_t groupChunks = lhsParts.size() / safeInputFactor;
    for (int64_t group = 0; group < outputFactor; ++group) {
      size_t offset = 2 * group * groupChunks;
      llvm::append_range(results, lhsParts.slice(offset, groupChunks));
      llvm::append_range(results, rhsParts.slice(offset, groupChunks));
    }
    for (int64_t group = 0; group < outputFactor; ++group) {
      size_t offset = (2 * group + 1) * groupChunks;
      llvm::append_range(results, lhsParts.slice(offset, groupChunks));
      llvm::append_range(results, rhsParts.slice(offset, groupChunks));
    }
  }

  LogicalResult validateZeroCopyResultParts(
      SourceOp op, ArrayRef<Value> results, TypeRange lowTypes,
      TypeRange highTypes, OneToNPatternRewriter &rewriter) const {
    SmallVector<Type> resultTypes;
    resultTypes.reserve(lowTypes.size() + highTypes.size());
    llvm::append_range(resultTypes, lowTypes);
    llvm::append_range(resultTypes, highTypes);
    bool resultArityMismatch = results.size() != resultTypes.size();
    if (resultArityMismatch) {
      return rewriter.notifyMatchFailure(
          op, "zero-copy interleave result arity mismatch");
    }
    for (auto [value, resultType] : llvm::zip_equal(results, resultTypes)) {
      bool resultTypeMismatch = value.getType() != resultType;
      if (resultTypeMismatch) {
        return rewriter.notifyMatchFailure(
            op, "zero-copy interleave part type mismatch");
      }
    }
    return success();
  }

  FailureOr<SmallVector<Value>> materializeZeroCopyResults(
      SourceOp op, ValueRange lhsParts, ValueRange rhsParts,
      TypeRange lowTypes, TypeRange highTypes, int64_t inputFactor,
      int64_t outputFactor, bool zeroCopyVintlv,
      OneToNPatternRewriter &rewriter) const {
    if (inputFactor <= 0) {
      return rewriter.notifyMatchFailure(
          op, "zero-copy interleave requires positive input factor");
    }
    int64_t safeInputFactor = inputFactor;
    SmallVector<Value> results;
    results.reserve(lhsParts.size() + rhsParts.size());
    if (zeroCopyVintlv) {
      bool invalidGroupCount =
          lhsParts.empty() || lhsParts.size() % (2 * safeInputFactor) != 0;
      if (invalidGroupCount) {
        return rewriter.notifyMatchFailure(
            op, "zero-copy vintlv expects input groups with even chunk count");
      }
      appendVintlvZeroCopyResults(results, lhsParts, rhsParts, inputFactor);
    } else {
      bool invalidGroupCount =
          lhsParts.empty() || lhsParts.size() % safeInputFactor != 0;
      if (invalidGroupCount) {
        return rewriter.notifyMatchFailure(
            op, "zero-copy vdintlv expects complete input layout groups");
      }
      appendVdintlvZeroCopyResults(results, lhsParts, rhsParts, inputFactor,
                                   outputFactor);
    }

    if (failed(validateZeroCopyResultParts(op, results, lowTypes, highTypes,
                                           rewriter))) {
      return failure();
    }
    return results;
  }

  LogicalResult lowerContiguous(
      SourceOp op, OneToNPatternRewriter &rewriter, ValueRange lhsParts,
      ValueRange rhsParts, ValueRange maskParts, ArrayRef<Type> lowTypes,
      ArrayRef<Type> highTypes) const {
    bool singleChunk = lhsParts.size() == 1 && rhsParts.size() == 1 &&
                       lowTypes.size() == 1 && highTypes.size() == 1;
    if (!singleChunk) {
      return rewriter.notifyMatchFailure(
          op, "single-chunk interleave expects one physical part");
    }
    bool invalidMaskPart =
        !maskParts.empty() && !isa<MaskType>(maskParts.front().getType());
    if (invalidMaskPart) {
      return rewriter.notifyMatchFailure(
          op, "single-chunk interleave mask part type mismatch");
    }
    bool invalidTypes =
        !isa<VRegType>(lowTypes.front()) || !isa<VRegType>(highTypes.front()) ||
        lhsParts.front().getType() != lowTypes.front() ||
        rhsParts.front().getType() != lowTypes.front() ||
        highTypes.front() != lowTypes.front();
    if (invalidTypes) {
      return rewriter.notifyMatchFailure(
          op, "single-chunk interleave part type mismatch");
    }
    auto interleave = rewriter.create<TargetOp>(
        op.getLoc(), lowTypes.front(), highTypes.front(), lhsParts.front(),
        rhsParts.front());
    SmallVector<Value, 2> results = {interleave.getLow(), interleave.getHigh()};
    return lowerBinaryPhysicalResults(op, results, rewriter,
                                      *this->getTypeConverter());
  }

public:

  LogicalResult matchAndRewrite(
      SourceOp op,
      typename OneToNOpConversionPattern<SourceOp>::OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter) const override {
    ValueRange lhsParts = adaptor.getLhs();
    ValueRange rhsParts = adaptor.getRhs();
    ValueRange maskParts = adaptor.getMask();
    FailureOr<std::pair<SmallVector<Type>, SmallVector<Type>>> resultTypes =
        getInterleaveResultTypes(op, lhsParts, rhsParts, maskParts, rewriter);
    if (failed(resultTypes)) {
      return failure();
    }
    SmallVector<Type> lowTypes = std::move(resultTypes->first);
    SmallVector<Type> highTypes = std::move(resultTypes->second);

    auto lhsType = cast<VMIVRegType>(op.getLhs().getType());
    auto rhsType = cast<VMIVRegType>(op.getRhs().getType());
    auto maskType = cast<VMIMaskType>(op.getMask().getType());
    auto lowType = cast<VMIVRegType>(op.getLow().getType());
    auto highType = cast<VMIVRegType>(op.getHigh().getType());
    FailureOr<VMIInterleaveLayoutFact> fact = getInterleaveLayoutFact(
        op, lhsType, rhsType, maskType, lowType, highType, rewriter);
    if (failed(fact)) {
      return failure();
    }

    return lowerInterleaveByLayout(
        op, rewriter, lhsParts, rhsParts, maskParts, lowTypes, highTypes,
        lhsType.getElementType(), *fact);
  }
};

struct OneToNVMIFmaOpPattern : OneToNOpConversionPattern<VMIFmaOp> {
  using OneToNOpConversionPattern<VMIFmaOp>::OneToNOpConversionPattern;

private:
  FailureOr<Value> lowerPart(VMIFmaOp op, Value lhs, Value rhs, Value acc,
                             Type resultType,
                             OneToNPatternRewriter &rewriter) const {
    auto vregType = dyn_cast<VRegType>(resultType);
    const bool invalidPart =
        !vregType || lhs.getType() != resultType || rhs.getType() != resultType ||
        acc.getType() != resultType;
    if (invalidPart) {
      (void)rewriter.notifyMatchFailure(
          op, "fma requires matching physical vreg parts");
      return failure();
    }
    FailureOr<Value> mask =
        createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter);
    if (failed(mask)) {
      (void)rewriter.notifyMatchFailure(op,
                                        "unsupported element type for fma");
      return failure();
    }
    return rewriter
        .create<VmulaOp>(op.getLoc(), resultType, acc, lhs, rhs, *mask)
        .getResult();
  }

public:
  LogicalResult
  matchAndRewrite(VMIFmaOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange lhsParts = adaptor.getLhs();
    ValueRange rhsParts = adaptor.getRhs();
    ValueRange accParts = adaptor.getAcc();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypesOrFailure(op, *this->getTypeConverter());
    if (failed(maybe_resultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    const bool invalidArity =
        lhsParts.size() != rhsParts.size() ||
        lhsParts.size() != accParts.size() ||
        lhsParts.size() != resultTypes.size();
    if (invalidArity) {
      return rewriter.notifyMatchFailure(op, "fma physical arity mismatch");
    }

    return lowerPointwisePhysicalParts(
        op, resultTypes, "fma physical arity mismatch", rewriter,
        [&](int64_t index, Type resultType) -> FailureOr<Value> {
          return lowerPart(op, lhsParts[index], rhsParts[index], accParts[index],
                           resultType, rewriter);
        },
        *this->getTypeConverter());
  }
};

struct OneToNVMIVexpdifOpPattern : OneToNOpConversionPattern<VMIVexpdifOp> {
  using OneToNOpConversionPattern<VMIVexpdifOp>::OneToNOpConversionPattern;

private:
  FailureOr<Value> lowerF32Part(VMIVexpdifOp op, Value x, Value max,
                                Value mask, Type resultType,
                                OneToNPatternRewriter &rewriter) const {
    auto vregType = dyn_cast<VRegType>(resultType);
    auto maskType = dyn_cast<MaskType>(mask.getType());
    const bool invalidPart =
        !vregType || !maskType || !vregType.getElementType().isF32() ||
        x.getType() != resultType || max.getType() != resultType ||
        maskType.getGranularity() != "b32";
    if (invalidPart) {
      (void)rewriter.notifyMatchFailure(
          op, "f32 vexpdif requires matching f32 parts and b32 masks");
      return failure();
    }
    return rewriter
        .create<VexpdifOp>(op.getLoc(), resultType, x, max, mask,
                           rewriter.getStringAttr("ODD"))
        .getResult();
  }

  FailureOr<Value> lowerF16Part(VMIVexpdifOp op, Value x, Value max,
                                Value mask, Type resultType, StringRef part,
                                OneToNPatternRewriter &rewriter) const {
    auto xType = dyn_cast<VRegType>(x.getType());
    auto resultVRegType = dyn_cast<VRegType>(resultType);
    auto maskType = dyn_cast<MaskType>(mask.getType());
    const bool invalidPart =
        !xType || !resultVRegType || !maskType ||
        !xType.getElementType().isF16() ||
        !resultVRegType.getElementType().isF32() ||
        max.getType() != x.getType() || maskType.getGranularity() != "b16";
    if (invalidPart) {
      (void)rewriter.notifyMatchFailure(
          op, "f16 vexpdif requires matching f16 parts and b16 masks");
      return failure();
    }
    return rewriter
        .create<VexpdifOp>(op.getLoc(), resultType, x, max, mask,
                           rewriter.getStringAttr(part))
        .getResult();
  }

  LogicalResult lowerF32(VMIVexpdifOp op, ValueRange xParts,
                         ValueRange maxParts, ValueRange maskParts,
                         ArrayRef<Type> resultTypes,
                         OneToNPatternRewriter &rewriter) const {
    const bool invalidArity = xParts.size() != resultTypes.size();
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, "f32 vexpdif requires one result per source part");
    }

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [x, max, mask, resultType] :
         llvm::zip_equal(xParts, maxParts, maskParts, resultTypes)) {
      FailureOr<Value> result =
          lowerF32Part(op, x, max, mask, resultType, rewriter);
      if (failed(result)) {
        return failure();
      }
      results.push_back(*result);
    }
    return lowerBinaryPhysicalResults(op, results, rewriter,
                                      *this->getTypeConverter());
  }

  LogicalResult lowerF16(VMIVexpdifOp op, ValueRange xParts,
                         ValueRange maxParts, ValueRange maskParts,
                         ArrayRef<Type> resultTypes,
                         OneToNPatternRewriter &rewriter) const {
    const bool invalidArity = resultTypes.size() != 2 * xParts.size();
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, "f16 vexpdif requires EVEN/ODD f32 result parts");
    }

    static constexpr StringRef kParts[] = {"EVEN", "ODD"};
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [partIndex, part] : llvm::enumerate(kParts)) {
      for (auto [chunkIndex, x] : llvm::enumerate(xParts)) {
        Value max = maxParts[chunkIndex];
        Value mask = maskParts[chunkIndex];
        Type resultType = resultTypes[partIndex * xParts.size() + chunkIndex];
        FailureOr<Value> result =
            lowerF16Part(op, x, max, mask, resultType, part, rewriter);
        if (failed(result)) {
          return failure();
        }
        results.push_back(*result);
      }
    }
    return replacePhysicalResults(rewriter, op, results,
                                  *this->getTypeConverter());
  }

  LogicalResult lowerBySourceElementType(
      VMIVexpdifOp op, ValueRange xParts, ValueRange maxParts,
      ValueRange maskParts, ArrayRef<Type> resultTypes,
      Type sourceElementType, OneToNPatternRewriter &rewriter) const {
    if (sourceElementType.isF32()) {
      return lowerF32(op, xParts, maxParts, maskParts, resultTypes, rewriter);
    }
    if (!sourceElementType.isF16()) {
      return rewriter.notifyMatchFailure(
          op, "f16 vexpdif requires EVEN/ODD f32 result parts");
    }
    return lowerF16(op, xParts, maxParts, maskParts, resultTypes, rewriter);
  }

public:
  LogicalResult
  matchAndRewrite(VMIVexpdifOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    const bool requiresPassthru =
        op.getPmode().has_value() && *op.getPmode() == "merge";
    if (requiresPassthru) {
      return rewriter.notifyMatchFailure(
          op, "merge predicate mode requires an explicit passthru lowering");
    }

    ValueRange xParts = adaptor.getX();
    ValueRange maxParts = adaptor.getMax();
    ValueRange maskParts = adaptor.getMask();
    FailureOr<SmallVector<Type>> maybeResultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybeResultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybeResultTypes);
    const bool invalidInputArity =
        xParts.size() != maxParts.size() || xParts.size() != maskParts.size();
    if (invalidInputArity) {
      return rewriter.notifyMatchFailure(op, "vexpdif physical arity mismatch");
    }

    auto sourceVMIType = cast<VMIVRegType>(op.getX().getType());
    return lowerBySourceElementType(
        op, xParts, maxParts, maskParts, resultTypes,
        sourceVMIType.getElementType(), rewriter);
  }
};

template <typename SourceOp, typename TargetOp, bool IsMaskResult = false>
struct OneToNVMIUnaryOpPattern : OneToNOpConversionPattern<SourceOp> {
  using OneToNOpConversionPattern<SourceOp>::OneToNOpConversionPattern;

private:
  FailureOr<Value> lowerPart(
      SourceOp op, Value source, Type resultType,
      OneToNPatternRewriter &rewriter) const {
    if constexpr (IsMaskResult) {
      auto maskType = dyn_cast<MaskType>(resultType);
      const bool invalidPart = !maskType || source.getType() != resultType;
      if (invalidPart) {
        return rewriter.notifyMatchFailure(
            op, "physical mask unary part type mismatch");
      }
      FailureOr<Value> seedMask =
          createAllTrueMask(op.getLoc(), maskType, rewriter);
      if (failed(seedMask)) {
        return rewriter.notifyMatchFailure(
            op, "unsupported mask type for all-true mask unary seed");
      }
      return rewriter
          .create<TargetOp>(op.getLoc(), resultType, source, *seedMask)
          .getResult();
    }
    auto vregType = dyn_cast<VRegType>(resultType);
    const bool invalidPart = !vregType || source.getType() != resultType;
    if (invalidPart) {
      return rewriter.notifyMatchFailure(
          op, "physical unary part type mismatch");
    }
    FailureOr<Value> mask =
        createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter);
    if (failed(mask)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported element type for all-true unary mask");
    }
    return rewriter
        .create<TargetOp>(op.getLoc(), resultType, source, *mask)
        .getResult();
  }

  LogicalResult lowerParts(SourceOp op, ValueRange sourceParts,
                           ArrayRef<Type> resultTypes,
                           OneToNPatternRewriter &rewriter) const {
    const bool invalidArity = sourceParts.size() != resultTypes.size();
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, IsMaskResult ? "physical mask unary arity mismatch"
                           : "physical unary arity mismatch");
    }
    return lowerPointwisePhysicalParts(
        op, resultTypes,
        IsMaskResult ? "physical mask unary arity mismatch"
                     : "physical unary arity mismatch",
        rewriter,
        [&](int64_t index, Type resultType) -> FailureOr<Value> {
          return lowerPart(op, sourceParts[index], resultType, rewriter);
        },
        *this->getTypeConverter());
  }

public:
  LogicalResult matchAndRewrite(
      SourceOp op,
      typename OneToNOpConversionPattern<SourceOp>::OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter) const override {
    return lowerWithConvertedResultTypes(
        op, 0, *this->getTypeConverter(), [&](ArrayRef<Type> resultTypes) {
          return lowerParts(op, adaptor.getSource(), resultTypes, rewriter);
        });
  }
};


template <typename SourceOp>
struct OneToNVMICmpOpPattern : OneToNOpConversionPattern<SourceOp> {
  using OneToNOpConversionPattern<SourceOp>::OneToNOpConversionPattern;

private:
  FailureOr<Value> lowerPart(
      SourceOp op, Value lhs, Value rhs, Type resultType,
      const VPTOCmpMode &cmpMode,
      OneToNPatternRewriter &rewriter) const {
    auto maskType = dyn_cast<MaskType>(resultType);
    auto lhsType = dyn_cast<VRegType>(lhs.getType());
    const bool invalidPart =
        !maskType || lhs.getType() != rhs.getType() || !lhsType;
    if (invalidPart) {
      (void)rewriter.notifyMatchFailure(op,
                                        "physical cmp part type mismatch");
      return failure();
    }
    FailureOr<Value> seedMask =
        createAllTrueMask(op.getLoc(), maskType, rewriter);
    if (failed(seedMask)) {
      (void)rewriter.notifyMatchFailure(
          op, "unsupported mask type for all-true cmp seed");
      return failure();
    }
    if (cmpMode.signedness) {
      FailureOr<VRegType> carrierType =
          getSignednessCarrierVRegType(lhsType, *cmpMode.signedness);
      if (failed(carrierType)) {
        (void)rewriter.notifyMatchFailure(
            op, "unsupported integer compare signedness carrier");
        return failure();
      }
      FailureOr<Value> carrierLhs =
          bitcastVReg(op.getLoc(), lhs, *carrierType, rewriter);
      FailureOr<Value> carrierRhs =
          bitcastVReg(op.getLoc(), rhs, *carrierType, rewriter);
      const bool failedCarriers = failed(carrierLhs) || failed(carrierRhs);
      if (failedCarriers) {
        (void)rewriter.notifyMatchFailure(
            op, "failed to materialize integer compare signedness carrier");
        return failure();
      }
      lhs = *carrierLhs;
      rhs = *carrierRhs;
    }
    return rewriter
        .create<VcmpOp>(op.getLoc(), resultType, lhs, rhs, *seedMask,
                        rewriter.getStringAttr(cmpMode.mode))
        .getResult();
  }

public:
  LogicalResult matchAndRewrite(
      SourceOp op,
      typename OneToNOpConversionPattern<SourceOp>::OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter) const override {
    std::optional<VPTOCmpMode> cmpMode =
        getVPTOCmpMode<SourceOp>(op.getPredicate());
    if (!cmpMode) {
      return op.emitOpError()
             << kVMIDiagUnsupportedPrefix << "compare predicate "
             << op.getPredicate()
             << " cannot be lowered to pto.vcmp; supported predicates are "
             << getSupportedComparePredicateMessage<SourceOp>();
    }

    ValueRange lhsParts = adaptor.getLhs();
    ValueRange rhsParts = adaptor.getRhs();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    const bool invalidArity = lhsParts.size() != rhsParts.size() ||
                              lhsParts.size() != resultTypes.size();
    if (invalidArity) {
      return rewriter.notifyMatchFailure(op, "physical cmp arity mismatch");
    }

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [lhs, rhs, resultType] :
         llvm::zip_equal(lhsParts, rhsParts, resultTypes)) {
      FailureOr<Value> result =
          lowerPart(op, lhs, rhs, resultType, *cmpMode, rewriter);
      if (failed(result)) {
        return failure();
      }
      results.push_back(*result);
    }

    replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
    return success();
  }
};

struct OneToNVMISelectOpPattern : OneToNOpConversionPattern<VMISelectOp> {
  using OneToNOpConversionPattern<VMISelectOp>::OneToNOpConversionPattern;

private:
  FailureOr<Value> lowerPart(VMISelectOp op, Value mask, Value trueValue,
                             Value falseValue, Type resultType,
                             OneToNPatternRewriter &rewriter) const {
    const bool invalidPart =
        !isa<MaskType>(mask.getType()) || trueValue.getType() != resultType ||
        falseValue.getType() != resultType || !isa<VRegType>(resultType);
    if (invalidPart) {
      (void)rewriter.notifyMatchFailure(
          op, "physical select part type mismatch");
      return failure();
    }
    return rewriter
        .create<VselOp>(op.getLoc(), resultType, trueValue, falseValue, mask)
        .getResult();
  }

public:
  LogicalResult
  matchAndRewrite(VMISelectOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange maskParts = adaptor.getMask();
    ValueRange trueParts = adaptor.getTrueValue();
    ValueRange falseParts = adaptor.getFalseValue();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    const bool invalidArity =
        maskParts.size() != trueParts.size() ||
        trueParts.size() != falseParts.size() ||
        trueParts.size() != resultTypes.size();
    if (invalidArity) {
      return rewriter.notifyMatchFailure(op, "physical select arity mismatch");
    }

    return lowerPointwisePhysicalParts(
        op, resultTypes, "physical select arity mismatch", rewriter,
        [&](int64_t index, Type resultType) -> FailureOr<Value> {
          return lowerPart(op, maskParts[index], trueParts[index],
                           falseParts[index], resultType, rewriter);
        },
        *this->getTypeConverter());
  }
};

struct OneToNVMIVselrOpPattern : OneToNOpConversionPattern<VMIVselrOp> {
  using OneToNOpConversionPattern<VMIVselrOp>::OneToNOpConversionPattern;

private:
  FailureOr<Value> lowerPart(VMIVselrOp op, Value source, Value index,
                             Type resultType,
                             OneToNPatternRewriter &rewriter) const {
    auto sourceType = dyn_cast<VRegType>(source.getType());
    auto indexType = dyn_cast<VRegType>(index.getType());
    auto resultVRegType = dyn_cast<VRegType>(resultType);
    const bool invalidPart =
        !sourceType || !indexType || !resultVRegType ||
        sourceType != resultVRegType ||
        sourceType.getElementCount() != indexType.getElementCount() ||
        pto::getPTOStorageElemBitWidth(sourceType.getElementType()) !=
            pto::getPTOStorageElemBitWidth(indexType.getElementType());
    if (invalidPart) {
      (void)rewriter.notifyMatchFailure(
          op, "vselr physical source/index/result type mismatch");
      return failure();
    }
    return rewriter
        .create<VselrOp>(op.getLoc(), resultVRegType, source, index)
        .getResult();
  }

public:
  LogicalResult
  matchAndRewrite(VMIVselrOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    ValueRange indexParts = adaptor.getIndex();
    FailureOr<SmallVector<Type>> maybeResultTypes =
        getConvertedResultTypesOrFailure(op, *this->getTypeConverter());
    if (failed(maybeResultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybeResultTypes);
    const bool invalidArity = sourceParts.size() != 1 ||
                              indexParts.size() != 1 || resultTypes.size() != 1;
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, "vselr supports only one physical source/index/result part");
    }
    FailureOr<Value> result = lowerPart(op, sourceParts.front(),
                                        indexParts.front(), resultTypes.front(),
                                        rewriter);
    if (failed(result)) {
      return failure();
    }
    return replaceSinglePhysicalResult(rewriter, op, *result,
                                       *this->getTypeConverter());
  }
};

struct OneToNVMIActivePrefixIndexOpPattern
    : OneToNOpConversionPattern<VMIActivePrefixIndexOp> {
  using OneToNOpConversionPattern<
      VMIActivePrefixIndexOp>::OneToNOpConversionPattern;

private:
  FailureOr<Value> lowerPart(VMIActivePrefixIndexOp op, Value mask,
                             Type resultType,
                             OneToNPatternRewriter &rewriter) const {
    auto vregType = dyn_cast<VRegType>(resultType);
    auto maskType = dyn_cast<MaskType>(mask.getType());
    const bool invalidPartTypes = !vregType || !maskType;
    if (invalidPartTypes) {
      (void)rewriter.notifyMatchFailure(
          op, "active_prefix_index requires physical vreg/mask parts");
      return failure();
    }
    auto intType = dyn_cast<IntegerType>(vregType.getElementType());
    const bool invalidElementType = !intType || !intType.isSignless();
    if (invalidElementType) {
      (void)rewriter.notifyMatchFailure(
          op, "active_prefix_index requires signless integer result part");
      return failure();
    }
    FailureOr<Value> seedMask =
        createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter);
    if (failed(seedMask)) {
      (void)rewriter.notifyMatchFailure(
          op, "unsupported element type for active_prefix_index seed mask");
      return failure();
    }
    Value zero = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0,
                                                       intType.getWidth());
    Value carrier =
        rewriter
            .create<VdupOp>(op.getLoc(), resultType, zero, *seedMask,
                            /*position=*/nullptr)
            .getResult();
    return rewriter
        .create<VusqzOp>(op.getLoc(), resultType, carrier, mask)
        .getResult();
  }

public:
  LogicalResult
  matchAndRewrite(VMIActivePrefixIndexOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange maskParts = adaptor.getMask();
    FailureOr<SmallVector<Type>> maybeResultTypes =
        getConvertedResultTypesOrFailure(op, *this->getTypeConverter());
    if (failed(maybeResultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybeResultTypes);
    const bool invalidArity = maskParts.size() != 1 || resultTypes.size() != 1;
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, "active_prefix_index supports only one physical part");
    }
    FailureOr<Value> result =
        lowerPart(op, maskParts.front(), resultTypes.front(), rewriter);
    if (failed(result)) {
      return failure();
    }
    return replaceSinglePhysicalResult(rewriter, op, *result,
                                       *this->getTypeConverter());
  }
};

struct OneToNVMICompressOpPattern : OneToNOpConversionPattern<VMICompressOp> {
  using OneToNOpConversionPattern<VMICompressOp>::OneToNOpConversionPattern;

private:
  FailureOr<Value> lowerPart(VMICompressOp op, Value source, Value mask,
                             Type resultType,
                             OneToNPatternRewriter &rewriter) const {
    auto resultVRegType = dyn_cast<VRegType>(resultType);
    const bool invalidPart =
        !resultVRegType || source.getType() != resultType ||
        !isa<MaskType>(mask.getType());
    if (invalidPart) {
      (void)rewriter.notifyMatchFailure(
          op, "compress requires physical source/mask/result parts");
      return failure();
    }
    return rewriter
        .create<VsqzOp>(op.getLoc(), resultVRegType, source, mask)
        .getResult();
  }

public:
  LogicalResult
  matchAndRewrite(VMICompressOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    ValueRange maskParts = adaptor.getMask();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypesOrFailure(op, *this->getTypeConverter());
    if (failed(maybe_resultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    const bool invalidArity = sourceParts.size() != 1 ||
                              maskParts.size() != 1 || resultTypes.size() != 1;
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, "compress supports only one physical part");
    }

    FailureOr<Value> result = lowerPart(op, sourceParts.front(),
                                        maskParts.front(), resultTypes.front(),
                                        rewriter);
    if (failed(result)) {
      return failure();
    }
    replaceOpWithFlatConvertedValues(rewriter, op, SmallVector<Value>{*result},
                                     *this->getTypeConverter());
    return success();
  }
};

struct OneToNVMICompressStoreOpPattern
    : OneToNOpConversionPattern<VMICompressStoreOp> {
  using OneToNOpConversionPattern<
      VMICompressStoreOp>::OneToNOpConversionPattern;

private:
  LogicalResult lowerStore(VMICompressStoreOp op, Value destination,
                           Value offset, Value value, Value mask,
                           OneToNPatternRewriter &rewriter) const {
    auto valueType = dyn_cast<VRegType>(value.getType());
    auto destinationType = dyn_cast<PtrType>(destination.getType());
    const bool invalidTypes =
        !valueType || !isa<MaskType>(mask.getType()) || !destinationType;
    if (invalidTypes) {
      return rewriter.notifyMatchFailure(
          op, "compress_store requires physical value/mask and ptr "
              "destination");
    }
    Value storeBase =
        rewriter
            .create<AddPtrOp>(op.getLoc(), destination.getType(), destination,
                              offset)
            .getResult();
    Value squeezed =
        rewriter.create<VsqzOp>(op.getLoc(), valueType, value, mask).getResult();
    auto align = rewriter.create<InitAlignOp>(
        op.getLoc(), AlignType::get(rewriter.getContext()));
    auto store = rewriter.create<VsturOp>(
        op.getLoc(), align.getResult().getType(), align.getResult(), squeezed,
        storeBase, rewriter.getStringAttr("POST_UPDATE"));
    rewriter.create<VstarOp>(op.getLoc(), store.getAlignOut(), storeBase);
    rewriter.eraseOp(op);
    return success();
  }

public:
  LogicalResult
  matchAndRewrite(VMICompressStoreOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    FailureOr<Value> destination = getSingleValue(
        op, adaptor.getDestination(),
        "compress_store destination must convert to one value", rewriter);
    FailureOr<Value> offset = getSingleValue(
        op, adaptor.getOffset(),
        "compress_store offset must convert to one value", rewriter);
    const bool failedAddress = failed(destination) || failed(offset);
    if (failedAddress) {
      return failure();
    }

    ValueRange valueParts = adaptor.getValue();
    ValueRange maskParts = adaptor.getMask();
    const bool invalidArity = valueParts.size() != 1 || maskParts.size() != 1;
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, "compress_store supports only one physical part");
    }

    return lowerStore(op, *destination, *offset, valueParts.front(),
                      maskParts.front(), rewriter);
  }
};

struct ReduceAddPhysicalPlan {
  VRegType resultType;
  MaskType maskType;
};

template <typename OpTy>
static FailureOr<ReduceAddPhysicalPlan> buildReduceAddPhysicalPlan(
    OpTy op, ValueRange sourceParts, ValueRange maskParts,
    TypeRange resultTypes, OneToNPatternRewriter &rewriter,
    StringRef diagnostic) {
  bool invalidArity = sourceParts.empty() || sourceParts.size() != maskParts.size() ||
                      resultTypes.size() != 1;
  if (invalidArity) {
    return rewriter.notifyMatchFailure(
        op, Twine(diagnostic) +
                " requires matching source/mask chunks and one result chunk");
  }
  auto resultType = dyn_cast<VRegType>(resultTypes.front());
  auto maskType = dyn_cast<MaskType>(maskParts.front().getType());
  if (!resultType || !maskType) {
    return rewriter.notifyMatchFailure(
        op, Twine(diagnostic) +
                " requires matching physical source/result vregs and one mask");
  }
  for (Value sourcePart : sourceParts) {
    bool sourceTypeMismatch = sourcePart.getType() != resultType;
    if (sourceTypeMismatch) {
      return rewriter.notifyMatchFailure(
          op, Twine(diagnostic) +
                  " requires every source chunk to match result vreg type");
    }
  }
  for (Value maskPart : maskParts) {
    bool maskTypeMismatch = maskPart.getType() != maskType;
    if (maskTypeMismatch) {
      return rewriter.notifyMatchFailure(
          op, Twine(diagnostic) +
                  " requires every mask chunk to have the same predicate type");
    }
  }
  return ReduceAddPhysicalPlan{resultType, maskType};
}

template <typename ReduceOp>
static LogicalResult lowerReduceAddParts(
    ReduceOp op, ValueRange sourceParts, ValueRange maskParts,
    VRegType resultType, MaskType maskType, StringRef firstLaneDiagnostic,
    OneToNPatternRewriter &rewriter, TypeConverter &typeConverter) {
  FailureOr<Value> combined = combineEquivalentMaskedParts<VaddOp>(
      op.getLoc(), sourceParts, maskParts, resultType, rewriter);
  if (succeeded(combined)) {
    Value reduced = rewriter
                        .create<VcaddOp>(op.getLoc(), resultType, *combined,
                                         maskParts.front())
                        .getResult();
    replaceOpWithFlatConvertedValues(rewriter, op, SmallVector<Value>{reduced},
                                     typeConverter);
    return success();
  }

  Value accumulator = rewriter
                          .create<VcaddOp>(op.getLoc(), resultType,
                                           sourceParts.front(),
                                           maskParts.front())
                          .getResult();
  const bool singlePart = sourceParts.size() == 1;
  if (singlePart) {
    replaceOpWithFlatConvertedValues(rewriter, op,
                                     SmallVector<Value>{accumulator},
                                     typeConverter);
    return success();
  }
  FailureOr<Value> firstLaneMask =
      createPrefixMask(op.getLoc(), maskType, "PAT_VL1", rewriter);
  if (failed(firstLaneMask)) {
    return rewriter.notifyMatchFailure(op, firstLaneDiagnostic);
  }
  for (size_t part = 1; part < sourceParts.size(); ++part) {
    Value reduced = rewriter
                        .create<VcaddOp>(op.getLoc(), resultType,
                                         sourceParts[part], maskParts[part])
                        .getResult();
    accumulator = rewriter
                      .create<VaddOp>(op.getLoc(), resultType, reduced,
                                      accumulator, *firstLaneMask)
                      .getResult();
  }
  replaceOpWithFlatConvertedValues(rewriter, op,
                                   SmallVector<Value>{accumulator},
                                   typeConverter);
  return success();
}

struct OneToNVMIReduceAddIOpPattern
    : OneToNOpConversionPattern<VMIReduceAddIOp> {
  using OneToNOpConversionPattern<VMIReduceAddIOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIReduceAddIOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    ValueRange maskParts = adaptor.getMask();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypesOrFailure(op, *this->getTypeConverter());
    if (failed(maybe_resultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    FailureOr<ReduceAddPhysicalPlan> plan = buildReduceAddPhysicalPlan(
        op, sourceParts, maskParts, resultTypes, rewriter, "reduce_addi");
    if (failed(plan)) {
      return failure();
    }
    return lowerReduceAddParts(
        op, sourceParts, maskParts, plan->resultType, plan->maskType,
        "failed to create reduce_addi first-lane mask", rewriter,
        *this->getTypeConverter());
  }
};

struct OneToNVMIReduceAddFOpPattern
    : OneToNOpConversionPattern<VMIReduceAddFOp> {
  using OneToNOpConversionPattern<VMIReduceAddFOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIReduceAddFOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    ValueRange maskParts = adaptor.getMask();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    FailureOr<ReduceAddPhysicalPlan> plan = buildReduceAddPhysicalPlan(
        op, sourceParts, maskParts, resultTypes, rewriter, "reduce_addf");
    if (failed(plan)) {
      return failure();
    }
    return lowerReduceAddParts(
        op, sourceParts, maskParts, plan->resultType, plan->maskType,
        "failed to create reduce_addf first-lane mask", rewriter,
        *this->getTypeConverter());
  }
};

enum class GroupReduceLoweringPlan {
  OneBlockVcgadd,
  TwoBlockDeinterleaved2VcgaddVadd,
  FourBlockDeinterleaved4VcgaddTree,
  FullDeinterleaved2VcaddRows,
  ContiguousVcaddRows,
};

FailureOr<GroupReduceLoweringPlan>
classifyGroupReduceLoweringPlan(VMIVRegType sourceType, VMIMaskType maskType,
                                VMIVRegType resultType, int64_t numGroups,
                                std::string *reason = nullptr) {
  VMILayoutSupport supports;
  FailureOr<VMIGroupReduceLayoutFact> fact =
      supports.getGroupReduceLayoutFactForLayouts(
          sourceType, maskType, resultType, numGroups, reason);
  if (failed(fact)) {
    return failure();
  }

  switch (fact->blockClass) {
  case VMIGroupBlockClass::QuarterBlock:
  case VMIGroupBlockClass::HalfBlock:
  case VMIGroupBlockClass::OneBlock:
    return GroupReduceLoweringPlan::OneBlockVcgadd;
  case VMIGroupBlockClass::TwoBlock:
    return GroupReduceLoweringPlan::TwoBlockDeinterleaved2VcgaddVadd;
  case VMIGroupBlockClass::FourBlock:
    return GroupReduceLoweringPlan::FourBlockDeinterleaved4VcgaddTree;
  case VMIGroupBlockClass::FullPartMultiple:
    bool deinterleavedSource =
        fact->sourceLayout && fact->sourceLayout.isDeinterleaved() &&
        fact->sourceLayout.getFactor() == 2;
    if (deinterleavedSource) {
      return GroupReduceLoweringPlan::FullDeinterleaved2VcaddRows;
    }
    return GroupReduceLoweringPlan::ContiguousVcaddRows;
  }
  llvm_unreachable("unknown group block class");
}



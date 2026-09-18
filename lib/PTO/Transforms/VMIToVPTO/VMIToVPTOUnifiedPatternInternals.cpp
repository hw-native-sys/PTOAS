// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#pragma once
//===- VMIToVPTOUnifiedPatternInternals.inc - unified mask-preserving patterns -*- C++ -*-===//
//===----------------------------------------------------------------------===//

template <typename Adaptor, typename = void>
struct UnifiedMaskedHasMask : std::false_type {};
template <typename Adaptor>
struct UnifiedMaskedHasMask<
    Adaptor, std::void_t<decltype(std::declval<Adaptor &>().getMask())>>
    : std::true_type {};

template <typename Adaptor, typename = void>
struct UnifiedMaskedHasSource : std::false_type {};
template <typename Adaptor>
struct UnifiedMaskedHasSource<
    Adaptor, std::void_t<decltype(std::declval<Adaptor &>().getSource())>>
    : std::true_type {};

template <typename Adaptor, typename = void>
struct UnifiedMaskedHasLhsRhs : std::false_type {};
template <typename Adaptor>
struct UnifiedMaskedHasLhsRhs<
    Adaptor, std::void_t<decltype(std::declval<Adaptor &>().getLhs()),
                         decltype(std::declval<Adaptor &>().getRhs())>>
    : std::true_type {};

/// True for ops whose predicate is a mandatory physical operand group instead of
/// an optional variadic mask.
template <typename SourceOp>
inline constexpr bool kUnifiedMaskedHasMandatoryMask =
    std::is_same_v<SourceOp, VMIVaxpyOp> ||
    std::is_same_v<SourceOp, VMIVlreluOp> ||
    std::is_same_v<SourceOp, VMIVpreluOp>;

/// Resolves the predicate that belongs to one physical result part, deriving an
/// all-active predicate when the unified op carries no explicit mask.
FailureOr<Value> getUnifiedPartMask(Operation *op, ValueRange maskParts,
                                    size_t index, Type resultType,
                                    const TypeConverter &typeConverter,
                                    PatternRewriter &rewriter) {
  auto dataType = dyn_cast<VRegType>(resultType);
  if (!dataType) {
    return failure();
  }
  return getUnifiedMaskPart(op, maskParts, index, dataType, typeConverter,
                            rewriter);
}

/// Lowers one unified mask-preserving op. Every operand group and the predicate
/// carry one physical part per result part; `create` then emits the target VPTO
/// op for a single part.
template <typename CreateFn>
LogicalResult lowerUnifiedMaskedOp(
    Operation *op, ArrayRef<ValueRange> operandGroups, ValueRange maskParts,
    TypeRange resultTypes, TypeConverter &typeConverter,
    OneToNPatternRewriter &rewriter, StringRef detail, CreateFn create) {
  size_t partCount = resultTypes.size();
  for (ValueRange parts : operandGroups) {
    if (parts.size() != partCount) {
      return rewriter.notifyMatchFailure(
          op, (detail + " physical arity mismatch").str());
    }
  }
  if (!maskParts.empty() && maskParts.size() != partCount) {
    return rewriter.notifyMatchFailure(
        op, (detail + " predicate arity mismatch").str());
  }

  SmallVector<Value> results;
  results.reserve(partCount);
  for (size_t index = 0; index < partCount; ++index) {
    Type resultType = resultTypes[index];
    SmallVector<Value> operands;
    operands.reserve(operandGroups.size());
    for (ValueRange parts : operandGroups) {
      if (parts[index].getType() != resultType) {
        return rewriter.notifyMatchFailure(
            op, (detail + " physical type mismatch").str());
      }
      operands.push_back(parts[index]);
    }
    FailureOr<Value> mask = getUnifiedPartMask(
        op, maskParts, index, resultType, typeConverter, rewriter);
    if (failed(mask)) {
      return rewriter.notifyMatchFailure(
          op, (detail + " data and predicate types do not match").str());
    }
    FailureOr<Value> result = create(op->getLoc(), resultType, operands, *mask);
    if (failed(result)) {
      return failure();
    }
    results.push_back(*result);
  }
  replaceOpWithFlatConvertedValues(rewriter, op, results, typeConverter);
  return success();
}

/// Collects the physical operand groups, the predicate parts and the scalar
/// operands of one unified op.
template <typename SourceOp, typename Adaptor>
LogicalResult collectUnifiedMaskedOperands(
    SourceOp op, Adaptor adaptor, OneToNPatternRewriter &rewriter,
    SmallVectorImpl<ValueRange> &operandGroups,
    SmallVectorImpl<Value> &maskParts, SmallVectorImpl<Value> &scalars) {
  if constexpr (std::is_same_v<SourceOp, VMIVaxpyOp>) {
    FailureOr<Value> alpha =
        getSingleValue(op, adaptor.getAlpha(),
                       "vaxpy alpha must convert to one value", rewriter);
    if (failed(alpha)) {
      return failure();
    }
    scalars.push_back(*alpha);
    operandGroups.push_back(adaptor.getX());
    operandGroups.push_back(adaptor.getAcc());
  } else if constexpr (std::is_same_v<SourceOp, VMIVlreluOp>) {
    FailureOr<Value> slope =
        getSingleValue(op, adaptor.getSlope(),
                       "vlrelu slope must convert to one value", rewriter);
    if (failed(slope)) {
      return failure();
    }
    scalars.push_back(*slope);
    operandGroups.push_back(adaptor.getX());
  } else if constexpr (std::is_same_v<SourceOp, VMIVpreluOp>) {
    operandGroups.push_back(adaptor.getX());
    operandGroups.push_back(adaptor.getAlpha());
  } else if constexpr (std::is_same_v<SourceOp, VMIVmulaOp>) {
    operandGroups.push_back(adaptor.getAcc());
    operandGroups.push_back(adaptor.getLhs());
    operandGroups.push_back(adaptor.getRhs());
  } else if constexpr (UnifiedMaskedHasLhsRhs<Adaptor>::value) {
    operandGroups.push_back(adaptor.getLhs());
    operandGroups.push_back(adaptor.getRhs());
  } else if constexpr (UnifiedMaskedHasSource<Adaptor>::value) {
    operandGroups.push_back(adaptor.getSource());
  } else {
    return failure();
  }

  if constexpr (UnifiedMaskedHasMask<Adaptor>::value) {
    if constexpr (kUnifiedMaskedHasMandatoryMask<SourceOp>) {
      ValueRange mask = adaptor.getMask();
      maskParts.append(mask.begin(), mask.end());
    } else {
      maskParts = flattenOneToNOperands(adaptor.getMask());
    }
  }
  return success();
}

/// Sign-bit mask of the bf16 integer abs form: clearing the sign bit on the
/// reinterpreted bf16 word computes the absolute value.
constexpr uint16_t kBF16AbsSignMask = 0x7FFF;

/// Lowers one bf16 `vabs` part through `vbitcast`/`vand`, because the hardware
/// abs form does not cover bf16 values.
FailureOr<Value> lowerBF16Vabs(Location loc, Value source, VRegType vregType,
                               Value mask, Type resultType,
                               PatternRewriter &rewriter) {
  auto i16Type = rewriter.getIntegerType(mlir::pto::kValue16);
  auto integerVRegType = VRegType::get(rewriter.getContext(),
                                       vregType.getElementCount(), i16Type);
  FailureOr<MaskType> nativeMaskType =
      getMaskTypeForVReg(integerVRegType, rewriter.getContext());
  if (failed(nativeMaskType)) {
    return failure();
  }
  FailureOr<Value> nativeMask =
      createAllTrueMask(loc, *nativeMaskType, rewriter);
  if (failed(nativeMask)) {
    return failure();
  }
  Value integerSource =
      rewriter.create<VbitcastOp>(loc, integerVRegType, source);
  Value signMaskScalar = rewriter.create<arith::ConstantOp>(
      loc, i16Type, rewriter.getIntegerAttr(i16Type, kBF16AbsSignMask));
  Value signMask =
      rewriter
          .create<VdupOp>(loc, integerVRegType, signMaskScalar, *nativeMask,
                          /*position=*/nullptr)
          .getResult();
  Value integerResult =
      rewriter
          .create<VandOp>(loc, integerVRegType, integerSource, signMask, mask)
          .getResult();
  return rewriter.create<VbitcastOp>(loc, resultType, integerResult)
      .getResult();
}

/// Bitcasts a shift count to its signed carrier, which VPTO shifts require
/// regardless of the signedness of the shifted value.
FailureOr<Value> castShiftCount(PatternRewriter &rewriter, Location loc,
                                Value count) {
  auto countType = dyn_cast<VRegType>(count.getType());
  auto countElement =
      countType ? dyn_cast<IntegerType>(countType.getElementType())
                : IntegerType();
  if (!countType || !countElement) {
    return failure();
  }
  auto signedElement =
      IntegerType::get(rewriter.getContext(), countElement.getWidth(),
                       IntegerType::SignednessSemantics::Signed);
  auto signedType = VRegType::get(rewriter.getContext(),
                                  countType.getElementCount(), signedElement);
  return bitcastVReg(loc, count, signedType, rewriter);
}

/// Number of physical operand groups a unified op feeds to its target op.
template <typename SourceOp, typename Adaptor>
constexpr size_t unifiedMaskedOperandCount() {
  if constexpr (std::is_same_v<SourceOp, VMIVmulaOp>) {
    return mlir::pto::kValue3;
  } else if constexpr (std::is_same_v<SourceOp, VMIVaxpyOp> ||
                       std::is_same_v<SourceOp, VMIVpreluOp>) {
    return mlir::pto::kValue2;
  } else if constexpr (std::is_same_v<SourceOp, VMIVlreluOp>) {
    return 1;
  } else if constexpr (UnifiedMaskedHasLhsRhs<Adaptor>::value) {
    return mlir::pto::kValue2;
  } else {
    return 1;
  }
}

/// Emits one physical part of the VPTO op that corresponds to `TargetOp`.
template <typename TargetOp, size_t OperandCount>
FailureOr<Value> createUnifiedMaskedTarget(PatternRewriter &rewriter,
                                           Location loc, Type resultType,
                                           ArrayRef<Value> operands,
                                           ArrayRef<Value> scalars, Value mask) {
  if constexpr (std::is_same_v<TargetOp, VshlOp> ||
                std::is_same_v<TargetOp, VshrOp>) {
    FailureOr<Value> count = castShiftCount(rewriter, loc, operands[1]);
    if (failed(count)) {
      return failure();
    }
    return rewriter
        .create<TargetOp>(loc, resultType, operands[0], *count, mask)
        .getResult();
  } else if constexpr (std::is_same_v<TargetOp, VmulaOp>) {
    return rewriter
        .create<VmulaOp>(loc, resultType, operands[0], operands[1], operands[2],
                         mask)
        .getResult();
  } else if constexpr (std::is_same_v<TargetOp, VaxpyOp>) {
    return rewriter
        .create<VaxpyOp>(loc, resultType, operands[0], operands[1], scalars[0],
                         mask)
        .getResult();
  } else if constexpr (std::is_same_v<TargetOp, VlreluOp>) {
    return rewriter
        .create<VlreluOp>(loc, resultType, operands[0], scalars[0], mask)
        .getResult();
  } else if constexpr (std::is_same_v<TargetOp, VpreluOp>) {
    return rewriter
        .create<VpreluOp>(loc, resultType, operands[0], operands[1], mask)
        .getResult();
  } else if constexpr (std::is_same_v<TargetOp, VabsOp>) {
    auto vregType = dyn_cast<VRegType>(resultType);
    if (!vregType) {
      return failure();
    }
    if (!vregType.getElementType().isBF16()) {
      return rewriter.create<VabsOp>(loc, resultType, operands[0], mask)
          .getResult();
    }
    return lowerBF16Vabs(loc, operands[0], vregType, mask, resultType, rewriter);
  } else if constexpr (OperandCount == 1) {
    return rewriter
        .create<TargetOp>(loc, resultType, operands[0], mask)
        .getResult();
  } else {
    return rewriter
        .create<TargetOp>(loc, resultType, operands[0], operands[1], mask)
        .getResult();
  }
}

/// Shared entry of the unified mask-preserving patterns: resolves the physical
/// result types before lowering.
template <typename SourceOp>
FailureOr<SmallVector<Type>> getUnifiedMaskedResultTypes(
    SourceOp op, const TypeConverter &typeConverter) {
  return getConvertedResultTypes(op, 0, typeConverter);
}

/// Lowering pattern shared by every unified op that must keep its predicate: the
/// operand shape and the target op are supplied by the template arguments.
template <typename SourceOp, typename TargetOp>
struct OneToNUnifiedMaskedOpPattern : OneToNOpConversionPattern<SourceOp> {
  using OneToNOpConversionPattern<SourceOp>::OneToNOpConversionPattern;
  using AdaptorType =
      typename OneToNOpConversionPattern<SourceOp>::OpAdaptor;

  LogicalResult matchAndRewrite(
      SourceOp op,
      typename OneToNOpConversionPattern<SourceOp>::OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter) const override {
    FailureOr<SmallVector<Type>> resultTypes =
        getUnifiedMaskedResultTypes(op, *this->getTypeConverter());
    if (failed(resultTypes)) {
      return failure();
    }
    SmallVector<ValueRange> operandGroups;
    SmallVector<Value> maskParts;
    SmallVector<Value> scalars;
    if (failed(collectUnifiedMaskedOperands(op, adaptor, rewriter, operandGroups,
                                            maskParts, scalars))) {
      return rewriter.notifyMatchFailure(
          op, "unified masked operand conversion failed");
    }
    return lowerUnifiedMaskedOp(
        op, operandGroups, maskParts, *resultTypes, *this->getTypeConverter(),
        rewriter, op->getName().getStringRef(),
        [&](Location loc, Type resultType, ArrayRef<Value> operands,
            Value mask) {
          return createUnifiedMaskedTarget<
              TargetOp, unifiedMaskedOperandCount<SourceOp, AdaptorType>()>(
              rewriter, loc, resultType, operands, scalars, mask);
        });
  }
};

// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#pragma once
//===- VMIToVPTOPatternInternals5.inc - VMIToVPTO internals -*- C++ -*-===//
//===----------------------------------------------------------------------===//

template <typename OpTy, typename GroupReduceOpTy, typename RowReduceOpTy,
          typename CombineOpTy>
struct OneToNVMIGroupReduceOpPattern : OneToNOpConversionPattern<OpTy> {
  using OneToNOpConversionPattern<OpTy>::OneToNOpConversionPattern;

private:
  FailureOr<Value> buildOneBlockGroupResult(
      OpTy op, Value sourcePart, Value maskPart, Type resultType,
      VRegType expectedResultType, MaskType expectedMaskType,
      OneToNPatternRewriter &rewriter) const {
    bool mismatchedTypes = sourcePart.getType() != expectedResultType ||
                           maskPart.getType() != expectedMaskType ||
                           resultType != expectedResultType;
    if (mismatchedTypes) {
      return rewriter.notifyMatchFailure(
          op, "vcg group_reduce path requires uniform physical chunk types");
    }
    return rewriter
        .create<GroupReduceOpTy>(op.getLoc(), expectedResultType, sourcePart,
                                 maskPart)
        .getResult();
  }

  LogicalResult lowerOneBlock(
      OpTy op, ValueRange sourceParts, ValueRange maskParts,
      TypeRange resultTypes, OneToNPatternRewriter &rewriter) const {
    bool invalidArity = sourceParts.size() != maskParts.size() ||
                        sourceParts.size() != resultTypes.size() ||
                        sourceParts.empty();
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, "vcg group_reduce path requires matching physical arity");
    }
    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    auto maskType = dyn_cast<MaskType>(maskParts.front().getType());
    if (!resultType || !maskType) {
      return rewriter.notifyMatchFailure(
          op, "vcg group_reduce path requires physical vreg/mask");
    }
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [sourceIndex, sourcePart] : llvm::enumerate(sourceParts)) {
      FailureOr<Value> result = buildOneBlockGroupResult(
          op, sourcePart, maskParts[sourceIndex], resultTypes[sourceIndex],
          resultType, maskType, rewriter);
      if (failed(result)) {
        return failure();
      }
      results.push_back(*result);
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<Value> buildTwoBlockGroupResult(
      OpTy op, ValueRange sourceParts, ValueRange maskParts,
      TypeRange resultTypes, int64_t resultIndex, int64_t resultPartCount,
      int64_t numGroups, VRegType resultType, MaskType maskType,
      OneToNPatternRewriter &rewriter) const {
    Value loSource = sourceParts[resultIndex];
    Value hiSource = sourceParts[resultPartCount + resultIndex];
    Value loMask = maskParts[resultIndex];
    Value hiMask = maskParts[resultPartCount + resultIndex];
    bool mismatchedTypes = resultTypes[resultIndex] != resultType ||
                           loSource.getType() != resultType ||
                           hiSource.getType() != resultType ||
                           loMask.getType() != maskType ||
                           hiMask.getType() != maskType;
    if (mismatchedTypes) {
      return rewriter.notifyMatchFailure(
          op, "two-block group_reduce requires uniform physical types");
    }
    int64_t activeGroups = std::min<int64_t>(8, numGroups - resultIndex * 8);
    FailureOr<Value> combineMask = createPrefixMaskForActiveLanes(
        op.getLoc(), maskType, activeGroups, rewriter);
    if (failed(combineMask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to create two-block group_reduce combine mask");
    }
    Value lo = rewriter
                   .create<GroupReduceOpTy>(op.getLoc(), resultType, loSource,
                                            loMask)
                   .getResult();
    Value hi = rewriter
                   .create<GroupReduceOpTy>(op.getLoc(), resultType, hiSource,
                                            hiMask)
                   .getResult();
    return rewriter
        .create<CombineOpTy>(op.getLoc(), resultType, lo, hi, *combineMask)
        .getResult();
  }

  LogicalResult lowerTwoBlock(
      OpTy op, ValueRange sourceParts, ValueRange maskParts,
      TypeRange resultTypes, int64_t numGroups,
      OneToNPatternRewriter &rewriter) const {
    int64_t resultPartCount = resultTypes.size();
    bool invalidArity = static_cast<int64_t>(sourceParts.size()) !=
                            resultPartCount * 2 ||
                        maskParts.size() != sourceParts.size();
    if (invalidArity) {
      return rewriter.notifyMatchFailure(op,
                                         "two-block group_reduce arity mismatch");
    }
    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    auto maskType = dyn_cast<MaskType>(maskParts.front().getType());
    if (!resultType || !maskType) {
      return rewriter.notifyMatchFailure(
          op, "two-block group_reduce requires physical vreg/mask");
    }

    SmallVector<Value> results;
    results.reserve(resultPartCount);
    for (int64_t resultIndex = 0; resultIndex < resultPartCount;
         ++resultIndex) {
      FailureOr<Value> result = buildTwoBlockGroupResult(
          op, sourceParts, maskParts, resultTypes, resultIndex, resultPartCount,
          numGroups, resultType, maskType, rewriter);
      if (failed(result)) {
        return failure();
      }
      results.push_back(*result);
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<Value> buildFourBlockGroupResult(
      OpTy op, ValueRange sourceParts, ValueRange maskParts,
      TypeRange resultTypes, int64_t resultIndex, int64_t resultPartCount,
      int64_t numGroups, VRegType resultType, MaskType maskType,
      OneToNPatternRewriter &rewriter) const {
    int64_t activeGroups = std::min<int64_t>(8, numGroups - resultIndex * 8);
    FailureOr<Value> combineMask = createPrefixMaskForActiveLanes(
        op.getLoc(), maskType, activeGroups, rewriter);
    if (failed(combineMask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to create four-block group_reduce combine mask");
    }
    SmallVector<Value, 4> partials;
    partials.reserve(4);
    for (int64_t part = 0; part < 4; ++part) {
      int64_t sourceIndex = part * resultPartCount + resultIndex;
      Value source = sourceParts[sourceIndex];
      Value mask = maskParts[sourceIndex];
      bool mismatchedTypes = resultTypes[resultIndex] != resultType ||
                             source.getType() != resultType ||
                             mask.getType() != maskType;
      if (mismatchedTypes) {
        return rewriter.notifyMatchFailure(
            op, "four-block group_reduce requires uniform physical types");
      }
      partials.push_back(rewriter
                             .create<GroupReduceOpTy>(op.getLoc(), resultType,
                                                      source, mask)
                             .getResult());
    }
    Value sum01 = rewriter
                      .create<CombineOpTy>(op.getLoc(), resultType, partials[0],
                                           partials[1], *combineMask)
                      .getResult();
    Value sum23 = rewriter
                      .create<CombineOpTy>(op.getLoc(), resultType, partials[2],
                                           partials[3], *combineMask)
                      .getResult();
    return rewriter
        .create<CombineOpTy>(op.getLoc(), resultType, sum01, sum23,
                             *combineMask)
        .getResult();
  }

  LogicalResult lowerFourBlock(
      OpTy op, ValueRange sourceParts, ValueRange maskParts,
      TypeRange resultTypes, int64_t numGroups,
      OneToNPatternRewriter &rewriter) const {
    int64_t resultPartCount = resultTypes.size();
    bool invalidArity = static_cast<int64_t>(sourceParts.size()) !=
                            resultPartCount * 4 ||
                        maskParts.size() != sourceParts.size();
    if (invalidArity) {
      return rewriter.notifyMatchFailure(op,
                                         "four-block group_reduce arity mismatch");
    }
    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    auto maskType = dyn_cast<MaskType>(maskParts.front().getType());
    if (!resultType || !maskType) {
      return rewriter.notifyMatchFailure(
          op, "four-block group_reduce requires physical vreg/mask");
    }

    SmallVector<Value> results;
    results.reserve(resultPartCount);
    for (int64_t resultIndex = 0; resultIndex < resultPartCount;
         ++resultIndex) {
      FailureOr<Value> result = buildFourBlockGroupResult(
          op, sourceParts, maskParts, resultTypes, resultIndex,
          resultPartCount, numGroups, resultType, maskType, rewriter);
      if (failed(result)) {
        return failure();
      }
      results.push_back(*result);
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<SmallVector<Value>> buildDeinterleaved2GroupResults(
      OpTy op, ValueRange sourceParts, ValueRange maskParts,
      int64_t groupCount, int64_t chunksPerGroup, int64_t chunksPerPart,
      VRegType sourcePartType, VRegType rowResultType, MaskType maskType,
      Value firstLaneMask, OneToNPatternRewriter &rewriter) const {
    SmallVector<Value> results;
    results.reserve(groupCount);
    for (int64_t group = 0; group < groupCount; ++group) {
      Value accumulator;
      for (int64_t chunk = 0; chunk < chunksPerGroup; ++chunk) {
        int64_t loIndex = group * chunksPerGroup + chunk;
        int64_t hiIndex = chunksPerPart + loIndex;
        bool mismatchedTypes =
            sourceParts[loIndex].getType() != sourcePartType ||
            sourceParts[hiIndex].getType() != sourcePartType ||
            maskParts[loIndex].getType() != maskType ||
            maskParts[hiIndex].getType() != maskType;
        if (mismatchedTypes) {
          return rewriter.notifyMatchFailure(
              op, "deinterleaved=2 group_reduce requires uniform physical "
                  "chunk types");
        }
        Value low = rewriter
                        .create<RowReduceOpTy>(op.getLoc(), rowResultType,
                                               sourceParts[loIndex],
                                               maskParts[loIndex])
                        .getResult();
        Value high = rewriter
                         .create<RowReduceOpTy>(op.getLoc(), rowResultType,
                                                sourceParts[hiIndex],
                                                maskParts[hiIndex])
                         .getResult();
        Value pair = rewriter
                         .create<CombineOpTy>(op.getLoc(), rowResultType, low,
                                              high, firstLaneMask)
                         .getResult();
        accumulator =
            accumulator
                ? rewriter
                      .create<CombineOpTy>(op.getLoc(), rowResultType, pair,
                                           accumulator, firstLaneMask)
                      .getResult()
                : pair;
      }
      results.push_back(accumulator);
    }
    return results;
  }

  FailureOr<SmallVector<Value>> restoreDeinterleaved2GroupResults(
      OpTy op, ArrayRef<Value> reducedResults, TypeRange resultTypes,
      VRegType resultType, OneToNPatternRewriter &rewriter) const {
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (Value reducedResult : reducedResults) {
      FailureOr<Value> finalResult =
          bitcastVReg(op.getLoc(), reducedResult, resultType, rewriter);
      if (failed(finalResult)) {
        return rewriter.notifyMatchFailure(
            op, "failed to restore deinterleaved=2 group result type");
      }
      results.push_back(*finalResult);
    }
    return results;
  }

  struct Deinterleaved2GroupReduceTypes {
    VRegType resultType;
    MaskType maskType;
    VRegType sourcePartType;
    VRegType rowResultType;
    MaskType rowMaskType;
  };

  FailureOr<Deinterleaved2GroupReduceTypes>
  getDeinterleaved2GroupReduceTypes(
      OpTy op, ValueRange sourceParts, ValueRange maskParts,
      TypeRange resultTypes, OneToNPatternRewriter &rewriter) const {
    for (Type resultType : resultTypes) {
      if (!isa<VRegType>(resultType)) {
        return rewriter.notifyMatchFailure(
            op, "deinterleaved=2 group_reduce result must be vreg");
      }
    }
    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    auto maskType = dyn_cast<MaskType>(maskParts.front().getType());
    if (!resultType || !maskType) {
      return rewriter.notifyMatchFailure(
          op, "deinterleaved=2 group_reduce requires physical vreg/mask");
    }
    auto sourcePartType = dyn_cast<VRegType>(sourceParts.front().getType());
    if (!sourcePartType) {
      return rewriter.notifyMatchFailure(
          op, "deinterleaved=2 group_reduce source must be vreg");
    }
    FailureOr<VRegType> rowResultType =
        getRowResultType(sourcePartType, resultType);
    if (failed(rowResultType)) {
      return rewriter.notifyMatchFailure(
          op, "failed to derive deinterleaved=2 row-reduction type");
    }
    FailureOr<MaskType> rowMaskType =
        getMaskTypeForVReg(*rowResultType, rewriter.getContext());
    if (failed(rowMaskType)) {
      return rewriter.notifyMatchFailure(
          op, "failed to derive deinterleaved=2 combine mask type");
    }
    return Deinterleaved2GroupReduceTypes{
        resultType, maskType, sourcePartType, *rowResultType, *rowMaskType};
  }

  FailureOr<std::pair<int64_t, int64_t>> validateFullDeinterleaved2Shape(
      OpTy op, VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
      ValueRange sourceParts, ValueRange maskParts, TypeRange resultTypes,
      int64_t groupSize, OneToNPatternRewriter &rewriter) const {
    VMILayoutAttr resultLayout = resultVMIType.getLayoutAttr();
    bool rowLocalSlots1Result = resultLayout && resultLayout.isGroupSlots() &&
                                resultLayout.getSlots() == 1;
    if (!rowLocalSlots1Result) {
      return rewriter.notifyMatchFailure(
          op, "deinterleaved=2 full group_reduce requires slots=1 result");
    }
    FailureOr<int64_t> lanesPerPart =
        getDataLanesPerPart(sourceVMIType.getElementType());
    if (failed(lanesPerPart)) {
      return rewriter.notifyMatchFailure(
          op, "deinterleaved=2 group_reduce requires known physical lanes");
    }
    int64_t safeLanesPerPart = *lanesPerPart > 0 ? *lanesPerPart : 1;
    int64_t safeGroupSize = groupSize > 0 ? groupSize : 1;
    bool invalidGroupSize = groupSize % (2 * safeLanesPerPart) != 0;
    if (invalidGroupSize) {
      return rewriter.notifyMatchFailure(
          op, "deinterleaved=2 group_reduce requires group size to be a "
              "multiple of two physical chunks");
    }
    int64_t groupCount = sourceVMIType.getElementCount() / safeGroupSize;
    int64_t chunksPerGroup = groupSize / (2 * safeLanesPerPart);
    int64_t chunksPerPart = groupCount * chunksPerGroup;
    bool invalidArity = sourceParts.size() != maskParts.size() ||
                        static_cast<int64_t>(sourceParts.size()) !=
                            2 * chunksPerPart ||
                        static_cast<int64_t>(resultTypes.size()) != groupCount;
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, "deinterleaved=2 group_reduce arity mismatch");
    }
    return std::make_pair(groupCount, chunksPerGroup);
  }

  LogicalResult lowerFullDeinterleaved2(
      OpTy op, VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
      ValueRange sourceParts, ValueRange maskParts, TypeRange resultTypes,
      int64_t groupSize, OneToNPatternRewriter &rewriter) const {
    FailureOr<std::pair<int64_t, int64_t>> shape =
        validateFullDeinterleaved2Shape(
            op, sourceVMIType, resultVMIType, sourceParts, maskParts,
            resultTypes, groupSize, rewriter);
    if (failed(shape)) {
      return failure();
    }
    int64_t groupCount = shape->first;
    int64_t chunksPerGroupPerPart = shape->second;
    int64_t chunksPerPart = groupCount * chunksPerGroupPerPart;
    FailureOr<Deinterleaved2GroupReduceTypes> types =
        getDeinterleaved2GroupReduceTypes(op, sourceParts, maskParts,
                                          resultTypes, rewriter);
    if (failed(types)) {
      return failure();
    }
    FailureOr<Value> firstLaneMask =
        createPrefixMask(op.getLoc(), types->rowMaskType, "PAT_VL1", rewriter);
    if (failed(firstLaneMask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to create deinterleaved=2 group_reduce lane mask");
    }
    FailureOr<SmallVector<Value>> reducedResults =
        buildDeinterleaved2GroupResults(
            op, sourceParts, maskParts, groupCount, chunksPerGroupPerPart,
            chunksPerPart, types->sourcePartType, types->rowResultType,
            types->maskType,
            *firstLaneMask, rewriter);
    if (failed(reducedResults)) {
      return failure();
    }
    FailureOr<SmallVector<Value>> results = restoreDeinterleaved2GroupResults(
        op, *reducedResults, resultTypes, types->resultType, rewriter);
    if (failed(results)) {
      return failure();
    }
    replaceOpWithFlatConvertedValues(rewriter, op, *results,
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<Value> buildContiguousGroupReduceResult(
      OpTy op, ValueRange sourceParts, ValueRange maskParts, int64_t group,
      int64_t chunksPerGroup, VRegType sourcePartType, VRegType rowResultType,
      MaskType maskType, Value firstLaneMask,
      OneToNPatternRewriter &rewriter) const {
    Value accumulator;
    for (int64_t chunk = 0; chunk < chunksPerGroup; ++chunk) {
      int64_t index = group * chunksPerGroup + chunk;
      bool mismatchedTypes = sourceParts[index].getType() != sourcePartType ||
                             maskParts[index].getType() != maskType;
      if (mismatchedTypes) {
        return rewriter.notifyMatchFailure(
            op, "group_reduce requires uniform physical chunk types");
      }
      Value reduced = rewriter
                          .create<RowReduceOpTy>(op.getLoc(), rowResultType,
                                                 sourceParts[index],
                                                 maskParts[index])
                          .getResult();
      accumulator = accumulator
                        ? rewriter
                              .create<CombineOpTy>(op.getLoc(), rowResultType,
                                                   reduced, accumulator,
                                                   firstLaneMask)
                              .getResult()
                        : reduced;
    }
    return accumulator;
  }

  FailureOr<SmallVector<Value>> buildContiguousGroupReduceResults(
      OpTy op, ValueRange sourceParts, ValueRange maskParts,
      int64_t groupCount, int64_t chunksPerGroup, VRegType sourcePartType,
      VRegType rowResultType, MaskType maskType, Value firstLaneMask,
      OneToNPatternRewriter &rewriter) const {
    SmallVector<Value> results;
    results.reserve(groupCount);
    for (int64_t group = 0; group < groupCount; ++group) {
      FailureOr<Value> result = buildContiguousGroupReduceResult(
          op, sourceParts, maskParts, group, chunksPerGroup, sourcePartType,
          rowResultType, maskType, firstLaneMask, rewriter);
      if (failed(result)) {
        return failure();
      }
      results.push_back(*result);
    }
    return results;
  }

  FailureOr<SmallVector<Value>> restoreContiguousGroupResults(
      OpTy op, ArrayRef<Value> reducedResults, TypeRange resultTypes,
      VRegType resultType, int64_t groupCount, int64_t chunksPerGroup,
      bool rowLocalSlots1Result,
      OneToNPatternRewriter &rewriter) const {
    SmallVector<Value> results(resultTypes.size());
    for (int64_t group = 0; group < groupCount; ++group) {
      FailureOr<Value> finalResult =
          bitcastVReg(op.getLoc(), reducedResults[group], resultType, rewriter);
      if (failed(finalResult)) {
        return rewriter.notifyMatchFailure(
            op, "failed to restore group result type");
      }
      int64_t destChunk = rowLocalSlots1Result ? group : group * chunksPerGroup;
      if (rowLocalSlots1Result) {
        results[destChunk] = *finalResult;
      } else {
        for (int64_t chunk = 0; chunk < chunksPerGroup; ++chunk) {
          results[destChunk + chunk] = *finalResult;
        }
      }
    }
    return results;
  }

  struct ContiguousGroupReduceTypes {
    VRegType resultType;
    MaskType maskType;
    VRegType sourcePartType;
    VRegType rowResultType;
    MaskType rowMaskType;
  };

  struct ContiguousGroupReduceShape {
    int64_t lanesPerPart;
    int64_t groupCount;
    int64_t chunksPerGroup;
    bool rowLocalSlots1Result;
    int64_t expectedResultParts;
  };

  FailureOr<ContiguousGroupReduceShape> getContiguousGroupReduceShape(
      OpTy op, VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
      ValueRange sourceParts, ValueRange maskParts, TypeRange resultTypes,
      int64_t groupSize, OneToNPatternRewriter &rewriter) const {
    int64_t lanesPerPart = 0;
    int64_t groupCount = 0;
    int64_t chunksPerGroup = 0;
    if (failed(checkContiguousFullGroupChunks(op, sourceVMIType, groupSize,
                                              &lanesPerPart, &groupCount,
                                              &chunksPerGroup, rewriter))) {
      return failure();
    }
    VMILayoutAttr resultLayout = resultVMIType.getLayoutAttr();
    bool rowLocalSlots1Result = resultLayout && resultLayout.isGroupSlots() &&
                                resultLayout.getNumGroups() == groupCount &&
                                resultLayout.getSlots() == 1;
    int64_t expectedResultParts =
        rowLocalSlots1Result ? groupCount : groupCount * chunksPerGroup;
    bool invalidArity =
        sourceParts.size() != maskParts.size() ||
        static_cast<int64_t>(sourceParts.size()) != groupCount * chunksPerGroup ||
        static_cast<int64_t>(resultTypes.size()) != expectedResultParts;
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, "group_reduce requires matching source/mask/result arity");
    }
    return ContiguousGroupReduceShape{lanesPerPart, groupCount, chunksPerGroup,
                                      rowLocalSlots1Result, expectedResultParts};
  }

  FailureOr<ContiguousGroupReduceTypes> getContiguousGroupReduceTypes(
      OpTy op, ValueRange sourceParts, ValueRange maskParts,
      TypeRange resultTypes, OneToNPatternRewriter &rewriter) const {
    for (Type resultType : resultTypes) {
      if (!isa<VRegType>(resultType)) {
        return rewriter.notifyMatchFailure(
            op, "group_reduce result must be vreg");
      }
    }
    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    auto maskType = dyn_cast<MaskType>(maskParts.front().getType());
    if (!resultType || !maskType) {
      return rewriter.notifyMatchFailure(
          op, "group_reduce requires physical vreg result and mask");
    }
    auto sourcePartType = dyn_cast<VRegType>(sourceParts.front().getType());
    if (!sourcePartType) {
      return rewriter.notifyMatchFailure(op,
                                         "group_reduce source must be vreg");
    }
    FailureOr<VRegType> rowResultType =
        getRowResultType(sourcePartType, resultType);
    if (failed(rowResultType)) {
      return rewriter.notifyMatchFailure(
          op, "failed to derive group row-reduction type");
    }
    FailureOr<MaskType> rowMaskType =
        getMaskTypeForVReg(*rowResultType, rewriter.getContext());
    if (failed(rowMaskType)) {
      return rewriter.notifyMatchFailure(
          op, "failed to derive group combine mask type");
    }
    return ContiguousGroupReduceTypes{
        resultType, maskType, sourcePartType, *rowResultType, *rowMaskType};
  }

  LogicalResult lowerRowLocalContiguousGroupResults(
      OpTy op, ValueRange sourceParts, ValueRange maskParts,
      TypeRange resultTypes, const ContiguousGroupReduceShape &shape,
      const ContiguousGroupReduceTypes &types, Value firstLaneMask,
      OneToNPatternRewriter &rewriter) const {
    // One contiguous group maps to one result chunk: materialize the row
    // reduction and its final type view per group so the emitted ops keep
    // master's per-group ordering.
    SmallVector<Value> results(resultTypes.size());
    for (int64_t group = 0;
         group < shape.groupCount &&
         static_cast<size_t>(group) < results.size();
         ++group) {
      FailureOr<Value> reduced = buildContiguousGroupReduceResult(
          op, sourceParts, maskParts, group, shape.chunksPerGroup,
          types.sourcePartType, types.rowResultType, types.maskType,
          firstLaneMask, rewriter);
      if (failed(reduced)) {
        return failure();
      }
      FailureOr<Value> finalResult =
          bitcastVReg(op.getLoc(), *reduced, types.resultType, rewriter);
      if (failed(finalResult)) {
        return rewriter.notifyMatchFailure(
            op, "failed to restore group result type");
      }
      results[group] = *finalResult;
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  LogicalResult lowerContiguousRows(
      OpTy op, VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
      ValueRange sourceParts, ValueRange maskParts, TypeRange resultTypes,
      int64_t groupSize, OneToNPatternRewriter &rewriter) const {
    FailureOr<ContiguousGroupReduceShape> shape =
        getContiguousGroupReduceShape(op, sourceVMIType, resultVMIType,
                                      sourceParts, maskParts, resultTypes,
                                      groupSize, rewriter);
    if (failed(shape)) {
      return failure();
    }
    FailureOr<ContiguousGroupReduceTypes> types =
        getContiguousGroupReduceTypes(op, sourceParts, maskParts, resultTypes,
                                      rewriter);
    if (failed(types)) {
      return failure();
    }
    FailureOr<Value> firstLaneMask =
        createPrefixMask(op.getLoc(), types->rowMaskType, "PAT_VL1", rewriter);
    if (failed(firstLaneMask)) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to create group_reduce masks");
    }
    if (shape->rowLocalSlots1Result) {
      return lowerRowLocalContiguousGroupResults(
          op, sourceParts, maskParts, resultTypes, *shape, *types,
          *firstLaneMask, rewriter);
    }
    return lowerBatchedContiguousGroupResults(
        op, sourceParts, maskParts, resultTypes, *shape, *types,
        *firstLaneMask, rewriter);
  }

  LogicalResult lowerBatchedContiguousGroupResults(
      OpTy op, ValueRange sourceParts, ValueRange maskParts,
      TypeRange resultTypes, const ContiguousGroupReduceShape &shape,
      const ContiguousGroupReduceTypes &types, Value firstLaneMask,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<SmallVector<Value>> reducedResults =
        buildContiguousGroupReduceResults(
            op, sourceParts, maskParts, shape.groupCount, shape.chunksPerGroup,
            types.sourcePartType, types.rowResultType, types.maskType,
            firstLaneMask, rewriter);
    if (failed(reducedResults)) {
      return failure();
    }
    FailureOr<SmallVector<Value>> results = restoreContiguousGroupResults(
        op, *reducedResults, resultTypes, types.resultType, shape.groupCount,
        shape.chunksPerGroup, shape.rowLocalSlots1Result, rewriter);
    if (failed(results)) {
      return failure();
    }
    replaceOpWithFlatConvertedValues(rewriter, op, *results,
                                     *this->getTypeConverter());
    return success();
  }

public:

  LogicalResult
  matchAndRewrite(OpTy op,
                  typename OneToNOpConversionPattern<OpTy>::OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto sourceVMIType = cast<VMIVRegType>(op.getSource().getType());
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    ValueRange sourceParts = adaptor.getSource();
    ValueRange maskParts = adaptor.getMask();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    bool failedResultTypeConversion = failed(maybe_resultTypes);
    if (failedResultTypeConversion) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);

    VMILayoutSupport supports;
    std::string supportReason;
    if (failed(getSupport(supports, op, &supportReason))) {
      return rewriter.notifyMatchFailure(
          op, Twine(op->getName().getStringRef()) +
                  " has no layout support: " + supportReason);
    }
    auto maskVMIType = cast<VMIMaskType>(op.getMask().getType());
    FailureOr<GroupReduceLoweringPlan> plan = classifyGroupReduceLoweringPlan(
        sourceVMIType, maskVMIType, resultVMIType,
        op.getNumGroupsAttr().getInt(), &supportReason);
    if (failed(plan)) {
      return rewriter.notifyMatchFailure(
          op, Twine(op->getName().getStringRef()) +
                  " has no lowering plan: " + supportReason);
    }

    FailureOr<int64_t> groupSize = getGroupSizeFromNumGroups(
        sourceVMIType, op.getNumGroupsAttr().getInt());
    if (failed(groupSize)) {
      return rewriter.notifyMatchFailure(
          op, "group reduce requires num_groups to evenly divide lane count");
    }

    return lowerByPlan(op, *plan, *groupSize, sourceVMIType, resultVMIType,
                       sourceParts, maskParts, resultTypes, rewriter);
  }

private:
  LogicalResult lowerByPlan(
      OpTy op, GroupReduceLoweringPlan plan, int64_t groupSize,
      VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
      ValueRange sourceParts, ValueRange maskParts, TypeRange resultTypes,
      OneToNPatternRewriter &rewriter) const {
    int64_t numGroups = op.getNumGroupsAttr().getInt();
    switch (plan) {
    case GroupReduceLoweringPlan::OneBlockVcgadd:
      return lowerOneBlock(op, sourceParts, maskParts, resultTypes, rewriter);
    case GroupReduceLoweringPlan::TwoBlockDeinterleaved2VcgaddVadd:
      return lowerTwoBlock(op, sourceParts, maskParts, resultTypes, numGroups,
                           rewriter);
    case GroupReduceLoweringPlan::FourBlockDeinterleaved4VcgaddTree:
      return lowerFourBlock(op, sourceParts, maskParts, resultTypes, numGroups,
                            rewriter);
    case GroupReduceLoweringPlan::FullDeinterleaved2VcaddRows:
      return lowerFullDeinterleaved2(
          op, sourceVMIType, resultVMIType, sourceParts, maskParts, resultTypes,
          groupSize, rewriter);
    case GroupReduceLoweringPlan::ContiguousVcaddRows:
      return lowerContiguousRows(op, sourceVMIType, resultVMIType, sourceParts,
                                 maskParts, resultTypes, groupSize, rewriter);
    }
    return rewriter.notifyMatchFailure(op, "unknown group_reduce lowering plan");
  }

  FailureOr<VRegType> getRowResultType(VRegType sourceType,
                                       VRegType resultType) const {
    if constexpr (std::is_same_v<OpTy, VMIGroupReduceAddIOp>) {
      return getVcaddResultType(sourceType);
    }
    return resultType;
  }

  LogicalResult getSupport(VMILayoutSupport &supports, VMIGroupReduceAddFOp op,
                           std::string *reason) const {
    return supports.getGroupReduceAddFSupport(op, reason);
  }

  LogicalResult getSupport(VMILayoutSupport &supports, VMIGroupReduceAddIOp op,
                           std::string *reason) const {
    return supports.getGroupReduceAddISupport(op, reason);
  }

  LogicalResult getSupport(VMILayoutSupport &supports, VMIGroupReduceMaxIOp op,
                           std::string *reason) const {
    return supports.getGroupReduceMaxISupport(op, reason);
  }

  LogicalResult getSupport(VMILayoutSupport &supports, VMIGroupReduceMaxFOp op,
                           std::string *reason) const {
    return supports.getGroupReduceMaxFSupport(op, reason);
  }

  LogicalResult getSupport(VMILayoutSupport &supports, VMIGroupReduceMinFOp op,
                           std::string *reason) const {
    return supports.getGroupReduceMinFSupport(op, reason);
  }

  LogicalResult getSupport(VMILayoutSupport &supports, VMIGroupReduceMinIOp op,
                           std::string *reason) const {
    return supports.getGroupReduceMinISupport(op, reason);
  }

  ;
};

struct OneToNVMIGroupBroadcastOpPattern
    : OneToNOpConversionPattern<VMIGroupBroadcastOp> {
  using OneToNOpConversionPattern<VMIGroupBroadcastOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIGroupBroadcastOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    FailureOr<VMIPhysicalConversionInput> input =
        getVMIPhysicalConversionInput(op, adaptor, *this->getTypeConverter());
    if (failed(input)) {
      return failure();
    }
    SmallVector<Value> results;
    if (failed(lowerGroupBroadcastParts(
            op, input->sourceParts, input->sourceVMIType,
            input->resultVMIType, input->resultTypes,
            op.getNumGroupsAttr().getInt(), rewriter, results))) {
      return failure();
    }

    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// VMI vdhist / vchist → VPTO dhistv2 / chistv2 lowering (shared template)
//===----------------------------------------------------------------------===//

template <typename VMIOp, typename VPTOHistOp>
static LogicalResult lowerHistogramChunk(
    VMIOp op, Value source, Value userMask, int64_t firstLane,
    int64_t lanesPerPart, SmallVectorImpl<Value> &halves,
    ArrayRef<Value> binConsts, VRegType partType,
    OneToNPatternRewriter &rewriter) {
  auto maskType = dyn_cast<MaskType>(userMask.getType());
  if (!maskType || !maskType.isB8()) {
    return rewriter.notifyMatchFailure(op, "expected b8 source mask");
  }
  Value chunkMask = userMask;
  int64_t activeLanes = std::min<int64_t>(
      lanesPerPart,
      cast<VMIVRegType>(op.getSource().getType()).getElementCount() - firstLane);
  if (activeLanes < lanesPerPart) {
    FailureOr<Value> validMask = createPrefixMaskForActiveLanes(
        op.getLoc(), maskType, activeLanes, rewriter);
    FailureOr<Value> allMask =
        createAllTrueMask(op.getLoc(), maskType, rewriter);
    bool failedMask = failed(validMask) || failed(allMask);
    if (failedMask) {
      return rewriter.notifyMatchFailure(
          op, "failed to materialize tail-valid b8 mask");
    }
    chunkMask = rewriter
                    .create<PandOp>(op.getLoc(), maskType, chunkMask, *validMask,
                                    *allMask)
                    .getResult();
  }
  for (size_t half = 0; half < halves.size(); ++half) {
    halves[half] = rewriter
                       .create<VPTOHistOp>(op.getLoc(), partType, halves[half],
                                            source, chunkMask, binConsts[half])
                       .getResult();
  }
  return success();
}

template <typename VMIOp>
struct HistogramPhysicalPlan {
  ValueRange sourceParts;
  ValueRange maskParts;
  SmallVector<Value, 2> halves;
  SmallVector<Value, 2> binConsts;
  VRegType partType;
  int64_t lanesPerPart;
  size_t halfCount;
};

template <typename VMIOp>
static FailureOr<HistogramPhysicalPlan<VMIOp>> prepareHistogramPhysicalPlan(
    VMIOp op,
    typename OneToNOpConversionPattern<VMIOp>::OpAdaptor adaptor,
    OneToNPatternRewriter &rewriter) {
  ValueRange accParts = adaptor.getAcc();
  ValueRange sourceParts = adaptor.getSource();
  ValueRange maskParts = adaptor.getMask();
  size_t halfCount = accParts.size();
  const bool invalidHalfCount = halfCount != 1 && halfCount != 2;
  if (invalidHalfCount) {
      (void)rewriter.notifyMatchFailure(op,
                                        "expected one or two accumulator parts");
      return failure();
  }
  const bool invalidSourceMaskArity =
      sourceParts.empty() || sourceParts.size() != maskParts.size();
  if (invalidSourceMaskArity) {
      (void)rewriter.notifyMatchFailure(op,
                                        "expected matching source/mask chunks");
      return failure();
  }
  auto partType = dyn_cast<VRegType>(accParts.front().getType());
  if (!partType) {
    (void)rewriter.notifyMatchFailure(op, "expected ui16 acc parts");
    return failure();
  }
  const bool mismatchedSecondHalf =
      halfCount == 2 && accParts[1].getType() != partType;
  if (mismatchedSecondHalf) {
    (void)rewriter.notifyMatchFailure(op, "expected matching ui16 acc parts");
    return failure();
  }
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  FailureOr<int64_t> lanesPerPart =
      getDataLanesPerPart(sourceType.getElementType());
  if (failed(lanesPerPart)) {
    (void)rewriter.notifyMatchFailure(op, "failed to compute source lanes");
    return failure();
  }
  Location loc = op.getLoc();
  SmallVector<Value, 2> binConsts;
  binConsts.push_back(createI32Constant(loc, 0, rewriter));
  if (halfCount == 2) {
    binConsts.push_back(createI32Constant(loc, 1, rewriter));
  }
  return HistogramPhysicalPlan<VMIOp>{
      sourceParts, maskParts,
      SmallVector<Value, 2>(accParts.begin(), accParts.end()),
      std::move(binConsts), partType, *lanesPerPart, halfCount};
}

template <typename VMIOp, typename VPTOHistOp>
static LogicalResult
lowerVMIHistogramToVPTO(VMIOp op,
                        typename OneToNOpConversionPattern<VMIOp>::OpAdaptor
                            adaptor,
                        TypeConverter *typeConverter,
                        OneToNPatternRewriter &rewriter) {
  FailureOr<HistogramPhysicalPlan<VMIOp>> plan =
      prepareHistogramPhysicalPlan(op, adaptor, rewriter);
  if (failed(plan)) {
    return failure();
  }

  for (size_t index = 0, e = plan->sourceParts.size(); index < e; ++index) {
    if (failed(lowerHistogramChunk<VMIOp, VPTOHistOp>(
            op, plan->sourceParts[index], plan->maskParts[index],
            static_cast<int64_t>(index) * plan->lanesPerPart,
            plan->lanesPerPart, plan->halves, plan->binConsts, plan->partType,
            rewriter))) {
      return failure();
    }
  }

  replaceOpWithFlatConvertedValues(
      rewriter, op,
      SmallVector<Value>(plan->halves.begin(),
                         plan->halves.begin() + plan->halfCount),
      *typeConverter);
  return success();
}

struct OneToNVMIVdhistOpPattern : OneToNOpConversionPattern<VMIVdhistOp> {
  using OneToNOpConversionPattern<VMIVdhistOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIVdhistOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    return lowerVMIHistogramToVPTO<VMIVdhistOp, Dhistv2Op>(
        op, adaptor, this->getTypeConverter(), rewriter);
  }
};

struct OneToNVMIVchistOpPattern : OneToNOpConversionPattern<VMIVchistOp> {
  using OneToNOpConversionPattern<VMIVchistOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIVchistOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    return lowerVMIHistogramToVPTO<VMIVchistOp, Chistv2Op>(
        op, adaptor, this->getTypeConverter(), rewriter);
  }
};

template <typename SourceOp, typename ChunkReduceOp, typename CombineOp>
struct OneToNVMIReduceMinMaxOpPattern : OneToNOpConversionPattern<SourceOp> {
  using OneToNOpConversionPattern<SourceOp>::OneToNOpConversionPattern;

private:
  LogicalResult lowerSequentialReduction(
      SourceOp op, ValueRange sourceParts, ValueRange maskParts,
      VRegType resultType, MaskType maskType,
      OneToNPatternRewriter &rewriter) const {
    Value accumulator =
        rewriter
            .create<ChunkReduceOp>(op.getLoc(), resultType, sourceParts.front(),
                                   maskParts.front())
            .getResult();
    const bool singlePart = sourceParts.size() == 1;
    if (singlePart) {
      replaceOpWithFlatConvertedValues(
          rewriter, op, SmallVector<Value>{accumulator},
          *this->getTypeConverter());
      return success();
    }
    FailureOr<Value> firstLaneMask =
        createPrefixMask(op.getLoc(), maskType, "PAT_VL1", rewriter);
    if (failed(firstLaneMask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to create min/max reduction first-lane mask");
    }
    for (size_t part = 1; part < sourceParts.size(); ++part) {
      Value reduced = rewriter
                          .create<ChunkReduceOp>(op.getLoc(), resultType,
                                                 sourceParts[part],
                                                 maskParts[part])
                          .getResult();
      accumulator = rewriter
                        .create<CombineOp>(op.getLoc(), resultType, reduced,
                                           accumulator, *firstLaneMask)
                        .getResult();
    }
    replaceOpWithFlatConvertedValues(
        rewriter, op, SmallVector<Value>{accumulator},
        *this->getTypeConverter());
    return success();
  }

  FailureOr<std::pair<VRegType, MaskType>> validatePhysicalParts(
      SourceOp op, ValueRange sourceParts, ValueRange maskParts,
      TypeRange resultTypes, OneToNPatternRewriter &rewriter) const {
    bool invalidArity = sourceParts.empty() || sourceParts.size() != maskParts.size() ||
                        resultTypes.size() != 1;
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, "min/max reduction requires matching source/mask chunks and one result chunk");
    }
    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    auto maskType = dyn_cast<MaskType>(maskParts.front().getType());
    if (!resultType || !maskType) {
      return rewriter.notifyMatchFailure(
          op, "min/max reduction requires matching physical source/result vregs and one mask");
    }
    for (Value sourcePart : sourceParts) {
      bool mismatch = sourcePart.getType() != resultType;
      if (mismatch) {
        return rewriter.notifyMatchFailure(
            op, "min/max reduction requires every source chunk to match result vreg type");
      }
    }
    for (Value maskPart : maskParts) {
      bool mismatch = maskPart.getType() != maskType;
      if (mismatch) {
        return rewriter.notifyMatchFailure(
            op, "min/max reduction requires every mask chunk to have the same predicate type");
      }
    }
    return std::make_pair(resultType, maskType);
  }

  LogicalResult lowerReduction(SourceOp op, ValueRange sourceParts,
                               ValueRange maskParts, VRegType resultType,
                               MaskType maskType,
                               OneToNPatternRewriter &rewriter) const {
    FailureOr<Value> combined = combineEquivalentMaskedParts<CombineOp>(
        op.getLoc(), sourceParts, maskParts, resultType, rewriter);
    if (succeeded(combined)) {
      Value reduced =
          rewriter
              .create<ChunkReduceOp>(op.getLoc(), resultType, *combined,
                                     maskParts.front())
              .getResult();
      replaceOpWithFlatConvertedValues(
          rewriter, op, SmallVector<Value>{reduced}, *this->getTypeConverter());
      return success();
    }

    return lowerSequentialReduction(op, sourceParts, maskParts, resultType,
                                    maskType, rewriter);
  }

public:

  LogicalResult matchAndRewrite(
      SourceOp op,
      typename OneToNOpConversionPattern<SourceOp>::OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    ValueRange maskParts = adaptor.getMask();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    bool failedResultTypeConversion = failed(maybe_resultTypes);
    if (failedResultTypeConversion) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    FailureOr<std::pair<VRegType, MaskType>> physical =
        validatePhysicalParts(op, sourceParts, maskParts, resultTypes, rewriter);
    if (failed(physical)) {
      return failure();
    }
    return lowerReduction(op, sourceParts, maskParts, physical->first,
                          physical->second, rewriter);
  }
};

struct OneToNVMIExtFOpPattern : OneToNOpConversionPattern<VMIExtFOp> {
  using OneToNOpConversionPattern<VMIExtFOp>::OneToNOpConversionPattern;

private:
  struct ExtFPhysicalPlan {
    VRegType sourceType;
    SmallVector<VRegType> resultTypes;
  };

  FailureOr<ExtFPhysicalPlan> buildPhysicalPlan(
      VMIExtFOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      OneToNPatternRewriter &rewriter) const {
    if (sourceParts.empty()) {
      return rewriter.notifyMatchFailure(
          op, "extf requires at least one physical source chunk");
    }
    auto sourceType = dyn_cast<VRegType>(sourceParts.front().getType());
    if (!sourceType) {
      return rewriter.notifyMatchFailure(op, "expected physical extf source");
    }
    for (Value sourcePart : sourceParts) {
      auto currentSourceType = dyn_cast<VRegType>(sourcePart.getType());
      if (!currentSourceType || currentSourceType != sourceType) {
        return rewriter.notifyMatchFailure(
            op, "extf source physical parts must have matching type");
      }
    }
    SmallVector<VRegType> resultVRegTypes;
    resultVRegTypes.reserve(resultTypes.size());
    for (Type resultType : resultTypes) {
      auto resultVRegType = dyn_cast<VRegType>(resultType);
      bool invalidFirstResult =
          resultVRegTypes.empty() &&
          (!resultVRegType ||
           !(resultVRegType.getElementType().isF32() ||
             pto::isPTOBF16x2Type(resultVRegType.getElementType())));
      bool mismatchedResult =
          !resultVRegTypes.empty() && resultVRegType != resultVRegTypes.front();
      if (invalidFirstResult || mismatchedResult) {
        return rewriter.notifyMatchFailure(
            op, "unsupported physical extf result type");
      }
      resultVRegTypes.push_back(resultVRegType);
    }
    return ExtFPhysicalPlan{sourceType, std::move(resultVRegTypes)};
  }

  struct ResultViewPlan {
    bool isPackedBF16x2;
    VRegType vcvtResultType;
  };

  ResultViewPlan buildResultViewPlan(ArrayRef<VRegType> resultTypes,
                                     OneToNPatternRewriter &rewriter) const {
    bool isPackedBF16x2 =
        pto::isPTOBF16x2Type(resultTypes.front().getElementType());
    VRegType vcvtResultType = resultTypes.front();
    if (isPackedBF16x2) {
      vcvtResultType = VRegType::get(
          rewriter.getContext(), resultTypes.front().getElementCount() * 2,
          BFloat16Type::get(rewriter.getContext()));
    }
    return ResultViewPlan{isPackedBF16x2, vcvtResultType};
  }

  static Value createVcvtResult(Location loc, VRegType resultType,
                                Value sourcePart, Value mask, StringAttr rnd,
                                StringAttr sat, StringAttr part,
                                bool resultIsPackedBF16x2,
                                VRegType vcvtResultVRegType,
                                OneToNPatternRewriter &rewriter) {
    VRegType vcvtType = resultIsPackedBF16x2 ? vcvtResultVRegType : resultType;
    Value vcvt = rewriter
                     .create<VcvtOp>(loc, vcvtType, sourcePart, mask, rnd, sat,
                                     part)
                     .getResult();
    if (!resultIsPackedBF16x2) {
      return vcvt;
    }
    return rewriter.create<VbitcastOp>(loc, resultType, vcvt).getResult();
  }

  LogicalResult lowerLaneStride(
      VMIExtFOp op, OneToNPatternRewriter &rewriter, ValueRange sourceParts,
      ArrayRef<VRegType> resultTypes, Value mask, StringRef part,
      bool resultIsPackedBF16x2, VRegType vcvtResultVRegType) const {
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [sourcePart, resultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      results.push_back(createVcvtResult(
          op.getLoc(), resultType, sourcePart, mask, /*rnd=*/nullptr,
          /*sat=*/nullptr, rewriter.getStringAttr(part),
          resultIsPackedBF16x2, vcvtResultVRegType, rewriter));
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  LogicalResult lowerPackedE2M1LaneStride2(
      VMIExtFOp op, ValueRange sourceParts, const ExtFPhysicalPlan &plan,
      OneToNPatternRewriter &rewriter) const {
    static constexpr StringRef kPacked2Parts[] = {"P0", "P2"};
    FailureOr<Value> mask = createSeedMask(op, plan.sourceType, rewriter);
    if (failed(mask)) {
      return failure();
    }
    ResultViewPlan viewPlan = buildResultViewPlan(plan.resultTypes, rewriter);
    return lowerFactor(op, rewriter, sourceParts, plan.resultTypes,
                       kPacked2Parts, 2, *mask, viewPlan.isPackedBF16x2,
                       viewPlan.vcvtResultType);
  }

  LogicalResult lowerFactor(
      VMIExtFOp op, OneToNPatternRewriter &rewriter, ValueRange sourceParts,
      ArrayRef<VRegType> resultTypes, ArrayRef<StringRef> parts,
      int64_t factor, Value mask, bool resultIsPackedBF16x2,
      VRegType vcvtResultVRegType) const {
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (int64_t partIndex = 0; partIndex < factor; ++partIndex) {
      for (auto [chunkIndex, sourcePart] : llvm::enumerate(sourceParts)) {
        VRegType resultType =
            resultTypes[partIndex * sourceParts.size() + chunkIndex];
        results.push_back(createVcvtResult(
            op.getLoc(), resultType, sourcePart, mask, /*rnd=*/nullptr,
            /*sat=*/nullptr, rewriter.getStringAttr(parts[partIndex]),
            resultIsPackedBF16x2, vcvtResultVRegType, rewriter));
      }
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  struct ExtFFactorPlan {
    ArrayRef<StringRef> parts;
    int64_t factor;
  };

  FailureOr<ExtFFactorPlan> buildFactorPlan(
      VMIExtFOp op, unsigned sourceBits, size_t sourcePartCount,
      size_t resultPartCount, OneToNPatternRewriter &rewriter) const {
    if (sourceBits == 16 && resultPartCount == 2 * sourcePartCount) {
      static constexpr StringRef kEvenOddParts[] = {"EVEN", "ODD"};
      return ExtFFactorPlan{ArrayRef<StringRef>(kEvenOddParts), 2};
    }
    if (sourceBits == 8 && resultPartCount == 4 * sourcePartCount) {
      static constexpr StringRef kPacked4Parts[] = {"P0", "P1", "P2", "P3"};
      return ExtFFactorPlan{ArrayRef<StringRef>(kPacked4Parts), 4};
    }
    return rewriter.notifyMatchFailure(
        op, "unsupported physical extf source/result width relation");
  }

  FailureOr<Value> createSeedMask(VMIExtFOp op, VRegType sourceType,
                                   OneToNPatternRewriter &rewriter) const {
    FailureOr<Value> mask =
        createAllTrueMaskForVReg(op.getLoc(), sourceType, rewriter);
    if (failed(mask)) {
      return rewriter.notifyMatchFailure(op, "failed to build extf seed mask");
    }
    return *mask;
  }

  LogicalResult lowerPhysicalExtF(
      VMIExtFOp op, ValueRange sourceParts, const ExtFPhysicalPlan &plan,
      VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
      OneToNPatternRewriter &rewriter) const {
    unsigned sourceBits =
        pto::getPTOStorageElemBitWidth(plan.sourceType.getElementType());
    // A packed bf16x2 physical result cannot be produced directly by
    // pto.vcvt (classifyVcvtElemType has no BF16x2 branch); the widest native
    // f4 conversion result element is bf16. Build the bf16 view type (2 bf16
    // lanes per bf16x2 lane) and reinterpret each vcvt result with a
    // physical-noop VbitcastOp, mirroring the source-side reinterpret in
    // OneToNVMITruncFOpPattern (viewVcvtSource).
    ResultViewPlan viewPlan = buildResultViewPlan(plan.resultTypes, rewriter);
    bool denseLaneStrideExtension =
        sourceLayout && resultLayout && sourceLayout.isContiguous() &&
        resultLayout.isContiguous() && resultLayout.getLaneStride() == 1 &&
        ((sourceBits == 16 && sourceLayout.getLaneStride() == 2) ||
         (sourceBits == 8 && sourceLayout.getLaneStride() == 4)) &&
        plan.resultTypes.size() == sourceParts.size();
    if (denseLaneStrideExtension) {
      StringRef part = sourceBits == 16 ? StringRef("EVEN") : StringRef("P0");
      FailureOr<Value> mask = createSeedMask(op, plan.sourceType, rewriter);
      if (failed(mask)) {
        return failure();
      }
      return lowerLaneStride(op, rewriter, sourceParts, plan.resultTypes, *mask,
                             part, viewPlan.isPackedBF16x2,
                             viewPlan.vcvtResultType);
    }

    // Packed f4E2M1x2 sources stored with lane_stride = 2 (UNPK_B8) have
    // valid bytes on the even lanes; P0 (lanes 0 mod 4) plus P2 (lanes 2 mod
    // 4) cover them while P1/P3 are zero-fill gaps, so widen through the
    // {P0, P2} part pair instead of the dense factor-4 selection.
    bool packedE2M1LaneStride2 =
        sourceLayout && sourceLayout.isContiguous() &&
        sourceLayout.getLaneStride() == 2 &&
        isa<pto::F4E2M1x2Type>(plan.sourceType.getElementType()) &&
        plan.resultTypes.size() == 2 * sourceParts.size();
    if (packedE2M1LaneStride2) {
      return lowerPackedE2M1LaneStride2(op, sourceParts, plan, rewriter);
    }

    FailureOr<ExtFFactorPlan> factorPlan = buildFactorPlan(
        op, sourceBits, sourceParts.size(), plan.resultTypes.size(), rewriter);
    if (failed(factorPlan)) {
      return failure();
    }
    FailureOr<Value> mask = createSeedMask(op, plan.sourceType, rewriter);
    if (failed(mask)) {
      return failure();
    }
    return lowerFactor(op, rewriter, sourceParts, plan.resultTypes,
                       factorPlan->parts, factorPlan->factor, *mask,
                       viewPlan.isPackedBF16x2,
                       viewPlan.vcvtResultType);
  }

public:

  LogicalResult
  matchAndRewrite(VMIExtFOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    FailureOr<VMIPhysicalConversionInput> input =
        getVMIPhysicalConversionInput(op, adaptor, *this->getTypeConverter());
    if (failed(input)) {
      return failure();
    }
    FailureOr<ExtFPhysicalPlan> plan =
        buildPhysicalPlan(op, input->sourceParts, input->resultTypes, rewriter);
    if (failed(plan)) {
      return failure();
    }
    VMILayoutAttr sourceLayout = input->sourceVMIType.getLayoutAttr();
    VMILayoutAttr resultLayout = input->resultVMIType.getLayoutAttr();
    return lowerPhysicalExtF(op, input->sourceParts, *plan, sourceLayout,
                             resultLayout, rewriter);
  }
};

static bool hasUnsupportedPackedTruncFConversion(Type sourceElementType,
                                                 Type resultElementType) {
  bool usesPackedCarrier =
      isVMIPackedFloatCarrierType(sourceElementType) ||
      isVMIPackedFloatCarrierType(resultElementType);
  return usesPackedCarrier &&
         !lookupVMIFpToFpContract(sourceElementType, resultElementType);
}

static bool hasGroupSlotTruncFLayouts(VMILayoutAttr sourceLayout,
                                      VMILayoutAttr resultLayout) {
  return sourceLayout && resultLayout && sourceLayout.isGroupSlots() &&
         resultLayout.isGroupSlots();
}

struct OneToNVMITruncFOpPattern : OneToNOpConversionPattern<VMITruncFOp> {
  using OneToNOpConversionPattern<VMITruncFOp>::OneToNOpConversionPattern;

private:
  struct TruncFPhysicalPlan {
    VRegType sourceType;
    SmallVector<VRegType> resultTypes;
    VRegType sourceViewType;
    unsigned sourceBits;
    unsigned resultBits;
    bool sourceIsPackedBF16x2;
  };

  struct TruncFNarrowingPlan {
    ArrayRef<StringRef> parts;
    int64_t sourceFactor;
    int64_t resultLaneStride;
  };

  FailureOr<VRegType> getUniformSourceType(
      VMITruncFOp op, ValueRange sourceParts,
      OneToNPatternRewriter &rewriter) const {
    if (sourceParts.empty()) {
      return rewriter.notifyMatchFailure(op,
                                         "truncf requires source chunks");
    }
    auto firstType = dyn_cast<VRegType>(sourceParts.front().getType());
    if (!firstType) {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical truncf source type");
    }
    for (Value sourcePart : sourceParts) {
      auto sourceType = dyn_cast<VRegType>(sourcePart.getType());
      if (!sourceType || sourceType != firstType) {
        return rewriter.notifyMatchFailure(
            op, "truncf source physical parts must have matching type");
      }
    }
    return firstType;
  }

  FailureOr<SmallVector<VRegType>> getUniformResultTypes(
      VMITruncFOp op, ArrayRef<Type> resultTypes,
      OneToNPatternRewriter &rewriter) const {
    SmallVector<VRegType> resultVRegTypes;
    resultVRegTypes.reserve(resultTypes.size());
    for (Type physicalResultType : resultTypes) {
      auto resultType = dyn_cast<VRegType>(physicalResultType);
      bool invalidType = !resultType ||
                         (!resultVRegTypes.empty() &&
                          resultType != resultVRegTypes.front());
      if (invalidType) {
        return rewriter.notifyMatchFailure(
            op, "unsupported physical truncf result type");
      }
      resultVRegTypes.push_back(resultType);
    }
    return resultVRegTypes;
  }

  FailureOr<TruncFPhysicalPlan> buildPhysicalPlan(
      VMITruncFOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      OneToNPatternRewriter &rewriter) const {
    if (resultTypes.empty()) {
      return rewriter.notifyMatchFailure(op, "truncf requires result chunks");
    }
    FailureOr<VRegType> sourceType =
        getUniformSourceType(op, sourceParts, rewriter);
    if (failed(sourceType)) {
      return failure();
    }
    unsigned sourceBits =
        pto::getPTOStorageElemBitWidth(sourceType->getElementType());
    if (sourceBits != 32 && sourceBits != 16) {
      return rewriter.notifyMatchFailure(
          op, "truncf source bit width must be 32 or 16");
    }
    FailureOr<SmallVector<VRegType>> resultVRegTypes =
        getUniformResultTypes(op, resultTypes, rewriter);
    if (failed(resultVRegTypes)) {
      return failure();
    }
    unsigned resultBits = pto::getPTOStorageElemBitWidth(
        resultVRegTypes->front().getElementType());
    if (resultBits == 0) {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical truncf result type");
    }
    bool sourceIsPackedBF16x2 =
        pto::isPTOBF16x2Type(sourceType->getElementType());
    VRegType sourceViewType = *sourceType;
    if (sourceIsPackedBF16x2) {
      sourceViewType = VRegType::get(
          rewriter.getContext(), sourceType->getElementCount() * 2,
          BFloat16Type::get(rewriter.getContext()));
    }
    return TruncFPhysicalPlan{*sourceType, std::move(*resultVRegTypes),
                              sourceViewType, sourceBits, resultBits,
                              sourceIsPackedBF16x2};
  }

  LogicalResult lowerDenseLaneStride(
      VMITruncFOp op, ValueRange sourceParts,
      ArrayRef<VRegType> resultTypes, StringRef part,
      bool sourceIsPackedBF16x2, VRegType vcvtSourceVRegType,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<Value> sourceMask =
        createAllTrueMaskForVReg(op.getLoc(), vcvtSourceVRegType, rewriter);
    if (failed(sourceMask)) {
      return rewriter.notifyMatchFailure(op, "failed to build truncf masks");
    }
    StringAttr rnd = rewriter.getStringAttr(
        getTruncFRoundMode(op, resultTypes.front().getElementType()));
    StringAttr sat = op->getAttrOfType<StringAttr>("saturate");
    StringAttr partAttr = rewriter.getStringAttr(part);
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [sourcePart, resultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      results.push_back(rewriter
                            .create<VcvtOp>(
                                op.getLoc(), resultType,
                                makeVcvtSourceView(
                                    op.getLoc(), sourcePart,
                                    sourceIsPackedBF16x2, vcvtSourceVRegType,
                                    rewriter),
                                *sourceMask, rnd, sat, partAttr)
                            .getResult());
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<Value> lowerGroupSlotTruncPart(
      VMITruncFOp op, Value sourcePart, Type physicalResultType,
      Value activeSlotMask, StringAttr sat,
      OneToNPatternRewriter &rewriter) const {
    auto sourceType = dyn_cast<VRegType>(sourcePart.getType());
    auto resultType = dyn_cast<VRegType>(physicalResultType);
    const bool invalidTypes =
        !sourceType || !sourceType.getElementType().isF32() || !resultType;
    if (invalidTypes) {
      (void)rewriter.notifyMatchFailure(
          op, "unsupported group-slot truncf physical type");
      return failure();
    }
    unsigned resultBits =
        pto::getPTOStorageElemBitWidth(resultType.getElementType());
    const bool unsupportedResultBits = resultBits != 16 && resultBits != 8;
    if (unsupportedResultBits) {
      (void)rewriter.notifyMatchFailure(
          op, "unsupported group-slot truncf physical type");
      return failure();
    }
    StringAttr part =
        rewriter.getStringAttr(resultBits == 16 ? "EVEN" : "P0");
    StringAttr rnd = rewriter.getStringAttr(
        getTruncFRoundMode(op, resultType.getElementType()));
    return rewriter
        .create<VcvtOp>(op.getLoc(), resultType, sourcePart, activeSlotMask,
                        rnd, sat, part)
        .getResult();
  }

  static Value makeVcvtSourceView(Location loc, Value sourcePart,
                                  bool sourceIsPackedBF16x2,
                                  VRegType vcvtSourceVRegType,
                                  OneToNPatternRewriter &rewriter) {
    if (!sourceIsPackedBF16x2) {
      return sourcePart;
    }
    if (auto vbc = sourcePart.getDefiningOp<VbitcastOp>()) {
      if (auto srcVReg = dyn_cast<VRegType>(vbc.getInput().getType());
          srcVReg && srcVReg.getElementType().isBF16()) {
        return vbc.getInput();
      }
    }
    return rewriter.create<VbitcastOp>(loc, vcvtSourceVRegType, sourcePart)
        .getResult();
  }

  LogicalResult lowerSameWidth(
      VMITruncFOp op, ValueRange sourceParts, ArrayRef<VRegType> resultTypes,
      VRegType sourceViewType, StringAttr sat,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<Value> sourceMask =
        createAllTrueMaskForVReg(op.getLoc(), sourceViewType, rewriter);
    if (failed(sourceMask)) {
      return rewriter.notifyMatchFailure(op, "failed to build truncf masks");
    }
    StringAttr rnd = rewriter.getStringAttr(
        getTruncFRoundMode(op, resultTypes.front().getElementType()));
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [sourcePart, resultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      results.push_back(rewriter
                            .create<VcvtOp>(op.getLoc(), resultType, sourcePart,
                                            *sourceMask, rnd, sat,
                                            /*part=*/nullptr)
                            .getResult());
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<Value> buildNarrowTruncResult(
      VMITruncFOp op, ValueRange sourceParts, VRegType resultType,
      ArrayRef<StringRef> allParts, int64_t chunkIndex, int64_t sourceFactor,
      int64_t resultLaneStride, VRegType sourceViewType,
      bool sourceIsPackedBF16x2, Value sourceMask, StringAttr rnd,
      StringAttr sat, OneToNPatternRewriter &rewriter) const {
    if (sourceFactor <= 0) {
      return rewriter.notifyMatchFailure(
          op, "narrow truncf requires a positive source factor");
    }
    int64_t safeSourceFactor = sourceFactor;
    FailureOr<Value> resultMask =
        createAllTrueMaskForVReg(op.getLoc(), resultType, rewriter);
    if (failed(resultMask)) {
      return failure();
    }
    SmallVector<Value> partials;
    partials.reserve(safeSourceFactor);
    for (int64_t partIndex = 0; partIndex < safeSourceFactor; ++partIndex) {
      Value sourcePart =
          sourceParts[partIndex * (sourceParts.size() / safeSourceFactor) +
                      chunkIndex];
      bool hasIndexedPart =
          partIndex * resultLaneStride < static_cast<int64_t>(allParts.size());
      StringRef part =
          hasIndexedPart ? allParts[partIndex * resultLaneStride]
                         : allParts[partIndex];
      partials.push_back(
          rewriter
              .create<VcvtOp>(
                  op.getLoc(), resultType,
                  makeVcvtSourceView(op.getLoc(), sourcePart,
                                     sourceIsPackedBF16x2, sourceViewType,
                                     rewriter),
                  sourceMask, rnd, sat, rewriter.getStringAttr(part))
              .getResult());
    }
    Value merged = partials.front();
    for (Value partial : llvm::drop_begin(partials)) {
      merged = rewriter
                   .create<VorOp>(op.getLoc(), resultType, merged, partial,
                                  *resultMask)
                   .getResult();
    }
    return merged;
  }

  LogicalResult lowerNarrow(
      VMITruncFOp op, ValueRange sourceParts,
      ArrayRef<VRegType> resultTypes, ArrayRef<StringRef> allParts,
      int64_t sourceFactor, int64_t resultLaneStride,
      VRegType sourceViewType, bool sourceIsPackedBF16x2,
      StringAttr rnd, StringAttr sat,
      OneToNPatternRewriter &rewriter) const {
    if (sourceFactor <= 0 || resultLaneStride <= 0 ||
        sourceParts.size() !=
            static_cast<size_t>(sourceFactor) * resultTypes.size()) {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical truncf source/result arity relation");
    }
    FailureOr<Value> sourceMask =
        createAllTrueMaskForVReg(op.getLoc(), sourceViewType, rewriter);
    if (failed(sourceMask)) {
      return rewriter.notifyMatchFailure(op, "failed to build truncf masks");
    }
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [chunkIndex, resultType] : llvm::enumerate(resultTypes)) {
      FailureOr<Value> result = buildNarrowTruncResult(
          op, sourceParts, resultType, allParts, chunkIndex, sourceFactor,
          resultLaneStride, sourceViewType, sourceIsPackedBF16x2, *sourceMask,
          rnd, sat, rewriter);
      if (failed(result)) {
        return rewriter.notifyMatchFailure(
            op, "failed to build truncf result mask");
      }
      results.push_back(*result);
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<TruncFNarrowingPlan> buildNarrowingPlan(
      VMITruncFOp op, unsigned sourceBits, unsigned resultBits,
      VMILayoutAttr resultLayout, size_t sourcePartCount,
      size_t resultPartCount, OneToNPatternRewriter &rewriter) const {
    ArrayRef<StringRef> parts;
    int64_t factor = 0;
    if (resultBits * 2 == sourceBits) {
      static constexpr StringRef kEvenOddParts[] = {"EVEN", "ODD"};
      parts = kEvenOddParts;
      factor = 2;
    } else if (resultBits * 4 == sourceBits) {
      static constexpr StringRef kPacked4Parts[] = {"P0", "P1", "P2", "P3"};
      parts = kPacked4Parts;
      factor = 4;
    } else {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical truncf source/result width relation");
    }
    int64_t resultLaneStride = resultLayout && resultLayout.isContiguous()
                                   ? resultLayout.getLaneStride()
                                   : 1;
    bool invalidResultLaneStride =
        resultLaneStride <= 0 || factor % resultLaneStride != 0;
    if (invalidResultLaneStride) {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical truncf result lane stride");
    }
    int64_t sourceFactor = factor / resultLaneStride;
    bool sourceArityMismatch =
        sourcePartCount != static_cast<size_t>(sourceFactor) * resultPartCount;
    if (sourceArityMismatch) {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical truncf source/result arity relation");
    }
    return TruncFNarrowingPlan{parts, sourceFactor, resultLaneStride};
  }

  LogicalResult lowerGroupSlotTrunc(
      VMITruncFOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
      VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
      OneToNPatternRewriter &rewriter) const {
    unsigned resultBits =
        pto::getPTOStorageElemBitWidth(resultVMIType.getElementType());
    bool invalidShape =
        sourceLayout.getNumGroups() != resultLayout.getNumGroups() ||
        sourceLayout.getSlots() != resultLayout.getSlots() ||
        (sourceLayout.getSlots() != 1 && sourceLayout.getSlots() != 8) ||
        !sourceVMIType.getElementType().isF32() ||
        (resultBits != 16 && resultBits != 8) ||
        sourceParts.size() != resultTypes.size();
    if (invalidShape) {
      return rewriter.notifyMatchFailure(op, "unsupported group-slot truncf shape");
    }
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    const char *activeSlotPattern =
        sourceLayout.getSlots() == 1 ? "PAT_VL1" : "PAT_VL8";
    FailureOr<Value> activeSlotMask = createPrefixMask(
        op.getLoc(), MaskType::get(rewriter.getContext(), "b32"),
        activeSlotPattern, rewriter);
    if (failed(activeSlotMask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to build group-slot truncf active slot mask");
    }
    StringAttr sat = op->getAttrOfType<StringAttr>("saturate");
    for (auto [sourcePart, physicalResultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      FailureOr<Value> result = lowerGroupSlotTruncPart(
          op, sourcePart, physicalResultType, *activeSlotMask, sat, rewriter);
      if (failed(result)) {
        return failure();
      }
      results.push_back(*result);
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<bool> tryLowerSameWidthTrunc(
      VMITruncFOp op, ValueRange sourceParts,
      const TruncFPhysicalPlan &physicalPlan, VMILayoutAttr sourceLayout,
      VMILayoutAttr resultLayout, OneToNPatternRewriter &rewriter) const {
    bool sameWidthContiguous =
        physicalPlan.sourceBits == physicalPlan.resultBits && sourceLayout &&
        resultLayout && sourceLayout.isContiguous() &&
        sourceLayout.getLaneStride() == 1 && resultLayout.isContiguous() &&
        resultLayout.getLaneStride() == 1 &&
        sourceParts.size() == physicalPlan.resultTypes.size();
    if (!sameWidthContiguous) {
      return false;
    }
    StringAttr sat = op->getAttrOfType<StringAttr>("saturate");
    if (failed(lowerSameWidth(op, sourceParts, physicalPlan.resultTypes,
                              physicalPlan.sourceViewType, sat, rewriter))) {
      return failure();
    }
    return true;
  }

  FailureOr<bool> tryLowerDenseLaneStrideTrunc(
      VMITruncFOp op, ValueRange sourceParts,
      const TruncFPhysicalPlan &physicalPlan, VMILayoutAttr sourceLayout,
      VMILayoutAttr resultLayout, OneToNPatternRewriter &rewriter) const {
    bool denseLaneStrideNarrowing =
        sourceLayout && resultLayout && sourceLayout.isContiguous() &&
        sourceLayout.getLaneStride() == 1 && resultLayout.isContiguous() &&
        resultLayout.getLaneStride() != 1 &&
        sourceParts.size() == physicalPlan.resultTypes.size();
    if (!denseLaneStrideNarrowing) {
      return false;
    }
    bool isEven32To16 = physicalPlan.resultBits == 16 &&
                          resultLayout.getLaneStride() == 2;
    bool isPacked32To8 = physicalPlan.resultBits == 8 &&
                           resultLayout.getLaneStride() == 4;
    bool isEven16To8 = physicalPlan.resultBits == 8 &&
                         resultLayout.getLaneStride() == 2;
    if (!isEven32To16 && !isPacked32To8 && !isEven16To8) {
      return rewriter.notifyMatchFailure(
          op, "unsupported dense lane_stride truncf result layout");
    }
    StringRef part = isPacked32To8 ? "P0" : "EVEN";
    if (failed(lowerDenseLaneStride(
            op, sourceParts, physicalPlan.resultTypes, part,
            physicalPlan.sourceIsPackedBF16x2, physicalPlan.sourceViewType,
            rewriter))) {
      return failure();
    }
    return true;
  }

  LogicalResult lowerNonGroupSlotTrunc(
      VMITruncFOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<TruncFPhysicalPlan> physicalPlan =
        buildPhysicalPlan(op, sourceParts, resultTypes, rewriter);
    if (failed(physicalPlan)) {
      return failure();
    }
    bool unsupportedGroupSlotLayout = sourceLayout && sourceLayout.isGroupSlots();
    if (unsupportedGroupSlotLayout) {
      return rewriter.notifyMatchFailure(
          op, "group-slot layout for non-f32 truncf not supported");
    }
    FailureOr<bool> sameWidth = tryLowerSameWidthTrunc(
        op, sourceParts, *physicalPlan, sourceLayout, resultLayout, rewriter);
    if (failed(sameWidth)) {
      return failure();
    }
    if (*sameWidth) {
      return success();
    }
    FailureOr<bool> denseLaneStride = tryLowerDenseLaneStrideTrunc(
        op, sourceParts, *physicalPlan, sourceLayout, resultLayout, rewriter);
    if (failed(denseLaneStride)) {
      return failure();
    }
    if (*denseLaneStride) {
      return success();
    }
    FailureOr<TruncFNarrowingPlan> narrowingPlan = buildNarrowingPlan(
        op, physicalPlan->sourceBits, physicalPlan->resultBits, resultLayout,
        sourceParts.size(),
        resultTypes.size(), rewriter);
    if (failed(narrowingPlan)) {
      return failure();
    }
    StringAttr rnd = rewriter.getStringAttr(
        getTruncFRoundMode(op, physicalPlan->resultTypes.front().getElementType()));
    StringAttr sat = op->getAttrOfType<StringAttr>("saturate");
    return lowerNarrow(op, sourceParts, physicalPlan->resultTypes,
                       narrowingPlan->parts,
                       narrowingPlan->sourceFactor,
                       narrowingPlan->resultLaneStride,
                       physicalPlan->sourceViewType,
                       physicalPlan->sourceIsPackedBF16x2, rnd, sat, rewriter);
  }

public:

  LogicalResult
  matchAndRewrite(VMITruncFOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto sourceVMIType = cast<VMIVRegType>(op.getSource().getType());
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    Type sourceElementType = sourceVMIType.getElementType();
    Type resultElementType = resultVMIType.getElementType();
    if (hasUnsupportedPackedTruncFConversion(sourceElementType,
                                             resultElementType)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported packed fp-to-fp truncf conversion");
    }
    ValueRange sourceParts = adaptor.getSource();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);

    VMILayoutAttr sourceLayout = sourceVMIType.getLayoutAttr();
    VMILayoutAttr resultLayout = resultVMIType.getLayoutAttr();
    if (hasGroupSlotTruncFLayouts(sourceLayout, resultLayout)) {
      return lowerGroupSlotTrunc(op, sourceParts, resultTypes, sourceLayout,
                                 resultLayout, sourceVMIType, resultVMIType,
                                 rewriter);
    }

    return lowerNonGroupSlotTrunc(op, sourceParts, resultTypes, sourceLayout,
                                  resultLayout, rewriter);
  }
};



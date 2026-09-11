// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#pragma once
//===- VMIToVPTOPatternInternals7.inc - VMIToVPTO internals -*- C++ -*-===//
//===----------------------------------------------------------------------===//

struct OneToNVMIChannelSplitOpPattern
    : OneToNOpConversionPattern<VMIChannelSplitOp> {
  using OneToNOpConversionPattern<VMIChannelSplitOp>::OneToNOpConversionPattern;

private:
  LogicalResult validateResultLayouts(
      VMIChannelSplitOp op, OneToNPatternRewriter &rewriter) const {
    return validateContiguousParts(
        op, op.getResults(), "channel_split requires contiguous result layouts",
        rewriter, isContiguousVMIVRegPart);
  }

public:

  LogicalResult
  matchAndRewrite(VMIChannelSplitOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    int64_t channels = op.getNumResults();
    bool unsupportedChannels = channels != 2 && channels != 4;
    if (unsupportedChannels) {
      return rewriter.notifyMatchFailure(
          op, "channel_split only supports 2 or 4 channels");
    }

    auto sourceType = cast<VMIVRegType>(op.getSource().getType());
    VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
    VMILayoutAttr channelLayout =
        VMILayoutAttr::getDeinterleaved(rewriter.getContext(), channels);
    bool invalidSourceLayout =
        !sourceLayout ||
        (!sourceLayout.isContiguous() && sourceLayout != channelLayout);
    if (invalidSourceLayout) {
      return rewriter.notifyMatchFailure(
          op,
          "channel_split requires contiguous or matching deinterleaved source "
          "layout");
    }
    if (failed(validateResultLayouts(op, rewriter))) {
      return failure();
    }

    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, *this->getTypeConverter());
    if (failed(maybe_resultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    FailureOr<SmallVector<Value>> results =
        materializeDataLayoutConversion(op, adaptor.getSource(), resultTypes,
                                        sourceLayout, channelLayout,
                                        sourceType.getElementType(), rewriter);
    if (failed(results)) {
      return failure();
    }

    replaceOpWithFlatConvertedValues(rewriter, op, *results, *this->getTypeConverter());
    return success();
  }
};

struct OneToNVMIChannelMergeOpPattern
    : OneToNOpConversionPattern<VMIChannelMergeOp> {
  using OneToNOpConversionPattern<VMIChannelMergeOp>::OneToNOpConversionPattern;

private:
  LogicalResult validateInputLayouts(
      VMIChannelMergeOp op, OneToNPatternRewriter &rewriter) const {
    return validateContiguousParts(
        op, op.getInputs(), "channel_merge requires contiguous input layouts",
        rewriter, isContiguousVMIVRegPart);
  }

  LogicalResult validateResultLayout(
      VMIChannelMergeOp op, VMILayoutAttr resultLayout,
      VMILayoutAttr channelLayout,
      OneToNPatternRewriter &rewriter) const {
    bool invalidResultLayout =
        !resultLayout ||
        (!resultLayout.isContiguous() && resultLayout != channelLayout);
    if (invalidResultLayout) {
      return rewriter.notifyMatchFailure(
          op,
          "channel_merge requires contiguous or matching deinterleaved result "
          "layout");
    }
    return success();
  }

  LogicalResult lowerChannelMerge(
      VMIChannelMergeOp op, OpAdaptor adaptor, VMILayoutAttr channelLayout,
      VMILayoutAttr resultLayout, Type resultElementType,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<SmallVector<Type>> maybeResultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybeResultTypes)) {
      return failure();
    }
    FailureOr<SmallVector<Value>> results = materializeDataLayoutConversion(
        op, flattenOneToNOperands(adaptor.getOperands()), *maybeResultTypes,
        channelLayout, resultLayout, resultElementType, rewriter);
    if (failed(results)) {
      return failure();
    }
    replaceOpWithFlatConvertedValues(rewriter, op, *results,
                                     *this->getTypeConverter());
    return success();
  }

public:

  LogicalResult
  matchAndRewrite(VMIChannelMergeOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    int64_t channels = op.getInputs().size();
    bool unsupportedChannels = channels != 2 && channels != 4;
    if (unsupportedChannels) {
      return rewriter.notifyMatchFailure(
          op, "channel_merge only supports 2 or 4 channels");
    }

    if (failed(validateInputLayouts(op, rewriter))) {
      return failure();
    }
    auto resultType = cast<VMIVRegType>(op.getResult().getType());
    VMILayoutAttr resultLayout = resultType.getLayoutAttr();
    auto channelLayout =
        VMILayoutAttr::getDeinterleaved(rewriter.getContext(), channels);
    if (failed(validateResultLayout(op, resultLayout, channelLayout, rewriter))) {
      return failure();
    }

    return lowerChannelMerge(op, adaptor, channelLayout, resultLayout,
                             resultType.getElementType(), rewriter);
  }
};

struct OneToNVMIShuffleOpPattern : OneToNOpConversionPattern<VMIShuffleOp> {
  using OneToNOpConversionPattern<VMIShuffleOp>::OneToNOpConversionPattern;

private:
  LogicalResult lowerForwarding(
      VMIShuffleOp op, OneToNPatternRewriter &rewriter, ValueRange sourceParts,
      ArrayRef<Type> resultTypes, ArrayRef<int64_t> sourceIndices) const {
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (int64_t sourceIndex : sourceIndices) {
      bool sourceOutOfBounds =
          sourceIndex < 0 || sourceIndex >= static_cast<int64_t>(sourceParts.size());
      if (sourceOutOfBounds) {
        return rewriter.notifyMatchFailure(
            op, "shuffle forwarding source part range is out of bounds");
      }
      results.push_back(sourceParts[sourceIndex]);
    }
    if (failed(verifyIdentityPartForwarding(op, results, resultTypes, rewriter))) {
      return failure();
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  LogicalResult lowerLane0Splat(
      VMIShuffleOp op, OneToNPatternRewriter &rewriter, ValueRange sourceParts,
      ArrayRef<Type> resultTypes, int64_t sourceIndex) const {
    bool sourceOutOfBounds =
        sourceIndex < 0 || sourceIndex >= static_cast<int64_t>(sourceParts.size());
    if (sourceOutOfBounds) {
      return rewriter.notifyMatchFailure(
          op, "shuffle lane0 splat source part range is out of bounds");
    }
    Value sourcePart = sourceParts[sourceIndex];
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (Type resultType : resultTypes) {
      auto sourceVRegType = dyn_cast<VRegType>(sourcePart.getType());
      auto resultVRegType = dyn_cast<VRegType>(resultType);
      bool invalidTypes = !sourceVRegType || !resultVRegType ||
                          sourceVRegType != resultVRegType;
      if (invalidTypes) {
        return rewriter.notifyMatchFailure(
            op, "shuffle lane0 splat requires matching physical vreg type");
      }
      FailureOr<Value> mask =
          createAllTrueMaskForVReg(op.getLoc(), resultVRegType, rewriter);
      if (failed(mask)) {
        return rewriter.notifyMatchFailure(
            op, "failed to create shuffle lane0 splat mask");
      }
      results.push_back(rewriter
                           .create<VdupOp>(op.getLoc(), resultType, sourcePart,
                                           *mask,
                                           rewriter.getStringAttr("LOWEST"))
                           .getResult());
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<Value> buildShuffleVselrResult(
      VMIShuffleOp op, ValueRange sourceParts, Type resultType,
      const ShuffleVselrPlan &plan,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<VRegType> sourceVRegType = getShuffleVselrSourceType(
        op, sourceParts, resultType, plan.sourceFlatIndex, rewriter);
    if (failed(sourceVRegType)) {
      return failure();
    }
    unsigned indexBits =
        pto::getPTOStorageElemBitWidth(sourceVRegType->getElementType());
    bool unsupportedIndexBits =
        indexBits != 8 && indexBits != 16 && indexBits != 32;
    if (unsupportedIndexBits) {
      return rewriter.notifyMatchFailure(
          op, "shuffle vselr requires 8/16/32-bit index elements");
    }
    auto indexElementType = IntegerType::get(rewriter.getContext(), indexBits);
    Type indexType = VRegType::get(rewriter.getContext(),
                                   sourceVRegType->getElementCount(),
                                   indexElementType);
    FailureOr<Value> base = createScalarOffsetConstant(
        op.getLoc(), indexElementType, plan.baseLane, rewriter);
    if (failed(base)) {
      return rewriter.notifyMatchFailure(
          op, "failed to materialize shuffle vselr index base");
    }
    StringAttr orderAttr =
        plan.descending ? rewriter.getStringAttr("DESC") : StringAttr{};
    Value indexVector =
        rewriter.create<VciOp>(op.getLoc(), indexType, *base, orderAttr)
            .getResult();
    return rewriter
        .create<VselrOp>(op.getLoc(), resultType,
                         sourceParts[plan.sourceFlatIndex], indexVector)
        .getResult();
  }

  FailureOr<VRegType> getShuffleVselrSourceType(
      VMIShuffleOp op, ValueRange sourceParts, Type resultType,
      int64_t sourceIndex, OneToNPatternRewriter &rewriter) const {
    bool sourceOutOfBounds =
        sourceIndex < 0 || sourceIndex >= static_cast<int64_t>(sourceParts.size());
    if (sourceOutOfBounds) {
      return rewriter.notifyMatchFailure(
          op, "shuffle vselr source part range is out of bounds");
    }
    auto sourceVRegType =
        dyn_cast<VRegType>(sourceParts[sourceIndex].getType());
    auto resultVRegType = dyn_cast<VRegType>(resultType);
    bool invalidTypes =
        !sourceVRegType || !resultVRegType ||
        sourceVRegType.getElementCount() != resultVRegType.getElementCount() ||
        sourceVRegType.getElementType() != resultVRegType.getElementType();
    if (invalidTypes) {
      return rewriter.notifyMatchFailure(
          op, "shuffle vselr source/result type mismatch");
    }
    return sourceVRegType;
  }

  LogicalResult lowerVselr(
      VMIShuffleOp op, OneToNPatternRewriter &rewriter, ValueRange sourceParts,
      ArrayRef<Type> resultTypes, ArrayRef<ShuffleVselrPlan> plans) const {
    bool arityMismatch = plans.size() != resultTypes.size();
    if (arityMismatch) {
      return rewriter.notifyMatchFailure(op, "shuffle vselr arity mismatch");
    }
    return lowerPointwisePhysicalParts(
        op, resultTypes, "shuffle vselr arity mismatch", rewriter,
        [&](int64_t index, Type resultType) -> FailureOr<Value> {
          return buildShuffleVselrResult(op, sourceParts, resultType,
                                         plans[index], rewriter);
        },
        *this->getTypeConverter());
  }

public:

  LogicalResult
  matchAndRewrite(VMIShuffleOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    return lowerWithConvertedResultTypes(
        op, 0, *this->getTypeConverter(), [&](ArrayRef<Type> resultTypes) {
          std::string splatReason;
          FailureOr<int64_t> splatSource =
              computeShuffleLane0SplatSourcePart(op, &splatReason);
          if (succeeded(splatSource)) {
            return lowerLane0Splat(op, rewriter, sourceParts, resultTypes,
                                   *splatSource);
          }

          std::string reason;
          FailureOr<SmallVector<int64_t>> sourceFlatIndices =
              computeShuffleForwardingSourceParts(op, &reason);
          if (succeeded(sourceFlatIndices)) {
            return lowerForwarding(op, rewriter, sourceParts, resultTypes,
                                   *sourceFlatIndices);
          }

          std::string vselrReason;
          FailureOr<SmallVector<ShuffleVselrPlan>> vselrPlans =
              computeShuffleVselrPlans(op, &vselrReason);
          if (failed(vselrPlans)) {
            return rewriter.notifyMatchFailure(
                op, Twine("shuffle vselr ") + vselrReason);
          }

          return lowerVselr(op, rewriter, sourceParts, resultTypes,
                            *vselrPlans);
        });
  }
};

Block *convertBranchDestBlock(Block *block, OneToNPatternRewriter &rewriter,
                              OneToNTypeConverter &typeConverter,
                              llvm::DenseMap<Block *, Block *> &converted) {
  auto [it, inserted] = converted.try_emplace(block, nullptr);
  if (!inserted) {
    return it->second;
  }

  OneToNTypeMapping argMapping(block->getArgumentTypes());
  if (failed(typeConverter.computeTypeMapping(block->getArgumentTypes(),
                                              argMapping)) ||
      !argMapping.hasNonIdentityConversion()) {
    it->second = block;
    return block;
  }

  Block *newBlock = rewriter.applySignatureConversion(block, argMapping);
  it->second = newBlock;
  return newBlock;
}

struct OneToNCFBranchOpPattern : OneToNOpConversionPattern<cf::BranchOp> {
  using OneToNOpConversionPattern<cf::BranchOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::BranchOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto *converter = getTypeConverter<OneToNTypeConverter>();
    llvm::DenseMap<Block *, Block *> convertedBlocks;
    Block *dest = convertBranchDestBlock(op.getDest(), rewriter, *converter,
                                         convertedBlocks);
    if (!adaptor.getOperandMapping().hasNonIdentityConversion() &&
        dest == op.getDest()) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<cf::BranchOp>(op, dest,
                                              adaptor.getFlatOperands());
    return success();
  }
};

struct OneToNCFCondBranchOpPattern
    : OneToNOpConversionPattern<cf::CondBranchOp> {
  using OneToNOpConversionPattern<cf::CondBranchOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::CondBranchOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto *converter = getTypeConverter<OneToNTypeConverter>();
    llvm::DenseMap<Block *, Block *> convertedBlocks;
    Block *trueDest = convertBranchDestBlock(op.getTrueDest(), rewriter,
                                             *converter, convertedBlocks);
    Block *falseDest = convertBranchDestBlock(op.getFalseDest(), rewriter,
                                              *converter, convertedBlocks);

    if (!adaptor.getOperandMapping().hasNonIdentityConversion() &&
        trueDest == op.getTrueDest() && falseDest == op.getFalseDest()) {
      return failure();
    }

    ValueRange condition = adaptor.getCondition();
    bool conditionArityMismatch = condition.size() != 1;
    if (conditionArityMismatch) {
      return rewriter.notifyMatchFailure(
          op, "condition converted to multiple values");
    }

    SmallVector<Value> trueOperands;
    SmallVector<Value> falseOperands;
    ValueRange flatOperands = adaptor.getFlatOperands();
    const OneToNTypeMapping &operandMapping = adaptor.getOperandMapping();
    unsigned operandIndex = 1;
    for (unsigned i = 0, e = op.getNumTrueOperands(); i < e; ++i) {
      llvm::append_range(trueOperands, operandMapping.getConvertedValues(
                                           flatOperands, operandIndex++));
    }
    for (unsigned i = 0, e = op.getNumFalseOperands(); i < e; ++i) {
      llvm::append_range(falseOperands, operandMapping.getConvertedValues(
                                            flatOperands, operandIndex++));
    }

    rewriter.replaceOpWithNewOp<cf::CondBranchOp>(op, condition.front(),
                                                  trueDest, trueOperands,
                                                  falseDest, falseOperands);
    return success();
  }
};

struct OneToNCFSwitchOpPattern : OneToNOpConversionPattern<cf::SwitchOp> {
  using OneToNOpConversionPattern<cf::SwitchOp>::OneToNOpConversionPattern;

private:
  static void collectSwitchOperandSegments(
      ArrayRef<int32_t> segmentSizes, ValueRange flatOperands,
      const OneToNTypeMapping &operandMapping, unsigned &operandIndex,
      SmallVectorImpl<SmallVector<Value>> &storage,
      SmallVectorImpl<ValueRange> &segments) {
    storage.reserve(segmentSizes.size());
    segments.reserve(segmentSizes.size());
    for (int32_t segmentSize : segmentSizes) {
      SmallVector<Value> operands;
      for (int32_t index = 0; index < segmentSize; ++index) {
        llvm::append_range(operands, operandMapping.getConvertedValues(
                                         flatOperands, operandIndex++));
      }
      storage.push_back(std::move(operands));
    }
    for (SmallVector<Value> &operands : storage) {
      segments.push_back(operands);
    }
  }

  static Block *convertSwitchDestination(
      Block *destination, OneToNPatternRewriter &rewriter,
      OneToNTypeConverter &converter, llvm::DenseMap<Block *, Block *> &blocks) {
    return convertBranchDestBlock(destination, rewriter, converter, blocks);
  }

  static bool switchDestinationsChanged(
      cf::SwitchOp op, Block *defaultDest, ArrayRef<Block *> caseDests) {
    if (defaultDest != op.getDefaultDestination()) {
      return true;
    }
    for (auto [oldDest, newDest] :
         llvm::zip(op.getCaseDestinations(), caseDests)) {
      if (oldDest != newDest) {
        return true;
      }
    }
    return false;
  }

public:
  LogicalResult
  matchAndRewrite(cf::SwitchOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto *converter = getTypeConverter<OneToNTypeConverter>();
    llvm::DenseMap<Block *, Block *> convertedBlocks;
    Block *defaultDest = convertSwitchDestination(
        op.getDefaultDestination(), rewriter, *converter, convertedBlocks);

    SmallVector<Block *> caseDests;
    caseDests.reserve(op.getCaseDestinations().size());
    for (Block *dest : op.getCaseDestinations()) {
      caseDests.push_back(convertSwitchDestination(dest, rewriter, *converter,
                                                   convertedBlocks));
    }

    bool changed = switchDestinationsChanged(op, defaultDest, caseDests) ||
                   adaptor.getOperandMapping().hasNonIdentityConversion();
    if (!changed) {
      return failure();
    }

    ValueRange flag = adaptor.getFlag();
    bool flagArityMismatch = flag.size() != 1;
    if (flagArityMismatch) {
      return rewriter.notifyMatchFailure(op,
                                         "flag converted to multiple values");
    }

    SmallVector<Value> defaultOperands;
    SmallVector<SmallVector<Value>> caseOperandStorage;
    SmallVector<ValueRange> caseOperands;
    ValueRange flatOperands = adaptor.getFlatOperands();
    const OneToNTypeMapping &operandMapping = adaptor.getOperandMapping();
    unsigned operandIndex = 1;
    for (unsigned i = 0, e = op.getDefaultOperands().size(); i < e; ++i) {
      llvm::append_range(defaultOperands, operandMapping.getConvertedValues(
                                              flatOperands, operandIndex++));
    }

    collectSwitchOperandSegments(op.getCaseOperandSegments(), flatOperands,
                                 operandMapping, operandIndex,
                                 caseOperandStorage, caseOperands);

    rewriter.replaceOpWithNewOp<cf::SwitchOp>(
        op, flag.front(), defaultDest, defaultOperands, op.getCaseValuesAttr(),
        caseDests, caseOperands);
    return success();
  }
};

struct OneToNSCFExecuteRegionOpPattern
    : OneToNOpConversionPattern<scf::ExecuteRegionOp> {
  using OneToNOpConversionPattern<
      scf::ExecuteRegionOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ExecuteRegionOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    const OneToNTypeMapping &resultMapping = adaptor.getResultMapping();
    for (unsigned i = 0, e = op->getNumResults(); i < e; ++i) {
      llvm::append_range(resultTypes, resultMapping.getConvertedTypes(i));
    }
    if (resultTypes == op->getResultTypes()) {
      return failure();
    }

    auto newOp =
        rewriter.create<scf::ExecuteRegionOp>(op.getLoc(), resultTypes);
    newOp->setAttrs(op->getAttrs());
    rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
                                newOp.getRegion().end());
    rewriter.replaceOp(op, newOp->getResults(), resultMapping);
    return success();
  }
};

struct OneToNSCFIndexSwitchOpPattern
    : OneToNOpConversionPattern<scf::IndexSwitchOp> {
  using OneToNOpConversionPattern<
      scf::IndexSwitchOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::IndexSwitchOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange arg = adaptor.getArg();
    bool selectorArityMismatch = arg.size() != 1;
    if (selectorArityMismatch) {
      return rewriter.notifyMatchFailure(
          op, "index_switch selector converted to multiple values");
    }

    SmallVector<Type> resultTypes;
    const OneToNTypeMapping &resultMapping = adaptor.getResultMapping();
    for (unsigned i = 0, e = op->getNumResults(); i < e; ++i) {
      llvm::append_range(resultTypes, resultMapping.getConvertedTypes(i));
    }
    if (resultTypes == op->getResultTypes()) {
      return failure();
    }

    auto newOp = rewriter.create<scf::IndexSwitchOp>(
        op.getLoc(), resultTypes, arg.front(), op.getCases(), op.getNumCases());
    newOp->setAttrs(op->getAttrs());
    rewriter.inlineRegionBefore(op.getDefaultRegion(), newOp.getDefaultRegion(),
                                newOp.getDefaultRegion().end());
    for (auto [srcRegion, dstRegion] :
         llvm::zip(op.getCaseRegions(), newOp.getCaseRegions())) {
      rewriter.inlineRegionBefore(srcRegion, dstRegion, dstRegion.end());
    }
    rewriter.replaceOp(op, newOp->getResults(), resultMapping);
    return success();
  }
};

static void populateVMIStructuralAndMemoryPatterns(
    VMIToVPTOTypeConverter &typeConverter, RewritePatternSet &patterns) {
  populateFuncTypeConversionPatterns(typeConverter, patterns);
  scf::populateSCFStructuralOneToNTypeConversions(typeConverter, patterns);
  patterns.add<OneToNCFBranchOpPattern, OneToNCFCondBranchOpPattern,
               OneToNCFSwitchOpPattern>(typeConverter, patterns.getContext());
  patterns.add<OneToNSCFExecuteRegionOpPattern, OneToNSCFIndexSwitchOpPattern>(
      typeConverter, patterns.getContext());
  patterns.add<OneToNVMIPackOpPattern, OneToNVMIUnpackOpPattern>(
      typeConverter, patterns.getContext());
  patterns.add<
      OneToNVMIEnsureLayoutOpPattern, OneToNVMIEnsureMaskLayoutOpPattern,
      OneToNVMIBroadcastOpPattern, OneToNVMIIotaOpPattern<VMIIotaOp>,
      OneToNVMIIotaOpPattern<VMIGroupIotaOp>,
      OneToNVMIConstantOpPattern, OneToNVMIConstantMaskOpPattern,
      OneToNVMICreateMaskOpPattern, OneToNVMICreateGroupMaskOpPattern,
      OneToNVMIBinaryOpPattern<VMIMaskAndOp, PandOp, /*IsMaskResult=*/true>,
      OneToNVMIBinaryOpPattern<VMIMaskOrOp, PorOp, /*IsMaskResult=*/true>,
      OneToNVMIBinaryOpPattern<VMIMaskXOrOp, PxorOp, /*IsMaskResult=*/true>,
      OneToNVMIUnaryOpPattern<VMIMaskNotOp, PnotOp, /*IsMaskResult=*/true>,
      OneToNVMILoadOpPattern,
      OneToNVMIDeinterleaveLoadOpPattern, OneToNVMIGroupLoadOpPattern,
      OneToNVMIGroupSlotLoadOpPattern, OneToNVMIStrideLoadOpPattern,
      OneToNVMIMaskedLoadOpPattern, OneToNVMIGatherOpPattern,
      OneToNVMIExpandLoadOpPattern, OneToNVMIStoreOpPattern,
      OneToNVMIInterleaveStoreOpPattern, OneToNVMIGroupStoreOpPattern,
      OneToNVMIMaskedStoreOpPattern, OneToNVMIStrideStoreOpPattern,
      OneToNVMIScatterOpPattern>(typeConverter, patterns.getContext());
}

static void populateVMIArithmeticPatterns(
    VMIToVPTOTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<OneToNVMIBinaryOpPattern<VMIAddFOp, VaddOp>,
      OneToNVMIBinaryOpPattern<VMIAddIOp, VaddOp>,
      OneToNVMIVaddcOpPattern, OneToNVMIVaddcsOpPattern,
      OneToNVMIBinaryOpPattern<VMISubFOp, VsubOp>,
      OneToNVMIBinaryOpPattern<VMISubIOp, VsubOp>,
      OneToNVMIBinaryOpPattern<VMIMulFOp, VmulOp>,
      OneToNVMIBinaryOpPattern<VMIMulIOp, VmulOp>,
      OneToNVMIVecScalarOpPattern<VMIAddSOp, VaddsOp>,
      OneToNVMIVecScalarOpPattern<VMIMulSOp, VmulsOp>,
      OneToNVMIVecScalarOpPattern<VMIMaxSOp, VmaxsOp>,
      OneToNVMIVecScalarOpPattern<VMIMinSOp, VminsOp>,
      OneToNVMIVecScalarOpPattern<VMIShlSOp, VshlsOp>,
      OneToNVMIVecScalarOpPattern<VMIShrSOp, VshrsOp>, OneToNVMIVmullOpPattern,
      OneToNVMIFmaOpPattern, OneToNVMIVexpdifOpPattern,
      OneToNVMIBinaryOpPattern<VMIDivFOp, VdivOp>,
      OneToNVMIBinaryOpPattern<VMIMinFOp, VminOp>,
      OneToNVMIBinaryOpPattern<VMIMinIOp, VminOp>,
      OneToNVMIBinaryOpPattern<VMIMaxFOp, VmaxOp>,
      OneToNVMIBinaryOpPattern<VMIMaxIOp, VmaxOp>,
      OneToNVMIUnaryOpPattern<VMINegFOp, VnegOp>,
      OneToNVMIUnaryOpPattern<VMINegIOp, VnegOp>,
      OneToNVMIUnaryOpPattern<VMIAbsFOp, VabsOp>,
      OneToNVMIUnaryOpPattern<VMIAbsIOp, VabsOp>,
      OneToNVMIUnaryOpPattern<VMISqrtOp, VsqrtOp>,
      OneToNVMIUnaryOpPattern<VMIExpOp, VexpOp>,
      OneToNVMIUnaryOpPattern<VMILnOp, VlnOp>,
      OneToNVMIUnaryOpPattern<VMIReluOp, VreluOp>,
      OneToNVMIBinaryOpPattern<VMIAndIOp, VandOp>,
      OneToNVMIBinaryOpPattern<VMIOrIOp, VorOp>,
      OneToNVMIBinaryOpPattern<VMIXOrIOp, VxorOp>,
      OneToNVMIShiftOpPattern<VMIShLIOp, VshlOp>,
      OneToNVMIShiftOpPattern<VMIShRUIOp, VshrOp>,
      OneToNVMIShiftOpPattern<VMIShRSIOp, VshrOp>,
      OneToNVMIUnaryOpPattern<VMINotOp, VnotOp>,
      OneToNVMICmpOpPattern<VMICmpFOp>, OneToNVMICmpOpPattern<VMICmpIOp>,
      OneToNVMISelectOpPattern, OneToNVMIVselrOpPattern,
      OneToNVMIActivePrefixIndexOpPattern,
      OneToNVMICompressOpPattern,
      OneToNVMICompressStoreOpPattern>(typeConverter, patterns.getContext());
}

static void populateVMIReductionAndConversionPatterns(
    VMIToVPTOTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<
      OneToNVMIReduceAddIOpPattern, OneToNVMIReduceAddFOpPattern,
      OneToNVMIGroupBroadcastOpPattern, OneToNVMIVdhistOpPattern,
      OneToNVMIVchistOpPattern,
      OneToNVMIReduceMinMaxOpPattern<VMIReduceMaxFOp, VcmaxOp, VmaxOp>,
      OneToNVMIReduceMinMaxOpPattern<VMIReduceMinFOp, VcminOp, VminOp>,
      OneToNVMIReduceMinMaxOpPattern<VMIReduceMaxIOp, VcmaxOp, VmaxOp>,
      OneToNVMIReduceMinMaxOpPattern<VMIReduceMinIOp, VcminOp, VminOp>,
      OneToNVMIExtFOpPattern, OneToNVMITruncFOpPattern,
      OneToNVMIExtIOpPattern<VMIExtSIOp>, OneToNVMIExtIOpPattern<VMIExtUIOp>,
      OneToNVMITruncIOpPattern, OneToNVMIFPToSIOpPattern,
      OneToNVMIFPToUIOpPattern,
      OneToNVMISIToFPOpPattern, OneToNVMIBitcastOpPattern,
      OneToNVMIInterleaveOpPattern<VMIVintlvOp, VintlvOp>,
      OneToNVMIInterleaveOpPattern<VMIVdintlvOp, VdintlvOp>,
      OneToNVMIChannelSplitOpPattern, OneToNVMIChannelMergeOpPattern,
      OneToNVMIShuffleOpPattern>(typeConverter, patterns.getContext());
  patterns.add<OneToNVMIGroupBroadcastLoadOpPattern>(
      typeConverter, patterns.getContext());
  patterns.add<
      OneToNVMIGroupReduceOpPattern<VMIGroupReduceAddFOp, VcgaddOp, VcaddOp,
                                    VaddOp>,
      OneToNVMIGroupReduceOpPattern<VMIGroupReduceAddIOp, VcgaddOp, VcaddOp,
                                    VaddOp>,
      OneToNVMIGroupReduceOpPattern<VMIGroupReduceMaxIOp, VcgmaxOp, VcmaxOp,
                                    VmaxOp>,
      OneToNVMIGroupReduceOpPattern<VMIGroupReduceMaxFOp, VcgmaxOp, VcmaxOp,
                                    VmaxOp>,
      OneToNVMIGroupReduceOpPattern<VMIGroupReduceMinIOp, VcgminOp, VcminOp,
                                    VminOp>,
      OneToNVMIGroupReduceOpPattern<VMIGroupReduceMinFOp, VcgminOp, VcminOp,
                                    VminOp>>(typeConverter,
                                             patterns.getContext());
  patterns.add<OneToNVMIEnsureMaskGranularityOpPattern>(
      typeConverter, patterns.getContext());
}

void populateVMIConversionPatterns(
    VMIToVPTOTypeConverter &typeConverter, RewritePatternSet &patterns) {
  populateVMIStructuralAndMemoryPatterns(typeConverter, patterns);
  populateVMIArithmeticPatterns(typeConverter, patterns);
  populateVMIReductionAndConversionPatterns(typeConverter, patterns);
}

static WalkResult verifyNoResidualCreateMask(Operation *op) {
  if (auto createMask = dyn_cast<VMICreateMaskOp>(op)) {
    if (!createMask.getActiveLanes().getDefiningOp<arith::ConstantOp>()) {
      createMask.emitError()
          << kVMIDiagUnsupportedPrefix
          << "dynamic pto.vmi.create_mask active_lanes could not be lowered "
             "by the current runtime predicate generation plan";
      return WalkResult::interrupt();
    }
  }
  return WalkResult::advance();
}

static WalkResult verifyNoResidualConstant(Operation *op) {
  if (auto constant = dyn_cast<VMIConstantOp>(op)) {
    auto denseAttr = dyn_cast<DenseElementsAttr>(constant.getValue());
    if (denseAttr && !denseAttr.isSplat()) {
      constant.emitError()
          << kVMIDiagUnsupportedPrefix
          << "non-splat pto.vmi.constant requires a vreg immediate or "
             "scratch materialization plan";
      return WalkResult::interrupt();
    }
  }
  return WalkResult::advance();
}

LogicalResult verifyNoResidualVMIIR(ModuleOp module) {
  WalkResult result = module.walk([](Operation *op) {
    if (WalkResult result = verifyNoResidualCreateMask(op);
        result.wasInterrupted()) {
      return result;
    }
    if (WalkResult result = verifyNoResidualConstant(op);
        result.wasInterrupted()) {
      return result;
    }
    bool hasResidualVMI = isVMIOp(op) || hasVMIType(op);
    if (hasResidualVMI) {
      op->emitError() << kVMIDiagResidualOpPrefix
                      << "failed to convert all VMI ops/types to VPTO";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

LogicalResult checkSupportedExtFShape(VMIExtFOp op,
                                      std::string *reason = nullptr) {
  VMILayoutSupport supports;
  if (failed(supports.getExtFSupport(op, reason))) {
    return failure();
  }
  return success();
}

LogicalResult checkSupportedTruncFShape(VMITruncFOp op,
                                        std::string *reason = nullptr) {
  VMILayoutSupport supports;
  if (failed(supports.getTruncFSupport(op, reason))) {
    return failure();
  }
  return success();
}

LogicalResult checkSupportedExtSIShape(VMIExtSIOp op,
                                       std::string *reason = nullptr) {
  VMILayoutSupport supports;
  if (failed(supports.getExtSISupport(op, reason))) {
    return failure();
  }
  return success();
}

LogicalResult checkSupportedExtUIShape(VMIExtUIOp op,
                                       std::string *reason = nullptr) {
  VMILayoutSupport supports;
  if (failed(supports.getExtUISupport(op, reason))) {
    return failure();
  }
  return success();
}

LogicalResult checkSupportedTruncIShape(VMITruncIOp op,
                                        std::string *reason = nullptr) {
  VMILayoutSupport supports;
  if (failed(supports.getTruncISupport(op, reason))) {
    return failure();
  }
  return success();
}

static LogicalResult checkSameWidthConversionArity(
    VMIVRegType sourceType, VMIVRegType resultType, StringRef conversionName,
    std::string *reason);

template <typename ShapeOp, typename ShapeCheck>
WalkResult verifySupportedShapeOp(ShapeOp op, ShapeCheck check,
                                  StringRef diagnostic);

// Shared shape-check prologue for fp<->int and int->fp conversions: requires
// both the source and result VMI vreg layouts to be assigned.
struct AssignedSourceResultLayouts {
  VMILayoutAttr sourceLayout;
  VMILayoutAttr resultLayout;
};

static FailureOr<AssignedSourceResultLayouts> getAssignedSourceResultLayouts(
    VMIVRegType sourceType, VMIVRegType resultType, std::string *reason) {
  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!sourceLayout || !resultLayout) {
    return emitFailure<AssignedSourceResultLayouts>(
        reason, "requires assigned source/result layouts");
  }
  return AssignedSourceResultLayouts{sourceLayout, resultLayout};
}

template <typename OpTy, typename ContractLookup>
LogicalResult checkSupportedFPToIntShape(OpTy op, StringRef conversionName,
                                         ContractLookup lookup,
                                         std::string *reason = nullptr) {
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  FailureOr<AssignedSourceResultLayouts> layouts =
      getAssignedSourceResultLayouts(sourceType, resultType, reason);
  if (failed(layouts)) {
    return failure();
  }

  Type srcElem = sourceType.getElementType();
  Type dstElem = resultType.getElementType();
  auto contract = lookup(srcElem, dstElem);
  if (!contract) {
    return emitLogicalFailure(reason, Twine("unsupported ") + conversionName +
                                          " conversion element type pair");
  }

  unsigned srcBits = pto::getPTOStorageElemBitWidth(srcElem);
  unsigned dstBits = pto::getPTOStorageElemBitWidth(dstElem);

  if (srcBits == dstBits) {
    // Same-width (f32→s32, f16→s16): layout equality + arity equality.
    if (failed(checkSameWidthConversionArity(sourceType, resultType,
                                             conversionName, reason))) {
      return failure();
    }
  } else {
    // Widen or narrow: use the cast-layout framework (same as extf/truncf).
    VMILayoutSupport layoutSupport;
    FailureOr<VMICastLayoutFact> fact =
        layoutSupport.getCastLayoutFactForLayouts(
            sourceType, resultType, layouts->sourceLayout,
            layouts->resultLayout, reason);
    if (failed(fact)) {
      return failure();
    }
  }

  return success();
}

static LogicalResult checkSameWidthConversionArity(
    VMIVRegType sourceType, VMIVRegType resultType, StringRef conversionName,
    std::string *reason) {
  bool layoutMismatch = sourceType.getLayoutAttr() != resultType.getLayoutAttr();
  if (layoutMismatch) {
    if (reason) {
      *reason = (Twine("same-width ") + conversionName +
                 " requires matching layouts")
                    .str();
    }
    return failure();
  }
  FailureOr<int64_t> sourceArity = getVMIPhysicalArity(sourceType);
  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  bool arityMismatch = failed(sourceArity) || failed(resultArity) ||
                       *sourceArity != *resultArity;
  if (arityMismatch) {
    if (reason) {
      *reason = (Twine("same-width ") + conversionName +
                 " requires matching physical arity")
                    .str();
    }
    return failure();
  }
  return success();
}

LogicalResult checkSupportedFPToSIShape(VMIFPToSIOp op,
                                        std::string *reason = nullptr) {
  return checkSupportedFPToIntShape(
      op, "fp-to-si",
      [](Type source, Type result) {
        return lookupVMIFpToSiContract(source, result);
      },
      reason);
}

LogicalResult checkSupportedFPToUIShape(VMIFPToUIOp op,
                                        std::string *reason = nullptr) {
  return checkSupportedFPToIntShape(
      op, "fp-to-ui",
      [](Type source, Type result) {
        return lookupVMIFpToUIContract(source, result);
      },
      reason);
}

LogicalResult checkSupportedSIToFPShape(VMISIToFPOp op,
                                        std::string *reason = nullptr) {
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  FailureOr<AssignedSourceResultLayouts> layouts =
      getAssignedSourceResultLayouts(sourceType, resultType, reason);
  if (failed(layouts)) {
    return failure();
  }
  unsigned srcBits = pto::getPTOStorageElemBitWidth(sourceType.getElementType());
  unsigned dstBits = pto::getPTOStorageElemBitWidth(resultType.getElementType());
  if (srcBits == 32 && dstBits == 32) {
    if (!resultType.getElementType().isF32()) {
      return emitLogicalFailure(reason, "requires f32 result element type");
    }
    if (failed(checkSameWidthConversionArity(sourceType, resultType,
                                             "si32->f32", reason))) {
      return failure();
    }
  } else if (srcBits == 8 && dstBits == 16) {
    if (!resultType.getElementType().isF16()) {
      return emitLogicalFailure(reason, "requires f16 result element type");
    }
    VMILayoutSupport layoutSupport;
    if (failed(layoutSupport.getCastLayoutFactForLayouts(
            sourceType, resultType, layouts->sourceLayout,
            layouts->resultLayout, reason))) {
      return failure();
    }
  } else {
    return emitLogicalFailure(reason, "supports only si32 -> f32 or si8 -> f16");
  }
  return success();
}

LogicalResult checkSupportedBitcastShape(VMIBitcastOp op, std::string *reason) {
  VMILayoutSupport supports;
  if (failed(supports.getBitcastSupport(op, reason))) {
    return failure();
  }
  return success();
}



struct ChannelShapePlan {
  int64_t channels;
  VMILayoutAttr expectedLayout;
};

template <typename ValidateFn>
static LogicalResult validateContiguousParts(
    Operation *op, ValueRange parts, StringRef failureMessage,
    OneToNPatternRewriter &rewriter, ValidateFn &&validate) {
  for (Value part : parts) {
    if (!validate(part)) {
      return rewriter.notifyMatchFailure(op, failureMessage);
    }
  }
  return success();
}

template <typename OpTy>
static LogicalResult lowerPhysicalBinaryWithCarryResults(
    OpTy op, SmallVectorImpl<Value> &results, SmallVectorImpl<Value> &carries,
    OneToNPatternRewriter &rewriter, TypeConverter &typeConverter) {
  results.append(carries);
  return replacePhysicalResults(rewriter, op, results, typeConverter);
}

template <typename OpTy>
static LogicalResult lowerBinaryPhysicalResults(
    OpTy op, SmallVectorImpl<Value> &results, OneToNPatternRewriter &rewriter,
    TypeConverter &typeConverter) {
  return replacePhysicalResults(rewriter, op, results, typeConverter);
}

template <typename OpTy, typename LowerFn>
static LogicalResult lowerPointwisePhysicalParts(
    OpTy op, ArrayRef<Type> resultTypes, StringRef arityMessage,
    OneToNPatternRewriter &rewriter, LowerFn &&lowerFn,
    TypeConverter &typeConverter) {
  SmallVector<Value> results;
  results.reserve(resultTypes.size());
  for (auto [index, resultType] : llvm::enumerate(resultTypes)) {
    FailureOr<Value> result = lowerFn(index, resultType);
    if (failed(result)) {
      return failure();
    }
    results.push_back(*result);
  }
  return replacePhysicalResults(rewriter, op, results, typeConverter);
}

static FailureOr<int64_t> getContiguousChannelInputArity(
    ValueRange inputs, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> FailureOr<int64_t> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  int64_t inputArity = 0;
  for (Value input : inputs) {
    auto inputType = dyn_cast<VMIVRegType>(input.getType());
    if (!inputType) {
      return fail("requires every input to be a VMI vreg");
    }
    VMILayoutAttr inputLayout = inputType.getLayoutAttr();
    if (!inputLayout || !inputLayout.isContiguous()) {
      return fail("requires every input layout to be contiguous");
    }
    FailureOr<int64_t> arity = getVMIPhysicalArity(inputType);
    if (failed(arity)) {
      return fail("requires computable input physical arity");
    }
    inputArity += *arity;
  }
  return inputArity;
}

static bool isContiguousVMIVRegPart(Value part) {
  auto partType = dyn_cast<VMIVRegType>(part.getType());
  VMILayoutAttr partLayout = partType.getLayoutAttr();
  return partType && partLayout && partLayout.isContiguous();
}

template <typename ChannelOp>
static FailureOr<ChannelShapePlan> buildChannelShapePlan(
    ChannelOp op, int64_t channels, StringRef operationName,
    std::string *reason) {
  auto fail = [&reason](const Twine &message)
      -> FailureOr<ChannelShapePlan> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  if (channels != 2 && channels != 4) {
    return fail(Twine("pto.vmi.") + operationName +
                " supports only 2 or 4 channels");
  }
  return ChannelShapePlan{
      channels, VMILayoutAttr::getDeinterleaved(op.getContext(), channels)};
}

static FailureOr<int64_t> checkChannelSplitResultShape(
    VMIChannelSplitOp op, int64_t sourceArity, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> FailureOr<int64_t> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  int64_t resultArity = 0;
  for (Value result : op.getResults()) {
    VMILayoutAttr resultLayout =
        cast<VMIVRegType>(result.getType()).getLayoutAttr();
    if (!resultLayout || !resultLayout.isContiguous()) {
      return fail("requires every result layout to be contiguous");
    }
    FailureOr<int64_t> arity =
        getVMIPhysicalArity(cast<VMIVRegType>(result.getType()));
    if (failed(arity)) {
      return fail("requires computable result physical arity");
    }
    resultArity += *arity;
  }
  if (sourceArity != resultArity) {
    return fail("requires source and result to have the same physical arity");
  }
  return resultArity;
}

static FailureOr<int64_t> checkChannelSplitSourceShape(
    VMIChannelSplitOp op, VMILayoutAttr expectedLayout,
    std::string *reason) {
  auto fail = [&reason](const Twine &message) -> FailureOr<int64_t> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  if (!sourceLayout) {
    return fail("requires assigned source layout");
  }
  bool invalidSourceLayout =
      !sourceLayout.isContiguous() && sourceLayout != expectedLayout;
  if (invalidSourceLayout) {
    return fail("requires source layout to be contiguous or matching deinterleaved channel layout");
  }
  FailureOr<int64_t> sourceArity = getVMIPhysicalArity(sourceType);
  if (failed(sourceArity)) {
    return fail("requires computable source physical arity");
  }
  return *sourceArity;
}

static LogicalResult checkChannelMergeResultShape(
    VMIChannelMergeOp op, VMILayoutAttr expectedLayout, int64_t inputArity,
    std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!resultLayout) {
    return fail("requires assigned result layout");
  }
  bool invalidResultLayout =
      !resultLayout.isContiguous() && resultLayout != expectedLayout;
  if (invalidResultLayout) {
    return fail("requires result layout to be contiguous or matching "
                "deinterleaved channel layout");
  }
  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  if (failed(resultArity)) {
    return fail("requires computable result physical arity");
  }
  if (*resultArity != inputArity) {
    return fail("requires source and result to have the same physical arity");
  }
  return success();
}

LogicalResult checkSupportedChannelSplitShape(VMIChannelSplitOp op,
                                              std::string *reason = nullptr) {
  FailureOr<ChannelShapePlan> plan =
      buildChannelShapePlan(op, op.getNumResults(), "channel_split", reason);
  if (failed(plan)) {
    return failure();
  }
  FailureOr<int64_t> sourceArity =
      checkChannelSplitSourceShape(op, plan->expectedLayout, reason);
  if (failed(sourceArity)) {
    return failure();
  }
  if (failed(checkChannelSplitResultShape(op, *sourceArity, reason))) {
    return failure();
  }

  return success();
}

LogicalResult checkSupportedChannelMergeShape(VMIChannelMergeOp op,
                                              std::string *reason = nullptr) {
  FailureOr<ChannelShapePlan> plan = buildChannelShapePlan(
      op, op.getInputs().size(), "channel_merge", reason);
  if (failed(plan)) {
    return failure();
  }
  FailureOr<int64_t> inputArity =
      getContiguousChannelInputArity(op.getInputs(), reason);
  if (failed(inputArity)) {
    return failure();
  }

  if (failed(checkChannelMergeResultShape(op, plan->expectedLayout,
                                          *inputArity, reason))) {
    return failure();
  }
  return success();
}

struct ActivePrefixIndexShapePlan {
  VMIMaskType maskType;
  VMIVRegType resultType;
};

static LogicalResult checkActivePrefixIndexLayouts(
    VMIMaskType maskType, VMIVRegType resultType, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  VMILayoutAttr maskLayout = maskType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!maskLayout || !resultLayout) {
    return fail("requires assigned mask and result layouts");
  }
  bool nonContiguousLayout = !maskLayout.isContiguous() ||
                             !resultLayout.isContiguous();
  if (nonContiguousLayout) {
    return fail("requires contiguous mask and result layouts");
  }
  return success();
}

static LogicalResult checkActivePrefixIndexPhysicalChunks(
    VMIMaskType maskType, VMIVRegType resultType, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  std::string resultFullReason;
  if (failed(checkFullDataPhysicalChunks(resultType, &resultFullReason))) {
    return fail(Twine("requires full result physical chunks so padding mask "
                      "lanes cannot affect the observable prefix; ") +
                resultFullReason);
  }
  std::string maskFullReason;
  if (failed(checkFullVMIPhysicalChunks(maskType, &maskFullReason))) {
    return fail(Twine("requires full mask physical chunks so padding mask "
                      "lanes cannot affect the observable prefix; ") +
                maskFullReason);
  }
  return success();
}

static LogicalResult checkActivePrefixIndexSingleChunk(
    VMIMaskType maskType, VMIVRegType resultType, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  FailureOr<int64_t> maskArity = getVMIPhysicalArity(maskType);
  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  bool missingArity = failed(maskArity) || failed(resultArity);
  if (missingArity) {
    return fail("requires computable mask and result physical arity");
  }
  if (*maskArity != 1 || *resultArity != 1) {
    return fail("requires a single physical chunk; multi-chunk prefix needs "
                "cross-chunk carry");
  }
  return success();
}

static FailureOr<ActivePrefixIndexShapePlan> buildActivePrefixIndexShapePlan(
    VMIActivePrefixIndexOp op, std::string *reason) {
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  if (failed(checkActivePrefixIndexLayouts(maskType, resultType, reason))) {
    return failure();
  }
  if (failed(checkActivePrefixIndexPhysicalChunks(maskType, resultType,
                                                  reason))) {
    return failure();
  }
  if (failed(checkActivePrefixIndexSingleChunk(maskType, resultType, reason))) {
    return failure();
  }

  return ActivePrefixIndexShapePlan{maskType, resultType};
}

LogicalResult
checkSupportedActivePrefixIndexShape(VMIActivePrefixIndexOp op,
                                     std::string *reason = nullptr) {
  FailureOr<ActivePrefixIndexShapePlan> plan =
      buildActivePrefixIndexShapePlan(op, reason);
  if (failed(plan)) {
    return failure();
  }
  return success();
}

struct CompressPhysicalShapePlan {
  VMIVRegType valueType;
  VMIMaskType maskType;
};

static FailureOr<CompressPhysicalShapePlan> buildCompressPhysicalShapePlan(
    VMIVRegType valueType, VMIMaskType maskType, StringRef fullChunkSuffix,
    StringRef arityMessage, std::string *reason) {
  FailureOr<AssignedValueMaskLayouts> layouts =
      getAssignedValueMaskLayouts(valueType, maskType, reason);
  if (failed(layouts)) {
    return failure();
  }
  bool nonContiguousInputs =
      !layouts->value.isContiguous() || !layouts->mask.isContiguous();
  if (nonContiguousInputs) {
    return emitFailure<CompressPhysicalShapePlan>(
        reason, "requires contiguous value and mask layouts");
  }
  std::string fullChunkReason;
  if (failed(checkFullDataPhysicalChunks(valueType, &fullChunkReason))) {
    return emitFailure<CompressPhysicalShapePlan>(
        reason, Twine("requires full physical chunks so padding mask lanes ") +
                    fullChunkSuffix + "; " + fullChunkReason);
  }
  FailureOr<int64_t> valueArity = getVMIPhysicalArity(valueType);
  FailureOr<int64_t> maskArity = getVMIPhysicalArity(maskType);
  bool invalidArity = failed(valueArity) || failed(maskArity) ||
                      *valueArity != 1 || *maskArity != 1;
  if (invalidArity) {
    return emitFailure<CompressPhysicalShapePlan>(reason, arityMessage);
  }
  return CompressPhysicalShapePlan{valueType, maskType};
}

static LogicalResult checkSupportedCompressResultShape(
    VMIVRegType resultType, StringRef layoutMessage,
    StringRef computableArityMessage, StringRef arityMessage,
    std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!resultLayout) {
    return fail(layoutMessage);
  }
  if (!resultLayout.isContiguous()) {
    return fail("requires contiguous result layouts");
  }
  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  if (failed(resultArity)) {
    return fail(computableArityMessage);
  }
  if (*resultArity != 1) {
    return fail(arityMessage);
  }
  return success();
}

static LogicalResult checkCompressStoreDestination(VMICompressStoreOp op,
                                                   std::string *reason) {
  if (isa<PtrType>(op.getDestination().getType())) {
    return success();
  }
  if (reason) {
    *reason = "requires !pto.ptr destination because pto.vstur is pointer-only";
  }
  return failure();
}

LogicalResult checkSupportedCompressShape(VMICompressOp op,
                                          std::string *reason = nullptr) {
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr maskLayout = maskType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!sourceLayout || !maskLayout || !resultLayout) {
    return emitLogicalFailure(
        reason, "requires assigned source, mask, and result layouts");
  }
  if (!sourceLayout.isContiguous() || !maskLayout.isContiguous() ||
      !resultLayout.isContiguous()) {
    return emitLogicalFailure(
        reason, "requires contiguous source, mask, and result layouts");
  }
  FailureOr<CompressPhysicalShapePlan> plan = buildCompressPhysicalShapePlan(
      sourceType, maskType,
      "cannot be squeezed into the result",
      "requires a single physical chunk; multi-chunk compress needs cross-"
      "chunk compaction",
      reason);
  if (failed(plan)) {
    return failure();
  }
  return checkSupportedCompressResultShape(
      resultType, "requires assigned result layouts",
      "requires computable source, mask, and result physical arity",
      "requires a single physical chunk; multi-chunk compress needs "
      "cross-chunk compaction",
      reason);
}

LogicalResult checkSupportedCompressStoreShape(
    VMICompressStoreOp op,
    std::string *reason = nullptr) {
  auto valueType = cast<VMIVRegType>(op.getValue().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  FailureOr<CompressPhysicalShapePlan> plan = buildCompressPhysicalShapePlan(
      valueType, maskType, "cannot be squeezed into memory",
      "requires a single physical chunk; multi-chunk compress_store needs "
      "cross-chunk compaction and SQZN state planning",
      reason);
  if (failed(plan)) {
    return failure();
  }

  return checkCompressStoreDestination(op, reason);
}

struct ReducePhysicalShapePlan {
  VMILayoutAttr sourceLayout;
  VMILayoutAttr maskLayout;
  VMILayoutAttr resultLayout;
  int64_t sourceArity;
  int64_t resultArity;
};

template <typename OpTy>
static LogicalResult checkReduceLayouts(OpTy op, VMILayoutAttr *sourceLayout,
                                        VMILayoutAttr *maskLayout,
                                        VMILayoutAttr *resultLayout,
                                        std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  *sourceLayout = sourceType.getLayoutAttr();
  *maskLayout = maskType.getLayoutAttr();
  *resultLayout = resultType.getLayoutAttr();
  if (!*sourceLayout || !*maskLayout || !*resultLayout) {
    return fail("requires assigned source, mask, and result layouts");
  }
  bool nonContiguousLayout = !sourceLayout->isContiguous() ||
                             !maskLayout->isContiguous() ||
                             !resultLayout->isContiguous();
  if (nonContiguousLayout) {
    return fail("requires contiguous source, mask, and result layouts");
  }
  return success();
}

static LogicalResult checkReducePhysicalArity(
    VMIVRegType sourceType, VMIMaskType maskType, VMIVRegType resultType,
    int64_t *sourceArity, int64_t *resultArity, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  FailureOr<int64_t> sourceParts = getVMIPhysicalArity(sourceType);
  FailureOr<int64_t> maskParts = getVMIPhysicalArity(maskType);
  FailureOr<int64_t> resultParts = getVMIPhysicalArity(resultType);
  bool cannotComputeArity = failed(sourceParts) || failed(maskParts) ||
                            failed(resultParts);
  if (cannotComputeArity) {
    return fail("requires computable physical arity");
  }
  bool mismatchedInputArity = *sourceParts < 1 || *maskParts != *sourceParts;
  if (mismatchedInputArity) {
    return fail("requires source and mask physical arity to match and be "
                "non-empty");
  }
  if (*resultParts != 1) {
    return fail("requires one result physical chunk");
  }
  *sourceArity = *sourceParts;
  *resultArity = *resultParts;
  return success();
}

template <typename OpTy>
static LogicalResult checkReduceSourceChunks(OpTy op, VMIVRegType sourceType,
                                             std::string *reason) {
  std::string fullChunkReason;
  if (succeeded(checkFullDataPhysicalChunks(sourceType, &fullChunkReason))) {
    return success();
  }
  if (reason) {
    *reason = (Twine("requires full source physical chunks so padding lanes "
                    "do not participate in the reduction; ") +
               fullChunkReason)
                  .str();
  }
  return failure();
}

template <typename OpTy>
static FailureOr<ReducePhysicalShapePlan> buildReducePhysicalShapePlan(
    OpTy op, std::string *reason) {
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  VMILayoutAttr sourceLayout;
  VMILayoutAttr maskLayout;
  VMILayoutAttr resultLayout;
  if (failed(checkReduceLayouts(op, &sourceLayout, &maskLayout, &resultLayout,
                                reason))) {
    return failure();
  }

  if (failed(checkReduceSourceChunks(op, sourceType, reason))) {
    return failure();
  }

  int64_t sourceArity;
  int64_t resultArity;
  if (failed(checkReducePhysicalArity(sourceType, maskType, resultType,
                                      &sourceArity, &resultArity, reason))) {
    return failure();
  }

  return ReducePhysicalShapePlan{sourceLayout, maskLayout, resultLayout,
                                 sourceArity, resultArity};
}

template <typename OpTy>
LogicalResult
checkSupportedReduceShape(OpTy op, bool requiresReassoc,
                          std::string *reason = nullptr) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  if (requiresReassoc && !op->hasAttr("reassoc")) {
    return fail("requires reassoc attr for pair-wise floating-point vcadd");
  }
  FailureOr<ReducePhysicalShapePlan> plan =
      buildReducePhysicalShapePlan(op, reason);
  if (failed(plan)) {
    return failure();
  }
  return success();
}

template <typename OpTy>
LogicalResult
checkSupportedGroupReduceShape(OpTy op, std::string *reason = nullptr) {
  VMILayoutSupport supports;
  if constexpr (std::is_same_v<OpTy, VMIGroupReduceAddFOp>) {
    if (succeeded(supports.getGroupReduceAddFSupport(op, reason))) {
      return success();
    }
  } else if constexpr (std::is_same_v<OpTy, VMIGroupReduceMaxFOp>) {
    if (succeeded(supports.getGroupReduceMaxFSupport(op, reason))) {
      return success();
    }
  } else if constexpr (std::is_same_v<OpTy, VMIGroupReduceMaxIOp>) {
    if (succeeded(supports.getGroupReduceMaxISupport(op, reason))) {
      return success();
    }
  } else if constexpr (std::is_same_v<OpTy, VMIGroupReduceMinFOp>) {
    if (succeeded(supports.getGroupReduceMinFSupport(op, reason))) {
      return success();
    }
  } else if constexpr (std::is_same_v<OpTy, VMIGroupReduceMinIOp>) {
    if (succeeded(supports.getGroupReduceMinISupport(op, reason))) {
      return success();
    }
  } else {
    if (succeeded(supports.getGroupReduceAddISupport(op, reason))) {
      return success();
    }
  }
  return failure();
}

struct GroupBroadcastShapePlan {
  VMILayoutAttr sourceLayout;
  VMILayoutAttr resultLayout;
  int64_t numGroups;
  int64_t lanesPerPart;
  int64_t groupSize;
  int64_t resultFactor;
};

static LogicalResult checkGroupBroadcastLogicalContract(
    VMIGroupBroadcastOp op, VMIVRegType sourceType, VMIVRegType resultType,
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
    int64_t numGroups, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  bool mismatchedElementTypes =
      sourceType.getElementType() != resultType.getElementType();
  if (mismatchedElementTypes) {
    return fail("requires source/result element type to match");
  }
  bool missingLayouts = !sourceLayout || !resultLayout;
  if (missingLayouts) {
    return fail("requires assigned source/result layouts");
  }
  bool invalidGroupCount = numGroups <= 0;
  if (invalidGroupCount) {
    return fail("requires positive num_groups");
  }
  int64_t safeNumGroups = numGroups;
  bool sourceLaneCountMismatch = sourceType.getElementCount() != numGroups;
  if (sourceLaneCountMismatch) {
    return fail("requires source lane count to match num_groups");
  }
  bool resultLaneCountMismatch =
      resultType.getElementCount() % safeNumGroups != 0;
  if (resultLaneCountMismatch) {
    return fail("requires num_groups to evenly divide result lane count");
  }
  bool sourceLayoutMismatch =
      !sourceLayout.isGroupSlots() || sourceLayout.getNumGroups() != numGroups;
  if (sourceLayoutMismatch) {
    return fail("requires matching num_groups source layout");
  }
  bool resultUsesGroupSlots = resultLayout.isGroupSlots();
  if (resultUsesGroupSlots) {
    return fail("requires dense result layout");
  }
  bool unsupportedSlots = sourceLayout.getSlots() > 0 &&
                          sourceLayout.getSlots() != 8 &&
                          sourceLayout.getSlots() != 1;
  if (unsupportedSlots) {
    return fail("supports only slots=8 or slots=1 group_broadcast source "
                "layouts");
  }
  return success();
}

static LogicalResult checkGroupBroadcastSupportContract(
    VMIGroupBroadcastOp op, std::string *reason) {
  VMILayoutSupport supports;
  std::string supportReason;
  if (failed(supports.getGroupBroadcastSupport(op, &supportReason))) {
    if (reason) {
      *reason = supportReason;
    }
    return failure();
  }
  return success();
}

static FailureOr<GroupBroadcastShapePlan> buildGroupBroadcastShapePlan(
    VMIGroupBroadcastOp op, std::string *reason) {
  auto fail = [&reason](const Twine &message)
      -> FailureOr<GroupBroadcastShapePlan> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  int64_t numGroups = op.getNumGroupsAttr().getInt();
  if (failed(checkGroupBroadcastLogicalContract(
          op, sourceType, resultType, sourceLayout, resultLayout, numGroups,
          reason))) {
    return failure();
  }
  if (failed(checkGroupBroadcastSupportContract(op, reason))) {
    return failure();
  }

  FailureOr<int64_t> lanesPerPart =
      getDataLanesPerPart(sourceType.getElementType());
  FailureOr<int64_t> resultLanesPerPart =
      getDataLanesPerPart(resultType.getElementType());
  if (failed(lanesPerPart) || failed(resultLanesPerPart) ||
      *lanesPerPart != *resultLanesPerPart) {
    return fail("requires matching physical lanes per part");
  }
  FailureOr<int64_t> groupSize = getGroupSizeFromNumGroups(
      resultType, numGroups, reason);
  if (failed(groupSize)) {
    return failure();
  }
  if (*lanesPerPart % *groupSize != 0 && *groupSize % *lanesPerPart != 0) {
    return fail("requires derived group size to divide or be a multiple of "
                "physical lanes per part");
  }

  FailureOr<int64_t> resultFactor = getDataLayoutFactor(resultType);
  if (failed(resultFactor)) {
    return fail("requires known result layout factor");
  }
  return GroupBroadcastShapePlan{sourceLayout, resultLayout, numGroups,
                                 *lanesPerPart, *groupSize, *resultFactor};
}

static LogicalResult checkGroupBroadcastResultShape(
    VMIVRegType resultType, VMILayoutAttr resultLayout, int64_t groupSize,
    int64_t lanesPerPart, int64_t resultFactor, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  bool laneStridedDense =
      resultLayout.isDense() && resultLayout.getLaneStride() > 1;
  if (!laneStridedDense) {
    std::string fullChunkReason;
    if (failed(checkFullDataPhysicalChunks(resultType, &fullChunkReason))) {
      return fail(Twine("requires full result physical chunks; ") +
                  fullChunkReason);
    }
  }
  if (resultFactor == 1) {
    return success();
  }
  FailureOr<int64_t> resultBlockElems =
      getVMILayoutBlockElems(resultType);
  bool blockFragmentSmallGroup =
      resultLayout.isBlockDeinterleaved() && succeeded(resultBlockElems) &&
      groupSize < lanesPerPart && lanesPerPart % *resultBlockElems == 0;
  bool deinterleavedSmallGroup =
      resultLayout.isDeinterleaved() &&
      groupSize < lanesPerPart && groupSize >= resultFactor &&
      groupSize % resultFactor == 0 &&
      lanesPerPart % (groupSize / resultFactor) == 0;
  if (blockFragmentSmallGroup || deinterleavedSmallGroup) {
    return success();
  }
  int64_t logicalSpanPerResultChunk = lanesPerPart * resultFactor;
  bool groupSpansMultipleChunks =
      groupSize < lanesPerPart || groupSize % logicalSpanPerResultChunk != 0;
  if (groupSpansMultipleChunks) {
    return fail("deinterleaved result requires every physical result chunk to "
                "stay within one logical group");
  }
  return success();
}

LogicalResult checkSupportedGroupBroadcastShape(
    VMIGroupBroadcastOp op,
    std::string *reason = nullptr) {
  FailureOr<GroupBroadcastShapePlan> plan =
      buildGroupBroadcastShapePlan(op, reason);
  if (failed(plan)) {
    return failure();
  }
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  return checkGroupBroadcastResultShape(
      resultType, plan->resultLayout, plan->groupSize, plan->lanesPerPart,
      plan->resultFactor, reason);
}

LogicalResult checkSupportedVdhistShape(VMIVdhistOp op,
                                       std::string *reason = nullptr) {
  VMILayoutSupport supports;
  if (succeeded(supports.getVdhistSupport(op, reason))) {
    return success();
  }
  return failure();
}

LogicalResult checkSupportedVchistShape(VMIVchistOp op,
                                       std::string *reason = nullptr) {
  VMILayoutSupport supports;
  if (succeeded(supports.getVchistSupport(op, reason))) {
    return success();
  }
  return failure();
}

struct VmullShapePlan {
  VMIVRegType dataType;
  VMILayoutAttr layout;
  int64_t arity;
};

static LogicalResult validateVmullDataLayout(
    VMIVRegType aType, VMIVRegType bType, VMIVRegType lowType,
    VMIVRegType highType, VMIMaskType maskType, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  if (aType != bType || aType != lowType || aType != highType) {
    return fail("requires identical a, b, low, and high VMI vreg types");
  }
  VMILayoutAttr layout = aType.getLayoutAttr();
  if (!layout) {
    return fail("requires an assigned data layout");
  }
  bool supportedLayout =
      layout.getLaneStride() == 1 &&
      (layout.isContiguous() ||
       (layout.isDeinterleaved() &&
        (layout.getFactor() == 2 || layout.getFactor() == 4)));
  if (!supportedLayout) {
    return fail("requires contiguous or deinterleaved factor 2/4 layout with "
                "lane_stride=1");
  }
  bool maskLayoutMismatch = maskType.getLayoutAttr() != layout;
  if (maskLayoutMismatch) {
    return fail("requires the mask and all four data values to share one layout");
  }
  bool unsupportedMaskGranularity = maskType.getGranularity() != "b32";
  if (unsupportedMaskGranularity) {
    return fail("requires b32 mask granularity");
  }
  return success();
}

static FailureOr<VmullShapePlan> validateVmullLogicalShape(
    VMIVmullOp op, VMIVRegType aType, VMIVRegType bType,
    VMIVRegType lowType, VMIVRegType highType, VMIMaskType maskType,
    std::string *reason) {
  auto fail = [&reason](const Twine &message)
      -> FailureOr<VmullShapePlan> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  auto elementType = dyn_cast<IntegerType>(aType.getElementType());
  bool unsupportedElementType =
      !elementType || elementType.getWidth() != 32 ||
      (!elementType.isSignless() && !elementType.isUnsigned());
  if (unsupportedElementType) {
    return fail("requires element type to be exactly i32 or ui32");
  }
  int64_t lanes = aType.getElementCount();
  if (lanes != 64 && lanes != 128 && lanes != 256) {
    return fail("requires logical lane count 64, 128, or 256");
  }
  if (failed(validateVmullDataLayout(aType, bType, lowType, highType, maskType,
                                     reason))) {
    return failure();
  }
  return VmullShapePlan{aType, aType.getLayoutAttr(), 0};
}

static FailureOr<VmullShapePlan> buildVmullShapePlan(
    VMIVmullOp op, std::string *reason) {
  auto aType = cast<VMIVRegType>(op.getA().getType());
  auto bType = cast<VMIVRegType>(op.getB().getType());
  auto lowType = cast<VMIVRegType>(op.getLow().getType());
  auto highType = cast<VMIVRegType>(op.getHigh().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());

  FailureOr<VmullShapePlan> logical = validateVmullLogicalShape(
      op, aType, bType, lowType, highType, maskType, reason);
  if (failed(logical)) {
    return failure();
  }

  FailureOr<int64_t> aArity = getVMIPhysicalArity(aType);
  FailureOr<int64_t> bArity = getVMIPhysicalArity(bType);
  FailureOr<int64_t> lowArity = getVMIPhysicalArity(lowType);
  FailureOr<int64_t> highArity = getVMIPhysicalArity(highType);
  FailureOr<int64_t> maskArity = getVMIPhysicalArity(maskType);
  bool missingArity =
      failed(aArity) || failed(bArity) || failed(lowArity) ||
      failed(highArity) || failed(maskArity) || *aArity < 1;
  if (missingArity) {
    if (reason) {
      *reason = "requires computable non-empty physical arity on every port";
    }
    return failure();
  }
  bool arityMismatch =
      *aArity != *bArity || *aArity != *lowArity || *aArity != *highArity ||
      *aArity != *maskArity;
  if (arityMismatch) {
    if (reason) {
      *reason = "requires matching physical arity on a, b, mask, low, and high";
    }
    return failure();
  }
  return VmullShapePlan{aType, logical->layout, *aArity};
}



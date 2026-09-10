// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMI_ops_group.cpp - VMI group/reduce/histogram ops -----------------------===//
//===----------------------------------------------------------------------===//

#include "VMIInternal.h"

using namespace mlir;
using namespace mlir::pto;

LogicalResult VMIGroupIotaOp::verify() {
  auto resultType = cast<VMIVRegType>(getResult().getType());
  Type elementType = resultType.getElementType();
  if (!isVMIIotaElementType(elementType)) {
    return emitOpError("requires result element type to be integer 8/16/32 "
                       "or f16/f32");
  }
  if (!isCompatibleVMIScalarForSemanticType(elementType, getBase().getType())) {
    return emitOpError("requires base type to match result element type");
  }
  if (std::optional<StringRef> order = getOrder()) {
    if (*order != "ASC" && *order != "DESC") {
      return emitOpError("requires order to be ASC or DESC");
    }
  }

  int64_t numGroups = getGroupAttr().getInt();
  if (numGroups <= 1) {
    return emitOpError("requires group greater than one");
  }
  if (resultType.getElementCount() % numGroups != 0) {
    return emitOpError("requires group to evenly divide result logical lane "
                       "count");
  }
  int64_t groupSize = resultType.getElementCount() / numGroups;
  FailureOr<int64_t> lanesPerPart = getDataLanesPerPart(elementType);
  if (succeeded(lanesPerPart) && groupSize % *lanesPerPart != 0 &&
      *lanesPerPart % groupSize != 0) {
    return emitOpError("requires group_size to divide or be a multiple of "
                       "physical lanes per part (")
           << *lanesPerPart << ")";
  }
  if (VMILayoutAttr layout = resultType.getLayoutAttr();
      layout && !layout.isContiguous()) {
    return emitOpError("requires contiguous result layout");
  }
  return success();
}

LogicalResult VMIReduceAddIOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIIntegerLikeType(sourceType.getElementType())) {
    return emitOpError("requires integer-like VMI source element type");
  }
  auto sourceIntegerType = dyn_cast<IntegerType>(sourceType.getElementType());
  if (!sourceIntegerType || sourceIntegerType.getWidth() != mlir::pto::kValue32) {
    return emitOpError("requires 32-bit integer source element type");
  }
  if (sourceType.getElementType() != resultType.getElementType()) {
    return emitOpError("requires source and result element types to match");
  }
  if (resultType.getElementCount() != 1) {
    return emitOpError("requires result to be a 1-lane VMI vector");
  }
  return verifyMaskMatchesData(getOperation(), maskType, sourceType);
}

LogicalResult VMIReduceAddFOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!getOperation()->hasAttr("reassoc")) {
    return emitOpError(
        "requires reassoc attr because VPTO vcadd performs pair-wise "
        "floating-point reduction");
  }
  if (!isVMIFloatLikeType(sourceType.getElementType())) {
    return emitOpError("requires floating-point-like VMI source element type");
  }
  if (!isVMIF16OrF32Type(sourceType.getElementType())) {
    return emitOpError("requires f16 or f32 source element type");
  }
  if (sourceType.getElementType() != resultType.getElementType()) {
    return emitOpError("requires source and result element types to match");
  }
  if (resultType.getElementCount() != 1) {
    return emitOpError("requires result to be a 1-lane VMI vector");
  }
  return verifyMaskMatchesData(getOperation(), maskType, sourceType);
}

template <typename OpTy> static LogicalResult verifyReduceMinMaxFOp(OpTy op) {
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  if (!isVMIFloatLikeType(sourceType.getElementType())) {
    return op.emitOpError(
        "requires floating-point-like VMI source element type");
  }
  if (!isVMIF16OrF32Type(sourceType.getElementType())) {
    return op.emitOpError("requires f16 or f32 source element type");
  }
  if (sourceType.getElementType() != resultType.getElementType()) {
    return op.emitOpError("requires source and result element types to match");
  }
  if (resultType.getElementCount() != 1) {
    return op.emitOpError("requires result to be a 1-lane VMI vector");
  }
  return verifyMaskMatchesData(op.getOperation(), maskType, sourceType);
}

LogicalResult VMIReduceMaxFOp::verify() { return verifyReduceMinMaxFOp(*this); }

LogicalResult VMIReduceMinFOp::verify() { return verifyReduceMinMaxFOp(*this); }

template <typename OpTy> static LogicalResult verifyReduceMinMaxIOp(OpTy op) {
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  auto sourceIntegerType = dyn_cast<IntegerType>(sourceType.getElementType());
  if (!sourceIntegerType ||
      !isVMIAnyI8I16I32Type(sourceType.getElementType())) {
    return op.emitOpError(
        "requires 8-bit, 16-bit, or 32-bit integer source element type");
  }
  if (sourceType.getElementType() != resultType.getElementType()) {
    return op.emitOpError("requires source and result element types to match");
  }
  if (resultType.getElementCount() != 1) {
    return op.emitOpError("requires result to be a 1-lane VMI vector");
  }
  return verifyMaskMatchesData(op.getOperation(), maskType, sourceType);
}

LogicalResult VMIReduceMaxIOp::verify() { return verifyReduceMinMaxIOp(*this); }

LogicalResult VMIReduceMinIOp::verify() { return verifyReduceMinMaxIOp(*this); }

template <typename OpTy>
static LogicalResult verifyGroupReduceCommon(OpTy op, VMIVRegType sourceType,
                                             VMIVRegType resultType,
                                             VMIMaskType maskType) {
  int64_t numGroups = op.getNumGroupsAttr().getInt();
  if (resultType.getElementCount() != numGroups) {
    return op.emitOpError(
        "requires result logical lane count to match num_groups");
  }
  if (sourceType.getElementType() != resultType.getElementType()) {
    return op.emitOpError("requires source and result element types to match");
  }
  if (auto sourceLayout = sourceType.getLayoutAttr()) {
    bool supportedSourceLayout =
        sourceLayout.isContiguous() ||
        (sourceLayout.isDenseSplit() &&
         (sourceLayout.getFactor() == mlir::pto::kValue2 ||
          sourceLayout.getFactor() == mlir::pto::kValue4));
    if (!supportedSourceLayout) {
      return op.emitOpError(
          "requires layout-assigned source to use contiguous layout or "
          "deinterleaved=2/4 or block_deinterleaved=2/4 layout");
    }
  }
  if (auto resultLayout = resultType.getLayoutAttr()) {
    if (!resultLayout.isGroupSlots() ||
        resultLayout.getNumGroups() != numGroups) {
      return op.emitOpError() << "requires layout-assigned result to use "
                                 "#pto.vmi.layout<num_groups = "
                              << numGroups << ">";
    }
  }
  if (failed(verifyMaskMatchesData(op.getOperation(), maskType, sourceType))) {
    return failure();
  }
  return verifyNumGroups(op.getOperation(), sourceType, numGroups);
}

template <typename OpTy>
static LogicalResult verifyGroupReduceFloatOp(OpTy op, bool requiresReassoc) {
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  if (requiresReassoc && !op->hasAttr("reassoc")) {
    return op.emitOpError(
        "requires reassoc attr because grouped lowering uses pair-wise "
        "floating-point reductions");
  }
  if (!isVMIFloatLikeType(sourceType.getElementType())) {
    return op.emitOpError(
        "requires floating-point-like VMI source element type");
  }
  if (!isVMIF16OrF32Type(sourceType.getElementType())) {
    return op.emitOpError("requires f16 or f32 source element type");
  }
  return verifyGroupReduceCommon(op, sourceType, resultType, maskType);
}

LogicalResult VMIGroupReduceAddFOp::verify() {
  return verifyGroupReduceFloatOp(*this, /*requiresReassoc=*/true);
}

LogicalResult VMIGroupReduceMaxFOp::verify() {
  return verifyGroupReduceFloatOp(*this, /*requiresReassoc=*/false);
}

LogicalResult VMIGroupReduceMinFOp::verify() {
  return verifyGroupReduceFloatOp(*this, /*requiresReassoc=*/false);
}

template <typename OpTy>
static LogicalResult verifyGroupReduceIntegerOp(OpTy op) {
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  if (!isVMIIntegerLikeType(sourceType.getElementType())) {
    return op.emitOpError("requires integer-like VMI source element type");
  }
  auto intType = dyn_cast<IntegerType>(sourceType.getElementType());
  if (!intType || !isVMIAnyI8I16I32Type(sourceType.getElementType())) {
    return op.emitOpError(
        "requires 8-bit, 16-bit, or 32-bit integer source element type");
  }
  return verifyGroupReduceCommon(op, sourceType, resultType, maskType);
}

LogicalResult VMIGroupReduceAddIOp::verify() {
  return verifyGroupReduceIntegerOp(*this);
}

LogicalResult VMIGroupReduceMaxIOp::verify() {
  return verifyGroupReduceIntegerOp(*this);
}

LogicalResult VMIGroupReduceMinIOp::verify() {
  return verifyGroupReduceIntegerOp(*this);
}

//===----------------------------------------------------------------------===//
// Group 5: vcadd / vcmax / vcmin verifiers
//===----------------------------------------------------------------------===//

LogicalResult VMIGroupBroadcastOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  int64_t numGroups = getNumGroupsAttr().getInt();
  if (numGroups <= 0) {
    return emitOpError("requires num_groups to be positive");
  }
  if (sourceType.getElementCount() != numGroups) {
    return emitOpError(
        "requires source logical lane count to match num_groups");
  }
  if (resultType.getElementCount() % numGroups != 0) {
    return emitOpError(
        "requires num_groups to evenly divide result logical lane count");
  }
  if (sourceType.getElementType() != resultType.getElementType()) {
    return emitOpError("requires source and result element types to match");
  }
  if (auto sourceLayout = sourceType.getLayoutAttr()) {
    if (!sourceLayout.isGroupSlots() ||
        sourceLayout.getNumGroups() != numGroups) {
      return emitOpError() << "requires layout-assigned source to use "
                              "#pto.vmi.layout<num_groups = "
                           << numGroups << ">";
    }
  }
  if (auto resultLayout = resultType.getLayoutAttr()) {
    if (resultLayout.isGroupSlots()) {
      return emitOpError(
          "requires layout-assigned result to use a dense VMI layout");
    }
  }
  return verifyNumGroups(getOperation(), resultType, numGroups);
}

template <typename OpTy>
static LogicalResult verifyVMIHistogramLayouts(OpTy op, VMIVRegType accType,
                                               VMIVRegType sourceType,
                                               VMIVRegType resultType,
                                               VMIMaskType maskType) {
  if (auto accLayout = accType.getLayoutAttr()) {
    if (!accLayout.isContiguous()) {
      return op.emitOpError("requires layout-assigned acc to use contiguous "
                            "layout");
    }
  }
  if (auto sourceLayout = sourceType.getLayoutAttr()) {
    if (!sourceLayout.isContiguous()) {
      return op.emitOpError("requires layout-assigned source to use contiguous "
                            "layout");
    }
  }
  if (auto resultLayout = resultType.getLayoutAttr()) {
    if (!resultLayout.isContiguous()) {
      return op.emitOpError("requires layout-assigned result to use "
                            "contiguous layout");
    }
  }
  if (auto maskLayout = maskType.getLayoutAttr()) {
    if (!maskLayout.isContiguous()) {
      return op.emitOpError("requires layout-assigned mask to use contiguous "
                            "layout");
    }
    if (maskType.getGranularity() != "b8") {
      return op.emitOpError("requires layout-assigned mask granularity b8");
    }
  }
  return success();
}

template <typename OpTy> static LogicalResult verifyVMIHistogramOp(OpTy op) {
  auto accType = cast<VMIVRegType>(op.getAcc().getType());
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());

  auto accElemType = dyn_cast<IntegerType>(accType.getElementType());
  auto sourceElemType = dyn_cast<IntegerType>(sourceType.getElementType());
  int64_t bins = accType.getElementCount();
  if (!accElemType || accElemType.getWidth() != mlir::pto::kValue16 ||
      !matchesVMIIntSemantics(accElemType, VMIIntSignSemantics::Unsigned) ||
      (bins != mlir::pto::kValue128 && bins != mlir::pto::kValue256)) {
    return op.emitOpError("requires acc type to be "
                          "!pto.vmi.vreg<128x{ui16|i16}> (Bin_N0-only) or "
                          "!pto.vmi.vreg<256x{ui16|i16}>");
  }
  if (resultType.getElementCount() != bins) {
    return op.emitOpError("requires result element count to match acc "
                          "(bins must be identical)");
  }
  if (resultType.getLayoutAttr() != accType.getLayoutAttr()) {
    return op.emitOpError("requires result layout attribute to match acc");
  }
  auto resultElemType = dyn_cast<IntegerType>(resultType.getElementType());
  if (!resultElemType || resultElemType.getWidth() != mlir::pto::kValue16 ||
      !matchesVMIIntSemantics(resultElemType, VMIIntSignSemantics::Unsigned)) {
    return op.emitOpError("requires result element type to be ui16 or i16 "
                          "(interpreted as unsigned)");
  }
  if (!sourceElemType || sourceElemType.getWidth() != mlir::pto::kValue8 ||
      !matchesVMIIntSemantics(sourceElemType, VMIIntSignSemantics::Unsigned)) {
    return op.emitOpError("requires source type to be "
                          "!pto.vmi.vreg<Nx{ui8|i8}> (interpreted as unsigned)");
  }
  if (maskType.getElementCount() != sourceType.getElementCount()) {
    return op.emitOpError("requires mask logical lane count to match source");
  }
  return verifyVMIHistogramLayouts(op, accType, sourceType, resultType,
                                   maskType);
}

LogicalResult VMIVdhistOp::verify() { return verifyVMIHistogramOp(*this); }

LogicalResult VMIVchistOp::verify() { return verifyVMIHistogramOp(*this); }

LogicalResult VMIChannelSplitOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  if (getResults().size() < mlir::pto::kValue2) {
    return emitOpError("requires at least two channel results");
  }
  auto firstResultType = cast<VMIVRegType>(getResults().front().getType());
  if (sourceType.getElementCount() !=
      static_cast<int64_t>(getResults().size()) *
          firstResultType.getElementCount()) {
    return emitOpError("requires source lane count to equal result count times "
                       "per-channel lane count");
  }
  for (Value result : getResults()) {
    auto resultType = cast<VMIVRegType>(result.getType());
    if (resultType.getElementCount() != firstResultType.getElementCount() ||
        resultType.getElementType() != sourceType.getElementType()) {
      return emitOpError("requires every channel result to have equal lane "
                         "count and source element type");
    }
  }
  if (isLayoutAssigned(sourceType) ||
      llvm::any_of(getResults(), [](Value result) {
        return isLayoutAssigned(cast<VMIVRegType>(result.getType()));
      })) {
    return verifyChannelSplitLayout(getOperation(), sourceType, getResults());
  }
  return success();
}

LogicalResult VMIChannelMergeOp::verify() {
  if (getInputs().size() < mlir::pto::kValue2) {
    return emitOpError("requires at least two channel inputs");
  }
  auto firstInputType = cast<VMIVRegType>(getInputs().front().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  for (Value input : getInputs()) {
    auto inputType = cast<VMIVRegType>(input.getType());
    if (inputType.getElementCount() != firstInputType.getElementCount() ||
        inputType.getElementType() != firstInputType.getElementType()) {
      return emitOpError("requires all channel inputs to have the same lane "
                         "count and element type");
    }
  }
  if (resultType.getElementCount() != static_cast<int64_t>(getInputs().size()) *
                                          firstInputType.getElementCount() ||
      resultType.getElementType() != firstInputType.getElementType()) {
    return emitOpError(
        "requires result lane count and element type to match merged channels");
  }
  if (isLayoutAssigned(resultType) ||
      llvm::any_of(getInputs(), [](Value input) {
        return isLayoutAssigned(cast<VMIVRegType>(input.getType()));
      })) {
    return verifyChannelMergeLayout(getOperation(), resultType, getInputs());
  }
  return success();
}

LogicalResult verifyVCReductionElementAndMask(Operation *op,
                                                     VMIVRegType sourceType,
                                                     VMIMaskType maskType,
                                                     bool &isFloat) {
  Type elemTy = sourceType.getElementType();
  if (failed(verifyBF16x2ComputeElementType(op, elemTy))) {
    return failure();
  }
  isFloat = isVMIFloatLikeType(elemTy);
  bool isInt = isVMIIntegerLikeType(elemTy);
  if (!isFloat && !isInt) {
    return op->emitOpError("requires integer-like or floating-point-like VMI "
                           "source element type");
  }
  return verifyMaskMatchesData(op, maskType, sourceType);
}

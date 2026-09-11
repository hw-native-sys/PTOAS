// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMI_ops_arith.cpp - VMI arithmetic/compare ops -----------------------===//
//===----------------------------------------------------------------------===//

#include "VMIInternal.h"

using namespace mlir;
using namespace mlir::pto;

LogicalResult VMIConstantOp::verify() {
  auto resultType = cast<VMIVRegType>(getResult().getType());
  auto denseAttr = dyn_cast<DenseElementsAttr>(getValue());
  if (!denseAttr) {
    return emitOpError("requires dense elements constant attribute");
  }
  if (denseAttr.getElementType() != resultType.getElementType()) {
    return emitOpError(
        "requires dense constant element type to match result element type");
  }
  if (denseAttr.getNumElements() != resultType.getElementCount()) {
    return emitOpError("requires dense constant element count to match result "
                       "logical lane count");
  }
  return success();
}

// Batch11: broadcast scalar mode shared
template <typename OpTy>
static LogicalResult verifyVMIBroadcastScalarMode(OpTy op, Type valueType,
                                                  VMIVRegType resultType) {
  if (valueType == resultType.getElementType()) {
    return success();
  }
  if (auto vregType = dyn_cast<VMIVRegType>(valueType)) {
    if (vregType.getElementCount() != 1) {
      return op.emitOpError("requires VMI vector input to have one logical lane");
    }
    if (vregType.getElementType() != resultType.getElementType()) {
      return op.emitOpError("requires VMI vector input element type to match "
                         "result element type");
    }
    return success();
  }
  return op.emitOpError("requires scalar or VMI vector input element type to "
                     "match result element type");
}

LogicalResult VMIBroadcastOp::verify() {
  auto resultType = cast<VMIVRegType>(getResult().getType());
  Type valueType = getValue().getType();
  return verifyVMIBroadcastScalarMode(*this, valueType, resultType);
}

// Batch6: 双胞胎 op 共用校验模板(谓词/中段检查由调用方提供, 消息逐字节不变)
static bool isSupportedVCmpPredicate(llvm::StringRef cmp);
template <typename OpTy>
static LogicalResult verifyVMIIotaLikeOp(OpTy op) {
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  Type elementType = resultType.getElementType();
  if (!isVMIIotaElementType(elementType)) {
    return op.emitOpError("requires result element type to be integer 8/16/32 "
                          "or f16/f32");
  }
  if (!isCompatibleVMIScalarForSemanticType(elementType, op.getBase().getType())) {
    return op.emitOpError("requires base type to match result element type");
  }
  if (std::optional<StringRef> order = op.getOrder()) {
    if (*order != "ASC" && *order != "DESC") {
      return op.emitOpError("requires order to be ASC or DESC");
    }
  }
  return success();
}

template <typename OpTy>
static LogicalResult verifyVMIInterleaveLikeOp(OpTy op) {
  auto lhsType = cast<VMIVRegType>(op.getLhs().getType());
  auto rhsType = cast<VMIVRegType>(op.getRhs().getType());
  auto lowType = cast<VMIVRegType>(op.getLow().getType());
  auto highType = cast<VMIVRegType>(op.getHigh().getType());
  if (failed(verifyAllSameVRegShapeAndLayoutPresence(
          op.getOperation(), {lhsType, rhsType, lowType, highType},
          /*requireSameElement=*/true))) {
    return failure();
  }
  if (failed(verifyVMIPmodeMask(op.getOperation(),
                                cast<VMIMaskType>(op.getMask().getType()),
                                lhsType, op.getPmode()))) {
    return failure();
  }
  return success();
}

template <typename OpTy>
static LogicalResult verifyVMIMaskLogicOp(OpTy op) {
  if (isa<VMIMaskType>(op.getLhs().getType())) {
    if (!op.getMask().empty()) {
      return op.emitOpError("mask logic op does not support predication mask");
    }
    if (auto pmode = op.getPmode()) {
      return op.emitOpError("mask logic op does not support pmode");
    }
    auto lhsType = cast<VMIMaskType>(op.getLhs().getType());
    auto rhsType = cast<VMIMaskType>(op.getRhs().getType());
    auto resultType = cast<VMIMaskType>(op.getResult().getType());
    return verifyAllSameMaskShapeLayoutAndGranularity(
        op.getOperation(), {lhsType, rhsType, resultType});
  }
  auto lhsType = cast<VMIVRegType>(op.getLhs().getType());
  auto rhsType = cast<VMIVRegType>(op.getRhs().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  if (!isVMIIntegerLikeType(lhsType.getElementType())) {
    return op.emitOpError("requires integer-like VMI element type");
  }
  if (failed(verifyElementwiseVRegOp(op.getOperation(), lhsType, rhsType, resultType))) {
    return failure();
  }
  if (failed(verifyVMIVariadicPmodeMask(op.getOperation(), op.getMask(),
                                        resultType, op.getPmode()))) {
    return failure();
  }
  return success();
}

template <typename OpTy, typename ExtraCheck>
static LogicalResult verifyVMICmpCommon(OpTy op, VMIVRegType dataType,
                                        VMIMaskType seedType,
                                        VMIMaskType resultType,
                                        ExtraCheck extraCheck) {
  Type eltTy = dataType.getElementType();
  if (failed(verifyBF16x2ComputeElementType(op.getOperation(), eltTy))) {
    return failure();
  }
  if (!isVMII8I16I32OrF16BF16F32Type(eltTy)) {
    return op.emitOpError(
        "requires i8, i16, i32, f16, bf16, or f32 VMI element type");
  }
  bool usesOrderedFloatPredicate = op.getCmp().starts_with("o");
  if (isVMIIntegerLikeType(eltTy) && usesOrderedFloatPredicate) {
    return op.emitOpError("requires integer compare predicate eq/ne/lt/le/gt/ge; "
                          "signedness is selected by the integer element type");
  }
  if (failed(extraCheck())) {
    return failure();
  }
  if (!isSupportedVCmpPredicate(op.getCmp())) {
    return op.emitOpError("unsupported compare predicate '")
           << op.getCmp() << "'; expected eq/ne/lt/le/gt/ge, "
           << "or oeq/one/olt/ole/ogt/oge";
  }
  if (auto pmode = op.getPmode()) {
    bool unsupportedPmode =
        pmode.value() != "zero" && pmode.value() != "merge";
    if (unsupportedPmode) {
      return op.emitOpError("unsupported pmode '")
             << pmode.value() << "'; expected \"zero\" or \"merge\"";
    }
  }
  if (failed(verifyMaskMatchesData(op.getOperation(), seedType, dataType))) {
    return failure();
  }
  if (seedType.getElementCount() != resultType.getElementCount()) {
    return op.emitOpError(
        "requires result mask lane count to match seed mask lane count");
  }
  return success();
}

template <typename OpTy>
static LogicalResult verifyVMIFloatUnarySourceOp(OpTy op) {
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  if (!isVMIF16OrF32Type(sourceType.getElementType())) {
    return op.emitOpError("requires f16 or f32 VMI element type");
  }
  if (failed(verifyFloatUnaryVRegOp(op.getOperation(), sourceType, resultType))) {
    return failure();
  }
  if (failed(verifyVMIVariadicPmodeMask(op.getOperation(), op.getMask(),
                                        resultType, op.getPmode()))) {
    return failure();
  }
  return success();
}

LogicalResult VMIIotaOp::verify() {
  return verifyVMIIotaLikeOp(*this);
}

LogicalResult VMICreateMaskOp::verify() {
  return success();
}

LogicalResult VMICreateGroupMaskOp::verify() {
  auto resultType = cast<VMIMaskType>(getResult().getType());
  int64_t numGroups = getNumGroupsAttr().getInt();
  int64_t groupSize = getGroupSizeAttr().getInt();
  if (numGroups <= 0) {
    return emitOpError("requires positive num_groups");
  }
  if (groupSize <= 0) {
    return emitOpError("requires positive group_size");
  }
  if (resultType.getElementCount() != numGroups * groupSize) {
    return emitOpError("requires result lane count to equal num_groups * "
                       "group_size");
  }
  return success();
}

LogicalResult VMIConstantMaskOp::verify() {
  auto resultType = cast<VMIMaskType>(getResult().getType());
  auto denseAttr = dyn_cast<DenseElementsAttr>(getValue());
  if (!denseAttr) {
    return emitOpError("requires dense elements mask constant attribute");
  }
  if (!denseAttr.getElementType().isInteger(1)) {
    return emitOpError("requires dense mask constant element type to be i1");
  }
  if (denseAttr.getNumElements() != resultType.getElementCount()) {
    return emitOpError("requires dense mask constant element count to match "
                       "result logical lane count");
  }
  return success();
}

LogicalResult VMIMaskAndOp::verify() {
  auto lhsType = cast<VMIMaskType>(getLhs().getType());
  auto rhsType = cast<VMIMaskType>(getRhs().getType());
  auto resultType = cast<VMIMaskType>(getResult().getType());
  return verifyAllSameMaskShapeLayoutAndGranularity(
      getOperation(), {lhsType, rhsType, resultType});
}

LogicalResult VMIMaskOrOp::verify() {
  auto lhsType = cast<VMIMaskType>(getLhs().getType());
  auto rhsType = cast<VMIMaskType>(getRhs().getType());
  auto resultType = cast<VMIMaskType>(getResult().getType());
  return verifyAllSameMaskShapeLayoutAndGranularity(
      getOperation(), {lhsType, rhsType, resultType});
}

LogicalResult VMIMaskXOrOp::verify() {
  auto lhsType = cast<VMIMaskType>(getLhs().getType());
  auto rhsType = cast<VMIMaskType>(getRhs().getType());
  auto resultType = cast<VMIMaskType>(getResult().getType());
  return verifyAllSameMaskShapeLayoutAndGranularity(
      getOperation(), {lhsType, rhsType, resultType});
}

LogicalResult VMIMaskNotOp::verify() {
  auto sourceType = cast<VMIMaskType>(getSource().getType());
  auto resultType = cast<VMIMaskType>(getResult().getType());
  return verifyAllSameMaskShapeLayoutAndGranularity(getOperation(),
                                                    {sourceType, resultType});
}

// Batch8: float elementwise 五兄弟共用校验
template <typename OpTy>
static LogicalResult verifyVMIFloatElementwiseOp(OpTy op) {
  auto lhsType = cast<VMIVRegType>(op.getLhs().getType());
  auto rhsType = cast<VMIVRegType>(op.getRhs().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  if (!isVMIFloatLikeType(lhsType.getElementType())) {
    return op.emitOpError("requires floating-point-like VMI element type");
  }
  if (!isVMIF16BF16OrF32Type(lhsType.getElementType())) {
    return op.emitOpError("requires f16, bf16, or f32 VMI element type");
  }
  return verifyElementwiseVRegOp(op.getOperation(), lhsType, rhsType, resultType);
}


LogicalResult VMIAddFOp::verify() { return verifyVMIFloatElementwiseOp(*this); }
// Batch8: 整数位运算六兄弟 + vcmax/vcmin 共用校验
template <typename OpTy>
static LogicalResult verifyVMIIntBitwiseOp(OpTy op) {
  auto lhsType = cast<VMIVRegType>(op.getLhs().getType());
  auto rhsType = cast<VMIVRegType>(op.getRhs().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  if (!isVMIIntegerLikeType(lhsType.getElementType())) {
    return op.emitOpError("requires integer-like VMI element type");
  }
  if (!isVMIAnyI8I16I32Type(lhsType.getElementType())) {
    return op.emitOpError("requires i8, i16, or i32 VMI element type");
  }
  return verifyElementwiseVRegOp(op.getOperation(), lhsType, rhsType, resultType);
}

template <typename OpTy>
static LogicalResult verifyVMIvcCmpOp(OpTy op) {
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  bool isFloat = false;
  if (failed(verifyVCReductionElementAndMask(op.getOperation(), sourceType,
                                             maskType, isFloat))) {
    return failure();
  }
  return verifyReductionGroupAndPmode(op.getOperation(), sourceType, resultType,
                                      op.getGroupAttr(), op.getPmode());
}


LogicalResult VMIAddIOp::verify() { return verifyVMIIntBitwiseOp(*this); }

LogicalResult VMISubFOp::verify() { return verifyVMIFloatElementwiseOp(*this); }
LogicalResult VMISubIOp::verify() { return verifyVMIIntBitwiseOp(*this); }

LogicalResult VMIMulFOp::verify() { return verifyVMIFloatElementwiseOp(*this); }
LogicalResult VMIMulIOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIIntegerLikeType(lhsType.getElementType())) {
    return emitOpError("requires integer-like VMI element type");
  }
  auto intType = dyn_cast<IntegerType>(lhsType.getElementType());
  bool supportedInteger =
      intType && (intType.getWidth() == mlir::pto::kValue16 || intType.getWidth() == mlir::pto::kValue32);
  if (!supportedInteger) {
    return emitOpError("requires i16 or i32 VMI element type");
  }
  return verifyElementwiseVRegOp(getOperation(), lhsType, rhsType, resultType);
}

LogicalResult VMIFmaOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto accType = cast<VMIVRegType>(getAcc().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIF16BF16OrF32Type(lhsType.getElementType())) {
    return emitOpError("requires f16, bf16, or f32 VMI element type");
  }
  return verifyFloatTernaryVRegOp(getOperation(), lhsType, rhsType, accType,
                                  resultType);
}

//===----------------------------------------------------------------------===//
// Legacy elementwise op verifiers (restored for backward compatibility).
//===----------------------------------------------------------------------===//

LogicalResult VMIDivFOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIFloatLikeType(lhsType.getElementType())) {
    return emitOpError("requires floating-point-like VMI element type");
  }
  if (!isVMIF16OrF32Type(lhsType.getElementType())) {
    return emitOpError("requires f16 or f32 VMI element type");
  }
  return verifyElementwiseVRegOp(getOperation(), lhsType, rhsType, resultType);
}

LogicalResult VMIMinFOp::verify() { return verifyVMIFloatElementwiseOp(*this); }
LogicalResult VMIMaxFOp::verify() { return verifyVMIFloatElementwiseOp(*this); }
LogicalResult VMIMinIOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIIntegerLikeType(lhsType.getElementType()) ||
      !isVMIAnyI8I16I32Type(lhsType.getElementType())) {
    return emitOpError("requires i8, i16, or i32 integer-like VMI element type");
  }
  return verifyElementwiseVRegOp(getOperation(), lhsType, rhsType, resultType);
}

LogicalResult VMIMaxIOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIIntegerLikeType(lhsType.getElementType()) ||
      !isVMIAnyI8I16I32Type(lhsType.getElementType())) {
    return emitOpError("requires i8, i16, or i32 integer-like VMI element type");
  }
  return verifyElementwiseVRegOp(getOperation(), lhsType, rhsType, resultType);
}

LogicalResult VMINegFOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIF16OrF32Type(sourceType.getElementType())) {
    return emitOpError("requires f16 or f32 VMI element type");
  }
  return verifyFloatUnaryVRegOp(getOperation(), sourceType, resultType);
}

LogicalResult VMINegIOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIAnyI8I16I32Type(sourceType.getElementType())) {
    return emitOpError("requires i8, i16, or i32 VMI element type");
  }
  return verifyAllSameVRegShapeAndLayout(getOperation(),
                                         {sourceType, resultType},
                                         /*requireSameElement=*/true);
}

LogicalResult VMIAbsFOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIF16OrF32Type(sourceType.getElementType())) {
    return emitOpError("requires f16 or f32 VMI element type");
  }
  return verifyFloatUnaryVRegOp(getOperation(), sourceType, resultType);
}

LogicalResult VMIAbsIOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIIntegerLikeType(sourceType.getElementType())) {
    return emitOpError("requires integer-like VMI element type");
  }
  if (!isVMISignedI8I16I32Type(sourceType.getElementType())) {
    return emitOpError("requires si8, si16, or si32 VMI element type");
  }
  return verifyAllSameVRegShapeAndLayout(getOperation(),
                                         {sourceType, resultType},
                                         /*requireSameElement=*/true);
}

LogicalResult VMISqrtOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIF16OrF32Type(sourceType.getElementType())) {
    return emitOpError("requires f16 or f32 VMI element type");
  }
  return verifyFloatUnaryVRegOp(getOperation(), sourceType, resultType);
}

LogicalResult VMIExpOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIF16OrF32Type(sourceType.getElementType())) {
    return emitOpError("requires f16 or f32 VMI element type");
  }
  return verifyFloatUnaryVRegOp(getOperation(), sourceType, resultType);
}

LogicalResult VMILnOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIF16OrF32Type(sourceType.getElementType())) {
    return emitOpError("requires f16 or f32 VMI element type");
  }
  return verifyFloatUnaryVRegOp(getOperation(), sourceType, resultType);
}

LogicalResult VMIReluOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  Type elementType = sourceType.getElementType();
  if (failed(verifySignedI32OrF16F32ElementType(getOperation(), elementType))) {
    return failure();
  }
  return verifyAllSameVRegShapeAndLayout(getOperation(),
                                         {sourceType, resultType},
                                         /*requireSameElement=*/true);
}

LogicalResult VMIAndIOp::verify() { return verifyVMIIntBitwiseOp(*this); }

LogicalResult VMIOrIOp::verify() { return verifyVMIIntBitwiseOp(*this); }

LogicalResult VMIXOrIOp::verify() { return verifyVMIIntBitwiseOp(*this); }

LogicalResult VMIShLIOp::verify() { return verifyVMIIntBitwiseOp(*this); }

LogicalResult VMIShRUIOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  auto integerType = dyn_cast<IntegerType>(lhsType.getElementType());
  if (!integerType || integerType.isSigned()) {
    return emitOpError(
        "requires signless or unsigned integer VMI element type");
  }
  if (!isVMIAnyI8I16I32Type(lhsType.getElementType())) {
    return emitOpError("requires i8, i16, or i32 VMI element type");
  }
  return verifyElementwiseVRegOp(getOperation(), lhsType, rhsType, resultType);
}

LogicalResult VMIShRSIOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMISignedI8I16I32Type(lhsType.getElementType())) {
    return emitOpError(
        "requires signed i8, i16, or i32 VMI element type");
  }
  return verifyElementwiseVRegOp(getOperation(), lhsType, rhsType, resultType);
}

LogicalResult VMINotOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIIntegerLikeType(sourceType.getElementType())) {
    return emitOpError("requires integer-like VMI element type");
  }
  if (!isVMIAnyI8I16I32Type(sourceType.getElementType())) {
    return emitOpError("requires i8, i16, or i32 VMI element type");
  }
  return verifyAllSameVRegShapeAndLayout(getOperation(),
                                         {sourceType, resultType},
                                         /*requireSameElement=*/true);
}

//===----------------------------------------------------------------------===//

LogicalResult VMICmpFOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIMaskType>(getResult().getType());
  if (!isVMIFloatLikeType(lhsType.getElementType())) {
    return emitOpError("requires floating-point-like VMI element type");
  }
  if (!isVMIF16BF16OrF32Type(lhsType.getElementType())) {
    return emitOpError("requires f16, bf16, or f32 VMI element type");
  }
  if (failed(verifyAllSameVRegShapeAndLayout(getOperation(), {lhsType, rhsType},
                                             /*requireSameElement=*/true))) {
    return failure();
  }
  return verifyMaskMatchesData(getOperation(), resultType, lhsType);
}

LogicalResult VMICmpIOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIMaskType>(getResult().getType());
  if (!isVMIIntegerLikeType(lhsType.getElementType())) {
    return emitOpError("requires integer-like VMI element type");
  }
  if (!isVMIAnyI8I16I32Type(lhsType.getElementType())) {
    return emitOpError("requires i8, i16, or i32 VMI element type");
  }
  if (failed(verifyAllSameVRegShapeAndLayout(getOperation(), {lhsType, rhsType},
                                             /*requireSameElement=*/true))) {
    return failure();
  }
  return verifyMaskMatchesData(getOperation(), resultType, lhsType);
}

LogicalResult VMISelectOp::verify() {
  auto maskType = cast<VMIMaskType>(getMask().getType());
  auto trueType = cast<VMIVRegType>(getTrueValue().getType());
  auto falseType = cast<VMIVRegType>(getFalseValue().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (failed(verifyAllSameVRegShapeAndLayout(getOperation(),
                                             {trueType, falseType, resultType},
                                             /*requireSameElement=*/true))) {
    return failure();
  }
  return verifyMaskMatchesData(getOperation(), maskType, resultType);
}

LogicalResult VMIActivePrefixIndexOp::verify() {
  auto maskType = cast<VMIMaskType>(getMask().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  auto resultIntType = dyn_cast<IntegerType>(resultType.getElementType());
  if (!resultIntType || !resultIntType.isSignless()) {
    return emitOpError("requires signless integer result element type");
  }
  unsigned resultWidth = resultIntType.getWidth();
  if (resultWidth != mlir::pto::kValue8 && resultWidth != mlir::pto::kValue16 && resultWidth != mlir::pto::kValue32) {
    return emitOpError("requires i8, i16, or i32 result element type");
  }
  return verifyMaskMatchesData(getOperation(), maskType, resultType);
}

LogicalResult VMICompressOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (failed(verifyAllSameVRegShapeAndLayout(getOperation(),
                                             {sourceType, resultType},
                                             /*requireSameElement=*/true))) {
    return failure();
  }
  return verifyMaskMatchesData(getOperation(), maskType, sourceType);
}

LogicalResult VMIShuffleOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (sourceType.getElementType() != resultType.getElementType()) {
    return emitOpError(
        "requires result element type to match source element type");
  }
  if (static_cast<int64_t>(getIndices().size()) != resultType.getElementCount()) {
    return emitOpError(
        "requires shuffle index count to match result logical lane count");
  }
  for (int64_t index : getIndices()) {
    if (index < 0 || index >= sourceType.getElementCount()) {
      return emitOpError("requires every shuffle index to select an existing "
                         "source logical lane");
    }
  }
  if (isLayoutAssigned(sourceType) || isLayoutAssigned(resultType)) {
    if (!isLayoutAssigned(sourceType) || !isLayoutAssigned(resultType)) {
      return emitOpError("requires either both source and result to carry "
                         "layout or neither to carry layout");
    }
  }
  return success();
}

LogicalResult verifyChannelSplitLayout(Operation *op,
                                              VMIVRegType sourceType,
                                              ValueRange results) {
  if (!isLayoutAssigned(sourceType)) {
    return op->emitOpError("requires layout-assigned channel_split source when "
                           "any channel result has layout");
  }
  for (Value result : results) {
    auto resultType = cast<VMIVRegType>(result.getType());
    if (!isLayoutAssigned(resultType)) {
      return op->emitOpError("requires every channel_split result to carry "
                             "layout when source has layout");
    }
    if (!cast<VMILayoutAttr>(resultType.getLayout()).isContiguous()) {
      return op->emitOpError(
          "requires layout-assigned channel_split results to be contiguous");
    }
  }
  int64_t channels = results.size();
  if (channels == mlir::pto::kValue2 || channels == mlir::pto::kValue4) {
    auto sourceLayout = cast<VMILayoutAttr>(sourceType.getLayout());
    auto expectedLayout =
        VMILayoutAttr::getDeinterleaved(op->getContext(), channels);
    if (!sourceLayout.isContiguous() && sourceLayout != expectedLayout) {
      return op->emitOpError("requires layout-assigned channel_split source to "
                             "be contiguous or deinterleaved by result count");
    }
  }
  return success();
}

LogicalResult verifyChannelMergeLayout(Operation *op,
                                              VMIVRegType resultType,
                                              ValueRange inputs) {
  if (!isLayoutAssigned(resultType)) {
    return op->emitOpError("requires layout-assigned channel_merge result when "
                           "any channel input has layout");
  }
  for (Value input : inputs) {
    auto inputType = cast<VMIVRegType>(input.getType());
    if (!isLayoutAssigned(inputType)) {
      return op->emitOpError("requires every channel_merge input to carry layout "
                             "when result has layout");
    }
    if (!cast<VMILayoutAttr>(inputType.getLayout()).isContiguous()) {
      return op->emitOpError(
          "requires layout-assigned channel_merge inputs to be contiguous");
    }
  }
  int64_t channels = inputs.size();
  if (channels == mlir::pto::kValue2 || channels == mlir::pto::kValue4) {
    auto resultLayout = cast<VMILayoutAttr>(resultType.getLayout());
    auto expectedLayout =
        VMILayoutAttr::getDeinterleaved(op->getContext(), channels);
    if (!resultLayout.isContiguous() && resultLayout != expectedLayout) {
      return op->emitOpError("requires layout-assigned channel_merge result to "
                             "be contiguous or deinterleaved by input count");
    }
  }
  return success();
}

LogicalResult VMIEnsureLayoutOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (sourceType.getElementCount() != resultType.getElementCount() ||
      sourceType.getElementType() != resultType.getElementType()) {
    return emitOpError("requires source and result to preserve VMI data shape "
                       "and element type");
  }
  if (!isLayoutAssigned(sourceType) || !isLayoutAssigned(resultType)) {
    return emitOpError("requires source and result to be layout-assigned");
  }
  return success();
}

LogicalResult VMIEnsureMaskLayoutOp::verify() {
  auto sourceType = cast<VMIMaskType>(getSource().getType());
  auto resultType = cast<VMIMaskType>(getResult().getType());
  if (sourceType.getElementCount() != resultType.getElementCount() ||
      sourceType.getGranularity() != resultType.getGranularity()) {
    return emitOpError("requires source and result to preserve VMI mask shape "
                       "and granularity");
  }
  if (!isLayoutAssigned(sourceType) || !isLayoutAssigned(resultType)) {
    return emitOpError("requires source and result to be layout-assigned");
  }
  return success();
}

LogicalResult VMIEnsureMaskGranularityOp::verify() {
  auto sourceType = cast<VMIMaskType>(getSource().getType());
  auto resultType = cast<VMIMaskType>(getResult().getType());
  if (sourceType.getElementCount() != resultType.getElementCount()) {
    return emitOpError(
        "requires source and result to preserve VMI mask lane count");
  }
  if (sourceType.isPred() || resultType.isPred()) {
    return emitOpError(
        "requires concrete source and result mask granularities");
  }
  if (isLayoutAssigned(sourceType) || isLayoutAssigned(resultType)) {
    if (!isLayoutAssigned(sourceType) || !isLayoutAssigned(resultType)) {
      return emitOpError("requires either both source and result to carry "
                         "layout or neither to carry layout");
    }
  }
  return success();
}

LogicalResult VMIUnpackOp::verify() {
  return verifyPhysicalParts(getOperation(), getSource().getType(),
                             getParts().getTypes());
}

LogicalResult VMIPackOp::verify() {
  return verifyPhysicalParts(getOperation(), getResult().getType(),
                             getParts().getTypes());
}

//===----------------------------------------------------------------------===//
// VMI vector-scalar op verifiers (vadds/vmuls/vmaxs/vmins/vshls/vshrs)
//===----------------------------------------------------------------------===//

// Batch11: vector-scalar tail shared
static LogicalResult verifyVecScalarShapeMaskPmode(
    Operation *op, VMIVRegType srcType, VMIVRegType resultType,
    VMIMaskType maskType, std::optional<StringRef> pmode) {
  if (failed(verifyAllSameVRegShapeAndLayout(op, {srcType, resultType},
                                             /*requireSameElement=*/true))) {
    return failure();
  }
  if (failed(verifyMaskMatchesData(op, maskType, resultType))) {
    return failure();
  }
  if (pmode.has_value()) {
    StringRef mode = pmode.value();
    if (mode != "merge" && mode != "zero") {
      return op->emitOpError("unsupported pmode '")
             << mode << "'; expected \"merge\" or \"zero\"";
    }
  }
  return success();
}

/// Shared verifier for VMI vector-scalar elementwise ops.
static LogicalResult
verifyVMIVectorScalarOp(Operation *op, VMIVRegType srcType,
                        Type scalarType, VMIVRegType resultType,
                        VMIMaskType maskType,
                        std::optional<StringRef> pmode) {
  Type eltTy = srcType.getElementType();
  if (failed(verifyBF16x2ComputeElementType(op, eltTy))) {
    return failure();
}
  if (!isVMIFloatLikeType(eltTy) && !isVMIIntegerLikeType(eltTy)) {
    return op->emitOpError(
        "requires floating-point-like or integer-like VMI element type");
}

  if (scalarType != eltTy) {
    return op->emitOpError(
        "requires scalar type to match vector element type, got scalar ")
           << scalarType << " vs vector element " << eltTy;
}

  return verifyVecScalarShapeMaskPmode(op, srcType, resultType, maskType,
                                       pmode);
}

/// Shared verifier for VMI vector-scalar integer-only shift ops.
static LogicalResult
verifyVMIVectorScalarShiftOp(Operation *op, VMIVRegType srcType,
                             Type scalarType, VMIVRegType resultType,
                             VMIMaskType maskType,
                             std::optional<StringRef> pmode) {
  Type eltTy = srcType.getElementType();
  if (!isVMIIntegerLikeType(eltTy)) {
    return op->emitOpError(
        "requires integer-like VMI element type for shift");
  }
  if (!scalarType.isSignlessInteger(mlir::pto::kValue16)) {
    return op->emitOpError("requires signless i16 shift amount");
  }
  return verifyVecScalarShapeMaskPmode(op, srcType, resultType, maskType, pmode);
}

LogicalResult VMIAddSOp::verify() {
  auto srcType = cast<VMIVRegType>(getSrc().getType());
  if (failed(verifyBF16x2ComputeElementType(
          getOperation(), srcType.getElementType()))) {
    return failure();
  }
  if (!isVMII8I16I32OrF16BF16F32Type(srcType.getElementType())) {
    return emitOpError(
        "requires i8, i16, i32, f16, bf16, or f32 VMI element type");
  }
  return verifyVMIVectorScalarOp(getOperation(),
      srcType, getScalar().getType(),
      cast<VMIVRegType>(getResult().getType()),
      cast<VMIMaskType>(getMask().getType()), getPmode());
}

LogicalResult VMIMulSOp::verify() {
  auto srcType = cast<VMIVRegType>(getSrc().getType());
  Type elementType = srcType.getElementType();
  auto intType = dyn_cast<IntegerType>(elementType);
  bool supportedInteger =
      intType && (intType.getWidth() == mlir::pto::kValue16 || intType.getWidth() == mlir::pto::kValue32);
  if (!supportedInteger && !isVMIF16OrF32Type(elementType)) {
    return emitOpError("requires i16, i32, f16, or f32 VMI element type");
  }
  return verifyVMIVectorScalarOp(getOperation(),
      srcType, getScalar().getType(),
      cast<VMIVRegType>(getResult().getType()),
      cast<VMIMaskType>(getMask().getType()), getPmode());
}

// Batch11: MaxS/MinS shared
template <typename OpTy>
static LogicalResult verifyVMIMaxMinSOp(OpTy op) {
  auto srcType = cast<VMIVRegType>(op.getSrc().getType());
  if (!isVMII8I16I32OrF16BF16F32Type(srcType.getElementType())) {
    return op.emitOpError(
        "requires i8, i16, i32, f16, bf16, or f32 VMI element type");
  }
  return verifyVMIVectorScalarOp(op.getOperation(),
      srcType, op.getScalar().getType(),
      cast<VMIVRegType>(op.getResult().getType()),
      cast<VMIMaskType>(op.getMask().getType()), op.getPmode());
}

LogicalResult VMIMaxSOp::verify() { return verifyVMIMaxMinSOp(*this); }

LogicalResult VMIMinSOp::verify() { return verifyVMIMaxMinSOp(*this); }

LogicalResult VMIShlSOp::verify() {
  return verifyVMIVectorScalarShiftOp(getOperation(),
      cast<VMIVRegType>(getSrc().getType()), getScalar().getType(),
      cast<VMIVRegType>(getResult().getType()),
      cast<VMIMaskType>(getMask().getType()), getPmode());
}

LogicalResult VMIShrSOp::verify() {
  return verifyVMIVectorScalarShiftOp(getOperation(),
      cast<VMIVRegType>(getSrc().getType()), getScalar().getType(),
      cast<VMIVRegType>(getResult().getType()),
      cast<VMIMaskType>(getMask().getType()), getPmode());
}

//===----------------------------------------------------------------------===//
// Unified (new) VMI op verifiers
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//

/// Returns true if `cmp` is a comparison predicate supported by VCMP.
static bool isSupportedVCmpPredicate(StringRef cmp) {
  return cmp == "eq" || cmp == "ne" || cmp == "lt" ||
         cmp == "le" || cmp == "gt" || cmp == "ge" ||
         cmp == "oeq" || cmp == "one" || cmp == "olt" ||
         cmp == "ole" || cmp == "ogt" || cmp == "oge";
}

//===--- Unified (new) VMI op verifiers ---===//

LogicalResult VMIVbrcOp::verify() {
  auto resultType = cast<VMIVRegType>(getResult().getType());
  Type valueType = getValue().getType();

  if (auto groupAttr = getGroupAttr()) {
    // Group broadcast mode
    int64_t numGroupsVal = groupAttr.getInt();
    auto vregType = dyn_cast<VMIVRegType>(valueType);
    if (!vregType) {
      return emitOpError("requires VMI vector input when num_groups is set");
    }
    if (vregType.getElementCount() != numGroupsVal) {
      return emitOpError()
             << "requires source logical lane count " << vregType.getElementCount()
             << " to match num_groups " << numGroupsVal;
    }
    if (vregType.getElementType() != resultType.getElementType()) {
      return emitOpError("requires source and result element types to match");
    }
    if (auto sourceLayout = vregType.getLayoutAttr()) {
      if (!sourceLayout.isGroupSlots() ||
          sourceLayout.getNumGroups() != numGroupsVal) {
        return emitOpError() << "requires layout-assigned source to use "
                                "#pto.vmi.layout<num_groups = "
                             << numGroupsVal << ">";
      }
    }
    if (auto resultLayout = resultType.getLayoutAttr()) {
      if (resultLayout.isGroupSlots()) {
        return emitOpError(
            "requires layout-assigned result to use a dense VMI layout");
      }
    }
    return verifyNumGroups(getOperation(), resultType, numGroupsVal);
  }

  // Scalar/1-lane broadcast mode (no num_groups)
  return verifyVMIBroadcastScalarMode(*this, valueType, resultType);
}

LogicalResult VMIVciOp::verify() {
  if (failed(verifyVMIIotaLikeOp(*this))) {
    return failure();
  }
  if (auto groupAttr = getGroupAttr()) {
    int64_t numGroups = groupAttr.getInt();
    if (numGroups <= 0) {
      return emitOpError("requires group to be positive");
    }
  }
  return success();
}

LogicalResult VMIPsetOp::verify() {
  auto resultType = cast<VMIMaskType>(getResult().getType());
  StringRef pattern = getPattern();
  if (pattern != "PAT_ALL") {
    return emitOpError("requires pattern to be \"PAT_ALL\"");
  }
  if (!resultType.isPred() && !isLayoutAssigned(resultType)) {
    return emitOpError("requires concrete mask result to carry layout");
  }
  return success();
}

LogicalResult VMIPgeOp::verify() {
  auto resultType = cast<VMIMaskType>(getResult().getType());
  StringRef pattern = getPattern();
  if (!pattern.starts_with("PAT_VL")) {
    return emitOpError("requires pattern to start with \"PAT_VL\"");
  }
  int64_t activeLanes;
  if (pattern.drop_front(mlir::pto::kValue6).getAsInteger(mlir::pto::kValue10, activeLanes)) {
    return emitOpError("requires pattern \"PAT_VL<n>\" with integer n");
  }
  if (activeLanes <= 0) {
    return emitOpError("requires positive n in pattern \"PAT_VL<n>\"");
  }
  if (activeLanes > resultType.getElementCount()) {
    return emitOpError("PAT_VL active lanes ") << activeLanes
                                               << " exceeds mask element count " << resultType.getElementCount();
  }
  if (!resultType.isPred() && !isLayoutAssigned(resultType)) {
    return emitOpError("requires concrete mask result to carry layout");
  }
  return success();
}

LogicalResult VMIPltOp::verify() {
  auto resultType = cast<VMIMaskType>(getMask().getType());
  auto scalarType = dyn_cast<IntegerType>(getScalar().getType());
  if (!scalarType || scalarType.getWidth() != mlir::pto::kValue32) {
    return emitOpError("requires i32 scalar input");
  }
  auto scalarOutType = dyn_cast<IntegerType>(getScalarOut().getType());
  if (!scalarOutType || scalarOutType.getWidth() != mlir::pto::kValue32) {
    return emitOpError("requires i32 scalar_out result");
  }
  if (!resultType.isPred() && !isLayoutAssigned(resultType)) {
    return emitOpError("requires concrete mask result to carry layout");
  }
  return success();
}

// Batch6: elementwise 二元 op 共用校验骨架(谓词与消息由调用方提供)
template <typename BinaryOpTy>
static LogicalResult verifyVMIBinaryComputeOp(
    BinaryOpTy op, llvm::function_ref<bool(mlir::Type)> checkElementType,
    llvm::StringRef elementTypeMessage) {
  auto lhsType = cast<VMIVRegType>(op.getLhs().getType());
  auto rhsType = cast<VMIVRegType>(op.getRhs().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  if (failed(verifyBF16x2ComputeElementType(op.getOperation(),
                                            lhsType.getElementType()))) {
    return failure();
  }
  if (!checkElementType(lhsType.getElementType())) {
    return op.emitOpError(elementTypeMessage);
  }
  if (failed(verifyElementwiseVRegOp(op.getOperation(), lhsType, rhsType,
                                     resultType))) {
    return failure();
  }
  if (failed(verifyVMIVariadicPmodeMask(op.getOperation(), op.getMask(),
                                        resultType, op.getPmode()))) {
    return failure();
  }
  return success();
}


LogicalResult VMIVaddOp::verify() {
  return verifyVMIBinaryComputeOp(
      *this, [](mlir::Type type) { return isVMII8I16I32OrF16BF16F32Type(type); },
      "requires i8, i16, i32, f16, bf16, or f32 VMI element type");
}

LogicalResult VMIVsubOp::verify() {
  return verifyVMIBinaryComputeOp(
      *this, [](mlir::Type type) { return isVMII8I16I32OrF16BF16F32Type(type); },
      "requires i8, i16, i32, f16, bf16, or f32 VMI element type");
}

LogicalResult VMIVmulOp::verify() {
  return verifyVMIBinaryComputeOp(
      *this, [](mlir::Type type) { return isVMII16I32OrF16BF16F32Type(type); },
      "requires i16, i32, f16, bf16, or f32 VMI element type");
}

LogicalResult VMIVdivOp::verify() {
  return verifyVMIBinaryComputeOp(
      *this, [](mlir::Type type) { return isVMIF16OrF32Type(type); },
      "requires f16 or f32 VMI element type");
}

LogicalResult VMIVminOp::verify() {
  return verifyVMIBinaryComputeOp(
      *this, [](mlir::Type type) { return isVMII8I16I32OrF16BF16F32Type(type); },
      "requires i8, i16, i32, f16, bf16, or f32 VMI element type");
}

LogicalResult VMIVmaxOp::verify() {
  return verifyVMIBinaryComputeOp(
      *this, [](mlir::Type type) { return isVMII8I16I32OrF16BF16F32Type(type); },
      "requires i8, i16, i32, f16, bf16, or f32 VMI element type");
}

// Batch11: unary source tail shared
template <typename OpTy>
static LogicalResult verifyUnarySourcePmodeTail(OpTy op, VMIVRegType sourceType,
                                                VMIVRegType resultType) {
  if (failed(verifyAllSameVRegShapeAndLayout(
          op.getOperation(), {sourceType, resultType},
          /*requireSameElement=*/true))) {
    return failure();
  }
  if (failed(verifyVMIVariadicPmodeMask(op.getOperation(), op.getMask(),
                                        resultType, op.getPmode()))) {
    return failure();
  }
  return success();
}

LogicalResult VMIVnegOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMII8I16I32OrF16F32Type(sourceType.getElementType())) {
    return emitOpError("requires i8, i16, i32, f16, or f32 VMI element type");
  }
  return verifyUnarySourcePmodeTail(*this, sourceType, resultType);
}

LogicalResult VMIVabsOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());

  Type eltTy = sourceType.getElementType();
  if (failed(verifyBF16x2ComputeElementType(getOperation(), eltTy))) {
    return failure();
  }
  bool supportedInteger = isVMISignedI8I16I32Type(eltTy);
  if (!supportedInteger && !isVMIF16BF16OrF32Type(eltTy)) {
    return emitOpError(
        "requires si8, si16, si32, f16, bf16, or f32 VMI element type");
  }

  if (failed(verifyAllSameVRegShapeAndLayout(
          getOperation(), {sourceType, resultType},
          /*requireSameElement=*/true))) {
    return failure();
}

  if (failed(verifyVMIVariadicPmodeMask(getOperation(), getMask(),
                                      resultType, getPmode()))) {
    return failure();
}
  return success();
}

LogicalResult VMIVsqrtOp::verify() {
  return verifyVMIFloatUnarySourceOp(*this);
}

LogicalResult VMIVexpOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (failed(verifyBF16x2ComputeElementType(
          getOperation(), sourceType.getElementType()))) {
    return failure();
  }
  if (!isVMIF16OrF32Type(sourceType.getElementType())) {
    return emitOpError("requires f16 or f32 VMI element type");
  }
  if (failed(verifyFloatUnaryVRegOp(getOperation(), sourceType, resultType))) {
    return failure();
  }
  if (failed(verifyVMIVariadicPmodeMask(getOperation(), getMask(),
                                        resultType, getPmode()))) {
    return failure();
  }
  return success();
}

LogicalResult VMIVlnOp::verify() {
  return verifyVMIFloatUnarySourceOp(*this);
}

LogicalResult VMIVreluOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  Type elementType = sourceType.getElementType();
  if (failed(verifySignedI32OrF16F32ElementType(getOperation(), elementType))) {
    return failure();
  }
  return verifyUnarySourcePmodeTail(*this, sourceType, resultType);
}

LogicalResult VMIVandOp::verify() {
  return verifyVMIMaskLogicOp(*this);
}

LogicalResult VMIVorOp::verify() {
  return verifyVMIMaskLogicOp(*this);
}

LogicalResult VMIVxorOp::verify() {
  return verifyVMIMaskLogicOp(*this);
}

// Batch11: elementwise tail shared
template <typename OpTy>
static LogicalResult verifyElementwisePmodeTail(OpTy op, VMIVRegType lhsType,
                                                VMIVRegType rhsType,
                                                VMIVRegType resultType) {
  if (failed(verifyElementwiseVRegOp(op.getOperation(), lhsType, rhsType, resultType))) {
    return failure();
  }
  if (failed(verifyVMIVariadicPmodeMask(op.getOperation(), op.getMask(),
                                        resultType, op.getPmode()))) {
    return failure();
  }
  return success();
}

LogicalResult VMIVshlOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIIntegerLikeType(lhsType.getElementType())) {
    return emitOpError("requires integer-like VMI element type");
  }
  return verifyElementwisePmodeTail(*this, lhsType, rhsType, resultType);
}

LogicalResult VMIVshrOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  auto integerType = dyn_cast<IntegerType>(lhsType.getElementType());
  if (!integerType) {
    return emitOpError("requires integer VMI element type");
  }
  return verifyElementwisePmodeTail(*this, lhsType, rhsType, resultType);
}

LogicalResult VMIVnotOp::verify() {
  if (isa<VMIMaskType>(getSource().getType())) {
    // Mask logic path: reject predication mask and pmode.
    if (!getMask().empty()) {
      return emitOpError("mask logic op does not support predication mask");
    }
    if (auto pmode = getPmode()) {
      return emitOpError("mask logic op does not support pmode");
    }
    auto sourceType = cast<VMIMaskType>(getSource().getType());
    auto resultType = cast<VMIMaskType>(getResult().getType());
    return verifyAllSameMaskShapeLayoutAndGranularity(
        getOperation(), {sourceType, resultType});
  }
  // VReg path.
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIIntegerLikeType(sourceType.getElementType())) {
    return emitOpError("requires integer-like VMI element type");
  }
  if (failed(verifyAllSameVRegShapeAndLayout(getOperation(),
                                             {sourceType, resultType},
                                             /*requireSameElement=*/true))) {
    return failure();
  }
  if (failed(verifyVMIVariadicPmodeMask(getOperation(), getMask(),
                                        resultType, getPmode()))) {
    return failure();
  }
  return success();
}

LogicalResult VMIvSelOp::verify() {
  auto maskType = cast<VMIMaskType>(getMask().getType());
  auto trueType = cast<VMIVRegType>(getTrueValue().getType());
  auto falseType = cast<VMIVRegType>(getFalseValue().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (failed(verifyAllSameVRegShapeAndLayout(getOperation(),
                                             {trueType, falseType, resultType},
                                             /*requireSameElement=*/true))) {
    return failure();
  }
  if (failed(verifyMaskMatchesData(getOperation(), maskType, resultType))) {
    return failure();
  }
  if (auto pmode = getPmode(); pmode.has_value()) {
    StringRef mode = pmode.value();
    if (mode != "merge" && mode != "zero") {
      return emitOpError("pmode must be \"merge\" or \"zero\", got \"")
             << mode << "\"";
    }
  }
  return success();
}

LogicalResult VMIvcaddOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  bool isFloat = false;
  if (failed(verifyVCReductionElementAndMask(getOperation(), sourceType,
                                             maskType, isFloat))) {
    return failure();
  }
  // Floating-point vcadd MUST carry reassoc
  if (isFloat && !getReassoc()) {
    return emitOpError("floating add-reduction requires reassoc attr");
  }
  return verifyReductionGroupAndPmode(getOperation(), sourceType, resultType,
                                      getGroupAttr(), getPmode());
}

LogicalResult VMIvcmaxOp::verify() { return verifyVMIvcCmpOp(*this); }

LogicalResult VMIvcminOp::verify() { return verifyVMIvcCmpOp(*this); }

LogicalResult VMIVexpdifOp::verify() {
  auto xType = cast<VMIVRegType>(getX().getType());
  auto maxType = cast<VMIVRegType>(getMax().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());

  if (!isVMIF16OrF32Type(xType.getElementType())) {
    return emitOpError("requires x element type to be f16 or f32");
  }

  if (maxType.getElementType() != xType.getElementType()) {
    return emitOpError("requires x and max to have the same element type");
  }

  auto resultElemType = dyn_cast<FloatType>(resultType.getElementType());
  if (!resultElemType || resultElemType.getWidth() != mlir::pto::kValue32) {
    return emitOpError("requires result element type to be f32");
  }

  if (xType.getElementCount() != maxType.getElementCount() ||
      xType.getElementCount() != resultType.getElementCount()) {
    return emitOpError(
        "requires x, max, and result logical lane counts to match");
  }

  if (failed(verifyMaskMatchesData(getOperation(), maskType, xType))) {
    return failure();
  }

  if (auto pmode = getPmode()) {
    if (pmode.value() != "merge" && pmode.value() != "zero") {
      return emitOpError("pmode must be 'merge' or 'zero'");
    }
  }
  return success();
}

LogicalResult VMIVaxpyOp::verify() {
  auto xType = cast<VMIVRegType>(getX().getType());
  auto accType = cast<VMIVRegType>(getAcc().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());

  if (failed(verifyBF16x2ComputeElementType(
          getOperation(), xType.getElementType()))) {
    return failure();
  }
  if (!isVMIFloatLikeType(xType.getElementType())) {
    return emitOpError("requires vector element type to be f16 or f32");
  }

  if (xType != accType || accType != resultType) {
    return emitOpError(
        "requires x, acc, and result to have identical VMI vreg types");
  }

  auto alphaType = cast<FloatType>(getAlpha().getType());
  if (alphaType != xType.getElementType()) {
    return emitOpError("requires alpha scalar type to match vector element type");
  }

  return verifyMaskMatchesDataPmode(*this, maskType, resultType);
}

LogicalResult VMIVlreluOp::verify() {
  auto xType = cast<VMIVRegType>(getX().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());

  if (failed(verifyBF16x2ComputeElementType(
          getOperation(), xType.getElementType()))) {
    return failure();
  }
  if (!isVMIFloatLikeType(xType.getElementType())) {
    return emitOpError("requires vector element type to be f16 or f32");
  }

  if (xType != resultType) {
    return emitOpError("requires x and result to have identical VMI vreg types");
  }

  auto slopeType = cast<FloatType>(getSlope().getType());
  if (slopeType != xType.getElementType()) {
    return emitOpError(
        "requires slope scalar type to match vector element type");
  }

  return verifyMaskMatchesDataPmode(*this, maskType, resultType);
}

LogicalResult VMIVpreluOp::verify() {
  auto xType = cast<VMIVRegType>(getX().getType());
  auto alphaType = cast<VMIVRegType>(getAlpha().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());

  if (failed(verifyBF16x2ComputeElementType(
          getOperation(), xType.getElementType()))) {
    return failure();
  }

  if (!isVMIFloatLikeType(xType.getElementType())) {
    return emitOpError("requires vector element type to be f16 or f32");
  }

  if (xType != alphaType || alphaType != resultType) {
    return emitOpError(
        "requires x, alpha, and result to have identical VMI vreg types");
  }

  return verifyMaskMatchesDataPmode(*this, maskType, resultType);
}

template <typename CarryOp>
static LogicalResult verifyVMIAddCarryOp(CarryOp op, VMIMaskType carryInType,
                                         bool hasCarryIn) {
  auto lhsType = cast<VMIVRegType>(op.getLhs().getType());
  auto rhsType = cast<VMIVRegType>(op.getRhs().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  auto carryType = cast<VMIMaskType>(op.getCarry().getType());

  auto integerType = dyn_cast<IntegerType>(lhsType.getElementType());
  if (!integerType || integerType.getWidth() != mlir::pto::kValue32) {
    return op.emitOpError("requires 32-bit integer vector element types");
  }

  if (failed(verifyAllSameVRegShapeAndLayout(
          op.getOperation(), {lhsType, rhsType, resultType},
          /*requireSameElement=*/true))) {
    return failure();
  }
  if (failed(verifyMaskMatchesData(op.getOperation(), maskType, lhsType)) ||
      failed(verifyMaskMatchesData(op.getOperation(), carryType, lhsType))) {
    return failure();
  }
  if (hasCarryIn &&
      failed(verifyMaskMatchesData(op.getOperation(), carryInType, lhsType))) {
    return failure();
  }

  SmallVector<VMIMaskType> masks{maskType, carryType};
  if (hasCarryIn) {
    masks.push_back(carryInType);
  }
  return verifyAllSameMaskShapeLayoutAndGranularity(op.getOperation(), masks);
}

LogicalResult VMIVaddcOp::verify() {
  return verifyVMIAddCarryOp(*this, VMIMaskType{}, /*hasCarryIn=*/false);
}

LogicalResult VMIVaddcsOp::verify() {
  return verifyVMIAddCarryOp(*this, cast<VMIMaskType>(getCarryIn().getType()),
                             /*hasCarryIn=*/true);
}

LogicalResult VMIVmullOp::verify() {
  auto aType = cast<VMIVRegType>(getA().getType());
  auto bType = cast<VMIVRegType>(getB().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());
  auto lowType = cast<VMIVRegType>(getLow().getType());
  auto highType = cast<VMIVRegType>(getHigh().getType());

  auto isLegalElementType = [](Type type) {
    auto integerType = dyn_cast<IntegerType>(type);
    return integerType && integerType.getWidth() == mlir::pto::kValue32 &&
           (integerType.isSignless() || integerType.isUnsigned());
  };
  if (!isLegalElementType(aType.getElementType()) ||
      !isLegalElementType(bType.getElementType()) ||
      !isLegalElementType(lowType.getElementType()) ||
      !isLegalElementType(highType.getElementType())) {
    return emitOpError(
        "requires a, b, low, and high element types to be exactly i32 or ui32");
  }

  if (aType != bType || aType != lowType || aType != highType) {
    return emitOpError(
        "requires a, b, low, and high to have identical VMI vreg types");
  }

  int64_t lanes = aType.getElementCount();
  if (lanes != mlir::pto::kValue64 && lanes != mlir::pto::kValue128 && lanes != mlir::pto::kValue256) {
    return emitOpError("requires logical lane count to be 64, 128, or 256");
  }

  if (failed(verifyMaskMatchesData(getOperation(), maskType, aType))) {
    return failure();
  }

  if (auto pmode = getPmode(); pmode && pmode.value() != "zero") {
    return emitOpError("pmode must be 'zero' when specified");
  }
  return success();
}

LogicalResult VMIVmulaOp::verify() {
  auto accType = cast<VMIVRegType>(getAcc().getType());
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());

  Type eltTy = accType.getElementType();
  if (failed(verifyBF16x2ComputeElementType(getOperation(), eltTy))) {
    return failure();
}
  if (!isVMIFloatLikeType(eltTy) && !isVMIIntegerLikeType(eltTy)) {
    return emitOpError(
        "requires floating-point-like or integer-like VMI element type");
}

  if (accType != lhsType || lhsType != rhsType || rhsType != resultType) {
    return emitOpError(
        "requires acc, lhs, rhs, and result to have identical VMI vreg types");
}

  if (failed(verifyVMIVariadicPmodeMask(getOperation(), getMask(),
                                        resultType, getPmode()))) {
    return failure();
}
  return success();
}

LogicalResult VMIVselrOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto indexType = cast<VMIVRegType>(getIndex().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (sourceType.getElementType() != resultType.getElementType()) {
    return emitOpError(
        "requires result element type to match source element type");
  }

  if (indexType.getElementCount() != resultType.getElementCount()) {
    return emitOpError(
        "requires index lane count to match result lane count");
  }

  if (!isa<IntegerType>(indexType.getElementType())) {
    return emitOpError("requires index element type to be integer");
  }

  unsigned sourceBits =
      pto::getPTOStorageElemBitWidth(sourceType.getElementType());
  unsigned indexBits =
      pto::getPTOStorageElemBitWidth(indexType.getElementType());
  if (sourceBits != mlir::pto::kValue8 && sourceBits != mlir::pto::kValue16 && sourceBits != mlir::pto::kValue32) {
    return emitOpError(
        "requires source/result element storage width to be 8, 16, or 32 bits");
  }
  if (indexBits != sourceBits) {
    return emitOpError(
        "requires index element storage width to match source element storage width");
  }

  bool sourceHasLayout = isLayoutAssigned(sourceType);
  bool indexHasLayout = isLayoutAssigned(indexType);
  bool resultHasLayout = isLayoutAssigned(resultType);
  if (sourceHasLayout != resultHasLayout) {
    return emitOpError("requires source and result to both carry layout or "
                       "neither carry layout");
  }
  if (indexHasLayout && !sourceHasLayout) {
    return emitOpError(
        "requires index to carry layout only when source does");
  }

  return success();
}

LogicalResult VMIVintlvOp::verify() {
  return verifyVMIInterleaveLikeOp(*this);
}

LogicalResult VMIVdintlvOp::verify() {
  return verifyVMIInterleaveLikeOp(*this);
}

//===----------------------------------------------------------------------===//
// VMIVabsOp verifier (unified fp/int abs, replaces absf/absi)
//===----------------------------------------------------------------------===//
// VMIVcmpOp / VMIVcmpsOp verifiers
LogicalResult VMIVcmpOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto seedType = cast<VMIMaskType>(getSeed().getType());
  auto resultType = cast<VMIMaskType>(getResult().getType());
  return verifyVMICmpCommon(
      *this, lhsType, seedType, resultType, [&]() -> LogicalResult {
        return verifyAllSameVRegShapeAndLayout(
            getOperation(), {lhsType, rhsType}, /*requireSameElement=*/true);
      });
}

LogicalResult VMIVcmpsOp::verify() {
  auto srcType = cast<VMIVRegType>(getSrc().getType());
  auto seedType = cast<VMIMaskType>(getSeed().getType());
  auto resultType = cast<VMIMaskType>(getResult().getType());
  return verifyVMICmpCommon(
      *this, srcType, seedType, resultType, [&]() -> LogicalResult {
        Type eltTy = srcType.getElementType();
        Type scalarTy = getScalar().getType();
        if (scalarTy != eltTy) {
          return emitOpError("requires scalar type to match vector element type, "
                             "got scalar ")
                 << scalarTy << " vs vector element " << eltTy;
        }
        return success();
      });
}

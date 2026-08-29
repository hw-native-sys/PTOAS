// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOGenericOps.cpp - PTO generic operation verifiers ----------------===//

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"

#include <optional>

using namespace mlir;

namespace mlir {
namespace pto {

static bool isInsideSimtExecutionScope(Operation *op) {
  auto funcOp = op->getParentOfType<func::FuncOp>();
  return (funcOp && funcOp->hasAttr(kPTOSimtEntryAttrName)) ||
         op->getParentOfType<SectionSimtOp>();
}

static std::optional<Rounding> getSimtRounding(FloatRoundingModeAttr attr) {
  if (!attr) {
    return Rounding::R;
  }
  switch (attr.getValue()) {
  case FloatRoundingMode::to_nearest_even:
    return Rounding::R;
  case FloatRoundingMode::to_nearest_away:
    return Rounding::A;
  case FloatRoundingMode::downward:
    return Rounding::F;
  case FloatRoundingMode::upward:
    return Rounding::C;
  case FloatRoundingMode::toward_zero:
    return Rounding::Z;
  case FloatRoundingMode::to_odd:
    return Rounding::O;
  case FloatRoundingMode::hybrid:
    return Rounding::H;
  }
  return std::nullopt;
}

static bool hasLowPrecisionConversionPayload(Type type) {
  return isPTOLowPrecisionType(getElementTypeOrSelf(type));
}

static bool requiresSimtHardwareConversion(Operation *op, Type srcType,
                                           Type dstType,
                                           FloatRoundingModeAttr rounding,
                                           Saturation saturation) {
  return isInsideSimtExecutionScope(op) &&
         (hasLowPrecisionConversionPayload(srcType) ||
          hasLowPrecisionConversionPayload(dstType) ||
          saturation == Saturation::Enable || rounding);
}

static LogicalResult
verifySimtConversionIfNeeded(Operation *op, Type srcType, Type dstType,
                             FloatRoundingModeAttr rounding,
                             Saturation saturation, Attribute signedness) {
  if (!requiresSimtHardwareConversion(op, srcType, dstType, rounding,
                                      saturation)) {
    return success();
  }
  auto simtRounding = getSimtRounding(rounding);
  if (!simtRounding) {
    return op->emitOpError("has an invalid conversion rounding mode");
  }
  return verifySimtConversionControls(op, srcType, dstType, *simtRounding,
                                      saturation, signedness);
}

static LogicalResult rejectSemanticAttrs(Operation *op,
                                         ArrayRef<StringRef> attrNames) {
  for (StringRef attrName : attrNames) {
    if (op->hasAttr(attrName)) {
      return op->emitOpError() << "does not accept " << attrName;
    }
  }
  return success();
}

LogicalResult ConstantOp::verify() {
  if (failed(rejectSemanticAttrs(*this, {"fastmath", "roundingmode",
                                         "overflowFlags", "signedness"}))) {
    return failure();
  }
  auto typedValue = dyn_cast<TypedAttr>(getValue());
  if (!typedValue) {
    return emitOpError() << "requires a typed integer, floating-point, or "
                            "builtin-vector value";
  }
  bool hasMismatchedType = typedValue.getType() != getResult().getType();
  if (hasMismatchedType) {
    return emitOpError() << "requires value type " << typedValue.getType()
                         << " to match result type " << getResult().getType();
  }
  if (!isa<IntegerAttr, FloatAttr, DenseElementsAttr>(typedValue)) {
    return emitOpError() << "requires an integer, floating-point, or dense "
                            "builtin-vector value";
  }
  return success();
}

static LogicalResult verifyIntegerArithmeticOp(Operation *op) {
  return rejectSemanticAttrs(op, {"fastmath", "roundingmode", "signedness"});
}

static LogicalResult verifyFloatArithmeticOp(Operation *op) {
  return rejectSemanticAttrs(op,
                             {"overflowFlags", "roundingmode", "signedness"});
}

LogicalResult AddIOp::verify() { return verifyIntegerArithmeticOp(*this); }
LogicalResult SubIOp::verify() { return verifyIntegerArithmeticOp(*this); }
LogicalResult MulIOp::verify() { return verifyIntegerArithmeticOp(*this); }
LogicalResult NegIOp::verify() { return verifyIntegerArithmeticOp(*this); }

LogicalResult AddFOp::verify() { return verifyFloatArithmeticOp(*this); }
LogicalResult SubFOp::verify() { return verifyFloatArithmeticOp(*this); }
LogicalResult MulFOp::verify() { return verifyFloatArithmeticOp(*this); }
LogicalResult NegFOp::verify() { return verifyFloatArithmeticOp(*this); }

LogicalResult AddUIExtendedOp::verify() {
  return rejectSemanticAttrs(
      *this, {"fastmath", "roundingmode", "overflowFlags", "signedness"});
}

LogicalResult MulExtendedOp::verify() {
  return rejectSemanticAttrs(*this,
                             {"fastmath", "roundingmode", "overflowFlags"});
}

static LogicalResult verifySignedIntegerBinaryOp(Operation *op) {
  return rejectSemanticAttrs(op, {"fastmath", "roundingmode", "overflowFlags"});
}

LogicalResult DivIOp::verify() { return verifySignedIntegerBinaryOp(*this); }
LogicalResult FloorDivOp::verify() {
  return verifySignedIntegerBinaryOp(*this);
}
LogicalResult CeilDivOp::verify() { return verifySignedIntegerBinaryOp(*this); }
LogicalResult RemIOp::verify() { return verifySignedIntegerBinaryOp(*this); }
LogicalResult ShrOp::verify() { return verifySignedIntegerBinaryOp(*this); }

static LogicalResult verifyPlainIntegerBinaryOp(Operation *op) {
  return rejectSemanticAttrs(
      op, {"fastmath", "roundingmode", "overflowFlags", "signedness"});
}

LogicalResult AndOp::verify() { return verifyPlainIntegerBinaryOp(*this); }
LogicalResult OrOp::verify() { return verifyPlainIntegerBinaryOp(*this); }
LogicalResult XorOp::verify() { return verifyPlainIntegerBinaryOp(*this); }

LogicalResult ShlOp::verify() {
  return rejectSemanticAttrs(*this, {"fastmath", "roundingmode", "signedness"});
}

LogicalResult DivFOp::verify() { return verifyFloatArithmeticOp(*this); }
LogicalResult RemFOp::verify() { return verifyFloatArithmeticOp(*this); }

LogicalResult CmpIOp::verify() {
  if (failed(rejectSemanticAttrs(
          *this, {"fastmath", "roundingmode", "overflowFlags"}))) {
    return failure();
  }
  switch (getPredicateAttr().getValue()) {
  case ScalarCmpPredicate::Eq:
  case ScalarCmpPredicate::Ne:
  case ScalarCmpPredicate::Lt:
  case ScalarCmpPredicate::Le:
  case ScalarCmpPredicate::Gt:
  case ScalarCmpPredicate::Ge:
    return success();
  default:
    return emitOpError() << "predicate "
                         << stringifyScalarCmpPredicate(
                                getPredicateAttr().getValue())
                         << " requires floating-point operands";
  }
}

LogicalResult CmpFOp::verify() {
  return rejectSemanticAttrs(*this,
                             {"signedness", "roundingmode", "overflowFlags"});
}

LogicalResult MaxIOp::verify() {
  return rejectSemanticAttrs(*this,
                             {"fastmath", "roundingmode", "overflowFlags"});
}

LogicalResult MinIOp::verify() {
  return rejectSemanticAttrs(*this,
                             {"fastmath", "roundingmode", "overflowFlags"});
}

LogicalResult AbsIOp::verify() {
  return rejectSemanticAttrs(*this,
                             {"fastmath", "roundingmode", "overflowFlags"});
}

static LogicalResult verifyFloatExtremumOp(Operation *op) {
  return rejectSemanticAttrs(op,
                             {"signedness", "roundingmode", "overflowFlags"});
}

LogicalResult MaxFOp::verify() { return verifyFloatExtremumOp(*this); }
LogicalResult MinFOp::verify() { return verifyFloatExtremumOp(*this); }
LogicalResult MaximumOp::verify() { return verifyFloatExtremumOp(*this); }
LogicalResult MinimumOp::verify() { return verifyFloatExtremumOp(*this); }
LogicalResult AbsFOp::verify() {
  return rejectSemanticAttrs(
      *this, {"signedness", "roundingmode", "overflowFlags", "fastmath"});
}

static LogicalResult verifyConversionShape(Operation *op, Type srcType,
                                           Type dstType) {
  auto srcVectorType = dyn_cast<VectorType>(srcType);
  auto dstVectorType = dyn_cast<VectorType>(dstType);
  bool mixesScalarAndVector =
      static_cast<bool>(srcVectorType) != static_cast<bool>(dstVectorType);
  bool hasOpaquePackedType = isPTOLowPrecisionType(srcType) ||
                             isPTOLowPrecisionType(dstType);
  if (mixesScalarAndVector && !hasOpaquePackedType) {
    return op->emitOpError() << "requires both types to be scalar or both to "
                                "be builtin vectors; got "
                             << srcType << " -> " << dstType;
  }
  if (srcVectorType && dstVectorType &&
      (srcVectorType.getShape() != dstVectorType.getShape() ||
       srcVectorType.getScalableDims() != dstVectorType.getScalableDims())) {
    return op->emitOpError() << "requires source and destination vectors to "
                                "have the same shape; got "
                             << srcType << " -> " << dstType;
  }
  return success();
}

LogicalResult ExtIOp::verify() {
  if (failed(rejectSemanticAttrs(
          *this, {"overflowFlags", "roundingmode", "fastmath"}))) {
    return failure();
  }
  if (failed(verifyConversionShape(getOperation(), getSrc().getType(),
                                   getDst().getType()))) {
    return failure();
  }
  auto srcType = cast<IntegerType>(getElementTypeOrSelf(getSrc().getType()));
  auto dstType = cast<IntegerType>(getElementTypeOrSelf(getDst().getType()));
  bool isNotExtension = srcType.getWidth() >= dstType.getWidth();
  if (isNotExtension) {
    return emitOpError(
        "requires destination integer width greater than source");
  }
  return success();
}

LogicalResult TruncIOp::verify() {
  if (failed(rejectSemanticAttrs(*this,
                                 {"signedness", "roundingmode", "fastmath"}))) {
    return failure();
  }
  if (failed(verifyConversionShape(getOperation(), getSrc().getType(),
                                   getDst().getType()))) {
    return failure();
  }
  auto srcType = cast<IntegerType>(getElementTypeOrSelf(getSrc().getType()));
  auto dstType = cast<IntegerType>(getElementTypeOrSelf(getDst().getType()));
  bool isNotTruncation = srcType.getWidth() <= dstType.getWidth();
  if (isNotTruncation) {
    return emitOpError("requires destination integer width less than source");
  }
  return success();
}

LogicalResult FToFOp::verify() {
  if (failed(rejectSemanticAttrs(*this, {"signedness", "overflowFlags"}))) {
    return failure();
  }
  bool isSimt = isInsideSimtExecutionScope(getOperation());
  bool lowPrecision = hasLowPrecisionConversionPayload(getSrc().getType()) ||
                      hasLowPrecisionConversionPayload(getDst().getType());
  bool requiresSimt = getSaturation() == Saturation::Enable || lowPrecision;
  if (requiresSimt && !isSimt) {
    return emitOpError(
        "saturation and PTO packed floating-point conversion are only valid "
        "in a SIMT execution scope");
  }
  bool usesSimtOnlyRounding =
      getRoundingmodeAttr() &&
      (getRoundingmode() == FloatRoundingMode::to_odd ||
       getRoundingmode() == FloatRoundingMode::hybrid);
  if (!isSimt && usesSimtOnlyRounding) {
    return emitOpError(
        "to_odd and hybrid rounding modes are only valid in a SIMT "
        "execution scope");
  }
  if (failed(verifyConversionShape(getOperation(), getSrc().getType(),
                                   getDst().getType()))) {
    return failure();
  }
  Type srcElement = getElementTypeOrSelf(getSrc().getType());
  Type dstElement = getElementTypeOrSelf(getDst().getType());
  if (srcElement == dstElement) {
    return emitOpError("requires different source and destination formats");
  }
  auto srcType = dyn_cast<FloatType>(srcElement);
  auto dstType = dyn_cast<FloatType>(dstElement);
  if (!requiresSimtHardwareConversion(getOperation(), getSrc().getType(),
                                      getDst().getType(), getRoundingmodeAttr(),
                                      getSaturation()) &&
      getRoundingmodeAttr() && srcType && dstType &&
      srcType.getWidth() <= dstType.getWidth()) {
    return emitOpError(
        "accepts a rounding mode only when the destination is narrower");
  }
  return verifySimtConversionIfNeeded(getOperation(), getSrc().getType(),
                                      getDst().getType(), getRoundingmodeAttr(),
                                      getSaturation(), {});
}

LogicalResult FToIOp::verify() {
  if (failed(rejectSemanticAttrs(*this, {"overflowFlags", "fastmath"}))) {
    return failure();
  }
  bool isSimt = isInsideSimtExecutionScope(getOperation());
  bool lowPrecision = hasLowPrecisionConversionPayload(getSrc().getType());
  bool requiresSimt = getSaturation() == Saturation::Enable || lowPrecision;
  if (requiresSimt && !isSimt) {
    return emitOpError(
        "saturation and PTO packed floating-point conversion are only valid "
        "in a SIMT execution scope");
  }
  bool hasRoundingMode = static_cast<bool>(getRoundingmodeAttr());
  if (hasRoundingMode && !isSimt) {
    return emitOpError("rounding mode is only valid in a SIMT execution scope "
                       "for floating-point-to-integer conversion");
  }
  if (failed(verifyConversionShape(getOperation(), getSrc().getType(),
                                   getDst().getType()))) {
    return failure();
  }
  return verifySimtConversionIfNeeded(getOperation(), getSrc().getType(),
                                      getDst().getType(), getRoundingmodeAttr(),
                                      getSaturation(), getSignednessAttr());
}

LogicalResult IToFOp::verify() {
  if (failed(rejectSemanticAttrs(*this, {"overflowFlags", "fastmath"}))) {
    return failure();
  }
  bool isSimt = isInsideSimtExecutionScope(getOperation());
  bool lowPrecision = hasLowPrecisionConversionPayload(getDst().getType());
  bool requiresSimt = getSaturation() == Saturation::Enable || lowPrecision;
  if (requiresSimt && !isSimt) {
    return emitOpError(
        "saturation and PTO packed floating-point conversion are only valid "
        "in a SIMT execution scope");
  }
  bool hasRoundingMode = static_cast<bool>(getRoundingmodeAttr());
  if (hasRoundingMode && !isSimt) {
    return emitOpError("rounding mode is only valid in a SIMT execution scope "
                       "for integer-to-floating-point conversion");
  }
  if (failed(verifyConversionShape(getOperation(), getSrc().getType(),
                                   getDst().getType()))) {
    return failure();
  }
  return verifySimtConversionIfNeeded(getOperation(), getSrc().getType(),
                                      getDst().getType(), getRoundingmodeAttr(),
                                      getSaturation(), getSignednessAttr());
}

static LogicalResult verifyIndexCast(Operation *op, Type srcType,
                                     Type dstType) {
  auto srcVector = dyn_cast<VectorType>(srcType);
  auto dstVector = dyn_cast<VectorType>(dstType);
  bool mixesScalarAndVector =
      static_cast<bool>(srcVector) != static_cast<bool>(dstVector);
  if (mixesScalarAndVector) {
    return op->emitOpError() << "requires both types to be scalar or both to "
                                "be builtin vectors; got "
                             << srcType << " -> " << dstType;
  }
  if (srcVector &&
      (srcVector.getShape() != dstVector.getShape() ||
       srcVector.getScalableDims() != dstVector.getScalableDims())) {
    return op->emitOpError() << "requires source and destination vectors to "
                                "have the same shape; got "
                             << srcType << " -> " << dstType;
  }
  Type srcElement = getElementTypeOrSelf(srcType);
  Type dstElement = getElementTypeOrSelf(dstType);
  bool convertsIndexToInteger =
      isa<IndexType>(srcElement) && isa<IntegerType>(dstElement);
  bool convertsIntegerToIndex =
      isa<IntegerType>(srcElement) && isa<IndexType>(dstElement);
  if (convertsIndexToInteger || convertsIntegerToIndex) {
    return success();
  }
  return op->emitOpError() << "requires exactly one index element type and one "
                              "integer element type; got "
                           << srcType << " -> " << dstType;
}

LogicalResult IndexCastOp::verify() {
  if (failed(rejectSemanticAttrs(
          *this, {"fastmath", "roundingmode", "overflowFlags"}))) {
    return failure();
  }
  return verifyIndexCast(getOperation(), getSrc().getType(),
                         getDst().getType());
}

LogicalResult SelectOp::verify() {
  if (failed(rejectSemanticAttrs(*this, {"fastmath", "roundingmode",
                                         "overflowFlags", "signedness"}))) {
    return failure();
  }
  Type conditionType = getCondition().getType();
  Type valueType = getTrueValue().getType();
  if (conditionType.isInteger(1)) {
    return success();
  }
  auto conditionVector = dyn_cast<VectorType>(conditionType);
  auto valueVector = dyn_cast<VectorType>(valueType);
  if (!conditionVector || !conditionVector.getElementType().isInteger(1)) {
    return emitOpError() << "requires an i1 or builtin-vector-of-i1 condition";
  }
  bool hasMismatchedVectorShape =
      !valueVector || conditionVector.getShape() != valueVector.getShape() ||
      conditionVector.getScalableDims() != valueVector.getScalableDims();
  if (hasMismatchedVectorShape) {
    return emitOpError()
           << "requires a vector condition to match the selected vector shape";
  }
  return success();
}

} // namespace pto
} // namespace mlir

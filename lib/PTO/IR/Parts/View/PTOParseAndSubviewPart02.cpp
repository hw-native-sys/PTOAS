// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Split from PTOParseAndSubview.cpp; kept as a fragment included by PTOParseAndSubview.cpp.
// Intentionally has no local includes.

using namespace mlir;
using namespace mlir::pto;

ParseResult mlir::pto::TRowExpandSubOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTRowExpandBinaryLikeOp(parser, result);
}

void mlir::pto::TRowExpandSubOp::print(OpAsmPrinter &p) {
  printTRowExpandBinaryLikeOp(p, getOperation(), getSrc0(), getSrc1(), getTmp(),
                              getDst());
}

ParseResult mlir::pto::TRowExpandExpdifOp::parse(OpAsmParser &parser,
                                                 OperationState &result) {
  return parseTRowExpandBinaryLikeOp(parser, result);
}

void mlir::pto::TRowExpandExpdifOp::print(OpAsmPrinter &p) {
  printTRowExpandBinaryLikeOp(p, getOperation(), getSrc0(), getSrc1(), getTmp(),
                              getDst());
}

ParseResult mlir::pto::TRowExpandMaxOp::parse(OpAsmParser &parser,
                                              OperationState &result) {
  return parseTRowExpandBinaryLikeOp(parser, result);
}

void mlir::pto::TRowExpandMaxOp::print(OpAsmPrinter &p) {
  printTRowExpandBinaryLikeOp(p, getOperation(), getSrc0(), getSrc1(), getTmp(),
                              getDst());
}

ParseResult mlir::pto::TRowExpandMinOp::parse(OpAsmParser &parser,
                                              OperationState &result) {
  return parseTRowExpandBinaryLikeOp(parser, result);
}

void mlir::pto::TRowExpandMinOp::print(OpAsmPrinter &p) {
  printTRowExpandBinaryLikeOp(p, getOperation(), getSrc0(), getSrc1(), getTmp(),
                              getDst());
}

static FailureOr<Type> verifyTRowExpandBinaryCore(Operation *op, Type src0Ty,
                                                  Type src1Ty, Type dstTy,
                                                  Type tmpTy, bool hasTmp) {
  if (failed(verifyTileBufCommon(op, src0Ty, "src0")) ||
      failed(verifyTileBufCommon(op, src1Ty, "src1")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst")))
    return failure();
  if (hasTmp && failed(verifyTileBufCommon(op, tmpTy, "tmp")))
    return failure();
  if (failed(verifyTileBufSameElemType(op, src0Ty, dstTy, "src0", "dst")))
    return failure();
  if (getElemTy(src0Ty) != getElemTy(src1Ty)) {
    op->emitOpError("expects src0 and src1 to have the same element type");
    return failure();
  }
  if (!isRowMajorTileBuf(dstTy)) {
    op->emitOpError("expects dst to use row-major layout");
    return failure();
  }
  return getElemTy(src0Ty);
}

static bool isTRowExpandBinaryElemSupported(Type elem, PTOArch targetArch,
                                            bool allowA2A3IntegerTypes) {
  if (elem.isF16() || elem.isF32())
    return true;
  if (targetArch == PTOArch::A5)
    return elem.isInteger(kPTOI8BitWidth) || elem.isInteger(kPTOI16BitWidth) || elem.isInteger(kPTOI32BitWidth);
  return allowA2A3IntegerTypes &&
         (elem.isInteger(kPTOI16BitWidth) || elem.isInteger(kPTOI32BitWidth));
}

static LogicalResult verifyTRowExpandBinaryElemType(Operation *op, Type elem,
                                                    PTOArch targetArch,
                                                    bool allowA2A3IntegerTypes,
                                                    StringRef a2a3Message,
                                                    StringRef a5Message) {
  if (isTRowExpandBinaryElemSupported(elem, targetArch,
                                      allowA2A3IntegerTypes))
    return success();
  if (targetArch == PTOArch::A5)
    return op->emitOpError(a5Message);
  return op->emitOpError(a2a3Message);
}

static LogicalResult verifyTRowExpandBinaryLikeOp(Operation *op, Type src0Ty,
                                                  Type src1Ty, Type dstTy,
                                                  Type tmpTy, bool hasTmp,
                                                  PTOArch targetArch,
                                                  bool allowA2A3IntegerTypes,
                                                  StringRef a2a3Message,
                                                  StringRef a5Message) {
  FailureOr<Type> elemOr =
      verifyTRowExpandBinaryCore(op, src0Ty, src1Ty, dstTy, tmpTy, hasTmp);
  if (failed(elemOr))
    return failure();
  return verifyTRowExpandBinaryElemType(op, *elemOr, targetArch,
                                        allowA2A3IntegerTypes, a2a3Message,
                                        a5Message);
}

static LogicalResult verifyTRowExpandAddSrc1Shape(Operation *op, Type src1Ty,
                                                  Type dstTy, Type elem) {
  auto src1Valid = getValidShapeVec(src1Ty);
  auto dstValid = getValidShapeVec(dstTy);
  if (src1Valid.size() != kPTORowColRank || dstValid.size() != kPTORowColRank)
    return op->emitOpError(
        "expects src1 and dst to have rank-2 valid_shape");
  if (src1Valid[0] != ShapedType::kDynamic &&
      dstValid[0] != ShapedType::kDynamic && src1Valid[0] != dstValid[0])
    return op->emitOpError(
        "expects src1 valid_shape[0] to equal dst valid_shape[0]");
  bool src1IsRowMajor = isRowMajorTileBuf(src1Ty);
  int64_t expectedCol =
      elem.isInteger(kPTOI8BitWidth) ? 32
                        : ((elem.isF16() || elem.isInteger(kPTOI16BitWidth)) ? 16 : 8);
  int64_t src1Col = src1Valid[1];
  if (src1IsRowMajor) {
    if (src1Col != ShapedType::kDynamic && src1Col != expectedCol)
      return op->emitOpError(
          "expects row-major src1 valid_shape[1] to be 32/sizeof(dtype)");
    return success();
  }
  if (src1Col != ShapedType::kDynamic && src1Col != 1)
    return op->emitOpError(
        "expects non-row-major src1 valid_shape[1] to be 1");
  return success();
}

mlir::LogicalResult mlir::pto::TRowExpandDivOp::verify() {
  auto verifyByArch = [this](PTOArch targetArch) -> LogicalResult {
    return verifyTRowExpandBinaryLikeOp(
        *this, getSrc0().getType(), getSrc1().getType(), getDst().getType(),
        getTmp() ? getTmp().getType() : Type{}, static_cast<bool>(getTmp()),
        targetArch, /*allowA2A3IntegerTypes=*/false,
        "expects element type to be f16 or f32",
        "expects A5 trowexpanddiv element type to be i8/i16/i32/f16/f32");
  };
  auto verifyA2A3 = [&verifyByArch]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&verifyByArch]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TRowExpandMulOp::verify() {
  auto verifyByArch = [this](PTOArch targetArch) -> LogicalResult {
    return verifyTRowExpandBinaryLikeOp(
        *this, getSrc0().getType(), getSrc1().getType(), getDst().getType(),
        getTmp() ? getTmp().getType() : Type{}, static_cast<bool>(getTmp()),
        targetArch, /*allowA2A3IntegerTypes=*/true,
        "expects A2/A3 trowexpandmul element type to be i16/i32/f16/f32",
        "expects A5 trowexpandmul element type to be i8/i16/i32/f16/f32");
  };
  auto verifyA2A3 = [&verifyByArch]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&verifyByArch]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TRowExpandSubOp::verify() {
  auto verifyByArch = [this](PTOArch targetArch) -> LogicalResult {
    return verifyTRowExpandBinaryLikeOp(
        *this, getSrc0().getType(), getSrc1().getType(), getDst().getType(),
        getTmp() ? getTmp().getType() : Type{}, static_cast<bool>(getTmp()),
        targetArch, /*allowA2A3IntegerTypes=*/true,
        "expects A2/A3 trowexpandsub element type to be i16/i32/f16/f32",
        "expects A5 trowexpandsub element type to be i8/i16/i32/f16/f32");
  };
  auto verifyA2A3 = [&verifyByArch]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&verifyByArch]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TRowExpandAddOp::verify() {
  auto verifyByArch = [this](PTOArch targetArch) -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    FailureOr<Type> elemOr =
        verifyTRowExpandBinaryCore(*this, src0Ty, src1Ty, dstTy, Type{}, false);
    if (failed(elemOr))
      return failure();
    if (failed(verifyTileBufSameValidShape(*this, src0Ty, dstTy, "src0", "dst")))
      return failure();
    if (!isRowMajorTileBuf(src0Ty))
      return emitOpError("expects src0 to use row-major layout");
    if (failed(verifyTRowExpandBinaryElemType(
            *this, *elemOr, targetArch, /*allowA2A3IntegerTypes=*/true,
            "expects A2/A3 trowexpandadd element type to be i16/i32/f16/f32",
            "expects A5 trowexpandadd element type to be i8/i16/i32/f16/f32")))
      return failure();
    return verifyTRowExpandAddSrc1Shape(*this, src1Ty, dstTy, *elemOr);
  };
  auto verifyA2A3 = [&verifyByArch]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&verifyByArch]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyTRowExpandReduceTypes(Operation *op, Type src0Ty,
                                                 Type src1Ty, Type dstTy,
                                                 Type tmpTy, bool hasTmp) {
  if (failed(verifyTileBufCommon(op, src0Ty, "src0")) ||
      failed(verifyTileBufCommon(op, src1Ty, "src1")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst")))
    return failure();
  if (!hasTmp)
    return success();
  if (failed(verifyTileBufCommon(op, tmpTy, "tmp")))
    return failure();
  if (getElemTy(tmpTy) != getElemTy(dstTy))
    return op->emitOpError()
           << "expects tmp and dst to have the same element type";
  return success();
}

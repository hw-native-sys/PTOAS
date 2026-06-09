// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Split from PTOVerifyArithmeticB.cpp; kept as a fragment included by PTOVerifyArithmeticB.cpp.
// Intentionally has no local includes.

using namespace mlir;
using namespace mlir::pto;

static LogicalResult verifyTCmpSA2A3(TCmpSOp op) {
  if (failed(verifyTCmpSCommon(op)))
    return failure();
  Type elemTy = getElemTy(op.getSrc().getType());
  if (!(elemTy.isInteger(kPTOI16BitWidth) || elemTy.isInteger(kPTOI32BitWidth) || elemTy.isF16() ||
        elemTy.isF32())) {
    return op.emitOpError(
        "expects A2/A3 tcmps input element type to be i16/i32/f16/f32");
  }
  return success();
}

static LogicalResult verifyTCmpSA5(TCmpSOp op) {
  if (failed(verifyTCmpSCommon(op)))
    return failure();
  Type elemTy = getElemTy(op.getSrc().getType());
  if (!(elemTy.isInteger(kPTOI8BitWidth) || elemTy.isInteger(kPTOI16BitWidth) || elemTy.isInteger(kPTOI32BitWidth) ||
        elemTy.isF16() || elemTy.isF32())) {
    return op.emitOpError(
        "expects A5 tcmps input element type to be i8/i16/i32/f16/f32");
  }
  return success();
}

LogicalResult pto::TCmpSOp::verify() {
  auto verifyA2A3 = [this]() -> LogicalResult { return verifyTCmpSA2A3(*this); };
  auto verifyA5 = [this]() -> LogicalResult { return verifyTCmpSA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
LogicalResult pto::TColExpandOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyNDStyleVecTile(*this, srcTy, "src")) ||
      failed(verifyNDStyleVecTile(*this, dstTy, "dst")))
    return failure();
  if (getElemTy(srcTy) != getElemTy(dstTy))
    return emitOpError("expects src and dst to have the same element type");
  if (!isSupportedVecElemType(getElemTy(srcTy), /*allowBf16=*/true,
                              /*allowInt8=*/true))
    return emitOpError("expects tcolexpand element type to be supported");
  auto srcValid = getValidShapeVec(getSrc());
  auto dstValid = getValidShapeVec(getDst());
  if (srcValid.size() != kPTORowColRank || dstValid.size() != kPTORowColRank)
    return emitOpError("expects src and dst to have rank-2 valid_shape");
  if (srcValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
      srcValid[1] != dstValid[1])
    return emitOpError("expects src and dst to have the same valid_shape[1]");
  return success();
}
static LogicalResult verifyTColExpandBinaryLikeOp(Operation *op, Type t0, Type t1,
                                                  Type td, PTOArch targetArch,
                                                  StringRef opName,
                                                  bool allowIntegerTypes) {
  if (!isPTOShapedLike(t0) || !isPTOShapedLike(t1) || !isPTOShapedLike(td))
    return op->emitOpError("expects src0/src1/dst to be PTO shaped-like types");

  Type e0 = getElemTy(t0);
  Type e1 = getElemTy(t1);
  Type ed = getElemTy(td);
  if (!e0 || !e1 || !ed)
    return op->emitOpError("failed to get element type for src0/src1/dst");

  auto isSupportedElem = [allowIntegerTypes, targetArch](Type elemTy) {
    if (elemTy.isF16() || elemTy.isF32())
      return true;
    if (!allowIntegerTypes)
      return false;
    if (elemTy.isInteger(kPTOI16BitWidth) || elemTy.isInteger(kPTOI32BitWidth))
      return true;
    return targetArch == PTOArch::A5 && elemTy.isInteger(kPTOI8BitWidth);
  };
  if (!isSupportedElem(e0) || !isSupportedElem(e1) || !isSupportedElem(ed)) {
    if (!allowIntegerTypes)
      return op->emitOpError() << "expects " << opName
                               << " element type to be f16 or f32";
    if (targetArch == PTOArch::A5)
      return op->emitOpError() << "expects A5 " << opName
                               << " element type to be i8/i16/i32/f16/f32";
    return op->emitOpError() << "expects A2/A3 " << opName
                             << " element type to be i16/i32/f16/f32";
  }

  if (failed(verifyTColExpandShapeAndLayout(op, t0, t1, td)))
    return failure();

  return success();
}
LogicalResult pto::TColExpandMulOp::verify() {
  PTOArch arch = getTargetArch(getOperation());
  return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                      getSrc1().getType(), getDst().getType(),
                                      arch, "tcolexpandmul",
                                      /*allowIntegerTypes=*/true);
}
LogicalResult pto::TColExpandAddOp::verify() {
  PTOArch arch = getTargetArch(getOperation());
  return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                      getSrc1().getType(), getDst().getType(),
                                      arch, "tcolexpandadd",
                                      /*allowIntegerTypes=*/true);
}
LogicalResult pto::TColExpandDivOp::verify() {
  auto verifyByArch = [this](PTOArch targetArch) -> LogicalResult {
    bool allowIntegerTypes = (targetArch == PTOArch::A5);
    return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        targetArch, "tcolexpanddiv",
                                        /*allowIntegerTypes=*/allowIntegerTypes);
  };
  auto verifyA2A3 = [&verifyByArch]() -> LogicalResult {
    return verifyByArch(PTOArch::A3);
  };
  auto verifyA5 = [&verifyByArch]() -> LogicalResult {
    return verifyByArch(PTOArch::A5);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
LogicalResult pto::TColExpandSubOp::verify() {
  PTOArch arch = getTargetArch(getOperation());
  return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                      getSrc1().getType(), getDst().getType(),
                                      arch, "tcolexpandsub",
                                      /*allowIntegerTypes=*/true);
}
LogicalResult pto::TColExpandExpdifOp::verify() {
  PTOArch arch = getTargetArch(getOperation());
  return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                      getSrc1().getType(), getDst().getType(),
                                      arch, "tcolexpandexpdif",
                                      /*allowIntegerTypes=*/false);
}
LogicalResult pto::TColExpandMaxOp::verify() {
  PTOArch arch = getTargetArch(getOperation());
  return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                      getSrc1().getType(), getDst().getType(),
                                      arch, "tcolexpandmax",
                                      /*allowIntegerTypes=*/true);
}
LogicalResult pto::TColExpandMinOp::verify() {
  PTOArch arch = getTargetArch(getOperation());
  return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                      getSrc1().getType(), getDst().getType(),
                                      arch, "tcolexpandmin",
                                      /*allowIntegerTypes=*/true);
}
LogicalResult pto::TColMaxOp::verify() {
  return verifyTColReductionOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(),
      /*requireNonZeroSrcOnA2A3=*/false, /*requireNonZeroSrcOnA5=*/true,
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tcolmax element type to be f16/f32/i16/i32",
      "expects A5 tcolmax element type to be i8/i16/i32/f16/bf16/f32");
}

LogicalResult pto::TColArgMaxOp::verify() {
  auto verifyByArch = [this]() -> LogicalResult {
    return verifyTColArgReductionOpCommon(*this, getSrc().getType(),
                                          getTmp().getType(), getDst().getType());
  };

  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

LogicalResult pto::TColMinOp::verify() {
  return verifyTColReductionOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(),
      /*requireNonZeroSrcOnA2A3=*/false, /*requireNonZeroSrcOnA5=*/true,
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tcolmin element type to be f16/f32/i16/i32",
      "expects A5 tcolmin element type to be i8/i16/i32/f16/bf16/f32");
}

LogicalResult pto::TColArgMinOp::verify() {
  auto verifyByArch = [this]() -> LogicalResult {
    return verifyTColArgReductionOpCommon(*this, getSrc().getType(),
                                          getTmp().getType(), getDst().getType());
  };

  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

static ParseResult parseTColSumFormatWithTmp(OpAsmParser &parser,
                                             OperationState &result,
                                             Type &srcTy, Type &tmpTy) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (parser.parseColonType(srcTy) || parser.parseComma() ||
      parser.parseType(tmpTy))
    return failure();
  return success();
}

static ParseResult parseTColSumFormatWithoutTmp(OpAsmParser &parser, Type &srcTy) {
  return parser.parseColonType(srcTy);
}

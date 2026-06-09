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

static LogicalResult verifyTRowExpandReduceElementType(Operation *op, Type src0Ty,
                                                       Type src1Ty, Type dstTy,
                                                       PTOArch targetArch,
                                                       StringRef opName,
                                                       bool allowIntegerTypes,
                                                       Type &elem) {
  elem = getElemTy(dstTy);
  if (!elem || getElemTy(src0Ty) != elem || getElemTy(src1Ty) != elem)
    return op->emitOpError(
        "expects src0, src1, and dst to have the same element type");
  bool supported = elem.isF16() || elem.isF32() ||
                   (allowIntegerTypes &&
                    (elem.isInteger(kPTOI16BitWidth) || elem.isInteger(kPTOI32BitWidth) ||
                     (targetArch == PTOArch::A5 && elem.isInteger(kPTOI8BitWidth))));
  if (supported)
    return success();
  if (!allowIntegerTypes)
    return op->emitOpError() << "expects " << opName
                             << " element type to be f16 or f32";
  if (targetArch == PTOArch::A5)
    return op->emitOpError() << "expects A5 " << opName
                             << " element type to be i8/i16/i32/f16/f32";
  return op->emitOpError() << "expects A2/A3 " << opName
                           << " element type to be i16/i32/f16/f32";
}

static bool validShapeMatches(ArrayRef<int64_t> lhs, ArrayRef<int64_t> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [l, r] : llvm::zip(lhs, rhs)) {
    if (l != ShapedType::kDynamic && r != ShapedType::kDynamic && l != r)
      return false;
  }
  return true;
}

static LogicalResult verifyNonZeroRank2ValidShape(Operation *op,
                                                  ArrayRef<int64_t> valid,
                                                  StringRef name) {
  if (valid.size() != kPTORowColRank)
    return op->emitOpError() << "expects " << name
                             << " to have rank-2 valid_shape";
  if (valid[0] != ShapedType::kDynamic && valid[0] == 0)
    return op->emitOpError() << "expects " << name
                             << " valid_shape[0] to be non-zero";
  if (valid[1] != ShapedType::kDynamic && valid[1] == 0)
    return op->emitOpError() << "expects " << name
                             << " valid_shape[1] to be non-zero";
  return success();
}

static LogicalResult verifyTRowExpandBroadcastOperand(
    Operation *op, Type elem, Type operandTy, ArrayRef<int64_t> operandValid,
    ArrayRef<int64_t> dstValid, StringRef operandName,
    bool requireNonRowMajor) {
  if (operandValid[0] != ShapedType::kDynamic &&
      dstValid[0] != ShapedType::kDynamic && operandValid[0] != dstValid[0]) {
    return op->emitOpError() << "expects " << operandName
                             << " valid_shape[0] to equal dst valid_shape[0]";
  }
  int64_t expectedCol =
      elem.isInteger(kPTOI8BitWidth) ? 32 : ((elem.isF16() || elem.isInteger(kPTOI16BitWidth)) ? 16 : 8);
  int64_t operandCol = operandValid[1];
  bool operandIsRowMajor = isRowMajorTileBuf(operandTy);
  if (requireNonRowMajor && operandIsRowMajor) {
    return op->emitOpError()
           << "expects " << operandName
           << " to use a non-row-major layout when tmp is present";
  }
  if (operandIsRowMajor) {
    if (operandCol != ShapedType::kDynamic && operandCol != expectedCol) {
      return op->emitOpError()
             << "expects row-major " << operandName
             << " valid_shape[1] to be 32/sizeof(dtype)";
    }
    return success();
  }
  if (operandCol != ShapedType::kDynamic && operandCol != 1) {
    return op->emitOpError() << "expects non-row-major " << operandName
                             << " valid_shape[1] to be 1";
  }
  return success();
}

static LogicalResult verifyTRowExpandFullAndBroadcast(
    Operation *op, Type elem, ArrayRef<int64_t> dstValid, Type fullTy,
    ArrayRef<int64_t> fullValid, StringRef fullName, Type broadcastTy,
    ArrayRef<int64_t> broadcastValid, StringRef broadcastName,
    bool requireNonRowMajorBroadcast) {
  if (!isRowMajorTileBuf(fullTy))
    return op->emitOpError() << "expects " << fullName
                             << " to use row-major layout when it matches dst";
  if (fullValid[0] != ShapedType::kDynamic &&
      dstValid[0] != ShapedType::kDynamic && fullValid[0] != dstValid[0])
    return op->emitOpError() << "expects " << fullName
                             << " valid_shape[0] to equal dst valid_shape[0]";
  if (fullValid[1] != ShapedType::kDynamic &&
      dstValid[1] != ShapedType::kDynamic && fullValid[1] != dstValid[1])
    return op->emitOpError() << "expects " << fullName
                             << " valid_shape[1] to equal dst valid_shape[1]";
  return verifyTRowExpandBroadcastOperand(op, elem, broadcastTy, broadcastValid,
                                          dstValid, broadcastName,
                                          requireNonRowMajorBroadcast);
}

static LogicalResult verifyTRowExpandReduceLikeOp(Operation *op, Type src0Ty,
                                                  Type src1Ty, Type dstTy,
                                                  Type tmpTy, bool hasTmp,
                                                  PTOArch targetArch,
                                                  StringRef opName,
                                                  bool allowIntegerTypes) {
  if (failed(verifyTRowExpandReduceTypes(op, src0Ty, src1Ty, dstTy, tmpTy,
                                         hasTmp)))
    return failure();
  Type elem;
  if (failed(verifyTRowExpandReduceElementType(op, src0Ty, src1Ty, dstTy,
                                               targetArch, opName,
                                               allowIntegerTypes, elem)))
    return failure();
  if (!isRowMajorTileBuf(dstTy))
    return op->emitOpError("expects dst to use row-major layout");

  auto src0Valid = getValidShapeVec(src0Ty);
  auto src1Valid = getValidShapeVec(src1Ty);
  auto dstValid = getValidShapeVec(dstTy);
  if (src0Valid.size() != kPTORowColRank || src1Valid.size() != kPTORowColRank || dstValid.size() != kPTORowColRank)
    return op->emitOpError(
        "expects src0, src1, and dst to have rank-2 valid_shape");
  if (failed(verifyNonZeroRank2ValidShape(op, dstValid, "dst")))
    return failure();

  const bool src0MatchesDst = validShapeMatches(src0Valid, dstValid);
  const bool src1MatchesDst = validShapeMatches(src1Valid, dstValid);
  if (hasTmp && targetArch == PTOArch::A5)
    return op->emitOpError("expects A5 form to omit tmp");
  const bool requireNonRowMajorBroadcast =
      hasTmp && targetArch == PTOArch::A3;

  if (src0MatchesDst &&
      succeeded(verifyTRowExpandFullAndBroadcast(
          op, elem, dstValid, src0Ty, src0Valid, "src0", src1Ty, src1Valid,
          "src1", requireNonRowMajorBroadcast)))
    return success();
  if (src1MatchesDst &&
      succeeded(verifyTRowExpandFullAndBroadcast(
          op, elem, dstValid, src1Ty, src1Valid, "src1", src0Ty, src0Valid,
          "src0", requireNonRowMajorBroadcast)))
    return success();

  return op->emitOpError()
         << "expects one of src0/src1 to match dst valid_shape"
         << " and the other to be a per-row scalar vector";
}

mlir::LogicalResult mlir::pto::TRowExpandExpdifOp::verify() {
  auto verifyA2A3 = [this]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        static_cast<bool>(getTmp()), PTOArch::A3,
                                        "trowexpandexpdif",
                                        /*allowIntegerTypes=*/false);
  };
  auto verifyA5 = [this]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        static_cast<bool>(getTmp()), PTOArch::A5,
                                        "trowexpandexpdif",
                                        /*allowIntegerTypes=*/false);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TRowExpandMaxOp::verify() {
  auto verifyA2A3 = [this]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        static_cast<bool>(getTmp()), PTOArch::A3,
                                        "trowexpandmax",
                                        /*allowIntegerTypes=*/true);
  };
  auto verifyA5 = [this]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        static_cast<bool>(getTmp()), PTOArch::A5,
                                        "trowexpandmax",
                                        /*allowIntegerTypes=*/true);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

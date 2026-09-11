// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOVecOpDataMove.h - shared vecop DataMove helpers ----------------===//
//===----------------------------------------------------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Shared helpers. They live in a detail namespace so the original
// unqualified MLIR/LLVM names keep resolving.
//
// DataMove helper group of the vecop per-instruction TUs.
// Internal to lib/PTO/IR/VPTO/vecop/detail; not installed.
//===----------------------------------------------------------------------===//

#ifndef PTO_IR_VPTO_VECOP_DETAIL_DATAMOVE_H
#define PTO_IR_VPTO_VECOP_DETAIL_DATAMOVE_H

#include "VPTOInternal.h"
#include "VPTOVecOpPredicate.h"

namespace mlir::pto::vecop_detail {

using namespace mlir;
using namespace mlir::pto;

  [[maybe_unused]] static bool isSupportedVdupPosition(std::optional<StringRef> position) {
    return !position || *position == "LOWEST" || *position == "HIGHEST";
  }

  [[maybe_unused]] static std::optional<StringRef> getVdupMaskGranularity(Type elementType) {
    if (auto intType = dyn_cast<IntegerType>(elementType)) {
      switch (intType.getWidth()) {
      case mlir::pto::kValue8:
        return StringRef("b8");
      case mlir::pto::kValue16:
        return StringRef("b16");
      case mlir::pto::kValue32:
        return StringRef("b32");
      default:
        return std::nullopt;
      }
    }
    if (elementType.isF16() || elementType.isBF16()) {
      return StringRef("b16");
    }
    if (elementType.isF32()) {
      return StringRef("b32");
    }
    return std::nullopt;
  }

  [[maybe_unused]] static bool isSupportedVtrcRoundMode(StringRef mode) {
    return mode == "R" || mode == "A" || mode == "F" || mode == "C" ||
           mode == "Z";
  }

  [[maybe_unused]] static bool isSupportedPartToken(StringRef part) {
    return part == "LOWER" || part == "HIGHER";
  }

  template <typename OpTy>
  [[maybe_unused]] static LogicalResult verifyPartGranularityOp(OpTy op, bool widening) {
    if (failed(verifyMaskTypeLike(op, op.getInput().getType(), "input type")) ||
        failed(verifyMaskTypeLike(op, op.getResult().getType(), "result type"))) {
      return failure();
    }
    if (!isSupportedPartToken(op.getPart())) {
      return op.emitOpError("requires part to be LOWER or HIGHER");
    }
    auto inputMaskType = cast<MaskType>(op.getInput().getType());
    auto resultMaskType = cast<MaskType>(op.getResult().getType());
    StringRef inputGranularity = inputMaskType.getGranularity();
    StringRef resultGranularity = resultMaskType.getGranularity();
    bool adjacent = widening
                        ? isMaskGranularityAdjacentWidening(inputGranularity, resultGranularity)
                        : isMaskGranularityAdjacentNarrowing(inputGranularity, resultGranularity);
    if (inputGranularity != resultGranularity && !adjacent) {
      return op.emitOpError("requires result mask granularity to match the input or ")
             << (widening ? "widen" : "narrow") << " by one step";
    }
    return success();
  }

  template <typename SelectOp>
  [[maybe_unused]] static LogicalResult verifyLaneSelectOp(SelectOp op) {
    if (failed(verifyVRegTypeLike(op, op.getSrc0().getType(), "src0 type")) ||
        failed(verifyVRegTypeLike(op, op.getSrc1().getType(), "src1 type")) ||
        failed(verifyVRegTypeLike(op, op.getResult().getType(), "result type"))) {
      return failure();
    }

    auto src0Type = cast<VRegType>(op.getSrc0().getType());
    auto src1Type = cast<VRegType>(op.getSrc1().getType());
    auto resultType = cast<VRegType>(op.getResult().getType());
    if (src0Type != resultType) {
      return op.emitOpError("requires src0 and result to have identical vector types");
    }
    if (src1Type.getElementCount() != src0Type.getElementCount()) {
      return op.emitOpError("requires src0/src1 to have identical element counts");
    }
    auto src1ElemType = dyn_cast<IntegerType>(src1Type.getElementType());
    if (!src1ElemType) {
      return op.emitOpError("requires src1 to use integer vector elements");
    }
    if (src1ElemType.getWidth() != getIntOrFloatBitWidth(src0Type.getElementType())) {
      return op.emitOpError("requires src1 integer element width to match src0 element width");
    }
    return success();
  }

  template <typename PairOp>
  [[maybe_unused]] static LogicalResult verifyPairVecResults(PairOp op) {
    if (failed(verifyVRegTypeLike(op, op.getLhs().getType(), "lhs type")) ||
        failed(verifyVRegTypeLike(op, op.getRhs().getType(), "rhs type")) ||
        failed(verifyVRegTypeLike(op, op.getLow().getType(), "low result type")) ||
        failed(verifyVRegTypeLike(op, op.getHigh().getType(), "high result type"))) {
      return failure();
    }
    if (op.getLhs().getType() != op.getRhs().getType() ||
        op.getLhs().getType() != op.getLow().getType() ||
        op.getLhs().getType() != op.getHigh().getType()) {
      return op.emitOpError("requires operands and results to share one vector type");
    }
    return success();
  }

  template <typename PartOp>
  [[maybe_unused]] static LogicalResult verifyPartVecOp(PartOp op) {
    if (failed(verifyVRegTypeLike(op, op.getLhs().getType(), "lhs type")) ||
        failed(verifyVRegTypeLike(op, op.getRhs().getType(), "rhs type")) ||
        failed(verifyVRegTypeLike(op, op.getResult().getType(), "result type"))) {
      return failure();
    }
    if (op.getLhs().getType() != op.getRhs().getType() ||
        op.getLhs().getType() != op.getResult().getType()) {
      return op.emitOpError("requires operands and result to share one vector type");
    }
    if (!isSupportedPartToken(op.getPart())) {
      return op.emitOpError("requires part to be LOWER or HIGHER");
    }
    return success();
  }

  template <typename UnpackOp>
  [[maybe_unused]] static LogicalResult verifyUnpackVecOp(UnpackOp op) {
    if (failed(verifyVRegTypeLike(op, op.getSrc().getType(), "src type")) ||
        failed(verifyVRegTypeLike(op, op.getResult().getType(), "result type"))) {
      return failure();
    }
    auto srcType = cast<VRegType>(op.getSrc().getType());
    auto resultType = cast<VRegType>(op.getResult().getType());
    Type srcElemType = srcType.getElementType();
    Type resultElemType = resultType.getElementType();
    if (!isa<IntegerType>(srcElemType) || !isa<IntegerType>(resultElemType)) {
      return op.emitOpError(
          "currently requires integer source and result element types");
    }
    if (srcType.getElementCount() != resultType.getElementCount() * mlir::pto::kValue2) {
      return op.emitOpError(
          "requires source element count to be twice the result element count");
    }
    unsigned srcWidth = getIntOrFloatBitWidth(srcElemType);
    unsigned resultWidth = getIntOrFloatBitWidth(resultElemType);
    if (srcWidth == 0 || srcWidth * mlir::pto::kValue2 != resultWidth) {
      return op.emitOpError(
          "requires result element width to be twice the source element width");
    }
    return success();
  }

  template <typename OpTy>
  [[maybe_unused]] static LogicalResult verifyVdupMaskGranularityLike(OpTy op, Type elemType) {
    auto expectedGranularity = getVdupMaskGranularity(elemType);
    if (!expectedGranularity) {
      return op.emitOpError(
          "requires element type with supported predicate granularity");
    }
    if (failed(verifyMaskTypeWithGranularityLike(op, op.getMask().getType(),
                                                 "mask type",
                                                 *expectedGranularity))) {
      return failure();
    }
    return success();
  }

  template <typename OpTy>
  [[maybe_unused]] static LogicalResult verifyPackedInterleaveGranularity(OpTy op,
                                                         llvm::StringRef gran) {
    if (failed(verifyMaskTypeWithGranularityLike(op, op.getLhs().getType(),
                                                 "lhs type", gran)) ||
        failed(verifyMaskTypeWithGranularityLike(op, op.getRhs().getType(),
                                                 "rhs type", gran)) ||
        failed(verifyMaskTypeWithGranularityLike(op, op.getLow().getType(),
                                                 "low type", gran)) ||
        failed(verifyMaskTypeWithGranularityLike(op, op.getHigh().getType(),
                                                 "high type", gran))) {
      return failure();
    }
    return success();
  }

} // namespace mlir::pto::vecop_detail

#endif // PTO_IR_VPTO_VECOP_DETAIL_DATAMOVE_H

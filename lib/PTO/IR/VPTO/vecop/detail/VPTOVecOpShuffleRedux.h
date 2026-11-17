// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOVecOpShuffleRedux.h - shared vecop ShuffleRedux helpers --------===//
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
// ShuffleRedux helper group of the vecop per-instruction TUs.
// Internal to lib/PTO/IR/VPTO/vecop/detail; not installed.
//===----------------------------------------------------------------------===//

#ifndef PTO_IR_VPTO_VECOP_DETAIL_SHUFFLEREDUX_H
#define PTO_IR_VPTO_VECOP_DETAIL_SHUFFLEREDUX_H

#include "PTO/Support/CodeConstants.h"
#include "VPTOInternal.h"
#include "VPTOVecOpArith.h"

namespace mlir::pto::vecop_detail {

using namespace mlir;
using namespace mlir::pto;

  inline bool isSupportedShuffleValueType(Type type) {
    if (auto intType = dyn_cast<IntegerType>(type)) {
      return intType.getWidth() == mlir::pto::kValue32 ||
             intType.getWidth() == mlir::pto::kValue64;
    }
    if (auto vecType = dyn_cast<VectorType>(type)) {
      return vecType.getRank() == 1 && vecType.getDimSize(0) == mlir::pto::kValue2 &&
             vecType.getElementType().isF16();
    }
    return type.isF16() || type.isF32();
  }

  // The ISA splits the 32-lane warp with a 5-bit lane_mask whose only legal
  // values are 31, 30, 28, 24, 16 and 0, i.e. group sizes 1, 2, 4, 8, 16, 32.
  inline bool isSupportedShuffleWidth(int64_t width) {
      return width == 1 || width == mlir::pto::kValue2 || width == mlir::pto::kValue4 || width == mlir::pto::kValue8 ||
             width == mlir::pto::kValue16 || width == mlir::pto::kValue32;
  }

  // `bound` bounds the source window inside a group (the ISA calls this field
  // max_ofst in SHFL Rm[12:8]): max_lane = group_base + bound % width. A
  // source lane is usable when src_lane <= max_lane for .down/.bfly/.idx and
  // when src_lane >= max_lane for .up; outside the window the hardware returns
  // the lane's own value. The ISA reduces the field modulo the group size, so
  // accepting bound >= width would silently wrap; reject it.
  inline LogicalResult verifyShuffleSemanticControl(Operation *op,
                                                    Type controlType,
                                                    IntegerAttr widthAttr,
                                                    IntegerAttr boundAttr,
                                                    StringRef ctrlName) {
    if (!isSupportedShuffleValueType(op->getResultTypes().front())) {
      return op->emitOpError()
             << "requires i32, i64, f16, f32 or vector<2xf16> value/result type";
    }
    if (!controlType.isInteger(mlir::pto::kValue32)) {
      return op->emitOpError() << "requires " << ctrlName
                               << " operand to be i32";
    }

    int64_t width = widthAttr.getInt();
    if (!isSupportedShuffleWidth(width)) {
      return op->emitOpError()
             << "requires width to be a power of two in 1, 2, 4, 8, 16 or "
                "32, got "
             << width;
    }
    if (boundAttr) {
      int64_t bound = boundAttr.getInt();
      if (bound < 0 || bound >= width) {
        return op->emitOpError()
               << "requires bound to be in [0, width) i.e. [0, " << width
               << "), got " << bound;
      }
    }
    return success();
  }

  inline LogicalResult verifyReduxSemanticAttrs(Operation *op,
                                                  bool acceptsSignedness) {
    for (StringRef attrName : {"fastmath", "roundingmode", "overflowFlags"}) {
      if (op->hasAttr(attrName)) {
        return op->emitOpError() << "does not accept " << attrName;
      }
    }
    if (!acceptsSignedness && op->hasAttr("signedness")) {
      return op->emitOpError() << "does not accept signedness";
    }
    return success();
  }

  inline LogicalResult verifyRowReductionElementType(Operation *op, Type type) {
    if (auto integerType = dyn_cast<IntegerType>(type)) {
      if (integerType.getWidth() == mlir::pto::kValue16 ||
          integerType.getWidth() == mlir::pto::kValue32) {
        return success();
      }
    } else if (type.isF16() || type.isF32()) {
      return success();
    }
    return op->emitOpError(
        "requires 16-bit or 32-bit integer, f16, or f32 vector element type");
  }

  template <typename ReductionOp>
  [[maybe_unused]] static LogicalResult verifyMinMaxReductionVecOp(ReductionOp op) {
    if (failed(verifyVRegTypeLike(op, op.getInput().getType(), "input")) ||
        failed(verifyVRegTypeLike(op, op.getResult().getType(), "result"))) {
      return failure();
    }
    if (op.getInput().getType() != op.getResult().getType()) {
      return op.emitOpError("input and result must have the same vector type");
    }
    return verifyRowReductionElementType(
        op.getOperation(), cast<VRegType>(op.getInput().getType()).getElementType());
  }

  template <typename ReductionOp>
  [[maybe_unused]] static LogicalResult verifyWideningReductionVecOp(ReductionOp op,
                                                    StringRef opName) {
    if (failed(verifyVRegTypeLike(op, op.getInput().getType(), "input")) ||
        failed(verifyVRegTypeLike(op, op.getResult().getType(), "result"))) {
      return failure();
    }

    auto inputType = dyn_cast<VRegType>(op.getInput().getType());
    auto resultType = dyn_cast<VRegType>(op.getResult().getType());
    if (!inputType || !resultType) {
      return failure();
    }

    Type inputElemType = inputType.getElementType();
    if (failed(verifyRowReductionElementType(op.getOperation(), inputElemType))) {
      return failure();
    }
    Type expectedResultElemType = inputElemType;
    int64_t expectedResultLanes = inputType.getElementCount();
    if (auto inputInt = dyn_cast<IntegerType>(inputElemType)) {
      if (inputInt.getWidth() == mlir::pto::kValue16) {
        expectedResultElemType = IntegerType::get(
            op.getContext(), mlir::pto::kValue32, inputInt.getSignedness());
        expectedResultLanes = inputType.getElementCount() / mlir::pto::kValue2;
      }
    }

    if (resultType.getElementCount() == expectedResultLanes &&
        resultType.getElementType() == expectedResultElemType) {
      return success();
    }

    return op.emitOpError() << opName << " expects result type !pto.vreg<"
                            << expectedResultLanes << "x"
                            << expectedResultElemType
                            << " for input element type " << inputElemType;
  }

  template <typename ReductionOp>
  [[maybe_unused]] static LogicalResult verifyReductionVecOp(ReductionOp op) {
    return verifyUnaryVecOp(op);
  }

  template <typename ReductionOp>
  [[maybe_unused]] static LogicalResult verifyGroupReductionVecOp(ReductionOp op) {
    if (failed(verifyReductionVecOp(op))) {
      return failure();
    }
    auto inputType = cast<VRegType>(op.getInput().getType());
    Type elemType = inputType.getElementType();
    if (auto intType = dyn_cast<IntegerType>(elemType)) {
      if (intType.getWidth() != mlir::pto::kValue8 &&
          intType.getWidth() != mlir::pto::kValue16 &&
          intType.getWidth() != mlir::pto::kValue32) {
        return op.emitOpError(
            "requires 8-bit, 16-bit, or 32-bit integer vector element type");
      }
      return success();
    }
    if (!elemType.isF16() && !elemType.isF32()) {
      return op.emitOpError("requires i16/i32/f16/f32 vector element type");
    }
    return success();
  }

} // namespace mlir::pto::vecop_detail

#endif // PTO_IR_VPTO_VECOP_DETAIL_SHUFFLEREDUX_H

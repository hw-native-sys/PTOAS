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

#include "VPTOInternal.h"
#include "VPTOVecOpArith.h"

namespace mlir::pto::vecop_detail {

using namespace mlir;
using namespace mlir::pto;

  [[maybe_unused]] static bool isSupportedShuffleValueType(Type type) {
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

  [[maybe_unused]] static bool isSupportedReduxValueType(Type type) {
    if (auto intType = dyn_cast<IntegerType>(type)) {
      return intType.getWidth() == mlir::pto::kValue32;
    }
    return type.isF16() || type.isF32();
  }

  [[maybe_unused]] static LogicalResult verifyShuffleSemanticControl(Operation *op,
                                                    Type controlType,
                                                    IntegerAttr widthAttr,
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
    if (width != mlir::pto::kValue16 && width != mlir::pto::kValue32) {
      return op->emitOpError() << "requires width to be 16 or 32";
    }
    return success();
  }

  [[maybe_unused]] static LogicalResult verifyReduxSemanticType(Operation *op, Type valueType,
                                               Attribute signednessAttr,
                                               bool requireSignedness) {
    if (!isSupportedReduxValueType(valueType)) {
      return op->emitOpError()
             << "requires i32, f16 or f32 value/result type";
    }

    auto intType = dyn_cast<IntegerType>(valueType);
    if (!intType) {
      if (signednessAttr) {
        return op->emitOpError()
               << "does not accept signedness for floating-point redux";
      }
      return success();
    }

    if (!signednessAttr && requireSignedness) {
      return op->emitOpError()
             << "requires explicit signedness for integer redux";
    }

    if (!signednessAttr) {
      return success();
    }

    auto signedness = cast<pto::SignednessAttr>(signednessAttr).getValue();
    (void)signedness;
    return success();
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
    Type expectedResultElemType = inputElemType;
    int64_t expectedResultLanes = inputType.getElementCount();
    if (auto inputInt = dyn_cast<IntegerType>(inputElemType)) {
      if (inputInt.getWidth() < mlir::pto::kValue8 ||
          inputInt.getWidth() > mlir::pto::kValue32) {
        return op.emitOpError(
            "requires 8-bit, 16-bit, or 32-bit integer vector element type");
      }
      if (inputInt.getWidth() == mlir::pto::kValue8) {
        expectedResultElemType =
            IntegerType::get(op.getContext(), mlir::pto::kValue16, inputInt.getSignedness());
        expectedResultLanes = inputType.getElementCount() / mlir::pto::kValue2;
      }
      if (inputInt.getWidth() == mlir::pto::kValue16) {
        expectedResultElemType =
            IntegerType::get(op.getContext(), mlir::pto::kValue32, inputInt.getSignedness());
        expectedResultLanes = inputType.getElementCount() / mlir::pto::kValue2;
      }
    } else if (!inputElemType.isF16() && !inputElemType.isF32()) {
      return op.emitOpError("requires i16/i32/f16/f32 vector element type");
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

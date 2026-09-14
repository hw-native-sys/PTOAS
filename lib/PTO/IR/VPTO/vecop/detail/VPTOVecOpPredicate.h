// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOVecOpPredicate.h - shared vecop Predicate helpers --------------===//
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
// Predicate helper group of the vecop per-instruction TUs.
// Internal to lib/PTO/IR/VPTO/vecop/detail; not installed.
//===----------------------------------------------------------------------===//

#ifndef PTO_IR_VPTO_VECOP_DETAIL_PREDICATE_H
#define PTO_IR_VPTO_VECOP_DETAIL_PREDICATE_H

#include "VPTOInternal.h"

namespace mlir::pto::vecop_detail {

using namespace mlir;
using namespace mlir::pto;

  template <typename PltOp>
  [[maybe_unused]] static LogicalResult verifyPredicateLaneCountOp(PltOp op,
                                                  StringRef granularity) {
    if (failed(verifyMaskTypeWithGranularityLike(op, op.getMask().getType(),
                                                 "mask type", granularity))) {
      return failure();
    }
    Type scalarType = op.getScalar().getType();
    auto scalarIntType = dyn_cast<IntegerType>(scalarType);
    if (!scalarIntType || scalarIntType.getWidth() != mlir::pto::kValue32) {
      return op.emitOpError("requires scalar to be i32");
    }
    if (op.getScalarOut().getType() != scalarType) {
      return op.emitOpError("requires scalar_out to match scalar type");
    }
    return success();
  }

  template <typename PltmOp>
  [[maybe_unused]] static LogicalResult verifyPredicateLoopBoundOp(PltmOp op,
                                                  StringRef granularity) {
    if (failed(verifyMaskTypeWithGranularityLike(op, op.getMask().getType(),
                                                 "mask type", granularity))) {
      return failure();
    }
    if (!op.getLoop().getType().isInteger(mlir::pto::kValue16)) {
      return op.emitOpError("requires loop operand to be i16");
    }
    if (!op.getBound().getType().isInteger(mlir::pto::kValue32)) {
      return op.emitOpError("requires bound operand to be i32");
    }
    return success();
  }

  [[maybe_unused]] static bool isMaskGranularityAdjacentWidening(StringRef inputGranularity,
                                                StringRef resultGranularity) {
    return (inputGranularity == "b8" && resultGranularity == "b16") ||
           (inputGranularity == "b16" && resultGranularity == "b32");
  }

  [[maybe_unused]] static bool isMaskGranularityAdjacentNarrowing(StringRef inputGranularity,
                                                 StringRef resultGranularity) {
    return (inputGranularity == "b16" && resultGranularity == "b8") ||
           (inputGranularity == "b32" && resultGranularity == "b16");
  }

  template <typename BinaryMaskOp>
  [[maybe_unused]] static LogicalResult verifyBinaryMaskOp(BinaryMaskOp op) {
    if (failed(verifyMaskTypeLike(op, op.getSrc0().getType(), "src0 type")) ||
        failed(verifyMaskTypeLike(op, op.getSrc1().getType(), "src1 type")) ||
        failed(verifyMaskTypeLike(op, op.getMask().getType(), "mask type")) ||
        failed(verifyMaskTypeLike(op, op.getResult().getType(), "result type"))) {
      return failure();
    }
    return success();
  }

  template <typename ExtremaOp>
  [[maybe_unused]] static LogicalResult verifyExtremaPredicateOp(ExtremaOp op) {
    if (failed(verifyVRegTypeLike(op, op.getInput().getType(), "input type")) ||
        failed(verifyMaskTypeLike(op, op.getMask().getType(), "mask type")) ||
        failed(verifyVRegTypeLike(op, op.getValue().getType(), "value type")) ||
        failed(verifyMaskTypeLike(op, op.getPredicate().getType(),
                                  "predicate type"))) {
      return failure();
    }
    if (op.getInput().getType() != op.getValue().getType()) {
      return op.emitOpError(
          "requires input and value result to share one vector type");
    }
    if (op.getMask().getType() != op.getPredicate().getType()) {
      return op.emitOpError(
          "requires mask and predicate result to share one mask type");
    }

    Type elemType = cast<VRegType>(op.getInput().getType()).getElementType();
    if (elemType.isF16() || elemType.isF32()) {
      return success();
    }
    auto intType = dyn_cast<IntegerType>(elemType);
    if (!intType || (intType.getWidth() != mlir::pto::kValue8 &&
                     intType.getWidth() != mlir::pto::kValue16 &&
                     intType.getWidth() != mlir::pto::kValue32)) {
      return op.emitOpError("requires i8/i16/i32/f16/f32 vector element type");
    }
    return success();
  }

  [[maybe_unused]] static bool isSupportedCmpMode(StringRef mode) {
    return mode == "eq" || mode == "ne" || mode == "lt" || mode == "le" ||
           mode == "gt" || mode == "ge";
  }

} // namespace mlir::pto::vecop_detail

#endif // PTO_IR_VPTO_VECOP_DETAIL_PREDICATE_H

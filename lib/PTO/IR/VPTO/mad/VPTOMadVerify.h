// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOMadVerify.h - Mad-family verification helpers ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Internal to lib/PTO/IR/VPTO/mad; not installed. Split out of
// VPTOMadInternal.h to keep header sizes within the codecheck budget.

#ifndef PTO_IR_VPTO_MAD_VERIFY_H
#define PTO_IR_VPTO_MAD_VERIFY_H

#include "VPTOInternal.h"

namespace mlir::pto::mad_detail {
using namespace mlir;
using namespace mlir::pto;

inline LogicalResult verifyMadPointerKinds(Operation *op, Type lhsTy, Type rhsTy,
                                           Type dstTy,
                                           std::optional<Type> biasTy = std::nullopt) {
  auto lhsType = dyn_cast<pto::PtrType>(lhsTy);
  auto rhsType = dyn_cast<pto::PtrType>(rhsTy);
  auto dstType = dyn_cast<pto::PtrType>(dstTy);
  if (!lhsType || !rhsType || !dstType) {
    return op->emitOpError("requires typed !pto.ptr lhs/rhs/dst operands");
  }

  const auto lhsAS = lhsType.getMemorySpace().getAddressSpace();
  const auto rhsAS = rhsType.getMemorySpace().getAddressSpace();
  const auto dstAS = dstType.getMemorySpace().getAddressSpace();

  const bool isStrongCube =
      lhsAS == pto::AddressSpace::LEFT && rhsAS == pto::AddressSpace::RIGHT &&
      dstAS == pto::AddressSpace::ACC;
  if (!isStrongCube) {
    return op->emitOpError("requires l0a/l0b/l0c-typed lhs/rhs/dst pointers");
  }

  if (!biasTy) {
    return success();
  }

  auto biasType = dyn_cast<pto::PtrType>(*biasTy);
  if (!biasType) {
    return op->emitOpError("requires typed !pto.ptr bias operand");
  }
  if (biasType.getMemorySpace().getAddressSpace() != pto::AddressSpace::BIAS) {
    return op->emitOpError("requires bias pointer in !pto.ptr<..., bt>");
  }
  if (biasType.getElementType() != dstType.getElementType()) {
    return op->emitOpError("requires bias element type to match dst element type");
  }
  return success();
}

inline LogicalResult verifyMadMxCommon(Operation *op, Type lhsTy, Type rhsTy,
                                       Type dstTy,
                                       std::optional<Type> biasTy = std::nullopt) {
  if (failed(verifyMadPointerKinds(op, lhsTy, rhsTy, dstTy, biasTy))) {
    return failure();
  }

  auto lhsType = cast<pto::PtrType>(lhsTy);
  auto rhsType = cast<pto::PtrType>(rhsTy);
  auto dstType = cast<pto::PtrType>(dstTy);
  const auto lhsAS = lhsType.getMemorySpace().getAddressSpace();
  const auto rhsAS = rhsType.getMemorySpace().getAddressSpace();
  const auto dstAS = dstType.getMemorySpace().getAddressSpace();
  const bool isStrongCube =
      lhsAS == pto::AddressSpace::LEFT && rhsAS == pto::AddressSpace::RIGHT &&
      dstAS == pto::AddressSpace::ACC;
  if (!isStrongCube) {
    return op->emitOpError("requires l0a/l0b/l0c-typed lhs/rhs/dst pointers");
  }

  if (!isMxElementType(lhsType.getElementType()) ||
      !isMxElementType(rhsType.getElementType())) {
    return op->emitOpError(
        "requires MX lhs/rhs element types (f8E4M3FN, f8E5M2, f4E1M2x2, or "
        "f4E2M1x2)");
  }
  return success();
}

// Validate the optional runtime flag operands against the op's static
// configuration: bias_init selects the BT buffer as the C-matrix source, so it
// is only meaningful on ops that carry a bias pointer, and a runtime operand

template <typename SemanticT>
static mlir::LogicalResult
verifyMadRuntimeFlagOperands(mlir::Operation *op, SemanticT semantic) {
  if (semantic.getBiasInitValueOrNull() && !semantic.hasBiasOperand()) {
    return op->emitOpError(
        "bias_init operand requires a bias pointer; use mad_bias/mad_mx_bias");
  }
  if (semantic.getUnitFlagValueOrNull() && op->hasAttr("unit_flag_mode")) {
    return op->emitOpError(
        "unit_flag operand and unit_flag(...) clause are mutually exclusive");
  }
  if (semantic.getDisableGemvValueOrNull() && op->hasAttr("disable_gemv")) {
    return op->emitOpError(
        "disable_gemv operand and disable_gemv clause are mutually exclusive");
  }
  return mlir::success();
}

// Collected parse state for the optional runtime flag operands.

inline LogicalResult verifyMadSemanticClauses(Operation *op, Type lhsTy,
                                              Type rhsTy, Type dstTy,
                                              std::optional<Type> biasTy,
                                              std::optional<pto::Tf32Mode> tf32Mode,
                                              std::optional<pto::MadSatMode> satMode,
                                              bool hasNDir) {
  if (failed(verifyMadPointerKinds(op, lhsTy, rhsTy, dstTy, biasTy))) {
    return failure();
  }

  if (auto semantic = dyn_cast<MadSemanticOpInterface>(op)) {
    if (failed(verifyMadRuntimeFlagOperands(op, semantic))) {
      return failure();
    }
  }

  auto lhsType = dyn_cast<pto::PtrType>(lhsTy);
  auto rhsType = dyn_cast<pto::PtrType>(rhsTy);
  auto dstType = dyn_cast<pto::PtrType>(dstTy);
  if (!lhsType || !rhsType || !dstType) {
    return op->emitOpError("requires typed !pto.ptr lhs/rhs/dst operands");
  }

  if (tf32Mode) {
    if (!(lhsType.getElementType().isF32() && rhsType.getElementType().isF32() &&
          dstType.getElementType().isF32())) {
      return op->emitOpError(
          "requires tf32_mode only for f32 lhs/rhs/dst element types");
    }
  }
  if (pto::isPTOHiFloat8Type(lhsType.getElementType()) !=
      pto::isPTOHiFloat8Type(rhsType.getElementType())) {
    return op->emitOpError(
        "requires lhs/rhs to both use hif8 or both use non-hif8 element types");
  }
  if (satMode) {
    auto isFloatLike = [](Type type) {
      if (isa<FloatType>(type)) {
        return true;
      }
      return pto::isPTOLowPrecisionType(type);
    };
    if (!(isFloatLike(lhsType.getElementType()) &&
          isFloatLike(rhsType.getElementType()) &&
          isFloatLike(dstType.getElementType()))) {
      return op->emitOpError(
          "requires sat/nosat only for floating lhs/rhs/dst element types");
    }
  }
  (void)hasNDir;
  return success();
}

} // namespace mlir::pto::mad_detail

#endif // PTO_IR_VPTO_MAD_VERIFY_H

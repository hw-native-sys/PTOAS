// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Scalar and vector-scalar math, reduction, histogram, and extrema
// callee-name templates for the CANN900 LLVM emitter. Split from
// VPTOCANN900LLVMEmitterTemplates.h to keep each header under the
// 500-line header-size limit.

#pragma once

#include "VPTOCANN900LLVMEmitterInternal.h"

namespace mlir::pto::detail {

template <typename ScalarOp> StringRef buildScalarIntrinsicCallee(MLIRContext *context);

template <> inline StringRef buildScalarIntrinsicCallee<pto::PrmtOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.prmt").getValue();
}

template <typename UnaryOp> FailureOr<StringRef> buildUnaryScalarMathCallee(MLIRContext *context, Type valueType);

template <> inline FailureOr<StringRef> buildUnaryScalarMathCallee<pto::AbsFOp>(MLIRContext *context, Type valueType) {
  std::string elem = getLLVMFloatBuiltinFragment(valueType);
  if (elem != "f16" && elem != "f32" && elem != "v2f16" && elem != "v2bf16") {
    return failure();
  }
  return StringAttr::get(context, "llvm.fabs." + elem).getValue();
}

template <> inline FailureOr<StringRef> buildUnaryScalarMathCallee<pto::ExpOp>(MLIRContext *context, Type valueType) {
  std::string elem = getLLVMFloatBuiltinFragment(valueType);
  if (elem != "f32" && elem != "f16" && elem != "v2f16") {
    return failure();
  }
  return StringAttr::get(context, "llvm.exp." + elem).getValue();
}

template <> inline FailureOr<StringRef> buildUnaryScalarMathCallee<pto::LogOp>(MLIRContext *context, Type valueType) {
  std::string elem = getLLVMFloatBuiltinFragment(valueType);
  if (elem != "f32" && elem != "f16" && elem != "v2f16") {
    return failure();
  }
  return StringAttr::get(context, "llvm.log." + elem).getValue();
}

template <> inline FailureOr<StringRef> buildUnaryScalarMathCallee<pto::CeilOp>(MLIRContext *context, Type valueType) {
  std::string elem = getScalarHIVMFloatShortFragment(valueType);
  if (elem.empty()) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.ceil." + elem).getValue();
}

template <> inline FailureOr<StringRef> buildUnaryScalarMathCallee<pto::FloorOp>(MLIRContext *context, Type valueType) {
  std::string elem = getScalarHIVMFloatShortFragment(valueType);
  if (elem.empty()) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.floor." + elem).getValue();
}

template <> inline FailureOr<StringRef> buildUnaryScalarMathCallee<pto::RintOp>(MLIRContext *context, Type valueType) {
  std::string elem = getScalarHIVMFloatShortFragment(valueType);
  if (elem.empty()) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.rint." + elem).getValue();
}

template <> inline FailureOr<StringRef> buildUnaryScalarMathCallee<pto::RoundOp>(MLIRContext *context, Type valueType) {
  std::string elem = getScalarHIVMFloatShortFragment(valueType);
  if (elem.empty()) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.round." + elem).getValue();
}

template <typename BinaryOp> FailureOr<StringRef> buildBinaryScalarMathCallee(MLIRContext *context, Type valueType);

template <> inline FailureOr<StringRef> buildBinaryScalarMathCallee<pto::PowOp>(MLIRContext *context, Type valueType) {
  std::string elem = getLLVMFloatBuiltinFragment(valueType);
  if (elem != "f32" && elem != "f16" && elem != "v2f16") {
    return failure();
  }
  return StringAttr::get(context, "llvm.pow." + elem).getValue();
}

template <typename VecScalarOp> StringRef getVecScalarMaskedStem() {
  if constexpr (std::is_same_v<VecScalarOp, pto::VmulsOp>) {
    return "vmuls";
  }
  if constexpr (std::is_same_v<VecScalarOp, pto::VaddsOp>) {
    return "vadds";
  }
  if constexpr (std::is_same_v<VecScalarOp, pto::VmaxsOp>) {
    return "vmaxs";
  }
  if constexpr (std::is_same_v<VecScalarOp, pto::VminsOp>) {
    return "vmins";
  }
  if constexpr (std::is_same_v<VecScalarOp, pto::VlreluOp>) {
    return "vlrelu";
  }
  if constexpr (std::is_same_v<VecScalarOp, pto::VshlsOp>) {
    return "vshls";
  }
  if constexpr (std::is_same_v<VecScalarOp, pto::VshrsOp>) {
    return "vshrs";
  }
  return {};
}

template <typename VecScalarOp> constexpr bool usesSignedVecScalarCANN900Callee() {
  return !std::is_same_v<VecScalarOp, pto::VlreluOp>;
}

template <typename ReductionOp> StringRef getReductionUnaryStem() {
  if constexpr (std::is_same_v<ReductionOp, pto::VcaddOp>) {
    return "vcadd";
  }
  if constexpr (std::is_same_v<ReductionOp, pto::VcmaxOp>) {
    return "vcmax";
  }
  if constexpr (std::is_same_v<ReductionOp, pto::VcminOp>) {
    return "vcmin";
  }
  if constexpr (std::is_same_v<ReductionOp, pto::VcgaddOp>) {
    return "vcgadd";
  }
  if constexpr (std::is_same_v<ReductionOp, pto::VcgmaxOp>) {
    return "vcgmax";
  }
  if constexpr (std::is_same_v<ReductionOp, pto::VcgminOp>) {
    return "vcgmin";
  }
  if constexpr (std::is_same_v<ReductionOp, pto::VcpaddOp>) {
    return "vcpadd";
  }
  return {};
}

template <typename HistOp> StringRef getHistogramCallee(MLIRContext *context) {
  if constexpr (std::is_same_v<HistOp, pto::Chistv2Op>) {
    return StringAttr::get(context, "llvm.hivm.chistv2.m").getValue();
  }
  if constexpr (std::is_same_v<HistOp, pto::Dhistv2Op>) {
    return StringAttr::get(context, "llvm.hivm.dhistv2.m").getValue();
  }
  return {};
}

template <typename ExtremaOp> StringRef getExtremaPredicateStem() {
  if constexpr (std::is_same_v<ExtremaOp, pto::VcbmaxOp>) {
    return "vcbmax";
  }
  if constexpr (std::is_same_v<ExtremaOp, pto::VcbminOp>) {
    return "vcbmin";
  }
  return {};
}

template <typename ExtremaOp> FailureOr<StringRef> buildExtremaPredicateCallee(MLIRContext *context, Type resultType) {
  return buildCANN900SignedModeTypedCallee(context, resultType, getExtremaPredicateStem<ExtremaOp>(), "x");
}

template <typename ReductionOp> constexpr bool usesSignedReductionCANN900Callee() {
  return !std::is_same_v<ReductionOp, pto::VcpaddOp>;
}

} // namespace mlir::pto::detail

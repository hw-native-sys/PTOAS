// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Predicate (mask/pack/load/store/interleave/set) callee-name templates
// for the CANN900 LLVM emitter. Split from VPTOCANN900LLVMEmitterTemplates.h
// to keep each header under the 500-line header-size limit.

#pragma once

#include "VPTOCANN900LLVMEmitterInternal.h"

namespace mlir::pto::detail {

template <typename Op> StringRef buildPredicatePairReorderCallee(MLIRContext *context);

template <> inline StringRef buildPredicatePairReorderCallee<pto::PdintlvB8Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pdintlv.b8").getValue();
}

template <> inline StringRef buildPredicatePairReorderCallee<pto::PdintlvB16Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pdintlv.b16").getValue();
}

template <> inline StringRef buildPredicatePairReorderCallee<pto::PdintlvB32Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pdintlv.b32").getValue();
}

template <> inline StringRef buildPredicatePairReorderCallee<pto::PintlvB8Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pintlv.b8").getValue();
}

template <> inline StringRef buildPredicatePairReorderCallee<pto::PintlvB16Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pintlv.b16").getValue();
}

template <> inline StringRef buildPredicatePairReorderCallee<pto::PintlvB32Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pintlv.b32").getValue();
}

template <typename StoreOp> StringRef getPredicateStoreCallee(MLIRContext *context, bool post);

template <> inline StringRef getPredicateStoreCallee<pto::PstiOp>(MLIRContext *context, bool post) {
  return buildPstiCallee(context, post);
}

template <> inline StringRef getPredicateStoreCallee<pto::PstsOp>(MLIRContext *context, bool post) {
  return buildPstsCallee(context, post);
}

template <typename LoadOp> StringRef getPredicateLoadCallee(MLIRContext *context, bool post);

template <> inline StringRef getPredicateLoadCallee<pto::PldiOp>(MLIRContext *context, bool post) {
  return buildPldiCallee(context, post);
}

template <> inline StringRef getPredicateLoadCallee<pto::PldsOp>(MLIRContext *context, bool post) {
  return buildPldsCallee(context, post);
}

template <typename PredicateMaskOp> StringRef getPredicateMaskCallee(MLIRContext *context);

template <> inline StringRef getPredicateMaskCallee<pto::PnotOp>(MLIRContext *context) {
  return buildPnotCallee(context);
}

template <> inline StringRef getPredicateMaskCallee<pto::PselOp>(MLIRContext *context) {
  return buildPselCallee(context);
}

template <> inline StringRef getPredicateMaskCallee<pto::PandOp>(MLIRContext *context) {
  return buildPandCallee(context);
}

template <> inline StringRef getPredicateMaskCallee<pto::PorOp>(MLIRContext *context) {
  return buildPorCallee(context);
}

template <> inline StringRef getPredicateMaskCallee<pto::PxorOp>(MLIRContext *context) {
  return buildPxorCallee(context);
}

template <typename PackOp> StringRef getPredicatePackCallee(MLIRContext *context);

template <> inline StringRef getPredicatePackCallee<pto::PpackOp>(MLIRContext *context) {
  return buildPpackCallee(context);
}

template <> inline StringRef getPredicatePackCallee<pto::PunpackOp>(MLIRContext *context) {
  return buildPunpackCallee(context);
}

template <typename PltOp> StringRef buildPltCallee(MLIRContext *context);

template <> inline StringRef buildPltCallee<pto::PltB8Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.plt.b8.v300").getValue();
}

template <> inline StringRef buildPltCallee<pto::PltB16Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.plt.b16.v300").getValue();
}

template <> inline StringRef buildPltCallee<pto::PltB32Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.plt.b32.v300").getValue();
}

template <typename PltmOp> StringRef buildPltmCallee(MLIRContext *context);

template <> inline StringRef buildPltmCallee<pto::PltmB8Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pltm.b8.v300").getValue();
}

template <> inline StringRef buildPltmCallee<pto::PltmB16Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pltm.b16.v300").getValue();
}

template <> inline StringRef buildPltmCallee<pto::PltmB32Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pltm.b32.v300").getValue();
}

template <typename PsetOp> StringRef buildPsetCallee(MLIRContext *context);

template <> inline StringRef buildPsetCallee<pto::PsetB8Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pset.b8").getValue();
}

template <> inline StringRef buildPsetCallee<pto::PsetB16Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pset.b16").getValue();
}

template <> inline StringRef buildPsetCallee<pto::PsetB32Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pset.b32").getValue();
}

template <typename PgeOp> StringRef buildPgeCallee(MLIRContext *context);

template <> inline StringRef buildPgeCallee<pto::PgeB8Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pge.b8").getValue();
}

template <> inline StringRef buildPgeCallee<pto::PgeB16Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pge.b16").getValue();
}

template <> inline StringRef buildPgeCallee<pto::PgeB32Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pge.b32").getValue();
}

} // namespace mlir::pto::detail

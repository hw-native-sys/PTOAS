// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOMadMxBiasOp.cpp - pto.mad_mx_bias verification -----------------===//
//===----------------------------------------------------------------------===//

#include "VPTOMadInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::mad_detail;

void MadMxBiasOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getLhsMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getRhsMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getBiasMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult MadMxBiasOp::verify() {
  if (failed(verifyMadMxCommon(*this, getLhs().getType(), getRhs().getType(),
                               getDst().getType(), getBias().getType()))) {
    return failure();
  }
  return verifyMadSemanticClauses(*this, getLhs().getType(), getRhs().getType(),
                                  getDst().getType(), getBias().getType(),
                                  std::nullopt, getSatMode(),
                                  (*this)->hasAttr("n_dir"));
}

ParseResult MadMxBiasOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseMadSemanticOpCommon<MadMxBiasOp>(parser, result, /*hasBias=*/true,
                                               /*parseTf32ModeClause=*/false);
}

// NOLINTNEXTLINE(readability-make-member-function-const): ODS-generated printer callback has a non-const signature.
void MadMxBiasOp::print(OpAsmPrinter &p) {
  printMadSemanticOpWithBias(p, *this, /*allowTf32Mode=*/false);
}

// NOLINTNEXTLINE(readability-make-member-function-const): ODS-generated interface callback has a non-const signature.
bool MadMxBiasOp::isMadMxFamily() { return true; }
// NOLINTNEXTLINE(readability-make-member-function-const): ODS-generated interface callback has a non-const signature.
bool MadMxBiasOp::hasBiasOperand() { return true; }
// NOLINTNEXTLINE(readability-make-member-function-const): ODS-generated interface callback has a non-const signature.
bool MadMxBiasOp::readsAccumulator() { return false; }
// NOLINTNEXTLINE(readability-make-member-function-const): ODS-generated interface callback has a non-const signature.
bool MadMxBiasOp::supportsTf32Mode() { return false; }
Value MadMxBiasOp::getBiasOrNull() { return getBias(); }
// NOLINTNEXTLINE(readability-make-member-function-const): ODS-generated interface callback has a non-const signature.
Attribute MadMxBiasOp::getTf32ModeAttr() { return {}; }

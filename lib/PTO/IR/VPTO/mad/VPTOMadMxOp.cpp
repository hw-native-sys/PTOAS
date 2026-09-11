// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOMadMxOp.cpp - pto.mad_mx verification --------------------------===//
//===----------------------------------------------------------------------===//

#include "VPTOMadInternal.h"

using namespace mlir;
using namespace mlir::pto;

void MadMxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getLhsMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getRhsMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult MadMxOp::verify() {
  if (failed(verifyMadMxCommon(*this, getLhs().getType(), getRhs().getType(),
                               getDst().getType()))) {
    return failure();
  }
  return verifyMadSemanticClauses(*this, getLhs().getType(), getRhs().getType(),
                                  getDst().getType(), std::nullopt, std::nullopt,
                                  getSatMode(),
                                  (*this)->hasAttr("n_dir"));
}

ParseResult MadMxOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseMadSemanticOpCommon<MadMxOp>(parser, result, /*hasBias=*/false,
                                           /*parseTf32ModeClause=*/false);
}

void MadMxOp::print(OpAsmPrinter &p) {
  printMadSemanticOpNoBias(p, *this, /*allowTf32Mode=*/false);
}

bool MadMxOp::isMadMxFamily() { return true; }
bool MadMxOp::hasBiasOperand() { return false; }
bool MadMxOp::readsAccumulator() { return false; }
bool MadMxOp::supportsTf32Mode() { return false; }
Value MadMxOp::getBiasOrNull() { return {}; }
Attribute MadMxOp::getTf32ModeAttr() { return {}; }

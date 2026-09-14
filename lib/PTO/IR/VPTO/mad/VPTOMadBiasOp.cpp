// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOMadBiasOp.cpp - pto.mad_bias verification ----------------------===//
//===----------------------------------------------------------------------===//

#include "VPTOMadInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::mad_detail;

void MadBiasOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  collectMadSemanticBiasEffects(*this, effects, /*accumulates=*/false);
}

LogicalResult MadBiasOp::verify() {
  std::optional<pto::Tf32Mode> tf32Mode;
  if (auto tf32ModeAttr =
          (*this)->getAttrOfType<pto::Tf32ModeAttr>("tf32_mode")) {
    tf32Mode = tf32ModeAttr.getValue();
  }
  return verifyMadSemanticClauses(*this, getLhs().getType(), getRhs().getType(),
                                  getDst().getType(), getBias().getType(),
                                  tf32Mode, getSatMode(),
                                  (*this)->hasAttr("n_dir"));
}

ParseResult MadBiasOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseMadSemanticOpCommon<MadBiasOp>(parser, result, /*hasBias=*/true,
                                             /*parseTf32ModeClause=*/true);
}

void MadBiasOp::print(OpAsmPrinter &p) {
  printMadSemanticOpWithBias(p, *this, /*allowTf32Mode=*/true);
}

bool MadBiasOp::isMadMxFamily() const { return false; }
bool MadBiasOp::hasBiasOperand() const { return true; }
bool MadBiasOp::readsAccumulator() const { return false; }
bool MadBiasOp::supportsTf32Mode() const { return true; }
Value MadBiasOp::getBiasOrNull() { return getBias(); }

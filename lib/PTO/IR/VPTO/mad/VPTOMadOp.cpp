// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOMadOp.cpp - pto.mad verification -------------------------------===//
//===----------------------------------------------------------------------===//

#include "VPTOMadInternal.h"

using namespace mlir;
using namespace mlir::pto;

void MadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getLhsMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getRhsMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult MadOp::verify() { return verifyMadSemanticWithTf32(*this); }

ParseResult MadOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseMadSemanticOpCommon<MadOp>(parser, result, /*hasBias=*/false,
                                         /*parseTf32ModeClause=*/true);
}

void MadOp::print(OpAsmPrinter &p) {
  printMadSemanticOpNoBias(p, *this, /*allowTf32Mode=*/true);
}

bool MadOp::isMadMxFamily() { return false; }
bool MadOp::hasBiasOperand() { return false; }
bool MadOp::readsAccumulator() { return false; }
bool MadOp::supportsTf32Mode() { return true; }
Value MadOp::getBiasOrNull() { return {}; }

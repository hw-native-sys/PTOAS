// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOMteGmL1.cpp - pto.MteGmL1 methods ------------------------------===//
//===----------------------------------------------------------------------===//

#include "VPTOMteInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::mte_detail;

void MteGmL1Op::build(OpBuilder &odsBuilder, OperationState &state, Value source,
                      Value destination, Value lenBurst,
                      pto::DmaLoopConfig nburst,
                      llvm::ArrayRef<pto::DmaLoopConfig> loops) {
  buildDmaLoopOp<MteGmL1Op>(odsBuilder, state, source, destination, lenBurst,
                            nburst, loops);
}

ParseResult MteGmL1Op::parse(OpAsmParser &parser, OperationState &result) {
  return parseDmaLoopOp<MteGmL1Op>(parser, result);
}

void MteGmL1Op::print(OpAsmPrinter &p) { printDmaLoopOp(p, *this); }

LogicalResult MteGmL1Op::verify() {
  if (failed(verifyCopyGmToUbufOp(*this, true))) {
    return failure();
  }
  return verifyDmaLoadStoreLoopGroups(
      getOperation(), getLoopCounts(), getLoopSrcStrides(),
      getLoopDstStrides());
}

void MteGmL1Op::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

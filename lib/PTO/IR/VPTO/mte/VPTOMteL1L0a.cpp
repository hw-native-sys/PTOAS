// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOMteL1L0a.cpp - pto.MteL1L0a methods ----------------------------===//
//===----------------------------------------------------------------------===//

#include "VPTOMteInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::mte_detail;

LogicalResult MteL1L0aOp::verify() {
  if (failed(verifyCubeBridgeLoadLikeOp(*this, AddressSpace::LEFT, "LEFT"))) {
    return failure();
  }
  return verifyMteL1L0LoadOperands(
      getOperation(), {getM(), getK(), getStartRow(), getStartCol()},
      {"m", "k", "start_row", "start_col"},
      {getMStart(), getKStart(), getMStep(), getKStep(), getSrcStride(),
       getDstStride()});
}

ParseResult MteL1L0aOp::parse(OpAsmParser &parser, OperationState &result) {
  static constexpr StringRef kShapeNames[] = {
      "m", "k", "start_row", "start_col"};
  static constexpr StringRef kFullNames[] = {
      "m_start", "k_start", "m_step", "k_step", "src_stride", "dst_stride"};
  return parseMteL1L0OptionalOperandsOp<MteL1L0aOp>(
      parser, result, kShapeNames, kFullNames);
}

void MteL1L0aOp::print(OpAsmPrinter &p) {
  static constexpr StringRef kShapeNames[] = {
      "m", "k", "start_row", "start_col"};
  static constexpr StringRef kFullNames[] = {
      "m_start", "k_start", "m_step", "k_step", "src_stride", "dst_stride"};
  printMteL1L0OptionalOperandsOp(
      p, getOperation(), getSource(), getDestination(),
      {getM(), getK(), getStartRow(), getStartCol()}, kShapeNames,
      {getMStart(), getKStart(), getMStep(), getKStep(), getSrcStride(),
       getDstStride()},
      kFullNames);
}

void MteL1L0aOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

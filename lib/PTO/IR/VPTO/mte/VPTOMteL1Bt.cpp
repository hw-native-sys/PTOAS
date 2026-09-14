// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOMteL1Bt.cpp - pto.MteL1Bt methods ------------------------------===//
//===----------------------------------------------------------------------===//

#include "VPTOMteInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::mte_detail;

void MteL1BtOp::build(OpBuilder &odsBuilder, OperationState &state, Value source,
                      Value destination, Value lenBurst,
                      pto::DmaLoopConfig nburst) {
  (void)odsBuilder;
  addDmaTripleOperandList(state, source, destination, lenBurst, nburst);
}

ParseResult MteL1BtOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseDmaTripleOp(parser, result);
}

void MteL1BtOp::print(OpAsmPrinter &p) {
  printDmaTripleOpFields(p, *this);
}

LogicalResult MteL1BtOp::verify() {
  auto getBufferElementType = [](Type type) -> Type {
    if (auto ptrType = dyn_cast<pto::PtrType>(type)) {
      return ptrType.getElementType();
    }
    if (auto memrefType = dyn_cast<BaseMemRefType>(type)) {
      return memrefType.getElementType();
    }
    return {};
  };

  if (!isBufferLike(getSource().getType()) ||
      !isBufferLike(getDestination().getType())) {
    return emitOpError("requires buffer-like source and destination");
  }
  if (getBufferAddressSpace(getSource().getType()) != pto::AddressSpace::MAT) {
    return emitOpError("requires MAT source");
  }
  if (getBufferAddressSpace(getDestination().getType()) != pto::AddressSpace::BIAS) {
    return emitOpError("requires BIAS destination");
  }

  Type srcElem = getBufferElementType(getSource().getType());
  Type dstElem = getBufferElementType(getDestination().getType());
  const bool isF32 = srcElem.isF32() && dstElem.isF32();
  const bool isI32 = isa<IntegerType>(srcElem) && isa<IntegerType>(dstElem) &&
                     cast<IntegerType>(srcElem).getWidth() == 32 &&
                     cast<IntegerType>(dstElem).getWidth() == 32;
  const bool isF16ToF32 = srcElem.isF16() && dstElem.isF32();
  const bool isBF16ToF32 = srcElem.isBF16() && dstElem.isF32();
  if (!isF32 && !isI32 && !isF16ToF32 && !isBF16ToF32) {
    return emitOpError(
        "expects one of f32->f32, i32->i32, f16->f32, or bf16->f32");
  }
  return success();
}

void MteL1BtOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

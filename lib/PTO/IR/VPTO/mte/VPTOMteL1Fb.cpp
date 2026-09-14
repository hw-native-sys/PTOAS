// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOMteL1Fb.cpp - pto.MteL1Fb methods ------------------------------===//
//===----------------------------------------------------------------------===//

#include "VPTOMteInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::mte_detail;

void MteL1FbOp::build(OpBuilder &odsBuilder, OperationState &state, Value source,
                      Value destination, Value lenBurst,
                      pto::DmaLoopConfig nburst) {
  (void)odsBuilder;
  addDmaTripleOperandList(state, source, destination, lenBurst, nburst);
}

ParseResult MteL1FbOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseDmaTripleOp(parser, result);
}

void MteL1FbOp::print(OpAsmPrinter &p) {
  printDmaTripleOpFields(p, *this);
}

LogicalResult MteL1FbOp::verify() {
  if (!isBufferLike(getSource().getType()) || !isBufferLike(getDestination().getType())) {
    return emitOpError(
        "requires typed !pto.ptr or memref source and destination");
  }

  auto getAddressSpace = [](Type type) -> std::optional<pto::AddressSpace> {
    if (auto ptrType = dyn_cast<pto::PtrType>(type)) {
      return ptrType.getMemorySpace().getAddressSpace();
    }
    if (auto memrefType = dyn_cast<BaseMemRefType>(type)) {
      Attribute memorySpace = memrefType.getMemorySpace();
      if (auto addrSpace = dyn_cast_or_null<pto::AddressSpaceAttr>(memorySpace)) {
        return addrSpace.getAddressSpace();
      }
      if (auto intAttr = dyn_cast_or_null<IntegerAttr>(memorySpace)) {
        return static_cast<pto::AddressSpace>(intAttr.getInt());
      }
    }
    return std::nullopt;
  };

  std::optional<pto::AddressSpace> sourceAS = getAddressSpace(getSource().getType());
  std::optional<pto::AddressSpace> destinationAS =
      getAddressSpace(getDestination().getType());
  if (!sourceAS || !destinationAS) {
    return emitOpError("requires source and destination with PTO address spaces");
  }
  if (*sourceAS != pto::AddressSpace::MAT) {
    return emitOpError("requires source in mat address space");
  }
  if (*destinationAS != pto::AddressSpace::SCALING) {
    return emitOpError("requires destination in scaling address space");
  }
  return success();
}

void MteL1FbOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

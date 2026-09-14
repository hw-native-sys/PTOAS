// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOMteUbGm.cpp - pto.MteUbGm methods ------------------------------===//
//===----------------------------------------------------------------------===//

#include "VPTOMteInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::mte_detail;

void MteUbGmOp::build(OpBuilder &odsBuilder, OperationState &state, Value source,
                       Value destination, Value lenBurst,
                       pto::DmaLoopConfig nburst, Value l2CacheCtl,
                       llvm::ArrayRef<pto::DmaLoopConfig> loops) {
  state.addOperands({source, destination, lenBurst, nburst.count,
                     nburst.srcStride, nburst.dstStride});
  if (l2CacheCtl) {
    state.addOperands(l2CacheCtl);
  }
  addDmaLoopConfigOperands(state, loops);

  state.addAttribute(
      getOperandSegmentSizeAttr(),
      odsBuilder.getDenseI32ArrayAttr(
          {1, 1, 1, 1, 1, 1, l2CacheCtl ? 1 : 0,
           static_cast<int32_t>(loops.size()),
           static_cast<int32_t>(loops.size()),
           static_cast<int32_t>(loops.size())}));
}

void MteUbGmOp::build(OpBuilder &odsBuilder, OperationState &state, Value source,
                       Value destination, Value lenBurst,
                       pto::DmaLoopConfig nburst, Value l2CacheCtl,
                       std::optional<pto::DmaLoopConfig> loop1,
                       std::optional<pto::DmaLoopConfig> loop2) {
  SmallVector<pto::DmaLoopConfig> loops;
  if (loop1) {
    loops.push_back(*loop1);
  }
  if (loop2) {
    loops.push_back(*loop2);
  }
  build(odsBuilder, state, source, destination, lenBurst, nburst, l2CacheCtl,
        loops);
}

ParseResult MteUbGmOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand source, destination, lenBurst, l2CacheCtl;
  bool hasL2CacheCtl = false;
  SmallVector<OpAsmParser::UnresolvedOperand> nburstOperands,
      loopCountOperands, loopSrcStrideOperands, loopDstStrideOperands;
  if (failed(parseMteUbGmBasicOperands(parser, source, destination, lenBurst,
                                       nburstOperands)) ||
      failed(parseMteUbGmL2CacheCtlOperand(parser, l2CacheCtl,
                                            hasL2CacheCtl)) ||
      failed(parseDmaLoopOperandGroups(parser, loopCountOperands,
                                       loopSrcStrideOperands,
                                       loopDstStrideOperands))) {
    return failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon()) {
    return failure();
  }
  Type sourceType, destinationType, lenBurstType, l2CacheCtlType;
  SmallVector<Type> nburstTypes, loopCountTypes, loopSrcStrideTypes,
      loopDstStrideTypes;
  if (failed(parseMteUbGmBasicTypes(parser, sourceType, destinationType,
                                    lenBurstType, nburstTypes))) {
    return failure();
  }
  if (hasL2CacheCtl) {
    if (parser.parseComma() || parser.parseType(l2CacheCtlType)) {
      return failure();
    }
  }
  if (failed(parseDmaLoopTypeGroups(parser, loopCountTypes,
                                    loopSrcStrideTypes, loopDstStrideTypes))) {
    return failure();
  }
  if (failed(verifyDmaLoopGroupConsistency(
          parser, loopCountOperands.size(), loopSrcStrideOperands.size(),
          loopDstStrideOperands.size(), loopCountTypes.size(),
          loopSrcStrideTypes.size(), loopDstStrideTypes.size()))) {
    return failure();
  }
  setMteUbGmSegmentSizes(result, hasL2CacheCtl, loopCountOperands.size());
  if (failed(resolveDmaTripleOperands(
          parser, result, hasL2CacheCtl, l2CacheCtl, l2CacheCtlType, source,
          sourceType, destination, destinationType, lenBurst, lenBurstType,
          nburstOperands, nburstTypes, loopCountOperands, loopCountTypes,
          loopSrcStrideOperands, loopSrcStrideTypes, loopDstStrideOperands,
          loopDstStrideTypes))) {
    return failure();
  }
  return success();
}

void MteUbGmOp::print(OpAsmPrinter &p) {
  p << " " << getSource() << ", " << getDestination() << ", "
          << getLenBurst();
  printDmaTripleGroup(p, "nburst", getNBurst(), getNburstSrcStride(),
                      getNburstDstStride());
  if (Value l2CacheCtl = getL2CacheCtl()) {
    p << " l2_cache_ctl(" << l2CacheCtl << ")";
  }
  for (auto [count, srcStride, dstStride] :
       llvm::zip(getLoopCounts(), getLoopSrcStrides(), getLoopDstStrides())) {
    printDmaTripleGroup(p, "loop", count, srcStride, dstStride);
  }
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << getSource().getType() << ", " << getDestination().getType()
          << ", " << getLenBurst().getType() << ", " << getNBurst().getType()
          << ", " << getNburstSrcStride().getType()
          << ", "
          << getNburstDstStride().getType();
  if (Value l2CacheCtl = getL2CacheCtl()) {
    p << ", " << l2CacheCtl.getType();
  }
  for (auto [count, srcStride, dstStride] :
       llvm::zip(getLoopCounts(), getLoopSrcStrides(), getLoopDstStrides())) {
    printDmaTripleTypes(p, "loop", count.getType(), srcStride.getType(),
                        dstStride.getType());
  }
}

void MteUbGmOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult MteUbGmOp::verify() {
  if (!isBufferLike(getSource().getType()) ||
      !isBufferLike(getDestination().getType())) {
    return emitOpError(
        "requires typed !pto.ptr or memref source and destination");
  }
  if (classifyMemoryRole(getSource().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getDestination().getType()) != MemoryRole::GM) {
    return emitOpError("requires UB source and GM destination");
  }
  int64_t sourceElemBytes = getBufferElementByteSize(getSource().getType());
  int64_t destinationElemBytes =
      getBufferElementByteSize(getDestination().getType());
  if (sourceElemBytes <= 0 || destinationElemBytes <= 0) {
    return emitOpError(
        "requires copy source and destination element types with known byte width");
  }
  if (sourceElemBytes != destinationElemBytes) {
    return emitOpError(
        "requires source and destination element byte widths to match");
  }
  if (Value l2CacheCtlValue = getL2CacheCtl()) {
    APInt l2CacheCtl;
    if (matchPattern(l2CacheCtlValue, m_ConstantInt(&l2CacheCtl)) &&
        (l2CacheCtl.isNegative() || l2CacheCtl.ugt(mlir::pto::kValue15))) {
      return emitOpError(
          "requires constant l2_cache_ctl to fit in range [0, 15]");
    }
  }
  return verifyDmaLoadStoreLoopGroups(
      getOperation(), getLoopCounts(), getLoopSrcStrides(),
      getLoopDstStrides());
}

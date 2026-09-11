// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOMteGmUb.cpp - pto.MteGmUb methods ------------------------------===//
//===----------------------------------------------------------------------===//

#include "VPTOMteInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::mte_detail;

void MteGmUbOp::build(OpBuilder &odsBuilder, OperationState &state, Value source,
                      Value destination, Value l2CacheCtl, Value lenBurst,
                      pto::DmaLoopConfig nburst,
                      llvm::ArrayRef<pto::DmaLoopConfig> loops,
                      std::optional<pto::DmaPadConfig> pad) {
  state.addOperands({source, destination, l2CacheCtl, lenBurst, nburst.count,
                     nburst.srcStride, nburst.dstStride});
  addDmaLoopConfigOperands(state, loops);
  bool hasPadCounts = pad && pad->leftCount && pad->rightCount;
  if (pad && static_cast<bool>(pad->leftCount) !=
                  static_cast<bool>(pad->rightCount)) {
    llvm::report_fatal_error(
        "mte_gm_ub pad config must provide both left and right counts, or omit both");
  }
  if (pad) {
    state.addOperands(pad->value);
    if (hasPadCounts) {
      state.addOperands({pad->leftCount, pad->rightCount});
    }
  }

  state.addAttribute(
      getOperandSegmentSizeAttr(),
      odsBuilder.getDenseI32ArrayAttr(
          {1, 1, 1, 1, 1, 1, 1,
           static_cast<int32_t>(loops.size()),
           static_cast<int32_t>(loops.size()),
           static_cast<int32_t>(loops.size()),
           pad ? 1 : 0, hasPadCounts ? 1 : 0, hasPadCounts ? 1 : 0}));
}

void MteGmUbOp::build(OpBuilder &odsBuilder, OperationState &state, Value source,
                      Value destination, Value l2CacheCtl, Value lenBurst,
                      pto::DmaLoopConfig nburst,
                      std::optional<pto::DmaLoopConfig> loop1,
                      std::optional<pto::DmaLoopConfig> loop2,
                      std::optional<pto::DmaPadConfig> pad) {
  SmallVector<pto::DmaLoopConfig> loops;
  if (loop1) {
    loops.push_back(*loop1);
  }
  if (loop2) {
    loops.push_back(*loop2);
  }
  build(odsBuilder, state, source, destination, l2CacheCtl, lenBurst, nburst,
        loops, pad);
}

ParseResult MteGmUbOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand source, destination, l2CacheCtl, lenBurst;
  SmallVector<OpAsmParser::UnresolvedOperand> nburstOperands;
  SmallVector<OpAsmParser::UnresolvedOperand> loopCountOperands;
  SmallVector<OpAsmParser::UnresolvedOperand> loopSrcStrideOperands;
  SmallVector<OpAsmParser::UnresolvedOperand> loopDstStrideOperands;
  SmallVector<OpAsmParser::UnresolvedOperand> padOperands;
  if (failed(parseMteGmUbBasicOperands(parser, source, destination,
                                       l2CacheCtl, lenBurst,
                                       nburstOperands)) ||
      failed(parseDmaLoopOperandGroups(parser, loopCountOperands,
                                       loopSrcStrideOperands,
                                       loopDstStrideOperands)) ||
      failed(parseDmaPadOperandGroup(parser, padOperands))) {
    return failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon()) {
    return failure();
  }
  Type sourceType, destinationType, l2CacheCtlType, lenBurstType;
  SmallVector<Type> nburstTypes, loopCountTypes, loopSrcStrideTypes,
      loopDstStrideTypes, padTypes;
  if (failed(parseMteGmUbBasicTypes(parser, sourceType, destinationType,
                                    l2CacheCtlType, lenBurstType,
                                    nburstTypes)) ||
      failed(parseDmaLoopAndPadTypeGroups(parser, loopCountTypes,
                                          loopSrcStrideTypes,
                                          loopDstStrideTypes, padTypes))) {
    return failure();
  }
  if (failed(verifyDmaLoopGroupConsistency(
          parser, loopCountOperands.size(), loopSrcStrideOperands.size(),
          loopDstStrideOperands.size(), loopCountTypes.size(),
          loopSrcStrideTypes.size(), loopDstStrideTypes.size()))) {
    return failure();
  }
  setMteGmUbSegmentSizes(result,
                         static_cast<int32_t>(loopCountOperands.size()),
                         padOperands.size());
  if (failed(resolveMteGmUbOperands(
          parser, result, source, sourceType, destination, destinationType,
          l2CacheCtl, l2CacheCtlType, lenBurst, lenBurstType, nburstOperands,
          nburstTypes, loopCountOperands, loopCountTypes,
          loopSrcStrideOperands, loopSrcStrideTypes, loopDstStrideOperands,
          loopDstStrideTypes, padOperands, padTypes))) {
    return failure();
  }
  return success();
}

void MteGmUbOp::print(OpAsmPrinter &p) {
  p << " " << getSource() << ", " << getDestination() << ", "
          << getL2CacheCtl() << ", " << getLenBurst();
  printDmaTripleGroup(p, "nburst", getNBurst(), getNburstSrcStride(),
                      getNburstDstStride());
  for (auto [count, srcStride, dstStride] :
       llvm::zip(getLoopCounts(), getLoopSrcStrides(), getLoopDstStrides())) {
    printDmaTripleGroup(p, "loop", count, srcStride, dstStride);
  }
  if (getPadValue()) {
    printDmaPadGroup(p, getPadValue(), getLeftPaddingCount(),
                     getRightPaddingCount());
  }
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << getSource().getType() << ", " << getDestination().getType()
          << ", " << getL2CacheCtl().getType() << ", " << getLenBurst().getType()
          << ", " << getNBurst().getType() << ", " << getNburstSrcStride().getType()
          << ", "
          << getNburstDstStride().getType();
  for (auto [count, srcStride, dstStride] :
       llvm::zip(getLoopCounts(), getLoopSrcStrides(), getLoopDstStrides())) {
    printDmaTripleTypes(p, "loop", count.getType(), srcStride.getType(),
                        dstStride.getType());
  }
  if (getPadValue()) {
    printDmaPadTypes(p, getPadValue().getType(),
                     getLeftPaddingCount() ? getLeftPaddingCount().getType() : Type{},
                     getRightPaddingCount() ? getRightPaddingCount().getType() : Type{});
  }
}

void MteGmUbOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult MteGmUbOp::verify() {
  if (failed(verifyCopyGmToUbufOp(*this, true))) {
    return failure();
  }
  if (failed(verifyDmaLoadStoreLoopGroups(
          getOperation(), getLoopCounts(), getLoopSrcStrides(),
          getLoopDstStrides()))) {
    return failure();
  }
  if (!getPadValue() && (getLeftPaddingCount() || getRightPaddingCount())) {
    return emitOpError() << "requires pad group to provide a pad value";
  }
  if (getPadValue() && static_cast<bool>(getLeftPaddingCount()) !=
                           static_cast<bool>(getRightPaddingCount())) {
    return emitOpError()
           << "requires pad group to provide both left and right counts, or omit both";
  }
  if (Value padValue = getPadValue()) {
    Type valueType = padValue.getType();
    if (!isSupportedMovPadScalarType(valueType)) {
      return emitOpError()
             << "expects pad value to be i8/i16/i32 or f16/bf16/f32 scalar, but got "
             << valueType;
    }
  }
  return success();
}

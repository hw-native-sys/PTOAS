// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOMte.cpp - VPTO MTE ops -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VPTOInternal.h"

using namespace mlir;
using namespace mlir::pto;

// Batch12: DmaLoopConfig expansion shared
static void addDmaLoopConfigOperands(OperationState &state,
                                     llvm::ArrayRef<pto::DmaLoopConfig> loops) {
  for (const pto::DmaLoopConfig &loop : loops) {
    state.addOperands(loop.count);
  }
  for (const pto::DmaLoopConfig &loop : loops) {
    state.addOperands(loop.srcStride);
  }
  for (const pto::DmaLoopConfig &loop : loops) {
    state.addOperands(loop.dstStride);
  }
}

void MteGmUbOp::build(OpBuilder &builder, OperationState &state, Value source,
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
      builder.getDenseI32ArrayAttr(
          {1, 1, 1, 1, 1, 1, 1,
           static_cast<int32_t>(loops.size()),
           static_cast<int32_t>(loops.size()),
           static_cast<int32_t>(loops.size()),
           pad ? 1 : 0, hasPadCounts ? 1 : 0, hasPadCounts ? 1 : 0}));
}

void MteGmUbOp::build(OpBuilder &builder, OperationState &state, Value source,
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
  build(builder, state, source, destination, l2CacheCtl, lenBurst, nburst,
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

void MteGmUbOp::print(OpAsmPrinter &printer) {
  printer << " " << getSource() << ", " << getDestination() << ", "
          << getL2CacheCtl() << ", " << getLenBurst();
  printDmaTripleGroup(printer, "nburst", getNBurst(), getNburstSrcStride(),
                      getNburstDstStride());
  for (auto [count, srcStride, dstStride] :
       llvm::zip(getLoopCounts(), getLoopSrcStrides(), getLoopDstStrides())) {
    printDmaTripleGroup(printer, "loop", count, srcStride, dstStride);
  }
  if (getPadValue()) {
    printDmaPadGroup(printer, getPadValue(), getLeftPaddingCount(),
                     getRightPaddingCount());
  }
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : " << getSource().getType() << ", " << getDestination().getType()
          << ", " << getL2CacheCtl().getType() << ", " << getLenBurst().getType()
          << ", " << getNBurst().getType() << ", " << getNburstSrcStride().getType()
          << ", "
          << getNburstDstStride().getType();
  for (auto [count, srcStride, dstStride] :
       llvm::zip(getLoopCounts(), getLoopSrcStrides(), getLoopDstStrides())) {
    printDmaTripleTypes(printer, "loop", count.getType(), srcStride.getType(),
                        dstStride.getType());
  }
  if (getPadValue()) {
    printDmaPadTypes(printer, getPadValue().getType(),
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

void MteUbUbOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult MteUbUbOp::verify() {
  if (!isBufferLike(getSource().getType()) || !isBufferLike(getDestination().getType())) {
    return emitOpError("requires pointer-like source and destination");
  }
  if (classifyMemoryRole(getSource().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getDestination().getType()) != MemoryRole::UB) {
    return emitOpError("requires UB-backed source and destination");
  }
  return success();
}

void MteUbL1Op::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult MteUbL1Op::verify() {
  if (!isBufferLike(getSource().getType()) || !isBufferLike(getDestination().getType())) {
    return emitOpError("requires pointer-like source and destination");
  }
  if (classifyMemoryRole(getSource().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getDestination().getType()) != MemoryRole::Other) {
    return emitOpError("requires UB-backed source and CBUF-backed destination");
  }
  return success();
}

void MteUbGmOp::build(OpBuilder &builder, OperationState &state, Value source,
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
      builder.getDenseI32ArrayAttr(
          {1, 1, 1, 1, 1, 1, l2CacheCtl ? 1 : 0,
           static_cast<int32_t>(loops.size()),
           static_cast<int32_t>(loops.size()),
           static_cast<int32_t>(loops.size())}));
}

void MteUbGmOp::build(OpBuilder &builder, OperationState &state, Value source,
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
  build(builder, state, source, destination, lenBurst, nburst, l2CacheCtl,
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

void MteUbGmOp::print(OpAsmPrinter &printer) {
  printer << " " << getSource() << ", " << getDestination() << ", "
          << getLenBurst();
  printDmaTripleGroup(printer, "nburst", getNBurst(), getNburstSrcStride(),
                      getNburstDstStride());
  if (Value l2CacheCtl = getL2CacheCtl()) {
    printer << " l2_cache_ctl(" << l2CacheCtl << ")";
  }
  for (auto [count, srcStride, dstStride] :
       llvm::zip(getLoopCounts(), getLoopSrcStrides(), getLoopDstStrides())) {
    printDmaTripleGroup(printer, "loop", count, srcStride, dstStride);
  }
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : " << getSource().getType() << ", " << getDestination().getType()
          << ", " << getLenBurst().getType() << ", " << getNBurst().getType()
          << ", " << getNburstSrcStride().getType()
          << ", "
          << getNburstDstStride().getType();
  if (Value l2CacheCtl = getL2CacheCtl()) {
    printer << ", " << l2CacheCtl.getType();
  }
  for (auto [count, srcStride, dstStride] :
       llvm::zip(getLoopCounts(), getLoopSrcStrides(), getLoopDstStrides())) {
    printDmaTripleTypes(printer, "loop", count.getType(), srcStride.getType(),
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

// Batch6: GmL1/L1Ub 双胞胎 op 共用 build/parse/print
template <typename OpTy>
static void buildDmaLoopOp(OpBuilder &builder, OperationState &state,
                           Value source, Value destination, Value lenBurst,
                           pto::DmaLoopConfig nburst,
                           llvm::ArrayRef<pto::DmaLoopConfig> loops) {
  state.addOperands(
      {source, destination, lenBurst, nburst.count, nburst.srcStride,
       nburst.dstStride});
  addDmaLoopConfigOperands(state, loops);
  state.addAttribute(
      OpTy::getOperandSegmentSizeAttr(),
      builder.getDenseI32ArrayAttr(
          {1, 1, 1, 1, 1, 1,
           static_cast<int32_t>(loops.size()),
           static_cast<int32_t>(loops.size()),
           static_cast<int32_t>(loops.size())}));
}

void MteGmL1Op::build(OpBuilder &builder, OperationState &state, Value source,
                      Value destination, Value lenBurst,
                      pto::DmaLoopConfig nburst,
                      llvm::ArrayRef<pto::DmaLoopConfig> loops) {
  buildDmaLoopOp<MteGmL1Op>(builder, state, source, destination, lenBurst,
                            nburst, loops);
}

void MteL1UbOp::build(OpBuilder &builder, OperationState &state, Value source,
                      Value destination, Value lenBurst,
                      pto::DmaLoopConfig nburst,
                      llvm::ArrayRef<pto::DmaLoopConfig> loops) {
  buildDmaLoopOp<MteL1UbOp>(builder, state, source, destination, lenBurst,
                            nburst, loops);
}


void MteGmL1FracOp::build(OpBuilder &builder, OperationState &state,
                           Value source, Value destination,
                           pto::CubeLoadFracMode mode,
                           pto::CubeLoadFracShapeConfig shape,
                           pto::CubeLoadFracSrcLayoutConfig srcLayout,
                           pto::CubeLoadFracDstGroupConfig dstGroup,
                           pto::CubeLoadFracCtrlConfig ctrl) {
  state.addOperands({source, destination, shape.nValue, shape.dValue,
                     srcLayout.srcInnerStride});
  state.addOperands({dstGroup.groupCount, dstGroup.dstLoop2Stride,
                     dstGroup.dstLoop3Stride, dstGroup.dstLoop4Stride,
                     ctrl.l2CacheCtrl, ctrl.smallc0En});
  bool hasSrcOuterStride = srcLayout.srcOuterStride.has_value();
  if (hasSrcOuterStride) {
    state.addOperands(*srcLayout.srcOuterStride);
  }

  state.addAttribute(getModeAttrName(state.name),
                     CubeLoadFracModeAttr::get(builder.getContext(), mode));
}

template <typename OpTy>
static ParseResult parseDmaLoopOp(OpAsmParser &parser,
                                  OperationState &result) {
  OpAsmParser::UnresolvedOperand source, destination, lenBurst;
  SmallVector<OpAsmParser::UnresolvedOperand> nburstOperands;
  SmallVector<OpAsmParser::UnresolvedOperand> loopCountOperands,
      loopSrcStrideOperands, loopDstStrideOperands;
  if (parseRequiredOperandWithComma(parser, source) ||
      parseRequiredOperandWithComma(parser, destination) ||
      parser.parseOperand(lenBurst) ||
      parseDmaTripleGroup(parser, "nburst", nburstOperands) ||
      parseDmaLoopOperandGroups(parser, loopCountOperands, loopSrcStrideOperands,
                                loopDstStrideOperands)) {
    return failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon()) {
    return failure();
  }
  Type sourceType, destinationType, lenBurstType;
  SmallVector<Type> nburstTypes, loopCountTypes, loopSrcStrideTypes,
      loopDstStrideTypes;
  if (parser.parseType(sourceType) || parser.parseComma() ||
      parser.parseType(destinationType) || parser.parseComma() ||
      parser.parseType(lenBurstType) || parser.parseComma() ||
      parseDmaTripleTypes(parser, nburstTypes) ||
      parseDmaLoopTypeGroups(parser, loopCountTypes, loopSrcStrideTypes,
                             loopDstStrideTypes)) {
    return failure();
  }
  int32_t loopGroupCount = static_cast<int32_t>(loopCountOperands.size());
  if (failed(verifyDmaLoopGroupConsistency(
          parser, loopCountOperands.size(), loopSrcStrideOperands.size(),
          loopDstStrideOperands.size(), loopCountTypes.size(),
          loopSrcStrideTypes.size(), loopDstStrideTypes.size()))) {
    return failure();
  }
  auto &segments = result.getOrAddProperties<typename OpTy::Properties>().operandSegmentSizes;
  llvm::copy(ArrayRef<int32_t>{1, 1, 1, 1, 1, 1,
                               loopGroupCount, loopGroupCount, loopGroupCount},
             segments.begin());
  if (failed(resolveDmaTripleOperands(
          parser, result, /*hasL2CacheCtl=*/false, /*l2CacheCtl=*/{},
          /*l2CacheCtlType=*/{}, source, sourceType, destination,
          destinationType, lenBurst, lenBurstType, nburstOperands,
          nburstTypes, loopCountOperands, loopCountTypes,
          loopSrcStrideOperands, loopSrcStrideTypes, loopDstStrideOperands,
          loopDstStrideTypes))) {
    return failure();
  }
  return success();
}

ParseResult MteGmL1Op::parse(OpAsmParser &parser, OperationState &result) {
  return parseDmaLoopOp<MteGmL1Op>(parser, result);
}

ParseResult MteL1UbOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseDmaLoopOp<MteL1UbOp>(parser, result);
}


ParseResult MteGmL1FracOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand source, destination;
  StringRef modeKeyword;
  SmallVector<OpAsmParser::UnresolvedOperand> shapeOperands;
  SmallVector<OpAsmParser::UnresolvedOperand> srcLayoutOperands;
  SmallVector<OpAsmParser::UnresolvedOperand> dstGroupOperands;
  SmallVector<OpAsmParser::UnresolvedOperand> ctrlOperands;
  if (failed(parseMteGmL1FracBasicOperands(parser, source, destination,
                                           modeKeyword, shapeOperands,
                                           srcLayoutOperands,
                                           dstGroupOperands, ctrlOperands))) {
    return failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon()) {
    return failure();
  }
  Type sourceType, destinationType;
  SmallVector<Type> shapeTypes, srcLayoutTypes, dstGroupTypes, ctrlTypes;
  if (failed(parseMteGmL1FracBasicTypes(parser, sourceType, destinationType,
                                        modeKeyword, shapeTypes,
                                        srcLayoutTypes, dstGroupTypes,
                                        ctrlTypes))) {
    return failure();
  }
  auto modeOr = parseCubeLoadFracModeKeyword(modeKeyword);
  if (failed(modeOr)) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected one of 'nd2nz' or 'dn2nz'");
  }
  if (failed(validateMteGmL1FracOperands(
          parser, shapeOperands.size(), shapeTypes.size(),
          srcLayoutOperands.size(), srcLayoutTypes.size(),
          dstGroupOperands.size(), dstGroupTypes.size(),
          ctrlOperands.size(), ctrlTypes.size()))) {
    return failure();
  }
  result.addAttribute(getModeAttrName(result.name),
                      CubeLoadFracModeAttr::get(parser.getContext(), *modeOr));
  if (failed(resolveMteGmL1FracOperands(parser, result, source, sourceType,
                                        destination, destinationType,
                                        shapeOperands, shapeTypes,
                                        srcLayoutOperands, srcLayoutTypes,
                                        dstGroupOperands, dstGroupTypes,
                                        ctrlOperands, ctrlTypes))) {
    return failure();
  }
  return success();
}

template <typename OpTy>
static void printDmaLoopOp(OpAsmPrinter &printer, OpTy op) {
  printer << " " << op.getSource() << ", " << op.getDestination() << ", "
          << op.getLenBurst();
  printDmaTripleGroup(printer, "nburst", op.getNBurst(),
                      op.getNburstSrcStride(), op.getNburstDstStride());
  for (auto [count, srcStride, dstStride] :
       llvm::zip(op.getLoopCounts(), op.getLoopSrcStrides(),
                 op.getLoopDstStrides())) {
    printDmaTripleGroup(printer, "loop", count, srcStride, dstStride);
  }
  printer.printOptionalAttrDict(op->getAttrs());
  printer << " : " << op.getSource().getType() << ", "
          << op.getDestination().getType() << ", " << op.getLenBurst().getType()
          << ", " << op.getNBurst().getType() << ", "
          << op.getNburstSrcStride().getType() << ", "
          << op.getNburstDstStride().getType();
  for (auto [count, srcStride, dstStride] :
       llvm::zip(op.getLoopCounts(), op.getLoopSrcStrides(),
                 op.getLoopDstStrides())) {
    printDmaTripleTypes(printer, "loop", count.getType(), srcStride.getType(),
                        dstStride.getType());
  }
}

void MteGmL1Op::print(OpAsmPrinter &printer) { printDmaLoopOp(printer, *this); }

void MteL1UbOp::print(OpAsmPrinter &printer) { printDmaLoopOp(printer, *this); }


// Batch6: Bt/Fb 双胞胎 op 共用 build/parse/print
static void addDmaTripleOperandList(OperationState &state, Value source,
                                    Value destination, Value lenBurst,
                                    pto::DmaLoopConfig nburst) {
  state.addOperands({source, destination, lenBurst, nburst.count,
                     nburst.srcStride, nburst.dstStride});
}

static ParseResult parseDmaTripleOp(OpAsmParser &parser,
                                    OperationState &result) {
  OpAsmParser::UnresolvedOperand source, destination, lenBurst;
  SmallVector<OpAsmParser::UnresolvedOperand> nburstOperands;
  if (parseRequiredOperandWithComma(parser, source) ||
      parseRequiredOperandWithComma(parser, destination) ||
      parser.parseOperand(lenBurst) ||
      parseDmaTripleGroup(parser, "nburst", nburstOperands) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon()) {
    return failure();
  }
  Type sourceType, destinationType, lenBurstType;
  SmallVector<Type> nburstTypes;
  if (parser.parseType(sourceType) || parser.parseComma() ||
      parser.parseType(destinationType) || parser.parseComma() ||
      parser.parseType(lenBurstType) || parser.parseComma() ||
      parseDmaTripleTypes(parser, nburstTypes)) {
    return failure();
  }
  if (parser.resolveOperand(source, sourceType, result.operands) ||
      parser.resolveOperand(destination, destinationType, result.operands) ||
      parser.resolveOperand(lenBurst, lenBurstType, result.operands) ||
      parser.resolveOperands(nburstOperands, nburstTypes,
                             parser.getCurrentLocation(), result.operands)) {
    return failure();
  }
  return success();
}

template <typename OpTy>
static void printDmaTripleOpFields(OpAsmPrinter &printer, OpTy op) {
  printer << " " << op.getSource() << ", " << op.getDestination() << ", "
          << op.getLenBurst();
  printDmaTripleGroup(printer, "nburst", op.getNBurst(), op.getNburstSrcGap(),
                      op.getNburstDstGap());
  printer.printOptionalAttrDict(op->getAttrs());
  printer << " : " << op.getSource().getType() << ", "
          << op.getDestination().getType() << ", " << op.getLenBurst().getType()
          << ", " << op.getNBurst().getType() << ", "
          << op.getNburstSrcGap().getType() << ", "
          << op.getNburstDstGap().getType();
}

void MteL1BtOp::build(OpBuilder &builder, OperationState &state, Value source,
                      Value destination, Value lenBurst,
                      pto::DmaLoopConfig nburst) {
  addDmaTripleOperandList(state, source, destination, lenBurst, nburst);
}

void MteL1FbOp::build(OpBuilder &builder, OperationState &state, Value source,
                      Value destination, Value lenBurst,
                      pto::DmaLoopConfig nburst) {
  addDmaTripleOperandList(state, source, destination, lenBurst, nburst);
}

ParseResult MteL1BtOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseDmaTripleOp(parser, result);
}

ParseResult MteL1FbOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseDmaTripleOp(parser, result);
}

void MteL1BtOp::print(OpAsmPrinter &printer) {
  printDmaTripleOpFields(printer, *this);
}

void MteL1FbOp::print(OpAsmPrinter &printer) {
  printDmaTripleOpFields(printer, *this);
}


void MteGmL1FracOp::print(OpAsmPrinter &printer) {
  printer << " " << getSource() << ", " << getDestination() << ", "
          << pto::stringifyCubeLoadFracMode(getMode());
  printer << ", shape(" << getNValue() << ", " << getDValue() << ")";
  printCubeLoadFracSrcLayoutGroup(printer, getSrcInnerStride(),
                                  getSrcOuterStride());
  printer << ", dst_group(" << getGroupCount() << ", " << getDstLoop2Stride()
          << ", " << getDstLoop3Stride() << ", " << getDstLoop4Stride()
          << ")";
  printer << ", ctrl(" << getL2CacheCtrl() << ", " << getSmallc0En() << ")";
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                /*elidedAttrs=*/{"operandSegmentSizes",
                                                 "mode"});
  printer << " : " << getSource().getType() << ", " << getDestination().getType()
          << ", " << pto::stringifyCubeLoadFracMode(getMode())
          << ", shape " << getNValue().getType() << ", " << getDValue().getType();
  printCubeLoadFracSrcLayoutTypes(
      printer, getSrcInnerStride().getType(),
      getSrcOuterStride() ? getSrcOuterStride().getType() : Type());
  printer << ", dst_group " << getGroupCount().getType() << ", "
          << getDstLoop2Stride().getType() << ", "
          << getDstLoop3Stride().getType() << ", "
          << getDstLoop4Stride().getType() << ", ctrl "
          << getL2CacheCtrl().getType() << ", " << getSmallc0En().getType();
}

LogicalResult MteGmL1Op::verify() {
  if (failed(verifyCopyGmToUbufOp(*this, true))) {
    return failure();
  }
  return verifyDmaLoadStoreLoopGroups(
      getOperation(), getLoopCounts(), getLoopSrcStrides(),
      getLoopDstStrides());
}

LogicalResult MteL1UbOp::verify() {
  if (failed(verifyCopyCbufToUbufLikeOp(*this))) {
    return failure();
  }
  return verifyDmaLoadStoreLoopGroups(
      getOperation(), getLoopCounts(), getLoopSrcStrides(),
      getLoopDstStrides());
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

LogicalResult MteGmL1FracOp::verify() {
  if (failed(verifyCopyGmToUbufOp(*this, true))) {
    return failure();
  }

  auto checkNonNegativeConst = [&](Value value, StringRef name) -> LogicalResult {
    APInt intValue;
    if (matchPattern(value, m_ConstantInt(&intValue)) && intValue.isNegative()) {
      return emitOpError() << name << " must be non-negative";
    }
    return success();
  };
  if (failed(checkNonNegativeConst(getGroupCount(), "group_count")) ||
      failed(checkNonNegativeConst(getSrcInnerStride(), "src_inner_stride")) ||
      failed(checkNonNegativeConst(getDstLoop2Stride(), "dst_loop2_stride")) ||
      failed(checkNonNegativeConst(getDstLoop3Stride(), "dst_loop3_stride")) ||
      failed(checkNonNegativeConst(getDstLoop4Stride(), "dst_loop4_stride")) ||
      (getSrcOuterStride() &&
       failed(checkNonNegativeConst(getSrcOuterStride(), "src_outer_stride")))) {
    return failure();
  }

  APInt groupCount;
  if (matchPattern(getGroupCount(), m_ConstantInt(&groupCount)) &&
      groupCount.isZero()) {
    return emitOpError("group_count must be greater than zero");
  }

  APInt smallc0En;
  APInt dValue;
  if (matchPattern(getSmallc0En(), m_ConstantInt(&smallc0En)) &&
      smallc0En.getBoolValue() && matchPattern(getDValue(), m_ConstantInt(&dValue)) &&
      dValue.ugt(mlir::pto::kValue4)) {
    return emitOpError("smallc0_en requires d_value <= 4");
  }

  return success();
}

void MteGmL1Op::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

void MteL1UbOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

void MteL1BtOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

void MteL1FbOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

void MteGmL1FracOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

ParseResult MteL0cL1Op::parse(OpAsmParser &parser, OperationState &result) {
  Builder builder(parser.getContext());
  StructuredAccStoreAsmState state;
  OpAsmParser::UnresolvedOperand source, destination, m, n, srcStride,
      dstStride;
  if (parseRequiredOperandWithComma(parser, source) ||
      parseRequiredOperandWithComma(parser, destination) ||
      parseRequiredOperandWithComma(parser, m) ||
      parseRequiredOperandWithComma(parser, n) ||
      parseRequiredOperandWithComma(parser, srcStride) ||
      parseRequiredOperandWithComma(parser, dstStride) ||
      parseStructuredAccStoreClauses(parser, state) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon()) {
    return failure();
  }
  Type sourceType, destinationType, mType, nType, srcStrideType, dstStrideType;
  if (failed(parseMteL0cL1Types(parser, sourceType, destinationType, mType,
                                 nType, srcStrideType, dstStrideType,
                                 state))) {
    return failure();
  }
  setMteL0cL1SegmentSizes(result, state);
  if (state.atomicType || state.atomicOp) {
    return parser.emitError(parser.getCurrentLocation(),
                            "atomic is only supported for mte_l0c_gm");
  }
  addStructuredAccStoreAttrs<MteL0cL1Op>(result, builder, state);
  if (failed(resolveMteL0cL1Operands(parser, result, source, sourceType,
                                      destination, destinationType, m, mType,
                                      n, nType, srcStride, srcStrideType,
                                      dstStride, dstStrideType, state))) {
    return failure();
  }
  return success();
}
// Shared structured-acc store clause + attribute-dictionary printing for the
// MteL0cL1/MteL0cGm print pair.
template <typename OpTy>
static void printStructuredAccStoreClausesAndAttrs(OpTy op,
                                                   OpAsmPrinter &printer) {
  printStructuredAccStoreClauses(printer, op.getUnitFlag(), op.getPreQuant(),
                                 op.getPreQuantMode(), op.getPreRelu(),
                                 op.getPreReluMode(), op.getClipValue(),
                                 op.getMode(), op.getSplit(),
                                 op.getLoop0SrcStride(), op.getLoop3Count(),
                                 op.getLoop3SrcStride(),
                                 op.getLoop3DstStride(), op.getSatMode(),
                                 op.getAtomicType(), op.getAtomicOp());
  printer.printOptionalAttrDict(op->getAttrs(),
                                /*elidedAttrs=*/{"operandSegmentSizes",
                                                 "mode",
                                                 "unit_flag",
                                                 "pre_quant_mode",
                                                 "pre_relu_mode",
                                                 "atomic_type",
                                                 "atomic_op",
                                                 "sat_mode"});
}

void MteL0cL1Op::print(OpAsmPrinter &printer) {
  printer << " " << getSource() << ", " << getDestination() << ", " << getM()
          << ", " << getN() << ", " << getSrcStride() << ", " << getDstStride();
  printStructuredAccStoreClausesAndAttrs(*this, printer);
  printer << " : " << getSource().getType() << ", " << getDestination().getType()
          << ", " << getM().getType() << ", " << getN().getType() << ", "
          << getSrcStride().getType() << ", " << getDstStride().getType();
  printStructuredAccStoreOptionalTypes(
      printer, getPreQuant(), getPreRelu(), getClipValue(), getSplit(),
      getLoop0SrcStride(), getLoop3Count(), getLoop3SrcStride(),
      getLoop3DstStride());
}

static LogicalResult verifyMteL1L0LoadOperands(
    Operation *op, ArrayRef<Value> shapeOperands,
    ArrayRef<StringRef> shapeNames, ArrayRef<Value> fullOperands) {
  const bool hasShape = llvm::any_of(shapeOperands, [](Value value) {
    return static_cast<bool>(value);
  });
  const bool hasFull = llvm::any_of(fullOperands, [](Value value) {
    return static_cast<bool>(value);
  });
  if (hasShape && hasFull) {
    return op->emitOpError(
        "cannot mix shape-derived operands with full control operands");
}
  if (!hasShape && !hasFull) {
    return op->emitOpError(
        "requires either all shape-derived operands or all full control operands");
}
  if (hasShape) {
    for (auto [value, name] : llvm::zip(shapeOperands, shapeNames)) {
      if (!value) {
        return op->emitOpError()
               << "shape-derived form requires " << name;
}
}
    return verifyCubeBridgeLoadStart(op, shapeOperands[2], shapeNames[2],
                                     shapeOperands[3], shapeNames[3]);
  }
  static constexpr StringRef kFullNames[] = {
      "m_start", "k_start", "m_step", "k_step", "src_stride", "dst_stride"};
  for (auto [value, name] : llvm::zip(fullOperands, kFullNames)) {
    if (!value) {
      return op->emitOpError() << "full control form requires " << name;
}
}
  constexpr int64_t kU16Max = 65535;
  constexpr int64_t kU8Max = 255;
  if (failed(verifyStaticControlRange(op, fullOperands[0], "m_start", 0,
                                      kU16Max)) ||
      failed(verifyStaticControlRange(op, fullOperands[1], "k_start", 0,
                                      kU16Max)) ||
      failed(verifyStaticControlRange(op, fullOperands[2], "m_step", 1,
                                      kU8Max)) ||
      failed(verifyStaticControlRange(op, fullOperands[3], "k_step", 1,
                                      kU8Max)) ||
      failed(verifyStaticControlRange(op, fullOperands[4], "src_stride", 1,
                                      kU16Max)) ||
      failed(verifyStaticControlRange(op, fullOperands[5], "dst_stride", 1,
                                      kU16Max))) {
    return failure();
}
  return success();
}

LogicalResult MteL0cL1Op::verify() {
  if (!isBufferLike(getSource().getType()) ||
      !isBufferLike(getDestination().getType())) {
    return emitOpError("requires buffer-like source and destination");
}
  std::optional<AddressSpace> sourceSpace =
      getBufferAddressSpace(getSource().getType());
  std::optional<AddressSpace> destinationSpace =
      getBufferAddressSpace(getDestination().getType());
  if (sourceSpace != AddressSpace::ACC || destinationSpace != AddressSpace::MAT) {
    return emitOpError("requires ACC source and MAT destination");
  }
  return verifyStructuredAccStoreLike(
      *this, getSource().getType(), getDestination().getType(), getPreQuant(), getPreRelu(),
      getClipValue(), getSplit(), getLoop0SrcStride(), getLoop3Count(),
      getLoop3SrcStride(), getLoop3DstStride(), getUnitFlag(),
      getPreQuantMode(), getPreReluMode(), getMode(), std::nullopt,
      std::nullopt, /*allowAtomic=*/false);
}

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

LogicalResult MteL1L0bOp::verify() {
  if (failed(verifyCubeBridgeLoadLikeOp(*this, AddressSpace::RIGHT, "RIGHT"))) {
    return failure();
  }
  return verifyMteL1L0LoadOperands(
      getOperation(), {getK(), getN(), getStartRow(), getStartCol()},
      {"k", "n", "start_row", "start_col"},
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

void MteL1L0aOp::print(OpAsmPrinter &printer) {
  static constexpr StringRef kShapeNames[] = {
      "m", "k", "start_row", "start_col"};
  static constexpr StringRef kFullNames[] = {
      "m_start", "k_start", "m_step", "k_step", "src_stride", "dst_stride"};
  printMteL1L0OptionalOperandsOp(
      printer, getOperation(), getSource(), getDestination(),
      {getM(), getK(), getStartRow(), getStartCol()}, kShapeNames,
      {getMStart(), getKStart(), getMStep(), getKStep(), getSrcStride(),
       getDstStride()},
      kFullNames);
}

ParseResult MteL1L0bOp::parse(OpAsmParser &parser, OperationState &result) {
  static constexpr StringRef kShapeNames[] = {
      "k", "n", "start_row", "start_col"};
  static constexpr StringRef kFullNames[] = {
      "m_start", "k_start", "m_step", "k_step", "src_stride", "dst_stride"};
  return parseMteL1L0OptionalOperandsOp<MteL1L0bOp>(
      parser, result, kShapeNames, kFullNames);
}

void MteL1L0bOp::print(OpAsmPrinter &printer) {
  static constexpr StringRef kShapeNames[] = {
      "k", "n", "start_row", "start_col"};
  static constexpr StringRef kFullNames[] = {
      "m_start", "k_start", "m_step", "k_step", "src_stride", "dst_stride"};
  printMteL1L0OptionalOperandsOp(
      printer, getOperation(), getSource(), getDestination(),
      {getK(), getN(), getStartRow(), getStartCol()}, kShapeNames,
      {getMStart(), getKStart(), getMStep(), getKStep(), getSrcStride(),
       getDstStride()},
      kFullNames);
}

ParseResult MteL1L0aMxOp::parse(OpAsmParser &parser, OperationState &result) {
  static constexpr StringRef kShapeNames[] = {
      "m", "k", "start_row", "start_col"};
  static constexpr StringRef kFullNames[] = {
      "x_start", "y_start", "x_step", "y_step", "src_stride", "dst_stride"};
  return parseMteL1L0OptionalOperandsOp<MteL1L0aMxOp>(
      parser, result, kShapeNames, kFullNames, "MX operands");
}

void MteL1L0aMxOp::print(OpAsmPrinter &printer) {
  static constexpr StringRef kShapeNames[] = {
      "m", "k", "start_row", "start_col"};
  static constexpr StringRef kFullNames[] = {
      "x_start", "y_start", "x_step", "y_step", "src_stride", "dst_stride"};
  printMteL1L0OptionalOperandsOp(
      printer, getOperation(), getSource(), getDestination(),
      {getM(), getK(), getStartRow(), getStartCol()}, kShapeNames,
      {getXStart(), getYStart(), getXStep(), getYStep(), getSrcStride(),
       getDstStride()},
      kFullNames);
}

LogicalResult MteL1L0aMxOp::verify() {
  if (failed(verifyCubeBridgeLoadLikeOp(*this, AddressSpace::LEFT, "LEFT"))) {
    return failure();
  }
  if (failed(verifyMxLoadOperands(
          getOperation(), {getM(), getK(), getStartRow(), getStartCol()},
          {"m", "k", "start_row", "start_col"},
          {getXStart(), getYStart(), getXStep(), getYStep(), getSrcStride(),
           getDstStride()}))) {
    return failure();
  }
  return verifyMxLoadAlignment(getOperation(), getSource(), getDestination());
}

ParseResult MteL1L0bMxOp::parse(OpAsmParser &parser, OperationState &result) {
  static constexpr StringRef kShapeNames[] = {
      "k", "n", "start_row", "start_col"};
  static constexpr StringRef kFullNames[] = {
      "x_start", "y_start", "x_step", "y_step", "src_stride", "dst_stride"};
  return parseMteL1L0OptionalOperandsOp<MteL1L0bMxOp>(
      parser, result, kShapeNames, kFullNames, "MX operands");
}

void MteL1L0bMxOp::print(OpAsmPrinter &printer) {
  static constexpr StringRef kShapeNames[] = {
      "k", "n", "start_row", "start_col"};
  static constexpr StringRef kFullNames[] = {
      "x_start", "y_start", "x_step", "y_step", "src_stride", "dst_stride"};
  printMteL1L0OptionalOperandsOp(
      printer, getOperation(), getSource(), getDestination(),
      {getK(), getN(), getStartRow(), getStartCol()}, kShapeNames,
      {getXStart(), getYStart(), getXStep(), getYStep(), getSrcStride(),
       getDstStride()},
      kFullNames);
}

LogicalResult MteL1L0bMxOp::verify() {
  if (failed(verifyCubeBridgeLoadLikeOp(*this, AddressSpace::RIGHT, "RIGHT"))) {
    return failure();
  }
  if (failed(verifyMxLoadOperands(
          getOperation(), {getK(), getN(), getStartRow(), getStartCol()},
          {"k", "n", "start_row", "start_col"},
          {getXStart(), getYStart(), getXStep(), getYStep(), getSrcStride(),
           getDstStride()}))) {
    return failure();
  }
  return verifyMxLoadAlignment(getOperation(), getSource(), getDestination());
}

void MteL1L0aOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

void MteL1L0bOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

void MteL1L0aMxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

void MteL1L0bMxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

void MteL0cL1Op::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

ParseResult MteL0cGmOp::parse(OpAsmParser &parser, OperationState &result) {
  Builder builder(parser.getContext());
  StructuredAccStoreAsmState state;
  OpAsmParser::UnresolvedOperand source, destination, m, n, srcStride,
      dstStride, sid, l2CacheCtrl;
  if (parseRequiredOperandWithComma(parser, source) ||
      parseRequiredOperandWithComma(parser, destination) ||
      parseRequiredOperandWithComma(parser, m) ||
      parseRequiredOperandWithComma(parser, n) ||
      parseRequiredOperandWithComma(parser, srcStride) ||
      parseRequiredOperandWithComma(parser, dstStride) ||
      parseRequiredOperandWithComma(parser, sid) ||
      parseRequiredOperandWithComma(parser, l2CacheCtrl) ||
      parseStructuredAccStoreClauses(parser, state) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon()) {
    return failure();
  }
  Type sourceType, destinationType, mType, nType, srcStrideType, dstStrideType,
      sidType, l2CacheCtrlType;
  if (failed(parseMteL0cGmTypes(parser, sourceType, destinationType, mType,
                                 nType, srcStrideType, dstStrideType,
                                 sidType, l2CacheCtrlType, state))) {
    return failure();
  }
  setMteL0cGmSegmentSizes(result, state);
  addStructuredAccStoreAttrs<MteL0cGmOp>(result, builder, state);
  if (failed(resolveMteL0cGmOperands(
          parser, result, source, sourceType, destination, destinationType,
          m, mType, n, nType, srcStride, srcStrideType, dstStride,
          dstStrideType, sid, sidType, l2CacheCtrl, l2CacheCtrlType, state))) {
    return failure();
  }
  return success();
}
void MteL0cGmOp::print(OpAsmPrinter &printer) {
  printer << " " << getSource() << ", " << getDestination() << ", " << getM()
          << ", " << getN() << ", " << getSrcStride() << ", "
          << getDstStride() << ", " << getSid() << ", " << getL2CacheCtrl();
  printStructuredAccStoreClausesAndAttrs(*this, printer);
  printer << " : " << getSource().getType() << ", " << getDestination().getType()
          << ", " << getM().getType() << ", " << getN().getType() << ", "
          << getSrcStride().getType() << ", " << getDstStride().getType()
          << ", " << getSid().getType() << ", " << getL2CacheCtrl().getType();
  printStructuredAccStoreOptionalTypes(
      printer, getPreQuant(), getPreRelu(), getClipValue(), getSplit(),
      getLoop0SrcStride(), getLoop3Count(), getLoop3SrcStride(),
      getLoop3DstStride());
}

LogicalResult MteL0cGmOp::verify() {
  if (!isBufferLike(getSource().getType()) ||
      !isBufferLike(getDestination().getType())) {
    return emitOpError("requires buffer-like source and destination");
  }
  std::optional<AddressSpace> sourceSpace =
      getBufferAddressSpace(getSource().getType());
  std::optional<AddressSpace> destinationSpace =
      getBufferAddressSpace(getDestination().getType());
  if (sourceSpace != AddressSpace::ACC || destinationSpace != AddressSpace::GM) {
    return emitOpError("requires ACC source and GM destination");
  }
  return verifyStructuredAccStoreLike(
      *this, getSource().getType(), getDestination().getType(), getPreQuant(), getPreRelu(),
      getClipValue(), getSplit(), getLoop0SrcStride(), getLoop3Count(),
      getLoop3SrcStride(), getLoop3DstStride(), getUnitFlag(),
      getPreQuantMode(), getPreReluMode(), getMode(), getAtomicType(),
      getAtomicOp(), /*allowAtomic=*/true);
}

void MteL0cGmOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

ParseResult MteL0cUbOp::parse(OpAsmParser &parser, OperationState &result) {
  Builder builder(parser.getContext());
  StructuredAccStoreAsmState state;
  OpAsmParser::UnresolvedOperand source, destination, m, n, srcStride,
      dstStride, subBlockId;
  bool hasSubBlockId = false;
  AccStoreUbDstMode dstMode = AccStoreUbDstMode::Single;
  if (failed(parseMteL0cUbBasicOperands(parser, source, destination, m,
                                        n, srcStride, dstStride))) {
    return failure();
  }
  if (failed(parseMteL0cUbDstMode(parser, dstMode, subBlockId, hasSubBlockId))) {
    return failure();
  }
  if (succeeded(parser.parseOptionalComma()) &&
      parseStructuredAccStoreClauses(parser, state)) {
    return failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon()) {
    return failure();
  }
  Type sourceType, destinationType, mType, nType, srcStrideType,
      dstStrideType, subBlockIdType;
  if (failed(parseMteL0cUbTypes(parser, sourceType, destinationType, mType,
                                nType, srcStrideType, dstStrideType,
                                hasSubBlockId, subBlockIdType, state))) {
    return failure();
  }
  setStructuredAccStoreSegmentSizes<MteL0cUbOp>(
      result, {1, 1, 1, 1, 1, 1, !state.preQuantOperands.empty() ? 1 : 0,
               !state.preReluOperands.empty() ? 1 : 0,
               !state.clipValueOperands.empty() ? 1 : 0,
               hasSubBlockId ? 1 : 0,
               !state.splitOperands.empty() ? 1 : 0,
               !state.loop0SrcStrideOperands.empty() ? 1 : 0,
               !state.loop3CountOperands.empty() ? 1 : 0,
               !state.loop3SrcStrideOperands.empty() ? 1 : 0,
               !state.loop3DstStrideOperands.empty() ? 1 : 0});
  if (state.atomicType || state.atomicOp) {
    return parser.emitError(parser.getCurrentLocation(),
                            "atomic is only supported for mte_l0c_gm");
  }
  addStructuredAccStoreAttrs<MteL0cUbOp>(result, builder, state);
  result.addAttribute("dst_mode", AccStoreUbDstModeAttr::get(builder.getContext(), dstMode));
  return resolveMteL0cUbOperands(parser, result, source, sourceType,
                                 destination, destinationType, m, mType, n,
                                 nType, srcStride, srcStrideType, dstStride,
                                 dstStrideType, hasSubBlockId, subBlockId,
                                 subBlockIdType, state);
}

void MteL0cUbOp::print(OpAsmPrinter &printer) {
  printer << " " << getSource() << ", " << getDestination() << ", " << getM()
          << ", " << getN() << ", " << getSrcStride() << ", "
          << getDstStride() << ", dst_mode(";
  switch (getDstMode()) {
  case AccStoreUbDstMode::Single:
    printer << getSubBlockid();
    break;
  case AccStoreUbDstMode::SplitM:
    printer << "split_m";
    break;
  case AccStoreUbDstMode::SplitN:
    printer << "split_n";
    break;
  }
  printer << ")";
  printStructuredAccStoreClauses(printer, getUnitFlag(), getPreQuant(),
                                 getPreQuantMode(), getPreRelu(),
                                 getPreReluMode(), getClipValue(), getMode(),
                                 getSplit(), getLoop0SrcStride(),
                                 getLoop3Count(), getLoop3SrcStride(),
                                 getLoop3DstStride(), getSatMode(),
                                 std::nullopt, std::nullopt);
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                /*elidedAttrs=*/{"operandSegmentSizes",
                                                 "mode",
                                                 "unit_flag",
                                                 "pre_quant_mode",
                                                 "pre_relu_mode",
                                                 "dst_mode",
                                                 "sat_mode"});
  printer << " : " << getSource().getType() << ", " << getDestination().getType()
          << ", " << getM().getType() << ", " << getN().getType() << ", "
          << getSrcStride().getType() << ", " << getDstStride().getType();
  if (getSubBlockid()) {
    printer << ", " << getSubBlockid().getType();
  }
  printStructuredAccStoreOptionalTypes(
      printer, getPreQuant(), getPreRelu(), getClipValue(), getSplit(),
      getLoop0SrcStride(), getLoop3Count(), getLoop3SrcStride(),
      getLoop3DstStride());
}

static LogicalResult verifyMteL0cUbBufferSpaces(MteL0cUbOp op) {
  if (!isBufferLike(op.getSource().getType()) ||
      !isBufferLike(op.getDestination().getType())) {
    return op.emitOpError("requires buffer-like source and destination");
  }
  std::optional<AddressSpace> sourceSpace =
      getBufferAddressSpace(op.getSource().getType());
  std::optional<AddressSpace> destinationSpace =
      getBufferAddressSpace(op.getDestination().getType());
  if (sourceSpace != AddressSpace::ACC || destinationSpace != AddressSpace::VEC) {
    return op.emitOpError("requires ACC source and UB destination");
  }
  return success();
}

static LogicalResult verifyMteL0cUbSubBlockId(MteL0cUbOp op) {
  if (!op.getSubBlockid()) {
    return op.emitOpError("dst_mode(%sub_blockid) requires a sub_blockid operand");
  }
  APInt subBlockId;
  if (matchPattern(op.getSubBlockid(), m_ConstantInt(&subBlockId)) &&
      subBlockId.ugt(1)) {
    return op.emitOpError("sub_blockid must be 0 or 1");
  }
  return success();
}

static LogicalResult verifyMteL0cUbSplitRestrictions(MteL0cUbOp op) {
  if (op.getPreQuant() || op.getPreRelu() || op.getClipValue() ||
      op.getPreQuantMode() || op.getPreReluMode() || op.getSplit() ||
      op.getLoop0SrcStride() || op.getLoop3Count() ||
      op.getLoop3SrcStride() || op.getLoop3DstStride()) {
    return op.emitOpError("dual destination mode cannot be combined with "
                          "pre_quant, pre_relu, clip, nz2dn, nz2nz, or loop3");
  }
  if (op.getMode() && *op.getMode() != AccStoreMode::Nz2nd) {
    return op.emitOpError("dual destination mode requires normal or nz2nd layout");
  }
  APInt mValue;
  APInt nValue;
  if (op.getDstMode() == AccStoreUbDstMode::SplitM &&
      matchPattern(op.getM(), m_ConstantInt(&mValue)) &&
      mValue.getZExtValue() % mlir::pto::kValue2 != 0) {
    return op.emitOpError("split-M dual destination requires m to be even");
  }
  if (op.getDstMode() == AccStoreUbDstMode::SplitN &&
      matchPattern(op.getN(), m_ConstantInt(&nValue)) &&
      nValue.getZExtValue() % mlir::pto::kValue32 != 0) {
    return op.emitOpError("split-N dual destination requires n to be a multiple of 32");
  }
  return success();
}

LogicalResult MteL0cUbOp::verify() {
  if (failed(verifyMteL0cUbBufferSpaces(*this)) ||
      failed(verifyStructuredAccStoreLike(
          *this, getSource().getType(), getDestination().getType(), getPreQuant(), getPreRelu(),
          getClipValue(), getSplit(), getLoop0SrcStride(), getLoop3Count(),
          getLoop3SrcStride(), getLoop3DstStride(), getUnitFlag(),
          getPreQuantMode(), getPreReluMode(), getMode(), std::nullopt,
          std::nullopt, /*allowAtomic=*/false))) {
    return failure();
  }
  if (getDstMode() == AccStoreUbDstMode::Single) {
    return verifyMteL0cUbSubBlockId(*this);
  }
  if (getSubBlockid()) {
    return emitOpError("split destination modes do not accept sub_blockid");
  }
  return verifyMteL0cUbSplitRestrictions(*this);
}

void MteL0cUbOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

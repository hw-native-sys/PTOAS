// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMI_ops_mem.cpp - VMI memory ops -----------------------===//
//===----------------------------------------------------------------------===//

#include "VMIInternal.h"

using namespace mlir;
using namespace mlir::pto;

// Batch8: 访存族双胞胎 op 共用校验
template <typename OpTy>
static LogicalResult verifyVMIStoreDataMask(OpTy op, VMIVRegType &valueType,
                                            VMIMaskType &maskType) {
  valueType = cast<VMIVRegType>(op.getValue().getType());
  maskType = cast<VMIMaskType>(op.getMask().getType());
  if (failed(verifyMemoryElementMatches(op.getOperation(),
                                        op.getDestination().getType(), valueType,
                                        "destination"))) {
    return failure();
  }
  if (failed(verifyUBBackedMemory(op.getOperation(), op.getDestination().getType(),
                                  "destination"))) {
    return failure();
  }
  return success();
}

template <typename OpTy>
static LogicalResult verifyVMIMaskedLoadLike(OpTy op) {
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  auto passthruType = cast<VMIVRegType>(op.getPassthru().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  if (failed(verifyMemoryElementMatches(op.getOperation(), op.getSource().getType(),
                                        resultType, "source"))) {
    return failure();
  }
  if (failed(verifyUBBackedMemory(op.getOperation(), op.getSource().getType(),
                                  "source"))) {
    return failure();
  }
  if (failed(verifyAllSameVRegShapeAndLayout(op.getOperation(),
                                             {passthruType, resultType},
                                             /*requireSameElement=*/true))) {
    return failure();
  }
  return verifyMaskMatchesData(op.getOperation(), maskType, resultType);
}

template <typename OpTy>
static LogicalResult verifyVMIGroupLoadMemory(OpTy op, VMIVRegType resultType) {
  if (failed(verifyMemoryElementMatches(op.getOperation(), op.getSource().getType(),
                                        resultType, "source"))) {
    return failure();
  }
  if (failed(verifyUBBackedMemory(op.getOperation(), op.getSource().getType(),
                                  "source"))) {
    return failure();
  }
  return success();
}


LogicalResult VMICompressStoreOp::verify() {
  VMIVRegType valueType;
  VMIMaskType maskType;
  if (failed(verifyVMIStoreDataMask(*this, valueType, maskType))) {
    return failure();
  }
  return verifyMaskMatchesData(getOperation(), maskType, valueType);
}

void VMICompressStoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VMILoadOp::verify() {
  if (failed(verifyMemoryElementMatches(
          getOperation(), getSource().getType(),
          cast<VMIVRegType>(getResult().getType()), "source"))) {
    return failure();
  }
  return verifyUBBackedMemory(getOperation(), getSource().getType(), "source");
}

void VMILoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VMIDeinterleaveLoadOp::verify() {
  auto lowType = cast<VMIVRegType>(getLow().getType());
  auto highType = cast<VMIVRegType>(getHigh().getType());
  if (failed(verifyAllSameVRegShapeAndLayout(getOperation(),
                                             {lowType, highType},
                                             /*requireSameElement=*/true))) {
    return failure();
  }
  if (failed(verifyMemoryElementMatches(getOperation(), getSource().getType(),
                                        lowType, "source"))) {
    return failure();
  }
  if (failed(verifyUBBackedMemory(getOperation(), getSource().getType(),
                                  "source"))) {
    return failure();
  }
  if (failed(verifyContiguousIfLayoutAssigned(getOperation(), lowType,
                                              "low result")) ||
      failed(verifyContiguousIfLayoutAssigned(getOperation(), highType,
                                              "high result"))) {
    return failure();
  }
  return success();
}

void VMIDeinterleaveLoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VMIGroupLoadOp::verify() {
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (failed(verifyMemoryElementMatches(getOperation(), getSource().getType(),
                                        resultType, "source"))) {
    return failure();
  }
  if (failed(verifyUBBackedMemory(getOperation(), getSource().getType(),
                                  "source"))) {
    return failure();
  }
  return verifyNumGroups(getOperation(), resultType,
                         getNumGroupsAttr().getInt());
}

void VMIGroupLoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VMIGroupSlotLoadOp::verify() {
  auto resultType = cast<VMIVRegType>(getResult().getType());
  int64_t numGroups = getNumGroupsAttr().getInt();
  if (resultType.getElementCount() != numGroups) {
    return emitOpError(
        "requires result logical lane count to match num_groups");
  }
  if (failed(verifyVMIGroupLoadMemory(*this, resultType))) {
    return failure();
  }
  if (auto resultLayout = resultType.getLayoutAttr()) {
    if (!resultLayout.isGroupSlots() ||
        resultLayout.getNumGroups() != numGroups) {
      return emitOpError() << "requires layout-assigned result to use "
                              "#pto.vmi.layout<num_groups = "
                           << numGroups << ">";
    }
  }
  return verifyNumGroups(getOperation(), resultType, numGroups);
}

void VMIGroupSlotLoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VMIGroupBroadcastLoadOp::verify() {
  auto resultType = cast<VMIVRegType>(getResult().getType());
  int64_t numGroups = getNumGroupsAttr().getInt();
  if (numGroups <= 0) {
    return emitOpError("requires num_groups to be positive");
  }
  if (resultType.getElementCount() % numGroups != 0) {
    return emitOpError(
        "requires num_groups to evenly divide result logical lane count");
  }
  if (failed(verifyVMIGroupLoadMemory(*this, resultType))) {
    return failure();
  }
  if (auto resultLayout = resultType.getLayoutAttr()) {
    if (resultLayout.isGroupSlots()) {
      return emitOpError(
          "requires layout-assigned result to use a dense VMI layout");
    }
  }
  return verifyNumGroups(getOperation(), resultType, numGroups);
}

void VMIGroupBroadcastLoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VMIMaskedLoadOp::verify() { return verifyVMIMaskedLoadLike(*this); }

void VMIMaskedLoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

static LogicalResult verifyVMIOffsetIndexType(Operation *op,
                                                  Type indexElementType,
                                                  StringRef valueNoun) {
  auto intType = dyn_cast_or_null<IntegerType>(indexElementType);
  if (!intType || intType.isSigned() ||
      (intType.getWidth() != mlir::pto::kValue16 &&
       intType.getWidth() != mlir::pto::kValue32)) {
    return op->emitOpError(
               "requires signless or unsigned 16-bit or 32-bit integer ")
           << valueNoun;
  }
  return success();
}

LogicalResult VMIGatherOp::verify() {
  auto indicesType = cast<VMIVRegType>(getIndices().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());
  auto passthruType = cast<VMIVRegType>(getPassthru().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (failed(verifyGatherMemoryElementMatches(getOperation(), getSource().getType(),
                                        resultType, "source"))) {
    return failure();
  }
  if (failed(verifyUBBackedMemory(getOperation(), getSource().getType(),
                                  "source"))) {
    return failure();
  }

  auto indexElementType = dyn_cast<IntegerType>(indicesType.getElementType());
  if (failed(verifyVMIOffsetIndexType(getOperation(), indexElementType,
                                      "indices"))) {
    return failure();
  }

  if (failed(verifyAllSameVRegShapeAndLayout(
          getOperation(), {indicesType, passthruType, resultType},
          /*requireSameElement=*/false))) {
    return failure();
  }
  if (failed(verifyAllSameVRegShapeAndLayout(getOperation(),
                                             {passthruType, resultType},
                                             /*requireSameElement=*/true))) {
    return failure();
  }

  if (indexElementType.getWidth() == mlir::pto::kValue16 &&
      !isSupported16BitGatherResult(
          getMemoryElementType(getSource().getType()),
          resultType.getElementType())) {
    return emitOpError(
        "requires i16/ui16/f16/bf16 result and passthru element type when "
        "using ui16 indices, or i8/ui8 -> i16/ui16 integer promotion");
  }
  return verifyMaskMatchesData(getOperation(), maskType, resultType);
}

void VMIGatherOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VMIExpandLoadOp::verify() { return verifyVMIMaskedLoadLike(*this); }

void VMIExpandLoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VMIStoreOp::verify() {
  if (failed(verifyMemoryElementMatches(
          getOperation(), getDestination().getType(),
          cast<VMIVRegType>(getValue().getType()), "destination"))) {
    return failure();
  }
  return verifyUBBackedMemory(getOperation(), getDestination().getType(),
                              "destination");
}

void VMIStoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VMIInterleaveStoreOp::verify() {
  auto lowType = cast<VMIVRegType>(getLow().getType());
  auto highType = cast<VMIVRegType>(getHigh().getType());
  if (failed(verifyAllSameVRegShapeAndLayout(getOperation(),
                                             {lowType, highType},
                                             /*requireSameElement=*/true))) {
    return failure();
  }
  if (failed(verifyMemoryElementMatches(getOperation(),
                                        getDestination().getType(), lowType,
                                        "destination"))) {
    return failure();
  }
  if (failed(verifyUBBackedMemory(getOperation(), getDestination().getType(),
                                  "destination"))) {
    return failure();
  }
  if (failed(verifyContiguousIfLayoutAssigned(getOperation(), lowType,
                                              "low input")) ||
      failed(verifyContiguousIfLayoutAssigned(getOperation(), highType,
                                              "high input"))) {
    return failure();
  }
  return success();
}

void VMIInterleaveStoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VMIGroupStoreOp::verify() {
  auto valueType = cast<VMIVRegType>(getValue().getType());
  if (!isPackedByteGroupStore(getDestination().getType(), valueType) &&
      failed(verifyMemoryElementMatches(getOperation(),
                                        getDestination().getType(), valueType,
                                        "destination"))) {
    return failure();
  }
  if (failed(verifyUBBackedMemory(getOperation(), getDestination().getType(),
                                  "destination"))) {
    return failure();
  }
  return verifyNumGroups(getOperation(), valueType,
                         getNumGroupsAttr().getInt());
}

void VMIGroupStoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VMIStrideLoadOp::verify() {
  auto resultType = cast<VMIVRegType>(getResult().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());
  if (failed(verifyMemoryElementMatches(getOperation(), getSource().getType(),
                                        resultType, "source"))) {
    return failure();
  }
  if (failed(verifyUBBackedMemory(getOperation(), getSource().getType(),
                                  "source"))) {
    return failure();
  }
  return verifyMaskMatchesData(getOperation(), maskType, resultType);
}

void VMIStrideLoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VMIMaskedStoreOp::verify() {
  VMIVRegType valueType;
  VMIMaskType maskType;
  if (failed(verifyVMIStoreDataMask(*this, valueType, maskType))) {
    return failure();
  }
  return verifyMaskMatchesData(getOperation(), maskType, valueType);
}

void VMIMaskedStoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VMIStrideStoreOp::verify() {
  VMIVRegType valueType;
  VMIMaskType maskType;
  if (failed(verifyVMIStoreDataMask(*this, valueType, maskType))) {
    return failure();
  }
  return verifyMaskMatchesData(getOperation(), maskType, valueType);
}

void VMIStrideStoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

//===----------------------------------------------------------------------===//

LogicalResult VMIScatterOp::verify() {
  VMIVRegType valueType;
  VMIMaskType maskType;
  if (failed(verifyVMIStoreDataMask(*this, valueType, maskType))) {
    return failure();
  }

  auto indicesType = cast<VMIVRegType>(getIndices().getType());
  auto indexElementType = dyn_cast<IntegerType>(indicesType.getElementType());
  if (!indexElementType || indexElementType.isSigned() ||
      (indexElementType.getWidth() != mlir::pto::kValue32 &&
       indexElementType.getWidth() != mlir::pto::kValue16)) {
    return emitOpError(
        "requires signless or unsigned 16-bit or 32-bit integer indices");
  }

  if (failed(verifyAllSameVRegShapeAndLayout(getOperation(),
                                             {valueType, indicesType},
                                             /*requireSameElement=*/false))) {
    return failure();
  }
  return verifyMaskMatchesData(getOperation(), maskType, valueType);
}

void VMIScatterOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

//===----------------------------------------------------------------------===//

static const std::set<StringRef> &validDistModes() {
  static const std::set<StringRef> modes = {"continuous", "dintlv", "brc"};
  return modes;
}

static const std::set<StringRef> &validStoreDistModes() {
  static const std::set<StringRef> modes = {"continuous", "intlv"};
  return modes;
}

static const std::set<StringRef> &validPModes() {
  static const std::set<StringRef> modes = {"zero", "merge"};
  return modes;
}

LogicalResult VMIVgatherOp::verify() {
  auto offsetsType = cast<VMIVRegType>(getOffsets().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());

  if (failed(verifyUBBackedMemory(getOperation(), getSource().getType(),
                                  "source"))) {
    return failure();
  }

  if (failed(verifyGatherMemoryElementMatches(getOperation(), getSource().getType(),
                                        resultType, "source"))) {
    return failure();
  }

  auto indexElementType =
      dyn_cast<IntegerType>(offsetsType.getElementType());
  if (failed(verifyVMIOffsetIndexType(getOperation(), indexElementType,
                                      "offsets"))) {
    return failure();
  }

  if (failed(verifyAllSameVRegShapeAndLayout(
          getOperation(), {offsetsType, resultType},
          /*requireSameElement=*/false))) {
    return failure();
  }
  if (failed(verifyMaskMatchesData(getOperation(), maskType, resultType))) {
    return failure();
  }

  // 16-bit offsets address the pto.vgather2 / b16 mask path.  It supports
  // i16/ui16/f16/bf16 same-width gather and i8/ui8 -> i16/ui16 promotion.
  if (indexElementType.getWidth() == mlir::pto::kValue16 &&
      !isSupported16BitGatherResult(
          getMemoryElementType(getSource().getType()),
          resultType.getElementType())) {
    return emitOpError(
        "requires i16/ui16/f16/bf16 result element type when using ui16 "
        "offsets, or i8/ui8 -> i16/ui16 integer promotion");
  }

  if (auto pmode = getPmode()) {
    if (pmode.value() != "merge" && pmode.value() != "zero") {
      return emitOpError("pmode must be 'merge' or 'zero'");
    }
  }
  return success();
}

void VMIVgatherOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VMIVgatherbOp::verify() {
  auto offsetsType = cast<VMIVRegType>(getOffsets().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());

  if (failed(verifyUBBackedMemory(getOperation(), getSource().getType(),
                                  "source"))) {
    return failure();
  }

  if (failed(verifyMemoryElementMatches(getOperation(), getSource().getType(),
                                        resultType, "source"))) {
    return failure();
  }

  auto indexElementType =
      dyn_cast<IntegerType>(offsetsType.getElementType());
  if (failed(verifyVMIOffsetIndexType(getOperation(), indexElementType,
                                      "offsets"))) {
    return failure();
  }

  return verifyMaskMatchesDataPmode(*this, maskType, resultType);
}

void VMIVgatherbOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VMIVscatterOp::verify() {
  auto valueType = cast<VMIVRegType>(getValue().getType());
  auto offsetsType = cast<VMIVRegType>(getOffsets().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());

  if (failed(verifyUBBackedMemory(getOperation(),
                                  getDestination().getType(), "destination"))) {
    return failure();
  }

  if (failed(verifyMemoryElementMatches(getOperation(),
                                        getDestination().getType(), valueType,
                                        "destination"))) {
    return failure();
  }

  auto indexElementType =
      dyn_cast<IntegerType>(offsetsType.getElementType());
  if (!indexElementType || indexElementType.isSigned() ||
      (indexElementType.getWidth() != mlir::pto::kValue32 &&
       indexElementType.getWidth() != mlir::pto::kValue16)) {
    return emitOpError(
        "requires signless or unsigned 16-bit or 32-bit integer offsets");
  }

  if (failed(verifyAllSameVRegShapeAndLayout(getOperation(),
                                             {valueType, offsetsType},
                                             /*requireSameElement=*/false))) {
    return failure();
  }
  if (failed(verifyMaskMatchesData(getOperation(), maskType, valueType))) {
    return failure();
  }

  if (auto pmode = getPmode()) {
    if (pmode.value() != "merge" && pmode.value() != "zero") {
      return emitOpError("pmode must be 'merge' or 'zero'");
    }
  }
  return success();
}

void VMIVscatterOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

// Parse pre-bracket operands: value(s) + optional [offset].
static ParseResult parseVStorePreBracket(
    OpAsmParser &parser, OpAsmParser::UnresolvedOperand &operand,
    OpAsmParser::UnresolvedOperand &offsetOperand,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &preBracketOperands) {
  if (parser.parseOperand(operand)) {
    return failure();
  }
  preBracketOperands.push_back(operand);
  bool consumedLSquare = false;
  while (!consumedLSquare) {
    if (succeeded(parser.parseOptionalLSquare())) {
      if (parser.parseOperand(offsetOperand) || parser.parseRSquare()) {
        return failure();
      }
      consumedLSquare = true;
      break;
    }
    if (parser.parseComma()) {
      return failure();
    }
    if (succeeded(parser.parseOptionalLSquare())) {
      if (parser.parseOperand(offsetOperand) || parser.parseRSquare()) {
        return failure();
      }
      consumedLSquare = true;
      break;
    }
    if (parser.parseOperand(operand)) {
      return failure();
    }
    preBracketOperands.push_back(operand);
  }
  return success();
}

// Disambiguate post-bracket operands into stride/block/mask.
static void disambiguateVStorePostBracket(
    bool hasGroup, size_t numPostOps, size_t nValues, size_t nTypes,
    bool &hasStride, bool &hasBlock, bool &hasMask,
    int &strideIdx, int &blockIdx, int &maskIdx) {
  hasStride = false;
  hasBlock = false;
  hasMask = false;
  strideIdx = -1;
  blockIdx = -1;
  maskIdx = -1;
  if (hasGroup) {
    // Group mode: post-bracket ops are stride[, mask]
    if (numPostOps >= 1) {
      hasStride = true;
      strideIdx = 0;
    }
    if (numPostOps >= mlir::pto::kValue2) {
      hasMask = true;
      maskIdx = 1;
    }
  } else if (numPostOps == mlir::pto::kValue2) {
    // Block-stride mode with a mask: block_stride, mask.
    hasBlock = true;
    hasMask = true;
    blockIdx = 0;
    maskIdx = 1;
  } else if (numPostOps == 1) {
    // The type list includes mask types, but never block-stride types.
    if (nTypes == nValues + mlir::pto::kValue2) {
      hasMask = true;
      maskIdx = 0;
    } else {
      hasBlock = true;
      blockIdx = 0;
    }
  }
}

// Resolve all parsed operands against their types.
static ParseResult resolveVStoreOperands(
    OpAsmParser &parser, OperationState &result,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &preBracketOperands,
    OpAsmParser::UnresolvedOperand offsetOperand,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &postBracketOps,
    SmallVectorImpl<Type> &types, size_t nValues, bool hasStride,
    bool hasBlock, bool hasMask, int strideIdx, int blockIdx,
    int maskIdx) {
  size_t expectedTypes = nValues + 1 + (hasMask ? 1 : 0);
  size_t nTypes = types.size();
  if (nTypes != expectedTypes) {
    return parser.emitError(parser.getCurrentLocation())
           << "expected " << expectedTypes << " types (" << nValues
           << " value(s), 1 destination" << (hasMask ? ", 1 mask" : "")
           << "), got " << nTypes;
  }
  for (size_t i = 0; i < nValues; ++i) {
    if (parser.resolveOperand(preBracketOperands[i], types[i], result.operands)) {
      return failure();
    }
  }
  Type destType = types[nValues];
  if (parser.resolveOperand(preBracketOperands[nValues], destType,
                            result.operands)) {
    return failure();
  }
  if (parser.resolveOperand(offsetOperand, parser.getBuilder().getIndexType(),
                            result.operands)) {
    return failure();
  }
  if (hasStride &&
      parser.resolveOperand(postBracketOps[strideIdx],
                            parser.getBuilder().getIndexType(),
                            result.operands)) {
    return failure();
  }
  if (hasBlock &&
      parser.resolveOperand(postBracketOps[blockIdx],
                            parser.getBuilder().getIntegerType(mlir::pto::kValue16),
                            result.operands)) {
    return failure();
  }
  if (hasMask) {
    Type maskType = types.back();
    if (parser.resolveOperand(postBracketOps[maskIdx], maskType,
                              result.operands)) {
      return failure();
    }
  }
  return success();
}

ParseResult VMIvStoreOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, mlir::pto::kValue4> preBracketOperands;
  OpAsmParser::UnresolvedOperand operand, offsetOperand;
  SmallVector<OpAsmParser::UnresolvedOperand, mlir::pto::kValue3> postBracketOps;
  if (failed(parseVStorePreBracket(parser, operand, offsetOperand,
                                   preBracketOperands))) {
    return failure();
  }
  if (preBracketOperands.empty()) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected at least one value and one destination");
  }
  // Optional post-bracket operands: stride, block_stride, and/or mask.
  // Up to 2, disambiguated after parsing attrs and the type list.
  while (succeeded(parser.parseOptionalComma())) {
    OpAsmParser::UnresolvedOperand postOp;
    if (parser.parseOperand(postOp)) {
      return failure();
    }
    postBracketOps.push_back(postOp);
    if (postBracketOps.size() >= mlir::pto::kValue3) {
      break;
    }
  }
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }
  SmallVector<Type, mlir::pto::kValue6> types;
  if (parser.parseColon() || parser.parseTypeList(types)) {
    return failure();
  }
  bool hasGroup = result.attributes.get("group") != nullptr;
  bool hasStride, hasBlock, hasMask;
  int strideIdx, blockIdx, maskIdx;
  disambiguateVStorePostBracket(hasGroup, postBracketOps.size(),
                                preBracketOperands.size() - 1, types.size(),
                                hasStride, hasBlock, hasMask,
                                strideIdx, blockIdx, maskIdx);
  size_t nValues = preBracketOperands.size() - 1;
  if (failed(resolveVStoreOperands(parser, result, preBracketOperands,
                                   offsetOperand, postBracketOps, types,
                                   nValues, hasStride, hasBlock, hasMask,
                                   strideIdx, blockIdx, maskIdx))) {
    return failure();
  }
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {static_cast<int32_t>(nValues), 1, 1,
                           hasStride ? 1 : 0, hasBlock ? 1 : 0,
                           hasMask ? 1 : 0}));
  return success();
}

// Shared stride/block_stride operand printing for vstore/vload.
static void printVStrideBlockStride(OpAsmPrinter &p, Value stride,
                                    Value blockStride) {
  if (stride) {
    p << ", ";
    p.printOperand(stride);
  }
  if (blockStride) {
    p << ", ";
    p.printOperand(blockStride);
  }
}

void VMIvStoreOp::print(OpAsmPrinter &p) {
  for (auto val : getValues()) {
    p << ' ' << val << ", ";
  }
  p << getDestination() << '[';
  p.printOperand(getOffset());
  p << ']';
  printVStrideBlockStride(p, getStride(), getBlockStride());
  if (!getMask().empty()) {
    p << ", ";
    p.printOperand(getMask()[0]);
  }
  p.printOptionalAttrDict((*this)->getAttrs(), {"operandSegmentSizes"});
  p << " : ";
  for (auto val : getValues()) {
    p << val.getType() << ", ";
  }
  p << getDestination().getType();
  if (!getMask().empty()) {
    p << ", " << getMask()[0].getType();
  }
}

static LogicalResult verifyBlockStrideExclusions(
    Operation *op, bool hasBlock, std::optional<StringRef> distMode,
    bool hasGroup) {
  if (hasBlock) {
    if (distMode) {
      return op->emitOpError(
          "block_stride and dist_mode are mutually exclusive");
    }
    if (hasGroup) {
      return op->emitOpError("block_stride and group are mutually exclusive");
    }
  }
  return success();
}

static LogicalResult verifyVStoreGroupAndBlockModes(
    Operation *op, bool hasGroup, std::optional<StringRef> distMode,
    bool hasStride, bool hasBlock, bool hasMask, int64_t numGroups,
    size_t nValues, ValueRange values) {
  if (hasGroup && distMode) {
    return op->emitOpError("group and dist_mode are mutually exclusive");
  }
  if (hasGroup && !hasStride) {
    return op->emitOpError("group requires a stride operand");
  }
  if (!hasGroup && hasStride) {
    return op->emitOpError("stride operand is only valid with group");
  }
  if (hasGroup && hasMask) {
    return op->emitOpError("group mode does not support mask operand");
  }
  if (hasGroup) {
    if (numGroups <= 0) {
      return op->emitOpError("group must be positive, got ") << numGroups;
    }
    if (nValues != 1) {
      return op->emitOpError("group mode requires exactly 1 value");
    }
    auto valueType = cast<VMIVRegType>(values[0].getType());
    if (failed(verifyNumGroups(op, valueType, numGroups))) {
      return failure();
    }
  }
  if (failed(verifyBlockStrideExclusions(op, hasBlock, distMode, hasGroup))) {
    return failure();
  }
  if (hasBlock && nValues != 1) {
    return op->emitOpError("block-stride mode requires exactly 1 value");
  }
  return success();
}

static LogicalResult verifyVStoreDistModeAndPmode(
    Operation *op, std::optional<StringRef> distMode, size_t nValues,
    size_t maskCount, std::optional<StringRef> pmode) {
  if (distMode && validStoreDistModes().find(*distMode) == validStoreDistModes().end()) {
    return op->emitOpError("invalid dist-mode: \"") << *distMode << "\"";
  }
  bool isIntlv = distMode && *distMode == "intlv";
  if (nValues < 1) {
    return op->emitOpError("requires at least 1 value");
  }
  if (isIntlv && nValues != mlir::pto::kValue2) {
    return op->emitOpError("dist-mode \"intlv\" requires exactly 2 values");
  }
  if (!isIntlv && nValues != 1) {
    return op->emitOpError("requires exactly 1 value for dist-mode \"")
           << (distMode ? *distMode : "continuous") << "\"";
  }
  if (maskCount > 1) {
    return op->emitOpError("at most one mask allowed");
  }
  if (pmode && validPModes().find(*pmode) == validPModes().end()) {
    return op->emitOpError("invalid pmode: \"") << *pmode << "\"";
  }
  if (pmode && *pmode != "zero") {
    return op->emitOpError("pmode \"merge\" is not supported for stores: the "
                           "legacy store lowering is mask-governed only and "
                           "cannot retain prior destination contents on inactive "
                           "lanes; omit pmode (defaults to \"zero\")");
  }
  return success();
}

static LogicalResult verifyVStoreValueMaskTypes(Operation *op, bool hasGroup,
                                                ValueRange values,
                                                Value destination,
                                                ValueRange mask) {
  auto valueType = cast<VMIVRegType>(values[0].getType());
  bool isPackedGroupStore =
      hasGroup &&
      isPackedByteGroupStore(destination.getType(), valueType);
  if (!isPackedGroupStore &&
      failed(verifyMemoryElementMatches(op, destination.getType(),
                                        valueType, "destination"))) {
    return failure();
  }
  if (values.size() == mlir::pto::kValue2) {
    auto loType = cast<VMIVRegType>(values[0].getType());
    auto hiType = cast<VMIVRegType>(values[1].getType());
    if (failed(verifyAllSameVRegShapeAndLayout(
            op, {loType, hiType}, /*requireSameElement=*/true))) {
      return failure();
    }
    if (failed(verifyContiguousIfLayoutAssigned(op, loType,
                                                "low input")) ||
        failed(verifyContiguousIfLayoutAssigned(op, hiType,
                                                "high input"))) {
      return failure();
    }
  }
  if (!mask.empty()) {
    auto maskType = cast<VMIMaskType>(mask[0].getType());
    if (failed(verifyMaskMatchesData(op, maskType, valueType))) {
      return failure();
    }
  }
  return success();
}

LogicalResult VMIvStoreOp::verify() {
  bool hasGroup = static_cast<bool>(getGroup());
  auto distMode = getDistMode();
  bool hasStride = static_cast<bool>(getStride());
  bool hasBlock = static_cast<bool>(getBlockStride());
  bool hasMask = !getMask().empty();
  size_t nValues = getValues().size();
  if (failed(verifyVStoreGroupAndBlockModes(
          getOperation(), hasGroup, distMode, hasStride, hasBlock, hasMask,
          hasGroup ? getGroupAttr().getInt() : 0, nValues, getValues()))) {
    return failure();
  }
  if (failed(verifyVStoreDistModeAndPmode(getOperation(), distMode, nValues,
                                         getMask().size(), getPmode()))) {
    return failure();
  }
  return verifyVStoreValueMaskTypes(getOperation(), hasGroup, getValues(),
                                   getDestination(), getMask());
}

void VMIvStoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VMIVsstbOp::verify() {
  auto valueType = cast<VMIVRegType>(getValue().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());
  if (failed(verifyMemoryElementMatches(getOperation(),
                                        getDestination().getType(), valueType,
                                        "destination")) ||
      failed(verifyUBBackedMemory(getOperation(), getDestination().getType(),
                                  "destination"))) {
    return failure();
  }
  if (auto pmode = getPmode(); pmode && validPModes().find(*pmode) == validPModes().end()) {
    return emitOpError("invalid pmode: \"") << *pmode << "\"";
  }
  if (auto pmode = getPmode(); pmode && *pmode != "zero") {
    return emitOpError("pmode \"merge\" is not supported for stores: the "
                       "legacy store lowering is mask-governed only and "
                       "cannot retain prior destination contents on inactive "
                       "blocks; omit pmode (defaults to \"zero\")");
  }
  return verifyMaskMatchesData(getOperation(), maskType, valueType);
}

void VMIVsstbOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

//===----------------------------------------------------------------------===//
// VMICvtOp — unified elementwise type conversion
//===----------------------------------------------------------------------===//
// VMIvLoadOp
static ParseResult parseVLoadOptionalPostOperand(
    OpAsmParser &parser, int &numPostBracket,
    OpAsmParser::UnresolvedOperand &postOp1) {
  // Optional comma-separated post-bracket operands.
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(postOp1)) {
      return failure();
    }
    numPostBracket = 1;
  }
  return success();
}

static void disambiguateVLoadPostOperands(
    OperationState &result, int numPostBracket,
    OpAsmParser::UnresolvedOperand postOp1,
    OpAsmParser::UnresolvedOperand &strideOperand,
    OpAsmParser::UnresolvedOperand &blockStrideOperand, bool &hasStride,
    bool &hasBlock) {
  // Disambiguate post-bracket operands
  // 1 operand + group attr → stride; otherwise → block_stride.
  if (numPostBracket == 1) {
    if (result.attributes.get("group")) {
      hasStride = true;
      strideOperand = postOp1;
    } else {
      hasBlock = true;
      blockStrideOperand = postOp1;
    }
  }
}

static ParseResult resolveVLoadOperandsAndFinalize(
    OpAsmParser &parser, OperationState &result,
    OpAsmParser::UnresolvedOperand sourceOperand,
    OpAsmParser::UnresolvedOperand offsetOperand,
    OpAsmParser::UnresolvedOperand strideOperand,
    OpAsmParser::UnresolvedOperand blockStrideOperand, bool hasStride,
    bool hasBlock, Type sourceType, ArrayRef<Type> resultTypes) {
  if (parser.resolveOperand(sourceOperand, sourceType, result.operands)) {
    return failure();
  }
  if (parser.resolveOperand(offsetOperand, parser.getBuilder().getIndexType(),
                            result.operands)) {
    return failure();
  }
  if (hasStride &&
      parser.resolveOperand(strideOperand, parser.getBuilder().getIndexType(),
                            result.operands)) {
    return failure();
  }
  if (hasBlock &&
      parser.resolveOperand(blockStrideOperand,
                            parser.getBuilder().getIntegerType(mlir::pto::kValue16),
                            result.operands)) {
    return failure();
  }
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {1, 1, hasStride ? 1 : 0, hasBlock ? 1 : 0}));
  result.addTypes(resultTypes);
  return success();
}

ParseResult VMIvLoadOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand sourceOperand;
  OpAsmParser::UnresolvedOperand offsetOperand;
  OpAsmParser::UnresolvedOperand strideOperand;
  OpAsmParser::UnresolvedOperand blockStrideOperand;
  // Parse: %source[%offset]
  if (parser.parseOperand(sourceOperand) || parser.parseLSquare() ||
      parser.parseOperand(offsetOperand) || parser.parseRSquare()) {
    return failure();
  }
  int numPostBracket = 0;
  OpAsmParser::UnresolvedOperand postOp1;
  if (failed(parseVLoadOptionalPostOperand(parser, numPostBracket,
                                           postOp1))) {
    return failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }
  Type sourceType;
  if (parser.parseColonType(sourceType)) {
    return failure();
  }
  if (parser.parseArrow()) {
    return failure();
  }
  SmallVector<Type, mlir::pto::kValue2> resultTypes;
  if (parser.parseTypeList(resultTypes)) {
    return failure();
  }
  bool hasStride = false;
  bool hasBlock = false;
  disambiguateVLoadPostOperands(result, numPostBracket, postOp1, strideOperand,
                                blockStrideOperand, hasStride, hasBlock);
  return resolveVLoadOperandsAndFinalize(
      parser, result, sourceOperand, offsetOperand, strideOperand,
      blockStrideOperand, hasStride, hasBlock, sourceType, resultTypes);
}

void VMIvLoadOp::print(OpAsmPrinter &p) {
  p << ' ' << getSource() << '[';
  p.printOperand(getOffset());
  p << ']';
  printVStrideBlockStride(p, getStride(), getBlockStride());
  p.printOptionalAttrDict((*this)->getAttrs(), {"operandSegmentSizes"});
  p << " : " << getSource().getType() << " -> " << getResults().getTypes();
}

static LogicalResult verifyVLoadGroupAndBlockModes(
    Operation *op, bool hasGroup, std::optional<StringRef> distMode,
    bool hasStride, bool hasBlock, int64_t numGroups, size_t nResults) {
  if (hasGroup && distMode && *distMode != "brc") {
    return op->emitOpError("group and dist_mode are mutually exclusive");
  }
  if (hasGroup && !hasStride) {
    return op->emitOpError("group requires a stride operand");
  }
  if (!hasGroup && hasStride) {
    return op->emitOpError("stride operand is only valid with group");
  }
  if (hasGroup) {
    if (numGroups <= 0) {
      return op->emitOpError("group must be positive, got ") << numGroups;
    }
    if (nResults != 1) {
      return op->emitOpError("group mode requires exactly 1 result");
    }
  }
  if (failed(verifyBlockStrideExclusions(op, hasBlock, distMode, hasGroup))) {
    return failure();
  }
  if (hasBlock && nResults != 1) {
    return op->emitOpError("block-stride mode requires exactly 1 result");
  }
  return success();
}

static LogicalResult verifyVLoadDistModeAndPmode(
    Operation *op, std::optional<StringRef> distMode, size_t nResults,
    std::optional<StringRef> pmode) {
  if (distMode && validDistModes().find(*distMode) == validDistModes().end()) {
    return op->emitOpError("invalid dist-mode: \"") << *distMode << "\"";
  }
  bool isDintlv = distMode && *distMode == "dintlv";
  if (isDintlv && nResults != mlir::pto::kValue2) {
    return op->emitOpError("dist-mode \"dintlv\" requires exactly 2 results");
  }
  if (!isDintlv && nResults != 1) {
    return op->emitOpError("requires exactly 1 result for dist-mode \"")
           << (distMode ? *distMode : "continuous") << "\"";
  }
  if (pmode && validPModes().find(*pmode) == validPModes().end()) {
    return op->emitOpError("invalid pmode: \"") << *pmode << "\"";
  }
  return success();
}

static LogicalResult verifyVLoadModeAndResultCounts(
    Operation *op, bool hasGroup, std::optional<StringRef> distMode,
    bool hasStride, bool hasBlock, int64_t numGroups, size_t nResults,
    std::optional<StringRef> pmode) {
  if (failed(verifyVLoadGroupAndBlockModes(op, hasGroup, distMode, hasStride,
                                           hasBlock, numGroups, nResults))) {
    return failure();
  }
  return verifyVLoadDistModeAndPmode(op, distMode, nResults, pmode);
}

static LogicalResult verifyVLoadResultTypes(
    Operation *op, ValueRange results, Type sourceType,
    std::optional<StringRef> distMode) {
  bool isDintlv = distMode && *distMode == "dintlv";
  for (auto res : results) {
    auto resType = cast<VMIVRegType>(res.getType());
    if (failed(verifyMemoryElementMatches(op, sourceType, resType,
                                          "source"))) {
      return failure();
    }
    if (isDintlv &&
        failed(verifyContiguousIfLayoutAssigned(op, resType, "result"))) {
      return failure();
    }
  }
  return success();
}

LogicalResult VMIvLoadOp::verify() {
  bool hasGroup = static_cast<bool>(getGroup());
  auto distMode = getDistMode();
  bool hasStride = static_cast<bool>(getStride());
  bool hasBlock = static_cast<bool>(getBlockStride());
  size_t nResults = getResults().size();
  if (failed(verifyVLoadModeAndResultCounts(
          getOperation(), hasGroup, distMode, hasStride, hasBlock,
          hasGroup ? getGroupAttr().getInt() : 0, nResults, getPmode()))) {
    return failure();
  }
  if (hasGroup) {
    auto resultType = cast<VMIVRegType>(getResults()[0].getType());
    if (failed(verifyNumGroups(getOperation(), resultType,
                               getGroupAttr().getInt()))) {
      return failure();
    }
  }
  return verifyVLoadResultTypes(getOperation(), getResults(),
                                getSource().getType(), distMode);
}

void VMIvLoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

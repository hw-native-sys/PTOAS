// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTODma.cpp - VPTO DMA ops -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VPTOInternal.h"

using namespace mlir;
using namespace mlir::pto;

bool isSupportedMovPadScalarType(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    return intType.isSignless() &&
           (intType.getWidth() == mlir::pto::kValue8 ||
            intType.getWidth() == mlir::pto::kValue16 ||
            intType.getWidth() == mlir::pto::kValue32);
  }
  if (auto floatType = dyn_cast<FloatType>(type)) {
    return floatType.isF16() || floatType.isBF16() || floatType.isF32();
  }
  return false;
}

static std::optional<unsigned> getDistElementWidth(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    return intType.getWidth();
  }
  if (type.isF16() || type.isBF16()) {
    return mlir::pto::kValue16;
  }
  if (type.isF32()) {
    return mlir::pto::kValue32;
  }
  if (type.isF64()) {
    return mlir::pto::kValue64;
  }
  return std::nullopt;
}

static bool isSupportedVldx2DistToken(StringRef dist, Type elementType) {
  return lookupVPTOMemoryDist(VPTOMemoryOpFamily::LoadX2, dist,
                              getDistElementWidth(elementType));
}

static bool isSupportedVldsDistToken(StringRef dist, Type elementType) {
  return lookupVPTOMemoryDist(VPTOMemoryOpFamily::Load, dist,
                              getDistElementWidth(elementType));
}

static bool isSupportedVstsDistToken(StringRef dist) {
  return lookupVPTOMemoryDist(VPTOMemoryOpFamily::Store, dist);
}

static bool isSupportedVstsx2DistToken(StringRef dist) {
  return lookupVPTOMemoryDist(VPTOMemoryOpFamily::StoreX2, dist);
}

static std::optional<StringRef>
getVstsMaskGranularityOverride(StringRef dist) {
  const auto *contract =
      lookupVPTOMemoryDist(VPTOMemoryOpFamily::Store, dist);
  if (!contract || contract->maskGranularity.empty()) {
    return std::nullopt;
  }
  return contract->maskGranularity;
}

// Shared kValue3 operand loop for the DMA triple-group parsers.
static ParseResult parseDmaTripleOperandLoop(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands) {
  for (int i = 0; i < mlir::pto::kValue3; ++i) {
    OpAsmParser::UnresolvedOperand operand;
    if (parser.parseOperand(operand)) {
      return failure();
    }
    operands.push_back(operand);
    if (i != mlir::pto::kValue2 && parser.parseComma()) {
      return failure();
    }
  }
  return success();
}

ParseResult parseDmaTripleGroup(
    OpAsmParser &parser, StringRef keyword,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands) {
  if (parser.parseKeyword(keyword) || parser.parseLParen()) {
    return failure();
  }
  if (failed(parseDmaTripleOperandLoop(parser, operands))) {
    return failure();
  }
  return parser.parseRParen();
}

ParseResult parseOptionalDmaTripleGroupAlias(
    OpAsmParser &parser, ArrayRef<StringRef> keywords,
    StringRef &parsedKeyword,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands) {
  parsedKeyword = {};
  for (StringRef keyword : keywords) {
    if (failed(parser.parseOptionalKeyword(keyword))) {
      continue;
    }
    parsedKeyword = keyword;
    if (parser.parseLParen()) {
      return failure();
    }
    if (failed(parseDmaTripleOperandLoop(parser, operands))) {
      return failure();
    }
    return parser.parseRParen();
  }
  return success();
}

static bool isDmaLoopKeyword(StringRef keyword) {
  if (keyword == "loop") {
    return true;
  }
  if (!keyword.consume_front("loop")) {
    return false;
  }
  if (keyword.empty()) {
    return false;
  }
  return llvm::all_of(keyword, llvm::isDigit);
}

ParseResult parseDmaTripleTypes(OpAsmParser &parser,
                                       SmallVectorImpl<Type> &types) {
  for (int i = 0; i < mlir::pto::kValue3; ++i) {
    Type type;
    if (parser.parseType(type)) {
      return failure();
    }
    types.push_back(type);
    if (i != mlir::pto::kValue2 && parser.parseComma()) {
      return failure();
    }
  }
  return success();
}

static ParseResult parseDmaPadTypes(OpAsmParser &parser,
                                    SmallVectorImpl<Type> &types) {
  Type valueType;
  if (parser.parseType(valueType)) {
    return failure();
  }
  types.push_back(valueType);
  if (succeeded(parser.parseOptionalComma())) {
    Type leftType;
    Type rightType;
    if (parser.parseType(leftType) || parser.parseComma() ||
        parser.parseType(rightType)) {
      return failure();
    }
    types.push_back(leftType);
    types.push_back(rightType);
  }
  return success();
}

ParseResult parseDmaPadOperandGroup(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &padOperands) {
  if (failed(parser.parseOptionalKeyword("pad"))) {
    return success();
  }
  if (parser.parseLParen()) {
    return failure();
  }
  OpAsmParser::UnresolvedOperand value;
  if (parser.parseOperand(value)) {
    return failure();
  }
  padOperands.push_back(value);
  if (succeeded(parser.parseOptionalComma())) {
    OpAsmParser::UnresolvedOperand left;
    OpAsmParser::UnresolvedOperand right;
    if (parser.parseOperand(left) || parser.parseComma() ||
        parser.parseOperand(right)) {
      return failure();
    }
    padOperands.push_back(left);
    padOperands.push_back(right);
  }
  if (parser.parseRParen()) {
    return failure();
  }
  return success();
}

ParseResult parseDmaLoopOperandGroups(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loopCountOperands,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loopSrcStrideOperands,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loopDstStrideOperands) {
  bool hasMore = true;
  while (hasMore) {
    StringRef parsedKeyword;
    SmallVector<OpAsmParser::UnresolvedOperand, mlir::pto::kValue3> loopGroupOperands;
    if (parseOptionalDmaTripleGroupAlias(parser, {"loop", "loop1", "loop2"},
                                         parsedKeyword, loopGroupOperands)) {
      return failure();
    }
    if (parsedKeyword.empty()) {
      hasMore = false;
      continue;
    }
    loopCountOperands.push_back(loopGroupOperands[0]);
    loopSrcStrideOperands.push_back(loopGroupOperands[1]);
    loopDstStrideOperands.push_back(loopGroupOperands[mlir::pto::kValue2]);
  }
  return success();
}

ParseResult parseDmaLoopTypeGroups(
    OpAsmParser &parser, SmallVectorImpl<Type> &loopCountTypes,
    SmallVectorImpl<Type> &loopSrcStrideTypes,
    SmallVectorImpl<Type> &loopDstStrideTypes) {
  while (succeeded(parser.parseOptionalComma())) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword)) {
      return failure();
    }
    if (!isDmaLoopKeyword(keyword)) {
      return parser.emitError(parser.getCurrentLocation(), "expected 'loop'");
    }
    SmallVector<Type> loopGroupTypes;
    if (parseDmaTripleTypes(parser, loopGroupTypes)) {
      return failure();
    }
    loopCountTypes.push_back(loopGroupTypes[0]);
    loopSrcStrideTypes.push_back(loopGroupTypes[1]);
    loopDstStrideTypes.push_back(loopGroupTypes[mlir::pto::kValue2]);
  }
  return success();
}

ParseResult parseDmaLoopAndPadTypeGroups(
    OpAsmParser &parser, SmallVectorImpl<Type> &loopCountTypes,
    SmallVectorImpl<Type> &loopSrcStrideTypes,
    SmallVectorImpl<Type> &loopDstStrideTypes,
    SmallVectorImpl<Type> &padTypes) {
  while (succeeded(parser.parseOptionalComma())) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword)) {
      return failure();
    }
    if (isDmaLoopKeyword(keyword)) {
      SmallVector<Type> loopGroupTypes;
      if (parseDmaTripleTypes(parser, loopGroupTypes)) {
        return failure();
      }
      loopCountTypes.push_back(loopGroupTypes[0]);
      loopSrcStrideTypes.push_back(loopGroupTypes[1]);
      loopDstStrideTypes.push_back(loopGroupTypes[mlir::pto::kValue2]);
      continue;
    }
    if (keyword == "pad") {
      if (!padTypes.empty() || parseDmaPadTypes(parser, padTypes)) {
        return failure();
      }
      continue;
    }
    return parser.emitError(parser.getCurrentLocation(),
                            "expected one of 'loop' or 'pad'");
  }
  return success();
}

ParseResult verifyDmaLoopGroupConsistency(
    OpAsmParser &parser, size_t countOperands, size_t srcStrideOperands,
    size_t dstStrideOperands, size_t countTypes, size_t srcStrideTypes,
    size_t dstStrideTypes) {
  if (countOperands != srcStrideOperands || countOperands != dstStrideOperands ||
      countTypes != srcStrideTypes || countTypes != dstStrideTypes) {
    return parser.emitError(parser.getCurrentLocation(),
                            "requires each loop group to provide count, src stride, and dst stride");
  }
  if (countOperands != countTypes) {
    return parser.emitError(parser.getCurrentLocation(),
                            "requires loop operand and type groups to match");
  }
  return success();
}

ParseResult resolveDmaBasicOperands(
    OpAsmParser &parser, OperationState &result,
    OpAsmParser::UnresolvedOperand source, Type sourceType,
    OpAsmParser::UnresolvedOperand destination, Type destinationType,
    OpAsmParser::UnresolvedOperand lenBurst, Type lenBurstType,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &nburstOperands,
    SmallVectorImpl<Type> &nburstTypes) {
  if (parser.resolveOperand(source, sourceType, result.operands) ||
      parser.resolveOperand(destination, destinationType, result.operands) ||
      parser.resolveOperand(lenBurst, lenBurstType, result.operands) ||
      parser.resolveOperands(nburstOperands, nburstTypes,
                             parser.getCurrentLocation(), result.operands)) {
    return failure();
  }
  return success();
}

ParseResult resolveDmaLoopOperands(
    OpAsmParser &parser, OperationState &result,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loopCountOperands,
    SmallVectorImpl<Type> &loopCountTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loopSrcStrideOperands,
    SmallVectorImpl<Type> &loopSrcStrideTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loopDstStrideOperands,
    SmallVectorImpl<Type> &loopDstStrideTypes) {
  auto loc = parser.getCurrentLocation();
  if (parser.resolveOperands(loopCountOperands, loopCountTypes, loc,
                             result.operands) ||
      parser.resolveOperands(loopSrcStrideOperands, loopSrcStrideTypes, loc,
                             result.operands) ||
      parser.resolveOperands(loopDstStrideOperands, loopDstStrideTypes, loc,
                             result.operands)) {
    return failure();
  }
  return success();
}

// Unified resolver for the DMA triple-group ops; covers the plain loop ops
// (hasL2CacheCtl=false) and the MteUbGm ops that interleave l2_cache_ctl
// between the basic operands and the loop groups (resolve order is kept).
ParseResult resolveDmaTripleOperands(
    OpAsmParser &parser, OperationState &result, bool hasL2CacheCtl,
    OpAsmParser::UnresolvedOperand l2CacheCtl, Type l2CacheCtlType,
    OpAsmParser::UnresolvedOperand source, Type sourceType,
    OpAsmParser::UnresolvedOperand destination, Type destinationType,
    OpAsmParser::UnresolvedOperand lenBurst, Type lenBurstType,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &nburstOperands,
    SmallVectorImpl<Type> &nburstTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loopCountOperands,
    SmallVectorImpl<Type> &loopCountTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loopSrcStrideOperands,
    SmallVectorImpl<Type> &loopSrcStrideTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loopDstStrideOperands,
    SmallVectorImpl<Type> &loopDstStrideTypes) {
  if (failed(resolveDmaBasicOperands(parser, result, source, sourceType,
                                     destination, destinationType, lenBurst,
                                     lenBurstType, nburstOperands,
                                     nburstTypes))) {
    return failure();
  }
  if (hasL2CacheCtl &&
      parser.resolveOperand(l2CacheCtl, l2CacheCtlType, result.operands)) {
    return failure();
  }
  return resolveDmaLoopOperands(parser, result, loopCountOperands,
                                loopCountTypes, loopSrcStrideOperands,
                                loopSrcStrideTypes, loopDstStrideOperands,
                                loopDstStrideTypes);
}

void printDmaTripleGroup(OpAsmPrinter &printer, StringRef keyword,
                                Value first, Value second, Value third) {
  printer << " " << keyword << "(" << first << ", " << second << ", " << third
          << ")";
}

void printDmaTripleTypes(OpAsmPrinter &printer, StringRef keyword,
                                Type first, Type second, Type third) {
  printer << ", " << keyword << " " << first << ", " << second << ", " << third;
}

void printDmaPadGroup(OpAsmPrinter &printer, Value value, Value left,
                             Value right) {
  printer << " pad(" << value;
  if (left || right) {
    printer << ", " << left << ", " << right;
  }
  printer << ")";
}

void printDmaPadTypes(OpAsmPrinter &printer, Type valueType,
                             Type leftType, Type rightType) {
  printer << ", pad " << valueType;
  if (leftType || rightType) {
    printer << ", " << leftType << ", " << rightType;
  }
}

ParseResult parseFixedKeywordOperandGroup(
    OpAsmParser &parser, StringRef keyword, int operandCount,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands) {
  if (parser.parseKeyword(keyword) || parser.parseLParen()) {
    return failure();
  }
  for (int i = 0; i < operandCount; ++i) {
    OpAsmParser::UnresolvedOperand operand;
    if (parser.parseOperand(operand)) {
      return failure();
    }
    operands.push_back(operand);
    if (i + 1 != operandCount && parser.parseComma()) {
      return failure();
    }
  }
  return parser.parseRParen();
}

ParseResult parseFixedKeywordTypes(OpAsmParser &parser, StringRef keyword,
                                          int typeCount,
                                          SmallVectorImpl<Type> &types) {
  if (parser.parseKeyword(keyword)) {
    return failure();
  }
  for (int i = 0; i < typeCount; ++i) {
    Type type;
    if (parser.parseType(type)) {
      return failure();
    }
    types.push_back(type);
    if (i + 1 != typeCount && parser.parseComma()) {
      return failure();
    }
  }
  return success();
}

template <typename DmaOp>
[[maybe_unused]] static LogicalResult verifyOptionalDmaLoopGroup(DmaOp op, Value count,
                                                Value srcStride,
                                                Value dstStride,
                                                StringRef name) {
  bool hasAny = static_cast<bool>(count) || static_cast<bool>(srcStride) ||
                static_cast<bool>(dstStride);
  bool hasAll = static_cast<bool>(count) && static_cast<bool>(srcStride) &&
                static_cast<bool>(dstStride);
  if (hasAny && !hasAll) {
    return op.emitOpError() << "requires " << name
                            << " group to provide count, src stride, and dst stride together";
  }
  return success();
}

LogicalResult verifyDmaLoadStoreLoopGroups(Operation *op,
                                                  ValueRange loopCounts,
                                                  ValueRange loopSrcStrides,
                                                  ValueRange loopDstStrides) {
  if (loopCounts.size() != loopSrcStrides.size() ||
      loopCounts.size() != loopDstStrides.size()) {
    return op->emitOpError()
           << "requires each loop group to provide count, src stride, and dst stride together";
  }
  return success();
}

void CopyGmToUbufOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult CopyGmToUbufOp::verify() {
  return verifyCopyGmToUbufOp(*this, true);
}

LogicalResult SetMovPadValOp::verify() {
  Type valueType = getValue().getType();
  if (isSupportedMovPadScalarType(valueType)) {
    return success();
  }
  return emitOpError()
         << "expects i8/i16/i32 or f16/bf16/f32 scalar operand, but got "
         << valueType;
}

LogicalResult CopyUbufToUbufOp::verify() {
  if (!isBufferLike(getSource().getType()) || !isBufferLike(getDestination().getType())) {
    return emitOpError("requires pointer-like source and destination");
  }
  if (classifyMemoryRole(getSource().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getDestination().getType()) != MemoryRole::UB) {
    return emitOpError("requires UB-backed source and destination");
  }
  return success();
}

void CopyCbufToUbufOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult CopyCbufToUbufOp::verify() {
  return verifyCopyCbufToUbufLikeOp(*this);
}

void CopyUbufToCbufOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult CopyUbufToCbufOp::verify() {
  if (!isBufferLike(getSource().getType()) || !isBufferLike(getDestination().getType())) {
    return emitOpError("requires pointer-like source and destination");
  }
  if (classifyMemoryRole(getSource().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getDestination().getType()) != MemoryRole::Other) {
    return emitOpError("requires UB-backed source and CBUF-backed destination");
  }
  return success();
}

void VldsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

template <typename LoadOp>
static LogicalResult verifyVldsCommon(LoadOp op) {
  if (!isBufferLike(op.getSource().getType())) {
    return op.emitOpError("requires a pointer-like source");
  }

  if (failed(verifyVRegTypeLike(op, op.getResult().getType(), "result type"))) {
    return failure();
  }

  MemoryRole sourceRole = classifyMemoryRole(op.getSource().getType());
  if (sourceRole == MemoryRole::GM) {
    return op.emitOpError("requires a UB-backed source");
  }

  if (op.getDistAttr()) {
    StringRef dist = *op.getDist();
    Type elementType =
        cast<VRegType>(op.getResult().getType()).getElementType();
    if (!isSupportedVldsDistToken(dist, elementType)) {
      return op.emitOpError(
          "supports only NORM, BRC_B8/B16/B32, US_B8/B16, DS_B8/B16, "
          "UNPK_B8/B16/B32, BRC_BLK, E2B_B16/B32, UNPK4, SPLT4CHN, and "
          "SPLT2CHN_B8/B16 load distributions");
    }
  }

  return success();
}

LogicalResult VldsOp::verify() {
  if (failed(verifyVldsCommon(*this))) {
    return failure();
  }
  if (Value updatedBase = getUpdatedBase()) {
    if (updatedBase.getType() != getSource().getType()) {
      return emitOpError("requires updated base result to match base type");
    }
  }
  return success();
}

void Vldsx2Op::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult Vldsx2Op::verify() {
  if (!isBufferLike(getSource().getType())) {
    return emitOpError("requires a pointer-like source");
  }
  if (classifyMemoryRole(getSource().getType()) == MemoryRole::GM) {
    return emitOpError("requires a UB-backed source");
  }
  if (!getOffset().getType().isIndex()) {
    return emitOpError("requires index offset");
  }
  if (failed(verifyVRegTypeLike(*this, getLow().getType(), "low result type")) ||
      failed(verifyVRegTypeLike(*this, getHigh().getType(), "high result type"))) {
    return failure();
  }
  if (getLow().getType() != getHigh().getType()) {
    return emitOpError("requires low/high results to share one vector type");
  }
  Type elementType = cast<VRegType>(getLow().getType()).getElementType();
  if (!isSupportedVldx2DistToken(getDist(), elementType)) {
    return emitOpError("requires a supported x2 load distribution token");
  }
  if (getUpdatedBase() &&
      getUpdatedBase().getType() != getSource().getType()) {
    return emitOpError("requires updated base result to match base type");
  }
  return success();
}

void VstsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

template <typename StoreOp>
static LogicalResult verifyVstsCommon(StoreOp op) {
  if (failed(verifyVRegTypeLike(op, op.getValue().getType(), "value type"))) {
    return failure();
  }

  if (!isBufferLike(op.getDestination().getType())) {
    return op.emitOpError("requires a pointer-like destination");
  }

  MemoryRole destinationRole = classifyMemoryRole(op.getDestination().getType());
  if (destinationRole == MemoryRole::GM) {
    return op.emitOpError("requires a UB-backed destination");
  }

  if (std::optional<StringRef> dist = op.getDist();
      dist && !isSupportedVstsDistToken(*dist)) {
    return op.emitOpError("requires a supported store distribution token");
  }
  if (std::optional<StringRef> dist = op.getDist()) {
    if (std::optional<StringRef> granularity =
            getVstsMaskGranularityOverride(*dist)) {
      if (failed(verifyMaskTypeWithGranularityLike(op, op.getMask().getType(),
                                                   "mask type", *granularity))) {
        return failure();
      }
    } else if (failed(verifyMaskTypeLike(op, op.getMask().getType(),
                                         "mask type"))) {
      return failure();
    }
  } else if (failed(verifyMaskTypeLike(op, op.getMask().getType(),
                                       "mask type"))) {
    return failure();
  }

  return success();
}

LogicalResult VstsOp::verify() {
  if (failed(verifyVstsCommon(*this))) {
    return failure();
  }
  if (getUpdatedBase() &&
      getUpdatedBase().getType() != getDestination().getType()) {
    return emitOpError("requires updated base result to match base type");
  }
  return success();
}
void Vstsx2Op::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getLowMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getHighMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult Vstsx2Op::verify() {
  if (failed(verifyVRegTypeLike(*this, getLow().getType(), "low value type")) ||
      failed(verifyVRegTypeLike(*this, getHigh().getType(), "high value type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type"))) {
    return failure();
  }
  if (getLow().getType() != getHigh().getType()) {
    return emitOpError("requires low/high values to share one vector type");
  }
  if (!isBufferLike(getDestination().getType())) {
    return emitOpError("requires a pointer-like destination");
  }
  if (classifyMemoryRole(getDestination().getType()) == MemoryRole::GM) {
    return emitOpError("requires a UB-backed destination");
  }
  if (!getOffset().getType().isIndex()) {
    return emitOpError("requires index offset");
  }
  if (!isSupportedVstsx2DistToken(getDist())) {
    return emitOpError("requires a supported x2 store distribution token");
  }
  return success();
}

void CopyUbufToGmOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult CopyUbufToGmOp::verify() {
  return verifyCopyGmToUbufOp(*this, false);
}

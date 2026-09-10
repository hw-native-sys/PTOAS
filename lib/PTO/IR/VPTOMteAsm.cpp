// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOMteAsm.cpp - VPTO MTE assembly format helpers -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VPTOInternal.h"

using namespace mlir;
using namespace mlir::pto;

[[maybe_unused]] static ParseResult parseMteL0cL1OptionalLoop3(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loop3CountOperands,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loop3SrcStrideOperands,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loop3DstStrideOperands) {
  StringRef parsedKeyword;
  SmallVector<OpAsmParser::UnresolvedOperand, mlir::pto::kValue3> loop3Operands;
  if (parseOptionalDmaTripleGroupAlias(parser, {"loop3"}, parsedKeyword,
                                       loop3Operands)) {
    return failure();
  }
  if (!parsedKeyword.empty()) {
    loop3CountOperands.push_back(loop3Operands[0]);
    loop3SrcStrideOperands.push_back(loop3Operands[1]);
    loop3DstStrideOperands.push_back(loop3Operands[mlir::pto::kValue2]);
  }
  return success();
}

[[maybe_unused]] static ParseResult parseMteL0cL1OptionalFpc(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &fpcOperands) {
  if (failed(parser.parseOptionalKeyword("fpc"))) {
    return success();
  }
  if (parser.parseLParen()) {
    return failure();
  }
  OpAsmParser::UnresolvedOperand operand;
  if (parser.parseOperand(operand) || parser.parseRParen()) {
    return failure();
  }
  fpcOperands.push_back(operand);
  return success();
}

[[maybe_unused]] static void printMteL0cL1OptionalFpc(OpAsmPrinter &printer,
                                                      Value fpc) {
  if (fpc) {
    printer << ", fpc(" << fpc << ")";
  }
}

[[maybe_unused]] static void
printMteL0cL1OptionalFpcType(OpAsmPrinter &printer, Type fpcType) {
  if (fpcType) {
    printer << ", fpc(" << fpcType << ")";
  }
}

[[maybe_unused]] static ParseResult parseMteL0cL1OptionalLoop3Types(
    OpAsmParser &parser, SmallVectorImpl<Type> &loop3CountTypes,
    SmallVectorImpl<Type> &loop3SrcStrideTypes,
    SmallVectorImpl<Type> &loop3DstStrideTypes, StringRef opName) {
  if (succeeded(parser.parseOptionalComma())) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword)) {
      return failure();
    }
    if (keyword != "loop3") {
      return parser.emitError(parser.getCurrentLocation(), "expected 'loop3'");
    }
    SmallVector<Type> loop3GroupTypes;
    if (parseDmaTripleTypes(parser, loop3GroupTypes)) {
      return failure();
    }
    loop3CountTypes.push_back(loop3GroupTypes[0]);
    loop3SrcStrideTypes.push_back(loop3GroupTypes[1]);
    loop3DstStrideTypes.push_back(loop3GroupTypes[mlir::pto::kValue2]);
    if (succeeded(parser.parseOptionalComma())) {
      return parser.emitError(parser.getCurrentLocation(),
                              (Twine(opName) +
                               " accepts at most one loop3 group")
                                  .str());
    }
  }
  return success();
}

[[maybe_unused]] static ParseResult resolveStructuredMteL0cL1OptionalOperands(
    OpAsmParser &parser, StructuredAccStoreAsmState &state,
    SmallVectorImpl<Value> &resolvedOperands, OperationState &result) {
  auto location = parser.getCurrentLocation();
  if (parser.resolveOperands(state.preQuantOperands, state.preQuantTypes,
                             location, result.operands) ||
      parser.resolveOperands(state.preReluOperands, state.preReluTypes,
                             location, result.operands) ||
      parser.resolveOperands(state.clipValueOperands, state.clipValueTypes,
                             location, result.operands) ||
      parser.resolveOperands(state.splitOperands, state.splitTypes, location,
                             result.operands) ||
      parser.resolveOperands(state.loop0SrcStrideOperands,
                             state.loop0SrcStrideTypes, location,
                             result.operands) ||
      parser.resolveOperands(state.loop3CountOperands, state.loop3CountTypes,
                             location, result.operands) ||
      parser.resolveOperands(state.loop3SrcStrideOperands,
                             state.loop3SrcStrideTypes, location,
                             result.operands) ||
      parser.resolveOperands(state.loop3DstStrideOperands,
                             state.loop3DstStrideTypes, location,
                             result.operands)) {
    return failure();
  }

  auto extractResolved = [&](SmallVectorImpl<OpAsmParser::UnresolvedOperand> &ops,
                             SmallVectorImpl<Type> &types) -> Value {
    if (ops.empty()) {
      return {};
    }
    return result.operands[resolvedOperands.size()];
  };
  resolvedOperands.push_back(extractResolved(state.preQuantOperands,
                                             state.preQuantTypes));
  resolvedOperands.push_back(extractResolved(state.preReluOperands,
                                             state.preReluTypes));
  resolvedOperands.push_back(extractResolved(state.clipValueOperands,
                                             state.clipValueTypes));
  resolvedOperands.push_back(extractResolved(state.splitOperands,
                                             state.splitTypes));
  resolvedOperands.push_back(extractResolved(state.loop0SrcStrideOperands,
                                             state.loop0SrcStrideTypes));
  resolvedOperands.push_back(extractResolved(state.loop3CountOperands,
                                             state.loop3CountTypes));
  resolvedOperands.push_back(extractResolved(state.loop3SrcStrideOperands,
                                             state.loop3SrcStrideTypes));
  resolvedOperands.push_back(extractResolved(state.loop3DstStrideOperands,
                                             state.loop3DstStrideTypes));
  return success();
}

ParseResult parseMteGmUbBasicOperands(
    OpAsmParser &parser, OpAsmParser::UnresolvedOperand &source,
    OpAsmParser::UnresolvedOperand &destination,
    OpAsmParser::UnresolvedOperand &l2CacheCtl,
    OpAsmParser::UnresolvedOperand &lenBurst,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &nburstOperands) {
  if (parseRequiredOperandWithComma(parser, source) ||
      parseRequiredOperandWithComma(parser, destination) ||
      parseRequiredOperandWithComma(parser, l2CacheCtl) ||
      parser.parseOperand(lenBurst) ||
      parseDmaTripleGroup(parser, "nburst", nburstOperands)) {
    return failure();
  }
  return success();
}

ParseResult parseMteGmUbBasicTypes(
    OpAsmParser &parser, Type &sourceType, Type &destinationType,
    Type &l2CacheCtlType, Type &lenBurstType,
    SmallVectorImpl<Type> &nburstTypes) {
  if (parser.parseType(sourceType) || parser.parseComma() ||
      parser.parseType(destinationType) || parser.parseComma() ||
      parser.parseType(l2CacheCtlType) || parser.parseComma() ||
      parser.parseType(lenBurstType) || parser.parseComma() ||
      parseDmaTripleTypes(parser, nburstTypes)) {
    return failure();
  }
  return success();
}

void setMteGmUbSegmentSizes(OperationState &result,
                                    int32_t loopGroupCount,
                                    size_t padOperandCount) {
  auto &segments =
      result.getOrAddProperties<MteGmUbOp::Properties>().operandSegmentSizes;
  llvm::copy(ArrayRef<int32_t>{1, 1, 1, 1, 1, 1, 1,
                               loopGroupCount, loopGroupCount, loopGroupCount,
                               static_cast<int32_t>(padOperandCount ? 1 : 0),
                               static_cast<int32_t>(padOperandCount == 3 ? 1 : 0),
                               static_cast<int32_t>(padOperandCount == 3 ? 1 : 0)},
              segments.begin());
}

ParseResult resolveMteGmUbOperands(
    OpAsmParser &parser, OperationState &result,
    OpAsmParser::UnresolvedOperand source, Type sourceType,
    OpAsmParser::UnresolvedOperand destination, Type destinationType,
    OpAsmParser::UnresolvedOperand l2CacheCtl, Type l2CacheCtlType,
    OpAsmParser::UnresolvedOperand lenBurst, Type lenBurstType,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &nburstOperands,
    SmallVectorImpl<Type> &nburstTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loopCountOperands,
    SmallVectorImpl<Type> &loopCountTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loopSrcStrideOperands,
    SmallVectorImpl<Type> &loopSrcStrideTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loopDstStrideOperands,
    SmallVectorImpl<Type> &loopDstStrideTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &padOperands,
    SmallVectorImpl<Type> &padTypes) {
  auto loc = parser.getCurrentLocation();
  if (parser.resolveOperand(source, sourceType, result.operands) ||
      parser.resolveOperand(destination, destinationType, result.operands) ||
      parser.resolveOperand(l2CacheCtl, l2CacheCtlType, result.operands) ||
      parser.resolveOperand(lenBurst, lenBurstType, result.operands) ||
      parser.resolveOperands(nburstOperands, nburstTypes, loc,
                             result.operands) ||
      failed(resolveDmaLoopOperands(parser, result, loopCountOperands,
                                    loopCountTypes, loopSrcStrideOperands,
                                    loopSrcStrideTypes, loopDstStrideOperands,
                                    loopDstStrideTypes)) ||
      parser.resolveOperands(padOperands, padTypes, loc, result.operands)) {
    return failure();
  }
  return success();
}

ParseResult parseMteUbGmBasicOperands(
    OpAsmParser &parser, OpAsmParser::UnresolvedOperand &source,
    OpAsmParser::UnresolvedOperand &destination,
    OpAsmParser::UnresolvedOperand &lenBurst,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &nburstOperands) {
  if (parseRequiredOperandWithComma(parser, source) ||
      parseRequiredOperandWithComma(parser, destination) ||
      parser.parseOperand(lenBurst) ||
      parseDmaTripleGroup(parser, "nburst", nburstOperands)) {
    return failure();
  }
  return success();
}

ParseResult parseMteUbGmBasicTypes(
    OpAsmParser &parser, Type &sourceType, Type &destinationType,
    Type &lenBurstType, SmallVectorImpl<Type> &nburstTypes) {
  if (parser.parseType(sourceType) || parser.parseComma() ||
      parser.parseType(destinationType) || parser.parseComma() ||
      parser.parseType(lenBurstType) || parser.parseComma() ||
      parseDmaTripleTypes(parser, nburstTypes)) {
    return failure();
  }
  return success();
}

ParseResult parseMteUbGmL2CacheCtlOperand(
    OpAsmParser &parser, OpAsmParser::UnresolvedOperand &l2CacheCtl,
    bool &hasL2CacheCtl) {
  hasL2CacheCtl = succeeded(parser.parseOptionalKeyword("l2_cache_ctl"));
  if (!hasL2CacheCtl) {
    return success();
  }
  if (parser.parseLParen() || parser.parseOperand(l2CacheCtl) ||
      parser.parseRParen()) {
    return failure();
  }
  return success();
}

void setMteUbGmSegmentSizes(OperationState &result, bool hasL2CacheCtl,
                                    size_t loopGroupCount) {
  auto &segments =
      result.getOrAddProperties<MteUbGmOp::Properties>().operandSegmentSizes;
  llvm::copy(ArrayRef<int32_t>{1, 1, 1, 1, 1, 1,
                               hasL2CacheCtl ? 1 : 0,
                               static_cast<int32_t>(loopGroupCount),
                               static_cast<int32_t>(loopGroupCount),
                               static_cast<int32_t>(loopGroupCount)},
              segments.begin());
}

ParseResult parseMteGmL1FracBasicOperands(
    OpAsmParser &parser, OpAsmParser::UnresolvedOperand &source,
    OpAsmParser::UnresolvedOperand &destination, StringRef &modeKeyword,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &shapeOperands,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &srcLayoutOperands,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &dstGroupOperands,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &ctrlOperands) {
  if (parseRequiredOperandWithComma(parser, source) ||
      parseRequiredOperandWithComma(parser, destination) ||
      parser.parseKeyword(&modeKeyword) ||
      failed(parseCubeLoadFracModeKeyword(modeKeyword)) ||
      parser.parseComma() ||
      parseFixedKeywordOperandGroup(parser, "shape", mlir::pto::kValue2,
                                    shapeOperands) ||
      parser.parseComma() ||
      parseCubeLoadFracSrcLayoutGroup(parser, srcLayoutOperands) ||
      parser.parseComma() ||
      parseFixedKeywordOperandGroup(parser, "dst_group", mlir::pto::kValue4,
                                    dstGroupOperands) ||
      parser.parseComma() ||
      parseFixedKeywordOperandGroup(parser, "ctrl", mlir::pto::kValue2,
                                    ctrlOperands)) {
    return failure();
  }
  return success();
}

ParseResult parseMteGmL1FracBasicTypes(
    OpAsmParser &parser, Type &sourceType, Type &destinationType,
    StringRef modeKeyword, SmallVectorImpl<Type> &shapeTypes,
    SmallVectorImpl<Type> &srcLayoutTypes,
    SmallVectorImpl<Type> &dstGroupTypes,
    SmallVectorImpl<Type> &ctrlTypes) {
  if (parser.parseType(sourceType) || parser.parseComma() ||
      parser.parseType(destinationType) || parser.parseComma() ||
      parser.parseKeyword(modeKeyword) || parser.parseComma() ||
      parseFixedKeywordTypes(parser, "shape", mlir::pto::kValue2,
                            shapeTypes) ||
      parser.parseComma() ||
      parseCubeLoadFracSrcLayoutTypes(parser, srcLayoutTypes) ||
      parser.parseComma() ||
      parseFixedKeywordTypes(parser, "dst_group", mlir::pto::kValue4,
                            dstGroupTypes) ||
      parser.parseComma() ||
      parseFixedKeywordTypes(parser, "ctrl", mlir::pto::kValue2, ctrlTypes)) {
    return failure();
  }
  return success();
}

ParseResult validateMteGmL1FracOperands(
    OpAsmParser &parser, size_t shapeOps, size_t shapeTypes,
    size_t srcLayoutOps, size_t srcLayoutTypes,
    size_t dstGroupOps, size_t dstGroupTypes,
    size_t ctrlOps, size_t ctrlTypes) {
  if (shapeOps != 2 || shapeTypes != 2) {
    return parser.emitError(parser.getCurrentLocation(),
                            "shape requires exactly two operands and types");
  }
  if (srcLayoutOps == 0 || srcLayoutOps > mlir::pto::kValue2 ||
      srcLayoutTypes == 0 || srcLayoutTypes > mlir::pto::kValue2) {
    return parser.emitError(parser.getCurrentLocation(),
                            "src_layout requires one or two operands and types");
  }
  if (dstGroupOps != 4 || dstGroupTypes != 4) {
    return parser.emitError(parser.getCurrentLocation(),
                            "dst_group requires exactly four operands and types");
  }
  if (ctrlOps != 2 || ctrlTypes != 2) {
    return parser.emitError(parser.getCurrentLocation(),
                            "ctrl requires exactly two operands and types");
  }
  if (srcLayoutOps != srcLayoutTypes) {
    return parser.emitError(parser.getCurrentLocation(),
                            "src_layout operand and type groups must match");
  }
  return success();
}

ParseResult resolveMteGmL1FracOperands(
    OpAsmParser &parser, OperationState &result,
    OpAsmParser::UnresolvedOperand source, Type sourceType,
    OpAsmParser::UnresolvedOperand destination, Type destinationType,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &shapeOperands,
    SmallVectorImpl<Type> &shapeTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &srcLayoutOperands,
    SmallVectorImpl<Type> &srcLayoutTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &dstGroupOperands,
    SmallVectorImpl<Type> &dstGroupTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &ctrlOperands,
    SmallVectorImpl<Type> &ctrlTypes) {
  bool hasSrcOuterStride = srcLayoutOperands.size() == 2;
  SmallVector<Type> flatTypes;
  SmallVector<OpAsmParser::UnresolvedOperand> flatOperands;
  flatOperands.append({shapeOperands[0], shapeOperands[1], srcLayoutOperands[0]});
  flatTypes.append({shapeTypes[0], shapeTypes[1], srcLayoutTypes[0]});
  flatOperands.append(dstGroupOperands.begin(), dstGroupOperands.end());
  flatTypes.append(dstGroupTypes.begin(), dstGroupTypes.end());
  flatOperands.append(ctrlOperands.begin(), ctrlOperands.end());
  flatTypes.append(ctrlTypes.begin(), ctrlTypes.end());
  if (hasSrcOuterStride) {
    flatOperands.push_back(srcLayoutOperands[1]);
    flatTypes.push_back(srcLayoutTypes[1]);
  }
  if (parser.resolveOperand(source, sourceType, result.operands) ||
      parser.resolveOperand(destination, destinationType, result.operands) ||
      parser.resolveOperands(flatOperands, flatTypes,
                             parser.getCurrentLocation(), result.operands)) {
    return failure();
  }
  return success();
}

void setMteL0cL1SegmentSizes(OperationState &result,
                                        const StructuredAccStoreAsmState &st) {
  setStructuredAccStoreSegmentSizes<MteL0cL1Op>(
      result, {1, 1, 1, 1, 1, 1, !st.preQuantOperands.empty() ? 1 : 0,
               !st.preReluOperands.empty() ? 1 : 0,
               !st.clipValueOperands.empty() ? 1 : 0,
               !st.splitOperands.empty() ? 1 : 0,
               !st.loop0SrcStrideOperands.empty() ? 1 : 0,
               !st.loop3CountOperands.empty() ? 1 : 0,
               !st.loop3SrcStrideOperands.empty() ? 1 : 0,
               !st.loop3DstStrideOperands.empty() ? 1 : 0});
}

ParseResult parseMteL0cL1Types(
    OpAsmParser &parser, Type &sourceType, Type &destinationType,
    Type &mType, Type &nType, Type &srcStrideType, Type &dstStrideType,
    StructuredAccStoreAsmState &state) {
  if (parser.parseType(sourceType) || parser.parseComma() ||
      parser.parseType(destinationType) || parser.parseComma() ||
      parser.parseType(mType) || parser.parseComma() ||
      parser.parseType(nType) || parser.parseComma() ||
      parser.parseType(srcStrideType) || parser.parseComma() ||
      parser.parseType(dstStrideType) ||
      parseStructuredAccStoreTailTypes(parser, state)) {
    return failure();
  }
  return success();
}

// Batch7: MteL0c resolve 共用前缀与尾段
static ParseResult resolveMteL0cPrefix(
    OpAsmParser &parser, OperationState &result,
    OpAsmParser::UnresolvedOperand source, Type sourceType,
    OpAsmParser::UnresolvedOperand destination, Type destinationType,
    OpAsmParser::UnresolvedOperand m, Type mType,
    OpAsmParser::UnresolvedOperand n, Type nType,
    OpAsmParser::UnresolvedOperand srcStride, Type srcStrideType,
    OpAsmParser::UnresolvedOperand dstStride, Type dstStrideType,
    StructuredAccStoreAsmState &state) {
  auto loc = parser.getCurrentLocation();
  if (parser.resolveOperand(source, sourceType, result.operands) ||
      parser.resolveOperand(destination, destinationType, result.operands) ||
      parser.resolveOperand(m, mType, result.operands) ||
      parser.resolveOperand(n, nType, result.operands) ||
      parser.resolveOperand(srcStride, srcStrideType, result.operands) ||
      parser.resolveOperand(dstStride, dstStrideType, result.operands) ||
      parser.resolveOperands(state.preQuantOperands, state.preQuantTypes,
                             loc, result.operands) ||
      parser.resolveOperands(state.preReluOperands, state.preReluTypes,
                             loc, result.operands) ||
      parser.resolveOperands(state.clipValueOperands, state.clipValueTypes,
                             loc, result.operands)) {
    return failure();
  }
  return success();
}

static ParseResult resolveMteL0cTail(OpAsmParser &parser,
                                     OperationState &result,
                                     StructuredAccStoreAsmState &state) {
  auto loc = parser.getCurrentLocation();
  if (parser.resolveOperands(state.splitOperands, state.splitTypes,
                             loc, result.operands) ||
      parser.resolveOperands(state.loop0SrcStrideOperands,
                              state.loop0SrcStrideTypes, loc,
                              result.operands) ||
      parser.resolveOperands(state.loop3CountOperands,
                              state.loop3CountTypes, loc,
                              result.operands) ||
      parser.resolveOperands(state.loop3SrcStrideOperands,
                              state.loop3SrcStrideTypes, loc,
                              result.operands) ||
      parser.resolveOperands(state.loop3DstStrideOperands,
                              state.loop3DstStrideTypes, loc,
                              result.operands)) {
    return failure();
  }
  return success();
}

ParseResult resolveMteL0cL1Operands(
    OpAsmParser &parser, OperationState &result,
    OpAsmParser::UnresolvedOperand source, Type sourceType,
    OpAsmParser::UnresolvedOperand destination, Type destinationType,
    OpAsmParser::UnresolvedOperand m, Type mType,
    OpAsmParser::UnresolvedOperand n, Type nType,
    OpAsmParser::UnresolvedOperand srcStride, Type srcStrideType,
    OpAsmParser::UnresolvedOperand dstStride, Type dstStrideType,
    StructuredAccStoreAsmState &state) {
  if (failed(resolveMteL0cPrefix(parser, result, source, sourceType,
                                 destination, destinationType, m, mType, n,
                                 nType, srcStride, srcStrideType, dstStride,
                                 dstStrideType, state)) ||
      failed(resolveMteL0cTail(parser, result, state))) {
    return failure();
  }
  return success();
}

void printMteL1L0OptionalOperandsOp(
    OpAsmPrinter &printer, Operation *operation, Value source, Value destination,
    ArrayRef<Value> shapeOperands, ArrayRef<StringRef> shapeNames,
    ArrayRef<Value> fullOperands, ArrayRef<StringRef> fullNames) {

  const bool hasShape = llvm::any_of(shapeOperands, [](Value value) {
    return static_cast<bool>(value);
  });
  const bool hasFull = llvm::any_of(fullOperands, [](Value value) {
    return static_cast<bool>(value);
  });
  const bool isShapeForm = hasShape && !hasFull &&
      llvm::all_of(shapeOperands, [](Value value) { return static_cast<bool>(value); });
  const bool isFullForm = hasFull && !hasShape &&
      llvm::all_of(fullOperands, [](Value value) { return static_cast<bool>(value); });

  printer << " " << source << ", " << destination;
  SmallVector<Value, mlir::pto::kValue10> printedOperands;
  if (isShapeForm) {
    for (Value value : shapeOperands) {
      printer << ", " << value;
      printedOperands.push_back(value);
    }
  } else if (isFullForm) {
    for (Value value : fullOperands) {
      printer << ", " << value;
      printedOperands.push_back(value);
    }
  } else {
    for (auto [index, value] : llvm::enumerate(shapeOperands)) {
      if (!value) {
        continue;
      }
      printer << ", " << shapeNames[index] << "(" << value << ")";
      printedOperands.push_back(value);
    }
    for (auto [index, value] : llvm::enumerate(fullOperands)) {
      if (!value) {
        continue;
      }
      printer << ", " << fullNames[index] << "(" << value << ")";
      printedOperands.push_back(value);
    }
  }

  printer.printOptionalAttrDict(operation->getAttrs(),
                                /*elidedAttrs=*/{"operandSegmentSizes"});
  printer << " : " << source.getType() << ", " << destination.getType();
  for (Value value : printedOperands) {
    printer << ", " << value.getType();
  }
}
void setMteL0cGmSegmentSizes(OperationState &result,
                                        const StructuredAccStoreAsmState &st) {
  setStructuredAccStoreSegmentSizes<MteL0cGmOp>(
      result, {1, 1, 1, 1, 1, 1, !st.preQuantOperands.empty() ? 1 : 0,
               !st.preReluOperands.empty() ? 1 : 0,
               !st.clipValueOperands.empty() ? 1 : 0, 1, 1,
               !st.splitOperands.empty() ? 1 : 0,
               !st.loop0SrcStrideOperands.empty() ? 1 : 0,
               !st.loop3CountOperands.empty() ? 1 : 0,
               !st.loop3SrcStrideOperands.empty() ? 1 : 0,
               !st.loop3DstStrideOperands.empty() ? 1 : 0});
}

ParseResult parseMteL0cGmTypes(
    OpAsmParser &parser, Type &sourceType, Type &destinationType,
    Type &mType, Type &nType, Type &srcStrideType, Type &dstStrideType,
    Type &sidType, Type &l2CacheCtrlType,
    StructuredAccStoreAsmState &state) {
  if (parser.parseType(sourceType) || parser.parseComma() ||
      parser.parseType(destinationType) || parser.parseComma() ||
      parser.parseType(mType) || parser.parseComma() ||
      parser.parseType(nType) || parser.parseComma() ||
      parser.parseType(srcStrideType) || parser.parseComma() ||
      parser.parseType(dstStrideType) || parser.parseComma() ||
      parser.parseType(sidType) || parser.parseComma() ||
      parser.parseType(l2CacheCtrlType) ||
      parseStructuredAccStoreTailTypes(parser, state)) {
    return failure();
  }
  return success();
}

ParseResult resolveMteL0cGmOperands(
    OpAsmParser &parser, OperationState &result,
    OpAsmParser::UnresolvedOperand source, Type sourceType,
    OpAsmParser::UnresolvedOperand destination, Type destinationType,
    OpAsmParser::UnresolvedOperand m, Type mType,
    OpAsmParser::UnresolvedOperand n, Type nType,
    OpAsmParser::UnresolvedOperand srcStride, Type srcStrideType,
    OpAsmParser::UnresolvedOperand dstStride, Type dstStrideType,
    OpAsmParser::UnresolvedOperand sid, Type sidType,
    OpAsmParser::UnresolvedOperand l2CacheCtrl, Type l2CacheCtrlType,
    StructuredAccStoreAsmState &state) {
  if (failed(resolveMteL0cPrefix(parser, result, source, sourceType,
                                 destination, destinationType, m, mType, n,
                                 nType, srcStride, srcStrideType, dstStride,
                                 dstStrideType, state)) ||
      parser.resolveOperand(sid, sidType, result.operands) ||
      parser.resolveOperand(l2CacheCtrl, l2CacheCtrlType, result.operands) ||
      failed(resolveMteL0cTail(parser, result, state))) {
    return failure();
  }
  return success();
}

ParseResult parseMteL0cUbBasicOperands(
    OpAsmParser &parser, OpAsmParser::UnresolvedOperand &source,
    OpAsmParser::UnresolvedOperand &destination,
    OpAsmParser::UnresolvedOperand &m, OpAsmParser::UnresolvedOperand &n,
    OpAsmParser::UnresolvedOperand &srcStride,
    OpAsmParser::UnresolvedOperand &dstStride) {
  if (parseRequiredOperandWithComma(parser, source) ||
      parseRequiredOperandWithComma(parser, destination) ||
      parseRequiredOperandWithComma(parser, m) ||
      parseRequiredOperandWithComma(parser, n) ||
      parseRequiredOperandWithComma(parser, srcStride) ||
      parseRequiredOperandWithComma(parser, dstStride)) {
    return failure();
  }
  return success();
}

ParseResult parseMteL0cUbTypes(
    OpAsmParser &parser, Type &sourceType, Type &destinationType, Type &mType,
    Type &nType, Type &srcStrideType, Type &dstStrideType, bool hasSubBlockId,
    Type &subBlockIdType, StructuredAccStoreAsmState &state) {
  if (parser.parseType(sourceType) || parser.parseComma() ||
      parser.parseType(destinationType) || parser.parseComma() ||
      parser.parseType(mType) || parser.parseComma() || parser.parseType(nType) ||
      parser.parseComma() || parser.parseType(srcStrideType) ||
      parser.parseComma() || parser.parseType(dstStrideType)) {
    return failure();
  }
  if (hasSubBlockId &&
      (parser.parseComma() || parser.parseType(subBlockIdType))) {
    return failure();
  }
  if (parseStructuredAccStoreTailTypes(parser, state)) {
    return failure();
  }
  return success();
}

ParseResult parseMteL0cUbDstMode(OpAsmParser &parser,
                                        AccStoreUbDstMode &dstMode,
                                        OpAsmParser::UnresolvedOperand &subBlockId,
                                        bool &hasSubBlockId) {
  if (parser.parseKeyword("dst_mode") || parser.parseLParen()) {
    return failure();
  }
  OptionalParseResult subBlockIdParse = parser.parseOptionalOperand(subBlockId);
  if (subBlockIdParse.has_value()) {
    if (failed(*subBlockIdParse)) {
      return failure();
    }
    hasSubBlockId = true;
  } else {
    StringRef dstModeKeyword;
    if (parser.parseKeyword(&dstModeKeyword)) {
      return failure();
    }
    if (dstModeKeyword == "split_m") {
      dstMode = AccStoreUbDstMode::SplitM;
    } else if (dstModeKeyword == "split_n") {
      dstMode = AccStoreUbDstMode::SplitN;
    } else {
      return parser.emitError(parser.getCurrentLocation(),
          "expected dst_mode(%sub_blockid), dst_mode(split_m), or dst_mode(split_n)");
    }
  }
  if (parser.parseRParen()) {
    return failure();
  }
  return success();
}

ParseResult resolveMteL0cUbOperands(
    OpAsmParser &parser, OperationState &result,
    OpAsmParser::UnresolvedOperand source, Type sourceType,
    OpAsmParser::UnresolvedOperand destination, Type destinationType,
    OpAsmParser::UnresolvedOperand m, Type mType,
    OpAsmParser::UnresolvedOperand n, Type nType,
    OpAsmParser::UnresolvedOperand srcStride, Type srcStrideType,
    OpAsmParser::UnresolvedOperand dstStride, Type dstStrideType,
    bool hasSubBlockId, OpAsmParser::UnresolvedOperand subBlockId,
    Type subBlockIdType, const StructuredAccStoreAsmState &state) {
  if (parser.resolveOperand(source, sourceType, result.operands) ||
      parser.resolveOperand(destination, destinationType, result.operands) ||
      parser.resolveOperand(m, mType, result.operands) ||
      parser.resolveOperand(n, nType, result.operands) ||
      parser.resolveOperand(srcStride, srcStrideType, result.operands) ||
      parser.resolveOperand(dstStride, dstStrideType, result.operands) ||
      parser.resolveOperands(state.preQuantOperands, state.preQuantTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(state.preReluOperands, state.preReluTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(state.clipValueOperands, state.clipValueTypes,
                             parser.getCurrentLocation(), result.operands) ||
      (hasSubBlockId &&
       parser.resolveOperand(subBlockId, subBlockIdType, result.operands)) ||
      parser.resolveOperands(state.splitOperands, state.splitTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(state.loop0SrcStrideOperands,
                             state.loop0SrcStrideTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(state.loop3CountOperands, state.loop3CountTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(state.loop3SrcStrideOperands,
                             state.loop3SrcStrideTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(state.loop3DstStrideOperands,
                             state.loop3DstStrideTypes,
                             parser.getCurrentLocation(), result.operands)) {
    return failure();
  }
  return success();
}

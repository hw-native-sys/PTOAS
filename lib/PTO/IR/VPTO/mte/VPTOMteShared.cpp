// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOMteShared.cpp - shared VPTO MTE externally linked helpers -------===//
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Externally linked MTE helpers declared in VPTOMteInternal.h (definitions
// here keep a single copy across the mte translation units); templated
// build/parse/print helpers remain in the internal header.
//===----------------------------------------------------------------------===//

#include "VPTOMteInternal.h"

#include "PTO/Support/CodeConstants.h"
#include "mlir/IR/Matchers.h"

namespace mlir::pto::mte_detail {

  void warnUnalignedBurstLengthWithoutPad(Operation *op, Value lenBurst,
                                          Value padValue,
                                          StringRef destinationSpace,
                                          StringRef remedy) {
    if (padValue) {
      return;
    }
    std::optional<int64_t> length = getConstantIntValue(lenBurst);
    if (!length || *length % kValue32 == 0) {
      return;
    }
    // The transfer itself stays valid: the 32B tail block simply carries source
    // data. Warn so the author can decide whether that is intended. The warning
    // goes through the location (not Operation::emitWarning, which appends the
    // operation itself and re-prints the IR while the verifier still runs).
    mlir::emitWarning(op->getLoc())
        << op->getName().getStringRef() << " len_burst (" << *length
        << " bytes) is not a multiple of " << kValue32 << ": the "
        << destinationSpace
        << " side of the transfer always writes whole 32B blocks, so the tail "
           "block [len_burst, roundUp(len_burst, 32)) carries source data; pass "
        << remedy << " to make it deterministic";
  }

  // Batch12: DmaLoopConfig expansion shared
  void addDmaLoopConfigOperands(OperationState &state,
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

  // Batch6: Bt/Fb 双胞胎 op 共用 build/parse/print
  void addDmaTripleOperandList(OperationState &state, Value source,
                                      Value destination, Value lenBurst,
                                      pto::DmaLoopConfig nburst) {
    state.addOperands({source, destination, lenBurst, nburst.count,
                       nburst.srcStride, nburst.dstStride});
  }

  ParseResult parseDmaTripleOp(OpAsmParser &parser,
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

  // Shape-derived operand positions: {m, k, start_row, start_col}.
  constexpr unsigned kShapeStartRowIndex = mlir::pto::kValue2;
  constexpr unsigned kShapeStartColIndex = mlir::pto::kValue3;
  // Full-control operand positions:
  // {m_start, k_start, m_step, k_step, src_stride, dst_stride}.
  constexpr unsigned kFullMStartIndex = 0;
  constexpr unsigned kFullKStartIndex = 1;
  constexpr unsigned kFullMStepIndex = mlir::pto::kValue2;
  constexpr unsigned kFullKStepIndex = mlir::pto::kValue3;
  constexpr unsigned kFullSrcStrideIndex = mlir::pto::kValue4;
  constexpr unsigned kFullDstStrideIndex = mlir::pto::kValue5;

  LogicalResult verifyMteL1L0LoadOperands(
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
      return verifyCubeBridgeLoadStart(
          op, shapeOperands[kShapeStartRowIndex],
          shapeNames[kShapeStartRowIndex], shapeOperands[kShapeStartColIndex],
          shapeNames[kShapeStartColIndex]);
    }
    [[maybe_unused]] static constexpr StringRef kFullNames[] = {
        "m_start", "k_start", "m_step", "k_step", "src_stride", "dst_stride"};
    for (auto [value, name] : llvm::zip(fullOperands, kFullNames)) {
      if (!value) {
        return op->emitOpError() << "full control form requires " << name;
  }
  }
    return verifyMteL1L0FullControlRanges(op, fullOperands);
  }

  LogicalResult verifyMteL1L0FullControlRanges(
      Operation *op, ArrayRef<Value> fullOperands) {
    constexpr int64_t kU16Max = 65535;
    constexpr int64_t kU8Max = 255;
    if (failed(verifyStaticControlRange(op, fullOperands[kFullMStartIndex],
                                        "m_start", 0, kU16Max)) ||
        failed(verifyStaticControlRange(op, fullOperands[kFullKStartIndex],
                                        "k_start", 0, kU16Max)) ||
        failed(verifyStaticControlRange(op, fullOperands[kFullMStepIndex],
                                        "m_step", 1, kU8Max)) ||
        failed(verifyStaticControlRange(op, fullOperands[kFullKStepIndex],
                                        "k_step", 1, kU8Max)) ||
        failed(verifyStaticControlRange(op, fullOperands[kFullSrcStrideIndex],
                                        "src_stride", 1, kU16Max)) ||
        failed(verifyStaticControlRange(op, fullOperands[kFullDstStrideIndex],
                                        "dst_stride", 1, kU16Max))) {
      return failure();
  }
    return success();
  }

  LogicalResult verifyMteL0cUbBufferSpaces(MteL0cUbOp op) {
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

  LogicalResult verifyMteL0cUbSubBlockId(MteL0cUbOp op) {
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

  LogicalResult verifyMteL0cUbSplitRestrictions(MteL0cUbOp op) {
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

} // namespace mlir::pto::mte_detail

// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOMteInternal.h - shared VPTO MTE asm/verify helpers -------------===//
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Shared MTE helpers (DMA loop/triple builders and parsers, structured-acc
// store clauses, L0c/L1/L0 operand verification). They live in a detail
// namespace so the original unqualified MLIR names keep resolving.
// Non-template helpers are declared here and defined once in VPTOMteShared.cpp;
// templated build/parse/print helpers stay in this header.
// Internal to lib/PTO/IR/VPTO/mte; not installed.
//===----------------------------------------------------------------------===//

#ifndef PTO_IR_VPTO_MTE_INTERNAL_H
#define PTO_IR_VPTO_MTE_INTERNAL_H

#include "VPTOInternal.h"

namespace mlir::pto::mte_detail {

using namespace mlir;
using namespace mlir::pto;

  // Batch12: DmaLoopConfig expansion shared
  void addDmaLoopConfigOperands(OperationState &state,
                                llvm::ArrayRef<pto::DmaLoopConfig> loops);

  // Batch6: Bt/Fb 双胞胎 op 共用 build/parse/print
  void addDmaTripleOperandList(OperationState &state, Value source,
                               Value destination, Value lenBurst,
                               pto::DmaLoopConfig nburst);

  ParseResult parseDmaTripleOp(OpAsmParser &parser, OperationState &result);

  LogicalResult verifyMteL1L0LoadOperands(Operation *op,
                                          ArrayRef<Value> shapeOperands,
                                          ArrayRef<StringRef> shapeNames,
                                          ArrayRef<Value> fullOperands);

  LogicalResult verifyMteL1L0FullControlRanges(
      Operation *op, ArrayRef<Value> fullOperands);

  LogicalResult verifyMteL0cUbBufferSpaces(MteL0cUbOp op);

  LogicalResult verifyMteL0cUbSubBlockId(MteL0cUbOp op);

  LogicalResult verifyMteL0cUbSplitRestrictions(MteL0cUbOp op);

  // Batch6: GmL1/L1Ub 双胞胎 op 共用 build/parse/print
  template <typename OpTy>
  [[maybe_unused]] static void buildDmaLoopOp(OpBuilder &odsBuilder, OperationState &state,
                             Value source, Value destination, Value lenBurst,
                             pto::DmaLoopConfig nburst,
                             llvm::ArrayRef<pto::DmaLoopConfig> loops) {
    state.addOperands(
        {source, destination, lenBurst, nburst.count, nburst.srcStride,
         nburst.dstStride});
    addDmaLoopConfigOperands(state, loops);
    state.addAttribute(
        OpTy::getOperandSegmentSizeAttr(),
        odsBuilder.getDenseI32ArrayAttr(
            {1, 1, 1, 1, 1, 1,
             static_cast<int32_t>(loops.size()),
             static_cast<int32_t>(loops.size()),
             static_cast<int32_t>(loops.size())}));
  }

  template <typename OpTy>
  [[maybe_unused]] static ParseResult parseDmaLoopOp(OpAsmParser &parser,
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

  template <typename OpTy>
  [[maybe_unused]] static void printDmaLoopOp(OpAsmPrinter &printer, OpTy op) {
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

  template <typename OpTy>
  [[maybe_unused]] static void printDmaTripleOpFields(OpAsmPrinter &printer, OpTy op) {
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

  // Shared structured-acc store clause + attribute-dictionary printing for the
  // MteL0cL1/MteL0cGm print pair.
  template <typename OpTy>
  [[maybe_unused]] static void printStructuredAccStoreClausesAndAttrs(OpTy op,
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

} // namespace mlir::pto::mte_detail

#endif // PTO_IR_VPTO_MTE_INTERNAL_H

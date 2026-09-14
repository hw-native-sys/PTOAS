// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOMadInternal.h - Mad-family op-common templates ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Internal to lib/PTO/IR/VPTO/mad; not installed. Verification helpers
// live in VPTOMadVerify.h and the custom-assembly parse/print helpers in
// VPTOMadAsm.h; this header keeps the per-op parse/print/verify templates.

#ifndef PTO_IR_VPTO_MAD_INTERNAL_H
#define PTO_IR_VPTO_MAD_INTERNAL_H

#include "VPTOMadAsm.h"
#include "VPTOMadVerify.h"

namespace mlir::pto::mad_detail {
using namespace mlir;
using namespace mlir::pto;

inline mlir::ParseResult
resolveMadRuntimeFlagOperands(mlir::OpAsmParser &parser,
                              mlir::OperationState &result,
                              const MadRuntimeFlagParseState &flags) {
  if (flags.hasUnitFlagValue &&
      parser.resolveOperand(flags.unitFlagValue, flags.unitFlagType,
                            result.operands)) {
    return mlir::failure();
  }
  if (flags.hasAccInitValue &&
      parser.resolveOperand(flags.accInitValue, flags.accInitType,
                            result.operands)) {
    return mlir::failure();
  }
  if (flags.hasDisableGemvValue &&
      parser.resolveOperand(flags.disableGemvValue, flags.disableGemvType,
                            result.operands)) {
    return mlir::failure();
  }
  if (flags.hasBiasInitValue &&
      parser.resolveOperand(flags.biasInitValue, flags.biasInitType,
                            result.operands)) {
    return mlir::failure();
  }
  return mlir::success();
}

template <typename OpT>
[[maybe_unused]] static mlir::ParseResult
parseMadSemanticOpCommon(mlir::OpAsmParser &parser, mlir::OperationState &result,
                         bool hasBias, bool parseTf32ModeClause) {
  mlir::OpAsmParser::UnresolvedOperand lhs, rhs, dst, bias;
  mlir::OpAsmParser::UnresolvedOperand m, n, k;
  if (parseRequiredOperandWithComma(parser, lhs) ||
      parseRequiredOperandWithComma(parser, rhs) ||
      parseRequiredOperandWithComma(parser, dst) ||
      (hasBias && parseRequiredOperandWithComma(parser, bias)) ||
      parseRequiredOperandWithComma(parser, m) ||
      parseRequiredOperandWithComma(parser, n) ||
      parser.parseOperand(k)) {
    return mlir::failure();
  }
  MadRuntimeFlagParseState flags;
  if (mlir::failed(parseMadRuntimeFlagClauses(parser, flags))) {
    return mlir::failure();
  }
  mlir::NamedAttrList attrs;
  if (mlir::failed(parseMadSemanticClauses(parser, attrs, parseTf32ModeClause))) {
    return mlir::failure();
  }
  if (parser.parseOptionalAttrDict(attrs) || parser.parseColon()) {
    return mlir::failure();
  }
  mlir::Type lhsType, rhsType, dstType, mType, nType, kType, biasType;
  if (mlir::failed(parseMadSemanticTypes(
          parser, hasBias, lhsType, rhsType, dstType, biasType, mType, nType,
          kType, flags))) {
    return mlir::failure();
  }
  result.addAttributes(attrs);
  prefillMadOperandSegmentSizes(parser, result, hasBias, flags);
  if (mlir::failed(resolveMadSemanticOperands(parser, result, hasBias, lhs,
                                              lhsType, rhs, rhsType, dst,
                                              dstType, bias, biasType, m, mType,
                                              n, nType, k, kType))) {
    return mlir::failure();
  }
  return resolveMadRuntimeFlagOperands(parser, result, flags);
}

template <typename OpT>
static void printMadRuntimeFlagClauses(mlir::OpAsmPrinter &printer, OpT op) {
  if (auto uf = op.getUnitFlagValue()) {
    printer << " unit_flag_value(" << uf << ")";
  }
  if (auto acc = op.getAccInitValue()) {
    printer << " acc_init(" << acc << ")";
  }
  if (auto gemv = op.getDisableGemvValue()) {
    printer << " disable_gemv_value(" << gemv << ")";
  }
  if (auto bias = op.getBiasInitValue()) {
    printer << " bias_init(" << bias << ")";
  }
}

template <typename OpT>
static void appendMadRuntimeFlagTypes(mlir::OpAsmPrinter &printer, OpT op) {
  if (auto uf = op.getUnitFlagValue()) {
    printer << ", " << uf.getType();
  }
  if (auto acc = op.getAccInitValue()) {
    printer << ", " << acc.getType();
  }
  if (auto gemv = op.getDisableGemvValue()) {
    printer << ", " << gemv.getType();
  }
  if (auto bias = op.getBiasInitValue()) {
    printer << ", " << bias.getType();
  }
}

template <typename OpT>
static void printMadSemanticOpNoBias(mlir::OpAsmPrinter &printer, OpT op,
                                     bool allowTf32Mode) {
  printer << ' ' << op.getLhs() << ", " << op.getRhs() << ", " << op.getDst()
          << ", " << op.getM() << ", " << op.getN() << ", " << op.getK();
  printMadRuntimeFlagClauses(printer, op);
  printMadSemanticClauses(printer, op, allowTf32Mode);
  printer.printOptionalAttrDict(op->getAttrs(),
                                getMadSemanticElidedAttrs(allowTf32Mode));
  printer << " : " << op.getLhs().getType() << ", " << op.getRhs().getType()
          << ", " << op.getDst().getType() << ", " << op.getM().getType()
          << ", " << op.getN().getType() << ", " << op.getK().getType();
  appendMadRuntimeFlagTypes(printer, op);
}

template <typename OpT>
static void printMadSemanticOpWithBias(mlir::OpAsmPrinter &printer, OpT op,
                                       bool allowTf32Mode) {
  printer << ' ' << op.getLhs() << ", " << op.getRhs() << ", " << op.getDst()
          << ", " << op.getBias() << ", " << op.getM() << ", " << op.getN()
          << ", " << op.getK();
  printMadRuntimeFlagClauses(printer, op);
  printMadSemanticClauses(printer, op, allowTf32Mode);
  printer.printOptionalAttrDict(op->getAttrs(),
                                getMadSemanticElidedAttrs(allowTf32Mode));
  printer << " : " << op.getLhs().getType() << ", " << op.getRhs().getType()
          << ", " << op.getDst().getType() << ", " << op.getBias().getType()
          << ", " << op.getM().getType() << ", " << op.getN().getType()
          << ", " << op.getK().getType();
  appendMadRuntimeFlagTypes(printer, op);
}

// Shared MemoryEffects for the mad semantic ops: lhs/rhs are read and dst is
// written. A runtime acc_init/bias_init operand may select the accumulate
// path or make the accumulator the C-matrix source, so dst becomes
// read+write conservatively; the accumulating ops (mad_acc/mad_mx_acc)
// always read+write dst.
template <typename OpT>
static void collectMadSemanticEffects(
    OpT op,
    mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<
        mlir::MemoryEffects::Effect>> &effects,
    bool accumulates) {
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &op.getLhsMutable());
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &op.getRhsMutable());
  effects.emplace_back(mlir::MemoryEffects::Write::get(), &op.getDstMutable());
  if (accumulates || op.getAccInitValue() || op.getBiasInitValue()) {
    effects.emplace_back(mlir::MemoryEffects::Read::get(), &op.getDstMutable());
  }
}

// Bias variants (mad_bias/mad_mx_bias) additionally read their bias pointer.
template <typename OpT>
static void collectMadSemanticBiasEffects(
    OpT op,
    mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<
        mlir::MemoryEffects::Effect>> &effects,
    bool accumulates) {
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &op.getBiasMutable());
  collectMadSemanticEffects(op, effects, accumulates);
}

// Batch7: Mad 家族共用 tf32_mode 读取 + 语义校验
template <typename OpTy>
static mlir::LogicalResult verifyMadSemanticWithTf32(OpTy op) {
  std::optional<mlir::pto::Tf32Mode> tf32Mode;
  if (auto tf32ModeAttr =
          op->template getAttrOfType<mlir::pto::Tf32ModeAttr>("tf32_mode")) {
    tf32Mode = tf32ModeAttr.getValue();
  }
  return verifyMadSemanticClauses(op, op.getLhs().getType(),
                                  op.getRhs().getType(), op.getDst().getType(),
                                  std::nullopt, tf32Mode, op.getSatMode(),
                                  op->hasAttr("n_dir"));
}

} // namespace mlir::pto::mad_detail

#endif // PTO_IR_VPTO_MAD_INTERNAL_H

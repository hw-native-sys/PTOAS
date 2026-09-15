// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOMadInternal.h - Mad-family declarations and templates ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Internal to lib/PTO/IR/VPTO/mad; not installed.

#ifndef PTO_IR_VPTO_MAD_INTERNAL_H
#define PTO_IR_VPTO_MAD_INTERNAL_H

#include "VPTOInternal.h"
// Shared Mad-family helpers. They live in a detail namespace so the
// per-instruction TUs (VPTOMad*Op.cpp) can `using namespace` them without
// polluting mlir::pto.
namespace mlir::pto::mad_detail {
using namespace mlir;
using namespace mlir::pto;


// Shared Mad-family helpers (defined in mad/VPTOMad.cpp).
mlir::LogicalResult verifyMadPointerKinds(mlir::Operation *op, mlir::Type lhsTy,
                                          mlir::Type rhsTy, mlir::Type dstTy,
                                          std::optional<mlir::Type> biasTy = std::nullopt);
mlir::LogicalResult verifyMadMxCommon(mlir::Operation *op, mlir::Type lhsTy,
                                      mlir::Type rhsTy, mlir::Type dstTy,
                                      std::optional<mlir::Type> biasTy = std::nullopt);
mlir::LogicalResult verifyMadSemanticClauses(mlir::Operation *op, mlir::Type lhsTy,
                                             mlir::Type rhsTy, mlir::Type dstTy,
                                             std::optional<mlir::Type> biasTy,
                                             std::optional<mlir::pto::Tf32Mode> tf32Mode,
                                             std::optional<mlir::pto::MadSatMode> satMode,
                                             bool hasNDir);
mlir::ParseResult parseMadSemanticClauses(mlir::OpAsmParser &parser,
                                          mlir::NamedAttrList &attrs,
                                          bool parseTf32ModeClause);
// Collected parse state for the optional runtime flag operands.
struct MadRuntimeFlagParseState {
  mlir::OpAsmParser::UnresolvedOperand unitFlagValue;
  mlir::OpAsmParser::UnresolvedOperand accInitValue;
  mlir::OpAsmParser::UnresolvedOperand disableGemvValue;
  mlir::OpAsmParser::UnresolvedOperand biasInitValue;
  mlir::Type unitFlagType;
  mlir::Type accInitType;
  mlir::Type disableGemvType;
  mlir::Type biasInitType;
  bool hasUnitFlagValue = false;
  bool hasAccInitValue = false;
  bool hasDisableGemvValue = false;
  bool hasBiasInitValue = false;
};

mlir::ParseResult parseMadSemanticTypes(
    mlir::OpAsmParser &parser, bool hasBias, mlir::Type &lhsType,
    mlir::Type &rhsType, mlir::Type &dstType, mlir::Type &biasType,
    mlir::Type &mType, mlir::Type &nType, mlir::Type &kType,
    MadRuntimeFlagParseState &flags);
mlir::ParseResult resolveMadSemanticOperands(
    mlir::OpAsmParser &parser, mlir::OperationState &result, bool hasBias,
    mlir::OpAsmParser::UnresolvedOperand lhs, mlir::Type lhsType,
    mlir::OpAsmParser::UnresolvedOperand rhs, mlir::Type rhsType,
    mlir::OpAsmParser::UnresolvedOperand dst, mlir::Type dstType,
    mlir::OpAsmParser::UnresolvedOperand bias, mlir::Type biasType,
    mlir::OpAsmParser::UnresolvedOperand m, mlir::Type mType,
    mlir::OpAsmParser::UnresolvedOperand n, mlir::Type nType,
    mlir::OpAsmParser::UnresolvedOperand k, mlir::Type kType);
void printMadSemanticClauses(mlir::OpAsmPrinter &printer, mlir::Operation *op,
                             bool allowTf32Mode);
llvm::ArrayRef<llvm::StringRef> getMadSemanticElidedAttrs(bool allowTf32Mode);


// Parse one `keyword(%operand)` runtime flag clause.
inline mlir::ParseResult
parseMadRuntimeFlagClause(mlir::OpAsmParser &parser, const char *keyword,
                          mlir::OpAsmParser::UnresolvedOperand &operand,
                          bool &present) {
  if (mlir::failed(parser.parseOptionalKeyword(keyword))) {
    return mlir::success();
  }
  present = true;
  return mlir::failure(parser.parseLParen() || parser.parseOperand(operand) ||
                       parser.parseRParen());
}

// Parse the four optional runtime flag clauses in operand order.
inline mlir::ParseResult
parseMadRuntimeFlagClauses(mlir::OpAsmParser &parser,
                           MadRuntimeFlagParseState &flags) {
  if (failed(parseMadRuntimeFlagClause(parser, "unit_flag_value",
                                       flags.unitFlagValue,
                                       flags.hasUnitFlagValue)) ||
      failed(parseMadRuntimeFlagClause(parser, "acc_init",
                                       flags.accInitValue,
                                       flags.hasAccInitValue)) ||
      failed(parseMadRuntimeFlagClause(parser, "disable_gemv_value",
                                       flags.disableGemvValue,
                                       flags.hasDisableGemvValue)) ||
      failed(parseMadRuntimeFlagClause(parser, "bias_init",
                                       flags.biasInitValue,
                                       flags.hasBiasInitValue))) {
    return mlir::failure();
  }
  return mlir::success();
}

// Prefill the mandatory operandSegmentSizes attribute with the canonical
// sizes when the input text omitted it.
inline void prefillMadOperandSegmentSizes(mlir::OpAsmParser &parser,
                                          mlir::OperationState &result,
                                          bool hasBias,
                                          const MadRuntimeFlagParseState &flags) {
  if (result.attributes.get("operandSegmentSizes")) {
    return;
  }
  int fixedCount = (hasBias ? 4 : 3) + 3;
  llvm::SmallVector<int32_t, 10> sizes(fixedCount, 1);
  sizes.push_back(flags.hasUnitFlagValue ? 1 : 0);
  sizes.push_back(flags.hasAccInitValue ? 1 : 0);
  sizes.push_back(flags.hasDisableGemvValue ? 1 : 0);
  sizes.push_back(flags.hasBiasInitValue ? 1 : 0);
  result.addAttribute(
      "operandSegmentSizes",
      mlir::DenseI32ArrayAttr::get(parser.getContext(), sizes));
}

// Resolve the runtime flag operands onto the operation, in operand order.
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
static void printMadOperandSegmentSizesIfNeeded(mlir::OpAsmPrinter &printer,
                                                OpT op) {
  if (op->getAttr("operandSegmentSizes")) {
    return;
  }
  bool any = op.getUnitFlagValue() || op.getAccInitValue() ||
             op.getDisableGemvValue() || op.getBiasInitValue();
  if (!any) {
    return;
  }
  llvm::SmallVector<int32_t, 10> sizes;
  auto push = [&sizes](mlir::Value v) { sizes.push_back(v ? 1 : 0); };
  push(op.getLhs());
  push(op.getRhs());
  push(op.getDst());
  if constexpr (std::is_same_v<OpT, mlir::pto::MadBiasOp> ||
                std::is_same_v<OpT, mlir::pto::MadMxBiasOp>) {
    push(op.getBias());
  }
  push(op.getM());
  push(op.getN());
  push(op.getK());
  push(op.getUnitFlagValue());
  push(op.getAccInitValue());
  push(op.getDisableGemvValue());
  push(op.getBiasInitValue());
  printer << " {operandSegmentSizes = array<i32:";
  llvm::interleave(
      sizes, printer, [&printer](int32_t s) { printer << " " << s; }, ",");
  printer << "}";
}

template <typename OpT>
static void printMadSemanticOpNoBias(mlir::OpAsmPrinter &printer, OpT op,
                                     bool allowTf32Mode) {
  printer << ' ' << op.getLhs() << ", " << op.getRhs() << ", " << op.getDst()
          << ", " << op.getM() << ", " << op.getN() << ", " << op.getK();
  printMadRuntimeFlagClauses(printer, op);
  printMadSemanticClauses(printer, op, allowTf32Mode);
  printMadOperandSegmentSizesIfNeeded(printer, op);
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
  printMadOperandSegmentSizesIfNeeded(printer, op);
  printer.printOptionalAttrDict(op->getAttrs(),
                                getMadSemanticElidedAttrs(allowTf32Mode));
  printer << " : " << op.getLhs().getType() << ", " << op.getRhs().getType()
          << ", " << op.getDst().getType() << ", " << op.getBias().getType()
          << ", " << op.getM().getType() << ", " << op.getN().getType()
          << ", " << op.getK().getType();
  appendMadRuntimeFlagTypes(printer, op);
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

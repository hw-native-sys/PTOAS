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
mlir::ParseResult parseMadSemanticTypes(mlir::OpAsmParser &parser, bool hasBias,
                                        mlir::Type &lhsType, mlir::Type &rhsType,
                                        mlir::Type &dstType, mlir::Type &biasType,
                                        mlir::Type &mType, mlir::Type &nType,
                                        mlir::Type &kType);
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
  mlir::NamedAttrList attrs;
  if (mlir::failed(parseMadSemanticClauses(parser, attrs, parseTf32ModeClause))) {
    return mlir::failure();
  }
  if (parser.parseOptionalAttrDict(attrs) || parser.parseColon()) {
    return mlir::failure();
  }
  mlir::Type lhsType, rhsType, dstType, mType, nType, kType, biasType;
  if (mlir::failed(parseMadSemanticTypes(parser, hasBias, lhsType, rhsType,
                                         dstType, biasType, mType, nType,
                                         kType))) {
    return mlir::failure();
  }
  result.addAttributes(attrs);
  if (mlir::failed(resolveMadSemanticOperands(parser, result, hasBias, lhs,
                                              lhsType, rhs, rhsType, dst,
                                              dstType, bias, biasType, m, mType,
                                              n, nType, k, kType))) {
    return mlir::failure();
  }
  return mlir::success();
}

template <typename OpT>
static void printMadSemanticOpNoBias(mlir::OpAsmPrinter &printer, OpT op,
                                     bool allowTf32Mode) {
  printer << ' ' << op.getLhs() << ", " << op.getRhs() << ", " << op.getDst()
          << ", " << op.getM() << ", " << op.getN() << ", " << op.getK();
  printMadSemanticClauses(printer, op, allowTf32Mode);
  printer.printOptionalAttrDict(op->getAttrs(),
                                getMadSemanticElidedAttrs(allowTf32Mode));
  printer << " : " << op.getLhs().getType() << ", " << op.getRhs().getType()
          << ", " << op.getDst().getType() << ", " << op.getM().getType()
          << ", " << op.getN().getType() << ", " << op.getK().getType();
}

template <typename OpT>
static void printMadSemanticOpWithBias(mlir::OpAsmPrinter &printer, OpT op,
                                       bool allowTf32Mode) {
  printer << ' ' << op.getLhs() << ", " << op.getRhs() << ", " << op.getDst()
          << ", " << op.getBias() << ", " << op.getM() << ", " << op.getN()
          << ", " << op.getK();
  printMadSemanticClauses(printer, op, allowTf32Mode);
  printer.printOptionalAttrDict(op->getAttrs(),
                                getMadSemanticElidedAttrs(allowTf32Mode));
  printer << " : " << op.getLhs().getType() << ", " << op.getRhs().getType()
          << ", " << op.getDst().getType() << ", " << op.getBias().getType()
          << ", " << op.getM().getType() << ", " << op.getN().getType()
          << ", " << op.getK().getType();
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

#endif // PTO_IR_VPTO_MAD_INTERNAL_H

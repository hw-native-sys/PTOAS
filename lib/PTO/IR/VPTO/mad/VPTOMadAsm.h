// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOMadAsm.h - Mad-family custom-assembly parse/print helpers ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Internal to lib/PTO/IR/VPTO/mad; not installed. Split out of
// VPTOMadInternal.h to keep header sizes within the codecheck budget.

#ifndef PTO_IR_VPTO_MAD_ASM_H
#define PTO_IR_VPTO_MAD_ASM_H

#include "VPTOInternal.h"

namespace mlir::pto::mad_detail {
using namespace mlir;
using namespace mlir::pto;

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

inline ParseResult parseMadFixedTypes(OpAsmParser &parser, bool hasBias,
                                      Type &lhsType, Type &rhsType,
                                      Type &dstType, Type &biasType,
                                      Type &mType, Type &nType, Type &kType) {
  if (parser.parseType(lhsType) || parser.parseComma() ||
      parser.parseType(rhsType) || parser.parseComma() ||
      parser.parseType(dstType) || parser.parseComma()) {
    return failure();
  }
  if (hasBias && (parser.parseType(biasType) || parser.parseComma())) {
    return failure();
  }
  return failure(parser.parseType(mType) || parser.parseComma() ||
                 parser.parseType(nType) || parser.parseComma() ||
                 parser.parseType(kType));
}

inline ParseResult parseMadFlagType(OpAsmParser &parser, bool present,
                                    Type &out) {
  if (!present) {
    return success();
  }
  return failure(parser.parseComma() || parser.parseType(out));
}

inline ParseResult parseMadSemanticTypes(
    OpAsmParser &parser, bool hasBias, Type &lhsType, Type &rhsType,
    Type &dstType, Type &biasType, Type &mType, Type &nType, Type &kType,
    MadRuntimeFlagParseState &flags) {
  if (failed(parseMadFixedTypes(parser, hasBias, lhsType, rhsType, dstType,
                                biasType, mType, nType, kType))) {
    return failure();
  }
  if (failed(parseMadFlagType(parser, flags.hasUnitFlagValue,
                              flags.unitFlagType)) ||
      failed(parseMadFlagType(parser, flags.hasAccInitValue,
                              flags.accInitType)) ||
      failed(parseMadFlagType(parser, flags.hasDisableGemvValue,
                              flags.disableGemvType)) ||
      failed(parseMadFlagType(parser, flags.hasBiasInitValue,
                              flags.biasInitType))) {
    return failure();
  }
  return success();
}

inline ParseResult resolveMadSemanticOperands(
    OpAsmParser &parser, OperationState &result, bool hasBias,
    OpAsmParser::UnresolvedOperand lhs, Type lhsType,
    OpAsmParser::UnresolvedOperand rhs, Type rhsType,
    OpAsmParser::UnresolvedOperand dst, Type dstType,
    OpAsmParser::UnresolvedOperand bias, Type biasType,
    OpAsmParser::UnresolvedOperand m, Type mType,
    OpAsmParser::UnresolvedOperand n, Type nType,
    OpAsmParser::UnresolvedOperand k, Type kType) {
  if (parser.resolveOperand(lhs, lhsType, result.operands) ||
      parser.resolveOperand(rhs, rhsType, result.operands) ||
      parser.resolveOperand(dst, dstType, result.operands)) {
    return failure();
  }
  if (hasBias) {
    if (parser.resolveOperand(bias, biasType, result.operands)) {
      return failure();
    }
  }
  if (parser.resolveOperand(m, mType, result.operands) ||
      parser.resolveOperand(n, nType, result.operands) ||
      parser.resolveOperand(k, kType, result.operands)) {
    return failure();
  }
  return success();
}

inline StringRef stringifyTf32ModeToken(pto::Tf32Mode mode) {
  switch (mode) {
  case pto::Tf32Mode::RoundEven:
    return "round_even";
  case pto::Tf32Mode::RoundAway:
    return "round_away";
  }
  llvm_unreachable("unexpected tf32 mode");
}

inline StringRef stringifyMadUnitFlagModeToken(pto::MadUnitFlagMode mode) {
  switch (mode) {
  case pto::MadUnitFlagMode::CheckOnly:
    return "check_only";
  case pto::MadUnitFlagMode::CheckAndSet:
    return "check_and_set";
  }
  llvm_unreachable("unexpected mad unit flag mode");
}

inline StringRef stringifyMadSatModeToken(pto::MadSatMode mode) {
  switch (mode) {
  case pto::MadSatMode::Sat:
    return "sat";
  case pto::MadSatMode::NoSat:
    return "nosat";
  }
  llvm_unreachable("unexpected mad sat mode");
}

inline void printMadSemanticClauses(OpAsmPrinter &printer, Operation *op,
                                    bool allowTf32Mode) {
  if (auto unitFlagMode = op->getAttrOfType<pto::MadUnitFlagModeAttr>(
          "unit_flag_mode")) {
    printer << " unit_flag("
            << stringifyMadUnitFlagModeToken(unitFlagMode.getValue()) << ")";
  }
  if (op->hasAttr("disable_gemv")) {
    printer << " disable_gemv";
  }
  if (auto satMode = op->getAttrOfType<pto::MadSatModeAttr>("sat_mode")) {
    printer << ' ' << stringifyMadSatModeToken(satMode.getValue());
  }
  if (allowTf32Mode) {
    if (auto tf32Mode = op->getAttrOfType<pto::Tf32ModeAttr>("tf32_mode")) {
      printer << " tf32_mode(" << stringifyTf32ModeToken(tf32Mode.getValue())
              << ")";
    }
  }
  if (op->hasAttr("n_dir")) {
    printer << " n_dir";
  }
}

inline ArrayRef<StringRef> getMadSemanticElidedAttrs(bool allowTf32Mode) {
  static constexpr StringRef kWithTf32[] = {"unit_flag_mode", "disable_gemv",
                                            "sat_mode", "tf32_mode", "n_dir"};
  static constexpr StringRef kWithoutTf32[] = {"unit_flag_mode",
                                               "disable_gemv", "sat_mode",
                                               "n_dir"};
  return allowTf32Mode ? ArrayRef<StringRef>(kWithTf32)
                       : ArrayRef<StringRef>(kWithoutTf32);
}


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


inline std::optional<pto::MadUnitFlagMode>
parseMadUnitFlagModeToken(StringRef token) {
  if (token == "check_only") {
    return pto::MadUnitFlagMode::CheckOnly;
  }
  if (token == "check_and_set") {
    return pto::MadUnitFlagMode::CheckAndSet;
  }
  return std::nullopt;
}

inline std::optional<pto::Tf32Mode> parseTf32ModeToken(StringRef token) {
  if (token == "round_even") {
    return pto::Tf32Mode::RoundEven;
  }
  if (token == "round_away") {
    return pto::Tf32Mode::RoundAway;
  }
  return std::nullopt;
}


inline ParseResult parseMadSemanticClauses(OpAsmParser &parser,
                                       NamedAttrList &attrs,
                                       bool parseTf32ModeClause) {
  StringRef unitFlagKeyword;
  if (failed(parser.parseOptionalKeyword("unit_flag"))) {
    /* no unit_flag clause */
  } else {
    if (parser.parseLParen() || parser.parseKeyword(&unitFlagKeyword) ||
        parser.parseRParen()) {
      return failure();
    }
    auto mode = parseMadUnitFlagModeToken(unitFlagKeyword);
    if (!mode) {
      return parser.emitError(parser.getCurrentLocation())
             << "expected unit_flag(check_only|check_and_set)";
    }
    attrs.set("unit_flag_mode",
              pto::MadUnitFlagModeAttr::get(parser.getContext(), *mode));
  }
  if (succeeded(parser.parseOptionalKeyword("disable_gemv"))) {
    attrs.set("disable_gemv", UnitAttr::get(parser.getContext()));
  }
  if (succeeded(parser.parseOptionalKeyword("sat"))) {
    attrs.set("sat_mode",
              pto::MadSatModeAttr::get(parser.getContext(),
                                       pto::MadSatMode::Sat));
  } else if (succeeded(parser.parseOptionalKeyword("nosat"))) {
    attrs.set("sat_mode",
              pto::MadSatModeAttr::get(parser.getContext(),
                                       pto::MadSatMode::NoSat));
  }
  if (parseTf32ModeClause &&
      succeeded(parser.parseOptionalKeyword("tf32_mode"))) {
    StringRef tf32Keyword;
    if (parser.parseLParen() || parser.parseKeyword(&tf32Keyword) ||
        parser.parseRParen()) {
      return failure();
    }
    auto mode = parseTf32ModeToken(tf32Keyword);
    if (!mode) {
      return parser.emitError(parser.getCurrentLocation())
             << "expected tf32_mode(round_even|round_away)";
    }
    attrs.set("tf32_mode", pto::Tf32ModeAttr::get(parser.getContext(), *mode));
  }
  if (succeeded(parser.parseOptionalKeyword("n_dir"))) {
    attrs.set("n_dir", UnitAttr::get(parser.getContext()));
  }
  return success();
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

} // namespace mlir::pto::mad_detail

#endif // PTO_IR_VPTO_MAD_ASM_H

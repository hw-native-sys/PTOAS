// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOMad.cpp - shared VPTO Mad semantic asm/verify helpers ----------===//
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Mad-family shared helpers: pointer-kind/semantic-clause verification and
// the unit_flag/disable_gemv/sat/tf32_mode/n_dir clause parse-print infra.
// Declarations live in mad/VPTOMadInternal.h; the per-instruction TUs
// (VPTOMad*Op.cpp) instantiate the header templates.
//===----------------------------------------------------------------------===//

#include "VPTOInternal.h"

using namespace mlir;
using namespace mlir::pto;

LogicalResult verifyMadPointerKinds(Operation *op, Type lhsTy, Type rhsTy,
                                           Type dstTy,
                                           std::optional<Type> biasTy) {
  auto lhsType = dyn_cast<pto::PtrType>(lhsTy);
  auto rhsType = dyn_cast<pto::PtrType>(rhsTy);
  auto dstType = dyn_cast<pto::PtrType>(dstTy);
  if (!lhsType || !rhsType || !dstType) {
    return op->emitOpError("requires typed !pto.ptr lhs/rhs/dst operands");
  }

  const auto lhsAS = lhsType.getMemorySpace().getAddressSpace();
  const auto rhsAS = rhsType.getMemorySpace().getAddressSpace();
  const auto dstAS = dstType.getMemorySpace().getAddressSpace();

  const bool isStrongCube =
      lhsAS == pto::AddressSpace::LEFT && rhsAS == pto::AddressSpace::RIGHT &&
      dstAS == pto::AddressSpace::ACC;
  if (!isStrongCube) {
    return op->emitOpError("requires l0a/l0b/l0c-typed lhs/rhs/dst pointers");
  }

  if (!biasTy) {
    return success();
  }

  auto biasType = dyn_cast<pto::PtrType>(*biasTy);
  if (!biasType) {
    return op->emitOpError("requires typed !pto.ptr bias operand");
  }
  if (biasType.getMemorySpace().getAddressSpace() != pto::AddressSpace::BIAS) {
    return op->emitOpError("requires bias pointer in !pto.ptr<..., bt>");
  }
  if (biasType.getElementType() != dstType.getElementType()) {
    return op->emitOpError("requires bias element type to match dst element type");
  }
  return success();
}

LogicalResult verifyMadMxCommon(Operation *op, Type lhsTy, Type rhsTy,
                                       Type dstTy,
                                       std::optional<Type> biasTy) {
  if (failed(verifyMadPointerKinds(op, lhsTy, rhsTy, dstTy, biasTy))) {
    return failure();
  }

  auto lhsType = cast<pto::PtrType>(lhsTy);
  auto rhsType = cast<pto::PtrType>(rhsTy);
  auto dstType = cast<pto::PtrType>(dstTy);
  const auto lhsAS = lhsType.getMemorySpace().getAddressSpace();
  const auto rhsAS = rhsType.getMemorySpace().getAddressSpace();
  const auto dstAS = dstType.getMemorySpace().getAddressSpace();
  const bool isStrongCube =
      lhsAS == pto::AddressSpace::LEFT && rhsAS == pto::AddressSpace::RIGHT &&
      dstAS == pto::AddressSpace::ACC;
  if (!isStrongCube) {
    return op->emitOpError("requires l0a/l0b/l0c-typed lhs/rhs/dst pointers");
  }

  if (!isMxElementType(lhsType.getElementType()) ||
      !isMxElementType(rhsType.getElementType())) {
    return op->emitOpError(
        "requires MX lhs/rhs element types (f8E4M3FN, f8E5M2, f4E1M2x2, or "
        "f4E2M1x2)");
  }
  return success();
}

static std::optional<pto::MadUnitFlagMode>
parseMadUnitFlagModeToken(StringRef token) {
  if (token == "check_only") {
    return pto::MadUnitFlagMode::CheckOnly;
  }
  if (token == "check_and_set") {
    return pto::MadUnitFlagMode::CheckAndSet;
  }
  return std::nullopt;
}

static StringRef stringifyMadUnitFlagModeToken(pto::MadUnitFlagMode mode) {
  switch (mode) {
  case pto::MadUnitFlagMode::CheckOnly:
    return "check_only";
  case pto::MadUnitFlagMode::CheckAndSet:
    return "check_and_set";
  }
  llvm_unreachable("unexpected mad unit flag mode");
}

static std::optional<pto::Tf32Mode> parseTf32ModeToken(StringRef token) {
  if (token == "round_even") {
    return pto::Tf32Mode::RoundEven;
  }
  if (token == "round_away") {
    return pto::Tf32Mode::RoundAway;
  }
  return std::nullopt;
}

static StringRef stringifyTf32ModeToken(pto::Tf32Mode mode) {
  switch (mode) {
  case pto::Tf32Mode::RoundEven:
    return "round_even";
  case pto::Tf32Mode::RoundAway:
    return "round_away";
  }
  llvm_unreachable("unexpected tf32 mode");
}

static StringRef stringifyMadSatModeToken(pto::MadSatMode mode) {
  switch (mode) {
  case pto::MadSatMode::Sat:
    return "sat";
  case pto::MadSatMode::NoSat:
    return "nosat";
  }
  llvm_unreachable("unexpected mad sat mode");
}

LogicalResult verifyMadSemanticClauses(Operation *op, Type lhsTy,
                                              Type rhsTy, Type dstTy,
                                              std::optional<Type> biasTy,
                                              std::optional<pto::Tf32Mode> tf32Mode,
                                              std::optional<pto::MadSatMode> satMode,
                                              bool hasNDir) {
  if (failed(verifyMadPointerKinds(op, lhsTy, rhsTy, dstTy, biasTy))) {
    return failure();
  }

  auto lhsType = dyn_cast<pto::PtrType>(lhsTy);
  auto rhsType = dyn_cast<pto::PtrType>(rhsTy);
  auto dstType = dyn_cast<pto::PtrType>(dstTy);
  if (!lhsType || !rhsType || !dstType) {
    return op->emitOpError("requires typed !pto.ptr lhs/rhs/dst operands");
  }

  if (tf32Mode) {
    if (!(lhsType.getElementType().isF32() && rhsType.getElementType().isF32() &&
          dstType.getElementType().isF32())) {
      return op->emitOpError(
          "requires tf32_mode only for f32 lhs/rhs/dst element types");
    }
  }
  if (pto::isPTOHiFloat8Type(lhsType.getElementType()) !=
      pto::isPTOHiFloat8Type(rhsType.getElementType())) {
    return op->emitOpError(
        "requires lhs/rhs to both use hif8 or both use non-hif8 element types");
  }
  if (satMode) {
    auto isFloatLike = [](Type type) {
      if (isa<FloatType>(type)) {
        return true;
      }
      return pto::isPTOLowPrecisionType(type);
    };
    if (!(isFloatLike(lhsType.getElementType()) &&
          isFloatLike(rhsType.getElementType()) &&
          isFloatLike(dstType.getElementType()))) {
      return op->emitOpError(
          "requires sat/nosat only for floating lhs/rhs/dst element types");
    }
  }
  (void)hasNDir;
  return success();
}

ParseResult parseMadSemanticClauses(OpAsmParser &parser,
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

ParseResult parseMadSemanticTypes(OpAsmParser &parser, bool hasBias,
                                         Type &lhsType, Type &rhsType,
                                         Type &dstType, Type &biasType,
                                         Type &mType, Type &nType,
                                         Type &kType) {
  if (parser.parseType(lhsType) || parser.parseComma() ||
      parser.parseType(rhsType) || parser.parseComma() ||
      parser.parseType(dstType) || parser.parseComma()) {
    return failure();
  }
  if (hasBias) {
    if (parser.parseType(biasType) || parser.parseComma()) {
      return failure();
    }
  }
  if (parser.parseType(mType) || parser.parseComma() ||
      parser.parseType(nType) || parser.parseComma() ||
      parser.parseType(kType)) {
    return failure();
  }
  return success();
}

ParseResult resolveMadSemanticOperands(
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

void printMadSemanticClauses(OpAsmPrinter &printer, Operation *op,
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

ArrayRef<StringRef> getMadSemanticElidedAttrs(bool allowTf32Mode) {
  static constexpr StringRef kWithTf32[] = {"unit_flag_mode", "disable_gemv",
                                            "sat_mode", "tf32_mode", "n_dir"};
  static constexpr StringRef kWithoutTf32[] = {"unit_flag_mode",
                                               "disable_gemv", "sat_mode",
                                               "n_dir"};
  return allowTf32Mode ? ArrayRef<StringRef>(kWithTf32)
                       : ArrayRef<StringRef>(kWithoutTf32);
}

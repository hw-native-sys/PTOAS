// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOStructuredAcc.cpp - shared structured acc-store helper definitions ===//
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Definitions of the shared structured acc-store helpers declared in
// structuredacc/VPTOStructuredAccInternal.h.
//===----------------------------------------------------------------------===//

#include "VPTOStructuredAccInternal.h"

namespace mlir::pto::structuredacc_detail {

using namespace mlir;
using namespace mlir::pto;

  FailureOr<AccStoreMode> parseAccStoreModeKeyword(StringRef keyword) {
    if (std::optional<AccStoreMode> mode = symbolizeAccStoreMode(keyword)) {
      return *mode;
    }
    return failure();
  }

  ParseResult parseAccStoreModeGroup(
      OpAsmParser &parser, StringRef &modeKeyword,
      SmallVectorImpl<OpAsmParser::UnresolvedOperand> &modeOperands) {
    if (parser.parseKeyword(&modeKeyword)) {
      return failure();
    }
    if (failed(parseAccStoreModeKeyword(modeKeyword))) {
      return parser.emitError(parser.getCurrentLocation(),
                              "expected one of 'nz2nd', 'nz2dn', or 'nz2nz'");
    }
    auto parseModeOperandWithParens = [&parser, &modeOperands]() -> ParseResult {
      OpAsmParser::UnresolvedOperand operand;
      if (parser.parseLParen() || parser.parseOperand(operand) || parser.parseRParen()) {
        return failure();
      }
      modeOperands.push_back(operand);
      return success();
    };
    auto parseModeOperandAfterLParen = [&parser, &modeOperands]() -> ParseResult {
      OpAsmParser::UnresolvedOperand operand;
      if (parser.parseOperand(operand) || parser.parseRParen()) {
        return failure();
      }
      modeOperands.push_back(operand);
      return success();
    };

    switch (*parseAccStoreModeKeyword(modeKeyword)) {
    case AccStoreMode::Nz2nd:
      return success();
    case AccStoreMode::Nz2dn:
      (void)parser.parseOptionalComma();
      if (succeeded(parser.parseOptionalKeyword("loop0_src_stride"))) {
        return parseModeOperandWithParens();
      }
      if (failed(parser.parseOptionalLParen())) {
        return success();
      }
      return parseModeOperandAfterLParen();
    case AccStoreMode::Nz2nz:
      (void)parser.parseOptionalComma();
      if (succeeded(parser.parseOptionalKeyword("split"))) {
        return parseModeOperandWithParens();
      }
      if (failed(parser.parseOptionalLParen())) {
        return success();
      }
      return parseModeOperandAfterLParen();
    }
    return success();
  }

  ParseResult
  parseAccStoreModeTypes(OpAsmParser &parser, StringRef modeKeyword,
                         SmallVectorImpl<Type> &modeTypes) {
    if (parser.parseKeyword(modeKeyword)) {
      return failure();
    }
    auto parseModeTypeWithParens = [&parser, &modeTypes]() -> ParseResult {
      Type modeType;
      if (parser.parseLParen() || parser.parseType(modeType) || parser.parseRParen()) {
        return failure();
      }
      modeTypes.push_back(modeType);
      return success();
    };
    auto parseModeTypeAfterLParen = [&parser, &modeTypes]() -> ParseResult {
      Type modeType;
      if (parser.parseType(modeType) || parser.parseRParen()) {
        return failure();
      }
      modeTypes.push_back(modeType);
      return success();
    };

    switch (*parseAccStoreModeKeyword(modeKeyword)) {
    case AccStoreMode::Nz2nd:
      return success();
    case AccStoreMode::Nz2dn:
      (void)parser.parseOptionalComma();
      if (succeeded(parser.parseOptionalKeyword("loop0_src_stride"))) {
        return parseModeTypeWithParens();
      }
      if (failed(parser.parseOptionalLParen())) {
        return success();
      }
      return parseModeTypeAfterLParen();
    case AccStoreMode::Nz2nz:
      (void)parser.parseOptionalComma();
      if (succeeded(parser.parseOptionalKeyword("split"))) {
        return parseModeTypeWithParens();
      }
      if (failed(parser.parseOptionalLParen())) {
        return success();
      }
      return parseModeTypeAfterLParen();
    }
    return success();
  }

  void printAccStoreModeGroup(OpAsmPrinter &printer,
                                                      AccStoreMode mode,
                                                      Value split,
                                                      Value loop0SrcStride) {
    printer << ", " << pto::stringifyAccStoreMode(mode);
    switch (mode) {
    case AccStoreMode::Nz2nd:
      return;
    case AccStoreMode::Nz2dn:
      if (loop0SrcStride) {
        printer << ", loop0_src_stride(" << loop0SrcStride << ")";
      }
      return;
    case AccStoreMode::Nz2nz:
      if (split) {
        printer << ", split(" << split << ")";
      }
      return;
    }
    llvm_unreachable("unexpected mte_l0c mode");
  }

  void printAccStoreModeTypes(OpAsmPrinter &printer,
                                                      AccStoreMode mode,
                                                      Type splitType,
                                                      Type loop0SrcStrideType) {
    printer << ", " << pto::stringifyAccStoreMode(mode);
    switch (mode) {
    case AccStoreMode::Nz2nd:
      return;
    case AccStoreMode::Nz2dn:
      if (loop0SrcStrideType) {
        printer << ", loop0_src_stride(" << loop0SrcStrideType << ")";
      }
      return;
    case AccStoreMode::Nz2nz:
      if (splitType) {
        printer << ", split(" << splitType << ")";
      }
      return;
    }
    llvm_unreachable("unexpected mte_l0c mode");
  }

  LogicalResult verifyAccStoreLikeModeOperands(
      Operation *op, AccStoreMode mode, Value split, Value loop0SrcStride,
      Value loop3Count, Value loop3SrcStride, Value loop3DstStride,
      StringRef nz2ndSplitError, StringRef nz2ndLoop0Error,
      StringRef nz2dnSplitError, StringRef nz2nzLoop0Error,
      StringRef nz2nzLoop3Error) {
    bool hasLoop3Count = static_cast<bool>(loop3Count);
    bool hasLoop3SrcStride = static_cast<bool>(loop3SrcStride);
    bool hasLoop3DstStride = static_cast<bool>(loop3DstStride);
    if ((hasLoop3Count != hasLoop3SrcStride) ||
        (hasLoop3Count != hasLoop3DstStride)) {
      return op->emitOpError(
          "requires loop3 count, src stride, and dst stride to appear together");
    }

    switch (mode) {
    case AccStoreMode::Nz2nd:
      if (split) {
        return op->emitOpError(nz2ndSplitError);
      }
      if (loop0SrcStride) {
        return op->emitOpError(nz2ndLoop0Error);
      }
      return success();
    case AccStoreMode::Nz2dn:
      if (split) {
        return op->emitOpError(nz2dnSplitError);
      }
      return success();
    case AccStoreMode::Nz2nz:
      if (loop0SrcStride) {
        return op->emitOpError(nz2nzLoop0Error);
      }
      if (loop3Count) {
        return op->emitOpError(nz2nzLoop3Error);
      }
      return success();
    }
    llvm_unreachable("unexpected mte_l0c mode");
  }

  

  bool isStructuredAccStoreVectorQuantMode(AccStoreQuantPreMode mode) {
    switch (mode) {
    case AccStoreQuantPreMode::QF322HIF8PreVec:
    case AccStoreQuantPreMode::QF322HIF8PreHybridVec:
    case AccStoreQuantPreMode::DEQS32IntVec:
    case AccStoreQuantPreMode::REQ8Vec:
    case AccStoreQuantPreMode::DEQF16Vec:
    case AccStoreQuantPreMode::QF322FP8PreVec:
    case AccStoreQuantPreMode::QF322F32PreVec:
    case AccStoreQuantPreMode::QF162B8PreVec:
    case AccStoreQuantPreMode::QF162S4PreVec:
    case AccStoreQuantPreMode::REQ4Vec:
    case AccStoreQuantPreMode::QF322B8PreVec:
    case AccStoreQuantPreMode::QF322S4PreVec:
    case AccStoreQuantPreMode::DEQS16Vec:
    case AccStoreQuantPreMode::QF162S16PreVec:
    case AccStoreQuantPreMode::QF322F16PreVec:
    case AccStoreQuantPreMode::QF322BF16PreVec:
    case AccStoreQuantPreMode::QS322BF16PreVec:
      return true;
    default:
      return false;
    }
  }

  bool isStructuredAccStoreScalingPayload(Value value) {
    auto ptrType = dyn_cast_or_null<PtrType>(value.getType());
    return ptrType &&
           ptrType.getMemorySpace().getAddressSpace() == AddressSpace::SCALING;
  }

  bool isStructuredAccStoreScalingPayloadType(Type type) {
    auto ptrType = dyn_cast_or_null<PtrType>(type);
    return ptrType &&
           ptrType.getMemorySpace().getAddressSpace() == AddressSpace::SCALING;
  }

  Type getStructuredAccStoreScalingElementType(Value value) {
    auto ptrType = dyn_cast_or_null<PtrType>(value.getType());
    if (!ptrType ||
        ptrType.getMemorySpace().getAddressSpace() != AddressSpace::SCALING) {
      return {};
    }
    return ptrType.getElementType();
  }

  bool isStructuredAccStoreIntegerPayload(Value value) {
    return value.getType().isSignlessInteger();
  }

  bool isStructuredAccStoreClipPayloadForUInt8(Type type) {
    auto intType = dyn_cast<IntegerType>(type);
    if (!intType || intType.getWidth() != mlir::pto::kValue16) {
      return false;
    }
    return intType.isUnsigned() || intType.isSignless();
  }

  bool isStructuredAccStoreClipPayloadForSignedInt(Type type) {
    auto intType = dyn_cast<IntegerType>(type);
    if (!intType) {
      return false;
    }
    unsigned width = intType.getWidth();
    if (width != mlir::pto::kValue4 && width != 8 && width != 16) {
      return false;
    }
    return intType.isSigned() || intType.isSignless();
  }

  bool isStructuredAccStoreFloatScalarPayloadType(Type type) {
    return type.isF16() || type.isF32() || type.isBF16();
  }

  bool isStructuredAccStoreFloatScalarPayload(Value value) {
    return isStructuredAccStoreFloatScalarPayloadType(value.getType());
  }

  bool isStructuredAccStoreIntegerPayloadType(Type type) {
    return type.isSignlessInteger();
  }

  bool isStructuredAccStoreClipSupportedElementType(Type type) {
    if (auto floatType = dyn_cast<FloatType>(type)) {
      return floatType.isF16();
    }
    auto intType = dyn_cast<IntegerType>(type);
    if (!intType) {
      return false;
    }
    if (intType.isUnsignedInteger(mlir::pto::kValue8)) {
      return true;
    }
    if (intType.isSignlessInteger(mlir::pto::kValue4) || intType.isSignlessInteger(mlir::pto::kValue8) ||
        intType.isSignlessInteger(mlir::pto::kValue16)) {
      return true;
    }
    if (intType.isSignedInteger(mlir::pto::kValue4) || intType.isSignedInteger(mlir::pto::kValue8) ||
        intType.isSignedInteger(mlir::pto::kValue16)) {
      return true;
    }
    return false;
  }

  LogicalResult verifyStructuredAccStoreClipPayload(Operation *op,
                                                          Type destinationElementType,
                                                          Value clipValue) {
    if (!clipValue) {
      return success();
    }

    Type clipType = clipValue.getType();
    if (destinationElementType.isF16()) {
      if (!clipType.isF16()) {
        return op->emitOpError("clip for f16 destination requires f16 payload");
      }
      return success();
    }

    auto intType = dyn_cast<IntegerType>(destinationElementType);
    if (!intType) {
      return op->emitOpError()
             << "clip requires destination element type to be f16, ui8, or signed 4/8/16-bit integer, got "
             << destinationElementType;
    }

    if (intType.isUnsignedInteger(mlir::pto::kValue8)) {
      if (!isStructuredAccStoreClipPayloadForUInt8(clipType)) {
        return op->emitOpError("clip for ui8 destination requires ui16/signless i16 payload");
      }
      return success();
    }

    if (intType.isSignlessInteger(mlir::pto::kValue4) ||
        intType.isSignlessInteger(mlir::pto::kValue8) ||
        intType.isSignlessInteger(mlir::pto::kValue16) ||
        intType.isSignedInteger(mlir::pto::kValue4) ||
        intType.isSignedInteger(mlir::pto::kValue8) ||
        intType.isSignedInteger(mlir::pto::kValue16)) {
      if (!isStructuredAccStoreClipPayloadForSignedInt(clipType)) {
        return op->emitOpError("clip for signed 4/8/16-bit destination requires signed/signless i4/i8/i16 payload");
      }
      return success();
    }

    return op->emitOpError()
           << "clip requires destination element type to be f16, ui8, or signed 4/8/16-bit integer, got "
           << destinationElementType;
  }

  bool isStructuredAccStoreFloatPreQuantMode(AccStoreQuantPreMode mode) {
    static constexpr AccStoreQuantPreMode kFloatModes[] = {
        AccStoreQuantPreMode::F32F16,
        AccStoreQuantPreMode::QF322HIF8PreVec,
        AccStoreQuantPreMode::QF322HIF8PreScalar,
        AccStoreQuantPreMode::QF322HIF8PreHybridVec,
        AccStoreQuantPreMode::QF322HIF8PreHybridScalar,
        AccStoreQuantPreMode::QF322FP8PreVec,
        AccStoreQuantPreMode::QF322FP8PreScalar,
        AccStoreQuantPreMode::QF322F32PreVec,
        AccStoreQuantPreMode::QF322F32PreScalar,
        AccStoreQuantPreMode::F32BF16,
        AccStoreQuantPreMode::QF162B8PreVec,
        AccStoreQuantPreMode::QF162B8PreScalar,
        AccStoreQuantPreMode::QF162S4PreVec,
        AccStoreQuantPreMode::QF162S4PreScalar,
        AccStoreQuantPreMode::QF322B8PreVec,
        AccStoreQuantPreMode::QF322B8PreScalar,
        AccStoreQuantPreMode::QF322S4PreVec,
        AccStoreQuantPreMode::QF322S4PreScalar,
        AccStoreQuantPreMode::QF322F16PreVec,
        AccStoreQuantPreMode::QF322F16PreScalar,
        AccStoreQuantPreMode::QF322BF16PreVec,
        AccStoreQuantPreMode::QF322BF16PreScalar,
    };
    return llvm::is_contained(kFloatModes, mode);
  }

  bool isStructuredAccStoreInt32PreQuantMode(AccStoreQuantPreMode mode) {
    switch (mode) {
    case AccStoreQuantPreMode::DEQS32IntVec:
    case AccStoreQuantPreMode::DEQS32IntScalar:
    case AccStoreQuantPreMode::REQ8Vec:
    case AccStoreQuantPreMode::REQ8Scalar:
    case AccStoreQuantPreMode::DEQF16Vec:
    case AccStoreQuantPreMode::DEQF16Scalar:
    case AccStoreQuantPreMode::DEQS16Vec:
    case AccStoreQuantPreMode::DEQS16Scalar:
    case AccStoreQuantPreMode::QF162S16PreVec:
    case AccStoreQuantPreMode::QF162S16PreScalar:
    case AccStoreQuantPreMode::QS322BF16PreVec:
    case AccStoreQuantPreMode::QS322BF16PreScalar:
      return true;
    default:
      return false;
    }
  }

  

  StructuredAccStoreDestinationFamily
  getStructuredAccStorePreQuantDestinationFamily(AccStoreQuantPreMode mode) {
    struct Entry {
      AccStoreQuantPreMode mode;
      StructuredAccStoreDestinationFamily family;
    };
    static constexpr Entry kEntries[] = {
        {AccStoreQuantPreMode::F32F16, StructuredAccStoreDestinationFamily::F16},
        {AccStoreQuantPreMode::QF322F16PreVec, StructuredAccStoreDestinationFamily::F16},
        {AccStoreQuantPreMode::QF322F16PreScalar, StructuredAccStoreDestinationFamily::F16},
        {AccStoreQuantPreMode::DEQF16Vec, StructuredAccStoreDestinationFamily::F16},
        {AccStoreQuantPreMode::DEQF16Scalar, StructuredAccStoreDestinationFamily::F16},
        {AccStoreQuantPreMode::F32BF16, StructuredAccStoreDestinationFamily::BF16},
        {AccStoreQuantPreMode::QF322BF16PreVec, StructuredAccStoreDestinationFamily::BF16},
        {AccStoreQuantPreMode::QF322BF16PreScalar, StructuredAccStoreDestinationFamily::BF16},
        {AccStoreQuantPreMode::QS322BF16PreVec, StructuredAccStoreDestinationFamily::BF16},
        {AccStoreQuantPreMode::QS322BF16PreScalar, StructuredAccStoreDestinationFamily::BF16},
        {AccStoreQuantPreMode::QF322F32PreVec, StructuredAccStoreDestinationFamily::F32},
        {AccStoreQuantPreMode::QF322F32PreScalar, StructuredAccStoreDestinationFamily::F32},
        {AccStoreQuantPreMode::QF322HIF8PreVec, StructuredAccStoreDestinationFamily::FP8},
        {AccStoreQuantPreMode::QF322HIF8PreScalar, StructuredAccStoreDestinationFamily::FP8},
        {AccStoreQuantPreMode::QF322HIF8PreHybridVec, StructuredAccStoreDestinationFamily::FP8},
        {AccStoreQuantPreMode::QF322HIF8PreHybridScalar, StructuredAccStoreDestinationFamily::FP8},
        {AccStoreQuantPreMode::QF322FP8PreVec, StructuredAccStoreDestinationFamily::FP8},
        {AccStoreQuantPreMode::QF322FP8PreScalar, StructuredAccStoreDestinationFamily::FP8},
        {AccStoreQuantPreMode::DEQS32IntVec, StructuredAccStoreDestinationFamily::I32},
        {AccStoreQuantPreMode::DEQS32IntScalar, StructuredAccStoreDestinationFamily::I32},
        {AccStoreQuantPreMode::QF162S16PreVec, StructuredAccStoreDestinationFamily::I16},
        {AccStoreQuantPreMode::QF162S16PreScalar, StructuredAccStoreDestinationFamily::I16},
        {AccStoreQuantPreMode::DEQS16Vec, StructuredAccStoreDestinationFamily::I16},
        {AccStoreQuantPreMode::DEQS16Scalar, StructuredAccStoreDestinationFamily::I16},
        {AccStoreQuantPreMode::QF162B8PreVec, StructuredAccStoreDestinationFamily::I8},
        {AccStoreQuantPreMode::QF162B8PreScalar, StructuredAccStoreDestinationFamily::I8},
        {AccStoreQuantPreMode::REQ8Vec, StructuredAccStoreDestinationFamily::I8},
        {AccStoreQuantPreMode::REQ8Scalar, StructuredAccStoreDestinationFamily::I8},
        {AccStoreQuantPreMode::QF322B8PreVec, StructuredAccStoreDestinationFamily::I8},
        {AccStoreQuantPreMode::QF322B8PreScalar, StructuredAccStoreDestinationFamily::I8},
        {AccStoreQuantPreMode::QF162S4PreVec, StructuredAccStoreDestinationFamily::I4},
        {AccStoreQuantPreMode::QF162S4PreScalar, StructuredAccStoreDestinationFamily::I4},
        {AccStoreQuantPreMode::REQ4Vec, StructuredAccStoreDestinationFamily::I4},
        {AccStoreQuantPreMode::REQ4Scalar, StructuredAccStoreDestinationFamily::I4},
        {AccStoreQuantPreMode::QF322S4PreVec, StructuredAccStoreDestinationFamily::I4},
        {AccStoreQuantPreMode::QF322S4PreScalar, StructuredAccStoreDestinationFamily::I4},
    };
    auto it = llvm::find_if(kEntries, [mode](const Entry &entry) {
      return entry.mode == mode;
    });
    return it != std::end(kEntries)
               ? it->family
               : StructuredAccStoreDestinationFamily::Any;
  }

  bool isStructuredAccStoreDestinationFamily(
      Type type, StructuredAccStoreDestinationFamily family) {
    switch (family) {
    case StructuredAccStoreDestinationFamily::Any:
      return true;
    case StructuredAccStoreDestinationFamily::F32:
      return type.isF32();
    case StructuredAccStoreDestinationFamily::F16:
      return type.isF16();
    case StructuredAccStoreDestinationFamily::BF16:
      return type.isBF16();
    case StructuredAccStoreDestinationFamily::I32:
      if (auto intType = dyn_cast<IntegerType>(type)) {
        return intType.getWidth() == mlir::pto::kValue32;
      }
      return false;
    case StructuredAccStoreDestinationFamily::I16:
      if (auto intType = dyn_cast<IntegerType>(type)) {
        return intType.getWidth() == mlir::pto::kValue16 && !intType.isUnsigned();
      }
      return false;
    case StructuredAccStoreDestinationFamily::I8:
      if (auto intType = dyn_cast<IntegerType>(type)) {
        return intType.getWidth() == mlir::pto::kValue8;
      }
      return false;
    case StructuredAccStoreDestinationFamily::I4:
      if (auto intType = dyn_cast<IntegerType>(type)) {
        return intType.getWidth() == mlir::pto::kValue4 && !intType.isUnsigned();
      }
      return false;
    case StructuredAccStoreDestinationFamily::FP8:
      return pto::isPTOFloat8Type(type) || pto::isPTOHiFloat8Type(type) ||
             pto::isPTOHiFloat8x2Type(type);
    }
    llvm_unreachable("unknown acc-store destination family");
  }

  ParseResult parseStructuredAccStoreUnitFlag(OpAsmParser &parser,
                                                     StructuredAccStoreAsmState &state) {
    if (state.unitFlag) {
      return parser.emitError(parser.getCurrentLocation(), "duplicate unit_flag clause");
    }
    StringRef keyword;
    if (parser.parseLParen() || parser.parseKeyword(&keyword) || parser.parseRParen()) {
      return failure();
    }
    if (keyword == "check_only") {
      state.unitFlag = AccStoreUnitFlagCtrl::CheckOnly;
    }
    else if (keyword == "check_and_clear") {
      state.unitFlag = AccStoreUnitFlagCtrl::CheckAndClear;
    }
    else {
      return parser.emitError(parser.getCurrentLocation(),
                              "expected 'check_only' or 'check_and_clear'");
  }
    return success();
  }

  ParseResult parseStructuredAccStorePreQuant(
      OpAsmParser &parser, StructuredAccStoreAsmState &state) {
    if (state.preQuantMode) {
      return parser.emitError(parser.getCurrentLocation(), "duplicate pre_quant clause");
    }
    OpAsmParser::UnresolvedOperand payload;
    StringRef modeKeyword;
    if (parser.parseLParen() || parser.parseOperand(payload) || parser.parseComma() ||
        parser.parseKeyword("mode") || parser.parseEqual() ||
        parser.parseKeyword(&modeKeyword) || parser.parseRParen()) {
      return failure();
    }
    auto mode = symbolizeAccStoreQuantPreMode(modeKeyword);
    if (!mode) {
      return parser.emitError(parser.getCurrentLocation(), "invalid pre_quant mode");
    }
    state.preQuantOperands.push_back(payload);
    state.preQuantMode = *mode;
    return success();
  }

  ParseResult parseStructuredAccStorePreRelu(
      OpAsmParser &parser, StructuredAccStoreAsmState &state) {
    if (state.preReluMode) {
      return parser.emitError(parser.getCurrentLocation(), "duplicate pre_relu clause");
    }
    StringRef modeKeyword;
    bool hasPayload = false;
    OpAsmParser::UnresolvedOperand payload;
    if (parser.parseLParen()) {
      return failure();
    }
    if (failed(parser.parseOptionalKeyword("mode"))) {
      hasPayload = true;
      if (parser.parseOperand(payload) || parser.parseComma() ||
          parser.parseKeyword("mode")) {
        return failure();
      }
    }
    if (parser.parseEqual() || parser.parseKeyword(&modeKeyword)) {
      return failure();
    }
    auto mode = symbolizeReluPreMode(modeKeyword);
    if (!mode) {
      return parser.emitError(parser.getCurrentLocation(), "invalid pre_relu mode");
    }
    if (succeeded(parser.parseOptionalComma())) {
      if (parser.parseKeyword("clip") || parser.parseEqual()) {
        return failure();
      }
      if (!state.clipValueOperands.empty()) {
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate clip payload in pre_relu clause");
      }
      OpAsmParser::UnresolvedOperand clipValue;
      if (parser.parseOperand(clipValue)) {
        return failure();
      }
      state.clipValueOperands.push_back(clipValue);
    }
    if (parser.parseRParen()) {
      return failure();
    }

    if (hasPayload) {
      state.preReluOperands.push_back(payload);
    }
    state.preReluMode = *mode;
    return success();
  }

  ParseResult parseStructuredAccStoreLayout(
      OpAsmParser &parser, StructuredAccStoreAsmState &state, StringRef keyword) {
    auto mode = parseAccStoreModeKeyword(keyword);
    if (failed(mode)) {
      return parser.emitError(parser.getCurrentLocation(),
                              "expected one of 'nz2nd', 'nz2dn', or 'nz2nz'");
    }
    if (state.mode) {
      return parser.emitError(parser.getCurrentLocation(), "duplicate layout clause");
    }
    state.mode = *mode;
    if (*mode == AccStoreMode::Nz2dn) {
      if (succeeded(parser.parseOptionalLParen())) {
        OpAsmParser::UnresolvedOperand operand;
        if (parser.parseOperand(operand) || parser.parseRParen()) {
          return failure();
        }
        state.loop0SrcStrideOperands.push_back(operand);
      }
    } else if (*mode == AccStoreMode::Nz2nz) {
      if (succeeded(parser.parseOptionalLParen())) {
        OpAsmParser::UnresolvedOperand operand;
        if (parser.parseOperand(operand) || parser.parseRParen()) {
          return failure();
        }
        state.splitOperands.push_back(operand);
      }
    }
    return success();
  }

  ParseResult parseStructuredAccStoreLoop3(
      OpAsmParser &parser, StructuredAccStoreAsmState &state) {
    if (!state.loop3CountOperands.empty()) {
      return parser.emitError(parser.getCurrentLocation(), "duplicate loop3 clause");
    }
    OpAsmParser::UnresolvedOperand count;
    OpAsmParser::UnresolvedOperand srcStride;
    OpAsmParser::UnresolvedOperand dstStride;
    if (parser.parseLParen() || parser.parseOperand(count) || parser.parseComma() ||
        parser.parseOperand(srcStride) || parser.parseComma() ||
        parser.parseOperand(dstStride) || parser.parseRParen()) {
      return failure();
    }
    state.loop3CountOperands.push_back(count);
    state.loop3SrcStrideOperands.push_back(srcStride);
    state.loop3DstStrideOperands.push_back(dstStride);
    return success();
  }

  ParseResult parseStructuredAccStoreAtomic(
      OpAsmParser &parser, StructuredAccStoreAsmState &state) {
    if (state.atomicType || state.atomicOp) {
      return parser.emitError(parser.getCurrentLocation(), "duplicate atomic clause");
    }
    StringRef typeKeyword;
    StringRef opKeyword;
    if (parser.parseLParen() || parser.parseKeyword("type") || parser.parseEqual() ||
        parser.parseKeyword(&typeKeyword) || parser.parseComma() ||
        parser.parseKeyword("op") || parser.parseEqual() ||
        parser.parseKeyword(&opKeyword) || parser.parseRParen()) {
      return failure();
    }
    auto type = symbolizeAccStoreAtomicType(typeKeyword);
    auto op = symbolizeAccStoreAtomicOp(opKeyword);
    if (!type) {
      return parser.emitError(parser.getCurrentLocation(), "invalid atomic type");
    }
    if (!op) {
      return parser.emitError(parser.getCurrentLocation(), "invalid atomic op");
    }
    state.atomicType = *type;
    state.atomicOp = *op;
    return success();
  }

  bool classifyStructuredAccStoreClause(
      StringRef keyword, StructuredAccStoreClauseKind &kind) {
    if (keyword == "unit_flag") {
      kind = StructuredAccStoreClauseKind::UnitFlag;
    } else if (keyword == "pre_quant") {
      kind = StructuredAccStoreClauseKind::PreQuant;
    } else if (keyword == "pre_relu") {
      kind = StructuredAccStoreClauseKind::PreRelu;
    } else if (keyword == "nz2nd" || keyword == "nz2dn" || keyword == "nz2nz") {
      kind = StructuredAccStoreClauseKind::Layout;
    } else if (keyword == "loop3") {
      kind = StructuredAccStoreClauseKind::Loop3;
    } else if (keyword == "sat" || keyword == "nosat") {
      kind = StructuredAccStoreClauseKind::Sat;
    } else if (keyword == "atomic") {
      kind = StructuredAccStoreClauseKind::Atomic;
    } else {
      return false;
    }
    return true;
  }

  ParseResult parseStructuredAccStoreSatClause(
      OpAsmParser &parser, StructuredAccStoreAsmState &state,
      StringRef keyword) {
    if (state.satMode) {
      return parser.emitError(parser.getCurrentLocation(), "duplicate sat/nosat clause");
    }
    if (keyword == "nosat") {
      state.satMode = AccStoreSatMode::NoSat;
      return success();
    }
    if (succeeded(parser.parseOptionalLParen())) {
      StringRef satOption;
      if (parser.parseKeyword(&satOption) || satOption != "preserve_nan") {
        return parser.emitError(parser.getCurrentLocation(),
                                "expected preserve_nan");
      }
      if (parser.parseRParen()) {
        return failure();
      }
      state.satMode = AccStoreSatMode::SatPreserveNan;
    } else {
      state.satMode = AccStoreSatMode::Sat;
    }
    return success();
  }

  ParseResult parseStructuredAccStoreClauseBody(
      OpAsmParser &parser, StructuredAccStoreAsmState &state,
      StructuredAccStoreClauseKind kind, StringRef keyword) {
    switch (kind) {
    case StructuredAccStoreClauseKind::UnitFlag:
      return parseStructuredAccStoreUnitFlag(parser, state);
    case StructuredAccStoreClauseKind::PreQuant:
      return parseStructuredAccStorePreQuant(parser, state);
    case StructuredAccStoreClauseKind::PreRelu:
      return parseStructuredAccStorePreRelu(parser, state);
    case StructuredAccStoreClauseKind::Layout:
      return parseStructuredAccStoreLayout(parser, state, keyword);
    case StructuredAccStoreClauseKind::Loop3:
      return parseStructuredAccStoreLoop3(parser, state);
    case StructuredAccStoreClauseKind::Atomic:
      return parseStructuredAccStoreAtomic(parser, state);
    case StructuredAccStoreClauseKind::Sat:
      return success();
    }
    return success();
  }

  ParseResult parseStructuredOptionalType(OpAsmParser &parser,
                                                 SmallVectorImpl<Type> &types) {
    Type type;
    if (parser.parseType(type)) {
      return failure();
    }
    types.push_back(type);
    return success();
  }

  // Validate pre_quant mode against source/destination element types.
  LogicalResult verifyStructuredPreQuant(
      Operation *op, Value preQuant, Type sourceElementType,
      Type destinationElementType,
      std::optional<AccStoreQuantPreMode> preQuantMode) {
    if (static_cast<bool>(preQuant) != static_cast<bool>(preQuantMode)) {
      return op->emitOpError("pre_quant requires payload and mode together");
    }
    // The no_convert keyword carries no quantization parameters; the
    // syntactic payload operand is ignored for compatibility with the
    // structured pre_quant clause form.
    if (!preQuantMode || *preQuantMode == AccStoreQuantPreMode::NoConvert) {
      return success();
    }
    if (isStructuredAccStoreVectorQuantMode(*preQuantMode)) {
      if (!isStructuredAccStoreScalingPayload(preQuant)) {
        return op->emitOpError("vector pre_quant mode requires scaling pointer payload");
      }
      if (!isStructuredAccStoreFloatScalarPayloadType(
              getStructuredAccStoreScalingElementType(preQuant))) {
        return op->emitOpError(
            "vector pre_quant mode requires scaling pointer element type to be f16, bf16, or f32");
      }
    } else if (!isStructuredAccStoreFloatScalarPayload(preQuant)) {
      return op->emitOpError(
          "scalar pre_quant mode requires f16/bf16/f32 payload");
    }
    auto emitIncompatibleQuantModeError =
        [op, preQuantMode, sourceElementType, destinationElementType]() -> LogicalResult {
      return op->emitOpError()
             << "pre_quant mode " << stringifyAccStoreQuantPreMode(*preQuantMode)
             << " is incompatible with source element type " << sourceElementType
             << " and destination element type " << destinationElementType;
    };
    if (*preQuantMode != AccStoreQuantPreMode::NoConvert) {
      if (isa<Float32Type>(sourceElementType)) {
        if (!isStructuredAccStoreFloatPreQuantMode(*preQuantMode)) {
          return emitIncompatibleQuantModeError();
        }
      } else if (sourceElementType.isSignlessInteger(mlir::pto::kValue32)) {
        if (!isStructuredAccStoreInt32PreQuantMode(*preQuantMode)) {
          return emitIncompatibleQuantModeError();
        }
      } else {
        return op->emitOpError()
               << "pre_quant requires source element type to be f32 or i32, got "
               << sourceElementType;
      }
      StructuredAccStoreDestinationFamily destinationFamily =
          getStructuredAccStorePreQuantDestinationFamily(*preQuantMode);
      if (!isStructuredAccStoreDestinationFamily(destinationElementType,
                                                 destinationFamily)) {
        return emitIncompatibleQuantModeError();
      }
    }
    return success();
  }

  // Validate pre_relu mode and payload.
  LogicalResult verifyStructuredPreRelu(Operation *op, Value preRelu,
                                               Value clipValue,
                                               std::optional<ReluPreMode> preReluMode) {
    if (!preReluMode) {
      if (preRelu) {
        return op->emitOpError("pre_relu payload requires pre_relu mode");
      }
      if (clipValue) {
        return op->emitOpError("clip requires pre_relu clause");
      }
      return success();
    }
    switch (*preReluMode) {
    case ReluPreMode::NoRelu:
      if (preRelu) {
        return op->emitOpError("mode does not accept pre_relu payload");
      }
      break;
    case ReluPreMode::NormalRelu:
      if (preRelu) {
        return op->emitOpError("mode does not accept pre_relu payload");
      }
      break;
    case ReluPreMode::ScalarRelu:
      if (!preRelu) {
        return op->emitOpError("scalar_relu requires payload");
      }
      if (!isStructuredAccStoreFloatScalarPayload(preRelu)) {
        return op->emitOpError("scalar_relu requires f16/bf16/f32 payload");
      }
      break;
    case ReluPreMode::VectorRelu:
      if (!preRelu) {
        return op->emitOpError("vector_relu requires payload");
      }
      if (!isStructuredAccStoreScalingPayload(preRelu)) {
        return op->emitOpError("vector_relu requires scaling pointer payload");
      }
      if (!isStructuredAccStoreFloatScalarPayloadType(
              getStructuredAccStoreScalingElementType(preRelu))) {
        return op->emitOpError(
            "vector_relu requires scaling pointer element type to be f16, bf16, or f32");
      }
      break;
    case ReluPreMode::Pwl:
      return op->emitOpError("pwl is not supported for target_profile mte_l0c_l1");
    }
    return success();
  }

  // unit_flag must be off for nz2dn when loop0_src_stride is not a constant 1.
  LogicalResult verifyAccStoreUnitFlagNz2dn(
      Operation *op, std::optional<AccStoreUnitFlagCtrl> unitFlag,
      Value loop0SrcStride) {
    if (!unitFlag || *unitFlag == AccStoreUnitFlagCtrl::Off) {
      return success();
    }
    APInt loop0Value;
    if (!matchPattern(loop0SrcStride, m_ConstantInt(&loop0Value)) ||
        !loop0Value.isOne()) {
      return op->emitOpError(
          "unit_flag must be off when nz2dn loop0_src_stride is not 1");
    }
    return success();
  }

  // Validate mode (nz2nd/nz2dn/nz2nz) and related attributes.
  LogicalResult verifyStructuredAccStoreMode(
      Operation *op, Value split, Value loop0SrcStride, Value loop3Count,
      Type destinationElementType, std::optional<AccStoreUnitFlagCtrl> unitFlag,
      std::optional<AccStoreMode> mode) {
    if (!mode) {
      if (split) { return op->emitOpError("split requires nz2nz"); }
      if (loop0SrcStride) { return op->emitOpError("loop0_src_stride requires nz2dn"); }
      if (loop3Count) { return op->emitOpError("loop3 requires nz2nd or nz2dn"); }
      return success();
    }
    switch (*mode) {
    case AccStoreMode::Nz2nd:
      if (split) {
        return op->emitOpError("nz2nd does not accept split");
      }
      if (loop0SrcStride) {
        return op->emitOpError("nz2nd does not accept loop0_src_stride");
      }
      break;
    case AccStoreMode::Nz2dn: {
      if (!loop0SrcStride) {
        return op->emitOpError("nz2dn requires loop0_src_stride");
      }
      if (split) {
        return op->emitOpError("nz2dn does not accept split");
      }
      if (failed(verifyAccStoreUnitFlagNz2dn(op, unitFlag, loop0SrcStride))) {
        return failure();
      }
      break;
    }
    case AccStoreMode::Nz2nz:
      if (loop0SrcStride) {
        return op->emitOpError("nz2nz does not accept loop0_src_stride");
      }
      if (loop3Count) {
        return op->emitOpError("loop3 requires nz2nd or nz2dn");
      }
      if (!isa<FloatType>(destinationElementType) ||
          !cast<FloatType>(destinationElementType).isF32()) {
        return op->emitOpError("nz2nz requires destination element type to be f32");
      }
      break;
    }
    return success();
  }

  void printStructuredAccStoreMode(OpAsmPrinter &printer,
                                          AccStoreMode mode, Value split,
                                          Value loop0SrcStride) {
    switch (mode) {
    case AccStoreMode::Nz2nd:
      printer << ", nz2nd";
      break;
    case AccStoreMode::Nz2dn:
      printer << ", nz2dn";
      if (loop0SrcStride) {
        printer << "(" << loop0SrcStride << ")";
      }
      break;
    case AccStoreMode::Nz2nz:
      printer << ", nz2nz";
      if (split) {
        printer << "(" << split << ")";
      }
      break;
    }
  }

  void printStructuredAccStoreSatMode(OpAsmPrinter &printer,
                                             AccStoreSatMode satMode) {
    switch (satMode) {
    case AccStoreSatMode::Sat:
      printer << ", sat";
      break;
    case AccStoreSatMode::NoSat:
      printer << ", nosat";
      break;
    case AccStoreSatMode::SatPreserveNan:
      printer << ", sat(preserve_nan)";
      break;
    }
  }

  ParseResult parseStructuredAccStoreOptionalType(
      OpAsmParser &parser, bool hasOperand, SmallVectorImpl<Type> &types) {
    if (!hasOperand) {
      return success();
    }
    if (parser.parseComma() || parseStructuredOptionalType(parser, types)) {
      return failure();
    }
    return success();
  }

} // namespace mlir::pto::structuredacc_detail

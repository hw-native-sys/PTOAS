// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOStructuredAccInternal.h - shared structured acc-store helpers --===//
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Shared structured acc-store helpers used by the L0c store ops. The
// definitions live in VPTOStructuredAcc.cpp inside a detail namespace so
// the original unqualified MLIR/LLVM names keep resolving.
// Internal to lib/PTO/IR/VPTO/structuredacc; not installed.
//===----------------------------------------------------------------------===//

#ifndef PTO_IR_VPTO_STRUCTUREDACC_INTERNAL_H
#define PTO_IR_VPTO_STRUCTUREDACC_INTERNAL_H

#include "VPTOInternal.h"

namespace mlir::pto::structuredacc_detail {

using namespace mlir;
using namespace mlir::pto;

enum class StructuredAccStoreClauseKind {
  UnitFlag = 0,
  PreQuant = 1,
  PreRelu = 2,
  Layout = 3,
  Loop3 = 4,
  Sat = 5,
  Atomic = 6
};

enum class StructuredAccStoreDestinationFamily {
  Any,
  F32,
  F16,
  BF16,
  I32,
  I16,
  I8,
  I4,
  FP8
};

  FailureOr<AccStoreMode> parseAccStoreModeKeyword(StringRef keyword);
  ParseResult parseAccStoreModeGroup( OpAsmParser &parser, StringRef &modeKeyword, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &modeOperands);
  ParseResult parseAccStoreModeTypes(OpAsmParser &parser, StringRef modeKeyword, SmallVectorImpl<Type> &modeTypes);
  void printAccStoreModeGroup(OpAsmPrinter &printer, AccStoreMode mode, Value split, Value loop0SrcStride);
  void printAccStoreModeTypes(OpAsmPrinter &printer, AccStoreMode mode, Type splitType, Type loop0SrcStrideType);
  LogicalResult verifyAccStoreLikeModeOperands( Operation *op, AccStoreMode mode, Value split, Value loop0SrcStride, Value loop3Count, Value loop3SrcStride, Value loop3DstStride, StringRef nz2ndSplitError, StringRef nz2ndLoop0Error, StringRef nz2dnSplitError, StringRef nz2nzLoop0Error, StringRef nz2nzLoop3Error);
  bool isStructuredAccStoreVectorQuantMode(AccStoreQuantPreMode mode);
  bool isStructuredAccStoreScalingPayload(Value value);
  bool isStructuredAccStoreScalingPayloadType(Type type);
  Type getStructuredAccStoreScalingElementType(Value value);
  bool isStructuredAccStoreIntegerPayload(Value value);
  bool isStructuredAccStoreClipPayloadForUInt8(Type type);
  bool isStructuredAccStoreClipPayloadForSignedInt(Type type);
  bool isStructuredAccStoreFloatScalarPayloadType(Type type);
  bool isStructuredAccStoreFloatScalarPayload(Value value);
  bool isStructuredAccStoreIntegerPayloadType(Type type);
  bool isStructuredAccStoreClipSupportedElementType(Type type);
  LogicalResult verifyStructuredAccStoreClipPayload(Operation *op, Type destinationElementType, Value clipValue);
  bool isStructuredAccStoreFloatPreQuantMode(AccStoreQuantPreMode mode);
  bool isStructuredAccStoreInt32PreQuantMode(AccStoreQuantPreMode mode);
  StructuredAccStoreDestinationFamily getStructuredAccStorePreQuantDestinationFamily(AccStoreQuantPreMode mode);
  bool isStructuredAccStoreDestinationFamily( Type type, StructuredAccStoreDestinationFamily family);
  ParseResult parseStructuredAccStoreUnitFlag(OpAsmParser &parser, StructuredAccStoreAsmState &state);
  ParseResult parseStructuredAccStorePreQuant( OpAsmParser &parser, StructuredAccStoreAsmState &state);
  ParseResult parseStructuredAccStorePreRelu( OpAsmParser &parser, StructuredAccStoreAsmState &state);
  ParseResult parseStructuredAccStoreLayout( OpAsmParser &parser, StructuredAccStoreAsmState &state, StringRef keyword);
  ParseResult parseStructuredAccStoreLoop3( OpAsmParser &parser, StructuredAccStoreAsmState &state);
  ParseResult parseStructuredAccStoreAtomic( OpAsmParser &parser, StructuredAccStoreAsmState &state);
  bool classifyStructuredAccStoreClause( StringRef keyword, StructuredAccStoreClauseKind &kind);
  ParseResult parseStructuredAccStoreSatClause( OpAsmParser &parser, StructuredAccStoreAsmState &state, StringRef keyword);
  ParseResult parseStructuredAccStoreClauseBody( OpAsmParser &parser, StructuredAccStoreAsmState &state, StructuredAccStoreClauseKind kind, StringRef keyword);
  ParseResult parseStructuredOptionalType(OpAsmParser &parser, SmallVectorImpl<Type> &types);
  LogicalResult verifyStructuredPreQuant( Operation *op, Value preQuant, Type sourceElementType, Type destinationElementType, std::optional<AccStoreQuantPreMode> preQuantMode);
  LogicalResult verifyStructuredPreRelu(Operation *op, Value preRelu, Value clipValue, std::optional<ReluPreMode> preReluMode);
  LogicalResult verifyAccStoreUnitFlagNz2dn( Operation *op, std::optional<AccStoreUnitFlagCtrl> unitFlag, Value loop0SrcStride);
  LogicalResult verifyStructuredAccStoreMode( Operation *op, Value split, Value loop0SrcStride, Value loop3Count, Type destinationElementType, std::optional<AccStoreUnitFlagCtrl> unitFlag, std::optional<AccStoreMode> mode);
  void printStructuredAccStoreMode(OpAsmPrinter &printer, AccStoreMode mode, Value split, Value loop0SrcStride);
  void printStructuredAccStoreSatMode(OpAsmPrinter &printer, AccStoreSatMode satMode);
  ParseResult parseStructuredAccStoreOptionalType( OpAsmParser &parser, bool hasOperand, SmallVectorImpl<Type> &types);

} // namespace mlir::pto::structuredacc_detail

#endif // PTO_IR_VPTO_STRUCTUREDACC_INTERNAL_H

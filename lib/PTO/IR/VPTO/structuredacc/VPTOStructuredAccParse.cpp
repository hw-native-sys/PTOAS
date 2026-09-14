// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOStructuredAccParse.cpp - structured acc-store asm parse entry points ===//
//===----------------------------------------------------------------------===//

#include "VPTOStructuredAccInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::structuredacc_detail;

ParseResult parseStructuredAccStoreClauses(
    OpAsmParser &parser, StructuredAccStoreAsmState &state) {
  int lastClause = -1;
  bool seenClause = false;
  bool hasMore = true;
  while (hasMore) {
    if (seenClause) {
      if (failed(parser.parseOptionalComma())) {
        hasMore = false;
        continue;
      }
    }
    StringRef keyword;
    OptionalParseResult optParseResult = parser.parseOptionalKeyword(&keyword);
    if (!optParseResult.has_value() || failed(*optParseResult)) {
      if (!seenClause) {
        hasMore = false;
        continue;
      }
      return parser.emitError(parser.getCurrentLocation(), "expected valid keyword");
    }
    seenClause = true;

    StructuredAccStoreClauseKind kind;
    if (!classifyStructuredAccStoreClause(keyword, kind)) {
      return parser.emitError(parser.getCurrentLocation(), "unknown mte_l0c clause");
    }
    if (static_cast<int>(kind) < lastClause) {
      return parser.emitError(parser.getCurrentLocation(),
                              "mte_l0c clauses must follow canonical order");
    }
    lastClause = static_cast<int>(kind);

    if (kind == StructuredAccStoreClauseKind::Sat) {
      if (failed(parseStructuredAccStoreSatClause(parser, state, keyword))) {
        return failure();
      }
      continue;
    }
    if (failed(parseStructuredAccStoreClauseBody(parser, state, kind, keyword))) {
      return failure();
    }
  }
  return success();
}

ParseResult parseStructuredAccStoreTailTypes(
    OpAsmParser &parser, StructuredAccStoreAsmState &state) {
  if (failed(parseStructuredAccStoreOptionalType(
          parser, !state.preQuantOperands.empty(), state.preQuantTypes)) ||
      failed(parseStructuredAccStoreOptionalType(
          parser, !state.preReluOperands.empty(), state.preReluTypes)) ||
      failed(parseStructuredAccStoreOptionalType(
          parser, !state.clipValueOperands.empty(), state.clipValueTypes)) ||
      failed(parseStructuredAccStoreOptionalType(
          parser, !state.splitOperands.empty(), state.splitTypes)) ||
      failed(parseStructuredAccStoreOptionalType(
          parser, !state.loop0SrcStrideOperands.empty(),
          state.loop0SrcStrideTypes))) {
    return failure();
  }
  if (!state.loop3CountOperands.empty() &&
      (parser.parseComma() ||
       parseStructuredOptionalType(parser, state.loop3CountTypes) ||
       parser.parseComma() ||
       parseStructuredOptionalType(parser, state.loop3SrcStrideTypes) ||
       parser.parseComma() ||
       parseStructuredOptionalType(parser, state.loop3DstStrideTypes))) {
    return failure();
  }
  return success();
}

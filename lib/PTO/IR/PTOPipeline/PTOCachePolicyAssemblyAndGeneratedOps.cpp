// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.


static ParseResult parseL1Cache(OpAsmParser &parser, L1CacheAttr &l1cache) {
  if (failed(parser.parseOptionalKeyword("l1cache"))) {
    l1cache = L1CacheAttr::get(parser.getContext(), L1Cache::Cache);
    return success();
  }

  StringRef keyword;
  if (parser.parseLParen() || parser.parseKeyword(&keyword) ||
      parser.parseRParen()) {
    return failure();
  }
  std::optional<L1Cache> parsed = symbolizeL1Cache(keyword);
  if (!parsed) {
    return parser.emitError(parser.getCurrentLocation())
           << "expected memory l1cache to be cache or uncache";
  }
  l1cache = L1CacheAttr::get(parser.getContext(), *parsed);
  return success();
}

static void printL1Cache(OpAsmPrinter &printer, Operation *op,
                         L1CacheAttr l1cache) {
  if (!l1cache) {
    return;
  }
  printer << "l1cache(" << stringifyL1Cache(l1cache.getValue()) << ")";
}

static ParseResult parseLdL2Cache(OpAsmParser &parser,
                                  LdL2CacheAttr &l2cache) {
  if (failed(parser.parseOptionalKeyword("l2cache"))) {
    l2cache = LdL2CacheAttr::get(parser.getContext(), LdL2Cache::NMFV);
    return success();
  }

  StringRef keyword;
  if (parser.parseLParen() || parser.parseKeyword(&keyword) ||
      parser.parseRParen()) {
    return failure();
  }
  std::optional<LdL2Cache> parsed = symbolizeLdL2Cache(keyword);
  if (!parsed) {
    return parser.emitError(parser.getCurrentLocation())
           << "expected load L2 cache control to be one of "
           << kLdL2CacheKeywords;
  }
  l2cache = LdL2CacheAttr::get(parser.getContext(), *parsed);
  return success();
}
static void printLdL2Cache(OpAsmPrinter &printer, Operation *op,
                           LdL2CacheAttr l2cache) {
  if (!l2cache) {
    return;
  }
  printer << "l2cache(" << stringifyLdL2Cache(l2cache.getValue()) << ")";
}

static ParseResult parseStL2Cache(OpAsmParser &parser,
                                  StL2CacheAttr &l2cache) {
  if (failed(parser.parseOptionalKeyword("l2cache"))) {
    l2cache = StL2CacheAttr::get(parser.getContext(), StL2Cache::NMFV);
    return success();
  }

  StringRef keyword;
  if (parser.parseLParen() || parser.parseKeyword(&keyword) ||
      parser.parseRParen()) {
    return failure();
  }
  std::optional<StL2Cache> parsed = symbolizeStL2Cache(keyword);
  if (!parsed) {
    return parser.emitError(parser.getCurrentLocation())
           << "expected store L2 cache control to be one of "
           << kStL2CacheKeywords;
  }
  l2cache = StL2CacheAttr::get(parser.getContext(), *parsed);
  return success();
}

static void printStL2Cache(OpAsmPrinter &printer, Operation *op,
                           StL2CacheAttr l2cache) {
  if (!l2cache) {
    return;
  }
  printer << "l2cache(" << stringifyStL2Cache(l2cache.getValue()) << ")";
}

// custom<StructPath>($path): a bracketed list of constant field indices, e.g.
// `[0, 2]`. Backs pto.struct_get / pto.struct_set so they read as `%s[0, 2]`.
static ParseResult parseStructPath(OpAsmParser &parser,
                                   DenseI64ArrayAttr &path) {
  SmallVector<int64_t> indices;
  if (parser.parseLSquare()) {
    return failure();
  }
  if (failed(parser.parseOptionalRSquare())) {
    do {
      int64_t idx = 0;
      if (parser.parseInteger(idx)) {
        return failure();
      }
      indices.push_back(idx);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRSquare()) {
      return failure();
    }
  }
  path = DenseI64ArrayAttr::get(parser.getContext(), indices);
  return success();
}

static void printStructPath(OpAsmPrinter &printer, Operation *op,
                            DenseI64ArrayAttr path) {
  printer << "[";
  llvm::ArrayRef<int64_t> indices = path.asArrayRef();
  for (size_t i = 0; i < indices.size(); ++i) {
    if (i) {
      printer << ", ";
    }
    printer << indices[i];
  }
  printer << "]";
}

// [Include 必须放在最后]
#include "PTO/IR/PTOInterfaces.cpp.inc"
#include "PTO/IR/VPTOInterfaces.cpp.inc"
#define GET_OP_CLASSES
#include "PTO/IR/PTOOps.cpp.inc"

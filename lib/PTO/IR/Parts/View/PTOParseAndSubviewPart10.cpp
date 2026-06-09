// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Split from PTOParseAndSubview.cpp; kept as a fragment included by PTOParseAndSubview.cpp.
// Intentionally has no local includes.

using namespace mlir;
using namespace mlir::pto;

ParseResult mlir::pto::SubViewOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  OpAsmParser::UnresolvedOperand source;
  SmallVec4<OpAsmParser::UnresolvedOperand> offsets;
  SmallVec2<OpAsmParser::UnresolvedOperand> valids;
  Type sourceTy;
  Type resultTy;
  bool hasExplicitResultTy = false;
  if (failed(parseSubViewSourceOffsetsAndSizes(parser, result, source, offsets)))
    return failure();
  if (failed(parseSubViewValids(parser, valids)))
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(sourceTy))
    return failure();
  if (failed(resolveSubViewSourceAndIndices(parser, result, source, sourceTy,
                                            resultTy, hasExplicitResultTy,
                                            offsets, valids)))
    return failure();

  int32_t hasValid = valids.empty() ? 0 : 1;
  addOperandSegmentSizesAttr(parser, result,
                             {1, static_cast<int32_t>(offsets.size()), hasValid,
                              hasValid});
  return finalizeSubViewResultTypes(parser, result, resultTy,
                                    hasExplicitResultTy);
}

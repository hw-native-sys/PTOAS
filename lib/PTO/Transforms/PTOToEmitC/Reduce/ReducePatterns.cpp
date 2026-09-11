// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- ReducePatterns.cpp - Reduce pattern registration --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "ReduceInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

// Aggregates the per-op-family pattern registrations for this domain.
static void populateReduceFamilyPatterns(RewritePatternSet &patterns,
                                      TypeConverter &typeConverter, MLIRContext *ctx) {
  populateReduceArithCmpIPatterns(patterns, typeConverter, ctx);
  populateReduceReduceMiscPatterns(patterns, typeConverter, ctx);
  populateReduceTInterleavePatterns(patterns, typeConverter, ctx);
  populateReduceTMathPatterns(patterns, typeConverter, ctx);
  populateReduceTPartPatterns(patterns, typeConverter, ctx);
  populateReduceTPrintTrapPatterns(patterns, typeConverter, ctx);
  populateReduceTRowExpandPatterns(patterns, typeConverter, ctx);
  populateReduceTRowReducePatterns(patterns, typeConverter, ctx);
  populateReduceTScatterPatterns(patterns, typeConverter, ctx);
  populateReduceTSelPatterns(patterns, typeConverter, ctx);
  populateReduceTShiftScalarPatterns(patterns, typeConverter, ctx);
}

void populateTensorReducePatterns(RewritePatternSet &patterns,
                              TypeConverter &typeConverter,
                              MLIRContext *ctx, PTOArch targetArch) {
  (void)targetArch;
  populateReduceFamilyPatterns(patterns, typeConverter, ctx);
}

} // namespace pto
} // namespace mlir

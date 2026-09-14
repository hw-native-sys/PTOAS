// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- TensorPatterns.cpp - Tensor pattern registration --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "TensorInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

// Aggregates the per-op-family pattern registrations for this domain.
static void populateTensorFamilyPatterns(RewritePatternSet &patterns,
                                      TypeConverter &typeConverter, MLIRContext *ctx,
                                      PTOArch targetArch) {
  populateTensorTActivationPatterns(patterns, typeConverter, ctx);
  populateTensorTAddPatterns(patterns, typeConverter, ctx);
  populateTensorTAndPatterns(patterns, typeConverter, ctx);
  populateTensorTCIPatterns(patterns, typeConverter, ctx);
  populateTensorTCmpPatterns(patterns, typeConverter, ctx);
  populateTensorTColExpandPatterns(patterns, typeConverter, ctx);
  populateTensorTColReducePatterns(patterns, typeConverter, ctx);
  populateTensorTConcatPatterns(patterns, typeConverter, ctx);
  populateTensorTCvtPatterns(patterns, typeConverter, ctx);
  populateTensorTDivPatterns(patterns, typeConverter, ctx);
  populateTensorTElemwisePatterns(patterns, typeConverter, ctx);
  populateTensorTExpPatterns(patterns, typeConverter, ctx);
  populateTensorTExtractInsertPatterns(patterns, typeConverter, ctx);
  populateTensorTFillPadPatterns(patterns, typeConverter, ctx);
  populateTensorTGatherPatterns(patterns, typeConverter, ctx);
  populateTensorTMinMaxPatterns(patterns, typeConverter, ctx);
  populateTensorTMovPatterns(patterns, typeConverter, ctx);
  populateTensorTMulPatterns(patterns, typeConverter, ctx);
  populateTensorTQuantPatterns(patterns, typeConverter, ctx, targetArch);
  populateTensorTRandomPatterns(patterns, typeConverter, ctx);
  populateTensorTSortPatterns(patterns, typeConverter, ctx);
  populateTensorTSubPatterns(patterns, typeConverter, ctx);
  populateTensorTTransPatterns(patterns, typeConverter, ctx);
  populateTensorTTriPatterns(patterns, typeConverter, ctx);
}

void populateTensorPatterns(RewritePatternSet &patterns,
                              TypeConverter &typeConverter,
                              MLIRContext *ctx, PTOArch targetArch) {
  (void)targetArch;
  populateTensorFamilyPatterns(patterns, typeConverter, ctx, targetArch);
}

} // namespace pto
} // namespace mlir

// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- ScalarMiscPatterns.cpp - ScalarMisc pattern registration --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "ScalarMiscInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

// Aggregates the per-op-family pattern registrations for this domain.
static void populateScalarMiscFamilyPatterns(RewritePatternSet &patterns,
                                      TypeConverter &typeConverter, MLIRContext *ctx,
                                      PTOArch targetArch) {
  populateScalarMiscAsyncSessionPatterns(patterns, typeConverter, ctx, targetArch);
  populateScalarMiscCommPatterns(patterns, typeConverter, ctx);
  populateScalarMiscGlobalEventArrayPatterns(patterns, typeConverter, ctx);
  populateScalarMiscLocalArrayPatterns(patterns, typeConverter, ctx);
  populateScalarMiscScalarMiscMiscPatterns(patterns, typeConverter, ctx);
  populateScalarMiscScalarMiscOpsPatterns(patterns, typeConverter, ctx);
  populateScalarMiscScalarPtrPatterns(patterns, typeConverter, ctx);
  populateScalarMiscStructPatterns(patterns, typeConverter, ctx);
}

void populateScalarMiscPatterns(RewritePatternSet &patterns,
                              TypeConverter &typeConverter,
                              MLIRContext *ctx, PTOArch targetArch) {
  (void)targetArch;
  populateScalarMiscFamilyPatterns(patterns, typeConverter, ctx, targetArch);
}

} // namespace pto
} // namespace mlir

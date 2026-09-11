// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- SyncCommPatterns.cpp - SyncComm pattern registration --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "SyncCommInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

// Aggregates the per-op-family pattern registrations for this domain.
static void populateSyncCommFamilyPatterns(RewritePatternSet &patterns,
                                      TypeConverter &typeConverter, MLIRContext *ctx,
                                      PTOArch targetArch) {
  populateSyncCommBlockIdxPatterns(patterns, typeConverter, ctx);
  populateSyncCommBufPoolPatterns(patterns, typeConverter, ctx);
  populateSyncCommFuncCallPatterns(patterns, typeConverter, ctx);
  populateSyncCommSyncPatterns(patterns, typeConverter, ctx, targetArch);
}

void populateSyncCommPatterns(RewritePatternSet &patterns,
                              TypeConverter &typeConverter,
                              MLIRContext *ctx, PTOArch targetArch) {
  (void)targetArch;
  populateSyncCommFamilyPatterns(patterns, typeConverter, ctx, targetArch);
}

} // namespace pto
} // namespace mlir

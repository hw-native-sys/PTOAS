// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Internal helper declarations shared within the SyncComm lowering domain.
#pragma once

#include "../PTOToEmitCEmitters.h"

namespace mlir {
namespace pto {

using namespace mlir;

bool tryConvertPipeAttrToToken(Attribute attr, std::string &token);
bool tryConvertEventAttrToToken(Attribute attr, std::string &token);
bool tryAssignSyncTokens(Attribute srcAttr, Attribute dstAttr, Attribute evtAttr, std::string &srcTok, std::string &dstTok, std::string &evtTok);
bool tryExtractSyncTokensFromNamedAttrs(Operation *op, StringRef srcName, StringRef dstName, StringRef evtName, std::string &srcTok, std::string &dstTok, std::string &evtTok);
bool tryExtractSyncTokensFromArrayAttr(Operation *op, StringRef attrName, std::string &srcTok, std::string &dstTok, std::string &evtTok);
bool tryExtractFallbackSyncTokens(Operation *op, std::string &srcTok, std::string &dstTok, std::string &evtTok);

void populateSyncCommBlockIdxPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateSyncCommBufPoolPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateSyncCommFuncCallPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateSyncCommSyncPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx,
                        PTOArch targetArch);

} // namespace pto
} // namespace mlir

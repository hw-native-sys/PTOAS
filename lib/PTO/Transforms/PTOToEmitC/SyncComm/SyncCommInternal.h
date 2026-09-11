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

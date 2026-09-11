// Internal helper declarations shared within the LoadStore lowering domain.
#pragma once

#include "../PTOToEmitCEmitters.h"

namespace mlir {
namespace pto {

using namespace mlir;

void emitTileCallAndReplace(Operation *op, ConversionPatternRewriter &rewriter, StringRef callee, ArrayAttr templateArgs, ValueRange operands, Value dst);

void populateLoadStoreTLoadPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateLoadStoreTMatmulPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateLoadStoreTStorePatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);

} // namespace pto
} // namespace mlir

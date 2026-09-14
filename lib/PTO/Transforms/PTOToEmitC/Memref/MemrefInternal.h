// Internal helper declarations shared within the Memref lowering domain.
#pragma once

#include "../PTOToEmitCEmitters.h"

namespace mlir {
namespace pto {

using namespace mlir;

Value ofrToEmitCIndexValue(ConversionPatternRewriter &rewriter, Location loc, Type indexTy, OpFoldResult ofr);
std::string memrefElemTypeToString(Type elemTy);

void populateMemrefReinterpretCastPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateMemrefSubviewPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);

} // namespace pto
} // namespace mlir

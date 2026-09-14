// Internal helper declarations shared within the Tensor lowering domain.
#pragma once

#include "../PTOToEmitCEmitters.h"

namespace mlir {
namespace pto {

using namespace mlir;

std::string cmpModeTok(pto::CmpModeAttr a);
std::string roundModeTok(mlir::pto::RoundModeAttr attr);
std::string saturationModeTok(mlir::pto::SaturationModeAttr attr);
StringRef getReluPreModeToken(pto::ReluPreMode mode);
StringRef getAccToVecModeToken(pto::AccToVecMode mode);
StringRef getTInsertModeToken(pto::TInsertMode mode);
void pushModeAndReluTemplateArgs(SmallVectorImpl<Attribute> &args, MLIRContext *ctx, pto::AccToVecModeAttr modeAttr, bool reluNonDefault, pto::ReluPreMode reluPreMode);
StringRef getTFillPadModeToken(pto::TFillPadLoweringKind loweringKind);
Value materializeOffsetAddress(ConversionPatternRewriter &rewriter, Location loc, MLIRContext *ctx, Value offset);
FailureOr<ArrayAttr> buildTQuantTemplateArgs(pto::TQuantOp op, ConversionPatternRewriter &rewriter, MLIRContext *ctx, Value dst, Value src, Value fp, Value tmp);
Value addressOfEmitCValue(ConversionPatternRewriter &rewriter, Location loc, MLIRContext *ctx, Value v, emitc::OpaqueType ot);
void appendModernMxTemplateArgs(pto::TQuantMxOp op, MLIRContext *ctx, SmallVectorImpl<Attribute> &out);
StringRef quantScaleAlgTok(pto::QuantScaleAlg alg);
StringRef vecStoreModeTok(pto::VecStoreMode mode);
void appendLegacyExpZzMxTemplateArgs( pto::TQuantMxOp op, MLIRContext *ctx, StringRef quantTypeStr, emitc::OpaqueType dstOT, emitc::OpaqueType srcOT, emitc::OpaqueType expOT, emitc::OpaqueType maxOT, emitc::OpaqueType scalingOT, SmallVectorImpl<Attribute> &out);

void populateTensorTActivationPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateTensorTAddPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateTensorTAndPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateTensorTCIPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateTensorTCmpPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateTensorTColExpandPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateTensorTColReducePatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateTensorTConcatPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateTensorTCvtPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateTensorTDivPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateTensorTElemwisePatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateTensorTExpPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateTensorTExtractInsertPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateTensorTFillPadPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateTensorTGatherPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateTensorTMinMaxPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateTensorTMovPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateTensorTMulPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateTensorTQuantPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx,
                        PTOArch targetArch);
void populateTensorTRandomPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateTensorTSortPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateTensorTSubPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateTensorTTransPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateTensorTTriPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);

} // namespace pto
} // namespace mlir

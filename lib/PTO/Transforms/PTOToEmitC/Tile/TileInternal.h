// Internal helper declarations shared within the Tile lowering domain.
#pragma once

#include "../PTOToEmitCEmitters.h"

namespace mlir {
namespace pto {

using namespace mlir;

const char *reinterpretCastTileRole(pto::AddressSpace as, Value source);
std::string buildReinterpretCastTileTypeString(MemRefType resMrTy, Type elemTy, const char *roleTok);
Value reinterpretCastBaseAddress(ConversionPatternRewriter &rewriter, Location loc, Value source, pto::AddressSpace as, StringRef elemTok, Type u64Ty);
void assignTileAddress(ConversionPatternRewriter &rewriter, Location loc, MLIRContext *ctx, Value tile, Value addr);
FailureOr<Value> createEmitCTileVariable(ConversionPatternRewriter &rewriter, Location loc, const TypeConverter *typeConverter, pto::TileBufType tileTy, bool initializeDynamicValidToShape = false);
std::pair<SmallVector<Value, 5>, SmallVector<Value, 5>> buildRuntime5DValues(ConversionPatternRewriter &rewriter, Location loc, ValueRange runtimeShape, ValueRange runtimeStrides, int64_t shift);


template <typename OpTy>
FailureOr<std::string>
resolvePipeTileConfigToken(OpTy op, PTOArch targetArch) {
  (void)targetArch;
  if constexpr (std::is_same_v<OpTy, mlir::pto::TPushOp>) {
    if (auto accPushEpilogue =
            getPipeInitAccPushEpilogue(getPipeInitDef(op.getPipeHandle()))) {
      auto pipeId = getFrontendPipeIdFromHandle(op.getPipeHandle());
      if (pipeId)
        return buildFixpipeConfigAliasName(*pipeId);
      auto configTokOr = buildFixpipeConfigTypeToken(accPushEpilogue);
      if (failed(configTokOr))
        return failure();
      return *configTokOr;
    }
  }
  return getTileSplitToken(op.getSplit());
}
void populateTilePartitionViewPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateTileSectionPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateTileTensorViewPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateTileTileAllocPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx,
                        PTOArch targetArch);
void populateTileTileMiscPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx,
                        PTOArch targetArch);

} // namespace pto
} // namespace mlir

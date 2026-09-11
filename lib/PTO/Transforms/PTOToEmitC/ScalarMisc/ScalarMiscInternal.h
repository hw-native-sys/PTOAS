// Internal helper declarations shared within the ScalarMisc lowering domain.
#pragma once

#include "../PTOToEmitCEmitters.h"

namespace mlir {
namespace pto {

using namespace mlir;

StringRef scatterAtomicTok(pto::ScatterAtomicOp atomic);
StringRef scatterOobTok(pto::ScatterOOB mode);
StringRef scatterConflictTok(pto::ScatterConflict mode);
StringRef coalesceTok(pto::Coalesce mode);
Value peelAllConversionCasts(Value v);
bool isTileLikeValue(Value v);
Type getStructFieldValueType(const TypeConverter *tc, Type fieldPtoTy);
FailureOr<Type> getStructMemberFieldType(mlir::pto::StructType structTy, int64_t index, const TypeConverter *tc);
FailureOr<Value> getStructAdaptorValue(ValueRange operands);
FailureOr<Value> buildStructMemberChain( ConversionPatternRewriter &rewriter, Location loc, const TypeConverter *tc, Value root, mlir::pto::StructType rootPtoTy, llvm::ArrayRef<int64_t> path);
FailureOr<Value> resolveStructMember(Operation *op, ValueRange adaptorOperands, Type structPtoTy, ArrayRef<int64_t> path, ConversionPatternRewriter &rewriter, const TypeConverter *typeConverter);


template <typename OpTy>
FailureOr<SmallVector<Value>> buildCommGroupGlobalTensors(
    ConversionPatternRewriter &rewriter, Location loc, OpTy op,
    ValueRange originalGroup, ValueRange emittedGroup) {
  SmallVector<Value> groupGTs;
  groupGTs.reserve(originalGroup.size());
  for (auto [orig, emitted] : llvm::zip(originalGroup, emittedGroup)) {
    FailureOr<Value> gt =
        buildCommGlobalTensorValue(rewriter, loc, orig, emitted, op.getOperation());
    if (failed(gt))
      return failure();
    groupGTs.push_back(*gt);
  }
  return groupGTs;
}
void populateScalarMiscAsyncSessionPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx,
                        PTOArch targetArch);
void populateScalarMiscCommPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateScalarMiscGlobalEventArrayPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateScalarMiscLocalArrayPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateScalarMiscScalarMiscMiscPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateScalarMiscScalarMiscOpsPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateScalarMiscScalarPtrPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateScalarMiscStructPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);

} // namespace pto
} // namespace mlir

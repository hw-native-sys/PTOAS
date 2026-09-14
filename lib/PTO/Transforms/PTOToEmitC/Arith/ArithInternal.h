// Internal helper declarations shared within the Arith lowering domain.
#pragma once

#include "../PTOToEmitCEmitters.h"

namespace mlir {
namespace pto {

using namespace mlir;

struct ScalarIntOpPrologue {
  Location loc;
  Type dstTy;
};

struct UnsignedBinaryOperands {
  emitc::OpaqueType uTy;
  Value lhs;
  Value rhs;
};

struct SignedDivParts {
  Value quotient;
  Value remainder;
  Value remainderNonZero;
  Value signsDiffer;
  Value signsSame;
};

struct ArithCmpFConfig {
  bool unordered = false;
  emitc::CmpPredicate predicate = emitc::CmpPredicate::eq;
};

unsigned getScalarIntOrIndexBitWidth(Type opTy);
bool isScalarIntOrIndex(Type opTy);
SignedDivParts buildSignedDivParts(ConversionPatternRewriter &rewriter, Location loc, Type dstTy, Value lhs, Value rhs);
LogicalResult emitWidenedInt(Operation *op, Value in, IntegerType srcIntTy, IntegerType dstIntTy, Type dstTy, bool isSigned, ConversionPatternRewriter &rewriter);
Value cmpFIsNaN(ConversionPatternRewriter &rewriter, Location loc, Value v);
Value cmpFIsNotNaN(ConversionPatternRewriter &rewriter, Location loc, Value v);
Value buildCmpFResult(const ArithCmpFConfig &config, ConversionPatternRewriter &rewriter, Location loc, Type i1Ty, Value lhs, Value rhs);
void applyFuncSpecifiers(func::FuncOp op, ConversionPatternRewriter &rewriter, emitc::FuncOp &emitcFunc);
InterCoreSyncCallDesc buildInterCoreSyncSetCallImpl(ConversionPatternRewriter &rewriter, Value msgVal, PTOArch targetArch, pto::PipeAttr pipeAttr);
FailureOr<UnsignedBinaryOperands> getUnsignedBinaryOperands(Operation *op, Value lhs, Value rhs, ConversionPatternRewriter &rewriter);
std::optional<Value> buildSpecialCmpFResult(arith::CmpFPredicate predicate, ConversionPatternRewriter &rewriter, Location loc, Type i1Ty, Value lhs, Value rhs);
std::optional<ArithCmpFConfig> getCmpFConfig(arith::CmpFPredicate predicate);

template <typename PatternTy, typename ArithOp>
static FailureOr<ScalarIntOpPrologue> resolveScalarIntPrologue(
    const PatternTy *pattern, ArithOp op, typename ArithOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) {
  (void)adaptor;
  if (!isScalarIntOrIndex(op.getType()))
    return rewriter.notifyMatchFailure(
        op, "expected scalar integer or index type");
  Type dstTy = pattern->getTypeConverter()->convertType(op.getType());
  if (!dstTy)
    return failure();
  return ScalarIntOpPrologue{op.getLoc(), dstTy};
}

template <typename EmitCOp, typename ArithOp>
static LogicalResult emitUnsignedBinaryAndReplace(
    ArithOp op, typename ArithOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter, Location loc, Type dstTy) {
  auto operandsOr = getUnsignedBinaryOperands(op.getOperation(),
                                              adaptor.getLhs(),
                                              adaptor.getRhs(), rewriter);
  if (failed(operandsOr))
    return failure();
  Value resU = rewriter.create<EmitCOp>(loc, operandsOr->uTy,
                                        operandsOr->lhs, operandsOr->rhs);
  rewriter.replaceOp(op, emitCCast(rewriter, loc, dstTy, resU));
  return success();
}

template <typename PatternTy, typename ArithOp, typename EmitCOp,
          typename EmitCI1Op>
static LogicalResult
emitScalarIntBinary(const PatternTy *pattern, ArithOp op,
                    typename ArithOp::Adaptor adaptor,
                    ConversionPatternRewriter &rewriter) {
  FailureOr<ScalarIntOpPrologue> prologue =
      resolveScalarIntPrologue(pattern, op, adaptor, rewriter);
  if (failed(prologue))
    return failure();
  auto [loc, dstTy] = *prologue;

  if (getScalarIntOrIndexBitWidth(op.getType()) == 1) {
    rewriter.replaceOpWithNewOp<EmitCI1Op>(op, op.getType(), adaptor.getLhs(),
                                           adaptor.getRhs());
    return success();
  }
  return emitUnsignedBinaryAndReplace<EmitCOp>(op, adaptor, rewriter, loc,
                                               dstTy);
}

void populateArithArithBinaryPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateArithArithCastPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateArithArithCmpPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateArithArithFloatMinMaxPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateArithArithIntBinaryPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateArithArithIntDivPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateArithArithIntMinMaxPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateArithArithMiscPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateArithArithShiftPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);

} // namespace pto
} // namespace mlir

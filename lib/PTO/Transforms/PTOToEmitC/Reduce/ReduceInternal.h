// Internal helper declarations shared within the Reduce lowering domain.
#pragma once

#include "../PTOToEmitCEmitters.h"

namespace mlir {
namespace pto {

using namespace mlir;

template <typename OpTy>
SmallVector<Value, 4>
collectRowExpandOperands(OpTy op, typename OpTy::Adaptor adaptor) {
  Value src0 = adaptor.getSrc0();
  Value src1 = adaptor.getSrc1();
  Value dst = adaptor.getDst();
  Value tmp = op.getTmp() ? adaptor.getTmp() : Value();

  SmallVector<Value, 4> operands;
  if (tmp)
    operands.assign({dst, src0, src1, tmp});
  else
    operands.assign({dst, src0, src1});
  return operands;
}

void populateReduceArithCmpIPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateReduceReduceMiscPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateReduceTInterleavePatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateReduceTMathPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateReduceTPartPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateReduceTPrintTrapPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateReduceTRowExpandPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateReduceTRowReducePatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateReduceTScatterPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateReduceTSelPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateReduceTShiftScalarPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);

} // namespace pto
} // namespace mlir

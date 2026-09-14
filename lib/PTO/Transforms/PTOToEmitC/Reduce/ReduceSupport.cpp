// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- ReduceSupport.cpp - Reduce lowering helpers --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "ReduceInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

// CANN Open Software License Agreement Version 2.0 (the "License").
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.

//===- PTOToEmitCTensorReduce.cpp - tensor elementwise op lowering ---------===//
//===----------------------------------------------------------------------===//






//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TOR DPS/memref op)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TORS DPS/memref op)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TPARTADD DPS/memref op)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TPARTMAX DPS/memref op)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TPARTMIN DPS/memref op)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TPARTMUL DPS/memref op)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TPRELU DPS/memref op)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TRECIP DPS/memref op)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TRELU DPS/memref op)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TREM DPS/memref op)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TREMS DPS/memref op)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TPOW DPS/memref op)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TPOWS DPS/memref op)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TROWEXPAND DPS/memref op)
//===----------------------------------------------------------------------===//

// Binary row-expand ops (add/sub/mul/div/max/min): emit
// TROWEXPAND<OP>(dst, src0, src1[, tmp]) with the optional tmp operand.
// LastUseAware selects the last-use-aware call so consumed tiles carry
// their pto.last_use marker into the generated C++ (mul/div keep the
// original behavior; the other flavors never emitted it).

// Gather the (dst, src0, src1[, tmp]) operand list shared by all binary
// row-expand lowerings.
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TROWMAX DPS/memref op)
//===----------------------------------------------------------------------===//

// Row-wise reduce ops (max/min/sum): emit TROW<OP>(dst, src, tmp) with a
// last-use-aware call so consumed tiles can be released eagerly.

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TROWMIN DPS/memref op)
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TROWSUM DPS/memref op)
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TRSQRT DPS/memref op)
// - no-tmp form : TRSQRT(dst, src)
// - tmp form    : TRSQRT(dst, src, tmp)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSCATTER DPS/memref op)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSEL DPS/memref op)
//===----------------------------------------------------------------------===//

// tsel/tsels lowering: TSEL(dst, mask, src0, src1[, tmp]) /
// TSELS(dst, mask, src, tmp, scalar).

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSHL DPS/memref op)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSHR DPS/memref op)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering for TSHLS/TSHRS DPS: shift by scalar)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (TSORT32 DPS/memref op: ins(src, idx[, tmp]) outs(dst))
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSQRT DPS/memref op)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSTORE_FP DPS/memref op)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSUBC DPS/memref op)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSUBS DPS/memref op)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSUBSC DPS/memref op)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TXOR DPS/memref op)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TXORS DPS/memref op)
//===----------------------------------------------------------------------===//

// pto.print "format", %scalar -> PRINTF("format", scalar)

// pto.trap -> TRAP()

// =============================================================================
// Arith CmpI -> EmitC Cmp
// =============================================================================

//===----------------------------------------------------------------------===//
// Section Op Lowering
//===----------------------------------------------------------------------===//


void replaceOrEraseWithOpaqueCallAndReturnDst(Operation *op, Value dst,
                                                     StringRef callee,
                                                     ArrayRef<Value> args,
                                                     ArrayAttr templateArgs,
                                                     ConversionPatternRewriter &rewriter) {
  createLastUseAwareOpaqueCall(rewriter, op, TypeRange{}, callee, args, ArrayAttr{}, templateArgs);
  if (op->getNumResults() == 1) {
    rewriter.replaceOp(op, dst);
  } else {
    rewriter.eraseOp(op);
  }
}



} // namespace pto
} // namespace mlir

// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- LoadStoreSupport.cpp - LoadStore lowering helpers --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "LoadStoreInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

// CANN Open Software License Agreement Version 2.0 (the "License").
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.

//===- PTOToEmitCLoadStore.cpp - tload/tstore/matmul lowering ---------===//
//===----------------------------------------------------------------------===//





//===----------------------------------------------------------------------===//
// pto.matmul_dps lowering (Simplified: No internal copy/sync)
//===----------------------------------------------------------------------===//
//
// Render `pto.tmatmul` as one of three forms depending on the optional
// `acc_phase` attribute:
//   * absent / Unspecified  -> `TMATMUL(dst, lhs, rhs)`
//   * Partial               -> `TMATMUL<pto::AccPhase::Partial>(dst, lhs, rhs)`
//   * Final                 -> `TMATMUL<pto::AccPhase::Final>(dst, lhs, rhs)`
// The Unspecified default keeps backward compatibility with all upstream IR

//===----------------------------------------------------------------------===//
//
// Render `pto.tmatmul` as one of three forms depending on the optional
// `acc_phase` attribute:
//   * absent / Unspecified  -> `TMATMUL(dst, lhs, rhs)`
//   * Partial               -> `TMATMUL<pto::AccPhase::Partial>(dst, lhs, rhs)`
//   * Final                 -> `TMATMUL<pto::AccPhase::Final>(dst, lhs, rhs)`
// The Unspecified default keeps backward compatibility with all upstream IR
// that does not yet emit an explicit phase attribute.

// Emit an opaque call for a DPS tile op and forward (or erase) the op: when
// the op has a result, it is replaced by its dst operand.
void emitTileCallAndReplace(Operation *op, ConversionPatternRewriter &rewriter,
                                   StringRef callee, ArrayAttr templateArgs,
                                   ValueRange operands, Value dst) {
  rewriter.create<emitc::CallOpaqueOp>(op->getLoc(), TypeRange{}, callee,
                                       /*args=*/ArrayAttr{},
                                       /*templateArgs=*/templateArgs, operands);
  if (op->getNumResults() == 1) {
    rewriter.replaceOp(op, dst);
  } else {
    rewriter.eraseOp(op);
  }
}

//===----------------------------------------------------------------------===//
// pto.tgemv lowering
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// pto.tgemv.acc lowering
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// pto.matmul_acc_dps lowering (Simplified: No internal copy/sync)
//===----------------------------------------------------------------------===//

ArrayAttr buildAccPhaseTemplateArgs(ConversionPatternRewriter &rewriter,
                                           pto::AccPhase phase) {
  StringRef tmpl;
  switch (phase) {
  case pto::AccPhase::Unspecified:
    return ArrayAttr{};
  case pto::AccPhase::Partial:
    tmpl = "pto::AccPhase::Partial";
    break;
  case pto::AccPhase::Final:
    tmpl = "pto::AccPhase::Final";
    break;
  }
  if (tmpl.empty())
    return ArrayAttr{};
  return rewriter.getArrayAttr(
      {emitc::OpaqueAttr::get(rewriter.getContext(), tmpl)});
}



} // namespace pto
} // namespace mlir

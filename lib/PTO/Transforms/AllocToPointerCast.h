// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- AllocToPointerCast.h --Convert memref.AllocOp to pto.pointercastOp-===//
#ifndef LLVM_PROJECT_ALLOCTOPOINTERCAST_H
#define LLVM_PROJECT_ALLOCTOPOINTERCAST_H
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/SmallSet.h"

namespace mlir {
namespace pto {
class MemrefAllocaOpToPointerCastOpPattern
    : public OpRewritePattern<memref::AllocOp> {
public:
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  /// map from buffer to its allocated addresses
  /// note: the buffer which does multibuffer n optimization will be allocated n
  /// addresses.
  DenseMap<Value, SmallVector<uint64_t>> buffer2Offsets;
  mutable uint64_t fallbackNextOffset = 0;

  explicit MemrefAllocaOpToPointerCastOpPattern(
      MLIRContext *context,
      DenseMap<Value, SmallVector<uint64_t>> buffer2Offsets)
      : OpRewritePattern<memref::AllocOp>(context),
        buffer2Offsets(std::move(buffer2Offsets)) {
    // Seed fallback offsets above any known planned offsets to reduce collisions.
    constexpr uint64_t kAlign = 4096;
    uint64_t maxOff = 0;
    for (const auto &kv : this->buffer2Offsets) {
      for (uint64_t off : kv.second)
        maxOff = std::max(maxOff, off);
    }
    fallbackNextOffset = ((maxOff + kAlign - 1) / kAlign) * kAlign;
  }
  LogicalResult matchAndRewrite(memref::AllocOp op,
                                PatternRewriter &rewriter) const final;
};

// class UpdateWorkSpaceAllocaOpOffsetPattern
//     : public OpRewritePattern<bishengir::memref_ext::AllocWorkspaceOp> {
// public:
//   using OpRewritePattern<
//       bishengir::memref_ext::AllocWorkspaceOp>::OpRewritePattern;

//   DenseMap<Value, SmallVector<uint64_t>> buffer2Offsets;

//   explicit UpdateWorkSpaceAllocaOpOffsetPattern(
//       MLIRContext *context,
//       DenseMap<Value, SmallVector<uint64_t>> buffer2Offsets)
//       : OpRewritePattern<bishengir::memref_ext::AllocWorkspaceOp>(context),
//         buffer2Offsets(buffer2Offsets) {}
//   LogicalResult matchAndRewrite(bishengir::memref_ext::AllocWorkspaceOp op,
//                                 PatternRewriter &rewriter) const final;
// };
} // namespace pto
} // namespace mlir

#endif // LLVM_PROJECT_ALLOCTOPOINTERCAST_H

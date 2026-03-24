//===- AllocToPointerCast.h --Convert pto.alloc_tile to pto.pointer_cast ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
#ifndef LLVM_PROJECT_ALLOCTOPOINTERCAST_H
#define LLVM_PROJECT_ALLOCTOPOINTERCAST_H
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "llvm/ADT/SmallSet.h"

namespace mlir {
namespace pto {
class AllocTileOpToPointerCastOpPattern
    : public OpRewritePattern<pto::AllocTileOp> {
public:
  using OpRewritePattern<pto::AllocTileOp>::OpRewritePattern;

  /// map from buffer to its allocated addresses
  /// note: the buffer which does multibuffer n optimization will be allocated n
  /// addresses.
  DenseMap<Value, SmallVector<uint64_t>> buffer2Offsets;
  mutable uint64_t fallbackNextOffset = 0;

  explicit AllocTileOpToPointerCastOpPattern(
      MLIRContext *context,
      DenseMap<Value, SmallVector<uint64_t>> buffer2Offsets)
      : OpRewritePattern<pto::AllocTileOp>(context),
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
  LogicalResult matchAndRewrite(pto::AllocTileOp op,
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

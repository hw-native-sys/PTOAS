//===- AllocToPointerCast.cpp - convert alloc_tile to pto.pointer_cast. -------//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AllocToPointerCast.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_ALLOCTOPOINTERCAST
#include "PTO/Transforms/Passes.h.inc"

} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {} // namespace

LogicalResult
AllocTileOpToPointerCastOpPattern::matchAndRewrite(pto::AllocTileOp op,
                                                   PatternRewriter &rewriter) const {
  // Manual-address alloc_tile is already fully bound and must not be remapped.
  if (op.getAddr())
    return failure();

  auto tileType = dyn_cast<pto::TileBufType>(op.getResult().getType());
  if (!tileType)
    return failure();

  // Keep config from the tile descriptor so lowering can generate the exact
  // Tile<...> type token (layout/fractal/pad) without memref-side recovery.
  TileBufConfigAttr configAttr = tileType.getConfigAttr();

  constexpr uint64_t kAlign = 4096;
  auto iter = buffer2Offsets.find(op.getResult());

  // If MemPlan didn't assign an address, synthesize a unique, aligned offset so
  // downstream PointerCast lowering won't crash on empty addrs.
  SmallVector<uint64_t> offsets;
  if (iter != buffer2Offsets.end())
    offsets = iter->second;

  if (offsets.empty()) {
    // Estimate tile size in bytes using the static tile descriptor.
    uint64_t bytes = kAlign;
    uint64_t elemBytes = 0;
    Type elemTy = tileType.getElementType();
    if (elemTy.isF16() || elemTy.isBF16())
      elemBytes = 2;
    else if (elemTy.isF32())
      elemBytes = 4;
    else if (auto it = dyn_cast<IntegerType>(elemTy))
      elemBytes = it.getWidth() / 8;

    if (elemBytes != 0) {
      uint64_t numel = 1;
      bool allStatic = true;
      for (int64_t d : tileType.getShape()) {
        if (d == ShapedType::kDynamic) {
          allStatic = false;
          break;
        }
        numel *= static_cast<uint64_t>(d);
      }
      if (allStatic && numel != 0)
        bytes = numel * elemBytes;
    }

    uint64_t stride = ((bytes + kAlign - 1) / kAlign) * kAlign;
    uint64_t off = fallbackNextOffset;
    fallbackNextOffset += std::max<uint64_t>(stride, kAlign);
    offsets.push_back(off);
  }

  SmallVector<Value> addrs;
  addrs.reserve(offsets.size());
  for (uint64_t offset : offsets) {
    auto constantIntOffsetOp =
        rewriter.create<arith::ConstantIntOp>(op->getLoc(), offset, 64);
    addrs.push_back(constantIntOffsetOp);
  }

  // Preserve valid-shape contract:
  // - dynamic valid dims: forward alloc_tile operands
  // - static valid dims: materialize constants from TileBufType
  // This keeps semantics identical to alloc_tile across PlanMemory rewrite.
  Value vRow, vCol;
  vRow = op.getValidRow();
  vCol = op.getValidCol();
  auto validShape = tileType.getValidShape();
  if (validShape.size() >= 2) {
    auto indexType = rewriter.getIndexType();
    Location loc = op.getLoc();
    if (!vRow && validShape[0] >= 0) {
      vRow = rewriter.create<arith::ConstantOp>(
          loc, indexType, rewriter.getIndexAttr(validShape[0]));
    }
    if (!vCol && validShape[1] >= 0) {
      vCol = rewriter.create<arith::ConstantOp>(
          loc, indexType, rewriter.getIndexAttr(validShape[1]));
    }
  }

  // Build tile-native pointer_cast with assigned physical address.
  auto ptoPointerCastOp = rewriter.create<pto::PointerCastOp>(
      op.getLoc(), tileType,
      ValueRange(addrs),      // addrs
      vRow ? vRow : Value(),  // valid_row
      vCol ? vCol : Value(),  // valid_col
      configAttr              // config from tile descriptor
  );

  rewriter.replaceOp(op, ptoPointerCastOp->getResults());
  return success();
}

// LogicalResult UpdateWorkSpaceAllocaOpOffsetPattern::matchAndRewrite(
//     bishengir::memref_ext::AllocWorkspaceOp op,
//     PatternRewriter &rewriter) const {
//   if (!op.getOffset().empty()) {
//     return failure();
//   }
//   auto iter = buffer2Offsets.find(op.getResult());
//   assert(iter != buffer2Offsets.end() && "address should be found");

//   SmallVector<Value> argOffset;
//   for (auto &offset : iter->second) {
//     Value newOffset =
//         rewriter.create<arith::ConstantIndexOp>(op->getLoc(), offset)
//             .getResult();
//     argOffset.push_back(newOffset);
//   }
//   auto allocWorkspaceOp =
//       rewriter.create<bishengir::memref_ext::AllocWorkspaceOp>(
//           op.getLoc(), op->getResultTypes(), op.getWorkspaceArg(),
//           op.getDynamicSize(), argOffset);
//   rewriter.replaceOp(op, allocWorkspaceOp->getResults());
//   return success();
// }

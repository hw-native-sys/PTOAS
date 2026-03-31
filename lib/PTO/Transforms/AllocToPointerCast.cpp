// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- AllocToPointerCast.cpp - convert memref.AllocOp to pto.pointercastOp.//
//===----------------------------------------------------------------------===//

#include "AllocToPointerCast.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_ALLOCTOPOINTERCAST
#include "PTO/Transforms/Passes.h.inc"

} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {} // namespace

namespace {
static TileBufConfigAttr inferBindTileConfig(memref::AllocOp op) {
  TileBufConfigAttr configAttr;
  for (Operation *user : op.getResult().getUsers()) {
    auto bind = dyn_cast<pto::BindTileOp>(user);
    if (!bind || bind.getSource() != op.getResult())
      continue;
    if (!configAttr) {
      configAttr = bind.getConfigAttr();
      continue;
    }
    if (configAttr != bind.getConfigAttr()) {
      op.emitWarning("alloc has multiple bind_tile users with different configs; "
                     "using the first one");
      break;
    }
  }
  return configAttr;
}

static SmallVector<uint64_t> getAllocatedOffsets(memref::AllocOp op,
                                                 BaseMemRefType memRefType,
                                                 const DenseMap<Value, SmallVector<uint64_t>> &buffer2Offsets,
                                                 uint64_t &fallbackNextOffset) {
  constexpr uint64_t kAlign = 4096;
  auto iter = buffer2Offsets.find(op.getResult());
  SmallVector<uint64_t> offsets;
  if (iter != buffer2Offsets.end())
    offsets = iter->second;

  if (offsets.empty()) {
    // Estimate buffer size (best-effort). Most PTO tile buffers are 32x32 and
    // naturally align to 4096 bytes.
    uint64_t bytes = kAlign;
    if (auto memrefTy = dyn_cast<MemRefType>(memRefType)) {
      uint64_t elemBytes = 0;
      Type elemTy = memrefTy.getElementType();
      if (elemTy.isF16())
        elemBytes = 2;
      else if (elemTy.isF32())
        elemBytes = 4;
      else if (auto it = dyn_cast<IntegerType>(elemTy))
        elemBytes = it.getWidth() / 8;

      if (elemBytes != 0) {
        uint64_t numel = 1;
        bool allStatic = true;
        for (int64_t d : memrefTy.getShape()) {
          if (d == ShapedType::kDynamic) {
            allStatic = false;
            break;
          }
          numel *= static_cast<uint64_t>(d);
        }
        if (allStatic && numel != 0)
          bytes = numel * elemBytes;
      }
    }
    uint64_t stride = ((bytes + kAlign - 1) / kAlign) * kAlign;
    uint64_t off = fallbackNextOffset;
    fallbackNextOffset += std::max<uint64_t>(stride, kAlign);
    offsets.push_back(off);
  }
  return offsets;
}

static std::pair<Value, Value> getDynamicValidShapeValues(memref::AllocOp op) {
  Value vRow;
  Value vCol;
  auto dynSizes = op.getDynamicSizes();
  if (dynSizes.size() >= 2) {
    vRow = dynSizes[0];
    vCol = dynSizes[1];
  } else if (dynSizes.size() == 1) {
    vCol = dynSizes[0];
  }
  return {vRow, vCol};
}
} // namespace

LogicalResult MemrefAllocaOpToPointerCastOpPattern::matchAndRewrite(
    memref::AllocOp op, PatternRewriter &rewriter) const {
  const auto &currentMemRefType = cast<BaseMemRefType>(op.getType());
  TileBufConfigAttr configAttr = inferBindTileConfig(op);
  SmallVector<uint64_t> offsets = getAllocatedOffsets(
      op, currentMemRefType, buffer2Offsets, fallbackNextOffset);
  SmallVector<Value> addrs;
  addrs.reserve(offsets.size());
  for (uint64_t offset : offsets) {
    auto constantIntOffsetOp =
        rewriter.create<arith::ConstantIntOp>(op->getLoc(), offset, 64);
    addrs.push_back(constantIntOffsetOp);
  }

  auto [vRow, vCol] = getDynamicValidShapeValues(op);
  auto ptoPointerCastOp = rewriter.create<pto::PointerCastOp>(
      op.getLoc(), currentMemRefType, ValueRange(addrs), vRow ? vRow : Value(),
      vCol ? vCol : Value(), configAttr);

  rewriter.replaceOp(op, ptoPointerCastOp->getResults());
  return success();
}

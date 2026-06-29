// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOInsertVMemBar.cpp - V-pipeline mem_bar insertion ----------------===//
//
// After tile ops are lowered to the structured `pto.tsub` / `pto.tcolexpandmul`
// form (and before `pto-expand-tile-op` expands them to `func.call`), each
// V-pipeline tile op carries:
//   - `OpPipeInterface`  → getPipe() == PIPE_V
//   - `DestinationStyleOpInterface` → precise read (dpsInput) / write (dpsInit)
//     operand sets
//   - tile operands that trace back to `pto.alloc_tile addr = <const>` with a
//     constant UB address.
//
// This pass analyzes RAW / WAW / WAR memory dependencies between V-pipeline
// tile ops whose `alloc_tile` address ranges overlap, and inserts `pto.mem_bar`
// barriers before the dependent op so that the later inlined `vsts` / `vlds`
// cannot reorder across the dependency.
//
// Scope:
//   - Only V-pipeline tile ops (PIPE_V). Cube / cross-pipe sync stays with
//     `pto-insert-sync` and friends.
//   - Only intra-block, intra-iteration dependencies (same enclosing block,
//     not across `scf.for` iterations).
//   - If a `pto.mem_bar` already sits between two dependent ops, no new
//     barrier is inserted.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include <optional>
#include <tuple>

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::pto;

namespace mlir {
namespace pto {
  #define GEN_PASS_DEF_PTOINSERTVMEMBAR
  #include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

namespace {

/// A half-open UB address range [base, base + size) for one tile operand.
struct AddrRange {
  uint64_t base = 0;
  uint64_t size = 0;
  bool valid = false; // false if address could not be resolved to a constant
};

static bool overlaps(const AddrRange &a, const AddrRange &b) {
  if (!a.valid || !b.valid)
    return true; // conservative: unknown address => may alias
  return a.base < b.base + b.size && b.base < a.base + a.size;
}

/// Resolve the element size in bytes for a tile element type.
static uint64_t getElementBytes(Type elemTy) {
  if (auto intTy = dyn_cast<IntegerType>(elemTy))
    return intTy.getWidth() / 8;
  if (auto floatTy = dyn_cast<FloatType>(elemTy))
    return floatTy.getWidth() / 8;
  // bf16 is a FloatType (width 16) handled above; fall back to 4 bytes.
  return 4;
}

/// Compute the byte size covered by a tile buffer from its TileBufType shape.
static uint64_t getTileByteSize(TileBufType tileTy) {
  auto shape = tileTy.getShape();
  uint64_t elems = 1;
  for (int64_t dim : shape) {
    if (dim <= 0)
      return 0; // dynamic => unknown size; caller treats as may-alias
    elems *= static_cast<uint64_t>(dim);
  }
  return elems * getElementBytes(tileTy.getElementType());
}

/// Trace a tile operand back to its `pto.alloc_tile` and resolve its UB
/// address range. Returns std::nullopt if the operand is not a tile_buf or
/// its address is not a constant.
static std::optional<AddrRange> resolveTileAddrRange(Value tileVal) {
  auto tileTy = dyn_cast<TileBufType>(tileVal.getType());
  if (!tileTy)
    return std::nullopt;

  AddrRange range;
  range.size = getTileByteSize(tileTy);

  // Trace back through defining op (alloc_tile produces the tile_buf).
  auto alloc = tileVal.getDefiningOp<pto::AllocTileOp>();
  if (!alloc) {
    // Not directly from alloc_tile (e.g. a subview / bind_tile). Be
    // conservative: treat as unknown address that may alias anything.
    range.valid = false;
    return range;
  }

  Value addrVal = alloc.getAddr();
  if (!addrVal) {
    range.valid = false;
    return range;
  }
  if (auto cInt = addrVal.getDefiningOp<arith::ConstantIntOp>()) {
    range.base = static_cast<uint64_t>(cInt.value());
    range.valid = true;
    return range;
  }
  range.valid = false;
  return range;
}

/// One memory access of a tile op: an address range plus whether it is a
/// read or a write.
struct TileAccess {
  AddrRange range;
  bool isWrite;
};

/// Collect the read and write tile accesses of a V-pipeline tile op via its
/// PTO_DpsInitOpInterface: the `getDpsInits()` operands are writes, all other
/// tile_buf operands are reads.
static SmallVector<TileAccess> collectTileAccesses(Operation *op) {
  SmallVector<TileAccess> accesses;
  auto dps = dyn_cast<pto::PTO_DpsInitOpInterface>(op);
  if (!dps)
    return accesses;

  // Build the set of write (init) operands.
  SmallPtrSet<OpOperand *, 4> writeOperands;
  MutableOperandRange inits = dps.getDpsInitsMutable();
  for (OpOperand &initOp : inits) {
    if (isa<TileBufType>(initOp.get().getType()))
      writeOperands.insert(&initOp);
  }

  for (OpOperand &operand : op->getOpOperands()) {
    if (!isa<TileBufType>(operand.get().getType()))
      continue;
    bool isWrite = writeOperands.count(&operand) > 0;
    if (auto range = resolveTileAddrRange(operand.get()))
      accesses.push_back({*range, isWrite});
  }
  return accesses;
}

static bool isVMemBarrier(Operation *op) { return isa<pto::MemBarOp>(op); }

/// Returns true if a pto.mem_bar exists in (opA, opB) (open on opA, closed on
/// opB) within the same block, i.e. after opA and strictly before opB.
static bool hasBarrierBetween(Block::iterator opA, Operation *opB) {
  Block *block = opB->getBlock();
  if (!block)
    return false;
  for (auto it = std::next(opA); it != block->end(); ++it) {
    if (&*it == opB)
      return false;
    if (isVMemBarrier(&*it))
      return true;
  }
  return false;
}

/// Pick the MemBarKind for a dependency between opA and opB.
///   RAW (opA write -> opB read)   : VST_VLD
///   WAW (opA write -> opB write)  : VST_VST
///   WAR (opA read  -> opB write)  : VLD_VST
static MemBarKind classifyDep(const SmallVector<TileAccess> &aAcc,
                              const SmallVector<TileAccess> &bAcc) {
  bool raw = false, waw = false, war = false;
  for (const auto &wa : aAcc) {
    if (!wa.isWrite)
      continue;
    for (const auto &rb : bAcc) {
      if (!overlaps(wa.range, rb.range))
        continue;
      if (rb.isWrite)
        waw = true;
      else
        raw = true;
    }
  }
  for (const auto &ra : aAcc) {
    if (ra.isWrite)
      continue;
    for (const auto &wb : bAcc) {
      if (!wb.isWrite)
        continue;
      if (overlaps(ra.range, wb.range))
        war = true;
    }
  }
  // Prefer VST_VLD (RAW) first, then VST_VST (WAW), then VLD_VST (WAR).
  if (raw)
    return MemBarKind::VST_VLD;
  if (waw)
    return MemBarKind::VST_VST;
  if (war)
    return MemBarKind::VLD_VST;
  return MemBarKind::VV_ALL;
}

static bool hasOverlapDep(const SmallVector<TileAccess> &aAcc,
                          const SmallVector<TileAccess> &bAcc) {
  for (const auto &a : aAcc)
    for (const auto &b : bAcc) {
      if (!overlaps(a.range, b.range))
        continue;
      // any cross-op pair that is not (read,read) is a dependency
      if (a.isWrite || b.isWrite)
        return true;
    }
  return false;
}

struct PTOInsertVMemBarPass
    : public mlir::pto::impl::PTOInsertVMemBarBase<PTOInsertVMemBarPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // Collect insertion points first, then build, so we don't mutate IR while
    // iterating. Each entry: <opB to insert before, MemBarKind>.
    SmallVector<std::pair<Operation *, MemBarKind>, 16> insertions;

    // Gather V-pipeline tile ops across the function, grouped by their
    // enclosing block. Dependencies are only intra-block / intra-iteration.
    struct VTileOp {
      Operation *op;
      SmallVector<TileAccess> accesses;
    };
    llvm::DenseMap<Block *, SmallVector<VTileOp, 16>> byBlock;
    func.walk([&](Operation *op) {
      if (!isa<pto::OpPipeInterface>(op))
        return;
      auto pipeIface = cast<pto::OpPipeInterface>(op);
      if (pipeIface.getPipe() != pto::PIPE::PIPE_V)
        return;
      if (!isa<pto::PTO_DpsInitOpInterface>(op))
        return;
      auto accesses = collectTileAccesses(op);
      if (accesses.empty())
        return;
      Block *block = op->getBlock();
      if (!block)
        return;
      byBlock[block].push_back({op, std::move(accesses)});
    });

    for (auto &kv : byBlock) {
      SmallVector<VTileOp, 16> &vOps = kv.second;
      if (vOps.size() < 2)
        continue;

      // For each pair (i < j) with an overlapping dependency and no existing
      // barrier between them, schedule a mem_bar before vOps[j].
      for (unsigned j = 1; j < vOps.size(); ++j) {
        MemBarKind kind = MemBarKind::VV_ALL;
        bool anyDep = false;
        for (unsigned i = 0; i < j; ++i) {
          if (!hasOverlapDep(vOps[i].accesses, vOps[j].accesses))
            continue;
          // Only the nearest preceding op with a dependency matters for
          // ordering: if a barrier already exists between vOps[i] and
          // vOps[j], later dependences on vOps[j] are still possible from
          // ops after vOps[i]; but a barrier between any i<j and j already
          // guards all writes from ops up to that barrier. To stay simple
          // and correct, we check the nearest i<j without an intervening
          // barrier by scanning from j-1 downward.
          anyDep = true;
          break;
        }
        if (!anyDep)
          continue;

        // Recompute the kind using the *nearest* dependent predecessor that
        // has no existing barrier between it and opB.
        for (int i = static_cast<int>(j) - 1; i >= 0; --i) {
          if (!hasOverlapDep(vOps[i].accesses, vOps[j].accesses))
            continue;
          if (hasBarrierBetween(Block::iterator(vOps[i].op), vOps[j].op))
            break; // an existing barrier already guards this opB from prior
          kind = classifyDep(vOps[i].accesses, vOps[j].accesses);
          insertions.emplace_back(vOps[j].op, kind);
          break;
        }
      }
    }

    OpBuilder builder(func.getContext());
    for (auto [opB, kind] : insertions) {
      builder.setInsertionPoint(opB);
      auto attr = pto::MemBarAttr::get(func.getContext(), kind);
      builder.create<pto::MemBarOp>(opB->getLoc(), attr);
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOInsertVMemBarPass() {
  return std::make_unique<PTOInsertVMemBarPass>();
}

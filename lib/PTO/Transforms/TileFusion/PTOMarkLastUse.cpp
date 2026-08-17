// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Support/CodeConstants.h"
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>
#include <optional>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOMARKLASTUSE
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

static constexpr llvm::StringLiteral kFusionGroupIdAttr =
    "pto.fusion.group_id";
static constexpr llvm::StringLiteral kFusionOrderAttr = "pto.fusion.order";
static constexpr llvm::StringLiteral kLastUseAttrName = "pto.last_use";

struct GroupSpanMember {
  Operation *op = nullptr;
  int64_t order = 0;
};

struct GroupSpan {
  Block *block = nullptr;
  int64_t groupId = -1;
  SmallVector<GroupSpanMember, mlir::pto::kValue8> members;
};

static bool isTileType(Type type) {
  return isa<pto::TileBufType>(type);
}

static bool isDpsInitOperand(OpOperand &operand) {
  Operation *owner = operand.getOwner();
  if (auto dpsIface = dyn_cast<pto::PTO_DpsInitOpInterface>(owner)) {
    for (OpOperand &init : dpsIface.getDpsInitsMutable()) {
      if (&init == &operand) {
        return true;
      }
    }
  }
  return false;
}

static bool isTileOperand(OpOperand &operand) {
  return isTileType(operand.get().getType());
}

static bool isTileInputOperand(OpOperand &operand) {
  return isTileOperand(operand) && !isDpsInitOperand(operand);
}

// The last-use mask is indexed by tile operand slots only, in source operand
// order after filtering out scalar operands. DPS init/output tile slots are
// preserved and always materialize as 0.
static SmallVector<OpOperand *, mlir::pto::kValue4> collectTileOperands(Operation *op) {
  SmallVector<OpOperand *, mlir::pto::kValue4> tileOperands;
  for (OpOperand &operand : op->getOpOperands()) {
    if (isTileOperand(operand)) {
      tileOperands.push_back(&operand);
    }
  }
  return tileOperands;
}

static std::optional<int64_t> getRequiredI64Attr(Operation *op,
                                                 StringRef attrName) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(attrName)) {
    return attr.getInt();
  }
  return std::nullopt;
}

static bool hasIncompleteFusionMetadata(Operation *op) {
  const bool hasGroupId = op->hasAttr(kFusionGroupIdAttr);
  const bool hasOrder = op->hasAttr(kFusionOrderAttr);
  return hasGroupId != hasOrder;
}

static LogicalResult
collectGroupSpansInBlock(Block &block, SmallVectorImpl<GroupSpan> &spans) {
  DenseMap<int64_t, unsigned> spanIndexByGroupId;

  GroupSpan current;

  auto flush = [&]() -> LogicalResult {
    if (current.members.empty()) {
      return success();
    }

    current.block = &block;
    auto [it, inserted] =
        spanIndexByGroupId.try_emplace(current.groupId, spans.size());
    if (!inserted) {
      spans[it->second].members.clear();
      current = GroupSpan();
      return success();
    }

    spans.push_back(std::move(current));
    current = GroupSpan();
    return success();
  };

  for (Operation &op : block) {
    if (hasIncompleteFusionMetadata(&op)) {
      op.emitError("expected pto.fusion.group_id and pto.fusion.order to "
                   "either both exist or both be absent");
      return failure();
    }

    std::optional<int64_t> groupId =
        getRequiredI64Attr(&op, kFusionGroupIdAttr);
    if (!groupId) {
      if (!current.members.empty()) {
        // Non-fusion ops between fusion group members are tolerated (e.g.
        // alloc_tile). Only flush when the next fusion op belongs to a
        // different group.
        continue;
      }
      if (failed(flush())) {
        return failure();
      }
      continue;
    }

    std::optional<int64_t> order = getRequiredI64Attr(&op, kFusionOrderAttr);
    if (!order) {
      op.emitError("missing required pto.fusion.order attribute");
      return failure();
    }

    if (current.members.empty()) {
      current.groupId = *groupId;
      current.members.push_back(GroupSpanMember{&op, *order});
      continue;
    }

    if (current.groupId != *groupId) {
      if (failed(flush())) {
        return failure();
      }
      current.groupId = *groupId;
    }

    if (!current.members.empty() && current.members.back().order >= *order) {
      op.emitError("expected contiguous fusion span to follow increasing "
                   "pto.fusion.order");
      return failure();
    }

    current.members.push_back(GroupSpanMember{&op, *order});
  }

  return flush();
}

static bool isSpanLocalLastUseCandidate(Value value, Operation *currentOp,
                                        Block *block) {
  if (!value) {
    return false;
  }

  for (OpOperand &use : value.getUses()) {
    Operation *user = use.getOwner();
    if (user == currentOp) {
      continue;
    }
    if (user->getBlock() != block) {
      return false;
    }
    if (currentOp->isBeforeInBlock(user)) {
      return false;
    }
  }
  return true;
}

static bool hasLaterUseAfterSpan(Value value, Operation *spanEnd, Block *block) {
  for (OpOperand &use : value.getUses()) {
    Operation *user = use.getOwner();
    if (user->getBlock() != block) {
      return true;
    }
    if (spanEnd->isBeforeInBlock(user)) {
      return true;
    }
  }
  return false;
}

static bool isHardSpanBarrier(Operation *op) {
  if (op->hasTrait<OpTrait::IsTerminator>() || !op->getRegions().empty()) {
    return true;
  }
  if (isa<CallOpInterface>(op)) {
    return true;
  }
  return false;
}

static bool hasHardBarrierInSpan(const GroupSpan &span) {
  if (span.members.size() < mlir::pto::kValue2) {
    return false;
  }
  for (size_t i = 0; i + 1 < span.members.size(); ++i) {
    Operation *cur = span.members[i].op;
    Operation *next = span.members[i + 1].op;
    for (Operation *cursor = cur->getNextNode(); cursor && cursor != next;
         cursor = cursor->getNextNode()) {
      if (isHardSpanBarrier(cursor)) {
        return true;
      }
    }
  }
  return false;
}

static void markGroupSpanLastUse(const GroupSpan &span) {
  if (span.members.empty()) {
    return;
  }

  if (hasHardBarrierInSpan(span)) {
    return;
  }

  Block &block = *span.block;
  Operation *spanEnd = span.members.back().op;
  for (const GroupSpanMember &member : span.members) {
    Operation &op = *member.op;
    SmallVector<OpOperand *, mlir::pto::kValue4> tileOperands = collectTileOperands(&op);
    if (tileOperands.empty()) {
      op.removeAttr(kLastUseAttrName);
      continue;
    }

    SmallVector<int64_t, mlir::pto::kValue8> lastUseMask;
    lastUseMask.reserve(tileOperands.size());
    for (OpOperand *operand : tileOperands) {
      if (!isTileInputOperand(*operand)) {
        lastUseMask.push_back(0);
        continue;
      }
      // isSpanLocalLastUseCandidate的检查范围大于hasLaterUseAfterSpan
      bool blockedByLaterSpanUse =
          !isSpanLocalLastUseCandidate(operand->get(), &op, &block);
      bool blockedByLaterPostSpanUse =
          hasLaterUseAfterSpan(operand->get(), spanEnd, &block);
      lastUseMask.push_back(
          (!blockedByLaterSpanUse && !blockedByLaterPostSpanUse) ? 1 : 0);
    }

    op.setAttr(kLastUseAttrName,
               Builder(op.getContext()).getDenseI64ArrayAttr(lastUseMask));
  }
}

static LogicalResult markRegionLastUse(Region &region) {
  for (Block &block : region.getBlocks()) {
    SmallVector<GroupSpan, mlir::pto::kValue8> spans;
    if (failed(collectGroupSpansInBlock(block, spans))) {
      return failure();
    }
    for (const GroupSpan &span : spans) {
      markGroupSpanLastUse(span);
    }

    for (Operation &op : block) {
      for (Region &nestedRegion : op.getRegions()) {
        if (failed(markRegionLastUse(nestedRegion))) {
          return failure();
        }
      }
    }
  }
  return success();
}

struct PTOMarkLastUsePass
    : public pto::impl::PTOMarkLastUseBase<PTOMarkLastUsePass> {
  using pto::impl::PTOMarkLastUseBase<
      PTOMarkLastUsePass>::PTOMarkLastUseBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal()) {
      return;
    }

    if (failed(markRegionLastUse(func.getRegion()))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOMarkLastUsePass() {
  return std::make_unique<PTOMarkLastUsePass>();
}

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
#include "PTO/Transforms/TileFusion/FusionAnalysis.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOFUSIONREGIONGEN
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

static constexpr llvm::StringLiteral kFusionGroupIdAttr =
    "pto.fusion.group_id";
static constexpr llvm::StringLiteral kFusionOrderAttr = "pto.fusion.order";
static constexpr llvm::StringLiteral kFusionRowUnrollAttr =
    "pto.fusion.row_unroll_factor";
static constexpr llvm::StringLiteral kFusionColUnrollAttr =
    "pto.fusion.col_unroll_factor";

struct GroupSpanMember {
  Operation *op = nullptr;
  int64_t order = 0;
};

struct GroupSpan {
  Block *block = nullptr;
  int64_t groupId = -1;
  SmallVector<GroupSpanMember, mlir::pto::kValue8> members;
};

struct GroupSpanInterface {
  SmallVector<Value, mlir::pto::kValue8> externallyVisibleValues;
  SmallVector<Operation *, mlir::pto::kValue8> localDefs;
};

struct FusionBlockAnalysisIndex {
  DenseMap<Operation *, unsigned> nodeIdByOp;
  DenseMap<unsigned, SmallVector<const pto::FusionWriteInstanceLiveness *, 2>>
      writeInstancesByProducerNode;
};

struct PreFusionAnalysisIndex {
  DenseMap<Block *, FusionBlockAnalysisIndex> blocks;
};

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
flushGroupSpan(Block &block, DenseMap<int64_t, unsigned> &spanIndexByGroupId,
               GroupSpan &current, SmallVectorImpl<GroupSpan> &spans) {
  if (current.members.empty()) {
    return success();
  }
  current.block = &block;
  auto [it, inserted] =
      spanIndexByGroupId.try_emplace(current.groupId, spans.size());
  if (!inserted) {
    current.members.front().op->emitError(
        "expected one contiguous span per pto.fusion.group_id within a basic "
        "block");
    return failure();
  }
  spans.push_back(std::move(current));
  current = GroupSpan();
  return success();
}

static LogicalResult
collectGroupSpansInBlock(Block &block, SmallVectorImpl<GroupSpan> &spans) {
  DenseMap<int64_t, unsigned> spanIndexByGroupId;
  GroupSpan current;

  for (Operation &op : block) {
    if (hasIncompleteFusionMetadata(&op)) {
      op.emitError("expected pto.fusion.group_id and pto.fusion.order to "
                   "either both exist or both be absent");
      return failure();
    }

    std::optional<int64_t> groupId =
        getRequiredI64Attr(&op, kFusionGroupIdAttr);
    if (!groupId) {
      if (failed(flushGroupSpan(block, spanIndexByGroupId, current, spans))) {
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
      if (failed(flushGroupSpan(block, spanIndexByGroupId, current, spans))) {
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

  return flushGroupSpan(block, spanIndexByGroupId, current, spans);
}

static bool isNestedInOp(Operation *op, Operation *ancestor) {
  for (Operation *cur = op; cur; cur = cur->getParentOp()) {
    if (cur == ancestor) {
      return true;
    }
  }
  return false;
}

static bool isNestedInSpan(Operation *op, const DenseSet<Operation *> &spanOps) {
  for (Operation *cur = op; cur; cur = cur->getParentOp()) {
    if (spanOps.contains(cur)) {
      return true;
    }
  }
  return false;
}

static void appendUniqueValue(SmallVectorImpl<Value> &values,
                              DenseSet<Value> &seen, Value value) {
  if (seen.insert(value).second) {
    values.push_back(value);
  }
}

static Operation *getTopLevelAncestorInBlock(Operation *op, Block *block) {
  for (Operation *cur = op; cur; cur = cur->getParentOp()) {
    if (cur->getBlock() == block) {
      return cur;
    }
  }
  return nullptr;
}

static bool canReplaceUseWithRegionResult(OpOperand &use, Operation *boundary) {
  Operation *topLevel =
      getTopLevelAncestorInBlock(use.getOwner(), boundary->getBlock());
  if (!topLevel || topLevel == boundary) {
    return false;
  }
  return boundary->isBeforeInBlock(topLevel);
}

static bool hasReplaceableUseOutsideSpan(Value value,
                                         const DenseSet<Operation *> &spanOps,
                                         Operation *boundary) {
  for (OpOperand &use : value.getUses()) {
    if (isNestedInSpan(use.getOwner(), spanOps)) {
      continue;
    }
    if (canReplaceUseWithRegionResult(use, boundary)) {
      return true;
    }
  }
  return false;
}

static bool hasAnyUseOutsideSpan(Value value,
                                 const DenseSet<Operation *> &spanOps) {
  for (OpOperand &use : value.getUses()) {
    if (!isNestedInSpan(use.getOwner(), spanOps)) {
      return true;
    }
  }
  return false;
}

static const FusionBlockAnalysisIndex *
getBlockAnalysisIndex(const PreFusionAnalysisIndex *analysisIndex, Block *block) {
  if (!analysisIndex) {
    return nullptr;
  }
  auto it = analysisIndex->blocks.find(block);
  if (it == analysisIndex->blocks.end()) {
    return nullptr;
  }
  return &it->second;
}

static const pto::FusionWriteInstanceLiveness *
getProducedWriteInstance(const FusionBlockAnalysisIndex *blockAnalysis,
                         Operation *op, unsigned tileOutputIndex) {
  if (!blockAnalysis) {
    return nullptr;
  }

  auto nodeIt = blockAnalysis->nodeIdByOp.find(op);
  if (nodeIt == blockAnalysis->nodeIdByOp.end()) {
    return nullptr;
  }

  auto writeIt =
      blockAnalysis->writeInstancesByProducerNode.find(nodeIt->second);
  if (writeIt == blockAnalysis->writeInstancesByProducerNode.end()) {
    return nullptr;
  }
  if (tileOutputIndex >= writeIt->second.size()) {
    return nullptr;
  }
  return writeIt->second[tileOutputIndex];
}

static bool
writeInstanceEscapesSpan(const pto::FusionWriteInstanceLiveness &writeInstance,
                         const DenseSet<unsigned> &spanNodeIds) {
  if (writeInstance.hasExternalUsers || writeInstance.escapesBlock ||
      writeInstance.hasLocalBoundaryUsers ||
      writeInstance.hasLocalHardBoundaryUsers) {
    return true;
  }

  for (unsigned consumerNode : writeInstance.consumerNodes) {
    if (!spanNodeIds.contains(consumerNode)) {
      return true;
    }
  }
  return false;
}

static bool canSinkAllocTileDefToRegion(Value value, const GroupSpan &span,
                                        const DenseSet<Operation *> &spanOps) {
  auto alloc = dyn_cast_or_null<pto::AllocTileOp>(value.getDefiningOp());
  if (!alloc || alloc->getBlock() != span.block) {
    return false;
  }

  Operation *firstOp = span.members.front().op;
  if (!alloc->isBeforeInBlock(firstOp)) {
    return false;
  }

  for (OpOperand &use : value.getUses()) {
    if (isNestedInSpan(use.getOwner(), spanOps)) {
      continue;
    }
    if (!canReplaceUseWithRegionResult(use, firstOp)) {
      return false;
    }
  }

  return true;
}

struct SpanInterfaceContext {
  DenseSet<Operation *> spanOps;
  DenseSet<unsigned> spanNodeIds;
  Operation *boundary = nullptr;
  const FusionBlockAnalysisIndex *blockAnalysis = nullptr;
};

static SpanInterfaceContext
buildSpanInterfaceContext(const GroupSpan &span,
                          const PreFusionAnalysisIndex *analysisIndex) {
  SpanInterfaceContext context;
  context.boundary = span.members.front().op;
  context.blockAnalysis = getBlockAnalysisIndex(analysisIndex, span.block);
  for (const GroupSpanMember &member : span.members) {
    context.spanOps.insert(member.op);
    if (!context.blockAnalysis) {
      continue;
    }
    auto nodeIt = context.blockAnalysis->nodeIdByOp.find(member.op);
    if (nodeIt != context.blockAnalysis->nodeIdByOp.end()) {
      context.spanNodeIds.insert(nodeIt->second);
    }
  }
  return context;
}

static void collectVisibleSpanValues(const GroupSpan &span,
                                     const SpanInterfaceContext &context,
                                     GroupSpanInterface &iface) {
  DenseSet<Value> seenOutputs;
  for (const GroupSpanMember &member : span.members) {
    for (Value result : member.op->getResults()) {
      if (hasReplaceableUseOutsideSpan(result, context.spanOps,
                                       context.boundary)) {
        appendUniqueValue(iface.externallyVisibleValues, seenOutputs, result);
      }
    }
    auto dpsIface = dyn_cast<pto::PTO_DpsInitOpInterface>(member.op);
    if (!dpsIface) {
      continue;
    }
    unsigned tileOutputIndex = 0;
    for (Value init : dpsIface.getDpsInits()) {
      if (!isa<pto::TileBufType>(init.getType())) {
        continue;
      }
      const pto::FusionWriteInstanceLiveness *writeInstance =
          getProducedWriteInstance(context.blockAnalysis, member.op,
                                   tileOutputIndex++);
      bool escapes = hasReplaceableUseOutsideSpan(
          init, context.spanOps, context.boundary);
      if (writeInstance) {
        escapes &= writeInstanceEscapesSpan(*writeInstance,
                                            context.spanNodeIds);
      }
      if (escapes) {
        appendUniqueValue(iface.externallyVisibleValues, seenOutputs, init);
      }
    }
  }
}

static void collectSpanLocalDefs(const GroupSpan &span,
                                 const SpanInterfaceContext &context,
                                 GroupSpanInterface &iface) {
  DenseSet<Value> visibleValues(iface.externallyVisibleValues.begin(),
                                iface.externallyVisibleValues.end());
  DenseSet<Operation *> seenLocalDefs;
  for (const GroupSpanMember &member : span.members) {
    auto dpsIface = dyn_cast<pto::PTO_DpsInitOpInterface>(member.op);
    if (!dpsIface) {
      continue;
    }
    for (Value init : dpsIface.getDpsInits()) {
      bool cannotSink =
          !isa<pto::TileBufType>(init.getType()) ||
          !canSinkAllocTileDefToRegion(init, span, context.spanOps) ||
          (hasAnyUseOutsideSpan(init, context.spanOps) &&
           !visibleValues.contains(init));
      if (cannotSink) {
        continue;
      }
      Operation *defOp = init.getDefiningOp();
      if (seenLocalDefs.insert(defOp).second) {
        iface.localDefs.push_back(defOp);
      }
    }
  }
}

static GroupSpanInterface
buildGroupSpanInterface(const GroupSpan &span,
                        const PreFusionAnalysisIndex *analysisIndex) {
  GroupSpanInterface iface;
  SpanInterfaceContext context =
      buildSpanInterfaceContext(span, analysisIndex);
  collectVisibleSpanValues(span, context, iface);
  collectSpanLocalDefs(span, context, iface);
  return iface;
}

static void replaceEscapingUsesOutsideRegion(pto::FusionRegionOp fusionRegion,
                                             ArrayRef<Value> oldValues) {
  for (auto [oldValueRef, newValue] :
       llvm::zip(oldValues, fusionRegion.getOutputs())) {
    Value oldValue = oldValueRef;
    oldValue.replaceUsesWithIf(newValue, [&](OpOperand &use) {
      return !isNestedInOp(use.getOwner(), fusionRegion.getOperation()) &&
             canReplaceUseWithRegionResult(use, fusionRegion.getOperation());
    });
  }
}

static void clearSpanFusionMetadata(const GroupSpan &span) {
  for (const GroupSpanMember &member : span.members) {
    member.op->removeAttr(kFusionGroupIdAttr);
    member.op->removeAttr(kFusionOrderAttr);
    member.op->removeAttr(kFusionRowUnrollAttr);
    member.op->removeAttr(kFusionColUnrollAttr);
  }
}

static FailureOr<std::optional<int64_t>>
getCommonSpanI64Attr(const GroupSpan &span, StringRef attrName) {
  std::optional<int64_t> commonUnroll;
  bool sawMissingAttr = false;
  for (const GroupSpanMember &member : span.members) {
    auto attr = member.op->getAttrOfType<IntegerAttr>(attrName);
    if (!attr) {
      sawMissingAttr = true;
      continue;
    }
    int64_t unroll = attr.getInt();
    if (unroll < 1) {
      member.op->emitError("expected ")
          << attrName << " to be a positive integer";
      return failure();
    }
    if (!commonUnroll) {
      commonUnroll = unroll;
      continue;
    }
    if (*commonUnroll != unroll) {
      member.op->emitError("inconsistent ")
          << attrName << " within one pto.fusion.group_id";
      return failure();
    }
  }
  if (commonUnroll && sawMissingAttr) {
    span.members.front().op->emitError("incomplete ")
        << attrName << " within one pto.fusion.group_id";
    return failure();
  }
  return commonUnroll;
}

struct SpanUnrollFactors {
  std::optional<int64_t> row;
  std::optional<int64_t> col;
};

static FailureOr<SpanUnrollFactors>
getSpanUnrollFactors(const GroupSpan &span) {
  FailureOr<std::optional<int64_t>> row =
      getCommonSpanI64Attr(span, kFusionRowUnrollAttr);
  if (failed(row)) {
    return failure();
  }
  FailureOr<std::optional<int64_t>> col =
      getCommonSpanI64Attr(span, kFusionColUnrollAttr);
  if (failed(col)) {
    return failure();
  }
  return SpanUnrollFactors{*row, *col};
}

static pto::FusionRegionOp
createFusionRegion(const GroupSpan &span, const GroupSpanInterface &iface,
                   const SpanUnrollFactors &unroll) {
  SmallVector<Type, mlir::pto::kValue8> outputTypes;
  for (Value output : iface.externallyVisibleValues) {
    outputTypes.push_back(output.getType());
  }
  OpBuilder builder(span.members.front().op);
  auto region = builder.create<pto::FusionRegionOp>(
      span.members.front().op->getLoc(), TypeRange(outputTypes));
  region->setAttr(kFusionGroupIdAttr, builder.getI64IntegerAttr(span.groupId));
  if (unroll.row) {
    region->setAttr(kFusionRowUnrollAttr,
                    builder.getI64IntegerAttr(*unroll.row));
  }
  if (unroll.col) {
    region->setAttr(kFusionColUnrollAttr,
                    builder.getI64IntegerAttr(*unroll.col));
  }
  return region;
}

static LogicalResult populateFusionRegion(pto::FusionRegionOp region,
                                          const GroupSpan &span,
                                          const GroupSpanInterface &iface) {
  Block *body = new Block();
  region.getBody().push_back(body);
  for (Operation *localDef : iface.localDefs) {
    localDef->moveBefore(body, body->end());
  }
  for (const GroupSpanMember &member : span.members) {
    member.op->moveBefore(body, body->end());
  }
  clearSpanFusionMetadata(span);
  OpBuilder::atBlockEnd(body).create<pto::YieldOp>(
      region.getLoc(), ValueRange(iface.externallyVisibleValues));
  return verify(region.getOperation());
}

static LogicalResult
encapsulateGroupSpan(const GroupSpan &span,
                     const PreFusionAnalysisIndex *analysisIndex) {
  if (span.members.empty()) {
    return success();
  }

  GroupSpanInterface iface = buildGroupSpanInterface(span, analysisIndex);
  FailureOr<SpanUnrollFactors> unroll = getSpanUnrollFactors(span);
  if (failed(unroll)) {
    return failure();
  }
  pto::FusionRegionOp fusionRegion = createFusionRegion(span, iface, *unroll);
  if (failed(populateFusionRegion(fusionRegion, span, iface))) {
    return failure();
  }
  replaceEscapingUsesOutsideRegion(fusionRegion, iface.externallyVisibleValues);
  return success();
}

static LogicalResult processRegion(Region &region,
                                   const PreFusionAnalysisIndex *analysisIndex) {
  for (Block &block : region.getBlocks()) {
    SmallVector<Region *, mlir::pto::kValue4> nestedRegions;
    for (Operation &op : block) {
      for (Region &nestedRegion : op.getRegions()) {
        nestedRegions.push_back(&nestedRegion);
      }
    }

    for (Region *nestedRegion : nestedRegions) {
      if (failed(processRegion(*nestedRegion, analysisIndex))) {
        return failure();
      }
    }

    SmallVector<GroupSpan, mlir::pto::kValue8> spans;
    if (failed(collectGroupSpansInBlock(block, spans))) {
      return failure();
    }

    for (const GroupSpan &span : spans) {
      if (failed(encapsulateGroupSpan(span, analysisIndex))) {
        return failure();
      }
    }
  }
  return success();
}

struct PTOFusionRegionGenPass
    : public pto::impl::PTOFusionRegionGenBase<PTOFusionRegionGenPass> {
  using pto::impl::PTOFusionRegionGenBase<
      PTOFusionRegionGenPass>::PTOFusionRegionGenBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal()) {
      return;
    }

    // Reuse the shared pre-fusion dataflow graph cached by the analysis
    // manager (built once, by FusionPlan or lazily here).  FusionRegionGen
    // consumes only compute-node ids and write-instance pointers, never the
    // iteration-domain classes, so it does not need the --enable-shape-inference
    // option and can use the cached DFG directly.
    const pto::PreFusionAnalysis &sharedAnalysis =
        getAnalysis<pto::PreFusionAnalysis>();
    if (!sharedAnalysis.isValid()) {
      signalPassFailure();
      return;
    }
    const pto::PreFusionAnalysisResult &analysis = sharedAnalysis.getResult();

    PreFusionAnalysisIndex analysisIndex;
    for (const pto::FusionBlockAnalysis &blockAnalysis : analysis.blocks) {
      FusionBlockAnalysisIndex &index = analysisIndex.blocks[blockAnalysis.block];
      for (const pto::FusionComputeNode &node : blockAnalysis.computeNodes) {
        index.nodeIdByOp.try_emplace(node.op, node.id);
      }
      for (const pto::FusionWriteInstanceLiveness &writeInstance :
           blockAnalysis.writeInstances) {
        if (!writeInstance.producerNode) {
          continue;
        }
        index.writeInstancesByProducerNode[*writeInstance.producerNode]
            .push_back(&writeInstance);
      }
    }

    if (failed(processRegion(func.getRegion(), &analysisIndex))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOFusionRegionGenPass() {
  return std::make_unique<PTOFusionRegionGenPass>();
}

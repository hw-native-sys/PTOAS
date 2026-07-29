// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This pass owns the bounded, cost-ranked search across competing DFG paths.
// PTOFusionPlan.cpp intentionally retains the original single-path greedy
// planner used by the production fusion pipeline.

#include "PTO/Transforms/TileFusion/FusionAnalysis.h"
#include "PTO/Transforms/TileFusion/FusionOpSemantics.h"

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"

#include <algorithm>
#include <optional>

namespace mlir {
namespace pto {
// Passes.h (included above) pulls in the global GEN_PASS_DECL block, which
// defines GEN_PASS_DECL_FUSIONPLANSEARCH and leaves it set.  Undef it before
// re-including the .inc for GEN_PASS_DEF so the options struct is not defined
// twice.
#undef GEN_PASS_DECL_FUSIONPLANSEARCH
#define GEN_PASS_DEF_FUSIONPLANSEARCH
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

static constexpr llvm::StringLiteral kFusionGroupIdAttr = "pto.fusion.group_id";
static constexpr llvm::StringLiteral kFusionOrderAttr = "pto.fusion.order";

struct PlannedFusionGroup {
  SmallVector<const pto::FusionComputeNode *, 8> members;
};

struct PlanningContext {
  const pto::FusionBlockAnalysis &blockAnalysis;
};

struct PlanningCost {
  int64_t dependencyBenefit = 0;
  int64_t loopMergeBenefit = 0;
  int64_t liveTilePenalty = 0;
  int64_t vfParameterPenalty = 0;
  bool rejectedForDynamicShape = false;

  int64_t total() const {
    return dependencyBenefit + loopMergeBenefit - liveTilePenalty -
           vfParameterPenalty;
  }
};

struct PlanningDecision {
  bool accept = false;
  PlanningCost cost;
};

static bool isCurrentlyPlannableOp(StringRef opName) {
  return llvm::StringSwitch<bool>(opName)
      .Cases("tmul", "tdiv", "tadd", "tsub", "tmax", "tmin", true)
      .Cases("tmuls", "tdivs", "tadds", "tsubs", "tmaxs", "tmins", true)
      .Case("texp", true)
      .Case("texpands", true)
      .Cases("trowexpandsub", "trowexpandmul", "trowexpanddiv", true)
      .Cases("trowsum", "trowmax", "trowmin", true)
      .Cases("tcolsum", "tcolmax", "tcolmin", true)
      .Default(false);
}

static bool
isProvenIterationDomain(const pto::FusionBlockAnalysis &blockAnalysis,
                        const pto::FusionComputeNode &node) {
  if (node.iterationDomainClass >= blockAnalysis.iterationDomainClasses.size())
    return false;
  return blockAnalysis.iterationDomainClasses[node.iterationDomainClass]
             .info.proof == pto::IterationDomainProof::Proven;
}

static bool hasHardBoundaryBetween(const pto::FusionComputeNode &a,
                                   const pto::FusionComputeNode &b) {
  const pto::FusionComputeNode &earlier = a.blockOrder < b.blockOrder ? a : b;
  const pto::FusionComputeNode &later = a.blockOrder < b.blockOrder ? b : a;

  Operation *cursor = earlier.op->getNextNode();
  while (cursor && cursor != later.op) {
    if (cursor->hasTrait<OpTrait::IsTerminator>() ||
        !cursor->getRegions().empty() || isa<CallOpInterface>(cursor))
      return true;
    cursor = cursor->getNextNode();
  }
  return false;
}

static bool
hasHardBoundaryToGroup(ArrayRef<const pto::FusionComputeNode *> group,
                       const pto::FusionComputeNode &candidate) {
  for (const pto::FusionComputeNode *member : group)
    if (hasHardBoundaryBetween(*member, candidate))
      return true;
  return false;
}

static bool dependsOnPreviousNode(const pto::FusionBlockAnalysis &blockAnalysis,
                                  const pto::FusionComputeNode &previous,
                                  const pto::FusionComputeNode &current) {
  for (unsigned edgeId : current.incomingEdges) {
    if (edgeId >= blockAnalysis.edges.size())
      continue;
    if (blockAnalysis.edges[edgeId].producerNode == previous.id)
      return true;
  }

  for (Value output : previous.semantics.tileOutputs)
    if (llvm::is_contained(current.semantics.tileInputs, output))
      return true;

  return false;
}

static SmallVector<const pto::FusionComputeNode *, 8>
buildStableInGroupOrder(ArrayRef<const pto::FusionComputeNode *> members) {
  SmallVector<const pto::FusionComputeNode *, 8> ordered(members.begin(),
                                                         members.end());
  llvm::stable_sort(ordered, [](const pto::FusionComputeNode *lhs,
                                const pto::FusionComputeNode *rhs) {
    if (lhs->blockOrder != rhs->blockOrder)
      return lhs->blockOrder < rhs->blockOrder;
    return lhs->id < rhs->id;
  });
  return ordered;
}

static void assignStableGroupMetadata(ArrayRef<PlannedFusionGroup> groups,
                                      MLIRContext *ctx, int64_t &nextGroupId) {
  SmallVector<const PlannedFusionGroup *, 8> orderedGroups;
  orderedGroups.reserve(groups.size());
  for (const PlannedFusionGroup &group : groups)
    orderedGroups.push_back(&group);

  llvm::stable_sort(orderedGroups, [](const PlannedFusionGroup *lhs,
                                      const PlannedFusionGroup *rhs) {
    const pto::FusionComputeNode *lhsFirst = lhs->members.front();
    const pto::FusionComputeNode *rhsFirst = rhs->members.front();
    if (lhsFirst->blockOrder != rhsFirst->blockOrder)
      return lhsFirst->blockOrder < rhsFirst->blockOrder;
    return lhsFirst->id < rhsFirst->id;
  });

  for (const PlannedFusionGroup *group : orderedGroups) {
    const int64_t groupId = nextGroupId++;
    SmallVector<const pto::FusionComputeNode *, 8> stableOrder =
        buildStableInGroupOrder(group->members);
    for (auto [order, node] : llvm::enumerate(stableOrder)) {
      node->op->setAttr(kFusionGroupIdAttr,
                        IntegerAttr::get(IntegerType::get(ctx, 64), groupId));
      node->op->setAttr(kFusionOrderAttr,
                        IntegerAttr::get(IntegerType::get(ctx, 64),
                                         static_cast<int64_t>(order)));
    }
  }
}

static bool isSupportedPlanningNode(const pto::FusionComputeNode &node) {
  return node.semantics.kind == pto::FusionOpKind::Compute &&
         isCurrentlyPlannableOp(node.semantics.opName);
}

static unsigned
countEdgesFromGroup(const pto::FusionBlockAnalysis &blockAnalysis,
                    ArrayRef<const pto::FusionComputeNode *> group,
                    const pto::FusionComputeNode &candidate) {
  DenseSet<unsigned> producerIds;
  for (const pto::FusionComputeNode *member : group)
    producerIds.insert(member->id);

  unsigned count = 0;
  for (unsigned edgeId : candidate.incomingEdges) {
    if (edgeId >= blockAnalysis.edges.size())
      continue;
    if (producerIds.contains(blockAnalysis.edges[edgeId].producerNode))
      ++count;
  }
  return count;
}

struct GroupFootprint {
  unsigned liveTileCount = 0;
  unsigned vfParameterCount = 0;
};

static bool
nodesHaveDirectDataFlowConnection(const pto::FusionBlockAnalysis &blockAnalysis,
                                  const pto::FusionComputeNode &lhs,
                                  const pto::FusionComputeNode &rhs) {
  for (unsigned edgeId : lhs.outgoingEdges) {
    if (edgeId >= blockAnalysis.edges.size())
      continue;
    if (blockAnalysis.edges[edgeId].consumerNode == rhs.id)
      return true;
  }

  for (unsigned edgeId : lhs.incomingEdges) {
    if (edgeId >= blockAnalysis.edges.size())
      continue;
    if (blockAnalysis.edges[edgeId].producerNode == rhs.id)
      return true;
  }

  for (Value output : lhs.semantics.tileOutputs)
    if (llvm::is_contained(rhs.semantics.tileInputs, output))
      return true;

  for (Value output : rhs.semantics.tileOutputs)
    if (llvm::is_contained(lhs.semantics.tileInputs, output))
      return true;

  return false;
}

static unsigned
countConnectionsToGroup(const pto::FusionBlockAnalysis &blockAnalysis,
                        ArrayRef<const pto::FusionComputeNode *> group,
                        const pto::FusionComputeNode &candidate) {
  unsigned connections = 0;
  for (const pto::FusionComputeNode *member : group)
    if (nodesHaveDirectDataFlowConnection(blockAnalysis, *member, candidate))
      ++connections;
  return connections;
}

static unsigned
countInternalEdges(const pto::FusionBlockAnalysis &blockAnalysis,
                   ArrayRef<const pto::FusionComputeNode *> group) {
  DenseSet<unsigned> memberIds;
  for (const pto::FusionComputeNode *member : group)
    memberIds.insert(member->id);

  return llvm::count_if(blockAnalysis.edges,
                        [&](const pto::FusionDFGEdge &edge) {
                          return memberIds.contains(edge.producerNode) &&
                                 memberIds.contains(edge.consumerNode);
                        });
}

static unsigned
countInternalConnections(const pto::FusionBlockAnalysis &blockAnalysis,
                         ArrayRef<const pto::FusionComputeNode *> group) {
  unsigned connections = 0;
  for (auto [index, lhs] : llvm::enumerate(group))
    for (const pto::FusionComputeNode *rhs : group.drop_front(index + 1))
      if (nodesHaveDirectDataFlowConnection(blockAnalysis, *lhs, *rhs))
        ++connections;
  return connections;
}

static GroupFootprint
computeGroupFootprint(ArrayRef<const pto::FusionComputeNode *> members) {
  DenseSet<Value> producedTiles;
  DenseSet<Value> touchedTiles;
  DenseSet<Value> externalInputs;

  for (const pto::FusionComputeNode *member : members) {
    for (Value output : member->semantics.tileOutputs) {
      producedTiles.insert(output);
      touchedTiles.insert(output);
    }
  }

  for (const pto::FusionComputeNode *member : members) {
    for (Value input : member->semantics.tileInputs) {
      touchedTiles.insert(input);
      if (!producedTiles.contains(input))
        externalInputs.insert(input);
    }
  }

  GroupFootprint footprint;
  footprint.liveTileCount = touchedTiles.size();
  footprint.vfParameterCount = externalInputs.size() + producedTiles.size();
  return footprint;
}

static PlanningCost
computePlanningCost(ArrayRef<const pto::FusionComputeNode *> members,
                    unsigned connectionCount, int64_t loopMergeBenefit,
                    unsigned liveTileLimit, unsigned vfParameterLimit) {
  GroupFootprint footprint = computeGroupFootprint(members);

  PlanningCost cost;
  cost.dependencyBenefit = 4 * static_cast<int64_t>(connectionCount);
  cost.loopMergeBenefit = loopMergeBenefit;
  cost.liveTilePenalty = std::max<int64_t>(
      0, static_cast<int64_t>(footprint.liveTileCount) - liveTileLimit);
  cost.vfParameterPenalty = std::max<int64_t>(
      0, static_cast<int64_t>(footprint.vfParameterCount) - vfParameterLimit);
  return cost;
}

class CostModel {
public:
  virtual ~CostModel() = default;

  virtual PlanningDecision
  evaluateSeed(const PlanningContext &ctx,
               const pto::FusionComputeNode &candidate) const = 0;

  virtual PlanningCost
  evaluateGroup(const PlanningContext &ctx,
                ArrayRef<const pto::FusionComputeNode *> members) const = 0;

  virtual PlanningDecision
  evaluateAppend(const PlanningContext &ctx,
                 ArrayRef<const pto::FusionComputeNode *> currentGroup,
                 const pto::FusionComputeNode &candidate) const = 0;
};

class ConservativeGreedyCostModel final : public CostModel {
public:
  PlanningDecision
  evaluateSeed(const PlanningContext &ctx,
               const pto::FusionComputeNode &candidate) const override {
    PlanningDecision decision;
    if (!isSupportedPlanningNode(candidate))
      return decision;

    if (!isProvenIterationDomain(ctx.blockAnalysis, candidate)) {
      decision.cost.rejectedForDynamicShape = true;
      return decision;
    }

    decision.accept = true;
    return decision;
  }

  PlanningCost evaluateGroup(
      const PlanningContext &ctx,
      ArrayRef<const pto::FusionComputeNode *> members) const override {
    return computePlanningCost(
        members, countInternalEdges(ctx.blockAnalysis, members),
        /*loopMergeBenefit=*/2, /*liveTileLimit=*/4, /*vfParameterLimit=*/6);
  }

  PlanningDecision
  evaluateAppend(const PlanningContext &ctx,
                 ArrayRef<const pto::FusionComputeNode *> currentGroup,
                 const pto::FusionComputeNode &candidate) const override {
    PlanningDecision seedDecision = evaluateSeed(ctx, candidate);
    if (!seedDecision.accept)
      return seedDecision;

    PlanningDecision decision;
    if (currentGroup.empty()) {
      decision.accept = true;
      return decision;
    }

    const pto::FusionComputeNode &previous = *currentGroup.back();
    const bool sameDomainClass =
        previous.iterationDomainClass == candidate.iterationDomainClass;
    const bool contiguousInBlock =
        candidate.blockOrder == previous.blockOrder + 1;
    const bool directlyDependent =
        dependsOnPreviousNode(ctx.blockAnalysis, previous, candidate);
    if (!sameDomainClass || !contiguousInBlock || !directlyDependent)
      return decision;

    SmallVector<const pto::FusionComputeNode *, 8> proposedGroup(
        currentGroup.begin(), currentGroup.end());
    proposedGroup.push_back(&candidate);
    decision.cost = computePlanningCost(
        proposedGroup,
        countEdgesFromGroup(ctx.blockAnalysis, currentGroup, candidate),
        /*loopMergeBenefit=*/2, /*liveTileLimit=*/4, /*vfParameterLimit=*/6);
    decision.accept = decision.cost.total() > 0;
    return decision;
  }
};

class ConservativeDAGGreedyCostModel final : public CostModel {
public:
  PlanningDecision
  evaluateSeed(const PlanningContext &ctx,
               const pto::FusionComputeNode &candidate) const override {
    PlanningDecision decision;
    if (!isSupportedPlanningNode(candidate))
      return decision;

    if (!isProvenIterationDomain(ctx.blockAnalysis, candidate)) {
      decision.cost.rejectedForDynamicShape = true;
      return decision;
    }

    decision.accept = true;
    return decision;
  }

  PlanningCost evaluateGroup(
      const PlanningContext &ctx,
      ArrayRef<const pto::FusionComputeNode *> members) const override {
    return computePlanningCost(
        members, countInternalConnections(ctx.blockAnalysis, members),
        /*loopMergeBenefit=*/4, /*liveTileLimit=*/10,
        /*vfParameterLimit=*/12);
  }

  PlanningDecision
  evaluateAppend(const PlanningContext &ctx,
                 ArrayRef<const pto::FusionComputeNode *> currentGroup,
                 const pto::FusionComputeNode &candidate) const override {
    PlanningDecision seedDecision = evaluateSeed(ctx, candidate);
    if (!seedDecision.accept)
      return seedDecision;

    PlanningDecision decision;
    if (currentGroup.empty()) {
      decision.accept = true;
      return decision;
    }

    if (currentGroup.front()->iterationDomainClass !=
        candidate.iterationDomainClass)
      return decision;

    if (hasHardBoundaryToGroup(currentGroup, candidate))
      return decision;

    const unsigned connectionCount =
        countConnectionsToGroup(ctx.blockAnalysis, currentGroup, candidate);
    if (connectionCount == 0)
      return decision;

    SmallVector<const pto::FusionComputeNode *, 8> proposedGroup(
        currentGroup.begin(), currentGroup.end());
    proposedGroup.push_back(&candidate);
    decision.cost = computePlanningCost(
        proposedGroup, connectionCount, /*loopMergeBenefit=*/4,
        /*liveTileLimit=*/10, /*vfParameterLimit=*/12);
    decision.accept = decision.cost.total() > 0;
    return decision;
  }
};

class StrategyEngine {
public:
  virtual ~StrategyEngine() = default;

  virtual SmallVector<PlannedFusionGroup, 8>
  planBlock(const PlanningContext &ctx, const CostModel &costModel) const = 0;
};

class CostGuidedSearchStrategyEngine final : public StrategyEngine {
  // Keep a bounded, deterministically ranked frontier to prevent exponential
  // compile-time growth. This selects the best group among retained candidates;
  // it does not guarantee the global optimum across every connected subset.
  static constexpr unsigned kMaxCandidatesPerGroupSize = 64;

  struct GroupCandidate {
    SmallVector<const pto::FusionComputeNode *, 8> members;
    DenseSet<unsigned> memberIds;
    PlanningCost cost;
  };

  static bool haveSameMembers(const GroupCandidate &lhs,
                              const GroupCandidate &rhs) {
    if (lhs.members.size() != rhs.members.size())
      return false;
    return llvm::equal(lhs.members, rhs.members,
                       [](const pto::FusionComputeNode *lhsNode,
                          const pto::FusionComputeNode *rhsNode) {
                         return lhsNode->id == rhsNode->id;
                       });
  }

  static bool isBetterCandidate(const GroupCandidate &lhs,
                                const GroupCandidate &rhs) {
    if (lhs.cost.total() != rhs.cost.total())
      return lhs.cost.total() > rhs.cost.total();
    if (lhs.members.size() != rhs.members.size())
      return lhs.members.size() > rhs.members.size();

    for (auto [lhsNode, rhsNode] : llvm::zip(lhs.members, rhs.members)) {
      if (lhsNode->blockOrder != rhsNode->blockOrder)
        return lhsNode->blockOrder < rhsNode->blockOrder;
      if (lhsNode->id != rhsNode->id)
        return lhsNode->id < rhsNode->id;
    }
    return false;
  }

  static void addOrUpdateCandidate(SmallVectorImpl<GroupCandidate> &candidates,
                                   GroupCandidate candidate) {
    auto existing = llvm::find_if(candidates, [&](const GroupCandidate &other) {
      return haveSameMembers(candidate, other);
    });
    if (existing == candidates.end()) {
      candidates.push_back(std::move(candidate));
      return;
    }
    if (isBetterCandidate(candidate, *existing))
      *existing = std::move(candidate);
  }

public:
  SmallVector<PlannedFusionGroup, 8>
  planBlock(const PlanningContext &ctx,
            const CostModel &costModel) const override {
    SmallVector<PlannedFusionGroup, 8> groups;
    DenseSet<unsigned> assignedNodes;

    for (const pto::FusionComputeNode &seed : ctx.blockAnalysis.computeNodes) {
      if (assignedNodes.contains(seed.id))
        continue;

      PlanningDecision seedDecision = costModel.evaluateSeed(ctx, seed);
      if (!seedDecision.accept)
        continue;

      GroupCandidate seedCandidate;
      seedCandidate.members.push_back(&seed);
      seedCandidate.memberIds.insert(seed.id);
      seedCandidate.cost = costModel.evaluateGroup(ctx, seedCandidate.members);

      SmallVector<GroupCandidate, 64> frontier;
      frontier.push_back(std::move(seedCandidate));
      std::optional<GroupCandidate> bestCandidate;

      while (!frontier.empty()) {
        SmallVector<GroupCandidate, 64> nextFrontier;
        for (const GroupCandidate &current : frontier) {
          for (const pto::FusionComputeNode &candidate :
               ctx.blockAnalysis.computeNodes) {
            if (assignedNodes.contains(candidate.id) ||
                current.memberIds.contains(candidate.id))
              continue;

            PlanningDecision appendDecision =
                costModel.evaluateAppend(ctx, current.members, candidate);
            if (!appendDecision.accept)
              continue;

            GroupCandidate successor = current;
            successor.members.push_back(&candidate);
            successor.members = buildStableInGroupOrder(successor.members);
            successor.memberIds.insert(candidate.id);
            successor.cost = costModel.evaluateGroup(ctx, successor.members);
            addOrUpdateCandidate(nextFrontier, std::move(successor));
          }
        }

        llvm::stable_sort(nextFrontier, isBetterCandidate);
        if (nextFrontier.size() > kMaxCandidatesPerGroupSize)
          nextFrontier.resize(kMaxCandidatesPerGroupSize);

        for (const GroupCandidate &candidate : nextFrontier)
          if (!bestCandidate || isBetterCandidate(candidate, *bestCandidate))
            bestCandidate = candidate;

        frontier = std::move(nextFrontier);
      }

      if (!bestCandidate || bestCandidate->members.size() < 2)
        continue;

      PlannedFusionGroup group;
      group.members = bestCandidate->members;
      groups.push_back(group);
      for (const pto::FusionComputeNode *member : group.members)
        assignedNodes.insert(member->id);
    }

    return groups;
  }
};

static void clearPlanningAttrs(func::FuncOp func) {
  func.walk([](Operation *op) {
    op->removeAttr(kFusionGroupIdAttr);
    op->removeAttr(kFusionOrderAttr);
  });
}

struct FusionPlanSearchPass
    : public pto::impl::FusionPlanSearchBase<FusionPlanSearchPass> {
  using pto::impl::FusionPlanSearchBase<
      FusionPlanSearchPass>::FusionPlanSearchBase;

  FusionPlanSearchPass() = default;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal())
      return;

    clearPlanningAttrs(func);

    // Reuse the shared (analysis-manager-cached) pre-fusion dataflow graph
    // rather than rebuilding it from scratch.  The DFG — compute nodes, edges,
    // liveness, write instances — is identical regardless of whether shape
    // inference is enabled, so it is built once by PreFusionAnalysis and shared
    // with FusionRegionGen via the analysis manager.  Only iteration-domain
    // inference depends on the --enable-shape-inference option, so we run that
    // separable step ourselves on a local copy of the cached graph.
    const pto::PreFusionAnalysis &sharedAnalysis =
        getAnalysis<pto::PreFusionAnalysis>();
    if (!sharedAnalysis.isValid()) {
      signalPassFailure();
      return;
    }
    pto::PreFusionAnalysisResult analysis = sharedAnalysis.getResult();
    if (failed(
            pto::inferIterationDomainClasses(analysis, enableShapeInference))) {
      signalPassFailure();
      return;
    }

    MLIRContext *ctx = &getContext();
    int64_t nextGroupId = 0;
    ConservativeDAGGreedyCostModel costModel;
    CostGuidedSearchStrategyEngine strategyEngine;

    for (const pto::FusionBlockAnalysis &blockAnalysis : analysis.blocks) {
      PlanningContext planningCtx{blockAnalysis};
      SmallVector<PlannedFusionGroup, 8> groups =
          strategyEngine.planBlock(planningCtx, costModel);
      assignStableGroupMetadata(groups, ctx, nextGroupId);
    }

    // The fusion metadata we annotate (group_id/order) is a planning *output*;
    // it does not alter tile semantics, operand types, aliasing or liveness,
    // so it cannot invalidate the shared PreFusionAnalysis DFG.  Preserve it so
    // the downstream FusionRegionGen pass reuses the cached graph instead of
    // rebuilding it.
    markAnalysesPreserved<pto::PreFusionAnalysis>();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createFusionPlanSearchPass() {
  return std::make_unique<FusionPlanSearchPass>();
}

std::unique_ptr<Pass> mlir::pto::createFusionPlanSearchPass(
    const pto::FusionPlanSearchOptions &options) {
  return std::make_unique<FusionPlanSearchPass>(options);
}

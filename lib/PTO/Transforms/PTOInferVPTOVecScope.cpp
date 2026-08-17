// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOInferVPTOVecScope.cpp ------------------------------------------===//
//
// VPTO automatic vecscope inference.
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/Passes.h"

#include "PTO/Support/CodeConstants.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include <optional>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOINFERVPTOVECSCOPE
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

enum class VPTOInferenceOpClass {
  Vector,
  SafeScalar,
  Boundary,
};

struct NestedRegionSummary {
  bool hasVectorOperation = false;
  bool hasBoundaryOperation = false;
};

struct EscapingMovedValue {
  Value value;
  Operation *producer = nullptr;
  Operation *user = nullptr;
  bool requiresDiagnostic = false;
};

struct ResultlessScopePlan {
  SmallVector<Operation *, mlir::pto::kValue16> hoistOps;
  SmallVector<Operation *, mlir::pto::kValue16> moveOps;
};

struct LogicalScopePlan {
  size_t begin = 0;
  size_t end = 0;
  ResultlessScopePlan plan;
};

using SegmentRematCache =
    llvm::DenseMap<Value, llvm::DenseMap<Operation *, Value>>;

static VPTOInferenceOpClass classifyOperationForInference(Operation *op);
static LogicalResult
buildResultlessScopePlan(ArrayRef<Operation *> ops, ResultlessScopePlan &plan,
                         EscapingMovedValue &escapingValue);
static LogicalResult inferVecScopesInRegion(Region &region,
                                            MLIRContext *context);

static bool isVecScopeType(Type type) {
  return isa<pto::VRegType, pto::MaskType, pto::AlignType>(type);
}

static bool isPTOOperation(Operation *op) {
  return op && op->getName().getStringRef().starts_with("pto.");
}

static bool isExplicitVectorScopeCarrier(Operation *op) {
  return isa<pto::VecScopeOp, pto::StrictVecScopeOp>(op);
}

static bool isForbiddenInsideInferredVectorScope(Operation *op) {
  // Bisheng cannot expand block-query results produced inside an AIV vector
  // scope. Keep these scalar queries outside the inferred scope and capture
  // their results instead.
  return isa<pto::VbitsortOp, pto::Vmrgsort4Op, pto::GetBlockIdxOp,
             pto::GetBlockNumOp>(op);
}

static bool isVectorScopeBoundaryOperation(Operation *op) {
  return isa<pto::BarrierOp, pto::BarrierSyncOp>(op);
}

static bool hasVecScopeTypedOperandOrResult(Operation *op) {
  for (Type type : op->getOperandTypes()) {
    if (isVecScopeType(type)) {
      return true;
    }
  }
  for (Type type : op->getResultTypes()) {
    if (isVecScopeType(type)) {
      return true;
    }
  }
  return false;
}

static bool requiresVectorScope(Operation *op) {
  if (!isPTOOperation(op)) {
    return false;
  }

  return hasVecScopeTypedOperandOrResult(op) ||
         isa<pto::MemBarOp, pto::SprclrOp>(op);
}

static bool isAtomicControlFlowCandidate(Operation *op) {
  // Structured control flow is kept as one inference unit. In particular,
  // scf.while carries vector-scope values through both regions and must not
  // be split into a resultless vecscope around only one of its regions.
  return isa<scf::IfOp, scf::ForOp, scf::WhileOp>(op);
}

static bool isSafeScalarOperation(Operation *op) {
  if (op->getNumRegions() != 0) {
    return false;
  }
  if (op->hasTrait<OpTrait::IsTerminator>()) {
    return false;
  }
  if (isa<func::CallOp>(op)) {
    return false;
  }
  if (isPTOOperation(op) && !isMemoryEffectFree(op)) {
    return false;
  }
  return isMemoryEffectFree(op);
}

static bool isRematerializableVecScopeProducer(Operation *op) {
  if (!op || op->getNumRegions() != 0) {
    return false;
  }
  if (op->hasTrait<OpTrait::IsTerminator>()) {
    return false;
  }
  if (isa<func::CallOp>(op)) {
    return false;
  }
  return isMemoryEffectFree(op);
}

static void summarizeNestedRegionForAtomicCluster(
    Region &region, NestedRegionSummary &summary) {
  for (Block &block : region) {
    for (Operation &op : block) {
      if (op.hasTrait<OpTrait::IsTerminator>()) {
        continue;
      }

      switch (classifyOperationForInference(&op)) {
      case VPTOInferenceOpClass::Vector:
        summary.hasVectorOperation = true;
        break;
      case VPTOInferenceOpClass::SafeScalar:
        break;
      case VPTOInferenceOpClass::Boundary:
        summary.hasBoundaryOperation = true;
        return;
      }
    }
  }
}

static bool canTreatAsAtomicControlFlow(Operation *op) {
  if (!isAtomicControlFlowCandidate(op)) {
    return false;
  }

  NestedRegionSummary summary;
  for (Region &region : op->getRegions()) {
    summarizeNestedRegionForAtomicCluster(region, summary);
    if (summary.hasBoundaryOperation) {
      return false;
    }
  }
  return summary.hasVectorOperation;
}

static VPTOInferenceOpClass classifyOperationForInference(Operation *op) {
  if (!op) {
    return VPTOInferenceOpClass::Boundary;
  }

  if (isExplicitVectorScopeCarrier(op)) {
    return VPTOInferenceOpClass::Boundary;
  }
  if (op->hasTrait<OpTrait::IsTerminator>()) {
    return VPTOInferenceOpClass::Boundary;
  }
  if (isa<func::CallOp>(op)) {
    return VPTOInferenceOpClass::Boundary;
  }
  if (isVectorScopeBoundaryOperation(op)) {
    return VPTOInferenceOpClass::Boundary;
  }
  if (isForbiddenInsideInferredVectorScope(op)) {
    return VPTOInferenceOpClass::Boundary;
  }

  if (requiresVectorScope(op)) {
    return VPTOInferenceOpClass::Vector;
  }

  if (canTreatAsAtomicControlFlow(op)) {
    return VPTOInferenceOpClass::Vector;
  }

  if (isSafeScalarOperation(op)) {
    return VPTOInferenceOpClass::SafeScalar;
  }

  return VPTOInferenceOpClass::Boundary;
}

static bool hasVectorOperation(ArrayRef<Operation *> ops) {
  return llvm::any_of(ops, [](Operation *op) {
    return classifyOperationForInference(op) == VPTOInferenceOpClass::Vector;
  });
}

static bool isUserInsideCluster(Operation *user,
                                const llvm::SmallPtrSetImpl<Operation *> &ops) {
  for (Operation *cur = user; cur; cur = cur->getParentOp()) {
    if (ops.contains(cur)) {
      return true;
    }
  }
  return false;
}

static bool anyUserIsMoved(Value result,
                           const llvm::SmallPtrSetImpl<Operation *> &movedOps) {
  for (Operation *user : result.getUsers()) {
    if (isUserInsideCluster(user, movedOps)) {
      return true;
    }
  }
  return false;
}

static llvm::SmallPtrSet<Operation *, mlir::pto::kValue16>
computeMovedOpsForResultlessScope(ArrayRef<Operation *> ops) {
  llvm::SmallPtrSet<Operation *, mlir::pto::kValue16> movedOps;
  for (Operation *op : ops) {
    if (classifyOperationForInference(op) == VPTOInferenceOpClass::Vector) {
      movedOps.insert(op);
    }
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation *op : llvm::reverse(ops)) {
      if (movedOps.contains(op) ||
          classifyOperationForInference(op) !=
              VPTOInferenceOpClass::SafeScalar) {
        continue;
      }

      bool hasMovedUser = false;
      bool allUsersMoved = true;
      for (Value result : op->getResults()) {
        for (Operation *user : result.getUsers()) {
          if (isUserInsideCluster(user, movedOps)) {
            hasMovedUser = true;
            continue;
          }
          if (!isUserInsideCluster(user, movedOps)) {
            allUsersMoved = false;
            break;
          }
        }
        if (!allUsersMoved) {
          break;
        }
      }

      if (hasMovedUser && allUsersMoved) {
        movedOps.insert(op);
        changed = true;
      }
    }
  }
  return movedOps;
}

static Operation *getAncestorInBlock(Operation *op, Block &block) {
  for (Operation *cur = op; cur; cur = cur->getParentOp()) {
    if (cur->getBlock() == &block) {
      return cur;
    }
  }
  return nullptr;
}

static FailureOr<Operation *>
cloneVecScopeProducerForUse(
    Value value, Operation *user, Operation *logicalScopeAnchor,
    SegmentRematCache &cache, MLIRContext *context,
    llvm::DenseMap<Operation *, Operation *> &clones) {
  auto result = dyn_cast<OpResult>(value);
  if (!result) {
    return failure();
  }

  if (auto cacheIt = cache.find(value); cacheIt != cache.end()) {
    auto anchorIt = cacheIt->second.find(logicalScopeAnchor);
    if (anchorIt != cacheIt->second.end()) {
      return anchorIt->second.getDefiningOp();
    }
  }

  Operation *producer = result.getOwner();
  auto existing = clones.find(producer);
  if (existing != clones.end()) {
    return existing->second;
  }

  if (!isRematerializableVecScopeProducer(producer)) {
    return failure();
  }

  IRMapping mapping;
  for (Value operand : producer->getOperands()) {
    if (!isVecScopeType(operand.getType())) {
      continue;
    }

    FailureOr<Operation *> clonedOperandProducer =
        cloneVecScopeProducerForUse(operand, user, logicalScopeAnchor, cache,
                                    context, clones);
    if (failed(clonedOperandProducer)) {
      return failure();
    }

    auto operandResult = cast<OpResult>(operand);
    mapping.map(operand, (*clonedOperandProducer)
                             ->getResult(operandResult.getResultNumber()));
  }

  IRRewriter rewriter(context);
  rewriter.setInsertionPoint(user);
  Operation *clone = rewriter.clone(*producer, mapping);
  clones.try_emplace(producer, clone);
  cache[value][logicalScopeAnchor] = clone->getResult(result.getResultNumber());
  return clone;
}

static void collectGreedyLogicalScopePlans(
    ArrayRef<Operation *> ops, SmallVectorImpl<LogicalScopePlan> &plans) {
  plans.clear();

  for (size_t begin = 0; begin < ops.size();) {
    size_t bestEnd = begin;
    ResultlessScopePlan bestPlan;

    for (size_t end = ops.size(); end > begin; --end) {
      ArrayRef<Operation *> candidate = ops.slice(begin, end - begin);
      if (!hasVectorOperation(candidate)) {
        continue;
      }

      ResultlessScopePlan plan;
      EscapingMovedValue candidateEscapingValue;
      if (succeeded(
              buildResultlessScopePlan(candidate, plan, candidateEscapingValue))) {
        bestEnd = end;
        bestPlan = std::move(plan);
        break;
      }
    }

    if (bestEnd == begin) {
      ++begin;
      continue;
    }

    plans.push_back(LogicalScopePlan{begin, bestEnd, std::move(bestPlan)});
    begin = bestEnd;
  }
}

static void assignLogicalScopeAnchorsForCluster(
    ArrayRef<Operation *> ops,
    llvm::DenseMap<Operation *, Operation *> &logicalScopeAnchors) {
  SmallVector<LogicalScopePlan, mlir::pto::kValue8> plans;
  collectGreedyLogicalScopePlans(ops, plans);

  llvm::DenseMap<Operation *, Operation *> scopeAnchorByMovedOp;
  for (const LogicalScopePlan &plan : plans) {
    if (plan.plan.moveOps.empty()) {
      continue;
    }
    Operation *scopeAnchor = plan.plan.moveOps.front();
    for (Operation *movedOp : plan.plan.moveOps) {
      scopeAnchorByMovedOp[movedOp] = scopeAnchor;
    }
  }

  Operation *currentNonScopeAnchor = nullptr;
  for (Operation *op : ops) {
    auto scopeIt = scopeAnchorByMovedOp.find(op);
    if (scopeIt != scopeAnchorByMovedOp.end()) {
      logicalScopeAnchors[op] = scopeIt->second;
      currentNonScopeAnchor = nullptr;
      continue;
    }

    if (!currentNonScopeAnchor) {
      currentNonScopeAnchor = op;
    }
    logicalScopeAnchors[op] = currentNonScopeAnchor;
  }
}

static llvm::DenseMap<Operation *, Operation *>
computeLogicalScopeAnchors(Block &block) {
  llvm::DenseMap<Operation *, Operation *> logicalScopeAnchors;
  SmallVector<Operation *, mlir::pto::kValue32> pending;

  auto flush = [&logicalScopeAnchors, &pending]() {
    if (pending.empty()) {
      return;
    }
    assignLogicalScopeAnchorsForCluster(pending, logicalScopeAnchors);
    pending.clear();
  };

  for (Operation &op : block) {
    switch (classifyOperationForInference(&op)) {
    case VPTOInferenceOpClass::Vector:
    case VPTOInferenceOpClass::SafeScalar:
      pending.push_back(&op);
      break;
    case VPTOInferenceOpClass::Boundary:
      flush();
      break;
    }
  }
  flush();
  return logicalScopeAnchors;
}

static LogicalResult rematerializeEscapingValueForUserSegments(
    Value value, const llvm::SmallPtrSetImpl<Operation *> &movedOps,
    Block &block, SegmentRematCache &cache, MLIRContext *context) {
  auto result = dyn_cast<OpResult>(value);
  if (!result) {
    return failure();
  }

  llvm::DenseMap<Operation *, Operation *> logicalScopeAnchors =
      computeLogicalScopeAnchors(block);
  llvm::DenseMap<Operation *, SmallVector<OpOperand *, mlir::pto::kValue4>> usesBySegment;

  for (OpOperand &use : result.getUses()) {
    Operation *user = use.getOwner();
    if (isUserInsideCluster(user, movedOps)) {
      continue;
    }

    Operation *ancestor = getAncestorInBlock(user, block);
    if (!ancestor) {
      return failure();
    }

    auto anchorIt = logicalScopeAnchors.find(ancestor);
    if (anchorIt == logicalScopeAnchors.end()) {
      return failure();
    }

    usesBySegment[anchorIt->second].push_back(&use);
  }

  if (usesBySegment.empty()) {
    return failure();
  }

  for (auto &entry : usesBySegment) {
    Operation *logicalScopeAnchor = entry.first;
    SmallVectorImpl<OpOperand *> &uses = entry.second;
    Value replacement;
    if (auto cacheIt = cache.find(value); cacheIt != cache.end()) {
      auto anchorIt = cacheIt->second.find(logicalScopeAnchor);
      if (anchorIt != cacheIt->second.end()) {
        replacement = anchorIt->second;
      }
    }

    if (!replacement) {
      if (!logicalScopeAnchor) {
        return failure();
      }

      llvm::DenseMap<Operation *, Operation *> clones;
      FailureOr<Operation *> clonedProducer =
          cloneVecScopeProducerForUse(value, logicalScopeAnchor,
                                      logicalScopeAnchor, cache, context,
                                      clones);
      if (failed(clonedProducer)) {
        return failure();
      }

      replacement = (*clonedProducer)->getResult(result.getResultNumber());
      cache[value][logicalScopeAnchor] = replacement;
    }

    for (OpOperand *use : uses) {
      use->set(replacement);
    }
  }

  return success();
}

static bool findEscapingMovedResult(
    const llvm::SmallPtrSetImpl<Operation *> &movedOps,
    EscapingMovedValue &escapingValue) {
  for (Operation *op : movedOps) {
    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (isUserInsideCluster(user, movedOps)) {
          continue;
        }

        escapingValue.value = result;
        escapingValue.producer = op;
        escapingValue.user = user;
        escapingValue.requiresDiagnostic = isVecScopeType(result.getType());
        return true;
      }
    }
  }
  return false;
}

static LogicalResult
emitEscapingVectorScopeValueError(const EscapingMovedValue &escapingValue) {
  Operation *producer = escapingValue.producer;
  if (!producer) {
    return failure();
  }

  InFlightDiagnostic diag = producer->emitOpError()
                            << "cannot infer resultless pto.vecscope because "
                               "VPTO vector-scope data cannot have external "
                               "users";
  if (escapingValue.value) {
    diag << "; escaping value type is " << escapingValue.value.getType();
  }
  if (escapingValue.user) {
    diag.attachNote(escapingValue.user->getLoc())
        << "external user is here";
  }
  return failure();
}

// classify which operations need to be moved into a vecscope, which can be hoisted out of the
// vecscope, and check for any vector-scope-typed values that would escape the vecscope if we were to
// move the candidate operations into a resultless vecscope. Returns failure if the candidate cluster
// is not suitable for vecscope inference.
static LogicalResult
buildResultlessScopePlan(ArrayRef<Operation *> ops, ResultlessScopePlan &plan,
                         EscapingMovedValue &escapingValue) {
  if (ops.empty() || !hasVectorOperation(ops)) {
    return failure();
  }

  llvm::SmallPtrSet<Operation *, mlir::pto::kValue16> movedOps =
      computeMovedOpsForResultlessScope(ops);
  if (movedOps.empty()) {
    return failure();
  }

  if (findEscapingMovedResult(movedOps, escapingValue)) {
    return failure();
  }

  llvm::SmallPtrSet<Operation *, mlir::pto::kValue16> hoistedOps;
  for (Operation *op : ops) {
    if (movedOps.contains(op) ||
        classifyOperationForInference(op) != VPTOInferenceOpClass::SafeScalar) {
      continue;
    }

    for (Value result : op->getResults()) {
      if (anyUserIsMoved(result, movedOps)) {
        hoistedOps.insert(op);
        break;
      }
    }
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation *op : llvm::reverse(ops)) {
      if (movedOps.contains(op) || hoistedOps.contains(op) ||
          classifyOperationForInference(op) !=
              VPTOInferenceOpClass::SafeScalar) {
        continue;
      }

      bool feedsHoistedOp = false;
      for (Value result : op->getResults()) {
        for (Operation *user : result.getUsers()) {
          if (isUserInsideCluster(user, hoistedOps)) {
            feedsHoistedOp = true;
            break;
          }
        }
        if (feedsHoistedOp) {
          break;
        }
      }

      if (feedsHoistedOp) {
        hoistedOps.insert(op);
        changed = true;
      }
    }
  }

  plan.hoistOps.clear();
  plan.moveOps.clear();
  for (Operation *op : ops) {
    if (hoistedOps.contains(op)) {
      plan.hoistOps.push_back(op);
    }
    if (movedOps.contains(op)) {
      plan.moveOps.push_back(op);
    }
  }
  return success();
}

static void wrapCluster(const ResultlessScopePlan &plan, MLIRContext *context) {
  if (plan.moveOps.empty()) {
    return;
  }

  Operation *first = plan.moveOps.front();
  Block *parentBlock = first->getBlock();

  IRRewriter rewriter(context);
  rewriter.setInsertionPoint(first);
  auto scope = rewriter.create<pto::VecScopeOp>(first->getLoc());
  scope.getBody().push_back(new Block());

  for (Operation *op : plan.hoistOps) {
    if (op->getBlock() == parentBlock && scope->isBeforeInBlock(op)) {
      op->moveBefore(scope);
    }
  }

  Block &scopeBody = scope.getBody().front();
  for (Operation *op : plan.moveOps) {
    scopeBody.getOperations().splice(scopeBody.end(),
                                     parentBlock->getOperations(),
                                     Block::iterator(op));
  }
}

static LogicalResult wrapGreedySubclusters(ArrayRef<Operation *> ops,
                                           MLIRContext *context) {
  for (size_t begin = 0; begin < ops.size();) {
    size_t bestEnd = begin;
    ResultlessScopePlan bestPlan;
    EscapingMovedValue escapingValue;
    bool sawEscapingMovedResult = false;

    for (size_t end = ops.size(); end > begin; --end) {
      ArrayRef<Operation *> candidate = ops.slice(begin, end - begin);
      if (!hasVectorOperation(candidate)) {
        continue;
      }

      // Prefer the largest suffix-preserving candidate that actually needs a
      // vecscope and can be moved into today's resultless pto.vecscope form.
      ResultlessScopePlan plan;
      EscapingMovedValue candidateEscapingValue;
      if (succeeded(buildResultlessScopePlan(candidate, plan,
                                             candidateEscapingValue))) {
        bestEnd = end;
        bestPlan = std::move(plan);
        break;
      }

      if (!sawEscapingMovedResult && candidateEscapingValue.producer) {
        escapingValue = candidateEscapingValue;
        sawEscapingMovedResult = true;
      }
    }

    if (bestEnd == begin) {
      if (classifyOperationForInference(ops[begin]) ==
              VPTOInferenceOpClass::Vector &&
          sawEscapingMovedResult && escapingValue.requiresDiagnostic) {
        return emitEscapingVectorScopeValueError(escapingValue);
      }
      ++begin;
      continue;
    }

    wrapCluster(bestPlan, context);
    begin = bestEnd;
  }
  return success();
}

static FailureOr<bool> fixOneEscapingSubcluster(ArrayRef<Operation *> ops,
                                                SegmentRematCache &cache,
                                                MLIRContext *context) {
  for (size_t begin = 0; begin < ops.size();) {
    size_t bestEnd = begin;
    EscapingMovedValue escapingValue;
    bool sawEscapingMovedResult = false;
    size_t escapingCandidateBegin = begin;
    size_t escapingCandidateEnd = begin;

    for (size_t end = ops.size(); end > begin; --end) {
      ArrayRef<Operation *> candidate = ops.slice(begin, end - begin);
      if (!hasVectorOperation(candidate)) {
        continue;
      }

      ResultlessScopePlan ignoredPlan;
      EscapingMovedValue candidateEscapingValue;
      if (succeeded(buildResultlessScopePlan(candidate, ignoredPlan,
                                             candidateEscapingValue))) {
        bestEnd = end;
        break;
      }

      if (!sawEscapingMovedResult && candidateEscapingValue.producer) {
        escapingValue = candidateEscapingValue;
        sawEscapingMovedResult = true;
        escapingCandidateBegin = begin;
        escapingCandidateEnd = end;
      }
    }

    if (bestEnd == begin) {
      if (classifyOperationForInference(ops[begin]) ==
              VPTOInferenceOpClass::Vector &&
          sawEscapingMovedResult && escapingValue.requiresDiagnostic) {
        ArrayRef<Operation *> escapingCandidate =
            ops.slice(escapingCandidateBegin,
                      escapingCandidateEnd - escapingCandidateBegin);
        llvm::SmallPtrSet<Operation *, mlir::pto::kValue16> movedOps =
            computeMovedOpsForResultlessScope(escapingCandidate);
        Block *block = ops.front()->getBlock();
        if (!block) {
          return false;
        }

        if (succeeded(rematerializeEscapingValueForUserSegments(
                escapingValue.value, movedOps, *block, cache, context))) {
          return true;
        }
        return false;
      }
      ++begin;
      continue;
    }

    begin = bestEnd;
  }

  return false;
}

static LogicalResult repairEscapingSubclusters(Block &block,
                                               MLIRContext *context) {
  constexpr unsigned kMaxRepairIterations = 256;
  SegmentRematCache cache;
  for (unsigned iteration = 0; iteration < kMaxRepairIterations; ++iteration) {
    bool changedInIteration = false;
    SmallVector<Operation *, mlir::pto::kValue32> pending;
    SmallVector<Operation *, mlir::pto::kValue32> ops;
    for (Operation &op : block) {
      ops.push_back(&op);
    }

    auto flush = [&]() -> FailureOr<bool> {
      FailureOr<bool> changed =
          fixOneEscapingSubcluster(pending, cache, context);
      pending.clear();
      return changed;
    };

    for (Operation *op : ops) {
      switch (classifyOperationForInference(op)) {
      case VPTOInferenceOpClass::Vector:
      case VPTOInferenceOpClass::SafeScalar:
        pending.push_back(op);
        break;
      case VPTOInferenceOpClass::Boundary: {
        FailureOr<bool> changed = flush();
        if (failed(changed)) {
          return failure();
        }
        changedInIteration |= *changed;
        break;
      }
      }
      if (changedInIteration) {
        break;
      }
    }

    if (changedInIteration) {
      continue;
    }

    FailureOr<bool> changed = flush();
    if (failed(changed)) {
      return failure();
    }
    if (!*changed) {
      return success();
    }
  }

  return failure();
}

static LogicalResult inferVecScopesInBlock(Block &block, MLIRContext *context) {
  if (failed(repairEscapingSubclusters(block, context))) {
    return failure();
  }

  SmallVector<Operation *, mlir::pto::kValue16> pending;

  auto flush = [&]() -> LogicalResult {
    if (failed(wrapGreedySubclusters(pending, context))) {
      return failure();
    }
    pending.clear();
    return success();
  };

  SmallVector<Operation *, mlir::pto::kValue32> ops;
  for (Operation &op : block) {
    ops.push_back(&op);
  }

  for (Operation *op : ops) {
    switch (classifyOperationForInference(op)) {
    case VPTOInferenceOpClass::Vector:
    case VPTOInferenceOpClass::SafeScalar:
      pending.push_back(op);
      continue;
    case VPTOInferenceOpClass::Boundary:
      if (failed(flush())) {
        return failure();
      }
      continue;
    }
  }
  if (failed(flush())) {
    return failure();
  }

  SmallVector<Operation *, mlir::pto::kValue32> remainingOps;
  for (Operation &op : block) {
    remainingOps.push_back(&op);
  }

  for (Operation *op : remainingOps) {
    if (isExplicitVectorScopeCarrier(op)) {
      continue;
    }
    for (Region &nested : op->getRegions()) {
      if (failed(inferVecScopesInRegion(nested, context))) {
        return failure();
      }
    }
  }
  return success();
}

static LogicalResult inferVecScopesInRegion(Region &region,
                                            MLIRContext *context) {
  for (Block &block : region) {
    if (failed(inferVecScopesInBlock(block, context))) {
      return failure();
    }
  }
  return success();
}

struct PTOInferVPTOVecScopePass
    : public pto::impl::PTOInferVPTOVecScopeBase<
          PTOInferVPTOVecScopePass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (failed(inferVecScopesInRegion(func.getBody(), &getContext()))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOInferVPTOVecScopePass() {
  return std::make_unique<PTOInferVPTOVecScopePass>();
}

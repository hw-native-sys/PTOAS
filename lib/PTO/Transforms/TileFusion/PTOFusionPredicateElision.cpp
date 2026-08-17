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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOFUSIONPREDICATEELISION
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

struct PltCandidate {
  Operation *op = nullptr;
  Value scalar;
  Value mask;
  Value scalarOut;
  unsigned bitWidth = 0;
  SmallVector<unsigned, mlir::pto::kValue4> dominatingCandidates;
};

struct FusionRegionPredicateContext {
  pto::FusionRegionOp fusionRegion;
  SmallVector<PltCandidate, mlir::pto::kValue8> pltCandidates;
};

enum class EquivalenceState : uint8_t {
  InProgress,
  Equivalent,
  NotEquivalent,
};

struct ValueEquivalenceContext {
  // Cache pairwise equivalence inside one fusion-region rewrite walk. This
  // keeps recursive loop-carried checks bounded and deterministic.
  llvm::DenseMap<Value, llvm::SmallDenseMap<Value, EquivalenceState, mlir::pto::kValue4>>
      states;
};

struct ForIterArgInfo {
  scf::ForOp forOp;
  unsigned iterArgIndex = 0;
};

struct PltScalarOutInfo {
  Value scalar;
  unsigned bitWidth = 0;
};

static bool areEquivalentValues(Value lhs, Value rhs,
                                ValueEquivalenceContext &context);

static void normalizeValuePair(Value &lhs, Value &rhs) {
  if (lhs.getAsOpaquePointer() > rhs.getAsOpaquePointer()) {
    Value tmp = lhs;
    lhs = rhs;
    rhs = tmp;
  }
}

static bool areSameValuePair(Value lhs, Value rhs, Value expectedLhs,
                             Value expectedRhs) {
  normalizeValuePair(lhs, rhs);
  normalizeValuePair(expectedLhs, expectedRhs);
  return lhs == expectedLhs && rhs == expectedRhs;
}

static std::optional<EquivalenceState>
lookupEquivalenceState(ValueEquivalenceContext &context, Value lhs, Value rhs) {
  normalizeValuePair(lhs, rhs);
  auto outerIt = context.states.find(lhs);
  if (outerIt == context.states.end()) {
    return std::nullopt;
  }
  auto innerIt = outerIt->second.find(rhs);
  if (innerIt == outerIt->second.end()) {
    return std::nullopt;
  }
  return innerIt->second;
}

static void setEquivalenceState(ValueEquivalenceContext &context, Value lhs,
                                Value rhs, EquivalenceState state) {
  normalizeValuePair(lhs, rhs);
  context.states[lhs][rhs] = state;
}

static std::optional<PltCandidate> buildPltCandidate(Operation *op) {
  if (auto plt = dyn_cast<pto::PltB8Op>(op)) {
    return PltCandidate{
        op, plt.getScalar(), plt.getMask(), plt.getScalarOut(), 8, {}};
  }
  if (auto plt = dyn_cast<pto::PltB16Op>(op)) {
    return PltCandidate{
        op, plt.getScalar(), plt.getMask(), plt.getScalarOut(), 16, {}};
  }
  if (auto plt = dyn_cast<pto::PltB32Op>(op)) {
    return PltCandidate{
        op, plt.getScalar(), plt.getMask(), plt.getScalarOut(), 32, {}};
  }
  return std::nullopt;
}

static std::optional<ForIterArgInfo> getForIterArgInfo(Value value) {
  auto arg = dyn_cast<BlockArgument>(value);
  if (!arg || arg.getArgNumber() == 0) {
    return std::nullopt;
  }

  auto forOp =
      dyn_cast_or_null<scf::ForOp>(arg.getParentRegion()->getParentOp());
  if (!forOp || arg.getOwner() != forOp.getBody()) {
    return std::nullopt;
  }

  unsigned iterArgIndex = arg.getArgNumber() - 1;
  if (iterArgIndex >= forOp.getInitArgs().size()) {
    return std::nullopt;
  }
  return ForIterArgInfo{forOp, iterArgIndex};
}

static std::optional<PltScalarOutInfo> getPltScalarOutInfo(Value value) {
  // Element-wise templates keep the remaining element count as index while
  // plt consumes and returns an integer scalar. Look through the cast used to
  // feed the scalar result back to scf.for.
  if (auto indexCast = value.getDefiningOp<arith::IndexCastOp>()) {
    value = indexCast.getIn();
  }

  auto result = dyn_cast<OpResult>(value);
  if (!result || result.getResultNumber() != 1) {
    return std::nullopt;
  }

  if (auto plt = dyn_cast<pto::PltB8Op>(result.getOwner())) {
    return PltScalarOutInfo{plt.getScalar(), 8};
  }
  if (auto plt = dyn_cast<pto::PltB16Op>(result.getOwner())) {
    return PltScalarOutInfo{plt.getScalar(), 16};
  }
  if (auto plt = dyn_cast<pto::PltB32Op>(result.getOwner())) {
    return PltScalarOutInfo{plt.getScalar(), 32};
  }
  return std::nullopt;
}

static bool areEquivalentLoopCarriedValues(Value lhs, Value rhs,
                                           ValueEquivalenceContext &context) {
  std::optional<ForIterArgInfo> lhsInfo = getForIterArgInfo(lhs);
  std::optional<ForIterArgInfo> rhsInfo = getForIterArgInfo(rhs);
  if (!lhsInfo || !rhsInfo) {
    return false;
  }
  if (lhsInfo->forOp != rhsInfo->forOp) {
    return false;
  }

  if (lhsInfo->forOp.getRegionIterArgs().size() !=
      lhsInfo->forOp.getInitArgs().size()) {
    return false;
  }
  if (lhsInfo->forOp.getRegionIterArgs().size() !=
      lhsInfo->forOp.getYieldedValues().size()) {
    return false;
  }

  ValueRange initArgs = lhsInfo->forOp.getInitArgs();
  if (!areEquivalentValues(initArgs[lhsInfo->iterArgIndex],
                           initArgs[rhsInfo->iterArgIndex], context)) {
    return false;
  }

  ValueRange yieldedValues = lhsInfo->forOp.getYieldedValues();
  std::optional<PltScalarOutInfo> lhsYieldInfo =
      getPltScalarOutInfo(yieldedValues[lhsInfo->iterArgIndex]);
  std::optional<PltScalarOutInfo> rhsYieldInfo =
      getPltScalarOutInfo(yieldedValues[rhsInfo->iterArgIndex]);
  if (!lhsYieldInfo || !rhsYieldInfo) {
    return false;
  }
  if (lhsYieldInfo->bitWidth != rhsYieldInfo->bitWidth) {
    return false;
  }

  Value lhsRecurrenceInput = lhsYieldInfo->scalar;
  Value rhsRecurrenceInput = rhsYieldInfo->scalar;
  if (auto indexCast = lhsRecurrenceInput.getDefiningOp<arith::IndexCastOp>()) {
    lhsRecurrenceInput = indexCast.getIn();
  }
  if (auto indexCast = rhsRecurrenceInput.getDefiningOp<arith::IndexCastOp>()) {
    rhsRecurrenceInput = indexCast.getIn();
  }

  // Stay conservative on unsupported cyclic proofs. The only accepted
  // recurrence cycle is the direct iter_arg -> plt.scalar_out self recursion
  // for the same value pair, optionally bridged by the index casts required by
  // the plt/scf type boundary; more complex cycles remain unsupported.
  if (areSameValuePair(lhs, rhs, lhsRecurrenceInput, rhsRecurrenceInput)) {
    return true;
  }

  return areEquivalentValues(lhsYieldInfo->scalar, rhsYieldInfo->scalar,
                             context);
}

static bool areEquivalentOperations(Operation *lhs, Operation *rhs,
                                    ValueEquivalenceContext &context) {
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->getName() != rhs->getName()) {
    return false;
  }
  if (lhs->getNumRegions() != 0 || rhs->getNumRegions() != 0) {
    return false;
  }
  if (lhs->getNumResults() != rhs->getNumResults()) {
    return false;
  }
  if (lhs->getNumOperands() != rhs->getNumOperands()) {
    return false;
  }
  if (lhs->getAttrDictionary() != rhs->getAttrDictionary()) {
    return false;
  }
  if (!isMemoryEffectFree(lhs) || !isMemoryEffectFree(rhs)) {
    return false;
  }
  if (!llvm::equal(lhs->getResultTypes(), rhs->getResultTypes())) {
    return false;
  }

  for (auto [lhsOperand, rhsOperand] :
       llvm::zip(lhs->getOperands(), rhs->getOperands())) {
    if (!areEquivalentValues(lhsOperand, rhsOperand, context)) {
      return false;
    }
  }
  return true;
}

static bool areEquivalentValues(Value lhs, Value rhs,
                                ValueEquivalenceContext &context) {
  if (lhs == rhs) {
    return true;
  }
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs.getType() != rhs.getType()) {
    return false;
  }

  if (std::optional<EquivalenceState> state =
          lookupEquivalenceState(context, lhs, rhs)) {
    return *state == EquivalenceState::Equivalent;
  }
  setEquivalenceState(context, lhs, rhs, EquivalenceState::InProgress);

  auto lhsArg = dyn_cast<BlockArgument>(lhs);
  auto rhsArg = dyn_cast<BlockArgument>(rhs);
  if (lhsArg || rhsArg) {
    bool equivalent =
        lhsArg && rhsArg &&
        ((lhsArg.getOwner() == rhsArg.getOwner() &&
          lhsArg.getArgNumber() == rhsArg.getArgNumber()) ||
         areEquivalentLoopCarriedValues(lhs, rhs, context));
    setEquivalenceState(context, lhs, rhs,
                        equivalent ? EquivalenceState::Equivalent
                                   : EquivalenceState::NotEquivalent);
    return equivalent;
  }

  bool equivalent =
      areEquivalentOperations(lhs.getDefiningOp(), rhs.getDefiningOp(), context);
  setEquivalenceState(context, lhs, rhs,
                      equivalent ? EquivalenceState::Equivalent
                                 : EquivalenceState::NotEquivalent);
  return equivalent;
}

static void populateDominatingCandidateIndices(
    MutableArrayRef<PltCandidate> candidates, DominanceInfo &dominanceInfo) {
  for (unsigned current = 0; current < candidates.size(); ++current) {
    for (unsigned previous = 0; previous < current; ++previous) {
      if (candidates[previous].bitWidth != candidates[current].bitWidth) {
        continue;
      }
      // Only reuse an earlier plt when its whole result pair dominates the
      // later one. This keeps replacement local and SSA-safe.
      if (!dominanceInfo.properlyDominates(candidates[previous].op,
                                           candidates[current].op)) {
        continue;
      }
      candidates[current].dominatingCandidates.push_back(previous);
    }
  }
}

static FusionRegionPredicateContext
buildFusionRegionPredicateContext(pto::FusionRegionOp fusionRegion,
                                  DominanceInfo &dominanceInfo) {
  FusionRegionPredicateContext context;
  context.fusionRegion = fusionRegion;

  fusionRegion.walk([&](Operation *op) -> WalkResult {
    if (op != fusionRegion.getOperation() && isa<pto::FusionRegionOp>(op)) {
      return WalkResult::skip();
    }

    if (std::optional<PltCandidate> candidate = buildPltCandidate(op)) {
      context.pltCandidates.push_back(std::move(*candidate));
    }
    return WalkResult::advance();
  });

  populateDominatingCandidateIndices(context.pltCandidates, dominanceInfo);
  return context;
}

static Value getCurrentScalarOperand(const PltCandidate &candidate) {
  return candidate.op ? candidate.op->getOperand(0) : Value();
}

static std::optional<unsigned>
findEquivalentDominatingCandidate(FusionRegionPredicateContext &context,
                                  ValueEquivalenceContext &valueContext,
                                  unsigned currentIndex,
                                  const llvm::DenseSet<unsigned> &erased) {
  const PltCandidate &current = context.pltCandidates[currentIndex];
  Value currentScalar = getCurrentScalarOperand(current);
  for (unsigned previousIndex : current.dominatingCandidates) {
    if (erased.contains(previousIndex)) {
      continue;
    }
    const PltCandidate &previous = context.pltCandidates[previousIndex];
    // Equivalence is checked on the scalar input; when it holds, both plt
    // results are reused as a pair.
    if (areEquivalentValues(getCurrentScalarOperand(previous), currentScalar,
                            valueContext)) {
      return previousIndex;
    }
  }
  return std::nullopt;
}

static bool
elideEquivalentPltCandidates(FusionRegionPredicateContext &context) {
  bool changed = false;
  llvm::DenseSet<unsigned> erased;
  SmallVector<Operation *, mlir::pto::kValue8> opsToErase;
  ValueEquivalenceContext valueContext;

  for (unsigned currentIndex = 0; currentIndex < context.pltCandidates.size();
       ++currentIndex) {
    if (erased.contains(currentIndex)) {
      continue;
    }

    std::optional<unsigned> previousIndex =
        findEquivalentDominatingCandidate(context, valueContext, currentIndex,
                                          erased);
    if (!previousIndex) {
      continue;
    }

    PltCandidate &current = context.pltCandidates[currentIndex];
    PltCandidate &previous = context.pltCandidates[*previousIndex];
    current.mask.replaceAllUsesWith(previous.mask);
    current.scalarOut.replaceAllUsesWith(previous.scalarOut);
    opsToErase.push_back(current.op);
    erased.insert(currentIndex);
    changed = true;
  }

  for (Operation *op : opsToErase) {
    op->erase();
  }

  return changed;
}

struct PTOFusionPredicateElisionPass
    : public pto::impl::PTOFusionPredicateElisionBase<
          PTOFusionPredicateElisionPass> {
  using pto::impl::PTOFusionPredicateElisionBase<
      PTOFusionPredicateElisionPass>::PTOFusionPredicateElisionBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal()) {
      return;
    }

    DominanceInfo &dominanceInfo = getAnalysis<DominanceInfo>();
    SmallVector<FusionRegionPredicateContext, mlir::pto::kValue4> fusionContexts;
    func.walk([&](pto::FusionRegionOp fusionRegion) {
      FusionRegionPredicateContext context =
          buildFusionRegionPredicateContext(fusionRegion, dominanceInfo);
      if (!context.pltCandidates.empty()) {
        fusionContexts.push_back(std::move(context));
      }
    });

    bool changed = false;
    for (FusionRegionPredicateContext &context : fusionContexts) {
      changed |= elideEquivalentPltCandidates(context);
    }

    if (!changed) {
      markAllAnalysesPreserved();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOFusionPredicateElisionPass() {
  return std::make_unique<PTOFusionPredicateElisionPass>();
}

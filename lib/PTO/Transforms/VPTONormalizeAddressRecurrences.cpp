// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This pass only creates reversible address-recurrence witnesses for loops
// nested under pto.vecscope, matching the ownership boundary consumed by
// VPTOSoftPostUpdate. Loops outside that boundary must remain untouched.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/VPTOPostUpdateUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/MathExtras.h"
#include <cstdint>
#include <limits>
#include <optional>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VPTONORMALIZEADDRESSRECURRENCES
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

static constexpr unsigned kAddressWidth = 16;
using AddressDomain = pto::PostUpdateAddressDomain;

struct ProvenRecurrence {
  Value source;
  int64_t initial;
  int64_t increment;
  AddressDomain domain;
  Operation *updateOp = nullptr;
};

struct RewriteTarget {
  Operation *owner;
  unsigned operandNumber;
  Operation *sourceUseOwner;
  unsigned sourceUseOperandNumber;
  unsigned candidate;
};

struct RecurrencePlan {
  ProvenRecurrence recurrence;
  SmallVector<RewriteTarget> targets;
  bool rejected = false;
};

struct CandidatePlan {
  SmallVector<unsigned> recurrences;
  bool rejected = false;
};

static bool isAlreadyPostUpdate(Operation *op,
                                const pto::PostUpdateOpInfo &info) {
  return op->getNumResults() > info.minResultsForPost;
}

static std::optional<int64_t> getConstant(Value value, AddressDomain domain) {
  APInt bits;
  if (!matchPattern(value, m_ConstantInt(&bits)) || bits.getBitWidth() > 64)
    return std::nullopt;
  if (domain == AddressDomain::Signed)
    return bits.getSExtValue();
  uint64_t unsignedValue = bits.getZExtValue();
  if (unsignedValue >
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    return std::nullopt;
  return static_cast<int64_t>(unsignedValue);
}

static std::optional<int64_t> getSignedConstant(Value value) {
  return getConstant(value, AddressDomain::Signed);
}

static std::optional<uint64_t> getConstantTripCount(scf::ForOp forOp) {
  auto lower = getSignedConstant(forOp.getLowerBound());
  auto upper = getSignedConstant(forOp.getUpperBound());
  auto step = getSignedConstant(forOp.getStep());
  if (!lower || !upper || !step || *step <= 0)
    return std::nullopt;
  if (*lower >= *upper)
    return 0;
  __int128 distance = static_cast<__int128>(*upper) - *lower;
  __int128 count = (distance + *step - 1) / *step;
  if (count > std::numeric_limits<uint64_t>::max())
    return std::nullopt;
  return static_cast<uint64_t>(count);
}

static std::optional<unsigned> getSupportedWidth(Type type) {
  if (type.isIndex())
    return 64;
  auto integerType = dyn_cast<IntegerType>(type);
  if (!integerType ||
      (integerType.getWidth() != 16 && integerType.getWidth() != 32))
    return std::nullopt;
  return integerType.getWidth();
}

static bool fitsDomain(__int128 value, unsigned width, AddressDomain domain) {
  __int128 lower = domain == AddressDomain::Signed
                       ? -(static_cast<__int128>(1) << (width - 1))
                       : 0;
  __int128 upper = domain == AddressDomain::Signed
                       ? (static_cast<__int128>(1) << (width - 1)) - 1
                       : (static_cast<__int128>(1) << width) - 1;
  return value >= lower && value <= upper;
}

// Prove the complete mathematical recurrence, including the update executed
// after the last body iteration. Endpoint checks suffice because the step is
// fixed and therefore the sequence is monotonic.
static bool recurrenceFits(Value source, int64_t initial, int64_t increment,
                           uint64_t tripCount, AddressDomain domain) {
  auto sourceWidth = getSupportedWidth(source.getType());
  bool incrementFits =
      domain == AddressDomain::Signed
          ? fitsDomain(increment, kAddressWidth, AddressDomain::Signed)
          : increment >=
                    -static_cast<int64_t>((uint64_t{1} << kAddressWidth) - 1) &&
                increment <=
                    static_cast<int64_t>((uint64_t{1} << kAddressWidth) - 1);
  if (!sourceWidth || !incrementFits)
    return false;
  __int128 final = static_cast<__int128>(initial) +
                   static_cast<__int128>(increment) * tripCount;
  __int128 minimum = std::min(static_cast<__int128>(initial), final);
  __int128 maximum = std::max(static_cast<__int128>(initial), final);
  return fitsDomain(minimum, *sourceWidth, domain) &&
         fitsDomain(maximum, *sourceWidth, domain) &&
         fitsDomain(minimum, kAddressWidth, domain) &&
         fitsDomain(maximum, kAddressWidth, domain);
}

static std::optional<ProvenRecurrence>
matchProvenRecurrence(Value value, scf::ForOp forOp, uint64_t tripCount,
                      AddressDomain domain) {
  if (!getSupportedWidth(value.getType()))
    return std::nullopt;

  int64_t initial;
  int64_t increment;
  Operation *updateOp = nullptr;
  if (value == forOp.getInductionVar()) {
    auto lower = getConstant(forOp.getLowerBound(), domain);
    auto step = getSignedConstant(forOp.getStep());
    if (!lower || !step)
      return std::nullopt;
    initial = *lower;
    increment = *step;
  } else {
    auto iterArg = dyn_cast<BlockArgument>(value);
    if (!iterArg || iterArg.getOwner() != forOp.getBody() ||
        iterArg.getArgNumber() == 0)
      return std::nullopt;
    unsigned index = iterArg.getArgNumber() - 1;
    auto init = getConstant(forOp.getInitArgs()[index], domain);
    if (!init || !forOp.getResult(index).use_empty())
      return std::nullopt;
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    Value yielded = yieldOp.getOperand(index);
    if (auto add = yielded.getDefiningOp<arith::AddIOp>()) {
      Value step;
      if (add.getLhs() == value)
        step = add.getRhs();
      else if (add.getRhs() == value)
        step = add.getLhs();
      auto constant = step ? getConstant(step, domain) : std::nullopt;
      if (!constant)
        return std::nullopt;
      increment = *constant;
      updateOp = add;
    } else if (auto sub = yielded.getDefiningOp<arith::SubIOp>()) {
      if (sub.getLhs() != value)
        return std::nullopt;
      auto constant = getConstant(sub.getRhs(), domain);
      if (!constant)
        return std::nullopt;
      __int128 negated = -static_cast<__int128>(*constant);
      if (negated < std::numeric_limits<int64_t>::min() ||
          negated > std::numeric_limits<int64_t>::max())
        return std::nullopt;
      increment = static_cast<int64_t>(negated);
      updateOp = sub;
    } else {
      return std::nullopt;
    }
    if (!yielded.hasOneUse() ||
        *yielded.getUsers().begin() != yieldOp.getOperation())
      return std::nullopt;
    initial = *init;
  }

  if (!recurrenceFits(value, initial, increment, tripCount, domain))
    return std::nullopt;
  return ProvenRecurrence{value, initial, increment, domain, updateOp};
}

class LoopPlanner {
public:
  LoopPlanner(scf::ForOp forOp, uint64_t tripCount)
      : forOp(forOp), tripCount(tripCount) {}

  void collect() {
    for (Operation &op : forOp.getBody()->without_terminator()) {
      const auto *info = pto::getPostUpdateOpInfo(&op);
      if (!info || isAlreadyPostUpdate(&op, *info))
        continue;
      analyzeCandidate(&op, *info);
    }
    rejectRecurrencesWithOtherUsers();
  }

  bool empty() const {
    return llvm::none_of(recurrences, [&](const RecurrencePlan &plan) {
      return !plan.rejected &&
             llvm::any_of(plan.targets, [&](const RewriteTarget &target) {
               return !candidates[target.candidate].rejected;
             });
    });
  }

  void rewrite(OpBuilder &builder) {
    SmallVector<unsigned> active;
    for (auto [index, plan] : llvm::enumerate(recurrences)) {
      if (plan.rejected)
        continue;
      if (llvm::any_of(plan.targets, [&](const RewriteTarget &target) {
            return !candidates[target.candidate].rejected;
          }))
        active.push_back(index);
    }
    if (active.empty())
      return;

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(forOp);
    SmallVector<Value> initArgs(forOp.getInitArgs().begin(),
                                forOp.getInitArgs().end());
    for (unsigned index : active)
      initArgs.push_back(builder.create<arith::ConstantIntOp>(
          forOp.getLoc(), recurrences[index].recurrence.initial,
          kAddressWidth));

    auto newFor = builder.create<scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), initArgs);
    newFor->setAttrs(forOp->getAttrs());

    IRMapping mapping;
    mapping.map(forOp.getInductionVar(), newFor.getInductionVar());
    for (auto [oldArg, newArg] : llvm::zip(
             forOp.getRegionIterArgs(), newFor.getRegionIterArgs().take_front(
                                            forOp.getNumRegionIterArgs())))
      mapping.map(oldArg, newArg);

    DenseMap<unsigned, Value> normalizedI16Values;
    DenseMap<std::pair<unsigned, Type>, Value> normalizedValues;
    unsigned originalArgCount = forOp.getNumRegionIterArgs();
    builder.setInsertionPointToStart(newFor.getBody());
    for (auto [activeIndex, recurrenceIndex] : llvm::enumerate(active)) {
      Value i16Value =
          newFor.getRegionIterArgs()[originalArgCount + activeIndex];
      normalizedI16Values[recurrenceIndex] = i16Value;
      normalizedValues[{recurrenceIndex, builder.getI16Type()}] = i16Value;
    }

    for (unsigned recurrenceIndex : active) {
      AddressDomain domain = recurrences[recurrenceIndex].recurrence.domain;
      for (const RewriteTarget &target : recurrences[recurrenceIndex].targets) {
        if (candidates[target.candidate].rejected)
          continue;
        Type wantedType =
            target.owner->getOperand(target.operandNumber).getType();
        auto key = std::make_pair(recurrenceIndex, wantedType);
        if (normalizedValues.contains(key))
          continue;
        Value i16Value = normalizedI16Values[recurrenceIndex];
        Value restored;
        if (wantedType.isIndex()) {
          restored = domain == AddressDomain::Signed
                         ? Value(builder.create<arith::IndexCastOp>(
                               forOp.getLoc(), wantedType, i16Value))
                         : Value(builder.create<arith::IndexCastUIOp>(
                               forOp.getLoc(), wantedType, i16Value));
        } else if (wantedType.isInteger(32)) {
          restored = domain == AddressDomain::Signed
                         ? Value(builder.create<arith::ExtSIOp>(
                               forOp.getLoc(), wantedType, i16Value))
                         : Value(builder.create<arith::ExtUIOp>(
                               forOp.getLoc(), wantedType, i16Value));
        } else {
          llvm_unreachable("unsupported canonical address target type");
        }
        normalizedValues[key] = restored;
      }
    }

    DenseMap<Operation *, Operation *> clonedOps;
    for (Operation &oldOp : forOp.getBody()->without_terminator()) {
      Operation *cloned = builder.clone(oldOp, mapping);
      clonedOps[&oldOp] = cloned;
    }

    for (unsigned recurrenceIndex : active) {
      RecurrencePlan &plan = recurrences[recurrenceIndex];
      for (const RewriteTarget &target : plan.targets) {
        if (candidates[target.candidate].rejected)
          continue;
        auto cloned = clonedOps.find(target.owner);
        if (cloned == clonedOps.end())
          continue;
        Type wantedType =
            target.owner->getOperand(target.operandNumber).getType();
        Operation *clonedOwner = cloned->second;
        Value original = clonedOwner->getOperand(target.operandNumber);
        Value canonical = normalizedValues[{recurrenceIndex, wantedType}];
        builder.setInsertionPoint(clonedOwner);
        Value witness = builder.create<pto::AddressRecurrenceWitnessOp>(
            clonedOwner->getLoc(), wantedType, original, canonical);
        clonedOwner->setOperand(target.operandNumber, witness);
      }
    }

    auto oldYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    SmallVector<Value> yields;
    for (Value value : oldYield.getOperands())
      yields.push_back(mapping.lookupOrDefault(value));
    builder.setInsertionPointToEnd(newFor.getBody());
    for (auto [activeIndex, recurrenceIndex] : llvm::enumerate(active)) {
      Value current =
          newFor.getRegionIterArgs()[originalArgCount + activeIndex];
      const ProvenRecurrence &recurrence =
          recurrences[recurrenceIndex].recurrence;
      if (recurrence.domain == AddressDomain::Unsigned &&
          recurrence.increment < 0) {
        Value decrement = builder.create<arith::ConstantIntOp>(
            forOp.getLoc(), -recurrence.increment, kAddressWidth);
        yields.push_back(
            builder.create<arith::SubIOp>(forOp.getLoc(), current, decrement,
                                          arith::IntegerOverflowFlags::nuw));
      } else {
        Value increment = builder.create<arith::ConstantIntOp>(
            forOp.getLoc(), recurrence.increment, kAddressWidth);
        arith::IntegerOverflowFlags flags =
            recurrence.domain == AddressDomain::Signed
                ? arith::IntegerOverflowFlags::nsw
                : arith::IntegerOverflowFlags::nuw;
        yields.push_back(builder.create<arith::AddIOp>(forOp.getLoc(), current,
                                                       increment, flags));
      }
    }
    builder.create<scf::YieldOp>(oldYield.getLoc(), yields);

    for (auto [oldResult, newResult] :
         llvm::zip(forOp.getResults(),
                   newFor.getResults().take_front(forOp.getNumResults())))
      oldResult.replaceAllUsesWith(newResult);
    forOp.erase();
  }

private:
  std::optional<unsigned>
  requestRecurrence(Value value, Operation *owner, unsigned operandNumber,
                    Operation *sourceUseOwner, unsigned sourceUseOperandNumber,
                    unsigned candidate, AddressDomain domain) {
    auto proven = matchProvenRecurrence(value, forOp, tripCount, domain);
    if (!proven)
      return std::nullopt;

    auto [it, inserted] = recurrenceIndices.try_emplace(
        std::make_pair(value, domain), recurrences.size());
    if (inserted)
      recurrences.push_back(RecurrencePlan{*proven, {}});
    unsigned index = it->second;
    auto &targets = recurrences[index].targets;
    if (llvm::none_of(targets, [&](const RewriteTarget &target) {
          return target.owner == owner &&
                 target.operandNumber == operandNumber &&
                 target.candidate == candidate;
        }))
      targets.push_back({owner, operandNumber, sourceUseOwner,
                         sourceUseOperandNumber, candidate});
    return index;
  }

  bool analyzeIntegerValue(Value value, Operation *owner,
                           unsigned operandNumber, unsigned candidate,
                           AddressDomain domain) {
    if (forOp.isDefinedOutsideOfLoop(value))
      return true;
    Value recurrenceValue = value;
    Operation *rewriteOwner = owner;
    unsigned rewriteOperand = operandNumber;
    Operation *sourceUseOwner = owner;
    unsigned sourceUseOperand = operandNumber;
    // Pointer advancement is index-typed even when the stateful op exposes an
    // i32 stride. Look through a signed, value-preserving cast so both uses
    // share one i16 shadow recurrence.
    if (auto cast = value.getDefiningOp<arith::IndexCastOp>()) {
      recurrenceValue = cast.getIn();
      rewriteOwner = cast;
      rewriteOperand = 0;
      sourceUseOwner = cast;
      sourceUseOperand = 0;
    }
    if (auto cast = value.getDefiningOp<arith::IndexCastUIOp>()) {
      if (domain != AddressDomain::Unsigned ||
          !cast.getIn().getType().isIndex() ||
          !cast.getType().isInteger(kAddressWidth))
        return false;
      recurrenceValue = cast.getIn();
      sourceUseOwner = cast;
      sourceUseOperand = 0;
    }
    auto recurrence =
        matchProvenRecurrence(recurrenceValue, forOp, tripCount, domain);
    if (!recurrence)
      return false;
    auto plan =
        requestRecurrence(recurrenceValue, rewriteOwner, rewriteOperand,
                          sourceUseOwner, sourceUseOperand, candidate, domain);
    if (!plan)
      return false;
    candidates[candidate].recurrences.push_back(*plan);
    return true;
  }

  bool analyzeBase(Value base, unsigned candidate) {
    if (forOp.isDefinedOutsideOfLoop(base))
      return true;

    if (auto addPtr = base.getDefiningOp<pto::AddPtrOp>()) {
      if (!forOp.isDefinedOutsideOfLoop(addPtr.getPtr()))
        return false;
      return analyzeIntegerValue(addPtr.getOffset(), addPtr, 1, candidate,
                                 AddressDomain::Signed);
    }

    auto iterArg = dyn_cast<BlockArgument>(base);
    if (!iterArg || iterArg.getOwner() != forOp.getBody() ||
        iterArg.getArgNumber() == 0)
      return false;
    unsigned index = iterArg.getArgNumber() - 1;
    if (!forOp.getResult(index).use_empty())
      return false;
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    auto addPtr = yieldOp.getOperand(index).getDefiningOp<pto::AddPtrOp>();
    if (!addPtr || addPtr.getPtr() != base || !addPtr->hasOneUse())
      return false;
    return analyzeIntegerValue(addPtr.getOffset(), addPtr, 1, candidate,
                               AddressDomain::Signed);
  }

  void analyzeCandidate(Operation *op, const pto::PostUpdateOpInfo &info) {
    unsigned candidate = candidates.size();
    candidates.emplace_back();

    Value base = op->getOperand(info.baseOperandIdx);
    if (!analyzeBase(base, candidate)) {
      candidates[candidate].rejected = true;
      return;
    }

    if (info.strideOperandIdx) {
      Value stride = op->getOperand(*info.strideOperandIdx);
      if (!analyzeIntegerValue(stride, op, *info.strideOperandIdx, candidate,
                               info.strideDomain)) {
        candidates[candidate].rejected = true;
        return;
      }
    }
  }

  void rejectRecurrencesWithOtherUsers() {
    DenseMap<Value, DenseSet<OpOperand *>> plannedUsesBySource;
    for (RecurrencePlan &plan : recurrences)
      for (RewriteTarget &target : plan.targets)
        plannedUsesBySource[plan.recurrence.source].insert(
            &target.sourceUseOwner->getOpOperand(
                target.sourceUseOperandNumber));

    for (RecurrencePlan &plan : recurrences) {
      const DenseSet<OpOperand *> &plannedUses =
          plannedUsesBySource[plan.recurrence.source];
      for (OpOperand &use : plan.recurrence.source.getUses()) {
        if (plannedUses.contains(&use) ||
            use.getOwner() == plan.recurrence.updateOp)
          continue;
        plan.rejected = true;
        break;
      }
    }

    bool changed;
    do {
      changed = false;
      for (auto [candidateIndex, candidate] : llvm::enumerate(candidates)) {
        if (candidate.rejected)
          continue;
        if (llvm::any_of(candidate.recurrences, [&](unsigned recurrence) {
              return recurrences[recurrence].rejected;
            })) {
          candidate.rejected = true;
          changed = true;
          for (unsigned recurrence : candidate.recurrences)
            recurrences[recurrence].rejected = true;
        }
      }
    } while (changed);
  }

  scf::ForOp forOp;
  uint64_t tripCount;
  SmallVector<CandidatePlan> candidates;
  SmallVector<RecurrencePlan> recurrences;
  DenseMap<std::pair<Value, AddressDomain>, unsigned> recurrenceIndices;
};

struct VPTONormalizeAddressRecurrencesPass
    : public pto::impl::VPTONormalizeAddressRecurrencesBase<
          VPTONormalizeAddressRecurrencesPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<scf::ForOp> loops;
    module.walk([&](scf::ForOp forOp) {
      if (forOp->getParentOfType<pto::VecScopeOp>())
        loops.push_back(forOp);
    });

    OpBuilder builder(&getContext());
    for (scf::ForOp forOp : loops) {
      auto tripCount = getConstantTripCount(forOp);
      if (!tripCount)
        continue;
      LoopPlanner planner(forOp, *tripCount);
      planner.collect();
      if (!planner.empty())
        planner.rewrite(builder);
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVPTONormalizeAddressRecurrencesPass() {
  return std::make_unique<VPTONormalizeAddressRecurrencesPass>();
}

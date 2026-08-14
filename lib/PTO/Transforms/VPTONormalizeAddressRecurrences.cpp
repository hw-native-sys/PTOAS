// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This independent canonicalization pass permanently narrows proven VPTO
// address recurrences to i16 without changing IV versus iter_arg structure. It
// does not depend on soft post-update accepting or consuming the result.

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
};

struct RecurrencePlan {
  ProvenRecurrence recurrence;
  SmallVector<RewriteTarget> targets;
  bool rejected = false;
};

struct SourceRewrite {
  Value source;
  int64_t initial;
  int64_t increment;
  SmallVector<unsigned> planIndices;
  std::optional<unsigned> iterArgIndex;
  bool needsSignedNoWrap = false;
  bool needsUnsignedNoWrap = false;
  bool compatible = true;
};

static std::optional<int64_t> getConstant(Value value, AddressDomain domain) {
  APInt bits;
  if (!matchPattern(value, m_ConstantInt(&bits)) || bits.getBitWidth() > 64)
    return std::nullopt;
  if (domain == AddressDomain::Signed)
    return bits.getSExtValue();
  uint64_t unsignedValue = bits.getZExtValue();
  if (unsignedValue >
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return std::nullopt;
  }
  return static_cast<int64_t>(unsignedValue);
}

static std::optional<int64_t> getSignedConstant(Value value) {
  return getConstant(value, AddressDomain::Signed);
}

static Value restoreAddressType(OpBuilder &builder, Location loc, Value value,
                                Type type, AddressDomain domain) {
  if (type.isInteger(kAddressWidth)) {
    return value;
  }
  if (type.isIndex()) {
    return domain == AddressDomain::Signed
               ? Value(builder.create<arith::IndexCastOp>(loc, type, value))
               : Value(builder.create<arith::IndexCastUIOp>(loc, type, value));
  }
  if (type.isInteger(32)) {
    return domain == AddressDomain::Signed
               ? Value(builder.create<arith::ExtSIOp>(loc, type, value))
               : Value(builder.create<arith::ExtUIOp>(loc, type, value));
  }
  llvm_unreachable("unsupported normalized address type");
}

static std::optional<uint64_t> getConstantTripCount(scf::ForOp forOp) {
  auto lower = getSignedConstant(forOp.getLowerBound());
  auto upper = getSignedConstant(forOp.getUpperBound());
  auto step = getSignedConstant(forOp.getStep());
  if (!lower || !upper || !step || *step <= 0) {
    return std::nullopt;
  }
  if (*lower >= *upper) {
    return 0;
  }
  __int128 distance = static_cast<__int128>(*upper) - *lower;
  __int128 count = (distance + *step - 1) / *step;
  if (count > std::numeric_limits<uint64_t>::max()) {
    return std::nullopt;
  }
  return static_cast<uint64_t>(count);
}

static std::optional<unsigned> getSupportedWidth(Type type) {
  if (type.isIndex()) {
    return 64;
  }
  auto integerType = dyn_cast<IntegerType>(type);
  if (!integerType ||
      (integerType.getWidth() != 16 && integerType.getWidth() != 32)) {
    return std::nullopt;
  }
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
  if (!sourceWidth || !incrementFits) {
    return false;
  }
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
  if (!getSupportedWidth(value.getType())) {
    return std::nullopt;
  }

  int64_t initial;
  int64_t increment;
  Operation *updateOp = nullptr;
  if (value == forOp.getInductionVar()) {
    auto lower = getConstant(forOp.getLowerBound(), domain);
    auto step = getSignedConstant(forOp.getStep());
    if (!lower || !step) {
      return std::nullopt;
    }
    initial = *lower;
    increment = *step;
  } else {
    auto iterArg = dyn_cast<BlockArgument>(value);
    if (!iterArg || iterArg.getOwner() != forOp.getBody() ||
        iterArg.getArgNumber() == 0) {
      return std::nullopt;
    }
    unsigned index = iterArg.getArgNumber() - 1;
    auto init = getConstant(forOp.getInitArgs()[index], domain);
    if (!init) {
      return std::nullopt;
    }
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    Value yielded = yieldOp.getOperand(index);
    if (auto add = yielded.getDefiningOp<arith::AddIOp>()) {
      Value step;
      if (add.getLhs() == value) {
        step = add.getRhs();
      } else if (add.getRhs() == value) {
        step = add.getLhs();
      }
      auto constant = step ? getConstant(step, domain) : std::nullopt;
      if (!constant) {
        return std::nullopt;
      }
      increment = *constant;
      updateOp = add;
    } else if (auto sub = yielded.getDefiningOp<arith::SubIOp>()) {
      if (sub.getLhs() != value) {
        return std::nullopt;
      }
      auto constant = getConstant(sub.getRhs(), domain);
      if (!constant) {
        return std::nullopt;
      }
      __int128 negated = -static_cast<__int128>(*constant);
      if (negated < std::numeric_limits<int64_t>::min() ||
          negated > std::numeric_limits<int64_t>::max()) {
        return std::nullopt;
      }
      increment = static_cast<int64_t>(negated);
      updateOp = sub;
    } else {
      return std::nullopt;
    }
    if (!yielded.hasOneUse() ||
        *yielded.getUsers().begin() != yieldOp.getOperation()) {
      return std::nullopt;
    }
    initial = *init;
  }

  if (!recurrenceFits(value, initial, increment, tripCount, domain)) {
    return std::nullopt;
  }
  return ProvenRecurrence{value, initial, increment, domain, updateOp};
}

class LoopPlanner {
public:
  LoopPlanner(scf::ForOp forOp, uint64_t tripCount)
      : forOp(forOp), tripCount(tripCount) {}

  void collect() {
    for (Operation &op : forOp.getBody()->without_terminator()) {
      const auto *info = pto::getPostUpdateOpInfo(&op);
      if (!info) {
        continue;
      }
      analyzeCandidate(&op, *info);
    }
  }

  bool empty() const {
    return llvm::none_of(recurrences, [&](const RecurrencePlan &plan) {
      return !plan.rejected && !plan.targets.empty();
    });
  }

  void rewrite(OpBuilder &builder) {
    DenseMap<Value, unsigned> sourceIndices;
    SmallVector<SourceRewrite> sources;
    for (auto [planIndex, plan] : llvm::enumerate(recurrences)) {
      if (plan.rejected || plan.targets.empty()) {
        continue;
      }
      auto [it, inserted] =
          sourceIndices.try_emplace(plan.recurrence.source, sources.size());
      if (inserted) {
        auto iterArg = dyn_cast<BlockArgument>(plan.recurrence.source);
        std::optional<unsigned> iterArgIndex;
        if (iterArg && iterArg.getArgNumber() > 0) {
          iterArgIndex = iterArg.getArgNumber() - 1;
        }
        sources.push_back({plan.recurrence.source,
                           plan.recurrence.initial,
                           plan.recurrence.increment,
                           {},
                           iterArgIndex});
      }
      SourceRewrite &source = sources[it->second];
      if (source.initial != plan.recurrence.initial ||
          source.increment != plan.recurrence.increment) {
        source.compatible = false;
        continue;
      }
      source.planIndices.push_back(planIndex);
      source.needsSignedNoWrap |=
          plan.recurrence.domain == AddressDomain::Signed;
      source.needsUnsignedNoWrap |=
          plan.recurrence.domain == AddressDomain::Unsigned;
    }

    bool narrowInductionVariable = false;
    SmallVector<unsigned> activeSources;
    for (auto [sourceIndex, source] : llvm::enumerate(sources)) {
      if (!source.compatible || source.planIndices.empty()) {
        continue;
      }
      if (source.source == forOp.getInductionVar()) {
        if (!pto::canNarrowLoopCounterToI16(forOp)) {
          continue;
        }
        narrowInductionVariable = true;
      }
      activeSources.push_back(sourceIndex);
    }
    if (activeSources.empty()) {
      return;
    }

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(forOp);
    Value lower = forOp.getLowerBound();
    Value upper = forOp.getUpperBound();
    Value step = forOp.getStep();
    if (narrowInductionVariable) {
      lower = builder.create<arith::ConstantIntOp>(
          forOp.getLoc(), *getSignedConstant(lower), kAddressWidth);
      upper = builder.create<arith::ConstantIntOp>(
          forOp.getLoc(), *getSignedConstant(upper), kAddressWidth);
      step = builder.create<arith::ConstantIntOp>(
          forOp.getLoc(), *getSignedConstant(step), kAddressWidth);
    }

    SmallVector<Value> initArgs(forOp.getInitArgs().begin(),
                                forOp.getInitArgs().end());
    for (unsigned sourceIndex : activeSources) {
      SourceRewrite &source = sources[sourceIndex];
      if (!source.iterArgIndex) {
        continue;
      }
      initArgs[*source.iterArgIndex] = builder.create<arith::ConstantIntOp>(
          forOp.getLoc(), source.initial, kAddressWidth);
    }

    auto newFor = builder.create<scf::ForOp>(forOp.getLoc(), lower, upper,
                                             step, initArgs);
    newFor->setAttrs(forOp->getAttrs());
    if (!newFor.getBody()->empty()) {
      newFor.getBody()->getTerminator()->erase();
    }

    IRMapping mapping;
    DenseMap<unsigned, Value> normalizedI16ValuesBySource;
    DenseMap<std::pair<unsigned, Type>, Value> normalizedValues;
    builder.setInsertionPointToStart(newFor.getBody());

    if (narrowInductionVariable) {
      unsigned sourceIndex = sourceIndices.lookup(forOp.getInductionVar());
      normalizedI16ValuesBySource[sourceIndex] = newFor.getInductionVar();
    } else {
      mapping.map(forOp.getInductionVar(), newFor.getInductionVar());
    }
    for (auto [argIndex, oldArg] :
         llvm::enumerate(forOp.getRegionIterArgs())) {
      auto sourceIt = sourceIndices.find(oldArg);
      bool isActive =
          sourceIt != sourceIndices.end() &&
          llvm::is_contained(activeSources, sourceIt->second);
      if (isActive) {
        normalizedI16ValuesBySource[sourceIt->second] =
            newFor.getRegionIterArgs()[argIndex];
      } else {
        mapping.map(oldArg, newFor.getRegionIterArgs()[argIndex]);
      }
    }

    for (unsigned sourceIndex : activeSources) {
      SourceRewrite &source = sources[sourceIndex];
      for (unsigned planIndex : source.planIndices) {
        AddressDomain domain = recurrences[planIndex].recurrence.domain;
        for (const RewriteTarget &target : recurrences[planIndex].targets) {
          Type wantedType =
              target.owner->getOperand(target.operandNumber).getType();
          auto key = std::make_pair(planIndex, wantedType);
          if (normalizedValues.contains(key)) {
            continue;
          }
          Value i16Value = normalizedI16ValuesBySource[sourceIndex];
          normalizedValues[key] = restoreAddressType(
              builder, forOp.getLoc(), i16Value, wantedType, domain);
        }
      }

      unsigned defaultPlan = source.planIndices.front();
      Type sourceType = source.source.getType();
      auto defaultKey = std::make_pair(defaultPlan, sourceType);
      if (!normalizedValues.contains(defaultKey)) {
        AddressDomain domain = recurrences[defaultPlan].recurrence.domain;
        Value i16Value = normalizedI16ValuesBySource[sourceIndex];
        normalizedValues[defaultKey] = restoreAddressType(
            builder, forOp.getLoc(), i16Value, sourceType, domain);
      }
      mapping.map(source.source, normalizedValues[defaultKey]);
    }

    DenseSet<Operation *> replacedUpdates;
    for (unsigned sourceIndex : activeSources) {
      for (unsigned planIndex : sources[sourceIndex].planIndices) {
        if (Operation *update = recurrences[planIndex].recurrence.updateOp) {
          replacedUpdates.insert(update);
        }
      }
    }

    DenseMap<Operation *, Operation *> clonedOps;
    for (Operation &oldOp : forOp.getBody()->without_terminator()) {
      if (replacedUpdates.contains(&oldOp)) {
        continue;
      }
      Operation *cloned = builder.clone(oldOp, mapping);
      clonedOps[&oldOp] = cloned;
    }

    for (unsigned sourceIndex : activeSources) {
      for (unsigned planIndex : sources[sourceIndex].planIndices) {
        RecurrencePlan &plan = recurrences[planIndex];
        for (const RewriteTarget &target : plan.targets) {
          auto cloned = clonedOps.find(target.owner);
          if (cloned == clonedOps.end()) {
            continue;
          }
          Type wantedType =
              target.owner->getOperand(target.operandNumber).getType();
          Value normalized = normalizedValues[{planIndex, wantedType}];
          cloned->second->setOperand(target.operandNumber, normalized);
        }
      }
    }

    auto oldYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    SmallVector<Value> yields;
    builder.setInsertionPointToEnd(newFor.getBody());
    for (auto [argIndex, yielded] : llvm::enumerate(oldYield.getOperands())) {
      Value oldArg = forOp.getRegionIterArgs()[argIndex];
      auto sourceIt = sourceIndices.find(oldArg);
      bool isActive =
          sourceIt != sourceIndices.end() &&
          llvm::is_contained(activeSources, sourceIt->second);
      if (!isActive) {
        yields.push_back(mapping.lookupOrDefault(yielded));
        continue;
      }

      SourceRewrite &source = sources[sourceIt->second];
      Value current = newFor.getRegionIterArgs()[argIndex];
      arith::IntegerOverflowFlags flags = arith::IntegerOverflowFlags::none;
      if (source.needsSignedNoWrap) {
        flags = flags | arith::IntegerOverflowFlags::nsw;
      }
      if (source.needsUnsignedNoWrap) {
        flags = flags | arith::IntegerOverflowFlags::nuw;
      }
      // INT16_MIN is representable as an addend, but its positive magnitude
      // is not representable as a signed i16 subtraction operand.
      bool useSignedMinimumAdd =
          source.needsSignedNoWrap &&
          source.increment == std::numeric_limits<int16_t>::min();
      if (source.increment < 0 && !useSignedMinimumAdd) {
        Value decrement = builder.create<arith::ConstantIntOp>(
            forOp.getLoc(), -source.increment, kAddressWidth);
        yields.push_back(
            builder.create<arith::SubIOp>(forOp.getLoc(), current, decrement,
                                          flags));
      } else {
        Value increment = builder.create<arith::ConstantIntOp>(
            forOp.getLoc(), source.increment, kAddressWidth);
        yields.push_back(builder.create<arith::AddIOp>(forOp.getLoc(), current,
                                                       increment, flags));
      }
    }
    builder.create<scf::YieldOp>(oldYield.getLoc(), yields);

    builder.setInsertionPointAfter(newFor);
    for (auto [resultIndex, oldResult] : llvm::enumerate(forOp.getResults())) {
      if (oldResult.use_empty()) {
        continue;
      }
      Value oldArg = forOp.getRegionIterArgs()[resultIndex];
      auto sourceIt = sourceIndices.find(oldArg);
      bool isActive =
          sourceIt != sourceIndices.end() &&
          llvm::is_contained(activeSources, sourceIt->second);
      if (!isActive) {
        oldResult.replaceAllUsesWith(newFor.getResult(resultIndex));
        continue;
      }
      SourceRewrite &source = sources[sourceIt->second];
      unsigned defaultPlan = source.planIndices.front();
      AddressDomain domain = recurrences[defaultPlan].recurrence.domain;
      Value restored = restoreAddressType(
          builder, forOp.getLoc(), newFor.getResult(resultIndex),
          oldResult.getType(), domain);
      oldResult.replaceAllUsesWith(restored);
    }
    forOp.erase();
  }

private:
  std::optional<unsigned>
  requestRecurrence(Value value, Operation *owner, unsigned operandNumber,
                    AddressDomain domain) {
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
                 target.operandNumber == operandNumber;
        }))
      targets.push_back({owner, operandNumber});
    return index;
  }

  bool analyzeIntegerValue(Value value, Operation *owner,
                           unsigned operandNumber,
                           AddressDomain domain) {
    if (forOp.isDefinedOutsideOfLoop(value)) {
      return true;
    }
    Value recurrenceValue = value;
    Operation *rewriteOwner = owner;
    unsigned rewriteOperand = operandNumber;
    // Pointer advancement is index-typed even when the stateful op exposes an
    // i32 stride. Look through a signed, value-preserving cast so both uses
    // share one canonical i16 recurrence.
    if (auto cast = value.getDefiningOp<arith::IndexCastOp>()) {
      recurrenceValue = cast.getIn();
      rewriteOwner = cast;
      rewriteOperand = 0;
    }
    if (auto cast = value.getDefiningOp<arith::IndexCastUIOp>()) {
      if (domain != AddressDomain::Unsigned ||
          !cast.getIn().getType().isIndex() ||
          !cast.getType().isInteger(kAddressWidth)) {
        return false;
      }
      recurrenceValue = cast.getIn();
    }
    auto recurrence =
        matchProvenRecurrence(recurrenceValue, forOp, tripCount, domain);
    if (!recurrence) {
      return false;
    }
    auto plan =
        requestRecurrence(recurrenceValue, rewriteOwner, rewriteOperand,
                          domain);
    return plan.has_value();
  }

  bool analyzeBase(Value base) {
    if (forOp.isDefinedOutsideOfLoop(base)) {
      return true;
    }

    if (auto addPtr = base.getDefiningOp<pto::AddPtrOp>()) {
      return analyzeIntegerValue(addPtr.getOffset(), addPtr, 1,
                                 AddressDomain::Signed);
    }

    auto iterArg = dyn_cast<BlockArgument>(base);
    if (!iterArg || iterArg.getOwner() != forOp.getBody() ||
        iterArg.getArgNumber() == 0) {
      return false;
    }
    unsigned index = iterArg.getArgNumber() - 1;
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    auto addPtr = yieldOp.getOperand(index).getDefiningOp<pto::AddPtrOp>();
    if (!addPtr) {
      return false;
    }
    // Pointer liveness and recurrence shape constrain post-update formation,
    // but not the independent range proof for the integer address leaf.
    return analyzeIntegerValue(addPtr.getOffset(), addPtr, 1,
                               AddressDomain::Signed);
  }

  void analyzeCandidate(Operation *op, const pto::PostUpdateOpInfo &info) {
    Value base = op->getOperand(info.baseOperandIdx);
    (void)analyzeBase(base);

    if (info.strideOperandIdx) {
      Value stride = op->getOperand(*info.strideOperandIdx);
      (void)analyzeIntegerValue(stride, op, *info.strideOperandIdx,
                                info.strideDomain);
    }
  }

  scf::ForOp forOp;
  uint64_t tripCount;
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
    module.walk([&](scf::ForOp forOp) { loops.push_back(forOp); });

    OpBuilder builder(&getContext());
    for (scf::ForOp forOp : loops) {
      auto tripCount = getConstantTripCount(forOp);
      if (!tripCount) {
        continue;
      }
      LoopPlanner planner(forOp, *tripCount);
      planner.collect();
      if (!planner.empty()) {
        planner.rewrite(builder);
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVPTONormalizeAddressRecurrencesPass() {
  return std::make_unique<VPTONormalizeAddressRecurrencesPass>();
}

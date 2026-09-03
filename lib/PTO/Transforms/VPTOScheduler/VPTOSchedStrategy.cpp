// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOSchedStrategy.cpp - VPTO scheduling strategy -----------------===//

#include "PTO/Transforms/VPTOScheduler/VPTOSchedStrategy.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <optional>

using namespace mlir;
using namespace mlir::pto;

namespace {

struct RankedCandidate {
  const VPTOSchedCandidate *candidate = nullptr;
  bool exceedsLimit = false;
  int64_t excessGrowthCost = 0;
  int64_t projectedExcessCost = 0;
  int64_t lookaheadExcessCost = 0;
  int64_t lookaheadRiskCost = 0;
  int64_t lookaheadEndCost = 0;
  int64_t highPressureProjectedCost = 0;
  int64_t highPressureReleaseCredit = 0;
  int64_t nearLimitProjectedCost = 0;
  int64_t nearLimitReleaseCredit = 0;
  int64_t pressureDeltaCost = 0;
  int64_t closureProjected = 0;
  int64_t closureReleaseCredit = 0;
  bool urgentCriticalPath = false;
  bool opensPressureFrontier = false;
  bool advancesPressureClosure = false;
};

struct RankingContext {
  SmallVector<bool> nearLimitPressureSets;
  SmallVector<bool> highPressureSets;
  unsigned longestCriticalPath = 0;
  unsigned urgentSlack = 0;
  bool hasNearLimitPressure = false;
  bool hasHighPressure = false;
};

static bool checkedMultiplyAdd(int64_t lhs, int64_t rhs, int64_t &total) {
  int64_t product = 0;
  int64_t updated = 0;
  if (llvm::MulOverflow(lhs, rhs, product)) {
    return false;
  }
  if (llvm::AddOverflow(total, product, updated)) {
    return false;
  }
  total = updated;
  return true;
}

static bool hasCompletePressureResult(const VPTOScheduleContext &context,
                                      const VPTOSchedCandidate &candidate) {
  size_t pressureSetCount = context.model.getPressureSets().size();
  return context.currentPressure.size() == pressureSetCount &&
         candidate.pressure.delta.size() == pressureSetCount &&
         candidate.pressure.released.size() == pressureSetCount &&
         candidate.pressure.introduced.size() == pressureSetCount &&
         candidate.pressure.projected.size() == pressureSetCount &&
         candidate.pressure.projectedExcess.size() == pressureSetCount &&
         candidate.lookaheadPeak.size() == pressureSetCount &&
         candidate.lookaheadEnd.size() == pressureSetCount;
}

static LogicalResult
validateCandidateContext(const VPTOScheduleContext &context,
                         const VPTOSchedCandidate &candidate,
                         std::string &detail) {
  bool invalidCandidate = !candidate.unit ||
                          candidate.direction != context.direction ||
                          candidate.issueCycle != context.issueCycle ||
                          !hasCompletePressureResult(context, candidate);
  if (invalidCandidate) {
    detail = "candidate does not match the current scheduling context";
    return failure();
  }
  if (!context.model.getSchedClass(candidate.unit->getOperation()).known) {
    detail = "candidate has an unknown scheduling class";
    return failure();
  }
  return success();
}

static void updateCriticalPathContext(const VPTOScheduleContext &context,
                                      const VPTOSchedCandidate &candidate,
                                      RankingContext &rankingContext) {
  VPTOSchedParameters parameters =
      context.model.getSchedParameters(candidate.unit->getOperation());
  if (candidate.criticalPath > rankingContext.longestCriticalPath) {
    rankingContext.longestCriticalPath = candidate.criticalPath;
    rankingContext.urgentSlack = parameters.writeLatency;
    return;
  }
  if (candidate.criticalPath == rankingContext.longestCriticalPath) {
    rankingContext.urgentSlack =
        std::max(rankingContext.urgentSlack, parameters.writeLatency);
  }
}

static LogicalResult populatePressureBands(const VPTOScheduleContext &context,
                                           RankingContext &rankingContext,
                                           std::string &detail) {
  for (auto [index, pressureSet] :
       llvm::enumerate(context.model.getPressureSets())) {
    if (context.currentPressure[index] < 0) {
      detail = "current pressure contains a negative value";
      return failure();
    }
    if (!pressureSet.limit) {
      continue;
    }
    int64_t limit = static_cast<int64_t>(*pressureSet.limit);
    // Redirect producer-heavy frontiers before the next instruction would
    // spill, while leaving room for equally safe critical-path candidates.
    bool nearLimit = context.currentPressure[index] * 2 >= limit;
    bool highPressure = context.currentPressure[index] * 3 >= limit * 2;
    rankingContext.nearLimitPressureSets[index] = nearLimit;
    rankingContext.highPressureSets[index] = highPressure;
    rankingContext.hasNearLimitPressure |= nearLimit;
    rankingContext.hasHighPressure |= highPressure;
  }
  return success();
}

static FailureOr<RankingContext>
buildRankingContext(const VPTOScheduleContext &context,
                    ArrayRef<VPTOSchedCandidate> candidates,
                    std::string &detail) {
  ArrayRef<VPTORegPressureSet> pressureSets = context.model.getPressureSets();
  bool pressureCountMatches =
      context.currentPressure.size() == pressureSets.size();
  if (!pressureCountMatches) {
    detail = "current pressure does not match target pressure sets";
    return failure();
  }
  if (context.closurePressureSet &&
      *context.closurePressureSet >= pressureSets.size()) {
    detail = "closure pressure set does not match the target model";
    return failure();
  }

  RankingContext rankingContext;
  rankingContext.nearLimitPressureSets.assign(pressureSets.size(), false);
  rankingContext.highPressureSets.assign(pressureSets.size(), false);
  for (const VPTOSchedCandidate &candidate : candidates) {
    if (failed(validateCandidateContext(context, candidate, detail))) {
      return failure();
    }
    updateCriticalPathContext(context, candidate, rankingContext);
  }
  if (failed(populatePressureBands(context, rankingContext, detail))) {
    return failure();
  }
  return rankingContext;
}

struct PressureScoreState {
  int64_t currentExcess = 0;
  int64_t projectedExcess = 0;
};

static FailureOr<PressureScoreState>
validatePressureScoreState(const VPTOScheduleContext &context,
                           const VPTOSchedCandidate &candidate,
                           const VPTORegPressureSet &pressureSet,
                           unsigned index, std::string &detail) {
  bool invalid = pressureSet.weight < 0 || pressureSet.spillCost < 0 ||
                 context.currentPressure[index] < 0 ||
                 candidate.pressure.released[index] < 0 ||
                 candidate.pressure.introduced[index] < 0 ||
                 candidate.pressure.projected[index] < 0 ||
                 candidate.pressure.projectedExcess[index] < 0;
  if (invalid) {
    detail = "pressure set or candidate contains an invalid scoring parameter";
    return failure();
  }
  int64_t expectedDelta = 0;
  int64_t expectedProjected = 0;
  bool inconsistent =
      llvm::SubOverflow(candidate.pressure.introduced[index],
                        candidate.pressure.released[index], expectedDelta) ||
      llvm::AddOverflow(context.currentPressure[index],
                        candidate.pressure.delta[index], expectedProjected) ||
      expectedDelta != candidate.pressure.delta[index] ||
      expectedProjected != candidate.pressure.projected[index];
  if (inconsistent) {
    detail = "candidate pressure snapshot is inconsistent or overflows";
    return failure();
  }
  PressureScoreState state;
  if (!pressureSet.limit) {
    if (candidate.pressure.projectedExcess[index] != 0) {
      detail = "unbounded pressure set has projected excess";
      return failure();
    }
    return state;
  }
  state.currentExcess =
      std::max<int64_t>(0, context.currentPressure[index] - *pressureSet.limit);
  state.projectedExcess = candidate.pressure.projectedExcess[index];
  int64_t expectedExcess = std::max<int64_t>(
      0, expectedProjected - static_cast<int64_t>(*pressureSet.limit));
  if (state.projectedExcess != expectedExcess) {
    detail = "candidate projected pressure excess is inconsistent";
    return failure();
  }
  return state;
}

static LogicalResult
accumulateBasePressureCosts(const VPTORegPressureSet &pressureSet,
                            const VPTOSchedCandidate &candidate, unsigned index,
                            const PressureScoreState &state,
                            RankedCandidate &rank, std::string &detail) {
  int64_t excessGrowth =
      std::max<int64_t>(0, state.projectedExcess - state.currentExcess);
  bool overflow =
      !checkedMultiplyAdd(pressureSet.spillCost, excessGrowth,
                          rank.excessGrowthCost) ||
      !checkedMultiplyAdd(pressureSet.spillCost, state.projectedExcess,
                          rank.projectedExcessCost) ||
      !checkedMultiplyAdd(pressureSet.weight, candidate.pressure.delta[index],
                          rank.pressureDeltaCost);
  if (overflow) {
    detail = "candidate pressure score overflow";
    return failure();
  }
  rank.exceedsLimit |= state.projectedExcess > 0;
  return success();
}

static LogicalResult
accumulatePressureBandCosts(const RankingContext &context,
                            const VPTORegPressureSet &pressureSet,
                            const VPTOSchedCandidate &candidate, unsigned index,
                            RankedCandidate &rank, std::string &detail) {
  if (context.nearLimitPressureSets[index]) {
    bool overflow = !checkedMultiplyAdd(pressureSet.spillCost,
                                        candidate.pressure.projected[index],
                                        rank.nearLimitProjectedCost) ||
                    !checkedMultiplyAdd(pressureSet.spillCost,
                                        candidate.pressure.released[index],
                                        rank.nearLimitReleaseCredit);
    if (overflow) {
      detail = "candidate near-limit pressure score overflow";
      return failure();
    }
  }
  if (context.highPressureSets[index]) {
    bool overflow = !checkedMultiplyAdd(pressureSet.spillCost,
                                        candidate.pressure.projected[index],
                                        rank.highPressureProjectedCost) ||
                    !checkedMultiplyAdd(pressureSet.spillCost,
                                        candidate.pressure.released[index],
                                        rank.highPressureReleaseCredit);
    if (overflow) {
      detail = "candidate high-pressure score overflow";
      return failure();
    }
  }
  return success();
}

static LogicalResult
accumulateLookaheadCosts(const RankingContext &context,
                         const VPTORegPressureSet &pressureSet,
                         const VPTOSchedCandidate &candidate, unsigned index,
                         RankedCandidate &rank, std::string &detail) {
  if (!context.nearLimitPressureSets[index]) {
    return success();
  }
  int64_t limit = static_cast<int64_t>(*pressureSet.limit);
  int64_t criticalThreshold = (limit + 1) / 2;
  int64_t bandWidth = std::max<int64_t>(1, (limit - criticalThreshold + 3) / 4);
  int64_t lookaheadPeak = candidate.lookaheadPeak[index];
  int64_t lookaheadEnd = candidate.lookaheadEnd[index];
  if (lookaheadPeak < 0 || lookaheadEnd < 0) {
    detail = "candidate contains negative lookahead pressure";
    return failure();
  }
  int64_t lookaheadExcess = std::max<int64_t>(0, lookaheadPeak - limit);
  int64_t lookaheadRisk =
      std::max<int64_t>(0, candidate.pressure.projected[index] -
                               criticalThreshold) /
      bandWidth;
  bool overflow = !checkedMultiplyAdd(pressureSet.spillCost, lookaheadExcess,
                                      rank.lookaheadExcessCost) ||
                  !checkedMultiplyAdd(pressureSet.weight, lookaheadRisk,
                                      rank.lookaheadRiskCost) ||
                  !checkedMultiplyAdd(pressureSet.weight, lookaheadEnd,
                                      rank.lookaheadEndCost);
  if (overflow) {
    detail = "candidate lookahead pressure score overflow";
    return failure();
  }
  return success();
}

static FailureOr<RankedCandidate>
rankCandidate(const VPTOScheduleContext &context,
              const RankingContext &rankingContext,
              const VPTOSchedCandidate &candidate, std::string &detail) {
  if (!candidate.unit || candidate.direction != context.direction ||
      candidate.issueCycle != context.issueCycle) {
    detail = "candidate does not match the current scheduling context";
    return failure();
  }
  if (!hasCompletePressureResult(context, candidate)) {
    detail = "candidate pressure does not match target pressure sets";
    return failure();
  }
  if (!context.model.getSchedClass(candidate.unit->getOperation()).known) {
    detail = "candidate has an unknown scheduling class";
    return failure();
  }
  RankedCandidate rank;
  rank.candidate = &candidate;
  rank.opensPressureFrontier = candidate.opensPressureFrontier;
  rank.advancesPressureClosure = candidate.advancesPressureClosure;
  for (auto [index, pressureSet] :
       llvm::enumerate(context.model.getPressureSets())) {
    FailureOr<PressureScoreState> state = validatePressureScoreState(
        context, candidate, pressureSet, index, detail);
    bool invalidScore =
        failed(state) ||
        failed(accumulateBasePressureCosts(pressureSet, candidate, index,
                                           *state, rank, detail)) ||
        failed(accumulatePressureBandCosts(rankingContext, pressureSet,
                                           candidate, index, rank, detail)) ||
        failed(accumulateLookaheadCosts(rankingContext, pressureSet, candidate,
                                        index, rank, detail));
    if (invalidScore) {
      return failure();
    }
  }
  unsigned criticalPathSlack =
      rankingContext.longestCriticalPath - candidate.criticalPath;
  rank.urgentCriticalPath = criticalPathSlack <= rankingContext.urgentSlack;
  if (context.closurePressureSet) {
    unsigned index = *context.closurePressureSet;
    rank.closureProjected = candidate.pressure.projected[index];
    rank.closureReleaseCredit = candidate.pressure.released[index];
  }
  return rank;
}

template <typename T> static std::optional<bool> preferLower(T lhs, T rhs) {
  if (lhs == rhs) {
    return std::nullopt;
  }
  return lhs < rhs;
}

template <typename T> static std::optional<bool> preferHigher(T lhs, T rhs) {
  if (lhs == rhs) {
    return std::nullopt;
  }
  return lhs > rhs;
}

static std::optional<bool> compareBasePressure(const RankedCandidate &lhs,
                                               const RankedCandidate &rhs) {
  if (lhs.exceedsLimit != rhs.exceedsLimit) {
    return !lhs.exceedsLimit;
  }
  if (auto result = preferLower(lhs.excessGrowthCost, rhs.excessGrowthCost)) {
    return result;
  }
  return preferLower(lhs.projectedExcessCost, rhs.projectedExcessCost);
}

static std::optional<bool> comparePressureClosure(const RankedCandidate &lhs,
                                                  const RankedCandidate &rhs) {
  if (lhs.advancesPressureClosure != rhs.advancesPressureClosure) {
    return lhs.advancesPressureClosure;
  }
  if (auto result = preferLower(lhs.closureProjected, rhs.closureProjected)) {
    return result;
  }
  return preferHigher(lhs.closureReleaseCredit, rhs.closureReleaseCredit);
}

static std::optional<bool> compareHighPressure(const RankedCandidate &lhs,
                                               const RankedCandidate &rhs) {
  if (auto result = preferLower(lhs.highPressureProjectedCost,
                                rhs.highPressureProjectedCost)) {
    return result;
  }
  return preferHigher(lhs.highPressureReleaseCredit,
                      rhs.highPressureReleaseCredit);
}

static std::optional<bool>
compareNearLimitPressure(const RankedCandidate &lhs,
                         const RankedCandidate &rhs) {
  if (auto result =
          preferLower(lhs.lookaheadExcessCost, rhs.lookaheadExcessCost)) {
    return result;
  }
  if (auto result = preferLower(lhs.lookaheadRiskCost, rhs.lookaheadRiskCost)) {
    return result;
  }
  if (lhs.urgentCriticalPath != rhs.urgentCriticalPath) {
    return lhs.urgentCriticalPath;
  }
  if (lhs.opensPressureFrontier != rhs.opensPressureFrontier) {
    return !lhs.opensPressureFrontier;
  }
  if (auto result = preferLower(lhs.lookaheadEndCost, rhs.lookaheadEndCost)) {
    return result;
  }
  if (auto result =
          preferLower(lhs.nearLimitProjectedCost, rhs.nearLimitProjectedCost)) {
    return result;
  }
  return preferHigher(lhs.nearLimitReleaseCredit, rhs.nearLimitReleaseCredit);
}

static bool isBetterCandidate(const RankedCandidate &lhs,
                              const RankedCandidate &rhs,
                              bool hasNearLimitPressure,
                              bool hasHighPressure,
                              bool hasPressureClosure) {
  if (auto result = compareBasePressure(lhs, rhs)) {
    return *result;
  }
  if (hasPressureClosure) {
    if (auto result = comparePressureClosure(lhs, rhs)) {
      return *result;
    }
  }
  if (hasHighPressure) {
    if (auto result = compareHighPressure(lhs, rhs)) {
      return *result;
    }
  }
  if (hasNearLimitPressure) {
    if (auto result = compareNearLimitPressure(lhs, rhs)) {
      return *result;
    }
  }
  if (lhs.candidate->criticalPath != rhs.candidate->criticalPath) {
    return lhs.candidate->criticalPath > rhs.candidate->criticalPath;
  }
  if (lhs.pressureDeltaCost != rhs.pressureDeltaCost) {
    return lhs.pressureDeltaCost < rhs.pressureDeltaCost;
  }
  return lhs.candidate->originalIndex < rhs.candidate->originalIndex;
}

static StringRef getBasePressureReason(const RankedCandidate &selected,
                                       const RankedCandidate &runnerUp) {
  if (selected.exceedsLimit != runnerUp.exceedsLimit) {
    return "pressure-safe-candidate";
  }
  if (selected.excessGrowthCost != runnerUp.excessGrowthCost) {
    return "lower-excess-growth";
  }
  if (selected.projectedExcessCost != runnerUp.projectedExcessCost) {
    return "lower-projected-excess";
  }
  return {};
}

static StringRef getClosureReason(const RankedCandidate &selected,
                                  const RankedCandidate &runnerUp) {
  if (selected.advancesPressureClosure != runnerUp.advancesPressureClosure) {
    return "advance-pressure-closure";
  }
  if (selected.closureProjected != runnerUp.closureProjected) {
    return "closure-pressure-preserving";
  }
  if (selected.closureReleaseCredit != runnerUp.closureReleaseCredit) {
    return "closure-live-range-closing";
  }
  return {};
}

static StringRef getHighPressureReason(const RankedCandidate &selected,
                                       const RankedCandidate &runnerUp) {
  if (selected.highPressureProjectedCost !=
      runnerUp.highPressureProjectedCost) {
    return "high-pressure-preserving";
  }
  if (selected.highPressureReleaseCredit !=
      runnerUp.highPressureReleaseCredit) {
    return "high-pressure-live-range-closing";
  }
  return {};
}

static StringRef getNearLimitReason(const RankedCandidate &selected,
                                    const RankedCandidate &runnerUp) {
  if (selected.lookaheadExcessCost != runnerUp.lookaheadExcessCost) {
    return "bounded-lookahead-avoids-excess";
  }
  if (selected.lookaheadRiskCost != runnerUp.lookaheadRiskCost) {
    return "bounded-lookahead-lower-risk";
  }
  if (selected.urgentCriticalPath != runnerUp.urgentCriticalPath) {
    return "urgent-critical-path";
  }
  if (selected.opensPressureFrontier != runnerUp.opensPressureFrontier) {
    return "continue-open-pressure-frontier";
  }
  if (selected.lookaheadEndCost != runnerUp.lookaheadEndCost) {
    return "bounded-lookahead-lower-ending-pressure";
  }
  if (selected.nearLimitProjectedCost != runnerUp.nearLimitProjectedCost) {
    if (selected.nearLimitReleaseCredit > runnerUp.nearLimitReleaseCredit) {
      return "near-limit-live-range-closing";
    }
    return "near-limit-pressure-preserving";
  }
  if (selected.nearLimitReleaseCredit != runnerUp.nearLimitReleaseCredit) {
    return "near-limit-live-range-closing";
  }
  return {};
}

static StringRef getDecisionReason(const RankedCandidate &selected,
                                   const RankedCandidate &runnerUp,
                                   bool hasNearLimitPressure,
                                   bool hasHighPressure,
                                   bool hasPressureClosure) {
  if (StringRef reason = getBasePressureReason(selected, runnerUp);
      !reason.empty()) {
    return reason;
  }
  if (hasPressureClosure) {
    if (StringRef reason = getClosureReason(selected, runnerUp);
        !reason.empty()) {
      return reason;
    }
  }
  if (hasHighPressure) {
    if (StringRef reason = getHighPressureReason(selected, runnerUp);
        !reason.empty()) {
      return reason;
    }
  }
  if (hasNearLimitPressure) {
    if (StringRef reason = getNearLimitReason(selected, runnerUp);
        !reason.empty()) {
      return reason;
    }
  }
  if (selected.candidate->criticalPath != runnerUp.candidate->criticalPath) {
    return "longer-critical-path";
  }
  if (selected.pressureDeltaCost != runnerUp.pressureDeltaCost) {
    return "lower-pressure-delta";
  }
  if (selected.candidate->originalIndex != runnerUp.candidate->originalIndex) {
    return "deterministic-tie-break";
  }
  return "stable-candidate-order";
}

} // namespace

FailureOr<VPTOSchedDecision>
VPTODefaultSchedStrategy::pickCandidate(const VPTOScheduleContext &context,
                                        ArrayRef<VPTOSchedCandidate> candidates,
                                        std::string &detail) const {
  if (candidates.empty()) {
    detail = "strategy received no candidates";
    return failure();
  }

  SmallVector<RankedCandidate> ranks;
  ranks.reserve(candidates.size());
  FailureOr<RankingContext> rankingContext =
      buildRankingContext(context, candidates, detail);
  if (failed(rankingContext)) {
    return failure();
  }
  for (const VPTOSchedCandidate &candidate : candidates) {
    FailureOr<RankedCandidate> rank =
        rankCandidate(context, *rankingContext, candidate, detail);
    if (failed(rank)) {
      return failure();
    }
    ranks.push_back(*rank);
  }

  const RankedCandidate *selected = &ranks.front();
  for (const RankedCandidate &rank : llvm::drop_begin(ranks)) {
    if (isBetterCandidate(rank, *selected,
                          rankingContext->hasNearLimitPressure,
                          rankingContext->hasHighPressure,
                          context.closurePressureSet.has_value())) {
      selected = &rank;
    }
  }

  const RankedCandidate *runnerUp = nullptr;
  for (const RankedCandidate &rank : ranks) {
    if (&rank == selected) {
      continue;
    }
    if (!runnerUp || isBetterCandidate(rank, *runnerUp,
                                       rankingContext->hasNearLimitPressure,
                                       rankingContext->hasHighPressure,
                                       context.closurePressureSet.has_value())) {
      runnerUp = &rank;
    }
  }

  StringRef reason = runnerUp ? getDecisionReason(
                                    *selected, *runnerUp,
                                    rankingContext->hasNearLimitPressure,
                                    rankingContext->hasHighPressure,
                                    context.closurePressureSet.has_value())
                              : StringRef("only-candidate");
  const VPTOSchedCandidate &candidate = *selected->candidate;
  return VPTOSchedDecision{candidate.unit, candidate.direction,
                           candidate.issueCycle, reason.str()};
}

const VPTOSchedStrategy &mlir::pto::getDefaultVPTOSchedStrategy() {
  static const VPTODefaultSchedStrategy strategy;
  return strategy;
}

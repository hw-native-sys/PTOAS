// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTORegPressureTracker.cpp - VPTO pressure tracking ---------------===//

#include "PTO/Transforms/VPTOScheduler/VPTORegPressureTracker.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::pto;

VPTORegPressureTracker::VPTORegPressureTracker(const VPTOSchedModel &model,
                                               const VPTOSchedDAG &dag,
                                               VPTOSchedDirection direction)
    : model(model), dag(dag), direction(direction) {
  for (auto [index, pressureSet] : llvm::enumerate(model.getPressureSets())) {
    pressureSetIndex.try_emplace(pressureSet.id, index);
  }
  for (Value liveOut : dag.getLiveOuts()) {
    liveOutValues.insert(getPressureRepresentative(liveOut));
  }
  current.assign(model.getPressureSets().size(), 0);
  peak.assign(model.getPressureSets().size(), 0);
  if (direction == VPTOSchedDirection::Top)
    initializeTop();
  else
    initializeBottom();
  peak = current;
}

Value VPTORegPressureTracker::getPressureRepresentative(Value value) const {
  while (value) {
    Operation *definingOp = value.getDefiningOp();
    if (!definingOp || !dag.lookup(definingOp)) {
      break;
    }
    Value representative = model.getPressureRepresentative(value);
    if (!representative || representative == value) {
      break;
    }
    value = representative;
  }
  return value;
}

bool VPTORegPressureTracker::isLive(Value value) const {
  return liveValues.contains(getPressureRepresentative(value));
}

void VPTORegPressureTracker::addValuePressure(
    Value value, int sign, MutableArrayRef<int64_t> values) const {
  value = getPressureRepresentative(value);
  for (const VPTORegPressureContribution &contribution :
       model.getPressure(value)) {
    auto found = pressureSetIndex.find(contribution.pressureSet);
    if (found == pressureSetIndex.end())
      continue;
    values[found->second] += sign * static_cast<int64_t>(contribution.units);
  }
}

bool VPTORegPressureTracker::isLiveOut(Value value) const {
  return liveOutValues.contains(getPressureRepresentative(value));
}

bool VPTORegPressureTracker::resultNeedsLiveness(Value value) const {
  value = getPressureRepresentative(value);
  return isLiveOut(value) || remainingUses.lookup(value) > 0;
}

void VPTORegPressureTracker::initializeTop() {
  for (Value liveIn : dag.getLiveIns()) {
    liveIn = getPressureRepresentative(liveIn);
    if (liveValues.insert(liveIn).second)
      addValuePressure(liveIn, 1, current);
  }
  for (const std::unique_ptr<VPTOSUnit> &unit : dag.getUnits()) {
    for (Value operand : unit->getOperation()->getOperands()) {
      operand = getPressureRepresentative(operand);
      ++remainingUses[operand];
    }
  }
}

void VPTORegPressureTracker::initializeBottom() {
  for (Value liveOut : dag.getLiveOuts()) {
    liveOut = getPressureRepresentative(liveOut);
    if (liveValues.insert(liveOut).second)
      addValuePressure(liveOut, 1, current);
  }
}

VPTORegPressureEvaluation
VPTORegPressureTracker::evaluateTop(const VPTOSUnit &unit) const {
  VPTORegPressureEvaluation evaluation;
  evaluation.delta.assign(current.size(), 0);
  evaluation.released.assign(current.size(), 0);
  evaluation.introduced.assign(current.size(), 0);

  DenseMap<Value, unsigned> candidateUses;
  for (Value operand : unit.getOperation()->getOperands()) {
    operand = getPressureRepresentative(operand);
    ++candidateUses[operand];
  }
  for (const auto &entry : candidateUses) {
    Value value = entry.first;
    if (liveValues.contains(value) && !isLiveOut(value) &&
        remainingUses.lookup(value) == entry.second) {
      addValuePressure(value, -1, evaluation.delta);
      addValuePressure(value, 1, evaluation.released);
    }
  }
  DenseSet<Value> candidateResults;
  for (Value result : unit.getOperation()->getResults()) {
    result = getPressureRepresentative(result);
    if (!candidateResults.insert(result).second) {
      continue;
    }
    bool neededAfterCandidate =
        isLiveOut(result) ||
        remainingUses.lookup(result) > candidateUses.lookup(result);
    bool introducesLiveResult =
        !liveValues.contains(result) && neededAfterCandidate;
    if (introducesLiveResult) {
      addValuePressure(result, 1, evaluation.delta);
      addValuePressure(result, 1, evaluation.introduced);
    }
  }
  refreshSummary(evaluation);
  return evaluation;
}

VPTORegPressureEvaluation
VPTORegPressureTracker::evaluateBottom(const VPTOSUnit &unit) const {
  VPTORegPressureEvaluation evaluation;
  evaluation.delta.assign(current.size(), 0);
  evaluation.released.assign(current.size(), 0);
  evaluation.introduced.assign(current.size(), 0);
  DenseSet<Value> results;
  for (Value result : unit.getOperation()->getResults()) {
    results.insert(getPressureRepresentative(result));
  }
  DenseSet<Value> operands;
  for (Value operand : unit.getOperation()->getOperands()) {
    operands.insert(getPressureRepresentative(operand));
  }
  DenseSet<Value> affectedValues;
  for (Value result : results) {
    affectedValues.insert(result);
  }
  for (Value operand : operands) {
    affectedValues.insert(operand);
  }
  for (Value value : affectedValues) {
    bool liveBefore = liveValues.contains(value);
    bool liveAfter = (liveBefore && !results.contains(value)) ||
                     operands.contains(value);
    if (liveBefore && !liveAfter) {
      addValuePressure(value, -1, evaluation.delta);
      addValuePressure(value, 1, evaluation.released);
    }
    if (!liveBefore && liveAfter) {
      addValuePressure(value, 1, evaluation.delta);
      addValuePressure(value, 1, evaluation.introduced);
    }
  }
  refreshSummary(evaluation);
  return evaluation;
}

void VPTORegPressureTracker::refreshSummary(
    VPTORegPressureEvaluation &evaluation) const {
  evaluation.projected.resize(current.size());
  evaluation.projectedExcess.assign(current.size(), 0);
  for (auto [index, pressureSet] : llvm::enumerate(model.getPressureSets())) {
    evaluation.projected[index] = current[index] + evaluation.delta[index];
    if (pressureSet.limit)
      evaluation.projectedExcess[index] =
          std::max<int64_t>(0, evaluation.projected[index] -
                                   static_cast<int64_t>(*pressureSet.limit));
  }
}

VPTORegPressureEvaluation
VPTORegPressureTracker::evaluate(const VPTOSUnit &unit) const {
  return direction == VPTOSchedDirection::Top ? evaluateTop(unit)
                                              : evaluateBottom(unit);
}

LogicalResult VPTORegPressureTracker::commit(const VPTOSUnit &unit) {
  VPTORegPressureEvaluation evaluation = evaluate(unit);
  for (auto [index, projected] : llvm::enumerate(evaluation.projected)) {
    if (projected < 0)
      return failure();
    current[index] = projected;
    peak[index] = std::max(peak[index], projected);
  }

  if (direction == VPTOSchedDirection::Top) {
    for (Value operand : unit.getOperation()->getOperands()) {
      operand = getPressureRepresentative(operand);
      auto found = remainingUses.find(operand);
      if (found == remainingUses.end() || found->second == 0)
        return failure();
      if (--found->second == 0 && !isLiveOut(operand))
        liveValues.erase(operand);
    }
    for (Value result : unit.getOperation()->getResults()) {
      result = getPressureRepresentative(result);
      if (resultNeedsLiveness(result))
        liveValues.insert(result);
    }
    return success();
  }

  for (Value result : unit.getOperation()->getResults()) {
    result = getPressureRepresentative(result);
    liveValues.erase(result);
  }
  for (Value operand : unit.getOperation()->getOperands()) {
    operand = getPressureRepresentative(operand);
    liveValues.insert(operand);
  }
  return success();
}

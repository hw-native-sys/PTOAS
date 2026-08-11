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
  for (auto [index, pressureSet] : llvm::enumerate(model.getPressureSets()))
    pressureSetIndex.try_emplace(pressureSet.id, index);
  current.assign(model.getPressureSets().size(), 0);
  peak.assign(model.getPressureSets().size(), 0);
  if (direction == VPTOSchedDirection::Top)
    initializeTop();
  else
    initializeBottom();
  peak = current;
}

void VPTORegPressureTracker::addValuePressure(
    Value value, int sign, MutableArrayRef<int64_t> values) const {
  for (const VPTORegPressureContribution &contribution :
       model.getPressure(value)) {
    auto found = pressureSetIndex.find(contribution.pressureSet);
    if (found == pressureSetIndex.end())
      continue;
    values[found->second] += sign * static_cast<int64_t>(contribution.units);
  }
}

bool VPTORegPressureTracker::isLiveOut(Value value) const {
  return llvm::is_contained(dag.getLiveOuts(), value);
}

bool VPTORegPressureTracker::resultNeedsLiveness(Value value) const {
  if (isLiveOut(value))
    return true;
  return llvm::any_of(value.getUsers(),
                      [&](Operation *user) { return dag.lookup(user); });
}

void VPTORegPressureTracker::initializeTop() {
  for (Value liveIn : dag.getLiveIns()) {
    if (liveValues.insert(liveIn).second)
      addValuePressure(liveIn, 1, current);
  }
  for (const std::unique_ptr<VPTOSUnit> &unit : dag.getUnits()) {
    for (Value operand : unit->getOperation()->getOperands())
      ++remainingUses[operand];
  }
}

void VPTORegPressureTracker::initializeBottom() {
  for (Value liveOut : dag.getLiveOuts()) {
    if (liveValues.insert(liveOut).second)
      addValuePressure(liveOut, 1, current);
  }
}

VPTORegPressureEvaluation
VPTORegPressureTracker::evaluateTop(const VPTOSUnit &unit) const {
  VPTORegPressureEvaluation evaluation;
  evaluation.delta.assign(current.size(), 0);

  DenseMap<Value, unsigned> candidateUses;
  for (Value operand : unit.getOperation()->getOperands())
    ++candidateUses[operand];
  for (const auto &entry : candidateUses) {
    Value value = entry.first;
    if (liveValues.contains(value) && !isLiveOut(value) &&
        remainingUses.lookup(value) == entry.second)
      addValuePressure(value, -1, evaluation.delta);
  }
  for (Value result : unit.getOperation()->getResults()) {
    if (!liveValues.contains(result) && resultNeedsLiveness(result))
      addValuePressure(result, 1, evaluation.delta);
  }
  updateSummary(evaluation);
  return evaluation;
}

VPTORegPressureEvaluation
VPTORegPressureTracker::evaluateBottom(const VPTOSUnit &unit) const {
  VPTORegPressureEvaluation evaluation;
  evaluation.delta.assign(current.size(), 0);
  for (Value result : unit.getOperation()->getResults()) {
    if (liveValues.contains(result))
      addValuePressure(result, -1, evaluation.delta);
  }
  DenseSet<Value> uniqueOperands;
  for (Value operand : unit.getOperation()->getOperands()) {
    if (uniqueOperands.insert(operand).second && !liveValues.contains(operand))
      addValuePressure(operand, 1, evaluation.delta);
  }
  updateSummary(evaluation);
  return evaluation;
}

void VPTORegPressureTracker::updateSummary(
    VPTORegPressureEvaluation &evaluation) const {
  evaluation.projected.resize(current.size());
  evaluation.projectedExcess.assign(current.size(), 0);
  for (auto [index, pressureSet] : llvm::enumerate(model.getPressureSets())) {
    evaluation.projected[index] = current[index] + evaluation.delta[index];
    if (pressureSet.limit)
      evaluation.projectedExcess[index] =
          std::max<int64_t>(0, evaluation.projected[index] -
                                   static_cast<int64_t>(*pressureSet.limit));
    evaluation.weightedDelta +=
        static_cast<double>(evaluation.delta[index]) * pressureSet.weight;
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
      auto found = remainingUses.find(operand);
      if (found == remainingUses.end() || found->second == 0)
        return failure();
      if (--found->second == 0 && !isLiveOut(operand))
        liveValues.erase(operand);
    }
    for (Value result : unit.getOperation()->getResults()) {
      if (resultNeedsLiveness(result))
        liveValues.insert(result);
    }
    return success();
  }

  for (Value result : unit.getOperation()->getResults())
    liveValues.erase(result);
  for (Value operand : unit.getOperation()->getOperands())
    liveValues.insert(operand);
  return success();
}

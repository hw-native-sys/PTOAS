// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOSchedResourceTracker.cpp - VPTO resource tracking -------------===//

#include "PTO/Transforms/VPTOScheduler/VPTOSchedResourceTracker.h"

#include "llvm/Support/raw_ostream.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::pto;

static unsigned getTimelineValue(ArrayRef<unsigned> timeline, unsigned cycle) {
  return cycle < timeline.size() ? timeline[cycle] : 0;
}

const VPTOSchedResource *
VPTOResourceTracker::findResource(VPTOSchedResourceID id) const {
  for (const VPTOSchedResource &resource : model.getResources()) {
    if (resource.id == id)
      return &resource;
  }
  return nullptr;
}

unsigned VPTOResourceTracker::getIssueOccupancy(unsigned cycle) const {
  return getTimelineValue(issueOccupancy, cycle);
}

unsigned VPTOResourceTracker::getResourceOccupancy(VPTOSchedResourceID resource,
                                                   unsigned cycle) const {
  auto found = resourceOccupancy.find(resource);
  return found == resourceOccupancy.end()
             ? 0
             : getTimelineValue(found->second, cycle);
}

bool VPTOResourceTracker::canReserve(const VPTOSchedParameters &parameters,
                                     unsigned cycle,
                                     std::string &reason) const {
  unsigned issueWidth = model.getMachineModel().issueWidth;
  if (parameters.microOps > issueWidth) {
    reason = "sched class exceeds machine issue width";
    return false;
  }
  if (getIssueOccupancy(cycle) + parameters.microOps > issueWidth) {
    return false;
  }

  for (const VPTOSchedResourceUse &use : parameters.resources) {
    const VPTOSchedResource *resource = findResource(use.resource);
    if (!resource) {
      reason = "sched class references an unknown resource";
      return false;
    }
    if (use.units > resource->units) {
      reason = "sched class requires more resource units than available";
      return false;
    }
    for (unsigned offset = 0; offset < use.duration; ++offset) {
      unsigned reservationCycle = cycle + use.acquireAt + offset;
      if (getResourceOccupancy(use.resource, reservationCycle) + use.units >
          resource->units)
        return false;
    }
  }
  return true;
}

VPTOResourceEvaluation
VPTOResourceTracker::evaluate(const VPTOSUnit &unit,
                              unsigned requestedCycle) const {
  VPTOResourceEvaluation evaluation;
  evaluation.earliestCycle = requestedCycle;
  VPTOSchedParameters parameters =
      model.getSchedParameters(unit.getOperation());
  std::string reason;

  // This upper bound protects analyze mode from malformed model data. Normal
  // resource conflicts resolve within the finite reservation timeline.
  constexpr unsigned kMaxResourceSearchCycles = 1U << 20;
  for (unsigned attempt = 0; attempt < kMaxResourceSearchCycles; ++attempt) {
    unsigned cycle = requestedCycle + attempt;
    reason.clear();
    if (canReserve(parameters, cycle, reason)) {
      evaluation.earliestCycle = cycle;
      evaluation.issueSlot = getIssueOccupancy(cycle);
      evaluation.stallCycles = attempt;
      return evaluation;
    }
    if (!reason.empty()) {
      evaluation.legal = false;
      evaluation.reason = std::move(reason);
      return evaluation;
    }
  }
  evaluation.legal = false;
  evaluation.reason = "resource search budget exceeded";
  return evaluation;
}

void VPTOResourceTracker::reserve(const VPTOSchedParameters &parameters,
                                  unsigned cycle) {
  if (issueOccupancy.size() <= cycle)
    issueOccupancy.resize(cycle + 1, 0);
  issueOccupancy[cycle] += parameters.microOps;

  for (const VPTOSchedResourceUse &use : parameters.resources) {
    SmallVector<unsigned> &timeline = resourceOccupancy[use.resource];
    unsigned endCycle = cycle + use.acquireAt + use.duration;
    if (timeline.size() < endCycle)
      timeline.resize(endCycle, 0);
    for (unsigned offset = 0; offset < use.duration; ++offset)
      timeline[cycle + use.acquireAt + offset] += use.units;
  }
}

LogicalResult VPTOResourceTracker::commit(const VPTOSUnit &unit,
                                          unsigned cycle) {
  VPTOResourceEvaluation evaluation = evaluate(unit, cycle);
  if (!evaluation.legal || evaluation.earliestCycle != cycle)
    return failure();
  reserve(model.getSchedParameters(unit.getOperation()), cycle);
  return success();
}

VPTOHazardResult VPTONullHazardRecognizer::check(const VPTOSUnit &unit,
                                                 VPTOSchedDirection direction,
                                                 unsigned cycle) const {
  (void)unit;
  (void)direction;
  return {/*legal=*/true, cycle, {}};
}

void VPTONullHazardRecognizer::commit(const VPTOSUnit &unit,
                                      VPTOSchedDirection direction,
                                      unsigned cycle) {
  (void)unit;
  (void)direction;
  (void)cycle;
}

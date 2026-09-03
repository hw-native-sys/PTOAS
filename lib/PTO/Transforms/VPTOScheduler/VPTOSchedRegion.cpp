// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOSchedRegion.cpp - VPTO scheduling regions ---------------------===//

#include "PTO/Transforms/VPTOScheduler/VPTOSchedRegion.h"

#include "PTO/IR/PTO.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::pto;

namespace {

static void addOperationUses(Operation &op, llvm::SetVector<Value> &live) {
  live.insert(op.getOperands().begin(), op.getOperands().end());
  for (Region &nestedRegion : op.getRegions()) {
    visitUsedValuesDefinedAbove(
        nestedRegion, nestedRegion,
        [&](OpOperand *operand) { live.insert(operand->get()); });
  }
}

static void addEnclosingLoopCaptures(Block &block,
                                     llvm::SetVector<Value> &live) {
  for (Operation *ancestor = block.getParentOp(); ancestor;
       ancestor = ancestor->getParentOp()) {
    auto loop = dyn_cast<LoopLikeOpInterface>(ancestor);
    if (!loop) {
      continue;
    }
    for (Region *loopRegion : loop.getLoopRegions()) {
      llvm::SetVector<Value> captures;
      getUsedValuesDefinedAbove(*loopRegion, *loopRegion, captures);
      live.insert(captures.begin(), captures.end());
    }
  }
}

static void populateLiveThroughs(Block &block,
                                 MutableArrayRef<VPTOSchedRegion> regions,
                                 const Liveness *liveness) {
  if (regions.empty()) {
    return;
  }

  DenseMap<Operation *, unsigned> regionStarts;
  DenseMap<Operation *, unsigned> regionEnds;
  for (auto [index, region] : llvm::enumerate(regions)) {
    regionStarts.try_emplace(region.operations.front(), index);
    regionEnds.try_emplace(region.operations.back(), index);
  }

  llvm::SetVector<Value> live;
  if (liveness) {
    if (const LivenessBlockInfo *blockInfo = liveness->getLiveness(&block)) {
      live.insert(blockInfo->out().begin(), blockInfo->out().end());
    }
  }
  addEnclosingLoopCaptures(block, live);

  SmallVector<DenseSet<Value>> liveAfter(regions.size());
  for (Operation &op : llvm::reverse(block)) {
    if (auto found = regionEnds.find(&op); found != regionEnds.end()) {
      liveAfter[found->second].insert(live.begin(), live.end());
    }

    for (Value result : op.getResults()) {
      live.remove(result);
    }
    addOperationUses(op, live);

    if (auto found = regionStarts.find(&op); found != regionStarts.end()) {
      VPTOSchedRegion &region = regions[found->second];
      for (Value value : live) {
        if (liveAfter[found->second].contains(value)) {
          region.liveThroughs.push_back(value);
        }
      }
    }
  }
}

} // namespace

static unsigned getClassIndex(VPTOSchedulingClass schedulingClass) {
  return static_cast<unsigned>(schedulingClass);
}

void VPTOSchedulingCoverage::record(Operation *op,
                                    const VPTOSchedulingSemantics &semantics) {
  VPTOSchedulingClass schedulingClass = semantics.schedulingClass;
  ++classCounts[getClassIndex(schedulingClass)];
  if (op && schedulingClass == VPTOSchedulingClass::Unsupported)
    ++unsupportedOps[op->getName().getStringRef()];
  if (schedulingClass == VPTOSchedulingClass::SchedulingBoundary)
    ++boundaryReasons[getVPTOSchedulingBoundaryReason(op)];
  if (op && schedulingClass == VPTOSchedulingClass::SchedulingBoundary &&
      !semantics.classificationKnown)
    ++unclassifiedOps[op->getName().getStringRef()];
}

unsigned
VPTOSchedulingCoverage::getCount(VPTOSchedulingClass schedulingClass) const {
  return classCounts[getClassIndex(schedulingClass)];
}

unsigned VPTOSchedulingCoverage::getUnclassifiedCount() const {
  unsigned count = 0;
  for (const auto &entry : unclassifiedOps)
    count += entry.getValue();
  return count;
}

std::string mlir::pto::getVPTOSchedulingBoundaryReason(Operation *op) {
  if (!op)
    return "block-boundary";
  if (op->hasTrait<OpTrait::IsTerminator>())
    return "terminator";
  if (op->getNumRegions() != 0)
    return "contains-regions";

  VPTOSchedulingClass schedulingClass = classifyVPTOSchedulingOp(op);
  std::string reason;
  llvm::raw_string_ostream os(reason);
  os << stringifyVPTOSchedulingClass(schedulingClass) << ':'
     << op->getName().getStringRef();
  return reason;
}

SmallVector<VPTOSchedRegion> VPTOSchedRegionBuilder::build(Block &block) const {
  SmallVector<VPTOSchedRegion> regions;
  SmallVector<Operation *> current;
  Operation *precedingBoundary = nullptr;
  std::string precedingReason = "block-start";

  auto flush = [&](Operation *followingBoundary, StringRef followingReason) {
    bool hasSchedulable = llvm::any_of(current, [](Operation *op) {
      return classifyVPTOSchedulingOp(op) == VPTOSchedulingClass::Schedulable;
    });
    if (hasSchedulable) {
      VPTOSchedRegion &region = regions.emplace_back();
      region.block = &block;
      region.index = regions.size() - 1;
      region.operations = current;
      region.precedingBoundary = precedingBoundary;
      region.followingBoundary = followingBoundary;
      region.precedingBoundaryReason = precedingReason;
      region.followingBoundaryReason = followingReason.str();
    }
    current.clear();
  };

  for (Operation &operation : block) {
    Operation *op = &operation;
    VPTOSchedulingSemantics semantics = getVPTOSchedulingSemantics(op);
    VPTOSchedulingClass schedulingClass = semantics.schedulingClass;
    if (coverage)
      coverage->record(op, semantics);

    if (schedulingClass == VPTOSchedulingClass::SchedulingBoundary ||
        schedulingClass == VPTOSchedulingClass::Unsupported) {
      std::string reason = getVPTOSchedulingBoundaryReason(op);
      flush(op, reason);
      precedingBoundary = op;
      precedingReason = std::move(reason);
      continue;
    }
    current.push_back(op);
  }

  flush(nullptr, "block-end");
  populateLiveThroughs(block, regions, liveness);
  return regions;
}

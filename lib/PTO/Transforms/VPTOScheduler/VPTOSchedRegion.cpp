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

#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::pto;

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
  return regions;
}

// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOSchedulerPass.cpp - VPTO scheduler driver ---------------------===//

#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/VPTOScheduler/VPTORegPressureTracker.h"
#include "PTO/Transforms/VPTOScheduler/VPTOSchedDAGBuilder.h"
#include "PTO/Transforms/VPTOScheduler/VPTOSchedResourceTracker.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <functional>
#include <mutex>
#include <string>
#include <utility>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VPTOSCHEDULER
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

static void printPressureVector(llvm::raw_ostream &os, StringRef label,
                                ArrayRef<int64_t> values,
                                const VPTOSchedModel &model) {
  os << ' ' << label << "={";
  for (auto [index, pressureSet] : llvm::enumerate(model.getPressureSets())) {
    if (index)
      os << ',';
    os << pressureSet.name << ':' << values[index];
  }
  os << '}';
}

static void printRegionReport(llvm::raw_ostream &os,
                              const VPTOSchedRegion &region, VPTOSchedDAG &dag,
                              const VPTOSchedModel &model) {
  VPTOSchedBoundary topBoundary(dag, model, VPTOSchedDirection::Top);
  unsigned topReady = topBoundary.getAvailable().size();
  VPTOSchedBoundary bottomBoundary(dag, model, VPTOSchedDirection::Bottom);
  unsigned bottomReady = bottomBoundary.getAvailable().size();

  unsigned knownClasses = 0;
  unsigned unknownClasses = 0;
  for (const std::unique_ptr<VPTOSUnit> &unit : dag.getUnits()) {
    if (model.getSchedClass(unit->getOperation()).known)
      ++knownClasses;
    else
      ++unknownClasses;
  }

  os << "vpto-scheduler: region=" << region.index
     << " before=" << region.precedingBoundaryReason
     << " after=" << region.followingBoundaryReason
     << " nodes=" << dag.getUnits().size() << " edges=" << dag.getEdges().size()
     << " live-ins=" << dag.getLiveIns().size()
     << " live-outs=" << dag.getLiveOuts().size() << " top-ready=" << topReady
     << " bottom-ready=" << bottomReady << " known-classes=" << knownClasses
     << " unknown-classes=" << unknownClasses << '\n';

  for (const std::unique_ptr<VPTOSUnit> &unit : dag.getUnits()) {
    const VPTOSchedClass &schedClass =
        model.getSchedClass(unit->getOperation());
    os << "vpto-scheduler: node=" << unit->getId()
       << " original-index=" << unit->getOriginalIndex()
       << " op=" << unit->getOperation()->getName().getStringRef()
       << " semantic="
       << stringifyVPTOSchedulingClass(unit->getSchedulingClass())
       << " sched-class=" << schedClass.name
       << " known=" << (schedClass.known ? "true" : "false")
       << " depth=" << unit->getDepth() << " height=" << unit->getHeight()
       << '\n';
  }
  for (const std::unique_ptr<VPTOSchedEdge> &edge : dag.getEdges()) {
    os << "vpto-scheduler: edge=" << edge->getPredecessor()->getId() << "->"
       << edge->getSuccessor()->getId()
       << " kind=" << stringifyVPTOSchedEdgeKind(edge->getKind())
       << " strength=" << stringifyVPTOSchedEdgeStrength(edge->getStrength())
       << " latency=" << edge->getLatency() << " reason=" << edge->getReason()
       << '\n';
  }

  VPTOResourceTracker &resourceTracker = topBoundary.getResourceTracker();
  VPTORegPressureTracker &pressureTracker = topBoundary.getPressureTracker();
  VPTOHazardRecognizer &hazardRecognizer = topBoundary.getHazardRecognizer();
  unsigned requestedCycle = 0;
  unsigned lastTimelineCycle = 0;
  for (const std::unique_ptr<VPTOSUnit> &unit : dag.getUnits()) {
    VPTOResourceEvaluation resource =
        resourceTracker.evaluate(*unit, requestedCycle);
    VPTOHazardResult hazard = hazardRecognizer.check(
        *unit, VPTOSchedDirection::Top, resource.earliestCycle);
    VPTORegPressureEvaluation pressure = pressureTracker.evaluate(*unit);
    if (!resource.legal || !hazard.legal) {
      os << "vpto-scheduler: fallback=tracker-rejected node=" << unit->getId()
         << " reason=" << (!resource.legal ? resource.reason : hazard.reason)
         << '\n';
      break;
    }

    unsigned cycle = std::max(resource.earliestCycle, hazard.earliestCycle);
    if (failed(resourceTracker.commit(*unit, cycle)) ||
        failed(pressureTracker.commit(*unit))) {
      os << "vpto-scheduler: fallback=tracker-commit-failed node="
         << unit->getId() << '\n';
      break;
    }
    hazardRecognizer.commit(*unit, VPTOSchedDirection::Top, cycle);

    const VPTOSchedClass &schedClass =
        model.getSchedClass(unit->getOperation());
    os << "vpto-scheduler: issue node=" << unit->getId() << " cycle=" << cycle
       << " slot=" << resource.issueSlot << " stall=" << resource.stallCycles
       << " sched-class=" << schedClass.name;
    printPressureVector(os, "delta", pressure.delta, model);
    printPressureVector(os, "current", pressureTracker.getCurrent(), model);
    printPressureVector(os, "peak", pressureTracker.getPeak(), model);
    os << '\n';

    lastTimelineCycle = std::max(lastTimelineCycle, cycle);
    for (const VPTOSchedResourceUse &use : schedClass.resources)
      lastTimelineCycle =
          std::max(lastTimelineCycle,
                   cycle + use.acquireAt + std::max(1U, use.duration) - 1);
    requestedCycle = cycle;
  }

  for (unsigned cycle = 0; cycle <= lastTimelineCycle; ++cycle) {
    os << "vpto-scheduler: timeline cycle=" << cycle
       << " issue=" << resourceTracker.getIssueOccupancy(cycle);
    for (const VPTOSchedResource &resource : model.getResources())
      os << ' ' << resource.name << '='
         << resourceTracker.getResourceOccupancy(resource.id, cycle);
    os << '\n';
  }
}

static void printCoverage(llvm::raw_ostream &os,
                          const VPTOSchedulingCoverage &coverage) {
  os << "vpto-scheduler: coverage schedulable="
     << coverage.getCount(VPTOSchedulingClass::Schedulable)
     << " structural=" << coverage.getCount(VPTOSchedulingClass::Structural)
     << " boundary="
     << coverage.getCount(VPTOSchedulingClass::SchedulingBoundary)
     << " unsupported=" << coverage.getCount(VPTOSchedulingClass::Unsupported)
     << " unclassified=" << coverage.getUnclassifiedCount()
     << '\n';

  SmallVector<std::pair<std::string, unsigned>> boundaryReasons;
  for (const auto &entry : coverage.boundaryReasons)
    boundaryReasons.emplace_back(entry.getKey().str(), entry.getValue());
  llvm::sort(boundaryReasons);
  for (const auto &[reason, count] : boundaryReasons)
    os << "vpto-scheduler: boundary-reason=" << reason << " count=" << count
       << '\n';

  SmallVector<std::pair<std::string, unsigned>> unsupported;
  for (const auto &entry : coverage.unsupportedOps)
    unsupported.emplace_back(entry.getKey().str(), entry.getValue());
  llvm::sort(unsupported);
  for (const auto &[name, count] : unsupported)
    os << "vpto-scheduler: unsupported-op=" << name << " count=" << count
       << '\n';

  SmallVector<std::pair<std::string, unsigned>> unclassified;
  for (const auto &entry : coverage.unclassifiedOps)
    unclassified.emplace_back(entry.getKey().str(), entry.getValue());
  llvm::sort(unclassified);
  for (const auto &[name, count] : unclassified)
    os << "vpto-scheduler: unclassified-op=" << name << " count=" << count
       << '\n';
}

static void analyzeFunction(func::FuncOp func, llvm::raw_ostream &os,
                            const VPTOSchedModel &model, StringRef mode) {
  SmallVector<Operation *> vecScopes;
  func.walk([&](Operation *op) {
    if (isa<VecScopeOp, StrictVecScopeOp>(op))
      vecScopes.push_back(op);
  });
  if (vecScopes.empty())
    return;

  const VPTOSchedMachineModel &machine = model.getMachineModel();
  os << "vpto-scheduler: function=" << func.getSymName() << " mode=" << mode
     << " target=" << machine.target << " model=" << machine.version
     << " issue-width=" << machine.issueWidth << '\n';

  VPTOSchedulingCoverage coverage;
  unsigned blockIndex = 0;
  std::function<void(Region &)> analyzeRegion = [&](Region &parentRegion) {
    for (Block &block : parentRegion) {
      VPTOSchedRegionBuilder regionBuilder(&coverage);
      SmallVector<VPTOSchedRegion> regions = regionBuilder.build(block);
      os << "vpto-scheduler: block=" << blockIndex++
         << " regions=" << regions.size() << '\n';
      for (const VPTOSchedRegion &region : regions) {
        VPTOSchedDAGBuilder dagBuilder(&model);
        FailureOr<std::unique_ptr<VPTOSchedDAG>> dag = dagBuilder.build(region);
        if (failed(dag)) {
          os << "vpto-scheduler: region=" << region.index
             << " fallback=dag-cycle\n";
          continue;
        }
        printRegionReport(os, region, **dag, model);
      }
      for (Operation &op : block) {
        if (isa<VecScopeOp, StrictVecScopeOp>(op))
          continue;
        for (Region &nestedRegion : op.getRegions())
          analyzeRegion(nestedRegion);
      }
    }
  };
  for (Operation *vecScope : vecScopes)
    analyzeRegion(vecScope->getRegion(0));
  printCoverage(os, coverage);
}

static StringAttr findTargetArchitecture(ModuleOp module) {
  for (ModuleOp current = module; current;
       current = current->getParentOfType<ModuleOp>())
    if (auto target = current->getAttrOfType<StringAttr>("pto.target_arch"))
      return target;
  return {};
}

struct VPTOSchedulerPass
    : public pto::impl::VPTOSchedulerBase<VPTOSchedulerPass> {
  using Base::Base;

  void runOnOperation() override {
    if (mode == "off")
      return;
    if (mode != "analyze" && mode != "on") {
      getOperation().emitError("unknown VPTO scheduler mode '") << mode << "'";
      return signalPassFailure();
    }
    StringAttr target = findTargetArchitecture(getOperation());
    if (!target) {
      getOperation().emitError(
          "VPTO scheduler requires target architecture 'a5', but neither "
          "this module nor an enclosing module defines 'pto.target_arch'");
      return signalPassFailure();
    }
    if (target.getValue() != "a5") {
      getOperation().emitError("VPTO scheduler requires target architecture "
                               "'a5', but module targets '")
          << target.getValue() << "'";
      return signalPassFailure();
    }
    if (auto kernelKind = getOperation()->getAttrOfType<FunctionKernelKindAttr>(
            FunctionKernelKindAttr::name);
        kernelKind && kernelKind.getKernelKind() != FunctionKernelKind::Vector)
      return;

    VPTOGenericA5SchedModel model;
    std::string report;
    llvm::raw_string_ostream os(report);
    getOperation().walk(
        [&](func::FuncOp func) { analyzeFunction(func, os, model, mode); });
    os.flush();

    // Nested module pass adaptors may execute sibling kernel modules in
    // parallel. Keep each module report intact even in that configuration.
    static std::mutex reportMutex;
    std::lock_guard<std::mutex> lock(reportMutex);
    llvm::errs() << report;
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::pto::createVPTOSchedulerPass(const VPTOSchedulerOptions &options) {
  return std::make_unique<VPTOSchedulerPass>(options);
}

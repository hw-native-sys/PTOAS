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
#include "PTO/Transforms/VPTOScheduler/VPTOScheduler.h"

#include "mlir/Analysis/Liveness.h"
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

static bool consumePressureReportWork(Operation *op,
                                      VPTOSchedulingBudget &budget) {
  if (!budget.consume(op->getNumOperands())) {
    return false;
  }
  if (!budget.consume(op->getNumResults())) {
    return false;
  }
  return budget.consume();
}

static void printOriginalOrderPressureReport(llvm::raw_ostream &os,
                                             const VPTOSchedDAG &dag,
                                             const VPTOSchedModel &model,
                                             VPTOSchedulingBudget &budget) {
  VPTORegPressureTracker tracker(model, dag, VPTOSchedDirection::Top);
  for (const std::unique_ptr<VPTOSUnit> &unit : dag.getUnits()) {
    Operation *op = unit->getOperation();
    if (!consumePressureReportWork(op, budget)) {
      os << "vpto-scheduler: fallback=pressure-report-budget node="
         << unit->getId() << " limit=" << budget.getLimit() << '\n';
      return;
    }
    VPTORegPressureEvaluation pressure = tracker.evaluate(*unit);
    if (failed(tracker.commit(*unit))) {
      os << "vpto-scheduler: fallback=pressure-tracker-commit-failed node="
         << unit->getId() << '\n';
      return;
    }
    os << "vpto-scheduler: pressure node=" << unit->getId();
    printPressureVector(os, "delta", pressure.delta, model);
    printPressureVector(os, "current", tracker.getCurrent(), model);
    printPressureVector(os, "peak", tracker.getPeak(), model);
    os << '\n';
  }
}

static void printRegionReport(llvm::raw_ostream &os,
                              const VPTOSchedRegion &region,
                              const VPTOSchedDAG &dag,
                              const VPTOSchedModel &model,
                              VPTOSchedulingBudget &budget) {
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
     << " live-outs=" << dag.getLiveOuts().size()
     << " known-classes=" << knownClasses
     << " unknown-classes=" << unknownClasses << '\n';

  for (const std::unique_ptr<VPTOSUnit> &unit : dag.getUnits()) {
    Operation *op = unit->getOperation();
    const VPTOSchedClass &schedClass =
        model.getSchedClass(op);
    VPTOSchedParameters parameters = model.getSchedParameters(op);
    os << "vpto-scheduler: node=" << unit->getId()
       << " original-index=" << unit->getOriginalIndex()
       << " op=" << op->getName().getStringRef()
       << " semantic="
       << stringifyVPTOSchedulingClass(unit->getSchedulingClass())
       << " sched-class=" << schedClass.name
       << " known=" << (schedClass.known ? "true" : "false")
       << " micro-ops=" << parameters.microOps
       << " write-latency=" << parameters.writeLatency
       << " resource-uses=" << parameters.resources.size()
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
  printOriginalOrderPressureReport(os, dag, model, budget);
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

static void printScheduleSummary(llvm::raw_ostream &os, unsigned blockIndex,
                                 const VPTOSchedRegion &region,
                                 const VPTOScheduleResult &result,
                                 const VPTOSchedModel &model,
                                 uint64_t workUnits) {
  os << "vpto-scheduler: schedule-result block=" << blockIndex
     << " region=" << region.index << " nodes=" << result.entries.size()
     << " work-units=" << workUnits
     << " pressure-idles="
     << llvm::count_if(result.entries, [](const VPTOScheduleEntry &entry) {
          return entry.pressureDrivenIdle;
        });
  printPressureVector(os, "peak", result.peakPressure, model);
  os << '\n';
}

static void printScheduleEntries(llvm::raw_ostream &os,
                                 const VPTOScheduleResult &result) {
  for (auto [position, entry] : llvm::enumerate(result.entries)) {
    os << "vpto-scheduler: result-position=" << position
       << " node=" << entry.unit->getId()
       << " original-index=" << entry.unit->getOriginalIndex()
       << " cycle=" << entry.issueCycle
       << " direction=" << stringifyVPTOSchedDirection(entry.direction)
       << " reason=" << entry.reason
       << " op=" << entry.unit->getOperation()->getName().getStringRef()
       << " pressure-idle=" << (entry.pressureDrivenIdle ? "true" : "false")
       << " recovery-idle=" << (entry.recoveryDrivenIdle ? "true" : "false")
       << '\n';
  }
}

static void printClosureDiagnostic(llvm::raw_ostream &os,
                                   const VPTOScheduleDiagnostic &diagnostic,
                                   const VPTOSchedModel &model) {
  if (!diagnostic.closurePressureSet) {
    os << " closure-set=none";
    return;
  }
  os << " closure-set=" << *diagnostic.closurePressureSet
     << " closure-bundle={";
  llvm::interleaveComma(diagnostic.closureBundleOriginalIndices, os);
  os << "} closure-target=" << diagnostic.closureTargetOriginalIndex.value_or(0)
     << " closure-size=" << diagnostic.closureGroupSize
     << " closure-steps=" << diagnostic.closureSteps
     << " closure-support=" << diagnostic.closureSupportPressure
     << " closure-effective-end=" << diagnostic.closureEffectiveEnd
     << " closure-net-relief=" << diagnostic.closureNetRelief;
  printPressureVector(os, "closure-peak", diagnostic.closurePeak, model);
  printPressureVector(os, "closure-end", diagnostic.closureEnd, model);
}

static void printCandidateDiagnostic(llvm::raw_ostream &os,
                                     const VPTOScheduleDiagnostic &diagnostic,
                                     const VPTOSchedModel &model) {
  os << " selected-cp=" << diagnostic.selectedCriticalPath
     << " selected-closure="
     << (diagnostic.selectedAdvancesClosure ? "true" : "false");
  printPressureVector(os, "selected-projected",
                      diagnostic.selectedProjectedPressure, model);
  printPressureVector(os, "selected-released",
                      diagnostic.selectedReleasedPressure, model);
  if (!diagnostic.safeAlternativeOriginalIndex) {
    os << " safe-alt=none";
    return;
  }
  os << " safe-alt=" << *diagnostic.safeAlternativeOriginalIndex
     << " safe-alt-cp=" << diagnostic.safeAlternativeCriticalPath
     << " safe-alt-opens-frontier="
     << (diagnostic.safeAlternativeOpensPressureFrontier ? "true" : "false");
  printPressureVector(os, "safe-alt-projected",
                      diagnostic.safeAlternativeProjectedPressure, model);
  printPressureVector(os, "safe-alt-released",
                      diagnostic.safeAlternativeReleasedPressure, model);
}

static void printScheduleDiagnostics(llvm::raw_ostream &os,
                                     const VPTOScheduleResult &result,
                                     const VPTOSchedModel &model) {
  for (auto [position, diagnosticValue] :
       llvm::enumerate(result.diagnostics)) {
    if (!diagnosticValue) {
      continue;
    }
    const VPTOScheduleDiagnostic &diagnostic = *diagnosticValue;
    os << "vpto-scheduler: decision-detail position=" << position
       << " candidates=" << diagnostic.candidateCount;
    printPressureVector(os, "current", diagnostic.currentPressure, model);
    printClosureDiagnostic(os, diagnostic, model);
    printCandidateDiagnostic(os, diagnostic, model);
    os << '\n';
  }
}

static void printScheduleResult(llvm::raw_ostream &os, unsigned blockIndex,
                                const VPTOSchedRegion &region,
                                const VPTOScheduleResult &result,
                                const VPTOSchedModel &model,
                                uint64_t workUnits) {
  printScheduleSummary(os, blockIndex, region, result, model, workUnits);
  printScheduleEntries(os, result);
  printScheduleDiagnostics(os, result, model);
}

static void emitRegionFailure(func::FuncOp func, unsigned blockIndex,
                              const VPTOSchedRegion &region,
                              const VPTOScheduleFailure &failure) {
  InFlightDiagnostic diagnostic =
      failure.kind == VPTOScheduleFailureKind::Budget ||
              failure.kind == VPTOScheduleFailureKind::InvalidModel
          ? func.emitRemark()
          : func.emitWarning();
  diagnostic << "VPTO scheduler skipped block " << blockIndex << " region "
             << region.index << ": "
             << stringifyVPTOScheduleFailureKind(failure.kind);
  if (!failure.name.empty()) {
    diagnostic << " " << failure.name << " actual=" << failure.actual
               << " limit=" << failure.limit;
  }
  if (!failure.detail.empty()) {
    diagnostic << " (" << failure.detail << ")";
  }
}

static void printRegionFailure(llvm::raw_ostream &os, unsigned blockIndex,
                               const VPTOSchedRegion &region,
                               const VPTOScheduleFailure &failure) {
  os << "vpto-scheduler: schedule-failure block=" << blockIndex
     << " region=" << region.index
     << " kind=" << stringifyVPTOScheduleFailureKind(failure.kind);
  if (!failure.name.empty()) {
    os << " name=" << failure.name << " actual=" << failure.actual
       << " limit=" << failure.limit;
  }
  if (!failure.detail.empty()) {
    os << " detail=" << failure.detail;
  }
  os << '\n';
}

static bool reportUnknownClasses(func::FuncOp func, unsigned blockIndex,
                                 const VPTOSchedRegion &region,
                                 const VPTOSchedDAG &dag,
                                 const VPTOSchedModel &model) {
  bool foundUnknown = false;
  for (const std::unique_ptr<VPTOSUnit> &unit : dag.getUnits()) {
    if (model.getSchedClass(unit->getOperation()).known) {
      continue;
    }
    foundUnknown = true;
    unit->getOperation()->emitRemark()
        << "VPTO scheduler skipped block " << blockIndex << " region "
        << region.index
        << " because original-index=" << unit->getOriginalIndex()
        << " op=" << unit->getOperation()->getName().getStringRef()
        << " has an unknown scheduling class";
  }
  return foundUnknown;
}

static void scheduleRegion(func::FuncOp func, llvm::raw_ostream &os,
                           unsigned blockIndex, const VPTOSchedRegion &region,
                           VPTOSchedDAG &dag, const VPTOSchedModel &model,
                           const VPTOSchedulerLimits &limits,
                           VPTOSchedulingBudget &budget, bool trace) {
  if (reportUnknownClasses(func, blockIndex, region, dag, model)) {
    return;
  }

  VPTOScheduleFailure failure;
  VPTOScheduler scheduler(model, dag, limits, budget, trace);
  FailureOr<VPTOScheduleResult> result = scheduler.schedule(failure);
  if (failed(result)) {
    if (trace) {
      printRegionFailure(os, blockIndex, region, failure);
    }
    emitRegionFailure(func, blockIndex, region, failure);
    return;
  }
  if (failed(verifyVPTOScheduleResult(dag, *result, budget, failure))) {
    if (trace) {
      printRegionFailure(os, blockIndex, region, failure);
    }
    emitRegionFailure(func, blockIndex, region, failure);
    return;
  }
  if (failed(replayVPTOScheduleResult(model, dag, *result, budget, failure))) {
    if (trace) {
      printRegionFailure(os, blockIndex, region, failure);
    }
    emitRegionFailure(func, blockIndex, region, failure);
    return;
  }
  if (failed(applyVPTOScheduleResult(dag, *result, budget, failure))) {
    if (trace) {
      printRegionFailure(os, blockIndex, region, failure);
    }
    emitRegionFailure(func, blockIndex, region, failure);
    return;
  }
  if (trace) {
    printScheduleResult(os, blockIndex, region, *result, model,
                        budget.getUsed());
  }
}

class FunctionSchedulerRunner {
public:
  FunctionSchedulerRunner(func::FuncOp func, llvm::raw_ostream &os,
                          const VPTOSchedModel &model, StringRef mode,
                          bool trace)
      : func(func), os(os), model(model), mode(mode), trace(trace),
        liveness(func) {}

  void run(ArrayRef<Operation *> vecScopes) {
    for (Operation *vecScope : vecScopes) {
      processRegion(vecScope->getRegion(0));
    }
  }

  const VPTOSchedulingCoverage &getCoverage() const { return coverage; }

private:
  void processRegion(Region &parentRegion);
  void processBlock(Block &block);
  void processSchedulingRegion(unsigned currentBlockIndex,
                               const VPTOSchedRegion &region);

  func::FuncOp func;
  llvm::raw_ostream &os;
  const VPTOSchedModel &model;
  StringRef mode;
  bool trace;
  VPTOSchedulingCoverage coverage;
  Liveness liveness;
  unsigned blockIndex = 0;
};

void FunctionSchedulerRunner::processSchedulingRegion(
    unsigned currentBlockIndex, const VPTOSchedRegion &region) {
  VPTOSchedulerLimits limits;
  VPTOSchedulingBudget schedulingBudget(limits.maxWorkUnits);
  VPTOScheduleFailure failure;
  VPTOSchedDAGBuilder dagBuilder(&model, limits, schedulingBudget);
  FailureOr<std::unique_ptr<VPTOSchedDAG>> dag =
      dagBuilder.build(region, failure);
  if (failed(dag)) {
    emitRegionFailure(func, currentBlockIndex, region, failure);
    return;
  }
  if (mode == "analyze" || trace) {
    VPTOSchedulingBudget reportBudget(limits.maxWorkUnits);
    printRegionReport(os, region, **dag, model, reportBudget);
  }
  if (mode != "analyze") {
    scheduleRegion(func, os, currentBlockIndex, region, **dag, model, limits,
                   schedulingBudget, trace);
  }
}

void FunctionSchedulerRunner::processBlock(Block &block) {
  unsigned currentBlockIndex = blockIndex++;
  VPTOSchedRegionBuilder regionBuilder(&coverage, &liveness);
  SmallVector<VPTOSchedRegion> regions = regionBuilder.build(block);
  if (mode == "analyze" || trace) {
    os << "vpto-scheduler: block=" << currentBlockIndex
       << " regions=" << regions.size() << '\n';
  }
  for (const VPTOSchedRegion &region : regions) {
    processSchedulingRegion(currentBlockIndex, region);
  }
  for (Operation &op : block) {
    if (isa<VecScopeOp, StrictVecScopeOp>(op)) {
      continue;
    }
    for (Region &nestedRegion : op.getRegions()) {
      processRegion(nestedRegion);
    }
  }
}

void FunctionSchedulerRunner::processRegion(Region &parentRegion) {
  for (Block &block : parentRegion) {
    processBlock(block);
  }
}

static void runFunction(func::FuncOp func, llvm::raw_ostream &os,
                        const VPTOSchedModel &model, StringRef mode,
                        bool trace) {
  SmallVector<Operation *> vecScopes;
  func.walk([&](Operation *op) {
    if (isa<VecScopeOp, StrictVecScopeOp>(op)) {
      vecScopes.push_back(op);
    }
  });
  if (vecScopes.empty()) {
    return;
  }

  const VPTOSchedMachineModel &machine = model.getMachineModel();
  if (mode == "analyze" || trace) {
    os << "vpto-scheduler: function=" << func.getSymName() << " mode=" << mode
       << " target=" << machine.target << " model=" << machine.version << '\n';
  }
  FunctionSchedulerRunner runner(func, os, model, mode, trace);
  runner.run(vecScopes);
  if (mode == "analyze" || trace) {
    printCoverage(os, runner.getCoverage());
  }
}

static StringAttr findTargetArchitecture(ModuleOp module) {
  for (ModuleOp current = module; current;
       current = current->getParentOfType<ModuleOp>()) {
    if (auto target = current->getAttrOfType<StringAttr>("pto.target_arch")) {
      return target;
    }
  }
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
        [&](func::FuncOp func) { runFunction(func, os, model, mode, trace); });
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

// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- pto-vpto-scheduler-tracker-test.cpp -------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/VPTOScheduler/VPTORegPressureTracker.h"
#include "PTO/Transforms/VPTOScheduler/VPTOSchedDAGBuilder.h"
#include "PTO/Transforms/VPTOScheduler/VPTOSchedResourceTracker.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::pto;

namespace {
enum ResourceID : unsigned {
  MultiUnitResource,
  SharedResource,
  DelayedResource
};
enum PressureSetID : unsigned { VectorPressure, PredicatePressure };

class TrackerTestModel final : public VPTOSchedModel {
public:
  TrackerTestModel() {
    machine.target = "test";
    machine.version = "tracker-test-v1";
    machine.issueWidth = 2;

    resources = {
        {MultiUnitResource, "multi-unit", 2, 0, {}},
        {SharedResource, "shared", 1, 0, {}},
        {DelayedResource, "delayed", 1, 0, {}},
    };
    pressureSets = {
        {VectorPressure, "vector", std::nullopt, 1.0, 1.0},
        {PredicatePressure, "predicate", 2, 2.0, 4.0},
    };
    schedClasses = {
        {0, "default", true, 1, 1, {}, {}},
        {1, "two-units", true, 1, 1, {{MultiUnitResource, 0, 1, 2}}, {}},
        {2, "shared-a", true, 1, 1, {{SharedResource, 0, 1, 1}}, {}},
        {3, "shared-b", true, 1, 1, {{SharedResource, 0, 1, 1}}, {}},
        {4, "single", true, 1, 1, {}, {}},
        {5, "delayed", true, 1, 1, {{DelayedResource, 1, 2, 1}}, {}},
        {6, "too-wide", true, 3, 1, {}, {}},
        {7, "unknown", false, 1, 1, {}, {}},
    };
  }

  const VPTOSchedMachineModel &getMachineModel() const override {
    return machine;
  }
  ArrayRef<VPTOSchedResource> getResources() const override {
    return resources;
  }
  ArrayRef<VPTORegPressureSet> getPressureSets() const override {
    return pressureSets;
  }
  const VPTOSchedClass &getSchedClass(Operation *op) const override {
    StringRef name = "default";
    if (auto attr = op->getAttrOfType<StringAttr>("test_class"))
      name = attr.getValue();
    for (const VPTOSchedClass &schedClass : schedClasses)
      if (schedClass.name == name)
        return schedClass;
    return schedClasses.back();
  }
  SmallVector<VPTORegPressureContribution>
  getPressure(Value value) const override {
    if (!value)
      return {};
    if (isa<VRegType>(value.getType()))
      return {{VectorPressure, 1}};
    if (isa<MaskType>(value.getType()))
      return {{PredicatePressure, 1}};
    return {};
  }

private:
  VPTOSchedMachineModel machine;
  SmallVector<VPTOSchedResource> resources;
  SmallVector<VPTORegPressureSet> pressureSets;
  SmallVector<VPTOSchedClass> schedClasses;
};

static bool check(bool condition, const Twine &message) {
  if (condition)
    return true;
  llvm::errs() << "FAIL: " << message << '\n';
  return false;
}

static OwningOpRef<ModuleOp> parseModule(MLIRContext &context,
                                         StringRef source) {
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(source, &context);
  if (!module || failed(verify(*module)))
    return {};
  return module;
}

static VecScopeOp findVecScope(ModuleOp module) {
  VecScopeOp result;
  module.walk([&](VecScopeOp scope) {
    if (!result)
      result = scope;
  });
  return result;
}

static bool testResourceTracker(MLIRContext &context,
                                const TrackerTestModel &model) {
  static constexpr StringLiteral source = R"mlir(
module attributes {pto.target_arch = "a5"} {
  func.func @resources() {
    pto.vecscope {
      pto.sprclr "AR" {test_class = "two-units"}
      pto.sprclr "AR" {test_class = "two-units"}
      pto.sprclr "AR" {test_class = "shared-a"}
      pto.sprclr "AR" {test_class = "shared-b"}
      pto.sprclr "AR" {test_class = "single"}
      pto.sprclr "AR" {test_class = "single"}
      pto.sprclr "AR" {test_class = "single"}
      pto.sprclr "AR" {test_class = "delayed"}
      pto.sprclr "AR" {test_class = "delayed"}
      pto.sprclr "AR" {test_class = "too-wide"}
    }
    return
  }
}
)mlir";

  OwningOpRef<ModuleOp> module = parseModule(context, source);
  if (!check(static_cast<bool>(module), "cannot parse resource fixture"))
    return false;
  VecScopeOp scope = findVecScope(*module);
  if (!check(static_cast<bool>(scope), "resource fixture has no vecscope"))
    return false;

  VPTOSchedRegion region;
  for (Operation &op : scope.getBody().front())
    region.operations.push_back(&op);
  VPTOSchedDAG dag(region);
  ArrayRef<std::unique_ptr<VPTOSUnit>> units = dag.getUnits();
  if (!check(units.size() == 10, "resource fixture unit count"))
    return false;

  VPTOResourceTracker multiUnit(model);
  bool ok = check(succeeded(multiUnit.commit(*units[0], 0)),
                  "commit two-unit reservation");
  VPTOResourceEvaluation secondMulti = multiUnit.evaluate(*units[1], 0);
  ok &= check(secondMulti.legal && secondMulti.earliestCycle == 1 &&
                  secondMulti.stallCycles == 1,
              "multi-unit capacity must stall one cycle");
  ok &= check(multiUnit.getResourceOccupancy(MultiUnitResource, 0) == 2,
              "multi-unit occupancy");
  if (!ok)
    return false;
  llvm::outs() << "resource multi-unit: pass\n";

  VPTOResourceTracker shared(model);
  ok = check(succeeded(shared.commit(*units[2], 0)),
             "commit first shared-resource user");
  VPTOResourceEvaluation secondShared = shared.evaluate(*units[3], 0);
  ok &= check(secondShared.legal && secondShared.earliestCycle == 1,
              "sched classes sharing a resource must conflict");
  if (!ok)
    return false;
  llvm::outs() << "resource shared: pass\n";

  VPTOResourceTracker issue(model);
  ok = check(succeeded(issue.commit(*units[4], 0)), "commit first issue slot");
  VPTOResourceEvaluation secondIssue = issue.evaluate(*units[5], 0);
  ok &= check(secondIssue.legal && secondIssue.earliestCycle == 0 &&
                  secondIssue.issueSlot == 1,
              "second issue slot");
  ok &=
      check(succeeded(issue.commit(*units[5], 0)), "commit second issue slot");
  VPTOResourceEvaluation thirdIssue = issue.evaluate(*units[6], 0);
  ok &= check(thirdIssue.legal && thirdIssue.earliestCycle == 1 &&
                  issue.getIssueOccupancy(0) == 2,
              "issue width must defer third micro-op");
  VPTOResourceEvaluation tooWide = issue.evaluate(*units[9], 0);
  ok &= check(!tooWide.legal &&
                  tooWide.reason == "sched class exceeds machine issue width",
              "sched class wider than machine must be rejected");
  if (!ok)
    return false;
  llvm::outs() << "resource issue-width: pass\n";

  VPTOResourceTracker reservation(model);
  ok = check(succeeded(reservation.commit(*units[7], 0)),
             "commit cross-cycle reservation");
  VPTOResourceEvaluation secondReservation = reservation.evaluate(*units[8], 0);
  ok &= check(secondReservation.legal && secondReservation.earliestCycle == 2 &&
                  secondReservation.stallCycles == 2,
              "cross-cycle reservation must defer overlapping use");
  ok &= check(reservation.getResourceOccupancy(DelayedResource, 0) == 0 &&
                  reservation.getResourceOccupancy(DelayedResource, 1) == 1 &&
                  reservation.getResourceOccupancy(DelayedResource, 2) == 1,
              "acquireAt and duration occupancy");
  if (!ok)
    return false;
  llvm::outs() << "resource reservation: pass\n";
  return true;
}

struct PressureFixture {
  OwningOpRef<ModuleOp> module;
  std::unique_ptr<VPTOSchedDAG> dag;
};

static FailureOr<PressureFixture>
buildPressureFixture(MLIRContext &context, const TrackerTestModel &model) {
  static constexpr StringLiteral source = R"mlir(
module attributes {pto.target_arch = "a5"} {
  func.func @pressure(%lhs: !pto.vreg<64xf32>, %rhs: !pto.vreg<64xf32>,
                      %active: !pto.mask<b32>, %dst: !pto.ptr<f32, ub>) {
    %c0 = arith.constant 0 : index
    pto.vecscope {
      %p0 = pto.vcmp %lhs, %rhs, %active, "lt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.mask<b32>
      %p1 = pto.vcmp %lhs, %rhs, %active, "gt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.mask<b32>
      %s0 = pto.vsel %lhs, %rhs, %p0 : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
      %s1 = pto.vsel %lhs, %rhs, %p1 : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
      %keep = pto.vsel %lhs, %rhs, %active : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
      pto.vsts %s0, %dst[%c0], %active : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
      pto.vsts %s1, %dst[%c0], %active : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
      pto.vsts %keep, %dst[%c0], %active : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
    }
    return
  }
}
)mlir";

  PressureFixture fixture;
  fixture.module = parseModule(context, source);
  if (!fixture.module)
    return failure();
  VecScopeOp scope = findVecScope(*fixture.module);
  if (!scope)
    return failure();

  VPTOSchedRegion region;
  for (Operation &op : scope.getBody().front()) {
    if (isa<VstsOp>(op))
      break;
    region.operations.push_back(&op);
  }
  VPTOSchedDAGBuilder builder(&model);
  FailureOr<std::unique_ptr<VPTOSchedDAG>> dag = builder.build(region);
  if (failed(dag))
    return failure();
  fixture.dag = std::move(*dag);
  return fixture;
}

static bool commitOrder(VPTORegPressureTracker &tracker,
                        ArrayRef<std::unique_ptr<VPTOSUnit>> units,
                        ArrayRef<unsigned> order) {
  for (unsigned index : order)
    if (failed(tracker.commit(*units[index])))
      return false;
  return true;
}

static bool testPressureTracker(MLIRContext &context,
                                const TrackerTestModel &model) {
  FailureOr<PressureFixture> fixture = buildPressureFixture(context, model);
  if (!check(succeeded(fixture), "cannot build pressure fixture"))
    return false;
  VPTOSchedDAG &dag = *fixture->dag;
  ArrayRef<std::unique_ptr<VPTOSUnit>> units = dag.getUnits();
  if (!check(units.size() == 5, "pressure fixture unit count"))
    return false;

  bool ok = check(dag.getLiveIns().size() == 3, "deduplicated live-ins") &&
            check(dag.getLiveOuts().size() == 3, "live-outs");
  VPTORegPressureTracker grouped(model, dag, VPTOSchedDirection::Top);
  ok &= check(grouped.getCurrent()[VectorPressure] == 2 &&
                  grouped.getCurrent()[PredicatePressure] == 1,
              "top tracker initializes live-in pressure");
  ok &= check(commitOrder(grouped, units, {0, 1}), "commit grouped compares");
  VPTORegPressureEvaluation lastUse = grouped.evaluate(*units[2]);
  ok &= check(lastUse.delta[PredicatePressure] == -1,
              "last predicate use pressure delta");
  Value p0 = units[0]->getOperation()->getResult(0);
  ok &= check(succeeded(grouped.commit(*units[2])) && !grouped.isLive(p0),
              "last use removes predicate liveness");
  ok &= check(commitOrder(grouped, units, {3, 4}), "finish grouped order");
  ok &= check(grouped.getPeak()[PredicatePressure] == 3,
              "grouped compare/select predicate peak");
  if (!ok)
    return false;
  llvm::outs() << "pressure live-in-out-last-use: pass\n";

  VPTORegPressureTracker interleaved(model, dag, VPTOSchedDirection::Top);
  ok = check(commitOrder(interleaved, units, {0, 2, 1, 3, 4}),
             "commit interleaved compare/select order");
  ok &= check(interleaved.getPeak()[PredicatePressure] == 2,
              "interleaved compare/select predicate peak");
  if (!ok)
    return false;
  llvm::outs() << "pressure compare-select: grouped=3 interleaved=2\n";

  VPTORegPressureTracker bottom(model, dag, VPTOSchedDirection::Bottom);
  ok = check(bottom.getCurrent()[VectorPressure] == 3 &&
                 bottom.getCurrent()[PredicatePressure] == 0,
             "bottom tracker initializes live-out pressure");
  VPTORegPressureEvaluation bottomFirst = bottom.evaluate(*units[4]);
  ok &= check(bottomFirst.delta[VectorPressure] == 1 &&
                  bottomFirst.delta[PredicatePressure] == 1,
              "bottom candidate delta");
  ok &=
      check(commitOrder(bottom, units, {4, 3, 2, 1, 0}), "commit bottom order");
  ok &= check(bottom.getCurrent()[VectorPressure] == 2 &&
                  bottom.getCurrent()[PredicatePressure] == 1,
              "bottom tracker finishes at live-in pressure");
  if (!ok)
    return false;
  llvm::outs() << "pressure bottom: pass\n";
  return true;
}
} // namespace

int main() {
  DialectRegistry registry;
  registry.insert<PTODialect, func::FuncDialect, arith::ArithDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  TrackerTestModel model;
  if (!testResourceTracker(context, model) ||
      !testPressureTracker(context, model))
    return 1;
  return 0;
}

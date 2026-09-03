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
#include "PTO/Transforms/VPTOScheduler/VPTOScheduler.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <limits>

using namespace mlir;
using namespace mlir::pto;

namespace {
enum ResourceID : unsigned {
  MultiUnitResource,
  SharedResource,
  DelayedResource
};
enum PressureSetID : unsigned {
  VectorPressure,
  PredicatePressure,
  UnboundedPressure
};

class TrackerTestModel final : public VPTOSchedModel {
public:
  explicit TrackerTestModel(bool trackUnboundedPressure = false,
                            unsigned predicateLimit = 2)
      : trackUnboundedPressure(trackUnboundedPressure) {
    machine.target = "test";
    machine.version = "tracker-test-v1";
    machine.issueWidth = 2;

    resources = {
        {MultiUnitResource, "multi-unit", 2, 0, {}},
        {SharedResource, "shared", 1, 0, {}},
        {DelayedResource, "delayed", 1, 0, {}},
    };
    pressureSets = {
        {VectorPressure, "vector", 8, 1, 1},
        {PredicatePressure, "predicate", predicateLimit, 2, 4},
    };
    if (trackUnboundedPressure) {
      pressureSets.push_back(
          {UnboundedPressure, "unbounded", std::nullopt, 4, 1});
    }
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
    if (!value) {
      return {};
    }
    if (isa<VRegType>(value.getType())) {
      return {{VectorPressure, 1}};
    }
    if (isa<MaskType>(value.getType())) {
      return {{PredicatePressure, 1}};
    }
    if (trackUnboundedPressure && value.getType().isIndex()) {
      return {{UnboundedPressure, 1}};
    }
    return {};
  }

private:
  bool trackUnboundedPressure;
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

static VPTOSchedRegion
buildPressureRegion(VecScopeOp scope,
                    std::optional<unsigned> maxOperations = std::nullopt,
                    bool stopBeforeStore = false) {
  VPTOSchedRegion region;
  region.block = &scope.getBody().front();
  for (Operation &op : scope.getBody().front()) {
    bool reachedLimit =
        maxOperations && region.operations.size() == *maxOperations;
    bool reachedBoundary = (stopBeforeStore && isa<VstsOp>(op)) || reachedLimit;
    if (reachedBoundary) {
      region.followingBoundary = &op;
      break;
    }
    if (op.hasTrait<OpTrait::IsTerminator>()) {
      break;
    }
    region.operations.push_back(&op);
  }
  return region;
}

static FailureOr<PressureFixture> buildPressureFixtureFromSource(
    MLIRContext &context, const VPTOSchedModel &model, StringRef source,
    std::optional<unsigned> maxOperations = std::nullopt,
    bool stopBeforeStore = false) {
  PressureFixture fixture;
  fixture.module = parseModule(context, source);
  if (!fixture.module) {
    return failure();
  }
  VecScopeOp scope = findVecScope(*fixture.module);
  if (!scope) {
    return failure();
  }
  VPTOSchedRegion region =
      buildPressureRegion(scope, maxOperations, stopBeforeStore);
  VPTOSchedDAGBuilder builder(&model);
  FailureOr<std::unique_ptr<VPTOSchedDAG>> dag = builder.build(region);
  if (failed(dag)) {
    return failure();
  }
  fixture.dag = std::move(*dag);
  return fixture;
}

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
  return buildPressureFixtureFromSource(context, model, source, std::nullopt,
                                        true);
}

static FailureOr<PressureFixture>
buildFanoutFixture(MLIRContext &context, const TrackerTestModel &model) {
  static constexpr StringLiteral source = R"mlir(
module attributes {pto.target_arch = "a5"} {
  func.func @fanout(%lhs: !pto.vreg<64xf32>,
                    %rhs: !pto.vreg<64xf32>,
                    %active: !pto.mask<b32>) {
    pto.vecscope {
      %root = pto.vadd %lhs, %rhs, %active : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
      %a = pto.vadd %root, %rhs, %active : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
      %b = pto.vadd %root, %rhs, %active : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
      %c = pto.vadd %root, %rhs, %active : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
    }
    return
  }
}
)mlir";
  return buildPressureFixtureFromSource(context, model, source);
}

static FailureOr<PressureFixture>
buildUnboundedPressureFixture(MLIRContext &context,
                              const TrackerTestModel &model) {
  static constexpr StringLiteral source = R"mlir(
module attributes {pto.target_arch = "a5"} {
  func.func @unbounded_pressure(%lhs: index, %rhs: index, %drop: index) {
    pto.vecscope {
      %grow = arith.addi %lhs, %rhs : index
      %discard = arith.index_cast %drop : index to i64
      %use0 = arith.addi %grow, %lhs : index
      %use1 = arith.addi %use0, %rhs : index
    }
    return
  }
}
)mlir";
  return buildPressureFixtureFromSource(context, model, source, 2);
}

static bool commitOrder(VPTORegPressureTracker &tracker,
                        ArrayRef<std::unique_ptr<VPTOSUnit>> units,
                        ArrayRef<unsigned> order) {
  for (unsigned index : order)
    if (failed(tracker.commit(*units[index])))
      return false;
  return true;
}

static FailureOr<PressureFixture>
buildCompletePressureFixture(MLIRContext &context, const VPTOSchedModel &model,
                             StringRef source) {
  return buildPressureFixtureFromSource(context, model, source);
}

static bool testTopPressureTracker(const TrackerTestModel &model,
                                   VPTOSchedDAG &dag) {
  ArrayRef<std::unique_ptr<VPTOSUnit>> units = dag.getUnits();
  Value active = units[0]->getOperation()->getOperand(2);
  bool ok = check(dag.getLiveIns().size() == 3, "deduplicated live-ins") &&
            check(dag.getLiveOuts().size() == 4, "live-through live-out") &&
            check(llvm::is_contained(dag.getLiveOuts(), active),
                  "live-in used after the region must remain live-out");
  VPTORegPressureTracker tracker(model, dag, VPTOSchedDirection::Top);
  ok &= check(tracker.getCurrent()[VectorPressure] == 2 &&
                  tracker.getCurrent()[PredicatePressure] == 1,
              "top tracker initializes live-in pressure");
  ok &= check(commitOrder(tracker, units, {0, 1}), "commit grouped compares");
  VPTORegPressureEvaluation lastUse = tracker.evaluate(*units[2]);
  ok &= check(lastUse.delta[PredicatePressure] == -1,
              "last predicate use pressure delta");
  Value p0 = units[0]->getOperation()->getResult(0);
  ok &= check(succeeded(tracker.commit(*units[2])) && !tracker.isLive(p0),
              "last use removes predicate liveness");
  ok &= check(commitOrder(tracker, units, {3, 4}), "finish grouped order");
  ok &= check(tracker.getPeak()[PredicatePressure] == 3,
              "grouped compare/select predicate peak");
  ok &= check(tracker.isLive(active) &&
                  tracker.getCurrent()[PredicatePressure] == 1,
              "top tracker must retain a live-through predicate");
  return ok;
}

static bool testBottomPressureTracker(const TrackerTestModel &model,
                                      VPTOSchedDAG &dag) {
  ArrayRef<std::unique_ptr<VPTOSUnit>> units = dag.getUnits();
  VPTORegPressureTracker tracker(model, dag, VPTOSchedDirection::Bottom);
  bool ok = check(tracker.getCurrent()[VectorPressure] == 3 &&
                      tracker.getCurrent()[PredicatePressure] == 1,
                  "bottom tracker initializes live-out pressure");
  VPTORegPressureEvaluation first = tracker.evaluate(*units[4]);
  ok &= check(first.delta[VectorPressure] == 1 &&
                  first.delta[PredicatePressure] == 0,
              "bottom candidate delta");
  ok &= check(commitOrder(tracker, units, {4, 3, 2, 1, 0}),
              "commit bottom order");
  ok &= check(tracker.getCurrent()[VectorPressure] == 2 &&
                  tracker.getCurrent()[PredicatePressure] == 1,
              "bottom tracker finishes at live-in pressure");
  return ok;
}

static bool testPressureTracker(MLIRContext &context,
                                const TrackerTestModel &model) {
  FailureOr<PressureFixture> fixture = buildPressureFixture(context, model);
  if (!check(succeeded(fixture), "cannot build pressure fixture")) {
    return false;
  }
  VPTOSchedDAG &dag = *fixture->dag;
  ArrayRef<std::unique_ptr<VPTOSUnit>> units = dag.getUnits();
  bool hasExpectedUnitCount =
      check(units.size() == 5, "pressure fixture unit count");
  if (!hasExpectedUnitCount) {
    return false;
  }
  if (!testTopPressureTracker(model, dag)) {
    return false;
  }
  llvm::outs() << "pressure live-in-out-last-use: pass\n";

  VPTORegPressureTracker interleaved(model, dag, VPTOSchedDirection::Top);
  bool ok = check(commitOrder(interleaved, units, {0, 2, 1, 3, 4}),
                  "commit interleaved compare/select order");
  ok &= check(interleaved.getPeak()[PredicatePressure] == 2,
              "interleaved compare/select predicate peak");
  if (!ok) {
    return false;
  }
  llvm::outs() << "pressure compare-select: grouped=3 interleaved=2\n";
  if (!testBottomPressureTracker(model, dag)) {
    return false;
  }
  llvm::outs() << "pressure bottom: pass\n";
  return true;
}

static bool testTopBitcastPressure(const VPTOGenericA5SchedModel &model,
                                   VPTOSchedDAG &dag) {
  ArrayRef<std::unique_ptr<VPTOSUnit>> units = dag.getUnits();
  VPTORegPressureTracker tracker(model, dag, VPTOSchedDirection::Top);
  bool ok = check(tracker.getCurrent()[VectorPressure] == 1 &&
                      tracker.getCurrent()[PredicatePressure] == 2,
                  "bitcast top pressure initializes physical live-ins");
  VPTORegPressureEvaluation vectorAlias = tracker.evaluate(*units[0]);
  ok &= check(vectorAlias.delta[VectorPressure] == 0 &&
                  vectorAlias.introduced[VectorPressure] == 0 &&
                  vectorAlias.released[VectorPressure] == 0,
              "vbitcast must preserve one physical live range");
  ok &= check(commitOrder(tracker, units, {0, 1, 2}),
              "commit vector bitcast chain");
  VPTORegPressureEvaluation predicateAlias = tracker.evaluate(*units[3]);
  ok &= check(predicateAlias.delta[PredicatePressure] == 0 &&
                  predicateAlias.introduced[PredicatePressure] == 0 &&
                  predicateAlias.released[PredicatePressure] == 0,
              "pbitcast must preserve one physical live range");
  ok &= check(commitOrder(tracker, units, {3, 4}),
              "commit predicate bitcast chain");
  return ok && check(tracker.getCurrent()[VectorPressure] == 0 &&
                         tracker.getCurrent()[PredicatePressure] == 0 &&
                         tracker.getPeak()[VectorPressure] == 1 &&
                         tracker.getPeak()[PredicatePressure] == 2,
                     "bitcast top pressure must not create view registers");
}

static bool testBottomBitcastPressure(const VPTOGenericA5SchedModel &model,
                                      VPTOSchedDAG &dag) {
  ArrayRef<std::unique_ptr<VPTOSUnit>> units = dag.getUnits();
  VPTORegPressureTracker tracker(model, dag, VPTOSchedDirection::Bottom);
  bool ok =
      check(commitOrder(tracker, units, {4}), "start bottom predicate chain");
  VPTORegPressureEvaluation predicateAlias = tracker.evaluate(*units[3]);
  ok &= check(predicateAlias.delta[PredicatePressure] == 0 &&
                  predicateAlias.introduced[PredicatePressure] == 0 &&
                  predicateAlias.released[PredicatePressure] == 0,
              "bottom pbitcast must preserve one physical live range");
  ok &= check(commitOrder(tracker, units, {3, 2, 1}),
              "advance bottom bitcast chains");
  VPTORegPressureEvaluation vectorAlias = tracker.evaluate(*units[0]);
  ok &= check(vectorAlias.delta[VectorPressure] == 0 &&
                  vectorAlias.introduced[VectorPressure] == 0 &&
                  vectorAlias.released[VectorPressure] == 0,
              "bottom vbitcast must preserve one physical live range");
  ok &= check(commitOrder(tracker, units, {0}), "finish bottom bitcast chain");
  return ok && check(tracker.getCurrent()[VectorPressure] == 1 &&
                         tracker.getCurrent()[PredicatePressure] == 2,
                     "bitcast bottom pressure must recover physical live-ins");
}

static bool testBitcastPressureAliasing(
    MLIRContext &context, const VPTOGenericA5SchedModel &model) {
  static constexpr StringLiteral source = R"mlir(
module attributes {pto.target_arch = "a5"} {
  func.func @bitcast_pressure(%input: !pto.vreg<64xf32>,
                              %minus_one: i32,
                              %active: !pto.mask<b32>,
                              %predicate: !pto.mask<b8>) {
    pto.vecscope {
      %bits = pto.vbitcast %input : !pto.vreg<64xf32> -> !pto.vreg<64xi32>
      %previous = pto.vadds %bits, %minus_one, %active : !pto.vreg<64xi32>, i32, !pto.mask<b32> -> !pto.vreg<64xi32>
      %view = pto.vbitcast %previous : !pto.vreg<64xi32> -> !pto.vreg<64xf32>
      %b16 = pto.pbitcast %predicate : !pto.mask<b8> -> !pto.mask<b16>
      %b32 = pto.pbitcast %b16 : !pto.mask<b16> -> !pto.mask<b32>
    }
    return
  }
}
)mlir";
  FailureOr<PressureFixture> fixture =
      buildCompletePressureFixture(context, model, source);
  bool invalidFixture =
      !check(succeeded(fixture), "cannot build bitcast-pressure fixture") ||
      !check(fixture->dag->getUnits().size() == 5,
             "bitcast-pressure fixture unit count");
  if (invalidFixture) {
    return false;
  }
  bool ok = testTopBitcastPressure(model, *fixture->dag) &&
            testBottomBitcastPressure(model, *fixture->dag);
  if (!ok) {
    return false;
  }
  llvm::outs() << "pressure bitcast-alias: pass\n";
  return true;
}

static constexpr StringLiteral kBitcastPressureCacheSource = R"mlir(
module attributes {pto.target_arch = "a5"} {
  func.func @bitcast_pressure_cache(%input: !pto.vreg<64xf32>,
                                    %float_one: f32,
                                    %int_one: i32,
                                    %direct_active: !pto.mask<b32>,
                                    %alias_active: !pto.mask<b32>) {
    pto.vecscope {
      %bits = pto.vbitcast %input : !pto.vreg<64xf32> -> !pto.vreg<64xi32>
      %direct = pto.vadds %input, %float_one, %direct_active
          : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.vreg<64xf32>
      %alias = pto.vadds %bits, %int_one, %alias_active
          : !pto.vreg<64xi32>, i32, !pto.mask<b32> -> !pto.vreg<64xi32>
    }
    return
  }
}
)mlir";

static bool
testBitcastPressureCacheInvalidation(MLIRContext &context,
                                     const VPTOGenericA5SchedModel &model) {
  FailureOr<PressureFixture> fixture =
      buildCompletePressureFixture(context, model, kBitcastPressureCacheSource);
  if (!check(succeeded(fixture),
             "cannot build bitcast-pressure-cache fixture") ||
      !check(fixture->dag->getUnits().size() == 3,
             "bitcast-pressure-cache fixture unit count")) {
    return false;
  }
  ArrayRef<std::unique_ptr<VPTOSUnit>> units = fixture->dag->getUnits();
  VPTOSchedBoundary boundary(*fixture->dag, model, VPTOSchedDirection::Top);
  VPTOSchedulingBudget budget(128);
  std::string detail;
  bool ok = check(succeeded(boundary.commit(*units[0], 0, budget, detail)),
                  "commit pressure-alias producer");
  VPTORegPressureEvaluation beforeDirect =
      boundary.evaluatePressure(*units[2]);
  ok &= check(beforeDirect.delta[VectorPressure] == 0,
              "cached alias pressure must retain another use");
  ok &= check(succeeded(boundary.commit(*units[1], 0, budget, detail)),
              "commit direct pressure-alias user");
  VPTORegPressureEvaluation afterDirect =
      boundary.evaluatePressure(*units[2]);
  ok &= check(afterDirect.delta[VectorPressure] == -1 &&
                  afterDirect.released[VectorPressure] == 1,
              "alias pressure cache must invalidate after a shared use");
  if (!ok) {
    return false;
  }
  llvm::outs() << "pressure bitcast-cache-invalidation: pass\n";
  return true;
}

static VPTOSchedCandidate makeStrategyCandidate(
    const VPTOSchedModel &model, VPTOSUnit &unit, unsigned criticalPath,
    unsigned originalIndex, ArrayRef<int64_t> current,
    int64_t predicateDelta, int64_t predicateReleased,
    int64_t predicateIntroduced) {
  VPTOSchedCandidate candidate;
  candidate.unit = &unit;
  candidate.criticalPath = criticalPath;
  candidate.originalIndex = originalIndex;
  candidate.pressure.delta = {0, predicateDelta};
  candidate.pressure.released = {0, predicateReleased};
  candidate.pressure.introduced = {0, predicateIntroduced};
  candidate.pressure.projected = {current[VectorPressure],
                                  current[PredicatePressure] + predicateDelta};
  candidate.pressure.projectedExcess = {0, 0};
  std::optional<unsigned> predicateLimit =
      model.getPressureSets()[PredicatePressure].limit;
  if (predicateLimit) {
    candidate.pressure.projectedExcess[PredicatePressure] =
        std::max<int64_t>(0, candidate.pressure.projected[PredicatePressure] -
                                static_cast<int64_t>(*predicateLimit));
  }
  candidate.lookaheadPeak = candidate.pressure.projected;
  candidate.lookaheadEnd = candidate.pressure.projected;
  candidate.lookaheadSteps = 1;
  candidate.opensPressureFrontier =
      predicateIntroduced > 0 && predicateReleased == 0;
  return candidate;
}

static bool checkStrategyDecision(const VPTOSchedStrategy &strategy,
                                  const VPTOScheduleContext &context,
                                  ArrayRef<VPTOSchedCandidate> candidates,
                                  VPTOSUnit *expectedUnit,
                                  StringRef expectedReason,
                                  const Twine &failureMessage,
                                  StringRef successMessage) {
  std::string detail;
  FailureOr<VPTOSchedDecision> decision =
      strategy.pickCandidate(context, candidates, detail);
  bool ok = check(succeeded(decision) && decision->unit == expectedUnit &&
                      decision->reason == expectedReason,
                  failureMessage);
  if (ok) {
    llvm::outs() << successMessage << '\n';
  }
  return ok;
}

static bool testNearLimitStrategy(const TrackerTestModel &model,
                                  VPTOSchedDAG &dag,
                                  ArrayRef<std::unique_ptr<VPTOSUnit>> units) {
  SmallVector<int64_t> pressure = {2, 3};
  VPTOSchedCandidate producer =
      makeStrategyCandidate(model, *units[0], 10, 0, pressure, 1, 0, 1);
  VPTOSchedCandidate consumer =
      makeStrategyCandidate(model, *units[2], 9, 2, pressure, -1, 1, 0);
  VPTOScheduleContext context{model, dag,      VPTOSchedDirection::Top,
                              0,     pressure, std::nullopt};
  return checkStrategyDecision(
      getDefaultVPTOSchedStrategy(), context, {producer, consumer},
      consumer.unit, "high-pressure-preserving",
      "near-limit strategy must close a live range within the critical-path "
      "window",
      "strategy near-limit closing: pass");
}

static bool
testLowPressureStrategy(const TrackerTestModel &model, VPTOSchedDAG &dag,
                        ArrayRef<std::unique_ptr<VPTOSUnit>> units) {
  SmallVector<int64_t> pressure = {2, 1};
  VPTOSchedCandidate producer =
      makeStrategyCandidate(model, *units[0], 10, 0, pressure, 1, 0, 1);
  VPTOSchedCandidate consumer =
      makeStrategyCandidate(model, *units[2], 9, 2, pressure, -1, 1, 0);
  VPTOScheduleContext context{model, dag,      VPTOSchedDirection::Top,
                              0,     pressure, std::nullopt};
  return checkStrategyDecision(
      getDefaultVPTOSchedStrategy(), context, {producer, consumer},
      producer.unit, "longer-critical-path",
      "low-pressure strategy must preserve critical-path priority",
      "strategy low-pressure critical path: pass");
}

static bool testAtLimitStrategy(const TrackerTestModel &model,
                                VPTOSchedDAG &dag,
                                ArrayRef<std::unique_ptr<VPTOSUnit>> units) {
  SmallVector<int64_t> pressure = {2, 4};
  VPTOSchedCandidate producer =
      makeStrategyCandidate(model, *units[0], 10, 0, pressure, 0, 0, 0);
  VPTOSchedCandidate consumer =
      makeStrategyCandidate(model, *units[2], 9, 2, pressure, -1, 1, 0);
  VPTOScheduleContext context{model, dag,      VPTOSchedDirection::Top,
                              0,     pressure, std::nullopt};
  return checkStrategyDecision(
      getDefaultVPTOSchedStrategy(), context, {producer, consumer},
      consumer.unit, "high-pressure-preserving",
      "critical-pressure strategy must prefer a live-range-closing candidate "
      "at the limit",
      "strategy no-producer critical path: pass");
}

static bool
testPressureOverridesLatency(const TrackerTestModel &model, VPTOSchedDAG &dag,
                             ArrayRef<std::unique_ptr<VPTOSUnit>> units) {
  SmallVector<int64_t> pressure = {2, 3};
  VPTOSchedCandidate producer =
      makeStrategyCandidate(model, *units[0], 10, 0, pressure, 1, 0, 1);
  VPTOSchedCandidate consumer =
      makeStrategyCandidate(model, *units[2], 8, 2, pressure, -1, 1, 0);
  VPTOScheduleContext context{model, dag,      VPTOSchedDirection::Top,
                              0,     pressure, std::nullopt};
  return checkStrategyDecision(
      getDefaultVPTOSchedStrategy(), context, {producer, consumer},
      consumer.unit, "high-pressure-preserving",
      "critical-pressure strategy must override latency when bounded "
      "lookahead crosses a pressure-risk band",
      "strategy urgent critical path: pass");
}

static bool
testClosureAndTieBreakStrategy(const TrackerTestModel &model, VPTOSchedDAG &dag,
                               ArrayRef<std::unique_ptr<VPTOSUnit>> units) {
  const VPTOSchedStrategy &strategy = getDefaultVPTOSchedStrategy();
  SmallVector<int64_t> nearLimit = {2, 3};
  VPTOSchedCandidate outside =
      makeStrategyCandidate(model, *units[0], 10, 0, nearLimit, 0, 0, 0);
  VPTOSchedCandidate inside =
      makeStrategyCandidate(model, *units[1], 10, 1, nearLimit, 0, 0, 0);
  inside.advancesPressureClosure = true;
  VPTOScheduleContext closureContext{model, dag,       VPTOSchedDirection::Top,
                                     0,     nearLimit, PredicatePressure};
  if (!checkStrategyDecision(
          strategy, closureContext, {outside, inside}, inside.unit,
          "advance-pressure-closure",
          "high-pressure strategy must advance the selected closure cone when "
          "immediate pressure risk is equal",
          "strategy closure-cone progress: pass")) {
    return false;
  }
  SmallVector<int64_t> low = {2, 1};
  VPTOSchedCandidate later =
      makeStrategyCandidate(model, *units[0], 10, 7, low, 1, 0, 1);
  VPTOSchedCandidate earlier =
      makeStrategyCandidate(model, *units[1], 10, 3, low, 1, 0, 1);
  VPTOScheduleContext lowContext{model, dag, VPTOSchedDirection::Top,
                                 0,     low, std::nullopt};
  return checkStrategyDecision(
      strategy, lowContext, {later, earlier}, earlier.unit,
      "deterministic-tie-break",
      "strategy must use original order as its deterministic tie-break",
      "strategy deterministic tie-break: pass");
}

static bool testPressureAwareStrategy(MLIRContext &context) {
  TrackerTestModel model(/*trackUnboundedPressure=*/false,
                         /*predicateLimit=*/4);
  FailureOr<PressureFixture> fixture = buildPressureFixture(context, model);
  bool invalidFixture =
      !check(succeeded(fixture), "cannot build strategy fixture") ||
      !check(fixture->dag->getUnits().size() >= 3,
             "strategy fixture unit count");
  if (invalidFixture) {
    return false;
  }
  ArrayRef<std::unique_ptr<VPTOSUnit>> units = fixture->dag->getUnits();
  return testNearLimitStrategy(model, *fixture->dag, units) &&
         testLowPressureStrategy(model, *fixture->dag, units) &&
         testAtLimitStrategy(model, *fixture->dag, units) &&
         testPressureOverridesLatency(model, *fixture->dag, units) &&
         testClosureAndTieBreakStrategy(model, *fixture->dag, units);
}

static bool testGenericA5PredicateLimit() {
  VPTOGenericA5SchedModel model;
  auto predicate = llvm::find_if(
      model.getPressureSets(),
      [](const VPTORegPressureSet &pressureSet) {
        return pressureSet.name == "predicate";
      });
  bool ok = check(predicate != model.getPressureSets().end() &&
                      predicate->limit && *predicate->limit == 7,
                  "generic A5 predicate pressure limit must remain 7");
  if (ok) {
    llvm::outs() << "model predicate-limit: 7\n";
  }
  return ok;
}

static bool testBoundary(MLIRContext &context, const TrackerTestModel &model) {
  FailureOr<PressureFixture> fixture = buildPressureFixture(context, model);
  if (!check(succeeded(fixture), "cannot build boundary fixture")) {
    return false;
  }
  VPTOSchedDAG &dag = *fixture->dag;
  ArrayRef<std::unique_ptr<VPTOSUnit>> units = dag.getUnits();
  VPTOSchedBoundary top(dag, model, VPTOSchedDirection::Top);
  VPTOSchedBoundary bottom(dag, model, VPTOSchedDirection::Bottom);
  VPTOSchedulingBudget commitBudget(128);

  bool ok = check(top.getAvailable().size() == 3,
                  "top boundary initial availability") &&
            check(bottom.getAvailable().size() == 3,
                  "bottom boundary initial availability");
  std::string detail;
  ok &= check(succeeded(top.commit(*units[0], 0, commitBudget, detail)),
              "top boundary commit");
  ok &= check(top.isScheduled(units[0].get()) &&
                  !bottom.isScheduled(units[0].get()),
              "boundaries must not share dependency state");
  const VPTOPendingUnit *nextPending = top.getNextPending();
  ok &= check(top.getPendingCount() == 1 && nextPending &&
                  nextPending->unit == units[2].get() &&
                  nextPending->readyCycle == 1,
              "boundary must defer a dependency by edge latency");

  VPTOSchedulingBudget exhaustedBudget(0);
  FailureOr<bool> exhaustedAdvance =
      top.advanceToNextPendingCycle(exhaustedBudget);
  ok &= check(failed(exhaustedAdvance) && top.getCurrentCycle() == 0 &&
                  top.getPendingCount() == 1,
              "boundary budget failure must not partially release pending");

  VPTOSchedulingBudget budget(8);
  FailureOr<bool> advanced = top.advanceToNextPendingCycle(budget);
  ok &= check(succeeded(advanced) && *advanced && top.getCurrentCycle() == 1 &&
                  llvm::is_contained(top.getAvailable(), units[2].get()),
              "boundary must release pending nodes at their ready cycle");
  if (ok) {
    llvm::outs() << "boundary independent-latency: pass\n";
  }
  return ok;
}

static bool
testBoundaryCommitBudget(VPTOSchedBoundary &boundary,
                         ArrayRef<std::unique_ptr<VPTOSUnit>> units) {
  SmallVector<int64_t> pressureBefore(
      boundary.getPressureTracker().getCurrent().begin(),
      boundary.getPressureTracker().getCurrent().end());
  VPTOSchedulingBudget exhaustedBudget(18);
  std::string detail;
  LogicalResult exhaustedCommit =
      boundary.commit(*units[0], 0, exhaustedBudget, detail);
  bool ok = check(failed(exhaustedCommit) && exhaustedBudget.hasExceeded(),
                  "fanout commit must stop at the work budget");
  ok &= check(boundary.isAvailable(units[0].get()) &&
                  !boundary.isScheduled(units[0].get()) &&
                  boundary.getPendingCount() == 0 &&
                  llvm::equal(pressureBefore,
                              boundary.getPressureTracker().getCurrent()),
              "budget failure must not partially commit boundary state");
  VPTOSchedulingBudget retryBudget(64);
  detail.clear();
  LogicalResult retryCommit =
      boundary.commit(*units[0], 0, retryBudget, detail);
  const VPTOPendingUnit *next = boundary.getNextPending();
  ok &= check(succeeded(retryCommit) && boundary.isScheduled(units[0].get()) &&
                  boundary.getPendingCount() == 3 && next &&
                  next->readyCycle == 1,
              "fanout retry must build an earliest-cycle pending bucket");
  return ok;
}

static bool testBoundaryReleaseBudget(VPTOSchedBoundary &boundary) {
  VPTOSchedulingBudget failureBudget(2);
  FailureOr<bool> failedAdvance =
      boundary.advanceToNextPendingCycle(failureBudget);
  bool ok = check(failed(failedAdvance) && failureBudget.hasExceeded() &&
                      boundary.getCurrentCycle() == 0 &&
                      boundary.getPendingCount() == 3,
                  "pending release budget failure must not mutate cycle "
                  "buckets");
  VPTOSchedulingBudget releaseBudget(64);
  FailureOr<bool> advanced = boundary.advanceToNextPendingCycle(releaseBudget);
  ok &= check(succeeded(advanced) && *advanced &&
                  boundary.getCurrentCycle() == 1 &&
                  boundary.getPendingCount() == 0 &&
                  boundary.getAvailable().size() == 3,
              "pending cycle bucket must release all earliest-cycle nodes");
  return ok;
}

static bool testBoundaryBudget(MLIRContext &context,
                               const TrackerTestModel &model) {
  FailureOr<PressureFixture> fixture = buildFanoutFixture(context, model);
  if (!check(succeeded(fixture), "cannot build boundary budget fixture")) {
    return false;
  }
  VPTOSchedDAG &dag = *fixture->dag;
  ArrayRef<std::unique_ptr<VPTOSUnit>> units = dag.getUnits();
  bool hasExpectedFanout =
      units.size() == 4 && units[0]->getSuccessors().size() == 3;
  if (!check(hasExpectedFanout, "boundary budget fixture fanout")) {
    return false;
  }
  VPTOSchedBoundary boundary(dag, model, VPTOSchedDirection::Top);
  bool ok = testBoundaryCommitBudget(boundary, units) &&
            testBoundaryReleaseBudget(boundary);
  if (ok) {
    llvm::outs() << "boundary fanout-budget-cycle-bucket: pass\n";
  }
  return ok;
}

static bool verifySchedulerResult(const TrackerTestModel &model,
                                  VPTOSchedDAG &dag,
                                  const VPTOScheduleResult &result,
                                  VPTOSchedulingBudget &budget,
                                  VPTOScheduleFailure &failure) {
  bool ok =
      check(!result.entries.empty() &&
                result.entries.front().direction == VPTOSchedDirection::Top &&
                !result.entries.front().reason.empty(),
            "scheduler result must preserve decision direction and "
            "reason");
  VPTOSchedulingBudget exhaustedBudget(0);
  VPTOScheduleFailure exhaustedFailure;
  ok &= check(failed(verifyVPTOScheduleResult(dag, result, exhaustedBudget,
                                              exhaustedFailure)) &&
                  exhaustedFailure.kind == VPTOScheduleFailureKind::Budget &&
                  exhaustedFailure.name == "work-units",
              "semantic verifier must honor the shared work-unit budget");
  ok &= check(succeeded(verifyVPTOScheduleResult(dag, result, budget, failure)),
              "scheduler result semantic verification");
  ok &= check(
      succeeded(replayVPTOScheduleResult(model, dag, result, budget, failure)),
      "scheduler result model replay");
  auto pressureIdle =
      llvm::find_if(result.entries, [](const VPTOScheduleEntry &entry) {
        return entry.pressureDrivenIdle;
      });
  ok &= check(pressureIdle != result.entries.end() &&
                  result.peakPressure[PredicatePressure] == 2,
              "scheduler must idle before exceeding predicate pressure");
  if (ok) {
    llvm::outs() << "scheduler verify-replay: pass\n";
  }
  return ok;
}

static bool testSchedulerReplayMetadata(const TrackerTestModel &model,
                                        VPTOSchedDAG &dag,
                                        const VPTOScheduleResult &result) {
  auto pressureIdle =
      llvm::find_if(result.entries, [](const VPTOScheduleEntry &entry) {
        return entry.pressureDrivenIdle;
      });
  VPTOScheduleResult missingIdle = result;
  size_t idlePosition =
      static_cast<size_t>(std::distance(result.entries.begin(), pressureIdle));
  missingIdle.entries[idlePosition].pressureDrivenIdle = false;
  VPTOSchedulingBudget missingBudget(128);
  VPTOScheduleFailure missingFailure;
  bool ok =
      check(failed(replayVPTOScheduleResult(model, dag, missingIdle,
                                            missingBudget, missingFailure)) &&
                missingFailure.kind == VPTOScheduleFailureKind::ModelReplay,
            "model replay must reject missing pressure-idle metadata");
  if (!ok) {
    return false;
  }
  llvm::outs() << "scheduler pressure-idle replay: pass\n";
  VPTOScheduleResult illegalIdle = result;
  illegalIdle.entries.front().pressureDrivenIdle = true;
  illegalIdle.entries.front().issueCycle = 1;
  VPTOSchedulingBudget illegalBudget(128);
  VPTOScheduleFailure illegalFailure;
  ok = check(failed(replayVPTOScheduleResult(model, dag, illegalIdle,
                                             illegalBudget, illegalFailure)) &&
                 illegalFailure.kind == VPTOScheduleFailureKind::ModelReplay,
             "model replay must reject pressure idle while a safe candidate "
             "is available");
  if (ok) {
    llvm::outs() << "scheduler illegal pressure-idle rejection: pass\n";
  }
  return ok;
}

static bool
testSchedulerApplyAndSemanticRejection(VPTOSchedDAG &dag,
                                       const VPTOScheduleResult &result) {
  VPTOSchedulingBudget exhaustedBudget(0);
  VPTOScheduleFailure applyFailure;
  bool ok = check(failed(applyVPTOScheduleResult(dag, result, exhaustedBudget,
                                                 applyFailure)) &&
                      applyFailure.kind == VPTOScheduleFailureKind::Budget &&
                      applyFailure.name == "work-units",
                  "apply must prepay node moves from the shared budget");
  VPTOSchedulingBudget unchangedBudget(128);
  VPTOScheduleFailure unchangedFailure;
  ok &= check(succeeded(verifyVPTOScheduleResult(dag, result, unchangedBudget,
                                                 unchangedFailure)),
              "apply budget failure must not partially reorder the region");
  if (!ok) {
    return false;
  }
  VPTOScheduleResult invalidResult = result;
  std::swap(invalidResult.entries.front(), invalidResult.entries.back());
  VPTOScheduleFailure invalidFailure;
  VPTOSchedulingBudget invalidBudget(128);
  ok = check(failed(verifyVPTOScheduleResult(dag, invalidResult, invalidBudget,
                                             invalidFailure)) &&
                 invalidFailure.kind ==
                     VPTOScheduleFailureKind::SemanticVerification,
             "semantic verifier must reject a dependency violation");
  if (ok) {
    llvm::outs() << "scheduler semantic rejection: pass\n";
  }
  return ok;
}

static bool testSchedulerWorkBudget(MLIRContext &context,
                                    const TrackerTestModel &model) {
  FailureOr<PressureFixture> fixture = buildPressureFixture(context, model);
  if (!check(succeeded(fixture), "cannot build budget fixture")) {
    return false;
  }
  VPTOSchedulerLimits limits;
  limits.maxWorkUnits = 1;
  VPTOSchedulingBudget budget(limits.maxWorkUnits);
  VPTOScheduleFailure failure;
  VPTOScheduler scheduler(model, *fixture->dag, limits, budget);
  bool ok = check(failed(scheduler.schedule(failure)) &&
                      failure.kind == VPTOScheduleFailureKind::Budget &&
                      failure.name == "work-units",
                  "scheduler must report the shared work-unit budget");
  if (ok) {
    llvm::outs() << "scheduler budget: pass\n";
  }
  return ok;
}

static bool testScheduler(MLIRContext &context, const TrackerTestModel &model) {
  FailureOr<PressureFixture> fixture = buildPressureFixture(context, model);
  if (!check(succeeded(fixture), "cannot build scheduler fixture")) {
    return false;
  }
  VPTOSchedulerLimits limits;
  limits.maxWorkUnits = 512;
  VPTOSchedulingBudget budget(limits.maxWorkUnits);
  VPTOScheduleFailure failure;
  VPTOScheduler scheduler(model, *fixture->dag, limits, budget);
  FailureOr<VPTOScheduleResult> result = scheduler.schedule(failure);
  if (!check(succeeded(result), "scheduler must produce a result")) {
    return false;
  }
  return verifySchedulerResult(model, *fixture->dag, *result, budget,
                               failure) &&
         testSchedulerReplayMetadata(model, *fixture->dag, *result) &&
         testSchedulerApplyAndSemanticRejection(*fixture->dag, *result) &&
         testSchedulerWorkBudget(context, model);
}

static bool testPressureNoPendingProgress(MLIRContext &context) {
  TrackerTestModel model(/*trackUnboundedPressure=*/false,
                         /*predicateLimit=*/0);
  FailureOr<PressureFixture> fixture = buildPressureFixture(context, model);
  if (!check(succeeded(fixture), "cannot build no-pending fixture")) {
    return false;
  }

  VPTOSchedulerLimits limits;
  VPTOSchedulingBudget budget(limits.maxWorkUnits);
  VPTOScheduleFailure failure;
  VPTOScheduler scheduler(model, *fixture->dag, limits, budget);
  FailureOr<VPTOScheduleResult> result = scheduler.schedule(failure);
  bool ok = check(succeeded(result) && !result->entries.empty() &&
                      result->entries.front().issueCycle == 0 &&
                      !result->entries.front().pressureDrivenIdle,
                  "all-over-limit scheduling must progress when no pending "
                  "event exists");
  if (ok) {
    llvm::outs() << "scheduler no-pending pressure progress: pass\n";
  }
  return ok;
}

static constexpr StringLiteral kPressureReliefSource = R"mlir(
module attributes {pto.target_arch = "a5"} {
  func.func @pressure_relief(
      %lhs: !pto.vreg<64xf32>, %rhs: !pto.vreg<64xf32>,
      %p0: !pto.mask<b32>, %p1: !pto.mask<b32>, %p2: !pto.mask<b32>) {
    pto.vecscope {
      pto.sprclr "AR"
      %relief = pto.vsel %lhs, %rhs, %p0 : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
      %delayed = pto.vsel %lhs, %rhs, %p1 : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
    }
    return
  }
}
)mlir";

static FailureOr<PressureFixture>
buildPressureReliefFixture(MLIRContext &context) {
  PressureFixture fixture;
  fixture.module = parseModule(context, kPressureReliefSource);
  if (!fixture.module) {
    return failure();
  }
  VecScopeOp scope = findVecScope(*fixture.module);
  func::FuncOp function =
      fixture.module->lookupSymbol<func::FuncOp>("pressure_relief");
  if (!scope || !function) {
    return failure();
  }
  VPTOSchedRegion region = buildPressureRegion(scope);
  fixture.dag = std::make_unique<VPTOSchedDAG>(region);
  ArrayRef<std::unique_ptr<VPTOSUnit>> units = fixture.dag->getUnits();
  bool invalidUnitCount = units.size() != 3;
  if (invalidUnitCount) {
    return failure();
  }
  for (unsigned argumentIndex = 2; argumentIndex != 5; ++argumentIndex) {
    fixture.dag->addLiveIn(function.getArgument(argumentIndex));
  }
  fixture.dag->addEdge(*units[0], *units[1], VPTOSchedEdgeKind::Artificial,
                       VPTOSchedEdgeStrength::Must, 0, "ready pressure relief");
  fixture.dag->addEdge(*units[0], *units[2], VPTOSchedEdgeKind::Artificial,
                       VPTOSchedEdgeStrength::Must, 10, "delayed alternative");
  if (failed(fixture.dag->computeCriticalPaths())) {
    return failure();
  }
  fixture.dag->resetDependencyCounts();
  return fixture;
}

static bool testPressureReliefDoesNotIdle(MLIRContext &context) {
  FailureOr<PressureFixture> fixture = buildPressureReliefFixture(context);
  if (!check(succeeded(fixture), "cannot build pressure-relief fixture")) {
    return false;
  }
  TrackerTestModel model(/*trackUnboundedPressure=*/false,
                         /*predicateLimit=*/1);
  VPTOSchedulerLimits limits;
  VPTOSchedulingBudget budget(limits.maxWorkUnits);
  VPTOScheduleFailure failure;
  VPTOScheduler scheduler(model, *fixture->dag, limits, budget);
  FailureOr<VPTOScheduleResult> result = scheduler.schedule(failure);
  bool ok = check(succeeded(result) && result->entries.size() == 3,
                  "pressure-relief scheduler result");
  ArrayRef<std::unique_ptr<VPTOSUnit>> units = fixture->dag->getUnits();
  ok &= check(result->entries[1].unit == units[1].get() &&
                  result->entries[1].issueCycle == 0 &&
                  !result->entries[1].pressureDrivenIdle,
              "pressure relief must run without waiting while still over "
              "the limit");
  if (ok) {
    llvm::outs() << "scheduler over-limit pressure relief: pass\n";
  }
  return ok;
}

static bool testUnboundedPressureScheduling(MLIRContext &context) {
  TrackerTestModel model(/*trackUnboundedPressure=*/true);
  FailureOr<PressureFixture> fixture =
      buildUnboundedPressureFixture(context, model);
  if (!check(succeeded(fixture), "cannot build unbounded-pressure fixture")) {
    return false;
  }

  VPTOSchedDAG &dag = *fixture->dag;
  ArrayRef<std::unique_ptr<VPTOSUnit>> units = dag.getUnits();
  bool hasExpectedUnitCount = units.size() == 2;
  if (!check(hasExpectedUnitCount, "unbounded-pressure fixture unit count")) {
    return false;
  }
  VPTORegPressureTracker tracker(model, dag, VPTOSchedDirection::Top);
  VPTORegPressureEvaluation grow = tracker.evaluate(*units[0]);
  VPTORegPressureEvaluation drop = tracker.evaluate(*units[1]);
  bool ok = check(grow.delta[UnboundedPressure] == 1 &&
                      drop.delta[UnboundedPressure] == -1 &&
                      grow.projectedExcess[UnboundedPressure] == 0 &&
                      drop.projectedExcess[UnboundedPressure] == 0,
                  "unbounded pressure must track delta without excess");

  VPTOSchedulerLimits limits;
  VPTOSchedulingBudget budget(limits.maxWorkUnits);
  VPTOScheduleFailure failure;
  VPTOScheduler scheduler(model, dag, limits, budget);
  FailureOr<VPTOScheduleResult> result = scheduler.schedule(failure);
  ok &= check(succeeded(result) && result->entries.size() == 2 &&
                  result->entries.front().unit->getOriginalIndex() == 1,
              "scheduler must use unbounded weighted pressure delta");
  if (ok) {
    llvm::outs() << "scheduler unbounded-pressure-delta: pass\n";
  }
  return ok;
}

struct RandomDAGEdgeSpec {
  unsigned predecessor = 0;
  unsigned successor = 0;
  unsigned latency = 0;
};

struct RandomDAGNodeSpec {
  std::array<int, 2> operandSources = {-1, -2};
};

struct RandomDAGSpec {
  uint32_t seed = 0;
  SmallVector<RandomDAGNodeSpec> nodes;
  SmallVector<RandomDAGEdgeSpec> edges;
};

class StableRandom final {
public:
  explicit StableRandom(uint32_t seed) : state(seed ? seed : 1) {}

  uint32_t next(uint32_t bound) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return bound == 0 ? 0 : state % bound;
  }

private:
  uint32_t state;
};

static bool hasEdge(const RandomDAGSpec &spec, unsigned predecessor,
                    unsigned successor) {
  return llvm::any_of(spec.edges, [&](const RandomDAGEdgeSpec &edge) {
    return edge.predecessor == predecessor && edge.successor == successor;
  });
}

static void addRandomDAGEdge(RandomDAGSpec &spec, unsigned predecessor,
                             unsigned successor, unsigned latency) {
  if (!hasEdge(spec, predecessor, successor)) {
    spec.edges.push_back({predecessor, successor, latency});
  }
}

static void assignRandomDAGOperands(RandomDAGSpec &spec) {
  for (auto [nodeIndex, node] : llvm::enumerate(spec.nodes)) {
    SmallVector<unsigned, 2> predecessors;
    for (const RandomDAGEdgeSpec &edge : spec.edges) {
      bool isOperandPredecessor =
          edge.successor == nodeIndex && predecessors.size() < 2;
      if (isOperandPredecessor) {
        predecessors.push_back(edge.predecessor);
      }
    }
    if (!predecessors.empty()) {
      node.operandSources[0] = static_cast<int>(predecessors[0]);
    }
    bool hasTwoPredecessors = predecessors.size() == 2;
    if (hasTwoPredecessors) {
      node.operandSources[1] = static_cast<int>(predecessors[1]);
    }
  }
}

static RandomDAGSpec generateRandomDAGSpec(uint32_t seed) {
  static constexpr size_t maxEdges = 24;
  StableRandom random(seed);
  RandomDAGSpec spec;
  spec.seed = seed;
  spec.nodes.resize(11 + random.next(3));

  // Node 0 stays isolated. The remaining fixed edges embed a chain, fan-out,
  // fan-in, diamond, and multi-layer paths before random forward edges are
  // added in original-index topological order.
  addRandomDAGEdge(spec, 1, 2, 1);
  addRandomDAGEdge(spec, 2, 3, 2);
  addRandomDAGEdge(spec, 1, 4, 1);
  addRandomDAGEdge(spec, 1, 5, 1);
  addRandomDAGEdge(spec, 4, 6, 0);
  addRandomDAGEdge(spec, 5, 6, 3);
  addRandomDAGEdge(spec, 2, 7, 1);
  addRandomDAGEdge(spec, 2, 8, 1);
  addRandomDAGEdge(spec, 7, 9, 2);
  addRandomDAGEdge(spec, 8, 9, 1);
  addRandomDAGEdge(spec, 6, 10, 1);
  addRandomDAGEdge(spec, 9, 10, 0);

  for (unsigned successor = 2; successor < spec.nodes.size(); ++successor) {
    for (unsigned predecessor = 1; predecessor < successor; ++predecessor) {
      bool atEdgeLimit = spec.edges.size() == maxEdges;
      if (atEdgeLimit) {
        break;
      }
      bool chooseEdge = random.next(5) == 0;
      if (chooseEdge) {
        addRandomDAGEdge(spec, predecessor, successor, random.next(4));
      }
    }
  }
  assignRandomDAGOperands(spec);
  return spec;
}

static void printRandomDAGSpec(raw_ostream &os, const RandomDAGSpec &spec) {
  os << "seed=" << spec.seed << " nodes=" << spec.nodes.size() << " [";
  for (auto [index, node] : llvm::enumerate(spec.nodes)) {
    if (index != 0) {
      os << ", ";
    }
    os << 'n' << index << '(';
    for (unsigned operandIndex = 0; operandIndex < 2; ++operandIndex) {
      if (operandIndex != 0) {
        os << ',';
      }
      int source = node.operandSources[operandIndex];
      if (source >= 0) {
        os << 'n' << source;
      } else {
        os << "arg" << (-source - 1);
      }
    }
    os << ')';
  }
  os << "] edges=[";
  for (auto [index, edge] : llvm::enumerate(spec.edges)) {
    if (index != 0) {
      os << ", ";
    }
    os << 'n' << edge.predecessor << "->n" << edge.successor << '@'
       << edge.latency;
  }
  os << "] pressure=index:1\n";
}

static bool checkRandom(bool condition, const RandomDAGSpec &spec,
                        const Twine &message) {
  if (condition) {
    return true;
  }
  llvm::errs() << "FAIL: random DAG " << message << '\n';
  printRandomDAGSpec(llvm::errs(), spec);
  return false;
}

struct RandomDAGFixture {
  OwningOpRef<ModuleOp> module;
  std::unique_ptr<VPTOSchedDAG> dag;
};

static std::string getRandomDAGOperandName(int source) {
  if (source >= 0) {
    return (Twine("%n") + Twine(source)).str();
  }
  return (Twine("%arg") + Twine(-source - 1)).str();
}

static std::string buildRandomDAGSource(const RandomDAGSpec &spec) {
  std::string source;
  llvm::raw_string_ostream os(source);
  os << "module {\n  func.func @random_dag(%arg0: index, %arg1: index) {\n";
  for (auto [index, node] : llvm::enumerate(spec.nodes)) {
    os << "    %n" << index << " = arith.addi "
       << getRandomDAGOperandName(node.operandSources[0]) << ", "
       << getRandomDAGOperandName(node.operandSources[1]) << " : index\n";
  }
  os << "    return\n  }\n}\n";
  os.flush();
  return source;
}

static FailureOr<VPTOSchedRegion>
buildRandomDAGRegion(RandomDAGFixture &fixture, const RandomDAGSpec &spec) {
  func::FuncOp function =
      fixture.module->lookupSymbol<func::FuncOp>("random_dag");
  if (!function) {
    return failure();
  }
  Block &block = function.getBody().front();
  VPTOSchedRegion region;
  region.block = &block;
  for (Operation &op : block) {
    if (op.hasTrait<OpTrait::IsTerminator>()) {
      region.followingBoundary = &op;
      break;
    }
    region.operations.push_back(&op);
  }
  bool invalidNodeCount = region.operations.size() != spec.nodes.size();
  if (invalidNodeCount) {
    return failure();
  }
  return region;
}

static LogicalResult addRandomDAGEdges(VPTOSchedDAG &dag,
                                       const RandomDAGSpec &spec) {
  ArrayRef<std::unique_ptr<VPTOSUnit>> units = dag.getUnits();
  for (const RandomDAGEdgeSpec &edge : spec.edges) {
    bool invalidEdge = edge.predecessor >= units.size() ||
                       edge.successor >= units.size() ||
                       edge.predecessor >= edge.successor;
    if (invalidEdge) {
      return failure();
    }
    dag.addEdge(*units[edge.predecessor], *units[edge.successor],
                VPTOSchedEdgeKind::Data, VPTOSchedEdgeStrength::Must,
                edge.latency, "random-dag differential edge");
  }
  return dag.computeCriticalPaths();
}

static FailureOr<RandomDAGFixture>
buildRandomDAGFixture(MLIRContext &context, const RandomDAGSpec &spec) {
  RandomDAGFixture fixture;
  fixture.module = parseModule(context, buildRandomDAGSource(spec));
  if (!fixture.module) {
    return failure();
  }
  FailureOr<VPTOSchedRegion> region = buildRandomDAGRegion(fixture, spec);
  if (failed(region)) {
    return failure();
  }
  fixture.dag = std::make_unique<VPTOSchedDAG>(*region);
  Block *block = region->block;
  fixture.dag->addLiveIn(block->getArgument(0));
  fixture.dag->addLiveIn(block->getArgument(1));
  if (failed(addRandomDAGEdges(*fixture.dag, spec))) {
    return failure();
  }
  fixture.dag->resetDependencyCounts();
  return fixture;
}

static bool verifyPermutationOracle(const VPTOSchedDAG &dag,
                                    const VPTOScheduleResult &result) {
  ArrayRef<std::unique_ptr<VPTOSUnit>> units = dag.getUnits();
  bool resultIsComplete = result.entries.size() == units.size();
  if (!resultIsComplete) {
    return false;
  }
  SmallVector<bool> seen(units.size(), false);
  for (const VPTOScheduleEntry &entry : result.entries) {
    bool invalidEntry = !entry.unit || entry.unit->getId() >= units.size();
    if (!invalidEntry) {
      invalidEntry = units[entry.unit->getId()].get() != entry.unit ||
                     seen[entry.unit->getId()];
    }
    if (invalidEntry) {
      return false;
    }
    seen[entry.unit->getId()] = true;
  }
  return llvm::all_of(seen, [](bool value) { return value; });
}

static SmallVector<unsigned>
getSchedulePositions(const VPTOSchedDAG &dag,
                     const VPTOScheduleResult &result) {
  SmallVector<unsigned> positions(dag.getUnits().size(),
                                  std::numeric_limits<unsigned>::max());
  for (auto [position, entry] : llvm::enumerate(result.entries)) {
    bool hasKnownPosition =
        entry.unit && entry.unit->getId() < positions.size();
    if (hasKnownPosition) {
      positions[entry.unit->getId()] = static_cast<unsigned>(position);
    }
  }
  return positions;
}

static bool verifyMustEdgesOracle(const VPTOSchedDAG &dag,
                                  const VPTOScheduleResult &result) {
  SmallVector<unsigned> positions = getSchedulePositions(dag, result);
  for (const std::unique_ptr<VPTOSchedEdge> &edge : dag.getEdges()) {
    bool violatesMustEdge =
        edge->isMust() && positions[edge->getPredecessor()->getId()] >=
                              positions[edge->getSuccessor()->getId()];
    if (violatesMustEdge) {
      return false;
    }
  }
  return true;
}

static bool verifyReadyCyclesOracle(const VPTOSchedDAG &dag,
                                    const VPTOScheduleResult &result) {
  SmallVector<unsigned> cycles(dag.getUnits().size(), 0);
  SmallVector<bool> scheduled(dag.getUnits().size(), false);
  for (const VPTOScheduleEntry &entry : result.entries) {
    unsigned readyCycle = 0;
    for (VPTOSchedEdge *edge : entry.unit->getPredecessors()) {
      if (!edge->isMust()) {
        continue;
      }
      if (!scheduled[edge->getPredecessor()->getId()]) {
        return false;
      }
      uint64_t edgeReady =
          static_cast<uint64_t>(cycles[edge->getPredecessor()->getId()]) +
          edge->getLatency();
      if (edgeReady > std::numeric_limits<unsigned>::max()) {
        return false;
      }
      readyCycle = std::max(readyCycle, static_cast<unsigned>(edgeReady));
    }
    if (entry.issueCycle < readyCycle) {
      return false;
    }
    cycles[entry.unit->getId()] = entry.issueCycle;
    scheduled[entry.unit->getId()] = true;
  }
  return true;
}

static bool needsRandomDAGLiveness(const VPTOSchedDAG &dag, Value value) {
  return llvm::any_of(value.getUsers(),
                      [&](Operation *user) { return dag.lookup(user); });
}

struct PressureOracleResult {
  SmallVector<SmallVector<int64_t>> currentAfter;
  SmallVector<int64_t> peak;
  SmallVector<int64_t> deltas;
  bool valid = true;
};

struct PressureOracleState {
  llvm::DenseMap<Value, unsigned> remainingUses;
  llvm::DenseSet<Value> liveValues;
  int64_t current = 0;
};

static PressureOracleState initializePressureOracle(const VPTOSchedDAG &dag) {
  PressureOracleState state;
  for (Value liveIn : dag.getLiveIns()) {
    if (state.liveValues.insert(liveIn).second && liveIn.getType().isIndex()) {
      ++state.current;
    }
  }
  for (const std::unique_ptr<VPTOSUnit> &unit : dag.getUnits()) {
    for (Value operand : unit->getOperation()->getOperands()) {
      ++state.remainingUses[operand];
    }
  }
  return state;
}

static int64_t
getOraclePressureDelta(const VPTOSchedDAG &dag, const VPTOScheduleEntry &entry,
                       const PressureOracleState &state,
                       const llvm::DenseMap<Value, unsigned> &candidateUses) {
  int64_t delta = 0;
  for (const auto &use : candidateUses) {
    bool isLastUse = state.liveValues.contains(use.first) &&
                     state.remainingUses.lookup(use.first) == use.second &&
                     use.first.getType().isIndex();
    if (isLastUse) {
      --delta;
    }
  }
  for (Value value : entry.unit->getOperation()->getResults()) {
    bool needsResult = !state.liveValues.contains(value) &&
                       needsRandomDAGLiveness(dag, value) &&
                       value.getType().isIndex();
    if (needsResult) {
      ++delta;
    }
  }
  return delta;
}

static bool updateOracleLiveness(const VPTOSchedDAG &dag,
                                 const VPTOScheduleEntry &entry,
                                 PressureOracleState &state) {
  for (Value operand : entry.unit->getOperation()->getOperands()) {
    auto found = state.remainingUses.find(operand);
    bool invalidUseCount =
        found == state.remainingUses.end() || found->second == 0;
    if (invalidUseCount) {
      return false;
    }
    --found->second;
    if (found->second == 0) {
      state.liveValues.erase(operand);
    }
  }
  for (Value value : entry.unit->getOperation()->getResults()) {
    if (needsRandomDAGLiveness(dag, value)) {
      state.liveValues.insert(value);
    }
  }
  return true;
}

static PressureOracleResult
computePressureOracle(const VPTOSchedDAG &dag,
                      const VPTOScheduleResult &result) {
  PressureOracleState state = initializePressureOracle(dag);
  PressureOracleResult oracle;
  oracle.peak = {0, 0, state.current};
  for (const VPTOScheduleEntry &entry : result.entries) {
    llvm::DenseMap<Value, unsigned> candidateUses;
    for (Value operand : entry.unit->getOperation()->getOperands()) {
      ++candidateUses[operand];
    }
    int64_t delta = getOraclePressureDelta(dag, entry, state, candidateUses);
    state.current += delta;
    if (state.current < 0) {
      oracle.valid = false;
      return oracle;
    }
    oracle.peak[UnboundedPressure] =
        std::max(oracle.peak[UnboundedPressure], state.current);
    oracle.deltas.push_back(delta);
    oracle.currentAfter.push_back({0, 0, state.current});
    if (!updateOracleLiveness(dag, entry, state)) {
      oracle.valid = false;
      return oracle;
    }
  }
  return oracle;
}

static bool verifyPressureOracle(const TrackerTestModel &model,
                                 const VPTOSchedDAG &dag,
                                 const VPTOScheduleResult &result) {
  // This oracle derives liveness directly from SSA uses and index types. It
  // deliberately does not call Boundary, replay, or model pressure queries.
  PressureOracleResult oracle = computePressureOracle(dag, result);
  if (!oracle.valid || !llvm::equal(oracle.peak, result.peakPressure)) {
    return false;
  }
  llvm::SmallDenseSet<int64_t, 8> distinctDeltas(oracle.deltas.begin(),
                                                 oracle.deltas.end());
  bool hasVariedDeltas = distinctDeltas.size() >= 2;
  if (!hasVariedDeltas) {
    return false;
  }
  VPTORegPressureTracker tracker(model, dag, VPTOSchedDirection::Top);
  for (auto [index, entry] : llvm::enumerate(result.entries)) {
    LogicalResult committed = tracker.commit(*entry.unit);
    bool commitSucceeded = succeeded(committed);
    bool currentMatches =
        llvm::equal(tracker.getCurrent(), oracle.currentAfter[index]);
    if (!commitSucceeded || !currentMatches) {
      return false;
    }
  }
  return llvm::equal(tracker.getPeak(), oracle.peak);
}

static bool hasValidDecisionMetadata(const VPTOScheduleResult &result) {
  static constexpr std::array<StringLiteral, 11> reasons = {
      "pressure-safe-candidate",
      "lower-excess-growth",
      "lower-projected-excess",
      "near-limit-live-range-closing",
      "near-limit-pressure-preserving",
      "urgent-critical-path",
      "longer-critical-path",
      "lower-pressure-delta",
      "deterministic-tie-break",
      "stable-candidate-order",
      "only-candidate"};
  return llvm::all_of(result.entries, [&](const VPTOScheduleEntry &entry) {
    return entry.direction == VPTOSchedDirection::Top &&
           llvm::is_contained(reasons, entry.reason);
  });
}

static bool schedulesMatch(const VPTOScheduleResult &lhs,
                           const VPTOScheduleResult &rhs) {
  bool summaryMatches = lhs.entries.size() == rhs.entries.size() &&
                        llvm::equal(lhs.peakPressure, rhs.peakPressure);
  if (!summaryMatches) {
    return false;
  }
  return llvm::all_of(llvm::zip(lhs.entries, rhs.entries), [](auto entries) {
    const VPTOScheduleEntry &left = std::get<0>(entries);
    const VPTOScheduleEntry &right = std::get<1>(entries);
    return left.unit == right.unit && left.direction == right.direction &&
           left.issueCycle == right.issueCycle && left.reason == right.reason &&
           left.pressureDrivenIdle == right.pressureDrivenIdle;
  });
}

static bool verifierRejects(const VPTOSchedDAG &dag,
                            const VPTOScheduleResult &result) {
  VPTOSchedulingBudget budget(1ULL << 20);
  VPTOScheduleFailure failure;
  return failed(verifyVPTOScheduleResult(dag, result, budget, failure)) &&
         failure.kind == VPTOScheduleFailureKind::SemanticVerification;
}

static bool replayRejects(const TrackerTestModel &model,
                          const VPTOSchedDAG &dag,
                          const VPTOScheduleResult &result) {
  VPTOSchedulingBudget budget(1ULL << 20);
  VPTOScheduleFailure failure;
  return failed(
             replayVPTOScheduleResult(model, dag, result, budget, failure)) &&
         failure.kind == VPTOScheduleFailureKind::ModelReplay;
}

static SmallVector<Operation *>
getCurrentRandomDAGOrder(const VPTOSchedDAG &dag) {
  SmallVector<Operation *> order;
  for (Operation &op : *dag.getRegion().block) {
    if (dag.lookup(&op)) {
      order.push_back(&op);
    }
  }
  return order;
}

static bool hasOriginalRandomDAGOrder(const VPTOSchedDAG &dag) {
  return llvm::equal(getCurrentRandomDAGOrder(dag), dag.getRegion().operations);
}

static bool verifyRandomDAGMutations(MLIRContext &context,
                                     const TrackerTestModel &model,
                                     const RandomDAGSpec &spec,
                                     RandomDAGFixture &fixture,
                                     const VPTOScheduleResult &result) {
  bool hasCompleteSummary =
      !result.entries.empty() && result.peakPressure.size() > UnboundedPressure;
  if (!hasCompleteSummary) {
    return false;
  }
  VPTOScheduleResult mutated = result;
  mutated.entries.pop_back();
  bool ok = verifierRejects(*fixture.dag, mutated);

  mutated = result;
  mutated.entries.back() = mutated.entries.front();
  ok &= verifierRejects(*fixture.dag, mutated);

  mutated = result;
  mutated.entries.front().unit = nullptr;
  ok &= verifierRejects(*fixture.dag, mutated);

  FailureOr<RandomDAGFixture> foreignFixture =
      buildRandomDAGFixture(context, spec);
  if (failed(foreignFixture)) {
    return false;
  }
  mutated = result;
  mutated.entries.front().unit = foreignFixture->dag->getUnits().front().get();
  ok &= verifierRejects(*fixture.dag, mutated);

  SmallVector<unsigned> positions = getSchedulePositions(*fixture.dag, result);
  const VPTOSchedEdge &edge = *fixture.dag->getEdges().front();
  mutated = result;
  std::swap(mutated.entries[positions[edge.getPredecessor()->getId()]],
            mutated.entries[positions[edge.getSuccessor()->getId()]]);
  ok &= verifierRejects(*fixture.dag, mutated);

  mutated = result;
  if (mutated.entries.back().issueCycle ==
      std::numeric_limits<unsigned>::max()) {
    return false;
  }
  ++mutated.entries.back().issueCycle;
  ok &= replayRejects(model, *fixture.dag, mutated);

  mutated = result;
  mutated.entries.front().direction = VPTOSchedDirection::Bottom;
  ok &= replayRejects(model, *fixture.dag, mutated);

  mutated = result;
  if (mutated.peakPressure[UnboundedPressure] ==
      std::numeric_limits<int64_t>::max()) {
    return false;
  }
  ++mutated.peakPressure[UnboundedPressure];
  ok &= replayRejects(model, *fixture.dag, mutated);
  return ok && hasOriginalRandomDAGOrder(*fixture.dag);
}

static bool verifyRandomScheduleBudgets(const TrackerTestModel &model,
                                        RandomDAGFixture &fixture,
                                        const VPTOSchedulerLimits &limits,
                                        const VPTOScheduleResult &result,
                                        uint64_t scheduleWork) {
  VPTOSchedulingBudget exactBudget(scheduleWork);
  VPTOScheduleFailure failure;
  VPTOScheduler exactScheduler(model, *fixture.dag, limits, exactBudget);
  FailureOr<VPTOScheduleResult> exactResult = exactScheduler.schedule(failure);
  bool ok = succeeded(exactResult) && schedulesMatch(result, *exactResult);
  std::array<uint64_t, 3> smallBudgets = {0, scheduleWork / 2,
                                          scheduleWork - 1};
  for (uint64_t limit : smallBudgets) {
    VPTOSchedulingBudget smallBudget(limit);
    VPTOScheduleFailure budgetFailure;
    VPTOScheduler scheduler(model, *fixture.dag, limits, smallBudget);
    ok &= failed(scheduler.schedule(budgetFailure)) &&
          budgetFailure.kind == VPTOScheduleFailureKind::Budget &&
          hasOriginalRandomDAGOrder(*fixture.dag);
  }
  return ok;
}

static bool verifyRandomVerifierBudgets(RandomDAGFixture &fixture,
                                        const VPTOScheduleResult &result) {
  VPTOSchedulingBudget measure(1ULL << 20);
  VPTOScheduleFailure failure;
  bool ok = succeeded(
      verifyVPTOScheduleResult(*fixture.dag, result, measure, failure));
  uint64_t work = measure.getUsed();
  if (work == 0) {
    return false;
  }
  VPTOSchedulingBudget exact(work);
  ok &=
      succeeded(verifyVPTOScheduleResult(*fixture.dag, result, exact, failure));
  VPTOSchedulingBudget shortBudget(work - 1);
  ok &= failed(verifyVPTOScheduleResult(*fixture.dag, result, shortBudget,
                                        failure)) &&
        failure.kind == VPTOScheduleFailureKind::Budget &&
        hasOriginalRandomDAGOrder(*fixture.dag);
  return ok;
}

static bool
verifyRandomReplayAndApplyBudgets(const TrackerTestModel &model,
                                  RandomDAGFixture &fixture,
                                  const VPTOScheduleResult &result) {
  VPTOSchedulingBudget measure(1ULL << 20);
  VPTOScheduleFailure failure;
  bool ok = succeeded(
      replayVPTOScheduleResult(model, *fixture.dag, result, measure, failure));
  uint64_t work = measure.getUsed();
  if (work == 0) {
    return false;
  }
  VPTOSchedulingBudget exact(work);
  ok &= succeeded(
      replayVPTOScheduleResult(model, *fixture.dag, result, exact, failure));
  VPTOSchedulingBudget shortBudget(work - 1);
  ok &= failed(replayVPTOScheduleResult(model, *fixture.dag, result,
                                        shortBudget, failure)) &&
        failure.kind == VPTOScheduleFailureKind::Budget &&
        hasOriginalRandomDAGOrder(*fixture.dag);
  VPTOSchedulingBudget applyShort(result.entries.size() - 1);
  ok &= failed(applyVPTOScheduleResult(*fixture.dag, result, applyShort,
                                       failure)) &&
        failure.kind == VPTOScheduleFailureKind::Budget &&
        hasOriginalRandomDAGOrder(*fixture.dag);
  return ok;
}

static bool verifyRandomDAGBudgets(const TrackerTestModel &model,
                                   RandomDAGFixture &fixture,
                                   const VPTOSchedulerLimits &limits,
                                   const VPTOScheduleResult &result,
                                   uint64_t scheduleWork) {
  if (scheduleWork == 0 || result.entries.empty()) {
    return false;
  }
  return verifyRandomScheduleBudgets(model, fixture, limits, result,
                                     scheduleWork) &&
         verifyRandomVerifierBudgets(fixture, result) &&
         verifyRandomReplayAndApplyBudgets(model, fixture, result);
}

static bool verifyRandomDAGApply(MLIRContext &context,
                                 const TrackerTestModel &model,
                                 const RandomDAGSpec &spec,
                                 const VPTOSchedulerLimits &limits) {
  FailureOr<RandomDAGFixture> fixture = buildRandomDAGFixture(context, spec);
  if (failed(fixture)) {
    return false;
  }
  VPTOSchedulingBudget scheduleBudget(1ULL << 20);
  VPTOScheduleFailure failure;
  VPTOScheduler scheduler(model, *fixture->dag, limits, scheduleBudget);
  FailureOr<VPTOScheduleResult> result = scheduler.schedule(failure);
  if (failed(result)) {
    return false;
  }
  VPTOSchedulingBudget applyBudget(result->entries.size());
  if (failed(applyVPTOScheduleResult(*fixture->dag, *result, applyBudget,
                                     failure))) {
    return false;
  }
  SmallVector<Operation *> expected;
  for (const VPTOScheduleEntry &entry : result->entries) {
    expected.push_back(entry.unit->getOperation());
  }
  return llvm::equal(getCurrentRandomDAGOrder(*fixture->dag), expected);
}

static bool testRandomDAGDifferential(MLIRContext &context) {
  static constexpr std::array<uint32_t, 8> seeds = {
      0x10203040U, 0x13579BDFU, 0x2468ACE0U, 0x31415926U,
      0x5EED1143U, 0x89ABCDEFU, 0xC001D00DU, 0xF00DBAAAU};
  TrackerTestModel model(/*trackUnboundedPressure=*/true);
  VPTOSchedulerLimits limits;
  limits.maxNodes = 13;
  limits.maxEdges = 24;
  limits.maxWorkUnits = 1ULL << 20;

  for (uint32_t seed : seeds) {
    RandomDAGSpec spec = generateRandomDAGSpec(seed);
    FailureOr<RandomDAGFixture> fixture = buildRandomDAGFixture(context, spec);
    if (!checkRandom(succeeded(fixture), spec, "fixture construction")) {
      return false;
    }
    VPTOSchedulingBudget budget(limits.maxWorkUnits);
    VPTOScheduleFailure failure;
    VPTOScheduler scheduler(model, *fixture->dag, limits, budget);
    FailureOr<VPTOScheduleResult> result = scheduler.schedule(failure);
    bool ok = checkRandom(succeeded(result), spec, "schedule success");
    if (!ok) {
      return false;
    }
    uint64_t scheduleWork = budget.getUsed();
    if (!checkRandom(verifyPermutationOracle(*fixture->dag, *result), spec,
                     "permutation oracle") ||
        !checkRandom(verifyMustEdgesOracle(*fixture->dag, *result), spec,
                     "Must-edge oracle") ||
        !checkRandom(verifyReadyCyclesOracle(*fixture->dag, *result), spec,
                     "ready-cycle oracle") ||
        !checkRandom(verifyPressureOracle(model, *fixture->dag, *result), spec,
                     "pressure oracle") ||
        !checkRandom(hasValidDecisionMetadata(*result), spec,
                     "decision metadata") ||
        !checkRandom(
            verifyRandomDAGMutations(context, model, spec, *fixture, *result),
            spec, "mutated result rejection") ||
        !checkRandom(verifyRandomDAGBudgets(model, *fixture, limits, *result,
                                            scheduleWork),
                     spec, "work-unit budgets") ||
        !checkRandom(verifyRandomDAGApply(context, model, spec, limits), spec,
                     "apply success")) {
      return false;
    }
  }
  llvm::outs() << "scheduler random-dag differential: pass seeds="
               << seeds.size() << " dags=" << seeds.size() << '\n';
  return true;
}

} // namespace

int main() {
  DialectRegistry registry;
  registry.insert<PTODialect, func::FuncDialect, arith::ArithDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  TrackerTestModel model;
  VPTOGenericA5SchedModel genericModel;
  if (!testResourceTracker(context, model) ||
      !testPressureTracker(context, model) ||
      !testBitcastPressureAliasing(context, genericModel) ||
      !testBitcastPressureCacheInvalidation(context, genericModel) ||
      !testPressureAwareStrategy(context) ||
      !testGenericA5PredicateLimit() || !testBoundary(context, model) ||
      !testBoundaryBudget(context, model) || !testScheduler(context, model) ||
      !testPressureNoPendingProgress(context) ||
      !testPressureReliefDoesNotIdle(context) ||
      !testUnboundedPressureScheduling(context) ||
      !testRandomDAGDifferential(context)) {
    return 1;
  }
  return 0;
}

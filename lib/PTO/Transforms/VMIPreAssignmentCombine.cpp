// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMIPreAssignmentCombine.cpp - Pre-assignment VMI combines ---------===//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VMIPREASSIGNMENTCOMBINE
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

static std::optional<int64_t> getConstantIndexValue(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
    return constant.value();
  if (auto constant = value.getDefiningOp<arith::ConstantOp>())
    if (auto integerAttr = dyn_cast<IntegerAttr>(constant.getValue()))
      return integerAttr.getInt();
  return std::nullopt;
}

static LogicalResult canonicalizeContiguousGroupLoads(ModuleOp module) {
  SmallVector<VMIGroupLoadOp> loads;
  module.walk([&](VMIGroupLoadOp load) {
    auto resultType = dyn_cast<VMIVRegType>(load.getResult().getType());
    if (!resultType)
      return;

    int64_t numGroups = load.getNumGroupsAttr().getInt();
    int64_t laneCount = resultType.getElementCount();
    if (numGroups <= 0 || laneCount % numGroups != 0)
      return;

    std::optional<int64_t> rowStride =
        getConstantIndexValue(load.getRowStride());
    if (rowStride && *rowStride == laneCount / numGroups)
      loads.push_back(load);
  });

  OpBuilder builder(module.getContext());
  for (VMIGroupLoadOp load : loads) {
    builder.setInsertionPoint(load);
    auto replacement = builder.create<VMILoadOp>(
        load.getLoc(), load.getResult().getType(), load.getSource(),
        load.getOffset());
    load.getResult().replaceAllUsesWith(replacement.getResult());
    load.erase();
  }
  return success();
}

static LogicalResult fuseGroupSlotBroadcastLoads(ModuleOp module) {
  SmallVector<VMIGroupBroadcastOp> broadcasts;
  module.walk([&](VMIGroupBroadcastOp broadcast) {
    auto load = broadcast.getSource().getDefiningOp<VMIGroupSlotLoadOp>();
    if (!load || !load.getResult().hasOneUse())
      return;
    if (load.getNumGroupsAttr().getInt() !=
        broadcast.getNumGroupsAttr().getInt())
      return;

    if (!isa<VMIVRegType>(broadcast.getResult().getType()))
      return;
    broadcasts.push_back(broadcast);
  });

  OpBuilder builder(module.getContext());
  for (VMIGroupBroadcastOp broadcast : broadcasts) {
    auto load = broadcast.getSource().getDefiningOp<VMIGroupSlotLoadOp>();
    if (!load)
      continue;

    builder.setInsertionPoint(broadcast);
    auto fused = builder.create<VMIGroupBroadcastLoadOp>(
        broadcast.getLoc(), broadcast.getResult().getType(), load.getSource(),
        load.getOffset(), load.getSourceGroupStride(),
        broadcast.getNumGroupsAttr());
    broadcast.getResult().replaceAllUsesWith(fused.getResult());
    broadcast.erase();
    if (load->use_empty())
      load.erase();
  }
  return success();
}

struct VMIPreAssignmentCombinePass
    : pto::impl::VMIPreAssignmentCombineBase<VMIPreAssignmentCombinePass> {
  void runOnOperation() override {
    if (failed(canonicalizeContiguousGroupLoads(getOperation())) ||
        failed(fuseGroupSlotBroadcastLoads(getOperation())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVMIPreAssignmentCombinePass() {
  return std::make_unique<VMIPreAssignmentCombinePass>();
}

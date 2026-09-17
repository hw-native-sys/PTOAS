// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMIExpandImplicitEnsureLayouts.cpp - E2B broadcast load expansion --===//
//===----------------------------------------------------------------------===//

// The layout solver selects E2B as a lowering preference for
// group_broadcast_load, but the materialized result layout can still be
// contiguous when the consumers only require a dense contiguous value. E2B
// lowering of a contiguous multi-part broadcast (for example 256xf32 with
// num_groups = 8) is not a legal single-packet form, so vmi-to-vpto would
// leave such an op unconverted. This pass makes the E2B preference explicit:
// the load result vreg is given the deinterleaved (d2/d4) layout the direct
// E2B fact describes and an ensure_layout d2/d4 -> contiguous is inserted
// immediately after the load, so vmi-to-vpto lowers the load as one E2B
// packet per part and then converts the value back to contiguous.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/VMILayoutSupport.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VMIEXPANDIMPLICITENSURELAYOUTS
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

// Returns the deinterleaved result layout that E2B direct lowering of p op
// can produce, or null if the op should keep its current form.
static VMILayoutAttr getE2BDeinterleavedResultLayout(VMIGroupBroadcastLoadOp op,
                                                     VMILayoutAttr current) {
  if (!current || !current.isContiguous() || current.getLaneStride() != 1) {
    return VMILayoutAttr();
  }
  auto resultType = dyn_cast<VMIVRegType>(op.getResult().getType());
  if (!resultType) {
    return VMILayoutAttr();
  }

  VMILayoutSupport supports;
  std::string reason;
  // The capability probe must be layout-agnostic.  The op still carries the
  // materialized (contiguous) result layout here -- the very state this pass
  // exists to expand -- but the direct-fact table only reports the E2B row
  // when the probed type has no layout yet or already has the E2B result
  // layout (the state this pass is meant to create).  Probing with the op
  // overload would therefore gate E2B on its own output.  Instead, probe the
  // underlying fact on a layout-less twin type carrying the same element
  // count/type, exactly as layout assignment does before it picks a layout.
  auto probeType = VMIVRegType::get(
      resultType.getContext(), resultType.getElementCount(),
      resultType.getElementType(), Attribute());
  FailureOr<VMIGroupBroadcastLoadDirectFact> fact =
      supports.getGroupBroadcastLoadDirectFact(
          probeType, op.getSource().getType(), op.getSourceGroupStride(),
          op.getNumGroupsAttr().getInt(), &reason);
  if (failed(fact)) {
    return VMILayoutAttr();
  }
  if (fact->kind != VMIGroupBroadcastLoadDirectKind::E2B) {
    return VMILayoutAttr();
  }
  VMILayoutAttr directLayout = fact->layout.resultLayout;
  if (!directLayout || !directLayout.isDeinterleaved()) {
    return VMILayoutAttr();
  }

  // The inserted ensure_layout must itself be supported by the registered
  // layout materialization tables.
  auto directType = VMIVRegType::get(op->getContext(),
                                     resultType.getElementCount(),
                                     resultType.getElementType(), directLayout);
  if (failed(supports.getEnsureLayoutFact(directType, resultType, &reason))) {
    return VMILayoutAttr();
  }
  return directLayout;
}

static LogicalResult expandImplicitEnsureLayouts(ModuleOp module) {
  SmallVector<VMIGroupBroadcastLoadOp> loads;
  module.walk([&](VMIGroupBroadcastLoadOp op) {
    auto resultType = dyn_cast<VMIVRegType>(op.getResult().getType());
    if (!resultType || op->use_empty()) {
      return;
    }
    if (getE2BDeinterleavedResultLayout(op, resultType.getLayoutAttr())) {
      loads.push_back(op);
    }
  });

  for (VMIGroupBroadcastLoadOp op : loads) {
    auto resultType = cast<VMIVRegType>(op.getResult().getType());
    VMILayoutAttr directLayout =
        getE2BDeinterleavedResultLayout(op, resultType.getLayoutAttr());
    if (!directLayout) {
      continue;
    }
    auto directType = VMIVRegType::get(op->getContext(),
                                       resultType.getElementCount(),
                                       resultType.getElementType(),
                                       directLayout);

    // Record the current users before the rewrite so the ensure result
    // replaces exactly those uses and not the ensure op source itself.
    SmallVector<OpOperand *> users;
    users.reserve(std::distance(op->use_begin(), op->use_end()));
    for (OpOperand &use : op.getResult().getUses()) {
      users.push_back(&use);
    }

    OpBuilder builder(op);
    builder.setInsertionPointAfter(op);
    auto ensure = builder.create<VMIEnsureLayoutOp>(op.getLoc(), resultType,
                                                    op.getResult());

    op.getResult().setType(directType);
    for (OpOperand *use : users) {
      use->set(ensure.getResult());
    }
  }
  return success();
}

struct VMIExpandImplicitEnsureLayoutsPass
    : pto::impl::VMIExpandImplicitEnsureLayoutsBase<
          VMIExpandImplicitEnsureLayoutsPass> {
  void runOnOperation() override {
    if (failed(expandImplicitEnsureLayouts(getOperation()))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVMIExpandImplicitEnsureLayoutsPass() {
  return std::make_unique<VMIExpandImplicitEnsureLayoutsPass>();
}

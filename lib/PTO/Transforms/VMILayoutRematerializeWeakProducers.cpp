// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMILayoutRematerializeWeakProducers.cpp - Rematerialize weak producers ===//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VMILAYOUTREMATERIALIZEWEAKPRODUCERS
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

static bool isLayoutPolymorphicProducer(Operation *op) {
  return isa<VMIConstantOp, VMIBroadcastOp, VMIIotaOp, VMICreateMaskOp,
             VMICreateGroupMaskOp, VMIConstantMaskOp>(op);
}

struct VMILayoutRematerializeWeakProducersPass
    : public mlir::pto::impl::VMILayoutRematerializeWeakProducersBase<
          VMILayoutRematerializeWeakProducersPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      VMILayoutRematerializeWeakProducersPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *, 16> producers;
    module.walk([&](Operation *op) {
      if (isLayoutPolymorphicProducer(op)) {
        producers.push_back(op);
      }
    });

    for (Operation *producer : producers) {
      const unsigned resultCount = producer->getNumResults();
      if (resultCount != 1) {
        continue;
      }
      Value result = producer->getResult(0);
      SmallVector<OpOperand *, 4> uses;
      for (OpOperand &use : result.getUses()) {
        uses.push_back(&use);
      }
      const size_t useCount = uses.size();
      if (useCount < 2) {
        continue;
      }

      DenseMap<Operation *, SmallVector<OpOperand *, 4>> usesByOwner;
      SmallVector<Operation *, 4> owners;
      for (OpOperand *use : uses) {
        Operation *owner = use->getOwner();
        const bool isNewOwner = usesByOwner.find(owner) == usesByOwner.end();
        if (isNewOwner) {
          owners.push_back(owner);
        }
        usesByOwner[owner].push_back(use);
      }

      for (size_t ownerIndex = 1; ownerIndex < owners.size(); ++ownerIndex) {
        Operation *owner = owners[ownerIndex];
        OpBuilder builder(owner);
        Operation *clone = builder.clone(*producer);
        for (OpOperand *use : usesByOwner[owner]) {
          use->set(clone->getResult(0));
        }
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::pto::createVMILayoutRematerializeWeakProducersPass() {
  return std::make_unique<VMILayoutRematerializeWeakProducersPass>();
}

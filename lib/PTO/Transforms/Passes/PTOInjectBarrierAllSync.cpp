// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOInjectBarrierAllSync.cpp - Conservative sync barriers -----------===//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/Passes.h"
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <iterator>

namespace mlir {
namespace pto {
namespace func = ::mlir::func;

#define GEN_PASS_DEF_PTOINJECTBARRIERALLSYNC
#include "PTO/Transforms/Passes.h.inc"

} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

static bool hasReadOrWriteMemoryEffect(Operation *op) {
  auto memEffect = dyn_cast<MemoryEffectOpInterface>(op);
  return memEffect && (memEffect.hasEffect<MemoryEffects::Read>() ||
                       memEffect.hasEffect<MemoryEffects::Write>());
}

static bool isPipeAllBarrier(Operation *op) {
  auto barrier = dyn_cast_or_null<pto::BarrierOp>(op);
  return barrier && barrier.getPipe().getPipe() == pto::PIPE::PIPE_ALL;
}

static bool hasPreviousPipeAllBarrier(Operation *op) {
  Block *block = op->getBlock();
  if (!block) {
    return false;
  }
  auto it = op->getIterator();
  if (it == block->begin()) {
    return false;
  }
  return isPipeAllBarrier(&*std::prev(it));
}

static bool shouldInjectBarrierAllBefore(Operation *op) {
  Dialect *dialect = op->getDialect();
  if (!dialect ||
      dialect->getNamespace() != pto::PTODialect::getDialectNamespace()) {
    return false;
  }

  return isa<pto::OpPipeInterface>(op) && hasReadOrWriteMemoryEffect(op);
}

struct PTOInjectBarrierAllSyncPass
    : public mlir::pto::impl::PTOInjectBarrierAllSyncBase<
          PTOInjectBarrierAllSyncPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    SmallVector<Operation *> insertionPoints;
    SmallVector<func::ReturnOp> tailInsertionPoints;
    bool sawMemoryEffectingPipeOp = false;

    func.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (shouldInjectBarrierAllBefore(op)) {
        sawMemoryEffectingPipeOp = true;
        if (hasPreviousPipeAllBarrier(op)) {
          return WalkResult::advance();
        }
        insertionPoints.push_back(op);
      }
      return WalkResult::advance();
    });

    if (sawMemoryEffectingPipeOp) {
      func.walk([&](func::ReturnOp ret) {
        if (!hasPreviousPipeAllBarrier(ret)) {
          tailInsertionPoints.push_back(ret);
        }
      });
    }

    OpBuilder builder(func.getContext());
    auto pipeAll = pto::PipeAttr::get(func.getContext(), pto::PIPE::PIPE_ALL);
    for (Operation *op : insertionPoints) {
      builder.setInsertionPoint(op);
      builder.create<pto::BarrierOp>(op->getLoc(), pipeAll);
    }
    for (func::ReturnOp ret : tailInsertionPoints) {
      builder.setInsertionPoint(ret);
      builder.create<pto::BarrierOp>(ret.getLoc(), pipeAll);
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOInjectBarrierAllSyncPass() {
  return std::make_unique<PTOInjectBarrierAllSyncPass>();
}

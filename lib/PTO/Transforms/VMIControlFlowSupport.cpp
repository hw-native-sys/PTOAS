// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMIControlFlowSupport.cpp - VMI SCF loop constraints --------------===//

#include "PTO/Transforms/VMIControlFlowSupport.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;

LogicalResult VMIControlFlowSupport::addForConstraints(
    scf::ForOp forOp, EquivalenceCallback addEquivalent) {
  scf::YieldOp yieldOp = nullptr;
  if (Block *body = forOp.getBody()) {
    yieldOp = dyn_cast<scf::YieldOp>(body->getTerminator());
  }

  for (auto [index, initArg] : llvm::enumerate(forOp.getInitArgs())) {
    Value anchor = initArg;
    if (index < forOp.getRegionIterArgs().size() &&
        failed(addEquivalent(anchor, forOp.getRegionIterArgs()[index], forOp))) {
      return failure();
    }
    if (yieldOp && index < yieldOp.getNumOperands() &&
        failed(addEquivalent(anchor, yieldOp.getOperand(index), forOp))) {
      return failure();
    }
    if (index < forOp.getNumResults() &&
        failed(addEquivalent(anchor, forOp.getResult(index), forOp))) {
      return failure();
    }
  }
  return success();
}

LogicalResult VMIControlFlowSupport::addWhileConstraints(
    scf::WhileOp whileOp, EquivalenceCallback addEquivalent) {
  if (whileOp.getBefore().empty() || whileOp.getAfter().empty()) {
    return success();
  }

  Block &beforeBlock = whileOp.getBefore().front();
  Block &afterBlock = whileOp.getAfter().front();
  auto conditionOp = dyn_cast<scf::ConditionOp>(beforeBlock.getTerminator());
  auto yieldOp = dyn_cast<scf::YieldOp>(afterBlock.getTerminator());
  if (!conditionOp || !yieldOp) {
    return success();
  }

  // The before region carries the initial operands back through the after
  // region's yield.  The condition's forwarded values are a separate carry
  // sequence for the after region and the while results.  They must not be
  // unioned by their positional index with the initial operands.
  for (auto [index, init] : llvm::enumerate(whileOp.getInits())) {
    if (index < whileOp.getBeforeArguments().size() &&
        failed(addEquivalent(init, whileOp.getBeforeArguments()[index],
                             whileOp))) {
      return failure();
    }
    if (index < yieldOp.getNumOperands() &&
        failed(addEquivalent(init, yieldOp.getOperand(index), whileOp))) {
      return failure();
    }
  }

  for (auto [index, forwarded] : llvm::enumerate(conditionOp.getArgs())) {
    if (index < afterBlock.getNumArguments() &&
        failed(addEquivalent(forwarded, afterBlock.getArgument(index),
                             whileOp))) {
      return failure();
    }
    if (index < whileOp.getNumResults() &&
        failed(addEquivalent(forwarded, whileOp.getResult(index), whileOp))) {
      return failure();
    }
  }
  return success();
}

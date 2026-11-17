// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMIControlFlowSupport.cpp - VMI SCF loop constraints --------------===//

#include "PTO/Support/CodeConstants.h"
#include "PTO/Transforms/VMIControlFlowSupport.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;

namespace {

using EquivalenceCallback = VMIControlFlowSupport::EquivalenceCallback;

LogicalResult matchControlFlowValues(
    ValueRange sources, ValueRange destinations, Operation* op, EquivalenceCallback addEquivalent)
{
    for (auto [index, source] : llvm::enumerate(sources)) {
        bool hasDestination = index < destinations.size();
        if (hasDestination && failed(addEquivalent(source, destinations[index], op))) {
            return failure();
        }
    }
    return success();
}

LogicalResult addBranchConstraints(BranchOpInterface branch, EquivalenceCallback addEquivalent)
{
    for (auto [index, destination] : llvm::enumerate(branch->getSuccessors())) {
        SuccessorOperands operands = branch.getSuccessorOperands(index);
        unsigned produced = operands.getProducedOperandCount();
        if (produced > destination->getNumArguments()) {
            return branch->emitError("branch produces too many destination arguments");
        }
        LogicalResult matched = matchControlFlowValues(
            operands.getForwardedOperands(), destination->getArguments().drop_front(produced), branch, addEquivalent);
        if (failed(matched)) {
            return failure();
        }
    }
    return success();
}

LogicalResult addRegionExitConstraints(
    RegionBranchOpInterface branch, Region& region, EquivalenceCallback addEquivalent)
{
    SmallVector<RegionSuccessor, mlir::pto::kValue4> successors;
    branch.getSuccessorRegions(region, successors);
    for (const RegionSuccessor& successor : successors) {
        for (Block& block : region) {
            auto terminator = dyn_cast<RegionBranchTerminatorOpInterface>(block.getTerminator());
            if (!terminator) {
                continue;
            }
            LogicalResult matched = matchControlFlowValues(
                terminator.getSuccessorOperands(RegionBranchPoint(successor)), successor.getSuccessorInputs(), branch,
                addEquivalent);
            if (failed(matched)) {
                return failure();
            }
        }
    }
    return success();
}

LogicalResult addRegionConstraints(RegionBranchOpInterface branch, EquivalenceCallback addEquivalent)
{
    SmallVector<RegionSuccessor, mlir::pto::kValue4> successors;
    branch.getSuccessorRegions(RegionBranchPoint::parent(), successors);
    for (const RegionSuccessor& successor : successors) {
        LogicalResult matched = matchControlFlowValues(
            branch.getEntrySuccessorOperands(RegionBranchPoint(successor)), successor.getSuccessorInputs(), branch,
            addEquivalent);
        if (failed(matched)) {
            return failure();
        }
    }
    for (Region& region : branch->getRegions()) {
        if (failed(addRegionExitConstraints(branch, region, addEquivalent))) {
            return failure();
        }
    }
    return success();
}

} // namespace

LogicalResult VMIControlFlowSupport::addConstraints(Operation* op, EquivalenceCallback addEquivalent)
{
    if (!op) {
        return success();
    }
    // Preserve the established loop-carried equivalence classes, including the
    // zero-trip for result and the two distinct carry sequences of scf.while.
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        return addForConstraints(forOp, addEquivalent);
    }
    if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
        return addWhileConstraints(whileOp, addEquivalent);
    }
    if (auto branch = dyn_cast<BranchOpInterface>(op)) {
        return addBranchConstraints(branch, addEquivalent);
    }
    if (auto branch = dyn_cast<RegionBranchOpInterface>(op)) {
        return addRegionConstraints(branch, addEquivalent);
    }
    return success();
}

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

SmallVector<Type> mlir::pto::getConsistentCallResultTypes(ModuleOp module,
                                                          func::FuncOp func) {
  SmallVector<Type> resultTypes;
  bool found = false;
  module.walk([func, &resultTypes, &found](func::CallOp call) mutable {
    bool callsFunction = call.getCallee() == func.getSymName();
    if (!callsFunction) {
      return;
    }
    if (!found) {
      resultTypes.assign(call.getResultTypes().begin(),
                         call.getResultTypes().end());
      found = true;
      return;
    }
    bool hasMatchingArity = resultTypes.size() == call.getNumResults();
    if (!hasMatchingArity) {
      return;
    }
    for (auto [index, type] : llvm::enumerate(call.getResultTypes())) {
      if (resultTypes[index] != type) {
        resultTypes[index] = {};
      }
    }
  });
  return found ? resultTypes : SmallVector<Type>{};
}

SmallVector<Type> mlir::pto::getFunctionInputTypes(func::FuncOp func) {
  SmallVector<Type> inputs;
  inputs.reserve(func.getNumArguments());
  for (BlockArgument argument : func.getArguments()) {
    inputs.push_back(argument.getType());
  }
  return inputs;
}

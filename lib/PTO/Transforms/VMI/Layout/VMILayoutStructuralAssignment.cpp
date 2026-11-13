// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Completes ABI/CFG transport layouts without making new operation-layout choices.
#include "VMILayoutStructuralAssignment.h"
#include "PTO/IR/VMIUtils.h"
#include "PTO/Transforms/VMIControlFlowSupport.h"
#include "PTO/Transforms/VMILayoutPropagation.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/TypeSwitch.h"
using namespace mlir;
using namespace mlir::pto;
namespace {
class VMILayoutStructuralAssignment {
public:
    VMILayoutStructuralAssignment(ModuleOp module, VMILayoutPropagator& propagator)
        : module(module), propagator(&propagator)
    {}
    LogicalResult complete()
    {
        if (failed(seedStructuralValues()) || failed(matchStructuralEdges()) || failed(matchStructuralLoops()) ||
            failed(seedTransportArguments()) || failed(seedUnplannedResults())) {
            return failure();
        }
        return success();
    }

private:
    LogicalResult seedStructuralValues()
    {
        LogicalResult structuralValues = success();
        module.walk([&](Operation* op) {
            bool isStructural =
                isa<scf::IfOp, scf::ForOp, scf::WhileOp, scf::YieldOp, scf::ConditionOp, cf::BranchOp, cf::CondBranchOp,
                    cf::SwitchOp, func::CallOp, func::ReturnOp>(op);
            if (failed(structuralValues) || !isStructural) {
                return;
            }
            auto seed = [&](Value value) {
                if (!value || !isa<VMIVRegType, VMIMaskType>(value.getType()) ||
                    propagator->getRequestedOrCurrentLayout(value)) {
                    return;
                }
                structuralValues = propagator->installPlanned(value, VMILayoutAttr::getContiguous(value.getContext()));
            };
            for (Value operand : op->getOperands()) {
                seed(operand);
            }
            if (isa<scf::WhileOp>(op)) {
                for (Value result : op->getResults()) {
                    seed(result);
                }
            }
        });

        return structuralValues;
    }

    void matchEdge(OpOperand& edge, Value destination)
    {
        VMILayoutAttr layout = propagator->getRequestedOrCurrentLayout(destination);
        // Structural transport results/arguments can be untyped before this
        // pass.  In that case the edge source is the validated layout seed;
        // install it on the destination first, then constrain the edge use.
        if (!layout) {
            layout = propagator->getRequestedOrCurrentLayout(edge.get());
        }
        if (!layout || failed(propagator->installPlanned(destination, layout)) ||
            failed(propagator->installPlanned(edge, layout))) {
            edgeStatus = failure();
        }
    }
    void matchBranch(cf::BranchOp branch)
    {
        for (auto [index, operand] : llvm::enumerate(branch.getDestOperands())) {
            (void)operand;
            if (index < branch.getDest()->getNumArguments()) {
                matchEdge(branch->getOpOperand(index + 0), branch.getDest()->getArgument(index));
            }
        }
    }
    void matchCondBranch(cf::CondBranchOp branch)
    {
        unsigned trueOffset = 1;
        for (auto [index, operand] : llvm::enumerate(branch.getTrueDestOperands())) {
            (void)operand;
            if (index < branch.getTrueDest()->getNumArguments()) {
                matchEdge(branch->getOpOperand(trueOffset + index), branch.getTrueDest()->getArgument(index));
            }
        }
        unsigned falseOffset = trueOffset + branch.getTrueDestOperands().size();
        for (auto [index, operand] : llvm::enumerate(branch.getFalseDestOperands())) {
            (void)operand;
            if (index < branch.getFalseDest()->getNumArguments()) {
                matchEdge(branch->getOpOperand(falseOffset + index), branch.getFalseDest()->getArgument(index));
            }
        }
    }
    void matchExecuteRegion(scf::ExecuteRegionOp execute)
    {
        for (Block& block : execute.getRegion()) {
            matchYield(&block, execute.getResults());
        }
    }
    void matchIndexSwitch(scf::IndexSwitchOp switchOp)
    {
        matchYield(&switchOp.getDefaultBlock(), switchOp.getResults());
        for (unsigned index = 0; index < switchOp.getNumCases(); ++index) {
            matchYield(&switchOp.getCaseBlock(index), switchOp.getResults());
        }
    }
    void matchYield(Block* block, ResultRange results)
    {
        if (!block) {
            return;
        }
        auto yield = dyn_cast<scf::YieldOp>(block->getTerminator());
        if (!yield) {
            return;
        }
        for (auto [index, operand] : llvm::enumerate(yield.getOperands())) {
            (void)operand;
            if (index < results.size()) {
                matchEdge(yield->getOpOperand(index), results[index]);
            }
        }
    }
    LogicalResult matchStructuralEdges()
    {
        edgeStatus = success();
        module.walk([&](Operation* op) {
            if (failed(edgeStatus)) {
                return;
            }
            llvm::TypeSwitch<Operation*>(op)
                .Case<cf::BranchOp>([&](auto branch) { matchBranch(branch); })
                .Case<cf::CondBranchOp>([&](auto branch) { matchCondBranch(branch); })
                .Case<scf::ExecuteRegionOp>([&](auto execute) { matchExecuteRegion(execute); })
                .Case<scf::IndexSwitchOp>([&](auto switchOp) { matchIndexSwitch(switchOp); });
        });
        return edgeStatus;
    }

    LogicalResult matchStructuralLoops()
    {
        LogicalResult structuralLoops = success();
        module.walk([&](scf::WhileOp whileOp) {
            if (failed(structuralLoops)) {
                return;
            }
            structuralLoops =
                VMIControlFlowSupport::addWhileConstraints(whileOp, [&](Value lhs, Value rhs, Operation*) {
                    VMILayoutAttr lhsLayout = propagator->getRequestedOrCurrentLayout(lhs);
                    VMILayoutAttr rhsLayout = propagator->getRequestedOrCurrentLayout(rhs);
                    VMILayoutAttr layout = lhsLayout ? lhsLayout : rhsLayout;
                    if (!layout) {
                        return success();
                    }
                    if (failed(propagator->installPlanned(lhs, layout)) ||
                        failed(propagator->installPlanned(rhs, layout))) {
                        return failure();
                    }
                    return success();
                });
        });

        return structuralLoops;
    }

    LogicalResult seedBlockArguments(Block& block)
    {
        for (BlockArgument arg : block.getArguments()) {
            if (!isa<VMIVRegType, VMIMaskType>(arg.getType()) || getVMIExplicitLayout(arg.getType()) ||
                propagator->getRequestedOrCurrentLayout(arg)) {
                continue;
            }
            if (failed(propagator->installPlanned(arg, VMILayoutAttr::getContiguous(arg.getContext())))) {
                return failure();
            }
        }
        return success();
    }

    LogicalResult seedTransportArguments()
    {
        LogicalResult transportArgs = success();
        module.walk([&](Operation* op) {
            if (failed(transportArgs)) {
                return;
            }
            for (Region& region : op->getRegions()) {
                for (Block& block : region) {
                    if (failed(seedBlockArguments(block))) {
                        transportArgs = failure();
                        return;
                    }
                }
            }
        });

        return transportArgs;
    }

    LogicalResult seedUnplannedResults()
    {
        LogicalResult unplannedResults = success();
        module.walk([&](Operation* op) {
            if (failed(unplannedResults)) {
                return;
            }
            for (OpResult result : op->getResults()) {
                if (!isa<VMIVRegType, VMIMaskType>(result.getType()) || getVMIExplicitLayout(result.getType())) {
                    continue;
                }
                if (propagator->getRequestedOrCurrentLayout(result)) {
                    continue;
                }
                if (failed(propagator->installPlanned(result, VMILayoutAttr::getContiguous(result.getContext())))) {
                    unplannedResults = failure();
                    return;
                }
            }
        });

        return unplannedResults;
    }
    LogicalResult edgeStatus = success();
    ModuleOp module;
    VMILayoutPropagator* propagator;
};
} // namespace
LogicalResult mlir::pto::completeVMILayoutTransportAssignments(ModuleOp module, VMILayoutPropagator& propagator)
{
    return VMILayoutStructuralAssignment(module, propagator).complete();
}

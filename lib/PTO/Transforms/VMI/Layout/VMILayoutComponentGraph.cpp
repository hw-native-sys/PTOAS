// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

// Build and traverse layout components, including non-SSA structural edges.

#include "VMILayoutComponentGraph.h"

#include "PTO/IR/VMIUtils.h"
#include "PTO/Transforms/VMILayoutPlanner.h"
#include "PTO/Transforms/VMILayoutConflictSolver.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace mlir::pto;

namespace {

static bool isLayoutValue(Value value) { return value && isa<VMIVRegType, VMIMaskType>(value.getType()); }

static void appendLayoutValues(Operation* op, SmallVectorImpl<Value>& values)
{
    for (Value operand : op->getOperands()) {
        if (isLayoutValue(operand)) {
            values.push_back(operand);
        }
    }
    for (Value result : op->getResults()) {
        if (isLayoutValue(result)) {
            values.push_back(result);
        }
    }
}

} // namespace

// Build both directions before visiting any seed. Structural endpoints are
// not necessarily operands/results of their owner (yield -> parent result,
// branch -> destination argument), so an SSA use walk alone is insufficient.
VMILayoutComponentGraph::VMILayoutComponentGraph(Operation* scope)
{
    VMILayoutStructuralEdgeProvider provider;
    scope->walk([&](Operation* op) {
        SmallVector<Value, mlir::pto::kValue4> values;
        appendLayoutValues(op, values);
        for (const VMILayoutEqualityConstraint& edge : provider.enumerateEdges(op)) {
            values.push_back(edge.source);
            values.push_back(edge.destination);
        }
        DenseSet<Value> seen;
        for (Value value : values) {
            if (seen.insert(value).second) {
                operationValues[op].push_back(value);
                valueOperations[value].push_back(op);
            }
        }
    });
}

void VMILayoutComponentGraph::collect(
    Value seed, SmallVectorImpl<Operation*>& ops, DenseSet<Value>& globallySeenValues) const
{
    SmallVector<Value, mlir::pto::kValue16> worklist{seed};
    SmallPtrSet<Operation*, mlir::pto::kValue16> seenOps;
    while (!worklist.empty()) {
        Value value = worklist.pop_back_val();
        if (!globallySeenValues.insert(value).second) {
            continue;
        }
        auto users = valueOperations.find(value);
        if (users == valueOperations.end()) {
            continue;
        }
        for (Operation* op : users->second) {
            if (!seenOps.insert(op).second) {
                continue;
            }
            ops.push_back(op);
            auto connected = operationValues.find(op);
            if (connected != operationValues.end()) {
                worklist.append(connected->second);
            }
        }
    }
}

SmallVector<unsigned, kValue8> mlir::pto::getVMILayoutSSAComponents(ArrayRef<VMILayoutSolverOp> ops)
{
    SmallVector<SmallVector<unsigned, mlir::pto::kValue4>, mlir::pto::kValue8> adjacency(ops.size());
    DenseMap<Operation*, unsigned> indices;
    for (auto [index, solverOp] : llvm::enumerate(ops)) {
        indices[solverOp.op] = index;
    }
    for (auto [consumerIndex, solverOp] : llvm::enumerate(ops)) {
        for (Value operand : solverOp.op->getOperands()) {
            auto producerIt = indices.find(operand.getDefiningOp());
            if (producerIt == indices.end()) {
                continue;
            }
            adjacency[consumerIndex].push_back(producerIt->second);
            adjacency[producerIt->second].push_back(consumerIndex);
        }
    }
    SmallVector<unsigned, mlir::pto::kValue8> components(ops.size(), 0);
    unsigned nextComponent = 0;
    llvm::SmallBitVector visited(ops.size());
    for (unsigned root = 0; root < ops.size(); ++root) {
        if (visited.test(root)) {
            continue;
        }
        SmallVector<unsigned, mlir::pto::kValue4> worklist{root};
        visited.set(root);
        while (!worklist.empty()) {
            unsigned current = worklist.pop_back_val();
            components[current] = nextComponent;
            for (unsigned neighbor : adjacency[current]) {
                if (!visited.test(neighbor)) {
                    visited.set(neighbor);
                    worklist.push_back(neighbor);
                }
            }
        }
        ++nextComponent;
    }

    return components;
}

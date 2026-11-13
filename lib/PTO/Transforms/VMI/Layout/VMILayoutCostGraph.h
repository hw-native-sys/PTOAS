// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Private physical cost graph, shared by operation pricing and materialization.
#ifndef PTO_TRANSFORMS_VMI_VMILAYOUTCOSTGRAPH_H
#define PTO_TRANSFORMS_VMI_VMILAYOUTCOSTGRAPH_H
#include "PTO/Transforms/VMILayoutCostModel.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/VMIUtils.h"
#include "PTO/Transforms/VMILayoutSupport.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSwitch.h"
#include <algorithm>
#include <limits>
#include <optional>
namespace mlir::pto::layout_cost {
inline bool isCarrierWiseArithmetic(Operation* op)
{
    return llvm::StringSwitch<bool>(op->getName().getStringRef())
#define VMI_MASKED_ARITHMETIC(Source, Target) .Case(Source::getOperationName(), true)
#define VMI_SCALAR_ARITHMETIC(Source, Target) VMI_MASKED_ARITHMETIC(Source, Target)
#include "PTO/Transforms/VMICarrierWiseArithmetic.def"
#undef VMI_SCALAR_ARITHMETIC
#undef VMI_MASKED_ARITHMETIC
        .Default(false);
}

enum class PhysicalActionKind {
    MemoryRead,
    MemoryWrite,
    Semantic,
    Compute,
    Interleave,
    Deinterleave,
    Unpack,
    Pack,
    Merge,
};

struct PhysicalValue {
    unsigned id = 0;
};

struct PhysicalAction {
    PhysicalActionKind kind = PhysicalActionKind::Semantic;
    Region* scope = nullptr;
    SmallVector<unsigned, mlir::pto::kValue4> inputs;
    std::string tag;
    SmallVector<PhysicalValue, mlir::pto::kValue4> results;
};

inline Region* getDynamicScope(Operation* op) { return op ? op->getParentRegion() : nullptr; }

inline FailureOr<int64_t> getArity(VMIVRegType type, VMILayoutAttr layout)
{
    if (!type || !layout || (layout.isGroupSlots() && type.getElementCount() != layout.getNumGroups())) {
        return failure();
    }
    auto assigned = VMIVRegType::get(type.getContext(), type.getElementCount(), type.getElementType(), layout);
    return getVMIPhysicalArity(assigned);
}

inline FailureOr<int64_t> getMaskArity(VMIMaskType type, VMILayoutAttr layout)
{
    if (!type || !layout) {
        return failure();
    }
    auto assigned = VMIMaskType::get(type.getContext(), type.getElementCount(), type.getGranularity(), layout);
    return getVMIPhysicalArity(assigned);
}

inline VMILayoutAttr getMaskCarrierLayout(VMIMaskType type, VMILayoutAttr layout)
{
    return type ? getVMIMaskCarrierLayout(layout) : VMILayoutAttr{};
}

inline bool isFullPhysicalShape(VMIVRegType type)
{
    auto lanes = getDataLanesPerPart(type.getElementType());
    return succeeded(lanes) && *lanes > 0 && type.getElementCount() % *lanes == 0;
}

class PhysicalGraph {
public:
    LogicalResult addIntrinsicRearrangementCost(int64_t cost, Region* scope)
    {
        const int64_t remaining = std::numeric_limits<int64_t>::max() - intrinsicRearrangementCost;
        if (cost < 0 || cost > remaining) {
            return failure();
        }
        intrinsicRearrangementCost += cost;
        intrinsicCosts.push_back({cost, scope});
        return success();
    }

    SmallVector<PhysicalValue, mlir::pto::kValue4> createInputs(unsigned count)
    {
        SmallVector<PhysicalValue, mlir::pto::kValue4> inputs;
        for (unsigned index = 0; index < count; ++index) {
            inputs.push_back(PhysicalValue{nextValueId++});
        }
        return inputs;
    }

    SmallVector<PhysicalValue, mlir::pto::kValue4> addAction(
        PhysicalActionKind kind, Region* scope, ArrayRef<PhysicalValue> inputs, StringRef tag, unsigned resultCount = 1)
    {
        SmallVector<unsigned, mlir::pto::kValue4> inputIds;
        for (PhysicalValue input : inputs) {
            inputIds.push_back(input.id);
        }
        PhysicalAction action;
        action.kind = kind;
        action.scope = scope;
        action.inputs = std::move(inputIds);
        action.tag = tag.str();
        for (unsigned result = 0; result < resultCount; ++result) {
            action.results.push_back(PhysicalValue{nextValueId++});
        }
        actions.push_back(std::move(action));
        return actions.back().results;
    }

    VMILayoutScopeCost getCost() const
    {
        VMILayoutScopeCost cost;
        cost.intrinsicRearrangements = intrinsicRearrangementCost;
        for (const auto& [value, scope] : intrinsicCosts) {
            cost.total = saturatingAdd(cost.total, saturatingMultiply(value, scopeWeight(scope)));
        }
        for (const PhysicalAction& action : actions) {
            // Memory actions break ties in verified compute/rearrangement cost.
            // Their count distinguishes dense and part stores without assuming one
            // memory action has the same latency as one arithmetic or shuffle op.
            switch (action.kind) {
                case PhysicalActionKind::MemoryRead:
                    ++cost.memoryReads;
                    cost.memoryTotal = saturatingAdd(cost.memoryTotal, scopeWeight(action.scope));
                    continue;
                case PhysicalActionKind::MemoryWrite:
                    ++cost.memoryWrites;
                    cost.memoryTotal = saturatingAdd(cost.memoryTotal, scopeWeight(action.scope));
                    continue;
                case PhysicalActionKind::Semantic:
                    ++cost.semantics;
                    continue;
                case PhysicalActionKind::Compute:
                    ++cost.semantics;
                    break;
                case PhysicalActionKind::Interleave:
                    ++cost.interleaves;
                    break;
                case PhysicalActionKind::Deinterleave:
                    ++cost.deinterleaves;
                    break;
                case PhysicalActionKind::Unpack:
                    ++cost.unpacks;
                    break;
                case PhysicalActionKind::Pack:
                    ++cost.packs;
                    break;
                case PhysicalActionKind::Merge:
                    ++cost.merges;
                    break;
            }
            int64_t frequency = scopeWeight(action.scope);
            cost.total = saturatingAdd(cost.total, frequency);
        }
        return cost;
    }

private:
    static int64_t saturatingAdd(int64_t lhs, int64_t rhs)
    {
        return lhs > std::numeric_limits<int64_t>::max() - rhs ? std::numeric_limits<int64_t>::max() : lhs + rhs;
    }

    static int64_t saturatingMultiply(int64_t lhs, int64_t rhs)
    {
        if (lhs == 0 || rhs == 0) {
            return 0;
        }
        return lhs > std::numeric_limits<int64_t>::max() / rhs ? std::numeric_limits<int64_t>::max() : lhs * rhs;
    }

    static int64_t loopIterations(scf::ForOp loop)
    {
        constexpr int64_t kEstimatedLoopIterations = 2;
        std::optional<int64_t> lower = getConstantIntValue(loop.getLowerBound());
        std::optional<int64_t> upper = getConstantIntValue(loop.getUpperBound());
        std::optional<int64_t> step = getConstantIntValue(loop.getStep());
        if (!lower || !upper || !step || *step <= 0) {
            return kEstimatedLoopIterations;
        }
        if (*upper <= *lower) {
            return 0;
        }
        // Unsigned subtraction represents the positive span even when signed
        // bounds straddle zero. Ceiling division avoids overflowing span + step.
        uint64_t span = static_cast<uint64_t>(*upper) - static_cast<uint64_t>(*lower);
        uint64_t unsignedStep = static_cast<uint64_t>(*step);
        uint64_t count = span / unsignedStep + (span % unsignedStep != 0 ? 1 : 0);
        return static_cast<int64_t>(std::min<uint64_t>(count, std::numeric_limits<int64_t>::max()));
    }

    static int64_t scopeWeight(Region* scope)
    {
        int64_t weight = 1;
        Operation* parent = scope ? scope->getParentOp() : nullptr;
        while (parent) {
            if (auto loop = dyn_cast<scf::ForOp>(parent)) {
                weight = saturatingMultiply(weight, loopIterations(loop));
            }
            parent = parent->getParentOp();
        }
        return weight;
    }

    SmallVector<PhysicalAction, mlir::pto::kValue16> actions;
    SmallVector<std::pair<int64_t, Region*>, mlir::pto::kValue4> intrinsicCosts;
    unsigned nextValueId = 1;
    int64_t intrinsicRearrangementCost = 0;
};

struct PhysicalStateStorage {
    struct SharedMaterialization {
        Value source;
        VMILayoutAttr sourceLayout;
        VMILayoutAttr resultLayout;
        Block* block = nullptr;
        SmallVector<PhysicalValue, mlir::pto::kValue4> result;
    };

    PhysicalGraph graph;
    DenseMap<Value, SmallVector<PhysicalValue, mlir::pto::kValue4>> values;
    DenseMap<Value, VMILayoutAttr> layouts;
    DenseMap<Operation*, VMILayoutOpRelation> selectedRelations;
    DenseSet<Operation*> builtOps;
    SmallVector<SharedMaterialization, mlir::pto::kValue4> sharedMaterializations;
};

using PhysicalValues = SmallVector<PhysicalValue, kValue4>;
using PhysicalInputs = SmallVector<PhysicalValues, kValue4>;
struct BroadcastLoadState {
    VMILayoutAttr layout;
    VMILayoutAttr producerLayout;
    int64_t arity;
    int64_t producerArity;
    FailureOr<VMIGroupBroadcastLoadDirectFact> direct;
    FailureOr<VMIGroupBroadcastLoadLayoutFact> fact;
};
struct DataMaterialization {
    Value sourceValue;
    ArrayRef<PhysicalValue> source;
    VMIVRegType type;
    VMILayoutAttr sourceLayout;
    VMILayoutAttr resultLayout;
    int64_t sourceArity;
    int64_t resultArity;
    Operation* placement;
    Region* scope;
};
class PlanGraphBuilder {
public:
    PlanGraphBuilder(ArrayRef<VMILayoutOpRelation> relations, const VMILayoutPlan& plan);

    PlanGraphBuilder(
        const PhysicalStateStorage& initial, const VMILayoutOpRelation& relation, const VMILayoutPlan& plan);

    FailureOr<VMILayoutScopeCost> build();

    FailureOr<PhysicalStateStorage> append(const VMILayoutOpRelation& relation);

private:
    FailureOr<PhysicalValues> deint4ToContiguous(ArrayRef<PhysicalValue> source, size_t resultCount, Region* scope);
    FailureOr<PhysicalValues> materializeMaskLaneStrideEndpoint(
        PhysicalValue input, VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout, int64_t sourceArity,
        int64_t resultArity, Region* scope);

    FailureOr<PhysicalValues> buildBlockArgumentValue(Value value);
    void seedStructuralOperand(OpOperand& operand, VMILayoutAttr targetLayout);
    FailureOr<PhysicalValues> buildMaskOperand(
        OpOperand& operand, ArrayRef<PhysicalValue> source, VMILayoutAttr layout, VMILayoutAttr targetLayout);
    FailureOr<PhysicalValues> materializeDeint4ToDeint2(const DataMaterialization& context);
    FailureOr<PhysicalValues> materializeDeint2ToDeint4(const DataMaterialization& context);
    FailureOr<PhysicalValues> materializeDensePacket(const DataMaterialization& context);
    FailureOr<PhysicalValues> materializeThroughContiguous(const DataMaterialization& context);
    FailureOr<PhysicalValues> materializeDenseEndpoint(const DataMaterialization& context);
    FailureOr<PhysicalValues> materializeDataRecipe(const DataMaterialization& context);
    FailureOr<PhysicalValues> materializeMaskDenseEndpoint(
        PhysicalValue input, VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout, int64_t sourceArity,
        int64_t resultArity, Region* scope);

    LogicalResult buildSameLayoutResult(
        Operation* op, OpResult result, VMILayoutAttr layout, int64_t arity, const PhysicalInputs& inputs);
    // predicate is the physical conversion predicate of a masked cast and empty
    // for an unmasked one; it is charged as an input of every conversion part so
    // the graph keeps the predicate dependency the lowering also emits.
    FailureOr<PhysicalValues> buildWideningCast(
        ArrayRef<PhysicalValue> source, int64_t resultArity, Region* scope, PhysicalActionKind conversionKind,
        ArrayRef<PhysicalValue> predicate = {});
    FailureOr<PhysicalValues> buildNarrowingCast(
        ArrayRef<PhysicalValue> source, int64_t resultArity, Region* scope, ArrayRef<PhysicalValue> predicate = {});
    PhysicalValues buildDenseDeinterleavedLoad(int64_t arity, int64_t factor, Region* scope);
    PhysicalValues buildBroadcastLoadActions(
        VMIGroupBroadcastLoadOp load, const VMILayoutOpRelation& relation, const BroadcastLoadState& state);
    LogicalResult buildRelationEndpoints(Operation* op, const VMILayoutOpRelation& relation);

    FailureOr<int64_t> getCompactReductionComputeCount(
        const VMIGroupReduceLayoutFact& fact, VMIVRegType source, VMIVRegType result, int64_t numGroups);

    FailureOr<int64_t> getGroupReductionComputeCount(Operation* op, const VMILayoutOpRelation& relation);

    FailureOr<SmallVector<PhysicalValue, mlir::pto::kValue4>> buildValue(Value value);

    FailureOr<SmallVector<PhysicalValue, mlir::pto::kValue4>> buildOperand(
        OpOperand& operand, VMILayoutAttr targetLayout);

    LogicalResult buildOp(Operation* op);

    LogicalResult buildRelation(const VMILayoutOpRelation& relation);

    LogicalResult buildGeneratedMask(const VMILayoutOpRelation& relation);

    LogicalResult buildStaticMask(const VMILayoutOpRelation& relation);

    LogicalResult buildEnsure(const VMILayoutOpRelation& relation);

    LogicalResult buildMaskGranularity(const VMILayoutOpRelation& relation);

    LogicalResult buildBitcast(const VMILayoutOpRelation& relation);

    LogicalResult buildProducer(const VMILayoutOpRelation& relation);

    LogicalResult buildShuffle(const VMILayoutOpRelation& relation);

    LogicalResult buildHistogram(const VMILayoutOpRelation& relation);

    LogicalResult buildActivePrefixIndex(const VMILayoutOpRelation& relation);

    LogicalResult buildCompress(const VMILayoutOpRelation& relation);

    LogicalResult buildExpandLoad(const VMILayoutOpRelation& relation);

    LogicalResult buildGroupBroadcast(const VMILayoutOpRelation& relation);

    LogicalResult buildGroupReduce(const VMILayoutOpRelation& relation);

    LogicalResult buildIndexedLoad(const VMILayoutOpRelation& relation);

    LogicalResult buildReduce(const VMILayoutOpRelation& relation);

    LogicalResult buildInterleave(const VMILayoutOpRelation& relation);

    LogicalResult buildChannelSplit(const VMILayoutOpRelation& relation);

    LogicalResult buildChannelMerge(const VMILayoutOpRelation& relation);

    LogicalResult buildMaskedLoad(const VMILayoutOpRelation& relation);

    LogicalResult buildGroupLoad(const VMILayoutOpRelation& relation);

    LogicalResult buildDeinterleaveLoad(const VMILayoutOpRelation& relation);

    LogicalResult buildGroupStore(const VMILayoutOpRelation& relation);

    LogicalResult buildInterleaveStore(const VMILayoutOpRelation& relation);

    LogicalResult buildIndexedStore(const VMILayoutOpRelation& relation);

    LogicalResult buildMaskedStore(const VMILayoutOpRelation& relation);

    LogicalResult buildGroupBroadcastLoad(VMIGroupBroadcastLoadOp load, const VMILayoutOpRelation& relation);

    LogicalResult buildLoad(VMILoadOp load, const VMILayoutOpRelation& relation);

    LogicalResult buildStore(VMIStoreOp store, const VMILayoutOpRelation& relation);

    // Zip/unzip preserve the logical layout but change the element width.
    // Their carrier arities need not match: one shuffle pairs one or two wide
    // registers with a low/high half-width register pair, as in VPTO lowering.
    static FailureOr<size_t> getPairedShuffleRatio(size_t wide, size_t half);

    LogicalResult buildPairedShuffle(Operation* op, const VMILayoutOpRelation& relation);

    LogicalResult buildUnzip(Operation* op, VMILayoutAttr layout);

    LogicalResult buildZip(Operation* op, VMILayoutAttr layout);

    LogicalResult buildSameLayoutOp(Operation* op, const VMILayoutOpRelation& relation);

    // Layouts the cast's source, optional conversion predicate and result ports
    // carry.  A masked cast gives its predicate operand the source layout: the
    // verifier pairs one predicate lane with each source lane in one layout, so
    // the predicate travels with the source and is charged through the same
    // buildOperand query as every other mask operand of the cost graph.
    struct CastPortLayouts {
        VMILayoutAttr source;
        std::optional<VMILayoutAttr> predicate;
        VMILayoutAttr result;
    };

    static std::optional<CastPortLayouts> resolveCastPortLayouts(Operation* op, const VMILayoutOpRelation& relation);

    LogicalResult buildCast(Operation* op, const VMILayoutOpRelation& relation);

    // Emit the conversion actions of one cast relation: a width change goes
    // through the widening or narrowing recipe, a same-width cast is one semantic
    // action per carrier.  predicate is empty for an unmasked cast.
    FailureOr<SmallVector<PhysicalValue, mlir::pto::kValue4>> buildCastConversion(
        Operation* op, ArrayRef<PhysicalValue> source, int64_t resultArity, VMIVRegType sourceType,
        VMIVRegType resultType, const VMILayoutOpRelation& relation, ArrayRef<PhysicalValue> predicate);

    FailureOr<SmallVector<PhysicalValue, mlir::pto::kValue4>> addMaskActions(
        PhysicalValue source, PhysicalActionKind kind, int64_t count, Region* scope, StringRef tag);

    FailureOr<SmallVector<PhysicalValue, mlir::pto::kValue4>> materializeMaskPackTree(
        PhysicalValue source, int64_t sourceArity, int64_t levels, Region* scope);

    FailureOr<SmallVector<PhysicalValue, mlir::pto::kValue4>> materializeMaskLayout(
        ArrayRef<PhysicalValue> source, VMIMaskType type, VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
        Operation* placement);

    FailureOr<SmallVector<PhysicalValue, mlir::pto::kValue4>> groupSlotsToGroupSlots(
        ArrayRef<PhysicalValue> source, VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout, Region* scope);

    FailureOr<SmallVector<PhysicalValue, mlir::pto::kValue4>> materialize(
        Value sourceValue, ArrayRef<PhysicalValue> source, VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
        Operation* placement);

    FailureOr<SmallVector<PhysicalValue, mlir::pto::kValue4>> deinterleavedToContiguous(
        ArrayRef<PhysicalValue> source, int64_t factor, size_t resultCount, Region* scope);

    FailureOr<SmallVector<PhysicalValue, mlir::pto::kValue4>> contiguousToDeinterleaved(
        ArrayRef<PhysicalValue> source, int64_t factor, size_t resultCount, Region* scope);

    FailureOr<SmallVector<PhysicalValue, mlir::pto::kValue4>> contiguousToLaneStride(
        ArrayRef<PhysicalValue> source, int64_t stride, size_t resultCount, Region* scope);

    FailureOr<SmallVector<PhysicalValue, mlir::pto::kValue4>> laneStrideToContiguous(
        ArrayRef<PhysicalValue> source, int64_t stride, Region* scope);

    const VMILayoutPlan& plan;
    DenseMap<Operation*, VMILayoutOpRelation> selectedRelations;
    DenseMap<Value, SmallVector<PhysicalValue, mlir::pto::kValue4>> values;
    DenseMap<Value, VMILayoutAttr> layouts;
    DenseSet<Operation*> builtOps;
    DenseSet<Operation*> buildingOps;
    SmallVector<PhysicalStateStorage::SharedMaterialization, mlir::pto::kValue4> sharedMaterializations;
    PhysicalGraph graph;
};
} // namespace mlir::pto::layout_cost
#endif // PTO_TRANSFORMS_VMI_VMILAYOUTCOSTGRAPH_H

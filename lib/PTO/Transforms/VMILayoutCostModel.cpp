// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- VMILayoutCostModel.cpp - VMI physical layout cost model -----------===//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/VMILayoutCostModel.h"

#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/VMIUtils.h"
#include "PTO/Transforms/VMILayoutSupport.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>

using namespace mlir;
using namespace mlir::pto;

namespace {

enum class PhysicalActionKind {
  MemoryRead,
  MemoryWrite,
  Semantic,
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
  Region *scope = nullptr;
  SmallVector<unsigned, mlir::pto::kValue4> inputs;
  std::string tag;
  SmallVector<PhysicalValue, mlir::pto::kValue4> results;
};

static Region *getDynamicScope(Operation *op) {
  return op ? op->getParentRegion() : nullptr;
}

static FailureOr<int64_t> getArity(VMIVRegType type, VMILayoutAttr layout) {
  if (!type || !layout) {
    return failure();
  }
  auto assigned = VMIVRegType::get(type.getContext(), type.getElementCount(),
                                   type.getElementType(), layout);
  return getVMIPhysicalArity(assigned);
}

static FailureOr<int64_t> getMaskArity(VMIMaskType type, VMILayoutAttr layout) {
  if (!type || !layout) {
    return failure();
  }
  auto assigned = VMIMaskType::get(type.getContext(), type.getElementCount(),
                                   type.getGranularity(), layout);
  return getVMIPhysicalArity(assigned);
}

static VMILayoutAttr getMaskCarrierLayout(VMIMaskType type,
                                          VMILayoutAttr layout) {
  if (!type || !layout) {
    return {};
  }
  MLIRContext *ctx = type.getContext();
  if (layout.isContiguous()) {
    return VMILayoutAttr::getContiguous(ctx);
  }
  if (layout.isDeinterleaved()) {
    return VMILayoutAttr::getDeinterleaved(ctx, layout.getFactor());
  }
  if (layout.isBlockDeinterleaved()) {
    return VMILayoutAttr::getBlockDeinterleaved(ctx, layout.getFactor());
  }
  if (layout.isGroupSlots()) {
    return VMILayoutAttr::getGroupSlots(ctx, layout.getNumGroups(),
                                        layout.getSlots());
  }
  return {};
}

static bool isFullPhysicalShape(VMIVRegType type) {
  auto lanes = getDataLanesPerPart(type.getElementType());
  return succeeded(lanes) && *lanes > 0 && type.getElementCount() % *lanes == 0;
}

class PhysicalGraph {
public:
  LogicalResult addIntrinsicRearrangementCost(int64_t cost) {
    if (cost < 0 || intrinsicRearrangementCost >
                        std::numeric_limits<int64_t>::max() - cost) {
      return failure();
    }
    intrinsicRearrangementCost += cost;
    return success();
  }

  SmallVector<PhysicalValue, mlir::pto::kValue4> createInputs(unsigned count) {
    SmallVector<PhysicalValue, mlir::pto::kValue4> inputs;
    for (unsigned index = 0; index < count; ++index) {
      inputs.push_back(PhysicalValue{nextValueId++});
    }
    return inputs;
  }

  SmallVector<PhysicalValue, mlir::pto::kValue4>
  addAction(PhysicalActionKind kind, Region *scope,
            ArrayRef<PhysicalValue> inputs, StringRef tag,
            unsigned resultCount = 1) {
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

  VMILayoutScopeCost getCost() const {
    VMILayoutScopeCost cost;
    cost.total = intrinsicRearrangementCost;
    for (const PhysicalAction &action : actions) {
      // Native memory traffic and semantic compute are not layout-conversion
      // costs.  Only actions that materialize a physical rearrangement (or a
      // merge used by one) contribute to the static solver objective.
      switch (action.kind) {
      case PhysicalActionKind::MemoryRead:
      case PhysicalActionKind::MemoryWrite:
      case PhysicalActionKind::Semantic:
        continue;
      case PhysicalActionKind::Interleave:
      case PhysicalActionKind::Deinterleave:
      case PhysicalActionKind::Unpack:
      case PhysicalActionKind::Pack:
      case PhysicalActionKind::Merge:
        break;
      }
      if (cost.total == std::numeric_limits<int64_t>::max()) {
        return cost;
      }
      ++cost.total;
    }
    return cost;
  }

private:
  SmallVector<PhysicalAction, mlir::pto::kValue16> actions;
  unsigned nextValueId = 1;
  int64_t intrinsicRearrangementCost = 0;
};

struct PhysicalStateStorage {
  struct SharedMaterialization {
    Value source;
    VMILayoutAttr sourceLayout;
    VMILayoutAttr resultLayout;
    Block *block = nullptr;
    SmallVector<PhysicalValue, mlir::pto::kValue4> result;
  };

  PhysicalGraph graph;
  DenseMap<Value, SmallVector<PhysicalValue, mlir::pto::kValue4>> values;
  DenseMap<Value, VMILayoutAttr> layouts;
  DenseMap<Operation *, VMILayoutOpRelation> selectedRelations;
  DenseSet<Operation *> builtOps;
  SmallVector<SharedMaterialization, mlir::pto::kValue4>
      sharedMaterializations;
};

class PlanGraphBuilder {
public:
  PlanGraphBuilder(ArrayRef<VMILayoutOpRelation> relations,
                   const VMILayoutPlan &plan)
      : plan(plan) {
    for (const VMILayoutOpRelation &relation : relations) {
      selectedRelations[relation.op] = relation;
    }
  }

  PlanGraphBuilder(const PhysicalStateStorage &initial,
                   const VMILayoutOpRelation &relation,
                   const VMILayoutPlan &plan)
      : plan(plan), values(initial.values), layouts(initial.layouts),
        builtOps(initial.builtOps),
        sharedMaterializations(initial.sharedMaterializations),
        graph(initial.graph) {
    selectedRelations = initial.selectedRelations;
    selectedRelations[relation.op] = relation;
  }

  FailureOr<VMILayoutScopeCost> build() {
    for (const auto &entry : selectedRelations) {
      if (failed(buildOp(entry.first))) {
        return failure();
      }
    }
    return graph.getCost();
  }

  FailureOr<PhysicalStateStorage> append(const VMILayoutOpRelation &relation) {
    if (failed(buildOp(relation.op))) {
      return failure();
    }
    for (const VMILayoutPortAssignment &port : relation.ports) {
      if (port.kind == VMILayoutPortKind::Result) {
        layouts[relation.op->getResult(port.index)] = port.layout;
      }
    }
    return PhysicalStateStorage{
        std::move(graph), std::move(values), std::move(layouts),
        std::move(selectedRelations), std::move(builtOps),
        std::move(sharedMaterializations)};
  }

private:
  FailureOr<SmallVector<PhysicalValue, mlir::pto::kValue4>>
  buildValue(Value value) {
    auto cached = values.find(value);
    if (cached != values.end()) {
      return cached->second;
    }
    Operation *definingOp = value.getDefiningOp();
    if (!definingOp) {
      auto type = dyn_cast<VMIVRegType>(value.getType());
      auto planned = plan.valueLayouts.find(value);
      VMILayoutAttr layout =
          planned == plan.valueLayouts.end()
              ? (type ? type.getLayoutAttr() : VMILayoutAttr{})
              : planned->second;
      if (auto mask = dyn_cast<VMIMaskType>(value.getType())) {
        if (planned == plan.valueLayouts.end()) {
          layout = mask.getLayoutAttr();
        }
        if (!layout) {
          return failure();
        }
        values[value] = graph.createInputs(1);
        layouts[value] = layout;
        return values.lookup(value);
      }
      auto arity = getArity(type, layout);
      if (!type || !layout || failed(arity) || *arity <= 0 ||
          static_cast<uint64_t>(*arity) >
              std::numeric_limits<unsigned>::max()) {
        return failure();
      }
      values[value] = graph.createInputs(static_cast<unsigned>(*arity));
      layouts[value] = layout;
      return values.lookup(value);
    }
    if (!selectedRelations.count(definingOp)) {
      auto type = dyn_cast<VMIVRegType>(value.getType());
      VMILayoutAttr layout = plan.valueLayouts.lookup(value);
      if (!layout) {
        layout = layouts.lookup(value);
      }
      if (type && layout) {
        auto arity = getArity(type, layout);
        if (succeeded(arity) && *arity > 0) {
          values[value] = graph.createInputs(static_cast<unsigned>(*arity));
          layouts[value] = layout;
          return values.lookup(value);
        }
      }
      if (isa<VMIMaskType>(value.getType()) && layout) {
        values[value] = graph.createInputs(1);
        layouts[value] = layout;
        return values.lookup(value);
      }
    }
    if (failed(buildOp(definingOp))) {
      return failure();
    }
    cached = values.find(value);
    return cached == values.end()
               ? FailureOr<SmallVector<PhysicalValue, mlir::pto::kValue4>>(
                     failure())
               : FailureOr<SmallVector<PhysicalValue, mlir::pto::kValue4>>(
                     cached->second);
  }

  FailureOr<SmallVector<PhysicalValue, mlir::pto::kValue4>>
  buildOperand(OpOperand &operand, VMILayoutAttr targetLayout) {
    Operation *definingOp = operand.get().getDefiningOp();
    bool isStructuralValue =
        isa<BlockArgument>(operand.get()) ||
        (definingOp && !selectedRelations.count(definingOp));
    if (isStructuralValue && !plan.valueLayouts.contains(operand.get()) &&
        !layouts.lookup(operand.get())) {
      auto type = dyn_cast<VMIVRegType>(operand.get().getType());
      if (type) {
        auto arity = getArity(type, targetLayout);
        if (succeeded(arity) && *arity > 0) {
          values[operand.get()] =
              graph.createInputs(static_cast<unsigned>(*arity));
          layouts[operand.get()] = targetLayout;
        }
      } else if (isa<VMIMaskType>(operand.get().getType())) {
        values[operand.get()] = graph.createInputs(1);
        layouts[operand.get()] = targetLayout;
      }
    }
    auto source = buildValue(operand.get());
    auto sourceLayout = plan.valueLayouts.find(operand.get());
    VMILayoutAttr layout = sourceLayout == plan.valueLayouts.end()
                               ? layouts.lookup(operand.get())
                               : sourceLayout->second;
    if (!layout && isa<VMIVRegType, VMIMaskType>(operand.get().getType())) {
      // Structural transport (block arguments and function boundaries) is
      // constrained by the assignment propagator rather than a VMI relation.
      // For an unassigned internal transport value, the consuming relation's
      // required layout is the only legal physical realization available to
      // the cost graph.
      layout = targetLayout;
    }
    if (failed(source) || !layout) {
      return failure();
    }
    if (isa<VMIMaskType>(operand.get().getType())) {
      if (layout != targetLayout) {
        VMILayoutSupport support;
        auto mask = cast<VMIMaskType>(operand.get().getType());
        auto sourceType =
            VMIMaskType::get(mask.getContext(), mask.getElementCount(),
                             mask.getGranularity(), layout);
        auto targetType =
            VMIMaskType::get(mask.getContext(), mask.getElementCount(),
                             mask.getGranularity(), targetLayout);
        auto ensureFact =
            support.getEnsureMaskLayoutFact(sourceType, targetType);
        if (failed(ensureFact)) {
          return failure();
        }
        if (ensureFact->forwardsPhysicalParts) {
          return *source;
        }
        return materializeMaskLayout(*source, sourceType, layout, targetLayout,
                                     operand.getOwner());
      }
      return *source;
    }
    if (layout != targetLayout) {
      Block *block = operand.getOwner()->getBlock();
      for (const PhysicalStateStorage::SharedMaterialization &shared :
           sharedMaterializations) {
        if (shared.source == operand.get() &&
            shared.sourceLayout == layout &&
            shared.resultLayout == targetLayout && shared.block == block) {
          return shared.result;
        }
      }
      auto result = materialize(operand.get(), *source, layout, targetLayout,
                                operand.getOwner());
      if (failed(result)) {
        return failure();
      }
      sharedMaterializations.push_back(
          {operand.get(), layout, targetLayout, block, *result});
      return result;
    }
    auto result = materialize(operand.get(), *source, layout, targetLayout,
                              operand.getOwner());
    return result;
  }

  LogicalResult buildOp(Operation *op) {
    if (builtOps.contains(op)) {
      return success();
    }
    if (!buildingOps.insert(op).second) {
      return failure();
    }
    auto relationIt = selectedRelations.find(op);
    if (relationIt == selectedRelations.end()) {
      return failure();
    }
    const VMILayoutOpRelation &relation = relationIt->second;
    if (relation.ports.empty() && !relation.endpoints.empty()) {
      for (const VMILayoutRelationEndpoint &endpoint : relation.endpoints) {
        if (!endpoint.value || !endpoint.layout) {
          return failure();
        }
        if (endpoint.use) {
          if (endpoint.use->getOwner() != op ||
              failed(buildOperand(*endpoint.use, endpoint.layout))) {
            return failure();
          }
          continue;
        }
        layouts[endpoint.value] = endpoint.layout;
        if (endpoint.value.getDefiningOp() != op ||
            values.contains(endpoint.value)) {
          continue;
        }
        if (auto type = dyn_cast<VMIVRegType>(endpoint.value.getType())) {
          auto arity = getArity(type, endpoint.layout);
          if (failed(arity) || *arity <= 0) {
            return failure();
          }
          values[endpoint.value] =
              graph.createInputs(static_cast<unsigned>(*arity));
          continue;
        }
        if (isa<VMIMaskType>(endpoint.value.getType())) {
          values[endpoint.value] = graph.createInputs(1);
        }
      }
      buildingOps.erase(op);
      builtOps.insert(op);
      return success();
    }
    LogicalResult result = buildRelation(relation);
    buildingOps.erase(op);
    if (succeeded(result)) {
      if (failed(graph.addIntrinsicRearrangementCost(
              relation.intrinsicRearrangementCost))) {
        return failure();
      }
      builtOps.insert(op);
    }
    return result;
  }

  LogicalResult buildRelation(const VMILayoutOpRelation &relation) {
    Operation *op = relation.op;
    if (auto generated = dyn_cast<VMICreateGroupMaskOp>(op)) {
      if (relation.ports.size() != 1 ||
          relation.ports.front().kind != VMILayoutPortKind::Result) {
        return failure();
      }
      VMILayoutAttr resultLayout = relation.ports.front().layout;
      VMILayoutSupport support;
      auto fact = support.getGeneratedMaskLayoutFact(op, resultLayout);
      auto type = cast<VMIMaskType>(generated.getResult().getType());
      if (failed(fact)) {
        return failure();
      }
      if (fact->generationLayout == fact->resultLayout) {
        values[generated.getResult()] = graph.createInputs(1);
        layouts[generated.getResult()] = resultLayout;
        return success();
      }
      auto arity = getMaskArity(type, fact->generationLayout);
      if (failed(arity) || *arity <= 0) {
        return failure();
      }
      auto source = graph.createInputs(1);
      auto converted = materializeMaskLayout(
          source, type, fact->generationLayout, fact->resultLayout, op);
      if (failed(converted)) {
        return failure();
      }
      values[generated.getResult()] = std::move(*converted);
      layouts[generated.getResult()] = resultLayout;
      return success();
    }
    if (isa<VMICreateMaskOp, VMIConstantMaskOp, VMIPsetOp, VMIPgeOp, VMIPltOp>(
            op)) {
      if (relation.ports.size() != 1 ||
          relation.ports.front().kind != VMILayoutPortKind::Result) {
        return failure();
      }
      values[op->getResult(relation.ports.front().index)] =
          graph.createInputs(1);
      layouts[op->getResult(relation.ports.front().index)] =
          relation.ports.front().layout;
      return success();
    }
    if (isa<VMIEnsureLayoutOp, VMIEnsureMaskLayoutOp>(op)) {
      if (relation.ports.size() != 2 ||
          relation.ports[0].kind != VMILayoutPortKind::Operand ||
          relation.ports[1].kind != VMILayoutPortKind::Result) {
        return failure();
      }
      auto source = buildOperand(op->getOpOperand(0), relation.ports[0].layout);
      if (failed(source)) {
        return failure();
      }
      SmallVector<PhysicalValue, mlir::pto::kValue4> result;
      if (auto mask = dyn_cast<VMIMaskType>(op->getOperand(0).getType())) {
        auto converted =
            materializeMaskLayout(*source, mask, relation.ports[0].layout,
                                  relation.ports[1].layout, op);
        if (failed(converted)) {
          return failure();
        }
        result = std::move(*converted);
      } else {
        auto converted =
            materialize(op->getOperand(0), *source, relation.ports[0].layout,
                        relation.ports[1].layout, op);
        if (failed(converted)) {
          return failure();
        }
        result = std::move(*converted);
      }
      values[op->getResult(0)] = std::move(result);
      layouts[op->getResult(0)] = relation.ports[1].layout;
      return success();
    }
    if (auto ensure = dyn_cast<VMIEnsureMaskGranularityOp>(op)) {
      if (relation.ports.size() != 2) {
        return failure();
      }
      auto source =
          buildOperand(ensure.getSourceMutable(), relation.ports[0].layout);
      if (failed(source) || source->empty()) {
        return failure();
      }
      auto resultType = cast<VMIMaskType>(ensure.getResult().getType());
      VMILayoutAttr sourceCarrier =
          getMaskCarrierLayout(resultType, relation.ports[0].layout);
      VMILayoutAttr resultCarrier =
          getMaskCarrierLayout(resultType, relation.ports[1].layout);
      auto converted = materializeMaskLayout(*source, resultType, sourceCarrier,
                                             resultCarrier, op);
      if (failed(converted)) {
        return failure();
      }
      values[ensure.getResult()] = std::move(*converted);
      layouts[ensure.getResult()] = relation.ports[1].layout;
      return success();
    }
    if (isa<VMIBitcastOp, VMIVinterpretCastOp>(op)) {
      if (relation.ports.size() != 2 ||
          relation.ports[0].kind != VMILayoutPortKind::Operand ||
          relation.ports[1].kind != VMILayoutPortKind::Result) {
        return failure();
      }
      auto source =
          buildOperand(op->getOpOperand(0), relation.ports[0].layout);
      auto sourceType = dyn_cast<VMIVRegType>(op->getOperand(0).getType());
      auto resultType = dyn_cast<VMIVRegType>(op->getResult(0).getType());
      auto arity = resultType ? getArity(resultType, relation.ports[1].layout)
                              : FailureOr<int64_t>(failure());
      if (failed(source) || failed(arity) || source->empty() || *arity <= 0) {
        return failure();
      }
      if (!sourceType || !resultType) {
        return failure();
      }
      SmallVector<PhysicalValue, mlir::pto::kValue4> results;
      for (int64_t index = 0; index < *arity; ++index) {
        results.push_back(graph.addAction(
            PhysicalActionKind::Semantic, getDynamicScope(op),
            {(*source)[static_cast<size_t>(index) % source->size()]},
            "bitcast")[0]);
      }
      values[op->getResult(0)] = std::move(results);
      layouts[op->getResult(0)] = relation.ports[1].layout;
      return success();
    }
    if (isa<VMIConstantOp, VMIBroadcastOp, VMIIotaOp, VMIGroupIotaOp, VMIVciOp,
            VMIVbrcOp>(op)) {
      if (relation.ports.size() != 1 ||
          relation.ports.front().kind != VMILayoutPortKind::Result) {
        return failure();
      }
      Value result = op->getResult(relation.ports.front().index);
      auto type = dyn_cast<VMIVRegType>(result.getType());
      auto arity = type ? getArity(type, relation.ports.front().layout)
                        : FailureOr<int64_t>(failure());
      if (failed(arity) || *arity <= 0) {
        return failure();
      }
      values[result] = graph.createInputs(static_cast<unsigned>(*arity));
      layouts[result] = relation.ports.front().layout;
      return success();
    }
    if (auto load = dyn_cast<VMIGroupBroadcastLoadOp>(op)) {
      return buildGroupBroadcastLoad(load, relation);
    }
    if (auto shuffle = dyn_cast<VMIShuffleOp>(op)) {
      if (relation.ports.size() != 2 ||
          relation.ports[0].kind != VMILayoutPortKind::Operand ||
          relation.ports[1].kind != VMILayoutPortKind::Result) {
        return failure();
      }
      auto source =
          buildOperand(shuffle.getSourceMutable(), relation.ports[0].layout);
      auto resultType = dyn_cast<VMIVRegType>(shuffle.getResult().getType());
      if (failed(source) || source->empty() || !resultType) {
        return failure();
      }
      FailureOr<int64_t> resultArity =
          getArity(resultType, relation.ports[1].layout);
      if (failed(resultArity) || *resultArity <= 0) {
        return failure();
      }
      SmallVector<PhysicalValue, mlir::pto::kValue4> results;
      for (int64_t index = 0; index < *resultArity; ++index) {
        results.push_back(graph.addAction(
            PhysicalActionKind::Semantic, getDynamicScope(op),
            {(*source)[static_cast<size_t>(index) % source->size()]},
            "shuffle")[0]);
      }
      values[shuffle.getResult()] = std::move(results);
      layouts[shuffle.getResult()] = relation.ports[1].layout;
      return success();
    }
    if (isa<VMIVdhistOp, VMIVchistOp, VMIVselrOp>(op)) {
      if (relation.ports.empty()) {
        return failure();
      }
      SmallVector<SmallVector<PhysicalValue, mlir::pto::kValue4>,
                  mlir::pto::kValue4>
          inputs;
      OpResult result;
      VMILayoutAttr resultLayout;
      for (const VMILayoutPortAssignment &port : relation.ports) {
        if (port.kind == VMILayoutPortKind::Operand) {
          auto input = buildOperand(op->getOpOperand(port.index), port.layout);
          if (failed(input) || input->empty()) {
            return failure();
          }
          inputs.push_back(std::move(*input));
        } else {
          if (result || port.index >= op->getNumResults()) {
            return failure();
          }
          result = op->getResult(port.index);
          resultLayout = port.layout;
        }
      }
      if (!result || !resultLayout || inputs.empty()) {
        return failure();
      }
      auto resultType = dyn_cast<VMIVRegType>(result.getType());
      auto arity = resultType ? getArity(resultType, resultLayout)
                              : FailureOr<int64_t>(failure());
      if (failed(arity) || *arity <= 0) {
        return failure();
      }
      SmallVector<PhysicalValue, mlir::pto::kValue4> results;
      for (int64_t index = 0; index < *arity; ++index) {
        SmallVector<PhysicalValue, mlir::pto::kValue4> args;
        for (const auto &input : inputs) {
          args.push_back(
              input.size() == 1
                  ? input.front()
                  : input[static_cast<size_t>(index) % input.size()]);
        }
        results.push_back(graph.addAction(PhysicalActionKind::Semantic,
                                          getDynamicScope(op), args,
                                          op->getName().getStringRef())[0]);
      }
      values[result] = std::move(results);
      layouts[result] = resultLayout;
      return success();
    }
    if (auto active = dyn_cast<VMIActivePrefixIndexOp>(op)) {
      if (relation.ports.size() != 2) {
        return failure();
      }
      auto mask =
          buildOperand(active.getMaskMutable(), relation.ports[0].layout);
      auto type = dyn_cast<VMIVRegType>(active.getResult().getType());
      FailureOr<int64_t> arity = failure();
      if (type) {
        arity = getArity(type, relation.ports[1].layout);
      }
      if (failed(mask) || failed(arity) || mask->empty() || *arity <= 0) {
        return failure();
      }
      SmallVector<PhysicalValue, mlir::pto::kValue4> result;
      for (int64_t index = 0; index < *arity; ++index) {
        result.push_back(graph.addAction(PhysicalActionKind::Semantic,
                                         getDynamicScope(op), {mask->front()},
                                         "active_prefix_index")[0]);
      }
      values[active.getResult()] = std::move(result);
      layouts[active.getResult()] = relation.ports[1].layout;
      return success();
    }
    if (auto compress = dyn_cast<VMICompressOp>(op)) {
      if (relation.ports.size() != 3) {
        return failure();
      }
      auto source =
          buildOperand(compress.getSourceMutable(), relation.ports[0].layout);
      auto mask =
          buildOperand(compress.getMaskMutable(), relation.ports[1].layout);
      auto type = dyn_cast<VMIVRegType>(compress.getResult().getType());
      if (failed(source) || failed(mask) || !type || source->empty() ||
          mask->empty()) {
        return failure();
      }
      auto arity = getArity(type, relation.ports[2].layout);
      if (failed(arity) || *arity <= 0) {
        return failure();
      }
      SmallVector<PhysicalValue, mlir::pto::kValue4> result;
      for (int64_t index = 0; index < *arity; ++index) {
        result.push_back(graph.addAction(
            PhysicalActionKind::Semantic, getDynamicScope(op),
            {(*source)[static_cast<size_t>(index) % source->size()],
             mask->front()},
            "compress")[0]);
      }
      values[compress.getResult()] = std::move(result);
      layouts[compress.getResult()] = relation.ports[2].layout;
      return success();
    }
    if (auto expand = dyn_cast<VMIExpandLoadOp>(op)) {
      if (relation.ports.size() != 3) {
        return failure();
      }
      auto mask =
          buildOperand(expand.getMaskMutable(), relation.ports[0].layout);
      auto passthru =
          buildOperand(expand.getPassthruMutable(), relation.ports[1].layout);
      auto type = dyn_cast<VMIVRegType>(expand.getResult().getType());
      if (failed(mask) || failed(passthru) || !type || mask->empty() ||
          passthru->empty()) {
        return failure();
      }
      auto arity = getArity(type, relation.ports[2].layout);
      if (failed(arity) || *arity <= 0) {
        return failure();
      }
      SmallVector<PhysicalValue, mlir::pto::kValue4> result;
      for (int64_t index = 0; index < *arity; ++index) {
        result.push_back(graph.addAction(
            PhysicalActionKind::MemoryRead, getDynamicScope(op),
            {mask->front(), passthru->front()}, "expand_load", 1)[0]);
      }
      values[expand.getResult()] = std::move(result);
      layouts[expand.getResult()] = relation.ports[2].layout;
      return success();
    }
    if (auto broadcast = dyn_cast<VMIGroupBroadcastOp>(op)) {
      if (relation.ports.size() != 2 ||
          relation.ports[0].kind != VMILayoutPortKind::Operand ||
          relation.ports[1].kind != VMILayoutPortKind::Result) {
        return failure();
      }
      auto source =
          buildOperand(broadcast.getSourceMutable(), relation.ports[0].layout);
      auto resultType = dyn_cast<VMIVRegType>(broadcast.getResult().getType());
      FailureOr<int64_t> resultArity = failure();
      if (resultType) {
        resultArity = getArity(resultType, relation.ports[1].layout);
      }
      if (failed(source) || source->empty() || failed(resultArity)) {
        return failure();
      }
      int64_t resultArityValue = *resultArity;
      if (resultArityValue <= 0) {
        return failure();
      }
      SmallVector<PhysicalValue, mlir::pto::kValue4> results;
      for (int64_t index = 0; index < resultArityValue; ++index) {
        results.push_back(graph.addAction(
            PhysicalActionKind::Semantic, getDynamicScope(op),
            {(*source)[static_cast<size_t>(index) % source->size()]},
            "group_broadcast")[0]);
      }
      values[broadcast.getResult()] = std::move(results);
      layouts[broadcast.getResult()] = relation.ports[1].layout;
      return success();
    }
    if (isa<VMIGroupReduceAddFOp, VMIGroupReduceMaxFOp, VMIGroupReduceMinFOp,
            VMIGroupReduceAddIOp, VMIGroupReduceMaxIOp, VMIGroupReduceMinIOp,
            VMIvcaddOp, VMIvcmaxOp, VMIvcminOp>(op)) {
      if (relation.ports.size() != 3 ||
          relation.ports[0].kind != VMILayoutPortKind::Operand ||
          relation.ports[1].kind != VMILayoutPortKind::Operand ||
          relation.ports[2].kind != VMILayoutPortKind::Result) {
        return failure();
      }
      auto source = buildOperand(op->getOpOperand(0), relation.ports[0].layout);
      auto mask = buildOperand(op->getOpOperand(1), relation.ports[1].layout);
      auto resultType = dyn_cast<VMIVRegType>(op->getResult(0).getType());
      FailureOr<int64_t> resultArity = failure();
      if (resultType) {
        resultArity = getArity(resultType, relation.ports[2].layout);
      }
      if (failed(source) || failed(mask) || failed(resultArity) ||
          source->empty() || mask->empty()) {
        return failure();
      }
      int64_t resultArityValue = *resultArity;
      if (resultArityValue <= 0) {
        return failure();
      }
      SmallVector<PhysicalValue, mlir::pto::kValue4> results;
      for (int64_t index = 0; index < resultArityValue; ++index) {
        results.push_back(graph.addAction(
            PhysicalActionKind::Semantic, getDynamicScope(op),
            {(*source)[static_cast<size_t>(index) % source->size()],
             mask->front()},
            "group_reduce")[0]);
      }
      values[op->getResult(0)] = std::move(results);
      layouts[op->getResult(0)] = relation.ports[2].layout;
      return success();
    }
    if (auto load = dyn_cast<VMILoadOp>(op)) {
      return buildLoad(load, relation);
    }
    if (isa<VMIStrideLoadOp, VMIGatherOp>(op)) {
      if (relation.ports.empty()) {
        return failure();
      }
      for (const VMILayoutPortAssignment &port : relation.ports) {
        if (port.kind != VMILayoutPortKind::Operand) {
          continue;
        }
        if (failed(buildOperand(op->getOpOperand(port.index), port.layout))) {
          return failure();
        }
      }
      auto result = op->getResult(0);
      auto type = dyn_cast<VMIVRegType>(result.getType());
      auto arity = type ? getArity(type, relation.ports.back().layout)
                        : FailureOr<int64_t>(failure());
      if (failed(arity) || *arity <= 0) {
        return failure();
      }
      SmallVector<PhysicalValue, mlir::pto::kValue4> out;
      for (int64_t i = 0; i < *arity; ++i) {
        out.push_back(graph.addAction(
            PhysicalActionKind::MemoryRead, getDynamicScope(op), {},
            isa<VMIGatherOp>(op) ? "gather" : "stride_load", 1)[0]);
      }
      values[result] = std::move(out);
      layouts[result] = relation.ports.back().layout;
      return success();
    }
    if (isa<VMIReduceAddIOp, VMIReduceAddFOp, VMIReduceMaxFOp, VMIReduceMinFOp,
            VMIReduceMaxIOp, VMIReduceMinIOp>(op)) {
      if (relation.ports.size() != 3) {
        return failure();
      }
      auto source = buildOperand(op->getOpOperand(0), relation.ports[0].layout);
      auto mask = buildOperand(op->getOpOperand(1), relation.ports[1].layout);
      auto type = dyn_cast<VMIVRegType>(op->getResult(0).getType());
      FailureOr<int64_t> arity = failure();
      if (type) {
        arity = getArity(type, relation.ports[2].layout);
      }
      if (failed(source) || failed(mask) || failed(arity) || source->empty() ||
          mask->empty()) {
        return failure();
      }
      int64_t arityValue = *arity;
      if (arityValue <= 0) {
        return failure();
      }
      SmallVector<PhysicalValue, mlir::pto::kValue4> result;
      for (int64_t i = 0; i < arityValue; ++i) {
        result.push_back(
            graph.addAction(PhysicalActionKind::Semantic, getDynamicScope(op),
                            {(*source)[i % source->size()], mask->front()},
                            op->getName().getStringRef())[0]);
      }
      values[op->getResult(0)] = std::move(result);
      layouts[op->getResult(0)] = relation.ports[2].layout;
      return success();
    }
    if (isa<VMIVintlvOp, VMIVdintlvOp>(op)) {
      if (relation.ports.size() != 5) {
        return failure();
      }
      auto lhs = buildOperand(op->getOpOperand(0), relation.ports[0].layout);
      auto rhs = buildOperand(op->getOpOperand(1), relation.ports[1].layout);
      auto mask = buildOperand(op->getOpOperand(2), relation.ports[2].layout);
      auto lowType = dyn_cast<VMIVRegType>(op->getResult(0).getType());
      auto highType = dyn_cast<VMIVRegType>(op->getResult(1).getType());
      FailureOr<int64_t> lowArity = failure();
      FailureOr<int64_t> highArity = failure();
      if (lowType) {
        lowArity = getArity(lowType, relation.ports[3].layout);
      }
      if (highType) {
        highArity = getArity(highType, relation.ports[4].layout);
      }
      if (failed(lhs) || failed(rhs) || failed(mask) || failed(lowArity) ||
          failed(highArity) || lhs->empty() || rhs->empty() || mask->empty()) {
        return failure();
      }
      int64_t lowArityValue = *lowArity;
      int64_t highArityValue = *highArity;
      if (lowArityValue <= 0 || highArityValue <= 0) {
        return failure();
      }
      SmallVector<PhysicalValue, mlir::pto::kValue4> low, high;
      for (int64_t i = 0; i < lowArityValue; ++i) {
        low.push_back(graph.addAction(
            PhysicalActionKind::Semantic, getDynamicScope(op),
            {(*lhs)[i % lhs->size()], (*rhs)[i % rhs->size()], mask->front()},
            op->getName().getStringRef())[0]);
      }
      for (int64_t i = 0; i < highArityValue; ++i) {
        high.push_back(graph.addAction(
            PhysicalActionKind::Semantic, getDynamicScope(op),
            {(*lhs)[i % lhs->size()], (*rhs)[i % rhs->size()], mask->front()},
            op->getName().getStringRef())[0]);
      }
      values[op->getResult(0)] = std::move(low);
      values[op->getResult(1)] = std::move(high);
      layouts[op->getResult(0)] = relation.ports[3].layout;
      layouts[op->getResult(1)] = relation.ports[4].layout;
      return success();
    }
    if (auto split = dyn_cast<VMIChannelSplitOp>(op)) {
      if (relation.ports.size() != split.getNumResults() + 1) {
        return failure();
      }
      auto source =
          buildOperand(split.getSourceMutable(), relation.ports[0].layout);
      if (failed(source) || source->empty()) {
        return failure();
      }
      int64_t sourceOffset = 0;
      for (unsigned i = 0; i < split.getNumResults(); ++i) {
        auto type = dyn_cast<VMIVRegType>(split.getResult(i).getType());
        auto arity = type ? getArity(type, relation.ports[i + 1].layout)
                          : FailureOr<int64_t>(failure());
        if (failed(arity) || *arity <= 0 ||
            sourceOffset > static_cast<int64_t>(source->size()) - *arity) {
          return failure();
        }
        SmallVector<PhysicalValue, mlir::pto::kValue4> result;
        for (int64_t p = 0; p < *arity; ++p) {
          result.push_back((*source)[sourceOffset + p]);
        }
        sourceOffset += *arity;
        values[split.getResult(i)] = std::move(result);
        layouts[split.getResult(i)] = relation.ports[i + 1].layout;
      }
      if (sourceOffset != static_cast<int64_t>(source->size())) {
        return failure();
      }
      return success();
    }
    if (auto merge = dyn_cast<VMIChannelMergeOp>(op)) {
      if (relation.ports.size() != merge.getInputs().size() + 1) {
        return failure();
      }
      SmallVector<SmallVector<PhysicalValue, mlir::pto::kValue4>,
                  mlir::pto::kValue4>
          inputs;
      for (unsigned i = 0; i < merge.getInputs().size(); ++i) {
        auto input =
            buildOperand(merge->getOpOperand(i), relation.ports[i].layout);
        if (failed(input) || input->empty()) {
          return failure();
        }
        inputs.push_back(std::move(*input));
      }
      auto type = dyn_cast<VMIVRegType>(merge.getResult().getType());
      auto arity = type ? getArity(type, relation.ports.back().layout)
                        : FailureOr<int64_t>(failure());
      if (failed(arity) || *arity <= 0) {
        return failure();
      }
      SmallVector<PhysicalValue, mlir::pto::kValue4> result;
      for (const auto &input : inputs) {
        result.append(input.begin(), input.end());
      }
      if (static_cast<int64_t>(result.size()) != *arity) {
        return failure();
      }
      values[merge.getResult()] = std::move(result);
      layouts[merge.getResult()] = relation.ports.back().layout;
      return success();
    }
    if (auto load = dyn_cast<VMIMaskedLoadOp>(op)) {
      if (relation.ports.size() != 3 ||
          relation.ports[0].kind != VMILayoutPortKind::Operand ||
          relation.ports[1].kind != VMILayoutPortKind::Operand ||
          relation.ports[2].kind != VMILayoutPortKind::Result) {
        return failure();
      }
      auto mask = buildOperand(load.getMaskMutable(), relation.ports[0].layout);
      auto passthru =
          buildOperand(load.getPassthruMutable(), relation.ports[1].layout);
      auto type = dyn_cast<VMIVRegType>(load.getResult().getType());
      FailureOr<int64_t> arity = failure();
      if (type) {
        arity = getArity(type, relation.ports[2].layout);
      }
      if (failed(mask) || failed(passthru) || failed(arity) || mask->empty() ||
          passthru->empty()) {
        return failure();
      }
      int64_t arityValue = *arity;
      if (arityValue <= 0) {
        return failure();
      }
      SmallVector<PhysicalValue, mlir::pto::kValue4> result;
      for (int64_t i = 0; i < arityValue; ++i) {
        result.push_back(graph.addAction(
            PhysicalActionKind::MemoryRead, getDynamicScope(op),
            {mask->front(), passthru->front()}, "masked_load", 1)[0]);
      }
      values[load.getResult()] = std::move(result);
      layouts[load.getResult()] = relation.ports[2].layout;
      return success();
    }
    if (isa<VMIGroupLoadOp, VMIGroupSlotLoadOp>(op)) {
      if (relation.ports.size() != 1 ||
          relation.ports.front().kind != VMILayoutPortKind::Result) {
        return failure();
      }
      auto result = op->getResult(0);
      auto type = dyn_cast<VMIVRegType>(result.getType());
      auto arity = type ? getArity(type, relation.ports.front().layout)
                        : FailureOr<int64_t>(failure());
      if (failed(arity) || *arity <= 0) {
        return failure();
      }
      SmallVector<PhysicalValue, mlir::pto::kValue4> valuesForResult;
      for (int64_t index = 0; index < *arity; ++index) {
        valuesForResult.push_back(graph.addAction(
            PhysicalActionKind::MemoryRead, getDynamicScope(op), {},
            isa<VMIGroupSlotLoadOp>(op) ? "group_slot_load" : "group_load",
            1)[0]);
      }
      values[result] = std::move(valuesForResult);
      layouts[result] = relation.ports.front().layout;
      return success();
    }
    if (auto load = dyn_cast<VMIDeinterleaveLoadOp>(op)) {
      if (relation.ports.size() != 2 ||
          relation.ports[0].kind != VMILayoutPortKind::Result ||
          relation.ports[1].kind != VMILayoutPortKind::Result) {
        return failure();
      }
      auto lowType = dyn_cast<VMIVRegType>(load.getLow().getType());
      auto highType = dyn_cast<VMIVRegType>(load.getHigh().getType());
      if (!lowType || !highType) {
        return failure();
      }
      auto lowArity = getArity(lowType, relation.ports[0].layout);
      auto highArity = getArity(highType, relation.ports[1].layout);
      if (failed(lowArity) || failed(highArity) || *lowArity <= 0 ||
          *highArity <= 0 || *lowArity != *highArity) {
        return failure();
      }
      SmallVector<PhysicalValue, mlir::pto::kValue4> low;
      SmallVector<PhysicalValue, mlir::pto::kValue4> high;
      for (int64_t index = 0; index < *lowArity; ++index) {
        auto loaded = graph.addAction(PhysicalActionKind::MemoryRead,
                                      getDynamicScope(op), {}, "vlds", 2);
        low.push_back(loaded[0]);
        high.push_back(loaded[1]);
      }
      values[load.getLow()] = std::move(low);
      values[load.getHigh()] = std::move(high);
      layouts[load.getLow()] = relation.ports[0].layout;
      layouts[load.getHigh()] = relation.ports[1].layout;
      return success();
    }
    if (auto store = dyn_cast<VMIGroupStoreOp>(op)) {
      if (relation.ports.size() != 1 ||
          relation.ports.front().kind != VMILayoutPortKind::Operand) {
        return failure();
      }
      auto input =
          buildOperand(store.getValueMutable(), relation.ports.front().layout);
      if (failed(input) || input->empty()) {
        return failure();
      }
      SmallVector<PhysicalValue, mlir::pto::kValue4> storeValues = *input;
      auto valueType = dyn_cast<VMIVRegType>(store.getValue().getType());
      if (!valueType) {
        return failure();
      }
      auto concreteType = VMIVRegType::get(
          valueType.getContext(), valueType.getElementCount(),
          valueType.getElementType(), relation.ports.front().layout);
      auto fact =
          VMILayoutSupport().getGroupStoreLayoutFact(store, concreteType);
      if (succeeded(fact) && fact->stagingLayout) {
        auto staged =
            materialize(store.getValue(), *input, relation.ports.front().layout,
                        fact->stagingLayout, op);
        if (failed(staged)) {
          return failure();
        }
        storeValues = std::move(*staged);
      }
      for (PhysicalValue value : storeValues) {
        graph.addAction(PhysicalActionKind::MemoryWrite, getDynamicScope(op),
                        {value}, "group_store", 0);
      }
      return success();
    }
    if (auto store = dyn_cast<VMIInterleaveStoreOp>(op)) {
      if (relation.ports.size() != 2 ||
          relation.ports[0].kind != VMILayoutPortKind::Operand ||
          relation.ports[1].kind != VMILayoutPortKind::Operand) {
        return failure();
      }
      auto lowType = cast<VMIVRegType>(store.getLow().getType());
      auto highType = cast<VMIVRegType>(store.getHigh().getType());
      auto support = VMILayoutSupport().getInterleaveStoreSupport(
          VMIVRegType::get(lowType.getContext(), lowType.getElementCount(),
                           lowType.getElementType(), relation.ports[0].layout),
          VMIVRegType::get(highType.getContext(), highType.getElementCount(),
                           highType.getElementType(),
                           relation.ports[1].layout));
      if (failed(support)) {
        return failure();
      }
      auto low = buildOperand(store.getLowMutable(), relation.ports[0].layout);
      auto high =
          buildOperand(store.getHighMutable(), relation.ports[1].layout);
      if (failed(low) || failed(high) || low->empty() || high->empty() ||
          low->size() != high->size()) {
        return failure();
      }
      for (size_t index = 0; index < low->size(); ++index) {
        graph.addAction(PhysicalActionKind::MemoryWrite, getDynamicScope(op),
                        {(*low)[index], (*high)[index]}, "interleave_store", 0);
      }
      return success();
    }
    if (auto store = dyn_cast<VMIStoreOp>(op)) {
      return buildStore(store, relation);
    }
    if (isa<VMIStrideStoreOp, VMIScatterOp, VMICompressStoreOp>(op)) {
      if (relation.ports.empty()) {
        return failure();
      }
      auto value =
          buildOperand(op->getOpOperand(0), relation.ports.front().layout);
      if (failed(value) || value->empty()) {
        return failure();
      }
      for (const VMILayoutPortAssignment &port : relation.ports) {
        if (port.kind != VMILayoutPortKind::Operand || port.index == 0) {
          continue;
        }
        if (failed(buildOperand(op->getOpOperand(port.index), port.layout))) {
          return failure();
        }
      }
      StringRef tag = isa<VMIScatterOp>(op)         ? "scatter"
                      : isa<VMICompressStoreOp>(op) ? "compress_store"
                                                    : "stride_store";
      for (PhysicalValue v : *value) {
        graph.addAction(PhysicalActionKind::MemoryWrite, getDynamicScope(op),
                        {v}, tag, 0);
      }
      return success();
    }
    if (auto store = dyn_cast<VMIMaskedStoreOp>(op)) {
      if (relation.ports.size() != 2 ||
          relation.ports[0].kind != VMILayoutPortKind::Operand ||
          relation.ports[1].kind != VMILayoutPortKind::Operand) {
        return failure();
      }
      auto value =
          buildOperand(store.getValueMutable(), relation.ports[0].layout);
      auto mask =
          buildOperand(store.getMaskMutable(), relation.ports[1].layout);
      if (failed(value) || failed(mask) || value->empty() || mask->empty()) {
        return failure();
      }
      auto valueType = dyn_cast<VMIVRegType>(store.getValue().getType());
      auto maskType = dyn_cast<VMIMaskType>(store.getMask().getType());
      if (!valueType || !maskType) {
        return failure();
      }
      SmallVector<PhysicalValue, mlir::pto::kValue4> storeValues = *value;
      SmallVector<PhysicalValue, mlir::pto::kValue4> storeMasks = *mask;
      if (relation.ports[0].layout.isDeinterleaved()) {
        // Dense deinterleaved masked stores canonicalize both data and mask to
        // contiguous physical chunks inside the op before issuing vsts.
        VMILayoutAttr contiguous =
            VMILayoutAttr::getContiguous(op->getContext());
        auto convertedValue = materialize(
            store.getValue(), *value, relation.ports[0].layout, contiguous, op);
        auto convertedMask = materializeMaskLayout(
            *mask, maskType, relation.ports[1].layout, contiguous, op);
        if (failed(convertedValue) || failed(convertedMask)) {
          return failure();
        }
        storeValues = std::move(*convertedValue);
        storeMasks = std::move(*convertedMask);
      }
      if (storeMasks.empty()) {
        return failure();
      }
      for (auto [index, v] : llvm::enumerate(storeValues)) {
        PhysicalValue m = storeMasks[index % storeMasks.size()];
        graph.addAction(PhysicalActionKind::MemoryWrite, getDynamicScope(op),
                        {v, m}, "masked_store", 0);
      }
      return success();
    }
    if (isVMILayoutCastOp(op)) {
      return buildCast(op, relation);
    }
    if (isVMISameLayoutOp(op)) {
      return buildSameLayoutOp(op, relation);
    }
    return failure();
  }

  LogicalResult buildGroupBroadcastLoad(VMIGroupBroadcastLoadOp load,
                                        const VMILayoutOpRelation &relation) {
    if (relation.ports.size() != 1 ||
        relation.ports.front().kind != VMILayoutPortKind::Result) {
      return failure();
    }
    auto type = cast<VMIVRegType>(load.getResult().getType());
    VMILayoutAttr layout = relation.ports.front().layout;
    auto arity = getArity(type, layout);
    if (failed(arity) || *arity <= 0) {
      return failure();
    }
    StringRef tag = relation.directProducer ? "e2b" : "brc";
    auto loaded = graph.addAction(PhysicalActionKind::MemoryRead,
                                  getDynamicScope(load), {}, tag, 1);
    SmallVector<PhysicalValue, mlir::pto::kValue4> aliases(*arity, loaded[0]);
    values[load.getResult()] = std::move(aliases);
    return success();
  }

  LogicalResult buildLoad(VMILoadOp load, const VMILayoutOpRelation &relation) {
    if (relation.ports.size() != 1) {
      return failure();
    }
    auto type = cast<VMIVRegType>(load.getResult().getType());
    VMILayoutAttr layout = relation.ports.front().layout;
    auto arity = getArity(type, layout);
    auto contiguousArity =
        getArity(type, VMILayoutAttr::getContiguous(type.getContext()));
    if (failed(arity) || failed(contiguousArity) || *arity <= 0) {
      return failure();
    }
    Region *scope = getDynamicScope(load);
    SmallVector<PhysicalValue, mlir::pto::kValue4> results;
    bool directDenseSplit =
        isFullPhysicalShape(type) && *arity <= *contiguousArity;
    if (layout.isDeinterleaved() && layout.getFactor() == 2 &&
        directDenseSplit && *arity % 2 == 0) {
      SmallVector<PhysicalValue, mlir::pto::kValue4> lows;
      SmallVector<PhysicalValue, mlir::pto::kValue4> highs;
      for (int64_t group = 0; group < *arity / 2; ++group) {
        auto loaded = graph.addAction(PhysicalActionKind::MemoryRead, scope, {},
                                      "vldsx2", 2);
        lows.push_back(loaded[0]);
        highs.push_back(loaded[1]);
      }
      results.append(lows);
      results.append(highs);
    } else if (layout.isDeinterleaved() && layout.getFactor() == 4 &&
               directDenseSplit && *arity % 4 == 0) {
      for (int64_t group = 0; group < *arity / 4; ++group) {
        auto first = graph.addAction(PhysicalActionKind::MemoryRead, scope, {},
                                     "vldsx2", 2);
        auto second = graph.addAction(PhysicalActionKind::MemoryRead, scope, {},
                                      "vldsx2", 2);
        auto even = graph.addAction(PhysicalActionKind::Deinterleave, scope,
                                    {first[0], second[0]}, "d2", 2);
        auto odd = graph.addAction(PhysicalActionKind::Deinterleave, scope,
                                   {first[1], second[1]}, "d2", 2);
        results.append({even[0], odd[0], even[1], odd[1]});
      }
    } else if (layout.isDeinterleaved()) {
      SmallVector<PhysicalValue, mlir::pto::kValue4> contiguous;
      for (int64_t part = 0; part < *contiguousArity; ++part) {
        contiguous.push_back(graph.addAction(PhysicalActionKind::MemoryRead,
                                             scope, {}, "vlds", 1)[0]);
      }
      auto converted = materialize(
          load.getResult(), contiguous,
          VMILayoutAttr::getContiguous(type.getContext()), layout, load);
      if (failed(converted)) {
        return failure();
      }
      results = std::move(*converted);
    } else {
      for (int64_t part = 0; part < *arity; ++part) {
        results.push_back(graph.addAction(PhysicalActionKind::MemoryRead, scope,
                                          {}, "vlds", 1)[0]);
      }
    }
    values[load.getResult()] = std::move(results);
    layouts[load.getResult()] = layout;
    return success();
  }

  LogicalResult buildStore(VMIStoreOp store,
                           const VMILayoutOpRelation &relation) {
    if (relation.ports.size() != 1) {
      return failure();
    }
    unsigned operandIndex = relation.ports.front().index;
    VMILayoutAttr layout = relation.ports.front().layout;
    auto input = buildOperand(store->getOpOperand(operandIndex), layout);
    auto type = cast<VMIVRegType>(store.getValue().getType());
    auto contiguousArity =
        getArity(type, VMILayoutAttr::getContiguous(type.getContext()));
    if (failed(input) || failed(contiguousArity)) {
      return failure();
    }
    Region *scope = getDynamicScope(store);
    if (layout.isDeinterleaved() && layout.getFactor() == 2 &&
        isFullPhysicalShape(type) && input->size() % 2 == 0 &&
        input->size() <= static_cast<size_t>(*contiguousArity)) {
      for (size_t part = 0; part < input->size() / 2; ++part) {
        graph.addAction(PhysicalActionKind::MemoryWrite, scope,
                        {(*input)[part], (*input)[input->size() / 2 + part]},
                        "vstsx2", 0);
      }
      return success();
    }
    if (layout.isContiguous() && layout.getLaneStride() > 1) {
      for (PhysicalValue value : *input) {
        graph.addAction(PhysicalActionKind::MemoryWrite, scope, {value}, "vsts",
                        0);
      }
      return success();
    }
    auto contiguous =
        materialize(store.getValue(), *input, layout,
                    VMILayoutAttr::getContiguous(type.getContext()), store);
    if (failed(contiguous)) {
      return failure();
    }
    for (PhysicalValue value : *contiguous) {
      graph.addAction(PhysicalActionKind::MemoryWrite, scope, {value}, "vsts",
                      0);
    }
    return success();
  }

  LogicalResult buildSameLayoutOp(Operation *op,
                                  const VMILayoutOpRelation &relation) {
    VMILayoutAttr candidateLayout;
    for (const VMILayoutPortAssignment &port : relation.ports) {
      if (port.kind == VMILayoutPortKind::Result) {
        candidateLayout = port.layout;
        break;
      }
    }
    if (failed(VMILayoutSupport().getSameLayoutRelationSupport(op,
                                                               candidateLayout))) {
      return failure();
    }
    SmallVector<SmallVector<PhysicalValue, mlir::pto::kValue4>,
                mlir::pto::kValue4>
        inputs;
    SmallVector<std::pair<OpResult, VMILayoutAttr>, mlir::pto::kValue4>
        resultsToBuild;
    for (const VMILayoutPortAssignment &port : relation.ports) {
      if (port.kind == VMILayoutPortKind::Operand) {
        auto input = buildOperand(op->getOpOperand(port.index), port.layout);
        if (failed(input)) {
          return failure();
        }
        inputs.push_back(std::move(*input));
      } else {
        resultsToBuild.emplace_back(cast<OpResult>(op->getResult(port.index)),
                                    port.layout);
      }
    }
    if (resultsToBuild.empty()) {
      return failure();
    }
    auto resultType = resultsToBuild.front().first.getType();
    VMILayoutAttr resultLayout = resultsToBuild.front().second;
    if (!resultLayout) {
      return failure();
    }
    int64_t arity = 0;
    int64_t inputArity = 0;
    if (isa<VMIMaskType>(resultType)) {
      arity = 1;
      for (const auto &input : inputs) {
        inputArity = std::max<int64_t>(inputArity, input.size());
      }
    } else if (auto type = dyn_cast<VMIVRegType>(resultType)) {
      auto physicalArity = getArity(type, resultLayout);
      if (failed(physicalArity)) {
        return failure();
      }
      arity = *physicalArity;
      inputArity = arity;
    } else {
      return failure();
    }
    if (arity <= 0 || llvm::any_of(inputs, [&](ArrayRef<PhysicalValue> input) {
          return input.size() != 1 &&
                 input.size() != static_cast<size_t>(inputArity);
        })) {
      return failure();
    }
    for (const auto &[result, layout] : resultsToBuild) {
      if (!layout) {
        return failure();
      }
      auto type = dyn_cast<VMIVRegType>(result.getType());
      int64_t resultArity = arity;
      if (isa<VMIMaskType>(result.getType())) {
        resultArity = 1;
      } else if (type) {
        auto physicalArity = getArity(type, layout);
        if (failed(physicalArity) || *physicalArity <= 0) {
          return failure();
        }
        resultArity = *physicalArity;
      }
      if (resultArity != 1 && resultArity != arity) {
        return failure();
      }
      SmallVector<PhysicalValue, mlir::pto::kValue4> resultValues;
      for (int64_t part = 0; part < resultArity; ++part) {
        SmallVector<PhysicalValue, mlir::pto::kValue4> partInputs;
        for (const auto &input : inputs) {
          partInputs.push_back(
              input.size() == 1
                  ? input.front()
                  : input[static_cast<size_t>(part) % input.size()]);
        }
        resultValues.push_back(
            graph.addAction(PhysicalActionKind::Semantic, getDynamicScope(op),
                            partInputs, op->getName().getStringRef(), 1)[0]);
      }
      values[result] = std::move(resultValues);
      layouts[result] = layout;
    }
    return success();
  }

  LogicalResult buildCast(Operation *op, const VMILayoutOpRelation &relation) {
    if (relation.ports.size() != 2 ||
        relation.ports[0].kind != VMILayoutPortKind::Operand ||
        relation.ports[1].kind != VMILayoutPortKind::Result) {
      return failure();
    }
    VMILayoutAttr sourceLayout = relation.ports[0].layout;
    VMILayoutAttr resultLayout = relation.ports[1].layout;
    auto source = buildOperand(op->getOpOperand(0), sourceLayout);
    auto sourceType = cast<VMIVRegType>(op->getOperand(0).getType());
    auto resultType = cast<VMIVRegType>(op->getResult(0).getType());
    auto resultArity = getArity(resultType, resultLayout);
    if (failed(source) || failed(resultArity) || source->empty() ||
        *resultArity <= 0) {
      return failure();
    }
    unsigned sourceBits =
        pto::getPTOStorageElemBitWidth(sourceType.getElementType());
    unsigned resultBits =
        pto::getPTOStorageElemBitWidth(resultType.getElementType());
    SmallVector<PhysicalValue, mlir::pto::kValue4> results;
    Region *scope = getDynamicScope(op);
    if (sourceBits < resultBits) {
      if (*resultArity % source->size() != 0) {
        return failure();
      }
      size_t outputsPerSource = *resultArity / source->size();
      for (size_t output = 0; output < outputsPerSource; ++output) {
        for (PhysicalValue input : *source) {
          std::string tag = (Twine("ext:") + Twine(output)).str();
          auto converted = graph.addAction(PhysicalActionKind::Semantic, scope,
                                           {input}, tag, 1);
          results.push_back(converted[0]);
        }
      }
    } else if (sourceBits > resultBits) {
      if (source->size() < static_cast<size_t>(*resultArity)) {
        return failure();
      }
      for (int64_t result = 0; result < *resultArity; ++result) {
        size_t begin = result * source->size() / *resultArity;
        size_t end = (result + 1) * source->size() / *resultArity;
        if (begin == end) {
          return failure();
        }
        SmallVector<PhysicalValue, mlir::pto::kValue4> partials;
        for (size_t sourceIndex = begin; sourceIndex < end; ++sourceIndex) {
          std::string tag =
              (Twine("trunc:") + Twine(sourceIndex - begin)).str();
          partials.push_back(graph.addAction(PhysicalActionKind::Semantic,
                                             scope, {(*source)[sourceIndex]},
                                             tag)[0]);
        }
        PhysicalValue merged = partials.front();
        for (PhysicalValue partial : llvm::drop_begin(partials)) {
          merged = graph.addAction(PhysicalActionKind::Merge, scope,
                                   {merged, partial}, "vor")[0];
        }
        results.push_back(merged);
      }
    } else {
      if (source->size() != static_cast<size_t>(*resultArity)) {
        return failure();
      }
      for (PhysicalValue input : *source) {
        results.push_back(graph.addAction(PhysicalActionKind::Semantic, scope,
                                          {input}, "cast")[0]);
      }
    }
    values[op->getResult(0)] = std::move(results);
    return success();
  }

  FailureOr<SmallVector<PhysicalValue, mlir::pto::kValue4>>
  addMaskActions(PhysicalValue source, PhysicalActionKind kind, int64_t count,
                 Region *scope, StringRef tag) {
    if (count < 0) {
      return failure();
    }
    PhysicalValue current = source;
    for (int64_t index = 0; index < count; ++index) {
      current = graph.addAction(kind, scope, {current}, tag, 1)[0];
    }
    return SmallVector<PhysicalValue, mlir::pto::kValue4>{current};
  }

  FailureOr<SmallVector<PhysicalValue, mlir::pto::kValue4>>
  materializeMaskPackTree(PhysicalValue source, int64_t sourceArity,
                          int64_t levels, Region *scope) {
    if (sourceArity <= 0 || levels <= 0) {
      return failure();
    }
    PhysicalValue current = source;
    int64_t carriers = sourceArity;
    for (int64_t level = 0; level < levels; ++level) {
      auto packed = addMaskActions(current, PhysicalActionKind::Pack, carriers,
                                   scope, "mask-pack");
      if (failed(packed)) {
        return failure();
      }
      current = packed->front();
      auto merged = addMaskActions(current, PhysicalActionKind::Merge,
                                   carriers / 2, scope, "mask-merge");
      if (failed(merged)) {
        return failure();
      }
      current = merged->front();
      carriers = (carriers + 1) / 2;
    }
    return SmallVector<PhysicalValue, mlir::pto::kValue4>{current};
  }

  FailureOr<SmallVector<PhysicalValue, mlir::pto::kValue4>>
  materializeMaskLayout(ArrayRef<PhysicalValue> source, VMIMaskType type,
                        VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
                        Operation *placement) {
    if (source.empty() || !type || !sourceLayout || !resultLayout) {
      return failure();
    }
    if (sourceLayout == resultLayout) {
      return SmallVector<PhysicalValue, mlir::pto::kValue4>(source);
    }
    auto sourceArity = getMaskArity(type, sourceLayout);
    auto resultArity = getMaskArity(type, resultLayout);
    if (failed(sourceArity) || failed(resultArity) || *sourceArity <= 0 ||
        *resultArity <= 0) {
      return failure();
    }

    Region *scope = getDynamicScope(placement);
    PhysicalValue input = source.front();
    bool sourceContiguous =
        sourceLayout.isContiguous() && sourceLayout.getLaneStride() == 1;
    bool resultContiguous =
        resultLayout.isContiguous() && resultLayout.getLaneStride() == 1;
    if (sourceContiguous && resultLayout.isDeinterleaved()) {
      int64_t factor = resultLayout.getFactor();
      if ((factor != 2 && factor != 4) || *resultArity % factor != 0) {
        return failure();
      }
      int64_t actions = factor == 2 ? *resultArity / 2 : *resultArity;
      return addMaskActions(input, PhysicalActionKind::Deinterleave, actions,
                            scope, "mask-deinterleave");
    }
    if (sourceLayout.isDeinterleaved() && resultContiguous) {
      int64_t factor = sourceLayout.getFactor();
      if ((factor != 2 && factor != 4) || *sourceArity % factor != 0) {
        return failure();
      }
      int64_t actions =
          factor == 2 ? (*resultArity + 1) / 2
                      : 2 * ((*resultArity + 3) / 4) + (*resultArity + 1) / 2;
      return addMaskActions(input, PhysicalActionKind::Interleave, actions,
                            scope, "mask-interleave");
    }
    if (sourceContiguous && resultLayout.isContiguous()) {
      int64_t stride = resultLayout.getLaneStride();
      if (stride != 2 && stride != 4) {
        return failure();
      }
      int64_t levels = stride == 2 ? 1 : 2;
      return addMaskActions(input, PhysicalActionKind::Unpack,
                            *resultArity * levels, scope, "mask-unpack");
    }
    // A lane-strided contiguous mask is still a contiguous carrier family;
    // only its physical packing differs.  Use the same pack tree as the
    // dense data materializer when normalizing it to unit stride.
    if (sourceLayout.isContiguous() && resultContiguous) {
      int64_t stride = sourceLayout.getLaneStride();
      if (stride != 2 && stride != 4) {
        return failure();
      }
      return materializeMaskPackTree(input, *sourceArity, stride == 2 ? 1 : 2,
                                     scope);
    }
    bool blockForwarding =
        (sourceContiguous && resultLayout.isBlockDeinterleaved()) ||
        (sourceLayout.isBlockDeinterleaved() && resultContiguous);
    if (blockForwarding) {
      return SmallVector<PhysicalValue, mlir::pto::kValue4>(source);
    }
    if (sourceLayout.isContiguous() || resultLayout.isContiguous()) {
      return failure();
    }
    VMILayoutAttr contiguous = VMILayoutAttr::getContiguous(type.getContext());
    auto first = materializeMaskLayout(source, type, sourceLayout, contiguous,
                                       placement);
    if (failed(first)) {
      return failure();
    }
    return materializeMaskLayout(*first, type, contiguous, resultLayout,
                                 placement);
  }

  FailureOr<SmallVector<PhysicalValue, mlir::pto::kValue4>>
  groupSlotsToGroupSlots(ArrayRef<PhysicalValue> source,
                         VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
                         Region *scope) {
    if (!sourceLayout.isGroupSlots() || !resultLayout.isGroupSlots() ||
        sourceLayout.getNumGroups() != resultLayout.getNumGroups() ||
        sourceLayout.getSlots() != mlir::pto::kValue8 ||
        resultLayout.getSlots() != mlir::pto::kValue8) {
      return failure();
    }
    int64_t sourceStride = sourceLayout.getLaneStride();
    int64_t resultStride = resultLayout.getLaneStride();
    if ((sourceStride != 1 && sourceStride != 2 && sourceStride != 4) ||
        (resultStride != 1 && resultStride != 2 && resultStride != 4)) {
      return failure();
    }
    PhysicalActionKind kind = sourceStride < resultStride
                                  ? PhysicalActionKind::Unpack
                                  : PhysicalActionKind::Pack;
    SmallVector<PhysicalValue, mlir::pto::kValue4> results;
    for (PhysicalValue input : source) {
      PhysicalValue current = input;
      int64_t currentStride = sourceStride;
      while (currentStride != resultStride) {
        current =
            graph.addAction(kind, scope, {current}, "group-slot-stride", 1)[0];
        currentStride =
            sourceStride < resultStride ? currentStride * 2 : currentStride / 2;
      }
      results.push_back(current);
    }
    return results;
  }

  FailureOr<SmallVector<PhysicalValue, mlir::pto::kValue4>>
  materialize(Value sourceValue, ArrayRef<PhysicalValue> source,
              VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
              Operation *placement) {
    auto type = dyn_cast<VMIVRegType>(sourceValue.getType());
    if (!type || !sourceLayout || !resultLayout) {
      return failure();
    }
    auto sourceArity = getArity(type, sourceLayout);
    auto resultArity = getArity(type, resultLayout);
    if (failed(sourceArity) || failed(resultArity) || *sourceArity <= 0 ||
        *resultArity <= 0 ||
        source.size() != static_cast<size_t>(*sourceArity)) {
      return failure();
    }
    if (sourceLayout == resultLayout) {
      return SmallVector<PhysicalValue, mlir::pto::kValue4>(source);
    }
    VMILayoutSupport supports;
    auto sourceType =
        VMIVRegType::get(type.getContext(), type.getElementCount(),
                         type.getElementType(), sourceLayout);
    auto resultType =
        VMIVRegType::get(type.getContext(), type.getElementCount(),
                         type.getElementType(), resultLayout);
    auto ensureFact = supports.getEnsureLayoutFact(sourceType, resultType);
    if (succeeded(ensureFact) && ensureFact->forwardsPhysicalParts) {
      if (*sourceArity != *resultArity) {
        return failure();
      }
      return SmallVector<PhysicalValue, mlir::pto::kValue4>(source);
    }
    Region *scope = getDynamicScope(placement);
    if (sourceLayout.isGroupSlots() && resultLayout.isGroupSlots()) {
      if (*sourceArity != *resultArity) {
        return failure();
      }
      return groupSlotsToGroupSlots(source, sourceLayout, resultLayout, scope);
    }
    if (sourceLayout.isDeinterleaved() && resultLayout.isContiguous() &&
        resultLayout.getLaneStride() == 1) {
      return deinterleavedToContiguous(source, sourceLayout.getFactor(),
                                       static_cast<size_t>(*resultArity),
                                       scope);
    }
    if (sourceLayout.isContiguous() && sourceLayout.getLaneStride() == 1 &&
        resultLayout.isDeinterleaved()) {
      return contiguousToDeinterleaved(source, resultLayout.getFactor(),
                                       static_cast<size_t>(*resultArity),
                                       scope);
    }
    // d4->d2 has a direct pairwise shuffle realization.
    if (sourceLayout.isDeinterleaved() &&
        resultLayout.isDeinterleaved() &&
        sourceLayout.getLaneStride() == 1 &&
        resultLayout.getLaneStride() == 1 &&
        sourceLayout.getFactor() == 4 && resultLayout.getFactor() == 2) {
      if (source.size() % 4 != 0 || *resultArity != *sourceArity) {
        return failure();
      }
      size_t chunks = source.size() / 4;
      SmallVector<PhysicalValue, mlir::pto::kValue4> results;
      results.reserve(source.size());
      SmallVector<PhysicalValue> lows, highs;
      for (size_t chunk = 0; chunk < chunks; ++chunk) {
          auto even = graph.addAction(
              PhysicalActionKind::Interleave,
              scope, {source[chunk], source[chunks * 2 + chunk]},
              "d4-d2-even", 2);
          auto odd = graph.addAction(
              PhysicalActionKind::Interleave,
              scope, {source[chunks + chunk], source[chunks * 3 + chunk]},
              "d4-d2-odd", 2);
          lows.push_back(even[0]);
          lows.push_back(odd[0]);
          highs.push_back(even[1]);
          highs.push_back(odd[1]);
      }
      results.append(lows);
      results.append(highs);
      return results;
    }
    // d2->d4 is the inverse of the direct d4->d2 pairwise shuffle above.
    if (sourceLayout.isDeinterleaved() &&
        resultLayout.isDeinterleaved() &&
        sourceLayout.getLaneStride() == 1 &&
        resultLayout.getLaneStride() == 1 &&
        sourceLayout.getFactor() == 2 && resultLayout.getFactor() == 4) {
      if (source.size() % 4 != 0 || *resultArity != *sourceArity) {
        return failure();
      }
      size_t chunks = source.size() / 4;
      SmallVector<PhysicalValue, mlir::pto::kValue4> results;
      results.reserve(source.size());
      SmallVector<PhysicalValue> part0, part1, part2, part3;
      for (size_t chunk = 0; chunk < chunks; ++chunk) {
        size_t lowIndex = 2 * chunk;
        size_t highIndex = 2 * chunks + lowIndex;
        auto even = graph.addAction(PhysicalActionKind::Deinterleave, scope,
                                    {source[lowIndex], source[highIndex]},
                                    "d2-d4-even", 2);
        auto odd = graph.addAction(PhysicalActionKind::Deinterleave, scope,
                                   {source[lowIndex + 1],
                                    source[highIndex + 1]},
                                   "d2-d4-odd", 2);
        part0.push_back(even[0]);
        part1.push_back(odd[0]);
        part2.push_back(even[1]);
        part3.push_back(odd[1]);
      }
      results.append(part0);
      results.append(part1);
      results.append(part2);
      results.append(part3);
      return results;
    }
    if (sourceLayout.isContiguous() && sourceLayout.getLaneStride() == 1 &&
        resultLayout.isContiguous() && resultLayout.getLaneStride() > 1) {
      return contiguousToLaneStride(source, resultLayout.getLaneStride(),
                                    static_cast<size_t>(*resultArity), scope);
    }
    if (sourceLayout.isContiguous() && sourceLayout.getLaneStride() > 1 &&
        resultLayout.isContiguous() && resultLayout.getLaneStride() == 1) {
      return laneStrideToContiguous(source, sourceLayout.getLaneStride(),
                                    scope);
    }
    bool deint2ToLaneStride2 =
        sourceLayout.isDeinterleaved() && sourceLayout.getFactor() == 2 &&
        sourceLayout.getLaneStride() == 1 && resultLayout.isContiguous() &&
        resultLayout.getLaneStride() == 2;
    bool laneStride2ToDeint2 =
        sourceLayout.isContiguous() && sourceLayout.getLaneStride() == 2 &&
        resultLayout.isDeinterleaved() && resultLayout.getFactor() == 2 &&
        resultLayout.getLaneStride() == 1;
    bool laneStride2ToLaneStride4 =
        sourceLayout.isContiguous() && sourceLayout.getLaneStride() == 2 &&
        resultLayout.isContiguous() && resultLayout.getLaneStride() == 4;
    bool laneStride4ToLaneStride2 =
        sourceLayout.isContiguous() && sourceLayout.getLaneStride() == 4 &&
        resultLayout.isContiguous() && resultLayout.getLaneStride() == 2;
    if (deint2ToLaneStride2 || laneStride2ToDeint2 ||
        laneStride2ToLaneStride4 || laneStride4ToLaneStride2) {
      VMILayoutAttr contiguous =
          VMILayoutAttr::getContiguous(type.getContext());
      auto first =
          materialize(sourceValue, source, sourceLayout, contiguous, placement);
      if (failed(first)) {
        return failure();
      }
      return materialize(sourceValue, *first, contiguous, resultLayout,
                         placement);
    }
    // Do not recurse through a contiguous intermediate when either side is
    // already contiguous.  Unsupported layout pairs (for example group-slot
    // to deinterleaved) would otherwise revisit the same pair indefinitely.
    if (sourceLayout.isContiguous() || resultLayout.isContiguous()) {
      return failure();
    }
    VMILayoutAttr contiguous = VMILayoutAttr::getContiguous(type.getContext());
    auto first =
        materialize(sourceValue, source, sourceLayout, contiguous, placement);
    if (failed(first)) {
      return failure();
    }
    return materialize(sourceValue, *first, contiguous, resultLayout,
                       placement);
  }

  FailureOr<SmallVector<PhysicalValue, mlir::pto::kValue4>>
  deinterleavedToContiguous(ArrayRef<PhysicalValue> source, int64_t factor,
                            size_t resultCount, Region *scope) {
    if (factor == 2 && source.size() % 2 == 0 && resultCount <= source.size()) {
      SmallVector<PhysicalValue, mlir::pto::kValue4> results;
      size_t groups = source.size() / 2;
      for (size_t group = 0; group < groups && results.size() < resultCount;
           ++group) {
        auto interleaved =
            graph.addAction(PhysicalActionKind::Interleave, scope,
                            {source[group], source[groups + group]}, "d2", 2);
        for (PhysicalValue value : interleaved) {
          if (results.size() == resultCount) {
            break;
          }
          results.push_back(value);
        }
      }
      return results;
    }
    if (factor != 4 || source.empty() || resultCount > source.size()) {
      return failure();
    }
    SmallVector<PhysicalValue, mlir::pto::kValue4> results;
    size_t groups = (resultCount + 3) / 4;
    size_t base = source.size() / 4;
    size_t remainder = source.size() % 4;
    auto getSource = [&](size_t part, size_t group) {
      size_t count = base + (part < remainder ? 1 : 0);
      size_t offset = part * base + std::min(part, remainder);
      return group < count ? source[offset + group] : source.back();
    };
    for (size_t group = 0; group < groups && results.size() < resultCount;
         ++group) {
      PhysicalValue p0 = getSource(0, group);
      PhysicalValue p1 = getSource(1, group);
      PhysicalValue p2 = getSource(2, group);
      PhysicalValue p3 = getSource(3, group);
      auto even = graph.addAction(PhysicalActionKind::Interleave, scope,
                                  {p0, p2}, "d4-even", 2);
      auto odd = graph.addAction(PhysicalActionKind::Interleave, scope,
                                 {p1, p3}, "d4-odd", 2);
      auto low = graph.addAction(PhysicalActionKind::Interleave, scope,
                                 {even[0], odd[0]}, "d4-low", 2);
      for (PhysicalValue value : low) {
        if (results.size() == resultCount) {
          break;
        }
        results.push_back(value);
      }
      if (results.size() < resultCount) {
        auto high = graph.addAction(PhysicalActionKind::Interleave, scope,
                                    {even[1], odd[1]}, "d4-high", 2);
        for (PhysicalValue value : high) {
          if (results.size() == resultCount) {
            break;
          }
          results.push_back(value);
        }
      }
    }
    return results;
  }

  FailureOr<SmallVector<PhysicalValue, mlir::pto::kValue4>>
  contiguousToDeinterleaved(ArrayRef<PhysicalValue> source, int64_t factor,
                            size_t resultCount, Region *scope) {
    if (factor == 2 && resultCount % 2 == 0 && !source.empty() &&
        source.size() <= resultCount) {
      SmallVector<PhysicalValue, mlir::pto::kValue4> lows;
      SmallVector<PhysicalValue, mlir::pto::kValue4> highs;
      for (size_t part = 0; part < resultCount; part += 2) {
        if (part >= source.size()) {
          return failure();
        }
        PhysicalValue higher =
            part + 1 < source.size() ? source[part + 1] : source[part];
        auto split = graph.addAction(PhysicalActionKind::Deinterleave, scope,
                                     {source[part], higher}, "d2", 2);
        lows.push_back(split[0]);
        highs.push_back(split[1]);
      }
      lows.append(highs);
      return lows;
    }
    if (factor != 4 || resultCount == 0 || resultCount % 4 != 0 ||
        source.empty() || source.size() > resultCount) {
      return failure();
    }
    SmallVector<PhysicalValue, mlir::pto::kValue4> part0;
    SmallVector<PhysicalValue, mlir::pto::kValue4> part1;
    SmallVector<PhysicalValue, mlir::pto::kValue4> part2;
    SmallVector<PhysicalValue, mlir::pto::kValue4> part3;
    size_t groups = resultCount / 4;
    for (size_t group = 0; group < groups; ++group) {
      auto getSource = [&](size_t offset) {
        return source[std::min(4 * group + offset, source.size() - 1)];
      };
      PhysicalValue s0 = getSource(0);
      PhysicalValue s1 = getSource(1);
      PhysicalValue s2 = getSource(2);
      PhysicalValue s3 = getSource(3);
      auto low = graph.addAction(PhysicalActionKind::Deinterleave, scope,
                                 {s0, s1}, "d4-low", 2);
      auto high = graph.addAction(PhysicalActionKind::Deinterleave, scope,
                                  {s2, s3}, "d4-high", 2);
      auto even = graph.addAction(PhysicalActionKind::Deinterleave, scope,
                                  {low[0], high[0]}, "d4-even", 2);
      auto odd = graph.addAction(PhysicalActionKind::Deinterleave, scope,
                                 {low[1], high[1]}, "d4-odd", 2);
      part0.push_back(even[0]);
      part1.push_back(odd[0]);
      part2.push_back(even[1]);
      part3.push_back(odd[1]);
    }
    part0.append(part1);
    part0.append(part2);
    part0.append(part3);
    return part0;
  }

  FailureOr<SmallVector<PhysicalValue, mlir::pto::kValue4>>
  contiguousToLaneStride(ArrayRef<PhysicalValue> source, int64_t stride,
                         size_t resultCount, Region *scope) {
    if ((stride != 2 && stride != 4) || source.empty() || resultCount == 0 ||
        (resultCount + stride - 1) / stride != source.size()) {
      return failure();
    }
    SmallVector<PhysicalValue, mlir::pto::kValue4> results;
    results.reserve(resultCount);
    for (size_t resultIndex = 0; resultIndex < resultCount; ++resultIndex) {
      int64_t part = resultIndex % stride;
      PhysicalValue current = source[resultIndex / stride];
      int64_t levels = stride == 4 ? 2 : 1;
      for (int64_t level = 0; level < levels; ++level) {
        int64_t selector =
            stride == 4 ? (level == 0 ? part / 2 : part % 2) : part;
        current = graph.addAction(PhysicalActionKind::Unpack, scope, {current},
                                  (Twine("level:") + Twine(level) +
                                   Twine(":selector:") + Twine(selector))
                                      .str(),
                                  1)[0];
      }
      results.push_back(current);
    }
    return results;
  }

  FailureOr<SmallVector<PhysicalValue, mlir::pto::kValue4>>
  laneStrideToContiguous(ArrayRef<PhysicalValue> source, int64_t stride,
                         Region *scope) {
    if ((stride != 2 && stride != 4) || source.empty()) {
      return failure();
    }
    SmallVector<PhysicalValue, mlir::pto::kValue4> results;
    for (size_t begin = 0; begin < source.size(); begin += stride) {
      SmallVector<PhysicalValue, mlir::pto::kValue4> level(
          source.slice(begin, std::min<size_t>(stride, source.size() - begin)));
      int64_t levels = stride == 4 ? 2 : 1;
      for (int64_t depth = 0; depth < levels; ++depth) {
        SmallVector<PhysicalValue, mlir::pto::kValue4> next;
        for (size_t index = 0; index < level.size(); index += 2) {
          PhysicalValue low = graph.addAction(PhysicalActionKind::Pack, scope,
                                              {level[index]}, "lower")[0];
          if (index + 1 == level.size()) {
            next.push_back(low);
            continue;
          }
          PhysicalValue high = graph.addAction(PhysicalActionKind::Pack, scope,
                                               {level[index + 1]}, "higher")[0];
          next.push_back(graph.addAction(PhysicalActionKind::Merge, scope,
                                         {low, high}, "vor")[0]);
        }
        level = std::move(next);
      }
      if (level.size() != 1) {
        return failure();
      }
      results.push_back(level.front());
    }
    return results;
  }

  const VMILayoutPlan &plan;
  DenseMap<Operation *, VMILayoutOpRelation> selectedRelations;
  DenseMap<Value, SmallVector<PhysicalValue, mlir::pto::kValue4>> values;
  DenseMap<Value, VMILayoutAttr> layouts;
  DenseSet<Operation *> builtOps;
  DenseSet<Operation *> buildingOps;
  SmallVector<PhysicalStateStorage::SharedMaterialization,
              mlir::pto::kValue4>
      sharedMaterializations;
  PhysicalGraph graph;
};

static void rememberCanonicalId(DenseMap<unsigned, unsigned> &canonicalIds,
                                unsigned physicalId) {
  if (!canonicalIds.contains(physicalId)) {
    canonicalIds[physicalId] = canonicalIds.size();
  }
}

static FailureOr<std::string>
buildContinuationKey(const PhysicalStateStorage &storage,
                     ArrayRef<Operation *> remainingOps) {
  constexpr unsigned maxObservableValues = 256;
  DenseSet<Value> seenValues;
  SmallVector<Value, mlir::pto::kValue16> liveValues;
  for (Operation *op : remainingOps) {
    for (Value operand : op->getOperands()) {
      if (storage.values.contains(operand) &&
          seenValues.insert(operand).second) {
        liveValues.push_back(operand);
      }
    }
  }

  DenseMap<unsigned, unsigned> canonicalIds;
  for (Value value : liveValues) {
    for (PhysicalValue part : storage.values.lookup(value)) {
      rememberCanonicalId(canonicalIds, part.id);
    }
  }
  if (canonicalIds.size() > maxObservableValues) {
    return failure();
  }

  std::string key;
  llvm::raw_string_ostream os(key);
  for (Value value : liveValues) {
    VMILayoutAttr layout = storage.layouts.lookup(value);
    if (!layout) {
      return failure();
    }
    os << "V{" << layout << ':';
    for (PhysicalValue part : storage.values.lookup(value)) {
      os << canonicalIds.lookup(part.id) << ',';
    }
    os << '}';
  }
  return os.str();
}

} // namespace

struct VMILayoutPhysicalState::Impl {
  PhysicalStateStorage storage;
};

VMILayoutPhysicalState::VMILayoutPhysicalState()
    : impl(std::make_shared<Impl>()) {}

VMILayoutPhysicalState::VMILayoutPhysicalState(const VMILayoutPhysicalState &) =
    default;
VMILayoutPhysicalState::VMILayoutPhysicalState(
    VMILayoutPhysicalState &&) noexcept = default;
VMILayoutPhysicalState &
VMILayoutPhysicalState::operator=(const VMILayoutPhysicalState &) = default;
VMILayoutPhysicalState &
VMILayoutPhysicalState::operator=(VMILayoutPhysicalState &&) noexcept = default;
VMILayoutPhysicalState::~VMILayoutPhysicalState() = default;

FailureOr<VMILayoutPhysicalState>
mlir::pto::appendVMILayoutPhysicalRelation(const VMILayoutPhysicalState &state,
                                           const VMILayoutOpRelation &relation,
                                           const VMILayoutPlan &partialPlan) {
  if (!state.impl || !relation.op) {
    return failure();
  }
  PlanGraphBuilder builder(state.impl->storage, relation, partialPlan);
  auto storage = builder.append(relation);
  if (failed(storage)) {
    return failure();
  }
  VMILayoutPhysicalState result;
  result.impl->storage = std::move(*storage);
  return result;
}

FailureOr<std::string>
mlir::pto::getVMILayoutContinuationKey(const VMILayoutPhysicalState &state,
                                       ArrayRef<Operation *> remainingOps) {
  if (!state.impl) {
    return failure();
  }
  return buildContinuationKey(state.impl->storage, remainingOps);
}

VMILayoutScopeCost
mlir::pto::getVMILayoutPhysicalCost(const VMILayoutPhysicalState &state) {
  if (state.impl) {
    return state.impl->storage.graph.getCost();
  }
  VMILayoutScopeCost empty;
  return empty;
}

FailureOr<VMILayoutScopeCost> mlir::pto::evaluateVMILayoutPlanCost(
    ArrayRef<VMILayoutOpRelation> selectedRelations,
    const VMILayoutPlan &plan) {
  return PlanGraphBuilder(selectedRelations, plan).build();
}

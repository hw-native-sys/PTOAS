// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMILayoutAssignment.cpp - Assign VMI layouts ----------------------===//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/VMIUtils.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/VMILayoutPropagation.h"
#include "PTO/Transforms/VMILayoutSupport.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VMILAYOUTASSIGNMENT
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

struct DataNode {
  Value value;
  VMIVRegType type;
  unsigned parent = 0;
  VMILayoutAttr naturalLayout;
  VMILayoutAttr preferredLayout;
};

struct MaskNode {
  Value value;
  VMIMaskType type;
  unsigned parent = 0;
  VMILayoutAttr requestedLayout;
};

enum class DataLayoutSeedPhase {
  Explicit,
  SeedStart,
  GroupLoad = SeedStart,
  Reduce,
  GroupSlotLoad,
  GroupBroadcast,
  GroupBroadcastLoad,
  Cast,
  WeakReduce,
  Store,
  Other,
  SeedEnd,
};

struct DataLayoutSeed {
  Value value;
  VMILayoutAttr layout;
  DataLayoutSeedPhase phase = DataLayoutSeedPhase::Other;
};

struct DataUseRequest {
  OpOperand *operand;
  VMILayoutAttr layout;
  bool late = false;
  DataLayoutSeedPhase phase = DataLayoutSeedPhase::Other;
};

struct MaskUseRequest {
  OpOperand *operand;
  VMILayoutAttr layout;
  DataLayoutSeedPhase phase = DataLayoutSeedPhase::Other;
};

struct GroupStoreUseRequest {
  VMIGroupStoreOp store;
};

static std::optional<int64_t> getConstantIndexValue(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
    return constant.value();
  if (auto constant = value.getDefiningOp<arith::ConstantOp>())
    if (auto integerAttr = dyn_cast<IntegerAttr>(constant.getValue()))
      return integerAttr.getInt();
  return std::nullopt;
}

static bool isLane0SplatShuffle(VMIShuffleOp op) {
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  ArrayRef<int64_t> indices = op.getIndices();
  return sourceType.getElementCount() == 1 && !indices.empty() &&
         llvm::all_of(indices, [](int64_t index) { return index == 0; });
}

bool containsVMIType(Type type) {
  if (isa<VMIVRegType, VMIMaskType>(type))
    return true;
  if (auto functionType = dyn_cast<FunctionType>(type)) {
    return llvm::any_of(functionType.getInputs(),
                        [](Type input) { return containsVMIType(input); }) ||
           llvm::any_of(functionType.getResults(),
                        [](Type result) { return containsVMIType(result); });
  }
  if (auto shapedType = dyn_cast<ShapedType>(type))
    return containsVMIType(shapedType.getElementType());
  return false;
}

struct LayoutSolver {
  explicit LayoutSolver(ModuleOp module)
      : module(module), ctx(module.getContext()) {}

  unsigned addDataValue(Value value) {
    auto type = dyn_cast<VMIVRegType>(value.getType());
    if (!type)
      return ~0u;
    auto [it, inserted] = dataIds.try_emplace(value, dataNodes.size());
    if (inserted) {
      dataNodes.push_back(
          DataNode{value, type, it->second, type.getLayoutAttr(), {}});
      if (type.getLayoutAttr())
        dataLayoutSeeds.push_back(DataLayoutSeed{
            value, type.getLayoutAttr(), DataLayoutSeedPhase::Explicit});
    }
    return it->second;
  }

  unsigned addMaskValue(Value value) {
    auto type = dyn_cast<VMIMaskType>(value.getType());
    if (!type)
      return ~0u;
    auto [it, inserted] = maskIds.try_emplace(value, maskNodes.size());
    if (inserted)
      maskNodes.push_back(
          MaskNode{value, type, it->second, type.getLayoutAttr()});
    return it->second;
  }

  unsigned find(unsigned id) {
    if (dataNodes[id].parent == id)
      return id;
    dataNodes[id].parent = find(dataNodes[id].parent);
    return dataNodes[id].parent;
  }

  unsigned findMask(unsigned id) {
    if (maskNodes[id].parent == id)
      return id;
    maskNodes[id].parent = findMask(maskNodes[id].parent);
    return maskNodes[id].parent;
  }

  LogicalResult unite(Value lhs, Value rhs, Operation *op) {
    (void)op;
    addDataValue(lhs);
    addDataValue(rhs);
    return success();
  }

  LogicalResult uniteDataEquivalent(Value lhs, Value rhs, Operation *op) {
    unsigned lhsId = addDataValue(lhs);
    unsigned rhsId = addDataValue(rhs);
    if (lhsId == ~0u || rhsId == ~0u)
      return success();
    unsigned lhsRoot = find(lhsId);
    unsigned rhsRoot = find(rhsId);
    if (lhsRoot == rhsRoot)
      return success();

    DataNode &lhsNode = dataNodes[lhsRoot];
    DataNode &rhsNode = dataNodes[rhsRoot];
    if (lhsNode.naturalLayout && rhsNode.naturalLayout &&
        lhsNode.naturalLayout != rhsNode.naturalLayout)
      return op->emitError()
             << kVMIDiagLayoutContractPrefix << "conflicting natural layouts "
             << lhsNode.naturalLayout << " and " << rhsNode.naturalLayout;
    if (lhsNode.preferredLayout && rhsNode.preferredLayout &&
        lhsNode.preferredLayout != rhsNode.preferredLayout)
      return op->emitError()
             << kVMIDiagLayoutContractPrefix << "conflicting preferred layouts "
             << lhsNode.preferredLayout << " and " << rhsNode.preferredLayout;

    rhsNode.parent = lhsRoot;
    if (!lhsNode.naturalLayout)
      lhsNode.naturalLayout = rhsNode.naturalLayout;
    if (!lhsNode.preferredLayout)
      lhsNode.preferredLayout = rhsNode.preferredLayout;
    return success();
  }

  LogicalResult uniteMask(Value lhs, Value rhs, Operation *op) {
    unsigned lhsId = addMaskValue(lhs);
    unsigned rhsId = addMaskValue(rhs);
    if (lhsId == ~0u || rhsId == ~0u)
      return success();
    unsigned lhsRoot = findMask(lhsId);
    unsigned rhsRoot = findMask(rhsId);
    if (lhsRoot == rhsRoot)
      return success();

    MaskNode &lhsNode = maskNodes[lhsRoot];
    MaskNode &rhsNode = maskNodes[rhsRoot];
    if (lhsNode.requestedLayout && rhsNode.requestedLayout &&
        lhsNode.requestedLayout != rhsNode.requestedLayout)
      return op->emitError()
             << kVMIDiagLayoutContractPrefix << "conflicting mask layouts "
             << lhsNode.requestedLayout << " and " << rhsNode.requestedLayout;
    rhsNode.parent = lhsRoot;
    if (!lhsNode.requestedLayout)
      lhsNode.requestedLayout = rhsNode.requestedLayout;
    return success();
  }

  LogicalResult
  setNaturalLayout(Value value, VMILayoutAttr layout, Operation *op,
                   DataLayoutSeedPhase phase = DataLayoutSeedPhase::Other) {
    unsigned id = addDataValue(value);
    if (id == ~0u || !layout)
      return success();
    unsigned root = find(id);
    VMILayoutAttr existing = dataNodes[root].naturalLayout;
    if (existing && existing != layout)
      return op->emitError()
             << kVMIDiagLayoutContractPrefix << "conflicting natural layouts "
             << existing << " and " << layout;
    dataNodes[root].naturalLayout = layout;
    dataLayoutSeeds.push_back(DataLayoutSeed{value, layout, phase});
    return success();
  }

  LogicalResult
  setPreferredLayout(Value value, VMILayoutAttr layout, Operation *op,
                     DataLayoutSeedPhase phase = DataLayoutSeedPhase::Other) {
    unsigned id = addDataValue(value);
    if (id == ~0u || !layout)
      return success();
    unsigned root = find(id);
    VMILayoutAttr existing = dataNodes[root].preferredLayout;
    if (existing && existing != layout)
      return op->emitError()
             << kVMIDiagLayoutContractPrefix << "conflicting preferred layouts "
             << existing << " and " << layout;
    dataNodes[root].preferredLayout = layout;
    dataLayoutSeeds.push_back(DataLayoutSeed{value, layout, phase});
    return success();
  }

  VMILayoutAttr getContiguousLayout() {
    return VMILayoutAttr::getContiguous(ctx);
  }

  VMILayoutAttr getPreferredDenseStoreLayout(VMIVRegType type) {
    VMILayoutSupport supports;
    FailureOr<VMIStoreLayoutFact> fact =
        supports.getPreferredStoreLayoutFact(type);
    if (failed(fact))
      return {};
    return fact->valueLayout;
  }

  bool hasDataLayoutSeed(Value value) {
    unsigned id = addDataValue(value);
    if (id == ~0u)
      return false;
    DataNode &node = dataNodes[find(id)];
    return static_cast<bool>(node.naturalLayout || node.preferredLayout);
  }

  FailureOr<VMIMaskedStoreLayoutFact>
  getPreferredDenseMaskedStoreLayout(VMIVRegType valueType,
                                     VMIMaskType maskType) {
    VMILayoutSupport supports;
    return supports.getPreferredMaskedStoreLayoutFact(valueType, maskType);
  }

  VMILayoutAttr getGroupSlotsLayout(int64_t numGroups) {
    return VMILayoutAttr::getGroupSlots(ctx, numGroups);
  }

  VMILayoutAttr getPreferredGroupSlotsLayout(VMIVRegType type,
                                             int64_t numGroups) {
    if (VMILayoutAttr existing = type.getLayoutAttr())
      if (existing.isGroupSlots() && existing.getSlots() > 0)
        return existing;
    VMILayoutSupport supports;
    FailureOr<VMIGroupReduceLayoutFact> fact =
        supports.getPreferredGroupReduceLayoutFact(type, numGroups);
    if (succeeded(fact))
      return fact->resultLayout;
    return getGroupSlotsLayout(numGroups);
  }

  VMILayoutAttr getPreferredGroupReduceSourceLayout(VMIVRegType type,
                                                    int64_t numGroups) {
    if (VMILayoutAttr existing = type.getLayoutAttr())
      return existing;
    VMILayoutSupport supports;
    FailureOr<VMIGroupReduceLayoutFact> fact =
        supports.getPreferredGroupReduceLayoutFact(type, numGroups);
    if (succeeded(fact))
      return fact->sourceLayout;
    return getContiguousLayout();
  }

  DataLayoutSeedPhase getGroupReduceUseSeedPhase(VMIVRegType sourceType,
                                                 int64_t numGroups,
                                                 VMIGroupReduceLayoutFact fact) {
    if (!fact.sourceLayout || !fact.sourceLayout.isContiguous() ||
        fact.sourceLayout.getLaneStride() != 1)
      return DataLayoutSeedPhase::Reduce;

    VMILayoutSupport supports;
    FailureOr<SmallVector<VMIGroupReduceLayoutFact, 4>> resultFacts =
        supports.getGroupReduceLayoutFactsForLayout(
            sourceType, numGroups, VMIGroupReduceLayoutPort::Result,
            fact.resultLayout);
    if (succeeded(resultFacts) && resultFacts->size() > 1)
      return DataLayoutSeedPhase::WeakReduce;
    return DataLayoutSeedPhase::Reduce;
  }

  VMILayoutAttr getPreferredGroupSlotLoadLayout(VMIGroupSlotLoadOp op) {
    auto type = cast<VMIVRegType>(op.getResult().getType());
    int64_t numGroups = op.getNumGroupsAttr().getInt();
    if (VMILayoutAttr existing = type.getLayoutAttr())
      if (existing.isGroupSlots() && existing.getSlots() > 0)
        return existing;
    std::optional<int64_t> sourceGroupStride =
        getConstantIndexValue(op.getSourceGroupStride());
    if (sourceGroupStride && *sourceGroupStride == 1)
      return VMILayoutAttr::getGroupSlots(ctx, numGroups, /*slots=*/8);
    return VMILayoutAttr::getGroupSlots(ctx, numGroups, /*slots=*/1);
  }

  VMILayoutAttr
  getPreferredGroupBroadcastLoadLayout(VMIGroupBroadcastLoadOp op) {
    auto type = cast<VMIVRegType>(op.getResult().getType());
    if (VMILayoutAttr existing = type.getLayoutAttr())
      return existing;

    VMILayoutSupport supports;
    FailureOr<VMIGroupBroadcastLoadDirectFact> fact =
        supports.getGroupBroadcastLoadDirectFact(
            type, op.getSource().getType(), op.getSourceGroupStride(),
            op.getNumGroupsAttr().getInt());
    if (failed(fact))
      return {};
    return fact->layout.resultLayout;
  }

  bool hasDirectGroupBroadcastLoadCandidate(VMIGroupBroadcastLoadOp op) {
    VMILayoutSupport supports;
    return succeeded(supports.getGroupBroadcastLoadDirectFact(
        cast<VMIVRegType>(op.getResult().getType()), op.getSource().getType(),
        op.getSourceGroupStride(), op.getNumGroupsAttr().getInt()));
  }

  VMILayoutAttr getPreferredGroupBroadcastSourceLayout(Value value,
                                                       int64_t numGroups) {
    auto type = dyn_cast<VMIVRegType>(value.getType());
    if (!type)
      return getContiguousLayout();
    if (VMILayoutAttr existing = type.getLayoutAttr())
      if (existing.isGroupSlots() && existing.getSlots() > 0)
        return existing;
    VMILayoutAttr solved = getDataLayout(value);
    if (solved && solved.isGroupSlots() && solved.getNumGroups() == numGroups &&
        solved.getSlots() > 0)
      return solved;
    if (type.getElementCount() == numGroups)
      return VMILayoutAttr::getGroupSlots(ctx, numGroups,
                                          numGroups >= 8 ? 8 : 1);
    if (auto load = value.getDefiningOp<VMIGroupSlotLoadOp>())
      return getPreferredGroupSlotLoadLayout(load);
    return getPreferredGroupSlotsLayout(type, numGroups);
  }

  VMILayoutAttr getPreferredGroupLoadResultLayout(VMIGroupLoadOp op) {
    auto type = cast<VMIVRegType>(op.getResult().getType());
    if (VMILayoutAttr existing = type.getLayoutAttr())
      return existing;

    int64_t numGroups = op.getNumGroupsAttr().getInt();
    if (numGroups <= 0 || type.getElementCount() % numGroups != 0)
      return getContiguousLayout();

    if (!type.getElementType().isF32())
      return getContiguousLayout();

    int64_t groupSize = type.getElementCount() / numGroups;
    std::optional<int64_t> rowStride = getConstantIndexValue(op.getRowStride());
    if (rowStride && *rowStride == groupSize)
      return getContiguousLayout();
    if (!rowStride || *rowStride <= 0 || *rowStride % 8 != 0)
      return getContiguousLayout();

    if (groupSize == 16)
      return VMILayoutAttr::getBlockDeinterleaved(ctx, 2);
    if (groupSize == 32)
      return VMILayoutAttr::getBlockDeinterleaved(ctx, 4);

    return getContiguousLayout();
  }

  LogicalResult validateGroupLoadLayoutPlan(VMIGroupLoadOp op) {
    auto type = cast<VMIVRegType>(op.getResult().getType());
    if (type.getLayoutAttr())
      return success();

    int64_t numGroups = op.getNumGroupsAttr().getInt();
    if (numGroups <= 0 || type.getElementCount() % numGroups != 0)
      return success();
    if (!type.getElementType().isF32())
      return success();

    int64_t groupSize = type.getElementCount() / numGroups;
    if (groupSize != 16 && groupSize != 32)
      return success();

    std::optional<int64_t> rowStride = getConstantIndexValue(op.getRowStride());
    if (rowStride && *rowStride == groupSize)
      return success();
    if (rowStride && *rowStride > 0 && *rowStride % 8 == 0)
      return success();

    return op.emitError()
           << kVMIDiagLayoutContractPrefix << "pto.vmi.group_load group_size "
           << groupSize
           << " requires constant positive row_stride divisible by 8 f32 "
              "elements for the block8 stride plan; stable gather fallback is "
              "not implemented";
  }

  VMILayoutAttr getPreferredGroupStoreUseLayout(
      Value value, int64_t numGroups, Value rowStride,
      llvm::function_ref<VMILayoutAttr(Value)> getKnownLayout) {
    auto type = dyn_cast<VMIVRegType>(value.getType());
    if (!type)
      return getContiguousLayout();
    if (VMILayoutAttr existing = type.getLayoutAttr())
      if (existing.isGroupSlots() && existing.getSlots() > 0)
        return existing;
    VMILayoutAttr known = getKnownLayout(value);
    if (known && known.isGroupSlots() && known.getNumGroups() == numGroups &&
        known.getSlots() > 0)
      return known;
    if (known && known.isDeinterleaved()) {
      if (known.getFactor() == 2)
        return known;
      if (known.getFactor() == 4)
        return VMILayoutAttr::getDeinterleaved(ctx, /*factor=*/2);
    }
    if (auto castOp = value.getDefiningOp()) {
      if (isa<VMIExtFOp, VMIExtSIOp, VMIExtUIOp, VMITruncFOp, VMITruncIOp>(
              castOp) &&
          castOp->getNumOperands() == 1 && castOp->getNumResults() == 1) {
        auto sourceType =
            dyn_cast<VMIVRegType>(castOp->getOperand(0).getType());
        auto resultType = dyn_cast<VMIVRegType>(castOp->getResult(0).getType());
        if (sourceType && resultType) {
          VMILayoutAttr sourceLayout = getKnownLayout(castOp->getOperand(0));
          if (!sourceLayout)
            sourceLayout = getDataLayout(castOp->getOperand(0));
          VMILayoutSupport supports;
          FailureOr<VMICastLayoutFact> fact =
              supports.getCastLayoutFactForSourceLayout(sourceType, resultType,
                                                        sourceLayout);
          if (succeeded(fact) && fact->resultLayout.isGroupSlots() &&
              fact->resultLayout.getNumGroups() == numGroups &&
              fact->resultLayout.getSlots() > 0)
            return fact->resultLayout;
        }
      }
    }
    VMILayoutAttr solved = getDataLayout(value);
    if (solved && solved.isGroupSlots() && solved.getNumGroups() == numGroups &&
        solved.getSlots() > 0)
      return solved;
    if (value.getDefiningOp<VMIGroupReduceAddFOp>() ||
        value.getDefiningOp<VMIGroupReduceMaxFOp>() ||
        value.getDefiningOp<VMIGroupReduceMinFOp>() ||
        value.getDefiningOp<VMIGroupReduceAddIOp>() ||
        value.getDefiningOp<VMIGroupReduceMaxIOp>() ||
        value.getDefiningOp<VMIGroupReduceMinIOp>())
      return getPreferredGroupSlotsLayout(type, numGroups);
    if (type.getElementCount() == numGroups) {
      std::optional<int64_t> stride = getConstantIndexValue(rowStride);
      bool packedSlots =
          stride && *stride == 1 && static_cast<int64_t>(numGroups) >= 8;
      return VMILayoutAttr::getGroupSlots(ctx, numGroups, packedSlots ? 8 : 1);
    }
    if (auto load = value.getDefiningOp<VMIGroupSlotLoadOp>())
      return getPreferredGroupSlotLoadLayout(load);
    return getContiguousLayout();
  }

  VMILayoutAttr getPreferredGroupStoreUseLayout(Value value, int64_t numGroups,
                                                Value rowStride) {
    return getPreferredGroupStoreUseLayout(
        value, numGroups, rowStride,
        [&](Value knownValue) { return getDataLayout(knownValue); });
  }

  VMILayoutAttr getDataLayout(Value value) {
    unsigned id = addDataValue(value);
    if (id == ~0u)
      return {};
    unsigned root = find(id);
    if (dataNodes[root].naturalLayout)
      return dataNodes[root].naturalLayout;
    if (dataNodes[root].preferredLayout)
      return dataNodes[root].preferredLayout;
    return getContiguousLayout();
  }

  void requestDataUse(OpOperand &operand, VMILayoutAttr layout,
                      bool late = false,
                      DataLayoutSeedPhase phase = DataLayoutSeedPhase::Other) {
    if (isa<VMIVRegType>(operand.get().getType())) {
      addDataValue(operand.get());
      dataUseRequests.push_back(DataUseRequest{&operand, layout, late, phase});
    }
  }

  LogicalResult constrainElementwiseBinary(OpOperand &lhs, OpOperand &rhs,
                                           Value result, Operation *op) {
    if (failed(unite(lhs.get(), rhs.get(), op)))
      return failure();
    return unite(lhs.get(), result, op);
  }

  LogicalResult
  requestMaskUse(OpOperand &operand, VMILayoutAttr layout, Operation *op,
                 DataLayoutSeedPhase phase = DataLayoutSeedPhase::Other) {
    if (!isa<VMIMaskType>(operand.get().getType()))
      return success();
    if (!layout)
      return op->emitError()
             << kVMIDiagLayoutContractPrefix
             << "cannot infer concrete mask use layout";
    maskUseRequests.push_back(MaskUseRequest{&operand, layout, phase});
    return success();
  }

  LogicalResult collect() {
    module.walk([&](Operation *op) {
      for (Value result : op->getResults()) {
        addDataValue(result);
        addMaskValue(result);
      }
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (BlockArgument arg : block.getArguments()) {
            addDataValue(arg);
            addMaskValue(arg);
          }
    });
    return success();
  }

  LogicalResult addConstraints() {
    WalkResult result = module.walk([&](Operation *op) -> WalkResult {
      if (auto maskAnd = dyn_cast<VMIMaskAndOp>(op)) {
        if (failed(uniteMask(maskAnd.getLhs(), maskAnd.getRhs(), op)) ||
            failed(uniteMask(maskAnd.getLhs(), maskAnd.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto maskOr = dyn_cast<VMIMaskOrOp>(op)) {
        if (failed(uniteMask(maskOr.getLhs(), maskOr.getRhs(), op)) ||
            failed(uniteMask(maskOr.getLhs(), maskOr.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto maskXor = dyn_cast<VMIMaskXOrOp>(op)) {
        if (failed(uniteMask(maskXor.getLhs(), maskXor.getRhs(), op)) ||
            failed(uniteMask(maskXor.getLhs(), maskXor.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto maskNot = dyn_cast<VMIMaskNotOp>(op)) {
        if (failed(uniteMask(maskNot.getSource(), maskNot.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto ensure = dyn_cast<VMIEnsureMaskLayoutOp>(op)) {
        if (failed(uniteMask(ensure.getSource(), ensure.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto ensure = dyn_cast<VMIEnsureMaskGranularityOp>(op)) {
        return WalkResult::advance();
      }
      if (auto addf = dyn_cast<VMIAddFOp>(op)) {
        if (failed(constrainElementwiseBinary(addf.getLhsMutable(),
                                              addf.getRhsMutable(),
                                              addf.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto addi = dyn_cast<VMIAddIOp>(op)) {
        if (failed(constrainElementwiseBinary(addi.getLhsMutable(),
                                              addi.getRhsMutable(),
                                              addi.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto subf = dyn_cast<VMISubFOp>(op)) {
        if (failed(constrainElementwiseBinary(subf.getLhsMutable(),
                                              subf.getRhsMutable(),
                                              subf.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto subi = dyn_cast<VMISubIOp>(op)) {
        if (failed(constrainElementwiseBinary(subi.getLhsMutable(),
                                              subi.getRhsMutable(),
                                              subi.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto mulf = dyn_cast<VMIMulFOp>(op)) {
        if (failed(constrainElementwiseBinary(mulf.getLhsMutable(),
                                              mulf.getRhsMutable(),
                                              mulf.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto muli = dyn_cast<VMIMulIOp>(op)) {
        if (failed(constrainElementwiseBinary(muli.getLhsMutable(),
                                              muli.getRhsMutable(),
                                              muli.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto vmull = dyn_cast<VMIVmullOp>(op)) {
        if (failed(constrainElementwiseBinary(vmull.getAMutable(),
                                              vmull.getBMutable(),
                                              vmull.getLow(), op)) ||
            failed(unite(vmull.getA(), vmull.getHigh(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto fma = dyn_cast<VMIFmaOp>(op)) {
        if (failed(unite(fma.getLhs(), fma.getRhs(), op)) ||
            failed(unite(fma.getLhs(), fma.getAcc(), op)) ||
            failed(unite(fma.getLhs(), fma.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto divf = dyn_cast<VMIDivFOp>(op)) {
        if (failed(constrainElementwiseBinary(divf.getLhsMutable(),
                                              divf.getRhsMutable(),
                                              divf.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto minf = dyn_cast<VMIMinFOp>(op)) {
        if (failed(constrainElementwiseBinary(minf.getLhsMutable(),
                                              minf.getRhsMutable(),
                                              minf.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto maxf = dyn_cast<VMIMaxFOp>(op)) {
        if (failed(constrainElementwiseBinary(maxf.getLhsMutable(),
                                              maxf.getRhsMutable(),
                                              maxf.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto negf = dyn_cast<VMINegFOp>(op)) {
        if (failed(unite(negf.getSource(), negf.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto absf = dyn_cast<VMIAbsFOp>(op)) {
        if (failed(unite(absf.getSource(), absf.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto absi = dyn_cast<VMIAbsIOp>(op)) {
        if (failed(unite(absi.getSource(), absi.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto sqrt = dyn_cast<VMISqrtOp>(op)) {
        if (failed(unite(sqrt.getSource(), sqrt.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto exp = dyn_cast<VMIExpOp>(op)) {
        if (failed(unite(exp.getSource(), exp.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto ln = dyn_cast<VMILnOp>(op)) {
        if (failed(unite(ln.getSource(), ln.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto relu = dyn_cast<VMIReluOp>(op)) {
        if (failed(unite(relu.getSource(), relu.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto fptosi = dyn_cast<VMIFPToSIOp>(op)) {
        if (failed(unite(fptosi.getSource(), fptosi.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto sitofp = dyn_cast<VMISIToFPOp>(op)) {
        if (failed(unite(sitofp.getSource(), sitofp.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto andi = dyn_cast<VMIAndIOp>(op)) {
        if (failed(constrainElementwiseBinary(andi.getLhsMutable(),
                                              andi.getRhsMutable(),
                                              andi.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto ori = dyn_cast<VMIOrIOp>(op)) {
        if (failed(constrainElementwiseBinary(
                ori.getLhsMutable(), ori.getRhsMutable(), ori.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto xori = dyn_cast<VMIXOrIOp>(op)) {
        if (failed(constrainElementwiseBinary(xori.getLhsMutable(),
                                              xori.getRhsMutable(),
                                              xori.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto shli = dyn_cast<VMIShLIOp>(op)) {
        if (failed(constrainElementwiseBinary(shli.getLhsMutable(),
                                              shli.getRhsMutable(),
                                              shli.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto shrui = dyn_cast<VMIShRUIOp>(op)) {
        if (failed(constrainElementwiseBinary(shrui.getLhsMutable(),
                                              shrui.getRhsMutable(),
                                              shrui.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto shrsi = dyn_cast<VMIShRSIOp>(op)) {
        if (failed(constrainElementwiseBinary(shrsi.getLhsMutable(),
                                              shrsi.getRhsMutable(),
                                              shrsi.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto notOp = dyn_cast<VMINotOp>(op)) {
        if (failed(unite(notOp.getSource(), notOp.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto cmpf = dyn_cast<VMICmpFOp>(op)) {
        if (failed(unite(cmpf.getLhs(), cmpf.getRhs(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto cmpi = dyn_cast<VMICmpIOp>(op)) {
        if (failed(unite(cmpi.getLhs(), cmpi.getRhs(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto select = dyn_cast<VMISelectOp>(op)) {
        if (failed(unite(select.getTrueValue(), select.getFalseValue(), op)) ||
            failed(unite(select.getTrueValue(), select.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto activePrefix = dyn_cast<VMIActivePrefixIndexOp>(op)) {
        if (failed(setNaturalLayout(activePrefix.getResult(),
                                    getContiguousLayout(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto compress = dyn_cast<VMICompressOp>(op)) {
        requestDataUse(compress.getSourceMutable(), getContiguousLayout());
        if (failed(setNaturalLayout(compress.getResult(), getContiguousLayout(),
                                    op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto reduce = dyn_cast<VMIReduceAddIOp>(op)) {
        requestDataUse(reduce.getSourceMutable(), getContiguousLayout(),
                       /*late=*/false, DataLayoutSeedPhase::Reduce);
        requestDataUse(reduce.getInitMutable(), getContiguousLayout(),
                       /*late=*/false, DataLayoutSeedPhase::Reduce);
        if (failed(requestMaskUse(reduce.getMaskMutable(),
                                  getContiguousLayout(), op)))
          return WalkResult::interrupt();
        if (failed(setNaturalLayout(reduce.getResult(), getContiguousLayout(),
                                    op, DataLayoutSeedPhase::Reduce)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto reduce = dyn_cast<VMIReduceAddFOp>(op)) {
        requestDataUse(reduce.getSourceMutable(), getContiguousLayout(),
                       /*late=*/false, DataLayoutSeedPhase::Reduce);
        requestDataUse(reduce.getInitMutable(), getContiguousLayout(),
                       /*late=*/false, DataLayoutSeedPhase::Reduce);
        if (failed(requestMaskUse(reduce.getMaskMutable(),
                                  getContiguousLayout(), op)))
          return WalkResult::interrupt();
        if (failed(setNaturalLayout(reduce.getResult(), getContiguousLayout(),
                                    op, DataLayoutSeedPhase::Reduce)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto reduce = dyn_cast<VMIReduceMaxFOp>(op)) {
        requestDataUse(reduce.getSourceMutable(), getContiguousLayout(),
                       /*late=*/false, DataLayoutSeedPhase::Reduce);
        requestDataUse(reduce.getInitMutable(), getContiguousLayout(),
                       /*late=*/false, DataLayoutSeedPhase::Reduce);
        if (failed(requestMaskUse(reduce.getMaskMutable(),
                                  getContiguousLayout(), op)))
          return WalkResult::interrupt();
        if (failed(setNaturalLayout(reduce.getResult(), getContiguousLayout(),
                                    op, DataLayoutSeedPhase::Reduce)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto reduce = dyn_cast<VMIReduceMinFOp>(op)) {
        requestDataUse(reduce.getSourceMutable(), getContiguousLayout(),
                       /*late=*/false, DataLayoutSeedPhase::Reduce);
        requestDataUse(reduce.getInitMutable(), getContiguousLayout(),
                       /*late=*/false, DataLayoutSeedPhase::Reduce);
        if (failed(requestMaskUse(reduce.getMaskMutable(),
                                  getContiguousLayout(), op)))
          return WalkResult::interrupt();
        if (failed(setNaturalLayout(reduce.getResult(), getContiguousLayout(),
                                    op, DataLayoutSeedPhase::Reduce)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto reduce = dyn_cast<VMIReduceMaxIOp>(op)) {
        requestDataUse(reduce.getSourceMutable(), getContiguousLayout(),
                       /*late=*/false, DataLayoutSeedPhase::Reduce);
        requestDataUse(reduce.getInitMutable(), getContiguousLayout(),
                       /*late=*/false, DataLayoutSeedPhase::Reduce);
        if (failed(requestMaskUse(reduce.getMaskMutable(),
                                  getContiguousLayout(), op)))
          return WalkResult::interrupt();
        if (failed(setNaturalLayout(reduce.getResult(), getContiguousLayout(),
                                    op, DataLayoutSeedPhase::Reduce)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto reduce = dyn_cast<VMIReduceMinIOp>(op)) {
        requestDataUse(reduce.getSourceMutable(), getContiguousLayout(),
                       /*late=*/false, DataLayoutSeedPhase::Reduce);
        requestDataUse(reduce.getInitMutable(), getContiguousLayout(),
                       /*late=*/false, DataLayoutSeedPhase::Reduce);
        if (failed(requestMaskUse(reduce.getMaskMutable(),
                                  getContiguousLayout(), op)))
          return WalkResult::interrupt();
        if (failed(setNaturalLayout(reduce.getResult(), getContiguousLayout(),
                                    op, DataLayoutSeedPhase::Reduce)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto reduce = dyn_cast<VMIGroupReduceAddFOp>(op)) {
        auto sourceType = cast<VMIVRegType>(reduce.getSource().getType());
        auto resultType = cast<VMIVRegType>(reduce.getResult().getType());
        int64_t numGroups = reduce.getNumGroupsAttr().getInt();
        VMILayoutSupport supports;
        FailureOr<VMIGroupReduceLayoutFact> fact =
            supports.getPreferredGroupReduceLayoutFact(sourceType, numGroups);
        VMILayoutAttr sourceLayout =
            succeeded(fact) ? fact->sourceLayout : getContiguousLayout();
        DataLayoutSeedPhase usePhase =
            succeeded(fact)
                ? getGroupReduceUseSeedPhase(sourceType, numGroups, *fact)
                : DataLayoutSeedPhase::Reduce;
        requestDataUse(reduce.getSourceMutable(), sourceLayout, /*late=*/false,
                       usePhase);
        if (failed(requestMaskUse(reduce.getMaskMutable(), sourceLayout, op,
                                  usePhase)))
          return WalkResult::interrupt();
        if (failed(setNaturalLayout(
                reduce.getResult(),
                succeeded(fact)
                    ? fact->resultLayout
                    : getPreferredGroupSlotsLayout(resultType, numGroups),
                op, DataLayoutSeedPhase::Reduce)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto reduce = dyn_cast<VMIGroupReduceMaxFOp>(op)) {
        auto sourceType = cast<VMIVRegType>(reduce.getSource().getType());
        auto resultType = cast<VMIVRegType>(reduce.getResult().getType());
        int64_t numGroups = reduce.getNumGroupsAttr().getInt();
        VMILayoutSupport supports;
        FailureOr<VMIGroupReduceLayoutFact> fact =
            supports.getPreferredGroupReduceLayoutFact(sourceType, numGroups);
        VMILayoutAttr sourceLayout =
            succeeded(fact) ? fact->sourceLayout : getContiguousLayout();
        DataLayoutSeedPhase usePhase =
            succeeded(fact)
                ? getGroupReduceUseSeedPhase(sourceType, numGroups, *fact)
                : DataLayoutSeedPhase::Reduce;
        requestDataUse(reduce.getSourceMutable(), sourceLayout, /*late=*/false,
                       usePhase);
        if (failed(requestMaskUse(reduce.getMaskMutable(), sourceLayout, op,
                                  usePhase)))
          return WalkResult::interrupt();
        if (failed(setNaturalLayout(
                reduce.getResult(),
                succeeded(fact)
                    ? fact->resultLayout
                    : getPreferredGroupSlotsLayout(resultType, numGroups),
                op, DataLayoutSeedPhase::Reduce)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto reduce = dyn_cast<VMIGroupReduceMinFOp>(op)) {
        auto sourceType = cast<VMIVRegType>(reduce.getSource().getType());
        auto resultType = cast<VMIVRegType>(reduce.getResult().getType());
        int64_t numGroups = reduce.getNumGroupsAttr().getInt();
        VMILayoutSupport supports;
        FailureOr<VMIGroupReduceLayoutFact> fact =
            supports.getPreferredGroupReduceLayoutFact(sourceType, numGroups);
        VMILayoutAttr sourceLayout =
            succeeded(fact) ? fact->sourceLayout : getContiguousLayout();
        DataLayoutSeedPhase usePhase =
            succeeded(fact)
                ? getGroupReduceUseSeedPhase(sourceType, numGroups, *fact)
                : DataLayoutSeedPhase::Reduce;
        requestDataUse(reduce.getSourceMutable(), sourceLayout, /*late=*/false,
                       usePhase);
        if (failed(requestMaskUse(reduce.getMaskMutable(), sourceLayout, op,
                                  usePhase)))
          return WalkResult::interrupt();
        if (failed(setNaturalLayout(
                reduce.getResult(),
                succeeded(fact)
                    ? fact->resultLayout
                    : getPreferredGroupSlotsLayout(resultType, numGroups),
                op, DataLayoutSeedPhase::Reduce)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto reduce = dyn_cast<VMIGroupReduceAddIOp>(op)) {
        auto sourceType = cast<VMIVRegType>(reduce.getSource().getType());
        auto resultType = cast<VMIVRegType>(reduce.getResult().getType());
        int64_t numGroups = reduce.getNumGroupsAttr().getInt();
        VMILayoutSupport supports;
        FailureOr<VMIGroupReduceLayoutFact> fact =
            supports.getPreferredGroupReduceLayoutFact(sourceType, numGroups);
        VMILayoutAttr sourceLayout =
            succeeded(fact) ? fact->sourceLayout : getContiguousLayout();
        DataLayoutSeedPhase usePhase =
            succeeded(fact)
                ? getGroupReduceUseSeedPhase(sourceType, numGroups, *fact)
                : DataLayoutSeedPhase::Reduce;
        requestDataUse(reduce.getSourceMutable(), sourceLayout, /*late=*/false,
                       usePhase);
        if (failed(requestMaskUse(reduce.getMaskMutable(), sourceLayout, op,
                                  usePhase)))
          return WalkResult::interrupt();
        if (failed(setNaturalLayout(
                reduce.getResult(),
                succeeded(fact)
                    ? fact->resultLayout
                    : getPreferredGroupSlotsLayout(resultType, numGroups),
                op, DataLayoutSeedPhase::Reduce)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto reduce = dyn_cast<VMIGroupReduceMaxIOp>(op)) {
        auto sourceType = cast<VMIVRegType>(reduce.getSource().getType());
        auto resultType = cast<VMIVRegType>(reduce.getResult().getType());
        int64_t numGroups = reduce.getNumGroupsAttr().getInt();
        VMILayoutSupport supports;
        FailureOr<VMIGroupReduceLayoutFact> fact =
            supports.getPreferredGroupReduceLayoutFact(sourceType, numGroups);
        VMILayoutAttr sourceLayout =
            succeeded(fact) ? fact->sourceLayout : getContiguousLayout();
        DataLayoutSeedPhase usePhase =
            succeeded(fact)
                ? getGroupReduceUseSeedPhase(sourceType, numGroups, *fact)
                : DataLayoutSeedPhase::Reduce;
        requestDataUse(reduce.getSourceMutable(), sourceLayout, /*late=*/false,
                       usePhase);
        if (failed(requestMaskUse(reduce.getMaskMutable(), sourceLayout, op,
                                  usePhase)))
          return WalkResult::interrupt();
        if (failed(setNaturalLayout(
                reduce.getResult(),
                succeeded(fact)
                    ? fact->resultLayout
                    : getPreferredGroupSlotsLayout(resultType, numGroups),
                op, DataLayoutSeedPhase::Reduce)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto reduce = dyn_cast<VMIGroupReduceMinIOp>(op)) {
        auto sourceType = cast<VMIVRegType>(reduce.getSource().getType());
        auto resultType = cast<VMIVRegType>(reduce.getResult().getType());
        int64_t numGroups = reduce.getNumGroupsAttr().getInt();
        VMILayoutSupport supports;
        FailureOr<VMIGroupReduceLayoutFact> fact =
            supports.getPreferredGroupReduceLayoutFact(sourceType, numGroups);
        VMILayoutAttr sourceLayout =
            succeeded(fact) ? fact->sourceLayout : getContiguousLayout();
        DataLayoutSeedPhase usePhase =
            succeeded(fact)
                ? getGroupReduceUseSeedPhase(sourceType, numGroups, *fact)
                : DataLayoutSeedPhase::Reduce;
        requestDataUse(reduce.getSourceMutable(), sourceLayout, /*late=*/false,
                       usePhase);
        if (failed(requestMaskUse(reduce.getMaskMutable(), sourceLayout, op,
                                  usePhase)))
          return WalkResult::interrupt();
        if (failed(setNaturalLayout(
                reduce.getResult(),
                succeeded(fact)
                    ? fact->resultLayout
                    : getPreferredGroupSlotsLayout(resultType, numGroups),
                op, DataLayoutSeedPhase::Reduce)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto broadcast = dyn_cast<VMIGroupBroadcastOp>(op)) {
        requestDataUse(
            broadcast.getSourceMutable(),
            getPreferredGroupBroadcastSourceLayout(
                broadcast.getSource(), broadcast.getNumGroupsAttr().getInt()),
            /*late=*/false, DataLayoutSeedPhase::GroupBroadcast);
        return WalkResult::advance();
      }
      if (auto hist = dyn_cast<VMIVdhistOp>(op)) {
        requestDataUse(hist.getAccMutable(), getContiguousLayout(),
                       /*late=*/false, DataLayoutSeedPhase::Reduce);
        requestDataUse(hist.getSourceMutable(), getContiguousLayout(),
                       /*late=*/false, DataLayoutSeedPhase::Reduce);
        if (failed(requestMaskUse(hist.getMaskMutable(), getContiguousLayout(),
                                  op, DataLayoutSeedPhase::Reduce)))
          return WalkResult::interrupt();
        if (failed(setNaturalLayout(hist.getResult(), getContiguousLayout(), op,
                                    DataLayoutSeedPhase::Reduce)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto hist = dyn_cast<VMIVchistOp>(op)) {
        requestDataUse(hist.getAccMutable(), getContiguousLayout(),
                       /*late=*/false, DataLayoutSeedPhase::Reduce);
        requestDataUse(hist.getSourceMutable(), getContiguousLayout(),
                       /*late=*/false, DataLayoutSeedPhase::Reduce);
        if (failed(requestMaskUse(hist.getMaskMutable(), getContiguousLayout(),
                                  op, DataLayoutSeedPhase::Reduce)))
          return WalkResult::interrupt();
        if (failed(setNaturalLayout(hist.getResult(), getContiguousLayout(), op,
                                    DataLayoutSeedPhase::Reduce)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto extf = dyn_cast<VMIExtFOp>(op)) {
        auto sourceType = cast<VMIVRegType>(extf.getSource().getType());
        auto resultType = cast<VMIVRegType>(extf.getResult().getType());
        VMILayoutSupport supports;
        FailureOr<VMICastLayoutFact> fact =
            supports.getPreferredCastLayoutFact(sourceType, resultType);
        if (succeeded(fact)) {
          if (failed(setPreferredLayout(extf.getResult(), fact->resultLayout,
                                        op, DataLayoutSeedPhase::Cast)))
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto extsi = dyn_cast<VMIExtSIOp>(op)) {
        auto sourceType = cast<VMIVRegType>(extsi.getSource().getType());
        auto resultType = cast<VMIVRegType>(extsi.getResult().getType());
        VMILayoutSupport supports;
        FailureOr<VMICastLayoutFact> fact =
            supports.getPreferredCastLayoutFact(sourceType, resultType);
        if (succeeded(fact)) {
          if (failed(setPreferredLayout(extsi.getResult(), fact->resultLayout,
                                        op, DataLayoutSeedPhase::Cast)))
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto extui = dyn_cast<VMIExtUIOp>(op)) {
        auto sourceType = cast<VMIVRegType>(extui.getSource().getType());
        auto resultType = cast<VMIVRegType>(extui.getResult().getType());
        VMILayoutSupport supports;
        FailureOr<VMICastLayoutFact> fact =
            supports.getPreferredCastLayoutFact(sourceType, resultType);
        if (succeeded(fact)) {
          if (failed(setPreferredLayout(extui.getResult(), fact->resultLayout,
                                        op, DataLayoutSeedPhase::Cast)))
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto truncf = dyn_cast<VMITruncFOp>(op)) {
        auto sourceType = cast<VMIVRegType>(truncf.getSource().getType());
        auto resultType = cast<VMIVRegType>(truncf.getResult().getType());
        VMILayoutSupport supports;
        FailureOr<VMICastLayoutFact> fact =
            supports.getPreferredCastLayoutFact(sourceType, resultType);
        VMILayoutAttr resultLayout = getContiguousLayout();
        if (succeeded(fact)) {
          resultLayout = fact->resultLayout;
        }
        if (failed(setPreferredLayout(truncf.getResult(), resultLayout, op,
                                      DataLayoutSeedPhase::Cast)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto trunci = dyn_cast<VMITruncIOp>(op)) {
        auto sourceType = cast<VMIVRegType>(trunci.getSource().getType());
        auto resultType = cast<VMIVRegType>(trunci.getResult().getType());
        VMILayoutSupport supports;
        FailureOr<VMICastLayoutFact> fact =
            supports.getPreferredCastLayoutFact(sourceType, resultType);
        VMILayoutAttr resultLayout = getContiguousLayout();
        if (succeeded(fact)) {
          resultLayout = fact->resultLayout;
        }
        if (failed(setPreferredLayout(trunci.getResult(), resultLayout, op,
                                      DataLayoutSeedPhase::Cast)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto bitcast = dyn_cast<VMIBitcastOp>(op)) {
        if (failed(unite(bitcast.getSource(), bitcast.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto vintlv = dyn_cast<VMIVintlvOp>(op)) {
        VMILayoutSupport supports;
        auto valueType = cast<VMIVRegType>(vintlv.getLow().getType());
        FailureOr<VMIInterleaveLayoutFact> fact =
            supports.getPreferredVintlvLayoutFact(valueType);
        if (failed(fact))
          return WalkResult::advance();
        requestDataUse(vintlv.getLhsMutable(), fact->lhsLayout);
        requestDataUse(vintlv.getRhsMutable(), fact->rhsLayout);
        if (failed(requestMaskUse(vintlv.getMaskMutable(), fact->maskLayout,
                                  op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto vdintlv = dyn_cast<VMIVdintlvOp>(op)) {
        VMILayoutSupport supports;
        auto valueType = cast<VMIVRegType>(vdintlv.getLow().getType());
        FailureOr<VMIInterleaveLayoutFact> fact =
            supports.getPreferredVdintlvLayoutFact(valueType);
        if (failed(fact))
          return WalkResult::advance();
        requestDataUse(vdintlv.getLhsMutable(), fact->lhsLayout);
        requestDataUse(vdintlv.getRhsMutable(), fact->rhsLayout);
        if (failed(requestMaskUse(vdintlv.getMaskMutable(), fact->maskLayout,
                                  op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto load = dyn_cast<VMIDeinterleaveLoadOp>(op)) {
        VMILayoutSupport supports;
        FailureOr<VMIDeinterleaveLoadLayoutFact> fact =
            supports.getPreferredDeinterleaveLoadLayoutFact(
                cast<VMIVRegType>(load.getLow().getType()));
        if (failed(fact))
          return WalkResult::advance();
        if (failed(setNaturalLayout(load.getLow(), fact->lowLayout, op)) ||
            failed(setNaturalLayout(load.getHigh(), fact->highLayout, op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto load = dyn_cast<VMIMaskedLoadOp>(op)) {
        requestDataUse(load.getPassthruMutable(), getContiguousLayout());
        if (failed(
                setNaturalLayout(load.getResult(), getContiguousLayout(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto gather = dyn_cast<VMIGatherOp>(op)) {
        requestDataUse(gather.getIndicesMutable(), getContiguousLayout());
        requestDataUse(gather.getPassthruMutable(), getContiguousLayout());
        if (failed(requestMaskUse(gather.getMaskMutable(),
                                  getContiguousLayout(), op)))
          return WalkResult::interrupt();
        if (failed(setNaturalLayout(gather.getResult(), getContiguousLayout(),
                                    op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto load = dyn_cast<VMIExpandLoadOp>(op)) {
        requestDataUse(load.getPassthruMutable(), getContiguousLayout());
        if (failed(
                setNaturalLayout(load.getResult(), getContiguousLayout(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto load = dyn_cast<VMIGroupLoadOp>(op)) {
        if (failed(validateGroupLoadLayoutPlan(load)))
          return WalkResult::interrupt();
        VMILayoutAttr layout = getPreferredGroupLoadResultLayout(load);
        if (layout.isContiguous() && layout.getLaneStride() == 1) {
          if (failed(setPreferredLayout(load.getResult(), layout, op)))
            return WalkResult::interrupt();
        } else if (failed(setNaturalLayout(load.getResult(), layout, op,
                                           DataLayoutSeedPhase::GroupLoad))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto load = dyn_cast<VMIGroupSlotLoadOp>(op)) {
        if (failed(setPreferredLayout(
                load.getResult(), getPreferredGroupSlotLoadLayout(load), op,
                DataLayoutSeedPhase::GroupSlotLoad)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto load = dyn_cast<VMIGroupBroadcastLoadOp>(op)) {
        DataLayoutSeedPhase phase =
            hasDirectGroupBroadcastLoadCandidate(load)
                ? DataLayoutSeedPhase::GroupBroadcastLoad
                : DataLayoutSeedPhase::Other;
        if (failed(setNaturalLayout(load.getResult(),
                                    getPreferredGroupBroadcastLoadLayout(load),
                                    op, phase)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto load = dyn_cast<VMIStrideLoadOp>(op)) {
        if (failed(
                setNaturalLayout(load.getResult(), getContiguousLayout(), op)))
          return WalkResult::interrupt();
        if (failed(requestMaskUse(load.getMaskMutable(), getContiguousLayout(),
                                  op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto store = dyn_cast<VMIStoreOp>(op)) {
        auto valueType = cast<VMIVRegType>(store.getValue().getType());
        if (!hasDataLayoutSeed(store.getValue()))
          if (VMILayoutAttr layout = getPreferredDenseStoreLayout(valueType))
            requestDataUse(store.getValueMutable(), layout, /*late=*/false,
                           DataLayoutSeedPhase::Store);
        return WalkResult::advance();
      }
      if (auto store = dyn_cast<VMIInterleaveStoreOp>(op)) {
        requestDataUse(store.getLowMutable(), getContiguousLayout());
        requestDataUse(store.getHighMutable(), getContiguousLayout());
        return WalkResult::advance();
      }
      if (auto store = dyn_cast<VMIGroupStoreOp>(op)) {
        addDataValue(store.getValue());
        groupStoreUseRequests.push_back(GroupStoreUseRequest{store});
        return WalkResult::advance();
      }
      if (auto store = dyn_cast<VMIMaskedStoreOp>(op)) {
        auto valueType = cast<VMIVRegType>(store.getValue().getType());
        auto maskType = cast<VMIMaskType>(store.getMask().getType());
        if (!hasDataLayoutSeed(store.getValue())) {
          FailureOr<VMIMaskedStoreLayoutFact> fact =
              getPreferredDenseMaskedStoreLayout(valueType, maskType);
          if (succeeded(fact)) {
            requestDataUse(store.getValueMutable(), fact->valueLayout,
                           /*late=*/false,
                           DataLayoutSeedPhase::Store);
            if (failed(requestMaskUse(store.getMaskMutable(), fact->maskLayout,
                                      op, DataLayoutSeedPhase::Store)))
              return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      }
      if (auto store = dyn_cast<VMIStrideStoreOp>(op)) {
        requestDataUse(store.getValueMutable(), getContiguousLayout());
        if (failed(requestMaskUse(store.getMaskMutable(), getContiguousLayout(),
                                  op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto scatter = dyn_cast<VMIScatterOp>(op)) {
        requestDataUse(scatter.getValueMutable(), getContiguousLayout());
        requestDataUse(scatter.getIndicesMutable(), getContiguousLayout());
        if (failed(requestMaskUse(scatter.getMaskMutable(),
                                  getContiguousLayout(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto store = dyn_cast<VMICompressStoreOp>(op)) {
        requestDataUse(store.getValueMutable(), getContiguousLayout());
        if (failed(requestMaskUse(store.getMaskMutable(), getContiguousLayout(),
                                  op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto split = dyn_cast<VMIChannelSplitOp>(op)) {
        int64_t channels = split.getNumResults();
        if (channels != 2 && channels != 4) {
          split.emitError() << kVMIDiagUnsupportedPrefix
                            << "pto.vmi.channel_split supports only 2 or 4 "
                               "channels";
          return WalkResult::interrupt();
        }
        requestDataUse(split.getSourceMutable(),
                       VMILayoutAttr::getDeinterleaved(ctx, channels));
        for (Value result : split.getResults())
          if (failed(setNaturalLayout(result, getContiguousLayout(), op)))
            return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto merge = dyn_cast<VMIChannelMergeOp>(op)) {
        int64_t channels = merge.getInputs().size();
        if (channels != 2 && channels != 4) {
          merge.emitError() << kVMIDiagUnsupportedPrefix
                            << "pto.vmi.channel_merge supports only 2 or 4 "
                               "channels";
          return WalkResult::interrupt();
        }
        for (OpOperand &input : merge.getInputsMutable())
          requestDataUse(input, getContiguousLayout());
        if (failed(setNaturalLayout(
                merge.getResult(),
                VMILayoutAttr::getDeinterleaved(ctx, channels), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto shuffle = dyn_cast<VMIShuffleOp>(op)) {
        auto sourceType = cast<VMIVRegType>(shuffle.getSource().getType());
        auto resultType = cast<VMIVRegType>(shuffle.getResult().getType());
        if (sourceType.hasLayout() || resultType.hasLayout())
          return WalkResult::advance();

        requestDataUse(shuffle.getSourceMutable(), getContiguousLayout());
        if (isLane0SplatShuffle(shuffle))
          return WalkResult::advance();
        if (failed(setNaturalLayout(shuffle.getResult(), getContiguousLayout(),
                                    op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        if (failed(addIfConstraints(ifOp)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto executeRegionOp = dyn_cast<scf::ExecuteRegionOp>(op)) {
        if (failed(addExecuteRegionConstraints(executeRegionOp)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto indexSwitchOp = dyn_cast<scf::IndexSwitchOp>(op)) {
        if (failed(addIndexSwitchConstraints(indexSwitchOp)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
        if (failed(addWhileConstraints(whileOp)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        if (failed(addForConstraints(forOp)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto branchOp = dyn_cast<cf::BranchOp>(op)) {
        if (failed(addBranchConstraints(branchOp.getDest(),
                                        branchOp.getDestOperands(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto condBranchOp = dyn_cast<cf::CondBranchOp>(op)) {
        if (failed(addBranchConstraints(condBranchOp.getTrueDest(),
                                        condBranchOp.getTrueDestOperands(),
                                        op)) ||
            failed(addBranchConstraints(condBranchOp.getFalseDest(),
                                        condBranchOp.getFalseDestOperands(),
                                        op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto switchOp = dyn_cast<cf::SwitchOp>(op)) {
        if (failed(addBranchConstraints(switchOp.getDefaultDestination(),
                                        switchOp.getDefaultOperands(), op)))
          return WalkResult::interrupt();
        for (auto [dest, operands] : llvm::zip(switchOp.getCaseDestinations(),
                                               switchOp.getCaseOperands())) {
          if (failed(addBranchConstraints(dest, operands, op)))
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
        if (failed(addReturnConstraints(returnOp)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto callOp = dyn_cast<func::CallOp>(op)) {
        if (failed(addCallConstraints(callOp)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (op->getName().getStringRef() == "func.call_indirect") {
        if (hasVMIValueTypes(op)) {
          op->emitError()
              << kVMIDiagLayoutContractPrefix
              << "VMI typed call requires a direct internal callee with a body";
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
        if (funcOp.empty() && hasVMIFunctionType(funcOp)) {
          funcOp.emitError()
              << kVMIDiagLayoutContractPrefix
              << "VMI typed function declaration requires an explicit "
                 "external ABI materialization plan";
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      return WalkResult::advance();
    });
    return failure(result.wasInterrupted());
  }

  LogicalResult uniteEquivalentValues(Value lhs, Value rhs, Operation *op) {
    if (failed(uniteDataEquivalent(lhs, rhs, op)))
      return failure();
    return uniteMask(lhs, rhs, op);
  }

  LogicalResult addIfConstraints(scf::IfOp ifOp) {
    for (OpResult result : ifOp->getResults()) {
      unsigned resultNo = result.getResultNumber();
      for (Region *region : {&ifOp.getThenRegion(), &ifOp.getElseRegion()}) {
        if (region->empty())
          continue;
        auto yieldOp = dyn_cast<scf::YieldOp>(region->front().getTerminator());
        if (!yieldOp || resultNo >= yieldOp.getNumOperands())
          continue;
        if (failed(uniteEquivalentValues(result, yieldOp.getOperand(resultNo),
                                         ifOp)))
          return failure();
      }
    }
    return success();
  }

  LogicalResult addYieldConstraints(ResultRange results, scf::YieldOp yieldOp,
                                    Operation *op) {
    for (auto [index, result] : llvm::enumerate(results)) {
      if (index >= yieldOp.getNumOperands())
        break;
      if (failed(uniteEquivalentValues(result, yieldOp.getOperand(index), op)))
        return failure();
    }
    return success();
  }

  LogicalResult addExecuteRegionConstraints(scf::ExecuteRegionOp executeOp) {
    WalkResult result = executeOp.getRegion().walk([&](scf::YieldOp yieldOp) {
      if (yieldOp->getParentOp() != executeOp.getOperation())
        return WalkResult::advance();
      if (failed(
              addYieldConstraints(executeOp->getResults(), yieldOp, executeOp)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    return failure(result.wasInterrupted());
  }

  LogicalResult addIndexSwitchConstraints(scf::IndexSwitchOp indexSwitchOp) {
    auto addBlockTerminator = [&](Block &block) -> LogicalResult {
      auto yieldOp = dyn_cast<scf::YieldOp>(block.getTerminator());
      if (!yieldOp)
        return success();
      return addYieldConstraints(indexSwitchOp->getResults(), yieldOp,
                                 indexSwitchOp);
    };

    if (failed(addBlockTerminator(indexSwitchOp.getDefaultBlock())))
      return failure();
    for (unsigned idx = 0, e = indexSwitchOp.getNumCases(); idx < e; ++idx)
      if (failed(addBlockTerminator(indexSwitchOp.getCaseBlock(idx))))
        return failure();
    return success();
  }

  LogicalResult addWhileConstraints(scf::WhileOp whileOp) {
    auto inits = whileOp.getInits();
    auto beforeArgs = whileOp.getBeforeArguments();
    Block &afterBlock = whileOp.getAfter().front();
    auto conditionOp =
        dyn_cast<scf::ConditionOp>(whileOp.getBefore().front().getTerminator());
    auto yieldOp = dyn_cast<scf::YieldOp>(afterBlock.getTerminator());

    for (auto [index, init] : llvm::enumerate(inits)) {
      Value anchor = init;
      if (index < beforeArgs.size() &&
          failed(uniteEquivalentValues(anchor, beforeArgs[index], whileOp)))
        return failure();
      if (conditionOp && index < conditionOp.getArgs().size() &&
          failed(uniteEquivalentValues(anchor, conditionOp.getArgs()[index],
                                       whileOp)))
        return failure();
      if (index < afterBlock.getNumArguments() &&
          failed(uniteEquivalentValues(anchor, afterBlock.getArgument(index),
                                       whileOp)))
        return failure();
      if (yieldOp && index < yieldOp.getNumOperands() &&
          failed(uniteEquivalentValues(anchor, yieldOp.getOperand(index),
                                       whileOp)))
        return failure();
      if (index < whileOp.getNumResults() &&
          failed(
              uniteEquivalentValues(anchor, whileOp.getResult(index), whileOp)))
        return failure();
    }
    return success();
  }

  LogicalResult addForConstraints(scf::ForOp forOp) {
    auto initArgs = forOp.getInitArgs();
    auto regionIterArgs = forOp.getRegionIterArgs();
    auto results = forOp.getResults();
    scf::YieldOp yieldOp = nullptr;
    if (Block *body = forOp.getBody())
      yieldOp = dyn_cast<scf::YieldOp>(body->getTerminator());

    for (auto [index, initArg] : llvm::enumerate(initArgs)) {
      Value anchor = initArg;
      if (index < regionIterArgs.size() &&
          failed(uniteEquivalentValues(anchor, regionIterArgs[index], forOp)))
        return failure();
      if (index < results.size() &&
          failed(uniteEquivalentValues(anchor, results[index], forOp)))
        return failure();
      if (yieldOp && index < yieldOp.getNumOperands() &&
          failed(
              uniteEquivalentValues(anchor, yieldOp.getOperand(index), forOp)))
        return failure();
    }
    return success();
  }

  LogicalResult addBranchConstraints(Block *dest, OperandRange operands,
                                     Operation *op) {
    if (!dest)
      return success();
    for (auto [index, operand] : llvm::enumerate(operands)) {
      if (index >= dest->getNumArguments())
        break;
      if (failed(uniteEquivalentValues(operand, dest->getArgument(index), op)))
        return failure();
    }
    return success();
  }

  LogicalResult addReturnConstraints(func::ReturnOp returnOp) {
    auto func = returnOp->getParentOfType<func::FuncOp>();
    if (!func)
      return success();

    auto it = firstReturnOperandsByFunc.find(func);
    if (it == firstReturnOperandsByFunc.end()) {
      SmallVector<Value> operands(returnOp.getOperands());
      firstReturnOperandsByFunc.try_emplace(func, std::move(operands));
      return success();
    }

    ArrayRef<Value> firstOperands = it->second;
    for (auto [index, operand] : llvm::enumerate(returnOp.getOperands())) {
      if (index >= firstOperands.size())
        break;
      if (failed(
              uniteEquivalentValues(firstOperands[index], operand, returnOp)))
        return failure();
    }
    return success();
  }

  bool hasVMIValueTypes(Operation *op) {
    return llvm::any_of(op->getOperandTypes(), containsVMIType) ||
           llvm::any_of(op->getResultTypes(), containsVMIType);
  }

  bool hasVMIFunctionType(func::FuncOp func) {
    FunctionType type = func.getFunctionType();
    return llvm::any_of(type.getInputs(), containsVMIType) ||
           llvm::any_of(type.getResults(), containsVMIType);
  }

  LogicalResult addCallConstraints(func::CallOp callOp) {
    if (!hasVMIValueTypes(callOp))
      return success();

    auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        callOp, callOp.getCalleeAttr());
    if (!callee || callee.empty())
      return callOp.emitError()
             << kVMIDiagLayoutContractPrefix
             << "VMI typed call requires a direct internal callee with a body";

    for (auto [operand, argument] :
         llvm::zip(callOp.getOperands(), callee.getArguments())) {
      if (failed(uniteEquivalentValues(operand, argument, callOp)))
        return failure();
    }

    SmallVector<func::ReturnOp> returns;
    callee.walk([&](func::ReturnOp returnOp) { returns.push_back(returnOp); });
    for (func::ReturnOp returnOp : returns) {
      for (auto [index, result] : llvm::enumerate(callOp.getResults())) {
        if (index >= returnOp.getNumOperands())
          break;
        if (failed(uniteEquivalentValues(result, returnOp.getOperand(index),
                                         callOp)))
          return failure();
      }
    }
    return success();
  }

  void rewriteDataTypes() {
    for (DataNode &node : dataNodes) {
      VMILayoutAttr layout = getDataLayout(node.value);
      node.value.setType(VMIVRegType::get(ctx, node.type.getElementCount(),
                                          node.type.getElementType(), layout));
    }
  }

  FailureOr<Value> materializeLayoutValue(Value value, Type targetType,
                                          Location loc, OpBuilder &builder) {
    if (value.getType() == targetType)
      return value;

    if (auto sourceType = dyn_cast<VMIVRegType>(value.getType())) {
      auto targetVRegType = dyn_cast<VMIVRegType>(targetType);
      if (!targetVRegType ||
          sourceType.getElementCount() != targetVRegType.getElementCount() ||
          sourceType.getElementType() != targetVRegType.getElementType())
        return failure();
      return builder.create<VMIEnsureLayoutOp>(loc, targetVRegType, value)
          .getResult();
    }

    if (auto sourceType = dyn_cast<VMIMaskType>(value.getType())) {
      auto targetMaskType = dyn_cast<VMIMaskType>(targetType);
      if (!targetMaskType ||
          sourceType.getElementCount() != targetMaskType.getElementCount() ||
          sourceType.getGranularity() != targetMaskType.getGranularity())
        return failure();
      return builder
          .create<VMIEnsureMaskLayoutOp>(loc, targetMaskType, value)
          .getResult();
    }

    return failure();
  }

  SmallVector<Type> getCallResultTypes(func::FuncOp func) {
    SmallVector<Type> resultTypes;
    bool found = false;
    module.walk([&](func::CallOp call) {
      if (call.getCallee() != func.getSymName())
        return;
      if (!found) {
        resultTypes.assign(call.getResultTypes().begin(),
                           call.getResultTypes().end());
        found = true;
        return;
      }
      if (resultTypes.size() != call.getNumResults())
        return;
      for (auto [index, type] : llvm::enumerate(call.getResultTypes()))
        if (index < resultTypes.size() && resultTypes[index] != type)
          resultTypes[index] = {};
    });
    return found ? resultTypes : SmallVector<Type>{};
  }

  LogicalResult materializeCallBoundaries() {
    IRRewriter rewriter(ctx);

    WalkResult callResult = module.walk([&](func::CallOp call) -> WalkResult {
      auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          call, call.getCalleeAttr());
      if (!callee || callee.empty())
        return WalkResult::advance();

      rewriter.setInsertionPoint(call);
      for (auto [index, operand] : llvm::enumerate(call.getOperands())) {
        if (index >= callee.getNumArguments())
          break;
        Type targetType = callee.getArgument(index).getType();
        if (!isa<VMIVRegType, VMIMaskType>(targetType))
          continue;
        FailureOr<Value> materialized =
            materializeLayoutValue(operand, targetType, call.getLoc(),
                                   rewriter);
        if (failed(materialized))
          return WalkResult::interrupt();
        call->setOperand(index, *materialized);
      }
      return WalkResult::advance();
    });
    if (callResult.wasInterrupted())
      return failure();

    WalkResult returnResult = module.walk([&](func::FuncOp func) -> WalkResult {
      SmallVector<Type> resultTypes = getCallResultTypes(func);
      if (resultTypes.empty())
        return WalkResult::advance();

      WalkResult nested = func.walk([&](func::ReturnOp ret) -> WalkResult {
        rewriter.setInsertionPoint(ret);
        for (auto [index, operand] : llvm::enumerate(ret.getOperands())) {
          if (index >= resultTypes.size())
            break;
          if (!resultTypes[index])
            continue;
          Type targetType = resultTypes[index];
          if (!isa<VMIVRegType, VMIMaskType>(targetType))
            continue;
          FailureOr<Value> materialized =
              materializeLayoutValue(operand, targetType, ret.getLoc(),
                                     rewriter);
          if (failed(materialized))
            return WalkResult::interrupt();
          ret->setOperand(index, *materialized);
        }
        return WalkResult::advance();
      });
      return nested.wasInterrupted() ? WalkResult::interrupt()
                                     : WalkResult::advance();
    });
    return failure(returnResult.wasInterrupted());
  }

  LogicalResult insertDataUseMaterializations() {
    OpBuilder builder(ctx);
    for (DataUseRequest request : dataUseRequests) {
      Value value = request.operand->get();
      auto sourceType = dyn_cast<VMIVRegType>(value.getType());
      if (!sourceType)
        continue;
      VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
      if (!sourceLayout)
        return request.operand->getOwner()->emitError()
               << kVMIDiagLayoutContractPrefix
               << "data use materialization requires layout-assigned source "
                  "type";
      if (sourceLayout == request.layout)
        continue;

      auto resultType =
          VMIVRegType::get(ctx, sourceType.getElementCount(),
                           sourceType.getElementType(), request.layout);
      builder.setInsertionPoint(request.operand->getOwner());
      auto ensure = builder.create<VMIEnsureLayoutOp>(
          request.operand->getOwner()->getLoc(), resultType, value);
      request.operand->set(ensure.getResult());
    }
    return success();
  }

  bool hasRequestedLayout(VMILayoutPropagator &propagator, Value value) {
    return static_cast<bool>(propagator.getRequestedLayout(value));
  }

  bool hasLayoutAssignment(VMILayoutPropagator &propagator, Value value) {
    return propagator.lookup(value) != nullptr;
  }

  LogicalResult requestDataLayoutSeeds(VMILayoutPropagator &propagator,
                                       DataLayoutSeedPhase phase,
                                       bool skipAlreadyRequested) {
    SmallVector<Value, 16> protectedValues;
    if (skipAlreadyRequested) {
      for (DataLayoutSeed seed : dataLayoutSeeds) {
        if (seed.phase != phase)
          continue;
        if (hasRequestedLayout(propagator, seed.value) &&
            !llvm::is_contained(protectedValues, seed.value))
          protectedValues.push_back(seed.value);
      }
    }

    for (DataLayoutSeed seed : dataLayoutSeeds) {
      if (seed.phase != phase)
        continue;
      if (llvm::is_contained(protectedValues, seed.value))
        continue;
      if (failed(propagator.request(seed.value, seed.layout)))
        return failure();
    }
    return success();
  }

  LogicalResult requestDataUseSeeds(VMILayoutPropagator &propagator,
                                    DataLayoutSeedPhase phase, bool late) {
    for (DataUseRequest request : dataUseRequests)
      if (request.phase == phase && request.late == late) {
        if (hasLayoutAssignment(propagator, request.operand->get())) {
          VMILayoutAttr assigned =
              propagator.getRequestedOrCurrentLayout(request.operand->get());
          if (propagator.canUseOperandLayout(*request.operand, assigned))
            continue;
        }
        if (failed(propagator.request(*request.operand, request.layout)))
          return failure();
      }
    return success();
  }

  LogicalResult requestMaskUseSeeds(VMILayoutPropagator &propagator,
                                    DataLayoutSeedPhase phase) {
    for (MaskUseRequest request : maskUseRequests)
      if (request.phase == phase) {
        if (hasLayoutAssignment(propagator, request.operand->get())) {
          VMILayoutAttr assigned =
              propagator.getRequestedOrCurrentLayout(request.operand->get());
          if (propagator.canUseOperandLayout(*request.operand, assigned))
            continue;
        }
        if (failed(propagator.request(*request.operand, request.layout)))
          return failure();
      }
    return success();
  }

  LogicalResult runSeedPhase(VMILayoutPropagator &propagator,
                             DataLayoutSeedPhase phase) {
    if (failed(requestDataLayoutSeeds(propagator, phase,
                                      /*skipAlreadyRequested=*/true)))
      return failure();
    if (failed(requestDataUseSeeds(propagator, phase, /*late=*/false)))
      return failure();
    if (failed(requestMaskUseSeeds(propagator, phase)))
      return failure();
    return propagator.run();
  }

  LogicalResult applyLayouts() {
    VMILayoutPropagator propagator(module);
    for (DataNode &node : dataNodes) {
      DataNode &root = dataNodes[find(dataIds.lookup(node.value))];
      propagator.addEquivalentValues(root.value, node.value);
    }
    for (MaskNode &node : maskNodes) {
      MaskNode &root = maskNodes[findMask(maskIds.lookup(node.value))];
      propagator.addEquivalentValues(root.value, node.value);
    }
    if (failed(requestDataLayoutSeeds(propagator, DataLayoutSeedPhase::Explicit,
                                      /*skipAlreadyRequested=*/false)))
      return failure();
    for (MaskNode &node : maskNodes) {
      MaskNode &root = maskNodes[findMask(maskIds.lookup(node.value))];
      if (root.requestedLayout &&
          failed(propagator.request(node.value, root.requestedLayout)))
        return failure();
    }
    if (failed(propagator.run()))
      return failure();

    for (int64_t phase = static_cast<int64_t>(DataLayoutSeedPhase::SeedStart);
         phase < static_cast<int64_t>(DataLayoutSeedPhase::SeedEnd); ++phase)
      if (failed(runSeedPhase(propagator,
                              static_cast<DataLayoutSeedPhase>(phase))))
        return failure();

    for (GroupStoreUseRequest request : groupStoreUseRequests) {
      VMIGroupStoreOp store = request.store;
      VMILayoutAttr layout = getPreferredGroupStoreUseLayout(
          store.getValue(), store.getNumGroupsAttr().getInt(),
          store.getRowStride(), [&](Value value) {
            return propagator.getRequestedOrCurrentLayout(value);
          });
      if (failed(propagator.request(store.getValueMutable(), layout)))
        return failure();
    }
    if (failed(propagator.run()))
      return failure();

    for (DataUseRequest request : dataUseRequests)
      if (request.late &&
          failed(propagator.request(*request.operand, request.layout)))
        return failure();
    if (failed(propagator.run()))
      return failure();

    for (DataNode &node : dataNodes)
      if (!propagator.getRequestedLayout(node.value) &&
          failed(propagator.request(node.value, getContiguousLayout())))
        return failure();
    for (MaskNode &node : maskNodes)
      if (!propagator.getRequestedLayout(node.value) &&
          failed(propagator.request(node.value, getContiguousLayout())))
        return failure();
    if (failed(propagator.run()))
      return failure();

    IRRewriter rewriter(ctx);
    return propagator.apply(rewriter);
  }

  void rewriteFunctionType() {
    module.walk([&](func::FuncOp func) {
      if (func.empty())
        return;

      SmallVector<Type> inputs;
      inputs.reserve(func.getNumArguments());
      for (BlockArgument arg : func.getArguments())
        inputs.push_back(arg.getType());

      SmallVector<Type> results;
      auto it = firstReturnOperandsByFunc.find(func);
      SmallVector<Type> callResultTypes = getCallResultTypes(func);
      if (!callResultTypes.empty()) {
        for (Type type : callResultTypes)
          results.push_back(type);
      } else if (it != firstReturnOperandsByFunc.end()) {
        for (Value operand : it->second)
          results.push_back(operand.getType());
      } else {
        FunctionType functionType = func.getFunctionType();
        for (Type type : functionType.getResults()) {
          if (auto vregType = dyn_cast<VMIVRegType>(type)) {
            results.push_back(VMIVRegType::get(ctx, vregType.getElementCount(),
                                               vregType.getElementType(),
                                               getContiguousLayout()));
          } else if (auto maskType = dyn_cast<VMIMaskType>(type)) {
            results.push_back(VMIMaskType::get(ctx, maskType.getElementCount(),
                                               "b32", getContiguousLayout()));
          } else {
            results.push_back(type);
          }
        }
      }

      func.setFunctionType(FunctionType::get(ctx, inputs, results));
    });
  }

  LogicalResult run() {
    if (failed(collect()))
      return failure();
    if (failed(addConstraints()))
      return failure();
    if (failed(applyLayouts()))
      return failure();
    if (failed(materializeCallBoundaries()))
      return failure();
    rewriteFunctionType();
    return validateVMILayoutAssignedIR(module, /*diagOS=*/nullptr,
                                       /*verifyHelperSupport=*/false);
  }

  ModuleOp module;
  MLIRContext *ctx;
  DenseMap<Value, unsigned> dataIds;
  DenseMap<Value, unsigned> maskIds;
  DenseMap<func::FuncOp, SmallVector<Value>> firstReturnOperandsByFunc;
  SmallVector<DataNode> dataNodes;
  SmallVector<MaskNode> maskNodes;
  SmallVector<DataLayoutSeed> dataLayoutSeeds;
  SmallVector<DataUseRequest> dataUseRequests;
  SmallVector<GroupStoreUseRequest> groupStoreUseRequests;
  SmallVector<MaskUseRequest> maskUseRequests;
};

struct VMILayoutAssignmentPass
    : public mlir::pto::impl::VMILayoutAssignmentBase<VMILayoutAssignmentPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VMILayoutAssignmentPass)

  void runOnOperation() override {
    if (failed(LayoutSolver(getOperation()).run()))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVMILayoutAssignmentPass() {
  return std::make_unique<VMILayoutAssignmentPass>();
}

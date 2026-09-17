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
#include "PTO/Support/CodeConstants.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/VMIControlFlowSupport.h"
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
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"

#include <type_traits>

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
  CompactCast,
  GroupStore,
  LaneStrideNarrowCast,
  // Cast layout preferences must get a chance to constrain downstream
  // cast/cast-back chains before the group-broadcast natural seed.
  // Otherwise a deinterleaved group-broadcast seed anchors the whole
  // equivalence class and the preferred C -> LS4 -> C path cannot be
  // selected.
  GroupBroadcastLoad,
  Cast,
  WeakReduce,
  Store,
  Other,
  GroupStoreFallback,
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

static std::optional<int64_t> getConstantIndexValue(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>()) {
    return constant.value();
  }
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto integerAttr = dyn_cast<IntegerAttr>(constant.getValue())) {
      return integerAttr.getInt();
    }
  }
  return std::nullopt;
}

static bool isLane0SplatShuffle(VMIShuffleOp op) {
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  ArrayRef<int64_t> indices = op.getIndices();
  return sourceType.getElementCount() == 1 && !indices.empty() &&
         llvm::all_of(indices, [](int64_t index) { return index == 0; });
}

bool containsVMIType(Type type) {
  if (isa<VMIVRegType, VMIMaskType>(type)) {
    return true;
  }
  if (auto functionType = dyn_cast<FunctionType>(type)) {
    return llvm::any_of(functionType.getInputs(),
                        [](Type input) { return containsVMIType(input); }) ||
           llvm::any_of(functionType.getResults(),
                        [](Type result) { return containsVMIType(result); });
  }
  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    return containsVMIType(shapedType.getElementType());
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Direction-spine recognition.
//
// A cast *spine* is a run of cast legs that are reachable from one another
// through layout-transparent VMI ops (the ops for which isVMISameLayoutOp is
// true: elementwise arithmetic, lane-local maths, bitcasts, ...).  Each leg is
// directed by the storage element bit width of its operand and result:
// widening is "up" (extf / extsi / extui) and narrowing is "down" (truncf /
// trunci).  The predicate below therefore never looks at the element type, the
// element count or the shape - only at the direction sequence of the spine.
//
// The recognised shape is a *closed nested round trip*:
//
//     <up, down, up, down>   with   elem(leg0.source) == elem(leg3.result)
//
// i.e. the spine leaves an element type, is quantised down and widened back
// twice, and returns to the element type it started from (bf16 -> f32 -> f8 ->
// f32 -> bf16).  Only such a chain has a deinterleaved round trip worth keeping
// consistent end to end.
//
// In addition the narrow intermediate (leg1.result) must be a *pure* round trip
// inside the window: every use of it must be the closing widening leg (leg2) of
// the same window and nothing else.  The deinterleaved family is only the right
// choice for such a handoff, because only then can the narrow value stay split
// over the four physical parts of its wide side and be read back one-to-one by
// the widening leg (see the spine-scoped composite cast rows in
// VMILayoutSupport).  A chain whose narrow value is also stored, or fed to an
// elementwise op, or handed to a cast outside the window, would have to be
// packed back into a contiguous narrow value - which on this target costs more
// than the dist-encoded UB loads and stores the flip removes.  Those chains
// therefore keep the pre-existing (lane-stride) family.
//
// Deliberately NOT recognised, so that these keep the pre-existing behaviour:
//   * <down, up>  - a single round trip (f32 -> f8 -> f32) is already optimal;
//   * <up>        - a chain that only widens has no round trip at all;
//   * <up, down>  - a chain that widens and narrows again without a closing
//                   widening leg never returns to a deinterleaved narrow form;
//   * <up, down, up, down> whose leg1.result has any use other than leg2.
//===----------------------------------------------------------------------===//

constexpr unsigned kDirectionSpineLegCount = 4;
constexpr unsigned kDirectionSpineMaxHops = 8;
constexpr unsigned kDirectionSpineSetInlineCapacity = 8;
constexpr unsigned kDirectionSpineSearchBudget = 512;

// Whether the direction-spine recognition is allowed to seed layouts.
//
// ON by default: the deinterleaved family it selects is the intended shape for
// the bf16 quantise/dequantise chain, and the two regression tests that used to
// assert the pre-existing layout for exactly that chain were updated together
// with this default flip:
//   test/lit/vmi_new/opt/fused_quant_dequant_vmi_opt.pto            (VPTO)
//   test/lit/vmi_new/vmi_layout_assignment_cast_roundtrip.pto       (ASSIGN)
// The ASSIGN test keeps a single-round-trip region per narrow element type as
// the counterexample set proving the predicate did not widen; the LOWER side of
// that test scopes its "no pto.vor" guard to those counterexamples, because
// merging a deinterleaved value back to contiguous order legitimately uses
// mask + or for the nested case only.
//
// Pass --vmi-prefer-cast-spine-round-trip=false to recover the pre-recognition
// layout for the closed nested round-trip chain (used by offline A/B scans).
//
// This switch is independent of --vmi-prefer-lane-stride-narrowing: a matched
// window is reseeded from the spine-scoped cast table, which pins the two outer
// legs (16<->32) to the deinterleaved family and, when the narrow handoff is
// pure, also pins the two inner legs (32<->16 or 32<->8) to the composite
// deinterleaved lane-stride form.  A window that does not match keeps the
// pre-existing tables - including the lane-stride narrowing ones, which that
// switch still controls.
static llvm::cl::opt<bool> preferCastSpineRoundTrip(
    "vmi-prefer-cast-spine-round-trip",
    llvm::cl::desc("Seed the deinterleaved family for closed nested round-trip "
                   "cast chains (widening, narrowing, widening, narrowing back "
                   "to the starting element type).  A chain is only recognised "
                   "when the narrow handoff is pure: the truncation result must "
                   "feed nothing but the matching extension of the same chain"),
    llvm::cl::init(true));

struct DirectionSpineLeg {
  Operation *op = nullptr;
  Value source;
  Value result;
  Type sourceElementType;
  Type resultElementType;
  bool widening = false;
};

static bool getDirectionSpineLeg(Operation *op, DirectionSpineLeg &leg) {
  Value source;
  Value result;
  if (auto extf = dyn_cast<VMIExtFOp>(op)) {
    source = extf.getSource();
    result = extf.getResult();
  } else if (auto extsi = dyn_cast<VMIExtSIOp>(op)) {
    source = extsi.getSource();
    result = extsi.getResult();
  } else if (auto extui = dyn_cast<VMIExtUIOp>(op)) {
    source = extui.getSource();
    result = extui.getResult();
  } else if (auto truncf = dyn_cast<VMITruncFOp>(op)) {
    source = truncf.getSource();
    result = truncf.getResult();
  } else if (auto trunci = dyn_cast<VMITruncIOp>(op)) {
    source = trunci.getSource();
    result = trunci.getResult();
  } else {
    return false;
  }

  auto sourceType = dyn_cast<VMIVRegType>(source.getType());
  auto resultType = dyn_cast<VMIVRegType>(result.getType());
  if (!sourceType || !resultType) {
    return false;
  }
  unsigned sourceBits =
      pto::getPTOStorageElemBitWidth(sourceType.getElementType());
  unsigned resultBits =
      pto::getPTOStorageElemBitWidth(resultType.getElementType());
  if (sourceBits == 0 || resultBits == 0 || sourceBits == resultBits) {
    return false;
  }

  leg.op = op;
  leg.source = source;
  leg.result = result;
  leg.sourceElementType = sourceType.getElementType();
  leg.resultElementType = resultType.getElementType();
  leg.widening = resultBits > sourceBits;
  return true;
}

// Whether the narrow intermediate produced by \p downLeg is handed straight to
// the closing widening leg \p upLeg of the same window and to nothing else.
// This is the sufficiency condition for the deinterleaved family (see the
// recognition comment above): the caller only needs the use-def list of one
// value, and it is decided before any seed exists.
static bool isPureNarrowHandoff(const DirectionSpineLeg &downLeg,
                                const DirectionSpineLeg &upLeg) {
  if (downLeg.result.use_empty()) {
    return false;
  }
  for (OpOperand &use : downLeg.result.getUses()) {
    if (use.getOwner() != upLeg.op) {
      return false;
    }
  }
  return true;
}

// Extend a partially built leg window forward from \p current.  Legs alternate
// up/down starting with up, and between two legs only layout-transparent ops
// are crossed (bounded by kDirectionSpineMaxHops per leg and by \p budget per
// search).  A complete window is recorded only when it closes - the fourth leg
// ends on the element type the first leg started from - and when its narrow
// handoff is pure.
static void extendDirectionSpineWindow(
    Value current, SmallVectorImpl<DirectionSpineLeg> &window,
    llvm::SmallPtrSetImpl<Operation *> &spineLegs, unsigned hops,
    unsigned &budget) {
  if (budget == 0) {
    return;
  }
  if (window.size() == kDirectionSpineLegCount) {
    if (window.front().sourceElementType == window.back().resultElementType &&
        isPureNarrowHandoff(window[1], window[2])) {
      for (const DirectionSpineLeg &leg : window) {
        spineLegs.insert(leg.op);
      }
    }
    return;
  }

  auto currentType = dyn_cast<VMIVRegType>(current.getType());
  if (!currentType) {
    return;
  }
  const bool expectWidening = window.size() % 2 == 0;
  for (OpOperand &use : current.getUses()) {
    if (budget == 0) {
      return;
    }
    --budget;
    Operation *user = use.getOwner();

    DirectionSpineLeg next;
    if (getDirectionSpineLeg(user, next) && next.source == current &&
        next.widening == expectWidening) {
      window.push_back(next);
      extendDirectionSpineWindow(next.result, window, spineLegs, /*hops=*/0,
                                 budget);
      window.pop_back();
      continue;
    }

    if (hops < kDirectionSpineMaxHops && isVMISameLayoutOp(user)) {
      for (Value result : user->getResults()) {
        auto resultType = dyn_cast<VMIVRegType>(result.getType());
        if (!resultType ||
            resultType.getElementType() != currentType.getElementType()) {
          continue;
        }
        extendDirectionSpineWindow(result, window, spineLegs, hops + 1, budget);
      }
    }
  }
}

// Record every cast op that is a leg of a closed nested round trip.
static void collectDirectionSpineLegs(
    ModuleOp module, llvm::SmallPtrSetImpl<Operation *> &spineLegs) {
  SmallVector<DirectionSpineLeg, kDirectionSpineLegCount> window;
  module.walk([&](Operation *op) {
    DirectionSpineLeg first;
    if (!getDirectionSpineLeg(op, first) || !first.widening) {
      return;
    }
    unsigned budget = kDirectionSpineSearchBudget;
    window.clear();
    window.push_back(first);
    extendDirectionSpineWindow(first.result, window, spineLegs, /*hops=*/0,
                               budget);
  });
}

// Whether \p op is a widening cast, i.e. the kind of op that closes a
// narrow->wide handoff.
static bool isSpineWideningCast(Operation *op) {
  return isa<VMIExtFOp, VMIExtSIOp, VMIExtUIOp>(op);
}

// Narrow->wide handoff inside a matched direction spine.
// The composite 32<->16 / 32<->8 spine rows keep the narrow value split over
// the four physical parts of its wide side so that the narrowing and the closing
// widening are per-chunk one-to-one.  That is only safe while the narrow value
// has no other consumer: an elementwise op, a store, a broadcast or any layout
// request for a packed/contiguous narrow value would disagree with the
// composite form and force an ensure_layout (or a residual op), which is
// strictly worse than the assembly it was meant to remove.  Requiring every use
// of the narrowing leg's result to be a widening cast that is itself a leg of
// the same spine is therefore a cheap, local sufficiency condition.
// The set is computed from the IR alone, before any seed exists, and is empty
// whenever the direction-spine recognition is off or did not match.
static void collectSpineScopedCasts(
    const llvm::SmallPtrSetImpl<Operation *> &spineLegs,
    llvm::SmallPtrSetImpl<Operation *> &scopedCasts) {
  for (Operation *op : spineLegs) {
    if (!isa<VMITruncFOp, VMITruncIOp>(op) || op->getNumResults() != 1) {
      continue;
    }
    Value narrow = op->getResult(0);
    if (narrow.use_empty()) {
      continue;
    }
    bool handoff = true;
    for (OpOperand &use : narrow.getUses()) {
      Operation *user = use.getOwner();
      if (!isSpineWideningCast(user) || !spineLegs.contains(user)) {
        handoff = false;
        break;
      }
    }
    if (!handoff) {
      continue;
    }
    scopedCasts.insert(op);
    for (OpOperand &use : narrow.getUses()) {
      scopedCasts.insert(use.getOwner());
    }
  }
}

struct LayoutSolver {
  explicit LayoutSolver(ModuleOp module)
      : module(module), ctx(module.getContext()) {}

  unsigned addDataValue(Value value) {
    auto type = dyn_cast<VMIVRegType>(value.getType());
    if (!type) {
      return ~0U;
    }
    auto [it, inserted] = dataIds.try_emplace(value, dataNodes.size());
    if (inserted) {
      dataNodes.push_back(
          DataNode{value, type, it->second, type.getLayoutAttr(), {}});
      if (type.getLayoutAttr()) {
        dataLayoutSeeds.push_back(DataLayoutSeed{
            value, type.getLayoutAttr(), DataLayoutSeedPhase::Explicit});
      }
    }
    return it->second;
  }

  unsigned addMaskValue(Value value) {
    auto type = dyn_cast<VMIMaskType>(value.getType());
    if (!type) {
      return ~0U;
    }
    auto [it, inserted] = maskIds.try_emplace(value, maskNodes.size());
    if (inserted) {
      maskNodes.push_back(
          MaskNode{value, type, it->second, type.getLayoutAttr()});
    }
    return it->second;
  }

  unsigned find(unsigned id) {
    if (dataNodes[id].parent == id) {
      return id;
    }
    dataNodes[id].parent = find(dataNodes[id].parent);
    return dataNodes[id].parent;
  }

  unsigned findMask(unsigned id) {
    if (maskNodes[id].parent == id) {
      return id;
    }
    maskNodes[id].parent = findMask(maskNodes[id].parent);
    return maskNodes[id].parent;
  }

  LogicalResult unite(Value lhs, Value rhs, const Operation *op) {
    (void)op;
    addDataValue(lhs);
    addDataValue(rhs);
    return success();
  }

  LogicalResult uniteDataEquivalent(Value lhs, Value rhs, Operation *op) {
    unsigned lhsId = addDataValue(lhs);
    unsigned rhsId = addDataValue(rhs);
    if (lhsId == ~0U || rhsId == ~0U) {
      return success();
    }
    unsigned lhsRoot = find(lhsId);
    unsigned rhsRoot = find(rhsId);
    if (lhsRoot == rhsRoot) {
      return success();
    }

    DataNode &lhsNode = dataNodes[lhsRoot];
    DataNode &rhsNode = dataNodes[rhsRoot];
    if (lhsNode.naturalLayout && rhsNode.naturalLayout &&
        lhsNode.naturalLayout != rhsNode.naturalLayout) {
      return op->emitError()
             << kVMIDiagLayoutContractPrefix << "conflicting natural layouts "
             << lhsNode.naturalLayout << " and " << rhsNode.naturalLayout;
    }
    if (lhsNode.preferredLayout && rhsNode.preferredLayout &&
        lhsNode.preferredLayout != rhsNode.preferredLayout) {
      return op->emitError()
             << kVMIDiagLayoutContractPrefix << "conflicting preferred layouts "
             << lhsNode.preferredLayout << " and " << rhsNode.preferredLayout;
    }

    rhsNode.parent = lhsRoot;
    if (!lhsNode.naturalLayout) {
      lhsNode.naturalLayout = rhsNode.naturalLayout;
    }
    if (!lhsNode.preferredLayout) {
      lhsNode.preferredLayout = rhsNode.preferredLayout;
    }
    return success();
  }

  LogicalResult uniteMask(Value lhs, Value rhs, Operation *op) {
    unsigned lhsId = addMaskValue(lhs);
    unsigned rhsId = addMaskValue(rhs);
    if (lhsId == ~0U || rhsId == ~0U) {
      return success();
    }
    unsigned lhsRoot = findMask(lhsId);
    unsigned rhsRoot = findMask(rhsId);
    if (lhsRoot == rhsRoot) {
      return success();
    }

    MaskNode &lhsNode = maskNodes[lhsRoot];
    MaskNode &rhsNode = maskNodes[rhsRoot];
    if (lhsNode.requestedLayout && rhsNode.requestedLayout &&
        lhsNode.requestedLayout != rhsNode.requestedLayout) {
      return op->emitError()
             << kVMIDiagLayoutContractPrefix << "conflicting mask layouts "
             << lhsNode.requestedLayout << " and " << rhsNode.requestedLayout;
    }
    rhsNode.parent = lhsRoot;
    if (!lhsNode.requestedLayout) {
      lhsNode.requestedLayout = rhsNode.requestedLayout;
    }
    return success();
  }

  LogicalResult
  setNaturalLayout(Value value, VMILayoutAttr layout, Operation *op,
                   DataLayoutSeedPhase phase = DataLayoutSeedPhase::Other) {
    unsigned id = addDataValue(value);
    if (id == ~0U || !layout) {
      return success();
    }
    unsigned root = find(id);
    VMILayoutAttr existing = dataNodes[root].naturalLayout;
    if (existing && existing != layout) {
      return op->emitError()
             << kVMIDiagLayoutContractPrefix << "conflicting natural layouts "
             << existing << " and " << layout;
    }
    dataNodes[root].naturalLayout = layout;
    dataLayoutSeeds.push_back(DataLayoutSeed{value, layout, phase});
    return success();
  }

  LogicalResult
  setPreferredLayout(Value value, VMILayoutAttr layout, Operation *op,
                     DataLayoutSeedPhase phase = DataLayoutSeedPhase::Other) {
    unsigned id = addDataValue(value);
    if (id == ~0U || !layout) {
      return success();
    }
    unsigned root = find(id);
    VMILayoutAttr existing = dataNodes[root].preferredLayout;
    if (existing && existing != layout) {
      return op->emitError()
             << kVMIDiagLayoutContractPrefix << "conflicting preferred layouts "
             << existing << " and " << layout;
    }
    dataNodes[root].preferredLayout = layout;
    dataLayoutSeeds.push_back(DataLayoutSeed{value, layout, phase});
    return success();
  }

  VMILayoutAttr getContiguousLayout() const {
    return VMILayoutAttr::getContiguous(ctx);
  }

  DataLayoutSeedPhase getCastSeedPhase(const VMICastLayoutFact &fact) const {
    if (fact.priority == VMICastLayoutPriority::High) {
      return DataLayoutSeedPhase::CompactCast;
    }
    if (fact.priority == VMICastLayoutPriority::LaneStrideNarrowing) {
      return DataLayoutSeedPhase::LaneStrideNarrowCast;
    }
    return DataLayoutSeedPhase::Cast;
  }

  // Whether the seed for \p value can still be applied without conflicting
  // with a preferred layout that is already pinned on the same value.
  bool canSeedPreferredLayout(Value value, VMILayoutAttr layout) {
    if (!layout) {
      return false;
    }
    unsigned id = addDataValue(value);
    if (id == ~0U) {
      return false;
    }
    VMILayoutAttr existing = dataNodes[find(id)].preferredLayout;
    return !existing || existing == layout;
  }

  // Cast layout fact used to seed one cast leg.  Only the legs the peephole
  // proved to be a narrow->wide handoff (the narrowing leg plus the widening
  // legs that consume it) are served from the spine-scoped table, which adds the
  // composite 32<->16 / 32<->8 rows on top of the layouts the generic tables
  // already offer.  Every other cast keeps the pre-existing table lookup and
  // phase unchanged.
  FailureOr<VMICastLayoutFact> getCastLayoutFactForSeed(Operation *op,
                                                        Value seedValue,
                                                        VMIVRegType sourceType,
                                                        VMIVRegType resultType) {
    VMILayoutSupport supports;
    if (!spineScopedCasts.contains(op)) {
      return supports.getPreferredCastLayoutFact(sourceType, resultType);
    }

    std::string reason;
    FailureOr<VMICastLayoutFact> spineFact =
        supports.getSpineScopedCastLayoutFact(sourceType, resultType, &reason);
    if (succeeded(spineFact) &&
        canSeedPreferredLayout(seedValue, spineFact->resultLayout)) {
      return spineFact;
    }
    return supports.getPreferredCastLayoutFact(sourceType, resultType);
  }

  VMILayoutAttr getPreferredDenseStoreLayout(VMIVRegType type) const {
    VMILayoutSupport supports;
    FailureOr<VMIStoreLayoutFact> fact =
        supports.getPreferredStoreLayoutFact(type);
    if (failed(fact)) {
      return {};
    }
    return fact->valueLayout;
  }

  bool hasDataLayoutSeed(Value value) {
    unsigned id = addDataValue(value);
    if (id == ~0U) {
      return false;
    }
    DataNode &node = dataNodes[find(id)];
    return static_cast<bool>(node.naturalLayout || node.preferredLayout);
  }

  FailureOr<VMIMaskedStoreLayoutFact>
  getPreferredDenseMaskedStoreLayout(VMIVRegType valueType,
                                     VMIMaskType maskType) const {
    VMILayoutSupport supports;
    return supports.getPreferredMaskedStoreLayoutFact(valueType, maskType);
  }

  VMILayoutAttr getGroupSlotsLayout(int64_t numGroups) const {
    return VMILayoutAttr::getGroupSlots(ctx, numGroups);
  }

  VMILayoutAttr getPreferredGroupSlotsLayout(VMIVRegType type,
                                             int64_t numGroups) const {
    if (VMILayoutAttr existing = type.getLayoutAttr()) {
      if (existing.isGroupSlots() && existing.getSlots() > 0) {
        return existing;
      }
    }
    VMILayoutSupport supports;
    FailureOr<VMIGroupReduceLayoutFact> fact =
        supports.getPreferredGroupReduceLayoutFact(type, numGroups);
    if (succeeded(fact)) {
      return fact->resultLayout;
    }
    return getGroupSlotsLayout(numGroups);
  }

  VMILayoutAttr getPreferredGroupReduceSourceLayout(VMIVRegType type,
                                                    int64_t numGroups) const {
    if (VMILayoutAttr existing = type.getLayoutAttr()) {
      return existing;
    }
    VMILayoutSupport supports;
    FailureOr<VMIGroupReduceLayoutFact> fact =
        supports.getPreferredGroupReduceLayoutFact(type, numGroups);
    if (succeeded(fact)) {
      return fact->sourceLayout;
    }
    return getContiguousLayout();
  }

  DataLayoutSeedPhase
  getGroupReduceUseSeedPhase(VMIVRegType sourceType, int64_t numGroups,
                             VMIGroupReduceLayoutFact fact) const {
    if (!fact.sourceLayout || !fact.sourceLayout.isContiguous() ||
        fact.sourceLayout.getLaneStride() != 1) {
      return DataLayoutSeedPhase::Reduce;
    }

    VMILayoutSupport supports;
    FailureOr<SmallVector<VMIGroupReduceLayoutFact, mlir::pto::kValue4>> resultFacts =
        supports.getGroupReduceLayoutFactsForLayout(
            sourceType, numGroups, VMIGroupReduceLayoutPort::Result,
            fact.resultLayout);
    if (succeeded(resultFacts) && resultFacts->size() > 1) {
      return DataLayoutSeedPhase::WeakReduce;
    }
    return DataLayoutSeedPhase::Reduce;
  }

  VMILayoutAttr getPreferredGroupSlotLoadLayout(VMIGroupSlotLoadOp op) const {
    auto type = cast<VMIVRegType>(op.getResult().getType());
    int64_t numGroups = op.getNumGroupsAttr().getInt();
    if (VMILayoutAttr existing = type.getLayoutAttr()) {
      if (existing.isGroupSlots() && existing.getSlots() > 0) {
        return existing;
      }
    }
    std::optional<int64_t> sourceGroupStride =
        getConstantIndexValue(op.getSourceGroupStride());
    if (sourceGroupStride && *sourceGroupStride == 1) {
      return VMILayoutAttr::getGroupSlots(ctx, numGroups, /*slots=*/mlir::pto::kValue8);
    }
    return VMILayoutAttr::getGroupSlots(ctx, numGroups, /*slots=*/1);
  }

  VMILayoutAttr
  getPreferredGroupBroadcastLoadLayout(VMIGroupBroadcastLoadOp op) const {
    auto type = cast<VMIVRegType>(op.getResult().getType());
    if (VMILayoutAttr existing = type.getLayoutAttr()) {
      return existing;
    }

    VMILayoutSupport supports;
    FailureOr<VMIGroupBroadcastLoadDirectFact> fact =
        supports.getGroupBroadcastLoadDirectFact(
            type, op.getSource().getType(), op.getSourceGroupStride(),
            op.getNumGroupsAttr().getInt());
    if (failed(fact)) {
      return {};
    }
    VMILayoutAttr directLayout = fact->layout.resultLayout;
    // A direct E2B packet fills exactly one physical part. When the
    // contiguous form of this broadcast spans multiple physical chunks, the
    // direct table can only offer a deinterleaved split layout. Prefer the
    // generic contiguous lowering (group_slots -> contiguous) so consumers
    // such as plain elementwise vmul can stay contiguous and avoid
    // vldsx2/vintlv-style deinterleave materialization.
    if (fact->kind == VMIGroupBroadcastLoadDirectKind::E2B &&
        directLayout.isDeinterleaved()) {
      VMILayoutAttr contiguous = getContiguousLayout();
      auto contiguousType = VMIVRegType::get(
          ctx, type.getElementCount(), type.getElementType(), contiguous);
      FailureOr<int64_t> contiguousArity = getVMIPhysicalArity(contiguousType);
      if (succeeded(contiguousArity) && *contiguousArity > 1) {
        FailureOr<
            SmallVector<VMIGroupBroadcastLayoutFact, mlir::pto::kValue4>>
            contiguousFacts = supports.getGroupBroadcastLayoutFactsForLayout(
                type, contiguousType, op.getNumGroupsAttr().getInt(),
                VMIGroupBroadcastLayoutPort::Result, contiguous);
        if (succeeded(contiguousFacts) && !contiguousFacts->empty()) {
          return contiguous;
        }
      }
    }
    return directLayout;
  }

  VMILayoutAttr getPreferredGroupBroadcastSourceLayout(Value value,
                                                       int64_t numGroups) {
    auto type = dyn_cast<VMIVRegType>(value.getType());
    if (!type) {
      return getContiguousLayout();
    }
    if (VMILayoutAttr existing = type.getLayoutAttr()) {
      if (existing.isGroupSlots() && existing.getSlots() > 0) {
        return existing;
      }
    }
    VMILayoutAttr solved = getDataLayout(value);
    if (solved && solved.isGroupSlots() && solved.getNumGroups() == numGroups &&
        solved.getSlots() > 0) {
      return solved;
    }
    if (type.getElementCount() == numGroups) {
      // Prefer the packed carrier for plastic producers, including partial
      // packets with fewer than eight groups.  This keeps the broadcast on
      // the single-source vselr path; explicit or otherwise fixed slots=1
      // values retain their layout and use the cross-source fallback.
      return VMILayoutAttr::getGroupSlots(ctx, numGroups, /*slots=*/mlir::pto::kValue8);
    }
    if (auto load = value.getDefiningOp<VMIGroupSlotLoadOp>()) {
      return getPreferredGroupSlotLoadLayout(load);
    }
    return getPreferredGroupSlotsLayout(type, numGroups);
  }

  VMILayoutAttr
  getPreferredGroupBroadcastResultLayout(VMIGroupBroadcastOp op) const {
    auto type = cast<VMIVRegType>(op.getResult().getType());
    if (VMILayoutAttr existing = type.getLayoutAttr()) {
      return existing;
    }

    VMILayoutSupport supports;
    return supports.getPreferredGroupBroadcastResultLayout(
        type, op.getNumGroupsAttr().getInt());
  }

  VMILayoutAttr getPreferredGroupLoadResultLayout(VMIGroupLoadOp op) const {
    auto type = cast<VMIVRegType>(op.getResult().getType());
    if (VMILayoutAttr existing = type.getLayoutAttr()) {
      return existing;
    }

    int64_t numGroups = op.getNumGroupsAttr().getInt();
    if (numGroups <= 0 || type.getElementCount() % numGroups != 0) {
      return getContiguousLayout();
    }

    if (!type.getElementType().isF32()) {
      return getContiguousLayout();
    }

    int64_t groupSize = type.getElementCount() / numGroups;
    std::optional<int64_t> rowStride = getConstantIndexValue(op.getRowStride());
    if (rowStride && *rowStride == groupSize) {
      return getContiguousLayout();
    }
    if (!rowStride || *rowStride <= 0 ||
        *rowStride % mlir::pto::kValue8 != 0) {
      return getContiguousLayout();
    }

    if (groupSize == mlir::pto::kValue16) {
      return VMILayoutAttr::getBlockDeinterleaved(ctx, mlir::pto::kValue2);
    }
    if (groupSize == mlir::pto::kValue32) {
      return VMILayoutAttr::getBlockDeinterleaved(ctx, mlir::pto::kValue4);
    }

    return getContiguousLayout();
  }

  LogicalResult validateGroupLoadLayoutPlan(VMIGroupLoadOp op) const {
    auto type = cast<VMIVRegType>(op.getResult().getType());
    if (type.getLayoutAttr()) {
      return success();
    }

    int64_t numGroups = op.getNumGroupsAttr().getInt();
    if (numGroups <= 0 || type.getElementCount() % numGroups != 0) {
      return success();
    }

    int64_t groupSize = type.getElementCount() / numGroups;
    std::optional<int64_t> rowStride = getConstantIndexValue(op.getRowStride());
    if (rowStride && *rowStride == groupSize) {
      // Packed rows alias the contiguous layout.
      return success();
    }
    // A group covering one physical part or more goes through the full-chunk
    // plan, which accepts a dynamic row stride: leave it to that plan.
    FailureOr<int64_t> lanesPerPart =
        getDataLanesPerPart(type.getElementType());
    if (succeeded(lanesPerPart) && groupSize >= *lanesPerPart) {
      return success();
    }

    // Strided rows: the block forms (`vsldb`) address whole 32-byte blocks, so
    // one group must span a whole number of them and the row stride must
    // advance by a whole number of them.  Sub-32B groups and non-block row
    // strides used to be emulated by the sub-chunk gather, which is gone; the
    // contract is spelled out here because this pass is the first to see the
    // shape.
    unsigned elementBits =
        pto::getPTOStorageElemBitWidth(type.getElementType());
    if (elementBits == 0 || elementBits % 8 != 0) {
      return success();
    }
    int64_t elementBytes = elementBits / 8;
    int64_t blockElements = kVMIVCGBlockBytes / elementBytes;
    bool granular = blockElements > 0 && groupSize > 0 &&
                    groupSize % blockElements == 0 && rowStride &&
                    *rowStride > 0 && *rowStride % blockElements == 0;
    if (granular) {
      return success();
    }

    std::string details;
    llvm::raw_string_ostream stream(details);
    stream << "group = "
           << (groupSize > 0 ? groupSize * elementBytes : 0) << " bytes, "
           << "row_stride = "
           << (rowStride && *rowStride > 0 ? *rowStride * elementBytes : 0)
           << " bytes";
    return op.emitError()
           << kVMIDiagLayoutContractPrefix
           << "pto.vmi.group_load with a row stride requires one group to be "
              "32, 64, or 128 bytes and the row stride to be a whole number of "
              "32-byte blocks with a 32-byte aligned source; got "
           << details;
  }

  VMILayoutAttr getDataLayout(Value value) {
    unsigned id = addDataValue(value);
    if (id == ~0U) {
      return {};
    }
    unsigned root = find(id);
    if (dataNodes[root].naturalLayout) {
      return dataNodes[root].naturalLayout;
    }
    if (dataNodes[root].preferredLayout) {
      return dataNodes[root].preferredLayout;
    }
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
    if (failed(unite(lhs.get(), rhs.get(), op))) {
      return failure();
    }
    return unite(lhs.get(), result, op);
  }

  LogicalResult
  requestMaskUse(OpOperand &operand, VMILayoutAttr layout, Operation *op,
                 DataLayoutSeedPhase phase = DataLayoutSeedPhase::Other) {
    if (!isa<VMIMaskType>(operand.get().getType())) {
      return success();
    }
    if (!layout) {
      return op->emitError()
             << kVMIDiagLayoutContractPrefix
             << "cannot infer concrete mask use layout";
    }
    maskUseRequests.push_back(MaskUseRequest{&operand, layout, phase});
    return success();
  }

  LogicalResult collect() {
    module.walk([this](Operation *op) {
      for (Value result : op->getResults()) {
        addDataValue(result);
        addMaskValue(result);
      }
      for (Region &region : op->getRegions()) {
        for (Block &block : region) {
          for (BlockArgument arg : block.getArguments()) {
            addDataValue(arg);
            addMaskValue(arg);
          }
        }
      }
    });
    return success();
  }

  static std::optional<WalkResult> constraintResult(LogicalResult result) {
    return failed(result) ? WalkResult::interrupt() : WalkResult::advance();
  }

  std::optional<WalkResult> addBasicConstraint(Operation *op) {
    return llvm::TypeSwitch<Operation *, std::optional<WalkResult>>(op)
        .Case<VMIGroupIotaOp>([this, op](auto groupIota) {
          return constraintResult(setNaturalLayout(
              groupIota.getResult(), getContiguousLayout(), op));
        })
        .Case<VMIMaskAndOp, VMIMaskOrOp, VMIMaskXOrOp>([this, op](auto maskOp) {
          bool failedToUnite =
              failed(uniteMask(maskOp.getLhs(), maskOp.getRhs(), op)) ||
              failed(uniteMask(maskOp.getLhs(), maskOp.getResult(), op));
          return constraintResult(failure(failedToUnite));
        })
        .Case<VMIMaskNotOp, VMIEnsureMaskLayoutOp>([this, op](auto maskOp) {
          return constraintResult(
              uniteMask(maskOp.getSource(), maskOp.getResult(), op));
        })
        .Case<VMIVaddcOp, VMIVsubcOp, VMIVaddcsOp, VMIVsubcsOp>(
            [this, op](auto binaryOp) {
              return constraintResult(constrainElementwiseBinary(
                  binaryOp.getLhsMutable(), binaryOp.getRhsMutable(),
                  binaryOp.getResult(), op));
            })
        .Case<VMIAddSOp, VMIMulSOp, VMIMaxSOp, VMIMinSOp, VMIShlSOp,
              VMIShrSOp>([this, op](auto scalarOp) {
          return constraintResult(
              unite(scalarOp.getSrc(), scalarOp.getResult(), op));
        })
        .Case<VMIBitcastOp>([this, op](auto unaryOp) {
          return constraintResult(
              unite(unaryOp.getSource(), unaryOp.getResult(), op));
        })
        .Default([](Operation *) { return std::nullopt; });
  }

  std::optional<WalkResult> addCompositeConstraint(Operation *op) {
    return llvm::TypeSwitch<Operation *, std::optional<WalkResult>>(op)
        .Case<VMIVmullOp>([this, op](VMIVmullOp vmull) {
          bool failedToUnite =
              failed(constrainElementwiseBinary(vmull.getAMutable(),
                                                vmull.getBMutable(),
                                                vmull.getLow(), op)) ||
              failed(unite(vmull.getA(), vmull.getHigh(), op));
          return constraintResult(failure(failedToUnite));
        })
        .Case<VMICmpFOp, VMICmpIOp>([this, op](auto compareOp) {
          return constraintResult(
              unite(compareOp.getLhs(), compareOp.getRhs(), op));
        })
        .Case<VMISelectOp>([this, op](VMISelectOp select) {
          bool failedToUnite =
              failed(unite(select.getTrueValue(), select.getFalseValue(), op)) ||
              failed(unite(select.getTrueValue(), select.getResult(), op));
          return constraintResult(failure(failedToUnite));
        })
        .Default([](Operation *) { return std::nullopt; });
  }

  template <typename CastOp>
  WalkResult addConversionConstraint(CastOp castOp, Operation *op) {
    auto sourceType = cast<VMIVRegType>(castOp.getSource().getType());
    auto resultType = cast<VMIVRegType>(castOp.getResult().getType());
    FailureOr<VMICastLayoutFact> fact =
        VMILayoutSupport().getPreferredCastLayoutFact(sourceType, resultType);
    VMILayoutAttr resultLayout =
        succeeded(fact) ? fact->resultLayout : getContiguousLayout();
    return *constraintResult(setPreferredLayout(
        castOp.getResult(), resultLayout, op, DataLayoutSeedPhase::Cast));
  }

  template <typename CastOp>
  WalkResult addExtensionConstraint(CastOp castOp, Operation *op) {
    auto sourceType = cast<VMIVRegType>(castOp.getSource().getType());
    auto resultType = cast<VMIVRegType>(castOp.getResult().getType());

    // A four-lane byte/halfword -> ui32 widening is a compact group-slot
    // operation.  Seed the result with the group-slot carrier so propagation
    // selects the existing group-slot extension lowering (and avoids creating
    // an unsupported group-slot -> dense lane-stride ensure_layout).
    if constexpr (std::is_same_v<CastOp, VMIExtUIOp>) {
      auto sourceInteger = dyn_cast<IntegerType>(sourceType.getElementType());
      auto resultInteger = dyn_cast<IntegerType>(resultType.getElementType());
      unsigned sourceBits =
          pto::getPTOStorageElemBitWidth(sourceType.getElementType());
      unsigned resultBits =
          pto::getPTOStorageElemBitWidth(resultType.getElementType());
      bool compactUI32Widen =
          sourceType.getElementCount() == 4 && sourceInteger && resultInteger &&
          !sourceType.getLayoutAttr() && !resultType.getLayoutAttr() &&
          (resultInteger.isUnsigned() || resultInteger.isSignless()) &&
          (sourceBits == 8 || sourceBits == 16) &&
          resultBits == 32;
      if (compactUI32Widen) {
        VMILayoutAttr compactLayout =
            VMILayoutAttr::getGroupSlots(ctx, /*numGroups=*/4,
                                         /*slots=*/mlir::pto::kValue8);
        return *constraintResult(setPreferredLayout(
            castOp.getResult(), compactLayout, op,
            DataLayoutSeedPhase::CompactCast));
      }
    }

    FailureOr<VMICastLayoutFact> fact = getCastLayoutFactForSeed(
        op, castOp.getResult(), sourceType, resultType);
    if (failed(fact)) {
      return WalkResult::advance();
    }
    return *constraintResult(setPreferredLayout(
        castOp.getResult(), fact->resultLayout, op, getCastSeedPhase(*fact)));
  }

  template <typename CastOp>
  WalkResult addTruncationConstraint(CastOp castOp, Operation *op) {
    auto sourceType = cast<VMIVRegType>(castOp.getSource().getType());
    auto resultType = cast<VMIVRegType>(castOp.getResult().getType());
    FailureOr<VMICastLayoutFact> fact = getCastLayoutFactForSeed(
        op, castOp.getResult(), sourceType, resultType);
    VMILayoutAttr resultLayout =
        succeeded(fact) ? fact->resultLayout : getContiguousLayout();
    DataLayoutSeedPhase phase = succeeded(fact) ? getCastSeedPhase(*fact)
                                                : DataLayoutSeedPhase::Cast;
    return *constraintResult(
        setPreferredLayout(castOp.getResult(), resultLayout, op, phase));
  }

  WalkResult addVexpdifConstraint(VMIVexpdifOp vexpdif, Operation *op) {
    auto sourceType = cast<VMIVRegType>(vexpdif.getX().getType());
    auto resultType = cast<VMIVRegType>(vexpdif.getResult().getType());
    if (failed(unite(vexpdif.getX(), vexpdif.getMax(), op))) {
      return WalkResult::interrupt();
    }
    if (sourceType.getElementType().isF32()) {
      return *constraintResult(
          unite(vexpdif.getX(), vexpdif.getResult(), op));
    }
    FailureOr<VMICastLayoutFact> fact =
        VMILayoutSupport().getPreferredCastLayoutFact(sourceType, resultType);
    if (failed(fact)) {
      return WalkResult::advance();
    }
    return *constraintResult(setPreferredLayout(
        vexpdif.getResult(), fact->resultLayout, op, getCastSeedPhase(*fact)));
  }

  std::optional<WalkResult> addCastConstraint(Operation *op) {
    return llvm::TypeSwitch<Operation *, std::optional<WalkResult>>(op)
        .Case<VMIFPToSIOp, VMIFPToUIOp, VMISIToFPOp>(
            [this, op](auto castOp) {
              return addConversionConstraint(castOp, op);
            })
        .Case<VMIExtFOp, VMIExtSIOp, VMIExtUIOp>([this, op](auto castOp) {
          return addExtensionConstraint(castOp, op);
        })
        .Case<VMITruncFOp, VMITruncIOp>([this, op](auto castOp) {
          return addTruncationConstraint(castOp, op);
        })
        .Case<VMIVexpdifOp>([this, op](VMIVexpdifOp vexpdif) {
          return addVexpdifConstraint(vexpdif, op);
        })
        .Default([](Operation *) { return std::nullopt; });
  }

  template <typename ReduceOp>
  WalkResult addReductionConstraint(ReduceOp reduce, Operation *op) {
    VMILayoutAttr layout = getContiguousLayout();
    requestDataUse(reduce.getSourceMutable(), layout, /*late=*/false,
                   DataLayoutSeedPhase::Reduce);
    bool failedToConstrain =
        failed(requestMaskUse(reduce.getMaskMutable(), layout, op)) ||
        failed(setNaturalLayout(reduce.getResult(), layout, op,
                                DataLayoutSeedPhase::Reduce));
    return *constraintResult(failure(failedToConstrain));
  }

  template <typename ReduceOp>
  WalkResult addGroupReductionConstraint(ReduceOp reduce, Operation *op) {
    auto sourceType = cast<VMIVRegType>(reduce.getSource().getType());
    auto resultType = cast<VMIVRegType>(reduce.getResult().getType());
    int64_t numGroups = reduce.getNumGroupsAttr().getInt();
    FailureOr<VMIGroupReduceLayoutFact> fact =
        VMILayoutSupport().getPreferredGroupReduceLayoutFact(sourceType,
                                                             numGroups);
    VMILayoutAttr sourceLayout =
        succeeded(fact) ? fact->sourceLayout : getContiguousLayout();
    DataLayoutSeedPhase usePhase =
        succeeded(fact)
            ? getGroupReduceUseSeedPhase(sourceType, numGroups, *fact)
            : DataLayoutSeedPhase::Reduce;
    requestDataUse(reduce.getSourceMutable(), sourceLayout, /*late=*/false,
                   usePhase);
    VMILayoutAttr resultLayout =
        succeeded(fact)
            ? fact->resultLayout
            : getPreferredGroupSlotsLayout(resultType, numGroups);
    bool failedToConstrain =
        failed(requestMaskUse(reduce.getMaskMutable(), sourceLayout, op,
                              usePhase)) ||
        failed(setNaturalLayout(reduce.getResult(), resultLayout, op,
                                DataLayoutSeedPhase::Reduce));
    return *constraintResult(failure(failedToConstrain));
  }

  template <typename HistogramOp>
  WalkResult addHistogramConstraint(HistogramOp histogram, Operation *op) {
    VMILayoutAttr layout = getContiguousLayout();
    requestDataUse(histogram.getAccMutable(), layout, /*late=*/false,
                   DataLayoutSeedPhase::Reduce);
    requestDataUse(histogram.getSourceMutable(), layout, /*late=*/false,
                   DataLayoutSeedPhase::Reduce);
    bool failedToConstrain =
        failed(requestMaskUse(histogram.getMaskMutable(), layout, op,
                              DataLayoutSeedPhase::Reduce)) ||
        failed(setNaturalLayout(histogram.getResult(), layout, op,
                                DataLayoutSeedPhase::Reduce));
    return *constraintResult(failure(failedToConstrain));
  }

  std::optional<WalkResult> addReductionConstraints(Operation *op) {
    return llvm::TypeSwitch<Operation *, std::optional<WalkResult>>(op)
        .Case<VMIReduceAddIOp, VMIReduceAddFOp, VMIReduceMaxFOp,
              VMIReduceMinFOp, VMIReduceMaxIOp, VMIReduceMinIOp>(
            [this, op](auto reduce) {
              return addReductionConstraint(reduce, op);
            })
        .Case<VMIGroupReduceAddFOp, VMIGroupReduceMaxFOp,
              VMIGroupReduceMinFOp, VMIGroupReduceAddIOp,
              VMIGroupReduceMaxIOp, VMIGroupReduceMinIOp>(
            [this, op](auto reduce) {
              return addGroupReductionConstraint(reduce, op);
            })
        .Case<VMIVdhistOp, VMIVchistOp>([this, op](auto histogram) {
          return addHistogramConstraint(histogram, op);
        })
        .Default([](Operation *) { return std::nullopt; });
  }

  WalkResult addVselrConstraint(VMIVselrOp vselr, Operation *op) {
    FailureOr<VMIVselrLayoutFact> fact =
        VMILayoutSupport().getPreferredVselrLayoutFact(vselr);
    if (failed(fact)) {
      return WalkResult::advance();
    }
    requestDataUse(vselr.getSourceMutable(), fact->sourceLayout);
    requestDataUse(vselr.getIndexMutable(), fact->indexLayout);
    return *constraintResult(
        setNaturalLayout(vselr.getResult(), fact->resultLayout, op));
  }

  WalkResult addGroupBroadcastConstraint(VMIGroupBroadcastOp broadcast,
                                         Operation *op) {
    requestDataUse(
        broadcast.getSourceMutable(),
        getPreferredGroupBroadcastSourceLayout(
            broadcast.getSource(), broadcast.getNumGroupsAttr().getInt()),
        /*late=*/false, DataLayoutSeedPhase::GroupBroadcast);
    return *constraintResult(setPreferredLayout(
        broadcast.getResult(), getPreferredGroupBroadcastResultLayout(broadcast),
        op, DataLayoutSeedPhase::GroupBroadcast));
  }

  std::optional<WalkResult> addSpecialComputeConstraint(Operation *op) {
    return llvm::TypeSwitch<Operation *, std::optional<WalkResult>>(op)
        .Case<VMIVselrOp>([this, op](auto vselr) {
          return addVselrConstraint(vselr, op);
        })
        .Case<VMIActivePrefixIndexOp>([this, op](auto activePrefix) {
          return constraintResult(setNaturalLayout(
              activePrefix.getResult(), getContiguousLayout(), op));
        })
        .Case<VMICompressOp>([this, op](auto compress) {
          requestDataUse(compress.getSourceMutable(), getContiguousLayout());
          return constraintResult(setNaturalLayout(
              compress.getResult(), getContiguousLayout(), op));
        })
        .Case<VMIGroupBroadcastOp>([this, op](auto broadcast) {
          return addGroupBroadcastConstraint(broadcast, op);
        })
        .Default([](Operation *) { return std::nullopt; });
  }

  template <typename InterleaveOp, typename Fact>
  WalkResult applyInterleaveFact(InterleaveOp interleave, const Fact &fact,
                                 Operation *op) {
    requestDataUse(interleave.getLhsMutable(), fact.lhsLayout);
    requestDataUse(interleave.getRhsMutable(), fact.rhsLayout);
    return *constraintResult(
        requestMaskUse(interleave.getMaskMutable(), fact.maskLayout, op));
  }

  std::optional<WalkResult> addInterleaveConstraint(Operation *op) {
    return llvm::TypeSwitch<Operation *, std::optional<WalkResult>>(op)
        .Case<VMIVintlvOp>([this, op](VMIVintlvOp interleave) {
          auto type = cast<VMIVRegType>(interleave.getLow().getType());
          FailureOr<VMIInterleaveLayoutFact> fact =
              VMILayoutSupport().getPreferredVintlvLayoutFact(type);
          return failed(fact) ? WalkResult::advance()
                              : applyInterleaveFact(interleave, *fact, op);
        })
        .Case<VMIVdintlvOp>([this, op](VMIVdintlvOp interleave) {
          auto type = cast<VMIVRegType>(interleave.getLow().getType());
          FailureOr<VMIInterleaveLayoutFact> fact =
              VMILayoutSupport().getPreferredVdintlvLayoutFact(type);
          return failed(fact) ? WalkResult::advance()
                              : applyInterleaveFact(interleave, *fact, op);
        })
        .Default([](Operation *) { return std::nullopt; });
  }

  WalkResult addDeinterleaveLoadConstraint(VMIDeinterleaveLoadOp load,
                                           Operation *op) {
    FailureOr<VMIDeinterleaveLoadLayoutFact> fact =
        VMILayoutSupport().getPreferredDeinterleaveLoadLayoutFact(
            cast<VMIVRegType>(load.getLow().getType()));
    if (failed(fact)) {
      return WalkResult::advance();
    }
    bool failedToSet =
        failed(setNaturalLayout(load.getLow(), fact->lowLayout, op)) ||
        failed(setNaturalLayout(load.getHigh(), fact->highLayout, op));
    return *constraintResult(failure(failedToSet));
  }

  WalkResult addGatherConstraint(VMIGatherOp gather, Operation *op) {
    VMILayoutAttr layout = getContiguousLayout();
    requestDataUse(gather.getIndicesMutable(), layout);
    requestDataUse(gather.getPassthruMutable(), layout);
    bool failedToConstrain =
        failed(requestMaskUse(gather.getMaskMutable(), layout, op)) ||
        failed(setNaturalLayout(gather.getResult(), layout, op));
    return *constraintResult(failure(failedToConstrain));
  }

  std::optional<WalkResult> addSimpleLoadConstraint(Operation *op) {
    return llvm::TypeSwitch<Operation *, std::optional<WalkResult>>(op)
        .Case<VMIDeinterleaveLoadOp>([this, op](auto load) {
          return addDeinterleaveLoadConstraint(load, op);
        })
        .Case<VMILoadOp>([this, op](auto load) {
          auto type = cast<VMIVRegType>(load.getResult().getType());
          FailureOr<int64_t> lanesPerPart =
              getDataLanesPerPart(type.getElementType());
          // Stock leaves plain VMILoadOp results unconstrained (the layout is
          // driven by consumer use-requests).  Seed a natural layout ONLY
          // when the packed/small-load lane-stride heuristic below actually
          // applies: seeding plain contiguous unconditionally would block
          // deinterleaved use-requests from propagating back to the load
          // (e.g. channel_split @128 f16 hitting the ensure-layout gap,
          // vmi_layout_assignment_store_prefer_lane_stride).
          // The heuristic is safe for packed float carriers (f4x2 /
          // hi-float8x2 / bf16x2, whose storage byte holds a pair of values)
          // and for >=16-bit dense elements (a lane-stride unpack keeps every
          // real element reachable through the EVEN part; pinned by the
          // ComputeMropeF16 capability guard).  Dense 8-bit elements are the
          // proven mine: an unpack-style distribution whose gap lanes read as
          // zeros (the e4m3 odd-column regression shape; hidden=128 configs
          // would hit it), so those keep no seed at all.
          Type loadElemTy = type.getElementType();
          bool isPackedCarrier = pto::isPTOFloat4PackedType(loadElemTy) ||
                                 pto::isPTOHiFloat8x2Type(loadElemTy) ||
                                 pto::isPTOBF16x2Type(loadElemTy);
          bool laneStrideSeedAllowed =
              isPackedCarrier ||
              pto::getPTOStorageElemBitWidth(loadElemTy) != 8;
          if (!laneStrideSeedAllowed || failed(lanesPerPart) ||
              type.getElementCount() >= *lanesPerPart ||
              *lanesPerPart % type.getElementCount() != 0) {
            return std::optional<WalkResult>(std::nullopt);
          }
          int64_t laneStride = *lanesPerPart / type.getElementCount();
          return constraintResult(setNaturalLayout(
              load.getResult(), VMILayoutAttr::getContiguous(ctx, laneStride),
              op));
        })
        .Case<VMIMaskedLoadOp, VMIExpandLoadOp>([this, op](auto load) {
          requestDataUse(load.getPassthruMutable(), getContiguousLayout());
          return constraintResult(setNaturalLayout(
              load.getResult(), getContiguousLayout(), op));
        })
        .Case<VMIGatherOp>([this, op](auto gather) {
          return addGatherConstraint(gather, op);
        })
        .Case<VMIStrideLoadOp>([this, op](auto load) {
          VMILayoutAttr layout = getContiguousLayout();
          bool failedToConstrain =
              failed(setNaturalLayout(load.getResult(), layout, op)) ||
              failed(requestMaskUse(load.getMaskMutable(), layout, op));
          return constraintResult(failure(failedToConstrain));
        })
        .Default([](Operation *) { return std::nullopt; });
  }

  WalkResult addGroupLoadConstraint(VMIGroupLoadOp load, Operation *op) {
    if (failed(validateGroupLoadLayoutPlan(load))) {
      return WalkResult::interrupt();
    }
    VMILayoutAttr layout = getPreferredGroupLoadResultLayout(load);
    bool isDenseLaneStride = layout.isContiguous() &&
                             layout.getLaneStride() == 1;
    LogicalResult result =
        isDenseLaneStride
            ? setPreferredLayout(load.getResult(), layout, op)
            : setNaturalLayout(load.getResult(), layout, op,
                               DataLayoutSeedPhase::GroupLoad);
    return *constraintResult(result);
  }

  WalkResult addGroupBroadcastLoadConstraint(VMIGroupBroadcastLoadOp load,
                                             Operation *op) {
    FailureOr<VMIGroupBroadcastLoadDirectFact> directFact =
        VMILayoutSupport().getGroupBroadcastLoadDirectFact(load);
    bool scalarBroadcast =
        succeeded(directFact) &&
        directFact->kind == VMIGroupBroadcastLoadDirectKind::BRC &&
        load.getNumGroupsAttr().getInt() == 1;
    if (scalarBroadcast) {
      return WalkResult::advance();
    }
    DataLayoutSeedPhase phase =
        succeeded(directFact) ? DataLayoutSeedPhase::GroupBroadcastLoad
                              : DataLayoutSeedPhase::Other;
    // A direct E2B layout is a lowering preference, not a semantic
    // natural-layout contract.  Keep it as a preferred seed so a
    // downstream cast chain (for example C -> LS4 -> C) can select its
    // compatible layout and materialize an ensure_layout for E2B when
    // needed, instead of anchoring the whole equivalent value class.
    return *constraintResult(setPreferredLayout(
        load.getResult(), getPreferredGroupBroadcastLoadLayout(load), op,
        phase));
  }

  std::optional<WalkResult> addGroupLoadConstraints(Operation *op) {
    return llvm::TypeSwitch<Operation *, std::optional<WalkResult>>(op)
        .Case<VMIGroupLoadOp>([this, op](auto load) {
          return addGroupLoadConstraint(load, op);
        })
        .Case<VMIGroupSlotLoadOp>([this, op](auto load) {
          return constraintResult(setPreferredLayout(
              load.getResult(), getPreferredGroupSlotLoadLayout(load), op,
              DataLayoutSeedPhase::GroupSlotLoad));
        })
        .Case<VMIGroupBroadcastLoadOp>([this, op](auto load) {
          return addGroupBroadcastLoadConstraint(load, op);
        })
        .Default([](Operation *) { return std::nullopt; });
  }

  WalkResult addDenseStoreConstraint(VMIStoreOp store) {
    auto valueType = cast<VMIVRegType>(store.getValue().getType());
    if (!hasDataLayoutSeed(store.getValue())) {
      VMILayoutAttr layout = getPreferredDenseStoreLayout(valueType);
      if (layout) {
        requestDataUse(store.getValueMutable(), layout, /*late=*/false,
                       DataLayoutSeedPhase::Store);
      }
    }
    return WalkResult::advance();
  }

  WalkResult addGroupStoreConstraint(VMIGroupStoreOp store) {
    auto valueType = cast<VMIVRegType>(store.getValue().getType());
    VMILayoutSupport supports;
    VMILayoutAttr highPriorityLayout;
    FailureOr<VMIGroupStoreLayoutFact> highPriorityFact =
        supports.getHighPriorityGroupStoreLayoutFact(store, valueType);
    if (succeeded(highPriorityFact)) {
      highPriorityLayout = highPriorityFact->valueLayout;
      requestDataUse(store.getValueMutable(), highPriorityLayout,
                     /*late=*/false, DataLayoutSeedPhase::GroupStore);
    }
    FailureOr<VMIGroupStoreLayoutFact> preferredFact =
        supports.getPreferredGroupStoreLayoutFact(store, valueType);
    bool hasDistinctFallback =
        succeeded(preferredFact) &&
        preferredFact->valueLayout != highPriorityLayout;
    if (hasDistinctFallback) {
      requestDataUse(store.getValueMutable(), preferredFact->valueLayout,
                     /*late=*/false,
                     DataLayoutSeedPhase::GroupStoreFallback);
    }
    return WalkResult::advance();
  }

  WalkResult addMaskedStoreConstraint(VMIMaskedStoreOp store, Operation *op) {
    if (hasDataLayoutSeed(store.getValue())) {
      return WalkResult::advance();
    }
    auto valueType = cast<VMIVRegType>(store.getValue().getType());
    auto maskType = cast<VMIMaskType>(store.getMask().getType());
    FailureOr<VMIMaskedStoreLayoutFact> fact =
        getPreferredDenseMaskedStoreLayout(valueType, maskType);
    if (failed(fact)) {
      return WalkResult::advance();
    }
    requestDataUse(store.getValueMutable(), fact->valueLayout,
                   /*late=*/false, DataLayoutSeedPhase::Store);
    return *constraintResult(requestMaskUse(
        store.getMaskMutable(), fact->maskLayout, op,
        DataLayoutSeedPhase::Store));
  }

  std::optional<WalkResult> addStoreConstraint(Operation *op) {
    return llvm::TypeSwitch<Operation *, std::optional<WalkResult>>(op)
        .Case<VMIStoreOp>([this](auto store) {
          return addDenseStoreConstraint(store);
        })
        .Case<VMIInterleaveStoreOp>([this](auto store) {
          requestDataUse(store.getLowMutable(), getContiguousLayout());
          requestDataUse(store.getHighMutable(), getContiguousLayout());
          return WalkResult::advance();
        })
        .Case<VMIGroupStoreOp>([this](auto store) {
          return addGroupStoreConstraint(store);
        })
        .Case<VMIMaskedStoreOp>([this, op](auto store) {
          return addMaskedStoreConstraint(store, op);
        })
        .Case<VMIStrideStoreOp, VMICompressStoreOp>([this, op](auto store) {
          VMILayoutAttr layout = getContiguousLayout();
          requestDataUse(store.getValueMutable(), layout);
          return constraintResult(
              requestMaskUse(store.getMaskMutable(), layout, op));
        })
        .Case<VMIScatterOp>([this, op](auto scatter) {
          VMILayoutAttr layout = getContiguousLayout();
          requestDataUse(scatter.getValueMutable(), layout);
          requestDataUse(scatter.getIndicesMutable(), layout);
          return constraintResult(
              requestMaskUse(scatter.getMaskMutable(), layout, op));
        })
        .Default([](Operation *) { return std::nullopt; });
  }

  WalkResult addChannelSplitConstraint(VMIChannelSplitOp split,
                                       Operation *op) {
    int64_t channels = split.getNumResults();
    bool unsupported = channels != mlir::pto::kValue2 &&
                       channels != mlir::pto::kValue4;
    if (unsupported) {
      split.emitError() << kVMIDiagUnsupportedPrefix
                        << "pto.vmi.channel_split supports only 2 or 4 channels";
      return WalkResult::interrupt();
    }
    requestDataUse(split.getSourceMutable(),
                   VMILayoutAttr::getDeinterleaved(ctx, channels));
    for (Value result : split.getResults()) {
      if (failed(setNaturalLayout(result, getContiguousLayout(), op))) {
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  }

  WalkResult addChannelMergeConstraint(VMIChannelMergeOp merge,
                                       Operation *op) {
    int64_t channels = merge.getInputs().size();
    bool unsupported = channels != mlir::pto::kValue2 &&
                       channels != mlir::pto::kValue4;
    if (unsupported) {
      merge.emitError() << kVMIDiagUnsupportedPrefix
                        << "pto.vmi.channel_merge supports only 2 or 4 channels";
      return WalkResult::interrupt();
    }
    for (OpOperand &input : merge.getInputsMutable()) {
      requestDataUse(input, getContiguousLayout());
    }
    return *constraintResult(setNaturalLayout(
        merge.getResult(), VMILayoutAttr::getDeinterleaved(ctx, channels), op));
  }

  WalkResult addShuffleConstraint(VMIShuffleOp shuffle, Operation *op) {
    auto sourceType = cast<VMIVRegType>(shuffle.getSource().getType());
    auto resultType = cast<VMIVRegType>(shuffle.getResult().getType());
    bool hasExplicitLayout = sourceType.hasLayout() || resultType.hasLayout();
    if (hasExplicitLayout) {
      return WalkResult::advance();
    }
    requestDataUse(shuffle.getSourceMutable(), getContiguousLayout());
    if (isLane0SplatShuffle(shuffle)) {
      return WalkResult::advance();
    }
    return *constraintResult(
        setNaturalLayout(shuffle.getResult(), getContiguousLayout(), op));
  }

  std::optional<WalkResult> addChannelConstraint(Operation *op) {
    return llvm::TypeSwitch<Operation *, std::optional<WalkResult>>(op)
        .Case<VMIChannelSplitOp>([this, op](auto split) {
          return addChannelSplitConstraint(split, op);
        })
        .Case<VMIChannelMergeOp>([this, op](auto merge) {
          return addChannelMergeConstraint(merge, op);
        })
        .Case<VMIShuffleOp>([this, op](auto shuffle) {
          return addShuffleConstraint(shuffle, op);
        })
        .Default([](Operation *) { return std::nullopt; });
  }

  LogicalResult addSwitchConstraints(cf::SwitchOp switchOp) {
    if (failed(addBranchConstraints(switchOp.getDefaultDestination(),
                                    switchOp.getDefaultOperands(), switchOp))) {
      return failure();
    }
    for (auto [destination, operands] :
         llvm::zip(switchOp.getCaseDestinations(), switchOp.getCaseOperands())) {
      if (failed(addBranchConstraints(destination, operands, switchOp))) {
        return failure();
      }
    }
    return success();
  }

  std::optional<WalkResult> addControlFlowConstraint(Operation *op) {
    return llvm::TypeSwitch<Operation *, std::optional<WalkResult>>(op)
        .Case<scf::IfOp>([this](auto controlOp) {
          return constraintResult(addIfConstraints(controlOp));
        })
        .Case<scf::ExecuteRegionOp>([this](auto controlOp) {
          return constraintResult(addExecuteRegionConstraints(controlOp));
        })
        .Case<scf::IndexSwitchOp>([this](auto controlOp) {
          return constraintResult(addIndexSwitchConstraints(controlOp));
        })
        .Case<scf::WhileOp>([this](auto controlOp) {
          return constraintResult(addWhileConstraints(controlOp));
        })
        .Case<scf::ForOp>([this](auto controlOp) {
          return constraintResult(addForConstraints(controlOp));
        })
        .Case<cf::BranchOp>([this, op](cf::BranchOp branch) {
          return constraintResult(addBranchConstraints(
              branch.getDest(), branch.getDestOperands(), op));
        })
        .Case<cf::CondBranchOp>([this, op](cf::CondBranchOp branch) {
          bool failedToConstrain =
              failed(addBranchConstraints(branch.getTrueDest(),
                                          branch.getTrueDestOperands(), op)) ||
              failed(addBranchConstraints(branch.getFalseDest(),
                                          branch.getFalseDestOperands(), op));
          return constraintResult(failure(failedToConstrain));
        })
        .Case<cf::SwitchOp>([this](auto switchOp) {
          return constraintResult(addSwitchConstraints(switchOp));
        })
        .Case<func::ReturnOp>([this](auto returnOp) {
          return constraintResult(addReturnConstraints(returnOp));
        })
        .Case<func::CallOp>([this](auto callOp) {
          return constraintResult(addCallConstraints(callOp));
        })
        .Default([](Operation *) { return std::nullopt; });
  }

  std::optional<WalkResult> addComputeConstraint(Operation *op) {
    if (auto result = addBasicConstraint(op)) {
      return result;
    }
    if (auto result = addCompositeConstraint(op)) {
      return result;
    }
    if (auto result = addCastConstraint(op)) {
      return result;
    }
    if (auto result = addReductionConstraints(op)) {
      return result;
    }
    if (auto result = addSpecialComputeConstraint(op)) {
      return result;
    }
    if (auto result = addInterleaveConstraint(op)) {
      return result;
    }
    return std::nullopt;
  }

  std::optional<WalkResult> addMemoryAndControlConstraint(Operation *op) {
    if (auto result = addSimpleLoadConstraint(op)) {
      return result;
    }
    if (auto result = addGroupLoadConstraints(op)) {
      return result;
    }
    if (auto result = addStoreConstraint(op)) {
      return result;
    }
    if (auto result = addChannelConstraint(op)) {
      return result;
    }
    if (auto result = addControlFlowConstraint(op)) {
      return result;
    }
    return std::nullopt;
  }

  WalkResult validateUnconstrainedOperation(Operation *op) const {
    bool isIndirectCall =
        op->getName().getStringRef() == "func.call_indirect";
    if (isIndirectCall) {
      if (hasVMIValueTypes(op)) {
        op->emitError()
            << kVMIDiagLayoutContractPrefix
            << "VMI typed call requires a direct internal callee with a body";
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }
    if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
      bool invalidDeclaration = funcOp.empty() && hasVMIFunctionType(funcOp);
      if (invalidDeclaration) {
        funcOp.emitError()
            << kVMIDiagLayoutContractPrefix
            << "VMI typed function declaration requires an explicit "
               "external ABI materialization plan";
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  }

  WalkResult addConstraint(Operation *op) {
    if (auto result = addComputeConstraint(op)) {
      return *result;
    }
    if (auto result = addMemoryAndControlConstraint(op)) {
      return *result;
    }
    return validateUnconstrainedOperation(op);
  }

  LogicalResult addConstraints() {
    WalkResult result =
        module.walk([this](Operation *op) { return addConstraint(op); });
    return failure(result.wasInterrupted());
  }

  LogicalResult uniteEquivalentValues(Value lhs, Value rhs, Operation *op) {
    if (failed(uniteDataEquivalent(lhs, rhs, op))) {
      return failure();
    }
    return uniteMask(lhs, rhs, op);
  }

  LogicalResult addIfConstraints(scf::IfOp ifOp) {
    for (OpResult result : ifOp->getResults()) {
      unsigned resultNo = result.getResultNumber();
      for (Region *region : {&ifOp.getThenRegion(), &ifOp.getElseRegion()}) {
        if (region->empty()) {
          continue;
        }
        auto yieldOp = dyn_cast<scf::YieldOp>(region->front().getTerminator());
        if (!yieldOp || resultNo >= yieldOp.getNumOperands()) {
          continue;
        }
        if (failed(uniteEquivalentValues(result, yieldOp.getOperand(resultNo),
                                         ifOp))) {
          return failure();
        }
      }
    }
    return success();
  }

  LogicalResult addYieldConstraints(ResultRange results, scf::YieldOp yieldOp,
                                    Operation *op) {
    for (auto [index, result] : llvm::enumerate(results)) {
      if (index >= yieldOp.getNumOperands()) {
        break;
      }
      if (failed(uniteEquivalentValues(result, yieldOp.getOperand(index), op))) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult addExecuteRegionConstraints(scf::ExecuteRegionOp executeOp) {
    WalkResult result = executeOp.getRegion().walk(
        [this, executeOp](scf::YieldOp yieldOp) mutable {
          const bool belongsToExecuteRegion =
              yieldOp->getParentOp() == executeOp.getOperation();
          if (!belongsToExecuteRegion) {
            return WalkResult::advance();
          }
          if (failed(addYieldConstraints(executeOp->getResults(), yieldOp,
                                         executeOp))) {
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
    return failure(result.wasInterrupted());
  }

  LogicalResult addIndexSwitchConstraints(scf::IndexSwitchOp indexSwitchOp) {
    auto addBlockTerminator =
        [this, indexSwitchOp](Block &block) mutable -> LogicalResult {
      auto yieldOp = dyn_cast<scf::YieldOp>(block.getTerminator());
      if (!yieldOp) {
        return success();
      }
      return addYieldConstraints(indexSwitchOp->getResults(), yieldOp,
                                 indexSwitchOp);
    };
    if (failed(addBlockTerminator(indexSwitchOp.getDefaultBlock()))) {
      return failure();
    }
    for (unsigned idx = 0, e = indexSwitchOp.getNumCases(); idx < e; ++idx) {
      if (failed(addBlockTerminator(indexSwitchOp.getCaseBlock(idx)))) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult addWhileConstraints(scf::WhileOp whileOp) {
    return VMIControlFlowSupport::addWhileConstraints(
        whileOp, [this](Value lhs, Value rhs, Operation *op) {
          return uniteEquivalentValues(lhs, rhs, op);
        });
  }

  LogicalResult addForConstraints(scf::ForOp forOp) {
    return VMIControlFlowSupport::addForConstraints(
        forOp, [this](Value lhs, Value rhs, Operation *op) {
          return uniteEquivalentValues(lhs, rhs, op);
        });
  }

  LogicalResult addBranchConstraints(Block *dest, OperandRange operands,
                                     Operation *op) {
    if (!dest) {
      return success();
    }
    for (auto [index, operand] : llvm::enumerate(operands)) {
      if (index >= dest->getNumArguments()) {
        break;
      }
      if (failed(uniteEquivalentValues(operand, dest->getArgument(index), op))) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult addReturnConstraints(func::ReturnOp returnOp) {
    auto func = returnOp->getParentOfType<func::FuncOp>();
    if (!func) {
      return success();
    }

    auto it = firstReturnOperandsByFunc.find(func);
    if (it == firstReturnOperandsByFunc.end()) {
      SmallVector<Value> operands(returnOp.getOperands());
      firstReturnOperandsByFunc.try_emplace(func, std::move(operands));
      return success();
    }

    ArrayRef<Value> firstOperands = it->second;
    for (auto [index, operand] : llvm::enumerate(returnOp.getOperands())) {
      if (index >= firstOperands.size()) {
        break;
      }
      if (failed(
              uniteEquivalentValues(firstOperands[index], operand, returnOp))) {
        return failure();
      }
    }
    return success();
  }

  bool hasVMIValueTypes(Operation *op) const {
    return llvm::any_of(op->getOperandTypes(), containsVMIType) ||
           llvm::any_of(op->getResultTypes(), containsVMIType);
  }

  bool hasVMIFunctionType(func::FuncOp func) const {
    FunctionType type = func.getFunctionType();
    return llvm::any_of(type.getInputs(), containsVMIType) ||
           llvm::any_of(type.getResults(), containsVMIType);
  }

  LogicalResult addCallConstraints(func::CallOp callOp) {
    if (!hasVMIValueTypes(callOp)) {
      return success();
    }

    auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        callOp, callOp.getCalleeAttr());
    if (!callee || callee.empty()) {
      return callOp.emitError()
             << kVMIDiagLayoutContractPrefix
             << "VMI typed call requires a direct internal callee with a body";
    }

    for (auto [operand, argument] :
         llvm::zip(callOp.getOperands(), callee.getArguments())) {
      if (failed(uniteEquivalentValues(operand, argument, callOp))) {
        return failure();
      }
    }

    SmallVector<func::ReturnOp> returns;
    callee.walk(
        [&returns](func::ReturnOp returnOp) { returns.push_back(returnOp); });
    for (func::ReturnOp returnOp : returns) {
      for (auto [index, result] : llvm::enumerate(callOp.getResults())) {
        if (index >= returnOp.getNumOperands()) {
          break;
        }
        if (failed(uniteEquivalentValues(result, returnOp.getOperand(index),
                                         callOp))) {
          return failure();
        }
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
                                          Location loc,
                                          OpBuilder &builder) const {
    if (value.getType() == targetType) {
      return value;
    }

    if (auto sourceType = dyn_cast<VMIVRegType>(value.getType())) {
      auto targetVRegType = dyn_cast<VMIVRegType>(targetType);
      if (!targetVRegType ||
          sourceType.getElementCount() != targetVRegType.getElementCount() ||
          sourceType.getElementType() != targetVRegType.getElementType()) {
        return failure();
      }
      return builder.create<VMIEnsureLayoutOp>(loc, targetVRegType, value)
          .getResult();
    }

    if (auto sourceType = dyn_cast<VMIMaskType>(value.getType())) {
      auto targetMaskType = dyn_cast<VMIMaskType>(targetType);
      if (!targetMaskType ||
          sourceType.getElementCount() != targetMaskType.getElementCount() ||
          sourceType.getGranularity() != targetMaskType.getGranularity()) {
        return failure();
      }
      return builder
          .create<VMIEnsureMaskLayoutOp>(loc, targetMaskType, value)
          .getResult();
    }

    return failure();
  }

  LogicalResult materializeCallOperands(IRRewriter &rewriter) {
    WalkResult result = module.walk([this, &rewriter](func::CallOp call) {
      auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          call, call.getCalleeAttr());
      if (!callee || callee.empty()) {
        return WalkResult::advance();
      }

      rewriter.setInsertionPoint(call);
      for (auto [index, operand] : llvm::enumerate(call.getOperands())) {
        if (index >= callee.getNumArguments()) {
          break;
        }
        Type targetType = callee.getArgument(index).getType();
        if (!isa<VMIVRegType, VMIMaskType>(targetType)) {
          continue;
        }
        FailureOr<Value> materialized =
            materializeLayoutValue(operand, targetType, call.getLoc(),
                                   rewriter);
        if (failed(materialized)) {
          return WalkResult::interrupt();
        }
        call->setOperand(index, *materialized);
      }
      return WalkResult::advance();
    });
    return failure(result.wasInterrupted());
  }

  WalkResult materializeReturnOperands(func::ReturnOp ret,
                                       ArrayRef<Type> resultTypes,
                                       IRRewriter &rewriter) const {
    rewriter.setInsertionPoint(ret);
    for (auto [index, operand] : llvm::enumerate(ret.getOperands())) {
      if (index >= resultTypes.size()) {
        break;
      }
      Type targetType = resultTypes[index];
      if (!targetType || !isa<VMIVRegType, VMIMaskType>(targetType)) {
        continue;
      }
      FailureOr<Value> materialized =
          materializeLayoutValue(operand, targetType, ret.getLoc(), rewriter);
      if (failed(materialized)) {
        return WalkResult::interrupt();
      }
      ret->setOperand(index, *materialized);
    }
    return WalkResult::advance();
  }

  LogicalResult materializeFunctionReturns(IRRewriter &rewriter) {
    WalkResult result = module.walk([this, &rewriter](func::FuncOp func) {
      SmallVector<Type> resultTypes =
          getConsistentCallResultTypes(module, func);
      if (resultTypes.empty()) {
        return WalkResult::advance();
      }
      WalkResult nested = func.walk([this, &resultTypes, &rewriter](
                                        func::ReturnOp ret) {
        return materializeReturnOperands(ret, resultTypes, rewriter);
      });
      return nested.wasInterrupted() ? WalkResult::interrupt()
                                     : WalkResult::advance();
    });
    return failure(result.wasInterrupted());
  }

  LogicalResult materializeCallBoundaries() {
    IRRewriter rewriter(ctx);
    if (failed(materializeCallOperands(rewriter))) {
      return failure();
    }
    return materializeFunctionReturns(rewriter);
  }

  LogicalResult insertDataUseMaterializations() {
    OpBuilder builder(ctx);
    for (DataUseRequest request : dataUseRequests) {
      Value value = request.operand->get();
      auto sourceType = dyn_cast<VMIVRegType>(value.getType());
      if (!sourceType) {
        continue;
      }
      VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
      if (!sourceLayout) {
        return request.operand->getOwner()->emitError()
               << kVMIDiagLayoutContractPrefix
               << "data use materialization requires layout-assigned source "
                  "type";
      }
      if (sourceLayout == request.layout) {
        continue;
      }

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

  bool hasRequestedLayout(VMILayoutPropagator &propagator, Value value) const {
    return static_cast<bool>(propagator.getRequestedLayout(value));
  }

  bool hasLayoutAssignment(VMILayoutPropagator &propagator, Value value) const {
    return propagator.lookup(value) != nullptr;
  }

  LogicalResult requestDataLayoutSeeds(VMILayoutPropagator &propagator,
                                       DataLayoutSeedPhase phase,
                                       bool skipAlreadyRequested) {
    SmallVector<Value, mlir::pto::kValue16> protectedValues;
    if (skipAlreadyRequested) {
      for (DataLayoutSeed seed : dataLayoutSeeds) {
        if (seed.phase != phase) {
          continue;
        }
        if (hasRequestedLayout(propagator, seed.value) &&
            !llvm::is_contained(protectedValues, seed.value)) {
          protectedValues.push_back(seed.value);
        }
      }
    }

    for (DataLayoutSeed seed : dataLayoutSeeds) {
      if (seed.phase != phase) {
        continue;
      }
      if (llvm::is_contained(protectedValues, seed.value)) {
        continue;
      }
      if (failed(propagator.request(seed.value, seed.layout))) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult requestDataUseSeeds(VMILayoutPropagator &propagator,
                                    DataLayoutSeedPhase phase, bool late) {
    for (DataUseRequest request : dataUseRequests) {
      if (request.phase == phase && request.late == late) {
        if (hasLayoutAssignment(propagator, request.operand->get())) {
          VMILayoutAttr assigned =
              propagator.getRequestedOrCurrentLayout(request.operand->get());
          if (propagator.canUseOperandLayout(*request.operand, assigned)) {
            continue;
          }
        }
        if (failed(propagator.request(*request.operand, request.layout))) {
          return failure();
        }
      }
    }
    return success();
  }

  LogicalResult requestMaskUseSeeds(VMILayoutPropagator &propagator,
                                    DataLayoutSeedPhase phase) {
    for (MaskUseRequest request : maskUseRequests) {
      if (request.phase == phase) {
        if (hasLayoutAssignment(propagator, request.operand->get())) {
          VMILayoutAttr assigned =
              propagator.getRequestedOrCurrentLayout(request.operand->get());
          if (propagator.canUseOperandLayout(*request.operand, assigned)) {
            continue;
          }
        }
        if (failed(propagator.request(*request.operand, request.layout))) {
          return failure();
        }
      }
    }
    return success();
  }

  LogicalResult runSeedPhase(VMILayoutPropagator &propagator,
                             DataLayoutSeedPhase phase) {
    if (failed(requestDataLayoutSeeds(propagator, phase,
                                      /*skipAlreadyRequested=*/true))) {
      return failure();
    }
    if (failed(requestDataUseSeeds(propagator, phase, /*late=*/false))) {
      return failure();
    }
    if (failed(requestMaskUseSeeds(propagator, phase))) {
      return failure();
    }
    return propagator.run();
  }

  void addEquivalentLayoutValues(VMILayoutPropagator &propagator) {
    for (DataNode &node : dataNodes) {
      DataNode &root = dataNodes[find(dataIds.lookup(node.value))];
      propagator.addEquivalentValues(root.value, node.value);
    }
    for (MaskNode &node : maskNodes) {
      MaskNode &root = maskNodes[findMask(maskIds.lookup(node.value))];
      propagator.addEquivalentValues(root.value, node.value);
    }
  }

  LogicalResult requestExplicitLayouts(VMILayoutPropagator &propagator) {
    if (failed(requestDataLayoutSeeds(propagator,
                                      DataLayoutSeedPhase::Explicit,
                                      /*skipAlreadyRequested=*/false))) {
      return failure();
    }
    for (MaskNode &node : maskNodes) {
      MaskNode &root = maskNodes[findMask(maskIds.lookup(node.value))];
      if (root.requestedLayout &&
          failed(propagator.request(node.value, root.requestedLayout))) {
        return failure();
      }
    }
    return propagator.run();
  }

  LogicalResult runLayoutSeedPhases(VMILayoutPropagator &propagator) {
    for (int64_t phase = static_cast<int64_t>(DataLayoutSeedPhase::SeedStart);
         phase < static_cast<int64_t>(DataLayoutSeedPhase::SeedEnd); ++phase) {
      if (failed(runSeedPhase(propagator,
                              static_cast<DataLayoutSeedPhase>(phase)))) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult requestLateLayouts(VMILayoutPropagator &propagator) {
    for (DataUseRequest request : dataUseRequests) {
      if (request.late &&
          failed(propagator.request(*request.operand, request.layout))) {
        return failure();
      }
    }
    return propagator.run();
  }

  LogicalResult requestFallbackLayouts(VMILayoutPropagator &propagator) {
    for (DataNode &node : dataNodes) {
      if (!propagator.getRequestedLayout(node.value) &&
          failed(propagator.request(node.value, getContiguousLayout()))) {
        return failure();
      }
    }
    for (MaskNode &node : maskNodes) {
      if (!propagator.getRequestedLayout(node.value) &&
          failed(propagator.request(node.value, getContiguousLayout()))) {
        return failure();
      }
    }
    return propagator.run();
  }

  LogicalResult applyLayouts() {
    VMILayoutPropagator propagator(module);
    propagator.setSpineScopedCastOps(&spineScopedCasts);
    addEquivalentLayoutValues(propagator);
    bool failedToPropagate = failed(requestExplicitLayouts(propagator)) ||
                             failed(runLayoutSeedPhases(propagator)) ||
                             failed(requestLateLayouts(propagator)) ||
                             failed(requestFallbackLayouts(propagator));
    if (failedToPropagate) {
      return failure();
    }

    IRRewriter rewriter(ctx);
    return propagator.apply(rewriter);
  }

  void rewriteFunctionType() {
    module.walk([this](func::FuncOp func) {
      if (func.empty()) {
        return;
      }

      SmallVector<Type> inputs = getFunctionInputTypes(func);
      SmallVector<Type> results;
      auto it = firstReturnOperandsByFunc.find(func);
      SmallVector<Type> callResultTypes =
          getConsistentCallResultTypes(module, func);
      if (!callResultTypes.empty()) {
        for (Type type : callResultTypes) {
          results.push_back(type);
        }
      } else if (it != firstReturnOperandsByFunc.end()) {
        for (Value operand : it->second) {
          results.push_back(operand.getType());
        }
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
    if (failed(collect())) {
      return failure();
    }
    // Pure IR analysis: it only needs the op structure, so it runs before the
    // constraint walk decides any seed.  With the recognition disabled
    // directionSpineLegs stays empty and every cast seed goes through the
    // pre-existing tables and phases.
    if (preferCastSpineRoundTrip) {
      collectDirectionSpineLegs(module, directionSpineLegs);
      collectSpineScopedCasts(directionSpineLegs, spineScopedCasts);
    }
    if (failed(addConstraints())) {
      return failure();
    }
    if (failed(applyLayouts())) {
      return failure();
    }
    if (failed(materializeCallBoundaries())) {
      return failure();
    }
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
  SmallVector<MaskUseRequest> maskUseRequests;
  // Cast ops that are legs of a closed nested round trip (see the
  // direction-spine recognition above).  Populated once per module, before the
  // constraint walk creates any seed.
  llvm::SmallPtrSet<Operation *, kDirectionSpineSetInlineCapacity> directionSpineLegs;
  // Subset of directionSpineLegs plus the widening legs that consume them: the
  // cast ops whose reconciliation must use the spine-scoped cast layout table.
  llvm::SmallPtrSet<Operation *, kDirectionSpineSetInlineCapacity> spineScopedCasts;
};

struct VMILayoutAssignmentPass
    : public mlir::pto::impl::VMILayoutAssignmentBase<VMILayoutAssignmentPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VMILayoutAssignmentPass)

  void runOnOperation() override {
    if (failed(LayoutSolver(getOperation()).run())) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVMILayoutAssignmentPass() {
  return std::make_unique<VMILayoutAssignmentPass>();
}

// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- SIMTPersistentFragmentAnalysis.cpp -------------------------------===//
//
// Read-only analysis for lexically persistent SIMT fragments.
//
// The published plan contains only the relationships needed by materialization;
// pointer-walk records and lookup maps remain local to this analysis.
//
//===----------------------------------------------------------------------===//

#include "SIMTPersistentFragmentAnalysis.h"

#include "PTO/Support/CodeConstants.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <cstdint>
#include <limits>
#include <utility>

using namespace mlir;

namespace mlir {
namespace pto {
namespace {

constexpr llvm::StringLiteral kPersistentAttrName = "pto.persistent";
constexpr int64_t kPersistentSlotLimit = 123;
constexpr int64_t kWideValueSlotWidth = 2;

struct PointerWorkItem {
  Value pointer;
  int64_t byteOffset;
};

// Allocation shape needed while normalizing a single pointer use graph. The
// published fragment plan only needs the resulting resident elements and does
// not retain these validation-only values.
struct FragmentShape {
  int64_t elementByteSize;
  int64_t totalByteSize;
};

// These records exist only while the pointer use graph is being normalized.
// The published plan stores the resulting AccessLane values under their
// owning resident element instead of retaining this fragment-level index.
struct NormalizedPersistentAccess {
  Operation *op;
  pto::SectionSimtOp section;
  int64_t elementOffset;
  unsigned laneIndex;
};

struct PersistentAccessDiscovery {
  SmallVector<NormalizedPersistentAccess> accesses;
  llvm::DenseMap<Operation *, SmallVector<unsigned>> accessIndices;
};

static bool isSupportedPersistentScalarType(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    return intType.getWidth() <= mlir::pto::kValue64;
  }
  return type.isF16() || type.isBF16() || type.isF32();
}

static FailureOr<int64_t> getSignedInt64(const APInt &value, Operation *anchor,
                                         StringRef description) {
  if (!value.isSignedIntN(mlir::pto::kValue64)) {
    anchor->emitOpError() << description << " does not fit in signed i64";
    return failure();
  }
  return value.getSExtValue();
}

static FailureOr<int64_t> getConstantInt64(Value value, Operation *anchor,
                                           StringRef description) {
  APInt constant;
  if (!matchPattern(value, m_ConstantInt(&constant))) {
    anchor->emitOpError() << description
                          << " must be a compile-time integer constant";
    return failure();
  }
  return getSignedInt64(constant, anchor, description);
}

static FailureOr<int64_t> getFixedTypeByteSize(Type type, Operation *anchor,
                                               StringRef description,
                                               const DataLayout &dataLayout) {
  llvm::TypeSize typeSize = dataLayout.getTypeSize(type);
  if (typeSize.isScalable() || typeSize.getFixedValue() == 0 ||
      typeSize.getFixedValue() >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    anchor->emitOpError() << description
                          << " must have a non-zero fixed byte size, got "
                          << type;
    return failure();
  }
  return static_cast<int64_t>(typeSize.getFixedValue());
}

static FailureOr<FragmentShape> getFragmentShape(LLVM::AllocaOp allocaOp,
                                                 const DataLayout &dataLayout) {
  Type elementType = allocaOp.getElemType();
  if (!isSupportedPersistentScalarType(elementType)) {
    return allocaOp.emitOpError()
           << "persistent SIMT fragment requires an integer scalar element "
              "type up to 64 bits or f16/bf16/f32, got "
           << elementType;
  }

  FailureOr<int64_t> elementCount = getConstantInt64(
      allocaOp.getArraySize(), allocaOp, "persistent fragment element count");
  if (failed(elementCount)) {
    return failure();
  }
  if (*elementCount <= 0) {
    return allocaOp.emitOpError(
        "persistent SIMT fragment element count must be positive");
  }

  FailureOr<int64_t> elementByteSize = getFixedTypeByteSize(
      elementType, allocaOp, "persistent fragment element type", dataLayout);
  if (failed(elementByteSize)) {
    return failure();
  }

  int64_t totalByteSize;
  if (llvm::MulOverflow(*elementCount, *elementByteSize, totalByteSize)) {
    return allocaOp.emitOpError(
        "persistent SIMT fragment byte size overflows signed i64");
  }

  return FragmentShape{*elementByteSize, totalByteSize};
}

static FailureOr<int64_t> getStaticGEPIndex(LLVM::GEPOp gep) {
  LLVM::GEPIndicesAdaptor<ValueRange> indices = gep.getIndices();
  if (indices.size() != 1) {
    gep.emitOpError(
        "persistent SIMT fragment currently requires llvm.getelementptr "
        "to have exactly one index");
    return failure();
  }

  auto index = indices[0];
  if (auto constant = llvm::dyn_cast_if_present<IntegerAttr>(index)) {
    return getSignedInt64(constant.getValue(), gep, "GEP index");
  }

  Value dynamicIndex = llvm::dyn_cast_if_present<Value>(index);
  if (!dynamicIndex) {
    gep.emitOpError("failed to decode persistent fragment GEP index");
    return failure();
  }
  return getConstantInt64(dynamicIndex, gep, "persistent fragment GEP index");
}

static FailureOr<int64_t> getStaticGEPByteOffset(LLVM::GEPOp gep,
                                                 const DataLayout &dataLayout) {
  FailureOr<int64_t> index = getStaticGEPIndex(gep);
  if (failed(index)) {
    return failure();
  }

  FailureOr<int64_t> elementByteSize = getFixedTypeByteSize(
      gep.getElemType(), gep, "GEP element type", dataLayout);
  if (failed(elementByteSize)) {
    return failure();
  }

  int64_t byteOffset;
  if (llvm::MulOverflow(*index, *elementByteSize, byteOffset)) {
    gep.emitOpError("persistent fragment GEP byte offset overflows signed i64");
    return failure();
  }
  return byteOffset;
}

// Reject vector memory semantics that cannot be preserved by lane splitting.
static LogicalResult validateScalarizableVectorAccess(Operation *access) {
  bool isVolatile = false;
  LLVM::AtomicOrdering ordering = LLVM::AtomicOrdering::not_atomic;
  if (auto load = dyn_cast<LLVM::LoadOp>(access)) {
    isVolatile = load.getVolatile_();
    ordering = load.getOrdering();
  } else {
    auto store = cast<LLVM::StoreOp>(access);
    isVolatile = store.getVolatile_();
    ordering = store.getOrdering();
  }
  if (isVolatile || ordering != LLVM::AtomicOrdering::not_atomic) {
    return access->emitOpError(
        "persistent SIMT fragment vector access cannot be volatile or "
        "atomic");
  }
  return success();
}

// Return the number of scalar element accesses represented by a load/store.
static FailureOr<unsigned>
getPersistentAccessLaneCount(Operation *access, Type accessType,
                             Type allocationElementType) {
  Type scalarType = accessType;
  unsigned laneCount = 1;
  if (auto vectorType = dyn_cast<VectorType>(accessType)) {
    if (vectorType.getRank() != 1 || vectorType.isScalable()) {
      access->emitOpError(
          "persistent SIMT fragment vector access must be fixed-length and "
          "one-dimensional");
      return failure();
    }
    int64_t vectorLaneCount = vectorType.getDimSize(0);
    if (vectorLaneCount != mlir::pto::kValue2 && vectorLaneCount != 4) {
      access->emitOpError()
          << "persistent SIMT fragment currently supports only 2- or 4-lane "
             "vector accesses, got "
          << vectorLaneCount << " lanes";
      return failure();
    }
    scalarType = vectorType.getElementType();
    laneCount = static_cast<unsigned>(vectorLaneCount);
    if (failed(validateScalarizableVectorAccess(access))) {
      return failure();
    }
  }

  if (!isSupportedPersistentScalarType(scalarType)) {
    return access->emitOpError()
           << "persistent SIMT fragment access requires an integer scalar "
              "element type up to 64 bits or f16/bf16/f32, got "
           << scalarType;
  }
  if (scalarType != allocationElementType) {
    return access->emitOpError()
           << "persistent SIMT fragment access element type " << scalarType
           << " must match alloca element type " << allocationElementType;
  }
  return laneCount;
}

static LogicalResult recordAccess(Operation *access, Type accessType,
                                  int64_t byteOffset, func::FuncOp parentFunc,
                                  LLVM::AllocaOp allocaOp,
                                  const FragmentShape &shape,
                                  PersistentAccessDiscovery &discovery) {
  pto::SectionSimtOp section = access->getParentOfType<pto::SectionSimtOp>();
  if (!section) {
    return access->emitOpError(
        "persistent SIMT fragment access must be inside pto.section.simt");
  }
  if (section->getParentOfType<func::FuncOp>() != parentFunc) {
    return access->emitOpError(
        "persistent SIMT fragment access must remain in its defining "
        "function");
  }
  if (!section.getBody().hasOneBlock() ||
      access->getBlock() != &section.getBody().front()) {
    return access->emitOpError(
        "persistent SIMT fragment access must be directly inside the "
        "single top-level block of the pto.section.simt body");
  }

  Type elementType = allocaOp.getElemType();
  FailureOr<unsigned> laneCount =
      getPersistentAccessLaneCount(access, accessType, elementType);
  if (failed(laneCount)) {
    return failure();
  }
  if (byteOffset < 0 || byteOffset % shape.elementByteSize != 0) {
    return access->emitOpError()
           << "persistent SIMT fragment byte offset " << byteOffset
           << " must be non-negative and aligned to element byte size "
           << shape.elementByteSize;
  }

  int64_t accessByteSize;
  if (llvm::MulOverflow(static_cast<int64_t>(*laneCount), shape.elementByteSize,
                        accessByteSize)) {
    return access->emitOpError(
        "persistent SIMT fragment access byte size overflows signed i64");
  }
  int64_t accessEnd;
  if (llvm::AddOverflow(byteOffset, accessByteSize, accessEnd) ||
      accessEnd > shape.totalByteSize) {
    if (*laneCount == 1) {
      return access->emitOpError()
             << "persistent SIMT fragment access at byte offset " << byteOffset
             << " exceeds allocation size " << shape.totalByteSize;
    }
    return access->emitOpError()
           << "persistent SIMT fragment access at byte offset " << byteOffset
           << " with " << *laneCount << " scalar lane(s) exceeds allocation "
           << "size " << shape.totalByteSize;
  }

  if (discovery.accessIndices.count(access)) {
    return access->emitOpError(
        "persistent SIMT fragment access was visited more than once");
  }

  int64_t firstElementOffset = byteOffset / shape.elementByteSize;
  SmallVector<unsigned> accessIndices;
  accessIndices.reserve(*laneCount);
  for (unsigned laneIndex = 0; laneIndex < *laneCount; ++laneIndex) {
    int64_t elementOffset;
    if (llvm::AddOverflow(firstElementOffset, static_cast<int64_t>(laneIndex),
                          elementOffset)) {
      return access->emitOpError(
          "persistent SIMT fragment lane element offset overflows signed "
          "i64");
    }
    accessIndices.push_back(static_cast<unsigned>(discovery.accesses.size()));
    discovery.accesses.push_back({access, section, elementOffset, laneIndex});
  }
  discovery.accessIndices.try_emplace(access, std::move(accessIndices));
  return success();
}

// Compare the lane topology of two inline SIMT sections.
static bool haveSameLaunchDimensions(pto::SectionSimtOp lhs,
                                     pto::SectionSimtOp rhs) {
  return lhs.getDimX() == rhs.getDimX() && lhs.getDimY() == rhs.getDimY() &&
         lhs.getDimZ() == rhs.getDimZ();
}

// Find the unique fragment access section that dominates every other access.
static FailureOr<pto::SectionSimtOp>
findInitSection(DominanceInfo &dominance,
                const PersistentMaterializationPlan &plan,
                const PersistentAccessDiscovery &discovery,
                PersistentFragmentAnalysis &fragment) {
  llvm::DenseSet<Operation *> accessedSectionSet;
  for (const NormalizedPersistentAccess &access : discovery.accesses) {
    if (!access.section) {
      return fragment.allocaOp.emitOpError(
          "persistent fragment access has no containing SIMT section");
    }
    pto::SectionSimtOp accessSection = access.section;
    accessedSectionSet.insert(accessSection.getOperation());
  }

  // Iterate the canonical section list to preserve function walk order without
  // storing a long-lived operation-to-index map in the analysis result.
  SmallVector<pto::SectionSimtOp> accessedSections;
  for (pto::SectionSimtOp section : plan.sections) {
    if (accessedSectionSet.contains(section.getOperation())) {
      accessedSections.push_back(section);
    }
  }
  if (accessedSections.size() != accessedSectionSet.size()) {
    return fragment.allocaOp.emitOpError(
        "failed to locate every persistent fragment access section in the "
        "materialization plan");
  }

  pto::SectionSimtOp initSection;
  for (pto::SectionSimtOp candidate : accessedSections) {
    bool dominatesAllAccessSections =
        llvm::all_of(accessedSections, [&](pto::SectionSimtOp section) {
          if (section == candidate) {
            return true;
          }
          return dominance.dominates(candidate.getOperation(),
                                     section.getOperation());
        });
    if (!dominatesAllAccessSections) {
      continue;
    }

    if (initSection) {
      fragment.allocaOp.emitOpError(
          "persistent SIMT fragment has multiple init section candidates "
          "that dominate every access section");
      return failure();
    }
    initSection = candidate;
  }

  if (!initSection) {
    fragment.allocaOp.emitOpError(
        "persistent SIMT fragment requires a unique init section that "
        "dominates every access section");
    return failure();
  }
  return initSection;
}

// Build the resident set from elements first stored by the init section.
static LogicalResult
collectResidentElements(pto::SectionSimtOp initSection,
                        const PersistentAccessDiscovery &discovery,
                        PersistentFragmentAnalysis &fragment,
                        llvm::DenseSet<int64_t> &residentElementSet) {
  for (Operation &op : initSection.getBody().front()) {
    auto accessIt = discovery.accessIndices.find(&op);
    if (accessIt == discovery.accessIndices.end()) {
      continue;
    }

    for (unsigned accessIndex : accessIt->second) {
      const NormalizedPersistentAccess &access =
          discovery.accesses[accessIndex];
      if (!residentElementSet.insert(access.elementOffset).second) {
        continue;
      }

      if (!isa<LLVM::StoreOp>(access.op)) {
        return access.op->emitOpError()
               << "persistent SIMT fragment element offset "
               << access.elementOffset
               << " must be initialized by a store before its first load in "
                  "the init section";
      }
      fragment.residentElements.push_back(
          {access.elementOffset, 0, /*accesses=*/{}});
    }
  }

  if (fragment.residentElements.empty()) {
    return initSection.emitOpError(
        "persistent SIMT fragment init section must initialize at least one "
        "element");
  }
  llvm::sort(fragment.residentElements, [](const ResidentElementPlan &lhs,
                                           const ResidentElementPlan &rhs) {
    return lhs.elementOffset < rhs.elementOffset;
  });
  return success();
}

// Reject accesses that would extend the init-defined resident set.
static LogicalResult
validateResidentAccesses(const llvm::DenseSet<int64_t> &residentElementSet,
                         const PersistentAccessDiscovery &discovery) {
  for (const NormalizedPersistentAccess &access : discovery.accesses) {
    if (!residentElementSet.contains(access.elementOffset)) {
      return access.op->emitOpError()
             << "persistent SIMT fragment element offset "
             << access.elementOffset
             << " is not initialized by the init section and is not part of "
                "residentElements";
    }
  }
  return success();
}

// Validate the launch topology of every section that carries the fragment.
static LogicalResult
validateCarrySectionDimensions(pto::SectionSimtOp initSection,
                               DominanceInfo &dominance,
                               const PersistentMaterializationPlan &plan,
                               PersistentFragmentAnalysis &fragment) {
  fragment.carrySections.clear();
  for (pto::SectionSimtOp section : plan.sections) {
    if (section == initSection) {
      continue;
    }
    // Only sections dominated by init can carry initialized fragment state.
    if (!dominance.dominates(initSection.getOperation(),
                             section.getOperation())) {
      continue;
    }
    if (!haveSameLaunchDimensions(initSection, section)) {
      return section.emitOpError()
             << "persistent SIMT fragment carry section launch dimensions ("
             << section.getDimX() << ", " << section.getDimY() << ", "
             << section.getDimZ() << ") must match init section dimensions ("
             << initSection.getDimX() << ", " << initSection.getDimY() << ", "
             << initSection.getDimZ() << ")";
    }
    fragment.carrySections.push_back(section);
  }

  return success();
}

// Identify the init section and validate its fixed resident set and carries.
static LogicalResult
analyzeResidentElements(DominanceInfo &dominance,
                        const PersistentMaterializationPlan &plan,
                        PersistentFragmentAnalysis &fragment,
                        const PersistentAccessDiscovery &discovery) {
  FailureOr<pto::SectionSimtOp> initSection =
      findInitSection(dominance, plan, discovery, fragment);
  if (failed(initSection)) {
    return failure();
  }
  fragment.initSection = *initSection;

  llvm::DenseSet<int64_t> residentElementSet;
  if (failed(collectResidentElements(*initSection, discovery, fragment,
                                     residentElementSet))) {
    return failure();
  }
  if (failed(validateResidentAccesses(residentElementSet, discovery))) {
    return failure();
  }

  return validateCarrySectionDimensions(*initSection, dominance, plan,
                                        fragment);
}

static FailureOr<int64_t> getPersistentSlotWidth(Type type, Operation *anchor) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    if (intType.getWidth() <= mlir::pto::kValue32) {
      return 1;
    }
    if (intType.getWidth() <= 64) {
      return kWideValueSlotWidth;
    }
  } else if (type.isF16() || type.isBF16() || type.isF32()) {
    return 1;
  }

  anchor->emitOpError() << "persistent SIMT fragment type " << type
                        << " has no keep/resume slot-width mapping";
  return failure();
}

static LogicalResult
allocatePersistentSlots(func::FuncOp func,
                        PersistentMaterializationPlan &plan) {
  int64_t nextSlot = 0;
  for (PersistentFragmentAnalysis &fragment : plan.fragments) {
    Type elementType = fragment.allocaOp.getElemType();
    FailureOr<int64_t> slotWidth =
        getPersistentSlotWidth(elementType, fragment.allocaOp);
    if (failed(slotWidth)) {
      return failure();
    }

    for (ResidentElementPlan &residentElement : fragment.residentElements) {
      int64_t elementOffset = residentElement.elementOffset;
      int64_t slot = nextSlot;
      if (*slotWidth == mlir::pto::kValue2 && slot % mlir::pto::kValue2 != 0) {
        ++slot;
      }

      if (slot > kPersistentSlotLimit - *slotWidth) {
        return fragment.allocaOp.emitOpError()
               << "persistent SIMT fragment slot allocation in function '"
               << func.getSymName() << "' exceeds the " << kPersistentSlotLimit
               << "-slot limit: element offset " << elementOffset << " of type "
               << elementType << " needs " << *slotWidth
               << " slot(s), next candidate slot is " << slot;
      }

      residentElement.slot = static_cast<unsigned>(slot);
      nextSlot = slot + *slotWidth;
    }
  }

  return success();
}

// Publish normalized accesses under their owning resident element. Iterating
// sections and block operations gives the final plan a stable
// section/order/lane traversal independent of pointer-use walk order.
static LogicalResult
materializeResidentAccessLanes(const PersistentMaterializationPlan &plan,
                               PersistentFragmentAnalysis &fragment,
                               const PersistentAccessDiscovery &discovery) {
  size_t materializedAccessCount = 0;
  for (pto::SectionSimtOp section : plan.sections) {
    if (!section.getBody().hasOneBlock()) {
      return section.emitOpError(
          "persistent fragment access materialization requires a "
          "single-block SIMT section");
    }
    for (Operation &op : section.getBody().front()) {
      auto accessIt = discovery.accessIndices.find(&op);
      if (accessIt == discovery.accessIndices.end()) {
        continue;
      }

      for (unsigned accessIndex : accessIt->second) {
        if (accessIndex >= discovery.accesses.size()) {
          return fragment.allocaOp.emitOpError(
              "persistent fragment discovery has an invalid access index");
        }
        const NormalizedPersistentAccess &access =
            discovery.accesses[accessIndex];
        if (access.op != &op || access.section != section) {
          return access.op->emitOpError(
              "persistent fragment normalized access does not match its "
              "section operation");
        }

        ResidentElementPlan *residentElement =
            fragment.findResidentElement(access.elementOffset);
        if (!residentElement) {
          return access.op->emitOpError()
                 << "persistent fragment normalized access element offset "
                 << access.elementOffset << " is not resident";
        }
        residentElement->accesses.push_back({access.op, access.laneIndex});
        ++materializedAccessCount;
      }
    }
  }

  if (materializedAccessCount != discovery.accesses.size()) {
    return fragment.allocaOp.emitOpError(
        "failed to materialize every normalized persistent fragment access "
        "under a resident element");
  }
  return success();
}

static LogicalResult
analyzePersistentFragment(LLVM::AllocaOp allocaOp, DominanceInfo &dominance,
                          const PersistentMaterializationPlan &plan,
                          PersistentFragmentAnalysis &fragment,
                          PersistentAccessDiscovery &discovery) {
  func::FuncOp parentFunc = allocaOp->getParentOfType<func::FuncOp>();
  if (!parentFunc) {
    return allocaOp.emitOpError("must be nested in a func.func");
  }

  if (!isa<UnitAttr>(allocaOp->getAttr(kPersistentAttrName))) {
    return allocaOp.emitOpError()
           << "expects '" << kPersistentAttrName << "' to be a unit attribute";
  }
  if (allocaOp->getParentOfType<pto::SectionSimtOp>()) {
    return allocaOp.emitOpError(
        "persistent SIMT fragment must be defined outside pto.section.simt");
  }
  if (parentFunc->hasAttr(pto::kPTOSimtEntryAttrName)) {
    return allocaOp.emitOpError(
        "persistent SIMT fragment must be defined before SIMT outlining");
  }

  DataLayout dataLayout = DataLayout::closest(allocaOp);
  FailureOr<FragmentShape> shape = getFragmentShape(allocaOp, dataLayout);
  if (failed(shape)) {
    return failure();
  }

  SmallVector<PointerWorkItem> pointerWorklist{{allocaOp.getRes(), 0}};
  llvm::DenseMap<Value, int64_t> pointerOffsets;

  while (!pointerWorklist.empty()) {
    PointerWorkItem item = pointerWorklist.pop_back_val();
    auto [offsetIt, inserted] =
        pointerOffsets.try_emplace(item.pointer, item.byteOffset);
    if (!inserted) {
      if (offsetIt->second != item.byteOffset) {
        return allocaOp.emitOpError(
            "persistent fragment pointer has inconsistent byte offsets");
      }
      continue;
    }

    for (OpOperand &use : item.pointer.getUses()) {
      Operation *user = use.getOwner();
      if (user->getParentOfType<func::FuncOp>() != parentFunc) {
        return user->emitOpError(
            "persistent SIMT fragment pointer must remain in its defining "
            "function");
      }
      if (!dominance.dominates(allocaOp.getOperation(), user)) {
        return user->emitOpError(
            "persistent SIMT fragment definition must dominate every use");
      }

      if (auto gep = dyn_cast<LLVM::GEPOp>(user)) {
        if (use.getOperandNumber() != 0) {
          return gep.emitOpError(
              "persistent SIMT fragment pointer must be the base of "
              "llvm.getelementptr");
        }

        FailureOr<int64_t> gepByteOffset =
            getStaticGEPByteOffset(gep, dataLayout);
        if (failed(gepByteOffset)) {
          return failure();
        }
        int64_t derivedByteOffset;
        if (llvm::AddOverflow(item.byteOffset, *gepByteOffset,
                              derivedByteOffset)) {
          return gep.emitOpError(
              "persistent fragment cumulative byte offset overflows signed "
              "i64");
        }
        pointerWorklist.push_back({gep.getRes(), derivedByteOffset});
        continue;
      }

      if (auto load = dyn_cast<LLVM::LoadOp>(user)) {
        if (failed(recordAccess(load, load.getRes().getType(), item.byteOffset,
                                parentFunc, allocaOp, *shape, discovery))) {
          return failure();
        }
        continue;
      }

      if (auto store = dyn_cast<LLVM::StoreOp>(user)) {
        if (use.getOperandNumber() != 1) {
          return store.emitOpError(
              "persistent SIMT fragment pointer must not be stored as a "
              "value");
        }
        if (failed(recordAccess(store, store.getValue().getType(),
                                item.byteOffset, parentFunc, allocaOp, *shape,
                                discovery))) {
          return failure();
        }
        continue;
      }

      return user->emitOpError()
             << "unsupported use of persistent SIMT fragment pointer by '"
             << user->getName() << "'";
    }
  }

  if (discovery.accesses.empty()) {
    return allocaOp.emitOpError(
        "persistent SIMT fragment requires at least one llvm.load or "
        "llvm.store inside pto.section.simt");
  }

  return analyzeResidentElements(dominance, plan, fragment, discovery);
}

} // namespace

SIMTPersistentFragmentAnalysis::SIMTPersistentFragmentAnalysis(
    func::FuncOp func) {
  SmallVector<LLVM::AllocaOp> persistentAllocas;
  func.walk([&](LLVM::AllocaOp allocaOp) {
    if (allocaOp->hasAttr(kPersistentAttrName)) {
      persistentAllocas.push_back(allocaOp);
    }
  });

  // A function without persistent allocations has a valid empty plan and does
  // not need to satisfy the PTO kernel-entry constraint.
  if (persistentAllocas.empty()) {
    plan.emplace();
    return;
  }

  if (!pto::isPTOEntryFunction(func)) {
    persistentAllocas.front().emitOpError(
        "persistent SIMT fragment must be defined in a PTO kernel entry");
    return;
  }

  PersistentMaterializationPlan candidate;
  func.walk([&](pto::SectionSimtOp section) {
    candidate.sections.push_back(section);
  });

  candidate.fragments.reserve(persistentAllocas.size());
  SmallVector<PersistentAccessDiscovery> accessDiscoveries;
  accessDiscoveries.reserve(persistentAllocas.size());
  DominanceInfo dominance(func);
  for (LLVM::AllocaOp allocaOp : persistentAllocas) {
    candidate.fragments.emplace_back(allocaOp);
    accessDiscoveries.emplace_back();
    if (failed(analyzePersistentFragment(allocaOp, dominance, candidate,
                                         candidate.fragments.back(),
                                         accessDiscoveries.back()))) {
      return;
    }
  }

  if (failed(allocatePersistentSlots(func, candidate))) {
    return;
  }
  for (unsigned fragmentIndex = 0;
       fragmentIndex < static_cast<unsigned>(candidate.fragments.size());
       ++fragmentIndex) {
    if (failed(materializeResidentAccessLanes(
            candidate, candidate.fragments[fragmentIndex],
            accessDiscoveries[fragmentIndex]))) {
      return;
    }
  }
  // Publish only a fully checked plan. No analysis result is visible to the
  // materialization pass after an intermediate failure.
  plan.emplace(std::move(candidate));
}

} // namespace pto
} // namespace mlir

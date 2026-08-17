// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOMaterializeSIMTPersistentFragment.cpp ---------------------------===//
//
// Materialize lexically persistent SIMT fragments into pto.keep/pto.resume.
//
// The read-only discovery and planning lives in
// SIMTPersistentFragmentAnalysis. This pass only consumes the cached plan and
// performs section-local IR materialization.
//
//===----------------------------------------------------------------------===//

#include "PTO/Support/CodeConstants.h"
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "SIMTPersistentFragmentAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Mem2Reg.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cassert>
#include <cstdint>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOMATERIALIZESIMTPERSISTENTFRAGMENT
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

using pto::AccessLane;
using pto::PersistentFragmentAnalysis;
using pto::PersistentMaterializationPlan;
using pto::ResidentElementPlan;

struct PersistentElementRewrite {
  LLVM::AllocaOp proxy;
  Value resumeValue;
};

struct PersistentLaneRewrite {
  // Scalar component index in the original access. Scalar accesses use lane 0.
  unsigned laneIndex;
  // Index into the section-local elements and parallel elementRewrites arrays.
  unsigned elementIndex;
};

struct PersistentElementWorkItem {
  const PersistentFragmentAnalysis *fragment = nullptr;
  const ResidentElementPlan *residentElement = nullptr;
  // Access lanes for this element that occur in the owning worklist section.
  SmallVector<AccessLane> accesses;
};

// A section-local, fully validated view of the immutable analysis plan. The
// worklist directly references fragment/element plans and owns only local lane
// bindings, so the transform does not rediscover them while mutating the body.
struct PersistentSectionWorklist {
  explicit PersistentSectionWorklist(pto::SectionSimtOp section)
      : section(section) {}

  pto::SectionSimtOp section;
  SmallVector<PersistentElementWorkItem> elements;
  llvm::DenseMap<Operation *, SmallVector<PersistentLaneRewrite>>
      laneRewritesByAccess;
};

struct PersistentTransformWorklist {
  SmallVector<PersistentSectionWorklist> sections;
};

// Redirect one scalar access to its section-local proxy.
static LogicalResult rewireScalarAccess(Operation *access,
                                        LLVM::AllocaOp proxy) {
  if (auto load = dyn_cast<LLVM::LoadOp>(access)) {
    load.getAddrMutable().assign(proxy.getRes());
    return success();
  }
  if (auto store = dyn_cast<LLVM::StoreOp>(access)) {
    store.getAddrMutable().assign(proxy.getRes());
    return success();
  }
  return access->emitOpError(
      "expected a scalar llvm.load or llvm.store during persistent "
      "fragment materialization");
}

// Return the scalar or vector value type carried by an LLVM memory access.
static FailureOr<Type> getPersistentAccessType(Operation *access) {
  if (auto load = dyn_cast<LLVM::LoadOp>(access)) {
    return load.getRes().getType();
  }
  if (auto store = dyn_cast<LLVM::StoreOp>(access)) {
    return store.getValue().getType();
  }
  access->emitOpError(
      "expected an llvm.load or llvm.store persistent fragment access");
  return failure();
}

// Validate and order each operation's lane-to-proxy rewrite before mutation.
static LogicalResult validateAccessRewritePlan(
    llvm::DenseMap<Operation *, SmallVector<PersistentLaneRewrite>>
        &laneRewritesByAccess) {
  for (auto &[access, laneRewrites] : laneRewritesByAccess) {
    FailureOr<Type> accessType = getPersistentAccessType(access);
    if (failed(accessType)) {
      return failure();
    }

    llvm::sort(laneRewrites, [](const PersistentLaneRewrite &lhs,
                                const PersistentLaneRewrite &rhs) {
      return lhs.laneIndex < rhs.laneIndex;
    });

    unsigned expectedLaneCount = 1;
    if (auto vectorType = dyn_cast<VectorType>(*accessType)) {
      expectedLaneCount = static_cast<unsigned>(vectorType.getNumElements());
    }

    if (laneRewrites.size() != expectedLaneCount) {
      return access->emitOpError()
             << "persistent fragment access expected " << expectedLaneCount
             << " scalar lane rewrite(s), got " << laneRewrites.size();
    }
    for (auto [expectedLaneIndex, laneRewrite] :
         llvm::enumerate(laneRewrites)) {
      if (laneRewrite.laneIndex != expectedLaneIndex) {
        return access->emitOpError()
               << "persistent fragment access has non-contiguous lane "
                  "mapping at lane "
               << expectedLaneIndex;
      }
    }
  }
  return success();
}

static bool
isFragmentActiveInSection(const PersistentFragmentAnalysis &fragment,
                          pto::SectionSimtOp section) {
  if (fragment.initSection == section) {
    return true;
  }
  return llvm::any_of(fragment.carrySections, [&](pto::SectionSimtOp carry) {
    return carry == section;
  });
}

// Build and validate all section-local rewrite bindings before touching IR.
// The analysis result already fixes each fragment's init/carry lifetime. This
// check only verifies that the temporary worklist faithfully materializes that
// immutable result.
static LogicalResult
validateSectionWorklist(const PersistentMaterializationPlan &plan,
                        PersistentSectionWorklist &sectionWorklist,
                        llvm::DenseMap<Operation *, llvm::DenseSet<unsigned>>
                            &assignedAccessLanes) {
  pto::SectionSimtOp section = sectionWorklist.section;
  if (!section || !section.getBody().hasOneBlock()) {
    return section ? section.emitOpError(
                         "persistent fragment transform requires a single-"
                         "block SIMT section")
                   : failure();
  }

  unsigned expectedElementIndex = 0;
  for (const PersistentFragmentAnalysis &fragment : plan.fragments) {
    if (!isFragmentActiveInSection(fragment, section)) {
      continue;
    }

    for (const ResidentElementPlan &expectedElement :
         fragment.residentElements) {
      if (expectedElementIndex >= sectionWorklist.elements.size()) {
        LLVM::AllocaOp allocaOp = fragment.allocaOp;
        return allocaOp.emitOpError(
            "persistent fragment section worklist is missing the complete "
            "resident element set");
      }

      const PersistentElementWorkItem &element =
          sectionWorklist.elements[expectedElementIndex];
      if (element.fragment != &fragment ||
          element.residentElement != &expectedElement) {
        return section.emitOpError(
            "persistent fragment section worklist does not preserve fragment "
            "and resident element order");
      }

      const ResidentElementPlan &residentElement = *element.residentElement;

      for (const AccessLane &accessLane : element.accesses) {
        if (!accessLane.op) {
          LLVM::AllocaOp allocaOp = fragment.allocaOp;
          return allocaOp.emitOpError(
              "persistent fragment worklist contains a null access "
              "operation");
        }

        bool isOwnedAccess = llvm::any_of(
            residentElement.accesses, [&](const AccessLane &candidate) {
              return candidate.op == accessLane.op &&
                     candidate.laneIndex == accessLane.laneIndex;
            });
        if (!isOwnedAccess) {
          return accessLane.op->emitOpError(
              "persistent fragment worklist access lane is owned by a "
              "different resident element");
        }

        pto::SectionSimtOp accessSection =
            accessLane.op->getParentOfType<pto::SectionSimtOp>();
        if (accessSection != section ||
            accessLane.op->getBlock() != &section.getBody().front()) {
          return accessLane.op->emitOpError(
              "persistent fragment access does not belong to its planned "
              "SIMT section body");
        }
        if (!assignedAccessLanes[accessLane.op]
                 .insert(accessLane.laneIndex)
                 .second) {
          return accessLane.op->emitOpError(
              "persistent fragment access lane was assigned to more than one "
              "transform work item");
        }
        // Invert the element-owned access into an operation-owned lane mapping.
        // elementIndex selects this lane's scalar proxy from elementRewrites.
        sectionWorklist.laneRewritesByAccess[accessLane.op].push_back(
            {accessLane.laneIndex, expectedElementIndex});
      }

      ++expectedElementIndex;
    }
  }

  if (expectedElementIndex != sectionWorklist.elements.size()) {
    return section.emitOpError(
        "persistent fragment section worklist contains an unexpected "
        "resident element");
  }

  return validateAccessRewritePlan(sectionWorklist.laneRewritesByAccess);
}

// Construct the complete transform worklist. No operation insertion, erase,
// or operand replacement is allowed before this function succeeds.
static LogicalResult
buildPersistentTransformWorklist(const PersistentMaterializationPlan &plan,
                                 PersistentTransformWorklist &worklist) {
  worklist.sections.clear();
  worklist.sections.reserve(plan.sections.size());
  for (pto::SectionSimtOp section : plan.sections) {
    worklist.sections.emplace_back(section);
  }

  llvm::DenseMap<Operation *, PersistentSectionWorklist *> worklistBySection;
  for (PersistentSectionWorklist &sectionWorklist : worklist.sections) {
    if (!sectionWorklist.section) {
      return failure();
    }
    auto [it, inserted] = worklistBySection.try_emplace(
        sectionWorklist.section.getOperation(), &sectionWorklist);
    (void)it;
    if (!inserted) {
      return sectionWorklist.section.emitOpError(
          "persistent fragment section appears more than once in the plan");
    }
  }

  // Append each fragment's complete resident set to its init and carry
  // sections. The outer section vector remains in function walk order;
  // appending fragments in alloca order preserves function-wide slot order.
  for (const PersistentFragmentAnalysis &fragment : plan.fragments) {
    LLVM::AllocaOp allocaOp = fragment.allocaOp;
    if (!fragment.initSection) {
      return allocaOp.emitOpError(
          "persistent fragment has no init section in its analysis plan");
    }

    llvm::DenseSet<Operation *> addedSections;
    auto addToSection = [&](pto::SectionSimtOp section) -> LogicalResult {
      if (!addedSections.insert(section.getOperation()).second) {
        return allocaOp.emitOpError(
            "persistent fragment init/carry sections contain a duplicate");
      }
      auto sectionIt = worklistBySection.find(section.getOperation());
      if (sectionIt == worklistBySection.end()) {
        return allocaOp.emitOpError(
            "persistent fragment init/carry section is not in the plan");
      }

      PersistentSectionWorklist &sectionWorklist = *sectionIt->second;
      for (const ResidentElementPlan &residentElement :
           fragment.residentElements) {
        SmallVector<AccessLane> localAccesses;
        for (const AccessLane &accessLane : residentElement.accesses) {
          pto::SectionSimtOp accessSection =
              accessLane.op
                  ? accessLane.op->getParentOfType<pto::SectionSimtOp>()
                  : pto::SectionSimtOp();
if (accessSection == section) {
          localAccesses.push_back(accessLane);
        }
        }
        sectionWorklist.elements.push_back(
            {&fragment, &residentElement, std::move(localAccesses)});
      }
      return success();
    };
    if (failed(addToSection(fragment.initSection))) {
      return failure();
    }
    for (pto::SectionSimtOp section : fragment.carrySections) {
      if (failed(addToSection(section))) {
        return failure();
      }
    }
  }

  llvm::DenseMap<Operation *, llvm::DenseSet<unsigned>> assignedAccessLanes;
  for (PersistentSectionWorklist &sectionWorklist : worklist.sections) {
    if (failed(validateSectionWorklist(plan, sectionWorklist,
                                       assignedAccessLanes))) {
      return failure();
    }
  }

  for (const PersistentFragmentAnalysis &fragment : plan.fragments) {
    for (const ResidentElementPlan &element : fragment.residentElements) {
      for (const AccessLane &accessLane : element.accesses) {
        if (!accessLane.op) {
          LLVM::AllocaOp allocaOp = fragment.allocaOp;
          return allocaOp.emitOpError(
              "persistent fragment analysis contains a null access "
              "operation");
        }
        auto assignedIt = assignedAccessLanes.find(accessLane.op);
        if (assignedIt == assignedAccessLanes.end() ||
            !assignedIt->second.contains(accessLane.laneIndex)) {
          accessLane.op->emitOpError(
              "persistent fragment access lane was not assigned to a "
              "transform work item");
          return failure();
        }
      }
    }
  }
  return success();
}

// Rewrite one analyzed access against its section-local scalar proxies.
static LogicalResult rewritePersistentAccess(
    Operation *access, ArrayRef<PersistentLaneRewrite> laneRewrites,
    MutableArrayRef<PersistentElementRewrite> elementRewrites) {
  FailureOr<Type> accessType = getPersistentAccessType(access);
  if (failed(accessType)) {
    return failure();
  }

  if (!isa<VectorType>(*accessType)) {
    assert(laneRewrites.size() == 1 && laneRewrites.front().laneIndex == 0 &&
           "scalar access must map to exactly one scalar lane");
    LLVM::AllocaOp proxy =
        elementRewrites[laneRewrites.front().elementIndex].proxy;
    assert(proxy && "locally accessed element must have a scalar proxy");
    return rewireScalarAccess(access, proxy);
  }

  OpBuilder builder(access);
  Location loc = access->getLoc();
  if (auto store = dyn_cast<LLVM::StoreOp>(access)) {
    for (const PersistentLaneRewrite &laneRewrite : laneRewrites) {
      LLVM::AllocaOp proxy = elementRewrites[laneRewrite.elementIndex].proxy;
      assert(proxy && "vector store lane must have a scalar proxy");
      Value laneIndex = builder.create<arith::ConstantIntOp>(
          loc, laneRewrite.laneIndex, /*width=*/32);
      Value laneValue = builder.create<LLVM::ExtractElementOp>(
          loc, store.getValue(), laneIndex);
      builder.create<LLVM::StoreOp>(loc, laneValue, proxy.getRes());
    }
    store.erase();
    return success();
  }

  auto load = cast<LLVM::LoadOp>(access);
  auto vectorType = cast<VectorType>(*accessType);
  Value rebuiltVector = builder.create<LLVM::PoisonOp>(loc, vectorType);
  for (const PersistentLaneRewrite &laneRewrite : laneRewrites) {
    LLVM::AllocaOp proxy = elementRewrites[laneRewrite.elementIndex].proxy;
    assert(proxy && "vector load lane must have a scalar proxy");
    Value laneValue = builder.create<LLVM::LoadOp>(
        loc, vectorType.getElementType(), proxy.getRes());
    Value laneIndex = builder.create<arith::ConstantIntOp>(
        loc, laneRewrite.laneIndex, /*width=*/32);
    rebuiltVector = builder.create<LLVM::InsertElementOp>(
        loc, vectorType, rebuiltVector, laneValue, laneIndex);
  }
  load.getRes().replaceAllUsesWith(rebuiltVector);
  load.erase();
  return success();
}

// Materialize all active persistent elements in one inline SIMT section.
static LogicalResult
materializeSection(const PersistentSectionWorklist &sectionWorklist,
                   const DataLayout &dataLayout, DominanceInfo &dominance) {
  const auto &elements = sectionWorklist.elements;
  if (elements.empty()) {
    return success();
  }

  pto::SectionSimtOp section = sectionWorklist.section;
  Block &body = section.getBody().front();
  SmallVector<PersistentElementRewrite> rewrites(elements.size());

  // The section-local access/lane bindings were completely validated before
  // the first section was modified.
  const auto &laneRewritesByAccess = sectionWorklist.laneRewritesByAccess;

  OpBuilder entryBuilder(section.getContext());
  entryBuilder.setInsertionPointToStart(&body);

  bool hasLocalAccess = !laneRewritesByAccess.empty();
  Value proxyArraySize;
  if (hasLocalAccess) {
    proxyArraySize = entryBuilder.create<arith::ConstantIntOp>(section.getLoc(),
                                                               1, /*width=*/mlir::pto::kValue32);
  }

  // Emit the complete resume prologue before proxy setup so the group remains
  // contiguous. Elements initialized by this section have no incoming value.
  for (auto [elementIndex, element] : llvm::enumerate(elements)) {
    const PersistentFragmentAnalysis &fragment = *element.fragment;
    const ResidentElementPlan &residentElement = *element.residentElement;
    LLVM::AllocaOp allocaOp = fragment.allocaOp;
    if (section == fragment.initSection) {
      continue;
    }

    rewrites[elementIndex].resumeValue =
        entryBuilder
            .create<pto::ResumeOp>(section.getLoc(), allocaOp.getElemType(),
                                   static_cast<uint64_t>(residentElement.slot))
            .getResult();
  }

  // A proxy gives generic mem2reg one scalar slot in the same region as all of
  // its accesses. Carry sections seed it from resume; init sections rely on
  // the previously validated first-store initialization.
  for (auto [elementIndex, element] : llvm::enumerate(elements)) {
    if (element.accesses.empty()) {
      continue;
    }

    const PersistentFragmentAnalysis &fragment = *element.fragment;
    LLVM::AllocaOp allocaOp = fragment.allocaOp;
    Location elementLoc = element.accesses.front().op->getLoc();
    LLVM::AllocaOp proxy = entryBuilder.create<LLVM::AllocaOp>(
        elementLoc, allocaOp.getRes().getType(), allocaOp.getElemType(),
        proxyArraySize);
    rewrites[elementIndex].proxy = proxy;

    if (section != fragment.initSection) {
      assert(rewrites[elementIndex].resumeValue &&
             "carry element must have a resume value");
      entryBuilder.create<LLVM::StoreOp>(
          elementLoc, rewrites[elementIndex].resumeValue, proxy.getRes());
    }
  }

  // Rewrite original accesses only after every section-local proxy exists.
  // Block order gives vector scalarization a stable operation-level traversal.
  size_t rewrittenAccessCount = 0;
  for (Operation &op : llvm::make_early_inc_range(body)) {
    auto accessIt = laneRewritesByAccess.find(&op);
    if (accessIt == laneRewritesByAccess.end()) {
      continue;
    }
    if (failed(rewritePersistentAccess(&op, accessIt->second, rewrites))) {
      return failure();
    }
    ++rewrittenAccessCount;
  }
  if (rewrittenAccessCount != laneRewritesByAccess.size()) {
    return section.emitOpError(
        "failed to rewrite every persistent fragment access in block order");
  }

  OpBuilder exitBuilder(section.getContext());
  exitBuilder.setInsertionPointToEnd(&body);

  // Compute every outgoing value before emitting keeps so the keep epilogue is
  // one contiguous group.
  SmallVector<Value> keepPayloads;
  keepPayloads.reserve(elements.size());
  for (auto [elementIndex, element] : llvm::enumerate(elements)) {
    const PersistentFragmentAnalysis &fragment = *element.fragment;
    LLVM::AllocaOp allocaOp = fragment.allocaOp;
    if (rewrites[elementIndex].proxy) {
      keepPayloads.push_back(
          exitBuilder
              .create<LLVM::LoadOp>(section.getLoc(), allocaOp.getElemType(),
                                    rewrites[elementIndex].proxy.getRes())
              .getResult());
      continue;
    }
    assert(rewrites[elementIndex].resumeValue &&
           "an element without local accesses must pass through resume");
    keepPayloads.push_back(rewrites[elementIndex].resumeValue);
  }

  for (auto [element, payload] : llvm::zip_equal(elements, keepPayloads)) {
    const ResidentElementPlan &residentElement = *element.residentElement;
    exitBuilder.create<pto::KeepOp>(
        section.getLoc(), payload, static_cast<uint64_t>(residentElement.slot));
  }

  // Promote each proxy independently so success guarantees that this specific
  // temporary allocation and all of its memory traffic were eliminated.
  OpBuilder promotionBuilder(section.getContext());
  promotionBuilder.setInsertionPointToStart(&body);
  for (auto [elementIndex, element] : llvm::enumerate(elements)) {
    LLVM::AllocaOp proxy = rewrites[elementIndex].proxy;
    if (!proxy) {
      continue;
    }
    SmallVector<PromotableAllocationOpInterface, 1> allocators{
        cast<PromotableAllocationOpInterface>(proxy.getOperation())};
    if (failed(tryToPromoteMemorySlots(allocators, promotionBuilder, dataLayout,
                                       dominance))) {
      return section.emitOpError()
             << "failed to promote scalar proxy for persistent fragment "
                "element offset "
             << element.residentElement->elementOffset;
    }
  }

  if (proxyArraySize && proxyArraySize.use_empty()) {
    proxyArraySize.getDefiningOp()->erase();
  }
  return success();
}

// Remove the now-dead GEP tree rooted at a persistent alloca result.
static LogicalResult eraseDeadPersistentGEPs(Value pointer) {
  SmallVector<LLVM::GEPOp> derivedPointers;
  for (Operation *user : pointer.getUsers()) {
    auto gep = dyn_cast<LLVM::GEPOp>(user);
    if (!gep) {
      return user->emitOpError(
          "persistent fragment access remained after scalar proxy promotion");
    }
    derivedPointers.push_back(gep);
  }

  for (LLVM::GEPOp gep : derivedPointers) {
    if (failed(eraseDeadPersistentGEPs(gep.getRes()))) {
      return failure();
    }
    if (!gep.getRes().use_empty()) {
      return gep.emitOpError(
          "persistent fragment derived pointer still has live uses after "
          "scalar proxy promotion");
    }
    gep.erase();
  }
  return success();
}

// Rewrite every planned section, then remove the original lexical carriers.
static LogicalResult
materializePersistentFragments(func::FuncOp func, DominanceInfo &dominance,
                               const PersistentMaterializationPlan &plan,
                               const PersistentTransformWorklist &worklist) {
  DataLayout dataLayout = DataLayout::closest(func);
  for (const PersistentSectionWorklist &sectionWorklist : worklist.sections) {
    if (failed(materializeSection(sectionWorklist, dataLayout, dominance))) {
      return failure();
    }
  }

  for (const PersistentFragmentAnalysis &fragment : plan.fragments) {
    LLVM::AllocaOp allocaOp = fragment.allocaOp;
    if (failed(eraseDeadPersistentGEPs(allocaOp.getRes()))) {
      return failure();
    }
    if (!allocaOp.getRes().use_empty()) {
      return allocaOp.emitOpError(
          "persistent fragment still has live uses after materialization");
    }
    allocaOp.erase();
  }
  return success();
}

struct PTOMaterializeSIMTPersistentFragmentPass
    : public pto::impl::PTOMaterializeSIMTPersistentFragmentBase<
          PTOMaterializeSIMTPersistentFragmentPass> {
  using pto::impl::PTOMaterializeSIMTPersistentFragmentBase<
      PTOMaterializeSIMTPersistentFragmentPass>::
      PTOMaterializeSIMTPersistentFragmentBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    const auto &analysis = getAnalysis<pto::SIMTPersistentFragmentAnalysis>();
    if (!analysis.isValid()) {
      signalPassFailure();
      return;
    }

    const PersistentMaterializationPlan &plan = analysis.getPlan();
    if (plan.fragments.empty()) {
      return;
    }

    // Convert the fragment/element-oriented analysis plan into a validated,
    // section/operation-oriented worklist before any IR mutation.
    PersistentTransformWorklist worklist;
    if (failed(buildPersistentTransformWorklist(plan, worklist))) {
      signalPassFailure();
      return;
    }
    DominanceInfo dominance(func);
    if (failed(
            materializePersistentFragments(func, dominance, plan, worklist))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::pto::createPTOMaterializeSIMTPersistentFragmentPass() {
  return std::make_unique<PTOMaterializeSIMTPersistentFragmentPass>();
}

// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VPTOSPLITCVMODULE
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

static bool hasKernelKind(ModuleOp module) {
  return module->hasAttr(FunctionKernelKindAttr::name);
}

static bool hasKernelKindChildModule(ModuleOp module) {
  return llvm::any_of(module.getOps<ModuleOp>(),
                      [](ModuleOp child) { return hasKernelKind(child); });
}

/// Returns true when at least one top-level function in \p module carries a
/// per-func `pto.kernel_kind` attribute. This is the "sugar" input form where
/// each kernel function declares its own kind instead of living under a
/// kind-tagged child module.
static bool hasKernelKindTopLevelFunc(ModuleOp module) {
  return llvm::any_of(module.getOps<func::FuncOp>(), [](func::FuncOp funcOp) {
    return funcOp->hasAttr(FunctionKernelKindAttr::name);
  });
}

static bool hasCVSections(ModuleOp module);

static bool isVPTOBackendModule(ModuleOp module) {
  auto backend = module->getAttrOfType<StringAttr>("pto.backend");
  return backend && backend.getValue() == "vpto";
}

static bool hasConflictingContainerAttrs(ModuleOp outer, ModuleOp child) {
  for (NamedAttribute attr : child->getAttrs()) {
    if (attr.getName() == SymbolTable::getSymbolAttrName()) {
      continue;
    }
    Attribute outerValue = outer->getAttr(attr.getName());
    if (outerValue && outerValue != attr.getValue()) {
      return true;
    }
  }
  return false;
}

static bool flattenSingleUnpartitionedChild(ModuleOp module) {
  SmallVector<Operation *> topLevelOps;
  for (Operation &op : module.getBodyRegion().front().getOperations()) {
    topLevelOps.push_back(&op);
  }
  if (topLevelOps.size() != 1) {
    return false;
  }

  auto child = dyn_cast<ModuleOp>(topLevelOps.front());
  if (!child || !isVPTOBackendModule(child) || hasKernelKind(child) ||
      !hasCVSections(child) || hasConflictingContainerAttrs(module, child)) {
    return false;
  }

  SmallVector<NamedAttribute> childAttrs(child->getAttrs().begin(),
                                         child->getAttrs().end());
  Region childBody;
  childBody.takeBody(child.getBodyRegion());
  child.erase();
  module.getBodyRegion().takeBody(childBody);
  for (NamedAttribute attr : childAttrs) {
    if (attr.getName() != SymbolTable::getSymbolAttrName()) {
      module->setAttr(attr.getName(), attr.getValue());
    }
  }
  return true;
}

static bool isSectionSplitCandidate(func::FuncOp funcOp);

static bool hasCVSections(ModuleOp module) {
  bool found = false;
  module.walk([&](func::FuncOp funcOp) {
    if (found || !isSectionSplitCandidate(funcOp)) {
      return WalkResult::advance();
    }
    WalkResult result = funcOp.walk([&](Operation *op) {
      if (isa<SectionCubeOp, SectionVectorOp>(op)) {
        found = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return result.wasInterrupted() ? WalkResult::interrupt()
                                   : WalkResult::advance();
  });
  return found;
}

static bool hasSectionKind(ModuleOp module, FunctionKernelKind kind) {
  bool found = false;
  module.walk([&](func::FuncOp funcOp) {
    if (found || !isSectionSplitCandidate(funcOp)) {
      return WalkResult::advance();
    }
    WalkResult result = funcOp.walk([&](Operation *op) {
      bool matches = kind == FunctionKernelKind::Cube
                         ? isa<SectionCubeOp>(op)
                         : isa<SectionVectorOp>(op);
      if (matches) {
        found = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return result.wasInterrupted() ? WalkResult::interrupt()
                                   : WalkResult::advance();
  });
  return found;
}

static bool hasSectionKind(func::FuncOp funcOp, FunctionKernelKind kind) {
  bool found = false;
  funcOp.walk([&](Operation *op) {
    bool matches = kind == FunctionKernelKind::Cube ? isa<SectionCubeOp>(op)
                                                    : isa<SectionVectorOp>(op);
    if (matches) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static bool hasAnySection(func::FuncOp funcOp) {
  bool found = false;
  funcOp.walk([&](Operation *op) {
    if (isa<SectionCubeOp, SectionVectorOp>(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static bool isSectionSplitCandidate(func::FuncOp funcOp) {
  return funcOp && !funcOp.isDeclaration() &&
         (pto::isPTOEntryFunction(funcOp) || hasAnySection(funcOp));
}

static LogicalResult verifyNoNestedSections(ModuleOp module) {
  LogicalResult status = success();
  module.walk([&](Operation *op) {
    if (failed(status) || !isa<SectionCubeOp, SectionVectorOp>(op)) {
      return WalkResult::advance();
    }
    Operation *parent = op->getParentOp();
    while (parent) {
      if (isa<SectionCubeOp, SectionVectorOp>(parent)) {
        status = op->emitError("nested pto.section.cube/vector is not allowed");
        return WalkResult::interrupt();
      }
      parent = parent->getParentOp();
    }
    return WalkResult::advance();
  });
  return status;
}

static void eraseUnusedSimtEntries(ModuleOp module) {
  SmallVector<ModuleOp> symbolTables{module};
  module.walk([&](ModuleOp nested) {
    if (nested != module) {
      symbolTables.push_back(nested);
    }
  });

  for (ModuleOp symbolTableModule : symbolTables) {
    SymbolTable symbolTable(symbolTableModule);
    SmallVector<func::FuncOp> deadEntries;
    for (func::FuncOp funcOp : symbolTableModule.getOps<func::FuncOp>()) {
      if (!funcOp->hasAttr(kPTOSimtEntryAttrName)) {
        continue;
      }
      auto uses = symbolTable.getSymbolUses(funcOp, symbolTableModule);
      if (uses && uses->empty()) {
        deadEntries.push_back(funcOp);
      }
    }
    for (func::FuncOp funcOp : deadEntries) {
      funcOp.erase();
    }
  }
}

static LogicalResult verifyExplicitKernelKindMatchesSections(ModuleOp module) {
  auto kindAttr = module->getAttrOfType<FunctionKernelKindAttr>(
      FunctionKernelKindAttr::name);
  if (!kindAttr) {
    return success();
  }
  bool expectsCube = kindAttr.getKernelKind() == FunctionKernelKind::Cube;
  LogicalResult status = success();
  module.walk([&](Operation *op) {
    if (failed(status)) {
      return WalkResult::interrupt();
    }
    bool isCube = isa<SectionCubeOp>(op);
    bool isVector = isa<SectionVectorOp>(op);
    if (!isCube && !isVector) {
      return WalkResult::advance();
    }
    if (isCube != expectsCube) {
      status = op->emitError(
          "conflicts with explicit pto.kernel_kind on its module");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return status;
}

static LogicalResult verifySectionSplitCandidatesUseSections(ModuleOp module) {
  LogicalResult status = success();
  module.walk([&](func::FuncOp funcOp) {
    if (failed(status) || !isSectionSplitCandidate(funcOp)) {
      return WalkResult::advance();
    }
    if (!hasAnySection(funcOp)) {
      status = funcOp.emitOpError(
          "must contain pto.section.cube or pto.section.vector in section "
          "input split by vpto-split-cv-module");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return status;
}

static void
eraseSectionSplitCandidatesWithoutSectionKind(ModuleOp module,
                                              FunctionKernelKind kind) {
  SmallVector<func::FuncOp> eraseFuncs;
  module.walk([&](func::FuncOp funcOp) {
    if (isSectionSplitCandidate(funcOp) && !hasSectionKind(funcOp, kind)) {
      eraseFuncs.push_back(funcOp);
    }
  });

  for (func::FuncOp funcOp : eraseFuncs) {
    funcOp.erase();
  }
}

static void replaceSectionWithBody(Operation *sectionOp) {
  Region &region = sectionOp->getRegion(0);
  Block &body = region.front();
  Block *parentBlock = sectionOp->getBlock();
  parentBlock->getOperations().splice(Block::iterator(sectionOp),
                                      body.getOperations());
  sectionOp->erase();
}

static void rewriteSectionsForKind(ModuleOp module, FunctionKernelKind kind) {
  SmallVector<Operation *> eraseSections;
  SmallVector<Operation *> inlineSections;
  module.walk([&](Operation *op) {
    if (kind == FunctionKernelKind::Cube) {
      if (isa<SectionVectorOp>(op)) {
        eraseSections.push_back(op);
      } else if (isa<SectionCubeOp>(op)) {
        inlineSections.push_back(op);
      }
    } else {
      if (isa<SectionCubeOp>(op)) {
        eraseSections.push_back(op);
      } else if (isa<SectionVectorOp>(op)) {
        inlineSections.push_back(op);
      }
    }
  });

  for (Operation *op : eraseSections) {
    op->erase();
  }
  for (Operation *op : inlineSections) {
    replaceSectionWithBody(op);
  }
  if (!eraseSections.empty() || !inlineSections.empty()) {
    eraseUnusedSimtEntries(module);
  }
}

static ModuleOp cloneModuleForKind(ModuleOp source, FunctionKernelKind kind,
                                   OpBuilder &builder) {
  auto cloned = cast<ModuleOp>(source->clone());
  cloned->setAttr(FunctionKernelKindAttr::name,
                  FunctionKernelKindAttr::get(cloned.getContext(), kind));
  eraseSectionSplitCandidatesWithoutSectionKind(cloned, kind);
  rewriteSectionsForKind(cloned, kind);
  builder.insert(cloned);
  return cloned;
}

static LogicalResult materializeExplicitKernelKindSections(ModuleOp module) {
  auto kindAttr = module->getAttrOfType<FunctionKernelKindAttr>(
      FunctionKernelKindAttr::name);
  if (!kindAttr) {
    return success();
  }
  if (failed(verifyNoNestedSections(module)) ||
      failed(verifyExplicitKernelKindMatchesSections(module))) {
    return failure();
  }
  rewriteSectionsForKind(module, kindAttr.getKernelKind());
  return success();
}

/// Group top-level functions carrying a per-func `pto.kernel_kind` attribute
/// into one child kernel submodule per distinct kind. The "sugar" input form
/// puts `pto.kernel_kind` directly on each kernel function instead of on a
/// surrounding kind-tagged child module. This step rewrites it into the
/// canonical container form so that `vpto-normalize-container` and the
/// downstream VPTO emitter only ever see kind-tagged child modules.
///
/// For each distinct kind, a child module is created carrying that
/// `pto.kernel_kind`. Every top-level function is cloned into each child
/// module, then functions whose per-func `pto.kernel_kind` does not match the
/// child's kind are erased. Functions without any `pto.kernel_kind` (helper
/// functions) are duplicated into every child module, matching the design in
/// docs/designs/vpto-section-sugar.md constraint 5. The per-func
/// `pto.kernel_kind` attribute is intentionally kept on the cloned functions:
/// although the child module already carries the kind, verifiers such as
/// getEnclosingFunctionKernelKind only read the enclosing func::FuncOp, so
/// stripping it would reject legitimate pipe ops after the split.
static LogicalResult splitPerFuncKernelKind(ModuleOp module) {
  if (!hasKernelKindTopLevelFunc(module)) {
    return success();
  }

  // Collect the distinct kernel kinds requested by top-level functions.
  SmallVector<FunctionKernelKind, 2> kinds;
  for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
    auto kindAttr = funcOp->getAttrOfType<FunctionKernelKindAttr>(
        FunctionKernelKindAttr::name);
    if (!kindAttr) {
      continue;
    }
    FunctionKernelKind kind = kindAttr.getKernelKind();
    if (llvm::find(kinds, kind) == kinds.end()) {
      kinds.push_back(kind);
    }
  }
  if (kinds.empty()) {
    return success();
  }

  // Preserve outer module attributes (e.g. pto.target_arch, pto.backend) and
  // drop the per-func kernel_kind tags from the source once we have cloned the
  // variants. Build the child modules inside the existing outer module body.
  //
  // All clones are produced from the original module state BEFORE any child
  // module is inserted. Otherwise each successive clone would also pick up the
  // child modules inserted by earlier iterations, causing nested duplication.
  SmallVector<ModuleOp> clones;
  clones.reserve(kinds.size());
  for (FunctionKernelKind kind : kinds) {
    auto cloned = cast<ModuleOp>(module->clone());
    cloned->setAttr(FunctionKernelKindAttr::name,
                    FunctionKernelKindAttr::get(cloned.getContext(), kind));

    SmallVector<func::FuncOp> eraseFuncs;
    for (func::FuncOp funcOp : cloned.getOps<func::FuncOp>()) {
      auto funcKindAttr = funcOp->getAttrOfType<FunctionKernelKindAttr>(
          FunctionKernelKindAttr::name);
      if (!funcKindAttr) {
        // Helper function: keep a copy in every kernel submodule.
        continue;
      }
      if (funcKindAttr.getKernelKind() != kind) {
        eraseFuncs.push_back(funcOp);
        continue;
      }
      // Keep the per-func pto.kernel_kind tag on the cloned function. The
      // child module also carries the kind, but several verifiers (see
      // getEnclosingFunctionKernelKind in PTO.cpp) only consult the enclosing
      // func::FuncOp, never the parent ModuleOp. Stripping the tag here would
      // make legitimate pipe ops (tpush/tpop/aic_initialize_pipe/...) be
      // rejected as "not inside a kernel_kind function" after the split.
    }
    for (func::FuncOp funcOp : eraseFuncs) {
      funcOp.erase();
    }
    clones.push_back(cloned);
  }

  // Remove the original top-level functions now that each has been cloned into
  // its matching kernel submodule.
  SmallVector<Operation *> originalTopLevel;
  for (Operation &op : module.getBodyRegion().front().getOperations()) {
    if (isa<func::FuncOp>(op)) {
      originalTopLevel.push_back(&op);
    }
  }
  for (Operation *op : originalTopLevel) {
    op->erase();
  }

  // Now that the original functions are gone, insert the kind-tagged child
  // modules into the outer container body.
  OpBuilder builder(module.getBody(), module.getBody()->end());
  for (ModuleOp cloned : clones) {
    builder.insert(cloned);
  }
  return success();
}

static LogicalResult splitCVModule(ModuleOp module) {
  flattenSingleUnpartitionedChild(module);
  if (hasKernelKind(module)) {
    return materializeExplicitKernelKindSections(module);
  }
  if (hasKernelKindChildModule(module)) {
    for (ModuleOp child : module.getOps<ModuleOp>()) {
      if (!hasKernelKind(child)) {
        continue;
      }
      if (failed(materializeExplicitKernelKindSections(child))) {
        return failure();
      }
    }
    return success();
  }
  // Per-func `pto.kernel_kind` sugar: each kernel function declares its own
  // kind. Group functions into kind-tagged child submodules before the
  // section-sugar path, which handles a different (section-based) input form.
  if (hasKernelKindTopLevelFunc(module)) {
    return splitPerFuncKernelKind(module);
  }
  if (!hasCVSections(module)) {
    return success();
  }
  if (failed(verifyNoNestedSections(module))) {
    return failure();
  }
  if (failed(verifySectionSplitCandidatesUseSections(module))) {
    return failure();
  }
  bool needVector = hasSectionKind(module, FunctionKernelKind::Vector);
  bool needCube = hasSectionKind(module, FunctionKernelKind::Cube);
  if (!needVector && !needCube) {
    return success();
  }

  SmallVector<NamedAttribute> outerAttrs;
  outerAttrs.reserve(module->getAttrs().size());
  for (NamedAttribute attr : module->getAttrs()) {
    if (attr.getName() != SymbolTable::getSymbolAttrName()) {
      outerAttrs.push_back(attr);
    }
  }

  auto outer = ModuleOp::create(module.getLoc());
  outer->setAttrs(DictionaryAttr::get(module.getContext(), outerAttrs));
  OpBuilder builder(outer.getBody(), outer.getBody()->end());
  if (needVector) {
    cloneModuleForKind(module, FunctionKernelKind::Vector, builder);
  }
  if (needCube) {
    cloneModuleForKind(module, FunctionKernelKind::Cube, builder);
  }

  module.getBodyRegion().takeBody(outer.getBodyRegion());
  module->setAttrs(outer->getAttrs());
  return success();
}

struct VPTOSplitCVModulePass
    : public mlir::pto::impl::VPTOSplitCVModuleBase<VPTOSplitCVModulePass> {
  void runOnOperation() override {
    if (failed(splitCVModule(getOperation()))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVPTOSplitCVModulePass() {
  return std::make_unique<VPTOSplitCVModulePass>();
}

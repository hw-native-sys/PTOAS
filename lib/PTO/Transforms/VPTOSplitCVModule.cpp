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

static bool isSectionSplitCandidate(func::FuncOp funcOp);

static bool hasCVSections(ModuleOp module) {
  bool found = false;
  module.walk([&](func::FuncOp funcOp) {
    if (found || !isSectionSplitCandidate(funcOp))
      return WalkResult::advance();
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
    if (found || !isSectionSplitCandidate(funcOp))
      return WalkResult::advance();
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
    if (failed(status) || !isa<SectionCubeOp, SectionVectorOp>(op))
      return WalkResult::advance();
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

static LogicalResult verifySectionSplitCandidatesUseSections(ModuleOp module) {
  LogicalResult status = success();
  module.walk([&](func::FuncOp funcOp) {
    if (failed(status) || !isSectionSplitCandidate(funcOp))
      return WalkResult::advance();
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

static LogicalResult verifyUniqueSectionKindsPerFunction(ModuleOp module) {
  LogicalResult status = success();
  module.walk([&](func::FuncOp funcOp) {
    if (failed(status) || !isSectionSplitCandidate(funcOp))
      return WalkResult::advance();
    unsigned cubeCount = 0;
    unsigned vectorCount = 0;
    funcOp.walk([&](Operation *op) {
      if (isa<SectionCubeOp>(op))
        ++cubeCount;
      if (isa<SectionVectorOp>(op))
        ++vectorCount;
    });
    if (cubeCount > 1) {
      status = funcOp.emitOpError("contains more than one pto.section.cube");
      return WalkResult::interrupt();
    }
    if (vectorCount > 1) {
      status = funcOp.emitOpError("contains more than one pto.section.vector");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return status;
}

static void eraseSectionSplitCandidatesWithoutSectionKind(ModuleOp module,
                                                          FunctionKernelKind kind) {
  SmallVector<func::FuncOp> eraseFuncs;
  module.walk([&](func::FuncOp funcOp) {
    if (isSectionSplitCandidate(funcOp) && !hasSectionKind(funcOp, kind))
      eraseFuncs.push_back(funcOp);
  });

  for (func::FuncOp funcOp : eraseFuncs)
    funcOp.erase();
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
      if (isa<SectionVectorOp>(op))
        eraseSections.push_back(op);
      else if (isa<SectionCubeOp>(op))
        inlineSections.push_back(op);
    } else {
      if (isa<SectionCubeOp>(op))
        eraseSections.push_back(op);
      else if (isa<SectionVectorOp>(op))
        inlineSections.push_back(op);
    }
  });

  for (Operation *op : eraseSections)
    op->erase();
  for (Operation *op : inlineSections)
    replaceSectionWithBody(op);
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

static LogicalResult splitCVModule(ModuleOp module) {
  if (auto kernelKind = module->getAttrOfType<FunctionKernelKindAttr>(
          FunctionKernelKindAttr::name)) {
    rewriteSectionsForKind(module, kernelKind.getKernelKind());
    return success();
  }
  if (hasKernelKindChildModule(module))
    return success();
  if (!hasCVSections(module))
    return success();
  if (failed(verifyNoNestedSections(module)))
    return failure();
  if (failed(verifySectionSplitCandidatesUseSections(module)))
    return failure();
  if (failed(verifyUniqueSectionKindsPerFunction(module)))
    return failure();

  bool needVector = hasSectionKind(module, FunctionKernelKind::Vector);
  bool needCube = hasSectionKind(module, FunctionKernelKind::Cube);
  if (!needVector && !needCube)
    return success();

  SmallVector<NamedAttribute> outerAttrs;
  outerAttrs.reserve(module->getAttrs().size());
  for (NamedAttribute attr : module->getAttrs())
    if (attr.getName() != SymbolTable::getSymbolAttrName())
      outerAttrs.push_back(attr);

  auto outer = ModuleOp::create(module.getLoc());
  outer->setAttrs(DictionaryAttr::get(module.getContext(), outerAttrs));
  OpBuilder builder(outer.getBody(), outer.getBody()->end());
  if (needVector)
    cloneModuleForKind(module, FunctionKernelKind::Vector, builder);
  if (needCube)
    cloneModuleForKind(module, FunctionKernelKind::Cube, builder);

  module.getBodyRegion().takeBody(outer.getBodyRegion());
  module->setAttrs(outer->getAttrs());
  return success();
}

struct VPTOSplitCVModulePass
    : public mlir::pto::impl::VPTOSplitCVModuleBase<VPTOSplitCVModulePass> {
  void runOnOperation() override {
    if (failed(splitCVModule(getOperation())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVPTOSplitCVModulePass() {
  return std::make_unique<VPTOSplitCVModulePass>();
}

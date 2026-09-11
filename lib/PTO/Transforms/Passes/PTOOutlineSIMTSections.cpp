// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOOutlineSIMTSections.cpp ----------------------------------------===//
//
// Outline inline pto.section.simt regions into pto.simt_entry helpers.
//
// PTODSL can keep SIMT code inline so PTOAS can still see values allocated in
// the outer kernel scope before materializing persistent SIMT fragments.  This
// pass is the late outline step that restores the VPTO backend's existing
// simt_entry + simt_launch form.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOOUTLINESIMTSECTIONS
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

static bool isDefinedInside(Operation *scope, Value value) {
  if (Operation *defOp = value.getDefiningOp()) {
    return scope->isAncestor(defOp);
  }

  auto blockArg = dyn_cast<BlockArgument>(value);
  if (!blockArg) {
    return false;
  }

  Operation *owner = blockArg.getOwner()->getParentOp();
  return owner && scope->isAncestor(owner);
}

static LogicalResult collectCaptures(pto::SectionSimtOp sectionOp,
                                     SmallVectorImpl<Value> &captures) {
  llvm::DenseSet<Value> seen;
  Operation *scope = sectionOp.getOperation();

  sectionOp.getBody().walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      if (isDefinedInside(scope, operand)) {
        continue;
      }
      if (Operation *defOp = operand.getDefiningOp()) {
        if (defOp->hasTrait<OpTrait::ConstantLike>()) {
          continue;
        }
      }
      if (seen.insert(operand).second) {
        captures.push_back(operand);
      }
    }
  });

  return success();
}

static void cloneExternalConstants(pto::SectionSimtOp sectionOp,
                                   OpBuilder &builder, IRMapping &mapping) {
  llvm::DenseSet<Operation *> seen;
  SmallVector<Operation *> constants;
  Operation *scope = sectionOp.getOperation();

  sectionOp.getBody().walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      if (!defOp || isDefinedInside(scope, operand) ||
          !defOp->hasTrait<OpTrait::ConstantLike>()) {
        continue;
      }
      if (seen.insert(defOp).second) {
        constants.push_back(defOp);
      }
    }
  });

  for (Operation *constant : constants) {
    builder.clone(*constant, mapping);
  }
}

static LogicalResult verifySectionCanBeOutlined(pto::SectionSimtOp sectionOp) {
  func::FuncOp parentFunc = sectionOp->getParentOfType<func::FuncOp>();
  if (!parentFunc) {
    return sectionOp.emitOpError("must be nested in a func.func");
  }

  if (parentFunc->hasAttr(pto::kPTOSimtEntryAttrName)) {
    return sectionOp.emitOpError()
           << "must not be nested in a function marked with '"
           << pto::kPTOSimtEntryAttrName << "'";
  }

  if (!sectionOp.getBody().hasOneBlock()) {
    return sectionOp.emitOpError("requires a single-block body");
  }

  if (sectionOp.getBody().front().getNumArguments() != 0) {
    return sectionOp.emitOpError("does not support region block arguments");
  }

  bool hasNestedSection = false;
  sectionOp.getBody().walk([&](pto::SectionSimtOp nested) {
    if (nested != sectionOp) {
      hasNestedSection = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (hasNestedSection) {
    return sectionOp.emitOpError("does not support nested pto.section.simt");
  }

  Operation *scope = sectionOp.getOperation();
  WalkResult escapeCheck = sectionOp.getBody().walk([&](Operation *op) {
    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (!scope->isAncestor(user)) {
          return WalkResult::interrupt();
        }
      }
    }
    return WalkResult::advance();
  });
  if (escapeCheck.wasInterrupted()) {
    return sectionOp.emitOpError(
        "does not support values defined in pto.section.simt escaping the "
        "section");
  }

  return success();
}

static std::string getUniqueHelperName(ModuleOp module, func::FuncOp parentFunc,
                                       unsigned &outlineIndex) {
  std::string parentName = parentFunc.getSymName().str();
  do {
    std::string candidate =
        (Twine(parentName) + "_simt_" + Twine(outlineIndex++)).str();
    if (!module.lookupSymbol<func::FuncOp>(candidate)) {
      return candidate;
    }
  } while (true);
}

static FailureOr<int64_t> getSimtThreadCount(pto::SectionSimtOp sectionOp) {
  constexpr int64_t kMaxSimtThreads = 2048;
  const int64_t dimX = sectionOp.getDimX();
  const int64_t dimY = sectionOp.getDimY();
  const int64_t dimZ = sectionOp.getDimZ();

  int64_t threadCount = 1;
  for (int64_t dimension : {dimX, dimY, dimZ}) {
    if (dimension <= 0 || threadCount > kMaxSimtThreads / dimension) {
      sectionOp.emitOpError()
          << "SIMT launch dimensions must have a positive product no greater "
             "than "
          << kMaxSimtThreads << ", got (" << dimX << ", " << dimY << ", "
          << dimZ << ")";
      return failure();
    }
    threadCount *= dimension;
  }
  return threadCount;
}

static func::FuncOp createOutlinedHelper(ModuleOp module,
                                         pto::SectionSimtOp sectionOp,
                                         StringRef helperName,
                                         int64_t maxThreads,
                                         ValueRange captures) {
  MLIRContext *ctx = module.getContext();
  Location loc = sectionOp.getLoc();

  SmallVector<Type> argTypes;
  argTypes.reserve(captures.size());
  for (Value capture : captures) {
    argTypes.push_back(capture.getType());
  }

  OpBuilder moduleBuilder(module.getBodyRegion());
  moduleBuilder.setInsertionPointToEnd(&module.getBodyRegion().front());
  auto helper = moduleBuilder.create<func::FuncOp>(
      loc, helperName, FunctionType::get(ctx, argTypes, TypeRange{}));
  helper.setPrivate();
  helper->setAttr(pto::kPTOSimtEntryAttrName, UnitAttr::get(ctx));

  // The outlined helper is compiled independently from the outer launch. Keep
  // its static lane count so the LLVM emitter can derive the correct SIMT
  // register budget instead of falling back to the 1024-thread default.
  helper->setAttr(pto::kPTOSimtMaxThreadsAttrName,
                  moduleBuilder.getI32IntegerAttr(maxThreads));

  Block *entry = helper.addEntryBlock();
  IRMapping mapping;
  for (auto [capture, arg] : llvm::zip(captures, entry->getArguments())) {
    mapping.map(capture, arg);
  }

  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(entry);
  cloneExternalConstants(sectionOp, bodyBuilder, mapping);
  for (Operation &op : sectionOp.getBody().front()) {
    bodyBuilder.clone(op, mapping);
  }
  bodyBuilder.create<func::ReturnOp>(loc);

  return helper;
}

static void replaceSectionWithLaunch(pto::SectionSimtOp sectionOp,
                                     StringRef helperName,
                                     ValueRange captures) {
  Location loc = sectionOp.getLoc();
  OpBuilder builder(sectionOp);
  Value dimX = builder.create<arith::ConstantIntOp>(loc, sectionOp.getDimX(),
                                                    /*width=*/32);
  Value dimY = builder.create<arith::ConstantIntOp>(loc, sectionOp.getDimY(),
                                                    /*width=*/32);
  Value dimZ = builder.create<arith::ConstantIntOp>(loc, sectionOp.getDimZ(),
                                                    /*width=*/32);
  builder.create<pto::SimtLaunchOp>(loc, helperName, dimX, dimY, dimZ,
                                    captures);
  sectionOp.erase();
}

static LogicalResult outlineSection(ModuleOp module,
                                    pto::SectionSimtOp sectionOp,
                                    unsigned &outlineIndex) {
  if (failed(verifySectionCanBeOutlined(sectionOp))) {
    return failure();
  }

  FailureOr<int64_t> maxThreads = getSimtThreadCount(sectionOp);
  if (failed(maxThreads)) {
    return failure();
  }

  SmallVector<Value> captures;
  if (failed(collectCaptures(sectionOp, captures))) {
    return failure();
  }

  func::FuncOp parentFunc = sectionOp->getParentOfType<func::FuncOp>();
  std::string helperName =
      getUniqueHelperName(module, parentFunc, outlineIndex);
  createOutlinedHelper(module, sectionOp, helperName, *maxThreads, captures);
  replaceSectionWithLaunch(sectionOp, helperName, captures);
  return success();
}

struct PTOOutlineSIMTSectionsPass
    : public pto::impl::PTOOutlineSIMTSectionsBase<PTOOutlineSIMTSectionsPass> {
  using pto::impl::PTOOutlineSIMTSectionsBase<
      PTOOutlineSIMTSectionsPass>::PTOOutlineSIMTSectionsBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<pto::SectionSimtOp> sections;
    module.walk(
        [&](pto::SectionSimtOp sectionOp) { sections.push_back(sectionOp); });

    unsigned outlineIndex = 0;
    for (pto::SectionSimtOp sectionOp : sections) {
      if (failed(outlineSection(module, sectionOp, outlineIndex))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOOutlineSIMTSectionsPass() {
  return std::make_unique<PTOOutlineSIMTSectionsPass>();
}

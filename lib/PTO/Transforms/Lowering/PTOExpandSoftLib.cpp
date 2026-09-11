// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOExpandSoftLib.cpp - Target soft-library materialization ----------===//
//
// Soft operation bodies live in lib/SoftOps and are authored in PTODSL.  This
// pass only selects and materializes the body at the final VPTO boundary; the
// normal PTOInlineLibCall pass expands the temporary call immediately after
// this pass.  Keeping the implementation out of this C++ file is important:
// the same library mechanism can later host other software ops.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Support/CodeConstants.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/SoftLibService.h"
#include "Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOEXPANDSOFTLIB
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

static constexpr llvm::StringLiteral kSoftLibInstanceAttr =
    "pto.softlib.instance";

// Element widths the A5 Software Library implements for pto.vdiv.  Signed and
// signless forms are both accepted; unsigned vectors are intentionally not.
static bool isSoftLibVdivIntegerVReg(Type type) {
  auto vreg = dyn_cast<VRegType>(type);
  if (!vreg) {
    return false;
  }
  auto integer = dyn_cast<IntegerType>(vreg.getElementType());
  if (!integer) {
    return false;
  }
  return (integer.getWidth() == mlir::pto::kValue16 ||
          integer.getWidth() == mlir::pto::kValue32) &&
         (integer.isSigned() || integer.isSignless());
}

// Integer-element pto.vdiv on A5 must be handled here (SoftOps or an explicit
// compile-time rejection).  Floating-point vdiv is native and is left alone.
static bool isIntegerVdiv(Operation *op) {
  auto vdiv = dyn_cast<VdivOp>(op);
  if (!vdiv) {
    return false;
  }
  auto vreg = dyn_cast<VRegType>(vdiv.getResult().getType());
  if (!vreg) {
    return false;
  }
  return isa<IntegerType>(vreg.getElementType());
}

static std::string getIntegerDtype(VdivOp op) {
  auto vreg = cast<VRegType>(op.getLhs().getType());
  auto integer = cast<IntegerType>(vreg.getElementType());
  return (integer.isSigned() ? "si" : "i") + std::to_string(integer.getWidth());
}

static std::string buildVdivRequestJson(VdivOp op) {
  auto vreg = cast<VRegType>(op.getLhs().getType());
  std::string json = "{\"dtype\":\"" + getIntegerDtype(op) + "\",\"lanes\":";
  json += std::to_string(vreg.getElementCount());
  json += ",\"mask\":\"";
  json += cast<MaskType>(op.getMask().getType()).getGranularity().str();
  json += "\"}";
  return json;
}

static std::string uniqueSoftLibName(ModuleOp module, StringRef base) {
  std::string stem = std::string("__pto_soft_") + base.str();
  unsigned suffix = 0;
  std::string name = stem;
  while (module.lookupSymbol<func::FuncOp>(name)) {
    name = stem + "_" + std::to_string(++suffix);
  }
  return name;
}

struct PTOExpandSoftLibPass
    : public mlir::pto::impl::PTOExpandSoftLibBase<PTOExpandSoftLibPass> {
  using PTOExpandSoftLibBase::PTOExpandSoftLibBase;

  llvm::DenseMap<Operation *, llvm::StringMap<func::FuncOp>> materialized;

  // Replaces `op` with a call to the previously materialized `callee` and
  // erases the original operation. Shared by the cache-hit and
  // freshly-materialized paths.
  void replaceWithSoftLibCall(Operation *op, func::FuncOp callee,
                              ValueRange operands, Value resultValue) {
    OpBuilder builder(op);
    auto call = builder.create<func::CallOp>(op->getLoc(), callee, operands);
    resultValue.replaceAllUsesWith(call.getResult(0));
    op->erase();
  }

  // Fast path: reuses a function previously materialized for the same
  // (target, op, operand specs) key within this module. Returns the callee
  // on hit, std::nullopt on miss.
  std::optional<func::FuncOp>
  findCachedSoftLibFunc(ModuleOp module, const std::string &cacheKey) {
    auto &moduleCache = materialized[module.getOperation()];
    auto cached = moduleCache.find(cacheKey);
    if (cached != moduleCache.end() && cached->second &&
        cached->second->getParentOp() == module.getOperation()) {
      return cached->second;
    }
    return std::nullopt;
  }

  // SoftLib import callback for materializeCall: clones every function from
  // the materialized `source` module into `module` under collision-free
  // names, renames all symbol uses, and reports the imported entry point
  // through `importedEntry`. Rejects a source that does not belong to
  // `context`, lacks the requested entry, or collides with existing names.
  LogicalResult
  importSoftLibFunctions(ModuleOp module, MLIRContext &context,
                         StringRef functionName, StringRef entrySymbol,
                         ModuleOp source,
                         func::FuncOp &importedEntry) {
    bool materializeSourceReady =
        source && source.getContext() == &context;
    if (!materializeSourceReady) {
      return failure();
    }
    func::FuncOp sourceEntry = source.lookupSymbol<func::FuncOp>(entrySymbol);
    if (!sourceEntry) {
      return failure();
    }

    SymbolTable symbols(module);
    SmallVector<func::FuncOp> sourceFunctions;
    for (func::FuncOp fn : source.getOps<func::FuncOp>()) {
      sourceFunctions.push_back(fn);
    }

    llvm::StringMap<std::string> renames;
    if (failed(planSoftLibRenames(symbols, sourceFunctions, sourceEntry,
                                  functionName, renames))) {
      return failure();
    }

    OpBuilder builder(&context);
    builder.setInsertionPointToEnd(module.getBody());
    SmallVector<func::FuncOp> cloned;
    for (func::FuncOp fn : sourceFunctions) {
      auto copy = cast<func::FuncOp>(builder.clone(*fn));
      copy.setName(renames.lookup(fn.getSymName()));
      copy.setVisibility(SymbolTable::Visibility::Private);
      copy->setAttr(kSoftLibInstanceAttr, UnitAttr::get(&context));
      cloned.push_back(copy);
    }
    if (failed(applySoftLibRenames(context, cloned, renames))) {
      return failure();
    }
    importedEntry = module.lookupSymbol<func::FuncOp>(functionName);
    return importedEntry ? success() : failure();
  }

  // Plans the collision-free import name for every source function: the
  // entry keeps `functionName`, the rest are namespaced under it.
  static LogicalResult
  planSoftLibRenames(const SymbolTable &symbols,
                     ArrayRef<func::FuncOp> sourceFunctions,
                     func::FuncOp sourceEntry, StringRef functionName,
                     llvm::StringMap<std::string> &renames) {
    for (func::FuncOp fn : sourceFunctions) {
      std::string name = fn == sourceEntry
                             ? functionName.str()
                             : functionName.str() + "__" + fn.getSymName().str();
      if (symbols.lookup(name)) {
        return failure();
      }
      renames[fn.getSymName()] = name;
    }
    return success();
  }

  // Rewrites every symbol use inside the cloned functions to the planned
  // import names.
  static LogicalResult
  applySoftLibRenames(MLIRContext &context, ArrayRef<func::FuncOp> cloned,
                      const llvm::StringMap<std::string> &renames) {
    for (func::FuncOp fn : cloned) {
      for (const auto &rename : renames) {
        if (failed(SymbolTable::replaceAllSymbolUses(
                StringAttr::get(&context, rename.getKey()),
                StringAttr::get(&context, rename.getValue()), fn))) {
          return failure();
        }
      }
    }
    return success();
  }

  LogicalResult materializeCall(
      Operation *op, ModuleOp module, MLIRContext &context, StringRef target,
      StringRef requestOp, StringRef requestSpecs, StringRef functionStem,
      ValueRange operands, Value resultValue,
      const std::shared_ptr<SoftLibService> &service) {
    SoftLibMaterializationRequest request;
    request.target = target.str();
    request.op = requestOp.str();
    request.operandSpecsJson = requestSpecs.str();
    auto &moduleCache = materialized[module.getOperation()];

    std::string cacheKey = target.str() + ":" + request.op + ":" +
                           request.operandSpecsJson;
    if (auto cached = findCachedSoftLibFunc(module, cacheKey)) {
      replaceWithSoftLibCall(op, *cached, operands, resultValue);
      return success();
    }

    std::string functionName = uniqueSoftLibName(module, functionStem);
    func::FuncOp importedEntry;
    LogicalResult materializationResult = service->materialize(
        request, context, [&](ModuleOp source, StringRef entrySymbol) {
          return importSoftLibFunctions(module, context, functionName,
                                        entrySymbol, source, importedEntry);
        });
    bool softLibReady = succeeded(materializationResult) && importedEntry;
    if (!softLibReady) {
      return op->emitError() << "failed to materialize SoftOps implementation for "
                             << requestOp,
             failure();
    }

    replaceWithSoftLibCall(op, importedEntry, operands, resultValue);
    moduleCache[cacheKey] = importedEntry;
    return success();
  }

  LogicalResult materializeTrig(Operation *op, ModuleOp module,
                                MLIRContext &context, StringRef target,
                                StringRef opName,
                                const std::shared_ptr<SoftLibService> &service) {
    return materializeCall(op, module, context, target, opName, "{}",
                           (opName == "pto.sin" ? "sin_f32" : "cos_f32"),
                           op->getOperands(), op->getResult(0), service);
  }

  LogicalResult materializeVdiv(VdivOp op, ModuleOp module,
                                MLIRContext &context, StringRef target,
                                const std::shared_ptr<SoftLibService> &service) {
    auto vreg = cast<VRegType>(op.getLhs().getType());
    std::string stem = "vdiv_" + getIntegerDtype(op) + "_" +
                       std::to_string(vreg.getElementCount());
    return materializeCall(op, module, context, target, "pto.vdiv",
                           buildVdivRequestJson(op), stem,
                           ValueRange{op.getLhs(), op.getRhs(), op.getMask()},
                           op.getResult(), service);
  }

  // Validates one integer pto.vdiv against the A5 Software Library contract
  // (i16 vectors with b16 masks, i32 with b32; all operands same type).
  bool checkVdivAgainstSoftLibContract(VdivOp vdiv) {
    auto resultVreg = dyn_cast<VRegType>(vdiv.getResult().getType());
    if (!resultVreg) {
      vdiv.emitError("A5 pto.vdiv requires a vector result");
      return false;
    }
    auto integer = dyn_cast<IntegerType>(resultVreg.getElementType());
    auto expectedMask = integer && integer.getWidth() == 16 ? "b16" : "b32";
    bool lhsLegal = isSoftLibVdivIntegerVReg(vdiv.getLhs().getType());
    bool rhsLegal = isSoftLibVdivIntegerVReg(vdiv.getRhs().getType());
    bool sameType = vdiv.getLhs().getType() == vdiv.getRhs().getType() &&
                    vdiv.getLhs().getType() == vdiv.getResult().getType();
    bool maskMatches =
        vdiv.getMask().getType() == MaskType::get(&getContext(), expectedMask);
    if (lhsLegal && rhsLegal && sameType && maskMatches) {
      return true;
    }
    vdiv.emitError() << "A5 integer pto.vdiv is not supported for "
                     << vdiv.getResult().getType() << " with mask "
                     << vdiv.getMask().getType() << "; only signed or "
                     << "signless i16 vectors with a b16 mask and i32 "
                     << "vectors with a b32 mask are materialized through "
                     << "the A5 Software Library, and f16/f32 pto.vdiv "
                     << "uses the native vector instruction";
    return false;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    ModuleOp topLevel = getTopLevelModuleOp(module);
    if (!isTargetArchA5(topLevel)) {
      return;
    }
    StringRef targetArch = "a5";
    if (auto attr = topLevel->getAttrOfType<StringAttr>("pto.target_arch")) {
      targetArch = attr.getValue();
    }
    SmallVector<Operation *> candidates;
    module.walk([&](Operation *op) {
      bool isSoftLibCandidate = isa<SinOp, CosOp>(op) || isIntegerVdiv(op);
      if (isSoftLibCandidate) {
        candidates.push_back(op);
      }
    });
    if (candidates.empty()) {
      return;
    }
    auto service = SoftLibRuntime::getService();
    if (!service) {
      module.emitError("PTOExpandSoftLib requires an initialized PTODSL SoftLib runtime");
      signalPassFailure();
      return;
    }
    for (Operation *op : candidates) {
      if (auto vdiv = dyn_cast<VdivOp>(op)) {
        if (!checkVdivAgainstSoftLibContract(vdiv)) {
          signalPassFailure();
          continue;
        }
        if (failed(materializeVdiv(vdiv, module, getContext(), targetArch,
                                   service))) {
          signalPassFailure();
        }
        continue;
      }
      if (!op->getResult(0).getType().isF32() || !op->getOperand(0).getType().isF32()) {
        op->emitError("A5 SoftOps pto.sin/pto.cos require f32 scalar operands");
        signalPassFailure();
        continue;
      }
      StringRef opName = isa<SinOp>(op) ? "pto.sin" : "pto.cos";
      if (failed(materializeTrig(op, module, getContext(), targetArch, opName,
                                 service))) {
        signalPassFailure();
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOExpandSoftLibPass() {
  return std::make_unique<PTOExpandSoftLibPass>();
}

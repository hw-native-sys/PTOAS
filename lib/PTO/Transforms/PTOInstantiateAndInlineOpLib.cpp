// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "PTOLowerToOpLibCalls.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOINLINEBACKENDHELPERS
#define GEN_PASS_DEF_PTOINLINELIBCALL
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

static constexpr llvm::StringLiteral kOpLibAttrInstVariantId =
    "pto.oplib.instance.variant_id";
static constexpr llvm::StringLiteral kOpLibAttrInstOp = "pto.oplib.instance.op";
static constexpr llvm::StringLiteral kOpLibAttrInstDType =
    "pto.oplib.instance.dtype";
static constexpr llvm::StringLiteral kErrInstanceBodyMissing =
    "E_OPLIB_INSTANCE_BODY_MISSING";

static bool isInstanceFunc(func::FuncOp fn) {
  return fn->hasAttr(kOpLibAttrInstVariantId);
}

static bool isTilelangInlineProcFunc(func::FuncOp fn) {
  return fn->hasAttr("pto.tilelang.inline_proc");
}

static bool isPTODSLSubkernelHelperFunc(func::FuncOp fn) {
  return pto::hasPTODSLSubkernelHelperMarker(fn);
}

static bool isTilelangTemplateFunc(func::FuncOp fn) {
  return fn->hasAttr("pto.tilelang.instance") && fn.isPrivate();
}

static bool isInlineableBackendHelperFunc(func::FuncOp fn) {
  return isPTODSLSubkernelHelperFunc(fn);
}

static bool isInlineableLibFunc(func::FuncOp fn) {
  // Keep OP-Lib behavior unchanged while TileLang private template helpers are
  // still handled on the VPTO tile-op expansion path, together with
  // TileLang inline_proc helpers that only become meaningful after ExpandTileOp.
  if (isInstanceFunc(fn) || isTilelangInlineProcFunc(fn))
    return true;
  return isTilelangTemplateFunc(fn);
}

static Value maybeUnwrapCastToExpected(Value operand, Type expectedType) {
  if (operand.getType() == expectedType)
    return operand;

  auto cast = operand.getDefiningOp<UnrealizedConversionCastOp>();
  if (!cast || cast->getNumOperands() != 1 || cast->getNumResults() != 1)
    return operand;

  if (cast.getOperand(0).getType() == expectedType)
    return cast.getOperand(0);
  return operand;
}

static Operation *cloneOpForInlineWithFix(OpBuilder &builder, Operation &op,
                                          IRMapping &mapping) {
  if (auto alloc = dyn_cast<pto::AllocTileOp>(&op)) {
    auto mapOperand = [&](Value operand, Type expectedType) -> Value {
      if (!operand)
        return Value();
      Value mapped = mapping.lookupOrNull(operand);
      if (!mapped)
        mapped = operand;
      return maybeUnwrapCastToExpected(mapped, expectedType);
    };

    Value mappedAddr = mapOperand(
        alloc.getAddr(), alloc.getAddr() ? alloc.getAddr().getType() : Type());
    Value mappedValidRow = mapOperand(
        alloc.getValidRow(),
        alloc.getValidRow() ? alloc.getValidRow().getType() : Type());
    Value mappedValidCol = mapOperand(
        alloc.getValidCol(),
        alloc.getValidCol() ? alloc.getValidCol().getType() : Type());

    auto cloned = builder.create<pto::AllocTileOp>(
        alloc.getLoc(), alloc.getType(), mappedAddr, mappedValidRow,
        mappedValidCol);
    cloned->setAttrs(alloc->getAttrs());
    return cloned.getOperation();
  }

  return builder.clone(op, mapping);
}

static void eraseDeadBridgeCasts(func::FuncOp func) {
  bool changed = true;
  while (changed) {
    changed = false;

    SmallVector<UnrealizedConversionCastOp, 8> deadUnrealized;
    func.walk([&](UnrealizedConversionCastOp cast) {
      if (cast->use_empty())
        deadUnrealized.push_back(cast);
    });

    SmallVector<memref::CastOp, 8> deadMemrefCasts;
    func.walk([&](memref::CastOp cast) {
      if (cast->use_empty())
        deadMemrefCasts.push_back(cast);
    });

    if (deadUnrealized.empty() && deadMemrefCasts.empty())
      break;

    for (UnrealizedConversionCastOp cast : llvm::reverse(deadUnrealized))
      cast.erase();
    for (memref::CastOp cast : llvm::reverse(deadMemrefCasts))
      cast.erase();
    changed = true;
  }
}

static LogicalResult inlineCall(func::CallOp call, func::FuncOp callee) {
  if (callee.isExternal())
    return call.emitOpError("callee must have a body before inlining");

  Block &entry = callee.getBody().front();
  if (entry.getNumArguments() != call.getNumOperands())
    return call.emitOpError("callee argument count mismatch during inlining");
  auto returnOp = dyn_cast<func::ReturnOp>(entry.getTerminator());
  if (!returnOp)
    return call.emitOpError("callee must terminate with func.return");
  if (returnOp.getNumOperands() != call.getNumResults())
    return call.emitOpError("callee return/result arity mismatch during inlining");

  OpBuilder builder(call);
  IRMapping mapping;
  for (auto [arg, operand] :
       llvm::zip(entry.getArguments(), call.getOperands()))
    mapping.map(arg, operand);

  for (Operation &op : entry.without_terminator()) {
    FailureOr<bool> handledOr =
        pto::tryCloneOpLibInlineBridgeOp(builder, op, mapping);
    if (failed(handledOr))
      return call.emitOpError("failed to remap OP-Lib inline bridge op");
    if (*handledOr)
      continue;

    Operation *newOp = cloneOpForInlineWithFix(builder, op, mapping);
    for (auto [oldRes, newRes] :
         llvm::zip(op.getResults(), newOp->getResults()))
      mapping.map(oldRes, newRes);
  }

  for (auto [callResult, returnOperand] :
       llvm::zip(call.getResults(), returnOp.getOperands())) {
    Value mapped = mapping.lookupOrNull(returnOperand);
    if (!mapped)
      mapped = returnOperand;
    callResult.replaceAllUsesWith(mapped);
  }

  call.erase();
  return success();
}

static void emitMissingInstanceBodyError(func::CallOp call, func::FuncOp callee) {
  call.emitError() << kErrInstanceBodyMissing
                   << ": OP-Lib instance body is missing for @"
                   << callee.getSymName();
  if (auto variant =
          callee->getAttrOfType<StringAttr>(kOpLibAttrInstVariantId)) {
    call.emitRemark() << "variant_id=" << variant.getValue();
  }
  if (auto op = callee->getAttrOfType<StringAttr>(kOpLibAttrInstOp)) {
    call.emitRemark() << "op=" << op.getValue();
  }
  if (auto dtype = callee->getAttrOfType<StringAttr>(kOpLibAttrInstDType)) {
    call.emitRemark() << "dtype=" << dtype.getValue();
  }
}

static SmallVector<ModuleOp, 4> collectFuncModules(ModuleOp root) {
  SmallVector<ModuleOp, 4> modules;
  modules.push_back(root);
  root.walk([&](ModuleOp nested) {
    if (nested != root)
      modules.push_back(nested);
  });
  return modules;
}

template <typename InlinePredicate>
static LogicalResult validateInlineableCalleesHaveBodies(
    ModuleOp module, InlinePredicate &&shouldInline) {
  for (ModuleOp funcModule : collectFuncModules(module)) {
    for (func::FuncOp func : funcModule.getOps<func::FuncOp>()) {
      if (func.isExternal() || func.empty())
        continue;

      bool failed = false;
      func.walk([&](func::CallOp call) {
        auto calleeAttr = call.getCalleeAttr();
        if (!calleeAttr)
          return;

        func::FuncOp callee =
            funcModule.lookupSymbol<func::FuncOp>(calleeAttr.getValue());
        if (!callee || !shouldInline(callee) || !callee.isExternal())
          return;

        emitMissingInstanceBodyError(call, callee);
        failed = true;
      });
      if (failed)
        return failure();
    }
  }

  return success();
}

template <typename InlinePredicate>
static LogicalResult inlineMatchingCalls(
    ModuleOp module, InlinePredicate &&shouldInline, bool debug,
    llvm::StringRef debugTag, int &inlinedCalls, int &touchedFuncs) {
  for (ModuleOp funcModule : collectFuncModules(module)) {
    for (func::FuncOp func : funcModule.getOps<func::FuncOp>()) {
      if (func.isExternal())
        continue;
      if (isInstanceFunc(func))
        continue;
      if (func.empty())
        continue;

      bool changedThisFunc = false;
      bool madeProgress = true;
      while (madeProgress) {
        madeProgress = false;

        SmallVector<func::CallOp, 16> calls;
        func.walk([&](func::CallOp call) { calls.push_back(call); });

        for (func::CallOp oldCall : calls) {
          if (!oldCall || !oldCall->getBlock())
            continue;

          auto calleeAttr = oldCall.getCalleeAttr();
          if (!calleeAttr)
            continue;

          func::FuncOp callee =
              funcModule.lookupSymbol<func::FuncOp>(calleeAttr.getValue());
          if (!callee || !shouldInline(callee))
            continue;

          if (callee.isExternal()) {
            oldCall.emitOpError("callee must have a body before inlining");
            return failure();
          }

          func::CallOp call = oldCall;
          SmallVector<Value, 4> concreteOperands;
          concreteOperands.reserve(call.getNumOperands());
          for (auto [operand, expectedTy] :
               llvm::zip(call.getOperands(),
                         callee.getFunctionType().getInputs())) {
            concreteOperands.push_back(
                maybeUnwrapCastToExpected(operand, expectedTy));
          }

          OpBuilder builder(call);
          auto newCall = builder.create<func::CallOp>(call.getLoc(), callee,
                                                      concreteOperands);
          if (call.getNumResults() != newCall.getNumResults()) {
            call.emitOpError("call result arity mismatch during inline staging");
            return failure();
          }
          for (auto [oldResult, newResult] :
               llvm::zip(call.getResults(), newCall.getResults()))
            oldResult.replaceAllUsesWith(newResult);
          call.erase();

          if (failed(inlineCall(newCall, callee)))
            return failure();

          ++inlinedCalls;
          changedThisFunc = true;
          madeProgress = true;
          if (debug) {
            llvm::errs() << debugTag << ": inlined @" << callee.getSymName()
                         << " into @" << func.getSymName() << "\n";
          }
        }
      }

      if (changedThisFunc) {
        eraseDeadBridgeCasts(func);
        ++touchedFuncs;
      }
    }
  }

  return success();
}

template <typename FuncPredicate>
static void eraseDeadMatchingPrivateFuncs(ModuleOp module,
                                          FuncPredicate &&predicate) {
  for (ModuleOp funcModule : collectFuncModules(module)) {
    SymbolTable symbolTable(funcModule);
    SmallVector<func::FuncOp, 8> deadFuncs;
    for (func::FuncOp func : funcModule.getOps<func::FuncOp>()) {
      if (!predicate(func))
        continue;
      if (func.isPublic())
        continue;
      auto uses = symbolTable.getSymbolUses(func, funcModule);
      if (uses && uses->empty())
        deadFuncs.push_back(func);
    }
    for (func::FuncOp func : deadFuncs)
      func.erase();
  }
}

struct PTOInlineBackendHelpersPass
    : public pto::impl::PTOInlineBackendHelpersBase<
          PTOInlineBackendHelpersPass> {
  using pto::impl::PTOInlineBackendHelpersBase<
      PTOInlineBackendHelpersPass>::PTOInlineBackendHelpersBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    int inlinedCalls = 0;
    int touchedFuncs = 0;
    if (failed(inlineMatchingCalls(module, isInlineableBackendHelperFunc, debug,
                                   "[op-fusion] inline-backend-helpers",
                                   inlinedCalls, touchedFuncs))) {
      signalPassFailure();
      return;
    }

    if (debug) {
      llvm::errs() << "[op-fusion] inline-backend-helpers touched "
                   << touchedFuncs << " function(s), inlined " << inlinedCalls
                   << " call(s)\n";
    }

    eraseDeadMatchingPrivateFuncs(module, isInlineableBackendHelperFunc);
  }
};

struct PTOInlineLibCallPass
    : public pto::impl::PTOInlineLibCallBase<PTOInlineLibCallPass> {
  using pto::impl::PTOInlineLibCallBase<
      PTOInlineLibCallPass>::PTOInlineLibCallBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    int inlinedCalls = 0;
    int touchedFuncs = 0;
    if (failed(validateInlineableCalleesHaveBodies(module, isInlineableLibFunc))) {
      signalPassFailure();
      return;
    }

    if (failed(inlineMatchingCalls(module, isInlineableLibFunc, debug,
                                   "[op-fusion] inline-libcall", inlinedCalls,
                                   touchedFuncs))) {
      signalPassFailure();
      return;
    }

    if (debug) {
      llvm::errs() << "[op-fusion] inline-libcall touched " << touchedFuncs
                   << " function(s), inlined " << inlinedCalls << " call(s)\n";
    }

    // Drop now-dead inline-able callees (private + uncalled) so downstream
    // backends never see leftover template/instance bodies.  This is needed
    // for TileLang templates whose tile_buf-typed parameters cannot be
    // legalized once their callers have been inlined.
    eraseDeadMatchingPrivateFuncs(module, isInlineableLibFunc);
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::pto::createPTOInlineLibCallPass(const PTOInlineLibCallOptions &options) {
  return std::make_unique<PTOInlineLibCallPass>(options);
}

std::unique_ptr<Pass> mlir::pto::createPTOInlineBackendHelpersPass(
    const PTOInlineBackendHelpersOptions &options) {
  return std::make_unique<PTOInlineBackendHelpersPass>(options);
}

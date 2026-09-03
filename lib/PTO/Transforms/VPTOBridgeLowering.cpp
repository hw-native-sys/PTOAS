// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- VPTOBridgeLowering.cpp - generic C++ interface bridge lowering ----===//
//===----------------------------------------------------------------------===//
//
// Generic bridge lowering pass of the VPTO C++ interface bridge. It validates
// resolved logical entries against the compiler-owned registry and
// mechanically lowers them into calls to concrete wrapper instances.
// bridge_object_create exclusively owns stateful object materialization:
// registry-bound size query, aligned stack storage, and void initialization.
//
// The route policy is also the routing check of last resort: any op still
// present in the IR that the whitelist routes to a wrapper entry was
// missed by its family pass, and is rejected with a diagnostic instead of
// silently flowing into the regular LLVM emission path.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/VPTOBridgeRegistry.h"
#include "PTO/Transforms/VPTOBridgeWhitelist.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

namespace mlir {
namespace pto {

#define GEN_PASS_DECL_VPTOBRIDGELOWERING
#define GEN_PASS_DEF_VPTOBRIDGELOWERING
#include "PTO/Transforms/Passes.h.inc"

namespace {

/// Converts the carrier types a bridge op may hold. These rules mirror the
/// PipeType/PtrType entries of the VPTO type converter
/// (VPTOCANN900LLVMEmitter.cpp convertVPTOType); the bridge lowering runs
/// before that converter and must agree with it so values flow into the
/// remaining PTO ops without extra casts.
class BridgeTypeConverter final : public TypeConverter {
public:
  explicit BridgeTypeConverter(MLIRContext *context) {
    addConversion([](Type type) -> Type {
      if (isa<pto::PipeType>(type)) {
        return LLVM::LLVMPointerType::get(type.getContext());
      }
      if (auto ptrType = dyn_cast<pto::PtrType>(type)) {
        return LLVM::LLVMPointerType::get(
            type.getContext(),
            static_cast<unsigned>(ptrType.getMemorySpace().getAddressSpace()));
      }
      return type;
    });
    addSourceMaterialization(materializeBridgeCast);
    addTargetMaterialization(materializeBridgeCast);
  }

private:
  static std::optional<Value> materializeBridgeCast(OpBuilder &builder,
                                                    Type resultType,
                                                    ValueRange inputs,
                                                    Location loc) {
    if (inputs.size() != 1) {
      return std::nullopt;
    }
    return builder
        .create<UnrealizedConversionCastOp>(loc, TypeRange{resultType}, inputs)
        .getResult(0);
  }
};

struct BridgeLoweringState {
  llvm::StringSet<> declaredEntries;
};

/// Creates the module-level private declaration of a wrapper entry the
/// first time it is called.
static LogicalResult
ensureWrapperDecl(ModuleOp module, BridgeLoweringState &state,
                  PatternRewriter &rewriter, Operation *anchor,
                  StringRef callee, TypeRange argTypes, TypeRange resultTypes) {
  FunctionType expected =
      FunctionType::get(module.getContext(), argTypes, resultTypes);
  auto existing = module.lookupSymbol<func::FuncOp>(callee);
  if (existing && existing.getFunctionType() != expected) {
    return anchor->emitError()
           << "VPTO bridge wrapper '" << callee << "' already has type "
           << existing.getFunctionType() << ", requested " << expected;
  }
  if (existing) {
    state.declaredEntries.insert(callee);
    return success();
  }
  state.declaredEntries.insert(callee);
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&module.getBodyRegion().front());
  auto decl = rewriter.create<func::FuncOp>(module.getLoc(), callee, expected);
  decl.setPrivate();
  return success();
}

/// Validates the fully assembled call argument list against the whitelist
/// ABI. Emits a diagnostic and returns failure on any mismatch.
static bool bridgeValueKindMatches(BridgeValueKind kind, Type type) {
  switch (kind) {
  case BridgeValueKind::I32:
    return type.isInteger(32);
  case BridgeValueKind::I64:
    return type.isInteger(64);
  case BridgeValueKind::Pointer:
    return isa<LLVM::LLVMPointerType, pto::PtrType>(type);
  case BridgeValueKind::PipeObject:
    return isa<LLVM::LLVMPointerType, pto::PipeType>(type);
  }
  return false;
}

static LogicalResult validateRegistryAbi(Operation *op,
                                         const BridgeFunctionDesc &desc,
                                         ValueRange callArgs) {
  if (callArgs.size() != desc.arguments.size()) {
    return op->emitError() << "VPTO bridge call to registry entry '"
                           << stringifyBridgeEntryId(desc.id) << "' passes "
                           << callArgs.size()
                           << " argument(s), registry ABI declares "
                           << desc.arguments.size();
  }
  for (auto [index, arg] : llvm::enumerate(callArgs)) {
    if (!bridgeValueKindMatches(desc.arguments[index], arg.getType())) {
      return op->emitError()
             << "VPTO bridge call to registry entry '"
             << stringifyBridgeEntryId(desc.id) << "' argument #" << index
             << " has type " << arg.getType()
             << ", which does not match the registry ABI";
    }
  }
  return success();
}

static LogicalResult validateRegistryResults(Operation *op,
                                             const BridgeFunctionDesc &desc,
                                             TypeRange resultTypes) {
  if (resultTypes.size() != desc.results.size()) {
    return op->emitError() << "VPTO bridge call to registry entry '"
                           << stringifyBridgeEntryId(desc.id) << "' declares "
                           << resultTypes.size()
                           << " result(s), registry ABI declares "
                           << desc.results.size();
  }
  for (auto [index, type] : llvm::enumerate(resultTypes)) {
    if (!bridgeValueKindMatches(desc.results[index], type)) {
      return op->emitError() << "VPTO bridge call to registry entry '"
                             << stringifyBridgeEntryId(desc.id) << "' result #"
                             << index << " has type " << type
                             << ", which does not match the registry ABI";
    }
  }
  return success();
}

static bool isResolvedSymbolForEntry(StringRef symbol,
                                     const BridgeFunctionDesc &desc) {
  if (symbol == desc.symbolBase) {
    return true;
  }
  return symbol.starts_with(desc.symbolBase) &&
         symbol.drop_front(desc.symbolBase.size()).starts_with("__");
}

class LowerBridgeObjectCreatePattern final
    : public OpConversionPattern<BridgeObjectCreateOp> {
public:
  LowerBridgeObjectCreatePattern(TypeConverter &converter, MLIRContext *context,
                                 BridgeLoweringState &state)
      : OpConversionPattern<BridgeObjectCreateOp>(converter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(BridgeObjectCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef entryId = op.getEntryAttr().getValue();
    const BridgeFunctionDesc *desc = findBridgeFunctionById(entryId);
    StringRef symbol =
        op.getCalleeAttr() ? op.getCalleeAttr().getValue() : StringRef();
    if (!desc || !desc->createsObject || symbol.empty()) {
      return op.emitError() << "bridge object requires a resolved registered "
                               "entry and callee: "
                            << entryId;
    }
    if (!isResolvedSymbolForEntry(symbol, *desc)) {
      return op.emitError()
             << "resolved callee '" << symbol
             << "' does not belong to bridge entry '" << entryId << "'";
    }
    if (desc->arguments.size() != adaptor.getArgs().size() ||
        desc->results.size() != 1 ||
        desc->results.front() != BridgeValueKind::PipeObject) {
      return op.emitError()
             << "bridge object operands/results do not match registry entry "
             << entryId;
    }
    if (failed(validateRegistryAbi(op, *desc, adaptor.getArgs()))) {
      return failure();
    }
    const BridgeFunctionDesc *sizeDesc =
        findBridgeFunction(BridgeEntryId::PipeSize);
    if (!sizeDesc) {
      return op.emitError("bridge registry has no object size entry");
    }
    StringRef sizeSymbol = sizeDesc->symbolBase;
    if (op.getSizeCalleeAttr()) {
      sizeSymbol = op.getSizeCalleeAttr().getValue();
    }
    if (!isResolvedSymbolForEntry(sizeSymbol, *sizeDesc)) {
      return op.emitError() << "resolved size callee '" << sizeSymbol
                            << "' does not belong to bridge entry 'pipe.size'";
    }
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op.getLoc();
    Value size = rewriter
                     .create<func::CallOp>(loc, sizeSymbol,
                                           rewriter.getI64Type(), ValueRange{})
                     .getResult(0);
    if (failed(ensureWrapperDecl(module, state, rewriter, op, sizeSymbol, {},
                                 {rewriter.getI64Type()}))) {
      return failure();
    }
    Value storage = rewriter.create<LLVM::AllocaOp>(
        loc, LLVM::LLVMPointerType::get(rewriter.getContext()),
        rewriter.getI8Type(), size, desc->objectAlignment);
    SmallVector<Value> args{storage};
    args.append(adaptor.getArgs().begin(), adaptor.getArgs().end());
    rewriter.create<func::CallOp>(loc, symbol, TypeRange{}, args);
    if (failed(ensureWrapperDecl(
            module, state, rewriter, op, symbol,
            llvm::map_to_vector<4>(args,
                                   [](Value arg) { return arg.getType(); }),
            {}))) {
      return failure();
    }
    rewriter.replaceOp(op, storage);
    return success();
  }

private:
  BridgeLoweringState &state;
};

class LowerBridgeCallPattern final : public OpConversionPattern<BridgeCallOp> {
public:
  LowerBridgeCallPattern(TypeConverter &converter, MLIRContext *context,
                         BridgeLoweringState &state)
      : OpConversionPattern<BridgeCallOp>(converter, context), state(state) {}

  LogicalResult
  matchAndRewrite(BridgeCallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    StringRef entryId = op.getEntryAttr().getValue();
    auto calleeAttr = op.getCalleeAttr();
    if (!calleeAttr) {
      return op.emitError() << "bridge call '" << entryId
                            << "' has no resolved concrete callee";
    }
    StringRef callee = calleeAttr.getValue();
    const BridgeFunctionDesc *registryDesc = findBridgeFunctionById(entryId);
    if (!registryDesc) {
      return op.emitError() << "VPTO bridge entry '" << entryId
                            << "' has no registered ABI entry";
    }
    if (!isResolvedSymbolForEntry(callee, *registryDesc)) {
      return op.emitError()
             << "resolved callee '" << callee
             << "' does not belong to bridge entry '" << entryId << "'";
    }
    ModuleOp module = op->getParentOfType<ModuleOp>();
    ValueRange callArgs = adaptor.getArgs();
    if (failed(validateRegistryAbi(op, *registryDesc, callArgs))) {
      return failure();
    }
    SmallVector<Type> resultTypes;
    for (Type resultType : op.getResultTypes()) {
      Type converted = getTypeConverter()->convertType(resultType);
      if (!converted) {
        return op.emitError() << "VPTO bridge call result type " << resultType
                              << " has no bridge conversion";
      }
      resultTypes.push_back(converted);
    }
    if (failed(validateRegistryResults(op, *registryDesc,
                                       TypeRange(resultTypes)))) {
      return failure();
    }

    func::CallOp call = rewriter.create<func::CallOp>(
        loc, callee, TypeRange(resultTypes), ValueRange(callArgs));
    if (failed(ensureWrapperDecl(
            module, state, rewriter, op, callee,
            llvm::map_to_vector<4>(callArgs,
                                   [](Value arg) { return arg.getType(); }),
            TypeRange(resultTypes)))) {
      return failure();
    }

    if (call.getNumResults() == 0) {
      rewriter.eraseOp(op);
      return success();
    }
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  BridgeLoweringState &state;
};

class LowerBridgeIntToPtrPattern final
    : public OpConversionPattern<BridgeIntToPtrOp> {
public:
  LowerBridgeIntToPtrPattern(TypeConverter &converter, MLIRContext *context)
      : OpConversionPattern<BridgeIntToPtrOp>(converter, context) {}

  LogicalResult
  matchAndRewrite(BridgeIntToPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type convertedResult =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!convertedResult || !isa<LLVM::LLVMPointerType>(convertedResult)) {
      return op.emitError()
             << "VPTO bridge inttoptr requires a result type that converts "
                "to an LLVM pointer, got "
             << op.getResult().getType();
    }
    rewriter.replaceOpWithNewOp<LLVM::IntToPtrOp>(op, convertedResult,
                                                  adaptor.getAddr());
    return success();
  }
};

struct VPTOBridgeLoweringPass final
    : public impl::VPTOBridgeLoweringBase<VPTOBridgeLoweringPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VPTOBridgeLoweringPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool hasBridgeOps = false;
    module.walk([&](Operation *op) {
      if (isa<BridgeCallOp, BridgeObjectCreateOp, BridgeIntToPtrOp>(op)) {
        hasBridgeOps = true;
      }
    });

    // Routing policy is only used to reject operations that should have
    // been claimed by a family/typed lowering pass.
    FailureOr<BridgeRoutePolicy> policyOr =
        loadBridgeRoutePolicy(whitelistPath, llvm::errs());
    if (failed(policyOr)) {
      signalPassFailure();
      return;
    }
    bool leftoversFound = false;
    module.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      StringRef family = name == "pto.tpush" || name == "pto.tpop" ||
                                 name == "pto.tfree" ||
                                 name == "pto.initialize_l2l_pipe"
                             ? "pipe"
                             : "cube";
      bool routed = family == "pipe" ? policyOr->routesFamily("pipe")
                                     : policyOr->routesOp("cube", name);
      if (!routed ||
          isa<BridgeCallOp, BridgeObjectCreateOp, BridgeIntToPtrOp>(op)) {
        return;
      }
      op->emitError()
          << "VPTO bridge: '" << name
          << "' is routed by policy but was not lowered into a bridge op";
      leftoversFound = true;
    });
    if (leftoversFound) {
      signalPassFailure();
      return;
    }

    if (!hasBridgeOps) {
      return;
    }

    BridgeTypeConverter converter(&getContext());
    ConversionTarget target(getContext());
    target.addIllegalOp<BridgeCallOp, BridgeObjectCreateOp, BridgeIntToPtrOp>();
    // Everything the patterns create (func.call, llvm.alloca, private
    // declarations) must be legal on the target, otherwise the conversion
    // driver rejects the generated operations and rolls the pattern back.
    target.markUnknownOpDynamicallyLegal([](Operation *op) { return true; });

    RewritePatternSet patterns(&getContext());
    BridgeLoweringState state{};
    patterns.add<LowerBridgeCallPattern>(converter, &getContext(), state);
    patterns.add<LowerBridgeObjectCreatePattern>(converter, &getContext(),
                                                 state);
    patterns.add<LowerBridgeIntToPtrPattern>(converter, &getContext());
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createVPTOBridgeLoweringPass() {
  return std::make_unique<VPTOBridgeLoweringPass>();
}

} // namespace pto
} // namespace mlir

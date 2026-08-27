// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOBridgeLowering.cpp - generic C++ interface bridge lowering ----===//
//===----------------------------------------------------------------------===//
//
// Generic bridge lowering pass of the VPTO C++ interface bridge. It knows
// nothing about individual PTO-ISA interface families: it validates each
// pto.bridge_call against the bridge whitelist and mechanically lowers it
// into a call to the wrapper entry, materializing the wrapper declaration
// at module level. Entries that carry `storage_size_callee` additionally
// synthesize the stateful-object pattern (size query + stack storage) so
// family passes can express "construct a template object on the kernel
// stack" without emitting LLVM dialect ops themselves.
//
// The whitelist is also the routing check of last resort: any op still
// present in the IR that the whitelist routes to a wrapper entry was
// missed by its family pass, and is rejected with a diagnostic instead of
// silently flowing into the regular LLVM emission path.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
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
  const BridgeWhitelist &whitelist;
  llvm::StringSet<> declaredEntries;
};

/// Creates the module-level private declaration of a wrapper entry the
/// first time it is called.
static void ensureWrapperDecl(ModuleOp module, BridgeLoweringState &state,
                              PatternRewriter &rewriter, StringRef callee,
                              TypeRange argTypes, TypeRange resultTypes) {
  if (state.declaredEntries.contains(callee)) {
    return;
  }
  state.declaredEntries.insert(callee);
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&module.getBodyRegion().front());
  auto decl = rewriter.create<func::FuncOp>(
      module.getLoc(), callee,
      FunctionType::get(module.getContext(), argTypes, resultTypes));
  decl.setPrivate();
}

/// Validates the fully assembled call argument list against the whitelist
/// ABI. Emits a diagnostic and returns failure on any mismatch.
static LogicalResult validateAbi(Operation *op, const BridgeWhitelistEntry &entry,
                                 ValueRange callArgs) {
  if (callArgs.size() != entry.abi.size()) {
    return op->emitError()
           << "VPTO bridge call to '" << entry.entry << "' passes "
           << callArgs.size() << " argument(s), whitelist ABI declares "
           << entry.abi.size();
  }
  for (auto [index, arg] : llvm::enumerate(callArgs)) {
    const BridgeAbiArg &abiArg = entry.abi[index];
    if (!bridgeAbiTypeMatches(abiArg.type, arg.getType())) {
      return op->emitError()
             << "VPTO bridge call to '" << entry.entry << "' argument #"
             << index << " has type " << arg.getType()
             << ", whitelist ABI declares '" << abiArg.type << "'";
    }
  }
  return success();
}

class LowerBridgeCallPattern final : public OpConversionPattern<BridgeCallOp> {
public:
  LowerBridgeCallPattern(TypeConverter &converter, MLIRContext *context,
                         BridgeLoweringState &state)
      : OpConversionPattern<BridgeCallOp>(converter, context), state(state) {}

  LogicalResult
  matchAndRewrite(BridgeCallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    StringRef callee = op.getCalleeAttr().getValue();
    const BridgeWhitelistEntry *entry = state.whitelist.findEntry(callee);
    if (!entry) {
      return op.emitError()
             << "VPTO bridge call to wrapper entry '" << callee
             << "' is not declared in the bridge whitelist";
    }
    ModuleOp module = op->getParentOfType<ModuleOp>();
    ValueRange operands = adaptor.getArgs();

    // Stateful entries synthesize their own storage: query the wrapper for
    // the object size and alloca it on the kernel stack. The storage value
    // replaces the bridge call result, which must be the only result.
    bool hasStorage = op.getStorageSizeCalleeAttr() != nullptr;
    SmallVector<Value> callArgs;
    Value storage;
    if (hasStorage) {
      if (op.getNumResults() != 1) {
        return op.emitError()
               << "VPTO bridge call with storage_size_callee must have "
                  "exactly one result (the storage handle)";
      }
      StringRef sizeCallee = op.getStorageSizeCalleeAttr().getValue();
      if (!state.whitelist.findEntry(sizeCallee)) {
        return op.emitError()
               << "VPTO bridge storage size callee '" << sizeCallee
               << "' is not declared in the bridge whitelist";
      }
      Value size = rewriter.create<func::CallOp>(loc, sizeCallee,
                                                 rewriter.getI64Type(),
                                                 ValueRange{})
                       .getResult(0);
      ensureWrapperDecl(module, state, rewriter, sizeCallee, /*argTypes=*/{},
                        /*resultTypes=*/{rewriter.getI64Type()});
      storage = rewriter.create<LLVM::AllocaOp>(
          loc, LLVM::LLVMPointerType::get(rewriter.getContext()),
          rewriter.getI8Type(), size, /*alignment=*/8);
      callArgs.push_back(storage);
    }
    callArgs.append(operands.begin(), operands.end());

    if (failed(validateAbi(op, *entry, callArgs))) {
      return failure();
    }

    SmallVector<Type> resultTypes;
    for (Type resultType : op.getResultTypes()) {
      Type converted = getTypeConverter()->convertType(resultType);
      if (!converted) {
        return op.emitError()
               << "VPTO bridge call result type " << resultType
               << " has no bridge conversion";
      }
      resultTypes.push_back(converted);
    }

    func::CallOp call = rewriter.create<func::CallOp>(
        loc, callee, TypeRange(resultTypes), ValueRange(callArgs));
    ensureWrapperDecl(module, state, rewriter, callee,
                      llvm::map_to_vector<4>(callArgs,
                                             [](Value arg) { return arg.getType(); }),
                      TypeRange(resultTypes));

    if (hasStorage) {
      rewriter.replaceOp(op, storage);
      return success();
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
    Type convertedResult = getTypeConverter()->convertType(op.getResult().getType());
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
      if (isa<BridgeCallOp, BridgeIntToPtrOp>(op)) {
        hasBridgeOps = true;
      }
    });

    // The whitelist always resolves through the formal chain (pass option,
    // PTOAS_VPTO_BRIDGE_WHITELIST, built-in default), so this pass always
    // validates; `whitelistName` is only for diagnostics.
    std::string whitelistName;
    FailureOr<BridgeWhitelist> whitelistOr =
        loadBridgeWhitelist(whitelistPath, llvm::errs(), &whitelistName);
    if (failed(whitelistOr)) {
      signalPassFailure();
      return;
    }
    BridgeWhitelist whitelist = std::move(*whitelistOr);

    // Routing check: an op the whitelist routes to a wrapper entry must
    // have been rewritten into bridge ops by the pass owning its lowering
    // channel. Leftovers mean that pass was skipped or missed the op;
    // reject them here instead of letting them flow into the regular
    // emission path. The diagnostic names the channel so the reader knows
    // which pass to look at.
    llvm::StringMap<const BridgeWhitelistEntry *> routedOps;
    for (const BridgeWhitelistEntry &entry : whitelist.bridgeOps) {
      if (entry.op != BridgeWhitelist::kInternalOp) {
        routedOps[entry.op] = &entry;
      }
    }
    bool leftoversFound = false;
    module.walk([&](Operation *op) {
      auto it = routedOps.find(op->getName().getStringRef());
      if (it == routedOps.end()) {
        return;
      }
      op->emitError()
          << "VPTO bridge: '" << it->first()
          << "' is routed to wrapper entry '" << it->second->entry
          << "' by the bridge whitelist '" << whitelistName
          << "' but was not lowered into a pto.bridge_call by "
          << (it->second->isDeclarative()
                  ? "the declarative bridge lowering"
                  : "its family pass");
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
    target.addIllegalOp<BridgeCallOp, BridgeIntToPtrOp>();
    // Everything the patterns create (func.call, llvm.alloca, private
    // declarations) must be legal on the target, otherwise the conversion
    // driver rejects the generated operations and rolls the pattern back.
    target.markUnknownOpDynamicallyLegal(
        [](Operation *op) { return true; });

    RewritePatternSet patterns(&getContext());
    BridgeLoweringState state{whitelist};
    patterns.add<LowerBridgeCallPattern>(converter, &getContext(), state);
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

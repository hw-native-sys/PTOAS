// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOLowerDeclarativeBridgeOps.cpp - typed Cube bridge lowering -----===//
//===----------------------------------------------------------------------===//
//
// The external policy only selects registered Cube ops. Typed adapters own
// PTO op semantics, the registry owns wrapper ABI ordering, and this pass
// materializes structured bridge calls without YAML operand indices.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/VPTOBridgeRegistry.h"
#include "PTO/Transforms/VPTOBridgeWhitelist.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace pto {

#define GEN_PASS_DECL_PTOLOWERDECLARATIVEBRIDGEOPS
#define GEN_PASS_DEF_PTOLOWERDECLARATIVEBRIDGEOPS
#include "PTO/Transforms/Passes.h.inc"

namespace {

static std::string canonicalBridgeInstanceKey(llvm::StringRef entryId,
                                              DictionaryAttr spec) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << entryId << "|";
  spec.print(os);
  os.flush();
  return text;
}

static DictionaryAttr buildStructuredTileSpec(OpBuilder &builder,
                                              TileBufType tileType) {
  SmallVector<NamedAttribute> fields = {
      builder.getNamedAttr("element_type",
                           TypeAttr::get(tileType.getElementType())),
      builder.getNamedAttr("shape",
                           builder.getDenseI64ArrayAttr(tileType.getShape())),
      builder.getNamedAttr(
          "valid_shape",
          builder.getDenseI64ArrayAttr(tileType.getValidShape())),
      builder.getNamedAttr(
          "b_layout", builder.getI32IntegerAttr(tileType.getBLayoutValueI32())),
      builder.getNamedAttr(
          "s_layout", builder.getI32IntegerAttr(tileType.getSLayoutValueI32())),
      builder.getNamedAttr(
          "s_fractal",
          builder.getI32IntegerAttr(tileType.getSFractalSizeI32()))};
  if (Attribute memorySpace = tileType.getMemorySpace()) {
    fields.push_back(builder.getNamedAttr("memory_space", memorySpace));
  }
  return DictionaryAttr::get(builder.getContext(), fields);
}

struct TMatmulBridgeAdapter {
  static Value getTile(TMatmulOp op, llvm::StringRef role) {
    if (role == "result_tile") {
      return op.getDst();
    }
    if (role == "left_tile") {
      return op.getLhs();
    }
    if (role == "right_tile") {
      return op.getRhs();
    }
    return nullptr;
  }
  static Attribute getAccPhase(TMatmulOp op) { return op.getAccPhaseAttr(); }
  static Value getResult(TMatmulOp op) {
    return op->getNumResults() == 1 ? op.getResult() : Value{};
  }
};

struct TGemvBridgeAdapter {
  static Value getTile(TGemvOp op, llvm::StringRef role) {
    if (role == "result_tile") {
      return op.getDst();
    }
    if (role == "left_tile") {
      return op.getLhs();
    }
    if (role == "right_tile") {
      return op.getRhs();
    }
    return nullptr;
  }
  static Attribute getAccPhase(TGemvOp op) { return op.getAccPhaseAttr(); }
  static Value getResult(TGemvOp op) {
    return op->getNumResults() == 1 ? op.getResult() : Value{};
  }
};

template <typename OpTy, typename Adapter>
class LowerDirectBridgeOp final : public OpRewritePattern<OpTy> {
public:
  LowerDirectBridgeOp(MLIRContext *context, const BridgeFunctionDesc &desc)
      : OpRewritePattern<OpTy>(context), desc(desc) {}

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    if (desc.renderer != BridgeRendererKind::CubeDirect ||
        desc.bindings.size() != desc.arguments.size()) {
      return op.emitError("Cube bridge registry entry is not a direct ABI");
    }

    SmallVector<Value> callArgs;
    SmallVector<NamedAttribute> structuredSpec;
    SmallVector<AllocTileOp> allocs;
    for (const BridgeAbiBinding &binding : desc.bindings) {
      Value tile = Adapter::getTile(op, binding.role);
      auto tileType = tile ? dyn_cast<TileBufType>(tile.getType()) : nullptr;
      if (!tileType) {
        return op.emitError() << "Cube bridge role '" << binding.role
                              << "' must be a tile_buf";
      }
      auto alloc = tile.template getDefiningOp<AllocTileOp>();
      if (!alloc || !alloc.getAddr()) {
        return op.emitError() << "Cube bridge role '" << binding.role
                              << "' must come from alloc_tile with a planned "
                                 "address";
      }
      callArgs.push_back(alloc.getAddr());
      structuredSpec.push_back(rewriter.getNamedAttr(
          binding.role, buildStructuredTileSpec(rewriter, tileType)));
      allocs.push_back(alloc);
    }
    structuredSpec.push_back(
        rewriter.getNamedAttr("acc_phase", Adapter::getAccPhase(op)));
    DictionaryAttr spec =
        DictionaryAttr::get(rewriter.getContext(), structuredSpec);
    auto call = rewriter.create<BridgeCallOp>(
        op.getLoc(), TypeRange{}, desc.symbolBase, nullptr, callArgs);
    StringRef entryId = stringifyBridgeEntryId(desc.id);
    call->setAttr("entry_id", rewriter.getStringAttr(entryId));
    call->setAttr("spec",
                  BridgeCubeSpecAttr::get(rewriter.getContext(), spec));
    call->setAttr("instance_key", rewriter.getStringAttr(
                                      canonicalBridgeInstanceKey(entryId, spec)));
    if (Value result = Adapter::getResult(op)) {
      result.replaceAllUsesWith(Adapter::getTile(op, "result_tile"));
    }
    rewriter.eraseOp(op);
    for (AllocTileOp alloc : allocs) {
      if (alloc.use_empty()) {
        rewriter.eraseOp(alloc);
      }
    }
    return success();
  }

private:
  const BridgeFunctionDesc &desc;
};

struct PTOLowerDeclarativeBridgeOpsPass final
    : public impl::PTOLowerDeclarativeBridgeOpsBase<
          PTOLowerDeclarativeBridgeOpsPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTOLowerDeclarativeBridgeOpsPass)

  void runOnOperation() override {
    FailureOr<BridgeRoutePolicy> policyOr =
        loadBridgeRoutePolicy(whitelistPath, llvm::errs());
    if (failed(policyOr)) {
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(&getContext());
    bool hasPatterns = false;
    if (policyOr->routesOp("cube", "pto.tmatmul")) {
      const BridgeFunctionDesc *desc =
          findBridgeFunction(BridgeEntryId::CubeTMatmul);
      if (!desc) {
        getOperation().emitError("registry has no tmatmul bridge entry");
        signalPassFailure();
        return;
      }
      patterns.add<LowerDirectBridgeOp<TMatmulOp, TMatmulBridgeAdapter>>(
          &getContext(), *desc);
      hasPatterns = true;
    }
    if (policyOr->routesOp("cube", "pto.tgemv")) {
      const BridgeFunctionDesc *desc =
          findBridgeFunction(BridgeEntryId::CubeTgemv);
      if (!desc) {
        getOperation().emitError("registry has no tgemv bridge entry");
        signalPassFailure();
        return;
      }
      patterns.add<LowerDirectBridgeOp<TGemvOp, TGemvBridgeAdapter>>(
          &getContext(), *desc);
      hasPatterns = true;
    }
    if (!hasPatterns) {
      return;
    }
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createPTOLowerDeclarativeBridgeOpsPass() {
  return std::make_unique<PTOLowerDeclarativeBridgeOpsPass>();
}

} // namespace pto
} // namespace mlir

// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software; you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may obtain a copy of the License at
// https://www.huawei.com/
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE.
// See LICENSE for more details.

//===- PTOLowerMatmulFamilyOps.cpp - MATMUL family bridge lowering -------===//
//===----------------------------------------------------------------------===//
//
// MATMUL family pass of the VPTO C++ interface bridge. It understands the
// semantics of the tile-world matmul ops (tmatmul / tmatmul.acc) and
// rewrites them into generic pto.bridge_call ops that carry only the
// wrapper callee name and the planned i64 addresses of the operand tiles.
// Unlike the pipe family there is no storage lifecycle and no address
// rebinding: the tile type information is spread over the operand tile
// types, and all of it is collected here into the per-function bridge
// specialization. The generic bridge lowering pass only sees the resulting
// bridge ops.
//
// Routing is whitelist driven: the wrapper callee of every converted op is
// looked up in the bridge whitelist by IR op name, so this pass holds no
// hardcoded wrapper entry names. A matmul op that the whitelist does not
// route is left untouched and keeps flowing through the regular tile-op
// expansion path, matching the pre-bridge behavior.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/VPTOBridgeTokens.h"
#include "PTO/Transforms/VPTOBridgeWhitelist.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace pto {
namespace {

/// Returns the C++ template token of a non-Unspecified accumulation phase,
/// or std::nullopt when the call renders without a template argument.
static std::optional<std::string> buildAccPhaseToken(AccPhase phase) {
  switch (phase) {
  case AccPhase::Unspecified:
    return std::nullopt;
  case AccPhase::Partial:
    return std::string("pto::AccPhase::Partial");
  case AccPhase::Final:
    return std::string("pto::AccPhase::Final");
  }
  return std::nullopt;
}

struct PTOLowerMatmulFamilyOpsPass final
    : public PassWrapper<PTOLowerMatmulFamilyOpsPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTOLowerMatmulFamilyOpsPass)

  PTOLowerMatmulFamilyOpsPass() = default;
  PTOLowerMatmulFamilyOpsPass(const PTOLowerMatmulFamilyOpsPass &other)
      : PassWrapper(other) {
    copyOptionValuesFrom(&other);
  }

  Option<std::string> whitelistPath{
      *this, "whitelist-path", llvm::cl::init(""),
      llvm::cl::desc("Path to the VPTO bridge whitelist YAML; falls back to "
                     "the PTOAS_VPTO_BRIDGE_WHITELIST environment variable, "
                     "then to the built-in default whitelist")};

  llvm::StringRef getArgument() const final {
    return "pto-lower-matmul-family-ops";
  }

  llvm::StringRef getDescription() const final {
    return "Lower tile-world matmul ops into generic VPTO bridge ops";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // Collect first; rewriting during the walk would invalidate the walker.
    SmallVector<TMatmulOp> matmuls;
    SmallVector<TMatmulAccOp> matmulAccs;
    func.walk([&](Operation *op) {
      if (auto matmul = dyn_cast<TMatmulOp>(op)) {
        matmuls.push_back(matmul);
      } else if (auto matmulAcc = dyn_cast<TMatmulAccOp>(op)) {
        matmulAccs.push_back(matmulAcc);
      }
    });
    if (matmuls.empty() && matmulAccs.empty()) {
      return;
    }

    // Whitelist-driven routing. Unlike the pipe family, matmul ops have a
    // non-bridge VPTO lowering (tile-op expansion to pto.mad), so a missing
    // routing entry simply leaves the op on the regular path instead of
    // erroring. The whitelist always resolves through the formal chain
    // (pass option, PTOAS_VPTO_BRIDGE_WHITELIST, built-in default); kernels
    // that want the mad expansion route the op out of the whitelist with an
    // explicit whitelist file.
    FailureOr<BridgeWhitelist> whitelistOr =
        loadBridgeWhitelist(whitelistPath, llvm::errs());
    if (failed(whitelistOr)) {
      signalPassFailure();
      return;
    }
    const BridgeWhitelist &whitelist = *whitelistOr;

    bool hadError = false;
    // Wrapper specialization fields collected while lowering this function;
    // stored as a function attribute once lowering succeeds. The module-level
    // wrapper generation pass merges the per-function specs deterministically
    // (the family pass instances may run concurrently).
    SmallVector<std::pair<std::string, std::string>> specFields;
    // Tile handles consumed by bridged matmul ops; erased once use-empty.
    SmallVector<AllocTileOp> bridgedAllocs;

    // Resolves a tile operand to its planned address. The bridge wrapper
    // binds each tile to the address at runtime, so every operand must be an
    // alloc_tile carrying a planned address.
    auto resolvePlannedTile = [&](Operation *op, Value tile,
                                  llvm::StringRef role) -> Value {
      auto tileTy = dyn_cast<TileBufType>(tile.getType());
      if (!tileTy) {
        op->emitError() << "VPTO matmul bridge: the " << role
                        << " operand must be a tile_buf";
        hadError = true;
        return nullptr;
      }
      auto alloc = tile.getDefiningOp<AllocTileOp>();
      if (!alloc || !alloc.getAddr()) {
        op->emitError() << "VPTO matmul bridge: the " << role
                        << " tile must come from an alloc_tile with a "
                           "planned address";
        hadError = true;
        return nullptr;
      }
      return alloc.getAddr();
    };

    // Collects the tile template token of a tile operand into the spec.
    auto collectTileToken = [&](Operation *op, Value tile,
                                llvm::StringLiteral specKey,
                                llvm::StringRef role) {
      auto tileTokOr = buildBridgeTileToken(cast<TileBufType>(tile.getType()));
      if (failed(tileTokOr)) {
        op->emitError() << "VPTO matmul bridge failed to build the " << role
                        << " tile template token";
        hadError = true;
        return;
      }
      specFields.emplace_back(specKey, *tileTokOr);
      if (auto alloc = tile.getDefiningOp<AllocTileOp>()) {
        bridgedAllocs.push_back(alloc);
      }
    };

    for (TMatmulOp matmul : matmuls) {
      const BridgeWhitelistEntry *entry = whitelist.findOp(
          matmul->getName().getStringRef());
      if (!entry) {
        continue;
      }
      Value dstAddr = resolvePlannedTile(matmul, matmul.getDst(), "result");
      Value lhsAddr = resolvePlannedTile(matmul, matmul.getLhs(), "left");
      Value rhsAddr = resolvePlannedTile(matmul, matmul.getRhs(), "right");
      if (!dstAddr || !lhsAddr || !rhsAddr) {
        continue;
      }
      if (matmul.getNumResults() > 0) {
        matmul.emitError("VPTO matmul bridge supports the buffer form "
                         "without a tensor result");
        hadError = true;
        continue;
      }
      collectTileToken(matmul, matmul.getLhs(), kBridgeSpecLeftTileKey,
                       "left");
      collectTileToken(matmul, matmul.getRhs(), kBridgeSpecRightTileKey,
                       "right");
      collectTileToken(matmul, matmul.getDst(), kBridgeSpecResultTileKey,
                       "result");
      if (auto phaseTok = buildAccPhaseToken(matmul.getAccPhase())) {
        specFields.emplace_back(kBridgeSpecAccPhaseKey, *phaseTok);
      }
      specFields.emplace_back(kBridgeSpecEntryMatmulKey, entry->entry);
      OpBuilder builder(matmul);
      builder.create<BridgeCallOp>(matmul.getLoc(), /*results=*/TypeRange{},
                                   /*callee=*/entry->entry,
                                   /*storage_size_callee=*/nullptr,
                                   /*args=*/ValueRange{dstAddr, lhsAddr,
                                                       rhsAddr});
      matmul.erase();
    }

    for (TMatmulAccOp matmulAcc : matmulAccs) {
      const BridgeWhitelistEntry *entry = whitelist.findOp(
          matmulAcc->getName().getStringRef());
      if (!entry) {
        continue;
      }
      Value dstAddr =
          resolvePlannedTile(matmulAcc, matmulAcc.getDst(), "result");
      Value accInAddr =
          resolvePlannedTile(matmulAcc, matmulAcc.getAccIn(), "accumulator");
      Value lhsAddr =
          resolvePlannedTile(matmulAcc, matmulAcc.getLhs(), "left");
      Value rhsAddr =
          resolvePlannedTile(matmulAcc, matmulAcc.getRhs(), "right");
      if (!dstAddr || !accInAddr || !lhsAddr || !rhsAddr) {
        continue;
      }
      if (matmulAcc.getNumResults() > 0) {
        matmulAcc.emitError("VPTO matmul bridge supports the buffer form "
                            "without a tensor result");
        hadError = true;
        continue;
      }
      collectTileToken(matmulAcc, matmulAcc.getLhs(), kBridgeSpecLeftTileKey,
                       "left");
      collectTileToken(matmulAcc, matmulAcc.getRhs(), kBridgeSpecRightTileKey,
                       "right");
      collectTileToken(matmulAcc, matmulAcc.getDst(), kBridgeSpecResultTileKey,
                       "result");
      collectTileToken(matmulAcc, matmulAcc.getAccIn(),
                       kBridgeSpecAccInTileKey, "accumulator");
      if (auto phaseTok = buildAccPhaseToken(matmulAcc.getAccPhase())) {
        specFields.emplace_back(kBridgeSpecAccPhaseKey, *phaseTok);
      }
      specFields.emplace_back(kBridgeSpecEntryMatmulAccKey, entry->entry);
      OpBuilder builder(matmulAcc);
      builder.create<BridgeCallOp>(
          matmulAcc.getLoc(), /*results=*/TypeRange{},
          /*callee=*/entry->entry, /*storage_size_callee=*/nullptr,
          /*args=*/ValueRange{dstAddr, accInAddr, lhsAddr, rhsAddr});
      matmulAcc.erase();
    }

    // Erase the tile handles consumed by the bridged matmul ops. Handles
    // with surviving users (e.g. a tile_buf_addr feeding a non-bridged op)
    // stay on the regular lowering path.
    for (AllocTileOp alloc : bridgedAllocs) {
      if (alloc.use_empty()) {
        alloc.erase();
      }
    }

    if (hadError) {
      signalPassFailure();
      return;
    }
    if (!specFields.empty()) {
      SmallVector<NamedAttribute> specAttrs;
      for (const auto &field : specFields) {
        specAttrs.push_back({StringAttr::get(func.getContext(), field.first),
                             StringAttr::get(func.getContext(), field.second)});
      }
      func->setAttr(kBridgeFuncSpecAttrName,
                    DictionaryAttr::get(func.getContext(), specAttrs));
    }
  }
};

} // namespace

std::unique_ptr<Pass> createPTOLowerMatmulFamilyOpsPass() {
  return std::make_unique<PTOLowerMatmulFamilyOpsPass>();
}

} // namespace pto
} // namespace mlir

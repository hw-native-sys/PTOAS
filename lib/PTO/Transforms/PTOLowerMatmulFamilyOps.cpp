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
// semantics of the tile-world matmul ops (tmatmul / tmatmul.acc and the
// bias/MX entry variants) and rewrites them into generic pto.bridge_call
// ops that carry only the wrapper callee name and the planned i64
// addresses of the operand tiles.
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
#include "llvm/ADT/StringMap.h"

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
    SmallVector<TMatmulBiasOp> matmulBiases;
    SmallVector<TMatmulMxOp> matmulMxes;
    SmallVector<TMatmulMxAccOp> matmulMxAccs;
    SmallVector<TMatmulMxBiasOp> matmulMxBiases;
    func.walk([&](Operation *op) {
      if (auto matmul = dyn_cast<TMatmulOp>(op)) {
        matmuls.push_back(matmul);
      } else if (auto matmulAcc = dyn_cast<TMatmulAccOp>(op)) {
        matmulAccs.push_back(matmulAcc);
      } else if (auto matmulBias = dyn_cast<TMatmulBiasOp>(op)) {
        matmulBiases.push_back(matmulBias);
      } else if (auto matmulMx = dyn_cast<TMatmulMxOp>(op)) {
        matmulMxes.push_back(matmulMx);
      } else if (auto matmulMxAcc = dyn_cast<TMatmulMxAccOp>(op)) {
        matmulMxAccs.push_back(matmulMxAcc);
      } else if (auto matmulMxBias = dyn_cast<TMatmulMxBiasOp>(op)) {
        matmulMxBiases.push_back(matmulMxBias);
      }
    });
    if (matmuls.empty() && matmulAccs.empty() && matmulBiases.empty() &&
        matmulMxes.empty() && matmulMxAccs.empty() && matmulMxBiases.empty()) {
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
    // The spec becomes a DictionaryAttr, so each key may appear at most
    // once. Repeat writes with the same token are harmless (multiple ops
    // sharing a tile shape or entry); a different token for a key already
    // written is a conflict the wrapper cannot render.
    llvm::StringMap<std::string> writtenSpecFields;
    auto addSpecField = [&](Operation *op, llvm::StringLiteral key,
                            std::string token) {
      auto inserted = writtenSpecFields.try_emplace(key, token);
      if (inserted.second) {
        specFields.emplace_back(key, token);
        return;
      }
      if (inserted.first->second != token) {
        op->emitError() << "VPTO matmul bridge spec field '" << key
                        << "' was already collected as '"
                        << inserted.first->second
                        << "'; the wrapper renders one token per spec field";
        hadError = true;
      }
    };
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
      addSpecField(op, specKey, *tileTokOr);
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
        addSpecField(matmul, kBridgeSpecAccPhaseKey, *phaseTok);
      }
      addSpecField(matmul, kBridgeSpecEntryMatmulKey, entry->entry);
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
        addSpecField(matmulAcc, kBridgeSpecAccPhaseKey, *phaseTok);
      }
      addSpecField(matmulAcc, kBridgeSpecEntryMatmulAccKey, entry->entry);
      OpBuilder builder(matmulAcc);
      builder.create<BridgeCallOp>(
          matmulAcc.getLoc(), /*results=*/TypeRange{},
          /*callee=*/entry->entry, /*storage_size_callee=*/nullptr,
          /*args=*/ValueRange{dstAddr, accInAddr, lhsAddr, rhsAddr});
      matmulAcc.erase();
    }

    for (TMatmulBiasOp matmulBias : matmulBiases) {
      const BridgeWhitelistEntry *entry = whitelist.findOp(
          matmulBias->getName().getStringRef());
      if (!entry) {
        continue;
      }
      Value dstAddr =
          resolvePlannedTile(matmulBias, matmulBias.getDst(), "result");
      Value lhsAddr = resolvePlannedTile(matmulBias, matmulBias.getA(), "left");
      Value rhsAddr =
          resolvePlannedTile(matmulBias, matmulBias.getB(), "right");
      Value biasAddr =
          resolvePlannedTile(matmulBias, matmulBias.getBias(), "bias");
      if (!dstAddr || !lhsAddr || !rhsAddr || !biasAddr) {
        continue;
      }
      if (matmulBias.getNumResults() > 0) {
        matmulBias.emitError("VPTO matmul bridge supports the buffer form "
                             "without a tensor result");
        hadError = true;
        continue;
      }
      collectTileToken(matmulBias, matmulBias.getA(), kBridgeSpecLeftTileKey,
                       "left");
      collectTileToken(matmulBias, matmulBias.getB(), kBridgeSpecRightTileKey,
                       "right");
      collectTileToken(matmulBias, matmulBias.getDst(),
                       kBridgeSpecResultTileKey, "result");
      collectTileToken(matmulBias, matmulBias.getBias(),
                       kBridgeSpecBiasTileKey, "bias");
      if (auto phaseTok = buildAccPhaseToken(matmulBias.getAccPhase())) {
        addSpecField(matmulBias, kBridgeSpecAccPhaseKey, *phaseTok);
      }
      addSpecField(matmulBias, kBridgeSpecEntryMatmulBiasKey, entry->entry);
      OpBuilder builder(matmulBias);
      builder.create<BridgeCallOp>(
          matmulBias.getLoc(), /*results=*/TypeRange{},
          /*callee=*/entry->entry, /*storage_size_callee=*/nullptr,
          /*args=*/ValueRange{dstAddr, lhsAddr, rhsAddr, biasAddr});
      matmulBias.erase();
    }

    for (TMatmulMxOp matmulMx : matmulMxes) {
      const BridgeWhitelistEntry *entry = whitelist.findOp(
          matmulMx->getName().getStringRef());
      if (!entry) {
        continue;
      }
      Value dstAddr = resolvePlannedTile(matmulMx, matmulMx.getDst(), "result");
      Value lhsAddr = resolvePlannedTile(matmulMx, matmulMx.getA(), "left");
      Value aScaleAddr =
          resolvePlannedTile(matmulMx, matmulMx.getAScale(), "left scale");
      Value rhsAddr = resolvePlannedTile(matmulMx, matmulMx.getB(), "right");
      Value bScaleAddr =
          resolvePlannedTile(matmulMx, matmulMx.getBScale(), "right scale");
      if (!dstAddr || !lhsAddr || !aScaleAddr || !rhsAddr || !bScaleAddr) {
        continue;
      }
      if (matmulMx.getNumResults() > 0) {
        matmulMx.emitError("VPTO matmul bridge supports the buffer form "
                           "without a tensor result");
        hadError = true;
        continue;
      }
      collectTileToken(matmulMx, matmulMx.getA(), kBridgeSpecLeftTileKey,
                       "left");
      collectTileToken(matmulMx, matmulMx.getB(), kBridgeSpecRightTileKey,
                       "right");
      collectTileToken(matmulMx, matmulMx.getDst(), kBridgeSpecResultTileKey,
                       "result");
      collectTileToken(matmulMx, matmulMx.getAScale(),
                       kBridgeSpecAScaleTileKey, "left scale");
      collectTileToken(matmulMx, matmulMx.getBScale(),
                       kBridgeSpecBScaleTileKey, "right scale");
      if (auto phaseTok = buildAccPhaseToken(matmulMx.getAccPhase())) {
        addSpecField(matmulMx, kBridgeSpecAccPhaseKey, *phaseTok);
      }
      addSpecField(matmulMx, kBridgeSpecEntryMatmulMxKey, entry->entry);
      OpBuilder builder(matmulMx);
      builder.create<BridgeCallOp>(
          matmulMx.getLoc(), /*results=*/TypeRange{},
          /*callee=*/entry->entry, /*storage_size_callee=*/nullptr,
          /*args=*/
          ValueRange{dstAddr, lhsAddr, aScaleAddr, rhsAddr, bScaleAddr});
      matmulMx.erase();
    }

    for (TMatmulMxAccOp matmulMxAcc : matmulMxAccs) {
      const BridgeWhitelistEntry *entry = whitelist.findOp(
          matmulMxAcc->getName().getStringRef());
      if (!entry) {
        continue;
      }
      Value dstAddr =
          resolvePlannedTile(matmulMxAcc, matmulMxAcc.getDst(), "result");
      Value cInAddr =
          resolvePlannedTile(matmulMxAcc, matmulMxAcc.getCIn(), "accumulator");
      Value lhsAddr = resolvePlannedTile(matmulMxAcc, matmulMxAcc.getA(),
                                         "left");
      Value aScaleAddr = resolvePlannedTile(matmulMxAcc,
                                            matmulMxAcc.getAScale(),
                                            "left scale");
      Value rhsAddr = resolvePlannedTile(matmulMxAcc, matmulMxAcc.getB(),
                                         "right");
      Value bScaleAddr = resolvePlannedTile(matmulMxAcc,
                                            matmulMxAcc.getBScale(),
                                            "right scale");
      if (!dstAddr || !cInAddr || !lhsAddr || !aScaleAddr || !rhsAddr ||
          !bScaleAddr) {
        continue;
      }
      if (matmulMxAcc.getNumResults() > 0) {
        matmulMxAcc.emitError("VPTO matmul bridge supports the buffer form "
                              "without a tensor result");
        hadError = true;
        continue;
      }
      collectTileToken(matmulMxAcc, matmulMxAcc.getA(),
                       kBridgeSpecLeftTileKey, "left");
      collectTileToken(matmulMxAcc, matmulMxAcc.getB(),
                       kBridgeSpecRightTileKey, "right");
      collectTileToken(matmulMxAcc, matmulMxAcc.getDst(),
                       kBridgeSpecResultTileKey, "result");
      collectTileToken(matmulMxAcc, matmulMxAcc.getCIn(),
                       kBridgeSpecAccInTileKey, "accumulator");
      collectTileToken(matmulMxAcc, matmulMxAcc.getAScale(),
                       kBridgeSpecAScaleTileKey, "left scale");
      collectTileToken(matmulMxAcc, matmulMxAcc.getBScale(),
                       kBridgeSpecBScaleTileKey, "right scale");
      if (auto phaseTok = buildAccPhaseToken(matmulMxAcc.getAccPhase())) {
        addSpecField(matmulMxAcc, kBridgeSpecAccPhaseKey, *phaseTok);
      }
      addSpecField(matmulMxAcc, kBridgeSpecEntryMatmulMxAccKey,
                   entry->entry);
      OpBuilder builder(matmulMxAcc);
      builder.create<BridgeCallOp>(
          matmulMxAcc.getLoc(), /*results=*/TypeRange{},
          /*callee=*/entry->entry, /*storage_size_callee=*/nullptr,
          /*args=*/
          ValueRange{dstAddr, cInAddr, lhsAddr, aScaleAddr, rhsAddr,
                     bScaleAddr});
      matmulMxAcc.erase();
    }

    for (TMatmulMxBiasOp matmulMxBias : matmulMxBiases) {
      const BridgeWhitelistEntry *entry = whitelist.findOp(
          matmulMxBias->getName().getStringRef());
      if (!entry) {
        continue;
      }
      Value dstAddr =
          resolvePlannedTile(matmulMxBias, matmulMxBias.getDst(), "result");
      Value lhsAddr =
          resolvePlannedTile(matmulMxBias, matmulMxBias.getA(), "left");
      Value aScaleAddr = resolvePlannedTile(matmulMxBias,
                                            matmulMxBias.getAScale(),
                                            "left scale");
      Value rhsAddr =
          resolvePlannedTile(matmulMxBias, matmulMxBias.getB(), "right");
      Value bScaleAddr = resolvePlannedTile(matmulMxBias,
                                            matmulMxBias.getBScale(),
                                            "right scale");
      Value biasAddr =
          resolvePlannedTile(matmulMxBias, matmulMxBias.getBias(), "bias");
      if (!dstAddr || !lhsAddr || !aScaleAddr || !rhsAddr || !bScaleAddr ||
          !biasAddr) {
        continue;
      }
      if (matmulMxBias.getNumResults() > 0) {
        matmulMxBias.emitError("VPTO matmul bridge supports the buffer form "
                               "without a tensor result");
        hadError = true;
        continue;
      }
      collectTileToken(matmulMxBias, matmulMxBias.getA(),
                       kBridgeSpecLeftTileKey, "left");
      collectTileToken(matmulMxBias, matmulMxBias.getB(),
                       kBridgeSpecRightTileKey, "right");
      collectTileToken(matmulMxBias, matmulMxBias.getDst(),
                       kBridgeSpecResultTileKey, "result");
      collectTileToken(matmulMxBias, matmulMxBias.getAScale(),
                       kBridgeSpecAScaleTileKey, "left scale");
      collectTileToken(matmulMxBias, matmulMxBias.getBScale(),
                       kBridgeSpecBScaleTileKey, "right scale");
      collectTileToken(matmulMxBias, matmulMxBias.getBias(),
                       kBridgeSpecBiasTileKey, "bias");
      // The MX-bias IR op carries no accumulation phase, so its wrapper
      // entry renders without a Phase template argument.
      addSpecField(matmulMxBias, kBridgeSpecEntryMatmulMxBiasKey,
                   entry->entry);
      OpBuilder builder(matmulMxBias);
      builder.create<BridgeCallOp>(
          matmulMxBias.getLoc(), /*results=*/TypeRange{},
          /*callee=*/entry->entry, /*storage_size_callee=*/nullptr,
          /*args=*/
          ValueRange{dstAddr, lhsAddr, aScaleAddr, rhsAddr, bScaleAddr,
                     biasAddr});
      matmulMxBias.erase();
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

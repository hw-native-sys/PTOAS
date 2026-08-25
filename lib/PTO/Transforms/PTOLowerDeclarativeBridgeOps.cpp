// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOLowerDeclarativeBridgeOps.cpp - declarative bridge lowering ----===//
//===----------------------------------------------------------------------===//
//
// Generic declarative lowering channel of the VPTO C++ interface bridge.
// It rewrites every whitelist entry marked `lowering: declarative` into a
// void pto.bridge_call using only the whitelist description: each abi row
// binds a wrapper argument to an IR operand position whose planned
// alloc_tile address becomes the i64 call argument, and the template
// specialization is collected from the operand tile types (keyed by the
// abi role) plus optional enum attributes (tmpl_map `source: attr` rows).
// No family semantics are understood here; ops needing storage lifecycle
// or address rebinding stay on their family pass (`lowering: family`).
//
// Routing is whitelist driven: an op the whitelist does not route (or
// routes to the family channel) is left untouched, so unrouted matmul ops
// keep flowing through the regular tile-op expansion path.
//
// The collected spec keys deliberately match the constants consumed by the
// wrapper generation pass: the role name is the tile spec key, the attr
// row `field` is the enum spec key, the entry spec key is derived
// from the op name (`pto.tmatmul.mx.acc` -> `entry.matmul_mx_acc`), and
// the reserved `core.<wrapper>` key carries the derived core guard of a
// wrapper declaration that omits `core`.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/VPTOBridgeSpecCollector.h"
#include "PTO/Transforms/VPTOBridgeTokens.h"
#include "PTO/Transforms/VPTOBridgeWhitelist.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include <algorithm>
#include <string>

namespace mlir {
namespace pto {

#define GEN_PASS_DECL_PTOLOWERDECLARATIVEBRIDGEOPS
#define GEN_PASS_DEF_PTOLOWERDECLARATIVEBRIDGEOPS
#include "PTO/Transforms/Passes.h.inc"

namespace {

/// Derives the wrapper entry spec key from the routed IR op name.
/// Tile-world ops carry the `pto.t` mnemonic prefix, which is not part of
/// the wrapper's entry naming (`pto.tmatmul.mx.acc` -> `entry.matmul_mx_acc`).
static std::string deriveEntrySpecKey(llvm::StringRef opName) {
  constexpr llvm::StringLiteral kTileWorldOpPrefix = "pto.t";
  if (!opName.consume_front(kTileWorldOpPrefix)) {
    opName.consume_front("pto.");
  }
  std::string key = ("entry." + opName).str();
  constexpr llvm::StringLiteral kEntryKeyPrefix = "entry.";
  std::replace(key.begin() + kEntryKeyPrefix.size(), key.end(), '.', '_');
  return key;
}

/// Derives the camelCase spelling of a snake_case whitelist field name.
/// ODS attribute names are camelCase ($accPhase) while the whitelist field
/// doubles as the spec key (acc_phase), so the attribute lookup tries both
/// spellings.
static std::string camelCaseFieldName(llvm::StringRef fieldName) {
  std::string camel;
  camel.reserve(fieldName.size());
  bool upperNext = false;
  for (char c : fieldName) {
    if (c == '_') {
      upperNext = true;
      continue;
    }
    camel.push_back(upperNext && c >= 'a' && c <= 'z'
                        ? static_cast<char>(c - 'a' + 'A')
                        : c);
    upperNext = false;
  }
  return camel;
}

/// Maps a bridged tile to the core kind its wrapper renders under when the
/// wrapper declaration omits `core`: VEC tiles run on the vector core, the
/// cube-family tile spaces (mat/left/right/acc/bias/scaling) on the cube
/// core. Tiles without a supported address space fail earlier in
/// buildBridgeTileToken, so the cube default here is unreachable for a
/// successfully collected tile.
static llvm::StringLiteral bridgeCoreKindForTile(Value tile) {
  auto tileTy = cast<TileBufType>(tile.getType());
  auto spaceAttr =
      dyn_cast_or_null<AddressSpaceAttr>(tileTy.getMemorySpace());
  if (spaceAttr && spaceAttr.getAddressSpace() == AddressSpace::VEC)
    return kBridgeWrapperCoreVec;
  return kBridgeWrapperCoreCube;
}

struct PTOLowerDeclarativeBridgeOpsPass final
    : public impl::PTOLowerDeclarativeBridgeOpsBase<
          PTOLowerDeclarativeBridgeOpsPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTOLowerDeclarativeBridgeOpsPass)

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // The whitelist always resolves through the formal chain (pass option,
    // PTOAS_VPTO_BRIDGE_WHITELIST, built-in default); kernels that want the
    // regular tile-op expansion route the op out of the whitelist with an
    // explicit whitelist file.
    FailureOr<BridgeWhitelist> whitelistOr =
        loadBridgeWhitelist(whitelistPath, llvm::errs());
    if (failed(whitelistOr)) {
      signalPassFailure();
      return;
    }
    const BridgeWhitelist &whitelist = *whitelistOr;

    // Collect the ops routed to the declarative channel first; rewriting
    // during the walk would invalidate the walker. Ops with no whitelist
    // entry or with a family entry are left untouched: unrouted ops keep
    // their non-bridge lowering (e.g. the matmul mad expansion), and family
    // entries are rewritten by their family pass.
    SmallVector<std::pair<Operation *, const BridgeWhitelistEntry *>> routed;
    func.walk([&](Operation *op) {
      const BridgeWhitelistEntry *entry =
          whitelist.findOp(op->getName().getStringRef());
      if (entry && entry->isDeclarative()) {
        routed.push_back({op, entry});
      }
    });
    if (routed.empty()) {
      return;
    }

    bool hadError = false;
    // Wrapper specialization fields collected while lowering this function;
    // stored as a function attribute once lowering succeeds. The module-level
    // wrapper generation pass merges the per-function specs deterministically
    // (the pass instances may run concurrently).
    BridgeSpecCollector spec;
    // Tile handles consumed by bridged ops; erased once use-empty.
    SmallVector<AllocTileOp> bridgedAllocs;

    // Resolves a tile operand to its planned address. The bridge wrapper
    // binds each tile to the address at runtime, so every operand must be an
    // alloc_tile carrying a planned address.
    auto resolvePlannedTile = [&](Operation *op, const BridgeAbiArg &abiArg,
                                  Value tile) -> Value {
      auto tileTy = dyn_cast<TileBufType>(tile.getType());
      if (!tileTy) {
        op->emitError() << "VPTO declarative bridge: operand #" << abiArg.operand
                        << " ('" << abiArg.arg << "', role " << abiArg.role
                        << ") must be a tile_buf";
        hadError = true;
        return nullptr;
      }
      auto alloc = tile.getDefiningOp<AllocTileOp>();
      if (!alloc || !alloc.getAddr()) {
        op->emitError() << "VPTO declarative bridge: operand #" << abiArg.operand
                        << " ('" << abiArg.arg << "', role " << abiArg.role
                        << ") tile must come from an alloc_tile with a "
                           "planned address";
        hadError = true;
        return nullptr;
      }
      return alloc.getAddr();
    };

    // Collects the tile template token of one abi-bound operand into the
    // spec under the operand's role, plus the core kind of the wrapper the
    // entry routes into: a wrapper declaration may omit `core`, in which
    // case the renderer picks the guard up from the reserved
    // `core.<wrapper>` spec key collected here.
    auto collectTileToken = [&](Operation *op, const BridgeAbiArg &abiArg,
                                Value tile,
                                const BridgeWhitelistEntry &entry) {
      auto tileTokOr = buildBridgeTileToken(cast<TileBufType>(tile.getType()));
      if (failed(tileTokOr)) {
        op->emitError() << "VPTO declarative bridge failed to build the "
                        << abiArg.role << " tile template token for operand '"
                        << abiArg.arg << "'";
        hadError = true;
        return;
      }
      spec.addField(op, abiArg.role, *tileTokOr);
      spec.addField(op, "core." + entry.wrapper,
                    bridgeCoreKindForTile(tile));
      if (auto alloc = tile.getDefiningOp<AllocTileOp>()) {
        bridgedAllocs.push_back(alloc);
      }
    };

    // Renders a tmpl_map attr row into the spec. The attribute is reflected
    // through the enum token interface; a missing attribute or the omit
    // case renders no template argument (e.g. the Unspecified accumulation
    // phase, and entries such as tmatmul.mx.bias that carry no phase).
    auto collectAttrToken = [&](Operation *op,
                                const BridgeTmplMapField &field) {
      Attribute attrValue = op->getAttr(field.field);
      if (!attrValue) {
        attrValue = op->getAttr(camelCaseFieldName(field.field));
      }
      if (!attrValue) {
        return;
      }
      auto enumTokenAttr = dyn_cast<EnumTokenAttr>(attrValue);
      if (!enumTokenAttr) {
        op->emitError() << "VPTO declarative bridge: attribute '"
                        << field.field
                        << "' must be a PTO enum attribute to feed a "
                           "template slot";
        hadError = true;
        return;
      }
      llvm::StringRef caseSymbol = enumTokenAttr.getEnumCaseSymbol();
      if (!field.omitValue.empty() && caseSymbol == field.omitValue) {
        return;
      }
      spec.addField(op, field.field, field.enumType + "::" + caseSymbol.str());
    };

    for (auto &[op, entry] : routed) {
      // Per-op error flag: one broken op must not keep the other routed ops
      // from lowering (matching the family pass continue semantics); the
      // pass fails at the end when any error was recorded.
      bool opFailed = false;
      if (op->getNumResults() > 0) {
        op->emitError("VPTO declarative bridge supports the buffer form "
                      "without a tensor result");
        hadError = true;
        continue;
      }
      // Resolve the call arguments in abi order; every operand must bind to
      // a planned tile address.
      SmallVector<Value> callArgs;
      for (const BridgeAbiArg &abiArg : entry->abi) {
        if (abiArg.operand >= static_cast<int64_t>(op->getNumOperands())) {
          op->emitError() << "VPTO declarative bridge: whitelist entry '"
                          << entry->entry << "' binds operand #"
                          << abiArg.operand << " but the op has only "
                          << op->getNumOperands() << " operands";
          hadError = true;
          opFailed = true;
          break;
        }
        Value tile = op->getOperand(abiArg.operand);
        Value addr = resolvePlannedTile(op, abiArg, tile);
        if (!addr) {
          opFailed = true;
          break;
        }
        callArgs.push_back(addr);
      }
      if (opFailed) {
        continue;
      }
      // Template specialization: one tile token per abi role, then the
      // enum attribute rows, then the wrapper entry name.
      for (const BridgeAbiArg &abiArg : entry->abi) {
        collectTileToken(op, abiArg, op->getOperand(abiArg.operand), *entry);
      }
      for (const BridgeTmplMapField &field : entry->tmplMap) {
        if (field.source == kAttrTmplMapSource) {
          collectAttrToken(op, field);
        }
      }
      spec.addField(op, deriveEntrySpecKey(op->getName().getStringRef()),
                    entry->entry);
      OpBuilder builder(op);
      builder.create<BridgeCallOp>(op->getLoc(), /*results=*/TypeRange{},
                                   /*callee=*/entry->entry,
                                   /*storage_size_callee=*/nullptr,
                                   /*args=*/callArgs);
      op->erase();
    }

    // Erase the tile handles consumed by the bridged ops. Handles with
    // surviving users (e.g. a tile_buf_addr feeding a non-bridged op) stay
    // on the regular lowering path.
    for (AllocTileOp alloc : bridgedAllocs) {
      if (alloc.use_empty()) {
        alloc.erase();
      }
    }

    if (hadError || spec.hadError()) {
      signalPassFailure();
      return;
    }
    spec.store(func);
  }
};

} // namespace

std::unique_ptr<Pass> createPTOLowerDeclarativeBridgeOpsPass() {
  return std::make_unique<PTOLowerDeclarativeBridgeOpsPass>();
}

} // namespace pto
} // namespace mlir

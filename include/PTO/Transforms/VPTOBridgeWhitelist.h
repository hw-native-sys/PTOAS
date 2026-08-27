// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOBridgeWhitelist.h - C++ bridge whitelist --------------*- C++ -*-===//
//===----------------------------------------------------------------------===//
//
// Declarative description of which IR ops are routed to the VPTO C++
// interface bridge and how their arguments map onto wrapper ABI values.
// The generic bridge lowering pass consumes this table to validate bridge
// calls; wrapper generation consumes it to synthesize wrapper sources.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_VPTOBRIDGEWHITELIST_H
#define MLIR_DIALECT_PTO_TRANSFORMS_VPTOBRIDGEWHITELIST_H

#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <vector>

namespace mlir {
namespace pto {

/// ABI argument of a wrapper entry. `type` is one of the supported carrier
/// tokens: "ptr", "i64", or "i32" (declarative entries may omit it, the
/// parser fills in the "i64" tile-address default). Declarative entries
/// additionally bind each argument to an IR operand position (`operand`,
/// positional because MLIR exposes no generic ODS operand-name
/// reflection) and the template role the operand's tile token is collected
/// under (`role`, which is also the spec key and the source of the entry's
/// tile typedef name). `arg` is the diagnostic label and rendered parameter
/// name (the ODS operand name); it defaults to the lowerCamelCase of the
/// role (see bridgeRoleParamName), so a binding registers with
/// operand/role only.
struct BridgeAbiArg {
  std::string type;
  int64_t operand = -1;
  std::string arg;
  std::string role;
};

/// Declarative template-argument mapping row: an IR field (`source` +
/// `field`) feeds a C++ template slot (`target`). Consumed by wrapper
/// generation to validate that the collected specialization covers the
/// declared slots; the authoritative token construction lives in
/// VPTOBridgeTokens. Declarative entries carry only `source: attr` rows
/// (an enum attribute mapped to a template slot, with `enumType` providing
/// the qualified C++ enum spelling and `omitValue` the case that renders
/// no template argument); their tile typedefs derive from the abi roles,
/// so tile rows are rejected at parse time.
struct BridgeTmplMapField {
  std::string source;
  std::string field;
  std::string target;
  std::string enumType;
  std::string omitValue;
};

/// One whitelist row: an IR op routed to a wrapper entry of a bridged
/// PTO-ISA C++ interface.
struct BridgeWhitelistEntry {
  /// IR op name, e.g. "pto.tpush". The whitelist is the routing table the
  /// family passes consult; "internal" marks wrapper-internal helpers
  /// (e.g. the size query entry) that are never routed from an IR op.
  std::string op;
  /// Wrapper source this entry is rendered into, e.g. "pipe". Entries
  /// sharing a wrapper share one C++ translation unit, one template
  /// specialization and one typedef set, so this is the unit of wrapper
  /// generation rather than a taxonomy of ops: the N IR ops of the pipe
  /// protocol (init/push/pop/free plus the internal size query) all name
  /// wrapper "pipe" and render into a single source. Wrapper generation
  /// dispatches on it to pick the renderer, and a module may use entries of
  /// exactly one wrapper. It also scopes the tmpl_map source validation of
  /// custom-channel entries.
  std::string wrapper;
  /// Lowering channel. "declarative" (the default) routes the op through the
  /// generic declarative bridge lowering: a mechanical operand-adapter
  /// mapping driven entirely by this table, needing no pass code. "custom"
  /// opts the entry out of that channel because it carries real family
  /// semantics (storage lifecycle, address rebinding) that only a dedicated
  /// pass can express.
  ///
  /// The default is deliberately the generic channel: adding a mechanically
  /// mapped interface family must cost zero pass code *and* zero ceremony,
  /// while the exception stays explicit. It also makes the common mistake
  /// self-detecting -- an entry that needs a custom pass but forgets the tag
  /// lacks the operand/arg/role bindings the declarative channel requires,
  /// so it is rejected at parse time with the missing field named, instead
  /// of surviving until the post-lowering leftover check.
  std::string lowering = "declarative";
  /// Wrapper entry name, e.g. "pto_vpto_pipe_push". This is the callee
  /// the generic bridge lowering emits. Optional for declarative routed
  /// entries, which default to a name derived from the op name (see
  /// deriveDefaultBridgeEntry); custom entries must declare it.
  std::string entry;
  /// C++ call spelling the generic declarative renderer emits for this
  /// entry, e.g. "pto::TMATMUL" or "TADD". Declarative routed entries
  /// only; the call arguments are the abi-bound tiles in declaration
  /// order. Optional: it defaults to a spelling derived from the op name
  /// (see deriveDefaultBridgeCall); a wrong derivation fails loudly when
  /// the generated wrapper source is compiled.
  std::string call;
  /// Template arguments rendered between the call spelling and its
  /// argument list. Each item is either the `field` of one of this
  /// entry's attr tmpl_map rows (rendered as the spec token collected
  /// for that field; the whole template argument list is omitted when
  /// the spec carries no token) or a literal qualified C++ spelling
  /// (contains "::"). Declarative routed entries only.
  std::vector<std::string> tmplArgs;
  /// Call-side ABI of the wrapper entry, including any synthesized
  /// arguments such as the storage pointer of stateful entries.
  std::vector<BridgeAbiArg> abi;
  /// Wrapper entry returning the size of the stateful object owned by this
  /// entry. Declared on stateful entries (e.g. the pipe init) and consumed
  /// by the family pass as the bridge call storage_size_callee.
  std::string storageSizeEntry;
  /// Declarative IR-field -> C++ template-slot mappings for wrapper
  /// generation. Optional; empty when the entry needs no template mapping.
  std::vector<BridgeTmplMapField> tmplMap;

  /// Returns whether the op lowers through the generic declarative channel.
  bool isDeclarative() const { return lowering == kLoweringDeclarative; }

  /// `lowering` value (the default) routing the op through the generic
  /// declarative bridge lowering.
  static constexpr llvm::StringLiteral kLoweringDeclarative = "declarative";
  /// `lowering` value opting the entry out of the declarative channel: a
  /// dedicated family pass owns the rewrite.
  static constexpr llvm::StringLiteral kLoweringCustom = "custom";
};

/// Declaration of a wrapper rendered by the generic declarative renderer:
/// the family knowledge no whitelist entry carries. `includes` lists the
/// PTO-ISA headers the wrapper translation unit includes; `core` selects
/// the core guard the entries render under ("cube" -> __DAV_CUBE__,
/// "vec" -> __DAV_VEC__, "both" -> no guard). `core` is optional: it
/// defaults to the kind of the tile address spaces the declarative
/// lowering collects for the wrapper (VEC tiles -> "vec", any cube-family
/// tile -> "cube"), so a single-core wrapper registers without it; a
/// wrapper whose used entries collected no tile declares it explicitly.
/// Wrappers whose entries carry `lowering: custom` own a dedicated
/// renderer and must not be declared here.
struct BridgeWrapperDecl {
  std::string name;
  std::vector<std::string> includes;
  std::string core;
};

/// Parsed whitelist document.
struct BridgeWhitelist {
  std::vector<BridgeWhitelistEntry> bridgeOps;
  std::vector<BridgeWrapperDecl> wrappers;

  /// Returns the wrapper declaration named `name`, or nullptr.
  const BridgeWrapperDecl *findWrapper(llvm::StringRef name) const {
    for (const BridgeWrapperDecl &decl : wrappers) {
      if (decl.name == name) {
        return &decl;
      }
    }
    return nullptr;
  }

  /// Returns whether any entry routing into the wrapper `name` carries
  /// `lowering: custom` (and thus needs a dedicated renderer).
  bool wrapperHasCustomEntry(llvm::StringRef name) const {
    for (const BridgeWhitelistEntry &entry : bridgeOps) {
      if (entry.wrapper == name && !entry.isDeclarative()) {
        return true;
      }
    }
    return false;
  }

  /// Returns the entry whose wrapper name is `entryName`, or nullptr.
  const BridgeWhitelistEntry *findEntry(llvm::StringRef entryName) const {
    for (const BridgeWhitelistEntry &entry : bridgeOps) {
      if (entry.entry == entryName) {
        return &entry;
      }
    }
    return nullptr;
  }

  /// Returns the entry routing the IR op `opName` (e.g. "pto.tpush"), or
  /// nullptr. Wrapper-internal helpers (op == "internal") are never routed.
  const BridgeWhitelistEntry *findOp(llvm::StringRef opName) const {
    for (const BridgeWhitelistEntry &entry : bridgeOps) {
      if (entry.op == opName && entry.op != kInternalOp) {
        return &entry;
      }
    }
    return nullptr;
  }

  /// Marker `op` value of wrapper-internal helper entries that no IR op
  /// routes to (e.g. the stateful-object size query).
  static constexpr llvm::StringLiteral kInternalOp = "internal";
};

/// Parses a whitelist YAML file. Diagnostics are written to `diagOS`.
/// Rejects unreadable files, YAML syntax errors, empty fields, duplicate
/// wrapper entry names, duplicate routed op names, unsupported ABI type
/// tokens, and dangling storage_size_entry references.
FailureOr<BridgeWhitelist> parseBridgeWhitelist(llvm::StringRef path,
                                                llvm::raw_ostream &diagOS);

/// Parses a whitelist YAML document already in memory; `sourceName` is used
/// in diagnostics (e.g. a file path or the built-in whitelist marker).
FailureOr<BridgeWhitelist>
parseBridgeWhitelistFromBuffer(llvm::StringRef content,
                               llvm::StringRef sourceName,
                               llvm::raw_ostream &diagOS);

/// Resolves the whitelist path from a pass `whitelist-path` option value,
/// falling back to the PTOAS_VPTO_BRIDGE_WHITELIST environment variable.
/// Returns an empty string when neither is configured.
std::string resolveBridgeWhitelistPath(llvm::StringRef optionValue);

/// Source name used in diagnostics when the built-in default whitelist is
/// in effect.
constexpr llvm::StringLiteral kBuiltinBridgeWhitelistSource =
    "<built-in vpto bridge whitelist>";

/// Loads the bridge whitelist through the formal resolution chain: pass
/// `whitelist-path` option, then PTOAS_VPTO_BRIDGE_WHITELIST, then the
/// built-in default whitelist (pipe family) shipped with ptoas.
/// Always returns a parsed whitelist unless the explicitly configured file
/// fails to parse. When `sourceName` is non-null it receives the resolved
/// source name (file path or the built-in marker) for diagnostics.
FailureOr<BridgeWhitelist> loadBridgeWhitelist(llvm::StringRef optionValue,
                                               llvm::raw_ostream &diagOS,
                                               std::string *sourceName = nullptr);

/// Returns whether `token` is one of the ABI carrier tokens accepted by the
/// generic bridge lowering ("ptr", "i64", "i32"). The set stays closed so
/// whitelist parsing and lowering agree on the carriers.
bool isSupportedBridgeAbiType(llvm::StringRef token);

/// Returns whether the ABI carrier token describes `type` after bridge
/// lowering conversion ("ptr" -> LLVM pointer, "i64"/"i32" -> the integer
/// widths).
bool bridgeAbiTypeMatches(llvm::StringRef token, Type type);

/// tmpl_map `source` tokens naming IR producers of a template argument.
constexpr llvm::StringLiteral kPipeInitTmplMapSource = "pipe.init";
constexpr llvm::StringLiteral kTileTmplMapSource = "tile";
/// tmpl_map `source` token mapping an enum attribute of the routed op to a
/// template slot.
constexpr llvm::StringLiteral kAttrTmplMapSource = "attr";

/// `core` values accepted by a wrapper declaration.
constexpr llvm::StringLiteral kBridgeWrapperCoreCube = "cube";
constexpr llvm::StringLiteral kBridgeWrapperCoreVec = "vec";
constexpr llvm::StringLiteral kBridgeWrapperCoreBoth = "both";

/// Derives the default wrapper entry name of a declarative routed entry
/// from its IR op name: strip the `pto.` prefix, drop the tile-world `t`
/// mnemonic lead, replace dots with underscores and prepend `pto_vpto_`
/// (`pto.tmatmul.mx.acc` -> `pto_vpto_matmul_mx_acc`).
std::string deriveDefaultBridgeEntry(llvm::StringRef opName);

/// Derives the default C++ call spelling of a declarative routed entry
/// from its IR op name: strip the `pto.` prefix (keeping the tile-world
/// `t` mnemonic lead the entry name drops), uppercase the remainder,
/// replace dots with underscores and qualify with `pto::`
/// (`pto.tadd` -> `pto::TADD`, `pto.tmatmul.mx` -> `pto::TMATMUL_MX`).
/// Variants whose interface call does not follow the convention declare
/// `call` explicitly.
std::string deriveDefaultBridgeCall(llvm::StringRef opName);

/// Renders the lowerCamelCase parameter name of an abi role
/// (`left_tile` -> `leftTile`, `a_scale_tile` -> `aScaleTile`), the
/// default `arg` of declarative abi bindings.
std::string bridgeRoleParamName(llvm::StringRef role);

/// Renders the CamelCase typedef target name of an abi role
/// (`left_tile` -> `LeftTile`, `a_scale_tile` -> `AScaleTile`). Tile
/// typedefs of declarative entries are role driven, so this is the single
/// source of the typedef names the wrapper bodies reference.
std::string bridgeRoleTypedefTarget(llvm::StringRef role);

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_VPTOBRIDGEWHITELIST_H

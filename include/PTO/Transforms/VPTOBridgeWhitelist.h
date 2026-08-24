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

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <vector>

namespace mlir {
namespace pto {

/// ABI argument of a wrapper entry. `type` is one of the supported carrier
/// tokens: "ptr", "i64", or "i32". Declarative entries additionally bind
/// each argument to an IR operand position (`operand`, positional because
/// MLIR exposes no generic ODS operand-name reflection), carry a
/// diagnostic label (`arg`, the ODS operand name) and the template role
/// the operand's tile token is collected under (`role`, which is also the
/// spec key and a valid tmpl_map source).
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
/// VPTOBridgeTokens. For declarative entries the tile sources name abi
/// roles; `source: attr` maps an enum attribute to a template slot, with
/// `enumType` providing the qualified C++ enum spelling and `omitValue`
/// the case that renders no template argument (e.g. an Unspecified
/// accumulation phase).
struct BridgeTmplMapField {
  std::string source;
  std::string field;
  std::string target;
  std::string enumType;
  std::string omitValue;
};

/// One whitelist row: an IR op routed to a wrapper entry of a PTO-ISA
/// interface family.
struct BridgeWhitelistEntry {
  /// IR op name, e.g. "pto.tpush". The whitelist is the routing table the
  /// family passes consult; "internal" marks wrapper-internal helpers
  /// (e.g. the size query entry) that are never routed from an IR op.
  std::string op;
  /// Interface family, e.g. "pipe". Selects the family pass and the wrapper
  /// template.
  std::string family;
  /// Lowering channel: "declarative" routes the op through the generic
  /// declarative bridge lowering (mechanical operand-adapter mapping, no
  /// family pass); "family" (the default) requires the family pass to
  /// rewrite the op into bridge ops.
  std::string lowering = "family";
  /// Wrapper entry name, e.g. "pto_vpto_pipe_push". This is the callee the
  /// generic bridge lowering emits.
  std::string entry;
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

  /// `lowering` value routing the op through the generic declarative
  /// bridge lowering instead of a family pass.
  static constexpr llvm::StringLiteral kLoweringDeclarative = "declarative";
  /// `lowering` value (the default) keeping the op on a family pass.
  static constexpr llvm::StringLiteral kLoweringFamily = "family";
};

/// Parsed whitelist document.
struct BridgeWhitelist {
  std::vector<BridgeWhitelistEntry> bridgeOps;

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

/// Loads the bridge whitelist through the formal resolution chain: pass
/// `whitelist-path` option, then PTOAS_VPTO_BRIDGE_WHITELIST, then the
/// built-in default whitelist (pipe + matmul families) shipped with ptoas.
/// Always returns a parsed whitelist unless the explicitly configured file
/// fails to parse.
FailureOr<BridgeWhitelist> loadBridgeWhitelist(llvm::StringRef optionValue,
                                               llvm::raw_ostream &diagOS);

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_VPTOBRIDGEWHITELIST_H

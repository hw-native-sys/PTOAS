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
/// tokens: "ptr", "i64", or "i32".
struct BridgeAbiArg {
  std::string type;
};

/// One whitelist row: an IR op routed to a wrapper entry of a PTO-ISA
/// interface family.
struct BridgeWhitelistEntry {
  /// IR op name, e.g. "pto.tpush". Routing metadata; the generic lowering
  /// validates bridge calls by `entry`, and wrapper generation consumes the
  /// full row.
  std::string op;
  /// Interface family, e.g. "pipe". Selects the family pass and the wrapper
  /// template.
  std::string family;
  /// Wrapper entry name, e.g. "pto_vpto_pipe_push". This is the callee the
  /// generic bridge lowering emits.
  std::string entry;
  /// Call-side ABI of the wrapper entry, including any synthesized
  /// arguments such as the storage pointer of stateful entries.
  std::vector<BridgeAbiArg> abi;
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
};

/// Parses a whitelist YAML file. Diagnostics are written to `diagOS`.
/// Rejects unreadable files, YAML syntax errors, duplicate wrapper entry
/// names, and unsupported ABI type tokens.
FailureOr<BridgeWhitelist> parseBridgeWhitelist(llvm::StringRef path,
                                                llvm::raw_ostream &diagOS);

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_VPTOBRIDGEWHITELIST_H

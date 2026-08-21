// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOBridgeWhitelist.cpp - C++ bridge whitelist ---------------------===//
//===----------------------------------------------------------------------===//
//
// YAML parsing and semantic validation for the VPTO C++ interface bridge
// whitelist (see include/PTO/Transforms/VPTOBridgeWhitelist.h).
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/VPTOBridgeWhitelist.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/YAMLTraits.h"

using namespace mlir;
using namespace mlir::pto;

namespace llvm {
namespace yaml {

template <> struct MappingTraits<BridgeAbiArg> {
  static void mapping(IO &io, BridgeAbiArg &arg) {
    io.mapRequired("type", arg.type);
  }
};

template <> struct MappingTraits<BridgeWhitelistEntry> {
  static void mapping(IO &io, BridgeWhitelistEntry &entry) {
    io.mapRequired("op", entry.op);
    io.mapRequired("family", entry.family);
    io.mapRequired("entry", entry.entry);
    io.mapOptional("abi", entry.abi);
  }
};

template <> struct MappingTraits<BridgeWhitelist> {
  static void mapping(IO &io, BridgeWhitelist &whitelist) {
    io.mapRequired("bridge_ops", whitelist.bridgeOps);
  }
};

} // namespace yaml
} // namespace llvm

LLVM_YAML_IS_SEQUENCE_VECTOR(BridgeAbiArg)
LLVM_YAML_IS_SEQUENCE_VECTOR(BridgeWhitelistEntry)

namespace {

/// ABI carrier tokens accepted by the generic bridge lowering. The token set
/// is intentionally small: it must stay a closed list so that both the
/// lowering and the future wrapper generator agree on the carriers.
bool isSupportedAbiType(StringRef type) {
  return type == "ptr" || type == "i64" || type == "i32";
}

} // namespace

FailureOr<BridgeWhitelist>
pto::parseBridgeWhitelist(llvm::StringRef path, llvm::raw_ostream &diagOS) {
  auto bufferOr = llvm::MemoryBuffer::getFile(path);
  if (!bufferOr) {
    diagOS << "VPTO bridge whitelist: cannot read '" << path
           << "': " << bufferOr.getError().message() << "\n";
    return failure();
  }

  BridgeWhitelist whitelist;
  llvm::yaml::Input input(bufferOr.get()->getBuffer());
  input >> whitelist;
  if (std::error_code error = input.error()) {
    diagOS << "VPTO bridge whitelist: cannot parse '" << path
           << "': " << error.message() << "\n";
    return failure();
  }

  llvm::StringSet<> seenEntries;
  for (const BridgeWhitelistEntry &entry : whitelist.bridgeOps) {
    if (!seenEntries.insert(entry.entry).second) {
      diagOS << "VPTO bridge whitelist: duplicate wrapper entry '"
             << entry.entry << "' in '" << path << "'\n";
      return failure();
    }
    for (const BridgeAbiArg &arg : entry.abi) {
      if (!isSupportedAbiType(arg.type)) {
        diagOS << "VPTO bridge whitelist: unsupported ABI type token '"
               << arg.type << "' for entry '" << entry.entry << "' in '"
               << path << "' (supported: ptr, i64, i32)\n";
        return failure();
      }
    }
  }
  return whitelist;
}

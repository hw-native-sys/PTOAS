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
#include <cstdlib>

using namespace mlir;
using namespace mlir::pto;

namespace llvm {
namespace yaml {

template <> struct MappingTraits<BridgeAbiArg> {
  static void mapping(IO &io, BridgeAbiArg &arg) {
    io.mapRequired("type", arg.type);
  }
};

template <> struct MappingTraits<BridgeTmplMapField> {
  static void mapping(IO &io, BridgeTmplMapField &field) {
    io.mapRequired("source", field.source);
    io.mapRequired("field", field.field);
    io.mapRequired("target", field.target);
  }
};

template <> struct MappingTraits<BridgeWhitelistEntry> {
  static void mapping(IO &io, BridgeWhitelistEntry &entry) {
    io.mapRequired("op", entry.op);
    io.mapRequired("family", entry.family);
    io.mapRequired("entry", entry.entry);
    io.mapOptional("abi", entry.abi);
    io.mapOptional("storage_size_entry", entry.storageSizeEntry);
    io.mapOptional("tmpl_map", entry.tmplMap);
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
LLVM_YAML_IS_SEQUENCE_VECTOR(BridgeTmplMapField)
LLVM_YAML_IS_SEQUENCE_VECTOR(BridgeWhitelistEntry)

namespace {

/// ABI carrier tokens accepted by the generic bridge lowering. The token set
/// is intentionally small: it must stay a closed list so that both the
/// lowering and the future wrapper generator agree on the carriers.
bool isSupportedAbiType(StringRef type) {
  return type == "ptr" || type == "i64" || type == "i32";
}

/// tmpl_map `source` tokens accepted for the pipe family. A source names the
/// IR producer of a template argument: the pipe init op attributes or a tile
/// operand's type.
bool isPipeTmplMapSource(StringRef source) {
  return source == "pipe.init" || source == "tile";
}

/// tmpl_map `source` tokens accepted for the matmul family. The tile types
/// are spread over the matmul operand roles, so each role is its own
/// source.
bool isMatmulTmplMapSource(StringRef source) {
  return source == "left_tile" || source == "right_tile" ||
         source == "result_tile" || source == "acc_in_tile";
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
  llvm::StringSet<> seenOps;
  for (const BridgeWhitelistEntry &entry : whitelist.bridgeOps) {
    if (entry.op.empty() || entry.family.empty() || entry.entry.empty()) {
      diagOS << "VPTO bridge whitelist: entry with op='" << entry.op
             << "', family='" << entry.family << "', entry='" << entry.entry
             << "' has an empty required field in '" << path << "'\n";
      return failure();
    }
    if (!seenEntries.insert(entry.entry).second) {
      diagOS << "VPTO bridge whitelist: duplicate wrapper entry '"
             << entry.entry << "' in '" << path << "'\n";
      return failure();
    }
    if (entry.op != BridgeWhitelist::kInternalOp &&
        !seenOps.insert(entry.op).second) {
      diagOS << "VPTO bridge whitelist: duplicate routed op '" << entry.op
             << "' in '" << path << "'\n";
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
    for (const BridgeTmplMapField &field : entry.tmplMap) {
      if (field.source.empty() || field.field.empty() ||
          field.target.empty()) {
        diagOS << "VPTO bridge whitelist: tmpl_map row of entry '"
               << entry.entry << "' has an empty source/field/target in '"
               << path << "'\n";
        return failure();
      }
      if (entry.family == "pipe" && !isPipeTmplMapSource(field.source)) {
        diagOS << "VPTO bridge whitelist: tmpl_map row of entry '"
               << entry.entry << "' uses unknown pipe-family source '"
               << field.source << "' in '" << path
               << "' (supported: pipe.init, tile)\n";
        return failure();
      }
      if (entry.family == "matmul" &&
          !isMatmulTmplMapSource(field.source)) {
        diagOS << "VPTO bridge whitelist: tmpl_map row of entry '"
               << entry.entry << "' uses unknown matmul-family source '"
               << field.source << "' in '" << path
               << "' (supported: left_tile, right_tile, result_tile, "
                  "acc_in_tile)\n";
        return failure();
      }
    }
  }
  for (const BridgeWhitelistEntry &entry : whitelist.bridgeOps) {
    if (!entry.storageSizeEntry.empty() &&
        !whitelist.findEntry(entry.storageSizeEntry)) {
      diagOS << "VPTO bridge whitelist: entry '" << entry.entry
             << "' declares storage_size_entry '" << entry.storageSizeEntry
             << "' which is not a declared wrapper entry in '" << path
             << "'\n";
      return failure();
    }
  }
  return whitelist;
}

std::string pto::resolveBridgeWhitelistPath(llvm::StringRef optionValue) {
  if (!optionValue.empty()) {
    return std::string(optionValue);
  }
  if (const char *envPath = std::getenv("PTOAS_VPTO_BRIDGE_WHITELIST")) {
    return envPath;
  }
  return {};
}

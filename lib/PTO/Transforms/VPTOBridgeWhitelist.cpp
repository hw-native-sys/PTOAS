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
#include "llvm/ADT/DenseSet.h"
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
    io.mapOptional("operand", arg.operand, (int64_t)-1);
    io.mapOptional("arg", arg.arg);
    io.mapOptional("role", arg.role);
  }
};

template <> struct MappingTraits<BridgeTmplMapField> {
  static void mapping(IO &io, BridgeTmplMapField &field) {
    io.mapRequired("source", field.source);
    io.mapRequired("field", field.field);
    io.mapRequired("target", field.target);
    io.mapOptional("enum_type", field.enumType);
    io.mapOptional("omit_value", field.omitValue);
  }
};

template <> struct MappingTraits<BridgeWhitelistEntry> {
  static void mapping(IO &io, BridgeWhitelistEntry &entry) {
    io.mapRequired("op", entry.op);
    io.mapRequired("family", entry.family);
    io.mapOptional("lowering", entry.lowering,
                   std::string(BridgeWhitelistEntry::kLoweringFamily));
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

/// tmpl_map `source` token mapping an enum attribute of the routed op to a
/// template slot. The token spelling is assembled from the whitelist
/// `enum_type` prefix and the attribute's enum case symbol.
constexpr llvm::StringLiteral kAttrTmplMapSource = "attr";

} // namespace

FailureOr<BridgeWhitelist>
pto::parseBridgeWhitelistFromBuffer(llvm::StringRef content,
                                    llvm::StringRef sourceName,
                                    llvm::raw_ostream &diagOS) {
  BridgeWhitelist whitelist;
  llvm::yaml::Input input(content);
  input >> whitelist;
  if (std::error_code error = input.error()) {
    diagOS << "VPTO bridge whitelist: cannot parse '" << sourceName
           << "': " << error.message() << "\n";
    return failure();
  }

  llvm::StringSet<> seenEntries;
  llvm::StringSet<> seenOps;
  for (const BridgeWhitelistEntry &entry : whitelist.bridgeOps) {
    if (entry.op.empty() || entry.family.empty() || entry.entry.empty()) {
      diagOS << "VPTO bridge whitelist: entry with op='" << entry.op
             << "', family='" << entry.family << "', entry='" << entry.entry
             << "' has an empty required field in '" << sourceName << "'\n";
      return failure();
    }
    if (entry.lowering != BridgeWhitelistEntry::kLoweringDeclarative &&
        entry.lowering != BridgeWhitelistEntry::kLoweringFamily) {
      diagOS << "VPTO bridge whitelist: entry '" << entry.entry
             << "' declares unsupported lowering '" << entry.lowering
             << "' in '" << sourceName << "' (supported: declarative, "
                "family)\n";
      return failure();
    }
    if (!seenEntries.insert(entry.entry).second) {
      diagOS << "VPTO bridge whitelist: duplicate wrapper entry '"
             << entry.entry << "' in '" << sourceName << "'\n";
      return failure();
    }
    if (entry.op != BridgeWhitelist::kInternalOp &&
        !seenOps.insert(entry.op).second) {
      diagOS << "VPTO bridge whitelist: duplicate routed op '" << entry.op
             << "' in '" << sourceName << "'\n";
      return failure();
    }
    // Declarative entries bind every abi argument to an IR operand position
    // and a template role; the role set is the valid source set of the
    // entry's tmpl_map tile rows.
    llvm::StringSet<> declarativeRoles;
    llvm::DenseSet<int64_t> declarativeOperands;
    if (entry.isDeclarative()) {
      for (const BridgeAbiArg &arg : entry.abi) {
        if (arg.operand < 0 || arg.arg.empty() || arg.role.empty()) {
          diagOS << "VPTO bridge whitelist: declarative entry '" << entry.entry
                 << "' has an abi argument without operand/arg/role binding "
                    "in '"
                 << sourceName << "'\n";
          return failure();
        }
        if (!declarativeOperands.insert(arg.operand).second) {
          diagOS << "VPTO bridge whitelist: declarative entry '" << entry.entry
                 << "' binds operand #" << arg.operand
                 << " more than once in '" << sourceName << "'\n";
          return failure();
        }
        declarativeRoles.insert(arg.role);
      }
    }
    for (const BridgeAbiArg &arg : entry.abi) {
      if (!isSupportedAbiType(arg.type)) {
        diagOS << "VPTO bridge whitelist: unsupported ABI type token '"
               << arg.type << "' for entry '" << entry.entry << "' in '"
               << sourceName << "' (supported: ptr, i64, i32)\n";
        return failure();
      }
    }
    for (const BridgeTmplMapField &field : entry.tmplMap) {
      if (field.source.empty() || field.field.empty() ||
          field.target.empty()) {
        diagOS << "VPTO bridge whitelist: tmpl_map row of entry '"
               << entry.entry << "' has an empty source/field/target in '"
               << sourceName << "'\n";
        return failure();
      }
      if (entry.isDeclarative()) {
        // Declarative sources are structural: tile rows name abi roles, the
        // attr row maps an enum attribute and must declare its C++ enum.
        if (field.source == kAttrTmplMapSource) {
          if (field.enumType.empty()) {
            diagOS << "VPTO bridge whitelist: tmpl_map attr row of entry '"
                   << entry.entry << "' lacks enum_type in '" << sourceName
                   << "'\n";
            return failure();
          }
        } else if (!declarativeRoles.count(field.source)) {
          diagOS << "VPTO bridge whitelist: tmpl_map row of entry '"
                 << entry.entry << "' uses source '" << field.source
                 << "' which is not an abi role of the declarative entry "
                    "(roles: ";
          llvm::ListSeparator sep;
          for (const BridgeAbiArg &arg : entry.abi) {
            diagOS << sep << arg.role;
          }
          diagOS << ") in '" << sourceName << "'\n";
          return failure();
        }
      } else if (entry.family == "pipe" &&
                 !isPipeTmplMapSource(field.source)) {
        diagOS << "VPTO bridge whitelist: tmpl_map row of entry '"
               << entry.entry << "' uses unknown pipe-family source '"
               << field.source << "' in '" << sourceName
               << "' (supported: pipe.init, tile)\n";
        return failure();
      }
    }
  }
  for (const BridgeWhitelistEntry &entry : whitelist.bridgeOps) {
    if (!entry.storageSizeEntry.empty() &&
        !whitelist.findEntry(entry.storageSizeEntry)) {
      diagOS << "VPTO bridge whitelist: entry '" << entry.entry
             << "' declares storage_size_entry '" << entry.storageSizeEntry
             << "' which is not a declared wrapper entry in '" << sourceName
             << "'\n";
      return failure();
    }
  }
  return whitelist;
}

FailureOr<BridgeWhitelist>
pto::parseBridgeWhitelist(llvm::StringRef path, llvm::raw_ostream &diagOS) {
  auto bufferOr = llvm::MemoryBuffer::getFile(path);
  if (!bufferOr) {
    diagOS << "VPTO bridge whitelist: cannot read '" << path
           << "': " << bufferOr.getError().message() << "\n";
    return failure();
  }
  return parseBridgeWhitelistFromBuffer(bufferOr.get()->getBuffer(), path,
                                        diagOS);
}

/// The built-in default whitelist covering the interface families bridged
/// today: the pipe family (C2V/V2C fifo) and the matmul family (TMATMUL /
/// TMATMUL_ACC and the bias/MX entry variants). It keeps `ptoas
/// --pto-backend=vpto` working out of the box; an explicit whitelist (pass
/// option or PTOAS_VPTO_BRIDGE_WHITELIST) always overrides it. End-to-end
/// cases under test/vpto/cases/kernels/ rely on this default, so adding a
/// bridged family requires extending it here.
/// Variant entries share the wrapper's one tile configuration, so every
/// matmul entry declares the full set of role tiles it renders; duplicate
/// targets deduplicate at render time.
/// The matmul entries use the declarative lowering channel: each abi row
/// binds a wrapper argument to an IR operand position and a tile role, and
/// an optional attr tmpl_map row maps the accPhase enum attribute.
static constexpr llvm::StringLiteral kDefaultBridgeWhitelistYaml = R"yaml(
bridge_ops:
  - op: pto.initialize_l2l_pipe
    family: pipe
    entry: pto_vpto_pipe_init
    storage_size_entry: pto_vpto_pipe_size
    abi:
      - type: ptr    # storage, synthesized by the bridge lowering
      - type: i32    # consumer local buffer address
    tmpl_map:
      - source: pipe.init
        field: pipe
        target: Pipe
  - op: pto.tpush
    family: pipe
    entry: pto_vpto_pipe_push
    abi:
      - type: ptr    # storage
      - type: i64    # producer tile address
    tmpl_map:
      - source: tile
        field: tile
        target: ProducerTile
  - op: pto.tpop
    family: pipe
    entry: pto_vpto_pipe_pop
    abi:
      - type: ptr    # storage
    tmpl_map:
      - source: tile
        field: tile
        target: ConsumerTile
  - op: pto.tfree
    family: pipe
    entry: pto_vpto_pipe_free
    abi:
      - type: ptr    # storage
  - op: internal    # wrapper-internal helper, not routed from an IR op
    family: pipe
    entry: pto_vpto_pipe_size
    abi: []
  - op: pto.tmatmul
    family: matmul
    lowering: declarative
    entry: pto_vpto_matmul
    abi:
      - {operand: 2, arg: dst, type: i64, role: result_tile}
      - {operand: 0, arg: lhs, type: i64, role: left_tile}
      - {operand: 1, arg: rhs, type: i64, role: right_tile}
    tmpl_map:
      - source: left_tile
        field: tile
        target: LeftTile
      - source: right_tile
        field: tile
        target: RightTile
      - source: result_tile
        field: tile
        target: ResultTile
      - source: attr
        field: acc_phase
        target: AccPhase
        enum_type: pto::AccPhase
        omit_value: Unspecified
  - op: pto.tmatmul.acc
    family: matmul
    lowering: declarative
    entry: pto_vpto_matmul_acc
    abi:
      - {operand: 3, arg: dst, type: i64, role: result_tile}
      - {operand: 0, arg: acc_in, type: i64, role: acc_in_tile}
      - {operand: 1, arg: lhs, type: i64, role: left_tile}
      - {operand: 2, arg: rhs, type: i64, role: right_tile}
    tmpl_map:
      - source: acc_in_tile
        field: tile
        target: AccInTile
      - source: attr
        field: acc_phase
        target: AccPhase
        enum_type: pto::AccPhase
        omit_value: Unspecified
  - op: pto.tmatmul.bias
    family: matmul
    lowering: declarative
    entry: pto_vpto_matmul_bias
    abi:
      - {operand: 3, arg: dst, type: i64, role: result_tile}
      - {operand: 0, arg: a, type: i64, role: left_tile}
      - {operand: 1, arg: b, type: i64, role: right_tile}
      - {operand: 2, arg: bias, type: i64, role: bias_tile}
    tmpl_map:
      - source: left_tile
        field: tile
        target: LeftTile
      - source: right_tile
        field: tile
        target: RightTile
      - source: result_tile
        field: tile
        target: ResultTile
      - source: bias_tile
        field: tile
        target: BiasTile
      - source: attr
        field: acc_phase
        target: AccPhase
        enum_type: pto::AccPhase
        omit_value: Unspecified
  - op: pto.tmatmul.mx
    family: matmul
    lowering: declarative
    entry: pto_vpto_matmul_mx
    abi:
      - {operand: 4, arg: dst, type: i64, role: result_tile}
      - {operand: 0, arg: a, type: i64, role: left_tile}
      - {operand: 1, arg: a_scale, type: i64, role: a_scale_tile}
      - {operand: 2, arg: b, type: i64, role: right_tile}
      - {operand: 3, arg: b_scale, type: i64, role: b_scale_tile}
    tmpl_map:
      - source: left_tile
        field: tile
        target: LeftTile
      - source: right_tile
        field: tile
        target: RightTile
      - source: result_tile
        field: tile
        target: ResultTile
      - source: a_scale_tile
        field: tile
        target: AScaleTile
      - source: b_scale_tile
        field: tile
        target: BScaleTile
      - source: attr
        field: acc_phase
        target: AccPhase
        enum_type: pto::AccPhase
        omit_value: Unspecified
  - op: pto.tmatmul.mx.acc
    family: matmul
    lowering: declarative
    entry: pto_vpto_matmul_mx_acc
    abi:
      - {operand: 5, arg: dst, type: i64, role: result_tile}
      - {operand: 0, arg: c_in, type: i64, role: acc_in_tile}
      - {operand: 1, arg: a, type: i64, role: left_tile}
      - {operand: 2, arg: a_scale, type: i64, role: a_scale_tile}
      - {operand: 3, arg: b, type: i64, role: right_tile}
      - {operand: 4, arg: b_scale, type: i64, role: b_scale_tile}
    tmpl_map:
      - source: left_tile
        field: tile
        target: LeftTile
      - source: right_tile
        field: tile
        target: RightTile
      - source: result_tile
        field: tile
        target: ResultTile
      - source: acc_in_tile
        field: tile
        target: AccInTile
      - source: a_scale_tile
        field: tile
        target: AScaleTile
      - source: b_scale_tile
        field: tile
        target: BScaleTile
      - source: attr
        field: acc_phase
        target: AccPhase
        enum_type: pto::AccPhase
        omit_value: Unspecified
  - op: pto.tmatmul.mx.bias
    family: matmul
    lowering: declarative
    entry: pto_vpto_matmul_mx_bias
    abi:
      - {operand: 5, arg: dst, type: i64, role: result_tile}
      - {operand: 0, arg: a, type: i64, role: left_tile}
      - {operand: 1, arg: a_scale, type: i64, role: a_scale_tile}
      - {operand: 2, arg: b, type: i64, role: right_tile}
      - {operand: 3, arg: b_scale, type: i64, role: b_scale_tile}
      - {operand: 4, arg: bias, type: i64, role: bias_tile}
    tmpl_map:
      - source: left_tile
        field: tile
        target: LeftTile
      - source: right_tile
        field: tile
        target: RightTile
      - source: result_tile
        field: tile
        target: ResultTile
      - source: a_scale_tile
        field: tile
        target: AScaleTile
      - source: b_scale_tile
        field: tile
        target: BScaleTile
      - source: bias_tile
        field: tile
        target: BiasTile
)yaml";

std::string pto::resolveBridgeWhitelistPath(llvm::StringRef optionValue) {
  if (!optionValue.empty()) {
    return std::string(optionValue);
  }
  if (const char *envPath = std::getenv("PTOAS_VPTO_BRIDGE_WHITELIST")) {
    return envPath;
  }
  return {};
}

FailureOr<BridgeWhitelist>
pto::loadBridgeWhitelist(llvm::StringRef optionValue,
                         llvm::raw_ostream &diagOS) {
  std::string path = resolveBridgeWhitelistPath(optionValue);
  if (!path.empty()) {
    return parseBridgeWhitelist(path, diagOS);
  }
  return parseBridgeWhitelistFromBuffer(
      kDefaultBridgeWhitelistYaml, "<built-in vpto bridge whitelist>",
      diagOS);
}

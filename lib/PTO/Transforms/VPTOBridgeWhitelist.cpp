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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/YAMLTraits.h"
#include <algorithm>
#include <cstdlib>

using namespace mlir;
using namespace mlir::pto;

namespace llvm {
namespace yaml {

template <> struct MappingTraits<BridgeAbiArg> {
  static void mapping(IO &io, BridgeAbiArg &arg) {
    io.mapOptional("type", arg.type);
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
    io.mapRequired("wrapper", entry.wrapper);
    io.mapOptional("lowering", entry.lowering,
                   std::string(BridgeWhitelistEntry::kLoweringDeclarative));
    io.mapOptional("entry", entry.entry);
    io.mapOptional("call", entry.call);
    io.mapOptional("tmpl_args", entry.tmplArgs);
    io.mapOptional("abi", entry.abi);
    io.mapOptional("storage_size_entry", entry.storageSizeEntry);
    io.mapOptional("tmpl_map", entry.tmplMap);
  }
};

template <> struct MappingTraits<BridgeWrapperDecl> {
  static void mapping(IO &io, BridgeWrapperDecl &decl) {
    io.mapRequired("name", decl.name);
    io.mapRequired("includes", decl.includes);
    io.mapOptional("core", decl.core);
  }
};

template <> struct MappingTraits<BridgeWhitelist> {
  static void mapping(IO &io, BridgeWhitelist &whitelist) {
    io.mapOptional("wrappers", whitelist.wrappers);
    io.mapRequired("bridge_ops", whitelist.bridgeOps);
  }
};

} // namespace yaml
} // namespace llvm

LLVM_YAML_IS_SEQUENCE_VECTOR(BridgeAbiArg)
LLVM_YAML_IS_SEQUENCE_VECTOR(BridgeTmplMapField)
LLVM_YAML_IS_SEQUENCE_VECTOR(BridgeWhitelistEntry)
LLVM_YAML_IS_SEQUENCE_VECTOR(BridgeWrapperDecl)

namespace {

/// tmpl_map `source` tokens accepted for the pipe wrapper. A source names
/// the IR producer of a template argument: the pipe init op attributes or a
/// tile operand's type.
bool isPipeTmplMapSource(StringRef source) {
  return source == kPipeInitTmplMapSource || source == kTileTmplMapSource;
}

} // namespace

bool pto::isSupportedBridgeAbiType(llvm::StringRef token) {
  return token == "ptr" || token == "i64" || token == "i32";
}

bool pto::bridgeAbiTypeMatches(llvm::StringRef token, Type type) {
  if (token == "ptr") {
    return isa<LLVM::LLVMPointerType>(type);
  }
  if (token == "i64") {
    return type.isInteger(64);
  }
  if (token == "i32") {
    return type.isInteger(32);
  }
  return false;
}

std::string pto::deriveDefaultBridgeEntry(llvm::StringRef opName) {
  constexpr llvm::StringLiteral kTileWorldOpPrefix = "pto.t";
  if (!opName.consume_front(kTileWorldOpPrefix)) {
    opName.consume_front("pto.");
  }
  std::string name = ("pto_vpto_" + opName).str();
  constexpr llvm::StringLiteral kNamePrefix = "pto_vpto_";
  std::replace(name.begin() + kNamePrefix.size(), name.end(), '.', '_');
  return name;
}

std::string pto::deriveDefaultBridgeCall(llvm::StringRef opName) {
  // Unlike the entry name, the interface call keeps the tile-world `t`
  // mnemonic lead (pto::TADD, pto::TMATMUL), so only the dialect prefix
  // is stripped.
  opName.consume_front("pto.");
  if (opName.empty()) {
    return {};
  }
  std::string name;
  name.reserve(opName.size());
  for (char c : opName) {
    name.push_back(c == '.' ? '_'
                            : static_cast<char>(llvm::toUpper(c)));
  }
  return "pto::" + name;
}

std::string pto::bridgeRoleParamName(llvm::StringRef role) {
  std::string name;
  name.reserve(role.size());
  bool upperNext = false;
  for (char c : role) {
    if (c == '_') {
      upperNext = true;
      continue;
    }
    name.push_back(upperNext && c >= 'a' && c <= 'z'
                       ? static_cast<char>(c - 'a' + 'A')
                       : c);
    upperNext = false;
  }
  return name;
}

std::string pto::bridgeRoleTypedefTarget(llvm::StringRef role) {
  std::string target;
  target.reserve(role.size());
  bool upperNext = true;
  for (char c : role) {
    if (c == '_') {
      upperNext = true;
      continue;
    }
    target.push_back(upperNext && c >= 'a' && c <= 'z'
                         ? static_cast<char>(c - 'a' + 'A')
                         : c);
    upperNext = false;
  }
  return target;
}

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
  for (BridgeWhitelistEntry &entry : whitelist.bridgeOps) {
    if (entry.op.empty() || entry.wrapper.empty()) {
      diagOS << "VPTO bridge whitelist: entry with op='" << entry.op
             << "', wrapper='" << entry.wrapper << "', entry='" << entry.entry
             << "' has an empty required field in '" << sourceName << "'\n";
      return failure();
    }
    if (entry.lowering != BridgeWhitelistEntry::kLoweringDeclarative &&
        entry.lowering != BridgeWhitelistEntry::kLoweringCustom) {
      diagOS << "VPTO bridge whitelist: entry '" << entry.entry
             << "' declares unsupported lowering '" << entry.lowering
             << "' in '" << sourceName << "' (supported: declarative, "
                "custom)\n";
      return failure();
    }
    // Routed declarative entries default their entry name from the op
    // name; custom entries and wrapper-internal helpers are named by hand
    // because no mechanical rule covers them.
    const bool declarativeChannel =
        entry.isDeclarative() && entry.op != BridgeWhitelist::kInternalOp;
    if (entry.entry.empty()) {
      if (declarativeChannel) {
        entry.entry = deriveDefaultBridgeEntry(entry.op);
      } else {
        diagOS << "VPTO bridge whitelist: entry with op='" << entry.op
               << "' declares no entry name in '" << sourceName
               << "' (only declarative entries default it from the op "
                  "name)\n";
        return failure();
      }
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
    // The call spelling and template arguments belong to the generic
    // declarative renderer; entries owned by a dedicated pass (or a
    // wrapper-internal helper) must not carry them.
    if (!declarativeChannel && (!entry.call.empty() || !entry.tmplArgs.empty())) {
      diagOS << "VPTO bridge whitelist: entry '" << entry.entry
             << "' declares call/tmpl_args but is not lowered through the "
                "declarative channel in '"
             << sourceName << "'\n";
      return failure();
    }
    if (declarativeChannel) {
      // The call spelling follows the op-name convention unless declared;
      // a derivation the interface does not follow is overridden with an
      // explicit `call`, and a wrong one fails loudly when the generated
      // wrapper source is compiled.
      if (entry.call.empty()) {
        entry.call = deriveDefaultBridgeCall(entry.op);
      }
      if (entry.call.empty()) {
        diagOS << "VPTO bridge whitelist: declarative entry '" << entry.entry
               << "' declares no call spelling in '" << sourceName
               << "' (the generic renderer needs the C++ call the entry "
                  "body emits)\n";
        return failure();
      }
      for (const std::string &tmplArg : entry.tmplArgs) {
        if (tmplArg.empty()) {
          diagOS << "VPTO bridge whitelist: declarative entry '"
                 << entry.entry << "' has an empty tmpl_args item in '"
                 << sourceName << "'\n";
          return failure();
        }
        // A qualified spelling is a literal template argument; anything
        // else must name an attr tmpl_map row of this entry whose
        // collected spec token feeds the slot.
        if (llvm::StringRef(tmplArg).contains("::")) {
          continue;
        }
        bool attrFieldDeclared = false;
        for (const BridgeTmplMapField &field : entry.tmplMap) {
          if (field.source == kAttrTmplMapSource && field.field == tmplArg) {
            attrFieldDeclared = true;
            break;
          }
        }
        if (!attrFieldDeclared) {
          diagOS << "VPTO bridge whitelist: declarative entry '"
                 << entry.entry << "' tmpl_args item '" << tmplArg
                 << "' is neither a qualified literal nor an attr "
                    "tmpl_map field of the entry in '"
                 << sourceName << "'\n";
          return failure();
        }
      }
    }
    // Declarative entries bind every abi argument to an IR operand position
    // and a template role; the role set names the entry's tile typedefs.
    // Wrapper-internal helpers are never routed from an IR op, so they
    // carry no operand bindings and the `lowering` value is meaningless
    // for them.
    llvm::StringSet<> declarativeRoles;
    llvm::DenseSet<int64_t> declarativeOperands;
    if (declarativeChannel) {
      for (BridgeAbiArg &arg : entry.abi) {
        if (arg.operand < 0 || arg.role.empty()) {
          // The declarative channel is the default, so the likeliest cause
          // is an entry that needs a dedicated pass but never opted out.
          diagOS << "VPTO bridge whitelist: declarative entry '" << entry.entry
                 << "' has an abi argument without operand/role binding "
                    "in '"
                 << sourceName
                 << "' (entries owned by a dedicated family pass must "
                    "declare 'lowering: custom')\n";
          return failure();
        }
        // The parameter name is a rendering concern: it defaults to the
        // lowerCamelCase of the role, tying the wrapper parameter to its
        // tile typedef.
        if (arg.arg.empty()) {
          arg.arg = bridgeRoleParamName(arg.role);
        }
        if (!declarativeOperands.insert(arg.operand).second) {
          diagOS << "VPTO bridge whitelist: declarative entry '" << entry.entry
                 << "' binds operand #" << arg.operand
                 << " more than once in '" << sourceName << "'\n";
          return failure();
        }
        declarativeRoles.insert(arg.role);
        // Tile addresses are the only carrier the declarative channel
        // emits, so the type defaults to i64 when omitted.
        if (arg.type.empty()) {
          arg.type = "i64";
        }
      }
    }
    for (const BridgeAbiArg &arg : entry.abi) {
      if (!isSupportedBridgeAbiType(arg.type)) {
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
      if (declarativeChannel) {
        // Declarative tile typedefs derive from the abi roles, so the only
        // tmpl_map rows the channel accepts map enum attributes; a tile
        // row here is the legacy spelling and would name a typedef target
        // the role-driven renderer never emits.
        if (field.source != kAttrTmplMapSource) {
          diagOS << "VPTO bridge whitelist: tmpl_map row of entry '"
                 << entry.entry << "' uses source '" << field.source
                 << "', but declarative entries only accept 'attr' rows "
                    "(tile typedefs derive from the abi roles) in '"
                 << sourceName << "'\n";
          return failure();
        }
        if (field.enumType.empty()) {
          diagOS << "VPTO bridge whitelist: tmpl_map attr row of entry '"
                 << entry.entry << "' lacks enum_type in '" << sourceName
                 << "'\n";
          return failure();
        }
      } else if (entry.wrapper == "pipe" &&
                 !isPipeTmplMapSource(field.source)) {
        diagOS << "VPTO bridge whitelist: tmpl_map row of entry '"
               << entry.entry << "' uses unknown pipe-wrapper source '"
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
  // Wrapper declarations feed the generic declarative renderer: every
  // declared wrapper must own at least one declarative routed entry and no
  // custom entry (custom wrappers own a dedicated renderer).
  llvm::StringSet<> seenWrappers;
  for (const BridgeWrapperDecl &decl : whitelist.wrappers) {
    if (decl.name.empty()) {
      diagOS << "VPTO bridge whitelist: wrapper declaration with an empty "
                "name in '"
             << sourceName << "'\n";
      return failure();
    }
    if (!seenWrappers.insert(decl.name).second) {
      diagOS << "VPTO bridge whitelist: duplicate wrapper declaration '"
             << decl.name << "' in '" << sourceName << "'\n";
      return failure();
    }
    if (decl.includes.empty()) {
      diagOS << "VPTO bridge whitelist: wrapper '" << decl.name
             << "' declares no includes in '" << sourceName << "'\n";
      return failure();
    }
    for (const std::string &include : decl.includes) {
      if (include.empty()) {
        diagOS << "VPTO bridge whitelist: wrapper '" << decl.name
               << "' has an empty include in '" << sourceName << "'\n";
        return failure();
      }
    }
    if (!decl.core.empty() && decl.core != kBridgeWrapperCoreCube &&
        decl.core != kBridgeWrapperCoreVec &&
        decl.core != kBridgeWrapperCoreBoth) {
      diagOS << "VPTO bridge whitelist: wrapper '" << decl.name
             << "' declares unsupported core '" << decl.core << "' in '"
             << sourceName << "' (supported: cube, vec, both; omit it to "
                "derive the guard from the routed tile kinds)\n";
      return failure();
    }
    if (whitelist.wrapperHasCustomEntry(decl.name)) {
      diagOS << "VPTO bridge whitelist: wrapper '" << decl.name
             << "' is declared in the wrappers section but carries "
                "'lowering: custom' entries owned by a dedicated renderer "
                "in '"
             << sourceName << "'\n";
      return failure();
    }
    bool hasDeclarativeEntry = false;
    for (const BridgeWhitelistEntry &entry : whitelist.bridgeOps) {
      if (entry.wrapper == decl.name && entry.isDeclarative() &&
          entry.op != BridgeWhitelist::kInternalOp) {
        hasDeclarativeEntry = true;
        break;
      }
    }
    if (!hasDeclarativeEntry) {
      diagOS << "VPTO bridge whitelist: wrapper '" << decl.name
             << "' is declared in the wrappers section but no declarative "
                "entry routes into it in '"
             << sourceName << "'\n";
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

/// The built-in default whitelist covering the wrappers bridged today:
/// The default bridge whitelist currently contains only the pipe family.
/// Pipe entries opt out of the declarative channel because storage lifecycle
/// and TPOP address rebinding require a dedicated lowering pass.
static constexpr llvm::StringLiteral kDefaultBridgeWhitelistYaml = R"yaml(
wrappers:
bridge_ops:
  - op: pto.initialize_l2l_pipe
    wrapper: pipe
    lowering: custom    # storage lifecycle: owned by the pipe family pass
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
    wrapper: pipe
    lowering: custom    # consumes the family pass storage SSA value
    entry: pto_vpto_pipe_push
    abi:
      - type: ptr    # storage
      - type: i64    # producer tile address
    tmpl_map:
      - source: tile
        field: tile
        target: ProducerTile
  - op: pto.tpop
    wrapper: pipe
    lowering: custom    # rebinds the tile address to the bridge call result
    entry: pto_vpto_pipe_pop
    abi:
      - type: ptr    # storage
    tmpl_map:
      - source: tile
        field: tile
        target: ConsumerTile
  - op: pto.tfree
    wrapper: pipe
    lowering: custom    # consumes the family pass storage SSA value
    entry: pto_vpto_pipe_free
    abi:
      - type: ptr    # storage
  - op: internal    # wrapper-internal helper, not routed from an IR op
    wrapper: pipe
    entry: pto_vpto_pipe_size
    abi: []
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
                         llvm::raw_ostream &diagOS, std::string *sourceName) {
  std::string path = resolveBridgeWhitelistPath(optionValue);
  if (sourceName) {
    *sourceName =
        path.empty() ? kBuiltinBridgeWhitelistSource.str() : path;
  }
  if (!path.empty()) {
    return parseBridgeWhitelist(path, diagOS);
  }
  return parseBridgeWhitelistFromBuffer(
      kDefaultBridgeWhitelistYaml, kBuiltinBridgeWhitelistSource, diagOS);
}

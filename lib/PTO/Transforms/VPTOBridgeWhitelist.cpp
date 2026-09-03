// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOBridgeWhitelist.cpp - bridge route policy ---------------------===//

#include "PTO/Transforms/VPTOBridgeWhitelist.h"
#include "PTO/Transforms/VPTOBridgeRegistry.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/YAMLTraits.h"
#include <cstdlib>

using namespace mlir;
using namespace mlir::pto;

namespace llvm {
namespace yaml {

template <> struct MappingTraits<BridgeFamilyPolicy> {
  static void mapping(IO &io, BridgeFamilyPolicy &policy) {
    io.mapOptional("enabled", policy.enabled, false);
    io.mapOptional("enabled_ops", policy.enabledOps);
  }
};

template <> struct MappingTraits<BridgePolicyFamilies> {
  static void mapping(IO &io, BridgePolicyFamilies &families) {
    io.mapOptional("pipe", families.pipe);
    io.mapOptional("cube", families.cube);
  }
};

template <> struct MappingTraits<BridgeRoutePolicy> {
  static void mapping(IO &io, BridgeRoutePolicy &policy) {
    io.mapRequired("version", policy.version);
    io.mapRequired("families", policy.families);
  }
};

} // namespace yaml
} // namespace llvm

namespace {

constexpr llvm::StringLiteral kDefaultBridgePolicyYaml = R"yaml(
version: 1
families:
  pipe:
    enabled: true
  cube:
    enabled_ops: []
)yaml";

} // namespace

bool BridgeRoutePolicy::routesFamily(llvm::StringRef family) const {
  if (family == "pipe") {
    return families.pipe.enabled;
  }
  if (family == "cube") {
    return families.cube.enabled;
  }
  return false;
}

bool BridgeRoutePolicy::routesOp(llvm::StringRef family,
                                 llvm::StringRef opName) const {
  const BridgeFamilyPolicy *policy = nullptr;
  if (family == "pipe") {
    policy = &families.pipe;
  } else if (family == "cube") {
    policy = &families.cube;
  } else {
    return false;
  }
  return policy->enabled || llvm::is_contained(policy->enabledOps, opName);
}

FailureOr<BridgeRoutePolicy> pto::parseBridgeRoutePolicyFromBuffer(
    llvm::StringRef content, llvm::StringRef sourceName,
    llvm::raw_ostream &diagOS) {
  for (llvm::StringRef key : {"bridge_ops", "wrappers", "entry", "abi",
                              "operand", "role", "call", "tmpl_args",
                              "tmpl_map", "includes", "storage_size_entry"}) {
    if (content.contains((key + ":").str())) {
      diagOS << "VPTO bridge policy: legacy key '" << key
             << "' is not allowed; policy YAML only selects routed families "
                "and ops in '" << sourceName << "'\n";
      return failure();
    }
  }

  BridgeRoutePolicy policy;
  llvm::yaml::Input input(content);
  input >> policy;
  if (std::error_code error = input.error()) {
    diagOS << "VPTO bridge policy: cannot parse '" << sourceName
           << "': " << error.message() << "\n";
    return failure();
  }
  if (policy.version != 1) {
    diagOS << "VPTO bridge policy: unsupported version " << policy.version
           << " in '" << sourceName << "' (supported: 1)\n";
    return failure();
  }
  if (!policy.families.pipe.enabledOps.empty()) {
    diagOS << "VPTO bridge policy: Pipe is routed as a family; pipe.enabled_ops "
              "is not supported in '" << sourceName << "'\n";
    return failure();
  }

  llvm::StringSet<> seenOps;
  for (const std::string &opName : policy.families.cube.enabledOps) {
    if (opName.empty() || !seenOps.insert(opName).second) {
      diagOS << "VPTO bridge policy: cube.enabled_ops contains an empty or "
                "duplicate op in '" << sourceName << "'\n";
      return failure();
    }
    const BridgeFunctionDesc *desc = findBridgeFunctionByOp(opName);
    if (!desc || desc->family != BridgeFamily::Cube) {
      diagOS << "VPTO bridge policy: cube.enabled_ops contains unregistered "
                "op '" << opName << "' in '" << sourceName << "'\n";
      return failure();
    }
  }
  return policy;
}

std::string pto::resolveBridgeWhitelistPath(llvm::StringRef optionValue) {
  if (!optionValue.empty()) {
    return optionValue.str();
  }
  if (const char *envPath = std::getenv("PTOAS_VPTO_BRIDGE_WHITELIST")) {
    return envPath;
  }
  return {};
}

FailureOr<BridgeRoutePolicy> pto::loadBridgeRoutePolicy(
    llvm::StringRef optionValue, llvm::raw_ostream &diagOS,
    std::string *sourceName) {
  std::string path = resolveBridgeWhitelistPath(optionValue);
  if (sourceName) {
    *sourceName = path.empty() ? kBuiltinBridgeWhitelistSource.str() : path;
  }
  if (path.empty()) {
    return parseBridgeRoutePolicyFromBuffer(
        kDefaultBridgePolicyYaml, kBuiltinBridgeWhitelistSource, diagOS);
  }
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    diagOS << "VPTO bridge policy: cannot read '" << path
           << "': " << buffer.getError().message() << "\n";
    return failure();
  }
  return parseBridgeRoutePolicyFromBuffer(buffer.get()->getBuffer(), path,
                                          diagOS);
}

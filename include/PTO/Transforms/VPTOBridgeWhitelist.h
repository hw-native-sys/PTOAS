// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOBridgeWhitelist.h - bridge route policy --------------*- C++ -*-===//
//===----------------------------------------------------------------------===//
//
// External configuration selects compiler-registered bridge families/ops.
// ABI, entry symbols, operand bindings and renderers are intentionally not
// configurable here.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_VPTOBRIDGEWHITELIST_H
#define MLIR_DIALECT_PTO_TRANSFORMS_VPTOBRIDGEWHITELIST_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <string>
#include <vector>

namespace mlir {
namespace pto {

struct BridgeFamilyPolicy {
  bool enabled = false;
  std::vector<std::string> enabledOps;
};

struct BridgePolicyFamilies {
  BridgeFamilyPolicy pipe;
  BridgeFamilyPolicy cube;
};

struct BridgeRoutePolicy {
  uint32_t version = 1;
  BridgePolicyFamilies families;

  bool routesFamily(llvm::StringRef family) const;
  bool routesOp(llvm::StringRef family, llvm::StringRef opName) const;
};

FailureOr<BridgeRoutePolicy>
parseBridgeRoutePolicyFromBuffer(llvm::StringRef content,
                                 llvm::StringRef sourceName,
                                 llvm::raw_ostream &diagOS);

FailureOr<BridgeRoutePolicy> loadBridgeRoutePolicy(
    llvm::StringRef optionValue, llvm::raw_ostream &diagOS,
    std::string *sourceName = nullptr);

std::string resolveBridgeWhitelistPath(llvm::StringRef optionValue);

constexpr llvm::StringLiteral kBuiltinBridgeWhitelistSource =
    "<built-in vpto bridge policy>";

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_VPTOBRIDGEWHITELIST_H

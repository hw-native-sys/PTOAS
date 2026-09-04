// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOBridgeRegistry.h - Strongly typed bridge ABI registry -*- C++
//-*-===//
//===----------------------------------------------------------------------===//
//
// Compiler-owned bridge ABI contracts. External policy selects registered
// entries but cannot redefine their symbols, signatures, or renderers.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_VPTOBRIDGEREGISTRY_H
#define MLIR_DIALECT_PTO_TRANSFORMS_VPTOBRIDGEREGISTRY_H

#include "PTO/IR/PTO.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>

namespace mlir {
namespace pto {

enum class BridgeFamily : uint8_t { Pipe, Cube };

enum class BridgeCoreKind : uint8_t { Any, Vector, Cube };

enum class BridgeValueKind : uint8_t {
  Pointer,
  I32,
  I64,
};

enum class BridgeRendererKind : uint8_t {
  Pipe,
  CubeDirect,
};

using BridgeEntryId = BridgeEntry;

struct BridgeAbiBinding {
  BridgeValueKind kind;
  int64_t operand = -1;
  llvm::StringLiteral argument;
  llvm::StringLiteral role;
};

struct BridgeFunctionDesc {
  BridgeEntryId id;
  BridgeFamily family;
  BridgeCoreKind core;
  BridgeRendererKind renderer;
  llvm::StringLiteral opName;
  llvm::StringLiteral symbolBase;
  llvm::StringLiteral callSpelling;
  llvm::ArrayRef<BridgeAbiBinding> bindings;
  llvm::ArrayRef<BridgeValueKind> arguments;
  llvm::ArrayRef<BridgeValueKind> results;
  bool createsObject = false;
  uint64_t objectAlignment = 0;
};

llvm::ArrayRef<BridgeFunctionDesc> getBridgeFunctionRegistry();

const BridgeFunctionDesc *findBridgeFunction(BridgeEntryId id);
const BridgeFunctionDesc *findBridgeFunctionByOp(llvm::StringRef opName);
const BridgeFunctionDesc *findBridgeFunctionBySymbol(llvm::StringRef symbol);

llvm::StringRef stringifyBridgeEntryId(BridgeEntryId id);
llvm::StringRef stringifyBridgeFamily(BridgeFamily family);
llvm::StringRef stringifyBridgeCore(BridgeCoreKind core);

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_VPTOBRIDGEREGISTRY_H

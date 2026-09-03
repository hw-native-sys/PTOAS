// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- VPTOBridgeTokens.h - C++ template token building ---------*- C++ -*-===//
//===----------------------------------------------------------------------===//
//
// Bridge-side construction of the PTO-ISA C++ template tokens used by the
// generated VPTO bridge wrapper. Both the IR-fact -> C++ spelling mapping
// rules and the bridge assembly rules (fully qualified spellings, NoneBox
// trailing-argument omission) live here.
//
// The tokens are fully qualified C++ type/constant spellings (e.g.
// "pto::TPipe<0, pto::Direction::DIR_C2V, 1024, 8, 2, false>") suitable for
// direct substitution into the generated wrapper source.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_VPTOBRIDGETOKENS_H
#define MLIR_DIALECT_PTO_TRANSFORMS_VPTOBRIDGETOKENS_H

#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace mlir {
namespace pto {

class InitializeL2LPipeOp;
class TileBufType;

/// Module attribute carrying the rendered bridge wrapper C++ source.
constexpr llvm::StringLiteral kBridgeWrapperSourceAttrName =
    "pto.vpto.bridge.wrapper_source";
constexpr llvm::StringLiteral kBridgeInstanceKeyAttrName = "instance_key";

constexpr llvm::StringLiteral kBridgeSpecPipeKey = "pipe";
constexpr llvm::StringLiteral kBridgeSpecProducerTileKey = "producer_tile";
constexpr llvm::StringLiteral kBridgeSpecConsumerTileKey = "consumer_tile";
constexpr llvm::StringLiteral kBridgeSpecSplitKey = "split";

/// Renders a supported MLIR element type as a PTO-ISA C++ type token.
FailureOr<std::string> buildBridgeElementTypeToken(Type elementType);

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_VPTOBRIDGETOKENS_H

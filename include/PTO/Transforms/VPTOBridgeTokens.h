// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOBridgeTokens.h - C++ template token building ---------*- C++ -*-===//
//===----------------------------------------------------------------------===//
//
// Bridge-side construction of the PTO-ISA C++ template tokens used by the
// generated VPTO bridge wrapper. These utilities are the bridge analogue of
// EmitC's token builders (see docs/designs/vpto-cpp-bridge-generalization.md
// section 3.3): they follow the same construction rules but are implemented
// independently so that neither backend depends on the other.
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

/// Module attribute carrying the collected pipe bridge specialization (a
/// DictionaryAttr of StringAttr token fields). Written by the pipe family
/// pass, consumed by the bridge wrapper generation pass.
constexpr llvm::StringLiteral kBridgeSpecAttrName = "pto.vpto.bridge.spec";

/// Function attribute carrying one function's pipe bridge specialization
/// (a DictionaryAttr with the same keys as the module spec). Written by the
/// pipe family pass; the wrapper generation pass merges the per-function
/// specs into the module spec and removes them. The family pass instances
/// may run concurrently, so the shared module attribute is only written by
/// the single-threaded module-level pass.
constexpr llvm::StringLiteral kBridgeFuncSpecAttrName =
    "pto.vpto.bridge.func_spec";

/// Module attribute carrying the rendered bridge wrapper C++ source (a
/// StringAttr). Written by the bridge wrapper generation pass, consumed by
/// object emission.
constexpr llvm::StringLiteral kBridgeWrapperSourceAttrName =
    "pto.vpto.bridge.wrapper_source";

/// Spec DictionaryAttr keys for the pipe bridge specialization.
constexpr llvm::StringLiteral kBridgeSpecPipeKey = "pipe";
constexpr llvm::StringLiteral kBridgeSpecProducerTileKey = "producer_tile";
constexpr llvm::StringLiteral kBridgeSpecConsumerTileKey = "consumer_tile";
constexpr llvm::StringLiteral kBridgeSpecSplitKey = "split";
constexpr llvm::StringLiteral kBridgeSpecEntryInitKey = "entry.init";
constexpr llvm::StringLiteral kBridgeSpecEntrySizeKey = "entry.size";
constexpr llvm::StringLiteral kBridgeSpecEntryPushKey = "entry.push";
constexpr llvm::StringLiteral kBridgeSpecEntryPopKey = "entry.pop";
constexpr llvm::StringLiteral kBridgeSpecEntryFreeKey = "entry.free";

/// Spec DictionaryAttr keys for the matmul bridge specialization. The tile
/// tokens come from the three (four for the accumulate form) tile operand
/// types; the acc phase token is only collected for a non-Unspecified
/// phase. The bias/MX entry variants add the bias and per-operand scale
/// tile tokens plus their own wrapper entry keys.
constexpr llvm::StringLiteral kBridgeSpecLeftTileKey = "left_tile";
constexpr llvm::StringLiteral kBridgeSpecRightTileKey = "right_tile";
constexpr llvm::StringLiteral kBridgeSpecResultTileKey = "result_tile";
constexpr llvm::StringLiteral kBridgeSpecAccInTileKey = "acc_in_tile";
constexpr llvm::StringLiteral kBridgeSpecBiasTileKey = "bias_tile";
constexpr llvm::StringLiteral kBridgeSpecAScaleTileKey = "a_scale_tile";
constexpr llvm::StringLiteral kBridgeSpecBScaleTileKey = "b_scale_tile";
constexpr llvm::StringLiteral kBridgeSpecAccPhaseKey = "acc_phase";
constexpr llvm::StringLiteral kBridgeSpecEntryMatmulKey = "entry.matmul";
constexpr llvm::StringLiteral kBridgeSpecEntryMatmulAccKey =
    "entry.matmul_acc";
constexpr llvm::StringLiteral kBridgeSpecEntryMatmulBiasKey =
    "entry.matmul_bias";
constexpr llvm::StringLiteral kBridgeSpecEntryMatmulMxKey =
    "entry.matmul_mx";
constexpr llvm::StringLiteral kBridgeSpecEntryMatmulMxAccKey =
    "entry.matmul_mx_acc";
constexpr llvm::StringLiteral kBridgeSpecEntryMatmulMxBiasKey =
    "entry.matmul_mx_bias";

/// Builds the fully qualified `pto::TPipe<flagBase, Direction, slotSize,
/// slotNum, localSlotNum, nosplit>` token from a local-to-local pipe init op.
/// Fails when the op lacks a flag_base attribute or carries an unsupported
/// dir_mask. The L2L pipe uses a fixed localSlotNum of 2.
FailureOr<std::string> buildBridgePipeToken(InitializeL2LPipeOp init);

/// Builds the fully qualified `pto::TileSplitAxis::TILE_*` token for a split
/// value (0..4). Fails for values outside that range.
FailureOr<std::string> buildBridgeTileSplitToken(int64_t split);

/// Builds the fully qualified `pto::Tile<TileType, dtype, Rows, Cols,
/// BLayout, RowValid, ColValid[, SLayout, SFractalSize]>` token from a tile
/// buffer type. The SLayout/SFractalSize template arguments are emitted only
/// for boxed (non-NoneBox) storage layouts, matching the wrapper's Tile
/// specializations. Fails when the type lacks a resolvable address space or
/// element type.
FailureOr<std::string> buildBridgeTileToken(TileBufType tile);

/// Builds the C++ element type token (e.g. "float", "half", "int8_t") for an
/// MLIR element type. Falls back to "float" for unrecognized types, mirroring
/// the EmitC element token behavior.
std::string buildBridgeElementTypeToken(Type elementType);

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_VPTOBRIDGETOKENS_H

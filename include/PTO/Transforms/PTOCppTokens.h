// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.huawei.com/
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOCppTokens.h - shared PTO-ISA C++ token mappings -------*- C++ -*-===//
//===----------------------------------------------------------------------===//
//
// Single source of truth for the pure mappings from IR facts (element
// types, enums, integer attribute values) to PTO-ISA C++ spellings. Both
// the EmitC backend and the VPTO C++ interface bridge render the same
// pto-isa template tokens; they share these mapping functions and keep
// their own assembly logic (which template arguments to emit, how the
// tokens are consumed).
//
// Every builder takes a `qualifier` prefix that is prepended to the
// pto-isa constant/type spelling: the bridge passes "pto::" (the wrapper
// is a standalone translation unit and always spells fully qualified
// names), the EmitC backend passes an empty string (its output relies on
// the surrounding namespace context).
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_PTOCPPTOKENS_H
#define MLIR_DIALECT_PTO_TRANSFORMS_PTOCPPTOKENS_H

#include "PTO/IR/PTO.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace mlir {
namespace pto {

/// Builds the C++ element type token (e.g. "float", "half", "int8_t") for
/// an MLIR element type. Falls back to "float" for unrecognized types.
std::string getPTOCppElementTypeToken(Type elementType);

/// Builds the `TileSplitAxis::TILE_*` token for a split value (0..4).
/// Fails for values outside that range.
FailureOr<std::string> getPTOCppTileSplitToken(int64_t split,
                                               llvm::StringRef qualifier);

/// Builds the `Direction::DIR_*` token for a local pipe dir_mask (1=C2V,
/// 2=V2C, 3=BOTH). The L2G2L "_GM" variants are an EmitC-side extension
/// and are not part of this core mapping. Fails for other masks.
FailureOr<std::string> getPTOCppDirectionToken(int8_t dirMask,
                                               llvm::StringRef qualifier);

/// Builds the `TileType::*` token for a local tile address space. Fails
/// for address spaces with no TileType mapping (e.g. global memory);
/// callers apply their own fallback policy for those.
FailureOr<std::string> getPTOCppTileTypeToken(AddressSpace addressSpace,
                                              llvm::StringRef qualifier);

/// Builds the `BLayout::*` token. Fails for values outside the closed set.
FailureOr<std::string> getPTOCppBLayoutToken(BLayout bLayout,
                                             llvm::StringRef qualifier);

/// Builds the `SLayout::*` token. Fails for values outside the closed set.
FailureOr<std::string> getPTOCppSLayoutToken(SLayout sLayout,
                                             llvm::StringRef qualifier);

/// Renders the `TPipe<flagBase, Direction, slotSize, slotNum, localSlotNum,
/// nosplit>` spelling; `dirTok` is an already rendered direction token.
std::string renderTPipeSpelling(int32_t flagBase, llvm::StringRef dirTok,
                                int32_t slotSize, int32_t slotNum,
                                int32_t localSlotNum, bool nosplit,
                                llvm::StringRef qualifier);

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_PTOCPPTOKENS_H

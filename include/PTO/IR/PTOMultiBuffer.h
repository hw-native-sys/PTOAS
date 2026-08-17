// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOMultiBuffer.h - Shared constants for multi-buffer ----*- C++ -*-===//
//
// Shared constants for the multi-buffer expression scheme:
//   - `kPtoMultiBufferAttrName` is the legacy memref-level slot-count
//     attribute.
//   - `kPtoMultiBufferAddrsAttrName` is the internal address list written
//     directly on tile-native `pto.alloc_multi_tile` by PlanMemory.
//   - `kPtoMultiBufferMaxNum` is the upper bound on the slot count N. It is
//     kept in lock-step with the InsertSync `kMaxMultiBufferCount`.
//
//===----------------------------------------------------------------------===//

#ifndef PTO_IR_PTOMULTIBUFFER_H
#define PTO_IR_PTOMULTIBUFFER_H

#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace pto {

/// Attribute name for multi-buffer depth (integer slot count N>=2).
inline constexpr llvm::StringLiteral kPtoMultiBufferAttrName =
    "pto.multi_buffer";

/// Internal DenseI64ArrayAttr containing one byte address per physical slot.
inline constexpr llvm::StringLiteral kPtoMultiBufferAddrsAttrName =
    "pto.multi_buffer_addrs";

/// Upper bound for N; must stay consistent with `kMaxMultiBufferCount` in
/// insert-sync.
inline constexpr unsigned kPtoMultiBufferMaxNum = 16;

/// Lower bound for N (a multi_tile_buf must have at least 2 slots).
inline constexpr unsigned kPtoMultiBufferMinNum = 2;

} // namespace pto
} // namespace mlir

#endif // PTO_IR_PTOMULTIBUFFER_H

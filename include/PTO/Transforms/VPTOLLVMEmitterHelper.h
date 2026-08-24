// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_VPTOLLVMEMITTERHELPER_H
#define MLIR_DIALECT_PTO_TRANSFORMS_VPTOLLVMEMITTERHELPER_H

#include "PTO/Transforms/VPTOLLVMEmitter.h"

namespace mlir {
class ConversionPatternRewriter;
}

namespace mlir::pto {

/// Target-ABI lowering for an index-denominated VPTO address offset. VPTO keeps
/// the logical offset as `index`; the i32 value below is only the intrinsic
/// encoding. When the logical offset cannot be represented by that encoding,
/// `base` and, for post-update operations, `updatedBase` preserve the complete
/// pointer-width address calculation while `intrinsicOffset` is zero.
struct VPTOLoweredAddressOffset {
  Value base;
  Value intrinsicOffset;
  Value updatedBase;
};

FailureOr<VPTOLoweredAddressOffset> lowerVPTOElementOffsetForIntrinsic(
    Operation *anchor, Value base, Value elementOffset, Type elementType,
    bool isPostUpdate, ::mlir::ConversionPatternRewriter &rewriter);

/// Lower the byte- or alignment-denominated index offset of plds/pldi/psts/psti.
/// The intrinsic encodes logical units in i32. If that encoding is insufficient,
/// the fallback uses the op-specific unit size to materialize the full address.
FailureOr<VPTOLoweredAddressOffset> lowerVPTOPredicateOffsetForIntrinsic(
    Operation *anchor, Value base, Value offset, bool isPostUpdate,
    ::mlir::ConversionPatternRewriter &rewriter);

} // namespace mlir::pto

#endif // MLIR_DIALECT_PTO_TRANSFORMS_VPTOLLVMEMITTERHELPER_H

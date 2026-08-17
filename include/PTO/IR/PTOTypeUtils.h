// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_IR_PTOTYPEUTILS_H
#define PTO_IR_PTOTYPEUTILS_H

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

namespace mlir::pto {

namespace detail {
template <typename MemRefT>
inline auto getPTOMemRefStridesAndOffsetImpl(
    MemRefT memTy, SmallVectorImpl<int64_t> &strides, int64_t &offset, int)
    -> decltype(memTy.getStridesAndOffset(strides, offset)) {
  return memTy.getStridesAndOffset(strides, offset);
}

template <typename MemRefT>
inline LogicalResult getPTOMemRefStridesAndOffsetImpl(
    MemRefT memTy, SmallVectorImpl<int64_t> &strides, int64_t &offset, long) {
  return getStridesAndOffset(memTy, strides, offset);
}
} // namespace detail

inline LogicalResult getPTOMemRefStridesAndOffset(
    MemRefType memTy, SmallVectorImpl<int64_t> &strides, int64_t &offset) {
  return detail::getPTOMemRefStridesAndOffsetImpl(memTy, strides, offset, 0);
}

bool isPTOFloat8Type(Type t);
bool isPTOFloat8E4M3LikeType(Type t);
bool isPTOFloat8E5M2LikeType(Type t);
bool isPTOHiFloat8Type(Type t);
bool isPTOF8E8M0Type(Type t);
bool isPTOHiFloat8x2Type(Type t);
bool isPTOBF16x2Type(Type t);
bool isPTOFloat4PackedType(Type t);
bool isPTOPackedLdgStgVectorType(Type t);
bool isPTOLowPrecisionType(Type t);

unsigned getPTOStorageElemBitWidth(Type t);
unsigned getPTOStorageElemByteSize(Type t);
unsigned getPTOPackedLdgStgTotalBits(Type t);

} // namespace mlir::pto

#endif // PTO_IR_PTOTYPEUTILS_H

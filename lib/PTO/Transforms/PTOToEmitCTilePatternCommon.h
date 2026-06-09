// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TO_EMITC_TILE_PATTERN_COMMON_H
#define PTO_TO_EMITC_TILE_PATTERN_COMMON_H

#include "llvm/ADT/SmallVector.h"

namespace mlir::pto {

constexpr unsigned kInlineCapacity2 = 2;
constexpr unsigned kInlineCapacity3 = 3;
constexpr unsigned kInlineCapacity4 = 4;
constexpr unsigned kInlineCapacity5 = 5;
constexpr unsigned kInlineCapacity7 = 7;
constexpr unsigned kInlineCapacity8 = 8;

template <typename T>
using SmallVec2 = llvm::SmallVector<T, kInlineCapacity2>;
template <typename T>
using SmallVec3 = llvm::SmallVector<T, kInlineCapacity3>;
template <typename T>
using SmallVec4 = llvm::SmallVector<T, kInlineCapacity4>;
template <typename T>
using SmallVec5 = llvm::SmallVector<T, kInlineCapacity5>;
template <typename T>
using SmallVec7 = llvm::SmallVector<T, kInlineCapacity7>;
template <typename T>
using SmallVec8 = llvm::SmallVector<T, kInlineCapacity8>;

} // namespace mlir::pto

#endif // PTO_TO_EMITC_TILE_PATTERN_COMMON_H

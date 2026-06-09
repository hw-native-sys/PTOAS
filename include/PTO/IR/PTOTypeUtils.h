// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_IR_PTOTYPEUTILS_H
#define PTO_IR_PTOTYPEUTILS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

#include <cstdint>

namespace mlir::pto {

inline constexpr unsigned kPTOByteBitWidth = 8;
inline constexpr unsigned kPTOI8BitWidth = 8;
inline constexpr unsigned kPTOI16BitWidth = 16;
inline constexpr unsigned kPTOI32BitWidth = 32;
inline constexpr unsigned kPTOI64BitWidth = 64;
inline constexpr unsigned kPTOI128BitWidth = 128;
inline constexpr unsigned kPTOPaddedTensorRank5D = 5;
inline constexpr int32_t kFractalSize16 = 16;
inline constexpr int32_t kFractalSize32 = 32;
inline constexpr int32_t kFractalSize512 = 512;
inline constexpr int32_t kFractalSize1024 = 1024;
inline constexpr unsigned kPTOByteSize = 1;
inline constexpr unsigned kPTOHalfWordBytes = 2;
inline constexpr unsigned kPTOWordBytes = 4;
inline constexpr unsigned kPTODoubleWordBytes = 8;

bool isPTOFloat8Type(Type t);
bool isPTOHiFloat8Type(Type t);
bool isPTOFloat4PackedType(Type t);
bool isPTOLowPrecisionType(Type t);

unsigned getPTOStorageElemBitWidth(Type t);
unsigned getPTOStorageElemByteSize(Type t);

} // namespace mlir::pto

#endif // PTO_IR_PTOTYPEUTILS_H

// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Support/CodeConstants.h"
#include "PTO/IR/PTOTypeUtils.h"

#include "PTO/IR/PTO.h"

using namespace mlir;
using namespace mlir::pto;

namespace {
constexpr unsigned kBitsPerByte = 8;
constexpr unsigned kPackedLdgStgBitWidth16 = 16;
constexpr unsigned kPackedLdgStgBitWidth32 = 32;
constexpr unsigned kPackedLdgStgBitWidth64 = 64;
} // namespace

bool mlir::pto::isPTOFloat8Type(Type t) {
  return isPTOFloat8E4M3LikeType(t) || isPTOFloat8E5M2LikeType(t);
}

bool mlir::pto::isPTOFloat8E4M3LikeType(Type t) {
  return isa<Float8E4M3Type, Float8E4M3FNType, Float8E4M3FNUZType,
             Float8E4M3B11FNUZType>(t);
}

bool mlir::pto::isPTOFloat8E5M2LikeType(Type t) {
  return isa<Float8E5M2Type, Float8E5M2FNUZType>(t);
}

bool mlir::pto::isPTOHiFloat8Type(Type t) { return isa<HiF8Type>(t); }

bool mlir::pto::isPTOF8E8M0Type(Type t) { return isa<F8E8M0Type>(t); }

bool mlir::pto::isPTOHiFloat8x2Type(Type t) { return isa<HiF8x2Type>(t); }

bool mlir::pto::isPTOBF16x2Type(Type t) { return isa<BF16x2Type>(t); }

bool mlir::pto::isPTOFloat4PackedType(Type t) {
  return isa<F4E1M2x2Type, F4E2M1x2Type>(t);
}

bool mlir::pto::isPTOPackedLdgStgVectorType(Type t) {
  // !pto.hif8x2 is a 2-byte packed hif8 value type (not a VectorType).
  if (isPTOHiFloat8x2Type(t)) {
    return true;
  }
  auto vecType = dyn_cast<VectorType>(t);
  if (!vecType || vecType.isScalable() || vecType.getRank() != 1) {
    return false;
  }
  int64_t lanes = vecType.getDimSize(0);
  Type elemType = vecType.getElementType();
  bool validElem = false;
  if (isPTOFloat8Type(elemType)) {
    validElem = lanes == mlir::pto::kValue2 || lanes == 4 || lanes == 8;
  } else {
    validElem =
        lanes == mlir::pto::kValue2 &&
        (elemType.isF16() || elemType.isBF16() || elemType.isF32());
  }
  if (!validElem) {
    if (auto intTy = dyn_cast<IntegerType>(elemType)) {
      unsigned w = intTy.getWidth();
      validElem = lanes == mlir::pto::kValue2 && (w == 8 || w == 16 || w == 32);
    }
  }
  if (!validElem) {
    return false;
  }
  unsigned totalBits =
      vecType.getDimSize(0) * getPTOStorageElemBitWidth(elemType);
  return totalBits == kPackedLdgStgBitWidth16 ||
         totalBits == kPackedLdgStgBitWidth32 ||
         totalBits == kPackedLdgStgBitWidth64;
}

unsigned mlir::pto::getPTOPackedLdgStgTotalBits(Type t) {
  if (isPTOHiFloat8x2Type(t)) {
    return getPTOStorageElemBitWidth(t); // 16
  }
  auto vecType = cast<VectorType>(t);
  return vecType.getDimSize(0) *
         getPTOStorageElemBitWidth(vecType.getElementType());
}

bool mlir::pto::isPTOLowPrecisionType(Type t) {
  return isPTOFloat8Type(t) || isPTOHiFloat8Type(t) || isPTOF8E8M0Type(t) ||
         isPTOHiFloat8x2Type(t) || isPTOFloat4PackedType(t) ||
         isPTOBF16x2Type(t);
}

unsigned mlir::pto::getPTOStorageElemBitWidth(Type t) {
  if (isPTOHiFloat8x2Type(t)) {
    return 16;
  }
  // bf16x2 is a 4-byte packed pair; special-case it before the generic
  // low-precision branch (which would otherwise report 8 bits).
  if (isPTOBF16x2Type(t)) {
    return 32;
  }
  if (isPTOLowPrecisionType(t)) {
    return kBitsPerByte;
}
  if (auto floatTy = dyn_cast<FloatType>(t)) {
    return floatTy.getWidth();
}
  if (auto intTy = dyn_cast<IntegerType>(t)) {
    return intTy.getWidth();
}
  return 0;
}

unsigned mlir::pto::getPTOStorageElemByteSize(Type t) {
  unsigned bitWidth = getPTOStorageElemBitWidth(t);
  return bitWidth == 0 ? 0 : bitWidth / kBitsPerByte;
}

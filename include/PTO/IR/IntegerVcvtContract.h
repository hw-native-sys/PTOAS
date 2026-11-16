// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software; you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See the License in the root of the software repository for the full text of the License.

#ifndef PTO_IR_INTEGER_VCVT_CONTRACT_H
#define PTO_IR_INTEGER_VCVT_CONTRACT_H

#include "mlir/IR/BuiltinTypes.h"

#include <optional>

namespace mlir::pto {

/// Return whether a VMI integer vcvt belongs to the directly lowered
/// 8/16/32-bit width-changing family. Other conversions retain the existing
/// unified-to-legacy lowering path.
inline bool isDirectIntegerVcvtWidthChange(Type sourceType, Type resultType) {
  auto sourceInteger = dyn_cast<IntegerType>(sourceType);
  auto resultInteger = dyn_cast<IntegerType>(resultType);
  auto supportedWidth = [](unsigned width) {
    return width == 8 || width == 16 || width == 32;
  };
  return sourceInteger && resultInteger && supportedWidth(sourceInteger.getWidth()) &&
         supportedWidth(resultInteger.getWidth()) && sourceInteger.getWidth() != resultInteger.getWidth();
}

enum class IntegerVcvtPartFamily {
  EvenOdd,
  Packed4,
};

struct IntegerVcvtContract {
    bool requiresSaturate;
    bool requiresPart;
    IntegerVcvtPartFamily partFamily;
    bool saturatesNegativeToZero = false;
};

/// Return the A5 VCVTII contract for one native integer type pair.
inline std::optional<IntegerVcvtContract> lookupIntegerVcvtContract(unsigned sourceBits, bool sourceSigned,
                                                                    unsigned resultBits, bool resultSigned) {
  if (sourceBits == 8 && resultBits == 16 && sourceSigned == resultSigned) {
    return IntegerVcvtContract{false, true, IntegerVcvtPartFamily::EvenOdd};
  }
  if (sourceBits == 8 && resultBits == 32 && sourceSigned == resultSigned) {
    return IntegerVcvtContract{false, true, IntegerVcvtPartFamily::Packed4};
  }
  if (sourceBits == 16 && resultBits == 32 && (sourceSigned || !resultSigned)) {
    return IntegerVcvtContract{false, true, IntegerVcvtPartFamily::EvenOdd, sourceSigned && !resultSigned};
  }
  if (sourceBits == 16 && resultBits == 8 && !resultSigned) {
    return IntegerVcvtContract{true, true, IntegerVcvtPartFamily::EvenOdd};
  }
  if (sourceBits == 32 && resultBits == 16) {
    return IntegerVcvtContract{true, true, IntegerVcvtPartFamily::EvenOdd};
  }
  if (sourceBits == 32 && resultBits == 8 && !resultSigned) {
    return IntegerVcvtContract{true, true, IntegerVcvtPartFamily::Packed4};
  }
  if (sourceBits == 32 && resultBits == 64 && sourceSigned && resultSigned) {
    return IntegerVcvtContract{false, true, IntegerVcvtPartFamily::EvenOdd};
  }
  if (sourceBits == 64 && resultBits == 32 && sourceSigned && resultSigned) {
    return IntegerVcvtContract{true, true, IntegerVcvtPartFamily::EvenOdd};
  }
  return std::nullopt;
}

/// Signed 8-bit destinations without a native VCVTII form can use an
/// equivalent bit-preserving NOSAT truncation followed by a type reinterpret.
inline bool isIntegerVcvtSignedByteAlias(unsigned sourceBits, unsigned resultBits, bool resultSigned) {
  return (sourceBits == 16 || sourceBits == 32) && resultBits == 8 && resultSigned;
}

} // namespace mlir::pto

#endif // PTO_IR_INTEGER_VCVT_CONTRACT_H

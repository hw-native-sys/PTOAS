// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- MemrefSupport.cpp - Memref lowering helpers --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "MemrefInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

// CANN Open Software License Agreement Version 2.0 (the "License").
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.

//===- PTOToEmitCMemref.cpp - memref/global-tensor/pointer lowering ---------===//
//===----------------------------------------------------------------------===//







// =============================================================================
// 4. MemRef SubView -> Explicit Shape/Stride Construction (Full Implementation)
// =============================================================================
// Lower an OpFoldResult to an int64_t EmitC value using the given index type.
Value ofrToEmitCIndexValue(ConversionPatternRewriter &rewriter,
                                  Location loc, Type indexTy,
                                  OpFoldResult ofr) {
  auto mkIndex = [&](int64_t v) -> Value {
    return rewriter.create<emitc::ConstantOp>(
        loc, indexTy, emitc::OpaqueAttr::get(rewriter.getContext(),
                                             std::to_string(v)));
  };
  auto asIndex = [&](Value value) -> Value {
    if (value.getType() == indexTy)
      return value;
    return rewriter.create<emitc::CastOp>(loc, indexTy, value).getResult();
  };
  if (isa<Value>(ofr)) {
    Value v = cast<Value>(ofr);
    Value rv = rewriter.getRemappedValue(v);
    return asIndex(rv);
  }
  if (isa<Attribute>(ofr)) {
    Attribute attr = cast<Attribute>(ofr);
    if (auto ia = dyn_cast<IntegerAttr>(attr))
      return mkIndex(getIntegerAttrSignedValue(ia));
  }
  return mkIndex(0);
}

// C++ scalar token for a MemRef element type, defaulting to float.
std::string memrefElemTypeToString(Type elemTy) {
  if (elemTy.isF16())
    return "half";
  if (elemTy.isBF16())
    return "bfloat16_t";
  if (elemTy.isF32())
    return "float";
  if (elemTy.isF64())
    return "double";
  if (elemTy.isInteger(8)) {
    if (elemTy.isSignlessInteger(8) || elemTy.isSignedInteger(8))
      return "int8_t";
    return "uint8_t";
  }
  if (elemTy.isInteger(16)) {
    if (elemTy.isSignlessInteger(16) || elemTy.isSignedInteger(16))
      return "int16_t";
    return "uint16_t";
  }
  if (elemTy.isInteger(32)) {
    if (elemTy.isSignlessInteger(32) || elemTy.isSignedInteger(32))
      return "int32_t";
    return "uint32_t";
  }
  if (elemTy.isInteger(64)) {
    return cast<IntegerType>(elemTy).isUnsigned() ? "uint64_t" : "int64_t";
  }
  return "float";
}

//===----------------------------------------------------------------------===//
// Helper: build GlobalTensor from a static MemRef (for TLOAD/TSTORE)
//===----------------------------------------------------------------------===//

























//===----------------------------------------------------------------------===//
// PTO pointer lowering
//===----------------------------------------------------------------------===



} // namespace pto
} // namespace mlir

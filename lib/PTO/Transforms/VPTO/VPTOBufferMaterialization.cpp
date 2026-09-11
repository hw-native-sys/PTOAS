// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/VPTOLowering.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::pto {
namespace {

static AddressSpaceAttr getNormalizedPtrMemorySpace(Attribute memorySpace,
                                                    MLIRContext *context) {
  if (auto addrSpace = dyn_cast_or_null<AddressSpaceAttr>(memorySpace)) {
    return addrSpace;
  }
  if (auto intAttr = dyn_cast_or_null<IntegerAttr>(memorySpace)) {
    return AddressSpaceAttr::get(context,
                                 static_cast<AddressSpace>(intAttr.getInt()));
  }
  return AddressSpaceAttr::get(context, AddressSpace::GM);
}

static Value materializeMemRefView(Value value, MemRefType memrefType,
                                   PatternRewriter &rewriter, Location loc) {
  if (value.getType() == memrefType) {
    return value;
  }
  return rewriter
      .create<UnrealizedConversionCastOp>(
          loc, TypeRange(ArrayRef<Type>{memrefType}), value)
      .getResult(0);
}

static Value materializeTileBufferView(Value value, PatternRewriter &rewriter,
                                       Location loc) {
  if (isa<BaseMemRefType>(value.getType())) {
    return value;
  }

  auto tileType = dyn_cast<TileBufType>(value.getType());
  if (!tileType) {
    return {};
  }

  auto memrefType =
      MemRefType::get(tileType.getShape(), tileType.getElementType(),
                      AffineMap(), tileType.getMemorySpace());
  return materializeMemRefView(value, memrefType, rewriter, loc);
}

} // namespace

Value materializeBufferPointer(Value value, Type elementType,
                               Attribute memorySpace,
                               PatternRewriter &rewriter, Location loc) {
  if (!value) {
    return {};
  }

  auto ptrMemorySpace =
      getNormalizedPtrMemorySpace(memorySpace, rewriter.getContext());
  auto ptrType = PtrType::get(rewriter.getContext(), elementType, ptrMemorySpace);
  if (value.getType() == ptrType) {
    return value;
  }

  Value memrefValue = materializeTileBufferView(value, rewriter, loc);
  auto memrefType = dyn_cast_or_null<MemRefType>(memrefValue.getType());
  if (!memrefValue || !memrefType) {
    return {};
  }
  return rewriter.create<CastPtrOp>(loc, ptrType, memrefValue).getResult();
}
} // namespace mlir::pto

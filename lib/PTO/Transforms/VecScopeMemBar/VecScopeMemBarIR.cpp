// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/VecScopeMemBar/VecScopeMemBarIR.h"
#include "PTO/IR/PTO.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::vecscopemembar;

namespace {

static bool hasUBEffect(Operation *op, bool wantRead, bool wantWrite) {
  if (!isa<pto::VectorMicroOpInterface>(op))
    return false;
  auto iface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!iface)
    return false;

  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 4> effects;
  iface.getEffects(effects);
  return llvm::any_of(effects, [&](const auto &effect) {
    Value value = effect.getValue();
    if (!value || !isUBBackedType(value.getType()))
      return false;
    return (wantRead && isa<MemoryEffects::Read>(effect.getEffect())) ||
           (wantWrite && isa<MemoryEffects::Write>(effect.getEffect()));
  });
}

} // namespace

bool vecscopemembar::isUBBackedType(Type type) {
  if (auto ptr = dyn_cast<pto::PtrType>(type))
    return ptr.getMemorySpace().getAddressSpace() == AddressSpace::VEC;
  auto memref = dyn_cast<BaseMemRefType>(type);
  if (!memref)
    return false;
  Attribute space = memref.getMemorySpace();
  if (auto attr = dyn_cast_or_null<pto::AddressSpaceAttr>(space))
    return attr.getAddressSpace() == AddressSpace::VEC;
  if (auto integer = dyn_cast_or_null<IntegerAttr>(space))
    return integer.getInt() == static_cast<int64_t>(AddressSpace::VEC);
  return false;
}

bool vecscopemembar::isUBVectorStore(Operation *op) {
  return hasUBEffect(op, /*wantRead=*/false, /*wantWrite=*/true);
}

bool vecscopemembar::isUBVectorLoad(Operation *op) {
  return hasUBEffect(op, /*wantRead=*/true, /*wantWrite=*/false);
}

bool vecscopemembar::isUBVectorMemoryOp(Operation *op) {
  return isUBVectorStore(op) || isUBVectorLoad(op);
}

SmallVector<Value, 2> vecscopemembar::getStoredValues(Operation *storeOp) {
  SmallVector<Value, 2> out;
  if (auto s = dyn_cast<pto::VstsOp>(storeOp)) {
    out.push_back(s.getValue());
  } else if (auto s = dyn_cast<pto::Vstsx2Op>(storeOp)) {
    out.push_back(s.getLow());
    out.push_back(s.getHigh());
  }
  return out;
}

SmallVector<Value, 2> vecscopemembar::getLoadedValues(Operation *loadOp) {
  SmallVector<Value, 2> out;
  if (auto l = dyn_cast<pto::VldsOp>(loadOp)) {
    if (Value r = l.getResult())
      out.push_back(r);
  } else if (auto l = dyn_cast<pto::Vldsx2Op>(loadOp)) {
    out.push_back(l.getLow());
    out.push_back(l.getHigh());
  }
  return out;
}

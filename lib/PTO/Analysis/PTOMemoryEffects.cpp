// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Analysis/PTOMemoryAccessAnalysis.h"
#include "PTO/IR/PTO.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::pto;

namespace {

// Scalar accesses use the ordinary PTO pointer load/store ops. Legacy SIMT
// scalar operations also implement SimtOpInterface; keep that spelling for
// compatibility with older IR while classifying PTOLoad/PTOStore directly.
static bool isScalarAccess(Operation* op) { return isa<pto::SimtOpInterface, pto::PTOLoadOp, pto::PTOStoreOp>(op); }

static bool hasUBEffect(Operation* op, bool wantRead, bool wantWrite)
{
    auto iface = dyn_cast<MemoryEffectOpInterface>(op);
    if (!iface) {
        return false;
    }

    SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 4> effects;
    iface.getEffects(effects);
    return llvm::any_of(effects, [&](const auto& effect) {
        Value value = effect.getValue();
        if (!value || !isUBBackedType(value.getType())) {
            return false;
        }
        return (wantRead && isa<MemoryEffects::Read>(effect.getEffect())) ||
               (wantWrite && isa<MemoryEffects::Write>(effect.getEffect()));
    });
}

} // namespace

bool pto::isUBBackedType(Type type)
{
    if (auto ptr = dyn_cast<pto::PtrType>(type)) {
        return ptr.getMemorySpace().getAddressSpace() == AddressSpace::VEC;
    }
    auto memref = dyn_cast<BaseMemRefType>(type);
    if (!memref) {
        return false;
    }
    Attribute space = memref.getMemorySpace();
    if (auto attr = dyn_cast_or_null<pto::AddressSpaceAttr>(space)) {
        return attr.getAddressSpace() == AddressSpace::VEC;
    }
    if (auto integer = dyn_cast_or_null<IntegerAttr>(space)) {
        return integer.getInt() == static_cast<int64_t>(AddressSpace::VEC);
    }
    return false;
}

bool pto::isUBVectorStore(Operation* op) { return !isScalarAccess(op) && hasUBEffect(op, false, true); }

bool pto::isUBVectorLoad(Operation* op) { return !isScalarAccess(op) && hasUBEffect(op, true, false); }

bool pto::isUBVectorMemoryOp(Operation* op) { return isUBVectorStore(op) || isUBVectorLoad(op); }

bool pto::isUBScalarMemoryOp(Operation* op) { return isScalarAccess(op) && hasUBEffect(op, true, true); }

bool pto::isUBScalarStore(Operation* op) { return isScalarAccess(op) && hasUBEffect(op, false, true); }

bool pto::isUBScalarLoad(Operation* op) { return isScalarAccess(op) && hasUBEffect(op, true, false); }

bool pto::isUBMemoryOp(Operation* op) { return isUBVectorMemoryOp(op) || isUBScalarMemoryOp(op); }

SmallVector<Value, 2> pto::getStoredValues(Operation* storeOp)
{
    SmallVector<Value, 2> out;
    if (auto s = dyn_cast<pto::PTOStoreOp>(storeOp)) {
        out.push_back(s.getValue());
    } else if (auto s = dyn_cast<pto::VscatterOp>(storeOp)) {
        out.push_back(s.getValue());
    } else if (auto s = dyn_cast<pto::VstsOp>(storeOp)) {
        out.push_back(s.getValue());
    } else if (auto s = dyn_cast<pto::Vstsx2Op>(storeOp)) {
        out.push_back(s.getLow());
        out.push_back(s.getHigh());
    }
    return out;
}

SmallVector<Value, 2> pto::getLoadedValues(Operation* loadOp)
{
    SmallVector<Value, 2> out;
    if (auto l = dyn_cast<pto::PTOLoadOp>(loadOp)) {
        out.push_back(l.getResult());
    } else if (isa<pto::Vgather2Op, pto::Vgather2BcOp, pto::VgatherbOp>(loadOp)) {
        out.push_back(loadOp->getResult(0));
    } else if (auto l = dyn_cast<pto::VldsOp>(loadOp)) {
        if (Value r = l.getResult()) {
            out.push_back(r);
        }
    } else if (auto l = dyn_cast<pto::Vldsx2Op>(loadOp)) {
        out.push_back(l.getLow());
        out.push_back(l.getHigh());
    }
    return out;
}

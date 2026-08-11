// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOScheduling.cpp - VPTO scheduling semantics --------------------===//

#include "PTO/IR/VPTOScheduling.h"
#include "PTO/IR/PTO.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::pto;

namespace {
static std::optional<PIPE> getExecutionPipe(Operation *op) {
  if (auto pipeOp = dyn_cast<OpPipeInterface>(op))
    return pipeOp.getPipe();
  if (isa<VectorMicroOpInterface>(op))
    return PIPE::PIPE_V;
  if (isa<CubeMicroOpInterface>(op))
    return PIPE::PIPE_M;
  if (isa<SimtOpInterface>(op))
    return PIPE::PIPE_S;
  // Raw MTE micro-ops do not all expose a precise OpPipeInterface yet. Use
  // PIPE_ALL as their conservative execution-pipe classification.
  if (isa<MteOpInterface>(op))
    return PIPE::PIPE_ALL;
  return std::nullopt;
}

static bool isMemoryAddress(Value value) {
  return value && isa<pto::PtrType, BaseMemRefType>(value.getType());
}

static Attribute getAddressSpace(Value value) {
  if (auto pointer = dyn_cast<pto::PtrType>(value.getType()))
    return pointer.getMemorySpace();
  if (auto memref = dyn_cast<BaseMemRefType>(value.getType()))
    return memref.getMemorySpace();
  return {};
}

static bool isStoreLikeName(StringRef name) {
  return name == "pto.store" || name == "pto.stg" || name == "pto.st_dev" ||
         name.starts_with("pto.vst") || name.starts_with("pto.pst");
}

static std::optional<int64_t> getElementByteSize(Value pointer) {
  Type elementType;
  if (auto pointerType = dyn_cast<pto::PtrType>(pointer.getType()))
    elementType = pointerType.getElementType();
  else if (auto memrefType = dyn_cast<BaseMemRefType>(pointer.getType()))
    elementType = memrefType.getElementType();
  if (!elementType)
    return std::nullopt;

  int64_t elementCount = 1;
  if (auto vectorType = dyn_cast<VectorType>(elementType)) {
    if (vectorType.isScalable())
      return std::nullopt;
    elementCount = vectorType.getNumElements();
    elementType = vectorType.getElementType();
  }
  if (!elementType.isIntOrFloat())
    return std::nullopt;
  unsigned bitWidth = elementType.getIntOrFloatBitWidth();
  if (bitWidth == 0 || bitWidth % 8 != 0)
    return std::nullopt;

  int64_t byteSize;
  if (llvm::MulOverflow(elementCount, static_cast<int64_t>(bitWidth / 8),
                        byteSize))
    return std::nullopt;
  return byteSize;
}

static std::optional<int64_t> getConstantOffset(Value offset) {
  APInt value;
  if (!matchPattern(offset, m_ConstantInt(&value)) || !value.isSignedIntN(64))
    return std::nullopt;
  return value.getSExtValue();
}

template <typename OpTy>
static void setStaticIndexedRange(OpTy op, VPTOMemoryAccess &access) {
  if (access.address != op.getPtr())
    return;
  std::optional<int64_t> elementOffset = getConstantOffset(op.getOffset());
  std::optional<int64_t> elementByteSize = getElementByteSize(access.address);
  if (!elementOffset || !elementByteSize)
    return;
  int64_t byteOffset;
  if (llvm::MulOverflow(*elementOffset, *elementByteSize, byteOffset))
    return;
  access.byteOffset = byteOffset;
  access.byteSize = *elementByteSize;
}

static void setStaticAccessRange(Operation *op, VPTOMemoryAccess &access) {
  if (auto load = dyn_cast<PTOLoadOp>(op))
    return setStaticIndexedRange(load, access);
  if (auto store = dyn_cast<PTOStoreOp>(op))
    return setStaticIndexedRange(store, access);
  if (auto load = dyn_cast<PTOLdgOp>(op))
    return setStaticIndexedRange(load, access);
  if (auto store = dyn_cast<PTOStgOp>(op))
    return setStaticIndexedRange(store, access);
  if (auto load = dyn_cast<PTOLdDevOp>(op))
    return setStaticIndexedRange(load, access);
  if (auto store = dyn_cast<PTOStDevOp>(op))
    return setStaticIndexedRange(store, access);
}

/// These operations have complete scheduler-specific state semantics and do
/// not access ordinary memory. This classification deliberately does not add
/// Pure: vector-memory barriers and register-state effects must remain visible
/// to the scheduler and to general IR transformations.
static bool hasKnownNoOrdinaryMemoryAccess(Operation *op) {
  return isa<MemBarOp, SprclrOp, GetCtrlOp, SetCtrlOp>(op);
}

static void collectMemoryAccesses(Operation *op,
                                  VPTOSchedulingSemantics &semantics) {
  SmallVectorImpl<VPTOMemoryAccess> &accesses = semantics.memoryAccesses;
  if (hasKnownNoOrdinaryMemoryAccess(op)) {
    semantics.memoryBehavior = VPTOMemoryBehavior::None;
    return;
  }

  auto memoryEffects = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memoryEffects) {
    if (isMemoryEffectFree(op)) {
      semantics.memoryBehavior = VPTOMemoryBehavior::None;
      return;
    }
    semantics.memoryBehavior = VPTOMemoryBehavior::Unknown;
    VPTOMemoryAccess access;
    access.writes = true;
    access.ordered = true;
    access.unknown = true;
    accesses.push_back(access);
    return;
  }

  semantics.memoryBehavior = VPTOMemoryBehavior::Explicit;
  SmallVector<MemoryEffects::EffectInstance> effects;
  memoryEffects.getEffects(effects);
  bool storeLike = isStoreLikeName(op->getName().getStringRef());
  for (const MemoryEffects::EffectInstance &effect : effects) {
    Value value = effect.getValue();
    if (value && !isMemoryAddress(value))
      continue;
    VPTOMemoryAccess access;
    access.address = value;
    access.addressSpace = value ? getAddressSpace(value) : Attribute();
    access.reads = isa<MemoryEffects::Read>(effect.getEffect());
    access.writes =
        isa<MemoryEffects::Write, MemoryEffects::Allocate, MemoryEffects::Free>(
            effect.getEffect());
    if (value && storeLike)
      access.writes = true;
    access.unknown = !value || (!access.reads && !access.writes);
    setStaticAccessRange(op, access);
    accesses.push_back(access);
  }

  bool ordered =
      llvm::any_of(semantics.effects, [](const VPTOSchedulingEffect &effect) {
        return effect.kind == VPTOSchedulingEffectKind::AtomicMemory ||
               effect.kind == VPTOSchedulingEffectKind::VolatileMemory;
      });
  if (!ordered) {
    if (accesses.empty())
      semantics.memoryBehavior = VPTOMemoryBehavior::None;
    return;
  }
  if (accesses.empty()) {
    VPTOMemoryAccess access;
    access.reads = true;
    access.writes = true;
    access.ordered = true;
    access.unknown = true;
    accesses.push_back(access);
  }
  for (VPTOMemoryAccess &access : accesses)
    access.ordered = true;
}
} // namespace

VPTOSchedulingSemantics
mlir::pto::getDefaultVPTOSchedulingSemantics(Operation *op) {
  VPTOSchedulingSemantics semantics;
  SmallVectorImpl<VPTOSchedulingEffect> &effects = semantics.effects;
  if (isa<MemBarOp>(op)) {
    effects.push_back(
        {VPTOSchedulingEffectKind::Barrier, "memory-order", Value()});
  }
  if (isa<AtomicCasOp, AtomicExchOp, AtomicAddOp, AtomicSubOp, AtomicMinOp,
          AtomicMaxOp, AtomicAndOp, AtomicOrOp, AtomicXorOp>(op))
    effects.push_back(
        {VPTOSchedulingEffectKind::AtomicMemory, "memory", Value()});
  if (op->hasAttr("volatile") || op->hasAttr("is_volatile"))
    effects.push_back(
        {VPTOSchedulingEffectKind::VolatileMemory, "memory", Value()});
  auto addPostUpdate = [&](Value updatedBase) {
    if (updatedBase)
      effects.push_back({VPTOSchedulingEffectKind::PostUpdate,
                         "updated-address", updatedBase});
  };
  if (auto typedOp = dyn_cast<VldsOp>(op))
    addPostUpdate(typedOp.getUpdatedBase());
  if (auto typedOp = dyn_cast<Vldsx2Op>(op))
    addPostUpdate(typedOp.getUpdatedBase());
  if (auto typedOp = dyn_cast<SprstiOp>(op))
    addPostUpdate(typedOp.getUpdatedBase());
  if (auto typedOp = dyn_cast<SprstsOp>(op))
    addPostUpdate(typedOp.getUpdatedBase());
  if (auto typedOp = dyn_cast<VldusOp>(op))
    addPostUpdate(typedOp.getUpdatedBase());
  if (auto typedOp = dyn_cast<PldsOp>(op))
    addPostUpdate(typedOp.getUpdatedBase());
  if (auto typedOp = dyn_cast<PldiOp>(op))
    addPostUpdate(typedOp.getUpdatedBase());
  if (auto typedOp = dyn_cast<PstiOp>(op))
    addPostUpdate(typedOp.getUpdatedBase());
  if (auto typedOp = dyn_cast<VstsOp>(op))
    addPostUpdate(typedOp.getUpdatedBase());
  if (auto typedOp = dyn_cast<PstsOp>(op))
    addPostUpdate(typedOp.getUpdatedBase());
  if (auto typedOp = dyn_cast<VsldbOp>(op))
    addPostUpdate(typedOp.getUpdatedBase());
  if (auto typedOp = dyn_cast<VsstbOp>(op))
    addPostUpdate(typedOp.getUpdatedBase());
  if (auto typedOp = dyn_cast<VstasOp>(op))
    addPostUpdate(typedOp.getUpdatedBase());
  if (auto sprclr = dyn_cast<SprclrOp>(op))
    effects.push_back(
        {VPTOSchedulingEffectKind::ImplicitWrite, sprclr.getSpr(), Value()});
  if (auto sprsti = dyn_cast<SprstiOp>(op))
    effects.push_back(
        {VPTOSchedulingEffectKind::ImplicitRead, sprsti.getSpr(), Value()});
  if (auto sprsts = dyn_cast<SprstsOp>(op))
    effects.push_back(
        {VPTOSchedulingEffectKind::ImplicitRead, sprsts.getSpr(), Value()});
  if (isa<GetCtrlOp>(op))
    effects.push_back(
        {VPTOSchedulingEffectKind::ImplicitRead, "ctrl", Value()});
  if (isa<SetCtrlOp>(op))
    effects.push_back(
        {VPTOSchedulingEffectKind::ImplicitWrite, "ctrl", Value()});

  collectMemoryAccesses(op, semantics);
  if (getExecutionPipe(op) || !semantics.effects.empty()) {
    semantics.schedulingClass = VPTOSchedulingClass::Schedulable;
    semantics.classificationKnown = true;
  }
  return semantics;
}

VPTOSchedulingSemantics mlir::pto::getVPTOSchedulingSemantics(Operation *op) {
  VPTOSchedulingSemantics semantics;
  if (!op)
    return semantics;
  if (op->hasTrait<OpTrait::IsTerminator>() || op->getNumRegions() != 0) {
    semantics.classificationKnown = true;
    return semantics;
  }

  if (auto scheduling = dyn_cast<VPTOSchedulingOpInterface>(op))
    return scheduling.getVPTOSchedulingSemantics();

  if (isMemoryEffectFree(op)) {
    semantics.schedulingClass = VPTOSchedulingClass::Structural;
    semantics.classificationKnown = true;
    semantics.memoryBehavior = VPTOMemoryBehavior::None;
    return semantics;
  }

  // Any operation that reaches this fallback has no explicit scheduling
  // classification. Keep it out of scheduling regions conservatively and let
  // function coverage report the missing classification. Unsupported is
  // reserved for operations whose scheduling interface returns it explicitly.
  return semantics;
}

VPTOSchedulingClass mlir::pto::classifyVPTOSchedulingOp(Operation *op) {
  return getVPTOSchedulingSemantics(op).schedulingClass;
}

StringRef mlir::pto::stringifyVPTOSchedulingClass(VPTOSchedulingClass value) {
  switch (value) {
  case VPTOSchedulingClass::Schedulable:
    return "schedulable";
  case VPTOSchedulingClass::Structural:
    return "structural";
  case VPTOSchedulingClass::SchedulingBoundary:
    return "boundary";
  case VPTOSchedulingClass::Unsupported:
    return "unsupported";
  }
  llvm_unreachable("unknown VPTO scheduling class");
}

// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- VPTOScheduling.cpp - VPTO scheduling semantics --------------------===//

#include "PTO/IR/VPTOScheduling.h"
#include "PTO/IR/PTO.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::pto;

VPTOSchedulingClass mlir::pto::getDefaultVPTOSchedulingClass(Operation *op) {
  (void)op;
  return VPTOSchedulingClass::Schedulable;
}

void mlir::pto::getDefaultVPTOSchedulingEffects(
    Operation *op, SmallVectorImpl<VPTOSchedulingEffect> &effects) {
  StringRef name = op->getName().getStringRef();
  if (name == "pto.syncthreads" || name == "pto.threadfence" ||
      name == "pto.threadfence_block") {
    effects.push_back(
        {VPTOSchedulingEffectKind::Barrier, "simt-memory", Value()});
  }
  if (name.contains("atomic"))
    effects.push_back(
        {VPTOSchedulingEffectKind::AtomicMemory, "memory", Value()});
  if (name == "pto.sprclr")
    effects.push_back(
        {VPTOSchedulingEffectKind::ImplicitWrite, "special-register", Value()});
  if (name == "pto.sprsti" || name == "pto.sprsts")
    effects.push_back(
        {VPTOSchedulingEffectKind::ImplicitRead, "special-register", Value()});
}

VPTOSchedulingClass mlir::pto::classifyVPTOSchedulingOp(Operation *op) {
  if (!op || op->hasTrait<OpTrait::IsTerminator>() || op->getNumRegions() != 0)
    return VPTOSchedulingClass::SchedulingBoundary;

  if (auto scheduling = dyn_cast<VPTOSchedulingOpInterface>(op))
    return scheduling.getVPTOSchedulingClass();

  if (isMemoryEffectFree(op))
    return VPTOSchedulingClass::Structural;

  if (op->getDialect() &&
      op->getDialect()->getNamespace() == PTODialect::getDialectNamespace())
    return VPTOSchedulingClass::Unsupported;

  return VPTOSchedulingClass::SchedulingBoundary;
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

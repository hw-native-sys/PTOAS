// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- VPTOScheduling.h - VPTO scheduling semantics -----------*- C++ -*-===//
//
// This file contains the IR-level contract used by the VPTO scheduler.  Ops
// describe only their local, non-SSA scheduling semantics here; dependency and
// alias decisions remain the responsibility of the scheduling DAG builder.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_VPTOSCHEDULING_H
#define MLIR_DIALECT_PTO_IR_VPTOSCHEDULING_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::pto {

enum class VPTOSchedulingClass {
  Schedulable,
  Structural,
  SchedulingBoundary,
  Unsupported,
};

enum class VPTOSchedulingEffectKind {
  ImplicitRead,
  ImplicitWrite,
  Barrier,
  Event,
  Pipe,
  BufferId,
  VolatileMemory,
  AtomicMemory,
  Unknown,
};

/// One op-local effect which is not represented by ordinary SSA def-use or by
/// MemoryEffectOpInterface.  `resource` names an implicit state domain; `value`
/// optionally carries a dynamic event/buffer/address identity.
struct VPTOSchedulingEffect {
  VPTOSchedulingEffectKind kind = VPTOSchedulingEffectKind::Unknown;
  llvm::StringRef resource;
  Value value;
};

/// Classify an operation at the VPTO emission scheduling boundary.  Unknown or
/// unsupported operations must be treated as region boundaries by clients.
VPTOSchedulingClass classifyVPTOSchedulingOp(Operation *op);

/// Default implementations used by the scheduling op interface carried by
/// VPTO micro-op families.  Individual ops can override the interface when a
/// future target needs more precise semantics.
VPTOSchedulingClass getDefaultVPTOSchedulingClass(Operation *op);
void getDefaultVPTOSchedulingEffects(
    Operation *op, SmallVectorImpl<VPTOSchedulingEffect> &effects);

llvm::StringRef stringifyVPTOSchedulingClass(VPTOSchedulingClass value);

} // namespace mlir::pto

#endif // MLIR_DIALECT_PTO_IR_VPTOSCHEDULING_H

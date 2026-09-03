// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace mlir::pto {

enum class VPTOSchedulingClass {
  Schedulable,
  Structural,
  SchedulingBoundary,
  Unsupported,
};

/// Number of `VPTOSchedulingClass` values; keep in sync with the enum above.
inline constexpr unsigned kNumVPTOSchedulingClasses = 4;

enum class VPTOSchedulingEffectKind {
  ImplicitRead,
  ImplicitWrite,
  Barrier,
  PostUpdate,
  VolatileMemory,
  AtomicMemory,
  Unknown,
};

/// Completeness of an operation's ordinary memory semantics. `Unknown` means
/// the operation lacks a complete declaration and must receive a conservative
/// memory access; it does not describe an access with an unknown address.
enum class VPTOMemoryBehavior {
  None,
  Explicit,
  Unknown,
};

/// One op-local effect which is not represented by ordinary SSA def-use or by
/// MemoryEffectOpInterface. `resource` names an implicit state domain and
/// `value` optionally carries an SSA identity such as a post-update address.
struct VPTOSchedulingEffect {
  VPTOSchedulingEffect() = default;
  VPTOSchedulingEffect(VPTOSchedulingEffectKind kind, llvm::StringRef resource,
                       Value value = {})
      : kind(kind), resource(resource), value(value) {}

  VPTOSchedulingEffectKind kind = VPTOSchedulingEffectKind::Unknown;
  llvm::StringRef resource;
  Value value;
};

/// One normalized memory access owned by an operation. The operation semantic
/// layer describes local access facts; alias roots and conflicts between two
/// accesses are derived later by the scheduling DAG builder.
struct VPTOMemoryAccess {
  Value address;
  Attribute addressSpace;
  std::optional<int64_t> byteOffset;
  std::optional<int64_t> byteSize;
  bool reads = false;
  bool writes = false;
  bool ordered = false;
  bool unknown = false;
};

/// Stable, operation-local input consumed by scheduling analyses. Clients must
/// not recover operation-specific scheduling facts from names or operand
/// positions after this structure has been produced.
struct VPTOSchedulingSemantics {
  VPTOSchedulingClass schedulingClass = VPTOSchedulingClass::SchedulingBoundary;
  bool classificationKnown = false;
  SmallVector<VPTOSchedulingEffect> effects;
  VPTOMemoryBehavior memoryBehavior = VPTOMemoryBehavior::Unknown;
  SmallVector<VPTOMemoryAccess> memoryAccesses;
};

/// Return the normalized semantics for any operation at the VPTO emission
/// scheduling boundary. Operations implementing VPTOSchedulingOpInterface
/// provide their semantic record through that interface; other operations are
/// classified conservatively.
VPTOSchedulingSemantics getVPTOSchedulingSemantics(Operation *op);

/// Classify an operation at the VPTO emission scheduling boundary.  Unknown or
/// unsupported operations must be treated as region boundaries by clients.
VPTOSchedulingClass classifyVPTOSchedulingOp(Operation *op);

/// Default implementations used by the scheduling op interface carried by
/// VPTO micro-op families.  Individual ops can override the interface when a
/// future target needs more precise semantics.
VPTOSchedulingSemantics getDefaultVPTOSchedulingSemantics(Operation *op);

llvm::StringRef stringifyVPTOSchedulingClass(VPTOSchedulingClass value);

} // namespace mlir::pto

#endif // MLIR_DIALECT_PTO_IR_VPTOSCHEDULING_H

// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMIControlFlowSupport.h - VMI SCF loop constraints ----------------===//
//===----------------------------------------------------------------------===//

#ifndef PTO_TRANSFORMS_VMICONTROLFLOWSUPPORT_H
#define PTO_TRANSFORMS_VMICONTROLFLOWSUPPORT_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/STLFunctionalExtras.h"

namespace mlir {

namespace pto {

/// Shared SCF loop-carried value constraints used by data-layout and mask-
/// granularity assignment.  The callback owns the domain-specific union/find
/// operation; this component only describes the SSA relationships of SCF
/// loops.
class VMIControlFlowSupport {
public:
  using EquivalenceCallback =
      llvm::function_ref<LogicalResult(Value, Value, Operation *)>;

  static LogicalResult addForConstraints(scf::ForOp forOp,
                                         EquivalenceCallback addEquivalent);
  static LogicalResult addWhileConstraints(
      scf::WhileOp whileOp, EquivalenceCallback addEquivalent);
};

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_VMICONTROLFLOWSUPPORT_H

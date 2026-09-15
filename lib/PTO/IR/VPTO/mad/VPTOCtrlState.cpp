// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOCtrlState.cpp - CTRL state interfaces and guard ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementations of StateAccessOpInterface for the explicit CTRL access ops
// and raw MAD consumers, StateGuardOpInterface for CtrlStateGuardOp, and the
// CtrlStateGuardOp verifier. These describe implicit CTRL state semantics;
// they do not decide where hardware get_ctrl/set_ctrl accesses are emitted.
//
//===----------------------------------------------------------------------===//

#include "VPTOInternal.h"

using namespace mlir;
using namespace mlir::pto;

//===----------------------------------------------------------------------===//
// StateAccessOpInterface: explicit CTRL access ops
//===----------------------------------------------------------------------===//

StateResource GetCtrlOp::getStateResource() { return StateResource::Ctrl; }

StateAccessKind GetCtrlOp::getStateAccessKind() { return StateAccessKind::Query; }

Value GetCtrlOp::getAccessedStateValue() { return getResult(); }

StateResource SetCtrlOp::getStateResource() { return StateResource::Ctrl; }

StateAccessKind SetCtrlOp::getStateAccessKind() { return StateAccessKind::Write; }

Value SetCtrlOp::getAccessedStateValue() { return getValue(); }

//===----------------------------------------------------------------------===//
// StateAccessOpInterface: raw MAD consumers
//===----------------------------------------------------------------------===//

StateResource MadRawOp::getStateResource() { return StateResource::Ctrl; }

StateAccessKind MadRawOp::getStateAccessKind() {
  return StateAccessKind::Consume;
}

Value MadRawOp::getAccessedStateValue() { return {}; }

StateResource MadBiasRawOp::getStateResource() { return StateResource::Ctrl; }

StateAccessKind MadBiasRawOp::getStateAccessKind() {
  return StateAccessKind::Consume;
}

Value MadBiasRawOp::getAccessedStateValue() { return {}; }

StateResource MadMxRawOp::getStateResource() { return StateResource::Ctrl; }

StateAccessKind MadMxRawOp::getStateAccessKind() {
  return StateAccessKind::Consume;
}

Value MadMxRawOp::getAccessedStateValue() { return {}; }

StateResource MadMxBiasRawOp::getStateResource() { return StateResource::Ctrl; }

StateAccessKind MadMxBiasRawOp::getStateAccessKind() {
  return StateAccessKind::Consume;
}

Value MadMxBiasRawOp::getAccessedStateValue() { return {}; }

//===----------------------------------------------------------------------===//
// StateGuardOpInterface and verifier: CtrlStateGuardOp
//===----------------------------------------------------------------------===//

StateResource CtrlStateGuardOp::getStateResource() {
  return StateResource::Ctrl;
}

uint64_t CtrlStateGuardOp::getControlledStateBits() {
  return static_cast<uint64_t>(getControlledBitsAttr().getInt());
}

uint64_t CtrlStateGuardOp::getRequiredStateBits() {
  return static_cast<uint64_t>(getRequiredBitsAttr().getInt());
}

Region &CtrlStateGuardOp::getGuardedBody() { return getBody(); }

// The guard itself touches no memory; the RecursiveMemoryEffects trait
// forwards this query to the wrapped consumer (the raw MAD).
void CtrlStateGuardOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getOperation()->getRegion(0).walk([&](Operation *op) {
    if (auto memOp = dyn_cast<MemoryEffectOpInterface>(op)) {
      memOp.getEffects(effects);
    }
  });
}

static llvm::StringLiteral accessKindName(StateAccessKind kind) {
  switch (kind) {
  case StateAccessKind::Query:
    return "Query";
  case StateAccessKind::Write:
    return "Write";
  case StateAccessKind::Consume:
  case StateAccessKind::Clobber:
    return "Clobber";
  }
  return "Clobber";
}

static LogicalResult verifyGuardBits(CtrlStateGuardOp op) {
  uint64_t controlled = op.getControlledStateBits();
  uint64_t required = op.getRequiredStateBits();
  if (controlled == 0) {
    return op.emitOpError("controlled_bits must be non-zero; an empty guard "
                          "controls no CTRL bits and has no legal use");
  }
  if (required & ~controlled) {
    return op.emitOpError()
           << "required_bits (" << required
           << ") must be a subset of controlled_bits (" << controlled << ")";
  }
  return success();
}

static LogicalResult verifyGuardBody(CtrlStateGuardOp op) {
  Block &body = op.getBody().front();
  size_t consumeCount = 0;
  for (Operation &bodyOp : body) {
    auto access = dyn_cast<StateAccessOpInterface>(bodyOp);
    if (!access || access.getStateResource() != StateResource::Ctrl) {
      continue;
    }
    if (access.getStateAccessKind() != StateAccessKind::Consume) {
      return bodyOp.emitOpError()
             << "inside ctrl_state_guard must only CTRL-Consume; found a "
             << accessKindName(access.getStateAccessKind()) << " access";
    }
    if (!isa<MadRawOpInterface>(bodyOp)) {
      return bodyOp.emitOpError(
          "inside ctrl_state_guard must implement MadRawOpInterface; "
          "new CTRL guard producers only need to widen this consumer check");
    }
    ++consumeCount;
  }
  if (consumeCount != 1) {
    return op.emitOpError() << "requires exactly one CTRL Consume operation "
                               "in the body, found "
                            << consumeCount;
  }
  return success();
}

static LogicalResult verifyGuardPlacement(CtrlStateGuardOp op) {
  if (!op.getBody().hasOneBlock()) {
    return op.emitOpError("body must be a single block");
  }
  if (!op->getParentOfType<func::FuncOp>()) {
    return op.emitOpError("must be nested inside a func.func");
  }
  if (isa_and_nonnull<CtrlStateGuardOp>(op->getParentOp())) {
    return op.emitOpError("must not be nested inside another ctrl_state_guard");
  }
  return success();
}

LogicalResult CtrlStateGuardOp::verify() {
  if (failed(verifyGuardBits(*this))) {
    return failure();
  }
  if (failed(verifyGuardBody(*this))) {
    return failure();
  }
  return verifyGuardPlacement(*this);
}

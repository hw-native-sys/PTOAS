// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOSimtLaunch.cpp - pto.SimtLaunch methods ------------------------===//
//===----------------------------------------------------------------------===//

#include "VPTOMemOpInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::memop_detail;

LogicalResult SimtLaunchOp::verify() {
  if (auto parentFunc = (*this)->getParentOfType<func::FuncOp>()) {
    if (parentFunc->hasAttr(pto::kPTOSimtEntryAttrName)) {
      return emitOpError()
             << "must not appear inside a function marked with '"
             << pto::kPTOSimtEntryAttrName
             << "'; launch the SIMT entry from an outer non-simt function";
    }
  }

  func::FuncOp callee =
      SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(*this, getCalleeAttr());
  if (!callee) {
    return emitOpError() << "'" << getCalleeAttr().getValue()
                         << "' does not reference a valid function";
  }

  if (!callee->hasAttr(pto::kPTOSimtEntryAttrName)) {
    return emitOpError() << "callee '" << getCalleeAttr().getValue()
                         << "' must be marked with '"
                         << pto::kPTOSimtEntryAttrName << "'";
  }

  FunctionType calleeType = callee.getFunctionType();
  if (!calleeType.getResults().empty()) {
    return emitOpError("requires a callee with no results");
  }

  if (calleeType.getNumInputs() != getArgs().size()) {
    return emitOpError("incorrect number of operands for callee");
  }

  for (auto [index, argType, operand] :
       llvm::enumerate(calleeType.getInputs(), getArgs())) {
    if (argType != operand.getType()) {
      return emitOpError("operand type mismatch: expected operand type ")
             << argType << ", but provided " << operand.getType()
             << " for operand number " << index;
    }
  }
  return success();
}

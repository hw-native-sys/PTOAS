// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMIMaskUtils.cpp - Shared VMI predicate / seed helpers -------------===//

#include "PTO/Transforms/VMIMaskUtils.h"

#include "PTO/IR/PTO.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"

using namespace mlir;
using namespace mlir::pto;

bool mlir::pto::isAllActiveSeed(Value seed) {
  Operation *def = seed.getDefiningOp();
  if (!def)
    return false;
  if (isa<VMIPsetOp>(def))
    return true;
  if (auto cm = dyn_cast<VMICreateMaskOp>(def)) {
    auto maskTy = cast<VMIMaskType>(cm.getResult().getType());
    if (auto cst = cm.getActiveLanes().getDefiningOp<arith::ConstantOp>())
      if (auto ia = dyn_cast<IntegerAttr>(cst.getValue()))
        return ia.getInt() >= maskTy.getElementCount();
  }
  return false;
}

bool mlir::pto::isAllInactiveSeed(Value seed) {
  Operation *def = seed.getDefiningOp();
  if (!def)
    return false;
  if (auto cm = dyn_cast<VMICreateMaskOp>(def)) {
    if (auto cst = cm.getActiveLanes().getDefiningOp<arith::ConstantOp>())
      if (auto ia = dyn_cast<IntegerAttr>(cst.getValue()))
        return ia.getInt() <= 0;
  }
  return false;
}

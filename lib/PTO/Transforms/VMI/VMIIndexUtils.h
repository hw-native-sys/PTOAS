// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#pragma once

// Shared helpers for VMI passes that inspect constant index operands.
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"

#include <optional>

namespace mlir {
namespace pto {

inline std::optional<int64_t> getConstantIndexValue(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>()) {
    return constant.value();
  }
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto integerAttr = dyn_cast<IntegerAttr>(constant.getValue())) {
      return integerAttr.getInt();
    }
  }
  return std::nullopt;
}

} // namespace pto
} // namespace mlir

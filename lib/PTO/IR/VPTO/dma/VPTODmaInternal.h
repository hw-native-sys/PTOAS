// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTODmaInternal.h - shared DMA dist-token width helper -------------===//
//===----------------------------------------------------------------------===//
//
// Internal to lib/PTO/IR/VPTO/dma; not installed.
//===----------------------------------------------------------------------===//

#ifndef PTO_IR_VPTO_DMA_INTERNAL_H
#define PTO_IR_VPTO_DMA_INTERNAL_H

#include "VPTOInternal.h"

inline std::optional<unsigned> getDmaDistElementWidth(mlir::Type type) {
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
    return intType.getWidth();
  }
  if (type.isF16() || type.isBF16()) {
    return mlir::pto::kValue16;
  }
  if (type.isF32()) {
    return mlir::pto::kValue32;
  }
  if (type.isF64()) {
    return mlir::pto::kValue64;
  }
  return std::nullopt;
}

#endif // PTO_IR_VPTO_DMA_INTERNAL_H

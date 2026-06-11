// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOUbOps.cpp --------------------------------------------------===//

#include "PTO/IR/PTO.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;

namespace mlir {
namespace pto {

//===----------------------------------------------------------------------===//
// UBSetMaskOp
//===----------------------------------------------------------------------===//

void UBSetMaskOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

LogicalResult UBSetMaskOp::verify() {
  return success();
}

//===----------------------------------------------------------------------===//
// UBSetMaskCountOp
//===----------------------------------------------------------------------===//

void UBSetMaskCountOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

LogicalResult UBSetMaskCountOp::verify() {
  return success();
}

//===----------------------------------------------------------------------===//
// UBSetMaskNormOp
//===----------------------------------------------------------------------===//

void UBSetMaskNormOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

LogicalResult UBSetMaskNormOp::verify() {
  return success();
}

} // namespace pto
} // namespace mlir

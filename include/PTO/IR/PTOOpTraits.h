// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef MLIR_DIALECT_PTO_IR_PTOOPTRAITS_H_
#define MLIR_DIALECT_PTO_IR_PTOOPTRAITS_H_

#include "mlir/IR/OpDefinition.h"

namespace mlir::pto::detail {
LogicalResult verifyFrontendInitPipeTrait(Operation *op);
LogicalResult verifyFrontendPopTrait(Operation *op);
} // namespace mlir::pto::detail

namespace mlir::OpTrait::pto {

template <typename ConcreteType>
class FrontendInitPipeVerify
    : public TraitBase<ConcreteType, FrontendInitPipeVerify> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return ::mlir::pto::detail::verifyFrontendInitPipeTrait(op);
  }
};

template <typename ConcreteType>
class FrontendPopVerify : public TraitBase<ConcreteType, FrontendPopVerify> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return ::mlir::pto::detail::verifyFrontendPopTrait(op);
  }
};

} // namespace mlir::OpTrait::pto

#endif // MLIR_DIALECT_PTO_IR_PTOOPTRAITS_H_

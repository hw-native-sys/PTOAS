// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMIMaskRematerialization.h - Shared VMI mask rematerialization ----===//
//===----------------------------------------------------------------------===//

#ifndef PTO_TRANSFORMS_VMI_VMIMASKREMATERIALIZATION_H
#define PTO_TRANSFORMS_VMI_VMIMASKREMATERIALIZATION_H

#include "PTO/IR/PTO.h"
#include "PTO/IR/VMIUtils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

#include <optional>

namespace mlir::pto::mask_rematerialization {

inline std::optional<Value> rematerializeMaskProducer(
    Value value, VMIMaskType resultType, Location loc, OpBuilder& builder)
{
    if (isVMIUnspecifiedValue(value.getDefiningOp())) {
        return builder.create<PTOUnspecifiedValueOp>(loc, resultType).getResult();
    }

    if (auto createMask = value.getDefiningOp<VMICreateMaskOp>()) {
        return builder.create<VMICreateMaskOp>(loc, resultType, createMask.getActiveLanes()).getResult();
    }

    if (auto createGroupMask = value.getDefiningOp<VMICreateGroupMaskOp>()) {
        return builder
            .create<VMICreateGroupMaskOp>(
                loc, resultType, createGroupMask.getActiveElemsPerGroup(), createGroupMask.getNumGroupsAttr(),
                createGroupMask.getGroupSizeAttr())
            .getResult();
    }

    if (auto constantMask = value.getDefiningOp<VMIConstantMaskOp>()) {
        return builder.create<VMIConstantMaskOp>(loc, resultType, constantMask.getValueAttr()).getResult();
    }

    return std::nullopt;
}

} // namespace mlir::pto::mask_rematerialization

#endif // PTO_TRANSFORMS_VMI_VMIMASKREMATERIALIZATION_H

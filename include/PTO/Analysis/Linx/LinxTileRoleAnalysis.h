// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// C++ API roles are inferred from uses; they are not IR storage locations.
#ifndef PTO_ANALYSIS_LINX_TILE_ROLE_ANALYSIS_H
#define PTO_ANALYSIS_LINX_TILE_ROLE_ANALYSIS_H
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
namespace mlir::pto::linx {
enum class TileRole { Vec, Left, Right };
class LinxTileRoleAnalysis {
public:
    LogicalResult run(ModuleOp module);
    TileRole getRole(Value value) const;
private:
    struct Requirement {
        TileRole role;
        Operation* origin;
    };
    LogicalResult require(Value value, TileRole role, Operation* origin);
    llvm::DenseMap<Value, Requirement> requirements;
};
} // namespace mlir::pto::linx
#endif

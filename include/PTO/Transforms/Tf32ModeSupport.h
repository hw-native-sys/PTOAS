// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_TF32MODESUPPORT_H
#define MLIR_DIALECT_PTO_TRANSFORMS_TF32MODESUPPORT_H

#include "PTO/IR/PTO.h"

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::pto {

/// True when `op` carries HF32 rounding semantics of its own: the tile matmul
/// family lowers `tf32_mode` to a CTRL scope in the EmitC backend, and the micro
/// MAD family lowers it to a CTRL state guard in the VPTO backend.
inline bool supportsTf32Mode(Operation *op)
{
    if (isa<TMatmulOp, TMatmulAccOp, TMatmulBiasOp>(op)) {
        return true;
    }
    return isa<MadSemanticOpInterface>(op);
}

/// Rejects `tf32_mode` on every operation that does not implement it. Ops accept
/// arbitrary attributes through their dictionary, so without this check a
/// precision request on, say, a `pto.tgemv` or a f16 matmul variant would be
/// silently ignored instead of diagnosed.
inline LogicalResult verifyTf32ModeSupported(Operation *op)
{
    if (!op->hasAttr("tf32_mode") || supportsTf32Mode(op)) {
        return success();
    }
    return op->emitOpError()
        << "does not support tf32_mode; only f32 tile matmul "
        << "(pto.tmatmul, pto.tmatmul.acc, pto.tmatmul.bias) and the "
        << "micro MAD family accept it";
}

} // namespace mlir::pto

#endif // MLIR_DIALECT_PTO_TRANSFORMS_TF32MODESUPPORT_H

// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#pragma once

#include <cstdint>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
class MLIRContext;
}

namespace ptobc {

/// Decode a PTOBC v0 buffer (the full file contents, including header)
/// into an MLIR module.
///
/// Throws std::runtime_error on malformed input.
mlir::OwningOpRef<mlir::ModuleOp>
decodePTOBCToModule(llvm::ArrayRef<uint8_t> fileBytes, mlir::MLIRContext &ctx);

} // namespace ptobc

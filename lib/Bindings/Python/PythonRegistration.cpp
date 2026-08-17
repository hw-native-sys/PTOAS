// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "mlir-c/Dialect/Arith.h"
#include "mlir-c/Dialect/Func.h"
#include "mlir-c/Dialect/LLVM.h"
#include "mlir-c/Dialect/Math.h"
#include "mlir-c/Dialect/MemRef.h"
#include "mlir-c/Dialect/SCF.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

PYBIND11_MODULE(_site_initialize_0, module) {
  module.doc() = "PTOAS MLIR dialect registration";
  module.def("register_dialects", [](MlirDialectRegistry registry) {
    mlirDialectHandleInsertDialect(mlirGetDialectHandle__arith__(), registry);
    mlirDialectHandleInsertDialect(mlirGetDialectHandle__func__(), registry);
    mlirDialectHandleInsertDialect(mlirGetDialectHandle__llvm__(), registry);
    mlirDialectHandleInsertDialect(mlirGetDialectHandle__math__(), registry);
    mlirDialectHandleInsertDialect(mlirGetDialectHandle__memref__(), registry);
    mlirDialectHandleInsertDialect(mlirGetDialectHandle__scf__(), registry);
  });
}

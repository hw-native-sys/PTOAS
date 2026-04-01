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

#include <string>

#include <mlir/IR/BuiltinOps.h>

namespace ptobc {

struct CanonicalPrintOptions {
  /// If true, print operations in MLIR generic form (quoted op names).
  bool generic = false;

  /// If true, keep MLIR's default float printing. If false, force scalar
  /// FloatAttr constants to be printed as hex bitpatterns (`0x... : f32`).
  bool keepMLIRFloatPrinting = false;

  /// If true, print `loc(...)` debug locations (parseable form).
  bool printDebugInfo = false;
};

/// Print a ModuleOp in a canonical, parseable `.pto` form.
///
/// Today this is implemented as: MLIR pretty printer + targeted canonicalization
/// of scalar float constants.
std::string printModuleCanonical(mlir::ModuleOp module,
                                 const CanonicalPrintOptions &opt = {});

} // namespace ptobc

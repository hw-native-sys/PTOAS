// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOOps.td - Aggregate PTO operation definitions --------*- tablegen -*-===//
//===----------------------------------------------------------------------===//
//
// Compatibility entry point for all PTO operation families. New operation
// definitions belong in their semantic family file.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_PTOOPS
#define MLIR_DIALECT_PTO_IR_PTOOPS

include "PTO/IR/PTOBaseOps.td"
include "PTO/IR/PTOTileOps.td"
include "PTO/IR/VMIOps.td"
include "PTO/IR/VPTOOps.td"
include "PTO/IR/VPTOUbOps.td"

#endif // MLIR_DIALECT_PTO_IR_PTOOPS

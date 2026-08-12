// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTO.h - PTO VPTO operation declarations --------------*- C++ -*-===//

#ifndef MLIR_DIALECT_PTO_IR_VPTO_H_
#define MLIR_DIALECT_PTO_IR_VPTO_H_

#include "PTO/IR/PTOCommon.h"

#define GET_OP_CLASSES
#include "PTO/IR/VPTOOps.h.inc"

#define GET_OP_CLASSES
#include "PTO/IR/VPTOUbOps.h.inc"

#endif // MLIR_DIALECT_PTO_IR_VPTO_H_

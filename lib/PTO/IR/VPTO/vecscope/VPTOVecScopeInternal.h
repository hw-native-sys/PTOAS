// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOVecScopeInternal.h - shared VPTOVecScope helpers ---------------===//
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Shared helpers. They live in a detail namespace so the original
// unqualified MLIR/LLVM names keep resolving.
// Internal to lib/PTO/IR/VPTO/vecscope; not installed.
//===----------------------------------------------------------------------===//

#ifndef PTO_IR_VPTO_VPTOVECSCOPE_INTERNAL_H
#define PTO_IR_VPTO_VPTOVECSCOPE_INTERNAL_H

#include "VPTOInternal.h"

namespace mlir::pto::vecscope_detail {

using namespace mlir;
using namespace mlir::pto;


} // namespace mlir::pto::vecscope_detail

#endif // PTO_IR_VPTO_VPTOVECSCOPE_INTERNAL_H

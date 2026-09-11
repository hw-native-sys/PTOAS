// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOMemOp.cpp - shared VPTO memop global helper --------------------===//
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Externally linked memop helpers declared in VPTOInternal.h; the per-instruction
// helpers live in memop/VPTOMemOpInternal.h.
//===----------------------------------------------------------------------===//

#include "VPTOInternal.h"

using namespace mlir;
using namespace mlir::pto;

bool isSupportedPredicatePattern(StringRef pattern) {
  return pattern == "PAT_ALL" || pattern == "PAT_VL1" || pattern == "PAT_VL2" ||
         pattern == "PAT_VL3" || pattern == "PAT_VL4" || pattern == "PAT_VL8" ||
         pattern == "PAT_VL16" || pattern == "PAT_VL32" ||
         pattern == "PAT_VL64" || pattern == "PAT_VL128" ||
         pattern == "PAT_M3" || pattern == "PAT_M4" || pattern == "PAT_H" ||
         pattern == "PAT_Q" || pattern == "PAT_ALLF";
}

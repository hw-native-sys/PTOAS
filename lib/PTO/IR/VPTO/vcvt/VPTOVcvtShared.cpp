// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOVcvtShared.cpp - shared VPTOVcvt externally linked helpers -----===//
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Externally linked helpers declared in VPTOInternal.h (definitions here keep
// their global scope/linkage); other shared helpers live in the internal header.
//===----------------------------------------------------------------------===//

#include "VPTOVcvtInternal.h"

using namespace mlir;
using namespace mlir::pto;

std::optional<StringRef> normalizeEvenOddPartToken(StringRef token) {
  if (token == "EVEN" || token == "PART_EVEN") {
    return StringRef("EVEN");
  }
  if (token == "ODD" || token == "PART_ODD") {
    return StringRef("ODD");
  }
  return std::nullopt;
}

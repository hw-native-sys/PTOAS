// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOVecOp.cpp - shared VPTOVecOp externally linked helpers ---------===//
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Externally linked helpers declared in VPTOInternal.h (definitions here keep
// their global scope/linkage); other shared helpers live in the internal header.
//===----------------------------------------------------------------------===//

#include "VPTOVecOpInternal.h"

using namespace mlir;
using namespace mlir::pto;

bool isMxElementType(Type type) {
  return isa<Float8E4M3FNType, Float8E5M2Type>(type) ||
         isa<pto::F4E1M2x2Type, pto::F4E2M1x2Type>(type);
}

bool isSupportedPostMode(StringRef mode) {
  return mode == "NO_POST_UPDATE" || mode == "POST_UPDATE";
}

bool isCompatibleScalarForSemanticType(Type semanticType,
                                              Type scalarType) {
  if (semanticType == scalarType) {
    return true;
  }

  auto semanticInt = dyn_cast<IntegerType>(semanticType);
  auto scalarInt = dyn_cast<IntegerType>(scalarType);
  if (!semanticInt || !scalarInt || semanticInt.getWidth() != scalarInt.getWidth()) {
    return false;
  }

  return isSemanticSignCompatible(semanticInt, scalarInt);
}

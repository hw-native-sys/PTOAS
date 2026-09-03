// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- VPTOBridgeTokens.cpp - C++ template token building ----------------===//
//===----------------------------------------------------------------------===//
//
// Implementation of the bridge-side PTO-ISA C++ template token builders. See
// include/PTO/Transforms/VPTOBridgeTokens.h. Both the IR-fact -> C++ spelling
// mapping rules and the bridge assembly rules (fully qualified spellings,
// NoneBox trailing-argument omission) live here.
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/VPTOBridgeTokens.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "llvm/ADT/Twine.h"
#include <string>

using namespace mlir;
using namespace mlir::pto;

using namespace mlir;
using namespace mlir::pto;

FailureOr<std::string> pto::buildBridgeElementTypeToken(Type elementType) {
  if (pto::isPTOFloat8E4M3LikeType(elementType))
    return std::string("float8_e4m3_t");
  if (pto::isPTOFloat8E5M2LikeType(elementType))
    return std::string("float8_e5m2_t");
  if (pto::isPTOF8E8M0Type(elementType))
    return std::string("float8_e8m0_t");
  if (isa<pto::HiF8Type>(elementType))
    return std::string("hifloat8_t");
  if (isa<pto::F4E1M2x2Type>(elementType))
    return std::string("float4_e1m2x2_t");
  if (isa<pto::F4E2M1x2Type>(elementType))
    return std::string("float4_e2m1x2_t");
  if (elementType.isF16())
    return std::string("half");
  if (elementType.isBF16())
    return std::string("bfloat16_t");
  if (elementType.isF32())
    return std::string("float");
  if (elementType.isF64())
    return std::string("double");
  if (elementType.isInteger(8))
    return (elementType.isSignlessInteger(8) || elementType.isSignedInteger(8))
               ? std::string("int8_t")
               : std::string("uint8_t");
  if (elementType.isInteger(16))
    return (elementType.isSignlessInteger(16) ||
            elementType.isSignedInteger(16))
               ? std::string("int16_t")
               : std::string("uint16_t");
  if (elementType.isInteger(32))
    return (elementType.isSignlessInteger(32) ||
            elementType.isSignedInteger(32))
               ? std::string("int32_t")
               : std::string("uint32_t");
  if (elementType.isInteger(64))
    return cast<IntegerType>(elementType).isUnsigned() ? std::string("uint64_t")
                                                       : std::string("int64_t");
  return failure();
}

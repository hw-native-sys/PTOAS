// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOStructuredAccPrint.cpp - structured acc-store asm print entry points ===//
//===----------------------------------------------------------------------===//

#include "VPTOStructuredAccInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::structuredacc_detail;

void printStructuredAccStoreClauses(
    OpAsmPrinter &printer, std::optional<AccStoreUnitFlagCtrl> unitFlag,
    Value preQuant,
    std::optional<AccStoreQuantPreMode> preQuantMode, Value preRelu,
    std::optional<ReluPreMode> preReluMode, Value clipValue,
    std::optional<AccStoreMode> mode, Value split, Value loop0SrcStride,
    Value loop3Count, Value loop3SrcStride, Value loop3DstStride,
    std::optional<AccStoreSatMode> satMode,
    std::optional<AccStoreAtomicType> atomicType,
    std::optional<AccStoreAtomicOp> atomicOp) {
  if (unitFlag && *unitFlag != AccStoreUnitFlagCtrl::Off) {
    printer << ", unit_flag("
            << (*unitFlag == AccStoreUnitFlagCtrl::CheckOnly ? "check_only"
                                                             : "check_and_clear")
            << ")";
  }
  if (preQuantMode) {
    printer << ", pre_quant(" << preQuant << ", mode = "
            << stringifyAccStoreQuantPreMode(*preQuantMode) << ")";
  }
  if (preReluMode) {
    printer << ", pre_relu(";
    if (preRelu) {
      printer << preRelu << ", ";
    }
    printer << "mode = " << stringifyReluPreMode(*preReluMode);
    if (clipValue) {
      printer << ", clip = " << clipValue;
    }
    printer << ")";
  }
  if (mode) {
    printStructuredAccStoreMode(printer, *mode, split, loop0SrcStride);
  }
  if (loop3Count) {
    printer << ", loop3(" << loop3Count << ", " << loop3SrcStride << ", "
            << loop3DstStride << ")";
  }
  if (satMode) {
    printStructuredAccStoreSatMode(printer, *satMode);
  }
  if (atomicType && atomicOp) {
    printer << ", atomic(type = " << stringifyAccStoreAtomicType(*atomicType)
            << ", op = " << stringifyAccStoreAtomicOp(*atomicOp) << ")";
  }
}

void printStructuredAccStoreOptionalTypes(
    OpAsmPrinter &printer, Value preQuant, Value preRelu, Value clipValue,
    Value split, Value loop0SrcStride, Value loop3Count, Value loop3SrcStride,
    Value loop3DstStride) {
  if (preQuant) {
    printer << ", " << preQuant.getType();
  }
  if (preRelu) {
    printer << ", " << preRelu.getType();
  }
  if (clipValue) {
    printer << ", " << clipValue.getType();
  }
  if (split) {
    printer << ", " << split.getType();
  }
  if (loop0SrcStride) {
    printer << ", " << loop0SrcStride.getType();
  }
  if (loop3Count) {
    printer << ", " << loop3Count.getType() << ", " << loop3SrcStride.getType()
            << ", " << loop3DstStride.getType();
  }
}

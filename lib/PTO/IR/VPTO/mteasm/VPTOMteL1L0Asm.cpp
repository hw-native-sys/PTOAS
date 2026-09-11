// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOMteL1L0Asm.cpp - pto.mte_l1_l0 optional-operand print helper ---===//
//===----------------------------------------------------------------------===//

#include "VPTOInternal.h"

using namespace mlir;
using namespace mlir::pto;

void printMteL1L0OptionalOperandsOp(
    OpAsmPrinter &printer, Operation *operation, Value source, Value destination,
    ArrayRef<Value> shapeOperands, ArrayRef<StringRef> shapeNames,
    ArrayRef<Value> fullOperands, ArrayRef<StringRef> fullNames) {
  const bool hasShape = llvm::any_of(shapeOperands, [](Value value) {
    return static_cast<bool>(value);
  });
  const bool hasFull = llvm::any_of(fullOperands, [](Value value) {
    return static_cast<bool>(value);
  });
  const bool isShapeForm = hasShape && !hasFull &&
      llvm::all_of(shapeOperands, [](Value value) { return static_cast<bool>(value); });
  const bool isFullForm = hasFull && !hasShape &&
      llvm::all_of(fullOperands, [](Value value) { return static_cast<bool>(value); });

  printer << " " << source << ", " << destination;
  SmallVector<Value, mlir::pto::kValue10> printedOperands;
  if (isShapeForm) {
    for (Value value : shapeOperands) {
      printer << ", " << value;
      printedOperands.push_back(value);
    }
  } else if (isFullForm) {
    for (Value value : fullOperands) {
      printer << ", " << value;
      printedOperands.push_back(value);
    }
  } else {
    for (auto [index, value] : llvm::enumerate(shapeOperands)) {
      if (!value) {
        continue;
      }
      printer << ", " << shapeNames[index] << "(" << value << ")";
      printedOperands.push_back(value);
    }
    for (auto [index, value] : llvm::enumerate(fullOperands)) {
      if (!value) {
        continue;
      }
      printer << ", " << fullNames[index] << "(" << value << ")";
      printedOperands.push_back(value);
    }
  }

  printer.printOptionalAttrDict(operation->getAttrs(),
                                /*elidedAttrs=*/{"operandSegmentSizes"});
  printer << " : " << source.getType() << ", " << destination.getType();
  for (Value value : printedOperands) {
    printer << ", " << value.getType();
  }
}

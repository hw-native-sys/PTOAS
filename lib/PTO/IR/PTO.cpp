// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTO.cpp - PTO Dialect ----------------------------------------------===//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOLayoutUtils.h"
#include "PTO/IR/PTOMultiBuffer.h"
#include "PTO/IR/PTOSyncUtils.h"
#include "PTO/IR/PTOTypeUtils.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <optional>
#include <tuple>

// Implementation is split into codecheck-sized fragments while retaining
// one translation unit and the original declaration order.
#include "Parts/PTOImports.cpp"
#include "Parts/PTOOpsPart01.cpp"
#include "Parts/PTOOpsPart02.cpp"
#include "Parts/PTOOpsPart03.cpp"
#include "Parts/PTOOpsPart04.cpp"
#include "Parts/PTOOpsPart05.cpp"
#include "Parts/PTOOpsPart06.cpp"
#include "Parts/PTOOpsPart07.cpp"
#include "Parts/PTOOpsPart08.cpp"
#include "Parts/PTOOpsPart09.cpp"
#include "Parts/PTOOpsPart10.cpp"
#include "Parts/PTOOpsPart11.cpp"
#include "Parts/PTOOpsPart12.cpp"
#include "Parts/PTOOpsPart13.cpp"
#include "Parts/PTOOpsPart14.cpp"
//===----------------------------------------------------------------------===//
// InferIntRangeInterface: PTO runtime query ops (i64)
//===----------------------------------------------------------------------===//

// get_block_idx returns the linear index of the current block within the task,
// documented as [0, BlockNum - 1]. BlockNum is a runtime launch parameter with
// no static IR representation, so report the conservative non-negative range
// [0, INT64_SIGNED_MAX] (signed max, not unsigned max: a sign flip in the
// unsigned range would widen the signed part back to the full range and carry
// no non-negative information).
void pto::GetBlockIdxOp::inferResultRanges(
    ::llvm::ArrayRef<::mlir::ConstantIntRanges> operandRanges,
    ::mlir::SetIntRangeFn setResultRange) {
  setResultRange(
      getResult(),
      ConstantIntRanges::fromUnsigned(APInt::getMinValue(64),
                                      APInt::getSignedMaxValue(64)));
}

// get_subblock_idx returns the vector-core ID, documented as [0, 1].
void pto::GetSubBlockIdxOp::inferResultRanges(
    ::llvm::ArrayRef<::mlir::ConstantIntRanges> operandRanges,
    ::mlir::SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 ConstantIntRanges::fromUnsigned(APInt(64, 0),
                                                 APInt(64, 1)));
}

// Block/subblock counts are non-negative. Do NOT claim >= 1: existing
// kernels guard with `cmpi sge block_num, 1` and rely on that comparison
// staying dynamic (a >= 1 range would fold the guard away).
void pto::GetBlockNumOp::inferResultRanges(
    ::llvm::ArrayRef<::mlir::ConstantIntRanges> operandRanges,
    ::mlir::SetIntRangeFn setResultRange) {
  setResultRange(
      getResult(),
      ConstantIntRanges::fromUnsigned(APInt::getMinValue(64),
                                      APInt::getSignedMaxValue(64)));
}

void pto::GetSubBlockNumOp::inferResultRanges(
    ::llvm::ArrayRef<::mlir::ConstantIntRanges> operandRanges,
    ::mlir::SetIntRangeFn setResultRange) {
  setResultRange(
      getResult(),
      ConstantIntRanges::fromUnsigned(APInt::getMinValue(64),
                                      APInt::getSignedMaxValue(64)));
}

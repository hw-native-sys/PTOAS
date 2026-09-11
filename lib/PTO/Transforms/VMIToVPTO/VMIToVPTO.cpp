// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMIToVPTO.cpp - Convert VMI to physical VPTO IR -------------------===//
//===----------------------------------------------------------------------===//

#include "PTO/Analysis/PTOAddressAnalysis.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/VMIUtils.h"
#include "PTO/IR/VPTOMemoryDist.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/VMILayoutSupport.h"
#include "PTO/Transforms/VPTOLowering.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Func/Transforms/OneToNFuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/OneToNTypeConversion.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <numeric>
#include <type_traits>
#include <tuple>
#include <variant>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VMITOVPTO
#include "PTO/Transforms/Passes.h.inc"

namespace {

#include "VMIToVPTOConversionInternals.cpp"
#include "VMIToVPTOMemoryInternals.cpp"
#include "VMIToVPTOMaskInternals.cpp"
#include "VMIToVPTODataLayoutInternals.cpp"
#include "VMIToVPTOPatternInternals0.cpp"
#include "VMIToVPTOPatternInternals1.cpp"
#include "VMIToVPTOPatternInternals2.cpp"
#include "VMIToVPTOPatternInternals3.cpp"
#include "VMIToVPTOPatternInternals4.cpp"
#include "VMIToVPTOPatternInternals5.cpp"
#include "VMIToVPTOPatternInternals6.cpp"
#include "VMIToVPTOPatternInternals7.cpp"
#include "VMIToVPTOPatternInternals8.cpp"
#include "VMIToVPTOPatternInternals9.cpp"
} // namespace

} // namespace pto
} // namespace mlir

std::unique_ptr<mlir::Pass> mlir::pto::createVMIToVPTOPass() {
  return std::make_unique<mlir::pto::VMIToVPTOPass>();
}

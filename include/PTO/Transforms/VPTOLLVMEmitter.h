// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_VPTOLLVMEMITTER_H
#define MLIR_DIALECT_PTO_TRANSFORMS_VPTOLLVMEMITTER_H

#include "PTO/Support/CANNVersion.h"

#include <memory>
#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::pto {

struct VPTOEmissionOptions {
  bool dumpVPTOIR = false;
  std::string targetTriple;
  std::string march;
  std::string aicoreArch;
  std::string defaultTargetCPU;
  std::string defaultTargetFeatures;
  CANNVersion cannVersion = kDefaultCANNVersion;
};

struct EmittedLLVMModule {
  void reset() {
    module.reset();
    context.reset();
  }

  std::unique_ptr<llvm::LLVMContext> context;
  std::unique_ptr<llvm::Module> module;
};

LogicalResult lowerVPTOModuleToLLVMIRText(
    ModuleOp module, const VPTOEmissionOptions &options, std::string &output,
    llvm::raw_ostream &diagOS);

LogicalResult lowerVPTOModuleToLLVMModules(
    ModuleOp module, const VPTOEmissionOptions &options,
    EmittedLLVMModule &cubeModule, EmittedLLVMModule &vectorModule,
    llvm::raw_ostream &diagOS);

LogicalResult lowerVPTOModuleToLLVMModulesBeta1(
    ModuleOp module, const VPTOEmissionOptions &options,
    EmittedLLVMModule &cubeModule, EmittedLLVMModule &vectorModule,
    llvm::raw_ostream &diagOS);

LogicalResult lowerVPTOModuleToLLVMModulesCANN900(
    ModuleOp module, const VPTOEmissionOptions &options,
    EmittedLLVMModule &cubeModule, EmittedLLVMModule &vectorModule,
    llvm::raw_ostream &diagOS);

} // namespace mlir::pto

#endif // MLIR_DIALECT_PTO_TRANSFORMS_VPTOLLVMEMITTER_H

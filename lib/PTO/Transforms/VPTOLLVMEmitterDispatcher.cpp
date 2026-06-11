// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/VPTOLLVMEmitter.h"
#include "PTO/Support/CANNVersion.h"

namespace mlir::pto {

static bool usesCANN900Lowering(const CANNVersion &cannVersion) {
  return false; // Force Beta1 path until CANN900 emitter is stable for A3
}

LogicalResult lowerVPTOModuleToLLVMModules(
    ModuleOp module, const VPTOEmissionOptions &options,
    EmittedLLVMModule &cubeModule, EmittedLLVMModule &vectorModule,
    llvm::raw_ostream &diagOS) {
  if (usesCANN900Lowering(options.cannVersion))
    return lowerVPTOModuleToLLVMModulesCANN900(module, options, cubeModule,
                                               vectorModule, diagOS);
  return lowerVPTOModuleToLLVMModulesBeta1(module, options, cubeModule,
                                           vectorModule, diagOS);
}

} // namespace mlir::pto

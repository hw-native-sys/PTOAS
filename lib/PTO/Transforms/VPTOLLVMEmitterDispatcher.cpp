// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Support/CodeConstants.h"
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/VPTOLLVMEmitter.h"
#include "PTO/Support/CANNVersion.h"

namespace mlir::pto {

static bool usesCANN900Lowering(const VPTOEmissionOptions &options) {
  const bool isC220 = options.march == "dav-c220-vec" ||
                      options.march == "dav-c220-cube";
  return !isC220 &&
         options.cannVersion >= CANNVersion::release(mlir::pto::kValue9, 0, 0);
}

static bool containsOp(ModuleOp module, bool wantLdDev, bool wantStDev) {
  bool found = false;
  module.walk([&](Operation *op) {
    if ((wantLdDev && isa<PTOLdDevOp>(op)) ||
        (wantStDev && isa<PTOStDevOp>(op))) {
      found = true;
    }
  });
  return found;
}

// ld_dev is required on A2/A3 for SDMA descriptor reads. st_dev is A5-only:
// A2/A3 cannot use it to ring the SDMA doorbell, and stores through that path
// are not reliable on that generation. Other non-c220 targets still need the
// 9.0.0 official lowering; the 9.0.0-beta.1 intrinsic set has not been checked.
static LogicalResult verifyLdStDevTarget(ModuleOp module,
                                         const VPTOEmissionOptions &options,
                                         llvm::raw_ostream &diagOS) {
  const bool isC220 = options.march == "dav-c220-vec" ||
                      options.march == "dav-c220-cube";
  if (isC220 && containsOp(module, false, true)) {
    diagOS << "VPTO LLVM emission failed: pto.st_dev is not supported on A2/A3\n";
    return failure();
  }
  if (!containsOp(module, true, true) || isC220 ||
      usesCANN900Lowering(options)) {
    return success();
  }

  diagOS << "VPTO LLVM emission failed: pto.ld_dev and pto.st_dev require "
            "CANN 9.0.0 or newer official lowering\n";
  return failure();
}

LogicalResult lowerVPTOModuleToLLVMModules(
    ModuleOp module, const VPTOEmissionOptions &options,
    EmittedLLVMModule &cubeModule, EmittedLLVMModule &vectorModule,
    llvm::raw_ostream &diagOS) {
  if (failed(verifyLdStDevTarget(module, options, diagOS))) {
    return failure();
  }
  if (usesCANN900Lowering(options)) {
    return lowerVPTOModuleToLLVMModulesCANN900(module, options, cubeModule,
                                               vectorModule, diagOS);
  }
  return lowerVPTOModuleToLLVMModulesBeta1(module, options, cubeModule,
                                           vectorModule, diagOS);
}

LogicalResult lowerVPTOModuleToLLVMIRText(
    ModuleOp module, const VPTOEmissionOptions &options, std::string &output,
    llvm::raw_ostream &diagOS) {
  output.clear();

  EmittedLLVMModule cubeModule;
  EmittedLLVMModule vectorModule;
  if (failed(
          lowerVPTOModuleToLLVMModules(module, options, cubeModule, vectorModule,
                                       diagOS))) {
    return failure();
  }

  llvm::raw_string_ostream os(output);
  bool printedAny = false;
  if (vectorModule.module) {
    vectorModule.module->print(os, nullptr);
    os << "\n";
    printedAny = true;
  }
  if (cubeModule.module) {
    if (printedAny) {
      os << "\n";
    }
    cubeModule.module->print(os, nullptr);
    os << "\n";
  }
  os.flush();
  // The LLVM dialect prints arg-memory effects with memory(...), while the
  // Bisheng LLVM 15 parser accepts only the equivalent legacy argmemonly
  // spelling.
  constexpr StringLiteral modernArgMemOnly = "memory(argmem: readwrite)";
  size_t offset = 0;
  while ((offset = output.find(modernArgMemOnly.str(), offset)) !=
         std::string::npos) {
    output.replace(offset, modernArgMemOnly.size(), "argmemonly");
    offset += StringRef("argmemonly").size();
  }
  return success();
}

} // namespace mlir::pto

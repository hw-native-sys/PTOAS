// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <string>
#include "BufidSyncAnalysis.h"
#include "BufidSyncCodegen.h"
#include "BufidSyncIdAlloc.h"
#include "PTO/IR/PTO.h"
#include "PTO/Support/CodeConstants.h"
#include "PTO/Transforms/InsertSync/MemoryDependentAnalyzer.h"
#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"
#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"


#define DEBUG_TYPE "pto-bufid-sync"

namespace mlir {
namespace pto {

#define GEN_PASS_DECL_PTOBUFIDSYNC
#define GEN_PASS_DEF_PTOBUFIDSYNC
#include "PTO/Transforms/Passes.h.inc"

namespace {
struct PTOBufidSyncPass
    : public impl::PTOBufidSyncBase<PTOBufidSyncPass> {
  PTOBufidSyncPass() = default;
  explicit PTOBufidSyncPass(const PTOBufidSyncOptions &options) {
    enableBufidSyncDebug = options.enableBufidSyncDebug;
  }
  void runOnOperation() override;
};
} // namespace

void PTOBufidSyncPass::runOnOperation() {
  func::FuncOp func = getOperation();

  bool hasExistingBufSync = false;
  func.walk([&](pto::GetBufOp) { hasExistingBufSync = true; });
  func.walk([&](pto::RlsBufOp) { hasExistingBufSync = true; });
  if (hasExistingBufSync) {
    LLVM_DEBUG(llvm::dbgs() << "bufid_sync: existing get_buf ops found, "
                               "skipping pass.\n");
    return;
  }
  if (enableBufidSyncDebug) {
    llvm::outs() << "[bufid_sync] STEP 0: Build SyncIR...\n";
  }
  SyncIRs syncIR;
  Buffer2MemInfoMap buffer2MemInfoMap;
  MemoryDependentAnalyzer memAnalyzer;

  PTOIRTranslator translator(syncIR, memAnalyzer, buffer2MemInfoMap, func,
                             SyncAnalysisMode::NORMALSYNC);
  translator.Build();
  if (enableBufidSyncDebug) {
    llvm::outs() << "[bufid_sync] STEP 0 done: syncIR size=" << syncIR.size() << "\n";
  }

  if (syncIR.empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "bufid_sync: SyncIR is empty, nothing to do.\n");
    return;
  }

  BufidSyncAnalysis analysis(syncIR, memAnalyzer, func, enableBufidSyncDebug);

  analysis.collectDependencies();
  analysis.classifyTiles();
  analysis.allocateVirtualBufIds();
  analysis.insertSyncOperations();
  analysis.optimizeSamePipeMerge();

  if (analysis.getOp2BufSync().empty()) {
    if (enableBufidSyncDebug) {
      llvm::outs() << "[bufid_sync] No sync operations to insert, done.\n";
    }
    return;
  }

  BufidSyncIdAlloc idAlloc(analysis.getVirtualBufIds(),
                           analysis.getOp2BufSync(), syncIR, mlir::pto::kValue32,
                           enableBufidSyncDebug);

  idAlloc.computeLifeIntervals();
  idAlloc.linearScanAllocate();
  idAlloc.compactPhysicalIds();

  if (idAlloc.needsReuse()) {
    idAlloc.reuseIds();
    idAlloc.compactPhysicalIds();
  }
  if (idAlloc.needsReuse()) {
    func.emitError("bufid_sync requires more than 32 physical buf ids after "
                   "reuse");
    signalPassFailure();
    return;
  }

  analysis.setLogicToPhysicalId(idAlloc.getLogicToPhysical());

  analysis.mergeGetRls();

  std::string validationError;
  if (!idAlloc.validateNoSamePhysicalIdNesting(&validationError)) {
    func.emitError("bufid_sync produced invalid physical bufid nesting: ")
        << validationError;
    signalPassFailure();
    return;
  }

  BufidSyncCodegen codegen(func, analysis.getOp2BufSync(), idAlloc);
  if (failed(codegen.run())) {
    signalPassFailure();
    return;
  }
}

std::unique_ptr<Pass>
createPTOBufidSyncPass(const PTOBufidSyncOptions &options) {
  return std::make_unique<PTOBufidSyncPass>(options);
}

} // namespace pto
} // namespace mlir

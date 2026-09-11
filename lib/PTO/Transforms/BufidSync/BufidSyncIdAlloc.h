// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_BUFIDSYNC_BUFIDSYNCIDALLOC_H
#define MLIR_DIALECT_PTO_TRANSFORMS_BUFIDSYNC_BUFIDSYNCIDALLOC_H

#include <map>
#include <string>
#include "BufidSyncAnalysis.h"


namespace mlir {
namespace pto {

class BufidSyncIdAlloc {
public:
  BufidSyncIdAlloc(SmallVector<VirtualBufId> &virtualBufIds,
                   DenseMap<Operation *, BufSyncPipeBuild> &op2BufSync,
                   const SyncIRs &syncIR, unsigned physicalBufIdCount,
                   bool debugEnabled = false)
      : virtualBufIds_(virtualBufIds), op2BufSync_(op2BufSync),
        syncIR_(syncIR), physicalBufIdCount_(physicalBufIdCount),
        debugEnabled_(debugEnabled) {}

  void computeLifeIntervals();
  void linearScanAllocate();
  bool needsReuse() const { return maxPhysicalIdUsed_ >= static_cast<int>(physicalBufIdCount_); }
  void reuseIds();
  void compactPhysicalIds();
  bool validateNoSamePhysicalIdNesting(std::string *error = nullptr) const;

  const DenseMap<int, int> &getLogicToPhysical() const { return logicToPhysical_; }

private:
  void collectPipeSignature(int logicId, SmallVector<PipelineType> &pipes) const;
  unsigned getOutermostLoopBegin(Operation *op) const;
  unsigned getOutermostLoopEnd(Operation *op) const;

  // reuseIds() helpers. A single reuse iteration (reuseIdsStep) selects the
  // best signature group, merges half of its logic ids onto donors, and
  // recompacts; it returns false when no further reuse is possible.
  bool reuseIdsStep(int iteration);
  void collectLogicIdPipes(
      DenseMap<int, SmallVector<PipelineType>> &logicIdPipes,
      DenseMap<int, unsigned> &logicIdFirstPos) const;
  std::string
  selectBestSigGroup(const std::map<std::string, SmallVector<int>> &sigGroups) const;
  bool isConsecutiveOnPipe(const SmallVector<int> &ids, PipelineType pipe) const;
  void applyMerges(const DenseMap<int, int> &mergeMap);

  SmallVector<VirtualBufId> &virtualBufIds_;
  DenseMap<Operation *, BufSyncPipeBuild> &op2BufSync_;
  const SyncIRs &syncIR_;
  unsigned physicalBufIdCount_;
  bool debugEnabled_;

  SmallVector<BufIdInterval> intervals_;
  DenseMap<int, int> logicToPhysical_;
  int maxPhysicalIdUsed_ = -1;
};

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_BUFIDSYNC_BUFIDSYNCIDALLOC_H

// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_BUFIDSYNC_BUFIDSYNCANALYSIS_H
#define MLIR_DIALECT_PTO_TRANSFORMS_BUFIDSYNC_BUFIDSYNCANALYSIS_H

#include <algorithm>
#include <optional>
#include "PTO/Transforms/InsertSync/MemoryDependentAnalyzer.h"
#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"
#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"


namespace mlir {
namespace pto {

struct TileInfo {
  const BaseMemInfo *memInfo;
  pto::AddressSpace scope;
  uint64_t baseAddr;
  uint64_t size;
  Value tileValue;
  Value rootBuffer;
};

struct DepPair {
  CompoundInstanceElement *srcElement;
  CompoundInstanceElement *dstElement;
  SmallVector<const BaseMemInfo *> depTiles;
  PipelineType srcPipe;
  PipelineType dstPipe;
};

struct VirtualBufId {
  int logicId;
  SmallVector<TileInfo> tiles;
  pto::AddressSpace scope;
};

enum class BufSyncType {
  GET_BUF,
  RLS_BUF
};

struct BufSyncOperation {
  BufSyncType type;
  PipelineType pipe;
  int logicId;
  unsigned syncIRIndex;
  unsigned depSyncIRIndex;
};

struct BufSyncPipeBuild {
  SmallVector<BufSyncOperation> pipeBefore;
  SmallVector<BufSyncOperation> pipeAfter;
};

struct BufIdInterval {
  int logicId;
  unsigned startPos;
  unsigned endPos;
  SmallVector<PipelineType> pipes;
};

inline void printTileValue(llvm::raw_ostream &os, Value val) {
  val.printAsOperand(os, OpPrintingFlags());
}

inline llvm::StringRef stringifyAddressSpaceShort(pto::AddressSpace as) {
  switch (as) {
  case pto::AddressSpace::GM:      return "gm";
  case pto::AddressSpace::MAT:    return "mat";
  case pto::AddressSpace::LEFT:   return "left";
  case pto::AddressSpace::RIGHT:  return "right";
  case pto::AddressSpace::ACC:    return "acc";
  case pto::AddressSpace::VEC:    return "vec";
  case pto::AddressSpace::BIAS:   return "bias";
  case pto::AddressSpace::SCALING:return "scaling";
  default: return "unknown";
  }
}

inline std::optional<pto::AddressSpace> getTileLocFromType(Value tileValue) {
  if (!tileValue) {
    return std::nullopt;
  }
  auto tileType = dyn_cast<pto::TileBufType>(tileValue.getType());
  if (tileType) {
    auto asAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(tileType.getMemorySpace());
    if (asAttr) {
      return asAttr.getAddressSpace();
    }
  }
  auto memrefType = dyn_cast<mlir::MemRefType>(tileValue.getType());
  if (memrefType) {
    auto asAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(memrefType.getMemorySpace());
    if (asAttr) {
      return asAttr.getAddressSpace();
    }
  }
  return std::nullopt;
}

inline void printTileLoc(llvm::raw_ostream &os, Value tileValue) {
  if (!tileValue) {
    return;
  }
  auto *defOp = tileValue.getDefiningOp();
  if (!defOp) {
    os << " BlockArg";
    return;
  }
  os << " defBy=" << defOp->getName().getStringRef();
  auto locAs = getTileLocFromType(tileValue);
  if (locAs) {
    os << " typeLoc=" << stringifyAddressSpaceShort(*locAs);
  }
}

inline void printTileInfo(llvm::raw_ostream &os, const TileInfo &t) {
  printTileValue(os, t.tileValue);
  os << " scope=" << static_cast<int>(t.scope)
     << "(" << stringifyAddressSpaceShort(t.scope) << ")";
  auto typeLoc = getTileLocFromType(t.tileValue);
  if (typeLoc && *typeLoc != t.scope) {
    os << " MISMATCH! typeLoc=" << stringifyAddressSpaceShort(*typeLoc);
  }
  os << " baseAddr=" << t.baseAddr << " size=" << t.size;
  printTileLoc(os, t.tileValue);
  os << " root=";
  printTileValue(os, t.rootBuffer);
  if (t.rootBuffer) {
    if (auto *defOp = t.rootBuffer.getDefiningOp()) {
      os << " rootDefBy=" << defOp->getName().getStringRef();
    } else {
      os << " rootDefBy=BlockArg";
    }
    os << " rootPtr=" << static_cast<const void *>(t.rootBuffer.getAsOpaquePointer());
  }
}

inline void printDepPairs(llvm::raw_ostream &os,
                          const SmallVector<DepPair> &depPairs) {
  os << "[bufid_sync] depPairs count: " << depPairs.size() << "\n";
  for (unsigned i = 0; i < depPairs.size(); ++i) {
    auto &dp = depPairs[i];
    os << "  depPair[" << i << "] src[syncIR=" << dp.srcElement->GetIndex()
       << "] dst[syncIR=" << dp.dstElement->GetIndex()
       << "] srcPipe=" << static_cast<int>(dp.srcPipe)
       << " dstPipe=" << static_cast<int>(dp.dstPipe) << "\n";
    for (unsigned j = 0; j < dp.depTiles.size(); ++j) {
      os << "    depTile[" << j << "] ";
      printTileValue(os, dp.depTiles[j]->baseBuffer);
      os << " scope=" << static_cast<int>(dp.depTiles[j]->scope)
         << " rootBuffer=";
      printTileValue(os, dp.depTiles[j]->rootBuffer);
      printTileLoc(os, dp.depTiles[j]->baseBuffer);
      os << "\n";
    }
  }
}

inline void printAllTiles(llvm::raw_ostream &os,
                          const SmallVector<TileInfo> &allTiles) {
  os << "[bufid_sync] allTiles count: " << allTiles.size() << "\n";
  for (unsigned i = 0; i < allTiles.size(); ++i) {
    os << "  tile[" << i << "] ";
    printTileInfo(os, allTiles[i]);
    os << "\n";
  }
}

inline void printTileGroups(llvm::raw_ostream &os,
                            const SmallVector<SmallVector<unsigned>> &tileGroups,
                            const SmallVector<TileInfo> &allTiles) {
  os << "[bufid_sync] tileGroups count: " << tileGroups.size() << "\n";
  for (unsigned i = 0; i < tileGroups.size(); ++i) {
    os << "  group[" << i << "] size=" << tileGroups[i].size() << " tiles=[";
    for (unsigned j = 0; j < tileGroups[i].size(); ++j) {
      if (j > 0) {
        os << " ; ";
      }
      printTileValue(os, allTiles[tileGroups[i][j]].tileValue);
    }
    os << "]\n";
  }
}

inline void printVirtualBufIds(llvm::raw_ostream &os,
                               const SmallVector<VirtualBufId> &virtualBufIds) {
  os << "[bufid_sync] VirtualBufId count: " << virtualBufIds.size() << "\n";
  for (auto &vbid : virtualBufIds) {
    os << "  logicId=" << vbid.logicId
       << " scope=" << static_cast<int>(vbid.scope)
       << " tileCount=" << vbid.tiles.size() << " tiles=[";
    for (unsigned i = 0; i < vbid.tiles.size(); ++i) {
      if (i > 0) {
        os << " ; ";
      }
      printTileValue(os, vbid.tiles[i].tileValue);
    }
    os << "]\n";
  }
}

inline void printOp2BufSyncEntry(llvm::raw_ostream &os, Operation *op,
                                 const BufSyncPipeBuild &build,
                                 unsigned printIdx) {
  auto firstSyncIdx = build.pipeBefore.empty()
                          ? (build.pipeAfter.empty() ? 0
                                                     : build.pipeAfter[0].syncIRIndex)
                          : build.pipeBefore[0].syncIRIndex;
  os << "  [" << printIdx << "][syncIR=" << firstSyncIdx << "] op: ";
  op->getName().print(os);
  if (op->getNumResults() > 0) {
    os << " ";
    op->getResult(0).printAsOperand(os, OpPrintingFlags());
  }
  os << " <- ";
  for (unsigned i = 0; i < op->getNumOperands(); ++i) {
    if (i > 0) {
      os << ", ";
    }
    op->getOperand(i).printAsOperand(os, OpPrintingFlags());
  }
  os << "\n    pipeBefore:";
  for (auto &s : build.pipeBefore) {
    os << " [GET_BUF pipe=" << static_cast<int>(s.pipe)
       << " logicId=" << s.logicId
       << " syncIR=" << s.syncIRIndex << "]";
  }
  os << "\n    pipeAfter:";
  for (auto &s : build.pipeAfter) {
    os << " [RLS_BUF pipe=" << static_cast<int>(s.pipe)
       << " logicId=" << s.logicId
       << " syncIR=" << s.syncIRIndex << "]";
  }
  os << "\n";
}

inline void printOp2BufSync(llvm::raw_ostream &os,
                            const DenseMap<Operation *, BufSyncPipeBuild> &op2BufSync,
                            func::FuncOp func, const char *title = nullptr) {
  if (title) {
    os << "[bufid_sync] " << title << ":\n";
  }

  SmallVector<Operation *> sortedOps;
  sortedOps.reserve(op2BufSync.size());
  for (auto &[op, build] : op2BufSync) {
    sortedOps.push_back(op);
  }

  DenseMap<const Operation *, unsigned> opOrder;
  unsigned orderIdx = 0;
  func.walk([&](Operation *op) { opOrder[op] = orderIdx++; });

  std::sort(sortedOps.begin(), sortedOps.end(),
            [&](const Operation *a, const Operation *b) {
              return opOrder[a] < opOrder[b];
            });

  os << "[bufid_sync] op2BufSync count: " << op2BufSync.size() << "\n";
  unsigned printIdx = 0;
  for (auto *op : sortedOps) {
    auto &build = op2BufSync.find(op)->second;
    printOp2BufSyncEntry(os, op, build, printIdx);
    ++printIdx;
  }
}

inline void printLifeIntervals(llvm::raw_ostream &os,
                               const SmallVector<BufIdInterval> &intervals) {
  os << "[bufid_sync] LifeIntervals count: " << intervals.size() << "\n";
  for (auto &iv : intervals) {
    os << "  logicId=" << iv.logicId
       << " [" << iv.startPos << "," << iv.endPos << "] pipes=";
    for (auto p : iv.pipes) {
      os << static_cast<int>(p) << ",";
    }
    os << "\n";
  }
}

inline void printLogicToPhysical(llvm::raw_ostream &os,
                                 const DenseMap<int, int> &logicToPhysical,
                                 const char *label = "") {
  os << "[bufid_sync] " << label << ":\n";
  for (auto &[lid, pid] : logicToPhysical) {
    os << "  logicId=" << lid << " -> physicalId=" << pid << "\n";
  }
}

// optimizeSamePipeMerge()/reuseIds() helper: copy `in` into `out`, remapping
// each logicId through mergeMap and dropping duplicates by (pipe, remapped
// logicId). Shared by BufidSyncAnalysis and BufidSyncIdAlloc.
void remapSyncList(const SmallVector<BufSyncOperation> &in,
                   const DenseMap<int, int> &mergeMap,
                   SmallVector<BufSyncOperation> &out);

class BufidSyncAnalysis {
public:
  BufidSyncAnalysis(SyncIRs &syncIR, MemoryDependentAnalyzer &memAnalyzer,
                    func::FuncOp func, bool debugEnabled = false)
      : syncIR_(syncIR), memAnalyzer_(memAnalyzer), func_(func),
        debugEnabled_(debugEnabled) {}

  void collectDependencies();
  void classifyTiles();
  void allocateVirtualBufIds();
  void insertSyncOperations();
  void optimizeSamePipeMerge();
  void mergeGetRls();

  const SmallVector<DepPair> &getDepPairs() const { return depPairs_; }
  const SmallVector<VirtualBufId> &getVirtualBufIds() const { return virtualBufIds_; }
  SmallVector<VirtualBufId> &getVirtualBufIds() { return virtualBufIds_; }
  const DenseMap<Operation *, BufSyncPipeBuild> &getOp2BufSync() const { return op2BufSync_; }
  DenseMap<Operation *, BufSyncPipeBuild> &getOp2BufSync() { return op2BufSync_; }
  const DenseMap<int, int> &getLogicToPhysicalId() const { return logicToPhysicalId_; }
  void setLogicToPhysicalId(const DenseMap<int, int> &mapping) { logicToPhysicalId_ = mapping; }

private:
  bool isGMDependency(const SmallVector<const BaseMemInfo *> &tiles) const;
  bool isSamePipe(const CompoundInstanceElement *a,
                  const CompoundInstanceElement *b) const;
  void collectTilesFromDepPairs();
  // collectDependencies() helper: run the three-way DepBetween probe between two
  // compounds, returning true (with `out` populated) on the first hit.
  bool detectDependency(CompoundInstanceElement *src,
                        CompoundInstanceElement *dst,
                        DepBaseMemInfoPairVec &out) const;
  // collectDependencies() helper: probe one ordered compound pair and build the
  // resulting DepPair, or std::nullopt when there is no local dependency.
  std::optional<DepPair> tryBuildDepPair(CompoundInstanceElement *src,
                                         CompoundInstanceElement *dst);
  // collectTilesFromDepPairs() helpers: gather raw tiles from depPairs_, then
  // drop duplicates that share (scope, baseAddr, size) and alias/root buffer.
  void collectRawTiles();
  DenseSet<unsigned> computeRemovedTileIndices() const;
  void dedupTiles();
  // optimizeSamePipeMerge() helpers.
  DenseMap<int, DenseSet<int>> computeLogicIdToPipeInts() const;
  DenseMap<int, int>
  computeSamePipeMergeMap(const DenseMap<int, DenseSet<int>> &logicIdToPipeInts) const;
  void accumulateMergesForOp(
      const DenseMap<int, SmallVector<int>> &pipeIntToLogicIds,
      const DenseMap<int, DenseSet<int>> &logicIdToPipeInts,
      DenseMap<int, int> &mergeMap) const;
  // accumulateMergesForOp() helpers: group a pipe's logic ids by their single
  // cross-pipe signature, then merge same-signature groups into one survivor.
  DenseMap<int, SmallVector<int>> computeCrossPipeSignatureGroups(
      int pipeInt, const SmallVector<int> &logicIds,
      const DenseMap<int, DenseSet<int>> &logicIdToPipeInts) const;
  void mergeSignatureGroups(const DenseMap<int, SmallVector<int>> &sigGroups,
                            DenseMap<int, int> &mergeMap) const;
  void rebuildOp2BufSyncWithMerge(const DenseMap<int, int> &mergeMap);
  int findBestVirtualBufId(const BaseMemInfo *tile) const;
  int findBestVirtualBufId(const DepPair &depPair) const;
  bool virtualBufIdContainsTile(const VirtualBufId &vbid,
                                const BaseMemInfo *tile) const;

  void bronKerbosch(const SmallVector<unsigned> &R, SmallVector<unsigned> P,
                    SmallVector<unsigned> X,
                    const SmallVector<SmallVector<bool>> &adjMatrix,
                    SmallVector<SmallVector<unsigned>> &maximalCliques);

  SyncIRs &syncIR_;
  MemoryDependentAnalyzer &memAnalyzer_;
  func::FuncOp func_;
  bool debugEnabled_;

  SmallVector<DepPair> depPairs_;
  SmallVector<TileInfo> allTiles_;
  SmallVector<SmallVector<unsigned>> tileGroups_;
  SmallVector<VirtualBufId> virtualBufIds_;
  DenseMap<Operation *, BufSyncPipeBuild> op2BufSync_;
  DenseMap<int, int> logicToPhysicalId_;
};

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_BUFIDSYNC_BUFIDSYNCANALYSIS_H

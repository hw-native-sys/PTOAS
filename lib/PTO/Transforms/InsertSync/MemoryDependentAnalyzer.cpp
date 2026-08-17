// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/InsertSync/MemoryDependentAnalyzer.h"
#include "PTO/Transforms/InsertSync/InsertSyncDebug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/Debug.h"
 
#define DEBUG_TYPE "pto-inject-sync"
 
using namespace mlir;
using namespace mlir::pto;

static bool isTraceEnabled() {
  return isInsertSyncDebugEnabled(InsertSyncDebugLevel::Trace);
}
 
// [Debug] 打印 Value 详细信息
static void printValueDebug(const char* tag, Value v) {
  if (!isTraceEnabled()) {
    return;
  }
  llvm::errs() << tag << ": ";
  if (!v) {
    llvm::errs() << "NULL\n";
    return;
  }
  
  if (auto *op = v.getDefiningOp()) {
    llvm::errs() << "OpResult defined by [" << op->getName() << "]";
  } else {
    llvm::errs() << "BlockArgument";
  }
  llvm::errs() << " | Type: " << v.getType() << "\n";
}
 
// [Fix & Debug] 增强版 GetRealRoot
static Value GetRealRoot(Value v) {
  const bool trace = isTraceEnabled();
  if (trace) {
    llvm::errs() << "  [Trace] GetRealRoot Start:\n";
    printValueDebug("    Current", v);
  }
  
  int depth = 0;
  const int maxDepth = 20;
 
  while (v && depth++ < maxDepth) {
    Operation *defOp = v.getDefiningOp();
    if (!defOp) {
      if (trace) {
        llvm::errs() << "    -> Reached BlockArgument. Stop.\n";
      }
        break; 
    }
 
    if (trace) {
      llvm::errs() << "    -> Hit Alloc/Other [" << defOp->getName()
                   << "]. Stop.\n";
    }
    break;
  }
  return v;
}

// Extract an absolute base address when a root is represented by an integer
// constant. Tile roots otherwise carry their planned addresses in
// `BaseMemInfo::baseAddresses`.
static uint64_t getRootBaseAddress(Value rootBuffer) {
  if (!rootBuffer) {
    return 0;
  }
  if (auto cst = rootBuffer.getDefiningOp<arith::ConstantOp>()) {
    if (auto attr = dyn_cast<IntegerAttr>(cst.getValue())) {
      return attr.getValue().getZExtValue();
    }
  }
  return 0;
}

static bool isIntegerConstant(Value value) {
  if (!value) {
    return false;
  }
  auto cst = value.getDefiningOp<arith::ConstantOp>();
  return cst && isa<IntegerAttr>(cst.getValue());
}

static bool hasKnownLocalAbsoluteAddress(const BaseMemInfo *info) {
  if (!info || info->baseAddresses.empty() || info->allocateSize == 0) {
    return false;
  }

  Value root = info->rootBuffer;
  if (!root) {
    return false;
  }

  if (auto alloc = root.getDefiningOp<pto::AllocTileOp>()) {
    return isIntegerConstant(alloc.getAddr());
  }

  if (isIntegerConstant(root)) {
    return true;
  }

  return false;
}

// Cross-root byte-range overlap check for local memory (UB/L1).
// `MemAlias` historically only checked byte-range overlap when the two
// `rootBuffer` SSA values were identical (same `alloc_tile`).
// When PlanMemory reuses the same physical UB region for two *different*
// allocations, their `rootBuffer` values differ even though their byte ranges
// overlap — so `MemAlias` returned false and InsertSync silently dropped the
// cross-pipe hazard (issue #934: MTE3 tstore racing a V-pipe write into an
// overlapping-but-different-root buffer).
// This helper re-derives the absolute address as `getRootBaseAddress(root) +
// baseAddresses[i]` and performs the same `maxStart < minEnd` overlap test
// across every address pair, regardless of root identity.
static bool isLocalBufferOverlapCrossRoot(const BaseMemInfo *a,
                                           const BaseMemInfo *b) {
  // Cross-root checks are only meaningful after physical addresses have been
  // materialized. For unknown-address roots keep the historical behavior:
  // different roots are treated as independent.
  if (!hasKnownLocalAbsoluteAddress(a) || !hasKnownLocalAbsoluteAddress(b)) {
    return false;
  }

  uint64_t rootBaseA = getRootBaseAddress(a->rootBuffer);
  uint64_t rootBaseB = getRootBaseAddress(b->rootBuffer);

  for (uint64_t addrA : a->baseAddresses) {
    for (uint64_t addrB : b->baseAddresses) {
      uint64_t aStart = rootBaseA + addrA;
      uint64_t bStart = rootBaseB + addrB;
      uint64_t aEnd = aStart + a->allocateSize;
      uint64_t bEnd = bStart + b->allocateSize;
      uint64_t maxStart = std::max(aStart, bStart);
      uint64_t minEnd = std::min(aEnd, bEnd);
      if (maxStart < minEnd) {
        return true;
      }
    }
  }
  return false;
}

bool MemoryDependentAnalyzer::DepBetween(
    const SmallVector<const BaseMemInfo *> &a,
    const SmallVector<const BaseMemInfo *> &b,
    DepBaseMemInfoPairVec &depBaseMemInfosVec) {
  // [Debug Log] 关键入口信息
  if (isTraceEnabled()) {
    llvm::errs() << "\n[DepBetween] Checking dependency...\n";
    llvm::errs() << "  Vec A Size: " << a.size() << "\n";
    llvm::errs() << "  Vec B Size: " << b.size() << "\n";
  }
 
  bool hasAlias = false;
  for (auto &i : a) {
    for (auto &j : b) {
      if (MemAlias(i, j)) {
        depBaseMemInfosVec.push_back(std::make_pair(i, j));
        hasAlias = true;
      }
    }
  }
  return hasAlias;
}
 
bool MemoryDependentAnalyzer::MemAlias(const BaseMemInfo *a,
                                       const BaseMemInfo *b) {
  pto::AddressSpace as = a->scope;
  pto::AddressSpace bs = b->scope;
 
  // [Debug Log] 打印比较对象
  if (isTraceEnabled()) {
    llvm::errs() << "  [MemAlias Check]\n";
    printValueDebug("    Root A", a->rootBuffer);
    printValueDebug("    Root B", b->rootBuffer);
    llvm::errs() << "    Scope A: " << static_cast<int>(as)
                 << ", Scope B: " << static_cast<int>(bs) << "\n";
  }
 
  if (as != bs) {
    if (isTraceEnabled()) {
      llvm::errs() << "    -> Scope Mismatch. False.\n";
    }
    return false;
  }

  if (a->aliasesUnknownRange || b->aliasesUnknownRange) {
    if (isTraceEnabled()) {
      llvm::errs() << "    -> Unknown range in same scope. True.\n";
    }
    return true;
  }
 
  // 1. GM 内存
  if (as == pto::AddressSpace::GM) {
    return isGMBufferOverlap(a, b);
  }
 
  // 2. Local Memory (UB/L1)
  // PTOPlanMemory writes physical addresses back to allocation roots. Once an
  // async MTE3 store has consumed the source SSA, a later allocation can reuse
  // the same physical range with a different alloc_tile root. Compare those
  // ranges directly when both sides carry known physical local addresses.
  if (a->hasKnownPhysicalAddresses && b->hasKnownPhysicalAddresses) {
    if (isTraceEnabled()) {
      llvm::errs() << "    -> Comparing known physical local ranges.\n";
    }
    if (a->baseAddresses.empty() || b->baseAddresses.empty()) {
      return true;
    }
    if (a->allocateSize == 0 || b->allocateSize == 0) {
      return true;
    }
    return isBufferAddressRangeOverlap(a, b);
  }

  if (a->rootBuffer == b->rootBuffer) {
    if (a->baseAddresses.empty() || b->baseAddresses.empty()) {
      return true;
    }
    if (a->allocateSize == 0 || b->allocateSize == 0) {
      return true;
    }
    return isBufferAddressRangeOverlap(a, b);
  }
 
  // 2.2 深层比较：穿透 View
  Value realRootA = GetRealRoot(a->rootBuffer);
  Value realRootB = GetRealRoot(b->rootBuffer);
 
  if (isTraceEnabled()) {
    llvm::errs() << "    [Deep Check] Surface Roots differ. Digging deeper...\n";
    printValueDebug("      Real Root A", realRootA);
    printValueDebug("      Real Root B", realRootB);
  }
 
  if (realRootA == realRootB && realRootA != nullptr) {
    if (isTraceEnabled()) {
      llvm::errs() << "      -> MATCH! Real roots are the same.\n";
    }
    if (a->baseAddresses.empty() || b->baseAddresses.empty()) {
      return true;
    }
    if (a->allocateSize == 0 || b->allocateSize == 0) {
      return true;
    }
      return isBufferAddressRangeOverlap(a, b);
  } else {
    if (isTraceEnabled()) {
      llvm::errs() << "      -> Mismatch. Real roots differ.\n";
    }
  }

  // 2.3 Cross-root absolute-address overlap check.
  //
  // Roots genuinely differ (different alloc_tile/address SSA). Historically
  // MemAlias returned false here, but PlanMemory can reuse the same physical
  // UB region for distinct allocations whose
  // byte ranges overlap. Re-derive the absolute address and check overlap so
  // a cross-pipe hazard (e.g. MTE3 tstore vs a V-pipe write into an
  // overlapping region) is not silently dropped (issue #934).
  bool crossRootOverlap = isLocalBufferOverlapCrossRoot(a, b);
  if (isTraceEnabled()) {
    llvm::errs() << "      -> Cross-root overlap check: "
                 << (crossRootOverlap ? "true" : "false") << "\n";
  }
  return crossRootOverlap;
}
 
bool MemoryDependentAnalyzer::isGMBufferOverlap(const BaseMemInfo *a,
                                                const BaseMemInfo *b) {
  if (a->baseAddresses.empty() || b->baseAddresses.empty()) {
    return true;
  }
  if (a->rootBuffer != b->rootBuffer) {
    Value realRootA = GetRealRoot(a->rootBuffer);
    Value realRootB = GetRealRoot(b->rootBuffer);
    if (realRootA != realRootB) {
        return false;
    }
    if (a->allocateSize == 0 || b->allocateSize == 0) {
      return true;
    }
    return isBufferAddressRangeOverlap(a, b);
  }

  if (a->allocateSize == 0 || b->allocateSize == 0) {
    return true;
  }

  return isBufferAddressRangeOverlap(a, b);
}
 
bool MemoryDependentAnalyzer::isBufferAddressRangeOverlap(
    const BaseMemInfo *a, const BaseMemInfo *b) {
  int aBaseAddressesSize = static_cast<int>(a->baseAddresses.size());
  int bBaseAddressesSize = static_cast<int>(b->baseAddresses.size());
  
  for (int i = 0; i < aBaseAddressesSize; i++) {
    for (int j = 0; j < bBaseAddressesSize; j++) {
      if (isBufferOverlap(a, b, i, j)) {
        return true;
      }
    }
  }
  return false;
}
 
bool MemoryDependentAnalyzer::isBufferOverlap(const BaseMemInfo *a,
                                              const BaseMemInfo *b, int aIndex,
                                              int bIndex) {
  uint64_t aStart = a->baseAddresses[aIndex];
  uint64_t bStart = b->baseAddresses[bIndex];
  uint64_t aEnd = aStart + a->allocateSize;
  uint64_t bEnd = bStart + b->allocateSize;
 
  uint64_t maxStart = std::max(aStart, bStart);
  uint64_t minEnd = std::min(aEnd, bEnd);
 
  return maxStart < minEnd;
}

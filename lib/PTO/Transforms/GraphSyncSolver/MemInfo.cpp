// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===------------- MemInfo.cpp ---- Graph Sync Solver ---------------------===//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/GraphSyncSolver/MemInfo.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Transforms/GraphSyncSolver/SyncSolverIR.h"
#include "../Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>

using namespace mlir;
using namespace pto::syncsolver;

namespace mlir::pto::syncsolver {

static std::optional<int64_t> getBufferBitSize(Value value) {
  auto shaped = dyn_cast<ShapedType>(value.getType());
  if (!shaped || !shaped.hasStaticShape()) {
    return ShapedType::kDynamic;
  }
  Type elementType = shaped.getElementType();
  auto bitWidth = getPTOStorageElemBitWidth(elementType);
  if (bitWidth == 0) {
    return ShapedType::kDynamic;
  }
  return shaped.getNumElements() * bitWidth;
}

llvm::SmallVector<int64_t> getAddresses(const llvm::SmallVector<Value> &addrs) {
  llvm::SmallVector<int64_t> offsets;
  for (auto addr : addrs) {
    auto constOp =
        llvm::dyn_cast_if_present<arith::ConstantOp>(addr.getDefiningOp());
    if (!constOp) {
      offsets.push_back(ShapedType::kDynamic);
      continue;
    }
    auto baseAddr =
        static_cast<int64_t>(cast<IntegerAttr>(constOp.getValue()).getInt());
    int64_t baseAddrInBits = baseAddr * pto::kBitsToByte;
    offsets.push_back(baseAddrInBits);
  }
  return offsets;
}

PointerLikeInfo getPointerLikeInfo(pto::PointerCastOp pointerCastOp) {
  PointerLikeInfo pointerLikeInfo(pointerCastOp);
  pointerLikeInfo.addresses = getAddresses(pointerCastOp.getAddrs());
  pointerLikeInfo.allocateSize = getBufferBitSize(pointerCastOp.getResult());
  if (!pointerLikeInfo.allocateSize.has_value()) {
    pointerCastOp.emitError("unknown buffer size");
    llvm_unreachable("unknown buffer size");
  }
  if (auto spaceAttr = GetBufferSpaceAttr(pointerCastOp.getResult())) {
    pointerLikeInfo.addressSpace = spaceAttr->getAddressSpace();
  }
  if (auto parentLoop = pointerCastOp->getParentOfType<LoopLikeOpInterface>()) {
    pointerLikeInfo.parentLoop = parentLoop;
  }
  return pointerLikeInfo;
}

MemInfo getMemInfo(Value val) {
  if (auto *defOp = val.getDefiningOp()) {
    if (auto pointerCastOp = llvm::dyn_cast<pto::PointerCastOp>(defOp)) {
      return MemInfo(val, getPointerLikeInfo(pointerCastOp));
    }
  }
  return MemInfo(val, isWorkSpaceFuncArgument(val));
}

MemInfo getMemInfo(const llvm::SmallVector<int64_t> &addrs) {
  MemInfo memInfo;
  memInfo.pointerLikeInfo = PointerLikeInfo();
  memInfo.pointerLikeInfo->addresses = addrs;
  memInfo.pointerLikeInfo->allocateSize = 1;
  memInfo.pointerLikeInfo->addressSpace = pto::AddressSpace::Zero;
  return memInfo;
}

static bool haveComparableAddressSpace(const PointerLikeInfo &lhs,
                                       const PointerLikeInfo &rhs) {
  return lhs.addressSpace.has_value() && rhs.addressSpace.has_value() &&
         lhs.addressSpace.value() == rhs.addressSpace.value();
}

static bool isIgnoredByEventId(std::optional<int64_t> eventIdNum, int64_t i,
                               int64_t j) {
  return eventIdNum.has_value() &&
         (i % eventIdNum.value()) == (j % eventIdNum.value());
}

static bool offsetsDefinitelySeparated(int64_t offset1, int64_t allocSz1,
                                       int64_t offset2) {
  return allocSz1 != ShapedType::kDynamic && offset1 + allocSz1 < offset2 + 1;
}

static bool doAddressesConflict(int64_t offset1, int64_t allocSz1,
                                int64_t offset2, int64_t allocSz2) {
  if (offset1 == ShapedType::kDynamic || offset2 == ShapedType::kDynamic)
    return true;
  if (offsetsDefinitelySeparated(offset1, allocSz1, offset2))
    return false;
  if (offsetsDefinitelySeparated(offset2, allocSz2, offset1))
    return false;
  return true;
}

bool PointerLikeInfo::checkConflict(const PointerLikeInfo &pointerLikeInfo1,
                                    const PointerLikeInfo &pointerLikeInfo2,
                                    std::optional<int64_t> lcmLen,
                                    std::optional<int64_t> eventIdNum) {
  if (!haveComparableAddressSpace(pointerLikeInfo1, pointerLikeInfo2))
    return false;

  auto &offsets1 = pointerLikeInfo1.addresses;
  auto &offsets2 = pointerLikeInfo2.addresses;
  auto sz1 = static_cast<int64_t>(offsets1.size());
  auto sz2 = static_cast<int64_t>(offsets2.size());

  int64_t len1 = sz1;
  int64_t len2 = sz2;
  if (lcmLen.has_value()) {
    len1 = lcmLen.value();
    len2 = lcmLen.value();
  }

  for (int64_t i = 0; i < len1; i++) {
    for (int64_t j = 0; j < len2; j++) {
      if (isIgnoredByEventId(eventIdNum, i, j))
        continue;

      auto offset1 = offsets1[i % sz1];
      auto offset2 = offsets2[j % sz2];
      ASSERT(pointerLikeInfo1.allocateSize.has_value());
      ASSERT(pointerLikeInfo2.allocateSize.has_value());
      auto allocSz1 = pointerLikeInfo1.allocateSize.value();
      auto allocSz2 = pointerLikeInfo2.allocateSize.value();
      if (doAddressesConflict(offset1, allocSz1, offset2, allocSz2))
        return true;
    }
  }
  return false;
}

bool MemInfo::checkConflict(const MemInfo &memInfo1, const MemInfo &memInfo2,
                            std::optional<int64_t> lcmLen,
                            std::optional<int64_t> eventIdNum) {
  if (memInfo1.pointerLikeInfo.has_value() &&
      memInfo2.pointerLikeInfo.has_value()) {
    return PointerLikeInfo::checkConflict(memInfo1.pointerLikeInfo.value(),
                                          memInfo2.pointerLikeInfo.value(),
                                          lcmLen, eventIdNum);
  }
  return memInfo1.value == memInfo2.value;
}

bool isWorkSpaceFuncArgument(Value value) {
  auto blockArg = dyn_cast_if_present<BlockArgument>(value);
  if (!blockArg) {
    return false;
  }
  auto *block = blockArg.getOwner();
  if (!block) {
    return false;
  }
  auto *region = block->getParent();
  if (!region) {
    return false;
  }
  auto *parentOp = region->getParentOp();
  if (!parentOp) {
    return false;
  }
  auto funcOp = dyn_cast<func::FuncOp>(parentOp);
  if (!funcOp) {
    return false;
  }
  return false;
}

} // namespace mlir::pto::syncsolver

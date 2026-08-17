// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//
// This file is derived from the HIVM GraphSyncSolver in AscendNPU-IR/bishengir
// (https://github.com/AscendNPU-IR/bishengir), licensed under the Apache
// License, Version 2.0.  Upstream copyright and license notice:
//   Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//       http://www.apache.org/licenses/LICENSE-2.0
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.

//===------------- MemInfo.cpp ---- Graph Sync Solver ---------------------===//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include "../Utils.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOMultiBuffer.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Support/CodeConstants.h"
#include "PTO/Transforms/GraphSyncSolver/MemInfo.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"


using namespace mlir;
using namespace pto::syncsolver;

namespace mlir::pto::syncsolver {

static std::optional<int64_t> getTileBufferBitSize(pto::TileBufType type) {
  auto bitWidth = getPTOStorageElemBitWidth(type.getElementType());
  if (bitWidth == 0) {
    return ShapedType::kDynamic;
  }

  ArrayRef<int64_t> shape = type.getShape();
  if (type.getCompactModeI32() ==
      static_cast<int32_t>(pto::CompactMode::RowPlusOne)) {
    if (shape.size() != mlir::pto::kValue2 || llvm::is_contained(shape, ShapedType::kDynamic)) {
      return ShapedType::kDynamic;
    }

    bool rowMajor = type.getBLayoutValueI32() ==
                    static_cast<int32_t>(pto::BLayout::RowMajor);
    int64_t major = rowMajor ? shape[0] : shape[1];
    int64_t minor = rowMajor ? shape[1] : shape[0];
    if (major == 0 || minor == 0) {
      return 0;
    }
    return ((major - 1) * (minor + 1) + minor) *
           static_cast<int64_t>(bitWidth);
  }

  int64_t numElements = 1;
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic) {
      return ShapedType::kDynamic;
    }
    numElements *= dim;
  }
  return numElements * bitWidth;
}

static std::optional<int64_t> getBufferBitSize(Value value) {
  if (auto multiType = dyn_cast<pto::MultiTileBufType>(value.getType())) {
    return getTileBufferBitSize(multiType.getSlotType());
  }
  if (auto tileType = dyn_cast<pto::TileBufType>(value.getType())) {
    return getTileBufferBitSize(tileType);
  }
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

static std::optional<pto::AddressSpace> getValueAddressSpace(Value value) {
  if (!value) {
    return std::nullopt;
  }
  if (auto space = pto::getPTOAddressSpaceAttr(value.getType())) {
    return space.getAddressSpace();
  }
  if (auto memRefType = dyn_cast<BaseMemRefType>(value.getType());
      memRefType && !memRefType.getMemorySpace()) {
    return pto::AddressSpace::GM;
  }
  if (isa<pto::TensorViewType, pto::PartitionTensorViewType>(value.getType())) {
    return pto::AddressSpace::GM;
  }
  return std::nullopt;
}

static MemInfo getConservativeIntToPtrMemInfo(pto::IntToPtrOp intToPtr) {
  PointerLikeInfo pointerLikeInfo(intToPtr);
  pointerLikeInfo.addressSpace = getValueAddressSpace(intToPtr.getResult());
  pointerLikeInfo.addresses.push_back(ShapedType::kDynamic);
  pointerLikeInfo.allocateSize = ShapedType::kDynamic;
  pointerLikeInfo.aliasesUnknownRange = true;
  return MemInfo(intToPtr.getResult(), pointerLikeInfo);
}

static std::optional<int64_t> getConstantI64(Value value) {
  IntegerAttr attr;
  if (!value || !matchPattern(value, m_Constant(&attr))) {
    return std::nullopt;
  }
  return attr.getValue().getSExtValue();
}

static PointerLikeInfo getPointerLikeInfo(pto::AllocMultiTileOp alloc) {
  PointerLikeInfo info(alloc);
  info.allocateSize = getBufferBitSize(alloc.getResult());
  auto slotType = alloc.getResult().getType().getSlotType();
  if (auto space = dyn_cast_or_null<pto::AddressSpaceAttr>(
          slotType.getMemorySpace())) {
    info.addressSpace = space.getAddressSpace();
  }

  if (auto planned = alloc->getAttrOfType<DenseI64ArrayAttr>(
          pto::kPtoMultiBufferAddrsAttrName)) {
    for (int64_t address : planned.asArrayRef()) {
      info.addresses.push_back(address * pto::kBitsToByte);
    }
  } else if (auto base = getConstantI64(alloc.getAddr())) {
    int64_t slotBits = info.allocateSize.value_or(ShapedType::kDynamic);
    for (uint32_t slot = 0; slot < alloc.getResult().getType().getCount(); ++slot) {
      info.addresses.push_back(*base * pto::kBitsToByte + slot * slotBits);
    }
  }
  if (auto loop = alloc->getParentOfType<LoopLikeOpInterface>()) {
    info.parentLoop = loop;
  }
  return info;
}

static MemInfo getMemInfoForMultiTileGet(pto::MultiTileGetOp get) {
  auto alloc = get.getSource().getDefiningOp<pto::AllocMultiTileOp>();
  if (!alloc) {
    return MemInfo(get.getResult(), isWorkSpaceFuncArgument(get.getResult()));
  }

  PointerLikeInfo info = getPointerLikeInfo(alloc);
  IntegerAttr slotAttr;
  if (matchPattern(get.getSlot(), m_Constant(&slotAttr)) &&
      info.addresses.size() > 1) {
    int64_t slot = slotAttr.getValue().getSExtValue();
    if (slot >= 0 && slot < static_cast<int64_t>(info.addresses.size())) {
      int64_t address = info.addresses[static_cast<size_t>(slot)];
      info.addresses.assign(1, address);
    }
  }
  if (auto loop = get->getParentOfType<LoopLikeOpInterface>()) {
    info.parentLoop = loop;
  }
  return MemInfo(get.getResult(), info);
}

MemInfo getMemInfo(Value val) {
  if (auto *defOp = val.getDefiningOp()) {
    if (auto intToPtr = llvm::dyn_cast<pto::IntToPtrOp>(defOp)) {
      return getConservativeIntToPtrMemInfo(intToPtr);
    }
    if (auto allocMulti = llvm::dyn_cast<pto::AllocMultiTileOp>(defOp)) {
      return MemInfo(val, getPointerLikeInfo(allocMulti));
    }
    if (auto multiGet = llvm::dyn_cast<pto::MultiTileGetOp>(defOp)) {
      return getMemInfoForMultiTileGet(multiGet);
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

bool PointerLikeInfo::checkConflict(const PointerLikeInfo &pointerLikeInfo1,
                                    const PointerLikeInfo &pointerLikeInfo2,
                                    std::optional<int64_t> lcmLen,
                                    std::optional<int64_t> eventIdNum) {
  if (!pointerLikeInfo1.addressSpace.has_value() ||
      !pointerLikeInfo2.addressSpace.has_value()) {
    return false;
  }
  if (pointerLikeInfo1.addressSpace.value() !=
      pointerLikeInfo2.addressSpace.value()) {
    return false;
  }

  if (pointerLikeInfo1.aliasesUnknownRange ||
      pointerLikeInfo2.aliasesUnknownRange) {
    return true;
  }

  auto &offsets1 = pointerLikeInfo1.addresses;
  auto &offsets2 = pointerLikeInfo2.addresses;
  auto sz1 = static_cast<int64_t>(offsets1.size());
  auto sz2 = static_cast<int64_t>(offsets2.size());
  if (sz1 == 0 || sz2 == 0) {
    return pointerLikeInfo1.addressSpace == pto::AddressSpace::GM ||
           pointerLikeInfo2.addressSpace == pto::AddressSpace::GM;
  }

  int64_t len1 = sz1;
  int64_t len2 = sz2;
  if (lcmLen.has_value()) {
    len1 = lcmLen.value();
    len2 = lcmLen.value();
  }

  for (int64_t i = 0; i < len1; i++) {
    for (int64_t j = 0; j < len2; j++) {
      if (eventIdNum.has_value()) {
        if ((i % eventIdNum.value()) == (j % eventIdNum.value())) {
          continue;
        }
      }

      auto offset1 = offsets1[i % sz1];
      auto offset2 = offsets2[j % sz2];
      if (offset1 == ShapedType::kDynamic || offset2 == ShapedType::kDynamic) {
        return true;
      }

      assert(pointerLikeInfo1.allocateSize.has_value());
      assert(pointerLikeInfo2.allocateSize.has_value());
      auto allocSz1 = pointerLikeInfo1.allocateSize.value();
      auto allocSz2 = pointerLikeInfo2.allocateSize.value();

      if ((allocSz1 != ShapedType::kDynamic) &&
          (offset1 + allocSz1 < offset2 + 1)) {
        continue;
      }
      if ((allocSz2 != ShapedType::kDynamic) &&
          (offset2 + allocSz2 < offset1 + 1)) {
        continue;
      }
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
  auto getUnknownAddressSpace =
      [](const MemInfo &memInfo) -> std::optional<pto::AddressSpace> {
    if (!memInfo.pointerLikeInfo ||
        !memInfo.pointerLikeInfo->aliasesUnknownRange) {
      return std::nullopt;
    }
    return memInfo.pointerLikeInfo->addressSpace;
  };
  auto aliasesAddressSpace =
      [](const MemInfo &memInfo, pto::AddressSpace space) -> bool {
    if (memInfo.pointerLikeInfo && memInfo.pointerLikeInfo->addressSpace &&
        *memInfo.pointerLikeInfo->addressSpace == space) {
      return true;
    }
    auto valueSpace = getValueAddressSpace(memInfo.value);
    return valueSpace && *valueSpace == space;
  };

  if (auto unknownSpace = getUnknownAddressSpace(memInfo1)) {
    if (aliasesAddressSpace(memInfo2, *unknownSpace)) {
      return true;
    }
  }
  if (auto unknownSpace = getUnknownAddressSpace(memInfo2)) {
    if (aliasesAddressSpace(memInfo1, *unknownSpace)) {
      return true;
    }
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

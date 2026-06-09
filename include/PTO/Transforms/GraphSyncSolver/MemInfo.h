// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===------------- MemInfo.h ---- Graph Sync Solver -----------------------===//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_MEMINFO_H
#define MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_MEMINFO_H

#include "PTO/IR/PTO.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::pto::syncsolver {

struct PointerLikeInfo {
  Operation *op{nullptr};
  llvm::SmallVector<int64_t> addresses;
  std::optional<int64_t> allocateSize;
  std::optional<pto::AddressSpace> addressSpace;
  LoopLikeOpInterface parentLoop{nullptr};

  PointerLikeInfo() = default;
  explicit PointerLikeInfo(Operation *op) : op(op) {}

  std::string str() const;

  static bool checkConflict(const PointerLikeInfo &pointerLikeInfo1,
                            const PointerLikeInfo &pointerLikeInfo2,
                            std::optional<int64_t> lcmLen = {},
                            std::optional<int64_t> eventIdNum = {});
};

struct MemInfo {
  Value value{nullptr};
  std::optional<PointerLikeInfo> pointerLikeInfo;
  bool isWorkSpace{false};

  MemInfo() = default;
  explicit MemInfo(Value value, bool isWorkSpace = false)
      : value(value), isWorkSpace(isWorkSpace) {}

  explicit MemInfo(Value value, PointerLikeInfo pointerLikeInfo,
                   bool isWorkSpace = false)
      : value(value), pointerLikeInfo(pointerLikeInfo),
        isWorkSpace(isWorkSpace) {}

  int64_t getSz() const {
    if (pointerLikeInfo.has_value()) {
      return pointerLikeInfo->addresses.size();
    }
    if (value != nullptr) {
      return 1;
    }
    return 0;
  }

  std::string str() const;

  static bool checkConflict(const MemInfo &memInfo1, const MemInfo &memInfo2,
                            std::optional<int64_t> lcmLen = {},
                            std::optional<int64_t> eventIdNum = {});
};

llvm::SmallVector<int64_t> getAddresses(const llvm::SmallVector<Value> &addrs);

PointerLikeInfo getPointerLikeInfo(pto::PointerCastOp pointerCastOp);

MemInfo getMemInfo(Value val);

MemInfo getMemInfo(const llvm::SmallVector<int64_t> &addrs);

bool isWorkSpaceFuncArgument(Value value);

} // namespace mlir::pto::syncsolver

#endif // MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_MEMINFO_H

// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- PTOSimtKeepResumeShared.h - SIMT keep/resume verify helpers --------===//
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Shared verification helpers for the SIMT keep/resume register-slot model,
// used by both the op-level verifiers (KeepOp::verify / ResumeOp::verify via
// the PTO.cpp translation unit) and the surface validation pass
// (PTOValidateVPTOIR.cpp). Previously each side kept a private copy.
// Internal to lib/PTO; not installed.
//
// Note on const: `dyn_cast<KeepOp/ResumeOp>` requires a non-const
// `Operation *` (MLIR op wrappers only construct from non-const operations),
// so `first` parameters stay non-const even where only traversal is done.
//===----------------------------------------------------------------------===//

#ifndef PTO_IR_SIMT_KEEP_RESUME_SHARED_H
#define PTO_IR_SIMT_KEEP_RESUME_SHARED_H

#include "PTO/IR/PTO.h"
#include "PTO/Support/CodeConstants.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace mlir::pto::simt_detail {

using namespace mlir;

inline constexpr int64_t kSimtKeepResumeSlotLimit = 123;

inline Operation *getFirstNonConstantLikeOp(Block *block) {
  if (!block) {
    return nullptr;
  }
  for (Operation &op : *block) {
    if (!op.hasTrait<OpTrait::ConstantLike>()) {
      return &op;
    }
  }
  return nullptr;
}

inline bool isOpInRange(const Operation *op, const Operation *first,
                          const Operation *last) {
  for (const Operation *cur = first; cur; cur = cur->getNextNode()) {
    if (cur == op) {
      return true;
    }
    if (cur == last) {
      return false;
    }
  }
  return false;
}

inline std::optional<unsigned> getSimtKeepResumeRegisterCount(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    if (intType.getWidth() <= mlir::pto::kValue32) {
      return 1;
    }
    if (intType.getWidth() == mlir::pto::kValue64) {
      return mlir::pto::kValue2;
    }
    return std::nullopt;
  }
  if (type.isF16() || type.isBF16() || type.isF32()) {
    return 1;
  }
  return std::nullopt;
}

inline Type getSimtKeepResumeValueType(KeepOp op) {
  return op.getPayload().getType();
}

inline Type getSimtKeepResumeValueType(ResumeOp op) {
  return op.getResult().getType();
}

template <typename OpT>
LogicalResult verifySimtKeepResumeSlotRange(OpT op) {
  std::optional<unsigned> registerCount =
      getSimtKeepResumeRegisterCount(getSimtKeepResumeValueType(op));
  if (!registerCount) {
    return success();
  }
  int64_t slot = op.getSlot();
  if (slot < 0 || slot >= kSimtKeepResumeSlotLimit) {
    return op.emitOpError()
           << "requires slot in range [0, "
           << (kSimtKeepResumeSlotLimit - 1) << "]";
  }
  if (*registerCount == mlir::pto::kValue2) {
    if ((slot % mlir::pto::kValue2) != 0) {
      return op.emitOpError()
             << "requires an even slot for 64-bit keep/resume values";
    }
    if (slot + 1 >= kSimtKeepResumeSlotLimit) {
      return op.emitOpError()
             << "requires slot in range [0, "
             << (kSimtKeepResumeSlotLimit - mlir::pto::kValue2)
             << "] for 64-bit keep/resume values";
    }
  }
  return success();
}

template <typename OpT>
bool overlapsEarlierSimtKeepResumeSlotUse(OpT op,
                                          SmallVectorImpl<int64_t> &used) {
  std::optional<unsigned> registerCount =
      getSimtKeepResumeRegisterCount(getSimtKeepResumeValueType(op));
  if (!registerCount) {
    return false;
  }
  int64_t slot = op.getSlot();
  for (int64_t word = slot; word < slot + *registerCount; ++word) {
    if (llvm::is_contained(used, word)) {
      return true;
    }
  }
  for (int64_t word = slot; word < slot + *registerCount; ++word) {
    used.push_back(word);
  }
  return false;
}

inline LogicalResult verifyUniqueResumeGroupSlots(ResumeOp current,
                                                  Operation *first) {
  SmallVector<int64_t, mlir::pto::kValue4> slots;
  for (Operation *cur = first; cur;
       cur = cur->getNextNode()) {
    auto resume = dyn_cast<ResumeOp>(cur);
    if (!resume) {
      break;
    }
    if (overlapsEarlierSimtKeepResumeSlotUse(resume, slots) &&
        resume.getOperation() == current.getOperation()) {
      return current.emitOpError()
             << "duplicates an earlier slot " << resume.getSlot()
             << " in the SIMT resume prologue group";
    }
  }
  return success();
}

inline LogicalResult verifyUniqueKeepGroupSlots(KeepOp current,
                                                Operation *first,
                                                Operation *last) {
  SmallVector<int64_t, mlir::pto::kValue4> slots;
  for (Operation *cur = first; cur;
       cur = cur->getNextNode()) {
    auto keep = dyn_cast<KeepOp>(cur);
    if (!keep) {
      break;
    }
    if (overlapsEarlierSimtKeepResumeSlotUse(keep, slots) &&
        keep.getOperation() == current.getOperation()) {
      return current.emitOpError()
             << "duplicates an earlier slot " << keep.getSlot()
             << " in the SIMT keep epilogue group";
    }
    if (cur == last) {
      break;
    }
  }
  return success();
}

inline bool isSupportedSimtKeepResumeType(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    return intType.getWidth() <= mlir::pto::kValue64;
  }
  return type.isF16() || type.isBF16() || type.isF32();
}

} // namespace mlir::pto::simt_detail

#endif // PTO_IR_SIMT_KEEP_RESUME_SHARED_H

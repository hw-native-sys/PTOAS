// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===------- UtilityDenseKeys.h ---- Graph Sync Solver Dense Keys ---------===//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_UTILITYDENSEKEYS_H
#define MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_UTILITYDENSEKEYS_H

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/GraphSyncSolver/SyncSolverIR.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#include <cstddef>
#include <cstdint>
#include <tuple>

namespace mlir::pto::syncsolver {
struct CorePipeInfo {
  pto::TCoreType coreType{pto::TCoreType::CUBE_OR_VECTOR};
  pto::PIPE pipe{pto::PIPE::PIPE_UNASSIGNED};

  CorePipeInfo() = default;

  CorePipeInfo(pto::TCoreType coreType, pto::PIPE pipe)
      : coreType(coreType), pipe(pipe) {}

  CorePipeInfo(std::pair<pto::TCoreType, pto::PIPE> corePipePair)
      : mlir::pto::syncsolver::CorePipeInfo(corePipePair.first,
                                             corePipePair.second) {}

  bool operator==(const CorePipeInfo &other) const {
    return std::tie(coreType, pipe) == std::tie(other.coreType, other.pipe);
  }

  bool operator!=(const CorePipeInfo &other) const { return !(*this == other); }

  bool operator<(const CorePipeInfo &other) const {
    return std::tie(coreType, pipe) < std::tie(other.coreType, other.pipe);
  }
};

struct CorePipeInfoKeyInfo {
  using CorePipePairTy = std::pair<pto::TCoreType, pto::PIPE>;

  static inline CorePipeInfo getEmptyKey() {
    return CorePipeInfo(llvm::DenseMapInfo<CorePipePairTy>::getEmptyKey());
  }

  static inline CorePipeInfo getTombstoneKey() {
    return CorePipeInfo(llvm::DenseMapInfo<CorePipePairTy>::getTombstoneKey());
  }

  static unsigned getHashValue(const CorePipeInfo &val) {
    return llvm::DenseMapInfo<CorePipePairTy>::getHashValue(
        {val.coreType, val.pipe});
  }

  static bool isEqual(const CorePipeInfo &lhs, const CorePipeInfo &rhs) {
    return lhs == rhs;
  }
};

using CorePipePairKey = std::tuple<CorePipeInfo, CorePipeInfo>;
using CorePipeEventKey = std::tuple<CorePipeInfo, CorePipeInfo, int64_t>;
using SyncScopePairKey =
    std::tuple<OperationBase *, OperationBase *, OperationBase *, CorePipeInfo,
               CorePipeInfo>;
constexpr size_t kCorePipePairFirstIndex = 0;
constexpr size_t kCorePipePairSecondIndex = 1;
constexpr size_t kCorePipeEventIdIndex = 2;
constexpr size_t kSyncScopeScopeIndex = 0;
constexpr size_t kSyncScopeProducerIndex = 1;
constexpr size_t kSyncScopeConsumerIndex = 2;
constexpr size_t kSyncScopeProducerPipeIndex = 3;
constexpr size_t kSyncScopeConsumerPipeIndex = 4;

static inline unsigned combineDenseHash(unsigned lhs, unsigned rhs) {
  return (lhs * 37U) ^ rhs;
}

struct CorePipePairKeyInfo {
  static inline CorePipePairKey getEmptyKey() {
    return {CorePipeInfoKeyInfo::getEmptyKey(),
            CorePipeInfoKeyInfo::getEmptyKey()};
  }

  static inline CorePipePairKey getTombstoneKey() {
    return {CorePipeInfoKeyInfo::getTombstoneKey(),
            CorePipeInfoKeyInfo::getTombstoneKey()};
  }

  static unsigned getHashValue(const CorePipePairKey &val) {
    return combineDenseHash(
        CorePipeInfoKeyInfo::getHashValue(std::get<kCorePipePairFirstIndex>(val)),
        CorePipeInfoKeyInfo::getHashValue(
            std::get<kCorePipePairSecondIndex>(val)));
  }

  static bool isEqual(const CorePipePairKey &lhs,
                      const CorePipePairKey &rhs) {
    return CorePipeInfoKeyInfo::isEqual(
               std::get<kCorePipePairFirstIndex>(lhs),
               std::get<kCorePipePairFirstIndex>(rhs)) &&
           CorePipeInfoKeyInfo::isEqual(
               std::get<kCorePipePairSecondIndex>(lhs),
               std::get<kCorePipePairSecondIndex>(rhs));
  }
};

struct CorePipeEventKeyInfo {
  static inline CorePipeEventKey getEmptyKey() {
    return {CorePipeInfoKeyInfo::getEmptyKey(),
            CorePipeInfoKeyInfo::getEmptyKey(),
            llvm::DenseMapInfo<int64_t>::getEmptyKey()};
  }

  static inline CorePipeEventKey getTombstoneKey() {
    return {CorePipeInfoKeyInfo::getTombstoneKey(),
            CorePipeInfoKeyInfo::getTombstoneKey(),
            llvm::DenseMapInfo<int64_t>::getTombstoneKey()};
  }

  static unsigned getHashValue(const CorePipeEventKey &val) {
    unsigned hash = CorePipePairKeyInfo::getHashValue(
        {std::get<kCorePipePairFirstIndex>(val),
         std::get<kCorePipePairSecondIndex>(val)});
    return combineDenseHash(hash,
                            llvm::DenseMapInfo<int64_t>::getHashValue(
                                std::get<kCorePipeEventIdIndex>(val)));
  }

  static bool isEqual(const CorePipeEventKey &lhs,
                      const CorePipeEventKey &rhs) {
    return CorePipePairKeyInfo::isEqual(
               {std::get<kCorePipePairFirstIndex>(lhs),
                std::get<kCorePipePairSecondIndex>(lhs)},
               {std::get<kCorePipePairFirstIndex>(rhs),
                std::get<kCorePipePairSecondIndex>(rhs)}) &&
           llvm::DenseMapInfo<int64_t>::isEqual(
               std::get<kCorePipeEventIdIndex>(lhs),
               std::get<kCorePipeEventIdIndex>(rhs));
  }
};

struct SyncScopePairKeyInfo {
  static inline SyncScopePairKey getEmptyKey() {
    return {llvm::DenseMapInfo<OperationBase *>::getEmptyKey(),
            llvm::DenseMapInfo<OperationBase *>::getEmptyKey(),
            llvm::DenseMapInfo<OperationBase *>::getEmptyKey(),
            CorePipeInfoKeyInfo::getEmptyKey(),
            CorePipeInfoKeyInfo::getEmptyKey()};
  }

  static inline SyncScopePairKey getTombstoneKey() {
    return {llvm::DenseMapInfo<OperationBase *>::getTombstoneKey(),
            llvm::DenseMapInfo<OperationBase *>::getTombstoneKey(),
            llvm::DenseMapInfo<OperationBase *>::getTombstoneKey(),
            CorePipeInfoKeyInfo::getTombstoneKey(),
            CorePipeInfoKeyInfo::getTombstoneKey()};
  }

  static unsigned getHashValue(const SyncScopePairKey &val) {
    unsigned hash =
        llvm::DenseMapInfo<OperationBase *>::getHashValue(
            std::get<kSyncScopeScopeIndex>(val));
    hash = combineDenseHash(hash,
                            llvm::DenseMapInfo<OperationBase *>::getHashValue(
                                std::get<kSyncScopeProducerIndex>(val)));
    hash = combineDenseHash(hash,
                            llvm::DenseMapInfo<OperationBase *>::getHashValue(
                                std::get<kSyncScopeConsumerIndex>(val)));
    hash = combineDenseHash(
        hash,
        CorePipeInfoKeyInfo::getHashValue(
            std::get<kSyncScopeProducerPipeIndex>(val)));
    return combineDenseHash(
        hash,
        CorePipeInfoKeyInfo::getHashValue(
            std::get<kSyncScopeConsumerPipeIndex>(val)));
  }

  static bool isEqual(const SyncScopePairKey &lhs,
                      const SyncScopePairKey &rhs) {
    return llvm::DenseMapInfo<OperationBase *>::isEqual(
               std::get<kSyncScopeScopeIndex>(lhs),
               std::get<kSyncScopeScopeIndex>(rhs)) &&
           llvm::DenseMapInfo<OperationBase *>::isEqual(
               std::get<kSyncScopeProducerIndex>(lhs),
               std::get<kSyncScopeProducerIndex>(rhs)) &&
           llvm::DenseMapInfo<OperationBase *>::isEqual(
               std::get<kSyncScopeConsumerIndex>(lhs),
               std::get<kSyncScopeConsumerIndex>(rhs)) &&
           CorePipeInfoKeyInfo::isEqual(
               std::get<kSyncScopeProducerPipeIndex>(lhs),
               std::get<kSyncScopeProducerPipeIndex>(rhs)) &&
           CorePipeInfoKeyInfo::isEqual(
               std::get<kSyncScopeConsumerPipeIndex>(lhs),
               std::get<kSyncScopeConsumerPipeIndex>(rhs));
  }
};

template <typename ValueT>
using CorePipeDenseMap =
    llvm::DenseMap<CorePipeInfo, ValueT, CorePipeInfoKeyInfo>;
template <typename ValueT>
using CorePipePairDenseMap =
    llvm::DenseMap<CorePipePairKey, ValueT, CorePipePairKeyInfo>;
using CorePipePairDenseSet =
    llvm::DenseSet<CorePipePairKey, CorePipePairKeyInfo>;
using CorePipeEventDenseSet =
    llvm::DenseSet<CorePipeEventKey, CorePipeEventKeyInfo>;
template <typename ValueT>
using SyncScopePairDenseMap =
    llvm::DenseMap<SyncScopePairKey, ValueT, SyncScopePairKeyInfo>;

} // namespace mlir::pto::syncsolver

#endif // MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_UTILITYDENSEKEYS_H

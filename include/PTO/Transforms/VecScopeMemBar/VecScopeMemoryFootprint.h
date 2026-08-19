// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TRANSFORMS_VECSCOPEMEMBAR_VECSCOPEMEMORYFOOTPRINT_H
#define PTO_TRANSFORMS_VECSCOPEMEMBAR_VECSCOPEMEMORYFOOTPRINT_H

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/VecScopeMemBar/VecScopeMemBarIR.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <utility>

namespace mlir::pto::vecscopemembar {

struct AffineByteExpr {
  int64_t constant = 0;
  SmallVector<std::pair<Value, int64_t>, 2> coefficients;
  bool exact = true;

  int64_t getCoeff(Value v) const;
  bool isConstant() const { return coefficients.empty(); }
  AffineByteExpr &scale(int64_t factor);
  AffineByteExpr &combine(const AffineByteExpr &other);
};

enum class MemoryRootKind {
  Absolute,
  ProvenAllocation,
  Symbolic,
  Unknown,
};

struct VecScopeMemoryFootprint {
  MemoryRootKind rootKind = MemoryRootKind::Unknown;
  Value root;
  std::optional<uint64_t> absoluteBase;
  std::optional<AddressSpace> addressSpace;
  AffineByteExpr byteOffset;
  std::optional<uint64_t> byteSize;
  bool forcesMayAlias = false;
};

enum class VecScopeAccessKind { Load, Store };

struct AccessOccurrence {
  unsigned id = 0;
  Operation *op = nullptr;
  VecScopeAccessKind kind = VecScopeAccessKind::Load;
  VecScopeMemoryFootprint footprint;
  SmallVector<unsigned, 2> loopNest;
  SmallVector<unsigned, 2> schedulePath;
  unsigned lexicalOrder = 0;
};

struct VecMemoryAccessDescriptor {
  VecScopeAccessKind kind = VecScopeAccessKind::Load;
  Value base;
  std::optional<AddressSpace> addressSpace;
  AffineByteExpr byteOffset;
  std::optional<uint64_t> conservativeByteSize;
  bool forcesMayAlias = false;
};

enum class VecScopeAliasResult {
  NoAlias,
  MayAlias,
  MustOrPartialAlias,
};

FailureOr<VecMemoryAccessDescriptor> buildAccessDescriptor(Operation *op,
                                                           ArrayRef<Value> ivs);

VecScopeAliasResult aliasSameIteration(const VecScopeMemoryFootprint &producer,
                                       const VecScopeMemoryFootprint &consumer);

VecScopeMemoryFootprint
footprintFromDescriptor(const VecMemoryAccessDescriptor &desc,
                        ArrayRef<Value> ivs);

} // namespace mlir::pto::vecscopemembar

#endif // PTO_TRANSFORMS_VECSCOPEMEMBAR_VECSCOPEMEMORYFOOTPRINT_H

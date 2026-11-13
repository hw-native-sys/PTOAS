// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_ANALYSIS_PTOMEMORYACCESSANALYSIS_H
#define PTO_ANALYSIS_PTOMEMORYACCESSANALYSIS_H

#include "PTO/IR/PTO.h"
#include "PTO/Analysis/PTOAddressAnalysis.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <utility>

namespace mlir::pto {

enum class PTOMemoryAccessUnit { Vector, Scalar };

struct PTOAffineByteExpr {
    int64_t constant = 0;
    SmallVector<std::pair<Value, int64_t>, 2> coefficients;
    bool exact = true;
    // Lossy expressions no longer describe the address. Faithful symbolic sums
    // may be bounded or cancelled; lossy ones must retain the hazard.
    bool lossy = false;

    int64_t getCoeff(Value v) const;
    bool isConstant() const { return coefficients.empty(); }
    PTOAffineByteExpr& scale(int64_t factor);
    PTOAffineByteExpr& combine(const PTOAffineByteExpr& other);
};

enum class PTOMemoryRootKind {
    Absolute,
    ProvenAllocation,
    Symbolic,
    Unknown,
};

struct PTOMemoryFootprint {
    PTOMemoryRootKind rootKind = PTOMemoryRootKind::Unknown;
    Value root;
    std::optional<uint64_t> absoluteBase;
    std::optional<AddressSpace> addressSpace;
    PTOAffineByteExpr byteOffset;
    std::optional<uint64_t> byteSize;
    bool forcesMayAlias = false;
    // The pointer and element-offset operands the access was issued with.
    Value accessBase;
    Value accessOffset;
    // Indexed offsets are covered by a certified lane domain, falling back to
    // the full unsigned hardware domain when value certification fails.
    bool offsetDomainCovered = false;
};

enum class PTOMemoryAccessKind { Load, Store };

struct PTOMemoryAccessDescriptor {
    PTOMemoryAccessKind kind = PTOMemoryAccessKind::Load;
    PTOMemoryAccessUnit unit = PTOMemoryAccessUnit::Vector;
    Value base;
    Value accessOffset;
    bool offsetDomainCovered = false;
    std::optional<AddressSpace> addressSpace;
    PTOAffineByteExpr byteOffset;
    std::optional<uint64_t> conservativeByteSize;
    bool forcesMayAlias = false;
    // Disjoint 32-byte pieces of a block-strided store, relative to byteOffset.
    // Empty means the descriptor represents a single interval.
    SmallVector<int64_t, 8> blockByteOffsets;
};

bool isUBBackedType(Type type);

bool isUBVectorStore(Operation* op);

bool isUBVectorLoad(Operation* op);

bool isUBVectorMemoryOp(Operation* op);

bool isUBScalarMemoryOp(Operation* op);

bool isUBScalarStore(Operation* op);

bool isUBScalarLoad(Operation* op);

bool isUBMemoryOp(Operation* op);

SmallVector<Value, 2> getStoredValues(Operation* storeOp);

SmallVector<Value, 2> getLoadedValues(Operation* loadOp);

enum class PTOMemoryAliasResult {
    NoAlias,
    MayAlias,
    MustOrPartialAlias,
};

// Preserve disjoint segments instead of replacing block-strided accesses by
// a contiguous hull. An empty result means unsupported, never no access.
SmallVector<PTOMemoryFootprint> getMemoryFootprints(Operation* op, ArrayRef<Value> ivs);
SmallVector<PTOMemoryFootprint> getMemoryFootprints(const PTOMemoryAccessDescriptor& descriptor, ArrayRef<Value> ivs);

FailureOr<PTOMemoryAccessDescriptor> buildMemoryAccessDescriptor(Operation* op, ArrayRef<Value> ivs);

std::optional<bool> memoryIntervalsOverlap(
    std::optional<int64_t> lhsOffset, std::optional<int64_t> lhsSize, std::optional<int64_t> rhsOffset,
    std::optional<int64_t> rhsSize);

PTOMemoryAliasResult aliasMemorySameIteration(const PTOMemoryFootprint& producer, const PTOMemoryFootprint& consumer);

PTOMemoryFootprint memoryFootprintFromDescriptor(const PTOMemoryAccessDescriptor& desc, ArrayRef<Value> ivs);

} // namespace mlir::pto

#endif // PTO_ANALYSIS_PTOMEMORYACCESSANALYSIS_H

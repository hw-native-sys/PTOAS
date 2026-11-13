// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Analysis/PTOMemoryAliasAnalysis.h"
#include "PTO/Analysis/PTOIntegerRange.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Support/CodeConstants.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <array>
#include <climits>
#include <limits>

using namespace mlir;
using namespace mlir::pto;

constexpr unsigned kBitsPerByte = CHAR_BIT;
constexpr unsigned kVisitedOperationInlineCapacity = mlir::pto::kValue8;

static std::optional<int64_t> getConstantInteger(Value value)
{
    IntegerAttr constant;
    bool invalidConstant =
        !value || !matchPattern(value, m_Constant(&constant)) || !constant.getValue().isSignedIntN(64);
    if (invalidConstant) {
        return std::nullopt;
    }
    return constant.getValue().getSExtValue();
}

std::optional<int64_t> pto::getExactByteDifference(const ResolvedMemoryAccess& from, const ResolvedMemoryAccess& to)
{
    if (!from.addressExpression || !to.addressExpression) {
        return std::nullopt;
    }
    Value base = from.addressExpression->currentBase;
    auto function = base.getParentRegion()->getParentOfType<func::FuncOp>();
    if (!function) {
        return std::nullopt;
    }
    PTOValueEvolutionAnalysis values(function);
    PTOAddressAnalysis addresses(function, values);
    auto difference = addresses.getDifferenceBytes(*from.addressExpression, *to.addressExpression, true);
    auto linear = difference ? normalizePTOLinearExpr(*difference.value) : std::nullopt;
    if (!linear || !linear->terms.empty()) {
        return std::nullopt;
    }
    return linear->constant;
}

/// Resolve a conservative byte-address range for pointer-producing operations
/// explicit at the VPTO scheduling boundary. Unknown transforms and possible
/// integer overflow deliberately fall back to may-alias.
static std::optional<IntegerRange> addDisplacement(int64_t address, const IntegerRange& displacement)
{
    IntegerRange result;
    bool overflow = llvm::AddOverflow(address, displacement.lowerInclusive, result.lowerInclusive) != 0 ||
                    llvm::AddOverflow(address, displacement.upperInclusive, result.upperInclusive) != 0;
    if (overflow) {
        return std::nullopt;
    }
    return result;
}

static std::optional<int64_t> getAddPtrElementByteSize(AddPtrOp addPtr)
{
    auto pointerType = dyn_cast<PtrType>(addPtr.getPtr().getType());
    if (!pointerType) {
        return std::nullopt;
    }
    Type elementType = pointerType.getElementType();
    bool invalidElementType = !elementType.isIntOrFloat() || elementType.getIntOrFloatBitWidth() % kBitsPerByte != 0;
    if (invalidElementType) {
        return std::nullopt;
    }
    return static_cast<int64_t>(elementType.getIntOrFloatBitWidth() / kBitsPerByte);
}

static LogicalResult accumulateAddPtrDisplacement(AddPtrOp addPtr, IntegerRange& displacement)
{
    std::optional<IntegerRange> elementOffset = [&]() -> std::optional<IntegerRange> {
        auto range = integer_range::inferPointIntegerRange(addPtr.getOffset());
        bool unsupportedRange = !range || !range->smin().isSignedIntN(64) || !range->smax().isSignedIntN(64);
        if (unsupportedRange) {
            return std::nullopt;
        }
        return IntegerRange{range->smin().getSExtValue(), range->smax().getSExtValue()};
    }();
    std::optional<int64_t> elementByteSize = getAddPtrElementByteSize(addPtr);
    if (!elementOffset || !elementByteSize) {
        return failure();
    }
    int64_t minimumByteOffset = 0;
    int64_t maximumByteOffset = 0;
    return failure(
        llvm::MulOverflow(elementOffset->lowerInclusive, *elementByteSize, minimumByteOffset) != 0 ||
        llvm::MulOverflow(elementOffset->upperInclusive, *elementByteSize, maximumByteOffset) != 0 ||
        llvm::AddOverflow(displacement.lowerInclusive, minimumByteOffset, displacement.lowerInclusive) != 0 ||
        llvm::AddOverflow(displacement.upperInclusive, maximumByteOffset, displacement.upperInclusive) != 0);
}

static std::optional<IntegerRange> getPointerAddressRange(Value value)
{
    SmallPtrSet<Operation*, kVisitedOperationInlineCapacity> visited;
    IntegerRange displacement;
    while (Operation* definingOp = value.getDefiningOp()) {
        if (!visited.insert(definingOp).second) {
            return std::nullopt;
        }
        if (auto cast = dyn_cast<CastPtrOp>(definingOp)) {
            Value input = cast.getInput();
            if (std::optional<int64_t> address = getConstantInteger(input)) {
                return addDisplacement(*address, displacement);
            }
            if (!isa<PtrType>(input.getType())) {
                return std::nullopt;
            }
            value = input;
            continue;
        }
        if (auto addPtr = dyn_cast<AddPtrOp>(definingOp)) {
            if (failed(accumulateAddPtrDisplacement(addPtr, displacement))) {
                return std::nullopt;
            }
            value = addPtr.getPtr();
            continue;
        }
        return std::nullopt;
    }
    return std::nullopt;
}

static Value getAliasRoot(Value value)
{
    SmallPtrSet<Operation*, mlir::pto::kValue8> visited;
    while (Operation* definingOp = value.getDefiningOp()) {
        if (!visited.insert(definingOp).second) {
            break;
        }
        std::optional<std::pair<Value, Value>> alias = getOperationAliasInfo(definingOp);
        if (!alias || alias->first != value || !alias->second) {
            break;
        }
        value = alias->second;
    }
    return value;
}

static const PTOAddressExpr* findUniqueAddressExpression(Value address, ArrayRef<PTOAddressExpr> expressions)
{
    const PTOAddressExpr* matching = nullptr;
    for (const PTOAddressExpr& expression : expressions) {
        if (expression.currentBase != address) {
            continue;
        }
        if (matching) {
            return nullptr;
        }
        matching = &expression;
    }
    return matching;
}

static void resolveAbsoluteByteRange(ResolvedMemoryAccess& access)
{
    std::optional<IntegerRange> pointerRange = getPointerAddressRange(access.semantics.address);
    if (!pointerRange || !access.semantics.byteOffset) {
        return;
    }
    IntegerRange byteRange;
    bool overflow =
        llvm::AddOverflow(pointerRange->lowerInclusive, *access.semantics.byteOffset, byteRange.lowerInclusive) != 0 ||
        llvm::AddOverflow(pointerRange->upperInclusive, *access.semantics.byteOffset, byteRange.upperInclusive) != 0;
    if (overflow) {
        return;
    }
    access.absoluteByteRange = byteRange;
    if (byteRange.lowerInclusive == byteRange.upperInclusive) {
        access.absoluteByteOffset = byteRange.lowerInclusive;
    }
}

static ResolvedMemoryAccess resolveMemoryAccess(
    const VPTOMemoryAccess& memoryAccess, ArrayRef<PTOAddressExpr> addressExpressions)
{
    ResolvedMemoryAccess access{memoryAccess, {}, std::nullopt, std::nullopt, std::nullopt};
    if (!access.semantics.address) {
        return access;
    }
    if (const PTOAddressExpr* expression = findUniqueAddressExpression(access.semantics.address, addressExpressions)) {
        access.addressExpression = *expression;
    }
    resolveAbsoluteByteRange(access);
    access.aliasRoot = getAliasRoot(access.semantics.address);
    if (access.aliasRoot != access.semantics.address) {
        access.semantics.byteOffset.reset();
    }
    return access;
}

SmallVector<ResolvedMemoryAccess> pto::resolveMemoryAccesses(
    Operation* operation, const VPTOSchedulingSemantics& semantics, PTOAddressAnalysis* addressAnalysis)
{
    SmallVector<PTOAddressExpr> addressExpressions;
    if (addressAnalysis) {
        auto result = addressAnalysis->getAddresses(operation);
        if (result) {
            addressExpressions = std::move(*result.value);
        }
    }
    SmallVector<ResolvedMemoryAccess> accesses;
    accesses.reserve(semantics.memoryAccesses.size());
    for (const VPTOMemoryAccess& memoryAccess : semantics.memoryAccesses) {
        accesses.push_back(resolveMemoryAccess(memoryAccess, addressExpressions));
    }
    return accesses;
}

static std::optional<bool> getExactDifferenceOverlap(const ResolvedMemoryAccess& lhs, const ResolvedMemoryAccess& rhs)
{
    bool hasPositiveSizes =
        lhs.semantics.byteSize && *lhs.semantics.byteSize > 0 && rhs.semantics.byteSize && *rhs.semantics.byteSize > 0;
    if (!hasPositiveSizes) {
        return std::nullopt;
    }
    std::optional<int64_t> rhsBegin = getExactByteDifference(lhs, rhs);
    return memoryIntervalsOverlap(0, lhs.semantics.byteSize, rhsBegin, rhs.semantics.byteSize);
}

static bool hasDisjointAbsoluteRanges(const ResolvedMemoryAccess& lhs, const ResolvedMemoryAccess& rhs)
{
    if (!lhs.absoluteByteRange || !lhs.semantics.byteSize || !rhs.absoluteByteRange || !rhs.semantics.byteSize) {
        return false;
    }
    int64_t lhsLatestEnd = 0;
    int64_t rhsLatestEnd = 0;
    bool overflow =
        llvm::AddOverflow(lhs.absoluteByteRange->upperInclusive, *lhs.semantics.byteSize, lhsLatestEnd) != 0 ||
        llvm::AddOverflow(rhs.absoluteByteRange->upperInclusive, *rhs.semantics.byteSize, rhsLatestEnd) != 0;
    return !overflow && (lhsLatestEnd <= rhs.absoluteByteRange->lowerInclusive ||
                         rhsLatestEnd <= lhs.absoluteByteRange->lowerInclusive);
}

bool pto::mayAlias(const ResolvedMemoryAccess& lhs, const ResolvedMemoryAccess& rhs)
{
    bool differentAddressSpaces = lhs.semantics.addressSpace && rhs.semantics.addressSpace &&
                                  lhs.semantics.addressSpace != rhs.semantics.addressSpace;
    if (differentAddressSpaces) {
        return false;
    }
    if (!lhs.semantics.address || !rhs.semantics.address) {
        return true;
    }
    if (std::optional<bool> overlap = getExactDifferenceOverlap(lhs, rhs)) {
        return *overlap;
    }
    if (hasDisjointAbsoluteRanges(lhs, rhs)) {
        return false;
    }
    if (std::optional<bool> overlap = memoryIntervalsOverlap(
            lhs.absoluteByteOffset, lhs.semantics.byteSize, rhs.absoluteByteOffset, rhs.semantics.byteSize)) {
        return *overlap;
    }
    bool hasSameAliasRoot = lhs.aliasRoot == rhs.aliasRoot;
    if (hasSameAliasRoot) {
        if (std::optional<bool> overlap = memoryIntervalsOverlap(
                lhs.semantics.byteOffset, lhs.semantics.byteSize, rhs.semantics.byteOffset, rhs.semantics.byteSize)) {
            return *overlap;
        }
    }
    // Different roots in the same physical space remain conservative: memory
    // planning may have assigned overlapping ranges to distinct SSA roots.
    return true;
}

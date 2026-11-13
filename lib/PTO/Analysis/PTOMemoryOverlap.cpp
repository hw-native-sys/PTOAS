// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Analysis/PTOMemoryAccessAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "llvm/Support/MathExtras.h"

#include <cstdint>

using namespace mlir;
using namespace mlir::pto;

using WideInt = __int128_t;

static bool intervalsOverlap(WideInt aLo, uint64_t aSize, WideInt bLo, uint64_t bSize)
{
    // Use a signed type wider than every input so adding an offset or a size
    // cannot wrap at 64 bits.
    WideInt aEnd = aLo + static_cast<WideInt>(aSize);
    WideInt bEnd = bLo + static_cast<WideInt>(bSize);
    return aLo < bEnd && bLo < aEnd;
}

namespace {

// The absolute start address of a footprint with a resolved absolute base and
// constant byte offset.
static WideInt absoluteStart(const PTOMemoryFootprint& fp)
{
    return static_cast<WideInt>(*fp.absoluteBase) + static_cast<WideInt>(fp.byteOffset.constant);
}

// Both absolute with closed ranges: compare intervals directly.
static std::optional<PTOMemoryAliasResult> aliasAbsoluteClosed(
    const PTOMemoryFootprint& producer, const PTOMemoryFootprint& consumer)
{
    if (producer.absoluteBase && consumer.absoluteBase && producer.byteSize && consumer.byteSize &&
        producer.byteOffset.exact && consumer.byteOffset.exact && producer.byteOffset.isConstant() &&
        consumer.byteOffset.isConstant()) {
        WideInt producerStart = absoluteStart(producer);
        WideInt consumerStart = absoluteStart(consumer);
        if (producerStart >= 0 && consumerStart >= 0 &&
            !intervalsOverlap(producerStart, *producer.byteSize, consumerStart, *consumer.byteSize)) {
            return PTOMemoryAliasResult::NoAlias;
        }
    }
    return std::nullopt;
}

// One side open-ended (byteSize unknown, e.g. vsstb post-update): the open
// side's base B is the lower bound of its write range. The other side's
// closed interval [a, a+s) is disjoint iff it ends at or before B (open side
// is consumer) or starts at/after B+s (open side is producer). Only the
// "ends before B" case is provable since the open upper bound is unknown.
static std::optional<PTOMemoryAliasResult> aliasAbsoluteOpenEnded(
    const PTOMemoryFootprint& producer, const PTOMemoryFootprint& consumer)
{
    if (!producer.absoluteBase || !consumer.absoluteBase || !producer.byteOffset.exact || !consumer.byteOffset.exact ||
        !producer.byteOffset.isConstant() || !consumer.byteOffset.isConstant()) {
        return std::nullopt;
    }
    WideInt prodStart = absoluteStart(producer);
    WideInt consStart = absoluteStart(consumer);
    if (prodStart < 0 || consStart < 0) {
        return std::nullopt;
    }
    // Producer open-ended (e.g. vsstb): consumer [consStart, consStart+s)
    // ends at or before prodStart.
    if (producer.forcesMayAlias && !producer.byteSize && consumer.byteSize &&
        consStart + static_cast<WideInt>(*consumer.byteSize) <= prodStart) {
        return PTOMemoryAliasResult::NoAlias;
    }
    // Consumer open-ended: producer [prodStart, prodStart+s) ends at or
    // before consStart.
    if (consumer.forcesMayAlias && !consumer.byteSize && producer.byteSize &&
        prodStart + static_cast<WideInt>(*producer.byteSize) <= consStart) {
        return PTOMemoryAliasResult::NoAlias;
    }
    return std::nullopt;
}

// Rule 2: gather/scatter forces MayAlias within same space, but only after
// trying to prove the (possibly open-ended) footprints disjoint at the root
// level. Disjoint absolute ranges and distinct proven allocations remain
// provable NoAlias; everything else stays MayAlias.
static PTOMemoryAliasResult aliasForcedMayAlias(const PTOMemoryFootprint& producer, const PTOMemoryFootprint& consumer)
{
    if (producer.rootKind == PTOMemoryRootKind::Absolute && consumer.rootKind == PTOMemoryRootKind::Absolute) {
        if (auto closed = aliasAbsoluteClosed(producer, consumer)) {
            return *closed;
        }
        if (auto open = aliasAbsoluteOpenEnded(producer, consumer)) {
            return *open;
        }
    }
    // Distinct proven allocations cannot alias even with forcesMayAlias.
    if (producer.rootKind == PTOMemoryRootKind::ProvenAllocation &&
        consumer.rootKind == PTOMemoryRootKind::ProvenAllocation && producer.root != consumer.root) {
        return PTOMemoryAliasResult::NoAlias;
    }
    return PTOMemoryAliasResult::MayAlias;
}

// Rule 3: both absolute -> compare absolute intervals.
static PTOMemoryAliasResult aliasBothAbsolute(const PTOMemoryFootprint& producer, const PTOMemoryFootprint& consumer)
{
    if (!producer.absoluteBase || !consumer.absoluteBase || !producer.byteSize || !consumer.byteSize ||
        !producer.byteOffset.exact || !consumer.byteOffset.exact || !producer.byteOffset.isConstant() ||
        !consumer.byteOffset.isConstant()) {
        return PTOMemoryAliasResult::MayAlias;
    }
    WideInt producerStart = absoluteStart(producer);
    WideInt consumerStart = absoluteStart(consumer);
    if (producerStart < 0 || consumerStart < 0) {
        return PTOMemoryAliasResult::MayAlias;
    }
    if (intervalsOverlap(producerStart, *producer.byteSize, consumerStart, *consumer.byteSize)) {
        return PTOMemoryAliasResult::MustOrPartialAlias;
    }
    return PTOMemoryAliasResult::NoAlias;
}

// Rule 5, same root -> compare the root-relative intervals.
static PTOMemoryAliasResult aliasSameRootRelative(
    const PTOMemoryFootprint& producer, const PTOMemoryFootprint& consumer)
{
    if (!producer.byteSize || !consumer.byteSize || !producer.byteOffset.exact || !consumer.byteOffset.exact) {
        return PTOMemoryAliasResult::MayAlias;
    }

    // Equal symbolic terms cancel for same-iteration comparisons only. Use
    // wide subtraction so extreme constant offsets cannot wrap into NoAlias.
    bool equalTerms = true;
    for (auto [value, coefficient] : producer.byteOffset.coefficients) {
        // This helper is also used for sibling-loop comparisons. Their IVs
        // describe independent executions and must stay in the relation model.
        auto arg = dyn_cast<BlockArgument>(value);
        bool loopArgument = arg && isa<scf::ForOp>(arg.getOwner()->getParentOp());
        equalTerms &= !loopArgument && consumer.byteOffset.getCoeff(value) == coefficient;
    }
    for (auto [value, coefficient] : consumer.byteOffset.coefficients) {
        equalTerms &= producer.byteOffset.getCoeff(value) == coefficient;
    }
    if (equalTerms) {
        WideInt pLo = static_cast<WideInt>(producer.byteOffset.constant);
        WideInt cLo = static_cast<WideInt>(consumer.byteOffset.constant);
        if (intervalsOverlap(pLo, *producer.byteSize, cLo, *consumer.byteSize)) {
            return PTOMemoryAliasResult::MustOrPartialAlias;
        }
        return PTOMemoryAliasResult::NoAlias;
    }

    // IV/symbolic offset terms: first version cannot prove disjointness across
    // the iteration domain -> MayAlias. Loop-carried Presburger reasoning lives
    // in the analysis module.
    return PTOMemoryAliasResult::MayAlias;
}

} // namespace

PTOMemoryAliasResult pto::aliasMemorySameIteration(
    const PTOMemoryFootprint& producer, const PTOMemoryFootprint& consumer)
{
    // Rule 1: different memory spaces -> NoAlias.
    if (producer.addressSpace && consumer.addressSpace && *producer.addressSpace != *consumer.addressSpace) {
        return PTOMemoryAliasResult::NoAlias;
    }

    // Rule 2: gather/scatter forces MayAlias within same space, but only after
    // trying to prove the (possibly open-ended) footprints disjoint at the root
    // level. This lets a post-update vsstb whose destination base resolves to
    // an absolute address be proven NoAlias against other absolute buffers that
    // live entirely below its write-range lower bound, and against distinct
    // proven allocations.
    if (producer.forcesMayAlias || consumer.forcesMayAlias) {
        return aliasForcedMayAlias(producer, consumer);
    }

    // Rule 3: both absolute -> compare absolute intervals.
    if (producer.rootKind == PTOMemoryRootKind::Absolute && consumer.rootKind == PTOMemoryRootKind::Absolute) {
        return aliasBothAbsolute(producer, consumer);
    }

    // Absolute vs symbolic/unknown: cannot prove disjoint -> MayAlias.
    if (producer.rootKind == PTOMemoryRootKind::Absolute || consumer.rootKind == PTOMemoryRootKind::Absolute) {
        return PTOMemoryAliasResult::MayAlias;
    }

    // Rule 7: unknown provenance -> MayAlias.
    if (producer.rootKind == PTOMemoryRootKind::Unknown || consumer.rootKind == PTOMemoryRootKind::Unknown) {
        return PTOMemoryAliasResult::MayAlias;
    }

    // Rule 6: different symbolic roots -> MayAlias.
    if (producer.rootKind == PTOMemoryRootKind::ProvenAllocation &&
        consumer.rootKind == PTOMemoryRootKind::ProvenAllocation && producer.root != consumer.root) {
        return PTOMemoryAliasResult::NoAlias;
    }

    // Generic symbolic roots (for example function arguments) may alias even
    // when represented by different SSA values.
    if (producer.root != consumer.root) {
        return PTOMemoryAliasResult::MayAlias;
    }

    // Rule 5: same root -> compare relative intervals.
    return aliasSameRootRelative(producer, consumer);
}

std::optional<bool> pto::memoryIntervalsOverlap(
    std::optional<int64_t> lhsOffset, std::optional<int64_t> lhsSize, std::optional<int64_t> rhsOffset,
    std::optional<int64_t> rhsSize)
{
    if (!lhsOffset || !lhsSize || !rhsOffset || !rhsSize || *lhsSize <= 0 || *rhsSize <= 0) {
        return std::nullopt;
    }
    int64_t lhsEnd;
    int64_t rhsEnd;
    bool overflow = llvm::AddOverflow(*lhsOffset, *lhsSize, lhsEnd) || llvm::AddOverflow(*rhsOffset, *rhsSize, rhsEnd);
    if (overflow) {
        return std::nullopt;
    }
    PTOMemoryFootprint lhs;
    lhs.rootKind = PTOMemoryRootKind::Symbolic;
    lhs.byteOffset.constant = *lhsOffset;
    lhs.byteSize = *lhsSize;
    PTOMemoryFootprint rhs;
    rhs.rootKind = PTOMemoryRootKind::Symbolic;
    rhs.byteOffset.constant = *rhsOffset;
    rhs.byteSize = *rhsSize;
    // Both offsets are already expressed in the caller's common coordinate
    // system. Null roots here denote that shared origin, not pointer evidence.
    auto alias = aliasMemorySameIteration(lhs, rhs);
    if (alias == PTOMemoryAliasResult::MayAlias) {
        return std::nullopt;
    }
    return alias != PTOMemoryAliasResult::NoAlias;
}

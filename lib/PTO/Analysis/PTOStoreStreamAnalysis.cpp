// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Proves the VPTO stream lifecycle independently of target selection.
// No cached attributes or trusted hints bypass the proof.
#include "PTO/Analysis/PTOStoreStreamAnalysis.h"
#include "PTO/Analysis/PTOIntegerRange.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "PTO/IR/VPTOAddressSemantics.h"
#include "PTOStoreStreamRange.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <algorithm>
#include <cstdint>

using namespace mlir;
using namespace mlir::pto;
namespace {

static bool isContinuous(VstusOp store, scf::ForOp loop, int64_t bytes)
{
    if (auto arg = dyn_cast<BlockArgument>(store.getBase())) {
        bool carried = arg.getOwner() == loop.getBody() && arg.getArgNumber() != 0;
        if (carried) {
            auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
            return yield.getOperand(arg.getArgNumber() - 1) == store.getBaseOut();
        }
    }
    auto function = loop->getParentOfType<func::FuncOp>();
    if (!function) {
        return false;
    }
    PTOValueEvolutionAnalysis values(function);
    PTOAddressAnalysis addresses(function, values);
    auto expressions = addresses.getAddresses(store);
    bool singleAddress = expressions && expressions.value->size() == 1;
    if (!singleAddress) {
        return false;
    }
    auto delta = addresses.getDeltaBytes(expressions.value->front(), loop);
    auto constant = delta ? foldPTOConstant(*delta.value) : std::nullopt;
    if (constant) {
        return *constant == bytes;
    }
    // Typed evolution may not know an enclosing IV's range. Certify the
    // complete address before using its affine coefficient as a fallback.
    PTOMemoryAccessDescriptor descriptor;
    descriptor.base = store.getBase();
    StreamDomains domains(loop);
    auto footprint = memoryFootprintFromDescriptor(descriptor, domains.ivs);
    auto step = integer_range::getConstantInt(loop.getStep());
    return step && footprint.byteOffset.exact && !footprint.byteOffset.lossy &&
           addressIsBounded(store.getBase(), loop) &&
           static_cast<__int128>(footprint.byteOffset.getCoeff(loop.getInductionVar())) * *step == bytes;
}

static bool otherAccessesDisjoint(
    scf::ForOp loop, VstusOp store, VstasOp flush, Operation* init, const PTOMemoryFootprint& envelope)
{
    for (Operation& op : loop.getBody()->without_terminator()) {
        if (&op == store || &op == flush || &op == init) {
            continue;
        }
        bool nested = op.getNumRegions() != 0;
        if (nested) {
            return false;
        }
        bool pure = isMemoryEffectFree(&op);
        if (pure) {
            continue;
        }
        // Only ordinary reads may overlap execution of a buffered stream.
        if (!isa<VldsOp, Vldsx2Op>(op)) {
            return false;
        }
        StreamDomains domains(loop);
        auto footprints = getMemoryFootprints(&op, domains.ivs);
        for (const auto& footprint : footprints) {
            auto bounded = boundFootprint(footprint, loop);
            auto closed = bounded ? closeEnclosingDomains(*bounded, loop) : std::nullopt;
            auto stream = closeEnclosingDomains(envelope, loop);
            bool disjoint = closed && stream &&
                            aliasMemorySameIteration(*stream, *closed) == PTOMemoryAliasResult::NoAlias;
            if (!disjoint) {
                return false;
            }
        }
        bool unknown = footprints.empty();
        if (unknown) {
            return false;
        }
    }
    return true;
}

static bool flushImmediatelyAfter(scf::ForOp loop, VstasOp flush)
{
    bool sibling = flush->getBlock() == loop->getBlock();
    if (!sibling) {
        return false;
    }
    for (Operation* op = loop->getNextNode(); op && op != flush; op = op->getNextNode()) {
        if (!isa<arith::ConstantOp>(op)) {
            return false;
        }
    }
    return loop->isBeforeInBlock(flush);
}
static VstusOp findSingleStore(scf::ForOp loop)
{
    VstusOp store;
    for (Operation& op : loop.getBody()->without_terminator()) {
        if (auto candidate = dyn_cast<VstusOp>(op)) {
            if (store) {
                return {};
            }
            store = candidate;
        }
    }
    return store;
}

static std::optional<int64_t> storeBytes(VstusOp store)
{
    auto advance = integer_range::getConstantInt(store.getOffset());
    auto pointer = dyn_cast<PtrType>(store.getBase().getType());
    auto payload = dyn_cast<VRegType>(store.getValue().getType());
    bool sameElement = pointer && payload && pointer.getElementType() == payload.getElementType();
    bool validLength = advance && *advance > 0 && payload && *advance <= payload.getElementCount();
    if (!sameElement || !validLength) {
        return std::nullopt;
    }
    int64_t bytes = *advance * static_cast<int64_t>(getPTOStorageElemByteSize(pointer.getElementType()));
    return bytes > 0 ? std::optional<int64_t>(bytes) : std::nullopt;
}

struct StreamEndpoints {
    Operation* init;
    Value align;
    Value base;
    bool carried;
};

static std::optional<StreamEndpoints> streamEndpoints(VstusOp store, scf::ForOp loop)
{
    auto alignArg = dyn_cast<BlockArgument>(store.getAlignIn());
    if (!alignArg) {
        return StreamEndpoints{
            store.getAlignIn().getDefiningOp<InitAlignOp>(), store.getAlignOut(), store.getBaseOut(), false};
    }
    auto baseArg = dyn_cast<BlockArgument>(store.getBase());
    if (
        alignArg.getOwner() != loop.getBody() || alignArg.getArgNumber() == 0 || !baseArg ||
        baseArg.getOwner() != loop.getBody() || baseArg.getArgNumber() == 0) {
        return std::nullopt;
    }
    unsigned alignIndex = alignArg.getArgNumber() - 1;
    unsigned baseIndex = baseArg.getArgNumber() - 1;
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    bool matchesYield =
        yield.getOperand(alignIndex) == store.getAlignOut() && yield.getOperand(baseIndex) == store.getBaseOut();
    if (!matchesYield) {
        return std::nullopt;
    }
    return StreamEndpoints{
        loop.getInits()[alignIndex].getDefiningOp<InitAlignOp>(), loop.getResult(alignIndex), loop.getResult(baseIndex),
        true};
}

static VstasOp findFlush(const StreamEndpoints& ends, VstusOp store, scf::ForOp loop)
{
    if (
        !ends.init || !ends.init->getResult(0).hasOneUse() || !ends.align.hasOneUse() || !ends.base.hasOneUse()) {
        return {};
    }
    auto flush = dyn_cast<VstasOp>(*ends.align.getUsers().begin());
    auto offset = flush ? integer_range::getConstantInt(flush.getOffset()) : std::nullopt;
    bool compatible = flush && offset && *offset == 0 && !flush.getUpdatedBase() && flush.getDestination() == ends.base;
    if (!compatible) {
        return {};
    }
    bool local = flush->getBlock() == loop.getBody() && store->isBeforeInBlock(flush);
    if (
        ends.carried ? !flushImmediatelyAfter(loop, flush) : !local) {
        return {};
    }
    return flush;
}
} // namespace

std::optional<PTOIsolatedStoreStream> pto::getIsolatedStoreStream(scf::ForOp loop)
{
    if (!loop || !isa<VecScopeOp, func::FuncOp, scf::ForOp, scf::IfOp>(loop->getParentOp())) {
        return std::nullopt;
    }
    auto function = loop->getParentOfType<func::FuncOp>();
    if (!function) {
        return std::nullopt;
    }
    PTOValueEvolutionAnalysis values(function);
    auto evolution = values.getEvolution(loop.getInductionVar(), loop);
    if (!evolution || !evolution.value->noWrap) {
        return std::nullopt;
    }
    VstusOp store = findSingleStore(loop);
    bool linear = store && store.getBaseOut() && store.getBaseOut().hasOneUse() && store.getAlignOut().hasOneUse();
    if (!linear) {
        return std::nullopt;
    }
    auto bytes = storeBytes(store);
    auto ends = streamEndpoints(store, loop);
    if (
        !bytes || !ends || !isContinuous(store, loop, *bytes)) {
        return std::nullopt;
    }
    VstasOp flush = findFlush(*ends, store, loop);
    auto envelope = transactionEnvelope(store, loop, *bytes);
    if (
        !flush || !envelope || !otherAccessesDisjoint(loop, store, flush, ends->init, *envelope)) {
        return std::nullopt;
    }
    return PTOIsolatedStoreStream{store, flush, *envelope};
}

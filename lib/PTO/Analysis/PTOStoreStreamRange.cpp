// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTOStoreStreamRange.h"
#include "PTO/IR/VPTOMemoryDist.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/VPTOAddressSemantics.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include <algorithm>
#include <climits>
using namespace mlir;
using namespace mlir::pto;
namespace {
constexpr int64_t kTransactionBytes = 32;
constexpr unsigned kAddressDepthLimit = 32;
}

// Only address forms whose complete scalar arithmetic is bounded are admitted.
bool pto::addressIsBounded(Value pointer, scf::ForOp loop, unsigned depth)
{
    if (depth == kAddressDepthLimit) {
        return false;
    }
    if (auto add = pointer.getDefiningOp<AddPtrOp>()) {
        StreamDomains domains(loop);
        auto range = integer_range::getAddressRange(add.getOffset(), domains.ivs, domains.bounds);
        return range && addressIsBounded(add.getPtr(), loop, depth + 1);
    }
    if (auto cast = pointer.getDefiningOp<CastPtrOp>()) {
        bool pointerInput = isa<PtrType>(cast.getInput().getType());
        if (pointerInput) {
            return addressIsBounded(cast.getInput(), loop, depth + 1);
        }
        auto constant = integer_range::getConstantInt(cast.getInput());
        return constant && *constant >= 0;
    }
    if (auto arg = dyn_cast<BlockArgument>(pointer)) {
        bool localArgument = arg.getOwner() == loop.getBody();
        if (!localArgument) {
            return loop.isDefinedOutsideOfLoop(pointer);
        }
        bool induction = arg.getArgNumber() == 0;
        if (induction) {
            return false;
        }
        unsigned index = arg.getArgNumber() - 1;
        auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
        Operation* producer = yield.getOperand(index).getDefiningOp();
        if (!producer) {
            return false;
        }
        auto semantics = getDefaultVPTOAddressSemantics(producer);
        auto update = semantics.postUpdate;
        auto amount = update && update->advanceOperand ? integer_range::getConstantInt(update->advanceOperand->get()) :
                                                         std::nullopt;
        return update && update->baseOperand->get() == pointer && amount && *amount > 0 &&
               addressIsBounded(loop.getInits()[index], loop, depth + 1);
    }
    return false;
}

// Collapse the fixed inner loop, retaining certified enclosing IV terms.
std::optional<PTOMemoryFootprint> pto::boundFootprint(PTOMemoryFootprint footprint, scf::ForOp loop)
{
    auto bounds = integer_range::getConstBounds(loop);
    auto count = bounds.exactTripCount();
    if (!count || *count <= 0 || !footprint.byteOffset.exact || footprint.byteOffset.lossy ||
        footprint.forcesMayAlias || !footprint.byteSize || !addressIsBounded(footprint.accessBase, loop)) {
        return std::nullopt;
    }
    StreamDomains domains(loop);
    if (footprint.accessOffset &&
        !integer_range::getAddressRange(footprint.accessOffset, domains.ivs, domains.bounds)) {
        return std::nullopt;
    }
    __int128 lo = footprint.byteOffset.constant;
    __int128 hi = lo;
    SmallVector<std::pair<Value, int64_t>, kValue2> retained;
    for (auto [iv, coefficient] : footprint.byteOffset.coefficients) {
        if (iv != loop.getInductionVar()) {
            if (!integer_range::getAddressRange(iv, domains.ivs, domains.bounds)) {
                return std::nullopt;
            }
            retained.emplace_back(iv, coefficient);
            continue;
        }
        __int128 first = static_cast<__int128>(bounds.lb) * coefficient;
        __int128 last =
            (static_cast<__int128>(bounds.lb) + (*count - 1) * static_cast<__int128>(bounds.step)) * coefficient;
        lo += std::min(first, last);
        hi += std::max(first, last);
    }
    hi += *footprint.byteSize;
    if (lo < INT64_MIN || hi > INT64_MAX || hi - lo > INT64_MAX) {
        return std::nullopt;
    }
    footprint.byteOffset = {};
    footprint.byteOffset.constant = static_cast<int64_t>(lo);
    footprint.byteOffset.coefficients = std::move(retained);
    footprint.byteSize = static_cast<uint64_t>(hi - lo);
    // The scalar domain check above certifies the collapsed stream offsets.
    footprint.offsetDomainCovered = true;
    return footprint;
}

// A valid ordinary load carries the dist's documented alignment contract.
// If its root-relative displacement is block-aligned, the shared root is too.
static bool hasBlockAlignedRoot(const PTOMemoryFootprint& footprint, scf::ForOp loop, VstusOp store)
{
    if (footprint.absoluteBase) {
        return *footprint.absoluteBase % kTransactionBytes == 0;
    }
    bool aligned = false;
    StreamDomains domains(loop);
    auto function = loop->getParentOfType<func::FuncOp>();
    function.walk([&](VldsOp load) {
        if (
            load->getBlock() != loop.getBody() && load->getBlock() != loop->getBlock()) {
            return;
        }
        Operation* streamAnchor = load->getBlock() == loop.getBody() ? store.getOperation() : loop.getOperation();
        if (!load->isBeforeInBlock(streamAnchor)) {
            return;
        }
        auto pointer = dyn_cast<PtrType>(load.getSource().getType());
        auto vector = dyn_cast<VRegType>(load.getResult().getType());
        if (!pointer || !vector || !pointer.getElementType().isIntOrFloat()) {
            return;
        }
        auto contract = lookupVPTOMemoryDist(
            VPTOMemoryOpFamily::Load, load.getDist().value_or("NORM"),
            pointer.getElementType().getIntOrFloatBitWidth());
        int64_t vectorBytes = vector.getElementCount() * getPTOStorageElemByteSize(vector.getElementType());
        auto alignment = contract ? contract->getRequiredAlignmentBytes(vectorBytes) : std::nullopt;
        if (!alignment || *alignment < kTransactionBytes || *alignment % kTransactionBytes != 0) {
            return;
        }
        auto accesses = getMemoryFootprints(load, domains.ivs);
        if (
            accesses.size() != 1) {
            return;
        }
        const auto& access = accesses.front();
        bool sameRoot = access.root == footprint.root && access.rootKind == footprint.rootKind &&
                        access.byteOffset.exact && !access.byteOffset.lossy;
        bool multiples = access.byteOffset.constant % kTransactionBytes == 0 &&
                         llvm::all_of(access.byteOffset.coefficients, [](const auto& term) {
                             return term.second % kTransactionBytes == 0;
                         });
        if (sameRoot && multiples) {
            aligned = true;
        }
    });
    return aligned;
}

std::optional<PTOMemoryFootprint> pto::transactionEnvelope(VstusOp store, scf::ForOp loop, int64_t bytes)
{
    PTOMemoryAccessDescriptor descriptor;
    descriptor.kind = PTOMemoryAccessKind::Store;
    descriptor.base = store.getBase();
    bool carried = false;
    if (auto argument = dyn_cast<BlockArgument>(store.getBase())) {
        carried = argument.getOwner() == loop.getBody() && argument.getArgNumber() != 0;
        if (carried) {
            descriptor.base = loop.getInits()[argument.getArgNumber() - 1];
        }
    }
    descriptor.addressSpace = AddressSpace::VEC;
    descriptor.conservativeByteSize = bytes;
    StreamDomains domains(loop);
    auto footprint = boundFootprint(memoryFootprintFromDescriptor(descriptor, domains.ivs), loop);
    if (!footprint) {
        return std::nullopt;
    }
    if (carried) {
        auto count = integer_range::getConstBounds(loop).exactTripCount();
        if (!count || *count <= 0) {
            return std::nullopt;
        }
        __int128 span = static_cast<__int128>(*footprint->byteSize) +
                        static_cast<__int128>(*count - 1) * bytes;
        if (span > INT64_MAX) {
            return std::nullopt;
        }
        footprint->byteSize = static_cast<uint64_t>(span);
    }
    // Cover any partial 32-byte blocks touched by the complete stream,
    // including a tail whose bytes are committed only by the final flush.
    bool aligned = hasBlockAlignedRoot(*footprint, loop, store) &&
                   footprint->byteOffset.constant >= 0 &&
                   llvm::all_of(footprint->byteOffset.coefficients, [](const auto& term) {
                       return term.second % kTransactionBytes == 0;
                   });
    __int128 lo = static_cast<__int128>(footprint->byteOffset.constant) - (kTransactionBytes - 1);
    __int128 size = static_cast<__int128>(*footprint->byteSize) + 2 * (kTransactionBytes - 1);
    if (aligned) {
        lo = footprint->byteOffset.constant / kTransactionBytes * kTransactionBytes;
        __int128 end = static_cast<__int128>(footprint->byteOffset.constant) + *footprint->byteSize;
        end = (end + kTransactionBytes - 1) / kTransactionBytes * kTransactionBytes;
        size = end - lo;
    }
    if (lo < INT64_MIN || size > INT64_MAX || lo + size > INT64_MAX) {
        return std::nullopt;
    }
    footprint->byteOffset.constant = static_cast<int64_t>(lo);
    footprint->byteSize = static_cast<uint64_t>(size);
    return footprint;
}

std::optional<PTOMemoryFootprint> pto::closeEnclosingDomains(PTOMemoryFootprint footprint, scf::ForOp loop)
{
    StreamDomains domains(loop);
    __int128 lo = footprint.byteOffset.constant;
    __int128 hi = lo;
    for (auto [iv, coefficient] : footprint.byteOffset.coefficients) {
        auto range = integer_range::getAddressRange(iv, domains.ivs, domains.bounds);
        if (!range) {
            return std::nullopt;
        }
        __int128 first = static_cast<__int128>(range->lo) * coefficient;
        __int128 last = static_cast<__int128>(range->hi) * coefficient;
        lo += std::min(first, last);
        hi += std::max(first, last);
        if (lo < INT64_MIN || hi > INT64_MAX) {
            return std::nullopt;
        }
    }
    hi += *footprint.byteSize;
    if (hi > INT64_MAX || hi - lo > INT64_MAX) {
        return std::nullopt;
    }
    footprint.byteOffset = {};
    footprint.byteOffset.constant = static_cast<int64_t>(lo);
    footprint.byteSize = static_cast<uint64_t>(hi - lo);
    return footprint;
}

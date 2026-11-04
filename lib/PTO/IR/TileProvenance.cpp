// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/TileProvenance.h"
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;

Value pto::getTileViewSource(Value tile)
{
    if (!tile || !isa<TileBufType>(tile.getType())) {
        return {};
    }
    if (auto view = dyn_cast_or_null<ViewLikeOpInterface>(tile.getDefiningOp())) {
        return view.getViewSource();
    }
    return {};
}

static Value getYieldSource(Region &region, unsigned index)
{
    if (!region.hasOneBlock() || region.front().empty()) {
        return {};
    }
    auto yield = dyn_cast<scf::YieldOp>(region.front().getTerminator());
    return yield && index < yield.getNumOperands() ? yield.getOperand(index) : Value();
}

static Value getWhileConditionSource(scf::WhileOp loop, unsigned index)
{
    Region &region = loop.getBefore();
    if (!region.hasOneBlock() || region.front().empty()) {
        return {};
    }
    auto condition = dyn_cast<scf::ConditionOp>(region.front().getTerminator());
    return condition && index < condition.getArgs().size() ? condition.getArgs()[index] : Value();
}

static pto::TileSources getLoopSources(scf::ForOp loop, unsigned index)
{
    Value yielded = getYieldSource(loop.getRegion(), index);
    if (!yielded || index >= loop.getInitArgs().size()) {
        return {};
    }
    return {pto::TileSourceKind::Loop,
            {{loop.getInitArgs()[index], {}, true},
             {yielded, {}, true}}};
}

static pto::TileSources getWhileBeforeSources(scf::WhileOp loop, unsigned index)
{
    Value yielded = getYieldSource(loop.getAfter(), index);
    if (!yielded || index >= loop.getInits().size()) {
        return {};
    }
    return {pto::TileSourceKind::Loop,
            {{loop.getInits()[index], {}, true}, {yielded, {}, true}}};
}

pto::TileSources pto::getTileSources(Value tile)
{
    if (!tile || !isa<TileBufType>(tile.getType())) {
        return {};
    }
    if (Value source = getTileViewSource(tile)) {
        return {TileSourceKind::View, {{source, {}, true}}};
    }
    if (auto argument = dyn_cast<BlockArgument>(tile)) {
        auto loop = dyn_cast_or_null<scf::ForOp>(argument.getOwner()->getParentOp());
        if (loop && argument.getArgNumber() > 0) {
            return getLoopSources(loop, argument.getArgNumber() - 1);
        }
        if (auto whileOp = dyn_cast_or_null<scf::WhileOp>(argument.getOwner()->getParentOp())) {
            unsigned index = argument.getArgNumber();
            if (argument.getParentRegion() == &whileOp.getBefore()) {
                return getWhileBeforeSources(whileOp, index);
            }
            return {TileSourceKind::Loop, {{getWhileConditionSource(whileOp, index), {}, true}}};
        }
        return {};
    }
    auto result = cast<OpResult>(tile);
    Operation *op = result.getOwner();
    unsigned index = result.getResultNumber();
    // Frontend pops bind FIFO-backed local descriptors before pipe lowering
    // materializes them as declare_tile plus TPOP (issue #1724).
    if (isa<AllocTileOp, DeclareTileOp, TPopFromAicOp, TPopFromAivOp>(op)) {
        return {TileSourceKind::LocalRoot, {}};
    }
    if (auto selection = dyn_cast<scf::IfOp>(op)) {
        Value thenSource = getYieldSource(selection.getThenRegion(), index);
        Value elseSource = getYieldSource(selection.getElseRegion(), index);
        if (!thenSource || !elseSource) {
            return {};
        }
        return {TileSourceKind::Selection,
                {{thenSource, selection.getCondition(), true},
                 {elseSource, selection.getCondition(), false}}};
    }
    if (auto selection = dyn_cast<arith::SelectOp>(op)) {
        return {TileSourceKind::Selection,
                {{selection.getTrueValue(), selection.getCondition(), true},
                 {selection.getFalseValue(), selection.getCondition(), false}}};
    }
    if (auto loop = dyn_cast<scf::ForOp>(op)) {
        return getLoopSources(loop, index);
    }
    if (auto loop = dyn_cast<scf::WhileOp>(op)) {
        return {TileSourceKind::Loop, {{getWhileConditionSource(loop, index), {}, true}}};
    }
    return {};
}

pto::TileOrigins pto::getTileOrigins(Value tile, bool followLoops)
{
    TileOrigins origins;
    SmallVector<Value> pending{tile};
    llvm::DenseSet<Value> visited;
    while (!pending.empty()) {
        Value current = pending.pop_back_val();
        if (!visited.insert(current).second) {
            continue;
        }
        TileSources sources = getTileSources(current);
        if (sources.kind == TileSourceKind::LocalRoot) {
            origins.roots.push_back(current);
        } else if (sources.kind == TileSourceKind::Unknown ||
                   (sources.kind == TileSourceKind::Loop && !followLoops)) {
            origins.complete = false;
        } else {
            for (const TileSourceEdge &edge : sources.edges) {
                pending.push_back(edge.source);
            }
        }
    }
    origins.complete &= !origins.roots.empty();
    return origins;
}

bool pto::isLocallyBoundTile(Value tile)
{
    // Mutation of loop-carried descriptors requires a separate backend contract.
    return getTileOrigins(tile, /*followLoops=*/false).complete;
}

namespace {
struct TileDescriptorResource final : SideEffects::Resource::Base<TileDescriptorResource> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileDescriptorResource)
    StringRef getName() final { return "pto.tile_descriptor"; }
};

using Effects = SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>;

void addDescriptorEffects(OpOperand &source, Value result, Effects &effects)
{
    auto type = dyn_cast<pto::TileBufType>(result.getType());
    if (!type) {
        return; // Scalar/vector bitcasts remain pure.
    }
    effects.emplace_back(MemoryEffects::Read::get(), &source);
    // Static valid extents do not make the address immutable: TPOP may
    // rebind a tile view. Preserve descriptor identity for every tile type.
    effects.emplace_back(MemoryEffects::Allocate::get(), cast<OpResult>(result),
                         TileDescriptorResource::get());
}
} // namespace

void pto::TReshapeOp::getEffects(Effects &effects)
{
    addDescriptorEffects(getSrcMutable(), getResult(), effects);
}

void pto::BitcastOp::getEffects(Effects &effects)
{
    addDescriptorEffects(getSrcMutable(), getResult(), effects);
}

Speculation::Speculatability pto::BitcastOp::getSpeculatability()
{
    return isa<TileBufType>(getResult().getType()) ? Speculation::NotSpeculatable
                                                 : Speculation::Speculatable;
}

void pto::SubViewOp::getEffects(Effects &effects)
{
    addDescriptorEffects(getSourceMutable(), getResult(), effects);
}

void pto::CopyTileDescriptorOp::getEffects(Effects &effects)
{
    addDescriptorEffects(getSourceMutable(), getResult(), effects);
}

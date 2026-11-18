// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/InsertSync/DeduplicateSlotSync.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <vector>

using namespace mlir;
using namespace mlir::pto;

namespace {

// Exercise protocol metadata directly: several lifetime distinctions have no
// source-IR spelling, so end-to-end tests cannot isolate these rejection cases.
struct Fixture {
    MLIRContext context;
    OwningOpRef<ModuleOp> module;
    Value slot, otherSlot, root, otherRoot;
    std::vector<std::unique_ptr<BaseMemInfo>> memories;
    SyncIRs ir;
    SyncOperations operations;

    Fixture()
    {
        context.loadDialect<PTODialect, func::FuncDialect>();
        module = parseSourceString<ModuleOp>(
            R"mlir(
          module {
            func.func @fixture(%slot: index, %other_slot: index,
                              %root: !pto.tile_buf<mat, 16x16xf16>,
                              %other_root: !pto.tile_buf<mat, 16x16xf16>) {
              return
            }
          }
        )mlir",
            &context);
        if (!module) {
            llvm::errs() << "cannot parse slot dedup unit fixture\n";
            std::exit(2);
        }
        auto function = module->lookupSymbol<func::FuncOp>("fixture");
        slot = function.getArgument(0);
        otherSlot = function.getArgument(1);
        root = function.getArgument(2);
        otherRoot = function.getArgument(3);
        memories.push_back(
            std::make_unique<BaseMemInfo>(root, root, AddressSpace::MAT, SmallVector<uint64_t>{0, 512}, 128, true));
        memories.push_back(
            std::make_unique<BaseMemInfo>(root, root, AddressSpace::MAT, SmallVector<uint64_t>{128, 640}, 128, true));
        memories.push_back(std::make_unique<BaseMemInfo>(
            otherRoot, otherRoot, AddressSpace::MAT, SmallVector<uint64_t>{1024, 1536}, 128, true));

        // Both inner and outer loop records are valid, so changing the loop
        // identity is tested separately from malformed-loop rejection.
        ir.push_back(std::make_unique<LoopInstanceElement>(0, 0, 7));
        ir.push_back(std::make_unique<LoopInstanceElement>(1, 1, 6));
        for (unsigned index = 2; index <= 5; ++index) {
            auto pipe = index < 4 ? PipelineType::PIPE_MTE2 : PipelineType::PIPE_MTE1;
            ir.push_back(std::make_unique<CompoundInstanceElement>(
                index, SmallVector<const BaseMemInfo*>{}, SmallVector<const BaseMemInfo*>{}, pipe,
                OperationName(index < 4 ? "pto.tload" : "pto.tmov", &context)));
        }
        ir.push_back(std::make_unique<LoopInstanceElement>(6, 1, 6, KindOfLoop::LOOP_END));
        ir.push_back(std::make_unique<LoopInstanceElement>(7, 0, 7, KindOfLoop::LOOP_END));
        addPair(0);
        addPair(1);
    }

    void addPair(unsigned memory, std::optional<int> loop = 6)
    {
        unsigned index = operations.size();
        auto set = std::make_unique<SyncOperation>(
            SyncOperation::TYPE::SET_EVENT, PipelineType::PIPE_MTE1, PipelineType::PIPE_MTE2, index, 4, loop);
        set->slotSSAExpr = slot;
        set->slotCount = 2;
        set->eventIdNum = 2;
        set->depRootBuffers.push_back(root);
        set->depMemInfos.push_back(memories[memory].get());
        set->SetDepSyncIRIndex(4 + memory);
        auto wait = set->GetMatchSync(2);
        ir[4]->pipeAfter.push_back(set.get());
        ir[2]->pipeBefore.push_back(wait.get());
        SmallVector<std::unique_ptr<SyncOperation>> pair;
        pair.push_back(std::move(set));
        pair.push_back(std::move(wait));
        operations.push_back(std::move(pair));
    }

    void addShiftedPair(std::optional<int> loop = 7)
    {
        // Append an unrelated, nonduplicate family after the merge candidates.
        // Its allocation can change when any earlier family releases IDs.
        addPair(2, loop);
        for (auto& op : operations.back()) {
            op->depRootBuffers = {otherRoot};
            op->SetDepSyncIRIndex(5);
        }
        operations.back()[1]->slotSSAExpr = otherSlot;
    }

    SyncOperation* set(unsigned pair = 1) { return operations[pair][0].get(); }
    SyncOperation* wait(unsigned pair = 1) { return operations[pair][1].get(); }

    void both(const std::function<void(SyncOperation&)>& mutate)
    {
        mutate(*set());
        mutate(*wait());
    }

    void detach(SyncOperation* sync)
    {
        for (auto& element : ir) {
            for (auto* list : {&element->pipeBefore, &element->pipeAfter})
                list->erase(std::remove(list->begin(), list->end(), sync), list->end());
        }
    }

    void move(SyncOperation* sync, unsigned anchor, bool before)
    {
        detach(sync);
        sync->SetSyncIRIndex(anchor);
        (before ? ir[anchor]->pipeBefore : ir[anchor]->pipeAfter).push_back(sync);
    }

    void rebuildPair(PipelineType source, PipelineType destination, std::optional<int> loop)
    {
        auto replacement =
            std::make_unique<SyncOperation>(SyncOperation::TYPE::SET_EVENT, source, destination, 1, 4, loop);
        replacement->slotSSAExpr = slot;
        replacement->slotCount = replacement->eventIdNum = 2;
        replacement->depRootBuffers = {root};
        replacement->depMemInfos = {memories[1].get()};
        replacement->SetDepSyncIRIndex(5);
        auto replacementWait = replacement->GetMatchSync(2);
        detach(set());
        detach(wait());
        operations[1].clear();
        operations[1].push_back(std::move(replacement));
        operations[1].push_back(std::move(replacementWait));
        ir[4]->pipeAfter.push_back(set());
        ir[2]->pipeBefore.push_back(wait());
    }

    void addBlockingWaitBetweenSets()
    {
        // A valid static forward pair: SET before the source anchor, WAIT
        // interleaved between the otherwise duplicate source SETs.
        unsigned index = operations.size();
        auto set = std::make_unique<SyncOperation>(
            SyncOperation::TYPE::SET_EVENT, PipelineType::PIPE_MTE2, PipelineType::PIPE_MTE1, index, 3, std::nullopt);
        auto wait = set->GetMatchSync(4);
        ir[3]->pipeAfter.push_back(set.get());
        ir[4]->pipeAfter.insert(ir[4]->pipeAfter.begin() + 1, wait.get());
        SmallVector<std::unique_ptr<SyncOperation>> pair;
        pair.push_back(std::move(set));
        pair.push_back(std::move(wait));
        operations.push_back(std::move(pair));
    }

    void addWaitAnchorBarrier()
    {
        auto barrier = std::make_unique<SyncOperation>(
            SyncOperation::TYPE::PIPE_BARRIER, PipelineType::PIPE_MTE2, PipelineType::PIPE_MTE2, operations.size(), 2,
            std::nullopt);
        ir[2]->pipeBefore.insert(ir[2]->pipeBefore.begin() + 1, barrier.get());
        SmallVector<std::unique_ptr<SyncOperation>> group;
        group.push_back(std::move(barrier));
        operations.push_back(std::move(group));
    }

    std::vector<const SyncOperation*> attachments() const
    {
        std::vector<const SyncOperation*> result;
        for (const auto& element : ir) {
            result.insert(result.end(), element->pipeBefore.begin(), element->pipeBefore.end());
            result.insert(result.end(), element->pipeAfter.begin(), element->pipeAfter.end());
        }
        return result;
    }
};

unsigned failures = 0;
unsigned cases = 0;

void check(bool condition, StringRef name, StringRef message)
{
    if (!condition) {
        ++failures;
        llvm::errs() << "FAIL: " << name << ": " << message << '\n';
    }
}

void reject(StringRef name, const std::function<void(Fixture&)>& mutate)
{
    ++cases;
    Fixture fixture;
    mutate(fixture);
    auto attachments = fixture.attachments();
    std::vector<bool> useless;
    for (const auto& pair : fixture.operations)
        for (const auto& op : pair)
            useless.push_back(op->uselessSync);
    check(deduplicateSlotEventFamilies(fixture.ir, fixture.operations) == 0, name, "unexpected merge");
    check(fixture.attachments() == attachments, name, "rejected protocol attachments changed");
    unsigned index = 0;
    for (const auto& pair : fixture.operations)
        for (const auto& op : pair)
            check(op->uselessSync == useless[index++], name, "rejected protocol liveness changed");
}

void configureAcceptedFixture(Fixture& fixture, bool masked, bool differentEndpointSelectors, unsigned lanes)
{
    uint32_t mask = (uint32_t{1} << lanes) - 1;
    for (auto& pair : fixture.operations) {
        if (differentEndpointSelectors)
            pair[1]->slotSSAExpr = fixture.otherSlot;
        for (auto& op : pair) {
            op->slotCount = op->eventIdNum = lanes;
            if (masked)
                op->slotBoundaryMasks = SlotEventBoundaryMasks{mask, mask};
        }
    }
}

void checkRetiredFamily(Fixture& fixture, const SyncOperations::value_type& pair, StringRef name)
{
    check(pair[1]->uselessSync, name, "only half of duplicate retired");
    auto attachments = fixture.attachments();
    check(
        std::find(attachments.begin(), attachments.end(), pair[0].get()) == attachments.end() &&
            std::find(attachments.begin(), attachments.end(), pair[1].get()) == attachments.end(),
        name, "retired pair still attached");
}

void checkLiveEndpoints(
    Fixture& fixture, const SyncOperations::value_type& pair, StringRef name, bool masked, unsigned lanes)
{
    uint32_t mask = (uint32_t{1} << lanes) - 1;
    check(!pair[1]->uselessSync, name, "live pair has dead wait");
    for (const auto& op : pair) {
        check(
            op->depMemInfos.size() == 2 && llvm::is_contained(op->depMemInfos, fixture.memories[0].get()) &&
                llvm::is_contained(op->depMemInfos, fixture.memories[1].get()),
            name, "dependency metadata was not unioned on both halves");
        check(op->depRootBuffers.size() == 1 && op->depRootBuffers[0] == fixture.root, name, "closed root changed");
        check(
            op->slotCount == lanes && op->eventIdNum == int(lanes) && op->eventIds.empty(), name,
            "lane policy changed");
        check(op->slotBoundaryMasks.has_value() == masked, name, "boundary policy changed");
        if (masked && op->slotBoundaryMasks)
            check(
                op->slotBoundaryMasks->prime == mask && op->slotBoundaryMasks->drain == mask, name,
                "boundary masks changed");
    }
}

void checkLiveLifetime(
    Fixture& fixture, const SyncOperations::value_type& pair, StringRef name, bool differentEndpointSelectors)
{
    check(
        pair[0]->GetSyncIRIndex() == 4 && pair[1]->GetSyncIRIndex() == 2 && pair[0]->GetForEndIndex() == 6 &&
            pair[1]->GetForEndIndex() == 6,
        name, "event lifetime anchors changed");
    check(
        (pair[0]->GetDepSyncIRIndex() == 4 || pair[0]->GetDepSyncIRIndex() == 5) &&
            pair[0]->GetDepSyncIRIndex() == pair[1]->GetDepSyncIRIndex(),
        name, "representative dependency source was replaced");
    check(
        pair[0]->slotSSAExpr == fixture.slot &&
            pair[1]->slotSSAExpr == (differentEndpointSelectors ? fixture.otherSlot : fixture.slot),
        name, "per-endpoint selector changed");
}

void accept(bool masked, bool differentEndpointSelectors, unsigned lanes = 2)
{
    ++cases;
    StringRef name = lanes == 3 ? "three-lane duplicate" : (masked ? "masked duplicate" : "legacy duplicate");
    Fixture fixture;
    configureAcceptedFixture(fixture, masked, differentEndpointSelectors, lanes);
    check(deduplicateSlotEventFamilies(fixture.ir, fixture.operations) == 1, name, "duplicate not merged");
    unsigned live = 0;
    for (const auto& pair : fixture.operations) {
        if (pair[0]->uselessSync) {
            checkRetiredFamily(fixture, pair, name);
            continue;
        }
        ++live;
        checkLiveEndpoints(fixture, pair, name, masked, lanes);
        checkLiveLifetime(fixture, pair, name, differentEndpointSelectors);
    }
    check(live == 1, name, "expected one complete surviving family");
    check(fixture.attachments().size() == 2, name, "expected one set and one wait");
    check(deduplicateSlotEventFamilies(fixture.ir, fixture.operations) == 0, name, "not idempotent");
}

void setAllBoundaryMasks(Fixture& fixture)
{
    for (auto& pair : fixture.operations)
        for (auto& sync : pair)
            sync->slotBoundaryMasks = SlotEventBoundaryMasks{3, 3};
}

void rejectLifetimeMismatches()
{
    reject("set anchor", [](Fixture& f) { f.move(f.set(), 5, false); });
    reject("wait anchor", [](Fixture& f) { f.move(f.wait(), 3, true); });
    reject("different loop", [](Fixture& f) { f.rebuildPair(PipelineType::PIPE_MTE1, PipelineType::PIPE_MTE2, 7); });
    reject(
        "no loop", [](Fixture& f) { f.rebuildPair(PipelineType::PIPE_MTE1, PipelineType::PIPE_MTE2, std::nullopt); });
    reject("direction", [](Fixture& f) { f.rebuildPair(PipelineType::PIPE_MTE3, PipelineType::PIPE_MTE2, 6); });
    reject("set selector", [](Fixture& f) {
        setAllBoundaryMasks(f);
        f.set()->slotSSAExpr = f.otherSlot;
    });
    reject("wait selector", [](Fixture& f) {
        setAllBoundaryMasks(f);
        f.wait()->slotSSAExpr = f.otherSlot;
    });
    reject("lane count", [](Fixture& f) { f.both([](auto& s) { s.slotCount = s.eventIdNum = 3; }); });
    reject("inconsistent lane count", [](Fixture& f) { f.set()->slotCount = 3; });
    reject("inconsistent event count", [](Fixture& f) { f.wait()->eventIdNum = 3; });
    reject("mask presence", [](Fixture& f) {
        f.both([](auto& s) { s.slotBoundaryMasks = SlotEventBoundaryMasks{3, 3}; });
    });
    reject("prime mask", [](Fixture& f) {
        setAllBoundaryMasks(f);
        f.both([](auto& s) { s.slotBoundaryMasks->prime = 1; });
    });
    reject("drain mask", [](Fixture& f) {
        setAllBoundaryMasks(f);
        f.both([](auto& s) { s.slotBoundaryMasks->drain = 1; });
    });
    reject("inconsistent masks", [](Fixture& f) { f.set()->slotBoundaryMasks = SlotEventBoundaryMasks{3, 3}; });
}

void rejectDependencyMismatches()
{
    reject("core", [](Fixture& f) { f.both([](auto& s) { s.syncCoreType = TCoreType::CUBE; }); });
    reject("inconsistent core", [](Fixture& f) { f.wait()->syncCoreType = TCoreType::CUBE; });
    reject("mechanism", [](Fixture& f) { f.both([](auto& s) { s.SetMechanism(SyncOperation::MECHANISM::BUFID); }); });
    reject("different root", [](Fixture& f) {
        f.both([&](auto& s) {
            s.depRootBuffers = {f.otherRoot};
            s.depMemInfos = {f.memories[2].get()};
        });
    });
    reject("mixed roots", [](Fixture& f) { f.both([&](auto& s) { s.depRootBuffers.push_back(f.otherRoot); }); });
    reject("foreign memory info", [](Fixture& f) {
        f.both([&](auto& s) { s.depMemInfos.push_back(f.memories[2].get()); });
    });
    reject("missing memory info", [](Fixture& f) { f.both([](auto& s) { s.depMemInfos.clear(); }); });
}

void rejectSpecialEventStates()
{
    reject("static pair", [](Fixture& f) {
        f.both([](auto& s) {
            s.slotCount = s.eventIdNum = 1;
            s.slotSSAExpr = {};
        });
    });
    reject("compensation", [](Fixture& f) { f.both([](auto& s) { s.isCompensation = true; }); });
    reject("compensation owner", [](Fixture& f) { f.both([](auto& s) { s.compensationOf = 0; }); });
    reject("allocated", [](Fixture& f) { f.both([](auto& s) { s.eventIds = {0, 1}; }); });
    reject("widened", [](Fixture& f) { f.both([](auto& s) { s.reuseCntForWiden = 1; }); });
    reject("reallocated", [](Fixture& f) { f.both([](auto& s) { s.reallocatedLoopHeadTailSync = true; }); });
    reject("widened ancestor", [](Fixture& f) { f.both([&](auto& s) { s.lowestCommonAncestorBuffer = f.root; }); });
    reject("tail barrier", [](Fixture& f) { f.both([](auto& s) { s.autoSyncTailBarrier = true; }); });
    reject("block value", [](Fixture& f) { f.both([&](auto& s) { s.block_sync_event_value = f.slot; }); });
}

void rejectAttachmentMismatches()
{
    reject("useless half", [](Fixture& f) { f.set()->uselessSync = true; });
    reject("wrong type", [](Fixture& f) { f.set()->SetType(SyncOperation::TYPE::WAIT_EVENT); });
    reject("missing attachment", [](Fixture& f) { f.detach(f.wait()); });
    reject("duplicate attachment", [](Fixture& f) { f.ir[4]->pipeAfter.push_back(f.set()); });
    reject("wrong side", [](Fixture& f) { f.move(f.set(), 4, true); });
    reject("attachment index mismatch", [](Fixture& f) { f.set()->SetSyncIRIndex(5); });
    reject("blocking wait between sets", [](Fixture& f) { f.addBlockingWaitBetweenSets(); });
    reject("barrier between waits", [](Fixture& f) { f.addWaitAnchorBarrier(); });
}

void rejectUnprovedRotations()
{
    reject("unmasked shifted duplicates", [](Fixture& f) { configureAcceptedFixture(f, false, true, 2); });
    reject("unrelated root and loop rotation", [](Fixture& f) { f.addShiftedPair(); });
    reject("allocated unrelated rotation", [](Fixture& f) {
        f.addShiftedPair();
        for (auto& op : f.operations.back())
            op->eventIds = {0, 1};
    });
    reject("rotation with only set mask", [](Fixture& f) {
        f.addShiftedPair();
        f.operations.back()[0]->slotBoundaryMasks = SlotEventBoundaryMasks{3, 3};
    });
    reject("rotation with only wait mask", [](Fixture& f) {
        f.addShiftedPair();
        f.operations.back()[1]->slotBoundaryMasks = SlotEventBoundaryMasks{3, 3};
    });
}

void acceptWithUnrelatedPair(bool masked, bool dead, bool forward)
{
    ++cases;
    StringRef name = masked ? "masked unrelated rotation" : (dead ? "dead unrelated rotation" : "forward pair");
    Fixture fixture;
    fixture.addShiftedPair(forward ? std::nullopt : std::optional<int>(7));
    for (auto& op : fixture.operations.back()) {
        op->uselessSync = dead;
        if (masked)
            op->slotBoundaryMasks = SlotEventBoundaryMasks{3, 3};
    }
    check(deduplicateSlotEventFamilies(fixture.ir, fixture.operations) == 1, name, "unrelated pair blocked merge");
    check(!fixture.set(0)->uselessSync && fixture.set()->uselessSync, name, "wrong duplicate retired");
    for (const auto& op : fixture.operations.back())
        check(op->uselessSync == dead, name, "unrelated pair liveness changed");
    check(deduplicateSlotEventFamilies(fixture.ir, fixture.operations) == 0, name, "not idempotent");
}

} // namespace

int main()
{
    accept(false, false);
    accept(true, false);
    accept(true, true);
    accept(true, false, 3);
    rejectLifetimeMismatches();
    rejectDependencyMismatches();
    rejectSpecialEventStates();
    rejectAttachmentMismatches();
    rejectUnprovedRotations();
    acceptWithUnrelatedPair(true, false, false);
    acceptWithUnrelatedPair(false, true, false);
    acceptWithUnrelatedPair(false, false, true);
    if (failures) {
        llvm::errs() << failures << " failure(s) in " << cases << " slot protocol cases\n";
        return 1;
    }
    llvm::outs() << "PASS: " << cases << " slot protocol cases\n";
    return 0;
}

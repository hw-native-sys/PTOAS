// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===----------------------------------------------------------------------===//
// UFTokenModel: structured unit-flag hand-off analysis for issue #1694.
//===----------------------------------------------------------------------===//
//
// The a2/a3/a5 unit flag is a single producer/consumer token per core. A Final matmul
// publishes it (check_and_set) and a Final store consumes it (check_and_clear).
// A phased operation with a Partial phase only observes the token and leaves it
// alone. Proving that a memory dependency is already ordered by this handshake
// needs two structural facts:
//
//   1. The producer's precondition holds: walking back along the same execution
//      path, the nearest prior token action is the opposite one (or the region
//      starts there, which is the pre-existing "no stale phase" contract).
//   2. No other token action sits between the pair (forward for a same-iteration
//      pair, circular through the loop body for a loop-carried pair).
//
// Walking the flat SyncIR index space gets both facts wrong once branches and
// loops are involved: it can walk into a mutually exclusive branch, or out of
// the loop body entirely and "prove" a pairing by reaching unrelated code. This
// file builds a small tree that keeps the nesting, so the backward and forward
// walks stop at the region boundary and wrap only around loop bodies.
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/InsertSync/UFTokenModel.h"

#include "PTO/IR/PTO.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/CallInterfaces.h"

#include <memory>
#include <utility>

using namespace mlir;
using namespace mlir::pto;

namespace mlir::pto {

UFTokenAction getUFTokenAction(const CompoundInstanceElement& compound)
{
    const UFInfo& ufInfo = compound.ufInfo;
    if (ufInfo.role == UFRole::MatmulProducer && ufInfo.accPhase && *ufInfo.accPhase == AccPhase::Final) {
        return UFTokenAction::Publish;
    }
    if (ufInfo.role == UFRole::AccStore && ufInfo.stPhase && *ufInfo.stPhase == STPhase::Final) {
        return UFTokenAction::Consume;
    }
    return UFTokenAction::None;
}

namespace {

UFEffect composeEffect(UFEffect lhs, UFEffect rhs)
{
    unsigned result = 0;
    for (unsigned input = 0; input < 2; ++input) {
        for (unsigned middle = 0; middle < 2; ++middle) {
            for (unsigned output = 0; output < 2; ++output) {
                const bool first = (static_cast<unsigned>(lhs) & (1U << (input * 2 + middle))) != 0;
                const bool second = (static_cast<unsigned>(rhs) & (1U << (middle * 2 + output))) != 0;
                if (first && second) {
                    result |= 1U << (input * 2 + output);
                }
            }
        }
    }
    return static_cast<UFEffect>(result);
}

bool isNeutralEffect(UFEffect effect)
{
    return effect == UFEffect::Neutral || effect == UFEffect::Clear || effect == UFEffect::Set;
}

// Every executable arm must accept the input and agree on its output. An
// absent else is identity, so an optional balanced pair retains its precondition.
UFEffect mergeEffects(UFEffect lhs, UFEffect rhs)
{
    return static_cast<UFEffect>(static_cast<unsigned>(lhs) & static_cast<unsigned>(rhs));
}

UFEffect effectOfAction(UFTokenAction action)
{
    switch (action) {
        case UFTokenAction::Publish:
            return UFEffect::Publish;
        case UFTokenAction::Consume:
            return UFEffect::Consume;
        case UFTokenAction::None:
            return UFEffect::Neutral;
    }
    return UFEffect::Unknown;
}

bool sameAccAccess(const BaseMemInfo* lhs, const BaseMemInfo* rhs)
{
    return lhs && rhs && lhs->scope == AddressSpace::ACC && rhs->scope == AddressSpace::ACC &&
           lhs->rootBuffer == rhs->rootBuffer;
}

bool accRegionsOverlap(const UFInfo& lhs, const UFInfo& rhs)
{
    return llvm::any_of(lhs.accAccesses, [&rhs](const BaseMemInfo* access) {
        return llvm::any_of(
            rhs.accAccesses, [access](const BaseMemInfo* other) { return sameAccAccess(access, other); });
    });
}

} // namespace

UFTokenModel::~UFTokenModel() = default;

void UFTokenModel::build()
{
    if (built_) {
        return;
    }
    roots_.clear();
    buildRange(0, static_cast<unsigned>(syncIR_.size()), roots_);

    auto rootSequence = std::make_unique<Sequence>();
    rootSequence->nodes = &roots_;
    Sequence* rootPtr = rootSequence.get();
    sequences_.push_back(std::move(rootSequence));

    assignParents(roots_, rootPtr);
    computeSequenceEffect(roots_);
    computeTokenFree(roots_);
    collectLeafInfo(roots_, rootPtr);
    built_ = true;
}

void UFTokenModel::buildRange(unsigned begin, unsigned end, std::vector<Node>& out)
{
    for (unsigned i = begin; i < end; ++i) {
        InstanceElement* element = syncIR_[i].get();
        const bool opaqueCall = element->elementOp && isa<CallOpInterface>(element->elementOp);
        const bool compoundOrCall = isa<CompoundInstanceElement>(element) || opaqueCall;
        if (compoundOrCall) {
            Node node;
            node.kind = Node::Kind::Compound;
            node.index = element->GetIndex();
            out.push_back(std::move(node));
            continue;
        }
        if (auto* loop = dyn_cast<LoopInstanceElement>(element)) {
            const bool malformedLoop =
                loop->getLoopKind() != KindOfLoop::LOOP_BEGIN || loop->endId >= syncIR_.size() || loop->endId < i;
            if (malformedLoop) {
                continue;
            }
            Node node;
            node.kind = Node::Kind::Loop;
            node.index = i;
            node.endIndex = loop->endId;
            buildRange(i + 1, loop->endId, node.body);
            out.push_back(std::move(node));
            i = loop->endId;
            continue;
        }
        if (auto* branch = dyn_cast<BranchInstanceElement>(element)) {
            const bool malformedBranch = branch->getBranchKind() != KindOfBranch::IF_BEGIN ||
                                         branch->endId >= syncIR_.size() || branch->branchId > branch->endId;
            if (malformedBranch) {
                continue;
            }
            Node node;
            node.kind = Node::Kind::If;
            node.index = i;
            node.endIndex = branch->endId;
            buildRange(i + 1, branch->branchId, node.body);
            if (branch->branchId + 1 <= branch->endId) {
                buildRange(branch->branchId + 1, branch->endId, node.elseBody);
            }
            out.push_back(std::move(node));
            i = branch->endId;
            continue;
        }
        // Placeholders, ELSE_BEGIN and IF_END markers carry no token state.
    }
}

void UFTokenModel::assignParents(std::vector<Node>& nodes, const Sequence* sequence)
{
    for (unsigned i = 0; i < nodes.size(); ++i) {
        Node& node = nodes[i];
        node.parentSequence = sequence;
        node.childIndex = i;

        if (node.kind == Node::Kind::Loop) {
            auto bodySequence = std::make_unique<Sequence>();
            bodySequence->nodes = &node.body;
            bodySequence->owner = &node;
            Sequence* bodyPtr = bodySequence.get();
            sequences_.push_back(std::move(bodySequence));
            node.bodySequence = bodyPtr;
            assignParents(node.body, bodyPtr);
        } else if (node.kind == Node::Kind::If) {
            auto thenSequence = std::make_unique<Sequence>();
            thenSequence->nodes = &node.body;
            thenSequence->owner = &node;
            Sequence* thenPtr = thenSequence.get();
            sequences_.push_back(std::move(thenSequence));
            node.bodySequence = thenPtr;

            auto elseSequence = std::make_unique<Sequence>();
            elseSequence->nodes = &node.elseBody;
            elseSequence->owner = &node;
            Sequence* elsePtr = elseSequence.get();
            sequences_.push_back(std::move(elseSequence));
            node.elseSequence = elsePtr;

            assignParents(node.body, thenPtr);
            assignParents(node.elseBody, elsePtr);
        }
    }
}

UFEffect UFTokenModel::computeSequenceEffect(std::vector<Node>& nodes)
{
    UFEffect effect = UFEffect::Neutral;
    for (Node& node : nodes) {
        effect = composeEffect(effect, computeNodeEffect(node));
    }
    return effect;
}

UFEffect UFTokenModel::computeNodeEffect(Node& node)
{
    switch (node.kind) {
        case Node::Kind::Compound:
            node.effect = isa_and_nonnull<CallOpInterface>(opOfNode(node))
                              ? UFEffect::Unknown : effectOfAction(actionOf(node.index));
            return node.effect;
        case Node::Kind::Loop: {
            if (auto whileOp = dyn_cast_or_null<scf::WhileOp>(opOfNode(node))) {
                return computeWhileEffect(node, whileOp);
            }
            node.bodyEffect = computeSequenceEffect(node.body);
            // A loop with a non-neutral body leaves the token in a different state on
            // every iteration. Its net effect is therefore not representable, even if
            // the last iteration happens to balance out.
            node.effect = isNeutralEffect(node.bodyEffect) ? node.bodyEffect : UFEffect::Unknown;
            loopBalanced_[node.endIndex] = isNeutralEffect(node.bodyEffect);
            return node.effect;
        }
        case Node::Kind::If:
            node.bodyEffect = computeSequenceEffect(node.body);
            node.elseEffect = computeSequenceEffect(node.elseBody);
            node.effect = mergeEffects(node.bodyEffect, node.elseEffect);
            ifEffect_[node.index] = node.effect;
            return node.effect;
    }
    return UFEffect::Unknown;
}

Operation* UFTokenModel::opOfNode(const Node& node) const
{
    if (node.index >= syncIR_.size()) {
        return nullptr;
    }
    return syncIR_[node.index]->elementOp;
}

UFEffect UFTokenModel::computeWhileEffect(Node& node, Operation* whileOp)
{
    auto whileBody = cast<scf::WhileOp>(whileOp);
    UFEffect before = UFEffect::Neutral;
    UFEffect after = UFEffect::Neutral;
    bool valid = true;
    for (Node& child : node.body) {
        Operation* childOp = opOfNode(child);
        if (!childOp) {
            valid = false;
            break;
        }
        const UFEffect childEffect = computeNodeEffect(child);
        Region* region = childOp->getParentRegion();
        if (region == &whileBody.getBefore()) {
            before = composeEffect(before, childEffect);
        } else if (region == &whileBody.getAfter()) {
            after = composeEffect(after, childEffect);
        } else {
            valid = false;
            break;
        }
    }
    // scf.while exits from the before region. Requiring both halves to be
    // neutral with compatible preconditions makes the exit state equal the
    // entry state and keeps every iteration's token traffic balanced. A composite-neutral but individually
    // non-neutral split (publish in before, consume in after) would leave the
    // flag set when the loop exits.
    const bool balanced = valid && isNeutralEffect(before) && isNeutralEffect(after) &&
                          isNeutralEffect(composeEffect(before, after));
    node.bodyEffect = composeEffect(before, after);
    node.effect = balanced ? node.bodyEffect : UFEffect::Unknown;
    loopBalanced_[node.endIndex] = balanced;
    return node.effect;
}

void UFTokenModel::computeTokenFree(std::vector<Node>& nodes)
{
    for (Node& node : nodes) {
        if (node.kind == Node::Kind::Compound) {
            node.tokenFree = node.effect == UFEffect::Neutral;
            continue;
        }
        if (node.kind == Node::Kind::Loop) {
            computeTokenFree(node.body);
            node.tokenFree = llvm::all_of(node.body, [](const Node& child) { return child.tokenFree; });
            continue;
        }
        computeTokenFree(node.body);
        computeTokenFree(node.elseBody);
        node.tokenFree = llvm::all_of(node.body, [](const Node& child) { return child.tokenFree; }) &&
                         llvm::all_of(node.elseBody, [](const Node& child) { return child.tokenFree; });
    }
}

void UFTokenModel::collectLeafInfo(const std::vector<Node>& nodes, const Sequence* sequence)
{
    for (const Node& node : nodes) {
        switch (node.kind) {
            case Node::Kind::Compound:
                leaves_[node.index] = &node;
                break;
            case Node::Kind::Loop:
                collectLeafInfo(node.body, node.bodySequence);
                break;
            case Node::Kind::If:
                collectLeafInfo(node.body, node.bodySequence);
                collectLeafInfo(node.elseBody, node.elseSequence);
                break;
        }
    }
}

UFTokenAction UFTokenModel::actionOf(unsigned compoundIndex) const
{
    if (compoundIndex >= syncIR_.size()) {
        return UFTokenAction::None;
    }
    auto* compound = dyn_cast_or_null<CompoundInstanceElement>(syncIR_[compoundIndex].get());
    return compound ? getUFTokenAction(*compound) : UFTokenAction::None;
}

const UFTokenModel::Node* UFTokenModel::findLeaf(unsigned compoundIndex) const
{
    auto it = leaves_.find(compoundIndex);
    return it == leaves_.end() ? nullptr : it->second;
}

bool UFTokenModel::sameRegion(const Node& lhs, const Node& rhs) const
{
    return lhs.parentSequence && lhs.parentSequence == rhs.parentSequence;
}

bool UFTokenModel::pairRegionUsable(const Node& lhs, const Node& rhs) const
{
    if (!sameRegion(lhs, rhs)) {
        return false;
    }
    const Sequence* sequence = lhs.parentSequence;
    const Node* owner = sequence->owner;
    if (owner && owner->kind == Node::Kind::Loop) {
        auto balanced = loopBalanced_.find(owner->endIndex);
        const bool balancedBody = balanced != loopBalanced_.end() && balanced->second;
        if (!balancedBody) {
            return false;
        }
    }
    // A failed branch merge means the two paths leave the token in different
    // states. No pair inside either branch may then be treated as a complete
    // hand-off: the test matrix requires `if { publish } else { publish; consume }`
    // to keep every event. Walk every ancestor If, not just the directly
    // enclosing one, because the pair is inside all of them.
    const Node* current = owner;
    while (current) {
        if (current->kind == Node::Kind::Loop && !isLoopBodyBalanced(current->endIndex)) {
            return false;
        }
        if (current->kind == Node::Kind::Loop && current->bodyEffect == UFEffect::Set &&
            !loopEntryIsSet(*current)) {
            return false;
        }
        if (current->kind == Node::Kind::If) {
            auto effect = ifEffect_.find(current->index);
            const bool unknownEffect = effect == ifEffect_.end() || effect->second == UFEffect::Unknown;
            if (unknownEffect) {
                return false;
            }
        }
        const Sequence* parentSequence = current->parentSequence;
        current = parentSequence ? parentSequence->owner : nullptr;
    }
    return true;
}

bool UFTokenModel::forwardSpanTokenFree(const Node& front, const Node& now) const
{
    if (!sameRegion(front, now)) {
        return false;
    }
    const Sequence* sequence = front.parentSequence;
    if (front.childIndex >= now.childIndex) {
        return false;
    }
    const std::vector<Node>& nodes = *sequence->nodes;
    for (unsigned i = front.childIndex + 1; i < now.childIndex; ++i) {
        if (!nodes[i].tokenFree) {
            return false;
        }
    }
    return true;
}

// Follow only the leaf's execution path up to the carrying loop body.
bool UFTokenModel::collectLoopChain(
    unsigned loopEndIndex, const Node& leaf,
    SmallVectorImpl<std::pair<const Sequence*, unsigned>>& chain) const
{
    chain.clear();
    const Sequence* sequence = leaf.parentSequence;
    unsigned childIndex = leaf.childIndex;
    while (sequence) {
        chain.push_back({sequence, childIndex});
        const Node* owner = sequence->owner;
        if (owner && owner->kind == Node::Kind::Loop && owner->endIndex == loopEndIndex) {
            return true;
        }
        if (!owner) {
            return false;
        }
        childIndex = owner->childIndex;
        sequence = owner->parentSequence;
    }
    return false;
}

bool UFTokenModel::spanNodesTokenFree(const std::vector<Node>& nodes, unsigned begin, size_t end) const
{
    for (size_t i = begin; i < end; ++i) {
        if (!nodes[i].tokenFree) {
            return false;
        }
    }
    return true;
}

bool UFTokenModel::circularSpanTokenFree(unsigned loopEndIndex, const Node& front, const Node& now) const
{
    if (!sameRegion(front, now)) {
        return false;
    }
    SmallVector<std::pair<const Sequence*, unsigned>, 8> frontChain;
    SmallVector<std::pair<const Sequence*, unsigned>, 8> nowChain;
    if (!collectLoopChain(loopEndIndex, front, frontChain)) {
        return false;
    }
    if (!collectLoopChain(loopEndIndex, now, nowChain)) {
        return false;
    }
    // Tail of this iteration, then head of the next, along the selected path.
    for (const auto& entry : frontChain) {
        const std::vector<Node>& nodes = *entry.first->nodes;
        if (!spanNodesTokenFree(nodes, entry.second + 1, nodes.size())) {
            return false;
        }
    }
    for (const auto& entry : nowChain) {
        if (!spanNodesTokenFree(*entry.first->nodes, 0, entry.second)) {
            return false;
        }
    }
    return true;
}

std::optional<UFEffect> UFTokenModel::priorTokenEffectBefore(const Node& compound) const
{
    const Sequence* sequence = compound.parentSequence;
    unsigned childIndex = compound.childIndex;
    while (sequence) {
        const std::vector<Node>& nodes = *sequence->nodes;
        for (unsigned i = childIndex; i > 0; --i) {
            UFEffect effect = nodes[i - 1].effect;
            if (effect != UFEffect::Neutral) {
                return effect;
            }
        }

        // A loop body is executed repeatedly: when the iteration start is reached,
        // the previous iteration's tail is the nearest prior action. This is the
        // circular walk the design's A2 requires; walking past the body into the
        // code before the loop would prove the pairing by accident.
        const Node* owner = sequence->owner;
        if (owner && owner->kind == Node::Kind::Loop) {
            for (unsigned i = static_cast<unsigned>(nodes.size()); i > childIndex + 1; --i) {
                UFEffect effect = nodes[i - 1].effect;
                if (effect != UFEffect::Neutral) {
                    return effect;
                }
            }
        }

        if (!owner) {
            return std::nullopt;
        }
        childIndex = owner->childIndex;
        sequence = owner->parentSequence;
    }
    return std::nullopt;
}

bool UFTokenModel::loopEntryIsSet(const Node& loop) const
{
    const Sequence* sequence = loop.parentSequence;
    unsigned childIndex = loop.childIndex;
    while (sequence) {
        const auto& nodes = *sequence->nodes;
        for (unsigned i = childIndex; i > 0; --i) {
            const Node& prior = nodes[i - 1];
            if (prior.effect == UFEffect::Neutral) {
                continue;
            }
            // Set->Set preserves an existing publication; it cannot create
            // one, especially when a preceding loop executes zero times.
            return prior.effect == UFEffect::Publish ||
                   (prior.effect == UFEffect::Set && loopEntryIsSet(prior));
        }
        const Node* owner = sequence->owner;
        if (!owner) {
            break;
        }
        childIndex = owner->childIndex;
        sequence = owner->parentSequence;
    }
    return false;
}

// Lift a leaf only through branches that summarize exactly its token action.
// Check its local tail/head too: the sibling branch alone cannot prove adjacency.
const UFTokenModel::Node* UFTokenModel::liftEndpoint(
    const Node& leaf, const Sequence* target, bool front) const
{
    const Node* node = &leaf;
    const UFEffect expected = effectOfAction(actionOf(leaf.index));
    while (node->parentSequence && node->parentSequence != target) {
        const Sequence* sequence = node->parentSequence;
        const Node* owner = sequence->owner;
        if (!owner) {
            return nullptr;
        }
        const bool branchEndpoint = owner->kind == Node::Kind::If && owner->effect == expected;
        const bool loopEndpoint = owner->kind == Node::Kind::Loop && owner->bodyEffect == UFEffect::Set &&
                                  loopEntryIsSet(*owner) && isLoopBodyBalanced(owner->endIndex);
        if (!branchEndpoint && !loopEndpoint) {
            return nullptr;
        }
        const auto& siblings = *sequence->nodes;
        const unsigned begin = front ? node->childIndex + 1 : 0;
        const size_t end = front ? siblings.size() : node->childIndex;
        if (!spanNodesTokenFree(siblings, begin, end)) {
            return nullptr;
        }
        node = owner;
    }
    return node->parentSequence == target ? node : nullptr;
}

bool UFTokenModel::isHandoffAdjacent(unsigned frontIndex, unsigned nowIndex, bool backEdge, unsigned loopEndIndex) const
{
    const Node* front = findLeaf(frontIndex);
    const Node* now = findLeaf(nowIndex);
    if (!front || !now) {
        return false;
    }
    const Node* originalFront = front;
    const UFTokenAction frontAction = actionOf(frontIndex);
    if (frontAction == UFTokenAction::None) {
        return false;
    }
    const bool usable = pairRegionUsable(*front, *front) && pairRegionUsable(*now, *now);
    if (!usable) {
        return false;
    }
    const bool needsLifting = !sameRegion(*front, *now) && !backEdge;
    if (needsLifting) {
        if (const Node* lifted = liftEndpoint(*front, now->parentSequence, true)) {
            front = lifted;
        } else {
            now = liftEndpoint(*now, front->parentSequence, false);
        }
    }
    if (!now || !sameRegion(*front, *now)) {
        return false;
    }

    if (backEdge) {
        const bool sameLoop =
            enclosingLoopEnd(frontIndex) == loopEndIndex && enclosingLoopEnd(nowIndex) == loopEndIndex;
        if (!sameLoop) {
            return false;
        }
        if (!isLoopBodyBalanced(loopEndIndex)) {
            return false;
        }
        if (!circularSpanTokenFree(loopEndIndex, *front, *now)) {
            return false;
        }
    } else if (!forwardSpanTokenFree(*front, *now)) {
        return false;
    }

    const Node* preconditionFront = front->kind == Node::Kind::Loop ? originalFront : front;
    return handoffPreconditionHolds(*preconditionFront, frontAction);
}

bool UFTokenModel::handoffPreconditionHolds(const Node& front, UFTokenAction action) const
{
    const std::optional<UFEffect> prior = priorTokenEffectBefore(front);
    if (!prior.has_value()) {
        // Reaching the region start is the pre-existing "no stale phase" contract:
        // the user's first phase determines the initial token state.
        return true;
    }
    if (*prior == UFEffect::Unknown) {
        return false;
    }
    const UFEffect expected = action == UFTokenAction::Publish ? UFEffect::Consume : UFEffect::Publish;
    const UFEffect balancedExpected = action == UFTokenAction::Publish ? UFEffect::Clear : UFEffect::Set;
    return *prior == expected || *prior == balancedExpected;
}

// Match a prologue/epilogue pair or one endpoint crossing a loop boundary.
const UFTokenModel::Node* UFTokenModel::pipelineBoundaryLoop(const Node& front, const Node& now) const
{
    if (sameRegion(front, now)) {
        if (now.childIndex != front.childIndex + 2 ||
            actionOf(front.index) != UFTokenAction::Publish || actionOf(now.index) != UFTokenAction::Consume) {
            return nullptr;
        }
        return &(*front.parentSequence->nodes)[front.childIndex + 1];
    }
    const Node* nowOwner = now.parentSequence ? now.parentSequence->owner : nullptr;
    if (nowOwner && sameRegion(front, *nowOwner)) {
        return pipelineInnerEndpointMatches(front, now, *nowOwner, true) ? nowOwner : nullptr;
    }
    const Node* frontOwner = front.parentSequence ? front.parentSequence->owner : nullptr;
    if (frontOwner && sameRegion(*frontOwner, now)) {
        return pipelineInnerEndpointMatches(front, now, *frontOwner, false) ? frontOwner : nullptr;
    }
    return nullptr;
}

bool UFTokenModel::pipelineInnerEndpointMatches(
    const Node& front, const Node& now, const Node& loop, bool entering) const
{
    const UFTokenAction expected = entering ? UFTokenAction::Publish : UFTokenAction::Consume;
    const bool matchingActions = actionOf(front.index) == expected && actionOf(now.index) == expected;
    if (!matchingActions) {
        return false;
    }
    if (entering) {
        return front.childIndex + 1 == loop.childIndex &&
               spanNodesTokenFree(*now.parentSequence->nodes, now.childIndex + 1,
                                  now.parentSequence->nodes->size());
    }
    return loop.childIndex + 1 == now.childIndex &&
           spanNodesTokenFree(*front.parentSequence->nodes, 0, front.childIndex);
}

// A Set->Set loop preserves a prologue publication through zero or more
// complete consume/publish iterations. This also orders same-pipe ACC reuse.
bool UFTokenModel::isPipelineBoundaryOrdered(unsigned frontIndex, unsigned nowIndex) const
{
    const Node* front = findLeaf(frontIndex);
    const Node* now = findLeaf(nowIndex);
    if (!front || !now || frontIndex >= nowIndex) {
        return false;
    }
    const bool usable = pairRegionUsable(*front, *front) && pairRegionUsable(*now, *now);
    if (!usable) {
        return false;
    }
    const Node* loop = pipelineBoundaryLoop(*front, *now);
    if (!loop || loop->kind != Node::Kind::Loop || loop->bodyEffect != UFEffect::Set || !loopEntryIsSet(*loop)) {
        return false;
    }
    return isLoopBodyBalanced(loop->endIndex);
}

bool UFTokenModel::isLoopBodyBalanced(unsigned loopEndIndex) const
{
    auto it = loopBalanced_.find(loopEndIndex);
    return it != loopBalanced_.end() && it->second;
}

std::optional<unsigned> UFTokenModel::enclosingLoopEnd(unsigned compoundIndex) const
{
    auto leaf = leaves_.find(compoundIndex);
    if (leaf == leaves_.end()) {
        return std::nullopt;
    }
    const Node* node = leaf->second;
    const Sequence* sequence = node->parentSequence;
    while (sequence) {
        const Node* owner = sequence->owner;
        if (owner && owner->kind == Node::Kind::Loop) {
            return owner->endIndex;
        }
        if (!owner) {
            return std::nullopt;
        }
        sequence = owner->parentSequence;
    }
    return std::nullopt;
}

void UFTokenModel::emitDeadlockDiagnostics() const
{
    emitIfDiagnostics(roots_);
    emitPartialPhaseDiagnostics(roots_);
    emitLoopEntryDiagnostics(roots_);
}

void UFTokenModel::emitLoopEntryDiagnostics(const std::vector<Node>& nodes) const
{
    for (const Node& node : nodes) {
        if (node.kind == Node::Kind::Loop && node.bodyEffect == UFEffect::Set && !loopEntryIsSet(node)) {
            if (Operation* op = opOfNode(node)) {
                op->emitWarning() << "consume-first unit-flag loop has no preceding publish: "
                                    "the first consume can wait forever for the unit flag (issue #1694)";
            }
        }
        emitLoopEntryDiagnostics(node.body);
        emitLoopEntryDiagnostics(node.elseBody);
    }
}

void UFTokenModel::emitIfDiagnostics(const std::vector<Node>& nodes) const
{
    for (const Node& node : nodes) {
        if (node.kind == Node::Kind::Loop) {
            emitIfDiagnostics(node.body);
            continue;
        }
        if (node.kind != Node::Kind::If) {
            continue;
        }
        emitIfDiagnostics(node.body);
        emitIfDiagnostics(node.elseBody);
        if (node.effect != UFEffect::Unknown || node.index >= syncIR_.size()) {
            continue;
        }
        Operation* op = syncIR_[node.index]->elementOp;
        if (!op) {
            continue;
        }
        const UFEffect thenEffect = node.bodyEffect;
        const UFEffect elseEffect = node.elseEffect;
        const bool thenConsumeElseNeutral = thenEffect == UFEffect::Consume && elseEffect == UFEffect::Neutral;
        const bool thenNeutralElseConsume = thenEffect == UFEffect::Neutral && elseEffect == UFEffect::Consume;
        const bool thenPublishElseNeutral = thenEffect == UFEffect::Publish && elseEffect == UFEffect::Neutral;
        const bool thenNeutralElsePublish = thenEffect == UFEffect::Neutral && elseEffect == UFEffect::Publish;
        const bool consistentBranches = thenEffect != UFEffect::Unknown && elseEffect != UFEffect::Unknown;
        if (thenConsumeElseNeutral || thenNeutralElseConsume) {
            op->emitWarning() << "conditional unit-flag consume: an iteration that skips the "
                                 "store leaves the unit flag set, so the next AccPhase::Final "
                                 "matmul's check_and_set can hang (issue #1694)";
        } else if (thenPublishElseNeutral || thenNeutralElsePublish) {
            op->emitWarning() << "conditional unit-flag publish: an iteration that skips the "
                                 "matmul leaves the fixpipe waiting for a flag that never arrives "
                                 "(issue #1694)";
        } else if (consistentBranches && thenEffect != elseEffect) {
            op->emitWarning() << "the two branches leave the unit-flag token in different states; "
                                 "pairing a matmul with a store across this branch can hang on the "
                                 "MAD's check_and_set (issue #1694)";
        }
    }
}

void UFTokenModel::emitPartialPhaseDiagnostics(const std::vector<Node>& nodes) const
{
    for (const Node& node : nodes) {
        if (node.kind == Node::Kind::Loop) {
            emitPartialPhaseDiagnostics(node.body);
            continue;
        }
        if (node.kind == Node::Kind::If) {
            emitPartialPhaseDiagnostics(node.body);
            emitPartialPhaseDiagnostics(node.elseBody);
            continue;
        }
        if (node.index >= syncIR_.size()) {
            continue;
        }
        auto* store = dyn_cast_or_null<CompoundInstanceElement>(syncIR_[node.index].get());
        if (!store || store->ufInfo.role != UFRole::AccStore || !store->ufInfo.stPhase ||
            *store->ufInfo.stPhase != STPhase::Partial) {
            continue;
        }
        const Sequence* sequence = node.parentSequence;
        if (!sequence) {
            continue;
        }
        const std::vector<Node>& siblings = *sequence->nodes;
        for (unsigned i = node.childIndex; i > 0; --i) {
            const Node& sibling = siblings[i - 1];
            if (sibling.kind != Node::Kind::Compound || sibling.index >= syncIR_.size()) {
                continue;
            }
            auto* matmul = dyn_cast_or_null<CompoundInstanceElement>(syncIR_[sibling.index].get());
            if (!matmul || matmul->ufInfo.role != UFRole::MatmulProducer ||
                !accRegionsOverlap(store->ufInfo, matmul->ufInfo)) {
                continue;
            }
            if (matmul->ufInfo.accPhase && *matmul->ufInfo.accPhase == AccPhase::Partial && store->elementOp) {
                store->elementOp->emitWarning() << "TSTORE with STPhase::Partial follows a TMATMUL with "
                                                   "AccPhase::Partial; the fixpipe waits for a unit flag that is "
                                                   "never set and the kernel can hang (issue #1694)";
            }
            break;
        }
    }
}

} // namespace mlir::pto

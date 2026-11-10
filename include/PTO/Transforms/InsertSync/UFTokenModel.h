// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TRANSFORMS_INSERTSYNC_UFTOKENMODEL_H
#define PTO_TRANSFORMS_INSERTSYNC_UFTOKENMODEL_H

#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>
#include <optional>
#include <vector>

namespace mlir::pto {

// The two operations that move the a2/a3/a5 unit-flag token. A phased operation that
// only observes the token (a Partial store, a Partial matmul) leaves the state
// alone and therefore has no action of its own.
enum class UFTokenAction { None, Publish, Consume };

UFTokenAction getUFTokenAction(const CompoundInstanceElement& compound);

// State transition summary of a region. Branches must agree on a valid
// entering state, and loops must preserve that state.
// Bit relation: Clear->Clear, Clear->Set, Set->Clear, Set->Set.
// Unlike a net count, this retains the precondition of a balanced handshake.
enum class UFEffect : unsigned {
    Unknown = 0,
    Clear = 1,
    Publish = 2,
    Consume = 4,
    Set = 8,
    Neutral = 9
};

/// A structured view of the flat SyncIR, carrying the loop/branch nesting that
/// `isUFHandoffAdjacent` needs. The flat index walk cannot tell a real prior
/// hand-off from an instruction in a mutually exclusive branch or in a previous
/// loop iteration; this model answers those questions structurally.
///
/// The a2/a3/a5 unit flag is a depth-one handshake: a Final matmul publishes it
/// (check_and_set) and a Final store consumes it (check_and_clear). A pair is
/// proven only when the producer's precondition is established by the nearest
/// prior action on the same execution path, and no other token action sits
/// between the pair. Both facts are computed per structured region instead of
/// by scanning the flat index space.
class UFTokenModel {
public:
    explicit UFTokenModel(const SyncIRs& syncIR) : syncIR_(syncIR) {}
    ~UFTokenModel();

    /// Build the tree, effects and region tables. Must run before any query.
    void build();

    bool isBuilt() const { return built_; }

    /// True when the structured model proves `front` hands the token to `now`.
    /// `backEdge` selects the loop-carried pairing in either direction,
    /// which adds the circular span and the loop-body
    /// balance requirement of the design's B phase.
    bool isHandoffAdjacent(unsigned frontIndex, unsigned nowIndex, bool backEdge, unsigned loopEndIndex) const;

    /// The loop body's net effect is neutral, i.e. it publishes as many tokens as
    /// it consumes on every iteration. A non-neutral body does not race here, it
    /// hangs the MAD's check_and_set, so the back edge may not be dropped.
    bool isLoopBodyBalanced(unsigned loopEndIndex) const;
    bool isPipelineBoundaryOrdered(unsigned frontIndex, unsigned nowIndex) const;

    /// Innermost loop body containing the compound, keyed by its end marker.
    std::optional<unsigned> enclosingLoopEnd(unsigned compoundIndex) const;

    /// Emit warnings for phase combinations that hang on hardware. These are
    /// user errors that no amount of event synchronization can fix, so pointing
    /// them out at compile time is better than a device timeout.
    void emitDeadlockDiagnostics() const;

private:
    struct Sequence;

    // One node of the structured SyncIR. A compound keeps its canonical SyncIR
    // index so state is attached to the same operation the flat index names.
    struct Node {
        enum class Kind { Compound, Loop, If };

        Kind kind{Kind::Compound};
        // Compound: canonical SyncIR index. Loop/If: begin marker index.
        unsigned index{0};
        // Loop/If: end marker index.
        unsigned endIndex{0};

        std::vector<Node> body;
        std::vector<Node> elseBody;

        const Sequence* parentSequence{nullptr};
        unsigned childIndex{0};
        const Sequence* bodySequence{nullptr};
        const Sequence* elseSequence{nullptr};

        UFEffect effect{UFEffect::Neutral};
        UFEffect bodyEffect{UFEffect::Neutral};
        UFEffect elseEffect{UFEffect::Neutral};
        bool tokenFree{true};
    };

    // A linear sequence of nodes: the function body, a loop body, or one arm of
    // an scf.if. `owner` is null for the function body.
    struct Sequence {
        const std::vector<Node>* nodes{nullptr};
        const Node* owner{nullptr};
    };

    UFTokenAction actionOf(unsigned compoundIndex) const;
    const Node* findLeaf(unsigned compoundIndex) const;
    bool sameRegion(const Node& lhs, const Node& rhs) const;
    bool pairRegionUsable(const Node& lhs, const Node& rhs) const;
    bool forwardSpanTokenFree(const Node& front, const Node& now) const;
    bool circularSpanTokenFree(unsigned loopEndIndex, const Node& front, const Node& now) const;
    bool collectLoopChain(unsigned loopEndIndex, const Node& leaf,
                          SmallVectorImpl<std::pair<const Sequence*, unsigned>>& chain) const;
    bool spanNodesTokenFree(const std::vector<Node>& nodes, unsigned begin,
                            size_t end) const;
    std::optional<UFEffect> priorTokenEffectBefore(const Node& compound) const;
    bool handoffPreconditionHolds(const Node& front, UFTokenAction action) const;
    bool loopEntryIsSet(const Node& loop) const;
    const Node* pipelineBoundaryLoop(const Node& front, const Node& now) const;
    bool pipelineInnerEndpointMatches(const Node& front, const Node& now, const Node& loop, bool entering) const;
    const Node* liftEndpoint(const Node& leaf, const Sequence* target, bool front) const;
    void emitLoopEntryDiagnostics(const std::vector<Node>& nodes) const;

    void buildRange(unsigned begin, unsigned end, std::vector<Node>& out);
    void assignParents(std::vector<Node>& nodes, const Sequence* sequence);
    UFEffect computeSequenceEffect(std::vector<Node>& nodes);
    UFEffect computeNodeEffect(Node& node);
    UFEffect computeWhileEffect(Node& node, Operation* whileOp);
    Operation* opOfNode(const Node& node) const;
    void computeTokenFree(std::vector<Node>& nodes);
    void collectLeafInfo(const std::vector<Node>& nodes, const Sequence* sequence);
    void emitIfDiagnostics(const std::vector<Node>& nodes) const;
    void emitPartialPhaseDiagnostics(const std::vector<Node>& nodes) const;

    const SyncIRs& syncIR_;
    bool built_{false};
    std::vector<Node> roots_;
    // Owns the stable Sequence objects referenced by leaves and parents.
    std::vector<std::unique_ptr<Sequence>> sequences_;
    llvm::DenseMap<unsigned, const Node*> leaves_;
    llvm::DenseMap<unsigned, bool> loopBalanced_;
    llvm::DenseMap<unsigned, UFEffect> ifEffect_;
};

} // namespace mlir::pto

#endif // PTO_TRANSFORMS_INSERTSYNC_UFTOKENMODEL_H

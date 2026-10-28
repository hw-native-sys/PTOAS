// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMILayoutSupport.h - VMI layout support queries ------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#ifndef PTO_TRANSFORMS_VMILAYOUTSUPPORT_H
#define PTO_TRANSFORMS_VMILAYOUTSUPPORT_H

#include "PTO/Support/CodeConstants.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/VPTOMemoryDist.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"

#include <string>

namespace mlir::pto {

/// True when `layout` is a group-slot packet that lives in a single physical
/// chunk and puts logical lane i on physical lane i.
/// A `num_groups = G, slots = S, lane_stride = 1` layout places logical lane i
/// at {part 0, chunk i / S, lane i % S}.  With `G <= S` every lane lands in
/// chunk 0 at lane i, and `G <= lanesPerPart` keeps that lane inside the
/// carrier, so the value occupies exactly one physical register whose live
/// lanes are 0..G-1.
bool isVMISingleCarrierGroupSlots(VMILayoutAttr layout, int64_t lanesPerPart);

/// Same as isVMISingleCarrierGroupSlots with the lane stride spelled out: the
/// packet places logical lane i at {part 0, chunk i / S, lane (i % S) * LS}.
/// A lane stride other than 1 is only meaningful for sub-32-bit elements, whose
/// several values share one carrier lane; a 32-bit element already fills a
/// carrier lane.  The supported strides therefore follow the packable carrier
/// chain (32 -> 16 -> 8): stride 2 up to 16-bit elements, stride 4 up to 8-bit.
/// Only the slot widths the lowering builds qualify (one or eight group slots
/// per part): a packet with more slots spreads its groups over several carriers
/// and is hand-written IR that no producer emits.
bool isVMISingleCarrierGroupSlotsWithStride(VMILayoutAttr layout,
                                            int64_t lanesPerPart,
                                            int64_t laneStride);

/// True when a dense value and a single-carrier group packet differ in lane
/// stride, so the carrier identity cannot bridge them directly and the dense
/// lane stride has to be normalized through a contiguous intermediate first.
/// Kept next to the single-carrier predicate so the ensure_layout query and the
/// materializer agree on which pairs are supported.
/// `elementType` bounds the bridge to the shapes the dense lane-stride
/// materialization covers (8/16-bit elements; lane stride 4 only for 8-bit).
/// One of the two strides also has to be the unit stride, because the dense
/// lane-stride materialization moves between the unit stride and a strided form
/// only; two non-unit strides would need two such moves.
bool needsVMIDenseLaneStrideGroupSlotBridge(VMILayoutAttr sourceLayout,
                                            VMILayoutAttr resultLayout,
                                            Type elementType,
                                            int64_t lanesPerPart);

/// True when a group-slot packet and a dense contiguous vector describe exactly
/// the same physical lanes of a single carrier, so converting between them is a
/// pure register forward with no pack/zip/shuffle.  Symmetric in `lhs`/`rhs`:
/// the relation holds in both directions.
/// Short dense vectors reach group operations through this relation: a group
/// reduce with one result group, or a group broadcast reading eight slot
/// values, is a genuine group packet, while the value a plain vload produces is
/// dense.  Stating the invariant keeps that bridge free of any group count
/// list.
bool isVMISingleCarrierGroupSlotAlias(VMILayoutAttr lhs, VMILayoutAttr rhs,
                                      int64_t lanesPerPart);

struct VMILoadLayoutFact {
  VMILayoutAttr resultLayout;
};

/// A pto.vmi.group_iota result.  The contiguous vci instruction is the physical
/// producer, so the only layout the op itself can produce is contiguous;
/// non-contiguous consumers are expressed by an explicit ensure_layout edge.
struct VMIGroupIotaLayoutFact {
    VMILayoutAttr resultLayout;
};

enum class VMIDeinterleaveLoadLayoutPort {
  Low,
  High,
};

struct VMIDeinterleaveLoadLayoutFact {
  VMILayoutAttr lowLayout;
  VMILayoutAttr highLayout;
};

struct VMIStoreLayoutFact {
  VMILayoutAttr valueLayout;
};

struct VMIMaskedStoreLayoutFact {
  VMILayoutAttr valueLayout;
  VMILayoutAttr maskLayout;
};

struct VMIMaskedLoadLayoutFact {
  VMILayoutAttr resultLayout;
  VMILayoutAttr maskLayout;
  VMILayoutAttr passthruLayout;
};

struct VMIEnsureLayoutFact {
  VMILayoutAttr sourceLayout;
  VMILayoutAttr resultLayout;
  // True when the conversion selects the same physical parts on both sides.
  // The register is then forwarded unchanged: no lane rearrangement runs and
  // the result arity equals the source arity.
  bool forwardsPhysicalParts = false;
};

struct VMIEnsureMaskLayoutFact {
  VMILayoutAttr sourceLayout;
  VMILayoutAttr resultLayout;
  // As in VMIEnsureLayoutFact: the predicate register itself is forwarded.
  bool forwardsPhysicalParts = false;
};

struct VMIGeneratedMaskLayoutFact {
    VMILayoutAttr generationLayout;
    VMILayoutAttr resultLayout;
};

enum class VMICastLayoutPort {
  Source,
  Result,
};

enum class VMICastLayoutPriority {
  Normal,
  High,
  LaneStrideNarrowing,
};

enum class VMIInterleaveLayoutPort {
  Lhs,
  Rhs,
  Mask,
  Low,
  High,
};

struct VMICastLayoutFact {
  VMILayoutAttr sourceLayout;
  VMILayoutAttr resultLayout;
  int64_t sourceBits = 0;
  int64_t resultBits = 0;
  VMICastLayoutPriority priority = VMICastLayoutPriority::Normal;
  // Number of layout-rearrangement instructions performed by the cast
  // lowering itself.  Numeric conversion instructions are not included.
  int64_t intrinsicRearrangementCost = 0;
};

struct VMIMaskGranularityCastLayoutFact {
  VMILayoutAttr sourceLayout;
  VMILayoutAttr resultLayout;
  int64_t sourceGranularityBits = 0;
  int64_t resultGranularityBits = 0;
  // Number of predicate-carrier steps the granularity conversion performs.
  int64_t intrinsicRearrangementCost = 0;
};

struct VMIInterleaveLayoutFact {
  VMILayoutAttr lhsLayout;
  VMILayoutAttr rhsLayout;
  VMILayoutAttr maskLayout;
  VMILayoutAttr lowLayout;
  VMILayoutAttr highLayout;
  int64_t elementCount = 0;
  int64_t lanesPerPart = 0;
};

struct VMIBitcastLayoutFact {
  VMILayoutAttr sourceLayout;
  VMILayoutAttr resultLayout;
};

enum class VMIGroupBlockClass {
  Compact,
  QuarterBlock,
  HalfBlock,
  OneBlock,
  TwoBlock,
  FourBlock,
  FullPartMultiple,
};

struct VMIGroupStoreLayoutFact {
  VMILayoutAttr valueLayout;
  // Layout the value has to be materialized into before the store when the
  // assigned value layout is not one the group_store lowering can write
  // directly; null when no staging step is needed.
  VMILayoutAttr stagingLayout;
  VMIGroupBlockClass blockClass = VMIGroupBlockClass::OneBlock;
  int64_t groupSize = 0;
  int64_t lanesPerPart = 0;
  int64_t vcgBlockElems = 0;
};

struct VMIGroupReduceLayoutFact {
  VMIGroupBlockClass blockClass = VMIGroupBlockClass::OneBlock;
  VMILayoutAttr sourceLayout;
  VMILayoutAttr maskLayout;
  VMILayoutAttr resultLayout;
  int64_t groupSize = 0;
  int64_t lanesPerPart = 0;
  int64_t vcgBlockElems = 0;
};

/// A non-grouped reduce (pto.vmi.reduce_*, lowered through the legacy vcadd /
/// vcmax / vcmin family).  The legacy lowering consumes complete physical
/// source chunks and produces a contiguous result, so the relation is the
/// contiguous identity.
struct VMIReduceLayoutFact {
    VMILayoutAttr sourceLayout;
    VMILayoutAttr maskLayout;
    VMILayoutAttr resultLayout;
};

struct VMIGroupBroadcastLayoutFact {
  VMIGroupBlockClass blockClass = VMIGroupBlockClass::OneBlock;
  VMILayoutAttr sourceLayout;
  VMILayoutAttr resultLayout;
  int64_t groupSize = 0;
  int64_t lanesPerPart = 0;
  int64_t vcgBlockElems = 0;
};

enum class VMIGroupBroadcastLoadDirectKind {
  E2B,
  BRC,
};

struct VMIGroupBroadcastLoadLayoutFact {
  VMIGroupBlockClass blockClass = VMIGroupBlockClass::OneBlock;
  VMILayoutAttr resultLayout;
  int64_t groupSize = 0;
  int64_t lanesPerPart = 0;
  int64_t vcgBlockElems = 0;
  int64_t elementBits = 0;
};

struct VMIGroupBroadcastLoadDirectFact {
  VMIGroupBroadcastLoadDirectKind kind = VMIGroupBroadcastLoadDirectKind::E2B;
  VMIGroupBroadcastLoadLayoutFact layout;
};

struct VMIGroupLoadLayoutFact {
  VMIGroupBlockClass blockClass = VMIGroupBlockClass::TwoBlock;
  VMILayoutAttr resultLayout;
  int64_t groupSize = 0;
};

struct VMIGroupSlotLayoutFact {
  VMILayoutAttr layout;
  int64_t numGroups = 0;
  int64_t slots = 0;
};

// Layout/shape contract shared by the relation provider and VPTO lowering for
// the two-source interleave store.  Memory-address legality remains in the
// lowering-specific access-plan checker.
struct VMIInterleaveStoreSupport {
    VMILayoutAttr lowLayout;
    VMILayoutAttr highLayout;
};

// Integer vcgadd widens its physical sums; other reductions keep their width.
enum class VMIGroupReduceKind { IntegerAdd, Other };
VMIGroupReduceKind getVMIGroupReduceKind(Operation *op);

enum class VMIGroupReduceLayoutPort {
  Source,
  Mask,
  Result,
};

enum class VMIGroupBroadcastLayoutPort {
  Source,
  Result,
};

struct VMIHistogramLayoutFact {
  VMILayoutAttr accLayout;
  VMILayoutAttr sourceLayout;
  VMILayoutAttr maskLayout;
  VMILayoutAttr resultLayout;
};

struct VMIVselrLayoutFact {
  VMILayoutAttr sourceLayout;
  VMILayoutAttr indexLayout;
  VMILayoutAttr resultLayout;
};

enum class VMIVexpdifLayoutPort {
    Source,
    Result,
};

/// A pto.vmi.vexpdif relation the VPTO lowering can realize.  The lowering
/// reads one 256-bit source register at a time, so a relation is legal only
/// when the physical parts of x/max, of the predicate, and of the result line
/// up the way the emitted pto.vexpdif ops produce them:
///
///  * An f32 source keeps its element width, so one source part produces one
///    result part and x, max, the mask, and the result share one layout.  The
///    layout must stay dense with lane_stride = 1: the mask follows the source
///    layout, and its physical granularity is granularity * lane_stride, which
///    the lowering accepts only while it still matches the data element width.
///  * An f16 source widens to f32.  One source part yields the even and the
///    odd lanes of that part as two result parts, so the result has to be
///    deinterleaved = 2 while x, max, and the mask keep their natural lane
///    order (contiguous).
///
/// sourceParts/resultParts are the physical part counts of the source and the
/// result under this relation.  The lowering also requires the mask to produce
/// exactly sourceParts parts and the element-width ratio to connect the two
/// counts, so a row only survives while those part counts agree.
struct VMIVexpdifLayoutFact {
    VMILayoutAttr sourceLayout;
    VMILayoutAttr resultLayout;
    int64_t sourceParts = 0;
    int64_t resultParts = 0;
    int64_t resultPartsPerSourcePart = 1;
    bool preferred = false;
};

class VMILayoutSupport {
public:
    FailureOr<VMILoadLayoutFact> getLoadLayoutFact(VMIVRegType resultType, std::string* reason = nullptr) const;

    FailureOr<VMIDeinterleaveLoadLayoutFact> getPreferredDeinterleaveLoadLayoutFact(
        VMIVRegType valueType, std::string* reason = nullptr) const;

    FailureOr<SmallVector<VMIDeinterleaveLoadLayoutFact, mlir::pto::kValue4>> getDeinterleaveLoadLayoutFactsForLayout(
        VMIVRegType valueType, VMIDeinterleaveLoadLayoutPort port, VMILayoutAttr layout,
        std::string* reason = nullptr) const;

    FailureOr<VMIDeinterleaveLoadLayoutFact> getDeinterleaveLoadLayoutFactForLayouts(
        VMIVRegType lowType, VMIVRegType highType, std::string* reason = nullptr) const;

    FailureOr<VMIStoreLayoutFact> getStoreLayoutFact(VMIVRegType valueType, std::string* reason = nullptr) const;

    FailureOr<VMIStoreLayoutFact> getPreferredStoreLayoutFact(
        VMIVRegType valueType, std::string* reason = nullptr) const;

    FailureOr<VMIMaskedStoreLayoutFact> getMaskedStoreLayoutFact(
        VMIVRegType valueType, VMIMaskType maskType, std::string* reason = nullptr) const;

    FailureOr<VMIMaskedStoreLayoutFact> getPreferredMaskedStoreLayoutFact(
        VMIVRegType valueType, VMIMaskType maskType, std::string* reason = nullptr) const;

    FailureOr<VMIMaskedLoadLayoutFact> getMaskedLoadLayoutFact(
        VMIVRegType resultType, VMIMaskType maskType, VMIVRegType passthruType, std::string* reason = nullptr) const;

    FailureOr<VMIEnsureLayoutFact> getEnsureLayoutFact(
        VMIVRegType sourceType, VMIVRegType resultType, std::string* reason = nullptr) const;

    FailureOr<VMIEnsureMaskLayoutFact> getEnsureMaskLayoutFact(
        VMIMaskType sourceType, VMIMaskType resultType, std::string* reason = nullptr) const;

    FailureOr<VMIGeneratedMaskLayoutFact> getGeneratedMaskLayoutFact(
        Operation* op, VMILayoutAttr resultLayout, std::string* reason = nullptr) const;

    // Preferred cast relation.  \p allowLaneStridePreference decides whether the
    // lane-stride *cost* rows may answer ahead of the default preferred table:
    // the one-chunk high-priority relations and the
    // kPreferredLaneStrideNarrowCastLayoutPatterns rows.  Passing false leaves
    // only the table's own default row for the width pair, which for every
    // widening pair is {c(), d(f)} - i.e. the narrow side stays contiguous and
    // the wide side takes the deinterleaved form the per-part vcvt actually
    // produces.  It is a caller-supplied fact rather than a property of the width
    // pair, because whether the lane stride is worth paying is decided by what the
    // narrow side is used for, not by the cast.
    FailureOr<VMICastLayoutFact> getPreferredCastLayoutFact(
        VMIVRegType sourceType, VMIVRegType resultType, std::string* reason = nullptr,
        bool allowLaneStridePreference = true) const;

    // Spine-*scoped* cast layout query.
    // These queries are the only way to reach kSpineScopedCastLayoutPatterns /
    // kSpineScopedLegalCastLayoutPatterns.  Those rows are not part of the
    // generic preferred/legal tables, so a caller that does not explicitly ask
    // for the spine-scoped variant can never observe them - the scope is enforced
    // by construction (the rows are reachable from this entry point only), not by
    // "the candidate happened not to be selected".
    // The direction-spine peephole in VMILayoutAssignment calls this variant for
    // the cast ops of a matched <up,down,up,down> chain whose narrowing leg is
    // handed straight to the closing widening leg.  The scoped table adds the
    // composite 16<->32 and 32<->8 rows (deinterleaved=4 -> deinterleaved=4 with
    // an explicit lane stride) that keep the narrow value split over the four
    // physical parts of its wide side, so the narrowing and the widening are
    // per-chunk 1:1 and need no pto.vor assembly.
    FailureOr<VMICastLayoutFact> getSpineScopedCastLayoutFact(
        VMIVRegType sourceType, VMIVRegType resultType, std::string* reason = nullptr) const;

    FailureOr<VMICastLayoutFact> getSpineScopedCastLayoutFactForLayouts(
        VMIVRegType sourceType, VMIVRegType resultType, VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
        std::string* reason = nullptr) const;

    FailureOr<SmallVector<VMICastLayoutFact, mlir::pto::kValue4>> getSpineScopedCastLayoutFactsForLayout(
        VMIVRegType sourceType, VMIVRegType resultType, VMICastLayoutPort port, VMILayoutAttr layout,
        std::string* reason = nullptr) const;

    FailureOr<SmallVector<VMICastLayoutFact, mlir::pto::kValue4>> getCastLayoutFactsForLayout(
        VMIVRegType sourceType, VMIVRegType resultType, VMICastLayoutPort port, VMILayoutAttr layout,
        std::string* reason = nullptr) const;

    FailureOr<VMICastLayoutFact> getCastLayoutFactForSourceLayout(
        VMIVRegType sourceType, VMIVRegType resultType, VMILayoutAttr sourceLayout,
        std::string* reason = nullptr) const;

    FailureOr<VMICastLayoutFact> getCastLayoutFactForResultLayout(
        VMIVRegType sourceType, VMIVRegType resultType, VMILayoutAttr resultLayout,
        std::string* reason = nullptr) const;

    FailureOr<VMICastLayoutFact> getCastLayoutFactForLayouts(
        VMIVRegType sourceType, VMIVRegType resultType, VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
        std::string* reason = nullptr) const;

    FailureOr<SmallVector<VMIMaskGranularityCastLayoutFact, mlir::pto::kValue4>>
    getMaskGranularityCastLayoutFactsForLayout(
        VMIMaskType sourceType, VMIMaskType resultType, VMICastLayoutPort port, VMILayoutAttr layout,
        std::string* reason = nullptr) const;

    FailureOr<VMIMaskGranularityCastLayoutFact> getMaskGranularityCastLayoutFactForLayouts(
        VMIMaskType sourceType, VMIMaskType resultType, VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
        std::string* reason = nullptr) const;

    FailureOr<VMILayoutAttr> getWidenSourceLayoutForResultLayout(
        VMIVRegType sourceType, VMIVRegType resultType, VMILayoutAttr requestedResultLayout,
        std::string* reason = nullptr) const;

    FailureOr<VMIInterleaveLayoutFact> getPreferredVintlvLayoutFact(
        VMIVRegType valueType, std::string* reason = nullptr) const;

    FailureOr<VMIInterleaveLayoutFact> getPreferredVdintlvLayoutFact(
        VMIVRegType valueType, std::string* reason = nullptr) const;

    FailureOr<SmallVector<VMIInterleaveLayoutFact, mlir::pto::kValue4>> getVintlvLayoutFactsForLayout(
        VMIVRegType valueType, VMIInterleaveLayoutPort port, VMILayoutAttr layout, std::string* reason = nullptr) const;

    FailureOr<SmallVector<VMIInterleaveLayoutFact, mlir::pto::kValue4>> getVdintlvLayoutFactsForLayout(
        VMIVRegType valueType, VMIInterleaveLayoutPort port, VMILayoutAttr layout, std::string* reason = nullptr) const;

    FailureOr<VMIInterleaveLayoutFact> getVintlvLayoutFactForLayouts(
        VMIVRegType lhsType, VMIVRegType rhsType, VMIMaskType maskType, VMIVRegType lowType, VMIVRegType highType,
        std::string* reason = nullptr) const;

    FailureOr<VMIInterleaveLayoutFact> getVdintlvLayoutFactForLayouts(
        VMIVRegType lhsType, VMIVRegType rhsType, VMIMaskType maskType, VMIVRegType lowType, VMIVRegType highType,
        std::string* reason = nullptr) const;

    FailureOr<VMIGroupSlotLayoutFact> getGroupSlotLoadLayoutFact(
        VMIVRegType resultType, int64_t numGroups, std::string* reason = nullptr) const;

    FailureOr<VMIInterleaveStoreSupport> getInterleaveStoreSupport(
        VMIVRegType lowType, VMIVRegType highType, std::string* reason = nullptr) const;

    FailureOr<VMIGroupLoadLayoutFact> getGroupLoadLayoutFact(VMIGroupLoadOp op, std::string* reason = nullptr) const;
    FailureOr<VMIGroupLoadLayoutFact> getGroupLoadLayoutFact(
        VMIVRegType resultType, Value rowStride, int64_t numGroups, std::string* reason = nullptr) const;

    FailureOr<VMIGroupSlotLayoutFact> getGroupStoreLayoutFact(
        VMIVRegType valueType, int64_t numGroups, std::string* reason = nullptr) const;

    FailureOr<VMIGroupStoreLayoutFact> getGroupStoreLayoutFact(
        VMIGroupStoreOp op, VMIVRegType valueType, std::string* reason = nullptr) const;

    FailureOr<SmallVector<VMIGroupStoreLayoutFact, mlir::pto::kValue4>> getGroupStoreLayoutFactsForLayout(
        VMIGroupStoreOp op, VMIVRegType valueType, VMILayoutAttr layout, std::string* reason = nullptr) const;

    FailureOr<VMIGroupStoreLayoutFact> getPreferredGroupStoreLayoutFact(
        VMIGroupStoreOp op, VMIVRegType valueType, std::string* reason = nullptr) const;

    FailureOr<VMIGroupStoreLayoutFact> getHighPriorityGroupStoreLayoutFact(
        VMIGroupStoreOp op, VMIVRegType valueType, std::string* reason = nullptr) const;

    LogicalResult getGroupOperationShapeSupport(Operation* op, std::string* reason = nullptr) const;

    // Shape-only capability checks for diagnostics before layout assignment.
    // Assigned source layouts are respected; lowering checks the full fact again.
    LogicalResult getGroupReduceShapeSupport(
        VMIGroupReduceKind kind, VMIVRegType sourceType, int64_t numGroups, std::string* reason = nullptr) const;
    FailureOr<VMIGroupBroadcastLayoutFact> getPreferredGroupBroadcastLayoutFact(
        VMIVRegType resultType, int64_t numGroups, std::string* reason = nullptr) const;
    VMILayoutAttr getPreferredGroupBroadcastResultLayout(VMIVRegType resultType, int64_t numGroups) const;
    LogicalResult getGroupBroadcastShapeSupport(
        VMIVRegType resultType, int64_t numGroups, std::string* reason = nullptr) const;

    FailureOr<VMIGroupReduceLayoutFact> getPreferredGroupReduceLayoutFact(
        VMIGroupReduceKind kind, VMIVRegType sourceType, int64_t numGroups, std::string* reason = nullptr) const;

    FailureOr<VMIGroupReduceLayoutFact> getGroupReduceLayoutFactForLayouts(
        VMIGroupReduceKind kind, VMIVRegType sourceType, VMIMaskType maskType, VMIVRegType resultType,
        int64_t numGroups, std::string* reason = nullptr) const;

    FailureOr<SmallVector<VMIGroupReduceLayoutFact, mlir::pto::kValue4>> getGroupReduceLayoutFactsForLayout(
        VMIGroupReduceKind kind, VMIVRegType sourceType, int64_t numGroups, VMIGroupReduceLayoutPort port,
        VMILayoutAttr layout, std::string* reason = nullptr) const;

    FailureOr<VMIGroupBroadcastLayoutFact> getGroupBroadcastLayoutFactForLayouts(
        VMIVRegType sourceType, VMIVRegType resultType, int64_t numGroups, std::string* reason = nullptr) const;

    FailureOr<SmallVector<VMIGroupBroadcastLayoutFact, mlir::pto::kValue4>> getGroupBroadcastLayoutFactsForLayout(
        VMIVRegType sourceType, VMIVRegType resultType, int64_t numGroups, VMIGroupBroadcastLayoutPort port,
        VMILayoutAttr layout, std::string* reason = nullptr) const;

    FailureOr<VMIGroupBroadcastLoadLayoutFact> getGroupBroadcastLoadLayoutFact(
        VMIGroupBroadcastLoadOp op, std::string* reason = nullptr) const;
    FailureOr<VMIGroupBroadcastLoadLayoutFact> getGroupBroadcastLoadLayoutFact(
        VMIVRegType resultType, Value sourceGroupStride, int64_t numGroups, std::string* reason = nullptr) const;
    FailureOr<VMIGroupBroadcastLoadDirectFact> getGroupBroadcastLoadDirectFact(
        VMIGroupBroadcastLoadOp op, std::string* reason = nullptr) const;
    FailureOr<VMIGroupBroadcastLoadDirectFact> getGroupBroadcastLoadDirectFact(
        VMIVRegType resultType, Type sourceType, Value sourceGroupStride, int64_t numGroups,
        std::string* reason = nullptr) const;

    FailureOr<VMIHistogramLayoutFact> getVdhistLayoutFact(VMIVdhistOp op, std::string* reason = nullptr) const;

    FailureOr<VMIHistogramLayoutFact> getVchistLayoutFact(VMIVchistOp op, std::string* reason = nullptr) const;

    /// Facts for every vexpdif table row whose source (or result) layout is
    /// p layout.  Every returned fact is a relation the VPTO lowering can
    /// realize for this operation's shape, so a planner that enumerates only
    /// these facts cannot select an unlowerable plan.
    FailureOr<SmallVector<VMIVexpdifLayoutFact, mlir::pto::kValue4>> getVexpdifLayoutFactsForLayout(
        VMIVexpdifOp op, VMIVexpdifLayoutPort port, VMILayoutAttr layout, std::string* reason = nullptr) const;

    /// The vexpdif table row this shape prefers.  Plans may pick another row,
    /// which the planner charges as a layout preference penalty.
    FailureOr<VMIVexpdifLayoutFact> getPreferredVexpdifLayoutFact(VMIVexpdifOp op, std::string* reason = nullptr) const;

    FailureOr<VMIVselrLayoutFact> getPreferredVselrLayoutFact(VMIVselrOp op, std::string* reason = nullptr) const;

    FailureOr<VMIVselrLayoutFact> getVselrLayoutFact(VMIVselrOp op, std::string* reason = nullptr) const;

    LogicalResult getVselrSupport(VMIVselrOp op, std::string* reason = nullptr) const;

    LogicalResult getGroupReduceAddFSupport(VMIGroupReduceAddFOp op, std::string* reason = nullptr) const;

    LogicalResult getGroupReduceMaxFSupport(VMIGroupReduceMaxFOp op, std::string* reason = nullptr) const;

    LogicalResult getGroupReduceMinFSupport(VMIGroupReduceMinFOp op, std::string* reason = nullptr) const;

    LogicalResult getGroupReduceAddISupport(VMIGroupReduceAddIOp op, std::string* reason = nullptr) const;

    LogicalResult getGroupReduceMaxISupport(VMIGroupReduceMaxIOp op, std::string* reason = nullptr) const;

    LogicalResult getGroupReduceMinISupport(VMIGroupReduceMinIOp op, std::string* reason = nullptr) const;

    LogicalResult getGroupBroadcastSupport(VMIGroupBroadcastOp op, std::string* reason = nullptr) const;

    LogicalResult getGroupBroadcastSupport(
        VMIVRegType sourceType, VMIVRegType resultType, int64_t numGroups, std::string* reason = nullptr) const;

    LogicalResult getGroupBroadcastLoadSupport(VMIGroupBroadcastLoadOp op, std::string* reason = nullptr) const;

    /// Whether the already-assigned layout pair of \p op has a registered
    /// lowering.  \p enableVmiFastmath reflects the state the lowering will
    /// actually run in: a dense same-layout BF16 narrowing pair is accepted only
    /// in the NaN-relaxed fastmath state (fastmath on, rounding "Z", saturate
    /// "NOSAT"), because that is the only lowering that produces a dense result.
    /// \p enableVmiFastmath is required: the state is not a property of the IR
    /// shape, so a defaulted value would grade the same IR differently depending
    /// on the call site.
    LogicalResult getTruncFSupport(VMITruncFOp op, std::string* reason, bool enableVmiFastmath) const;

    LogicalResult getExtFSupport(VMIExtFOp op, std::string* reason = nullptr) const;

    LogicalResult getVUnzipSupport(VMIVUnzipOp op, std::string* reason = nullptr) const;

    LogicalResult getVZipSupport(VMIVZipOp op, std::string* reason = nullptr) const;

    LogicalResult getExtSISupport(VMIExtSIOp op, std::string* reason = nullptr) const;

    LogicalResult getExtUISupport(VMIExtUIOp op, std::string* reason = nullptr) const;

    LogicalResult getTruncISupport(VMITruncIOp op, std::string* reason = nullptr) const;

    FailureOr<VMIBitcastLayoutFact> getBitcastLayoutFact(VMIBitcastOp op, std::string* reason = nullptr) const;

    FailureOr<SmallVector<VMIBitcastLayoutFact, mlir::pto::kValue4>> getBitcastLayoutFactsForLayout(
        VMIVRegType sourceType, VMIVRegType resultType, VMICastLayoutPort port, VMILayoutAttr layout,
        std::string* reason = nullptr) const;

    LogicalResult getBitcastSupport(VMIBitcastOp op, std::string* reason = nullptr) const;

    LogicalResult getVdhistSupport(VMIVdhistOp op, std::string* reason = nullptr) const;

    LogicalResult getVchistSupport(VMIVchistOp op, std::string* reason = nullptr) const;

    /// Whether the operation realizes the op-qualified same-layout relation for
    /// \p layout.  Shared by the relation provider and the cost model so a
    /// layout a lowering rejects is never planned.
    LogicalResult getSameLayoutRelationSupport(
        Operation* op, VMILayoutAttr layout, std::string* reason = nullptr) const;

    //===--------------------------------------------------------------------===//
    // Queries the costed layout solver drives.

    // The planner enumerates *candidate* relations instead of validating one
    // assigned relation, so it needs the full fact set of a table (every row the
    // shape admits) rather than the single row an assigned layout selects.  These
    // queries are additive: no upstream pass calls them, so they cannot widen or
    // narrow any decision upstream's own layout path takes.  Where upstream has a
    // per-layout entry point the body below is reconstructed from it, so the set
    // of legal relations stays defined by upstream's tables and pattern DSL.
    //===--------------------------------------------------------------------===//

    /// Every dense load layout the load table admits for \p resultType's element
    /// type and count, in table order, deduplicated by layout.  Reconstructed
    /// from getLoadLayoutFact over the same kDenseLoadLayoutPatterns rows without
    /// requiring an assigned result layout.
    FailureOr<SmallVector<VMILoadLayoutFact, mlir::pto::kValue4>> getLoadLayoutFacts(
        VMIVRegType resultType, std::string* reason = nullptr) const;

    /// Every dense store layout the store table admits for \p valueType's element
    /// type and count, in table order, deduplicated by layout.  Reconstructed
    /// from getStoreLayoutFact over upstream's kDenseStoreLayoutPatterns without
    /// requiring an assigned value layout.  The fork's three extra preferred
    /// "full-width contiguous" rows are deliberately not injected: for this
    /// enumeration they are duplicates of upstream's {8,16,32}-bit contiguous row,
    /// and upstream's own getPreferredStoreLayoutFact is the only query they would
    /// change.
    FailureOr<SmallVector<VMIStoreLayoutFact, mlir::pto::kValue4>> getStoreLayoutFacts(
        VMIVRegType valueType, std::string* reason = nullptr) const;

    /// Every legal cast relation for the source/result element types, keyed on the
    /// source side.  Reconstructed from getCastLayoutFactsForLayout by driving it
    /// once per source layout the legal cast table can materialize for the width
    /// pair, so the relations are exactly the ones upstream's shared cast funnel
    /// defines (including its CastTypeClass constraint) and upstream's table stays
    /// the single source of truth.

    /// Two fork behaviours are preserved because this entry point belongs to the
    /// solver, not to a shared funnel (plan 11.5): the group count is taken from
    /// whichever side carries a group-slot layout, and a row whose intrinsic
    /// rearrangement cost cannot be derived is not a candidate.
    FailureOr<SmallVector<VMICastLayoutFact, mlir::pto::kValue4>> getCastLayoutFacts(
        VMIVRegType sourceType, VMIVRegType resultType, std::string* reason = nullptr) const;

    /// The cast relation of an equal-storage-width numeric cast: an identity
    /// relation, so it has no row in any cast table.  Upstream has no counterpart;
    /// the body is the fork's (fork VMILayoutSupport.cpp:2444-2471) and the
    /// relation is reachable only from the solver, which uses it for
    /// pto.vmi.fptosi / fptoui / sitofp at equal width.
    FailureOr<VMICastLayoutFact> getSameWidthCastLayoutFact(
        VMIVRegType sourceType, VMIVRegType resultType, std::string* reason = nullptr) const;

    /// Validate the operands of a solver-side cast relation: the operation and
    /// both assigned layouts must be present.  The relation itself is the cast
    /// table's to define, so this entry point carries no layout rule of its own.

    /// This is a solver-side precondition, not a diagnostic.  Upstream has no
    /// counterpart; the body is the fork's (fork
    /// VMILayoutSupport.cpp:2042-2063).
    LogicalResult validateCastOperationRelation(
        Operation* op, VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout, std::string* reason = nullptr) const;

    /// The layout relation of a non-grouped reduce (pto.vmi.reduce_* lowered
    /// through the legacy vcadd / vcmax / vcmin family): the contiguous identity,
    /// bounded by "the contiguous source fills complete physical chunks" so the
    /// planner never advertises a relation the legacy lowering must reject by
    /// folding padding lanes into the result.  Upstream has no counterpart for the
    /// non-grouped reduce family; the body is the fork's (fork
    /// VMILayoutSupport.cpp:1526-1557).
    FailureOr<VMIReduceLayoutFact> getReduceLayoutFactForLayouts(
        VMIVRegType sourceType, VMIMaskType maskType, VMIVRegType resultType, std::string* reason = nullptr) const;

    /// Every interleave layout table row the vector shape admits, in table order.
    /// Reconstructed from getVintlvLayoutFactsForLayout / getVdintlvLayoutFactsForLayout
    /// over upstream's kVintlvLayoutPatterns / kVdintlvLayoutPatterns by dropping
    /// the port-layout filter, so the rows are upstream's and no fork row is
    /// injected.

    /// Upstream's rows are wider than the fork's: they carry 64-bit elements
    /// (bits<8,16,32,64> vs the fork's bits<8,16,32>), and injecting the fork's
    /// narrower versions ahead of them would silently drop 64-bit support and
    /// return duplicate facts (this enumeration has no by-layout dedup, exactly as
    /// in the fork).
    FailureOr<SmallVector<VMIInterleaveLayoutFact, mlir::pto::kValue4>> getVintlvLayoutFacts(
        VMIVRegType valueType, std::string* reason = nullptr) const;
    FailureOr<SmallVector<VMIInterleaveLayoutFact, mlir::pto::kValue4>> getVdintlvLayoutFacts(
        VMIVRegType valueType, std::string* reason = nullptr) const;

    /// Every group_broadcast_load layout table row this operation's shape, group
    /// count, element width and source_group_stride admit, in table order,
    /// deduplicated by result layout.  Reconstructed from the table predicates
    /// upstream's singular getGroupBroadcastLoadLayoutFact already uses
    /// (matchesGroupBlockPattern / matchesElementCountPattern /
    /// matchesElementBitsPattern / matchesGroupBroadcastLoadMemoryPattern) without
    /// that query's assigned-result-layout filter, which is what makes it singular.

    /// Row audit (verified by set diff, fork 11 rows vs upstream 15): the fork has
    /// exactly one unique row, {gb(1), bits<16,32>, memContiguous, d(4)}, and it is
    /// deliberately NOT injected.  That row is not a candidate this enumeration is
    /// missing: upstream's shared gate getGroupBroadcastLoadSupport -- the check the
    /// lowering (VMIToVPTO/VMIToVPTOMemoryInternals.cpp:957) and the IR validator
    /// (Passes/PTOValidateVMIIR.cpp:640) run -- is defined as
    /// "succeeded(getGroupBroadcastLoadLayoutFact(op))"
    /// (VMILayoutSupportQueryHelpers.inc:238-242), i.e. it reads this same table.
    /// Offering the row from the planner would therefore let the costed solver
    /// select a group_broadcast_load relation the lowering rejects, which is the
    /// failure mode this port exists to avoid.  The matching fork-only
    /// kGroupBroadcastLoadDirectPatterns row (E2B, G<8>, gb(1), bits<16,32>,
    /// memContiguous, d(4)) is gone upstream for the same reason, and the planner
    /// pairs this enumeration with that direct fact (fork
    /// VMILayoutPlanner.cpp:2114-2134) to pick the E2B producer layout, so the two
    /// rows can only be restored together -- and only with lowering support, which
    /// belongs to stage 7, not here.
    FailureOr<SmallVector<VMIGroupBroadcastLoadLayoutFact, mlir::pto::kValue4>> getGroupBroadcastLoadLayoutFacts(
        VMIGroupBroadcastLoadOp op, std::string* reason = nullptr) const;

    /// The one layout a group_iota result can carry.  group_iota is the internal
    /// lowering of grouped vci and its physical producer is the contiguous vci
    /// instruction, so the table has a single row; a non-contiguous consumer is
    /// expressed by an explicit ensure_layout edge, and an already-assigned
    /// non-contiguous layout is rejected.  Upstream has no counterpart; the body is
    /// the fork's (fork VMILayoutSupport.cpp:2934-2956).
    FailureOr<SmallVector<VMIGroupIotaLayoutFact, mlir::pto::kValue4>> getGroupIotaLayoutFacts(
        VMIVRegType resultType, std::string* reason = nullptr) const;

    /// The histogram layout the whole table agrees on, ignoring any assigned
    /// layout.  Upstream has no counterpart: getVdhistLayoutFact /
    /// getVchistLayoutFact need the assigned layouts to answer, while the planner
    /// needs the single admissible relation when nothing is assigned yet
    /// (fork VMILayoutPlanner.cpp:2160-2163).  Both ops read the same
    /// kVdhistLayoutPatterns table, as upstream's own singular queries do.  The
    /// bodies are the fork's (fork VMILayoutSupport.cpp:4142-4172).
    FailureOr<VMIHistogramLayoutFact> getPreferredVdhistLayoutFact(VMIVdhistOp op, std::string* reason = nullptr) const;
    FailureOr<VMIHistogramLayoutFact> getPreferredVchistLayoutFact(VMIVchistOp op, std::string* reason = nullptr) const;

    /// Stride-aware group_slot_load fact, the fork's signature.  Upstream's
    /// getGroupSlotLoadLayoutFact(resultType, numGroups, reason) is left untouched;
    /// this overload adds the source_group_stride checks the fork needs and that
    /// upstream's query does not make, so no upstream caller's acceptance changes:
    /// every added check is guarded by "the stride operand is present", so the two
    /// queries agree wherever upstream calls them, and only the solver -- which
    /// always has the operand -- sees the stricter fork rule.

    /// slots = 8 reads one whole group per block load and therefore requires a
    /// constant unit stride; slots = 1 reads each slot from its own element address
    /// with a scalar broadcast load, which needs an 8/16/32-bit element type and a
    /// constant positive stride but has no block-alignment requirement.
    FailureOr<VMIGroupSlotLayoutFact> getGroupSlotLoadLayoutFact(
        VMIVRegType resultType, Value sourceGroupStride, int64_t numGroups, std::string* reason = nullptr) const;

    //===--------------------------------------------------------------------===//
    // Signature reconciliations recorded, deliberately without new code.

    // getPreferredCastLayoutFact: upstream's declaration already ends in a
    // defaulted bool allowLaneStridePreference = true (declared above), so the
    // fork's three-argument call (fork VMILayoutSupport.h:335-337, called at fork
    // VMILayoutPlanner.cpp:1244) compiles unchanged and needs no overload, no
    // default argument and no fork-only variant.  The two implementations also
    // agree in the default configuration: both trees carry the same
    // "vmi-prefer-lane-stride-narrowing" command-line flag defaulting to true
    // (upstream VMILayoutSupport.cpp:87, fork VMILayoutSupport.cpp:54), and with it
    // set both take the one-chunk high-priority / lane-stride-narrow path before
    // the preferred table.  Upstream adds one guard the fork does not have --
    // isDeinterleavedNarrowingFallbackSupported (VMILayoutSupport.cpp:933-999),
    // which keeps the lane-stride relation for shapes whose deinterleaved fallback
    // row has no materialization even when the flag is switched off, instead of
    // switching to a relation the ensure_layout table cannot express.  That is a
    // safety guard, so upstream's version is adopted as is; the fork's preference
    // order is not re-derived.  Preference rows are the solver's last tie-break
    // layer (plan 2.1) and never decide legality, so this changes no candidate set.
    // If the fork's own preference order is ever needed, it belongs in a fork-only
    // entry point the planner drives, not in this shared query.
    // getGroupReduceLayoutFactsForLayout and getPreferredGroupReduceLayoutFact:
    // upstream's take a leading VMIGroupReduceKind (declared above), the fork's do
    // not (fork VMILayoutSupport.h:466-472, 494-498).  No kind-less overload is
    // added, because the kind cannot be guessed at those call sites: it is computed
    // by getVMIGroupReduceKind(op) and its only effect is to swap the result layout
    // of a 16-bit integer group sum to the native i16 row
    // (VMILayoutSupportMaterialization.inc:347-353, upstream 6d744afb5), so a
    // kind-less shim defaulting to Other would silently drop that row for
    // pto.vmi.group_reduce.addi.  When the planner is ported (stage 2c) its five
    // call sites (fork VMILayoutPlanner.cpp:810, 2418, 2444, 2497, 2522) pass
    // getVMIGroupReduceKind(op) as the first argument; for pto.vmi.vcadd / vcmax /
    // vcmin that helper returns Other, which is what upstream itself does for those
    // ops (VMILayoutSupportQueryHelpers.inc:175-225).
    //===--------------------------------------------------------------------===//
};

/// Whether a direct memory-dist access of `family` with token `dist` is legal
/// at this address: the contract must exist for the register's element width and
/// the address must be provably aligned to the contract's alignment rule.

/// The lowering may only emit the direct form when this holds, and the cost model
/// may only charge a layout as directly realized by such an access when it holds,
/// so both sides must ask the same question: this declaration is the shared
/// answer. Sharing it is the point - the model previously mirrored only the
/// *shape* of the direct load form and stayed silent about the address, which
/// made deinterleaved loads whose address is not provably aligned look cheaper
/// than the lowering can make them.
bool isDirectMemoryDistAddressLegal(
    Value base, Value offset, Type addressElementType, VRegType registerType, VPTOMemoryOpFamily family,
    StringRef dist);

/// True when an already assigned BF16/F32 cast pair preserves one of the
/// regular physical lane maps that the pairwise intlv/dintlv lowering can
/// preserve.  This is validation-only: it is not a layout candidate and must
/// not participate in layout assignment or propagation.
bool isBF16SameLayoutCastPair(
    VMIVRegType sourceType, VMIVRegType resultType, VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout);

/// True for an already assigned F32 deinterleaved=4 -> BF16 contiguous
/// narrowing. This is validator-only and is not a layout-solver candidate.
bool isDeinterleaved4ToContiguousBF16CastPair(
    VMIVRegType sourceType, VMIVRegType resultType, VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout);

/// True when the element types and element count name the dense (64/128/256
/// element, non group-slot) F32 -> BF16 narrowing family.  Only this family has
/// a dense same-layout `{c, c}` relation, and whether that relation is real
/// depends on how the cast lowers, not on the width pair: only the NaN-relaxed
/// fastmath lowering produces a dense BF16 result from one physical source
/// part.  The strict per-part vcvt writes the converted lanes at lane_stride 2,
/// so for this family the dense relation is state-qualified and must never be
/// decided by a stateless preference table.
bool isDenseBF16NarrowingCast(VMIVRegType sourceType, VMIVRegType resultType);

/// True for the dense same-layout pair of the family above, i.e. the
/// `{c(1), c(1)}` relation whose result claims one dense BF16 value per lane.
bool isDenseBF16NarrowingSameLayoutPair(
    VMIVRegType sourceType, VMIVRegType resultType, VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout);

/// True when the pair above has one source part per result part.  That is the
/// only shape whose dense lowering is exclusively the NaN-relaxed fastmath
/// path: with twice as many source parts as result parts the paired vcvt form
/// is honest in any state, so it does not require the fastmath condition.
bool requiresBF16NaNRelaxedLowering(
    VMIVRegType sourceType, VMIVRegType resultType, VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout);

/// The same question for a cast whose layouts are not assigned yet: whether the
/// dense same-layout pair of this narrowing family is the matching-arity (K=1)
/// shape, i.e. the only shape whose dense lowering needs the NaN-relaxed
/// fastmath state.  Decided from the dense contiguous layouts both sides start
/// from, so the stateless cast seed can be gated by the same predicate the
/// candidate set uses.
bool requiresBF16NaNRelaxedLowering(VMIVRegType sourceType, VMIVRegType resultType);

/// True when \p op is in the NaN-relaxed fastmath state that realizes the dense
/// BF16 narrowing relation: fastmath enabled plus rounding "Z" and saturate
/// "NOSAT".  Every other state must stay on the lane_stride 2 relation, which
/// the per-part vcvt genuinely implements.
bool hasBF16NaNRelaxedFastmathState(Operation* op, bool enableVmiFastmath);

/// The state clause shared by every diagnostic about the dense BF16 narrowing
/// relation: the required condition and the observed one, quoted consistently so
/// users and CI can grep one string:
/// ```
/// requires vmi-fastmath enabled with rounding="Z" and saturate="NOSAT"; got
/// vmi-fastmath <enabled|disabled>, rounding="<mode>", saturate="<mode>"
/// ```
std::string describeBF16NarrowingState(Operation* op, bool enableVmiFastmath);

/// True when the direct E2B packet backing \p fact can be materialized at the
/// address of \p op.  E2B needs a stricter base-address alignment than the
/// per-group BRC fallback, so layout selection and the implicit ensure-layout
/// expansion must agree with the lowering's E2B address gate.
bool isGroupBroadcastLoadDirectAddressLegal(VMIGroupBroadcastLoadOp op, const VMIGroupBroadcastLoadDirectFact& fact);

/// When the direct E2B layout of \p op cannot be materialized because its base
/// address is not provably aligned, return the contiguous layout the per-group
/// BRC path can lower instead, or null when the E2B layout still applies.
VMILayoutAttr getUnalignedE2BFallbackLayout(
    VMIGroupBroadcastLoadOp op, VMIVRegType type, const VMIGroupBroadcastLoadDirectFact& fact,
    const VMILayoutSupport& supports);

} // namespace mlir::pto

#endif // PTO_TRANSFORMS_VMILAYOUTSUPPORT_H

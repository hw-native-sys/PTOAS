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
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"

#include <string>

namespace mlir::pto {

/// True when `layout` is a group-slot packet that lives in a single physical
/// chunk and puts logical lane i on physical lane i.
///
/// A `num_groups = G, slots = S, lane_stride = 1` layout places logical lane i
/// at {part 0, chunk i / S, lane i % S}.  With `G <= S` every lane lands in
/// chunk 0 at lane i, and `G <= lanesPerPart` keeps that lane inside the
/// carrier, so the value occupies exactly one physical register whose live
/// lanes are 0..G-1.
bool isVMISingleCarrierGroupSlots(VMILayoutAttr layout, int64_t lanesPerPart);

/// Same as isVMISingleCarrierGroupSlots with the lane stride spelled out: the
/// packet places logical lane i at {part 0, chunk i / S, lane (i % S) * LS}.
///
/// A lane stride other than 1 is only meaningful for sub-32-bit elements, whose
/// several values share one carrier lane; a 32-bit element already fills a
/// carrier lane.  The supported strides therefore follow the packable carrier
/// chain (32 -> 16 -> 8): stride 2 up to 16-bit elements, stride 4 up to 8-bit.
///
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
///
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
};

struct VMIEnsureMaskLayoutFact {
  VMILayoutAttr sourceLayout;
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
};

struct VMIMaskGranularityCastLayoutFact {
  VMILayoutAttr sourceLayout;
  VMILayoutAttr resultLayout;
  int64_t sourceGranularityBits = 0;
  int64_t resultGranularityBits = 0;
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

class VMILayoutSupport {
public:
  FailureOr<VMILoadLayoutFact>
  getLoadLayoutFact(VMIVRegType resultType,
                    std::string *reason = nullptr) const;

  FailureOr<VMIDeinterleaveLoadLayoutFact>
  getPreferredDeinterleaveLoadLayoutFact(
      VMIVRegType valueType, std::string *reason = nullptr) const;

  FailureOr<SmallVector<VMIDeinterleaveLoadLayoutFact, mlir::pto::kValue4>>
  getDeinterleaveLoadLayoutFactsForLayout(
      VMIVRegType valueType, VMIDeinterleaveLoadLayoutPort port,
      VMILayoutAttr layout, std::string *reason = nullptr) const;

  FailureOr<VMIDeinterleaveLoadLayoutFact>
  getDeinterleaveLoadLayoutFactForLayouts(
      VMIVRegType lowType, VMIVRegType highType,
      std::string *reason = nullptr) const;

  FailureOr<VMIStoreLayoutFact>
  getStoreLayoutFact(VMIVRegType valueType,
                     std::string *reason = nullptr) const;

  FailureOr<VMIStoreLayoutFact>
  getPreferredStoreLayoutFact(VMIVRegType valueType,
                              std::string *reason = nullptr) const;

  FailureOr<VMIMaskedStoreLayoutFact>
  getMaskedStoreLayoutFact(VMIVRegType valueType, VMIMaskType maskType,
                           std::string *reason = nullptr) const;

  FailureOr<VMIMaskedStoreLayoutFact>
  getPreferredMaskedStoreLayoutFact(VMIVRegType valueType,
                                    VMIMaskType maskType,
                                    std::string *reason = nullptr) const;

  FailureOr<VMIMaskedLoadLayoutFact>
  getMaskedLoadLayoutFact(VMIVRegType resultType, VMIMaskType maskType,
                          VMIVRegType passthruType,
                          std::string *reason = nullptr) const;

  FailureOr<VMIEnsureLayoutFact>
  getEnsureLayoutFact(VMIVRegType sourceType, VMIVRegType resultType,
                      std::string *reason = nullptr) const;

  FailureOr<VMIEnsureMaskLayoutFact>
  getEnsureMaskLayoutFact(VMIMaskType sourceType, VMIMaskType resultType,
                          std::string *reason = nullptr) const;

  FailureOr<VMICastLayoutFact>
  getPreferredCastLayoutFact(VMIVRegType sourceType, VMIVRegType resultType,
                             std::string *reason = nullptr) const;

  // Spine-*scoped* cast layout query.
  //
  // These queries are the only way to reach kSpineScopedCastLayoutPatterns /
  // kSpineScopedLegalCastLayoutPatterns.  Those rows are not part of the
  // generic preferred/legal tables, so a caller that does not explicitly ask
  // for the spine-scoped variant can never observe them - the scope is enforced
  // by construction (the rows are reachable from this entry point only), not by
  // "the candidate happened not to be selected".
  //
  // The direction-spine peephole in VMILayoutAssignment calls this variant for
  // the cast ops of a matched <up,down,up,down> chain whose narrowing leg is
  // handed straight to the closing widening leg.  The scoped table adds the
  // composite 16<->32 and 32<->8 rows (deinterleaved=4 -> deinterleaved=4 with
  // an explicit lane stride) that keep the narrow value split over the four
  // physical parts of its wide side, so the narrowing and the widening are
  // per-chunk 1:1 and need no pto.vor assembly.
  FailureOr<VMICastLayoutFact> getSpineScopedCastLayoutFact(
      VMIVRegType sourceType, VMIVRegType resultType,
      std::string *reason = nullptr) const;

  FailureOr<VMICastLayoutFact> getSpineScopedCastLayoutFactForLayouts(
      VMIVRegType sourceType, VMIVRegType resultType,
      VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
      std::string *reason = nullptr) const;

  FailureOr<SmallVector<VMICastLayoutFact, mlir::pto::kValue4>>
  getSpineScopedCastLayoutFactsForLayout(VMIVRegType sourceType,
                                         VMIVRegType resultType,
                                         VMICastLayoutPort port,
                                         VMILayoutAttr layout,
                                         std::string *reason = nullptr) const;

  FailureOr<SmallVector<VMICastLayoutFact, mlir::pto::kValue4>>
  getCastLayoutFactsForLayout(VMIVRegType sourceType, VMIVRegType resultType,
                              VMICastLayoutPort port, VMILayoutAttr layout,
                              std::string *reason = nullptr) const;

  FailureOr<VMICastLayoutFact> getCastLayoutFactForSourceLayout(
      VMIVRegType sourceType, VMIVRegType resultType,
      VMILayoutAttr sourceLayout, std::string *reason = nullptr) const;

  FailureOr<VMICastLayoutFact> getCastLayoutFactForResultLayout(
      VMIVRegType sourceType, VMIVRegType resultType,
      VMILayoutAttr resultLayout, std::string *reason = nullptr) const;

  FailureOr<VMICastLayoutFact> getCastLayoutFactForLayouts(
      VMIVRegType sourceType, VMIVRegType resultType, VMILayoutAttr sourceLayout,
      VMILayoutAttr resultLayout, std::string *reason = nullptr) const;

  FailureOr<SmallVector<VMIMaskGranularityCastLayoutFact, mlir::pto::kValue4>>
  getMaskGranularityCastLayoutFactsForLayout(
      VMIMaskType sourceType, VMIMaskType resultType, VMICastLayoutPort port,
      VMILayoutAttr layout, std::string *reason = nullptr) const;

  FailureOr<VMIMaskGranularityCastLayoutFact>
  getMaskGranularityCastLayoutFactForLayouts(
      VMIMaskType sourceType, VMIMaskType resultType,
      VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
      std::string *reason = nullptr) const;

  FailureOr<VMILayoutAttr> getWidenSourceLayoutForResultLayout(
      VMIVRegType sourceType, VMIVRegType resultType,
      VMILayoutAttr requestedResultLayout, std::string *reason = nullptr) const;

  FailureOr<VMIInterleaveLayoutFact>
  getPreferredVintlvLayoutFact(VMIVRegType valueType,
                               std::string *reason = nullptr) const;

  FailureOr<VMIInterleaveLayoutFact>
  getPreferredVdintlvLayoutFact(VMIVRegType valueType,
                                std::string *reason = nullptr) const;

  FailureOr<SmallVector<VMIInterleaveLayoutFact, mlir::pto::kValue4>>
  getVintlvLayoutFactsForLayout(VMIVRegType valueType,
                                VMIInterleaveLayoutPort port,
                                VMILayoutAttr layout,
                                std::string *reason = nullptr) const;

  FailureOr<SmallVector<VMIInterleaveLayoutFact, mlir::pto::kValue4>>
  getVdintlvLayoutFactsForLayout(VMIVRegType valueType,
                                 VMIInterleaveLayoutPort port,
                                 VMILayoutAttr layout,
                                 std::string *reason = nullptr) const;

  FailureOr<VMIInterleaveLayoutFact> getVintlvLayoutFactForLayouts(
      VMIVRegType lhsType, VMIVRegType rhsType, VMIMaskType maskType,
      VMIVRegType lowType, VMIVRegType highType,
      std::string *reason = nullptr) const;

  FailureOr<VMIInterleaveLayoutFact> getVdintlvLayoutFactForLayouts(
      VMIVRegType lhsType, VMIVRegType rhsType, VMIMaskType maskType,
      VMIVRegType lowType, VMIVRegType highType,
      std::string *reason = nullptr) const;

  FailureOr<VMIGroupSlotLayoutFact>
  getGroupSlotLoadLayoutFact(VMIVRegType resultType, int64_t numGroups,
                             std::string *reason = nullptr) const;

  FailureOr<VMIGroupLoadLayoutFact>
  getGroupLoadLayoutFact(VMIGroupLoadOp op,
                         std::string *reason = nullptr) const;
  FailureOr<VMIGroupLoadLayoutFact>
  getGroupLoadLayoutFact(VMIVRegType resultType, Value rowStride,
                         int64_t numGroups,
                         std::string *reason = nullptr) const;

  FailureOr<VMIGroupSlotLayoutFact>
  getGroupStoreLayoutFact(VMIVRegType valueType, int64_t numGroups,
                          std::string *reason = nullptr) const;

  FailureOr<VMIGroupStoreLayoutFact>
  getGroupStoreLayoutFact(VMIGroupStoreOp op, VMIVRegType valueType,
                          std::string *reason = nullptr) const;

  FailureOr<SmallVector<VMIGroupStoreLayoutFact, mlir::pto::kValue4>>
  getGroupStoreLayoutFactsForLayout(VMIGroupStoreOp op,
                                    VMIVRegType valueType,
                                    VMILayoutAttr layout,
                                    std::string *reason = nullptr) const;

  FailureOr<VMIGroupStoreLayoutFact>
  getPreferredGroupStoreLayoutFact(VMIGroupStoreOp op, VMIVRegType valueType,
                                   std::string *reason = nullptr) const;

  FailureOr<VMIGroupStoreLayoutFact>
  getHighPriorityGroupStoreLayoutFact(VMIGroupStoreOp op,
                                      VMIVRegType valueType,
                                      std::string *reason = nullptr) const;

  LogicalResult getGroupOperationShapeSupport(Operation *op,
                                               std::string *reason = nullptr) const;

  // Shape-only capability checks for diagnostics before layout assignment.
  // Assigned source layouts are respected; lowering checks the full fact again.
  LogicalResult getGroupReduceShapeSupport(VMIVRegType sourceType,
                                           int64_t numGroups,
                                           std::string *reason = nullptr) const;
  FailureOr<VMIGroupBroadcastLayoutFact>
  getPreferredGroupBroadcastLayoutFact(VMIVRegType resultType, int64_t numGroups,
                                       std::string *reason = nullptr) const;
  VMILayoutAttr getPreferredGroupBroadcastResultLayout(
      VMIVRegType resultType, int64_t numGroups) const;
  LogicalResult getGroupBroadcastShapeSupport(VMIVRegType resultType,
                                              int64_t numGroups,
                                              std::string *reason = nullptr) const;

  FailureOr<VMIGroupReduceLayoutFact>
  getPreferredGroupReduceLayoutFact(VMIVRegType sourceType, int64_t numGroups,
                                    std::string *reason = nullptr) const;

  FailureOr<VMIGroupReduceLayoutFact> getGroupReduceLayoutFactForLayouts(
      VMIVRegType sourceType, VMIMaskType maskType, VMIVRegType resultType,
      int64_t numGroups, std::string *reason = nullptr) const;

  FailureOr<SmallVector<VMIGroupReduceLayoutFact, mlir::pto::kValue4>>
  getGroupReduceLayoutFactsForLayout(VMIVRegType sourceType,
                                     int64_t numGroups,
                                     VMIGroupReduceLayoutPort port,
                                     VMILayoutAttr layout,
                                     std::string *reason = nullptr) const;

  FailureOr<VMIGroupBroadcastLayoutFact>
  getGroupBroadcastLayoutFactForLayouts(VMIVRegType sourceType,
                                        VMIVRegType resultType,
                                        int64_t numGroups,
                                        std::string *reason = nullptr) const;

  FailureOr<SmallVector<VMIGroupBroadcastLayoutFact, mlir::pto::kValue4>>
  getGroupBroadcastLayoutFactsForLayout(VMIVRegType sourceType,
                                        VMIVRegType resultType,
                                        int64_t numGroups,
                                        VMIGroupBroadcastLayoutPort port,
                                        VMILayoutAttr layout,
                                        std::string *reason = nullptr) const;

  FailureOr<VMIGroupBroadcastLoadLayoutFact>
  getGroupBroadcastLoadLayoutFact(VMIGroupBroadcastLoadOp op,
                                  std::string *reason = nullptr) const;
  FailureOr<VMIGroupBroadcastLoadLayoutFact>
  getGroupBroadcastLoadLayoutFact(VMIVRegType resultType,
                                  Value sourceGroupStride, int64_t numGroups,
                                  std::string *reason = nullptr) const;
  FailureOr<VMIGroupBroadcastLoadDirectFact> getGroupBroadcastLoadDirectFact(
      VMIGroupBroadcastLoadOp op, std::string *reason = nullptr) const;
  FailureOr<VMIGroupBroadcastLoadDirectFact> getGroupBroadcastLoadDirectFact(
      VMIVRegType resultType, Type sourceType, Value sourceGroupStride,
      int64_t numGroups, std::string *reason = nullptr) const;

  FailureOr<VMIHistogramLayoutFact>
  getVdhistLayoutFact(VMIVdhistOp op, std::string *reason = nullptr) const;

  FailureOr<VMIHistogramLayoutFact>
  getVchistLayoutFact(VMIVchistOp op, std::string *reason = nullptr) const;

  FailureOr<VMIVselrLayoutFact>
  getPreferredVselrLayoutFact(VMIVselrOp op,
                              std::string *reason = nullptr) const;

  FailureOr<VMIVselrLayoutFact>
  getVselrLayoutFact(VMIVselrOp op,
                     std::string *reason = nullptr) const;

  LogicalResult getVselrSupport(VMIVselrOp op,
                                std::string *reason = nullptr) const;

  LogicalResult getGroupReduceAddFSupport(VMIGroupReduceAddFOp op,
                                          std::string *reason = nullptr) const;

  LogicalResult getGroupReduceMaxFSupport(VMIGroupReduceMaxFOp op,
                                          std::string *reason = nullptr) const;

  LogicalResult getGroupReduceMinFSupport(VMIGroupReduceMinFOp op,
                                          std::string *reason = nullptr) const;

  LogicalResult getGroupReduceAddISupport(VMIGroupReduceAddIOp op,
                                          std::string *reason = nullptr) const;

  LogicalResult getGroupReduceMaxISupport(VMIGroupReduceMaxIOp op,
                                          std::string *reason = nullptr) const;

  LogicalResult getGroupReduceMinISupport(VMIGroupReduceMinIOp op,
                                          std::string *reason = nullptr) const;

  LogicalResult getGroupBroadcastSupport(VMIGroupBroadcastOp op,
                                         std::string *reason = nullptr) const;

  LogicalResult getGroupBroadcastSupport(VMIVRegType sourceType,
                                         VMIVRegType resultType,
                                         int64_t numGroups,
                                         std::string *reason = nullptr) const;

  LogicalResult getGroupBroadcastLoadSupport(
      VMIGroupBroadcastLoadOp op, std::string *reason = nullptr) const;

  LogicalResult getTruncFSupport(VMITruncFOp op,
                                 std::string *reason = nullptr) const;

  LogicalResult getExtFSupport(VMIExtFOp op,
                               std::string *reason = nullptr) const;

  LogicalResult getExtSISupport(VMIExtSIOp op,
                                std::string *reason = nullptr) const;

  LogicalResult getExtUISupport(VMIExtUIOp op,
                                std::string *reason = nullptr) const;

  LogicalResult getTruncISupport(VMITruncIOp op,
                                 std::string *reason = nullptr) const;

  FailureOr<VMIBitcastLayoutFact>
  getBitcastLayoutFact(VMIBitcastOp op,
                       std::string *reason = nullptr) const;

  FailureOr<SmallVector<VMIBitcastLayoutFact, mlir::pto::kValue4>>
  getBitcastLayoutFactsForLayout(VMIVRegType sourceType,
                                 VMIVRegType resultType,
                                 VMICastLayoutPort port,
                                 VMILayoutAttr layout,
                                 std::string *reason = nullptr) const;

  LogicalResult getBitcastSupport(VMIBitcastOp op,
                                  std::string *reason = nullptr) const;

  LogicalResult getVdhistSupport(VMIVdhistOp op,
                                std::string *reason = nullptr) const;

  LogicalResult getVchistSupport(VMIVchistOp op,
                                std::string *reason = nullptr) const;
};

} // namespace mlir::pto

#endif // PTO_TRANSFORMS_VMILAYOUTSUPPORT_H

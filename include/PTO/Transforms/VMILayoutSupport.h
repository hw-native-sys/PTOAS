// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- VMILayoutSupport.h - VMI layout support queries ------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#ifndef PTO_TRANSFORMS_VMILAYOUTSUPPORT_H
#define PTO_TRANSFORMS_VMILAYOUTSUPPORT_H

#include "PTO/IR/PTO.h"
#include "PTO/Support/CodeConstants.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"

#include <string>

namespace mlir::pto {

struct VMILoadLayoutFact {
  VMILayoutAttr resultLayout;
};

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
  bool forwardsPhysicalParts = false;
};

struct VMIEnsureMaskLayoutFact {
  VMILayoutAttr sourceLayout;
  VMILayoutAttr resultLayout;
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
  QuarterBlock,
  HalfBlock,
  OneBlock,
  TwoBlock,
  FourBlock,
  FullPartMultiple,
};

struct VMIGroupStoreLayoutFact {
  VMILayoutAttr valueLayout;
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
  FailureOr<SmallVector<VMILoadLayoutFact, mlir::pto::kValue4>>
  getLoadLayoutFacts(VMIVRegType resultType,
                     std::string *reason = nullptr) const;

  FailureOr<SmallVector<VMIGroupIotaLayoutFact, mlir::pto::kValue4>>
  getGroupIotaLayoutFacts(VMIVRegType resultType,
                          std::string *reason = nullptr) const;

  FailureOr<VMILoadLayoutFact>
  getLoadLayoutFact(VMIVRegType resultType,
                    std::string *reason = nullptr) const;

  FailureOr<VMIDeinterleaveLoadLayoutFact>
  getPreferredDeinterleaveLoadLayoutFact(VMIVRegType valueType,
                                         std::string *reason = nullptr) const;

  FailureOr<SmallVector<VMIDeinterleaveLoadLayoutFact, mlir::pto::kValue4>>
  getDeinterleaveLoadLayoutFactsForLayout(VMIVRegType valueType,
                                          VMIDeinterleaveLoadLayoutPort port,
                                          VMILayoutAttr layout,
                                          std::string *reason = nullptr) const;

  FailureOr<VMIDeinterleaveLoadLayoutFact>
  getDeinterleaveLoadLayoutFactForLayouts(VMIVRegType lowType,
                                          VMIVRegType highType,
                                          std::string *reason = nullptr) const;

  FailureOr<VMIStoreLayoutFact>
  getStoreLayoutFact(VMIVRegType valueType,
                     std::string *reason = nullptr) const;

  FailureOr<SmallVector<VMIStoreLayoutFact, mlir::pto::kValue4>>
  getStoreLayoutFacts(VMIVRegType valueType,
                      std::string *reason = nullptr) const;

  FailureOr<VMIStoreLayoutFact>
  getPreferredStoreLayoutFact(VMIVRegType valueType,
                              std::string *reason = nullptr) const;

  FailureOr<VMIMaskedStoreLayoutFact>
  getMaskedStoreLayoutFact(VMIVRegType valueType, VMIMaskType maskType,
                           std::string *reason = nullptr) const;

  FailureOr<VMIMaskedStoreLayoutFact>
  getPreferredMaskedStoreLayoutFact(VMIVRegType valueType, VMIMaskType maskType,
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

  FailureOr<VMIGeneratedMaskLayoutFact>
  getGeneratedMaskLayoutFact(Operation *op, VMILayoutAttr resultLayout,
                             std::string *reason = nullptr) const;

  FailureOr<VMICastLayoutFact>
  getPreferredCastLayoutFact(VMIVRegType sourceType, VMIVRegType resultType,
                             std::string *reason = nullptr) const;

  FailureOr<SmallVector<VMICastLayoutFact, mlir::pto::kValue4>>
  getCastLayoutFacts(VMIVRegType sourceType, VMIVRegType resultType,
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

  FailureOr<VMICastLayoutFact>
  getCastLayoutFactForLayouts(VMIVRegType sourceType, VMIVRegType resultType,
                              VMILayoutAttr sourceLayout,
                              VMILayoutAttr resultLayout,
                              std::string *reason = nullptr) const;

  // Validate operation-family capabilities that are not expressible by the
  // storage-width/layout relation alone.  This is shared by planning and
  // lowering so a legal relation cannot be exposed to one and rejected by the
  // other.
  LogicalResult
  validateCastOperationRelation(Operation *op, VMILayoutAttr sourceLayout,
                                VMILayoutAttr resultLayout,
                                std::string *reason = nullptr) const;

  FailureOr<VMICastLayoutFact>
  getSameWidthCastLayoutFact(VMIVRegType sourceType, VMIVRegType resultType,
                             std::string *reason = nullptr) const;

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
  FailureOr<SmallVector<VMIInterleaveLayoutFact, mlir::pto::kValue4>>
  getVintlvLayoutFacts(VMIVRegType valueType,
                       std::string *reason = nullptr) const;

  FailureOr<VMIInterleaveLayoutFact>
  getPreferredVdintlvLayoutFact(VMIVRegType valueType,
                                std::string *reason = nullptr) const;
  FailureOr<SmallVector<VMIInterleaveLayoutFact, mlir::pto::kValue4>>
  getVdintlvLayoutFacts(VMIVRegType valueType,
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

  FailureOr<VMIInterleaveLayoutFact>
  getVintlvLayoutFactForLayouts(VMIVRegType lhsType, VMIVRegType rhsType,
                                VMIMaskType maskType, VMIVRegType lowType,
                                VMIVRegType highType,
                                std::string *reason = nullptr) const;

  FailureOr<VMIInterleaveLayoutFact>
  getVdintlvLayoutFactForLayouts(VMIVRegType lhsType, VMIVRegType rhsType,
                                 VMIMaskType maskType, VMIVRegType lowType,
                                 VMIVRegType highType,
                                 std::string *reason = nullptr) const;

  FailureOr<VMIGroupSlotLayoutFact>
  getGroupSlotLoadLayoutFact(VMIVRegType resultType, Value sourceGroupStride,
                             int64_t numGroups,
                             std::string *reason = nullptr) const;

  FailureOr<VMIInterleaveStoreSupport>
  getInterleaveStoreSupport(VMIVRegType lowType, VMIVRegType highType,
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
  getGroupStoreLayoutFactsForLayout(VMIGroupStoreOp op, VMIVRegType valueType,
                                    VMILayoutAttr layout,
                                    std::string *reason = nullptr) const;

  FailureOr<VMIGroupStoreLayoutFact>
  getPreferredGroupStoreLayoutFact(VMIGroupStoreOp op, VMIVRegType valueType,
                                   std::string *reason = nullptr) const;

  FailureOr<VMIGroupStoreLayoutFact>
  getHighPriorityGroupStoreLayoutFact(VMIGroupStoreOp op, VMIVRegType valueType,
                                      std::string *reason = nullptr) const;

  FailureOr<VMIGroupReduceLayoutFact>
  getPreferredGroupReduceLayoutFact(VMIVRegType sourceType, int64_t numGroups,
                                    std::string *reason = nullptr) const;

  FailureOr<VMIGroupReduceLayoutFact> getGroupReduceLayoutFactForLayouts(
      VMIVRegType sourceType, VMIMaskType maskType, VMIVRegType resultType,
      int64_t numGroups, std::string *reason = nullptr) const;

  FailureOr<VMIReduceLayoutFact>
  getReduceLayoutFactForLayouts(VMIVRegType sourceType, VMIMaskType maskType,
                                VMIVRegType resultType,
                                std::string *reason = nullptr) const;

  /// Returns failure when a contiguous source value does not fill its physical
  /// chunks, so its padding lanes cannot be told apart from real data.  The
  /// reduce family and the compress legality check share this rule; the reason
  /// is a neutral requirement fragment because each op family prefixes its own
  /// subject rather than borrowing the other's wording.
  LogicalResult checkSourceFillsPhysicalChunks(VMIVRegType sourceType,
                                               std::string *reason = nullptr) const;

  /// Returns true when the scalar broadcast load path can read this element
  /// width.  A slots=1 group slot holds one group value that a consumer
  /// broadcasts, so it materializes as one scalar broadcast load per slot.
  /// The planner and the lowering both gate on this predicate, so keep the
  /// element width rule in this single definition.
  static bool isScalarBroadcastLoadElementType(Type elementType);

  FailureOr<SmallVector<VMIGroupReduceLayoutFact, mlir::pto::kValue4>>
  getGroupReduceLayoutFactsForLayout(VMIVRegType sourceType, int64_t numGroups,
                                     VMIGroupReduceLayoutPort port,
                                     VMILayoutAttr layout,
                                     std::string *reason = nullptr) const;

  FailureOr<VMIGroupBroadcastLayoutFact> getGroupBroadcastLayoutFactForLayouts(
      VMIVRegType sourceType, VMIVRegType resultType, int64_t numGroups,
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
  FailureOr<SmallVector<VMIGroupBroadcastLoadLayoutFact, mlir::pto::kValue4>>
  getGroupBroadcastLoadLayoutFacts(VMIGroupBroadcastLoadOp op,
                                   std::string *reason = nullptr) const;
  FailureOr<VMIGroupBroadcastLoadLayoutFact>
  getGroupBroadcastLoadLayoutFact(VMIVRegType resultType,
                                  Value sourceGroupStride, int64_t numGroups,
                                  std::string *reason = nullptr) const;
  FailureOr<VMIGroupBroadcastLoadDirectFact>
  getGroupBroadcastLoadDirectFact(VMIGroupBroadcastLoadOp op,
                                  std::string *reason = nullptr) const;
  FailureOr<VMIGroupBroadcastLoadDirectFact>
  getGroupBroadcastLoadDirectFact(VMIVRegType resultType, Type sourceType,
                                  Value sourceGroupStride, int64_t numGroups,
                                  std::string *reason = nullptr) const;

  FailureOr<VMIHistogramLayoutFact>
  getPreferredVdhistLayoutFact(VMIVdhistOp op,
                               std::string *reason = nullptr) const;

  FailureOr<VMIHistogramLayoutFact>
  getPreferredVchistLayoutFact(VMIVchistOp op,
                               std::string *reason = nullptr) const;

  FailureOr<VMIHistogramLayoutFact>
  getVdhistLayoutFact(VMIVdhistOp op, std::string *reason = nullptr) const;

  FailureOr<VMIHistogramLayoutFact>
  getVchistLayoutFact(VMIVchistOp op, std::string *reason = nullptr) const;

  FailureOr<VMIVselrLayoutFact>
  getPreferredVselrLayoutFact(VMIVselrOp op,
                              std::string *reason = nullptr) const;

  FailureOr<VMIVselrLayoutFact>
  getVselrLayoutFact(VMIVselrOp op, std::string *reason = nullptr) const;

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

  LogicalResult
  getGroupBroadcastLoadSupport(VMIGroupBroadcastLoadOp op,
                               std::string *reason = nullptr) const;

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
  getBitcastLayoutFact(VMIBitcastOp op, std::string *reason = nullptr) const;

  FailureOr<SmallVector<VMIBitcastLayoutFact, mlir::pto::kValue4>>
  getBitcastLayoutFactsForLayout(VMIVRegType sourceType, VMIVRegType resultType,
                                 VMICastLayoutPort port, VMILayoutAttr layout,
                                 std::string *reason = nullptr) const;

  LogicalResult getBitcastSupport(VMIBitcastOp op,
                                  std::string *reason = nullptr) const;

  LogicalResult getVdhistSupport(VMIVdhistOp op,
                                 std::string *reason = nullptr) const;

  LogicalResult getVchistSupport(VMIVchistOp op,
                                 std::string *reason = nullptr) const;

  /// Facts for every vexpdif table row whose source (or result) layout is
  /// p layout.  Every returned fact is a relation the VPTO lowering can
  /// realize for this operation's shape, so a planner that enumerates only
  /// these facts cannot select an unlowerable plan.
  FailureOr<SmallVector<VMIVexpdifLayoutFact, mlir::pto::kValue4>>
  getVexpdifLayoutFactsForLayout(VMIVexpdifOp op, VMIVexpdifLayoutPort port,
                                 VMILayoutAttr layout,
                                 std::string *reason = nullptr) const;

  /// The vexpdif table row this shape prefers.  Plans may pick another row,
  /// which the planner charges as a layout preference penalty.
  FailureOr<VMIVexpdifLayoutFact>
  getPreferredVexpdifLayoutFact(VMIVexpdifOp op,
                                std::string *reason = nullptr) const;

  LogicalResult
  getSameLayoutRelationSupport(Operation *op, VMILayoutAttr layout,
                               std::string *reason = nullptr) const;
};

} // namespace mlir::pto

#endif // PTO_TRANSFORMS_VMILAYOUTSUPPORT_H

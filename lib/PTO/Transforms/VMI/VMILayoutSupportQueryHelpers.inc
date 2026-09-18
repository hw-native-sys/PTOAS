// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef VMILAYOUTSUPPORTQUERYHELPERS_INC
#define VMILAYOUTSUPPORTQUERYHELPERS_INC

//===- VMILayoutSupportQueryHelpers.inc - VMI query helpers -*- C++ -*-===//
//===----------------------------------------------------------------------===//

namespace {

struct GroupBroadcastLoadQuery {
  VMIVRegType resultType;
  VMILayoutAttr resultLayout;
  std::optional<int64_t> stride;
  GroupLayoutKey key;
  int64_t numGroups;
  unsigned elementBits;
};

inline bool matchesGroupBroadcastLoadPattern(
    const GroupBroadcastLoadLayoutPattern &pattern,
    const GroupBroadcastLoadQuery &query) {
  return matchesGroupBlockPattern(pattern.block, query.key) &&
         matchesElementCountPattern(pattern.numGroups, query.numGroups) &&
         matchesElementBitsPattern(pattern.elementBits, query.elementBits) &&
         matchesGroupBroadcastLoadMemoryPattern(pattern.memory, query.stride,
                                                query.elementBits) &&
         matchesLayoutPattern(query.resultType.getContext(),
                              pattern.resultLayout, query.resultLayout,
                              query.numGroups);
}

inline VMIGroupBroadcastLoadLayoutFact materializeGroupBroadcastLoadFact(
    const GroupBroadcastLoadLayoutPattern &pattern,
    const GroupBroadcastLoadQuery &query) {
  return {getGroupBlockClassFromPattern(pattern.block),
          query.resultLayout,
          query.key.groupSize,
          query.key.lanesPerPart,
          query.key.vcgBlockElems,
          static_cast<int64_t>(query.elementBits)};
}

inline bool matchesGroupBroadcastLoadDirectPattern(
    const GroupBroadcastLoadDirectPattern &pattern,
    const GroupBroadcastLoadQuery &query) {
  return matchesElementCountPattern(pattern.numGroups, query.numGroups) &&
         matchesGroupBlockPattern(pattern.block, query.key) &&
         matchesElementBitsPattern(pattern.elementBits, query.elementBits) &&
         matchesGroupBroadcastLoadMemoryPattern(pattern.memory, query.stride,
                                                query.elementBits);
}

struct GroupStoreMaterializationQuery {
  VMIVRegType valueType;
  VMIVRegType sourceType;
  GroupLayoutKey key;
  std::optional<int64_t> rowStride;
};

inline void appendMaterializableGroupStoreFacts(
    const VMILayoutSupport &support,
    const GroupStoreMaterializationQuery &query,
    SmallVectorImpl<VMIGroupStoreLayoutFact> &facts) {
  MLIRContext *ctx = query.valueType.getContext();
  for (const GroupStoreLayoutPattern &pattern : kGroupStoreLayoutPatterns) {
    if (!matchesGroupStoreLayoutPattern(pattern, query.valueType, query.key,
                                        query.rowStride)) {
      continue;
    }
    VMILayoutAttr useLayout =
        materializeLayoutPattern(ctx, pattern.valueLayout);
    bool duplicate = llvm::any_of(
        facts, [useLayout](const VMIGroupStoreLayoutFact &fact) {
          return fact.valueLayout == useLayout;
        });
    if (!useLayout || duplicate) {
      continue;
    }
    auto useType = VMIVRegType::get(
        ctx, query.valueType.getElementCount(),
        query.valueType.getElementType(), useLayout);
    if (failed(support.getEnsureLayoutFact(query.sourceType, useType,
                                           nullptr))) {
      continue;
    }
    VMIGroupStoreLayoutFact fact =
        materializeGroupStoreLayoutFact(ctx, pattern, query.key);
    fact.valueLayout = useLayout;
    facts.push_back(fact);
  }
}

inline std::optional<VMIGroupStoreLayoutFact>
getPreferredGroupSlotStoreFact(const VMILayoutSupport &support,
                               VMIGroupStoreOp op, VMIVRegType valueType,
                               int64_t numGroups) {
  bool hasOneLanePerGroup = valueType.getElementCount() == numGroups;
  if (!hasOneLanePerGroup) {
    return std::nullopt;
  }
  std::optional<int64_t> rowStride = getConstantIndexValue(op.getRowStride());
  // A unit row stride writes the group values back to back, which is what a
  // dense short vector already holds in one carrier, so prefer the packed slot
  // form for it.  One carrier per group stays as the fallback: a group reduce
  // over whole parts really does produce one part per group, and only the
  // slots = 1 group_store lowering can pack that back together.
  SmallVector<int64_t, mlir::pto::kValue2> slotCandidates;
  if (rowStride && *rowStride == 1) {
    slotCandidates.push_back(mlir::pto::kValue8);
  }
  slotCandidates.push_back(1);
  for (int64_t slots : slotCandidates) {
    VMILayoutAttr layout = VMILayoutAttr::getGroupSlots(
        valueType.getContext(), numGroups, slots);
    auto assignedType = VMIVRegType::get(
        valueType.getContext(), valueType.getElementCount(),
        valueType.getElementType(), layout);
    if (succeeded(support.getGroupStoreLayoutFact(op, assignedType,
                                                  nullptr))) {
      return VMIGroupStoreLayoutFact{layout};
    }
  }
  return std::nullopt;
}

inline const GroupStoreLayoutPattern *selectPreferredGroupStorePattern(
    VMIVRegType valueType, GroupLayoutKey key,
    std::optional<int64_t> rowStride) {
  const GroupStoreLayoutPattern *selected = nullptr;
  for (const GroupStoreLayoutPattern &pattern : kGroupStoreLayoutPatterns) {
    if (pattern.priority == GroupStoreLayoutPriority::LegalOnly ||
        !matchesGroupStoreLayoutPattern(pattern, valueType, key, rowStride)) {
      continue;
    }
    bool hasHigherPriority =
        !selected || static_cast<unsigned>(pattern.priority) >
                         static_cast<unsigned>(selected->priority);
    if (hasHigherPriority) {
      selected = &pattern;
    }
  }
  return selected;
}

// Pick the unique cast layout fact whose result layout equals \p resultLayout
// from a successful source-side cast query.  The generic and the spine-scoped
// cast reconciliation share this step; the caller supplies the diagnostics.
static FailureOr<VMICastLayoutFact> selectCastLayoutFactForResultLayout(
    ArrayRef<VMICastLayoutFact> facts, VMILayoutAttr resultLayout,
    StringRef ambiguousMessage, StringRef noMatchMessage,
    std::string *reason) {
  std::optional<VMICastLayoutFact> selected;
  for (const VMICastLayoutFact &fact : facts) {
    if (fact.resultLayout != resultLayout) {
      continue;
    }
    if (selected) {
      if (reason) {
        *reason = ambiguousMessage.str();
      }
      return failure();
    }
    selected = fact;
  }
  if (!selected) {
    if (reason) {
      *reason = noMatchMessage.str();
    }
    return failure();
  }
  return *selected;
}

} // namespace

//===----------------------------------------------------------------------===//
// Single-carrier group packet predicates
//===----------------------------------------------------------------------===//

namespace {

/// Carrier widths the narrowing pack chain can express: the packs are 32 -> 16
/// and 16 -> 8, so only 8/16/32-bit carriers have a form to pack or unpack with.
constexpr unsigned kCarrierBits8 = 8;
constexpr unsigned kCarrierBits16 = 16;
constexpr unsigned kCarrierBits32 = 32;

/// Lane strides of a dense lane-strided value: two or four element lanes share
/// the carrier a single unit-stride lane occupies.
constexpr int64_t kLaneStridePair = 2;
constexpr int64_t kLaneStrideQuad = 4;

/// True for a carrier width the narrowing pack chain can express.
static bool isPackableCarrierBits(unsigned bits) {
  return bits == kCarrierBits8 || bits == kCarrierBits16 ||
         bits == kCarrierBits32;
}

/// True when a dense lane-strided value of `elementBits`-wide elements has a
/// packable carrier at `laneStride`: both the element carrier and the
/// lane-strided carrier (element bits times the stride) have to be expressible.
static bool isPackableLaneStride(unsigned elementBits, int64_t laneStride) {
  if (laneStride == 1) {
    return isPackableCarrierBits(elementBits);
  }
  if (laneStride != kLaneStridePair && laneStride != kLaneStrideQuad) {
    return false;
  }
  return isPackableCarrierBits(elementBits *
                               static_cast<unsigned>(laneStride));
}

} // namespace

bool isVMISingleCarrierGroupSlotsWithStride(VMILayoutAttr layout,
                                            int64_t lanesPerPart,
                                            int64_t laneStride) {
  bool usablePacket = layout && layout.isGroupSlots() && lanesPerPart > 0 &&
                      laneStride > 0;
  if (!usablePacket) {
    return false;
  }
  int64_t numGroups = layout.getNumGroups();
  int64_t slots = layout.getSlots();
  // The group-slot forms the lowering builds carry eight group slots per part
  // (or one), and a packet with more slots is hand-written IR whose groups are
  // spread over several carriers, so it is not a single-carrier packet.
  bool usablePacketForm =
      slots > 0 && slots <= kValue8 && layout.getLaneStride() == laneStride &&
      numGroups <= slots && numGroups <= lanesPerPart;
  if (!usablePacketForm) {
    return false;
  }
  // Group g sits at lane (g % slots) * laneStride, so the lane stride widens the
  // packet's footprint: the last group still has to land inside one carrier.
  // The divisional form keeps an out-of-range attribute from overflowing the
  // product (numGroups - 1 <= (lanesPerPart - 1) / laneStride is equivalent for
  // the non-negative values admitted above).
  return numGroups - 1 <= (lanesPerPart - 1) / laneStride;
}

bool isVMISingleCarrierGroupSlots(VMILayoutAttr layout, int64_t lanesPerPart) {
  return isVMISingleCarrierGroupSlotsWithStride(layout, lanesPerPart, 1);
}

bool needsVMIDenseLaneStrideGroupSlotBridge(VMILayoutAttr sourceLayout,
                                            VMILayoutAttr resultLayout,
                                            Type elementType,
                                            int64_t lanesPerPart) {
  if (!sourceLayout || !resultLayout) {
    return false;
  }
  bool sourceIsPacket = isVMISingleCarrierGroupSlotsWithStride(
      sourceLayout, lanesPerPart, sourceLayout.getLaneStride());
  bool resultIsPacket = isVMISingleCarrierGroupSlotsWithStride(
      resultLayout, lanesPerPart, resultLayout.getLaneStride());
  bool denseToPacket = sourceLayout.isContiguous() && resultIsPacket;
  bool packetToDense = sourceIsPacket && resultLayout.isContiguous();
  if (!denseToPacket && !packetToDense) {
    return false;
  }
  int64_t sourceStride = sourceLayout.getLaneStride();
  int64_t resultStride = resultLayout.getLaneStride();
  if (sourceStride == resultStride) {
    // Same lane stride: the carrier identity already covers the pair.
    return false;
  }

  // Two strides that are both non-unit would need two dense steps, which no
  // materialization covers: only unit <-> stride moves exist, so such a pair is
  // rejected here rather than admitted and left residual by the lowering.
  if (sourceStride != 1 && resultStride != 1) {
    return false;
  }

  // The dense side moves between the two lane strides through the dense
  // lane-stride materialization, whose carrier chain has to stay expressible:
  // both the element carrier and the lane-strided carrier have to be packable
  // widths, i.e. 8/16/32 bits (there is no 64-bit pack form).
  unsigned elementBits = pto::getPTOStorageElemBitWidth(elementType);
  return isPackableLaneStride(elementBits, sourceStride) &&
         isPackableLaneStride(elementBits, resultStride);
}

bool isVMISingleCarrierGroupSlotAlias(VMILayoutAttr lhs, VMILayoutAttr rhs,
                                      int64_t lanesPerPart) {
  // A single-carrier packet places group g at lane (g % slots) * lane_stride
  // and a dense contiguous value places logical lane i at lane i * lane_stride,
  // so both describe the same carrier lanes whenever the lane strides agree.
  // The stride is part of the shared mapping, not a condition for the identity.
  auto isCarrierPair = [lanesPerPart](VMILayoutAttr packet,
                                      VMILayoutAttr dense) {
    return dense && dense.isContiguous() &&
           isVMISingleCarrierGroupSlotsWithStride(packet, lanesPerPart,
                                                  dense.getLaneStride());
  };
  return isCarrierPair(lhs, rhs) || isCarrierPair(rhs, lhs);
}
//===----------------------------------------------------------------------===//
// Support-query adapters
//===----------------------------------------------------------------------===//

inline LogicalResult getGroupReduceAddSupportImpl(VMIVRegType sourceType,
                                                  VMIMaskType maskType,
                                                  VMIVRegType resultType,
                                                  int64_t numGroups,
                                                  std::string *reason) {
  FailureOr<VMIGroupReduceLayoutFact> fact =
      VMILayoutSupport().getGroupReduceLayoutFactForLayouts(
          sourceType, maskType, resultType, numGroups, reason);
  if (failed(fact)) {
    return failure();
  }
  return success();
}

LogicalResult
VMILayoutSupport::getGroupReduceAddFSupport(VMIGroupReduceAddFOp op,
                                            std::string *reason) const {
  return getGroupReduceAddSupportImpl(
      cast<VMIVRegType>(op.getSource().getType()),
      cast<VMIMaskType>(op.getMask().getType()),
      cast<VMIVRegType>(op.getResult().getType()),
      op.getNumGroupsAttr().getInt(), reason);
}

LogicalResult
VMILayoutSupport::getGroupReduceMaxFSupport(VMIGroupReduceMaxFOp op,
                                            std::string *reason) const {
  return getGroupReduceAddSupportImpl(
      cast<VMIVRegType>(op.getSource().getType()),
      cast<VMIMaskType>(op.getMask().getType()),
      cast<VMIVRegType>(op.getResult().getType()),
      op.getNumGroupsAttr().getInt(), reason);
}

LogicalResult
VMILayoutSupport::getGroupReduceMinFSupport(VMIGroupReduceMinFOp op,
                                            std::string *reason) const {
  return getGroupReduceAddSupportImpl(
      cast<VMIVRegType>(op.getSource().getType()),
      cast<VMIMaskType>(op.getMask().getType()),
      cast<VMIVRegType>(op.getResult().getType()),
      op.getNumGroupsAttr().getInt(), reason);
}

LogicalResult
VMILayoutSupport::getGroupReduceAddISupport(VMIGroupReduceAddIOp op,
                                            std::string *reason) const {
  return getGroupReduceAddSupportImpl(
      cast<VMIVRegType>(op.getSource().getType()),
      cast<VMIMaskType>(op.getMask().getType()),
      cast<VMIVRegType>(op.getResult().getType()),
      op.getNumGroupsAttr().getInt(), reason);
}

LogicalResult
VMILayoutSupport::getGroupReduceMaxISupport(VMIGroupReduceMaxIOp op,
                                            std::string *reason) const {
  return getGroupReduceAddSupportImpl(
      cast<VMIVRegType>(op.getSource().getType()),
      cast<VMIMaskType>(op.getMask().getType()),
      cast<VMIVRegType>(op.getResult().getType()),
      op.getNumGroupsAttr().getInt(), reason);
}

LogicalResult
VMILayoutSupport::getGroupReduceMinISupport(VMIGroupReduceMinIOp op,
                                            std::string *reason) const {
  return getGroupReduceAddSupportImpl(
      cast<VMIVRegType>(op.getSource().getType()),
      cast<VMIMaskType>(op.getMask().getType()),
      cast<VMIVRegType>(op.getResult().getType()),
      op.getNumGroupsAttr().getInt(), reason);
}

LogicalResult VMILayoutSupport::getGroupBroadcastSupport(
    VMIGroupBroadcastOp op, std::string *reason) const {
  return getGroupBroadcastSupport(cast<VMIVRegType>(op.getSource().getType()),
                                  cast<VMIVRegType>(op.getResult().getType()),
                                  op.getNumGroupsAttr().getInt(), reason);
}

LogicalResult
VMILayoutSupport::getGroupBroadcastLoadSupport(VMIGroupBroadcastLoadOp op,
                                               std::string *reason) const {
  return success(succeeded(getGroupBroadcastLoadLayoutFact(op, reason)));
}

LogicalResult VMILayoutSupport::getGroupBroadcastSupport(
    VMIVRegType sourceType, VMIVRegType resultType, int64_t numGroups,
    std::string *reason) const {
  return success(succeeded(getGroupBroadcastLayoutFactForLayouts(
      sourceType, resultType, numGroups, reason)));
}

// Checker-side acceptance for the two composite 32<->8 spine rows.
// This predicate is deliberately *not* part of the cast candidate path: it is
// consulted only by the "is this assigned pair supportable" validators below
// (getTruncFSupport / getExtFSupport and the matching residual-op guards in
// VMIToVPTO), never by the layout solver or by the propagation transfer, which
// reach the same rows exclusively through the spine-scoped queries.  Accepting
// the pair here is what makes an already-seeded composite pair lowerable; it
// cannot make any chain reach the pair on its own.
static bool isSpineScopedCastLayoutPair(VMIVRegType sourceType,
                                        VMIVRegType resultType,
                                        VMILayoutAttr sourceLayout,
                                        VMILayoutAttr resultLayout) {
  if (!sourceLayout || !resultLayout) {
    return false;
  }
  int64_t sourceBits =
      pto::getPTOStorageElemBitWidth(sourceType.getElementType());
  int64_t resultBits =
      pto::getPTOStorageElemBitWidth(resultType.getElementType());
  MLIRContext *ctx = sourceType.getContext();
  int64_t numGroups =
      sourceLayout.isGroupSlots() ? sourceLayout.getNumGroups() : 0;
  for (const LegalCastLayoutPattern &pattern :
       kSpineScopedLegalCastLayoutPatterns) {
    if (!matchesElementBitsPattern(pattern.sourceBits, sourceBits) ||
        !matchesElementBitsPattern(pattern.resultBits, resultBits)) {
      continue;
    }
    if (matchesLayoutPattern(ctx, pattern.sourceLayout, sourceLayout,
                             numGroups) &&
        matchesLayoutPattern(ctx, pattern.resultLayout, resultLayout,
                             numGroups)) {
      return true;
    }
  }
  return false;
}

inline LogicalResult getNarrowCastSupport(VMIVRegType sourceType,
                                          VMIVRegType resultType,
                                          std::string *reason) {
  if (succeeded(VMILayoutSupport().getCastLayoutFactForLayouts(
          sourceType, resultType, sourceType.getLayoutAttr(),
          resultType.getLayoutAttr(), reason))) {
    return success();
  }
  return success(isSpineScopedCastLayoutPair(
      sourceType, resultType, sourceType.getLayoutAttr(),
      resultType.getLayoutAttr()));
}

LogicalResult VMILayoutSupport::getTruncFSupport(VMITruncFOp op,
                                                 std::string *reason) const {
  return getNarrowCastSupport(cast<VMIVRegType>(op.getSource().getType()),
                              cast<VMIVRegType>(op.getResult().getType()),
                              reason);
}

LogicalResult VMILayoutSupport::getExtFSupport(VMIExtFOp op,
                                               std::string *reason) const {
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  if (succeeded(getCastLayoutFactForLayouts(
          sourceType, resultType, sourceType.getLayoutAttr(),
          resultType.getLayoutAttr(), reason))) {
    return success();
  }
  return success(isSpineScopedCastLayoutPair(
      sourceType, resultType, sourceType.getLayoutAttr(),
      resultType.getLayoutAttr()));
}

template <typename OpT>
inline LogicalResult getExtISupportImpl(OpT op, std::string *reason) {
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  if (succeeded(VMILayoutSupport().getCastLayoutFactForLayouts(
          sourceType, resultType, sourceType.getLayoutAttr(),
          resultType.getLayoutAttr(), reason))) {
    return success();
  }
  return success(isSpineScopedCastLayoutPair(
      sourceType, resultType, sourceType.getLayoutAttr(),
      resultType.getLayoutAttr()));
}

LogicalResult VMILayoutSupport::getExtSISupport(VMIExtSIOp op,
                                                std::string *reason) const {
  return getExtISupportImpl(op, reason);
}

LogicalResult VMILayoutSupport::getExtUISupport(VMIExtUIOp op,
                                                std::string *reason) const {
  return getExtISupportImpl(op, reason);
}

LogicalResult
VMILayoutSupport::getTruncISupport(VMITruncIOp op, std::string *reason) const {
  return getNarrowCastSupport(cast<VMIVRegType>(op.getSource().getType()),
                              cast<VMIVRegType>(op.getResult().getType()),
                              reason);
}

#endif // VMILAYOUTSUPPORTQUERYHELPERS_INC

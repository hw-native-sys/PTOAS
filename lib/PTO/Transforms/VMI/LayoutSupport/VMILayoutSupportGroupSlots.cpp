// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/VMILayoutSupport.h"

#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Support/CodeConstants.h"

namespace mlir {
namespace pto {

//===----------------------------------------------------------------------===//
// Single-carrier group packet predicates
//===----------------------------------------------------------------------===//
// Defined here rather than in VMILayoutSupportQueryHelpers.inc: these have
// external linkage (declared in PTO/Transforms/VMILayoutSupport.h) so their
// definitions must not live in a header-included fragment.

/// Carrier widths the narrowing pack chain can express: the packs are 32 -> 16
/// and 16 -> 8, so only 8/16/32-bit carriers have a form to pack or unpack with.
namespace {
constexpr unsigned kCarrierBits8 = 8;
constexpr unsigned kCarrierBits16 = 16;
constexpr unsigned kCarrierBits32 = 32;

/// Lane strides of a dense lane-strided value: two or four element lanes share
/// the carrier a single unit-stride lane occupies.
constexpr int64_t kLaneStridePair = 2;
constexpr int64_t kLaneStrideQuad = 4;

/// True for a carrier width the narrowing pack chain can express.
bool isPackableCarrierBits(unsigned bits) {
  return bits == kCarrierBits8 || bits == kCarrierBits16 ||
         bits == kCarrierBits32;
}

/// True when a dense lane-strided value of `elementBits`-wide elements has a
/// packable carrier at `laneStride`: both the element carrier and the
/// lane-strided carrier (element bits times the stride) have to be expressible.
bool isPackableLaneStride(unsigned elementBits, int64_t laneStride) {
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

} // namespace pto
} // namespace mlir

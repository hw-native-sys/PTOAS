// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef VMILAYOUTSUPPORTCASTTABLES_INC
#define VMILAYOUTSUPPORTCASTTABLES_INC

struct PreferredCastLayoutPattern {
  ElementBitsPattern sourceBits;
  ElementBitsPattern resultBits;
  int64_t elementCount = 0; // 0 means the default row for this bit-width pair.
  LayoutPattern sourceLayout;
  LayoutPattern resultLayout;
  CastTypeClass typeClass = CastTypeClass::Any;
  bool numGroupsFromElementCount = false;
};

struct HighPriorityCastLayoutPattern {
  ElementBitsPattern sourceBits;
  ElementBitsPattern resultBits;
  PhysicalChunkCountPattern sourceChunks;
  PhysicalChunkCountPattern resultChunks;
  LayoutPattern sourceLayout;
  LayoutPattern resultLayout;
  CastTypeClass typeClass = CastTypeClass::Any;
  ElementCountPattern elementCounts = anyN();
  bool numGroupsFromElementCount = false;
};

struct LegalCastLayoutPattern {
  ElementBitsPattern sourceBits;
  ElementBitsPattern resultBits;
  LayoutPattern sourceLayout;
  LayoutPattern resultLayout;
  CastTypeClass typeClass = CastTypeClass::Any;
  ElementCountPattern elementCounts = anyN();
};

struct LegalMaskGranularityCastLayoutPattern {
  MaskGranularityPattern sourceGranularity;
  MaskGranularityPattern resultGranularity;
  LayoutPattern sourceLayout;
  LayoutPattern resultLayout;
};

struct InterleaveLayoutPattern {
  ElementBitsPattern elementBits;
  PhysicalChunkCountPattern chunks;
  int64_t maxPhysicalParts = 0;
  LayoutPattern lhsLayout;
  LayoutPattern rhsLayout;
  LayoutPattern maskLayout;
  LayoutPattern lowLayout;
  LayoutPattern highLayout;
};

static constexpr PreferredCastLayoutPattern kPreferredCastLayoutPatterns[] = {
    // BF16/F32 use matching layouts for dense vectors and group-slot
    // reduction results. The short-vector rows materialize one group per
    // logical result lane; the bitwidth pair class keeps F16/F32 and integer
    // casts on their existing preferences.
    {bits<16>(), bits<32>(), 2, gs(8), gs(8), CastTypeClass::BF16F32, true},
    {bits<16>(), bits<32>(), 4, gs(8), gs(8), CastTypeClass::BF16F32, true},
    {bits<16>(), bits<32>(), 8, gs(8), gs(8), CastTypeClass::BF16F32, true},
    {bits<32>(), bits<16>(), 2, gs(8), gs(8), CastTypeClass::BF16F32, true},
    {bits<32>(), bits<16>(), 4, gs(8), gs(8), CastTypeClass::BF16F32, true},
    {bits<32>(), bits<16>(), 8, gs(8), gs(8), CastTypeClass::BF16F32, true},
    {bits<16>(), bits<32>(), 64, c(), c(), CastTypeClass::BF16F32},
    {bits<16>(), bits<32>(), 128, c(), c(), CastTypeClass::BF16F32},
    {bits<16>(), bits<32>(), 256, c(), c(), CastTypeClass::BF16F32},
    {bits<32>(), bits<16>(), 64, c(), c(), CastTypeClass::BF16F32},
    {bits<32>(), bits<16>(), 128, c(), c(), CastTypeClass::BF16F32},
    {bits<32>(), bits<16>(), 256, c(), c(), CastTypeClass::BF16F32},
    // Exact rows override the default legal relation for small shapes where the
    // compact lane-stride form is the natural cast layout.
    {bits<16>(), bits<32>(), 64, ls(2), c()},
    // Dense FP8/HiF8 widening prefers the compact UNPK4-style carrier over
    // the default dense {c(), d(4)} factor-4 relation. The latter emits four
    // part-wise conversions and then recombines them; ls(4) -> contiguous
    // keeps the valid byte lanes directly addressable by the one-part vcvt.
    {bits<8>(), bits<32>(), 64, ls(4), c(), CastTypeClass::DenseFP8ToF32},
    {bits<8>(), bits<32>(), 128, ls(4), c(), CastTypeClass::DenseFP8ToF32},
    {bits<8>(), bits<32>(), 256, ls(4), c(), CastTypeClass::DenseFP8ToF32},
    // fp4 pair-cast row: f4E2M1x2 (8-bit storage) -> bf16x2 (32-bit pair) at
    // 128 elements, lowered as UNPK4 + a single P0 vcvt. Dense FP8/HiF8 uses
    // the DenseFP8ToF32 rows above and independently prefers ls(4) -> c.
    {bits<8>(), bits<32>(), 128, ls(4), c(),
     CastTypeClass::F4E2M1x2ToBF16x2},
    {bits<32>(), bits<16>(), 64, c(), ls(2)},
    {bits<32>(), bits<8>(), 64, c(), ls(4)},
    {bits<32>(), bits<8>(), 128, d(2), ls(2)},

    // Default rows for the storage-width cast families.
    {bits<8>(), bits<16>(), 0, c(), d(2)},
    {bits<16>(), bits<32>(), 0, c(), d(2)},
    {bits<8>(), bits<32>(), 0, c(), d(4)},
    {bits<16>(), bits<8>(), 0, d(2), c()},
    {bits<32>(), bits<16>(), 0, d(2), c()},
    {bits<32>(), bits<8>(), 0, d(4), c()},
};

static constexpr PreferredCastLayoutPattern
    kPreferredLaneStrideNarrowCastLayoutPatterns[] = {
        {bits<16>(), bits<8>(), 0, c(), ls(2)},
        {bits<32>(), bits<16>(), 64, c(), c(), CastTypeClass::BF16F32},
        {bits<32>(), bits<16>(), 128, c(), c(), CastTypeClass::BF16F32},
        {bits<32>(), bits<16>(), 256, c(), c(), CastTypeClass::BF16F32},
        {bits<32>(), bits<16>(), 0, c(), ls(2)},
        {bits<32>(), bits<8>(), 0, c(), ls(4)},
};

static constexpr HighPriorityCastLayoutPattern
    kHighPriorityCastLayoutPatterns[] = {
        // BF16/F32 use equal layouts for the supported dense one-chunk case
        // and for short group-slot reduction results.
        {bits<16>(), bits<32>(), chunk<1>(), chunk<1>(), c(), c(),
         CastTypeClass::BF16F32, N<64>()},
        {bits<32>(), bits<16>(), chunk<1>(), chunk<1>(), c(), c(),
         CastTypeClass::BF16F32, N<64>()},
        {bits<16>(), bits<32>(), chunk<1>(), chunk<1>(), gs(8), gs(8),
         CastTypeClass::BF16F32, N<2, 4, 8>(), true},
        {bits<32>(), bits<16>(), chunk<1>(), chunk<1>(), gs(8), gs(8),
         CastTypeClass::BF16F32, N<2, 4, 8>(), true},
        // One-chunk widening relations.
        {bits<8>(), bits<16>(), chunk<1>(), chunk<1>(), ls(2), c()},
        {bits<16>(), bits<32>(), chunk<1>(), chunk<1>(), ls(2), c()},
        {bits<8>(), bits<32>(), chunk<1>(), chunk<1>(), ls(4), c()},

        // One-chunk narrowing relations.
        {bits<16>(), bits<8>(), chunk<1>(), chunk<1>(), c(), ls(2)},
        {bits<32>(), bits<16>(), chunk<1>(), chunk<1>(), c(), ls(2)},
        {bits<32>(), bits<8>(), chunk<1>(), chunk<1>(), c(), ls(4)},
};

static constexpr LegalCastLayoutPattern kLegalCastLayoutPatterns[] = {
    // Same-layout BF16/F32 casts are legal for the fast and strict vcvt
    // lowering paths; group-slot short reductions are limited to 2/4/8 lanes.
    {bits<16>(), bits<32>(), c(), c(), CastTypeClass::BF16F32,
     N<64, 128, 256>()},
    {bits<32>(), bits<16>(), c(), c(), CastTypeClass::BF16F32,
     N<64, 128, 256>()},
    {bits<16>(), bits<32>(), gs(8), gs(8), CastTypeClass::BF16F32,
     N<2, 4, 8>()},
    {bits<32>(), bits<16>(), gs(8), gs(8), CastTypeClass::BF16F32,
     N<2, 4, 8>()},
    // Same-width fp-to-fp (bf16 -> f16): dense contiguous 1:1.
    {bits<16>(), bits<16>(), c(), c(), CastTypeClass::Float},

    // 2x widening.
    {bits<8>(), bits<16>(), c(), d(2)},
    {bits<8>(), bits<16>(), ls(2), c()},
    {bits<8>(), bits<16>(), d(2), d(4)},
    {bits<16>(), bits<32>(), c(), d(2)},
    {bits<16>(), bits<32>(), ls(2), c()},
    {bits<16>(), bits<32>(), d(2), d(4)},

    // 2x narrowing.
    {bits<16>(), bits<8>(), d(2), c()},
    {bits<16>(), bits<8>(), c(), ls(2)},
    {bits<16>(), bits<8>(), d(4), d(2)},
    {bits<32>(), bits<16>(), d(2), c()},
    {bits<32>(), bits<16>(), c(), ls(2)},
    {bits<32>(), bits<16>(), d(4), d(2)},

    // 4x widening/narrowing.
    {bits<8>(), bits<32>(), c(), d(4)},
    {bits<8>(), bits<32>(), ls(2), d(2)},
    {bits<8>(), bits<32>(), ls(4), c()},
    {bits<32>(), bits<8>(), d(4), c()},
    {bits<32>(), bits<8>(), c(), ls(4)},
    {bits<32>(), bits<8>(), d(2), ls(2)},

    // Group-slot casts keep the row-local group layout.  num_groups is
    // inherited from the anchor layout at query time.  Packed narrowing records
    // the selected sub-lane stride on the result; widening is the inverse.
    {bits<8>(), bits<16>(), gs(1), gs(1)},
    {bits<8>(), bits<16>(), gs(8, 2), gs(8)},
    {bits<16>(), bits<32>(), gs(1), gs(1)},
    {bits<16>(), bits<32>(), gs(8, 2), gs(8)},
    {bits<8>(), bits<32>(), gs(1), gs(1)},
    {bits<8>(), bits<32>(), gs(8, 4), gs(8)},
    {bits<8>(), bits<16>(), gs(2), gs(2), CastTypeClass::Integer},
    {bits<8>(), bits<16>(), gs(4), gs(4), CastTypeClass::Integer},
    {bits<8>(), bits<16>(), gs(8), gs(8), CastTypeClass::Integer},
    {bits<16>(), bits<32>(), gs(2), gs(2), CastTypeClass::Integer},
    {bits<16>(), bits<32>(), gs(4), gs(4), CastTypeClass::Integer},
    {bits<16>(), bits<32>(), gs(8), gs(8), CastTypeClass::Integer},
    {bits<8>(), bits<32>(), gs(2), gs(2), CastTypeClass::Integer},
    {bits<8>(), bits<32>(), gs(4), gs(4), CastTypeClass::Integer},
    {bits<8>(), bits<32>(), gs(8), gs(8), CastTypeClass::Integer},
    {bits<16>(), bits<8>(), gs(1), gs(1)},
    {bits<16>(), bits<8>(), gs(8), gs(8, 2)},
    {bits<16>(), bits<8>(), gs(8, 2), gs(8, 4), CastTypeClass::IntegerToUnsigned},
    {bits<32>(), bits<16>(), gs(1), gs(1)},
    {bits<32>(), bits<16>(), gs(8), gs(8, 2)},
    {bits<32>(), bits<8>(), gs(1), gs(1)},
    {bits<32>(), bits<8>(), gs(8), gs(8, 4)},
};

static constexpr LegalMaskGranularityCastLayoutPattern
    kLegalMaskGranularityCastLayoutPatterns[] = {
        // Same-width fp-to-fp (bf16 -> f16): dense contiguous 1:1.
        {mb16(), mb16(), c(), c()},

        // 2x widening.
        {mb8(), mb16(), c(), d(2)},
        {mb8(), mb16(), ls(2), c()},
        {mb8(), mb16(), d(2), d(4)},
        {mb16(), mb32(), c(), d(2)},
        {mb16(), mb32(), ls(2), c()},
        {mb16(), mb32(), d(2), d(4)},

        // 2x narrowing.
        {mb16(), mb8(), d(2), c()},
        {mb16(), mb8(), c(), ls(2)},
        {mb16(), mb8(), d(4), d(2)},
        {mb32(), mb16(), c(), c()},
        {mb32(), mb16(), d(2), c()},
        {mb32(), mb16(), c(), ls(2)},
        {mb32(), mb16(), d(4), d(2)},

        // 4x widening/narrowing.
        {mb8(), mb32(), c(), d(4)},
        {mb8(), mb32(), ls(2), d(2)},
        {mb8(), mb32(), ls(4), c()},
        {mb32(), mb8(), d(4), c()},
        {mb32(), mb8(), c(), ls(4)},
        {mb32(), mb8(), d(2), ls(2)},

        // Group-slot casts keep the row-local group layout.
        {mb8(), mb16(), gs(1), gs(1)},
        {mb8(), mb16(), gs(8, 2), gs(8)},
        {mb16(), mb32(), gs(1), gs(1)},
        {mb16(), mb32(), gs(8, 2), gs(8)},
        {mb8(), mb32(), gs(1), gs(1)},
        {mb8(), mb32(), gs(8, 4), gs(8)},
        {mb16(), mb8(), gs(1), gs(1)},
        {mb16(), mb8(), gs(8), gs(8, 2)},
        {mb16(), mb8(), gs(8, 2), gs(8, 4)},
        {mb32(), mb16(), gs(1), gs(1)},
        {mb32(), mb16(), gs(8), gs(8, 2)},
        {mb32(), mb8(), gs(1), gs(1)},
        {mb32(), mb8(), gs(8), gs(8, 4)},
};

static constexpr InterleaveLayoutPattern kVdintlvLayoutPatterns[] = {
    {bits<8, 16, 32, 64>(), chunk<2, 4>(), 0, d(2), d(2), d(2), c(),
     c()},
    {bits<8, 16, 32, 64>(), chunk<4>(), 0, d(4), d(4), d(4), d(2),
     d(2)},
    {bits<8, 16, 32, 64>(), chunk<1>(), 0, c(), c(), c(), c(), c()},

    // A dense lane-stride layout is a dense vector of wider carrier slots.
    // Preserve the layout by selecting the interleave instruction whose lane
    // width is element_bits * lane_stride.  The direct hardware operation is
    // valid only while the logical vector occupies one carrier register;
    // wider shapes need a cross-register layout relation instead.
    {bits<8, 16>(), chunk<1>(), 1, ls(2), ls(2), ls(2), ls(2), ls(2)},
    {bits<8>(), chunk<1>(), 1, ls(4), ls(4), ls(4), ls(4), ls(4)},
};

static constexpr InterleaveLayoutPattern kVintlvLayoutPatterns[] = {
    {bits<8, 16, 32, 64>(), chunk<2, 4>(), 0, c(), c(), c(), d(2),
     d(2)},
    {bits<8, 16, 32, 64>(), chunk<4>(), 0, d(2), d(2), d(2), d(4),
     d(4)},
    {bits<8, 16, 32, 64>(), chunk<1>(), 0, c(), c(), c(), c(), c()},
    {bits<8, 16>(), chunk<1>(), 1, ls(2), ls(2), ls(2), ls(2), ls(2)},
    {bits<8>(), chunk<1>(), 1, ls(4), ls(4), ls(4), ls(4), ls(4)},
};

#endif // VMILAYOUTSUPPORTCASTTABLES_INC

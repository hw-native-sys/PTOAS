// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.


//===- VMILayoutSupportTables.inc - VMI layout support internals -*- C++ -*-===//
//===----------------------------------------------------------------------===//

// Rule tables
//===----------------------------------------------------------------------===//

struct EnsureLayoutPattern {
  ElementBitsPattern elementBits;
  ElementCountPattern elementCounts;
  LayoutPattern sourceLayout;
  LayoutPattern resultLayout;
};

static constexpr EnsureLayoutPattern kEnsureLayoutPatterns[] = {
    // A one-element dense value and a one-group, one-slot value select the
    // same sole physical carrier lane.
    {bits<8, 16, 32, 64>(), N<1>(), c(), gs(1)},
    {bits<8, 16, 32, 64>(), N<1>(), gs(1), c()},

    {bits<16>(), N<256>(), c(), d(2)},
    {bits<32>(), N<128, 256>(), c(), d(2)},
    {bits<64>(), N<64, 128, 256>(), c(), d(2)},
    {bits<16>(), N<256>(), d(2), c()},
    {bits<32>(), N<128, 256>(), d(2), c()},
    {bits<64>(), N<64, 128, 256>(), d(2), c()},

    {bits<32>(), N<256>(), c(), d(4)},
    {bits<64>(), N<128, 256>(), c(), d(4)},
    {bits<32>(), N<256>(), d(4), c()},
    {bits<64>(), N<128, 256>(), d(4), c()},

    {bits<32>(), N<256, 512>(), d(2), d(4)},
    {bits<64>(), N<128, 256>(), d(2), d(4)},
    {bits<32>(), N<256, 512>(), d(4), d(2)},
    {bits<64>(), N<128, 256>(), d(4), d(2)},

    {bits<8, 16>(), anyN(), c(), ls(2)},
    {bits<8, 16>(), anyN(), ls(2), c()},
    {bits<8>(), anyN(), c(), ls(4)},
    {bits<8>(), anyN(), ls(4), c()},
    {bits<16>(), N<256>(), d(2), ls(2)},
    {bits<16>(), N<256>(), ls(2), d(2)},
    {bits<8>(), anyN(), ls(2), ls(4)},
    {bits<8>(), anyN(), ls(4), ls(2)},

    // Group-slot lane-stride materialization keeps num_groups and slots=8.
    // Each physical part carries at most eight row-local group values, so the
    // conversion is independent of the total logical element count.
    {bits<8, 16>(), anyN(), gs(8), gs(8, 2)},
    {bits<8, 16>(), anyN(), gs(8, 2), gs(8)},
    {bits<8>(), anyN(), gs(8), gs(8, 4)},
    {bits<8>(), anyN(), gs(8, 4), gs(8)},
    {bits<8>(), anyN(), gs(8, 2), gs(8, 4)},
    {bits<8>(), anyN(), gs(8, 4), gs(8, 2)},
};

struct EnsureMaskLayoutPattern {
  MaskGranularityPattern granularity;
  ElementCountPattern elementCounts;
  LayoutPattern sourceLayout;
  LayoutPattern resultLayout;
};

static constexpr EnsureMaskLayoutPattern kEnsureMaskLayoutPatterns[] = {
    {mb16(), N<256>(), c(), d(2)},
    {mb32(), N<128, 256, 512>(), c(), d(2)},
    {mb32(), N<128>(), c(), bd(2)},
    {mb16(), N<256>(), d(2), c()},
    {mb32(), N<128, 256, 512>(), d(2), c()},
    {mb32(), N<128>(), bd(2), c()},

    {mb32(), N<256>(), c(), d(4)},
    {mb32(), N<256>(), c(), bd(4)},
    {mb32(), N<256>(), d(4), c()},
    {mb32(), N<256>(), bd(4), c()},

    {mb8(), N<1, 2, 4, 8, 64, 128>(), c(), ls(2)},
    {mb16(), N<1, 2, 4, 8, 64>(), c(), ls(2)},
    {mb32(), N<1, 2, 4, 8, 64>(), c(), ls(2)},
    {mb8(), N<1, 2, 4, 8, 64, 128>(), ls(2), c()},
    {mb16(), N<1, 2, 4, 8, 64>(), ls(2), c()},
    {mb32(), N<1, 2, 4, 8, 64>(), ls(2), c()},

    {mb8(), N<1, 2, 4, 8, 64>(), c(), ls(4)},
    {mb16(), N<1, 2, 4, 8>(), c(), ls(4)},
    {mb32(), N<1, 2, 4, 8>(), c(), ls(4)},
    {mb8(), N<1, 2, 4, 8, 64>(), ls(4), c()},
    {mb16(), N<1, 2, 4, 8>(), ls(4), c()},
    {mb32(), N<1, 2, 4, 8>(), ls(4), c()},
};

struct GroupBlockClassPattern {
  GroupBlockPattern block;
  VMIGroupBlockClass blockClass;
};

static constexpr GroupBlockClassPattern kGroupBlockClassPatterns[] = {
    {gb(1, 4), VMIGroupBlockClass::QuarterBlock},
    {gb(1, 2), VMIGroupBlockClass::HalfBlock},
    {gb(1), VMIGroupBlockClass::OneBlock},
    {gb(2), VMIGroupBlockClass::TwoBlock},
    {gb(4), VMIGroupBlockClass::FourBlock},
    {gbFull(), VMIGroupBlockClass::FullPartMultiple},
    {gbFull(2), VMIGroupBlockClass::FullPartMultiple},
    {gbFull(4), VMIGroupBlockClass::FullPartMultiple},
};

struct GroupReduceLayoutPattern {
  GroupBlockPattern block;
  LayoutPattern sourceLayout;
  LayoutPattern resultLayout;
};

static constexpr GroupReduceLayoutPattern kGroupReduceLayoutPatterns[] = {
    {gb(1, 4), ls(4), gs(8)},
    {gb(1, 2), ls(2), gs(8)},
    {gb(1), c(), gs(8)},
    {gb(2), d(2), gs(8)},
    {gb(2), bd(2), gs(8)},
    {gb(4), d(4), gs(8)},
    {gb(4), bd(4), gs(8)},
    {gbFull(), c(), gs(1)},
    {gbFull(2), d(2), gs(1)},
    {gbFull(4), d(4), gs(1)},
};

struct PreferredCastLayoutPattern {
  ElementBitsPattern sourceBits;
  ElementBitsPattern resultBits;
  int64_t elementCount = 0; // 0 means the default row for this bit-width pair.
  LayoutPattern sourceLayout;
  LayoutPattern resultLayout;
  CastTypeClass typeClass = CastTypeClass::Any;
};

struct HighPriorityCastLayoutPattern {
  ElementBitsPattern sourceBits;
  ElementBitsPattern resultBits;
  PhysicalChunkCountPattern sourceChunks;
  PhysicalChunkCountPattern resultChunks;
  LayoutPattern sourceLayout;
  LayoutPattern resultLayout;
  CastTypeClass typeClass = CastTypeClass::Any;
};

struct LegalCastLayoutPattern {
  ElementBitsPattern sourceBits;
  ElementBitsPattern resultBits;
  LayoutPattern sourceLayout;
  LayoutPattern resultLayout;
  CastTypeClass typeClass = CastTypeClass::Any;
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
    // Exact rows override the default legal relation for small shapes where the
    // compact lane-stride form is the natural cast layout.
    {bits<16>(), bits<32>(), 64, ls(2), c()},
    {bits<8>(), bits<32>(), 64, ls(4), c()},
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
        {bits<32>(), bits<16>(), 0, c(), ls(2)},
        {bits<32>(), bits<8>(), 0, c(), ls(4)},
};

static constexpr HighPriorityCastLayoutPattern
    kHighPriorityCastLayoutPatterns[] = {
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

struct DenseMemoryLayoutPattern {
  ElementBitsPattern elementBits;
  LayoutPattern layout;
  ElementCountPattern elementCounts = anyN();
  bool preferred = false;
};

static constexpr DenseMemoryLayoutPattern kDenseLoadLayoutPatterns[] = {
    {bits<8, 16, 32>(), c()},  {bits<8, 16, 32>(), ls(2)}, {bits<8>(), ls(4)},
    {bits<8, 16, 32>(), d(2)}, {bits<8, 16, 32>(), d(4)},
};

struct DeinterleaveLoadLayoutPattern {
  ElementBitsPattern elementBits;
  LayoutPattern lowLayout;
  LayoutPattern highLayout;
};

static constexpr DeinterleaveLoadLayoutPattern
    kDeinterleaveLoadLayoutPatterns[] = {
        {bits<8, 16, 32>(), c(), c()},
};

static constexpr DenseMemoryLayoutPattern kDenseStoreLayoutPatterns[] = {
    {bits<8, 16, 32>(), c()},
    {bits<8>(), ls(4), N<64>(), /*preferred=*/true},
    {bits<8>(), ls(2), N<128>(), /*preferred=*/true},
    {bits<16>(), ls(2), N<64>(), /*preferred=*/true},
    {bits<8, 16, 32>(), ls(2)},
    {bits<8>(), ls(4)},
    {bits<8, 16, 32>(), d(2)},
    {bits<8, 16, 32>(), d(4)},
};

struct DenseMaskedStoreLayoutPattern {
  ElementBitsPattern elementBits;
  LayoutPattern valueLayout;
  LayoutPattern maskLayout;
  ElementCountPattern elementCounts = anyN();
  bool preferred = false;
};

struct DenseMaskedLoadLayoutPattern {
  ElementBitsPattern elementBits;
  LayoutPattern resultLayout;
  LayoutPattern maskLayout;
  LayoutPattern passthruLayout;
};

static constexpr DenseMaskedStoreLayoutPattern
    kDenseMaskedStoreLayoutPatterns[] = {
        {bits<8, 16, 32>(), c(), c()},
        {bits<8>(), ls(4), ls(4), N<64>(), /*preferred=*/true},
        {bits<8>(), ls(2), ls(2), N<128>(), /*preferred=*/true},
        {bits<16>(), ls(2), ls(2), N<64>(), /*preferred=*/true},
        {bits<8, 16>(), ls(2), ls(2)},
        {bits<8>(), ls(4), ls(4)},
        {bits<8, 16, 32>(), d(2), d(2)},
        {bits<8, 16, 32>(), d(4), d(4)},
};

static constexpr DenseMaskedLoadLayoutPattern
    kDenseMaskedLoadLayoutPatterns[] = {
        {bits<8, 16, 32>(), c(), c(), c()},
};

struct GroupLoadLayoutPattern {
  ElementBitsPattern elementBits;
  GroupBlockPattern block = gb(2);
  GroupMemoryPattern memory = memAny();
  LayoutPattern resultLayout;
};

static constexpr GroupLoadLayoutPattern kGroupLoadLayoutPatterns[] = {
    {bits<8, 16, 32>(), gb(1, 4), memContiguous(), c()},
    {bits<8, 16, 32>(), gb(1, 2), memContiguous(), c()},
    {bits<8, 16, 32>(), gb(1), memContiguous(), c()},
    {bits<8, 16, 32>(), gb(2), memContiguous(), c()},
    {bits<8, 16, 32>(), gb(4), memContiguous(), c()},
    {bits<8, 16, 32>(), gbFull(), memAny(), c()},
    {bits<32>(), gb(2), memBlockAligned(), bd(2)},
    {bits<32>(), gb(4), memBlockAligned(), bd(4)},
};

struct GroupSlotMemoryLayoutPattern {
  LayoutPattern layout;
};

static constexpr GroupSlotMemoryLayoutPattern kGroupSlotMemoryLayoutPatterns[] =
    {
        {gs(1)},
        {gs(8)},
        {gs(8, 2)},
        {gs(8, 4)},
};

enum class GroupStoreLayoutPriority {
  LegalOnly,
  Fallback,
  High,
};

struct GroupStoreLayoutPattern {
  ElementBitsPattern elementBits;
  GroupBlockPattern block;
  GroupMemoryPattern memory = memAny();
  LayoutPattern valueLayout;
  GroupStoreLayoutPriority priority = GroupStoreLayoutPriority::LegalOnly;
};

static constexpr GroupStoreLayoutPattern kGroupStoreLayoutPatterns[] = {
    // One 32B block per group requires packed contiguous values for the
    // block-strided store path when row_stride preserves 32B alignment.
    {bits<8, 16, 32>(), gb(1), memBlockAligned(), c(),
     GroupStoreLayoutPriority::High},
    // Full physical group chunks use ordinary contiguous stores.
    {bits<8, 16, 32>(), gbFull(), memAny(), c(),
     GroupStoreLayoutPriority::Fallback},
    // Two-part deinterleaved values can be stored directly with vstsx2.
    {bits<8, 16, 32>(), gbFull(2), memAny(), d(2)},
};

struct GroupBroadcastLoadLayoutPattern {
  GroupBlockPattern block;
  ElementBitsPattern elementBits;
  GroupMemoryPattern memory = memContiguous();
  LayoutPattern resultLayout;
  ElementCountPattern numGroups = anyG();
};

static constexpr GroupBroadcastLoadLayoutPattern
    kGroupBroadcastLoadLayoutPatterns[] = {
        {gb(1, 4), bits<8, 16, 32>(), memContiguous(), ls(4)},
        {gb(1, 2), bits<8, 16, 32>(), memContiguous(), ls(2)},
        {gb(1), bits<8, 16, 32>(), memContiguous(), c()},
        {gb(2), bits<8, 16, 32>(), memContiguous(), c()},
        {gb(2), bits<8, 16, 32>(), memContiguous(), d(2)},
        {gb(4), bits<8, 16, 32>(), memContiguous(), c()},
        {gb(4), bits<8, 16, 32>(), memContiguous(), d(4)},
        {gbFull(), bits<8, 16, 32>(), memAny(), c()},
        // A single-group BRC is a scalar broadcast. Its equal-valued result
        // may use a deinterleaved physical layout without data rearrangement.
        {gbFull(), bits<8, 16, 32>(), memAny(), d(2), G<1>()},
        {gbFull(), bits<8, 16, 32>(), memAny(), d(4), G<1>()},
};

struct GroupBroadcastLoadDirectPattern {
  VMIGroupBroadcastLoadDirectKind kind;
  ElementCountPattern numGroups;
  GroupBlockPattern block;
  ElementBitsPattern elementBits;
  GroupMemoryPattern memory = memContiguous();
  LayoutPattern resultLayout;
};

static constexpr GroupBroadcastLoadDirectPattern
    kGroupBroadcastLoadDirectPatterns[] = {
        {VMIGroupBroadcastLoadDirectKind::E2B, G<8>(), gb(1), bits<16, 32>(),
         memContiguous(), c()},
        {VMIGroupBroadcastLoadDirectKind::E2B, G<8>(), gb(2), bits<16, 32>(),
         memContiguous(), d(2)},
        {VMIGroupBroadcastLoadDirectKind::E2B, G<8>(), gb(4), bits<16, 32>(),
         memContiguous(), d(4)},
        {VMIGroupBroadcastLoadDirectKind::BRC, anyG(), gbFull(),
         bits<8, 16, 32>(), memAny(), c()},
        // Scalar BRC can be emitted once per physical result chunk, so its
        // equal-valued result is also directly available as d2/d4.
        {VMIGroupBroadcastLoadDirectKind::BRC, G<1>(), gbFull(),
         bits<8, 16, 32>(), memAny(), d(2)},
        {VMIGroupBroadcastLoadDirectKind::BRC, G<1>(), gbFull(),
         bits<8, 16, 32>(), memAny(), d(4)},
};

struct GroupBroadcastLayoutPattern {
  GroupBlockPattern block;
  LayoutPattern sourceLayout;
  LayoutPattern resultLayout;
};

static constexpr GroupBroadcastLayoutPattern kGroupBroadcastLayoutPatterns[] = {
    {gb(1, 4), gs(8), ls(4)},
    {gb(1, 2), gs(8), ls(2)},
    {gb(1), gs(8), c()},
    {gb(1), gs(8), ls(2)},
    {gb(1), gs(8, 2), c()},
    {gb(1), gs(8, 4), c()},
    {gb(2), gs(8), c()},
    {gb(2), gs(8), d(2)},
    {gb(2), gs(8), bd(2)},
    // bf16 G=4 on L=128: groupSize=32 = two 32B blocks; slots=1 source.
    {gb(2), gs(1), c()},
    {gb(4), gs(8), c()},
    {gb(4), gs(8), d(4)},
    {gb(4), gs(8), bd(4)},
    {gb(4), gs(1), c()},
    {gbFull(), gs(8), c()},
    {gbFull(), gs(1), c()},
    {gbFull(2), gs(1), d(2)},
    {gbFull(4), gs(1), d(4)},
};

struct VselrLayoutPattern {
  ElementBitsPattern elementBits;
  ElementCountPattern sourceElements;
  ElementCountPattern indexElements;
  ElementCountPattern resultElements;
  PhysicalChunkCountPattern sourceChunks;
  PhysicalChunkCountPattern indexChunks;
  PhysicalChunkCountPattern resultChunks;
  LayoutPattern sourceLayout;
  LayoutPattern indexLayout;
  LayoutPattern resultLayout;
};

static constexpr VselrLayoutPattern kVselrLayoutPatterns[] = {
    {bits<8>(), N<64>(), N<64>(), N<64>(), chunk<1>(), chunk<1>(),
     chunk<1>(), c(), c(), c()},
    {bits<8>(), N<128>(), N<128>(), N<128>(), chunk<1>(), chunk<1>(),
     chunk<1>(), c(), c(), c()},
    {bits<8>(), N<256>(), N<256>(), N<256>(), chunk<1>(), chunk<1>(),
     chunk<1>(), c(), c(), c()},
    {bits<16>(), N<64>(), N<64>(), N<64>(), chunk<1>(), chunk<1>(),
     chunk<1>(), c(), c(), c()},
    {bits<16>(), N<128>(), N<128>(), N<128>(), chunk<1>(), chunk<1>(),
     chunk<1>(), c(), c(), c()},
    {bits<32>(), N<64>(), N<64>(), N<64>(), chunk<1>(), chunk<1>(),
     chunk<1>(), c(), c(), c()},
};

struct HistogramLayoutPattern {
  LayoutPattern accLayout;
  LayoutPattern sourceLayout;
  LayoutPattern maskLayout;
  LayoutPattern resultLayout;
};

static constexpr HistogramLayoutPattern kVdhistLayoutPatterns[] = {
    {c(), c(), c(), c()},
};

struct WidthChangingBitcastLayoutPattern {
  LayoutPattern layout;
};

static constexpr WidthChangingBitcastLayoutPattern
    kWidthChangingBitcastLayoutPatterns[] = {
    {c()},
};

//===----------------------------------------------------------------------===//

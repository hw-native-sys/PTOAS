// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef VMILAYOUTSUPPORTTABLES_INC
#define VMILAYOUTSUPPORTTABLES_INC

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
    // A group packet whose groups fit in one physical chunk and a dense short
    // vector select the same lanes of the same sole physical carrier, so the
    // conversion forwards the register unchanged in either direction.  Dense
    // short vectors are how group reduce results and group broadcast sources
    // reach and leave memory, so this bridge is stated as the invariant rather
    // than as the group counts any one producer happens to emit.
    {bits<8, 16, 32, 64>(), anyN(), gsFit(), c()},
    {bits<8, 16, 32, 64>(), anyN(), c(), gsFit()},

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

    // The lane-stride carrier is element bits times the stride and has to be
    // packable (32 -> 16 -> 8 only), so stride 2 stops at 16-bit elements and
    // stride 4 at 8-bit ones.
    {bits<8, 16>(), anyN(), c(), ls(2)},
    {bits<8, 16>(), anyN(), ls(2), c()},
    {bits<8>(), anyN(), c(), ls(4)},
    {bits<8>(), anyN(), ls(4), c()},
    {bits<16>(), N<256>(), d(2), ls(2)},
    {bits<16>(), N<256>(), ls(2), d(2)},
    {bits<8>(), anyN(), ls(2), ls(4)},
    {bits<8>(), anyN(), ls(4), ls(2)},

    // A lane-strided dense value and a unit-stride group packet do not select
    // the same carrier lanes, so they cannot use the carrier-identity bridge
    // above: the dense lane stride is normalized onto the packet's carrier
    // first, which materializeDataLayoutViaContiguous decomposes through the
    // contiguous form.  The rows stay explicit because an ensure_layout row is
    // a whole-layout pair, and they need no element count of their own: a
    // packet whose groups fit in one carrier has num_groups <= slots and the
    // produced forms carry at most eight group slots, so the dense value holds
    // at most eight elements and occupies a single part.  A longer dense value
    // spans several parts, no longer shares a carrier with the packet and has
    // no producer, so both the predicate and the materializer reject it instead
    // of admitting a pair they cannot lower.
    {bits<8, 16>(), anyN(), ls(2), gsFit()},
    {bits<8, 16>(), anyN(), gsFit(), ls(2)},
    {bits<8>(), anyN(), ls(4), gsFit()},
    {bits<8>(), anyN(), gsFit(), ls(4)},

    // The same bridge for a unit-stride dense value against a packet that
    // carries its groups at a lane stride: the short grouped read/write path
    // produces both directions.
    {bits<8, 16>(), anyN(), c(), gsFitStride(2)},
    {bits<8, 16>(), anyN(), gsFitStride(2), c()},
    {bits<8>(), anyN(), c(), gsFitStride(4)},
    {bits<8>(), anyN(), gsFitStride(4), c()},

    // The same carrier at a non-unit lane stride on both sides is the carrier
    // identity itself, so it needs no normalization.  Only the 8-bit stride-4
    // pair is reachable: the packed f8 read/write path is what puts a value in
    // a lane-strided carrier and then hands it to a group packet, while no
    // producer selects a lane-strided 16-bit dense value against a stride-2
    // packet.
    {bits<8>(), anyN(), ls(4), gsFitStride(4)},
    {bits<8>(), anyN(), gsFitStride(4), ls(4)},

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
    {gbCompact(), VMIGroupBlockClass::Compact},
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
    // A one-carrier group packet row used to sit here:
    //   {gbCompact(), c(), gs(8)}
    // It is removed (upstream MR) because the dense masked-row fallback below
    // covers every shape it matched: the reachable rows above already claim the
    // native one-block / ratio shapes, and anything left -- small or eight-bit
    // groups with no native row -- lowers through the slots=1 dense fallback.
    // Do not add a dense one-carrier row back (for example {gb(4), c(), gs(8)}
    // for 128xf32 group=4); the slots=1 form is the contract, pinned by
    // vmi_to_vpto_group_reduce_gb4_contiguous_slots1.pto.
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
    // fp4 pair-cast row: f4E2M1x2 (8-bit storage) -> bf16x2 (32-bit pair) at
    // 128 elements, lowered as UNPK4 + a single P0 vcvt.  Restricted to the
    // exact packed pair so dense 8-bit -> 32-bit casts at 128 elements keep
    // the stock default row {c(), d(4)}: both forms are value-correct for
    // dense data (UNPK4/P0 selects exactly the valid lanes), but letting the
    // dense cast take this row changes solver outcomes and conflict
    // diagnostics (vmi_layout_assignment_multi_return_conflict_invalid.pto).
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
    // A group load whose source row stride differs from its group size is not
    // contiguous: the rows above pair `memContiguous()` (row_stride ==
    // group_size) with the stride-free full-chunk lowering.  A strided source
    // therefore needs its own row before layout assignment can hand the result
    // a layout at all.  The result layout is unchanged (`c()`), and the
    // admissible source shapes are the ones the per-group address plans
    // implement.  Those plans address whole 32-byte blocks, so the table only
    // admits groups of 32, 64, or 128 bytes (`gb(1)`, `gb(2)`, `gb(4)`; larger
    // groups are claimed by the `gbFull()` row above and the full-chunk plan).
    // Sub-32B strided groups (`gb(1, 2)` / `gb(1, 4)`) have no lowering and are
    // rejected with the 32-byte granularity contract by
    // `describeStridedGroupLoadGranularityViolation`, as is a row stride that is
    // not a whole number of 32-byte blocks.  Listing these after the contiguous
    // rows keeps first-match precedence for the packed layout.
    {bits<8, 16, 32>(), gb(1), memAny(), c()},
    {bits<8, 16, 32>(), gb(2), memAny(), c()},
    {bits<8, 16, 32>(), gb(4), memAny(), c()},
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
        // A constant non-unit group stride only relocates each group's base and
        // the per-group BRC plan addresses groups individually, so every dense
        // sub-full-part broadcast admits an arbitrary constant stride.
        {gb(1), bits<8, 16, 32>(), memAny(), c()},
        {gb(2), bits<8, 16, 32>(), memAny(), c()},
        {gb(4), bits<8, 16, 32>(), memAny(), c()},
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
        // A dense sub-full-part group is materialized by one lane-zero BRC
        // load per logical group plus a prefix-mask select tree, so it needs
        // only element alignment and never a 32B-aligned pto.vsldb base.
        {VMIGroupBroadcastLoadDirectKind::BRC, anyG(), gb(1), bits<8, 16, 32>(), memAny(), c()},
        {VMIGroupBroadcastLoadDirectKind::BRC, anyG(), gb(2), bits<8, 16, 32>(), memAny(), c()},
        {VMIGroupBroadcastLoadDirectKind::BRC, anyG(), gb(4), bits<8, 16, 32>(), memAny(), c()},
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
    {gbCompact(), gs(8), c()},
    {gbCompact(), gs(1), c()},
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

//===----------------------------------------------------------------------===//

#endif // VMILAYOUTSUPPORTTABLES_INC

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
    {bits<8, 16, 32, 64>(), N<1>(), c(), gs(1)},
    {bits<8, 16, 32, 64>(), N<1>(), gs(1), c()},

    // A 128-element 16-bit value is a sub-chunk carrier: the tail group
    // reduction reaches its deinterleaved=2 source only through this bridge,
    // so the 16-bit row must admit the shorter element count the 32-bit row
    // already does.
    {bits<16>(), N<128, 256>(), c(), d(2)},
    {bits<32>(), N<128, 256>(), c(), d(2)},
    {bits<64>(), N<64, 128, 256>(), c(), d(2)},
    {bits<16>(), N<128, 256>(), d(2), c()},
    {bits<32>(), N<128, 256>(), d(2), c()},
    {bits<64>(), N<64, 128, 256>(), d(2), c()},

    {bits<16>(), N<256, 768>(), c(), d(4)},
    {bits<32>(), N<256>(), c(), d(4)},
    {bits<32>(), N<512>(), c(), d(4)},
    {bits<32>(), N<128>(), c(), d(4)},
    {bits<64>(), N<128, 256>(), c(), d(4)},
    {bits<16>(), N<256, 768>(), d(4), c()},
    {bits<32>(), N<256>(), d(4), c()},
    {bits<32>(), N<512>(), d(4), c()},
    {bits<32>(), N<128>(), d(4), c()},
    {bits<64>(), N<128, 256>(), d(4), c()},

    {bits<32>(), N<128, 256, 512>(), d(2), d(4)},
    {bits<64>(), N<128, 256>(), d(2), d(4)},
    {bits<32>(), N<128, 256, 512>(), d(4), d(2)},
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

    // Matching non-unit strides describe the same physical lanes. In
    // particular, a native i16 sum packet can feed another half-block reduce
    // through an ls(2) view without packing or reading the high halfwords.
    {bits<8, 16>(), anyN(), ls(2), gsFitStride(2)},
    {bits<8, 16>(), anyN(), gsFitStride(2), ls(2)},
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

// Block-deinterleaved predicates reorder 32B blocks. Their conversion uses
// explicit vector selection and comparison, never identity forwarding.
static constexpr EnsureMaskLayoutPattern kEnsureMaskLayoutPatterns[] = {
    {mb32(), N<128>(), c(), bd(2)},
    {mb32(), N<128>(), bd(2), c()},
    {mb32(), N<256>(), c(), bd(4)},
    {mb32(), N<256>(), bd(4), c()},
    {mb16(), N<128, 256>(), c(), d(2)},
    {mb32(), N<128, 256, 512>(), c(), d(2)},
    {mb16(), N<128, 256>(), d(2), c()},
    {mb32(), N<128, 256, 512>(), d(2), c()},

    {mb32(), N<256>(), c(), d(4)},
    {mb16(), N<256, 768>(), c(), d(4)},
    {mb32(), N<256>(), d(4), c()},
    {mb16(), N<256, 768>(), d(4), c()},

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

// A generated group mask is built by a predicate-generating instruction that
// writes one predicate per physical part in a fixed layout.  When the assigned
// result layout differs from that generation layout, the mask has to be
// rearranged first, so the cost model has to know both layouts.  A row whose
// two layouts are equal is the identity: the mask is generated in place.
enum class GeneratedMaskKind {
    GroupConstant,
    GroupDynamic,
};

struct GeneratedMaskStagingPattern {
    GeneratedMaskKind kind;
    LayoutPattern resultLayout;
    LayoutPattern generationLayout;
};

static constexpr GeneratedMaskStagingPattern kGeneratedMaskStagingPatterns[] = {
    {GeneratedMaskKind::GroupDynamic, d(2), c()},
    {GeneratedMaskKind::GroupDynamic, d(4), c()},
    {GeneratedMaskKind::GroupDynamic, bd(4), bd(4)},
    {GeneratedMaskKind::GroupConstant, bd(4), bd(4)},
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
  LayoutPattern integer16AddResultLayout = resultLayout;
};

// The fourth column is the native layout of 16-bit integer sums. Scalar row
// results keep their default slots=1 form.
static constexpr GroupReduceLayoutPattern kGroupReduceLayoutPatterns[] = {
    {gb(1, 4), ls(4), gs(8), gs(8, 2)},
    {gb(1, 2), ls(2), gs(8), gs(8, 2)},
    {gb(1), c(), gs(8), gs(8, 2)},
    {gb(2), d(2), gs(8), gs(8, 2)},
    {gb(2), bd(2), gs(8), gs(8, 2)},
    {gb(4), d(4), gs(8), gs(8, 2)},
    {gb(4), bd(4), gs(8), gs(8, 2)},
    {gbFull(), c(), gs(1)},
    {gbFull(2), d(2), gs(1)},
    {gbFull(4), d(4), gs(1)},
    // A one-carrier group packet row used to sit here:
    //   {gbCompact(), c(), gs(8)}
    // It is removed (upstream MR) because the dense masked-row fallback below
    // covers every shape it matched: the reachable rows above already claim the
    // native one-block / ratio shapes, and remaining supported groups with no
    // native row lower through the slots=1 dense fallback.
    // Do not add a dense one-carrier row back (for example {gb(4), c(), gs(8)}
    // for 128xf32 group=4); the slots=1 form is the contract, pinned by
    // vmi_to_vpto_group_reduce_gb4_contiguous_slots1.pto.
};

#include "VMILayoutSupportCastTables.inc"

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
  bool requiresF32 = false;
};

static constexpr GroupLoadLayoutPattern kGroupLoadLayoutPatterns[] = {
    {bits<8, 16, 32>(), gb(1, 4), memContiguous(), c()},
    {bits<8, 16, 32>(), gb(1, 2), memContiguous(), c()},
    {bits<8, 16, 32>(), gb(1), memContiguous(), c()},
    {bits<8, 16, 32>(), gb(2), memContiguous(), c()},
    {bits<8, 16, 32>(), gb(4), memContiguous(), c()},
    {bits<8, 16, 32>(), gbFull(), memAny(), c()},
    // Strided sub-part loads use f32 32B block reads. A one-block group
    // fills contiguous carriers; larger groups split their blocks across
    // carriers and therefore produce block-deinterleaved layouts. Declaring
    // the latter contiguous would skip the required block permutation.
    {bits<32>(), gb(1), memBlockAligned(), c(), true},
    {bits<32>(), gb(2), memBlockAligned(), bd(2), true},
    {bits<32>(), gb(4), memBlockAligned(), bd(4), true},
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

static constexpr GroupBroadcastLoadLayoutPattern kGroupBroadcastLoadLayoutPatterns[] = {
    {gb(1, 4), bits<8, 16, 32>(), memContiguous(), ls(4)},
    {gb(1, 2), bits<8, 16, 32>(), memContiguous(), ls(2)},
    {gb(1), bits<8, 16, 32>(), memContiguous(), c()},
    {gb(1), bits<16, 32>(), memContiguous(), d(4)},
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
    // Same single-element-broadcast argument as the gbFull rows below: a
    // contiguous lane_stride = 2 carrier needs no data rearrangement.
    {gb(4), bits<8, 16, 32>(), memContiguous(), ls(2), G<1>()},
    {gbFull(), bits<8, 16, 32>(), memAny(), c()},
    // A single-group BRC is a scalar broadcast. Its equal-valued result
    // may use a deinterleaved physical layout without data rearrangement.
    {gbFull(), bits<8, 16, 32>(), memAny(), d(2), G<1>()},
    {gbFull(), bits<8, 16, 32>(), memAny(), d(4), G<1>()},
    // Same argument for the lane-strided physical carriers: every lane of a
    // single-group broadcast holds the same element, so a contiguous
    // lane_stride = 2 result needs no rearrangement either. This row also
    // supports the scalar broadcast-load form synthesized from a one-lane
    // load followed by an ungrouped broadcast.
    {gbFull(), bits<8, 16, 32>(), memAny(), ls(2), G<1>()},
    // A scalar broadcast fills each carrier even for sub-block results.
    // Its sole group base is independent of the source group stride.
    {gb(1, 2), bits<8, 16, 32>(), memAny(), c(), G<1>()},
    {gb(1, 4), bits<8, 16, 32>(), memAny(), c(), G<1>()},
};

struct GroupBroadcastLoadDirectPattern {
  VMIGroupBroadcastLoadDirectKind kind;
  ElementCountPattern numGroups;
  GroupBlockPattern block;
  ElementBitsPattern elementBits;
  GroupMemoryPattern memory = memContiguous();
  LayoutPattern resultLayout;
};

static constexpr GroupBroadcastLoadDirectPattern kGroupBroadcastLoadDirectPatterns[] = {
    // E2B always reads one full eight-group packet; a packet that covers
    // more groups than the result needs keeps its surplus groups outside
    // the result layout.  The surplus read stays inside the full physical
    // chunk the load safety policy already grants, so a four-group result
    // (for example a 128-column scale tail) may use the same packet.
    {VMIGroupBroadcastLoadDirectKind::E2B, G<8, 4>(), gb(1), bits<16, 32>(), memContiguous(), c()},
    {VMIGroupBroadcastLoadDirectKind::E2B, G<8, 4>(), gb(2), bits<16, 32>(), memContiguous(), d(2)},
    {VMIGroupBroadcastLoadDirectKind::E2B, G<8, 4>(), gb(4), bits<16, 32>(), memContiguous(), d(4)},
    // A dense sub-full-part group is materialized by one lane-zero BRC
    // load per logical group plus a prefix-mask select tree, so it needs
    // only element alignment and never a 32B-aligned pto.vsldb base.
    {VMIGroupBroadcastLoadDirectKind::BRC, anyG(), gb(1), bits<8, 16, 32>(), memAny(), c()},
    {VMIGroupBroadcastLoadDirectKind::BRC, anyG(), gb(2), bits<8, 16, 32>(), memAny(), c()},
    {VMIGroupBroadcastLoadDirectKind::BRC, anyG(), gb(4), bits<8, 16, 32>(), memAny(), c()},
    {VMIGroupBroadcastLoadDirectKind::BRC, anyG(), gbFull(), bits<8, 16, 32>(), memAny(), c()},
    // Scalar BRC can be emitted once per physical result chunk, so its
    // equal-valued result is also directly available as d2/d4.
    {VMIGroupBroadcastLoadDirectKind::BRC, G<1>(), gbFull(), bits<8, 16, 32>(), memAny(), d(2)},
    {VMIGroupBroadcastLoadDirectKind::BRC, G<1>(), gbFull(), bits<8, 16, 32>(), memAny(), d(4)},
    // Match the full-part scalar lane-stride row in the support table.
    // Each physical result carrier receives the same scalar BRC load.
    {VMIGroupBroadcastLoadDirectKind::BRC, G<1>(), gbFull(), bits<8, 16, 32>(), memAny(), ls(2)},
    // A 256-lane part on an 8-bit element type covers 32 VCG block
    // elements, so a 128-lane single-group broadcast is a gb(4) group (half
    // a part), not gbFull.  It is still a scalar broadcast whose
    // equal-valued result may be carried contiguously with lane_stride = 2.
    {VMIGroupBroadcastLoadDirectKind::BRC, G<1>(), gb(4), bits<8, 16, 32>(), memAny(), ls(2)},
    // Short scalar results also use one whole-part BRC per carrier.
    {VMIGroupBroadcastLoadDirectKind::BRC, G<1>(), gb(1, 2), bits<8, 16, 32>(), memAny(), c()},
    {VMIGroupBroadcastLoadDirectKind::BRC, G<1>(), gb(1, 4), bits<8, 16, 32>(), memAny(), c()},
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

// Supported pto.vmi.vexpdif relations.  A row describes the layout of x and
// max (the mask always shares it, because the VPTO lowering requires the mask
// to keep the data's lane order and to keep matching its element width) and
// the layout of the result.  See VMIVexpdifLayoutFact for why the supported
// set is this small.
struct VexpdifLayoutPattern {
    ElementBitsPattern sourceBits;
    ElementBitsPattern resultBits;
    LayoutPattern sourceLayout;
    LayoutPattern resultLayout;
    bool preferred = false;
    ElementCountPattern elementCounts = anyN();
};

static constexpr VexpdifLayoutPattern kVexpdifLayoutPatterns[] = {
    // An f32 source keeps its element width: x/max, the mask, and the result
    // stay in one lane_stride = 1 layout.
    {bits<32>(), bits<32>(), c(), c(), true},
    {bits<32>(), bits<32>(), d(2), d(2)},
    {bits<32>(), bits<32>(), d(4), d(4)},
    {bits<32>(), bits<32>(), bd(2), bd(2)},
    {bits<32>(), bits<32>(), bd(4), bd(4)},
    // An f16 source widens: the two physical results of one source part are
    // exactly the even and the odd lanes of that part.
    {bits<16>(), bits<32>(), c(), d(2), true},
    // Below one full f16 physical part (the public lane counts 1, 2, 4, 8,
    // and 64) the deinterleaved=2 result halves cannot fill whole parts and
    // no ensure_layout bridge back to a contiguous consumer exists, so the
    // VMIToVPTO lowering instead computes the EVEN and ODD halves on the
    // packed source part and merges them with one vintlv: the result stays
    // contiguous with one result part per source part.
    {bits<16>(), bits<32>(), c(), c(), false, N<1, 2, 4, 8, 64>()},
};

// A lane-strided source is not in the table on purpose.  The mask shares the
// source layout, and a mask granularity of b16/b32 with lane_stride = 2/4
// becomes a b32/b64/b128 physical predicate carrier, which the mask carrier
// conversion and the pto.vexpdif mask granularity check both reject.  The
// same lane_stride = 1 rule keeps every result part pair an even/odd lane
// split of its own source part.

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

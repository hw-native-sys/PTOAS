// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMILayoutSupport.cpp - VMI layout support queries --------------===//
//===----------------------------------------------------------------------===//
//
// This file is the central table-driven source for VMI layout support facts.
// Keep file-level responsibilities separated as follows:
//
// 1. Layout pattern DSL:
//    Define compact syntax for expressing layouts and table keys only.
// 2. Query key derivation helpers:
//    Extract op operands and derive normalized keys used to query the tables.
//    Do not add new layout support facts in this section.
// 3. Rule tables:
//    Add legal/preferred layout relations here.  New support facts should be
//    visible as table rows instead of being hidden in query helper branches.
// 4. Table matching and materialization helpers:
//    Convert table rows into facts and compare derived keys with row keys.
//    Do not add new layout support facts in this section.
// 5. Query implementations:
//    Query functions should consume the tables, derive table keys, and check
//    op-level preconditions only.  Shape/type limits that define layout support
//    must be visible as table keys or table rows, not hidden in query branches.
//
// When adding a new family of support rules, first extend the shared pattern
// DSL if needed, then add table rows, then expose them through a small query.
// Avoid local mini-DSLs or ad-hoc support logic that duplicates table facts.

#include "PTO/Transforms/VMILayoutSupport.h"

#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/VMIUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;
using namespace mlir::pto;

namespace {

static llvm::cl::opt<bool> preferLaneStrideNarrowing(
    "vmi-prefer-lane-stride-narrowing",
    llvm::cl::desc(
        "Prefer continuous-to-lane-stride layouts for 2x/4x VMI narrowing"),
    llvm::cl::init(true));

//===----------------------------------------------------------------------===//
// Layout pattern DSL
//===----------------------------------------------------------------------===//

enum class LayoutPatternKind {
  Contiguous,
  LaneStride,
  Deinterleaved,
  BlockDeinterleaved,
  GroupSlots,
};

struct LayoutPattern {
  LayoutPatternKind kind = LayoutPatternKind::Contiguous;
  int64_t value = 1;
  int64_t laneStride = 1;
};

enum class GroupBlockPatternKind {
  Ratio,
  FullPartMultiple,
};

enum class GroupMemoryPatternKind {
  Any,
  Contiguous,
  BlockAligned,
};

struct GroupBlockPattern {
  GroupBlockPatternKind kind = GroupBlockPatternKind::Ratio;
  int64_t numerator = 1;
  int64_t denominator = 1;
};

struct GroupMemoryPattern {
  GroupMemoryPatternKind kind = GroupMemoryPatternKind::Contiguous;
};

struct ElementBitsPattern {
  uint64_t mask = 0;
};

struct ElementCountPattern {
  int64_t values[32] = {};
  int64_t count = 0;
  bool any = false;
};

struct MaskGranularityPattern {
  uint8_t mask = 0;
};

struct PhysicalChunkCountPattern {
  int64_t values[4] = {};
  int64_t count = 0;
};

template <int64_t... Counts>
static constexpr PhysicalChunkCountPattern chunk() {
  static_assert(sizeof...(Counts) <= 4, "too many physical chunk counts");
  static_assert(((Counts == 1 || Counts == 2 || Counts == 4) && ...),
                "unsupported physical chunk count");
  return {{Counts...}, static_cast<int64_t>(sizeof...(Counts))};
}

static constexpr uint64_t elementBitsMask(int64_t bits) {
  return bits == 8    ? 1ull << 0
         : bits == 16 ? 1ull << 1
         : bits == 32 ? 1ull << 2
         : bits == 64 ? 1ull << 3
                      : 0;
}

template <int64_t... Bits> static constexpr ElementBitsPattern bits() {
  return {((uint64_t{0} | elementBitsMask(Bits)) | ...)};
}

template <int64_t... Counts> static constexpr ElementCountPattern N() {
  static_assert(sizeof...(Counts) <= 32, "too many element count patterns");
  return {{Counts...}, static_cast<int64_t>(sizeof...(Counts)), false};
}

template <int64_t... Counts> static constexpr ElementCountPattern G() {
  return N<Counts...>();
}

static constexpr ElementCountPattern anyN() { return {{}, 0, true}; }
static constexpr ElementCountPattern anyG() { return anyN(); }

static constexpr MaskGranularityPattern mb8() { return {1u << 0}; }
static constexpr MaskGranularityPattern mb16() { return {1u << 1}; }
static constexpr MaskGranularityPattern mb32() { return {1u << 2}; }

static bool matchesElementBitsPattern(ElementBitsPattern pattern,
                                      int64_t bits) {
  uint64_t mask = elementBitsMask(bits);
  return mask != 0 && (pattern.mask & mask) != 0;
}

static bool matchesElementBitsPattern(ElementBitsPattern pattern,
                                      Type elementType) {
  unsigned elementBits = pto::getPTOStorageElemBitWidth(elementType);
  return matchesElementBitsPattern(pattern, elementBits);
}

static bool matchesElementCountPattern(ElementCountPattern pattern,
                                       int64_t count) {
  if (pattern.any)
    return true;
  for (int64_t i = 0; i < pattern.count; ++i)
    if (pattern.values[i] == count)
      return true;
  return false;
}

static bool matchesMaskGranularityPattern(MaskGranularityPattern pattern,
                                          StringRef granularity) {
  uint8_t mask = granularity == "b8"    ? 1u << 0
                 : granularity == "b16" ? 1u << 1
                 : granularity == "b32" ? 1u << 2
                                        : 0;
  return mask != 0 && (pattern.mask & mask) != 0;
}

static VMILayoutAttr materializeLayoutPattern(MLIRContext *ctx,
                                              LayoutPattern pattern,
                                              int64_t numGroups = 0) {
  switch (pattern.kind) {
  case LayoutPatternKind::Contiguous:
    return VMILayoutAttr::getContiguous(ctx);
  case LayoutPatternKind::LaneStride:
    return VMILayoutAttr::getContiguous(ctx, pattern.value);
  case LayoutPatternKind::Deinterleaved:
    return VMILayoutAttr::getDeinterleaved(ctx, pattern.value);
  case LayoutPatternKind::BlockDeinterleaved:
    return VMILayoutAttr::getBlockDeinterleaved(ctx, pattern.value);
  case LayoutPatternKind::GroupSlots:
    return numGroups > 0
               ? VMILayoutAttr::getGroupSlots(ctx, numGroups, pattern.value,
                                              pattern.laneStride)
               : VMILayoutAttr();
  }
  llvm_unreachable("unknown layout pattern kind");
}

static bool matchesLayoutPattern(MLIRContext *ctx, LayoutPattern pattern,
                                 VMILayoutAttr layout, int64_t numGroups = 0) {
  if (!layout)
    return false;
  return materializeLayoutPattern(ctx, pattern, numGroups) == layout;
}

static constexpr LayoutPattern c() {
  return {LayoutPatternKind::Contiguous, 1, 1};
}

static constexpr LayoutPattern ls(int64_t laneStride) {
  return {LayoutPatternKind::LaneStride, laneStride, 1};
}

static constexpr LayoutPattern d(int64_t factor) {
  return {LayoutPatternKind::Deinterleaved, factor, 1};
}

static constexpr LayoutPattern bd(int64_t factor) {
  return {LayoutPatternKind::BlockDeinterleaved, factor, 1};
}

static constexpr LayoutPattern gs(int64_t slots, int64_t laneStride = 1) {
  return {LayoutPatternKind::GroupSlots, slots, laneStride};
}

static constexpr GroupMemoryPattern memAny() {
  return {GroupMemoryPatternKind::Any};
}

static constexpr GroupMemoryPattern memContiguous() {
  return {GroupMemoryPatternKind::Contiguous};
}

static constexpr GroupMemoryPattern memBlockAligned() {
  return {GroupMemoryPatternKind::BlockAligned};
}

static constexpr GroupBlockPattern gb(int64_t numerator) {
  return {GroupBlockPatternKind::Ratio, numerator, 1};
}

static constexpr GroupBlockPattern gb(int64_t numerator, int64_t denominator) {
  return {GroupBlockPatternKind::Ratio, numerator, denominator};
}

static constexpr GroupBlockPattern gbFull(int64_t fullPartMultiple = 1) {
  return {GroupBlockPatternKind::FullPartMultiple, fullPartMultiple, 1};
}

//===----------------------------------------------------------------------===//
// Query key derivation helpers.  Keep new layout facts in the rule table
// section below; helpers in this section should only derive normalized keys.
//===----------------------------------------------------------------------===//

static std::optional<int64_t> getConstantIndexValue(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
    return constant.value();
  if (auto constant = value.getDefiningOp<arith::ConstantIntOp>()) {
    if (constant.getType().isIndex())
      return constant.value();
  }
  return std::nullopt;
}

static FailureOr<int64_t> getGroupSizeFromNumGroups(VMIVRegType type,
                                                    int64_t numGroups,
                                                    std::string *reason) {
  auto fail = [&](const Twine &message) -> FailureOr<int64_t> {
    if (reason)
      *reason = message.str();
    return failure();
  };
  if (numGroups <= 0)
    return fail("requires num_groups to be positive");
  if (type.getElementCount() % numGroups != 0)
    return fail("requires num_groups to evenly divide logical lane count");
  return type.getElementCount() / numGroups;
}

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

    {bits<8>(), N<1, 2, 4, 8, 64, 128>(), c(), ls(2)},
    {bits<16>(), N<1, 2, 4, 8, 64>(), c(), ls(2)},
    {bits<8>(), N<1, 2, 4, 8, 64, 128>(), ls(2), c()},
    {bits<16>(), N<1, 2, 4, 8, 64>(), ls(2), c()},

    {bits<8>(), N<1, 2, 4, 8, 64>(), c(), ls(4)},
    {bits<8>(), N<1, 2, 4, 8, 64>(), ls(4), c()},

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
};

struct LegalCastLayoutPattern {
  ElementBitsPattern sourceBits;
  ElementBitsPattern resultBits;
  LayoutPattern sourceLayout;
  LayoutPattern resultLayout;
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

static constexpr LegalCastLayoutPattern kLegalCastLayoutPatterns[] = {
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
    {bits<16>(), bits<32>(), gs(8), gs(8)},
    {bits<8>(), bits<32>(), gs(8), gs(8)},
    {bits<16>(), bits<8>(), gs(1), gs(1)},
    {bits<16>(), bits<8>(), gs(8), gs(8, 2)},
    {bits<32>(), bits<16>(), gs(1), gs(1)},
    {bits<32>(), bits<16>(), gs(8), gs(8, 2)},
    {bits<32>(), bits<8>(), gs(1), gs(1)},
    {bits<32>(), bits<8>(), gs(8), gs(8, 4)},
};

static constexpr LegalMaskGranularityCastLayoutPattern
    kLegalMaskGranularityCastLayoutPatterns[] = {
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
        {mb16(), mb32(), gs(8), gs(8)},
        {mb8(), mb32(), gs(8), gs(8)},
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

struct SupplementalCastLayoutPattern {
  ElementBitsPattern sourceBits;
  ElementBitsPattern resultBits;
  LayoutPattern sourceLayout;
  LayoutPattern resultLayout;
};

static constexpr SupplementalCastLayoutPattern
    kSupplementalIntegerExtLayoutPatterns[] = {
    {bits<8>(), bits<32>(), gs(8), gs(8)},
    {bits<16>(), bits<32>(), gs(8), gs(8)},
};

static constexpr SupplementalCastLayoutPattern
    kSupplementalNarrowCastLayoutPatterns[] = {
    {bits<32>(), bits<8>(), gs(8), gs(8)},
    {bits<32>(), bits<16>(), gs(8), gs(8)},
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

struct GroupBroadcastLoadLayoutPattern {
  GroupBlockPattern block;
  ElementBitsPattern elementBits;
  GroupMemoryPattern memory = memContiguous();
  LayoutPattern resultLayout;
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
    {gb(1), gs(8, 2), c()},
    {gb(1), gs(8, 4), c()},
    {gb(2), gs(8), c()},
    {gb(2), gs(8), d(2)},
    {gb(2), gs(8), bd(2)},
    {gb(4), gs(8), c()},
    {gb(4), gs(8), d(4)},
    {gb(4), gs(8), bd(4)},
    {gbFull(), gs(8), c()},
    {gbFull(), gs(1), c()},
    {gbFull(2), gs(1), d(2)},
    {gbFull(4), gs(1), d(4)},
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
// Table matching and materialization helpers
//===----------------------------------------------------------------------===//

static VMIDeinterleaveLoadLayoutFact materializeDeinterleaveLoadLayoutFact(
    MLIRContext *ctx, const DeinterleaveLoadLayoutPattern &pattern) {
  return VMIDeinterleaveLoadLayoutFact{
      materializeLayoutPattern(ctx, pattern.lowLayout),
      materializeLayoutPattern(ctx, pattern.highLayout)};
}

static bool isSameGroupBlockPattern(GroupBlockPattern lhs,
                                    GroupBlockPattern rhs) {
  return lhs.kind == rhs.kind && lhs.numerator == rhs.numerator &&
         lhs.denominator == rhs.denominator;
}

static VMIGroupBlockClass
getGroupBlockClassFromPattern(GroupBlockPattern pattern) {
  for (const GroupBlockClassPattern &row : kGroupBlockClassPatterns)
    if (isSameGroupBlockPattern(pattern, row.block))
      return row.blockClass;
  llvm_unreachable("unsupported group block pattern");
}

static bool matchesGroupBroadcastLoadMemoryPattern(
    GroupMemoryPattern pattern, std::optional<int64_t> stride,
    int64_t elementBits) {
  switch (pattern.kind) {
  case GroupMemoryPatternKind::Any:
    return true;
  case GroupMemoryPatternKind::Contiguous:
    if (!stride)
      return false;
    return *stride == 1;
  case GroupMemoryPatternKind::BlockAligned: {
    if (!stride)
      return false;
    if (elementBits <= 0 || 256 % elementBits != 0)
      return false;
    int64_t alignedStrideElems = 256 / elementBits;
    return *stride > 0 && *stride % alignedStrideElems == 0;
  }
  }
  llvm_unreachable("unknown group memory pattern kind");
}

static bool matchesGroupLoadMemoryPattern(GroupMemoryPattern pattern,
                                          std::optional<int64_t> rowStride,
                                          int64_t groupSize,
                                          int64_t elementBits) {
  switch (pattern.kind) {
  case GroupMemoryPatternKind::Any:
    return true;
  case GroupMemoryPatternKind::Contiguous:
    return rowStride && *rowStride == groupSize;
  case GroupMemoryPatternKind::BlockAligned: {
    if (!rowStride || elementBits <= 0 || 256 % elementBits != 0)
      return false;
    int64_t alignedStrideElems = 256 / elementBits;
    return *rowStride > 0 && *rowStride % alignedStrideElems == 0;
  }
  }
  llvm_unreachable("unknown group memory pattern kind");
}

static bool isSupportedGroupSlotMemoryLayout(VMILayoutAttr layout,
                                             int64_t numGroups) {
  if (!layout || !layout.isGroupSlots() || layout.getNumGroups() != numGroups ||
      layout.getSlots() <= 0)
    return false;
  for (const GroupSlotMemoryLayoutPattern &pattern :
       kGroupSlotMemoryLayoutPatterns)
    if (matchesLayoutPattern(layout.getContext(), pattern.layout, layout,
                             numGroups))
      return true;
  return false;
}

static FailureOr<VMIGroupBlockClass> getGroupBlockClass(int64_t groupSize,
                                                        int64_t vcgBlockElems) {
  if (vcgBlockElems <= 0)
    return failure();

  for (const GroupBlockClassPattern &row : kGroupBlockClassPatterns) {
    GroupBlockPattern block = row.block;
    if (block.kind == GroupBlockPatternKind::FullPartMultiple) {
      int64_t fullPartElems = 8 * vcgBlockElems;
      if (groupSize >= fullPartElems && groupSize % fullPartElems == 0)
        return row.blockClass;
      continue;
    }

    int64_t numerator = vcgBlockElems * block.numerator;
    if (block.denominator <= 0 || numerator % block.denominator != 0)
      continue;
    if (groupSize == numerator / block.denominator)
      return row.blockClass;
  }
  return failure();
}

struct GroupLayoutKey {
  int64_t groupSize = 0;
  int64_t lanesPerPart = 0;
  int64_t vcgBlockElems = 0;
  VMIGroupBlockClass blockClass = VMIGroupBlockClass::OneBlock;
};

struct InterleaveLayoutKey {
  int64_t elementCount = 0;
  int64_t lanesPerPart = 0;
  int64_t physicalChunkCount = 0;
};

static bool matchesPhysicalChunkCountPattern(
    PhysicalChunkCountPattern pattern, InterleaveLayoutKey key) {
  if (key.physicalChunkCount <= 0)
    return false;
  for (int64_t i = 0; i < pattern.count; ++i)
    if (pattern.values[i] == key.physicalChunkCount)
      return true;
  return false;
}

static bool matchesInterleaveLayoutPattern(
    const InterleaveLayoutPattern &pattern, VMIVRegType valueType,
    InterleaveLayoutKey key) {
  if (!matchesElementBitsPattern(pattern.elementBits,
                                 valueType.getElementType()) ||
      !matchesPhysicalChunkCountPattern(pattern.chunks, key))
    return false;
  if (pattern.maxPhysicalParts <= 0)
    return true;

  VMILayoutAttr lhsLayout =
      materializeLayoutPattern(valueType.getContext(), pattern.lhsLayout);
  auto lhsType = VMIVRegType::get(valueType.getContext(),
                                  valueType.getElementCount(),
                                  valueType.getElementType(), lhsLayout);
  FailureOr<int64_t> physicalParts = getVMIPhysicalArity(lhsType);
  return succeeded(physicalParts) &&
         *physicalParts <= pattern.maxPhysicalParts;
}

static FailureOr<InterleaveLayoutKey>
buildInterleaveLayoutKey(VMIVRegType valueType, std::string *reason) {
  auto fail = [&](const Twine &message) -> FailureOr<InterleaveLayoutKey> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  FailureOr<int64_t> lanesPerPart =
      getDataLanesPerPart(valueType.getElementType());
  if (failed(lanesPerPart))
    return fail("interleave layout requires element type with known physical "
                "lanes per part");
  int64_t elementCount = valueType.getElementCount();
  if (elementCount <= 0)
    return fail("interleave layout requires positive logical lane count");
  int64_t physicalChunkCount =
      elementCount <= *lanesPerPart
          ? 1
          : (elementCount % *lanesPerPart == 0
                 ? elementCount / *lanesPerPart
                 : 0);
  return InterleaveLayoutKey{elementCount, *lanesPerPart, physicalChunkCount};
}

static bool matchesGroupBlockPattern(GroupBlockPattern pattern,
                                     GroupLayoutKey key) {
  if (pattern.kind == GroupBlockPatternKind::FullPartMultiple) {
    if (pattern.numerator <= 0)
      return false;
    int64_t fullPartElems = key.lanesPerPart * pattern.numerator;
    return key.groupSize >= fullPartElems &&
           key.groupSize % fullPartElems == 0;
  }

  int64_t numerator = key.vcgBlockElems * pattern.numerator;
  if (pattern.denominator <= 0 || numerator % pattern.denominator != 0)
    return false;
  return key.groupSize == numerator / pattern.denominator;
}

static FailureOr<GroupLayoutKey>
buildGroupLayoutKey(VMIVRegType type, int64_t numGroups,
                    const Twine &unsupportedGroupSizeReason,
                    std::string *reason) {
  auto fail = [&](const Twine &message) -> FailureOr<GroupLayoutKey> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  FailureOr<int64_t> groupSize =
      getGroupSizeFromNumGroups(type, numGroups, reason);
  if (failed(groupSize))
    return failure();
  FailureOr<int64_t> lanesPerPart = getDataLanesPerPart(type.getElementType());
  if (failed(lanesPerPart) || *lanesPerPart % 8 != 0)
    return fail("requires element type with known 32B VCG block width");

  int64_t vcgBlockElems = *lanesPerPart / 8;
  FailureOr<VMIGroupBlockClass> blockClass =
      getGroupBlockClass(*groupSize, vcgBlockElems);
  if (failed(blockClass))
    return fail(unsupportedGroupSizeReason);

  return GroupLayoutKey{*groupSize, *lanesPerPart, vcgBlockElems, *blockClass};
}

static VMIGroupReduceLayoutFact
materializeGroupReduceLayoutFact(MLIRContext *ctx,
                                 const GroupReduceLayoutPattern &pattern,
                                 int64_t groupSize, int64_t lanesPerPart,
                                 int64_t vcgBlockElems, int64_t numGroups) {
  VMIGroupReduceLayoutFact fact;
  fact.blockClass = getGroupBlockClassFromPattern(pattern.block);
  fact.sourceLayout = materializeLayoutPattern(ctx, pattern.sourceLayout);
  fact.maskLayout = fact.sourceLayout;
  fact.resultLayout =
      materializeLayoutPattern(ctx, pattern.resultLayout, numGroups);
  fact.groupSize = groupSize;
  fact.lanesPerPart = lanesPerPart;
  fact.vcgBlockElems = vcgBlockElems;
  return fact;
}

static VMIGroupBroadcastLayoutFact materializeGroupBroadcastLayoutFact(
    MLIRContext *ctx, const GroupBroadcastLayoutPattern &pattern,
    int64_t groupSize, int64_t lanesPerPart, int64_t vcgBlockElems,
    int64_t numGroups) {
  VMIGroupBroadcastLayoutFact fact;
  fact.blockClass = getGroupBlockClassFromPattern(pattern.block);
  fact.sourceLayout =
      materializeLayoutPattern(ctx, pattern.sourceLayout, numGroups);
  fact.resultLayout =
      materializeLayoutPattern(ctx, pattern.resultLayout, numGroups);
  fact.groupSize = groupSize;
  fact.lanesPerPart = lanesPerPart;
  fact.vcgBlockElems = vcgBlockElems;
  return fact;
}

static VMIInterleaveLayoutFact materializeInterleaveLayoutFact(
    MLIRContext *ctx, const InterleaveLayoutPattern &pattern,
    InterleaveLayoutKey key) {
  VMIInterleaveLayoutFact fact;
  fact.lhsLayout = materializeLayoutPattern(ctx, pattern.lhsLayout);
  fact.rhsLayout = materializeLayoutPattern(ctx, pattern.rhsLayout);
  fact.maskLayout = materializeLayoutPattern(ctx, pattern.maskLayout);
  fact.lowLayout = materializeLayoutPattern(ctx, pattern.lowLayout);
  fact.highLayout = materializeLayoutPattern(ctx, pattern.highLayout);
  fact.elementCount = key.elementCount;
  fact.lanesPerPart = key.lanesPerPart;
  return fact;
}

} // namespace

//===----------------------------------------------------------------------===//
// Query implementations
//===----------------------------------------------------------------------===//

FailureOr<VMIGroupReduceLayoutFact>
VMILayoutSupport::getPreferredGroupReduceLayoutFact(VMIVRegType sourceType,
                                                    int64_t numGroups,
                                                    std::string *reason) const {
  auto fail = [&](const Twine &message) -> FailureOr<VMIGroupReduceLayoutFact> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  FailureOr<GroupLayoutKey> key = buildGroupLayoutKey(
      sourceType, numGroups,
      "group_reduce layout supports group sizes of 1/4, 1/2, 1, 2, or 4 "
      "32B VCG blocks, or full physical chunk multiples",
      reason);
  if (failed(key))
    return failure();

  for (const GroupReduceLayoutPattern &pattern : kGroupReduceLayoutPatterns) {
    if (!matchesGroupBlockPattern(pattern.block, *key))
      continue;
    return materializeGroupReduceLayoutFact(sourceType.getContext(), pattern,
                                            key->groupSize, key->lanesPerPart,
                                            key->vcgBlockElems, numGroups);
  }

  return fail("group_reduce layout supports group sizes of 1/4, 1/2, 1, 2, "
              "or 4 32B VCG blocks, or full physical chunk multiples");
}

FailureOr<VMIGroupReduceLayoutFact>
VMILayoutSupport::getGroupReduceLayoutFactForLayouts(
    VMIVRegType sourceType, VMIMaskType maskType, VMIVRegType resultType,
    int64_t numGroups, std::string *reason) const {
  auto fail = [&](const Twine &message) -> FailureOr<VMIGroupReduceLayoutFact> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr maskLayout = maskType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!sourceLayout || !maskLayout || !resultLayout)
    return fail("requires assigned source, mask, and result layouts");

  FailureOr<GroupLayoutKey> key = buildGroupLayoutKey(
      sourceType, numGroups,
      "group_reduce layout table has no row for this group size", reason);
  if (failed(key))
    return failure();

  for (const GroupReduceLayoutPattern &pattern : kGroupReduceLayoutPatterns) {
    if (!matchesGroupBlockPattern(pattern.block, *key))
      continue;
    VMIGroupReduceLayoutFact candidate = materializeGroupReduceLayoutFact(
        sourceType.getContext(), pattern, key->groupSize, key->lanesPerPart,
        key->vcgBlockElems, numGroups);
    if (candidate.sourceLayout == sourceLayout &&
        candidate.maskLayout == maskLayout &&
        candidate.resultLayout == resultLayout)
      return candidate;
  }

  return fail("group_reduce source/mask/result layouts do not match a legal "
              "layout table row for the group size");
}

FailureOr<SmallVector<VMIGroupReduceLayoutFact, 4>>
VMILayoutSupport::getGroupReduceLayoutFactsForLayout(
    VMIVRegType sourceType, int64_t numGroups, VMIGroupReduceLayoutPort port,
    VMILayoutAttr layout, std::string *reason) const {
  auto fail = [&](const Twine &message)
      -> FailureOr<SmallVector<VMIGroupReduceLayoutFact, 4>> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  if (!layout)
    return fail("requires assigned group_reduce layout query port");

  FailureOr<GroupLayoutKey> key = buildGroupLayoutKey(
      sourceType, numGroups,
      "group_reduce layout table has no row for this group size", reason);
  if (failed(key))
    return failure();

  SmallVector<VMIGroupReduceLayoutFact, 4> facts;
  for (const GroupReduceLayoutPattern &pattern : kGroupReduceLayoutPatterns) {
    if (!matchesGroupBlockPattern(pattern.block, *key))
      continue;
    VMIGroupReduceLayoutFact candidate = materializeGroupReduceLayoutFact(
        sourceType.getContext(), pattern, key->groupSize, key->lanesPerPart,
        key->vcgBlockElems, numGroups);

    VMILayoutAttr candidateLayout;
    switch (port) {
    case VMIGroupReduceLayoutPort::Source:
      candidateLayout = candidate.sourceLayout;
      break;
    case VMIGroupReduceLayoutPort::Mask:
      candidateLayout = candidate.maskLayout;
      break;
    case VMIGroupReduceLayoutPort::Result:
      candidateLayout = candidate.resultLayout;
      break;
    }
    if (candidateLayout == layout)
      facts.push_back(candidate);
  }

  if (facts.empty())
    return fail("group_reduce layout query port does not match a legal layout "
                "table row for the group size");
  return facts;
}

FailureOr<VMIGroupBroadcastLayoutFact>
VMILayoutSupport::getGroupBroadcastLayoutFactForLayouts(
    VMIVRegType sourceType, VMIVRegType resultType, int64_t numGroups,
    std::string *reason) const {
  auto fail =
      [&](const Twine &message) -> FailureOr<VMIGroupBroadcastLayoutFact> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!sourceLayout || !resultLayout)
    return fail("requires assigned source/result layouts");

  FailureOr<GroupLayoutKey> key = buildGroupLayoutKey(
      resultType, numGroups,
      "group_broadcast layout table has no row for this group size", reason);
  if (failed(key))
    return failure();

  for (const GroupBroadcastLayoutPattern &pattern :
       kGroupBroadcastLayoutPatterns) {
    if (!matchesGroupBlockPattern(pattern.block, *key))
      continue;
    VMIGroupBroadcastLayoutFact candidate = materializeGroupBroadcastLayoutFact(
        sourceType.getContext(), pattern, key->groupSize, key->lanesPerPart,
        key->vcgBlockElems, numGroups);
    if (candidate.sourceLayout == sourceLayout &&
        candidate.resultLayout == resultLayout)
      return candidate;
  }

  return fail("source/result layouts do not match a supported group_broadcast "
              "table row");
}

FailureOr<SmallVector<VMIGroupBroadcastLayoutFact, 4>>
VMILayoutSupport::getGroupBroadcastLayoutFactsForLayout(
    VMIVRegType sourceType, VMIVRegType resultType, int64_t numGroups,
    VMIGroupBroadcastLayoutPort port, VMILayoutAttr layout,
    std::string *reason) const {
  auto fail = [&](const Twine &message)
      -> FailureOr<SmallVector<VMIGroupBroadcastLayoutFact, 4>> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  if (!layout)
    return fail("requires assigned group_broadcast layout query port");

  FailureOr<GroupLayoutKey> key = buildGroupLayoutKey(
      resultType, numGroups,
      "group_broadcast layout table has no row for this group size", reason);
  if (failed(key))
    return failure();

  SmallVector<VMIGroupBroadcastLayoutFact, 4> facts;
  for (const GroupBroadcastLayoutPattern &pattern :
       kGroupBroadcastLayoutPatterns) {
    if (!matchesGroupBlockPattern(pattern.block, *key))
      continue;
    VMIGroupBroadcastLayoutFact candidate = materializeGroupBroadcastLayoutFact(
        sourceType.getContext(), pattern, key->groupSize, key->lanesPerPart,
        key->vcgBlockElems, numGroups);

    VMILayoutAttr candidateLayout;
    switch (port) {
    case VMIGroupBroadcastLayoutPort::Source:
      candidateLayout = candidate.sourceLayout;
      break;
    case VMIGroupBroadcastLayoutPort::Result:
      candidateLayout = candidate.resultLayout;
      break;
    }
    if (candidateLayout == layout)
      facts.push_back(candidate);
  }

  if (facts.empty())
    return fail("group_broadcast layout query port does not match a legal "
                "layout table row for the group size");
  return facts;
}

FailureOr<VMIGroupBroadcastLoadLayoutFact>
VMILayoutSupport::getGroupBroadcastLoadLayoutFact(VMIGroupBroadcastLoadOp op,
                                                  std::string *reason) const {
  return getGroupBroadcastLoadLayoutFact(
      cast<VMIVRegType>(op.getResult().getType()), op.getSourceGroupStride(),
      op.getNumGroupsAttr().getInt(), reason);
}

FailureOr<VMIGroupBroadcastLoadLayoutFact>
VMILayoutSupport::getGroupBroadcastLoadLayoutFact(VMIVRegType resultType,
                                                  Value sourceGroupStride,
                                                  int64_t numGroups,
                                                  std::string *reason) const {
  auto fail =
      [&](const Twine &message) -> FailureOr<VMIGroupBroadcastLoadLayoutFact> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!resultLayout)
    return fail("requires assigned result layout");

  unsigned elementBits =
      pto::getPTOStorageElemBitWidth(resultType.getElementType());
  if (elementBits == 0)
    return fail("group_broadcast_load requires known element bit width");
  std::optional<int64_t> stride =
      getConstantIndexValue(sourceGroupStride);

  FailureOr<GroupLayoutKey> key = buildGroupLayoutKey(
      resultType, numGroups,
      "group_broadcast_load layout table has no row for this group size",
      reason);
  if (failed(key))
    return failure();

  for (const GroupBroadcastLoadLayoutPattern &pattern :
       kGroupBroadcastLoadLayoutPatterns) {
    if (!matchesGroupBlockPattern(pattern.block, *key))
      continue;
    if (!matchesElementBitsPattern(pattern.elementBits, elementBits))
      continue;
    if (!matchesGroupBroadcastLoadMemoryPattern(pattern.memory, stride,
                                               elementBits))
      continue;
    if (!matchesLayoutPattern(resultType.getContext(), pattern.resultLayout,
                              resultLayout, numGroups))
      continue;
    return VMIGroupBroadcastLoadLayoutFact{
        getGroupBlockClassFromPattern(pattern.block),
        resultLayout,
        key->groupSize,
        key->lanesPerPart,
        key->vcgBlockElems,
        static_cast<int64_t>(elementBits)};
  }

  int64_t alignedStrideElems = 256 / elementBits;
  return fail(Twine("group_broadcast_load requires a table row for result "
                    "layout, group size, and either constant unit "
                    "source_group_stride or constant positive "
                    "source_group_stride divisible by ") +
              Twine(alignedStrideElems) + " elements");
}

FailureOr<VMIGroupBroadcastLoadDirectFact>
VMILayoutSupport::getGroupBroadcastLoadDirectFact(VMIGroupBroadcastLoadOp op,
                                                  std::string *reason) const {
  return getGroupBroadcastLoadDirectFact(
      cast<VMIVRegType>(op.getResult().getType()), op.getSource().getType(),
      op.getSourceGroupStride(), op.getNumGroupsAttr().getInt(), reason);
}

FailureOr<VMIGroupBroadcastLoadDirectFact>
VMILayoutSupport::getGroupBroadcastLoadDirectFact(
    VMIVRegType resultType, Type sourceType, Value sourceGroupStride,
    int64_t numGroups, std::string *reason) const {
  auto fail =
      [&](const Twine &message) -> FailureOr<VMIGroupBroadcastLoadDirectFact> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  if (!isa<PtrType>(sourceType))
    return fail("group_broadcast_load direct lowering requires !pto.ptr source");

  unsigned elementBits =
      pto::getPTOStorageElemBitWidth(resultType.getElementType());
  if (elementBits == 0)
    return fail("group_broadcast_load requires known element bit width");
  std::optional<int64_t> stride = getConstantIndexValue(sourceGroupStride);

  FailureOr<GroupLayoutKey> key = buildGroupLayoutKey(
      resultType, numGroups,
      "group_broadcast_load preferred layout table has no row for this group "
      "size",
      reason);
  if (failed(key))
    return failure();

  VMILayoutAttr existing = resultType.getLayoutAttr();
  for (const GroupBroadcastLoadDirectPattern &pattern :
       kGroupBroadcastLoadDirectPatterns) {
    if (!matchesElementCountPattern(pattern.numGroups, numGroups))
      continue;
    if (!matchesGroupBlockPattern(pattern.block, *key))
      continue;
    if (!matchesElementBitsPattern(pattern.elementBits, elementBits))
      continue;
    if (!matchesGroupBroadcastLoadMemoryPattern(pattern.memory, stride,
                                               elementBits))
      continue;
    VMILayoutAttr resultLayout = materializeLayoutPattern(
        resultType.getContext(), pattern.resultLayout, numGroups);
    if (existing && existing != resultLayout)
      continue;
    return VMIGroupBroadcastLoadDirectFact{
        pattern.kind,
        VMIGroupBroadcastLoadLayoutFact{
            getGroupBlockClassFromPattern(pattern.block),
            resultLayout,
            key->groupSize,
            key->lanesPerPart,
            key->vcgBlockElems,
            static_cast<int64_t>(elementBits)}};
  }

  return fail("group_broadcast_load has no preferred direct lowering layout "
              "table row");
}

static std::pair<int64_t, int64_t> getCastElementBits(VMIVRegType sourceType,
                                                      VMIVRegType resultType) {
  unsigned sourceBits =
      pto::getPTOStorageElemBitWidth(sourceType.getElementType());
  unsigned resultBits =
      pto::getPTOStorageElemBitWidth(resultType.getElementType());
  return std::pair<int64_t, int64_t>(sourceBits, resultBits);
}

static VMICastLayoutFact makeCastLayoutFact(int64_t sourceBits,
                                            int64_t resultBits,
                                            VMILayoutAttr sourceLayout,
                                            VMILayoutAttr resultLayout) {
  VMICastLayoutFact fact;
  fact.sourceBits = sourceBits;
  fact.resultBits = resultBits;
  fact.sourceLayout = sourceLayout;
  fact.resultLayout = resultLayout;
  return fact;
}

static int64_t getMaskGranularityBits(StringRef granularity) {
  if (granularity == "b8")
    return 8;
  if (granularity == "b16")
    return 16;
  if (granularity == "b32")
    return 32;
  return 0;
}

static VMIMaskGranularityCastLayoutFact
makeMaskGranularityCastLayoutFact(int64_t sourceBits, int64_t resultBits,
                                  VMILayoutAttr sourceLayout,
                                  VMILayoutAttr resultLayout) {
  VMIMaskGranularityCastLayoutFact fact;
  fact.sourceGranularityBits = sourceBits;
  fact.resultGranularityBits = resultBits;
  fact.sourceLayout = sourceLayout;
  fact.resultLayout = resultLayout;
  return fact;
}

static FailureOr<VMICastLayoutFact> getPreferredCastLayoutFactImpl(
    ArrayRef<PreferredCastLayoutPattern> patterns, VMIVRegType sourceType,
    VMIVRegType resultType, StringRef tableName, std::string *reason) {
  auto [sourceBits, resultBits] = getCastElementBits(sourceType, resultType);

  const PreferredCastLayoutPattern *selected = nullptr;
  bool selectedIsExact = false;
  int64_t elementCount = sourceType.getElementCount();
  for (const PreferredCastLayoutPattern &pattern : patterns) {
    if (!matchesElementBitsPattern(pattern.sourceBits, sourceBits) ||
        !matchesElementBitsPattern(pattern.resultBits, resultBits))
      continue;
    bool isExact = pattern.elementCount != 0;
    if (isExact && pattern.elementCount != elementCount)
      continue;
    if (!selected || (isExact && !selectedIsExact)) {
      selected = &pattern;
      selectedIsExact = isExact;
      continue;
    }
    if (isExact == selectedIsExact) {
      if (reason)
        *reason =
            (Twine(tableName) + " has ambiguous matching rows").str();
      return failure();
    }
  }

  if (!selected) {
    if (reason)
      *reason = (Twine("requires a matching ") + tableName + " row").str();
    return failure();
  }

  MLIRContext *ctx = sourceType.getContext();
  return makeCastLayoutFact(sourceBits, resultBits,
                            materializeLayoutPattern(ctx,
                                                     selected->sourceLayout),
                            materializeLayoutPattern(ctx,
                                                     selected->resultLayout));
}

static FailureOr<VMICastLayoutFact>
getPreferredLaneStrideNarrowCastLayoutFactImpl(VMIVRegType sourceType,
                                               VMIVRegType resultType,
                                               std::string *reason) {
  return getPreferredCastLayoutFactImpl(
      kPreferredLaneStrideNarrowCastLayoutPatterns, sourceType, resultType,
      "preferred lane-stride narrow cast layout table", reason);
}

FailureOr<VMICastLayoutFact> VMILayoutSupport::getPreferredCastLayoutFact(
    VMIVRegType sourceType, VMIVRegType resultType, std::string *reason) const {
  if (preferLaneStrideNarrowing) {
    FailureOr<VMICastLayoutFact> laneStrideFact =
        getPreferredLaneStrideNarrowCastLayoutFactImpl(sourceType, resultType,
                                                       reason);
    if (succeeded(laneStrideFact))
      return laneStrideFact;
  }
  return getPreferredCastLayoutFactImpl(
      kPreferredCastLayoutPatterns, sourceType, resultType,
      "preferred cast layout table", reason);
}

FailureOr<SmallVector<VMICastLayoutFact, 4>>
VMILayoutSupport::getCastLayoutFactsForLayout(VMIVRegType sourceType,
                                              VMIVRegType resultType,
                                              VMICastLayoutPort port,
                                              VMILayoutAttr layout,
                                              std::string *reason) const {
  auto fail = [&](const Twine &message)
      -> FailureOr<SmallVector<VMICastLayoutFact, 4>> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto [sourceBits, resultBits] = getCastElementBits(sourceType, resultType);
  MLIRContext *ctx = sourceType.getContext();
  SmallVector<VMICastLayoutFact, 4> facts;

  int64_t numGroups =
      layout && layout.isGroupSlots() ? layout.getNumGroups() : 0;
  for (const LegalCastLayoutPattern &pattern : kLegalCastLayoutPatterns) {
    if (!matchesElementBitsPattern(pattern.sourceBits, sourceBits) ||
        !matchesElementBitsPattern(pattern.resultBits, resultBits))
      continue;

    VMILayoutAttr sourceLayout =
        materializeLayoutPattern(ctx, pattern.sourceLayout, numGroups);
    VMILayoutAttr resultLayout =
        materializeLayoutPattern(ctx, pattern.resultLayout, numGroups);
    if (!sourceLayout || !resultLayout)
      continue;

    if (port == VMICastLayoutPort::Source && sourceLayout != layout)
      continue;
    if (port == VMICastLayoutPort::Result && resultLayout != layout)
      continue;

    facts.push_back(
        makeCastLayoutFact(sourceBits, resultBits, sourceLayout, resultLayout));
  }

  if (facts.empty()) {
    if (port == VMICastLayoutPort::Source)
      return fail("requires a legal cast relation for the source layout");
    return fail("requires a legal cast relation for the result layout");
  }
  return facts;
}

static FailureOr<VMICastLayoutFact>
getUniqueCastLayoutFact(FailureOr<SmallVector<VMICastLayoutFact, 4>> facts,
                        std::string *reason) {
  auto fail = [&](const Twine &message) -> FailureOr<VMICastLayoutFact> {
    if (reason)
      *reason = message.str();
    return failure();
  };
  if (failed(facts))
    return failure();
  if (facts->empty())
    return fail("cast layout query produced no layout facts");
  if (facts->size() != 1)
    return fail("cast layout query produced ambiguous layout facts");
  return facts->front();
}

FailureOr<VMICastLayoutFact> VMILayoutSupport::getCastLayoutFactForSourceLayout(
    VMIVRegType sourceType, VMIVRegType resultType, VMILayoutAttr sourceLayout,
    std::string *reason) const {
  return getUniqueCastLayoutFact(
      getCastLayoutFactsForLayout(sourceType, resultType,
                                  VMICastLayoutPort::Source, sourceLayout,
                                  reason),
      reason);
}

FailureOr<VMICastLayoutFact> VMILayoutSupport::getCastLayoutFactForResultLayout(
    VMIVRegType sourceType, VMIVRegType resultType, VMILayoutAttr resultLayout,
    std::string *reason) const {
  return getUniqueCastLayoutFact(
      getCastLayoutFactsForLayout(sourceType, resultType,
                                  VMICastLayoutPort::Result, resultLayout,
                                  reason),
      reason);
}

FailureOr<VMICastLayoutFact> VMILayoutSupport::getCastLayoutFactForLayouts(
    VMIVRegType sourceType, VMIVRegType resultType, VMILayoutAttr sourceLayout,
    VMILayoutAttr resultLayout, std::string *reason) const {
  auto fail = [&](const Twine &message) -> FailureOr<VMICastLayoutFact> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  FailureOr<SmallVector<VMICastLayoutFact, 4>> facts =
      getCastLayoutFactsForLayout(sourceType, resultType,
                                  VMICastLayoutPort::Source, sourceLayout,
                                  reason);
  if (failed(facts))
    return failure();

  std::optional<VMICastLayoutFact> selected;
  for (const VMICastLayoutFact &fact : *facts) {
    if (fact.resultLayout != resultLayout)
      continue;
    if (selected)
      return fail("cast layout query produced ambiguous layout facts");
    selected = fact;
  }
  if (!selected)
    return fail("source/result layouts do not match a legal cast table row");
  return *selected;
}

FailureOr<SmallVector<VMIMaskGranularityCastLayoutFact, 4>>
VMILayoutSupport::getMaskGranularityCastLayoutFactsForLayout(
    VMIMaskType sourceType, VMIMaskType resultType, VMICastLayoutPort port,
    VMILayoutAttr layout, std::string *reason) const {
  auto fail = [&](const Twine &message)
      -> FailureOr<SmallVector<VMIMaskGranularityCastLayoutFact, 4>> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  if (sourceType.getElementCount() != resultType.getElementCount())
    return fail("requires source and result mask lane counts to match");
  if (!VMIMaskType::isConcreteGranularity(sourceType.getGranularity()) ||
      !VMIMaskType::isConcreteGranularity(resultType.getGranularity()))
    return fail("requires concrete b8/b16/b32 source and result "
                "granularities");

  int64_t sourceBits = getMaskGranularityBits(sourceType.getGranularity());
  int64_t resultBits = getMaskGranularityBits(resultType.getGranularity());
  if (sourceBits == 0 || resultBits == 0)
    return fail("requires supported source/result mask granularities");

  MLIRContext *ctx = sourceType.getContext();
  int64_t numGroups =
      layout && layout.isGroupSlots() ? layout.getNumGroups() : 0;
  SmallVector<VMIMaskGranularityCastLayoutFact, 4> facts;
  for (const LegalMaskGranularityCastLayoutPattern &pattern :
       kLegalMaskGranularityCastLayoutPatterns) {
    if (!matchesMaskGranularityPattern(pattern.sourceGranularity,
                                       sourceType.getGranularity()) ||
        !matchesMaskGranularityPattern(pattern.resultGranularity,
                                       resultType.getGranularity()))
      continue;

    VMILayoutAttr sourceLayout =
        materializeLayoutPattern(ctx, pattern.sourceLayout, numGroups);
    VMILayoutAttr resultLayout =
        materializeLayoutPattern(ctx, pattern.resultLayout, numGroups);
    if (!sourceLayout || !resultLayout)
      continue;

    if (port == VMICastLayoutPort::Source && sourceLayout != layout)
      continue;
    if (port == VMICastLayoutPort::Result && resultLayout != layout)
      continue;

    facts.push_back(makeMaskGranularityCastLayoutFact(
        sourceBits, resultBits, sourceLayout, resultLayout));
  }

  if (facts.empty()) {
    if (port == VMICastLayoutPort::Source)
      return fail("requires a legal mask granularity cast relation for the "
                  "source layout");
    return fail("requires a legal mask granularity cast relation for the "
                "result layout");
  }
  return facts;
}

FailureOr<VMIMaskGranularityCastLayoutFact>
VMILayoutSupport::getMaskGranularityCastLayoutFactForLayouts(
    VMIMaskType sourceType, VMIMaskType resultType, VMILayoutAttr sourceLayout,
    VMILayoutAttr resultLayout, std::string *reason) const {
  auto fail =
      [&](const Twine &message) -> FailureOr<VMIMaskGranularityCastLayoutFact> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  FailureOr<SmallVector<VMIMaskGranularityCastLayoutFact, 4>> facts =
      getMaskGranularityCastLayoutFactsForLayout(
          sourceType, resultType, VMICastLayoutPort::Source, sourceLayout,
          reason);
  if (failed(facts))
    return failure();

  std::optional<VMIMaskGranularityCastLayoutFact> selected;
  for (const VMIMaskGranularityCastLayoutFact &fact : *facts) {
    if (fact.resultLayout != resultLayout)
      continue;
    if (selected)
      return fail("mask granularity cast layout query produced ambiguous "
                  "layout facts");
    selected = fact;
  }
  if (!selected)
    return fail("source/result layouts do not match a legal mask granularity "
                "cast table row");
  return *selected;
}

FailureOr<VMILayoutAttr> VMILayoutSupport::getWidenSourceLayoutForResultLayout(
    VMIVRegType sourceType, VMIVRegType resultType,
    VMILayoutAttr requestedResultLayout, std::string *reason) const {
  FailureOr<VMICastLayoutFact> fact = getCastLayoutFactForResultLayout(
      sourceType, resultType, requestedResultLayout, reason);
  if (failed(fact))
    return failure();
  return fact->sourceLayout;
}

static FailureOr<VMIInterleaveLayoutFact> getPreferredInterleaveLayoutFactImpl(
    ArrayRef<InterleaveLayoutPattern> patterns, VMIVRegType valueType,
    std::string *reason) {
  auto fail = [&](const Twine &message) -> FailureOr<VMIInterleaveLayoutFact> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  FailureOr<InterleaveLayoutKey> key =
      buildInterleaveLayoutKey(valueType, reason);
  if (failed(key))
    return failure();

  for (const InterleaveLayoutPattern &pattern : patterns) {
    if (!matchesInterleaveLayoutPattern(pattern, valueType, *key))
      continue;
    return materializeInterleaveLayoutFact(valueType.getContext(), pattern,
                                           *key);
  }

  return fail("requires a preferred interleave layout table row");
}

static FailureOr<SmallVector<VMIInterleaveLayoutFact, 4>>
getInterleaveLayoutFactsForLayoutImpl(
    ArrayRef<InterleaveLayoutPattern> patterns, VMIVRegType valueType,
    VMIInterleaveLayoutPort port, VMILayoutAttr layout,
    std::string *reason) {
  auto fail = [&](const Twine &message)
      -> FailureOr<SmallVector<VMIInterleaveLayoutFact, 4>> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  if (!layout)
    return fail("requires assigned interleave layout query port");

  FailureOr<InterleaveLayoutKey> key =
      buildInterleaveLayoutKey(valueType, reason);
  if (failed(key))
    return failure();

  SmallVector<VMIInterleaveLayoutFact, 4> facts;
  for (const InterleaveLayoutPattern &pattern : patterns) {
    if (!matchesInterleaveLayoutPattern(pattern, valueType, *key))
      continue;
    VMIInterleaveLayoutFact candidate =
        materializeInterleaveLayoutFact(valueType.getContext(), pattern, *key);

    VMILayoutAttr candidateLayout;
    switch (port) {
    case VMIInterleaveLayoutPort::Lhs:
      candidateLayout = candidate.lhsLayout;
      break;
    case VMIInterleaveLayoutPort::Rhs:
      candidateLayout = candidate.rhsLayout;
      break;
    case VMIInterleaveLayoutPort::Mask:
      candidateLayout = candidate.maskLayout;
      break;
    case VMIInterleaveLayoutPort::Low:
      candidateLayout = candidate.lowLayout;
      break;
    case VMIInterleaveLayoutPort::High:
      candidateLayout = candidate.highLayout;
      break;
    }
    if (candidateLayout == layout)
      facts.push_back(candidate);
  }

  if (facts.empty())
    return fail("interleave layout query port does not match a legal layout "
                "table row for the vector shape");
  return facts;
}

static FailureOr<VMIInterleaveLayoutFact> getInterleaveLayoutFactForLayoutsImpl(
    ArrayRef<InterleaveLayoutPattern> patterns, VMIVRegType lhsType,
    VMIVRegType rhsType, VMIMaskType maskType, VMIVRegType lowType,
    VMIVRegType highType, std::string *reason) {
  auto fail = [&](const Twine &message) -> FailureOr<VMIInterleaveLayoutFact> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  if (lhsType.getElementCount() != rhsType.getElementCount() ||
      lhsType.getElementCount() != lowType.getElementCount() ||
      lhsType.getElementCount() != highType.getElementCount() ||
      lhsType.getElementCount() != maskType.getElementCount())
    return fail("interleave layout requires all ports to share logical lane "
                "count");
  if (lhsType.getElementType() != rhsType.getElementType() ||
      lhsType.getElementType() != lowType.getElementType() ||
      lhsType.getElementType() != highType.getElementType())
    return fail("interleave layout requires all data ports to share element "
                "type");

  VMILayoutAttr lhsLayout = lhsType.getLayoutAttr();
  VMILayoutAttr rhsLayout = rhsType.getLayoutAttr();
  VMILayoutAttr maskLayout = maskType.getLayoutAttr();
  VMILayoutAttr lowLayout = lowType.getLayoutAttr();
  VMILayoutAttr highLayout = highType.getLayoutAttr();
  if (!lhsLayout || !rhsLayout || !maskLayout || !lowLayout || !highLayout)
    return fail("requires assigned lhs/rhs/mask/low/high layouts");

  FailureOr<SmallVector<VMIInterleaveLayoutFact, 4>> facts =
      getInterleaveLayoutFactsForLayoutImpl(
          patterns, lhsType, VMIInterleaveLayoutPort::Lhs, lhsLayout, reason);
  if (failed(facts))
    return failure();

  std::optional<VMIInterleaveLayoutFact> selected;
  for (const VMIInterleaveLayoutFact &fact : *facts) {
    if (fact.rhsLayout != rhsLayout || fact.maskLayout != maskLayout ||
        fact.lowLayout != lowLayout || fact.highLayout != highLayout)
      continue;
    if (selected)
      return fail("interleave layout query produced ambiguous layout facts");
    selected = fact;
  }
  if (!selected)
    return fail("lhs/rhs/mask/low/high layouts do not match a legal "
                "interleave layout table row");
  return *selected;
}

FailureOr<VMIInterleaveLayoutFact>
VMILayoutSupport::getPreferredVintlvLayoutFact(
    VMIVRegType valueType, std::string *reason) const {
  return getPreferredInterleaveLayoutFactImpl(kVintlvLayoutPatterns, valueType,
                                              reason);
}

FailureOr<VMIInterleaveLayoutFact>
VMILayoutSupport::getPreferredVdintlvLayoutFact(
    VMIVRegType valueType, std::string *reason) const {
  return getPreferredInterleaveLayoutFactImpl(kVdintlvLayoutPatterns, valueType,
                                              reason);
}

FailureOr<SmallVector<VMIInterleaveLayoutFact, 4>>
VMILayoutSupport::getVintlvLayoutFactsForLayout(
    VMIVRegType valueType, VMIInterleaveLayoutPort port, VMILayoutAttr layout,
    std::string *reason) const {
  return getInterleaveLayoutFactsForLayoutImpl(
      kVintlvLayoutPatterns, valueType, port, layout, reason);
}

FailureOr<SmallVector<VMIInterleaveLayoutFact, 4>>
VMILayoutSupport::getVdintlvLayoutFactsForLayout(
    VMIVRegType valueType, VMIInterleaveLayoutPort port, VMILayoutAttr layout,
    std::string *reason) const {
  return getInterleaveLayoutFactsForLayoutImpl(
      kVdintlvLayoutPatterns, valueType, port, layout, reason);
}

FailureOr<VMIInterleaveLayoutFact>
VMILayoutSupport::getVintlvLayoutFactForLayouts(
    VMIVRegType lhsType, VMIVRegType rhsType, VMIMaskType maskType,
    VMIVRegType lowType, VMIVRegType highType, std::string *reason) const {
  return getInterleaveLayoutFactForLayoutsImpl(
      kVintlvLayoutPatterns, lhsType, rhsType, maskType, lowType, highType,
      reason);
}

FailureOr<VMIInterleaveLayoutFact>
VMILayoutSupport::getVdintlvLayoutFactForLayouts(
    VMIVRegType lhsType, VMIVRegType rhsType, VMIMaskType maskType,
    VMIVRegType lowType, VMIVRegType highType, std::string *reason) const {
  return getInterleaveLayoutFactForLayoutsImpl(
      kVdintlvLayoutPatterns, lhsType, rhsType, maskType, lowType, highType,
      reason);
}

FailureOr<VMILoadLayoutFact>
VMILayoutSupport::getLoadLayoutFact(VMIVRegType resultType,
                                    std::string *reason) const {
  auto fail = [&](const Twine &message) -> FailureOr<VMILoadLayoutFact> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  VMILayoutAttr layout = resultType.getLayoutAttr();
  if (!layout)
    return fail("requires assigned result layout");
  for (const DenseMemoryLayoutPattern &pattern : kDenseLoadLayoutPatterns) {
    if (!matchesElementBitsPattern(pattern.elementBits,
                                   resultType.getElementType()))
      continue;
    if (!matchesElementCountPattern(pattern.elementCounts,
                                    resultType.getElementCount()))
      continue;
    if (!matchesLayoutPattern(resultType.getContext(), pattern.layout, layout))
      continue;
    return VMILoadLayoutFact{layout};
  }

  return fail("result layout does not match a supported dense load table row");
}

FailureOr<VMIDeinterleaveLoadLayoutFact>
VMILayoutSupport::getPreferredDeinterleaveLoadLayoutFact(
    VMIVRegType valueType, std::string *reason) const {
  auto fail =
      [&](const Twine &message) -> FailureOr<VMIDeinterleaveLoadLayoutFact> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  for (const DeinterleaveLoadLayoutPattern &pattern :
       kDeinterleaveLoadLayoutPatterns) {
    if (!matchesElementBitsPattern(pattern.elementBits,
                                   valueType.getElementType()))
      continue;
    return materializeDeinterleaveLoadLayoutFact(valueType.getContext(),
                                                 pattern);
  }
  return fail("requires a preferred deinterleave_load layout table row");
}

FailureOr<SmallVector<VMIDeinterleaveLoadLayoutFact, 4>>
VMILayoutSupport::getDeinterleaveLoadLayoutFactsForLayout(
    VMIVRegType valueType, VMIDeinterleaveLoadLayoutPort port,
    VMILayoutAttr layout, std::string *reason) const {
  auto fail = [&](const Twine &message)
      -> FailureOr<SmallVector<VMIDeinterleaveLoadLayoutFact, 4>> {
    if (reason)
      *reason = message.str();
    return failure();
  };
  if (!layout)
    return fail("requires assigned deinterleave_load layout query port");

  SmallVector<VMIDeinterleaveLoadLayoutFact, 4> facts;
  for (const DeinterleaveLoadLayoutPattern &pattern :
       kDeinterleaveLoadLayoutPatterns) {
    if (!matchesElementBitsPattern(pattern.elementBits,
                                   valueType.getElementType()))
      continue;
    VMIDeinterleaveLoadLayoutFact candidate =
        materializeDeinterleaveLoadLayoutFact(valueType.getContext(), pattern);
    VMILayoutAttr candidateLayout =
        port == VMIDeinterleaveLoadLayoutPort::Low ? candidate.lowLayout
                                                   : candidate.highLayout;
    if (candidateLayout == layout)
      facts.push_back(candidate);
  }
  if (facts.empty())
    return fail("deinterleave_load layout query port does not match a legal "
                "layout table row");
  return facts;
}

FailureOr<VMIDeinterleaveLoadLayoutFact>
VMILayoutSupport::getDeinterleaveLoadLayoutFactForLayouts(
    VMIVRegType lowType, VMIVRegType highType, std::string *reason) const {
  auto fail =
      [&](const Twine &message) -> FailureOr<VMIDeinterleaveLoadLayoutFact> {
    if (reason)
      *reason = message.str();
    return failure();
  };
  if (lowType.getElementCount() != highType.getElementCount() ||
      lowType.getElementType() != highType.getElementType())
    return fail("deinterleave_load layout requires low/high to share shape "
                "and element type");

  VMILayoutAttr lowLayout = lowType.getLayoutAttr();
  VMILayoutAttr highLayout = highType.getLayoutAttr();
  if (!lowLayout || !highLayout)
    return fail("requires assigned low/high layouts");

  FailureOr<SmallVector<VMIDeinterleaveLoadLayoutFact, 4>> facts =
      getDeinterleaveLoadLayoutFactsForLayout(
          lowType, VMIDeinterleaveLoadLayoutPort::Low, lowLayout, reason);
  if (failed(facts))
    return failure();
  for (const VMIDeinterleaveLoadLayoutFact &fact : *facts)
    if (fact.highLayout == highLayout)
      return fact;
  return fail("low/high layouts do not match a legal deinterleave_load layout "
              "table row");
}

FailureOr<VMIStoreLayoutFact>
VMILayoutSupport::getStoreLayoutFact(VMIVRegType valueType,
                                     std::string *reason) const {
  auto fail = [&](const Twine &message) -> FailureOr<VMIStoreLayoutFact> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  VMILayoutAttr layout = valueType.getLayoutAttr();
  if (!layout)
    return fail("requires assigned value layout");
  for (const DenseMemoryLayoutPattern &pattern : kDenseStoreLayoutPatterns) {
    if (!matchesElementBitsPattern(pattern.elementBits,
                                   valueType.getElementType()))
      continue;
    if (!matchesElementCountPattern(pattern.elementCounts,
                                    valueType.getElementCount()))
      continue;
    if (!matchesLayoutPattern(valueType.getContext(), pattern.layout, layout))
      continue;
    return VMIStoreLayoutFact{layout};
  }

  return fail("value layout does not match a supported dense store table row");
}

FailureOr<VMIStoreLayoutFact>
VMILayoutSupport::getPreferredStoreLayoutFact(VMIVRegType valueType,
                                              std::string *reason) const {
  auto fail = [&](const Twine &message) -> FailureOr<VMIStoreLayoutFact> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  if (valueType.getLayoutAttr())
    return getStoreLayoutFact(valueType, reason);

  for (const DenseMemoryLayoutPattern &pattern : kDenseStoreLayoutPatterns) {
    if (!pattern.preferred)
      continue;
    if (!matchesElementBitsPattern(pattern.elementBits,
                                   valueType.getElementType()))
      continue;
    if (!matchesElementCountPattern(pattern.elementCounts,
                                    valueType.getElementCount()))
      continue;
    VMILayoutAttr layout =
        materializeLayoutPattern(valueType.getContext(), pattern.layout);
    if (!layout)
      continue;
    return VMIStoreLayoutFact{layout};
  }

  return fail("value type does not match a preferred dense store table row");
}

FailureOr<VMIMaskedStoreLayoutFact> VMILayoutSupport::getMaskedStoreLayoutFact(
    VMIVRegType valueType, VMIMaskType maskType, std::string *reason) const {
  auto fail = [&](const Twine &message) -> FailureOr<VMIMaskedStoreLayoutFact> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  VMILayoutAttr valueLayout = valueType.getLayoutAttr();
  VMILayoutAttr maskLayout = maskType.getLayoutAttr();
  if (!valueLayout || !maskLayout)
    return fail("requires assigned value/mask layouts");
  for (const DenseMaskedStoreLayoutPattern &pattern :
       kDenseMaskedStoreLayoutPatterns) {
    if (!matchesElementBitsPattern(pattern.elementBits,
                                   valueType.getElementType()))
      continue;
    if (!matchesElementCountPattern(pattern.elementCounts,
                                    valueType.getElementCount()))
      continue;
    if (!matchesLayoutPattern(valueType.getContext(), pattern.valueLayout,
                              valueLayout))
      continue;
    if (!matchesLayoutPattern(maskType.getContext(), pattern.maskLayout,
                              maskLayout))
      continue;
    return VMIMaskedStoreLayoutFact{valueLayout, maskLayout};
  }

  return fail("value/mask layouts do not match a supported dense masked store "
              "table row");
}

FailureOr<VMIMaskedStoreLayoutFact>
VMILayoutSupport::getPreferredMaskedStoreLayoutFact(
    VMIVRegType valueType, VMIMaskType maskType, std::string *reason) const {
  auto fail =
      [&](const Twine &message) -> FailureOr<VMIMaskedStoreLayoutFact> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  VMILayoutAttr existingValueLayout = valueType.getLayoutAttr();
  VMILayoutAttr existingMaskLayout = maskType.getLayoutAttr();
  if (existingValueLayout && existingMaskLayout)
    return getMaskedStoreLayoutFact(valueType, maskType, reason);

  for (const DenseMaskedStoreLayoutPattern &pattern :
       kDenseMaskedStoreLayoutPatterns) {
    if (!pattern.preferred)
      continue;
    if (!matchesElementBitsPattern(pattern.elementBits,
                                   valueType.getElementType()))
      continue;
    if (!matchesElementCountPattern(pattern.elementCounts,
                                    valueType.getElementCount()))
      continue;

    VMILayoutAttr valueLayout =
        materializeLayoutPattern(valueType.getContext(), pattern.valueLayout);
    VMILayoutAttr maskLayout =
        materializeLayoutPattern(maskType.getContext(), pattern.maskLayout);
    if (!valueLayout || !maskLayout)
      continue;
    if (existingValueLayout && existingValueLayout != valueLayout)
      continue;
    if (existingMaskLayout && existingMaskLayout != maskLayout)
      continue;
    return VMIMaskedStoreLayoutFact{valueLayout, maskLayout};
  }

  return fail("value/mask types do not match a preferred dense masked store "
              "table row");
}

FailureOr<VMIMaskedLoadLayoutFact> VMILayoutSupport::getMaskedLoadLayoutFact(
    VMIVRegType resultType, VMIMaskType maskType, VMIVRegType passthruType,
    std::string *reason) const {
  auto fail = [&](const Twine &message) -> FailureOr<VMIMaskedLoadLayoutFact> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  VMILayoutAttr maskLayout = maskType.getLayoutAttr();
  VMILayoutAttr passthruLayout = passthruType.getLayoutAttr();
  if (!resultLayout || !maskLayout || !passthruLayout)
    return fail("requires assigned result/mask/passthru layouts");
  for (const DenseMaskedLoadLayoutPattern &pattern :
       kDenseMaskedLoadLayoutPatterns) {
    if (!matchesElementBitsPattern(pattern.elementBits,
                                   resultType.getElementType()))
      continue;
    if (!matchesLayoutPattern(resultType.getContext(), pattern.resultLayout,
                              resultLayout))
      continue;
    if (!matchesLayoutPattern(maskType.getContext(), pattern.maskLayout,
                              maskLayout))
      continue;
    if (!matchesLayoutPattern(passthruType.getContext(),
                              pattern.passthruLayout, passthruLayout))
      continue;
    return VMIMaskedLoadLayoutFact{resultLayout, maskLayout, passthruLayout};
  }

  return fail("result/mask/passthru layouts do not match a supported dense "
              "masked_load table row");
}

static LogicalResult matchEnsureLayoutPattern(VMIVRegType sourceType,
                                              VMIVRegType resultType,
                                              VMILayoutAttr sourceLayout,
                                              VMILayoutAttr resultLayout,
                                              std::string *reason) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };
  if (!sourceLayout || !resultLayout)
    return fail("requires assigned source/result layouts");
  if (sourceLayout == resultLayout)
    return success();

  int64_t numGroups =
      sourceLayout.isGroupSlots()
          ? sourceLayout.getNumGroups()
          : (resultLayout.isGroupSlots() ? resultLayout.getNumGroups() : 0);

  for (const EnsureLayoutPattern &pattern : kEnsureLayoutPatterns) {
    if (!matchesElementBitsPattern(pattern.elementBits,
                                   sourceType.getElementType()))
      continue;
    if (!matchesElementCountPattern(pattern.elementCounts,
                                    sourceType.getElementCount()))
      continue;
    if (!matchesLayoutPattern(sourceType.getContext(), pattern.sourceLayout,
                              sourceLayout, numGroups))
      continue;
    if (!matchesLayoutPattern(resultType.getContext(), pattern.resultLayout,
                              resultLayout, numGroups))
      continue;
    return success();
  }

  return fail("source/result layouts do not match a supported ensure_layout "
              "table row");
}

static LogicalResult matchEnsureMaskLayoutPattern(VMIMaskType sourceType,
                                                  VMIMaskType resultType,
                                                  VMILayoutAttr sourceLayout,
                                                  VMILayoutAttr resultLayout,
                                                  std::string *reason) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };
  if (!sourceLayout || !resultLayout)
    return fail("requires assigned source/result layouts");
  if (sourceLayout == resultLayout)
    return success();

  for (const EnsureMaskLayoutPattern &pattern : kEnsureMaskLayoutPatterns) {
    if (!matchesMaskGranularityPattern(pattern.granularity,
                                       sourceType.getGranularity()))
      continue;
    if (!matchesElementCountPattern(pattern.elementCounts,
                                    sourceType.getElementCount()))
      continue;
    if (!matchesLayoutPattern(sourceType.getContext(), pattern.sourceLayout,
                              sourceLayout))
      continue;
    if (!matchesLayoutPattern(resultType.getContext(), pattern.resultLayout,
                              resultLayout))
      continue;
    return success();
  }

  return fail("source/result mask layouts do not match a supported "
              "ensure_mask_layout table row");
}

FailureOr<VMIEnsureLayoutFact> VMILayoutSupport::getEnsureLayoutFact(
    VMIVRegType sourceType, VMIVRegType resultType, std::string *reason) const {
  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (failed(matchEnsureLayoutPattern(sourceType, resultType, sourceLayout,
                                      resultLayout, reason)))
    return failure();
  return VMIEnsureLayoutFact{sourceLayout, resultLayout};
}

FailureOr<VMIEnsureMaskLayoutFact> VMILayoutSupport::getEnsureMaskLayoutFact(
    VMIMaskType sourceType, VMIMaskType resultType, std::string *reason) const {
  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (failed(matchEnsureMaskLayoutPattern(sourceType, resultType, sourceLayout,
                                          resultLayout, reason)))
    return failure();
  return VMIEnsureMaskLayoutFact{sourceLayout, resultLayout};
}

FailureOr<VMIGroupSlotLayoutFact> VMILayoutSupport::getGroupSlotLoadLayoutFact(
    VMIVRegType resultType, int64_t numGroups, std::string *reason) const {
  auto fail = [&](const Twine &message) -> FailureOr<VMIGroupSlotLayoutFact> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  VMILayoutAttr layout = resultType.getLayoutAttr();
  if (!layout)
    return fail("requires assigned result layout");

  if (!isSupportedGroupSlotMemoryLayout(layout, numGroups))
    return fail("result layout does not match a supported group_slot_load "
                "table row");

  return VMIGroupSlotLayoutFact{layout, numGroups, layout.getSlots()};
}

FailureOr<VMIGroupLoadLayoutFact>
VMILayoutSupport::getGroupLoadLayoutFact(VMIGroupLoadOp op,
                                         std::string *reason) const {
  return getGroupLoadLayoutFact(cast<VMIVRegType>(op.getResult().getType()),
                                op.getRowStride(),
                                op.getNumGroupsAttr().getInt(), reason);
}

FailureOr<VMIGroupLoadLayoutFact> VMILayoutSupport::getGroupLoadLayoutFact(
    VMIVRegType resultType, Value rowStride, int64_t numGroups,
    std::string *reason) const {
  auto fail = [&](const Twine &message) -> FailureOr<VMIGroupLoadLayoutFact> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  VMILayoutAttr layout = resultType.getLayoutAttr();
  if (!layout)
    return fail("requires assigned result layout");

  unsigned elementBits =
      pto::getPTOStorageElemBitWidth(resultType.getElementType());
  if (elementBits == 0)
    return fail("group_load requires known element bit width");
  std::optional<int64_t> stride = getConstantIndexValue(rowStride);

  FailureOr<GroupLayoutKey> key = buildGroupLayoutKey(
      resultType, numGroups,
      "group_load layout table has no row for this group size", reason);
  if (failed(key))
    return failure();

  for (const GroupLoadLayoutPattern &pattern : kGroupLoadLayoutPatterns) {
    if (!matchesElementBitsPattern(pattern.elementBits, elementBits))
      continue;
    if (!matchesGroupBlockPattern(pattern.block, *key))
      continue;
    if (!matchesGroupLoadMemoryPattern(pattern.memory, stride, key->groupSize,
                                       elementBits))
      continue;
    if (!matchesLayoutPattern(resultType.getContext(), pattern.resultLayout,
                              layout))
      continue;
    return VMIGroupLoadLayoutFact{
        getGroupBlockClassFromPattern(pattern.block), layout, key->groupSize};
  }

  return fail("result layout, group size, and row_stride do not match a "
              "supported group_load table row");
}

FailureOr<VMIGroupSlotLayoutFact> VMILayoutSupport::getGroupStoreLayoutFact(
    VMIVRegType valueType, int64_t numGroups, std::string *reason) const {
  auto fail = [&](const Twine &message) -> FailureOr<VMIGroupSlotLayoutFact> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  VMILayoutAttr layout = valueType.getLayoutAttr();
  if (!layout)
    return fail("requires assigned value layout");
  if (!isSupportedGroupSlotMemoryLayout(layout, numGroups))
    return fail("value layout does not match a supported group_store table "
                "row");
  return VMIGroupSlotLayoutFact{layout, numGroups, layout.getSlots()};
}

LogicalResult getGroupReduceAddSupportImpl(VMIVRegType sourceType,
                                           VMIMaskType maskType,
                                           VMIVRegType resultType,
                                           int64_t numGroups,
                                           std::string *reason) {
  FailureOr<VMIGroupReduceLayoutFact> fact =
      VMILayoutSupport().getGroupReduceLayoutFactForLayouts(
          sourceType, maskType, resultType, numGroups, reason);
  if (failed(fact))
    return failure();
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

static LogicalResult matchSupplementalCastLayoutPattern(
    VMIVRegType sourceType, VMIVRegType resultType,
    ArrayRef<SupplementalCastLayoutPattern> patterns,
    std::string *reason) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!sourceLayout || !resultLayout)
    return fail("requires assigned source/result layouts");

  auto [sourceBits, resultBits] = getCastElementBits(sourceType, resultType);
  int64_t numGroups =
      sourceLayout.isGroupSlots()
          ? sourceLayout.getNumGroups()
          : (resultLayout.isGroupSlots() ? resultLayout.getNumGroups() : 0);
  MLIRContext *ctx = sourceType.getContext();
  for (const SupplementalCastLayoutPattern &pattern : patterns) {
    if (!matchesElementBitsPattern(pattern.sourceBits, sourceBits) ||
        !matchesElementBitsPattern(pattern.resultBits, resultBits))
      continue;
    if (!matchesLayoutPattern(ctx, pattern.sourceLayout, sourceLayout,
                              numGroups))
      continue;
    if (!matchesLayoutPattern(ctx, pattern.resultLayout, resultLayout,
                              numGroups))
      continue;
    return success();
  }

  return fail("source/result layouts do not match a supplemental cast table row");
}

static LogicalResult getNarrowCastSupport(VMIVRegType sourceType,
                                          VMIVRegType resultType,
                                          std::string *reason) {
  VMILayoutSupport support;
  if (succeeded(support.getCastLayoutFactForLayouts(
          sourceType, resultType, sourceType.getLayoutAttr(),
          resultType.getLayoutAttr(), reason)))
    return success();
  return matchSupplementalCastLayoutPattern(
      sourceType, resultType, kSupplementalNarrowCastLayoutPatterns, reason);
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
  return success(succeeded(getCastLayoutFactForLayouts(
      sourceType, resultType, sourceType.getLayoutAttr(),
      resultType.getLayoutAttr(), reason)));
}

template <typename OpT>
static LogicalResult getExtISupportImpl(OpT op, std::string *reason) {
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());

  FailureOr<VMICastLayoutFact> fact =
      VMILayoutSupport().getCastLayoutFactForLayouts(
          sourceType, resultType, sourceType.getLayoutAttr(),
          resultType.getLayoutAttr(), reason);
  if (succeeded(fact))
    return success();

  return matchSupplementalCastLayoutPattern(
      sourceType, resultType, kSupplementalIntegerExtLayoutPatterns, reason);
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

FailureOr<VMIBitcastLayoutFact>
VMILayoutSupport::getBitcastLayoutFact(VMIBitcastOp op,
                                       std::string *reason) const {
  auto fail = [&](const Twine &message) -> FailureOr<VMIBitcastLayoutFact> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!sourceLayout || !resultLayout)
    return fail("requires assigned source and result layouts");
  if (sourceLayout != resultLayout)
    return fail("requires matching source and result layouts");

  int64_t numGroups =
      sourceLayout.isGroupSlots() ? sourceLayout.getNumGroups() : 0;
  unsigned sourceElementBits =
      pto::getPTOStorageElemBitWidth(sourceType.getElementType());
  unsigned resultElementBits =
      pto::getPTOStorageElemBitWidth(resultType.getElementType());
  if (sourceElementBits == 0 || resultElementBits == 0)
    return fail("requires source and result with known storage element width");
  // Equal-width bitcast is layout-transparent for any identical layout.  Only
  // width-changing bitcast needs a table row because not every layout has a
  // representation-preserving carrier reinterpretation across element widths.
  if (sourceElementBits != resultElementBits) {
    bool matchedLayout = false;
    MLIRContext *ctx = op.getContext();
    for (const WidthChangingBitcastLayoutPattern &pattern :
         kWidthChangingBitcastLayoutPatterns) {
      if (matchesLayoutPattern(ctx, pattern.layout, sourceLayout, numGroups) &&
          matchesLayoutPattern(ctx, pattern.layout, resultLayout, numGroups)) {
        matchedLayout = true;
        break;
      }
    }
    if (!matchedLayout)
      return fail("width-changing bitcast layout does not match a bitcast "
                  "layout table row");
  }

  return VMIBitcastLayoutFact{sourceLayout, resultLayout};
}

LogicalResult VMILayoutSupport::getBitcastSupport(VMIBitcastOp op,
                                                  std::string *reason) const {
  return getBitcastLayoutFact(op, reason);
}

template <typename OpTy>
static FailureOr<VMIHistogramLayoutFact>
getHistogramLayoutFactImpl(OpTy op, ArrayRef<HistogramLayoutPattern> patterns,
                           StringRef opName, std::string *reason) {
  auto fail = [&](const Twine &message) -> FailureOr<VMIHistogramLayoutFact> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  if (patterns.empty())
    return fail(opName + " histogram layout table has no row");

  auto accType = cast<VMIVRegType>(op.getAcc().getType());
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());

  VMILayoutAttr accLayout = accType.getLayoutAttr();
  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr maskLayout = maskType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!accLayout || !sourceLayout || !maskLayout || !resultLayout)
    return fail("requires assigned acc/source/mask/result layouts");

  MLIRContext *ctx = op.getContext();
  for (const HistogramLayoutPattern &pattern : patterns) {
    if (!matchesLayoutPattern(ctx, pattern.accLayout, accLayout) ||
        !matchesLayoutPattern(ctx, pattern.sourceLayout, sourceLayout) ||
        !matchesLayoutPattern(ctx, pattern.maskLayout, maskLayout) ||
        !matchesLayoutPattern(ctx, pattern.resultLayout, resultLayout))
      continue;

    VMIHistogramLayoutFact fact;
    fact.accLayout = accLayout;
    fact.sourceLayout = sourceLayout;
    fact.maskLayout = maskLayout;
    fact.resultLayout = resultLayout;
    return fact;
  }

  return fail(opName + " acc/source/mask/result layouts do not match a "
                       "histogram layout table row");
}

FailureOr<VMIHistogramLayoutFact>
VMILayoutSupport::getVdhistLayoutFact(VMIVdhistOp op,
                                     std::string *reason) const {
  return getHistogramLayoutFactImpl(op, kVdhistLayoutPatterns, "vdhist", reason);
}

FailureOr<VMIHistogramLayoutFact>
VMILayoutSupport::getVchistLayoutFact(VMIVchistOp op,
                                     std::string *reason) const {
  // vchist shares the same layout constraints as vdhist (same base class, same
  // signature).  When kVdhistLayoutPatterns is updated, review whether vchist
  // should inherit the new patterns.
  return getHistogramLayoutFactImpl(op, kVdhistLayoutPatterns, "vchist", reason);
}

LogicalResult
VMILayoutSupport::getVdhistSupport(VMIVdhistOp op, std::string *reason) const {
  return getVdhistLayoutFact(op, reason);
}

LogicalResult
VMILayoutSupport::getVchistSupport(VMIVchistOp op, std::string *reason) const {
  return getVchistLayoutFact(op, reason);
}

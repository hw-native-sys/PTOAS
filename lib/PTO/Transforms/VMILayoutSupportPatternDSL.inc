// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.


//===- VMILayoutSupportPatternDSL.inc - VMI layout support internals -*- C++ -*-===//
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

enum class CastTypeClass {
  // The layout relation is independent of the source/result value semantics.
  Any,
  // Both source and result are floating-point-like VMI element types.
  Float,
  // Both source and result are integer element types.
  Integer,
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
  int64_t values[mlir::pto::kValue4] = {};
  int64_t count = 0;
};

template <int64_t... Counts>
static constexpr PhysicalChunkCountPattern chunk() {
  static_assert(sizeof...(Counts) <= mlir::pto::kValue4,
                "too many physical chunk counts");
  static_assert(((Counts == 1 || Counts == mlir::pto::kValue2 ||
                  Counts == mlir::pto::kValue4) && ...),
                "unsupported physical chunk count");
  return {{Counts...}, static_cast<int64_t>(sizeof...(Counts))};
}

static constexpr uint64_t elementBitsMask(int64_t bits) {
  return bits == mlir::pto::kValue8    ? 1ull << 0
         : bits == mlir::pto::kValue16 ? 1ull << 1
         : bits == mlir::pto::kValue32 ? 1ull << mlir::pto::kValue2
         : bits == mlir::pto::kValue64 ? 1ull << mlir::pto::kValue3
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

static constexpr MaskGranularityPattern mb8() { return {1U << 0}; }
static constexpr MaskGranularityPattern mb16() { return {1U << 1}; }
static constexpr MaskGranularityPattern mb32() { return {1U << 2}; }

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

static bool matchesCastTypeClass(CastTypeClass typeClass, Type sourceType,
                                 Type resultType) {
  switch (typeClass) {
  case CastTypeClass::Any:
    return true;
  case CastTypeClass::Float:
    return (isa<FloatType>(sourceType) ||
            pto::isPTOLowPrecisionType(sourceType)) &&
           (isa<FloatType>(resultType) ||
            pto::isPTOLowPrecisionType(resultType));
  case CastTypeClass::Integer:
    return isa<IntegerType>(sourceType) && isa<IntegerType>(resultType);
  }
  llvm_unreachable("unknown cast type class");
}

static bool matchesElementCountPattern(ElementCountPattern pattern,
                                       int64_t count) {
  if (pattern.any) {
    return true;
  }
  for (int64_t i = 0; i < pattern.count; ++i) {
    if (pattern.values[i] == count) {
      return true;
    }
  }
  return false;
}

static bool matchesPhysicalChunkCountPattern(
    PhysicalChunkCountPattern pattern, int64_t chunkCount) {
  if (chunkCount <= 0) {
    return false;
  }
  for (int64_t i = 0; i < pattern.count; ++i) {
    if (pattern.values[i] == chunkCount) {
      return true;
    }
  }
  return false;
}

static bool matchesMaskGranularityPattern(MaskGranularityPattern pattern,
                                          StringRef granularity) {
  uint8_t mask = granularity == "b8"    ? 1U << 0
                 : granularity == "b16" ? 1U << 1
                 : granularity == "b32" ? 1U << 2
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
  if (!layout) {
    return false;
  }
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
  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>()) {
    return constant.value();
  }
  if (auto constant = value.getDefiningOp<arith::ConstantIntOp>()) {
    if (constant.getType().isIndex()) {
      return constant.value();
    }
  }
  return std::nullopt;
}

static FailureOr<int64_t> getGroupSizeFromNumGroups(VMIVRegType type,
                                                    int64_t numGroups,
                                                    std::string *reason) {
  auto fail = [&](const Twine &message) -> FailureOr<int64_t> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  if (numGroups <= 0) {
    return fail("requires num_groups to be positive");
  }
  bool evenlyDivides = type.getElementCount() % numGroups == 0;
  if (!evenlyDivides) {
    return fail("requires num_groups to evenly divide logical lane count");
  }
  return type.getElementCount() / numGroups;
}

//===----------------------------------------------------------------------===//

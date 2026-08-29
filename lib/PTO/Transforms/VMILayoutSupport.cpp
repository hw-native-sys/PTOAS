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

#include "PTO/Support/CodeConstants.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/VMIUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"

namespace mlir {
namespace pto {

namespace {

constexpr int64_t kLayoutBlockBitWidth = 256;

static llvm::cl::opt<bool> preferLaneStrideNarrowing(
    "vmi-prefer-lane-stride-narrowing",
    llvm::cl::desc(
        "Prefer continuous-to-lane-stride layouts for 2x/4x VMI narrowing"),
    llvm::cl::init(true));

//===----------------------------------------------------------------------===//
#include "VMILayoutSupportPatternDSL.inc"
#include "VMILayoutSupportTables.inc"
#include "VMILayoutSupportMaterialization.inc"
} // namespace

#include "VMILayoutSupportQueryHelpers.inc"

//===----------------------------------------------------------------------===//
// Query implementations
//===----------------------------------------------------------------------===//

FailureOr<VMIVselrLayoutFact>
VMILayoutSupport::getPreferredVselrLayoutFact(
    VMIVselrOp op, std::string *reason) const {
  auto fail = [&](const Twine &message) -> FailureOr<VMIVselrLayoutFact> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  for (const VselrLayoutPattern &pattern : kVselrLayoutPatterns) {
    VMIVselrLayoutFact fact =
        materializeVselrLayoutFact(op.getContext(), pattern);
    if (matchesVselrLayoutPattern(pattern, op, fact)) {
      return fact;
    }
  }
  return fail("vselr requires contiguous layouts and supports N=64, 128, or "
              "256 for 8-bit, N=64 or 128 for 16-bit, and N=64 for 32-bit "
              "elements");
}

FailureOr<VMIVselrLayoutFact>
VMILayoutSupport::getVselrLayoutFact(VMIVselrOp op,
                                     std::string *reason) const {
  auto fail = [&](const Twine &message) -> FailureOr<VMIVselrLayoutFact> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto indexType = cast<VMIVRegType>(op.getIndex().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  VMIVselrLayoutFact assignedFact{sourceType.getLayoutAttr(),
                                  indexType.getLayoutAttr(),
                                  resultType.getLayoutAttr()};
  if (!assignedFact.sourceLayout || !assignedFact.indexLayout ||
      !assignedFact.resultLayout) {
    return fail("vselr requires assigned source/index/result layouts");
  }

  for (const VselrLayoutPattern &pattern : kVselrLayoutPatterns) {
    VMIVselrLayoutFact tableFact =
        materializeVselrLayoutFact(op.getContext(), pattern);
    if (assignedFact.sourceLayout != tableFact.sourceLayout ||
        assignedFact.indexLayout != tableFact.indexLayout ||
        assignedFact.resultLayout != tableFact.resultLayout ||
        !matchesVselrLayoutPattern(pattern, op, assignedFact)) {
      continue;
    }
    return assignedFact;
  }
  return fail("vselr requires contiguous layouts and supports N=64, 128, or "
              "256 for 8-bit, N=64 or 128 for 16-bit, and N=64 for 32-bit "
              "elements");
}

LogicalResult
VMILayoutSupport::getVselrSupport(VMIVselrOp op,
                                  std::string *reason) const {
  return getVselrLayoutFact(op, reason);
}

FailureOr<VMIGroupReduceLayoutFact>
VMILayoutSupport::getPreferredGroupReduceLayoutFact(VMIVRegType sourceType,
                                                    int64_t numGroups,
                                                    std::string *reason) const {
  auto fail = [&](const Twine &message) -> FailureOr<VMIGroupReduceLayoutFact> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  FailureOr<GroupLayoutKey> key = buildGroupLayoutKey(
      sourceType, numGroups,
      "group_reduce layout supports group sizes of 1/4, 1/2, 1, 2, or 4 "
      "32B VCG blocks, or full physical chunk multiples",
      reason);
  if (failed(key)) {
    return failure();
  }

  for (const GroupReduceLayoutPattern &pattern : kGroupReduceLayoutPatterns) {
    if (!matchesGroupBlockPattern(pattern.block, *key)) {
      continue;
    }
    return materializeGroupReduceLayoutFact(sourceType.getContext(), pattern,
                                            *key, numGroups);
  }

  return fail("group_reduce layout supports group sizes of 1/4, 1/2, 1, 2, "
              "or 4 32B VCG blocks, or full physical chunk multiples");
}

FailureOr<VMIGroupReduceLayoutFact>
VMILayoutSupport::getGroupReduceLayoutFactForLayouts(
    VMIVRegType sourceType, VMIMaskType maskType, VMIVRegType resultType,
    int64_t numGroups, std::string *reason) const {
  auto fail = [&](const Twine &message) -> FailureOr<VMIGroupReduceLayoutFact> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr maskLayout = maskType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!sourceLayout || !maskLayout || !resultLayout) {
    return fail("requires assigned source, mask, and result layouts");
  }

  FailureOr<GroupLayoutKey> key = buildGroupLayoutKey(
      sourceType, numGroups,
      "group_reduce layout table has no row for this group size", reason);
  if (failed(key)) {
    return failure();
  }

  for (const GroupReduceLayoutPattern &pattern : kGroupReduceLayoutPatterns) {
    if (!matchesGroupBlockPattern(pattern.block, *key)) {
      continue;
    }
    VMIGroupReduceLayoutFact candidate = materializeGroupReduceLayoutFact(
        sourceType.getContext(), pattern, *key, numGroups);
    if (candidate.sourceLayout == sourceLayout &&
        candidate.maskLayout == maskLayout &&
        candidate.resultLayout == resultLayout) {
      return candidate;
    }
  }

  return fail("group_reduce source/mask/result layouts do not match a legal "
              "layout table row for the group size");
}

FailureOr<SmallVector<VMIGroupReduceLayoutFact, mlir::pto::kValue4>>
VMILayoutSupport::getGroupReduceLayoutFactsForLayout(
    VMIVRegType sourceType, int64_t numGroups, VMIGroupReduceLayoutPort port,
    VMILayoutAttr layout, std::string *reason) const {
  auto fail = [&](const Twine &message)
      -> FailureOr<SmallVector<VMIGroupReduceLayoutFact, 4>> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  if (!layout) {
    return fail("requires assigned group_reduce layout query port");
  }

  FailureOr<GroupLayoutKey> key = buildGroupLayoutKey(
      sourceType, numGroups,
      "group_reduce layout table has no row for this group size", reason);
  if (failed(key)) {
    return failure();
  }

  SmallVector<VMIGroupReduceLayoutFact, mlir::pto::kValue4> facts;
  for (const GroupReduceLayoutPattern &pattern : kGroupReduceLayoutPatterns) {
    if (!matchesGroupBlockPattern(pattern.block, *key)) {
      continue;
    }
    VMIGroupReduceLayoutFact candidate = materializeGroupReduceLayoutFact(
        sourceType.getContext(), pattern, *key, numGroups);

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
    if (candidateLayout == layout) {
      facts.push_back(candidate);
    }
  }

  if (facts.empty()) {
    return fail("group_reduce layout query port does not match a legal layout "
                "table row for the group size");
  }
  return facts;
}

FailureOr<VMIGroupBroadcastLayoutFact>
VMILayoutSupport::getGroupBroadcastLayoutFactForLayouts(
    VMIVRegType sourceType, VMIVRegType resultType, int64_t numGroups,
    std::string *reason) const {
  auto fail =
      [&](const Twine &message) -> FailureOr<VMIGroupBroadcastLayoutFact> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!sourceLayout || !resultLayout) {
    return fail("requires assigned source/result layouts");
  }

  FailureOr<GroupLayoutKey> key = buildGroupLayoutKey(
      resultType, numGroups,
      "group_broadcast layout table has no row for this group size", reason);
  if (failed(key)) {
    return failure();
  }

  for (const GroupBroadcastLayoutPattern &pattern :
       kGroupBroadcastLayoutPatterns) {
    if (!matchesGroupBlockPattern(pattern.block, *key)) {
      continue;
    }
    VMIGroupBroadcastLayoutFact candidate = materializeGroupBroadcastLayoutFact(
        sourceType.getContext(), pattern, *key, numGroups);
    if (candidate.sourceLayout == sourceLayout &&
        candidate.resultLayout == resultLayout) {
      return candidate;
    }
  }

  return fail("source/result layouts do not match a supported group_broadcast "
              "table row");
}

FailureOr<SmallVector<VMIGroupBroadcastLayoutFact, mlir::pto::kValue4>>
VMILayoutSupport::getGroupBroadcastLayoutFactsForLayout(
    VMIVRegType sourceType, VMIVRegType resultType, int64_t numGroups,
    VMIGroupBroadcastLayoutPort port, VMILayoutAttr layout,
    std::string *reason) const {
  auto fail = [&](const Twine &message)
      -> FailureOr<SmallVector<VMIGroupBroadcastLayoutFact, 4>> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  if (!layout) {
    return fail("requires assigned group_broadcast layout query port");
  }

  FailureOr<GroupLayoutKey> key = buildGroupLayoutKey(
      resultType, numGroups,
      "group_broadcast layout table has no row for this group size", reason);
  if (failed(key)) {
    return failure();
  }

  SmallVector<VMIGroupBroadcastLayoutFact, mlir::pto::kValue4> facts;
  for (const GroupBroadcastLayoutPattern &pattern :
       kGroupBroadcastLayoutPatterns) {
    if (!matchesGroupBlockPattern(pattern.block, *key)) {
      continue;
    }
    VMIGroupBroadcastLayoutFact candidate = materializeGroupBroadcastLayoutFact(
        sourceType.getContext(), pattern, *key, numGroups);

    VMILayoutAttr candidateLayout;
    switch (port) {
    case VMIGroupBroadcastLayoutPort::Source:
      candidateLayout = candidate.sourceLayout;
      break;
    case VMIGroupBroadcastLayoutPort::Result:
      candidateLayout = candidate.resultLayout;
      break;
    }
    if (candidateLayout == layout) {
      facts.push_back(candidate);
    }
  }

  if (facts.empty()) {
    return fail("group_broadcast layout query port does not match a legal "
                "layout table row for the group size");
  }
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
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!resultLayout) {
    return fail("requires assigned result layout");
  }

  unsigned elementBits =
      pto::getPTOStorageElemBitWidth(resultType.getElementType());
  if (elementBits == 0) {
    return fail("group_broadcast_load requires known element bit width");
  }
  std::optional<int64_t> stride =
      getConstantIndexValue(sourceGroupStride);

  FailureOr<GroupLayoutKey> key = buildGroupLayoutKey(
      resultType, numGroups,
      "group_broadcast_load layout table has no row for this group size",
      reason);
  if (failed(key)) {
    return failure();
  }

  GroupBroadcastLoadQuery query{resultType, resultLayout, stride, *key,
                                numGroups, elementBits};
  for (const GroupBroadcastLoadLayoutPattern &pattern :
       kGroupBroadcastLoadLayoutPatterns) {
    if (!matchesGroupBroadcastLoadPattern(pattern, query)) {
      continue;
    }
    return materializeGroupBroadcastLoadFact(pattern, query);
  }

  int64_t alignedStrideElems = kLayoutBlockBitWidth / elementBits;
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
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  if (!isa<PtrType>(sourceType)) {
    return fail("group_broadcast_load direct lowering requires !pto.ptr source");
  }

  unsigned elementBits =
      pto::getPTOStorageElemBitWidth(resultType.getElementType());
  if (elementBits == 0) {
    return fail("group_broadcast_load requires known element bit width");
  }
  std::optional<int64_t> stride = getConstantIndexValue(sourceGroupStride);

  FailureOr<GroupLayoutKey> key = buildGroupLayoutKey(
      resultType, numGroups,
      "group_broadcast_load preferred layout table has no row for this group "
      "size",
      reason);
  if (failed(key)) {
    return failure();
  }

  VMILayoutAttr existing = resultType.getLayoutAttr();
  GroupBroadcastLoadQuery query{resultType, existing, stride, *key, numGroups,
                                elementBits};
  for (const GroupBroadcastLoadDirectPattern &pattern :
       kGroupBroadcastLoadDirectPatterns) {
    if (!matchesGroupBroadcastLoadDirectPattern(pattern, query)) {
      continue;
    }
    VMILayoutAttr resultLayout = materializeLayoutPattern(
        resultType.getContext(), pattern.resultLayout, numGroups);
    if (existing && existing != resultLayout) {
      continue;
    }
    return VMIGroupBroadcastLoadDirectFact{
        pattern.kind,
        VMIGroupBroadcastLoadLayoutFact{
            getGroupBlockClassFromPattern(pattern.block),
            resultLayout,
            query.key.groupSize,
            query.key.lanesPerPart,
            query.key.vcgBlockElems,
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
                                            VMILayoutAttr resultLayout,
                                            VMICastLayoutPriority priority =
                                                VMICastLayoutPriority::Normal) {
  VMICastLayoutFact fact;
  fact.sourceBits = sourceBits;
  fact.resultBits = resultBits;
  fact.sourceLayout = sourceLayout;
  fact.resultLayout = resultLayout;
  fact.priority = priority;
  return fact;
}

static FailureOr<VMICastLayoutFact>
getHighPriorityCastLayoutFactImpl(VMIVRegType sourceType,
                                  VMIVRegType resultType,
                                  bool allowLaneStrideNarrowing,
                                  std::string *reason) {
  auto [sourceBits, resultBits] = getCastElementBits(sourceType, resultType);
  if (!allowLaneStrideNarrowing && sourceBits > resultBits) {
    return failure();
  }

  MLIRContext *ctx = sourceType.getContext();
  std::optional<VMICastLayoutFact> selected;
  for (const HighPriorityCastLayoutPattern &pattern :
       kHighPriorityCastLayoutPatterns) {
    if (!matchesElementBitsPattern(pattern.sourceBits, sourceBits) ||
        !matchesElementBitsPattern(pattern.resultBits, resultBits) ||
        !matchesCastTypeClass(pattern.typeClass, sourceType.getElementType(),
                              resultType.getElementType())) {
      continue;
    }

    VMILayoutAttr sourceLayout =
        materializeLayoutPattern(ctx, pattern.sourceLayout);
    VMILayoutAttr resultLayout =
        materializeLayoutPattern(ctx, pattern.resultLayout);
    auto assignedSourceType = VMIVRegType::get(
        ctx, sourceType.getElementCount(), sourceType.getElementType(),
        sourceLayout);
    auto assignedResultType = VMIVRegType::get(
        ctx, resultType.getElementCount(), resultType.getElementType(),
        resultLayout);
    FailureOr<int64_t> sourceArity = getVMIPhysicalArity(assignedSourceType);
    FailureOr<int64_t> resultArity = getVMIPhysicalArity(assignedResultType);
    if (failed(sourceArity) || failed(resultArity) ||
        !matchesPhysicalChunkCountPattern(pattern.sourceChunks,
                                          *sourceArity) ||
        !matchesPhysicalChunkCountPattern(pattern.resultChunks,
                                          *resultArity)) {
      continue;
    }
    if (selected) {
      if (reason) {
        *reason = "high-priority cast layout table has ambiguous matching rows";
      }
      return failure();
    }
    selected = makeCastLayoutFact(sourceBits, resultBits, sourceLayout,
                                  resultLayout,
                                  VMICastLayoutPriority::High);
  }
  if (!selected) {
    if (reason) {
      *reason = "requires a matching high-priority cast layout table row";
    }
    return failure();
  }
  return *selected;
}

static int64_t getMaskGranularityBits(StringRef granularity) {
  if (granularity == "b8") {
    return mlir::pto::kValue8;
  }
  if (granularity == "b16") {
    return mlir::pto::kValue16;
  }
  if (granularity == "b32") {
    return mlir::pto::kValue32;
  }
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

struct PreferredCastPatternQuery {
  int64_t sourceBits;
  int64_t resultBits;
  int64_t elementCount;
  Type sourceElementType;
  Type resultElementType;
  StringRef tableName;
  std::string *reason;
};

struct PreferredCastLayoutRequest {
  VMIVRegType sourceType;
  VMIVRegType resultType;
  StringRef tableName;
  std::string *reason;
  VMICastLayoutPriority priority;
};

static FailureOr<const PreferredCastLayoutPattern *>
selectPreferredCastLayoutPattern(
    ArrayRef<PreferredCastLayoutPattern> patterns,
    const PreferredCastPatternQuery &query) {
  const PreferredCastLayoutPattern *selected = nullptr;
  bool selectedIsExact = false;
  for (const PreferredCastLayoutPattern &pattern : patterns) {
    bool elementBitsMismatch =
        !matchesElementBitsPattern(pattern.sourceBits, query.sourceBits) ||
        !matchesElementBitsPattern(pattern.resultBits, query.resultBits);
    if (elementBitsMismatch ||
        !matchesCastTypeClass(pattern.typeClass, query.sourceElementType,
                              query.resultElementType)) {
      continue;
    }
    bool isExact = pattern.elementCount != 0;
    if (isExact && pattern.elementCount != query.elementCount) {
      continue;
    }
    if (!selected || (isExact && !selectedIsExact)) {
      selected = &pattern;
      selectedIsExact = isExact;
      continue;
    }
    if (isExact == selectedIsExact) {
      if (query.reason) {
        *query.reason =
            (Twine(query.tableName) + " has ambiguous matching rows").str();
      }
      return failure();
    }
  }

  if (!selected) {
    if (query.reason) {
      *query.reason =
          (Twine("requires a matching ") + query.tableName + " row").str();
    }
    return failure();
  }
  return selected;
}

static FailureOr<VMICastLayoutFact> getPreferredCastLayoutFactImpl(
    ArrayRef<PreferredCastLayoutPattern> patterns,
    const PreferredCastLayoutRequest &request) {
  auto [sourceBits, resultBits] =
      getCastElementBits(request.sourceType, request.resultType);
  PreferredCastPatternQuery query{
      sourceBits,
      resultBits,
      request.sourceType.getElementCount(),
      request.sourceType.getElementType(),
      request.resultType.getElementType(),
      request.tableName,
      request.reason};
  FailureOr<const PreferredCastLayoutPattern *> selected =
      selectPreferredCastLayoutPattern(patterns, query);
  if (failed(selected)) {
    return failure();
  }

  MLIRContext *ctx = request.sourceType.getContext();
  return makeCastLayoutFact(sourceBits, resultBits,
                            materializeLayoutPattern(ctx,
                                                     (*selected)->sourceLayout),
                            materializeLayoutPattern(ctx,
                                                     (*selected)->resultLayout),
                            request.priority);
}

static FailureOr<VMICastLayoutFact>
getPreferredLaneStrideNarrowCastLayoutFactImpl(VMIVRegType sourceType,
                                               VMIVRegType resultType,
                                               std::string *reason) {
  return getPreferredCastLayoutFactImpl(
      kPreferredLaneStrideNarrowCastLayoutPatterns,
      {sourceType, resultType,
       "preferred lane-stride narrow cast layout table", reason,
       VMICastLayoutPriority::LaneStrideNarrowing});
}

FailureOr<VMICastLayoutFact> VMILayoutSupport::getPreferredCastLayoutFact(
    VMIVRegType sourceType, VMIVRegType resultType, std::string *reason) const {
  FailureOr<VMICastLayoutFact> highPriorityFact =
      getHighPriorityCastLayoutFactImpl(sourceType, resultType,
                                        preferLaneStrideNarrowing, reason);
  if (succeeded(highPriorityFact)) {
    return highPriorityFact;
  }
  if (preferLaneStrideNarrowing) {
    FailureOr<VMICastLayoutFact> laneStrideFact =
        getPreferredLaneStrideNarrowCastLayoutFactImpl(sourceType, resultType,
                                                       reason);
    if (succeeded(laneStrideFact)) {
      return laneStrideFact;
    }
  }
  return getPreferredCastLayoutFactImpl(
      kPreferredCastLayoutPatterns,
      {sourceType, resultType, "preferred cast layout table", reason,
       VMICastLayoutPriority::Normal});
}

FailureOr<SmallVector<VMICastLayoutFact, mlir::pto::kValue4>>
VMILayoutSupport::getCastLayoutFactsForLayout(VMIVRegType sourceType,
                                              VMIVRegType resultType,
                                              VMICastLayoutPort port,
                                              VMILayoutAttr layout,
                                              std::string *reason) const {
  auto fail = [&](const Twine &message)
      -> FailureOr<SmallVector<VMICastLayoutFact, 4>> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  auto [sourceBits, resultBits] = getCastElementBits(sourceType, resultType);
  MLIRContext *ctx = sourceType.getContext();
  SmallVector<VMICastLayoutFact, mlir::pto::kValue4> facts;

  int64_t numGroups =
      layout && layout.isGroupSlots() ? layout.getNumGroups() : 0;
  for (const LegalCastLayoutPattern &pattern : kLegalCastLayoutPatterns) {
    if (!matchesElementBitsPattern(pattern.sourceBits, sourceBits) ||
        !matchesElementBitsPattern(pattern.resultBits, resultBits) ||
        !matchesCastTypeClass(pattern.typeClass, sourceType.getElementType(),
                              resultType.getElementType())) {
      continue;
    }

    VMILayoutAttr sourceLayout =
        materializeLayoutPattern(ctx, pattern.sourceLayout, numGroups);
    VMILayoutAttr resultLayout =
        materializeLayoutPattern(ctx, pattern.resultLayout, numGroups);
    if (!sourceLayout || !resultLayout) {
      continue;
    }

    if (port == VMICastLayoutPort::Source && sourceLayout != layout) {
      continue;
    }
    if (port == VMICastLayoutPort::Result && resultLayout != layout) {
      continue;
    }

    facts.push_back(
        makeCastLayoutFact(sourceBits, resultBits, sourceLayout, resultLayout));
  }

  if (facts.empty()) {
    if (port == VMICastLayoutPort::Source) {
      return fail("requires a legal cast relation for the source layout");
    }
    return fail("requires a legal cast relation for the result layout");
  }
  return facts;
}

static FailureOr<VMICastLayoutFact>
getUniqueCastLayoutFact(FailureOr<SmallVector<VMICastLayoutFact, mlir::pto::kValue4>> facts,
                        std::string *reason) {
  auto fail = [&](const Twine &message) -> FailureOr<VMICastLayoutFact> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  if (failed(facts)) {
    return failure();
  }
  if (facts->empty()) {
    return fail("cast layout query produced no layout facts");
  }
  if (facts->size() != 1) {
    return fail("cast layout query produced ambiguous layout facts");
  }
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
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  FailureOr<SmallVector<VMICastLayoutFact, mlir::pto::kValue4>> facts =
      getCastLayoutFactsForLayout(sourceType, resultType,
                                  VMICastLayoutPort::Source, sourceLayout,
                                  reason);
  if (failed(facts)) {
    return failure();
  }

  std::optional<VMICastLayoutFact> selected;
  for (const VMICastLayoutFact &fact : *facts) {
    if (fact.resultLayout != resultLayout) {
      continue;
    }
    if (selected) {
      return fail("cast layout query produced ambiguous layout facts");
    }
    selected = fact;
  }
  if (!selected) {
    return fail("source/result layouts do not match a legal cast table row");
  }
  return *selected;
}

struct MaskGranularityCastQuery {
  int64_t sourceBits;
  int64_t resultBits;
  int64_t numGroups;
};

static FailureOr<MaskGranularityCastQuery> buildMaskGranularityCastQuery(
    VMIMaskType sourceType, VMIMaskType resultType, VMILayoutAttr layout,
    std::string *reason) {
  auto fail = [&](const Twine &message) -> FailureOr<MaskGranularityCastQuery> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  if (sourceType.getElementCount() != resultType.getElementCount()) {
    return fail("requires source and result mask lane counts to match");
  }
  bool hasConcreteGranularities =
      VMIMaskType::isConcreteGranularity(sourceType.getGranularity()) &&
      VMIMaskType::isConcreteGranularity(resultType.getGranularity());
  if (!hasConcreteGranularities) {
    return fail("requires concrete b8/b16/b32 source and result "
                "granularities");
  }
  int64_t sourceBits = getMaskGranularityBits(sourceType.getGranularity());
  int64_t resultBits = getMaskGranularityBits(resultType.getGranularity());
  if (sourceBits == 0 || resultBits == 0) {
    return fail("requires supported source/result mask granularities");
  }
  int64_t numGroups =
      layout && layout.isGroupSlots() ? layout.getNumGroups() : 0;
  return MaskGranularityCastQuery{sourceBits, resultBits, numGroups};
}

static bool matchesMaskGranularityCastPattern(
    const LegalMaskGranularityCastLayoutPattern &pattern,
    VMIMaskType sourceType, VMIMaskType resultType) {
  return matchesMaskGranularityPattern(pattern.sourceGranularity,
                                       sourceType.getGranularity()) &&
         matchesMaskGranularityPattern(pattern.resultGranularity,
                                       resultType.getGranularity());
}

FailureOr<SmallVector<VMIMaskGranularityCastLayoutFact, mlir::pto::kValue4>>
VMILayoutSupport::getMaskGranularityCastLayoutFactsForLayout(
    VMIMaskType sourceType, VMIMaskType resultType, VMICastLayoutPort port,
    VMILayoutAttr layout, std::string *reason) const {
  auto fail = [&](const Twine &message)
      -> FailureOr<SmallVector<VMIMaskGranularityCastLayoutFact, 4>> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  FailureOr<MaskGranularityCastQuery> query =
      buildMaskGranularityCastQuery(sourceType, resultType, layout, reason);
  if (failed(query)) {
    return failure();
  }

  MLIRContext *ctx = sourceType.getContext();
  SmallVector<VMIMaskGranularityCastLayoutFact, mlir::pto::kValue4> facts;
  for (const LegalMaskGranularityCastLayoutPattern &pattern :
       kLegalMaskGranularityCastLayoutPatterns) {
    if (!matchesMaskGranularityCastPattern(pattern, sourceType, resultType)) {
      continue;
    }

    VMILayoutAttr sourceLayout =
        materializeLayoutPattern(ctx, pattern.sourceLayout, query->numGroups);
    VMILayoutAttr resultLayout =
        materializeLayoutPattern(ctx, pattern.resultLayout, query->numGroups);
    if (!sourceLayout || !resultLayout) {
      continue;
    }

    if (port == VMICastLayoutPort::Source && sourceLayout != layout) {
      continue;
    }
    if (port == VMICastLayoutPort::Result && resultLayout != layout) {
      continue;
    }

    facts.push_back(makeMaskGranularityCastLayoutFact(
        query->sourceBits, query->resultBits, sourceLayout, resultLayout));
  }

  if (facts.empty()) {
    if (port == VMICastLayoutPort::Source) {
      return fail("requires a legal mask granularity cast relation for the "
                  "source layout");
    }
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
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  FailureOr<SmallVector<VMIMaskGranularityCastLayoutFact, mlir::pto::kValue4>> facts =
      getMaskGranularityCastLayoutFactsForLayout(
          sourceType, resultType, VMICastLayoutPort::Source, sourceLayout,
          reason);
  if (failed(facts)) {
    return failure();
  }

  std::optional<VMIMaskGranularityCastLayoutFact> selected;
  for (const VMIMaskGranularityCastLayoutFact &fact : *facts) {
    if (fact.resultLayout != resultLayout) {
      continue;
    }
    if (selected) {
      return fail("mask granularity cast layout query produced ambiguous "
                  "layout facts");
    }
    selected = fact;
  }
  if (!selected) {
    return fail("source/result layouts do not match a legal mask granularity "
                "cast table row");
  }
  return *selected;
}

FailureOr<VMILayoutAttr> VMILayoutSupport::getWidenSourceLayoutForResultLayout(
    VMIVRegType sourceType, VMIVRegType resultType,
    VMILayoutAttr requestedResultLayout, std::string *reason) const {
  FailureOr<VMICastLayoutFact> fact = getCastLayoutFactForResultLayout(
      sourceType, resultType, requestedResultLayout, reason);
  if (failed(fact)) {
    return failure();
  }
  return fact->sourceLayout;
}

static FailureOr<VMIInterleaveLayoutFact> getPreferredInterleaveLayoutFactImpl(
    ArrayRef<InterleaveLayoutPattern> patterns, VMIVRegType valueType,
    std::string *reason) {
  auto fail = [&](const Twine &message) -> FailureOr<VMIInterleaveLayoutFact> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  FailureOr<InterleaveLayoutKey> key =
      buildInterleaveLayoutKey(valueType, reason);
  if (failed(key)) {
    return failure();
  }

  for (const InterleaveLayoutPattern &pattern : patterns) {
    if (!matchesInterleaveLayoutPattern(pattern, valueType, *key)) {
      continue;
    }
    return materializeInterleaveLayoutFact(valueType.getContext(), pattern,
                                           *key);
  }

  return fail("requires a preferred interleave layout table row");
}

static FailureOr<SmallVector<VMIInterleaveLayoutFact, mlir::pto::kValue4>>
getInterleaveLayoutFactsForLayoutImpl(
    ArrayRef<InterleaveLayoutPattern> patterns, VMIVRegType valueType,
    VMIInterleaveLayoutPort port, VMILayoutAttr layout,
    std::string *reason) {
  auto fail = [&](const Twine &message)
      -> FailureOr<SmallVector<VMIInterleaveLayoutFact, 4>> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  if (!layout) {
    return fail("requires assigned interleave layout query port");
  }

  FailureOr<InterleaveLayoutKey> key =
      buildInterleaveLayoutKey(valueType, reason);
  if (failed(key)) {
    return failure();
  }

  SmallVector<VMIInterleaveLayoutFact, mlir::pto::kValue4> facts;
  for (const InterleaveLayoutPattern &pattern : patterns) {
    if (!matchesInterleaveLayoutPattern(pattern, valueType, *key)) {
      continue;
    }
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
    if (candidateLayout == layout) {
      facts.push_back(candidate);
    }
  }

  if (facts.empty()) {
    return fail("interleave layout query port does not match a legal layout "
                "table row for the vector shape");
  }
  return facts;
}

struct InterleaveTypes {
  VMIVRegType lhs;
  VMIVRegType rhs;
  VMIMaskType mask;
  VMIVRegType low;
  VMIVRegType high;
};

struct InterleaveLayouts {
  VMILayoutAttr lhs;
  VMILayoutAttr rhs;
  VMILayoutAttr mask;
  VMILayoutAttr low;
  VMILayoutAttr high;
};

static LogicalResult validateInterleaveTypes(InterleaveTypes types,
                                             std::string *reason) {
  auto fail = [reason](const Twine &message) {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  int64_t elementCount = types.lhs.getElementCount();
  bool laneCountsMatch = elementCount == types.rhs.getElementCount() &&
                         elementCount == types.low.getElementCount() &&
                         elementCount == types.high.getElementCount() &&
                         elementCount == types.mask.getElementCount();
  if (!laneCountsMatch) {
    return fail("interleave layout requires all ports to share logical lane "
                "count");
  }
  Type elementType = types.lhs.getElementType();
  bool elementTypesMatch = elementType == types.rhs.getElementType() &&
                           elementType == types.low.getElementType() &&
                           elementType == types.high.getElementType();
  if (!elementTypesMatch) {
    return fail("interleave layout requires all data ports to share element "
                "type");
  }
  return success();
}

static FailureOr<InterleaveLayouts>
getInterleaveLayouts(InterleaveTypes types, std::string *reason) {
  InterleaveLayouts layouts{types.lhs.getLayoutAttr(), types.rhs.getLayoutAttr(),
                            types.mask.getLayoutAttr(), types.low.getLayoutAttr(),
                            types.high.getLayoutAttr()};
  bool allAssigned = layouts.lhs && layouts.rhs && layouts.mask && layouts.low &&
                     layouts.high;
  if (!allAssigned) {
    if (reason) {
      *reason = "requires assigned lhs/rhs/mask/low/high layouts";
    }
    return failure();
  }
  return layouts;
}

static bool matchesInterleaveLayouts(const VMIInterleaveLayoutFact &fact,
                                     InterleaveLayouts layouts) {
  return fact.rhsLayout == layouts.rhs && fact.maskLayout == layouts.mask &&
         fact.lowLayout == layouts.low && fact.highLayout == layouts.high;
}

static FailureOr<VMIInterleaveLayoutFact> getInterleaveLayoutFactForLayoutsImpl(
    ArrayRef<InterleaveLayoutPattern> patterns, InterleaveTypes types,
    std::string *reason) {
  auto fail = [&](const Twine &message) -> FailureOr<VMIInterleaveLayoutFact> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  if (failed(validateInterleaveTypes(types, reason))) {
    return failure();
  }
  FailureOr<InterleaveLayouts> layouts = getInterleaveLayouts(types, reason);
  if (failed(layouts)) {
    return failure();
  }

  FailureOr<SmallVector<VMIInterleaveLayoutFact, mlir::pto::kValue4>> facts =
      getInterleaveLayoutFactsForLayoutImpl(
          patterns, types.lhs, VMIInterleaveLayoutPort::Lhs, layouts->lhs,
          reason);
  if (failed(facts)) {
    return failure();
  }

  std::optional<VMIInterleaveLayoutFact> selected;
  for (const VMIInterleaveLayoutFact &fact : *facts) {
    if (!matchesInterleaveLayouts(fact, *layouts)) {
      continue;
    }
    if (selected) {
      return fail("interleave layout query produced ambiguous layout facts");
    }
    selected = fact;
  }
  if (!selected) {
    return fail("lhs/rhs/mask/low/high layouts do not match a legal "
                "interleave layout table row");
  }
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

FailureOr<SmallVector<VMIInterleaveLayoutFact, mlir::pto::kValue4>>
VMILayoutSupport::getVintlvLayoutFactsForLayout(
    VMIVRegType valueType, VMIInterleaveLayoutPort port, VMILayoutAttr layout,
    std::string *reason) const {
  return getInterleaveLayoutFactsForLayoutImpl(
      kVintlvLayoutPatterns, valueType, port, layout, reason);
}

FailureOr<SmallVector<VMIInterleaveLayoutFact, mlir::pto::kValue4>>
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
      kVintlvLayoutPatterns,
      InterleaveTypes{lhsType, rhsType, maskType, lowType, highType}, reason);
}

FailureOr<VMIInterleaveLayoutFact>
VMILayoutSupport::getVdintlvLayoutFactForLayouts(
    VMIVRegType lhsType, VMIVRegType rhsType, VMIMaskType maskType,
    VMIVRegType lowType, VMIVRegType highType, std::string *reason) const {
  return getInterleaveLayoutFactForLayoutsImpl(
      kVdintlvLayoutPatterns,
      InterleaveTypes{lhsType, rhsType, maskType, lowType, highType}, reason);
}

FailureOr<VMILoadLayoutFact>
VMILayoutSupport::getLoadLayoutFact(VMIVRegType resultType,
                                    std::string *reason) const {
  auto fail = [&](const Twine &message) -> FailureOr<VMILoadLayoutFact> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  VMILayoutAttr layout = resultType.getLayoutAttr();
  if (!layout) {
    return fail("requires assigned result layout");
  }
  for (const DenseMemoryLayoutPattern &pattern : kDenseLoadLayoutPatterns) {
    if (!matchesElementBitsPattern(pattern.elementBits,
                                   resultType.getElementType())) {
      continue;
    }
    if (!matchesElementCountPattern(pattern.elementCounts,
                                    resultType.getElementCount())) {
      continue;
    }
    if (!matchesLayoutPattern(resultType.getContext(), pattern.layout, layout)) {
      continue;
    }
    return VMILoadLayoutFact{layout};
  }

  return fail("result layout does not match a supported dense load table row");
}

FailureOr<VMIDeinterleaveLoadLayoutFact>
VMILayoutSupport::getPreferredDeinterleaveLoadLayoutFact(
    VMIVRegType valueType, std::string *reason) const {
  auto fail =
      [&](const Twine &message) -> FailureOr<VMIDeinterleaveLoadLayoutFact> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  for (const DeinterleaveLoadLayoutPattern &pattern :
       kDeinterleaveLoadLayoutPatterns) {
    if (!matchesElementBitsPattern(pattern.elementBits,
                                   valueType.getElementType())) {
      continue;
    }
    return materializeDeinterleaveLoadLayoutFact(valueType.getContext(),
                                                 pattern);
  }
  return fail("requires a preferred deinterleave_load layout table row");
}

FailureOr<SmallVector<VMIDeinterleaveLoadLayoutFact, mlir::pto::kValue4>>
VMILayoutSupport::getDeinterleaveLoadLayoutFactsForLayout(
    VMIVRegType valueType, VMIDeinterleaveLoadLayoutPort port,
    VMILayoutAttr layout, std::string *reason) const {
  auto fail = [&](const Twine &message)
      -> FailureOr<SmallVector<VMIDeinterleaveLoadLayoutFact, 4>> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  if (!layout) {
    return fail("requires assigned deinterleave_load layout query port");
  }

  SmallVector<VMIDeinterleaveLoadLayoutFact, mlir::pto::kValue4> facts;
  for (const DeinterleaveLoadLayoutPattern &pattern :
       kDeinterleaveLoadLayoutPatterns) {
    if (!matchesElementBitsPattern(pattern.elementBits,
                                   valueType.getElementType())) {
      continue;
    }
    VMIDeinterleaveLoadLayoutFact candidate =
        materializeDeinterleaveLoadLayoutFact(valueType.getContext(), pattern);
    VMILayoutAttr candidateLayout =
        port == VMIDeinterleaveLoadLayoutPort::Low ? candidate.lowLayout
                                                   : candidate.highLayout;
    if (candidateLayout == layout) {
      facts.push_back(candidate);
    }
  }
  if (facts.empty()) {
    return fail("deinterleave_load layout query port does not match a legal "
                "layout table row");
  }
  return facts;
}

FailureOr<VMIDeinterleaveLoadLayoutFact>
VMILayoutSupport::getDeinterleaveLoadLayoutFactForLayouts(
    VMIVRegType lowType, VMIVRegType highType, std::string *reason) const {
  auto fail =
      [&](const Twine &message) -> FailureOr<VMIDeinterleaveLoadLayoutFact> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  if (lowType.getElementCount() != highType.getElementCount() ||
      lowType.getElementType() != highType.getElementType()) {
    return fail("deinterleave_load layout requires low/high to share shape "
                "and element type");
  }

  VMILayoutAttr lowLayout = lowType.getLayoutAttr();
  VMILayoutAttr highLayout = highType.getLayoutAttr();
  if (!lowLayout || !highLayout) {
    return fail("requires assigned low/high layouts");
  }

  FailureOr<SmallVector<VMIDeinterleaveLoadLayoutFact, mlir::pto::kValue4>> facts =
      getDeinterleaveLoadLayoutFactsForLayout(
          lowType, VMIDeinterleaveLoadLayoutPort::Low, lowLayout, reason);
  if (failed(facts)) {
    return failure();
  }
  for (const VMIDeinterleaveLoadLayoutFact &fact : *facts) {
    if (fact.highLayout == highLayout) {
      return fact;
    }
  }
  return fail("low/high layouts do not match a legal deinterleave_load layout "
              "table row");
}

FailureOr<VMIStoreLayoutFact>
VMILayoutSupport::getStoreLayoutFact(VMIVRegType valueType,
                                     std::string *reason) const {
  auto fail = [&](const Twine &message) -> FailureOr<VMIStoreLayoutFact> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  VMILayoutAttr layout = valueType.getLayoutAttr();
  if (!layout) {
    return fail("requires assigned value layout");
  }
  for (const DenseMemoryLayoutPattern &pattern : kDenseStoreLayoutPatterns) {
    if (!matchesElementBitsPattern(pattern.elementBits,
                                   valueType.getElementType())) {
      continue;
    }
    if (!matchesElementCountPattern(pattern.elementCounts,
                                    valueType.getElementCount())) {
      continue;
    }
    if (!matchesLayoutPattern(valueType.getContext(), pattern.layout, layout)) {
      continue;
    }
    return VMIStoreLayoutFact{layout};
  }

  return fail("value layout does not match a supported dense store table row");
}

FailureOr<VMIStoreLayoutFact>
VMILayoutSupport::getPreferredStoreLayoutFact(VMIVRegType valueType,
                                              std::string *reason) const {
  auto fail = [&](const Twine &message) -> FailureOr<VMIStoreLayoutFact> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  if (valueType.getLayoutAttr()) {
    return getStoreLayoutFact(valueType, reason);
  }

  for (const DenseMemoryLayoutPattern &pattern : kDenseStoreLayoutPatterns) {
    if (!pattern.preferred) {
      continue;
    }
    if (!matchesElementBitsPattern(pattern.elementBits,
                                   valueType.getElementType())) {
      continue;
    }
    if (!matchesElementCountPattern(pattern.elementCounts,
                                    valueType.getElementCount())) {
      continue;
    }
    VMILayoutAttr layout =
        materializeLayoutPattern(valueType.getContext(), pattern.layout);
    if (!layout) {
      continue;
    }
    return VMIStoreLayoutFact{layout};
  }

  return fail("value type does not match a preferred dense store table row");
}

FailureOr<VMIMaskedStoreLayoutFact> VMILayoutSupport::getMaskedStoreLayoutFact(
    VMIVRegType valueType, VMIMaskType maskType, std::string *reason) const {
  auto fail = [&](const Twine &message) -> FailureOr<VMIMaskedStoreLayoutFact> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  VMILayoutAttr valueLayout = valueType.getLayoutAttr();
  VMILayoutAttr maskLayout = maskType.getLayoutAttr();
  if (!valueLayout || !maskLayout) {
    return fail("requires assigned value/mask layouts");
  }
  for (const DenseMaskedStoreLayoutPattern &pattern :
       kDenseMaskedStoreLayoutPatterns) {
    if (!matchesElementBitsPattern(pattern.elementBits,
                                   valueType.getElementType())) {
      continue;
    }
    if (!matchesElementCountPattern(pattern.elementCounts,
                                    valueType.getElementCount())) {
      continue;
    }
    if (!matchesLayoutPattern(valueType.getContext(), pattern.valueLayout,
                              valueLayout)) {
      continue;
    }
    if (!matchesLayoutPattern(maskType.getContext(), pattern.maskLayout,
                              maskLayout)) {
      continue;
    }
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
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  VMILayoutAttr existingValueLayout = valueType.getLayoutAttr();
  VMILayoutAttr existingMaskLayout = maskType.getLayoutAttr();
  if (existingValueLayout && existingMaskLayout) {
    return getMaskedStoreLayoutFact(valueType, maskType, reason);
  }

  for (const DenseMaskedStoreLayoutPattern &pattern :
       kDenseMaskedStoreLayoutPatterns) {
    if (!pattern.preferred) {
      continue;
    }
    if (!matchesElementBitsPattern(pattern.elementBits,
                                   valueType.getElementType())) {
      continue;
    }
    if (!matchesElementCountPattern(pattern.elementCounts,
                                    valueType.getElementCount())) {
      continue;
    }

    VMILayoutAttr valueLayout =
        materializeLayoutPattern(valueType.getContext(), pattern.valueLayout);
    VMILayoutAttr maskLayout =
        materializeLayoutPattern(maskType.getContext(), pattern.maskLayout);
    if (!valueLayout || !maskLayout) {
      continue;
    }
    if (existingValueLayout && existingValueLayout != valueLayout) {
      continue;
    }
    if (existingMaskLayout && existingMaskLayout != maskLayout) {
      continue;
    }
    return VMIMaskedStoreLayoutFact{valueLayout, maskLayout};
  }

  return fail("value/mask types do not match a preferred dense masked store "
              "table row");
}

FailureOr<VMIMaskedLoadLayoutFact> VMILayoutSupport::getMaskedLoadLayoutFact(
    VMIVRegType resultType, VMIMaskType maskType, VMIVRegType passthruType,
    std::string *reason) const {
  auto fail = [&](const Twine &message) -> FailureOr<VMIMaskedLoadLayoutFact> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  VMILayoutAttr maskLayout = maskType.getLayoutAttr();
  VMILayoutAttr passthruLayout = passthruType.getLayoutAttr();
  if (!resultLayout || !maskLayout || !passthruLayout) {
    return fail("requires assigned result/mask/passthru layouts");
  }
  for (const DenseMaskedLoadLayoutPattern &pattern :
       kDenseMaskedLoadLayoutPatterns) {
    if (!matchesElementBitsPattern(pattern.elementBits,
                                   resultType.getElementType())) {
      continue;
    }
    if (!matchesLayoutPattern(resultType.getContext(), pattern.resultLayout,
                              resultLayout)) {
      continue;
    }
    if (!matchesLayoutPattern(maskType.getContext(), pattern.maskLayout,
                              maskLayout)) {
      continue;
    }
    if (!matchesLayoutPattern(passthruType.getContext(),
                              pattern.passthruLayout, passthruLayout)) {
      continue;
    }
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
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  if (!sourceLayout || !resultLayout) {
    return fail("requires assigned source/result layouts");
  }
  if (sourceLayout == resultLayout) {
    return success();
  }

  int64_t numGroups =
      sourceLayout.isGroupSlots()
          ? sourceLayout.getNumGroups()
          : (resultLayout.isGroupSlots() ? resultLayout.getNumGroups() : 0);

  for (const EnsureLayoutPattern &pattern : kEnsureLayoutPatterns) {
    if (!matchesElementBitsPattern(pattern.elementBits,
                                   sourceType.getElementType())) {
      continue;
    }
    if (!matchesElementCountPattern(pattern.elementCounts,
                                    sourceType.getElementCount())) {
      continue;
    }
    if (!matchesLayoutPattern(sourceType.getContext(), pattern.sourceLayout,
                              sourceLayout, numGroups)) {
      continue;
    }
    if (!matchesLayoutPattern(resultType.getContext(), pattern.resultLayout,
                              resultLayout, numGroups)) {
      continue;
    }
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
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  if (!sourceLayout || !resultLayout) {
    return fail("requires assigned source/result layouts");
  }
  if (sourceLayout == resultLayout) {
    return success();
  }

  for (const EnsureMaskLayoutPattern &pattern : kEnsureMaskLayoutPatterns) {
    if (!matchesMaskGranularityPattern(pattern.granularity,
                                       sourceType.getGranularity())) {
      continue;
    }
    if (!matchesElementCountPattern(pattern.elementCounts,
                                    sourceType.getElementCount())) {
      continue;
    }
    if (!matchesLayoutPattern(sourceType.getContext(), pattern.sourceLayout,
                              sourceLayout)) {
      continue;
    }
    if (!matchesLayoutPattern(resultType.getContext(), pattern.resultLayout,
                              resultLayout)) {
      continue;
    }
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
                                      resultLayout, reason))) {
    return failure();
  }
  return VMIEnsureLayoutFact{sourceLayout, resultLayout};
}

FailureOr<VMIEnsureMaskLayoutFact> VMILayoutSupport::getEnsureMaskLayoutFact(
    VMIMaskType sourceType, VMIMaskType resultType, std::string *reason) const {
  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (failed(matchEnsureMaskLayoutPattern(sourceType, resultType, sourceLayout,
                                          resultLayout, reason))) {
    return failure();
  }
  return VMIEnsureMaskLayoutFact{sourceLayout, resultLayout};
}

FailureOr<VMIGroupSlotLayoutFact> VMILayoutSupport::getGroupSlotLoadLayoutFact(
    VMIVRegType resultType, int64_t numGroups, std::string *reason) const {
  auto fail = [&](const Twine &message) -> FailureOr<VMIGroupSlotLayoutFact> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  VMILayoutAttr layout = resultType.getLayoutAttr();
  if (!layout) {
    return fail("requires assigned result layout");
  }

  if (!isSupportedGroupSlotMemoryLayout(layout, numGroups)) {
    return fail("result layout does not match a supported group_slot_load "
                "table row");
  }

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
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  VMILayoutAttr layout = resultType.getLayoutAttr();
  if (!layout) {
    return fail("requires assigned result layout");
  }

  unsigned elementBits =
      pto::getPTOStorageElemBitWidth(resultType.getElementType());
  if (elementBits == 0) {
    return fail("group_load requires known element bit width");
  }
  std::optional<int64_t> stride = getConstantIndexValue(rowStride);

  FailureOr<GroupLayoutKey> key = buildGroupLayoutKey(
      resultType, numGroups,
      "group_load layout table has no row for this group size", reason);
  if (failed(key)) {
    return failure();
  }

  for (const GroupLoadLayoutPattern &pattern : kGroupLoadLayoutPatterns) {
    if (!matchesElementBitsPattern(pattern.elementBits, elementBits)) {
      continue;
    }
    if (!matchesGroupBlockPattern(pattern.block, *key)) {
      continue;
    }
    if (!matchesGroupLoadMemoryPattern(pattern.memory, stride, key->groupSize,
                                       elementBits)) {
      continue;
    }
    if (!matchesLayoutPattern(resultType.getContext(), pattern.resultLayout,
                              layout)) {
      continue;
    }
    return VMIGroupLoadLayoutFact{
        getGroupBlockClassFromPattern(pattern.block), layout, key->groupSize};
  }

  return fail("result layout, group size, and row_stride do not match a "
              "supported group_load table row");
}

FailureOr<VMIGroupSlotLayoutFact> VMILayoutSupport::getGroupStoreLayoutFact(
    VMIVRegType valueType, int64_t numGroups, std::string *reason) const {
  auto fail = [&](const Twine &message) -> FailureOr<VMIGroupSlotLayoutFact> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  VMILayoutAttr layout = valueType.getLayoutAttr();
  if (!layout) {
    return fail("requires assigned value layout");
  }
  if (!isSupportedGroupSlotMemoryLayout(layout, numGroups)) {
    return fail("value layout does not match a supported group_store table "
                "row");
  }
  return VMIGroupSlotLayoutFact{layout, numGroups, layout.getSlots()};
}

FailureOr<VMIGroupStoreLayoutFact> VMILayoutSupport::getGroupStoreLayoutFact(
    VMIGroupStoreOp op, VMIVRegType valueType, std::string *reason) const {
  auto fail = [&](const Twine &message) -> FailureOr<VMIGroupStoreLayoutFact> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  VMILayoutAttr layout = valueType.getLayoutAttr();
  if (!layout) {
    return fail("requires assigned value layout");
  }

  int64_t numGroups = op.getNumGroupsAttr().getInt();
  if (layout.isGroupSlots()) {
    if (failed(getGroupStoreLayoutFact(valueType, numGroups, reason))) {
      return failure();
    }
    return VMIGroupStoreLayoutFact{layout};
  }

  if (pto::getPTOStorageElemBitWidth(valueType.getElementType()) == 0) {
    return fail("group_store requires known element bit width");
  }

  FailureOr<GroupLayoutKey> key = buildGroupLayoutKey(
      valueType, numGroups,
      "group_store layout table has no row for this group size", reason);
  if (failed(key)) {
    return failure();
  }

  std::optional<int64_t> rowStride =
      getConstantIndexValue(op.getRowStride());
  for (const GroupStoreLayoutPattern &pattern : kGroupStoreLayoutPatterns) {
    if (!matchesGroupStoreLayoutPattern(pattern, valueType, *key, rowStride)) {
      continue;
    }
    if (!matchesLayoutPattern(valueType.getContext(), pattern.valueLayout,
                              layout)) {
      continue;
    }
    VMIGroupStoreLayoutFact fact =
        materializeGroupStoreLayoutFact(valueType.getContext(), pattern, *key);
    fact.valueLayout = layout;
    return fact;
  }

  return fail("value layout, group size, and row_stride do not match a "
              "supported group_store table row");
}


FailureOr<SmallVector<VMIGroupStoreLayoutFact, mlir::pto::kValue4>>
VMILayoutSupport::getGroupStoreLayoutFactsForLayout(
    VMIGroupStoreOp op, VMIVRegType valueType, VMILayoutAttr layout,
    std::string *reason) const {
  auto fail = [&](const Twine &message)
      -> FailureOr<SmallVector<VMIGroupStoreLayoutFact, 4>> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  if (!layout) {
    return fail("requires assigned group_store value layout");
  }

  MLIRContext *ctx = valueType.getContext();
  auto sourceType = VMIVRegType::get(ctx, valueType.getElementCount(),
                                    valueType.getElementType(), layout);
  FailureOr<VMIGroupStoreLayoutFact> directFact =
      getGroupStoreLayoutFact(op, sourceType, nullptr);
  if (succeeded(directFact)) {
    return SmallVector<VMIGroupStoreLayoutFact, mlir::pto::kValue4>{*directFact};
  }

  FailureOr<GroupLayoutKey> key = buildGroupLayoutKey(
      valueType, op.getNumGroupsAttr().getInt(),
      "group_store layout table has no row for this group size", reason);
  if (failed(key)) {
    return failure();
  }

  std::optional<int64_t> rowStride =
      getConstantIndexValue(op.getRowStride());
  SmallVector<VMIGroupStoreLayoutFact, mlir::pto::kValue4> facts;
  appendMaterializableGroupStoreFacts(
      *this, {valueType, sourceType, *key, rowStride}, facts);

  if (facts.empty()) {
    return fail("value layout cannot be used directly or materialized to a "
                "supported group_store layout table row");
  }
  return facts;
}

FailureOr<VMIGroupStoreLayoutFact>
VMILayoutSupport::getPreferredGroupStoreLayoutFact(
    VMIGroupStoreOp op, VMIVRegType valueType, std::string *reason) const {
  auto fail = [&](const Twine &message) -> FailureOr<VMIGroupStoreLayoutFact> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  int64_t numGroups = op.getNumGroupsAttr().getInt();
  std::optional<VMIGroupStoreLayoutFact> slotFact =
      getPreferredGroupSlotStoreFact(*this, op, valueType, numGroups);
  if (slotFact) {
    return *slotFact;
  }

  if (pto::getPTOStorageElemBitWidth(valueType.getElementType()) == 0) {
    return fail("group_store requires known element bit width");
  }
  FailureOr<GroupLayoutKey> key = buildGroupLayoutKey(
      valueType, numGroups,
      "group_store preferred layout table has no row for this group size",
      reason);
  if (failed(key)) {
    return failure();
  }

  std::optional<int64_t> rowStride =
      getConstantIndexValue(op.getRowStride());
  const GroupStoreLayoutPattern *selected =
      selectPreferredGroupStorePattern(valueType, *key, rowStride);
  if (selected) {
    return materializeGroupStoreLayoutFact(valueType.getContext(), *selected,
                                           *key);
  }

  return fail("value type, group size, and row_stride do not match a "
              "preferred group_store table row");
}

FailureOr<VMIGroupStoreLayoutFact>
VMILayoutSupport::getHighPriorityGroupStoreLayoutFact(
    VMIGroupStoreOp op, VMIVRegType valueType, std::string *reason) const {
  auto fail = [&](const Twine &message) -> FailureOr<VMIGroupStoreLayoutFact> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  if (pto::getPTOStorageElemBitWidth(valueType.getElementType()) == 0) {
    return fail("group_store requires known element bit width");
  }
  FailureOr<GroupLayoutKey> key = buildGroupLayoutKey(
      valueType, op.getNumGroupsAttr().getInt(),
      "high-priority group_store layout table has no row for this group size",
      reason);
  if (failed(key)) {
    return failure();
  }

  std::optional<int64_t> rowStride =
      getConstantIndexValue(op.getRowStride());
  for (const GroupStoreLayoutPattern &pattern : kGroupStoreLayoutPatterns) {
    if (pattern.priority != GroupStoreLayoutPriority::High) {
      continue;
    }
    if (!matchesGroupStoreLayoutPattern(pattern, valueType, *key, rowStride)) {
      continue;
    }
    VMIGroupStoreLayoutFact fact = materializeGroupStoreLayoutFact(
        valueType.getContext(), pattern, *key);
    if (!fact.valueLayout) {
      continue;
    }
    return fact;
  }

  return fail("value type, group size, and row_stride do not match a "
              "high-priority group_store table row");
}

FailureOr<VMIBitcastLayoutFact>
VMILayoutSupport::getBitcastLayoutFact(VMIBitcastOp op,
                                       std::string *reason) const {
  auto fail = [&](const Twine &message) -> FailureOr<VMIBitcastLayoutFact> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!sourceLayout || !resultLayout) {
    return fail("requires assigned source and result layouts");
  }
  if (sourceLayout != resultLayout) {
    return fail("requires matching source and result layouts");
  }

  int64_t numGroups =
      sourceLayout.isGroupSlots() ? sourceLayout.getNumGroups() : 0;
  unsigned sourceElementBits =
      pto::getPTOStorageElemBitWidth(sourceType.getElementType());
  unsigned resultElementBits =
      pto::getPTOStorageElemBitWidth(resultType.getElementType());
  if (sourceElementBits == 0 || resultElementBits == 0) {
    return fail("requires source and result with known storage element width");
  }
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
    if (!matchedLayout) {
      return fail("width-changing bitcast layout does not match a bitcast "
                  "layout table row");
    }
  }

  return VMIBitcastLayoutFact{sourceLayout, resultLayout};
}

FailureOr<SmallVector<VMIBitcastLayoutFact, mlir::pto::kValue4>>
VMILayoutSupport::getBitcastLayoutFactsForLayout(
    VMIVRegType sourceType, VMIVRegType resultType, VMICastLayoutPort port,
    VMILayoutAttr layout, std::string *reason) const {
  auto fail = [&](const Twine &message)
      -> FailureOr<SmallVector<VMIBitcastLayoutFact, 4>> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  if (!layout) {
    return fail("requires an assigned bitcast query layout");
  }

  unsigned sourceElementBits =
      pto::getPTOStorageElemBitWidth(sourceType.getElementType());
  unsigned resultElementBits =
      pto::getPTOStorageElemBitWidth(resultType.getElementType());
  if (sourceElementBits == 0 || resultElementBits == 0) {
    return fail("requires source and result with known storage element width");
  }

  if (sourceElementBits == resultElementBits) {
    return SmallVector<VMIBitcastLayoutFact, mlir::pto::kValue4>{
        VMIBitcastLayoutFact{layout, layout}};
  }

  int64_t numGroups = layout.isGroupSlots() ? layout.getNumGroups() : 0;
  MLIRContext *ctx = sourceType.getContext();
  SmallVector<VMIBitcastLayoutFact, mlir::pto::kValue4> facts;
  for (const WidthChangingBitcastLayoutPattern &pattern :
       kWidthChangingBitcastLayoutPatterns) {
    VMILayoutAttr candidate =
        materializeLayoutPattern(ctx, pattern.layout, numGroups);
    if (!candidate) {
      continue;
    }
    if (port == VMICastLayoutPort::Source && candidate != layout) {
      continue;
    }
    if (port == VMICastLayoutPort::Result && candidate != layout) {
      continue;
    }
    facts.push_back(VMIBitcastLayoutFact{candidate, candidate});
  }

  if (facts.empty()) {
    if (port == VMICastLayoutPort::Source) {
      return fail(
          "requires a legal width-changing bitcast source layout relation");
    }
    return fail(
        "requires a legal width-changing bitcast result layout relation");
  }
  return facts;
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
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  if (patterns.empty()) {
    return fail(opName + " histogram layout table has no row");
  }

  auto accType = cast<VMIVRegType>(op.getAcc().getType());
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());

  VMILayoutAttr accLayout = accType.getLayoutAttr();
  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr maskLayout = maskType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!accLayout || !sourceLayout || !maskLayout || !resultLayout) {
    return fail("requires assigned acc/source/mask/result layouts");
  }

  MLIRContext *ctx = op.getContext();
  for (const HistogramLayoutPattern &pattern : patterns) {
    if (!matchesLayoutPattern(ctx, pattern.accLayout, accLayout) ||
        !matchesLayoutPattern(ctx, pattern.sourceLayout, sourceLayout) ||
        !matchesLayoutPattern(ctx, pattern.maskLayout, maskLayout) ||
        !matchesLayoutPattern(ctx, pattern.resultLayout, resultLayout)) {
      continue;
    }

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

} // namespace pto
} // namespace mlir

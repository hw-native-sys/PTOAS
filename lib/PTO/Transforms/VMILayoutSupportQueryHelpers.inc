// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.


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

static bool matchesGroupBroadcastLoadPattern(
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

static VMIGroupBroadcastLoadLayoutFact materializeGroupBroadcastLoadFact(
    const GroupBroadcastLoadLayoutPattern &pattern,
    const GroupBroadcastLoadQuery &query) {
  return {getGroupBlockClassFromPattern(pattern.block),
          query.resultLayout,
          query.key.groupSize,
          query.key.lanesPerPart,
          query.key.vcgBlockElems,
          static_cast<int64_t>(query.elementBits)};
}

static bool matchesGroupBroadcastLoadDirectPattern(
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

static void appendMaterializableGroupStoreFacts(
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
        facts, [&](const VMIGroupStoreLayoutFact &fact) {
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

static std::optional<VMIGroupStoreLayoutFact>
getPreferredGroupSlotStoreFact(const VMILayoutSupport &support,
                               VMIGroupStoreOp op, VMIVRegType valueType,
                               int64_t numGroups) {
  bool hasOneLanePerGroup = valueType.getElementCount() == numGroups;
  if (!hasOneLanePerGroup) {
    return std::nullopt;
  }
  std::optional<int64_t> rowStride = getConstantIndexValue(op.getRowStride());
  bool packedSlots = rowStride && *rowStride == 1 &&
                     numGroups >= mlir::pto::kValue8;
  int64_t slots = packedSlots ? mlir::pto::kValue8 : 1;
  VMILayoutAttr layout = VMILayoutAttr::getGroupSlots(
      valueType.getContext(), numGroups, slots);
  auto assignedType = VMIVRegType::get(
      valueType.getContext(), valueType.getElementCount(),
      valueType.getElementType(), layout);
  if (failed(support.getGroupStoreLayoutFact(op, assignedType, nullptr))) {
    return std::nullopt;
  }
  return VMIGroupStoreLayoutFact{layout};
}

static const GroupStoreLayoutPattern *selectPreferredGroupStorePattern(
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

} // namespace

//===----------------------------------------------------------------------===//
// Support-query adapters
//===----------------------------------------------------------------------===//

static LogicalResult getGroupReduceAddSupportImpl(VMIVRegType sourceType,
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

static LogicalResult getNarrowCastSupport(VMIVRegType sourceType,
                                          VMIVRegType resultType,
                                          std::string *reason) {
  return success(succeeded(VMILayoutSupport().getCastLayoutFactForLayouts(
      sourceType, resultType, sourceType.getLayoutAttr(),
      resultType.getLayoutAttr(), reason)));
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
  return success(succeeded(VMILayoutSupport().getCastLayoutFactForLayouts(
      sourceType, resultType, sourceType.getLayoutAttr(),
      resultType.getLayoutAttr(), reason)));
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

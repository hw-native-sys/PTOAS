// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMILayoutSupportVisitors.inc - Lazy layout fact visitors -*- C++ -*-===//
//===----------------------------------------------------------------------===//
//
// This file owns constraint-aware traversal of the layout support tables.
// Exact, anchored, preferred, and operation-level queries should filter these
// visitors instead of materializing a temporary vector of every relation.
//
//===----------------------------------------------------------------------===//

namespace {

template <typename Fact>
using LayoutFactVisitor = llvm::function_ref<void(const Fact &)>;

template <typename Fact, typename Predicate>
static size_t visitMatchingFacts(ArrayRef<Fact> facts, Predicate predicate,
                                 LayoutFactVisitor<Fact> visitor) {
  size_t count = 0;
  for (const Fact &fact : facts) {
    if (!predicate(fact)) {
      continue;
    }
    visitor(fact);
    ++count;
  }
  return count;
}

static FailureOr<size_t> failFactVisit(const Twine &message,
                                       std::string *reason) {
  if (reason) {
    *reason = message.str();
  }
  return failure();
}

static size_t visitDenseMemoryFacts(
    ArrayRef<DenseMemoryLayoutPattern> patterns, VMIVRegType type,
    llvm::function_ref<bool(const DenseMemoryLayoutPattern &)> filter,
    LayoutFactVisitor<VMILayoutAttr> visitor) {
  size_t count = 0;
  for (const DenseMemoryLayoutPattern &pattern : patterns) {
    if (!matchesElementBitsPattern(pattern.elementBits,
                                   type.getElementType()) ||
        !matchesElementCountPattern(pattern.elementCounts,
                                    type.getElementCount()) ||
        !filter(pattern)) {
      continue;
    }
    VMILayoutAttr layout =
        materializeLayoutPattern(type.getContext(), pattern.layout);
    if (!layout) {
      continue;
    }
    visitor(layout);
    ++count;
  }
  return count;
}

static size_t
visitLoadLayoutFacts(VMIVRegType resultType,
                     LayoutFactVisitor<VMILoadLayoutFact> visitor) {
  return visitDenseMemoryFacts(
      kDenseLoadLayoutPatterns, resultType,
      [](const DenseMemoryLayoutPattern &) { return true; },
      [&](VMILayoutAttr layout) { visitor(VMILoadLayoutFact{layout}); });
}

static size_t visitStoreLayoutFacts(
    VMIVRegType valueType,
    llvm::function_ref<bool(const DenseMemoryLayoutPattern &)> filter,
    LayoutFactVisitor<VMIStoreLayoutFact> visitor) {
  return visitDenseMemoryFacts(
      kDenseStoreLayoutPatterns, valueType, filter,
      [&](VMILayoutAttr layout) { visitor(VMIStoreLayoutFact{layout}); });
}

static size_t visitDeinterleaveLoadLayoutFacts(
    VMIVRegType valueType,
    LayoutFactVisitor<VMIDeinterleaveLoadLayoutFact> visitor) {
  size_t count = 0;
  for (const DeinterleaveLoadLayoutPattern &pattern :
       kDeinterleaveLoadLayoutPatterns) {
    if (!matchesElementBitsPattern(pattern.elementBits,
                                   valueType.getElementType())) {
      continue;
    }
    visitor(
        materializeDeinterleaveLoadLayoutFact(valueType.getContext(), pattern));
    ++count;
  }
  return count;
}

static size_t visitMaskedStoreLayoutFacts(
    VMIVRegType valueType,
    llvm::function_ref<bool(const DenseMaskedStoreLayoutPattern &)> filter,
    LayoutFactVisitor<VMIMaskedStoreLayoutFact> visitor) {
  size_t count = 0;
  for (const DenseMaskedStoreLayoutPattern &pattern :
       kDenseMaskedStoreLayoutPatterns) {
    if (!matchesElementBitsPattern(pattern.elementBits,
                                   valueType.getElementType()) ||
        !matchesElementCountPattern(pattern.elementCounts,
                                    valueType.getElementCount()) ||
        !filter(pattern)) {
      continue;
    }
    visitor(
        {materializeLayoutPattern(valueType.getContext(), pattern.valueLayout),
         materializeLayoutPattern(valueType.getContext(), pattern.maskLayout)});
    ++count;
  }
  return count;
}

static size_t
visitMaskedLoadLayoutFacts(VMIVRegType resultType,
                           LayoutFactVisitor<VMIMaskedLoadLayoutFact> visitor) {
  size_t count = 0;
  for (const DenseMaskedLoadLayoutPattern &pattern :
       kDenseMaskedLoadLayoutPatterns) {
    if (!matchesElementBitsPattern(pattern.elementBits,
                                   resultType.getElementType())) {
      continue;
    }
    visitor(
        {materializeLayoutPattern(resultType.getContext(),
                                  pattern.resultLayout),
         materializeLayoutPattern(resultType.getContext(), pattern.maskLayout),
         materializeLayoutPattern(resultType.getContext(),
                                  pattern.passthruLayout)});
    ++count;
  }
  return count;
}

static FailureOr<size_t> visitInterleaveLayoutFacts(
    ArrayRef<InterleaveLayoutPattern> patterns, VMIVRegType valueType,
    LayoutFactVisitor<VMIInterleaveLayoutFact> visitor, std::string *reason) {
  FailureOr<InterleaveLayoutKey> key =
      buildInterleaveLayoutKey(valueType, reason);
  if (failed(key)) {
    return failure();
  }
  size_t count = 0;
  for (const InterleaveLayoutPattern &pattern : patterns) {
    if (!matchesInterleaveLayoutPattern(pattern, valueType, *key)) {
      continue;
    }
    visitor(
        materializeInterleaveLayoutFact(valueType.getContext(), pattern, *key));
    ++count;
  }
  return count;
}

static SmallVector<int64_t, 2>
getDomainGroupCounts(ArrayRef<VMILayoutRelationPortDomain> domains) {
  SmallVector<int64_t, 2> groupCounts;
  for (const VMILayoutRelationPortDomain &domain : domains) {
    for (VMILayoutAttr layout : domain.layouts) {
      bool hasNewGroupCount =
          layout && layout.isGroupSlots() && layout.getNumGroups() > 0 &&
          !llvm::is_contained(groupCounts, layout.getNumGroups());
      if (!hasNewGroupCount) {
        continue;
      }
      groupCounts.push_back(layout.getNumGroups());
    }
  }
  return groupCounts;
}

static size_t visitCastLayoutFactsForGroupCount(
    VMIVRegType sourceType, VMIVRegType resultType, int64_t numGroups,
    LayoutFactVisitor<VMICastLayoutFact> visitor) {
  auto [sourceBits, resultBits] = getCastElementBits(sourceType, resultType);
  size_t count = 0;
  for (const LegalCastLayoutPattern &pattern : kLegalCastLayoutPatterns) {
    bool groupParameterized =
        pattern.sourceLayout.kind == LayoutPatternKind::GroupSlots ||
        pattern.resultLayout.kind == LayoutPatternKind::GroupSlots;
    if (groupParameterized != (numGroups != 0)) {
      continue;
    }
    bool matchesType =
        matchesElementBitsPattern(pattern.sourceBits, sourceBits) &&
        matchesElementBitsPattern(pattern.resultBits, resultBits) &&
        matchesCastTypeClass(pattern.typeClass, sourceType.getElementType(),
                             resultType.getElementType());
    if (!matchesType) {
      continue;
    }
    VMILayoutAttr sourceLayout = materializeLayoutPattern(
        sourceType.getContext(), pattern.sourceLayout, numGroups);
    VMILayoutAttr resultLayout = materializeLayoutPattern(
        sourceType.getContext(), pattern.resultLayout, numGroups);
    if (!sourceLayout || !resultLayout) {
      continue;
    }
    visitor(
        makeCastLayoutFact(sourceBits, resultBits, sourceLayout, resultLayout));
    ++count;
  }
  return count;
}

static FailureOr<size_t>
visitCastLayoutFacts(VMIVRegType sourceType, VMIVRegType resultType,
                     ArrayRef<VMILayoutRelationPortDomain> domains,
                     LayoutFactVisitor<VMICastLayoutFact> visitor,
                     std::string *reason) {
  SmallVector<int64_t, 2> groupCounts = getDomainGroupCounts(domains);
  if (domains.empty()) {
    return failFactVisit(
        "cast relation has parameterized group-slot rows and requires a "
        "finite port domain",
        reason);
  }
  size_t count =
      visitCastLayoutFactsForGroupCount(sourceType, resultType, 0, visitor);
  for (int64_t numGroups : groupCounts) {
    count += visitCastLayoutFactsForGroupCount(sourceType, resultType,
                                               numGroups, visitor);
  }
  return count;
}

static FailureOr<size_t> visitMaskGranularityCastLayoutFacts(
    VMIMaskType sourceType, VMIMaskType resultType,
    ArrayRef<VMILayoutRelationPortDomain> domains,
    LayoutFactVisitor<VMIMaskGranularityCastLayoutFact> visitor,
    std::string *reason) {
  bool hasConcreteMatchingTypes =
      sourceType.getElementCount() == resultType.getElementCount() &&
      VMIMaskType::isConcreteGranularity(sourceType.getGranularity()) &&
      VMIMaskType::isConcreteGranularity(resultType.getGranularity());
  if (!hasConcreteMatchingTypes) {
    return failFactVisit(
        "mask granularity cast requires equal lane counts and concrete "
        "source/result granularities",
        reason);
  }
  if (domains.empty()) {
    return failFactVisit(
        "mask granularity cast has parameterized group-slot rows and "
        "requires a finite port domain",
        reason);
  }
  int64_t sourceBits = getMaskGranularityBits(sourceType.getGranularity());
  int64_t resultBits = getMaskGranularityBits(resultType.getGranularity());
  SmallVector<int64_t, 3> groupCounts{0};
  llvm::append_range(groupCounts, getDomainGroupCounts(domains));
  size_t count = 0;
  for (int64_t numGroups : groupCounts) {
    for (const LegalMaskGranularityCastLayoutPattern &pattern :
         kLegalMaskGranularityCastLayoutPatterns) {
      bool groupParameterized =
          pattern.sourceLayout.kind == LayoutPatternKind::GroupSlots ||
          pattern.resultLayout.kind == LayoutPatternKind::GroupSlots;
      if (groupParameterized != (numGroups != 0)) {
        continue;
      }
      if (!matchesMaskGranularityCastPattern(pattern, sourceType, resultType)) {
        continue;
      }
      VMILayoutAttr sourceLayout = materializeLayoutPattern(
          sourceType.getContext(), pattern.sourceLayout, numGroups);
      VMILayoutAttr resultLayout = materializeLayoutPattern(
          sourceType.getContext(), pattern.resultLayout, numGroups);
      if (!sourceLayout || !resultLayout) {
        continue;
      }
      visitor(makeMaskGranularityCastLayoutFact(sourceBits, resultBits,
                                                sourceLayout, resultLayout));
      ++count;
    }
  }
  return count;
}

static FailureOr<size_t> visitEnsureLayoutFacts(
    VMIVRegType valueType, ArrayRef<VMILayoutRelationPortDomain> domains,
    LayoutFactVisitor<VMIEnsureLayoutFact> visitor, std::string *reason) {
  if (domains.empty()) {
    return failFactVisit(
        "ensure_layout includes a polymorphic identity relation and requires "
        "a finite port domain",
        reason);
  }
  SmallVector<VMILayoutAttr, 8> anchors;
  for (const VMILayoutRelationPortDomain &domain : domains) {
    for (VMILayoutAttr layout : domain.layouts) {
      if (layout && !llvm::is_contained(anchors, layout)) {
        anchors.push_back(layout);
      }
    }
  }
  size_t count = 0;
  for (VMILayoutAttr layout : anchors) {
    visitor({layout, layout});
    ++count;
  }
  SmallVector<int64_t, 3> groupCounts{0};
  llvm::append_range(groupCounts, getDomainGroupCounts(domains));
  for (int64_t numGroups : groupCounts) {
    for (const EnsureLayoutPattern &pattern : kEnsureLayoutPatterns) {
      bool groupParameterized =
          pattern.sourceLayout.kind == LayoutPatternKind::GroupSlots ||
          pattern.resultLayout.kind == LayoutPatternKind::GroupSlots;
      if (groupParameterized != (numGroups != 0)) {
        continue;
      }
      if (!matchesElementBitsPattern(pattern.elementBits,
                                     valueType.getElementType()) ||
          !matchesElementCountPattern(pattern.elementCounts,
                                      valueType.getElementCount())) {
        continue;
      }
      VMILayoutAttr sourceLayout = materializeLayoutPattern(
          valueType.getContext(), pattern.sourceLayout, numGroups);
      VMILayoutAttr resultLayout = materializeLayoutPattern(
          valueType.getContext(), pattern.resultLayout, numGroups);
      if (!sourceLayout || !resultLayout) {
        continue;
      }
      visitor({sourceLayout, resultLayout});
      ++count;
    }
  }
  return count;
}

static FailureOr<size_t> visitEnsureMaskLayoutFacts(
    VMIMaskType maskType, ArrayRef<VMILayoutRelationPortDomain> domains,
    LayoutFactVisitor<VMIEnsureMaskLayoutFact> visitor, std::string *reason) {
  if (domains.empty()) {
    return failFactVisit(
        "ensure_mask_layout includes a polymorphic identity relation and "
        "requires a finite port domain",
        reason);
  }
  SmallVector<VMILayoutAttr, 8> anchors;
  for (const VMILayoutRelationPortDomain &domain : domains) {
    for (VMILayoutAttr layout : domain.layouts) {
      if (layout && !llvm::is_contained(anchors, layout)) {
        anchors.push_back(layout);
      }
    }
  }
  size_t count = 0;
  for (VMILayoutAttr layout : anchors) {
    visitor({layout, layout});
    ++count;
  }
  for (const EnsureMaskLayoutPattern &pattern : kEnsureMaskLayoutPatterns) {
    if (!matchesMaskGranularityPattern(pattern.granularity,
                                       maskType.getGranularity()) ||
        !matchesElementCountPattern(pattern.elementCounts,
                                    maskType.getElementCount())) {
      continue;
    }
    visitor(
        {materializeLayoutPattern(maskType.getContext(), pattern.sourceLayout),
         materializeLayoutPattern(maskType.getContext(),
                                  pattern.resultLayout)});
    ++count;
  }
  return count;
}

static FailureOr<size_t>
visitBitcastLayoutFacts(VMIVRegType sourceType, VMIVRegType resultType,
                        ArrayRef<VMILayoutRelationPortDomain> domains,
                        LayoutFactVisitor<VMIBitcastLayoutFact> visitor,
                        std::string *reason) {
  unsigned sourceBits =
      pto::getPTOStorageElemBitWidth(sourceType.getElementType());
  unsigned resultBits =
      pto::getPTOStorageElemBitWidth(resultType.getElementType());
  if (sourceBits == 0 || resultBits == 0) {
    return failFactVisit(
        "bitcast requires known source/result storage element widths", reason);
  }
  if (sourceBits != resultBits) {
    size_t count = 0;
    for (const WidthChangingBitcastLayoutPattern &pattern :
         kWidthChangingBitcastLayoutPatterns) {
      VMILayoutAttr layout =
          materializeLayoutPattern(sourceType.getContext(), pattern.layout);
      visitor({layout, layout});
      ++count;
    }
    return count;
  }
  if (domains.empty()) {
    return failFactVisit(
        "equal-width bitcast is layout-polymorphic and requires a finite port "
        "domain",
        reason);
  }
  SmallVector<VMILayoutAttr, 8> anchors;
  for (const VMILayoutRelationPortDomain &domain : domains) {
    for (VMILayoutAttr layout : domain.layouts) {
      if (layout && !llvm::is_contained(anchors, layout)) {
        anchors.push_back(layout);
      }
    }
  }
  for (VMILayoutAttr layout : anchors) {
    visitor({layout, layout});
  }
  return anchors.size();
}

static FailureOr<size_t>
visitGroupReduceLayoutFacts(VMIVRegType sourceType, int64_t numGroups,
                            LayoutFactVisitor<VMIGroupReduceLayoutFact> visitor,
                            std::string *reason) {
  FailureOr<GroupLayoutKey> key = buildGroupLayoutKey(
      sourceType, numGroups,
      "group_reduce layout table has no row for this group size", reason);
  if (failed(key)) {
    return failure();
  }
  size_t count = 0;
  for (const GroupReduceLayoutPattern &pattern : kGroupReduceLayoutPatterns) {
    if (!matchesGroupBlockPattern(pattern.block, *key)) {
      continue;
    }
    visitor(materializeGroupReduceLayoutFact(sourceType.getContext(), pattern,
                                             *key, numGroups));
    ++count;
  }
  return count;
}

static FailureOr<size_t> visitGroupBroadcastLayoutFacts(
    VMIVRegType resultType, int64_t numGroups,
    LayoutFactVisitor<VMIGroupBroadcastLayoutFact> visitor,
    std::string *reason) {
  FailureOr<GroupLayoutKey> key = buildGroupLayoutKey(
      resultType, numGroups,
      "group_broadcast layout table has no row for this group size", reason);
  if (failed(key)) {
    return failure();
  }
  size_t count = 0;
  for (const GroupBroadcastLayoutPattern &pattern :
       kGroupBroadcastLayoutPatterns) {
    if (!matchesGroupBlockPattern(pattern.block, *key)) {
      continue;
    }
    visitor(materializeGroupBroadcastLayoutFact(resultType.getContext(),
                                                pattern, *key, numGroups));
    ++count;
  }
  return count;
}

static FailureOr<size_t> visitGroupLoadLayoutFacts(
    VMIVRegType resultType, Value rowStride, int64_t numGroups,
    LayoutFactVisitor<VMIGroupLoadLayoutFact> visitor, std::string *reason) {
  unsigned elementBits =
      pto::getPTOStorageElemBitWidth(resultType.getElementType());
  if (elementBits == 0) {
    return failFactVisit("group_load requires known element bit width", reason);
  }
  FailureOr<GroupLayoutKey> key = buildGroupLayoutKey(
      resultType, numGroups,
      "group_load layout table has no row for this group size", reason);
  if (failed(key)) {
    return failure();
  }
  std::optional<int64_t> stride = getConstantIndexValue(rowStride);
  size_t count = 0;
  for (const GroupLoadLayoutPattern &pattern : kGroupLoadLayoutPatterns) {
    bool matches =
        matchesElementBitsPattern(pattern.elementBits, elementBits) &&
        matchesGroupBlockPattern(pattern.block, *key) &&
        matchesGroupLoadMemoryPattern(pattern.memory, stride, key->groupSize,
                                      elementBits);
    if (!matches) {
      continue;
    }
    visitor({getGroupBlockClassFromPattern(pattern.block),
             materializeLayoutPattern(resultType.getContext(),
                                      pattern.resultLayout),
             key->groupSize});
    ++count;
  }
  return count;
}

static FailureOr<size_t> visitDirectGroupStoreLayoutFacts(
    VMIGroupStoreOp op, VMIVRegType valueType,
    LayoutFactVisitor<VMIGroupStoreLayoutFact> visitor, std::string *reason) {
  int64_t numGroups = op.getNumGroupsAttr().getInt();
  size_t count = 0;
  for (const GroupSlotMemoryLayoutPattern &pattern :
       kGroupSlotMemoryLayoutPatterns) {
    VMILayoutAttr layout = materializeLayoutPattern(valueType.getContext(),
                                                    pattern.layout, numGroups);
    if (!layout) {
      continue;
    }
    visitor(VMIGroupStoreLayoutFact{layout});
    ++count;
  }
  FailureOr<GroupLayoutKey> key = buildGroupLayoutKey(
      valueType, numGroups,
      "group_store layout table has no row for this group size", reason);
  if (failed(key)) {
    return count == 0 ? FailureOr<size_t>(failure()) : FailureOr<size_t>(count);
  }
  std::optional<int64_t> stride = getConstantIndexValue(op.getRowStride());
  for (const GroupStoreLayoutPattern &pattern : kGroupStoreLayoutPatterns) {
    if (!matchesGroupStoreLayoutPattern(pattern, valueType, *key, stride)) {
      continue;
    }
    visitor(
        materializeGroupStoreLayoutFact(valueType.getContext(), pattern, *key));
    ++count;
  }
  return count;
}

static size_t
visitGroupSlotLayoutFacts(MLIRContext *context, int64_t numGroups,
                          LayoutFactVisitor<VMIGroupSlotLayoutFact> visitor) {
  size_t count = 0;
  for (const GroupSlotMemoryLayoutPattern &pattern :
       kGroupSlotMemoryLayoutPatterns) {
    VMILayoutAttr layout =
        materializeLayoutPattern(context, pattern.layout, numGroups);
    if (!layout) {
      continue;
    }
    visitor({layout, numGroups, layout.getSlots()});
    ++count;
  }
  return count;
}

static FailureOr<size_t> visitGroupBroadcastLoadLayoutFacts(
    VMIGroupBroadcastLoadOp op,
    LayoutFactVisitor<VMIGroupBroadcastLoadLayoutFact> visitor,
    std::string *reason) {
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  unsigned elementBits =
      pto::getPTOStorageElemBitWidth(resultType.getElementType());
  if (elementBits == 0) {
    return failFactVisit(
        "group_broadcast_load requires known element bit width", reason);
  }
  int64_t numGroups = op.getNumGroupsAttr().getInt();
  FailureOr<GroupLayoutKey> key = buildGroupLayoutKey(
      resultType, numGroups,
      "group_broadcast_load layout table has no row for this group size",
      reason);
  if (failed(key)) {
    return failure();
  }
  std::optional<int64_t> stride =
      getConstantIndexValue(op.getSourceGroupStride());
  size_t count = 0;
  for (const GroupBroadcastLoadLayoutPattern &pattern :
       kGroupBroadcastLoadLayoutPatterns) {
    VMILayoutAttr layout = materializeLayoutPattern(
        resultType.getContext(), pattern.resultLayout, numGroups);
    GroupBroadcastLoadQuery query{resultType, layout,    stride,
                                  *key,       numGroups, elementBits};
    if (!matchesGroupBroadcastLoadPattern(pattern, query)) {
      continue;
    }
    visitor(materializeGroupBroadcastLoadFact(pattern, query));
    ++count;
  }
  return count;
}

static size_t
visitVselrLayoutFacts(VMIVselrOp op,
                      LayoutFactVisitor<VMIVselrLayoutFact> visitor) {
  size_t count = 0;
  for (const VselrLayoutPattern &pattern : kVselrLayoutPatterns) {
    VMIVselrLayoutFact fact =
        materializeVselrLayoutFact(op.getContext(), pattern);
    if (!matchesVselrLayoutPattern(pattern, op, fact)) {
      continue;
    }
    visitor(fact);
    ++count;
  }
  return count;
}

static size_t
visitHistogramLayoutFacts(MLIRContext *context,
                          ArrayRef<HistogramLayoutPattern> patterns,
                          LayoutFactVisitor<VMIHistogramLayoutFact> visitor) {
  size_t count = 0;
  for (const HistogramLayoutPattern &pattern : patterns) {
    visitor({materializeLayoutPattern(context, pattern.accLayout),
             materializeLayoutPattern(context, pattern.sourceLayout),
             materializeLayoutPattern(context, pattern.maskLayout),
             materializeLayoutPattern(context, pattern.resultLayout)});
    ++count;
  }
  return count;
}

} // namespace

namespace {

static LogicalResult
validatePortDomains(Operation *op,
                    ArrayRef<VMILayoutRelationPortDomain> domains,
                    std::string *reason) {
  auto fail = [&](const Twine &message) {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  for (size_t i = 0; i < domains.size(); ++i) {
    const VMILayoutRelationPortDomain &domain = domains[i];
    unsigned portCount = domain.kind == VMILayoutRelationPortKind::Operand
                             ? op->getNumOperands()
                             : op->getNumResults();
    if (domain.portNumber >= portCount) {
      return fail("layout relation domain references an out-of-range port");
    }
    Type type = domain.kind == VMILayoutRelationPortKind::Operand
                    ? op->getOperand(domain.portNumber).getType()
                    : op->getResult(domain.portNumber).getType();
    if (!isa<VMIVRegType, VMIMaskType>(type)) {
      return fail("layout relation domain references a non-layout port");
    }
    bool hasConcreteDomain =
        !domain.layouts.empty() &&
        !llvm::is_contained(domain.layouts, VMILayoutAttr());
    if (!hasConcreteDomain) {
      return fail("layout relation domain must contain concrete layouts");
    }
    for (size_t j = 0; j < i; ++j) {
      if (domains[j].kind == domain.kind &&
          domains[j].portNumber == domain.portNumber) {
        return fail("layout relation domain contains a duplicate port");
      }
    }
  }
  return success();
}

static VMILayoutRelationFact makeOperationFact(
    Operation *op,
    std::initializer_list<std::pair<unsigned, VMILayoutAttr>> operands,
    std::initializer_list<std::pair<unsigned, VMILayoutAttr>> results) {
  VMILayoutRelationFact fact;
  fact.operandLayouts.resize(op->getNumOperands());
  fact.resultLayouts.resize(op->getNumResults());
  for (auto [port, layout] : operands) {
    fact.operandLayouts[port] = layout;
  }
  for (auto [port, layout] : results) {
    fact.resultLayouts[port] = layout;
  }
  return fact;
}

static bool matchesPortDomains(const VMILayoutRelationFact &fact,
                               ArrayRef<VMILayoutRelationPortDomain> domains) {
  for (const VMILayoutRelationPortDomain &domain : domains) {
    VMILayoutAttr layout = domain.kind == VMILayoutRelationPortKind::Operand
                               ? fact.operandLayouts[domain.portNumber]
                               : fact.resultLayouts[domain.portNumber];
    if (!layout || !llvm::is_contained(domain.layouts, layout)) {
      return false;
    }
  }
  return true;
}

static LogicalResult validateInterleaveOperation(Operation *op,
                                                 std::string *reason) {
  auto lhs = cast<VMIVRegType>(op->getOperand(0).getType());
  auto rhs = cast<VMIVRegType>(op->getOperand(1).getType());
  auto mask = cast<VMIMaskType>(op->getOperand(2).getType());
  auto low = cast<VMIVRegType>(op->getResult(0).getType());
  auto high = cast<VMIVRegType>(op->getResult(1).getType());
  bool laneCountsMatch = lhs.getElementCount() == rhs.getElementCount() &&
                         lhs.getElementCount() == mask.getElementCount() &&
                         lhs.getElementCount() == low.getElementCount() &&
                         lhs.getElementCount() == high.getElementCount();
  bool elementTypesMatch = lhs.getElementType() == rhs.getElementType() &&
                           lhs.getElementType() == low.getElementType() &&
                           lhs.getElementType() == high.getElementType();
  if (laneCountsMatch && elementTypesMatch) {
    return success();
  }
  if (reason) {
    *reason = "interleave relation requires equal lane counts and matching "
              "data element types";
  }
  return failure();
}

static bool isSameLayoutRelationOp(Operation *op) {
  return isa<VMIAddFOp, VMIAddIOp, VMISubFOp, VMISubIOp, VMIMulFOp, VMIMulIOp,
             VMIVaddcOp, VMIVaddcsOp, VMIAddSOp, VMIMulSOp, VMIMaxSOp,
             VMIMinSOp, VMIShlSOp, VMIShrSOp, VMIVmullOp, VMIFmaOp, VMIDivFOp,
             VMIMinFOp, VMIMinIOp, VMIMaxFOp, VMIMaxIOp, VMINegFOp, VMIAbsFOp,
             VMIAbsIOp, VMISqrtOp, VMIExpOp, VMILnOp, VMIReluOp, VMIAndIOp,
             VMIOrIOp, VMIXOrIOp, VMIShLIOp, VMIShRUIOp, VMIShRSIOp, VMINotOp,
             VMICmpFOp, VMICmpIOp, VMISelectOp, VMIMaskAndOp, VMIMaskOrOp,
             VMIMaskXOrOp, VMIMaskNotOp, VMIActivePrefixIndexOp, VMICompressOp,
             VMIExpandLoadOp>(op);
}

static bool isFreeResultLayoutRelationOp(Operation *op) {
  return isa<VMIConstantOp, VMIBroadcastOp, VMIIotaOp, VMICreateMaskOp,
             VMICreateGroupMaskOp, VMIConstantMaskOp>(op);
}

static bool isSameWidthNumericCastRelationOp(Operation *op) {
  bool hasSupportedSignature =
      isa<VMIFPToSIOp, VMIFPToUIOp, VMISIToFPOp>(op) &&
      op->getNumOperands() == 1 && op->getNumResults() == 1;
  if (!hasSupportedSignature) {
    return false;
  }
  auto sourceType = dyn_cast<VMIVRegType>(op->getOperand(0).getType());
  auto resultType = dyn_cast<VMIVRegType>(op->getResult(0).getType());
  if (!sourceType || !resultType) {
    return false;
  }
  unsigned sourceBits =
      pto::getPTOStorageElemBitWidth(sourceType.getElementType());
  unsigned resultBits =
      pto::getPTOStorageElemBitWidth(resultType.getElementType());
  return sourceBits != 0 && sourceBits == resultBits;
}

static SmallVector<VMILayoutAttr, 8>
getDomainLayouts(ArrayRef<VMILayoutRelationPortDomain> domains) {
  SmallVector<VMILayoutAttr, 8> layouts;
  for (const VMILayoutRelationPortDomain &domain : domains) {
    for (VMILayoutAttr layout : domain.layouts) {
      if (!llvm::is_contained(layouts, layout)) {
        layouts.push_back(layout);
      }
    }
  }
  return layouts;
}

static VMILayoutRelationFact makeSameLayoutFact(Operation *op,
                                                VMILayoutAttr layout) {
  VMILayoutRelationFact fact;
  fact.operandLayouts.resize(op->getNumOperands());
  fact.resultLayouts.resize(op->getNumResults());
  for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
    if (isa<VMIVRegType, VMIMaskType>(operand.getType())) {
      fact.operandLayouts[index] = layout;
    }
  }
  for (auto [index, result] : llvm::enumerate(op->getResults())) {
    if (isa<VMIVRegType, VMIMaskType>(result.getType())) {
      fact.resultLayouts[index] = layout;
    }
  }
  return fact;
}

static FailureOr<size_t> visitSameLayoutRelationFacts(
    Operation *op, ArrayRef<VMILayoutRelationPortDomain> domains,
    VMILayoutRelationFactVisitor visitor, std::string *reason) {
  SmallVector<VMILayoutAttr, 8> layouts = getDomainLayouts(domains);
  if (layouts.empty()) {
    return failFactVisit(
        "same-layout relation requires a finite layout port domain", reason);
  }
  for (VMILayoutAttr layout : layouts) {
    visitor(makeSameLayoutFact(op, layout));
  }
  return layouts.size();
}

static FailureOr<size_t> visitFreeResultRelationFacts(
    Operation *op, ArrayRef<VMILayoutRelationPortDomain> domains,
    VMILayoutRelationFactVisitor visitor, std::string *reason) {
  bool hasLayoutResult =
      op->getNumResults() == 1 &&
      isa<VMIVRegType, VMIMaskType>(op->getResult(0).getType());
  if (!hasLayoutResult) {
    return failFactVisit("free-result relation requires one layout result",
                         reason);
  }
  const VMILayoutRelationPortDomain *resultDomain = nullptr;
  for (const VMILayoutRelationPortDomain &domain : domains) {
    if (domain.kind == VMILayoutRelationPortKind::Result &&
        domain.portNumber == 0) {
      resultDomain = &domain;
      break;
    }
  }
  if (!resultDomain) {
    return failFactVisit(
        "free-result relation requires a finite result port domain", reason);
  }
  for (VMILayoutAttr layout : resultDomain->layouts) {
    visitor(makeOperationFact(op, {}, {{0, layout}}));
  }
  return resultDomain->layouts.size();
}

} // namespace

FailureOr<size_t> VMILayoutSupport::visitLayoutRelationFacts(
    Operation *op, ArrayRef<VMILayoutRelationPortDomain> portDomains,
    VMILayoutRelationFactVisitor visitor, std::string *reason) const {
  if (!op) {
    return failFactVisit("layout relation query requires an operation", reason);
  }
  if (failed(validatePortDomains(op, portDomains, reason))) {
    return failure();
  }

  size_t count = 0;
  auto emit = [&](const VMILayoutRelationFact &fact) {
    if (!matchesPortDomains(fact, portDomains)) {
      return;
    }
    visitor(fact);
    ++count;
  };

  FailureOr<size_t> visited = failure();
  if (isFreeResultLayoutRelationOp(op)) {
    visited = visitFreeResultRelationFacts(op, portDomains, emit, reason);
  } else if (isa<VMIGroupIotaOp>(op)) {
    VMILayoutAttr layout = VMILayoutAttr::getContiguous(op->getContext());
    visited = size_t{1};
    emit(makeOperationFact(op, {}, {{0, layout}}));
  } else if (isSameLayoutRelationOp(op) ||
             isSameWidthNumericCastRelationOp(op)) {
    visited = visitSameLayoutRelationFacts(op, portDomains, emit, reason);
  } else if (auto vexpdif = dyn_cast<VMIVexpdifOp>(op)) {
    auto sourceType = cast<VMIVRegType>(vexpdif.getX().getType());
    auto resultType = cast<VMIVRegType>(vexpdif.getResult().getType());
    if (sourceType.getElementType().isF32()) {
      visited = visitSameLayoutRelationFacts(op, portDomains, emit, reason);
    } else {
      visited = visitCastLayoutFacts(
          sourceType, resultType, portDomains,
          [&](const VMICastLayoutFact &fact) {
            emit(makeOperationFact(op,
                                   {{0, fact.sourceLayout},
                                    {1, fact.sourceLayout},
                                    {2, fact.sourceLayout}},
                                   {{0, fact.resultLayout}}));
          },
          reason);
    }
  } else if (auto load = dyn_cast<VMILoadOp>(op)) {
    auto type = cast<VMIVRegType>(load.getResult().getType());
    visited = visitLoadLayoutFacts(type, [&](const VMILoadLayoutFact &fact) {
      emit(makeOperationFact(op, {}, {{0, fact.resultLayout}}));
    });
  } else if (auto load = dyn_cast<VMIDeinterleaveLoadOp>(op)) {
    auto type = cast<VMIVRegType>(load.getLow().getType());
    visited = visitDeinterleaveLoadLayoutFacts(
        type, [&](const VMIDeinterleaveLoadLayoutFact &fact) {
          emit(makeOperationFact(op, {},
                                 {{0, fact.lowLayout}, {1, fact.highLayout}}));
        });
  } else if (auto store = dyn_cast<VMIStoreOp>(op)) {
    auto type = cast<VMIVRegType>(store.getValue().getType());
    visited = visitStoreLayoutFacts(
        type, [](const DenseMemoryLayoutPattern &) { return true; },
        [&](const VMIStoreLayoutFact &fact) {
          emit(makeOperationFact(op, {{0, fact.valueLayout}}, {}));
        });
  } else if (auto store = dyn_cast<VMIMaskedStoreOp>(op)) {
    auto type = cast<VMIVRegType>(store.getValue().getType());
    visited = visitMaskedStoreLayoutFacts(
        type, [](const DenseMaskedStoreLayoutPattern &) { return true; },
        [&](const VMIMaskedStoreLayoutFact &fact) {
          emit(makeOperationFact(
              op, {{0, fact.valueLayout}, {3, fact.maskLayout}}, {}));
        });
  } else if (auto load = dyn_cast<VMIMaskedLoadOp>(op)) {
    auto type = cast<VMIVRegType>(load.getResult().getType());
    visited = visitMaskedLoadLayoutFacts(
        type, [&](const VMIMaskedLoadLayoutFact &fact) {
          emit(makeOperationFact(
              op, {{2, fact.maskLayout}, {3, fact.passthruLayout}},
              {{0, fact.resultLayout}}));
        });
  } else if (auto load = dyn_cast<VMIGroupLoadOp>(op)) {
    auto type = cast<VMIVRegType>(load.getResult().getType());
    visited = visitGroupLoadLayoutFacts(
        type, load.getRowStride(), load.getNumGroupsAttr().getInt(),
        [&](const VMIGroupLoadLayoutFact &fact) {
          emit(makeOperationFact(op, {}, {{0, fact.resultLayout}}));
        },
        reason);
  } else if (auto load = dyn_cast<VMIGroupSlotLoadOp>(op)) {
    visited = visitGroupSlotLayoutFacts(
        op->getContext(), load.getNumGroupsAttr().getInt(),
        [&](const VMIGroupSlotLayoutFact &fact) {
          emit(makeOperationFact(op, {}, {{0, fact.layout}}));
        });
  } else if (auto load = dyn_cast<VMIGroupBroadcastLoadOp>(op)) {
    visited = visitGroupBroadcastLoadLayoutFacts(
        load,
        [&](const VMIGroupBroadcastLoadLayoutFact &fact) {
          emit(makeOperationFact(op, {}, {{0, fact.resultLayout}}));
        },
        reason);
  } else if (auto store = dyn_cast<VMIGroupStoreOp>(op)) {
    auto type = cast<VMIVRegType>(store.getValue().getType());
    visited = visitDirectGroupStoreLayoutFacts(
        store, type,
        [&](const VMIGroupStoreLayoutFact &fact) {
          emit(makeOperationFact(op, {{0, fact.valueLayout}}, {}));
        },
        reason);
  } else if (isa<VMIGroupReduceAddFOp, VMIGroupReduceMaxFOp,
                 VMIGroupReduceMinFOp, VMIGroupReduceAddIOp,
                 VMIGroupReduceMaxIOp, VMIGroupReduceMinIOp>(op)) {
    auto sourceType = cast<VMIVRegType>(op->getOperand(0).getType());
    int64_t numGroups = op->getAttrOfType<IntegerAttr>("num_groups").getInt();
    visited = visitGroupReduceLayoutFacts(
        sourceType, numGroups,
        [&](const VMIGroupReduceLayoutFact &fact) {
          emit(makeOperationFact(op,
                                 {{0, fact.sourceLayout}, {1, fact.maskLayout}},
                                 {{0, fact.resultLayout}}));
        },
        reason);
  } else if (auto broadcast = dyn_cast<VMIGroupBroadcastOp>(op)) {
    auto resultType = cast<VMIVRegType>(broadcast.getResult().getType());
    visited = visitGroupBroadcastLayoutFacts(
        resultType, broadcast.getNumGroupsAttr().getInt(),
        [&](const VMIGroupBroadcastLayoutFact &fact) {
          emit(makeOperationFact(op, {{0, fact.sourceLayout}},
                                 {{0, fact.resultLayout}}));
        },
        reason);
  } else if (auto select = dyn_cast<VMIVselrOp>(op)) {
    visited =
        visitVselrLayoutFacts(select, [&](const VMIVselrLayoutFact &fact) {
          emit(makeOperationFact(
              op, {{0, fact.sourceLayout}, {1, fact.indexLayout}},
              {{0, fact.resultLayout}}));
        });
  } else if (auto gather = dyn_cast<VMIGatherOp>(op)) {
    VMILayoutAttr layout = VMILayoutAttr::getContiguous(op->getContext());
    visited = size_t{1};
    emit(makeOperationFact(op, {{1, layout}, {2, layout}, {3, layout}},
                           {{0, layout}}));
  } else if (isa<VMIVintlvOp, VMIVdintlvOp>(op)) {
    if (failed(validateInterleaveOperation(op, reason))) {
      return failure();
    }
    auto type = cast<VMIVRegType>(op->getOperand(0).getType());
    ArrayRef<InterleaveLayoutPattern> patterns =
        isa<VMIVintlvOp>(op) ? ArrayRef(kVintlvLayoutPatterns)
                             : ArrayRef(kVdintlvLayoutPatterns);
    visited = visitInterleaveLayoutFacts(
        patterns, type,
        [&](const VMIInterleaveLayoutFact &fact) {
          emit(makeOperationFact(
              op,
              {{0, fact.lhsLayout}, {1, fact.rhsLayout}, {2, fact.maskLayout}},
              {{0, fact.lowLayout}, {1, fact.highLayout}}));
        },
        reason);
  } else if (isa<VMIVdhistOp, VMIVchistOp>(op)) {
    visited = visitHistogramLayoutFacts(
        op->getContext(), kVdhistLayoutPatterns,
        [&](const VMIHistogramLayoutFact &fact) {
          emit(makeOperationFact(op,
                                 {{0, fact.accLayout},
                                  {1, fact.sourceLayout},
                                  {2, fact.maskLayout}},
                                 {{0, fact.resultLayout}}));
        });
  } else if (auto bitcast = dyn_cast<VMIBitcastOp>(op)) {
    auto sourceType = cast<VMIVRegType>(bitcast.getSource().getType());
    auto resultType = cast<VMIVRegType>(bitcast.getResult().getType());
    visited = visitBitcastLayoutFacts(
        sourceType, resultType, portDomains,
        [&](const VMIBitcastLayoutFact &fact) {
          emit(makeOperationFact(op, {{0, fact.sourceLayout}},
                                 {{0, fact.resultLayout}}));
        },
        reason);
  } else if (auto ensure = dyn_cast<VMIEnsureLayoutOp>(op)) {
    auto type = cast<VMIVRegType>(ensure.getSource().getType());
    visited = visitEnsureLayoutFacts(
        type, portDomains,
        [&](const VMIEnsureLayoutFact &fact) {
          emit(makeOperationFact(op, {{0, fact.sourceLayout}},
                                 {{0, fact.resultLayout}}));
        },
        reason);
  } else if (auto ensure = dyn_cast<VMIEnsureMaskLayoutOp>(op)) {
    auto type = cast<VMIMaskType>(ensure.getSource().getType());
    visited = visitEnsureMaskLayoutFacts(
        type, portDomains,
        [&](const VMIEnsureMaskLayoutFact &fact) {
          emit(makeOperationFact(op, {{0, fact.sourceLayout}},
                                 {{0, fact.resultLayout}}));
        },
        reason);
  } else if (auto ensure = dyn_cast<VMIEnsureMaskGranularityOp>(op)) {
    auto sourceType = cast<VMIMaskType>(ensure.getSource().getType());
    auto resultType = cast<VMIMaskType>(ensure.getResult().getType());
    visited = visitMaskGranularityCastLayoutFacts(
        sourceType, resultType, portDomains,
        [&](const VMIMaskGranularityCastLayoutFact &fact) {
          emit(makeOperationFact(op, {{0, fact.sourceLayout}},
                                 {{0, fact.resultLayout}}));
        },
        reason);
  } else if (isa<VMIExtFOp, VMITruncFOp, VMIExtSIOp, VMIExtUIOp, VMITruncIOp,
                 VMIFPToSIOp, VMIFPToUIOp, VMISIToFPOp>(op)) {
    auto sourceType = cast<VMIVRegType>(op->getOperand(0).getType());
    auto resultType = cast<VMIVRegType>(op->getResult(0).getType());
    visited = visitCastLayoutFacts(
        sourceType, resultType, portDomains,
        [&](const VMICastLayoutFact &fact) {
          emit(makeOperationFact(op, {{0, fact.sourceLayout}},
                                 {{0, fact.resultLayout}}));
        },
        reason);
  } else {
    return failFactVisit(
        Twine(
            "operation has no finite VMILayoutSupport relation enumerator: ") +
            op->getName().getStringRef(),
        reason);
  }

  if (failed(visited)) {
    return failure();
  }
  return count;
}

FailureOr<size_t> VMILayoutSupport::visitLayoutConversionFacts(
    Type valueType, ArrayRef<VMILayoutAttr> sourceLayouts,
    ArrayRef<VMILayoutAttr> resultLayouts,
    VMILayoutConversionFactVisitor visitor, std::string *reason) const {
  if (!isa<VMIVRegType, VMIMaskType>(valueType)) {
    return failFactVisit("layout conversion requires a VReg or mask type",
                         reason);
  }
  bool hasEndpointDomain = !sourceLayouts.empty() || !resultLayouts.empty();
  if (!hasEndpointDomain) {
    return failFactVisit("layout conversion requires a finite endpoint domain",
                         reason);
  }
  bool hasNullEndpoint =
      llvm::is_contained(sourceLayouts, VMILayoutAttr()) ||
      llvm::is_contained(resultLayouts, VMILayoutAttr());
  if (hasNullEndpoint) {
    return failFactVisit(
        "layout conversion endpoint domains must contain concrete layouts",
        reason);
  }

  VMILayoutRelationPortDomain domains[] = {
      {VMILayoutRelationPortKind::Operand, 0, sourceLayouts},
      {VMILayoutRelationPortKind::Result, 0, resultLayouts}};
  ArrayRef<VMILayoutRelationPortDomain> activeDomains = domains;
  if (sourceLayouts.empty()) {
    activeDomains = ArrayRef(domains).drop_front();
  } else if (resultLayouts.empty()) {
    activeDomains = ArrayRef(domains).drop_back();
  }

  size_t count = 0;
  auto emit = [&](VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout) {
    bool sourceMatches = sourceLayouts.empty() ||
                         llvm::is_contained(sourceLayouts, sourceLayout);
    bool resultMatches = resultLayouts.empty() ||
                         llvm::is_contained(resultLayouts, resultLayout);
    if (!sourceMatches || !resultMatches) {
      return;
    }
    visitor({sourceLayout, resultLayout});
    ++count;
  };

  FailureOr<size_t> visited = failure();
  if (auto vregType = dyn_cast<VMIVRegType>(valueType)) {
    visited = visitEnsureLayoutFacts(
        vregType, activeDomains,
        [&](const VMIEnsureLayoutFact &fact) {
          emit(fact.sourceLayout, fact.resultLayout);
        },
        reason);
  } else {
    visited = visitEnsureMaskLayoutFacts(
        cast<VMIMaskType>(valueType), activeDomains,
        [&](const VMIEnsureMaskLayoutFact &fact) {
          emit(fact.sourceLayout, fact.resultLayout);
        },
        reason);
  }
  if (failed(visited)) {
    return failure();
  }
  return count;
}

// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef VMILAYOUTSUPPORTGROUPCAPABILITIES_INC
#define VMILAYOUTSUPPORTGROUPCAPABILITIES_INC

// Shape-only checks share the executable layout facts used by assignment and
// lowering. Keep diagnostics about the logical request, not internal layouts.
LogicalResult VMILayoutSupport::getGroupReduceShapeSupport(
    VMIGroupReduceKind kind, VMIVRegType sourceType, int64_t numGroups,
    std::string *reason) const {
  bool supported;
  if (VMILayoutAttr layout = sourceType.getLayoutAttr()) {
    supported = succeeded(getGroupReduceLayoutFactsForLayout(
        kind, sourceType, numGroups, VMIGroupReduceLayoutPort::Source, layout));
  } else {
    supported = succeeded(
        getPreferredGroupReduceLayoutFact(kind, sourceType, numGroups));
  }
  if (supported) {
    return success();
  }
  if (reason) {
    reason->clear();
    llvm::raw_string_ostream os(*reason);
    os << "type=" << sourceType.getElementType()
       << ", VL=" << sourceType.getElementCount() << ", group=" << numGroups;
    auto size = getGroupSizeFromNumGroups(sourceType, numGroups, nullptr);
    if (succeeded(size)) {
      os << ", bytes/group="
         << *size * getPTOStorageElemBitWidth(sourceType.getElementType()) /
                mlir::pto::kValue8;
    }
    auto integerType = dyn_cast<IntegerType>(sourceType.getElementType());
    if (integerType && integerType.getWidth() == mlir::pto::kValue8) {
      os << ": 8-bit integer reductions are not supported; explicitly convert "
            "the source to a supported 16-bit or 32-bit type";
    } else {
      os << ": no executable A5 grouped-reduction layout; requires supported "
            "16/32-bit elements, predicate width, and source/mask/result layouts";
    }
  }
  return failure();
}

FailureOr<VMIGroupBroadcastLayoutFact>
VMILayoutSupport::getPreferredGroupBroadcastLayoutFact(
    VMIVRegType resultType, int64_t numGroups, std::string *reason) const {
  FailureOr<GroupLayoutKey> key = buildGroupLayoutKey(
      resultType, numGroups, "no registered grouped-broadcast shape", reason);
  if (failed(key)) {
    return failure();
  }
  for (const GroupBroadcastLayoutPattern &pattern : kGroupBroadcastLayoutPatterns) {
    if (matchesGroupBroadcastLayoutPattern(
            pattern, *key, resultType.getElementType())) {
      return materializeGroupBroadcastLayoutFact(
          resultType.getContext(), pattern, *key, numGroups);
    }
  }
  return failWithReason<FailureOr<VMIGroupBroadcastLayoutFact>>(
      reason, "no registered grouped-broadcast shape");
}

VMILayoutAttr VMILayoutSupport::getPreferredGroupBroadcastResultLayout(
    VMIVRegType resultType, int64_t numGroups) const {
  FailureOr<GroupLayoutKey> key =
      buildGroupLayoutKey(resultType, numGroups, "", nullptr);
  if (failed(key)) {
    return {};
  }
  // Preserve the memory table's fast packed-store layouts when the broadcast
  // supports them. In particular, f16 L=64 keeps its native half-block path.
  for (const DenseMemoryLayoutPattern &memory : kDenseStoreLayoutPatterns) {
    if (!memory.preferred ||
        !matchesElementBitsPattern(memory.elementBits, resultType.getElementType()) ||
        !matchesElementCountPattern(memory.elementCounts, key->elementCount)) {
      continue;
    }
    VMILayoutAttr preferred =
        materializeLayoutPattern(resultType.getContext(), memory.layout);
    for (const GroupBroadcastLayoutPattern &pattern : kGroupBroadcastLayoutPatterns) {
      if (matchesGroupBroadcastLayoutPattern(
            pattern, *key, resultType.getElementType()) &&
          materializeLayoutPattern(resultType.getContext(), pattern.resultLayout) == preferred) {
        return preferred;
      }
    }
  }
  if (matchesCompactGroupShape(key->elementCount, key->vcgBlockElems, key->groupSize)) {
    return VMILayoutAttr::getContiguous(resultType.getContext());
  }
  // Preserve the established half-carrier broadcast outside the dense fallback.
  if (key->elementCount < key->lanesPerPart && key->groupSize == key->vcgBlockElems) {
    return VMILayoutAttr::getContiguous(resultType.getContext(), mlir::pto::kValue2);
  }
  return {};
}

LogicalResult VMILayoutSupport::getGroupBroadcastShapeSupport(
    VMIVRegType resultType, int64_t numGroups, std::string *reason) const {
  return success(succeeded(
      getPreferredGroupBroadcastLayoutFact(resultType, numGroups, reason)));
}

LogicalResult VMILayoutSupport::getGroupOperationShapeSupport(
    Operation *op, std::string *reason) const {
  auto reduce = [this, reason](auto groupReduce) {
    return getGroupReduceShapeSupport(
        getVMIGroupReduceKind(groupReduce),
        cast<VMIVRegType>(groupReduce.getSource().getType()),
        groupReduce.getNumGroupsAttr().getInt(), reason);
  };
  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
      .Case<VMIGroupReduceAddFOp, VMIGroupReduceAddIOp, VMIGroupReduceMaxFOp,
            VMIGroupReduceMaxIOp, VMIGroupReduceMinFOp, VMIGroupReduceMinIOp>(
          reduce)
      .Case<VMIGroupBroadcastOp>([&](VMIGroupBroadcastOp broadcast) {
        return getGroupBroadcastShapeSupport(
            cast<VMIVRegType>(broadcast.getResult().getType()),
            broadcast.getNumGroupsAttr().getInt(), reason);
      })
      .Default([](Operation *) { return success(); });
}

#endif // VMILAYOUTSUPPORTGROUPCAPABILITIES_INC

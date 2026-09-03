// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.


//===- VMILayoutSupportMaterialization.inc - VMI layout support internals -*- C++ -*-===//
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
  for (const GroupBlockClassPattern &row : kGroupBlockClassPatterns) {
    if (isSameGroupBlockPattern(pattern, row.block)) {
      return row.blockClass;
    }
  }
  llvm_unreachable("unsupported group block pattern");
}

static bool matchesGroupBroadcastLoadMemoryPattern(
    GroupMemoryPattern pattern, std::optional<int64_t> stride,
    int64_t elementBits) {
  switch (pattern.kind) {
  case GroupMemoryPatternKind::Any:
    return true;
  case GroupMemoryPatternKind::Contiguous:
    if (!stride) {
      return false;
    }
    return *stride == 1;
  case GroupMemoryPatternKind::BlockAligned: {
    if (!stride) {
      return false;
    }
    if (elementBits <= 0 || kLayoutBlockBitWidth % elementBits != 0) {
      return false;
    }
    int64_t alignedStrideElems = kLayoutBlockBitWidth / elementBits;
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
    if (!rowStride || elementBits <= 0 ||
        kLayoutBlockBitWidth % elementBits != 0) {
      return false;
    }
    int64_t alignedStrideElems = kLayoutBlockBitWidth / elementBits;
    return *rowStride > 0 && *rowStride % alignedStrideElems == 0;
  }
  }
  llvm_unreachable("unknown group memory pattern kind");
}

static bool isSupportedGroupSlotMemoryLayout(VMILayoutAttr layout,
                                             int64_t numGroups) {
  bool hasSupportedShape =
      layout && layout.isGroupSlots() && layout.getNumGroups() == numGroups &&
      layout.getSlots() > 0;
  if (!hasSupportedShape) {
    return false;
  }
  for (const GroupSlotMemoryLayoutPattern &pattern :
       kGroupSlotMemoryLayoutPatterns) {
    if (matchesLayoutPattern(layout.getContext(), pattern.layout, layout,
                             numGroups)) {
      return true;
    }
  }
  return false;
}

static FailureOr<VMIGroupBlockClass> getGroupBlockClass(int64_t groupSize,
                                                        int64_t vcgBlockElems) {
  if (vcgBlockElems <= 0) {
    return failure();
  }

  for (const GroupBlockClassPattern &row : kGroupBlockClassPatterns) {
    GroupBlockPattern block = row.block;
    if (block.kind == GroupBlockPatternKind::FullPartMultiple) {
      int64_t fullPartElems = 8 * vcgBlockElems;
      if (groupSize >= fullPartElems && groupSize % fullPartElems == 0) {
        return row.blockClass;
      }
      continue;
    }

    int64_t numerator = vcgBlockElems * block.numerator;
    if (block.denominator <= 0 || numerator % block.denominator != 0) {
      continue;
    }
    if (groupSize == numerator / block.denominator) {
      return row.blockClass;
    }
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
  return matchesPhysicalChunkCountPattern(pattern, key.physicalChunkCount);
}

static bool matchesInterleaveLayoutPattern(
    const InterleaveLayoutPattern &pattern, VMIVRegType valueType,
    InterleaveLayoutKey key) {
  if (!matchesElementBitsPattern(pattern.elementBits,
                                 valueType.getElementType()) ||
      !matchesPhysicalChunkCountPattern(pattern.chunks, key)) {
    return false;
  }
  if (pattern.maxPhysicalParts <= 0) {
    return true;
  }

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
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  FailureOr<int64_t> lanesPerPart =
      getDataLanesPerPart(valueType.getElementType());
  if (failed(lanesPerPart)) {
    return fail("interleave layout requires element type with known physical "
                "lanes per part");
  }
  int64_t elementCount = valueType.getElementCount();
  if (elementCount <= 0) {
    return fail("interleave layout requires positive logical lane count");
  }
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
    if (pattern.numerator <= 0) {
      return false;
    }
    int64_t fullPartElems = key.lanesPerPart * pattern.numerator;
    return key.groupSize >= fullPartElems &&
           key.groupSize % fullPartElems == 0;
  }

  int64_t numerator = key.vcgBlockElems * pattern.numerator;
  if (pattern.denominator <= 0 || numerator % pattern.denominator != 0) {
    return false;
  }
  return key.groupSize == numerator / pattern.denominator;
}

static bool matchesGroupStoreLayoutPattern(
    const GroupStoreLayoutPattern &pattern, VMIVRegType valueType,
    GroupLayoutKey key, std::optional<int64_t> rowStride) {
  unsigned elementBits =
      pto::getPTOStorageElemBitWidth(valueType.getElementType());
  return elementBits != 0 &&
         matchesElementBitsPattern(pattern.elementBits,
                                   valueType.getElementType()) &&
         matchesGroupBlockPattern(pattern.block, key) &&
         matchesGroupLoadMemoryPattern(pattern.memory, rowStride,
                                       key.groupSize, elementBits);
}

static FailureOr<GroupLayoutKey>
buildGroupLayoutKey(VMIVRegType type, int64_t numGroups,
                    const Twine &unsupportedGroupSizeReason,
                    std::string *reason) {
  auto fail = [&](const Twine &message) -> FailureOr<GroupLayoutKey> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  FailureOr<int64_t> groupSize =
      getGroupSizeFromNumGroups(type, numGroups, reason);
  if (failed(groupSize)) {
    return failure();
  }
  FailureOr<int64_t> lanesPerPart = getDataLanesPerPart(type.getElementType());
  bool hasWholeVcgBlocks =
      succeeded(lanesPerPart) &&
      *lanesPerPart % mlir::pto::kValue8 == 0;
  if (!hasWholeVcgBlocks) {
    return fail("requires element type with known 32B VCG block width");
  }

  int64_t vcgBlockElems = *lanesPerPart / mlir::pto::kValue8;
  FailureOr<VMIGroupBlockClass> blockClass =
      getGroupBlockClass(*groupSize, vcgBlockElems);
  if (failed(blockClass)) {
    return fail(unsupportedGroupSizeReason);
  }

  return GroupLayoutKey{*groupSize, *lanesPerPart, vcgBlockElems, *blockClass};
}

static VMIGroupReduceLayoutFact
materializeGroupReduceLayoutFact(MLIRContext *ctx,
                                 const GroupReduceLayoutPattern &pattern,
                                 GroupLayoutKey key, int64_t numGroups) {
  VMIGroupReduceLayoutFact fact;
  fact.blockClass = getGroupBlockClassFromPattern(pattern.block);
  fact.sourceLayout = materializeLayoutPattern(ctx, pattern.sourceLayout);
  fact.maskLayout = fact.sourceLayout;
  fact.resultLayout =
      materializeLayoutPattern(ctx, pattern.resultLayout, numGroups);
  fact.groupSize = key.groupSize;
  fact.lanesPerPart = key.lanesPerPart;
  fact.vcgBlockElems = key.vcgBlockElems;
  return fact;
}

static VMIGroupBroadcastLayoutFact materializeGroupBroadcastLayoutFact(
    MLIRContext *ctx, const GroupBroadcastLayoutPattern &pattern,
    GroupLayoutKey key, int64_t numGroups) {
  VMIGroupBroadcastLayoutFact fact;
  fact.blockClass = getGroupBlockClassFromPattern(pattern.block);
  fact.sourceLayout =
      materializeLayoutPattern(ctx, pattern.sourceLayout, numGroups);
  fact.resultLayout =
      materializeLayoutPattern(ctx, pattern.resultLayout, numGroups);
  fact.groupSize = key.groupSize;
  fact.lanesPerPart = key.lanesPerPart;
  fact.vcgBlockElems = key.vcgBlockElems;
  return fact;
}

static VMIGroupStoreLayoutFact materializeGroupStoreLayoutFact(
    MLIRContext *ctx, const GroupStoreLayoutPattern &pattern,
    GroupLayoutKey key) {
  VMIGroupStoreLayoutFact fact;
  fact.valueLayout = materializeLayoutPattern(ctx, pattern.valueLayout);
  fact.blockClass = getGroupBlockClassFromPattern(pattern.block);
  fact.groupSize = key.groupSize;
  fact.lanesPerPart = key.lanesPerPart;
  fact.vcgBlockElems = key.vcgBlockElems;
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

static VMIVselrLayoutFact
materializeVselrLayoutFact(MLIRContext *ctx,
                           const VselrLayoutPattern &pattern) {
  return VMIVselrLayoutFact{
      materializeLayoutPattern(ctx, pattern.sourceLayout),
      materializeLayoutPattern(ctx, pattern.indexLayout),
      materializeLayoutPattern(ctx, pattern.resultLayout)};
}

static FailureOr<int64_t> getPhysicalArityForLayout(VMIVRegType type,
                                                     VMILayoutAttr layout) {
  auto assignedType = VMIVRegType::get(
      type.getContext(), type.getElementCount(), type.getElementType(), layout);
  return getVMIPhysicalArity(assignedType);
}

static bool matchesVselrLayoutPattern(const VselrLayoutPattern &pattern,
                                      VMIVselrOp op,
                                      const VMIVselrLayoutFact &fact) {
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto indexType = cast<VMIVRegType>(op.getIndex().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  unsigned sourceBits =
      pto::getPTOStorageElemBitWidth(sourceType.getElementType());
  unsigned indexBits =
      pto::getPTOStorageElemBitWidth(indexType.getElementType());
  bool typesMatch =
      sourceType.getElementType() == resultType.getElementType() &&
      indexType.getElementCount() == resultType.getElementCount() &&
      sourceBits == indexBits;
  bool patternMatches =
      matchesElementBitsPattern(pattern.elementBits, sourceBits) &&
      matchesElementCountPattern(pattern.sourceElements,
                                 sourceType.getElementCount()) &&
      matchesElementCountPattern(pattern.indexElements,
                                 indexType.getElementCount()) &&
      matchesElementCountPattern(pattern.resultElements,
                                 resultType.getElementCount());
  if (!typesMatch || !patternMatches) {
    return false;
  }

  FailureOr<int64_t> sourceArity =
      getPhysicalArityForLayout(sourceType, fact.sourceLayout);
  FailureOr<int64_t> indexArity =
      getPhysicalArityForLayout(indexType, fact.indexLayout);
  FailureOr<int64_t> resultArity =
      getPhysicalArityForLayout(resultType, fact.resultLayout);
  return succeeded(sourceArity) && succeeded(indexArity) &&
         succeeded(resultArity) &&
         matchesPhysicalChunkCountPattern(pattern.sourceChunks,
                                          *sourceArity) &&
         matchesPhysicalChunkCountPattern(pattern.indexChunks, *indexArity) &&
         matchesPhysicalChunkCountPattern(pattern.resultChunks, *resultArity);
}

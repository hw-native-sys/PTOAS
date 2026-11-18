// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMILayoutSupportBF16.cpp - BF16/F32 cast layout support ---===//
//===----------------------------------------------------------------------===//

#include "PTO/Support/CodeConstants.h"
#include "PTO/Transforms/VMILayoutSupport.h"

#include "PTO/IR/VMIUtils.h"

#include <utility>

namespace mlir::pto {

bool isDenseBF16NarrowingCast(VMIVRegType sourceType,
                              VMIVRegType resultType)
{
    if (!sourceType || !resultType ||
        !sourceType.getElementType().isF32() ||
        !resultType.getElementType().isBF16() ||
        sourceType.getElementCount() != resultType.getElementCount()) {
        return false;
    }
    int64_t elementCount = sourceType.getElementCount();
    return elementCount == mlir::pto::kValue2 || elementCount == mlir::pto::kValue4 ||
           elementCount == mlir::pto::kValue8 || elementCount == mlir::pto::kValue64 ||
           elementCount == mlir::pto::kValue128 || elementCount == mlir::pto::kValue256;
}

bool isDenseBF16NarrowingSameLayoutPair(VMIVRegType sourceType,
                                        VMIVRegType resultType,
                                        VMILayoutAttr sourceLayout,
                                        VMILayoutAttr resultLayout)
{
    if (!isDenseBF16NarrowingCast(sourceType, resultType) || !sourceLayout ||
        !resultLayout || sourceLayout != resultLayout ||
        (!sourceLayout.isContiguous() && !sourceLayout.isDeinterleaved() &&
         !sourceLayout.isGroupSlots())) {
        return false;
    }
    // A group-slot layout is dense in its own way. The compact slots=1 form
    // converts one group row-locally, so its dense result is honest in every
    // state. A packet form (slots=8) is different: its narrowing is realized by
    // the dedicated group-slot lowering, which only produces the dense result in
    // the NaN-relaxed fastmath state; every other state writes the converted
    // lanes at lane_stride 2 (same as the dense rows). The relation is therefore
    // state-qualified for packet group slots too - leaving it ungated let the
    // solver pick a dense result the strict lowering cannot produce (#1656).
    // The lane_stride=2 packet is the honest sparse form, not this relation.
    return !sourceLayout.isGroupSlots() ||
           (sourceLayout.getLaneStride() == 1 &&
            sourceLayout.getSlots() == mlir::pto::kValue8);
}

bool requiresBF16NaNRelaxedLowering(VMIVRegType sourceType,
                                    VMIVRegType resultType,
                                    VMILayoutAttr sourceLayout,
                                    VMILayoutAttr resultLayout)
{
    if (!isDenseBF16NarrowingSameLayoutPair(sourceType, resultType, sourceLayout,
                                            resultLayout)) {
        return false;
    }
    MLIRContext *context = sourceType.getContext();
    auto assignedSource = VMIVRegType::get(
        context, sourceType.getElementCount(), sourceType.getElementType(),
        sourceLayout);
    auto assignedResult = VMIVRegType::get(
        context, resultType.getElementCount(), resultType.getElementType(),
        resultLayout);
    FailureOr<int64_t> sourceArity = getVMIPhysicalArity(assignedSource);
    FailureOr<int64_t> resultArity = getVMIPhysicalArity(assignedResult);
    return succeeded(sourceArity) && succeeded(resultArity) &&
           *sourceArity == *resultArity;
}

bool requiresBF16NaNRelaxedLowering(VMIVRegType sourceType,
                                    VMIVRegType resultType)
{
    if (!isDenseBF16NarrowingCast(sourceType, resultType)) {
        return false;
    }
    VMILayoutAttr contiguous = VMILayoutAttr::getContiguous(sourceType.getContext());
    return requiresBF16NaNRelaxedLowering(sourceType, resultType, contiguous,
                                          contiguous);
}

bool hasBF16NaNRelaxedFastmathState(Operation *op, bool enableVmiFastmath)
{
    if (!op || !enableVmiFastmath) {
        return false;
    }
    StringAttr rounding = op->getAttrOfType<StringAttr>("rounding");
    StringAttr saturate = op->getAttrOfType<StringAttr>("saturate");
    return rounding && rounding.getValue() == "Z" && saturate &&
           saturate.getValue() == "NOSAT";
}

std::string describeBF16NarrowingState(Operation *op, bool enableVmiFastmath)
{
    std::string description;
    llvm::raw_string_ostream stream(description);
    stream << "vmi-fastmath enabled with rounding=\"Z\" and saturate=\"NOSAT\"; "
              "got vmi-fastmath "
           << (enableVmiFastmath ? "enabled" : "disabled");
    StringAttr rounding =
        op ? op->getAttrOfType<StringAttr>("rounding") : StringAttr();
    StringAttr saturate =
        op ? op->getAttrOfType<StringAttr>("saturate") : StringAttr();
    stream << ", rounding=\""
           << (rounding ? rounding.getValue() : StringRef("<none>"))
           << "\", saturate=\""
           << (saturate ? saturate.getValue() : StringRef("<none>")) << "\"";
    return description;
}

// Dense and group-slot shapes the pairwise BF16/F32 cast supports.
static bool isSupportedBF16CastLogicalShape(VMIVRegType resultType,
                                            VMILayoutAttr sourceLayout,
                                            int64_t elementCount) {
  bool supportedDenseShape = elementCount == 1 || elementCount == 2 ||
                             elementCount == 4 || elementCount == 8 ||
                             elementCount == 64 || elementCount == 128 ||
                             elementCount == 256;
  bool supportedGroupSlotShape =
      sourceLayout.isGroupSlots() &&
      (elementCount == 2 || elementCount == 4 || elementCount == 8);
  return elementCount == resultType.getElementCount() &&
         (supportedDenseShape || supportedGroupSlotShape);
}

// Physical arities of the assigned source/result types, or failure.
static FailureOr<std::pair<int64_t, int64_t>>
getAssignedCastArities(VMIVRegType sourceType, VMIVRegType resultType,
                       VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
                       int64_t elementCount) {
  auto assignedSource = VMIVRegType::get(
      sourceType.getContext(), elementCount, sourceType.getElementType(),
      sourceLayout);
  auto assignedResult = VMIVRegType::get(
      resultType.getContext(), elementCount, resultType.getElementType(),
      resultLayout);
  FailureOr<int64_t> sourceArity = getVMIPhysicalArity(assignedSource);
  FailureOr<int64_t> resultArity = getVMIPhysicalArity(assignedResult);
  if (failed(sourceArity) || failed(resultArity) || *sourceArity <= 0 ||
      *resultArity <= 0) {
    return failure();
  }
  return std::make_pair(*sourceArity, *resultArity);
}

// Shared candidate/lowering contract for pairwise BF16/F32 casts.
// Regular layouts preserve the partition and sparse lane index; block layouts
// change their block size with element width and are deliberately excluded.
bool isBF16SameLayoutCastPair(VMIVRegType sourceType,
                              VMIVRegType resultType,
                              VMILayoutAttr sourceLayout,
                              VMILayoutAttr resultLayout) {
  if (!sourceType || !resultType) {
    return false;
  }
  bool regularLayout =
      sourceLayout && resultLayout && sourceLayout == resultLayout &&
      (sourceLayout.isContiguous() || sourceLayout.isDeinterleaved());
  bool supportedGroupSlots =
      sourceLayout && resultLayout && sourceLayout == resultLayout &&
      sourceLayout.isGroupSlots() &&
      (sourceLayout.getSlots() == 1 || sourceLayout.getSlots() == 8) &&
      (sourceType.getElementCount() == 2 ||
       sourceType.getElementCount() == 4 ||
       sourceType.getElementCount() == 8) &&
      sourceLayout.getNumGroups() == sourceType.getElementCount();
  bool supportedLaneStride =
      sourceLayout && (sourceLayout.getLaneStride() == 1 ||
                       sourceLayout.getLaneStride() == kValue2 ||
                       sourceLayout.getLaneStride() == kValue4);
  bool unsupportedLayout = !regularLayout && !supportedGroupSlots;
  if (unsupportedLayout || !supportedLaneStride) {
    return false;
  }
  bool isBF16ToF32 = sourceType.getElementType().isBF16() &&
                    resultType.getElementType().isF32();
  bool isF32ToBF16 = sourceType.getElementType().isF32() &&
                    resultType.getElementType().isBF16();
  if (!isBF16ToF32 && !isF32ToBF16) {
    return false;
  }
  int64_t elementCount = sourceType.getElementCount();
  if (!isSupportedBF16CastLogicalShape(resultType, sourceLayout, elementCount)) {
    return false;
  }

  FailureOr<std::pair<int64_t, int64_t>> arities = getAssignedCastArities(
      sourceType, resultType, sourceLayout, resultLayout, elementCount);
  if (failed(arities)) {
    return false;
  }
  int64_t wideArity = isF32ToBF16 ? arities->first : arities->second;
  int64_t narrowArity = isF32ToBF16 ? arities->second : arities->first;
  return wideArity == narrowArity ||
         (wideArity % kValue2 == 0 && wideArity / kValue2 == narrowArity);
}


// Checker-side acceptance for already-assigned F32 deinterleaved=4 -> BF16
// contiguous narrowing. This pair is intentionally not a legal-table
// candidate: the current solver cannot materialize the parallel d(4) source
// from a contiguous seed without first choosing an unsupported c -> c cast
// pair. The lowering handles the pair when a producer has already selected
// the d(4) source layout.
bool isDeinterleaved4ToContiguousBF16CastPair(VMIVRegType sourceType,
                                              VMIVRegType resultType,
                                              VMILayoutAttr sourceLayout,
                                              VMILayoutAttr resultLayout) {
  if (!sourceType || !resultType || !sourceLayout || !resultLayout) {
    return false;
  }
  int64_t elementCount = sourceType.getElementCount();
  return sourceType.getElementType().isF32() && resultType.getElementType().isBF16() &&
         resultType.getElementCount() == elementCount &&
         (elementCount == mlir::pto::kValue128 || elementCount == mlir::pto::kValue256) &&
         sourceLayout.isDeinterleaved() && sourceLayout.getFactor() == mlir::pto::kValue4 &&
         sourceLayout.getLaneStride() == 1 && resultLayout.isContiguous() && resultLayout.getLaneStride() == 1;
}

} // namespace mlir::pto

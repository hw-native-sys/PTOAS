// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMILayoutSupportBF16.cpp - BF16/F32 cast layout support ---===//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/VMILayoutSupport.h"

#include "PTO/IR/VMIUtils.h"

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
    return elementCount == 64 || elementCount == 128 || elementCount == 256;
}

bool isDenseBF16NarrowingSameLayoutPair(VMIVRegType sourceType,
                                        VMIVRegType resultType,
                                        VMILayoutAttr sourceLayout,
                                        VMILayoutAttr resultLayout)
{
    if (!isDenseBF16NarrowingCast(sourceType, resultType) || !sourceLayout ||
        !resultLayout || sourceLayout != resultLayout ||
        !sourceLayout.isContiguous() ||
        sourceLayout.getLaneStride() != 1) {
        return false;
    }
    // A group-slot layout is dense in its own way, but its narrowing is realized
    // by the dedicated group-slot lowering and is not the state-qualified dense
    // relation.
    return !sourceLayout.isGroupSlots();
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

// Validation-only acceptance for an already assigned BF16/F32 cast pair.
// Regular contiguous, deinterleaved, and supported group-slot lane maps
// preserve their padding and part order through the pairwise physical
// intlv/dintlv lowering. Layout candidates remain defined by the cast tables.
bool isBF16SameLayoutCastPair(VMIVRegType sourceType,
                              VMIVRegType resultType,
                              VMILayoutAttr sourceLayout,
                              VMILayoutAttr resultLayout) {
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
  bool supportedDenseShape =
      elementCount == 64 || elementCount == 128 || elementCount == 256;
  bool supportedGroupSlotShape =
      sourceLayout.isGroupSlots() &&
      (elementCount == 2 || elementCount == 4 || elementCount == 8);
  bool supportedLogicalShape = elementCount == resultType.getElementCount() &&
                               (supportedDenseShape || supportedGroupSlotShape);
  if (!supportedLogicalShape) {
    return false;
  }

  FailureOr<int64_t> sourceArity = getVMIPhysicalArity(sourceType);
  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  bool validArity = succeeded(sourceArity) && succeeded(resultArity) &&
                    *sourceArity > 0 && *resultArity > 0;
  if (!validArity) {
    return false;
  }
  int64_t wideArity = isF32ToBF16 ? *sourceArity : *resultArity;
  int64_t narrowArity = isF32ToBF16 ? *resultArity : *sourceArity;
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
  return sourceType.getElementType().isF32() &&
         resultType.getElementType().isBF16() &&
         resultType.getElementCount() == elementCount &&
         (elementCount == 128 || elementCount == 256) &&
         sourceLayout.isDeinterleaved() && sourceLayout.getFactor() == 4 &&
         sourceLayout.getLaneStride() == 1 && resultLayout.isContiguous() &&
         resultLayout.getLaneStride() == 1;
}

} // namespace mlir::pto

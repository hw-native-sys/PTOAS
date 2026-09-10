// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Internal shared helper definitions for VMI.
// This header is internal to lib/PTO/IR and not installed.
// Each including translation unit gets its own internal copy of these helpers.

#ifndef PTO_IR_VMI_INTERNAL_H
#define PTO_IR_VMI_INTERNAL_H

// Batch6: 由 VMI_ops.cpp 上移的类型
enum class CvtDirection { FpWiden, FpNarrow, FpToSi, FpToUi, SiToFp, IntWiden, IntNarrow };

#include <optional>
#include <set>
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/VMIUtils.h"
#include "PTO/Support/CodeConstants.h"


using namespace mlir;
using namespace mlir::pto;

enum class VMIIntSignSemantics { Unsigned, Signed, Any };

// Batch6: 以下 helper 定义迁至 VMI_helpers.cpp(外部链接)
std::string formatVMIVRegType(int64_t elementCount, Type elementType, Attribute layout);
std::string formatVMIMaskType(int64_t elementCount, StringRef granularity, Attribute layout);
bool matchesVMIIntSemantics(IntegerType intType, VMIIntSignSemantics semantics);
bool isCompatibleVMIScalarForSemanticType(Type semanticType, Type scalarType);
LogicalResult parseOptionalVMILayout(AsmParser &parser, Attribute &layout);
FailureOr<VMILayoutAttr> getAssignedVMILayout(Type type);
int64_t getMaskGranularityBitWidth(StringRef granularity);
StringRef getMaskGranularityForBitWidth(int64_t bits);
FailureOr<StringRef> getVMIMaskPhysicalGranularity(VMIMaskType type);
FailureOr<int64_t> getPhysicalLanesPerPart(Type type);
FailureOr<int64_t> getDenseLaneStride(Type type);
LogicalResult verifyAllSameVRegShapeAndLayout(Operation *op, ArrayRef<VMIVRegType> types, bool requireSameElement);
LogicalResult verifyAllSameVRegShapeAndLayoutPresence( Operation *op, ArrayRef<VMIVRegType> types, bool requireSameElement);
LogicalResult verifyFloatUnaryVRegOp(Operation *op, VMIVRegType source, VMIVRegType result);
LogicalResult verifyFloatTernaryVRegOp(Operation *op, VMIVRegType lhs, VMIVRegType rhs, VMIVRegType acc, VMIVRegType result);
LogicalResult verifyAllSameMaskShapeLayoutAndGranularity(Operation *op, ArrayRef<VMIMaskType> types);
LogicalResult verifyMaskMatchesData(Operation *op, VMIMaskType maskType, VMIVRegType dataType);
bool isUBBackedMemoryType(Type type);
LogicalResult verifyMemoryElementMatches(Operation *op, Type memoryType, VMIVRegType dataType, StringRef role);
bool isVMI8To16GatherPair(Type sourceElemType, Type resultElemType);
LogicalResult verifyGatherMemoryElementMatches( Operation *op, Type memoryType, VMIVRegType dataType, StringRef role);
bool isSameWidth16BitGatherPair(Type sourceElemType, Type resultElemType);
LogicalResult verifyContiguousIfLayoutAssigned(Operation *op, VMIVRegType type, StringRef role);
bool isPackedByteGroupStore(Type memoryType, VMIVRegType dataType);
LogicalResult verifyNumGroups(Operation *op, VMIVRegType type, int64_t numGroups);
LogicalResult verifyPhysicalVRegParts(Operation *op, VMIVRegType vregType, TypeRange physicalTypes);
LogicalResult verifyPhysicalMaskParts(Operation *op, VMIMaskType maskType, TypeRange physicalTypes);
LogicalResult verifyPhysicalParts(Operation *op, Type vmiType, TypeRange physicalTypes);
std::optional<int64_t> mapDenseLogicalLaneToPartIndex(int64_t elementCount, int64_t factor, int64_t blockElems, int64_t logicalLane, int64_t &part);
std::optional<int64_t> mapDensePartIndexToLogicalLane(int64_t elementCount, int64_t factor, int64_t blockElems, int64_t part, int64_t indexInPart);
int64_t getDenseLogicalLanesInPart(int64_t elementCount, int64_t factor, int64_t blockElems, int64_t part);
LogicalResult verifyReductionGroupAndPmode( Operation *op, VMIVRegType sourceType, VMIVRegType resultType, IntegerAttr groupAttr, std::optional<StringRef> pmode);

namespace {


[[maybe_unused]] static bool isSupportedVMIElementType(Type type) {
  return isa<IntegerType, FloatType, IndexType>(type) ||
         pto::isPTOLowPrecisionType(type);
}

[[maybe_unused]] static bool isVMIFloatLikeType(Type type) {
  return isa<FloatType>(type) || pto::isPTOLowPrecisionType(type);
}

[[maybe_unused]] static bool involvesBF16x2(Type sourceType, Type resultType) {
  return pto::isPTOBF16x2Type(sourceType) ||
         pto::isPTOBF16x2Type(resultType);
}

[[maybe_unused]] static bool isVMIPackedFloatCarrierType(Type type) {
  return pto::isPTOHiFloat8x2Type(type) ||
         pto::isPTOFloat4PackedType(type) ||
         pto::isPTOBF16x2Type(type);
}

[[maybe_unused]] static bool involvesVMIPackedFloatCarrier(Type sourceType, Type resultType) {
  return isVMIPackedFloatCarrierType(sourceType) ||
         isVMIPackedFloatCarrierType(resultType);
}

[[maybe_unused]] static LogicalResult verifyBF16x2ComputeElementType(Operation *op, Type type) {
  if (pto::isPTOBF16x2Type(type)) {
    return op->emitOpError(
        "does not support bf16x2 VMI element type; bf16x2 is conversion-only");
}
  return success();
}

[[maybe_unused]] static bool isVMIIntegerLikeType(Type type) {
  return isa<IntegerType, IndexType>(type);
}

[[maybe_unused]] static bool isVMIF16OrF32Type(Type type) {
  return type.isF16() || type.isF32();
}

[[maybe_unused]] static bool isVMIF16BF16OrF32Type(Type type) {
  return type.isF16() || type.isBF16() || type.isF32();
}

[[maybe_unused]] static bool isVMIPredicateMaskableElementType(Type type) {
  unsigned elementBits = pto::getPTOStorageElemBitWidth(type);
  return elementBits == mlir::pto::kValue8 || elementBits == mlir::pto::kValue16 || elementBits == mlir::pto::kValue32;
}

[[maybe_unused]] static bool isVMIAnyI8I16I32Type(Type type) {
  auto integerType = dyn_cast<IntegerType>(type);
  if (!integerType) {
    return false;
  }
  return integerType.getWidth() == mlir::pto::kValue8 || integerType.getWidth() == mlir::pto::kValue16 ||
         integerType.getWidth() == mlir::pto::kValue32;
}

[[maybe_unused]] static bool isVMII8I16I32OrF16BF16F32Type(Type type) {
  return isVMIAnyI8I16I32Type(type) || isVMIF16BF16OrF32Type(type);
}

[[maybe_unused]] static bool isVMII16I32OrF16BF16F32Type(Type type) {
  auto intType = dyn_cast<IntegerType>(type);
  bool supportedInteger =
      intType && (intType.getWidth() == mlir::pto::kValue16 || intType.getWidth() == mlir::pto::kValue32);
  return supportedInteger || isVMIF16BF16OrF32Type(type);
}

[[maybe_unused]] static bool isVMII8I16I32OrF16F32Type(Type type) {
  return isVMIAnyI8I16I32Type(type) || isVMIF16OrF32Type(type);
}

[[maybe_unused]] static bool isVMISignedI8I16I32Type(Type type) {
  auto integerType = dyn_cast<IntegerType>(type);
  if (!integerType || !integerType.isSigned()) {
    return false;
  }
  return integerType.getWidth() == mlir::pto::kValue8 || integerType.getWidth() == mlir::pto::kValue16 ||
         integerType.getWidth() == mlir::pto::kValue32;
}

[[maybe_unused]] static bool isVMISignedIntegerType(Type type) {
  auto integerType = dyn_cast<IntegerType>(type);
  return integerType && integerType.isSigned();
}

[[maybe_unused]] static bool isVMIUnsignedOrSignlessIntegerType(Type type) {
  auto integerType = dyn_cast<IntegerType>(type);
  return integerType && (integerType.isUnsigned() || integerType.isSignless());
}

// ---------------------------------------------------------------------------
// VMI integer element type sign-semantics helper
//
// CONVENTION: VMI op verifiers that need "unsigned semantics" or "signed
// semantics" on an integer element type MUST route the sign check through
// matchesVMIIntSemantics(...) instead of calling IntegerType::isUnsigned()
// / isSigned() directly.
//
// Signless integers are treated as equivalent to UNSIGNED only. They are
// NOT accepted for signed semantics: signed hardware ops require an
// explicitly signed integer type, to avoid silent sign-extension bugs when
// a producer happens to emit a signless value.
//
// Width / kind / IntegerType-cast checks stay inline at each callsite;
// only the sign-semantics decision is centralized here.
// ---------------------------------------------------------------------------



[[maybe_unused]] static bool isVMIIotaElementType(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    return intType.getWidth() == mlir::pto::kValue8 || intType.getWidth() == mlir::pto::kValue16 ||
           intType.getWidth() == mlir::pto::kValue32;
  }
  return type.isF16() || type.isF32();
}


[[maybe_unused]] static unsigned getVMIElementBitWidth(Type type) {
  if (isa<IndexType>(type)) {
    return mlir::pto::kValue64;
  }
  return pto::getPTOStorageElemBitWidth(type);
}

[[maybe_unused]] static int64_t divideCeilNonNegative(int64_t value, int64_t divisor) {
  if (divisor <= 0) {
    return 0;
  }
  return value == 0 ? 0 : (value + divisor - 1) / divisor;
}


[[maybe_unused]] static FailureOr<int64_t> getVMIElementCount(Type type) {
  if (auto vregType = dyn_cast<VMIVRegType>(type)) {
    return vregType.getElementCount();
  }
  if (auto maskType = dyn_cast<VMIMaskType>(type)) {
    return maskType.getElementCount();
  }
  return failure();
}


[[maybe_unused]] static FailureOr<int64_t> getLayoutFactor(Type type) {
  FailureOr<VMILayoutAttr> layout = getAssignedVMILayout(type);
  if (failed(layout)) {
    return failure();
  }
  return (*layout).isDenseSplit() ? (*layout).getFactor() : 1;
}

[[maybe_unused]] static FailureOr<int64_t> getLayoutBlockElems(Type type) {
  return getVMILayoutBlockElems(type);
}






[[maybe_unused]] static bool isLayoutAssigned(VMIVRegType type) {
  return static_cast<bool>(type.getLayoutAttr());
}

[[maybe_unused]] static bool isLayoutAssigned(VMIMaskType type) {
  return static_cast<bool>(type.getLayoutAttr());
}



[[maybe_unused]] static LogicalResult verifyElementwiseVRegOp(Operation *op, VMIVRegType lhs,
                                             VMIVRegType rhs,
                                             VMIVRegType result) {
  return verifyAllSameVRegShapeAndLayout(op, {lhs, rhs, result},
                                         /*requireSameElement=*/true);
}






[[maybe_unused]] static Type getMemoryElementType(Type type) {
  if (auto ptrType = dyn_cast<PtrType>(type)) {
    return ptrType.getElementType();
  }
  if (auto memrefType = dyn_cast<MemRefType>(type)) {
    return memrefType.getElementType();
  }
  return {};
}


[[maybe_unused]] static LogicalResult verifyUBBackedMemory(Operation *op, Type memoryType,
                                          StringRef role) {
  if (isUBBackedMemoryType(memoryType)) {
    return success();
  }
  return op->emitOpError() << "requires memory " << role
                           << " to be UB-backed";
}


// 8->16 gather promotion is a zero-extension (unsigned) operation. signless
// i8/i16 are accepted and treated as unsigned bytes; sign-extension is not
// supported (see VMIVgatherOp / Vgather2Op description).



[[maybe_unused]] static bool isSupported16BitGatherResult(Type sourceElemType,
                                         Type resultElemType) {
  // New 8 -> 16 path: i8/ui8 -> i16/ui16 with matching integer semantics.
  if (isVMI8To16GatherPair(sourceElemType, resultElemType)) {
    return true;
  }
  return isSameWidth16BitGatherPair(sourceElemType, resultElemType);
}











} // namespace

// Batch6: VMI_ops 拆分跨文件声明(定义分布在 VMI_ops/VMI_ops_mem/VMI_ops_cvt/VMI_ops_group)
LogicalResult verifyChannelMergeLayout(Operation *op, VMIVRegType resultType, ValueRange inputs);
LogicalResult verifyChannelSplitLayout(Operation *op, VMIVRegType sourceType, ValueRange results);
LogicalResult verifyVCReductionElementAndMask(Operation *op, VMIVRegType sourceType, VMIMaskType maskType, bool &isFloat);

// Batch6 补充
LogicalResult verifyVMIVariadicPmodeMask(Operation *op, ValueRange maskParts, VMIVRegType dataType, std::optional<StringRef> pmode);
LogicalResult verifySignedI32OrF16F32ElementType(Operation *op, Type elementType);

LogicalResult verifyVMIPmodeMask(Operation *op, VMIMaskType maskType, VMIVRegType dataType, std::optional<StringRef> pmode);
// Batch12: MaskMatchesData + pmode tail shared
template <typename OpTy>
static LogicalResult verifyMaskMatchesDataPmode(OpTy op, VMIMaskType maskType,
                                                VMIVRegType resultType) {
  if (failed(verifyMaskMatchesData(op.getOperation(), maskType, resultType))) {
    return failure();
  }
  if (auto pmode = op.getPmode()) {
    if (pmode.value() != "merge" && pmode.value() != "zero") {
      return op.emitOpError("pmode must be 'merge' or 'zero'");
    }
  }
  return success();
}

#endif // PTO_IR_VMI_INTERNAL_H

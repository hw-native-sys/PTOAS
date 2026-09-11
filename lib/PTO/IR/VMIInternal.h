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

enum class VMIIntSignSemantics { Unsigned, Signed, Any };

// Batch6: 以下 helper 定义迁至 VMI_helpers.cpp(外部链接)
std::string formatVMIVRegType(int64_t elementCount, mlir::Type elementType, mlir::Attribute layout);
std::string formatVMIMaskType(int64_t elementCount, llvm::StringRef granularity, mlir::Attribute layout);
bool matchesVMIIntSemantics(mlir::IntegerType intType, VMIIntSignSemantics semantics);
bool isCompatibleVMIScalarForSemanticType(mlir::Type semanticType, mlir::Type scalarType);
mlir::LogicalResult parseOptionalVMILayout(mlir::AsmParser &parser, mlir::Attribute &layout);
llvm::FailureOr<mlir::pto::VMILayoutAttr> getAssignedVMILayout(mlir::Type type);
int64_t getMaskGranularityBitWidth(llvm::StringRef granularity);
llvm::StringRef getMaskGranularityForBitWidth(int64_t bits);
llvm::FailureOr<llvm::StringRef> getVMIMaskPhysicalGranularity(mlir::pto::VMIMaskType type);
llvm::FailureOr<int64_t> getPhysicalLanesPerPart(mlir::Type type);
llvm::FailureOr<int64_t> getDenseLaneStride(mlir::Type type);
mlir::LogicalResult verifyAllSameVRegShapeAndLayout(mlir::Operation *op, llvm::ArrayRef<mlir::pto::VMIVRegType> types, bool requireSameElement);
mlir::LogicalResult verifyAllSameVRegShapeAndLayoutPresence( mlir::Operation *op, llvm::ArrayRef<mlir::pto::VMIVRegType> types, bool requireSameElement);
mlir::LogicalResult verifyFloatUnaryVRegOp(mlir::Operation *op, mlir::pto::VMIVRegType source, mlir::pto::VMIVRegType result);
mlir::LogicalResult verifyFloatTernaryVRegOp(mlir::Operation *op, mlir::pto::VMIVRegType lhs, mlir::pto::VMIVRegType rhs, mlir::pto::VMIVRegType acc, mlir::pto::VMIVRegType result);
mlir::LogicalResult verifyAllSameMaskShapeLayoutAndGranularity(mlir::Operation *op, llvm::ArrayRef<mlir::pto::VMIMaskType> types);
mlir::LogicalResult verifyMaskMatchesData(mlir::Operation *op, mlir::pto::VMIMaskType maskType, mlir::pto::VMIVRegType dataType);
bool isUBBackedMemoryType(mlir::Type type);
mlir::LogicalResult verifyMemoryElementMatches(mlir::Operation *op, mlir::Type memoryType, mlir::pto::VMIVRegType dataType, llvm::StringRef role);
bool isVMI8To16GatherPair(mlir::Type sourceElemType, mlir::Type resultElemType);
mlir::LogicalResult verifyGatherMemoryElementMatches( mlir::Operation *op, mlir::Type memoryType, mlir::pto::VMIVRegType dataType, llvm::StringRef role);
bool isSameWidth16BitGatherPair(mlir::Type sourceElemType, mlir::Type resultElemType);
mlir::LogicalResult verifyContiguousIfLayoutAssigned(mlir::Operation *op, mlir::pto::VMIVRegType type, llvm::StringRef role);
bool isPackedByteGroupStore(mlir::Type memoryType, mlir::pto::VMIVRegType dataType);
mlir::LogicalResult verifyNumGroups(mlir::Operation *op, mlir::pto::VMIVRegType type, int64_t numGroups);
mlir::LogicalResult verifyPhysicalVRegParts(mlir::Operation *op, mlir::pto::VMIVRegType vregType, mlir::TypeRange physicalTypes);
mlir::LogicalResult verifyPhysicalMaskParts(mlir::Operation *op, mlir::pto::VMIMaskType maskType, mlir::TypeRange physicalTypes);
mlir::LogicalResult verifyPhysicalParts(mlir::Operation *op, mlir::Type vmiType, mlir::TypeRange physicalTypes);
std::optional<int64_t> mapDenseLogicalLaneToPartIndex(int64_t elementCount, int64_t factor, int64_t blockElems, int64_t logicalLane, int64_t &part);
std::optional<int64_t> mapDensePartIndexToLogicalLane(int64_t elementCount, int64_t factor, int64_t blockElems, int64_t part, int64_t indexInPart);
int64_t getDenseLogicalLanesInPart(int64_t elementCount, int64_t factor, int64_t blockElems, int64_t part);
mlir::LogicalResult verifyReductionGroupAndPmode( mlir::Operation *op, mlir::pto::VMIVRegType sourceType, mlir::pto::VMIVRegType resultType, mlir::IntegerAttr groupAttr, std::optional<llvm::StringRef> pmode);

namespace {


[[maybe_unused]] inline bool isSupportedVMIElementType(mlir::Type type) {
  return mlir::isa<mlir::IntegerType, mlir::FloatType, mlir::IndexType>(type) ||
         mlir::pto::isPTOLowPrecisionType(type);
}

[[maybe_unused]] inline bool isVMIFloatLikeType(mlir::Type type) {
  return mlir::isa<mlir::FloatType>(type) || mlir::pto::isPTOLowPrecisionType(type);
}

[[maybe_unused]] inline bool involvesBF16x2(mlir::Type sourceType, mlir::Type resultType) {
  return mlir::pto::isPTOBF16x2Type(sourceType) ||
         mlir::pto::isPTOBF16x2Type(resultType);
}

[[maybe_unused]] inline bool isVMIPackedFloatCarrierType(mlir::Type type) {
  return mlir::pto::isPTOHiFloat8x2Type(type) ||
         mlir::pto::isPTOFloat4PackedType(type) ||
         mlir::pto::isPTOBF16x2Type(type);
}

[[maybe_unused]] inline bool involvesVMIPackedFloatCarrier(mlir::Type sourceType, mlir::Type resultType) {
  return isVMIPackedFloatCarrierType(sourceType) ||
         isVMIPackedFloatCarrierType(resultType);
}

[[maybe_unused]] inline mlir::LogicalResult verifyBF16x2ComputeElementType(mlir::Operation *op, mlir::Type type) {
  if (mlir::pto::isPTOBF16x2Type(type)) {
    return op->emitOpError(
        "does not support bf16x2 VMI element type; bf16x2 is conversion-only");
}
  return mlir::success();
}

[[maybe_unused]] inline bool isVMIIntegerLikeType(mlir::Type type) {
  return mlir::isa<mlir::IntegerType, mlir::IndexType>(type);
}

[[maybe_unused]] inline bool isVMIF16OrF32Type(mlir::Type type) {
  return type.isF16() || type.isF32();
}

[[maybe_unused]] inline bool isVMIF16BF16OrF32Type(mlir::Type type) {
  return type.isF16() || type.isBF16() || type.isF32();
}

[[maybe_unused]] inline bool isVMIPredicateMaskableElementType(mlir::Type type) {
  unsigned elementBits = mlir::pto::getPTOStorageElemBitWidth(type);
  return elementBits == mlir::pto::kValue8 || elementBits == mlir::pto::kValue16 || elementBits == mlir::pto::kValue32;
}

[[maybe_unused]] inline bool isVMIAnyI8I16I32Type(mlir::Type type) {
  auto integerType = mlir::dyn_cast<mlir::IntegerType>(type);
  if (!integerType) {
    return false;
  }
  return integerType.getWidth() == mlir::pto::kValue8 || integerType.getWidth() == mlir::pto::kValue16 ||
         integerType.getWidth() == mlir::pto::kValue32;
}

[[maybe_unused]] inline bool isVMII8I16I32OrF16BF16F32Type(mlir::Type type) {
  return isVMIAnyI8I16I32Type(type) || isVMIF16BF16OrF32Type(type);
}

[[maybe_unused]] inline bool isVMII16I32OrF16BF16F32Type(mlir::Type type) {
  auto intType = mlir::dyn_cast<mlir::IntegerType>(type);
  bool supportedInteger =
      intType && (intType.getWidth() == mlir::pto::kValue16 || intType.getWidth() == mlir::pto::kValue32);
  return supportedInteger || isVMIF16BF16OrF32Type(type);
}

[[maybe_unused]] inline bool isVMII8I16I32OrF16F32Type(mlir::Type type) {
  return isVMIAnyI8I16I32Type(type) || isVMIF16OrF32Type(type);
}

[[maybe_unused]] inline bool isVMISignedI8I16I32Type(mlir::Type type) {
  auto integerType = mlir::dyn_cast<mlir::IntegerType>(type);
  if (!integerType || !integerType.isSigned()) {
    return false;
  }
  return integerType.getWidth() == mlir::pto::kValue8 || integerType.getWidth() == mlir::pto::kValue16 ||
         integerType.getWidth() == mlir::pto::kValue32;
}

[[maybe_unused]] inline bool isVMISignedIntegerType(mlir::Type type) {
  auto integerType = mlir::dyn_cast<mlir::IntegerType>(type);
  return integerType && integerType.isSigned();
}

[[maybe_unused]] inline bool isVMIUnsignedOrSignlessIntegerType(mlir::Type type) {
  auto integerType = mlir::dyn_cast<mlir::IntegerType>(type);
  return integerType && (integerType.isUnsigned() || integerType.isSignless());
}

// ---------------------------------------------------------------------------
// VMI integer element type sign-semantics helper
//
// CONVENTION: VMI op verifiers that need "unsigned semantics" or "signed
// semantics" on an integer element type MUST route the sign check through
// matchesVMIIntSemantics(...) instead of calling mlir::IntegerType::isUnsigned()
// / isSigned() directly.
//
// Signless integers are treated as equivalent to UNSIGNED only. They are
// NOT accepted for signed semantics: signed hardware ops require an
// explicitly signed integer type, to avoid silent sign-extension bugs when
// a producer happens to emit a signless value.
//
// Width / kind / mlir::IntegerType-cast checks stay inline at each callsite;
// only the sign-semantics decision is centralized here.
// ---------------------------------------------------------------------------
[[maybe_unused]] inline bool isVMIIotaElementType(mlir::Type type) {
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
    return intType.getWidth() == mlir::pto::kValue8 || intType.getWidth() == mlir::pto::kValue16 ||
           intType.getWidth() == mlir::pto::kValue32;
  }
  return type.isF16() || type.isF32();
}


[[maybe_unused]] inline unsigned getVMIElementBitWidth(mlir::Type type) {
  if (mlir::isa<mlir::IndexType>(type)) {
    return mlir::pto::kValue64;
  }
  return mlir::pto::getPTOStorageElemBitWidth(type);
}

[[maybe_unused]] inline int64_t divideCeilNonNegative(int64_t value, int64_t divisor) {
  if (divisor <= 0) {
    return 0;
  }
  return value == 0 ? 0 : (value + divisor - 1) / divisor;
}


[[maybe_unused]] inline llvm::FailureOr<int64_t> getVMIElementCount(mlir::Type type) {
  if (auto vregType = mlir::dyn_cast<mlir::pto::VMIVRegType>(type)) {
    return vregType.getElementCount();
  }
  if (auto maskType = mlir::dyn_cast<mlir::pto::VMIMaskType>(type)) {
    return maskType.getElementCount();
  }
  return mlir::failure();
}

[[maybe_unused]] inline llvm::FailureOr<int64_t> getLayoutFactor(mlir::Type type) {
  llvm::FailureOr<mlir::pto::VMILayoutAttr> layout = getAssignedVMILayout(type);
  if (mlir::failed(layout)) {
    return mlir::failure();
  }
  return (*layout).isDenseSplit() ? (*layout).getFactor() : 1;
}

[[maybe_unused]] inline llvm::FailureOr<int64_t> getLayoutBlockElems(mlir::Type type) {
  return mlir::pto::getVMILayoutBlockElems(type);
}

[[maybe_unused]] inline bool isLayoutAssigned(mlir::pto::VMIVRegType type) {
  return static_cast<bool>(type.getLayoutAttr());
}

[[maybe_unused]] inline bool isLayoutAssigned(mlir::pto::VMIMaskType type) {
  return static_cast<bool>(type.getLayoutAttr());
}

[[maybe_unused]] inline mlir::LogicalResult verifyElementwiseVRegOp(mlir::Operation *op, mlir::pto::VMIVRegType lhs,
                                             mlir::pto::VMIVRegType rhs,
                                             mlir::pto::VMIVRegType result) {
  return verifyAllSameVRegShapeAndLayout(op, {lhs, rhs, result},
                                         /*requireSameElement=*/true);
}

[[maybe_unused]] inline mlir::Type getMemoryElementType(mlir::Type type) {
  if (auto ptrType = mlir::dyn_cast<mlir::pto::PtrType>(type)) {
    return ptrType.getElementType();
  }
  if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type)) {
    return memrefType.getElementType();
  }
  return {};
}

[[maybe_unused]] inline mlir::LogicalResult verifyUBBackedMemory(mlir::Operation *op, mlir::Type memoryType,
                                          llvm::StringRef role) {
  if (isUBBackedMemoryType(memoryType)) {
    return mlir::success();
  }
  return op->emitOpError() << "requires memory " << role
                           << " to be UB-backed";
}

// 8->16 gather promotion is a zero-extension (unsigned) operation. signless
// i8/i16 are accepted and treated as unsigned bytes; sign-extension is not
// supported (see VMIVgatherOp / Vgather2Op description).
[[maybe_unused]] inline bool isSupported16BitGatherResult(mlir::Type sourceElemType,
                                         mlir::Type resultElemType) {
  // New 8 -> 16 path: i8/ui8 -> i16/ui16 with matching integer semantics.
  if (isVMI8To16GatherPair(sourceElemType, resultElemType)) {
    return true;
  }
  return isSameWidth16BitGatherPair(sourceElemType, resultElemType);
}

} // namespace

// Batch6: VMI_ops 拆分跨文件声明(定义分布在 VMI_ops/VMI_ops_mem/VMI_ops_cvt/VMI_ops_group)
mlir::LogicalResult verifyChannelMergeLayout(mlir::Operation *op, mlir::pto::VMIVRegType resultType, mlir::ValueRange inputs);
mlir::LogicalResult verifyChannelSplitLayout(mlir::Operation *op, mlir::pto::VMIVRegType sourceType, mlir::ValueRange results);
mlir::LogicalResult verifyVCReductionElementAndMask(mlir::Operation *op, mlir::pto::VMIVRegType sourceType, mlir::pto::VMIMaskType maskType, bool &isFloat);

// Batch6 补充
mlir::LogicalResult verifyVMIVariadicPmodeMask(mlir::Operation *op, mlir::ValueRange maskParts, mlir::pto::VMIVRegType dataType, std::optional<llvm::StringRef> pmode);
mlir::LogicalResult verifySignedI32OrF16F32ElementType(mlir::Operation *op, mlir::Type elementType);

mlir::LogicalResult verifyVMIPmodeMask(mlir::Operation *op, mlir::pto::VMIMaskType maskType, mlir::pto::VMIVRegType dataType, std::optional<llvm::StringRef> pmode);
// Batch12: MaskMatchesData + pmode tail shared
template <typename OpTy>
static mlir::LogicalResult verifyMaskMatchesDataPmode(OpTy op, mlir::pto::VMIMaskType maskType,
                                                mlir::pto::VMIVRegType resultType) {
  if (mlir::failed(verifyMaskMatchesData(op.getOperation(), maskType, resultType))) {
    return mlir::failure();
  }
  if (auto pmode = op.getPmode()) {
    if (pmode.value() != "merge" && pmode.value() != "zero") {
      return op.emitOpError("pmode must be 'merge' or 'zero'");
    }
  }
  return mlir::success();
}

#endif // PTO_IR_VMI_INTERNAL_H

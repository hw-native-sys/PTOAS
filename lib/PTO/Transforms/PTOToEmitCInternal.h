// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_PTOTOEMITCINTERNAL_H
#define MLIR_DIALECT_PTO_TRANSFORMS_PTOTOEMITCINTERNAL_H

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ErrorHandling.h"

#include <optional>
#include <string>

namespace mlir::pto {

Value peelUnrealized(Value v);

Value makeEmitCOpaqueConstant(ConversionPatternRewriter &rewriter,
                              Location loc, Type type,
                              llvm::StringRef literal);

Value makeEmitCIntConstant(ConversionPatternRewriter &rewriter, Location loc,
                           Type type, int64_t value);

Value emitCCast(ConversionPatternRewriter &rewriter, Location loc, Type dstType,
                Value src);

Value castSignlessIntToUnsignedSameWidth(ConversionPatternRewriter &rewriter,
                                         Location loc, Value v,
                                         unsigned bitWidth);

inline constexpr int64_t kMrgSortExecutedNumRank = 1;
inline constexpr int64_t kMrgSortExecutedNumCount = 4;

inline constexpr int32_t kSLayout1 = 1;
inline constexpr int32_t kSLayout2 = 2;
inline constexpr int32_t kPadValue1 = 1;
inline constexpr int32_t kPadValue2 = 2;
inline constexpr int32_t kPadValue3 = 3;

inline constexpr int kTileTemplateRank2D = 2;
inline constexpr int64_t kFloat4PackedDimExpansion2 = 2;

inline emitc::OpaqueType getSignedIntOpaqueType(MLIRContext *ctx,
                                                unsigned bitWidth) {
  switch (bitWidth) {
  case 1:
    return emitc::OpaqueType::get(ctx, "int8_t");
  case kPTOI8BitWidth:
    return emitc::OpaqueType::get(ctx, "int8_t");
  case kPTOI16BitWidth:
    return emitc::OpaqueType::get(ctx, "int16_t");
  case kPTOI32BitWidth:
    return emitc::OpaqueType::get(ctx, "int32_t");
  case kPTOI64BitWidth:
    return emitc::OpaqueType::get(ctx, "int64_t");
  case kPTOI128BitWidth:
    return emitc::OpaqueType::get(ctx, "__int128");
  default:
    llvm::errs() << "[Debug] Unsupported signed integer bitwidth: " << bitWidth
                 << "\n";
    return emitc::OpaqueType::get(ctx, "int64_t");
  }
}

inline emitc::OpaqueType getUnsignedIntOpaqueType(MLIRContext *ctx,
                                                  unsigned bitWidth) {
  switch (bitWidth) {
  case 1:
    return emitc::OpaqueType::get(ctx, "uint8_t");
  case kPTOI8BitWidth:
    return emitc::OpaqueType::get(ctx, "uint8_t");
  case kPTOI16BitWidth:
    return emitc::OpaqueType::get(ctx, "uint16_t");
  case kPTOI32BitWidth:
    return emitc::OpaqueType::get(ctx, "uint32_t");
  case kPTOI64BitWidth:
    return emitc::OpaqueType::get(ctx, "uint64_t");
  case kPTOI128BitWidth:
    return emitc::OpaqueType::get(ctx, "unsigned __int128");
  default:
    llvm::errs() << "[Debug] Unsupported unsigned integer bitwidth: "
                 << bitWidth << "\n";
    return emitc::OpaqueType::get(ctx, "uint64_t");
  }
}

inline emitc::OpaqueType getWiderSignedIntOpaqueType(MLIRContext *ctx,
                                                     unsigned bitWidth) {
  switch (bitWidth) {
  case 1:
  case kPTOI8BitWidth:
    return getSignedIntOpaqueType(ctx, kPTOI16BitWidth);
  case kPTOI16BitWidth:
    return getSignedIntOpaqueType(ctx, kPTOI32BitWidth);
  case kPTOI32BitWidth:
    return getSignedIntOpaqueType(ctx, kPTOI64BitWidth);
  case kPTOI64BitWidth:
    return getSignedIntOpaqueType(ctx, kPTOI128BitWidth);
  default:
    return getSignedIntOpaqueType(ctx, kPTOI128BitWidth);
  }
}

inline emitc::OpaqueType getWiderUnsignedIntOpaqueType(MLIRContext *ctx,
                                                       unsigned bitWidth) {
  switch (bitWidth) {
  case 1:
  case kPTOI8BitWidth:
    return getUnsignedIntOpaqueType(ctx, kPTOI16BitWidth);
  case kPTOI16BitWidth:
    return getUnsignedIntOpaqueType(ctx, kPTOI32BitWidth);
  case kPTOI32BitWidth:
    return getUnsignedIntOpaqueType(ctx, kPTOI64BitWidth);
  case kPTOI64BitWidth:
    return getUnsignedIntOpaqueType(ctx, kPTOI128BitWidth);
  default:
    return getUnsignedIntOpaqueType(ctx, kPTOI128BitWidth);
  }
}

inline FailureOr<std::string>
buildEmitCOpaqueConstantLiteral(Type targetType, Attribute valueAttr) {
  auto opaqueTy = dyn_cast<emitc::OpaqueType>(targetType);
  if (!opaqueTy)
    return failure();
  if (opaqueTy.getValue() == "pto::MrgSortExecutedNumList") {
    auto dense = dyn_cast_or_null<DenseIntElementsAttr>(valueAttr);
    if (!dense)
      return failure();

    auto vecTy = dyn_cast<VectorType>(dense.getType());
    if (!vecTy || vecTy.getRank() != kMrgSortExecutedNumRank ||
        vecTy.getNumElements() != kMrgSortExecutedNumCount ||
        !vecTy.getElementType().isInteger(kPTOI16BitWidth))
      return failure();

    std::string literal;
    llvm::raw_string_ostream os(literal);
    os << "pto::MrgSortExecutedNumList{";
    bool first = true;
    for (APInt elem : dense.getValues<APInt>()) {
      if (!first)
        os << ", ";
      first = false;
      os << elem.getZExtValue();
    }
    os << "}";
    os.flush();
    return literal;
  }

  return failure();
}

std::string getEmitCScalarTypeToken(Type elemTy);

int64_t getEmitCScalarByteWidth(Type elemTy);

Value peelEmitCCasts(Value v);

bool isEmitCTileLikeValue(Value v);

Value scalePackedTileDynamicDim(ConversionPatternRewriter &rewriter,
                                Location loc, Type elemTy,
                                pto::BLayout blayout, Value emitted,
                                int dimIdx);

Value buildTileCtorDimValue(ConversionPatternRewriter &rewriter, Location loc,
                            Value emitted, int64_t fallback);

std::string getTileRoleToken(Attribute memorySpace);

inline std::string
getTileBufBLayoutToken(pto::TileBufConfigAttr configAttr) {
  std::string blTok = "BLayout::RowMajor";
  if (auto blAttr = dyn_cast<BLayoutAttr>(configAttr.getBLayout())) {
    if (static_cast<int32_t>(blAttr.getValue()) == 1)
      blTok = "BLayout::ColMajor";
  }
  return blTok;
}

inline std::string
getTileBufSLayoutToken(pto::TileBufConfigAttr configAttr) {
  std::string slTok = "SLayout::NoneBox";
  if (auto slAttr = dyn_cast<SLayoutAttr>(configAttr.getSLayout())) {
    int32_t slVal = static_cast<int32_t>(slAttr.getValue());
    if (slVal == kSLayout1)
      slTok = "SLayout::RowMajor";
    else if (slVal == kSLayout2)
      slTok = "SLayout::ColMajor";
  }
  return slTok;
}

inline std::string
getTileBufPadToken(pto::TileBufConfigAttr configAttr) {
  std::string padTok = "PadValue::Null";
  if (auto padAttr = dyn_cast<PadValueAttr>(configAttr.getPad())) {
    switch (static_cast<int32_t>(padAttr.getValue())) {
    case kPadValue1:
      padTok = "PadValue::Zero";
      break;
    case kPadValue2:
      padTok = "PadValue::Max";
      break;
    case kPadValue3:
      padTok = "PadValue::Min";
      break;
    default:
      padTok = "PadValue::Null";
      break;
    }
  }
  return padTok;
}

std::string getTileBufCompactToken(pto::TileBufConfigAttr configAttr);

Value castAddressToU64(ConversionPatternRewriter &rewriter, Location loc,
                       Value value);

llvm::StringRef addrSpaceQualifier(pto::AddressSpace as);

bool tileDataReturnsIntegralAddress(pto::AddressSpace as);

int64_t multiplyOrDynamic(int64_t lhs, int64_t rhs);

inline pto::BLayout getTileBufBLayoutValue(
    pto::TileBufConfigAttr configAttr) {
  if (auto blAttr = dyn_cast<BLayoutAttr>(configAttr.getBLayout()))
    return blAttr.getValue();
  return pto::BLayout::RowMajor;
}

inline int64_t renderTileTemplateDim(int64_t rawDim, Type elemTy,
                                     pto::BLayout blayout, int dimIdx) {
  if (!(dimIdx >= 0 && dimIdx < kTileTemplateRank2D)) {
    llvm::report_fatal_error(
        "renderTileTemplateDim expects a rank-2 rows/cols dimension index");
  }
  if (rawDim == ShapedType::kDynamic)
    return rawDim;
  if (!pto::isPTOFloat4PackedType(elemTy))
    return rawDim;
  int packedDim = blayout == pto::BLayout::ColMajor ? 0 : 1;
  return dimIdx == packedDim ? rawDim * kFloat4PackedDimExpansion2 : rawDim;
}

std::optional<std::string> getEmitCTileTypeString(pto::TileBufType type);

bool isSetFFTsPointerLikeType(Type ty);

bool isEmitCGlobalTensorLikeType(Type ty);

std::string getGlobalTensorTypeStringFromShapeAndStrides(
    Type elemTy, ArrayRef<int64_t> shape, ArrayRef<int64_t> strides,
    llvm::StringRef layoutEnum = "pto::Layout::ND");

std::string getElemTypeStringForGT(Type elemTy);

SmallVector<int64_t> buildRowMajorStrides(ArrayRef<int64_t> shape);

void buildGlobalTensorShapeAndStride(ArrayRef<int64_t> shape,
                                     ArrayRef<int64_t> strides,
                                     SmallVectorImpl<int64_t> &shape5D,
                                     SmallVectorImpl<int64_t> &stride5D);

std::string joinIntTemplateParams(ArrayRef<int64_t> values);

std::optional<mlir::pto::Layout>
resolveLayoutForGlobalTensor(Operation *anchor, Value basePtr);

std::string layoutToEmitCString(mlir::pto::Layout layout);

Value buildGlobalTensorFromMemref(ConversionPatternRewriter &rewriter,
                                  Location loc, Value basePtr,
                                  MemRefType mrTy, Operation *anchor);

FailureOr<std::string> buildTPipeTokenFromInitOp(Operation *op,
                                                 PTOArch targetArch);

Value castToGMBytePointer(ConversionPatternRewriter &rewriter, Location loc,
                          Value value);

Value materializeTensorViewDataPointer(ConversionPatternRewriter &rewriter,
                                       Location loc, Value value,
                                       Type originalType);

Value materializeAddressAsPointer(ConversionPatternRewriter &rewriter,
                                  Location loc, Value addr,
                                  pto::AddressSpace as,
                                  llvm::StringRef elemTok);

Value applyStaticMemrefOffset(ConversionPatternRewriter &rewriter,
                              Location loc, Value basePtr, int64_t offset);

FailureOr<Value> buildAsyncScratchTileValue(ConversionPatternRewriter &rewriter,
                                            Location loc, Value originalScratch,
                                            Value emittedScratch);

bool needsA5NoSplitVectorGuard(Operation *op);

Value materializeTileDataValue(ConversionPatternRewriter &rewriter,
                               Location loc, Value tile,
                               pto::AddressSpace as,
                               llvm::StringRef elemTypeToken);

void populatePTOToEmitCArithPatterns(RewritePatternSet &patterns,
                                     TypeConverter &typeConverter,
                                     MLIRContext *ctx);

void populatePTOToEmitCTilePatterns(RewritePatternSet &patterns,
                                    TypeConverter &typeConverter,
                                    MLIRContext *ctx);

void populatePTOToEmitCTileExtraPatterns(RewritePatternSet &patterns,
                                         TypeConverter &typeConverter,
                                         MLIRContext *ctx);

void populatePTOToEmitCTileMaterializationPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *ctx);

void populatePTOToEmitCSyncPatterns(RewritePatternSet &patterns,
                                    TypeConverter &typeConverter,
                                    MLIRContext *ctx, PTOArch targetArch);

void populatePTOToEmitCCommPatterns(RewritePatternSet &patterns,
                                    TypeConverter &typeConverter,
                                    MLIRContext *ctx, PTOArch targetArch);

void populatePTOToEmitCKernelOpPatterns(RewritePatternSet &patterns,
                                        TypeConverter &typeConverter,
                                        MLIRContext *ctx);

LogicalResult runPTOToEmitCSCFPreLowering(ModuleOp mop, MLIRContext *ctx);

void populatePTOToEmitCControlFlowPatterns(RewritePatternSet &patterns,
                                           TypeConverter &typeConverter,
                                           MLIRContext *ctx);

void populatePTOToEmitCSubviewPatterns(RewritePatternSet &patterns,
                                       TypeConverter &typeConverter,
                                       MLIRContext *ctx);

void populatePTOToEmitCSimpleOpPatterns(RewritePatternSet &patterns,
                                        TypeConverter &typeConverter,
                                        MLIRContext *ctx);

void populatePTOToEmitCRuntimeOpPatterns(RewritePatternSet &patterns,
                                         TypeConverter &typeConverter,
                                         MLIRContext *ctx, PTOArch targetArch);

void populatePTOToEmitCMemoryOpPatterns(RewritePatternSet &patterns,
                                        TypeConverter &typeConverter,
                                        MLIRContext *ctx);

} // namespace mlir::pto

#endif // MLIR_DIALECT_PTO_TRANSFORMS_PTOTOEMITCINTERNAL_H

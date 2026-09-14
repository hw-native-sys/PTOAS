// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- TileSupport.cpp - Tile lowering helpers --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "TileInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

// CANN Open Software License Agreement Version 2.0 (the "License").
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.

//===- PTOToEmitCTile.cpp - tile/view/pipe lowering ---------===//
//===----------------------------------------------------------------------===//





// Resolve the third TPUSH/TPOP template token: the fixpipe config alias
// when the pipe carries an acc-push epilogue, otherwise the split mode.
// tpush/tpop lowering: resolve the TPipe/tile/split template tokens and emit
// T(PUSH|POP)<pipe, tile, split>(pipeHandle, tile[, aivSubblockId]).

//===----------------------------------------------------------------------===//
// populate patterns

// Tile role token for a non-GM reinterpret_cast target address space.
// Scaling tiles infer their role from the source value when possible.
const char *reinterpretCastTileRole(pto::AddressSpace as,
                                           Value source) {
  if (as == pto::AddressSpace::SCALING)
    if (const char *inferredRole = inferScalingRoleFromValue(source))
      return inferredRole;
  return tileRoleToken(pto::AddressSpaceAttr::get(source.getContext(), as));
}

// Conservative Tile<...> type string for a reinterpret_cast target: the
// result shape (fallback 32x32) with a default config.
std::string buildReinterpretCastTileTypeString(MemRefType resMrTy,
                                                      Type elemTy,
                                                      const char *roleTok) {
  int64_t rows = 32, cols = 32;
  if (resMrTy.getRank() >= 2 && resMrTy.hasStaticShape()) {
    rows = resMrTy.getDimSize(0);
    cols = resMrTy.getDimSize(1);
  }
  int64_t templateRows =
      renderTileTemplateDim(rows, elemTy, pto::BLayout::RowMajor, 0);
  int64_t templateCols =
      renderTileTemplateDim(cols, elemTy, pto::BLayout::RowMajor, 1);
  return std::string("Tile<") + roleTok + ", " +
         getEmitCScalarTypeToken(elemTy) + ", " + std::to_string(templateRows) +
         ", " + std::to_string(templateCols) + ", BLayout::RowMajor, " +
         std::to_string(templateRows) + ", " + std::to_string(templateCols) +
         ", SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>";
}

// Resolve the underlying u64 address of a non-GM reinterpret_cast source:
// tiles contribute their `.data()` pointer, plain pointers pass through.
Value reinterpretCastBaseAddress(ConversionPatternRewriter &rewriter,
                                       Location loc, Value source,
                                       pto::AddressSpace as, StringRef elemTok,
                                       Type u64Ty) {
  Value rawPtr = source;
  if (auto ot = dyn_cast<emitc::OpaqueType>(source.getType())) {
    // Only Tiles have a `.data()` member. For plain address-space pointers
    // (e.g. `__ubuf__ float*`), use the pointer value directly.
    if (ot.getValue().starts_with("Tile<"))
      rawPtr = materializeTileDataValue(rewriter, loc, source, as, elemTok);
  }

  return coerceToU64Address(rewriter, loc, rawPtr, u64Ty);
}

//===----------------------------------------------------------------------===//
// pto.taddc lowering -> TADDC(dst, src0, src1, src2)
//===----------------------------------------------------------------------===//

// Reinterpret an alloc_tile address operand as u64 and TASSIGN it to the
// freshly created tile variable.
void assignTileAddress(ConversionPatternRewriter &rewriter,
                              Location loc, MLIRContext *ctx, Value tile,
                              Value addr) {
  addr = peelUnrealized(addr);
  auto u64Ty = emitc::OpaqueType::get(ctx, "uint64_t");
  bool isPointerLike =
      isa<emitc::PointerType>(addr.getType()) ||
      (isa<emitc::OpaqueType>(addr.getType()) &&
       cast<emitc::OpaqueType>(addr.getType()).getValue().ends_with("*"));
  if (isPointerLike) {
    auto rcU64 =
        rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, "uint64_t")});
    addr = rewriter
               .create<emitc::CallOpaqueOp>(loc, u64Ty, "reinterpret_cast",
                                            ArrayAttr{}, rcU64,
                                            ValueRange{addr})
               .getResult(0);
  } else if (addr.getType() != u64Ty) {
    addr = rewriter.create<emitc::CastOp>(loc, u64Ty, addr).getResult();
  }

  rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "TASSIGN",
                                       ArrayAttr{}, ArrayAttr{},
                                       ValueRange{tile, addr});
}

FailureOr<Value>
createEmitCTileVariable(ConversionPatternRewriter &rewriter, Location loc,
                        const TypeConverter *typeConverter,
                        pto::TileBufType tileTy,
                        bool initializeDynamicValidToShape) {
  auto tileTypeString = getEmitCTileTypeString(tileTy);
  if (!tileTypeString)
    return failure();

  Type convertedTy = typeConverter->convertType(tileTy);
  if (!convertedTy)
    convertedTy = emitc::OpaqueType::get(rewriter.getContext(), *tileTypeString);

  if (initializeDynamicValidToShape && tileTy.hasDynamicValid()) {
    auto shape = tileTy.getShape();
    if (shape.size() != 2 || llvm::is_contained(shape, ShapedType::kDynamic))
      return failure();
    Type i32Ty = emitc::OpaqueType::get(rewriter.getContext(), "int32_t");
    pto::BLayout blayout = getTileBufBLayoutValue(tileTy.getConfigAttr());
    SmallVector<Value, 2> constructorArgs;
    constructorArgs.push_back(makeEmitCIntConstant(
        rewriter, loc, i32Ty,
        renderTileTemplateDim(shape[0], tileTy.getElementType(), blayout, 0)));
    constructorArgs.push_back(makeEmitCIntConstant(
        rewriter, loc, i32Ty,
        renderTileTemplateDim(shape[1], tileTy.getElementType(), blayout, 1)));
    return rewriter
        .create<emitc::CallOpaqueOp>(loc, convertedTy, *tileTypeString,
                                     ArrayAttr{}, ArrayAttr{}, constructorArgs)
        .getResult(0);
  }

  Value tile = rewriter
                   .create<emitc::VariableOp>(
                       loc, getEmitCVariableResultType(convertedTy),
                       emitc::OpaqueAttr::get(rewriter.getContext(), ""))
                   .getResult();
  return loadEmitCVariableIfNeeded(rewriter, loc, tile);
}

//===----------------------------------------------------------------------===//
// SCF Control-Flow Pre-Lowering
//
// EmitC translation supports `emitc.for`/`emitc.if` plus CFG-style
// `cf.br`/`cf.cond_br`. Upstream SCFToEmitC patterns only cover `scf.for` and
// `scf.if`, so we pre-lower some SCF ops into those supported forms.
//===----------------------------------------------------------------------===//


FailureOr<Value> buildGlobalTensorViewFromPointer(
    ConversionPatternRewriter &rewriter, Location loc, Value ptr, Type elemTy,
    ArrayRef<int64_t> shape, ArrayRef<int64_t> strides,
    std::optional<SpecialGlobalTensorTypeSpec> specialSpec,
    StringRef layoutEnum) {
  if (llvm::any_of(shape, [](int64_t dim) {
        return dim == ShapedType::kDynamic;
      }))
    return failure();

  auto *ctx = rewriter.getContext();
  SmallVector<int64_t> rowMajorStrides;
  ArrayRef<int64_t> effectiveStrides = strides;
  if (effectiveStrides.empty()) {
    rowMajorStrides = buildRowMajorStrides(shape);
    effectiveStrides = rowMajorStrides;
  }
  SmallVector<int64_t, 5> shape5D;
  SmallVector<int64_t, 5> stride5D;
  buildGlobalTensorShapeAndStride(shape, effectiveStrides, shape5D, stride5D);

  std::string shapeType;
  std::string strideType;
  if (specialSpec) {
    shapeType = specialSpec->shapeTypeExpr;
    strideType = specialSpec->strideTypeExpr;
    layoutEnum = specialSpec->layoutEnum;
  } else {
    shapeType = "pto::Shape<" + joinIntTemplateParams(shape5D) + ">";
    strideType = "pto::Stride<" + joinIntTemplateParams(stride5D) + ">";
  }
  auto shapeVal = rewriter
                      .create<emitc::CallOpaqueOp>(
                          loc, emitc::OpaqueType::get(ctx, shapeType),
                          shapeType, ArrayAttr{}, ArrayAttr{}, ValueRange{})
                      .getResult(0);
  auto strideVal = rewriter
                       .create<emitc::CallOpaqueOp>(
                           loc, emitc::OpaqueType::get(ctx, strideType),
                           strideType, ArrayAttr{}, ArrayAttr{}, ValueRange{})
                       .getResult(0);

  // Keep the GlobalTensor template descriptors identical to the constructor
  // arguments, including the specialized MX shape and stride types.
  std::string gtTypeStr =
      "GlobalTensor<" + getElemTypeStringForGT(elemTy) + ", " + shapeType +
      ", " + strideType + ", " + layoutEnum.str() + ">";
  auto gtType = emitc::OpaqueType::get(ctx, gtTypeStr);
  auto gt = rewriter.create<emitc::CallOpaqueOp>(
      loc, gtType, gtTypeStr, ArrayAttr{}, ArrayAttr{},
      ValueRange{ptr, shapeVal, strideVal});
  return gt.getResult(0);
}

// Right-align runtime shape/stride values to 5 dims: leading shape dims are
// 1, leading strides derive by tight packing (or 1 in the fully-dynamic case).
std::pair<SmallVector<Value, 5>, SmallVector<Value, 5>>
buildRuntime5DValues(ConversionPatternRewriter &rewriter, Location loc,
                     ValueRange runtimeShape, ValueRange runtimeStrides,
                     int64_t shift) {
  SmallVector<Value, 5> shapeValues;
  SmallVector<Value, 5> strideValues;
  for (int64_t dim = 0; dim < shift; ++dim)
    shapeValues.push_back(makeViewIndexConstant(rewriter, loc, 1));
  for (Value value : runtimeShape)
    shapeValues.push_back(castViewIndexToEmitC(rewriter, loc, value));

  strideValues.resize(5);
  for (auto [index, value] : llvm::enumerate(runtimeStrides))
    strideValues[shift + static_cast<int64_t>(index)] =
        castViewIndexToEmitC(rewriter, loc, value);
  if (shift == 5) {
    for (int64_t dim = 0; dim < 5; ++dim)
      strideValues[dim] = makeViewIndexConstant(rewriter, loc, 1);
  } else {
    for (int64_t dim = shift - 1; dim >= 0; --dim) {
      strideValues[dim] =
          rewriter
              .create<emitc::MulOp>(loc, strideValues[dim + 1].getType(),
                                    shapeValues[dim + 1],
                                    strideValues[dim + 1])
              .getResult();
    }
  }
  return {shapeValues, strideValues};
}

FailureOr<Value> buildRuntimeGlobalTensor(
    ConversionPatternRewriter &rewriter, Location loc, Value ptr, Type elemTy,
    ArrayRef<int64_t> staticShape, ValueRange runtimeShape,
    ValueRange runtimeStrides, StringRef layoutEnum) {
  if (staticShape.size() > 5 || runtimeShape.size() != staticShape.size() ||
      runtimeStrides.size() != staticShape.size())
    return failure();

  SmallVector<int64_t, 5> shape5D(5, 1);
  SmallVector<int64_t, 5> stride5D(5, -1);
  int64_t shift = 5 - static_cast<int64_t>(staticShape.size());
  for (auto [index, dim] : llvm::enumerate(staticShape))
    shape5D[shift + static_cast<int64_t>(index)] =
        ShapedType::isDynamic(dim) ? -1 : dim;

  std::string shapeType = "pto::Shape<" + joinIntTemplateParams(shape5D) + ">";
  std::string strideType =
      "pto::Stride<" + joinIntTemplateParams(stride5D) + ">";
  auto [shapeValues, strideValues] =
      buildRuntime5DValues(rewriter, loc, runtimeShape, runtimeStrides, shift);

  Value shape = rewriter
                    .create<emitc::CallOpaqueOp>(
                        loc, emitc::OpaqueType::get(rewriter.getContext(),
                                                   shapeType),
                        shapeType, ArrayAttr{}, ArrayAttr{}, shapeValues)
                    .getResult(0);
  Value stride = rewriter
                     .create<emitc::CallOpaqueOp>(
                         loc, emitc::OpaqueType::get(rewriter.getContext(),
                                                    strideType),
                         strideType, ArrayAttr{}, ArrayAttr{}, strideValues)
                     .getResult(0);
  auto resultType = getRuntimeGlobalTensorOpaqueType(
      rewriter.getContext(), elemTy, staticShape, layoutEnum);
  return rewriter
      .create<emitc::CallOpaqueOp>(loc, resultType, resultType.getValue(),
                                   ArrayAttr{}, ArrayAttr{},
                                   ValueRange{ptr, shape, stride})
      .getResult(0);
}

FailureOr<Value> buildSyncAllGlobalTensorFromPointer(
    ConversionPatternRewriter &rewriter, Location loc, Value ptr, Type elemTy) {
  constexpr int64_t kWorkspaceElements = 16;
  SmallVector<int64_t, 1> shape{kWorkspaceElements};
  SmallVector<int64_t, 1> strides{1};
  return buildGlobalTensorViewFromPointer(rewriter, loc, ptr, elemTy, shape,
                                          strides);
}

Value castViewIndexToEmitC(ConversionPatternRewriter &rewriter,
                                  Location loc, Value value) {
  Type indexTy = emitc::OpaqueType::get(rewriter.getContext(), "int64_t");
  value = peelUnrealized(value);
  if (value.getType() == indexTy)
    return value;
  return rewriter.create<emitc::CastOp>(loc, indexTy, value).getResult();
}

Value getRuntimeGlobalTensorMetadata(
    ConversionPatternRewriter &rewriter, Location loc, Value tensor,
    Value logicalDim, int64_t rank, bool isStride) {
  Value dim = castViewIndexToEmitC(rewriter, loc, logicalDim);
  int64_t shift = 5 - rank;
  if (shift != 0) {
    dim = rewriter
              .create<emitc::AddOp>(loc, dim.getType(), dim,
                                    makeViewIndexConstant(rewriter, loc, shift))
              .getResult();
  }
  StringRef marker = isStride ? StringRef("PTOAS__GLOBAL_TENSOR_GET_STRIDE")
                              : StringRef("PTOAS__GLOBAL_TENSOR_GET_SHAPE");
  return rewriter
      .create<emitc::CallOpaqueOp>(
          loc, dim.getType(), marker, ArrayAttr{}, ArrayAttr{},
          ValueRange{tensor, dim})
      .getResult(0);
}

std::optional<int64_t> getStaticIndexLikeValue(Value value) {
  if (!value)
    return std::nullopt;
  if (auto cst = value.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value();
  if (auto cst = value.getDefiningOp<arith::ConstantIntOp>())
    return cst.value();
  if (auto cst = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue()))
      return getIntegerAttrSignedValue(intAttr);
  }
  return std::nullopt;
}

Value makeViewIndexConstant(ConversionPatternRewriter &rewriter,
                                   Location loc, int64_t value) {
  return makeEmitCIntConstant(
      rewriter, loc, emitc::OpaqueType::get(rewriter.getContext(), "int64_t"),
      value);
}

bool parseIntegerTemplateList(StringRef token, StringRef marker,
                                     SmallVectorImpl<int64_t> &values) {
  size_t pos = token.find(marker);
  if (pos == StringRef::npos)
    return false;
  pos += marker.size();
  size_t end = token.find('>', pos);
  if (end == StringRef::npos)
    return false;

  SmallVector<StringRef, 8> parts;
  token.slice(pos, end).split(parts, ',');
  values.clear();
  for (StringRef part : parts) {
    int64_t value = 0;
    if (part.trim().getAsInteger(10, value))
      return false;
    values.push_back(value);
  }
  return true;
}

bool partitionViewHasStaticResultShape(pto::PartitionViewOp op) {
  auto resTy = dyn_cast<pto::PartitionTensorViewType>(op.getResult().getType());
  if (!resTy) {
    return false;
  }

  int64_t sourceRank = 0;
  if (auto srcTy = dyn_cast<pto::TensorViewType>(op.getSource().getType())) {
    sourceRank = srcTy.getRank();
  } else if (auto srcTy =
                 dyn_cast<pto::PartitionTensorViewType>(op.getSource().getType())) {
    sourceRank = srcTy.getRank();
  } else {
    return false;
  }

  if (op.getOffsets().size() != static_cast<size_t>(sourceRank) ||
      op.getSizes().size() != static_cast<size_t>(sourceRank)) {
    return false;
  }

  for (auto [idx, value] : llvm::enumerate(op.getSizes())) {
    auto cst = getStaticIndexLikeValue(value);
    if (!cst) {
      return false;
    }
    int64_t resultDim = resTy.getShape()[idx];
    if (resultDim != ShapedType::kDynamic && resultDim != *cst) {
      return false;
    }
  }
  return true;
}


FailureOr<std::string> getPipeDataTypeToken(Value value) {
  auto opaqueTy = dyn_cast<emitc::OpaqueType>(value.getType());
  if (!opaqueTy)
    return failure();
  StringRef token = opaqueTy.getValue();
  if (!token.contains("Tile<") && !token.contains("GlobalTensor<"))
    return failure();
  return token.str();
}


LogicalResult getStaticTensorViewStrides(
    Value source, Value convertedSource, int64_t rank,
    SmallVectorImpl<int64_t> &strides) {
  strides.clear();

  if (auto makeView = source.getDefiningOp<pto::MakeTensorViewOp>()) {
    if (static_cast<int64_t>(makeView.getStrides().size()) != rank)
      return failure();
    for (Value strideValue : makeView.getStrides()) {
      auto cst = getStaticIndexLikeValue(strideValue);
      if (!cst)
        return failure();
      strides.push_back(*cst);
    }
    return success();
  }

  Value src = peelUnrealized(convertedSource);
  if (auto opaqueTy = dyn_cast<emitc::OpaqueType>(src.getType())) {
    SmallVector<int64_t, 5> stride5D;
    StringRef token = opaqueTy.getValue();
    if ((parseIntegerTemplateList(token, "pto::Stride<", stride5D) ||
         parseIntegerTemplateList(token, "Stride<", stride5D)) &&
        static_cast<int64_t>(stride5D.size()) >= rank) {
      strides.append(stride5D.end() - rank, stride5D.end());
      return success();
    }
  }

  return failure();
}



} // namespace pto
} // namespace mlir

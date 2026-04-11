//===- PTOToVPTOLowering.cpp - PTO to VPTO lowering helpers --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/VPTOLowering.h"

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOSyncUtils.h"

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/ADT/APFloat.h"

#include <optional>
#include <utility>

namespace mlir {
namespace pto {

namespace {

constexpr StringLiteral kLoweredLoopScopeAttrName = "llvm.loop.aivector_scope";

struct ResolvedTensorView {
  Value root;
  Attribute layoutAttr;
  SmallVector<OpFoldResult> shape;
  SmallVector<OpFoldResult> strides;
  OpFoldResult offsetElems;
};

struct VecNdTransferPlan {
  Value outerCount;
  Value outerSrcStrideElems;
  Value outerDstStrideElems;
  Value loop2Size;
  Value loop1Size;
  Value loop2FirstStrideBytes;
  Value loop2SecondStrideBytes;
  Value loop1FirstStrideBytes;
  Value loop1SecondStrideBytes;
  Value nBurst;
  Value lenBurst;
  Value firstStrideBytes;
  Value secondStrideBytes;
};

struct VPTORowReduceContract {
  StringRef family;
  VPTOTileDomain srcDomain = VPTOTileDomain::Vec;
  VPTOTileDomain dstDomain = VPTOTileDomain::Vec;
  StringRef srcLayout;
  StringRef dstLayout;
  Type elementType;
  Value validRowsValue;
  Value validColsValue;
  int64_t validRows = ShapedType::kDynamic;
  int64_t validCols = ShapedType::kDynamic;
  int64_t dstValidCols = ShapedType::kDynamic;
  VPTOLoopScopeContract loopScope;
};

struct VPTOColReduceContract {
  StringRef family;
  VPTOTileDomain srcDomain = VPTOTileDomain::Vec;
  VPTOTileDomain dstDomain = VPTOTileDomain::Vec;
  StringRef srcLayout;
  StringRef dstLayout;
  Type elementType;
  Value validRowsValue;
  Value validColsValue;
  int64_t validRows = ShapedType::kDynamic;
  int64_t validCols = ShapedType::kDynamic;
  int64_t dstValidRows = ShapedType::kDynamic;
  int64_t dstValidCols = ShapedType::kDynamic;
  bool isBinary = false;
  Value tmp;
  VPTOLoopScopeContract loopScope;
};

struct VPTOPartContract {
  StringRef family;
  VPTOTileDomain src0Domain = VPTOTileDomain::Vec;
  VPTOTileDomain src1Domain = VPTOTileDomain::Vec;
  VPTOTileDomain dstDomain = VPTOTileDomain::Vec;
  StringRef src0Layout;
  StringRef src1Layout;
  StringRef dstLayout;
  Type elementType;
  Value src0ValidRowsValue;
  Value src0ValidColsValue;
  Value src1ValidRowsValue;
  Value src1ValidColsValue;
  Value dstValidRowsValue;
  Value dstValidColsValue;
  int64_t src0ValidRows = ShapedType::kDynamic;
  int64_t src0ValidCols = ShapedType::kDynamic;
  int64_t src1ValidRows = ShapedType::kDynamic;
  int64_t src1ValidCols = ShapedType::kDynamic;
  int64_t dstValidRows = ShapedType::kDynamic;
  int64_t dstValidCols = ShapedType::kDynamic;
  VPTOLoopScopeContract loopScope;
};

struct VPTOExpandContract {
  StringRef family;
  VPTOTileDomain srcDomain = VPTOTileDomain::Vec;
  VPTOTileDomain dstDomain = VPTOTileDomain::Vec;
  StringRef srcLayout;
  StringRef dstLayout;
  Type elementType;
  Value srcValidRowsValue;
  Value srcValidColsValue;
  Value dstValidRowsValue;
  Value dstValidColsValue;
  int64_t srcValidRows = ShapedType::kDynamic;
  int64_t srcValidCols = ShapedType::kDynamic;
  int64_t dstValidRows = ShapedType::kDynamic;
  int64_t dstValidCols = ShapedType::kDynamic;
  VPTOLoopScopeContract loopScope;
};

StringRef inferVecTransferLayoutFromTile(StringRef explicitLayout,
                                         StringRef tileLayout) {
  if (explicitLayout != "nd")
    return explicitLayout;
  if (tileLayout == "col_major")
    return "dn";
  return "nd";
}

int64_t getElementByteSize(Type type);
Value materializeIndexValue(Value maybeValue, int64_t fallback,
                            PatternRewriter &rewriter, Location loc);
Value materializeI64Value(Value maybeValue, int64_t fallback,
                          PatternRewriter &rewriter, Location loc);

LogicalResult emitUnresolvedInstalledA5BaselineError(Operation *op,
                                                     StringRef family) {
  return op->emitOpError()
         << family
         << " lowering is intentionally unresolved until the installed A5 PTO "
            "helper baseline is located and traced";
}

std::optional<int64_t> getConstInt(Value value) {
  if (!value)
    return std::nullopt;

  if (auto constIndex = value.getDefiningOp<arith::ConstantIndexOp>())
    return constIndex.value();
  if (auto constInt = value.getDefiningOp<arith::ConstantIntOp>())
    return constInt.value();
  if (auto constOp = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
      return intAttr.getInt();
  }
  return std::nullopt;
}

std::optional<int64_t> getConstInt(OpFoldResult value) {
  if (auto attr = dyn_cast<Attribute>(value)) {
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      return intAttr.getInt();
    return std::nullopt;
  }
  return getConstInt(cast<Value>(value));
}

Value materializeIndexOfr(OpFoldResult value, PatternRewriter &rewriter,
                          Location loc) {
  if (auto attr = dyn_cast<Attribute>(value)) {
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      return rewriter.create<arith::ConstantIndexOp>(loc, intAttr.getInt());
    return {};
  }
  Value v = cast<Value>(value);
  if (v.getType().isIndex())
    return v;
  if (isa<IntegerType>(v.getType()))
    return rewriter.create<arith::IndexCastUIOp>(loc, rewriter.getIndexType(), v);
  return {};
}

Value materializeI64Ofr(OpFoldResult value, PatternRewriter &rewriter,
                        Location loc) {
  if (auto attr = dyn_cast<Attribute>(value)) {
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      return rewriter.create<arith::ConstantIntOp>(loc, intAttr.getInt(), 64);
    return {};
  }
  return materializeI64Value(cast<Value>(value), ShapedType::kDynamic, rewriter, loc);
}

Value materializeIndexBuilder(OpFoldResult value, PatternRewriter &rewriter, Location loc) {
  if (auto attr = dyn_cast<Attribute>(value)) {
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      return rewriter.create<arith::ConstantIndexOp>(loc, intAttr.getInt());
    return {};
  }
  Value v = cast<Value>(value);
  if (v.getType().isIndex())
    return v;
  if (isa<IntegerType>(v.getType()))
    return rewriter.create<arith::IndexCastUIOp>(loc, rewriter.getIndexType(), v);
  return {};
}

Value createI64Mul(Value lhs, Value rhs, PatternRewriter &rewriter, Location loc) {
  if (!lhs || !rhs)
    return {};
  if (std::optional<int64_t> lhsConst = getConstInt(lhs)) {
    if (std::optional<int64_t> rhsConst = getConstInt(rhs))
      return rewriter.create<arith::ConstantIntOp>(loc, (*lhsConst) * (*rhsConst), 64);
  }
  return rewriter.create<arith::MulIOp>(loc, lhs, rhs);
}

Value createI64Add(Value lhs, Value rhs, PatternRewriter &rewriter, Location loc) {
  if (!lhs || !rhs)
    return {};
  if (std::optional<int64_t> lhsConst = getConstInt(lhs)) {
    if (std::optional<int64_t> rhsConst = getConstInt(rhs))
      return rewriter.create<arith::ConstantIntOp>(loc, (*lhsConst) + (*rhsConst), 64);
  }
  return rewriter.create<arith::AddIOp>(loc, lhs, rhs);
}

OpFoldResult addOfr(OpFoldResult lhs, OpFoldResult rhs, PatternRewriter &rewriter,
                    Location loc) {
  if (auto lhsConst = getConstInt(lhs)) {
    if (auto rhsConst = getConstInt(rhs))
      return rewriter.getIndexAttr((*lhsConst) + (*rhsConst));
  }
  Value lhsValue = materializeIndexBuilder(lhs, rewriter, loc);
  Value rhsValue = materializeIndexBuilder(rhs, rewriter, loc);
  if (!lhsValue || !rhsValue)
    return {};
  return rewriter.create<arith::AddIOp>(loc, lhsValue, rhsValue).getResult();
}

OpFoldResult multiplyOfr(OpFoldResult lhs, OpFoldResult rhs, PatternRewriter &rewriter,
                         Location loc) {
  if (auto lhsConst = getConstInt(lhs)) {
    if (auto rhsConst = getConstInt(rhs))
      return rewriter.getIndexAttr((*lhsConst) * (*rhsConst));
  }
  Value lhsValue = materializeIndexBuilder(lhs, rewriter, loc);
  Value rhsValue = materializeIndexBuilder(rhs, rewriter, loc);
  if (!lhsValue || !rhsValue)
    return {};
  return rewriter.create<arith::MulIOp>(loc, lhsValue, rhsValue).getResult();
}

bool resolveTensorView(Value value, ResolvedTensorView &info, PatternRewriter &rewriter,
                       Location loc) {
  if (!value)
    return false;

  if (auto part = value.getDefiningOp<PartitionViewOp>()) {
    if (!resolveTensorView(part.getSource(), info, rewriter, loc))
      return false;
    SmallVector<OpFoldResult> offsets;
    offsets.reserve(part.getOffsets().size());
    for (Value offset : part.getOffsets())
      offsets.push_back(offset);
    if (offsets.size() != info.strides.size())
      return false;
    OpFoldResult totalOffset = info.offsetElems;
    for (auto [offset, stride] : llvm::zip(offsets, info.strides)) {
      OpFoldResult term = multiplyOfr(offset, stride, rewriter, loc);
      if (!term)
        return false;
      totalOffset = addOfr(totalOffset, term, rewriter, loc);
      if (!totalOffset)
        return false;
    }
    info.offsetElems = totalOffset;
    info.shape.clear();
    for (Value size : part.getSizes())
      info.shape.push_back(size);
    return true;
  }

  if (auto source = value.getDefiningOp<MakeTensorViewOp>()) {
    info.root = source.getPtr();
    info.layoutAttr = source.getLayoutAttr();
    info.shape.assign(source.getShape().begin(), source.getShape().end());
    info.strides.assign(source.getStrides().begin(), source.getStrides().end());
    info.offsetElems = rewriter.getIndexAttr(0);
    return true;
  }

  if (auto subview = value.getDefiningOp<memref::SubViewOp>()) {
    ResolvedTensorView parent;
    Value source = subview.getSource();
    if (auto reinterpret = source.getDefiningOp<memref::ReinterpretCastOp>()) {
      Value root = reinterpret.getSource();
      while (true) {
        if (auto cast = root.getDefiningOp<memref::CastOp>()) {
          root = cast.getSource();
          continue;
        }
        break;
      }
      parent.root = root;
      if (Attribute layout = reinterpret->getAttr("layout"))
        parent.layoutAttr = layout;
      auto parentShapes =
          getMixedValues(reinterpret.getStaticSizes(), reinterpret.getSizes(), rewriter);
      auto parentStrides =
          getMixedValues(reinterpret.getStaticStrides(), reinterpret.getStrides(), rewriter);
      auto offsets =
          getMixedValues(reinterpret.getStaticOffsets(), reinterpret.getOffsets(), rewriter);
      parent.shape.assign(parentShapes.begin(), parentShapes.end());
      parent.strides.assign(parentStrides.begin(), parentStrides.end());
      parent.offsetElems =
          offsets.empty() ? OpFoldResult(rewriter.getIndexAttr(0)) : offsets.front();
    } else if (!resolveTensorView(source, parent, rewriter, loc)) {
      return false;
    }

    if (parent.strides.empty()) {
      auto sourceType = dyn_cast<MemRefType>(source.getType());
      if (!sourceType)
        return false;
      SmallVector<int64_t> strides;
      int64_t offset = 0;
      if (failed(getStridesAndOffset(sourceType, strides, offset))) {
        strides.assign(sourceType.getRank(), 1);
        int64_t running = 1;
        for (int i = sourceType.getRank() - 1; i >= 0; --i) {
          strides[i] = running;
          int64_t dim = sourceType.getShape()[i];
          if (dim != ShapedType::kDynamic)
            running *= dim;
        }
      }
      for (int64_t stride : strides)
        parent.strides.push_back(rewriter.getIndexAttr(stride == ShapedType::kDynamic ? 1 : stride));
      parent.offsetElems = rewriter.getIndexAttr(offset);
      parent.root = source;
    }

    info = parent;
    if (subview.getMixedOffsets().size() != info.strides.size())
      return false;

    OpFoldResult totalOffset = info.offsetElems;
    for (auto [offset, stride] : llvm::zip(subview.getMixedOffsets(), info.strides)) {
      OpFoldResult term = multiplyOfr(offset, stride, rewriter, loc);
      if (!term)
        return false;
      totalOffset = addOfr(totalOffset, term, rewriter, loc);
      if (!totalOffset)
        return false;
    }

    SmallVector<OpFoldResult> newStrides;
    newStrides.reserve(info.strides.size());
    for (auto [srcStride, step] : llvm::zip(info.strides, subview.getMixedStrides())) {
      OpFoldResult product = multiplyOfr(srcStride, step, rewriter, loc);
      if (!product)
        return false;
      newStrides.push_back(product);
    }

    info.offsetElems = totalOffset;
    info.shape.assign(subview.getMixedSizes().begin(), subview.getMixedSizes().end());
    info.strides = std::move(newStrides);
    return true;
  }

  if (auto reinterpret = value.getDefiningOp<memref::ReinterpretCastOp>()) {
    Value root = reinterpret.getSource();
    while (true) {
      if (auto cast = root.getDefiningOp<memref::CastOp>()) {
        root = cast.getSource();
        continue;
      }
      if (auto unrealized = root.getDefiningOp<UnrealizedConversionCastOp>()) {
        if (!unrealized.getInputs().empty()) {
          root = unrealized.getInputs().front();
          continue;
        }
      }
      break;
    }
    info.root = root;
    if (Attribute layout = reinterpret->getAttr("layout"))
      info.layoutAttr = layout;
    auto reinterpretShapes =
        getMixedValues(reinterpret.getStaticSizes(), reinterpret.getSizes(), rewriter);
    auto reinterpretStrides =
        getMixedValues(reinterpret.getStaticStrides(), reinterpret.getStrides(), rewriter);
    auto offsets =
        getMixedValues(reinterpret.getStaticOffsets(), reinterpret.getOffsets(), rewriter);
    info.shape.assign(reinterpretShapes.begin(), reinterpretShapes.end());
    info.strides.assign(reinterpretStrides.begin(), reinterpretStrides.end());
    if (!offsets.empty()) {
      if (offsets.size() != 1)
        return false;
      info.offsetElems = offsets.front();
    } else {
      info.offsetElems = rewriter.getIndexAttr(0);
    }
    return true;
  }

  if (auto cast = value.getDefiningOp<memref::CastOp>())
    return resolveTensorView(cast.getSource(), info, rewriter, loc);

  if (auto memrefType = dyn_cast<MemRefType>(value.getType())) {
    info.root = value;
    info.shape.clear();
    for (int64_t dim : memrefType.getShape())
      info.shape.push_back(rewriter.getIndexAttr(dim == ShapedType::kDynamic ? 1 : dim));
    SmallVector<int64_t> strides;
    int64_t offset = 0;
    if (failed(getStridesAndOffset(memrefType, strides, offset))) {
      strides.assign(memrefType.getRank(), 1);
      int64_t running = 1;
      for (int i = memrefType.getRank() - 1; i >= 0; --i) {
        strides[i] = running;
        int64_t dim = memrefType.getShape()[i];
        if (dim != ShapedType::kDynamic)
          running *= dim;
      }
      offset = 0;
    }
    info.strides.clear();
    for (int64_t stride : strides)
      info.strides.push_back(rewriter.getIndexAttr(stride == ShapedType::kDynamic ? 1 : stride));
    info.offsetElems = rewriter.getIndexAttr(offset);
    return true;
  }

  return false;
}

void normalizeMixedGlobalShapeAndStride(ArrayRef<OpFoldResult> shape,
                                        ArrayRef<OpFoldResult> strides,
                                        SmallVectorImpl<OpFoldResult> &globalShape,
                                        SmallVectorImpl<OpFoldResult> &globalStride,
                                        PatternRewriter &rewriter, Location loc) {
  constexpr int64_t kRank = 5;
  globalShape.assign(kRank, rewriter.getIndexAttr(1));
  globalStride.assign(kRank, rewriter.getIndexAttr(1));

  size_t rank = std::min(shape.size(), strides.size());
  rank = std::min<size_t>(rank, kRank);
  size_t base = kRank - rank;
  for (size_t i = 0; i < rank; ++i) {
    globalShape[base + i] = shape[shape.size() - rank + i];
    globalStride[base + i] = strides[strides.size() - rank + i];
  }

  for (int i = static_cast<int>(kRank) - 2; i >= 0; --i) {
    if (i >= static_cast<int>(base))
      continue;
    OpFoldResult product = multiplyOfr(globalStride[i + 1], globalShape[i + 1], rewriter, loc);
    if (!product)
      product = rewriter.getIndexAttr(ShapedType::kDynamic);
    globalStride[i] = product;
  }
}

Value adjustPointerByElemOffset(Value ptr, Value elemOffsetI64, int64_t elemBytes,
                                PatternRewriter &rewriter, Location loc) {
  if (!ptr || !elemOffsetI64 || elemBytes <= 0)
    return {};

  Value offset = elemOffsetI64.getType().isIndex()
                     ? rewriter.create<arith::IndexCastUIOp>(
                           loc, rewriter.getI64Type(), elemOffsetI64)
                     : elemOffsetI64;
  Value byteOffset = offset;
  if (elemBytes != 1) {
    Value elemBytesValue = rewriter.create<arith::ConstantIntOp>(loc, elemBytes, 64);
    byteOffset = createI64Mul(offset, elemBytesValue, rewriter, loc);
  }
  if (auto ptrType = dyn_cast<PtrType>(ptr.getType())) {
    auto bytePtrType = PtrType::get(rewriter.getContext(), rewriter.getI8Type(),
                                    ptrType.getMemorySpace());
    Value bytePtr = ptrType == bytePtrType
                        ? ptr
                        : rewriter.create<CastPtrOp>(loc, bytePtrType, ptr).getResult();
    Value byteOffsetIndex =
        byteOffset.getType().isIndex()
            ? byteOffset
            : rewriter.create<arith::IndexCastUIOp>(loc, rewriter.getIndexType(),
                                                    byteOffset);
    return rewriter.create<AddPtrOp>(loc, bytePtrType, bytePtr, byteOffsetIndex);
  }
  return {};
}

Value castPtrToElementType(Value ptr, Type elementType, PatternRewriter &rewriter,
                           Location loc) {
  auto ptrType = dyn_cast_or_null<PtrType>(ptr.getType());
  if (!ptrType || !elementType)
    return {};
  auto targetType =
      PtrType::get(rewriter.getContext(), elementType, ptrType.getMemorySpace());
  if (targetType == ptrType)
    return ptr;
  return rewriter.create<CastPtrOp>(loc, targetType, ptr).getResult();
}

Type getCopyTransferElementType(Type elementType, Builder &builder) {
  if (getElementByteSize(elementType) == 8)
    return builder.getI32Type();
  return elementType;
}

LogicalResult buildVecNdLoadPlan(ArrayRef<OpFoldResult> shape,
                                 ArrayRef<OpFoldResult> strides, int64_t tileCols,
                                 Value validColsValue, int64_t validCols,
                                 Type elementType, PatternRewriter &rewriter,
                                 Location loc, VecNdTransferPlan &plan) {
  if (tileCols == ShapedType::kDynamic)
    return failure();
  int64_t elemBytes = getElementByteSize(elementType);
  if (elemBytes <= 0)
    return failure();

  SmallVector<OpFoldResult> globalShape;
  SmallVector<OpFoldResult> globalStride;
  normalizeMixedGlobalShapeAndStride(shape, strides, globalShape, globalStride, rewriter, loc);

  auto toI64 = [&](OpFoldResult ofr) { return materializeI64Ofr(ofr, rewriter, loc); };
  Value gShape0 = toI64(globalShape[0]);
  Value gShape1 = toI64(globalShape[1]);
  Value gShape2 = toI64(globalShape[2]);
  Value gShape3 = toI64(globalShape[3]);
  Value gStride0 = toI64(globalStride[0]);
  Value gStride1 = toI64(globalStride[1]);
  Value gStride2 = toI64(globalStride[2]);
  Value gStride3 = toI64(globalStride[3]);
  Value validColsI64 = materializeI64Value(validColsValue, validCols, rewriter, loc);
  if (!gShape0 || !gShape1 || !gShape2 || !gShape3 || !gStride0 || !gStride1 ||
      !gStride2 || !gStride3 || !validColsI64)
    return failure();

  Value tileColsI64 = rewriter.create<arith::ConstantIntOp>(loc, tileCols, 64);
  Value elemBytesI64 = rewriter.create<arith::ConstantIntOp>(loc, elemBytes, 64);
  Value dstStride2 = createI64Mul(gShape3, tileColsI64, rewriter, loc);
  Value dstStride1 = createI64Mul(gShape2, dstStride2, rewriter, loc);
  Value dstStride0 = createI64Mul(gShape1, dstStride1, rewriter, loc);

  plan.outerCount = gShape0;
  plan.outerSrcStrideElems = gStride0;
  plan.outerDstStrideElems = dstStride0;
  plan.loop2Size = gShape1;
  plan.loop1Size = gShape2;
  plan.loop2FirstStrideBytes = createI64Mul(dstStride1, elemBytesI64, rewriter, loc);
  plan.loop2SecondStrideBytes = createI64Mul(gStride1, elemBytesI64, rewriter, loc);
  plan.loop1FirstStrideBytes = createI64Mul(dstStride2, elemBytesI64, rewriter, loc);
  plan.loop1SecondStrideBytes = createI64Mul(gStride2, elemBytesI64, rewriter, loc);
  plan.nBurst = gShape3;
  plan.lenBurst = createI64Mul(validColsI64, elemBytesI64, rewriter, loc);
  plan.firstStrideBytes = createI64Mul(gStride3, elemBytesI64, rewriter, loc);
  plan.secondStrideBytes = createI64Mul(tileColsI64, elemBytesI64, rewriter, loc);
  return success();
}

LogicalResult buildVecDnLoadPlan(ArrayRef<OpFoldResult> shape,
                                 ArrayRef<OpFoldResult> strides, int64_t tileRows,
                                 Value validRowsValue, int64_t validRows,
                                 Type elementType, PatternRewriter &rewriter,
                                 Location loc, VecNdTransferPlan &plan) {
  if (tileRows == ShapedType::kDynamic)
    return failure();
  int64_t elemBytes = getElementByteSize(elementType);
  if (elemBytes <= 0)
    return failure();

  SmallVector<OpFoldResult> globalShape;
  SmallVector<OpFoldResult> globalStride;
  normalizeMixedGlobalShapeAndStride(shape, strides, globalShape, globalStride,
                                     rewriter, loc);

  auto toI64 = [&](OpFoldResult ofr) { return materializeI64Ofr(ofr, rewriter, loc); };
  Value gShape0 = toI64(globalShape[0]);
  Value gShape1 = toI64(globalShape[1]);
  Value gShape2 = toI64(globalShape[2]);
  Value gShape4 = toI64(globalShape[4]);
  Value gStride0 = toI64(globalStride[0]);
  Value gStride1 = toI64(globalStride[1]);
  Value gStride2 = toI64(globalStride[2]);
  Value gStride4 = toI64(globalStride[4]);
  Value validRowsI64 = materializeI64Value(validRowsValue, validRows, rewriter, loc);
  if (!gShape0 || !gShape1 || !gShape2 || !gShape4 || !gStride0 || !gStride1 ||
      !gStride2 || !gStride4 || !validRowsI64)
    return failure();

  Value tileRowsI64 = rewriter.create<arith::ConstantIntOp>(loc, tileRows, 64);
  Value elemBytesI64 = rewriter.create<arith::ConstantIntOp>(loc, elemBytes, 64);
  Value dstStride2 = createI64Mul(gShape4, tileRowsI64, rewriter, loc);
  Value dstStride1 = createI64Mul(gShape2, dstStride2, rewriter, loc);
  Value dstStride0 = createI64Mul(gShape1, dstStride1, rewriter, loc);

  plan.outerCount = gShape0;
  plan.outerSrcStrideElems = gStride0;
  plan.outerDstStrideElems = dstStride0;
  plan.loop2Size = gShape1;
  plan.loop1Size = gShape2;
  plan.loop2FirstStrideBytes = createI64Mul(dstStride1, elemBytesI64, rewriter, loc);
  plan.loop2SecondStrideBytes = createI64Mul(gStride1, elemBytesI64, rewriter, loc);
  plan.loop1FirstStrideBytes = createI64Mul(dstStride2, elemBytesI64, rewriter, loc);
  plan.loop1SecondStrideBytes = createI64Mul(gStride2, elemBytesI64, rewriter, loc);
  plan.nBurst = gShape4;
  plan.lenBurst = createI64Mul(validRowsI64, elemBytesI64, rewriter, loc);
  plan.firstStrideBytes = createI64Mul(gStride4, elemBytesI64, rewriter, loc);
  plan.secondStrideBytes = createI64Mul(tileRowsI64, elemBytesI64, rewriter, loc);
  return success();
}

LogicalResult buildVecNdStorePlan(ArrayRef<OpFoldResult> shape,
                                  ArrayRef<OpFoldResult> strides, int64_t tileCols,
                                  Value validColsValue, int64_t validCols,
                                  Type elementType, PatternRewriter &rewriter,
                                  Location loc, VecNdTransferPlan &plan) {
  if (failed(buildVecNdLoadPlan(shape, strides, tileCols, validColsValue, validCols,
                                elementType, rewriter, loc, plan)))
    return failure();
  std::swap(plan.outerSrcStrideElems, plan.outerDstStrideElems);
  std::swap(plan.loop2FirstStrideBytes, plan.loop2SecondStrideBytes);
  std::swap(plan.loop1FirstStrideBytes, plan.loop1SecondStrideBytes);
  return success();
}

LogicalResult buildVecDnStorePlan(ArrayRef<OpFoldResult> shape,
                                  ArrayRef<OpFoldResult> strides, int64_t tileRows,
                                  Value validRowsValue, int64_t validRows,
                                  Type elementType, PatternRewriter &rewriter,
                                  Location loc, VecNdTransferPlan &plan) {
  if (tileRows == ShapedType::kDynamic)
    return failure();
  int64_t elemBytes = getElementByteSize(elementType);
  if (elemBytes <= 0)
    return failure();

  SmallVector<OpFoldResult> globalShape;
  SmallVector<OpFoldResult> globalStride;
  normalizeMixedGlobalShapeAndStride(shape, strides, globalShape, globalStride,
                                     rewriter, loc);

  auto toI64 = [&](OpFoldResult ofr) { return materializeI64Ofr(ofr, rewriter, loc); };
  Value gShape0 = toI64(globalShape[0]);
  Value gShape1 = toI64(globalShape[1]);
  Value gShape2 = toI64(globalShape[2]);
  Value gShape4 = toI64(globalShape[4]);
  Value gStride0 = toI64(globalStride[0]);
  Value gStride1 = toI64(globalStride[1]);
  Value gStride2 = toI64(globalStride[2]);
  Value gStride4 = toI64(globalStride[4]);
  Value validRowsI64 = materializeI64Value(validRowsValue, validRows, rewriter, loc);
  if (!gShape0 || !gShape1 || !gShape2 || !gShape4 || !gStride0 || !gStride1 ||
      !gStride2 || !gStride4 || !validRowsI64)
    return failure();

  Value tileRowsI64 = rewriter.create<arith::ConstantIntOp>(loc, tileRows, 64);
  Value elemBytesI64 = rewriter.create<arith::ConstantIntOp>(loc, elemBytes, 64);
  Value outerSrcStride =
      createI64Mul(createI64Mul(createI64Mul(gShape1, gShape2, rewriter, loc),
                                gShape4, rewriter, loc),
                   tileRowsI64, rewriter, loc);
  Value loop1SrcStride =
      createI64Mul(createI64Mul(tileRowsI64, gShape4, rewriter, loc), elemBytesI64,
                   rewriter, loc);
  Value loop2SrcStride =
      createI64Mul(createI64Mul(createI64Mul(gShape2, tileRowsI64, rewriter, loc),
                                gShape4, rewriter, loc),
                   elemBytesI64, rewriter, loc);

  plan.outerCount = gShape0;
  plan.outerSrcStrideElems = outerSrcStride;
  plan.outerDstStrideElems = gStride0;
  plan.loop2Size = gShape1;
  plan.loop1Size = gShape2;
  plan.loop2FirstStrideBytes = loop2SrcStride;
  plan.loop2SecondStrideBytes = createI64Mul(gStride1, elemBytesI64, rewriter, loc);
  plan.loop1FirstStrideBytes = loop1SrcStride;
  plan.loop1SecondStrideBytes = createI64Mul(gStride2, elemBytesI64, rewriter, loc);
  plan.nBurst = gShape4;
  plan.lenBurst = createI64Mul(validRowsI64, elemBytesI64, rewriter, loc);
  plan.firstStrideBytes = createI64Mul(gStride4, elemBytesI64, rewriter, loc);
  plan.secondStrideBytes = createI64Mul(tileRowsI64, elemBytesI64, rewriter, loc);
  return success();
}

StringRef stringifyTileLayout(TileBufType type) {
  if (auto layoutAttr = dyn_cast_or_null<BLayoutAttr>(type.getBLayoutAttr())) {
    switch (layoutAttr.getValue()) {
    case BLayout::RowMajor:
      return "row_major";
    case BLayout::ColMajor:
      return "col_major";
    }
  }
  return "row_major";
}

StringRef stringifyTileLayoutConfig(TileBufConfigAttr config) {
  if (!config)
    return "row_major";
  if (auto layoutAttr = dyn_cast_or_null<BLayoutAttr>(config.getBLayout())) {
    switch (layoutAttr.getValue()) {
    case BLayout::RowMajor:
      return "row_major";
    case BLayout::ColMajor:
      return "col_major";
    }
  }
  return "row_major";
}

StringRef stringifyPadModeAttr(PadModeAttr padMode) {
  if (!padMode)
    return "none";

  switch (padMode.getPadmode()) {
  case PadMode::PadNull:
    return "none";
  case PadMode::PadFirstElem:
    return "first_elem";
  case PadMode::PadValue:
    return "value";
  }
  return "none";
}

StringRef stringifyLayoutAttr(Attribute layoutAttr) {
  if (auto attr = dyn_cast_or_null<LayoutAttr>(layoutAttr))
    return stringifyLayout(attr.getLayout());
  return "nd";
}

PipeAttr stringifyPipeAttr(PipeAttr pipe, PatternRewriter &rewriter) {
  return PipeAttr::get(rewriter.getContext(), pipe.getPipe());
}

EventAttr stringifyEventAttr(EventAttr event, PatternRewriter &rewriter) {
  return EventAttr::get(rewriter.getContext(), event.getEvent());
}

StringRef stringifyCmpModeAttr(CmpModeAttr cmpMode) {
  if (!cmpMode)
    return "eq";
  switch (cmpMode.getValue()) {
  case CmpMode::EQ:
    return "eq";
  case CmpMode::NE:
    return "ne";
  case CmpMode::LT:
    return "lt";
  case CmpMode::LE:
    return "le";
  case CmpMode::GT:
    return "gt";
  case CmpMode::GE:
    return "ge";
  }
  return "eq";
}

StringRef stringifyElementTypeFragment(Type type) {
  if (!type)
    return "unknown";
  if (type.isF16())
    return "f16";
  if (type.isBF16())
    return "bf16";
  if (type.isF32())
    return "f32";
  if (auto intType = dyn_cast<IntegerType>(type)) {
    if (intType.isUnsigned())
      switch (intType.getWidth()) {
      case 8:
        return "u8";
      case 16:
        return "u16";
      case 32:
        return "u32";
      case 64:
        return "u64";
      default:
        break;
      }
    switch (intType.getWidth()) {
    case 8:
      return "s8";
    case 16:
      return "s16";
    case 32:
      return "s32";
    case 64:
      return "s64";
    default:
      break;
    }
  }
  return "unknown";
}

StringRef stringifyCopyTransferTypeFragment(Type type) {
  switch (getElementByteSize(type)) {
  case 1:
    return "u8";
  case 2:
    return "u16";
  case 4:
  case 8:
    return "u32";
  default:
    return stringifyElementTypeFragment(type);
  }
}

static bool isSupportedPackedCmp32ElementType(Type type) {
  if (!type)
    return false;
  if (type.isF32())
    return true;
  auto intType = dyn_cast<IntegerType>(type);
  return intType && intType.getWidth() == 32;
}

VPTOTileDomain deriveTileDomain(Attribute memorySpace) {
  if (auto addrSpace = dyn_cast_or_null<AddressSpaceAttr>(memorySpace)) {
    switch (addrSpace.getAddressSpace()) {
    case AddressSpace::ACC:
      return VPTOTileDomain::Acc;
    case AddressSpace::MAT:
      return VPTOTileDomain::Mat;
    case AddressSpace::VEC:
    default:
      return VPTOTileDomain::Vec;
    }
  }
  if (auto intAttr = dyn_cast_or_null<IntegerAttr>(memorySpace)) {
    switch (intAttr.getInt()) {
    case static_cast<int64_t>(AddressSpace::ACC):
      return VPTOTileDomain::Acc;
    case static_cast<int64_t>(AddressSpace::MAT):
      return VPTOTileDomain::Mat;
    default:
      return VPTOTileDomain::Vec;
    }
  }
  return VPTOTileDomain::Vec;
}

void getValidShape(TileBufType type, int64_t &rows, int64_t &cols) {
  ArrayRef<int64_t> validShape = type.getValidShape();
  rows = validShape.size() > 0 ? validShape[0] : ShapedType::kDynamic;
  cols = validShape.size() > 1 ? validShape[1] : ShapedType::kDynamic;
}

static std::pair<Value, Value> getIfResultYieldedValues(Value value) {
  auto result = dyn_cast<OpResult>(value);
  if (!result)
    return {Value(), Value()};
  auto ifOp = dyn_cast<scf::IfOp>(result.getOwner());
  if (!ifOp)
    return {Value(), Value()};
  unsigned resultNumber = result.getResultNumber();
  auto thenYield = dyn_cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
  auto elseYield = dyn_cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());
  if (!thenYield || !elseYield)
    return {Value(), Value()};
  if (resultNumber >= thenYield.getNumOperands() ||
      resultNumber >= elseYield.getNumOperands())
    return {Value(), Value()};
  return {thenYield.getOperand(resultNumber), elseYield.getOperand(resultNumber)};
}

static bool equalOrBothNull(Value lhs, Value rhs) {
  if (!lhs && !rhs)
    return true;
  if (!lhs || !rhs)
    return false;
  if (lhs == rhs)
    return true;
  auto lhsConst = getConstInt(lhs);
  auto rhsConst = getConstInt(rhs);
  return lhsConst && rhsConst && *lhsConst == *rhsConst;
}

TileBufConfigAttr lookupTileConfig(Value value) {
  if (!value)
    return {};
  if (auto bind = value.getDefiningOp<BindTileOp>())
    return bind.getConfig();
  if (auto cast = value.getDefiningOp<PointerCastOp>())
    return cast.getConfig().value_or(TileBufConfigAttr{});
  if (auto subview = value.getDefiningOp<memref::SubViewOp>())
    return lookupTileConfig(subview.getSource());
  if (auto reinterpret = value.getDefiningOp<memref::ReinterpretCastOp>())
    return lookupTileConfig(reinterpret.getSource());
  if (auto cast = value.getDefiningOp<memref::CastOp>())
    return lookupTileConfig(cast.getSource());
  if (auto [thenValue, elseValue] = getIfResultYieldedValues(value);
      thenValue && elseValue) {
    TileBufConfigAttr thenConfig = lookupTileConfig(thenValue);
    TileBufConfigAttr elseConfig = lookupTileConfig(elseValue);
    if (thenConfig && elseConfig && thenConfig == elseConfig)
      return thenConfig;
  }
  return {};
}

bool hasStructuredTileDriver(Value value) {
  if (!value)
    return false;
  if (isa<TileBufType>(value.getType()))
    return true;
  if (value.getDefiningOp<BindTileOp>())
    return true;
  if (auto subview = value.getDefiningOp<memref::SubViewOp>())
    return hasStructuredTileDriver(subview.getSource());
  if (auto reinterpret = value.getDefiningOp<memref::ReinterpretCastOp>())
    return hasStructuredTileDriver(reinterpret.getSource());
  if (auto cast = value.getDefiningOp<memref::CastOp>())
    return hasStructuredTileDriver(cast.getSource());
  if (auto [thenValue, elseValue] = getIfResultYieldedValues(value);
      thenValue && elseValue) {
    return hasStructuredTileDriver(thenValue) && hasStructuredTileDriver(elseValue);
  }
  return false;
}

void lookupValidDims(Value value, Value &validRow, Value &validCol) {
  if (!value) {
    validRow = {};
    validCol = {};
    return;
  }
  if (auto bind = value.getDefiningOp<BindTileOp>()) {
    validRow = bind.getValidRow();
    validCol = bind.getValidCol();
    return;
  }
  if (auto cast = value.getDefiningOp<PointerCastOp>()) {
    validRow = cast.getValidRow();
    validCol = cast.getValidCol();
    return;
  }
  if (auto subview = value.getDefiningOp<memref::SubViewOp>()) {
    lookupValidDims(subview.getSource(), validRow, validCol);
    return;
  }
  if (auto reinterpret = value.getDefiningOp<memref::ReinterpretCastOp>()) {
    lookupValidDims(reinterpret.getSource(), validRow, validCol);
    return;
  }
  if (auto cast = value.getDefiningOp<memref::CastOp>()) {
    lookupValidDims(cast.getSource(), validRow, validCol);
    return;
  }
  if (auto [thenValue, elseValue] = getIfResultYieldedValues(value);
      thenValue && elseValue) {
    Value thenRow;
    Value thenCol;
    Value elseRow;
    Value elseCol;
    lookupValidDims(thenValue, thenRow, thenCol);
    lookupValidDims(elseValue, elseRow, elseCol);
    validRow = equalOrBothNull(thenRow, elseRow) ? thenRow : Value();
    validCol = equalOrBothNull(thenCol, elseCol) ? thenCol : Value();
    return;
  }
  validRow = {};
  validCol = {};
}

Type getElementType(Value value) {
  Type type = value.getType();
  if (auto tileType = dyn_cast<TileBufType>(type))
    return tileType.getElementType();
  if (auto memrefType = dyn_cast<MemRefType>(type))
    return memrefType.getElementType();
  if (auto ptrType = dyn_cast<PtrType>(type))
    return ptrType.getElementType();
  return {};
}

Attribute getMemorySpace(Value value) {
  Type type = value.getType();
  if (auto tileType = dyn_cast<TileBufType>(type))
    return tileType.getMemorySpace();
  if (auto memrefType = dyn_cast<MemRefType>(type))
    return memrefType.getMemorySpace();
  if (auto ptrType = dyn_cast<PtrType>(type))
    return ptrType.getMemorySpace();
  return {};
}

StringRef deriveTileLayout(Value value) {
  if (auto tileType = dyn_cast<TileBufType>(value.getType()))
    return stringifyTileLayout(tileType);
  return stringifyTileLayoutConfig(lookupTileConfig(value));
}

void deriveValidShape(Value value, int64_t &rows, int64_t &cols) {
  if (auto tileType = dyn_cast<TileBufType>(value.getType())) {
    getValidShape(tileType, rows, cols);
    return;
  }

  Value validRow;
  Value validCol;
  lookupValidDims(value, validRow, validCol);
  rows = getConstInt(validRow).value_or(ShapedType::kDynamic);
  cols = getConstInt(validCol).value_or(ShapedType::kDynamic);
  if (rows != ShapedType::kDynamic && cols != ShapedType::kDynamic)
    return;
  if (!hasStructuredTileDriver(value))
    return;

  auto shapedType = dyn_cast<ShapedType>(value.getType());
  if (!shapedType || !shapedType.hasRank())
    return;

  ArrayRef<int64_t> shape = shapedType.getShape();
  if (shape.empty()) {
    if (rows == ShapedType::kDynamic)
      rows = 1;
    if (cols == ShapedType::kDynamic)
      cols = 1;
    return;
  }
  if (shape.size() == 1) {
    if (rows == ShapedType::kDynamic)
      rows = 1;
    if (cols == ShapedType::kDynamic)
      cols = shape.front();
    return;
  }

  if (cols == ShapedType::kDynamic)
    cols = shape.back();
  if (rows == ShapedType::kDynamic) {
    int64_t flatRows = 1;
    for (int64_t dim : shape.drop_back()) {
      if (dim == ShapedType::kDynamic) {
        flatRows = ShapedType::kDynamic;
        break;
      }
      flatRows *= dim;
    }
    rows = flatRows;
  }
}

void deriveValidShapeValues(Value value, Value &rows, Value &cols) {
  if (auto tileType = dyn_cast<TileBufType>(value.getType())) {
    ArrayRef<int64_t> validShape = tileType.getValidShape();
    rows = {};
    cols = {};
    (void)validShape;
    lookupValidDims(value, rows, cols);
    return;
  }
  lookupValidDims(value, rows, cols);
}

void appendStaticSizes(ValueRange values, SmallVectorImpl<int64_t> &out,
                       bool &hasDynamic) {
  out.clear();
  hasDynamic = false;
  out.reserve(values.size());
  for (Value value : values) {
    if (std::optional<int64_t> constant = getConstInt(value)) {
      out.push_back(*constant);
      continue;
    }
    out.push_back(ShapedType::kDynamic);
    hasDynamic = true;
  }
}

int64_t getElementByteSize(Type type) {
  if (auto floatType = dyn_cast<FloatType>(type))
    return (floatType.getWidth() + 7) / 8;
  if (auto intType = dyn_cast<IntegerType>(type))
    return (intType.getWidth() + 7) / 8;
  return 0;
}

Value materializeIndexValue(Value maybeValue, int64_t fallback,
                            PatternRewriter &rewriter, Location loc) {
  if (maybeValue)
    return maybeValue;
  if (fallback != ShapedType::kDynamic)
    return rewriter.create<arith::ConstantIndexOp>(loc, fallback);
  return {};
}

Value materializeI64Value(Value maybeValue, int64_t fallback,
                          PatternRewriter &rewriter, Location loc) {
  if (maybeValue) {
    Type type = maybeValue.getType();
    if (type.isIndex())
      return rewriter.create<arith::IndexCastUIOp>(loc, rewriter.getI64Type(), maybeValue);
    if (type.isInteger(64))
      return maybeValue;
    if (auto intType = dyn_cast<IntegerType>(type))
      return rewriter.create<arith::ExtUIOp>(loc, rewriter.getI64Type(), maybeValue);
  }
  if (fallback != ShapedType::kDynamic)
    return rewriter.create<arith::ConstantIntOp>(loc, fallback, 64);
  return {};
}

void recordStaticValues(ValueRange values, SmallVectorImpl<int64_t> &out) {
  out.clear();
  out.reserve(values.size());
  for (Value value : values)
    out.push_back(getConstInt(value).value_or(ShapedType::kDynamic));
}

void recordStaticSizes(ArrayRef<OpFoldResult> values,
                       SmallVectorImpl<int64_t> &out, bool &hasDynamic) {
  out.clear();
  hasDynamic = false;
  out.reserve(values.size());
  for (OpFoldResult value : values) {
    if (auto attr = dyn_cast<Attribute>(value)) {
      if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
        out.push_back(intAttr.getInt());
        continue;
      }
    } else if (std::optional<int64_t> constant =
                   getConstInt(cast<Value>(value))) {
      out.push_back(*constant);
      continue;
    }
    out.push_back(ShapedType::kDynamic);
    hasDynamic = true;
  }
}

void mergeSubviewTrace(VPTOPartitionTrace &trace, ArrayRef<int64_t> offsets,
                       ArrayRef<int64_t> sizes, bool hasDynamicOffsets,
                       bool hasDynamicSizes) {
  if (trace.offsets.empty()) {
    trace.offsets.assign(offsets.begin(), offsets.end());
    trace.hasDynamicOffsets = hasDynamicOffsets;
  } else {
    size_t count = std::min(trace.offsets.size(), offsets.size());
    for (size_t i = 0; i < count; ++i) {
      if (trace.offsets[i] == ShapedType::kDynamic ||
          offsets[i] == ShapedType::kDynamic) {
        trace.offsets[i] = ShapedType::kDynamic;
        trace.hasDynamicOffsets = true;
        continue;
      }
      trace.offsets[i] += offsets[i];
    }
    trace.hasDynamicOffsets = trace.hasDynamicOffsets || hasDynamicOffsets;
  }

  trace.sizes.assign(sizes.begin(), sizes.end());
  trace.hasDynamicSizes = hasDynamicSizes;
}

Value resolveTensorViewBase(Value value, Attribute &layoutAttr,
                            SmallVectorImpl<int64_t> &shape,
                            SmallVectorImpl<int64_t> &strides) {
  if (!value)
    return {};

  if (auto part = value.getDefiningOp<PartitionViewOp>()) {
    return resolveTensorViewBase(part.getSource(), layoutAttr, shape, strides);
  }

  if (auto source = value.getDefiningOp<MakeTensorViewOp>()) {
    layoutAttr = source.getLayoutAttr();
    auto tensorType = dyn_cast<TensorViewType>(source.getResult().getType());
    shape.assign(tensorType.getShape().begin(), tensorType.getShape().end());
    recordStaticValues(source.getStrides(), strides);
    return source.getPtr();
  }

  if (auto subview = value.getDefiningOp<memref::SubViewOp>()) {
    Value base =
        resolveTensorViewBase(subview.getSource(), layoutAttr, shape, strides);
    if (shape.empty()) {
      bool hasDynamicSizes = false;
      recordStaticSizes(subview.getMixedSizes(), shape, hasDynamicSizes);
    }
    return base ? base : value;
  }

  if (auto reinterpret = value.getDefiningOp<memref::ReinterpretCastOp>()) {
    if (Attribute layout = reinterpret->getAttr("layout"))
      layoutAttr = layout;
    if (shape.empty()) {
      bool hasDynamicSizes = false;
      recordStaticSizes(reinterpret.getMixedSizes(), shape, hasDynamicSizes);
    }
    if (strides.empty()) {
      bool hasDynamicStrides = false;
      recordStaticSizes(reinterpret.getMixedStrides(), strides,
                        hasDynamicStrides);
    }
    Value base =
        resolveTensorViewBase(reinterpret.getSource(), layoutAttr, shape, strides);
    return base ? base : value;
  }

  if (auto cast = value.getDefiningOp<memref::CastOp>()) {
    Value base =
        resolveTensorViewBase(cast.getSource(), layoutAttr, shape, strides);
    return base ? base : value;
  }

  if (auto memrefType = dyn_cast<MemRefType>(value.getType())) {
    if (shape.empty())
      shape.assign(memrefType.getShape().begin(), memrefType.getShape().end());
    if (strides.empty()) {
      int64_t offset = 0;
      if (failed(mlir::getStridesAndOffset(memrefType, strides, offset)))
        strides.assign(shape.size(), ShapedType::kDynamic);
    }
    return value;
  }

  return {};
}

pto::VRegType getVPTOVRegType(MLIRContext *context, Type elementType) {
  unsigned bitWidth = 0;
  if (auto floatType = dyn_cast<FloatType>(elementType))
    bitWidth = floatType.getWidth();
  else if (auto intType = dyn_cast<IntegerType>(elementType))
    bitWidth = intType.getWidth();

  if (bitWidth == 0 || 2048 % bitWidth != 0)
    return {};
  return pto::VRegType::get(context, 2048 / bitWidth, elementType);
}

pto::MaskType getVPTOMaskType(MLIRContext *context, StringRef granularity) {
  return pto::MaskType::get(context, granularity);
}

pto::MaskType getVPTOMaskTypeForElementType(MLIRContext *context,
                                            Type elementType) {
  unsigned bitWidth = 0;
  if (auto floatType = dyn_cast<FloatType>(elementType))
    bitWidth = floatType.getWidth();
  else if (auto intType = dyn_cast<IntegerType>(elementType))
    bitWidth = intType.getWidth();

  switch (bitWidth) {
  case 8:
    return getVPTOMaskType(context, "b8");
  case 16:
    return getVPTOMaskType(context, "b16");
  case 32:
    return getVPTOMaskType(context, "b32");
  default:
    return {};
  }
}

ArrayAttr asI64ArrayAttr(Builder &builder, ArrayRef<int64_t> values) {
  SmallVector<Attribute> attrs;
  attrs.reserve(values.size());
  for (int64_t value : values)
    attrs.push_back(builder.getI64IntegerAttr(value));
  return builder.getArrayAttr(attrs);
}

void normalizeToPTOGlobalShapeAndStride(ArrayRef<int64_t> shape,
                                        ArrayRef<int64_t> strides,
                                        SmallVectorImpl<int64_t> &globalShape,
                                        SmallVectorImpl<int64_t> &globalStride) {
  constexpr int64_t kRank = 5;
  globalShape.assign(kRank, 1);
  globalStride.assign(kRank, 1);

  size_t shapeRank = std::min<size_t>(shape.size(), kRank);
  size_t strideRank = std::min<size_t>(strides.size(), kRank);
  size_t rank = std::min(shapeRank, strideRank);
  size_t base = kRank - rank;

  for (size_t i = 0; i < rank; ++i) {
    globalShape[base + i] = shape[shape.size() - rank + i];
    globalStride[base + i] = strides[strides.size() - rank + i];
  }

  for (int i = static_cast<int>(kRank) - 2; i >= 0; --i) {
    if (i >= static_cast<int>(base))
      continue;
    if (globalStride[i + 1] == ShapedType::kDynamic ||
        globalShape[i + 1] == ShapedType::kDynamic) {
      globalStride[i] = ShapedType::kDynamic;
      continue;
    }
    globalStride[i] = globalStride[i + 1] * globalShape[i + 1];
  }
}

int64_t packLoopStrideConfig(int64_t first, int64_t second) {
  return (static_cast<int64_t>(first) << 40) | static_cast<int64_t>(second);
}

int64_t packLoopSizeConfig(int64_t loop2, int64_t loop1) {
  return (static_cast<int64_t>(loop2) << 21) | static_cast<int64_t>(loop1);
}

LogicalResult deriveVecNDTransferConfig(ArrayRef<int64_t> shape,
                                        ArrayRef<int64_t> strides,
                                        StringRef tileLayout, Type elementType,
                                        int64_t validRows, int64_t validCols,
                                        SmallVectorImpl<int64_t> &globalShape,
                                        SmallVectorImpl<int64_t> &globalStride,
                                        int64_t &nBurst, int64_t &lenBurst,
                                        int64_t &gmStrideBytes,
                                        int64_t &ubStrideBytes,
                                        int64_t &loop1Size,
                                        int64_t &loop2Size,
                                        int64_t &loop1FirstStrideBytes,
                                        int64_t &loop1SecondStrideBytes,
                                        int64_t &loop2FirstStrideBytes,
                                        int64_t &loop2SecondStrideBytes) {
  if (tileLayout != "row_major")
    return failure();

  int64_t elemBytes = getElementByteSize(elementType);
  if (elemBytes <= 0)
    return failure();

  normalizeToPTOGlobalShapeAndStride(shape, strides, globalShape, globalStride);
  if (globalShape.size() != 5 || globalStride.size() != 5)
    return failure();
  if (llvm::any_of(globalShape, [](int64_t v) { return v == ShapedType::kDynamic; }) ||
      llvm::any_of(globalStride, [](int64_t v) { return v == ShapedType::kDynamic; }))
    return failure();
  nBurst = globalShape[3];
  lenBurst = (validCols == ShapedType::kDynamic) ? ShapedType::kDynamic
                                                 : validCols * elemBytes;
  gmStrideBytes = globalStride[3] * elemBytes;
  ubStrideBytes = globalShape[4] * elemBytes;

  int64_t dstStride2 = globalShape[3] * validCols;
  int64_t dstStride1 = globalShape[2] * dstStride2;

  loop2Size = globalShape[1];
  loop1Size = globalShape[2];
  loop2FirstStrideBytes = dstStride1 * elemBytes;
  loop2SecondStrideBytes = globalStride[1] * elemBytes;
  loop1FirstStrideBytes = dstStride2 * elemBytes;
  loop1SecondStrideBytes = globalStride[2] * elemBytes;
  return success();
}

std::pair<int64_t, int64_t> getStaticTileRowsCols(Value value) {
  if (auto shapedType = dyn_cast<ShapedType>(value.getType())) {
    ArrayRef<int64_t> shape = shapedType.getShape();
    if (shape.size() >= 2)
      return {shape[shape.size() - 2], shape[shape.size() - 1]};
  }
  return {ShapedType::kDynamic, ShapedType::kDynamic};
}

Value materializeStaticOrDynamicDimAsIndex(Value value, int64_t dim,
                                           unsigned dimPos,
                                           PatternRewriter &rewriter,
                                           Location loc) {
  if (dim != ShapedType::kDynamic)
    return rewriter.create<arith::ConstantIndexOp>(loc, dim);
  if (isa<MemRefType>(value.getType()))
    return rewriter.create<memref::DimOp>(loc, value, dimPos);
  return {};
}

LogicalResult materializeShapeBackedValidShapeValues(Value value, Value &rows,
                                                     Value &cols,
                                                     PatternRewriter &rewriter,
                                                     Location loc) {
  rows = {};
  cols = {};

  auto shapedType = dyn_cast<ShapedType>(value.getType());
  if (!shapedType || !shapedType.hasRank() || !hasStructuredTileDriver(value))
    return failure();

  ArrayRef<int64_t> shape = shapedType.getShape();
  if (shape.empty()) {
    rows = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    cols = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    return success();
  }
  if (shape.size() == 1) {
    rows = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    cols = materializeStaticOrDynamicDimAsIndex(value, shape.front(), 0, rewriter, loc);
    return success(cols != nullptr);
  }

  cols = materializeStaticOrDynamicDimAsIndex(value, shape.back(), shape.size() - 1,
                                              rewriter, loc);
  if (!cols)
    return failure();

  Value flatRows = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  for (auto [idx, dim] : llvm::enumerate(shape.drop_back())) {
    Value dimValue =
        materializeStaticOrDynamicDimAsIndex(value, dim, idx, rewriter, loc);
    if (!dimValue)
      return failure();
    flatRows = rewriter.create<arith::MulIOp>(loc, flatRows, dimValue);
  }
  rows = flatRows;
  return success();
}

LogicalResult resolveExecutionValidShape(Value carrier, Value &rowsValue,
                                         Value &colsValue, int64_t &rows,
                                         int64_t &cols,
                                         PatternRewriter &rewriter,
                                         Location loc) {
  rowsValue = materializeIndexValue(rowsValue, rows, rewriter, loc);
  colsValue = materializeIndexValue(colsValue, cols, rewriter, loc);
  if (rowsValue && colsValue)
    return success();

  if (succeeded(materializeShapeBackedValidShapeValues(carrier, rowsValue, colsValue,
                                                       rewriter, loc))) {
    deriveValidShape(carrier, rows, cols);
    return success(rowsValue && colsValue);
  }
  return failure();
}

Attribute getGmMemorySpace(MLIRContext *context) {
  return AddressSpaceAttr::get(context, AddressSpace::GM);
}

AddressSpaceAttr getNormalizedPtrMemorySpace(Attribute memorySpace,
                                             MLIRContext *context) {
  if (auto addrSpace = dyn_cast_or_null<AddressSpaceAttr>(memorySpace))
    return addrSpace;
  if (auto intAttr = dyn_cast_or_null<IntegerAttr>(memorySpace))
    return AddressSpaceAttr::get(context,
                                 static_cast<AddressSpace>(intAttr.getInt()));
  return AddressSpaceAttr::get(context, AddressSpace::GM);
}

Value materializeMemRefView(Value value, ArrayRef<int64_t> shape, Type elementType,
                            Attribute memorySpace,
                            PatternRewriter &rewriter, Location loc) {
  auto memrefType =
      MemRefType::get(shape, elementType, AffineMap(), memorySpace);
  if (value.getType() == memrefType)
    return value;
  return rewriter
      .create<UnrealizedConversionCastOp>(
          loc, TypeRange(ArrayRef<Type>{memrefType}), value)
      .getResult(0);
}

Value materializeTileBufferView(Value value, PatternRewriter &rewriter,
                                Location loc) {
  if (auto memrefType = dyn_cast<BaseMemRefType>(value.getType()))
    return value;

  auto tileType = dyn_cast<TileBufType>(value.getType());
  if (!tileType)
    return {};

  return materializeMemRefView(value, tileType.getShape(), tileType.getElementType(),
                               tileType.getMemorySpace(), rewriter, loc);
}

} // namespace

Value materializeBufferPointer(Value value, Type elementType,
                               Attribute memorySpace,
                               PatternRewriter &rewriter, Location loc) {
  if (!value)
    return {};

  auto ptrMemorySpace =
      getNormalizedPtrMemorySpace(memorySpace, rewriter.getContext());
  auto ptrType = PtrType::get(rewriter.getContext(), elementType, ptrMemorySpace);

  if (value.getType() == ptrType)
    return value;

  if (auto bind = value.getDefiningOp<BindTileOp>())
    return materializeBufferPointer(bind.getSource(), elementType, memorySpace,
                                    rewriter, loc);

  if (auto cast = value.getDefiningOp<PointerCastOp>()) {
    if (cast.getAddrs().empty())
      return {};
    return rewriter.create<CastPtrOp>(loc, ptrType, cast.getAddrs().front())
        .getResult();
  }

  Value memrefValue = materializeTileBufferView(value, rewriter, loc);
  auto memrefType = dyn_cast_or_null<MemRefType>(memrefValue.getType());
  if (!memrefValue || !memrefType)
    return {};
  return rewriter.create<CastPtrOp>(loc, ptrType, memrefValue).getResult();
}

namespace {

Value materializeBufferLikeAddress(Value value, Type elementType,
                                   Attribute memorySpace,
                                   PatternRewriter &rewriter, Location loc) {
  if (!value)
    return {};

  if (auto bind = value.getDefiningOp<BindTileOp>())
    return materializeBufferLikeAddress(bind.getSource(), elementType, memorySpace,
                                        rewriter, loc);

  // Keep memref semantics through the VPTO mainline whenever possible.
  Value memrefValue = materializeTileBufferView(value, rewriter, loc);
  if (memrefValue && isa<BaseMemRefType>(memrefValue.getType()))
    return memrefValue;

  return materializeBufferPointer(value, elementType, memorySpace, rewriter, loc);
}

Value offsetBufferPointer(Value basePtr, Type elementType, Value elementOffset,
                          PatternRewriter &rewriter, Location loc) {
  if (!basePtr)
    return {};

  if (auto ptrType = dyn_cast<PtrType>(basePtr.getType())) {
    Value offsetIndex =
        elementOffset.getType().isIndex()
            ? elementOffset
            : rewriter.create<arith::IndexCastUIOp>(loc,
                                                    rewriter.getIndexType(),
                                                    elementOffset);
    return rewriter.create<AddPtrOp>(loc, ptrType, basePtr, offsetIndex);
  }
  return {};
}

Value buildPackedCountI64(PatternRewriter &rewriter, Location loc,
                          ArrayRef<Value> counts) {
  Value packed = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
  for (auto [idx, count] : llvm::enumerate(counts)) {
    Value countI64 = count.getType().isIndex()
                         ? rewriter.create<arith::IndexCastUIOp>(
                               loc, rewriter.getI64Type(), count)
                         : count;
    if (idx != 0) {
      Value shift = rewriter.create<arith::ConstantIntOp>(loc, idx * 16, 64);
      countI64 = rewriter.create<arith::ShLIOp>(loc, countI64, shift);
    }
    packed = rewriter.create<arith::OrIOp>(loc, packed, countI64);
  }
  return packed;
}

Value buildCeilDivPositiveI64(PatternRewriter &rewriter, Location loc, Value lhs,
                              int64_t rhs) {
  Value rhsValue = rewriter.create<arith::ConstantIntOp>(loc, rhs, 64);
  Value rhsMinusOne = rewriter.create<arith::ConstantIntOp>(loc, rhs - 1, 64);
  Value biased = rewriter.create<arith::AddIOp>(loc, lhs, rhsMinusOne);
  return rewriter.create<arith::DivUIOp>(loc, biased, rhsValue);
}

VPTOPartitionTrace extractPartitionTrace(Value value) {
  VPTOPartitionTrace trace;
  if (auto part = value.getDefiningOp<PartitionViewOp>()) {
    appendStaticSizes(part.getOffsets(), trace.offsets, trace.hasDynamicOffsets);
    appendStaticSizes(part.getSizes(), trace.sizes, trace.hasDynamicSizes);
    return trace;
  }
  if (auto subview = value.getDefiningOp<memref::SubViewOp>()) {
    trace = extractPartitionTrace(subview.getSource());
    SmallVector<int64_t> offsets;
    SmallVector<int64_t> sizes;
    bool hasDynamicOffsets = false;
    bool hasDynamicSizes = false;
    recordStaticSizes(subview.getMixedOffsets(), offsets, hasDynamicOffsets);
    recordStaticSizes(subview.getMixedSizes(), sizes, hasDynamicSizes);
    mergeSubviewTrace(trace, offsets, sizes, hasDynamicOffsets, hasDynamicSizes);
    return trace;
  }
  if (auto reinterpret = value.getDefiningOp<memref::ReinterpretCastOp>())
    return extractPartitionTrace(reinterpret.getSource());
  if (auto cast = value.getDefiningOp<memref::CastOp>())
    return extractPartitionTrace(cast.getSource());
  if (auto unrealized = value.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (!unrealized.getInputs().empty())
      return extractPartitionTrace(unrealized.getInputs().front());
  }
  return trace;
}

VPTOLoadContract extractTLoadContract(TLoadOp op) {
  VPTOLoadContract contract;
  contract.trace = extractPartitionTrace(op.getSrc());
  contract.elementType = getElementType(op.getDst());

  Attribute layoutAttr;
  Value base = resolveTensorViewBase(op.getSrc(), layoutAttr, contract.sourceShape,
                                     contract.sourceStrides);
  (void)base;
  contract.sourceLayout = stringifyLayoutAttr(layoutAttr);

  contract.tileLayout = deriveTileLayout(op.getDst());
  contract.tileDomain = deriveTileDomain(getMemorySpace(op.getDst()));
  deriveValidShapeValues(op.getDst(), contract.validRowsValue, contract.validColsValue);
  deriveValidShape(op.getDst(), contract.validRows, contract.validCols);
  contract.padMode = stringifyPadModeAttr(op.getPadModeAttr());
  contract.padValue = op.getPadValue();
  contract.leftPaddingNum = op.getLeftPaddingNum();
  contract.rightPaddingNum = op.getRightPaddingNum();
  contract.initOutBuffer = op.getInitOutBuffer();
  contract.initCondition = op.getInitCondition();
  return contract;
}

VPTOUnaryContract extractTAbsContract(TAbsOp op) {
  VPTOUnaryContract contract;
  contract.family = "abs";
  contract.tileDomain = deriveTileDomain(getMemorySpace(op.getSrc()));
  contract.tileLayout = deriveTileLayout(op.getSrc());
  deriveValidShapeValues(op.getSrc(), contract.validRowsValue, contract.validColsValue);
  deriveValidShape(op.getSrc(), contract.validRows, contract.validCols);
  contract.elementType = getElementType(op.getSrc());
  contract.loopScope.kind = VPTOLoopScopeKind::AIVVectorScope;
  contract.loopScope.loweredAttr = kLoweredLoopScopeAttrName;
  contract.loopScope.loopDepth = 0;
  return contract;
}

VPTOBinaryContract buildBinaryContract(StringRef family, Value src0) {
  VPTOBinaryContract contract;
  contract.family = family;
  contract.tileDomain = deriveTileDomain(getMemorySpace(src0));
  contract.tileLayout = deriveTileLayout(src0);
  deriveValidShapeValues(src0, contract.validRowsValue, contract.validColsValue);
  deriveValidShape(src0, contract.validRows, contract.validCols);
  contract.elementType = getElementType(src0);
  contract.loopScope.kind = VPTOLoopScopeKind::AIVVectorScope;
  contract.loopScope.loweredAttr = kLoweredLoopScopeAttrName;
  contract.loopScope.loopDepth = 0;
  return contract;
}

VPTOBinaryContract extractTAddContract(TAddOp op) {
  return buildBinaryContract("add", op.getSrc0());
}

VPTOBinaryContract extractTSubContract(TSubOp op) {
  return buildBinaryContract("sub", op.getSrc0());
}

VPTOBinaryContract extractTMulContract(TMulOp op) {
  return buildBinaryContract("mul", op.getSrc0());
}

VPTOBinaryContract extractTDivContract(TDivOp op) {
  return buildBinaryContract("div", op.getSrc0());
}

VPTOUnaryContract buildUnaryContract(StringRef family, Value src) {
  VPTOUnaryContract contract;
  contract.family = family;
  contract.tileDomain = deriveTileDomain(getMemorySpace(src));
  contract.tileLayout = deriveTileLayout(src);
  deriveValidShapeValues(src, contract.validRowsValue, contract.validColsValue);
  deriveValidShape(src, contract.validRows, contract.validCols);
  contract.elementType = getElementType(src);
  contract.loopScope.kind = VPTOLoopScopeKind::AIVVectorScope;
  contract.loopScope.loweredAttr = kLoweredLoopScopeAttrName;
  contract.loopScope.loopDepth = 0;
  return contract;
}

VPTOUnaryContract extractTExpContract(TExpOp op) {
  return buildUnaryContract("exp", op.getSrc());
}

VPTOUnaryContract extractTLogContract(TLogOp op) {
  return buildUnaryContract("log", op.getSrc());
}

VPTOUnaryContract extractTSqrtContract(TSqrtOp op) {
  return buildUnaryContract("sqrt", op.getSrc());
}

VPTOUnaryContract extractTRecipContract(TRecipOp op) {
  return buildUnaryContract("recip", op.getSrc());
}

VPTOUnaryContract extractTReluContract(TReluOp op) {
  return buildUnaryContract("relu", op.getSrc());
}

VPTOUnaryContract extractTNotContract(TNotOp op) {
  return buildUnaryContract("not", op.getSrc());
}

static FailureOr<StringAttr> stringifyA5RoundMode(TCvtOp op,
                                                  PatternRewriter &rewriter) {
  switch (op.getRmode()) {
  case RoundMode::NONE:
  case RoundMode::RINT:
  case RoundMode::CAST_RINT:
    return rewriter.getStringAttr("ROUND_R");
  case RoundMode::ROUND:
    return rewriter.getStringAttr("ROUND_A");
  case RoundMode::FLOOR:
    return rewriter.getStringAttr("ROUND_F");
  case RoundMode::CEIL:
    return rewriter.getStringAttr("ROUND_C");
  case RoundMode::TRUNC:
    return rewriter.getStringAttr("ROUND_Z");
  case RoundMode::ODD:
    return rewriter.getStringAttr("ROUND_O");
  }
  return failure();
}

enum class VPTOCvtLoweringKind {
  Vtrc,
  F32ToBF16,
  F16ToF32,
  BF16ToF32,
};

static FailureOr<VPTOCvtLoweringKind> classifyA5CvtLowering(Type srcElemType,
                                                            Type dstElemType) {
  if (srcElemType.isF32() && dstElemType.isF32())
    return VPTOCvtLoweringKind::Vtrc;
  if (srcElemType.isF32() && dstElemType.isBF16())
    return VPTOCvtLoweringKind::F32ToBF16;
  if (srcElemType.isF16() && dstElemType.isF32())
    return VPTOCvtLoweringKind::F16ToF32;
  if (srcElemType.isBF16() && dstElemType.isF32())
    return VPTOCvtLoweringKind::BF16ToF32;
  return failure();
}

VPTOUnaryContract extractTExpandSContract(TExpandsOp op) {
  VPTOUnaryContract contract;
  contract.family = "expands";
  contract.tileDomain = deriveTileDomain(getMemorySpace(op.getDst()));
  contract.tileLayout = deriveTileLayout(op.getDst());
  deriveValidShapeValues(op.getDst(), contract.validRowsValue,
                         contract.validColsValue);
  deriveValidShape(op.getDst(), contract.validRows, contract.validCols);
  contract.elementType = getElementType(op.getDst());
  contract.loopScope.kind = VPTOLoopScopeKind::AIVVectorScope;
  contract.loopScope.loweredAttr = kLoweredLoopScopeAttrName;
  contract.loopScope.loopDepth = 0;
  return contract;
}

VPTOExpandContract extractTRowExpandContract(TRowExpandOp op) {
  VPTOExpandContract contract;
  contract.family = "rowexpand";
  contract.srcDomain = deriveTileDomain(getMemorySpace(op.getSrc()));
  contract.dstDomain = deriveTileDomain(getMemorySpace(op.getDst()));
  contract.srcLayout = deriveTileLayout(op.getSrc());
  contract.dstLayout = deriveTileLayout(op.getDst());
  contract.elementType = getElementType(op.getSrc());
  deriveValidShapeValues(op.getSrc(), contract.srcValidRowsValue,
                         contract.srcValidColsValue);
  deriveValidShape(op.getSrc(), contract.srcValidRows, contract.srcValidCols);
  deriveValidShapeValues(op.getDst(), contract.dstValidRowsValue,
                         contract.dstValidColsValue);
  deriveValidShape(op.getDst(), contract.dstValidRows, contract.dstValidCols);
  contract.loopScope.kind = VPTOLoopScopeKind::AIVVectorScope;
  contract.loopScope.loweredAttr = kLoweredLoopScopeAttrName;
  contract.loopScope.loopDepth = 0;
  return contract;
}

VPTOExpandContract extractTColExpandContract(TColExpandOp op) {
  VPTOExpandContract contract;
  contract.family = "colexpand";
  contract.srcDomain = deriveTileDomain(getMemorySpace(op.getSrc()));
  contract.dstDomain = deriveTileDomain(getMemorySpace(op.getDst()));
  contract.srcLayout = deriveTileLayout(op.getSrc());
  contract.dstLayout = deriveTileLayout(op.getDst());
  contract.elementType = getElementType(op.getSrc());
  deriveValidShapeValues(op.getSrc(), contract.srcValidRowsValue,
                         contract.srcValidColsValue);
  deriveValidShape(op.getSrc(), contract.srcValidRows, contract.srcValidCols);
  deriveValidShapeValues(op.getDst(), contract.dstValidRowsValue,
                         contract.dstValidColsValue);
  deriveValidShape(op.getDst(), contract.dstValidRows, contract.dstValidCols);
  contract.loopScope.kind = VPTOLoopScopeKind::AIVVectorScope;
  contract.loopScope.loweredAttr = kLoweredLoopScopeAttrName;
  contract.loopScope.loopDepth = 0;
  return contract;
}

VPTORowReduceContract extractTRowReduceContract(Value src, Value dst,
                                                StringRef family) {
  VPTORowReduceContract contract;
  contract.family = family;
  contract.srcDomain = deriveTileDomain(getMemorySpace(src));
  contract.dstDomain = deriveTileDomain(getMemorySpace(dst));
  contract.srcLayout = deriveTileLayout(src);
  contract.dstLayout = deriveTileLayout(dst);
  contract.elementType = getElementType(src);
  deriveValidShapeValues(src, contract.validRowsValue, contract.validColsValue);
  deriveValidShape(src, contract.validRows, contract.validCols);
  int64_t dstRows = ShapedType::kDynamic;
  deriveValidShape(dst, dstRows, contract.dstValidCols);
  contract.loopScope.kind = VPTOLoopScopeKind::AIVVectorScope;
  contract.loopScope.loweredAttr = kLoweredLoopScopeAttrName;
  contract.loopScope.loopDepth = 0;
  return contract;
}

VPTORowReduceContract extractTRowMaxContract(TRowMaxOp op) {
  return extractTRowReduceContract(op.getSrc(), op.getDst(), "rowmax");
}

VPTORowReduceContract extractTRowMinContract(TRowMinOp op) {
  return extractTRowReduceContract(op.getSrc(), op.getDst(), "rowmin");
}

VPTORowReduceContract extractTRowSumContract(TRowSumOp op) {
  return extractTRowReduceContract(op.getSrc(), op.getDst(), "rowsum");
}

VPTOColReduceContract extractTColReduceContract(Value src, Value dst,
                                                StringRef family) {
  VPTOColReduceContract contract;
  contract.family = family;
  contract.srcDomain = deriveTileDomain(getMemorySpace(src));
  contract.dstDomain = deriveTileDomain(getMemorySpace(dst));
  contract.srcLayout = deriveTileLayout(src);
  contract.dstLayout = deriveTileLayout(dst);
  contract.elementType = getElementType(src);
  deriveValidShapeValues(src, contract.validRowsValue, contract.validColsValue);
  deriveValidShape(src, contract.validRows, contract.validCols);
  deriveValidShape(dst, contract.dstValidRows, contract.dstValidCols);
  contract.loopScope.kind = VPTOLoopScopeKind::AIVVectorScope;
  contract.loopScope.loweredAttr = kLoweredLoopScopeAttrName;
  contract.loopScope.loopDepth = 0;
  return contract;
}

VPTOColReduceContract extractTColMaxContract(TColMaxOp op) {
  return extractTColReduceContract(op.getSrc(), op.getDst(), "colmax");
}

VPTOColReduceContract extractTColMinContract(TColMinOp op) {
  return extractTColReduceContract(op.getSrc(), op.getDst(), "colmin");
}

VPTOColReduceContract extractTColSumContract(TColSumOp op) {
  VPTOColReduceContract contract =
      extractTColReduceContract(op.getSrc(), op.getDst(), "colsum");
  contract.isBinary = op.getIsBinary();
  contract.tmp = op.getTmp();
  return contract;
}

VPTOPartContract extractTPartContract(Value src0, Value src1, Value dst,
                                      StringRef family) {
  VPTOPartContract contract;
  contract.family = family;
  contract.src0Domain = deriveTileDomain(getMemorySpace(src0));
  contract.src1Domain = deriveTileDomain(getMemorySpace(src1));
  contract.dstDomain = deriveTileDomain(getMemorySpace(dst));
  contract.src0Layout = deriveTileLayout(src0);
  contract.src1Layout = deriveTileLayout(src1);
  contract.dstLayout = deriveTileLayout(dst);
  contract.elementType = getElementType(dst);
  deriveValidShapeValues(src0, contract.src0ValidRowsValue, contract.src0ValidColsValue);
  deriveValidShapeValues(src1, contract.src1ValidRowsValue, contract.src1ValidColsValue);
  deriveValidShapeValues(dst, contract.dstValidRowsValue, contract.dstValidColsValue);
  deriveValidShape(src0, contract.src0ValidRows, contract.src0ValidCols);
  deriveValidShape(src1, contract.src1ValidRows, contract.src1ValidCols);
  deriveValidShape(dst, contract.dstValidRows, contract.dstValidCols);
  contract.loopScope.kind = VPTOLoopScopeKind::AIVVectorScope;
  contract.loopScope.loweredAttr = kLoweredLoopScopeAttrName;
  contract.loopScope.loopDepth = 0;
  return contract;
}

VPTOPartContract extractTPartAddContract(TPartAddOp op) {
  return extractTPartContract(op.getSrc0(), op.getSrc1(), op.getDst(), "partadd");
}

VPTOPartContract extractTPartMaxContract(TPartMaxOp op) {
  return extractTPartContract(op.getSrc0(), op.getSrc1(), op.getDst(), "partmax");
}

VPTOPartContract extractTPartMinContract(TPartMinOp op) {
  return extractTPartContract(op.getSrc0(), op.getSrc1(), op.getDst(), "partmin");
}

VPTOStoreContract extractTStoreContract(TStoreOp op) {
  VPTOStoreContract contract;
  contract.trace = extractPartitionTrace(op.getDst());

  contract.srcDomain = deriveTileDomain(getMemorySpace(op.getSrc()));
  deriveValidShapeValues(op.getSrc(), contract.validRowsValue, contract.validColsValue);
  deriveValidShape(op.getSrc(), contract.validRows, contract.validCols);
  contract.elementType = getElementType(op.getSrc());

  Attribute layoutAttr;
  Value base = resolveTensorViewBase(op.getDst(), layoutAttr,
                                     contract.destinationShape,
                                     contract.destinationStrides);
  (void)base;
  contract.destinationLayout = stringifyLayoutAttr(layoutAttr);
  return contract;
}

void attachLoadContractAttrs(Operation *op, const VPTOLoadContract &contract) {
  Builder builder(op->getContext());
  SmallVector<int64_t> globalShape;
  SmallVector<int64_t> globalStride;
  normalizeToPTOGlobalShapeAndStride(contract.sourceShape, contract.sourceStrides,
                                     globalShape, globalStride);
  op->setAttr("g_shape", asI64ArrayAttr(builder, globalShape));
  op->setAttr("g_strides", asI64ArrayAttr(builder, globalStride));
}

void attachStoreContractAttrs(Operation *op, const VPTOStoreContract &contract) {
  Builder builder(op->getContext());
  SmallVector<int64_t> globalShape;
  SmallVector<int64_t> globalStride;
  normalizeToPTOGlobalShapeAndStride(contract.destinationShape,
                                     contract.destinationStrides, globalShape,
                                     globalStride);
  op->setAttr("g_shape", asI64ArrayAttr(builder, globalShape));
  op->setAttr("g_strides", asI64ArrayAttr(builder, globalStride));
}

LogicalResult lowerUnsupportedAccStore(Location loc) {
  emitError(loc) << "TSTORE ACC lowering TODO for vpto backend";
  return failure();
}

LogicalResult lowerUnsupportedMatStore(Location loc) {
  emitError(loc) << "TSTORE MAT lowering TODO for vpto backend";
  return failure();
}

} // namespace

FailureOr<pto::VecScopeOp>
createLoopScopeRegion(Location loc, const VPTOLoopScopeContract &contract,
                      PatternRewriter &rewriter) {
  if (contract.kind == VPTOLoopScopeKind::None)
    return failure();
  if (contract.kind != VPTOLoopScopeKind::AIVVectorScope)
    return failure();

  auto vecScope = rewriter.create<pto::VecScopeOp>(loc);
  vecScope.getBody().push_back(new Block());
  return vecScope;
}

void set_loop2_stride_outtoub(Operation *copyOp, int64_t dstStride,
                              int64_t srcStride, Builder &builder) {
  copyOp->setAttr("pto.set_loop2_stride_outtoub",
                  builder.getI64IntegerAttr(
                      packLoopStrideConfig(dstStride, srcStride)));
}

void set_loop1_stride_outtoub(Operation *copyOp, int64_t dstStride,
                              int64_t srcStride, Builder &builder) {
  copyOp->setAttr("pto.set_loop1_stride_outtoub",
                  builder.getI64IntegerAttr(
                      packLoopStrideConfig(dstStride, srcStride)));
}

void set_loop_size_outtoub(Operation *copyOp, int64_t loop2, int64_t loop1,
                           Builder &builder) {
  copyOp->setAttr("pto.set_loop_size_outtoub",
                  builder.getI64IntegerAttr(packLoopSizeConfig(loop2, loop1)));
}

void set_loop2_stride_ubtoout(Operation *copyOp, int64_t srcStride,
                              int64_t dstStride, Builder &builder) {
  copyOp->setAttr("pto.set_loop2_stride_ubtoout",
                  builder.getI64IntegerAttr(
                      packLoopStrideConfig(srcStride, dstStride)));
}

void set_loop1_stride_ubtoout(Operation *copyOp, int64_t srcStride,
                              int64_t dstStride, Builder &builder) {
  copyOp->setAttr("pto.set_loop1_stride_ubtoout",
                  builder.getI64IntegerAttr(
                      packLoopStrideConfig(srcStride, dstStride)));
}

void set_loop_size_ubtoout(Operation *copyOp, int64_t loop2, int64_t loop1,
                           Builder &builder) {
  copyOp->setAttr("pto.set_loop_size_ubtoout",
                  builder.getI64IntegerAttr(packLoopSizeConfig(loop2, loop1)));
}

LogicalResult programCopyGmToUbLoops(Operation *copyOp,
                                     const VPTOLoadContract &contract,
                                     Builder &builder) {
  SmallVector<int64_t> globalShape;
  SmallVector<int64_t> globalStride;
  int64_t nBurst = 0, lenBurst = 0, gmStrideBytes = 0, ubStrideBytes = 0;
  int64_t loop1Size = 0, loop2Size = 0;
  int64_t loop1DstStrideBytes = 0, loop1SrcStrideBytes = 0;
  int64_t loop2DstStrideBytes = 0, loop2SrcStrideBytes = 0;
  if (failed(deriveVecNDTransferConfig(contract.sourceShape, contract.sourceStrides,
                                       contract.tileLayout, contract.elementType,
                                       contract.validRows, contract.validCols,
                                       globalShape, globalStride, nBurst, lenBurst,
                                       gmStrideBytes, ubStrideBytes, loop1Size,
                                       loop2Size, loop1DstStrideBytes,
                                       loop1SrcStrideBytes, loop2DstStrideBytes,
                                       loop2SrcStrideBytes)))
    return failure();

  set_loop2_stride_outtoub(copyOp, loop2DstStrideBytes, loop2SrcStrideBytes, builder);
  set_loop1_stride_outtoub(copyOp, loop1DstStrideBytes, loop1SrcStrideBytes, builder);
  set_loop_size_outtoub(copyOp, loop2Size, loop1Size, builder);
  return success();
}

LogicalResult programCopyUbToGmLoops(Operation *copyOp,
                                     const VPTOStoreContract &contract,
                                     Builder &builder) {
  SmallVector<int64_t> globalShape;
  SmallVector<int64_t> globalStride;
  int64_t nBurst = 0, lenBurst = 0, burstDstStrideBytes = 0, burstSrcStrideBytes = 0;
  int64_t loop1Size = 0, loop2Size = 0;
  int64_t loop1SrcStrideBytes = 0, loop1DstStrideBytes = 0;
  int64_t loop2SrcStrideBytes = 0, loop2DstStrideBytes = 0;
  if (failed(deriveVecNDTransferConfig(contract.destinationShape,
                                       contract.destinationStrides,
                                       "row_major", contract.elementType,
                                       contract.validRows, contract.validCols,
                                       globalShape, globalStride, nBurst, lenBurst,
                                       burstDstStrideBytes, burstSrcStrideBytes,
                                       loop1Size, loop2Size, loop1SrcStrideBytes,
                                       loop1DstStrideBytes, loop2SrcStrideBytes,
                                       loop2DstStrideBytes)))
    return failure();

  set_loop_size_ubtoout(copyOp, loop2Size, loop1Size, builder);
  set_loop1_stride_ubtoout(copyOp, loop1SrcStrideBytes, loop1DstStrideBytes, builder);
  set_loop2_stride_ubtoout(copyOp, loop2SrcStrideBytes, loop2DstStrideBytes, builder);
  return success();
}

int64_t deriveStaticRowStride(Value value) {
  StringRef layout = deriveTileLayout(value);
  if (layout == "col_major")
    return 1;

  if (auto tileType = dyn_cast<TileBufType>(value.getType())) {
    ArrayRef<int64_t> shape = tileType.getShape();
    if (shape.size() >= 2)
      return shape[shape.size() - 1];
  }
  if (auto shapedType = dyn_cast<ShapedType>(value.getType())) {
    ArrayRef<int64_t> shape = shapedType.getShape();
    if (shape.size() >= 2)
      return shape[shape.size() - 1];
  }
  return ShapedType::kDynamic;
}

int64_t deriveStaticShapeDim(Value value, unsigned dim) {
  if (auto tileType = dyn_cast<TileBufType>(value.getType())) {
    ArrayRef<int64_t> shape = tileType.getShape();
    if (dim < shape.size())
      return shape[dim];
  }
  if (auto shapedType = dyn_cast<ShapedType>(value.getType())) {
    ArrayRef<int64_t> shape = shapedType.getShape();
    if (dim < shape.size())
      return shape[dim];
  }
  return ShapedType::kDynamic;
}

int64_t deriveStaticTileCols(Value value) {
  if (auto tileType = dyn_cast<TileBufType>(value.getType())) {
    ArrayRef<int64_t> shape = tileType.getShape();
    if (!shape.empty())
      return shape.back();
  }
  if (auto shapedType = dyn_cast<ShapedType>(value.getType())) {
    ArrayRef<int64_t> shape = shapedType.getShape();
    if (!shape.empty())
      return shape.back();
  }
  return ShapedType::kDynamic;
}

Value buildFullWidthColsCondition(ArrayRef<int64_t> tileCols,
                                  Value validColsValue,
                                  PatternRewriter &rewriter, Location loc) {
  Value condition;
  for (int64_t tileCol : tileCols) {
    if (tileCol == ShapedType::kDynamic)
      return {};
    Value tileColValue = rewriter.create<arith::ConstantIndexOp>(loc, tileCol);
    Value isFullWidth = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, validColsValue, tileColValue);
    condition = condition ? rewriter.create<arith::AndIOp>(loc, condition, isFullWidth)
                          : isFullWidth;
  }
  return condition;
}

Value buildMinIndexValue(PatternRewriter &rewriter, Location loc, Value lhs,
                         Value rhs) {
  auto lhsLtRhs = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                                 lhs, rhs);
  return rewriter.create<arith::SelectOp>(loc, lhsLtRhs, lhs, rhs);
}

struct PredicateMaterialization {
  Value mask;
  Value nextScalar;
};

PredicateMaterialization buildPredicateForLaneCount(PatternRewriter &rewriter,
                                                    Location loc,
                                                    Type elementType,
                                                    Value laneCount) {
  auto maskType = getVPTOMaskTypeForElementType(rewriter.getContext(), elementType);
  Value laneCountI32 = laneCount;
  if (laneCount.getType().isIndex()) {
    laneCountI32 =
        rewriter.create<arith::IndexCastUIOp>(loc, rewriter.getI32Type(), laneCount);
  } else if (auto intType = dyn_cast<IntegerType>(laneCount.getType())) {
    if (intType.getWidth() < 32)
      laneCountI32 = rewriter.create<arith::ExtUIOp>(loc, rewriter.getI32Type(), laneCount);
    else if (intType.getWidth() > 32)
      laneCountI32 =
          rewriter.create<arith::TruncIOp>(loc, rewriter.getI32Type(), laneCount);
  }
  unsigned bitWidth = 0;
  if (auto intType = dyn_cast<IntegerType>(elementType))
    bitWidth = intType.getWidth();
  else if (auto floatType = dyn_cast<FloatType>(elementType))
    bitWidth = floatType.getWidth();
  if (bitWidth == 8) {
    auto plt = rewriter.create<pto::PltB8Op>(loc, maskType, rewriter.getI32Type(),
                                              laneCountI32);
    return {plt.getMask(), plt.getScalarOut()};
  }
  if (bitWidth == 16) {
    auto plt = rewriter.create<pto::PltB16Op>(loc, maskType, rewriter.getI32Type(),
                                               laneCountI32);
    return {plt.getMask(), plt.getScalarOut()};
  }
  if (bitWidth == 32) {
    auto plt = rewriter.create<pto::PltB32Op>(loc, maskType, rewriter.getI32Type(),
                                               laneCountI32);
    return {plt.getMask(), plt.getScalarOut()};
  }
  llvm_unreachable("unsupported element type for predicate lane-count lowering");
}

Value buildPredicateMaskForLaneCount(PatternRewriter &rewriter, Location loc,
                                     Type elementType, Value laneCount) {
  return buildPredicateForLaneCount(rewriter, loc, elementType, laneCount).mask;
}

Value buildAllPredicateMask(PatternRewriter &rewriter, Location loc,
                            Type elementType) {
  auto maskType = getVPTOMaskTypeForElementType(rewriter.getContext(), elementType);
  StringAttr allPattern = rewriter.getStringAttr("PAT_ALL");
  unsigned bitWidth = 0;
  if (auto intType = dyn_cast<IntegerType>(elementType))
    bitWidth = intType.getWidth();
  else if (auto floatType = dyn_cast<FloatType>(elementType))
    bitWidth = floatType.getWidth();
  if (bitWidth == 8)
    return rewriter.create<pto::PsetB8Op>(loc, maskType, allPattern).getResult();
  if (bitWidth == 16)
    return rewriter.create<pto::PsetB16Op>(loc, maskType, allPattern).getResult();
  if (bitWidth == 32)
    return rewriter.create<pto::PsetB32Op>(loc, maskType, allPattern).getResult();
  llvm_unreachable("unsupported element type for full predicate mask lowering");
}

LogicalResult buildMaskedVectorStore(PatternRewriter &rewriter, Location loc,
                                     Value value, Value dstBuffer,
                                     Value dstOffset, Value activeLanes,
                                     int64_t vectorWidth) {
  auto vecType = cast<pto::VRegType>(value.getType());
  Value mask = buildPredicateMaskForLaneCount(rewriter, loc,
                                              vecType.getElementType(),
                                              activeLanes);
  rewriter.create<pto::VstsOp>(loc, value, dstBuffer, dstOffset, StringAttr(),
                                mask);
  return success();
}

Attribute buildRowReduceInitValue(Type elementType, StringRef family,
                                  Builder &builder) {
  if (!isa<FloatType>(elementType))
    return {};

  if (family == "rowsum")
    return builder.getFloatAttr(elementType, 0.0);

  const llvm::fltSemantics &semantics = [&]() -> const llvm::fltSemantics & {
    if (elementType.isF16())
      return llvm::APFloat::IEEEhalf();
    if (elementType.isBF16())
      return llvm::APFloat::BFloat();
    return llvm::APFloat::IEEEsingle();
  }();
  bool negative = family == "rowmax";
  return builder.getFloatAttr(elementType, llvm::APFloat::getInf(semantics, negative));
}

Attribute buildPartPadValue(Type elementType, StringRef family, Builder &builder) {
  if (family == "partadd")
    return builder.getZeroAttr(elementType);
  if (isa<FloatType>(elementType)) {
    const llvm::fltSemantics &semantics = [&]() -> const llvm::fltSemantics & {
      if (elementType.isF16())
        return llvm::APFloat::IEEEhalf();
      if (elementType.isBF16())
        return llvm::APFloat::BFloat();
      return llvm::APFloat::IEEEsingle();
    }();
    bool negative = family == "partmax";
    return builder.getFloatAttr(elementType, llvm::APFloat::getInf(semantics, negative));
  }
  if (auto intType = dyn_cast<IntegerType>(elementType)) {
    unsigned width = intType.getWidth();
    if (intType.isUnsigned()) {
      if (family == "partmax")
        return builder.getIntegerAttr(elementType, 0);
      return builder.getIntegerAttr(elementType, llvm::APInt::getAllOnes(width));
    }
    if (family == "partmax")
      return builder.getIntegerAttr(elementType, llvm::APInt::getSignedMinValue(width));
    return builder.getIntegerAttr(elementType, llvm::APInt::getSignedMaxValue(width));
  }
  return {};
}

Attribute buildFillPadValue(Type elementType, PadValueAttr padAttr, Builder &builder) {
  if (!padAttr)
    return {};

  switch (padAttr.getValue()) {
  case PadValue::Null:
    return {};
  case PadValue::Zero:
    return builder.getZeroAttr(elementType);
  case PadValue::Max:
    if (isa<FloatType>(elementType)) {
      const llvm::fltSemantics &semantics = [&]() -> const llvm::fltSemantics & {
        if (elementType.isF16())
          return llvm::APFloat::IEEEhalf();
        if (elementType.isBF16())
          return llvm::APFloat::BFloat();
        return llvm::APFloat::IEEEsingle();
      }();
      return builder.getFloatAttr(elementType,
                                  llvm::APFloat::getLargest(semantics));
    }
    if (auto intType = dyn_cast<IntegerType>(elementType)) {
      unsigned width = intType.getWidth();
      return intType.isUnsigned()
                 ? builder.getIntegerAttr(elementType,
                                          llvm::APInt::getMaxValue(width))
                 : builder.getIntegerAttr(elementType,
                                          llvm::APInt::getSignedMaxValue(width));
    }
    return {};
  case PadValue::Min:
    if (isa<FloatType>(elementType)) {
      const llvm::fltSemantics &semantics = [&]() -> const llvm::fltSemantics & {
        if (elementType.isF16())
          return llvm::APFloat::IEEEhalf();
        if (elementType.isBF16())
          return llvm::APFloat::BFloat();
        return llvm::APFloat::IEEEsingle();
      }();
      auto value = llvm::APFloat::getLargest(semantics);
      value.changeSign();
      return builder.getFloatAttr(elementType, value);
    }
    if (auto intType = dyn_cast<IntegerType>(elementType)) {
      unsigned width = intType.getWidth();
      return intType.isUnsigned()
                 ? builder.getIntegerAttr(elementType, llvm::APInt(width, 0))
                 : builder.getIntegerAttr(elementType,
                                          llvm::APInt::getSignedMinValue(width));
    }
    return {};
  }
  return {};
}

LogicalResult buildRowReduceVecScope(StringRef family,
                                     const VPTORowReduceContract &contract,
                                     VPTOLoweringStrategy strategy, Value src,
                                     Value dst,
                                     PatternRewriter &rewriter, Location loc) {
  auto vecType = getVPTOVRegType(rewriter.getContext(), contract.elementType);
  if (!vecType)
    return emitError(loc) << "unsupported VPTO row-reduce element type";

  Value srcBuffer = materializeBufferPointer(src, contract.elementType,
                                             getMemorySpace(src), rewriter, loc);
  Value dstBuffer = materializeBufferPointer(dst, contract.elementType,
                                             getMemorySpace(dst), rewriter, loc);
  if (!srcBuffer || !dstBuffer)
    return emitError(loc) << "requires pointer-backed tile buffers for row-reduce lowering";

  if (contract.validRows == ShapedType::kDynamic ||
      contract.validCols == ShapedType::kDynamic)
    return emitError(loc) << family << " lowering currently requires static valid rows and cols";

  int64_t srcRowStride = deriveStaticRowStride(src);
  int64_t dstRowStride = deriveStaticRowStride(dst);
  if (srcRowStride == ShapedType::kDynamic || dstRowStride == ShapedType::kDynamic)
    return emitError(loc) << family << " lowering requires static row strides";

  Attribute initValue = buildRowReduceInitValue(contract.elementType, family, rewriter);
  if (!initValue)
    return emitError(loc) << family << " lowering supports only f16 and f32 element types";

  auto getRowReduceStoreDist = [&]() -> StringAttr {
    if (contract.elementType.isF16() || contract.elementType.isBF16())
      return rewriter.getStringAttr("1PT");
    if (contract.elementType.isF32())
      return rewriter.getStringAttr("1PT");
    return {};
  };
  StringAttr storeDist = getRowReduceStoreDist();
  if (!storeDist)
    return emitError(loc) << family << " lowering supports only f16 and f32 row-reduce stores";

  int64_t vectorWidth = vecType.getElementCount();
  int64_t repeatTimes = llvm::divideCeil(contract.validCols, vectorWidth);

  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value rowsUpper = rewriter.create<arith::ConstantIndexOp>(loc, contract.validRows);
  Value srcRowStrideValue = rewriter.create<arith::ConstantIndexOp>(loc, srcRowStride);
  Value dstRowStrideValue = rewriter.create<arith::ConstantIndexOp>(loc, dstRowStride);
  Value vectorWidthValue = rewriter.create<arith::ConstantIndexOp>(loc, vectorWidth);
  Value initScalar = rewriter.create<arith::ConstantOp>(loc, cast<TypedAttr>(initValue));

  FailureOr<pto::VecScopeOp> vecScope =
      createLoopScopeRegion(loc, contract.loopScope, rewriter);
  if (failed(vecScope))
    return emitError(loc) << "failed to create AIV vector scope region";

  OpBuilder::InsertionGuard aivGuard(rewriter);
  rewriter.setInsertionPointToStart(&(*vecScope).getBody().front());
  Value dstPredicate =
      buildPredicateMaskForLaneCount(rewriter, loc, contract.elementType, c1);
  Value validColsValue =
      rewriter.create<arith::ConstantIndexOp>(loc, contract.validCols);

  if (strategy == VPTOLoweringStrategy::PostUpdate) {
    auto rowLoop =
        rewriter.create<scf::ForOp>(loc, c0, rowsUpper, c1, ValueRange{dstBuffer});

    OpBuilder::InsertionGuard rowGuard(rewriter);
    rewriter.setInsertionPointToStart(rowLoop.getBody());
    Value row = rowLoop.getInductionVar();
    Value dstPtr = rowLoop.getRegionIterArgs().front();
    Value rowBase = rewriter.create<arith::MulIOp>(loc, row, srcRowStrideValue);
    Value srcPtr =
        adjustPointerByElemOffset(srcBuffer, rowBase, getElementByteSize(contract.elementType),
                                  rewriter, loc);
    Value acc = rewriter.create<pto::VbrOp>(loc, vecType, initScalar);
    Value remainingCols = rewriter.create<arith::ConstantIntOp>(
        loc, contract.validCols, 32);
    for (int64_t repeatIndex = 0; repeatIndex < repeatTimes; ++repeatIndex) {
      auto predicateState =
          buildPredicateForLaneCount(rewriter, loc, contract.elementType, remainingCols);
      Value srcPredicate = predicateState.mask;
      auto srcVecOp = rewriter.create<pto::VldsPostOp>(
          loc, TypeRange{vecType, srcPtr.getType()}, srcPtr, vectorWidthValue,
          rewriter.getStringAttr("NORM"));
      Value srcVec = srcVecOp.getResult();
      srcPtr = srcVecOp.getUpdatedSource();

      Value reduced;
      if (family == "rowsum")
        reduced = rewriter.create<pto::VcaddOp>(loc, vecType, srcVec, srcPredicate);
      else if (family == "rowmax")
        reduced = rewriter.create<pto::VcmaxOp>(loc, vecType, srcVec, srcPredicate);
      else if (family == "rowmin")
        reduced = rewriter.create<pto::VcminOp>(loc, vecType, srcVec, srcPredicate);
      else
        return emitError(loc) << "unsupported VPTO row-reduce family: " << family;

      Value fullMask = buildAllPredicateMask(rewriter, loc, contract.elementType);
      if (family == "rowsum")
        acc = rewriter.create<pto::VaddOp>(loc, vecType, acc, reduced, fullMask);
      else if (family == "rowmax")
        acc = rewriter.create<pto::VmaxOp>(loc, vecType, acc, reduced, fullMask);
      else
        acc = rewriter.create<pto::VminOp>(loc, vecType, acc, reduced, fullMask);
      remainingCols = predicateState.nextScalar;
    }

    auto storeOp = rewriter.create<pto::VstsPostOp>(loc, dstPtr.getType(), acc, dstPtr,
                                                     dstRowStrideValue, storeDist,
                                                     dstPredicate);
    Value nextDstPtr = storeOp.getUpdatedDestination();
    rewriter.create<scf::YieldOp>(loc, nextDstPtr);
    return success();
  }

  auto rowLoop = rewriter.create<scf::ForOp>(loc, c0, rowsUpper, c1);
  OpBuilder::InsertionGuard rowGuard(rewriter);
  rewriter.setInsertionPointToStart(rowLoop.getBody());
  Value row = rowLoop.getInductionVar();
  Value rowBase = rewriter.create<arith::MulIOp>(loc, row, srcRowStrideValue);
  Value acc = rewriter.create<pto::VbrOp>(loc, vecType, initScalar);
  for (int64_t repeatIndex = 0; repeatIndex < repeatTimes; ++repeatIndex) {
    Value repeat = rewriter.create<arith::ConstantIndexOp>(loc, repeatIndex);
    Value repeatBase =
        rewriter.create<arith::MulIOp>(loc, repeat, vectorWidthValue);
    Value srcOffset =
        rewriter.create<arith::AddIOp>(loc, rowBase, repeatBase);
    Value remainingCols =
        rewriter.create<arith::SubIOp>(loc, validColsValue, repeatBase);
    Value srcPredicate = buildPredicateMaskForLaneCount(
        rewriter, loc, contract.elementType, remainingCols);
    Value srcVec =
        rewriter.create<pto::VldsOp>(loc, vecType, srcBuffer, srcOffset,
                                      StringAttr())
            .getResult();

    Value reduced;
    if (family == "rowsum")
      reduced = rewriter.create<pto::VcaddOp>(loc, vecType, srcVec, srcPredicate);
    else if (family == "rowmax")
      reduced = rewriter.create<pto::VcmaxOp>(loc, vecType, srcVec, srcPredicate);
    else if (family == "rowmin")
      reduced = rewriter.create<pto::VcminOp>(loc, vecType, srcVec, srcPredicate);
    else
      return emitError(loc) << "unsupported VPTO row-reduce family: " << family;

    Value fullMask = buildAllPredicateMask(rewriter, loc, contract.elementType);
    if (family == "rowsum")
      acc = rewriter.create<pto::VaddOp>(loc, vecType, acc, reduced, fullMask);
    else if (family == "rowmax")
      acc = rewriter.create<pto::VmaxOp>(loc, vecType, acc, reduced, fullMask);
    else
      acc = rewriter.create<pto::VminOp>(loc, vecType, acc, reduced, fullMask);
  }

  Value dstOffset = rewriter.create<arith::MulIOp>(loc, row, dstRowStrideValue);
  rewriter.create<pto::VstsOp>(loc, acc, dstBuffer, dstOffset, storeDist,
                                dstPredicate);
  return success();
}

LogicalResult buildColReduceVecScope(StringRef family,
                                     const VPTOColReduceContract &contract,
                                     Value src, Value dst, Value tmp,
                                     PatternRewriter &rewriter, Location loc) {
  auto vecType = getVPTOVRegType(rewriter.getContext(), contract.elementType);
  if (!vecType)
    return emitError(loc) << "unsupported VPTO col-reduce element type";

  Value srcBuffer = materializeBufferPointer(src, contract.elementType,
                                             getMemorySpace(src), rewriter, loc);
  Value dstBuffer = materializeBufferPointer(dst, contract.elementType,
                                             getMemorySpace(dst), rewriter, loc);
  if (!srcBuffer || !dstBuffer)
    return emitError(loc) << "requires pointer-backed tile buffers for col-reduce lowering";

  Value tmpBuffer;
  if (contract.isBinary) {
    tmpBuffer = materializeBufferPointer(tmp, contract.elementType, getMemorySpace(tmp),
                                         rewriter, loc);
    if (!tmpBuffer)
      return emitError(loc) << "binary colsum lowering requires pointer-backed tmp tile";
  }

  int64_t srcRowStride = deriveStaticRowStride(src);
  int64_t dstRowStride = deriveStaticRowStride(dst);
  int64_t tmpRowStride =
      contract.isBinary ? deriveStaticRowStride(tmp) : ShapedType::kDynamic;
  if (srcRowStride == ShapedType::kDynamic || dstRowStride == ShapedType::kDynamic ||
      (contract.isBinary && tmpRowStride == ShapedType::kDynamic))
    return emitError(loc) << family << " lowering requires static row strides";

  Attribute initValue = buildRowReduceInitValue(contract.elementType, family, rewriter);
  if (!initValue)
    return emitError(loc) << family << " lowering supports only f16 and f32 element types";

  int64_t vectorWidth = vecType.getElementCount();
  int64_t repeatTimes = llvm::divideCeil(contract.validCols, vectorWidth);

  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value repeatUpper = rewriter.create<arith::ConstantIndexOp>(loc, repeatTimes);
  Value rowUpper = rewriter.create<arith::ConstantIndexOp>(loc, contract.validRows);
  Value srcRowStrideValue = rewriter.create<arith::ConstantIndexOp>(loc, srcRowStride);
  Value dstRowStrideValue = rewriter.create<arith::ConstantIndexOp>(loc, dstRowStride);
  Value vectorWidthValue = rewriter.create<arith::ConstantIndexOp>(loc, vectorWidth);
  Value initScalar = rewriter.create<arith::ConstantOp>(loc, cast<TypedAttr>(initValue));

  FailureOr<pto::VecScopeOp> vecScope =
      createLoopScopeRegion(loc, contract.loopScope, rewriter);
  if (failed(vecScope))
    return emitError(loc) << "failed to create AIV vector scope region";

  OpBuilder::InsertionGuard aivGuard(rewriter);
  rewriter.setInsertionPointToStart(&(*vecScope).getBody().front());
  auto chunkLoop = rewriter.create<scf::ForOp>(loc, c0, repeatUpper, c1);

  OpBuilder::InsertionGuard chunkGuard(rewriter);
  rewriter.setInsertionPointToStart(chunkLoop.getBody());
  Value chunk = chunkLoop.getInductionVar();
  Value chunkOffset = rewriter.create<arith::MulIOp>(loc, chunk, vectorWidthValue);

  if (!contract.isBinary) {
    Value firstRowOffset = chunkOffset;
    Value acc0 =
        rewriter.create<pto::VldsOp>(loc, vecType, srcBuffer, firstRowOffset, StringAttr()).getResult();
    auto rowLoop = rewriter.create<scf::ForOp>(loc, c1, rowUpper, c1, ValueRange{acc0});
    OpBuilder::InsertionGuard rowGuard(rewriter);
    rewriter.setInsertionPointToStart(rowLoop.getBody());
    Value row = rowLoop.getInductionVar();
    Value acc = rowLoop.getRegionIterArgs().front();
    Value rowBase = rewriter.create<arith::MulIOp>(loc, row, srcRowStrideValue);
    Value srcOffset = rewriter.create<arith::AddIOp>(loc, rowBase, chunkOffset);
    Value srcVec =
        rewriter.create<pto::VldsOp>(loc, vecType, srcBuffer, srcOffset, StringAttr()).getResult();
    Value nextAcc;
    Value fullMask = buildAllPredicateMask(rewriter, loc, contract.elementType);
    if (family == "colmax")
      nextAcc = rewriter.create<pto::VmaxOp>(loc, vecType, acc, srcVec, fullMask);
    else if (family == "colmin")
      nextAcc = rewriter.create<pto::VminOp>(loc, vecType, acc, srcVec, fullMask);
    else
      nextAcc = rewriter.create<pto::VaddOp>(loc, vecType, acc, srcVec, fullMask);
    rewriter.create<scf::YieldOp>(loc, nextAcc);

    rewriter.setInsertionPointAfter(rowLoop);
    Value dstOffset = chunkOffset;
    rewriter.create<pto::VstsOp>(
        loc, rowLoop.getResult(0), dstBuffer, dstOffset, StringAttr(),
        buildAllPredicateMask(rewriter, loc, contract.elementType));
    return success();
  }

  Value tmpRowStrideValue = rewriter.create<arith::ConstantIndexOp>(loc, tmpRowStride);
  auto reducePair = [&](Value lhs, Value rhs) -> Value {
    return rewriter.create<pto::VaddOp>(
        loc, vecType, lhs, rhs, buildAllPredicateMask(rewriter, loc, contract.elementType))
        .getResult();
  };

  int64_t nLoopStatic = contract.validRows / 2;
  bool remainStatic = (contract.validRows % 2) != 0;
  Value pairUpper = rewriter.create<arith::ConstantIndexOp>(loc, nLoopStatic);
  auto pairLoop = rewriter.create<scf::ForOp>(loc, c0, pairUpper, c1);
  {
    OpBuilder::InsertionGuard pairGuard(rewriter);
    rewriter.setInsertionPointToStart(pairLoop.getBody());
    Value pair = pairLoop.getInductionVar();
    Value row0 = rewriter.create<arith::MulIOp>(
        loc, rewriter.create<arith::MulIOp>(loc, pair, rewriter.create<arith::ConstantIndexOp>(loc, 2)),
        srcRowStrideValue);
    Value row1 = rewriter.create<arith::MulIOp>(
        loc, rewriter.create<arith::AddIOp>(loc,
                                            rewriter.create<arith::MulIOp>(loc, pair, rewriter.create<arith::ConstantIndexOp>(loc, 2)),
                                            c1),
        srcRowStrideValue);
    Value src0Offset = rewriter.create<arith::AddIOp>(loc, row0, chunkOffset);
    Value src1Offset = rewriter.create<arith::AddIOp>(loc, row1, chunkOffset);
    Value lhs = rewriter.create<pto::VldsOp>(loc, vecType, srcBuffer, src0Offset, StringAttr()).getResult();
    Value rhs = rewriter.create<pto::VldsOp>(loc, vecType, srcBuffer, src1Offset, StringAttr()).getResult();
    Value sum = reducePair(lhs, rhs);
    Value tmpOffset = rewriter.create<arith::MulIOp>(loc, pair, tmpRowStrideValue);
    rewriter.create<pto::VstsOp>(loc, sum, tmpBuffer, tmpOffset, StringAttr(),
                                  buildAllPredicateMask(rewriter, loc,
                                                        contract.elementType));
  }

  if (remainStatic && nLoopStatic > 0) {
    Value lastRowOffset = rewriter.create<arith::AddIOp>(
        loc,
        rewriter.create<arith::MulIOp>(
            loc, rewriter.create<arith::ConstantIndexOp>(loc, contract.validRows - 1),
            srcRowStrideValue),
        chunkOffset);
    Value tmpOffset = rewriter.create<arith::MulIOp>(
        loc, rewriter.create<arith::ConstantIndexOp>(loc, nLoopStatic - 1), tmpRowStrideValue);
    Value lhs = rewriter.create<pto::VldsOp>(loc, vecType, srcBuffer, lastRowOffset, StringAttr()).getResult();
    Value rhs = rewriter.create<pto::VldsOp>(loc, vecType, tmpBuffer, tmpOffset, StringAttr()).getResult();
    Value sum = reducePair(lhs, rhs);
    rewriter.create<pto::VstsOp>(loc, sum, tmpBuffer, tmpOffset, StringAttr(),
                                  buildAllPredicateMask(rewriter, loc,
                                                        contract.elementType));
  }

  int64_t currentRows = nLoopStatic;
  while (currentRows > 1) {
    int64_t nextRows = currentRows / 2;
    bool remain = (currentRows % 2) != 0;
    Value nextUpper = rewriter.create<arith::ConstantIndexOp>(loc, nextRows);
    auto foldLoop = rewriter.create<scf::ForOp>(loc, c0, nextUpper, c1);
    OpBuilder::InsertionGuard foldGuard(rewriter);
    rewriter.setInsertionPointToStart(foldLoop.getBody());
    Value pair = foldLoop.getInductionVar();
    Value idx2 = rewriter.create<arith::MulIOp>(
        loc, pair, rewriter.create<arith::ConstantIndexOp>(loc, 2));
    Value idx2p1 = rewriter.create<arith::AddIOp>(loc, idx2, c1);
    Value lhsOff = rewriter.create<arith::MulIOp>(loc, idx2, tmpRowStrideValue);
    Value rhsOff = rewriter.create<arith::MulIOp>(loc, idx2p1, tmpRowStrideValue);
    Value lhs = rewriter.create<pto::VldsOp>(loc, vecType, tmpBuffer, lhsOff, StringAttr()).getResult();
    Value rhs = rewriter.create<pto::VldsOp>(loc, vecType, tmpBuffer, rhsOff, StringAttr()).getResult();
    Value sum = reducePair(lhs, rhs);
    Value outOff = rewriter.create<arith::MulIOp>(loc, pair, tmpRowStrideValue);
    rewriter.create<pto::VstsOp>(loc, sum, tmpBuffer, outOff, StringAttr(),
                                  buildAllPredicateMask(rewriter, loc,
                                                        contract.elementType));

    rewriter.setInsertionPointAfter(foldLoop);
    if (remain && nextRows > 0) {
      Value lhsOff = rewriter.create<arith::MulIOp>(
          loc, rewriter.create<arith::ConstantIndexOp>(loc, nextRows - 1), tmpRowStrideValue);
      Value rhsOff = rewriter.create<arith::MulIOp>(
          loc, rewriter.create<arith::ConstantIndexOp>(loc, 2 * nextRows), tmpRowStrideValue);
      Value lhs = rewriter.create<pto::VldsOp>(loc, vecType, tmpBuffer, lhsOff, StringAttr()).getResult();
      Value rhs = rewriter.create<pto::VldsOp>(loc, vecType, tmpBuffer, rhsOff, StringAttr()).getResult();
      Value sum = reducePair(lhs, rhs);
      rewriter.create<pto::VstsOp>(loc, sum, tmpBuffer, lhsOff, StringAttr(),
                                    buildAllPredicateMask(rewriter, loc,
                                                          contract.elementType));
    }
    currentRows = nextRows;
  }

  Value finalVec;
  if (currentRows == 0) {
    finalVec = rewriter.create<pto::VbrOp>(loc, vecType, initScalar).getResult();
  } else {
    finalVec = rewriter.create<pto::VldsOp>(loc, vecType, tmpBuffer, c0, StringAttr()).getResult();
  }
  Value dstOffset = chunkOffset;
  rewriter.create<pto::VstsOp>(loc, finalVec, dstBuffer, dstOffset, StringAttr(),
                                buildAllPredicateMask(rewriter, loc,
                                                      contract.elementType));
  return success();
}

LogicalResult buildPartFill(StringRef family, const VPTOPartContract &contract,
                            Value dstBuffer, pto::VRegType vecType,
                            int64_t dstStride, PatternRewriter &rewriter,
                            Location loc) {
  Attribute initValue = buildPartPadValue(contract.elementType, family, rewriter);
  if (!initValue)
    return emitError(loc) << "unsupported pad value for " << family;
  int64_t vectorWidth = vecType.getElementCount();
  int64_t repeatTimes = llvm::divideCeil(contract.dstValidCols, vectorWidth);
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value rowsUpper = rewriter.create<arith::ConstantIndexOp>(loc, contract.dstValidRows);
  Value repeatUpper = rewriter.create<arith::ConstantIndexOp>(loc, repeatTimes);
  Value dstStrideValue = rewriter.create<arith::ConstantIndexOp>(loc, dstStride);
  Value vectorWidthValue = rewriter.create<arith::ConstantIndexOp>(loc, vectorWidth);
  Value initScalar = rewriter.create<arith::ConstantOp>(loc, cast<TypedAttr>(initValue));
  Value initVec = rewriter.create<pto::VbrOp>(loc, vecType, initScalar);
  auto rowLoop = rewriter.create<scf::ForOp>(loc, c0, rowsUpper, c1);
  OpBuilder::InsertionGuard rowGuard(rewriter);
  rewriter.setInsertionPointToStart(rowLoop.getBody());
  auto chunkLoop = rewriter.create<scf::ForOp>(loc, c0, repeatUpper, c1);
  OpBuilder::InsertionGuard chunkGuard(rewriter);
  rewriter.setInsertionPointToStart(chunkLoop.getBody());
  Value row = rowLoop.getInductionVar();
  Value chunk = chunkLoop.getInductionVar();
  Value rowBase = rewriter.create<arith::MulIOp>(loc, row, dstStrideValue);
  Value chunkBase = rewriter.create<arith::MulIOp>(loc, chunk, vectorWidthValue);
  Value dstOffset = rewriter.create<arith::AddIOp>(loc, rowBase, chunkBase);
  rewriter.create<pto::VstsOp>(loc, initVec, dstBuffer, dstOffset, StringAttr(),
                                buildAllPredicateMask(rewriter, loc,
                                                      vecType.getElementType()));
  rewriter.setInsertionPointAfter(chunkLoop);
  return success();
}

LogicalResult buildPartCopyRegion(Value srcBuffer, Value dstBuffer, pto::VRegType vecType,
                                  int64_t srcStride, int64_t dstStride,
                                  int64_t startRow, int64_t validRows,
                                  int64_t validCols, PatternRewriter &rewriter,
                                  Location loc) {
  int64_t vectorWidth = vecType.getElementCount();
  int64_t repeatTimes = llvm::divideCeil(validCols, vectorWidth);
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value rowsUpper = rewriter.create<arith::ConstantIndexOp>(loc, validRows);
  Value repeatUpper = rewriter.create<arith::ConstantIndexOp>(loc, repeatTimes);
  Value srcStrideValue = rewriter.create<arith::ConstantIndexOp>(loc, srcStride);
  Value dstStrideValue = rewriter.create<arith::ConstantIndexOp>(loc, dstStride);
  Value vectorWidthValue = rewriter.create<arith::ConstantIndexOp>(loc, vectorWidth);
  Value startRowValue = rewriter.create<arith::ConstantIndexOp>(loc, startRow);
  auto rowLoop = rewriter.create<scf::ForOp>(loc, startRowValue, rowsUpper, c1);
  OpBuilder::InsertionGuard rowGuard(rewriter);
  rewriter.setInsertionPointToStart(rowLoop.getBody());
  auto chunkLoop = rewriter.create<scf::ForOp>(loc, c0, repeatUpper, c1);
  OpBuilder::InsertionGuard chunkGuard(rewriter);
  rewriter.setInsertionPointToStart(chunkLoop.getBody());
  Value row = rowLoop.getInductionVar();
  Value chunk = chunkLoop.getInductionVar();
  Value rowSrc = rewriter.create<arith::MulIOp>(loc, row, srcStrideValue);
  Value rowDst = rewriter.create<arith::MulIOp>(loc, row, dstStrideValue);
  Value chunkBase = rewriter.create<arith::MulIOp>(loc, chunk, vectorWidthValue);
  Value srcOffset = rewriter.create<arith::AddIOp>(loc, rowSrc, chunkBase);
  Value dstOffset = rewriter.create<arith::AddIOp>(loc, rowDst, chunkBase);
  Value vec = rewriter.create<pto::VldsOp>(loc, vecType, srcBuffer, srcOffset, StringAttr()).getResult();
  rewriter.create<pto::VstsOp>(loc, vec, dstBuffer, dstOffset, StringAttr(),
                                buildAllPredicateMask(rewriter, loc,
                                                      vecType.getElementType()));
  rewriter.setInsertionPointAfter(chunkLoop);
  return success();
}

LogicalResult buildPartBinaryRegion(StringRef family, Value src0Buffer, Value src1Buffer,
                                    Value dstBuffer, pto::VRegType vecType,
                                    int64_t src0Stride, int64_t src1Stride,
                                    int64_t dstStride, int64_t validRows,
                                    int64_t validCols, PatternRewriter &rewriter,
                                    Location loc) {
  int64_t vectorWidth = vecType.getElementCount();
  int64_t repeatTimes = llvm::divideCeil(validCols, vectorWidth);
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value rowsUpper = rewriter.create<arith::ConstantIndexOp>(loc, validRows);
  Value repeatUpper = rewriter.create<arith::ConstantIndexOp>(loc, repeatTimes);
  Value src0StrideValue = rewriter.create<arith::ConstantIndexOp>(loc, src0Stride);
  Value src1StrideValue = rewriter.create<arith::ConstantIndexOp>(loc, src1Stride);
  Value dstStrideValue = rewriter.create<arith::ConstantIndexOp>(loc, dstStride);
  Value vectorWidthValue = rewriter.create<arith::ConstantIndexOp>(loc, vectorWidth);
  auto rowLoop = rewriter.create<scf::ForOp>(loc, c0, rowsUpper, c1);
  OpBuilder::InsertionGuard rowGuard(rewriter);
  rewriter.setInsertionPointToStart(rowLoop.getBody());
  auto chunkLoop = rewriter.create<scf::ForOp>(loc, c0, repeatUpper, c1);
  OpBuilder::InsertionGuard chunkGuard(rewriter);
  rewriter.setInsertionPointToStart(chunkLoop.getBody());
  Value row = rowLoop.getInductionVar();
  Value chunk = chunkLoop.getInductionVar();
  Value chunkBase = rewriter.create<arith::MulIOp>(loc, chunk, vectorWidthValue);
  Value rowSrc0 = rewriter.create<arith::MulIOp>(loc, row, src0StrideValue);
  Value rowSrc1 = rewriter.create<arith::MulIOp>(loc, row, src1StrideValue);
  Value rowDst = rewriter.create<arith::MulIOp>(loc, row, dstStrideValue);
  Value src0Offset = rewriter.create<arith::AddIOp>(loc, rowSrc0, chunkBase);
  Value src1Offset = rewriter.create<arith::AddIOp>(loc, rowSrc1, chunkBase);
  Value dstOffset = rewriter.create<arith::AddIOp>(loc, rowDst, chunkBase);
  Value lhs = rewriter.create<pto::VldsOp>(loc, vecType, src0Buffer, src0Offset, StringAttr()).getResult();
  Value rhs = rewriter.create<pto::VldsOp>(loc, vecType, src1Buffer, src1Offset, StringAttr()).getResult();
  Value fullMask = buildAllPredicateMask(rewriter, loc, vecType.getElementType());
  Value out;
  if (family == "partadd")
    out = rewriter.create<pto::VaddOp>(loc, vecType, lhs, rhs, fullMask);
  else if (family == "partmax")
    out = rewriter.create<pto::VmaxOp>(loc, vecType, lhs, rhs, fullMask);
  else if (family == "partmin")
    out = rewriter.create<pto::VminOp>(loc, vecType, lhs, rhs, fullMask);
  else
    return emitError(loc) << "unsupported part family: " << family;
  rewriter.create<pto::VstsOp>(loc, out, dstBuffer, dstOffset, StringAttr(),
                                buildAllPredicateMask(rewriter, loc,
                                                      vecType.getElementType()));
  rewriter.setInsertionPointAfter(chunkLoop);
  return success();
}

LogicalResult buildPartVecScope(StringRef family, const VPTOPartContract &contract,
                                Value src0, Value src1, Value dst,
                                PatternRewriter &rewriter, Location loc) {
  auto vecType = getVPTOVRegType(rewriter.getContext(), contract.elementType);
  if (!vecType)
    return emitError(loc) << "unsupported VPTO part element type";
  Value src0Buffer = materializeBufferLikeAddress(src0, contract.elementType,
                                                  getMemorySpace(src0), rewriter, loc);
  Value src1Buffer = materializeBufferLikeAddress(src1, contract.elementType,
                                                  getMemorySpace(src1), rewriter, loc);
  Value dstBuffer = materializeBufferLikeAddress(dst, contract.elementType,
                                                 getMemorySpace(dst), rewriter, loc);
  if (!src0Buffer || !src1Buffer || !dstBuffer)
    return emitError(loc) << "requires pointer-backed tile buffers for part lowering";
  int64_t src0Stride = deriveStaticRowStride(src0);
  int64_t src1Stride = deriveStaticRowStride(src1);
  int64_t dstStride = deriveStaticRowStride(dst);
  if (src0Stride == ShapedType::kDynamic || src1Stride == ShapedType::kDynamic ||
      dstStride == ShapedType::kDynamic)
    return emitError(loc) << family << " lowering requires static row strides";

  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  FailureOr<pto::VecScopeOp> vecScope =
      createLoopScopeRegion(loc, contract.loopScope, rewriter);
  if (failed(vecScope))
    return emitError(loc) << "failed to create AIV vector scope region";
  OpBuilder::InsertionGuard aivGuard(rewriter);
  rewriter.setInsertionPointToStart(&(*vecScope).getBody().front());

  auto condSrc0EqDst = contract.src0ValidRows == contract.dstValidRows &&
                       contract.src0ValidCols == contract.dstValidCols;
  auto condSrc0RowLtDst = contract.src0ValidRows < contract.dstValidRows &&
                          contract.src0ValidCols == contract.dstValidCols;
  auto condSrc0ColLtDst = contract.src0ValidRows <= contract.dstValidRows &&
                          contract.src0ValidCols < contract.dstValidCols;
  auto condSrc1EqDst = contract.src1ValidRows == contract.dstValidRows &&
                       contract.src1ValidCols == contract.dstValidCols;
  auto condSrc1RowLtDst = contract.src1ValidRows < contract.dstValidRows &&
                          contract.src1ValidCols == contract.dstValidCols;
  auto condSrc1ColLtDst = contract.src1ValidRows <= contract.dstValidRows &&
                          contract.src1ValidCols < contract.dstValidCols;

  if (family == "partadd") {
    if (condSrc0EqDst && condSrc1EqDst)
      return buildPartBinaryRegion(family, src0Buffer, src1Buffer, dstBuffer, vecType,
                                   src0Stride, src1Stride, dstStride,
                                   contract.dstValidRows, contract.dstValidCols,
                                   rewriter, loc);
    if (condSrc0ColLtDst && condSrc1EqDst) {
      if (failed(buildPartCopyRegion(src1Buffer, dstBuffer, vecType, src1Stride, dstStride,
                                     0, contract.src1ValidRows, contract.dstValidCols,
                                     rewriter, loc)))
        return failure();
      if (contract.src0ValidCols != 0)
        return buildPartBinaryRegion(family, src0Buffer, dstBuffer, dstBuffer, vecType,
                                     src0Stride, dstStride, dstStride,
                                     contract.src0ValidRows, contract.src0ValidCols,
                                     rewriter, loc);
      return success();
    }
    if (condSrc0RowLtDst && condSrc1EqDst) {
      if (contract.src0ValidRows != 0 &&
          failed(buildPartBinaryRegion(family, src0Buffer, src1Buffer, dstBuffer, vecType,
                                       src0Stride, src1Stride, dstStride,
                                       contract.src0ValidRows, contract.src0ValidCols,
                                       rewriter, loc)))
        return failure();
      return buildPartCopyRegion(src1Buffer, dstBuffer, vecType, src1Stride, dstStride,
                                 contract.src0ValidRows, contract.src1ValidRows,
                                 contract.dstValidCols, rewriter, loc);
    }
    if (condSrc1ColLtDst && condSrc0EqDst) {
      if (failed(buildPartCopyRegion(src0Buffer, dstBuffer, vecType, src0Stride, dstStride,
                                     0, contract.src0ValidRows, contract.dstValidCols,
                                     rewriter, loc)))
        return failure();
      if (contract.src1ValidCols != 0)
        return buildPartBinaryRegion(family, src1Buffer, dstBuffer, dstBuffer, vecType,
                                     src1Stride, dstStride, dstStride,
                                     contract.src1ValidRows, contract.src1ValidCols,
                                     rewriter, loc);
      return success();
    }
    if (condSrc1RowLtDst && condSrc0EqDst) {
      if (contract.src1ValidRows != 0 &&
          failed(buildPartBinaryRegion(family, src0Buffer, src1Buffer, dstBuffer, vecType,
                                       src0Stride, src1Stride, dstStride,
                                       contract.src1ValidRows, contract.src1ValidCols,
                                       rewriter, loc)))
        return failure();
      return buildPartCopyRegion(src0Buffer, dstBuffer, vecType, src0Stride, dstStride,
                                 contract.src1ValidRows, contract.src0ValidRows,
                                 contract.dstValidCols, rewriter, loc);
    }
    return emitError(loc) << "partadd lowering only supports PTO-covered destination-equality/extension cases";
  }

  bool condDstGeSrc = contract.src0ValidRows <= contract.dstValidRows &&
                      contract.src0ValidCols <= contract.dstValidCols &&
                      contract.src1ValidRows <= contract.dstValidRows &&
                      contract.src1ValidCols <= contract.dstValidCols;
  if (!condDstGeSrc)
    return emitError(loc) << family << " lowering only supports dst >= src0/src1 shape relation";
  if (failed(buildPartFill(family, contract, dstBuffer, vecType, dstStride, rewriter, loc)))
    return failure();
  if (failed(buildPartCopyRegion(src0Buffer, dstBuffer, vecType, src0Stride, dstStride,
                                 0, contract.src0ValidRows, contract.src0ValidCols,
                                 rewriter, loc)))
    return failure();
  return buildPartBinaryRegion(family, dstBuffer, src1Buffer, dstBuffer, vecType,
                               dstStride, src1Stride, dstStride,
                               contract.src1ValidRows, contract.src1ValidCols,
                               rewriter, loc);
}

LogicalResult buildUnaryVecScope(StringRef family,
                                 const VPTOUnaryContract &contract,
                                 VPTOLoweringStrategy strategy, Value src,
                                 Value dst, PatternRewriter &rewriter,
                                 Location loc) {
  auto vecType = getVPTOVRegType(rewriter.getContext(), contract.elementType);
  if (!vecType)
    return emitError(loc) << "unsupported VPTO unary element type";

  Value srcBuffer = materializeBufferPointer(src, contract.elementType,
                                             getMemorySpace(src), rewriter, loc);
  Value dstBuffer = materializeBufferPointer(dst, contract.elementType,
                                             getMemorySpace(dst), rewriter, loc);
  if (!srcBuffer || !dstBuffer)
    return emitError(loc) << "requires pointer-backed tile buffers for unary lowering";

  int64_t vectorWidth = vecType.getElementCount();
  Value validRowsValue;
  Value validColsValue;
  int64_t validRows = ShapedType::kDynamic;
  int64_t validCols = ShapedType::kDynamic;
  deriveValidShapeValues(dst, validRowsValue, validColsValue);
  deriveValidShape(dst, validRows, validCols);
  if (failed(resolveExecutionValidShape(dst, validRowsValue, validColsValue, validRows,
                                        validCols, rewriter, loc)))
    return emitError(loc) << "unary lowering requires valid rows and cols";

  int64_t srcStride = deriveStaticRowStride(src);
  int64_t dstStride = deriveStaticRowStride(dst);
  int64_t srcCols = deriveStaticTileCols(src);
  int64_t dstCols = deriveStaticTileCols(dst);
  if (srcStride == ShapedType::kDynamic || dstStride == ShapedType::kDynamic ||
      srcCols == ShapedType::kDynamic || dstCols == ShapedType::kDynamic)
    return emitError(loc) << "unary lowering requires static row strides and cols";

  auto buildUnaryValue = [&](Value loaded, Value predicate) -> FailureOr<Value> {
    if (family == "abs")
      return rewriter.create<pto::VabsOp>(loc, vecType, loaded, predicate).getResult();
    if (family == "exp")
      return rewriter.create<pto::VexpOp>(loc, vecType, loaded, predicate).getResult();
    if (family == "log")
      return rewriter.create<pto::VlnOp>(loc, vecType, loaded, predicate).getResult();
    if (family == "sqrt")
      return rewriter.create<pto::VsqrtOp>(loc, vecType, loaded, predicate).getResult();
    if (family == "relu")
      return rewriter.create<pto::VreluOp>(loc, vecType, loaded, predicate).getResult();
    if (family == "not")
      return rewriter.create<pto::VnotOp>(loc, vecType, loaded, predicate).getResult();
    return failure();
  };

  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value totalElementsValue =
      rewriter.create<arith::MulIOp>(loc, validRowsValue, validColsValue);
  Value vectorStepValue =
      rewriter.create<arith::ConstantIndexOp>(loc, vectorWidth);
  Value srcStrideValue = rewriter.create<arith::ConstantIndexOp>(loc, srcStride);
  Value dstStrideValue = rewriter.create<arith::ConstantIndexOp>(loc, dstStride);
  Value scalarInit = rewriter.create<arith::IndexCastUIOp>(loc, rewriter.getI32Type(),
                                                           totalElementsValue);
  Value rowScalarInit = rewriter.create<arith::IndexCastUIOp>(loc, rewriter.getI32Type(),
                                                              validColsValue);
  Value fullWidthCond =
      buildFullWidthColsCondition({srcCols, dstCols}, validColsValue, rewriter, loc);
  if (!fullWidthCond)
    return emitError(loc) << "unary lowering could not materialize full-width selector";

  FailureOr<pto::VecScopeOp> vecScope =
      createLoopScopeRegion(loc, contract.loopScope, rewriter);
  if (failed(vecScope))
    return emitError(loc) << "failed to create AIV vector scope region";

  OpBuilder::InsertionGuard aivGuard(rewriter);
  rewriter.setInsertionPointToStart(&(*vecScope).getBody().front());
  auto ifOp = rewriter.create<scf::IfOp>(loc, TypeRange{}, fullWidthCond,
                                         /*withElseRegion=*/true);
  rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
  {
    scf::ForOp chunkLoop;
    if (strategy == VPTOLoweringStrategy::PostUpdate) {
      chunkLoop = rewriter.create<scf::ForOp>(
          loc, c0, totalElementsValue, vectorStepValue,
          ValueRange{srcBuffer, dstBuffer, scalarInit});
    } else {
      chunkLoop = rewriter.create<scf::ForOp>(loc, c0, totalElementsValue,
                                              vectorStepValue,
                                              ValueRange{scalarInit});
    }
    rewriter.setInsertionPointToStart(chunkLoop.getBody());
    Value remaining = chunkLoop.getRegionIterArgs().back();
    PredicateMaterialization predicateState =
        buildPredicateForLaneCount(rewriter, loc, contract.elementType, remaining);
    Value loadBase = srcBuffer;
    Value storeBase = dstBuffer;
    Value loadOffset = chunkLoop.getInductionVar();
    Value storeOffset = chunkLoop.getInductionVar();
    if (strategy == VPTOLoweringStrategy::PostUpdate) {
      loadBase = chunkLoop.getRegionIterArgs()[0];
      storeBase = chunkLoop.getRegionIterArgs()[1];
      loadOffset = vectorStepValue;
      storeOffset = vectorStepValue;
    }
    Value loaded;
    Value nextSrc = {};
    if (strategy == VPTOLoweringStrategy::PostUpdate) {
      auto vlds = rewriter.create<pto::VldsPostOp>(loc, vecType, loadBase.getType(),
                                                    loadBase, loadOffset, StringAttr());
      loaded = vlds.getResult();
      nextSrc = vlds.getUpdatedSource();
    } else {
      auto vlds =
          rewriter.create<pto::VldsOp>(loc, vecType, loadBase, loadOffset, StringAttr());
      loaded = vlds.getResult();
    }
    FailureOr<Value> computed = buildUnaryValue(loaded, predicateState.mask);
    if (failed(computed))
      return emitError(loc) << "unsupported VPTO unary family: " << family;
    if (strategy == VPTOLoweringStrategy::PostUpdate) {
      auto vsts = rewriter.create<pto::VstsPostOp>(loc, storeBase.getType(), *computed,
                                                    storeBase, storeOffset, StringAttr(),
                                                    predicateState.mask);
      Value nextDst = vsts.getUpdatedDestination();
      rewriter.create<scf::YieldOp>(
          loc, ValueRange{nextSrc, nextDst, predicateState.nextScalar});
    } else {
      rewriter.create<pto::VstsOp>(loc, *computed, storeBase, storeOffset,
                                    StringAttr(), predicateState.mask);
      rewriter.create<scf::YieldOp>(loc, ValueRange{predicateState.nextScalar});
    }
  }

  rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
  {
    Value repeatUpper = rewriter.create<arith::CeilDivUIOp>(loc, validColsValue,
                                                            vectorStepValue);
    auto rowLoop = rewriter.create<scf::ForOp>(loc, c0, validRowsValue, c1);
    rewriter.setInsertionPointToStart(rowLoop.getBody());
    Value row = rowLoop.getInductionVar();
    Value srcRowBase = rewriter.create<arith::MulIOp>(loc, row, srcStrideValue);
    Value dstRowBase = rewriter.create<arith::MulIOp>(loc, row, dstStrideValue);
    auto repeatLoop = rewriter.create<scf::ForOp>(loc, c0, repeatUpper, c1,
                                                  ValueRange{rowScalarInit});
    rewriter.setInsertionPointToStart(repeatLoop.getBody());
    Value remaining = repeatLoop.getRegionIterArgs()[0];
    PredicateMaterialization predicateState =
        buildPredicateForLaneCount(rewriter, loc, contract.elementType, remaining);
    Value chunkBase =
        rewriter.create<arith::MulIOp>(loc, repeatLoop.getInductionVar(), vectorStepValue);
    Value srcOffset = rewriter.create<arith::AddIOp>(loc, srcRowBase, chunkBase);
    Value dstOffset = rewriter.create<arith::AddIOp>(loc, dstRowBase, chunkBase);
    auto loaded =
        rewriter.create<pto::VldsOp>(loc, vecType, srcBuffer, srcOffset, StringAttr());
    FailureOr<Value> computed =
        buildUnaryValue(loaded.getResult(), predicateState.mask);
    if (failed(computed))
      return emitError(loc) << "unsupported VPTO unary family: " << family;
    rewriter.create<pto::VstsOp>(loc, *computed, dstBuffer, dstOffset,
                                  StringAttr(), predicateState.mask);
    rewriter.create<scf::YieldOp>(loc, ValueRange{predicateState.nextScalar});
  }
  rewriter.setInsertionPointAfter(ifOp);

  return success();
}

LogicalResult buildBinaryVecScope(StringRef family,
                                  const VPTOBinaryContract &contract,
                                  VPTOLoweringStrategy strategy, Value src0,
                                  Value src1, Value dst,
                                  PatternRewriter &rewriter, Location loc) {
  auto vecType = getVPTOVRegType(rewriter.getContext(), contract.elementType);
  if (!vecType)
    return emitError(loc) << "unsupported VPTO binary element type";

  Value src0Buffer = materializeBufferPointer(src0, contract.elementType,
                                              getMemorySpace(src0), rewriter, loc);
  Value src1Buffer = materializeBufferPointer(src1, contract.elementType,
                                              getMemorySpace(src1), rewriter, loc);
  Value dstBuffer = materializeBufferPointer(dst, contract.elementType,
                                             getMemorySpace(dst), rewriter, loc);
  if (!src0Buffer || !src1Buffer || !dstBuffer)
    return emitError(loc) << "requires pointer-backed tile buffers for binary lowering";

  int64_t vectorWidth = vecType.getElementCount();
  Value validRowsValue = contract.validRowsValue;
  Value validColsValue = contract.validColsValue;
  int64_t validRows = contract.validRows;
  int64_t validCols = contract.validCols;
  if (failed(resolveExecutionValidShape(dst, validRowsValue, validColsValue, validRows,
                                        validCols, rewriter, loc)))
    return emitError(loc) << "binary lowering requires valid rows and cols";

  int64_t src0Stride = deriveStaticRowStride(src0);
  int64_t src1Stride = deriveStaticRowStride(src1);
  int64_t dstStride = deriveStaticRowStride(dst);
  int64_t src0Cols = deriveStaticTileCols(src0);
  int64_t src1Cols = deriveStaticTileCols(src1);
  int64_t dstCols = deriveStaticTileCols(dst);
  if (src0Stride == ShapedType::kDynamic || src1Stride == ShapedType::kDynamic ||
      dstStride == ShapedType::kDynamic || src0Cols == ShapedType::kDynamic ||
      src1Cols == ShapedType::kDynamic || dstCols == ShapedType::kDynamic)
    return emitError(loc) << "binary lowering requires static row strides and cols";

  auto buildBinaryValue = [&](Value lhs, Value rhs, Value mask) -> FailureOr<Value> {
    if (family == "add")
      return rewriter.create<pto::VaddOp>(loc, vecType, lhs, rhs, mask).getResult();
    if (family == "sub")
      return rewriter.create<pto::VsubOp>(loc, vecType, lhs, rhs, mask).getResult();
    if (family == "mul")
      return rewriter.create<pto::VmulOp>(loc, vecType, lhs, rhs, mask).getResult();
    if (family == "div")
      return rewriter.create<pto::VdivOp>(loc, vecType, lhs, rhs, mask).getResult();
    if (family == "max")
      return rewriter.create<pto::VmaxOp>(loc, vecType, lhs, rhs, mask).getResult();
    if (family == "min")
      return rewriter.create<pto::VminOp>(loc, vecType, lhs, rhs, mask).getResult();
    if (family == "and")
      return rewriter.create<pto::VandOp>(loc, vecType, lhs, rhs, mask).getResult();
    if (family == "or")
      return rewriter.create<pto::VorOp>(loc, vecType, lhs, rhs, mask).getResult();
    if (family == "xor")
      return rewriter.create<pto::VxorOp>(loc, vecType, lhs, rhs, mask).getResult();
    return failure();
  };

  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value totalElementsValue =
      rewriter.create<arith::MulIOp>(loc, validRowsValue, validColsValue);
  Value vectorStepValue =
      rewriter.create<arith::ConstantIndexOp>(loc, vectorWidth);
  Value src0StrideValue = rewriter.create<arith::ConstantIndexOp>(loc, src0Stride);
  Value src1StrideValue = rewriter.create<arith::ConstantIndexOp>(loc, src1Stride);
  Value dstStrideValue = rewriter.create<arith::ConstantIndexOp>(loc, dstStride);
  Value scalarInit = rewriter.create<arith::IndexCastUIOp>(loc, rewriter.getI32Type(),
                                                           totalElementsValue);
  Value rowScalarInit = rewriter.create<arith::IndexCastUIOp>(loc, rewriter.getI32Type(),
                                                              validColsValue);
  bool sameShapeLinearPath = src0Stride == dstStride && src1Stride == dstStride &&
                             src0Cols == dstCols && src1Cols == dstCols;
  Value fullWidthCond = buildFullWidthColsCondition(
      {src0Cols, src1Cols, dstCols}, validColsValue, rewriter, loc);
  if (!fullWidthCond)
    return emitError(loc) << "binary lowering could not materialize full-width selector";
  Value use1DCond = sameShapeLinearPath ? fullWidthCond : Value();

  FailureOr<pto::VecScopeOp> vecScope =
      createLoopScopeRegion(loc, contract.loopScope, rewriter);
  if (failed(vecScope))
    return emitError(loc) << "failed to create AIV vector scope region";

  OpBuilder::InsertionGuard aivGuard(rewriter);
  rewriter.setInsertionPointToStart(&(*vecScope).getBody().front());
  auto emit1DBody = [&]() -> LogicalResult {
    scf::ForOp chunkLoop;
    if (strategy == VPTOLoweringStrategy::PostUpdate) {
      chunkLoop = rewriter.create<scf::ForOp>(
          loc, c0, totalElementsValue, vectorStepValue,
          ValueRange{src0Buffer, src1Buffer, dstBuffer, scalarInit});
    } else {
      chunkLoop = rewriter.create<scf::ForOp>(loc, c0, totalElementsValue,
                                              vectorStepValue,
                                              ValueRange{scalarInit});
    }
    rewriter.setInsertionPointToStart(chunkLoop.getBody());
    Value remaining = chunkLoop.getRegionIterArgs().back();
    PredicateMaterialization predicateState =
        buildPredicateForLaneCount(rewriter, loc, contract.elementType, remaining);
    Value lhsBase = src0Buffer;
    Value rhsBase = src1Buffer;
    Value dstBase = dstBuffer;
    Value loadOffset = chunkLoop.getInductionVar();
    Value storeOffset = chunkLoop.getInductionVar();
    if (strategy == VPTOLoweringStrategy::PostUpdate) {
      lhsBase = chunkLoop.getRegionIterArgs()[0];
      rhsBase = chunkLoop.getRegionIterArgs()[1];
      dstBase = chunkLoop.getRegionIterArgs()[2];
      loadOffset = vectorStepValue;
      storeOffset = vectorStepValue;
    }
    Value lhsValue;
    Value rhsValue;
    Value nextSrc0 = {};
    Value nextSrc1 = {};
    if (strategy == VPTOLoweringStrategy::PostUpdate) {
      auto lhs = rewriter.create<pto::VldsPostOp>(loc, vecType, lhsBase.getType(),
                                                   lhsBase, loadOffset, StringAttr());
      auto rhs = rewriter.create<pto::VldsPostOp>(loc, vecType, rhsBase.getType(),
                                                   rhsBase, loadOffset, StringAttr());
      lhsValue = lhs.getResult();
      rhsValue = rhs.getResult();
      nextSrc0 = lhs.getUpdatedSource();
      nextSrc1 = rhs.getUpdatedSource();
    } else {
      auto lhs =
          rewriter.create<pto::VldsOp>(loc, vecType, lhsBase, loadOffset, StringAttr());
      auto rhs =
          rewriter.create<pto::VldsOp>(loc, vecType, rhsBase, loadOffset, StringAttr());
      lhsValue = lhs.getResult();
      rhsValue = rhs.getResult();
    }
    FailureOr<Value> computed = buildBinaryValue(lhsValue, rhsValue, predicateState.mask);
    if (failed(computed))
      return emitError(loc) << "unsupported VPTO binary family: " << family;
    if (strategy == VPTOLoweringStrategy::PostUpdate) {
      auto vsts = rewriter.create<pto::VstsPostOp>(loc, dstBase.getType(), *computed,
                                                    dstBase, storeOffset, StringAttr(),
                                                    predicateState.mask);
      Value nextDst = vsts.getUpdatedDestination();
      rewriter.create<scf::YieldOp>(
          loc,
          ValueRange{nextSrc0, nextSrc1, nextDst, predicateState.nextScalar});
    } else {
      rewriter.create<pto::VstsOp>(loc, *computed, dstBase, storeOffset,
                                    StringAttr(), predicateState.mask);
      rewriter.create<scf::YieldOp>(loc, ValueRange{predicateState.nextScalar});
    }
    return success();
  };

  auto emit2DBody = [&]() -> LogicalResult {
    Value repeatUpper = rewriter.create<arith::CeilDivUIOp>(loc, validColsValue,
                                                            vectorStepValue);
    auto rowLoop = rewriter.create<scf::ForOp>(loc, c0, validRowsValue, c1);
    rewriter.setInsertionPointToStart(rowLoop.getBody());
    Value row = rowLoop.getInductionVar();
    Value src0RowBase = rewriter.create<arith::MulIOp>(loc, row, src0StrideValue);
    Value src1RowBase = rewriter.create<arith::MulIOp>(loc, row, src1StrideValue);
    Value dstRowBase = rewriter.create<arith::MulIOp>(loc, row, dstStrideValue);

    if (strategy == VPTOLoweringStrategy::NoPostUpdate) {
      auto repeatLoop = rewriter.create<scf::ForOp>(loc, c0, repeatUpper, c1,
                                                    ValueRange{rowScalarInit});
      rewriter.setInsertionPointToStart(repeatLoop.getBody());
      Value remaining = repeatLoop.getRegionIterArgs()[0];
      PredicateMaterialization predicateState =
          buildPredicateForLaneCount(rewriter, loc, contract.elementType, remaining);
      Value chunkBase =
          rewriter.create<arith::MulIOp>(loc, repeatLoop.getInductionVar(), vectorStepValue);
      Value src0Offset = rewriter.create<arith::AddIOp>(loc, src0RowBase, chunkBase);
      Value src1Offset = rewriter.create<arith::AddIOp>(loc, src1RowBase, chunkBase);
      Value dstOffset = rewriter.create<arith::AddIOp>(loc, dstRowBase, chunkBase);
      auto lhs = rewriter.create<pto::VldsOp>(loc, vecType, src0Buffer, src0Offset,
                                               StringAttr());
      auto rhs = rewriter.create<pto::VldsOp>(loc, vecType, src1Buffer, src1Offset,
                                               StringAttr());
      FailureOr<Value> computed =
          buildBinaryValue(lhs.getResult(), rhs.getResult(), predicateState.mask);
      if (failed(computed))
        return emitError(loc) << "unsupported VPTO binary family: " << family;
      rewriter.create<pto::VstsOp>(loc, *computed, dstBuffer, dstOffset,
                                    StringAttr(), predicateState.mask);
      rewriter.create<scf::YieldOp>(loc, ValueRange{predicateState.nextScalar});
      return success();
    }

    auto repeatLoop = rewriter.create<scf::ForOp>(loc, c0, repeatUpper, c1);
    rewriter.setInsertionPointToStart(repeatLoop.getBody());
    Value chunkBase =
        rewriter.create<arith::MulIOp>(loc, repeatLoop.getInductionVar(), vectorStepValue);
    Value src0Offset = rewriter.create<arith::AddIOp>(loc, src0RowBase, chunkBase);
    Value src1Offset = rewriter.create<arith::AddIOp>(loc, src1RowBase, chunkBase);
    Value dstOffset = rewriter.create<arith::AddIOp>(loc, dstRowBase, chunkBase);
    Value nextChunk = rewriter.create<arith::AddIOp>(loc, chunkBase, vectorStepValue);
    Value exceeds =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, nextChunk, validColsValue);
    Value tailCount = rewriter.create<arith::SubIOp>(loc, validColsValue, chunkBase);
    Value activeLanes =
        rewriter.create<arith::SelectOp>(loc, exceeds, tailCount, vectorStepValue);
    Value predicate = buildPredicateMaskForLaneCount(rewriter, loc,
                                                     contract.elementType, activeLanes);
    auto lhs =
        rewriter.create<pto::VldsOp>(loc, vecType, src0Buffer, src0Offset, StringAttr());
    auto rhs =
        rewriter.create<pto::VldsOp>(loc, vecType, src1Buffer, src1Offset, StringAttr());
    FailureOr<Value> computed = buildBinaryValue(lhs.getResult(), rhs.getResult(), predicate);
    if (failed(computed))
      return emitError(loc) << "unsupported VPTO binary family: " << family;
    rewriter.create<pto::VstsOp>(loc, *computed, dstBuffer, dstOffset,
                                  StringAttr(), predicate);
    return success();
  };

  if (use1DCond) {
    auto ifOp = rewriter.create<scf::IfOp>(loc, TypeRange{}, use1DCond,
                                           /*withElseRegion=*/true);
    rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
    if (failed(emit1DBody()))
      return failure();
    rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
    if (failed(emit2DBody()))
      return failure();
    rewriter.setInsertionPointAfter(ifOp);
  } else {
    if (failed(emit2DBody()))
      return failure();
  }
  return success();
}

LogicalResult buildExpandScalarVecScope(const VPTOUnaryContract &contract,
                                        Value scalar, Value dst,
                                        PatternRewriter &rewriter,
                                        Location loc) {
  auto vecType = getVPTOVRegType(rewriter.getContext(), contract.elementType);
  if (!vecType)
    return emitError(loc) << "unsupported VPTO expands element type";

  Value dstBuffer = materializeBufferPointer(dst, contract.elementType,
                                             getMemorySpace(dst), rewriter, loc);
  if (!dstBuffer)
    return emitError(loc) << "requires pointer-backed tile buffer for expands lowering";

  Value validRowsValue = materializeIndexValue(contract.validRowsValue,
                                               contract.validRows, rewriter, loc);
  Value validColsValue = materializeIndexValue(contract.validColsValue,
                                               contract.validCols, rewriter, loc);
  if (!validRowsValue || !validColsValue)
    return emitError(loc) << "expands lowering requires valid rows and cols";

  int64_t vectorWidth = vecType.getElementCount();
  int64_t dstStride = deriveStaticRowStride(dst);
  int64_t dstCols = deriveStaticTileCols(dst);
  if (dstStride == ShapedType::kDynamic || dstCols == ShapedType::kDynamic)
    return emitError(loc) << "expands lowering requires static destination row stride and cols";

  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value totalElementsValue =
      rewriter.create<arith::MulIOp>(loc, validRowsValue, validColsValue);
  Value vectorStepValue =
      rewriter.create<arith::ConstantIndexOp>(loc, vectorWidth);
  Value dstStrideValue = rewriter.create<arith::ConstantIndexOp>(loc, dstStride);
  Value fullWidthCond =
      buildFullWidthColsCondition({dstCols}, validColsValue, rewriter, loc);
  if (!fullWidthCond)
    return emitError(loc) << "expands lowering could not materialize full-width selector";

  FailureOr<pto::VecScopeOp> vecScope =
      createLoopScopeRegion(loc, contract.loopScope, rewriter);
  if (failed(vecScope))
    return emitError(loc) << "failed to create AIV vector scope region";

  OpBuilder::InsertionGuard aivGuard(rewriter);
  rewriter.setInsertionPointToStart(&(*vecScope).getBody().front());
  auto ifOp = rewriter.create<scf::IfOp>(loc, TypeRange{}, fullWidthCond,
                                         /*withElseRegion=*/true);

  rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
  {
    Value scalarInit = rewriter.create<arith::IndexCastUIOp>(
        loc, rewriter.getI32Type(), totalElementsValue);
    auto chunkLoop = rewriter.create<scf::ForOp>(
        loc, c0, totalElementsValue, vectorStepValue,
        ValueRange{dstBuffer, scalarInit});
    rewriter.setInsertionPointToStart(chunkLoop.getBody());
    Value dstPtr = chunkLoop.getRegionIterArgs()[0];
    Value remaining = chunkLoop.getRegionIterArgs()[1];
    PredicateMaterialization predicateState = buildPredicateForLaneCount(
        rewriter, loc, contract.elementType, remaining);
    Value computed =
        rewriter.create<pto::VdupOp>(loc, vecType, scalar, predicateState.mask, StringAttr());
    auto vsts = rewriter.create<pto::VstsPostOp>(loc, dstPtr.getType(), computed, dstPtr,
                                                  vectorStepValue, StringAttr(),
                                                  predicateState.mask);
    Value nextDst = vsts.getUpdatedDestination();
    rewriter.create<scf::YieldOp>(
        loc, ValueRange{nextDst, predicateState.nextScalar});
  }

  rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
  {
    Value repeatUpper = rewriter.create<arith::CeilDivUIOp>(loc, validColsValue,
                                                            vectorStepValue);
    auto rowLoop = rewriter.create<scf::ForOp>(loc, c0, validRowsValue, c1);
    rewriter.setInsertionPointToStart(rowLoop.getBody());
    Value row = rowLoop.getInductionVar();
    Value rowBase = rewriter.create<arith::MulIOp>(loc, row, dstStrideValue);
    auto repeatLoop = rewriter.create<scf::ForOp>(loc, c0, repeatUpper, c1);
    rewriter.setInsertionPointToStart(repeatLoop.getBody());
    Value repeat = repeatLoop.getInductionVar();
    Value chunkBase = rewriter.create<arith::MulIOp>(loc, repeat, vectorStepValue);
    Value dstOffset = rewriter.create<arith::AddIOp>(loc, rowBase, chunkBase);
    Value remainingCols =
        rewriter.create<arith::SubIOp>(loc, validColsValue, chunkBase);
    Value activeLanes =
        buildMinIndexValue(rewriter, loc, remainingCols, vectorStepValue);
    Value predicate = buildPredicateMaskForLaneCount(
        rewriter, loc, contract.elementType, activeLanes);
    Value computed =
        rewriter.create<pto::VdupOp>(loc, vecType, scalar, predicate, StringAttr());
    rewriter.create<pto::VstsOp>(loc, computed, dstBuffer, dstOffset,
                                  StringAttr(), predicate);
  }

  rewriter.setInsertionPointAfter(ifOp);
  return success();
}

LogicalResult buildScalarUnaryVecScope(StringRef family,
                                       const VPTOUnaryContract &contract,
                                       VPTOLoweringStrategy strategy,
                                       Value src, Value scalar, Value dst,
                                       PatternRewriter &rewriter,
                                       Location loc) {
  auto vecType = getVPTOVRegType(rewriter.getContext(), contract.elementType);
  if (!vecType)
    return emitError(loc) << "unsupported VPTO scalar-unary element type";

  Value srcBuffer = materializeBufferPointer(src, contract.elementType,
                                             getMemorySpace(src), rewriter, loc);
  Value dstBuffer = materializeBufferPointer(dst, contract.elementType,
                                             getMemorySpace(dst), rewriter, loc);
  if (!srcBuffer || !dstBuffer)
    return emitError(loc)
           << "requires pointer-backed tile buffers for scalar-unary lowering";

  Value validRowsValue = materializeIndexValue(contract.validRowsValue,
                                               contract.validRows, rewriter, loc);
  Value validColsValue = materializeIndexValue(contract.validColsValue,
                                               contract.validCols, rewriter, loc);
  if (!validRowsValue || !validColsValue)
    return emitError(loc) << family << " lowering requires valid rows and cols";

  int64_t vectorWidth = vecType.getElementCount();
  int64_t srcStride = deriveStaticRowStride(src);
  int64_t dstStride = deriveStaticRowStride(dst);
  int64_t srcCols = deriveStaticTileCols(src);
  int64_t dstCols = deriveStaticTileCols(dst);
  if (srcStride == ShapedType::kDynamic || dstStride == ShapedType::kDynamic ||
      srcCols == ShapedType::kDynamic || dstCols == ShapedType::kDynamic)
    return emitError(loc)
           << family << " lowering requires static src/dst row stride and cols";

  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value totalElementsValue =
      rewriter.create<arith::MulIOp>(loc, validRowsValue, validColsValue);
  Value vectorStepValue =
      rewriter.create<arith::ConstantIndexOp>(loc, vectorWidth);
  Value srcStrideValue = rewriter.create<arith::ConstantIndexOp>(loc, srcStride);
  Value dstStrideValue = rewriter.create<arith::ConstantIndexOp>(loc, dstStride);
  Value fullWidthCond = buildFullWidthColsCondition(
      {srcCols, dstCols}, validColsValue, rewriter, loc);
  if (!fullWidthCond)
    return emitError(loc) << family << " lowering could not materialize full-width selector";

  FailureOr<pto::VecScopeOp> vecScope =
      createLoopScopeRegion(loc, contract.loopScope, rewriter);
  if (failed(vecScope))
    return emitError(loc) << "failed to create AIV vector scope region";

  OpBuilder::InsertionGuard aivGuard(rewriter);
  rewriter.setInsertionPointToStart(&(*vecScope).getBody().front());
  auto ifOp = rewriter.create<scf::IfOp>(loc, TypeRange{}, fullWidthCond,
                                         /*withElseRegion=*/true);

  rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
  {
    auto emitComputed = [&](Value loadedVec, Value predicate) -> FailureOr<Value> {
      if (family == "adds")
        return rewriter.create<pto::VaddsOp>(loc, vecType, loadedVec, scalar, predicate).getResult();
      if (family == "maxs")
        return rewriter.create<pto::VmaxsOp>(loc, vecType, loadedVec, scalar, predicate).getResult();
      if (family == "mins")
        return rewriter.create<pto::VminsOp>(loc, vecType, loadedVec, scalar, predicate).getResult();
      if (family == "muls")
        return rewriter.create<pto::VmulsOp>(loc, vecType, loadedVec, scalar, predicate).getResult();
      if (family == "lrelu")
        return rewriter.create<pto::VlreluOp>(loc, vecType, loadedVec, scalar, predicate).getResult();
      return failure();
    };

    if (strategy == VPTOLoweringStrategy::NoPostUpdate) {
      auto chunkLoop =
          rewriter.create<scf::ForOp>(loc, c0, totalElementsValue, vectorStepValue);
      rewriter.setInsertionPointToStart(chunkLoop.getBody());
      Value offset = chunkLoop.getInductionVar();
      Value remaining = rewriter.create<arith::SubIOp>(loc, totalElementsValue, offset);
      Value activeLanes =
          buildMinIndexValue(rewriter, loc, remaining, vectorStepValue);
      Value predicate = buildPredicateMaskForLaneCount(
          rewriter, loc, contract.elementType, activeLanes);
      auto loaded =
          rewriter.create<pto::VldsOp>(loc, vecType, srcBuffer, offset, StringAttr());
      FailureOr<Value> computed = emitComputed(loaded.getResult(), predicate);
      if (failed(computed))
        return emitError(loc) << "unsupported VPTO scalar-unary family: " << family;
      rewriter.create<pto::VstsOp>(loc, *computed, dstBuffer, offset, StringAttr(),
                                    predicate);
    } else {
      Value scalarInit = rewriter.create<arith::IndexCastUIOp>(
          loc, rewriter.getI32Type(), totalElementsValue);
      auto chunkLoop = rewriter.create<scf::ForOp>(
          loc, c0, totalElementsValue, vectorStepValue,
          ValueRange{srcBuffer, dstBuffer, scalarInit});
      rewriter.setInsertionPointToStart(chunkLoop.getBody());
      Value srcPtr = chunkLoop.getRegionIterArgs()[0];
      Value dstPtr = chunkLoop.getRegionIterArgs()[1];
      Value remaining = chunkLoop.getRegionIterArgs()[2];
      PredicateMaterialization predicateState = buildPredicateForLaneCount(
          rewriter, loc, contract.elementType, remaining);
      auto loaded = rewriter.create<pto::VldsPostOp>(loc, vecType, srcPtr.getType(), srcPtr,
                                                      vectorStepValue, StringAttr());
      FailureOr<Value> computed = emitComputed(loaded.getResult(), predicateState.mask);
      if (failed(computed))
        return emitError(loc) << "unsupported VPTO scalar-unary family: " << family;
      auto vsts = rewriter.create<pto::VstsPostOp>(loc, dstPtr.getType(), *computed, dstPtr,
                                                    vectorStepValue, StringAttr(),
                                                    predicateState.mask);
      Value nextSrc = loaded.getUpdatedSource();
      Value nextDst = vsts.getUpdatedDestination();
      rewriter.create<scf::YieldOp>(
          loc, ValueRange{nextSrc, nextDst, predicateState.nextScalar});
    }
  }

  rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
  {
    Value repeatUpper = rewriter.create<arith::CeilDivUIOp>(loc, validColsValue,
                                                            vectorStepValue);
    auto rowLoop = rewriter.create<scf::ForOp>(loc, c0, validRowsValue, c1);
    rewriter.setInsertionPointToStart(rowLoop.getBody());
    Value row = rowLoop.getInductionVar();
    Value srcRowBase = rewriter.create<arith::MulIOp>(loc, row, srcStrideValue);
    Value dstRowBase = rewriter.create<arith::MulIOp>(loc, row, dstStrideValue);
    auto repeatLoop = rewriter.create<scf::ForOp>(loc, c0, repeatUpper, c1);
    rewriter.setInsertionPointToStart(repeatLoop.getBody());
    Value repeat = repeatLoop.getInductionVar();
    Value chunkBase = rewriter.create<arith::MulIOp>(loc, repeat, vectorStepValue);
    Value srcOffset = rewriter.create<arith::AddIOp>(loc, srcRowBase, chunkBase);
    Value dstOffset = rewriter.create<arith::AddIOp>(loc, dstRowBase, chunkBase);
    Value predicate;
    if (strategy == VPTOLoweringStrategy::NoPostUpdate) {
      predicate =
          buildPredicateMaskForLaneCount(rewriter, loc, contract.elementType, validColsValue);
    } else {
      Value remainingCols =
          rewriter.create<arith::SubIOp>(loc, validColsValue, chunkBase);
      Value activeLanes =
          buildMinIndexValue(rewriter, loc, remainingCols, vectorStepValue);
      predicate = buildPredicateMaskForLaneCount(
          rewriter, loc, contract.elementType, activeLanes);
    }
    auto loaded =
        rewriter.create<pto::VldsOp>(loc, vecType, srcBuffer, srcOffset, StringAttr());
    Value computed;
    if (family == "adds")
      computed = rewriter.create<pto::VaddsOp>(loc, vecType, loaded.getResult(), scalar, predicate);
    else if (family == "maxs")
      computed = rewriter.create<pto::VmaxsOp>(loc, vecType, loaded.getResult(), scalar, predicate);
    else if (family == "mins")
      computed = rewriter.create<pto::VminsOp>(loc, vecType, loaded.getResult(), scalar, predicate);
    else if (family == "muls")
      computed = rewriter.create<pto::VmulsOp>(loc, vecType, loaded.getResult(), scalar, predicate);
    else if (family == "lrelu")
      computed = rewriter.create<pto::VlreluOp>(loc, vecType, loaded.getResult(), scalar, predicate);
    else
      return emitError(loc) << "unsupported VPTO scalar-unary family: " << family;
    rewriter.create<pto::VstsOp>(loc, computed, dstBuffer, dstOffset,
                                  StringAttr(), predicate);
  }

  rewriter.setInsertionPointAfter(ifOp);
  return success();
}

LogicalResult buildScalarBitwiseVecScope(StringRef family,
                                         const VPTOUnaryContract &contract,
                                         Value src, Value scalar, Value dst,
                                         PatternRewriter &rewriter,
                                         Location loc) {
  auto vecType = getVPTOVRegType(rewriter.getContext(), contract.elementType);
  if (!vecType)
    return emitError(loc) << "unsupported VPTO scalar-bitwise element type";

  Value srcBuffer = materializeBufferPointer(src, contract.elementType,
                                             getMemorySpace(src), rewriter, loc);
  Value dstBuffer = materializeBufferPointer(dst, contract.elementType,
                                             getMemorySpace(dst), rewriter, loc);
  if (!srcBuffer || !dstBuffer)
    return emitError(loc)
           << "requires pointer-backed tile buffers for scalar-bitwise lowering";

  Value validRowsValue;
  Value validColsValue;
  int64_t validRows = ShapedType::kDynamic;
  int64_t validCols = ShapedType::kDynamic;
  deriveValidShapeValues(dst, validRowsValue, validColsValue);
  deriveValidShape(dst, validRows, validCols);
  if (failed(resolveExecutionValidShape(dst, validRowsValue, validColsValue, validRows,
                                        validCols, rewriter, loc)))
    return emitError(loc) << family << " lowering requires valid rows and cols";

  int64_t vectorWidth = vecType.getElementCount();
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value totalElementsValue =
      rewriter.create<arith::MulIOp>(loc, validRowsValue, validColsValue);
  Value vectorStepValue =
      rewriter.create<arith::ConstantIndexOp>(loc, vectorWidth);
  Value vectorWidthValue =
      rewriter.create<arith::ConstantIndexOp>(loc, vectorWidth);

  FailureOr<pto::VecScopeOp> vecScope =
      createLoopScopeRegion(loc, contract.loopScope, rewriter);
  if (failed(vecScope))
    return emitError(loc) << "failed to create AIV vector scope region";

  OpBuilder::InsertionGuard aivGuard(rewriter);
  rewriter.setInsertionPointToStart(&(*vecScope).getBody().front());
  auto chunkLoop =
      rewriter.create<scf::ForOp>(loc, c0, totalElementsValue, vectorStepValue);

  OpBuilder::InsertionGuard chunkGuard(rewriter);
  rewriter.setInsertionPointToStart(chunkLoop.getBody());
  Value offset = chunkLoop.getInductionVar();
  Value remaining = rewriter.create<arith::SubIOp>(loc, totalElementsValue, offset);
  Value activeLanes =
      buildMinIndexValue(rewriter, loc, remaining, vectorWidthValue);
  Value predicate =
      buildPredicateMaskForLaneCount(rewriter, loc, contract.elementType, activeLanes);
  Value scalarVec =
      rewriter.create<pto::VdupOp>(loc, vecType, scalar, predicate, StringAttr());
  auto loaded = rewriter.create<pto::VldsOp>(loc, vecType, srcBuffer, offset,
                                              StringAttr());

  Value computed;
  if (family == "ands")
    computed =
        rewriter.create<pto::VandOp>(loc, vecType, loaded.getResult(), scalarVec, predicate);
  else if (family == "ors")
    computed =
        rewriter.create<pto::VorOp>(loc, vecType, loaded.getResult(), scalarVec, predicate);
  else if (family == "xors")
    computed =
        rewriter.create<pto::VxorOp>(loc, vecType, loaded.getResult(), scalarVec, predicate);
  else
    return emitError(loc) << "unsupported VPTO scalar-bitwise family: " << family;
  rewriter.create<pto::VstsOp>(loc, computed, dstBuffer, offset, StringAttr(),
                                predicate);
  return success();
}

static bool isVPTOShapedLikeValue(Value value) {
  Type type = value.getType();
  return isa<BaseMemRefType, RankedTensorType, pto::PartitionTensorViewType,
             pto::TileBufType>(type);
}

LogicalResult buildScalarDivVecScope(const VPTOUnaryContract &contract,
                                     VPTOLoweringStrategy strategy,
                                     Value src, Value scalar, Value dst,
                                     bool scalarFirst,
                                     PatternRewriter &rewriter, Location loc) {
  auto vecType = getVPTOVRegType(rewriter.getContext(), contract.elementType);
  if (!vecType)
    return emitError(loc) << "unsupported VPTO divs element type";

  Value srcBuffer = materializeBufferPointer(src, contract.elementType,
                                             getMemorySpace(src), rewriter, loc);
  Value dstBuffer = materializeBufferPointer(dst, contract.elementType,
                                             getMemorySpace(dst), rewriter, loc);
  if (!srcBuffer || !dstBuffer)
    return emitError(loc)
           << "requires pointer-backed tile buffers for divs lowering";

  Value validRowsValue = materializeIndexValue(contract.validRowsValue,
                                               contract.validRows, rewriter, loc);
  Value validColsValue = materializeIndexValue(contract.validColsValue,
                                               contract.validCols, rewriter, loc);
  if (!validRowsValue || !validColsValue)
    return emitError(loc) << "divs lowering requires valid rows and cols";

  int64_t vectorWidth = vecType.getElementCount();
  int64_t srcStride = deriveStaticRowStride(src);
  int64_t dstStride = deriveStaticRowStride(dst);
  int64_t srcCols = deriveStaticTileCols(src);
  int64_t dstCols = deriveStaticTileCols(dst);
  if (srcStride == ShapedType::kDynamic || dstStride == ShapedType::kDynamic ||
      srcCols == ShapedType::kDynamic || dstCols == ShapedType::kDynamic)
    return emitError(loc)
           << "divs lowering requires static src/dst row stride and cols";

  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value totalElementsValue =
      rewriter.create<arith::MulIOp>(loc, validRowsValue, validColsValue);
  Value vectorStepValue =
      rewriter.create<arith::ConstantIndexOp>(loc, vectorWidth);
  Value srcStrideValue = rewriter.create<arith::ConstantIndexOp>(loc, srcStride);
  Value dstStrideValue = rewriter.create<arith::ConstantIndexOp>(loc, dstStride);
  Value fullWidthCond = buildFullWidthColsCondition(
      {srcCols, dstCols}, validColsValue, rewriter, loc);
  if (!fullWidthCond)
    return emitError(loc) << "divs lowering could not materialize full-width selector";

  auto buildDivValue = [&](Value loaded, Value predicate) -> FailureOr<Value> {
    if (contract.elementType.isF32()) {
      if (scalarFirst) {
        Value scalarVec =
            rewriter.create<pto::VdupOp>(loc, vecType, scalar, predicate, StringAttr());
        return rewriter.create<pto::VdivOp>(loc, vecType, scalarVec, loaded, predicate)
            .getResult();
      }
      Value one = rewriter.create<arith::ConstantOp>(
          loc, contract.elementType,
          rewriter.getFloatAttr(contract.elementType, 1.0));
      Value reciprocal = rewriter.create<arith::DivFOp>(loc, one, scalar);
      return rewriter.create<pto::VmulsOp>(loc, vecType, loaded, reciprocal, predicate).getResult();
    }
    if (contract.elementType.isF16()) {
      Value scalarVec =
          rewriter.create<pto::VdupOp>(loc, vecType, scalar, predicate, StringAttr());
      return scalarFirst
                 ? rewriter.create<pto::VdivOp>(loc, vecType, scalarVec, loaded, predicate)
                       .getResult()
                 : rewriter.create<pto::VdivOp>(loc, vecType, loaded, scalarVec, predicate)
                       .getResult();
    }
    return failure();
  };

  FailureOr<pto::VecScopeOp> vecScope =
      createLoopScopeRegion(loc, contract.loopScope, rewriter);
  if (failed(vecScope))
    return emitError(loc) << "failed to create AIV vector scope region";

  OpBuilder::InsertionGuard aivGuard(rewriter);
  rewriter.setInsertionPointToStart(&(*vecScope).getBody().front());
  auto ifOp = rewriter.create<scf::IfOp>(loc, TypeRange{}, fullWidthCond,
                                         /*withElseRegion=*/true);

  rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
  {
    if (strategy == VPTOLoweringStrategy::NoPostUpdate) {
      auto chunkLoop =
          rewriter.create<scf::ForOp>(loc, c0, totalElementsValue, vectorStepValue);
      rewriter.setInsertionPointToStart(chunkLoop.getBody());
      Value offset = chunkLoop.getInductionVar();
      Value remaining = rewriter.create<arith::SubIOp>(loc, totalElementsValue, offset);
      Value activeLanes =
          buildMinIndexValue(rewriter, loc, remaining, vectorStepValue);
      Value predicate = buildPredicateMaskForLaneCount(
          rewriter, loc, contract.elementType, activeLanes);
      auto loaded =
          rewriter.create<pto::VldsOp>(loc, vecType, srcBuffer, offset, StringAttr());
      FailureOr<Value> computed = buildDivValue(loaded.getResult(), predicate);
      if (failed(computed))
        return emitError(loc)
               << "divs lowering currently supports only f16 and f32 element types";
      rewriter.create<pto::VstsOp>(loc, *computed, dstBuffer, offset, StringAttr(),
                                    predicate);
    } else {
      Value scalarInit = rewriter.create<arith::IndexCastUIOp>(
          loc, rewriter.getI32Type(), totalElementsValue);
      auto chunkLoop = rewriter.create<scf::ForOp>(
          loc, c0, totalElementsValue, vectorStepValue,
          ValueRange{srcBuffer, dstBuffer, scalarInit});
      rewriter.setInsertionPointToStart(chunkLoop.getBody());
      Value srcPtr = chunkLoop.getRegionIterArgs()[0];
      Value dstPtr = chunkLoop.getRegionIterArgs()[1];
      Value remaining = chunkLoop.getRegionIterArgs()[2];
      PredicateMaterialization predicateState = buildPredicateForLaneCount(
          rewriter, loc, contract.elementType, remaining);
      auto loaded = rewriter.create<pto::VldsPostOp>(loc, vecType, srcPtr.getType(), srcPtr,
                                                      vectorStepValue, StringAttr());
      FailureOr<Value> computed = buildDivValue(loaded.getResult(), predicateState.mask);
      if (failed(computed))
        return emitError(loc)
               << "divs lowering currently supports only f16 and f32 element types";
      auto vsts = rewriter.create<pto::VstsPostOp>(loc, dstPtr.getType(), *computed, dstPtr,
                                                    vectorStepValue, StringAttr(),
                                                    predicateState.mask);
      Value nextSrc = loaded.getUpdatedSource();
      Value nextDst = vsts.getUpdatedDestination();
      rewriter.create<scf::YieldOp>(
          loc, ValueRange{nextSrc, nextDst, predicateState.nextScalar});
    }
  }

  rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
  {
    Value repeatUpper = rewriter.create<arith::CeilDivUIOp>(loc, validColsValue,
                                                            vectorStepValue);
    auto rowLoop = rewriter.create<scf::ForOp>(loc, c0, validRowsValue, c1);
    rewriter.setInsertionPointToStart(rowLoop.getBody());
    Value row = rowLoop.getInductionVar();
    Value srcRowBase = rewriter.create<arith::MulIOp>(loc, row, srcStrideValue);
    Value dstRowBase = rewriter.create<arith::MulIOp>(loc, row, dstStrideValue);
    auto repeatLoop = rewriter.create<scf::ForOp>(loc, c0, repeatUpper, c1);
    rewriter.setInsertionPointToStart(repeatLoop.getBody());
    Value repeat = repeatLoop.getInductionVar();
    Value chunkBase = rewriter.create<arith::MulIOp>(loc, repeat, vectorStepValue);
    Value srcOffset = rewriter.create<arith::AddIOp>(loc, srcRowBase, chunkBase);
    Value dstOffset = rewriter.create<arith::AddIOp>(loc, dstRowBase, chunkBase);
    Value predicate;
    if (strategy == VPTOLoweringStrategy::NoPostUpdate) {
      predicate =
          buildPredicateMaskForLaneCount(rewriter, loc, contract.elementType, validColsValue);
    } else {
      Value remainingCols =
          rewriter.create<arith::SubIOp>(loc, validColsValue, chunkBase);
      Value activeLanes =
          buildMinIndexValue(rewriter, loc, remainingCols, vectorStepValue);
      predicate = buildPredicateMaskForLaneCount(
          rewriter, loc, contract.elementType, activeLanes);
    }
    auto loaded =
        rewriter.create<pto::VldsOp>(loc, vecType, srcBuffer, srcOffset, StringAttr());
    FailureOr<Value> computed = buildDivValue(loaded.getResult(), predicate);
    if (failed(computed))
      return emitError(loc)
             << "divs lowering currently supports only f16 and f32 element types";
    rewriter.create<pto::VstsOp>(loc, *computed, dstBuffer, dstOffset,
                                  StringAttr(), predicate);
  }

  rewriter.setInsertionPointAfter(ifOp);
  return success();
}

LogicalResult checkExpandContract(Operation *op,
                                  const VPTOExpandContract &contract) {
  bool hasPrecheckFailure = false;
  if (contract.srcDomain != VPTOTileDomain::Vec ||
      contract.dstDomain != VPTOTileDomain::Vec) {
    op->emitOpError() << contract.family
                      << " lowering requires vec source and destination";
    hasPrecheckFailure = true;
  }
  if (contract.srcLayout != "row_major" || contract.dstLayout != "row_major") {
    op->emitOpError() << contract.family
                      << " lowering requires row-major source and destination tile layout";
    hasPrecheckFailure = true;
  }
  if (!contract.elementType ||
      (!contract.elementType.isF16() && !contract.elementType.isF32())) {
    op->emitOpError() << contract.family
                      << " lowering currently supports only f16 and f32 element types";
    hasPrecheckFailure = true;
  }
  auto isStatic = [](int64_t value) { return value != ShapedType::kDynamic; };
  if (!isStatic(contract.srcValidRows) || !isStatic(contract.srcValidCols) ||
      !isStatic(contract.dstValidRows) || !isStatic(contract.dstValidCols)) {
    op->emitOpError() << contract.family
                      << " lowering currently requires static source and destination valid shapes";
    hasPrecheckFailure = true;
  }
  return failure(hasPrecheckFailure);
}

LogicalResult buildRowExpandVecScope(const VPTOExpandContract &contract,
                                     VPTOLoweringStrategy strategy, Value src, Value dst,
                                     PatternRewriter &rewriter, Location loc) {
  auto vecType = getVPTOVRegType(rewriter.getContext(), contract.elementType);
  if (!vecType)
    return emitError(loc) << "unsupported VPTO rowexpand element type";

  Value srcBuffer = materializeBufferPointer(src, contract.elementType,
                                             getMemorySpace(src), rewriter, loc);
  Value dstBuffer = materializeBufferPointer(dst, contract.elementType,
                                             getMemorySpace(dst), rewriter, loc);
  if (!srcBuffer || !dstBuffer)
    return emitError(loc)
           << "requires pointer-backed tile buffers for rowexpand lowering";

  auto [srcRows, srcCols] = getStaticTileRowsCols(src);
  auto [dstRows, dstCols] = getStaticTileRowsCols(dst);
  if (srcCols == ShapedType::kDynamic || dstCols == ShapedType::kDynamic ||
      srcRows == ShapedType::kDynamic || dstRows == ShapedType::kDynamic)
    return emitError(loc) << "rowexpand lowering requires static physical tile shape";

  int64_t vectorWidth = vecType.getElementCount();
  Value validRowsValue = materializeIndexValue(
      contract.dstValidRowsValue, contract.dstValidRows, rewriter, loc);
  Value validColsValue = materializeIndexValue(
      contract.dstValidColsValue, contract.dstValidCols, rewriter, loc);
  if (!validRowsValue || !validColsValue)
    return emitError(loc) << "rowexpand lowering requires valid rows and cols";

  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value srcStrideValue = rewriter.create<arith::ConstantIndexOp>(loc, srcCols);
  Value dstStrideValue = rewriter.create<arith::ConstantIndexOp>(loc, dstCols);
  Value vectorStepValue =
      rewriter.create<arith::ConstantIndexOp>(loc, vectorWidth);
  Value rowScalarInit = rewriter.create<arith::IndexCastUIOp>(loc, rewriter.getI32Type(),
                                                              validColsValue);
  FailureOr<pto::VecScopeOp> vecScope =
      createLoopScopeRegion(loc, contract.loopScope, rewriter);
  if (failed(vecScope))
    return emitError(loc) << "failed to create AIV vector scope region";

  OpBuilder::InsertionGuard aivGuard(rewriter);
  rewriter.setInsertionPointToStart(&(*vecScope).getBody().front());
  Value repeatUpper = rewriter.create<arith::CeilDivUIOp>(loc, validColsValue,
                                                          vectorStepValue);
  if (strategy == VPTOLoweringStrategy::NoPostUpdate) {
    auto rowLoop = rewriter.create<scf::ForOp>(loc, c0, validRowsValue, c1);
    rewriter.setInsertionPointToStart(rowLoop.getBody());
    Value row = rowLoop.getInductionVar();
    Value srcOffset = rewriter.create<arith::MulIOp>(loc, row, srcStrideValue);
    Value dstBase = rewriter.create<arith::MulIOp>(loc, row, dstStrideValue);
    auto loaded =
        rewriter.create<pto::VldsOp>(loc, vecType, srcBuffer, srcOffset, StringAttr());
    Value fullMask = buildAllPredicateMask(rewriter, loc, contract.elementType);
    Value expanded = rewriter.create<pto::VdupOp>(
        loc, vecType, loaded.getResult(), fullMask, rewriter.getStringAttr("LOWEST"));
    auto chunkLoop = rewriter.create<scf::ForOp>(loc, c0, repeatUpper, c1,
                                                 ValueRange{rowScalarInit});
    rewriter.setInsertionPointToStart(chunkLoop.getBody());
    Value remaining = chunkLoop.getRegionIterArgs()[0];
    PredicateMaterialization predicateState =
        buildPredicateForLaneCount(rewriter, loc, contract.elementType, remaining);
    Value chunkBase =
        rewriter.create<arith::MulIOp>(loc, chunkLoop.getInductionVar(), vectorStepValue);
    Value dstOffset = rewriter.create<arith::AddIOp>(loc, dstBase, chunkBase);
    rewriter.create<pto::VstsOp>(loc, expanded, dstBuffer, dstOffset, StringAttr(),
                                  predicateState.mask);
    rewriter.create<scf::YieldOp>(loc, ValueRange{predicateState.nextScalar});
    return success();
  }

  auto rowLoop =
      rewriter.create<scf::ForOp>(loc, c0, validRowsValue, c1, ValueRange{srcBuffer, dstBuffer});
  rewriter.setInsertionPointToStart(rowLoop.getBody());
  Value srcPtr = rowLoop.getRegionIterArgs()[0];
  Value dstPtr = rowLoop.getRegionIterArgs()[1];
  auto loaded = rewriter.create<pto::VldsPostOp>(loc, vecType, srcPtr.getType(), srcPtr,
                                                  srcStrideValue, StringAttr());
  Value fullMask = buildAllPredicateMask(rewriter, loc, contract.elementType);
  Value expanded = rewriter.create<pto::VdupOp>(
      loc, vecType, loaded.getResult(), fullMask, rewriter.getStringAttr("LOWEST"));
  auto chunkLoop = rewriter.create<scf::ForOp>(loc, c0, repeatUpper, c1,
                                               ValueRange{dstPtr, rowScalarInit});
  rewriter.setInsertionPointToStart(chunkLoop.getBody());
  Value dstChunkPtr = chunkLoop.getRegionIterArgs()[0];
  Value remaining = chunkLoop.getRegionIterArgs()[1];
  PredicateMaterialization predicateState =
      buildPredicateForLaneCount(rewriter, loc, contract.elementType, remaining);
  auto vsts = rewriter.create<pto::VstsPostOp>(loc, dstChunkPtr.getType(), expanded,
                                                dstChunkPtr, vectorStepValue, StringAttr(),
                                                predicateState.mask);
  Value nextDstChunkPtr = vsts.getUpdatedDestination();
  rewriter.create<scf::YieldOp>(loc, ValueRange{nextDstChunkPtr, predicateState.nextScalar});

  rewriter.setInsertionPointAfter(chunkLoop);
  Value rowAdvance = rewriter.create<arith::MulIOp>(loc, repeatUpper, vectorStepValue);
  Value dstPad = rewriter.create<arith::SubIOp>(loc, dstStrideValue, rowAdvance);
  Value nextDstPtr =
      offsetBufferPointer(dstPtr, contract.elementType, dstPad, rewriter, loc);
  Value nextSrcPtr = loaded.getUpdatedSource();
  rewriter.create<scf::YieldOp>(loc, ValueRange{nextSrcPtr, nextDstPtr});
  return success();
}

LogicalResult buildColExpandVecScope(const VPTOExpandContract &contract,
                                     Value src, Value dst,
                                     PatternRewriter &rewriter, Location loc) {
  auto vecType = getVPTOVRegType(rewriter.getContext(), contract.elementType);
  if (!vecType)
    return emitError(loc) << "unsupported VPTO colexpand element type";

  Value srcBuffer = materializeBufferPointer(src, contract.elementType,
                                             getMemorySpace(src), rewriter, loc);
  Value dstBuffer = materializeBufferPointer(dst, contract.elementType,
                                             getMemorySpace(dst), rewriter, loc);
  if (!srcBuffer || !dstBuffer)
    return emitError(loc)
           << "requires pointer-backed tile buffers for colexpand lowering";

  auto [dstRows, dstCols] = getStaticTileRowsCols(dst);
  if (dstRows == ShapedType::kDynamic || dstCols == ShapedType::kDynamic)
    return emitError(loc)
           << "colexpand lowering requires static physical destination tile shape";

  int64_t vectorWidth = vecType.getElementCount();
  Value validRowsValue = materializeIndexValue(
      contract.dstValidRowsValue, contract.dstValidRows, rewriter, loc);
  Value validColsValue = materializeIndexValue(
      contract.dstValidColsValue, contract.dstValidCols, rewriter, loc);
  if (!validRowsValue || !validColsValue)
    return emitError(loc) << "colexpand lowering requires valid rows and cols";

  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value dstStrideValue = rewriter.create<arith::ConstantIndexOp>(loc, dstCols);
  Value vectorStepValue =
      rewriter.create<arith::ConstantIndexOp>(loc, vectorWidth);
  FailureOr<pto::VecScopeOp> vecScope =
      createLoopScopeRegion(loc, contract.loopScope, rewriter);
  if (failed(vecScope))
    return emitError(loc) << "failed to create AIV vector scope region";

  OpBuilder::InsertionGuard aivGuard(rewriter);
  rewriter.setInsertionPointToStart(&(*vecScope).getBody().front());
  auto rowLoop = rewriter.create<scf::ForOp>(loc, c0, validRowsValue, c1);
  rewriter.setInsertionPointToStart(rowLoop.getBody());
  auto chunkLoop =
      rewriter.create<scf::ForOp>(loc, c0, validColsValue, vectorStepValue);
  rewriter.setInsertionPointToStart(chunkLoop.getBody());

  Value dstBase =
      rewriter.create<arith::MulIOp>(loc, rowLoop.getInductionVar(), dstStrideValue);
  Value dstOffset =
      rewriter.create<arith::AddIOp>(loc, dstBase, chunkLoop.getInductionVar());
  auto loaded = rewriter.create<pto::VldsOp>(
      loc, vecType, srcBuffer, chunkLoop.getInductionVar(), StringAttr());
  rewriter.create<pto::VstsOp>(loc, loaded.getResult(), dstBuffer, dstOffset,
                                StringAttr(),
                                buildAllPredicateMask(rewriter, loc,
                                                      contract.elementType));
  return success();
}

LogicalResult checkGenericUnaryContract(Operation *op,
                                        const VPTOUnaryContract &contract,
                                        Value dst,
                                        function_ref<bool(Type)> typePredicate,
                                        StringRef supportedTypeText) {
  int64_t dstRows = ShapedType::kDynamic;
  int64_t dstCols = ShapedType::kDynamic;
  deriveValidShape(dst, dstRows, dstCols);
  StringRef dstLayout = deriveTileLayout(dst);
  VPTOTileDomain dstDomain = deriveTileDomain(getMemorySpace(dst));

  bool hasPrecheckFailure = false;
  if (contract.tileDomain != VPTOTileDomain::Vec || dstDomain != VPTOTileDomain::Vec) {
    op->emitOpError() << contract.family << " lowering requires tile domain vec";
    hasPrecheckFailure = true;
  }
  if (contract.tileLayout != "row_major" || dstLayout != "row_major") {
    op->emitOpError() << contract.family << " lowering requires row-major tile layout";
    hasPrecheckFailure = true;
  }
  if (contract.validRows != ShapedType::kDynamic &&
      dstRows != ShapedType::kDynamic && dstRows > contract.validRows) {
    op->emitOpError() << contract.family
                      << " lowering requires destination valid rows not to exceed source";
    hasPrecheckFailure = true;
  }
  if (contract.validCols != ShapedType::kDynamic &&
      dstCols != ShapedType::kDynamic && dstCols > contract.validCols) {
    op->emitOpError() << contract.family
                      << " lowering requires destination valid cols not to exceed source";
    hasPrecheckFailure = true;
  }
  if (!contract.elementType || !typePredicate(contract.elementType)) {
    op->emitOpError()
        << contract.family << " lowering supports only " << supportedTypeText;
    hasPrecheckFailure = true;
  }
  return failure(hasPrecheckFailure);
}

LogicalResult checkGenericBinaryContract(
    Operation *op, const VPTOBinaryContract &contract, Value src1, Value dst,
    function_ref<bool(Type)> typePredicate, StringRef supportedTypeText) {
  StringRef src1Layout = deriveTileLayout(src1);
  StringRef dstLayout = deriveTileLayout(dst);
  VPTOTileDomain src1Domain = deriveTileDomain(getMemorySpace(src1));
  VPTOTileDomain dstDomain = deriveTileDomain(getMemorySpace(dst));

  bool hasPrecheckFailure = false;
  if (contract.tileDomain != VPTOTileDomain::Vec || src1Domain != VPTOTileDomain::Vec ||
      dstDomain != VPTOTileDomain::Vec) {
    op->emitOpError() << contract.family << " lowering requires tile domain vec";
    hasPrecheckFailure = true;
  }
  if (contract.tileLayout != "row_major" || src1Layout != "row_major" ||
      dstLayout != "row_major") {
    op->emitOpError() << contract.family << " lowering requires row-major tile layout";
    hasPrecheckFailure = true;
  }
  if (!contract.elementType || !typePredicate(contract.elementType)) {
    op->emitOpError()
        << contract.family << " lowering supports only " << supportedTypeText;
    hasPrecheckFailure = true;
  }
  return failure(hasPrecheckFailure);
}

LogicalResult checkRowReduceContract(Operation *op,
                                     const VPTORowReduceContract &contract,
                                     Value dst) {
  int64_t dstRows = ShapedType::kDynamic;
  int64_t dstCols = ShapedType::kDynamic;
  deriveValidShape(dst, dstRows, dstCols);

  bool hasPrecheckFailure = false;
  if (contract.srcDomain != VPTOTileDomain::Vec ||
      contract.dstDomain != VPTOTileDomain::Vec) {
    op->emitOpError() << contract.family << " lowering requires vec source and destination";
    hasPrecheckFailure = true;
  }
  if (contract.srcLayout != "row_major") {
    op->emitOpError() << contract.family << " lowering requires row-major source tile layout";
    hasPrecheckFailure = true;
  }
  if (contract.dstLayout != "row_major" && contract.dstLayout != "col_major") {
    op->emitOpError() << contract.family
                      << " lowering requires row-major or col-major destination tile layout";
    hasPrecheckFailure = true;
  }
  if (!contract.elementType || (!contract.elementType.isF16() && !contract.elementType.isF32())) {
    op->emitOpError() << contract.family << " lowering supports only f16 and f32 element types";
    hasPrecheckFailure = true;
  }
  if (contract.validRows == ShapedType::kDynamic ||
      contract.validCols == ShapedType::kDynamic) {
    op->emitOpError() << contract.family
                      << " lowering currently requires static source valid rows and cols";
    hasPrecheckFailure = true;
  }
  if (contract.validRows != dstRows) {
    op->emitOpError() << contract.family
                      << " lowering requires destination valid rows to match source valid rows";
    hasPrecheckFailure = true;
  }
  if (dstCols != 1) {
    op->emitOpError() << contract.family
                      << " lowering requires destination valid cols to equal 1";
    hasPrecheckFailure = true;
  }
  if (contract.dstLayout == "col_major") {
    auto [dstRowsPhysical, dstColsPhysical] = getStaticTileRowsCols(dst);
    (void)dstRowsPhysical;
    if (dstColsPhysical != 1) {
      op->emitOpError() << contract.family
                        << " lowering requires col-major destinations to use physical cols == 1";
      hasPrecheckFailure = true;
    }
  }
  return failure(hasPrecheckFailure);
}

LogicalResult checkColReduceContract(Operation *op,
                                     const VPTOColReduceContract &contract,
                                     Value dst) {
  int64_t dstRows = ShapedType::kDynamic;
  int64_t dstCols = ShapedType::kDynamic;
  deriveValidShape(dst, dstRows, dstCols);

  bool hasPrecheckFailure = false;
  if (contract.srcDomain != VPTOTileDomain::Vec ||
      contract.dstDomain != VPTOTileDomain::Vec) {
    op->emitOpError() << contract.family << " lowering requires vec source and destination";
    hasPrecheckFailure = true;
  }
  if (contract.srcLayout != "row_major" || contract.dstLayout != "row_major") {
    op->emitOpError() << contract.family
                      << " lowering requires row-major source and destination tile layout";
    hasPrecheckFailure = true;
  }
  if (!contract.elementType ||
      (!contract.elementType.isF16() && !contract.elementType.isF32())) {
    op->emitOpError() << contract.family << " lowering supports only f16 and f32 element types";
    hasPrecheckFailure = true;
  }
  if (contract.validRows == ShapedType::kDynamic ||
      contract.validCols == ShapedType::kDynamic) {
    op->emitOpError() << contract.family
                      << " lowering currently requires static source valid rows and cols";
    hasPrecheckFailure = true;
  }
  if (dstRows != 1) {
    op->emitOpError() << contract.family
                      << " lowering requires destination valid rows to equal 1";
    hasPrecheckFailure = true;
  }
  if (dstCols != contract.validCols) {
    op->emitOpError() << contract.family
                      << " lowering requires destination valid cols to match source valid cols";
    hasPrecheckFailure = true;
  }
  if (contract.isBinary && !contract.tmp) {
    op->emitOpError() << contract.family << " lowering requires tmp for binary path";
    hasPrecheckFailure = true;
  }
  return failure(hasPrecheckFailure);
}

LogicalResult checkPartContract(Operation *op, const VPTOPartContract &contract) {
  bool hasPrecheckFailure = false;
  if (contract.src0Domain != VPTOTileDomain::Vec ||
      contract.src1Domain != VPTOTileDomain::Vec ||
      contract.dstDomain != VPTOTileDomain::Vec) {
    op->emitOpError() << contract.family << " lowering requires vec source and destination";
    hasPrecheckFailure = true;
  }
  if (contract.src0Layout != "row_major" || contract.src1Layout != "row_major" ||
      contract.dstLayout != "row_major") {
    op->emitOpError() << contract.family
                      << " lowering requires row-major source and destination tile layout";
    hasPrecheckFailure = true;
  }
  if (!contract.elementType)
    hasPrecheckFailure = true;
  else if (contract.family == "partadd") {
    bool ok = contract.elementType.isF16() || contract.elementType.isF32() ||
              contract.elementType.isBF16();
    if (auto intType = dyn_cast<IntegerType>(contract.elementType))
      ok = intType.getWidth() == 8 || intType.getWidth() == 16 ||
           intType.getWidth() == 32;
    if (!ok) {
      op->emitOpError() << contract.family
                        << " lowering supports f16, f32, bf16, and 8/16/32-bit integers";
      hasPrecheckFailure = true;
    }
  } else {
    bool ok = contract.elementType.isF16() || contract.elementType.isF32() ||
              contract.elementType.isBF16();
    if (auto intType = dyn_cast<IntegerType>(contract.elementType))
      ok = intType.getWidth() == 8 || intType.getWidth() == 16 ||
           intType.getWidth() == 32;
    if (!ok) {
      op->emitOpError() << contract.family
                        << " lowering supports f16, f32, bf16, and 8/16/32-bit integers";
      hasPrecheckFailure = true;
    }
  }
  auto allStatic = [&](int64_t a, int64_t b) {
    return a != ShapedType::kDynamic && b != ShapedType::kDynamic;
  };
  if (!allStatic(contract.src0ValidRows, contract.src0ValidCols) ||
      !allStatic(contract.src1ValidRows, contract.src1ValidCols) ||
      !allStatic(contract.dstValidRows, contract.dstValidCols)) {
    op->emitOpError() << contract.family
                      << " lowering currently requires static source and destination valid shapes";
    hasPrecheckFailure = true;
  }
  return failure(hasPrecheckFailure);
}

LogicalResult lowerTLOAD(TLoadOp op, PatternRewriter &rewriter) {
  VPTOLoadContract contract = extractTLoadContract(op);
  if (contract.tileDomain != VPTOTileDomain::Vec)
    return op.emitOpError("currently supports only VEC TLOAD lowering");

  ResolvedTensorView sourceView;
  if (!resolveTensorView(op.getSrc(), sourceView, rewriter, op.getLoc()))
    return op.emitOpError("requires a recoverable source tensor view for VPTO lowering");

  StringRef sourceLayout =
      inferVecTransferLayoutFromTile(stringifyLayoutAttr(sourceView.layoutAttr),
                                     contract.tileLayout);
  bool isNdLoad = contract.tileLayout == "row_major" && sourceLayout == "nd";
  bool isDnLoad = contract.tileLayout == "col_major" && sourceLayout == "dn";
  if (!isNdLoad && !isDnLoad)
    return op.emitOpError("currently supports only ND row_major or DN col_major vec TLOAD lowering");

  Value sourceBuffer =
      materializeBufferPointer(sourceView.root, getElementType(sourceView.root),
                               getGmMemorySpace(rewriter.getContext()), rewriter,
                               op.getLoc());
  Value destinationBuffer =
      materializeBufferPointer(op.getDst(), contract.elementType,
                               getMemorySpace(op.getDst()), rewriter, op.getLoc());
  if (!sourceBuffer || !destinationBuffer)
    return op.emitOpError("requires A5-compatible source and destination buffers");

  auto [tileRows, tileCols] = getStaticTileRowsCols(op.getDst());
  (void)tileRows;
  bool ubPad = contract.padMode != "none" || contract.padValue ||
               contract.leftPaddingNum || contract.rightPaddingNum;
  Value validRowsValue =
      materializeI64Value(contract.validRowsValue, contract.validRows, rewriter,
                          op.getLoc());
  Value validColsValue =
      materializeI64Value(contract.validColsValue, contract.validCols, rewriter,
                          op.getLoc());
  Value sidValue = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0, 64);
  int64_t elemBytes = getElementByteSize(contract.elementType);
  if ((isNdLoad && tileCols == ShapedType::kDynamic) ||
      (isDnLoad && tileRows == ShapedType::kDynamic) || elemBytes <= 0)
    return op.emitOpError("requires static tile shape for A5-compatible transfer arguments");
  VecNdTransferPlan plan;
  LogicalResult planResult =
      isNdLoad ? buildVecNdLoadPlan(sourceView.shape, sourceView.strides, tileCols,
                                    contract.validColsValue, contract.validCols,
                                    contract.elementType, rewriter, op.getLoc(), plan)
               : buildVecDnLoadPlan(sourceView.shape, sourceView.strides, tileRows,
                                    contract.validRowsValue, contract.validRows,
                                    contract.elementType, rewriter, op.getLoc(), plan);
  if (failed(planResult))
    return op.emitOpError("requires PTO-compatible vec copy_gm_to_ubuf arguments");
  Value leftPaddingValue = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0, 64);
  Value rightPaddingValue = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0, 64);
  Value cacheCtlValue = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0, 64);
  if (!validRowsValue || !validColsValue)
    return op.emitOpError("requires valid rows and cols for A5-compatible transfer arguments");
  Value sourceOffset =
      materializeI64Ofr(sourceView.offsetElems, rewriter, op.getLoc());
  if (!sourceOffset)
    return op.emitOpError("requires a materializable source offset for VPTO lowering");
  Value sourceBase = adjustPointerByElemOffset(sourceBuffer, sourceOffset, elemBytes,
                                              rewriter, op.getLoc());
  if (!sourceBase)
    return op.emitOpError("failed to materialize source base pointer");

  rewriter.create<pto::SetLoop2StrideOutToUbOp>(
      op.getLoc(), plan.loop2FirstStrideBytes, plan.loop2SecondStrideBytes);
  rewriter.create<pto::SetLoop1StrideOutToUbOp>(
      op.getLoc(), plan.loop1FirstStrideBytes, plan.loop1SecondStrideBytes);
  rewriter.create<pto::SetLoopSizeOutToUbOp>(op.getLoc(), plan.loop2Size,
                                              plan.loop1Size);

  auto emitCopy = [&](Value srcPtr, Value dstPtr) {
    Type transferElementType =
        getCopyTransferElementType(contract.elementType, rewriter);
    Value typedSrcPtr =
        castPtrToElementType(srcPtr, transferElementType, rewriter, op.getLoc());
    Value typedDstPtr =
        castPtrToElementType(dstPtr, transferElementType, rewriter, op.getLoc());
    if (!typedSrcPtr || !typedDstPtr)
      return failure();
    Value dataSelectBitValue =
        rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI1Type(),
                                           rewriter.getBoolAttr(ubPad));
    rewriter.create<pto::CopyGmToUbufOp>(
        op.getLoc(), typedSrcPtr, typedDstPtr, sidValue, plan.nBurst,
        plan.lenBurst, leftPaddingValue, rightPaddingValue, dataSelectBitValue,
        cacheCtlValue, plan.firstStrideBytes, plan.secondStrideBytes);
    return success();
  };

  if (std::optional<int64_t> outerConst = getConstInt(plan.outerCount); outerConst && *outerConst == 1) {
    return emitCopy(sourceBase, destinationBuffer);
  }

  Value c0 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
  Value outerUpper =
      rewriter.create<arith::IndexCastUIOp>(op.getLoc(), rewriter.getIndexType(),
                                            plan.outerCount);
  auto outerLoop = rewriter.create<scf::ForOp>(op.getLoc(), c0, outerUpper, c1);
  rewriter.setInsertionPointToStart(outerLoop.getBody());
  Value ivI64 = rewriter.create<arith::IndexCastUIOp>(op.getLoc(), rewriter.getI64Type(),
                                                      outerLoop.getInductionVar());
  Value srcStep = createI64Mul(ivI64, plan.outerSrcStrideElems, rewriter, op.getLoc());
  Value dstStep = createI64Mul(ivI64, plan.outerDstStrideElems, rewriter, op.getLoc());
  Value iterSrc = adjustPointerByElemOffset(sourceBase, srcStep, elemBytes, rewriter,
                                            op.getLoc());
  Value iterDst = adjustPointerByElemOffset(destinationBuffer, dstStep, elemBytes, rewriter,
                                            op.getLoc());
  return emitCopy(iterSrc, iterDst);
}

LogicalResult lowerTABS(TAbsOp op, PatternRewriter &rewriter,
                        VPTOLoweringStrategy strategy) {
  VPTOUnaryContract contract = extractTAbsContract(op);
  if (failed(checkGenericUnaryContract(
          op, contract, op.getDst(),
          [](Type type) { return type.isF16() || type.isF32(); }, "f16 and f32 element types")))
    return failure();

  return buildUnaryVecScope("abs", contract, strategy, op.getSrc(), op.getDst(),
                            rewriter, op.getLoc());
}

LogicalResult lowerTADD(TAddOp op, PatternRewriter &rewriter,
                        VPTOLoweringStrategy strategy) {
  VPTOBinaryContract contract = extractTAddContract(op);
  deriveValidShapeValues(op.getDst(), contract.validRowsValue, contract.validColsValue);
  deriveValidShape(op.getDst(), contract.validRows, contract.validCols);
  if (failed(checkGenericBinaryContract(
          op, contract, op.getSrc1(), op.getDst(),
          [](Type type) {
            if (type.isF16() || type.isF32() || type.isBF16())
              return true;
            if (auto intType = dyn_cast<IntegerType>(type))
              return intType.getWidth() == 8 || intType.getWidth() == 16 ||
                     intType.getWidth() == 32;
            return false;
          },
          "f16, f32, bf16, and 8/16/32-bit integer element types")))
    return failure();
  return buildBinaryVecScope("add", contract, strategy, op.getSrc0(),
                             op.getSrc1(), op.getDst(), rewriter, op.getLoc());
}

LogicalResult lowerTSUB(TSubOp op, PatternRewriter &rewriter,
                        VPTOLoweringStrategy strategy) {
  VPTOBinaryContract contract = extractTSubContract(op);
  deriveValidShapeValues(op.getDst(), contract.validRowsValue, contract.validColsValue);
  deriveValidShape(op.getDst(), contract.validRows, contract.validCols);
  if (failed(checkGenericBinaryContract(
          op, contract, op.getSrc1(), op.getDst(),
          [](Type type) {
            if (type.isF16() || type.isF32() || type.isBF16())
              return true;
            if (auto intType = dyn_cast<IntegerType>(type))
              return intType.getWidth() == 8 || intType.getWidth() == 16 ||
                     intType.getWidth() == 32;
            return false;
          },
          "f16, f32, bf16, and 8/16/32-bit integer element types")))
    return failure();
  return buildBinaryVecScope("sub", contract, strategy, op.getSrc0(),
                             op.getSrc1(), op.getDst(), rewriter, op.getLoc());
}

LogicalResult lowerTMUL(TMulOp op, PatternRewriter &rewriter,
                        VPTOLoweringStrategy strategy) {
  VPTOBinaryContract contract = extractTMulContract(op);
  deriveValidShapeValues(op.getDst(), contract.validRowsValue, contract.validColsValue);
  deriveValidShape(op.getDst(), contract.validRows, contract.validCols);
  if (failed(checkGenericBinaryContract(
          op, contract, op.getSrc1(), op.getDst(),
          [](Type type) {
            if (type.isF16() || type.isF32() || type.isBF16())
              return true;
            if (auto intType = dyn_cast<IntegerType>(type))
              return intType.getWidth() == 8 || intType.getWidth() == 16 ||
                     intType.getWidth() == 32;
            return false;
          },
          "f16, f32, bf16, and 8/16/32-bit integer element types")))
    return failure();
  return buildBinaryVecScope("mul", contract, strategy, op.getSrc0(),
                             op.getSrc1(), op.getDst(), rewriter, op.getLoc());
}

LogicalResult lowerTDIV(TDivOp op, PatternRewriter &rewriter,
                        VPTOLoweringStrategy strategy) {
  VPTOBinaryContract contract = extractTDivContract(op);
  deriveValidShapeValues(op.getDst(), contract.validRowsValue, contract.validColsValue);
  deriveValidShape(op.getDst(), contract.validRows, contract.validCols);
  if (failed(checkGenericBinaryContract(
          op, contract, op.getSrc1(), op.getDst(),
          [](Type type) {
            if (type.isF16() || type.isF32())
              return true;
            if (auto intType = dyn_cast<IntegerType>(type))
              return intType.getWidth() == 16 || intType.getWidth() == 32;
            return false;
          },
          "f16, f32, and 16/32-bit integer element types")))
    return failure();
  return buildBinaryVecScope("div", contract, strategy, op.getSrc0(),
                             op.getSrc1(), op.getDst(), rewriter, op.getLoc());
}

LogicalResult lowerTMAX(TMaxOp op, PatternRewriter &rewriter,
                        VPTOLoweringStrategy strategy) {
  VPTOBinaryContract contract = buildBinaryContract("max", op.getSrc0());
  deriveValidShapeValues(op.getDst(), contract.validRowsValue, contract.validColsValue);
  deriveValidShape(op.getDst(), contract.validRows, contract.validCols);
  if (failed(checkGenericBinaryContract(
          op, contract, op.getSrc1(), op.getDst(),
          [](Type type) {
            if (type.isF16() || type.isF32() || type.isBF16())
              return true;
            if (auto intType = dyn_cast<IntegerType>(type))
              return intType.getWidth() == 8 || intType.getWidth() == 16 ||
                     intType.getWidth() == 32;
            return false;
          },
          "f16, f32, bf16, and 8/16/32-bit integer element types")))
    return failure();
  return buildBinaryVecScope("max", contract, strategy, op.getSrc0(),
                             op.getSrc1(), op.getDst(), rewriter, op.getLoc());
}

LogicalResult lowerTMIN(TMinOp op, PatternRewriter &rewriter,
                        VPTOLoweringStrategy strategy) {
  VPTOBinaryContract contract = buildBinaryContract("min", op.getSrc0());
  deriveValidShapeValues(op.getDst(), contract.validRowsValue, contract.validColsValue);
  deriveValidShape(op.getDst(), contract.validRows, contract.validCols);
  if (failed(checkGenericBinaryContract(
          op, contract, op.getSrc1(), op.getDst(),
          [](Type type) {
            if (type.isF16() || type.isF32() || type.isBF16())
              return true;
            if (auto intType = dyn_cast<IntegerType>(type))
              return intType.getWidth() == 8 || intType.getWidth() == 16 ||
                     intType.getWidth() == 32;
            return false;
          },
          "f16, f32, bf16, and 8/16/32-bit integer element types")))
    return failure();
  return buildBinaryVecScope("min", contract, strategy, op.getSrc0(),
                             op.getSrc1(), op.getDst(), rewriter, op.getLoc());
}

LogicalResult lowerTAND(TAndOp op, PatternRewriter &rewriter,
                        VPTOLoweringStrategy strategy) {
  VPTOBinaryContract contract = buildBinaryContract("and", op.getSrc0());
  deriveValidShapeValues(op.getDst(), contract.validRowsValue, contract.validColsValue);
  deriveValidShape(op.getDst(), contract.validRows, contract.validCols);
  if (failed(checkGenericBinaryContract(
          op, contract, op.getSrc1(), op.getDst(),
          [](Type type) {
            if (auto intType = dyn_cast<IntegerType>(type))
              return intType.getWidth() == 8 || intType.getWidth() == 16 ||
                     intType.getWidth() == 32;
            return false;
          },
          "8/16/32-bit integer element types")))
    return failure();
  return buildBinaryVecScope("and", contract, strategy, op.getSrc0(),
                             op.getSrc1(), op.getDst(), rewriter, op.getLoc());
}

LogicalResult lowerTANDS(TAndSOp op, PatternRewriter &rewriter) {
  return emitUnresolvedInstalledA5BaselineError(op, "tands");
}

LogicalResult lowerTOR(TOrOp op, PatternRewriter &rewriter,
                       VPTOLoweringStrategy strategy) {
  VPTOBinaryContract contract = buildBinaryContract("or", op.getSrc0());
  deriveValidShapeValues(op.getDst(), contract.validRowsValue, contract.validColsValue);
  deriveValidShape(op.getDst(), contract.validRows, contract.validCols);
  if (failed(checkGenericBinaryContract(
          op, contract, op.getSrc1(), op.getDst(),
          [](Type type) {
            if (auto intType = dyn_cast<IntegerType>(type))
              return intType.getWidth() == 8 || intType.getWidth() == 16 ||
                     intType.getWidth() == 32;
            return false;
          },
          "8/16/32-bit integer element types")))
    return failure();
  return buildBinaryVecScope("or", contract, strategy, op.getSrc0(),
                             op.getSrc1(), op.getDst(), rewriter, op.getLoc());
}

LogicalResult lowerTORS(TOrSOp op, PatternRewriter &rewriter) {
  return emitUnresolvedInstalledA5BaselineError(op, "tors");
}

LogicalResult lowerTXOR(TXorOp op, PatternRewriter &rewriter,
                        VPTOLoweringStrategy strategy) {
  VPTOBinaryContract contract = buildBinaryContract("xor", op.getSrc0());
  deriveValidShapeValues(op.getDst(), contract.validRowsValue, contract.validColsValue);
  deriveValidShape(op.getDst(), contract.validRows, contract.validCols);
  if (failed(checkGenericBinaryContract(
          op, contract, op.getSrc1(), op.getDst(),
          [](Type type) {
            if (auto intType = dyn_cast<IntegerType>(type))
              return intType.getWidth() == 8 || intType.getWidth() == 16 ||
                     intType.getWidth() == 32;
            return false;
          },
          "8/16/32-bit integer element types")))
    return failure();
  return buildBinaryVecScope("xor", contract, strategy, op.getSrc0(),
                             op.getSrc1(), op.getDst(), rewriter, op.getLoc());
}

LogicalResult lowerTXORS(TXorSOp op, PatternRewriter &rewriter) {
  return emitUnresolvedInstalledA5BaselineError(op, "txors");
}

LogicalResult lowerTEXP(TExpOp op, PatternRewriter &rewriter,
                        VPTOLoweringStrategy strategy) {
  VPTOUnaryContract contract = extractTExpContract(op);
  if (failed(checkGenericUnaryContract(
          op, contract, op.getDst(),
          [](Type type) { return type.isF16() || type.isF32(); }, "f16 and f32 element types")))
    return failure();
  return buildUnaryVecScope("exp", contract, strategy, op.getSrc(), op.getDst(),
                            rewriter, op.getLoc());
}

LogicalResult lowerTLOG(TLogOp op, PatternRewriter &rewriter,
                        VPTOLoweringStrategy strategy) {
  VPTOUnaryContract contract = extractTLogContract(op);
  if (failed(checkGenericUnaryContract(
          op, contract, op.getDst(),
          [](Type type) { return type.isF16() || type.isF32(); }, "f16 and f32 element types")))
    return failure();
  return buildUnaryVecScope("log", contract, strategy, op.getSrc(), op.getDst(),
                            rewriter, op.getLoc());
}

LogicalResult lowerTSQRT(TSqrtOp op, PatternRewriter &rewriter,
                         VPTOLoweringStrategy strategy) {
  VPTOUnaryContract contract = extractTSqrtContract(op);
  if (failed(checkGenericUnaryContract(
          op, contract, op.getDst(),
          [](Type type) { return type.isF16() || type.isF32(); }, "f16 and f32 element types")))
    return failure();
  return buildUnaryVecScope("sqrt", contract, strategy, op.getSrc(), op.getDst(),
                            rewriter, op.getLoc());
}

LogicalResult lowerTRSQRT(TRsqrtOp op, PatternRewriter &rewriter) {
  VPTOUnaryContract contract = buildUnaryContract("rsqrt", op.getSrc());
  if (failed(checkGenericUnaryContract(
          op, contract, op.getDst(),
          [](Type type) { return type.isF16() || type.isF32(); }, "f16 and f32 element types")))
    return failure();

  auto vecType = getVPTOVRegType(rewriter.getContext(), contract.elementType);
  if (!vecType)
    return op.emitOpError("trsqrt lowering requires a supported VPTO vector element type");

  Value srcBuffer = materializeBufferPointer(op.getSrc(), contract.elementType,
                                             getMemorySpace(op.getSrc()), rewriter,
                                             op.getLoc());
  Value dstBuffer = materializeBufferPointer(op.getDst(), contract.elementType,
                                             getMemorySpace(op.getDst()), rewriter,
                                             op.getLoc());
  if (!srcBuffer || !dstBuffer)
    return op.emitOpError("trsqrt lowering requires pointer-backed tile buffers");

  Value validRowsValue = materializeIndexValue(contract.validRowsValue,
                                               contract.validRows, rewriter, op.getLoc());
  Value validColsValue = materializeIndexValue(contract.validColsValue,
                                               contract.validCols, rewriter, op.getLoc());
  if (!validRowsValue || !validColsValue)
    return op.emitOpError("trsqrt lowering requires valid rows and cols");

  int64_t srcRowStride = deriveStaticRowStride(op.getSrc());
  int64_t dstRowStride = deriveStaticRowStride(op.getDst());
  if (srcRowStride == ShapedType::kDynamic || dstRowStride == ShapedType::kDynamic)
    return op.emitOpError("trsqrt lowering requires static row-major row strides");

  int64_t vectorWidth = vecType.getElementCount();
  Value c0 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
  Value srcRowStrideValue =
      rewriter.create<arith::ConstantIndexOp>(op.getLoc(), srcRowStride);
  Value dstRowStrideValue =
      rewriter.create<arith::ConstantIndexOp>(op.getLoc(), dstRowStride);
  Value vectorStepValue =
      rewriter.create<arith::ConstantIndexOp>(op.getLoc(), vectorWidth);
  TypedAttr oneAttr = FloatAttr::get(contract.elementType, 1.0);
  Value one = rewriter.create<arith::ConstantOp>(op.getLoc(), oneAttr);

  FailureOr<pto::VecScopeOp> vecScope =
      createLoopScopeRegion(op.getLoc(), contract.loopScope, rewriter);
  if (failed(vecScope))
    return op.emitOpError("failed to create AIV vector scope region");

  OpBuilder::InsertionGuard aivGuard(rewriter);
  rewriter.setInsertionPointToStart(&(*vecScope).getBody().front());
  auto rowLoop = rewriter.create<scf::ForOp>(op.getLoc(), c0, validRowsValue, c1);
  rewriter.setInsertionPointToStart(rowLoop.getBody());
  Value fullMask = buildAllPredicateMask(rewriter, op.getLoc(), vecType.getElementType());
  auto ones =
      rewriter.create<pto::VdupOp>(op.getLoc(), vecType, one, fullMask, StringAttr());
  auto chunkLoop =
      rewriter.create<scf::ForOp>(op.getLoc(), c0, validColsValue, vectorStepValue);
  rewriter.setInsertionPointToStart(chunkLoop.getBody());
  Value srcRowBase = rewriter.create<arith::MulIOp>(
      op.getLoc(), rowLoop.getInductionVar(), srcRowStrideValue);
  Value dstRowBase = rewriter.create<arith::MulIOp>(
      op.getLoc(), rowLoop.getInductionVar(), dstRowStrideValue);
  Value chunkOffset = chunkLoop.getInductionVar();
  Value srcOffset =
      rewriter.create<arith::AddIOp>(op.getLoc(), srcRowBase, chunkOffset);
  Value dstOffset =
      rewriter.create<arith::AddIOp>(op.getLoc(), dstRowBase, chunkOffset);
  Value remaining = rewriter.create<arith::SubIOp>(op.getLoc(), validColsValue, chunkOffset);
  Value predicate =
      buildPredicateMaskForLaneCount(rewriter, op.getLoc(), contract.elementType, remaining);
  auto loaded = rewriter.create<pto::VldsOp>(op.getLoc(), vecType, srcBuffer,
                                              srcOffset, StringAttr());
  auto sqrt = rewriter.create<pto::VsqrtOp>(op.getLoc(), vecType, loaded.getResult(),
                                             predicate);
  auto result = rewriter.create<pto::VdivOp>(op.getLoc(), vecType, ones.getResult(),
                                              sqrt.getResult(), predicate);
  rewriter.create<pto::VstsOp>(
      op.getLoc(), result.getResult(), dstBuffer, dstOffset, StringAttr(), predicate);
  return success();
}

LogicalResult lowerTRECIP(TRecipOp op, PatternRewriter &rewriter,
                          VPTOLoweringStrategy strategy) {
  VPTOUnaryContract contract = extractTRecipContract(op);
  if (failed(checkGenericUnaryContract(
          op, contract, op.getDst(),
          [](Type type) { return type.isF16() || type.isF32(); }, "f16 and f32 element types")))
    return failure();
  return buildUnaryVecScope("recip", contract, strategy, op.getSrc(),
                            op.getDst(), rewriter, op.getLoc());
}

LogicalResult lowerTNEG(TNegOp op, PatternRewriter &rewriter,
                        VPTOLoweringStrategy strategy) {
  VPTOUnaryContract contract = buildUnaryContract("muls", op.getSrc());
  if (failed(checkGenericUnaryContract(
          op, contract, op.getDst(),
          [](Type type) {
            if (type.isF16() || type.isF32())
              return true;
            if (auto intType = dyn_cast<IntegerType>(type))
              return intType.getWidth() == 16 || intType.getWidth() == 32;
            return false;
          },
          "f16, f32, and 16/32-bit integer element types")))
    return failure();

  TypedAttr negOneAttr;
  if (contract.elementType.isF16())
    negOneAttr = FloatAttr::get(contract.elementType, -1.0);
  else if (contract.elementType.isF32())
    negOneAttr = FloatAttr::get(contract.elementType, -1.0);
  else if (auto intType = dyn_cast<IntegerType>(contract.elementType))
    negOneAttr = IntegerAttr::get(intType, -1);
  else
    return op.emitOpError("tneg lowering requires scalar element type");

  Value negOne = rewriter.create<arith::ConstantOp>(op.getLoc(), negOneAttr);
  return buildScalarUnaryVecScope("muls", contract, strategy, op.getSrc(), negOne,
                                  op.getDst(), rewriter, op.getLoc());
}

LogicalResult lowerTLRELU(TLReluOp op, PatternRewriter &rewriter,
                          VPTOLoweringStrategy strategy) {
  VPTOUnaryContract contract = buildUnaryContract("lrelu", op.getSrc());
  if (failed(checkGenericUnaryContract(
          op, contract, op.getDst(),
          [](Type type) { return type.isF16() || type.isF32(); },
          "f16 and f32 element types")))
    return failure();
  if (op.getSlope().getType() != contract.elementType)
    return op.emitOpError("tlrelu lowering requires slope type to match source element type");
  return buildScalarUnaryVecScope("lrelu", contract, strategy, op.getSrc(), op.getSlope(),
                                  op.getDst(), rewriter, op.getLoc());
}

LogicalResult lowerTCVT(TCvtOp op, PatternRewriter &rewriter) {
  VPTOUnaryContract contract = buildUnaryContract("cvt", op.getSrc());
  if (failed(checkGenericUnaryContract(
          op, contract, op.getDst(),
          [](Type type) { return type.isF16() || type.isF32() || type.isBF16(); },
          "f16, f32, or bf16 element type")))
    return failure();

  Type dstElementType = getElementType(op.getDst());
  FailureOr<VPTOCvtLoweringKind> loweringKind =
      classifyA5CvtLowering(contract.elementType, dstElementType);
  if (failed(loweringKind))
    return op.emitOpError(
        "current tcvt lowering supports only f32->f32, f32->bf16, f16->f32, and bf16->f32");

  FailureOr<StringAttr> roundMode = stringifyA5RoundMode(op, rewriter);
  if (failed(roundMode))
    return op.emitOpError("tcvt lowering does not recognize the requested round mode");

  auto srcVecType = getVPTOVRegType(rewriter.getContext(), contract.elementType);
  auto dstVecType = getVPTOVRegType(rewriter.getContext(), dstElementType);
  if (!srcVecType || !dstVecType)
    return op.emitOpError("tcvt lowering requires legal VPTO vector types");

  Value srcBuffer = materializeBufferPointer(op.getSrc(), contract.elementType,
                                             getMemorySpace(op.getSrc()), rewriter,
                                             op.getLoc());
  Value dstBuffer = materializeBufferPointer(op.getDst(), dstElementType,
                                             getMemorySpace(op.getDst()), rewriter,
                                             op.getLoc());
  if (!srcBuffer || !dstBuffer)
    return op.emitOpError("tcvt lowering requires pointer-backed tile buffers");

  Value validRowsValue = materializeIndexValue(contract.validRowsValue,
                                               contract.validRows, rewriter,
                                               op.getLoc());
  Value validColsValue = materializeIndexValue(contract.validColsValue,
                                               contract.validCols, rewriter,
                                               op.getLoc());
  if (!validRowsValue || !validColsValue)
    return op.emitOpError("tcvt lowering requires valid rows and cols");

  int64_t vectorWidth = dstVecType.getElementCount();
  if (contract.validRows != ShapedType::kDynamic &&
      contract.validCols != ShapedType::kDynamic) {
    int64_t totalElements = contract.validRows * contract.validCols;
    if (totalElements % vectorWidth != 0)
      return op.emitOpError(
          "tcvt lowering requires total valid elements divisible by vector width");
  }

  Value c0 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
  Value totalElementsValue =
      rewriter.create<arith::MulIOp>(op.getLoc(), validRowsValue, validColsValue);
  Value vectorStepValue =
      rewriter.create<arith::ConstantIndexOp>(op.getLoc(), vectorWidth);

  FailureOr<pto::VecScopeOp> vecScope =
      createLoopScopeRegion(op.getLoc(), contract.loopScope, rewriter);
  if (failed(vecScope))
    return op.emitOpError("failed to create AIV vector scope region");

  OpBuilder::InsertionGuard aivGuard(rewriter);
  rewriter.setInsertionPointToStart(&(*vecScope).getBody().front());
  auto chunkLoop =
      rewriter.create<scf::ForOp>(op.getLoc(), c0, totalElementsValue, vectorStepValue);
  OpBuilder::InsertionGuard chunkGuard(rewriter);
  rewriter.setInsertionPointToStart(chunkLoop.getBody());
  Value offset = chunkLoop.getInductionVar();
  switch (*loweringKind) {
  case VPTOCvtLoweringKind::Vtrc: {
    auto loaded =
        rewriter.create<pto::VldsOp>(op.getLoc(), srcVecType, srcBuffer, offset, StringAttr());
    Value mask = buildAllPredicateMask(rewriter, op.getLoc(), dstElementType);
    Value converted = rewriter.create<pto::VtrcOp>(op.getLoc(), dstVecType,
                                                    loaded.getResult(), mask,
                                                    *roundMode);
    rewriter.create<pto::VstsOp>(
        op.getLoc(), converted, dstBuffer, offset, StringAttr(),
        buildAllPredicateMask(rewriter, op.getLoc(), dstElementType));
    break;
  }
  case VPTOCvtLoweringKind::F32ToBF16: {
    Value halfStep = rewriter.create<arith::ConstantIndexOp>(
        op.getLoc(), srcVecType.getElementCount());
    Value upperOffset =
        rewriter.create<arith::AddIOp>(op.getLoc(), offset, halfStep);
    auto lower =
        rewriter.create<pto::VldsOp>(op.getLoc(), srcVecType, srcBuffer, offset, StringAttr());
    auto upper = rewriter.create<pto::VldsOp>(op.getLoc(), srcVecType, srcBuffer,
                                               upperOffset, StringAttr());
    Value odd = rewriter.create<pto::VcvtOp>(
        op.getLoc(), dstVecType, upper.getResult(), *roundMode,
        rewriter.getStringAttr("RS_ENABLE"), rewriter.getStringAttr("PART_ODD"));
    Value even = rewriter.create<pto::VcvtOp>(
        op.getLoc(), dstVecType, lower.getResult(), *roundMode,
        rewriter.getStringAttr("RS_ENABLE"), rewriter.getStringAttr("PART_EVEN"));
    Value merged =
        rewriter.create<pto::VorOp>(
            op.getLoc(), dstVecType, even, odd,
            buildAllPredicateMask(rewriter, op.getLoc(), dstElementType));
    rewriter.create<pto::VstsOp>(
        op.getLoc(), merged, dstBuffer, offset, StringAttr(),
        buildAllPredicateMask(rewriter, op.getLoc(), dstElementType));
    break;
  }
  case VPTOCvtLoweringKind::F16ToF32: {
    auto loaded = rewriter.create<pto::VldsOp>(
        op.getLoc(), srcVecType, srcBuffer, offset, rewriter.getStringAttr("UNPK_B16"));
    Value converted = rewriter.create<pto::VcvtOp>(
        op.getLoc(), dstVecType, loaded.getResult(), StringAttr(),
        StringAttr(), rewriter.getStringAttr("PART_EVEN"));
    rewriter.create<pto::VstsOp>(
        op.getLoc(), converted, dstBuffer, offset, StringAttr(),
        buildAllPredicateMask(rewriter, op.getLoc(), dstElementType));
    break;
  }
  case VPTOCvtLoweringKind::BF16ToF32: {
    auto loaded = rewriter.create<pto::VldsOp>(
        op.getLoc(), srcVecType, srcBuffer, offset, rewriter.getStringAttr("UNPK_B16"));
    Value converted = rewriter.create<pto::VcvtOp>(
        op.getLoc(), dstVecType, loaded.getResult(), StringAttr(),
        StringAttr(), rewriter.getStringAttr("PART_EVEN"));
    rewriter.create<pto::VstsOp>(
        op.getLoc(), converted, dstBuffer, offset, StringAttr(),
        buildAllPredicateMask(rewriter, op.getLoc(), dstElementType));
    break;
  }
  }
  return success();
}

template <typename CompareEmitter>
LogicalResult buildPackedCmp32VecScope(StringRef family,
                                       const VPTOBinaryContract &contract,
                                       Value dst, Value dstBuffer,
                                       PatternRewriter &rewriter, Location loc,
                                       CompareEmitter emitCompare) {
  auto vecType = getVPTOVRegType(rewriter.getContext(), contract.elementType);
  if (!vecType)
    return emitError(loc) << family << " lowering requires a supported vector element type";

  Value validRowsValue = materializeIndexValue(contract.validRowsValue,
                                               contract.validRows, rewriter, loc);
  Value validColsValue = materializeIndexValue(contract.validColsValue,
                                               contract.validCols, rewriter, loc);
  if (!validRowsValue || !validColsValue)
    return emitError(loc) << family << " lowering requires valid rows and cols";
  if (contract.validRows == ShapedType::kDynamic ||
      contract.validCols == ShapedType::kDynamic)
    return emitError(loc) << family << " lowering currently requires static valid rows and cols";

  int64_t totalElements = contract.validRows * contract.validCols;
  constexpr int64_t repeatElem = 64;
  int64_t repeatTimes = (totalElements + repeatElem - 1) / repeatElem;
  int64_t pairedRepeats = repeatTimes / 2;
  int64_t remainRepeats = repeatTimes % 2;

  auto compareMaskType =
      getVPTOMaskTypeForElementType(rewriter.getContext(), contract.elementType);
  auto packedMaskType = getVPTOMaskType(rewriter.getContext(), "b8");
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value pairUpper = rewriter.create<arith::ConstantIndexOp>(loc, pairedRepeats);
  Value repeatStep = rewriter.create<arith::ConstantIndexOp>(loc, repeatElem);
  Value pairSrcStride = rewriter.create<arith::ConstantIndexOp>(loc, repeatElem * 2);
  Value pairDstStride = rewriter.create<arith::ConstantIndexOp>(loc, 4);
  Value laneCount = rewriter.create<arith::ConstantIntOp>(loc, repeatElem, 32);
  Value totalRemaining = rewriter.create<arith::ConstantIntOp>(loc, totalElements, 32);

  FailureOr<pto::VecScopeOp> vecScope =
      createLoopScopeRegion(loc, contract.loopScope, rewriter);
  if (failed(vecScope))
    return emitError(loc) << "failed to create AIV vector scope region";

  OpBuilder::InsertionGuard aivGuard(rewriter);
  rewriter.setInsertionPointToStart(&(*vecScope).getBody().front());
  auto pairLoop =
      rewriter.create<scf::ForOp>(loc, c0, pairUpper, c1, ValueRange{totalRemaining});
  rewriter.setInsertionPointToStart(pairLoop.getBody());
  Value remaining = pairLoop.getRegionIterArgs().front();
  Value pairBase = rewriter.create<arith::MulIOp>(loc, pairLoop.getInductionVar(),
                                                  pairSrcStride);
  Value pairNext = rewriter.create<arith::AddIOp>(loc, pairBase, repeatStep);
  Value dstOffset = rewriter.create<arith::MulIOp>(loc, pairLoop.getInductionVar(),
                                                   pairDstStride);
  Value dstBase = adjustPointerByElemOffset(dstBuffer, dstOffset, 4, rewriter, loc);
  Value dstZero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto pairMask0 = rewriter.create<pto::PltB32Op>(loc, compareMaskType,
                                                   rewriter.getI32Type(),
                                                   remaining);
  auto pairMask1 = rewriter.create<pto::PltB32Op>(loc, compareMaskType,
                                                   rewriter.getI32Type(),
                                                   pairMask0.getScalarOut());
  Value cmp0 = emitCompare(rewriter, loc, pairBase, pairMask0.getMask());
  Value cmp1 = emitCompare(rewriter, loc, pairNext, pairMask1.getMask());
  Value packedCmp0 = rewriter
                         .create<pto::PpackOp>(loc, packedMaskType, cmp0,
                                               rewriter.getStringAttr("LOWER"))
                         .getResult();
  Value packedCmp1 = rewriter
                         .create<pto::PpackOp>(loc, packedMaskType, cmp1,
                                               rewriter.getStringAttr("LOWER"))
                         .getResult();
  auto interleaved = rewriter.create<pto::PdintlvB8Op>(
      loc, packedMaskType, packedMaskType, packedCmp0, packedCmp1);
  rewriter.create<pto::PstsOp>(loc, interleaved.getLow(), dstBase, dstZero,
                               "NORM");
  rewriter.create<scf::YieldOp>(loc, pairMask1.getScalarOut());

  if (remainRepeats == 0)
    return success();

  rewriter.setInsertionPointAfter(pairLoop);
  Value tailBase = rewriter.create<arith::ConstantIndexOp>(loc, pairedRepeats * repeatElem * 2);
  Value tailDst = rewriter.create<arith::ConstantIndexOp>(loc, pairedRepeats * 4);
  Value tailDstBase = adjustPointerByElemOffset(dstBuffer, tailDst, 4, rewriter, loc);
  Value tailDstZero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto tailMask = rewriter.create<pto::PltB32Op>(loc, compareMaskType,
                                                  rewriter.getI32Type(),
                                                  pairLoop.getResult(0));
  Value tailCmp = emitCompare(rewriter, loc, tailBase, tailMask.getMask());
  Value packedTail = rewriter
                         .create<pto::PpackOp>(loc, packedMaskType, tailCmp,
                                                rewriter.getStringAttr("LOWER"))
                         .getResult();
  rewriter.create<pto::PstsOp>(loc, packedTail, tailDstBase, tailDstZero,
                               "NORM");
  return success();
}

LogicalResult lowerTCmpS(TCmpSOp op, PatternRewriter &rewriter) {
  VPTOBinaryContract contract = buildBinaryContract("cmps", op.getSrc());
  deriveValidShapeValues(op.getDst(), contract.validRowsValue, contract.validColsValue);
  deriveValidShape(op.getDst(), contract.validRows, contract.validCols);

  if (contract.tileDomain != VPTOTileDomain::Vec ||
      deriveTileDomain(getMemorySpace(op.getDst())) != VPTOTileDomain::Vec)
    return op.emitOpError("tcmps lowering requires tile domain vec");
  if (contract.tileLayout != "row_major" || deriveTileLayout(op.getDst()) != "row_major")
    return op.emitOpError("tcmps lowering requires row-major tile layout");
  if (contract.validRows == ShapedType::kDynamic ||
      contract.validCols == ShapedType::kDynamic)
    return op.emitOpError("tcmps lowering requires static valid shape");
  int64_t dstRows = ShapedType::kDynamic;
  int64_t dstCols = ShapedType::kDynamic;
  deriveValidShape(op.getDst(), dstRows, dstCols);
  if (contract.validRows != dstRows || contract.validCols != dstCols)
    return op.emitOpError("tcmps lowering requires matching source and destination valid region");
  if (!isSupportedPackedCmp32ElementType(contract.elementType))
    return op.emitOpError("tcmps lowering currently supports only 32-bit source tiles");
  auto dstElemType = dyn_cast_or_null<IntegerType>(getElementType(op.getDst()));
  if (!dstElemType || !dstElemType.isUnsignedInteger(8))
    return op.emitOpError("tcmps lowering currently requires ui8 destination tiles");
  if (op.getScalar().getType() != contract.elementType)
    return op.emitOpError("tcmps lowering requires scalar type to match source element type");

  Value srcBuffer = materializeBufferPointer(op.getSrc(), contract.elementType,
                                             getMemorySpace(op.getSrc()), rewriter,
                                             op.getLoc());
  Value dstBuffer = materializeBufferPointer(op.getDst(), getElementType(op.getDst()),
                                             getMemorySpace(op.getDst()), rewriter,
                                             op.getLoc());
  if (!srcBuffer || !dstBuffer)
    return op.emitOpError("tcmps lowering requires pointer-backed tile buffers");

  StringAttr cmpMode = rewriter.getStringAttr(stringifyCmpModeAttr(op.getCmpModeAttr()));
  return buildPackedCmp32VecScope(
      "tcmps", contract, op.getDst(), dstBuffer, rewriter, op.getLoc(),
      [&](PatternRewriter &nestedRewriter, Location nestedLoc, Value offset,
          Value mask) -> Value {
        auto vecType =
            getVPTOVRegType(nestedRewriter.getContext(), contract.elementType);
        auto loaded =
            nestedRewriter.create<pto::VldsOp>(nestedLoc, vecType, srcBuffer, offset, StringAttr());
        return nestedRewriter
            .create<pto::VcmpsOp>(nestedLoc,
                                   getVPTOMaskTypeForElementType(
                                       nestedRewriter.getContext(),
                                       contract.elementType),
                                   loaded.getResult(), op.getScalar(), mask, cmpMode)
            .getResult();
      });
}

LogicalResult lowerTCmp(TCmpOp op, PatternRewriter &rewriter) {
  VPTOBinaryContract contract = buildBinaryContract("cmp", op.getSrc0());
  deriveValidShapeValues(op.getDst(), contract.validRowsValue, contract.validColsValue);
  deriveValidShape(op.getDst(), contract.validRows, contract.validCols);

  if (contract.tileDomain != VPTOTileDomain::Vec ||
      deriveTileDomain(getMemorySpace(op.getSrc1())) != VPTOTileDomain::Vec ||
      deriveTileDomain(getMemorySpace(op.getDst())) != VPTOTileDomain::Vec)
    return op.emitOpError("tcmp lowering requires tile domain vec");
  if (contract.tileLayout != "row_major" || deriveTileLayout(op.getSrc1()) != "row_major" ||
      deriveTileLayout(op.getDst()) != "row_major")
    return op.emitOpError("tcmp lowering requires row-major tile layout");
  if (contract.validRows == ShapedType::kDynamic ||
      contract.validCols == ShapedType::kDynamic)
    return op.emitOpError("tcmp lowering requires static valid shape");
  int64_t src1Rows = ShapedType::kDynamic;
  int64_t src1Cols = ShapedType::kDynamic;
  int64_t dstRows = ShapedType::kDynamic;
  int64_t dstCols = ShapedType::kDynamic;
  deriveValidShape(op.getSrc1(), src1Rows, src1Cols);
  deriveValidShape(op.getDst(), dstRows, dstCols);
  if (contract.validRows != src1Rows || contract.validCols != src1Cols ||
      contract.validRows != dstRows || contract.validCols != dstCols)
    return op.emitOpError("tcmp lowering requires matching source and destination valid region");
  if (!isSupportedPackedCmp32ElementType(contract.elementType))
    return op.emitOpError("tcmp lowering currently supports only 32-bit source tiles");
  if (getElementType(op.getSrc1()) != contract.elementType)
    return op.emitOpError("tcmp lowering requires src1 element type to match src0");
  auto dstElemType = dyn_cast_or_null<IntegerType>(getElementType(op.getDst()));
  if (!dstElemType || !dstElemType.isUnsignedInteger(8))
    return op.emitOpError("tcmp lowering currently requires ui8 destination tiles");

  Value src0Buffer = materializeBufferPointer(op.getSrc0(), contract.elementType,
                                              getMemorySpace(op.getSrc0()), rewriter,
                                              op.getLoc());
  Value src1Buffer = materializeBufferPointer(op.getSrc1(), contract.elementType,
                                              getMemorySpace(op.getSrc1()), rewriter,
                                              op.getLoc());
  Value dstBuffer = materializeBufferPointer(op.getDst(), getElementType(op.getDst()),
                                             getMemorySpace(op.getDst()), rewriter,
                                             op.getLoc());
  if (!src0Buffer || !src1Buffer || !dstBuffer)
    return op.emitOpError("tcmp lowering requires pointer-backed tile buffers");

  StringAttr cmpMode = rewriter.getStringAttr(stringifyCmpModeAttr(op.getCmpModeAttr()));
  return buildPackedCmp32VecScope(
      "tcmp", contract, op.getDst(), dstBuffer, rewriter, op.getLoc(),
      [&](PatternRewriter &nestedRewriter, Location nestedLoc, Value offset,
          Value mask) -> Value {
        auto vecType =
            getVPTOVRegType(nestedRewriter.getContext(), contract.elementType);
        auto lhs =
            nestedRewriter.create<pto::VldsOp>(nestedLoc, vecType, src0Buffer, offset, StringAttr());
        auto rhs =
            nestedRewriter.create<pto::VldsOp>(nestedLoc, vecType, src1Buffer, offset, StringAttr());
        return nestedRewriter
            .create<pto::VcmpOp>(nestedLoc,
                                  getVPTOMaskTypeForElementType(
                                      nestedRewriter.getContext(),
                                      contract.elementType),
                                  lhs.getResult(), rhs.getResult(), mask, cmpMode)
            .getResult();
      });
}

LogicalResult lowerTCI(TCIOp op, PatternRewriter &rewriter) {
  Type elementType = getElementType(op.getDst());
  auto intType = dyn_cast_or_null<IntegerType>(elementType);
  if (!intType || (intType.getWidth() != 16 && intType.getWidth() != 32))
    return op.emitOpError("tci lowering requires i16 or i32 destination element type");
  if (deriveTileDomain(getMemorySpace(op.getDst())) != VPTOTileDomain::Vec)
    return op.emitOpError("tci lowering requires tile domain vec");
  if (deriveTileLayout(op.getDst()) != "row_major")
    return op.emitOpError("tci lowering requires row-major tile layout");

  int64_t validRows = ShapedType::kDynamic;
  int64_t validCols = ShapedType::kDynamic;
  Value validRowsValue;
  Value validColsValue;
  deriveValidShapeValues(op.getDst(), validRowsValue, validColsValue);
  deriveValidShape(op.getDst(), validRows, validCols);
  if (validRows != 1)
    return op.emitOpError("tci lowering currently requires valid rows == 1");

  Value dstBuffer = materializeBufferPointer(op.getDst(), elementType,
                                             getMemorySpace(op.getDst()), rewriter,
                                             op.getLoc());
  if (!dstBuffer)
    return op.emitOpError("tci lowering requires pointer-backed destination tile buffer");

  Value upperBound = materializeIndexValue(validColsValue, validCols, rewriter, op.getLoc());
  if (!upperBound)
    return op.emitOpError("tci lowering requires valid cols");

  Value c0 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
  auto loop = rewriter.create<scf::ForOp>(op.getLoc(), c0, upperBound, c1);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(loop.getBody());
  Value iv = loop.getInductionVar();
  Value ivAsElem = rewriter.create<arith::IndexCastOp>(op.getLoc(), intType, iv);
  Value stored =
      op.getDescending()
          ? rewriter.create<arith::SubIOp>(op.getLoc(), op.getS(), ivAsElem).getResult()
          : rewriter.create<arith::AddIOp>(op.getLoc(), op.getS(), ivAsElem).getResult();
  rewriter.create<pto::StoreScalarOp>(op.getLoc(), dstBuffer, iv, stored);
  return success();
}

LogicalResult lowerTRELU(TReluOp op, PatternRewriter &rewriter,
                         VPTOLoweringStrategy strategy) {
  VPTOUnaryContract contract = extractTReluContract(op);
  if (failed(checkGenericUnaryContract(
          op, contract, op.getDst(),
          [](Type type) {
            return type.isF16() || type.isF32() ||
                   (isa<IntegerType>(type) && cast<IntegerType>(type).getWidth() == 32);
          },
          "f16, f32, and i32 element types")))
    return failure();
  return buildUnaryVecScope("relu", contract, strategy, op.getSrc(),
                            op.getDst(), rewriter, op.getLoc());
}

LogicalResult lowerTNOT(TNotOp op, PatternRewriter &rewriter,
                        VPTOLoweringStrategy strategy) {
  VPTOUnaryContract contract = extractTNotContract(op);
  if (failed(checkGenericUnaryContract(
          op, contract, op.getDst(),
          [](Type type) {
            if (type.isF16() || type.isF32() || type.isBF16())
              return true;
            if (auto intType = dyn_cast<IntegerType>(type))
              return intType.getWidth() == 8 || intType.getWidth() == 16 ||
                     intType.getWidth() == 32;
            return false;
          },
          "f16, f32, bf16, and 8/16/32-bit integer element types")))
    return failure();
  return buildUnaryVecScope("not", contract, strategy, op.getSrc(), op.getDst(),
                            rewriter, op.getLoc());
}

LogicalResult lowerTTRANS(TTransOp op, PatternRewriter &rewriter) {
  VPTOUnaryContract contract = buildUnaryContract("trans", op.getSrc());
  int64_t dstRows = ShapedType::kDynamic;
  int64_t dstCols = ShapedType::kDynamic;
  deriveValidShape(op.getDst(), dstRows, dstCols);

  if (contract.tileDomain != VPTOTileDomain::Vec ||
      deriveTileDomain(getMemorySpace(op.getDst())) != VPTOTileDomain::Vec)
    return op.emitOpError("ttrans lowering requires tile domain vec");
  if (contract.tileLayout != "row_major" || deriveTileLayout(op.getDst()) != "row_major")
    return op.emitOpError("ttrans lowering requires row-major tile layout");
  if (contract.validRows == ShapedType::kDynamic || contract.validCols == ShapedType::kDynamic ||
      dstRows == ShapedType::kDynamic || dstCols == ShapedType::kDynamic)
    return op.emitOpError("ttrans lowering requires static valid shape");
  if (contract.validRows != dstCols || contract.validCols != dstRows)
    return op.emitOpError("ttrans lowering requires transposed source/destination valid shape");
  if (contract.elementType != getElementType(op.getDst()))
    return op.emitOpError("ttrans lowering requires matching source/destination element type");

  int64_t elemBytes = getElementByteSize(contract.elementType);
  int64_t srcStride = deriveStaticRowStride(op.getSrc());
  int64_t dstStride = deriveStaticRowStride(op.getDst());
  if (elemBytes != 4)
    return op.emitOpError("ttrans lowering currently supports only b32 element types");
  if (srcStride == ShapedType::kDynamic || dstStride == ShapedType::kDynamic)
    return op.emitOpError("ttrans lowering requires static source/destination row stride");

  auto dataVecType = getVPTOVRegType(rewriter.getContext(), contract.elementType);
  auto indexElemType = rewriter.getIntegerType(32);
  auto indexVecType = getVPTOVRegType(rewriter.getContext(), indexElemType);
  if (!dataVecType || !indexVecType)
    return op.emitOpError("ttrans lowering requires supported VPTO vector types");

  Value srcBuffer = materializeBufferPointer(op.getSrc(), contract.elementType,
                                             getMemorySpace(op.getSrc()), rewriter,
                                             op.getLoc());
  Value dstBuffer = materializeBufferPointer(op.getDst(), contract.elementType,
                                             getMemorySpace(op.getDst()), rewriter,
                                             op.getLoc());
  if (!srcBuffer || !dstBuffer)
    return op.emitOpError("ttrans lowering requires pointer-backed tile buffers");

  constexpr int64_t repeatBytes = 256;
  constexpr int64_t blockBytes = 32;
  int64_t elementsPerRepeat = repeatBytes / elemBytes;
  int64_t blockSizeElem = blockBytes / elemBytes;
  int64_t alignedRows =
      llvm::divideCeil(contract.validRows, blockSizeElem) * blockSizeElem;
  int64_t repeatTimes = llvm::divideCeil(alignedRows, elementsPerRepeat);

  Value c0 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
  Value colsUpper = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), contract.validCols);
  Value chunkUpper = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), repeatTimes);
  Value elementsPerRepeatValue =
      rewriter.create<arith::ConstantIndexOp>(op.getLoc(), elementsPerRepeat);
  Value dstStrideValue = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), dstStride);
  Value srcStrideI32 = rewriter.create<arith::ConstantIntOp>(op.getLoc(), srcStride, 32);

  FailureOr<pto::VecScopeOp> vecScope =
      createLoopScopeRegion(op.getLoc(), contract.loopScope, rewriter);
  if (failed(vecScope))
    return op.emitOpError("failed to create AIV vector scope region");

  OpBuilder::InsertionGuard aivGuard(rewriter);
  rewriter.setInsertionPointToStart(&(*vecScope).getBody().front());
  auto colLoop = rewriter.create<scf::ForOp>(op.getLoc(), c0, colsUpper, c1);
  rewriter.setInsertionPointToStart(colLoop.getBody());
  auto chunkLoop = rewriter.create<scf::ForOp>(op.getLoc(), c0, chunkUpper, c1);
  rewriter.setInsertionPointToStart(chunkLoop.getBody());

  Value chunkBase = rewriter.create<arith::MulIOp>(op.getLoc(), chunkLoop.getInductionVar(),
                                                   elementsPerRepeatValue);
  Value colI32 = rewriter.create<arith::IndexCastOp>(op.getLoc(), indexElemType,
                                                     colLoop.getInductionVar());
  Value chunkBaseI32 =
      rewriter.create<arith::IndexCastOp>(op.getLoc(), indexElemType, chunkBase);
  auto indices =
      rewriter.create<pto::VciOp>(op.getLoc(), indexVecType, chunkBaseI32,
                                   rewriter.getStringAttr("INC_ORDER"));
  Value fullMask = buildAllPredicateMask(rewriter, op.getLoc(), indexElemType);
  auto scaled = rewriter.create<pto::VmulsOp>(op.getLoc(), indexVecType,
                                               indices.getResult(), srcStrideI32, fullMask);
  auto offsets = rewriter.create<pto::VaddsOp>(op.getLoc(), indexVecType,
                                                scaled.getResult(), colI32, fullMask);
  Value fullActiveLanes =
      rewriter.create<arith::ConstantIndexOp>(op.getLoc(),
                                              dataVecType.getElementCount());
  auto gathered =
      rewriter.create<pto::Vgather2Op>(op.getLoc(), dataVecType, srcBuffer,
                                        offsets.getResult(), fullActiveLanes);
  Value dstBase =
      rewriter.create<arith::MulIOp>(op.getLoc(), colLoop.getInductionVar(), dstStrideValue);
  Value dstOffset = rewriter.create<arith::AddIOp>(op.getLoc(), dstBase, chunkBase);
  rewriter.create<pto::VstsOp>(
      op.getLoc(), gathered.getResult(), dstBuffer, dstOffset, StringAttr(),
      buildAllPredicateMask(rewriter, op.getLoc(), contract.elementType));
  return success();
}

template <typename FillPadOpTy>
LogicalResult lowerTFillPadCommon(FillPadOpTy op, PatternRewriter &rewriter,
                                  bool allowDstExpand) {
  VPTOUnaryContract contract = buildUnaryContract("fillpad", op.getSrc());
  int64_t dstRows = ShapedType::kDynamic;
  int64_t dstCols = ShapedType::kDynamic;
  deriveValidShape(op.getDst(), dstRows, dstCols);

  if (contract.tileDomain != VPTOTileDomain::Vec ||
      deriveTileDomain(getMemorySpace(op.getDst())) != VPTOTileDomain::Vec)
    return op.emitOpError("fillpad lowering requires tile domain vec");
  if (contract.tileLayout != "row_major" || deriveTileLayout(op.getDst()) != "row_major")
    return op.emitOpError("fillpad lowering requires row-major tile layout");
  if (contract.validRows == ShapedType::kDynamic || contract.validCols == ShapedType::kDynamic ||
      dstRows == ShapedType::kDynamic || dstCols == ShapedType::kDynamic)
    return op.emitOpError("fillpad lowering requires static valid shape");
  if (!allowDstExpand && (contract.validRows != dstRows || contract.validCols != dstCols))
    return op.emitOpError("tfillpad lowering requires matching source/destination valid shape");
  if (allowDstExpand && (dstRows < contract.validRows || dstCols < contract.validCols))
    return op.emitOpError("tfillpad_expand lowering requires dst shape >= src shape");
  if (contract.elementType != getElementType(op.getDst()))
    return op.emitOpError("fillpad lowering requires matching source/destination element type");

  int64_t srcStride = deriveStaticRowStride(op.getSrc());
  int64_t dstStride = deriveStaticRowStride(op.getDst());
  if (srcStride == ShapedType::kDynamic || dstStride == ShapedType::kDynamic)
    return op.emitOpError("fillpad lowering requires static source/destination row stride");

  auto vecType = getVPTOVRegType(rewriter.getContext(), contract.elementType);
  if (!vecType)
    return op.emitOpError("fillpad lowering requires supported VPTO vector element type");

  Value srcBuffer = materializeBufferPointer(op.getSrc(), contract.elementType,
                                             getMemorySpace(op.getSrc()), rewriter,
                                             op.getLoc());
  Value dstBuffer = materializeBufferPointer(op.getDst(), contract.elementType,
                                             getMemorySpace(op.getDst()), rewriter,
                                             op.getLoc());
  if (!srcBuffer || !dstBuffer)
    return op.emitOpError("fillpad lowering requires pointer-backed tile buffers");

  auto config = lookupTileConfig(op.getDst());
  PadValueAttr padAttr = config ? dyn_cast<PadValueAttr>(config.getPad()) : PadValueAttr{};
  Attribute padValueAttr = buildFillPadValue(contract.elementType, padAttr, rewriter);
  if (!padValueAttr)
    return op.emitOpError("fillpad lowering requires a concrete non-null dst pad value");
  Value padScalar = rewriter.create<arith::ConstantOp>(op.getLoc(), cast<TypedAttr>(padValueAttr));
  Value fullMask = buildAllPredicateMask(rewriter, op.getLoc(), vecType.getElementType());
  auto padVec =
      rewriter.create<pto::VdupOp>(op.getLoc(), vecType, padScalar, fullMask, StringAttr());

  int64_t vectorWidth = vecType.getElementCount();
  int64_t padCols = dstCols - contract.validCols;
  int64_t padRows = dstRows - contract.validRows;

  Value c0 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
  Value srcRowsUpper = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), contract.validRows);
  Value srcColsUpper = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), contract.validCols);
  Value dstRowsUpper = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), dstRows);
  Value vectorStep = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), vectorWidth);
  Value srcStrideValue = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), srcStride);
  Value dstStrideValue = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), dstStride);
  Value validColsValue = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), contract.validCols);
  Value dstColsValue = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), dstCols);

  FailureOr<pto::VecScopeOp> vecScope =
      createLoopScopeRegion(op.getLoc(), contract.loopScope, rewriter);
  if (failed(vecScope))
    return op.emitOpError("failed to create AIV vector scope region");

  OpBuilder::InsertionGuard aivGuard(rewriter);
  rewriter.setInsertionPointToStart(&(*vecScope).getBody().front());

  auto rowLoop = rewriter.create<scf::ForOp>(op.getLoc(), c0, srcRowsUpper, c1);
  rewriter.setInsertionPointToStart(rowLoop.getBody());
  Value srcRowBase = rewriter.create<arith::MulIOp>(op.getLoc(), rowLoop.getInductionVar(),
                                                    srcStrideValue);
  Value dstRowBase = rewriter.create<arith::MulIOp>(op.getLoc(), rowLoop.getInductionVar(),
                                                    dstStrideValue);

  auto copyChunkLoop =
      rewriter.create<scf::ForOp>(op.getLoc(), c0, srcColsUpper, vectorStep);
  rewriter.setInsertionPointToStart(copyChunkLoop.getBody());
  Value copyOffset =
      rewriter.create<arith::AddIOp>(op.getLoc(), srcRowBase, copyChunkLoop.getInductionVar());
  auto loaded = rewriter.create<pto::VldsOp>(op.getLoc(), vecType, srcBuffer,
                                              copyOffset, StringAttr());
  Value copyDstOffset =
      rewriter.create<arith::AddIOp>(op.getLoc(), dstRowBase, copyChunkLoop.getInductionVar());
  Value copyRemaining =
      rewriter.create<arith::SubIOp>(op.getLoc(), validColsValue, copyChunkLoop.getInductionVar());
  auto copyNeedsClamp = rewriter.create<arith::CmpIOp>(op.getLoc(), arith::CmpIPredicate::slt,
                                                       copyRemaining, vectorStep);
  Value copyActiveLanes =
      rewriter.create<arith::SelectOp>(op.getLoc(), copyNeedsClamp, copyRemaining, vectorStep);
  Value copyMask = buildPredicateMaskForLaneCount(
      rewriter, op.getLoc(), contract.elementType, copyActiveLanes);
  rewriter.create<pto::VstsOp>(op.getLoc(), loaded.getResult(), dstBuffer,
                                copyDstOffset, StringAttr(), copyMask);

  rewriter.setInsertionPointAfter(copyChunkLoop);
  if (padCols > 0) {
    Value padColsUpper = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), padCols);
    auto padColLoop = rewriter.create<scf::ForOp>(op.getLoc(), c0, padColsUpper, vectorStep);
    rewriter.setInsertionPointToStart(padColLoop.getBody());
    Value padDstStart = rewriter.create<arith::AddIOp>(op.getLoc(), dstRowBase, validColsValue);
    Value padDstOffset = rewriter.create<arith::AddIOp>(op.getLoc(), padDstStart,
                                                        padColLoop.getInductionVar());
    Value padRemaining =
        rewriter.create<arith::SubIOp>(op.getLoc(), padColsUpper, padColLoop.getInductionVar());
    auto padNeedsClamp = rewriter.create<arith::CmpIOp>(op.getLoc(), arith::CmpIPredicate::slt,
                                                        padRemaining, vectorStep);
    Value padActiveLanes =
        rewriter.create<arith::SelectOp>(op.getLoc(), padNeedsClamp, padRemaining, vectorStep);
    Value padMask = buildPredicateMaskForLaneCount(
        rewriter, op.getLoc(), contract.elementType, padActiveLanes);
    rewriter.create<pto::VstsOp>(op.getLoc(), padVec.getResult(), dstBuffer,
                                  padDstOffset, StringAttr(), padMask);
  }

  rewriter.setInsertionPointAfter(rowLoop);
  if (padRows > 0) {
    Value bottomStart = rewriter.create<arith::MulIOp>(op.getLoc(), srcRowsUpper, dstStrideValue);
    Value bottomElements =
        rewriter.create<arith::SubIOp>(op.getLoc(),
                                       rewriter.create<arith::MulIOp>(op.getLoc(), dstRowsUpper,
                                                                      dstColsValue),
                                       bottomStart);
    auto bottomLoop = rewriter.create<scf::ForOp>(op.getLoc(), c0, bottomElements, vectorStep);
    rewriter.setInsertionPointToStart(bottomLoop.getBody());
    Value bottomDstOffset =
        rewriter.create<arith::AddIOp>(op.getLoc(), bottomStart, bottomLoop.getInductionVar());
    Value bottomRemaining =
        rewriter.create<arith::SubIOp>(op.getLoc(), bottomElements, bottomLoop.getInductionVar());
    auto bottomNeedsClamp = rewriter.create<arith::CmpIOp>(
        op.getLoc(), arith::CmpIPredicate::slt, bottomRemaining, vectorStep);
    Value bottomActiveLanes = rewriter.create<arith::SelectOp>(
        op.getLoc(), bottomNeedsClamp, bottomRemaining, vectorStep);
    Value bottomMask = buildPredicateMaskForLaneCount(
        rewriter, op.getLoc(), contract.elementType, bottomActiveLanes);
    rewriter.create<pto::VstsOp>(op.getLoc(), padVec.getResult(), dstBuffer,
                                  bottomDstOffset, StringAttr(), bottomMask);
  }

  return success();
}

LogicalResult lowerTFILLPAD(TFillPadOp op, PatternRewriter &rewriter) {
  return lowerTFillPadCommon(op, rewriter, /*allowDstExpand=*/false);
}

LogicalResult lowerTFILLPADExpand(TFillPadExpandOp op, PatternRewriter &rewriter) {
  return lowerTFillPadCommon(op, rewriter, /*allowDstExpand=*/true);
}

LogicalResult lowerTExpandS(TExpandsOp op, PatternRewriter &rewriter) {
  VPTOUnaryContract contract = extractTExpandSContract(op);
  if (contract.tileDomain != VPTOTileDomain::Vec)
    return op.emitOpError("expands lowering requires tile domain vec");
  if (contract.tileLayout != "row_major")
    return op.emitOpError("expands lowering requires row-major tile layout");
  if (!contract.elementType)
    return op.emitOpError("expands lowering requires a concrete element type");

  Type scalarType = op.getScalar().getType();
  if (scalarType != contract.elementType)
    return op.emitOpError("expands lowering requires scalar type to match destination element type");

  if (!(contract.elementType.isF16() || contract.elementType.isF32() ||
        contract.elementType.isBF16())) {
    if (auto intType = dyn_cast<IntegerType>(contract.elementType)) {
      unsigned width = intType.getWidth();
      if (width != 8 && width != 16 && width != 32)
        return op.emitOpError("expands lowering supports only f16, f32, bf16, and 8/16/32-bit integer element types");
    } else {
      return op.emitOpError("expands lowering supports only scalar integer or floating-point element types");
    }
  }

  return buildExpandScalarVecScope(contract, op.getScalar(), op.getDst(),
                                   rewriter, op.getLoc());
}

LogicalResult lowerTGather(TGatherOp op, PatternRewriter &rewriter) {
  auto requireVecRowMajor = [&](Value value, StringRef role) -> LogicalResult {
    if (deriveTileDomain(getMemorySpace(value)) != VPTOTileDomain::Vec)
      return op.emitOpError() << "tgather lowering requires vec tile domain for "
                              << role;
    if (deriveTileLayout(value) != "row_major")
      return op.emitOpError() << "tgather lowering requires row-major layout for "
                              << role;
    return success();
  };

  if (failed(requireVecRowMajor(op.getSrc(), "src")) ||
      failed(requireVecRowMajor(op.getDst(), "dst")))
    return failure();

  Type dataElementType = getElementType(op.getSrc());
  if (dataElementType != getElementType(op.getDst()))
    return op.emitOpError("tgather lowering requires matching src/dst element type");

  auto dataVecType = getVPTOVRegType(rewriter.getContext(), dataElementType);
  if (!dataVecType)
    return op.emitOpError("tgather lowering requires supported VPTO data type");

  Value srcBuffer = materializeBufferPointer(op.getSrc(), dataElementType,
                                             getMemorySpace(op.getSrc()), rewriter,
                                             op.getLoc());
  Value dstBuffer = materializeBufferPointer(op.getDst(), dataElementType,
                                             getMemorySpace(op.getDst()), rewriter,
                                             op.getLoc());
  if (!srcBuffer || !dstBuffer)
    return op.emitOpError("tgather lowering requires pointer-backed tile buffers");

  int64_t srcStride = deriveStaticRowStride(op.getSrc());
  int64_t dstStride = deriveStaticRowStride(op.getDst());
  if (srcStride == ShapedType::kDynamic || dstStride == ShapedType::kDynamic)
    return op.emitOpError("tgather lowering requires static row stride");

  Value c0 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
  VPTOLoopScopeContract loopScope;
  loopScope.kind = VPTOLoopScopeKind::AIVVectorScope;
  loopScope.loweredAttr = kLoweredLoopScopeAttrName;
  loopScope.loopDepth = 0;

  FailureOr<pto::VecScopeOp> vecScope =
      createLoopScopeRegion(op.getLoc(), loopScope, rewriter);
  if (failed(vecScope))
    return op.emitOpError("failed to create AIV vector scope region");

  OpBuilder::InsertionGuard aivGuard(rewriter);
  rewriter.setInsertionPointToStart(&(*vecScope).getBody().front());

  if (Value indices = op.getIndices()) {
    if (failed(requireVecRowMajor(indices, "indices")))
      return failure();

    Type indexElementType = getElementType(indices);
    auto indexIntegerType = dyn_cast<IntegerType>(indexElementType);
    auto indexVecType = getVPTOVRegType(rewriter.getContext(), indexElementType);
    if (!indexIntegerType || !indexVecType)
      return op.emitOpError("tgather index lowering requires integer indices with supported VPTO vector type");
    if (indexVecType.getElementCount() != dataVecType.getElementCount())
      return op.emitOpError("tgather index lowering currently requires matching data/index vector widths");

    Value indexBuffer = materializeBufferPointer(indices, indexElementType,
                                                getMemorySpace(indices), rewriter,
                                                op.getLoc());
    if (!indexBuffer)
      return op.emitOpError("tgather index lowering requires pointer-backed indices tile");

    int64_t indexStride = deriveStaticRowStride(indices);
    if (indexStride == ShapedType::kDynamic)
      return op.emitOpError("tgather index lowering requires static index row stride");

    Value validRowsValue;
    Value validColsValue;
    int64_t validRows = ShapedType::kDynamic;
    int64_t validCols = ShapedType::kDynamic;
    deriveValidShapeValues(op.getDst(), validRowsValue, validColsValue);
    deriveValidShape(op.getDst(), validRows, validCols);
    if (failed(resolveExecutionValidShape(op.getDst(), validRowsValue, validColsValue,
                                          validRows, validCols, rewriter, op.getLoc())))
      return op.emitOpError("tgather index lowering requires valid dst shape");

    int64_t chunkWidth = indexVecType.getElementCount();
    Value chunkStep = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), chunkWidth);
    Value dstStrideValue =
        rewriter.create<arith::ConstantIndexOp>(op.getLoc(), dstStride);
    Value indexStrideValue =
        rewriter.create<arith::ConstantIndexOp>(op.getLoc(), indexStride);

    auto rowLoop = rewriter.create<scf::ForOp>(op.getLoc(), c0, validRowsValue, c1);
    rewriter.setInsertionPointToStart(rowLoop.getBody());
    auto chunkLoop =
        rewriter.create<scf::ForOp>(op.getLoc(), c0, validColsValue, chunkStep);
    rewriter.setInsertionPointToStart(chunkLoop.getBody());

    Value row = rowLoop.getInductionVar();
    Value chunkBase = chunkLoop.getInductionVar();
    Value remaining =
        rewriter.create<arith::SubIOp>(op.getLoc(), validColsValue, chunkBase);
    Value activeLanes =
        buildMinIndexValue(rewriter, op.getLoc(), remaining, chunkStep);

    Value dstRowBase =
        rewriter.create<arith::MulIOp>(op.getLoc(), row, dstStrideValue);
    Value indexRowBase =
        rewriter.create<arith::MulIOp>(op.getLoc(), row, indexStrideValue);
    Value indexOffset =
        rewriter.create<arith::AddIOp>(op.getLoc(), indexRowBase, chunkBase);
    auto offsetVector = rewriter.create<pto::VldsOp>(op.getLoc(), indexVecType,
                                                      indexBuffer, indexOffset,
                                                      StringAttr());
    auto gathered = rewriter.create<pto::Vgather2Op>(
        op.getLoc(), dataVecType, srcBuffer, offsetVector.getResult(), activeLanes);
    Value dstOffset =
        rewriter.create<arith::AddIOp>(op.getLoc(), dstRowBase, chunkBase);
    return buildMaskedVectorStore(rewriter, op.getLoc(), gathered.getResult(),
                                  dstBuffer, dstOffset, activeLanes, chunkWidth);
  }

  auto maskPattern = op.getMaskPatternAttr();
  if (!maskPattern)
    return op.emitOpError("tgather lowering requires indices or maskPattern");
  if (maskPattern.getValue() != MaskPattern::P1111)
    return op.emitOpError("tgather mask lowering currently supports only maskPattern=P1111");

  Value validRowsValue;
  Value validColsValue;
  int64_t validRows = ShapedType::kDynamic;
  int64_t validCols = ShapedType::kDynamic;
  deriveValidShapeValues(op.getSrc(), validRowsValue, validColsValue);
  deriveValidShape(op.getSrc(), validRows, validCols);
  if (failed(resolveExecutionValidShape(op.getSrc(), validRowsValue, validColsValue,
                                        validRows, validCols, rewriter, op.getLoc())))
    return op.emitOpError("tgather mask lowering requires valid src shape");

  int64_t chunkWidth = dataVecType.getElementCount();
  Value chunkStep = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), chunkWidth);
  Value srcStrideValue =
      rewriter.create<arith::ConstantIndexOp>(op.getLoc(), srcStride);
  Value dstStrideValue =
      rewriter.create<arith::ConstantIndexOp>(op.getLoc(), dstStride);

  auto rowLoop = rewriter.create<scf::ForOp>(op.getLoc(), c0, validRowsValue, c1);
  rewriter.setInsertionPointToStart(rowLoop.getBody());
  auto chunkLoop =
      rewriter.create<scf::ForOp>(op.getLoc(), c0, validColsValue, chunkStep);
  rewriter.setInsertionPointToStart(chunkLoop.getBody());

  Value row = rowLoop.getInductionVar();
  Value chunkBase = chunkLoop.getInductionVar();
  Value remaining =
      rewriter.create<arith::SubIOp>(op.getLoc(), validColsValue, chunkBase);
  Value activeLanes = buildMinIndexValue(rewriter, op.getLoc(), remaining, chunkStep);

  Value srcRowBase =
      rewriter.create<arith::MulIOp>(op.getLoc(), row, srcStrideValue);
  Value dstRowBase =
      rewriter.create<arith::MulIOp>(op.getLoc(), row, dstStrideValue);
  Value srcOffset =
      rewriter.create<arith::AddIOp>(op.getLoc(), srcRowBase, chunkBase);
  auto loaded = rewriter.create<pto::VldsOp>(op.getLoc(), dataVecType, srcBuffer,
                                              srcOffset, StringAttr());
  Value dstOffset =
      rewriter.create<arith::AddIOp>(op.getLoc(), dstRowBase, chunkBase);
  return buildMaskedVectorStore(rewriter, op.getLoc(), loaded.getResult(), dstBuffer,
                                dstOffset, activeLanes, chunkWidth);
}

LogicalResult lowerTGatherB(TGatherBOp op, PatternRewriter &rewriter) {
  auto requireVecRowMajor = [&](Value value, StringRef role) -> LogicalResult {
    if (deriveTileDomain(getMemorySpace(value)) != VPTOTileDomain::Vec)
      return op.emitOpError() << "tgatherb lowering requires vec tile domain for "
                              << role;
    if (deriveTileLayout(value) != "row_major")
      return op.emitOpError() << "tgatherb lowering requires row-major layout for "
                              << role;
    return success();
  };

  if (failed(requireVecRowMajor(op.getSrc(), "src")) ||
      failed(requireVecRowMajor(op.getOffsets(), "offsets")) ||
      failed(requireVecRowMajor(op.getDst(), "dst")))
    return failure();

  Type dataElementType = getElementType(op.getDst());
  if (getElementType(op.getSrc()) != dataElementType)
    return op.emitOpError("tgatherb lowering requires matching src/dst element type");

  auto offsetIntegerType = dyn_cast<IntegerType>(getElementType(op.getOffsets()));
  if (!offsetIntegerType || offsetIntegerType.getWidth() != 32 ||
      !offsetIntegerType.isUnsigned())
    return op.emitOpError("tgatherb lowering currently requires unsigned 32-bit offsets");

  auto dataVecType = getVPTOVRegType(rewriter.getContext(), dataElementType);
  auto offsetVecType =
      getVPTOVRegType(rewriter.getContext(), getElementType(op.getOffsets()));
  if (!dataVecType || !offsetVecType)
    return op.emitOpError("tgatherb lowering requires supported VPTO vector types");

  Value srcBuffer = materializeBufferPointer(op.getSrc(), dataElementType,
                                             getMemorySpace(op.getSrc()), rewriter,
                                             op.getLoc());
  Value dstBuffer = materializeBufferPointer(op.getDst(), dataElementType,
                                             getMemorySpace(op.getDst()), rewriter,
                                             op.getLoc());
  Value offsetBuffer =
      materializeBufferPointer(op.getOffsets(), getElementType(op.getOffsets()),
                               getMemorySpace(op.getOffsets()), rewriter, op.getLoc());
  if (!srcBuffer || !dstBuffer || !offsetBuffer)
    return op.emitOpError("tgatherb lowering requires pointer-backed tile buffers");

  int64_t dstStride = deriveStaticRowStride(op.getDst());
  int64_t offsetStride = deriveStaticRowStride(op.getOffsets());
  int64_t staticRows = deriveStaticShapeDim(op.getDst(), 0);
  int64_t staticCols = deriveStaticShapeDim(op.getDst(), 1);
  if (dstStride == ShapedType::kDynamic || offsetStride == ShapedType::kDynamic ||
      staticRows == ShapedType::kDynamic || staticCols == ShapedType::kDynamic)
    return op.emitOpError("tgatherb lowering requires static tile shape and row stride");

  Value validRowsValue;
  Value validColsValue;
  int64_t validRows = ShapedType::kDynamic;
  int64_t validCols = ShapedType::kDynamic;
  deriveValidShapeValues(op.getDst(), validRowsValue, validColsValue);
  deriveValidShape(op.getDst(), validRows, validCols);
  if (failed(resolveExecutionValidShape(op.getDst(), validRowsValue, validColsValue,
                                        validRows, validCols, rewriter, op.getLoc())))
    return op.emitOpError("tgatherb lowering requires valid dst shape");

  unsigned elemBytes = dataElementType.getIntOrFloatBitWidth() / 8;
  int64_t elementsPerRepeat = 256 / elemBytes;
  int64_t blockSizeElem = 32 / elemBytes;
  int64_t staticRepeatTimes = llvm::divideCeil(staticCols, elementsPerRepeat);

  Value c0 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
  Value elementsPerRepeatValue =
      rewriter.create<arith::ConstantIndexOp>(op.getLoc(), elementsPerRepeat);
  Value blockSizeElemValue =
      rewriter.create<arith::ConstantIndexOp>(op.getLoc(), blockSizeElem);
  Value dstStrideValue =
      rewriter.create<arith::ConstantIndexOp>(op.getLoc(), dstStride);
  Value offsetStrideValue =
      rewriter.create<arith::ConstantIndexOp>(op.getLoc(), offsetStride);

  VPTOLoopScopeContract loopScope;
  loopScope.kind = VPTOLoopScopeKind::AIVVectorScope;
  loopScope.loweredAttr = kLoweredLoopScopeAttrName;
  loopScope.loopDepth = 0;

  FailureOr<pto::VecScopeOp> vecScope =
      createLoopScopeRegion(op.getLoc(), loopScope, rewriter);
  if (failed(vecScope))
    return op.emitOpError("failed to create AIV vector scope region");

  OpBuilder::InsertionGuard aivGuard(rewriter);
  rewriter.setInsertionPointToStart(&(*vecScope).getBody().front());

  if (staticRepeatTimes > staticRows) {
    auto rowLoop = rewriter.create<scf::ForOp>(op.getLoc(), c0, validRowsValue, c1);
    rewriter.setInsertionPointToStart(rowLoop.getBody());
    auto chunkLoop = rewriter.create<scf::ForOp>(op.getLoc(), c0, validColsValue,
                                                 elementsPerRepeatValue);
    rewriter.setInsertionPointToStart(chunkLoop.getBody());

    Value row = rowLoop.getInductionVar();
    Value chunkBase = chunkLoop.getInductionVar();
    Value remaining =
        rewriter.create<arith::SubIOp>(op.getLoc(), validColsValue, chunkBase);
    Value activeLanes = buildMinIndexValue(rewriter, op.getLoc(), remaining,
                                           elementsPerRepeatValue);
    Value rowOffsetBase =
        rewriter.create<arith::MulIOp>(op.getLoc(), row, offsetStrideValue);
    Value rowDstBase =
        rewriter.create<arith::MulIOp>(op.getLoc(), row, dstStrideValue);
    Value offsetChunkBase =
        rewriter.create<arith::FloorDivSIOp>(op.getLoc(), chunkBase,
                                             blockSizeElemValue);
    Value offsetLoadOffset =
        rewriter.create<arith::AddIOp>(op.getLoc(), rowOffsetBase, offsetChunkBase);
    auto offsets = rewriter.create<pto::VldsOp>(op.getLoc(), offsetVecType,
                                                 offsetBuffer, offsetLoadOffset,
                                                 StringAttr());
    auto gathered = rewriter.create<pto::VgatherbOp>(
        op.getLoc(), dataVecType, srcBuffer, offsets.getResult(), activeLanes);
    Value dstOffset =
        rewriter.create<arith::AddIOp>(op.getLoc(), rowDstBase, chunkBase);
    return buildMaskedVectorStore(rewriter, op.getLoc(), gathered.getResult(),
                                  dstBuffer, dstOffset, activeLanes,
                                  dataVecType.getElementCount());
  }

  auto chunkLoop = rewriter.create<scf::ForOp>(op.getLoc(), c0, validColsValue,
                                               elementsPerRepeatValue);
  rewriter.setInsertionPointToStart(chunkLoop.getBody());
  auto rowLoop = rewriter.create<scf::ForOp>(op.getLoc(), c0, validRowsValue, c1);
  rewriter.setInsertionPointToStart(rowLoop.getBody());

  Value chunkBase = chunkLoop.getInductionVar();
  Value row = rowLoop.getInductionVar();
  Value remaining =
      rewriter.create<arith::SubIOp>(op.getLoc(), validColsValue, chunkBase);
  Value activeLanes = buildMinIndexValue(rewriter, op.getLoc(), remaining,
                                         elementsPerRepeatValue);
  Value rowOffsetBase =
      rewriter.create<arith::MulIOp>(op.getLoc(), row, offsetStrideValue);
  Value rowDstBase =
      rewriter.create<arith::MulIOp>(op.getLoc(), row, dstStrideValue);
  Value offsetChunkBase =
      rewriter.create<arith::FloorDivSIOp>(op.getLoc(), chunkBase,
                                           blockSizeElemValue);
  Value offsetLoadOffset =
      rewriter.create<arith::AddIOp>(op.getLoc(), rowOffsetBase, offsetChunkBase);
  auto offsets = rewriter.create<pto::VldsOp>(op.getLoc(), offsetVecType, offsetBuffer,
                                               offsetLoadOffset, StringAttr());
  auto gathered = rewriter.create<pto::VgatherbOp>(
      op.getLoc(), dataVecType, srcBuffer, offsets.getResult(), activeLanes);
  Value dstOffset =
      rewriter.create<arith::AddIOp>(op.getLoc(), chunkBase, rowDstBase);
  return buildMaskedVectorStore(rewriter, op.getLoc(), gathered.getResult(),
                                dstBuffer, dstOffset, activeLanes,
                                dataVecType.getElementCount());
}

LogicalResult lowerTScatter(TScatterOp op, PatternRewriter &rewriter) {
  auto requireVecRowMajor = [&](Value value, StringRef role) -> LogicalResult {
    if (deriveTileDomain(getMemorySpace(value)) != VPTOTileDomain::Vec)
      return op.emitOpError() << "tscatter lowering requires vec tile domain for "
                              << role;
    if (deriveTileLayout(value) != "row_major")
      return op.emitOpError() << "tscatter lowering requires row-major layout for "
                              << role;
    return success();
  };

  if (failed(requireVecRowMajor(op.getSrc(), "src")) ||
      failed(requireVecRowMajor(op.getIndexes(), "indexes")) ||
      failed(requireVecRowMajor(op.getDst(), "dst")))
    return failure();

  Type dataElementType = getElementType(op.getSrc());
  if (dataElementType != getElementType(op.getDst()))
    return op.emitOpError("tscatter lowering requires matching src/dst element type");

  Type indexElementType = getElementType(op.getIndexes());
  auto indexIntegerType = dyn_cast<IntegerType>(indexElementType);
  if (!indexIntegerType || indexIntegerType.getWidth() != 32)
    return op.emitOpError("tscatter lowering currently requires 32-bit integer indexes");

  auto dataVecType = getVPTOVRegType(rewriter.getContext(), dataElementType);
  auto indexVecType = getVPTOVRegType(rewriter.getContext(), indexElementType);
  if (!dataVecType || !indexVecType ||
      dataVecType.getElementCount() != indexVecType.getElementCount())
    return op.emitOpError("tscatter lowering currently requires matching data/index vector widths");

  Value srcBuffer = materializeBufferPointer(op.getSrc(), dataElementType,
                                             getMemorySpace(op.getSrc()), rewriter,
                                             op.getLoc());
  Value dstBuffer = materializeBufferPointer(op.getDst(), dataElementType,
                                             getMemorySpace(op.getDst()), rewriter,
                                             op.getLoc());
  Value indexBuffer = materializeBufferPointer(op.getIndexes(), indexElementType,
                                               getMemorySpace(op.getIndexes()), rewriter,
                                               op.getLoc());
  if (!srcBuffer || !dstBuffer || !indexBuffer)
    return op.emitOpError("tscatter lowering requires pointer-backed tile buffers");

  int64_t srcStride = deriveStaticRowStride(op.getSrc());
  int64_t indexStride = deriveStaticRowStride(op.getIndexes());
  if (srcStride == ShapedType::kDynamic || indexStride == ShapedType::kDynamic)
    return op.emitOpError("tscatter lowering requires static src/index row stride");

  Value validRowsValue;
  Value validColsValue;
  int64_t validRows = ShapedType::kDynamic;
  int64_t validCols = ShapedType::kDynamic;
  deriveValidShapeValues(op.getIndexes(), validRowsValue, validColsValue);
  deriveValidShape(op.getIndexes(), validRows, validCols);
  if (failed(resolveExecutionValidShape(op.getIndexes(), validRowsValue, validColsValue,
                                        validRows, validCols, rewriter, op.getLoc())))
    return op.emitOpError("tscatter lowering requires valid index shape");

  Value c0 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
  Value chunkStep = rewriter.create<arith::ConstantIndexOp>(
      op.getLoc(), indexVecType.getElementCount());
  Value srcStrideValue =
      rewriter.create<arith::ConstantIndexOp>(op.getLoc(), srcStride);
  Value indexStrideValue =
      rewriter.create<arith::ConstantIndexOp>(op.getLoc(), indexStride);

  VPTOLoopScopeContract loopScope;
  loopScope.kind = VPTOLoopScopeKind::AIVVectorScope;
  loopScope.loweredAttr = kLoweredLoopScopeAttrName;
  loopScope.loopDepth = 0;

  FailureOr<pto::VecScopeOp> vecScope =
      createLoopScopeRegion(op.getLoc(), loopScope, rewriter);
  if (failed(vecScope))
    return op.emitOpError("failed to create AIV vector scope region");

  OpBuilder::InsertionGuard aivGuard(rewriter);
  rewriter.setInsertionPointToStart(&(*vecScope).getBody().front());
  auto rowLoop = rewriter.create<scf::ForOp>(op.getLoc(), c0, validRowsValue, c1);
  rewriter.setInsertionPointToStart(rowLoop.getBody());
  auto chunkLoop =
      rewriter.create<scf::ForOp>(op.getLoc(), c0, validColsValue, chunkStep);
  rewriter.setInsertionPointToStart(chunkLoop.getBody());

  Value row = rowLoop.getInductionVar();
  Value chunkBase = chunkLoop.getInductionVar();
  Value remaining =
      rewriter.create<arith::SubIOp>(op.getLoc(), validColsValue, chunkBase);
  Value activeLanes =
      buildMinIndexValue(rewriter, op.getLoc(), remaining, chunkStep);

  Value srcRowBase =
      rewriter.create<arith::MulIOp>(op.getLoc(), row, srcStrideValue);
  Value indexRowBase =
      rewriter.create<arith::MulIOp>(op.getLoc(), row, indexStrideValue);
  Value srcOffset =
      rewriter.create<arith::AddIOp>(op.getLoc(), srcRowBase, chunkBase);
  Value indexOffset =
      rewriter.create<arith::AddIOp>(op.getLoc(), indexRowBase, chunkBase);
  auto srcVector = rewriter.create<pto::VldsOp>(op.getLoc(), dataVecType, srcBuffer,
                                                 srcOffset, StringAttr());
  auto indexVector = rewriter.create<pto::VldsOp>(op.getLoc(), indexVecType, indexBuffer,
                                                   indexOffset, StringAttr());
  rewriter.create<pto::VscatterOp>(op.getLoc(), srcVector.getResult(), dstBuffer,
                                    indexVector.getResult(), activeLanes);
  return success();
}

LogicalResult lowerTMrgSort(TMrgSortOp op, PatternRewriter &rewriter) {
  auto requireVecRowMajor = [&](Value value, StringRef role) -> LogicalResult {
    if (deriveTileDomain(getMemorySpace(value)) != VPTOTileDomain::Vec)
      return op.emitOpError() << "tmrgsort lowering requires vec tile domain for "
                              << role;
    if (deriveTileLayout(value) != "row_major")
      return op.emitOpError() << "tmrgsort lowering requires row-major layout for "
                              << role;
    return success();
  };
  auto requireOneRow = [&](Value value, StringRef role) -> LogicalResult {
    if (deriveStaticShapeDim(value, 0) != 1)
      return op.emitOpError() << "tmrgsort lowering requires rows==1 for " << role;
    return success();
  };

  Location loc = op.getLoc();
  if (op.isFormat1()) {
    Value src = op.getSrcs().front();
    Value dst = op.getDsts().front();
    if (failed(requireVecRowMajor(src, "src")) || failed(requireVecRowMajor(dst, "dst")) ||
        failed(requireOneRow(src, "src")) || failed(requireOneRow(dst, "dst")))
      return failure();

    Type elementType = getElementType(src);
    if (elementType != getElementType(dst))
      return op.emitOpError("tmrgsort format1 requires matching src/dst element type");
    if (!(elementType.isF16() || elementType.isF32()))
      return op.emitOpError("tmrgsort format1 currently supports only f16/f32");

    Value srcBuffer = materializeBufferPointer(src, elementType, getMemorySpace(src),
                                              rewriter, loc);
    Value dstBuffer = materializeBufferPointer(dst, elementType, getMemorySpace(dst),
                                              rewriter, loc);
    if (!srcBuffer || !dstBuffer)
      return op.emitOpError("tmrgsort format1 requires pointer-backed tile buffers");

    Value blockLen = op.getBlockLen();
    if (!blockLen)
      return op.emitOpError("tmrgsort format1 requires blockLen");
    Value blockLenI64;
    if (blockLen.getType().isIndex())
      blockLenI64 =
          rewriter.create<arith::IndexCastUIOp>(loc, rewriter.getI64Type(), blockLen);
    else
      blockLenI64 =
          rewriter.create<arith::ExtUIOp>(loc, rewriter.getI64Type(), blockLen);
    Value blockLenIndex =
        rewriter.create<arith::IndexCastUIOp>(loc, rewriter.getIndexType(), blockLenI64);

    Value validRowsValue;
    Value validColsValue;
    int64_t validRows = ShapedType::kDynamic;
    int64_t validCols = ShapedType::kDynamic;
    deriveValidShapeValues(src, validRowsValue, validColsValue);
    deriveValidShape(src, validRows, validCols);
    Value validColsI64 = materializeI64Value(validColsValue, validCols, rewriter, loc);

    int64_t elemBytes = getElementByteSize(elementType);
    Value numStructures = rewriter.create<arith::ShRSIOp>(
        loc, rewriter.getI64Type(),
        rewriter.create<arith::MulIOp>(
            loc, blockLenI64, rewriter.create<arith::ConstantIntOp>(loc, elemBytes, 64)),
        rewriter.create<arith::ConstantIntOp>(loc, 3, 64));
    Value count = buildPackedCountI64(rewriter, loc,
                                      {numStructures, numStructures, numStructures, numStructures});
    Value repeatTimes = rewriter.create<arith::DivUIOp>(
        loc, validColsI64,
        rewriter.create<arith::MulIOp>(
            loc, blockLenI64, rewriter.create<arith::ConstantIntOp>(loc, 4, 64)));
    Value config = rewriter.create<arith::OrIOp>(
        loc, repeatTimes, rewriter.create<arith::ConstantIntOp>(loc, 0b1111 << 8, 64));

    Value src0 = srcBuffer;
    Value src1 = offsetBufferPointer(srcBuffer, elementType, blockLenIndex, rewriter, loc);
    Value src2 = offsetBufferPointer(
        srcBuffer, elementType,
        rewriter.create<arith::MulIOp>(loc, blockLenIndex,
                                       rewriter.create<arith::ConstantIndexOp>(loc, 2)),
        rewriter, loc);
    Value src3 = offsetBufferPointer(
        srcBuffer, elementType,
        rewriter.create<arith::MulIOp>(loc, blockLenIndex,
                                       rewriter.create<arith::ConstantIndexOp>(loc, 3)),
        rewriter, loc);
    rewriter.create<pto::Vmrgsort4Op>(loc, dstBuffer, src0, src1, src2, src3, count,
                                       config);
    return success();
  }

  if (!op.isFormat2())
    return op.emitOpError("unsupported tmrgsort format for current vpto backend");
  if (op.getExhausted())
    return op.emitOpError("tmrgsort format2 exhausted=true is not yet supported");
  if (op.getSrcs().size() != 4 || op.getDsts().size() != 2)
    return op.emitOpError("tmrgsort format2 currently requires exactly 4 srcs and 2 dsts");

  Type elementType = getElementType(op.getSrcs().front());
  if (!(elementType.isF16() || elementType.isF32()))
    return op.emitOpError("tmrgsort format2 currently supports only f16/f32");

  SmallVector<Value> srcBuffers;
  SmallVector<Value> srcCounts;
  srcBuffers.reserve(4);
  srcCounts.reserve(4);
  for (Value src : op.getSrcs()) {
    if (failed(requireVecRowMajor(src, "src")) || failed(requireOneRow(src, "src")))
      return failure();
    if (getElementType(src) != elementType)
      return op.emitOpError("tmrgsort format2 requires matching source element types");

    Value srcBuffer =
        materializeBufferPointer(src, elementType, getMemorySpace(src), rewriter, loc);
    if (!srcBuffer)
      return op.emitOpError("tmrgsort format2 requires pointer-backed source tiles");
    srcBuffers.push_back(srcBuffer);

    Value rowsValue;
    Value colsValue;
    int64_t rows = ShapedType::kDynamic;
    int64_t cols = ShapedType::kDynamic;
    deriveValidShapeValues(src, rowsValue, colsValue);
    deriveValidShape(src, rows, cols);
    Value colsI64 = materializeI64Value(colsValue, cols, rewriter, loc);
    srcCounts.push_back(rewriter.create<arith::ShRSIOp>(
        loc, rewriter.getI64Type(), colsI64,
        rewriter.create<arith::ConstantIntOp>(loc, elementType.isF32() ? 1 : 2, 64)));
  }

  Value dst = op.getDsts()[0];
  Value tmp = op.getDsts()[1];
  if (failed(requireVecRowMajor(dst, "dst")) || failed(requireVecRowMajor(tmp, "tmp")) ||
      failed(requireOneRow(dst, "dst")) || failed(requireOneRow(tmp, "tmp")))
    return failure();
  if (getElementType(dst) != elementType || getElementType(tmp) != elementType)
    return op.emitOpError("tmrgsort format2 requires matching dst/tmp element types");

  Value dstBuffer =
      materializeBufferPointer(dst, elementType, getMemorySpace(dst), rewriter, loc);
  Value tmpBuffer =
      materializeBufferPointer(tmp, elementType, getMemorySpace(tmp), rewriter, loc);
  if (!dstBuffer || !tmpBuffer)
    return op.emitOpError("tmrgsort format2 requires pointer-backed dst/tmp tiles");

  Value count = buildPackedCountI64(rewriter, loc, srcCounts);
  Value config =
      rewriter.create<arith::ConstantIntOp>(loc, 1 | (0b1111 << 8), 64);
  rewriter.create<pto::Vmrgsort4Op>(loc, tmpBuffer, srcBuffers[0], srcBuffers[1],
                                     srcBuffers[2], srcBuffers[3], count, config);

  Value dstRowsValue;
  Value dstColsValue;
  int64_t dstRows = ShapedType::kDynamic;
  int64_t dstCols = ShapedType::kDynamic;
  deriveValidShapeValues(dst, dstRowsValue, dstColsValue);
  deriveValidShape(dst, dstRows, dstCols);
  Value dstColsI64 = materializeI64Value(dstColsValue, dstCols, rewriter, loc);
  int64_t elemBytes = getElementByteSize(elementType);
  Value lenBurst = buildCeilDivPositiveI64(
      rewriter, loc,
      rewriter.create<arith::MulIOp>(
          loc, dstColsI64, rewriter.create<arith::ConstantIntOp>(loc, elemBytes, 64)),
      32);
  Value zeroI64 = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
  Value oneI64 = rewriter.create<arith::ConstantIntOp>(loc, 1, 64);
  rewriter.create<pto::CopyUbufToUbufOp>(loc, tmpBuffer, dstBuffer, zeroI64, oneI64,
                                          lenBurst, zeroI64, zeroI64);
  return success();
}

LogicalResult lowerTSort32(TSort32Op op, PatternRewriter &rewriter) {
  auto requireVecRowMajor = [&](Value value, StringRef role) -> LogicalResult {
    if (deriveTileDomain(getMemorySpace(value)) != VPTOTileDomain::Vec)
      return op.emitOpError() << "tsort32 lowering requires vec tile domain for "
                              << role;
    if (deriveTileLayout(value) != "row_major")
      return op.emitOpError() << "tsort32 lowering requires row-major layout for "
                              << role;
    return success();
  };

  if (failed(requireVecRowMajor(op.getSrc(), "src")) ||
      failed(requireVecRowMajor(op.getDst(), "dst")) ||
      failed(requireVecRowMajor(op.getIdx(), "idx")))
    return failure();

  Type dataType = getElementType(op.getSrc());
  if (dataType != getElementType(op.getDst()))
    return op.emitOpError("tsort32 lowering requires matching src/dst element type");
  if (!(dataType.isF16() || dataType.isF32()))
    return op.emitOpError("tsort32 lowering currently supports only f16/f32 data");
  auto idxType = dyn_cast<IntegerType>(getElementType(op.getIdx()));
  if (!idxType || idxType.getWidth() != 32 || !idxType.isUnsigned())
    return op.emitOpError("tsort32 lowering currently requires u32 index tile");

  Value srcBuffer =
      materializeBufferPointer(op.getSrc(), dataType, getMemorySpace(op.getSrc()),
                               rewriter, op.getLoc());
  Value dstBuffer =
      materializeBufferPointer(op.getDst(), dataType, getMemorySpace(op.getDst()),
                               rewriter, op.getLoc());
  Value idxBuffer = materializeBufferPointer(op.getIdx(), getElementType(op.getIdx()),
                                             getMemorySpace(op.getIdx()), rewriter,
                                             op.getLoc());
  if (!srcBuffer || !dstBuffer || !idxBuffer)
    return op.emitOpError("tsort32 lowering requires pointer-backed tiles");

  int64_t srcStride = deriveStaticRowStride(op.getSrc());
  int64_t dstStride = deriveStaticRowStride(op.getDst());
  int64_t idxStride = deriveStaticRowStride(op.getIdx());
  if (srcStride == ShapedType::kDynamic || dstStride == ShapedType::kDynamic ||
      idxStride == ShapedType::kDynamic)
    return op.emitOpError("tsort32 lowering requires static row stride");

  Value validRowsValue;
  Value validColsValue;
  int64_t validRows = ShapedType::kDynamic;
  int64_t validCols = ShapedType::kDynamic;
  deriveValidShapeValues(op.getSrc(), validRowsValue, validColsValue);
  deriveValidShape(op.getSrc(), validRows, validCols);
  if (validCols == ShapedType::kDynamic || (validCols % 32) != 0)
    return op.emitOpError("tsort32 lowering currently requires static validCol divisible by 32");

  int64_t idxValidRows = ShapedType::kDynamic;
  int64_t idxValidCols = ShapedType::kDynamic;
  deriveValidShape(op.getIdx(), idxValidRows, idxValidCols);
  bool idxBroadcast = idxValidRows == 1;

  Value c0 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
  Value repeatNumPerRow =
      rewriter.create<arith::ConstantIndexOp>(op.getLoc(), validCols / 32);
  Value srcStrideValue = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), srcStride);
  Value dstStrideValue = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), dstStride);
  Value idxStrideValue =
      rewriter.create<arith::ConstantIndexOp>(op.getLoc(), idxBroadcast ? 0 : idxStride);

  auto rowLoop = rewriter.create<scf::ForOp>(op.getLoc(), c0, validRowsValue, c1);
  rewriter.setInsertionPointToStart(rowLoop.getBody());
  Value row = rowLoop.getInductionVar();
  Value srcOffset = rewriter.create<arith::MulIOp>(op.getLoc(), row, srcStrideValue);
  Value dstOffset = rewriter.create<arith::MulIOp>(op.getLoc(), row, dstStrideValue);
  Value idxOffset = rewriter.create<arith::MulIOp>(op.getLoc(), row, idxStrideValue);
  Value rowSrcPtr =
      offsetBufferPointer(srcBuffer, dataType, srcOffset, rewriter, op.getLoc());
  Value rowDstPtr =
      offsetBufferPointer(dstBuffer, dataType, dstOffset, rewriter, op.getLoc());
  Value rowIdxPtr = offsetBufferPointer(idxBuffer, getElementType(op.getIdx()), idxOffset,
                                        rewriter, op.getLoc());
  rewriter.create<pto::VbitsortOp>(op.getLoc(), rowDstPtr, rowSrcPtr, rowIdxPtr,
                                    repeatNumPerRow);
  return success();
}

LogicalResult lowerTMulS(TMulSOp op, PatternRewriter &rewriter,
                         VPTOLoweringStrategy strategy) {
  VPTOUnaryContract contract = buildUnaryContract("muls", op.getSrc0());
  if (failed(checkGenericUnaryContract(
          op, contract, op.getDst(),
          [](Type type) {
            if (type.isF16() || type.isF32())
              return true;
            if (auto intType = dyn_cast<IntegerType>(type))
              return intType.getWidth() == 16 || intType.getWidth() == 32;
            return false;
          },
          "f16, f32, and 16/32-bit integer element types")))
    return failure();
  if (op.getScalar().getType() != contract.elementType)
    return op.emitOpError("tmuls lowering requires scalar type to match source element type");
  return buildScalarUnaryVecScope("muls", contract, strategy, op.getSrc0(), op.getScalar(),
                                  op.getDst(), rewriter, op.getLoc());
}

LogicalResult lowerTSelS(TSelSOp op, PatternRewriter &rewriter) {
  VPTOBinaryContract contract = buildBinaryContract("sels", op.getSrc());
  deriveValidShapeValues(op.getDst(), contract.validRowsValue, contract.validColsValue);
  deriveValidShape(op.getDst(), contract.validRows, contract.validCols);
  if (failed(checkGenericBinaryContract(
          op, contract, op.getTmp(), op.getDst(),
          [](Type type) {
            if (type.isF16() || type.isF32() || type.isBF16())
              return true;
            if (auto intType = dyn_cast<IntegerType>(type))
              return intType.getWidth() == 8 || intType.getWidth() == 16 ||
                     intType.getWidth() == 32;
            return false;
          },
          "f16, f32, bf16, and 8/16/32-bit integer element types")))
    return failure();

  auto selectModeType = dyn_cast<IntegerType>(op.getScalar().getType());
  if (!selectModeType)
    return op.emitOpError("tsels lowering requires integer selectMode");

  auto vecType = getVPTOVRegType(rewriter.getContext(), contract.elementType);
  if (!vecType)
    return op.emitOpError("tsels lowering requires a supported VPTO vector element type");

  Value src0Buffer = materializeBufferPointer(op.getSrc(), contract.elementType,
                                              getMemorySpace(op.getSrc()), rewriter,
                                              op.getLoc());
  Value src1Buffer = materializeBufferPointer(op.getTmp(), contract.elementType,
                                              getMemorySpace(op.getTmp()), rewriter,
                                              op.getLoc());
  Value dstBuffer = materializeBufferPointer(op.getDst(), contract.elementType,
                                             getMemorySpace(op.getDst()), rewriter,
                                             op.getLoc());
  if (!src0Buffer || !src1Buffer || !dstBuffer)
    return op.emitOpError("tsels lowering requires pointer-backed tile buffers");

  Value validRowsValue = materializeIndexValue(contract.validRowsValue,
                                               contract.validRows, rewriter, op.getLoc());
  Value validColsValue = materializeIndexValue(contract.validColsValue,
                                               contract.validCols, rewriter, op.getLoc());
  if (!validRowsValue || !validColsValue)
    return op.emitOpError("tsels lowering requires valid rows and cols");

  int64_t vectorWidth = vecType.getElementCount();
  if (contract.validRows != ShapedType::kDynamic &&
      contract.validCols != ShapedType::kDynamic) {
    int64_t totalElements = contract.validRows * contract.validCols;
    if (totalElements % vectorWidth != 0)
      return op.emitOpError(
          "tsels lowering currently requires total valid elements divisible by vector width");
  }

  Value c0 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
  Value totalElementsValue =
      rewriter.create<arith::MulIOp>(op.getLoc(), validRowsValue, validColsValue);
  Value vectorStepValue =
      rewriter.create<arith::ConstantIndexOp>(op.getLoc(), vectorWidth);

  FailureOr<pto::VecScopeOp> vecScope =
      createLoopScopeRegion(op.getLoc(), contract.loopScope, rewriter);
  if (failed(vecScope))
    return op.emitOpError("failed to create AIV vector scope region");

  OpBuilder::InsertionGuard aivGuard(rewriter);
  rewriter.setInsertionPointToStart(&(*vecScope).getBody().front());

  Value selectOne = rewriter.create<arith::ConstantOp>(
      op.getLoc(), IntegerAttr::get(selectModeType, 1));
  Value isAll = rewriter.create<arith::CmpIOp>(op.getLoc(), arith::CmpIPredicate::eq,
                                               op.getScalar(), selectOne);
  auto ifOp = rewriter.create<scf::IfOp>(
      op.getLoc(), TypeRange{getVPTOMaskTypeForElementType(rewriter.getContext(), contract.elementType)}, isAll,
      /*withElseRegion=*/true);
  rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
  Value allMask = rewriter
                      .create<pto::PsetB8Op>(op.getLoc(),
                                              getVPTOMaskTypeForElementType(rewriter.getContext(), contract.elementType),
                                              rewriter.getStringAttr("PAT_ALL"))
                      .getResult();
  rewriter.create<scf::YieldOp>(op.getLoc(), allMask);
  rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
  Value allfMask = rewriter
                       .create<pto::PsetB8Op>(op.getLoc(),
                                               getVPTOMaskTypeForElementType(rewriter.getContext(), contract.elementType),
                                               rewriter.getStringAttr("PAT_ALLF"))
                       .getResult();
  rewriter.create<scf::YieldOp>(op.getLoc(), allfMask);

  rewriter.setInsertionPointAfter(ifOp);
  auto chunkLoop =
      rewriter.create<scf::ForOp>(op.getLoc(), c0, totalElementsValue, vectorStepValue);
  rewriter.setInsertionPointToStart(chunkLoop.getBody());
  Value offset = chunkLoop.getInductionVar();
  Value mask = ifOp.getResult(0);
  auto src0Vec = rewriter.create<pto::VldsOp>(op.getLoc(), vecType, src0Buffer,
                                               offset, StringAttr());
  auto src1Vec = rewriter.create<pto::VldsOp>(op.getLoc(), vecType, src1Buffer,
                                               offset, StringAttr());
  Value selected = rewriter
                       .create<pto::VselOp>(op.getLoc(), vecType, src0Vec.getResult(),
                                             src1Vec.getResult(), mask)
                       .getResult();
  rewriter.create<pto::VstsOp>(
      op.getLoc(), selected, dstBuffer, offset, StringAttr(),
      buildAllPredicateMask(rewriter, op.getLoc(), contract.elementType));
  return success();
}

LogicalResult lowerTSel(TSelOp op, PatternRewriter &rewriter) {
  VPTOBinaryContract contract = buildBinaryContract("tsel", op.getSrc0());
  deriveValidShapeValues(op.getDst(), contract.validRowsValue, contract.validColsValue);
  deriveValidShape(op.getDst(), contract.validRows, contract.validCols);

  int64_t src1Rows = ShapedType::kDynamic;
  int64_t src1Cols = ShapedType::kDynamic;
  int64_t dstRows = ShapedType::kDynamic;
  int64_t dstCols = ShapedType::kDynamic;
  int64_t maskRows = ShapedType::kDynamic;
  int64_t maskCols = ShapedType::kDynamic;
  deriveValidShape(op.getSrc1(), src1Rows, src1Cols);
  deriveValidShape(op.getDst(), dstRows, dstCols);
  deriveValidShape(op.getMask(), maskRows, maskCols);

  if (contract.tileDomain != VPTOTileDomain::Vec ||
      deriveTileDomain(getMemorySpace(op.getSrc1())) != VPTOTileDomain::Vec ||
      deriveTileDomain(getMemorySpace(op.getDst())) != VPTOTileDomain::Vec ||
      deriveTileDomain(getMemorySpace(op.getMask())) != VPTOTileDomain::Vec)
    return op.emitOpError("tsel lowering requires tile domain vec");
  if (contract.tileLayout != "row_major" || deriveTileLayout(op.getSrc1()) != "row_major" ||
      deriveTileLayout(op.getDst()) != "row_major" || deriveTileLayout(op.getMask()) != "row_major")
    return op.emitOpError("tsel lowering requires row-major tile layout");
  if (contract.validRows == ShapedType::kDynamic ||
      contract.validCols == ShapedType::kDynamic)
    return op.emitOpError("tsel lowering requires static valid shape");
  if (contract.validRows != src1Rows || contract.validCols != src1Cols ||
      contract.validRows != dstRows || contract.validCols != dstCols ||
      contract.validRows != maskRows || contract.validCols != maskCols)
    return op.emitOpError("tsel lowering requires matching source, mask, and destination valid region");
  if (!contract.elementType || !contract.elementType.isF32())
    return op.emitOpError("tsel lowering currently supports only f32 data tiles");
  auto maskElemType = dyn_cast_or_null<IntegerType>(getElementType(op.getMask()));
  if (!maskElemType || maskElemType.getWidth() != 8)
    return op.emitOpError("tsel lowering currently requires i8 mask tiles");

  auto [tileRows, tileCols] = getStaticTileRowsCols(op.getDst());
  auto [maskTileRows, maskTileCols] = getStaticTileRowsCols(op.getMask());
  if (tileRows == ShapedType::kDynamic || tileCols == ShapedType::kDynamic ||
      maskTileRows == ShapedType::kDynamic || maskTileCols == ShapedType::kDynamic)
    return op.emitOpError("tsel lowering requires static tile rows and cols");
  Value maskBuffer = materializeBufferPointer(op.getMask(), getElementType(op.getMask()),
                                              getMemorySpace(op.getMask()), rewriter,
                                              op.getLoc());
  Value src0Buffer = materializeBufferPointer(op.getSrc0(), contract.elementType,
                                              getMemorySpace(op.getSrc0()), rewriter,
                                              op.getLoc());
  Value src1Buffer = materializeBufferPointer(op.getSrc1(), contract.elementType,
                                              getMemorySpace(op.getSrc1()), rewriter,
                                              op.getLoc());
  Value dstBuffer = materializeBufferPointer(op.getDst(), contract.elementType,
                                             getMemorySpace(op.getDst()), rewriter,
                                             op.getLoc());
  if (!maskBuffer || !src0Buffer || !src1Buffer || !dstBuffer)
    return op.emitOpError("tsel lowering requires pointer-backed tile buffers");

  auto vecType = getVPTOVRegType(rewriter.getContext(), contract.elementType);
  if (!vecType)
    return op.emitOpError("tsel lowering requires a supported VPTO vector element type");

  Value c0 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
  Value validRowsValue = materializeIndexValue(contract.validRowsValue, contract.validRows,
                                               rewriter, op.getLoc());
  if (!validRowsValue)
    return op.emitOpError("tsel lowering requires valid rows");
  Value rowStride = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), tileCols);
  Value maskStride = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), maskTileCols);
  constexpr int64_t elementsPerRepeat = 64;
  constexpr int64_t unrollConstant = 2;
  int64_t repeatTimes = (contract.validCols + elementsPerRepeat - 1) / elementsPerRepeat;
  int64_t pairedRepeatTimes = repeatTimes / unrollConstant;
  int64_t remainRepeat = repeatTimes % unrollConstant;
  int64_t repeatIdxBase = pairedRepeatTimes * unrollConstant;

  FailureOr<pto::VecScopeOp> vecScope =
      createLoopScopeRegion(op.getLoc(), contract.loopScope, rewriter);
  if (failed(vecScope))
    return op.emitOpError("failed to create AIV vector scope region");

  OpBuilder::InsertionGuard aivGuard(rewriter);
  rewriter.setInsertionPointToStart(&(*vecScope).getBody().front());
  auto splitMaskType = getVPTOMaskType(rewriter.getContext(), "b16");
  Value fullMask = rewriter
                       .create<pto::PsetB16Op>(op.getLoc(), splitMaskType,
                                                rewriter.getStringAttr("PAT_ALL"))
                       .getResult();
  auto rowLoop = rewriter.create<scf::ForOp>(op.getLoc(), c0, validRowsValue, c1);
  rewriter.setInsertionPointToStart(rowLoop.getBody());
  Value rowBase = rewriter.create<arith::MulIOp>(op.getLoc(), rowLoop.getInductionVar(), rowStride);
  Value maskBase = rewriter.create<arith::MulIOp>(op.getLoc(), rowLoop.getInductionVar(), maskStride);

  for (int64_t j = 0; j < pairedRepeatTimes; ++j) {
    int64_t repeatIdx = j * unrollConstant;
    int64_t colOffset0 = repeatIdx * elementsPerRepeat;
    int64_t colOffset1 = colOffset0 + elementsPerRepeat;
    int64_t maskOffsetImm = repeatIdx * 8;
    int64_t count0 = std::min<int64_t>(elementsPerRepeat, contract.validCols - colOffset0);
    int64_t count1 = std::min<int64_t>(elementsPerRepeat, contract.validCols - colOffset1);

    Value maskOffset = rewriter.create<arith::AddIOp>(
        op.getLoc(), maskBase,
        rewriter.create<arith::ConstantIndexOp>(op.getLoc(), maskOffsetImm));
    Value rawMask = rewriter
                        .create<pto::PldsOp>(op.getLoc(),
                                              splitMaskType,
                                              maskBuffer, maskOffset,
                                              rewriter.getStringAttr("US"))
                        .getResult();
    auto splitMask = rewriter.create<pto::PintlvB16Op>(
        op.getLoc(), splitMaskType, splitMaskType, rawMask, fullMask);

    Value dataOffset0 = rewriter.create<arith::AddIOp>(
        op.getLoc(), rowBase,
        rewriter.create<arith::ConstantIndexOp>(op.getLoc(), colOffset0));
    auto lhs0 = rewriter.create<pto::VldsOp>(op.getLoc(), vecType, src0Buffer,
                                              dataOffset0, StringAttr());
    auto rhs0 = rewriter.create<pto::VldsOp>(op.getLoc(), vecType, src1Buffer,
                                              dataOffset0, StringAttr());
    Value selected0 = rewriter
                          .create<pto::VselOp>(op.getLoc(), vecType, lhs0.getResult(),
                                                rhs0.getResult(), splitMask.getLow())
                          .getResult();
    Value storeMask0 = buildPredicateMaskForLaneCount(
        rewriter, op.getLoc(), contract.elementType,
        rewriter.create<arith::ConstantIndexOp>(op.getLoc(), count0));
    rewriter.create<pto::VstsOp>(op.getLoc(), selected0, dstBuffer, dataOffset0,
                                  StringAttr(), storeMask0);

    Value dataOffset1 = rewriter.create<arith::AddIOp>(
        op.getLoc(), rowBase,
        rewriter.create<arith::ConstantIndexOp>(op.getLoc(), colOffset1));
    auto lhs1 = rewriter.create<pto::VldsOp>(op.getLoc(), vecType, src0Buffer,
                                              dataOffset1, StringAttr());
    auto rhs1 = rewriter.create<pto::VldsOp>(op.getLoc(), vecType, src1Buffer,
                                              dataOffset1, StringAttr());
    Value selected1 = rewriter
                          .create<pto::VselOp>(op.getLoc(), vecType, lhs1.getResult(),
                                                rhs1.getResult(), splitMask.getHigh())
                          .getResult();
    Value storeMask1 = buildPredicateMaskForLaneCount(
        rewriter, op.getLoc(), contract.elementType,
        rewriter.create<arith::ConstantIndexOp>(op.getLoc(), count1));
    rewriter.create<pto::VstsOp>(op.getLoc(), selected1, dstBuffer, dataOffset1,
                                  StringAttr(), storeMask1);
  }

  for (int64_t j = 0; j < remainRepeat; ++j) {
    int64_t repeatIdx = repeatIdxBase + j;
    int64_t colOffset = repeatIdx * elementsPerRepeat;
    int64_t count = std::max<int64_t>(0, contract.validCols - colOffset);
    int64_t maskOffsetImm = repeatIdx * 8;

    Value maskOffset = rewriter.create<arith::AddIOp>(
        op.getLoc(), maskBase,
        rewriter.create<arith::ConstantIndexOp>(op.getLoc(), maskOffsetImm));
    Value rawMask = rewriter
                        .create<pto::PldsOp>(op.getLoc(),
                                              splitMaskType,
                                              maskBuffer, maskOffset,
                                              rewriter.getStringAttr("US"))
                        .getResult();
    Value unpackedMask = rewriter
                             .create<pto::PunpackOp>(
                                 op.getLoc(), splitMaskType,
                                 rawMask, rewriter.getStringAttr("LOWER"))
                             .getResult();
    Value dataOffset = rewriter.create<arith::AddIOp>(
        op.getLoc(), rowBase,
        rewriter.create<arith::ConstantIndexOp>(op.getLoc(), colOffset));
    auto lhs = rewriter.create<pto::VldsOp>(op.getLoc(), vecType, src0Buffer,
                                             dataOffset, StringAttr());
    auto rhs = rewriter.create<pto::VldsOp>(op.getLoc(), vecType, src1Buffer,
                                             dataOffset, StringAttr());
    Value selected = rewriter
                         .create<pto::VselOp>(op.getLoc(), vecType, lhs.getResult(),
                                               rhs.getResult(), unpackedMask)
                         .getResult();
    Value storeMask = buildPredicateMaskForLaneCount(
        rewriter, op.getLoc(), contract.elementType,
        rewriter.create<arith::ConstantIndexOp>(op.getLoc(), count));
    rewriter.create<pto::VstsOp>(op.getLoc(), selected, dstBuffer, dataOffset,
                                  StringAttr(), storeMask);
  }
  return success();
}

LogicalResult lowerTDivS(TDivSOp op, PatternRewriter &rewriter,
                         VPTOLoweringStrategy strategy) {
  Value tileOperand;
  Value scalarOperand;
  bool scalarFirst = false;
  if (isVPTOShapedLikeValue(op.getSrc()) && !isVPTOShapedLikeValue(op.getScalar())) {
    tileOperand = op.getSrc();
    scalarOperand = op.getScalar();
  } else if (!isVPTOShapedLikeValue(op.getSrc()) &&
             isVPTOShapedLikeValue(op.getScalar())) {
    tileOperand = op.getScalar();
    scalarOperand = op.getSrc();
    scalarFirst = true;
  } else {
    return op.emitOpError(
        "divs lowering requires exactly one shaped operand and one scalar operand");
  }

  VPTOUnaryContract contract = buildUnaryContract("divs", tileOperand);
  if (failed(checkGenericUnaryContract(
          op, contract, op.getDst(),
          [](Type type) { return type.isF16() || type.isF32(); },
          "f16 and f32 element types")))
    return failure();
  if (scalarOperand.getType() != contract.elementType)
    return op.emitOpError(
        "divs lowering requires scalar type to match source element type");
  return buildScalarDivVecScope(contract, strategy, tileOperand, scalarOperand, op.getDst(),
                                scalarFirst, rewriter, op.getLoc());
}

LogicalResult lowerTAddS(TAddSOp op, PatternRewriter &rewriter,
                         VPTOLoweringStrategy strategy) {
  VPTOUnaryContract contract = buildUnaryContract("adds", op.getSrc());
  if (failed(checkGenericUnaryContract(
          op, contract, op.getDst(),
          [](Type type) {
            if (type.isF16() || type.isF32() || type.isBF16())
              return true;
            if (auto intType = dyn_cast<IntegerType>(type))
              return intType.getWidth() == 16 || intType.getWidth() == 32;
            return false;
          },
          "f16, f32, bf16, and 16/32-bit integer element types")))
    return failure();
  if (op.getScalar().getType() != contract.elementType)
    return op.emitOpError("tadds lowering requires scalar type to match source element type");
  return buildScalarUnaryVecScope("adds", contract, strategy, op.getSrc(), op.getScalar(),
                                  op.getDst(), rewriter, op.getLoc());
}

LogicalResult lowerTAddC(TAddCOp op, PatternRewriter &rewriter) {
  VPTOBinaryContract first = buildBinaryContract("add", op.getSrc0());
  deriveValidShapeValues(op.getDst(), first.validRowsValue, first.validColsValue);
  deriveValidShape(op.getDst(), first.validRows, first.validCols);
  if (failed(checkGenericBinaryContract(
          op, first, op.getSrc1(), op.getDst(),
          [](Type type) {
            if (type.isF16() || type.isF32() || type.isBF16())
              return true;
            if (auto intType = dyn_cast<IntegerType>(type))
              return intType.getWidth() == 8 || intType.getWidth() == 16 ||
                     intType.getWidth() == 32;
            return false;
          },
          "f16, f32, bf16, and 8/16/32-bit integer element types")))
    return failure();
  if (failed(buildBinaryVecScope("add", first, VPTOLoweringStrategy::PostUpdate,
                                 op.getSrc0(), op.getSrc1(), op.getDst(),
                                 rewriter, op.getLoc())))
    return failure();

  VPTOBinaryContract second = buildBinaryContract("add", op.getDst());
  deriveValidShapeValues(op.getDst(), second.validRowsValue, second.validColsValue);
  deriveValidShape(op.getDst(), second.validRows, second.validCols);
  if (failed(checkGenericBinaryContract(
          op, second, op.getSrc2(), op.getDst(),
          [](Type type) {
            if (type.isF16() || type.isF32() || type.isBF16())
              return true;
            if (auto intType = dyn_cast<IntegerType>(type))
              return intType.getWidth() == 8 || intType.getWidth() == 16 ||
                     intType.getWidth() == 32;
            return false;
          },
          "f16, f32, bf16, and 8/16/32-bit integer element types")))
    return failure();
  return buildBinaryVecScope("add", second, VPTOLoweringStrategy::PostUpdate,
                             op.getDst(), op.getSrc2(), op.getDst(), rewriter,
                             op.getLoc());
}

LogicalResult lowerTAddSC(TAddSCOp op, PatternRewriter &rewriter) {
  return emitUnresolvedInstalledA5BaselineError(op, "taddsc");
}

LogicalResult lowerTSubC(TSubCOp op, PatternRewriter &rewriter) {
  VPTOBinaryContract first = buildBinaryContract("sub", op.getSrc0());
  deriveValidShapeValues(op.getDst(), first.validRowsValue, first.validColsValue);
  deriveValidShape(op.getDst(), first.validRows, first.validCols);
  if (failed(checkGenericBinaryContract(
          op, first, op.getSrc1(), op.getDst(),
          [](Type type) {
            if (type.isF16() || type.isF32() || type.isBF16())
              return true;
            if (auto intType = dyn_cast<IntegerType>(type))
              return intType.getWidth() == 8 || intType.getWidth() == 16 ||
                     intType.getWidth() == 32;
            return false;
          },
          "f16, f32, bf16, and 8/16/32-bit integer element types")))
    return failure();
  if (failed(buildBinaryVecScope("sub", first, VPTOLoweringStrategy::PostUpdate,
                                 op.getSrc0(), op.getSrc1(), op.getDst(),
                                 rewriter, op.getLoc())))
    return failure();

  VPTOBinaryContract second = buildBinaryContract("add", op.getDst());
  deriveValidShapeValues(op.getDst(), second.validRowsValue, second.validColsValue);
  deriveValidShape(op.getDst(), second.validRows, second.validCols);
  if (failed(checkGenericBinaryContract(
          op, second, op.getSrc2(), op.getDst(),
          [](Type type) {
            if (type.isF16() || type.isF32() || type.isBF16())
              return true;
            if (auto intType = dyn_cast<IntegerType>(type))
              return intType.getWidth() == 8 || intType.getWidth() == 16 ||
                     intType.getWidth() == 32;
            return false;
          },
          "f16, f32, bf16, and 8/16/32-bit integer element types")))
    return failure();
  return buildBinaryVecScope("add", second, VPTOLoweringStrategy::PostUpdate,
                             op.getDst(), op.getSrc2(), op.getDst(), rewriter,
                             op.getLoc());
}

LogicalResult lowerTSubS(TSubSOp op, PatternRewriter &rewriter,
                         VPTOLoweringStrategy strategy) {
  (void)rewriter;
  (void)strategy;
  return emitUnresolvedInstalledA5BaselineError(op, "tsubs");
}

LogicalResult lowerTSubSC(TSubSCOp op, PatternRewriter &rewriter) {
  return emitUnresolvedInstalledA5BaselineError(op, "tsubsc");
}

LogicalResult lowerTMaxS(TMaxSOp op, PatternRewriter &rewriter,
                         VPTOLoweringStrategy strategy) {
  VPTOUnaryContract contract = buildUnaryContract("maxs", op.getSrc());
  if (failed(checkGenericUnaryContract(
          op, contract, op.getDst(),
          [](Type type) { return type.isF32(); }, "f32 element type")))
    return failure();
  if (op.getScalar().getType() != contract.elementType)
    return op.emitOpError("tmaxs lowering requires scalar type to match source element type");
  return buildScalarUnaryVecScope("maxs", contract, strategy, op.getSrc(), op.getScalar(),
                                  op.getDst(), rewriter, op.getLoc());
}

LogicalResult lowerTMinS(TMinSOp op, PatternRewriter &rewriter,
                         VPTOLoweringStrategy strategy) {
  VPTOUnaryContract contract = buildUnaryContract("mins", op.getSrc());
  if (failed(checkGenericUnaryContract(
          op, contract, op.getDst(),
          [](Type type) { return type.isF32(); }, "f32 element type")))
    return failure();
  if (op.getScalar().getType() != contract.elementType)
    return op.emitOpError("tmins lowering requires scalar type to match source element type");
  return buildScalarUnaryVecScope("mins", contract, strategy, op.getSrc(), op.getScalar(),
                                  op.getDst(), rewriter, op.getLoc());
}

LogicalResult lowerTRowMax(TRowMaxOp op, PatternRewriter &rewriter,
                           VPTOLoweringStrategy strategy) {
  VPTORowReduceContract contract = extractTRowMaxContract(op);
  if (failed(checkRowReduceContract(op, contract, op.getDst())))
    return failure();
  return buildRowReduceVecScope("rowmax", contract, strategy, op.getSrc(), op.getDst(),
                                rewriter, op.getLoc());
}

LogicalResult lowerTRowMin(TRowMinOp op, PatternRewriter &rewriter,
                           VPTOLoweringStrategy strategy) {
  VPTORowReduceContract contract = extractTRowMinContract(op);
  if (failed(checkRowReduceContract(op, contract, op.getDst())))
    return failure();
  return buildRowReduceVecScope("rowmin", contract, strategy, op.getSrc(), op.getDst(),
                                rewriter, op.getLoc());
}

LogicalResult lowerTRowSum(TRowSumOp op, PatternRewriter &rewriter,
                           VPTOLoweringStrategy strategy) {
  VPTORowReduceContract contract = extractTRowSumContract(op);
  if (failed(checkRowReduceContract(op, contract, op.getDst())))
    return failure();
  return buildRowReduceVecScope("rowsum", contract, strategy, op.getSrc(), op.getDst(),
                                rewriter, op.getLoc());
}

LogicalResult lowerTColMax(TColMaxOp op, PatternRewriter &rewriter) {
  VPTOColReduceContract contract = extractTColMaxContract(op);
  if (failed(checkColReduceContract(op, contract, op.getDst())))
    return failure();
  return buildColReduceVecScope("colmax", contract, op.getSrc(), op.getDst(),
                                Value(), rewriter, op.getLoc());
}

LogicalResult lowerTColMin(TColMinOp op, PatternRewriter &rewriter) {
  VPTOColReduceContract contract = extractTColMinContract(op);
  if (failed(checkColReduceContract(op, contract, op.getDst())))
    return failure();
  return buildColReduceVecScope("colmin", contract, op.getSrc(), op.getDst(),
                                Value(), rewriter, op.getLoc());
}

LogicalResult lowerTColSum(TColSumOp op, PatternRewriter &rewriter) {
  VPTOColReduceContract contract = extractTColSumContract(op);
  if (failed(checkColReduceContract(op, contract, op.getDst())))
    return failure();
  return buildColReduceVecScope("colsum", contract, op.getSrc(), op.getDst(),
                                op.getTmp(), rewriter, op.getLoc());
}

LogicalResult lowerTRowExpand(TRowExpandOp op, PatternRewriter &rewriter,
                              VPTOLoweringStrategy strategy) {
  VPTOExpandContract contract = extractTRowExpandContract(op);
  if (failed(checkExpandContract(op, contract)))
    return failure();
  if (contract.srcValidRows != contract.dstValidRows)
    return op.emitOpError()
           << "rowexpand lowering requires source and destination valid rows to match";
  return buildRowExpandVecScope(contract, strategy, op.getSrc(), op.getDst(), rewriter,
                                op.getLoc());
}

LogicalResult lowerTColExpand(TColExpandOp op, PatternRewriter &rewriter) {
  VPTOExpandContract contract = extractTColExpandContract(op);
  if (failed(checkExpandContract(op, contract)))
    return failure();
  if (contract.srcValidCols != contract.dstValidCols)
    return op.emitOpError()
           << "colexpand lowering requires source and destination valid cols to match";
  return buildColExpandVecScope(contract, op.getSrc(), op.getDst(), rewriter,
                                op.getLoc());
}

template <typename OpTy>
LogicalResult lowerTRowExpandBinaryLike(OpTy op, PatternRewriter &rewriter,
                                        StringRef family,
                                        VPTOLoweringStrategy strategy) {
  Type elementType = getElementType(op.getDst());
  if (!elementType || (!elementType.isF16() && !elementType.isF32()))
    return op.emitOpError() << family
                            << " lowering currently supports only f16 and f32 element types";

  if (deriveTileDomain(getMemorySpace(op.getDst())) != VPTOTileDomain::Vec ||
      deriveTileDomain(getMemorySpace(op.getSrc0())) != VPTOTileDomain::Vec ||
      deriveTileDomain(getMemorySpace(op.getSrc1())) != VPTOTileDomain::Vec)
    return op.emitOpError() << family << " lowering requires vec tile domain";
  if (deriveTileLayout(op.getDst()) != "row_major")
    return op.emitOpError() << family << " lowering requires row-major dst layout";

  int64_t dstValidRows = ShapedType::kDynamic;
  int64_t dstValidCols = ShapedType::kDynamic;
  int64_t src0ValidRows = ShapedType::kDynamic;
  int64_t src0ValidCols = ShapedType::kDynamic;
  int64_t src1ValidRows = ShapedType::kDynamic;
  int64_t src1ValidCols = ShapedType::kDynamic;
  deriveValidShape(op.getDst(), dstValidRows, dstValidCols);
  deriveValidShape(op.getSrc0(), src0ValidRows, src0ValidCols);
  deriveValidShape(op.getSrc1(), src1ValidRows, src1ValidCols);
  if (dstValidRows == ShapedType::kDynamic || dstValidCols == ShapedType::kDynamic ||
      src0ValidRows == ShapedType::kDynamic || src0ValidCols == ShapedType::kDynamic ||
      src1ValidRows == ShapedType::kDynamic || src1ValidCols == ShapedType::kDynamic)
    return op.emitOpError() << family
                            << " lowering currently requires static valid shapes";

  bool src0EqDst = op.getSrc0().getType() == op.getDst().getType();
  bool src1EqDst = op.getSrc1().getType() == op.getDst().getType();
  if (!src0EqDst && !src1EqDst)
    return op.emitOpError() << family
                            << " lowering requires src0 or src1 to match dst tile type";

  Value baseSrc = src0EqDst ? op.getSrc0() : op.getSrc1();
  Value expandSrc = src0EqDst ? op.getSrc1() : op.getSrc0();
  StringRef expandLayout = deriveTileLayout(expandSrc);
  int64_t expandValidRows = src0EqDst ? src1ValidRows : src0ValidRows;
  int64_t expandValidCols = src0EqDst ? src1ValidCols : src0ValidCols;
  if (expandValidRows != dstValidRows)
    return op.emitOpError() << family
                            << " lowering requires expand operand valid rows to match dst";

  int64_t elemBytes = getElementByteSize(elementType);
  bool expandIsRowMajor = expandLayout == "row_major" && expandValidCols == 32 / elemBytes;
  bool expandIsColMajor = expandLayout == "col_major" && expandValidCols == 1;
  if (!expandIsRowMajor && !expandIsColMajor)
    return op.emitOpError() << family
                            << " lowering requires PTO A5-compatible expand operand shape";

  auto vecType = getVPTOVRegType(rewriter.getContext(), elementType);
  if (!vecType)
    return op.emitOpError() << family
                            << " lowering requires a legal VPTO vector type";

  Value baseBuffer = materializeBufferPointer(baseSrc, elementType,
                                              getMemorySpace(baseSrc), rewriter,
                                              op.getLoc());
  Value expandBuffer = materializeBufferPointer(expandSrc, elementType,
                                                getMemorySpace(expandSrc), rewriter,
                                                op.getLoc());
  Value dstBuffer = materializeBufferPointer(op.getDst(), elementType,
                                             getMemorySpace(op.getDst()), rewriter,
                                             op.getLoc());
  if (!baseBuffer || !expandBuffer || !dstBuffer)
    return op.emitOpError() << family
                            << " lowering requires pointer-backed tile buffers";

  int64_t dstRowStride = deriveStaticRowStride(op.getDst());
  int64_t baseRowStride = deriveStaticRowStride(baseSrc);
  int64_t expandRowStride = deriveStaticRowStride(expandSrc);
  if (dstRowStride == ShapedType::kDynamic || baseRowStride == ShapedType::kDynamic ||
      expandRowStride == ShapedType::kDynamic)
    return op.emitOpError() << family << " lowering requires static row strides";

  Value c0 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
  Value rowsUpper = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), dstValidRows);
  Value colsUpper = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), dstValidCols);
  Value vectorStep =
      rewriter.create<arith::ConstantIndexOp>(op.getLoc(), vecType.getElementCount());
  Value baseStrideValue =
      rewriter.create<arith::ConstantIndexOp>(op.getLoc(), baseRowStride);
  Value expandStrideValue =
      rewriter.create<arith::ConstantIndexOp>(op.getLoc(), expandRowStride);
  Value dstStrideValue =
      rewriter.create<arith::ConstantIndexOp>(op.getLoc(), dstRowStride);
  Value blockSizeValue =
      rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 32 / elemBytes);

  VPTOLoopScopeContract loopScope;
  loopScope.kind = VPTOLoopScopeKind::AIVVectorScope;
  loopScope.loweredAttr = kLoweredLoopScopeAttrName;
  loopScope.loopDepth = 0;

  auto buildRowExpandValue = [&](Value baseVec, Value expandedVec,
                                 Value predicate) -> FailureOr<Value> {
    if (family == "trowexpandmul")
      return rewriter.create<pto::VmulOp>(op.getLoc(), vecType, baseVec,
                                           expandedVec, predicate)
          .getResult();
    if (family == "trowexpanddiv") {
      if (src0EqDst)
        return rewriter.create<pto::VdivOp>(op.getLoc(), vecType, baseVec,
                                             expandedVec, predicate)
            .getResult();
      return rewriter.create<pto::VdivOp>(op.getLoc(), vecType, expandedVec,
                                           baseVec, predicate)
          .getResult();
    }
    if (family == "trowexpandsub") {
      if (src0EqDst)
        return rewriter.create<pto::VsubOp>(op.getLoc(), vecType, baseVec,
                                             expandedVec, predicate)
            .getResult();
      return rewriter.create<pto::VsubOp>(op.getLoc(), vecType, expandedVec,
                                           baseVec, predicate)
          .getResult();
    }
    return failure();
  };

  FailureOr<pto::VecScopeOp> vecScope =
      createLoopScopeRegion(op.getLoc(), loopScope, rewriter);
  if (failed(vecScope))
    return op.emitOpError("failed to create AIV vector scope region");

  OpBuilder::InsertionGuard aivGuard(rewriter);
  rewriter.setInsertionPointToStart(&(*vecScope).getBody().front());
  auto rowLoop = rewriter.create<scf::ForOp>(op.getLoc(), c0, rowsUpper, c1);
  rewriter.setInsertionPointToStart(rowLoop.getBody());
  Value row = rowLoop.getInductionVar();
  Value baseRowOffset = rewriter.create<arith::MulIOp>(op.getLoc(), row, baseStrideValue);
  Value dstRowOffset = rewriter.create<arith::MulIOp>(op.getLoc(), row, dstStrideValue);
  Value expandRowOffset = expandIsRowMajor
                              ? rewriter.create<arith::MulIOp>(op.getLoc(), row, blockSizeValue)
                              : rewriter.create<arith::MulIOp>(op.getLoc(), row, expandStrideValue);

  Value expandVec;
  if (expandIsColMajor) {
    Value fullMask = buildAllPredicateMask(rewriter, op.getLoc(), elementType);
    Value expandScalar =
        rewriter.create<pto::UvldOp>(op.getLoc(), vecType, expandBuffer,
                                      expandRowOffset);
    expandVec = rewriter
                    .create<pto::VdupOp>(op.getLoc(), vecType, expandScalar, fullMask,
                                          StringAttr())
                    .getResult();
  } else {
    expandVec = rewriter
                    .create<pto::VldsOp>(op.getLoc(), vecType, expandBuffer, expandRowOffset,
                                          rewriter.getStringAttr("BLK"))
                    .getResult();
  }

  auto colLoop = rewriter.create<scf::ForOp>(op.getLoc(), c0, colsUpper, vectorStep);
  rewriter.setInsertionPointToStart(colLoop.getBody());
  Value col = colLoop.getInductionVar();
  Value remainingCols = rewriter.create<arith::SubIOp>(op.getLoc(), colsUpper, col);
  Value needsTailMask = rewriter.create<arith::CmpIOp>(
      op.getLoc(), arith::CmpIPredicate::slt, remainingCols, vectorStep);
  Value activeLanes = rewriter.create<arith::SelectOp>(op.getLoc(), needsTailMask,
                                                       remainingCols, vectorStep);
  Value baseOffset = rewriter.create<arith::AddIOp>(op.getLoc(), baseRowOffset, col);
  Value dstOffset = rewriter.create<arith::AddIOp>(op.getLoc(), dstRowOffset, col);
  Value storeMask =
      buildPredicateMaskForLaneCount(rewriter, op.getLoc(), elementType, activeLanes);
  Value baseVec =
      rewriter.create<pto::VldsOp>(op.getLoc(), vecType, baseBuffer, baseOffset, StringAttr());
  FailureOr<Value> computed =
      buildRowExpandValue(baseVec, expandVec, storeMask);
  if (failed(computed))
    return op.emitOpError() << "unsupported rowexpand binary family";
  rewriter.create<pto::VstsOp>(op.getLoc(), *computed, dstBuffer, dstOffset,
                                StringAttr(), storeMask);
  rewriter.create<scf::YieldOp>(op.getLoc());
  return success();
}

LogicalResult lowerTRowExpandMul(TRowExpandMulOp op, PatternRewriter &rewriter,
                                 VPTOLoweringStrategy strategy) {
  return lowerTRowExpandBinaryLike(op, rewriter, "trowexpandmul", strategy);
}

LogicalResult lowerTRowExpandDiv(TRowExpandDivOp op, PatternRewriter &rewriter,
                                 VPTOLoweringStrategy strategy) {
  return lowerTRowExpandBinaryLike(op, rewriter, "trowexpanddiv", strategy);
}

LogicalResult lowerTRowExpandSub(TRowExpandSubOp op, PatternRewriter &rewriter,
                                 VPTOLoweringStrategy strategy) {
  return lowerTRowExpandBinaryLike(op, rewriter, "trowexpandsub", strategy);
}

LogicalResult lowerTPartAdd(TPartAddOp op, PatternRewriter &rewriter) {
  VPTOPartContract contract = extractTPartAddContract(op);
  if (failed(checkPartContract(op, contract)))
    return failure();
  return buildPartVecScope("partadd", contract, op.getSrc0(), op.getSrc1(),
                           op.getDst(), rewriter, op.getLoc());
}

LogicalResult lowerTPartMax(TPartMaxOp op, PatternRewriter &rewriter) {
  VPTOPartContract contract = extractTPartMaxContract(op);
  if (failed(checkPartContract(op, contract)))
    return failure();
  return buildPartVecScope("partmax", contract, op.getSrc0(), op.getSrc1(),
                           op.getDst(), rewriter, op.getLoc());
}

LogicalResult lowerTPartMin(TPartMinOp op, PatternRewriter &rewriter) {
  VPTOPartContract contract = extractTPartMinContract(op);
  if (failed(checkPartContract(op, contract)))
    return failure();
  return buildPartVecScope("partmin", contract, op.getSrc0(), op.getSrc1(),
                           op.getDst(), rewriter, op.getLoc());
}

LogicalResult lowerTSTORE(TStoreOp op, PatternRewriter &rewriter) {
  VPTOStoreContract contract = extractTStoreContract(op);

  switch (contract.srcDomain) {
  case VPTOTileDomain::Acc:
    return lowerUnsupportedAccStore(op.getLoc());
  case VPTOTileDomain::Mat:
    return lowerUnsupportedMatStore(op.getLoc());
  case VPTOTileDomain::Vec:
    break;
  }

  ResolvedTensorView destinationView;
  if (!resolveTensorView(op.getDst(), destinationView, rewriter, op.getLoc()))
    return op.emitOpError("requires a recoverable destination tensor view for VPTO lowering");

  StringRef sourceTileLayout = deriveTileLayout(op.getSrc());
  StringRef destinationLayout =
      inferVecTransferLayoutFromTile(stringifyLayoutAttr(destinationView.layoutAttr),
                                     sourceTileLayout);
  bool isNdStore = sourceTileLayout == "row_major" && destinationLayout == "nd";
  bool isDnStore = sourceTileLayout == "col_major" && destinationLayout == "dn";
  if (!isNdStore && !isDnStore)
    return op.emitOpError("currently supports only ND row_major or DN col_major vec TSTORE lowering");

  Value sourceBuffer =
      materializeBufferPointer(op.getSrc(), contract.elementType,
                               getMemorySpace(op.getSrc()), rewriter, op.getLoc());
  Value destinationBuffer =
      materializeBufferPointer(destinationView.root, getElementType(destinationView.root),
                               getGmMemorySpace(rewriter.getContext()), rewriter,
                               op.getLoc());
  if (!sourceBuffer || !destinationBuffer)
    return op.emitOpError("requires A5-compatible source and destination buffers");

  auto [tileRows, tileCols] = getStaticTileRowsCols(op.getSrc());
  Value validRowsValue =
      materializeI64Value(contract.validRowsValue, contract.validRows, rewriter,
                          op.getLoc());
  Value validColsValue =
      materializeI64Value(contract.validColsValue, contract.validCols, rewriter,
                          op.getLoc());
  Value sidValue = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0, 64);
  int64_t elemBytes = getElementByteSize(contract.elementType);
  if ((isNdStore && tileCols == ShapedType::kDynamic) ||
      (isDnStore && tileRows == ShapedType::kDynamic) || elemBytes <= 0)
    return op.emitOpError("requires static tile shape for A5-compatible transfer arguments");
  VecNdTransferPlan plan;
  LogicalResult planResult =
      isNdStore ? buildVecNdStorePlan(destinationView.shape, destinationView.strides,
                                      tileCols, contract.validColsValue,
                                      contract.validCols, contract.elementType,
                                      rewriter, op.getLoc(), plan)
                : buildVecDnStorePlan(destinationView.shape, destinationView.strides,
                                      tileRows, contract.validRowsValue,
                                      contract.validRows, contract.elementType,
                                      rewriter, op.getLoc(), plan);
  if (failed(planResult))
    return op.emitOpError("requires PTO-compatible vec copy_ubuf_to_gm arguments");
  Value reservedValue = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0, 64);
  if (!validRowsValue || !validColsValue)
    return op.emitOpError("requires valid rows and cols for A5-compatible transfer arguments");
  Value destinationOffset =
      materializeI64Ofr(destinationView.offsetElems, rewriter, op.getLoc());
  if (!destinationOffset)
    return op.emitOpError("requires a materializable destination offset for VPTO lowering");
  Value destinationBase =
      adjustPointerByElemOffset(destinationBuffer, destinationOffset, elemBytes, rewriter,
                                op.getLoc());
  if (!destinationBase)
    return op.emitOpError("failed to materialize destination base pointer");

  rewriter.create<pto::SetLoopSizeUbToOutOp>(op.getLoc(), plan.loop2Size,
                                              plan.loop1Size);
  rewriter.create<pto::SetLoop1StrideUbToOutOp>(
      op.getLoc(), plan.loop1FirstStrideBytes, plan.loop1SecondStrideBytes);
  rewriter.create<pto::SetLoop2StrideUbToOutOp>(
      op.getLoc(), plan.loop2FirstStrideBytes, plan.loop2SecondStrideBytes);

  auto emitCopy = [&](Value srcPtr, Value dstPtr) {
    Type transferElementType =
        getCopyTransferElementType(contract.elementType, rewriter);
    Value typedSrcPtr =
        castPtrToElementType(srcPtr, transferElementType, rewriter, op.getLoc());
    Value typedDstPtr =
        castPtrToElementType(dstPtr, transferElementType, rewriter, op.getLoc());
    if (!typedSrcPtr || !typedDstPtr)
      return failure();
    rewriter.create<pto::CopyUbufToGmOp>(
        op.getLoc(), typedSrcPtr, typedDstPtr, sidValue, plan.nBurst,
        plan.lenBurst, reservedValue, plan.firstStrideBytes,
        plan.secondStrideBytes);
    return success();
  };

  if (std::optional<int64_t> outerConst = getConstInt(plan.outerCount); outerConst && *outerConst == 1) {
    return emitCopy(sourceBuffer, destinationBase);
  }

  Value c0 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
  Value outerUpper =
      rewriter.create<arith::IndexCastUIOp>(op.getLoc(), rewriter.getIndexType(),
                                            plan.outerCount);
  auto outerLoop = rewriter.create<scf::ForOp>(op.getLoc(), c0, outerUpper, c1);
  rewriter.setInsertionPointToStart(outerLoop.getBody());
  Value ivI64 = rewriter.create<arith::IndexCastUIOp>(op.getLoc(), rewriter.getI64Type(),
                                                      outerLoop.getInductionVar());
  Value srcStep = createI64Mul(ivI64, plan.outerSrcStrideElems, rewriter, op.getLoc());
  Value dstStep = createI64Mul(ivI64, plan.outerDstStrideElems, rewriter, op.getLoc());
  Value iterSrc = adjustPointerByElemOffset(sourceBuffer, srcStep, elemBytes, rewriter,
                                            op.getLoc());
  Value iterDst = adjustPointerByElemOffset(destinationBase, dstStep, elemBytes, rewriter,
                                            op.getLoc());
  return emitCopy(iterSrc, iterDst);
}

LogicalResult lowerSetFlag(SetFlagOp op, PatternRewriter &rewriter) {
  rewriter.create<pto::SetFlagOp>(op.getLoc(),
                                   stringifyPipeAttr(op.getSrcPipe(), rewriter),
                                   stringifyPipeAttr(op.getDstPipe(), rewriter),
                                   stringifyEventAttr(op.getEventId(), rewriter));
  return success();
}

LogicalResult lowerWaitFlag(WaitFlagOp op, PatternRewriter &rewriter) {
  rewriter.create<pto::WaitFlagOp>(op.getLoc(),
                                    stringifyPipeAttr(op.getSrcPipe(), rewriter),
                                    stringifyPipeAttr(op.getDstPipe(), rewriter),
                                    stringifyEventAttr(op.getEventId(), rewriter));
  return success();
}

LogicalResult lowerBarrier(BarrierOp op, PatternRewriter &rewriter) {
  rewriter.create<pto::BarrierOp>(op.getLoc(),
                                       stringifyPipeAttr(op.getPipe(), rewriter));
  return success();
}

static FailureOr<PipeAttr> stringifyConcreteSyncPipeAttr(Attribute opTypeAttr,
                                                         PatternRewriter &rewriter) {
  if (auto pipeAttr = dyn_cast<PipeAttr>(opTypeAttr))
    return PipeAttr::get(rewriter.getContext(), pipeAttr.getPipe());
  auto opTypeOr = parseSyncOpTypeLikeAttr(opTypeAttr);
  if (failed(opTypeOr))
    return failure();
  PIPE pipe = mapSyncOpTypeToPipe(*opTypeOr);
  if (!isConcreteSyncPipe(pipe))
    return failure();
  return PipeAttr::get(rewriter.getContext(), pipe);
}

LogicalResult lowerGetBuf(GetBufOp op, PatternRewriter &rewriter) {
  FailureOr<PipeAttr> pipeAttr =
      stringifyConcreteSyncPipeAttr(op.getOpTypeAttr(), rewriter);
  if (failed(pipeAttr))
    return op.emitOpError("get_buf expects SyncOpType/PipeEventType that maps to a concrete pipe");

  rewriter.create<pto::GetBufOp>(op.getLoc(), Attribute(*pipeAttr),
                                 static_cast<uint32_t>(op.getBufId()),
                                 static_cast<uint32_t>(op.getMode()));
  return success();
}

LogicalResult lowerRlsBuf(RlsBufOp op, PatternRewriter &rewriter) {
  FailureOr<PipeAttr> pipeAttr =
      stringifyConcreteSyncPipeAttr(op.getOpTypeAttr(), rewriter);
  if (failed(pipeAttr))
    return op.emitOpError("rls_buf expects SyncOpType/PipeEventType that maps to a concrete pipe");

  rewriter.create<pto::RlsBufOp>(op.getLoc(), Attribute(*pipeAttr),
                                 static_cast<uint32_t>(op.getBufId()),
                                 static_cast<uint32_t>(op.getMode()));
  return success();
}

namespace {

static Type convertVPTOBoundaryMemRefType(Type type) {
  auto memrefType = dyn_cast<BaseMemRefType>(type);
  if (!memrefType)
    return type;
  auto memorySpace = dyn_cast_or_null<AddressSpaceAttr>(memrefType.getMemorySpace());
  if (!memorySpace)
    return {};
  return PtrType::get(type.getContext(), memrefType.getElementType(), memorySpace);
}

static LogicalResult eraseDeadVPTOMemRefScaffold(ModuleOp module) {
  bool erasedAny = true;
  while (erasedAny) {
    erasedAny = false;
    SmallVector<Operation *> deadOps;
    module.walk([&](Operation *op) {
      if (!op->use_empty())
        return;
      if (isa<memref::ReinterpretCastOp, memref::SubViewOp,
              memref::MemorySpaceCastOp>(op))
        deadOps.push_back(op);
    });
    for (Operation *op : deadOps) {
      op->erase();
      erasedAny = true;
    }
  }
  return success();
}

static LogicalResult verifyNoResidualVPTOMemRefs(ModuleOp module,
                                                 llvm::raw_ostream *diagOS) {
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    for (Type input : func.getFunctionType().getInputs()) {
      if (!isa<BaseMemRefType>(input))
        continue;
      if (diagOS)
        *diagOS << "VPTO ptr-only boundary failed: residual memref argument in "
                << func.getName() << ": " << input << "\n";
      return failure();
    }
    for (Type result : func.getFunctionType().getResults()) {
      if (!isa<BaseMemRefType>(result))
        continue;
      if (diagOS)
        *diagOS << "VPTO ptr-only boundary failed: residual memref result in "
                << func.getName() << ": " << result << "\n";
      return failure();
    }
  }

  WalkResult walk = module.walk([&](Operation *op) {
    auto hasResidualMemRef = [](TypeRange types) {
      return llvm::any_of(types, [](Type type) {
        return isa<BaseMemRefType>(type);
      });
    };
    if (hasResidualMemRef(op->getOperandTypes()) ||
        hasResidualMemRef(op->getResultTypes())) {
      if (diagOS) {
        *diagOS << "VPTO ptr-only boundary failed: residual memref-typed op "
                << op->getName() << "\n";
        op->print(*diagOS);
        *diagOS << "\n";
      }
      return WalkResult::interrupt();
    }
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (BlockArgument arg : block.getArguments()) {
          if (!isa<BaseMemRefType>(arg.getType()))
            continue;
          if (diagOS)
            *diagOS << "VPTO ptr-only boundary failed: residual memref block "
                    << "argument in op " << op->getName() << ": "
                    << arg.getType() << "\n";
          return WalkResult::interrupt();
        }
      }
    }
    return WalkResult::advance();
  });
  return walk.wasInterrupted() ? failure() : success();
}

} // namespace

LogicalResult convertVPTOFunctionBoundariesToPtr(ModuleOp module,
                                                 llvm::raw_ostream *diagOS) {
  // VPTO kernels use ptr-only entry semantics: the function ABI keeps only the
  // same-space base pointer, while shape/stride/offset stay in live SSA and
  // address calculations inside the body.
  if (failed(eraseDeadVPTOMemRefScaffold(module)))
    return failure();

  bool sawFailure = false;
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    if (func.isExternal())
      continue;

    FunctionType functionType = func.getFunctionType();
    SmallVector<Type> newInputs(functionType.getInputs().begin(),
                                functionType.getInputs().end());
    bool changed = false;

    for (auto [idx, inputType] : llvm::enumerate(functionType.getInputs())) {
      auto memrefType = dyn_cast<BaseMemRefType>(inputType);
      if (!memrefType)
        continue;

      Type newType = convertVPTOBoundaryMemRefType(inputType);
      if (!newType) {
        if (diagOS)
          *diagOS << "VPTO ptr-only boundary failed: unsupported memref "
                  << "argument type in " << func.getName() << ": "
                  << inputType << "\n";
        sawFailure = true;
        continue;
      }

      BlockArgument arg = func.getArgument(idx);
      SmallVector<Operation *> users(arg.getUsers().begin(), arg.getUsers().end());
      arg.setType(newType);
      newInputs[idx] = newType;
      changed = true;

      for (Operation *user : users) {
        if (auto cast = dyn_cast<CastPtrOp>(user)) {
          if (cast.getInput() != arg)
            continue;
          if (cast.getResult().getType() == newType) {
            cast.getResult().replaceAllUsesWith(arg);
            cast.erase();
          }
          continue;
        }

        if (isa<memref::ReinterpretCastOp, memref::SubViewOp,
                memref::MemorySpaceCastOp>(user) &&
            user->use_empty()) {
          user->erase();
          continue;
        }

        if (diagOS) {
          *diagOS << "VPTO ptr-only boundary failed: argument " << idx
                  << " of " << func.getName()
                  << " still feeds a memref-dependent user after ptr rewrite:\n";
          user->print(*diagOS);
          *diagOS << "\n";
        }
        sawFailure = true;
      }
    }

    for (Type resultType : functionType.getResults()) {
      if (!isa<BaseMemRefType>(resultType))
        continue;
      if (diagOS)
        *diagOS << "VPTO ptr-only boundary failed: memref result is unsupported "
                << "for " << func.getName() << ": " << resultType << "\n";
      sawFailure = true;
    }

    if (changed) {
      func.setFunctionType(
          FunctionType::get(module.getContext(), newInputs, functionType.getResults()));
    }
  }

  if (sawFailure)
    return failure();

  if (failed(eraseDeadVPTOMemRefScaffold(module)))
    return failure();
  return verifyNoResidualVPTOMemRefs(module, diagOS);
}

} // namespace pto
} // namespace mlir

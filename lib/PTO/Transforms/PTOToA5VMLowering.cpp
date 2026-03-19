//===- PTOToA5VMLowering.cpp - PTO to A5VM lowering helpers --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/A5VMLowering.h"

#include "PTO/IR/A5VM.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/ErrorHandling.h"

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

int64_t getElementByteSize(Type type);
Value materializeIndexValue(Value maybeValue, int64_t fallback,
                            PatternRewriter &rewriter, Location loc);
Value materializeI64Value(Value maybeValue, int64_t fallback,
                          PatternRewriter &rewriter, Location loc);

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
  if (!ptr || !elemOffsetI64)
    return {};
  Value offsetBytes = elemOffsetI64;
  if (elemBytes != 1) {
    Value elemBytesValue = rewriter.create<arith::ConstantIntOp>(loc, elemBytes, 64);
    offsetBytes = createI64Mul(elemOffsetI64, elemBytesValue, rewriter, loc);
  }
  Value baseInt = rewriter.create<LLVM::PtrToIntOp>(loc, rewriter.getI64Type(), ptr);
  Value adjusted = createI64Add(baseInt, offsetBytes, rewriter, loc);
  return rewriter.create<LLVM::IntToPtrOp>(loc, ptr.getType(), adjusted);
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

StringAttr stringifyPipeAttr(PipeAttr pipe, PatternRewriter &rewriter) {
  return rewriter.getStringAttr(stringifyPIPE(pipe.getPipe()));
}

StringAttr stringifyEventAttr(EventAttr event, PatternRewriter &rewriter) {
  return rewriter.getStringAttr(stringifyEVENT(event.getEvent()));
}

A5VMTileDomain deriveTileDomain(Attribute memorySpace) {
  if (auto addrSpace = dyn_cast_or_null<AddressSpaceAttr>(memorySpace)) {
    switch (addrSpace.getAddressSpace()) {
    case AddressSpace::ACC:
      return A5VMTileDomain::Acc;
    case AddressSpace::MAT:
      return A5VMTileDomain::Mat;
    case AddressSpace::VEC:
    default:
      return A5VMTileDomain::Vec;
    }
  }
  if (auto intAttr = dyn_cast_or_null<IntegerAttr>(memorySpace)) {
    switch (intAttr.getInt()) {
    case static_cast<int64_t>(AddressSpace::ACC):
      return A5VMTileDomain::Acc;
    case static_cast<int64_t>(AddressSpace::MAT):
      return A5VMTileDomain::Mat;
    default:
      return A5VMTileDomain::Vec;
    }
  }
  return A5VMTileDomain::Vec;
}

void getValidShape(TileBufType type, int64_t &rows, int64_t &cols) {
  ArrayRef<int64_t> validShape = type.getValidShape();
  rows = validShape.size() > 0 ? validShape[0] : ShapedType::kDynamic;
  cols = validShape.size() > 1 ? validShape[1] : ShapedType::kDynamic;
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
  return {};
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

void mergeSubviewTrace(A5VMPartitionTrace &trace, ArrayRef<int64_t> offsets,
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

a5vm::VecType getA5VMVecType(MLIRContext *context, Type elementType) {
  unsigned bitWidth = 0;
  if (auto floatType = dyn_cast<FloatType>(elementType))
    bitWidth = floatType.getWidth();
  else if (auto intType = dyn_cast<IntegerType>(elementType))
    bitWidth = intType.getWidth();

  if (bitWidth == 0 || 2048 % bitWidth != 0)
    return {};
  return a5vm::VecType::get(context, 2048 / bitWidth, elementType);
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

Attribute getGmMemorySpace(MLIRContext *context) {
  return AddressSpaceAttr::get(context, AddressSpace::GM);
}

unsigned getLLVMAddressSpace(Attribute memorySpace) {
  if (auto addrSpace = dyn_cast_or_null<AddressSpaceAttr>(memorySpace))
    return static_cast<unsigned>(addrSpace.getAddressSpace());
  if (auto intAttr = dyn_cast_or_null<IntegerAttr>(memorySpace))
    return static_cast<unsigned>(intAttr.getInt());
  return static_cast<unsigned>(AddressSpace::GM);
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

Value materializeBufferPointer(Value value, Type elementType,
                               Attribute memorySpace,
                               PatternRewriter &rewriter, Location loc) {
  if (!value)
    return {};

  if (auto bind = value.getDefiningOp<BindTileOp>())
    return materializeBufferPointer(bind.getSource(), elementType, memorySpace,
                                    rewriter, loc);

  if (auto cast = value.getDefiningOp<PointerCastOp>()) {
    if (cast.getAddrs().empty())
      return {};
    auto ptrType =
        LLVM::LLVMPointerType::get(rewriter.getContext(),
                                   getLLVMAddressSpace(memorySpace));
    return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, cast.getAddrs().front());
  }

  Value memrefValue = materializeTileBufferView(value, rewriter, loc);
  if (!memrefValue || !isa<BaseMemRefType>(memrefValue.getType()))
    return {};

  Value ptrAsIndex =
      rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, memrefValue);
  Value ptrAsI64 =
      rewriter.create<arith::IndexCastUIOp>(loc, rewriter.getI64Type(), ptrAsIndex);
  auto ptrType =
      LLVM::LLVMPointerType::get(rewriter.getContext(),
                                 getLLVMAddressSpace(memorySpace));
  return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrAsI64);
}

A5VMPartitionTrace extractPartitionTrace(Value value) {
  A5VMPartitionTrace trace;
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

A5VMLoadContract extractTLoadContract(TLoadOp op) {
  A5VMLoadContract contract;
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

A5VMUnaryContract extractTAbsContract(TAbsOp op) {
  A5VMUnaryContract contract;
  contract.family = "abs";
  contract.tileDomain = deriveTileDomain(getMemorySpace(op.getSrc()));
  contract.tileLayout = deriveTileLayout(op.getSrc());
  deriveValidShapeValues(op.getSrc(), contract.validRowsValue, contract.validColsValue);
  deriveValidShape(op.getSrc(), contract.validRows, contract.validCols);
  contract.elementType = getElementType(op.getSrc());
  contract.loopScope.kind = A5VMLoopScopeKind::AIVVectorScope;
  contract.loopScope.loweredAttr = kLoweredLoopScopeAttrName;
  contract.loopScope.loopDepth = 0;
  return contract;
}

A5VMBinaryContract buildBinaryContract(StringRef family, Value src0) {
  A5VMBinaryContract contract;
  contract.family = family;
  contract.tileDomain = deriveTileDomain(getMemorySpace(src0));
  contract.tileLayout = deriveTileLayout(src0);
  deriveValidShapeValues(src0, contract.validRowsValue, contract.validColsValue);
  deriveValidShape(src0, contract.validRows, contract.validCols);
  contract.elementType = getElementType(src0);
  contract.loopScope.kind = A5VMLoopScopeKind::AIVVectorScope;
  contract.loopScope.loweredAttr = kLoweredLoopScopeAttrName;
  contract.loopScope.loopDepth = 0;
  return contract;
}

A5VMBinaryContract extractTAddContract(TAddOp op) {
  return buildBinaryContract("add", op.getSrc0());
}

A5VMBinaryContract extractTSubContract(TSubOp op) {
  return buildBinaryContract("sub", op.getSrc0());
}

A5VMBinaryContract extractTMulContract(TMulOp op) {
  return buildBinaryContract("mul", op.getSrc0());
}

A5VMBinaryContract extractTDivContract(TDivOp op) {
  return buildBinaryContract("div", op.getSrc0());
}

A5VMUnaryContract buildUnaryContract(StringRef family, Value src) {
  A5VMUnaryContract contract;
  contract.family = family;
  contract.tileDomain = deriveTileDomain(getMemorySpace(src));
  contract.tileLayout = deriveTileLayout(src);
  deriveValidShapeValues(src, contract.validRowsValue, contract.validColsValue);
  deriveValidShape(src, contract.validRows, contract.validCols);
  contract.elementType = getElementType(src);
  contract.loopScope.kind = A5VMLoopScopeKind::AIVVectorScope;
  contract.loopScope.loweredAttr = kLoweredLoopScopeAttrName;
  contract.loopScope.loopDepth = 0;
  return contract;
}

A5VMUnaryContract extractTExpContract(TExpOp op) {
  return buildUnaryContract("exp", op.getSrc());
}

A5VMUnaryContract extractTLogContract(TLogOp op) {
  return buildUnaryContract("log", op.getSrc());
}

A5VMUnaryContract extractTSqrtContract(TSqrtOp op) {
  return buildUnaryContract("sqrt", op.getSrc());
}

A5VMUnaryContract extractTRecipContract(TRecipOp op) {
  return buildUnaryContract("recip", op.getSrc());
}

A5VMUnaryContract extractTReluContract(TReluOp op) {
  return buildUnaryContract("relu", op.getSrc());
}

A5VMUnaryContract extractTNotContract(TNotOp op) {
  return buildUnaryContract("not", op.getSrc());
}

A5VMStoreContract extractTStoreContract(TStoreOp op) {
  A5VMStoreContract contract;
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

void attachLoadContractAttrs(Operation *op, const A5VMLoadContract &contract) {
  Builder builder(op->getContext());
  SmallVector<int64_t> globalShape;
  SmallVector<int64_t> globalStride;
  normalizeToPTOGlobalShapeAndStride(contract.sourceShape, contract.sourceStrides,
                                     globalShape, globalStride);
  op->setAttr("g_shape", asI64ArrayAttr(builder, globalShape));
  op->setAttr("g_strides", asI64ArrayAttr(builder, globalStride));
}

void attachStoreContractAttrs(Operation *op, const A5VMStoreContract &contract) {
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
  emitError(loc) << "TSTORE ACC lowering TODO for a5vm backend";
  return failure();
}

LogicalResult lowerUnsupportedMatStore(Location loc) {
  emitError(loc) << "TSTORE MAT lowering TODO for a5vm backend";
  return failure();
}

} // namespace

LogicalResult attachLoopScopeMetadata(LoopLikeOpInterface loop,
                                      const A5VMLoopScopeContract &contract,
                                      PatternRewriter &rewriter) {
  if (!loop)
    return failure();
  if (contract.kind == A5VMLoopScopeKind::None)
    return success();
  if (contract.kind != A5VMLoopScopeKind::AIVVectorScope)
    return failure();

  Operation *loopOp = loop.getOperation();
  loopOp->setAttr(contract.loweredAttr, rewriter.getUnitAttr());
  return success();
}

void set_loop2_stride_outtoub(Operation *copyOp, int64_t dstStride,
                              int64_t srcStride, Builder &builder) {
  copyOp->setAttr("a5vm.set_loop2_stride_outtoub",
                  builder.getI64IntegerAttr(
                      packLoopStrideConfig(dstStride, srcStride)));
}

void set_loop1_stride_outtoub(Operation *copyOp, int64_t dstStride,
                              int64_t srcStride, Builder &builder) {
  copyOp->setAttr("a5vm.set_loop1_stride_outtoub",
                  builder.getI64IntegerAttr(
                      packLoopStrideConfig(dstStride, srcStride)));
}

void set_loop_size_outtoub(Operation *copyOp, int64_t loop2, int64_t loop1,
                           Builder &builder) {
  copyOp->setAttr("a5vm.set_loop_size_outtoub",
                  builder.getI64IntegerAttr(packLoopSizeConfig(loop2, loop1)));
}

void set_loop2_stride_ubtoout(Operation *copyOp, int64_t srcStride,
                              int64_t dstStride, Builder &builder) {
  copyOp->setAttr("a5vm.set_loop2_stride_ubtoout",
                  builder.getI64IntegerAttr(
                      packLoopStrideConfig(srcStride, dstStride)));
}

void set_loop1_stride_ubtoout(Operation *copyOp, int64_t srcStride,
                              int64_t dstStride, Builder &builder) {
  copyOp->setAttr("a5vm.set_loop1_stride_ubtoout",
                  builder.getI64IntegerAttr(
                      packLoopStrideConfig(srcStride, dstStride)));
}

void set_loop_size_ubtoout(Operation *copyOp, int64_t loop2, int64_t loop1,
                           Builder &builder) {
  copyOp->setAttr("a5vm.set_loop_size_ubtoout",
                  builder.getI64IntegerAttr(packLoopSizeConfig(loop2, loop1)));
}

LogicalResult programCopyGmToUbLoops(Operation *copyOp,
                                     const A5VMLoadContract &contract,
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
                                     const A5VMStoreContract &contract,
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

LogicalResult buildUnaryVecScope(StringRef family,
                                 const A5VMUnaryContract &contract, Value src,
                                 Value dst, PatternRewriter &rewriter,
                                 Location loc) {
  auto vecType = getA5VMVecType(rewriter.getContext(), contract.elementType);
  if (!vecType)
    return emitError(loc) << "unsupported A5VM unary element type";

  Value srcBuffer = materializeBufferPointer(src, contract.elementType,
                                             getMemorySpace(src), rewriter, loc);
  Value dstBuffer = materializeBufferPointer(dst, contract.elementType,
                                             getMemorySpace(dst), rewriter, loc);
  if (!srcBuffer || !dstBuffer)
    return emitError(loc) << "requires pointer-backed tile buffers for unary lowering";

  int64_t vectorWidth = vecType.getElementCount();
  Value validRowsValue =
      materializeIndexValue(contract.validRowsValue, contract.validRows, rewriter, loc);
  Value validColsValue =
      materializeIndexValue(contract.validColsValue, contract.validCols, rewriter, loc);
  if (!validRowsValue || !validColsValue)
    return emitError(loc) << "unary lowering requires valid rows and cols";

  if (contract.validRows != ShapedType::kDynamic &&
      contract.validCols != ShapedType::kDynamic) {
    int64_t totalElements = contract.validRows * contract.validCols;
    if (totalElements % vectorWidth != 0)
      return emitError(loc)
             << "unary lowering requires total valid elements divisible by vector width";
  }

  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value totalElementsValue =
      rewriter.create<arith::MulIOp>(loc, validRowsValue, validColsValue);
  Value vectorStepValue =
      rewriter.create<arith::ConstantIndexOp>(loc, vectorWidth);

  auto aivScopeLoop = rewriter.create<scf::ForOp>(loc, c0, c1, c1);
  if (failed(attachLoopScopeMetadata(aivScopeLoop, contract.loopScope, rewriter)))
    return emitError(loc) << "failed to attach AIV loop scope metadata";

  OpBuilder::InsertionGuard aivGuard(rewriter);
  rewriter.setInsertionPointToStart(aivScopeLoop.getBody());
  auto chunkLoop =
      rewriter.create<scf::ForOp>(loc, c0, totalElementsValue, vectorStepValue);

  OpBuilder::InsertionGuard chunkGuard(rewriter);
  rewriter.setInsertionPointToStart(chunkLoop.getBody());
  Value offset = chunkLoop.getInductionVar();
  auto vlds = rewriter.create<a5vm::VldsOp>(loc, vecType, srcBuffer, offset);
  Value computed;
  if (family == "abs")
    computed = rewriter.create<a5vm::VabsOp>(loc, vecType, vlds.getResult());
  else if (family == "exp")
    computed = rewriter.create<a5vm::VexpOp>(loc, vecType, vlds.getResult());
  else if (family == "log")
    computed = rewriter.create<a5vm::VlnOp>(loc, vecType, vlds.getResult());
  else if (family == "sqrt")
    computed = rewriter.create<a5vm::VsqrtOp>(loc, vecType, vlds.getResult());
  else if (family == "recip")
    computed = rewriter.create<a5vm::VrecOp>(loc, vecType, vlds.getResult());
  else if (family == "relu")
    computed = rewriter.create<a5vm::VreluOp>(loc, vecType, vlds.getResult());
  else if (family == "not")
    computed = rewriter.create<a5vm::VnotOp>(loc, vecType, vlds.getResult());
  else
    return emitError(loc) << "unsupported A5VM unary family: " << family;
  rewriter.create<a5vm::VstsOp>(loc, computed, dstBuffer, offset, StringAttr());

  return success();
}

LogicalResult buildBinaryVecScope(StringRef family,
                                  const A5VMBinaryContract &contract,
                                  Value src0, Value src1, Value dst,
                                  PatternRewriter &rewriter, Location loc) {
  auto vecType = getA5VMVecType(rewriter.getContext(), contract.elementType);
  if (!vecType)
    return emitError(loc) << "unsupported A5VM binary element type";

  Value src0Buffer = materializeBufferPointer(src0, contract.elementType,
                                              getMemorySpace(src0), rewriter, loc);
  Value src1Buffer = materializeBufferPointer(src1, contract.elementType,
                                              getMemorySpace(src1), rewriter, loc);
  Value dstBuffer = materializeBufferPointer(dst, contract.elementType,
                                             getMemorySpace(dst), rewriter, loc);
  if (!src0Buffer || !src1Buffer || !dstBuffer)
    return emitError(loc) << "requires pointer-backed tile buffers for binary lowering";

  int64_t vectorWidth = vecType.getElementCount();
  Value validRowsValue =
      materializeIndexValue(contract.validRowsValue, contract.validRows, rewriter, loc);
  Value validColsValue =
      materializeIndexValue(contract.validColsValue, contract.validCols, rewriter, loc);
  if (!validRowsValue || !validColsValue)
    return emitError(loc) << "binary lowering requires valid rows and cols";

  if (contract.validRows != ShapedType::kDynamic &&
      contract.validCols != ShapedType::kDynamic) {
    int64_t totalElements = contract.validRows * contract.validCols;
    if (totalElements % vectorWidth != 0)
      return emitError(loc)
             << "binary lowering requires total valid elements divisible by vector width";
  }

  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value totalElementsValue =
      rewriter.create<arith::MulIOp>(loc, validRowsValue, validColsValue);
  Value vectorStepValue =
      rewriter.create<arith::ConstantIndexOp>(loc, vectorWidth);

  auto aivScopeLoop = rewriter.create<scf::ForOp>(loc, c0, c1, c1);
  if (failed(attachLoopScopeMetadata(aivScopeLoop, contract.loopScope, rewriter)))
    return emitError(loc) << "failed to attach AIV loop scope metadata";

  OpBuilder::InsertionGuard aivGuard(rewriter);
  rewriter.setInsertionPointToStart(aivScopeLoop.getBody());
  auto chunkLoop =
      rewriter.create<scf::ForOp>(loc, c0, totalElementsValue, vectorStepValue);

  OpBuilder::InsertionGuard chunkGuard(rewriter);
  rewriter.setInsertionPointToStart(chunkLoop.getBody());
  Value offset = chunkLoop.getInductionVar();
  auto lhs = rewriter.create<a5vm::VldsOp>(loc, vecType, src0Buffer, offset);
  auto rhs = rewriter.create<a5vm::VldsOp>(loc, vecType, src1Buffer, offset);

  Value computed;
  if (family == "add")
    computed = rewriter.create<a5vm::VaddOp>(loc, vecType, lhs.getResult(), rhs.getResult());
  else if (family == "sub")
    computed = rewriter.create<a5vm::VsubOp>(loc, vecType, lhs.getResult(), rhs.getResult());
  else if (family == "mul")
    computed = rewriter.create<a5vm::VmulOp>(loc, vecType, lhs.getResult(), rhs.getResult());
  else if (family == "div")
    computed = rewriter.create<a5vm::VdivOp>(loc, vecType, lhs.getResult(), rhs.getResult());
  else
    return emitError(loc) << "unsupported A5VM binary family: " << family;

  rewriter.create<a5vm::VstsOp>(loc, computed, dstBuffer, offset, StringAttr());
  return success();
}

LogicalResult checkGenericUnaryContract(Operation *op,
                                        const A5VMUnaryContract &contract,
                                        Value dst,
                                        function_ref<bool(Type)> typePredicate,
                                        StringRef supportedTypeText) {
  int64_t dstRows = ShapedType::kDynamic;
  int64_t dstCols = ShapedType::kDynamic;
  deriveValidShape(dst, dstRows, dstCols);
  StringRef dstLayout = deriveTileLayout(dst);
  A5VMTileDomain dstDomain = deriveTileDomain(getMemorySpace(dst));

  bool hasPrecheckFailure = false;
  if (contract.tileDomain != A5VMTileDomain::Vec || dstDomain != A5VMTileDomain::Vec) {
    op->emitOpError() << contract.family << " lowering requires tile domain vec";
    hasPrecheckFailure = true;
  }
  if (contract.tileLayout != "row_major" || dstLayout != "row_major") {
    op->emitOpError() << contract.family << " lowering requires row-major tile layout";
    hasPrecheckFailure = true;
  }
  if (contract.validRows != dstRows || contract.validCols != dstCols) {
    op->emitOpError()
        << contract.family
        << " lowering requires matching source and destination valid region";
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
    Operation *op, const A5VMBinaryContract &contract, Value src1, Value dst,
    function_ref<bool(Type)> typePredicate, StringRef supportedTypeText) {
  StringRef src1Layout = deriveTileLayout(src1);
  StringRef dstLayout = deriveTileLayout(dst);
  A5VMTileDomain src1Domain = deriveTileDomain(getMemorySpace(src1));
  A5VMTileDomain dstDomain = deriveTileDomain(getMemorySpace(dst));

  bool hasPrecheckFailure = false;
  if (contract.tileDomain != A5VMTileDomain::Vec || src1Domain != A5VMTileDomain::Vec ||
      dstDomain != A5VMTileDomain::Vec) {
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

LogicalResult lowerTLOAD(TLoadOp op, PatternRewriter &rewriter) {
  A5VMLoadContract contract = extractTLoadContract(op);
  if (contract.tileDomain != A5VMTileDomain::Vec)
    return op.emitOpError("currently supports only VEC TLOAD lowering");
  if (contract.tileLayout != "row_major" || contract.sourceLayout != "nd")
    return op.emitOpError("currently supports only row_major ND vec TLOAD lowering");

  ResolvedTensorView sourceView;
  if (!resolveTensorView(op.getSrc(), sourceView, rewriter, op.getLoc()))
    return op.emitOpError("requires a recoverable source tensor view for A5VM lowering");

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
  if (tileCols == ShapedType::kDynamic || elemBytes <= 0)
    return op.emitOpError("requires static tile columns for A5-compatible transfer arguments");
  VecNdTransferPlan plan;
  if (failed(buildVecNdLoadPlan(sourceView.shape, sourceView.strides, tileCols,
                                contract.validColsValue, contract.validCols,
                                contract.elementType, rewriter, op.getLoc(), plan)))
    return op.emitOpError("requires PTO-compatible vec ND2ND copy_gm_to_ubuf arguments");
  Value leftPaddingValue = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0, 64);
  Value rightPaddingValue = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0, 64);
  Value cacheCtlValue = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0, 64);
  if (!validRowsValue || !validColsValue)
    return op.emitOpError("requires valid rows and cols for A5-compatible transfer arguments");
  Value sourceOffset =
      materializeI64Ofr(sourceView.offsetElems, rewriter, op.getLoc());
  if (!sourceOffset)
    return op.emitOpError("requires a materializable source offset for A5VM lowering");
  Value sourceBase = adjustPointerByElemOffset(sourceBuffer, sourceOffset, elemBytes,
                                              rewriter, op.getLoc());
  if (!sourceBase)
    return op.emitOpError("failed to materialize source base pointer");

  rewriter.create<a5vm::SetLoop2StrideOutToUbOp>(
      op.getLoc(), plan.loop2FirstStrideBytes, plan.loop2SecondStrideBytes);
  rewriter.create<a5vm::SetLoop1StrideOutToUbOp>(
      op.getLoc(), plan.loop1FirstStrideBytes, plan.loop1SecondStrideBytes);
  rewriter.create<a5vm::SetLoopSizeOutToUbOp>(op.getLoc(), plan.loop2Size,
                                              plan.loop1Size);

  auto emitCopy = [&](Value srcPtr, Value dstPtr) {
    rewriter.create<a5vm::CopyGmToUbufOp>(
        op.getLoc(), srcPtr, dstPtr, validRowsValue, validColsValue, sidValue,
        plan.nBurst, plan.lenBurst, leftPaddingValue, rightPaddingValue,
        cacheCtlValue, plan.firstStrideBytes, plan.secondStrideBytes,
        rewriter.getStringAttr(contract.sourceLayout), rewriter.getBoolAttr(ubPad),
        rewriter.getBoolAttr(ubPad));
  };

  if (std::optional<int64_t> outerConst = getConstInt(plan.outerCount); outerConst && *outerConst == 1) {
    emitCopy(sourceBase, destinationBuffer);
    return success();
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
  emitCopy(iterSrc, iterDst);
  return success();
}

LogicalResult lowerTABS(TAbsOp op, PatternRewriter &rewriter) {
  A5VMUnaryContract contract = extractTAbsContract(op);
  if (failed(checkGenericUnaryContract(
          op, contract, op.getDst(),
          [](Type type) { return type.isF16() || type.isF32(); }, "f16 and f32 element types")))
    return failure();

  return buildUnaryVecScope("abs", contract, op.getSrc(), op.getDst(), rewriter,
                            op.getLoc());
}

LogicalResult lowerTADD(TAddOp op, PatternRewriter &rewriter) {
  A5VMBinaryContract contract = extractTAddContract(op);
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
  return buildBinaryVecScope("add", contract, op.getSrc0(), op.getSrc1(),
                             op.getDst(), rewriter, op.getLoc());
}

LogicalResult lowerTSUB(TSubOp op, PatternRewriter &rewriter) {
  A5VMBinaryContract contract = extractTSubContract(op);
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
  return buildBinaryVecScope("sub", contract, op.getSrc0(), op.getSrc1(),
                             op.getDst(), rewriter, op.getLoc());
}

LogicalResult lowerTMUL(TMulOp op, PatternRewriter &rewriter) {
  A5VMBinaryContract contract = extractTMulContract(op);
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
  return buildBinaryVecScope("mul", contract, op.getSrc0(), op.getSrc1(),
                             op.getDst(), rewriter, op.getLoc());
}

LogicalResult lowerTDIV(TDivOp op, PatternRewriter &rewriter) {
  A5VMBinaryContract contract = extractTDivContract(op);
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
  return buildBinaryVecScope("div", contract, op.getSrc0(), op.getSrc1(),
                             op.getDst(), rewriter, op.getLoc());
}

LogicalResult lowerTEXP(TExpOp op, PatternRewriter &rewriter) {
  A5VMUnaryContract contract = extractTExpContract(op);
  if (failed(checkGenericUnaryContract(
          op, contract, op.getDst(),
          [](Type type) { return type.isF16() || type.isF32(); }, "f16 and f32 element types")))
    return failure();
  return buildUnaryVecScope("exp", contract, op.getSrc(), op.getDst(), rewriter,
                            op.getLoc());
}

LogicalResult lowerTLOG(TLogOp op, PatternRewriter &rewriter) {
  A5VMUnaryContract contract = extractTLogContract(op);
  if (failed(checkGenericUnaryContract(
          op, contract, op.getDst(),
          [](Type type) { return type.isF16() || type.isF32(); }, "f16 and f32 element types")))
    return failure();
  return buildUnaryVecScope("log", contract, op.getSrc(), op.getDst(), rewriter,
                            op.getLoc());
}

LogicalResult lowerTSQRT(TSqrtOp op, PatternRewriter &rewriter) {
  A5VMUnaryContract contract = extractTSqrtContract(op);
  if (failed(checkGenericUnaryContract(
          op, contract, op.getDst(),
          [](Type type) { return type.isF16() || type.isF32(); }, "f16 and f32 element types")))
    return failure();
  return buildUnaryVecScope("sqrt", contract, op.getSrc(), op.getDst(), rewriter,
                            op.getLoc());
}

LogicalResult lowerTRECIP(TRecipOp op, PatternRewriter &rewriter) {
  A5VMUnaryContract contract = extractTRecipContract(op);
  if (failed(checkGenericUnaryContract(
          op, contract, op.getDst(),
          [](Type type) { return type.isF16() || type.isF32(); }, "f16 and f32 element types")))
    return failure();
  return buildUnaryVecScope("recip", contract, op.getSrc(), op.getDst(), rewriter,
                            op.getLoc());
}

LogicalResult lowerTRELU(TReluOp op, PatternRewriter &rewriter) {
  A5VMUnaryContract contract = extractTReluContract(op);
  if (failed(checkGenericUnaryContract(
          op, contract, op.getDst(),
          [](Type type) {
            return type.isF16() || type.isF32() ||
                   (isa<IntegerType>(type) && cast<IntegerType>(type).getWidth() == 32);
          },
          "f16, f32, and i32 element types")))
    return failure();
  return buildUnaryVecScope("relu", contract, op.getSrc(), op.getDst(), rewriter,
                            op.getLoc());
}

LogicalResult lowerTNOT(TNotOp op, PatternRewriter &rewriter) {
  A5VMUnaryContract contract = extractTNotContract(op);
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
  return buildUnaryVecScope("not", contract, op.getSrc(), op.getDst(), rewriter,
                            op.getLoc());
}

LogicalResult lowerTSTORE(TStoreOp op, PatternRewriter &rewriter) {
  A5VMStoreContract contract = extractTStoreContract(op);

  switch (contract.srcDomain) {
  case A5VMTileDomain::Acc:
    return lowerUnsupportedAccStore(op.getLoc());
  case A5VMTileDomain::Mat:
    return lowerUnsupportedMatStore(op.getLoc());
  case A5VMTileDomain::Vec:
    break;
  }
  if (contract.destinationLayout != "nd")
    return op.emitOpError("currently supports only ND destination TSTORE lowering");

  ResolvedTensorView destinationView;
  if (!resolveTensorView(op.getDst(), destinationView, rewriter, op.getLoc()))
    return op.emitOpError("requires a recoverable destination tensor view for A5VM lowering");

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
  (void)tileRows;
  Value validRowsValue =
      materializeI64Value(contract.validRowsValue, contract.validRows, rewriter,
                          op.getLoc());
  Value validColsValue =
      materializeI64Value(contract.validColsValue, contract.validCols, rewriter,
                          op.getLoc());
  Value sidValue = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0, 64);
  int64_t elemBytes = getElementByteSize(contract.elementType);
  if (tileCols == ShapedType::kDynamic || elemBytes <= 0)
    return op.emitOpError("requires static tile columns for A5-compatible transfer arguments");
  VecNdTransferPlan plan;
  if (failed(buildVecNdStorePlan(destinationView.shape, destinationView.strides, tileCols,
                                 contract.validColsValue, contract.validCols,
                                 contract.elementType, rewriter, op.getLoc(), plan)))
    return op.emitOpError("requires PTO-compatible vec ND2ND copy_ubuf_to_gm arguments");
  Value reservedValue = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0, 64);
  if (!validRowsValue || !validColsValue)
    return op.emitOpError("requires valid rows and cols for A5-compatible transfer arguments");
  Value destinationOffset =
      materializeI64Ofr(destinationView.offsetElems, rewriter, op.getLoc());
  if (!destinationOffset)
    return op.emitOpError("requires a materializable destination offset for A5VM lowering");
  Value destinationBase =
      adjustPointerByElemOffset(destinationBuffer, destinationOffset, elemBytes, rewriter,
                                op.getLoc());
  if (!destinationBase)
    return op.emitOpError("failed to materialize destination base pointer");

  rewriter.create<a5vm::SetLoopSizeUbToOutOp>(op.getLoc(), plan.loop2Size,
                                              plan.loop1Size);
  rewriter.create<a5vm::SetLoop1StrideUbToOutOp>(
      op.getLoc(), plan.loop1FirstStrideBytes, plan.loop1SecondStrideBytes);
  rewriter.create<a5vm::SetLoop2StrideUbToOutOp>(
      op.getLoc(), plan.loop2FirstStrideBytes, plan.loop2SecondStrideBytes);

  auto emitCopy = [&](Value srcPtr, Value dstPtr) {
    rewriter.create<a5vm::CopyUbufToGmOp>(
        op.getLoc(), srcPtr, dstPtr, validRowsValue, validColsValue, sidValue,
        plan.nBurst, plan.lenBurst, reservedValue, plan.firstStrideBytes,
        plan.secondStrideBytes, rewriter.getStringAttr(contract.destinationLayout));
  };

  if (std::optional<int64_t> outerConst = getConstInt(plan.outerCount); outerConst && *outerConst == 1) {
    emitCopy(sourceBuffer, destinationBase);
    return success();
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
  emitCopy(iterSrc, iterDst);
  return success();
}

LogicalResult lowerSetFlag(SetFlagOp op, PatternRewriter &rewriter) {
  rewriter.create<a5vm::SetFlagOp>(op.getLoc(),
                                   stringifyPipeAttr(op.getSrcPipe(), rewriter),
                                   stringifyPipeAttr(op.getDstPipe(), rewriter),
                                   stringifyEventAttr(op.getEventId(), rewriter));
  return success();
}

LogicalResult lowerWaitFlag(WaitFlagOp op, PatternRewriter &rewriter) {
  rewriter.create<a5vm::WaitFlagOp>(op.getLoc(),
                                    stringifyPipeAttr(op.getSrcPipe(), rewriter),
                                    stringifyPipeAttr(op.getDstPipe(), rewriter),
                                    stringifyEventAttr(op.getEventId(), rewriter));
  return success();
}

LogicalResult lowerBarrier(BarrierOp op, PatternRewriter &rewriter) {
  rewriter.create<a5vm::PipeBarrierOp>(op.getLoc(),
                                       stringifyPipeAttr(op.getPipe(), rewriter));
  return success();
}

} // namespace pto
} // namespace mlir

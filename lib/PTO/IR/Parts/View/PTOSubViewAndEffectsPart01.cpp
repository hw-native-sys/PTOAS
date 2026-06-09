// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Split from PTOSubViewAndEffects.cpp; kept as a fragment included by PTOSubViewAndEffects.cpp.
// Intentionally has no local includes.

using namespace mlir;
using namespace mlir::pto;

void mlir::pto::SubViewOp::print(OpAsmPrinter &p) {
  p << " " << getSource() << "[";
  p.printOperands(getOffsets());
  p << "] sizes " << getSizes();
  if (getValidRow()) {
    p << " valid [" << getValidRow() << ", " << getValidCol() << "]";
  }
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes", "sizes"});
  p << " : " << getSource().getType() << " -> " << getResult().getType();
}

static std::optional<ArrayAttr> getSubViewSizeAttr(DictionaryAttr attributes,
                                                   OpaqueProperties properties) {
  if (properties) {
    const auto *prop = properties.as<SubViewOp::Properties *>();
    if (prop && prop->sizes)
      return prop->sizes;
  }
  if (attributes)
    return attributes.getAs<ArrayAttr>("sizes");
  return std::nullopt;
}

static SmallVector<int64_t> collectSubviewShape(ArrayAttr sizeAttr) {
  SmallVector<int64_t> subviewShape;
  for (auto attr : sizeAttr)
    subviewShape.push_back(llvm::cast<IntegerAttr>(attr).getInt());
  return subviewShape;
}

struct SubViewExplicitValidOperands {
  Value row;
  Value col;
};

static void decodeSubviewExplicitValidOperandsFromSegments(
    SubViewExplicitValidOperands &explicitValids, ValueRange operands,
    DictionaryAttr attributes) {
  if (!attributes)
    return;
  auto segAttr = attributes.getAs<DenseI32ArrayAttr>("operandSegmentSizes");
  if (!segAttr)
    return;
  ArrayRef<int32_t> segs = segAttr.asArrayRef();
  if (segs.size() != kNumber4)
    return;
  int32_t srcSeg = segs[0];
  int32_t offSeg = segs[1];
  int32_t vRowSeg = segs[2];
  int32_t vColSeg = segs[3];
  if (srcSeg != 1 || offSeg < 0 || (vRowSeg != 0 && vRowSeg != 1) ||
      (vColSeg != 0 && vColSeg != 1))
    return;
  size_t idx = static_cast<size_t>(srcSeg + offSeg);
  if (vRowSeg == 1 && idx < operands.size())
    explicitValids.row = operands[idx++];
  if (vColSeg == 1 && idx < operands.size())
    explicitValids.col = operands[idx];
}

static SubViewExplicitValidOperands decodeSubviewExplicitValidOperands(
    ValueRange operands, DictionaryAttr attributes, int64_t rank) {
  SubViewExplicitValidOperands explicitValids;
  decodeSubviewExplicitValidOperandsFromSegments(explicitValids, operands,
                                                 attributes);
  if (!explicitValids.row && !explicitValids.col && rank == kPTORowColRank) {
    size_t expectedWithoutValid = static_cast<size_t>(1 + rank);
    if (operands.size() >= expectedWithoutValid + kNumber2) {
      explicitValids.row = operands[expectedWithoutValid];
      explicitValids.col = operands[expectedWithoutValid + 1];
    }
  }
  return explicitValids;
}

static SmallVector<int64_t> inferSubviewValidShape(ArrayRef<int64_t> subviewShape,
                                                   Value explicitVRow,
                                                   Value explicitVCol) {
  constexpr int64_t kDynamicValidDim = -1;
  SmallVector<int64_t> validShape;
  for (size_t i = 0, e = subviewShape.size(); i < e; ++i) {
    int64_t vdim = subviewShape[i];
    Value explicitV = (i == 0) ? explicitVRow : (i == 1 ? explicitVCol : Value());
    if (explicitV) {
      auto cst = getConstIndexValue(explicitV);
      vdim = cst ? std::min<int64_t>(*cst, subviewShape[i]) : kDynamicValidDim;
    }
    validShape.push_back(vdim);
  }
  return validShape;
}

LogicalResult SubViewOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  (void)location;
  (void)regions;
  if (operands.empty())
    return failure();
  auto sourceType = llvm::dyn_cast<TileBufType>(operands[0].getType());
  if (!sourceType)
    return failure();

  auto sizeAttr = getSubViewSizeAttr(attributes, properties);
  if (!sizeAttr)
    return failure();
  auto subviewShape = collectSubviewShape(*sizeAttr);
  if (subviewShape.size() != sourceType.getShape().size())
    return failure();

  auto explicitValids = decodeSubviewExplicitValidOperands(
      operands, attributes, static_cast<int64_t>(subviewShape.size()));
  auto validShape = inferSubviewValidShape(subviewShape, explicitValids.row,
                                           explicitValids.col);

  auto cfg = sourceType.getConfigAttr();
  if (!cfg)
    cfg = TileBufConfigAttr::getDefault(context);

  auto canonicalValidShape = canonicalizeTileBufValidShape(validShape);
  inferredReturnTypes.push_back(TileBufType::get(
      context, subviewShape, sourceType.getElementType(),
      sourceType.getMemorySpace(), canonicalValidShape, cfg));
  return success();
}

// =============================================================================
// SubViewOp verifier
// =============================================================================
static bool getConstIndex(Value v, int64_t &out) {
  if (auto cOp = v.getDefiningOp<arith::ConstantIndexOp>()) {
    out = cOp.value();
    return true;
  }
  if (auto cInt = v.getDefiningOp<arith::ConstantIntOp>()) {
    out = cInt.value();
    return true;
  }
  if (auto cOp = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(cOp.getValue())) {
      out = ia.getInt();
      return true;
    }
  }
  if (auto castOp = v.getDefiningOp<arith::IndexCastOp>())
    return getConstIndex(castOp.getIn(), out);
  if (auto extOp = v.getDefiningOp<arith::ExtSIOp>())
    return getConstIndex(extOp.getIn(), out);
  if (auto extOp = v.getDefiningOp<arith::ExtUIOp>())
    return getConstIndex(extOp.getIn(), out);
  if (auto truncOp = v.getDefiningOp<arith::TruncIOp>())
    return getConstIndex(truncOp.getIn(), out);
  return false;
}

static bool readLayoutI32(Attribute attr, int32_t &out) {
  if (auto layoutAttr = dyn_cast<BLayoutAttr>(attr)) {
    out = static_cast<int32_t>(layoutAttr.getValue());
    return true;
  }
  if (auto layoutAttr = dyn_cast<SLayoutAttr>(attr)) {
    out = static_cast<int32_t>(layoutAttr.getValue());
    return true;
  }
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    out = static_cast<int32_t>(intAttr.getInt());
    return true;
  }
  return false;
}

static LogicalResult computeBoxedInnerShape(Type elemTy, int32_t fractalSize,
                                            int32_t slayout, int64_t &innerRows,
                                            int64_t &innerCols) {
  int64_t elemBytes = static_cast<int64_t>(getElemByteSize(elemTy));
  if (elemBytes <= 0)
    return failure();
  if (fractalSize == kFractalSize1024) {
    innerRows = kFractalSize16;
    innerCols = kFractalSize16;
    return success();
  }
  if (fractalSize == kFractalSize32) {
    innerRows = kFractalSize16;
    innerCols = kFractalSize32 / kFractalSize16;
    return success();
  }
  if (fractalSize == kFractalSize512 &&
      slayout == static_cast<int32_t>(SLayout::RowMajor)) {
    innerRows = kFractalSize16;
    innerCols = kFractalSize32 / elemBytes;
    return success();
  }
  if (fractalSize == kFractalSize512 &&
      slayout == static_cast<int32_t>(SLayout::ColMajor)) {
    innerRows = kFractalSize32 / elemBytes;
    innerCols = kFractalSize16;
    return success();
  }
  return failure();
}

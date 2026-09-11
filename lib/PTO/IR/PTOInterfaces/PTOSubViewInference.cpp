// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

LogicalResult SubViewOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  if (operands.empty())
    return failure();
  auto sourceType = dyn_cast<TileBufType>(operands[0].getType());
  ArrayAttr sizeAttr = getSubViewSizeAttr(attributes, properties);
  if (!sourceType || !sizeAttr)
    return failure();
  SmallVector<int64_t> subviewShape;
  for (Attribute attr : sizeAttr)
    subviewShape.push_back(cast<IntegerAttr>(attr).getInt());
  if (subviewShape.size() != sourceType.getShape().size())
    return failure();
  auto [row, col] = getSubViewExplicitValids(
      operands, attributes, static_cast<int64_t>(subviewShape.size()));
  SmallVector<int64_t> validShape =
      getSubViewValidShape(subviewShape, row, col);

  auto cfg = sourceType.getConfigAttr();
  if (!cfg)
    cfg = TileBufConfigAttr::getDefault(context);
  auto canonicalValidShape = canonicalizeTileBufValidShape(validShape);
  auto resultType = TileBufType::get(
      context, subviewShape, sourceType.getElementType(),
      sourceType.getMemorySpace(), canonicalValidShape, cfg);

  inferredReturnTypes.push_back(resultType);
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
  if (auto castOp = v.getDefiningOp<arith::IndexCastOp>()) {
    return getConstIndex(castOp.getIn(), out);
  }
  if (auto extOp = v.getDefiningOp<arith::ExtSIOp>()) {
    return getConstIndex(extOp.getIn(), out);
  }
  if (auto extOp = v.getDefiningOp<arith::ExtUIOp>()) {
    return getConstIndex(extOp.getIn(), out);
  }
  if (auto truncOp = v.getDefiningOp<arith::TruncIOp>()) {
    return getConstIndex(truncOp.getIn(), out);
  }
  return false;
}

static LogicalResult computeInnerShape(TileBufConfigAttr cfg, Type elemTy,
                                       int64_t &innerRows, int64_t &innerCols,
                                       bool &boxed, int32_t &bl, int32_t &sl) {
  bl = 0;
  sl = 0;
  int32_t fr = 512;
  (void)readBLayoutValue(cfg.getBLayout(), bl);
  (void)readSLayoutValue(cfg.getSLayout(), sl);
  if (auto attr = dyn_cast<IntegerAttr>(cfg.getSFractalSize())) {
    fr = static_cast<int32_t>(attr.getInt());
  }

  boxed = (sl != 0);
  if (!boxed) {
    innerRows = 1;
    innerCols = 1;
    return success();
  }

  int64_t elemBytes = static_cast<int64_t>(getElemByteSize(elemTy));
  if (elemBytes <= 0) {
    return failure();
  }

  if (fr == mlir::pto::kValue1024) {
      innerRows = mlir::pto::kValue16;
      innerCols = mlir::pto::kValue16;
      return success();
  }
  if (fr == mlir::pto::kValue32) {
      innerRows = mlir::pto::kValue16;
      innerCols = mlir::pto::kValue2;
      return success();
  }
  if (fr == mlir::pto::kValue512) {
      if (sl == 1) {
          innerRows = mlir::pto::kValue16;
          innerCols = mlir::pto::kValue32 / elemBytes;
          return success();
      }
      if (sl == mlir::pto::kValue2) {
          innerRows = mlir::pto::kValue32 / elemBytes;
          innerCols = mlir::pto::kValue16;
          return success();
      }
  }
  return failure();
}

struct SubViewInfo {
  int64_t sizeR = 0, sizeC = 0;
  int64_t offR = 0, offC = 0;
  bool offRConst = false, offCConst = false;
};

static LogicalResult verifySubViewSizesAndOffsets(SubViewOp op,
                                                  SubViewInfo &info) {
  auto sizesAttr = op.getSizes();
  if (!sizesAttr || sizesAttr.size() != mlir::pto::kValue2) {
      return op.emitOpError("subview expects 2D sizes");
  }
  info.sizeR = cast<IntegerAttr>(sizesAttr[0]).getInt();
  info.sizeC = cast<IntegerAttr>(sizesAttr[1]).getInt();
  if (info.sizeR <= 0 || info.sizeC <= 0) {
    return op.emitOpError("subview sizes must be positive");
  }
  if (op.getOffsets().size() != mlir::pto::kValue2) {
      return op.emitOpError("subview expects 2D offsets");
  }

  info.offRConst = getConstIndex(op.getOffsets()[0], info.offR);
  info.offCConst = getConstIndex(op.getOffsets()[1], info.offC);
  if (info.offRConst && info.offR < 0) {
    return op.emitOpError("subview offsets must be non-negative");
  }
  if (info.offCConst && info.offC < 0) {
    return op.emitOpError("subview offsets must be non-negative");
  }
  return success();
}

static LogicalResult verifySubViewValidBounds(SubViewOp op, int64_t sizeR,
                                              int64_t sizeC) {
  bool hasValidRow = static_cast<bool>(op.getValidRow());
  bool hasValidCol = static_cast<bool>(op.getValidCol());
  if (hasValidRow != hasValidCol) {
    return op.emitOpError(
        "subview expects valid_row and valid_col to be both present or both absent");
  }

  if (hasValidRow) {
    int64_t vRow = 0, vCol = 0;
    if (getConstIndex(op.getValidRow(), vRow)) {
      if (vRow < 0) {
        return op.emitOpError("valid_row must be non-negative when constant");
      }
      if (vRow > sizeR) {
        return op.emitOpError("valid_row must be <= subview row size");
      }
    }
    if (getConstIndex(op.getValidCol(), vCol)) {
      if (vCol < 0) {
        return op.emitOpError("valid_col must be non-negative when constant");
      }
      if (vCol > sizeC) {
        return op.emitOpError("valid_col must be <= subview col size");
      }
    }
  }
  return success();
}
static LogicalResult verifySubViewShapeAndConfig(SubViewOp op, TileBufType srcTy,
                                                 TileBufType dstTy, int64_t sizeR,
                                                 int64_t sizeC) {
  auto dstShape = dstTy.getShape();
  if (dstShape.size() != mlir::pto::kValue2) {
      return op.emitOpError("expects result to be rank-2");
  }
  auto srcShape = srcTy.getShape();
  if (srcShape.size() != mlir::pto::kValue2) {
      return op.emitOpError("expects source to be rank-2");
  }
  if (dstShape[0] != sizeR || dstShape[1] != sizeC) {
    return op.emitOpError("expects result shape to match subview sizes");
  }

  if (dstTy.getElementType() != srcTy.getElementType()) {
    return op.emitOpError("expects result element type to match source");
  }
  if (dstTy.getMemorySpace() != srcTy.getMemorySpace()) {
    return op.emitOpError("expects result address space to match source");
  }
  auto srcCfg = srcTy.getConfigAttr();
  if (!srcCfg) {
    srcCfg = TileBufConfigAttr::getDefault(op.getContext());
  }
  auto dstCfg = dstTy.getConfigAttr();
  if (!dstCfg) {
    dstCfg = TileBufConfigAttr::getDefault(op.getContext());
  }
  if (dstCfg != srcCfg) {
    return op.emitOpError("expects result tile config to match source");
  }
  return success();
}

static LogicalResult verifySubViewValidShape(SubViewOp op, TileBufType dstTy,
                                             int64_t sizeR, int64_t sizeC) {
  // Design choice: when valid[...] is omitted, infer result valid_shape from
  // subview sizes directly. We intentionally do not constrain it by source
  // valid_shape to allow user-controlled subview semantics.

  auto expectedValidDim = [&](Value explicitValid, int64_t defaultSize) {
    if (!explicitValid) {
      return defaultSize;
    }
    int64_t c = 0;
    if (getConstIndex(explicitValid, c)) {
      return std::min<int64_t>(c, defaultSize);
    }
    return ShapedType::kDynamic;
  };
  int64_t expectedVRow = expectedValidDim(op.getValidRow(), sizeR);
  int64_t expectedVCol = expectedValidDim(op.getValidCol(), sizeC);
  auto dstValid = dstTy.getValidShape();
  if (dstValid.size() != mlir::pto::kValue2) {
      return op.emitOpError("expects result to have rank-2 valid_shape");
  }
  // With the valid operand omitted, the result type is authoritative for the
  // valid extent: accept any static value in [0, size] (this subsumes both the
  // full-size default and the v=0 no-op-replay empty marker). A dynamic result valid still
  // requires an explicit operand to supply the runtime extent, so it stays
  // rejected on this path.
  bool rowInferred = !op.getValidRow() && dstValid[0] != ShapedType::kDynamic &&
                     dstValid[0] >= 0 && dstValid[0] <= sizeR;
  bool colInferred = !op.getValidCol() && dstValid[1] != ShapedType::kDynamic &&
                     dstValid[1] >= 0 && dstValid[1] <= sizeC;
  if (dstValid[0] != expectedVRow && !rowInferred) {
    return op.emitOpError("expects result valid_shape[0] to match inferred/explicit valid_row");
  }
  if (dstValid[1] != expectedVCol && !colInferred) {
    return op.emitOpError("expects result valid_shape[1] to match inferred/explicit valid_col");
  }
  return success();
}

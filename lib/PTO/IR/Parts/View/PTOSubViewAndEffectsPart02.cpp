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

static LogicalResult computeInnerShape(TileBufConfigAttr cfg, Type elemTy,
                                       int64_t &innerRows, int64_t &innerCols,
                                       bool &boxed, int32_t &bl, int32_t &sl) {
  bl = 0;
  sl = 0;
  int32_t fractalSize = 512;
  (void)readLayoutI32(cfg.getBLayout(), bl);
  (void)readLayoutI32(cfg.getSLayout(), sl);
  if (auto attr = dyn_cast<IntegerAttr>(cfg.getSFractalSize()))
    fractalSize = static_cast<int32_t>(attr.getInt());

  boxed = sl != 0;
  if (!boxed) {
    innerRows = 1;
    innerCols = 1;
    return success();
  }
  return computeBoxedInnerShape(elemTy, fractalSize, sl, innerRows, innerCols);
}

struct SubViewVerifyInfo {
  TileBufType srcTy;
  TileBufType dstTy;
  int64_t sizeR;
  int64_t sizeC;
  int64_t offR = 0;
  int64_t offC = 0;
  bool offRConst = false;
  bool offCConst = false;
};

static FailureOr<SubViewVerifyInfo> verifySubviewOperandsAndSizes(SubViewOp op) {
  auto srcTy = llvm::dyn_cast<TileBufType>(op.getSource().getType());
  auto dstTy = llvm::dyn_cast<TileBufType>(op.getResult().getType());
  if (!srcTy || !dstTy)
    return op.emitOpError("expects tile_buf src and tile_buf result"), failure();
  if (srcTy.getRank() != kPTORowColRank || dstTy.getRank() != kPTORowColRank)
    return op.emitOpError("expects rank-2 tilebuf for src/dst"), failure();

  auto sizesAttr = op.getSizes();
  if (!sizesAttr || sizesAttr.size() != kPTORowColRank)
    return op.emitOpError("subview expects 2D sizes"), failure();
  int64_t sizeR = cast<IntegerAttr>(sizesAttr[0]).getInt();
  int64_t sizeC = cast<IntegerAttr>(sizesAttr[1]).getInt();
  if (sizeR <= 0 || sizeC <= 0)
    return op.emitOpError("subview sizes must be positive"), failure();
  if (op.getOffsets().size() != kPTORowColRank)
    return op.emitOpError("subview expects 2D offsets"), failure();

  SubViewVerifyInfo info{srcTy, dstTy, sizeR, sizeC};
  info.offRConst = getConstIndex(op.getOffsets()[0], info.offR);
  info.offCConst = getConstIndex(op.getOffsets()[1], info.offC);
  if ((info.offRConst && info.offR < 0) || (info.offCConst && info.offC < 0)) {
    return op.emitOpError("subview offsets must be non-negative"), failure();
  }
  return info;
}

static LogicalResult verifySubviewExplicitValids(SubViewOp op, int64_t sizeR,
                                                 int64_t sizeC) {
  bool hasValidRow = static_cast<bool>(op.getValidRow());
  bool hasValidCol = static_cast<bool>(op.getValidCol());
  if (hasValidRow != hasValidCol) {
    return op.emitOpError(
        "subview expects valid_row and valid_col to be both present or both absent");
  }
  if (!hasValidRow)
    return success();

  int64_t vRow = 0;
  int64_t vCol = 0;
  if (getConstIndex(op.getValidRow(), vRow)) {
    if (vRow <= 0)
      return op.emitOpError("valid_row must be positive when constant");
    if (vRow > sizeR)
      return op.emitOpError("valid_row must be <= subview row size");
  }
  if (getConstIndex(op.getValidCol(), vCol)) {
    if (vCol <= 0)
      return op.emitOpError("valid_col must be positive when constant");
    if (vCol > sizeC)
      return op.emitOpError("valid_col must be <= subview col size");
  }
  return success();
}

static LogicalResult verifySubviewResultType(SubViewOp op, const SubViewVerifyInfo &info) {
  auto dstShape = info.dstTy.getShape();
  auto srcShape = info.srcTy.getShape();
  if (dstShape.size() != kPTORowColRank)
    return op.emitOpError("expects result to be rank-2");
  if (srcShape.size() != kPTORowColRank)
    return op.emitOpError("expects source to be rank-2");
  if (dstShape[0] != info.sizeR || dstShape[1] != info.sizeC)
    return op.emitOpError("expects result shape to match subview sizes");
  if (info.dstTy.getElementType() != info.srcTy.getElementType())
    return op.emitOpError("expects result element type to match source");
  if (info.dstTy.getMemorySpace() != info.srcTy.getMemorySpace())
    return op.emitOpError("expects result address space to match source");

  auto srcCfg = info.srcTy.getConfigAttr();
  if (!srcCfg)
    srcCfg = TileBufConfigAttr::getDefault(op.getContext());
  auto dstCfg = info.dstTy.getConfigAttr();
  if (!dstCfg)
    dstCfg = TileBufConfigAttr::getDefault(op.getContext());
  if (dstCfg != srcCfg)
    return op.emitOpError("expects result tile config to match source");
  return success();
}

static int64_t getSubviewExpectedValidDim(Value explicitValid, int64_t defaultSize) {
  if (!explicitValid)
    return defaultSize;
  int64_t constantValue = 0;
  if (getConstIndex(explicitValid, constantValue))
    return std::min<int64_t>(constantValue, defaultSize);
  return ShapedType::kDynamic;
}

static LogicalResult verifySubviewResultValidShape(SubViewOp op,
                                                   const SubViewVerifyInfo &info) {
  auto dstValid = info.dstTy.getValidShape();
  if (dstValid.size() != kPTORowColRank)
    return op.emitOpError("expects result to have rank-2 valid_shape");
  int64_t expectedVRow = getSubviewExpectedValidDim(op.getValidRow(), info.sizeR);
  int64_t expectedVCol = getSubviewExpectedValidDim(op.getValidCol(), info.sizeC);
  if (dstValid[0] != expectedVRow)
    return op.emitOpError(
        "expects result valid_shape[0] to match inferred/explicit valid_row");
  if (dstValid[1] != expectedVCol)
    return op.emitOpError(
        "expects result valid_shape[1] to match inferred/explicit valid_col");
  return success();
}

static bool hasStaticRank2Shape(ArrayRef<int64_t> shape) {
  return shape.size() == kPTORowColRank && shape[0] != ShapedType::kDynamic &&
         shape[1] != ShapedType::kDynamic;
}

static LogicalResult verifyBoxedSubviewMajorConstraint(
    SubViewOp op, const SubViewVerifyInfo &info, ArrayRef<int64_t> srcShape,
    int32_t bl) {
  if (bl == 0) {
    if (info.sizeC != srcShape[1])
      return op.emitOpError("boxed RowMajor subview must keep full cols");
    if (!info.offCConst || info.offC != 0)
      return op.emitOpError(
          "boxed RowMajor subview requires static col offset = 0");
    return success();
  }
  if (bl == 1) {
    if (info.sizeR != srcShape[0])
      return op.emitOpError("boxed ColMajor subview must keep full rows");
    if (!info.offRConst || info.offR != 0)
      return op.emitOpError(
          "boxed ColMajor subview requires static row offset = 0");
  }
  return success();
}

static LogicalResult verifyBoxedSubviewLayout(SubViewOp op,
                                              const SubViewVerifyInfo &info) {
  auto cfg = info.srcTy.getConfigAttr();
  if (!cfg)
    cfg = TileBufConfigAttr::getDefault(op.getContext());

  int64_t innerRows = 1;
  int64_t innerCols = 1;
  bool boxed = false;
  int32_t bl = 0;
  int32_t sl = 0;
  if (failed(computeInnerShape(cfg, info.srcTy.getElementType(), innerRows,
                               innerCols, boxed, bl, sl))) {
    return op.emitOpError("unsupported tile layout for subview");
  }
  if (!boxed)
    return success();

  if (info.sizeR % innerRows != 0 || info.sizeC % innerCols != 0) {
    return op.emitOpError(
        "boxed layout subview sizes must be multiples of inner shape");
  }
  if (info.offRConst && info.offR % innerRows != 0)
    return op.emitOpError(
        "boxed layout subview offsets must be multiples of inner shape");
  if (info.offCConst && info.offC % innerCols != 0)
    return op.emitOpError(
        "boxed layout subview offsets must be multiples of inner shape");

  auto srcShape = info.srcTy.getShape();
  if (!hasStaticRank2Shape(srcShape)) {
    return op.emitOpError("boxed layout subview requires static source shape");
  }
  return verifyBoxedSubviewMajorConstraint(op, info, srcShape, bl);
}


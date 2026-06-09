// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Split from PTOVerifyArithmeticC.cpp; kept as a fragment included by PTOVerifyArithmeticC.cpp.
// Intentionally has no local includes.

using namespace mlir;
using namespace mlir::pto;

static LogicalResult verifyTMovFpA5(const TMovFpCommonInfo &info,
                                    TMovFPOp op) {
  if (info.srcTb && !isColMajorRowMajorNZTileBuf(info.srcTb))
    return op.emitOpError(
        "expects src to use blayout=col_major and slayout=row_major");
  return success();
}

mlir::LogicalResult mlir::pto::TMovFPOp::verify() {
  auto verifyA2A3 = [this]() -> LogicalResult {
    auto common = verifyTMovFpCommon(*this);
    if (failed(common))
      return failure();
    return verifyTMovFpA2A3(*common, *this);
  };
  auto verifyA5 = [this]() -> LogicalResult {
    auto common = verifyTMovFpCommon(*this);
    if (failed(common))
      return failure();
    return verifyTMovFpA5(*common, *this);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
// 辅助函数：获取 Rank，支持 ShapedType 和 PTO TileTypes
static int64_t getRankHelper(Type t) {
  if (auto s = dyn_cast<ShapedType>(t)) return s.getRank();
  if (auto tile = dyn_cast<pto::TileBufType>(t)) return tile.getRank();
  if (auto view = dyn_cast<pto::PartitionTensorViewType>(t)) return view.getRank();
  return -1;
}

static LogicalResult verifyMatmulLike(Operation *op, Type aTy, Type bTy, Type dstTy, bool checkRank = true) {
  // 1. 检查类型 (ShapedType 或 Tile 类型)
  bool aValid = isa<ShapedType, pto::TileBufType, pto::PartitionTensorViewType>(aTy);
  bool bValid = isa<ShapedType, pto::TileBufType, pto::PartitionTensorViewType>(bTy);
  bool dValid = isa<ShapedType, pto::TileBufType, pto::PartitionTensorViewType>(dstTy);
  if (!aValid || !bValid || !dValid)
    return op->emitOpError("expects inputs/outputs to be shaped types or PTO tile types");

  if (checkRank) {
    int64_t aRank = getRankHelper(aTy);
    int64_t bRank = getRankHelper(bTy);
    int64_t dRank = getRankHelper(dstTy);
    // 检查 Rank 一致性
    if (aRank != -1 && dRank != -1 && aRank != dRank)
      return op->emitOpError("expects a and dst to have the same rank");
    if (bRank != -1 && dRank != -1 && bRank != dRank)
      return op->emitOpError("expects b and dst to have the same rank");
  }

  return success();
}

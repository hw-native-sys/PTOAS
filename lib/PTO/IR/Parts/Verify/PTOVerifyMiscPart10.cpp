// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Split from PTOVerifyMisc.cpp; kept as a fragment included by PTOVerifyMisc.cpp.
// Intentionally has no local includes.

using namespace mlir;
using namespace mlir::pto;

static LogicalResult verifyTRowExpandCommon(TRowExpandOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyNDStyleVecTile(op, dstTy, "dst")))
    return failure();
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  if (!srcSpace || *srcSpace != pto::AddressSpace::VEC)
    return op.emitOpError("expects src to be in the vec address space");
  if (auto srcTb = dyn_cast<pto::TileBufType>(srcTy)) {
    if (srcTb.getSLayoutValueI32() !=
        static_cast<int32_t>(pto::SLayout::NoneBox)) {
      return op.emitOpError("expects src to use the none_box slayout");
    }
  }
  if (getElemTy(srcTy) != getElemTy(dstTy))
    return op.emitOpError("expects src and dst to have the same element type");
  if (!isSupportedVecElemType(getElemTy(srcTy), /*allowBf16=*/true,
                              /*allowInt8=*/true)) {
    return op.emitOpError("expects trowexpand element type to be supported");
  }
  auto srcValid = getValidShapeVec(op.getSrc());
  auto dstValid = getValidShapeVec(op.getDst());
  if (srcValid.size() != kPTORowColRank || dstValid.size() != kPTORowColRank)
    return op.emitOpError("expects src and dst to have rank-2 valid_shape");
  if (srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
      srcValid[0] != dstValid[0]) {
    return op.emitOpError("expects src and dst to have the same valid_shape[0]");
  }
  Operation *opBase = op.getOperation();
  auto checkNonZero = [opBase](ArrayRef<int64_t> valid, StringRef name)
      -> LogicalResult {
    if (valid[0] != ShapedType::kDynamic && valid[0] == 0)
      return opBase->emitOpError()
             << "expects " << name << " valid_shape[0] to be non-zero";
    if (valid[1] != ShapedType::kDynamic && valid[1] == 0)
      return opBase->emitOpError()
             << "expects " << name << " valid_shape[1] to be non-zero";
    return success();
  };
  if (failed(checkNonZero(srcValid, "src")) ||
      failed(checkNonZero(dstValid, "dst")))
    return failure();
  return success();
}

mlir::LogicalResult mlir::pto::TRowExpandOp::verify() {
  auto verifyA2A3 = [this]() -> LogicalResult {
    return verifyTRowExpandCommon(*this);
  };
  auto verifyA5 = [this]() -> LogicalResult {
    return verifyTRowExpandCommon(*this);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

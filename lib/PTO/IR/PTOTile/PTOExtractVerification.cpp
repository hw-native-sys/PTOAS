// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

static LogicalResult verifyTExtractA2A3Acc(TExtractOp op,
                                           const TExtractCommon &c) {
  if (*c.dstSpace != pto::AddressSpace::MAT) {
    return op.emitOpError("expects A2/A3 acc-source textract dst to use loc=mat");
  }
  if (c.srcTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
      c.srcTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
    return op.emitOpError("expects A2/A3 acc-source textract src to use blayout=col_major and slayout=row_major");
  }
  if (c.dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
      c.dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
    return op.emitOpError("expects A2/A3 acc-source textract dst to use blayout=col_major and slayout=row_major");
  }
  if (c.dstTb.getSFractalSizeI32() != mlir::pto::kValue512) {
      return op.emitOpError("expects A2/A3 acc-source textract dst fractal size to be 512");
  }
  const bool hasFp = static_cast<bool>(op.getFp());
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  if (hasFp || hasPreQuantScalar) {
    if (!isA2A3AccQuantExtractTypePair(c.srcElem, c.dstElem)) {
      return op.emitOpError(
          "expects A2/A3 acc preQuantScalar textract element types to be "
          "(src=f32,dst=i8) or (src=i32,dst=i8/f16/i16)");
    }
  } else if (!isA2A3AccCastExtractTypePair(c.srcElem, c.dstElem)) {
    return op.emitOpError(
        "expects A2/A3 acc textract element types to be src=f32, dst=f16/bf16");
  }
  return success();
}

static LogicalResult verifyTExtractA2A3Mat(TExtractOp op,
                                           const TExtractCommon &c) {
  if (*c.srcSpace != pto::AddressSpace::MAT) {
    return op.emitOpError("expects A2/A3 textract src to use loc=mat, loc=acc, or loc=vec");
  }
  if (*c.dstSpace != pto::AddressSpace::LEFT &&
      *c.dstSpace != pto::AddressSpace::RIGHT) {
    return op.emitOpError("expects A2/A3 textract dst to use loc=left, loc=right, loc=mat, or loc=vec");
  }
  if (!hasMatExtractSourceLayoutA2A3(c.srcTb)) {
    return op.emitOpError("expects A2/A3 textract src to use a supported mat blayout/slayout combination");
  }
  if (*c.dstSpace == pto::AddressSpace::LEFT) {
    if (c.dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
        c.dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
      return op.emitOpError("expects A2/A3 left dst to use row_major blayout and row_major slayout");
    }
  } else {
    if (c.dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
        c.dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::ColMajor)) {
      return op.emitOpError("expects A2/A3 right dst to use row_major blayout and col_major slayout");
    }
  }
  return success();
}

static LogicalResult verifyTExtractA2A3(TExtractOp op) {
  auto common = verifyTExtractCommon(op, /*allowLowPrecision=*/false);
  if (failed(common)) {
    return failure();
  }
  const TExtractCommon &c = *common;
  const bool hasFp = static_cast<bool>(op.getFp());
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  const bool hasRelu = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
  if (!isA2A3ExtractElemType(c.dstElem) && !(hasFp && c.dstElem.isInteger(mlir::pto::kValue16))) {
      return op.emitOpError("expects A2/A3 textract element type to be i8/f16/bf16/f32");
  }
  if (failed(verifyTExtractFpFormLoc(op, c.srcSpace))) {
    return failure();
  }
  if (op.getAccToVecModeAttr()) {
    return op.emitOpError("expects accToVecMode only on A5 acc->vec textract forms");
  }
  if (c.srcSpace && c.dstSpace && *c.srcSpace == pto::AddressSpace::VEC &&
      *c.dstSpace == pto::AddressSpace::VEC) {
    if (hasPreQuantScalar || hasRelu) {
      return op.emitOpError("expects vec->vec textract to use the base form without preQuantScalar or reluPreMode");
    }
    return success();
  }
  if (!c.srcSpace || !c.dstSpace) {
    return op.emitOpError("expects src and dst to have explicit loc");
  }
  if (*c.srcSpace == pto::AddressSpace::ACC) {
    return verifyTExtractA2A3Acc(op, c);
  }
  return verifyTExtractA2A3Mat(op, c);
}

static bool isTExtractA5SupportedPair(pto::AddressSpace srcSpace,
                                      pto::AddressSpace dstSpace) {
  return (srcSpace == pto::AddressSpace::MAT &&
          (dstSpace == pto::AddressSpace::LEFT ||
           dstSpace == pto::AddressSpace::RIGHT ||
           dstSpace == pto::AddressSpace::SCALING)) ||
         (srcSpace == pto::AddressSpace::VEC &&
          (dstSpace == pto::AddressSpace::MAT ||
           dstSpace == pto::AddressSpace::VEC)) ||
         (srcSpace == pto::AddressSpace::ACC &&
          (dstSpace == pto::AddressSpace::MAT ||
           dstSpace == pto::AddressSpace::VEC));
}

static LogicalResult verifyTExtractA5Mat(TExtractOp op,
                                         const TExtractCommon &c) {
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  const bool hasRelu = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
  if (hasPreQuantScalar || hasRelu) {
    return op.emitOpError("expects mat-source textract to use the base form without preQuantScalar or reluPreMode");
  }
  if (!hasMatExtractSourceLayoutA5(c.srcTb, *c.dstSpace)) {
    return op.emitOpError("expects A5 textract src to use a supported mat blayout/slayout combination");
  }
  if (*c.dstSpace == pto::AddressSpace::LEFT) {
    if (c.dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
        c.dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
      return op.emitOpError("expects A5 left dst to use col_major blayout and row_major slayout");
    }
  } else if (*c.dstSpace == pto::AddressSpace::RIGHT) {
    if (c.dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
        c.dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::ColMajor)) {
      return op.emitOpError("expects A5 right dst to use row_major blayout and col_major slayout");
    }
  }
  return success();
}

static LogicalResult verifyTExtractA5Acc(TExtractOp op,
                                         const TExtractCommon &c) {
  if (c.srcTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
      c.srcTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
    return op.emitOpError("expects A5 acc-source textract src to use blayout=col_major and slayout=row_major");
  }
  if (*c.dstSpace == pto::AddressSpace::MAT) {
    if (c.dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
        c.dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
      return op.emitOpError("expects A5 acc-source textract dst to use blayout=col_major and slayout=row_major");
    }
  } else {
    if (!isRowMajorNoneBoxND(c.dstTb)) {
      return op.emitOpError("expects A5 acc->vec textract dst to use ND layout (blayout=row_major, slayout=none_box)");
    }
  }
  const bool hasFp = static_cast<bool>(op.getFp());
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  if (hasFp || hasPreQuantScalar) {
    if (!isA5AccQuantExtractTypePair(c.srcElem, c.dstElem)) {
      return op.emitOpError(
          "expects A5 acc preQuantScalar textract element types to be "
          "(src=f32,dst=i8/fp8/f16/bf16/f32) or (src=i32,dst=i8/f16/bf16)");
    }
  } else if (!isA5AccCastExtractTypePair(c.srcElem, c.dstElem)) {
    return op.emitOpError(
        "expects A5 acc textract element types to be "
        "(src=f32,dst=f16/bf16/f32) or (src=i32,dst=i32)");
  }
  return success();
}

static LogicalResult verifyTExtractA5(TExtractOp op) {
  auto common = verifyTExtractCommon(op, /*allowLowPrecision=*/true);
  if (failed(common)) {
    return failure();
  }
  const TExtractCommon &c = *common;
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  const bool hasRelu = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
  if (!isA5ExtractElemType(c.dstElem)) {
    return op.emitOpError("expects A5 textract element type to be an fp8/f16/bf16/f32 or int8 family type");
  }
  if (failed(verifyTExtractFpFormLoc(op, c.srcSpace))) {
    return failure();
  }
  if (op.getAccToVecModeAttr() &&
      (!c.srcSpace || !c.dstSpace || *c.srcSpace != pto::AddressSpace::ACC ||
       *c.dstSpace != pto::AddressSpace::VEC)) {
    return op.emitOpError("expects accToVecMode only on A5 acc->vec textract forms");
  }
  if (!c.srcSpace || !c.dstSpace) {
    return op.emitOpError("expects src and dst to have explicit loc");
  }
  if (!isTExtractA5SupportedPair(*c.srcSpace, *c.dstSpace)) {
    return op.emitOpError("expects A5 textract to use a supported src/dst loc pair");
  }
  if (*c.srcSpace == pto::AddressSpace::MAT) {
    return verifyTExtractA5Mat(op, c);
  }
  if (*c.srcSpace == pto::AddressSpace::VEC &&
      *c.dstSpace == pto::AddressSpace::VEC) {
    if (hasPreQuantScalar || hasRelu) {
      return op.emitOpError("expects vec-source textract to use the base form without preQuantScalar or reluPreMode");
    }
    if (!isRowMajorNoneBoxND(c.srcTb) || !isRowMajorNoneBoxND(c.dstTb)) {
      return op.emitOpError(
          "expects A5 vec->vec textract src/dst to use ND layout "
          "(blayout=row_major, slayout=none_box)");
    }
    return success();
  }
  if (*c.srcSpace == pto::AddressSpace::ACC) {
    return verifyTExtractA5Acc(op, c);
  }
  return success();
}

mlir::LogicalResult mlir::pto::TExtractOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTExtractA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTExtractA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
static bool isA5VectorPreQuantTypePair(Type srcElem, Type dstElem);
static bool isA2A3AccCastInsertTypePair(Type srcElem, Type dstElem) {
  return srcElem.isF32() && (dstElem.isF16() || dstElem.isBF16());
}

static bool isA2A3AccQuantInsertTypePair(Type srcElem, Type dstElem) {
  return isA2A3AccQuantTypePair(srcElem, dstElem);
}

static bool isColMajorRowMajorNZ(pto::TileBufType ty) {
  return ty.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) &&
         ty.getSLayoutValueI32() == static_cast<int32_t>(pto::SLayout::RowMajor);
}

static bool isA5SupportedVecElemType(Type ty) {
  if (isPTOFloat8Type(ty) || isPTOHiFloat8Type(ty) || isPTOFloat4PackedType(ty)) {
    return true;
  }
  if (auto it = dyn_cast<IntegerType>(ty)) {
    return it.getWidth() == 8 || it.getWidth() == 32;
  }
  if (auto ft = dyn_cast<FloatType>(ty)) {
      return ft.getWidth() == mlir::pto::kValue8 || ft.isF16() || ft.isBF16() || ft.isF32();
  }
  return false;
}

static bool isA2A3VecInsertElemType(Type ty) {
    return ty.isInteger(mlir::pto::kValue8) || ty.isF16() || ty.isBF16() || ty.isF32();
}

using TInsertCommon = TileTransferCommon;

static FailureOr<TInsertCommon> verifyTInsertCommon(TInsertOp op,
                                                    bool allowLowPrecision) {
  return verifyTileTransferCommon(
      op, op.getSrc(), op.getDst(), op.getIndexRow(), op.getIndexCol(),
      allowLowPrecision, /*includeIndexAndIntOpsInConstFold=*/true,
      /*insert=*/true);
}

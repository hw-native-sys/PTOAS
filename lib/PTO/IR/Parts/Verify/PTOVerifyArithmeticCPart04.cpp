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

mlir::LogicalResult mlir::pto::TExtractFPOp::verify() {
  auto verifyA2A3 = [this]() -> LogicalResult {
    return verifyVectorPreQuantTransferOp(
        getOperation(), getSrc(), getFp(), getDst(), getIndexRow(),
        getIndexCol(),
        /*isInsertOp=*/false,
        /*requireDstFractal512=*/true, isA2A3VectorPreQuantTypePair,
        "expects A2/A3 textract_fp element types to be (src=f32,dst=i8) "
        "or (src=i32,dst=i8/f16/i16)");
  };
  auto verifyA5 = [this]() -> LogicalResult {
    return verifyVectorPreQuantTransferOp(
        getOperation(), getSrc(), getFp(), getDst(), getIndexRow(),
        getIndexCol(),
        /*isInsertOp=*/false,
        /*requireDstFractal512=*/false, isA5VectorPreQuantTypePair,
        "expects A5 textract_fp element types to be (src=f32,dst=i8/fp8/f16/bf16/f32) "
        "or (src=i32,dst=i8/f16/bf16)");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TInsertFPOp::verify() {
  auto verifyA2A3 = [this]() -> LogicalResult {
    return verifyVectorPreQuantTransferOp(
        getOperation(), getSrc(), getFp(), getDst(), getIndexRow(),
        getIndexCol(),
        /*isInsertOp=*/true,
        /*requireDstFractal512=*/true, isA2A3VectorPreQuantTypePair,
        "expects A2/A3 tinsert_fp element types to be (src=f32,dst=i8) "
        "or (src=i32,dst=i8/f16/i16)");
  };
  auto verifyA5 = [this]() -> LogicalResult {
    return verifyVectorPreQuantTransferOp(
        getOperation(), getSrc(), getFp(), getDst(), getIndexRow(),
        getIndexCol(),
        /*isInsertOp=*/true,
        /*requireDstFractal512=*/false, isA5VectorPreQuantTypePair,
        "expects A5 tinsert_fp element types to be (src=f32,dst=i8/fp8/f16/bf16/f32) "
        "or (src=i32,dst=i8/f16/bf16)");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static int64_t getFillPadElemBytes(Type type) {
  unsigned elemBytes = getPTOStorageElemByteSize(type);
  return elemBytes == 0 ? -1 : static_cast<int64_t>(elemBytes);
}

static LogicalResult verifyTFillPadMatHomogeneousConstraint(Operation *op,
                                                            Type srcTy,
                                                            Type dstTy,
                                                            llvm::StringRef opName) {
  if (opName != "tfillpad")
    return success();
  auto srcTb = mlir::dyn_cast<mlir::pto::TileBufType>(srcTy);
  auto dstTb = mlir::dyn_cast<mlir::pto::TileBufType>(dstTy);
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  if (!(srcTb && dstTb && srcSpace && dstSpace &&
        *srcSpace == mlir::pto::AddressSpace::MAT &&
        *dstSpace == mlir::pto::AddressSpace::MAT && srcTb != dstTb)) {
    return success();
  }

  auto dimToStr = [](int64_t dim) -> std::string {
    return dim == ShapedType::kDynamic ? "?" : std::to_string(dim);
  };
  SmallVec4<std::string> mismatchFields;
  auto srcValid = getValidShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (srcValid.size() == kPTORowColRank && dstValid.size() == kPTORowColRank) {
    if (srcValid[0] != dstValid[0])
      mismatchFields.push_back("v_row (" + dimToStr(srcValid[0]) + " vs " +
                               dimToStr(dstValid[0]) + ")");
    if (srcValid[1] != dstValid[1])
      mismatchFields.push_back("v_col (" + dimToStr(srcValid[1]) + " vs " +
                               dimToStr(dstValid[1]) + ")");
  }
  if (srcTb.getPadValueI32() != dstTb.getPadValueI32()) {
    mismatchFields.push_back("pad (" + std::to_string(srcTb.getPadValueI32()) +
                             " vs " + std::to_string(dstTb.getPadValueI32()) +
                             ")");
  }

  auto diag = op->emitError()
              << "expects src/dst tile types to be lowerable to TFILLPAD "
                 "for loc=mat";
  if (!mismatchFields.empty())
    diag << "; mismatching fields: " << llvm::join(mismatchFields, ", ");
  diag << "\n  src: " << srcTy;
  diag << "\n  dst: " << dstTy;
  diag << "\n  note: heterogeneous TFILLPAD overload is only available for loc=vec";
  return failure();
}

static LogicalResult verifyTFillPadDstPad(Operation *op, Type dstTy,
                                          llvm::StringRef opName) {
  if (auto dstTileTy = mlir::dyn_cast<mlir::pto::TileBufType>(dstTy)) {
    auto padAttr =
        mlir::dyn_cast<mlir::pto::PadValueAttr>(dstTileTy.getPadValueAttr());
    if (!padAttr || padAttr.getValue() == mlir::pto::PadValue::Null)
      return op->emitError() << "expects dst PadVal != Null for " << opName;
  }
  return success();
}

static LogicalResult verifyTFillPadShapeCompatibility(Operation *op,
                                                      ArrayRef<int64_t> srcShape,
                                                      ArrayRef<int64_t> dstShape,
                                                      bool allowDstExpand,
                                                      llvm::StringRef opName) {
  if (!allowDstExpand) {
    if (srcShape != dstShape) {
      return op->emitError()
             << "expects src and dst to have the same static shape for "
             << opName;
    }
    return mlir::success();
  }
  if (srcShape[0] > dstShape[0] || srcShape[1] > dstShape[1]) {
    return op->emitError()
           << "expects dst static shape to be >= src static shape for "
           << opName;
  }
  return mlir::success();
}

static mlir::LogicalResult verifyTFillPadLike(Operation *op, Type srcTy, Type dstTy,
                                              bool allowDstExpand,
                                              llvm::StringRef opName) {
  if (!isPTOShapedLike(srcTy) || !isPTOShapedLike(dstTy))
    return op->emitError("expects src/dst to be PTO shaped-like types");

  auto srcShape = getShapeVec(srcTy);
  auto dstShape = getShapeVec(dstTy);
  if (srcShape.size() != kPTORowColRank || dstShape.size() != kPTORowColRank)
    return op->emitError("expects rank-2 shaped types for src/dst");

  int64_t srcB = getFillPadElemBytes(getElemTy(srcTy));
  int64_t dstB = getFillPadElemBytes(getElemTy(dstTy));
  if (srcB < 0 || dstB < 0)
    return op->emitError("unsupported element type (expects int/float element types)");
  if (srcB != dstB)
    return op->emitError("expects sizeof(src element) == sizeof(dst element)");
  if (!(srcB == static_cast<int64_t>(kPTOByteSize) ||
        srcB == static_cast<int64_t>(kPTOHalfWordBytes) ||
        srcB == static_cast<int64_t>(kPTOWordBytes)))
    return op->emitError("expects element size to be 1, 2, or 4 bytes");
  if (failed(verifyTFillPadMatHomogeneousConstraint(op, srcTy, dstTy, opName)) ||
      failed(verifyTFillPadDstPad(op, dstTy, opName)) ||
      failed(verifyTFillPadShapeCompatibility(op, srcShape, dstShape,
                                              allowDstExpand, opName))) {
    return failure();
  }
  return mlir::success();
}

mlir::LogicalResult mlir::pto::TFillPadOp::verify() {
  return verifyTFillPadLike(getOperation(), getSrc().getType(), getDst().getType(),
                            /*allowDstExpand=*/false, "tfillpad");
}

mlir::LogicalResult mlir::pto::TFillPadExpandOp::verify() {
  return verifyTFillPadLike(getOperation(), getSrc().getType(), getDst().getType(),
                            /*allowDstExpand=*/true, "tfillpad_expand");
}

mlir::LogicalResult mlir::pto::TFillPadInplaceOp::verify() {
  return verifyTFillPadLike(getOperation(), getSrc().getType(), getDst().getType(),
                            /*allowDstExpand=*/false, "tfillpad_inplace");
}

struct GatherSrcDstCommon {
  Type srcTy;
  Type dstTy;
  Type srcElem;
  Type dstElem;
};

struct GatherIndexCommon {
  GatherSrcDstCommon base;
  Type idxTy;
  Type tmpTy;
  IntegerType idxElem;
};

struct GatherCompareCommon {
  GatherSrcDstCommon base;
  Type cdstTy;
  Type tmpTy;
  Type cdstElem;
  pto::CmpMode cmpMode;
};

static bool isSupportedGatherElemTypeA5Index(Type ty) {
  if (ty.isF16() || ty.isF32())
    return true;
  if (auto it = dyn_cast<IntegerType>(ty))
    return it.getWidth() == kPTOI8BitWidth || it.getWidth() == kPTOI16BitWidth || it.getWidth() == kPTOI32BitWidth;
  return false;
}

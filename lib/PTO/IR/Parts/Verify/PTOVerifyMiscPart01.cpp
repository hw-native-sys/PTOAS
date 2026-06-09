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

LogicalResult LoadScalarOp::verify() {
  Type ptrTy = getPtr().getType();
  Type elemTy;
  if (auto pty = dyn_cast<mlir::pto::PtrType>(ptrTy)) {
    elemTy = pty.getElementType();
  } else if (auto memTy = dyn_cast<MemRefType>(ptrTy)) {
    elemTy = memTy.getElementType();
    if (!isGmAddressSpaceAttr(memTy.getMemorySpace()))
      return emitOpError() << "scalar load only supports GM address space pointers";
  } else {
    return emitOpError("expects ptr to be !pto.ptr or memref type");
  }

  if (getValue().getType() != elemTy)
    return emitOpError("expects result type to match ptr element type");

  return success();
}
// ---- StoreScalarOp ----
LogicalResult StoreScalarOp::verify() {
  Type ptrTy = getPtr().getType();
  Type elemTy;
  if (auto pty = dyn_cast<mlir::pto::PtrType>(ptrTy)) {
    elemTy = pty.getElementType();
  } else if (auto memTy = dyn_cast<MemRefType>(ptrTy)) {
    elemTy = memTy.getElementType();
    if (!isGmAddressSpaceAttr(memTy.getMemorySpace()))
      return emitOpError() << "scalar store only supports GM address space pointers";
  } else {
    return emitOpError("expects ptr to be !pto.ptr or memref type");
  }

  if (getValue().getType() != elemTy)
    return emitOpError("expects value type to match ptr element type");

  return success();
}

// ---- GetBufOp / RlsBufOp ----
static LogicalResult verifyBufSyncOp(Operation *op, Attribute opTypeAttr,
                                     IntegerAttr bufIdAttr, IntegerAttr modeAttr) {
  if (!opTypeAttr)
    return op->emitOpError("expects 'op_type' attribute");

  auto opTypeOr = parseSyncOpTypeLikeAttr(opTypeAttr);
  if (failed(opTypeOr)) {
    auto diag =
        op->emitOpError("expects 'op_type' to be pipe_event_type/sync_op_type, got ");
    diag << opTypeAttr;
    return failure();
  }
  pto::PIPE pipe = mapSyncOpTypeToPipe(*opTypeOr);
  if (!isConcreteSyncPipe(pipe))
    return op->emitOpError("expects 'op_type' to map to a concrete pipe, not PIPE_ALL/PIPE_UNASSIGNED");

  if (!bufIdAttr)
    return op->emitOpError("expects 'buf_id' attribute");
  static constexpr int64_t kPTOSyncMinBufferId = 0;
  static constexpr int64_t kPTOSyncMaxBufferId = 31;
  int64_t bufId = bufIdAttr.getInt();
  if (bufId < kPTOSyncMinBufferId || bufId > kPTOSyncMaxBufferId)
    return op->emitOpError("expects 'buf_id' in range [0, 31]");

  if (modeAttr) {
    int64_t mode = modeAttr.getInt();
    if (mode < 0)
      return op->emitOpError("expects 'mode' to be non-negative");
  }

  return success();
}

LogicalResult GetBufOp::verify() {
  return verifyBufSyncOp(getOperation(), getOpTypeAttr(), getBufIdAttr(),
                         getModeAttr());
}

LogicalResult RlsBufOp::verify() {
  return verifyBufSyncOp(getOperation(), getOpTypeAttr(), getBufIdAttr(),
                         getModeAttr());
}
// ---- TOp ----

static LogicalResult verifyMatmulBiasLikeOp(Operation *op, Type aTy, Type bTy,
                                            Type biasTy, Type dstTy,
                                            bool useGemvOperands) {
  if (useGemvOperands) {
    if (failed(verifyGemvTileOperands(op, aTy, bTy, dstTy)))
      return failure();
  } else {
    if (failed(verifyMatTileOperands(op, aTy, bTy, dstTy)))
      return failure();
  }
  if (failed(verifyMatBiasTile(op, biasTy, dstTy)))
    return failure();
  if (failed(verifyMatmulTypeTriple(op, getElemTy(aTy), getElemTy(bTy),
                                    getElemTy(dstTy))))
    return failure();
  return verifyMatmulLike(op, aTy, bTy, dstTy);
}

template <typename ExtraVerifyFn>
static LogicalResult verifyMatmulMxA2A3LikeOp(Operation *op, Type aScaleTy,
                                              Type bScaleTy, Type aTy, Type bTy,
                                              Type dstTy,
                                              ExtraVerifyFn extraVerify) {
  if (failed(verifyTileBufCommon(op, aScaleTy, "a_scale")) ||
      failed(verifyTileBufCommon(op, bScaleTy, "b_scale")))
    return failure();
  if (failed(extraVerify()))
    return failure();
  return verifyMatmulLike(op, aTy, bTy, dstTy);
}

template <typename VerifyBaseFn>
static LogicalResult verifyMatmulMxA5LikeOp(Operation *op, Type aTy, Type bTy,
                                            Type dstTy,
                                            VerifyBaseFn verifyBase) {
  if (failed(verifyBase()))
    return failure();
  return verifyA5MxTypeTriple(op, aTy, bTy, dstTy, "a", "b", "dst");
}

LogicalResult TGemvBiasOp::verify() {
  auto verifyA2A3 = [this]() -> LogicalResult {
    return verifyMatmulBiasLikeOp(*this, getA().getType(), getB().getType(),
                                  getBias().getType(), getDst().getType(),
                                  /*useGemvOperands=*/true);
  };
  auto verifyA5 = [&verifyA2A3]() -> LogicalResult { return verifyA2A3(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TGemvMxOp::verify() {
  auto verifyA2A3 = [this]() -> LogicalResult {
    return emitOpError("tgemv.mx is only supported on A5 targets");
  };
  auto verifyA5 = [this]() -> LogicalResult {
    if (failed(verifyScaleTileMatchesOperand(*this, getAScale().getType(),
                                             getA().getType(), "a_scale", "a")) ||
        failed(verifyScaleTileMatchesOperand(*this, getBScale().getType(),
                                             getB().getType(), "b_scale", "b")) ||
        failed(verifyGemvTileOperands(*this, getA().getType(), getB().getType(),
                                      getDst().getType())))
      return failure();
    if (failed(verifyA5MxTypeTriple(*this, getA().getType(), getB().getType(),
                                    getDst().getType(), "a", "b", "dst")))
      return failure();
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TGemvMxAccOp::verify() {
  auto verifyA2A3 = [this]() -> LogicalResult {
    return emitOpError("tgemv.mx.acc is only supported on A5 targets");
  };
  auto verifyA5 = [this]() -> LogicalResult {
    if (failed(verifyAccTileCommon(*this, getCIn().getType(), "c_in")) ||
        failed(verifyScaleTileMatchesOperand(*this, getAScale().getType(),
                                             getA().getType(), "a_scale", "a")) ||
        failed(verifyScaleTileMatchesOperand(*this, getBScale().getType(),
                                             getB().getType(), "b_scale", "b")) ||
        failed(verifyGemvTileOperands(*this, getA().getType(), getB().getType(),
                                      getDst().getType())))
      return failure();
    if (failed(verifyA5MxTypeTriple(*this, getA().getType(), getB().getType(),
                                    getDst().getType(), "a", "b", "dst")))
      return failure();
    if (failed(verifyTileBufSameElemType(*this, getCIn().getType(),
                                             getDst().getType(), "c_in", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, getCIn().getType(),
                                           getDst().getType(), "c_in", "dst")))
      return failure();
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TGemvMxBiasOp::verify() {
  auto verifyA2A3 = [this]() -> LogicalResult {
    return emitOpError("tgemv.mx.bias is only supported on A5 targets");
  };
  auto verifyA5 = [this]() -> LogicalResult {
    if (failed(verifyScaleTileMatchesOperand(*this, getAScale().getType(),
                                             getA().getType(), "a_scale", "a")) ||
        failed(verifyScaleTileMatchesOperand(*this, getBScale().getType(),
                                             getB().getType(), "b_scale", "b")) ||
        failed(verifyGemvTileOperands(*this, getA().getType(), getB().getType(),
                                      getDst().getType())) ||
        failed(verifyMatBiasTile(*this, getBias().getType(), getDst().getType(),
                                 /*requireFloatBias=*/true)))
      return failure();
    if (failed(verifyA5MxTypeTriple(*this, getA().getType(), getB().getType(),
                                    getDst().getType(), "a", "b", "dst")))
      return failure();
    auto biasShape = getShapeVec(getBias().getType());
    auto dstShape = getShapeVec(getDst().getType());
    if (biasShape.size() != kPTORowColRank ||
        dstShape.size() != kPTORowColRank)
      return emitOpError("expects bias and dst to be rank-2 for tgemv.mx.bias");
    if (biasShape[1] != ShapedType::kDynamic && dstShape[1] != ShapedType::kDynamic &&
        biasShape[1] != dstShape[1])
      return emitOpError("expects bias and dst to have the same column shape");
    if (failed(verifyTileBufSameValidShape(*this, getBias().getType(),
                                           getDst().getType(), "bias", "dst")))
      return failure();
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

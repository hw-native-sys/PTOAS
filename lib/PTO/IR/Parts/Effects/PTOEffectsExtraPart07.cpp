// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Split from PTOEffectsExtra.cpp; kept as a fragment included by PTOEffectsExtra.cpp.
// Intentionally has no local includes.

using namespace mlir;
using namespace mlir::pto;

#define PTO_DEFINE_OP_VERIFY(OpName) LogicalResult OpName::verify()

LogicalResult CommTScatterOp::verify() {
  if (failed(verifyRootedCommTileTransfer(getOperation(), getSrc(), getGroup(),
                                          getRoot(), getPing(), getPong())))
    return failure();
  if (getElemTy(getSrc().getType()) != getElemTy(getGroup().front().getType()))
    return emitOpError("expects src element type to match group member type");
  return success();
}

LogicalResult TReduceOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  if (failed(verifyCommGlobalLike(*this, getDst(), "dst")) ||
      failed(verifyCommStagingTileLike(*this, getAcc(), "acc")) ||
      failed(verifyCommStagingTileLike(*this, getRecvPing(), "recv_ping")) ||
      failed(verifyCommPingPongSameType(*this, getRecvPing(), getRecvPong(),
                                        "recv_ping", "recv_pong")) ||
      failed(verifyCommGlobalGroup(*this, getGroup(), "group")))
    return failure();
  if (getRoot() >= static_cast<uint32_t>(getGroup().size()))
    return emitOpError("expects root to index into group operands");
  if (getElemTy(getDst().getType()) != getElemTy(getGroup().front().getType()))
    return emitOpError("expects dst element type to match group member type");
  if (getAcc().getType() != getRecvPing().getType())
    return emitOpError("expects acc and recv_ping to have identical types");
  if (getElemTy(getAcc().getType()) != getElemTy(getDst().getType()))
    return emitOpError("expects accumulator/receive tiles to match dst element type");
  return success();
}

PTO_DEFINE_OP_VERIFY(AicInitializePipeOp) {
  return verifyFrontendInitCommon(*this, FunctionKernelKind::Cube, "cube");
}

PTO_DEFINE_OP_VERIFY(AivInitializePipeOp) {
  return verifyFrontendInitCommon(*this, FunctionKernelKind::Vector, "vector");
}

LogicalResult TAllocToAivOp::verify() {
  if (failed(verifyFrontendSplitOp(getOperation(), FunctionKernelKind::Cube,
                                   "cube", getId(), getSplit())))
    return failure();
  if (failed(verifyFrontendDataOpDirection(getOperation(), getId(),
                                           /*expectC2V=*/true)))
    return failure();
  return verifyFrontendTensorEntryMatchesInit(getOperation(), getId(),
                                              getEntry().getType());
}

LogicalResult TAllocToAicOp::verify() {
  if (failed(verifyFrontendSplitOp(getOperation(), FunctionKernelKind::Vector,
                                   "vector", getId(), getSplit())))
    return failure();
  if (failed(verifyFrontendDataOpDirection(getOperation(), getId(),
                                           /*expectC2V=*/false)))
    return failure();
  return verifyFrontendTensorEntryMatchesInit(getOperation(), getId(),
                                              getEntry().getType());
}

LogicalResult TPushToAivOp::verify() {
  if (failed(verifyFrontendSplitOp(getOperation(), FunctionKernelKind::Cube,
                                   "cube", getId(), getSplit())))
    return failure();
  if (failed(verifyFrontendDataOpDirection(getOperation(), getId(),
                                           /*expectC2V=*/true)))
    return failure();
  return verifyFrontendTensorEntryMatchesInit(getOperation(), getId(),
                                              getTile().getType());
}

LogicalResult TPushToAicOp::verify() {
  if (failed(verifyFrontendSplitOp(getOperation(), FunctionKernelKind::Vector,
                                   "vector", getId(), getSplit())))
    return failure();
  if (failed(verifyFrontendDataOpDirection(getOperation(), getId(),
                                           /*expectC2V=*/false)))
    return failure();
  return verifyFrontendTensorEntryMatchesInit(getOperation(), getId(),
                                              getTile().getType());
}

PTO_DEFINE_OP_VERIFY(TPopFromAicOp) {
  return verifyFrontendPopOp(*this, FunctionKernelKind::Vector, "vector",
                             /*expectC2V=*/true);
}

PTO_DEFINE_OP_VERIFY(TPopFromAivOp) {
  return verifyFrontendPopOp(*this, FunctionKernelKind::Cube, "cube",
                             /*expectC2V=*/false);
}

#undef PTO_DEFINE_OP_VERIFY

LogicalResult TFreeFromAicOp::verify() {
  if (failed(verifyFrontendSplitOp(getOperation(), FunctionKernelKind::Vector,
                                   "vector", getId(), getSplit())))
    return failure();
  if (failed(verifyFrontendDataOpDirection(getOperation(), getId(),
                                           /*expectC2V=*/true)))
    return failure();
  if (getEntry())
    return verifyFrontendTensorEntryMatchesInit(getOperation(), getId(),
                                                getEntry().getType());
  return success();
}

LogicalResult TFreeFromAivOp::verify() {
  if (failed(verifyFrontendSplitOp(getOperation(), FunctionKernelKind::Cube,
                                   "cube", getId(), getSplit())))
    return failure();
  if (failed(verifyFrontendDataOpDirection(getOperation(), getId(),
                                           /*expectC2V=*/false)))
    return failure();
  if (getEntry())
    return verifyFrontendTensorEntryMatchesInit(getOperation(), getId(),
                                                getEntry().getType());
  return success();
}

LogicalResult InitializeL2G2LPipeOp::verify() {
  if (failed(verifyPipeShape(getOperation(), getDirMask(), getSlotSize(),
                             getSlotNum(),
                             getFlagBaseAttr()
                                 ? std::optional<int32_t>(getFlagBaseAttr().getInt())
                                 : std::nullopt)))
    return failure();

  if (!getLocalAddr()) {
    if (getPeerLocalAddr())
      return emitOpError("'peer_local_addr' requires 'local_addr'");
    if (getLocalSlotNumAttr())
      return emitOpError(
          "'local_slot_num' is only allowed when 'local_addr' is present");
    return success();
  }

  if (auto localSlotNumAttr = getLocalSlotNumAttr()) {
    int32_t localSlotNum = localSlotNumAttr.getInt();
    if (localSlotNum <= 0)
      return emitOpError("expects 'local_slot_num' to be greater than 0");
    if (static_cast<uint32_t>(localSlotNum) > getSlotNum())
      return emitOpError(
          "expects 'local_slot_num' to be less than or equal to slot_num");
  }

  if (getDirMask() == kPTOFrontendDirMaskBidirectional && !getPeerLocalAddr())
    return emitOpError("expects 'peer_local_addr' when dir_mask is 3");
  if (getDirMask() != kPTOFrontendDirMaskBidirectional && getPeerLocalAddr())
    return emitOpError("'peer_local_addr' is only allowed when dir_mask is 3");
  return success();
}

LogicalResult InitializeL2LPipeOp::verify() {
  if (failed(verifyPipeShape(getOperation(), getDirMask(), getSlotSize(),
                              getSlotNum(),
                              getFlagBaseAttr()
                                  ? std::optional<int32_t>(getFlagBaseAttr().getInt())
                                  : std::nullopt)))
    return failure();

  if (getDirMask() == kPTOFrontendDirMaskBidirectional && !getPeerLocalAddr())
    return emitOpError("expects 'peer_local_addr' when dir_mask is 3");
  if (getDirMask() != kPTOFrontendDirMaskBidirectional && getPeerLocalAddr())
    return emitOpError("'peer_local_addr' is only allowed when dir_mask is 3");
  return success();
}

LogicalResult TPushOp::verify() {
  if (failed(verifyFrontendPipeTileAccess(getOperation(), getPipeHandle(),
                                          getSplit(), getTile().getType())))
    return failure();
  if (!isa<TensorViewType>(getTile().getType()) &&
      getPipe() == pto::PIPE::PIPE_UNASSIGNED)
    return emitOpError("tile type must map to a supported producer pipe");
  return success();
}

LogicalResult TAllocOp::verify() {
  if (!isInsideSectionOrAttributedKernel(getOperation()))
    return emitOpError("must be inside pto.section.cube/vector or a kernel_kind function");
  if (failed(verifyPipeHandleProducer(getOperation(), getPipeHandle())))
    return failure();
  if (failed(verifyTensorEntryMatchesInternalPipeInit(
          getOperation(), getPipeHandle(), getEntry().getType())))
    return failure();
  return verifySplitAttr(getOperation(), getSplit());
}

LogicalResult TPopOp::verify() {
  if (failed(verifyFrontendPipeTileAccess(getOperation(), getPipeHandle(),
                                          getSplit(), getTile().getType())))
    return failure();
  if (!isa<TensorViewType>(getTile().getType()) &&
      getPipe() == pto::PIPE::PIPE_UNASSIGNED)
    return emitOpError(
        "tile type and target arch must map to a supported consumer pipe");
  return success();
}

LogicalResult TFreeOp::verify() {
  if (!isInsideSectionOrAttributedKernel(getOperation()))
    return emitOpError("must be inside pto.section.cube/vector or a kernel_kind function");
  if (failed(verifyPipeHandleProducer(getOperation(), getPipeHandle())))
    return failure();
  if (getEntry() &&
      failed(verifyTensorEntryMatchesInternalPipeInit(
          getOperation(), getPipeHandle(), getEntry().getType())))
    return failure();
  return verifySplitAttr(getOperation(), getSplit());
}

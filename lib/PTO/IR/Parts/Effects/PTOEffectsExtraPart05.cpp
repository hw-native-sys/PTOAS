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

static LogicalResult verifyTensorEntryByteSize(Operation *op,
                                               TensorViewType entryViewTy,
                                               InitializeL2G2LPipeOp initOp) {
  auto entryElemCount = getStaticElementCount(entryViewTy.getShape());
  if (!entryElemCount)
    return success();

  uint64_t elemBytes = getElemByteSize(entryViewTy.getElementType());
  if (elemBytes == 0)
    return success();

  uint64_t entryBytes = *entryElemCount * elemBytes;
  int8_t split = getTensorEntrySplit(op);
  uint64_t slotBytes = static_cast<uint64_t>(initOp.getSlotSize());
  bool isSplitEntry = split != 0;
  bool byteSizeMatches =
      entryBytes == slotBytes || (isSplitEntry && entryBytes * 2 == slotBytes);
  if (!byteSizeMatches) {
    return op->emitOpError()
           << "expects pipe entry byte size to match initialize_l2g2l_pipe "
              "slot_size"
           << (isSplitEntry ? " or half slot_size for split entries" : "")
           << " (got entry byte size = " << entryBytes
           << ", slot_size = " << initOp.getSlotSize() << ")";
  }
  return success();
}

static LogicalResult verifyTensorEntryMatchesInternalPipeInit(Operation *op,
                                                              Value pipeHandle,
                                                              Type entryTy) {
  auto entryViewTy = dyn_cast<TensorViewType>(entryTy);
  if (!entryViewTy)
    return success();

  auto initOp = verifyTensorEntryInternalPipeHandle(op, pipeHandle);
  if (failed(initOp) ||
      failed(verifyTensorEntrySlotType(op, entryViewTy, *initOp)) ||
      failed(verifyTensorEntryByteSize(op, entryViewTy, *initOp)))
    return failure();
  return success();
}

static LogicalResult verifyAsyncSessionScratch(Operation *op, Type scratchTy) {
  if (!isa<pto::TileBufType, MemRefType>(scratchTy))
    return op->emitOpError("expects scratch to be tile_buf or memref type");

  auto scratchSpace = getPTOMemorySpaceEnum(scratchTy);
  if (!scratchSpace || *scratchSpace != pto::AddressSpace::VEC)
    return op->emitOpError("expects scratch to be in vec address space");

  auto scratchShape = getShapeVec(scratchTy);
  if (scratchShape.empty() || scratchShape.size() > kPTORowColRank)
    return op->emitOpError("expects scratch to be rank-1 or rank-2");
  for (int64_t dim : scratchShape) {
    if (dim == ShapedType::kDynamic)
      return op->emitOpError("expects scratch to have a static shape");
  }

  auto scratchBytes = getStaticByteSize(scratchTy);
  if (!scratchBytes)
    return op->emitOpError("expects scratch byte size to be statically known");
  if (*scratchBytes < sizeof(uint64_t))
    return op->emitOpError("expects scratch to provide at least 8 bytes");
  return success();
}

static LogicalResult verifyAsyncSessionWorkspace(Operation *op, Type workspaceTy) {
  Type workspaceElemTy;
  if (auto ptrTy = dyn_cast<pto::PtrType>(workspaceTy)) {
    workspaceElemTy = ptrTy.getElementType();
  } else if (auto memTy = dyn_cast<MemRefType>(workspaceTy)) {
    workspaceElemTy = memTy.getElementType();
    if (!isGmAddressSpaceAttr(memTy.getMemorySpace()))
      return op->emitOpError("expects workspace to be in GM address space");
  } else {
    return op->emitOpError("expects workspace to be !pto.ptr or memref type");
  }
  if (!isByteIntegerType(workspaceElemTy))
    return op->emitOpError("expects workspace element type to be an 8-bit integer");
  return success();
}

static LogicalResult verifyAsyncSessionAttrs(BuildAsyncSessionOp op) {
  static constexpr int64_t kPTOAsyncSessionMinSyncId = 0;
  static constexpr int64_t kPTOAsyncSessionMaxSyncId = 7;
  if (auto syncIdAttr = op.getSyncIdAttr()) {
    int64_t syncId = syncIdAttr.getInt();
    if (syncId < kPTOAsyncSessionMinSyncId ||
        syncId > kPTOAsyncSessionMaxSyncId)
      return op.emitOpError("expects sync_id in range [0, 7]");
  }
  if (auto blockBytesAttr = op.getBlockBytesAttr()) {
    if (blockBytesAttr.getInt() <= 0)
      return op.emitOpError("expects block_bytes to be greater than 0");
  }
  if (auto commBlockOffsetAttr = op.getCommBlockOffsetAttr()) {
    if (commBlockOffsetAttr.getInt() < 0)
      return op.emitOpError("expects comm_block_offset to be non-negative");
  }
  if (auto queueNumAttr = op.getQueueNumAttr()) {
    if (queueNumAttr.getInt() <= 0)
      return op.emitOpError("expects queue_num to be greater than 0");
  }
  if (auto channelGroupIdxAttr = op.getChannelGroupIdxAttr()) {
    APInt value = channelGroupIdxAttr.getValue();
    if (value.isNegative())
      return op.emitOpError("expects channel_group_idx to be non-negative");
    if (value.ugt(UINT32_MAX))
      return op.emitOpError("expects channel_group_idx to fit in uint32");
  }
  return success();
}

LogicalResult BuildAsyncSessionOp::verify() {
  if (failed(verifyAsyncSessionScratch(getOperation(), getScratch().getType())) ||
      failed(verifyAsyncSessionWorkspace(getOperation(), getWorkspace().getType())) ||
      failed(verifyAsyncSessionAttrs(*this))) {
    return failure();
  }
  return success();
}

static LogicalResult verifyAsyncTransferOp(Operation *op, Value dst, Value src) {
  Type dstElemTy = getElemTy(dst.getType());
  Type srcElemTy = getElemTy(src.getType());
  if (!dstElemTy || !srcElemTy)
    return op->emitOpError("expects src and dst to have element types");
  if (dstElemTy != srcElemTy)
    return op->emitOpError("expects src and dst to have the same element type");
  if (failed(verifyAsyncFlatContiguous1DGMViewLike(op, dst, "dst")) ||
      failed(verifyAsyncFlatContiguous1DGMViewLike(op, src, "src")))
    return failure();
  if (getShapeVec(dst.getType()) != getShapeVec(src.getType()))
    return op->emitOpError("expects src and dst to have the same static shape");
  return success();
}

static LogicalResult verifyCommTransferWithStaging(Operation *op, Value dst,
                                                   Value src, Value ping,
                                                   Value pong) {
  if (shouldBypassDecodedMemrefVerifier(op))
    return success();
  if (failed(verifyCommGlobalLike(op, dst, "dst")) ||
      failed(verifyCommGlobalLike(op, src, "src")) ||
      failed(verifyCommStagingTileLike(op, ping, "ping")) ||
      failed(verifyCommPingPongSameType(op, ping, pong, "ping", "pong")))
    return failure();
  if (getElemTy(dst.getType()) != getElemTy(src.getType()))
    return op->emitOpError("expects src and dst to have the same element type");
  if (getShapeVec(dst.getType()) != getShapeVec(src.getType()))
    return op->emitOpError("expects src and dst to have the same static shape");
  if (getElemTy(ping.getType()) != getElemTy(src.getType()))
    return op->emitOpError("expects staging tile element type to match src/dst");
  return success();
}

static LogicalResult verifyRootedCommTileTransfer(Operation *op, Value src,
                                                  OperandRange group,
                                                  uint32_t root, Value ping,
                                                  Value pong) {
  if (shouldBypassDecodedMemrefVerifier(op))
    return success();
  if (failed(verifyCommGlobalLike(op, src, "src")) ||
      failed(verifyCommStagingTileLike(op, ping, "ping")) ||
      failed(verifyCommPingPongSameType(op, ping, pong, "ping", "pong")) ||
      failed(verifyCommGlobalGroup(op, group, "group")))
    return failure();
  if (root >= static_cast<uint32_t>(group.size()))
    return op->emitOpError("expects root to index into group operands");
  if (getElemTy(ping.getType()) != getElemTy(src.getType()))
    return op->emitOpError("expects staging tile element type to match src");
  return success();
}

static LogicalResult verifyFrontendPipeTileAccess(Operation *op, Value pipeHandle,
                                                  int64_t split, Type tileTy) {
  if (!isInsideSectionOrAttributedKernel(op))
    return op->emitOpError(
        "must be inside pto.section.cube/vector or a kernel_kind function");
  if (failed(verifyPipeHandleProducer(op, pipeHandle)))
    return failure();
  if (failed(verifySplitAttr(op, split)))
    return failure();
  return verifyTensorEntryMatchesInternalPipeInit(op, pipeHandle, tileTy);
}

template <typename PongRangeT>
static void addOptionalPongWriteEffect(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects,
    PongRangeT pongRange) {
  if (auto it = pongRange.begin(); it != pongRange.end())
    addEffect(effects, &*it, MemoryEffects::Write::get());
}

LogicalResult TPutAsyncOp::verify() {
  return verifyAsyncTransferOp(getOperation(), getDst(), getSrc());
}

LogicalResult TGetAsyncOp::verify() {
  return verifyAsyncTransferOp(getOperation(), getDst(), getSrc());
}

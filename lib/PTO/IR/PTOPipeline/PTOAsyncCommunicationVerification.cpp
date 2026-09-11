// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

static bool getTensorLikeElementAndShape(Type ty, Type &elementType,
                                         ArrayRef<int64_t> &shape) {
  if (auto tvTy = dyn_cast<TensorViewType>(ty)) {
    elementType = tvTy.getElementType();
    shape = tvTy.getShape();
    return true;
  }
  return false;
}

static int8_t getInternalPipeEntrySplit(Operation *op) {
  if (auto alloc = dyn_cast<TAllocOp>(op))
    return alloc.getSplit();
  if (auto push = dyn_cast<TPushOp>(op))
    return push.getSplit();
  if (auto pop = dyn_cast<TPopOp>(op))
    return pop.getSplit();
  if (auto free = dyn_cast<TFreeOp>(op))
    return free.getSplit();
  return 0;
}

static LogicalResult verifyInternalPipeEntryShape(
    Operation *op, TensorViewType entryTy, Type slotElementType,
    ArrayRef<int64_t> slotShape) {
  if (slotElementType != entryTy.getElementType())
    return op->emitOpError(
        "expects pipe entry element type to match initialize_l2g2l_pipe gm_addr element type");
  if (slotShape.size() != static_cast<size_t>(entryTy.getRank()))
    return op->emitOpError(
        "expects pipe entry rank to match initialize_l2g2l_pipe gm_addr rank");
  for (auto [index, entryDim] : llvm::enumerate(entryTy.getShape())) {
    int64_t slotDim = slotShape[index];
    if (slotDim != ShapedType::kDynamic && entryDim != ShapedType::kDynamic &&
        slotDim != entryDim)
      return op->emitOpError()
             << "expects pipe entry dimension " << index
             << " to match initialize_l2g2l_pipe gm_addr dimension " << slotDim;
  }
  return success();
}

static LogicalResult verifyInternalPipeEntryBytes(
    Operation *op, TensorViewType entryTy, InitializeL2G2LPipeOp initOp) {
  auto elementCount = getStaticElementCount(entryTy.getShape());
  uint64_t elemBytes = getElemByteSize(entryTy.getElementType());
  if (!elementCount || elemBytes == 0)
    return success();
  uint64_t entryBytes = *elementCount * elemBytes;
  uint64_t slotBytes = static_cast<uint64_t>(initOp.getSlotSize());
  bool split = getInternalPipeEntrySplit(op) != 0;
  if (entryBytes == slotBytes || (split && entryBytes * mlir::pto::kValue2 == slotBytes))
      return success();
  return op->emitOpError()
         << "expects pipe entry byte size to match initialize_l2g2l_pipe slot_size"
         << (split ? " or half slot_size for split entries" : "")
         << " (got entry byte size = " << entryBytes
         << ", slot_size = " << initOp.getSlotSize() << ")";
}

static LogicalResult verifyTensorEntryMatchesInternalPipeInit(Operation *op,
                                                              Value pipeHandle,
                                                              Type entryTy) {
  auto entryViewTy = dyn_cast<TensorViewType>(entryTy);
  if (!entryViewTy) {
    return success();
  }

  auto initOp = pipeHandle.getDefiningOp<InitializeL2G2LPipeOp>();
  if (!initOp) {
    return op->emitOpError()
           << "expects !pto.tensor_view pipe entry to use a pipe produced by "
              "pto.initialize_l2g2l_pipe";
  }
  if (initOp.getLocalAddr()) {
    return op->emitOpError()
           << "expects !pto.tensor_view pipe entry to use global-only "
              "pto.initialize_l2g2l_pipe without local_addr";
  }

  Type slotElementType;
  ArrayRef<int64_t> slotShape;
  if (!getTensorLikeElementAndShape(initOp.getGmAddr().getType(),
slotElementType, slotShape)) {
    return op->emitOpError()
           << "expects !pto.tensor_view pipe entry to use "
              "pto.initialize_l2g2l_pipe gm_addr with tensor_view slot type";
  }

  if (failed(verifyInternalPipeEntryShape(op, entryViewTy, slotElementType,
                                          slotShape)))
    return failure();
  return verifyInternalPipeEntryBytes(op, entryViewTy, initOp);
}

static LogicalResult verifyAsyncSessionScratch(BuildAsyncSessionOp op) {
  Type scratchTy = op.getScratch().getType();
  if (!isa<pto::TileBufType>(scratchTy)) {
    return op.emitOpError("expects scratch to be tile_buf type");
  }

  auto scratchSpace = getPTOMemorySpaceEnum(scratchTy);
  if (!scratchSpace || *scratchSpace != pto::AddressSpace::VEC) {
    return op.emitOpError("expects scratch to be in vec address space");
  }

  auto scratchShape = getShapeVec(scratchTy);
  if (scratchShape.empty() || scratchShape.size() > mlir::pto::kValue2) {
      return op.emitOpError("expects scratch to be rank-1 or rank-2");
  }
  for (int64_t dim : scratchShape) {
    if (dim == ShapedType::kDynamic) {
      return op.emitOpError("expects scratch to have a static shape");
    }
  }

  auto scratchBytes = getStaticByteSize(scratchTy);
  if (!scratchBytes) {
    return op.emitOpError("expects scratch byte size to be statically known");
  }
  if (*scratchBytes < sizeof(uint64_t)) {
    return op.emitOpError("expects scratch to provide at least 8 bytes");
  }
  return success();
}

static LogicalResult verifyAsyncSessionAttrs(BuildAsyncSessionOp op) {
  if (auto attr = op.getSyncIdAttr()) {
      if (attr.getInt() < 0 || attr.getInt() > mlir::pto::kValue7)
          return op.emitOpError("expects sync_id in range [0, 7]");
  }
  if (auto attr = op.getBlockBytesAttr(); attr && attr.getInt() <= 0)
    return op.emitOpError("expects block_bytes to be greater than 0");
  if (auto attr = op.getCommBlockOffsetAttr(); attr && attr.getInt() < 0)
    return op.emitOpError("expects comm_block_offset to be non-negative");
  if (auto attr = op.getQueueNumAttr(); attr && attr.getInt() <= 0)
    return op.emitOpError("expects queue_num to be greater than 0");
  if (auto attr = op.getChannelGroupIdxAttr()) {
    llvm::APInt value = attr.getValue();
    if (value.isNegative())
      return op.emitOpError("expects channel_group_idx to be non-negative");
    if (value.ugt(UINT32_MAX))
      return op.emitOpError("expects channel_group_idx to fit in uint32");
  }
  return success();
}

LogicalResult BuildAsyncSessionOp::verify() {
  if (failed(verifyAsyncSessionScratch(*this)))
    return failure();
  auto workspaceTy = dyn_cast<pto::PtrType>(getWorkspace().getType());
  if (!workspaceTy) {
    return emitOpError("expects workspace to be !pto.ptr type");
  }
  Type workspaceElemTy = workspaceTy.getElementType();
  if (!isByteIntegerType(workspaceElemTy)) {
    return emitOpError("expects workspace element type to be an 8-bit integer");
  }

  return verifyAsyncSessionAttrs(*this);
}

static LogicalResult verifyAsyncTransferOp(Operation *op, Value dst, Value src) {
  Type dstElemTy = getElemTy(dst.getType());
  Type srcElemTy = getElemTy(src.getType());
  if (!dstElemTy || !srcElemTy) {
    return op->emitOpError("expects src and dst to have element types");
  }
  if (dstElemTy != srcElemTy) {
    return op->emitOpError("expects src and dst to have the same element type");
  }
  if (failed(verifyAsyncFlatContiguous1DGMViewLike(op, dst, "dst")) ||
      failed(verifyAsyncFlatContiguous1DGMViewLike(op, src, "src"))) {
    return failure();
  }
  if (getShapeVec(dst.getType()) != getShapeVec(src.getType())) {
    return op->emitOpError("expects src and dst to have the same static shape");
  }
  return success();
}

LogicalResult TPutAsyncOp::verify() {
  return verifyAsyncTransferOp(getOperation(), getDst(), getSrc());
}

LogicalResult TGetAsyncOp::verify() {
  return verifyAsyncTransferOp(getOperation(), getDst(), getSrc());
}

static LogicalResult verifyCommTransferOp(Operation *op, Value dst, Value src,
                                          Value ping, Value pong) {
  if (failed(verifyCommGlobalLike(
          op, dst, "dst", CommGlobalShapePolicy::AllowDynamicPartitionView)) ||
      failed(verifyCommGlobalLike(
          op, src, "src", CommGlobalShapePolicy::AllowDynamicPartitionView)) ||
      failed(verifyCommStagingTileLike(op, ping, "ping")) ||
      failed(verifyCommPingPongSameType(op, ping, pong, "ping", "pong")))
    return failure();
  if (getElemTy(dst.getType()) != getElemTy(src.getType()))
    return op->emitOpError(
        "expects src and dst to have the same element type");
  if (getShapeVec(dst.getType()) != getShapeVec(src.getType()))
    return op->emitOpError(
        "expects src and dst to have the same static/dynamic shape signature");
  if (getElemTy(ping.getType()) != getElemTy(src.getType()))
    return op->emitOpError(
        "expects staging tile element type to match src/dst");
  return success();
}

LogicalResult TPutOp::verify() {
  return verifyCommTransferOp(getOperation(), getDst(), getSrc(), getPing(),
                              getPong());
}

LogicalResult TGetOp::verify() {
  return verifyCommTransferOp(getOperation(), getDst(), getSrc(), getPing(),
                              getPong());
}

static LogicalResult verifyCommSignalI32(Operation *op, Value signal,
                                         Value value, StringRef valueName) {
  if (failed(verifyCommSignalLike(op, signal, "signal")))
    return failure();
  auto valueTy = dyn_cast<IntegerType>(value.getType());
  if (!valueTy || valueTy.getWidth() != mlir::pto::kValue32)
      return op->emitOpError() << "expects " << valueName << " to be i32";
  return success();
}

LogicalResult TNotifyOp::verify() {
  return verifyCommSignalI32(getOperation(), getSignal(), getValue(), "value");
}

LogicalResult TWaitOp::verify() {
  return verifyCommSignalI32(getOperation(), getSignal(), getCmpValue(),
                             "cmp_value");
}

LogicalResult TTestOp::verify() {
  return verifyCommSignalI32(getOperation(), getSignal(), getCmpValue(),
                             "cmp_value");
}

static LogicalResult verifySyncAllWorkspaceCapacity(Operation *op,
                                                    ArrayRef<int64_t> shape,
                                                    StringRef name) {
  if (shape.empty())
    return op->emitOpError() << "expects " << name << " to have rank >= 1";
  if (llvm::any_of(shape, [](int64_t dim) {
        return dim != ShapedType::kDynamic && dim <= 0;
      }))
    return op->emitOpError() << "expects " << name << " shape to be positive";
  if (llvm::is_contained(shape, ShapedType::kDynamic))
    return success();
  int64_t capacity = 1;
  for (int64_t dim : shape) {
    int64_t product = 0;
    if (llvm::MulOverflow(capacity, dim, product)) {
      capacity = std::numeric_limits<int64_t>::max();
      break;
    }
    capacity = product;
  }
  if (capacity < mlir::pto::kValue16)
      return op->emitOpError() << "expects " << name
                               << " to contain at least 16 i32 elements (64 bytes), but static capacity is "
                               << capacity;
  return success();
}

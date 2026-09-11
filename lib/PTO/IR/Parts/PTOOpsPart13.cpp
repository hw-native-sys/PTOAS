// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This implementation fragment is included by PTO.cpp and intentionally is
// not listed as a separate CMake translation unit.

static LogicalResult verifyInternalOddSplitSupport(Operation *op,
                                                   Value pipeHandle,
                                                   int64_t split,
                                                   bool producerSide) {
  if (!isOddSplit(split)) {
    return success();
  }

  bool isCubeSide = isInsideCubeKernelOrSection(op);
  bool isVectorSide = isInsideVectorKernelOrSection(op);
  bool isC2VSide = producerSide ? isCubeSide : isVectorSide;
  bool isV2CSide = producerSide ? isVectorSide : isCubeSide;
  int8_t directionMask = isC2VSide ? 1 : (isV2CSide ? 2 : 0);
  if (isV2CSide && getTargetArch(op) == PTOArch::A5) {
    return op->emitOpError(
        "supports odd V2C split modes (split = 3 or 4) only on a2/a3");
  }
  auto initOp = pipeHandle.getDefiningOp<InitializeL2G2LPipeOp>();
  Value consumerBuffer;
  if (initOp && directionMask != 0) {
      consumerBuffer = directionMask == mlir::pto::kValue2 && initOp.getDirMask() == mlir::pto::kValue3 ?
                           initOp.getPeerLocalAddr() :
                           initOp.getLocalAddr();
  }
  if (!initOp || directionMask == 0 ||
      (initOp.getDirMask() & directionMask) == 0 || !consumerBuffer) {
    return op->emitOpError(
        "supports odd split modes (split = 3 or 4) only for a "
        "pto.initialize_l2g2l_pipe whose dir_mask enables the operation "
        "direction and provides its local consumer buffer");
  }
  return success();
}

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

static LogicalResult verifySyncAllGmWorkspace(Operation *op, Value workspace,
                                              StringRef name) {
  Type ty = workspace.getType();
  Type elemType;
  SmallVector<int64_t, mlir::pto::kValue4> shape;
  if (auto ptrTy = dyn_cast<pto::PtrType>(ty)) {
    if (ptrTy.getMemorySpace().getAddressSpace() != pto::AddressSpace::GM) {
      return op->emitOpError() << "expects " << name
                               << " to be in GM address space";
    }
    elemType = ptrTy.getElementType();
  } else if (isa<pto::TensorViewType, pto::PartitionTensorViewType>(ty)) {
    elemType = getElemTy(ty);
    shape = getShapeVec(ty);
  } else {
    return op->emitOpError()
           << "expects " << name
           << " to be a GM ptr/tensor_view/partition_view";
  }

  auto elemTy = dyn_cast<IntegerType>(elemType);
  if (!elemTy || elemTy.getWidth() != mlir::pto::kValue32) {
      return op->emitOpError() << "expects " << name << " element type to be i32";
  }

  // A pointer does not carry capacity metadata. It is lowered as the fixed
  // 16 x i32 workspace required by PTO-ISA; allocation size remains a runtime
  // responsibility.
  if (isa<pto::PtrType>(ty)) {
    return success();
  }

  return verifySyncAllWorkspaceCapacity(op, shape, name);
}

LogicalResult SyncAllOp::verify() {
  bool hasGm = static_cast<bool>(getGmWorkspace());
  auto mode = getMode().getValue();
  if (mode == pto::SyncAllMode::Hard) {
    if (hasGm || getUsedCores()) {
      return emitOpError(
          "expects hard syncall to have no gm_workspace or used_cores");
    }
    return success();
  }

  if (!hasGm) {
    return emitOpError("expects soft syncall to provide gm_workspace");
  }
  if (failed(verifySyncAllGmWorkspace(getOperation(), getGmWorkspace(),
                                      "gm_workspace"))) {
    return failure();
  }

  return success();
}

static LogicalResult verifyRootedCommOp(Operation *op, Value data,
                                        StringRef dataName, Value ping,
                                        Value pong, ValueRange group,
                                        uint32_t root, bool exactGroupType) {
  if (failed(verifyCommGlobalLike(op, data, dataName)) ||
      failed(verifyCommStagingTileLike(op, ping, "ping")) ||
      failed(verifyCommPingPongSameType(op, ping, pong, "ping", "pong")) ||
      failed(verifyCommGlobalGroup(op, group, "group")))
    return failure();
  if (root >= static_cast<uint32_t>(group.size()))
    return op->emitOpError("expects root to index into group operands");
  Type dataType = data.getType();
  Type groupType = group.front().getType();
  bool groupMatches = exactGroupType
                          ? dataType == groupType
                          : getElemTy(dataType) == getElemTy(groupType);
  if (!groupMatches)
    return op->emitOpError()
           << "expects " << dataName
           << (exactGroupType ? " type" : " element type")
           << " to match group member type";
  if (getElemTy(ping.getType()) != getElemTy(dataType))
    return op->emitOpError()
           << "expects staging tile element type to match " << dataName;
  return success();
}

LogicalResult TBroadcastOp::verify() {
  return verifyRootedCommOp(getOperation(), getSrc(), "src", getPing(),
                            getPong(), getGroup(), getRoot(),
                            /*exactGroupType=*/true);
}

LogicalResult CommTGatherOp::verify() {
  return verifyRootedCommOp(getOperation(), getDst(), "dst", getPing(),
                            getPong(), getGroup(), getRoot(),
                            /*exactGroupType=*/false);
}

LogicalResult CommTScatterOp::verify() {
  return verifyRootedCommOp(getOperation(), getSrc(), "src", getPing(),
                            getPong(), getGroup(), getRoot(),
                            /*exactGroupType=*/false);
}

LogicalResult TReduceOp::verify() {
  if (failed(verifyCommGlobalLike(*this, getDst(), "dst")) ||
      failed(verifyCommStagingTileLike(*this, getAcc(), "acc")) ||
      failed(verifyCommStagingTileLike(*this, getRecvPing(), "recv_ping")) ||
      failed(verifyCommPingPongSameType(*this, getRecvPing(), getRecvPong(),
                                        "recv_ping", "recv_pong")) ||
      failed(verifyCommGlobalGroup(*this, getGroup(), "group"))) {
    return failure();
  }
  if (getRoot() >= static_cast<uint32_t>(getGroup().size())) {
    return emitOpError("expects root to index into group operands");
  }
  if (getElemTy(getDst().getType()) != getElemTy(getGroup().front().getType())) {
    return emitOpError("expects dst element type to match group member type");
  }
  if (getAcc().getType() != getRecvPing().getType()) {
    return emitOpError("expects acc and recv_ping to have identical types");
  }
  if (getElemTy(getAcc().getType()) != getElemTy(getDst().getType())) {
    return emitOpError("expects accumulator/receive tiles to match dst element type");
  }
  return success();
}

LogicalResult AicInitializePipeOp::verify() {
  if (failed(verifyFrontendInitCommon(*this, FunctionKernelKind::Cube, "cube"))) {
    return failure();
  }

  auto accPushEpilogue = getAccPushEpilogueAttr();
  if (!accPushEpilogue) {
    return success();
  }

  auto peerConsumerInitOr = lookupFixpipePeerConsumerInit(*this);
  if (failed(peerConsumerInitOr)) {
    return emitOpError()
           << "expects peer consumer function to contain a matching "
              "aiv_initialize_pipe with the same consumer buffer contract";
  }

  Operation *peerConsumerInit = *peerConsumerInitOr;
  auto peerAccPushEpilogue = getAccPushEpilogueFromInitOp(peerConsumerInit);
  if (!peerAccPushEpilogue) {
    return emitOpError()
           << "expects peer consumer pipe init to also have "
              "'acc_push_epilogue' for fixpipe contract consistency";
  }

  if (peerAccPushEpilogue.getLayout() != accPushEpilogue.getLayout()) {
    return emitOpError()
           << "expects acc_push_epilogue.layout to match peer consumer "
           << "(producer has " << stringifyFixpipeLayout(accPushEpilogue.getLayout())
           << ", consumer has " << stringifyFixpipeLayout(peerAccPushEpilogue.getLayout())
           << ")";
  }
  if (peerAccPushEpilogue.getQuant() != accPushEpilogue.getQuant()) {
    return emitOpError()
           << "expects acc_push_epilogue.quant to match peer consumer "
           << "(producer has " << stringifyFixpipeQuant(accPushEpilogue.getQuant())
           << ", consumer has " << stringifyFixpipeQuant(peerAccPushEpilogue.getQuant())
           << ")";
  }
  if (peerAccPushEpilogue.getRelu() != accPushEpilogue.getRelu()) {
    return emitOpError()
           << "expects acc_push_epilogue.relu to match peer consumer "
           << "(producer has " << stringifyFixpipeRelu(accPushEpilogue.getRelu())
           << ", consumer has " << stringifyFixpipeRelu(peerAccPushEpilogue.getRelu())
           << ")";
  }

  return success();
}

static FailureOr<func::FuncOp> findFixpipePeerProducer(
    AivInitializePipeOp op, func::FuncOp consumer, StringRef &bufferName) {
  if (!op.getC2vConsumerBuf()) {
    op.emitOpError(
        "expects fixpipe consumer pipe to have 'c2v_consumer_buf'");
    return failure();
  }
  Operation *definition = op.getC2vConsumerBuf().getDefiningOp();
  if (dyn_cast_or_null<ImportReservedBufferOp>(definition)) {
    op.emitOpError(
        "expects consumer-side fixpipe pipe to use reserve_buffer, not import_reserved_buffer");
    return failure();
  }
  auto reserve = dyn_cast_or_null<ReserveBufferOp>(definition);
  if (!reserve) {
    op.emitOpError(
        "expects fixpipe pipe 'c2v_consumer_buf' to trace to reserve_buffer or "
        "import_reserved_buffer for peer contract verification");
    return failure();
  }
  bufferName = reserve.getName();
  ModuleOp module = consumer->getParentOfType<ModuleOp>();
  if (!module) {
    op.emitOpError(
        "must be nested under a module for fixpipe contract verification");
    return failure();
  }
  SmallVector<ImportReservedBufferOp> imports;
  module.walk([&](ImportReservedBufferOp candidate) {
    auto peer = lookupPeerFuncAcrossContainer(candidate,
                                               candidate.getPeerFuncAttr());
    if (candidate.getName() == bufferName && peer == consumer)
      imports.push_back(candidate);
  });
  if (imports.empty()) {
    op.emitOpError() << "cannot find peer import_reserved_buffer for consumer buffer '"
                     << bufferName << "'";
    return failure();
  }
  if (imports.size() > 1) {
    op.emitOpError() << "finds multiple peer import_reserved_buffer ops for consumer buffer '"
                     << bufferName << "'";
    return failure();
  }
  return imports.front()->getParentOfType<func::FuncOp>();
}

static FailureOr<uint32_t> getFixpipePeerProducerId(
    AivInitializePipeOp op, Operation *init, StringRef bufferName,
    func::FuncOp consumer) {
  if (auto frontend = dyn_cast<AicInitializePipeOp>(init)) {
    if (!frontend.getC2vConsumerBuf()) {
      op.emitOpError(
          "expects peer producer aic_initialize_pipe to have 'c2v_consumer_buf'");
      return failure();
    }
    auto imported = dyn_cast_or_null<ImportReservedBufferOp>(
        frontend.getC2vConsumerBuf().getDefiningOp());
    if (!imported) {
      op.emitOpError(
          "expects peer producer aic_initialize_pipe to use import_reserved_buffer for c2v_consumer_buf");
      return failure();
    }
    auto peer = lookupPeerFuncAcrossContainer(imported, imported.getPeerFuncAttr());
    if (imported.getName() != bufferName || peer != consumer) {
      op.emitOpError()
          << "cannot find matching producer aic_initialize_pipe for buffer '"
          << bufferName << "' in peer function";
      return failure();
    }
    return frontend.getId();
  }
  if (isa<InitializeL2LPipeOp, InitializeL2G2LPipeOp>(init)) {
    auto id = init->getAttrOfType<IntegerAttr>(kFrontendPipeIdAttrName);
    if (!id) {
      op.emitOpError()
          << "expects lowered peer producer fixpipe pipe to retain "
          << kFrontendPipeIdAttrName;
      return failure();
    }
    return static_cast<uint32_t>(id.getInt());
  }
  op.emitOpError(
      "expects peer producer fixpipe contract to resolve to frontend or lowered aic_initialize_pipe");
  return failure();
}

static LogicalResult verifyAivPeerEpilogue(
    AivInitializePipeOp op, AccPushEpilogueAttr consumer,
    AccPushEpilogueAttr producer) {
  if (!producer)
    return op.emitOpError(
        "expects peer producer pipe init to also have 'acc_push_epilogue' for fixpipe contract consistency");
  if (producer.getLayout() != consumer.getLayout())
    return op.emitOpError()
           << "expects acc_push_epilogue.layout to match peer producer "
           << "(consumer has " << stringifyFixpipeLayout(consumer.getLayout())
           << ", producer has " << stringifyFixpipeLayout(producer.getLayout())
           << ")";
  if (producer.getQuant() != consumer.getQuant())
    return op.emitOpError()
           << "expects acc_push_epilogue.quant to match peer producer "
           << "(consumer has " << stringifyFixpipeQuant(consumer.getQuant())
           << ", producer has " << stringifyFixpipeQuant(producer.getQuant())
           << ")";
  if (producer.getRelu() != consumer.getRelu())
    return op.emitOpError()
           << "expects acc_push_epilogue.relu to match peer producer "
           << "(consumer has " << stringifyFixpipeRelu(consumer.getRelu())
           << ", producer has " << stringifyFixpipeRelu(producer.getRelu())
           << ")";
  return success();
}

static FailureOr<pto::TileBufType> getAivFixpipeConsumerTile(
    AivInitializePipeOp op, func::FuncOp consumer,
    AccPushEpilogueAttr epilogue) {
  SmallVector<TPopFromAicOp> pops;
  consumer.walk([&](TPopFromAicOp pop) {
    if (pop.getId() == op.getId())
      pops.push_back(pop);
  });
  if (pops.empty()) {
    op.emitOpError() << "expects at least one tpop_from_aic for fixpipe pipe id = "
                     << op.getId() << " to resolve the consumer entry type";
    return failure();
  }
  Type type = pops.front().getTile().getType();
  if (llvm::any_of(llvm::drop_begin(pops),
                   [&](TPopFromAicOp pop) { return pop.getTile().getType() != type; })) {
    op.emitOpError() << "expects all tpop_from_aic results for fixpipe pipe id = "
                     << op.getId() << " to use the same tile type";
    return failure();
  }
  auto tile = dyn_cast<pto::TileBufType>(type);
  if (!tile) {
    op.emitOpError("expects fixpipe consumer tpop result to be !pto.tile_buf");
    return failure();
  }
  if (!matchesFixpipeConsumerElementType(epilogue.getQuant(),
                                          tile.getElementType())) {
    op.emitOpError()
        << "expects consumer element type to match acc_push_epilogue.quant "
        << stringifyFixpipeQuant(epilogue.getQuant());
    return failure();
  }
  if (!matchesFixpipeConsumerLayout(epilogue.getLayout(), tile)) {
    op.emitOpError()
        << "expects consumer tile layout to match acc_push_epilogue.layout "
        << stringifyFixpipeLayout(epilogue.getLayout());
    return failure();
  }
  return tile;
}

static std::optional<uint32_t> getPipeInitSlotSize(Operation *init) {
  if (auto value = dyn_cast_or_null<AicInitializePipeOp>(init))
    return value.getSlotSize();
  if (auto value = dyn_cast_or_null<AivInitializePipeOp>(init))
    return value.getSlotSize();
  if (auto value = dyn_cast_or_null<InitializeL2LPipeOp>(init))
    return value.getSlotSize();
  if (auto value = dyn_cast_or_null<InitializeL2G2LPipeOp>(init))
    return value.getSlotSize();
  return std::nullopt;
}

static LogicalResult verifyAivFixpipeCapacity(AivInitializePipeOp op,
                                              Operation *producer,
                                              pto::TileBufType tile) {
  auto required = getStaticTileByteSize(tile);
  if (!required)
    return success();
  if (static_cast<uint64_t>(op.getSlotSize()) < *required)
    return op.emitOpError()
           << "expects consumer-side fixpipe slot_size to be at least "
           << *required
           << " bytes for the resolved post-fixpipe consumer entry";
  auto producerSize = getPipeInitSlotSize(producer);
  if (producerSize && static_cast<uint64_t>(*producerSize) < *required)
    return op.emitOpError()
           << "expects peer producer fixpipe slot_size to be at least "
           << *required
           << " bytes for the resolved post-fixpipe consumer entry";
  return success();
}

static LogicalResult verifyAivProducerPushTypes(
    AivInitializePipeOp op, func::FuncOp producer, uint32_t producerId,
    FixpipeQuant quant, Type consumerElem) {
  bool mismatch = false;
  producer.walk([&](TPushToAivOp push) {
    auto src = dyn_cast<pto::TileBufType>(push.getTile().getType());
    if (push.getId() == producerId && src &&
        !matchesFixpipeProducerAndConsumerTypes(quant, src.getElementType(),
                                                consumerElem)) {
      mismatch = true;
      op.emitOpError()
          << "expects producer source element type and consumer tpop result "
             "element type to satisfy acc_push_epilogue.quant "
          << stringifyFixpipeQuant(quant);
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return success(!mismatch);
}

static LogicalResult verifyAivFixpipe(AivInitializePipeOp op,
                                      AccPushEpilogueAttr epilogue) {
  auto consumer = op->getParentOfType<func::FuncOp>();
  if (!consumer)
    return op.emitOpError("must be nested under a func.func");
  StringRef bufferName;
  auto producer = findFixpipePeerProducer(op, consumer, bufferName);
  if (failed(producer) || !*producer)
    return failure();
  auto producerInit = lookupFixpipePeerProducerInit(
      op, *producer, bufferName, consumer);
  if (failed(producerInit))
    return failure();
  auto producerId =
      getFixpipePeerProducerId(op, *producerInit, bufferName, consumer);
  if (failed(producerId) ||
      failed(verifyAivPeerEpilogue(
          op, epilogue, getAccPushEpilogueFromInitOp(*producerInit))))
    return failure();
  auto tile = getAivFixpipeConsumerTile(op, consumer, epilogue);
  if (failed(tile) ||
      failed(verifyAivFixpipeCapacity(op, *producerInit, *tile)))
    return failure();
  return verifyAivProducerPushTypes(op, *producer, *producerId,
                                    epilogue.getQuant(), tile->getElementType());
}
LogicalResult AivInitializePipeOp::verify() {
  if (failed(
          verifyFrontendInitCommon(*this, FunctionKernelKind::Vector, "vector")))
    return failure();
  auto epilogue = getAccPushEpilogueAttr();
  return epilogue ? verifyAivFixpipe(*this, epilogue) : success();
}

static LogicalResult verifyFrontendDataCommon(
    Operation *op, FunctionKernelKind kind, StringRef kernelName, uint32_t id,
    uint32_t split, bool expectC2V) {
  if (failed(verifyNoUnpublishedFixpipeFrontendAttrs(op)) ||
      failed(verifyFrontendSplitOp(op, kind, kernelName, id, split,
                                   expectC2V)) ||
      failed(verifyFrontendDataOpDirection(op, id, expectC2V)))
    return failure();
  return success();
}

static LogicalResult verifyFrontendFixpipeSplitZero(
    Operation *op, uint32_t id, uint32_t split, bool lookForAicInit,
    StringRef operationName) {
  auto funcOp = op->getParentOfType<func::FuncOp>();
  if (!funcOp)
    return success();
  auto init = lookupFrontendInitOpById(op, funcOp, id);
  if (failed(init))
    return success();
  bool isFixpipe = false;
  if (lookForAicInit) {
    auto initOp = dyn_cast<AicInitializePipeOp>(*init);
    isFixpipe = initOp && static_cast<bool>(initOp.getAccPushEpilogueAttr());
  } else {
    auto initOp = dyn_cast<AivInitializePipeOp>(*init);
    isFixpipe = initOp && static_cast<bool>(initOp.getAccPushEpilogueAttr());
  }
  if (isFixpipe && split != 0)
    return op->emitOpError()
           << "expects fixpipe " << operationName << " to have split = 0";
  return success();
}

LogicalResult TAllocToAivOp::verify() {
  if (failed(verifyFrontendDataCommon(getOperation(), FunctionKernelKind::Cube,
                                      "cube", getId(), getSplit(), true)) ||
      failed(verifyFrontendFixpipeSplitZero(
          getOperation(), getId(), getSplit(), /*lookForAicInit=*/true,
          "TALLOC")))
    return failure();
  if (failed(verifyOddSplitTileEntry(getOperation(), getSplit(),
                                     getEntry().getType())))
    return failure();
  return verifyFrontendTensorEntryMatchesInit(getOperation(), getId(),
                                              getEntry().getType());
}

LogicalResult TAllocToAicOp::verify() {
  if (failed(verifyFrontendDataCommon(
          getOperation(), FunctionKernelKind::Vector, "vector", getId(),
          getSplit(), false)) ||
      failed(verifyOddSplitTileEntry(getOperation(), getSplit(),
                                     getEntry().getType())))
    return failure();
  return verifyFrontendTensorEntryMatchesInit(getOperation(), getId(),
                                              getEntry().getType());
}

static LogicalResult verifyTPushToAivBase(TPushToAivOp op) {
  if (failed(verifyNoUnpublishedFixpipeFrontendAttrs(op)) ||
      failed(verifyFrontendSplitOp(op, FunctionKernelKind::Cube, "cube",
                                   op.getId(), op.getSplit(), true)) ||
      failed(verifyFrontendDataOpDirection(op, op.getId(), true)) ||
      failed(verifyFrontendTensorEntryMatchesInit(
          op, op.getId(), op.getTile().getType())) ||
      failed(verifyOddSplitTileEntry(op, op.getSplit(),
                                     op.getTile().getType())) ||
      failed(verifyFullTileSplitParity(op, op.getSplit(),
                                       op.getTile().getType())))
    return failure();
  return success();
}

static bool hasPrecedingFixpipeQuantConfig(TPushToAivOp op,
                                           bool scalarQuant) {
  for (Operation &candidate : op->getBlock()->getOperations()) {
    if (&candidate == op.getOperation())
      break;
    if (scalarQuant) {
      auto setQuant = dyn_cast<SetQuantScalarOp>(&candidate);
      if (setQuant && setQuant.getId() == op.getId())
        return true;
    } else {
      auto setQuant = dyn_cast<SetQuantVectorOp>(&candidate);
      if (setQuant && setQuant.getId() == op.getId())
        return true;
    }
  }
  return false;
}

static LogicalResult verifyTPushToAivFixpipe(
    TPushToAivOp op, AccPushEpilogueAttr epilogue) {
  auto tile = dyn_cast<pto::TileBufType>(op.getTile().getType());
  if (!tile)
    return op.emitOpError(
        "expects fixpipe TPUSH source tile to be a tile type");
  auto space = getPTOMemorySpaceEnum(tile);
  if (!space || *space != pto::AddressSpace::ACC)
    return op.emitOpError("expects fixpipe TPUSH source tile to use loc=acc");
  if (op.getSplit() != 0)
    return op.emitOpError("expects fixpipe TPUSH to have split = 0");
  auto quant = epilogue.getQuant();
  if (!matchesFixpipeProducerElementType(quant, tile.getElementType()))
    return op.emitOpError()
           << "expects fixpipe TPUSH source element type to match "
           << "acc_push_epilogue.quant mode requirements";
  bool scalar = isScalarFixpipeQuant(quant);
  bool vector = isVectorFixpipeQuant(quant);
  if (!scalar && !vector)
    return success();
  if (hasPrecedingFixpipeQuantConfig(op, scalar))
    return success();
  if (scalar)
    return op.emitOpError()
           << "expects a preceding pto.set_quant_scalar with id = "
           << op.getId() << " in the same block";
  return op.emitOpError()
         << "expects a preceding pto.set_quant_vector with id = " << op.getId()
         << " in the same block";
}

static LogicalResult verifyTPushToAivFixpipeIfPresent(TPushToAivOp op) {
  auto func = op->getParentOfType<func::FuncOp>();
  if (!func)
    return op.emitOpError("must be nested under a func.func");
  auto init = lookupFrontendInitOpById(op, func, op.getId());
  if (failed(init))
    return failure();
  auto aicInit = dyn_cast<AicInitializePipeOp>(*init);
  if (!aicInit || !aicInit.getAccPushEpilogueAttr())
    return success();
  return verifyTPushToAivFixpipe(op, aicInit.getAccPushEpilogueAttr());
}

LogicalResult TPushToAivOp::verify() {
  if (failed(verifyTPushToAivBase(*this)))
    return failure();
  return verifyTPushToAivFixpipeIfPresent(*this);
}
LogicalResult TPushToAicOp::verify() {
  if (failed(verifyNoUnpublishedFixpipeFrontendAttrs(getOperation()))) {
    return failure();
  }
  if (failed(verifyFrontendSplitOp(getOperation(), FunctionKernelKind::Vector,
                                   "vector", getId(), getSplit(),
                                   /*expectC2V=*/false))) {
    return failure();
  }
  if (failed(verifyFrontendDataOpDirection(getOperation(), getId(),
                                           /*expectC2V=*/false))) {
    return failure();
  }
  if (failed(verifyFrontendTensorEntryMatchesInit(getOperation(), getId(),
                                                  getTile().getType()))) {
    return failure();
  }
  if (failed(verifyOddSplitTileEntry(getOperation(), getSplit(),
                                     getTile().getType()))) {
    return failure();
  }
  return verifyAivSubblockIdOperand(getOperation(), getAivSubblockid(),
                                    getSplit(), getTile().getType());
}

LogicalResult TPopFromAicOp::verify() {
  if (failed(verifyNoUnpublishedFixpipeFrontendAttrs(getOperation()))) {
    return failure();
  }
  if (failed(verifyFrontendPopOp(*this, FunctionKernelKind::Vector, "vector",
                                 /*expectC2V=*/true))) {
    return failure();
  }
  if (failed(verifyAivSubblockIdOperand(getOperation(), getAivSubblockid(),
                                        getSplit(), getTile().getType()))) {
    return failure();
  }
  return verifyFixpipeConsumerType(getOperation(), getId(), getTile().getType());
}

LogicalResult TPopFromAivOp::verify() {
  if (failed(verifyNoUnpublishedFixpipeFrontendAttrs(getOperation()))) {
    return failure();
  }
  return verifyFrontendPopOp(*this, FunctionKernelKind::Cube, "cube",
                             /*expectC2V=*/false);
}

static LogicalResult verifyFrontendFreeOp(
    Operation *op, FunctionKernelKind kind, StringRef kernelName, uint32_t id,
    uint32_t split, bool expectC2V, Value entry) {
  if (failed(verifyFrontendDataCommon(op, kind, kernelName, id, split,
                                      expectC2V)) ||
      failed(verifyFrontendFixpipeSplitZero(
          op, id, split, /*lookForAicInit=*/false, "TFREE")))
    return failure();
  if (!entry)
    return success();
  if (failed(verifyOddSplitTileEntry(op, split, entry.getType())))
    return failure();
  return verifyFrontendTensorEntryMatchesInit(op, id, entry.getType());
}

LogicalResult TFreeFromAicOp::verify() {
  return verifyFrontendFreeOp(getOperation(), FunctionKernelKind::Vector,
                              "vector", getId(), getSplit(), true, getEntry());
}

LogicalResult TFreeFromAivOp::verify() {
  return verifyFrontendFreeOp(getOperation(), FunctionKernelKind::Cube,
                              "cube", getId(), getSplit(), false, getEntry());
}

static FailureOr<AccPushEpilogueAttr> getSetQuantEpilogue(Operation *op,
                                                         uint32_t id) {
  if (failed(verifyNoUnpublishedFixpipeFrontendAttrs(op)))
    return failure();
  auto funcOp = op->getParentOfType<func::FuncOp>();
  if (!funcOp) {
    op->emitOpError("must be nested under a func.func");
    return failure();
  }
  auto init = lookupFrontendOrLoweredInitOpById(op, funcOp, id);
  if (failed(init))
    return failure();
  AccPushEpilogueAttr epilogue;
  if (auto value = dyn_cast<AicInitializePipeOp>(*init)) {
    epilogue = value.getAccPushEpilogueAttr();
  } else if (auto value = dyn_cast<InitializeL2LPipeOp>(*init)) {
    epilogue = value.getAccPushEpilogueAttr();
  } else if (auto value = dyn_cast<InitializeL2G2LPipeOp>(*init)) {
    epilogue = value.getAccPushEpilogueAttr();
  } else {
    op->emitOpError() << "expects 'id' = " << id
                      << " to reference an aic_initialize_pipe or lowered producer pipe";
    return failure();
  }
  if (!epilogue) {
    op->emitOpError() << "expects 'id' = " << id
                      << " to reference a fixpipe pipe (with acc_push_epilogue)";
    return failure();
  }
  return epilogue;
}

LogicalResult SetQuantScalarOp::verify() {
  auto epilogue = getSetQuantEpilogue(getOperation(), getId());
  if (failed(epilogue))
    return failure();
  if (!isScalarFixpipeQuant(epilogue->getQuant()))
    return emitOpError()
           << "expects 'id' = " << getId()
           << " to reference a pipe with scalar quantization mode, but found non-scalar mode";
  if (!getScale().getType().isF32())
    return emitOpError("expects 'scale' to be f32");
  return success();
}

LogicalResult SetQuantVectorOp::verify() {
  auto epilogue = getSetQuantEpilogue(getOperation(), getId());
  if (failed(epilogue))
    return failure();
  if (!isVectorFixpipeQuant(epilogue->getQuant()))
    return emitOpError()
           << "expects 'id' = " << getId()
           << " to reference a pipe with vector quantization mode, but found non-vector mode";
  Type scalingTy = getScalingTile().getType();
  if (!isa<pto::TileBufType>(scalingTy)) {
    return emitOpError("expects 'scaling_tile' to be a tile type");
  }
  auto scalingSpace = getPTOMemorySpaceEnum(scalingTy);
  if (!scalingSpace || *scalingSpace != pto::AddressSpace::SCALING) {
    return emitOpError("expects 'scaling_tile' to use loc=scaling");
  }
  Type scalingElemTy = getElemTy(scalingTy);
  PTOArch arch = getTargetArch(getOperation());
  if (!isFixpipeQuantPayloadElemType(scalingElemTy, arch)) {
    if (arch == PTOArch::A3) {
      return emitOpError(
          "expects 'scaling_tile' element type to be packed i64/ui64/si64 on A3");
    }
    return emitOpError(
        "expects 'scaling_tile' element type to be packed i64/ui64/si64 on A5");
  }

  return success();
}

static LogicalResult verifyPeerLocalAddrForDirMask(Operation *op,
                                                   uint32_t dirMask,
                                                   Value peerLocalAddr) {
    if (dirMask == mlir::pto::kValue3 && !peerLocalAddr)
        return op->emitOpError("expects 'peer_local_addr' when dir_mask is 3");
    if (dirMask != mlir::pto::kValue3 && peerLocalAddr)
        return op->emitOpError("'peer_local_addr' is only allowed when dir_mask is 3");
    return success();
}

LogicalResult InitializeL2G2LPipeOp::verify() {
  if (failed(verifyPipeShape(getOperation(), getDirMask(), getSlotSize(),
                             getSlotNum(),
                             getFlagBaseAttr()
                                 ? std::optional<int32_t>(getFlagBaseAttr().getInt())
                                 : std::nullopt))) {
    return failure();
  }

  if (!getLocalAddr()) {
    if (getPeerLocalAddr()) {
      return emitOpError("'peer_local_addr' requires 'local_addr'");
    }
    if (getLocalSlotNumAttr()) {
      return emitOpError(
          "'local_slot_num' is only allowed when 'local_addr' is present");
    }
    return success();
  }

  if (auto localSlotNumAttr = getLocalSlotNumAttr()) {
    int32_t localSlotNum = localSlotNumAttr.getInt();
    if (localSlotNum <= 0) {
      return emitOpError("expects 'local_slot_num' to be greater than 0");
    }
    if (static_cast<uint32_t>(localSlotNum) > getSlotNum()) {
      return emitOpError(
          "expects 'local_slot_num' to be less than or equal to slot_num");
    }
  }

  return verifyPeerLocalAddrForDirMask(getOperation(), getDirMask(),
                                       getPeerLocalAddr());
}

LogicalResult InitializeL2LPipeOp::verify() {
  if (failed(verifyPipeShape(getOperation(), getDirMask(), getSlotSize(),
                             getSlotNum(),
                             getFlagBaseAttr()
                                 ? std::optional<int32_t>(getFlagBaseAttr().getInt())
                                 : std::nullopt))) {
    return failure();
  }

  return verifyPeerLocalAddrForDirMask(getOperation(), getDirMask(),
                                       getPeerLocalAddr());
}

static LogicalResult verifyInternalPipeBase(Operation *op, Value pipeHandle,
                                            uint32_t split,
                                            bool producerSide) {
  if (!isInsideSectionOrAttributedKernel(op))
    return op->emitOpError(
        "must be inside pto.section.cube/vector or a kernel_kind function");
  if (failed(verifyPipeHandleProducer(op, pipeHandle)) ||
      failed(verifySplitAttr(op, split)) ||
      failed(verifyInternalOddSplitSupport(op, pipeHandle, split,
                                           producerSide)))
    return failure();
  return success();
}

static LogicalResult verifyInternalTileOp(
    Operation *op, Value pipeHandle, uint32_t split, bool producerSide,
    Value tile, Value aivSubblockId, pto::PIPE pipe, StringRef pipeError) {
  if (failed(verifyInternalPipeBase(op, pipeHandle, split, producerSide)) ||
      failed(verifyOddSplitTileEntry(op, split, tile.getType())))
    return failure();
  if (isInsideCubeKernelOrSection(op) &&
      failed(verifyFullTileSplitParity(op, split, tile.getType())))
    return failure();
  if (failed(verifyAivSubblockIdOperand(op, aivSubblockId, split,
                                        tile.getType())) ||
      failed(verifyTensorEntryMatchesInternalPipeInit(
          op, pipeHandle, tile.getType())))
    return failure();
  if (!isa<TensorViewType>(tile.getType()) &&
      pipe == pto::PIPE::PIPE_UNASSIGNED)
    return op->emitOpError(pipeError);
  return success();
}

LogicalResult TPushOp::verify() {
  return verifyInternalTileOp(
      getOperation(), getPipeHandle(), getSplit(), /*producerSide=*/true,
      getTile(), getAivSubblockid(), getPipe(),
      "tile type must map to a supported producer pipe");
}

LogicalResult TAllocOp::verify() {
  if (failed(verifyInternalPipeBase(getOperation(), getPipeHandle(), getSplit(),
                                    /*producerSide=*/true)) ||
      failed(verifyOddSplitTileEntry(getOperation(), getSplit(),
                                     getEntry().getType())) ||
      failed(verifyTensorEntryMatchesInternalPipeInit(
          getOperation(), getPipeHandle(), getEntry().getType())))
    return failure();
  return success();
}

LogicalResult TPopOp::verify() {
  return verifyInternalTileOp(
      getOperation(), getPipeHandle(), getSplit(), /*producerSide=*/false,
      getTile(), getAivSubblockid(), getPipe(),
      "tile type and target arch must map to a supported consumer pipe");
}

LogicalResult TFreeOp::verify() {
  if (failed(verifyInternalPipeBase(getOperation(), getPipeHandle(), getSplit(),
                                    /*producerSide=*/false)))
    return failure();
  if (getEntry() &&
      failed(verifyOddSplitTileEntry(getOperation(), getSplit(),
                                     getEntry().getType()))) {
    return failure();
  }
  if (getEntry() &&
      failed(verifyTensorEntryMatchesInternalPipeInit(
          getOperation(), getPipeHandle(), getEntry().getType()))) {
    return failure();
  }
  return success();
}

struct TFreeParseState {
  OpAsmParser::UnresolvedOperand first;
  OpAsmParser::UnresolvedOperand pipe;
  Type firstTy;
  Type pipeTy;
  bool hasEntry = false;
};

static ParseResult parseTFreeOperands(OpAsmParser &parser,
                                      TFreeParseState &state) {
  if (parser.parseLParen() || parser.parseOperand(state.first))
    return failure();
  state.hasEntry = succeeded(parser.parseOptionalComma());
  if (!state.hasEntry) {
    if (parser.parseColonType(state.pipeTy) || parser.parseRParen())
      return failure();
    state.pipe = state.first;
    return success();
  }
  if (parser.parseOperand(state.pipe) || parser.parseColonType(state.firstTy) ||
      parser.parseComma() || parser.parseType(state.pipeTy) ||
      parser.parseRParen())
    return failure();
  return success();
}

static ParseResult parseTFreeAttributes(OpAsmParser &parser,
                                        NamedAttrList &attrs) {
  if (parser.parseLBrace() || parser.parseKeyword("split") ||
      parser.parseEqual())
    return failure();
  IntegerAttr splitAttr;
  if (parser.parseAttribute(splitAttr, parser.getBuilder().getI8Type(),
                            "split", attrs) ||
      parser.parseRBrace() || parser.parseOptionalAttrDict(attrs))
    return failure();
  return success();
}

static ParseResult resolveTFreeOperands(OpAsmParser &parser,
                                        OperationState &result,
                                        const TFreeParseState &state) {
  if (state.hasEntry &&
      parser.resolveOperand(state.first, state.firstTy, result.operands))
    return failure();
  return parser.resolveOperand(state.pipe, state.pipeTy, result.operands);
}

ParseResult TFreeOp::parse(OpAsmParser &parser, OperationState &result) {
  TFreeParseState state;
  NamedAttrList attrs;
  if (failed(parseTFreeOperands(parser, state)) ||
      failed(parseTFreeAttributes(parser, attrs))) {
    return failure();
  }
  result.addAttributes(attrs);
  return resolveTFreeOperands(parser, result, state);
}

void TFreeOp::print(OpAsmPrinter &p) {
  p << "(";
  if (getEntry()) {
    p << getEntry() << ", " << getPipeHandle() << " : "
      << getEntry().getType() << ", " << getPipeHandle().getType();
  } else {
    p << getPipeHandle() << " : " << getPipeHandle().getType();
  }
  p << ") {split = " << static_cast<int32_t>(getSplit()) << "}";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"split"});
}

static func::FuncOp getParentFunc(Operation *op) {
  return op ? op->getParentOfType<func::FuncOp>() : func::FuncOp();
}

static constexpr int64_t kSimtKeepResumeSlotLimit = 123;

static Operation *getFirstNonConstantLikeOp(Block *block) {
  if (!block) {
    return nullptr;
  }
  for (Operation &op : *block) {
    if (!op.hasTrait<OpTrait::ConstantLike>()) {
      return &op;
    }
  }
  return nullptr;
}

static bool isOpInRange(Operation *op, Operation *first, Operation *last) {
  for (Operation *cur = first; cur; cur = cur->getNextNode()) {
    if (cur == op) {
      return true;
    }
    if (cur == last) {
      return false;
    }
  }
  return false;
}

static std::optional<unsigned> getSimtKeepResumeRegisterCount(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
      if (intType.getWidth() <= mlir::pto::kValue32) {
          return 1;
      }
      if (intType.getWidth() == mlir::pto::kValue64) {
          return mlir::pto::kValue2;
      }
    return std::nullopt;
  }
  if (type.isF16() || type.isBF16() || type.isF32()) {
    return 1;
  }
  return std::nullopt;
}

static Type getSimtKeepResumeValueType(KeepOp op) {
  return op.getPayload().getType();
}

static Type getSimtKeepResumeValueType(ResumeOp op) {
  return op.getResult().getType();
}

template <typename OpT>
static LogicalResult verifySimtKeepResumeSlotRange(OpT op) {
  std::optional<unsigned> registerCount =
      getSimtKeepResumeRegisterCount(getSimtKeepResumeValueType(op));
  if (!registerCount) {
    return success();
  }
  int64_t slot = op.getSlot();
  if (slot < 0 || slot >= kSimtKeepResumeSlotLimit) {
    return op.emitOpError()
           << "requires slot in range [0, "
           << (kSimtKeepResumeSlotLimit - 1) << "]";
  }
  if (*registerCount == mlir::pto::kValue2) {
      if ((slot % mlir::pto::kValue2) != 0) {
          return op.emitOpError() << "requires an even slot for 64-bit keep/resume values";
      }
      if (slot + 1 >= kSimtKeepResumeSlotLimit) {
          return op.emitOpError() << "requires slot in range [0, " << (kSimtKeepResumeSlotLimit - mlir::pto::kValue2)
                                  << "] for 64-bit keep/resume values";
      }
  }
  return success();
}

template <typename OpT>
static bool overlapsEarlierSimtKeepResumeSlotUse(OpT op,
                                                 SmallVectorImpl<int64_t> &used) {
  std::optional<unsigned> registerCount =
      getSimtKeepResumeRegisterCount(getSimtKeepResumeValueType(op));
  if (!registerCount) {
    return false;
  }
  int64_t slot = op.getSlot();
  for (int64_t word = slot; word < slot + *registerCount; ++word) {
    if (llvm::is_contained(used, word)) {
      return true;
    }
  }
  for (int64_t word = slot; word < slot + *registerCount; ++word) {
    used.push_back(word);
  }
  return false;
}

static LogicalResult verifyUniqueResumeGroupSlots(ResumeOp current,
                                                  Operation *first) {
    SmallVector<int64_t, mlir::pto::kValue4> slots;
    for (Operation* cur = first; cur; cur = cur->getNextNode()) {
        auto resume = dyn_cast<ResumeOp>(cur);
        if (!resume) {
            break;
        }
        if (overlapsEarlierSimtKeepResumeSlotUse(resume, slots) && resume.getOperation() == current.getOperation()) {
            return current.emitOpError() << "duplicates an earlier slot " << resume.getSlot()
                                         << " in the SIMT resume prologue group";
        }
    }
  return success();
}

static LogicalResult verifyUniqueKeepGroupSlots(KeepOp current,
                                                Operation *first,
                                                Operation *last) {
    SmallVector<int64_t, mlir::pto::kValue4> slots;
    for (Operation* cur = first; cur; cur = cur->getNextNode()) {
        auto keep = dyn_cast<KeepOp>(cur);
        if (!keep) {
            break;
        }
        if (overlapsEarlierSimtKeepResumeSlotUse(keep, slots) && keep.getOperation() == current.getOperation()) {
            return current.emitOpError() << "duplicates an earlier slot " << keep.getSlot()
                                         << " in the SIMT keep epilogue group";
        }
        if (cur == last) {
            break;
        }
    }
  return success();
}

static bool isSupportedSimtKeepResumeType(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
      return intType.getWidth() <= mlir::pto::kValue64;
  }
  return type.isF16() || type.isBF16() || type.isF32();
}

static bool isInsideSimtExecutionScope(Operation *op) {
  func::FuncOp func = getParentFunc(op);
  return (func && func->hasAttr(pto::kPTOSimtEntryAttrName)) ||
         op->getParentOfType<pto::SectionSimtOp>();
}

static LogicalResult verifyInsideSimtExecutionScope(Operation *op) {
  if (!isInsideSimtExecutionScope(op)) {
    return op->emitOpError("must appear inside a function marked with '")
           << pto::kPTOSimtEntryAttrName
           << "' or inside pto.section.simt";
  }
  return success();
}

static LogicalResult verifySimtKeepResumeCommon(Operation *op, int64_t slot) {
  if (!isInsideSimtExecutionScope(op)) {
    return op->emitOpError("must appear inside a function marked with '")
           << pto::kPTOSimtEntryAttrName << "' or inside pto.section.simt";
  }
  if (slot < 0 || slot >= kSimtKeepResumeSlotLimit) {
    return op->emitOpError("requires slot in range [0, ")
           << (kSimtKeepResumeSlotLimit - 1) << "]";
  }
  return success();
}

LogicalResult SyncthreadsOp::verify() {
  return verifyInsideSimtExecutionScope(getOperation());
}

LogicalResult ThreadfenceOp::verify() {
  return verifyInsideSimtExecutionScope(getOperation());
}

LogicalResult ThreadfenceBlockOp::verify() {
  return verifyInsideSimtExecutionScope(getOperation());
}

void SyncthreadsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

void ThreadfenceOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

void ThreadfenceBlockOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

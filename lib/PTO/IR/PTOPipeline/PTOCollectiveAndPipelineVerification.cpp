// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

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

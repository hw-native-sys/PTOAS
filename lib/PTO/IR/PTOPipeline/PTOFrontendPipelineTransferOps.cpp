// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

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

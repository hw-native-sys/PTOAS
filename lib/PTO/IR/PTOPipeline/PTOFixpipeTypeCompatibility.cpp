// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

static bool matchesFixpipeConsumerElementType(FixpipeQuant quant,
                                              Type resultElemType) {
  switch (quant) {
  case FixpipeQuant::NoConvert:
      return resultElemType.isF32() || resultElemType.isInteger(mlir::pto::kValue32);
  case FixpipeQuant::F32F16:
  case FixpipeQuant::DEQF16Scalar:
  case FixpipeQuant::DEQF16Vec:
  case FixpipeQuant::QF322F16PreScalar:
    return resultElemType.isF16();
  case FixpipeQuant::F32BF16:
  case FixpipeQuant::QF322BF16PreScalar:
  case FixpipeQuant::QS322BF16PreScalar:
  case FixpipeQuant::QS322BF16PreVec:
    return resultElemType.isBF16();
  case FixpipeQuant::REQ8Scalar:
  case FixpipeQuant::QF322B8PreScalar:
    return isSignedOrUnsignedI8(resultElemType);
  case FixpipeQuant::REQ8Vec:
  case FixpipeQuant::QF322B8PreVec:
    return isSignedI8(resultElemType);
  case FixpipeQuant::QF322HIF8PreScalar:
    return isa<HiF8Type>(resultElemType);
  case FixpipeQuant::QF322FP8PreScalar:
    return isPTOFloat8E4M3LikeType(resultElemType);
  }
  llvm_unreachable("unhandled FixpipeQuant");
}

static bool isFixpipeQuantPayloadElemType(Type elemTy, PTOArch arch) {
  if (!elemTy) {
    return false;
  }
  bool isPackedI64 = elemTy.isUnsignedInteger(64) ||
                     elemTy.isSignlessInteger(64) ||
                     elemTy.isSignedInteger(64);
  // SET_QUANT_VECTOR passes each column to the hardware as a packed 64-bit
  // control word. The frontend does not convert floating-point elements into
  // that representation, so accepting f16/bf16/f32 would produce invalid
  // quantization parameters on both A3 and A5.
  return (arch == PTOArch::A3 || arch == PTOArch::A5) && isPackedI64;
}

static bool matchesFixpipeProducerElementType(FixpipeQuant quant,
                                              Type srcElemType) {
  switch (quant) {
  case FixpipeQuant::NoConvert:
      return srcElemType.isF32() || srcElemType.isInteger(mlir::pto::kValue32);
  case FixpipeQuant::F32F16:
  case FixpipeQuant::F32BF16:
  case FixpipeQuant::QF322B8PreScalar:
  case FixpipeQuant::QF322B8PreVec:
  case FixpipeQuant::QF322F16PreScalar:
  case FixpipeQuant::QF322BF16PreScalar:
  case FixpipeQuant::QF322HIF8PreScalar:
  case FixpipeQuant::QF322FP8PreScalar:
    return srcElemType.isF32();
  case FixpipeQuant::REQ8Scalar:
  case FixpipeQuant::REQ8Vec:
  case FixpipeQuant::DEQF16Scalar:
  case FixpipeQuant::DEQF16Vec:
  case FixpipeQuant::QS322BF16PreScalar:
  case FixpipeQuant::QS322BF16PreVec:
      return srcElemType.isInteger(mlir::pto::kValue32);
  }
  llvm_unreachable("unhandled FixpipeQuant");
}

static bool matchesFixpipeProducerAndConsumerTypes(FixpipeQuant quant,
                                                   Type srcElemType,
                                                   Type dstElemType) {
  return matchesFixpipeProducerElementType(quant, srcElemType) &&
         matchesFixpipeConsumerElementType(quant, dstElemType) &&
         (quant != FixpipeQuant::NoConvert || srcElemType == dstElemType);
}

static bool isUnpublishedFixpipeFrontendAttrName(StringRef name) {
  return llvm::StringSwitch<bool>(name)
      .Case("stPhase", true)
      .Case("st_phase", true)
      .Case("atomicType", true)
      .Case("atomic_type", true)
      .Case("subBlockId", true)
      .Case("subBlockid", true)
      .Case("sub_blockid", true)
      .Case("clipReluMode", true)
      .Case("clip_relu_mode", true)
      .Case("isChannelSplit", true)
      .Case("is_channel_split", true)
      .Case("channelSplit", true)
      .Case("channel_split", true)
      .Default(false);
}

static LogicalResult verifyNoUnpublishedFixpipeFrontendAttrs(Operation *op) {
  for (NamedAttribute attr : op->getAttrs()) {
    StringRef name = attr.getName().getValue();
    if (!isUnpublishedFixpipeFrontendAttrName(name)) {
      continue;
    }
    return op->emitOpError()
           << "does not allow unpublished fixpipe attr '" << name
           << "'; STPhase / AtomicType / SubBlockId / ClipReluMode / "
              "IsChannelSplit are not part of the PTOIR frontend surface";
  }
  return success();
}

static std::optional<uint64_t> getStaticTileByteSize(TileBufType tileTy) {
  auto shape = getShapeVec(tileTy);
  auto elemCount = getStaticElementCount(shape);
  uint64_t elemBytes = getElemByteSize(tileTy.getElementType());
  if (!elemCount || elemBytes == 0) {
    return std::nullopt;
  }
  return *elemCount * elemBytes;
}

static LogicalResult verifyFixpipePeerPopTypes(Operation *tpopOp,
                                               func::FuncOp funcOp, int32_t id,
                                               Type resultTileType) {
  bool mismatch = false;
  funcOp.walk([&](TPopFromAicOp otherPop) {
    if (otherPop.getOperation() == tpopOp ||
        otherPop.getId() != static_cast<uint32_t>(id))
      return WalkResult::advance();
    if (otherPop.getTile().getType() != resultTileType) {
      mismatch = true;
      tpopOp->emitOpError()
          << "expects all tpop_from_aic results for fixpipe pipe id = " << id
          << " to use the same tile type";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return success(!mismatch);
}

static LogicalResult verifyFixpipeConsumerType(Operation *tpopOp, int32_t id,
                                                Type resultTileType) {
  auto funcOp = tpopOp->getParentOfType<func::FuncOp>();
  if (!funcOp) {
    return success(); // Already checked elsewhere
  }

  // Look up the consumer-side init
  auto initOr = lookupFrontendInitOpById(tpopOp, funcOp, id);
  if (failed(initOr)) {
    return failure();
  }

  Operation *initOp = *initOr;
  auto aivInit = dyn_cast<AivInitializePipeOp>(initOp);
  if (!aivInit) {
    return success(); // Not consumer init, skip
  }

  auto accPushEpilogue = aivInit.getAccPushEpilogueAttr();
  if (!accPushEpilogue) {
    return success(); // Not a fixpipe, skip
  }

  if (auto tpop = dyn_cast<TPopFromAicOp>(tpopOp); tpop && tpop.getSplit() != 0) {
    return tpop.emitOpError("expects fixpipe TPOP to have split = 0");
  }

  // Rule 11: At least one tpop must exist (checked by counting all tpops for this pipe)
  // Rule 12: Verify result element type matches expected type from quant mode
  auto tileTy = dyn_cast<pto::TileBufType>(resultTileType);
  if (!tileTy) {
    return tpopOp->emitOpError(
        "expects fixpipe TPOP result to be a tile type");
  }

  Type resultElemType = tileTy.getElementType();
  auto quant = accPushEpilogue.getQuant();
  if (failed(
          verifyFixpipePeerPopTypes(tpopOp, funcOp, id, resultTileType))) {
    return failure();
  }

  if (!matchesFixpipeConsumerElementType(quant, resultElemType)) {
    return tpopOp->emitOpError()
           << "expects consumer element type to match acc_push_epilogue.quant "
           << stringifyFixpipeQuant(quant);
  }

  if (!matchesFixpipeConsumerLayout(accPushEpilogue.getLayout(), tileTy)) {
    return tpopOp->emitOpError()
           << "expects consumer tile layout to match acc_push_epilogue.layout "
           << stringifyFixpipeLayout(accPushEpilogue.getLayout());
  }

  return success();
}


static LogicalResult verifyPipeShape(Operation *op, int8_t dirMask, int32_t slotSize,
                                     int32_t slotNum,
                                     std::optional<int32_t> flagBase) {
  constexpr int32_t kMaxHardwareFlagIds = 16;
  if (dirMask != 1 && dirMask != mlir::pto::kValue2 && dirMask != mlir::pto::kValue3) {
      return op->emitOpError("expects 'dir_mask' to be 1, 2, or 3");
  }
  if (slotSize <= 0) {
    return op->emitOpError("expects 'slot_size' to be greater than 0");
  }
  if (slotNum <= 0) {
    return op->emitOpError("expects 'slot_num' to be greater than 0");
  }
  if (flagBase && *flagBase < 0) {
    return op->emitOpError("expects 'flag_base' to be non-negative when present");
  }
  if (flagBase) {
    int32_t flagWidth = dirMask == 3 ? 4 : 2;
    if (*flagBase + flagWidth > kMaxHardwareFlagIds) {
      return op->emitOpError()
             << "requires 'flag_base' and dir_mask to fit within "
             << kMaxHardwareFlagIds << " hardware flag ids";
    }
  }

  return success();
}

static LogicalResult verifyPipeHandleProducer(Operation *op, Value pipeHandle) {
  if (!isa<pto::PipeType>(pipeHandle.getType())) {
    return op->emitOpError("expects pipe operand type !pto.pipe");
  }
  if (!pipeHandle.getDefiningOp<InitializeL2LPipeOp>() &&
      !pipeHandle.getDefiningOp<InitializeL2G2LPipeOp>()) {
    return op->emitOpError(
        "pipe_handle must be produced by pto.initialize_l2l_pipe or "
        "pto.initialize_l2g2l_pipe");
  }
  return success();
}

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

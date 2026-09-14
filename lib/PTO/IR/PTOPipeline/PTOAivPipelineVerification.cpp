// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

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

// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.


static FailureOr<Operation *>
lookupFixpipePeerConsumerInit(AicInitializePipeOp producerInit) {
  if (!producerInit.getC2vConsumerBuf()) {
    return failure();
  }
  auto importOp = dyn_cast_or_null<ImportReservedBufferOp>(
      producerInit.getC2vConsumerBuf().getDefiningOp());
  if (!importOp) {
    return failure();
  }

  auto peerConsumerFunc =
      lookupPeerFuncAcrossContainer(importOp.getOperation(),
                                    importOp.getPeerFuncAttr());
  if (!peerConsumerFunc) {
    return failure();
  }

  StringRef bufferName = importOp.getName();
  Operation *matchedInit = nullptr;
  unsigned matchedInitCount = 0;
  peerConsumerFunc.walk([&](Operation *candidate) {
    if (auto aivInit = dyn_cast<AivInitializePipeOp>(candidate)) {
      if (!aivInit.getC2vConsumerBuf()) {
        return WalkResult::advance();
      }
      auto reserveOp = dyn_cast_or_null<ReserveBufferOp>(
          aivInit.getC2vConsumerBuf().getDefiningOp());
      if (!reserveOp || reserveOp.getName() != bufferName) {
        return WalkResult::advance();
      }
      matchedInit = candidate;
      ++matchedInitCount;
      return WalkResult::advance();
    }

    if (!matchesLoweredFixpipePeerContract(candidate, peerConsumerFunc,
                                           bufferName)) {
      return WalkResult::advance();
    }

    matchedInit = candidate;
    ++matchedInitCount;
    return WalkResult::advance();
  });

  if (matchedInitCount != 1) {
    return failure();
  }
  return matchedInit;
}

static FailureOr<Operation *>
lookupFixpipePeerProducerInit(AivInitializePipeOp consumerInit,
                              func::FuncOp peerProducerFunc,
                              StringRef bufferName,
                              func::FuncOp currentConsumerFunc) {
  Operation *matchedInit = nullptr;
  unsigned matchedInitCount = 0;
  peerProducerFunc.walk([&](Operation *candidate) {
    if (auto aicInit = dyn_cast<AicInitializePipeOp>(candidate)) {
      if (!aicInit.getC2vConsumerBuf()) {
        return WalkResult::advance();
      }

      auto importOp = dyn_cast_or_null<ImportReservedBufferOp>(
          aicInit.getC2vConsumerBuf().getDefiningOp());
      if (!importOp) {
        return WalkResult::advance();
      }

      auto peerConsumerFunc =
          lookupPeerFuncAcrossContainer(importOp.getOperation(),
                                        importOp.getPeerFuncAttr());
      if (importOp.getName() != bufferName ||
          peerConsumerFunc != currentConsumerFunc) {
        return WalkResult::advance();
      }

      matchedInit = candidate;
      ++matchedInitCount;
      return WalkResult::advance();
    }

    if (!matchesLoweredFixpipePeerContract(candidate, currentConsumerFunc,
                                           bufferName)) {
      return WalkResult::advance();
    }

    matchedInit = candidate;
    ++matchedInitCount;
    return WalkResult::advance();
  });

  if (matchedInitCount != 1) {
    consumerInit.emitOpError()
        << "expects peer producer function to contain a matching "
           "aic_initialize_pipe with the same consumer buffer contract";
    return failure();
  }

  return matchedInit;
}

static std::pair<Operation *, unsigned> findFrontendInitById(
    func::FuncOp funcOp, int32_t id) {
  Operation *matched = nullptr;
  unsigned count = 0;
  funcOp.walk([&](Operation *candidate) {
    if (auto aic = dyn_cast<AicInitializePipeOp>(candidate)) {
      if (aic.getId() == static_cast<uint32_t>(id)) {
        matched = candidate;
        ++count;
      }
      return WalkResult::advance();
    }
    if (auto aiv = dyn_cast<AivInitializePipeOp>(candidate)) {
      if (aiv.getId() == static_cast<uint32_t>(id)) {
        matched = candidate;
        ++count;
      }
      return WalkResult::advance();
    }
    return WalkResult::advance();
  });
  return {matched, count};
}

static std::pair<Operation *, unsigned> findLoweredInitById(
    func::FuncOp funcOp, int32_t id) {
  Operation *matched = nullptr;
  unsigned count = 0;
  funcOp.walk([&](Operation *candidate) {
    if (!isa<InitializeL2LPipeOp, InitializeL2G2LPipeOp>(candidate)) {
      return WalkResult::advance();
    }
    auto frontendIdAttr =
        candidate->getAttrOfType<IntegerAttr>(kFrontendPipeIdAttrName);
    if (!frontendIdAttr || frontendIdAttr.getInt() != id) {
      return WalkResult::advance();
    }
    matched = candidate;
    ++count;
    return WalkResult::advance();
  });
  return {matched, count};
}

static FailureOr<Operation *> lookupFrontendOrLoweredInitOpById(
    Operation *op, func::FuncOp funcOp, int32_t id) {
  auto [frontend, frontendCount] = findFrontendInitById(funcOp, id);
  if (frontendCount == 1)
    return frontend;
  if (frontendCount > 1) {
    op->emitOpError() << "expects 'id' = " << id
                      << " to match exactly one frontend initialize_pipe op in the same function";
    return failure();
  }
  auto [lowered, loweredCount] = findLoweredInitById(funcOp, id);
  if (loweredCount == 0) {
    op->emitOpError() << "expects 'id' = " << id
                      << " to match a frontend or lowered initialize_pipe op in the same function";
    return failure();
  }
  if (loweredCount > 1) {
    op->emitOpError() << "expects 'id' = " << id
                      << " to match exactly one frontend or lowered initialize_pipe op in the same function";
    return failure();
  }
  return lowered;
}

template <typename InitOpT>
static bool supportsFrontendOddDirection(Operation *op, InitOpT init,
                                         bool expectC2V) {
  PTOArch arch = getTargetArch(op);
  bool gmBacking = static_cast<bool>(init.getGmSlotTensor()) ||
                   (arch != PTOArch::A5 &&
                    static_cast<bool>(init.getGmSlotBuffer()));
  int8_t directionMask = expectC2V ? 1 : 2;
  Value buffer = expectC2V ? init.getC2vConsumerBuf()
                           : init.getV2cConsumerBuf();
  return (init.getDirMask() & directionMask) != 0 && gmBacking && buffer;
}

static FailureOr<func::FuncOp> getRequiredParentFunc(Operation *op) {
  auto funcOp = op->getParentOfType<func::FuncOp>();
  if (!funcOp) {
    op->emitOpError("must be nested under a func.func");
    return failure();
  }
  return funcOp;
}

static FailureOr<Operation *> getRequiredFrontendInit(Operation *op,
                                                      int32_t id) {
  auto funcOp = getRequiredParentFunc(op);
  if (failed(funcOp))
    return failure();
  return lookupFrontendInitOpById(op, *funcOp, id);
}

static LogicalResult verifyFrontendSplitOp(Operation *op,
                                           FunctionKernelKind expected,
                                           StringRef kernelName,
                                           int32_t id,
                                           int64_t split,
                                           bool expectC2V) {
  if (failed(verifyFrontendKernelKind(op, expected, kernelName))) {
    return failure();
  }
  if (id < 0) {
    return op->emitOpError("expects 'id' to be non-negative");
  }
  if (failed(verifySplitAttr(op, split))) {
    return failure();
  }
  if (!isOddSplit(split)) {
    return success();
  }

  auto initOr = getRequiredFrontendInit(op, id);
  if (failed(initOr)) {
    return failure();
  }
  if (!expectC2V && getTargetArch(op) == PTOArch::A5) {
    return op->emitOpError(
        "supports odd V2C split modes (split = 3 or 4) only on a2/a3");
  }

  bool supported = false;
  if (auto aic = dyn_cast<AicInitializePipeOp>(*initOr)) {
    supported = supportsFrontendOddDirection(op, aic, expectC2V);
  } else {
    supported = supportsFrontendOddDirection(
        op, cast<AivInitializePipeOp>(*initOr), expectC2V);
  }
  if (!supported) {
    return op->emitOpError()
           << "supports odd split modes (split = 3 or 4) only for a "
              "GM-backed tile pipe whose dir_mask enables "
           << (expectC2V ? "C2V" : "V2C")
           << " and provides the matching consumer buffer";
  }
  return success();
}

static LogicalResult verifyOddSplitTileEntry(Operation *op, int64_t split,
                                             Type entryTy) {
  if (isOddSplit(split) && isa<TensorViewType>(entryTy)) {
    return op->emitOpError(
        "supports odd split modes (split = 3 or 4) only for tile entries; "
        "the pinned pto-isa does not implement odd GlobalTensor offsets");
  }
  return success();
}

static LogicalResult verifyFullTileSplitParity(Operation *op, int64_t split,
                                               Type entryTy) {
  if (split == 0) {
    return success();
  }

  ArrayRef<int64_t> shape;
  if (auto tileTy = dyn_cast<TileBufType>(entryTy)) {
    shape = tileTy.getValidShape();
  } else if (auto viewTy = dyn_cast<TensorViewType>(entryTy)) {
    shape = viewTy.getShape();
  } else {
    return success();
}
if (shape.size() != mlir::pto::kValue2) {
    return success();
}

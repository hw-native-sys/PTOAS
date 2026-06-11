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

template <typename InitOpT>
static LogicalResult verifyFrontendInitCommon(InitOpT op,
                                              FunctionKernelKind expected,
                                              StringRef kernelName) {
  if (failed(verifyFrontendKernelKind(op.getOperation(), expected, kernelName)))
    return failure();

  auto funcOp = op->template getParentOfType<func::FuncOp>();
  if (!funcOp)
    return op.emitOpError("must be nested under a func.func");

  if (op.getId() < 0)
    return op.emitOpError("expects 'id' to be non-negative");

  unsigned sameIdInitCount = countFrontendInitOpsWithSameId<InitOpT>(
      funcOp, op.getId());
  if (sameIdInitCount > 1) {
    return op.emitOpError(
        "requires 'id' to be unique across frontend initialize_pipe ops in the function");
  }

  int8_t dirMask = op.getDirMask();
  if (dirMask != kPTOFrontendDirMaskC2V &&
      dirMask != kPTOFrontendDirMaskV2C &&
      dirMask != kPTOFrontendDirMaskBidirectional)
    return op.emitOpError("expects 'dir_mask' to be 1, 2, or 3");
  if (op.getSlotSize() <= 0)
    return op.emitOpError("expects 'slot_size' to be greater than 0");

  bool hasGlobalSlotTensor = static_cast<bool>(op.getGmSlotTensor());
  bool hasC2vConsumerBuf = static_cast<bool>(op.getC2vConsumerBuf());
  bool hasV2cConsumerBuf = static_cast<bool>(op.getV2cConsumerBuf());
  if (hasGlobalSlotTensor) {
    return verifyFrontendInitGlobalTensorForm(op, dirMask, hasC2vConsumerBuf,
                                              hasV2cConsumerBuf);
  }
  return verifyFrontendInitLocalPipeForm(op, dirMask, hasC2vConsumerBuf,
                                         hasV2cConsumerBuf);
}

static ReserveBufferOp findReserveBufferByName(func::FuncOp funcOp,
                                               StringRef name) {
  ReserveBufferOp found;
  funcOp.walk([&found, name](ReserveBufferOp reserveOp) {
    if (reserveOp.getName() != name)
      return WalkResult::advance();
    found = reserveOp;
    return WalkResult::interrupt();
  });
  return found;
}

LogicalResult ReserveBufferOp::verify() {
  auto funcOp = getOperation()->getParentOfType<func::FuncOp>();
  if (!funcOp)
    return emitOpError("must be nested under a func.func");

  if (getSize() <= 0)
    return emitOpError("expects 'size' to be greater than 0");

  auto location = getLocation().getAddressSpace();
  if (location != AddressSpace::VEC && location != AddressSpace::MAT)
    return emitOpError("expects 'location' to be #pto.address_space<vec> or #pto.address_space<mat>");

  if (!getAutoAlloc() && !getBaseAttr())
    return emitOpError("expects 'base' when 'auto' is false");

  if (auto baseAttr = getBaseAttr(); baseAttr && baseAttr.getInt() < 0)
    return emitOpError("expects 'base' to be non-negative when present");

  unsigned sameNameCount = 0;
  funcOp.walk([this, &sameNameCount](ReserveBufferOp reserveOp) {
    if (reserveOp.getName() == getName())
      ++sameNameCount;
  });
  if (sameNameCount > 1)
    return emitOpError("requires 'name' to be unique within the function");

  return success();
}

LogicalResult ImportReservedBufferOp::verify() {
  auto funcOp = getOperation()->getParentOfType<func::FuncOp>();
  if (!funcOp)
    return emitOpError("must be nested under a func.func");

  auto peerFunc = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
      getOperation(), getPeerFuncAttr());
  if (!peerFunc)
    return emitOpError("expects 'peer_func' to reference an existing func.func");

  unsigned sameImportCount = 0;
  funcOp.walk([this, &sameImportCount](ImportReservedBufferOp importOp) {
    if (importOp.getName() == getName() &&
        importOp.getPeerFuncAttr() == getPeerFuncAttr()) {
      ++sameImportCount;
    }
  });
  if (sameImportCount > 1) {
    return emitOpError(
        "requires (name, peer_func) to be unique within the function");
  }

  if (!findReserveBufferByName(peerFunc, getName()))
    return emitOpError("expects matching peer reserve_buffer to exist");

  return success();
}

static FailureOr<Operation *> lookupFrontendInitOpById(Operation *op,
                                                       func::FuncOp funcOp,
                                                       int32_t id) {
  Operation *matchedInit = nullptr;
  unsigned matchedInitCount = 0;
  funcOp.walk([id, &matchedInit, &matchedInitCount](Operation *candidate) {
    if (auto aic = dyn_cast<AicInitializePipeOp>(candidate)) {
      if (aic.getId() == static_cast<uint32_t>(id)) {
        matchedInit = candidate;
        ++matchedInitCount;
      }
      return WalkResult::advance();
    }
    if (auto aiv = dyn_cast<AivInitializePipeOp>(candidate)) {
      if (aiv.getId() == static_cast<uint32_t>(id)) {
        matchedInit = candidate;
        ++matchedInitCount;
      }
      return WalkResult::advance();
    }
    return WalkResult::advance();
  });

  if (matchedInitCount == 0) {
    op->emitOpError() << "expects 'id' = " << id
                      << " to match a frontend initialize_pipe op in the same function";
    return failure();
  }
  if (matchedInitCount > 1) {
    op->emitOpError() << "expects 'id' = " << id
                      << " to match exactly one frontend initialize_pipe op in the same function";
    return failure();
  }
  return matchedInit;
}

static LogicalResult verifyFrontendSplitOp(Operation *op,
                                           FunctionKernelKind expected,
                                           StringRef kernelName,
                                           int32_t id,
                                           int64_t split) {
  if (failed(verifyFrontendKernelKind(op, expected, kernelName)))
    return failure();
  if (id < 0)
    return op->emitOpError("expects 'id' to be non-negative");
  return verifySplitAttr(op, split);
}

static FailureOr<int8_t> lookupFrontendInitDirMaskById(Operation *op,
                                                       func::FuncOp funcOp,
                                                       int32_t id) {
  auto initOr = lookupFrontendInitOpById(op, funcOp, id);
  if (failed(initOr))
    return failure();
  if (auto aic = dyn_cast<AicInitializePipeOp>(*initOr))
    return aic.getDirMask();
  return cast<AivInitializePipeOp>(*initOr).getDirMask();
}

static LogicalResult verifyFrontendDataOpDirection(Operation *op, int32_t id,
                                                   bool expectC2V) {
  auto funcOp = op->getParentOfType<func::FuncOp>();
  if (!funcOp)
    return op->emitOpError("must be nested under a func.func");

  auto dirMaskOr = lookupFrontendInitDirMaskById(op, funcOp, id);
  if (failed(dirMaskOr))
    return failure();

  int8_t dirMask = *dirMaskOr;
  if (expectC2V && dirMask != kPTOFrontendDirMaskC2V &&
      dirMask != kPTOFrontendDirMaskBidirectional) {
    return op->emitOpError()
           << "expects 'id' = " << id
           << " to reference initialize_pipe with dir_mask = 1 or 3";
  }
  if (!expectC2V && dirMask != kPTOFrontendDirMaskV2C &&
      dirMask != kPTOFrontendDirMaskBidirectional) {
    return op->emitOpError()
           << "expects 'id' = " << id
           << " to reference initialize_pipe with dir_mask = 2 or 3";
  }
  return success();
}

static Value getFrontendInitGmSlotTensor(Operation *initOp) {
  if (auto aic = dyn_cast<AicInitializePipeOp>(initOp))
    return aic.getGmSlotTensor();
  return cast<AivInitializePipeOp>(initOp).getGmSlotTensor();
}

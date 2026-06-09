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

static LogicalResult verifyFrontendTensorEntryMatchesInit(Operation *op,
                                                          int32_t id,
                                                          Type entryTy) {
  auto entryViewTy = dyn_cast<TensorViewType>(entryTy);
  if (!entryViewTy)
    return success();

  auto funcOp = op->getParentOfType<func::FuncOp>();
  if (!funcOp)
    return op->emitOpError("must be nested under a func.func");

  auto initOr = lookupFrontendInitOpById(op, funcOp, id);
  if (failed(initOr))
    return failure();
  Value gmSlotTensor = getFrontendInitGmSlotTensor(*initOr);
  if (!gmSlotTensor) {
    return op->emitOpError()
           << "expects 'id' = " << id
           << " to reference initialize_pipe with 'gm_slot_tensor' when the "
              "pipe entry is !pto.tensor_view";
  }

  auto slotTensorTy = dyn_cast<TensorViewType>(gmSlotTensor.getType());
  if (!slotTensorTy)
    return op->emitOpError("expects 'gm_slot_tensor' to be !pto.tensor_view");
  if (slotTensorTy.getElementType() != entryViewTy.getElementType()) {
    return op->emitOpError()
           << "expects pipe entry element type to match gm_slot_tensor element type";
  }
  if (slotTensorTy.getRank() != entryViewTy.getRank()) {
    return op->emitOpError()
           << "expects pipe entry rank to match gm_slot_tensor rank";
  }

  ArrayRef<int64_t> slotShape = slotTensorTy.getShape();
  ArrayRef<int64_t> entryShape = entryViewTy.getShape();
  for (auto [idx, entryDim] : llvm::enumerate(entryShape)) {
    int64_t slotDim = slotShape[idx];
    if (slotDim == ShapedType::kDynamic ||
        entryDim == ShapedType::kDynamic || slotDim == entryDim)
      continue;
    return op->emitOpError()
           << "expects pipe entry dimension " << idx
           << " to match gm_slot_tensor dimension " << slotDim;
  }
  return success();
}

template <typename FrontendPopOpT>
static LogicalResult verifyFrontendPopOp(FrontendPopOpT op,
                                         FunctionKernelKind expected,
                                         StringRef kernelName,
                                         bool expectC2V) {
  if (failed(verifyFrontendSplitOp(op.getOperation(), expected, kernelName,
                                   op.getId(),
                                   op.getSplit())))
    return failure();
  if (failed(verifyFrontendDataOpDirection(op.getOperation(), op.getId(),
                                           expectC2V)))
    return failure();
  if (failed(verifyFrontendTensorEntryMatchesInit(op.getOperation(), op.getId(),
                                                  op.getTile().getType())))
    return failure();

  bool hasValidRow = static_cast<bool>(op.getValidRow());
  bool hasValidCol = static_cast<bool>(op.getValidCol());
  if (hasValidRow != hasValidCol)
    return op.emitOpError(
        "expects valid_row and valid_col operands to be provided together");
  if (!hasValidRow)
    return success();

  if (isa<TensorViewType>(op.getTile().getType()))
    return op.emitOpError(
        "does not accept valid_row/valid_col when result is !pto.tensor_view");

  auto tileTy = dyn_cast<TileBufType>(op.getTile().getType());
  if (!tileTy)
    return op.emitOpError(
        "expects tile result to be !pto.tile_buf when valid_row/valid_col operands are provided");
  if (!tileTy.hasDynamicValid())
    return op.emitOpError(
        "expects tile result to have dynamic validShape (?, ?) when valid_row/valid_col operands are provided");
  return success();
}

static LogicalResult verifyPipeShape(Operation *op, int8_t dirMask, int32_t slotSize,
                                     int32_t slotNum,
                                     std::optional<int32_t> flagBase) {
  constexpr int32_t kMaxHardwareFlagIds = 16;
  static constexpr int32_t kPTOFrontendMinSlotNum = 4;
  static constexpr int32_t kPTOFrontendMaxSlotNum = 8;
  static constexpr int32_t kPTOFrontendUnidirectionalFlagWidth = 2;
  static constexpr int32_t kPTOFrontendBidirectionalFlagWidth = 4;
  if (dirMask != kPTOFrontendDirMaskC2V &&
      dirMask != kPTOFrontendDirMaskV2C &&
      dirMask != kPTOFrontendDirMaskBidirectional)
    return op->emitOpError("expects 'dir_mask' to be 1, 2, or 3");
  if (slotSize <= 0)
    return op->emitOpError("expects 'slot_size' to be greater than 0");
  if (slotNum != kPTOFrontendMinSlotNum &&
      slotNum != kPTOFrontendMaxSlotNum)
    return op->emitOpError("expects 'slot_num' to be 4 or 8");
  if (flagBase && *flagBase < 0)
    return op->emitOpError("expects 'flag_base' to be non-negative when present");
  if (flagBase) {
    int32_t flagWidth = dirMask == kPTOFrontendDirMaskBidirectional
                            ? kPTOFrontendBidirectionalFlagWidth
                            : kPTOFrontendUnidirectionalFlagWidth;
    if (*flagBase + flagWidth > kMaxHardwareFlagIds) {
      return op->emitOpError()
             << "requires 'flag_base' and dir_mask to fit within "
             << kMaxHardwareFlagIds << " hardware flag ids";
    }
  }

  return success();
}

static LogicalResult verifyPipeHandleProducer(Operation *op, Value pipeHandle) {
  if (!isa<pto::PipeType>(pipeHandle.getType()))
    return op->emitOpError("expects pipe operand type !pto.pipe");
  if (!pipeHandle.getDefiningOp<InitializeL2LPipeOp>() &&
      !pipeHandle.getDefiningOp<InitializeL2G2LPipeOp>()) {
    return op->emitOpError(
        "pipe_handle must be produced by pto.initialize_l2l_pipe or "
        "pto.initialize_l2g2l_pipe");
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
  if (auto memrefTy = dyn_cast<MemRefType>(ty)) {
    elementType = memrefTy.getElementType();
    shape = memrefTy.getShape();
    return true;
  }
  return false;
}

static FailureOr<InitializeL2G2LPipeOp> verifyTensorEntryInternalPipeHandle(
    Operation *op, Value pipeHandle) {
  auto initOp = pipeHandle.getDefiningOp<InitializeL2G2LPipeOp>();
  if (!initOp) {
    op->emitOpError()
        << "expects !pto.tensor_view pipe entry to use a pipe produced by "
           "pto.initialize_l2g2l_pipe";
    return failure();
  }
  if (initOp.getLocalAddr()) {
    op->emitOpError()
        << "expects !pto.tensor_view pipe entry to use global-only "
           "pto.initialize_l2g2l_pipe without local_addr";
    return failure();
  }
  return initOp;
}

static LogicalResult verifyTensorEntrySlotType(Operation *op,
                                               TensorViewType entryViewTy,
                                               InitializeL2G2LPipeOp initOp) {
  Type slotElementType;
  ArrayRef<int64_t> slotShape;
  if (!getTensorLikeElementAndShape(initOp.getGmAddr().getType(),
                                    slotElementType, slotShape)) {
    return op->emitOpError()
           << "expects !pto.tensor_view pipe entry to use "
              "pto.initialize_l2g2l_pipe gm_addr with tensor/memref slot type";
  }
  if (slotElementType != entryViewTy.getElementType()) {
    return op->emitOpError()
           << "expects pipe entry element type to match initialize_l2g2l_pipe "
              "gm_addr element type";
  }
  if (slotShape.size() != static_cast<size_t>(entryViewTy.getRank())) {
    return op->emitOpError()
           << "expects pipe entry rank to match initialize_l2g2l_pipe gm_addr "
              "rank";
  }

  ArrayRef<int64_t> entryShape = entryViewTy.getShape();
  for (auto [idx, entryDim] : llvm::enumerate(entryShape)) {
    int64_t slotDim = slotShape[idx];
    if (slotDim == ShapedType::kDynamic || entryDim == ShapedType::kDynamic ||
        slotDim == entryDim) {
      continue;
    }
    return op->emitOpError()
           << "expects pipe entry dimension " << idx
           << " to match initialize_l2g2l_pipe gm_addr dimension " << slotDim;
  }
  return success();
}

static int8_t getTensorEntrySplit(Operation *op) {
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

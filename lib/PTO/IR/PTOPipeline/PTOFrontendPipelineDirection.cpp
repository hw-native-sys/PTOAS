// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

  bool splitRows = split == 1 || split == 3;
  int64_t axisSize = shape[splitRows ? 0 : 1];
  if (axisSize == ShapedType::kDynamic) {
    return success();
  }

  bool expectOdd = isOddSplit(split);
  if ((axisSize % mlir::pto::kValue2 != 0) != expectOdd) {
      return op->emitOpError() << "expects a statically " << (expectOdd ? "odd" : "even") << " valid-"
                               << (splitRows ? "row" : "column") << " count for split = " << split;
  }
  return success();
}

static LogicalResult verifyAivSubblockIdOperand(Operation *op,
                                                Value aivSubblockId,
                                                int64_t split,
                                                Type pipeEntryType) {
  if (!aivSubblockId) {
    return success();
  }

  if (split == 0) {
    return op->emitOpError(
        "expects 'aiv_subblockid' only when 'split' is 1, 2, 3, or 4");
  }

  if (isa<TensorViewType>(pipeEntryType)) {
    return op->emitOpError(
        "does not support 'aiv_subblockid' for !pto.tensor_view pipe entries");
  }

  auto addrSpace = getPTOMemorySpaceEnum(pipeEntryType);
  if (!addrSpace || *addrSpace != AddressSpace::VEC) {
    return op->emitOpError(
        "expects 'aiv_subblockid' only on AIV-side vector tile pipe entries");
  }

  return success();
}

static FailureOr<int8_t> lookupFrontendInitDirMaskById(Operation *op,
                                                       func::FuncOp funcOp,
                                                       int32_t id) {
  auto initOr = lookupFrontendInitOpById(op, funcOp, id);
  if (failed(initOr)) {
    return failure();
  }
  if (auto aic = dyn_cast<AicInitializePipeOp>(*initOr)) {
    return aic.getDirMask();
  }
  return cast<AivInitializePipeOp>(*initOr).getDirMask();
}

static LogicalResult verifyFrontendDataOpDirection(Operation *op, int32_t id,
                                                   bool expectC2V) {
  auto funcOp = op->getParentOfType<func::FuncOp>();
  if (!funcOp) {
    return op->emitOpError("must be nested under a func.func");
  }

  auto dirMaskOr = lookupFrontendInitDirMaskById(op, funcOp, id);
  if (failed(dirMaskOr)) {
    return failure();
  }

  int8_t dirMask = *dirMaskOr;
  if (expectC2V && dirMask != 1 && dirMask != mlir::pto::kValue3) {
      return op->emitOpError() << "expects 'id' = " << id << " to reference initialize_pipe with dir_mask = 1 or 3";
  }
  if (!expectC2V && dirMask != mlir::pto::kValue2 && dirMask != mlir::pto::kValue3) {
      return op->emitOpError() << "expects 'id' = " << id << " to reference initialize_pipe with dir_mask = 2 or 3";
  }
  return success();
}

static Value getFrontendInitGmSlotTensor(Operation *initOp) {
  if (auto aic = dyn_cast<AicInitializePipeOp>(initOp)) {
    return aic.getGmSlotTensor();
  }
  return cast<AivInitializePipeOp>(initOp).getGmSlotTensor();
}

static LogicalResult verifyFrontendTensorEntryMatchesInit(Operation *op,
                                                          int32_t id,
                                                          Type entryTy) {
  auto entryViewTy = dyn_cast<TensorViewType>(entryTy);
  if (!entryViewTy) {
    return success();
  }

  auto initOr = getRequiredFrontendInit(op, id);
  if (failed(initOr)) {
    return failure();
  }
  Value gmSlotTensor = getFrontendInitGmSlotTensor(*initOr);
  if (!gmSlotTensor) {
    return op->emitOpError()
           << "expects 'id' = " << id
           << " to reference initialize_pipe with 'gm_slot_tensor' when the "
              "pipe entry is !pto.tensor_view";
  }

  auto slotTensorTy = dyn_cast<TensorViewType>(gmSlotTensor.getType());
  if (!slotTensorTy) {
    return op->emitOpError("expects 'gm_slot_tensor' to be !pto.tensor_view");
  }
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
        entryDim == ShapedType::kDynamic || slotDim == entryDim) {
      continue;
    }
    return op->emitOpError()
           << "expects pipe entry dimension " << idx
           << " to match gm_slot_tensor dimension " << slotDim;
  }
  return success();
}

template <typename FrontendPopOpT>
static LogicalResult verifyFrontendPopValidOperands(FrontendPopOpT op,
                                                    bool expectC2V) {
  bool hasValidRow = static_cast<bool>(op.getValidRow());
  bool hasValidCol = static_cast<bool>(op.getValidCol());
  if (hasValidRow != hasValidCol)
    return op.emitOpError(
        "expects valid_row and valid_col operands to be provided together");
  if (expectC2V && isOddSplit(op.getSplit()) && !hasValidRow &&
      !isa<TensorViewType>(op.getTile().getType()))
    return op.emitOpError(
        "expects odd C2V split tpop to provide per-sub-core valid_row and "
        "valid_col operands");
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

template <typename FrontendPopOpT>
static LogicalResult verifyFrontendPopOp(FrontendPopOpT op,
                                         FunctionKernelKind expected,
                                         StringRef kernelName,
                                         bool expectC2V) {
  if (failed(verifyFrontendSplitOp(op.getOperation(), expected, kernelName,
                                   op.getId(),
                                   op.getSplit(), expectC2V))) {
    return failure();
  }
  if (failed(verifyFrontendDataOpDirection(op.getOperation(), op.getId(),
                                           expectC2V))) {
    return failure();
  }
  if (failed(verifyOddSplitTileEntry(op.getOperation(), op.getSplit(),
                                     op.getTile().getType()))) {
    return failure();
  }
  if (failed(verifyFrontendTensorEntryMatchesInit(op.getOperation(), op.getId(),
                                                  op.getTile().getType()))) {
    return failure();
  }
  if (!expectC2V &&
      failed(verifyFullTileSplitParity(op.getOperation(), op.getSplit(),
                                       op.getTile().getType()))) {
    return failure();
  }

  return verifyFrontendPopValidOperands(op, expectC2V);
}

static bool isScalarFixpipeQuant(FixpipeQuant quant) {
  switch (quant) {
  case FixpipeQuant::DEQF16Scalar:
  case FixpipeQuant::REQ8Scalar:
  case FixpipeQuant::QF322B8PreScalar:
  case FixpipeQuant::QF322F16PreScalar:
  case FixpipeQuant::QF322BF16PreScalar:
  case FixpipeQuant::QS322BF16PreScalar:
  case FixpipeQuant::QF322HIF8PreScalar:
  case FixpipeQuant::QF322FP8PreScalar:
    return true;
  default:
    return false;
  }
}

static bool isVectorFixpipeQuant(FixpipeQuant quant) {
  switch (quant) {
  case FixpipeQuant::DEQF16Vec:
  case FixpipeQuant::REQ8Vec:
  case FixpipeQuant::QF322B8PreVec:
  case FixpipeQuant::QS322BF16PreVec:
    return true;
  default:
    return false;
  }
}

static bool matchesFixpipeConsumerLayout(FixpipeLayout layout,
                                         TileBufType tileTy) {
  auto memorySpace = dyn_cast_or_null<AddressSpaceAttr>(tileTy.getMemorySpace());
  if (!memorySpace || memorySpace.getAddressSpace() != AddressSpace::VEC) {
    return false;
  }

  int32_t bLayout = tileTy.getBLayoutValueI32();
  int32_t sLayout = tileTy.getSLayoutValueI32();
  switch (layout) {
  case FixpipeLayout::NZ2ND:
    return bLayout == static_cast<int32_t>(BLayout::RowMajor) &&
           sLayout == static_cast<int32_t>(SLayout::NoneBox);
  case FixpipeLayout::NZ2DN:
    return bLayout == static_cast<int32_t>(BLayout::ColMajor) &&
           sLayout == static_cast<int32_t>(SLayout::NoneBox);
  case FixpipeLayout::NZ2NZ:
    return bLayout == static_cast<int32_t>(BLayout::ColMajor) &&
           sLayout == static_cast<int32_t>(SLayout::RowMajor);
  }
  llvm_unreachable("unhandled FixpipeLayout");
}

static bool isSignedOrUnsignedI8(Type ty) {
  if (auto intTy = dyn_cast<IntegerType>(ty)) {
      return intTy.getWidth() == mlir::pto::kValue8 && (intTy.isSigned() || intTy.isUnsigned());
  }
  return false;
}

static bool isSignedI8(Type ty) {
  if (auto intTy = dyn_cast<IntegerType>(ty)) {
      return intTy.isSignedInteger(mlir::pto::kValue8);
  }
  return false;
}

// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

void mlir::pto::SyncAllOp::print(OpAsmPrinter &p) {
  SmallVector<Value, 2> operands;
  if (getGmWorkspace()) {
    operands.push_back(getGmWorkspace());
  }
  if (getUsedCores()) {
    operands.push_back(getUsedCores());
  }

  p << "(";
  if (!operands.empty()) {
    p.printOperands(operands);
    p << " : ";
    llvm::interleaveComma(operands, p,
                          [&](Value operand) { p.printType(operand.getType()); });
  }
  p << ") mode = " << getMode() << ", core_type = " << getCoreType();
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes", "mode",
                                           "core_type"});
}
LogicalResult mlir::pto::SyncWaitOp::verify() {
  return verifySyncSetWaitCommon(getOperation(), getPipe(), getEventIdAttr(),
                                 getEventIdDyn(), getFftsModeAttr(),
                                 "sync.wait");
}

static LogicalResult verifyNamedSyncEventOp(Operation *op, PipeAttr pipe,
                                            IntegerAttr eventIdAttr,
                                            Value eventIdDyn,
                                            int64_t maxEventId,
                                            StringRef opName) {
  const bool hasStaticEventId = eventIdAttr != nullptr;
  const bool hasDynamicEventId = static_cast<bool>(eventIdDyn);
  if (hasStaticEventId == hasDynamicEventId) {
    return op->emitOpError()
           << "expects exactly one event-id form: static attr or dynamic index operand";
  }
  const bool staticEventIdOutOfRange =
      hasStaticEventId &&
      (eventIdAttr.getInt() < 0 || eventIdAttr.getInt() > maxEventId);
  if (staticEventIdOutOfRange) {
    return op->emitOpError() << "expects static event_id in [0, " << maxEventId
                             << "], but got " << eventIdAttr.getInt();
  }
  switch (pipe.getPipe()) {
  case PIPE::PIPE_FIX:
  case PIPE::PIPE_MTE1:
  case PIPE::PIPE_MTE2:
  case PIPE::PIPE_MTE3:
  case PIPE::PIPE_V:
    return success();
  default:
    return op->emitOpError() << opName << " expects pipe to be one of "
                              << "<PIPE_FIX>, <PIPE_MTE1>, <PIPE_MTE2>, "
                              << "<PIPE_MTE3>, <PIPE_V>";
  }
}

ParseResult mlir::pto::SetCrossBlockOp::parse(OpAsmParser &parser,
                                              OperationState &result) {
  return parseSyncEventOpCommon(parser, result,
                                SetCrossBlockOp::getPipeAttrName(result.name),
                                SetCrossBlockOp::getEventIdAttrName(result.name));
}

void mlir::pto::SetCrossBlockOp::print(OpAsmPrinter &p) {
  printSyncEventOpCommon(p, getOperation(), getPipe(), getEventIdAttr(),
                         getEventIdDyn(), getPipeAttrName().getValue(),
                         getEventIdAttrName().getValue());
}

LogicalResult mlir::pto::SetCrossBlockOp::verify() {
  if (IntegerAttr mode = getFftsModeAttr()) {
    int64_t modeValue = mode.getInt();
    if (modeValue != 0) {
      return emitOpError() << "requires ffts_mode 0, but got "
                           << modeValue;
    }
  }
  return verifyNamedSyncEventOp(getOperation(), getPipe(), getEventIdAttr(),
                                getEventIdDyn(), 15, "pto.set_cross_block");
}

ParseResult mlir::pto::WaitCrossBlockOp::parse(OpAsmParser &parser,
                                             OperationState &result) {
  return parseSyncEventOpCommon(parser, result,
                                WaitCrossBlockOp::getPipeAttrName(result.name),
                                WaitCrossBlockOp::getEventIdAttrName(result.name));
}

void mlir::pto::WaitCrossBlockOp::print(OpAsmPrinter &p) {
  printSyncEventOpCommon(p, getOperation(), getPipe(), getEventIdAttr(),
                         getEventIdDyn(), getPipeAttrName().getValue(),
                         getEventIdAttrName().getValue());
}

LogicalResult mlir::pto::WaitCrossBlockOp::verify() {
  return verifyNamedSyncEventOp(getOperation(), getPipe(), getEventIdAttr(),
                                getEventIdDyn(), 15, "pto.wait_cross_block");
}

ParseResult mlir::pto::SetIntraBlockOp::parse(OpAsmParser &parser,
                                               OperationState &result) {
  return parseSyncEventOpCommon(parser, result,
                                SetIntraBlockOp::getPipeAttrName(result.name),
                                SetIntraBlockOp::getEventIdAttrName(result.name));
}

void mlir::pto::SetIntraBlockOp::print(OpAsmPrinter &p) {
  printSyncEventOpCommon(p, getOperation(), getPipe(), getEventIdAttr(),
                         getEventIdDyn(), getPipeAttrName().getValue(),
                         getEventIdAttrName().getValue());
}

LogicalResult mlir::pto::SetIntraBlockOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyNamedSyncEventOp(getOperation(), getPipe(), getEventIdAttr(),
                                  getEventIdDyn(), 15,
                                  "pto.set_intra_block");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyNamedSyncEventOp(getOperation(), getPipe(), getEventIdAttr(),
                                  getEventIdDyn(), 31,
                                  "pto.set_intra_block");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

ParseResult mlir::pto::WaitIntraBlockOp::parse(OpAsmParser &parser,
                                               OperationState &result) {
  return parseSyncEventOpCommon(parser, result,
                                WaitIntraBlockOp::getPipeAttrName(result.name),
                                WaitIntraBlockOp::getEventIdAttrName(result.name));
}

void mlir::pto::WaitIntraBlockOp::print(OpAsmPrinter &p) {
  printSyncEventOpCommon(p, getOperation(), getPipe(), getEventIdAttr(),
                         getEventIdDyn(), getPipeAttrName().getValue(),
                         getEventIdAttrName().getValue());
}

LogicalResult mlir::pto::WaitIntraBlockOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyNamedSyncEventOp(getOperation(), getPipe(), getEventIdAttr(),
                                  getEventIdDyn(), 15,
                                  "pto.wait_intra_block");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyNamedSyncEventOp(getOperation(), getPipe(), getEventIdAttr(),
                                  getEventIdDyn(), 31,
                                  "pto.wait_intra_block");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

using StoreTypes = std::pair<pto::TileBufType, pto::PartitionTensorViewType>;

static std::optional<int64_t> getStaticElemCount(ArrayRef<int64_t> shape) {
  int64_t total = 1;
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic || dim <= 0 ||
        total > std::numeric_limits<int64_t>::max() / dim)
      return std::nullopt;
    total *= dim;
  }
  return total;
}

static LogicalResult verifyTStoreFootprint(TStoreOp op,
                                           pto::TileBufType srcTile,
                                           pto::PartitionTensorViewType dstPart) {
  if (op.getFp())
    return success();
  auto dstCount = getStaticElemCount(dstPart.getShape());
  auto srcCount = getStaticElemCount(srcTile.getValidShape());
  if (dstCount && srcCount && *dstCount != *srcCount)
    return op.emitOpError() << "expects dst static element count (" << *dstCount
                            << ") to match src valid_shape static element count ("
                            << *srcCount << ")";
  return success();
}

static FailureOr<StoreTypes> verifyTStoreCommon(TStoreOp op,
                                                bool allowLowPrecision) {
  auto srcTile = dyn_cast<pto::TileBufType>(op.getSrc().getType());
  auto dstPart = dyn_cast<pto::PartitionTensorViewType>(op.getDst().getType());
  if (!srcTile || !dstPart) {
    op.emitOpError(
        "expects src to be !pto.tile_buf and dst to be !pto.partition_tensor_view");
    return failure();
  }
  if (failed(verifyTileBufCommon(op, srcTile, "src", allowLowPrecision)))
    return failure();
  if (op.getFp()) {
    Type fpTy = op.getFp().getType();
    if (failed(verifyTileBufCommon(op, fpTy, "fp", allowLowPrecision)))
      return failure();
    auto fpSpace = getPTOMemorySpaceEnum(fpTy);
    if (!fpSpace || *fpSpace != pto::AddressSpace::SCALING) {
      op.emitOpError("expects fp to use loc=scaling");
      return failure();
    }
  }
  if (failed(verifyShapeSign(op, dstPart.getShape(), "dst shape", true)) ||
      failed(verifyShapeSign(op, srcTile.getValidShape(), "src valid_shape",
                             false)) ||
      failed(verifyTStoreFootprint(op, srcTile, dstPart)))
    return failure();
  return StoreTypes(srcTile, dstPart);
}

static LogicalResult verifyTStoreForms(TStoreOp op, pto::AddressSpace space) {
  bool hasQuant = op.getFp() || op.getPreQuantScalar();
  if (hasQuant && space != pto::AddressSpace::ACC)
    return op.emitOpError(
        "expects fp/preQuantScalar form to use loc=acc src");
  if (op.getReluPreMode() != pto::ReluPreMode::NoRelu &&
      space != pto::AddressSpace::ACC)
    return op.emitOpError("expects reluPreMode form to use loc=acc src");
  return success();
}

static LogicalResult verifyTStoreA2VecMat(TStoreOp op,
                                          pto::TileBufType srcTile,
                                          Type dstElem) {
  if (op.getFp() || op.getPreQuantScalar())
    return op.emitOpError(
        "expects fp/preQuantScalar form to use loc=acc src");
  Type srcElem = srcTile.getElementType();
  if (isPTOLowPrecisionType(dstElem))
    return op.emitOpError(
        "expects A2/A3 vec/mat tstore low-precision dst element types to be unsupported");
  if (!isSupportedLoadStoreElemTypeA2A3(srcElem))
    return op.emitOpError(
        "expects A2/A3 vec/mat tstore src element type to be i8/i16/i32/i64/u64/f16/bf16/f32");
  if (getElemByteSize(srcElem) != getElemByteSize(dstElem))
    return op.emitOpError(
        "expects A2/A3 vec/mat tstore src and dst element types to have the same bitwidth");
  return success();
}

static LogicalResult verifyTStoreA2AccTypes(TStoreOp op, Type srcElem,
                                            Type dstElem) {
  if (!(srcElem.isInteger(32) || srcElem.isF32()))
    return op.emitOpError(
        "expects A2/A3 acc tstore src element type to be i32 or f32");
  if (op.getPreQuantScalar() && srcElem.isInteger(32) &&
      !(dstElem.isInteger(8) || dstElem.isF16()))
    return op.emitOpError(
        "expects A2/A3 acc preQuantScalar tstore dst type to be i8/ui8/f16");
  if (op.getPreQuantScalar() && srcElem.isF32() && !dstElem.isInteger(8))
    return op.emitOpError(
        "expects A2/A3 acc preQuantScalar tstore dst type to be i8/ui8");
  if (!op.getPreQuantScalar() && !op.getFp() &&
      !(dstElem.isInteger(32) || dstElem.isF32() || dstElem.isF16() ||
        dstElem.isBF16()))
    return op.emitOpError(
        "expects A2/A3 acc tstore dst element type to be i32/f32/f16/bf16");
  return success();
}

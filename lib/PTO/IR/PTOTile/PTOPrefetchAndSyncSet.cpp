// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

static LogicalResult verifyTPrefetchImpl(TPrefetchOp op,
                                         bool allowLowPrecision) {
    Type srcTy = op.getSrc().getType();
    Type dstTy = op.getDst().getType();

    Type srcElem;
    Type dstElem;

    auto srcPart = dyn_cast<pto::PartitionTensorViewType>(srcTy);
    if (!srcPart) {
      return op.emitOpError("expects src to be !pto.partition_tensor_view");
    }
    if (failed(verifyStaticShapeSign(op, srcPart.getShape(), "src shape",
                                     /*requirePositive=*/true))) {
      return failure();
    }
    srcElem = srcPart.getElementType();

    auto dstTile = dyn_cast<pto::TileBufType>(dstTy);
    if (!dstTile) {
      return op.emitOpError("expects dst to be !pto.tile_buf");
    }
    if (failed(verifyTileBufCommon(op, dstTile, "dst", allowLowPrecision))) {
      return failure();
    }
    if (failed(verifyStaticShapeSign(op, dstTile.getValidShape(),
                                     "dst valid_shape",
                                     /*requirePositive=*/false))) {
      return failure();
    }
    auto dstSpace = getPTOMemorySpaceEnum(dstTile);
    if (!dstSpace || (*dstSpace != pto::AddressSpace::VEC &&
                      *dstSpace != pto::AddressSpace::MAT)) {
      return op.emitOpError("expects dst to use loc=vec or loc=mat");
    }
    dstElem = dstTile.getElementType();

    if (getElemByteSize(srcElem) != getElemByteSize(dstElem)) {
      return op.emitOpError("expects src and dst element types to have the same element size");
    }
    if (!allowLowPrecision &&
        (isPTOLowPrecisionType(srcElem) || isPTOLowPrecisionType(dstElem))) {
      return op.emitOpError("expects A2/A3 tprefetch low-precision element types to be unsupported");
    }
    if (allowLowPrecision &&
        (!isA5TLoadStoreTransferElemType(srcElem) ||
         !isA5TLoadStoreTransferElemType(dstElem))) {
      return op.emitOpError("expects A5 tprefetch element types to be i8/i16/i32/i64/f16/bf16/f32/f8/hif8/fp4");
    }
    return success();
}

LogicalResult TPrefetchOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTPrefetchImpl(*this, /*allowLowPrecision=*/false);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTPrefetchImpl(*this, /*allowLowPrecision=*/true);
  };
  switch (getVerifierTargetArch(getOperation())) {
  case VerifierTargetArch::A2A3:
    return verifyA2A3();
  case VerifierTargetArch::A5:
    return verifyA5();
  }
  return failure();
}

LogicalResult MakePrefetchAsyncContextOp::verify() {
  auto ptrTy = dyn_cast<pto::PtrType>(getWorkspace().getType());
  if (!ptrTy) {
    return emitOpError("expects workspace to be !pto.ptr<i8>");
  }
  Type elemTy = ptrTy.getElementType();
  if (!isByteIntegerType(elemTy)) {
    return emitOpError("expects workspace element type to be an 8-bit integer");
  }
  return success();
}

LogicalResult TPrefetchAsyncOp::verify() {
  if (failed(verifyAsyncFlatContiguous1DGMViewLike(getOperation(), getSrc(),
                                                   "src"))) {
    return failure();
  }
  return success();
}

LogicalResult mlir::pto::SetFFTsOp::verify() {
  auto ptrTy = llvm::dyn_cast<mlir::pto::PtrType>(getFfts().getType());
  if (!ptrTy) {
    return emitOpError("expects a !pto.ptr operand");
  }

  if (!ptrTy.getElementType().isInteger(64) &&
      !ptrTy.getElementType().isInteger(8)) {
    return emitOpError("expects element type i64 (or i8)");
  }

  return mlir::success();
}

ParseResult mlir::pto::SyncSetOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  return parseSyncEventOpCommon(parser, result,
                                SyncSetOp::getPipeAttrName(result.name),
                                SyncSetOp::getEventIdAttrName(result.name));
}

void mlir::pto::SyncSetOp::print(OpAsmPrinter &p) {
  printSyncEventOpCommon(p, getOperation(), getPipe(), getEventIdAttr(),
                         getEventIdDyn(), getPipeAttrName().getValue(),
                         getEventIdAttrName().getValue());
}

static LogicalResult verifySyncSetWaitCommon(
    Operation *op, PipeAttr pipe, IntegerAttr eventIdAttr, Value eventIdDyn,
    IntegerAttr fftsModeAttr, StringRef opName) {
  if ((eventIdAttr != nullptr) == static_cast<bool>(eventIdDyn))
    return op->emitOpError(
        "expects exactly one event-id form: static attr or dynamic index operand");
  if (fftsModeAttr &&
      (fftsModeAttr.getInt() < 0 || fftsModeAttr.getInt() > 2))
    return op->emitOpError()
           << "requires ffts_mode in range [0, 2], but got "
           << fftsModeAttr.getInt();
  auto verifyA2A3 = [&]() -> LogicalResult { return success(); };
  auto verifyA5 = [&]() -> LogicalResult {
    if (eventIdAttr &&
        (eventIdAttr.getInt() < 0 || eventIdAttr.getInt() > 15))
      return op->emitOpError()
             << "A5 " << opName
             << " expects static FFTS event_id in [0, 15], but got "
             << eventIdAttr.getInt();
    switch (pipe.getPipe()) {
    case PIPE::PIPE_FIX:
    case PIPE::PIPE_MTE1:
    case PIPE::PIPE_MTE2:
    case PIPE::PIPE_MTE3:
    case PIPE::PIPE_V:
      return success();
    default:
      return op->emitOpError()
             << "A5 " << opName << " expects pipe to be one of "
             << "<PIPE_FIX>, <PIPE_MTE1>, <PIPE_MTE2>, <PIPE_MTE3>, <PIPE_V>";
    }
  };
  return dispatchVerifierByArch(op, verifyA2A3, verifyA5);
}

LogicalResult mlir::pto::SyncSetOp::verify() {
  return verifySyncSetWaitCommon(getOperation(), getPipe(), getEventIdAttr(),
                                 getEventIdDyn(), getFftsModeAttr(), "sync.set");
}

ParseResult mlir::pto::SyncWaitOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  return parseSyncEventOpCommon(parser, result,
                                SyncWaitOp::getPipeAttrName(result.name),
                                SyncWaitOp::getEventIdAttrName(result.name));
}

void mlir::pto::SyncWaitOp::print(OpAsmPrinter &p) {
  printSyncEventOpCommon(p, getOperation(), getPipe(), getEventIdAttr(),
                         getEventIdDyn(), getPipeAttrName().getValue(),
                         getEventIdAttrName().getValue());
}

static ParseResult parseSyncAllOperands(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
    SmallVectorImpl<Type> &operandTypes) {
  if (parser.parseLParen()) {
    return failure();
  }
  if (failed(parser.parseOptionalRParen())) {
    if (parser.parseOperandList(operands) ||
        parser.parseColonTypeList(operandTypes) || parser.parseRParen()) {
      return failure();
    }
    if (operands.size() != operandTypes.size()) {
      return parser.emitError(parser.getCurrentLocation())
             << "expects the same number of operands and operand types";
    }
  }
  return success();
}

static ParseResult parseSyncAllAttributes(OpAsmParser &parser,
                                          OperationState &result,
                                          pto::SyncAllModeAttr &mode) {
  Attribute modeAttr;
  Attribute coreTypeAttr;
  if (parser.parseKeyword("mode") || parser.parseEqual() ||
      parser.parseAttribute(modeAttr) || parser.parseComma() ||
      parser.parseKeyword("core_type") || parser.parseEqual() ||
      parser.parseAttribute(coreTypeAttr)) {
    return failure();
  }
  mode = dyn_cast<pto::SyncAllModeAttr>(modeAttr);
  if (!mode) {
    return parser.emitError(parser.getCurrentLocation())
           << "expects mode to be #pto.sync_all_mode<...>";
  }

  auto coreType = dyn_cast<pto::SyncCoreTypeAttr>(coreTypeAttr);
  if (!coreType) {
    return parser.emitError(parser.getCurrentLocation())
           << "expects core_type to be #pto.sync_core_type<...>";
  }

  result.addAttribute("mode", mode);
  result.addAttribute("core_type", coreType);
  return parser.parseOptionalAttrDict(result.attributes);
}

static ParseResult resolveSyncAllOperands(
    OpAsmParser &parser, OperationState &result, pto::SyncAllModeAttr mode,
    ArrayRef<OpAsmParser::UnresolvedOperand> operands,
    ArrayRef<Type> operandTypes) {
  switch (mode.getValue()) {
  case pto::SyncAllMode::Hard:
    if (!operands.empty()) {
      return parser.emitError(parser.getCurrentLocation())
             << "expects hard syncall to have no operands";
    }
    result.addAttribute("operandSegmentSizes",
                        parser.getBuilder().getDenseI32ArrayAttr({0, 0}));
    return success();
  case pto::SyncAllMode::Soft:
    break;
  }

  if (operands.size() != 1 && operands.size() != 2) {
    return parser.emitError(parser.getCurrentLocation())
           << "expects soft syncall to have gm_workspace and optional "
              "used_cores";
  }
  if (parser.resolveOperand(operands[0], operandTypes[0], result.operands)) {
    return failure();
  }
  if (operands.size() == 2 &&
      parser.resolveOperand(operands[1], operandTypes[1], result.operands)) {
    return failure();
  }
  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr(
          {1, operands.size() == 2 ? 1 : 0}));
  return success();
}

ParseResult mlir::pto::SyncAllOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 2> operands;
  SmallVector<Type, 2> operandTypes;
  pto::SyncAllModeAttr mode;
  if (failed(parseSyncAllOperands(parser, operands, operandTypes)) ||
      failed(parseSyncAllAttributes(parser, result, mode))) {
    return failure();
  }
  return resolveSyncAllOperands(parser, result, mode, operands, operandTypes);
}

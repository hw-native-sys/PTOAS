// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

static LogicalResult verifyTMovGeneric(TMovOp op, bool isA5) {
  if (failed(verifyTMovGenericPreconditions(op, isA5)) ||
      failed(verifyTMovGenericPairing(op, isA5)) ||
      failed(verifyTMovGenericFpForm(op, isA5))) {
    return failure();
  }
  return success();
}

static LogicalResult verifyTMovImpl(TMovOp op, bool isA5) {
  Value fp = op.getFp();
  if (fp && !getPTOMemorySpaceEnum(fp.getType())) {
    return op.emitOpError("expects the third tile to have an explicit address space");
  }
  if (classifyTMovForm(fp) == TMovForm::XToZz) {
    return verifyTMovXToZz(op, isA5);
  }
  return verifyTMovGeneric(op, isA5);
}

mlir::LogicalResult mlir::pto::TMovOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTMovImpl(*this, /*isA5=*/false);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTMovImpl(*this, /*isA5=*/true);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

// 辅助函数：获取 Rank，支持 ShapedType 和 PTO TileTypes
static int64_t getRankHelper(Type t) {
  if (auto s = dyn_cast<RankedTensorType>(t)) {
    return s.getRank();
  }
  if (auto tile = dyn_cast<pto::TileBufType>(t)) {
    return tile.getRank();
  }
  if (auto view = dyn_cast<pto::PartitionTensorViewType>(t)) {
    return view.getRank();
  }
  return -1;
}

static LogicalResult verifyMatmulLike(Operation *op, Type aTy, Type bTy, Type dstTy, bool checkRank = true) {
  // 1. 检查类型 (Tensor 或 Tile 类型)
  bool aValid = isa<RankedTensorType, pto::TileBufType, pto::PartitionTensorViewType>(aTy);
  bool bValid = isa<RankedTensorType, pto::TileBufType, pto::PartitionTensorViewType>(bTy);
  bool dValid = isa<RankedTensorType, pto::TileBufType, pto::PartitionTensorViewType>(dstTy);

  if (!aValid || !bValid || !dValid) {
    return op->emitOpError("expects inputs/outputs to be tensors or PTO tile types");
  }

  if (checkRank) {
    int64_t aRank = getRankHelper(aTy);
    int64_t bRank = getRankHelper(bTy);
    int64_t dRank = getRankHelper(dstTy);

    // 检查 Rank 一致性
    if (aRank != -1 && dRank != -1 && aRank != dRank) {
      return op->emitOpError("expects a and dst to have the same rank");
    }
    if (bRank != -1 && dRank != -1 && bRank != dRank) {
      return op->emitOpError("expects b and dst to have the same rank");
    }
  }

  return success();
}

static LogicalResult verifyScalarPointerAccess(Operation *op, Value ptr,
                                               Type valueType,
                                               StringRef valueName) {
  auto ptrType = dyn_cast<mlir::pto::PtrType>(ptr.getType());
  if (!ptrType)
    return op->emitOpError("expects ptr to be !pto.ptr type");
  if (valueType != ptrType.getElementType())
    return op->emitOpError()
           << "expects " << valueName << " type to match ptr element type";
  return success();
}

// ---- LoadScalarOp ----
LogicalResult LoadScalarOp::verify() {
  return verifyScalarPointerAccess(getOperation(), getPtr(),
                                   getValue().getType(), "result");
}
// ---- StoreScalarOp ----
LogicalResult StoreScalarOp::verify() {
  return verifyScalarPointerAccess(getOperation(), getPtr(),
                                   getValue().getType(), "value");
}

// ---- CmoCacheInvalidOp ----
static bool isGmOrDefaultAddressSpace(pto::AddressSpace space) {
  return space == pto::AddressSpace::GM || space == pto::AddressSpace::Zero;
}

static bool isGmOrDefaultCmoAddressType(Type type) {
  if (auto ptrTy = dyn_cast<mlir::pto::PtrType>(type)) {
    return isGmOrDefaultAddressSpace(ptrTy.getMemorySpace().getAddressSpace());
  }
  if (isa<mlir::pto::TensorViewType, mlir::pto::PartitionTensorViewType>(type)) {
    return true;
  }
  return false;
}

ParseResult CmoCacheInvalidOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  if (succeeded(parser.parseOptionalKeyword("all"))) {
    AddressSpaceAttr spaceAttr;
    if (parser.parseAttribute(spaceAttr, "space", result.attributes) ||
        parser.parseOptionalAttrDict(result.attributes)) {
      return failure();
    }
    return success();
  }

  OpAsmParser::UnresolvedOperand addr;
  Type addrTy;
  if (parser.parseOperand(addr) ||
      parser.parseKeyword("single_cache_line") ||
      parser.parseColonType(addrTy) ||
      parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  if (parser.resolveOperand(addr, addrTy, result.operands)) {
    return failure();
  }

  if (!result.attributes.get("space")) {
    result.addAttribute(
        "space", AddressSpaceAttr::get(parser.getContext(), AddressSpace::GM));
  }
  return success();
}

void CmoCacheInvalidOp::print(OpAsmPrinter &p) {
  if (Value addr = getAddr()) {
    p << " " << addr << " single_cache_line";
    p << " : " << addr.getType();
    p.printOptionalAttrDict((*this)->getAttrs(),
                            /*elidedAttrs=*/{"space"});
    return;
  }

  p << " all " << getSpace();
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"space"});
}

LogicalResult CmoCacheInvalidOp::verify() {
  if (!isGmOrDefaultAddressSpace(getSpace().getAddressSpace())) {
    return emitOpError("only supports GM cache maintenance");
  }

  if (Value addr = getAddr()) {
    if (!isGmOrDefaultCmoAddressType(addr.getType())) {
      return emitOpError("single_cache_line address expects a GM pointer or GM tensor view");
    }
  }

  return success();
}

// ---- GetBufOp / RlsBufOp ----
static FailureOr<pto::PIPE> getConcreteSyncPipe(Operation *op,
                                                Attribute opTypeAttr) {
  if (!opTypeAttr) {
    op->emitOpError("expects 'op_type' attribute");
    return failure();
  }
  pto::PIPE pipe = pto::PIPE::PIPE_UNASSIGNED;
  if (auto pipeAttr = dyn_cast<PipeAttr>(opTypeAttr)) {
    pipe = pipeAttr.getPipe();
  } else {
    auto opType = parseSyncOpTypeLikeAttr(opTypeAttr);
    if (failed(opType)) {
      op->emitOpError(
          "expects 'op_type' to be pipe_event_type/sync_op_type/pipe, got ")
          << opTypeAttr;
      return failure();
    }
    pipe = mapSyncOpTypeToPipe(*opType);
  }
  if (!isConcreteSyncPipe(pipe)) {
    op->emitOpError(
        "expects 'op_type' to map to a concrete pipe, not PIPE_ALL/PIPE_UNASSIGNED");
    return failure();
  }
  return pipe;
}

static LogicalResult verifyOptionalSyncMode(Operation *op,
                                            IntegerAttr modeAttr) {
  if (modeAttr && modeAttr.getInt() < 0)
    return op->emitOpError("expects 'mode' to be non-negative");
  return success();
}

static LogicalResult verifyBufSyncOp(Operation *op, Attribute opTypeAttr,
                                     IntegerAttr bufIdAttr,
                                     IntegerAttr modeAttr) {
  if (failed(getConcreteSyncPipe(op, opTypeAttr)))
    return failure();

  if (!bufIdAttr) {
    return op->emitOpError("expects 'buf_id' attribute");
  }
  int64_t bufId = bufIdAttr.getInt();
  if (bufId < 0 || bufId > 31) {
    return op->emitOpError("expects 'buf_id' in range [0, 31]");
  }

  return verifyOptionalSyncMode(op, modeAttr);
}

LogicalResult GetBufOp::verify() {
  return verifyBufSyncOp(getOperation(), getOpTypeAttr(), getBufIdAttr(),
                         getModeAttr());
}

LogicalResult RlsBufOp::verify() {
  return verifyBufSyncOp(getOperation(), getOpTypeAttr(), getBufIdAttr(),
                         getModeAttr());
}

// ---- GetBufDynOp / RlsBufDynOp ----
static LogicalResult verifyBufDynSyncOp(Operation *op, Attribute opTypeAttr,
                                        Value bufId, IntegerAttr modeAttr) {
  if (failed(getConcreteSyncPipe(op, opTypeAttr)))
    return failure();
  if (!bufId) {
    return op->emitOpError("expects 'buf_id' operand");
  }
  return verifyOptionalSyncMode(op, modeAttr);
}

LogicalResult GetBufDynOp::verify() {
  return verifyBufDynSyncOp(getOperation(), getOpTypeAttr(), getBufId(),
                            getModeAttr());
}

LogicalResult RlsBufDynOp::verify() {
  return verifyBufDynSyncOp(getOperation(), getOpTypeAttr(), getBufId(),
                            getModeAttr());
}

static ParseResult parseLegacyOrAttrMemBar(OpAsmParser &parser,
                                           MemBarAttr &attr) {
  auto loc = parser.getCurrentLocation();
  std::string token;
  if (succeeded(parser.parseOptionalString(&token))) {
    auto kind = symbolizeMemBarKind(token);
    if (!kind) {
      return parser.emitError(loc) << "invalid membar token: " << token;
    }
    attr = MemBarAttr::get(parser.getContext(), *kind);
    return success();
  }

  Attribute parsed;
  if (failed(parser.parseAttribute(parsed))) {
    return failure();
  }
  auto memBarAttr = dyn_cast<MemBarAttr>(parsed);
  if (!memBarAttr) {
    return parser.emitError(loc, "expected membar attribute");
  }
  attr = memBarAttr;
  return success();
}

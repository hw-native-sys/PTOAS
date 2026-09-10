// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This implementation fragment is included by PTO.cpp and intentionally is
// not listed as a separate CMake translation unit.

static std::optional<uint64_t>
getLocalAddressAlignmentBytes(Attribute memorySpace) {
  auto addrSpace = dyn_cast_or_null<AddressSpaceAttr>(memorySpace);
  if (!addrSpace) {
    return std::nullopt;
  }

  // Keep this verifier as a conservative front-line guard for explicit local
  // tile addresses. PTO-ISA's buffer_limits.hpp defines the baseline
  // TASSIGN<Addr> alignment as 32 bytes for local tile memories. For L0 tile
  // bases, PTOAS level3/manual IR historically uses a 4096-bit (512-byte)
  // granularity; fuller per-arch/per-layout bounds checks belong in PTO-ISA.
  switch (addrSpace.getAddressSpace()) {
  case AddressSpace::VEC:
  case AddressSpace::MAT:
  case AddressSpace::BIAS:
  case AddressSpace::SCALING:
    return 32;
  case AddressSpace::LEFT:
  case AddressSpace::RIGHT:
  case AddressSpace::ACC:
    return 512;
  case AddressSpace::GM:
  case AddressSpace::Zero:
    return std::nullopt;
  }
  return std::nullopt;
}

static LogicalResult verifyConstantLocalAddress(Operation *op, Value addr,
                                                Attribute memorySpace,
                                                int addrIndex = -1) {
  std::optional<uint64_t> alignment =
      getLocalAddressAlignmentBytes(memorySpace);
  if (!alignment || *alignment == 0) {
    return success();
  }

  std::optional<int64_t> constantAddr = mlir::getConstantIntValue(addr);
  if (!constantAddr) {
    return success();
  }

  auto emitAddrError = [&]() {
    InFlightDiagnostic diag = op->emitOpError();
    if (addrIndex >= 0) {
      diag << "addr[" << addrIndex << "]";
    } else {
      diag << "addr";
}
    return diag;
  };

  if (*constantAddr < 0) {
    return emitAddrError() << " must be non-negative, got " << *constantAddr;
  }

  uint64_t unsignedAddr = static_cast<uint64_t>(*constantAddr);
  if ((unsignedAddr % *alignment) != 0) {
    return emitAddrError()
           << " must be aligned to " << *alignment
           << " bytes for local tile memory space, got " << unsignedAddr;
  }

  return success();
}

LogicalResult AllocTileOp::verify() {
  auto ty = getResult().getType(); // TileBufType

  if (failed(verifyTileBufLayoutConstraints(*this, ty, "result"))) {
    return failure();
  }

  if (failed(verifyConstantLocalAddress(getOperation(), getAddr(),
                                        ty.getMemorySpace()))) {
    return failure();
  }

  // op 上有没有传 operands
  bool hasVR = getValidRow() != nullptr;
  bool hasVC = getValidCol() != nullptr;

  // type 上的 validShape
  auto vs = ty.getValidShape();
  if (vs.size() != 2) {
    return emitOpError("result tile_buf must have rank-2 validShape");
  }

  // TileBuf valid dims use a negative sentinel (e.g. '?' / -1). Be robust to
  // any negative value (some code may materialize MLIR dynamic sentinels).
  bool needVR = (vs[0] < 0);
  bool needVC = (vs[1] < 0);

  // 你要求的：v_row=?, v_col=? 时必须同时给两个
  // （这条规则由下面两句自然实现）
  if (hasVR != needVR) {
    return emitOpError() << "valid_row operand "
                         << (needVR ? "is required" : "must be absent")
                         << " because result type v_row is "
                         << (needVR ? "?" : std::to_string(vs[0]));
  }

  if (hasVC != needVC) {
    return emitOpError() << "valid_col operand "
                         << (needVC ? "is required" : "must be absent")
                         << " because result type v_col is "
                         << (needVC ? "?" : std::to_string(vs[1]));
  }

  return success();
}

//===----------------------------------------------------------------------===//
// AllocMultiTileOp / MultiTileGetOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyMultiTileSlot(AllocMultiTileOp op,
                                         pto::MultiTileBufType mtbTy) {
  TileBufType slotTy = mtbTy.getSlotType();
  if (!slotTy)
    return op.emitOpError("multi_tile_buf slot type must be non-null");
  Type elemTy = slotTy.getElementType();
  if (isPTOLowPrecisionType(elemTy))
    return op.emitOpError() << "slot dtype " << elemTy
                            << " is not supported by pto.alloc_multi_tile yet";
  if (failed(verifyTileBufLayoutConstraints(op, slotTy, "slot")) ||
      failed(verifyConstantLocalAddress(op, op.getAddr(),
                                        slotTy.getMemorySpace())))
    return failure();
  if (slotTy.getCompactModeI32() ==
      static_cast<int32_t>(mlir::pto::CompactMode::RowPlusOne))
    return op.emitOpError()
           << "multi_tile_buf slot uses row_plus_one compaction, whose padded "
              "storage footprint exceeds product(shape) and would overlap "
              "adjacent multi-buffer slots; use a compact (non-row_plus_one) "
              "slot layout or a single pto.alloc_tile";
  return success();
}

static LogicalResult verifyMultiTileValidShape(AllocMultiTileOp op,
                                               TileBufType slotTy) {
  auto vs = slotTy.getValidShape();
  if (vs.size() != 2)
    return op.emitOpError("slot tile_buf must have rank-2 validShape");
  bool needVR = (vs[0] < 0);
  bool needVC = (vs[1] < 0);
  if (static_cast<bool>(op.getValidRow()) != needVR)
    return op.emitOpError() << "valid_row operand "
                         << (needVR ? "is required" : "must be absent")
                         << " because slot v_row is "
                         << (needVR ? "?" : std::to_string(vs[0]));
  if (static_cast<bool>(op.getValidCol()) != needVC)
    return op.emitOpError() << "valid_col operand "
                         << (needVC ? "is required" : "must be absent")
                         << " because slot v_col is "
                         << (needVC ? "?" : std::to_string(vs[1]));
  return success();
}

static LogicalResult verifyMultiTilePlannedAddresses(AllocMultiTileOp op,
                                                     pto::MultiTileBufType mtbTy) {
  Attribute rawAddrs = op->getAttr(pto::kPtoMultiBufferAddrsAttrName);
  if (!rawAddrs)
    return success();
  uint32_t count = mtbTy.getCount();
  auto addrs = dyn_cast<DenseI64ArrayAttr>(rawAddrs);
  if (!addrs)
    return op.emitOpError() << "expects internal '" << pto::kPtoMultiBufferAddrsAttrName
                            << "' to be a dense i64 array";
  if (op.getAddr())
    return op.emitOpError() << "cannot carry both base 'addr' and internal '"
                            << pto::kPtoMultiBufferAddrsAttrName << "'";
  if (addrs.size() != count)
    return op.emitOpError() << "expects " << count
                            << " planned slot addresses, got " << addrs.size();
  TileBufType slotTy = mtbTy.getSlotType();
  uint64_t slotBytes = getPTOStorageElemByteSize(slotTy.getElementType());
  for (int64_t dim : slotTy.getShape()) {
    if (dim == ShapedType::kDynamic)
      return op.emitOpError(
          "planned multi-buffer addresses require a static slot shape");
    slotBytes *= static_cast<uint64_t>(dim);
  }
  for (auto [lhsIdx, lhs] : llvm::enumerate(addrs.asArrayRef())) {
    if (lhs < 0)
      return op.emitOpError("planned slot addresses must be non-negative");
    for (size_t rhsIdx = lhsIdx + 1;
         rhsIdx < static_cast<size_t>(addrs.size()); ++rhsIdx) {
      uint64_t lhsBegin = lhs;
      uint64_t rhsBegin = addrs[rhsIdx];
      if (std::max(lhsBegin, rhsBegin) <
          std::min(lhsBegin + slotBytes, rhsBegin + slotBytes))
        return op.emitOpError() << "planned slots " << lhsIdx << " and "
                                << rhsIdx << " overlap";
    }
  }
  return success();
}

LogicalResult AllocMultiTileOp::verify() {
  auto mtbTy = getResult().getType();
  if (!mtbTy)
    return emitOpError("result must be `!pto.multi_tile_buf`");
  if (failed(verifyMultiTileSlot(*this, mtbTy)) ||
      failed(verifyMultiTileValidShape(*this, mtbTy.getSlotType())))
    return failure();
  uint32_t count = mtbTy.getCount();
  if (count < kPtoMultiBufferMinNum || count > kPtoMultiBufferMaxNum)
    return emitOpError() << "multi_tile_buf count must be in ["
                         << kPtoMultiBufferMinNum << ", "
                         << kPtoMultiBufferMaxNum << "] (got " << count << ")";
  return verifyMultiTilePlannedAddresses(*this, mtbTy);
}

LogicalResult MultiTileGetOp::verify() {
  auto srcTy = getSource().getType();
  auto resultTy = getResult().getType();
  if (!srcTy || !resultTy) {
    return emitOpError("source and result types must be non-null");
  }

  if (srcTy.getSlotType() != resultTy) {
    return emitOpError()
           << "result tile_buf must match the multi_tile_buf slot type: "
           << "expected " << srcTy.getSlotType() << ", got " << resultTy;
  }

  // If slot is an `arith.constant`, check it is in range.
  if (auto slotDef = getSlot().getDefiningOp<arith::ConstantOp>()) {
    if (auto attr = llvm::dyn_cast<IntegerAttr>(slotDef.getValue())) {
      int64_t slotVal = attr.getValue().getSExtValue();
      int64_t count = static_cast<int64_t>(srcTy.getCount());
      if (slotVal < 0 || slotVal >= count) {
        return emitOpError()
               << "constant slot " << slotVal
               << " is out of range for multi_tile_buf count=" << count;
      }
    }
  }

  return success();
}

LogicalResult TAssignOp::verify() {
  if (getTile().getType() != getResult().getType()) {
    return emitOpError("result type must match tile operand type");
  }

  auto tileTy = dyn_cast<TileBufType>(getTile().getType());
  if (!tileTy) {
    return emitOpError("expects tile operand and result to be !pto.tile_buf");
  }

  if (failed(verifyConstantLocalAddress(getOperation(), getAddr(),
                                        tileTy.getMemorySpace()))) {
    return failure();
  }

  return success();
}

using LoadTypes =
    std::pair<pto::PartitionTensorViewType, pto::TileBufType>;

static LogicalResult verifyShapeSign(Operation *op, ArrayRef<int64_t> shape,
                                     StringRef name, bool positive) {
  for (auto [index, dim] : llvm::enumerate(shape)) {
    bool invalid = positive ? dim <= 0 : dim < 0;
    if (dim != ShapedType::kDynamic && invalid)
      return op->emitOpError() << "expects " << name << "[" << index
                               << "] to be "
                               << (positive ? "positive" : "non-negative");
  }
  return success();
}

static FailureOr<LoadTypes> verifyTLoadCommon(TLoadOp op,
                                              bool allowLowPrecision) {
  auto srcPart = dyn_cast<pto::PartitionTensorViewType>(op.getSrc().getType());
  auto dstTile = dyn_cast<pto::TileBufType>(op.getDst().getType());
  if (!srcPart || !dstTile) {
    op.emitOpError(
        "expects src to be !pto.partition_tensor_view and dst to be !pto.tile_buf");
    return failure();
  }
  if (failed(verifyTileBufCommon(op, dstTile, "dst", allowLowPrecision)) ||
      failed(verifyShapeSign(op, srcPart.getShape(), "src shape", true)) ||
      failed(verifyShapeSign(op, dstTile.getValidShape(), "dst valid_shape",
                             false)))
    return failure();
  return LoadTypes(srcPart, dstTile);
}

static LogicalResult verifyTLoadA2A3(TLoadOp op) {
  auto common = verifyTLoadCommon(op, /*allowLowPrecision=*/false);
  if (failed(common))
    return failure();
  auto [srcPart, dstTile] = *common;
  Type srcElem = srcPart.getElementType();
  Type dstElem = dstTile.getElementType();
  if (isPTOLowPrecisionType(srcElem) || isPTOLowPrecisionType(dstElem))
    return op.emitOpError(
        "expects A2/A3 tload low-precision element types to be unsupported");
  if (!isSupportedLoadStoreElemTypeA2A3(dstElem))
    return op.emitOpError(
        "expects A2/A3 tload dst element type to be i8/i16/i32/i64/u64/f16/bf16/f32");
  auto dstSpace = getPTOMemorySpaceEnum(dstTile);
  if (!dstSpace || (*dstSpace != pto::AddressSpace::VEC &&
                    *dstSpace != pto::AddressSpace::MAT))
    return op.emitOpError("expects A2/A3 tload dst to use loc=vec or loc=mat");
  if (getElemByteSize(srcElem) != getElemByteSize(dstElem))
    return op.emitOpError(
        "expects src and dst element types to have the same bitwidth");
  return success();
}

static bool hasA5LoadStoreLayout(pto::TileBufType tile) {
  int32_t bl = tile.getBLayoutValueI32();
  int32_t sl = tile.getSLayoutValueI32();
  bool isND = bl == static_cast<int32_t>(pto::BLayout::RowMajor) &&
              sl == static_cast<int32_t>(pto::SLayout::NoneBox);
  bool isDN = bl == static_cast<int32_t>(pto::BLayout::ColMajor) &&
              sl == static_cast<int32_t>(pto::SLayout::NoneBox);
  bool isNZ = bl == static_cast<int32_t>(pto::BLayout::ColMajor) &&
              sl == static_cast<int32_t>(pto::SLayout::RowMajor);
  return isND || isDN || isNZ;
}

static LogicalResult verifyTLoadA5Types(TLoadOp op, Type srcElem,
                                        pto::TileBufType dstTile) {
  Type dstElem = dstTile.getElementType();
  unsigned srcBytes = getElemByteSize(srcElem);
  unsigned dstBytes = getElemByteSize(dstElem);
  if (srcBytes != dstBytes)
    return op.emitOpError(
        "expects src and dst element types to have the same element size");
  if (!(dstBytes == 1 || dstBytes == 2 || dstBytes == 4 || dstBytes == 8))
    return op.emitOpError(
        "expects A5 tload dst element size to be 1, 2, 4, or 8 bytes");
  if (!isA5TLoadStoreTransferElemType(srcElem))
    return op.emitOpError(
        "expects A5 tload src element type to be i8/i16/i32/i64/f16/bf16/f32/f8/hif8/fp4");
  if (!isA5TLoadStoreTransferElemType(dstElem))
    return op.emitOpError(
        "expects A5 tload dst element type to be i8/i16/i32/i64/f16/bf16/f32/f8/hif8/fp4");
  auto pad = dstTile.getPadValueI32();
  if (dstElem.isInteger(64) &&
      pad != static_cast<int32_t>(pto::PadValue::Null) &&
      pad != static_cast<int32_t>(pto::PadValue::Zero))
    return op.emitOpError(
        "expects A5 i64/u64 tload dst pad to be null or zero");
  return success();
}

static LogicalResult verifyTLoadA5(TLoadOp op) {
  auto common = verifyTLoadCommon(op, /*allowLowPrecision=*/true);
  if (failed(common))
    return failure();
  auto [srcPart, dstTile] = *common;
  if (failed(verifyTLoadA5Types(op, srcPart.getElementType(), dstTile)))
    return failure();
  auto dstSpace = getPTOMemorySpaceEnum(dstTile);
  if (dstSpace && *dstSpace == pto::AddressSpace::VEC &&
      !hasA5LoadStoreLayout(dstTile))
    return op.emitOpError(
        "expects A5 tload vec dst layout to be ND, DN, or NZ");
  return success();
}

LogicalResult TLoadOp::verify() {
  auto verifyA2A3 = [&]() { return verifyTLoadA2A3(*this); };
  auto verifyA5 = [&]() { return verifyTLoadA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyStaticShapeSign(Operation *op, ArrayRef<int64_t> shape,
                                           StringRef name, bool requirePositive) {
  return verifyShapeSign(op, shape, name, requirePositive);
}

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

static LogicalResult verifyTStoreA2Acc(TStoreOp op,
                                       pto::TileBufType srcTile,
                                       Type dstElem) {
  if (failed(verifyTStoreA2AccTypes(op, srcTile.getElementType(), dstElem)))
    return failure();
  auto shape = srcTile.getShape();
  if (shape[1] != ShapedType::kDynamic &&
      (shape[1] < 1 || shape[1] > 4095))
    return op.emitOpError(
        "expects A2/A3 acc tstore src cols to be in [1, 4095]");
  auto valid = srcTile.getValidShape();
  if (valid[1] != ShapedType::kDynamic &&
      (valid[1] < 0 || valid[1] > 4095))
    return op.emitOpError(
        "expects A2/A3 acc tstore src valid_shape[1] to be in [0, 4095]");
  return success();
}

static LogicalResult verifyTStoreA2A3(TStoreOp op) {
  auto common = verifyTStoreCommon(op, /*allowLowPrecision=*/false);
  if (failed(common))
    return failure();
  auto [srcTile, dstPart] = *common;
  auto space = getPTOMemorySpaceEnum(srcTile);
  if (!space || (*space != pto::AddressSpace::VEC &&
                 *space != pto::AddressSpace::MAT &&
                 *space != pto::AddressSpace::ACC))
    return op.emitOpError(
        "expects A2/A3 tstore src to use loc=vec, loc=mat, or loc=acc");
  if (failed(verifyTStoreForms(op, *space)))
    return failure();
  if (*space == pto::AddressSpace::ACC)
    return verifyTStoreA2Acc(op, srcTile, dstPart.getElementType());
  return verifyTStoreA2VecMat(op, srcTile, dstPart.getElementType());
}

static LogicalResult verifyTStoreA5Vec(TStoreOp op,
                                       pto::TileBufType srcTile,
                                       Type dstElem) {
  if (op.getFp() || op.getPreQuantScalar())
    return op.emitOpError(
        "expects fp/preQuantScalar form to use loc=acc src");
  Type srcElem = srcTile.getElementType();
  if (!isA5TLoadStoreTransferElemType(srcElem))
    return op.emitOpError(
        "expects A5 vec tstore src element type to be i8/i16/i32/i64/f16/bf16/f32/f8/hif8/fp4");
  if (getElemByteSize(srcElem) != getElemByteSize(dstElem))
    return op.emitOpError(
        "expects A5 vec tstore src and dst element types to have the same bitwidth");
  auto shape = srcTile.getShape();
  bool special = shape.size() == 2 && (shape[0] == 1 || shape[1] == 1);
  if (!special && !hasA5LoadStoreLayout(srcTile))
    return op.emitOpError(
        "expects A5 vec tstore src layout to be ND, DN, or NZ (or special case with 1 row/col)");
  return success();
}

static LogicalResult verifyTStoreA5Acc(TStoreOp op, Type srcElem,
                                       Type dstElem) {
  if (!(srcElem.isInteger(32) || srcElem.isF32()))
    return op.emitOpError(
        "expects A5 acc tstore src element type to be i32 or f32");
  if (op.getPreQuantScalar() &&
      !isA5AccStorePreQuantDstType(srcElem, dstElem))
    return op.emitOpError(
        "expects A5 acc preQuantScalar tstore dst type to be i8/ui8/f16/bf16/f32/hif8/f8E4M3");
  if (!op.getPreQuantScalar() && !op.getFp() &&
      !(dstElem.isInteger(32) || dstElem.isF32() || dstElem.isF16() ||
        dstElem.isBF16()))
    return op.emitOpError(
        "expects A5 acc tstore dst element type to be i32/f32/f16/bf16");
  return success();
}

static LogicalResult verifyTStoreA5(TStoreOp op) {
  auto common = verifyTStoreCommon(op, /*allowLowPrecision=*/true);
  if (failed(common))
    return failure();
  auto [srcTile, dstPart] = *common;
  auto space = getPTOMemorySpaceEnum(srcTile);
  if (!space || (*space != pto::AddressSpace::VEC &&
                 *space != pto::AddressSpace::ACC))
    return op.emitOpError("expects A5 tstore src to use loc=vec or loc=acc");
  if (failed(verifyTStoreForms(op, *space)))
    return failure();
  if (*space == pto::AddressSpace::VEC)
    return verifyTStoreA5Vec(op, srcTile, dstPart.getElementType());
  return verifyTStoreA5Acc(op, srcTile.getElementType(),
                           dstPart.getElementType());
}

LogicalResult TStoreOp::verify() {
  bool hasFp = static_cast<bool>(getFp());
  bool hasPreQuant = static_cast<bool>(getPreQuantScalar());
  if (hasFp && hasPreQuant)
    return emitOpError(
        "expects fp and preQuantScalar to be mutually exclusive");
  if (hasFp && getStPhase() != pto::STPhase::Unspecified)
    return emitOpError("expects fp form to use the default stPhase");
  auto verifyA2A3 = [&]() { return verifyTStoreA2A3(*this); };
  auto verifyA5 = [&]() { return verifyTStoreA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult pto::TAbsOp::verify() {
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyVecTileUnaryOp(*this, srcTy, dstTy, "src", "dst",
                                  /*allowBf16=*/false,
                                  /*allowInt8=*/false)) ||
      failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst"))) {
    return failure();
  }
  Type elemTy = getElemTy(srcTy);
  if (!(elemTy.isF16() || elemTy.isF32())) {
    return emitOpError() << "expects element type to be f16 or f32";
  }

  return success();
}
// PTO.cpp

static bool isPTOShapedLike(Type ty) {
  return mlir::isa<RankedTensorType, pto::TensorViewType, pto::TileBufType,
                pto::PartitionTensorViewType>(ty);
}

static bool isTileLikeType(Type ty) {
  return isa<pto::TileBufType>(ty);
}

static Type getElemTy(Type ty) {
  if (auto tt = mlir::dyn_cast<RankedTensorType>(ty)) {
    return tt.getElementType();
  }
  if (auto tv = mlir::dyn_cast<pto::TensorViewType>(ty)) {
    return tv.getElementType();
  }
  if (auto tb = mlir::dyn_cast<pto::TileBufType>(ty)) {
    return tb.getElementType();
  }
  if (auto tv = mlir::dyn_cast<pto::PartitionTensorViewType>(ty)) {
    return tv.getElementType();
  }
  return Type();
}

static SmallVector<int64_t, 4> getShapeVec(Type ty) {
  SmallVector<int64_t, 4> s;
  if (auto tt = mlir::dyn_cast<RankedTensorType>(ty)) {
    return SmallVector<int64_t, 4>(tt.getShape().begin(), tt.getShape().end());
  }
  if (auto tv = mlir::dyn_cast<pto::TensorViewType>(ty)) {
    return SmallVector<int64_t, 4>(tv.getShape().begin(), tv.getShape().end());
  }
  if (auto tb = mlir::dyn_cast<pto::TileBufType>(ty)) {
    return SmallVector<int64_t, 4>(tb.getShape().begin(), tb.getShape().end());
  }
  if (auto tv = mlir::dyn_cast<pto::PartitionTensorViewType>(ty)) {
    return SmallVector<int64_t, 4>(tv.getShape().begin(), tv.getShape().end());
  }
  return {};
}

static SmallVector<int64_t, 4> getValidShapeVec(Type ty) {
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    return SmallVector<int64_t, 4>(tb.getValidShape().begin(), tb.getValidShape().end());
  }
  return getShapeVec(ty);
}

static int64_t getLogicalTileDim(int64_t rawDim, Type elemTy,
                                 std::optional<pto::BLayout> blayout,
                                 unsigned dimIdx) {
  if (rawDim == ShapedType::kDynamic || !isPTOFloat4PackedType(elemTy)) {
    return rawDim;
  }
  pto::BLayout layout = blayout.value_or(pto::BLayout::RowMajor);
  unsigned packedDim = layout == pto::BLayout::ColMajor ? 0 : 1;
  return dimIdx == packedDim ? rawDim * 2 : rawDim;
}

static std::optional<pto::BLayout> getTileBufBLayout(Type ty) {
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    return static_cast<pto::BLayout>(tb.getBLayoutValueI32());
  }
  return std::nullopt;
}

static SmallVector<int64_t, 4> getLogicalTileExtentVec(Type ty,
                                                       bool useValidShape) {
  SmallVector<int64_t, 4> dims =
      useValidShape ? getValidShapeVec(ty) : getShapeVec(ty);
  if (!isTileLikeType(ty) || dims.size() != 2) {
    return dims;
  }

  Type elemTy = getElemTy(ty);
  auto blayout = getTileBufBLayout(ty);
  for (unsigned i = 0; i < dims.size(); ++i) {
    dims[i] = getLogicalTileDim(dims[i], elemTy, blayout, i);
  }
  return dims;
}

static SmallVector<int64_t, 4> getValidShapeVec(Value value) {
  if (!value) {
    return {};
  }
  auto valid = getValidShapeVec(value.getType());
  return valid;
}

static SmallVector<int64_t, 4> getMatmulLogicalShapeVec(Type ty) {
  auto shape = getShapeVec(ty);
  auto valid = getValidShapeVec(ty);
  if (!isa<pto::TileBufType>(ty) || shape.size() != valid.size()) {
    return shape;
  }

  for (size_t i = 0, e = shape.size(); i < e; ++i) {
    if (valid[i] != ShapedType::kDynamic) {
      shape[i] = valid[i];
    }
  }
  return shape;
}

static bool isByteIntegerType(Type ty) {
  auto intTy = dyn_cast<IntegerType>(ty);
  return intTy && intTy.getWidth() == 8;
}

static FailureOr<SmallVector<int64_t, 4>> getGlobalLikeShape(
    Operation *op, Type ty, StringRef name) {
  if (!isa<pto::TensorViewType, pto::PartitionTensorViewType>(ty)) {
    op->emitOpError()
        << "expects " << name << " to be a tensor_view or partition_view";
    return failure();
  }
  SmallVector<int64_t, 4> shape = getShapeVec(ty);
  if (shape.empty()) {
    op->emitOpError() << "expects " << name << " to have rank >= 1";
    return failure();
  }
  return shape;
}

static LogicalResult verifyAsyncFlatContiguous1DGMViewLike(Operation *op,
                                                           Value value,
                                                           StringRef name) {
  auto shape = getGlobalLikeShape(op, value.getType(), name);
  if (failed(shape))
    return failure();
  for (int64_t dim : *shape) {
    if (dim == ShapedType::kDynamic) {
      return op->emitOpError() << "expects " << name
                               << " to have a static shape";
    }
  }

  bool logical1D = true;
  for (int i = 0, e = static_cast<int>(shape->size()) - 1; i < e; ++i) {
    logical1D &= (*shape)[i] == 1;
  }
  if (!logical1D) {
    return op->emitOpError()
           << "expects " << name
           << " to be a static flat contiguous logical 1D GM view";
  }

  return success();
}

enum class CommGlobalShapePolicy {
  StaticOnly,
  AllowDynamicPartitionView,
};

static LogicalResult verifyCommGlobalLike(
    Operation *op, Value value, StringRef name,
    CommGlobalShapePolicy policy = CommGlobalShapePolicy::StaticOnly) {
  Type ty = value.getType();
  auto shape = getGlobalLikeShape(op, ty, name);
  if (failed(shape))
    return failure();

  bool opAllowsDynamic =
      policy == CommGlobalShapePolicy::AllowDynamicPartitionView;
  bool isAllowedDynamicType = isa<pto::PartitionTensorViewType>(ty);
  for (int64_t dim : *shape) {
    if (dim == ShapedType::kDynamic) {
      if (!opAllowsDynamic) {
        return op->emitOpError()
               << "does not support dynamic dimensions on " << name;
      }
      if (!isAllowedDynamicType) {
        return op->emitOpError() << "allows dynamic dimensions on " << name
                                 << " only for partition_tensor_view";
      }
      continue;
    }
    if (dim <= 0) {
      return op->emitOpError() << "expects every static dimension of " << name
                               << " to be positive";
    }
  }
  return success();
}

static LogicalResult verifyCommSignalLike(Operation *op, Value value,
                                          StringRef name) {
  if (failed(verifyCommGlobalLike(op, value, name))) {
    return failure();
  }
  Type elemTy = getElemTy(value.getType());
  if (!elemTy || !elemTy.isSignlessInteger(32)) {
    return op->emitOpError() << "expects " << name
                             << " element type to be i32";
  }
  return success();
}

static LogicalResult verifyCommStagingTileLike(Operation *op, Value value,
                                               StringRef name) {
  Type ty = value.getType();
  if (!isa<pto::TileBufType>(ty)) {
    return op->emitOpError() << "expects " << name << " to be a tile_buf";
  }
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::VEC) {
    return op->emitOpError() << "expects " << name
                             << " to be in vec address space";
  }
  SmallVector<int64_t, 4> shape = getShapeVec(ty);
  if (shape.empty()) {
    return op->emitOpError() << "expects " << name << " to have rank >= 1";
  }
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic || dim <= 0) {
      return op->emitOpError() << "expects " << name
                               << " to have a positive static shape";
    }
  }
  return success();
}

static LogicalResult verifyCommGlobalGroup(Operation *op, ValueRange group,
                                           StringRef name) {
  if (group.empty()) {
    return op->emitOpError() << "expects at least one " << name << " operand";
  }
  Type groupTy = group.front().getType();
  for (auto it : llvm::enumerate(group)) {
    if (failed(verifyCommGlobalLike(op, it.value(),
                                    (name + "[" + Twine(it.index()) + "]").str()))) {
      return failure();
    }
    if (it.value().getType() != groupTy) {
      return op->emitOpError() << "expects all " << name
                               << " operands to have identical types";
    }
  }
  return success();
}

static LogicalResult verifyCommPingPongSameType(Operation *op, Value ping,
                                                Value pong, StringRef pingName,
                                                StringRef pongName) {
  if (!pong) {
    return success();
  }
  if (failed(verifyCommStagingTileLike(op, ping, pingName)) ||
      failed(verifyCommStagingTileLike(op, pong, pongName))) {
    return failure();
  }
  if (ping.getType() != pong.getType()) {
    return op->emitOpError() << "expects " << pingName << " and " << pongName
                             << " to have identical types";
  }
  return success();
}

static std::optional<uint64_t> getStaticByteSize(Type ty) {
  SmallVector<int64_t, 4> shape = getShapeVec(ty);
  if (shape.empty()) {
    return std::nullopt;
  }
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic || dim < 0) {
      return std::nullopt;
    }
  }

  Type elemTy = getElemTy(ty);
  uint64_t elemBytes = getElemByteSize(elemTy);
  if (elemBytes == 0) {
    return std::nullopt;
  }

  uint64_t total = elemBytes;
  for (int64_t dim : shape) {
    total *= static_cast<uint64_t>(dim);
  }
  return total;
}

static LogicalResult verifyTmpCapacityAtLeast(Operation *op, Type tmpTy,
                                              uint64_t requiredBytes,
                                              StringRef tmpName) {
  auto actualBytes = getStaticByteSize(tmpTy);
  if (!actualBytes) {
    return op->emitOpError()
           << "expects " << tmpName << " to have statically known byte capacity";
  }
  if (*actualBytes < requiredBytes) {
    return op->emitOpError()
           << "expects " << tmpName << " capacity to be at least "
           << requiredBytes << " bytes, but got " << *actualBytes << " bytes";
  }
  return success();
}

static std::optional<pto::AddressSpace> getPTOMemorySpaceEnum(Type ty) {
  if (auto ptr = dyn_cast<pto::PtrType>(ty)) {
    return ptr.getMemorySpace().getAddressSpace();
  }
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    if (auto as = dyn_cast_or_null<pto::AddressSpaceAttr>(tb.getMemorySpace())) {
      return as.getAddressSpace();
    }
    return std::nullopt;
  }
  return std::nullopt;
}

TMovForm mlir::pto::classifyTMovForm(Value fp) {
  if (!fp)
    return TMovForm::NoTileAux;
  auto space = getPTOMemorySpaceEnum(fp.getType());
  if (!space)
    return TMovForm::XToZz;
  return *space == pto::AddressSpace::SCALING ? TMovForm::Fp
                                              : TMovForm::XToZz;
}

[[maybe_unused]] static bool isRank2TileBuf(Type ty) {
  auto tb = dyn_cast<pto::TileBufType>(ty);
  return tb && tb.getRank() == 2 && tb.getValidShape().size() == 2;
}

static bool isSupportedVecElemType(Type ty, bool allowBf16,
                                   bool allowInt8) {
  if (ty.isF16() || ty.isF32()) {
    return true;
  }
  if (allowBf16 && ty.isBF16()) {
    return true;
  }
  if (auto it = dyn_cast<IntegerType>(ty)) {
    switch (it.getWidth()) {
    case 32:
    case 16:
      return true;
    case 8:
      return allowInt8;
    default:
      return false;
    }
  }
  return false;
}

static bool isSupportedMGatherMScatterIndexElemType(Type ty) {
  auto it = dyn_cast<IntegerType>(ty);
  if (!it || it.getWidth() != 32) {
    return false;
  }
  return true;
}

static bool isSupportedMGatherMScatterPayloadElemType(Operation *op, Type ty) {
  if (isSupportedVecElemType(ty, /*allowBf16=*/true, /*allowInt8=*/true)) {
    return true;
  }
  if (!isTargetArchA5(op)) {
    return false;
  }
  return isPTOHiFloat8Type(ty) || isPTOFloat8Type(ty);
}

static bool isSupportedMScatterAtomicPayloadElemType(Type ty,
                                                     pto::ScatterAtomicOp atomic) {
  auto intTy = dyn_cast<IntegerType>(ty);
  switch (atomic) {
  case pto::ScatterAtomicOp::None:
    return true;
  case pto::ScatterAtomicOp::Add:
    return ty.isF16() || ty.isF32() ||
           (intTy && intTy.getWidth() == 32);
  case pto::ScatterAtomicOp::Max:
  case pto::ScatterAtomicOp::Min:
    return ty.isF32() ||
           (intTy && intTy.getWidth() == 32);
  }
  llvm_unreachable("Unknown ScatterAtomicOp");
}

static LogicalResult verifyMGatherMScatterMemOperand(Operation *op,
                                                     Value memValue,
                                                     Type dataElemTy,
                                                     StringRef dataOperandLabel) {
  Type memTy = memValue.getType();
  Type memElem = getElemTy(memTy);
  if (!memElem || memElem != dataElemTy) {
    return op->emitOpError() << "expects mem element type to match "
                             << dataOperandLabel << " element type";
  }

  if (!isa<pto::PartitionTensorViewType>(memTy)) {
    return op->emitOpError("expects mem to be !pto.partition_tensor_view");
  }
  if (auto layout = getLogicalViewLayout(memValue)) {
    if (*layout != pto::Layout::ND) {
      return op->emitOpError(
          "expects mem partition view to use ND logical layout when layout "
          "can be inferred");
    }
  }
  return success();
}

static bool hasCompatibleKnownExtent(int64_t lhs, int64_t rhs);
static bool isKnownUnitExtent(int64_t value);
static bool isKnownZeroOrUnitExtent(int64_t value);
static bool hasCompatibleKnownExtentOrZero(int64_t lhs, int64_t rhs);

static LogicalResult verifyMGatherMScatterCoalesce(
    Operation *op, StringRef dataName, std::optional<pto::Coalesce> coalesce,
    bool baseRow, bool row1xR, bool rowRx1, bool elem) {
  if (!coalesce) {
    if (baseRow || elem)
      return success();
    return op->emitOpError()
           << "expects idx valid_shape to be [" << dataName
           << ".valid_row, 0|1] or match " << dataName
           << " valid_shape when coalesce is omitted";
  }
  if (*coalesce == pto::Coalesce::Row && (row1xR || rowRx1))
    return success();
  if (*coalesce == pto::Coalesce::Elem && elem)
    return success();
  if (*coalesce == pto::Coalesce::Row)
    return op->emitOpError()
           << "expects row-coalesce idx valid_shape to be [0|1, " << dataName
           << ".valid_row] or [" << dataName << ".valid_row, 0|1]";
  return op->emitOpError()
         << "expects elem-coalesce idx valid_shape to match " << dataName
         << " valid_shape";
}

static LogicalResult verifyMGatherMScatterTileShape(Operation *op, Type dataTy,
                                                    Type idxTy,
                                                    StringRef dataName,
                                                    std::optional<pto::Coalesce> coalesce) {
  auto dataValid = getValidShapeVec(dataTy);
  auto idxValid = getValidShapeVec(idxTy);
  if (dataValid.size() != 2 || idxValid.size() != 2) {
    return op->emitOpError() << "expects " << dataName
                             << " and idx to have rank-2 valid_shape";
  }

  auto idxTile = dyn_cast<pto::TileBufType>(idxTy);
  if (!idxTile) {
    return op->emitOpError("expects idx to be a tile_buf type");
  }

  const bool idxRowMajor =
      idxTile.getBLayoutValueI32() ==
      static_cast<int32_t>(pto::BLayout::RowMajor);
  const bool idxColMajor =
      idxTile.getBLayoutValueI32() ==
      static_cast<int32_t>(pto::BLayout::ColMajor);

  const bool rowCoalesce1xR =
      idxRowMajor && isKnownZeroOrUnitExtent(idxValid[0]) &&
      hasCompatibleKnownExtent(idxValid[1], dataValid[0]);
  const bool rowCoalesceRx1 =
      idxColMajor && hasCompatibleKnownExtent(idxValid[0], dataValid[0]) &&
      isKnownZeroOrUnitExtent(idxValid[1]);
  const bool baseRowCoalesce =
      idxRowMajor && hasCompatibleKnownExtent(idxValid[0], dataValid[0]) &&
      isKnownZeroOrUnitExtent(idxValid[1]);
  const bool elemCoalesce =
      hasCompatibleKnownExtent(idxValid[0], dataValid[0]) &&
      hasCompatibleKnownExtent(idxValid[1], dataValid[1]);

  return verifyMGatherMScatterCoalesce(
      op, dataName, coalesce, baseRowCoalesce, rowCoalesce1xR,
      rowCoalesceRx1, elemCoalesce);
}

template <typename AttrT>
static AttrT getPTOOpAttr(Operation *op, StringRef name) {
  if (Attribute propsAttr = op->getPropertiesAsAttribute()) {
    if (auto props = dyn_cast<DictionaryAttr>(propsAttr)) {
      if (auto attr = dyn_cast_or_null<AttrT>(props.get(name))) {
        return attr;
      }
    }
  }
  return dyn_cast_or_null<AttrT>(op->getRawDictionaryAttrs().get(name));
}

template <typename OpTy>
static ParseResult parsePTOInherentAttrs(OpAsmParser &parser,
                                         OperationState &result,
                                         NamedAttrList &parsedAttrs,
                                         ArrayRef<StringRef> inherentAttrNames) {
  auto attrLoc = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDict(parsedAttrs)) {
    return failure();
  }

  auto &properties = result.getOrAddProperties<typename OpTy::Properties>();
  OpTy::populateDefaultProperties(result.name, properties);
  if (failed(OpTy::setPropertiesFromAttr(
          properties, parsedAttrs.getDictionary(parser.getContext()), [&] {
            return parser.emitError(attrLoc)
                   << "'" << result.name.getStringRef() << "' op ";
          }))) {
    return failure();
  }

  for (StringRef attrName : inherentAttrNames) {
    parsedAttrs.erase(attrName);
  }
  result.attributes = parsedAttrs;
  return success();
}

static NamedAttrList getNonInherentAttrs(Operation *op,
                                         ArrayRef<StringRef> inherentAttrNames) {
  NamedAttrList attrs;
  for (NamedAttribute attr : op->getRawDictionaryAttrs()) {
    if (llvm::is_contained(inherentAttrNames, attr.getName().getValue())) {
      continue;
    }
    attrs.append(attr);
  }
  return attrs;
}

static pto::CoalesceAttr getMGatherCoalesceAttrIfPresent(pto::MGatherOp op) {
  return dyn_cast_or_null<pto::CoalesceAttr>(op.getProperties().coalesce);
}

static pto::GatherOOBAttr getMGatherGatherOobAttrIfPresent(pto::MGatherOp op) {
  return dyn_cast_or_null<pto::GatherOOBAttr>(op.getProperties().gatherOob);
}

static pto::GatherOOB getGatherOobOrDefault(pto::MGatherOp op) {
  if (auto attr = getMGatherGatherOobAttrIfPresent(op)) {
    return attr.getValue();
  }
  return pto::GatherOOB::Undefined;
}

static pto::CoalesceAttr getMScatterCoalesceAttrIfPresent(pto::MScatterOp op) {
  return dyn_cast_or_null<pto::CoalesceAttr>(op.getProperties().coalesce);
}

static pto::ScatterAtomicOpAttr
getMScatterScatterAtomicOpAttrIfPresent(pto::MScatterOp op) {
  return dyn_cast_or_null<pto::ScatterAtomicOpAttr>(
      op.getProperties().scatterAtomicOp);
}

static pto::ScatterOOBAttr getMScatterScatterOobAttrIfPresent(
    pto::MScatterOp op) {
  return dyn_cast_or_null<pto::ScatterOOBAttr>(op.getProperties().scatterOob);
}

static pto::ScatterConflictAttr getMScatterScatterConflictAttrIfPresent(
    pto::MScatterOp op) {
  return dyn_cast_or_null<pto::ScatterConflictAttr>(
      op.getProperties().scatterConflict);
}

static std::optional<pto::Coalesce> getCoalesceIfPresent(pto::MGatherOp op) {
  if (auto attr = getMGatherCoalesceAttrIfPresent(op)) {
    return attr.getValue();
  }
  return std::nullopt;
}

static std::optional<pto::Coalesce> getCoalesceIfPresent(pto::MScatterOp op) {
  if (auto attr = getMScatterCoalesceAttrIfPresent(op)) {
    return attr.getValue();
  }
  return std::nullopt;
}

static pto::ScatterAtomicOp getScatterAtomicOpOrDefault(pto::MScatterOp op) {
  if (auto attr = getMScatterScatterAtomicOpAttrIfPresent(op)) {
    return attr.getValue();
  }
  return pto::ScatterAtomicOp::None;
}

static pto::ScatterOOB getScatterOobOrDefault(pto::MScatterOp op) {
  if (auto attr = getMScatterScatterOobAttrIfPresent(op)) {
    return attr.getValue();
  }
  return pto::ScatterOOB::Undefined;
}

static pto::ScatterConflictAttr getScatterConflictAttrIfPresent(
    pto::MScatterOp op) {
  return getMScatterScatterConflictAttrIfPresent(op);
}

static Value getTPrintTmpIfPresent(pto::TPrintOp op) {
  return op->getNumOperands() > 1 ? op->getOperand(1) : Value();
}

static LogicalResult verifyMGatherMScatterIdxTile(Operation *op, Type ty,
                                                  StringRef name) {
  if (failed(verifyTileBufInVec(op, ty, name)))
    return failure();
  auto tb = dyn_cast<pto::TileBufType>(ty);
  if (!tb) {
    return op->emitOpError() << "expects " << name << " to be a tile_buf type";
  }
  int32_t blayout = tb.getBLayoutValueI32();
  if (blayout != static_cast<int32_t>(pto::BLayout::RowMajor) &&
      blayout != static_cast<int32_t>(pto::BLayout::ColMajor)) {
    return op->emitOpError() << "expects " << name
                             << " to use row_major or col_major blayout";
  }
  if (tb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox)) {
    return op->emitOpError() << "expects " << name
                             << " to use the none_box slayout";
  }
  return success();
}

static bool isA5TLoadStoreTransferElemType(Type ty) {
  return ty.isInteger(8) || ty.isInteger(16) || ty.isInteger(32) ||
         ty.isInteger(64) || ty.isF16() || ty.isBF16() || ty.isF32() ||
         isPTOLowPrecisionType(ty);
}

static bool isA5AccStorePreQuantDstType(Type srcElem, Type dstElem) {
  if (srcElem.isInteger(32)) {
    return dstElem.isInteger(8) || dstElem.isF16() || dstElem.isBF16();
  }
  if (!srcElem.isF32()) {
    return false;
  }
  return dstElem.isInteger(8) || dstElem.isF16() || dstElem.isBF16() ||
         dstElem.isF32() || isPTOHiFloat8Type(dstElem) ||
         isPTOFloat8E4M3LikeType(dstElem);
}

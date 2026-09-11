// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

static LogicalResult verifyTHistogramShapes(THistogramOp op,
                                            const THistogramState &state,
                                            int64_t byte) {
  auto srcShape = getShapeVec(state.src);
  auto idxShape = getShapeVec(state.idx);
  auto dstShape = getShapeVec(state.dst);
  auto srcValid = getValidShapeVec(state.src);
  auto idxValid = getValidShapeVec(state.idx);
  auto dstValid = getValidShapeVec(state.dst);
  if (srcShape.size() != 2 || idxShape.size() != 2 || dstShape.size() != 2 ||
      srcValid.size() != 2 || idxValid.size() != 2 || dstValid.size() != 2)
    return op.emitOpError(
        "expects src, idx, and dst to have rank-2 shape and valid_shape");
  if (!hasCompatibleKnownExtent(srcShape[0], dstShape[0]) ||
      !hasCompatibleKnownExtent(srcValid[0], dstValid[0]))
    return op.emitOpError("expects dst rows and valid rows to match src");
  LogicalResult idxResult = state.srcIsUi16
                                ? verifyTHistogramUi16Idx(op, state, byte)
                                : verifyTHistogramUi32Idx(op, state, byte);
  if (failed(idxResult))
    return failure();
  if (dstShape[1] != ShapedType::kDynamic && dstShape[1] < 256)
    return op.emitOpError("expects dst shape[1] to be at least 256");
  if (dstValid[1] != ShapedType::kDynamic && dstValid[1] != 0 &&
      dstValid[1] < 256)
    return op.emitOpError(
        "expects dst valid_shape[1] to be 0 or at least 256");
  return success();
}

static LogicalResult verifyTHistogramA5(THistogramOp op, int64_t byte) {
  auto state = verifyTHistogramTypes(op);
  if (failed(state))
    return failure();
  return verifyTHistogramShapes(op, *state, byte);
}

LogicalResult THistogramOp::verify() {
  auto byte = getTHistogramByte(*this);
  if (failed(byte))
    return failure();

  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("thistogram is only supported on A5");
  };
  auto verifyA5 = [&]() { return verifyTHistogramA5(*this, *byte); };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyTGetScaleAddrShape(TGetScaleAddrOp op,
                                              pto::AddressSpace srcSpace,
                                              ArrayRef<int64_t> srcShape,
                                              ArrayRef<int64_t> srcValid,
                                              ArrayRef<int64_t> dstShape,
                                              ArrayRef<int64_t> dstValid) {
  if (srcSpace == pto::AddressSpace::LEFT) {
    int64_t scaleK = ceilDivKnown(srcValid[1], 32);
    if (!hasCompatibleKnownExtent(dstShape[0], srcShape[0]) ||
        !hasCompatibleKnownExtent(dstShape[1], scaleK) ||
        !hasCompatibleKnownExtent(dstValid[0], srcValid[0]) ||
        !hasCompatibleKnownExtent(dstValid[1], scaleK))
      return op.emitOpError(
          "expects dst shape/valid_shape to be [M, ceil(K/32)]");
    return success();
  }
  int64_t scaleK = ceilDivKnown(srcValid[0], 32);
  if (!hasCompatibleKnownExtent(dstShape[0], scaleK) ||
      !hasCompatibleKnownExtent(dstShape[1], srcShape[1]) ||
      !hasCompatibleKnownExtent(dstValid[0], scaleK) ||
      !hasCompatibleKnownExtent(dstValid[1], srcValid[1]))
    return op.emitOpError(
        "expects dst shape/valid_shape to be [ceil(K/32), N]");
  return success();
}

static LogicalResult verifyTGetScaleAddrA5(TGetScaleAddrOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, srcTy, "src", true)) ||
      failed(verifyTileBufCommon(op, dstTy, "dst", true)))
    return failure();
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  if (!srcSpace || (*srcSpace != pto::AddressSpace::LEFT &&
                    *srcSpace != pto::AddressSpace::RIGHT))
    return op.emitOpError(
        "expects src to be in the left or right address space");
  if (!dstSpace || *dstSpace != pto::AddressSpace::SCALING)
    return op.emitOpError("expects dst to be in the scaling address space");
  auto srcShape = getShapeVec(srcTy);
  auto dstShape = getShapeVec(dstTy);
  auto srcValid = getValidShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (srcShape.size() != 2 || dstShape.size() != 2 || srcValid.size() != 2 ||
      dstValid.size() != 2)
    return op.emitOpError(
        "expects src/dst to have rank-2 shape and valid_shape");
  return verifyTGetScaleAddrShape(op, *srcSpace, srcShape, srcValid, dstShape,
                                  dstValid);
}

LogicalResult TGetScaleAddrOp::verify() {
  auto verifyA2A3 = [&]() {
    return emitOpError("tget_scale_addr is only supported on A5");
  };
  auto verifyA5 = [&]() { return verifyTGetScaleAddrA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

// ---- MScatterOp ----
ParseResult mlir::pto::MScatterOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  OpAsmParser::UnresolvedOperand src;
  OpAsmParser::UnresolvedOperand idx;
  OpAsmParser::UnresolvedOperand mem;
  Type srcTy, idxTy, memTy;
  NamedAttrList parsedAttrs;

  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(src) || parser.parseComma() ||
      parser.parseOperand(idx) || parser.parseColonType(srcTy) ||
      parser.parseComma() || parser.parseType(idxTy) || parser.parseRParen() ||
      parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(mem) || parser.parseColonType(memTy) ||
      parser.parseRParen() ||
      parsePTOInherentAttrs<MScatterOp>(
          parser, result, parsedAttrs,
          {"coalesce", "scatterAtomicOp", "scatterOob", "scatterConflict"})) {
    return failure();
  }

  if (parser.resolveOperand(src, srcTy, result.operands) ||
      parser.resolveOperand(idx, idxTy, result.operands) ||
      parser.resolveOperand(mem, memTy, result.operands)) {
    return failure();
  }
  return success();
}

void mlir::pto::MScatterOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc() << ", " << getIdx() << " : "
    << getSrc().getType() << ", ";
  p.printStrippedAttrOrType(getIdx().getType());
  p << ") outs(" << getMem() << " : ";
  p.printStrippedAttrOrType(getMem().getType());
  p << ")";

  NamedAttrList attrs = getNonInherentAttrs(
      getOperation(),
      {"coalesce", "scatterAtomicOp", "scatterOob", "scatterConflict"});
  if (auto coalesceAttr = getMScatterCoalesceAttrIfPresent(*this)) {
    attrs.append("coalesce", coalesceAttr);
  }
  if (auto scatterAtomicAttr = getMScatterScatterAtomicOpAttrIfPresent(*this);
      scatterAtomicAttr &&
      scatterAtomicAttr.getValue() != pto::ScatterAtomicOp::None) {
    attrs.append("scatterAtomicOp", scatterAtomicAttr);
  }
  if (auto scatterOobAttr = getMScatterScatterOobAttrIfPresent(*this);
      scatterOobAttr &&
      scatterOobAttr.getValue() != pto::ScatterOOB::Undefined) {
    attrs.append("scatterOob", scatterOobAttr);
  }
  if (auto scatterConflictAttr =
          getMScatterScatterConflictAttrIfPresent(*this)) {
    attrs.append("scatterConflict", scatterConflictAttr);
  }
  p.printOptionalAttrDict(attrs.getAttrs());
}
static LogicalResult verifyMScatterAttrs(
    MScatterOp op, Type srcElem, std::optional<pto::Coalesce> coalesce) {
  pto::ScatterAtomicOp atomic = getScatterAtomicOpOrDefault(op);
  pto::ScatterOOB oob = getScatterOobOrDefault(op);
  if (!coalesce &&
      (atomic != pto::ScatterAtomicOp::None ||
       oob != pto::ScatterOOB::Undefined ||
       getScatterConflictAttrIfPresent(op)))
    return op.emitOpError(
        "expects coalesce when scatterAtomicOp/scatterOob/scatterConflict is specified");
  if (getScatterConflictAttrIfPresent(op) &&
      !isTargetArchA5(op.getOperation()))
    return op.emitOpError("expects scatterConflict only on A5 targets");
  if (!isSupportedMScatterAtomicPayloadElemType(srcElem, atomic))
    return op.emitOpError(
        "expects scatterAtomicOp-compatible src element type: add supports "
        "i32/ui32/f16/f32, max/min support signless i32/f32");
  return success();
}

LogicalResult MScatterOp::verify() {
  Type srcTy = getSrc().getType();
  Type idxTy = getIdx().getType();
  Type memTy = getMem().getType();

  if (getPTOTypeRank(srcTy) == -1 || getPTOTypeRank(idxTy) == -1 ||
      getPTOTypeRank(memTy) == -1) {
    return emitOpError("expects src, idx, and mem to use supported PTO shapes");
  }

  if (failed(verifyNDStyleVecTile(
          *this, srcTy, "src",
          /*allowLowPrecision=*/isTargetArchA5(getOperation()))) ||
      failed(verifyMGatherMScatterIdxTile(getOperation(), idxTy, "idx"))) {
    return failure();
  }

  auto coalesce = getCoalesceIfPresent(*this);

  Type srcElem = getElemTy(srcTy);
  Type idxElem = getElemTy(idxTy);
  if (!srcElem || !idxElem) {
    return emitOpError("failed to resolve element types for src or idx");
  }

  if (!isSupportedMGatherMScatterPayloadElemType(getOperation(), srcElem)) {
    return emitOpError(
        "expects src element type to be i8/ui8/i16/ui16/i32/ui32/f16/bf16/f32 "
        "(and on A5 targets also float8_e4m3/float8_e5m2 family types)");
  }

  if (!isSupportedMGatherMScatterIndexElemType(idxElem)) {
    return emitOpError("expects idx element type to be signless i32");
  }

  if (failed(verifyMGatherMScatterMemOperand(getOperation(), getMem(), srcElem,
                                             "src"))) {
    return failure();
  }

  if (failed(verifyMGatherMScatterTileShape(getOperation(), srcTy, idxTy, "src",
                                            coalesce))) {
    return failure();
  }

  return verifyMScatterAttrs(*this, srcElem, coalesce);
}

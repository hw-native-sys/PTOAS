// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

// ---- MGatherOp ----
// GM -> L1 (cube Mat) gather verifier. The destination is an L1 (loc=mat) tile
// in NZ layout; the index is a GM tensor (the cube core cannot read UB on A5),
// and Coalesce::Elem carries a contiguous GM scratch workspace. Mirrors the
// pto-isa MGATHER GM -> L1 overloads / MGatherCheckGm2L1.
static FailureOr<Type> verifyMGatherGm2L1Dst(Operation *op, Value dst) {
  Type dstTy = dst.getType();
  auto dstTb = dyn_cast<pto::TileBufType>(dstTy);
  if (!dstTb)
    return op->emitOpError("expects GM->L1 mgather dst to be a tile_buf");
  if (!isColMajorRowMajorNZTileBuf(dstTb))
    return op->emitOpError("expects GM->L1 mgather dst (loc=mat) to use "
                           "blayout=col_major and slayout=row_major (NZ)");
  if (dstTb.getSFractalSizeI32() != 512)
    return op->emitOpError("expects GM->L1 mgather dst fractal size to be 512");
  Type dstElem = getElemTy(dstTy);
  if (!dstElem)
    return op->emitOpError("failed to resolve GM->L1 mgather dst element type");
  if (!isSupportedMGatherMScatterPayloadElemType(op, dstElem))
    return op->emitOpError(
        "expects GM->L1 mgather dst element type to be "
        "i8/ui8/i16/ui16/i32/ui32/f16/bf16/f32 (and on A5 targets also "
        "float8_e4m3/float8_e5m2 family types)");
  unsigned elemBytes =
      std::max<unsigned>(1u, dstElem.getIntOrFloatBitWidth() / 8u);
  int64_t kC0 = 32 / static_cast<int64_t>(elemBytes);
  auto dstShape = getShapeVec(dstTy);
  if (dstShape.size() == 2) {
    if (kC0 > 0 && dstShape[1] != ShapedType::kDynamic &&
        dstShape[1] % kC0 != 0) {
      return op->emitOpError()
             << "expects GM->L1 mgather dst padded cols to be a multiple of "
             << kC0 << " (C0 = 32 / sizeof(elem))";
    }
    if (dstShape[0] != ShapedType::kDynamic && dstShape[0] % 16 != 0) {
      return op->emitOpError("expects GM->L1 mgather dst padded rows to be a "
                             "multiple of 16 (FRACTAL_NZ_ROW)");
    }
  }

  return dstElem;
}

static LogicalResult verifyMGatherGm2L1Idx(Operation *op, Value idx) {
  Type idxTy = idx.getType();
  if (isa<pto::TileBufType>(idxTy))
    return op->emitOpError("expects GM->L1 mgather idx to be a GM tensor "
                           "partition_tensor_view, not a tile_buf");
  if (!isa<pto::PartitionTensorViewType>(idxTy))
    return op->emitOpError(
        "expects GM->L1 mgather idx to be a partition_tensor_view");
  Type idxElem = getElemTy(idxTy);
  if (!idxElem || !isSupportedMGatherMScatterIndexElemType(idxElem))
    return op->emitOpError("expects GM->L1 mgather idx element type to be i32");
  return success();
}

static LogicalResult verifyMGatherGm2L1Scratch(
    Operation *op, Value scratch, Type dstElem,
    std::optional<pto::Coalesce> coalesce) {
  if (!coalesce)
    return op->emitOpError("expects GM->L1 mgather to specify an explicit "
                           "coalesce attribute (row or elem)");
  if (*coalesce == pto::Coalesce::Elem) {
    if (!scratch)
      return op->emitOpError("expects GM->L1 mgather with coalesce=elem to "
                             "provide a GM scratch operand");
    Type scTy = scratch.getType();
    if (!isa<pto::PartitionTensorViewType>(scTy))
      return op->emitOpError(
          "expects GM->L1 mgather scratch to be a partition_tensor_view");
    Type scElem = getElemTy(scTy);
    if (!scElem || scElem != dstElem)
      return op->emitOpError("expects GM->L1 mgather scratch element type to "
                             "match dst element type");
    return success();
  }
  if (scratch)
    return op->emitOpError("expects GM->L1 mgather with coalesce=row to omit "
                           "the scratch operand");
  return success();
}

static LogicalResult verifyMGatherGm2L1(Operation *op, Value mem, Value idx,
                                        Value dst, Value scratch,
                                        std::optional<pto::Coalesce> coalesce) {
  auto dstElem = verifyMGatherGm2L1Dst(op, dst);
  if (failed(dstElem) ||
      failed(verifyMGatherMScatterMemOperand(op, mem, *dstElem, "dst")) ||
      failed(verifyMGatherGm2L1Idx(op, idx)))
    return failure();
  return verifyMGatherGm2L1Scratch(op, scratch, *dstElem, coalesce);
}
static ParseResult parseMGatherInputs(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
    SmallVectorImpl<Type> &types) {
  if (parser.parseKeyword("ins") || parser.parseLParen()) {
    return failure();
  }
  do {
    OpAsmParser::UnresolvedOperand operand;
    if (parser.parseOperand(operand)) {
      return failure();
    }
    operands.push_back(operand);
  } while (succeeded(parser.parseOptionalComma()));
  if (operands.size() < 2 || operands.size() > 3) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expects mgather ins(mem, idx[, scratch])");
  }
  if (parser.parseColon()) {
    return failure();
  }
  do {
    Type type;
    if (parser.parseType(type)) {
      return failure();
    }
    types.push_back(type);
  } while (succeeded(parser.parseOptionalComma()));
  if (operands.size() != types.size()) {
    return parser.emitError(
        parser.getCurrentLocation(),
        "expects the number of ins operands to match the number of ins types");
  }
  return success();
}

static ParseResult resolveMGatherOperands(
    OpAsmParser &parser, OperationState &result,
    ArrayRef<OpAsmParser::UnresolvedOperand> inputs, ArrayRef<Type> inputTypes,
    OpAsmParser::UnresolvedOperand dst, Type dstTy) {
  if (parser.resolveOperand(inputs[0], inputTypes[0], result.operands) ||
      parser.resolveOperand(inputs[1], inputTypes[1], result.operands) ||
      parser.resolveOperand(dst, dstTy, result.operands)) {
    return failure();
  }
  if (inputs.size() == 3 &&
      parser.resolveOperand(inputs[2], inputTypes[2], result.operands)) {
    return failure();
  }
  return success();
}

ParseResult mlir::pto::MGatherOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 3> insOperands;
  SmallVector<Type, 3> insTypes;
  OpAsmParser::UnresolvedOperand dst;
  Type dstTy;
  NamedAttrList parsedAttrs;

  if (failed(parseMGatherInputs(parser, insOperands, insTypes))) {
    return failure();
  }

  if (parser.parseRParen() || parser.parseKeyword("outs") ||
      parser.parseLParen() || parser.parseOperand(dst) ||
      parser.parseColonType(dstTy) || parser.parseRParen() ||
      parsePTOInherentAttrs<MGatherOp>(
          parser, result, parsedAttrs, {"coalesce", "gatherOob"})) {
    return failure();
  }

  if (failed(resolveMGatherOperands(parser, result, insOperands, insTypes, dst,
                                    dstTy))) {
    return failure();
  }
  return success();
}

void mlir::pto::MGatherOp::print(OpAsmPrinter &p) {
  p << " ins(" << getMem() << ", " << getIdx();
  if (auto scratch = getScratch()) {
    p << ", " << scratch;
  }
  p << " : ";
  p.printStrippedAttrOrType(getMem().getType());
  p << ", ";
  p.printStrippedAttrOrType(getIdx().getType());
  if (auto scratch = getScratch()) {
    p << ", ";
    p.printStrippedAttrOrType(scratch.getType());
  }
  p << ") outs(" << getDst() << " : " << getDst().getType() << ")";

  NamedAttrList attrs =
      getNonInherentAttrs(getOperation(), {"coalesce", "gatherOob"});
  if (auto coalesceAttr = getMGatherCoalesceAttrIfPresent(*this)) {
    attrs.append("coalesce", coalesceAttr);
  }
  if (auto gatherOobAttr = getMGatherGatherOobAttrIfPresent(*this);
      gatherOobAttr &&
      gatherOobAttr.getValue() != pto::GatherOOB::Undefined) {
    attrs.append("gatherOob", gatherOobAttr);
  }
  p.printOptionalAttrDict(attrs.getAttrs());
}

static LogicalResult verifyMGatherGm2Ub(MGatherOp op) {
  Type idxTy = op.getIdx().getType();
  Type dstTy = op.getDst().getType();
  if (op.getScratch())
    return op.emitOpError(
        "expects scratch operand only on GM->L1 (loc=mat) mgather");
  if (failed(verifyNDStyleVecTile(
          op, dstTy, "dst",
          /*allowLowPrecision=*/isTargetArchA5(op.getOperation()))) ||
      failed(verifyMGatherMScatterIdxTile(op, idxTy, "idx")))
    return failure();
  auto coalesce = getCoalesceIfPresent(op);
  Type dstElem = getElemTy(dstTy);
  Type idxElem = getElemTy(idxTy);
  if (!dstElem || !idxElem)
    return op.emitOpError("failed to resolve element types for dst or idx");
  if (!isSupportedMGatherMScatterPayloadElemType(op, dstElem))
    return op.emitOpError(
        "expects dst element type to be i8/ui8/i16/ui16/i32/ui32/f16/bf16/f32 "
        "(and on A5 targets also float8_e4m3/float8_e5m2 family types)");
  if (!isSupportedMGatherMScatterIndexElemType(idxElem))
    return op.emitOpError("expects idx element type to be signless i32");
  if (failed(verifyMGatherMScatterMemOperand(op, op.getMem(), dstElem, "dst")) ||
      failed(verifyMGatherMScatterTileShape(op, dstTy, idxTy, "dst", coalesce)))
    return failure();
  if (getGatherOobOrDefault(op) != pto::GatherOOB::Undefined && !coalesce)
    return op.emitOpError("expects coalesce when gatherOob is specified");
  return success();
}

LogicalResult MGatherOp::verify() {
  Type memTy = getMem().getType();
  Type idxTy = getIdx().getType();
  Type dstTy = getDst().getType();
  if (getPTOTypeRank(memTy) == -1 || getPTOTypeRank(idxTy) == -1 ||
      getPTOTypeRank(dstTy) == -1)
    return emitOpError("expects mem, idx, and dst to use supported PTO shapes");
  auto space = getPTOMemorySpaceEnum(dstTy);
  if (isa<pto::TileBufType>(dstTy) && space &&
      *space == pto::AddressSpace::MAT) {
    std::optional<pto::Coalesce> coalesce;
    if (auto coalesceAttr = getCoalesceAttr()) {
      coalesce = coalesceAttr.getValue();
    }
    return verifyMGatherGm2L1(getOperation(), getMem(), getIdx(), getDst(),
                              getScratch(), coalesce);
  }
  return verifyMGatherGm2Ub(*this);
}

void mlir::pto::TCvtOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc();
  if (getTmp()) {
    p << ", " << getTmp();
  }
  Builder builder(getContext());
  NamedAttrList attrs;
  for (auto attr : (*this)->getAttrs()) {
    if (attr.getName() == "sat_mode") {
      attrs.set(builder.getStringAttr("satmode"), attr.getValue());
      continue;
    }
    attrs.set(attr.getName(), attr.getValue());
  }
  p.printOptionalAttrDict(attrs.getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes"});
  p << " : " << getSrc().getType();
  if (getTmp()) {
    p << ", " << getTmp().getType();
  }
  p << ") outs(" << getDst() << " : " << getDst().getType() << ")";
}

// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

static ParseResult parseBufSyncOp(OpAsmParser &parser, OperationState &result) {
  Attribute opTypeAttr;
  IntegerAttr bufIdAttr;
  IntegerAttr modeAttr;

  auto loc = parser.getCurrentLocation();
  std::string token;
  if (succeeded(parser.parseOptionalString(&token))) {
    if (auto pipe = symbolizePIPE(token)) {
      opTypeAttr = PipeAttr::get(parser.getContext(), *pipe);
    } else if (auto opType = symbolizeSyncOpType(token)) {
      opTypeAttr = PipeEventTypeAttr::get(parser.getContext(), *opType);
    } else {
      return parser.emitError(loc) << "invalid get_buf/rls_buf token: " << token;
}

    if (parser.parseComma() || parseI32LiteralAttr(parser, bufIdAttr)) {
      return failure();
    }
    if (failed(parseOptionalSyncMode(parser, modeAttr)))
      return failure();
  } else if (succeeded(parser.parseOptionalLSquare())) {
    if (parser.parseAttribute(opTypeAttr) || parser.parseComma() ||
        parseI32LiteralAttr(parser, bufIdAttr)) {
      return failure();
    }
    if (failed(parseOptionalSyncMode(parser, modeAttr)))
      return failure();
    if (parser.parseRSquare()) {
      return failure();
    }
  } else {
    return parser.emitError(loc, "expected string pipe/op_type or '['");
  }

  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }
  result.addAttribute("op_type", opTypeAttr);
  result.addAttribute("buf_id", bufIdAttr);
  result.addAttribute("mode", modeAttr);
  return success();
}

static void printBufSyncOp(OpAsmPrinter &p, Attribute opTypeAttr,
                           IntegerAttr bufIdAttr, IntegerAttr modeAttr,
                           ArrayRef<NamedAttribute> attrs) {
  if (auto pipeAttr = dyn_cast<PipeAttr>(opTypeAttr)) {
    p << " \"" << stringifyPIPE(pipeAttr.getPipe()) << "\", "
      << bufIdAttr.getInt() << ", " << modeAttr.getInt();
  } else if (auto pipeEventType = dyn_cast<PipeEventTypeAttr>(opTypeAttr)) {
    p << "[" << opTypeAttr << ", " << bufIdAttr.getInt() << ", "
      << modeAttr.getInt() << "]";
  } else if (auto syncOpType = dyn_cast<SyncOpTypeAttr>(opTypeAttr)) {
    p << "[" << opTypeAttr << ", " << bufIdAttr.getInt() << ", "
      << modeAttr.getInt() << "]";
  } else {
    p << "[" << opTypeAttr << ", " << bufIdAttr.getInt() << ", "
      << modeAttr.getInt() << "]";
  }
  p.printOptionalAttrDict(attrs, {"op_type", "buf_id", "mode"});
}

ParseResult GetBufOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseBufSyncOp(parser, result);
}

void GetBufOp::print(OpAsmPrinter &p) {
  printBufSyncOp(p, getOpTypeAttr(), getBufIdAttr(), getModeAttr(),
                 (*this)->getAttrs());
}

ParseResult RlsBufOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseBufSyncOp(parser, result);
}

void RlsBufOp::print(OpAsmPrinter &p) {
  printBufSyncOp(p, getOpTypeAttr(), getBufIdAttr(), getModeAttr(),
                 (*this)->getAttrs());
}

// ---- GetBufDynOp / RlsBufDynOp parse/print ----
static ParseResult parseBufDynSyncOp(OpAsmParser &parser,
                                     OperationState &result) {
  Attribute opTypeAttr;
  IntegerAttr modeAttr;
  auto loc = parser.getCurrentLocation();
  std::string token;
  bool bracketed = false;
  if (succeeded(parser.parseOptionalString(&token))) {
    if (auto pipe = symbolizePIPE(token)) {
      opTypeAttr = PipeAttr::get(parser.getContext(), *pipe);
    } else if (auto opType = symbolizeSyncOpType(token)) {
      opTypeAttr = PipeEventTypeAttr::get(parser.getContext(), *opType);
    } else {
      return parser.emitError(loc)
             << "invalid get_buf_dyn/rls_buf_dyn token: " << token;
    }
  } else if (succeeded(parser.parseOptionalLSquare())) {
    bracketed = true;
    if (parser.parseAttribute(opTypeAttr)) {
      return failure();
    }
  } else {
    return parser.emitError(loc, "expected string pipe/op_type or '['");
  }
  OpAsmParser::UnresolvedOperand bufOperand;
  if (parser.parseComma() || parser.parseOperand(bufOperand) ||
      parser.resolveOperand(bufOperand, parser.getBuilder().getIndexType(),
                            result.operands)) {
    return failure();
  }
  if (succeeded(parser.parseOptionalComma())) {
    if (parseI32LiteralAttr(parser, modeAttr)) {
      return failure();
    }
  } else {
    modeAttr = IntegerAttr::get(IntegerType::get(parser.getContext(), 32), 0);
  }
  if ((bracketed && parser.parseRSquare()) ||
      parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }
  result.addAttribute("op_type", opTypeAttr);
  result.addAttribute("mode", modeAttr);
  return success();
}

static void printBufDynSyncOp(OpAsmPrinter &p, Attribute opTypeAttr,
                              Value bufId, IntegerAttr modeAttr,
                              ArrayRef<NamedAttribute> attrs) {
  if (auto pipeAttr = dyn_cast<PipeAttr>(opTypeAttr)) {
    p << " \"" << stringifyPIPE(pipeAttr.getPipe()) << "\", " << bufId << ", "
      << modeAttr.getInt();
  } else {
    p << "[" << opTypeAttr << ", " << bufId << ", " << modeAttr.getInt()
      << "]";
  }
  p.printOptionalAttrDict(attrs, {"op_type", "mode"});
}

ParseResult GetBufDynOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseBufDynSyncOp(parser, result);
}

void GetBufDynOp::print(OpAsmPrinter &p) {
  printBufDynSyncOp(p, getOpTypeAttr(), getBufId(), getModeAttr(),
                    (*this)->getAttrs());
}

ParseResult RlsBufDynOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseBufDynSyncOp(parser, result);
}

void RlsBufDynOp::print(OpAsmPrinter &p) {
  printBufDynSyncOp(p, getOpTypeAttr(), getBufId(), getModeAttr(),
                    (*this)->getAttrs());
}
// ---- TOp ----
static LogicalResult verifyMatBiasCommon(Operation *op, Type a, Type b,
                                         Type bias, Type dst, bool isGemv,
                                         bool allowLowPrecision = false) {
  LogicalResult operands =
      isGemv ? verifyGemvTileOperands(op, a, b, dst)
             : verifyMatTileOperands(op, a, b, dst, allowLowPrecision);
  if (failed(operands) || failed(verifyMatBiasTile(op, bias, dst)) ||
      failed(verifyMatmulTypeTriple(op, getElemTy(a), getElemTy(b),
                                    getElemTy(dst))))
    return failure();
  return verifyMatmulLike(op, a, b, dst);
}

LogicalResult TGemvBiasOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyMatBiasCommon(getOperation(), getA().getType(),
                               getB().getType(), getBias().getType(),
                               getDst().getType(), true);
  };
  auto verifyA5 = [&]() -> LogicalResult { return verifyA2A3(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyA5MxGemvOperands(Operation *op, Type a, Type b,
                                            Type dst, Type aScale,
                                            Type bScale) {
  if (failed(verifyA5MxGemvTileOperands(op, a, b, dst)) ||
      failed(verifyA5MxGemvScaleTile(op, aScale, a, b, "a_scale",
                                     /*isLeftScale=*/true)) ||
      failed(verifyA5MxGemvScaleTile(op, bScale, a, b, "b_scale",
                                     /*isLeftScale=*/false)))
    return failure();
  return success();
}

static LogicalResult verifyA5Only(
    Operation *op, StringRef opName,
    llvm::function_ref<LogicalResult()> verifyA5Body) {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return op->emitOpError() << opName << " is only supported on A5 targets";
  };
  return dispatchVerifierByArch(op, verifyA2A3, verifyA5Body);
}

static LogicalResult verifyA5MxTypeAndMatmulShape(Operation *op, Type a,
                                                  Type b, Type dst) {
  if (failed(verifyA5MxTypeTriple(op, a, b, dst, "lhs", "rhs", "dst")))
    return failure();
  return verifyMatmulLike(op, a, b, dst);
}

static LogicalResult verifyA5MxAccumulator(Operation *op, Type cIn,
                                           Type dst) {
  if (failed(verifyTileBufSameElemType(op, cIn, dst, "c_in", "dst")) ||
      failed(verifyTileBufSameValidShape(op, cIn, dst, "c_in", "dst")))
    return failure();
  return success();
}

static LogicalResult verifyA5MxAccCommon(Operation *op, Type a, Type b,
                                         Type cIn, Type dst, Type aScale,
                                         Type bScale, bool isGemv);
static LogicalResult verifyA5MxBiasBase(Operation *op, Type a, Type b,
                                        Type bias, Type dst, Type aScale,
                                        Type bScale, bool isGemv);

LogicalResult TGemvMxOp::verify() {
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyA5MxGemvOperands(
            getOperation(), getA().getType(), getB().getType(),
            getDst().getType(), getAScale().getType(),
            getBScale().getType())))
      return failure();
    return verifyA5MxTypeAndMatmulShape(
        getOperation(), getA().getType(), getB().getType(), getDst().getType());
  };
  return verifyA5Only(getOperation(), "tgemv.mx", verifyA5);
}

LogicalResult TGemvMxAccOp::verify() {
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyA5MxAccCommon(
        getOperation(), getA().getType(), getB().getType(), getCIn().getType(),
        getDst().getType(), getAScale().getType(), getBScale().getType(), true);
  };
  return verifyA5Only(getOperation(), "tgemv.mx.acc", verifyA5);
}

LogicalResult TGemvMxBiasOp::verify() {
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyA5MxBiasBase(
            getOperation(), getA().getType(), getB().getType(),
            getBias().getType(), getDst().getType(), getAScale().getType(),
            getBScale().getType(), true)))
      return failure();
    auto biasShape = getShapeVec(getBias().getType());
    auto dstShape = getShapeVec(getDst().getType());
    if (biasShape.size() != 2 || dstShape.size() != 2) {
      return emitOpError("expects bias and dst to be rank-2 for tgemv.mx.bias");
    }
    if (biasShape[1] != ShapedType::kDynamic && dstShape[1] != ShapedType::kDynamic &&
        biasShape[1] != dstShape[1]) {
      return emitOpError("expects bias and dst to have the same column shape");
    }
    if (failed(verifyTileBufSameValidShape(*this, getBias().getType(),
                                           getDst().getType(), "bias", "dst"))) {
      return failure();
    }
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  return verifyA5Only(getOperation(), "tgemv.mx.bias", verifyA5);
}

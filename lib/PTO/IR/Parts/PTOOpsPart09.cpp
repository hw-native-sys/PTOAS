// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This implementation fragment is included by PTO.cpp and intentionally is
// not listed as a separate CMake translation unit.

void TScatterOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  if (getIndexes()) {
    auto idx = getIndexesMutable();
    if (!idx.empty())
      PTO_ADD_READ(idx[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

// Select: Read(mask, src0, src1) -> Write(tmp on A2/A3, dst)
void TSelOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getMaskMutable());
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  // A5 lowering does not consume tmp for TSEL; modeling tmp as a scratch
  // write inflates local-memory planning and can trigger false vec-overflow
  // diagnostics.
  if (getTargetArch(getOperation()) != PTOArch::A5)
    PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

// TSELS: Read(mask, src) -> Write(tmp on A2/A3, dst)
void TSelSOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getMaskMutable());
  PTO_ADD_READ(getSrcMutable());
  // A5 lowering does not consume tmp for TSELS; modeling tmp as a scratch
  // write inflates local-memory planning and can trigger false vec-overflow
  // diagnostics.
  if (getTargetArch(getOperation()) != PTOArch::A5)
    PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_BINARY_EFFECTS(TShlOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TShrOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TShlSOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TShrSOp, getSrcMutable(), getDstMutable())

// TSORT32: Read(src, idx) -> Write(dst [, tmp])
void TSort32Op::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getIdxMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty())
    PTO_ADD_WRITE(tmp[0]);
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_UNARY_EFFECTS(TSqrtOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TSubOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_TERNARY_EFFECTS(TSubCOp, getSrc0Mutable(), getSrc1Mutable(), getSrc2Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TSubSOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TSubSCOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())

// TXORS: Read(src) -> Write(tmp on A2/A3, dst)
void TXorSOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  // A5 lowering does not consume tmp for TXORS; modeling tmp as a scratch
  // write inflates local-memory planning and can trigger false vec-overflow
  // diagnostics.
  if (getTargetArch(getOperation()) != PTOArch::A5)
    PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

// TXOR: Read(src0, src1) -> Write(tmp on A2/A3, dst)
void TXorOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  // A5 lowering does not consume tmp for TXOR; modeling tmp as a scratch
  // write inflates local-memory planning and can trigger false vec-overflow
  // diagnostics.
  if (getTargetArch(getOperation()) != PTOArch::A5)
    PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

// TTRANS: Read(src) -> Write(tmp, dst)
void TTransOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TPrintOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  if (!getTmpMutable().empty())
    PTO_ADD_WRITE(getTmpMutable()[0]);
  PTO_ADD_WRITE(getSrcMutable());
}

#undef PTO_DEFINE_TERNARY_EFFECTS
#undef PTO_DEFINE_BINARY_EFFECTS
#undef PTO_DEFINE_UNARY_EFFECTS
#undef PTO_ADD_WRITE
#undef PTO_ADD_READ

// === TMatmulOp ===
// Read: lhs, rhs, (bias), Write: dst
void TMatmulOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  // Singleton -> 直接取地址
  addEffect(effects, &getLhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getRhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TMatmulAccOp ===
// Read: acc_in, lhs, rhs, Write: dst
void TMatmulAccOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAccInMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getLhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getRhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TMatmulBiasOp ===
// Read: a, b, bias, Write: dst
void TMatmulBiasOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  // 这里的 bias 是必选的 AnyType:$bias，所以是 Singleton
  addEffect(effects, &getBiasMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TGemvOp ===
// Read: lhs, rhs, Write: dst
void TGemvOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getLhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getRhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TGemvAccOp ===
// Read: acc_in, lhs, rhs, Write: dst
void TGemvAccOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAccInMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getLhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getRhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TGemvBiasOp ===
// Read: a, b, bias, Write: dst
void TGemvBiasOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBiasMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TGemvMxOp ===
// Read: a, a_scale, b, b_scale, Write: dst
void TGemvMxOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TGemvMxAccOp ===
// Read: c_in, a, a_scale, b, b_scale, Write: dst
void TGemvMxAccOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getCInMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TGemvMxBiasOp ===
// Read: a, a_scale, b, b_scale, bias, Write: dst
void TGemvMxBiasOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBiasMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TMatmulOp ===
void TMatmulMxOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TMatmulAccMxOp ===
// Read: acc_in, lhs, rhs, Write: dst
void TMatmulMxAccOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getCInMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TMatmulBiasMxOp ===
// Read: a, b, bias, Write: dst
void TMatmulMxBiasOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBScaleMutable(), MemoryEffects::Read::get());
  // 这里的 bias 是必选的 AnyType:$bias，所以是 Singleton
  addEffect(effects, &getBiasMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

static bool isInsideSectionCube(Operation *op) {
  return op->getParentOfType<pto::SectionCubeOp>() != nullptr;
}

static bool isInsideSectionVector(Operation *op) {
  return op->getParentOfType<pto::SectionVectorOp>() != nullptr;
}

static std::optional<FunctionKernelKind>
getEnclosingFunctionKernelKind(Operation *op) {
  auto funcOp = op->getParentOfType<func::FuncOp>();
  if (!funcOp)
    return std::nullopt;

  auto kernelKindAttr =
      funcOp->getAttrOfType<FunctionKernelKindAttr>(
          FunctionKernelKindAttr::name);
  if (!kernelKindAttr)
    return std::nullopt;

  return kernelKindAttr.getKernelKind();
}

static bool isInsideSectionOrAttributedKernel(Operation *op) {
  return isInsideSectionCube(op) || isInsideSectionVector(op) ||
         getEnclosingFunctionKernelKind(op).has_value();
}

static LogicalResult verifySplitAttr(Operation *op, int64_t split) {
  if (split < 0 || split > 2)
    return op->emitOpError("expects 'split' to be 0, 1, or 2");
  return success();
}

static LogicalResult verifyFrontendKernelKind(Operation *op,
                                              FunctionKernelKind expected,
                                              StringRef kernelName) {
  if (isInsideSectionCube(op)) {
    if (expected == FunctionKernelKind::Cube)
      return success();
    return op->emitOpError("must be inside a ")
           << kernelName << " kernel function or section";
  }
  if (isInsideSectionVector(op)) {
    if (expected == FunctionKernelKind::Vector)
      return success();
    return op->emitOpError("must be inside a ")
           << kernelName << " kernel function or section";
  }

  std::optional<FunctionKernelKind> kernelKind =
      getEnclosingFunctionKernelKind(op);
  if (!kernelKind || *kernelKind != expected) {
    return op->emitOpError("must be inside a ")
           << kernelName << " kernel function or section";
  }
  return success();
}

static ParseResult parseFrontendInitializePipeOp(OpAsmParser &parser,
                                                 OperationState &result) {
  NamedAttrList attrs;
  bool sawId = false;
  bool sawDirMask = false;
  bool sawSlotSize = false;
  bool sawSlotNum = false;
  bool sawLocalSlotNum = false;
  bool sawNoSplit = false;

  if (parser.parseLBrace())
    return failure();

  while (failed(parser.parseOptionalRBrace())) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword) || parser.parseEqual())
      return failure();

    if (keyword == "id") {
      if (sawId)
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'id' clause");
      IntegerAttr idAttr;
      if (parser.parseAttribute(idAttr, parser.getBuilder().getI32Type(), "id",
                                attrs))
        return failure();
      sawId = true;
    } else if (keyword == "dir_mask") {
      if (sawDirMask)
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'dir_mask' clause");
      IntegerAttr dirMaskAttr;
      if (parser.parseAttribute(dirMaskAttr, parser.getBuilder().getI8Type(),
                                "dir_mask", attrs))
        return failure();
      sawDirMask = true;
    } else if (keyword == "slot_size") {
      if (sawSlotSize)
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'slot_size' clause");
      IntegerAttr slotSizeAttr;
      if (parser.parseAttribute(slotSizeAttr, parser.getBuilder().getI32Type(),
                                "slot_size", attrs))
        return failure();
      sawSlotSize = true;
    } else if (keyword == "slot_num") {
      if (sawSlotNum)
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'slot_num' clause");
      IntegerAttr slotNumAttr;
      if (parser.parseAttribute(slotNumAttr, parser.getBuilder().getI32Type(),
                                "slot_num", attrs))
        return failure();
      sawSlotNum = true;
    } else if (keyword == "local_slot_num") {
      if (sawLocalSlotNum)
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'local_slot_num' clause");
      IntegerAttr localSlotNumAttr;
      if (parser.parseAttribute(localSlotNumAttr, parser.getBuilder().getI32Type(),
                                "local_slot_num", attrs))
        return failure();
      sawLocalSlotNum = true;
    } else if (keyword == "nosplit") {
      if (sawNoSplit)
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'nosplit' clause");
      BoolAttr noSplitAttr;
      if (parser.parseAttribute(noSplitAttr, "nosplit", attrs))
        return failure();
      sawNoSplit = true;
    } else {
      return parser.emitError(parser.getCurrentLocation())
             << "unexpected keyword '" << keyword << "'";
    }

    if (succeeded(parser.parseOptionalRBrace()))
      break;
    if (parser.parseComma())
      return failure();
  }

  if (!sawDirMask)
    return parser.emitError(parser.getNameLoc(), "expected 'dir_mask' clause");
  if (!sawSlotSize)
    return parser.emitError(parser.getNameLoc(), "expected 'slot_size' clause");
  if (!sawId)
    attrs.set("id", parser.getBuilder().getI32IntegerAttr(0));

  OpAsmParser::UnresolvedOperand gmSlotBuffer;
  OpAsmParser::UnresolvedOperand gmSlotTensor;
  OpAsmParser::UnresolvedOperand c2vConsumerBuf;
  OpAsmParser::UnresolvedOperand v2cConsumerBuf;
  Type gmSlotBufferTy;
  Type gmSlotTensorTy;
  Type c2vConsumerBufTy;
  Type v2cConsumerBufTy;
  bool hasGmSlotBuffer = false;
  bool hasGmSlotTensor = false;
  bool hasC2vConsumerBuf = false;
  bool hasV2cConsumerBuf = false;

  if (parser.parseLParen())
    return failure();
  while (failed(parser.parseOptionalRParen())) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword) || parser.parseEqual())
      return failure();

    if (keyword == "gm_slot_buffer") {
      if (hasGmSlotBuffer)
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'gm_slot_buffer' operand");
      if (parser.parseOperand(gmSlotBuffer) ||
          parser.parseColonType(gmSlotBufferTy))
        return failure();
      hasGmSlotBuffer = true;
    } else if (keyword == "gm_slot_tensor") {
      if (hasGmSlotTensor)
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'gm_slot_tensor' operand");
      if (parser.parseOperand(gmSlotTensor) ||
          parser.parseColonType(gmSlotTensorTy))
        return failure();
      hasGmSlotTensor = true;
    } else if (keyword == "c2v_consumer_buf") {
      if (hasC2vConsumerBuf)
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'c2v_consumer_buf' operand");
      if (parser.parseOperand(c2vConsumerBuf) ||
          parser.parseColonType(c2vConsumerBufTy))
        return failure();
      hasC2vConsumerBuf = true;
    } else if (keyword == "v2c_consumer_buf") {
      if (hasV2cConsumerBuf)
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'v2c_consumer_buf' operand");
      if (parser.parseOperand(v2cConsumerBuf) ||
          parser.parseColonType(v2cConsumerBufTy))
        return failure();
      hasV2cConsumerBuf = true;
    } else {
      return parser.emitError(parser.getCurrentLocation())
             << "unexpected initialize_pipe operand '" << keyword << "'";
    }

    if (succeeded(parser.parseOptionalRParen()))
      break;
    if (parser.parseComma())
      return failure();
  }

  if (parser.parseOptionalAttrDict(attrs))
    return failure();

  result.addAttributes(attrs);
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {hasGmSlotBuffer ? 1 : 0, hasGmSlotTensor ? 1 : 0,
                           hasC2vConsumerBuf ? 1 : 0,
                           hasV2cConsumerBuf ? 1 : 0}));
  if (hasGmSlotBuffer &&
      parser.resolveOperand(gmSlotBuffer, gmSlotBufferTy, result.operands))
    return failure();
  if (hasGmSlotTensor &&
      parser.resolveOperand(gmSlotTensor, gmSlotTensorTy, result.operands))
    return failure();
  if (hasC2vConsumerBuf &&
      parser.resolveOperand(c2vConsumerBuf, c2vConsumerBufTy, result.operands))
    return failure();
  if (hasV2cConsumerBuf &&
      parser.resolveOperand(v2cConsumerBuf, v2cConsumerBufTy, result.operands))
    return failure();
  return success();
}

template <typename InitOpT>
static void printFrontendInitializePipeOp(InitOpT op, OpAsmPrinter &p) {
  p << " {";
  bool needsComma = false;
  auto printClause = [&](StringRef keyword, auto value) {
    if (needsComma)
      p << ", ";
    p << keyword << " = " << value;
    needsComma = true;
  };

  if (op.getId() != 0)
    printClause("id", op.getId());
  printClause("dir_mask", static_cast<int32_t>(op.getDirMask()));
  printClause("slot_size", op.getSlotSize());
  if (auto slotNumAttr = op.getSlotNumAttr())
    printClause("slot_num", slotNumAttr.getInt());
  if (auto localSlotNumAttr = op.getLocalSlotNumAttr())
    printClause("local_slot_num", localSlotNumAttr.getInt());
  if (auto noSplitAttr = op.getNosplitAttr())
    printClause("nosplit", noSplitAttr.getValue() ? "true" : "false");
  p << "}";

  p << "(";
  bool needsOperandComma = false;
  auto printOperandClause = [&](StringRef keyword, Value value) {
    if (needsOperandComma)
      p << ", ";
    p << keyword << " = " << value << " : " << value.getType();
    needsOperandComma = true;
  };
  if (op.getGmSlotBuffer()) {
    printOperandClause("gm_slot_buffer", op.getGmSlotBuffer());
  }
  if (op.getGmSlotTensor())
    printOperandClause("gm_slot_tensor", op.getGmSlotTensor());
  if (op.getC2vConsumerBuf())
    printOperandClause("c2v_consumer_buf", op.getC2vConsumerBuf());
  if (op.getV2cConsumerBuf())
    printOperandClause("v2c_consumer_buf", op.getV2cConsumerBuf());
  p << ")";
  p.printOptionalAttrDict(
      op->getAttrs(),
      /*elidedAttrs=*/{"id", "dir_mask", "slot_size", "slot_num",
                       "local_slot_num",
                       "nosplit", "operandSegmentSizes"});
}

static std::optional<uint64_t>
getStaticElementCount(ArrayRef<int64_t> shape) {
  uint64_t count = 1;
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic || dim < 0)
      return std::nullopt;
    count *= static_cast<uint64_t>(dim);
  }
  return count;
}

static bool isSameOrHalfSlotByteSize(uint64_t tensorBytes, uint64_t slotBytes) {
  return tensorBytes == slotBytes || tensorBytes * 2 == slotBytes;
}

static LogicalResult verifyFrontendGlobalSlotTensor(Operation *op, Value tensor,
                                                    int8_t dirMask,
                                                    int32_t slotSize) {
  (void)dirMask;
  auto tvTy = dyn_cast<TensorViewType>(tensor.getType());
  if (!tvTy)
    return op->emitOpError("expects 'gm_slot_tensor' to be !pto.tensor_view");

  ArrayRef<int64_t> shape = tvTy.getShape();
  if (shape.empty())
    return op->emitOpError(
        "expects 'gm_slot_tensor' to describe one slot entry tensor");

  if (auto elemCount = getStaticElementCount(shape)) {
    uint64_t elemBytes = getElemByteSize(tvTy.getElementType());
    if (elemBytes != 0) {
      uint64_t tensorBytes = *elemCount * elemBytes;
      if (!isSameOrHalfSlotByteSize(tensorBytes,
                                    static_cast<uint64_t>(slotSize))) {
        return op->emitOpError()
               << "expects 'slot_size' to equal gm_slot_tensor byte size "
                  "or twice gm_slot_tensor byte size for split GlobalTensor "
                  "entries (got slot_size = "
               << slotSize << ", gm_slot_tensor byte size = " << tensorBytes
               << ")";
      }
    }
  }

  return success();
}

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

  unsigned sameIdInitCount = 0;
  funcOp.walk([&](Operation *candidate) {
    if (auto aic = dyn_cast<AicInitializePipeOp>(candidate)) {
      if (aic.getId() == op.getId())
        ++sameIdInitCount;
      return;
    }
    if (auto aiv = dyn_cast<AivInitializePipeOp>(candidate))
      if (aiv.getId() == op.getId())
        ++sameIdInitCount;
  });
  if (sameIdInitCount > 1) {
    return op.emitOpError(
        "requires 'id' to be unique across frontend initialize_pipe ops in the function");
  }

  int8_t dirMask = op.getDirMask();
  if (dirMask != 1 && dirMask != 2 && dirMask != 3)
    return op.emitOpError("expects 'dir_mask' to be 1, 2, or 3");
  if (op.getSlotSize() <= 0)
    return op.emitOpError("expects 'slot_size' to be greater than 0");
  int32_t slotNum = dirMask == 3 ? 4 : 8;
  if (auto slotNumAttr = op.getSlotNumAttr()) {
    slotNum = slotNumAttr.getInt();
    if (slotNum <= 0)
      return op.emitOpError("expects 'slot_num' to be greater than 0");
  }
  PTOArch arch = getTargetArch(op.getOperation());

  bool hasGlobalSlotTensor = static_cast<bool>(op.getGmSlotTensor());
  bool hasGmSlotBuffer = static_cast<bool>(op.getGmSlotBuffer());
  bool hasC2vConsumerBuf = static_cast<bool>(op.getC2vConsumerBuf());
  bool hasV2cConsumerBuf = static_cast<bool>(op.getV2cConsumerBuf());
  if (hasGlobalSlotTensor) {
    if (hasGmSlotBuffer || hasC2vConsumerBuf || hasV2cConsumerBuf) {
      return op.emitOpError(
          "globaltensor pipe init expects only 'gm_slot_tensor' and no "
          "'gm_slot_buffer', 'c2v_consumer_buf', or 'v2c_consumer_buf'");
    }
    if (op.getLocalSlotNumAttr())
      return op.emitOpError(
          "globaltensor pipe init does not use 'local_slot_num'");
    return verifyFrontendGlobalSlotTensor(
        op.getOperation(), op.getGmSlotTensor(), dirMask, op.getSlotSize());
  }

  if (!hasC2vConsumerBuf && !hasV2cConsumerBuf) {
    return op.emitOpError(
        "expects local pipe init to provide at least one consumer buffer "
        "operand; use 'gm_slot_tensor' for globaltensor pipe entries");
  }
  if (dirMask == 1 && !hasC2vConsumerBuf) {
    return op.emitOpError(
        "expects 'c2v_consumer_buf' when dir_mask is 1");
  }
  if (dirMask == 2 && !hasV2cConsumerBuf) {
    return op.emitOpError(
        "expects 'v2c_consumer_buf' when dir_mask is 2");
  }
  if (dirMask == 3 && (!hasC2vConsumerBuf || !hasV2cConsumerBuf)) {
    return op.emitOpError(
        "expects both 'c2v_consumer_buf' and 'v2c_consumer_buf' when dir_mask is 3");
  }

  if (auto localSlotNumAttr = op.getLocalSlotNumAttr()) {
    if (arch == PTOArch::A5)
      return op.emitOpError(
          "'local_slot_num' is only supported for a2/a3 frontend pipe lowering");
    int32_t localSlotNum = localSlotNumAttr.getInt();
    if (localSlotNum <= 0)
      return op.emitOpError("expects 'local_slot_num' to be greater than 0");
    if (localSlotNum > slotNum) {
      return op.emitOpError()
             << "expects 'local_slot_num' to be less than or equal to slot_num ("
             << slotNum << ") for dir_mask = " << static_cast<int>(dirMask);
    }
  }

  return success();
}

ParseResult AicInitializePipeOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  return parseFrontendInitializePipeOp(parser, result);
}

void AicInitializePipeOp::print(OpAsmPrinter &p) {
  printFrontendInitializePipeOp(*this, p);
}

ParseResult AivInitializePipeOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  return parseFrontendInitializePipeOp(parser, result);
}

void AivInitializePipeOp::print(OpAsmPrinter &p) {
  printFrontendInitializePipeOp(*this, p);
}

ReserveBufferOp mlir::pto::findReserveBufferByName(func::FuncOp funcOp,
                                                   StringRef name) {
  ReserveBufferOp found;
  funcOp.walk([&](ReserveBufferOp reserveOp) {
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
  funcOp.walk([&](ReserveBufferOp reserveOp) {
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

  auto peerFunc = lookupPeerFuncAcrossContainer(getOperation(), getPeerFuncAttr());
  if (!peerFunc)
    return emitOpError("expects 'peer_func' to reference an existing func.func");

  unsigned sameImportCount = 0;
  funcOp.walk([&](ImportReservedBufferOp importOp) {
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
  funcOp.walk([&](Operation *candidate) {
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
  if (expectC2V && dirMask != 1 && dirMask != 3) {
    return op->emitOpError()
           << "expects 'id' = " << id
           << " to reference initialize_pipe with dir_mask = 1 or 3";
  }
  if (!expectC2V && dirMask != 2 && dirMask != 3) {
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
  if (dirMask != 1 && dirMask != 2 && dirMask != 3)
    return op->emitOpError("expects 'dir_mask' to be 1, 2, or 3");
  if (slotSize <= 0)
    return op->emitOpError("expects 'slot_size' to be greater than 0");
  if (slotNum <= 0)
    return op->emitOpError("expects 'slot_num' to be greater than 0");
  if (flagBase && *flagBase < 0)
    return op->emitOpError("expects 'flag_base' to be non-negative when present");
  if (flagBase) {
    int32_t flagWidth = dirMask == 3 ? 4 : 2;
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

static LogicalResult verifyTensorEntryMatchesInternalPipeInit(Operation *op,
                                                              Value pipeHandle,
                                                              Type entryTy) {
  auto entryViewTy = dyn_cast<TensorViewType>(entryTy);
  if (!entryViewTy)
    return success();

  auto initOp = pipeHandle.getDefiningOp<InitializeL2G2LPipeOp>();
  if (!initOp) {
    return op->emitOpError()
           << "expects !pto.tensor_view pipe entry to use a pipe produced by "
              "pto.initialize_l2g2l_pipe";
  }
  if (initOp.getLocalAddr()) {
    return op->emitOpError()
           << "expects !pto.tensor_view pipe entry to use global-only "
              "pto.initialize_l2g2l_pipe without local_addr";
  }

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
    if (slotDim == ShapedType::kDynamic ||
        entryDim == ShapedType::kDynamic || slotDim == entryDim)
      continue;
    return op->emitOpError()
           << "expects pipe entry dimension " << idx
           << " to match initialize_l2g2l_pipe gm_addr dimension "
           << slotDim;
  }

  if (auto entryElemCount = getStaticElementCount(entryShape)) {
    uint64_t elemBytes = getElemByteSize(entryViewTy.getElementType());
    uint64_t entryBytes = *entryElemCount * elemBytes;
    if (elemBytes != 0) {
      int8_t split = 0;
      if (auto alloc = dyn_cast<TAllocOp>(op))
        split = alloc.getSplit();
      else if (auto push = dyn_cast<TPushOp>(op))
        split = push.getSplit();
      else if (auto pop = dyn_cast<TPopOp>(op))
        split = pop.getSplit();
      else if (auto free = dyn_cast<TFreeOp>(op))
        split = free.getSplit();

      uint64_t slotBytes = static_cast<uint64_t>(initOp.getSlotSize());
      bool isSplitEntry = split != 0;
      bool byteSizeMatches =
          entryBytes == slotBytes || (isSplitEntry && entryBytes * 2 == slotBytes);
      if (!byteSizeMatches) {
        return op->emitOpError()
               << "expects pipe entry byte size to match initialize_l2g2l_pipe "
                  "slot_size"
               << (isSplitEntry ? " or half slot_size for split entries" : "")
               << " (got entry byte size = " << entryBytes
               << ", slot_size = " << initOp.getSlotSize() << ")";
      }
    }
  }

  return success();
}

LogicalResult BuildAsyncSessionOp::verify() {
  Type scratchTy = getScratch().getType();
  if (!isa<pto::TileBufType, MemRefType>(scratchTy))
    return emitOpError("expects scratch to be tile_buf or memref type");

  auto scratchSpace = getPTOMemorySpaceEnum(scratchTy);
  if (!scratchSpace || *scratchSpace != pto::AddressSpace::VEC)
    return emitOpError("expects scratch to be in vec address space");

  auto scratchShape = getShapeVec(scratchTy);
  if (scratchShape.empty() || scratchShape.size() > 2)
    return emitOpError("expects scratch to be rank-1 or rank-2");
  for (int64_t dim : scratchShape) {
    if (dim == ShapedType::kDynamic)
      return emitOpError("expects scratch to have a static shape");
  }

  auto scratchBytes = getStaticByteSize(scratchTy);
  if (!scratchBytes)
    return emitOpError("expects scratch byte size to be statically known");
  if (*scratchBytes < sizeof(uint64_t))
    return emitOpError("expects scratch to provide at least 8 bytes");

  Type workspaceElemTy;
  Type workspaceTy = getWorkspace().getType();
  if (auto ptrTy = dyn_cast<pto::PtrType>(workspaceTy)) {
    workspaceElemTy = ptrTy.getElementType();
  } else if (auto memTy = dyn_cast<MemRefType>(workspaceTy)) {
    workspaceElemTy = memTy.getElementType();
    if (!isGmAddressSpaceAttr(memTy.getMemorySpace()))
      return emitOpError("expects workspace to be in GM address space");
  } else {
    return emitOpError("expects workspace to be !pto.ptr or memref type");
  }
  if (!isByteIntegerType(workspaceElemTy))
    return emitOpError("expects workspace element type to be an 8-bit integer");

  if (auto syncIdAttr = getSyncIdAttr()) {
    int64_t syncId = syncIdAttr.getInt();
    if (syncId < 0 || syncId > 7)
      return emitOpError("expects sync_id in range [0, 7]");
  }
  if (auto blockBytesAttr = getBlockBytesAttr()) {
    if (blockBytesAttr.getInt() <= 0)
      return emitOpError("expects block_bytes to be greater than 0");
  }
  if (auto commBlockOffsetAttr = getCommBlockOffsetAttr()) {
    if (commBlockOffsetAttr.getInt() < 0)
      return emitOpError("expects comm_block_offset to be non-negative");
  }
  if (auto queueNumAttr = getQueueNumAttr()) {
    if (queueNumAttr.getInt() <= 0)
      return emitOpError("expects queue_num to be greater than 0");
  }
  if (auto channelGroupIdxAttr = getChannelGroupIdxAttr()) {
    APInt value = channelGroupIdxAttr.getValue();
    if (value.isNegative())
      return emitOpError("expects channel_group_idx to be non-negative");
    if (value.ugt(UINT32_MAX))
      return emitOpError("expects channel_group_idx to fit in uint32");
  }

  return success();
}

static LogicalResult verifyAsyncTransferOp(Operation *op, Value dst, Value src) {
  Type dstElemTy = getElemTy(dst.getType());
  Type srcElemTy = getElemTy(src.getType());
  if (!dstElemTy || !srcElemTy)
    return op->emitOpError("expects src and dst to have element types");
  if (dstElemTy != srcElemTy)
    return op->emitOpError("expects src and dst to have the same element type");
  if (failed(verifyAsyncFlatContiguous1DGMViewLike(op, dst, "dst")) ||
      failed(verifyAsyncFlatContiguous1DGMViewLike(op, src, "src")))
    return failure();
  if (getShapeVec(dst.getType()) != getShapeVec(src.getType()))
    return op->emitOpError("expects src and dst to have the same static shape");
  return success();
}

LogicalResult TPutAsyncOp::verify() {
  return verifyAsyncTransferOp(getOperation(), getDst(), getSrc());
}

LogicalResult TGetAsyncOp::verify() {
  return verifyAsyncTransferOp(getOperation(), getDst(), getSrc());
}

LogicalResult TPutOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  if (failed(verifyCommGlobalLike(*this, getDst(), "dst")) ||
      failed(verifyCommGlobalLike(*this, getSrc(), "src")) ||
      failed(verifyCommStagingTileLike(*this, getPing(), "ping")) ||
      failed(verifyCommPingPongSameType(*this, getPing(), getPong(), "ping",
                                        "pong")))
    return failure();
  if (getElemTy(getDst().getType()) != getElemTy(getSrc().getType()))
    return emitOpError("expects src and dst to have the same element type");
  if (getShapeVec(getDst().getType()) != getShapeVec(getSrc().getType()))
    return emitOpError("expects src and dst to have the same static shape");
  if (getElemTy(getPing().getType()) != getElemTy(getSrc().getType()))
    return emitOpError("expects staging tile element type to match src/dst");
  return success();
}

LogicalResult TGetOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  if (failed(verifyCommGlobalLike(*this, getDst(), "dst")) ||
      failed(verifyCommGlobalLike(*this, getSrc(), "src")) ||
      failed(verifyCommStagingTileLike(*this, getPing(), "ping")) ||
      failed(verifyCommPingPongSameType(*this, getPing(), getPong(), "ping",
                                        "pong")))
    return failure();
  if (getElemTy(getDst().getType()) != getElemTy(getSrc().getType()))
    return emitOpError("expects src and dst to have the same element type");
  if (getShapeVec(getDst().getType()) != getShapeVec(getSrc().getType()))
    return emitOpError("expects src and dst to have the same static shape");
  if (getElemTy(getPing().getType()) != getElemTy(getSrc().getType()))
    return emitOpError("expects staging tile element type to match src/dst");
  return success();
}

LogicalResult TNotifyOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  if (failed(verifyCommSignalLike(*this, getSignal(), "signal")))
    return failure();
  auto valueTy = dyn_cast<IntegerType>(getValue().getType());
  if (!valueTy || valueTy.getWidth() != 32)
    return emitOpError("expects value to be i32");
  return success();
}

LogicalResult TWaitOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  if (failed(verifyCommSignalLike(*this, getSignal(), "signal")))
    return failure();
  auto cmpTy = dyn_cast<IntegerType>(getCmpValue().getType());
  if (!cmpTy || cmpTy.getWidth() != 32)
    return emitOpError("expects cmp_value to be i32");
  return success();
}

LogicalResult TTestOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  if (failed(verifyCommSignalLike(*this, getSignal(), "signal")))
    return failure();
  auto cmpTy = dyn_cast<IntegerType>(getCmpValue().getType());
  if (!cmpTy || cmpTy.getWidth() != 32)
    return emitOpError("expects cmp_value to be i32");
  return success();
}

static LogicalResult verifySyncAllGmWorkspace(Operation *op, Value workspace,
                                              StringRef name) {
  Type ty = workspace.getType();
  if (!isa<MemRefType, pto::TensorViewType, pto::PartitionTensorViewType>(ty))
    return op->emitOpError() << "expects " << name
                             << " to be a GM memref/tensor_view/partition_view";

  if (auto memTy = dyn_cast<MemRefType>(ty)) {
    if (!memTy.hasRank())
      return op->emitOpError() << "expects " << name << " to be ranked";
    if (!isGmAddressSpaceAttr(memTy.getMemorySpace()))
      return op->emitOpError() << "expects " << name
                               << " to be in GM address space";
  }

  auto elemTy = dyn_cast<IntegerType>(getElemTy(ty));
  if (!elemTy || elemTy.getWidth() != 32)
    return op->emitOpError() << "expects " << name
                             << " element type to be i32";

  SmallVector<int64_t, 4> shape = getShapeVec(ty);
  if (shape.empty())
    return op->emitOpError() << "expects " << name << " to have rank >= 1";
  for (int64_t dim : shape) {
    if (dim != ShapedType::kDynamic && dim <= 0)
      return op->emitOpError() << "expects " << name
                               << " shape to be positive";
  }
  return success();
}

static LogicalResult verifySyncAllTileWorkspace(Operation *op, Value workspace,
                                                StringRef name,
                                                pto::AddressSpace expectedSpace) {
  Type ty = workspace.getType();
  if (!isa<pto::TileBufType, MemRefType>(ty))
    return op->emitOpError() << "expects " << name
                             << " to be tile_buf or memref type";

  if (isa<pto::TileBufType>(ty) && failed(verifyTileBufCommon(op, ty, name)))
    return failure();

  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != expectedSpace)
    return op->emitOpError() << "expects " << name << " to be in "
                             << (expectedSpace == pto::AddressSpace::VEC
                                     ? "vec"
                                     : "mat")
                             << " address space";

  Type elemTy = getElemTy(ty);
  auto intTy = dyn_cast_or_null<IntegerType>(elemTy);
  if (!intTy || intTy.getWidth() != 32)
    return op->emitOpError() << "expects " << name
                             << " element type to be i32";

  auto shape = getShapeVec(ty);
  if (shape.empty() || shape.size() > 2)
    return op->emitOpError() << "expects " << name
                             << " to be rank-1 or rank-2";
  for (int64_t dim : shape) {
    if (dim != ShapedType::kDynamic && dim <= 0)
      return op->emitOpError() << "expects " << name
                               << " shape to be positive";
  }
  return success();
}

LogicalResult SyncAllOp::verify() {
  bool hasGm = static_cast<bool>(getGmWorkspace());
  bool hasUb = static_cast<bool>(getUbWorkspace());
  bool hasL1 = static_cast<bool>(getL1Workspace());
  auto mode = getMode().getValue();
  auto coreType = getCoreType().getValue();

  if (mode == pto::SyncAllMode::Hard) {
    if (hasGm || hasUb || hasL1 || getUsedCores())
      return emitOpError(
          "expects hard syncall to have no workspace operands or used_cores");
    return success();
  }

  if (!hasGm)
    return emitOpError("expects soft syncall to provide gm_workspace");
  if (failed(verifySyncAllGmWorkspace(getOperation(), getGmWorkspace(),
                                      "gm_workspace")))
    return failure();

  if (auto used = getUsedCores()) {
    auto intTy = dyn_cast<IntegerType>(used.getType());
    if (!intTy || intTy.getWidth() != 32)
      return emitOpError("expects used_cores to be i32");
  }

  switch (coreType) {
  case pto::SyncCoreType::AIVOnly:
    if (!hasUb || hasL1)
      return emitOpError("expects soft AIV-only syncall to use gm_workspace "
                         "+ ub_workspace only");
    return verifySyncAllTileWorkspace(getOperation(), getUbWorkspace(),
                                      "ub_workspace",
                                      pto::AddressSpace::VEC);
  case pto::SyncCoreType::AICOnly:
    if (hasUb || !hasL1)
      return emitOpError("expects soft AIC-only syncall to use gm_workspace "
                         "+ l1_workspace only");
    return verifySyncAllTileWorkspace(getOperation(), getL1Workspace(),
                                      "l1_workspace",
                                      pto::AddressSpace::MAT);
  case pto::SyncCoreType::Mix:
    if (!hasUb || !hasL1)
      return emitOpError("expects soft mixed syncall to use gm_workspace + "
                         "ub_workspace + l1_workspace");
    if (failed(verifySyncAllTileWorkspace(getOperation(), getUbWorkspace(),
                                          "ub_workspace",
                                          pto::AddressSpace::VEC)))
      return failure();
    return verifySyncAllTileWorkspace(getOperation(), getL1Workspace(),
                                      "l1_workspace",
                                      pto::AddressSpace::MAT);
  }

  llvm_unreachable("unhandled SyncCoreType");
}

LogicalResult TBroadcastOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  if (failed(verifyCommGlobalLike(*this, getSrc(), "src")) ||
      failed(verifyCommStagingTileLike(*this, getPing(), "ping")) ||
      failed(verifyCommPingPongSameType(*this, getPing(), getPong(), "ping",
                                        "pong")) ||
      failed(verifyCommGlobalGroup(*this, getGroup(), "group")))
    return failure();
  if (getRoot() >= static_cast<uint32_t>(getGroup().size()))
    return emitOpError("expects root to index into group operands");
  if (getSrc().getType() != getGroup().front().getType())
    return emitOpError("expects src type to match group member type");
  if (getElemTy(getPing().getType()) != getElemTy(getSrc().getType()))
    return emitOpError("expects staging tile element type to match src");
  return success();
}

LogicalResult CommTGatherOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  if (failed(verifyCommGlobalLike(*this, getDst(), "dst")) ||
      failed(verifyCommStagingTileLike(*this, getPing(), "ping")) ||
      failed(verifyCommPingPongSameType(*this, getPing(), getPong(), "ping",
                                        "pong")) ||
      failed(verifyCommGlobalGroup(*this, getGroup(), "group")))
    return failure();
  if (getRoot() >= static_cast<uint32_t>(getGroup().size()))
    return emitOpError("expects root to index into group operands");
  if (getElemTy(getDst().getType()) != getElemTy(getGroup().front().getType()))
    return emitOpError("expects dst element type to match group member type");
  if (getElemTy(getPing().getType()) != getElemTy(getDst().getType()))
    return emitOpError("expects staging tile element type to match dst");
  return success();
}

LogicalResult CommTScatterOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  if (failed(verifyCommGlobalLike(*this, getSrc(), "src")) ||
      failed(verifyCommStagingTileLike(*this, getPing(), "ping")) ||
      failed(verifyCommPingPongSameType(*this, getPing(), getPong(), "ping",
                                        "pong")) ||
      failed(verifyCommGlobalGroup(*this, getGroup(), "group")))
    return failure();
  if (getRoot() >= static_cast<uint32_t>(getGroup().size()))
    return emitOpError("expects root to index into group operands");
  if (getElemTy(getSrc().getType()) != getElemTy(getGroup().front().getType()))
    return emitOpError("expects src element type to match group member type");
  if (getElemTy(getPing().getType()) != getElemTy(getSrc().getType()))
    return emitOpError("expects staging tile element type to match src");
  return success();
}

LogicalResult TReduceOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  if (failed(verifyCommGlobalLike(*this, getDst(), "dst")) ||
      failed(verifyCommStagingTileLike(*this, getAcc(), "acc")) ||
      failed(verifyCommStagingTileLike(*this, getRecvPing(), "recv_ping")) ||
      failed(verifyCommPingPongSameType(*this, getRecvPing(), getRecvPong(),
                                        "recv_ping", "recv_pong")) ||
      failed(verifyCommGlobalGroup(*this, getGroup(), "group")))
    return failure();
  if (getRoot() >= static_cast<uint32_t>(getGroup().size()))
    return emitOpError("expects root to index into group operands");
  if (getElemTy(getDst().getType()) != getElemTy(getGroup().front().getType()))
    return emitOpError("expects dst element type to match group member type");
  if (getAcc().getType() != getRecvPing().getType())
    return emitOpError("expects acc and recv_ping to have identical types");
  if (getElemTy(getAcc().getType()) != getElemTy(getDst().getType()))
    return emitOpError("expects accumulator/receive tiles to match dst element type");
  return success();
}

LogicalResult AicInitializePipeOp::verify() {
  return verifyFrontendInitCommon(*this, FunctionKernelKind::Cube, "cube");
}

LogicalResult AivInitializePipeOp::verify() {
  return verifyFrontendInitCommon(*this, FunctionKernelKind::Vector, "vector");
}

LogicalResult TAllocToAivOp::verify() {
  if (failed(verifyFrontendSplitOp(getOperation(), FunctionKernelKind::Cube,
                                   "cube", getId(), getSplit())))
    return failure();
  if (failed(verifyFrontendDataOpDirection(getOperation(), getId(),
                                           /*expectC2V=*/true)))
    return failure();
  return verifyFrontendTensorEntryMatchesInit(getOperation(), getId(),
                                              getEntry().getType());
}

LogicalResult TAllocToAicOp::verify() {
  if (failed(verifyFrontendSplitOp(getOperation(), FunctionKernelKind::Vector,
                                   "vector", getId(), getSplit())))
    return failure();
  if (failed(verifyFrontendDataOpDirection(getOperation(), getId(),
                                           /*expectC2V=*/false)))
    return failure();
  return verifyFrontendTensorEntryMatchesInit(getOperation(), getId(),
                                              getEntry().getType());
}

LogicalResult TPushToAivOp::verify() {
  if (failed(verifyFrontendSplitOp(getOperation(), FunctionKernelKind::Cube,
                                   "cube", getId(), getSplit())))
    return failure();
  if (failed(verifyFrontendDataOpDirection(getOperation(), getId(),
                                           /*expectC2V=*/true)))
    return failure();
  return verifyFrontendTensorEntryMatchesInit(getOperation(), getId(),
                                              getTile().getType());
}

LogicalResult TPushToAicOp::verify() {
  if (failed(verifyFrontendSplitOp(getOperation(), FunctionKernelKind::Vector,
                                   "vector", getId(), getSplit())))
    return failure();
  if (failed(verifyFrontendDataOpDirection(getOperation(), getId(),
                                           /*expectC2V=*/false)))
    return failure();
  return verifyFrontendTensorEntryMatchesInit(getOperation(), getId(),
                                              getTile().getType());
}

LogicalResult TPopFromAicOp::verify() {
  return verifyFrontendPopOp(*this, FunctionKernelKind::Vector, "vector",
                             /*expectC2V=*/true);
}

LogicalResult TPopFromAivOp::verify() {
  return verifyFrontendPopOp(*this, FunctionKernelKind::Cube, "cube",
                             /*expectC2V=*/false);
}

LogicalResult TFreeFromAicOp::verify() {
  if (failed(verifyFrontendSplitOp(getOperation(), FunctionKernelKind::Vector,
                                   "vector", getId(), getSplit())))
    return failure();
  if (failed(verifyFrontendDataOpDirection(getOperation(), getId(),
                                           /*expectC2V=*/true)))
    return failure();
  if (getEntry())
    return verifyFrontendTensorEntryMatchesInit(getOperation(), getId(),
                                                getEntry().getType());
  return success();
}

LogicalResult TFreeFromAivOp::verify() {
  if (failed(verifyFrontendSplitOp(getOperation(), FunctionKernelKind::Cube,
                                   "cube", getId(), getSplit())))
    return failure();
  if (failed(verifyFrontendDataOpDirection(getOperation(), getId(),
                                           /*expectC2V=*/false)))
    return failure();
  if (getEntry())
    return verifyFrontendTensorEntryMatchesInit(getOperation(), getId(),
                                                getEntry().getType());
  return success();
}

LogicalResult InitializeL2G2LPipeOp::verify() {
  if (failed(verifyPipeShape(getOperation(), getDirMask(), getSlotSize(),
                             getSlotNum(),
                             getFlagBaseAttr()
                                 ? std::optional<int32_t>(getFlagBaseAttr().getInt())
                                 : std::nullopt)))
    return failure();

  if (!getLocalAddr()) {
    if (getPeerLocalAddr())
      return emitOpError("'peer_local_addr' requires 'local_addr'");
    if (getLocalSlotNumAttr())
      return emitOpError(
          "'local_slot_num' is only allowed when 'local_addr' is present");
    return success();
  }

  if (auto localSlotNumAttr = getLocalSlotNumAttr()) {
    int32_t localSlotNum = localSlotNumAttr.getInt();
    if (localSlotNum <= 0)
      return emitOpError("expects 'local_slot_num' to be greater than 0");
    if (static_cast<uint32_t>(localSlotNum) > getSlotNum())
      return emitOpError(
          "expects 'local_slot_num' to be less than or equal to slot_num");
  }

  if (getDirMask() == 3 && !getPeerLocalAddr())
    return emitOpError("expects 'peer_local_addr' when dir_mask is 3");
  if (getDirMask() != 3 && getPeerLocalAddr())
    return emitOpError("'peer_local_addr' is only allowed when dir_mask is 3");
  return success();
}

LogicalResult InitializeL2LPipeOp::verify() {
  if (failed(verifyPipeShape(getOperation(), getDirMask(), getSlotSize(),
                              getSlotNum(),
                              getFlagBaseAttr()
                                  ? std::optional<int32_t>(getFlagBaseAttr().getInt())
                                  : std::nullopt)))
    return failure();

  if (getDirMask() == 3 && !getPeerLocalAddr())
    return emitOpError("expects 'peer_local_addr' when dir_mask is 3");
  if (getDirMask() != 3 && getPeerLocalAddr())
    return emitOpError("'peer_local_addr' is only allowed when dir_mask is 3");
  return success();
}

LogicalResult TPushOp::verify() {
  if (!isInsideSectionOrAttributedKernel(getOperation()))
    return emitOpError("must be inside pto.section.cube/vector or a kernel_kind function");
  if (failed(verifyPipeHandleProducer(getOperation(), getPipeHandle())))
    return failure();
  if (failed(verifySplitAttr(getOperation(), getSplit())))
    return failure();
  if (failed(verifyTensorEntryMatchesInternalPipeInit(
          getOperation(), getPipeHandle(), getTile().getType())))
    return failure();
  if (!isa<TensorViewType>(getTile().getType()) &&
      getPipe() == pto::PIPE::PIPE_UNASSIGNED)
    return emitOpError("tile type must map to a supported producer pipe");
  return success();
}

LogicalResult TAllocOp::verify() {
  if (!isInsideSectionOrAttributedKernel(getOperation()))
    return emitOpError("must be inside pto.section.cube/vector or a kernel_kind function");
  if (failed(verifyPipeHandleProducer(getOperation(), getPipeHandle())))
    return failure();
  if (failed(verifyTensorEntryMatchesInternalPipeInit(
          getOperation(), getPipeHandle(), getEntry().getType())))
    return failure();
  return verifySplitAttr(getOperation(), getSplit());
}

LogicalResult TPopOp::verify() {
  if (!isInsideSectionOrAttributedKernel(getOperation()))
    return emitOpError("must be inside pto.section.cube/vector or a kernel_kind function");
  if (failed(verifyPipeHandleProducer(getOperation(), getPipeHandle())))
    return failure();
  if (failed(verifySplitAttr(getOperation(), getSplit())))
    return failure();
  if (failed(verifyTensorEntryMatchesInternalPipeInit(
          getOperation(), getPipeHandle(), getTile().getType())))
    return failure();
  if (!isa<TensorViewType>(getTile().getType()) &&
      getPipe() == pto::PIPE::PIPE_UNASSIGNED)
    return emitOpError(
        "tile type and target arch must map to a supported consumer pipe");
  return success();
}

LogicalResult TFreeOp::verify() {
  if (!isInsideSectionOrAttributedKernel(getOperation()))
    return emitOpError("must be inside pto.section.cube/vector or a kernel_kind function");
  if (failed(verifyPipeHandleProducer(getOperation(), getPipeHandle())))
    return failure();
  if (getEntry() &&
      failed(verifyTensorEntryMatchesInternalPipeInit(
          getOperation(), getPipeHandle(), getEntry().getType())))
    return failure();
  return verifySplitAttr(getOperation(), getSplit());
}

ParseResult TFreeOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand first;
  OpAsmParser::UnresolvedOperand pipe;
  Type firstTy;
  Type pipeTy;
  bool hasEntry = false;

  if (parser.parseLParen() || parser.parseOperand(first))
    return failure();

  if (succeeded(parser.parseOptionalComma())) {
    hasEntry = true;
    if (parser.parseOperand(pipe) || parser.parseColonType(firstTy) ||
        parser.parseComma() || parser.parseType(pipeTy) || parser.parseRParen())
      return failure();
  } else {
    if (parser.parseColonType(pipeTy) || parser.parseRParen())
      return failure();
    pipe = first;
  }

  NamedAttrList attrs;
  if (parser.parseLBrace() || parser.parseKeyword("split") ||
      parser.parseEqual())
    return failure();
  IntegerAttr splitAttr;
  if (parser.parseAttribute(splitAttr, parser.getBuilder().getI8Type(),
                            "split", attrs) ||
      parser.parseRBrace() || parser.parseOptionalAttrDict(attrs))
    return failure();

  result.addAttributes(attrs);
  if (hasEntry &&
      parser.resolveOperand(first, firstTy, result.operands))
    return failure();
  if (parser.resolveOperand(pipe, pipeTy, result.operands))
    return failure();
  return success();
}

void TFreeOp::print(OpAsmPrinter &p) {
  p << "(";
  if (getEntry()) {
    p << getEntry() << ", " << getPipeHandle() << " : "
      << getEntry().getType() << ", " << getPipeHandle().getType();
  } else {
    p << getPipeHandle() << " : " << getPipeHandle().getType();
  }
  p << ") {split = " << static_cast<int32_t>(getSplit()) << "}";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"split"});
}

static func::FuncOp getParentFunc(Operation *op) {
  return op ? op->getParentOfType<func::FuncOp>() : func::FuncOp();
}

static constexpr int64_t kSimtKeepResumeSlotLimit = 123;

static Operation *getFirstNonConstantLikeOp(Block *block) {
  if (!block)
    return nullptr;
  for (Operation &op : *block) {
    if (!op.hasTrait<OpTrait::ConstantLike>())
      return &op;
  }
  return nullptr;
}

static bool isOpInRange(Operation *op, Operation *first, Operation *last) {
  for (Operation *cur = first; cur; cur = cur->getNextNode()) {
    if (cur == op)
      return true;
    if (cur == last)
      return false;
  }
  return false;
}

static std::optional<unsigned> getSimtKeepResumeRegisterCount(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    if (intType.getWidth() <= 32)
      return 1;
    if (intType.getWidth() == 64)
      return 2;
    return std::nullopt;
  }
  if (type.isF16() || type.isBF16() || type.isF32())
    return 1;
  return std::nullopt;
}

static Type getSimtKeepResumeValueType(KeepOp op) {
  return op.getPayload().getType();
}

static Type getSimtKeepResumeValueType(ResumeOp op) {
  return op.getResult().getType();
}

template <typename OpT>
static LogicalResult verifySimtKeepResumeSlotRange(OpT op) {
  std::optional<unsigned> registerCount =
      getSimtKeepResumeRegisterCount(getSimtKeepResumeValueType(op));
  if (!registerCount)
    return success();
  int64_t slot = op.getSlot();
  if (slot < 0 || slot >= kSimtKeepResumeSlotLimit)
    return op.emitOpError()
           << "requires slot in range [0, "
           << (kSimtKeepResumeSlotLimit - 1) << "]";
  if (*registerCount == 2) {
    if ((slot % 2) != 0)
      return op.emitOpError()
             << "requires an even slot for 64-bit keep/resume values";
    if (slot + 1 >= kSimtKeepResumeSlotLimit)
      return op.emitOpError()
             << "requires slot in range [0, "
             << (kSimtKeepResumeSlotLimit - 2)
             << "] for 64-bit keep/resume values";
  }
  return success();
}

template <typename OpT>
static bool overlapsEarlierSimtKeepResumeSlotUse(OpT op,
                                                 SmallVectorImpl<int64_t> &used) {
  std::optional<unsigned> registerCount =
      getSimtKeepResumeRegisterCount(getSimtKeepResumeValueType(op));
  if (!registerCount)
    return false;
  int64_t slot = op.getSlot();
  for (int64_t word = slot; word < slot + *registerCount; ++word) {
    if (llvm::is_contained(used, word))
      return true;
  }
  for (int64_t word = slot; word < slot + *registerCount; ++word)
    used.push_back(word);
  return false;
}

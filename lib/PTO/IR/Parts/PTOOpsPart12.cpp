// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This implementation fragment is included by PTO.cpp and intentionally is
// not listed as a separate CMake translation unit.

void TPrintOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  if (!getTmpMutable().empty()) {
    PTO_ADD_WRITE(getTmpMutable()[0]);
  }
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

static bool isInsideTileOpHelper(Operation *op) {
  auto funcOp = op->getParentOfType<func::FuncOp>();
  return funcOp && funcOp->hasAttr("pto.tileop.helper");
}

static std::optional<FunctionKernelKind>
getEnclosingFunctionKernelKind(Operation *op) {
  auto funcOp = op->getParentOfType<func::FuncOp>();
  if (!funcOp) {
    return std::nullopt;
  }

  auto kernelKindAttr =
      funcOp->getAttrOfType<FunctionKernelKindAttr>(
          FunctionKernelKindAttr::name);
  if (!kernelKindAttr) {
    return std::nullopt;
  }

  return kernelKindAttr.getKernelKind();
}

static bool isInsideSectionOrAttributedKernel(Operation *op) {
  return isInsideSectionCube(op) || isInsideSectionVector(op) ||
         isInsideTileOpHelper(op) || getEnclosingFunctionKernelKind(op).has_value();
}

static LogicalResult verifySplitAttr(Operation *op, int64_t split) {
    if (split < 0 || split > mlir::pto::kValue4) {
        return op->emitOpError("expects 'split' to be 0, 1, 2, 3, or 4");
    }
  return success();
}

static bool isOddSplit(int64_t split) { return split == mlir::pto::kValue3 || split == mlir::pto::kValue4; }

static bool isInsideCubeKernelOrSection(Operation *op) {
  if (isInsideSectionCube(op)) {
    return true;
  }
  auto kernelKind = getEnclosingFunctionKernelKind(op);
  return kernelKind && *kernelKind == FunctionKernelKind::Cube;
}

static bool isInsideVectorKernelOrSection(Operation *op) {
  if (isInsideSectionVector(op)) {
    return true;
  }
  auto kernelKind = getEnclosingFunctionKernelKind(op);
  return kernelKind && *kernelKind == FunctionKernelKind::Vector;
}

static LogicalResult verifyFrontendKernelKind(Operation *op,
                                              FunctionKernelKind expected,
                                              StringRef kernelName) {
  if (isInsideTileOpHelper(op)) {
    return success();
  }
  if (isInsideSectionCube(op)) {
    if (expected == FunctionKernelKind::Cube) {
      return success();
    }
    return op->emitOpError("must be inside a ")
           << kernelName << " kernel function or section";
  }
  if (isInsideSectionVector(op)) {
    if (expected == FunctionKernelKind::Vector) {
      return success();
    }
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

struct FrontendInitParseState {
  NamedAttrList attrs;
  SmallVector<OpAsmParser::UnresolvedOperand, mlir::pto::kValue4> operands =
      SmallVector<OpAsmParser::UnresolvedOperand, 4>(4);
  SmallVector<Type, mlir::pto::kValue4> types = SmallVector<Type, mlir::pto::kValue4>(mlir::pto::kValue4);
  std::array<bool, mlir::pto::kValue7> sawAttr{};
  std::array<bool, mlir::pto::kValue4> sawOperand{};
};

static int getFrontendInitAttrIndex(StringRef keyword) {
    return llvm::StringSwitch<int>(keyword)
        .Case("id", 0)
        .Case("dir_mask", 1)
        .Case("slot_size", mlir::pto::kValue2)
        .Case("slot_num", mlir::pto::kValue3)
        .Case("local_slot_num", mlir::pto::kValue4)
        .Case("nosplit", mlir::pto::kValue5)
        .Case("acc_push_epilogue", 6)
        .Default(-1);
}

static ParseResult parseFrontendInitAttrClause(OpAsmParser &parser,
                                               FrontendInitParseState &state) {
  StringRef keyword;
  if (parser.parseKeyword(&keyword) || parser.parseEqual())
    return failure();
  int index = getFrontendInitAttrIndex(keyword);
  if (index < 0)
    return parser.emitError(parser.getCurrentLocation())
           << "unexpected keyword '" << keyword << "'";
  if (state.sawAttr[index])
    return parser.emitError(parser.getCurrentLocation())
           << "duplicate '" << keyword << "' clause";
  state.sawAttr[index] = true;
  if (index <= mlir::pto::kValue4) {
      IntegerAttr value;
      Type type = index == 1 ? parser.getBuilder().getI8Type() : parser.getBuilder().getI32Type();
      return parser.parseAttribute(value, type, keyword, state.attrs);
  }
  if (index == mlir::pto::kValue5) {
      BoolAttr value;
      return parser.parseAttribute(value, "nosplit", state.attrs);
  }
  AccPushEpilogueAttr value;
  return parser.parseAttribute(value, "acc_push_epilogue", state.attrs);
}

static ParseResult parseFrontendInitAttrs(OpAsmParser &parser,
                                          FrontendInitParseState &state) {
  if (parser.parseLBrace())
    return failure();
  while (failed(parser.parseOptionalRBrace())) {
    if (failed(parseFrontendInitAttrClause(parser, state)))
      return failure();
    if (succeeded(parser.parseOptionalRBrace()))
      break;
    if (parser.parseComma())
      return failure();
  }
  if (!state.sawAttr[1])
    return parser.emitError(parser.getNameLoc(), "expected 'dir_mask' clause");
  if (!state.sawAttr[2])
    return parser.emitError(parser.getNameLoc(), "expected 'slot_size' clause");
  if (!state.sawAttr[0])
    state.attrs.set("id", parser.getBuilder().getI32IntegerAttr(0));
  return success();
}

static int getFrontendInitOperandIndex(StringRef keyword) {
    return llvm::StringSwitch<int>(keyword)
        .Case("gm_slot_buffer", 0)
        .Case("gm_slot_tensor", 1)
        .Case("c2v_consumer_buf", mlir::pto::kValue2)
        .Case("v2c_consumer_buf", mlir::pto::kValue3)
        .Default(-1);
}

static ParseResult parseFrontendInitOperands(OpAsmParser &parser,
                                             FrontendInitParseState &state) {
  if (parser.parseLParen())
    return failure();
  while (failed(parser.parseOptionalRParen())) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword) || parser.parseEqual())
      return failure();
    int index = getFrontendInitOperandIndex(keyword);
    if (index < 0)
      return parser.emitError(parser.getCurrentLocation())
             << "unexpected initialize_pipe operand '" << keyword << "'";
    if (state.sawOperand[index])
      return parser.emitError(parser.getCurrentLocation())
             << "duplicate '" << keyword << "' operand";
    if (parser.parseOperand(state.operands[index]) ||
        parser.parseColonType(state.types[index]))
      return failure();
    state.sawOperand[index] = true;
    if (succeeded(parser.parseOptionalRParen()))
      break;
    if (parser.parseComma())
      return failure();
  }
  return success();
}

static ParseResult resolveFrontendInitOperands(OpAsmParser &parser,
                                               OperationState &result,
                                               FrontendInitParseState &state) {
  result.addAttributes(state.attrs);
  SmallVector<int32_t, mlir::pto::kValue4> segments;
  for (bool present : state.sawOperand)
    segments.push_back(present ? 1 : 0);
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(segments));
  for (unsigned i = 0; i < state.operands.size(); ++i) {
    if (state.sawOperand[i] &&
        parser.resolveOperand(state.operands[i], state.types[i],
                              result.operands))
      return failure();
  }
  return success();
}

static ParseResult parseFrontendInitializePipeOp(OpAsmParser &parser,
                                                 OperationState &result) {
  FrontendInitParseState state;
  if (failed(parseFrontendInitAttrs(parser, state)) ||
      failed(parseFrontendInitOperands(parser, state)) ||
      parser.parseOptionalAttrDict(state.attrs))
    return failure();
  return resolveFrontendInitOperands(parser, result, state);
}

template <typename InitOpT>
static void printFrontendInitAttrs(InitOpT op, OpAsmPrinter &p) {
  p << " {";
  bool needsComma = false;
  auto printClause = [&](StringRef keyword, auto value) {
    if (needsComma) {
      p << ", ";
    }
    p << keyword << " = " << value;
    needsComma = true;
  };

  printClause("id", op.getId());
  printClause("dir_mask", static_cast<int32_t>(op.getDirMask()));
  printClause("slot_size", op.getSlotSize());
  if (auto slotNumAttr = op.getSlotNumAttr()) {
    printClause("slot_num", slotNumAttr.getInt());
  }
  if (auto localSlotNumAttr = op.getLocalSlotNumAttr()) {
    printClause("local_slot_num", localSlotNumAttr.getInt());
  }
  if (auto noSplitAttr = op.getNosplitAttr()) {
    printClause("nosplit", noSplitAttr.getValue() ? "true" : "false");
  }
  if (auto accPushEpilogueAttr = op.getAccPushEpilogueAttr()) {
    printClause("acc_push_epilogue", accPushEpilogueAttr);
  }
  p << "}";
}

template <typename InitOpT>
static void printFrontendInitOperands(InitOpT op, OpAsmPrinter &p) {
  p << "(";
  bool needsOperandComma = false;
  auto printOperandClause = [&](StringRef keyword, Value value) {
    if (needsOperandComma) {
      p << ", ";
    }
    p << keyword << " = " << value << " : " << value.getType();
    needsOperandComma = true;
  };
  if (op.getGmSlotBuffer()) {
    printOperandClause("gm_slot_buffer", op.getGmSlotBuffer());
  }
  if (op.getGmSlotTensor()) {
    printOperandClause("gm_slot_tensor", op.getGmSlotTensor());
  }
  if (op.getC2vConsumerBuf()) {
    printOperandClause("c2v_consumer_buf", op.getC2vConsumerBuf());
  }
  if (op.getV2cConsumerBuf()) {
    printOperandClause("v2c_consumer_buf", op.getV2cConsumerBuf());
  }
  p << ")";
}

template <typename InitOpT>
static void printFrontendInitializePipeOp(InitOpT op, OpAsmPrinter &p) {
  printFrontendInitAttrs(op, p);
  printFrontendInitOperands(op, p);
  p.printOptionalAttrDict(
      op->getAttrs(),
      /*elidedAttrs=*/{"id", "dir_mask", "slot_size", "slot_num",
                       "local_slot_num", "acc_push_epilogue",
                       "nosplit", "operandSegmentSizes"});
}

static std::optional<uint64_t>
getStaticElementCount(ArrayRef<int64_t> shape) {
  uint64_t count = 1;
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic || dim < 0) {
      return std::nullopt;
    }
    count *= static_cast<uint64_t>(dim);
  }
  return count;
}

static bool isSameOrHalfSlotByteSize(uint64_t tensorBytes, uint64_t slotBytes) {
    return tensorBytes == slotBytes || tensorBytes * mlir::pto::kValue2 == slotBytes;
}

static LogicalResult verifyFrontendGlobalSlotTensor(Operation *op, Value tensor,
                                                    int8_t dirMask,
                                                    int32_t slotSize) {
  (void)dirMask;
  auto tvTy = dyn_cast<TensorViewType>(tensor.getType());
  if (!tvTy) {
    return op->emitOpError("expects 'gm_slot_tensor' to be !pto.tensor_view");
  }

  ArrayRef<int64_t> shape = tvTy.getShape();
  if (shape.empty()) {
    return op->emitOpError(
        "expects 'gm_slot_tensor' to describe one slot entry tensor");
  }

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
static FailureOr<int32_t> verifyFrontendInitIdentity(InitOpT op,
                                                     FunctionKernelKind expected,
                                                     StringRef kernelName) {
  if (failed(verifyNoUnpublishedFixpipeFrontendAttrs(op.getOperation())) ||
      failed(verifyFrontendKernelKind(op.getOperation(), expected, kernelName)))
    return failure();
  auto funcOp = op->template getParentOfType<func::FuncOp>();
  if (!funcOp) {
    op.emitOpError("must be nested under a func.func");
    return failure();
  }
  if (op.getId() < 0) {
    op.emitOpError("expects 'id' to be non-negative");
    return failure();
  }
  unsigned matches = 0;
  funcOp.walk([&](Operation *candidate) {
    if (auto init = dyn_cast<AicInitializePipeOp>(candidate))
      matches += init.getId() == op.getId();
    else if (auto init = dyn_cast<AivInitializePipeOp>(candidate))
      matches += init.getId() == op.getId();
  });
  if (matches > 1) {
    op.emitOpError(
        "requires 'id' to be unique across frontend initialize_pipe ops in the function");
    return failure();
  }
  int8_t dirMask = op.getDirMask();
  if (dirMask != 1 && dirMask != mlir::pto::kValue2 && dirMask != mlir::pto::kValue3) {
      op.emitOpError("expects 'dir_mask' to be 1, 2, or 3");
      return failure();
  }
  if (op.getSlotSize() <= 0) {
    op.emitOpError("expects 'slot_size' to be greater than 0");
    return failure();
  }
  int32_t slotNum = op.getSlotNumAttr()
                        ? op.getSlotNumAttr().getInt()
                        : (dirMask == 3 ? 4 : 8);
  if (slotNum <= 0) {
    op.emitOpError("expects 'slot_num' to be greater than 0");
    return failure();
  }
  return slotNum;
}

template <typename InitOpT>
static FailureOr<bool> verifyFrontendInitGlobalBacking(InitOpT op,
                                                       PTOArch arch) {
  if (!op.getGmSlotTensor())
    return false;
  if (op.getGmSlotBuffer()) {
    op.emitOpError("'gm_slot_tensor' cannot be combined with 'gm_slot_buffer'");
    return failure();
  }
  bool c2v = static_cast<bool>(op.getC2vConsumerBuf());
  bool v2c = static_cast<bool>(op.getV2cConsumerBuf());
  int8_t dirMask = op.getDirMask();
  bool supportedC2V = dirMask == 1 && c2v && !v2c;
  bool supportedA2A3V2C = arch != PTOArch::A5 &&
      ((dirMask == 2 && !c2v && v2c) || (dirMask == 3 && c2v && v2c));
  if ((c2v || v2c) && !supportedC2V && !supportedA2A3V2C) {
    op.emitOpError(
        "GM-backed tile pipe init supports dir_mask = 1 with "
        "'c2v_consumer_buf' on all targets and dir_mask = 2/3 with "
        "matching consumer buffers only on a2/a3");
    return failure();
  }
  if (failed(verifyFrontendGlobalSlotTensor(op, op.getGmSlotTensor(), dirMask,
                                            op.getSlotSize())))
    return failure();
  bool globalOnly = !c2v && !v2c;
  if (globalOnly && op.getLocalSlotNumAttr()) {
    op.emitOpError("globaltensor pipe init does not use 'local_slot_num'");
    return failure();
  }
  return globalOnly;
}

template <typename InitOpT>
static LogicalResult verifyFrontendInitLocalBuffers(InitOpT op) {
  bool c2v = static_cast<bool>(op.getC2vConsumerBuf());
  bool v2c = static_cast<bool>(op.getV2cConsumerBuf());
  int8_t dirMask = op.getDirMask();
  if (!c2v && !v2c)
    return op.emitOpError(
        "expects local pipe init to provide at least one consumer buffer "
        "operand; use 'gm_slot_tensor' for globaltensor pipe entries");
  if (dirMask == 1 && !c2v)
    return op.emitOpError(
        "expects 'c2v_consumer_buf' when dir_mask is 1");
  if (dirMask == mlir::pto::kValue2 && !v2c)
      return op.emitOpError("expects 'v2c_consumer_buf' when dir_mask is 2");
  if (dirMask == mlir::pto::kValue3 && (!c2v || !v2c))
      return op.emitOpError("expects both 'c2v_consumer_buf' and 'v2c_consumer_buf' when dir_mask is 3");
  return success();
}

template <typename InitOpT>
static LogicalResult verifyFrontendInitLocalSlots(InitOpT op, PTOArch arch,
                                                  int32_t slotNum) {
  auto attr = op.getLocalSlotNumAttr();
  if (!attr)
    return success();
  if (arch == PTOArch::A5)
    return op.emitOpError(
        "'local_slot_num' is only supported for a2/a3 frontend pipe lowering");
  int32_t localSlots = attr.getInt();
  if (localSlots <= 0)
    return op.emitOpError("expects 'local_slot_num' to be greater than 0");
  if (localSlots > slotNum)
    return op.emitOpError()
           << "expects 'local_slot_num' to be less than or equal to slot_num ("
           << slotNum << ") for dir_mask = "
           << static_cast<int>(op.getDirMask());
  return success();
}

static bool isAllowedFrontendFixpipeQuant(pto::FixpipeQuant quant) {
    static constexpr pto::FixpipeQuant kAllowedQuants[] = {
        pto::FixpipeQuant::NoConvert,
        pto::FixpipeQuant::F32F16,
        pto::FixpipeQuant::F32BF16,
        pto::FixpipeQuant::REQ8Scalar,
        pto::FixpipeQuant::REQ8Vec,
        pto::FixpipeQuant::DEQF16Scalar,
        pto::FixpipeQuant::DEQF16Vec,
        pto::FixpipeQuant::QF322B8PreScalar,
        pto::FixpipeQuant::QF322B8PreVec,
        pto::FixpipeQuant::QF322F16PreScalar,
        pto::FixpipeQuant::QF322BF16PreScalar,
        pto::FixpipeQuant::QS322BF16PreScalar,
        pto::FixpipeQuant::QS322BF16PreVec,
        pto::FixpipeQuant::QF322HIF8PreScalar,
        pto::FixpipeQuant::QF322FP8PreScalar,
    };
    if (llvm::is_contained(kAllowedQuants, quant)) {
        return true;
    }
    llvm_unreachable("unhandled FixpipeQuant");
}

static bool isA5OnlyFrontendFixpipeQuant(pto::FixpipeQuant quant) {
  return quant == pto::FixpipeQuant::QS322BF16PreScalar ||
         quant == pto::FixpipeQuant::QS322BF16PreVec ||
         quant == pto::FixpipeQuant::QF322HIF8PreScalar ||
         quant == pto::FixpipeQuant::QF322FP8PreScalar;
}

template <typename InitOpT>
static LogicalResult verifyFrontendInitFixpipe(InitOpT op, PTOArch arch) {
  auto epilogue = op.getAccPushEpilogueAttr();
  if (!epilogue)
    return success();
  if (op.getDirMask() != 1)
    return op.emitOpError(
        "expects fixpipe pipe (with 'acc_push_epilogue') to have dir_mask = 1 (C2V only)");
  if (!op.getNosplit())
    return op.emitOpError(
        "expects fixpipe pipe (with 'acc_push_epilogue') to have nosplit = true");
  if (op.getC2vConsumerBuf()) {
    Operation *def = op.getC2vConsumerBuf().getDefiningOp();
    if (!def || !isa<ReserveBufferOp, ImportReservedBufferOp>(def))
      return op.emitOpError(
          "expects fixpipe pipe 'c2v_consumer_buf' to trace to reserve_buffer or "
          "import_reserved_buffer for peer contract verification");
  }
  auto relu = epilogue.getRelu();
  if (relu != pto::FixpipeRelu::NoRelu &&
      relu != pto::FixpipeRelu::NormalRelu)
    return op.emitOpError(
        "expects 'acc_push_epilogue.relu' to be 'no_relu' or 'normal_relu' in v1");
  auto quant = epilogue.getQuant();
  if (!isAllowedFrontendFixpipeQuant(quant))
    return op.emitOpError(
        "expects 'acc_push_epilogue.quant' to be one of the v1 allowed quantization modes");
  if (arch != PTOArch::A5 && isA5OnlyFrontendFixpipeQuant(quant))
    return op.emitOpError(
        quant == pto::FixpipeQuant::QS322BF16PreScalar ||
                quant == pto::FixpipeQuant::QS322BF16PreVec
            ? "expects 'qs322bf16_pre_*' quantization modes to be used only on A5 target"
            : "expects 'qf322hif8_pre_scalar'/'qf322fp8_pre_scalar' to be used only on A5 target");
  return success();
}
template <typename InitOpT>
static LogicalResult verifyFrontendInitCommon(InitOpT op,
                                              FunctionKernelKind expected,
                                              StringRef kernelName) {
  auto slotNumResult = verifyFrontendInitIdentity(op, expected, kernelName);
  if (failed(slotNumResult))
    return failure();
  int32_t slotNum = *slotNumResult;
  PTOArch arch = getTargetArch(op.getOperation());
  auto globalOnly = verifyFrontendInitGlobalBacking(op, arch);
  if (failed(globalOnly))
    return failure();
  if (*globalOnly)
    return success();
  if (failed(verifyFrontendInitLocalBuffers(op)) ||
      failed(verifyFrontendInitLocalSlots(op, arch, slotNum)))
    return failure();

  return verifyFrontendInitFixpipe(op, arch);
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
    if (reserveOp.getName() != name) {
      return WalkResult::advance();
    }
    found = reserveOp;
    return WalkResult::interrupt();
  });
  return found;
}

LogicalResult ReserveBufferOp::verify() {
  auto funcOp = getOperation()->getParentOfType<func::FuncOp>();
  if (!funcOp) {
    return emitOpError("must be nested under a func.func");
  }

  if (getSize() <= 0) {
    return emitOpError("expects 'size' to be greater than 0");
  }

  auto location = getLocation().getAddressSpace();
  if (location != AddressSpace::VEC && location != AddressSpace::MAT) {
    return emitOpError("expects 'location' to be #pto.address_space<vec> or #pto.address_space<mat>");
  }

  if (!getAutoAlloc() && !getBaseAttr()) {
    return emitOpError("expects 'base' when 'auto' is false");
  }

  if (auto baseAttr = getBaseAttr(); baseAttr && baseAttr.getInt() < 0) {
    return emitOpError("expects 'base' to be non-negative when present");
  }

  unsigned sameNameCount = 0;
  funcOp.walk([&](ReserveBufferOp reserveOp) {
    if (reserveOp.getName() == getName()) {
      ++sameNameCount;
    }
  });
  if (sameNameCount > 1) {
    return emitOpError("requires 'name' to be unique within the function");
  }

  return success();
}

LogicalResult ImportReservedBufferOp::verify() {
  auto funcOp = getOperation()->getParentOfType<func::FuncOp>();
  if (!funcOp) {
    return emitOpError("must be nested under a func.func");
  }

  auto peerFunc = lookupPeerFuncAcrossContainer(getOperation(), getPeerFuncAttr());
  if (!peerFunc) {
    return emitOpError("expects 'peer_func' to reference an existing func.func");
  }

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

  if (!findReserveBufferByName(peerFunc, getName())) {
    return emitOpError("expects matching peer reserve_buffer to exist");
  }

  return success();
}

constexpr llvm::StringLiteral kFrontendPipeIdAttrName = "__pto.frontend_id";
constexpr llvm::StringLiteral kPipePeerOwnerFuncAttrName =
    "__pto.peer_owner_func";
constexpr llvm::StringLiteral kPipePeerReserveNameAttrName =
    "__pto.peer_reserve_name";
constexpr llvm::StringLiteral kPipePeerDirMaskAttrName = "__pto.peer_dir_mask";

struct FixpipeQuantStateResource
    : public SideEffects::Resource::Base<FixpipeQuantStateResource> {
  StringRef getName() final { return "PTOFixpipeQuantState"; }
};

static IntegerAttr getFixpipeQuantStateIdAttr(Operation *op, int32_t id) {
    return IntegerAttr::get(IntegerType::get(op->getContext(), mlir::pto::kValue32), id);
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

static std::optional<int32_t> getFrontendPipeIdFromHandle(Value pipeHandle) {
  if (!pipeHandle) {
    return std::nullopt;
  }
  Operation *defOp = pipeHandle.getDefiningOp();
  if (!defOp) {
    return std::nullopt;
  }
  auto frontendIdAttr = defOp->getAttrOfType<IntegerAttr>(kFrontendPipeIdAttrName);
  if (!frontendIdAttr) {
    return std::nullopt;
  }
  return static_cast<int32_t>(frontendIdAttr.getInt());
}

static pto::AccPushEpilogueAttr getAccPushEpilogueFromInitOp(Operation *initOp) {
  if (!initOp) {
    return {};
  }
  if (auto aicInit = dyn_cast<AicInitializePipeOp>(initOp)) {
    return aicInit.getAccPushEpilogueAttr();
  }
  if (auto aivInit = dyn_cast<AivInitializePipeOp>(initOp)) {
    return aivInit.getAccPushEpilogueAttr();
  }
  if (auto l2lInit = dyn_cast<InitializeL2LPipeOp>(initOp)) {
    return l2lInit.getAccPushEpilogueAttr();
  }
  if (auto l2g2lInit = dyn_cast<InitializeL2G2LPipeOp>(initOp)) {
    return l2g2lInit.getAccPushEpilogueAttr();
  }
  return {};
}

static bool matchesLoweredFixpipePeerContract(Operation *initOp,
                                              func::FuncOp expectedOwnerFunc,
                                              StringRef expectedReserveName) {
  if (!initOp || !isa<InitializeL2LPipeOp, InitializeL2G2LPipeOp>(initOp)) {
    return false;
  }

  auto ownerAttr =
      initOp->getAttrOfType<FlatSymbolRefAttr>(kPipePeerOwnerFuncAttrName);
  auto reserveAttr =
      initOp->getAttrOfType<StringAttr>(kPipePeerReserveNameAttrName);
  auto dirMaskAttr =
      initOp->getAttrOfType<IntegerAttr>(kPipePeerDirMaskAttrName);
  if (!ownerAttr || !reserveAttr || !dirMaskAttr) {
    return false;
  }

  if (ownerAttr.getValue() != expectedOwnerFunc.getSymName() ||
      reserveAttr.getValue() != expectedReserveName ||
      dirMaskAttr.getInt() != 1) {
    return false;
  }

  return static_cast<bool>(getAccPushEpilogueFromInitOp(initOp));
}

static FailureOr<Operation *> lookupFrontendOrLoweredInitOpById(
    Operation *op, func::FuncOp funcOp, int32_t id);

static FailureOr<Operation *>
lookupFixpipePeerConsumerInit(AicInitializePipeOp producerInit) {
  if (!producerInit.getC2vConsumerBuf()) {
    return failure();
  }
  auto importOp = dyn_cast_or_null<ImportReservedBufferOp>(
      producerInit.getC2vConsumerBuf().getDefiningOp());
  if (!importOp) {
    return failure();
  }

  auto peerConsumerFunc =
      lookupPeerFuncAcrossContainer(importOp.getOperation(),
                                    importOp.getPeerFuncAttr());
  if (!peerConsumerFunc) {
    return failure();
  }

  StringRef bufferName = importOp.getName();
  Operation *matchedInit = nullptr;
  unsigned matchedInitCount = 0;
  peerConsumerFunc.walk([&](Operation *candidate) {
    if (auto aivInit = dyn_cast<AivInitializePipeOp>(candidate)) {
      if (!aivInit.getC2vConsumerBuf()) {
        return WalkResult::advance();
      }
      auto reserveOp = dyn_cast_or_null<ReserveBufferOp>(
          aivInit.getC2vConsumerBuf().getDefiningOp());
      if (!reserveOp || reserveOp.getName() != bufferName) {
        return WalkResult::advance();
      }
      matchedInit = candidate;
      ++matchedInitCount;
      return WalkResult::advance();
    }

    if (!matchesLoweredFixpipePeerContract(candidate, peerConsumerFunc,
                                           bufferName)) {
      return WalkResult::advance();
    }

    matchedInit = candidate;
    ++matchedInitCount;
    return WalkResult::advance();
  });

  if (matchedInitCount != 1) {
    return failure();
  }
  return matchedInit;
}

static FailureOr<Operation *>
lookupFixpipePeerProducerInit(AivInitializePipeOp consumerInit,
                              func::FuncOp peerProducerFunc,
                              StringRef bufferName,
                              func::FuncOp currentConsumerFunc) {
  Operation *matchedInit = nullptr;
  unsigned matchedInitCount = 0;
  peerProducerFunc.walk([&](Operation *candidate) {
    if (auto aicInit = dyn_cast<AicInitializePipeOp>(candidate)) {
      if (!aicInit.getC2vConsumerBuf()) {
        return WalkResult::advance();
      }

      auto importOp = dyn_cast_or_null<ImportReservedBufferOp>(
          aicInit.getC2vConsumerBuf().getDefiningOp());
      if (!importOp) {
        return WalkResult::advance();
      }

      auto peerConsumerFunc =
          lookupPeerFuncAcrossContainer(importOp.getOperation(),
                                        importOp.getPeerFuncAttr());
      if (importOp.getName() != bufferName ||
          peerConsumerFunc != currentConsumerFunc) {
        return WalkResult::advance();
      }

      matchedInit = candidate;
      ++matchedInitCount;
      return WalkResult::advance();
    }

    if (!matchesLoweredFixpipePeerContract(candidate, currentConsumerFunc,
                                           bufferName)) {
      return WalkResult::advance();
    }

    matchedInit = candidate;
    ++matchedInitCount;
    return WalkResult::advance();
  });

  if (matchedInitCount != 1) {
    consumerInit.emitOpError()
        << "expects peer producer function to contain a matching "
           "aic_initialize_pipe with the same consumer buffer contract";
    return failure();
  }

  return matchedInit;
}

static std::pair<Operation *, unsigned> findFrontendInitById(
    func::FuncOp funcOp, int32_t id) {
  Operation *matched = nullptr;
  unsigned count = 0;
  funcOp.walk([&](Operation *candidate) {
    if (auto aic = dyn_cast<AicInitializePipeOp>(candidate)) {
      if (aic.getId() == static_cast<uint32_t>(id)) {
        matched = candidate;
        ++count;
      }
      return WalkResult::advance();
    }
    if (auto aiv = dyn_cast<AivInitializePipeOp>(candidate)) {
      if (aiv.getId() == static_cast<uint32_t>(id)) {
        matched = candidate;
        ++count;
      }
      return WalkResult::advance();
    }
    return WalkResult::advance();
  });
  return {matched, count};
}

static std::pair<Operation *, unsigned> findLoweredInitById(
    func::FuncOp funcOp, int32_t id) {
  Operation *matched = nullptr;
  unsigned count = 0;
  funcOp.walk([&](Operation *candidate) {
    if (!isa<InitializeL2LPipeOp, InitializeL2G2LPipeOp>(candidate)) {
      return WalkResult::advance();
    }
    auto frontendIdAttr =
        candidate->getAttrOfType<IntegerAttr>(kFrontendPipeIdAttrName);
    if (!frontendIdAttr || frontendIdAttr.getInt() != id) {
      return WalkResult::advance();
    }
    matched = candidate;
    ++count;
    return WalkResult::advance();
  });
  return {matched, count};
}

static FailureOr<Operation *> lookupFrontendOrLoweredInitOpById(
    Operation *op, func::FuncOp funcOp, int32_t id) {
  auto [frontend, frontendCount] = findFrontendInitById(funcOp, id);
  if (frontendCount == 1)
    return frontend;
  if (frontendCount > 1) {
    op->emitOpError() << "expects 'id' = " << id
                      << " to match exactly one frontend initialize_pipe op in the same function";
    return failure();
  }
  auto [lowered, loweredCount] = findLoweredInitById(funcOp, id);
  if (loweredCount == 0) {
    op->emitOpError() << "expects 'id' = " << id
                      << " to match a frontend or lowered initialize_pipe op in the same function";
    return failure();
  }
  if (loweredCount > 1) {
    op->emitOpError() << "expects 'id' = " << id
                      << " to match exactly one frontend or lowered initialize_pipe op in the same function";
    return failure();
  }
  return lowered;
}

template <typename InitOpT>
static bool supportsFrontendOddDirection(Operation *op, InitOpT init,
                                         bool expectC2V) {
  PTOArch arch = getTargetArch(op);
  bool gmBacking = static_cast<bool>(init.getGmSlotTensor()) ||
                   (arch != PTOArch::A5 &&
                    static_cast<bool>(init.getGmSlotBuffer()));
  int8_t directionMask = expectC2V ? 1 : 2;
  Value buffer = expectC2V ? init.getC2vConsumerBuf()
                           : init.getV2cConsumerBuf();
  return (init.getDirMask() & directionMask) != 0 && gmBacking && buffer;
}

static FailureOr<func::FuncOp> getRequiredParentFunc(Operation *op) {
  auto funcOp = op->getParentOfType<func::FuncOp>();
  if (!funcOp) {
    op->emitOpError("must be nested under a func.func");
    return failure();
  }
  return funcOp;
}

static FailureOr<Operation *> getRequiredFrontendInit(Operation *op,
                                                      int32_t id) {
  auto funcOp = getRequiredParentFunc(op);
  if (failed(funcOp))
    return failure();
  return lookupFrontendInitOpById(op, *funcOp, id);
}

static LogicalResult verifyFrontendSplitOp(Operation *op,
                                           FunctionKernelKind expected,
                                           StringRef kernelName,
                                           int32_t id,
                                           int64_t split,
                                           bool expectC2V) {
  if (failed(verifyFrontendKernelKind(op, expected, kernelName))) {
    return failure();
  }
  if (id < 0) {
    return op->emitOpError("expects 'id' to be non-negative");
  }
  if (failed(verifySplitAttr(op, split))) {
    return failure();
  }
  if (!isOddSplit(split)) {
    return success();
  }

  auto initOr = getRequiredFrontendInit(op, id);
  if (failed(initOr)) {
    return failure();
  }
  if (!expectC2V && getTargetArch(op) == PTOArch::A5) {
    return op->emitOpError(
        "supports odd V2C split modes (split = 3 or 4) only on a2/a3");
  }

  bool supported = false;
  if (auto aic = dyn_cast<AicInitializePipeOp>(*initOr)) {
    supported = supportsFrontendOddDirection(op, aic, expectC2V);
  } else {
    supported = supportsFrontendOddDirection(
        op, cast<AivInitializePipeOp>(*initOr), expectC2V);
  }
  if (!supported) {
    return op->emitOpError()
           << "supports odd split modes (split = 3 or 4) only for a "
              "GM-backed tile pipe whose dir_mask enables "
           << (expectC2V ? "C2V" : "V2C")
           << " and provides the matching consumer buffer";
  }
  return success();
}

static LogicalResult verifyOddSplitTileEntry(Operation *op, int64_t split,
                                             Type entryTy) {
  if (isOddSplit(split) && isa<TensorViewType>(entryTy)) {
    return op->emitOpError(
        "supports odd split modes (split = 3 or 4) only for tile entries; "
        "the pinned pto-isa does not implement odd GlobalTensor offsets");
  }
  return success();
}

static LogicalResult verifyFullTileSplitParity(Operation *op, int64_t split,
                                               Type entryTy) {
  if (split == 0) {
    return success();
  }

  ArrayRef<int64_t> shape;
  if (auto tileTy = dyn_cast<TileBufType>(entryTy)) {
    shape = tileTy.getValidShape();
  } else if (auto viewTy = dyn_cast<TensorViewType>(entryTy)) {
    shape = viewTy.getShape();
  } else {
    return success();
}
if (shape.size() != mlir::pto::kValue2) {
    return success();
}

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

static bool matchesFixpipeConsumerElementType(FixpipeQuant quant,
                                              Type resultElemType) {
  switch (quant) {
  case FixpipeQuant::NoConvert:
      return resultElemType.isF32() || resultElemType.isInteger(mlir::pto::kValue32);
  case FixpipeQuant::F32F16:
  case FixpipeQuant::DEQF16Scalar:
  case FixpipeQuant::DEQF16Vec:
  case FixpipeQuant::QF322F16PreScalar:
    return resultElemType.isF16();
  case FixpipeQuant::F32BF16:
  case FixpipeQuant::QF322BF16PreScalar:
  case FixpipeQuant::QS322BF16PreScalar:
  case FixpipeQuant::QS322BF16PreVec:
    return resultElemType.isBF16();
  case FixpipeQuant::REQ8Scalar:
  case FixpipeQuant::QF322B8PreScalar:
    return isSignedOrUnsignedI8(resultElemType);
  case FixpipeQuant::REQ8Vec:
  case FixpipeQuant::QF322B8PreVec:
    return isSignedI8(resultElemType);
  case FixpipeQuant::QF322HIF8PreScalar:
    return isa<HiF8Type>(resultElemType);
  case FixpipeQuant::QF322FP8PreScalar:
    return isPTOFloat8E4M3LikeType(resultElemType);
  }
  llvm_unreachable("unhandled FixpipeQuant");
}

static bool isFixpipeQuantPayloadElemType(Type elemTy, PTOArch arch) {
  if (!elemTy) {
    return false;
  }
  bool isPackedI64 = elemTy.isUnsignedInteger(64) ||
                     elemTy.isSignlessInteger(64) ||
                     elemTy.isSignedInteger(64);
  // SET_QUANT_VECTOR passes each column to the hardware as a packed 64-bit
  // control word. The frontend does not convert floating-point elements into
  // that representation, so accepting f16/bf16/f32 would produce invalid
  // quantization parameters on both A3 and A5.
  return (arch == PTOArch::A3 || arch == PTOArch::A5) && isPackedI64;
}

static bool matchesFixpipeProducerElementType(FixpipeQuant quant,
                                              Type srcElemType) {
  switch (quant) {
  case FixpipeQuant::NoConvert:
      return srcElemType.isF32() || srcElemType.isInteger(mlir::pto::kValue32);
  case FixpipeQuant::F32F16:
  case FixpipeQuant::F32BF16:
  case FixpipeQuant::QF322B8PreScalar:
  case FixpipeQuant::QF322B8PreVec:
  case FixpipeQuant::QF322F16PreScalar:
  case FixpipeQuant::QF322BF16PreScalar:
  case FixpipeQuant::QF322HIF8PreScalar:
  case FixpipeQuant::QF322FP8PreScalar:
    return srcElemType.isF32();
  case FixpipeQuant::REQ8Scalar:
  case FixpipeQuant::REQ8Vec:
  case FixpipeQuant::DEQF16Scalar:
  case FixpipeQuant::DEQF16Vec:
  case FixpipeQuant::QS322BF16PreScalar:
  case FixpipeQuant::QS322BF16PreVec:
      return srcElemType.isInteger(mlir::pto::kValue32);
  }
  llvm_unreachable("unhandled FixpipeQuant");
}

static bool matchesFixpipeProducerAndConsumerTypes(FixpipeQuant quant,
                                                   Type srcElemType,
                                                   Type dstElemType) {
  return matchesFixpipeProducerElementType(quant, srcElemType) &&
         matchesFixpipeConsumerElementType(quant, dstElemType) &&
         (quant != FixpipeQuant::NoConvert || srcElemType == dstElemType);
}

static bool isUnpublishedFixpipeFrontendAttrName(StringRef name) {
  return llvm::StringSwitch<bool>(name)
      .Case("stPhase", true)
      .Case("st_phase", true)
      .Case("atomicType", true)
      .Case("atomic_type", true)
      .Case("subBlockId", true)
      .Case("subBlockid", true)
      .Case("sub_blockid", true)
      .Case("clipReluMode", true)
      .Case("clip_relu_mode", true)
      .Case("isChannelSplit", true)
      .Case("is_channel_split", true)
      .Case("channelSplit", true)
      .Case("channel_split", true)
      .Default(false);
}

static LogicalResult verifyNoUnpublishedFixpipeFrontendAttrs(Operation *op) {
  for (NamedAttribute attr : op->getAttrs()) {
    StringRef name = attr.getName().getValue();
    if (!isUnpublishedFixpipeFrontendAttrName(name)) {
      continue;
    }
    return op->emitOpError()
           << "does not allow unpublished fixpipe attr '" << name
           << "'; STPhase / AtomicType / SubBlockId / ClipReluMode / "
              "IsChannelSplit are not part of the PTOIR frontend surface";
  }
  return success();
}

static std::optional<uint64_t> getStaticTileByteSize(TileBufType tileTy) {
  auto shape = getShapeVec(tileTy);
  auto elemCount = getStaticElementCount(shape);
  uint64_t elemBytes = getElemByteSize(tileTy.getElementType());
  if (!elemCount || elemBytes == 0) {
    return std::nullopt;
  }
  return *elemCount * elemBytes;
}

static LogicalResult verifyFixpipePeerPopTypes(Operation *tpopOp,
                                               func::FuncOp funcOp, int32_t id,
                                               Type resultTileType) {
  bool mismatch = false;
  funcOp.walk([&](TPopFromAicOp otherPop) {
    if (otherPop.getOperation() == tpopOp ||
        otherPop.getId() != static_cast<uint32_t>(id))
      return WalkResult::advance();
    if (otherPop.getTile().getType() != resultTileType) {
      mismatch = true;
      tpopOp->emitOpError()
          << "expects all tpop_from_aic results for fixpipe pipe id = " << id
          << " to use the same tile type";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return success(!mismatch);
}

static LogicalResult verifyFixpipeConsumerType(Operation *tpopOp, int32_t id,
                                                Type resultTileType) {
  auto funcOp = tpopOp->getParentOfType<func::FuncOp>();
  if (!funcOp) {
    return success(); // Already checked elsewhere
  }

  // Look up the consumer-side init
  auto initOr = lookupFrontendInitOpById(tpopOp, funcOp, id);
  if (failed(initOr)) {
    return failure();
  }

  Operation *initOp = *initOr;
  auto aivInit = dyn_cast<AivInitializePipeOp>(initOp);
  if (!aivInit) {
    return success(); // Not consumer init, skip
  }

  auto accPushEpilogue = aivInit.getAccPushEpilogueAttr();
  if (!accPushEpilogue) {
    return success(); // Not a fixpipe, skip
  }

  if (auto tpop = dyn_cast<TPopFromAicOp>(tpopOp); tpop && tpop.getSplit() != 0) {
    return tpop.emitOpError("expects fixpipe TPOP to have split = 0");
  }

  // Rule 11: At least one tpop must exist (checked by counting all tpops for this pipe)
  // Rule 12: Verify result element type matches expected type from quant mode
  auto tileTy = dyn_cast<pto::TileBufType>(resultTileType);
  if (!tileTy) {
    return tpopOp->emitOpError(
        "expects fixpipe TPOP result to be a tile type");
  }

  Type resultElemType = tileTy.getElementType();
  auto quant = accPushEpilogue.getQuant();
  if (failed(
          verifyFixpipePeerPopTypes(tpopOp, funcOp, id, resultTileType))) {
    return failure();
  }

  if (!matchesFixpipeConsumerElementType(quant, resultElemType)) {
    return tpopOp->emitOpError()
           << "expects consumer element type to match acc_push_epilogue.quant "
           << stringifyFixpipeQuant(quant);
  }

  if (!matchesFixpipeConsumerLayout(accPushEpilogue.getLayout(), tileTy)) {
    return tpopOp->emitOpError()
           << "expects consumer tile layout to match acc_push_epilogue.layout "
           << stringifyFixpipeLayout(accPushEpilogue.getLayout());
  }

  return success();
}


static LogicalResult verifyPipeShape(Operation *op, int8_t dirMask, int32_t slotSize,
                                     int32_t slotNum,
                                     std::optional<int32_t> flagBase) {
  constexpr int32_t kMaxHardwareFlagIds = 16;
  if (dirMask != 1 && dirMask != mlir::pto::kValue2 && dirMask != mlir::pto::kValue3) {
      return op->emitOpError("expects 'dir_mask' to be 1, 2, or 3");
  }
  if (slotSize <= 0) {
    return op->emitOpError("expects 'slot_size' to be greater than 0");
  }
  if (slotNum <= 0) {
    return op->emitOpError("expects 'slot_num' to be greater than 0");
  }
  if (flagBase && *flagBase < 0) {
    return op->emitOpError("expects 'flag_base' to be non-negative when present");
  }
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
  if (!isa<pto::PipeType>(pipeHandle.getType())) {
    return op->emitOpError("expects pipe operand type !pto.pipe");
  }
  if (!pipeHandle.getDefiningOp<InitializeL2LPipeOp>() &&
      !pipeHandle.getDefiningOp<InitializeL2G2LPipeOp>()) {
    return op->emitOpError(
        "pipe_handle must be produced by pto.initialize_l2l_pipe or "
        "pto.initialize_l2g2l_pipe");
  }
  return success();
}

// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

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

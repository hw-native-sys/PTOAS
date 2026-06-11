// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Split from PTOEffectsExtra.cpp; kept as a fragment included by PTOEffectsExtra.cpp.
// Intentionally has no local includes.

using namespace mlir;
using namespace mlir::pto;

static constexpr int64_t kPTOFrontendSplitMin = 0;
static constexpr int64_t kPTOFrontendSplitMax = 2;
static constexpr int8_t kPTOFrontendDirMaskC2V = 1;
static constexpr int8_t kPTOFrontendDirMaskV2C = 2;
static constexpr int8_t kPTOFrontendDirMaskBidirectional = 3;
static constexpr uint64_t kPTOFrontendHalfSlotByteMultiplier = 2;
static constexpr int32_t kPTOFrontendBidirectionalLoweredSlotNum = 4;
static constexpr int32_t kPTOFrontendUnidirectionalLoweredSlotNum = 8;

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
  if (split < kPTOFrontendSplitMin || split > kPTOFrontendSplitMax)
    return op->emitOpError("expects 'split' to be 0, 1, or 2");
  return success();
}

static LogicalResult verifyFrontendKernelKind(Operation *op,
                                              FunctionKernelKind expected,
                                              StringRef kernelName) {
  auto kernelKind = getEnclosingFunctionKernelKind(op);
  if (!kernelKind || *kernelKind != expected) {
    return op->emitOpError("must be inside a ")
           << kernelName << " kernel function";
  }
  return success();
}

struct FrontendInitAttrState {
  bool sawId = false;
  bool sawDirMask = false;
  bool sawSlotSize = false;
  bool sawLocalSlotNum = false;
  bool sawNoSplit = false;
};

static ParseResult parseFrontendInitI32AttrClause(OpAsmParser &parser,
                                                  IntegerAttr &attr,
                                                  bool &seen,
                                                  StringRef keyword,
                                                  Type attrType) {
  if (seen) {
    return parser.emitError(parser.getCurrentLocation())
           << "duplicate '" << keyword << "' clause";
  }
  if (parser.parseAttribute(attr, attrType))
    return failure();
  seen = true;
  return success();
}

static ParseResult parseFrontendInitBoolAttrClause(OpAsmParser &parser,
                                                   BoolAttr &attr,
                                                   bool &seen,
                                                   StringRef keyword) {
  if (seen) {
    return parser.emitError(parser.getCurrentLocation())
           << "duplicate '" << keyword << "' clause";
  }
  if (parser.parseAttribute(attr))
    return failure();
  seen = true;
  return success();
}

static ParseResult parseFrontendInitializePipeAttrClause(
    OpAsmParser &parser, StringRef keyword, IntegerAttr &idAttr,
    IntegerAttr &dirMaskAttr, IntegerAttr &slotSizeAttr,
    IntegerAttr &localSlotNumAttr, BoolAttr &nosplitAttr,
    FrontendInitAttrState &state) {
  Builder &builder = parser.getBuilder();
  if (keyword == "id") {
    return parseFrontendInitI32AttrClause(parser, idAttr, state.sawId, keyword,
                                          builder.getI32Type());
  }
  if (keyword == "dir_mask") {
    return parseFrontendInitI32AttrClause(parser, dirMaskAttr,
                                          state.sawDirMask, keyword,
                                          builder.getI8Type());
  }
  if (keyword == "slot_size") {
    return parseFrontendInitI32AttrClause(parser, slotSizeAttr,
                                          state.sawSlotSize, keyword,
                                          builder.getI32Type());
  }
  if (keyword == "local_slot_num") {
    return parseFrontendInitI32AttrClause(parser, localSlotNumAttr,
                                          state.sawLocalSlotNum, keyword,
                                          builder.getI32Type());
  }
  if (keyword == "nosplit") {
    return parseFrontendInitBoolAttrClause(parser, nosplitAttr,
                                           state.sawNoSplit, keyword);
  }
  return parser.emitError(parser.getCurrentLocation())
         << "unexpected keyword '" << keyword << "'";
}

static ParseResult parseFrontendInitializePipeAttrs(
    OpAsmParser &parser, IntegerAttr &idAttr, IntegerAttr &dirMaskAttr,
    IntegerAttr &slotSizeAttr, IntegerAttr &localSlotNumAttr,
    BoolAttr &nosplitAttr) {
  FrontendInitAttrState state;
  if (parser.parseLBrace())
    return failure();
  while (failed(parser.parseOptionalRBrace())) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword) || parser.parseEqual() ||
        failed(parseFrontendInitializePipeAttrClause(
            parser, keyword, idAttr, dirMaskAttr, slotSizeAttr,
            localSlotNumAttr, nosplitAttr, state))) {
      return failure();
    }
    if (succeeded(parser.parseOptionalRBrace()))
      break;
    if (parser.parseComma())
      return failure();
  }
  if (!state.sawDirMask)
    return parser.emitError(parser.getNameLoc(), "expected 'dir_mask' clause");
  if (!state.sawSlotSize)
    return parser.emitError(parser.getNameLoc(), "expected 'slot_size' clause");
  if (!state.sawId)
    idAttr = parser.getBuilder().getI32IntegerAttr(0);
  return success();
}

struct FrontendInitOperandState {
  bool hasGmSlotBuffer = false;
  bool hasGmSlotTensor = false;
  bool hasC2vConsumerBuf = false;
  bool hasV2cConsumerBuf = false;
};

static ParseResult parseFrontendInitializePipeOperandValue(
    OpAsmParser &parser, StringRef keyword,
    std::optional<OpAsmParser::UnresolvedOperand> &target, Type &targetTy,
    bool &seen) {
  if (seen) {
    return parser.emitError(parser.getCurrentLocation())
           << "duplicate '" << keyword << "' operand";
  }
  OpAsmParser::UnresolvedOperand operand;
  if (parser.parseOperand(operand) || parser.parseColonType(targetTy))
    return failure();
  target = operand;
  seen = true;
  return success();
}

static ParseResult parseFrontendInitializePipeOperandClause(
    OpAsmParser &parser, StringRef keyword,
    std::optional<OpAsmParser::UnresolvedOperand> &gmSlotBuffer,
    Type &gmSlotBufferTy,
    std::optional<OpAsmParser::UnresolvedOperand> &gmSlotTensor,
    Type &gmSlotTensorTy,
    std::optional<OpAsmParser::UnresolvedOperand> &c2vConsumerBuf,
    Type &c2vConsumerBufTy,
    std::optional<OpAsmParser::UnresolvedOperand> &v2cConsumerBuf,
    Type &v2cConsumerBufTy, FrontendInitOperandState &state) {
  if (keyword == "gm_slot_buffer")
    return parseFrontendInitializePipeOperandValue(
        parser, keyword, gmSlotBuffer, gmSlotBufferTy,
        state.hasGmSlotBuffer);
  if (keyword == "gm_slot_tensor")
    return parseFrontendInitializePipeOperandValue(
        parser, keyword, gmSlotTensor, gmSlotTensorTy,
        state.hasGmSlotTensor);
  if (keyword == "c2v_consumer_buf")
    return parseFrontendInitializePipeOperandValue(
        parser, keyword, c2vConsumerBuf, c2vConsumerBufTy,
        state.hasC2vConsumerBuf);
  if (keyword == "v2c_consumer_buf")
    return parseFrontendInitializePipeOperandValue(
        parser, keyword, v2cConsumerBuf, v2cConsumerBufTy,
        state.hasV2cConsumerBuf);
  return parser.emitError(parser.getCurrentLocation())
         << "unexpected initialize_pipe operand '" << keyword << "'";
}

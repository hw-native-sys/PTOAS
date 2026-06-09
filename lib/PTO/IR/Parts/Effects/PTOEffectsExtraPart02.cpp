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

static ParseResult parseFrontendInitializePipeOperands(
    OpAsmParser &parser, FrontendInitOperandState &state) {
  if (parser.parseLParen())
    return failure();
  while (failed(parser.parseOptionalRParen())) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword) || parser.parseEqual() ||
        failed(parseFrontendInitializePipeOperandClause(parser, keyword, state))) {
      return failure();
    }
    if (succeeded(parser.parseOptionalRParen()))
      break;
    if (parser.parseComma())
      return failure();
  }
  return success();
}

static ParseResult resolveFrontendInitializePipeOperands(
    OpAsmParser &parser, OperationState &result, NamedAttrList &attrs,
    const FrontendInitOperandState &state) {
  result.addAttributes(attrs);
  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr(
          {state.hasGmSlotBuffer ? 1 : 0, state.hasGmSlotTensor ? 1 : 0,
           state.hasC2vConsumerBuf ? 1 : 0, state.hasV2cConsumerBuf ? 1 : 0}));
  if (state.hasGmSlotBuffer &&
      parser.resolveOperand(state.gmSlotBuffer, state.gmSlotBufferTy,
                            result.operands))
    return failure();
  if (state.hasGmSlotTensor &&
      parser.resolveOperand(state.gmSlotTensor, state.gmSlotTensorTy,
                            result.operands))
    return failure();
  if (state.hasC2vConsumerBuf &&
      parser.resolveOperand(state.c2vConsumerBuf, state.c2vConsumerBufTy,
                            result.operands))
    return failure();
  if (state.hasV2cConsumerBuf &&
      parser.resolveOperand(state.v2cConsumerBuf, state.v2cConsumerBufTy,
                            result.operands))
    return failure();
  return success();
}

static ParseResult parseFrontendInitializePipeOp(OpAsmParser &parser,
                                                 OperationState &result) {
  NamedAttrList attrs;
  FrontendInitAttrState attrState;
  FrontendInitOperandState operandState;
  if (failed(parseFrontendInitializePipeAttrs(parser, attrs, attrState)) ||
      failed(parseFrontendInitializePipeOperands(parser, operandState)) ||
      parser.parseOptionalAttrDict(attrs) ||
      failed(resolveFrontendInitializePipeOperands(parser, result, attrs,
                                                  operandState))) {
    return failure();
  }
  return success();
}

template <typename InitOpT>
static void printFrontendInitializePipeOp(InitOpT op, OpAsmPrinter &p) {
  p << " {";
  bool needsComma = false;
  auto printClause = [&needsComma, &p](StringRef keyword, auto value) {
    if (needsComma)
      p << ", ";
    p << keyword << " = " << value;
    needsComma = true;
  };

  if (op.getId() != 0)
    printClause("id", op.getId());
  printClause("dir_mask", static_cast<int32_t>(op.getDirMask()));
  printClause("slot_size", op.getSlotSize());
  if (auto localSlotNumAttr = op.getLocalSlotNumAttr())
    printClause("local_slot_num", localSlotNumAttr.getInt());
  if (auto noSplitAttr = op.getNosplitAttr())
    printClause("nosplit", noSplitAttr.getValue() ? "true" : "false");
  p << "}";

  p << "(";
  bool needsOperandComma = false;
  auto printOperandClause = [&needsOperandComma, &p](StringRef keyword,
                                                     Value value) {
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
      /*elidedAttrs=*/{"id", "dir_mask", "slot_size", "local_slot_num",
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
  return tensorBytes == slotBytes ||
         tensorBytes * kPTOFrontendHalfSlotByteMultiplier == slotBytes;
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
static unsigned countFrontendInitOpsWithSameId(func::FuncOp funcOp,
                                               uint32_t id) {
  unsigned sameIdInitCount = 0;
  funcOp.walk([id, &sameIdInitCount](Operation *candidate) {
    if (auto aic = dyn_cast<AicInitializePipeOp>(candidate)) {
      if (aic.getId() == id)
        ++sameIdInitCount;
      return;
    }
    if (auto aiv = dyn_cast<AivInitializePipeOp>(candidate)) {
      if (aiv.getId() == id)
        ++sameIdInitCount;
    }
  });
  return sameIdInitCount;
}

template <typename InitOpT>
static LogicalResult verifyFrontendInitGlobalTensorForm(
    InitOpT op, int8_t dirMask, bool hasC2vConsumerBuf, bool hasV2cConsumerBuf) {
  if (op.getGmSlotBuffer() || hasC2vConsumerBuf || hasV2cConsumerBuf) {
    return op.emitOpError(
        "globaltensor pipe init expects only 'gm_slot_tensor' and no "
        "'gm_slot_buffer', 'c2v_consumer_buf', or 'v2c_consumer_buf'");
  }
  if (op.getLocalSlotNumAttr())
    return op.emitOpError("globaltensor pipe init does not use 'local_slot_num'");
  if (getTargetArch(op.getOperation()) == PTOArch::A5) {
    return op.emitOpError(
        "globaltensor pipe entries are supported for a2/a3 l2g2l pipes");
  }
  return verifyFrontendGlobalSlotTensor(op.getOperation(), op.getGmSlotTensor(),
                                        dirMask, op.getSlotSize());
}

template <typename InitOpT>
static LogicalResult verifyFrontendInitLocalPipeForm(
    InitOpT op, int8_t dirMask, bool hasC2vConsumerBuf, bool hasV2cConsumerBuf) {
  if (hasC2vConsumerBuf != hasV2cConsumerBuf) {
    return op.emitOpError(
        "expects 'c2v_consumer_buf' and 'v2c_consumer_buf' to be provided together");
  }
  if (!hasC2vConsumerBuf) {
    return op.emitOpError(
        "expects local pipe init to provide 'c2v_consumer_buf' and "
        "'v2c_consumer_buf'; use 'gm_slot_tensor' for globaltensor pipe entries");
  }
  if (auto localSlotNumAttr = op.getLocalSlotNumAttr()) {
    int32_t localSlotNum = localSlotNumAttr.getInt();
    if (localSlotNum <= 0)
      return op.emitOpError("expects 'local_slot_num' to be greater than 0");
    int32_t loweredSlotNum = dirMask == kPTOFrontendDirMaskBidirectional
                                 ? kPTOFrontendBidirectionalLoweredSlotNum
                                 : kPTOFrontendUnidirectionalLoweredSlotNum;
    if (localSlotNum > loweredSlotNum) {
      return op.emitOpError()
             << "expects 'local_slot_num' to be less than or equal to "
             << loweredSlotNum << " for dir_mask = " << static_cast<int>(dirMask);
    }
  }
  return success();
}

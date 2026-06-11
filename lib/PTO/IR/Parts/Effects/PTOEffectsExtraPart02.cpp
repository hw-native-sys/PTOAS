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
    OpAsmParser &parser,
    std::optional<OpAsmParser::UnresolvedOperand> &gmSlotBuffer,
    Type &gmSlotBufferTy,
    std::optional<OpAsmParser::UnresolvedOperand> &gmSlotTensor,
    Type &gmSlotTensorTy,
    std::optional<OpAsmParser::UnresolvedOperand> &c2vConsumerBuf,
    Type &c2vConsumerBufTy,
    std::optional<OpAsmParser::UnresolvedOperand> &v2cConsumerBuf,
    Type &v2cConsumerBufTy) {
  FrontendInitOperandState state;
  if (parser.parseLParen())
    return failure();
  while (failed(parser.parseOptionalRParen())) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword) || parser.parseEqual() ||
        failed(parseFrontendInitializePipeOperandClause(
            parser, keyword, gmSlotBuffer, gmSlotBufferTy, gmSlotTensor,
            gmSlotTensorTy, c2vConsumerBuf, c2vConsumerBufTy, v2cConsumerBuf,
            v2cConsumerBufTy, state))) {
      return failure();
    }
    if (succeeded(parser.parseOptionalRParen()))
      break;
    if (parser.parseComma())
      return failure();
  }
  return success();
}

static ParseResult parseFrontendInitializePipe(
    OpAsmParser &parser, IntegerAttr &idAttr, IntegerAttr &dirMaskAttr,
    IntegerAttr &slotSizeAttr, IntegerAttr &localSlotNumAttr,
    BoolAttr &nosplitAttr,
    std::optional<OpAsmParser::UnresolvedOperand> &gmSlotBuffer,
    Type &gmSlotBufferTy,
    std::optional<OpAsmParser::UnresolvedOperand> &gmSlotTensor,
    Type &gmSlotTensorTy,
    std::optional<OpAsmParser::UnresolvedOperand> &c2vConsumerBuf,
    Type &c2vConsumerBufTy,
    std::optional<OpAsmParser::UnresolvedOperand> &v2cConsumerBuf,
    Type &v2cConsumerBufTy) {
  if (failed(parseFrontendInitializePipeAttrs(
          parser, idAttr, dirMaskAttr, slotSizeAttr, localSlotNumAttr,
          nosplitAttr)) ||
      failed(parseFrontendInitializePipeOperands(
          parser, gmSlotBuffer, gmSlotBufferTy, gmSlotTensor, gmSlotTensorTy,
          c2vConsumerBuf, c2vConsumerBufTy, v2cConsumerBuf,
          v2cConsumerBufTy))) {
    return failure();
  }
  return success();
}

static void printFrontendInitializePipeAttrClause(OpAsmPrinter &printer,
                                                  bool &needsComma,
                                                  StringRef keyword,
                                                  Attribute attr) {
  if (needsComma)
    printer << ", ";
  printer << keyword << " = " << attr;
  needsComma = true;
}

template <typename InitOpT>
static void printFrontendInitializePipe(
    OpAsmPrinter &printer, InitOpT op, IntegerAttr idAttr,
    IntegerAttr dirMaskAttr, IntegerAttr slotSizeAttr,
    IntegerAttr localSlotNumAttr, BoolAttr nosplitAttr, Value gmSlotBuffer,
    Type gmSlotBufferTy, Value gmSlotTensor, Type gmSlotTensorTy,
    Value c2vConsumerBuf, Type c2vConsumerBufTy, Value v2cConsumerBuf,
    Type v2cConsumerBufTy) {
  (void)op;
  printer << " {";
  bool needsComma = false;
  if (idAttr && idAttr.getValue().getSExtValue() != 0) {
    printFrontendInitializePipeAttrClause(printer, needsComma, "id", idAttr);
  }
  printFrontendInitializePipeAttrClause(printer, needsComma, "dir_mask",
                                        dirMaskAttr);
  printFrontendInitializePipeAttrClause(printer, needsComma, "slot_size",
                                        slotSizeAttr);
  if (localSlotNumAttr) {
    printFrontendInitializePipeAttrClause(printer, needsComma,
                                          "local_slot_num", localSlotNumAttr);
  }
  if (nosplitAttr) {
    printFrontendInitializePipeAttrClause(printer, needsComma, "nosplit",
                                          nosplitAttr);
  }
  printer << "}";

  printer << "(";
  bool needsOperandComma = false;
  auto printOperandClause = [&needsOperandComma,
                             &printer](StringRef keyword, Value value,
                                       Type type) {
    if (needsOperandComma)
      printer << ", ";
    printer << keyword << " = " << value << " : " << type;
    needsOperandComma = true;
  };
  if (gmSlotBuffer)
    printOperandClause("gm_slot_buffer", gmSlotBuffer, gmSlotBufferTy);
  if (gmSlotTensor)
    printOperandClause("gm_slot_tensor", gmSlotTensor, gmSlotTensorTy);
  if (c2vConsumerBuf)
    printOperandClause("c2v_consumer_buf", c2vConsumerBuf, c2vConsumerBufTy);
  if (v2cConsumerBuf)
    printOperandClause("v2c_consumer_buf", v2cConsumerBuf, v2cConsumerBufTy);
  printer << ")";
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

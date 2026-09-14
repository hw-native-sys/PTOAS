// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

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

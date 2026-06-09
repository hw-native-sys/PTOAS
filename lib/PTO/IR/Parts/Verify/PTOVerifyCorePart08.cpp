// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Split from PTOVerifyCore.cpp; kept as a fragment included by PTOVerifyCore.cpp.
// Intentionally has no local includes.

using namespace mlir;
using namespace mlir::pto;

LogicalResult TPrefetchAsyncOp::verify() {
  if (failed(verifyAsyncFlatContiguous1DGMViewLike(getOperation(), getSrc(),
                                                   "src")))
    return failure();
  return success();
}

LogicalResult mlir::pto::SetFFTsOp::verify() {
  auto mr = llvm::dyn_cast<mlir::MemRefType>(getFfts().getType());
  if (!mr)
    return emitOpError("expects a memref operand");
  if (!mr.getElementType().isInteger(kPTOI64BitWidth) && !mr.getElementType().isInteger(kPTOI8BitWidth))
    return emitOpError("expects element type i64 (or i8)");
  return mlir::success();
}

ParseResult mlir::pto::SyncSetOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  return parseSyncEventOpCommon(parser, result,
                                SyncSetOp::getPipeAttrName(result.name),
                                SyncSetOp::getEventIdAttrName(result.name));
}

void mlir::pto::SyncSetOp::print(OpAsmPrinter &p) {
  printSyncEventOpCommon(p, getOperation(), getPipe(), getEventIdAttr(),
                         getEventIdDyn(), getPipeAttrName().getValue(),
                         getEventIdAttrName().getValue());
}

LogicalResult mlir::pto::SyncSetOp::verify() {
  bool hasStatic = getEventIdAttr() != nullptr;
  bool hasDynamic = static_cast<bool>(getEventIdDyn());
  if (hasStatic == hasDynamic)
    return emitOpError()
           << "expects exactly one event-id form: static attr or dynamic index operand";
  if (IntegerAttr fftsModeAttr = getFftsModeAttr()) {
    static constexpr int64_t kPTOFftsModeMin = 0;
    static constexpr int64_t kPTOFftsModeMax = 2;
    int64_t fftsMode = fftsModeAttr.getInt();
    if (fftsMode < kPTOFftsModeMin || fftsMode > kPTOFftsModeMax)
      return emitOpError() << "requires ffts_mode in range [0, 2], but got "
                           << fftsMode;
  }

  auto verifyA2A3 = []() -> LogicalResult { return success(); };
  auto verifyA5 = [this]() -> LogicalResult {
    switch (getPipe().getPipe()) {
    case PIPE::PIPE_FIX:
    case PIPE::PIPE_MTE3:
      return success();
    default:
      return emitOpError()
             << "A5 sync.set expects pipe to be one of <PIPE_FIX>, <PIPE_MTE3>";
    }
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

ParseResult mlir::pto::SyncWaitOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  return parseSyncEventOpCommon(parser, result,
                                SyncWaitOp::getPipeAttrName(result.name),
                                SyncWaitOp::getEventIdAttrName(result.name));
}

void mlir::pto::SyncWaitOp::print(OpAsmPrinter &p) {
  printSyncEventOpCommon(p, getOperation(), getPipe(), getEventIdAttr(),
                         getEventIdDyn(), getPipeAttrName().getValue(),
                         getEventIdAttrName().getValue());
}

static ParseResult parseSyncAllOptionalOperands(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
    SmallVectorImpl<Type> &operandTypes) {
  if (parser.parseLParen())
    return failure();
  if (failed(parser.parseOptionalRParen())) {
    if (parser.parseOperandList(operands) ||
        parser.parseColonTypeList(operandTypes) || parser.parseRParen())
      return failure();
    if (operands.size() != operandTypes.size()) {
      return parser.emitError(parser.getCurrentLocation())
             << "expects the same number of operands and operand types";
    }
  }
  return success();
}

static ParseResult parseSyncAllModeAndCoreType(OpAsmParser &parser,
                                               OperationState &result,
                                               SyncAllModeAttr &mode,
                                               SyncCoreTypeAttr &coreType) {
  Attribute modeAttr;
  Attribute coreTypeAttr;
  if (parser.parseKeyword("mode") || parser.parseEqual() ||
      parser.parseAttribute(modeAttr) || parser.parseComma() ||
      parser.parseKeyword("core_type") || parser.parseEqual() ||
      parser.parseAttribute(coreTypeAttr))
    return failure();
  mode = dyn_cast<pto::SyncAllModeAttr>(modeAttr);
  if (!mode)
    return parser.emitError(parser.getCurrentLocation())
           << "expects mode to be #pto.sync_all_mode<...>";
  coreType = dyn_cast<pto::SyncCoreTypeAttr>(coreTypeAttr);
  if (!coreType)
    return parser.emitError(parser.getCurrentLocation())
           << "expects core_type to be #pto.sync_core_type<...>";
  result.addAttribute("mode", mode);
  result.addAttribute("core_type", coreType);
  return parser.parseOptionalAttrDict(result.attributes);
}

static void addSyncAllSegmentSizes(OpAsmParser &parser, OperationState &result,
                                   int32_t gm, int32_t ub, int32_t l1,
                                   int32_t used) {
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {gm, ub, l1, used}));
}

static ParseResult resolveSyncAllSoftOperands(
    OpAsmParser &parser, OperationState &result,
    ArrayRef<OpAsmParser::UnresolvedOperand> operands,
    ArrayRef<Type> operandTypes, int32_t gm, int32_t ub, int32_t l1) {
  int32_t required = gm + ub + l1;
  bool hasUsedCores = operands.size() == static_cast<size_t>(required + 1);
  if (operands.size() != static_cast<size_t>(required) && !hasUsedCores)
    return failure();
  for (int32_t i = 0; i < required; ++i) {
    if (parser.resolveOperand(operands[i], operandTypes[i], result.operands))
      return failure();
  }
  if (hasUsedCores &&
      parser.resolveOperand(operands[required], operandTypes[required],
                            result.operands))
    return failure();
  addSyncAllSegmentSizes(parser, result, gm, ub, l1, hasUsedCores ? 1 : 0);
  return success();
}

ParseResult mlir::pto::SyncAllOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  SmallVec4<OpAsmParser::UnresolvedOperand> operands;
  SmallVec4<Type> operandTypes;
  SyncAllModeAttr mode;
  SyncCoreTypeAttr coreType;
  if (failed(parseSyncAllOptionalOperands(parser, operands, operandTypes)) ||
      failed(parseSyncAllModeAndCoreType(parser, result, mode, coreType)))
    return failure();

  switch (mode.getValue()) {
  case pto::SyncAllMode::Hard:
    if (!operands.empty())
      return parser.emitError(parser.getCurrentLocation())
             << "expects hard syncall to have no operands";
    addSyncAllSegmentSizes(parser, result, 0, 0, 0, 0);
    return success();
  case pto::SyncAllMode::Soft:
    break;
  }

  switch (coreType.getValue()) {
  case pto::SyncCoreType::AIVOnly:
    if (operands.size() != kNumber2 && operands.size() != kNumber3)
      return parser.emitError(parser.getCurrentLocation())
             << "expects soft AIV-only syncall to have gm_workspace, "
                "ub_workspace, and optional used_cores";
    return resolveSyncAllSoftOperands(parser, result, operands, operandTypes, 1,
                                      1, 0);
  case pto::SyncCoreType::AICOnly:
    if (operands.size() != kNumber2 && operands.size() != kNumber3)
      return parser.emitError(parser.getCurrentLocation())
             << "expects soft AIC-only syncall to have gm_workspace, "
                "l1_workspace, and optional used_cores";
    return resolveSyncAllSoftOperands(parser, result, operands, operandTypes, 1,
                                      0, 1);
  case pto::SyncCoreType::Mix:
    if (operands.size() != kNumber3 && operands.size() != kNumber4)
      return parser.emitError(parser.getCurrentLocation())
             << "expects soft mixed syncall to have gm_workspace, "
                "ub_workspace, l1_workspace, and optional used_cores";
    return resolveSyncAllSoftOperands(parser, result, operands, operandTypes, 1,
                                      1, 1);
  }

  llvm_unreachable("unhandled SyncCoreType");
}

void mlir::pto::SyncAllOp::print(OpAsmPrinter &p) {
  SmallVec4<Value> operands;
  if (getGmWorkspace())
    operands.push_back(getGmWorkspace());
  if (getUbWorkspace())
    operands.push_back(getUbWorkspace());
  if (getL1Workspace())
    operands.push_back(getL1Workspace());
  if (getUsedCores())
    operands.push_back(getUsedCores());

  p << "(";
  if (!operands.empty()) {
    p.printOperands(operands);
    p << " : ";
    llvm::interleaveComma(operands, p, [&p](Value operand) {
      p.printType(operand.getType());
    });
  }
  p << ") mode = " << getMode() << ", core_type = " << getCoreType();
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes", "mode",
                                           "core_type"});
}

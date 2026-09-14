// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

static LogicalResult verifyInternalTileOp(
    Operation *op, Value pipeHandle, uint32_t split, bool producerSide,
    Value tile, Value aivSubblockId, pto::PIPE pipe, StringRef pipeError) {
  if (failed(verifyInternalPipeBase(op, pipeHandle, split, producerSide)) ||
      failed(verifyOddSplitTileEntry(op, split, tile.getType())))
    return failure();
  if (isInsideCubeKernelOrSection(op) &&
      failed(verifyFullTileSplitParity(op, split, tile.getType())))
    return failure();
  if (failed(verifyAivSubblockIdOperand(op, aivSubblockId, split,
                                        tile.getType())) ||
      failed(verifyTensorEntryMatchesInternalPipeInit(
          op, pipeHandle, tile.getType())))
    return failure();
  if (!isa<TensorViewType>(tile.getType()) &&
      pipe == pto::PIPE::PIPE_UNASSIGNED)
    return op->emitOpError(pipeError);
  return success();
}

LogicalResult TPushOp::verify() {
  return verifyInternalTileOp(
      getOperation(), getPipeHandle(), getSplit(), /*producerSide=*/true,
      getTile(), getAivSubblockid(), getPipe(),
      "tile type must map to a supported producer pipe");
}

LogicalResult TAllocOp::verify() {
  if (failed(verifyInternalPipeBase(getOperation(), getPipeHandle(), getSplit(),
                                    /*producerSide=*/true)) ||
      failed(verifyOddSplitTileEntry(getOperation(), getSplit(),
                                     getEntry().getType())) ||
      failed(verifyTensorEntryMatchesInternalPipeInit(
          getOperation(), getPipeHandle(), getEntry().getType())))
    return failure();
  return success();
}

LogicalResult TPopOp::verify() {
  return verifyInternalTileOp(
      getOperation(), getPipeHandle(), getSplit(), /*producerSide=*/false,
      getTile(), getAivSubblockid(), getPipe(),
      "tile type and target arch must map to a supported consumer pipe");
}

LogicalResult TFreeOp::verify() {
  if (failed(verifyInternalPipeBase(getOperation(), getPipeHandle(), getSplit(),
                                    /*producerSide=*/false)))
    return failure();
  if (getEntry() &&
      failed(verifyOddSplitTileEntry(getOperation(), getSplit(),
                                     getEntry().getType()))) {
    return failure();
  }
  if (getEntry() &&
      failed(verifyTensorEntryMatchesInternalPipeInit(
          getOperation(), getPipeHandle(), getEntry().getType()))) {
    return failure();
  }
  return success();
}

struct TFreeParseState {
  OpAsmParser::UnresolvedOperand first;
  OpAsmParser::UnresolvedOperand pipe;
  Type firstTy;
  Type pipeTy;
  bool hasEntry = false;
};

static ParseResult parseTFreeOperands(OpAsmParser &parser,
                                      TFreeParseState &state) {
  if (parser.parseLParen() || parser.parseOperand(state.first))
    return failure();
  state.hasEntry = succeeded(parser.parseOptionalComma());
  if (!state.hasEntry) {
    if (parser.parseColonType(state.pipeTy) || parser.parseRParen())
      return failure();
    state.pipe = state.first;
    return success();
  }
  if (parser.parseOperand(state.pipe) || parser.parseColonType(state.firstTy) ||
      parser.parseComma() || parser.parseType(state.pipeTy) ||
      parser.parseRParen())
    return failure();
  return success();
}

static ParseResult parseTFreeAttributes(OpAsmParser &parser,
                                        NamedAttrList &attrs) {
  if (parser.parseLBrace() || parser.parseKeyword("split") ||
      parser.parseEqual())
    return failure();
  IntegerAttr splitAttr;
  if (parser.parseAttribute(splitAttr, parser.getBuilder().getI8Type(),
                            "split", attrs) ||
      parser.parseRBrace() || parser.parseOptionalAttrDict(attrs))
    return failure();
  return success();
}

static ParseResult resolveTFreeOperands(OpAsmParser &parser,
                                        OperationState &result,
                                        const TFreeParseState &state) {
  if (state.hasEntry &&
      parser.resolveOperand(state.first, state.firstTy, result.operands))
    return failure();
  return parser.resolveOperand(state.pipe, state.pipeTy, result.operands);
}

ParseResult TFreeOp::parse(OpAsmParser &parser, OperationState &result) {
  TFreeParseState state;
  NamedAttrList attrs;
  if (failed(parseTFreeOperands(parser, state)) ||
      failed(parseTFreeAttributes(parser, attrs))) {
    return failure();
  }
  result.addAttributes(attrs);
  return resolveTFreeOperands(parser, result, state);
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
  if (!block) {
    return nullptr;
  }
  for (Operation &op : *block) {
    if (!op.hasTrait<OpTrait::ConstantLike>()) {
      return &op;
    }
  }
  return nullptr;
}

static bool isOpInRange(Operation *op, Operation *first, Operation *last) {
  for (Operation *cur = first; cur; cur = cur->getNextNode()) {
    if (cur == op) {
      return true;
    }
    if (cur == last) {
      return false;
    }
  }
  return false;
}

static std::optional<unsigned> getSimtKeepResumeRegisterCount(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
      if (intType.getWidth() <= mlir::pto::kValue32) {
          return 1;
      }
      if (intType.getWidth() == mlir::pto::kValue64) {
          return mlir::pto::kValue2;
      }
    return std::nullopt;
  }
  if (type.isF16() || type.isBF16() || type.isF32()) {
    return 1;
  }
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
  if (!registerCount) {
    return success();
  }
  int64_t slot = op.getSlot();
  if (slot < 0 || slot >= kSimtKeepResumeSlotLimit) {
    return op.emitOpError()
           << "requires slot in range [0, "
           << (kSimtKeepResumeSlotLimit - 1) << "]";
  }
  if (*registerCount == mlir::pto::kValue2) {
      if ((slot % mlir::pto::kValue2) != 0) {
          return op.emitOpError() << "requires an even slot for 64-bit keep/resume values";
      }
      if (slot + 1 >= kSimtKeepResumeSlotLimit) {
          return op.emitOpError() << "requires slot in range [0, " << (kSimtKeepResumeSlotLimit - mlir::pto::kValue2)
                                  << "] for 64-bit keep/resume values";
      }
  }
  return success();
}

template <typename OpT>
static bool overlapsEarlierSimtKeepResumeSlotUse(OpT op,
                                                 SmallVectorImpl<int64_t> &used) {
  std::optional<unsigned> registerCount =
      getSimtKeepResumeRegisterCount(getSimtKeepResumeValueType(op));
  if (!registerCount) {
    return false;
  }
  int64_t slot = op.getSlot();
  for (int64_t word = slot; word < slot + *registerCount; ++word) {
    if (llvm::is_contained(used, word)) {
      return true;
    }
  }
  for (int64_t word = slot; word < slot + *registerCount; ++word) {
    used.push_back(word);
  }
  return false;
}

static LogicalResult verifyUniqueResumeGroupSlots(ResumeOp current,
                                                  Operation *first) {
    SmallVector<int64_t, mlir::pto::kValue4> slots;
    for (Operation* cur = first; cur; cur = cur->getNextNode()) {
        auto resume = dyn_cast<ResumeOp>(cur);
        if (!resume) {
            break;
        }
        if (overlapsEarlierSimtKeepResumeSlotUse(resume, slots) && resume.getOperation() == current.getOperation()) {
            return current.emitOpError() << "duplicates an earlier slot " << resume.getSlot()
                                         << " in the SIMT resume prologue group";
        }
    }
  return success();
}

static LogicalResult verifyUniqueKeepGroupSlots(KeepOp current,
                                                Operation *first,
                                                Operation *last) {
    SmallVector<int64_t, mlir::pto::kValue4> slots;
    for (Operation* cur = first; cur; cur = cur->getNextNode()) {
        auto keep = dyn_cast<KeepOp>(cur);
        if (!keep) {
            break;
        }
        if (overlapsEarlierSimtKeepResumeSlotUse(keep, slots) && keep.getOperation() == current.getOperation()) {
            return current.emitOpError() << "duplicates an earlier slot " << keep.getSlot()
                                         << " in the SIMT keep epilogue group";
        }
        if (cur == last) {
            break;
        }
    }
  return success();
}

static bool isSupportedSimtKeepResumeType(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
      return intType.getWidth() <= mlir::pto::kValue64;
  }
  return type.isF16() || type.isBF16() || type.isF32();
}

// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This implementation fragment is included by PTO.cpp and intentionally is
// not listed as a separate CMake translation unit.

static Operation *getLastSimtPayloadOp(KeepOp op) {
  Block *block = op->getBlock();
  bool insideSection =
      op->getParentOfType<SectionSimtOp>() != nullptr;
  if (insideSection)
    return block->empty() ? nullptr : &block->back();
  if (Operation *terminator = block->getTerminator()) {
    if (isa<func::ReturnOp>(terminator))
      return terminator->getPrevNode();
  }
  return nullptr;
}

static LogicalResult verifyKeepEpiloguePosition(KeepOp op) {
  Operation *lastPayloadOp = getLastSimtPayloadOp(op);
  if (!lastPayloadOp) {
    return op.emitOpError(
        "must be placed in the SIMT epilogue before func.return or the end "
        "of pto.section.simt");
  }

  Operation *cur = lastPayloadOp;
  while (cur && isa<SyncthreadsOp>(cur)) {
    cur = cur->getPrevNode();
  }
  Operation *lastKeep = cur;
  if (!lastKeep || !isa<KeepOp>(lastKeep)) {
    return op.emitOpError()
           << "must be placed in the SIMT epilogue before func.return; only "
              "'pto.syncthreads' may appear between the final 'pto.keep' group "
              "and func.return or the end of pto.section.simt";
  }

  Operation *firstKeep = lastKeep;
  while (Operation *prev = firstKeep->getPrevNode()) {
    if (!isa<KeepOp>(prev)) {
      break;
    }
    firstKeep = prev;
  }
  if (!isOpInRange(op, firstKeep, lastKeep)) {
    return op.emitOpError()
           << "must be in the contiguous SIMT keep epilogue group immediately "
              "before optional 'pto.syncthreads' and func.return or the end "
              "of pto.section.simt";
  }
  return verifyUniqueKeepGroupSlots(op, firstKeep, lastKeep);
}

LogicalResult KeepOp::verify() {
  if (failed(verifySimtKeepResumeCommon(getOperation(), getSlot())))
    return failure();
  if (!isSupportedSimtKeepResumeType(getPayload().getType()))
    return emitOpError()
           << "supports integer scalar payloads up to 64 bits and "
              "f16/bf16/f32 payloads";
  if (failed(verifySimtKeepResumeSlotRange(*this)))
    return failure();
  return verifyKeepEpiloguePosition(*this);
}

LogicalResult ResumeOp::verify() {
  if (failed(verifySimtKeepResumeCommon(getOperation(), getSlot()))) {
    return failure();
  }
  if (!isSupportedSimtKeepResumeType(getResult().getType())) {
    return emitOpError()
           << "supports integer scalar results up to 64 bits and "
              "f16/bf16/f32 results";
  }
  if (failed(verifySimtKeepResumeSlotRange(*this))) {
    return failure();
  }
  Block *block = getOperation()->getBlock();
  Operation *first = getFirstNonConstantLikeOp(block);
  if (!first || !isa<ResumeOp>(first)) {
    return emitOpError()
           << "must be in the contiguous SIMT resume prologue group after "
              "constant-like operations";
  }

  bool found = false;
  for (Operation *cur = first; cur; cur = cur->getNextNode()) {
    if (!isa<ResumeOp>(cur)) {
      break;
    }
    if (cur == getOperation()) {
      found = true;
      break;
    }
  }
  if (!found) {
    return emitOpError()
           << "must be in the contiguous SIMT resume prologue group after "
              "constant-like operations";
  }
  if (failed(verifyUniqueResumeGroupSlots(*this, first))) {
    return failure();
  }
  return success();
}

using MemoryEffectList =
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>;

static void addReadWriteEffects(MemoryEffectList &effects,
                                OpOperand &operand) {
  addEffect(effects, &operand, MemoryEffects::Read::get());
  addEffect(effects, &operand, MemoryEffects::Write::get());
}

static void addOptionalReadWriteEffects(MemoryEffectList &effects,
                                        mlir::MutableOperandRange operands) {
  if (auto it = operands.begin(); it != operands.end())
    addReadWriteEffects(effects, *it);
}

static void addPingPongEffects(MemoryEffectList &effects, OpOperand &ping,
                               mlir::MutableOperandRange pong) {
  addReadWriteEffects(effects, ping);
  addOptionalReadWriteEffects(effects, pong);
}

static void addGroupEffects(MemoryEffectList &effects,
                            mlir::MutableOperandRange group,
                            MemoryEffects::Effect *effect) {
  for (OpOperand &operand : group)
    addEffect(effects, &operand, effect);
}

static void addAsyncTransferEffects(MemoryEffectList &effects,
                                    OpOperand &dst, OpOperand &src,
                                    OpOperand &session, OpResult event) {
  addEffect(effects, &dst, MemoryEffects::Write::get());
  addEffect(effects, &src, MemoryEffects::Read::get());
  addEffect(effects, &session, MemoryEffects::Read::get());
  addEffect(effects, event, MemoryEffects::Write::get());
}

static void addSyncTransferEffects(MemoryEffectList &effects,
                                   OpOperand &dst, OpOperand &src,
                                   OpOperand &ping,
                                   mlir::MutableOperandRange pong) {
  addEffect(effects, &dst, MemoryEffects::Write::get());
  addEffect(effects, &src, MemoryEffects::Read::get());
  addPingPongEffects(effects, ping, pong);
}

void BuildAsyncSessionOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getScratchMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getWorkspaceMutable(), MemoryEffects::Read::get());
  addEffect(effects, getOperation()->getOpResult(0), MemoryEffects::Write::get());
}

void TPutAsyncOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addAsyncTransferEffects(effects, getDstMutable(), getSrcMutable(),
                          getSessionMutable(), getOperation()->getOpResult(0));
}

void TGetAsyncOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addAsyncTransferEffects(effects, getDstMutable(), getSrcMutable(),
                          getSessionMutable(), getOperation()->getOpResult(0));
}

void TPutOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addSyncTransferEffects(effects, getDstMutable(), getSrcMutable(),
                         getPingMutable(), getPongMutable());
}

void TGetOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addSyncTransferEffects(effects, getDstMutable(), getSrcMutable(),
                         getPingMutable(), getPongMutable());
}

void TNotifyOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getSignalMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getValueMutable(), MemoryEffects::Read::get());
}

void TWaitOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getSignalMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getCmpValueMutable(), MemoryEffects::Read::get());
}

void TTestOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getSignalMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getCmpValueMutable(), MemoryEffects::Read::get());
  addEffect(effects, getOperation()->getOpResult(0), MemoryEffects::Write::get());
}

void TBroadcastOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addPingPongEffects(effects, getPingMutable(), getPongMutable());
  addGroupEffects(effects, getGroupMutable(), MemoryEffects::Write::get());
}

void CommTGatherOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
  addPingPongEffects(effects, getPingMutable(), getPongMutable());
  addGroupEffects(effects, getGroupMutable(), MemoryEffects::Read::get());
}

void CommTScatterOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addPingPongEffects(effects, getPingMutable(), getPongMutable());
  addGroupEffects(effects, getGroupMutable(), MemoryEffects::Write::get());
}

void TReduceOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
  addReadWriteEffects(effects, getAccMutable());
  addPingPongEffects(effects, getRecvPingMutable(), getRecvPongMutable());
  addGroupEffects(effects, getGroupMutable(), MemoryEffects::Read::get());
}

void WaitAsyncEventOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getEventMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getSessionMutable(), MemoryEffects::Read::get());
  addEffect(effects, getOperation()->getOpResult(0), MemoryEffects::Write::get());
}

void TestAsyncEventOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getEventMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getSessionMutable(), MemoryEffects::Read::get());
  addEffect(effects, getOperation()->getOpResult(0), MemoryEffects::Write::get());
}

void InitializeL2G2LPipeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getGmAddrMutable(), MemoryEffects::Read::get());
  auto localAddr = getLocalAddrMutable();
  if (!localAddr.empty()) {
    addEffect(effects, &*localAddr.begin(), MemoryEffects::Read::get());
  }
  auto peerLocalAddr = getPeerLocalAddrMutable();
  if (!peerLocalAddr.empty()) {
    addEffect(effects, &*peerLocalAddr.begin(), MemoryEffects::Read::get());
  }
  addEffect(effects, getOperation()->getOpResult(0), MemoryEffects::Write::get());
}

void InitializeL2LPipeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getLocalAddrMutable(), MemoryEffects::Read::get());
  addEffect(effects, getOperation()->getOpResult(0), MemoryEffects::Write::get());
}

void TPushOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getTileMutable(), MemoryEffects::Read::get());
  auto aivSubblockId = getAivSubblockidMutable();
  if (!aivSubblockId.empty()) {
    addEffect(effects, &*aivSubblockId.begin(), MemoryEffects::Read::get());
  }
  addEffect(effects, &getPipeHandleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPipeHandleMutable(), MemoryEffects::Write::get());
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());

  if (auto pipeId = getFrontendPipeIdFromHandle(getPipeHandle())) {
    auto accPushEpilogue =
        getAccPushEpilogueFromInitOp(getPipeHandle().getDefiningOp());
    if (accPushEpilogue &&
        (isScalarFixpipeQuant(accPushEpilogue.getQuant()) ||
         isVectorFixpipeQuant(accPushEpilogue.getQuant()))) {
      effects.emplace_back(MemoryEffects::Read::get(),
                           getFixpipeQuantStateIdAttr(getOperation(), *pipeId),
                           FixpipeQuantStateResource::get());
    }
  }
}

void TPushToAivOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getTileMutable(), MemoryEffects::Read::get());

  auto funcOp = getOperation()->getParentOfType<func::FuncOp>();
  if (!funcOp) {
    return;
  }

  auto initOr = lookupFrontendInitOpById(getOperation(), funcOp, getId());
  if (failed(initOr)) {
    return;
  }

  auto aicInit = dyn_cast<AicInitializePipeOp>(*initOr);
  if (!aicInit) {
    return;
  }

  auto accPushEpilogue = aicInit.getAccPushEpilogueAttr();
  if (!accPushEpilogue) {
    return;
  }

  auto quant = accPushEpilogue.getQuant();
  if (isScalarFixpipeQuant(quant) || isVectorFixpipeQuant(quant)) {
    effects.emplace_back(MemoryEffects::Read::get(),
                         getFixpipeQuantStateIdAttr(getOperation(), getId()),
                         FixpipeQuantStateResource::get());
  }
}

void TAllocOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getEntryMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getPipeHandleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPipeHandleMutable(), MemoryEffects::Write::get());
}

void TPopOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getPipeHandleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPipeHandleMutable(), MemoryEffects::Write::get());
  auto aivSubblockId = getAivSubblockidMutable();
  if (!aivSubblockId.empty()) {
    addEffect(effects, &*aivSubblockId.begin(), MemoryEffects::Read::get());
  }
  addEffect(effects, &getTileMutable(), MemoryEffects::Write::get());
}

void TFreeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  auto entry = getEntryMutable();
  if (!entry.empty()) {
    addEffect(effects, &*entry.begin(), MemoryEffects::Read::get());
  }
  addEffect(effects, &getPipeHandleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPipeHandleMutable(), MemoryEffects::Write::get());
}

void SetQuantScalarOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getScaleMutable(), MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       getFixpipeQuantStateIdAttr(getOperation(), getId()),
                       FixpipeQuantStateResource::get());
}

void SetQuantVectorOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getScalingTileMutable(), MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       getFixpipeQuantStateIdAttr(getOperation(), getId()),
                       FixpipeQuantStateResource::get());
}

static constexpr const char kConvertRoundingKeywords[] = "r/a/f/c/z/o/h";

static ParseResult parseConvertRounding(OpAsmParser &parser,
                                        RoundingAttr &roundingAttr) {
  StringRef roundingKeyword;
  if (parser.parseKeyword("round") || parser.parseLParen() ||
      parser.parseKeyword(&roundingKeyword) || parser.parseRParen()) {
    return failure();
  }
  std::optional<Rounding> rounding = symbolizeRounding(roundingKeyword);
  if (!rounding) {
    return parser.emitError(parser.getCurrentLocation())
           << "expected convert rounding to be one of "
           << kConvertRoundingKeywords;
  }
  roundingAttr = RoundingAttr::get(parser.getContext(), *rounding);
  return success();
}

static void printConvertRounding(OpAsmPrinter &printer, Operation *op,
                                 RoundingAttr rounding) {
  printer << "round(" << stringifyRounding(rounding.getValue()) << ")";
}

static ParseResult parseConvertSaturation(OpAsmParser &parser,
                                          SaturationAttr &saturationAttr) {
  StringRef saturationKeyword;
  if (parser.parseKeyword(&saturationKeyword)) {
    return failure();
  }
  std::optional<Saturation> saturation =
      symbolizeSaturation(saturationKeyword);
  if (!saturation) {
    return parser.emitError(parser.getCurrentLocation())
           << "expected convert saturation to be sat or nosat";
  }
  saturationAttr = SaturationAttr::get(parser.getContext(), *saturation);
  return success();
}

static void printConvertSaturation(OpAsmPrinter &printer, Operation *op,
                                   SaturationAttr saturation) {
  printer << stringifySaturation(saturation.getValue());
}

static ParseResult parseSignedness(OpAsmParser &parser,
                                   SignednessAttr &signedness) {
  StringRef signednessKeyword;
  if (parser.parseKeyword(&signednessKeyword)) {
    return failure();
  }
  std::optional<Signedness> parsed = symbolizeSignedness(signednessKeyword);
  if (!parsed) {
    return parser.emitError(parser.getCurrentLocation())
           << "expected signedness to be signed or unsigned";
  }
  signedness = SignednessAttr::get(parser.getContext(), *parsed);
  return success();
}

static void printSignedness(OpAsmPrinter &printer, Operation *op,
                            SignednessAttr signedness) {
  printer << stringifySignedness(signedness.getValue());
}

static OptionalParseResult parseOptionalSignedness(OpAsmParser &parser,
                                                   SignednessAttr &signedness) {
  if (succeeded(parser.parseOptionalKeyword("signed"))) {
    signedness = SignednessAttr::get(parser.getContext(), Signedness::Signed);
    return success();
  }
  if (succeeded(parser.parseOptionalKeyword("unsigned"))) {
    signedness =
        SignednessAttr::get(parser.getContext(), Signedness::Unsigned);
    return success();
  }
  return std::nullopt;
}

static void printOptionalSignedness(OpAsmPrinter &printer, Operation *op,
                                    SignednessAttr signedness) {
  printer << stringifySignedness(signedness.getValue());
}

static constexpr const char kLdL2CacheKeywords[] =
    "nmfv/nmlv/nmprs/nmpref/nakeep/naclean/nadrop/idsfv/idslv/idsprs/"
    "idspref/exfv/exlv/exprs/expref";

static constexpr const char kStL2CacheKeywords[] =
    "nmfv/nmlv/nmprs/nmred/naci/napw/napi/nared/wbhfv/wbhlv/wbhprs/"
    "wbhred/wtsfv/wtslv/wtsprs/wtsred";

static ParseResult parseL1Cache(OpAsmParser &parser, L1CacheAttr &l1cache) {
  if (failed(parser.parseOptionalKeyword("l1cache"))) {
    l1cache = L1CacheAttr::get(parser.getContext(), L1Cache::Cache);
    return success();
  }

  StringRef keyword;
  if (parser.parseLParen() || parser.parseKeyword(&keyword) ||
      parser.parseRParen()) {
    return failure();
  }
  std::optional<L1Cache> parsed = symbolizeL1Cache(keyword);
  if (!parsed) {
    return parser.emitError(parser.getCurrentLocation())
           << "expected memory l1cache to be cache or uncache";
  }
  l1cache = L1CacheAttr::get(parser.getContext(), *parsed);
  return success();
}

static void printL1Cache(OpAsmPrinter &printer, Operation *op,
                         L1CacheAttr l1cache) {
  if (!l1cache) {
    return;
  }
  printer << "l1cache(" << stringifyL1Cache(l1cache.getValue()) << ")";
}

static ParseResult parseLdL2Cache(OpAsmParser &parser,
                                  LdL2CacheAttr &l2cache) {
  if (failed(parser.parseOptionalKeyword("l2cache"))) {
    l2cache = LdL2CacheAttr::get(parser.getContext(), LdL2Cache::NMFV);
    return success();
  }

  StringRef keyword;
  if (parser.parseLParen() || parser.parseKeyword(&keyword) ||
      parser.parseRParen()) {
    return failure();
  }
  std::optional<LdL2Cache> parsed = symbolizeLdL2Cache(keyword);
  if (!parsed) {
    return parser.emitError(parser.getCurrentLocation())
           << "expected load L2 cache control to be one of "
           << kLdL2CacheKeywords;
  }
  l2cache = LdL2CacheAttr::get(parser.getContext(), *parsed);
  return success();
}
static void printLdL2Cache(OpAsmPrinter &printer, Operation *op,
                           LdL2CacheAttr l2cache) {
  if (!l2cache) {
    return;
  }
  printer << "l2cache(" << stringifyLdL2Cache(l2cache.getValue()) << ")";
}

static ParseResult parseStL2Cache(OpAsmParser &parser,
                                  StL2CacheAttr &l2cache) {
  if (failed(parser.parseOptionalKeyword("l2cache"))) {
    l2cache = StL2CacheAttr::get(parser.getContext(), StL2Cache::NMFV);
    return success();
  }

  StringRef keyword;
  if (parser.parseLParen() || parser.parseKeyword(&keyword) ||
      parser.parseRParen()) {
    return failure();
  }
  std::optional<StL2Cache> parsed = symbolizeStL2Cache(keyword);
  if (!parsed) {
    return parser.emitError(parser.getCurrentLocation())
           << "expected store L2 cache control to be one of "
           << kStL2CacheKeywords;
  }
  l2cache = StL2CacheAttr::get(parser.getContext(), *parsed);
  return success();
}

static void printStL2Cache(OpAsmPrinter &printer, Operation *op,
                           StL2CacheAttr l2cache) {
  if (!l2cache) {
    return;
  }
  printer << "l2cache(" << stringifyStL2Cache(l2cache.getValue()) << ")";
}

// custom<StructPath>($path): a bracketed list of constant field indices, e.g.
// `[0, 2]`. Backs pto.struct_get / pto.struct_set so they read as `%s[0, 2]`.
static ParseResult parseStructPath(OpAsmParser &parser,
                                   DenseI64ArrayAttr &path) {
  SmallVector<int64_t> indices;
  if (parser.parseLSquare()) {
    return failure();
  }
  if (failed(parser.parseOptionalRSquare())) {
    do {
      int64_t idx = 0;
      if (parser.parseInteger(idx)) {
        return failure();
      }
      indices.push_back(idx);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRSquare()) {
      return failure();
    }
  }
  path = DenseI64ArrayAttr::get(parser.getContext(), indices);
  return success();
}

static void printStructPath(OpAsmPrinter &printer, Operation *op,
                            DenseI64ArrayAttr path) {
  printer << "[";
  llvm::ArrayRef<int64_t> indices = path.asArrayRef();
  for (size_t i = 0; i < indices.size(); ++i) {
    if (i) {
      printer << ", ";
    }
    printer << indices[i];
  }
  printer << "]";
}

// [Include 必须放在最后]
#include "PTO/IR/PTOInterfaces.cpp.inc"
#include "PTO/IR/VPTOInterfaces.cpp.inc"
#define GET_OP_CLASSES
#include "PTO/IR/PTOOps.cpp.inc"

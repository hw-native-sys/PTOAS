// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

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
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());

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

void TPushToAicOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getTileMutable(), MemoryEffects::Read::get());
  auto aivSubblockId = getAivSubblockidMutable();
  if (!aivSubblockId.empty()) {
    addEffect(effects, &*aivSubblockId.begin(), MemoryEffects::Read::get());
  }
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
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

static OptionalParseResult
parseOptionalGenericRounding(OpAsmParser &parser,
                             FloatRoundingModeAttr &roundingAttr) {
  if (failed(parser.parseOptionalKeyword("round"))) {
    return std::nullopt;
  }
  if (parser.parseLParen()) {
    return failure();
  }
  StringRef token;
  bool parseFailed = failed(parser.parseKeyword(&token)) ||
                     failed(parser.parseRParen());
  if (parseFailed) {
    return failure();
  }
  static const llvm::StringMap<FloatRoundingMode> modes = {
      {"r", FloatRoundingMode::to_nearest_even},
      {"a", FloatRoundingMode::to_nearest_away},
      {"f", FloatRoundingMode::downward},
      {"c", FloatRoundingMode::upward},
      {"z", FloatRoundingMode::toward_zero},
      {"o", FloatRoundingMode::to_odd},
      {"h", FloatRoundingMode::hybrid},
  };
  auto it = modes.find(token);
  if (it == modes.end()) {
    return parser.emitError(parser.getCurrentLocation())
           << "expected rounding r/a/f/c/z/o/h";
  }
  roundingAttr = FloatRoundingModeAttr::get(parser.getContext(), it->second);
  return success();
}

static void printOptionalGenericRounding(OpAsmPrinter &printer, [[maybe_unused]] Operation *op,
                                         FloatRoundingModeAttr rounding) {
  if (!rounding) {
    return;
  }
  static constexpr StringLiteral tokens[] = {"r", "f", "c", "z", "a", "o", "h"};
  printer << "round(" << tokens[static_cast<unsigned>(rounding.getValue())]
          << ")";
}

static OptionalParseResult
parseOptionalSaturation(OpAsmParser &parser, SaturationAttr &saturationAttr) {
  if (succeeded(parser.parseOptionalKeyword("sat"))) {
    saturationAttr =
        SaturationAttr::get(parser.getContext(), Saturation::Enable);
  } else if (succeeded(parser.parseOptionalKeyword("nosat"))) {
    saturationAttr =
        SaturationAttr::get(parser.getContext(), Saturation::Disable);
  } else {
    return std::nullopt;
  }
  return success();
}

static void printOptionalSaturation(OpAsmPrinter &printer, [[maybe_unused]] const Operation *op,
                                    SaturationAttr saturation) {
  bool isEnabled =
      saturation && saturation.getValue() != Saturation::Disable;
  if (isEnabled) {
    printer << "sat";
  }
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

static void printSignedness(OpAsmPrinter &printer, const Operation *op,
                            SignednessAttr signedness) {
  (void)op;
  printer << stringifySignedness(signedness.getValue());
}

static ParseResult
parseScalarCmpPredicate(OpAsmParser &parser,
                        ScalarCmpPredicateAttr &predicateAttr) {
  StringRef predicateKeyword;
  if (parser.parseKeyword(&predicateKeyword)) {
    return failure();
  }
  std::optional<ScalarCmpPredicate> predicate =
      symbolizeScalarCmpPredicate(predicateKeyword);
  if (!predicate) {
    return parser.emitError(parser.getCurrentLocation())
           << "expected scalar comparison predicate to be one of "
              "eq, ne, lt, le, gt, ge, or an explicit floating-point "
              "predicate";
  }
  predicateAttr = ScalarCmpPredicateAttr::get(parser.getContext(), *predicate);
  return success();
}

static void printScalarCmpPredicate(OpAsmPrinter &printer, [[maybe_unused]] Operation *op,
                                    ScalarCmpPredicateAttr predicate) {
  printer << stringifyScalarCmpPredicate(predicate.getValue());
}

static Type getPTOI1SameShape(Type type) {
  if (auto vectorType = dyn_cast<VectorType>(type)) {
    return VectorType::Builder(vectorType)
        .setElementType(IntegerType::get(type.getContext(), 1));
  }
  return IntegerType::get(type.getContext(), 1);
}

static ParseResult parseSelectType(OpAsmParser &parser, Type &conditionType,
                                   Type &resultType) {
  Type firstType;
  if (failed(parser.parseType(firstType))) {
    return failure();
  }
  if (succeeded(parser.parseOptionalComma())) {
    conditionType = firstType;
    return parser.parseType(resultType);
  }
  resultType = firstType;
  conditionType = getPTOI1SameShape(resultType);
  return success();
}

static void printSelectType(OpAsmPrinter &printer, [[maybe_unused]] Operation *op,
                            Type conditionType, Type resultType) {
  if (conditionType != getPTOI1SameShape(resultType)) {
    printer << conditionType << ", ";
  }
  printer << resultType;
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

static void printOptionalSignedness(OpAsmPrinter &printer, const Operation *op,
                                    SignednessAttr signedness) {
  (void)op;
  printer << stringifySignedness(signedness.getValue());
}

static constexpr const char kLdL2CacheKeywords[] =
    "nmfv/nmlv/nmprs/nmpref/nakeep/naclean/nadrop/idsfv/idslv/idsprs/"
    "idspref/exfv/exlv/exprs/expref";

static constexpr const char kStL2CacheKeywords[] =
    "nmfv/nmlv/nmprs/nmred/naci/napw/napi/nared/wbhfv/wbhlv/wbhprs/"
    "wbhred/wtsfv/wtslv/wtsprs/wtsred";

// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "VPTOCANN900LLVMEmitterInternal.h"

namespace mlir::pto::detail {

[[maybe_unused]] Value getI1Constant(OpBuilder &builder, Location loc, bool value) {
  return builder.create<arith::ConstantOp>(loc, builder.getIntegerAttr(builder.getI1Type(), value ? 1 : 0)).getResult();
}

bool isMxElementType(Type ty) {
  if (auto floatType = dyn_cast<FloatType>(ty)) {
    return floatType.getWidth() == kBits8;
  }
  if (isa<pto::F4E1M2x2Type, pto::F4E2M1x2Type>(ty)) {
    return true;
  }
  std::string typeText;
  llvm::raw_string_ostream os(typeText);
  ty.print(os);
  os.flush();
  return StringRef(typeText).starts_with("f8");
}

std::string getMadMxElementFragment(Type type) {
  if (type.isF16()) {
    return "f16";
  }
  if (type.isBF16()) {
    return "bf16";
  }

  std::string typeText;
  llvm::raw_string_ostream os(typeText);
  type.print(os);
  os.flush();

  std::string lower = StringRef(typeText).lower();
  if (StringRef(lower).contains("e4m3")) {
    return "e4m3";
  }
  if (StringRef(lower).contains("e5m2")) {
    return "e5m2";
  }
  if (StringRef(lower).contains("hif4")) {
    return "hif4";
  }
  if (StringRef(lower).contains("e2m1x2")) {
    return "e2m1x2";
  }
  if (StringRef(lower).contains("e1m2x2")) {
    return "e1m2x2";
  }
  return {};
}

FailureOr<StringRef> buildMadMxCalleeName(MLIRContext *context, Type lhsElem, Type rhsElem) {
  std::string lhs = getMadMxElementFragment(lhsElem);
  std::string rhs = getMadMxElementFragment(rhsElem);
  if (lhs.empty() || rhs.empty()) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.MMAD.MX." + lhs + rhs).getValue();
}

bool isSignedOrSignlessInteger(IntegerType intType, unsigned width) {
  return intType && intType.getWidth() == width && (intType.isSigned() || intType.isSignless());
}

std::string getMadRhsFragment(Type type) {
  if (type.isF16()) {
    return "f16";
  }
  if (type.isBF16()) {
    return "bf16";
  }
  if (type.isF32()) {
    return "f32";
  }
  if (isMadE4M3ElementType(type)) {
    return "e4m3";
  }
  if (isMadE5M2ElementType(type)) {
    return "e5m2";
  }
  if (pto::isPTOHiFloat8Type(type)) {
    return "hif8";
  }
  if (auto intType = dyn_cast<IntegerType>(type)) {
    if (isSignedOrSignlessInteger(intType, kBits4)) {
      return "s4";
    }
    if (isSignedOrSignlessInteger(intType, kBits8)) {
      return "s8";
    }
    if (intType.isUnsigned() && intType.getWidth() == kBits2) {
      return "u2";
    }
  }

  std::string typeText;
  llvm::raw_string_ostream os(typeText);
  type.print(os);
  os.flush();
  std::string lower = StringRef(typeText).lower();
  if (StringRef(lower).contains("e8m0")) {
    return "e8m0";
  }
  return {};
}

bool isMadE4M3ElementType(Type type) { return pto::isPTOFloat8E4M3LikeType(type); }

bool isMadE5M2ElementType(Type type) { return pto::isPTOFloat8E5M2LikeType(type); }

ArrayRef<MadCalleeContract> getMadCalleeContracts() {
  static constexpr MadCalleeContract contracts[] = {
      {"f16", "f16", "f32", "llvm.hivm.MAD.f162f32.c310"},    {"f16", "f16", "f16", "llvm.hivm.MAD.f162f16"},
      {"f16", "f16", "s32", "llvm.hivm.MAD.f162s32.1952"},    {"bf16", "bf16", "f32", "llvm.hivm.MAD.bf162f32.c310"},
      {"f32", "f32", "f32", "llvm.hivm.MAD.f322f32.c310"},    {"s8", "s8", "s32", "llvm.hivm.MAD.s8.c310"},
      {"e4m3", "e4m3", "f32", "llvm.hivm.MAD.e4m3e4m3.c310"}, {"e4m3", "e5m2", "f32", "llvm.hivm.MAD.e4m3e5m2.c310"},
      {"e5m2", "e4m3", "f32", "llvm.hivm.MAD.e5m2e4m3.c310"}, {"e5m2", "e5m2", "f32", "llvm.hivm.MAD.e5m2e5m2.c310"},
      {"hif8", "hif8", "f32", "llvm.hivm.MAD.e4m3e4m3.c310"}, {"f16", "s4", "", "llvm.hivm.MAD.f16s4.c310"},
      {"f16", "s8", "", "llvm.hivm.MAD.f16s8.c310"},          {"f16", "u2", "", "llvm.hivm.MAD.f16u2"},
      {"f16", "e8m0", "", "llvm.hivm.MAD.f16e8m0.c310"},
  };
  return contracts;
}

FailureOr<StringRef> buildMadTypedCalleeName(MLIRContext *context, Type lhsElem, Type rhsElem, Type dstElem) {
  if (pto::isPTOHiFloat8Type(lhsElem) && pto::isPTOHiFloat8Type(rhsElem) && dstElem.isF32()) {
    return StringAttr::get(context, "llvm.hivm.MAD.e4m3e4m3.c310").getValue();
  }
  std::string lhs = getMadLhsFragment(lhsElem);
  std::string rhs = getMadRhsFragment(rhsElem);
  std::string dst = getMadDstFragment(dstElem);
  for (const MadCalleeContract &contract : getMadCalleeContracts()) {
    if (contract.lhs == lhs && contract.rhs == rhs && (contract.dst.empty() || contract.dst == dst)) {
      return StringAttr::get(context, contract.callee).getValue();
    }
  }
  return failure();
}

FailureOr<StringRef> buildLaneTypedCallee(MLIRContext *context, Type resultType, StringRef stem, StringRef suffix) {
  std::string vec = getElementTypeFragment(getElementTypeFromVectorLike(resultType));
  auto lanes = getElementCountFromVectorLike(resultType);
  if (vec.empty() || !lanes) {
    return failure();
  }

  return StringAttr::get(context, "llvm.hivm." + stem.str() + ".v" + std::to_string(*lanes) + vec + suffix.str())
      .getValue();
}

std::string getCANN900VectorElementFragment(Type type) {
  if (type.isF16()) {
    return "f16";
  }
  if (type.isBF16()) {
    return "bf16";
  }
  if (type.isF32()) {
    return "f32";
  }
  if (std::string lowPrecision = getLowPrecisionElementFragment(type); !lowPrecision.empty()) {
    return lowPrecision;
  }
  if (auto intType = dyn_cast<IntegerType>(type)) {
    return "i" + std::to_string(intType.getWidth());
  }
  return {};
}

std::string getCANN900VectorTypeFragment(Type vectorType) {
  std::string elem = getCANN900VectorElementFragment(getElementTypeFromVectorLike(vectorType));
  auto lanes = getElementCountFromVectorLike(vectorType);
  if (elem.empty() || !lanes) {
    return {};
  }
  return "v" + std::to_string(*lanes) + elem;
}

std::string getCANN900SignednessFragment(Type elemType) {
  if (elemType.isF16() || elemType.isBF16() || elemType.isF32()) {
    return "s";
  }
  if (auto intType = dyn_cast<IntegerType>(elemType)) {
    return intType.isUnsigned() ? "u" : "s";
  }
  return {};
}

FailureOr<StringRef> buildCANN900ModeTypedCallee(MLIRContext *context, Type vectorType, StringRef stem,
                                                 StringRef mode) {
  std::string vec = getCANN900VectorTypeFragment(vectorType);
  if (vec.empty()) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm." + stem.str() + "." + mode.str() + "." + vec).getValue();
}

FailureOr<StringRef> buildCANN900SignedModeTypedCallee(MLIRContext *context, Type vectorType, StringRef stem,
                                                       StringRef mode) {
  std::string vec = getCANN900VectorTypeFragment(vectorType);
  std::string signedness = getCANN900SignednessFragment(getElementTypeFromVectorLike(vectorType));
  if (vec.empty() || signedness.empty()) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm." + stem.str() + "." + signedness + "." + mode.str() + "." + vec)
      .getValue();
}

FailureOr<StringRef> buildCANN900WideningReductionCallee(MLIRContext *context, Type inputType, Type resultType,
                                                         StringRef stem, StringRef mode) {
  std::string inputVec = getCANN900VectorTypeFragment(inputType);
  std::string resultVec = getCANN900VectorTypeFragment(resultType);
  std::string signedness = getCANN900SignednessFragment(getElementTypeFromVectorLike(inputType));
  if (inputVec.empty() || resultVec.empty() || signedness.empty()) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm." + stem.str() + "." + signedness + "." + mode.str() + "." + resultVec +
                                      "." + inputVec)
      .getValue();
}

std::string getCANN900MemoryElementTypeFragment(Type type) {
  if (pto::isPTOHiFloat8Type(type)) {
    return "s8";
  }
  return getMemoryElementTypeFragment(type);
}

bool isLowpPayloadElementType(Type type) {
  return pto::isPTOFloat8Type(type) || pto::isPTOHiFloat8Type(type) || pto::isPTOFloat4PackedType(type);
}

std::optional<LowpPayloadABI> getLowpPayloadABI(Type elementType, MLIRContext *context) {
  if (!isLowpPayloadElementType(elementType)) {
    return std::nullopt;
  }
  return LowpPayloadABI{IntegerType::get(context, 8), "u8"};
}

std::string getDirectLowpVLogicElementFragment(Type type) {
  if (pto::isPTOFloat8E4M3LikeType(type)) {
    return "fp8e4m3";
  }
  if (pto::isPTOFloat8E5M2LikeType(type)) {
    return "fp8e5m2";
  }
  return {};
}

FailureOr<StringRef> buildDirectLowpVLogicCallee(MLIRContext *context, Type vectorType, StringRef stem,
                                                 StringRef mode) {
  Type elementType = getElementTypeFromVectorLike(vectorType);
  auto lanes = getElementCountFromVectorLike(vectorType);
  std::string elem = getDirectLowpVLogicElementFragment(elementType);
  if (elem.empty() || !lanes) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm." + stem.str() + "." + mode.str() + ".v" + std::to_string(*lanes) + elem)
      .getValue();
}

FailureOr<StringRef> buildLowpPayloadVLogicCallee(MLIRContext *context, Type vectorType, StringRef stem,
                                                  StringRef mode) {
  Type elementType = getElementTypeFromVectorLike(vectorType);
  auto lanes = getElementCountFromVectorLike(vectorType);
  std::optional<LowpPayloadABI> abi = getLowpPayloadABI(elementType, context);
  if (!abi || !lanes) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm." + stem.str() + "." + mode.str() + ".v" + std::to_string(*lanes) +
                                      abi->intrinsicElementFragment.str())
      .getValue();
}

Type getLowpPayloadCarrierType(Type vectorLikeType, MLIRContext *context) {
  Type elementType = getElementTypeFromVectorLike(vectorLikeType);
  std::optional<LowpPayloadABI> abi = getLowpPayloadABI(elementType, context);
  if (!abi) {
    return {};
  }
  auto lanes = getElementCountFromVectorLike(vectorLikeType);
  if (!lanes) {
    return {};
  }
  return VectorType::get({*lanes}, abi->llvmElementType);
}

Type getPayloadABIType(Type semanticType, Type convertedType, MLIRContext *context) {
  if (Type carrierType = getLowpPayloadCarrierType(semanticType, context)) {
    return carrierType;
  }
  return convertedType;
}

Value castToPayloadABI(Location loc, Value value, Type semanticType, ConversionPatternRewriter &rewriter) {
  Type carrierType = getLowpPayloadCarrierType(semanticType, rewriter.getContext());
  if (!carrierType || carrierType == value.getType()) {
    return value;
  }
  return rewriter.create<LLVM::BitcastOp>(loc, carrierType, value);
}

Value castFromPayloadABI(Location loc, Value value, Type semanticType, Type convertedType,
                         ConversionPatternRewriter &rewriter) {
  Type carrierType = getLowpPayloadCarrierType(semanticType, rewriter.getContext());
  if (!carrierType || carrierType == convertedType) {
    return value;
  }
  return rewriter.create<LLVM::BitcastOp>(loc, convertedType, value);
}

std::string getAtomicElementTypeFragment(Type type, Attribute signednessAttr) {
  if (auto vecType = dyn_cast<VectorType>(type)) {
    if (vecType.getRank() != 1 || vecType.getDimSize(0) != 2) {
      return {};
    }
    if (vecType.getElementType().isF16()) {
      return "f16x2";
    }
    if (vecType.getElementType().isBF16()) {
      return "bf16x2";
    }
    return {};
  }
  if (type.isF16()) {
    return "fp16";
  }
  if (type.isBF16()) {
    return "bf16";
  }
  if (type.isF32()) {
    return "fp32";
  }
  auto intType = dyn_cast<IntegerType>(type);
  if (!intType) {
    return {};
  }
  if (intType.getWidth() != kBits32 && intType.getWidth() != kBits64) {
    return {};
  }
  if (signednessAttr) {
    auto signedness = cast<pto::SignednessAttr>(signednessAttr).getValue();
    return std::string(signedness == pto::Signedness::Unsigned ? "u" : "s") + std::to_string(intType.getWidth());
  }
  return std::string(intType.isUnsigned() ? "u" : "s") + std::to_string(intType.getWidth());
}

std::string getL0LoadElementFragment(Type type) {
  std::string elem = getElementTypeFragment(type);
  if (!elem.empty()) {
    return elem;
  }

  std::string typeText;
  llvm::raw_string_ostream os(typeText);
  type.print(os);
  os.flush();
  std::string lower = StringRef(typeText).lower();
  if (StringRef(lower).contains("e4m3") || StringRef(lower).contains("e5m2") || StringRef(lower).contains("e8m0") ||
      StringRef(lower).contains("hif8") || StringRef(lower).contains("e1m2x2") || StringRef(lower).contains("e2m1x2")) {
    return "s8";
  }
  return {};
}

std::string getShuffleIntrinsicTypeFragment(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    switch (intType.getWidth()) {
    case kBits32:
      return "i32";
    case kBits64:
      return "i64";
    default:
      return {};
    }
  }
  if (type.isF16()) {
    return "f16";
  }
  if (type.isF32()) {
    return "f32";
  }
  if (auto vecType = dyn_cast<VectorType>(type)) {
    if (vecType.getRank() == 1 && vecType.getDimSize(0) == kVectorPairLaneCount && vecType.getElementType().isF16()) {
      return "v2f16";
    }
  }
  return {};
}

std::string getReduxIntrinsicTypeFragment(Type type, Attribute signednessAttr) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    if (intType.getWidth() != kBits32) {
      return {};
    }
    bool isUnsigned = false;
    if (signednessAttr) {
      isUnsigned = cast<pto::SignednessAttr>(signednessAttr).getValue() == pto::Signedness::Unsigned;
    }
    return isUnsigned ? "u32" : "s32";
  }
  if (type.isF16()) {
    return "f16";
  }
  if (type.isF32()) {
    return "f32";
  }
  return {};
}

FailureOr<Value> normalizeVdupScalarOperand(OpBuilder &builder, Location loc, Value input, Type resultType) {
  auto intType = dyn_cast<IntegerType>(input.getType());
  if (!intType || intType.getWidth() != kBits8) {
    return input;
  }

  Type resultElemType = getElementTypeFromVectorLike(resultType);
  std::string resultElemFragment = getElementTypeFragment(resultElemType);
  if (resultElemFragment != "s8" && resultElemFragment != "u8") {
    return input;
  }

  if (intType.isSignless()) {
    return input;
  }

  Type signlessType = builder.getIntegerType(intType.getWidth());
  return builder.create<UnrealizedConversionCastOp>(loc, TypeRange{signlessType}, input).getResult(0);
}

Value normalizeByteScalarOperandForCANN900VectorCall(OpBuilder &builder, Location loc, Value input,
                                                     Type semanticElementType) {
  (void)semanticElementType;
  auto intType = dyn_cast<IntegerType>(input.getType());
  if (!intType || intType.getWidth() != kBits8 || intType.isSignless()) {
    return input;
  }

  Type signlessType = builder.getIntegerType(kBits8);
  return builder.create<UnrealizedConversionCastOp>(loc, TypeRange{signlessType}, input).getResult(0);
}

bool isCompatibleScalarForSemanticType(Type semanticType, Type scalarType) {
  if (semanticType == scalarType) {
    return true;
  }

  auto semanticInt = dyn_cast<IntegerType>(semanticType);
  auto scalarInt = dyn_cast<IntegerType>(scalarType);
  if (!semanticInt || !scalarInt || semanticInt.getWidth() != scalarInt.getWidth()) {
    return false;
  }

  if (semanticInt.isSigned()) {
    return scalarInt.isSigned() || scalarInt.isSignless();
  }
  if (semanticInt.isUnsigned()) {
    return scalarInt.isUnsigned() || scalarInt.isSignless();
  }
  return scalarInt.isSignless();
}

std::string getNd2NzCopyElementFragment(Type elementType) {
  if (!elementType) {
    return {};
  }
  std::string typeText;
  llvm::raw_string_ostream os(typeText);
  elementType.print(os);
  os.flush();
  std::string lower = StringRef(typeText).lower();
  if (StringRef(lower).contains("e4m3") || StringRef(lower).contains("e5m2") || StringRef(lower).contains("e8m0") ||
      StringRef(lower).contains("hif8")) {
    return "U8";
  }
  if (StringRef(lower).contains("e1m2x2") || StringRef(lower).contains("e2m1x2")) {
    return "U8";
  }

  if (elementType.isF16() || elementType.isBF16()) {
    return "U16";
  }
  if (elementType.isF32()) {
    return "U32";
  }
  if (auto intType = dyn_cast<IntegerType>(elementType)) {
    switch (intType.getWidth()) {
    case kBits8:
      return "U8";
    case kBits16:
      return "U16";
    case kBits32:
      return "U32";
    default:
      return {};
    }
  }
  return {};
}

std::optional<uint64_t> parsePredicatePatternImmediate(StringRef pattern) {
  static constexpr std::pair<StringRef, uint64_t> patternImmediates[] = {
      {"PAT_ALL", 0},  {"PAT_VL1", 1},   {"PAT_VL2", 2},   {"PAT_VL3", 3},   {"PAT_VL4", 4},
      {"PAT_VL8", 5},  {"PAT_VL16", 6},  {"PAT_VL32", 7},  {"PAT_VL64", 8},  {"PAT_VL128", 9},
      {"PAT_M3", 10},  {"PAT_M4", 11},   {"PAT_H", 12},    {"PAT_Q", 13},    {"PAT_ALLF", 15},
  };
  for (const auto &[name, immediate] : patternImmediates) {
    if (pattern == name) {
      return immediate;
    }
  }
  return std::nullopt;
}

std::optional<uint64_t> parseHiLoPartImmediate(StringRef part) {
  if (part == "LOWER") {
    return 0;
  }
  if (part == "HIGHER") {
    return 1;
  }
  return std::nullopt;
}

std::optional<int32_t> parsePostModeImmediate(StringRef mode) {
  if (mode == "NO_POST_UPDATE") {
    return 0;
  }
  if (mode == "POST_UPDATE") {
    return 1;
  }
  return std::nullopt;
}

std::optional<uint64_t> parsePipeImmediate(StringRef pipe) {
  static constexpr std::pair<StringRef, uint64_t> pipeImmediates[] = {
      {"PIPE_S", 0},    {"PIPE_V", 1},    {"PIPE_M", 2},    {"PIPE_MTE1", 3}, {"PIPE_MTE2", 4},
      {"PIPE_MTE3", 5}, {"PIPE_ALL", 6},  {"PIPE_MTE4", 7}, {"PIPE_MTE5", 8}, {"PIPE_V2", 9},
      {"PIPE_FIX", 10}, {"VIRTUAL_PIPE_MTE2_L1A", 11}, {"VIRTUAL_PIPE_MTE2_L1B", 12},
  };
  for (const auto &[name, immediate] : pipeImmediates) {
    if (pipe == name) {
      return immediate;
    }
  }
  return std::nullopt;
}

std::optional<uint64_t> parseEventImmediate(StringRef event) {
  if (!event.consume_front("EVENT_ID")) {
    return std::nullopt;
  }
  uint64_t value = 0;
  if (event.getAsInteger(kRadixDecimal, value)) {
    return std::nullopt;
  }
  return value;
}

std::optional<uint64_t> parseSprImmediate(StringRef spr) {
  if (spr == "AR") {
    return kSprArImmediate;
  }
  return std::nullopt;
}

std::optional<unsigned> getDistElementWidth(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    return intType.getWidth();
  }
  if (isLowpPayloadElementType(type)) {
    return kBits8;
  }
  if (type.isF16() || type.isBF16()) {
    return kBits16;
  }
  if (type.isF32()) {
    return kBits32;
  }
  if (type.isF64()) {
    return kBits64;
  }
  // bf16x2 is a 32-bit packed pair; its dist width is 32 (i32/align4 ABI).
  if (pto::isPTOBF16x2Type(type)) {
    return kBits32;
  }
  return std::nullopt;
}

VcvtElemKind classifyVcvtElemType(Type type) {
  if (type.isF16()) {
    return VcvtElemKind::F16;
  }
  if (type.isBF16()) {
    return VcvtElemKind::BF16;
  }
  if (type.isF32()) {
    return VcvtElemKind::F32;
  }
  if (pto::isPTOFloat8E4M3LikeType(type)) {
    return VcvtElemKind::F8E4M3;
  }
  if (pto::isPTOFloat8E5M2LikeType(type)) {
    return VcvtElemKind::F8E5M2;
  }
  if (pto::isPTOHiFloat8Type(type)) {
    return VcvtElemKind::HiF8;
  }
  if (isa<pto::F4E1M2x2Type>(type)) {
    return VcvtElemKind::F4E1M2x2;
  }
  if (isa<pto::F4E2M1x2Type>(type)) {
    return VcvtElemKind::F4E2M1x2;
  }
  if (auto intType = dyn_cast<IntegerType>(type)) {
    switch (intType.getWidth()) {
    case kBits8:
      return intType.isUnsigned() ? VcvtElemKind::U8 : VcvtElemKind::S8;
    case kBits16:
      return intType.isUnsigned() ? VcvtElemKind::U16 : VcvtElemKind::S16;
    case kBits32:
      return intType.isUnsigned() ? VcvtElemKind::U32 : VcvtElemKind::S32;
    case kBits64:
      return intType.isUnsigned() ? VcvtElemKind::Invalid : VcvtElemKind::S64;
    default:
      return VcvtElemKind::Invalid;
    }
  }
  return VcvtElemKind::Invalid;
}

struct VcvtContractEntry {
  VcvtElemKind src;
  VcvtElemKind dst;
  VcvtContract contract;
};

constexpr VcvtContractEntry kVcvtContractEntries[] = {
    {VcvtElemKind::F32, VcvtElemKind::F8E4M3, {"llvm.hivm.vcvtff.f322f8e4m3.x", true, true, true, 32, false}},
    {VcvtElemKind::F32, VcvtElemKind::F8E5M2, {"llvm.hivm.vcvtff.f322f8e5m2.x", true, true, true, 32, false}},
    {VcvtElemKind::F32, VcvtElemKind::HiF8, {"llvm.hivm.vcvtff.f322hif8.x", true, true, true, 32, false}},
    {VcvtElemKind::F32, VcvtElemKind::F16, {"llvm.hivm.vcvtff.f322f16.x", true, true, true, 32, false}},
    {VcvtElemKind::F32, VcvtElemKind::BF16, {"llvm.hivm.vcvtff.f322bf16.x", true, true, true, 32, false}},
    {VcvtElemKind::F32, VcvtElemKind::S16, {"llvm.hivm.vcvtfi.f322s16.x", true, true, true, 32, false}},
    {VcvtElemKind::F32, VcvtElemKind::S32, {"llvm.hivm.vcvtfi.f322s32.x", true, true, false, 32, false}},
    {VcvtElemKind::F32, VcvtElemKind::S64, {"llvm.hivm.vcvtfi.f322s64.x", true, true, true, 32, false}},
    {VcvtElemKind::F16, VcvtElemKind::F8E4M3, {"llvm.hivm.vcvtff.f162f8e4m3.x", true, true, true, 16, false}},
    {VcvtElemKind::F16, VcvtElemKind::F8E5M2, {"llvm.hivm.vcvtff.f162f8e5m2.x", true, true, true, 16, false}},
    {VcvtElemKind::F16, VcvtElemKind::HiF8, {"llvm.hivm.vcvtff.f162hif8.x", true, true, true, 16, false}},
    {VcvtElemKind::F16, VcvtElemKind::F32, {"llvm.hivm.vcvtff.f162f32.x", false, false, true, 16, false}},
    {VcvtElemKind::F16, VcvtElemKind::BF16, {"llvm.hivm.vcvtff.f162bf16.x", true, false, false, 16, false}},
    {VcvtElemKind::F16, VcvtElemKind::S32, {"llvm.hivm.vcvtfi.f162s32.x", true, false, true, 16, false}},
    {VcvtElemKind::F16, VcvtElemKind::S16, {"llvm.hivm.vcvtfi.f162s16.x", true, true, false, 16, false}},
    {VcvtElemKind::F16, VcvtElemKind::S8, {"llvm.hivm.vcvtfi.f162s8.x", true, true, true, 16, false}},
    {VcvtElemKind::F16, VcvtElemKind::U8, {"llvm.hivm.vcvtfi.f162u8.x", true, true, true, 16, false}},
    {VcvtElemKind::BF16, VcvtElemKind::F8E4M3, {"llvm.hivm.vcvtff.bf162f8e4m3.x", true, true, true, 16, false}},
    {VcvtElemKind::BF16, VcvtElemKind::F8E5M2, {"llvm.hivm.vcvtff.bf162f8e5m2.x", true, true, true, 16, false}},
    {VcvtElemKind::BF16, VcvtElemKind::F4E1M2x2, {"llvm.hivm.vcvtff2.bf162f4e1m2x2.x", true, false, true, 16, false}},
    {VcvtElemKind::BF16, VcvtElemKind::F4E2M1x2, {"llvm.hivm.vcvtff2.bf162f4e2m1x2.x", true, false, true, 16, false}},
    {VcvtElemKind::BF16, VcvtElemKind::F16, {"llvm.hivm.vcvtff.bf162f16.x", true, true, false, 16, true}},
    {VcvtElemKind::BF16, VcvtElemKind::F32, {"llvm.hivm.vcvtff.bf162f32.x", false, false, true, 16, false}},
    {VcvtElemKind::BF16, VcvtElemKind::S32, {"llvm.hivm.vcvtfi.bf162s32.x", true, true, true, 16, false}},
    {VcvtElemKind::U8, VcvtElemKind::F16, {"llvm.hivm.vcvtif.u82f16.x", false, false, true, 8, false}},
    {VcvtElemKind::U8, VcvtElemKind::U16, {"llvm.hivm.vcvtii.u82u16.x", false, false, true, 8, false}},
    {VcvtElemKind::U8, VcvtElemKind::U32, {"llvm.hivm.vcvtii.u82u32.x", false, false, true, 8, false}},
    {VcvtElemKind::S8, VcvtElemKind::F16, {"llvm.hivm.vcvtif.s82f16.x", false, false, true, 8, false}},
    {VcvtElemKind::S8, VcvtElemKind::S16, {"llvm.hivm.vcvtii.s82s16.x", false, false, true, 8, false}},
    {VcvtElemKind::S8, VcvtElemKind::S32, {"llvm.hivm.vcvtii.s82s32.x", false, false, true, 8, false}},
    {VcvtElemKind::U16, VcvtElemKind::U8, {"llvm.hivm.vcvtii.u162u8.x", false, true, true, 16, false}},
    {VcvtElemKind::U16, VcvtElemKind::U32, {"llvm.hivm.vcvtii.u162u32.x", false, false, true, 16, false}},
    {VcvtElemKind::S16, VcvtElemKind::F16, {"llvm.hivm.vcvtif.s162f16.x", true, false, false, 16, false}},
    {VcvtElemKind::S16, VcvtElemKind::F32, {"llvm.hivm.vcvtif.s162f32.x", false, false, true, 16, false}},
    {VcvtElemKind::S16, VcvtElemKind::U8, {"llvm.hivm.vcvtii.s162u8.x", false, true, true, 16, false}},
    {VcvtElemKind::S16, VcvtElemKind::U32, {"llvm.hivm.vcvtii.s162u32.x", false, false, true, 16, false}},
    {VcvtElemKind::S16, VcvtElemKind::S32, {"llvm.hivm.vcvtii.s162s32.x", false, false, true, 16, false}},
    {VcvtElemKind::U32, VcvtElemKind::U8, {"llvm.hivm.vcvtii.u322u8.x", false, true, true, 32, false}},
    {VcvtElemKind::U32, VcvtElemKind::U16, {"llvm.hivm.vcvtii.u322u16.x", false, true, true, 32, false}},
    {VcvtElemKind::U32, VcvtElemKind::S16, {"llvm.hivm.vcvtii.u322s16.x", false, true, true, 32, false}},
    {VcvtElemKind::S32, VcvtElemKind::F32, {"llvm.hivm.vcvtif.s322f32.x", true, false, false, 32, false}},
    {VcvtElemKind::S32, VcvtElemKind::U8, {"llvm.hivm.vcvtii.s322u8.x", false, true, true, 32, false}},
    {VcvtElemKind::S32, VcvtElemKind::U16, {"llvm.hivm.vcvtii.s322u16.x", false, true, true, 32, false}},
    {VcvtElemKind::S32, VcvtElemKind::S16, {"llvm.hivm.vcvtii.s322s16.x", false, true, true, 32, false}},
    {VcvtElemKind::S32, VcvtElemKind::S64, {"llvm.hivm.vcvtii.s322s64.x", false, false, true, 32, false}},
    {VcvtElemKind::S64, VcvtElemKind::F32, {"llvm.hivm.vcvtif.s642f32.x", true, false, true, 32, false}},
    {VcvtElemKind::S64, VcvtElemKind::S32, {"llvm.hivm.vcvtii.s642s32.x", false, true, true, 32, false}},
    {VcvtElemKind::F8E4M3, VcvtElemKind::F32, {"llvm.hivm.vcvtff.f8e4m32f32.x", false, false, true, 8, false}},
    {VcvtElemKind::F8E5M2, VcvtElemKind::F32, {"llvm.hivm.vcvtff.f8e5m22f32.x", false, false, true, 8, false}},
    {VcvtElemKind::HiF8, VcvtElemKind::F32, {"llvm.hivm.vcvtff.hif82f32.x", false, false, true, 8, false}},
    {VcvtElemKind::F4E1M2x2, VcvtElemKind::BF16, {"llvm.hivm.vcvtff2.f4e1m2x22bf16.x", false, false, true, 8, false}},
    {VcvtElemKind::F4E2M1x2, VcvtElemKind::BF16, {"llvm.hivm.vcvtff2.f4e2m1x22bf16.x", false, false, true, 8, false}},
};

std::optional<VcvtContract> lookupVcvtContract(VcvtElemKind src, VcvtElemKind dst) {
  for (const VcvtContractEntry &entry : kVcvtContractEntries) {
    if (entry.src == src && entry.dst == dst) {
      return entry.contract;
    }
  }
  return std::nullopt;
}
// VSQZ #st hint must only be set when the compacted vector feeds VSTUR.
// Emitting #st=1 without a matching VSTUR consumer can deadlock hardware queues.
uint64_t determineVsqzStoreHint(pto::VsqzOp vsqz) {
  Value result = vsqz.getResult();
  for (Operation *user : result.getUsers()) {
    auto vstur = dyn_cast<pto::VsturOp>(user);
    if (!vstur) {
      continue;
    }
    if (vstur.getValue() == result) {
      return 1;
    }
  }
  return 0;
}

std::optional<uint64_t> parseLoadDistImmediate(StringRef dist, Type elementType) {
  const auto *contract = lookupVPTOMemoryDist(VPTOMemoryOpFamily::Load, dist,
                                              getDistElementWidth(elementType));
  return contract ? std::optional<uint64_t>(contract->a5Immediate)
                  : std::nullopt;
}

} // namespace mlir::pto::detail

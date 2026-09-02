// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Support/CodeConstants.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <algorithm>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VPTOEXPANDWRAPPEROPS
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

enum class DmaArch { A2A3, A5 };

constexpr uint64_t kMxScaleAddressShift = 4;
constexpr uint64_t kSubBlockShift = 18;
constexpr uint64_t kQuantBlockBitShift = 29;
constexpr uint64_t kClipReluShift = 30;
constexpr uint64_t kQuantFieldShift = 34;
constexpr uint64_t kReluModeShift = 39;
constexpr uint64_t kChannelSplitShift = 42;
constexpr uint64_t kNz2ndShift = 43;

static DmaArch getDmaArch(ModuleOp mod) {
  if (!mod) {
    return DmaArch::A2A3;
  }
  auto arch = mod->getAttrOfType<StringAttr>("pto.target_arch");
  if (arch && arch.getValue() == "a5") {
    return DmaArch::A5;
  }
  return DmaArch::A2A3;
}

static pto::AddressSpaceAttr getPointerMemorySpace(Attribute memorySpace,
                                                   MLIRContext *ctx) {
  if (auto addrSpace = dyn_cast_or_null<pto::AddressSpaceAttr>(memorySpace)) {
    return addrSpace;
  }
  if (auto intAttr = dyn_cast_or_null<IntegerAttr>(memorySpace)) {
    return pto::AddressSpaceAttr::get(
        ctx, static_cast<pto::AddressSpace>(intAttr.getInt()));
  }
  return pto::AddressSpaceAttr::get(ctx, pto::AddressSpace::GM);
}

static bool hasZeroReinterpretOffset(memref::ReinterpretCastOp op) {
  for (int64_t offset : op.getStaticOffsets()) {
    if (ShapedType::isDynamic(offset) || offset != 0) {
      return false;
    }
  }
  return true;
}

static Value materializeBufferPointer(Value value, PatternRewriter &rewriter,
                                      Location loc);

static Value materializeTypedPointer(Value value, pto::PtrType ptrType,
                                     PatternRewriter &rewriter, Location loc) {
  auto cast = value.getDefiningOp<UnrealizedConversionCastOp>();
  if (!cast) {
    return value;
  }
  bool isSingleValueCast =
      cast->getNumOperands() == 1 && cast->getNumResults() == 1;
  if (!isSingleValueCast) {
    return {};
  }
  Value basePtr = materializeBufferPointer(cast.getOperand(0), rewriter, loc);
  bool needsPointerCast = basePtr && basePtr.getType() != ptrType;
  if (!needsPointerCast) {
    return basePtr;
  }
  return rewriter.create<pto::CastPtrOp>(loc, ptrType, basePtr).getResult();
}

static Value materializeReinterpretPointer(memref::ReinterpretCastOp cast,
                                           PatternRewriter &rewriter,
                                           Location loc) {
  auto resultType = dyn_cast<MemRefType>(cast.getType());
  if (!resultType || !hasZeroReinterpretOffset(cast)) {
    return {};
  }
  Value basePtr = materializeBufferPointer(cast.getSource(), rewriter, loc);
  if (!basePtr) {
    return {};
  }
  auto ptrType = pto::PtrType::get(
      rewriter.getContext(), resultType.getElementType(),
      getPointerMemorySpace(resultType.getMemorySpace(),
                            rewriter.getContext()));
  bool alreadyHasPointerType = basePtr.getType() == ptrType;
  if (alreadyHasPointerType) {
    return basePtr;
  }
  return rewriter.create<pto::CastPtrOp>(loc, ptrType, basePtr).getResult();
}

static Value materializeBufferPointer(Value value, PatternRewriter &rewriter,
                                      Location loc) {
  if (!value) {
    return {};
  }

  if (auto ptrType = dyn_cast<pto::PtrType>(value.getType())) {
    return materializeTypedPointer(value, ptrType, rewriter, loc);
  }

  if (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast->getNumOperands() != 1 || cast->getNumResults() != 1) {
      return {};
    }
    return materializeBufferPointer(cast.getOperand(0), rewriter, loc);
  }

  if (auto cast = value.getDefiningOp<memref::CastOp>()) {
    return materializeBufferPointer(cast.getSource(), rewriter, loc);
  }

  if (auto cast = value.getDefiningOp<memref::MemorySpaceCastOp>()) {
    return materializeBufferPointer(cast.getSource(), rewriter, loc);
  }

  if (auto cast = value.getDefiningOp<memref::ReinterpretCastOp>()) {
    if (Value pointer = materializeReinterpretPointer(cast, rewriter, loc)) {
      return pointer;
    }
  }

  auto memrefType = dyn_cast<MemRefType>(value.getType());
  if (!memrefType) {
    return {};
  }

  auto ptrType =
      pto::PtrType::get(rewriter.getContext(), memrefType.getElementType(),
                        getPointerMemorySpace(memrefType.getMemorySpace(),
                                              rewriter.getContext()));
  return rewriter.create<pto::CastPtrOp>(loc, ptrType, value).getResult();
}

static Type getBufferElementType(Type type) {
  if (auto ptrType = dyn_cast<pto::PtrType>(type)) {
    return ptrType.getElementType();
  }
  if (auto memrefType = dyn_cast<BaseMemRefType>(type)) {
    return memrefType.getElementType();
  }
  return {};
}

static Value offsetBufferPointer(Value basePtr, Value elementOffset,
                                 PatternRewriter &rewriter, Location loc) {
  if (!basePtr) {
    return {};
  }

  Value offsetIndex = elementOffset;
  if (!offsetIndex.getType().isIndex()) {
    offsetIndex = rewriter.create<arith::IndexCastUIOp>(loc,
                                                        rewriter.getIndexType(),
                                                        elementOffset);
  }
  return rewriter.create<pto::AddPtrOp>(loc, basePtr.getType(), basePtr,
                                        offsetIndex);
}

static bool isKnownOne(Value value) {
  APInt intValue;
  return value && matchPattern(value, m_ConstantInt(&intValue)) &&
         intValue.isOne();
}

static bool shouldRestoreDmaLoopSize(Value loop1Count, Value loop2Count) {
  if (!loop1Count) {
    return false;
  }
  return !isKnownOne(loop1Count) || !isKnownOne(loop2Count);
}

static SmallVector<pto::DmaLoopConfig> collectLoopConfigs(ValueRange counts,
                                                          ValueRange srcStrides,
                                                          ValueRange dstStrides) {
  SmallVector<pto::DmaLoopConfig> loops;
  loops.reserve(counts.size());
  for (auto [count, srcStride, dstStride] :
       llvm::zip(counts, srcStrides, dstStrides)) {
    loops.push_back({count, srcStride, dstStride});
  }
  return loops;
}

static Value offsetPointerByBytes(Value basePtr, Value byteOffset,
                                  PatternRewriter &rewriter, Location loc) {
  if (!basePtr) {
    return {};
  }

  Value basePtrValue = materializeBufferPointer(basePtr, rewriter, loc);
  auto ptrType = dyn_cast_or_null<pto::PtrType>(basePtrValue.getType());
  if (!ptrType) {
    return {};
  }

  APInt constOffset;
  if (matchPattern(byteOffset, m_ConstantInt(&constOffset)) &&
      constOffset.isZero()) {
    return basePtrValue;
  }

  auto bytePtrType =
      pto::PtrType::get(rewriter.getContext(), rewriter.getI8Type(),
                        ptrType.getMemorySpace());
  Value bytePtr =
      rewriter.create<pto::CastPtrOp>(loc, bytePtrType, basePtrValue);
  Value offsetIndex = byteOffset;
  if (!offsetIndex.getType().isIndex()) {
    offsetIndex =
        rewriter.create<arith::IndexCastUIOp>(loc, rewriter.getIndexType(),
                                              offsetIndex);
  }
  Value advanced =
      rewriter.create<pto::AddPtrOp>(loc, bytePtrType, bytePtr, offsetIndex);
  return rewriter.create<pto::CastPtrOp>(loc, ptrType, advanced);
}

[[maybe_unused]] static Value materializeFpcValue(Value fpc,
                                                  PatternRewriter &rewriter,
                                                  Location loc) {
  if (!fpc) {
    return {};
  }
  if (fpc.getType().isInteger(mlir::pto::kValue64)) {
    return fpc;
  }
  if (isa<pto::PtrType>(fpc.getType())) {
    return rewriter.create<pto::CastPtrOp>(loc, rewriter.getI64Type(), fpc);
  }
  return {};
}

static Value materializeI64Value(Value value, PatternRewriter &rewriter,
                                 Location loc) {
  if (!value) {
    return {};
  }
  if (value.getType().isInteger(mlir::pto::kValue64)) {
    return value;
  }
  if (auto intType = dyn_cast<IntegerType>(value.getType())) {
    return rewriter.create<arith::ExtUIOp>(loc, rewriter.getI64Type(), value);
  }
  if (isa<pto::PtrType>(value.getType())) {
    return rewriter.create<pto::CastPtrOp>(loc, rewriter.getI64Type(), value);
  }
  return {};
}

static Value materializeAccStoreScalarPayload(Value value,
                                              PatternRewriter &rewriter,
                                              Location loc) {
  if (!value) {
    return {};
  }
  if (Value raw = materializeI64Value(value, rewriter, loc)) {
    return raw;
  }

  Type type = value.getType();
  Value f32Value = value;
  if (type.isF16() || type.isBF16()) {
    f32Value = rewriter.create<arith::ExtFOp>(loc, rewriter.getF32Type(), value);
  } else if (!type.isF32()) {
    return {};
  }

  Value bitsI32 = rewriter.create<arith::BitcastOp>(loc, rewriter.getI32Type(), f32Value);
  return rewriter.create<arith::ExtUIOp>(loc, rewriter.getI64Type(), bitsI32);
}

static Value materializeAccStoreClipPayload(Value value, Type destinationElementType,
                                            PatternRewriter &rewriter,
                                            Location loc) {
  if (!value) {
    return {};
  }

  if (value.getType().isF16()) {
    Value bitsI16 =
        rewriter.create<arith::BitcastOp>(loc, rewriter.getI16Type(), value);
    return rewriter.create<arith::ExtUIOp>(loc, rewriter.getI64Type(), bitsI16);
  }

  auto intType = dyn_cast<IntegerType>(value.getType());
  if (!intType) {
    return {};
  }

  Value widened;
  if (auto dstIntType = dyn_cast<IntegerType>(destinationElementType);
      dstIntType && dstIntType.isUnsignedInteger(mlir::pto::kValue8)) {
    widened = rewriter.create<arith::ExtUIOp>(loc, rewriter.getI64Type(), value);
  } else {
    widened = rewriter.create<arith::ExtSIOp>(loc, rewriter.getI64Type(), value);
  }

  Value mask = rewriter.create<arith::ConstantIntOp>(loc, 0xFFFF, mlir::pto::kValue64);
  return rewriter.create<arith::AndIOp>(loc, widened, mask);
}

static Value getI64Constant(Location loc, PatternRewriter &rewriter,
                            uint64_t value) {
  return rewriter.create<arith::ConstantIntOp>(loc, value, mlir::pto::kValue64);
}

static Value deriveMxScaleDestination(Value dataDestination,
                                      PatternRewriter &rewriter,
                                      Location loc) {
  auto ptrType = dyn_cast<pto::PtrType>(dataDestination.getType());
  if (!ptrType) {
    return {};
  }

  Value dataAddress = rewriter.create<pto::CastPtrOp>(
      loc, rewriter.getI64Type(), dataDestination);
  Value scaleAddress = rewriter.create<arith::ShRUIOp>(
      loc, dataAddress,
      getI64Constant(loc, rewriter, kMxScaleAddressShift));
  return rewriter.create<pto::CastPtrOp>(loc, ptrType, scaleAddress);
}

static Value buildAccStoreOptionalEnumValue(Location loc,
                                            std::optional<uint32_t> value,
                                            PatternRewriter &rewriter) {
  return getI64Constant(loc, rewriter, value.value_or(0));
}

static bool isVectorQuantMode(pto::AccStoreQuantPreMode mode) {
  switch (mode) {
  case pto::AccStoreQuantPreMode::QF322HIF8PreVec:
  case pto::AccStoreQuantPreMode::QF322HIF8PreHybridVec:
  case pto::AccStoreQuantPreMode::DEQS32IntVec:
  case pto::AccStoreQuantPreMode::REQ8Vec:
  case pto::AccStoreQuantPreMode::DEQF16Vec:
  case pto::AccStoreQuantPreMode::QF322FP8PreVec:
  case pto::AccStoreQuantPreMode::QF322F32PreVec:
  case pto::AccStoreQuantPreMode::QF162B8PreVec:
  case pto::AccStoreQuantPreMode::QF162S4PreVec:
  case pto::AccStoreQuantPreMode::REQ4Vec:
  case pto::AccStoreQuantPreMode::QF322B8PreVec:
  case pto::AccStoreQuantPreMode::QF322S4PreVec:
  case pto::AccStoreQuantPreMode::DEQS16Vec:
  case pto::AccStoreQuantPreMode::QF162S16PreVec:
  case pto::AccStoreQuantPreMode::QF322F16PreVec:
  case pto::AccStoreQuantPreMode::QF322BF16PreVec:
  case pto::AccStoreQuantPreMode::QS322BF16PreVec:
    return true;
  default:
    return false;
  }
}

static Value encodeFixpipeBufferAddress(Location loc, Value address,
                                        uint64_t unitShift,
                                        PatternRewriter &rewriter) {
  Value segmentMask = getI64Constant(loc, rewriter, 0xffff);
  Value fieldMask = getI64Constant(loc, rewriter, 0xff);
  Value segmentOffset =
      rewriter.create<arith::AndIOp>(loc, address, segmentMask);
  Value scaledAddress = rewriter.create<arith::ShRUIOp>(
      loc, segmentOffset, getI64Constant(loc, rewriter, unitShift));
  return rewriter.create<arith::AndIOp>(loc, scaledAddress, fieldMask);
}

struct AccStorePreOpConfig {
  Value preQuant;
  std::optional<pto::AccStoreQuantPreMode> preQuantMode;
  Value preRelu;
  std::optional<pto::ReluPreMode> preReluMode;
  Value clipValue;
  Type destinationElementType;
};

static Value buildAccStoreFpcValue(Location loc,
                                   const AccStorePreOpConfig &config,
    PatternRewriter &rewriter) {
  Value quantAddress;
  if (config.preQuantMode && isVectorQuantMode(*config.preQuantMode)) {
    if (Value quantPointer =
            materializeI64Value(config.preQuant, rewriter, loc)) {
      quantAddress = encodeFixpipeBufferAddress(
          loc, quantPointer, mlir::pto::kValue7, rewriter);
    }
  }
  Value reluAddress;
  bool usesVectorRelu =
      config.preReluMode &&
      *config.preReluMode == pto::ReluPreMode::VectorRelu;
  if (usesVectorRelu) {
    if (Value reluPointer =
            materializeI64Value(config.preRelu, rewriter, loc)) {
      reluAddress = encodeFixpipeBufferAddress(
          loc, reluPointer, mlir::pto::kValue6, rewriter);
    }
  }
  if (!quantAddress && !reluAddress) {
    return {};
  }
  Value fpc = getI64Constant(loc, rewriter, 0);
  if (quantAddress) {
    Value quantBits = rewriter.create<arith::ShLIOp>(
        loc, quantAddress, getI64Constant(loc, rewriter, mlir::pto::kValue8));
    fpc = rewriter.create<arith::OrIOp>(loc, fpc, quantBits);
  }
  if (reluAddress) {
    Value mask = getI64Constant(loc, rewriter, 0xff);
    Value reluBits = rewriter.create<arith::AndIOp>(loc, reluAddress, mask);
    fpc = rewriter.create<arith::OrIOp>(loc, fpc, reluBits);
  }
  return fpc;
}

static void configureAccStoreScalarPreOps(
    Location loc, const AccStorePreOpConfig &config,
    PatternRewriter &rewriter) {
  bool hasScalarQuant =
      config.preQuantMode &&
      *config.preQuantMode != pto::AccStoreQuantPreMode::NoConvert &&
      !isVectorQuantMode(*config.preQuantMode);
  if (hasScalarQuant) {
    if (Value quantValue =
            materializeAccStoreScalarPayload(config.preQuant, rewriter, loc)) {
      rewriter.create<pto::SetQuantPreOp>(loc, quantValue);
    }
  }
  bool hasScalarRelu =
      config.preReluMode &&
      *config.preReluMode == pto::ReluPreMode::ScalarRelu;
  if (hasScalarRelu) {
    if (Value reluAlpha =
            materializeAccStoreScalarPayload(config.preRelu, rewriter, loc)) {
      rewriter.create<pto::SetReluAlphaOp>(loc, reluAlpha);
    }
  }
  if (config.clipValue) {
    Value clip = materializeAccStoreClipPayload(
        config.clipValue, config.destinationElementType, rewriter, loc);
    if (clip) {
      rewriter.create<pto::SetFixClipReluOp>(loc, clip);
    }
  }
}

struct AccStoreCtrlConfig {
  bool allowAtomic;
  std::optional<pto::AccStoreAtomicType> atomicType;
  std::optional<pto::AccStoreAtomicOp> atomicOp;
  std::optional<pto::AccStoreSatMode> satMode;
};

static Value configureAccStoreCtrl(Location loc,
                                   const AccStoreCtrlConfig &config,
                                   PatternRewriter &rewriter) {
  bool hasAtomic =
      config.allowAtomic && config.atomicType && config.atomicOp;
  if (!hasAtomic && !config.satMode) {
    return {};
  }

  Value originalCtrl = rewriter.create<pto::GetCtrlOp>(loc);
  Value ctrl = originalCtrl;
  uint64_t clearMaskValue = 0;
  if (hasAtomic) {
    clearMaskValue |= (static_cast<uint64_t>(0x7) << mlir::pto::kValue6) |
                      (static_cast<uint64_t>(0x3) << mlir::pto::kValue9);
  }
  if (config.satMode) {
    clearMaskValue |= (static_cast<uint64_t>(1) << mlir::pto::kValue48) |
                      (static_cast<uint64_t>(1) << mlir::pto::kValue50);
  }
  Value clearMask = getI64Constant(loc, rewriter, clearMaskValue);
  Value fullMask = getI64Constant(loc, rewriter, ~static_cast<uint64_t>(0));
  Value keepMask = rewriter.create<arith::XOrIOp>(loc, clearMask, fullMask);
  ctrl = rewriter.create<arith::AndIOp>(loc, ctrl, keepMask);

  if (hasAtomic) {
    uint64_t atomicBits =
        (static_cast<uint64_t>(static_cast<uint32_t>(*config.atomicType))
         << mlir::pto::kValue6) |
        (static_cast<uint64_t>(static_cast<uint32_t>(*config.atomicOp))
         << mlir::pto::kValue9);
    ctrl = rewriter.create<arith::OrIOp>(loc, ctrl,
                                         getI64Constant(loc, rewriter, atomicBits));
  }
  if (config.satMode &&
      *config.satMode == pto::AccStoreSatMode::NoSat) {
    ctrl = rewriter.create<arith::OrIOp>(
        loc, ctrl, getI64Constant(loc, rewriter,
                                  static_cast<uint64_t>(1) << mlir::pto::kValue48));
  }
  if (config.satMode &&
      *config.satMode == pto::AccStoreSatMode::SatPreserveNan) {
    ctrl = rewriter.create<arith::OrIOp>(
        loc, ctrl, getI64Constant(loc, rewriter,
                                  static_cast<uint64_t>(1) << mlir::pto::kValue50));
  }
  rewriter.create<pto::SetCtrlOp>(loc, ctrl);
  return originalCtrl;
}

static Value buildAccumulatedByteOffset(Location loc, Value baseOffset,
                                        Value indexI64, Value stride,
                                        PatternRewriter &rewriter) {
  Value delta = rewriter.create<arith::MulIOp>(loc, indexI64, stride);
  return rewriter.create<arith::AddIOp>(loc, baseOffset, delta);
}

static Value packLoopPair(Location loc, Value low, Value high,
                          PatternRewriter &rewriter) {
  Value shift = rewriter.create<arith::ConstantIntOp>(loc, 40, mlir::pto::kValue64);
  Value highShifted = rewriter.create<arith::ShLIOp>(loc, high, shift);
  return rewriter.create<arith::OrIOp>(loc, highShifted, low);
}

static Value packLoopSize(Location loc, Value loop2, Value loop1,
                          PatternRewriter &rewriter) {
  Value shift = rewriter.create<arith::ConstantIntOp>(loc, 21, mlir::pto::kValue64);
  Value loop2Shifted = rewriter.create<arith::ShLIOp>(loc, loop2, shift);
  return rewriter.create<arith::OrIOp>(loc, loop2Shifted, loop1);
}

static Value castIntegerLikeTo(Location loc, Value value, Type targetType,
                               PatternRewriter &rewriter) {
  if (value.getType() == targetType) {
    return value;
  }

  auto targetInt = dyn_cast<IntegerType>(targetType);
  if (value.getType().isIndex() && targetInt) {
    return rewriter.create<arith::IndexCastOp>(loc, targetType, value);
  }
  if (auto sourceInt = dyn_cast<IntegerType>(value.getType())) {
    if (targetInt) {
      if (sourceInt.getWidth() < targetInt.getWidth()) {
        return rewriter.create<arith::ExtUIOp>(loc, targetType, value);
      }
      if (sourceInt.getWidth() > targetInt.getWidth()) {
        return rewriter.create<arith::TruncIOp>(loc, targetType, value);
      }
      return value;
    }
    if (targetType.isIndex()) {
      return rewriter.create<arith::IndexCastOp>(loc, targetType, value);
    }
  }

  return {};
}

struct MadXtConfig {
  Value m;
  Value n;
  Value k;
  std::optional<pto::MadUnitFlagMode> unitFlagMode;
  bool disableGemv;
  bool cmatrixSource;
  bool cmatrixInit;
};

static FailureOr<Value> packMadXt(Location loc, const MadXtConfig &config,
                                  PatternRewriter &rewriter) {
  Type i64Ty = rewriter.getI64Type();
  Value mI64 = castIntegerLikeTo(loc, config.m, i64Ty, rewriter);
  Value nI64 = castIntegerLikeTo(loc, config.n, i64Ty, rewriter);
  Value kI64 = castIntegerLikeTo(loc, config.k, i64Ty, rewriter);
  if (!mI64 || !nI64 || !kI64) {
    return failure();
  }

  auto constant = [&rewriter, loc](uint64_t value) -> Value {
    return rewriter.create<arith::ConstantIntOp>(loc, value, mlir::pto::kValue64);
  };
  auto shl = [&rewriter, loc, &constant](Value value,
                                         uint64_t amount) -> Value {
    return rewriter.create<arith::ShLIOp>(loc, value, constant(amount));
  };
  auto bitOr = [&rewriter, loc](Value lhs, Value rhs) -> Value {
    return rewriter.create<arith::OrIOp>(loc, lhs, rhs);
  };

  Value xt = mI64;
  xt = bitOr(xt, shl(kI64, mlir::pto::kValue12));
  xt = bitOr(xt, shl(nI64, mlir::pto::kValue24));
  if (config.unitFlagMode) {
    uint64_t unitFlagCtrl =
        *config.unitFlagMode == pto::MadUnitFlagMode::CheckOnly
            ? mlir::pto::kValue2
            : mlir::pto::kValue3;
    xt = bitOr(xt, shl(constant(unitFlagCtrl), mlir::pto::kValue55));
  }
  if (config.disableGemv) {
    xt = bitOr(xt, shl(constant(1), mlir::pto::kValue61));
  }
  if (config.cmatrixSource) {
    xt = bitOr(xt, shl(constant(1), mlir::pto::kValue62));
  }
  if (config.cmatrixInit) {
    xt = bitOr(xt, shl(constant(1), mlir::pto::kValue63));
  }
  return xt;
}

static Value setCtrlBit(Location loc, Value ctrl, unsigned bitIndex, bool value,
                        PatternRewriter &rewriter) {
  Value bit = rewriter.create<arith::ConstantIntOp>(loc, bitIndex, mlir::pto::kValue64);
  if (value) {
    return rewriter.create<pto::Sbitset1Op>(loc, ctrl, bit).getResult();
  }
  return rewriter.create<pto::Sbitset0Op>(loc, ctrl, bit).getResult();
}

struct MadCtrlConfig {
  bool isHif8;
  std::optional<pto::Tf32Mode> tf32Mode;
  std::optional<pto::MadSatMode> satMode;
  bool hasNDir;
};

static Value buildMadSemanticCtrl(Location loc, Value ctrl,
                                  const MadCtrlConfig &config,
                                  PatternRewriter &rewriter) {
  ctrl =
      setCtrlBit(loc, ctrl, mlir::pto::kValue45, config.isHif8, rewriter);
  if (config.tf32Mode) {
    ctrl = setCtrlBit(loc, ctrl, mlir::pto::kValue46, true, rewriter);
    ctrl = setCtrlBit(loc, ctrl, mlir::pto::kValue47,
                      *config.tf32Mode == pto::Tf32Mode::RoundAway, rewriter);
  } else {
    ctrl = setCtrlBit(loc, ctrl, mlir::pto::kValue46, false, rewriter);
    ctrl = setCtrlBit(loc, ctrl, mlir::pto::kValue47, false, rewriter);
  }
  if (config.satMode) {
    bool noSaturation = *config.satMode == pto::MadSatMode::NoSat;
    ctrl = setCtrlBit(loc, ctrl, mlir::pto::kValue48, noSaturation, rewriter);
  }
  ctrl = setCtrlBit(loc, ctrl, mlir::pto::kValue51, config.hasNDir, rewriter);
  return ctrl;
}

struct Mte2NzConfig {
  Value groupCount;
  Value dstLoop2Stride;
  Value dstLoop3Stride;
  Value dstLoop4Stride;
};

static Value packMte2NzPara(Location loc, const Mte2NzConfig &config,
                            PatternRewriter &rewriter) {
  Value shift16 = rewriter.create<arith::ConstantIntOp>(loc, 16, mlir::pto::kValue64);
  Value shift32 = rewriter.create<arith::ConstantIntOp>(loc, 32, mlir::pto::kValue64);
  Value shift48 = rewriter.create<arith::ConstantIntOp>(loc, 48, mlir::pto::kValue64);
  Value loop2Bits =
      rewriter.create<arith::ShLIOp>(loc, config.dstLoop2Stride, shift16);
  Value loop3Bits =
      rewriter.create<arith::ShLIOp>(loc, config.dstLoop3Stride, shift32);
  Value loop4Bits =
      rewriter.create<arith::ShLIOp>(loc, config.dstLoop4Stride, shift48);
  Value low =
      rewriter.create<arith::OrIOp>(loc, config.groupCount, loop2Bits);
  Value high = rewriter.create<arith::OrIOp>(loc, loop3Bits, loop4Bits);
  return rewriter.create<arith::OrIOp>(loc, low, high);
}

struct CopyMatrixXmConfig {
  Value sid;
  Value nSize;
  Value mSize;
  Value dstStride;
};

static Value packCopyMatrixCcToGmXm(Location loc,
                                    const CopyMatrixXmConfig &config,
                                    PatternRewriter &rewriter) {
  Value nShift4 = rewriter.create<arith::ConstantIntOp>(loc, 4, mlir::pto::kValue64);
  Value mShift16 = rewriter.create<arith::ConstantIntOp>(loc, 16, mlir::pto::kValue64);
  Value dstShift32 = rewriter.create<arith::ConstantIntOp>(loc, 32, mlir::pto::kValue64);
  Value nBits =
      rewriter.create<arith::ShLIOp>(loc, config.nSize, nShift4);
  Value mBits =
      rewriter.create<arith::ShLIOp>(loc, config.mSize, mShift16);
  Value dstStrideBits =
      rewriter.create<arith::ShLIOp>(loc, config.dstStride, dstShift32);
  Value sidMask = rewriter.create<arith::ConstantIntOp>(loc, 0xf, mlir::pto::kValue64);
  Value sidBits = rewriter.create<arith::AndIOp>(loc, config.sid, sidMask);
  Value xmLow = rewriter.create<arith::OrIOp>(loc, sidBits, nBits);
  xmLow = rewriter.create<arith::OrIOp>(loc, xmLow, mBits);
  return rewriter.create<arith::OrIOp>(loc, xmLow, dstStrideBits);
}

struct AccStoreModeConfig {
  Value channelLoop0Stride;
  Value nz2nd;
  Value channelSplit;
  Value nz2dn;
};

struct AccStorePackedFields {
  Value clipRelu;
  Value unitFlag;
  Value quantMode;
  Value reluMode;
};

struct ExtractedFieldConfig {
  uint64_t sourceShift;
  uint64_t mask;
  uint64_t targetShift;
};

struct CopyMatrixCcToGmXtConfig {
  Value srcStride;
  Value l2CacheCtrl;
  AccStorePackedFields fields;
  AccStoreModeConfig mode;
};

struct CopyMatrixCcToUbConfig1 {
  Value srcStride;
  Value dualDstMode;
  Value subBlockId;
  AccStorePackedFields fields;
  AccStoreModeConfig mode;
};

static Value packMaskedField(Location loc, Value value, uint64_t mask,
                             uint64_t shift, PatternRewriter &rewriter) {
  Value masked = rewriter.create<arith::AndIOp>(
      loc, value, getI64Constant(loc, rewriter, mask));
  return rewriter.create<arith::ShLIOp>(
      loc, masked, getI64Constant(loc, rewriter, shift));
}

static Value packExtractedField(Location loc, Value value,
                                const ExtractedFieldConfig &config,
                                PatternRewriter &rewriter) {
  Value extracted = rewriter.create<arith::ShRUIOp>(
      loc, value, getI64Constant(loc, rewriter, config.sourceShift));
  return packMaskedField(loc, extracted, config.mask, config.targetShift,
                         rewriter);
}

static Value mergePackedFields(Location loc, Value base,
                               ArrayRef<Value> fields,
                               PatternRewriter &rewriter) {
  Value packed = base;
  for (Value field : fields) {
    packed = rewriter.create<arith::OrIOp>(loc, packed, field);
  }
  return packed;
}

static Value packAccStoreCommonBits(Location loc,
                                    const AccStorePackedFields &fields,
                                    const AccStoreModeConfig &mode,
                                    PatternRewriter &rewriter) {
  SmallVector<Value, mlir::pto::kValue8> packedFields{
      packMaskedField(loc, fields.clipRelu, 0x3, kClipReluShift,
                      rewriter),
      packMaskedField(loc, fields.unitFlag, 0x3, mlir::pto::kValue32,
                      rewriter),
      packExtractedField(
          loc, fields.quantMode,
          {mlir::pto::kValue5, 0x1, kQuantBlockBitShift}, rewriter),
      packMaskedField(loc, fields.quantMode, 0x1f, kQuantFieldShift,
                      rewriter),
      packMaskedField(loc, fields.reluMode, 0x7, kReluModeShift,
                      rewriter),
      packMaskedField(loc, mode.channelSplit, 0x1, kChannelSplitShift,
                      rewriter),
      packMaskedField(loc, mode.nz2nd, 0x1, kNz2ndShift, rewriter),
      packMaskedField(loc, mode.nz2dn, 0x1, mlir::pto::kValue62, rewriter)};
  return mergePackedFields(loc, getI64Constant(loc, rewriter, 0), packedFields,
                           rewriter);
}

static Value
packCopyMatrixCcToGmXt(Location loc,
                       const CopyMatrixCcToGmXtConfig &config,
                       PatternRewriter &rewriter) {
  Value l2CacheBits = packMaskedField(
      loc, config.l2CacheCtrl, 0xf, mlir::pto::kValue16, rewriter);
  Value commonBits =
      packAccStoreCommonBits(loc, config.fields, config.mode, rewriter);
  return mergePackedFields(loc, config.srcStride, {l2CacheBits, commonBits},
                           rewriter);
}

static Value
packCopyMatrixCcToUbConfig1(Location loc,
                            const CopyMatrixCcToUbConfig1 &config,
                            PatternRewriter &rewriter) {
  Value dualDstBits = packMaskedField(
      loc, config.dualDstMode, 0x3, mlir::pto::kValue16, rewriter);
  Value subBlockBits = packMaskedField(
      loc, config.subBlockId, 0x1, kSubBlockShift, rewriter);
  Value commonBits =
      packAccStoreCommonBits(loc, config.fields, config.mode, rewriter);
  return mergePackedFields(loc, config.srcStride,
                           {dualDstBits, subBlockBits, commonBits}, rewriter);
}

static Value packLoop3Config(Location loc, Value count, Value srcStride,
                             Value dstStride, PatternRewriter &rewriter) {
  Value srcShift16 = rewriter.create<arith::ConstantIntOp>(loc, 16, mlir::pto::kValue64);
  Value dstShift32 = rewriter.create<arith::ConstantIntOp>(loc, 32, mlir::pto::kValue64);
  Value srcBits = rewriter.create<arith::ShLIOp>(loc, srcStride, srcShift16);
  Value dstBits = rewriter.create<arith::ShLIOp>(loc, dstStride, dstShift32);
  Value low = rewriter.create<arith::OrIOp>(loc, count, srcBits);
  return rewriter.create<arith::OrIOp>(loc, low, dstBits);
}

static Value packChannelConfig(Location loc, Value loop0SrcStride,
                               PatternRewriter &rewriter) {
  Value shift48 = rewriter.create<arith::ConstantIntOp>(loc, 48, mlir::pto::kValue64);
  return rewriter.create<arith::ShLIOp>(loc, loop0SrcStride, shift48);
}

struct LoadCbufToCbControl {
  Value mStart;
  Value kStart;
  Value mStep;
  Value kStep;
  Value srcStride;
  Value dstStride;
};

struct LoadCbufToMxControl {
  Value xStartPosition;
  Value yStartPosition;
  Value xStep;
  Value yStep;
  Value srcStride;
  Value dstStride;
};

struct CbufControlMath {
  Location loc;
  PatternRewriter &rewriter;

  Value constant(uint64_t value) const {
    return rewriter.create<arith::ConstantIntOp>(loc, value, mlir::pto::kValue64);
  }

  Value ceilDiv(Value value, uint64_t divisor) const {
    Value sum = rewriter.create<arith::AddIOp>(loc, value,
                                               constant(divisor - 1));
    return rewriter.create<arith::DivUIOp>(loc, sum, constant(divisor));
  }
};

struct LoadCbufControlQuery {
  Location loc;
  Value outerSize;
  Value kSize;
  Type elementType;
  Value outerStart;
  Value kStart;
  bool transpose;
  PatternRewriter &rewriter;
};

static Value scalePackedKCoordinate(Value coordinate, bool isFp4Packed,
                                    const CbufControlMath &math) {
  if (!isFp4Packed) {
    return coordinate;
  }
  return math.rewriter.create<arith::DivUIOp>(math.loc, coordinate,
                                               math.constant(mlir::pto::kValue2));
}

static Value deriveCbufKStep(Value byteExtent, bool isFp4Packed,
                             const CbufControlMath &math) {
  uint64_t blockElements = isFp4Packed ? mlir::pto::kValue64
                                       : mlir::pto::kValue32;
  return math.ceilDiv(byteExtent, blockElements);
}

static FailureOr<LoadCbufToCbControl>
deriveLoadCbufControl(const LoadCbufControlQuery &query) {
  unsigned elementBits = pto::getPTOStorageElemBitWidth(query.elementType);
  bool hasWholeBytes =
      elementBits != 0 && elementBits % mlir::pto::kValue8 == 0;
  if (!hasWholeBytes) {
    return failure();
  }
  uint64_t elementBytes = elementBits / mlir::pto::kValue8;
  bool isFp4Packed = pto::isPTOFloat4PackedType(query.elementType);
  CbufControlMath math{query.loc, query.rewriter};
  Value kStart = scalePackedKCoordinate(query.kStart, isFp4Packed, math);
  if (!query.transpose) {
    Value outerStep = math.ceilDiv(query.outerSize, mlir::pto::kValue16);
    Value kBytes = query.rewriter.create<arith::MulIOp>(
        query.loc, query.kSize, math.constant(elementBytes));
    Value kStep = deriveCbufKStep(kBytes, isFp4Packed, math);
    return LoadCbufToCbControl{query.outerStart, kStart, outerStep, kStep,
                               outerStep, outerStep};
  }

  uint64_t c0Size = isFp4Packed
                        ? mlir::pto::kValue64
                        : std::max<uint64_t>(mlir::pto::kValue16,
                                             mlir::pto::kValue32 / elementBytes);
  Value outerAlign = math.ceilDiv(query.outerSize, c0Size);
  outerAlign = query.rewriter.create<arith::MulIOp>(
      query.loc, outerAlign, math.constant(c0Size));
  Value kAlign = math.ceilDiv(query.kSize, c0Size);
  kAlign = query.rewriter.create<arith::MulIOp>(
      query.loc, kAlign, math.constant(c0Size));
  Value outerStep = math.ceilDiv(kAlign, mlir::pto::kValue16);
  Value outerBytes = query.rewriter.create<arith::MulIOp>(
      query.loc, outerAlign, math.constant(elementBytes));
  Value kStep = deriveCbufKStep(outerBytes, isFp4Packed, math);
  Value srcStride = math.ceilDiv(kAlign, mlir::pto::kValue16);
  Value dstStride = math.ceilDiv(outerAlign, mlir::pto::kValue16);
  return LoadCbufToCbControl{query.outerStart, kStart, outerStep, kStep,
                             srcStride, dstStride};
}

enum class CbufMxSide { Left, Right };

struct LoadCbufMxControlQuery {
  Location loc;
  Value outerSize;
  Value kSize;
  Type elementType;
  Value startRow;
  Value startCol;
  CbufMxSide side;
  PatternRewriter &rewriter;
};

static FailureOr<LoadCbufToMxControl>
deriveLoadCbufMxControl(const LoadCbufMxControlQuery &query) {
  unsigned elementBits = pto::getPTOStorageElemBitWidth(query.elementType);
  bool hasWholeBytes =
      elementBits != 0 && elementBits % mlir::pto::kValue8 == 0;
  if (!hasWholeBytes) {
    return failure();
  }
  uint64_t elementBytes = elementBits / mlir::pto::kValue8;
  CbufControlMath math{query.loc, query.rewriter};
  Value kGroups = math.ceilDiv(query.kSize, mlir::pto::kValue32);
  Value xStep = math.ceilDiv(query.outerSize, mlir::pto::kValue16);
  Value packedGroups = query.rewriter.create<arith::MulIOp>(
      query.loc, kGroups, math.constant(elementBytes));
  Value packedStride = math.ceilDiv(packedGroups, mlir::pto::kValue2);
  Value yStep = packedStride;
  Value yStride = packedStride;
  if (query.side == CbufMxSide::Right) {
    yStride = math.ceilDiv(kGroups, mlir::pto::kValue2);
  }
  return LoadCbufToMxControl{query.startRow, query.startCol, xStep, yStep,
                             yStride, yStride};
}

static Value extractConfigLow40(Location loc, Value packed,
                                PatternRewriter &rewriter) {
  Value lowMask =
      rewriter.create<arith::ConstantIntOp>(loc, 0xffffffffffULL, mlir::pto::kValue64);
  return rewriter.create<arith::AndIOp>(loc, packed, lowMask);
}

static Value extractConfigHigh24(Location loc, Value packed,
                                 PatternRewriter &rewriter) {
  Value shift40 = rewriter.create<arith::ConstantIntOp>(loc, 40, mlir::pto::kValue64);
  return rewriter.create<arith::ShRUIOp>(loc, packed, shift40);
}

struct DmaLoopOffsets {
  Value source;
  Value destination;
};

template <typename BodyBuilder>
static void buildSoftwareLoopNest(PatternRewriter &rewriter, Location loc,
                                  ArrayRef<pto::DmaLoopConfig> loops,
                                  DmaLoopOffsets offsets,
                                  BodyBuilder &&buildLeaf) {
  if (loops.empty()) {
    buildLeaf(offsets.source, offsets.destination);
    return;
  }

  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value count = rewriter.create<arith::IndexCastUIOp>(loc, rewriter.getIndexType(),
                                                      loops.front().count);
  scf::ForOp forOp = rewriter.create<scf::ForOp>(loc, c0, count, c1);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(forOp.getBody());
    Value ivI64 =
        rewriter.create<arith::IndexCastUIOp>(loc, rewriter.getI64Type(),
                                              forOp.getInductionVar());
    Value nextSrcOffset = buildAccumulatedByteOffset(
        loc, offsets.source, ivI64, loops.front().srcStride, rewriter);
    Value nextDstOffset = buildAccumulatedByteOffset(
        loc, offsets.destination, ivI64, loops.front().dstStride, rewriter);
    buildSoftwareLoopNest(rewriter, loc, loops.drop_front(),
                          {nextSrcOffset, nextDstOffset}, buildLeaf);
  }
}

struct DmaLoopPlan {
  SmallVector<pto::DmaLoopConfig> softwareLoops;
  Value loop1Count;
  Value loop2Count;
};

static DmaLoopPlan configureLoadToUbLoops(
    pto::MteGmUbOp op, DmaArch dmaArch,
    ArrayRef<pto::DmaLoopConfig> loops, Value one,
    PatternRewriter &rewriter) {
  DmaLoopPlan plan{{}, {}, one};
  if (dmaArch != DmaArch::A5) {
    plan.softwareLoops.append(loops.begin(), loops.end());
    plan.softwareLoops.push_back(
        {op.getNBurst(), op.getNburstSrcStride(), op.getNburstDstStride()});
    return plan;
  }

  ArrayRef<pto::DmaLoopConfig> hardwareLoops =
      loops.take_front(mlir::pto::kValue2);
  ArrayRef<pto::DmaLoopConfig> softwareLoops =
      loops.drop_front(hardwareLoops.size());
  plan.softwareLoops.append(softwareLoops.begin(), softwareLoops.end());
  bool hasTwoHardwareLoops = hardwareLoops.size() == mlir::pto::kValue2;
  if (hasTwoHardwareLoops) {
    rewriter.create<pto::SetLoop2StrideOutToUbOp>(
        op.getLoc(), hardwareLoops[0].srcStride, hardwareLoops[0].dstStride);
    plan.loop2Count = hardwareLoops[0].count;
    plan.loop1Count = hardwareLoops[1].count;
    rewriter.create<pto::SetLoop1StrideOutToUbOp>(
        op.getLoc(), hardwareLoops[1].srcStride, hardwareLoops[1].dstStride);
  } else if (hardwareLoops.size() == 1) {
    plan.loop1Count = hardwareLoops[0].count;
    rewriter.create<pto::SetLoop1StrideOutToUbOp>(
        op.getLoc(), hardwareLoops[0].srcStride, hardwareLoops[0].dstStride);
  }
  if (plan.loop1Count) {
    rewriter.create<pto::SetLoopSizeOutToUbOp>(op.getLoc(), plan.loop2Count,
                                               plan.loop1Count);
  }
  return plan;
}

static DmaLoopPlan configureStoreFromUbLoops(
    pto::MteUbGmOp op, DmaArch dmaArch,
    ArrayRef<pto::DmaLoopConfig> loops, Value one,
    PatternRewriter &rewriter) {
  DmaLoopPlan plan{{}, {}, one};
  if (dmaArch != DmaArch::A5) {
    plan.softwareLoops.append(loops.begin(), loops.end());
    plan.softwareLoops.push_back(
        {op.getNBurst(), op.getNburstSrcStride(), op.getNburstDstStride()});
    return plan;
  }

  ArrayRef<pto::DmaLoopConfig> hardwareLoops =
      loops.take_front(mlir::pto::kValue2);
  ArrayRef<pto::DmaLoopConfig> softwareLoops =
      loops.drop_front(hardwareLoops.size());
  plan.softwareLoops.append(softwareLoops.begin(), softwareLoops.end());
  bool hasTwoHardwareLoops = hardwareLoops.size() == mlir::pto::kValue2;
  if (hasTwoHardwareLoops) {
    rewriter.create<pto::SetLoop2StrideUbToOutOp>(
        op.getLoc(), hardwareLoops[0].srcStride, hardwareLoops[0].dstStride);
    plan.loop2Count = hardwareLoops[0].count;
    plan.loop1Count = hardwareLoops[1].count;
    rewriter.create<pto::SetLoop1StrideUbToOutOp>(
        op.getLoc(), hardwareLoops[1].srcStride, hardwareLoops[1].dstStride);
  } else if (hardwareLoops.size() == 1) {
    plan.loop1Count = hardwareLoops[0].count;
    rewriter.create<pto::SetLoop1StrideUbToOutOp>(
        op.getLoc(), hardwareLoops[0].srcStride, hardwareLoops[0].dstStride);
  }
  if (plan.loop1Count) {
    rewriter.create<pto::SetLoopSizeUbToOutOp>(op.getLoc(), plan.loop2Count,
                                               plan.loop1Count);
  }
  return plan;
}

struct DmaPaddingConfig {
  Value left;
  Value right;
  Value dataSelect;
  bool enabled;
};

static DmaPaddingConfig configureDmaPadding(pto::MteGmUbOp op,
                                            PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  Value left = op.getLeftPaddingCount();
  Value right = op.getRightPaddingCount();
  if (!left) {
    left = rewriter.create<arith::ConstantIntOp>(loc, 0,
                                                 mlir::pto::kValue64);
  }
  if (!right) {
    right = rewriter.create<arith::ConstantIntOp>(loc, 0,
                                                  mlir::pto::kValue64);
  }
  bool enabled = static_cast<bool>(op.getPadValue());
  Value dataSelect = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI1Type(), rewriter.getBoolAttr(enabled));
  if (Value padValue = op.getPadValue()) {
    rewriter.create<pto::SetMovPadValOp>(loc, padValue);
  }
  return {left, right, dataSelect, enabled};
}

static DmaLoopPlan configureLoadToL1Loops(
    pto::MteGmL1Op op, ArrayRef<pto::DmaLoopConfig> loops, Value one,
    PatternRewriter &rewriter) {
  ArrayRef<pto::DmaLoopConfig> hardwareLoops =
      loops.take_front(mlir::pto::kValue2);
  ArrayRef<pto::DmaLoopConfig> softwareLoops =
      loops.drop_front(hardwareLoops.size());
  DmaLoopPlan plan{
      SmallVector<pto::DmaLoopConfig>(softwareLoops.rbegin(),
                                      softwareLoops.rend()),
      {}, one};
  bool hasTwoHardwareLoops = hardwareLoops.size() == mlir::pto::kValue2;
  if (hasTwoHardwareLoops) {
    rewriter.create<pto::SetLoop2StrideOutToL1Op>(
        op.getLoc(), packLoopPair(op.getLoc(), hardwareLoops[0].srcStride,
                                  hardwareLoops[0].dstStride, rewriter));
    plan.loop2Count = hardwareLoops[0].count;
    plan.loop1Count = hardwareLoops[1].count;
    rewriter.create<pto::SetLoop1StrideOutToL1Op>(
        op.getLoc(), packLoopPair(op.getLoc(), hardwareLoops[1].srcStride,
                                  hardwareLoops[1].dstStride, rewriter));
  } else if (hardwareLoops.size() == 1) {
    plan.loop1Count = hardwareLoops[0].count;
    rewriter.create<pto::SetLoop1StrideOutToL1Op>(
        op.getLoc(), packLoopPair(op.getLoc(), hardwareLoops[0].srcStride,
                                  hardwareLoops[0].dstStride, rewriter));
  }
  if (plan.loop1Count) {
    Value loopSize = packLoopSize(op.getLoc(), plan.loop2Count,
                                  plan.loop1Count, rewriter);
    rewriter.create<pto::SetLoopSizeOutToL1Op>(op.getLoc(), loopSize);
  }
  return plan;
}

struct ExpandUvldPattern : public OpRewritePattern<pto::UvldOp> {
  using OpRewritePattern<pto::UvldOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::UvldOp op,
                                PatternRewriter &rewriter) const override {
    auto vecType = dyn_cast<pto::VRegType>(op.getResult().getType());
    if (!vecType) {
      return failure();
    }

    Value basePtr = materializeBufferPointer(op.getSource(), rewriter, op.getLoc());
    if (!basePtr) {
      return op.emitOpError(
          "requires a recoverable pointer base for uvld expansion");
    }

    Value loadPtr =
        offsetBufferPointer(basePtr, op.getOffset(), rewriter, op.getLoc());
    auto alignType = pto::AlignType::get(rewriter.getContext());
    Value align =
        rewriter.create<pto::VldasOp>(op.getLoc(), alignType, loadPtr);
    auto load = rewriter.create<pto::VldusOp>(
        op.getLoc(), TypeRange{vecType, alignType},
        ValueRange{loadPtr, align});
    rewriter.replaceOp(op, load.getResult());
    return success();
  }
};

enum class MadRawKind { Ordinary, OrdinaryBias, Mx, MxBias };

static MadRawKind deriveMadRawKind(pto::MadSemanticOpInterface op) {
  if (op.isMadMxFamily()) {
    return op.hasBiasOperand() ? MadRawKind::MxBias : MadRawKind::Mx;
  }
  return op.hasBiasOperand() ? MadRawKind::OrdinaryBias
                             : MadRawKind::Ordinary;
}

static LogicalResult emitMadRawOp(pto::MadSemanticOpInterface op,
                                  MadRawKind kind, Value xt,
                                  PatternRewriter &rewriter) {
  Location loc = op->getLoc();
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();
  Value dst = op.getDst();
  switch (kind) {
  case MadRawKind::Ordinary:
    rewriter.create<pto::MadRawOp>(loc, lhs, rhs, dst, xt);
    return success();
  case MadRawKind::OrdinaryBias:
    rewriter.create<pto::MadBiasRawOp>(loc, lhs, rhs, dst, op.getBiasOrNull(),
                                       xt);
    return success();
  case MadRawKind::Mx:
    rewriter.create<pto::MadMxRawOp>(loc, lhs, rhs, dst, xt);
    return success();
  case MadRawKind::MxBias:
    rewriter.create<pto::MadMxBiasRawOp>(loc, lhs, rhs, dst,
                                         op.getBiasOrNull(), xt);
    return success();
  }
  return failure();
}

static LogicalResult lowerMadSemanticOp(pto::MadSemanticOpInterface op,
                                        PatternRewriter &rewriter) {
  std::optional<pto::MadUnitFlagMode> unitFlagMode;
  if (auto unitFlagModeAttr =
          dyn_cast_or_null<pto::MadUnitFlagModeAttr>(op.getUnitFlagModeAttr())) {
    unitFlagMode = unitFlagModeAttr.getValue();
  }

  std::optional<pto::Tf32Mode> tf32Mode;
  if (op.supportsTf32Mode()) {
    if (auto tf32ModeAttr =
            dyn_cast_or_null<pto::Tf32ModeAttr>(op.getTf32ModeAttr())) {
      tf32Mode = tf32ModeAttr.getValue();
    }
  }

  std::optional<pto::MadSatMode> satMode;
  if (auto satModeAttr =
          dyn_cast_or_null<pto::MadSatModeAttr>(op.getSatModeAttr())) {
    satMode = satModeAttr.getValue();
  }

  bool isHif8 = false;
  if (auto lhsPtr = dyn_cast<pto::PtrType>(op.getLhs().getType())) {
    isHif8 = pto::isPTOHiFloat8Type(lhsPtr.getElementType());
  }

  Location loc = op->getLoc();
  Value ctrlSaved = rewriter.create<pto::GetCtrlOp>(loc).getResult();
  Value ctrlForOp = buildMadSemanticCtrl(
      loc, ctrlSaved, {isHif8, tf32Mode, satMode, op.getNDir()}, rewriter);
  rewriter.create<pto::SetCtrlOp>(loc, ctrlForOp);

  FailureOr<Value> xt = packMadXt(
      loc,
      {op.getM(), op.getN(), op.getK(), unitFlagMode, op.getDisableGemv(),
       op.initializesAccumulatorWithBias(),
       op.initializesAccumulatorWithZero()},
      rewriter);
  if (failed(xt)) {
    return rewriter.notifyMatchFailure(op, "failed to pack mad xt");
  }

  if (failed(emitMadRawOp(op, deriveMadRawKind(op), *xt, rewriter))) {
    return rewriter.notifyMatchFailure(op, "failed to emit mad raw op");
  }

  rewriter.create<pto::SetCtrlOp>(loc, ctrlSaved);
  rewriter.eraseOp(op);
  return success();
}

template <typename SemanticOp>
class ExpandMadSemanticPattern final : public OpRewritePattern<SemanticOp> {
public:
  explicit ExpandMadSemanticPattern(MLIRContext *context)
      : OpRewritePattern<SemanticOp>(context) {}

  LogicalResult matchAndRewrite(SemanticOp op,
                                PatternRewriter &rewriter) const override {
    auto semantic = dyn_cast<pto::MadSemanticOpInterface>(op.getOperation());
    if (!semantic) {
      return failure();
    }
    return lowerMadSemanticOp(semantic, rewriter);
  }
};

struct ExpandDmaLoadPattern : public OpRewritePattern<pto::MteGmUbOp> {
  DmaArch dmaArch;
  explicit ExpandDmaLoadPattern(MLIRContext *ctx, DmaArch arch)
      : OpRewritePattern(ctx), dmaArch(arch) {}

  LogicalResult matchAndRewrite(pto::MteGmUbOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, mlir::pto::kValue64);
    Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, mlir::pto::kValue64);
    SmallVector<pto::DmaLoopConfig> loops =
        collectLoopConfigs(op.getLoopCounts(), op.getLoopSrcStrides(),
                           op.getLoopDstStrides());
    DmaLoopPlan loopPlan =
        configureLoadToUbLoops(op, dmaArch, loops, one, rewriter);
    DmaPaddingConfig padding = configureDmaPadding(op, rewriter);

    Value effectiveNBurst = (dmaArch == DmaArch::A5) ? op.getNBurst() : one;

    buildSoftwareLoopNest(
        rewriter, loc, loopPlan.softwareLoops, {zero, zero},
        [op, &rewriter, loc, zero, effectiveNBurst,
         padding](Value srcOffset, Value dstOffset) mutable {
          Value source = offsetPointerByBytes(op.getSource(), srcOffset, rewriter, loc);
          Value destination =
              offsetPointerByBytes(op.getDestination(), dstOffset, rewriter, loc);
          auto copyOp = rewriter.create<pto::CopyGmToUbufOp>(
              loc, source, destination, zero, effectiveNBurst, op.getLenBurst(),
              padding.left, padding.right, padding.dataSelect,
              op.getL2CacheCtl(),
              op.getNburstSrcStride(), op.getNburstDstStride());
          if (padding.enabled) {
            copyOp->setAttr("has_pad", UnitAttr::get(copyOp->getContext()));
          }
        });
    if (dmaArch == DmaArch::A5 &&
        shouldRestoreDmaLoopSize(loopPlan.loop1Count, loopPlan.loop2Count)) {
      rewriter.create<pto::SetLoopSizeOutToUbOp>(loc, one, one);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct ExpandDmaStorePattern : public OpRewritePattern<pto::MteUbGmOp> {
  DmaArch dmaArch;
  explicit ExpandDmaStorePattern(MLIRContext *ctx, DmaArch arch)
      : OpRewritePattern(ctx), dmaArch(arch) {}

  LogicalResult matchAndRewrite(pto::MteUbGmOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, mlir::pto::kValue64);
    Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, mlir::pto::kValue64);
    SmallVector<pto::DmaLoopConfig> loops =
        collectLoopConfigs(op.getLoopCounts(), op.getLoopSrcStrides(),
                           op.getLoopDstStrides());
    DmaLoopPlan loopPlan =
        configureStoreFromUbLoops(op, dmaArch, loops, one, rewriter);

    Value effectiveNBurst = (dmaArch == DmaArch::A5) ? op.getNBurst() : one;

    buildSoftwareLoopNest(
        rewriter, loc, loopPlan.softwareLoops, {zero, zero},
        [op, &rewriter, loc, zero, effectiveNBurst](Value srcOffset,
                                                    Value dstOffset) mutable {
          Value source = offsetPointerByBytes(op.getSource(), srcOffset, rewriter, loc);
          Value destination =
              offsetPointerByBytes(op.getDestination(), dstOffset, rewriter, loc);
          Value l2CacheCtl = op.getL2CacheCtl() ? op.getL2CacheCtl() : zero;
          rewriter.create<pto::CopyUbufToGmOp>(
              loc, source, destination, zero, effectiveNBurst, op.getLenBurst(),
              l2CacheCtl, op.getNburstDstStride(), op.getNburstSrcStride());
        });
    if (dmaArch == DmaArch::A5 &&
        shouldRestoreDmaLoopSize(loopPlan.loop1Count, loopPlan.loop2Count)) {
      rewriter.create<pto::SetLoopSizeUbToOutOp>(loc, one, one);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct ExpandMteUbUbPattern : public OpRewritePattern<pto::MteUbUbOp> {
  using OpRewritePattern<pto::MteUbUbOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::MteUbUbOp op,
                                PatternRewriter &rewriter) const override {
    Value zero = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0, mlir::pto::kValue64);
    rewriter.replaceOpWithNewOp<pto::CopyUbufToUbufOp>(
        op, op.getSource(), op.getDestination(), zero, op.getNBurst(),
        op.getLenBurst(), op.getSrcStride(), op.getDstStride());
    return success();
  }
};

struct ExpandMteUbL1Pattern : public OpRewritePattern<pto::MteUbL1Op> {
  using OpRewritePattern<pto::MteUbL1Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::MteUbL1Op op,
                                PatternRewriter &rewriter) const override {
    Value zero = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0, mlir::pto::kValue64);
    rewriter.replaceOpWithNewOp<pto::CopyUbufToCbufOp>(
        op, op.getSource(), op.getDestination(), zero, op.getNBurst(),
        op.getLenBurst(), op.getSrcStride(), op.getDstStride());
    return success();
  }
};

struct ExpandCubeLoadPattern : public OpRewritePattern<pto::MteGmL1Op> {
  using OpRewritePattern<pto::MteGmL1Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::MteGmL1Op op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, mlir::pto::kValue64);
    Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, mlir::pto::kValue64);
    SmallVector<pto::DmaLoopConfig> loops =
        collectLoopConfigs(op.getLoopCounts(), op.getLoopSrcStrides(),
                           op.getLoopDstStrides());
    DmaLoopPlan loopPlan = configureLoadToL1Loops(op, loops, one, rewriter);
    buildSoftwareLoopNest(
        rewriter, loc, loopPlan.softwareLoops, {zero, zero},
        [op, &rewriter, loc](Value srcOffset, Value dstOffset) mutable {
          Value source =
              offsetPointerByBytes(op.getSource(), srcOffset, rewriter, loc);
          Value destination = offsetPointerByBytes(op.getDestination(), dstOffset,
                                                   rewriter, loc);
          rewriter.create<pto::CopyGmToCbufOp>(
              loc, source, destination, op.getNBurst(), op.getLenBurst(),
              op.getNburstSrcStride(), op.getNburstDstStride());
        });
    bool restoreLoopSize =
        loopPlan.loop1Count &&
        (!isKnownOne(loopPlan.loop1Count) ||
         !isKnownOne(loopPlan.loop2Count));
    if (restoreLoopSize) {
      rewriter.create<pto::SetLoopSizeOutToL1Op>(
          loc, packLoopSize(loc, one, one, rewriter));
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct ExpandRawFillL1Pattern : public OpRewritePattern<pto::RawFillL1Op> {
  using OpRewritePattern<pto::RawFillL1Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::RawFillL1Op op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    const unsigned viewWidth = op.getFillWordBits() == 16 ? 16 : 32;

    Value basePtr = materializeBufferPointer(op.getDst(), rewriter, loc);
    if (!basePtr || !isa<pto::PtrType>(basePtr.getType())) {
      return rewriter.notifyMatchFailure(op, "failed to materialize dst ptr");
    }

    Value targetPtr =
        offsetPointerByBytes(basePtr, op.getByteOffset(), rewriter, loc);
    if (!targetPtr) {
      return rewriter.notifyMatchFailure(op, "failed to apply byte offset");
    }

    auto dstPtrType = cast<pto::PtrType>(targetPtr.getType());
    Type viewElement = IntegerType::get(rewriter.getContext(), viewWidth,
                                        IntegerType::Unsigned);
    auto viewType = pto::PtrType::get(rewriter.getContext(), viewElement,
                                      dstPtrType.getMemorySpace());
    Value viewPtr = targetPtr;
    const bool needsViewCast = targetPtr.getType() != viewType;
    if (needsViewCast) {
      viewPtr = rewriter.create<pto::CastPtrOp>(loc, viewType, targetPtr);
    }

    rewriter.create<pto::CreateCbufMatrixOp>(
        loc, viewPtr, op.getRawValue(), op.getRepeatTimes(),
        op.getBlockNum_32b(), op.getDstGap_32b(),
        static_cast<uint64_t>(op.getFillWordBits()));
    rewriter.eraseOp(op);
    return success();
  }
};

struct ExpandCubeStorePattern : public OpRewritePattern<pto::MteL1UbOp> {
  using OpRewritePattern<pto::MteL1UbOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::MteL1UbOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, mlir::pto::kValue64);
    SmallVector<pto::DmaLoopConfig> loops =
        collectLoopConfigs(op.getLoopCounts(), op.getLoopSrcStrides(),
                           op.getLoopDstStrides());
    SmallVector<pto::DmaLoopConfig> swLoopNestOrder(loops.rbegin(),
                                                    loops.rend());
    buildSoftwareLoopNest(
        rewriter, loc, swLoopNestOrder, {zero, zero},
        [op, &rewriter, loc, zero](Value srcOffset, Value dstOffset) mutable {
          Value source =
              offsetPointerByBytes(op.getSource(), srcOffset, rewriter, loc);
          Value destination =
              offsetPointerByBytes(op.getDestination(), dstOffset, rewriter, loc);
          rewriter.create<pto::CopyCbufToUbufOp>(
              loc, source, destination, zero, op.getNBurst(), op.getLenBurst(),
              op.getNburstSrcStride(), op.getNburstDstStride());
        });
    rewriter.eraseOp(op);
    return success();
  }
};

struct ExpandBiasLoadPattern : public OpRewritePattern<pto::MteL1BtOp> {
  using OpRewritePattern<pto::MteL1BtOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::MteL1BtOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value source = materializeBufferPointer(op.getSource(), rewriter, loc);
    Value destination =
        materializeBufferPointer(op.getDestination(), rewriter, loc);
    auto sourceType = dyn_cast_or_null<pto::PtrType>(source.getType());
    if (!sourceType) {
      return rewriter.notifyMatchFailure(op, "expected pointer-like source");
    }
    if (!destination) {
      return rewriter.notifyMatchFailure(op, "expected pointer-like destination");
    }

    Value convControl = rewriter.create<arith::ConstantIntOp>(
        loc, sourceType.getElementType().isF16() ? 1 : 0, 1);
    rewriter.replaceOpWithNewOp<pto::CopyCbufToBtOp>(
        op, source, destination, convControl, op.getNBurst(),
        op.getLenBurst(), op.getNburstSrcGap(), op.getNburstDstGap());
    return success();
  }
};

struct ExpandFpLoadPattern : public OpRewritePattern<pto::MteL1FbOp> {
  using OpRewritePattern<pto::MteL1FbOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::MteL1FbOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value source = materializeBufferPointer(op.getSource(), rewriter, loc);
    Value destination =
        materializeBufferPointer(op.getDestination(), rewriter, loc);
    if (!source || !destination) {
      return rewriter.notifyMatchFailure(op, "expected pointer-like operands");
    }

    rewriter.replaceOpWithNewOp<pto::CopyCbufToFbufOp>(
        op, source, destination, op.getNBurst(),
        op.getLenBurst(), op.getNburstSrcGap(), op.getNburstDstGap());
    return success();
  }
};

struct ExpandCubeLoadFracPattern : public OpRewritePattern<pto::MteGmL1FracOp> {
  using OpRewritePattern<pto::MteGmL1FracOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::MteGmL1FracOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, mlir::pto::kValue64);
    Value mte2NzPara = packMte2NzPara(
        loc,
        {op.getGroupCount(), op.getDstLoop2Stride(), op.getDstLoop3Stride(),
         op.getDstLoop4Stride()},
        rewriter);
    rewriter.create<pto::SetMte2NzParaOp>(loc, mte2NzPara);

    Value srcOuterStride = op.getSrcOuterStride() ? op.getSrcOuterStride() : zero;
    Value source = materializeBufferPointer(op.getSource(), rewriter, loc);
    Value destination =
        materializeBufferPointer(op.getDestination(), rewriter, loc);
    switch (op.getMode()) {
    case pto::CubeLoadFracMode::Nd2nz:
      rewriter.create<pto::CopyGmToCbufMultiNd2NzOp>(
          loc, source, destination, zero, op.getSrcInnerStride(),
          op.getL2CacheCtrl(), op.getNValue(), op.getDValue(), srcOuterStride,
          op.getSmallc0En());
      break;
    case pto::CubeLoadFracMode::Dn2nz:
      rewriter.create<pto::CopyGmToCbufMultiDn2NzOp>(
          loc, source, destination, zero, op.getSrcInnerStride(),
          op.getL2CacheCtrl(), op.getNValue(), op.getDValue(), srcOuterStride,
          op.getSmallc0En());
      break;
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct ExpandLeftLoadPattern : public OpRewritePattern<pto::MteL1L0aOp> {
  using OpRewritePattern<pto::MteL1L0aOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::MteL1L0aOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value source = materializeBufferPointer(op.getSource(), rewriter, loc);
    Value destination =
        materializeBufferPointer(op.getDestination(), rewriter, loc);
    auto sourceType = dyn_cast_or_null<pto::PtrType>(source.getType());
    if (!sourceType) {
      return rewriter.notifyMatchFailure(op, "expected typed L1 source");
    }
    Type elementType = sourceType.getElementType();
    if (!destination) {
      return rewriter.notifyMatchFailure(op, "expected pointer-like destination");
    }
    FailureOr<LoadCbufToCbControl> control =
        [op, loc, elementType,
         &rewriter]() mutable -> FailureOr<LoadCbufToCbControl> {
      if (op.getMStart()) {
        return LoadCbufToCbControl{op.getMStart(), op.getKStart(),
                                   op.getMStep(), op.getKStep(),
                                   op.getSrcStride(), op.getDstStride()};
      }
      return deriveLoadCbufControl(
          {loc, op.getM(), op.getK(), elementType, op.getStartRow(),
           op.getStartCol(), op.getTranspose(), rewriter});
    }();
    if (failed(control)) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to derive load_cbuf_to_ca control");
    }
    if (pto::isPTOFloat4PackedType(elementType)) {
      rewriter.create<pto::LoadCbufToCaS4Op>(
          loc, source, destination, control->mStart,
          control->kStart, control->mStep, control->kStep,
          control->srcStride, control->dstStride,
          rewriter.create<arith::ConstantIntOp>(loc, op.getTranspose(),
                                                mlir::pto::kValue64));
    } else {
      auto load = rewriter.create<pto::LoadCbufToCaOp>(
          loc, source, destination, control->mStart,
          control->kStart, control->mStep, control->kStep,
          control->srcStride, control->dstStride);
      load->setAttr("transpose", rewriter.getBoolAttr(op.getTranspose()));
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct ExpandRightLoadPattern : public OpRewritePattern<pto::MteL1L0bOp> {
  using OpRewritePattern<pto::MteL1L0bOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::MteL1L0bOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value source = materializeBufferPointer(op.getSource(), rewriter, loc);
    Value destination =
        materializeBufferPointer(op.getDestination(), rewriter, loc);
    auto sourceType = dyn_cast_or_null<pto::PtrType>(source.getType());
    if (!sourceType) {
      return rewriter.notifyMatchFailure(op, "expected typed L1 source");
    }
    Type elementType = sourceType.getElementType();
    if (!destination) {
      return rewriter.notifyMatchFailure(op, "expected pointer-like destination");
    }
    FailureOr<LoadCbufToCbControl> control =
        [op, loc, elementType,
         &rewriter]() mutable -> FailureOr<LoadCbufToCbControl> {
      if (op.getMStart()) {
        return LoadCbufToCbControl{op.getMStart(), op.getKStart(),
                                   op.getMStep(), op.getKStep(),
                                   op.getSrcStride(), op.getDstStride()};
      }
      return deriveLoadCbufControl(
          {loc, op.getN(), op.getK(), elementType, op.getStartRow(),
           op.getStartCol(), op.getTranspose(), rewriter});
    }();
    if (failed(control)) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to derive load_cbuf_to_cb control");
    }
    if (pto::isPTOFloat4PackedType(elementType)) {
      rewriter.create<pto::LoadCbufToCbS4Op>(
          loc, source, destination, control->mStart,
          control->kStart, control->mStep, control->kStep,
          control->srcStride, control->dstStride,
          rewriter.create<arith::ConstantIntOp>(loc, op.getTranspose(),
                                                mlir::pto::kValue64));
    } else {
      auto load = rewriter.create<pto::LoadCbufToCbOp>(
          loc, source, destination, control->mStart,
          control->kStart, control->mStep, control->kStep,
          control->srcStride, control->dstStride);
      load->setAttr("transpose", rewriter.getBoolAttr(op.getTranspose()));
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct ExpandLeftLoadMxPattern : public OpRewritePattern<pto::MteL1L0aMxOp> {
  using OpRewritePattern<pto::MteL1L0aMxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::MteL1L0aMxOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value source = materializeBufferPointer(op.getSource(), rewriter, loc);
    Value destination =
        materializeBufferPointer(op.getDestination(), rewriter, loc);
    auto sourceType = dyn_cast_or_null<pto::PtrType>(source.getType());
    if (!sourceType) {
      return rewriter.notifyMatchFailure(op, "expected typed L1 source");
    }
    if (!destination) {
      return rewriter.notifyMatchFailure(op, "expected pointer-like destination");
    }
    destination = deriveMxScaleDestination(destination, rewriter, loc);
    if (!destination) {
      return rewriter.notifyMatchFailure(
          op, "failed to derive MX scale destination pointer");
    }

    LoadCbufToMxControl control;
    if (op.getXStart()) {
      if (!op.getYStart() || !op.getXStep() || !op.getYStep() ||
          !op.getSrcStride() || !op.getDstStride()) {
        return rewriter.notifyMatchFailure(op,
                                           "expected complete full MX operands");
      }
      control = {op.getXStart(), op.getYStart(), op.getXStep(), op.getYStep(),
                 op.getSrcStride(), op.getDstStride()};
    } else {
      if (!op.getM() || !op.getK() || !op.getStartRow() || !op.getStartCol()) {
        return rewriter.notifyMatchFailure(
            op, "expected complete shape-derived MX operands");
      }
      FailureOr<LoadCbufToMxControl> derived = deriveLoadCbufMxControl(
          {loc, op.getM(), op.getK(), sourceType.getElementType(),
           op.getStartRow(), op.getStartCol(), CbufMxSide::Left, rewriter});
      if (failed(derived)) {
        return rewriter.notifyMatchFailure(
            op, "failed to derive load_cbuf_to_ca_mx control");
      }
      control = *derived;
    }

    rewriter.create<pto::LoadCbufToCaMxOp>(
        loc, source, destination, control.xStartPosition,
        control.yStartPosition, control.xStep, control.yStep,
        control.srcStride, control.dstStride);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ExpandRightLoadMxPattern : public OpRewritePattern<pto::MteL1L0bMxOp> {
  using OpRewritePattern<pto::MteL1L0bMxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::MteL1L0bMxOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value source = materializeBufferPointer(op.getSource(), rewriter, loc);
    Value destination =
        materializeBufferPointer(op.getDestination(), rewriter, loc);
    auto sourceType = dyn_cast_or_null<pto::PtrType>(source.getType());
    if (!sourceType) {
      return rewriter.notifyMatchFailure(op, "expected typed L1 source");
    }
    if (!destination) {
      return rewriter.notifyMatchFailure(op, "expected pointer-like destination");
    }
    destination = deriveMxScaleDestination(destination, rewriter, loc);
    if (!destination) {
      return rewriter.notifyMatchFailure(
          op, "failed to derive MX scale destination pointer");
    }

    LoadCbufToMxControl control;
    if (op.getXStart()) {
      if (!op.getYStart() || !op.getXStep() || !op.getYStep() ||
          !op.getSrcStride() || !op.getDstStride()) {
        return rewriter.notifyMatchFailure(op,
                                           "expected complete full MX operands");
      }
      control = {op.getXStart(), op.getYStart(), op.getXStep(), op.getYStep(),
                 op.getSrcStride(), op.getDstStride()};
    } else {
      if (!op.getK() || !op.getN() || !op.getStartRow() || !op.getStartCol()) {
        return rewriter.notifyMatchFailure(
            op, "expected complete shape-derived MX operands");
      }
      FailureOr<LoadCbufToMxControl> derived = deriveLoadCbufMxControl(
          {loc, op.getN(), op.getK(), sourceType.getElementType(),
           op.getStartRow(), op.getStartCol(), CbufMxSide::Right, rewriter});
      if (failed(derived)) {
        return rewriter.notifyMatchFailure(
            op, "failed to derive load_cbuf_to_cb_mx control");
      }
      control = *derived;
    }

    rewriter.create<pto::LoadCbufToCbMxOp>(
        loc, source, destination, control.xStartPosition,
        control.yStartPosition, control.xStep, control.yStep,
        control.srcStride, control.dstStride);
    rewriter.eraseOp(op);
    return success();
  }
};

struct AccStorePointers {
  Value source;
  Value destination;
};

template <typename StoreOp>
static AccStorePointers materializeAccStorePointers(
    StoreOp op, PatternRewriter &rewriter) {
  return {materializeBufferPointer(op.getSource(), rewriter, op.getLoc()),
          materializeBufferPointer(op.getDestination(), rewriter,
                                   op.getLoc())};
}

template <typename StoreOp>
static void configureAccStorePreOps(StoreOp op, PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  AccStorePreOpConfig config{
      op.getPreQuant(), op.getPreQuantMode(), op.getPreRelu(),
      op.getPreReluMode(), op.getClipValue(),
      getBufferElementType(op.getDestination().getType())};
  configureAccStoreScalarPreOps(loc, config, rewriter);
  Value fpc = buildAccStoreFpcValue(loc, config, rewriter);
  if (fpc) {
    rewriter.create<pto::SetFpcOp>(loc, fpc);
  }
}

template <typename StoreOp>
static pto::DmaLoopConfig getAccStoreHardwareLoop(StoreOp op, Value zero,
                                                   Value one) {
  if (Value count = op.getLoop3Count()) {
    return {count, op.getLoop3SrcStride(), op.getLoop3DstStride()};
  }
  return {one, zero, zero};
}

template <typename StoreOp>
static AccStoreModeConfig getAccStoreModeConfig(StoreOp op, Value zero,
                                                 Value one) {
  AccStoreModeConfig config{zero, zero, zero, zero};
  std::optional<pto::AccStoreMode> mode = op.getMode();
  if (!mode) {
    config.nz2nd = one;
    return config;
  }
  switch (*mode) {
  case pto::AccStoreMode::Nz2nd:
    config.nz2nd = one;
    break;
  case pto::AccStoreMode::Nz2dn:
    config.nz2dn = one;
    config.channelLoop0Stride =
        op.getLoop0SrcStride() ? op.getLoop0SrcStride() : one;
    break;
  case pto::AccStoreMode::Nz2nz:
    config.channelSplit = op.getSplit() ? op.getSplit() : zero;
    break;
  }
  return config;
}

static void emitAccStoreLoopConfig(Location loc, pto::DmaLoopConfig loop,
                                   Value channelLoop0Stride,
                                   PatternRewriter &rewriter) {
  Value loopConfig = packLoop3Config(loc, loop.count, loop.srcStride,
                                     loop.dstStride, rewriter);
  Value channelConfig =
      packChannelConfig(loc, channelLoop0Stride, rewriter);
  rewriter.create<pto::SetLoop3ParaOp>(
      loc, extractConfigLow40(loc, loopConfig, rewriter),
      extractConfigHigh24(loc, loopConfig, rewriter));
  rewriter.create<pto::SetChannelParaOp>(
      loc, extractConfigLow40(loc, channelConfig, rewriter),
      extractConfigHigh24(loc, channelConfig, rewriter));
}

template <typename StoreOp>
static AccStorePackedFields getAccStorePackedFields(
    StoreOp op, PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  auto encode = [loc, &rewriter](auto mode) {
    return buildAccStoreOptionalEnumValue(
        loc,
        mode ? std::optional<uint32_t>(static_cast<uint32_t>(*mode))
             : std::nullopt,
        rewriter);
  };
  return {getI64Constant(loc, rewriter, op.getClipValue() ? 1 : 0),
          encode(op.getUnitFlag()), encode(op.getPreQuantMode()),
          encode(op.getPreReluMode())};
}

static void restoreAccStoreCtrl(Location loc, Value originalCtrl,
                                PatternRewriter &rewriter) {
  if (originalCtrl) {
    rewriter.create<pto::SetCtrlOp>(loc, originalCtrl);
  }
}

struct ExpandAccStorePattern : public OpRewritePattern<pto::MteL0cL1Op> {
  using OpRewritePattern<pto::MteL0cL1Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::MteL0cL1Op op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    AccStorePointers pointers = materializeAccStorePointers(op, rewriter);
    if (!pointers.source || !pointers.destination) {
      return rewriter.notifyMatchFailure(op, "expected pointer-like operands");
    }
    Value zero = getI64Constant(loc, rewriter, 0);
    Value one = getI64Constant(loc, rewriter, 1);
    configureAccStorePreOps(op, rewriter);
    Value originalCtrl = configureAccStoreCtrl(
        loc, {false, std::nullopt, std::nullopt, op.getSatMode()}, rewriter);
    pto::DmaLoopConfig hardwareLoop =
        getAccStoreHardwareLoop(op, zero, one);
    AccStoreModeConfig mode = getAccStoreModeConfig(op, zero, one);
    emitAccStoreLoopConfig(loc, hardwareLoop, mode.channelLoop0Stride,
                           rewriter);
    AccStorePackedFields fields = getAccStorePackedFields(op, rewriter);
    Value xm = packCopyMatrixCcToGmXm(
        loc, {zero, op.getN(), op.getM(), op.getDstStride()}, rewriter);
    Value xt = packCopyMatrixCcToGmXt(
        loc, {op.getSrcStride(), zero, fields, mode}, rewriter);
    rewriter.create<pto::CopyMatrixCcToCbufOp>(
        loc, pointers.source, pointers.destination, xm, xt);
    restoreAccStoreCtrl(loc, originalCtrl, rewriter);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ExpandAccStoreGmPattern : public OpRewritePattern<pto::MteL0cGmOp> {
  using OpRewritePattern<pto::MteL0cGmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::MteL0cGmOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    AccStorePointers pointers = materializeAccStorePointers(op, rewriter);
    if (!pointers.source || !pointers.destination) {
      return rewriter.notifyMatchFailure(op, "expected pointer-like operands");
    }
    Value zero = getI64Constant(loc, rewriter, 0);
    Value one = getI64Constant(loc, rewriter, 1);
    configureAccStorePreOps(op, rewriter);
    Value originalCtrl = configureAccStoreCtrl(
        loc, {true, op.getAtomicType(), op.getAtomicOp(), op.getSatMode()},
        rewriter);
    pto::DmaLoopConfig hardwareLoop =
        getAccStoreHardwareLoop(op, zero, one);
    AccStoreModeConfig mode = getAccStoreModeConfig(op, zero, one);
    emitAccStoreLoopConfig(loc, hardwareLoop, mode.channelLoop0Stride,
                           rewriter);
    AccStorePackedFields fields = getAccStorePackedFields(op, rewriter);
    Value xm = packCopyMatrixCcToGmXm(
        loc, {op.getSid(), op.getN(), op.getM(), op.getDstStride()}, rewriter);
    Value xt = packCopyMatrixCcToGmXt(
        loc, {op.getSrcStride(), op.getL2CacheCtrl(), fields, mode}, rewriter);
    rewriter.create<pto::CopyMatrixCcToGmOp>(
        loc, pointers.source, pointers.destination, xm, xt);
    restoreAccStoreCtrl(loc, originalCtrl, rewriter);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ExpandAccStoreUbPattern : public OpRewritePattern<pto::MteL0cUbOp> {
  using OpRewritePattern<pto::MteL0cUbOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::MteL0cUbOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    AccStorePointers pointers = materializeAccStorePointers(op, rewriter);
    if (!pointers.source || !pointers.destination) {
      return rewriter.notifyMatchFailure(op, "expected pointer-like operands");
    }
    Value zero = getI64Constant(loc, rewriter, 0);
    Value one = getI64Constant(loc, rewriter, 1);
    configureAccStorePreOps(op, rewriter);
    Value originalCtrl = configureAccStoreCtrl(
        loc, {false, std::nullopt, std::nullopt, op.getSatMode()}, rewriter);
    pto::DmaLoopConfig hardwareLoop =
        getAccStoreHardwareLoop(op, zero, one);
    AccStoreModeConfig mode = getAccStoreModeConfig(op, zero, one);
    emitAccStoreLoopConfig(loc, hardwareLoop, mode.channelLoop0Stride,
                           rewriter);
    AccStorePackedFields fields = getAccStorePackedFields(op, rewriter);

    Value dualDstMode =
        getI64Constant(loc, rewriter, static_cast<int64_t>(op.getDstMode()));
    Value subBlockId = op.getSubBlockid() ? op.getSubBlockid() : zero;
    Value config0 = packCopyMatrixCcToGmXm(
        loc, {zero, op.getN(), op.getM(), op.getDstStride()}, rewriter);
    Value config1 = packCopyMatrixCcToUbConfig1(
        loc, {op.getSrcStride(), dualDstMode, subBlockId, fields, mode},
        rewriter);
    rewriter.create<pto::CopyMatrixCcToUbOp>(
        loc, pointers.source, pointers.destination, config0, config1);
    restoreAccStoreCtrl(loc, originalCtrl, rewriter);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ExpandSimtLaunchPattern : public OpRewritePattern<pto::SimtLaunchOp> {
  using OpRewritePattern<pto::SimtLaunchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::SimtLaunchOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    rewriter.create<pto::StoreVfSimtInfoOp>(loc, op.getDimZ(), op.getDimY(),
                                            op.getDimX());
    rewriter.create<func::CallOp>(loc, op.getCalleeAttr(), TypeRange{},
                                  op.getArgs());
    rewriter.eraseOp(op);
    return success();
  }
};

struct AtomicCtrlUpdate {
  uint64_t mask;
  uint64_t value;
};

template <typename AtomicConfigOp> static AtomicCtrlUpdate getAtomicCtrlUpdate();

// CCE set_atomic_* configures CTRL[10:6]. Dtype occupies [8:6] and the
// reduction operation occupies [10:9]. This matches the structured L0C-to-GM
// FIXP atomic CTRL encoding used by configureAccStoreCtrl above.
template <> AtomicCtrlUpdate getAtomicCtrlUpdate<pto::SetAtomicAddOp>() { return {0x3ULL << 9, 0x0ULL << 9}; }
template <> AtomicCtrlUpdate getAtomicCtrlUpdate<pto::SetAtomicMaxOp>() { return {0x3ULL << 9, 0x1ULL << 9}; }
template <> AtomicCtrlUpdate getAtomicCtrlUpdate<pto::SetAtomicMinOp>() { return {0x3ULL << 9, 0x2ULL << 9}; }
template <> AtomicCtrlUpdate getAtomicCtrlUpdate<pto::SetAtomicNoneOp>() { return {0x7ULL << 6, 0}; }
template <> AtomicCtrlUpdate getAtomicCtrlUpdate<pto::SetAtomicF32Op>() { return {0x7ULL << 6, 0x1ULL << 6}; }
template <> AtomicCtrlUpdate getAtomicCtrlUpdate<pto::SetAtomicF16Op>() { return {0x7ULL << 6, 0x2ULL << 6}; }
template <> AtomicCtrlUpdate getAtomicCtrlUpdate<pto::SetAtomicS16Op>() { return {0x7ULL << 6, 0x3ULL << 6}; }
template <> AtomicCtrlUpdate getAtomicCtrlUpdate<pto::SetAtomicS32Op>() { return {0x7ULL << 6, 0x4ULL << 6}; }
template <> AtomicCtrlUpdate getAtomicCtrlUpdate<pto::SetAtomicS8Op>() { return {0x7ULL << 6, 0x5ULL << 6}; }
template <> AtomicCtrlUpdate getAtomicCtrlUpdate<pto::SetAtomicBF16Op>() { return {0x7ULL << 6, 0x6ULL << 6}; }

template <typename AtomicConfigOp>
struct ExpandAtomicConfigPattern
    : public OpRewritePattern<AtomicConfigOp> {
  using OpRewritePattern<AtomicConfigOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AtomicConfigOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    AtomicCtrlUpdate update = getAtomicCtrlUpdate<AtomicConfigOp>();
    Value ctrl = rewriter.create<pto::GetCtrlOp>(loc);
    Value clearMask = getI64Constant(loc, rewriter, ~update.mask);
    Value value = getI64Constant(loc, rewriter, update.value);
    Value updated = rewriter.create<arith::AndIOp>(loc, ctrl, clearMask);
    updated = rewriter.create<arith::OrIOp>(loc, updated, value);
    rewriter.create<pto::SetCtrlOp>(loc, updated);
    rewriter.eraseOp(op);
    return success();
  }
};

struct VPTOExpandWrapperOpsPass
    : public pto::impl::VPTOExpandWrapperOpsBase<VPTOExpandWrapperOpsPass> {
  using pto::impl::VPTOExpandWrapperOpsBase<
      VPTOExpandWrapperOpsPass>::VPTOExpandWrapperOpsBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, pto::PTODialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal()) {
      return;
    }

    DmaArch dmaArch = getDmaArch(func->getParentOfType<ModuleOp>());

    RewritePatternSet patterns(&getContext());
    patterns.add(std::make_unique<ExpandDmaLoadPattern>(&getContext(), dmaArch));
    patterns.add(std::make_unique<ExpandDmaStorePattern>(&getContext(), dmaArch));
    patterns.add<ExpandUvldPattern,
                 ExpandMteUbUbPattern, ExpandMteUbL1Pattern, ExpandCubeLoadPattern,
                 ExpandCubeStorePattern, ExpandBiasLoadPattern,
                 ExpandFpLoadPattern,
                 ExpandCubeLoadFracPattern, ExpandLeftLoadPattern,
                 ExpandRightLoadPattern, ExpandLeftLoadMxPattern,
                 ExpandRightLoadMxPattern, ExpandAccStorePattern,
                 ExpandAccStoreGmPattern,
                 ExpandAccStoreUbPattern,
                 ExpandRawFillL1Pattern,
                 ExpandSimtLaunchPattern,
                 ExpandAtomicConfigPattern<pto::SetAtomicAddOp>,
                 ExpandAtomicConfigPattern<pto::SetAtomicMaxOp>,
                 ExpandAtomicConfigPattern<pto::SetAtomicMinOp>,
                 ExpandAtomicConfigPattern<pto::SetAtomicNoneOp>,
                 ExpandAtomicConfigPattern<pto::SetAtomicF32Op>,
                 ExpandAtomicConfigPattern<pto::SetAtomicF16Op>,
                 ExpandAtomicConfigPattern<pto::SetAtomicBF16Op>,
                 ExpandAtomicConfigPattern<pto::SetAtomicS32Op>,
                 ExpandAtomicConfigPattern<pto::SetAtomicS16Op>,
                 ExpandAtomicConfigPattern<pto::SetAtomicS8Op>,
                 ExpandMadSemanticPattern<pto::MadOp>,
                 ExpandMadSemanticPattern<pto::MadAccOp>,
                 ExpandMadSemanticPattern<pto::MadBiasOp>,
                 ExpandMadSemanticPattern<pto::MadMxOp>,
                 ExpandMadSemanticPattern<pto::MadMxAccOp>,
                 ExpandMadSemanticPattern<pto::MadMxBiasOp>>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVPTOExpandWrapperOpsPass() {
  return std::make_unique<VPTOExpandWrapperOpsPass>();
}

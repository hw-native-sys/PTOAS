// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

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

static DmaArch getDmaArch(ModuleOp mod) {
  if (!mod)
    return DmaArch::A2A3;
  auto arch = mod->getAttrOfType<StringAttr>("pto.target_arch");
  if (arch && arch.getValue() == "a5")
    return DmaArch::A5;
  return DmaArch::A2A3;
}

static pto::AddressSpaceAttr getPointerMemorySpace(Attribute memorySpace,
                                                   MLIRContext *ctx) {
  if (auto addrSpace = dyn_cast_or_null<pto::AddressSpaceAttr>(memorySpace))
    return addrSpace;
  if (auto intAttr = dyn_cast_or_null<IntegerAttr>(memorySpace))
    return pto::AddressSpaceAttr::get(
        ctx, static_cast<pto::AddressSpace>(intAttr.getInt()));
  return pto::AddressSpaceAttr::get(ctx, pto::AddressSpace::GM);
}

static Value materializeBufferPointer(Value value, PatternRewriter &rewriter,
                                      Location loc) {
  if (!value)
    return {};

  if (isa<pto::PtrType>(value.getType()))
    return value;

  auto memrefType = dyn_cast<MemRefType>(value.getType());
  if (!memrefType)
    return {};

  auto ptrType =
      pto::PtrType::get(rewriter.getContext(), memrefType.getElementType(),
                        getPointerMemorySpace(memrefType.getMemorySpace(),
                                              rewriter.getContext()));
  return rewriter.create<pto::CastPtrOp>(loc, ptrType, value).getResult();
}

static Type getBufferElementType(Type type) {
  if (auto ptrType = dyn_cast<pto::PtrType>(type))
    return ptrType.getElementType();
  if (auto memrefType = dyn_cast<BaseMemRefType>(type))
    return memrefType.getElementType();
  return {};
}

static Value offsetBufferPointer(Value basePtr, Type elementType,
                                 Value elementOffset,
                                 PatternRewriter &rewriter, Location loc) {
  if (!basePtr)
    return {};

  Value offsetIndex = elementOffset;
  if (!offsetIndex.getType().isIndex())
    offsetIndex = rewriter.create<arith::IndexCastUIOp>(loc,
                                                        rewriter.getIndexType(),
                                                        elementOffset);
  return rewriter.create<pto::AddPtrOp>(loc, basePtr.getType(), basePtr,
                                        offsetIndex);
}

static bool isKnownOne(Value value) {
  APInt intValue;
  return value && matchPattern(value, m_ConstantInt(&intValue)) &&
         intValue.isOne();
}

static bool shouldRestoreDmaLoopSize(Value loop1Count, Value loop2Count) {
  if (!loop1Count)
    return false;
  return !isKnownOne(loop1Count) || !isKnownOne(loop2Count);
}

static SmallVector<pto::DmaLoopConfig> collectLoopConfigs(ValueRange counts,
                                                          ValueRange srcStrides,
                                                          ValueRange dstStrides) {
  SmallVector<pto::DmaLoopConfig> loops;
  loops.reserve(counts.size());
  for (auto [count, srcStride, dstStride] :
       llvm::zip(counts, srcStrides, dstStrides))
    loops.push_back({count, srcStride, dstStride});
  return loops;
}

static Value offsetPointerByBytes(Value basePtr, Value byteOffset,
                                  PatternRewriter &rewriter, Location loc) {
  if (!basePtr)
    return {};

  Value basePtrValue = materializeBufferPointer(basePtr, rewriter, loc);
  auto ptrType = dyn_cast_or_null<pto::PtrType>(basePtrValue.getType());
  if (!ptrType)
    return {};

  APInt constOffset;
  if (matchPattern(byteOffset, m_ConstantInt(&constOffset)) && constOffset.isZero())
    return basePtrValue;

  auto bytePtrType =
      pto::PtrType::get(rewriter.getContext(), rewriter.getI8Type(),
                        ptrType.getMemorySpace());
  Value bytePtr =
      rewriter.create<pto::CastPtrOp>(loc, bytePtrType, basePtrValue);
  Value offsetIndex = byteOffset;
  if (!offsetIndex.getType().isIndex())
    offsetIndex =
        rewriter.create<arith::IndexCastUIOp>(loc, rewriter.getIndexType(),
                                              offsetIndex);
  Value advanced =
      rewriter.create<pto::AddPtrOp>(loc, bytePtrType, bytePtr, offsetIndex);
  return rewriter.create<pto::CastPtrOp>(loc, ptrType, advanced);
}

[[maybe_unused]] static Value materializeFpcValue(Value fpc,
                                                  PatternRewriter &rewriter,
                                                  Location loc) {
  if (!fpc)
    return {};
  if (fpc.getType().isInteger(64))
    return fpc;
  if (isa<pto::PtrType>(fpc.getType()))
    return rewriter.create<pto::CastPtrOp>(loc, rewriter.getI64Type(), fpc);
  return {};
}

static Value materializeI64Value(Value value, PatternRewriter &rewriter,
                                 Location loc) {
  if (!value)
    return {};
  if (value.getType().isInteger(64))
    return value;
  if (auto intType = dyn_cast<IntegerType>(value.getType()))
    return rewriter.create<arith::ExtUIOp>(loc, rewriter.getI64Type(), value);
  if (isa<pto::PtrType>(value.getType()))
    return rewriter.create<pto::CastPtrOp>(loc, rewriter.getI64Type(), value);
  return {};
}

static Value materializeAccStoreScalarPayload(Value value,
                                              PatternRewriter &rewriter,
                                              Location loc) {
  if (!value)
    return {};
  if (Value raw = materializeI64Value(value, rewriter, loc))
    return raw;

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
  if (!value)
    return {};

  if (value.getType().isF16()) {
    Value bitsI16 =
        rewriter.create<arith::BitcastOp>(loc, rewriter.getI16Type(), value);
    return rewriter.create<arith::ExtUIOp>(loc, rewriter.getI64Type(), bitsI16);
  }

  auto intType = dyn_cast<IntegerType>(value.getType());
  if (!intType)
    return {};

  Value widened;
  if (auto dstIntType = dyn_cast<IntegerType>(destinationElementType);
      dstIntType && dstIntType.isUnsignedInteger(8)) {
    widened = rewriter.create<arith::ExtUIOp>(loc, rewriter.getI64Type(), value);
  } else {
    widened = rewriter.create<arith::ExtSIOp>(loc, rewriter.getI64Type(), value);
  }

  Value mask = rewriter.create<arith::ConstantIntOp>(loc, 0xFFFF, 64);
  return rewriter.create<arith::AndIOp>(loc, widened, mask);
}

static Value getI64Constant(Location loc, PatternRewriter &rewriter,
                            uint64_t value) {
  return rewriter.create<arith::ConstantIntOp>(loc, value, 64);
}

static Value buildAccStoreOptionalEnumValue(Location loc,
                                            std::optional<uint32_t> value,
                                            PatternRewriter &rewriter) {
  return getI64Constant(loc, rewriter, value.value_or(0));
}

static Value buildAccStoreFpcValue(Location loc, Value preQuant,
                                   std::optional<pto::AccStoreQuantPreMode> preQuantMode,
                                   Value preRelu,
                                   std::optional<pto::ReluPreMode> preReluMode,
                                   PatternRewriter &rewriter) {
  auto encodeFixpipeBufferAddr = [&](Value addr, uint64_t unitShift) -> Value {
    Value segmentMask = getI64Constant(loc, rewriter, 0xffff);
    Value fieldMask = getI64Constant(loc, rewriter, 0xff);
    Value segmentOffset = rewriter.create<arith::AndIOp>(loc, addr, segmentMask);
    Value scaledAddr = rewriter.create<arith::ShRUIOp>(
        loc, segmentOffset, getI64Constant(loc, rewriter, unitShift));
    return rewriter.create<arith::AndIOp>(loc, scaledAddr, fieldMask);
  };

  Value quantAddr;
  if (preQuantMode) {
    switch (*preQuantMode) {
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
      if (Value quantPtr = materializeI64Value(preQuant, rewriter, loc))
        quantAddr = encodeFixpipeBufferAddr(quantPtr, /*unitShift=*/7);
      break;
    default:
      break;
    }
  }

  Value reluAddr;
  if (preReluMode && *preReluMode == pto::ReluPreMode::VectorRelu) {
    if (Value reluPtr = materializeI64Value(preRelu, rewriter, loc))
      reluAddr = encodeFixpipeBufferAddr(reluPtr, /*unitShift=*/6);
  }

  if (!quantAddr && !reluAddr)
    return {};

  Value mask = getI64Constant(loc, rewriter, 0xff);
  Value fpc = getI64Constant(loc, rewriter, 0);
  if (quantAddr) {
    Value quantShift = getI64Constant(loc, rewriter, 8);
    Value quantBits = rewriter.create<arith::ShLIOp>(loc, quantAddr, quantShift);
    fpc = rewriter.create<arith::OrIOp>(loc, fpc, quantBits);
  }
  if (reluAddr) {
    Value reluBits = rewriter.create<arith::AndIOp>(loc, reluAddr, mask);
    fpc = rewriter.create<arith::OrIOp>(loc, fpc, reluBits);
  }
  return fpc;
}

static void configureAccStoreScalarPreOps(Location loc, Value preQuant,
                                          std::optional<pto::AccStoreQuantPreMode> preQuantMode,
                                          Value preRelu,
                                          std::optional<pto::ReluPreMode> preReluMode,
                                          Value clipValue,
                                          Type destinationElementType,
                                          PatternRewriter &rewriter) {
  auto isVectorQuantMode = [](pto::AccStoreQuantPreMode mode) {
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
  };

  if (preQuantMode && !isVectorQuantMode(*preQuantMode)) {
    if (Value quantValue = materializeAccStoreScalarPayload(preQuant, rewriter, loc))
      rewriter.create<pto::SetQuantPreOp>(loc, quantValue);
  }
  if (preReluMode && *preReluMode == pto::ReluPreMode::ScalarRelu) {
    if (Value reluAlpha = materializeAccStoreScalarPayload(preRelu, rewriter, loc))
      rewriter.create<pto::SetReluAlphaOp>(loc, reluAlpha);
  }
  if (clipValue) {
    if (Value clip = materializeAccStoreClipPayload(clipValue,
                                                    destinationElementType,
                                                    rewriter, loc))
      rewriter.create<pto::SetFixClipReluOp>(loc, clip);
  }
}

static Value configureAccStoreCtrl(Location loc, bool allowAtomic,
                                   std::optional<pto::AccStoreAtomicType> atomicType,
                                   std::optional<pto::AccStoreAtomicOp> atomicOp,
                                   std::optional<pto::AccStoreSatMode> satMode,
                                   PatternRewriter &rewriter) {
  if ((!allowAtomic || !atomicType || !atomicOp) && !satMode)
    return {};

  Value originalCtrl = rewriter.create<pto::GetCtrlOp>(loc);
  Value ctrl = originalCtrl;
  uint64_t clearMaskValue = 0;
  if (allowAtomic && atomicType && atomicOp)
    clearMaskValue |= (static_cast<uint64_t>(0x7) << 6) |
                      (static_cast<uint64_t>(0x3) << 9);
  if (satMode)
    clearMaskValue |= (static_cast<uint64_t>(1) << 48) |
                      (static_cast<uint64_t>(1) << 50);
  Value clearMask = getI64Constant(loc, rewriter, clearMaskValue);
  Value fullMask = getI64Constant(loc, rewriter, ~static_cast<uint64_t>(0));
  Value keepMask = rewriter.create<arith::XOrIOp>(loc, clearMask, fullMask);
  ctrl = rewriter.create<arith::AndIOp>(loc, ctrl, keepMask);

  if (allowAtomic && atomicType && atomicOp) {
    uint64_t atomicBits = (static_cast<uint64_t>(static_cast<uint32_t>(*atomicType)) << 6) |
                          (static_cast<uint64_t>(static_cast<uint32_t>(*atomicOp)) << 9);
    ctrl = rewriter.create<arith::OrIOp>(loc, ctrl,
                                         getI64Constant(loc, rewriter, atomicBits));
  }
  if (satMode && *satMode == pto::AccStoreSatMode::NoSat) {
    ctrl = rewriter.create<arith::OrIOp>(
        loc, ctrl, getI64Constant(loc, rewriter,
                                  static_cast<uint64_t>(1) << 48));
  }
  if (satMode && *satMode == pto::AccStoreSatMode::SatPreserveNan) {
    ctrl = rewriter.create<arith::OrIOp>(
        loc, ctrl, getI64Constant(loc, rewriter,
                                  static_cast<uint64_t>(1) << 50));
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
  Value shift = rewriter.create<arith::ConstantIntOp>(loc, 40, 64);
  Value highShifted = rewriter.create<arith::ShLIOp>(loc, high, shift);
  return rewriter.create<arith::OrIOp>(loc, highShifted, low);
}

static Value packLoopSize(Location loc, Value loop2, Value loop1,
                          PatternRewriter &rewriter) {
  Value shift = rewriter.create<arith::ConstantIntOp>(loc, 21, 64);
  Value loop2Shifted = rewriter.create<arith::ShLIOp>(loc, loop2, shift);
  return rewriter.create<arith::OrIOp>(loc, loop2Shifted, loop1);
}

static Value castIntegerLikeTo(Location loc, Value value, Type targetType,
                               PatternRewriter &rewriter) {
  if (value.getType() == targetType)
    return value;

  auto targetInt = dyn_cast<IntegerType>(targetType);
  if (value.getType().isIndex() && targetInt)
    return rewriter.create<arith::IndexCastOp>(loc, targetType, value);
  if (auto sourceInt = dyn_cast<IntegerType>(value.getType())) {
    if (targetInt) {
      if (sourceInt.getWidth() < targetInt.getWidth())
        return rewriter.create<arith::ExtUIOp>(loc, targetType, value);
      if (sourceInt.getWidth() > targetInt.getWidth())
        return rewriter.create<arith::TruncIOp>(loc, targetType, value);
      return value;
    }
    if (targetType.isIndex())
      return rewriter.create<arith::IndexCastOp>(loc, targetType, value);
  }

  return {};
}

static FailureOr<Value> packMadXt(Location loc, Value m, Value n, Value k,
                                  std::optional<pto::MadUnitFlagMode> unitFlagMode,
                                  bool disableGemv, bool cmatrixSource,
                                  bool cmatrixInit,
                                  PatternRewriter &rewriter) {
  Type i64Ty = rewriter.getI64Type();
  Value mI64 = castIntegerLikeTo(loc, m, i64Ty, rewriter);
  Value nI64 = castIntegerLikeTo(loc, n, i64Ty, rewriter);
  Value kI64 = castIntegerLikeTo(loc, k, i64Ty, rewriter);
  if (!mI64 || !nI64 || !kI64)
    return failure();

  auto constant = [&](uint64_t value) -> Value {
    return rewriter.create<arith::ConstantIntOp>(loc, value, 64);
  };
  auto shl = [&](Value value, uint64_t amount) -> Value {
    return rewriter.create<arith::ShLIOp>(loc, value, constant(amount));
  };
  auto bitOr = [&](Value lhs, Value rhs) -> Value {
    return rewriter.create<arith::OrIOp>(loc, lhs, rhs);
  };

  Value xt = mI64;
  xt = bitOr(xt, shl(kI64, 12));
  xt = bitOr(xt, shl(nI64, 24));
  if (unitFlagMode) {
    uint64_t unitFlagCtrl =
        *unitFlagMode == pto::MadUnitFlagMode::CheckOnly ? 2 : 3;
    xt = bitOr(xt, shl(constant(unitFlagCtrl), 55));
  }
  if (disableGemv)
    xt = bitOr(xt, shl(constant(1), 61));
  if (cmatrixSource)
    xt = bitOr(xt, shl(constant(1), 62));
  if (cmatrixInit)
    xt = bitOr(xt, shl(constant(1), 63));
  return xt;
}

static Value setCtrlBit(Location loc, Value ctrl, unsigned bitIndex, bool value,
                        PatternRewriter &rewriter) {
  Value bit = rewriter.create<arith::ConstantIntOp>(loc, bitIndex, 64);
  if (value)
    return rewriter.create<pto::Sbitset1Op>(loc, ctrl, bit).getResult();
  return rewriter.create<pto::Sbitset0Op>(loc, ctrl, bit).getResult();
}

static Value buildMadSemanticCtrl(Location loc, Value ctrl,
                                  bool isHif8,
                                  std::optional<pto::Tf32Mode> tf32Mode,
                                  std::optional<pto::MadSatMode> satMode,
                                  bool hasNDir,
                                  PatternRewriter &rewriter) {
  ctrl = setCtrlBit(loc, ctrl, 45, isHif8, rewriter);
  if (tf32Mode) {
    ctrl = setCtrlBit(loc, ctrl, 46, true, rewriter);
    ctrl = setCtrlBit(loc, ctrl, 47,
                      *tf32Mode == pto::Tf32Mode::RoundAway, rewriter);
  } else {
    ctrl = setCtrlBit(loc, ctrl, 46, false, rewriter);
    ctrl = setCtrlBit(loc, ctrl, 47, false, rewriter);
  }
  if (satMode)
    ctrl = setCtrlBit(loc, ctrl, 48, *satMode == pto::MadSatMode::NoSat,
                      rewriter);
  ctrl = setCtrlBit(loc, ctrl, 51, hasNDir, rewriter);
  return ctrl;
}

static Value packMte2NzPara(Location loc, Value groupCount, Value dstLoop2Stride,
                            Value dstLoop3Stride, Value dstLoop4Stride,
                            PatternRewriter &rewriter) {
  Value shift16 = rewriter.create<arith::ConstantIntOp>(loc, 16, 64);
  Value shift32 = rewriter.create<arith::ConstantIntOp>(loc, 32, 64);
  Value shift48 = rewriter.create<arith::ConstantIntOp>(loc, 48, 64);
  Value loop2Bits =
      rewriter.create<arith::ShLIOp>(loc, dstLoop2Stride, shift16);
  Value loop3Bits =
      rewriter.create<arith::ShLIOp>(loc, dstLoop3Stride, shift32);
  Value loop4Bits =
      rewriter.create<arith::ShLIOp>(loc, dstLoop4Stride, shift48);
  Value low = rewriter.create<arith::OrIOp>(loc, groupCount, loop2Bits);
  Value high = rewriter.create<arith::OrIOp>(loc, loop3Bits, loop4Bits);
  return rewriter.create<arith::OrIOp>(loc, low, high);
}

static Value packCopyMatrixCcToGmXm(Location loc, Value sid, Value nSize,
                                    Value mSize, Value dstStride,
                                    PatternRewriter &rewriter) {
  Value nShift4 = rewriter.create<arith::ConstantIntOp>(loc, 4, 64);
  Value mShift16 = rewriter.create<arith::ConstantIntOp>(loc, 16, 64);
  Value dstShift32 = rewriter.create<arith::ConstantIntOp>(loc, 32, 64);
  Value nBits = rewriter.create<arith::ShLIOp>(loc, nSize, nShift4);
  Value mBits = rewriter.create<arith::ShLIOp>(loc, mSize, mShift16);
  Value dstStrideBits = rewriter.create<arith::ShLIOp>(loc, dstStride, dstShift32);
  Value sidMask = rewriter.create<arith::ConstantIntOp>(loc, 0xf, 64);
  Value sidBits = rewriter.create<arith::AndIOp>(loc, sid, sidMask);
  Value xmLow = rewriter.create<arith::OrIOp>(loc, sidBits, nBits);
  xmLow = rewriter.create<arith::OrIOp>(loc, xmLow, mBits);
  return rewriter.create<arith::OrIOp>(loc, xmLow, dstStrideBits);
}

static Value packCopyMatrixCcToGmXt(Location loc, Value srcStride,
                                    Value clipReluPre, Value unitFlagCtrl,
                                    Value quantPre, Value reluPreMode,
                                    Value l2CacheCtrl,
                                    Value nz2ndEn, Value channelSplitEn,
                                    Value nz2dnEn,
                                    PatternRewriter &rewriter) {
  Value l2CacheShift16 = rewriter.create<arith::ConstantIntOp>(loc, 16, 64);
  Value clipReluShift30 = rewriter.create<arith::ConstantIntOp>(loc, 30, 64);
  Value unitFlagShift32 = rewriter.create<arith::ConstantIntOp>(loc, 32, 64);
  Value quantBlockBitShift29 =
      rewriter.create<arith::ConstantIntOp>(loc, 29, 64);
  Value quantFieldShift34 = rewriter.create<arith::ConstantIntOp>(loc, 34, 64);
  Value reluShift39 = rewriter.create<arith::ConstantIntOp>(loc, 39, 64);
  Value channelSplitShift42 =
      rewriter.create<arith::ConstantIntOp>(loc, 42, 64);
  Value nz2ndShift43 = rewriter.create<arith::ConstantIntOp>(loc, 43, 64);
  Value nz2dnShift62 = rewriter.create<arith::ConstantIntOp>(loc, 62, 64);

  Value quantShift5 = rewriter.create<arith::ConstantIntOp>(loc, 5, 64);
  Value quantLowMask = rewriter.create<arith::ConstantIntOp>(loc, 0x1f, 64);
  Value quantBitMask = rewriter.create<arith::ConstantIntOp>(loc, 0x1, 64);
  Value clipReluMask = rewriter.create<arith::ConstantIntOp>(loc, 0x3, 64);
  Value l2CacheMask = rewriter.create<arith::ConstantIntOp>(loc, 0xf, 64);
  Value unitFlagMask = rewriter.create<arith::ConstantIntOp>(loc, 0x3, 64);
  Value reluMask = rewriter.create<arith::ConstantIntOp>(loc, 0x7, 64);

  Value l2CacheBits = rewriter.create<arith::AndIOp>(loc, l2CacheCtrl, l2CacheMask);
  l2CacheBits =
      rewriter.create<arith::ShLIOp>(loc, l2CacheBits, l2CacheShift16);

  Value clipReluBits =
      rewriter.create<arith::AndIOp>(loc, clipReluPre, clipReluMask);
  clipReluBits =
      rewriter.create<arith::ShLIOp>(loc, clipReluBits, clipReluShift30);

  Value unitFlagBits = rewriter.create<arith::AndIOp>(loc, unitFlagCtrl, unitFlagMask);
  unitFlagBits =
      rewriter.create<arith::ShLIOp>(loc, unitFlagBits, unitFlagShift32);

  Value quantBlockBit = rewriter.create<arith::ShRUIOp>(loc, quantPre, quantShift5);
  quantBlockBit =
      rewriter.create<arith::AndIOp>(loc, quantBlockBit, quantBitMask);
  quantBlockBit = rewriter.create<arith::ShLIOp>(loc, quantBlockBit,
                                                 quantBlockBitShift29);

  Value quantField = rewriter.create<arith::AndIOp>(loc, quantPre, quantLowMask);
  quantField =
      rewriter.create<arith::ShLIOp>(loc, quantField, quantFieldShift34);

  Value reluBits = rewriter.create<arith::AndIOp>(loc, reluPreMode, reluMask);
  reluBits = rewriter.create<arith::ShLIOp>(loc, reluBits, reluShift39);

  Value channelSplitBits =
      rewriter.create<arith::AndIOp>(loc, channelSplitEn, quantBitMask);
  channelSplitBits = rewriter.create<arith::ShLIOp>(loc, channelSplitBits,
                                                    channelSplitShift42);

  Value nz2ndBits = rewriter.create<arith::AndIOp>(loc, nz2ndEn, quantBitMask);
  nz2ndBits =
      rewriter.create<arith::ShLIOp>(loc, nz2ndBits, nz2ndShift43);

  Value nz2dnBits = rewriter.create<arith::AndIOp>(loc, nz2dnEn, quantBitMask);
  nz2dnBits =
      rewriter.create<arith::ShLIOp>(loc, nz2dnBits, nz2dnShift62);

  Value xt = rewriter.create<arith::OrIOp>(loc, srcStride, l2CacheBits);
  xt = rewriter.create<arith::OrIOp>(loc, xt, clipReluBits);
  xt = rewriter.create<arith::OrIOp>(loc, xt, unitFlagBits);
  xt = rewriter.create<arith::OrIOp>(loc, xt, quantBlockBit);
  xt = rewriter.create<arith::OrIOp>(loc, xt, quantField);
  xt = rewriter.create<arith::OrIOp>(loc, xt, reluBits);
  xt = rewriter.create<arith::OrIOp>(loc, xt, channelSplitBits);
  xt = rewriter.create<arith::OrIOp>(loc, xt, nz2ndBits);
  return rewriter.create<arith::OrIOp>(loc, xt, nz2dnBits);
}

static Value packCopyMatrixCcToUbConfig1(Location loc, Value srcStride,
                                         Value dualDstMode, Value subBlockId,
                                         Value clipReluPre, Value unitFlagCtrl,
                                         Value quantPre, Value reluPreMode,
                                         Value nz2ndEn, Value channelSplitEn,
                                         Value nz2dnEn,
                                         PatternRewriter &rewriter) {
  Value dualDstShift16 = rewriter.create<arith::ConstantIntOp>(loc, 16, 64);
  Value subBlockShift18 = rewriter.create<arith::ConstantIntOp>(loc, 18, 64);
  Value clipReluShift30 = rewriter.create<arith::ConstantIntOp>(loc, 30, 64);
  Value unitFlagShift32 = rewriter.create<arith::ConstantIntOp>(loc, 32, 64);
  Value quantBlockBitShift29 =
      rewriter.create<arith::ConstantIntOp>(loc, 29, 64);
  Value quantFieldShift34 = rewriter.create<arith::ConstantIntOp>(loc, 34, 64);
  Value reluShift39 = rewriter.create<arith::ConstantIntOp>(loc, 39, 64);
  Value channelSplitShift42 =
      rewriter.create<arith::ConstantIntOp>(loc, 42, 64);
  Value nz2ndShift43 = rewriter.create<arith::ConstantIntOp>(loc, 43, 64);
  Value nz2dnShift62 = rewriter.create<arith::ConstantIntOp>(loc, 62, 64);

  Value dualDstMask = rewriter.create<arith::ConstantIntOp>(loc, 0x3, 64);
  Value subBlockMask = rewriter.create<arith::ConstantIntOp>(loc, 0x1, 64);
  Value quantShift5 = rewriter.create<arith::ConstantIntOp>(loc, 5, 64);
  Value quantLowMask = rewriter.create<arith::ConstantIntOp>(loc, 0x1f, 64);
  Value quantBitMask = rewriter.create<arith::ConstantIntOp>(loc, 0x1, 64);
  Value clipReluMask = rewriter.create<arith::ConstantIntOp>(loc, 0x3, 64);
  Value unitFlagMask = rewriter.create<arith::ConstantIntOp>(loc, 0x3, 64);
  Value reluMask = rewriter.create<arith::ConstantIntOp>(loc, 0x7, 64);

  Value dualDstBits = rewriter.create<arith::AndIOp>(loc, dualDstMode, dualDstMask);
  dualDstBits =
      rewriter.create<arith::ShLIOp>(loc, dualDstBits, dualDstShift16);

  Value subBlockBits = rewriter.create<arith::AndIOp>(loc, subBlockId, subBlockMask);
  subBlockBits =
      rewriter.create<arith::ShLIOp>(loc, subBlockBits, subBlockShift18);

  Value clipReluBits =
      rewriter.create<arith::AndIOp>(loc, clipReluPre, clipReluMask);
  clipReluBits =
      rewriter.create<arith::ShLIOp>(loc, clipReluBits, clipReluShift30);

  Value unitFlagBits = rewriter.create<arith::AndIOp>(loc, unitFlagCtrl, unitFlagMask);
  unitFlagBits =
      rewriter.create<arith::ShLIOp>(loc, unitFlagBits, unitFlagShift32);

  Value quantBlockBit = rewriter.create<arith::ShRUIOp>(loc, quantPre, quantShift5);
  quantBlockBit =
      rewriter.create<arith::AndIOp>(loc, quantBlockBit, quantBitMask);
  quantBlockBit = rewriter.create<arith::ShLIOp>(loc, quantBlockBit,
                                                 quantBlockBitShift29);

  Value quantField = rewriter.create<arith::AndIOp>(loc, quantPre, quantLowMask);
  quantField =
      rewriter.create<arith::ShLIOp>(loc, quantField, quantFieldShift34);

  Value reluBits = rewriter.create<arith::AndIOp>(loc, reluPreMode, reluMask);
  reluBits = rewriter.create<arith::ShLIOp>(loc, reluBits, reluShift39);

  Value channelSplitBits =
      rewriter.create<arith::AndIOp>(loc, channelSplitEn, quantBitMask);
  channelSplitBits = rewriter.create<arith::ShLIOp>(loc, channelSplitBits,
                                                    channelSplitShift42);

  Value nz2ndBits = rewriter.create<arith::AndIOp>(loc, nz2ndEn, quantBitMask);
  nz2ndBits =
      rewriter.create<arith::ShLIOp>(loc, nz2ndBits, nz2ndShift43);

  Value nz2dnBits = rewriter.create<arith::AndIOp>(loc, nz2dnEn, quantBitMask);
  nz2dnBits =
      rewriter.create<arith::ShLIOp>(loc, nz2dnBits, nz2dnShift62);

  Value config1 = rewriter.create<arith::OrIOp>(loc, srcStride, dualDstBits);
  config1 = rewriter.create<arith::OrIOp>(loc, config1, subBlockBits);
  config1 = rewriter.create<arith::OrIOp>(loc, config1, clipReluBits);
  config1 = rewriter.create<arith::OrIOp>(loc, config1, unitFlagBits);
  config1 = rewriter.create<arith::OrIOp>(loc, config1, quantBlockBit);
  config1 = rewriter.create<arith::OrIOp>(loc, config1, quantField);
  config1 = rewriter.create<arith::OrIOp>(loc, config1, reluBits);
  config1 = rewriter.create<arith::OrIOp>(loc, config1, channelSplitBits);
  config1 = rewriter.create<arith::OrIOp>(loc, config1, nz2ndBits);
  return rewriter.create<arith::OrIOp>(loc, config1, nz2dnBits);
}

static Value packLoop3Config(Location loc, Value count, Value srcStride,
                             Value dstStride, PatternRewriter &rewriter) {
  Value srcShift16 = rewriter.create<arith::ConstantIntOp>(loc, 16, 64);
  Value dstShift32 = rewriter.create<arith::ConstantIntOp>(loc, 32, 64);
  Value srcBits = rewriter.create<arith::ShLIOp>(loc, srcStride, srcShift16);
  Value dstBits = rewriter.create<arith::ShLIOp>(loc, dstStride, dstShift32);
  Value low = rewriter.create<arith::OrIOp>(loc, count, srcBits);
  return rewriter.create<arith::OrIOp>(loc, low, dstBits);
}

static Value packChannelConfig(Location loc, Value loop0SrcStride,
                               PatternRewriter &rewriter) {
  Value shift48 = rewriter.create<arith::ConstantIntOp>(loc, 48, 64);
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

static FailureOr<LoadCbufToCbControl>
deriveLoadCbufToCbControl(Location loc, Value k, Value n, Type elementType,
                          Value mStart, Value kStart, bool transpose,
                          PatternRewriter &rewriter) {
  unsigned elemBitWidth = elementType.getIntOrFloatBitWidth();
  if (elemBitWidth == 0 || (elemBitWidth % 8) != 0)
    return failure();
  uint64_t elemBytes = elemBitWidth / 8;

  auto constant = [&](uint64_t value) -> Value {
    return rewriter.create<arith::ConstantIntOp>(loc, value, 64);
  };
  auto ceilDivConst = [&](Value value, uint64_t divisor) -> Value {
    Value bias = constant(divisor - 1);
    Value sum = rewriter.create<arith::AddIOp>(loc, value, bias);
    return rewriter.create<arith::DivUIOp>(loc, sum, constant(divisor));
  };

  if (!transpose) {
    Value mStep = ceilDivConst(n, 16);
    Value kBytes = rewriter.create<arith::MulIOp>(loc, k, constant(elemBytes));
    Value kStep = ceilDivConst(kBytes, 32);
    Value stride = ceilDivConst(n, 16);
    return LoadCbufToCbControl{mStart, kStart, mStep, kStep, stride, stride};
  }

  uint64_t c0Size = std::max<uint64_t>(16, 32 / elemBytes);
  Value kAlign = ceilDivConst(k, c0Size);
  kAlign = rewriter.create<arith::MulIOp>(loc, kAlign, constant(c0Size));
  Value nAlign = ceilDivConst(n, c0Size);
  nAlign = rewriter.create<arith::MulIOp>(loc, nAlign, constant(c0Size));
  Value mStep = ceilDivConst(kAlign, 16);
  Value nBytes = rewriter.create<arith::MulIOp>(loc, nAlign, constant(elemBytes));
  Value kStep = ceilDivConst(nBytes, 32);
  Value srcStride = ceilDivConst(kAlign, 16);
  Value dstStride = ceilDivConst(nAlign, 16);
  return LoadCbufToCbControl{mStart, kStart, mStep, kStep, srcStride, dstStride};
}

static FailureOr<LoadCbufToCbControl>
deriveLoadCbufToCaControl(Location loc, Value m, Value k, Type elementType,
                          Value mStart, Value kStart, bool transpose,
                          PatternRewriter &rewriter) {
  unsigned elemBitWidth = elementType.getIntOrFloatBitWidth();
  if (elemBitWidth == 0 || (elemBitWidth % 8) != 0)
    return failure();
  uint64_t elemBytes = elemBitWidth / 8;

  auto constant = [&](uint64_t value) -> Value {
    return rewriter.create<arith::ConstantIntOp>(loc, value, 64);
  };
  auto ceilDivConst = [&](Value value, uint64_t divisor) -> Value {
    Value bias = constant(divisor - 1);
    Value sum = rewriter.create<arith::AddIOp>(loc, value, bias);
    return rewriter.create<arith::DivUIOp>(loc, sum, constant(divisor));
  };

  if (!transpose) {
    Value mStep = ceilDivConst(m, 16);
    Value kBytes = rewriter.create<arith::MulIOp>(loc, k, constant(elemBytes));
    Value kStep = ceilDivConst(kBytes, 32);
    Value stride = ceilDivConst(m, 16);
    return LoadCbufToCbControl{mStart, kStart, mStep, kStep, stride, stride};
  }

  uint64_t c0Size = std::max<uint64_t>(16, 32 / elemBytes);
  Value mAlign = ceilDivConst(m, c0Size);
  mAlign = rewriter.create<arith::MulIOp>(loc, mAlign, constant(c0Size));
  Value kAlign = ceilDivConst(k, c0Size);
  kAlign = rewriter.create<arith::MulIOp>(loc, kAlign, constant(c0Size));
  Value mStep = ceilDivConst(kAlign, 16);
  Value mBytes = rewriter.create<arith::MulIOp>(loc, mAlign, constant(elemBytes));
  Value kStep = ceilDivConst(mBytes, 32);
  Value srcStride = ceilDivConst(kAlign, 16);
  Value dstStride = ceilDivConst(mAlign, 16);
  return LoadCbufToCbControl{mStart, kStart, mStep, kStep, srcStride, dstStride};
}

static Value extractConfigLow40(Location loc, Value packed,
                                PatternRewriter &rewriter) {
  Value lowMask =
      rewriter.create<arith::ConstantIntOp>(loc, 0xffffffffffULL, 64);
  return rewriter.create<arith::AndIOp>(loc, packed, lowMask);
}

static Value extractConfigHigh24(Location loc, Value packed,
                                 PatternRewriter &rewriter) {
  Value shift40 = rewriter.create<arith::ConstantIntOp>(loc, 40, 64);
  return rewriter.create<arith::ShRUIOp>(loc, packed, shift40);
}

template <typename BodyBuilder>
static void buildSoftwareLoopNest(PatternRewriter &rewriter, Location loc,
                                  ArrayRef<pto::DmaLoopConfig> loops,
                                  Value srcOffset, Value dstOffset,
                                  BodyBuilder &&buildLeaf) {
  if (loops.empty()) {
    buildLeaf(srcOffset, dstOffset);
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
        loc, srcOffset, ivI64, loops.front().srcStride, rewriter);
    Value nextDstOffset = buildAccumulatedByteOffset(
        loc, dstOffset, ivI64, loops.front().dstStride, rewriter);
    buildSoftwareLoopNest(rewriter, loc, loops.drop_front(), nextSrcOffset,
                          nextDstOffset, buildLeaf);
  }
}

struct ExpandUvldPattern : public OpRewritePattern<pto::UvldOp> {
  using OpRewritePattern<pto::UvldOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::UvldOp op,
                                PatternRewriter &rewriter) const override {
    auto vecType = dyn_cast<pto::VRegType>(op.getResult().getType());
    if (!vecType)
      return failure();

    Value basePtr = materializeBufferPointer(op.getSource(), rewriter, op.getLoc());
    if (!basePtr)
      return op.emitOpError(
          "requires a recoverable pointer base for uvld expansion");

    Value loadPtr = offsetBufferPointer(basePtr, vecType.getElementType(),
                                       op.getOffset(), rewriter, op.getLoc());
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
  if (op.isMadMxFamily())
    return op.hasBiasOperand() ? MadRawKind::MxBias : MadRawKind::Mx;
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
          dyn_cast_or_null<pto::MadUnitFlagModeAttr>(op.getUnitFlagModeAttr()))
    unitFlagMode = unitFlagModeAttr.getValue();

  std::optional<pto::Tf32Mode> tf32Mode;
  if (op.supportsTf32Mode()) {
    if (auto tf32ModeAttr =
            dyn_cast_or_null<pto::Tf32ModeAttr>(op.getTf32ModeAttr()))
      tf32Mode = tf32ModeAttr.getValue();
  }

  std::optional<pto::MadSatMode> satMode;
  if (auto satModeAttr =
          dyn_cast_or_null<pto::MadSatModeAttr>(op.getSatModeAttr()))
    satMode = satModeAttr.getValue();

  bool isHif8 = false;
  if (auto lhsPtr = dyn_cast<pto::PtrType>(op.getLhs().getType()))
    isHif8 = pto::isPTOHiFloat8Type(lhsPtr.getElementType());

  Location loc = op->getLoc();
  Value ctrlSaved = rewriter.create<pto::GetCtrlOp>(loc).getResult();
  Value ctrlForOp = buildMadSemanticCtrl(loc, ctrlSaved, isHif8, tf32Mode,
                                         satMode, op.getNDir(), rewriter);
  rewriter.create<pto::SetCtrlOp>(loc, ctrlForOp);

  FailureOr<Value> xt =
      packMadXt(loc, op.getM(), op.getN(), op.getK(), unitFlagMode,
                op.getDisableGemv(), op.initializesAccumulatorWithBias(),
                op.initializesAccumulatorWithZero(), rewriter);
  if (failed(xt))
    return rewriter.notifyMatchFailure(op, "failed to pack mad xt");

  if (failed(emitMadRawOp(op, deriveMadRawKind(op), *xt, rewriter)))
    return rewriter.notifyMatchFailure(op, "failed to emit mad raw op");

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
    if (!semantic)
      return failure();
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
    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
    Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 64);
    SmallVector<pto::DmaLoopConfig> loops =
        collectLoopConfigs(op.getLoopCounts(), op.getLoopSrcStrides(),
                           op.getLoopDstStrides());

    SmallVector<pto::DmaLoopConfig> allLoops;
    ArrayRef<pto::DmaLoopConfig> hwLoops;
    ArrayRef<pto::DmaLoopConfig> swLoops;
    Value loop1Count;
    Value loop2Size = one;

    if (dmaArch == DmaArch::A5) {
      hwLoops = ArrayRef<pto::DmaLoopConfig>(loops).take_front(2);
      swLoops = ArrayRef<pto::DmaLoopConfig>(loops).drop_front(hwLoops.size());
      if (hwLoops.size() == 2) {
        rewriter.create<pto::SetLoop2StrideOutToUbOp>(
            loc, hwLoops[0].srcStride, hwLoops[0].dstStride);
        loop2Size = hwLoops[0].count;
        loop1Count = hwLoops[1].count;
        rewriter.create<pto::SetLoop1StrideOutToUbOp>(
            loc, hwLoops[1].srcStride, hwLoops[1].dstStride);
        rewriter.create<pto::SetLoopSizeOutToUbOp>(loc, loop2Size, loop1Count);
      } else if (hwLoops.size() == 1) {
        loop1Count = hwLoops[0].count;
        rewriter.create<pto::SetLoop1StrideOutToUbOp>(
            loc, hwLoops[0].srcStride, hwLoops[0].dstStride);
        rewriter.create<pto::SetLoopSizeOutToUbOp>(loc, loop2Size, loop1Count);
      }
    } else {
      allLoops.append(loops);
      pto::DmaLoopConfig nburstCfg;
      nburstCfg.count = op.getNBurst();
      nburstCfg.srcStride = op.getNburstSrcStride();
      nburstCfg.dstStride = op.getNburstDstStride();
      allLoops.push_back(nburstCfg);
      swLoops = ArrayRef<pto::DmaLoopConfig>(allLoops);
    }

    Value leftPadding = op.getLeftPaddingCount();
    if (!leftPadding)
      leftPadding = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
    Value rightPadding = op.getRightPaddingCount();
    if (!rightPadding)
      rightPadding = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
    Value dataSelect = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI1Type(),
        rewriter.getBoolAttr(static_cast<bool>(op.getPadValue())));

    bool hasPad = static_cast<bool>(op.getPadValue());
    if (Value padValue = op.getPadValue())
      rewriter.create<pto::SetMovPadValOp>(loc, padValue);

    Value effectiveNBurst = (dmaArch == DmaArch::A5) ? op.getNBurst() : one;

    buildSoftwareLoopNest(
        rewriter, loc, swLoops, zero, zero,
        [&](Value srcOffset, Value dstOffset) {
          Value source = offsetPointerByBytes(op.getSource(), srcOffset, rewriter, loc);
          Value destination =
              offsetPointerByBytes(op.getDestination(), dstOffset, rewriter, loc);
          auto copyOp = rewriter.create<pto::CopyGmToUbufOp>(
              loc, source, destination, zero, effectiveNBurst, op.getLenBurst(),
              leftPadding, rightPadding, dataSelect, op.getL2CacheCtl(),
              op.getNburstSrcStride(), op.getNburstDstStride());
          if (hasPad)
            copyOp->setAttr("has_pad", UnitAttr::get(copyOp->getContext()));
        });
    if (dmaArch == DmaArch::A5 &&
        shouldRestoreDmaLoopSize(loop1Count, loop2Size))
      rewriter.create<pto::SetLoopSizeOutToUbOp>(loc, one, one);
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
    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
    Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 64);
    SmallVector<pto::DmaLoopConfig> loops =
        collectLoopConfigs(op.getLoopCounts(), op.getLoopSrcStrides(),
                           op.getLoopDstStrides());

    SmallVector<pto::DmaLoopConfig> allLoops;
    ArrayRef<pto::DmaLoopConfig> hwLoops;
    ArrayRef<pto::DmaLoopConfig> swLoops;
    Value loop1Count;
    Value loop2Size = one;

    if (dmaArch == DmaArch::A5) {
      hwLoops = ArrayRef<pto::DmaLoopConfig>(loops).take_front(2);
      swLoops =
          ArrayRef<pto::DmaLoopConfig>(loops).drop_front(hwLoops.size());
      if (hwLoops.size() == 2) {
        rewriter.create<pto::SetLoop2StrideUbToOutOp>(
            loc, hwLoops[0].srcStride, hwLoops[0].dstStride);
        loop2Size = hwLoops[0].count;
        loop1Count = hwLoops[1].count;
        rewriter.create<pto::SetLoop1StrideUbToOutOp>(
            loc, hwLoops[1].srcStride, hwLoops[1].dstStride);
        rewriter.create<pto::SetLoopSizeUbToOutOp>(loc, loop2Size, loop1Count);
      } else if (hwLoops.size() == 1) {
        loop1Count = hwLoops[0].count;
        rewriter.create<pto::SetLoop1StrideUbToOutOp>(
            loc, hwLoops[0].srcStride, hwLoops[0].dstStride);
        rewriter.create<pto::SetLoopSizeUbToOutOp>(loc, loop2Size, loop1Count);
      }
    } else {
      allLoops.append(loops);
      pto::DmaLoopConfig nburstCfg;
      nburstCfg.count = op.getNBurst();
      nburstCfg.srcStride = op.getNburstSrcStride();
      nburstCfg.dstStride = op.getNburstDstStride();
      allLoops.push_back(nburstCfg);
      swLoops = ArrayRef<pto::DmaLoopConfig>(allLoops);
    }

    Value effectiveNBurst = (dmaArch == DmaArch::A5) ? op.getNBurst() : one;

    buildSoftwareLoopNest(
        rewriter, loc, swLoops, zero, zero,
        [&](Value srcOffset, Value dstOffset) {
          Value source = offsetPointerByBytes(op.getSource(), srcOffset, rewriter, loc);
          Value destination =
              offsetPointerByBytes(op.getDestination(), dstOffset, rewriter, loc);
          rewriter.create<pto::CopyUbufToGmOp>(
              loc, source, destination, zero, effectiveNBurst, op.getLenBurst(),
              zero, op.getNburstDstStride(), op.getNburstSrcStride());
        });
    if (dmaArch == DmaArch::A5 &&
        shouldRestoreDmaLoopSize(loop1Count, loop2Size))
      rewriter.create<pto::SetLoopSizeUbToOutOp>(loc, one, one);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ExpandMteUbUbPattern : public OpRewritePattern<pto::MteUbUbOp> {
  using OpRewritePattern<pto::MteUbUbOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::MteUbUbOp op,
                                PatternRewriter &rewriter) const override {
    Value zero = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0, 64);
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
    Value zero = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0, 64);
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
    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
    Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 64);
    SmallVector<pto::DmaLoopConfig> loops =
        collectLoopConfigs(op.getLoopCounts(), op.getLoopSrcStrides(),
                           op.getLoopDstStrides());
    ArrayRef<pto::DmaLoopConfig> hwLoops =
        ArrayRef<pto::DmaLoopConfig>(loops).take_front(2);
    ArrayRef<pto::DmaLoopConfig> swLoops =
        ArrayRef<pto::DmaLoopConfig>(loops).drop_front(hwLoops.size());

    Value loop1Count;
    Value loop2Count = one;
    if (hwLoops.size() == 2) {
      rewriter.create<pto::SetLoop2StrideOutToL1Op>(
          loc,
          packLoopPair(loc, hwLoops[0].srcStride, hwLoops[0].dstStride,
                       rewriter));
      loop2Count = hwLoops[0].count;
      loop1Count = hwLoops[1].count;
      rewriter.create<pto::SetLoop1StrideOutToL1Op>(
          loc,
          packLoopPair(loc, hwLoops[1].srcStride, hwLoops[1].dstStride,
                       rewriter));
      rewriter.create<pto::SetLoopSizeOutToL1Op>(
          loc, packLoopSize(loc, loop2Count, loop1Count, rewriter));
    } else if (hwLoops.size() == 1) {
      loop1Count = hwLoops[0].count;
      rewriter.create<pto::SetLoop1StrideOutToL1Op>(
          loc,
          packLoopPair(loc, hwLoops[0].srcStride, hwLoops[0].dstStride,
                       rewriter));
      rewriter.create<pto::SetLoopSizeOutToL1Op>(
          loc, packLoopSize(loc, loop2Count, loop1Count, rewriter));
    }

    SmallVector<pto::DmaLoopConfig> swLoopNestOrder(swLoops.rbegin(),
                                                    swLoops.rend());
    buildSoftwareLoopNest(
        rewriter, loc, swLoopNestOrder, zero, zero,
        [&](Value srcOffset, Value dstOffset) {
          Value source =
              offsetPointerByBytes(op.getSource(), srcOffset, rewriter, loc);
          Value destination = offsetPointerByBytes(op.getDestination(), dstOffset,
                                                   rewriter, loc);
          rewriter.create<pto::CopyGmToCbufOp>(
              loc, source, destination, op.getNBurst(), op.getLenBurst(),
              op.getNburstSrcStride(), op.getNburstDstStride());
        });
    if (loop1Count && (!isKnownOne(loop1Count) || !isKnownOne(loop2Count)))
      rewriter.create<pto::SetLoopSizeOutToL1Op>(
          loc, packLoopSize(loc, one, one, rewriter));
    rewriter.eraseOp(op);
    return success();
  }
};

struct ExpandCubeStorePattern : public OpRewritePattern<pto::MteL1UbOp> {
  using OpRewritePattern<pto::MteL1UbOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::MteL1UbOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
    SmallVector<pto::DmaLoopConfig> loops =
        collectLoopConfigs(op.getLoopCounts(), op.getLoopSrcStrides(),
                           op.getLoopDstStrides());
    SmallVector<pto::DmaLoopConfig> swLoopNestOrder(loops.rbegin(),
                                                    loops.rend());
    buildSoftwareLoopNest(
        rewriter, loc, swLoopNestOrder, zero, zero,
        [&](Value srcOffset, Value dstOffset) {
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
    auto sourceType = dyn_cast<pto::PtrType>(
        materializeBufferPointer(op.getSource(), rewriter, loc).getType());
    if (!sourceType)
      return rewriter.notifyMatchFailure(op, "expected pointer-like source");

    Value convControl = rewriter.create<arith::ConstantIntOp>(
        loc, sourceType.getElementType().isF16() ? 1 : 0, 1);
    rewriter.replaceOpWithNewOp<pto::CopyCbufToBtOp>(
        op, op.getSource(), op.getDestination(), convControl, op.getNBurst(),
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
    if (!source || !destination)
      return rewriter.notifyMatchFailure(op, "expected pointer-like operands");

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
    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
    Value mte2NzPara = packMte2NzPara(
        loc, op.getGroupCount(), op.getDstLoop2Stride(), op.getDstLoop3Stride(),
        op.getDstLoop4Stride(), rewriter);
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
    auto sourceType = dyn_cast<pto::PtrType>(op.getSource().getType());
    if (!sourceType)
      return rewriter.notifyMatchFailure(op, "expected typed L1 source");
    FailureOr<LoadCbufToCbControl> control = deriveLoadCbufToCaControl(
        loc, op.getM(), op.getK(), sourceType.getElementType(),
        op.getStartRow(), op.getStartCol(), op.getTranspose(), rewriter);
    if (failed(control))
      return rewriter.notifyMatchFailure(op,
                                         "failed to derive load_cbuf_to_ca control");
    auto load = rewriter.create<pto::LoadCbufToCaOp>(
        loc, op.getSource(), op.getDestination(), control->mStart,
        control->kStart, control->mStep, control->kStep, control->srcStride,
        control->dstStride);
    load->setAttr("transpose", rewriter.getBoolAttr(op.getTranspose()));
    rewriter.eraseOp(op);
    return success();
  }
};

struct ExpandRightLoadPattern : public OpRewritePattern<pto::MteL1L0bOp> {
  using OpRewritePattern<pto::MteL1L0bOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::MteL1L0bOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto sourceType = dyn_cast<pto::PtrType>(op.getSource().getType());
    if (!sourceType)
      return rewriter.notifyMatchFailure(op, "expected typed L1 source");
    FailureOr<LoadCbufToCbControl> control = deriveLoadCbufToCbControl(
        loc, op.getK(), op.getN(), sourceType.getElementType(),
        op.getStartRow(), op.getStartCol(), op.getTranspose(), rewriter);
    if (failed(control))
      return rewriter.notifyMatchFailure(op,
                                         "failed to derive load_cbuf_to_cb control");
    auto load = rewriter.create<pto::LoadCbufToCbOp>(
        loc, op.getSource(), op.getDestination(), control->mStart,
        control->kStart, control->mStep, control->kStep, control->srcStride,
        control->dstStride);
    load->setAttr("transpose", rewriter.getBoolAttr(op.getTranspose()));
    rewriter.eraseOp(op);
    return success();
  }
};

struct ExpandLeftLoadMxPattern : public OpRewritePattern<pto::MteL1L0aMxOp> {
  using OpRewritePattern<pto::MteL1L0aMxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::MteL1L0aMxOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<pto::LoadCbufToCaMxOp>(op.getLoc(), op.getSource(),
                                           op.getDestination(), op.getM(),
                                           op.getK());
    rewriter.eraseOp(op);
    return success();
  }
};

struct ExpandRightLoadMxPattern : public OpRewritePattern<pto::MteL1L0bMxOp> {
  using OpRewritePattern<pto::MteL1L0bMxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::MteL1L0bMxOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto sourceType = dyn_cast<pto::PtrType>(op.getSource().getType());
    if (!sourceType)
      return rewriter.notifyMatchFailure(op, "expected typed L1 source");

    unsigned elemBitWidth = sourceType.getElementType().getIntOrFloatBitWidth();
    if (elemBitWidth == 0 || (elemBitWidth % 8) != 0)
      return rewriter.notifyMatchFailure(op, "unsupported element type");
    uint64_t elemBytes = elemBitWidth / 8;

    auto constant = [&](uint64_t value) -> Value {
      return rewriter.create<arith::ConstantIntOp>(loc, value, 64);
    };
    auto ceilDivConst = [&](Value value, uint64_t divisor) -> Value {
      Value bias = constant(divisor - 1);
      Value sum = rewriter.create<arith::AddIOp>(loc, value, bias);
      return rewriter.create<arith::DivUIOp>(loc, sum, constant(divisor));
    };

    Value zero = constant(0);
    Value one = constant(1);
    Value yStep = ceilDivConst(
        rewriter.create<arith::MulIOp>(loc, op.getK(), constant(elemBytes)), 32);
    Value stride = ceilDivConst(op.getN(), 16);

    rewriter.create<pto::LoadCbufToCbMxOp>(
        loc, op.getSource(), op.getDestination(), zero, zero, one, yStep, stride,
        stride);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ExpandAccStorePattern : public OpRewritePattern<pto::MteL0cL1Op> {
  using OpRewritePattern<pto::MteL0cL1Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::MteL0cL1Op op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zero = getI64Constant(loc, rewriter, 0);
    Value one = getI64Constant(loc, rewriter, 1);
    configureAccStoreScalarPreOps(loc, op.getPreQuant(), op.getPreQuantMode(),
                                  op.getPreRelu(), op.getPreReluMode(),
                                  op.getClipValue(),
                                  getBufferElementType(op.getDestination().getType()),
                                  rewriter);
    if (Value fpc = buildAccStoreFpcValue(loc, op.getPreQuant(),
                                          op.getPreQuantMode(),
                                          op.getPreRelu(),
                                          op.getPreReluMode(), rewriter))
      rewriter.create<pto::SetFpcOp>(loc, fpc);
    Value originalCtrl =
        configureAccStoreCtrl(loc, /*allowAtomic=*/false, std::nullopt,
                              std::nullopt, op.getSatMode(), rewriter);
    pto::DmaLoopConfig hwLoop{one, zero, zero};
    if (Value loop3Count = op.getLoop3Count()) {
      hwLoop = {loop3Count, op.getLoop3SrcStride(), op.getLoop3DstStride()};
    }

    Value channelLoop0Stride = zero;
    Value nz2ndEn = zero;
    Value channelSplitEn = zero;
    Value nz2dnEn = zero;
    if (auto mode = op.getMode()) {
      switch (*mode) {
      case pto::AccStoreMode::Nz2nd:
        nz2ndEn = one;
        break;
      case pto::AccStoreMode::Nz2dn:
        nz2dnEn = one;
        channelLoop0Stride = op.getLoop0SrcStride() ? op.getLoop0SrcStride() : one;
        break;
      case pto::AccStoreMode::Nz2nz:
        channelSplitEn = op.getSplit() ? op.getSplit() : zero;
        break;
      }
    } else {
      nz2ndEn = one;
    }

    Value loop3Config = packLoop3Config(loc, hwLoop.count, hwLoop.srcStride,
                                        hwLoop.dstStride, rewriter);
    Value channelConfig =
        packChannelConfig(loc, channelLoop0Stride, rewriter);
    rewriter.create<pto::SetLoop3ParaOp>(
        loc, extractConfigLow40(loc, loop3Config, rewriter),
        extractConfigHigh24(loc, loop3Config, rewriter));
    rewriter.create<pto::SetChannelParaOp>(
        loc, extractConfigLow40(loc, channelConfig, rewriter),
        extractConfigHigh24(loc, channelConfig, rewriter));
    Value clipReluPre = getI64Constant(loc, rewriter, op.getClipValue() ? 1 : 0);
    Value unitFlagCtrl = buildAccStoreOptionalEnumValue(
        loc,
        op.getUnitFlag()
            ? std::optional<uint32_t>(static_cast<uint32_t>(*op.getUnitFlag()))
            : std::nullopt,
        rewriter);
    Value quantPreMode = buildAccStoreOptionalEnumValue(
        loc,
        op.getPreQuantMode()
            ? std::optional<uint32_t>(static_cast<uint32_t>(*op.getPreQuantMode()))
            : std::nullopt,
        rewriter);
    Value reluPreMode = buildAccStoreOptionalEnumValue(
        loc,
        op.getPreReluMode()
            ? std::optional<uint32_t>(static_cast<uint32_t>(*op.getPreReluMode()))
            : std::nullopt,
        rewriter);
    Value xm =
        packCopyMatrixCcToGmXm(loc, zero, op.getN(), op.getM(),
                               op.getDstStride(), rewriter);
    Value xt = packCopyMatrixCcToGmXt(
        loc, op.getSrcStride(), clipReluPre, unitFlagCtrl, quantPreMode,
        reluPreMode, zero, nz2ndEn, channelSplitEn, nz2dnEn,
        rewriter);
    rewriter.create<pto::CopyMatrixCcToCbufOp>(loc, op.getSource(),
                                               op.getDestination(), xm, xt);
    if (originalCtrl)
      rewriter.create<pto::SetCtrlOp>(loc, originalCtrl);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ExpandAccStoreGmPattern : public OpRewritePattern<pto::MteL0cGmOp> {
  using OpRewritePattern<pto::MteL0cGmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::MteL0cGmOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zero = getI64Constant(loc, rewriter, 0);
    Value one = getI64Constant(loc, rewriter, 1);
    configureAccStoreScalarPreOps(loc, op.getPreQuant(), op.getPreQuantMode(),
                                  op.getPreRelu(), op.getPreReluMode(),
                                  op.getClipValue(),
                                  getBufferElementType(op.getDestination().getType()),
                                  rewriter);
    if (Value fpc = buildAccStoreFpcValue(loc, op.getPreQuant(),
                                          op.getPreQuantMode(),
                                          op.getPreRelu(),
                                          op.getPreReluMode(), rewriter))
      rewriter.create<pto::SetFpcOp>(loc, fpc);
    Value originalCtrl =
        configureAccStoreCtrl(loc, /*allowAtomic=*/true, op.getAtomicType(),
                              op.getAtomicOp(), op.getSatMode(), rewriter);
    pto::DmaLoopConfig hwLoop{one, zero, zero};
    if (Value loop3Count = op.getLoop3Count()) {
      hwLoop = {loop3Count, op.getLoop3SrcStride(), op.getLoop3DstStride()};
    }

    Value channelLoop0Stride = zero;
    Value nz2ndEn = zero;
    Value channelSplitEn = zero;
    Value nz2dnEn = zero;
    if (auto mode = op.getMode()) {
      switch (*mode) {
      case pto::AccStoreMode::Nz2nd:
        nz2ndEn = one;
        break;
      case pto::AccStoreMode::Nz2dn:
        nz2dnEn = one;
        channelLoop0Stride =
            op.getLoop0SrcStride() ? op.getLoop0SrcStride() : one;
        break;
      case pto::AccStoreMode::Nz2nz:
        channelSplitEn = op.getSplit() ? op.getSplit() : zero;
        break;
      }
    } else {
      nz2ndEn = one;
    }

    Value loop3Config = packLoop3Config(loc, hwLoop.count, hwLoop.srcStride,
                                        hwLoop.dstStride, rewriter);
    Value channelConfig = packChannelConfig(loc, channelLoop0Stride, rewriter);
    rewriter.create<pto::SetLoop3ParaOp>(
        loc, extractConfigLow40(loc, loop3Config, rewriter),
        extractConfigHigh24(loc, loop3Config, rewriter));
    rewriter.create<pto::SetChannelParaOp>(
        loc, extractConfigLow40(loc, channelConfig, rewriter),
        extractConfigHigh24(loc, channelConfig, rewriter));
    Value clipReluPre = getI64Constant(loc, rewriter, op.getClipValue() ? 1 : 0);
    Value unitFlagCtrl = buildAccStoreOptionalEnumValue(
        loc,
        op.getUnitFlag()
            ? std::optional<uint32_t>(static_cast<uint32_t>(*op.getUnitFlag()))
            : std::nullopt,
        rewriter);
    Value quantPreMode = buildAccStoreOptionalEnumValue(
        loc,
        op.getPreQuantMode()
            ? std::optional<uint32_t>(static_cast<uint32_t>(*op.getPreQuantMode()))
            : std::nullopt,
        rewriter);
    Value reluPreMode = buildAccStoreOptionalEnumValue(
        loc,
        op.getPreReluMode()
            ? std::optional<uint32_t>(static_cast<uint32_t>(*op.getPreReluMode()))
            : std::nullopt,
        rewriter);
    Value xm = packCopyMatrixCcToGmXm(loc, op.getSid(), op.getN(), op.getM(),
                                      op.getDstStride(), rewriter);
    Value xt = packCopyMatrixCcToGmXt(
        loc, op.getSrcStride(), clipReluPre, unitFlagCtrl, quantPreMode,
        reluPreMode, op.getL2CacheCtrl(), nz2ndEn, channelSplitEn,
        nz2dnEn, rewriter);
    rewriter.create<pto::CopyMatrixCcToGmOp>(loc, op.getSource(),
                                             op.getDestination(), xm, xt);
    if (originalCtrl)
      rewriter.create<pto::SetCtrlOp>(loc, originalCtrl);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ExpandAccStoreUbPattern : public OpRewritePattern<pto::MteL0cUbOp> {
  using OpRewritePattern<pto::MteL0cUbOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::MteL0cUbOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zero = getI64Constant(loc, rewriter, 0);
    Value one = getI64Constant(loc, rewriter, 1);
    configureAccStoreScalarPreOps(loc, op.getPreQuant(), op.getPreQuantMode(),
                                  op.getPreRelu(), op.getPreReluMode(),
                                  op.getClipValue(),
                                  getBufferElementType(op.getDestination().getType()),
                                  rewriter);
    if (Value fpc = buildAccStoreFpcValue(loc, op.getPreQuant(),
                                          op.getPreQuantMode(),
                                          op.getPreRelu(),
                                          op.getPreReluMode(), rewriter))
      rewriter.create<pto::SetFpcOp>(loc, fpc);
    Value originalCtrl =
        configureAccStoreCtrl(loc, /*allowAtomic=*/false, std::nullopt,
                              std::nullopt, op.getSatMode(), rewriter);
    pto::DmaLoopConfig hwLoop{one, zero, zero};
    if (Value loop3Count = op.getLoop3Count()) {
      hwLoop = {loop3Count, op.getLoop3SrcStride(), op.getLoop3DstStride()};
    }

    Value channelLoop0Stride = zero;
    Value nz2ndEn = zero;
    Value channelSplitEn = zero;
    Value nz2dnEn = zero;
    if (auto mode = op.getMode()) {
      switch (*mode) {
      case pto::AccStoreMode::Nz2nd:
        nz2ndEn = one;
        break;
      case pto::AccStoreMode::Nz2dn:
        nz2dnEn = one;
        channelLoop0Stride = op.getLoop0SrcStride() ? op.getLoop0SrcStride() : one;
        break;
      case pto::AccStoreMode::Nz2nz:
        channelSplitEn = op.getSplit() ? op.getSplit() : zero;
        break;
      }
    } else {
      nz2ndEn = one;
    }

    Value loop3Config = packLoop3Config(loc, hwLoop.count, hwLoop.srcStride,
                                        hwLoop.dstStride, rewriter);
    Value channelConfig =
        packChannelConfig(loc, channelLoop0Stride, rewriter);
    rewriter.create<pto::SetLoop3ParaOp>(
        loc, extractConfigLow40(loc, loop3Config, rewriter),
        extractConfigHigh24(loc, loop3Config, rewriter));
    rewriter.create<pto::SetChannelParaOp>(
        loc, extractConfigLow40(loc, channelConfig, rewriter),
        extractConfigHigh24(loc, channelConfig, rewriter));
    Value clipReluPre = getI64Constant(loc, rewriter, op.getClipValue() ? 1 : 0);
    Value unitFlagCtrl = buildAccStoreOptionalEnumValue(
        loc,
        op.getUnitFlag()
            ? std::optional<uint32_t>(static_cast<uint32_t>(*op.getUnitFlag()))
            : std::nullopt,
        rewriter);
    Value quantPreMode = buildAccStoreOptionalEnumValue(
        loc,
        op.getPreQuantMode()
            ? std::optional<uint32_t>(static_cast<uint32_t>(*op.getPreQuantMode()))
            : std::nullopt,
        rewriter);
    Value reluPreMode = buildAccStoreOptionalEnumValue(
        loc,
        op.getPreReluMode()
            ? std::optional<uint32_t>(static_cast<uint32_t>(*op.getPreReluMode()))
            : std::nullopt,
        rewriter);

    Value dualDstMode =
        getI64Constant(loc, rewriter, static_cast<int64_t>(op.getDstMode()));
    Value subBlockId = op.getSubBlockid() ? op.getSubBlockid() : zero;
    Value config0 = packCopyMatrixCcToGmXm(loc, zero, op.getN(), op.getM(),
                                           op.getDstStride(), rewriter);
    Value config1 = packCopyMatrixCcToUbConfig1(
        loc, op.getSrcStride(), dualDstMode, subBlockId,
        clipReluPre, unitFlagCtrl, quantPreMode, reluPreMode, nz2ndEn,
        channelSplitEn, nz2dnEn, rewriter);
    rewriter.create<pto::CopyMatrixCcToUbOp>(loc, op.getSource(),
                                             op.getDestination(), config0,
                                             config1);
    if (originalCtrl)
      rewriter.create<pto::SetCtrlOp>(loc, originalCtrl);
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
    if (func.isExternal())
      return;

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
                 ExpandSimtLaunchPattern,
                 ExpandMadSemanticPattern<pto::MadOp>,
                 ExpandMadSemanticPattern<pto::MadAccOp>,
                 ExpandMadSemanticPattern<pto::MadBiasOp>,
                 ExpandMadSemanticPattern<pto::MadMxOp>,
                 ExpandMadSemanticPattern<pto::MadMxAccOp>,
                 ExpandMadSemanticPattern<pto::MadMxBiasOp>>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVPTOExpandWrapperOpsPass() {
  return std::make_unique<VPTOExpandWrapperOpsPass>();
}

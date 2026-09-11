// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "VPTOCANN900LLVMEmitterInternal.h"

namespace mlir::pto::detail {

FailureOr<Value> packShiftedFields(Operation *anchor, Value base, ArrayRef<std::pair<Value, uint64_t>> fields) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Value result = castIntegerLikeTo(anchor, base, builder.getI64Type());
  if (!result) {
    return failure();
  }
  for (const auto &[field, shift] : fields) {
    Value value = castIntegerLikeTo(anchor, field, builder.getI64Type());
    if (!value) {
      return failure();
    }
    Value shifted =
        builder.create<arith::ShLIOp>(anchor->getLoc(), value, getI64Constant(builder, anchor->getLoc(), shift));
    result = builder.create<arith::OrIOp>(anchor->getLoc(), result, shifted);
  }
  return result;
}

std::optional<uint64_t> parseLoadX2DistImmediate(StringRef dist, Type elementType) {
  const auto *contract = lookupVPTOMemoryDist(VPTOMemoryOpFamily::LoadX2, dist,
                                              getDistElementWidth(elementType));
  return contract ? std::optional<uint64_t>(contract->a5Immediate)
                  : std::nullopt;
}

std::optional<uint64_t> parseStoreDistImmediate(StringRef dist, Type elementType) {
  const auto *contract = lookupVPTOMemoryDist(
      VPTOMemoryOpFamily::Store, dist,
      dist.empty() ? getDistElementWidth(elementType) : std::nullopt);
  return contract ? std::optional<uint64_t>(contract->a5Immediate)
                  : std::nullopt;
}

bool isMaskOnlyUsedByOnePointStores(Value mask) {
  return !mask.use_empty() && llvm::all_of(mask.getUsers(), [](Operation *user) {
    auto store = dyn_cast<pto::VstsOp>(user);
    return store && store.getDist() && isOnePointStoreDist(*store.getDist());
  });
}

std::optional<uint64_t> parseStoreX2DistImmediate(StringRef dist, Type) {
  const auto *contract =
      lookupVPTOMemoryDist(VPTOMemoryOpFamily::StoreX2, dist);
  return contract ? std::optional<uint64_t>(contract->a5Immediate)
                  : std::nullopt;
}

Value packBlockRepeatStride(Operation *anchor, Value blockStride, Value repeatStride) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);

  Value blockI32 = castIntegerLikeTo(anchor, blockStride, builder.getI32Type());
  Value repeatI32 = castIntegerLikeTo(anchor, repeatStride, builder.getI32Type());
  if (!blockI32 || !repeatI32) {
    return {};
  }

  auto c16 = builder.create<arith::ConstantIntOp>(anchor->getLoc(), kBlockRepeatStrideBlockShift, 32);
  auto blockShifted = builder.create<arith::ShLIOp>(anchor->getLoc(), blockI32, c16);
  return builder.create<arith::OrIOp>(anchor->getLoc(), blockShifted, repeatI32).getResult();
}

std::optional<uint64_t> parseOrderImmediate(StringRef order) {
  if (order.empty() || order == "ASC") {
    return 0;
  }
  if (order == "DESC") {
    return 1;
  }
  return std::nullopt;
}

FailureOr<Value> packCopyGmToUbConfig0(Operation *anchor, ValueRange operands) {
  if (operands.size() != kGmToUbConfig0OperandCount) {
    return failure();
  }

  SmallVector<std::pair<Value, uint64_t>, 6> fields = {
      {operands[3], kGmToUbBurstNumShift},
      {operands[4], kGmToUbBurstLenShift},
      {operands[5], kGmToUbLeftPaddingShift},
      {operands[6], kGmToUbRightPaddingShift},
      {operands[7], kGmToUbDataSelectShift},
      {operands[8], kGmToUbCacheCtlShift}};
  return packShiftedFields(anchor, operands[2], fields);
}

FailureOr<Value> packCopyGmToUbConfig1(Operation *anchor, ValueRange operands) {
  if (operands.size() != kGmToUbConfig0OperandCount) {
    return failure();
  }
  return packLoopPair(anchor, operands[9], operands[10]);
}

[[maybe_unused]] FailureOr<Value> packCopyGmToUbConfig0(Operation *anchor, Value sid, Value nBurst, Value lenBurst,
                                                        Value leftPadding, Value rightPadding, Value dataSelect,
                                                        Value cacheCtl) {
  SmallVector<Value, kGmToUbConfig0OperandCount> operands(kGmToUbConfig0OperandCount);
  operands[2] = sid;
  operands[3] = nBurst;
  operands[4] = lenBurst;
  operands[5] = leftPadding;
  operands[6] = rightPadding;
  operands[7] = dataSelect;
  operands[8] = cacheCtl;
  return packCopyGmToUbConfig0(anchor, operands);
}

FailureOr<Value> packCopyUbToGmConfig0(Operation *anchor, ValueRange operands) {
  if (operands.size() != kUbToGmConfig0OperandCount) {
    return failure();
  }
  SmallVector<std::pair<Value, uint64_t>, 3> fields = {
      {operands[3], kUbToGmBurstNumShift}, {operands[4], kUbToGmBurstLenShift},
      {operands[5], kUbToGmL2CacheCtrlShift}};
  return packShiftedFields(anchor, operands[2], fields);
}

FailureOr<Value> packCopyUbToGmConfig1(Operation *anchor, ValueRange operands) {
  if (operands.size() != kUbToGmConfig0OperandCount) {
    return failure();
  }
  return packLoopPair(anchor, operands[6], operands[7]);
}

[[maybe_unused]] FailureOr<Value> packCopyUbToGmConfig0(Operation *anchor, Value sid, Value nBurst, Value lenBurst,
                                                        Value l2CacheCtl) {
  SmallVector<Value, kUbToGmConfig0OperandCount> operands(kUbToGmConfig0OperandCount);
  operands[2] = sid;
  operands[3] = nBurst;
  operands[4] = lenBurst;
  operands[5] = l2CacheCtl;
  return packCopyUbToGmConfig0(anchor, operands);
}

FailureOr<Value> packCopyUbToUbConfig(Operation *anchor, ValueRange operands) {
  if (operands.size() != kUbToUbConfigOperandCount) {
    return failure();
  }
  SmallVector<std::pair<Value, uint64_t>, 3> fields = {
      {operands[4], kUbToUbNBurstShift}, {operands[5], kUbToUbLenBurstShift},
      {operands[6], kUbToUbDstGapShift}};
  return packShiftedFields(anchor, operands[3], fields);
}

FailureOr<Value> packCopyCbufToUbConfig(Operation *anchor, ValueRange operands) {
  if (operands.size() != kCbufToUbConfigOperandCount) {
    return failure();
  }
  SmallVector<std::pair<Value, uint64_t>, 4> fields = {
      {operands[3], kCbufToUbNBurstShift}, {operands[4], kCbufToUbLenBurstShift},
      {operands[5], kCbufToUbSrcGapShift}, {operands[6], kCbufToUbDstGapShift}};
  return packShiftedFields(anchor, operands[2], fields);
}

FailureOr<Value> packCopyUbToCbufConfig(Operation *anchor, ValueRange operands) {
  if (operands.size() != kCbufToUbConfigOperandCount) {
    return failure();
  }
  return packCopyCbufToUbConfig(anchor, operands);
}

namespace {
// Shared plumbing for the copy/load config packers: an insertion-point
// builder plus the i64 operand conversion and shifted-field composition
// helpers shared by every config layout.
class ConfigPacker {
public:
  explicit ConfigPacker(Operation *anchor) : anchor(anchor), builder(anchor), loc(anchor->getLoc()) {
    builder.setInsertionPoint(anchor);
  }

  Value i64Constant(uint64_t value) { return getI64Constant(builder, loc, value); }

  Value i64Operand(ValueRange operands, unsigned idx) {
    return castIntegerLikeTo(anchor, operands[idx], builder.getI64Type());
  }

  Value i64Operand(Value operand) { return castIntegerLikeTo(anchor, operand, builder.getI64Type()); }

  Value shl(Value value, uint64_t amount) {
    return builder.create<arith::ShLIOp>(loc, value, i64Constant(amount));
  }

  Value bitOr(Value lhs, Value rhs) { return builder.create<arith::OrIOp>(loc, lhs, rhs); }

private:
  Operation *anchor;
  OpBuilder builder;
  Location loc;
};
} // namespace

FailureOr<Value> packCopyGmToCbufConfig0(Operation *anchor, Value nBurst, Value lenBurst) {
  ConfigPacker packer(anchor);
  Value nBurstI64 = packer.i64Operand(nBurst);
  Value lenBurstI64 = packer.i64Operand(lenBurst);
  if (!nBurstI64 || !lenBurstI64) {
    return failure();
  }
  Value config0 = packer.i64Constant(0); // sid
  config0 = packer.bitOr(config0, packer.shl(nBurstI64, kGmToCbufBurstNumShift));     // burst_num[24:4]
  config0 = packer.bitOr(config0, packer.shl(lenBurstI64, kGmToCbufBurstLenShift));  // burst_len[45:25]
  return config0;
}

// Packs a (srcStride, dstStride) pair with the dst field shifted by dstShift;
// shared by the copy_gm_to_cbuf config1 and load_cbuf_to_* config1 layouts.
static FailureOr<Value> packSrcDstStrides(Operation *anchor, Value srcStride, Value dstStride, uint64_t dstShift) {
  ConfigPacker packer(anchor);
  Value srcStrideI64 = packer.i64Operand(srcStride);
  Value dstStrideI64 = packer.i64Operand(dstStride);
  if (!srcStrideI64 || !dstStrideI64) {
    return failure();
  }
  return packer.bitOr(srcStrideI64, packer.shl(dstStrideI64, dstShift));
}

FailureOr<Value> packCopyGmToCbufConfig1(Operation *anchor, Value srcStride, Value dstStride) {
  // config1 packs burst_src_stride[39:0] and burst_dst_stride[60:40].
  return packSrcDstStrides(anchor, srcStride, dstStride, kGmToCbufDstStrideShift);
}

FailureOr<Value> packCopyGmToCbufMultiConfig0(Operation *anchor, Value sid, Value loop1SrcStride, Value l2CacheCtl,
                                              Value nValue) {
  ConfigPacker packer(anchor);
  Value sidI64 = packer.i64Operand(sid);
  Value loop1SrcStrideI64 = packer.i64Operand(loop1SrcStride);
  Value l2CacheCtlI64 = packer.i64Operand(l2CacheCtl);
  Value nValueI64 = packer.i64Operand(nValue);
  if (!sidI64 || !loop1SrcStrideI64 || !l2CacheCtlI64 || !nValueI64) {
    return failure();
  }
  Value config0 = sidI64;
  config0 = packer.bitOr(config0, packer.shl(loop1SrcStrideI64, kGmToCbufMultiLoop1SrcStrideShift));
  config0 = packer.bitOr(config0, packer.shl(l2CacheCtlI64, kGmToCbufMultiL2CacheCtrlShift));
  config0 = packer.bitOr(config0, packer.shl(nValueI64, kGmToCbufMultiNValueShift));
  return config0;
}

FailureOr<Value> packCopyGmToCbufMultiConfig1(Operation *anchor, Value dValue, Value loop4SrcStride, Value smallC0En) {
  ConfigPacker packer(anchor);
  Value dValueI64 = packer.i64Operand(dValue);
  Value loop4SrcStrideI64 = packer.i64Operand(loop4SrcStride);
  Value smallC0EnI64 = packer.i64Operand(smallC0En);
  if (!dValueI64 || !loop4SrcStrideI64 || !smallC0EnI64) {
    return failure();
  }
  Value config1 = dValueI64;
  config1 = packer.bitOr(config1, packer.shl(loop4SrcStrideI64, kGmToCbufMultiLoop4SrcStrideShift));
  config1 = packer.bitOr(config1, packer.shl(smallC0EnI64, kGmToCbufMultiSmallC0EnShift));
  return config1;
}

FailureOr<Value> packCopyCbufToBtConfig(Operation *anchor, Value convControl, Value nBurst, Value lenBurst,
                                        Value sourceGap, Value dstGap) {
  ConfigPacker packer(anchor);
  Value zero = packer.i64Constant(0);
  SmallVector<std::pair<Value, uint64_t>, 5> fields = {
      {convControl, kCbufToBtConvControlShift}, {nBurst, kCbufToBtNBurstShift},
      {lenBurst, kCbufToBtLenBurstShift}, {sourceGap, kCbufToBtSourceGapShift},
      {dstGap, kCbufToBtDstGapShift}};
  return packShiftedFields(anchor, zero, fields);
}

FailureOr<Value> packCopyCbufToFbufConfig(Operation *anchor, Value nBurst, Value lenBurst, Value sourceGap,
                                          Value dstGap) {
  ConfigPacker packer(anchor);
  Value nBurstI64 = packer.i64Operand(nBurst);
  Value lenBurstI64 = packer.i64Operand(lenBurst);
  Value sourceGapI64 = packer.i64Operand(sourceGap);
  Value dstGapI64 = packer.i64Operand(dstGap);
  if (!nBurstI64 || !lenBurstI64 || !sourceGapI64 || !dstGapI64) {
    return failure();
  }
  Value config = packer.shl(nBurstI64, kCbufToFbufNBurstShift);
  config = packer.bitOr(config, packer.shl(lenBurstI64, kCbufToFbufLenBurstShift));
  config = packer.bitOr(config, packer.shl(sourceGapI64, kCbufToFbufSourceGapShift));
  config = packer.bitOr(config, packer.shl(dstGapI64, kCbufToFbufDstGapShift));
  return config;
}

// Packs the (mStart, kStart, mStep, kStep) 2D-load tile descriptor shared by
// the load_cbuf_to_s4/ca/cb config0 layouts.
static FailureOr<Value> packLoadCbufTileConfig0(Operation *anchor, Value mStart, Value kStart, Value mStep,
                                                Value kStep) {
  ConfigPacker packer(anchor);
  Value mStartI64 = packer.i64Operand(mStart);
  Value kStartI64 = packer.i64Operand(kStart);
  Value mStepI64 = packer.i64Operand(mStep);
  Value kStepI64 = packer.i64Operand(kStep);
  if (!mStartI64 || !kStartI64 || !mStepI64 || !kStepI64) {
    return failure();
  }
  Value config0 = mStartI64;
  config0 = packer.bitOr(config0, packer.shl(kStartI64, kLoadCbufKStartShift));
  config0 = packer.bitOr(config0, packer.shl(mStepI64, kLoadCbufMStepShift));
  config0 = packer.bitOr(config0, packer.shl(kStepI64, kLoadCbufKStepShift));
  return config0;
}

FailureOr<Value> packLoadCbufToS4Config0(Operation *anchor, Value mStart, Value kStart, Value mStep, Value kStep) {
  return packLoadCbufTileConfig0(anchor, mStart, kStart, mStep, kStep);
}

FailureOr<Value> packLoadCbufToS4Config1(Operation *anchor, Value srcStride, Value dstStride) {
  return packSrcDstStrides(anchor, srcStride, dstStride, kLoadCbufDstStrideShift);
}

FailureOr<Value> packLoadCbufToCaConfig0(Operation *anchor, Value mStart, Value kStart, Value mStep, Value kStep) {
  return packLoadCbufTileConfig0(anchor, mStart, kStart, mStep, kStep);
}

FailureOr<Value> packLoadCbufToCaConfig1(Operation *anchor, Value srcStride, Value dstStride) {
  return packSrcDstStrides(anchor, srcStride, dstStride, kLoadCbufDstStrideShift);
}

FailureOr<Value> packLoadCbufToCbConfig0(Operation *anchor, Value mStart, Value kStart, Value mStep, Value kStep) {
  return packLoadCbufTileConfig0(anchor, mStart, kStart, mStep, kStep);
}

FailureOr<Value> packLoadCbufToCbConfig1(Operation *anchor, Value srcStride, Value dstStride) {
  return packSrcDstStrides(anchor, srcStride, dstStride, kLoadCbufDstStrideShift);
}

Value buildMadBiasDestination(Operation *anchor, ConversionPatternRewriter &rewriter, Value dst, Value bias) {
  Type i64Ty = rewriter.getI64Type();
  Value dstAddr = rewriter.create<LLVM::PtrToIntOp>(anchor->getLoc(), i64Ty, dst);
  Value biasAddr = rewriter.create<LLVM::PtrToIntOp>(anchor->getLoc(), i64Ty, bias);
  Value lowMask = getI64Constant(rewriter, anchor->getLoc(), kMadBiasAddressMask);
  Value dstLow = rewriter.create<arith::AndIOp>(anchor->getLoc(), dstAddr, lowMask);
  Value biasLow = rewriter.create<arith::AndIOp>(anchor->getLoc(), biasAddr, lowMask);
  Value biasHigh =
      rewriter.create<arith::ShLIOp>(anchor->getLoc(), biasLow, getI64Constant(rewriter, anchor->getLoc(), kMadBiasHighWordShift));
  Value packed = rewriter.create<arith::OrIOp>(anchor->getLoc(), dstLow, biasHigh);
  return rewriter.create<LLVM::IntToPtrOp>(anchor->getLoc(), dst.getType(), packed);
}

FailureOr<Value> packVbitsortConfig(Operation *anchor, Value repeatTimes) {
  ConfigPacker packer(anchor);
  Value repeatI64 = packer.i64Operand(repeatTimes);
  if (!repeatI64) {
    return failure();
  }
  return packer.shl(repeatI64, kBitsortRepeatShift);
}

[[maybe_unused]] FailureOr<Value> materializeDynamicPltMask(ConversionPatternRewriter &rewriter, LoweringState &state,
                                                            Location loc, Value laneCount, Type vectorElemType) {
  Type i32Type = rewriter.getI32Type();
  Value laneCountI32 = laneCount;
  if (laneCountI32.getType() != i32Type) {
    laneCountI32 = castIntegerLikeTo(rewriter.getInsertionBlock()->getParentOp(), laneCountI32, i32Type);
    if (!laneCountI32) {
      return failure();
    }
  }

  StringRef calleeName;
  if (vectorElemType.isF32()) {
    calleeName = StringRef("llvm.hivm.plt.b32.v300");
  } else if (vectorElemType.isF16() || vectorElemType.isBF16()) {
    calleeName = StringRef("llvm.hivm.plt.b16.v300");
  } else if (auto intType = dyn_cast<IntegerType>(vectorElemType)) {
    if (intType.getWidth() == kBits32) {
      calleeName = StringRef("llvm.hivm.plt.b32.v300");
    } else if (intType.getWidth() == kBits16) {
      calleeName = StringRef("llvm.hivm.plt.b16.v300");
    } else if (intType.getWidth() == kBits8) {
      calleeName = StringRef("llvm.hivm.plt.b8.v300");
    }
  }
  if (calleeName.empty()) {
    return failure();
  }

  Type maskType = VectorType::get({kPltMaskVectorLength}, rewriter.getI1Type());
  auto funcType = rewriter.getFunctionType(TypeRange{i32Type}, TypeRange{maskType, i32Type});
  auto call = rewriter.create<func::CallOp>(loc, calleeName, funcType.getResults(), ValueRange{laneCountI32});
  state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
  return call.getResult(0);
}

} // namespace mlir::pto::detail

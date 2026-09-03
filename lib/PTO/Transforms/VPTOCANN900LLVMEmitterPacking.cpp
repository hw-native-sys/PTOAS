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

bool isOnePointStoreDist(StringRef dist) {
  const auto *contract = lookupVPTOMemoryDist(VPTOMemoryOpFamily::Store, dist);
  return contract && contract->isOnePointStore();
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

  auto c16 = builder.create<arith::ConstantIntOp>(anchor->getLoc(), 16, 32);
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

FailureOr<Value> packLoopPair(Operation *anchor, Value low, Value high) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);

  Value lowI64 = castIntegerLikeTo(anchor, low, builder.getI64Type());
  Value highI64 = castIntegerLikeTo(anchor, high, builder.getI64Type());
  if (!lowI64 || !highI64) {
    return failure();
  }

  Value shift = getI64Constant(builder, anchor->getLoc(), 40);
  Value highShifted = builder.create<arith::ShLIOp>(anchor->getLoc(), highI64, shift).getResult();
  return builder.create<arith::OrIOp>(anchor->getLoc(), highShifted, lowI64).getResult();
}

FailureOr<Value> packLoopSize(Operation *anchor, Value loop2, Value loop1) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);

  Value loop2I64 = castIntegerLikeTo(anchor, loop2, builder.getI64Type());
  Value loop1I64 = castIntegerLikeTo(anchor, loop1, builder.getI64Type());
  if (!loop2I64 || !loop1I64) {
    return failure();
  }

  Value shift = getI64Constant(builder, anchor->getLoc(), 21);
  Value loop2Shifted = builder.create<arith::ShLIOp>(anchor->getLoc(), loop2I64, shift).getResult();
  return builder.create<arith::OrIOp>(anchor->getLoc(), loop2Shifted, loop1I64).getResult();
}

FailureOr<Value> packCopyGmToUbConfig0(Operation *anchor, ValueRange operands) {
  if (operands.size() != 11) {
    return failure();
  }

  SmallVector<std::pair<Value, uint64_t>, 6> fields = {{operands[3], 4},  {operands[4], 25}, {operands[5], 46},
                                                       {operands[6], 52}, {operands[7], 58}, {operands[8], 60}};
  return packShiftedFields(anchor, operands[2], fields);
}

FailureOr<Value> packCopyGmToUbConfig1(Operation *anchor, ValueRange operands) {
  if (operands.size() != 11) {
    return failure();
  }
  return packLoopPair(anchor, operands[9], operands[10]);
}

[[maybe_unused]] FailureOr<Value> packCopyGmToUbConfig0(Operation *anchor, Value sid, Value nBurst, Value lenBurst,
                                                        Value leftPadding, Value rightPadding, Value dataSelect,
                                                        Value cacheCtl) {
  SmallVector<Value, 11> operands(11);
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
  if (operands.size() != 8) {
    return failure();
  }

  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  auto getI64Operand = [&](unsigned idx) -> Value {
    return castIntegerLikeTo(anchor, operands[idx], builder.getI64Type());
  };

  Value sid = getI64Operand(2);
  Value nBurst = getI64Operand(3);
  Value lenBurst = getI64Operand(4);
  Value l2CacheCtl = getI64Operand(5);
  if (!sid || !nBurst || !lenBurst || !l2CacheCtl) {
    return failure();
  }

  auto shl = [&](Value value, uint64_t amount) -> Value {
    return builder.create<arith::ShLIOp>(loc, value, getI64Constant(builder, loc, amount));
  };
  auto bitOr = [&](Value lhs, Value rhs) -> Value { return builder.create<arith::OrIOp>(loc, lhs, rhs); };

  Value config = sid;
  config = bitOr(config, shl(nBurst, 4));
  config = bitOr(config, shl(lenBurst, 25));
  config = bitOr(config, shl(l2CacheCtl, 60));
  return config;
}

FailureOr<Value> packCopyUbToGmConfig1(Operation *anchor, ValueRange operands) {
  if (operands.size() != 8) {
    return failure();
  }
  return packLoopPair(anchor, operands[6], operands[7]);
}

[[maybe_unused]] FailureOr<Value> packCopyUbToGmConfig0(Operation *anchor, Value sid, Value nBurst, Value lenBurst,
                                                        Value l2CacheCtl) {
  SmallVector<Value, 8> operands(8);
  operands[2] = sid;
  operands[3] = nBurst;
  operands[4] = lenBurst;
  operands[5] = l2CacheCtl;
  return packCopyUbToGmConfig0(anchor, operands);
}

FailureOr<Value> packCopyUbToUbConfig(Operation *anchor, ValueRange operands) {
  if (operands.size() != 7) {
    return failure();
  }
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  auto getI64Operand = [&](unsigned idx) -> Value {
    return castIntegerLikeTo(anchor, operands[idx], builder.getI64Type());
  };

  Value nBurst = getI64Operand(3);
  Value lenBurst = getI64Operand(4);
  Value srcStride = getI64Operand(5);
  Value dstStride = getI64Operand(6);
  if (!nBurst || !lenBurst || !srcStride || !dstStride) {
    return failure();
  }

  auto shl = [&](Value value, uint64_t amount) -> Value {
    return builder.create<arith::ShLIOp>(loc, value, getI64Constant(builder, loc, amount));
  };
  auto bitOr = [&](Value lhs, Value rhs) -> Value { return builder.create<arith::OrIOp>(loc, lhs, rhs); };

  Value config = nBurst;
  config = bitOr(config, shl(lenBurst, 16));
  config = bitOr(config, shl(srcStride, 32));
  config = bitOr(config, shl(dstStride, 48));
  return config;
}

FailureOr<Value> packCopyCbufToUbConfig(Operation *anchor, ValueRange operands) {
  if (operands.size() != 7) {
    return failure();
  }
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  auto getI64Operand = [&](unsigned idx) -> Value {
    return castIntegerLikeTo(anchor, operands[idx], builder.getI64Type());
  };

  Value sid = getI64Operand(2);
  Value nBurst = getI64Operand(3);
  Value lenBurst = getI64Operand(4);
  Value srcStride = getI64Operand(5);
  Value dstStride = getI64Operand(6);
  if (!sid || !nBurst || !lenBurst || !srcStride || !dstStride) {
    return failure();
  }

  auto shl = [&](Value value, uint64_t amount) -> Value {
    return builder.create<arith::ShLIOp>(loc, value, getI64Constant(builder, loc, amount));
  };
  auto bitOr = [&](Value lhs, Value rhs) -> Value { return builder.create<arith::OrIOp>(loc, lhs, rhs); };

  Value config = sid;
  config = bitOr(config, shl(nBurst, 4));
  config = bitOr(config, shl(lenBurst, 16));
  config = bitOr(config, shl(srcStride, 32));
  config = bitOr(config, shl(dstStride, 48));
  return config;
}

FailureOr<Value> packCopyUbToCbufConfig(Operation *anchor, ValueRange operands) {
  if (operands.size() != 7) {
    return failure();
  }
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  auto getI64Operand = [&](unsigned idx) -> Value {
    return castIntegerLikeTo(anchor, operands[idx], builder.getI64Type());
  };

  Value sid = getI64Operand(2);
  Value nBurst = getI64Operand(3);
  Value lenBurst = getI64Operand(4);
  Value srcStride = getI64Operand(5);
  Value dstStride = getI64Operand(6);
  if (!sid || !nBurst || !lenBurst || !srcStride || !dstStride) {
    return failure();
  }

  auto shl = [&](Value value, uint64_t amount) -> Value {
    return builder.create<arith::ShLIOp>(loc, value, getI64Constant(builder, loc, amount));
  };
  auto bitOr = [&](Value lhs, Value rhs) -> Value { return builder.create<arith::OrIOp>(loc, lhs, rhs); };

  Value config = sid;
  config = bitOr(config, shl(nBurst, 4));
  config = bitOr(config, shl(lenBurst, 16));
  config = bitOr(config, shl(srcStride, 32));
  config = bitOr(config, shl(dstStride, 48));
  return config;
}

FailureOr<Value> packCopyGmToCbufConfig0(Operation *anchor, Value nBurst, Value lenBurst) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  Value nBurstI64 = castIntegerLikeTo(anchor, nBurst, builder.getI64Type());
  Value lenBurstI64 = castIntegerLikeTo(anchor, lenBurst, builder.getI64Type());
  if (!nBurstI64 || !lenBurstI64) {
    return failure();
  }

  auto shl = [&](Value value, uint64_t amount) -> Value {
    return builder.create<arith::ShLIOp>(loc, value, getI64Constant(builder, loc, amount));
  };
  auto bitOr = [&](Value lhs, Value rhs) -> Value { return builder.create<arith::OrIOp>(loc, lhs, rhs); };

  Value config0 = getI64Constant(builder, loc, 0); // sid
  config0 = bitOr(config0, shl(nBurstI64, 4));     // burst_num[24:4]
  config0 = bitOr(config0, shl(lenBurstI64, 25));  // burst_len[45:25]
  return config0;
}

FailureOr<Value> packCopyGmToCbufConfig1(Operation *anchor, Value srcStride, Value dstStride) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  Value srcStrideI64 = castIntegerLikeTo(anchor, srcStride, builder.getI64Type());
  Value dstStrideI64 = castIntegerLikeTo(anchor, dstStride, builder.getI64Type());
  if (!srcStrideI64 || !dstStrideI64) {
    return failure();
  }

  auto shl = [&](Value value, uint64_t amount) -> Value {
    return builder.create<arith::ShLIOp>(loc, value, getI64Constant(builder, loc, amount));
  };
  auto bitOr = [&](Value lhs, Value rhs) -> Value { return builder.create<arith::OrIOp>(loc, lhs, rhs); };

  // config1 packs burst_src_stride[39:0] and burst_dst_stride[60:40].
  return bitOr(srcStrideI64, shl(dstStrideI64, 40));
}

FailureOr<Value> packCopyGmToCbufMultiConfig0(Operation *anchor, Value sid, Value loop1SrcStride, Value l2CacheCtl,
                                              Value nValue) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  Value sidI64 = castIntegerLikeTo(anchor, sid, builder.getI64Type());
  Value loop1SrcStrideI64 = castIntegerLikeTo(anchor, loop1SrcStride, builder.getI64Type());
  Value l2CacheCtlI64 = castIntegerLikeTo(anchor, l2CacheCtl, builder.getI64Type());
  Value nValueI64 = castIntegerLikeTo(anchor, nValue, builder.getI64Type());
  if (!sidI64 || !loop1SrcStrideI64 || !l2CacheCtlI64 || !nValueI64) {
    return failure();
  }

  auto shl = [&](Value value, uint64_t amount) -> Value {
    return builder.create<arith::ShLIOp>(loc, value, getI64Constant(builder, loc, amount));
  };
  auto bitOr = [&](Value lhs, Value rhs) -> Value { return builder.create<arith::OrIOp>(loc, lhs, rhs); };

  Value config0 = sidI64;
  config0 = bitOr(config0, shl(loop1SrcStrideI64, 4));
  config0 = bitOr(config0, shl(l2CacheCtlI64, 44));
  config0 = bitOr(config0, shl(nValueI64, 48));
  return config0;
}

FailureOr<Value> packCopyGmToCbufMultiConfig1(Operation *anchor, Value dValue, Value loop4SrcStride, Value smallC0En) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  Value dValueI64 = castIntegerLikeTo(anchor, dValue, builder.getI64Type());
  Value loop4SrcStrideI64 = castIntegerLikeTo(anchor, loop4SrcStride, builder.getI64Type());
  Value smallC0EnI64 = castIntegerLikeTo(anchor, smallC0En, builder.getI64Type());
  if (!dValueI64 || !loop4SrcStrideI64 || !smallC0EnI64) {
    return failure();
  }

  auto shl = [&](Value value, uint64_t amount) -> Value {
    return builder.create<arith::ShLIOp>(loc, value, getI64Constant(builder, loc, amount));
  };
  auto bitOr = [&](Value lhs, Value rhs) -> Value { return builder.create<arith::OrIOp>(loc, lhs, rhs); };

  Value config1 = dValueI64;
  config1 = bitOr(config1, shl(loop4SrcStrideI64, 21));
  config1 = bitOr(config1, shl(smallC0EnI64, 61));
  return config1;
}

FailureOr<Value> packCopyCbufToBtConfig(Operation *anchor, Value convControl, Value nBurst, Value lenBurst,
                                        Value sourceGap, Value dstGap) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Value zero = getI64Constant(builder, anchor->getLoc(), 0);
  SmallVector<std::pair<Value, uint64_t>, 5> fields = {
      {convControl, 3}, {nBurst, 4}, {lenBurst, 16}, {sourceGap, 32}, {dstGap, 48}};
  return packShiftedFields(anchor, zero, fields);
}

FailureOr<Value> packCopyCbufToFbufConfig(Operation *anchor, Value nBurst, Value lenBurst, Value sourceGap,
                                          Value dstGap) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  Value nBurstI64 = castIntegerLikeTo(anchor, nBurst, builder.getI64Type());
  Value lenBurstI64 = castIntegerLikeTo(anchor, lenBurst, builder.getI64Type());
  Value sourceGapI64 = castIntegerLikeTo(anchor, sourceGap, builder.getI64Type());
  Value dstGapI64 = castIntegerLikeTo(anchor, dstGap, builder.getI64Type());
  if (!nBurstI64 || !lenBurstI64 || !sourceGapI64 || !dstGapI64) {
    return failure();
  }

  auto shl = [&](Value value, uint64_t amount) -> Value {
    return builder.create<arith::ShLIOp>(loc, value, getI64Constant(builder, loc, amount));
  };
  auto bitOr = [&](Value lhs, Value rhs) -> Value { return builder.create<arith::OrIOp>(loc, lhs, rhs); };

  Value config = shl(nBurstI64, 4);
  config = bitOr(config, shl(lenBurstI64, 16));
  config = bitOr(config, shl(sourceGapI64, 32));
  config = bitOr(config, shl(dstGapI64, 48));
  return config;
}

FailureOr<Value> packLoadCbufToS4Config0(Operation *anchor, Value mStart, Value kStart, Value mStep, Value kStep) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  Value mStartI64 = castIntegerLikeTo(anchor, mStart, builder.getI64Type());
  Value kStartI64 = castIntegerLikeTo(anchor, kStart, builder.getI64Type());
  Value mStepI64 = castIntegerLikeTo(anchor, mStep, builder.getI64Type());
  Value kStepI64 = castIntegerLikeTo(anchor, kStep, builder.getI64Type());
  if (!mStartI64 || !kStartI64 || !mStepI64 || !kStepI64) {
    return failure();
  }

  auto shl = [&](Value value, uint64_t amount) -> Value {
    return builder.create<arith::ShLIOp>(loc, value, getI64Constant(builder, loc, amount));
  };
  auto bitOr = [&](Value lhs, Value rhs) -> Value { return builder.create<arith::OrIOp>(loc, lhs, rhs); };

  Value config0 = mStartI64;
  config0 = bitOr(config0, shl(kStartI64, 16));
  config0 = bitOr(config0, shl(mStepI64, 32));
  config0 = bitOr(config0, shl(kStepI64, 40));
  return config0;
}

FailureOr<Value> packLoadCbufToS4Config1(Operation *anchor, Value srcStride, Value dstStride) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  Value srcStrideI64 = castIntegerLikeTo(anchor, srcStride, builder.getI64Type());
  Value dstStrideI64 = castIntegerLikeTo(anchor, dstStride, builder.getI64Type());
  if (!srcStrideI64 || !dstStrideI64) {
    return failure();
  }

  auto shl = [&](Value value, uint64_t amount) -> Value {
    return builder.create<arith::ShLIOp>(loc, value, getI64Constant(builder, loc, amount));
  };
  return builder.create<arith::OrIOp>(loc, srcStrideI64, shl(dstStrideI64, 16)).getResult();
}

FailureOr<Value> packLoadCbufToCaConfig0(Operation *anchor, Value mStart, Value kStart, Value mStep, Value kStep) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  Value mStartI64 = castIntegerLikeTo(anchor, mStart, builder.getI64Type());
  Value kStartI64 = castIntegerLikeTo(anchor, kStart, builder.getI64Type());
  Value mStepI64 = castIntegerLikeTo(anchor, mStep, builder.getI64Type());
  Value kStepI64 = castIntegerLikeTo(anchor, kStep, builder.getI64Type());
  if (!mStartI64 || !kStartI64 || !mStepI64 || !kStepI64) {
    return failure();
  }

  auto shl = [&](Value value, uint64_t amount) -> Value {
    return builder.create<arith::ShLIOp>(loc, value, getI64Constant(builder, loc, amount));
  };
  auto bitOr = [&](Value lhs, Value rhs) -> Value { return builder.create<arith::OrIOp>(loc, lhs, rhs); };

  Value config0 = mStartI64;
  config0 = bitOr(config0, shl(kStartI64, 16));
  config0 = bitOr(config0, shl(mStepI64, 32));
  config0 = bitOr(config0, shl(kStepI64, 40));
  return config0;
}

FailureOr<Value> packLoadCbufToCaConfig1(Operation *anchor, Value srcStride, Value dstStride) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  Value srcStrideI64 = castIntegerLikeTo(anchor, srcStride, builder.getI64Type());
  Value dstStrideI64 = castIntegerLikeTo(anchor, dstStride, builder.getI64Type());
  if (!srcStrideI64 || !dstStrideI64) {
    return failure();
  }

  auto shl = [&](Value value, uint64_t amount) -> Value {
    return builder.create<arith::ShLIOp>(loc, value, getI64Constant(builder, loc, amount));
  };
  return builder.create<arith::OrIOp>(loc, srcStrideI64, shl(dstStrideI64, 16)).getResult();
}

FailureOr<Value> packLoadCbufToCbConfig0(Operation *anchor, Value mStart, Value kStart, Value mStep, Value kStep) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  Value mStartI64 = castIntegerLikeTo(anchor, mStart, builder.getI64Type());
  Value kStartI64 = castIntegerLikeTo(anchor, kStart, builder.getI64Type());
  Value mStepI64 = castIntegerLikeTo(anchor, mStep, builder.getI64Type());
  Value kStepI64 = castIntegerLikeTo(anchor, kStep, builder.getI64Type());
  if (!mStartI64 || !kStartI64 || !mStepI64 || !kStepI64) {
    return failure();
  }

  auto shl = [&](Value value, uint64_t amount) -> Value {
    return builder.create<arith::ShLIOp>(loc, value, getI64Constant(builder, loc, amount));
  };
  auto bitOr = [&](Value lhs, Value rhs) -> Value { return builder.create<arith::OrIOp>(loc, lhs, rhs); };

  Value config0 = mStartI64;
  config0 = bitOr(config0, shl(kStartI64, 16));
  config0 = bitOr(config0, shl(mStepI64, 32));
  config0 = bitOr(config0, shl(kStepI64, 40));
  return config0;
}

FailureOr<Value> packLoadCbufToCbConfig1(Operation *anchor, Value srcStride, Value dstStride) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  Value srcStrideI64 = castIntegerLikeTo(anchor, srcStride, builder.getI64Type());
  Value dstStrideI64 = castIntegerLikeTo(anchor, dstStride, builder.getI64Type());
  if (!srcStrideI64 || !dstStrideI64) {
    return failure();
  }

  auto shl = [&](Value value, uint64_t amount) -> Value {
    return builder.create<arith::ShLIOp>(loc, value, getI64Constant(builder, loc, amount));
  };
  return builder.create<arith::OrIOp>(loc, srcStrideI64, shl(dstStrideI64, 16)).getResult();
}

Value buildMadBiasDestination(Operation *anchor, ConversionPatternRewriter &rewriter, Value dst, Value bias) {
  Type i64Ty = rewriter.getI64Type();
  Value dstAddr = rewriter.create<LLVM::PtrToIntOp>(anchor->getLoc(), i64Ty, dst);
  Value biasAddr = rewriter.create<LLVM::PtrToIntOp>(anchor->getLoc(), i64Ty, bias);
  Value lowMask = getI64Constant(rewriter, anchor->getLoc(), 0xffffffffULL);
  Value dstLow = rewriter.create<arith::AndIOp>(anchor->getLoc(), dstAddr, lowMask);
  Value biasLow = rewriter.create<arith::AndIOp>(anchor->getLoc(), biasAddr, lowMask);
  Value biasHigh =
      rewriter.create<arith::ShLIOp>(anchor->getLoc(), biasLow, getI64Constant(rewriter, anchor->getLoc(), 32));
  Value packed = rewriter.create<arith::OrIOp>(anchor->getLoc(), dstLow, biasHigh);
  return rewriter.create<LLVM::IntToPtrOp>(anchor->getLoc(), dst.getType(), packed);
}

FailureOr<Value> packVbitsortConfig(Operation *anchor, Value repeatTimes) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  Value repeatI64 = castIntegerLikeTo(anchor, repeatTimes, builder.getI64Type());
  if (!repeatI64) {
    return failure();
  }
  return builder.create<arith::ShLIOp>(loc, repeatI64, getI64Constant(builder, loc, 56)).getResult();
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
    if (intType.getWidth() == 32) {
      calleeName = StringRef("llvm.hivm.plt.b32.v300");
    } else if (intType.getWidth() == 16) {
      calleeName = StringRef("llvm.hivm.plt.b16.v300");
    } else if (intType.getWidth() == 8) {
      calleeName = StringRef("llvm.hivm.plt.b8.v300");
    }
  }
  if (calleeName.empty()) {
    return failure();
  }

  Type maskType = VectorType::get({256}, rewriter.getI1Type());
  auto funcType = rewriter.getFunctionType(TypeRange{i32Type}, TypeRange{maskType, i32Type});
  auto call = rewriter.create<func::CallOp>(loc, calleeName, funcType.getResults(), ValueRange{laneCountI32});
  state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
  return call.getResult(0);
}

} // namespace mlir::pto::detail

// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- ReinterpretCast.cpp - Memref ReinterpretCast op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "../Tile/TileInternal.h"
#include "MemrefInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct ReinterpretCastToEmitC : public OpConversionPattern<memref::ReinterpretCastOp> {
  using OpConversionPattern<memref::ReinterpretCastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(memref::ReinterpretCastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resMrTy = dyn_cast<MemRefType>(op.getType());
    if (!resMrTy)
      return failure();

    auto asAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(resMrTy.getMemorySpace());
    const bool isGm = (!asAttr || asAttr.getAddressSpace() == pto::AddressSpace::GM);

    // GM: keep pointer arithmetic.
    if (isGm)
      return emitGmReinterpretCast(op, adaptor, rewriter);

    // UB/L1/L0 tiles: materialize a new Tile view by assigning an adjusted
    // underlying pointer (in elements).
    return emitTileReinterpretCast(op, adaptor, rewriter, resMrTy, asAttr);
  }

  // GM lowering: fold into pointer arithmetic (emitc.add) plus an optional
  // PTOAS__ADDPTR_TRACE call.
  LogicalResult emitGmReinterpretCast(memref::ReinterpretCastOp op,
                                      OpAdaptor adaptor,
                                      ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    bool emitAddPtrTrace = op->hasAttr("pto.addptr_trace");
    Value source = adaptor.getSource();
    auto offsets = adaptor.getOffsets();
    Value offsetVal = offsets.empty() ? Value() : offsets[0];
    auto mixedOffsets = op.getMixedOffsets();
    std::optional<int64_t> constantOffset =
        mixedOffsets.empty() ? std::nullopt
                             : getConstantIntValue(mixedOffsets.front());
    const bool isZeroOffset = constantOffset && *constantOffset == 0;

    if (!offsetVal || (isZeroOffset && !emitAddPtrTrace)) {
      rewriter.replaceOp(op, source);
      return success();
    }

    Type resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType)
      return failure();

    auto addOp = rewriter.create<emitc::AddOp>(loc, resultType, source, offsetVal);
    if (emitAddPtrTrace) {
      rewriter.setInsertionPointAfter(addOp);
      rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{}, "PTOAS__ADDPTR_TRACE",
          ArrayAttr{}, ArrayAttr{},
          ValueRange{addOp.getResult(), source, offsetVal});
    }
    rewriter.replaceOp(op, addOp.getResult());
    return success();
  }

  // UB/L1/L0 tile lowering: build the tile Variable, compute the adjusted
  // base address, and bind it with TASSIGN.
  LogicalResult emitTileReinterpretCast(
      memref::ReinterpretCastOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter, MemRefType resMrTy,
      pto::AddressSpaceAttr asAttr) const {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    Value source = adaptor.getSource();
    auto offsets = adaptor.getOffsets();
    Value offsetVal = offsets.empty() ? Value() : offsets[0];
    auto mixedOffsets = op.getMixedOffsets();
    std::optional<int64_t> constantOffset =
        mixedOffsets.empty() ? std::nullopt
                             : getConstantIntValue(mixedOffsets.front());
    const bool isZeroOffset = constantOffset && *constantOffset == 0;

    pto::AddressSpace as = asAttr.getAddressSpace();

    // Element type token.
    Type elemTy = resMrTy.getElementType();
    std::string elemTok = getEmitCScalarTypeToken(elemTy);
    int64_t elemBytes = getEmitCScalarByteWidth(elemTy);

    const char *roleTok = reinterpretCastTileRole(as, source);
    std::string tileTypeStr =
        buildReinterpretCastTileTypeString(resMrTy, elemTy, roleTok);

    auto tileType = emitc::OpaqueType::get(ctx, tileTypeStr);
    Value tile = rewriter
                     .create<emitc::VariableOp>(loc,
                                                getEmitCVariableResultType(tileType),
                                                emitc::OpaqueAttr::get(ctx, ""))
                     .getResult();
    tile = loadEmitCVariableIfNeeded(rewriter, loc, tile);

    auto u64Ty = emitc::OpaqueType::get(ctx, "uint64_t");
    Value baseAddr = reinterpretCastBaseAddress(rewriter, loc, source, as,
                                               elemTok, u64Ty);

    Value addr = baseAddr;
    if (offsetVal && !isZeroOffset) {
      Value offU64 = offsetVal;
      if (offU64.getType() != u64Ty)
        offU64 = rewriter.create<emitc::CastOp>(loc, u64Ty, offU64).getResult();

      auto bytesAttr = emitc::OpaqueAttr::get(ctx, std::to_string(elemBytes));
      Value bytesVal = rewriter.create<emitc::ConstantOp>(loc, u64Ty, bytesAttr);
      Value byteOff = rewriter.create<emitc::MulOp>(loc, u64Ty, offU64, bytesVal);
      addr = rewriter.create<emitc::AddOp>(loc, u64Ty, baseAddr, byteOff);
    }

    rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "TASSIGN",
                                         /*args=*/ArrayAttr{},
                                         /*templateArgs=*/ArrayAttr{},
                                         /*operands=*/ValueRange{tile, addr});

    rewriter.replaceOp(op, tile);
    return success();
  }
};

struct MemRefCastToEmitC : public OpConversionPattern<memref::CastOp> {
  using OpConversionPattern<memref::CastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(memref::CastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getSource());
    return success();
  }
};

void populateMemrefReinterpretCastPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<MemRefCastToEmitC>(typeConverter, ctx);
  patterns.add<ReinterpretCastToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir

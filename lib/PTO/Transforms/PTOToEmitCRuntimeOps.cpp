// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOToEmitCRuntimeOps.cpp -------------------------------------------===//
//===----------------------------------------------------------------------===//

#include "PTOToEmitCInternal.h"

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"

#include <string>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::pto;

namespace mlir::pto {
namespace {

static constexpr unsigned kPTOIndexBitWidth = 32;
static constexpr int64_t kPTOTileSplitNone = 0;
static constexpr int64_t kPTOTileSplitUpDown = 1;
static constexpr int64_t kPTOTileSplitLeftRight = 2;
static constexpr int8_t kPTOFrontendDirMaskC2V = 1;
static constexpr int8_t kPTOFrontendDirMaskV2C = 2;
static constexpr int8_t kPTOFrontendDirMaskBidirectional = 3;
static constexpr int32_t kPTOFrontendLocalSlotNum = 2;
static constexpr int64_t kNumber2 = 2;

static FailureOr<std::string> getTileSplitToken(int64_t split) {
  switch (split) {
  case kPTOTileSplitNone:
    return std::string("TileSplitAxis::TILE_NO_SPLIT");
  case kPTOTileSplitUpDown:
    return std::string("TileSplitAxis::TILE_UP_DOWN");
  case kPTOTileSplitLeftRight:
    return std::string("TileSplitAxis::TILE_LEFT_RIGHT");
  default:
    return failure();
  }
}

static FailureOr<std::string>
getTPipeDirectionToken(bool isL2G2L, int8_t dirMask, PTOArch targetArch) {
  if (dirMask == kPTOFrontendDirMaskC2V) {
    if (isL2G2L && targetArch == PTOArch::A5)
      return std::string("Direction::DIR_C2V_GM");
    return std::string("Direction::DIR_C2V");
  }
  if (dirMask == kPTOFrontendDirMaskV2C) {
    if (isL2G2L && targetArch == PTOArch::A5)
      return std::string("Direction::DIR_V2C_GM");
    return std::string("Direction::DIR_V2C");
  }
  if (dirMask == kPTOFrontendDirMaskBidirectional)
    return std::string("Direction::DIR_BOTH");
  return failure();
}

static std::string buildTPipeToken(int32_t flagBase, llvm::StringRef dirTok,
                                   int32_t slotSize, int32_t slotNum,
                                   int32_t localSlotNum, bool nosplit) {
  std::string token = "TPipe<" + std::to_string(flagBase) + ", " + dirTok.str() +
                      ", " + std::to_string(slotSize) + ", " +
                      std::to_string(slotNum);
  token += ", " + std::to_string(localSlotNum);
  token += nosplit ? ", true" : ", false";
  token += ">";
  return token;
}

} // namespace

FailureOr<std::string> buildTPipeTokenFromInitOp(Operation *op,
                                                        PTOArch targetArch) {
  if (auto initOp = dyn_cast<pto::InitializeL2G2LPipeOp>(op)) {
    if (!initOp.getFlagBaseAttr())
      return failure();
    auto dirTok =
        getTPipeDirectionToken(/*isL2G2L=*/true, initOp.getDirMask(), targetArch);
    if (failed(dirTok))
      return failure();
    int32_t localSlotNum = initOp.getLocalSlotNumAttr()
                               ? initOp.getLocalSlotNumAttr().getInt()
                               : initOp.getSlotNum();
    return buildTPipeToken(initOp.getFlagBaseAttr().getInt(), *dirTok,
                           initOp.getSlotSize(), initOp.getSlotNum(),
                           localSlotNum,
                           initOp.getNosplitAttr() &&
                               initOp.getNosplitAttr().getValue());
  }

  if (auto initOp = dyn_cast<pto::InitializeL2LPipeOp>(op)) {
    if (!initOp.getFlagBaseAttr())
      return failure();
    auto dirTok =
        getTPipeDirectionToken(/*isL2G2L=*/false, initOp.getDirMask(), targetArch);
    if (failed(dirTok))
      return failure();
    return buildTPipeToken(initOp.getFlagBaseAttr().getInt(), *dirTok,
                           initOp.getSlotSize(), initOp.getSlotNum(),
                           kPTOFrontendLocalSlotNum,
                           initOp.getNosplitAttr() &&
                               initOp.getNosplitAttr().getValue());
  }

  return failure();
}


namespace {

static FailureOr<std::string> getTPipeTokenFromValue(Value pipeHandle,
                                                     PTOArch targetArch) {
  pipeHandle = peelUnrealized(pipeHandle);
  Operation *def = pipeHandle.getDefiningOp();
  if (!def)
    return failure();
  return buildTPipeTokenFromInitOp(def, targetArch);
}

static FailureOr<std::string> getPipeDataTypeToken(Value value) {
  auto opaqueTy = dyn_cast<emitc::OpaqueType>(value.getType());
  if (!opaqueTy)
    return failure();
  StringRef token = opaqueTy.getValue();
  if (!token.contains("Tile<") && !token.contains("GlobalTensor<"))
    return failure();
  return token.str();
}

struct PTOTAllocToEmitC : public OpConversionPattern<mlir::pto::TAllocOp> {
  PTOTAllocToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                   PTOArch targetArch)
      : OpConversionPattern<mlir::pto::TAllocOp>(typeConverter, ctx),
        targetArch(targetArch) {}

  LogicalResult matchAndRewrite(mlir::pto::TAllocOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto pipeTok = getTPipeTokenFromValue(op.getPipeHandle(), targetArch);
    if (failed(pipeTok))
      return rewriter.notifyMatchFailure(op, "failed to resolve pipe token");
    Value entry = peelUnrealized(adaptor.getEntry());
    auto entryTok = getPipeDataTypeToken(entry);
    if (failed(entryTok))
      return rewriter.notifyMatchFailure(op, "failed to resolve entry token");
    auto splitTok = getTileSplitToken(op.getSplit());
    if (failed(splitTok))
      return rewriter.notifyMatchFailure(op, "failed to resolve split token");

    std::string callee =
        "TALLOC<" + *pipeTok + ", " + *entryTok + ", " + *splitTok + ">";
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, callee, ArrayAttr{}, ArrayAttr{},
        ValueRange{peelUnrealized(adaptor.getPipeHandle()), entry});
    return success();
  }

  PTOArch targetArch;
};

struct PTOTPushToEmitC : public OpConversionPattern<mlir::pto::TPushOp> {
  PTOTPushToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                  PTOArch targetArch)
      : OpConversionPattern<mlir::pto::TPushOp>(typeConverter, ctx),
        targetArch(targetArch) {}

  struct PipeTileRuntimeCall {
    std::string callee;
    Value pipeHandle;
    Value tile;
  };

  template <typename PipeTileOp, typename AdaptorT>
static FailureOr<PipeTileRuntimeCall> buildPipeTileRuntimeCall(
    PipeTileOp op, AdaptorT adaptor, PTOArch targetArch, StringRef calleeBase) {
    auto pipeTok = getTPipeTokenFromValue(op.getPipeHandle(), targetArch);
    if (failed(pipeTok))
      return failure();
    Value convertedTile = peelUnrealized(adaptor.getTile());
    auto tileTok = getPipeDataTypeToken(convertedTile);
    if (failed(tileTok))
      return failure();
    auto splitTok = getTileSplitToken(op.getSplit());
    if (failed(splitTok))
      return failure();

    return PipeTileRuntimeCall{
        (Twine(calleeBase) + "<" + *pipeTok + ", " + *tileTok + ", " +
         *splitTok + ">")
            .str(),
        peelUnrealized(adaptor.getPipeHandle()), convertedTile};
  }

  static LogicalResult lowerPipeTileRuntimeCall(
      Operation *op, const PipeTileRuntimeCall &call,
      ConversionPatternRewriter &rewriter) {
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, call.callee, ArrayAttr{}, ArrayAttr{},
        ValueRange{call.pipeHandle, call.tile});
    return success();
  }

  LogicalResult matchAndRewrite(mlir::pto::TPushOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto call = buildPipeTileRuntimeCall(op, adaptor, targetArch, "TPUSH");
    if (failed(call))
      return rewriter.notifyMatchFailure(op, "failed to resolve pipe token");
    return lowerPipeTileRuntimeCall(op, *call, rewriter);
  }

  PTOArch targetArch;
};

struct PTOTPopToEmitC : public OpConversionPattern<mlir::pto::TPopOp> {
  PTOTPopToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                 PTOArch targetArch)
      : OpConversionPattern<mlir::pto::TPopOp>(typeConverter, ctx),
        targetArch(targetArch) {}

  LogicalResult matchAndRewrite(mlir::pto::TPopOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto call = PTOTPushToEmitC::buildPipeTileRuntimeCall(
        op, adaptor, targetArch, "TPOP");
    if (failed(call))
      return rewriter.notifyMatchFailure(op, "failed to resolve pipe token");
    return PTOTPushToEmitC::lowerPipeTileRuntimeCall(op, *call, rewriter);
  }

  PTOArch targetArch;
};

struct PTOTFreeToEmitC : public OpConversionPattern<mlir::pto::TFreeOp> {
  PTOTFreeToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                  PTOArch targetArch)
      : OpConversionPattern<mlir::pto::TFreeOp>(typeConverter, ctx),
        targetArch(targetArch) {}

  LogicalResult matchAndRewrite(mlir::pto::TFreeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto pipeTok = getTPipeTokenFromValue(op.getPipeHandle(), targetArch);
    if (failed(pipeTok))
      return rewriter.notifyMatchFailure(op, "failed to resolve pipe token");
    auto splitTok = getTileSplitToken(op.getSplit());
    if (failed(splitTok))
      return rewriter.notifyMatchFailure(op, "failed to resolve split token");

    SmallVector<Value> operands{peelUnrealized(adaptor.getPipeHandle())};
    std::string callee;
    if (op.getEntry()) {
      Value entry = peelUnrealized(adaptor.getEntry());
      auto entryTok = getPipeDataTypeToken(entry);
      if (failed(entryTok))
        return rewriter.notifyMatchFailure(op, "failed to resolve entry token");
      callee = "TFREE<" + *pipeTok + ", " + *entryTok + ", " + *splitTok + ">";
      operands.push_back(entry);
    } else {
      callee = "TFREE<" + *pipeTok + ", " + *splitTok + ">";
    }
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, callee, ArrayAttr{}, ArrayAttr{}, operands);
    return success();
  }

  PTOArch targetArch;
};

//===----------------------------------------------------------------------===//
// populate patterns
//===----------------------------------------------------------------------===
static const char *getReinterpretCastTileRoleToken(pto::AddressSpace as) {
  switch (as) {
  case pto::AddressSpace::LEFT:
    return "TileType::Left";
  case pto::AddressSpace::RIGHT:
    return "TileType::Right";
  case pto::AddressSpace::ACC:
    return "TileType::Acc";
  case pto::AddressSpace::BIAS:
    return "TileType::Bias";
  case pto::AddressSpace::MAT:
    return "TileType::Mat";
  case pto::AddressSpace::SCALING:
    return "TileType::Scaling";
  case pto::AddressSpace::VEC:
  case pto::AddressSpace::GM:
  case pto::AddressSpace::Zero:
    return "TileType::Vec";
  }
  llvm_unreachable("unhandled reinterpret_cast tile role");
}

static std::string buildReinterpretCastTileTypeString(MemRefType resMrTy,
                                                      Type elemTy,
                                                      pto::AddressSpace as) {
  std::string elemTok = getEmitCScalarTypeToken(elemTy);
  int64_t rows = 32;
  int64_t cols = 32;
  if (resMrTy.getRank() >= kNumber2 && resMrTy.hasStaticShape()) {
    rows = resMrTy.getDimSize(0);
    cols = resMrTy.getDimSize(1);
  }
  int64_t templateRows =
      renderTileTemplateDim(rows, elemTy, pto::BLayout::RowMajor, 0);
  int64_t templateCols =
      renderTileTemplateDim(cols, elemTy, pto::BLayout::RowMajor, 1);

  return std::string("Tile<") + getReinterpretCastTileRoleToken(as) + ", " +
         elemTok + ", " + std::to_string(templateRows) + ", " +
         std::to_string(templateCols) + ", BLayout::RowMajor, " +
         std::to_string(templateRows) + ", " + std::to_string(templateCols) +
         ", SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>";
}

static Value buildReinterpretCastBaseAddr(ConversionPatternRewriter &rewriter,
                                          Location loc, Value source,
                                          pto::AddressSpace as,
                                          StringRef elemTok) {
  Value rawPtr = source;
  if (isEmitCTileLikeValue(source))
    rawPtr = materializeTileDataValue(rewriter, loc, source, as, elemTok);
  return castAddressToU64(rewriter, loc, rawPtr);
}

static LogicalResult rewriteGmReinterpretCast(memref::ReinterpretCastOp op,
                                              Value source, Value offsetVal,
                                              bool emitAddPtrTrace,
                                              ConversionPatternRewriter &rewriter,
                                              const TypeConverter *typeConverter) {
  if (!offsetVal) {
    rewriter.replaceOp(op, source);
    return success();
  }

  Type resultType = typeConverter->convertType(op.getType());
  if (!resultType)
    return failure();

  auto addOp =
      rewriter.create<emitc::AddOp>(op.getLoc(), resultType, source, offsetVal);
  if (emitAddPtrTrace) {
    rewriter.setInsertionPointAfter(addOp);
    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "PTOAS__ADDPTR_TRACE", ArrayAttr{},
        ArrayAttr{}, ValueRange{addOp.getResult(), source, offsetVal});
  }
  rewriter.replaceOp(op, addOp.getResult());
  return success();
}

struct ReinterpretCastToEmitC : public OpConversionPattern<memref::ReinterpretCastOp> {
  using OpConversionPattern<memref::ReinterpretCastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(memref::ReinterpretCastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    auto resMrTy = dyn_cast<MemRefType>(op.getType());
    if (!resMrTy)
      return failure();

    auto asAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(resMrTy.getMemorySpace());
    const bool isGm = (!asAttr || asAttr.getAddressSpace() == pto::AddressSpace::GM);

    bool emitAddPtrTrace = op->hasAttr("pto.addptr_trace");
    Value source = peelUnrealized(adaptor.getSource());
    auto offsets = adaptor.getOffsets();
    Value offsetVal = offsets.empty() ? Value() : offsets[0];

    // GM: keep pointer arithmetic.
    if (isGm)
      return rewriteGmReinterpretCast(op, source, offsetVal, emitAddPtrTrace,
                                      rewriter, getTypeConverter());

    // UB/L1/L0 tiles: materialize a new Tile view by assigning an adjusted
    // underlying pointer (in elements).
    pto::AddressSpace as = asAttr.getAddressSpace();

    // Element type token.
    Type elemTy = resMrTy.getElementType();
    std::string elemTok = getEmitCScalarTypeToken(elemTy);
    int64_t elemBytes = getEmitCScalarByteWidth(elemTy);
    std::string tileTypeStr =
        buildReinterpretCastTileTypeString(resMrTy, elemTy, as);

    auto tileType = emitc::OpaqueType::get(ctx, tileTypeStr);
    Value tile = rewriter
                     .create<emitc::VariableOp>(loc, tileType,
                                                emitc::OpaqueAttr::get(ctx, ""))
                     .getResult();

    auto u64Ty = emitc::OpaqueType::get(ctx, "uint64_t");
    Value baseAddr =
        buildReinterpretCastBaseAddr(rewriter, loc, source, as, elemTok);
    Value addr = baseAddr;
    if (offsetVal) {
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
//===----------------------------------------------------------------------===//
// pto.taddc lowering -> TADDC(dst, src0, src1, src2)
//===----------------------------------------------------------------------===//

struct PTOTAddCToTADDC : public OpConversionPattern<pto::TAddCOp> {
  using OpConversionPattern<pto::TAddCOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TAddCOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value src2 = peelUnrealized(adaptor.getSrc2());
    Value dst  = peelUnrealized(adaptor.getDst());

    // pto-isa does not provide NPU implementation for TADDC yet.
    // Decompose: dst = src0 + src1 + src2
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TADD",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src0, src1});
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TADD",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, dst, src2});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.tadds lowering -> TADDS(dst, src, scalar)
//===----------------------------------------------------------------------===//

struct PTOAddSToTADDS : public OpConversionPattern<pto::TAddSOp> {
  using OpConversionPattern<pto::TAddSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TAddSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src    = peelUnrealized(adaptor.getSrc());
    Value dst    = peelUnrealized(adaptor.getDst());
    Value scalar = peelUnrealized(adaptor.getScalar());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TADDS",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src, scalar});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.taddsc lowering -> TADDSC(dst, src0, scalar, src1)
//===----------------------------------------------------------------------===//

struct PTOAddSCToTADDSC : public OpConversionPattern<pto::TAddSCOp> {
  using OpConversionPattern<pto::TAddSCOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TAddSCOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src0    = peelUnrealized(adaptor.getSrc0());
    Value scalar  = peelUnrealized(adaptor.getScalar());
    Value src1    = peelUnrealized(adaptor.getSrc1());
    Value dst     = peelUnrealized(adaptor.getDst());

    // pto-isa does not provide NPU implementation for TADDSC yet.
    // Decompose: dst = src0 + scalar + src1
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TADDS",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src0, scalar});
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TADD",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, dst, src1});

    rewriter.eraseOp(op);
    return success();
  }
};
// Tile/vector PTO op conversion patterns live in PTOToEmitCTilePatterns.cpp.

struct PTOPrintOpToEmitC : public OpConversionPattern<pto::PrintOp> {
  using OpConversionPattern<pto::PrintOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::PrintOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    std::string fmt = op.getFormat().str();
    if (fmt.empty())
      fmt = "%f";
    std::string quoted = "\"";
    for (char c : fmt) {
      if (c == '"' || c == '\\')
        quoted += '\\';
      else if (c == '\n')
        quoted += "\\n";
      else if (c == '\t')
        quoted += "\\t";
      else
        quoted += c;
    }
    quoted += "\"";

    Value scalar = peelUnrealized(adaptor.getScalar());
    auto argsAttr = rewriter.getArrayAttr(
        {emitc::OpaqueAttr::get(ctx, quoted),
         IntegerAttr::get(IndexType::get(ctx), 0)});
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "cce::printf",
        /*args=*/argsAttr,
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{scalar});

    rewriter.eraseOp(op);
    return success();
  }
};

// pto.trap -> TRAP()
struct PTOTrapOpToEmitC : public OpConversionPattern<pto::TrapOp> {
  using OpConversionPattern<pto::TrapOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TrapOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "trap",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{});

    rewriter.eraseOp(op);
    return success();
  }
};

// =============================================================================
// Arith CmpI -> EmitC Cmp
// =============================================================================
class ArithCmpIToEmitC : public OpConversionPattern<arith::CmpIOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::CmpIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // 将 arith.cmpi 转换为 emitc.cmp
    // 映射 Predicate: eq -> equal, slt -> less, etc.
    emitc::CmpPredicate emitcPred = emitc::CmpPredicate::eq;
    const bool isUnsignedPred =
        op.getPredicate() == arith::CmpIPredicate::ult ||
        op.getPredicate() == arith::CmpIPredicate::ule ||
        op.getPredicate() == arith::CmpIPredicate::ugt ||
        op.getPredicate() == arith::CmpIPredicate::uge;
    switch (op.getPredicate()) {
      case arith::CmpIPredicate::eq:  emitcPred = emitc::CmpPredicate::eq; break;
      case arith::CmpIPredicate::ne:  emitcPred = emitc::CmpPredicate::ne; break;
      case arith::CmpIPredicate::slt: emitcPred = emitc::CmpPredicate::lt; break;
      case arith::CmpIPredicate::sle: emitcPred = emitc::CmpPredicate::le; break;
      case arith::CmpIPredicate::sgt: emitcPred = emitc::CmpPredicate::gt; break;
      case arith::CmpIPredicate::sge: emitcPred = emitc::CmpPredicate::ge; break;
      // ... 处理无符号比较 (ult, ule 等) ...
      case arith::CmpIPredicate::ult: emitcPred = emitc::CmpPredicate::lt; break;
      case arith::CmpIPredicate::ule: emitcPred = emitc::CmpPredicate::le; break;
      case arith::CmpIPredicate::ugt: emitcPred = emitc::CmpPredicate::gt; break;
      case arith::CmpIPredicate::uge: emitcPred = emitc::CmpPredicate::ge; break;
    }

    Type resTy = getTypeConverter()->convertType(op.getType());
    if (!resTy)
      return failure();

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    if (isUnsignedPred) {
      Type opTy = op.getLhs().getType();
      auto intTy = dyn_cast<IntegerType>(opTy);
      const bool isIndex = isa<IndexType>(opTy);
      if (!intTy && !isIndex)
        return rewriter.notifyMatchFailure(
            op, "expected scalar integer or index operands");

      const unsigned bitWidth =
          intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);
      if (bitWidth != 1) {
        lhs = castSignlessIntToUnsignedSameWidth(rewriter, loc, lhs, bitWidth);
        rhs = castSignlessIntToUnsignedSameWidth(rewriter, loc, rhs, bitWidth);
      }
    }

    rewriter.replaceOpWithNewOp<emitc::CmpOp>(
        op,
        /*resultType=*/resTy, // i1 -> bool/i1
        emitcPred,
        lhs,
        rhs
    );
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Section Op Lowering
//===----------------------------------------------------------------------===//
static bool isA5NoSplitPipeOp(Operation *op) {
  if (auto talloc = dyn_cast<pto::TAllocOp>(op))
    return talloc.getSplit() == 0;
  if (auto tpush = dyn_cast<pto::TPushOp>(op))
    return tpush.getSplit() == 0;
  if (auto tpop = dyn_cast<pto::TPopOp>(op))
    return tpop.getSplit() == 0;
  if (auto tfree = dyn_cast<pto::TFreeOp>(op))
    return tfree.getSplit() == 0;
  if (auto tpush = dyn_cast<pto::TPushToAivOp>(op))
    return tpush.getSplit() == 0;
  if (auto tpush = dyn_cast<pto::TPushToAicOp>(op))
    return tpush.getSplit() == 0;
  if (auto talloc = dyn_cast<pto::TAllocToAivOp>(op))
    return talloc.getSplit() == 0;
  if (auto talloc = dyn_cast<pto::TAllocToAicOp>(op))
    return talloc.getSplit() == 0;
  if (auto tpop = dyn_cast<pto::TPopFromAicOp>(op))
    return tpop.getSplit() == 0;
  if (auto tpop = dyn_cast<pto::TPopFromAivOp>(op))
    return tpop.getSplit() == 0;
  if (auto tfree = dyn_cast<pto::TFreeFromAicOp>(op))
    return tfree.getSplit() == 0;
  if (auto tfree = dyn_cast<pto::TFreeFromAivOp>(op))
    return tfree.getSplit() == 0;
  return false;
}

static bool hasExplicitSubblockControl(Operation *op) {
  bool hasControl = false;
  op->walk([&](Operation *nested) {
    if (isa<pto::GetSubBlockIdxOp, pto::GetSubBlockNumOp>(nested)) {
      hasControl = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return hasControl;
}

} // namespace

bool needsA5NoSplitVectorGuard(Operation *op) {
  auto arch = getTargetArch(op);
  if (arch != PTOArch::A5)
    return false;
  bool isVectorScope = isa<pto::SectionVectorOp>(op);
  if (auto func = dyn_cast<func::FuncOp>(op)) {
    if (auto kernelKindAttr =
            func->getAttrOfType<FunctionKernelKindAttr>(
                FunctionKernelKindAttr::name)) {
      isVectorScope =
          kernelKindAttr.getKernelKind() == FunctionKernelKind::Vector;
    }
  }
  if (!isVectorScope)
    return false;
  if (hasExplicitSubblockControl(op))
    return false;

  bool hasNoSplitPipe = false;
  op->walk([&](Operation *nested) {
    if (!isA5NoSplitPipeOp(nested))
      return WalkResult::advance();
    hasNoSplitPipe = true;
    return WalkResult::interrupt();
  });
  return hasNoSplitPipe;
}


void populatePTOToEmitCRuntimeOpPatterns(RewritePatternSet &patterns,
                                         TypeConverter &typeConverter,
                                         MLIRContext *ctx, PTOArch targetArch) {
  patterns.add<PTOTAllocToEmitC>(typeConverter, ctx, targetArch);
  patterns.add<PTOTPushToEmitC>(typeConverter, ctx, targetArch);
  patterns.add<PTOTPopToEmitC>(typeConverter, ctx, targetArch);
  patterns.add<PTOTFreeToEmitC>(typeConverter, ctx, targetArch);
  patterns.add<ReinterpretCastToEmitC>(typeConverter, ctx);
  patterns.add<PTOTAddCToTADDC>(typeConverter, ctx);
  patterns.add<PTOAddSToTADDS>(typeConverter, ctx);
  patterns.add<PTOAddSCToTADDSC>(typeConverter, ctx);
  patterns.add<PTOPrintOpToEmitC>(typeConverter, ctx);
  patterns.add<PTOTrapOpToEmitC>(typeConverter, ctx);
  patterns.add<ArithCmpIToEmitC>(typeConverter, ctx);
}

} // namespace mlir::pto

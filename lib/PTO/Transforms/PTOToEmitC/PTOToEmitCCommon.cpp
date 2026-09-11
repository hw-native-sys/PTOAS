// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOToEmitCCommon.cpp - shared helper definitions ---------===//
//===----------------------------------------------------------------------===//

#include "PTOToEmitCEmitters.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

FailureOr<Value>
adaptCallOperandForEmitC(const TypeConverter *typeConverter,
                         ConversionPatternRewriter &rewriter, Location loc,
                         Type originalCalleeArgTy, Value originalOperand,
                         Value loweredOperand) {
  Type elemTy;
  std::optional<pto::AddressSpace> as;
  if (auto ptrTy = dyn_cast<pto::PtrType>(originalCalleeArgTy)) {
    elemTy = ptrTy.getElementType();
    as = getAddressSpaceOrGM(ptrTy.getMemorySpace());
  } else if (auto memrefTy = dyn_cast<MemRefType>(originalCalleeArgTy)) {
    elemTy = memrefTy.getElementType();
    if (auto asAttr =
            dyn_cast_or_null<pto::AddressSpaceAttr>(memrefTy.getMemorySpace())) {
      as = asAttr.getAddressSpace();
    } else {
      as = pto::AddressSpace::GM;
    }
  }

  if (elemTy && as) {
    std::string elemTokStorage = getEmitCScalarTypeToken(elemTy);
    StringRef elemTok(elemTokStorage);

    auto materializeForCall = [&](Value tileLike) -> FailureOr<Value> {
      Value extracted =
          materializeTileDataValue(rewriter, loc, tileLike, *as, elemTok);
      if (!typeConverter)
        return extracted;
      Type targetTy = typeConverter->convertType(originalCalleeArgTy);
      if (!targetTy)
        return failure();
      if (extracted.getType() == targetTy)
        return extracted;
      return rewriter.create<emitc::CastOp>(loc, targetTy, extracted)
          .getResult();
    };

    if (auto tileBufAddr = originalOperand.getDefiningOp<pto::TileBufAddrOp>()) {
      Value tileValue = loweredOperand;
      if (!isEmitCTileLikeType(tileValue.getType()) && tileBufAddr.getSrc())
        tileValue = tileBufAddr.getSrc();
      if (isEmitCTileLikeType(tileValue.getType()))
        return materializeForCall(tileValue);
    }

    if (isEmitCTileLikeType(loweredOperand.getType()))
      return materializeForCall(loweredOperand);
  }

  return loweredOperand;
}

const char *addrSpaceQualifier(pto::AddressSpace as) {
  switch (as) {
  case pto::AddressSpace::Zero:
    return "__gm__";
  case pto::AddressSpace::VEC:
    return "__ubuf__";
  case pto::AddressSpace::GM:
    return "__gm__";
  case pto::AddressSpace::MAT:
    return "__cbuf__";
  case pto::AddressSpace::LEFT:
    return "__ca__";
  case pto::AddressSpace::RIGHT:
    return "__cb__";
  case pto::AddressSpace::ACC:
    return "__cc__";
  case pto::AddressSpace::BIAS:
    // Bias tiles are special in pto-isa; keep a safe fallback qualifier.
    return "__gm__";
  case pto::AddressSpace::SCALING:
    // pto-isa TileType::Scaling maps to __fbuf__ (see pto/common/memory.hpp).
    return "__fbuf__";
  }
  return "__gm__";
}

// Collect ordered name hints from a FusedLoc's metadata attribute; returns
// true when the metadata carried usable hints.
static bool appendFusedLocMetadataHints(Attribute metadata,
                                        SmallVectorImpl<std::string> &hints) {
  if (auto strAttr = dyn_cast<StringAttr>(metadata)) {
    std::string raw = strAttr.getValue().str();
    if (!raw.empty())
      hints.push_back(std::move(raw));
    return true;
  }
  if (auto arrayAttr = dyn_cast<ArrayAttr>(metadata)) {
    for (Attribute attr : arrayAttr) {
      auto strAttr = dyn_cast<StringAttr>(attr);
      if (!strAttr)
        continue;
      std::string raw = strAttr.getValue().str();
      if (!raw.empty())
        hints.push_back(std::move(raw));
    }
    return !hints.empty();
  }
  return false;
}

void appendRawLocationNameHints(Location loc,
                                       SmallVectorImpl<std::string> &hints) {
  if (auto nameLoc = dyn_cast<NameLoc>(loc)) {
    std::string raw = nameLoc.getName().getValue().str();
    if (!raw.empty())
      hints.push_back(std::move(raw));
    return;
  }

  if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
    if (Attribute metadata = fusedLoc.getMetadata())
      appendFusedLocMetadataHints(metadata, hints);

    // Only metadata explicitly attached by PTOAS name-hint recovery carries an
    // ordered result-name list. Ordinary fused child locations are debug
    // provenance, not result-indexed name hints.
    return;
  }

  if (auto callSiteLoc = dyn_cast<CallSiteLoc>(loc)) {
    appendRawLocationNameHints(callSiteLoc.getCallee(), hints);
    if (hints.empty())
      appendRawLocationNameHints(callSiteLoc.getCaller(), hints);
  }
}

Value applyStaticMemrefOffset(ConversionPatternRewriter &rewriter,
                                     Location loc, Value basePtr,
                                     int64_t offset) {
  if (offset == 0)
    return basePtr;
  auto *ctx = rewriter.getContext();
  Type u32Ty = emitc::OpaqueType::get(ctx, "unsigned");
  auto offVal = rewriter.create<emitc::ConstantOp>(
      loc, u32Ty, emitc::OpaqueAttr::get(ctx, std::to_string(offset)));
  return rewriter.create<emitc::AddOp>(loc, basePtr.getType(), basePtr, offVal);
}

// Return the operand as a tile value when it already is one (directly or
// after peeling conversion casts); otherwise return an empty Value.
static Value resolveScratchTileValue(Value emittedScratch) {
  Value scratch = emittedScratch;
  if (auto opaqueTy = dyn_cast<emitc::OpaqueType>(scratch.getType())) {
    StringRef typeStr = opaqueTy.getValue();
    if (typeStr.contains("Tile<") || typeStr.contains("ConvTile<"))
      return scratch;
  }
  scratch = peelUnrealized(scratch);
  if (auto opaqueTy = dyn_cast<emitc::OpaqueType>(scratch.getType())) {
    StringRef typeStr = opaqueTy.getValue();
    if (typeStr.contains("Tile<") || typeStr.contains("ConvTile<"))
      return scratch;
  }
  return Value();
}

FailureOr<Value> buildAsyncScratchTileValue(
    ConversionPatternRewriter &rewriter, Location loc, Value originalScratch,
    Value emittedScratch) {
  Value tile = resolveScratchTileValue(emittedScratch);
  if (tile)
    return tile;

  auto memTy = dyn_cast<MemRefType>(originalScratch.getType());
  if (!memTy)
    return failure();

  ArrayRef<int64_t> shape = memTy.getShape();
  if (!memTy.hasStaticShape() || shape.empty() || shape.size() > 2)
    return failure();

  int64_t rows = shape.size() == 1 ? 1 : shape[0];
  int64_t cols = shape.size() == 1 ? shape[0] : shape[1];

  auto *ctx = rewriter.getContext();
  pto::TileBufConfigAttr configAttr = pto::TileBufConfigAttr::getDefault(ctx);
  int32_t fractal = 512;
  if (auto frAttr = dyn_cast<IntegerAttr>(configAttr.getSFractalSize()))
    fractal = static_cast<int32_t>(getIntegerAttrSignedValue(frAttr));

  Type elemTy = memTy.getElementType();
  pto::BLayout blayout = getTileBufBLayoutValue(configAttr);
  int64_t templateRows = renderTileTemplateDim(rows, elemTy, blayout, 0);
  int64_t templateCols = renderTileTemplateDim(cols, elemTy, blayout, 1);
  std::string elemTypeStr = getEmitCScalarTypeToken(elemTy);
  std::string tileTypeStr =
      "Tile<TileType::Vec, " + elemTypeStr + ", " +
      std::to_string(templateRows) + ", " + std::to_string(templateCols) +
      ", " + tileBufBLayoutToken(configAttr) + ", " +
      std::to_string(templateRows) + ", " + std::to_string(templateCols) +
      ", " + tileBufSLayoutToken(configAttr) + ", " +
      std::to_string(fractal) + ", " + tileBufPadToken(configAttr) + ">";

  tile = rewriter
             .create<emitc::VariableOp>(
                 loc, getEmitCVariableResultType(
                          emitc::OpaqueType::get(ctx, tileTypeStr)),
                 emitc::OpaqueAttr::get(ctx, ""))
             .getResult();
  tile = loadEmitCVariableIfNeeded(rewriter, loc, tile);
  Value scratch = peelUnrealized(emittedScratch);
  auto addr = rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, "uint64_t")});
  Value scratchAddr =
      rewriter
          .create<emitc::CallOpaqueOp>(loc, emitc::OpaqueType::get(ctx, "uint64_t"),
                                       "reinterpret_cast", ArrayAttr{}, addr,
                                       ValueRange{scratch})
          .getResult(0);
  rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "TASSIGN",
                                       ArrayAttr{}, ArrayAttr{},
                                       ValueRange{tile, scratchAddr});
  return tile;
}

SmallVector<unsigned, 4>
buildDefaultLastUseTileSlotOrder(Operation *op) {
  SmallVector<unsigned, 4> dpsInitTileOperands;
  SmallVector<unsigned, 4> nonDpsTileOperands;
  for (OpOperand &operand : op->getOpOperands()) {
    if (!isa<pto::TileBufType>(operand.get().getType()))
      continue;
    if (isDpsInitOperand(operand)) {
      dpsInitTileOperands.push_back(operand.getOperandNumber());
    } else {
      nonDpsTileOperands.push_back(operand.getOperandNumber());
    }
  }

  // Most tile intrinsics lower as `CALLEE(dst, src0, src1, ...)`. When an op
  // has exactly one DPS init tile, treat that output slot as the leading
  // emitted tile operand so `[[pto::last_use(...)]]` aligns with the final
  // intrinsic call order.
  if (dpsInitTileOperands.size() == 1) {
    SmallVector<unsigned, 4> ordered{dpsInitTileOperands.front()};
    ordered.append(nonDpsTileOperands.begin(), nonDpsTileOperands.end());
    return ordered;
  }

  SmallVector<unsigned, 4> ordered = std::move(nonDpsTileOperands);
  ordered.append(dpsInitTileOperands.begin(), dpsInitTileOperands.end());
  return ordered;
}

FailureOr<std::string> buildEmitCOpaqueConstantLiteral(Type targetType,
                                                              Attribute valueAttr) {
  auto opaqueTy = dyn_cast<emitc::OpaqueType>(targetType);
  if (!opaqueTy)
    return failure();

  if (opaqueTy.getValue() == "pto::MrgSortExecutedNumList") {
    auto dense = dyn_cast_or_null<DenseIntElementsAttr>(valueAttr);
    if (!dense)
      return failure();

    auto vecTy = dyn_cast<VectorType>(dense.getType());
    if (!vecTy || vecTy.getRank() != 1 || vecTy.getNumElements() != 4 ||
        !vecTy.getElementType().isInteger(16))
      return failure();

    std::string literal;
    llvm::raw_string_ostream os(literal);
    os << "pto::MrgSortExecutedNumList{";
    bool first = true;
    for (APInt elem : dense.getValues<APInt>()) {
      if (!first)
        os << ", ";
      first = false;
      os << getAPIntUnsignedValue(elem);
    }
    os << "}";
    os.flush();
    return literal;
  }

  return failure();
}

std::string buildFixpipeConfigAliasName(int32_t pipeId) {
  return "Pipe" + std::to_string(pipeId) + "FixpipeConfig";
}

FailureOr<std::string>
buildFixpipeConfigTypeToken(AccPushEpilogueAttr accPushEpilogue) {
  auto layoutTok = getFixpipeLayoutToken(accPushEpilogue.getLayout());
  auto quantTok = getFixpipeQuantToken(accPushEpilogue.getQuant());
  auto reluTok = getFixpipeReluToken(accPushEpilogue.getRelu());
  if (failed(layoutTok) || failed(quantTok) || failed(reluTok))
    return failure();
  return "FixpipeParams<" + *layoutTok + ", " + *quantTok + ", " + *reluTok +
         ">";
}

static GlobalTensorTypeNames getGlobalTensorTypeNames(Operation *anchor,
                                                       StringRef tag) {
  // The type-alias names are keyed on the anchor op pointer. When a single op
  // wraps more than one GM memref as a GlobalTensor (e.g. GM->L1 mgather wraps
  // mem + idx + scratch), an extra `tag` keeps the emitted `using` aliases
  // distinct so the generated C++ does not redefine a type with a new value.
  std::string suffix = "_" + std::to_string(reinterpret_cast<uintptr_t>(anchor));
  if (!tag.empty())
    suffix += "_" + tag.str();
  return {
      "GTShape" + suffix,
      "GTStride" + suffix,
      "GT" + suffix,
      "GT" + suffix + "_layout",
  };
}

// Emit the `using` aliases for a GlobalTensor's Shape/Stride types plus the
// constexpr Layout constant, returning the layout enum spelling used.
static std::string
emitGlobalTensorAliases(ConversionPatternRewriter &rewriter, Location loc,
                        const GlobalTensorTypeNames &names,
                        const SmallVector<int64_t, 5> &shape5D,
                        const SmallVector<int64_t, 5> &stride5D,
                        const std::optional<SpecialGlobalTensorTypeSpec> &spec,
                        Operation *anchor, Value basePtr,
                        ArrayRef<int64_t> shape, ArrayRef<int64_t> strides,
                        Type elemTy) {
  std::string layoutEnum;
  if (spec) {
    rewriter.create<emitc::VerbatimOp>(
        loc, "using " + names.shapeTypeName + " = " + spec->shapeTypeExpr + ";");
    rewriter.create<emitc::VerbatimOp>(
        loc, "using " + names.strideTypeName + " = " + spec->strideTypeExpr + ";");
    layoutEnum = spec->layoutEnum;
  } else {
    rewriter.create<emitc::VerbatimOp>(
        loc, "using " + names.shapeTypeName + " = pto::Shape<" +
                 joinIntTemplateParams(shape5D) + ">;");
    rewriter.create<emitc::VerbatimOp>(
        loc, "using " + names.strideTypeName + " = pto::Stride<" +
                 joinIntTemplateParams(stride5D) + ">;");
    layoutEnum = resolveGlobalTensorLayout(anchor, basePtr, shape, strides,
                                           elemTy);
  }
  return layoutEnum;
}

Value buildGlobalTensorFromMemref(ConversionPatternRewriter &rewriter,
                                         Location loc, Value basePtr,
                                         MemRefType mrTy,
                                         Operation *anchor, StringRef tag) {
  auto *ctx = rewriter.getContext();

  ArrayRef<int64_t> shape = mrTy.getShape();
  if (!hasStaticShape(mrTy))
    return Value();

  SmallVector<int64_t> strides;
  int64_t offset = 0;
  if (!getStaticMemrefLayout(mrTy, strides, offset))
    return Value();

  Value ptr = applyStaticMemrefOffset(rewriter, loc, basePtr, offset);
  GlobalTensorTypeNames names = getGlobalTensorTypeNames(anchor, tag);
  std::string elemTypeStr = getElemTypeStringForGT(mrTy.getElementType());
  SmallVector<int64_t, 5> shape5D;
  SmallVector<int64_t, 5> stride5D;
  buildGlobalTensorShapeAndStride(shape, strides, shape5D, stride5D);

  std::string layoutEnum =
      emitGlobalTensorAliases(rewriter, loc, names, shape5D, stride5D,
                              getSpecialScaleGlobalTensorTypeSpec(anchor, mrTy),
                              anchor, basePtr, shape, strides,
                              mrTy.getElementType());

  rewriter.create<emitc::VerbatimOp>(loc, "constexpr pto::Layout " +
                                              names.layoutConstName + " = " +
                                              layoutEnum + ";");

  auto shapeTypeOpaque = emitc::OpaqueType::get(ctx, names.shapeTypeName);
  auto strideTypeOpaque = emitc::OpaqueType::get(ctx, names.strideTypeName);
  auto shapeInstOp = rewriter.create<emitc::CallOpaqueOp>(
      loc, shapeTypeOpaque, names.shapeTypeName, ArrayAttr{}, ArrayAttr{},
      ValueRange{});
  auto strideInstOp = rewriter.create<emitc::CallOpaqueOp>(
      loc, strideTypeOpaque, names.strideTypeName, ArrayAttr{}, ArrayAttr{},
      ValueRange{});

  rewriter.create<emitc::VerbatimOp>(
      loc, "using " + names.tensorTypeName + " = GlobalTensor<" + elemTypeStr +
               ", " + names.shapeTypeName + ", " + names.strideTypeName +
               ", " + names.layoutConstName + ">;");
  auto gtType = emitc::OpaqueType::get(ctx, names.tensorTypeName);

  SmallVector<Value> gtArgs;
  gtArgs.push_back(ptr);
  gtArgs.push_back(shapeInstOp.getResult(0));
  gtArgs.push_back(strideInstOp.getResult(0));

  auto gtInst = rewriter.create<emitc::CallOpaqueOp>(
      loc, gtType, names.tensorTypeName, ArrayAttr{}, ArrayAttr{},
      ValueRange(gtArgs));

  return gtInst.getResult(0);
}



void buildGlobalTensorShapeAndStride(ArrayRef<int64_t> shape,
                                            ArrayRef<int64_t> strides,
                                            SmallVectorImpl<int64_t> &shape5D,
                                            SmallVectorImpl<int64_t> &stride5D) {
  shape5D.assign(5, 1);
  stride5D.assign(5, 1);
  int rank = static_cast<int>(shape.size());
  int shift = 5 - rank;
  for (int i = 0; i < rank && i < 5; ++i) {
    shape5D[shift + i] = shape[i];
    stride5D[shift + i] = strides[i];
  }
  for (int i = 3; i >= 0; --i) {
    if (i >= shift)
      continue;
    stride5D[i] = multiplyOrDynamic(shape5D[i + 1], stride5D[i + 1]);
  }
}

std::optional<std::string> buildLastUseMarkerCallee(
    Operation *op, StringRef callee, ArrayRef<unsigned> tileSlotOrder) {
  auto lastUseAttr = dyn_cast_or_null<DenseI64ArrayAttr>(
      op->getAttr(kLastUseAttrName));
  if (!lastUseAttr)
    return std::nullopt;

  SmallVector<unsigned, 4> originalTileOperands = collectTileOperandNumbers(op);
  ArrayRef<int64_t> originalBits = lastUseAttr.asArrayRef();
  if (originalTileOperands.size() != originalBits.size())
    return std::nullopt;

  SmallVector<unsigned, 4> defaultTileSlotOrder;
  if (tileSlotOrder.empty()) {
    defaultTileSlotOrder = buildDefaultLastUseTileSlotOrder(op);
    tileSlotOrder = defaultTileSlotOrder;
  }
  if (tileSlotOrder.size() != originalBits.size())
    return std::nullopt;

  SmallVector<int64_t, 4> reorderedBits;
  reorderedBits.reserve(tileSlotOrder.size());
  for (unsigned operandNumber : tileSlotOrder) {
    bool found = false;
    for (auto [idx, originalOperandNumber] : llvm::enumerate(originalTileOperands)) {
      if (originalOperandNumber != operandNumber)
        continue;
      reorderedBits.push_back(originalBits[idx]);
      found = true;
      break;
    }
    if (!found)
      return std::nullopt;
  }

  std::string marker = kLastUseMarkerPrefix.str();
  marker.append(callee.str());
  marker.append("__");
  bool first = true;
  for (int64_t bit : reorderedBits) {
    if (!first)
      marker.append("__");
    first = false;
    marker.append(std::to_string(bit));
  }
  return marker;
}

SmallVector<int64_t> buildRowMajorStrides(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> strides(shape.size(), 1);
  int64_t running = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[i] = running;
    running = multiplyOrDynamic(running, shape[i]);
  }
  return strides;
}

FailureOr<std::string>
getTPipeDirectionToken(bool isL2G2L, int8_t dirMask, PTOArch targetArch) {
  if (dirMask == 1) {
    if (isL2G2L && targetArch == PTOArch::A5)
      return std::string("Direction::DIR_C2V_GM");
    return std::string("Direction::DIR_C2V");
  }
  if (dirMask == 2) {
    if (isL2G2L && targetArch == PTOArch::A5)
      return std::string("Direction::DIR_V2C_GM");
    return std::string("Direction::DIR_V2C");
  }
  if (dirMask == 3)
    return std::string("Direction::DIR_BOTH");
  return failure();
}

std::string buildTPipeToken(int32_t flagBase, llvm::StringRef dirTok,
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

FailureOr<std::string> buildTPipeTokenFromInitOp(Operation *op,
                                                        PTOArch targetArch) {
  if (auto initOp = dyn_cast<pto::InitializeL2G2LPipeOp>(op)) {
    if (!initOp.getFlagBaseAttr())
      return failure();
    auto dirTok =
        getTPipeDirectionToken(/*isL2G2L=*/true, initOp.getDirMask(), targetArch);
    if (failed(dirTok))
      return failure();
    int32_t localSlotNum =
        initOp.getLocalSlotNumAttr()
            ? static_cast<int32_t>(
                  getIntegerAttrSignedValue(initOp.getLocalSlotNumAttr()))
            : initOp.getSlotNum();
    return buildTPipeToken(
        static_cast<int32_t>(getIntegerAttrSignedValue(initOp.getFlagBaseAttr())),
        *dirTok, initOp.getSlotSize(), initOp.getSlotNum(), localSlotNum,
        initOp.getNosplitAttr() && initOp.getNosplitAttr().getValue());
  }

  if (auto initOp = dyn_cast<pto::InitializeL2LPipeOp>(op)) {
    if (!initOp.getFlagBaseAttr())
      return failure();
    auto dirTok =
        getTPipeDirectionToken(/*isL2G2L=*/false, initOp.getDirMask(), targetArch);
    if (failed(dirTok))
      return failure();
    return buildTPipeToken(
        static_cast<int32_t>(getIntegerAttrSignedValue(initOp.getFlagBaseAttr())),
        *dirTok, initOp.getSlotSize(), initOp.getSlotNum(), 2,
        initOp.getNosplitAttr() && initOp.getNosplitAttr().getValue());
  }

  return failure();
}

Value castSignlessIntToUnsignedSameWidth(ConversionPatternRewriter &rewriter,
                                                Location loc, Value v,
                                                unsigned bitWidth) {
  auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
  return emitCCast(rewriter, loc, uTy, v);
}

Value castToGMBytePointer(ConversionPatternRewriter &rewriter,
                                 Location loc, Value value) {
  auto *ctx = rewriter.getContext();
  auto targetTy =
      emitc::PointerType::get(emitc::OpaqueType::get(ctx, "__gm__ uint8_t"));
  if (value.getType() == targetTy) {
    return value;
  }

  auto castTyAttr =
      rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, "__gm__ uint8_t*")});
  if (isSetFFTsPointerLikeType(value.getType())) {
    return rewriter
        .create<emitc::CallOpaqueOp>(loc, targetTy, "reinterpret_cast",
                                     ArrayAttr{}, castTyAttr,
                                     ValueRange{value})
        .getResult(0);
  }
  return rewriter.create<emitc::CastOp>(loc, targetTy, value).getResult();
}

void collectStructTypes(Type t, llvm::SetVector<pto::StructType> &out) {
  auto st = dyn_cast<pto::StructType>(t);
  if (!st || out.contains(st))
    return;
  for (Type f : st.getFieldTypes())
    collectStructTypes(f, out);
  out.insert(st);
}

SmallVector<unsigned, 4> collectTileOperandNumbers(Operation *op) {
  SmallVector<unsigned, 4> tileOperandNumbers;
  for (OpOperand &operand : op->getOpOperands()) {
    if (isa<pto::TileBufType>(operand.get().getType()))
      tileOperandNumbers.push_back(operand.getOperandNumber());
  }
  return tileOperandNumbers;
}

void createOpaqueCall(ConversionPatternRewriter &rewriter, Location loc,
                             TypeRange resultTypes, StringRef callee,
                             ArrayAttr args, ArrayAttr templateArgs,
                             ValueRange operands) {
  rewriter.create<emitc::CallOpaqueOp>(loc, resultTypes, callee, args,
                                       templateArgs, operands);
}

void createLastUseAwareOpaqueCall(
    ConversionPatternRewriter &rewriter, Operation *op, TypeRange resultTypes,
    StringRef callee, ValueRange operands, ArrayAttr args,
    ArrayAttr templateArgs, ArrayRef<unsigned> tileSlotOrder) {
  std::string calleeStorage;
  StringRef effectiveCallee =
      getLastUseAwareCallee(op, callee, calleeStorage, tileSlotOrder);
  createOpaqueCall(rewriter, op->getLoc(), resultTypes, effectiveCallee, args,
                   templateArgs, operands);
}

Value emitCCast(ConversionPatternRewriter &rewriter, Location loc,
                       Type dstType, Value src) {
  if (src.getType() == dstType)
    return src;
  return rewriter.createOrFold<emitc::CastOp>(loc, dstType, src);
}

bool isDeadPureEmitCValueOp(Operation *op) {
  if (op->getNumResults() == 0)
    return false;
  if (!llvm::all_of(op->getResults(),
                    [](Value result) { return result.use_empty(); }))
    return false;

  if (auto call = dyn_cast<emitc::CallOpaqueOp>(op)) {
    StringRef callee = call.getCallee();
    return callee.starts_with("pto::Shape<") ||
           callee.starts_with("pto::Stride<") ||
           callee.starts_with("GlobalTensor<");
  }

  return isa<emitc::AddOp, emitc::MulOp, emitc::CastOp, emitc::ConstantOp>(op);
}

void eraseDeadPureEmitCValueOps(ModuleOp module) {
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> deadOps;
    module.walk([&](Operation *op) {
      if (isDeadPureEmitCValueOp(op)) {
        deadOps.push_back(op);
      }
    });
    for (Operation *op : llvm::reverse(deadOps)) {
      op->erase();
      changed = true;
    }
  }
}

FailureOr<Operation *> findPeerFixpipeConsumerInit(Operation *producerInit) {
  auto ownerFuncAttr =
      producerInit->getAttrOfType<FlatSymbolRefAttr>(kPipePeerOwnerFuncAttrName);
  auto reserveNameAttr =
      producerInit->getAttrOfType<StringAttr>(kPipePeerReserveNameAttrName);
  auto dirMaskAttr =
      producerInit->getAttrOfType<IntegerAttr>(kPipePeerDirMaskAttrName);
  if (!ownerFuncAttr || !reserveNameAttr || !dirMaskAttr ||
      dirMaskAttr.getInt() != 1)
    return failure();

  auto peerFunc =
      lookupPeerFuncAcrossContainer(producerInit, ownerFuncAttr);
  if (!peerFunc)
    return failure();

  Operation *matchedInit = nullptr;
  unsigned matchedInitCount = 0;
  peerFunc.walk([&](Operation *candidate) {
    if (!isa<InitializeL2LPipeOp, InitializeL2G2LPipeOp>(candidate))
      return WalkResult::advance();

    if (!getPipeInitAccPushEpilogue(candidate))
      return WalkResult::advance();

    auto candidateOwnerFuncAttr =
        candidate->getAttrOfType<FlatSymbolRefAttr>(kPipePeerOwnerFuncAttrName);
    auto candidateReserveNameAttr =
        candidate->getAttrOfType<StringAttr>(kPipePeerReserveNameAttrName);
    auto candidateDirMaskAttr =
        candidate->getAttrOfType<IntegerAttr>(kPipePeerDirMaskAttrName);
    if (!candidateOwnerFuncAttr || !candidateReserveNameAttr ||
        !candidateDirMaskAttr)
      return WalkResult::advance();

    if (candidateOwnerFuncAttr != ownerFuncAttr ||
        candidateReserveNameAttr != reserveNameAttr ||
        candidateDirMaskAttr.getInt() != dirMaskAttr.getInt())
      return WalkResult::advance();

    auto candidateFunc = candidate->getParentOfType<func::FuncOp>();
    if (!candidateFunc || candidateFunc != peerFunc)
      return WalkResult::advance();

    matchedInit = candidate;
    ++matchedInitCount;
    return WalkResult::advance();
  });
  if (matchedInitCount != 1 || !matchedInit)
    return failure();
  return matchedInit;
}

int64_t getAPIntSignedValue(const APInt &value) {
  return value.getBitWidth() == 0 ? 0 : value.getSExtValue();
}

uint64_t getAPIntUnsignedValue(const APInt &value) {
  return value.getBitWidth() == 0 ? 0 : value.getZExtValue();
}

pto::AddressSpace getAddressSpaceOrGM(Attribute memorySpace) {
  if (auto asAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(memorySpace))
    return asAttr.getAddressSpace();
  return pto::AddressSpace::GM;
}

std::string getElemTypeStringForGT(Type elemTy) {
  return getEmitCScalarTypeToken(elemTy);
}

emitc::PointerType getEmitCPointerType(MLIRContext *ctx,
                                      StringRef pointeeTypeStr) {
  return emitc::PointerType::get(emitc::OpaqueType::get(ctx, pointeeTypeStr));
}

emitc::PointerType getEmitCPointerType(MLIRContext *ctx, StringRef qualifier,
                                      StringRef elemTypeStr) {
  return getEmitCPointerType(ctx, (qualifier + " " + elemTypeStr).str());
}

int64_t getEmitCScalarByteWidth(Type elemTy) {
  if (pto::getPTOStorageElemByteSize(elemTy) == 1)
    return 1;
  if (elemTy.isF16() || elemTy.isBF16() || elemTy.isInteger(16))
    return 2;
  if (elemTy.isF32() || elemTy.isInteger(32))
    return 4;
  if (elemTy.isF64() || elemTy.isInteger(64))
    return 8;
  return 4;
}

// Map low-precision (FP8/FP4 family) types to their EmitC scalar token.
// Returns std::nullopt for non-low-precision types.
static std::optional<StringRef> getLowPrecisionToken(Type elemTy) {
  if (pto::isPTOFloat8E4M3LikeType(elemTy))
    return StringRef("float8_e4m3_t");
  if (pto::isPTOFloat8E5M2LikeType(elemTy))
    return StringRef("float8_e5m2_t");
  if (isF8E8M0ElemType(elemTy))
    return StringRef("float8_e8m0_t");
  if (isa<pto::HiF8Type>(elemTy))
    return StringRef("hifloat8_t");
  if (isa<pto::F4E1M2x2Type>(elemTy))
    return StringRef("float4_e1m2x2_t");
  if (isa<pto::F4E2M1x2Type>(elemTy))
    return StringRef("float4_e2m1x2_t");
  return std::nullopt;
}

// Map standard float types to their EmitC scalar token.
static std::optional<StringRef> getFloatToken(Type elemTy) {
  if (elemTy.isF16())
    return StringRef("half");
  if (elemTy.isBF16())
    return StringRef("bfloat16_t");
  if (elemTy.isF32())
    return StringRef("float");
  if (elemTy.isF64())
    return StringRef("double");
  return std::nullopt;
}

// Map integer types to their EmitC scalar token; signless and signed map to
// signed C types, unsigned maps to unsigned C types.
static std::optional<StringRef> getIntegerToken(Type elemTy) {
  if (elemTy.isInteger(8)) {
    return (elemTy.isSignlessInteger(8) || elemTy.isSignedInteger(8))
               ? StringRef("int8_t")
               : StringRef("uint8_t");
  }
  if (elemTy.isInteger(16)) {
    return (elemTy.isSignlessInteger(16) || elemTy.isSignedInteger(16))
               ? StringRef("int16_t")
               : StringRef("uint16_t");
  }
  if (elemTy.isInteger(32)) {
    return (elemTy.isSignlessInteger(32) || elemTy.isSignedInteger(32))
               ? StringRef("int32_t")
               : StringRef("uint32_t");
  }
  if (elemTy.isInteger(64)) {
    return cast<IntegerType>(elemTy).isUnsigned() ? StringRef("uint64_t")
                                                  : StringRef("int64_t");
  }
  return std::nullopt;
}

std::string getEmitCScalarTypeToken(Type elemTy) {
  if (auto lowPrec = getLowPrecisionToken(elemTy))
    return lowPrec->str();
  if (auto flt = getFloatToken(elemTy))
    return flt->str();
  if (auto intTok = getIntegerToken(elemTy))
    return intTok->str();
  return "float";
}

std::optional<std::string> getEmitCTileTypeString(pto::TileBufType type) {
  if (type.getRank() != 2)
    return std::nullopt;
  auto validShape = type.getValidShape();
  if (validShape.size() != 2)
    return std::nullopt;

  Type elemTy = type.getElementType();
  auto configAttr = type.getConfigAttr();
  pto::BLayout blayout = getTileBufBLayoutValue(configAttr);
  ArrayRef<int64_t> shape = type.getShape();
  int64_t rows = shape[0];
  int64_t cols = shape[1];

  auto render = [&](int64_t dim, int dimIdx) {
    return renderTileTemplateDim(dim, elemTy, blayout, dimIdx);
  };

  std::string vrowTok =
      validShape[0] == ShapedType::kDynamic
          ? "-1"
          : std::to_string(render(validShape[0], 0));
  std::string vcolTok =
      validShape[1] == ShapedType::kDynamic
          ? "-1"
          : std::to_string(render(validShape[1], 1));

  if (auto asAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(type.getMemorySpace())) {
    if (isLowPrecisionCubeOperandType(elemTy)) {
      if (asAttr.getAddressSpace() == pto::AddressSpace::LEFT &&
          shape[0] != 1 &&
          validShape[1] != ShapedType::kDynamic) {
        vcolTok = std::to_string(render(cols, 1));
      } else if (asAttr.getAddressSpace() == pto::AddressSpace::RIGHT &&
                 validShape[0] != ShapedType::kDynamic) {
        vrowTok = std::to_string(render(rows, 0));
      }
    }
  }

  int32_t fractal = 512;
  if (auto frAttr = dyn_cast<IntegerAttr>(configAttr.getSFractalSize()))
    fractal = static_cast<int32_t>(getIntegerAttrSignedValue(frAttr));

  return std::string("Tile<") +
         tileRoleToken(type.getMemorySpace(), elemTy, type.getConfigAttr()) + ", " +
         getEmitCScalarTypeToken(elemTy) + ", " +
         std::to_string(render(rows, 0)) + ", " +
         std::to_string(render(cols, 1)) + ", " +
         tileBufBLayoutToken(configAttr) + ", " + vrowTok + ", " + vcolTok +
         ", " + tileBufSLayoutToken(configAttr) + ", " +
         std::to_string(fractal) + ", " + tileBufPadToken(configAttr) + ", " +
         tileBufCompactToken(configAttr) + ">";
}

Type getEmitCVariableResultType(Type valueType) {
  return valueType;
}

FailureOr<std::string> getFixpipeLayoutToken(FixpipeLayout layout) {
  switch (layout) {
  case FixpipeLayout::NZ2ND:
    return std::string("LayoutMode_t::NZ2ND");
  case FixpipeLayout::NZ2DN:
    return std::string("LayoutMode_t::NZ2DN");
  case FixpipeLayout::NZ2NZ:
    return std::string("LayoutMode_t::NZ2NZ");
  }
  return failure();
}

FailureOr<std::string> getFixpipeQuantToken(FixpipeQuant quant) {
  switch (quant) {
  case FixpipeQuant::NoConvert:
    return std::string("QuantMode_t::NoQuant");
  case FixpipeQuant::F32F16:
    return std::string("QuantMode_t::F322F16");
  case FixpipeQuant::F32BF16:
    return std::string("QuantMode_t::F322BF16");
  case FixpipeQuant::REQ8Scalar:
    return std::string("QuantMode_t::REQ8");
  case FixpipeQuant::REQ8Vec:
    return std::string("QuantMode_t::VREQ8");
  case FixpipeQuant::DEQF16Scalar:
    return std::string("QuantMode_t::DEQF16");
  case FixpipeQuant::DEQF16Vec:
    return std::string("QuantMode_t::VDEQF16");
  case FixpipeQuant::QF322B8PreScalar:
    return std::string("QuantMode_t::QF322B8_PRE");
  case FixpipeQuant::QF322B8PreVec:
    return std::string("QuantMode_t::VQF322B8_PRE");
  case FixpipeQuant::QF322F16PreScalar:
    return std::string("QuantMode_t::QF322F16_PRE");
  case FixpipeQuant::QF322BF16PreScalar:
    return std::string("QuantMode_t::QF322BF16_PRE");
  case FixpipeQuant::QS322BF16PreScalar:
    return std::string("QuantMode_t::QS322BF16_PRE");
  case FixpipeQuant::QS322BF16PreVec:
    return std::string("QuantMode_t::VQS322BF16_PRE");
  case FixpipeQuant::QF322HIF8PreScalar:
    return std::string("QuantMode_t::QF322HIF8_PRE");
  case FixpipeQuant::QF322FP8PreScalar:
    return std::string("QuantMode_t::QF322FP8_PRE");
  }
  return failure();
}

FailureOr<std::string> getFixpipeReluToken(FixpipeRelu relu) {
  switch (relu) {
  case FixpipeRelu::NoRelu:
    return std::string("ReluPreMode::NoRelu");
  case FixpipeRelu::NormalRelu:
    return std::string("ReluPreMode::NormalRelu");
  }
  return failure();
}

int getGlobalTensorElementBytes(Type elemTy) {
  return static_cast<int>(getPTOStorageElemByteSize(elemTy));
}

std::string getGlobalTensorTypeStringFromShapeAndStrides(
    Type elemTy, ArrayRef<int64_t> shape, ArrayRef<int64_t> strides,
    StringRef layoutEnum) {
  SmallVector<int64_t, 5> shape5D;
  SmallVector<int64_t, 5> stride5D;
  buildGlobalTensorShapeAndStride(shape, strides, shape5D, stride5D);

  std::string elemTypeStr = getElemTypeStringForGT(elemTy);
  std::string shapeType = "pto::Shape<" + joinIntTemplateParams(shape5D) + ">";
  std::string strideType =
      "pto::Stride<" + joinIntTemplateParams(stride5D) + ">";
  return "GlobalTensor<" + elemTypeStr + ", " + shapeType + ", " +
         strideType + ", " + layoutEnum.str() + ">";
}

Location getIndexedNameHintLoc(Location fallbackLoc, unsigned index) {
  SmallVector<std::string, 4> hints;
  appendRawLocationNameHints(fallbackLoc, hints);
  if (index >= hints.size() || hints[index].empty())
    return fallbackLoc;
  return NameLoc::get(StringAttr::get(fallbackLoc.getContext(), hints[index]),
                      fallbackLoc);
}

int64_t getIntegerAttrSignedValue(IntegerAttr attr) {
  return getAPIntSignedValue(attr.getValue());
}

StringRef getLastUseAwareCallee(Operation *op, StringRef callee,
                                 std::string &storage,
                                 ArrayRef<unsigned> tileSlotOrder) {
  std::optional<std::string> marker =
      buildLastUseMarkerCallee(op, callee, tileSlotOrder);
  if (!marker)
    return callee;
  storage = std::move(*marker);
  return storage;
}

std::optional<mlir::pto::Layout> getLayoutAttrFromOp(Operation *op) {
  if (!op) {
    return std::nullopt;
  }
  if (auto attr = op->getAttrOfType<mlir::pto::LayoutAttr>("layout")) {
    return attr.getLayout();
  }
  return std::nullopt;
}

std::optional<mlir::pto::Layout> getLayoutAttrFromViewType(Type type) {
  if (auto tensorView = dyn_cast<pto::TensorViewType>(type)) {
    if (auto layout = tensorView.getLayoutAttr()) {
      return layout.getLayout();
    }
  }
  if (auto partitionView = dyn_cast<pto::PartitionTensorViewType>(type)) {
    if (auto layout = partitionView.getLayoutAttr()) {
      return layout.getLayout();
    }
  }
  return std::nullopt;
}

emitc::OpaqueType getRuntimeGlobalTensorOpaqueType(
    MLIRContext *ctx, Type elemTy, ArrayRef<int64_t> shape,
    StringRef layoutEnum) {
  SmallVector<int64_t, 5> shape5D(5, 1);
  SmallVector<int64_t, 5> stride5D(5, -1);
  int64_t shift = 5 - static_cast<int64_t>(shape.size());
  for (auto [index, dim] : llvm::enumerate(shape))
    shape5D[shift + static_cast<int64_t>(index)] =
        ShapedType::isDynamic(dim) ? -1 : dim;

  std::string elemTypeStr = getElemTypeStringForGT(elemTy);
  std::string shapeType = "pto::Shape<" + joinIntTemplateParams(shape5D) + ">";
  std::string strideType =
      "pto::Stride<" + joinIntTemplateParams(stride5D) + ">";
  return emitc::OpaqueType::get(
      ctx, "GlobalTensor<" + elemTypeStr + ", " + shapeType + ", " +
               strideType + ", " + layoutEnum.str() + ">");
}

emitc::OpaqueType getSignedIntOpaqueType(MLIRContext *ctx,
                                                unsigned bitWidth) {
  switch (bitWidth) {
  case 1:
    return emitc::OpaqueType::get(ctx, "int8_t");
  case 8:
    return emitc::OpaqueType::get(ctx, "int8_t");
  case 16:
    return emitc::OpaqueType::get(ctx, "int16_t");
  case 32:
    return emitc::OpaqueType::get(ctx, "int32_t");
  case 64:
    return emitc::OpaqueType::get(ctx, "int64_t");
  case 128:
    return emitc::OpaqueType::get(ctx, "__int128");
  default:
    llvm::errs() << "[Debug] Unsupported signed integer bitwidth: " << bitWidth
                 << "\n";
    return emitc::OpaqueType::get(ctx, "int64_t");
  }
}

Value getSourceEmitCVariable(Value value) {
  if (value.getDefiningOp<emitc::VariableOp>())
    return value;
  return {};
}

std::optional<SpecialGlobalTensorTypeSpec>
getSpecialGlobalTensorTypeSpecForLayout(std::optional<mlir::pto::Layout> layout,
                                        ArrayRef<int64_t> shape, Type elemTy) {
  if (!layout || !isF8E8M0ElemType(elemTy) || shape.size() != 2)
    return std::nullopt;

  auto alignUp = [](int64_t value, int64_t align) -> int64_t {
    if (value < 0 || align == 0)
      return value;
    return ((value + align - 1) / align) * align;
  };

  std::string elemTypeStr = getEmitCScalarTypeToken(elemTy);
  switch (*layout) {
  case mlir::pto::Layout::MX_A_ZZ: {
    int64_t rows = alignUp(shape[0], 16);
    int64_t cols = alignUp(shape[1], 2);
    return SpecialGlobalTensorTypeSpec{
        "TileShape2D<" + elemTypeStr + ", " + std::to_string(rows) + ", " +
            std::to_string(cols) + ", pto::Layout::MX_A_ZZ>",
        "BaseShape2D<" + elemTypeStr + ", " + std::to_string(rows) + ", " +
            std::to_string(cols) + ", pto::Layout::MX_A_ZZ>",
        "pto::Layout::MX_A_ZZ",
    };
  }
  case mlir::pto::Layout::MX_B_NN: {
    int64_t rows = alignUp(shape[0], 2);
    int64_t cols = alignUp(shape[1], 16);
    return SpecialGlobalTensorTypeSpec{
        "TileShape2D<" + elemTypeStr + ", " + std::to_string(rows) + ", " +
            std::to_string(cols) + ", pto::Layout::MX_B_NN>",
        "BaseShape2D<" + elemTypeStr + ", " + std::to_string(rows) + ", " +
            std::to_string(cols) + ", pto::Layout::MX_B_NN>",
        "pto::Layout::MX_B_NN",
    };
  }
  default:
    return std::nullopt;
  }
}

std::optional<SpecialGlobalTensorTypeSpec>
getSpecialScaleGlobalTensorTypeSpec(Operation *anchor, MemRefType mrTy) {
  auto load = dyn_cast_or_null<pto::TLoadOp>(anchor);
  if (!load)
    return std::nullopt;
  return getSpecialScaleGlobalTensorTypeSpecForTileValue(
      load.getDst(), mrTy.getShape(), mrTy.getElementType());
}

std::optional<SpecialGlobalTensorTypeSpec>
getSpecialScaleGlobalTensorTypeSpecForTileValue(Value dstValue,
                                                ArrayRef<int64_t> shape,
                                                Type elemTy) {
  dstValue = peelUnrealized(dstValue);

  auto dstTileTy = dyn_cast<pto::TileBufType>(dstValue.getType());
  if (!dstTileTy)
    return std::nullopt;

  auto dstSpace = dyn_cast_or_null<pto::AddressSpaceAttr>(
      dstTileTy.getMemorySpace());
  if (!dstSpace || dstSpace.getAddressSpace() != pto::AddressSpace::MAT)
    return std::nullopt;

  ArrayRef<int64_t> effectiveShape = dstTileTy.getShape();
  if (effectiveShape.empty())
    effectiveShape = shape;
  auto config = dstTileTy.getConfigAttr();
  if (!isF8E8M0ElemType(elemTy))
    return std::nullopt;
  if (effectiveShape.size() != 2)
    return std::nullopt;

  pto::BLayout blayout = getTileBufBLayoutValue(config);
  pto::SLayout slayout = getTileBufSLayoutValue(config);
  std::string elemTypeStr = getEmitCScalarTypeToken(elemTy);

  if (blayout == pto::BLayout::RowMajor &&
      slayout == pto::SLayout::RowMajor) {
    if (effectiveShape[0] == 1)
      return std::nullopt;
    return SpecialGlobalTensorTypeSpec{
        "TileShape2D<" + elemTypeStr + ", " +
            std::to_string(effectiveShape[0]) + ", " +
            std::to_string(effectiveShape[1]) + ", pto::Layout::MX_A_ZZ>",
        "BaseShape2D<" + elemTypeStr + ", " +
            std::to_string(effectiveShape[0]) + ", " +
            std::to_string(effectiveShape[1]) + ", pto::Layout::MX_A_ZZ>",
        "pto::Layout::MX_A_ZZ",
    };
  }

  if (blayout == pto::BLayout::ColMajor &&
      slayout == pto::SLayout::ColMajor) {
    return SpecialGlobalTensorTypeSpec{
        "TileShape2D<" + elemTypeStr + ", " +
            std::to_string(effectiveShape[0]) + ", " +
            std::to_string(effectiveShape[1]) + ", pto::Layout::MX_B_NN>",
        "BaseShape2D<" + elemTypeStr + ", " +
            std::to_string(effectiveShape[0]) + ", " +
            std::to_string(effectiveShape[1]) + ", pto::Layout::MX_B_NN>",
        "pto::Layout::MX_B_NN",
    };
  }

  return std::nullopt;
}

bool getStaticMemrefLayout(MemRefType mrTy, SmallVectorImpl<int64_t> &strides,
                                  int64_t &offset) {
  if (failed(
          mlir::pto::getPTOMemRefStridesAndOffset(mrTy, strides, offset))) {
    strides.clear();
    int64_t stride = 1;
    ArrayRef<int64_t> shape = mrTy.getShape();
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
      strides.push_back(stride);
      stride *= shape[i];
    }
    std::reverse(strides.begin(), strides.end());
    offset = 0;
  }
  return offset != ShapedType::kDynamic &&
         llvm::none_of(strides, [](int64_t strideValue) {
           return strideValue == ShapedType::kDynamic;
         });
}

// Stable, content-derived C++ type name for a !pto.struct, e.g.
// !pto.struct<f16, i8> -> "PtoStruct_f16_i8". A pure function of the type, so
// the type converter and the file-scope definition emitter agree without any
// shared state.
std::string getStructTypeName(pto::StructType st) {
  std::string s = "PtoStruct";
  for (Type f : st.getFieldTypes())
    s += "_" + mangleStructFieldType(f);
  return s;
}

FailureOr<std::string> getTPipeTokenFromValue(Value pipeHandle,
                                                     PTOArch targetArch) {
  pipeHandle = peelUnrealized(pipeHandle);
  Operation *def = pipeHandle.getDefiningOp();
  if (!def)
    return failure();
  return buildTPipeTokenFromInitOp(def, targetArch);
}

pto::BLayout getTileBufBLayoutValue(pto::TileBufConfigAttr configAttr) {
  if (auto blAttr = dyn_cast<BLayoutAttr>(configAttr.getBLayout()))
    return blAttr.getValue();
  return pto::BLayout::RowMajor;
}

pto::SLayout getTileBufSLayoutValue(pto::TileBufConfigAttr configAttr) {
  if (auto slAttr = dyn_cast<SLayoutAttr>(configAttr.getSLayout()))
    return slAttr.getValue();
  return pto::SLayout::NoneBox;
}

Type getTileDataResultType(MLIRContext *ctx, pto::AddressSpace as,
                                  StringRef elemTok) {
  if (tileDataReturnsIntegralAddress(as))
    return emitc::OpaqueType::get(ctx, "uint64_t");
  return getEmitCPointerType(ctx, addrSpaceQualifier(as), elemTok);
}

FailureOr<std::string> getTileSplitToken(int64_t split) {
  switch (split) {
  case 0:
    return std::string("TileSplitAxis::TILE_NO_SPLIT");
  case 1:
    return std::string("TileSplitAxis::TILE_UP_DOWN");
  case 2:
    return std::string("TileSplitAxis::TILE_LEFT_RIGHT");
  case 3:
    return std::string("TileSplitAxis::TILE_UP_DOWN_ODD");
  case 4:
    return std::string("TileSplitAxis::TILE_LEFT_RIGHT_ODD");
  default:
    return failure();
  }
}

emitc::OpaqueType getUnsignedIntOpaqueType(MLIRContext *ctx,
                                                  unsigned bitWidth) {
  switch (bitWidth) {
  case 1:
    return emitc::OpaqueType::get(ctx, "uint8_t");
  case 8:
    return emitc::OpaqueType::get(ctx, "uint8_t");
  case 16:
    return emitc::OpaqueType::get(ctx, "uint16_t");
  case 32:
    return emitc::OpaqueType::get(ctx, "uint32_t");
  case 64:
    return emitc::OpaqueType::get(ctx, "uint64_t");
  case 128:
    return emitc::OpaqueType::get(ctx, "unsigned __int128");
  default:
    llvm::errs() << "[Debug] Unsupported unsigned integer bitwidth: "
                 << bitWidth << "\n";
    return emitc::OpaqueType::get(ctx, "uint64_t");
  }
}

emitc::OpaqueType getWiderSignedIntOpaqueType(MLIRContext *ctx,
                                                     unsigned bitWidth) {
  switch (bitWidth) {
  case 1:
  case 8:
    return getSignedIntOpaqueType(ctx, 16);
  case 16:
    return getSignedIntOpaqueType(ctx, 32);
  case 32:
    return getSignedIntOpaqueType(ctx, 64);
  case 64:
    return getSignedIntOpaqueType(ctx, 128);
  default:
    return getSignedIntOpaqueType(ctx, 128);
  }
}

emitc::OpaqueType getWiderUnsignedIntOpaqueType(MLIRContext *ctx,
                                                       unsigned bitWidth) {
  switch (bitWidth) {
  case 1:
  case 8:
    return getUnsignedIntOpaqueType(ctx, 16);
  case 16:
    return getUnsignedIntOpaqueType(ctx, 32);
  case 32:
    return getUnsignedIntOpaqueType(ctx, 64);
  case 64:
    return getUnsignedIntOpaqueType(ctx, 128);
  default:
    return getUnsignedIntOpaqueType(ctx, 128);
  }
}

bool hasStaticShape(MemRefType mrTy) {
  return llvm::none_of(mrTy.getShape(), [](int64_t dim) {
    return dim == ShapedType::kDynamic;
  });
}

std::string inferFallbackGlobalTensorLayout(ArrayRef<int64_t> shape,
                                                   ArrayRef<int64_t> strides,
                                                   Type elemTy) {
  auto layout =
      inferLayout5D(shape, strides, getGlobalTensorElementBytes(elemTy));
  return layoutToEmitCString(layout.value_or(Layout::ND));
}

const char *inferScalingRoleFromValue(Value value) {
  auto opaqueTy = dyn_cast<emitc::OpaqueType>(value.getType());
  if (!opaqueTy)
    return nullptr;
  StringRef token = opaqueTy.getValue();
  if (token.contains("TileType::ScaleLeft"))
    return "TileType::ScaleLeft";
  if (token.contains("TileType::ScaleRight"))
    return "TileType::ScaleRight";
  if (token.contains("TileType::Scaling"))
    return "TileType::Scaling";
  return nullptr;
}

LogicalResult insertFixpipeConfigAliases(ModuleOp mop) {
  for (auto funcOp : mop.getOps<func::FuncOp>()) {
    llvm::DenseSet<int32_t> seenIds;
    SmallVector<std::pair<int32_t, std::string>> aliases;
    bool aliasBuildFailed = false;
    funcOp.walk([&](TPushOp tpush) {
      auto accPushEpilogue = getPipeInitAccPushEpilogue(getPipeInitDef(tpush.getPipeHandle()));
      auto pipeId = getFrontendPipeIdFromHandle(tpush.getPipeHandle());
      if (!accPushEpilogue || !pipeId || !seenIds.insert(*pipeId).second)
        return WalkResult::advance();
      auto configTok = buildFixpipeConfigTypeToken(accPushEpilogue);
      if (failed(configTok)) {
        aliasBuildFailed = true;
        return WalkResult::interrupt();
      }
      aliases.emplace_back(*pipeId, *configTok);
      return WalkResult::advance();
    });
    if (aliasBuildFailed)
      return failure();

    if (aliases.empty())
      continue;

    if (funcOp.empty()) {
      funcOp.emitError("cannot insert fixpipe config aliases into an external "
                       "function");
      return failure();
    }

    OpBuilder builder(funcOp.getContext());
    builder.setInsertionPointToStart(&funcOp.front());
    for (const auto &[pipeId, configTok] : aliases) {
      std::string line =
          "using " + buildFixpipeConfigAliasName(pipeId) + " = " + configTok + ";";
      builder.create<emitc::VerbatimOp>(
          funcOp.getLoc(), builder.getStringAttr(line));
    }
  }
  return success();
}

bool isDpsInitOperand(OpOperand &operand) {
  Operation *owner = operand.getOwner();
  if (auto dpsIface = dyn_cast<pto::PTO_DpsInitOpInterface>(owner)) {
    for (OpOperand &init : dpsIface.getDpsInitsMutable()) {
      if (&init == &operand)
        return true;
    }
  }
  return false;
}

bool isEmitCGlobalTensorLikeType(Type ty) {
  auto opaqueTy = dyn_cast<emitc::OpaqueType>(ty);
  return opaqueTy && opaqueTy.getValue().contains("GlobalTensor<");
}

bool isEmitCPointerLikeType(Type ty) {
  if (isa<emitc::PointerType>(ty))
    return true;
  if (auto opaqueTy = dyn_cast<emitc::OpaqueType>(ty))
    return opaqueTy.getValue().ends_with("*");
  return false;
}

bool isEmitCTileLikeType(Type ty) {
  auto opaqueTy = dyn_cast<emitc::OpaqueType>(ty);
  if (!opaqueTy)
    return false;
  StringRef value = opaqueTy.getValue();
  return value.contains("Tile<") || value.contains("ConvTile<");
}

bool isF8E8M0ElemType(Type elemTy) {
  return mlir::pto::isPTOF8E8M0Type(elemTy);
}

bool isLowPrecisionCubeOperandType(Type elemTy) {
  return pto::isPTOFloat8Type(elemTy) || isa<pto::F4E1M2x2Type>(elemTy) ||
         isa<pto::F4E2M1x2Type>(elemTy);
}

bool isSetFFTsPointerLikeType(Type ty) {
  return isEmitCPointerLikeType(ty);
}

std::string joinIntTemplateParams(ArrayRef<int64_t> values) {
  std::string result;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0)
      result += ", ";
    result += std::to_string(values[i]);
  }
  return result;
}

std::string layoutToEmitCString(mlir::pto::Layout layout) {
  switch (layout) {
  case mlir::pto::Layout::ND:
    return "pto::Layout::ND";
  case mlir::pto::Layout::DN:
    return "pto::Layout::DN";
  case mlir::pto::Layout::NZ:
    return "pto::Layout::NZ";
  case mlir::pto::Layout::MX_A_ZZ:
    return "pto::Layout::MX_A_ZZ";
  case mlir::pto::Layout::MX_B_NN:
    return "pto::Layout::MX_B_NN";
  }
  return "pto::Layout::ND";
}

Value loadEmitCVariableIfNeeded(OpBuilder &builder, Location loc,
                                       Value value) {
  (void)builder;
  (void)loc;
  return value;
}

Value makeEmitCIntConstant(ConversionPatternRewriter &rewriter,
                                  Location loc, Type type, int64_t value) {
  return makeEmitCOpaqueConstant(rewriter, loc, type, std::to_string(value));
}

Value makeEmitCOpaqueConstant(ConversionPatternRewriter &rewriter,
                                     Location loc, Type type,
                                     llvm::StringRef literal) {
  auto attr = emitc::OpaqueAttr::get(rewriter.getContext(), literal);
  return rewriter.create<emitc::ConstantOp>(loc, type, attr);
}

std::string mangleStructFieldType(Type t) {
  if (auto st = dyn_cast<pto::StructType>(t)) {
    std::string s = "S";
    for (Type f : st.getFieldTypes())
      s += "_" + mangleStructFieldType(f);
    return s + "_E";
  }
  std::string spelling;
  llvm::raw_string_ostream os(spelling);
  t.print(os);
  return sanitizeIdentifier(os.str());
}

Value materializeAddressAsPointer(ConversionPatternRewriter &rewriter,
                                         Location loc, Value addr,
                                         pto::AddressSpace as,
                                         StringRef elemTok) {
  auto *ctx = rewriter.getContext();
  std::string ptrTyStr =
      std::string(addrSpaceQualifier(as)) + " " + elemTok.str() + "*";
  auto ptrTy = getEmitCPointerType(ctx, addrSpaceQualifier(as), elemTok);
  if (isSetFFTsPointerLikeType(addr.getType())) {
    if (addr.getType() == ptrTy)
      return addr;
    return rewriter.create<emitc::CastOp>(loc, ptrTy, addr).getResult();
  }
  auto castTyAttr =
      rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, ptrTyStr)});
  return rewriter
      .create<emitc::CallOpaqueOp>(loc, ptrTy, "reinterpret_cast",
                                   ArrayAttr{}, castTyAttr,
                                   ValueRange{addr})
      .getResult(0);
}

Value materializeGlobalTensorDataPointer(
    ConversionPatternRewriter &rewriter, Location loc, Value value,
    Type sourceType) {
  Type loweredType = value.getType();
  if (!isEmitCGlobalTensorLikeType(loweredType))
    return value;

  Type elemType;
  if (auto tvTy = dyn_cast<pto::TensorViewType>(sourceType)) {
    elemType = tvTy.getElementType();
  } else if (auto partitionTy =
                 dyn_cast<pto::PartitionTensorViewType>(sourceType)) {
    elemType = partitionTy.getElementType();
  } else if (auto memrefTy = dyn_cast<MemRefType>(sourceType)) {
    elemType = memrefTy.getElementType();
  } else {
    return value;
  }

  auto *ctx = rewriter.getContext();
  std::string elemTypeStr = getElemTypeStringForGT(elemType);
  auto ptrTy = emitc::PointerType::get(
      emitc::OpaqueType::get(ctx, "__gm__ " + elemTypeStr));
  return rewriter
      .create<emitc::CallOpaqueOp>(loc, ptrTy, "PTOAS__GLOBAL_TENSOR_DATA",
                                   ArrayAttr{}, ArrayAttr{}, ValueRange{value})
      .getResult(0);
}

Value materializeTileDataValue(ConversionPatternRewriter &rewriter,
                                      Location loc, Value tile,
                                      pto::AddressSpace as,
                                      StringRef elemTok) {
  auto rawTy = getTileDataResultType(rewriter.getContext(), as, elemTok);
  return rewriter
      .create<emitc::CallOpaqueOp>(loc, rawTy, "PTOAS__TILE_DATA",
                                   ArrayAttr{}, ArrayAttr{},
                                   ValueRange{tile})
      .getResult(0);
}

Value maybeWrapGlobalMemrefAsGlobalTensor(
    ConversionPatternRewriter &rewriter, Location loc, Value loweredValue,
    Type originalType, Operation *anchor, StringRef tag) {
  auto mrTy = dyn_cast<MemRefType>(originalType);
  if (!mrTy) {
    return loweredValue;
  }

  bool isGlobal = true;
  if (auto asAttr =
          dyn_cast_or_null<pto::AddressSpaceAttr>(mrTy.getMemorySpace())) {
    auto as = asAttr.getAddressSpace();
    isGlobal = (as == pto::AddressSpace::GM || as == pto::AddressSpace::Zero);
  }
  if (!isGlobal) {
    return loweredValue;
  }

  Type loweredType = loweredValue.getType();
  if (isEmitCGlobalTensorLikeType(loweredType)) {
    return loweredValue;
  }

  if (Value gt = buildGlobalTensorFromMemref(rewriter, loc, loweredValue, mrTy,
                                             anchor, tag)) {
    return gt;
  }
  return loweredValue;
}



int64_t multiplyOrDynamic(int64_t lhs, int64_t rhs) {
  if (lhs < 0 || rhs < 0)
    return -1;
  return lhs * rhs;
}

bool isA5NoSplitPipeOp(Operation *op) {
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

bool hasExplicitSubblockControl(Operation *op) {
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

Value peelGlobalTensorConversionBridge(Value value) {
  auto cast = value.getDefiningOp<UnrealizedConversionCastOp>();
  if (!cast || cast->getNumOperands() != 1 || cast->getNumResults() != 1)
    return value;

  Value input = cast.getOperand(0);
  if (isEmitCGlobalTensorLikeType(input.getType()) &&
      isEmitCGlobalTensorLikeType(value.getType()))
    return input;
  return value;
}

// Clone the active fixpipe set-quant producer in front of `tpush` so the
// consumer sees a fresh scalar/vector binding. No-op when no producer is
// active for this pipe.
static LogicalResult rematerializeFixpipeQuantForPush(
    TPushOp tpush,
    llvm::DenseMap<int32_t, SetQuantScalarOp> &activeScalarById,
    llvm::DenseMap<int32_t, SetQuantVectorOp> &activeVectorById) {
  auto accPushEpilogue =
      getPipeInitAccPushEpilogue(getPipeInitDef(tpush.getPipeHandle()));
  auto pipeId = getFrontendPipeIdFromHandle(tpush.getPipeHandle());
  if (!accPushEpilogue || !pipeId)
    return success();

  OpBuilder builder(tpush);
  if (isScalarFixpipeQuant(accPushEpilogue.getQuant())) {
    auto it = activeScalarById.find(*pipeId);
    if (it == activeScalarById.end())
      return success();
    auto consumerTileTy = resolveFixpipeConsumerTileType(tpush.getPipeHandle());
    if (failed(consumerTileTy)) {
      tpush.emitOpError(
          "failed to resolve peer consumer tile type for fixpipe quant "
          "rematerialization");
      return failure();
    }
    Operation *cloned = builder.clone(*it->second.getOperation());
    cloned->setAttr(kEmitCScalarOutTypeAttrName,
                    builder.getStringAttr(
                        getEmitCScalarTypeToken((*consumerTileTy).getElementType())));
  } else if (isVectorFixpipeQuant(accPushEpilogue.getQuant())) {
    auto it = activeVectorById.find(*pipeId);
    if (it != activeVectorById.end())
      builder.clone(*it->second.getOperation());
  }
  return success();
}

static LogicalResult processFixpipeQuantBlock(Block &block,
                                            SmallVectorImpl<Operation *> &eraseList) {
  llvm::DenseMap<int32_t, SetQuantScalarOp> activeScalarById;
  llvm::DenseMap<int32_t, SetQuantVectorOp> activeVectorById;
  SmallVector<Operation *> originalOps;
  for (Operation &op : block)
    originalOps.push_back(&op);

  for (Operation *op : originalOps) {
    if (auto setQuantScalar = dyn_cast<SetQuantScalarOp>(op)) {
      activeScalarById[setQuantScalar.getId()] = setQuantScalar;
      eraseList.push_back(op);
    } else if (auto setQuantVector = dyn_cast<SetQuantVectorOp>(op)) {
      activeVectorById[setQuantVector.getId()] = setQuantVector;
      eraseList.push_back(op);
    } else if (auto tpush = dyn_cast<TPushOp>(op)) {
      if (failed(rematerializeFixpipeQuantForPush(tpush, activeScalarById,
                                               activeVectorById)))
        return failure();
    }

    for (Region &region : op->getRegions()) {
      for (Block &nestedBlock : region) {
        if (failed(processFixpipeQuantBlock(nestedBlock, eraseList)))
          return failure();
      }
    }
  }
  return success();
}


std::string renderStructDef(pto::StructType st) {
  std::string s = "struct " + getStructTypeName(st) + " {\n";
  for (auto [i, f] : llvm::enumerate(st.getFieldTypes()))
    s += "  " + renderStructFieldDecl(f, "f" + std::to_string(i)) + "\n";
  return s + "};";
}

LogicalResult rematerializeFixpipeQuantBindings(ModuleOp mop) {
  SmallVector<Operation *> eraseList;
  for (auto funcOp : mop.getOps<func::FuncOp>()) {
    for (Block &block : funcOp.getBlocks()) {
      if (failed(processFixpipeQuantBlock(block, eraseList)))
        return failure();
    }
  }

  for (Operation *op : eraseList)
    op->erase();
  return success();
}


std::string renderStructFieldDecl(Type fieldTy,
                                         const std::string &name) {
  if (auto st = dyn_cast<pto::StructType>(fieldTy))
    return getStructTypeName(st) + " " + name + ";";
  return getEmitCScalarTypeToken(fieldTy) + " " + name + ";";
}

int64_t renderTileTemplateDim(int64_t rawDim, Type elemTy,
                                     pto::BLayout blayout, int dimIdx) {
  if (dimIdx < 0 || dimIdx >= 2)
    return rawDim;
  if (rawDim == ShapedType::kDynamic)
    return rawDim;
  if (!pto::isPTOFloat4PackedType(elemTy))
    return rawDim;
  int packedDim = blayout == pto::BLayout::ColMajor ? 0 : 1;
  return dimIdx == packedDim ? rawDim * 2 : rawDim;
}

FailureOr<TileBufType> resolveFixpipeConsumerTileType(Value pipeHandle) {
  Operation *producerInit = getPipeInitDef(pipeHandle);
  if (!producerInit)
    return failure();

  Type resolvedType;
  bool hasMismatch = false;

  auto collectFromFunc = [&](func::FuncOp funcOp,
                             llvm::function_ref<bool(pto::TPopOp)> matchesPop) {
    funcOp.walk([&](pto::TPopOp pop) {
      if (!matchesPop(pop))
        return WalkResult::advance();
      if (!resolvedType) {
        resolvedType = pop.getTile().getType();
        return WalkResult::advance();
      }
      if (resolvedType != pop.getTile().getType()) {
        hasMismatch = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
  };

  auto peerInitOr = findPeerFixpipeConsumerInit(producerInit);
  if (failed(peerInitOr))
    return failure();

  Value peerPipe = (*peerInitOr)->getResult(0);
  collectFromFunc((*peerInitOr)->getParentOfType<func::FuncOp>(),
                  [&](pto::TPopOp pop) {
                    return peelUnrealized(pop.getPipeHandle()) == peerPipe;
                  });

  if (hasMismatch || !resolvedType)
    return failure();
  auto tileTy = dyn_cast<TileBufType>(resolvedType);
  if (!tileTy)
    return failure();
  return tileTy;
}

std::string resolveGlobalTensorLayout(Operation *anchor, Value basePtr,
                                             ArrayRef<int64_t> shape,
                                             ArrayRef<int64_t> strides,
                                             Type elemTy) {
  if (auto layout = resolveLayoutForGlobalTensor(anchor, basePtr))
    return layoutToEmitCString(*layout);
  return inferFallbackGlobalTensorLayout(shape, strides, elemTy);
}

std::optional<mlir::pto::Layout>
resolveLayoutForGlobalTensor(Operation *anchor, Value basePtr) {
  if (auto layout = getLayoutAttrFromOp(anchor))
    return layout;
  return resolveLayoutFromValueChain(basePtr);
}

std::optional<mlir::pto::Layout> resolveLayoutFromValueChain(Value v) {
  v = peelUnrealized(v);
  while (v) {
    if (auto layout = getLayoutAttrFromViewType(v.getType())) {
      return layout;
    }
    Operation *def = v.getDefiningOp();
    if (!def) {
      break;
    }
    if (auto layout = getLayoutAttrFromOp(def)) {
      return layout;
    }
    if (auto partition = dyn_cast<pto::PartitionViewOp>(def)) {
      v = peelUnrealized(partition.getSource());
      continue;
    }
    if (auto subview = dyn_cast<memref::SubViewOp>(def)) {
      v = peelUnrealized(subview.getSource());
      continue;
    }
    if (auto reinterpret = dyn_cast<memref::ReinterpretCastOp>(def)) {
      v = peelUnrealized(reinterpret.getSource());
      continue;
    }
    if (auto cast = dyn_cast<memref::CastOp>(def)) {
      v = peelUnrealized(cast.getSource());
      continue;
    }
    if (auto unrealized = dyn_cast<UnrealizedConversionCastOp>(def)) {
      if (unrealized->getNumOperands() == 0)
        break;
      v = peelUnrealized(unrealized.getOperand(0));
      continue;
    }
    break;
  }
  return std::nullopt;
}

std::string sanitizeIdentifier(std::string s) {
  for (char &c : s) {
    bool ok = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') ||
              (c >= 'A' && c <= 'Z') || c == '_';
    if (!ok)
      c = '_';
  }
  return s;
}

const char *scalingRoleToken(Type elemTy,
                                    pto::TileBufConfigAttr configAttr) {
  if (!isF8E8M0ElemType(elemTy))
    return "TileType::Scaling";
  pto::BLayout bl = getTileBufBLayoutValue(configAttr);
  pto::SLayout sl = getTileBufSLayoutValue(configAttr);
  if (bl == pto::BLayout::RowMajor && sl == pto::SLayout::RowMajor)
    return "TileType::ScaleLeft";
  if (bl == pto::BLayout::ColMajor && sl == pto::SLayout::ColMajor)
    return "TileType::ScaleRight";
  return "TileType::Scaling";
}

std::string tileBufBLayoutToken(pto::TileBufConfigAttr configAttr) {
  std::string blTok = "BLayout::RowMajor";
  if (auto blAttr = dyn_cast<BLayoutAttr>(configAttr.getBLayout())) {
    if (static_cast<int32_t>(blAttr.getValue()) == 1)
      blTok = "BLayout::ColMajor";
  }
  return blTok;
}

std::string tileBufCompactToken(pto::TileBufConfigAttr configAttr) {
  std::string compactTok = "CompactMode::Null";
  if (auto compactAttr = dyn_cast<CompactModeAttr>(configAttr.getCompactMode())) {
    switch (static_cast<int32_t>(compactAttr.getValue())) {
    case 1:
      compactTok = "CompactMode::Normal";
      break;
    case 2:
      compactTok = "CompactMode::RowPlusOne";
      break;
    default:
      compactTok = "CompactMode::Null";
      break;
    }
  }
  return compactTok;
}

std::string tileBufPadToken(pto::TileBufConfigAttr configAttr) {
  std::string padTok = "PadValue::Null";
  if (auto padAttr = dyn_cast<PadValueAttr>(configAttr.getPad())) {
    switch (static_cast<int32_t>(padAttr.getValue())) {
    case 1:
      padTok = "PadValue::Zero";
      break;
    case 2:
      padTok = "PadValue::Max";
      break;
    case 3:
      padTok = "PadValue::Min";
      break;
    default:
      padTok = "PadValue::Null";
      break;
    }
  }
  return padTok;
}

std::string tileBufSLayoutToken(pto::TileBufConfigAttr configAttr) {
  std::string slTok = "SLayout::NoneBox";
  if (auto slAttr = dyn_cast<SLayoutAttr>(configAttr.getSLayout())) {
    int32_t slVal = static_cast<int32_t>(slAttr.getValue());
    slTok = (slVal == 1) ? "SLayout::RowMajor"
                         : (slVal == 2) ? "SLayout::ColMajor"
                                        : "SLayout::NoneBox";
  }
  return slTok;
}

bool tileDataReturnsIntegralAddress(pto::AddressSpace as) {
  return as == pto::AddressSpace::BIAS;
}

const char *tileRoleToken(Attribute memorySpace, std::optional<Type> elemType,
                          std::optional<pto::TileBufConfigAttr> configAttr) {
  if (auto asAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(memorySpace)) {
    switch (asAttr.getAddressSpace()) {
    case pto::AddressSpace::VEC:
      return "TileType::Vec";
    case pto::AddressSpace::MAT:
      return "TileType::Mat";
    case pto::AddressSpace::LEFT:
      return "TileType::Left";
    case pto::AddressSpace::RIGHT:
      return "TileType::Right";
    case pto::AddressSpace::ACC:
      return "TileType::Acc";
    case pto::AddressSpace::BIAS:
      return "TileType::Bias";
    case pto::AddressSpace::SCALING:
      if (elemType && configAttr)
        return scalingRoleToken(*elemType, *configAttr);
      return "TileType::Scaling";
    case pto::AddressSpace::GM:
    case pto::AddressSpace::Zero:
      return "TileType::Vec";
    }
  }
  return "TileType::Vec";
}

std::string evtTokFromEventAttr(mlir::pto::EventAttr a) {
  return mlir::pto::stringifyEVENT(a.getEvent()).str();
}

std::string pipeTokFromPipeAttr(mlir::pto::PipeAttr a) {
  return mlir::pto::stringifyPIPE(a.getPipe()).str();
}

std::string pipeTokFromPipeEnum(mlir::pto::PIPE p) {
  return mlir::pto::stringifyPIPE(p).str();
}

std::string maskPatternTok(mlir::pto::MaskPatternAttr a) {
  auto v = a.getValue(); // enum
  return (std::string("pto::MaskPattern::") + mlir::pto::stringifyMaskPattern(v).str());
}

PTOToEmitCTypeConverter::PTOToEmitCTypeConverter(MLIRContext *Ctx,
                                                   PTOArch targetArch) {
  (void)targetArch;
  registerBasicConversions(Ctx);
  registerPTOValueConversions(Ctx);
  registerMemRefConversions(Ctx);
  registerFunctionAndMaterializations();
}


void PTOToEmitCTypeConverter::registerBasicConversions(MLIRContext *Ctx) {
  registerFloatConversions(Ctx);
  registerIntegerConversions(Ctx);
}

// Floating-point and low-precision element types.
void PTOToEmitCTypeConverter::registerFloatConversions(MLIRContext *Ctx) {
// 1. 基本类型 (f32, i32, index)
// ---------------------------------------------------------
addConversion([Ctx](FloatType type) -> Type {
  if (pto::isPTOFloat8E4M3LikeType(type)) {
    return emitc::OpaqueType::get(Ctx, "float8_e4m3_t");
  }
  if (pto::isPTOFloat8E5M2LikeType(type)) {
    return emitc::OpaqueType::get(Ctx, "float8_e5m2_t");
  }
  if (type.isF32()) {
    return emitc::OpaqueType::get(Ctx, "float");
  }
  if (type.isF16()) {
    return emitc::OpaqueType::get(Ctx, "half");
  }
  if (type.isBF16()) {
    return emitc::OpaqueType::get(Ctx, "bfloat16_t");
  }
  if (type.isF64()) {
    return emitc::OpaqueType::get(Ctx, "double");
  }
  llvm::errs() << "[Debug] Unsupported FloatType: " << type << "\n";
  return Type{};
});

addConversion([Ctx](pto::HiF8Type) -> Type {
  return emitc::OpaqueType::get(Ctx, "hifloat8_t");
});
addConversion([Ctx](Type type) -> std::optional<Type> {
  if (isF8E8M0ElemType(type))
    return emitc::OpaqueType::get(Ctx, "float8_e8m0_t");
  return std::nullopt;
});
addConversion([Ctx](pto::F4E1M2x2Type) -> Type {
  return emitc::OpaqueType::get(Ctx, "float4_e1m2x2_t");
});
addConversion([Ctx](pto::F4E2M1x2Type) -> Type {
  return emitc::OpaqueType::get(Ctx, "float4_e2m1x2_t");
});
}

// Integer/index types, including the 64-bit index lowering.
void PTOToEmitCTypeConverter::registerIntegerConversions(MLIRContext *Ctx) {addConversion([Ctx](IntegerType type) -> Type {
  if (type.getWidth() == 1)
    return type;

  // Prefer fixed-width C types. Preserve signedness if the MLIR integer is
  // explicitly signed/unsigned; treat signless as signed by default.
  const bool isUnsigned = type.isUnsignedInteger();
  switch (type.getWidth()) {
  case 8:
    return emitc::OpaqueType::get(Ctx, isUnsigned ? "uint8_t" : "int8_t");
  case 16:
    return emitc::OpaqueType::get(Ctx,
                                  isUnsigned ? "uint16_t" : "int16_t");
  case 32:
    return emitc::OpaqueType::get(Ctx,
                                  isUnsigned ? "uint32_t" : "int32_t");
  case 64:
    return emitc::OpaqueType::get(Ctx,
                                  isUnsigned ? "uint64_t" : "int64_t");
  default:
    llvm::errs() << "[Debug] Unsupported IntegerType width: "
                 << type.getWidth() << "\n";
    return emitc::OpaqueType::get(Ctx, "int32_t"); // Fallback
  }
});

addConversion([Ctx](IndexType type) -> Type {
  return emitc::OpaqueType::get(Ctx, "int64_t");
});

// vector<4xi16> (e.g. TMRGSORT executedNumList) -> pto::MrgSortExecutedNumList
addConversion([Ctx](VectorType type) -> Type {
  if (type.getRank() == 1 && type.getNumElements() == 4 &&
      type.getElementType().isInteger(16))
    return emitc::OpaqueType::get(Ctx, "pto::MrgSortExecutedNumList");
  return Type{};
});

// ---------------------------------------------------------
}

void PTOToEmitCTypeConverter::registerPTOValueConversions(MLIRContext *Ctx) {
  registerPointerAndStructConversions(Ctx);
  registerRuntimeValueConversions(Ctx);
}

// Pipe/event/array/struct/view value types.
void PTOToEmitCTypeConverter::registerPointerAndStructConversions(MLIRContext *Ctx) {
// 2. PTO 特殊类型 (透传或转换)
// ---------------------------------------------------------
addConversion([](emitc::OpaqueType type) { return type; });
addConversion([](emitc::PointerType type) { return type; });

// ---------------------------------------------------------
// 2.5 PtrType 转换 (指针类型)
// ---------------------------------------------------------
addConversion([this, Ctx](pto::PtrType type) -> std::optional<Type> {
  Type elemType = type.getElementType();
  Type newElemType = convertType(elemType);
  if (!newElemType)
    return std::nullopt;

  std::string elemTypeStr;
  if (auto opq = dyn_cast<emitc::OpaqueType>(newElemType)) {
    elemTypeStr = opq.getValue().str();
  } else {
    llvm::errs() << "  [Error] PtrType elem type is not OpaqueType: "
                 << newElemType << "\n";
    return std::nullopt;
  }

  std::string qualifier =
      addrSpaceQualifier(getAddressSpaceOrGM(type.getMemorySpace()));

  return getEmitCPointerType(Ctx, qualifier, elemTypeStr);
});
}

// Runtime handle types (async sessions, views, tile buffers, pipes).
void PTOToEmitCTypeConverter::registerRuntimeValueConversions(MLIRContext *Ctx) {
  registerViewAndHandleConversions(Ctx);
  registerTileConversions(Ctx);
}

// Views, async handles, pipes, events, arrays, structs.
void PTOToEmitCTypeConverter::registerViewAndHandleConversions(MLIRContext *Ctx) {addConversion([Ctx](pto::PipeType type) -> Type {
  (void)type;
  return emitc::OpaqueType::get(Ctx, "auto");
});

addConversion([Ctx](pto::EventIdArrayType type) -> Type {
  std::string tok = "PTOAS_EventIdArray<" + std::to_string(type.getSize()) + ">";
  return emitc::OpaqueType::get(Ctx, tok);
});

// !pto.local_array<D1 x D2 x ... x T> -> !emitc.array<D1 x D2 x ... x T>.
// Variables of this type render as `T a[D1][D2]...;` in the emitted C++.
addConversion([this](pto::LocalArrayType type) -> std::optional<Type> {
  Type convertedElem = convertType(type.getElementType());
  if (!convertedElem)
    return std::nullopt;
  return emitc::ArrayType::get(type.getShape(), convertedElem);
});

// !pto.struct<...> -> !emitc.opaque<"PtoStruct_...">. The matching C++
// `struct PtoStruct_... { ... };` definition is emitted at file scope by
// the pass (see runOnOperation), keyed on the same content-derived name.
// A struct is carried as a pointer to its storage. It cannot be carried by
// value (emitc.member needs an lvalue, so every field write would land in a
// copy), and it cannot be carried as an lvalue either: emitc.func rejects
// an lvalue argument outright, and the C++ emitter refuses one on func.func
// too, which would make a struct impossible to pass to a helper function.
// A pointer is legal in a signature and still names the caller's storage.
addConversion([Ctx](pto::StructType type) -> Type {
  return emitc::PointerType::get(
      emitc::OpaqueType::get(Ctx, getStructTypeName(type)));
});

addConversion([Ctx](pto::AsyncSessionType type) -> Type {
  (void)type;
  return emitc::OpaqueType::get(Ctx, "pto::comm::AsyncSession");
});

addConversion([Ctx](pto::AsyncEventType type) -> Type {
  (void)type;
  return emitc::OpaqueType::get(Ctx, "pto::comm::AsyncEvent");
});

addConversion([Ctx](pto::PrefetchAsyncContextType type) -> Type {
  (void)type;
  return emitc::OpaqueType::get(Ctx, "pto::PrefetchAsyncContext");
});

addConversion([Ctx](pto::TensorViewType type) -> Type {
  std::string layout = type.getLayoutAttr()
                           ? layoutToEmitCString(
                                 type.getLayoutAttr().getLayout())
                           : "pto::Layout::ND";
  return getRuntimeGlobalTensorOpaqueType(Ctx, type.getElementType(),
                                          type.getShape(), layout);
});

addConversion([Ctx](pto::PartitionTensorViewType type) -> Type {
  std::string layout = type.getLayoutAttr()
                           ? layoutToEmitCString(
                                 type.getLayoutAttr().getLayout())
                           : "pto::Layout::ND";
  return getRuntimeGlobalTensorOpaqueType(Ctx, type.getElementType(),
                                          type.getShape(), layout);
});
}

// Tile buffer types (tile_buf/alloc_tile handles).
void PTOToEmitCTypeConverter::registerTileConversions(MLIRContext *Ctx) {addConversion([Ctx](pto::TileBufType type) -> std::optional<Type> {
  auto typeString = getEmitCTileTypeString(type);
  if (!typeString)
    return std::nullopt;
  return emitc::OpaqueType::get(Ctx, *typeString);
});

// ---------------------------------------------------------
}

void PTOToEmitCTypeConverter::registerMemRefConversions(MLIRContext *Ctx) {
// 3. MemRef 转换 (Debug 重点)
// ---------------------------------------------------------
addConversion([this, Ctx](MemRefType type) -> std::optional<Type> {
  LLVM_DEBUG(llvm::dbgs() << "Converting MemRef: " << type << "\n");

  // A. 转换元素类型
  Type elemType = type.getElementType();
  Type newElemType = convertType(elemType); 
  if (!newElemType) {
    llvm::errs() << "  [Error] Failed to convert element type: " << elemType << "\n";
    return std::nullopt;
  }
  
  // 获取元素类型的字符串
  std::string elemTypeStr;
  if (auto opq = dyn_cast<emitc::OpaqueType>(newElemType)) {
    elemTypeStr = opq.getValue().str();
  } else {
     llvm::errs() << "  [Error] Converted element type is not OpaqueType: " << newElemType << "\n";
     return std::nullopt;
  }

  // B. 处理 Memory Space
  std::string qualifier = "";
  Attribute memorySpace = type.getMemorySpace();
  
  if (!memorySpace) {
     qualifier = "__gm__";
  } else if (auto ptoAttr = dyn_cast<pto::AddressSpaceAttr>(memorySpace)) {
     qualifier = addrSpaceQualifier(ptoAttr.getAddressSpace());
  } else {
     llvm::errs() << "  [Warning] Unknown MemorySpace Attribute type: " << memorySpace << "\n";
     qualifier = "__gm__"; // Fallback
  }

  std::string finalTypeStr = qualifier + " " + elemTypeStr;
  LLVM_DEBUG(llvm::dbgs() << "  [Success] -> " << finalTypeStr << "*\n");
  
  return getEmitCPointerType(Ctx, finalTypeStr);
});

// ---------------------------------------------------------

}

void PTOToEmitCTypeConverter::registerFunctionAndMaterializations() {
// 4. Function & Materialization
// ---------------------------------------------------------
addConversion([this](FunctionType type) -> Type {
  SmallVector<Type> inputs;
  if (failed(convertTypes(type.getInputs(), inputs))) {
    return Type{};
  }
  SmallVector<Type> results;
  if (failed(convertTypes(type.getResults(), results))) {
    return Type{};
  }
  return FunctionType::get(type.getContext(), inputs, results);
});

auto materializeCast = [](OpBuilder &Builder, Type ResultType,
                          ValueRange Inputs, Location Loc) -> Value {
  if (Inputs.size() != 1) {
    return Value();
  }
  return Builder.create<UnrealizedConversionCastOp>(Loc, ResultType, Inputs[0]).getResult(0);
};

  addSourceMaterialization(materializeCast);
  addTargetMaterialization(materializeCast);
}

} // namespace pto
} // namespace mlir

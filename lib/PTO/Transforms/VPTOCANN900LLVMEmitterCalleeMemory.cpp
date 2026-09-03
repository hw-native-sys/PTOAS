// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "VPTOCANN900LLVMEmitterInternal.h"

namespace mlir::pto::detail {

FailureOr<StringRef> buildCopyGmToUbCallee(MLIRContext *context, Type sourceType) {
  auto ptrType = dyn_cast<pto::PtrType>(sourceType);
  if (!ptrType) {
    return failure();
  }
  Type elementType = ptrType.getElementType();
  if ((isa<IntegerType>(elementType) && cast<IntegerType>(elementType).getWidth() == 64) || elementType.isF64()) {
    return StringAttr::get(context, "llvm.hivm.MOV.OUT.TO.UB.ALIGN.V2.s32.DV").getValue();
  }
  std::string elem = getCopyElementFragment(elementType);
  if (elem.empty()) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.MOV.OUT.TO.UB.ALIGN.V2." + elem + ".DV").getValue();
}

StringRef buildCopyUbToGmCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.MOV.UB.TO.OUT.ALIGN.V2.DV").getValue();
}

StringRef buildCopyUbToUbCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.MOV.UB.TO.UB.v310").getValue();
}

StringRef buildCopyCbufToUbCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.MOV.L1.TO.UB.v310").getValue();
}

StringRef buildCopyUbToCbufCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.MOV.UB.TO.L1.v310").getValue();
}

FailureOr<StringRef> buildOrdinaryMadCallee(MLIRContext *context, pto::MadRawOpInterface op) {
  auto lhsType = dyn_cast<pto::PtrType>(op.getLhs().getType());
  auto rhsType = dyn_cast<pto::PtrType>(op.getRhs().getType());
  auto dstType = dyn_cast<pto::PtrType>(op.getDst().getType());
  if (!lhsType || !rhsType || !dstType) {
    return failure();
  }

  return buildMadTypedCalleeName(context, lhsType.getElementType(), rhsType.getElementType(), dstType.getElementType());
}

FailureOr<StringRef> buildMxMadCallee(MLIRContext *context, pto::MadRawOpInterface op) {
  auto lhsType = dyn_cast<pto::PtrType>(op.getLhs().getType());
  auto rhsType = dyn_cast<pto::PtrType>(op.getRhs().getType());
  if (!lhsType || !rhsType) {
    return failure();
  }
  if (isMxElementType(lhsType.getElementType()) && isMxElementType(rhsType.getElementType())) {
    return buildMadMxCalleeName(context, lhsType.getElementType(), rhsType.getElementType());
  }
  return failure();
}

FailureOr<StringRef> buildCopyGmToCbufCallee(MLIRContext *context, Type sourceType) {
  auto ptrType = dyn_cast<pto::PtrType>(sourceType);
  if (!ptrType) {
    return failure();
  }
  std::string elem = getCopyElementFragment(ptrType.getElementType());
  if (elem.empty()) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.MOV.OUT.TO.L1.ALIGN.V2." + elem + ".DV").getValue();
}

FailureOr<StringRef> buildCopyGmToCbufMultiNd2NzCallee(MLIRContext *context, Type sourceType) {
  auto ptrType = dyn_cast<pto::PtrType>(sourceType);
  if (!ptrType) {
    return failure();
  }
  std::string elem = getNd2NzCopyElementFragment(ptrType.getElementType());
  if (elem.empty()) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.MOV.OUT.TO.L1.MULTI.ND2NZ." + elem + ".V310").getValue();
}

std::string getDn2NzCopyElementFragment(Type type) {
  auto ptrType = dyn_cast<pto::PtrType>(type);
  if (!ptrType) {
    return {};
  }

  Type elementType = ptrType.getElementType();
  std::string typeText;
  llvm::raw_string_ostream os(typeText);
  elementType.print(os);
  os.flush();
  std::string lower = StringRef(typeText).lower();
  if (StringRef(lower).contains("e4m3") || StringRef(lower).contains("e5m2") || StringRef(lower).contains("e8m0") ||
      StringRef(lower).contains("hif8")) {
    return "u8";
  }

  if (elementType.isF16() || elementType.isBF16()) {
    return "u16";
  }
  if (elementType.isF32()) {
    return "u32";
  }

  if (auto intType = dyn_cast<IntegerType>(elementType)) {
    switch (intType.getWidth()) {
    case 8:
      return "u8";
    case 16:
      return "u16";
    case 32:
      return "u32";
    default:
      return {};
    }
  }
  return {};
}

FailureOr<StringRef> buildCopyGmToCbufMultiDn2NzCallee(MLIRContext *context, Type sourceType) {
  auto ptrType = dyn_cast<pto::PtrType>(sourceType);
  if (!ptrType) {
    return failure();
  }
  std::string elem = getDn2NzCopyElementFragment(sourceType);
  if (elem.empty()) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.MOV.OUT.TO.L1.MULTI.DN2NZ." + elem).getValue();
}

FailureOr<StringRef> buildLoadCbufToCaCallee(MLIRContext *context, Type sourceType) {
  auto ptrType = dyn_cast<pto::PtrType>(sourceType);
  if (!ptrType) {
    return failure();
  }
  std::string elem = getL0LoadElementFragment(ptrType.getElementType());
  if (elem.empty()) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.LOAD.L1.TO.L0A.2Dv2." + elem).getValue();
}

FailureOr<StringRef> buildLoadCbufToCbCallee(MLIRContext *context, Type sourceType) {
  auto ptrType = dyn_cast<pto::PtrType>(sourceType);
  if (!ptrType) {
    return failure();
  }
  std::string elem = getL0LoadElementFragment(ptrType.getElementType());
  if (elem.empty()) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.LOAD.L1.TO.L0B.2Dv2." + elem).getValue();
}

FailureOr<StringRef> buildLoadCbufToCaS4Callee(MLIRContext *context, Type sourceType) {
  auto ptrType = dyn_cast<pto::PtrType>(sourceType);
  if (!ptrType) {
    return failure();
  }
  Type elementType = ptrType.getElementType();
  if (!isa<pto::F4E1M2x2Type, pto::F4E2M1x2Type>(elementType)) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.LOAD.L1.TO.L0A.2Dv2.s4").getValue();
}

FailureOr<StringRef> buildLoadCbufToCbS4Callee(MLIRContext *context, Type sourceType) {
  auto ptrType = dyn_cast<pto::PtrType>(sourceType);
  if (!ptrType) {
    return failure();
  }
  Type elementType = ptrType.getElementType();
  if (!isa<pto::F4E1M2x2Type, pto::F4E2M1x2Type>(elementType)) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.LOAD.L1.TO.L0B.2Dv2.s4").getValue();
}

StringRef buildLoadCbufToCaMxCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.LOAD.L1.TO.L0A.MX.2Dv2.v").getValue();
}

[[maybe_unused]] StringRef buildLoadCbufToCbMxCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.LOAD.L1.TO.L0B.MX.2Dv2.v").getValue();
}

StringRef buildCopyMatrixCcToGmCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.FIX.L0C.TO.OUT.f32.EXT").getValue();
}

StringRef buildCopyMatrixCcToCbufCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.FIX.L0C.TO.L1.f32.EXT").getValue();
}

FailureOr<StringRef> buildCopyMatrixCcToUbCallee(MLIRContext *context, Type destinationType) {
  auto ptrType = dyn_cast<pto::PtrType>(destinationType);
  if (!ptrType) {
    return failure();
  }
  Type dstElem = ptrType.getElementType();
  if (dstElem.isF16()) {
    return StringAttr::get(context, "llvm.hivm.FIX.L0C.TO.UB.f322f16.EXT").getValue();
  }
  if (dstElem.isF32()) {
    return StringAttr::get(context, "llvm.hivm.FIX.L0C.TO.UB.f32.EXT").getValue();
  }
  return failure();
}

FailureOr<StringRef> buildCopyCbufToBtCallee(pto::CopyCbufToBtOp op) {
  auto ptrType = dyn_cast<pto::PtrType>(op.getSource().getType());
  if (!ptrType) {
    return failure();
  }
  Type srcElem = ptrType.getElementType();
  if (srcElem.isF16()) {
    return StringAttr::get(op.getContext(), "llvm.hivm.MOV.L1.TO.BT.f16").getValue();
  }
  if (srcElem.isBF16()) {
    return StringAttr::get(op.getContext(), "llvm.hivm.MOV.L1.TO.BT.bf16").getValue();
  }
  if (srcElem.isF32()) {
    return StringAttr::get(op.getContext(), "llvm.hivm.MOV.L1.TO.BT.f32").getValue();
  }
  if (auto intType = dyn_cast<IntegerType>(srcElem); intType && intType.getWidth() == 32) {
    return StringAttr::get(op.getContext(), "llvm.hivm.MOV.L1.TO.BT.s32").getValue();
  }
  return failure();
}

StringRef buildCopyCbufToFbufCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.MOV.L1.TO.FB.v220").getValue();
}

StringRef buildPstiCallee(MLIRContext *context, bool post) {
  return StringAttr::get(context, post ? "llvm.hivm.psti.post.b8" : "llvm.hivm.psti.b8").getValue();
}

StringRef buildPstsCallee(MLIRContext *context, bool post) {
  return StringAttr::get(context, post ? "llvm.hivm.psts.post.b8" : "llvm.hivm.psts.b8").getValue();
}

StringRef buildPldiCallee(MLIRContext *context, bool post) {
  return StringAttr::get(context, post ? "llvm.hivm.pldi.post.b8" : "llvm.hivm.pldi.b8").getValue();
}

StringRef buildPldsCallee(MLIRContext *context, bool post) {
  return StringAttr::get(context, post ? "llvm.hivm.plds.post.b8" : "llvm.hivm.plds.b8").getValue();
}

StringRef buildPnotCallee(MLIRContext *context) { return StringAttr::get(context, "llvm.hivm.pnot.z").getValue(); }

StringRef buildPselCallee(MLIRContext *context) { return StringAttr::get(context, "llvm.hivm.psel").getValue(); }

StringRef buildPandCallee(MLIRContext *context) { return StringAttr::get(context, "llvm.hivm.pand.z").getValue(); }

StringRef buildPorCallee(MLIRContext *context) { return StringAttr::get(context, "llvm.hivm.por.z").getValue(); }

StringRef buildPxorCallee(MLIRContext *context) { return StringAttr::get(context, "llvm.hivm.pxor.z").getValue(); }

StringRef buildPpackCallee(MLIRContext *context) { return StringAttr::get(context, "llvm.hivm.ppack.z").getValue(); }

StringRef buildPunpackCallee(MLIRContext *context) { return StringAttr::get(context, "llvm.hivm.punpack").getValue(); }

FailureOr<StringRef> buildInterleaveCallee(MLIRContext *context, Type resultType, StringRef stem) {
  // bf16x2 has no dedicated vintlv/vdintlv intrinsic. It is a 32-bit packed
  // pair lowered to i32 at the LLVM ABI, and (de)interleave is a bit-level
  // lane shuffle, so the <N x i32> intrinsic serves the <N x bf16x2> type.
  if (pto::isPTOBF16x2Type(getElementTypeFromVectorLike(resultType))) {
    auto lanes = getElementCountFromVectorLike(resultType);
    if (lanes) {
      return StringAttr::get(context, "llvm.hivm." + stem.str() + ".v" + std::to_string(*lanes) + "i32").getValue();
    }
  }
  std::string vec = getCANN900VectorTypeFragment(resultType);
  if (vec.empty()) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm." + stem.str() + "." + vec).getValue();
}

FailureOr<StringRef> buildUnpackCallee(MLIRContext *context, Type inputType, Type resultType, StringRef stem) {
  (void)inputType;
  std::string vec = getCANN900VectorTypeFragment(resultType);
  if (vec.empty()) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm." + stem.str() + "." + vec).getValue();
}

FailureOr<StringRef> buildVpackCallee(MLIRContext *context, Type inputType, Type resultType) {
  (void)resultType;
  std::string vec = getCANN900VectorTypeFragment(inputType);
  if (vec.empty()) {
    return failure();
  }

  return StringAttr::get(context, "llvm.hivm.vpack.x." + vec).getValue();
}

FailureOr<StringRef> buildVsqzCallee(MLIRContext *context, Type resultType) {
  return buildCANN900ModeTypedCallee(context, resultType, "vsqz", "x");
}

FailureOr<StringRef> buildVusqzCallee(MLIRContext *context, Type resultType) {
  return buildCANN900ModeTypedCallee(context, resultType, "vusqz", "m");
}

FailureOr<StringRef> buildVmulaCallee(MLIRContext *context, Type resultType) {
  return buildCANN900SignedModeTypedCallee(context, resultType, "vmula", "m");
}

FailureOr<StringRef> buildVmullCallee(MLIRContext *context, Type resultType) {
  return buildLaneTypedCallee(context, resultType, "vmull", "");
}

FailureOr<StringRef> buildVldsCallee(MLIRContext *context, Type resultType) {
  std::string vec = getMemoryElementTypeFragment(getElementTypeFromVectorLike(resultType));
  auto lanes = getElementCountFromVectorLike(resultType);
  if (vec.empty() || !lanes) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.vldsx1.v" + std::to_string(*lanes) + vec).getValue();
}

FailureOr<StringRef> buildVldsx2Callee(MLIRContext *context, Type resultType, bool post) {
  std::string vec = getMemoryElementTypeFragment(getElementTypeFromVectorLike(resultType));
  auto lanes = getElementCountFromVectorLike(resultType);
  if (vec.empty() || !lanes) {
    return failure();
  }
  return StringAttr::get(context,
                         "llvm.hivm.vldsx2" + std::string(post ? ".post" : "") + ".v" + std::to_string(*lanes) + vec)
      .getValue();
}

FailureOr<StringRef> buildBlockStridedMemoryCallee(MLIRContext *context, Type vectorType, StringRef stem, bool post) {
  Type elementType = getElementTypeFromVectorLike(vectorType);
  auto lanes = getElementCountFromVectorLike(vectorType);
  if (!elementType || !lanes) {
    return failure();
  }

  std::string element;
  if (auto intType = dyn_cast<IntegerType>(elementType)) {
    element = "i" + std::to_string(intType.getWidth());
  } else if (isLowpPayloadElementType(elementType)) {
    element = "i8";
  } else {
    element = getMemoryElementTypeFragment(elementType);
  }
  if (element.empty()) {
    return failure();
  }

  return StringAttr::get(context, "llvm.hivm." + stem.str() + std::string(post ? ".post" : "") + ".v" +
                                      std::to_string(*lanes) + element)
      .getValue();
}

FailureOr<StringRef> buildVsldbCallee(MLIRContext *context, Type resultType, bool post) {
  return buildBlockStridedMemoryCallee(context, resultType, "vsldb", post);
}

FailureOr<StringRef> buildVstsCallee(MLIRContext *context, Type valueType) {
  std::string vec = getMemoryElementTypeFragment(getElementTypeFromVectorLike(valueType));
  auto lanes = getElementCountFromVectorLike(valueType);
  if (vec.empty() || !lanes) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.vstsx1.v" + std::to_string(*lanes) + vec).getValue();
}

FailureOr<StringRef> buildVstsx2Callee(MLIRContext *context, Type valueType) {
  Type elementType = getElementTypeFromVectorLike(valueType);
  auto lanes = getElementCountFromVectorLike(valueType);
  if (!elementType || !lanes) {
    return failure();
  }

  std::string element = getMemoryElementTypeFragment(elementType);
  if (element.empty()) {
    return failure();
  }

  return StringAttr::get(context, "llvm.hivm.vstsx2.v" + std::to_string(*lanes) + element).getValue();
}

FailureOr<StringRef> buildVsstbCallee(MLIRContext *context, Type valueType, bool post) {
  return buildBlockStridedMemoryCallee(context, valueType, "vsstb", post);
}

Type getVgather2SourceElementType(Type sourceType) {
  if (auto ptrType = dyn_cast<pto::PtrType>(sourceType)) {
    return ptrType.getElementType();
  }
  if (auto memrefType = dyn_cast<BaseMemRefType>(sourceType)) {
    return memrefType.getElementType();
  }
  return {};
}

FailureOr<StringRef> buildVgather2Callee(MLIRContext *context, Type sourceType, Type resultType) {
  Type sourceElemType = getVgather2SourceElementType(sourceType);
  Type resultElemType = getElementTypeFromVectorLike(resultType);
  auto lanes = getElementCountFromVectorLike(resultType);
  if (!sourceElemType || !resultElemType || !lanes) {
    return failure();
  }

  std::string vec;
  int64_t intrinsicLanes = *lanes;
  if (pto::getPTOStorageElemBitWidth(sourceElemType) == 8) {
    vec = getElementTypeFragment(sourceElemType);
    intrinsicLanes *= 2;
  } else {
    vec = getElementTypeFragment(resultElemType);
  }
  if (vec.empty()) {
    return failure();
  }

  return StringAttr::get(context, "llvm.hivm.vgather2.v300.v" + std::to_string(intrinsicLanes) + vec).getValue();
}

std::optional<uint64_t> getFixedVectorBitWidth(Type type) {
  auto vectorType = dyn_cast<VectorType>(type);
  if (!vectorType || vectorType.getRank() != 1 || vectorType.isScalable()) {
    return std::nullopt;
  }
  int64_t lanes = vectorType.getDimSize(0);
  if (lanes <= 0) {
    return std::nullopt;
  }
  auto elementType = dyn_cast<IntegerType>(vectorType.getElementType());
  if (!elementType) {
    return std::nullopt;
  }
  return static_cast<uint64_t>(lanes) * elementType.getWidth();
}

FailureOr<Type> getVgather2OffsetsCarrierType(PatternRewriter &rewriter, Type sourceType, Type resultType,
                                              Type offsetsType) {
  Type sourceElemType = getVgather2SourceElementType(sourceType);
  Type elementType = getElementTypeFromVectorLike(resultType);
  auto lanes = getElementCountFromVectorLike(resultType);
  if (!sourceElemType || !elementType || !lanes || *lanes <= 0) {
    return failure();
  }

  Type carrierType = offsetsType;
  if (pto::getPTOStorageElemBitWidth(elementType) == 16) {
    if (*lanes % 2 != 0) {
      return failure();
    }
    carrierType = VectorType::get({*lanes / 2}, rewriter.getI32Type());
  }

  std::optional<uint64_t> offsetsBits = getFixedVectorBitWidth(offsetsType);
  std::optional<uint64_t> carrierBits = getFixedVectorBitWidth(carrierType);
  if (!offsetsBits || !carrierBits || *offsetsBits != *carrierBits) {
    return failure();
  }
  return carrierType;
}

FailureOr<StringRef> buildVgather2BcCallee(MLIRContext *context, Type resultType) {
  return buildLaneTypedCallee(context, resultType, "vgather2.bc", "");
}

FailureOr<StringRef> buildVgatherbCallee(MLIRContext *context, Type resultType) {
  return buildLaneTypedCallee(context, resultType, "vgatherb.v310", "");
}

FailureOr<StringRef> buildVscatterCallee(MLIRContext *context, Type valueType) {
  return buildLaneTypedCallee(context, valueType, "vscatter", ".v300");
}

FailureOr<Type> getVscatterOffsetsCarrierType(Type offsetsType) { return offsetsType; }

FailureOr<StringRef> buildVaxpyCallee(MLIRContext *context, Type resultType) {
  return buildCANN900ModeTypedCallee(context, resultType, "vaxpy", "m");
}

FailureOr<StringRef> buildVmulscvtCallee(MLIRContext *context, Type inputType, Type resultType) {
  auto inputElemType = getElementTypeFromVectorLike(inputType);
  auto resultElemType = getElementTypeFromVectorLike(resultType);
  auto inputLanes = getElementCountFromVectorLike(inputType);
  auto resultLanes = getElementCountFromVectorLike(resultType);
  if (!inputElemType || !resultElemType || !inputLanes || !resultLanes) {
    return failure();
  }
  if (!inputElemType.isF32() || !resultElemType.isF16() || *inputLanes != 64 || *resultLanes != 128) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.vmulscvt.v128f16").getValue();
}

FailureOr<StringRef> buildVciCallee(MLIRContext *context, Type resultType) {
  std::string vec = getCANN900VectorTypeFragment(resultType);
  if (vec.empty()) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.vci." + vec).getValue();
}

FailureOr<StringRef> buildVtrcCallee(MLIRContext *context, Type resultType) {
  std::string vec = getElementTypeFragment(getElementTypeFromVectorLike(resultType));
  auto lanes = getElementCountFromVectorLike(resultType);
  if (vec.empty() || !lanes) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.vtrc." + vec + ".x").getValue();
}

FailureOr<StringRef> buildVexpdifCallee(MLIRContext *context, Type inputType, Type resultType) {
  Type inputElem = getElementTypeFromVectorLike(inputType);
  Type resultElem = getElementTypeFromVectorLike(resultType);
  auto srcLanes = getElementCountFromVectorLike(inputType);
  if (!srcLanes) {
    return failure();
  }
  if (inputElem.isF16() && resultElem.isF32() && *srcLanes == 128) {
    return StringAttr::get(context, "llvm.hivm.vexpdif.interleave.v128f16").getValue();
  }
  if (inputElem.isF32() && resultElem.isF32() && *srcLanes == 64) {
    return StringAttr::get(context, "llvm.hivm.vexpdif.v64f32").getValue();
  }
  return failure();
}

FailureOr<StringRef> buildVbitsortCallee(MLIRContext *context, pto::VbitsortOp op) {
  Type sourceElemType = cast<pto::PtrType>(op.getSource().getType()).getElementType();
  if (sourceElemType.isF16()) {
    return StringAttr::get(context, "llvm.hivm.VBS32.V300.f16").getValue();
  }
  if (sourceElemType.isF32()) {
    return StringAttr::get(context, "llvm.hivm.VBS32.V300.f32").getValue();
  }
  return failure();
}

FailureOr<StringRef> buildVmrgsort4Callee(MLIRContext *context, pto::Vmrgsort4Op op) {
  Type elemType = cast<pto::PtrType>(op.getDestination().getType()).getElementType();
  if (elemType.isF16()) {
    return StringAttr::get(context, "llvm.hivm.VMRGSORT.f16.V300").getValue();
  }
  if (elemType.isF32()) {
    return StringAttr::get(context, "llvm.hivm.VMRGSORT.f32.V300").getValue();
  }
  return failure();
}

FailureOr<Value> packVmrgsort4SourceAddr(Operation *anchor, Value source0, Value source1, Value source2, Value source3,
                                         Type elemType) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();
  unsigned addrShift = 0;
  if (elemType.isF16()) {
    addrShift = 3;
  } else if (elemType.isF32()) {
    addrShift = 3;
  } else {
    return failure();
  }

  auto packOne = [&](Value source, uint64_t laneShift) -> FailureOr<Value> {
    FailureOr<Value> ubPtr = reinterpretPointerToAddrSpace(anchor, source, 6);
    if (failed(ubPtr)) {
      return failure();
    }
    Value asInt = builder.create<LLVM::PtrToIntOp>(loc, builder.getI64Type(), *ubPtr);
    Value shifted = builder.create<arith::ShRUIOp>(loc, asInt, getI64Constant(builder, loc, addrShift));
    Value masked = builder.create<arith::AndIOp>(loc, shifted, getI64Constant(builder, loc, 0xFFFFULL));
    if (laneShift == 0) {
      return masked;
    }
    return builder.create<arith::ShLIOp>(loc, masked, getI64Constant(builder, loc, laneShift)).getResult();
  };

  FailureOr<Value> low0 = packOne(source0, 0);
  FailureOr<Value> low1 = packOne(source1, 16);
  FailureOr<Value> low2 = packOne(source2, 32);
  FailureOr<Value> low3 = packOne(source3, 48);
  if (failed(low0) || failed(low1) || failed(low2) || failed(low3)) {
    return failure();
  }

  Value packed01 = builder.create<arith::OrIOp>(loc, *low0, *low1);
  Value packed23 = builder.create<arith::OrIOp>(loc, *low2, *low3);
  Value packed = builder.create<arith::OrIOp>(loc, packed01, packed23);
  Type ubPtrTy = LLVM::LLVMPointerType::get(anchor->getContext(), 6);
  return builder.create<LLVM::IntToPtrOp>(loc, ubPtrTy, packed).getResult();
}

FailureOr<VcvtContract> buildVcvtContract(pto::VcvtOp op) {
  Type inputElemType = getElementTypeFromVectorLike(op.getInput().getType());
  Type resultElemType = getElementTypeFromVectorLike(op.getResult().getType());
  if (!inputElemType || !resultElemType) {
    return failure();
  }
  auto contract = lookupVcvtContract(classifyVcvtElemType(inputElemType), classifyVcvtElemType(resultElemType));
  if (!contract) {
    return failure();
  }
  return *contract;
}

bool needsV300CtrlModeForVPTOFunc(func::FuncOp funcOp) {
  if (!pto::isPTOEntryFunction(funcOp) || funcOp.getBlocks().empty()) {
    return false;
  }

  bool needsCtrlSetup = false;
  funcOp.walk([&](pto::VcvtOp vcvtOp) {
    FailureOr<VcvtContract> contract = buildVcvtContract(vcvtOp);
    if (succeeded(contract) && (*contract).requiresSat) {
      needsCtrlSetup = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return needsCtrlSetup;
}

FailureOr<Value> encodeMovPadValue(Location loc, Value value, ConversionPatternRewriter &rewriter) {
  Type type = value.getType();
  Value payload = value;
  unsigned bitWidth = 0;

  if (auto intType = dyn_cast<IntegerType>(type)) {
    bitWidth = intType.getWidth();
  } else if (auto floatType = dyn_cast<FloatType>(type)) {
    bitWidth = floatType.getWidth();
    auto intType = rewriter.getIntegerType(bitWidth);
    payload = rewriter.create<arith::BitcastOp>(loc, intType, value);
  } else {
    return failure();
  }

  if (bitWidth != 8 && bitWidth != 16 && bitWidth != 32) {
    return failure();
  }

  return rewriter.create<arith::ExtUIOp>(loc, rewriter.getI64Type(), payload).getResult();
}

StringRef buildMemBarCallee(MemBarKind kind, MLIRContext *context) {
  switch (kind) {
  case MemBarKind::VV_ALL:
    return StringAttr::get(context, "llvm.hivm.mem.bar.vv.all").getValue();
  case MemBarKind::VST_VLD:
    return StringAttr::get(context, "llvm.hivm.mem.bar.vst.vld").getValue();
  case MemBarKind::VLD_VST:
    return StringAttr::get(context, "llvm.hivm.mem.bar.vld.vst").getValue();
  case MemBarKind::VST_VST:
    return StringAttr::get(context, "llvm.hivm.mem.bar.vst.vst").getValue();
  case MemBarKind::VS_ALL:
    return StringAttr::get(context, "llvm.hivm.mem.bar.vs.all").getValue();
  case MemBarKind::VST_LD:
    return StringAttr::get(context, "llvm.hivm.mem.bar.vst.ld").getValue();
  case MemBarKind::VLD_ST:
    return StringAttr::get(context, "llvm.hivm.mem.bar.vld.st").getValue();
  case MemBarKind::VST_ST:
    return StringAttr::get(context, "llvm.hivm.mem.bar.vst.st").getValue();
  case MemBarKind::SV_ALL:
    return StringAttr::get(context, "llvm.hivm.mem.bar.sv.all").getValue();
  case MemBarKind::ST_VLD:
    return StringAttr::get(context, "llvm.hivm.mem.bar.st.vld").getValue();
  case MemBarKind::LD_VST:
    return StringAttr::get(context, "llvm.hivm.mem.bar.ld.vst").getValue();
  case MemBarKind::ST_VST:
    return StringAttr::get(context, "llvm.hivm.mem.bar.st.vst").getValue();
  case MemBarKind::SS_ALL:
    return StringAttr::get(context, "llvm.hivm.mem.bar.ss.all").getValue();
  case MemBarKind::ST_LD:
    return StringAttr::get(context, "llvm.hivm.mem.bar.st.ld").getValue();
  case MemBarKind::LD_ST:
    return StringAttr::get(context, "llvm.hivm.mem.bar.ld.st").getValue();
  case MemBarKind::ST_ST:
    return StringAttr::get(context, "llvm.hivm.mem.bar.st.st").getValue();
  }
  llvm_unreachable("unexpected membar kind");
}

uint64_t getDsbMemImmediate(DsbMem kind) { return static_cast<uint64_t>(kind); }

uint64_t getDcciCacheLineImmediate(DcciCacheLine kind) { return static_cast<uint64_t>(kind); }

uint64_t getDcciDstImmediate(DcciDst kind) { return static_cast<uint64_t>(kind); }

StringRef buildDcciCallee(unsigned addressSpace, bool hasDst, MLIRContext *context) {
  if (addressSpace == static_cast<unsigned>(pto::AddressSpace::GM)) {
    return StringAttr::get(context, hasDst ? "llvm.hivm.DCCI.DST" : "llvm.hivm.DCCI").getValue();
  }
  if (addressSpace == static_cast<unsigned>(pto::AddressSpace::VEC)) {
    return StringAttr::get(context, hasDst ? "llvm.hivm.DCCI.DST.UB" : "llvm.hivm.DCCI.UB").getValue();
  }
  llvm_unreachable("unexpected dcci address space");
}

StringRef buildBufDynSyncCallee(MLIRContext *context, bool isGetBuf) {
  return StringAttr::get(context, isGetBuf ? "llvm.hivm.GET.BUF.mode" : "llvm.hivm.RLS.BUF.mode").getValue();
}

LogicalResult materializeDecls(ModuleOp module, ArrayRef<PlannedDecl> plannedDecls, llvm::raw_ostream &diagOS) {
  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToStart(&module.getBodyRegion().front());
  for (const PlannedDecl &decl : plannedDecls) {
    if (func::FuncOp existing = module.lookupSymbol<func::FuncOp>(decl.name)) {
      if (existing.getFunctionType() != decl.type) {
        diagOS << "VPTO LLVM emission failed: conflicting declaration for " << decl.name << "\n";
        return failure();
      }
      continue;
    }
    auto func = builder.create<func::FuncOp>(module.getLoc(), decl.name, decl.type);
    func.setPrivate();
  }
  return success();
}

} // namespace mlir::pto::detail

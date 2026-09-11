// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- TensorSupport.cpp - Tensor lowering helpers --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "TensorInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

// CANN Open Software License Agreement Version 2.0 (the "License").
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.

//===- PTOToEmitCTensor.cpp - tensor elementwise op lowering ---------===//
//===----------------------------------------------------------------------===//






//===----------------------------------------------------------------------===//
// pto.tadds lowering -> TADDS(dst, src, scalar)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// pto.taddsc lowering -> TADDSC(dst, src0, scalar, src1)
//===----------------------------------------------------------------------===//

std::string cmpModeTok(pto::CmpModeAttr a) {
  // 生成 "CmpMode::GT" 这种 token
  auto m = a.getValue(); // 取 enum
  switch (m) {
    case pto::CmpMode::EQ: return "CmpMode::EQ";
    case pto::CmpMode::NE: return "CmpMode::NE";
    case pto::CmpMode::LT: return "CmpMode::LT";
    case pto::CmpMode::LE: return "CmpMode::LE";
    case pto::CmpMode::GT: return "CmpMode::GT";
    case pto::CmpMode::GE: return "CmpMode::GE";
  }
  return "CmpMode::EQ";
}

// Binary col-expand ops (add/sub/mul/div/max/min): emit
// TCOLEXPAND<OP>(dst, src0, src1).

std::string roundModeTok(mlir::pto::RoundModeAttr attr) {
  using RM = mlir::pto::RoundMode;
  switch (attr.getValue()) {
  case RM::NONE:      return "RoundMode::CAST_NONE";
  case RM::RINT:      return "RoundMode::CAST_RINT";
  case RM::ROUND:     return "RoundMode::CAST_ROUND";
  case RM::FLOOR:     return "RoundMode::CAST_FLOOR";
  case RM::CEIL:      return "RoundMode::CAST_CEIL";
  case RM::TRUNC:     return "RoundMode::CAST_TRUNC";
  case RM::ODD:       return "RoundMode::CAST_ODD";
  case RM::CAST_RINT: return "RoundMode::CAST_RINT";
  }
  return "RoundMode::CAST_RINT";
}
std::string saturationModeTok(mlir::pto::SaturationModeAttr attr) {
  using SM = mlir::pto::SaturationMode;
  switch (attr.getValue()) {
  case SM::ON:  return "SaturationMode::ON";
  case SM::OFF: return "SaturationMode::OFF";
  }
  return "SaturationMode::OFF";
}

//===----------------------------------------------------------------------===//
// pto.tdiv lowering -> TDIV(dst, src0, src1)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// pto.tdivs lowering -> TDIVS(dst, src, scalar)
// Preserve source order from textual parse:
// ins(tile, scalar)   -> TDIVS(dst, tile, scalar)
// ins(scalar, tile)   -> TDIVS(dst, scalar, tile)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// pto.texp lowering -> TEXP(dst, src)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// pto.texpands lowering -> TEXPANDS(dst, scalar)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// pto.textract lowering -> TEXTRACT(dst, src, indexRow, indexCol)
//===----------------------------------------------------------------------===//

StringRef getReluPreModeToken(pto::ReluPreMode mode) {
  switch (mode) {
  case pto::ReluPreMode::NoRelu:
    return "ReluPreMode::NoRelu";
  case pto::ReluPreMode::NormalRelu:
    return "ReluPreMode::NormalRelu";
  case pto::ReluPreMode::ScalarRelu:
    return "ReluPreMode::ScalarRelu";
  case pto::ReluPreMode::VectorRelu:
    return "ReluPreMode::VectorRelu";
  case pto::ReluPreMode::Pwl:
    return "ReluPreMode::Pwl";
  }
  llvm_unreachable("unknown ReluPreMode");
}

StringRef getAccToVecModeToken(pto::AccToVecMode mode) {
  switch (mode) {
  case pto::AccToVecMode::SingleModeVec0:
    return "pto::AccToVecMode::SingleModeVec0";
  case pto::AccToVecMode::SingleModeVec1:
    return "pto::AccToVecMode::SingleModeVec1";
  case pto::AccToVecMode::DualModeSplitM:
    return "pto::AccToVecMode::DualModeSplitM";
  case pto::AccToVecMode::DualModeSplitN:
    return "pto::AccToVecMode::DualModeSplitN";
  }
  llvm_unreachable("unknown AccToVecMode");
}

StringRef getTInsertModeToken(pto::TInsertMode mode) {
  switch (mode) {
  case pto::TInsertMode::SPLIT2:
    return "pto::TInsertMode::SPLIT2";
  case pto::TInsertMode::SPLIT4:
    return "pto::TInsertMode::SPLIT4";
  }
  llvm_unreachable("unknown TInsertMode");
}

// Append the acc-to-vec mode (when present) and relu-pre-mode template
// tokens shared by the TMOV overload family.
void pushModeAndReluTemplateArgs(SmallVectorImpl<Attribute> &args,
                                        MLIRContext *ctx,
                                        pto::AccToVecModeAttr modeAttr,
                                        bool reluNonDefault,
                                        pto::ReluPreMode reluPreMode) {
  if (modeAttr)
    args.push_back(emitc::OpaqueAttr::get(
        ctx, getAccToVecModeToken(modeAttr.getValue())));
  if (modeAttr || reluNonDefault)
    args.push_back(
        emitc::OpaqueAttr::get(ctx, getReluPreModeToken(reluPreMode)));
}

StringRef getTFillPadModeToken(pto::TFillPadLoweringKind loweringKind) {
  switch (loweringKind) {
  case pto::TFillPadLoweringKind::Normal:
    return "pto::TFillPadMode::Normal";
  case pto::TFillPadLoweringKind::InPlace:
    return "pto::TFillPadMode::InPlace";
  case pto::TFillPadLoweringKind::Expand:
    return "pto::TFillPadMode::Expand";
  }
  llvm_unreachable("unknown TFillPadLoweringKind");
}

//===----------------------------------------------------------------------===//
// pto.tgather lowering
// - Index form  : TGATHER(dst, src0, indices, tmp)
// - Compare form: TGATHER<DstT, SrcT, CDstT, TmpT, CmpMode::GT, 7>(dst, src0, kValue, cdst, tmp)
// - Mask form : TGATHER<dstTileTok, srcTileTok, pto::MaskPattern::Pxxxx>(dst, src0)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TLOG lowering to EmitC (PTOConvert.cpp)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TLRELU lowering to EmitC (PTOConvert.cpp)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TMAX lowering to EmitC (PTOConvert.cpp)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TMAXS lowering to EmitC (PTOConvert.cpp)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TMIN lowering to EmitC (PTOConvert.cpp)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TMINS lowering to EmitC (PTOConvert.cpp)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TMINS lowering to EmitC (fix APFloat -> FloatAttr)  (PTOConvert.cpp)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering for TMOV op -> EmitC)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TMOV_FP DPS/memref op)
//===----------------------------------------------------------------------===//

// Materialize an lvalue for `offset` (reusing an existing emitc.variable when
// possible) and return its address; TQuant/TQuantMx pass &offset to the
// intrinsic.
Value materializeOffsetAddress(ConversionPatternRewriter &rewriter,
                                      Location loc, MLIRContext *ctx,
                                      Value offset) {
  Type offsetValueTy = offset.getType();
  Value offsetLValue = getSourceEmitCVariable(offset);
  if (!offsetLValue) {
    offsetLValue =
        rewriter
            .create<emitc::VariableOp>(
                loc, getEmitCVariableResultType(offsetValueTy),
                emitc::OpaqueAttr::get(ctx, ""))
            .getResult();
    rewriter.create<emitc::AssignOp>(loc, offsetLValue, offset);
  }
  return rewriter
      .create<emitc::ApplyOp>(
          loc, emitc::PointerType::get(offsetValueTy), "&", offsetLValue)
      .getResult();
}

// TQUANT template arguments: QuantType, dst/src (and fp, tmp) opaque type
// spellings; empty when any operand is not opaque.
FailureOr<ArrayAttr>
buildTQuantTemplateArgs(pto::TQuantOp op,
                        ConversionPatternRewriter &rewriter,
                        MLIRContext *ctx, Value dst, Value src, Value fp,
                        Value tmp) {
  auto dstOT = mlir::dyn_cast<emitc::OpaqueType>(dst.getType());
  auto srcOT = mlir::dyn_cast<emitc::OpaqueType>(src.getType());
  auto fpOT = mlir::dyn_cast<emitc::OpaqueType>(fp.getType());
  if (!(dstOT && srcOT && fpOT))
    return ArrayAttr{};

  auto quantTypeTok = [&]() -> StringRef {
    switch (op.getQuantType()) {
    case pto::QuantType::INT8_SYM:
      return "pto::QuantType::INT8_SYM";
    case pto::QuantType::INT8_ASYM:
      return "pto::QuantType::INT8_ASYM";
    case pto::QuantType::MXFP8:
    case pto::QuantType::MXFP4_E2M1:
      break;
    }
    llvm_unreachable("unknown QuantType");
  };

  SmallVector<Attribute, 5> args{
      emitc::OpaqueAttr::get(ctx, quantTypeTok()),
      emitc::OpaqueAttr::get(ctx, dstOT.getValue().str()),
      emitc::OpaqueAttr::get(ctx, srcOT.getValue().str()),
      emitc::OpaqueAttr::get(ctx, fpOT.getValue().str()),
  };
  if (tmp) {
    auto tmpOT = mlir::dyn_cast<emitc::OpaqueType>(tmp.getType());
    if (!tmpOT)
      return rewriter.notifyMatchFailure(
          op, "tquant tmp lowering expects opaque tmp type");
    args.push_back(emitc::OpaqueAttr::get(ctx, tmpOT.getValue().str()));
  }
  return rewriter.getArrayAttr(args);
}

// Take the address of an EmitC value: reuse an existing emitc.variable
// lvalue when possible, otherwise materialize a temporary.
Value addressOfEmitCValue(ConversionPatternRewriter &rewriter,
                                 Location loc, MLIRContext *ctx, Value v,
                                 emitc::OpaqueType ot) {
  if (Value variable = getSourceEmitCVariable(v))
    return rewriter
        .create<emitc::ApplyOp>(loc, emitc::PointerType::get(v.getType()), "&",
                                variable)
        .getResult();

  Value tmp = rewriter
                  .create<emitc::VariableOp>(
                      loc, getEmitCVariableResultType(ot),
                      emitc::OpaqueAttr::get(ctx, ""))
                  .getResult();
  rewriter.create<emitc::AssignOp>(loc, tmp, v);
  return rewriter
      .create<emitc::ApplyOp>(loc, emitc::PointerType::get(ot), "&", tmp)
      .getResult();
}

// Modern (non exp_zz) TQUANT-MX template tokens: group axis, the
// OCP/NV algorithm variant, and the optional interleave flag.
void appendModernMxTemplateArgs(pto::TQuantMxOp op,
                                       MLIRContext *ctx,
                                       SmallVectorImpl<Attribute> &out) {
  const StringRef axisTok =
      op.getGrpAxis() == pto::MxGroupAxis::Axis0 ? "0" : "1";
  StringRef algTok;
  if (op.getQuantType() == pto::QuantType::MXFP8)
    algTok = op.getQuantScaleAlg() == pto::QuantScaleAlg::NV
                 ? "pto::MxQuantAlg::NvMxFp8E4M3"
                 : "pto::MxQuantAlg::OcpMxFp8E4M3";
  else
    algTok = op.getQuantScaleAlg() == pto::QuantScaleAlg::NV
                 ? "pto::MxQuantAlg::NvMxFp4E2M1"
                 : "pto::MxQuantAlg::OcpMxFp4E2M1";
  out.push_back(emitc::OpaqueAttr::get(ctx, axisTok));
  out.push_back(emitc::OpaqueAttr::get(ctx, algTok));
  if (op.getInterleave())
    out.push_back(emitc::OpaqueAttr::get(ctx, "true"));
}

StringRef quantScaleAlgTok(pto::QuantScaleAlg alg) {
  switch (alg) {
  case pto::QuantScaleAlg::OCP:
    return "pto::QuantScaleAlg::OCP";
  case pto::QuantScaleAlg::NV:
    return "pto::QuantScaleAlg::NV";
  }
  llvm_unreachable("unknown QuantScaleAlg");
}

StringRef vecStoreModeTok(pto::VecStoreMode mode) {
  switch (mode) {
  case pto::VecStoreMode::ND:
    return "pto::VecStoreMode::ND";
  case pto::VecStoreMode::NZ:
    return "pto::VecStoreMode::NZ";
  }
  llvm_unreachable("unknown VecStoreMode");
}

// Deprecated fused TQUANT-MX form: retain the existing PTO-ISA overload and
// complete tile-type template list for wire/API compatibility.
void appendLegacyExpZzMxTemplateArgs(
    pto::TQuantMxOp op, MLIRContext *ctx, StringRef quantTypeStr,
    emitc::OpaqueType dstOT, emitc::OpaqueType srcOT,
    emitc::OpaqueType expOT, emitc::OpaqueType maxOT,
    emitc::OpaqueType scalingOT, SmallVectorImpl<Attribute> &out) {
  out.push_back(emitc::OpaqueAttr::get(ctx, quantTypeStr));
  if (auto storeMode = op.getStoreMode())
    out.push_back(emitc::OpaqueAttr::get(ctx, vecStoreModeTok(*storeMode)));
  out.push_back(emitc::OpaqueAttr::get(ctx, dstOT.getValue().str()));
  out.push_back(emitc::OpaqueAttr::get(ctx, srcOT.getValue().str()));
  out.push_back(emitc::OpaqueAttr::get(ctx, expOT.getValue().str()));
  out.push_back(emitc::OpaqueAttr::get(ctx, maxOT.getValue().str()));
  out.push_back(emitc::OpaqueAttr::get(ctx, scalingOT.getValue().str()));
  if (!op.getStoreMode() && op.getQuantScaleAlg() != pto::QuantScaleAlg::OCP)
    out.push_back(emitc::OpaqueAttr::get(ctx, quantScaleAlgTok(op.getQuantScaleAlg())));
}

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TMRGSORT DPS/memref op)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TMUL DPS/memref op)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TMULS DPS/memref op)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TNEG DPS/memref op)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TNOT DPS/memref op)
//===----------------------------------------------------------------------===//



} // namespace pto
} // namespace mlir

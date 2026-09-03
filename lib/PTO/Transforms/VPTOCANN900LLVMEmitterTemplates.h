// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#pragma once

#include "VPTOCANN900LLVMEmitterInternal.h"

namespace mlir::pto::detail {

template <typename UnaryOp> StringRef getUnaryMaskedStem() {
  if constexpr (std::is_same_v<UnaryOp, pto::VabsOp>) {
    return "vabs";
  }
  if constexpr (std::is_same_v<UnaryOp, pto::VexpOp>) {
    return "vexp";
  }
  if constexpr (std::is_same_v<UnaryOp, pto::VlnOp>) {
    return "vln";
  }
  if constexpr (std::is_same_v<UnaryOp, pto::VnegOp>) {
    return "vneg";
  }
  if constexpr (std::is_same_v<UnaryOp, pto::VsqrtOp>) {
    return "vsqrt";
  }
  if constexpr (std::is_same_v<UnaryOp, pto::VreluOp>) {
    return "vrelu";
  }
  if constexpr (std::is_same_v<UnaryOp, pto::VnotOp>) {
    return "vnot";
  }
  return {};
}

template <typename UnaryOp> FailureOr<StringRef> buildUnaryMaskedCallee(MLIRContext *context, Type resultType) {
  StringRef stem = getUnaryMaskedStem<UnaryOp>();
  if (stem.empty()) {
    return failure();
  }
  return buildCANN900ModeTypedCallee(context, resultType, stem, "x");
}

template <typename BinaryOp> StringRef getBinaryMaskedStem() {
  if constexpr (std::is_same_v<BinaryOp, pto::VaddOp>) {
    return "vadd";
  }
  if constexpr (std::is_same_v<BinaryOp, pto::VsubOp>) {
    return "vsub";
  }
  if constexpr (std::is_same_v<BinaryOp, pto::VmulOp>) {
    return "vmul";
  }
  if constexpr (std::is_same_v<BinaryOp, pto::VdivOp>) {
    return "vdiv";
  }
  if constexpr (std::is_same_v<BinaryOp, pto::VmaxOp>) {
    return "vmax";
  }
  if constexpr (std::is_same_v<BinaryOp, pto::VminOp>) {
    return "vmin";
  }
  if constexpr (std::is_same_v<BinaryOp, pto::VandOp>) {
    return "vand";
  }
  if constexpr (std::is_same_v<BinaryOp, pto::VorOp>) {
    return "vor";
  }
  if constexpr (std::is_same_v<BinaryOp, pto::VxorOp>) {
    return "vxor";
  }
  if constexpr (std::is_same_v<BinaryOp, pto::VshlOp>) {
    return "vshl";
  }
  if constexpr (std::is_same_v<BinaryOp, pto::VshrOp>) {
    return "vshr";
  }
  if constexpr (std::is_same_v<BinaryOp, pto::VpreluOp>) {
    return "vprelu";
  }
  return {};
}

template <typename TernaryOp> StringRef getTernaryMaskedStem() {
  if constexpr (std::is_same_v<TernaryOp, pto::VmaddOp>) {
    return "vmadd";
  }
  return {};
}

template <typename BinaryOp> constexpr bool usesSignedBinaryCANN900Callee() {
  return !std::is_same_v<BinaryOp, pto::VandOp> && !std::is_same_v<BinaryOp, pto::VorOp> &&
         !std::is_same_v<BinaryOp, pto::VxorOp> && !std::is_same_v<BinaryOp, pto::VpreluOp>;
}

template <typename TernaryOp> constexpr bool usesSignedTernaryCANN900Callee() { return false; }

template <typename CarryOp> StringRef getCarryBinaryStem() {
  if constexpr (std::is_same_v<CarryOp, pto::VaddcOp>) {
    return "vaddc";
  }
  if constexpr (std::is_same_v<CarryOp, pto::VsubcOp>) {
    return "vsubc";
  }
  if constexpr (std::is_same_v<CarryOp, pto::VaddcsOp>) {
    return "vaddcs";
  }
  if constexpr (std::is_same_v<CarryOp, pto::VsubcsOp>) {
    return "vsubcs";
  }
  return {};
}

template <typename CarryOp> constexpr bool hasCarryInput() {
  return std::is_same_v<CarryOp, pto::VaddcsOp> || std::is_same_v<CarryOp, pto::VsubcsOp>;
}

template <typename QueryOp> StringRef buildRuntimeQueryCallee(MLIRContext *context);

template <> inline StringRef buildRuntimeQueryCallee<pto::GetCtrlOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.GET.CTRL").getValue();
}

template <> inline StringRef buildRuntimeQueryCallee<pto::GetVms4SrOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.GET.VMS4.SR").getValue();
}

template <> inline StringRef buildRuntimeQueryCallee<pto::GetTidXOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.TID.X").getValue();
}

template <> inline StringRef buildRuntimeQueryCallee<pto::GetTidYOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.TID.Y").getValue();
}

template <> inline StringRef buildRuntimeQueryCallee<pto::GetTidZOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.TID.Z").getValue();
}

template <> inline StringRef buildRuntimeQueryCallee<pto::GetBlockDimXOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.BLOCK.DIM.X").getValue();
}

template <> inline StringRef buildRuntimeQueryCallee<pto::GetBlockDimYOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.BLOCK.DIM.Y").getValue();
}

template <> inline StringRef buildRuntimeQueryCallee<pto::GetBlockDimZOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.BLOCK.DIM.Z").getValue();
}

template <> inline StringRef buildRuntimeQueryCallee<pto::GetGridDimXOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.GRID.DIM.X").getValue();
}

template <> inline StringRef buildRuntimeQueryCallee<pto::GetGridDimYOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.GRID.DIM.Y").getValue();
}

template <> inline StringRef buildRuntimeQueryCallee<pto::GetGridDimZOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.GRID.DIM.Z").getValue();
}

template <> inline StringRef buildRuntimeQueryCallee<pto::GetBlockIdxXOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.BLOCK.IDX.X").getValue();
}

template <> inline StringRef buildRuntimeQueryCallee<pto::GetBlockIdxYOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.BLOCK.IDX.Y").getValue();
}

template <> inline StringRef buildRuntimeQueryCallee<pto::GetBlockIdxZOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.BLOCK.IDX.Z").getValue();
}

template <> inline StringRef buildRuntimeQueryCallee<pto::GetVecCoreIdOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.tpe.get.VECCOREID").getValue();
}

template <> inline StringRef buildRuntimeQueryCallee<pto::GetLaneIdOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.laneID").getValue();
}

template <> inline StringRef buildRuntimeQueryCallee<pto::GetClock32Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.CLOCK32").getValue();
}

template <> inline StringRef buildRuntimeQueryCallee<pto::GetClock64Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.CLOCK64").getValue();
}

template <> inline StringRef buildRuntimeQueryCallee<pto::GetLaneMaskEqOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.LANEMASK.EQ").getValue();
}

template <> inline StringRef buildRuntimeQueryCallee<pto::GetLaneMaskLeOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.LANEMASK.LE").getValue();
}

template <> inline StringRef buildRuntimeQueryCallee<pto::GetLaneMaskLtOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.LANEMASK.LT").getValue();
}

template <> inline StringRef buildRuntimeQueryCallee<pto::GetLaneMaskGeOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.LANEMASK.GE").getValue();
}

template <> inline StringRef buildRuntimeQueryCallee<pto::GetLaneMaskGtOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.LANEMASK.GT").getValue();
}

template <typename SprStoreOp> StringRef buildSprStoreCallee(MLIRContext *context, bool post);

template <> inline StringRef buildSprStoreCallee<pto::SprstiOp>(MLIRContext *context, bool post) {
  return buildSprstiCallee(context, post);
}

template <> inline StringRef buildSprStoreCallee<pto::SprstsOp>(MLIRContext *context, bool post) {
  return buildSprstsCallee(context, post);
}

template <typename ConfigOp> StringRef buildUnaryConfigCallee(MLIRContext *context);

template <> inline StringRef buildUnaryConfigCallee<pto::SetCtrlOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.CTRL").getValue();
}

template <typename VoteOp> StringRef buildVoteCallee(MLIRContext *context);

template <> inline StringRef buildVoteCallee<pto::VoteAllOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.vote.all").getValue();
}

template <> inline StringRef buildVoteCallee<pto::VoteAnyOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.vote.any").getValue();
}

template <> inline StringRef buildVoteCallee<pto::VoteUniOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.vote.uni").getValue();
}

template <> inline StringRef buildVoteCallee<pto::VoteBallotOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.vote.ballot").getValue();
}

template <typename BinaryOp> StringRef buildBinaryI64PureCallee(MLIRContext *context);

template <> inline StringRef buildBinaryI64PureCallee<pto::Sbitset0Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SBITSET0").getValue();
}

template <> inline StringRef buildBinaryI64PureCallee<pto::Sbitset1Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SBITSET1").getValue();
}

template <typename ShuffleOp> FailureOr<StringRef> buildShuffleCallee(MLIRContext *context, Type valueType);

template <> inline FailureOr<StringRef> buildShuffleCallee<pto::ShuffleIdxOp>(MLIRContext *context, Type valueType) {
  std::string elem = getShuffleIntrinsicTypeFragment(valueType);
  if (elem.empty()) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.shfl.idx." + elem).getValue();
}

template <> inline FailureOr<StringRef> buildShuffleCallee<pto::ShuffleUpOp>(MLIRContext *context, Type valueType) {
  std::string elem = getShuffleIntrinsicTypeFragment(valueType);
  if (elem.empty()) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.shfl.up." + elem).getValue();
}

template <> inline FailureOr<StringRef> buildShuffleCallee<pto::ShuffleDownOp>(MLIRContext *context, Type valueType) {
  std::string elem = getShuffleIntrinsicTypeFragment(valueType);
  if (elem.empty()) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.shfl.down." + elem).getValue();
}

template <> inline FailureOr<StringRef> buildShuffleCallee<pto::ShuffleBflyOp>(MLIRContext *context, Type valueType) {
  std::string elem = getShuffleIntrinsicTypeFragment(valueType);
  if (elem.empty()) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.shfl.bfly." + elem).getValue();
}

template <typename ReduxOp>
FailureOr<StringRef> buildReduxCallee(MLIRContext *context, Type valueType, Attribute signednessAttr);

template <>
inline FailureOr<StringRef> buildReduxCallee<pto::ReduxAddOp>(MLIRContext *context, Type valueType,
                                                              Attribute signednessAttr) {
  std::string elem = getReduxIntrinsicTypeFragment(valueType, signednessAttr);
  if (elem.empty()) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.redux.add." + elem).getValue();
}

template <>
inline FailureOr<StringRef> buildReduxCallee<pto::ReduxMaxOp>(MLIRContext *context, Type valueType,
                                                              Attribute signednessAttr) {
  std::string elem = getReduxIntrinsicTypeFragment(valueType, signednessAttr);
  if (elem.empty()) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.redux.max." + elem).getValue();
}

template <>
inline FailureOr<StringRef> buildReduxCallee<pto::ReduxMinOp>(MLIRContext *context, Type valueType,
                                                              Attribute signednessAttr) {
  std::string elem = getReduxIntrinsicTypeFragment(valueType, signednessAttr);
  if (elem.empty()) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.redux.min." + elem).getValue();
}

template <typename ScalarOp> StringRef buildScalarIntrinsicCallee(MLIRContext *context);

template <> inline StringRef buildScalarIntrinsicCallee<pto::PrmtOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.prmt").getValue();
}

template <typename UnaryOp> FailureOr<StringRef> buildUnaryScalarMathCallee(MLIRContext *context, Type valueType);

template <> inline FailureOr<StringRef> buildUnaryScalarMathCallee<pto::AbsFOp>(MLIRContext *context, Type valueType) {
  std::string elem = getLLVMFloatBuiltinFragment(valueType);
  if (elem != "f16" && elem != "f32" && elem != "v2f16" && elem != "v2bf16") {
    return failure();
  }
  return StringAttr::get(context, "llvm.fabs." + elem).getValue();
}

template <> inline FailureOr<StringRef> buildUnaryScalarMathCallee<pto::ExpOp>(MLIRContext *context, Type valueType) {
  std::string elem = getLLVMFloatBuiltinFragment(valueType);
  if (elem != "f32" && elem != "f16" && elem != "v2f16") {
    return failure();
  }
  return StringAttr::get(context, "llvm.exp." + elem).getValue();
}

template <> inline FailureOr<StringRef> buildUnaryScalarMathCallee<pto::LogOp>(MLIRContext *context, Type valueType) {
  std::string elem = getLLVMFloatBuiltinFragment(valueType);
  if (elem != "f32" && elem != "f16" && elem != "v2f16") {
    return failure();
  }
  return StringAttr::get(context, "llvm.log." + elem).getValue();
}

template <> inline FailureOr<StringRef> buildUnaryScalarMathCallee<pto::CeilOp>(MLIRContext *context, Type valueType) {
  std::string elem = getScalarHIVMFloatShortFragment(valueType);
  if (elem.empty()) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.ceil." + elem).getValue();
}

template <> inline FailureOr<StringRef> buildUnaryScalarMathCallee<pto::FloorOp>(MLIRContext *context, Type valueType) {
  std::string elem = getScalarHIVMFloatShortFragment(valueType);
  if (elem.empty()) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.floor." + elem).getValue();
}

template <> inline FailureOr<StringRef> buildUnaryScalarMathCallee<pto::RintOp>(MLIRContext *context, Type valueType) {
  std::string elem = getScalarHIVMFloatShortFragment(valueType);
  if (elem.empty()) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.rint." + elem).getValue();
}

template <> inline FailureOr<StringRef> buildUnaryScalarMathCallee<pto::RoundOp>(MLIRContext *context, Type valueType) {
  std::string elem = getScalarHIVMFloatShortFragment(valueType);
  if (elem.empty()) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.round." + elem).getValue();
}

template <typename BinaryOp> FailureOr<StringRef> buildBinaryScalarMathCallee(MLIRContext *context, Type valueType);

template <> inline FailureOr<StringRef> buildBinaryScalarMathCallee<pto::FMinOp>(MLIRContext *context, Type valueType) {
  std::string elem = getLLVMFloatBuiltinFragment(valueType);
  if (elem != "f16" && elem != "f32" && elem != "bf16" && elem != "v2f16" && elem != "v2bf16") {
    return failure();
  }
  return StringAttr::get(context, "llvm.minnum." + elem).getValue();
}

template <> inline FailureOr<StringRef> buildBinaryScalarMathCallee<pto::FMaxOp>(MLIRContext *context, Type valueType) {
  std::string elem = getLLVMFloatBuiltinFragment(valueType);
  if (elem != "f16" && elem != "f32" && elem != "bf16" && elem != "v2f16" && elem != "v2bf16") {
    return failure();
  }
  return StringAttr::get(context, "llvm.maxnum." + elem).getValue();
}

template <> inline FailureOr<StringRef> buildBinaryScalarMathCallee<pto::PowOp>(MLIRContext *context, Type valueType) {
  std::string elem = getLLVMFloatBuiltinFragment(valueType);
  if (elem != "f32" && elem != "f16" && elem != "v2f16") {
    return failure();
  }
  return StringAttr::get(context, "llvm.pow." + elem).getValue();
}

template <typename VecScalarOp> StringRef getVecScalarMaskedStem() {
  if constexpr (std::is_same_v<VecScalarOp, pto::VmulsOp>) {
    return "vmuls";
  }
  if constexpr (std::is_same_v<VecScalarOp, pto::VaddsOp>) {
    return "vadds";
  }
  if constexpr (std::is_same_v<VecScalarOp, pto::VmaxsOp>) {
    return "vmaxs";
  }
  if constexpr (std::is_same_v<VecScalarOp, pto::VminsOp>) {
    return "vmins";
  }
  if constexpr (std::is_same_v<VecScalarOp, pto::VlreluOp>) {
    return "vlrelu";
  }
  if constexpr (std::is_same_v<VecScalarOp, pto::VshlsOp>) {
    return "vshls";
  }
  if constexpr (std::is_same_v<VecScalarOp, pto::VshrsOp>) {
    return "vshrs";
  }
  return {};
}

template <typename VecScalarOp> constexpr bool usesSignedVecScalarCANN900Callee() {
  return !std::is_same_v<VecScalarOp, pto::VlreluOp>;
}

template <typename ReductionOp> StringRef getReductionUnaryStem() {
  if constexpr (std::is_same_v<ReductionOp, pto::VcaddOp>) {
    return "vcadd";
  }
  if constexpr (std::is_same_v<ReductionOp, pto::VcmaxOp>) {
    return "vcmax";
  }
  if constexpr (std::is_same_v<ReductionOp, pto::VcminOp>) {
    return "vcmin";
  }
  if constexpr (std::is_same_v<ReductionOp, pto::VcgaddOp>) {
    return "vcgadd";
  }
  if constexpr (std::is_same_v<ReductionOp, pto::VcgmaxOp>) {
    return "vcgmax";
  }
  if constexpr (std::is_same_v<ReductionOp, pto::VcgminOp>) {
    return "vcgmin";
  }
  if constexpr (std::is_same_v<ReductionOp, pto::VcpaddOp>) {
    return "vcpadd";
  }
  return {};
}

template <typename HistOp> StringRef getHistogramCallee(MLIRContext *context) {
  if constexpr (std::is_same_v<HistOp, pto::Chistv2Op>) {
    return StringAttr::get(context, "llvm.hivm.chistv2.m").getValue();
  }
  if constexpr (std::is_same_v<HistOp, pto::Dhistv2Op>) {
    return StringAttr::get(context, "llvm.hivm.dhistv2.m").getValue();
  }
  return {};
}

template <typename ExtremaOp> StringRef getExtremaPredicateStem() {
  if constexpr (std::is_same_v<ExtremaOp, pto::VcbmaxOp>) {
    return "vcbmax";
  }
  if constexpr (std::is_same_v<ExtremaOp, pto::VcbminOp>) {
    return "vcbmin";
  }
  return {};
}

template <typename ExtremaOp> FailureOr<StringRef> buildExtremaPredicateCallee(MLIRContext *context, Type resultType) {
  return buildCANN900SignedModeTypedCallee(context, resultType, getExtremaPredicateStem<ExtremaOp>(), "x");
}

template <typename ReductionOp> constexpr bool usesSignedReductionCANN900Callee() {
  return !std::is_same_v<ReductionOp, pto::VcpaddOp>;
}

template <typename Op> StringRef buildPredicatePairReorderCallee(MLIRContext *context);

template <> inline StringRef buildPredicatePairReorderCallee<pto::PdintlvB8Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pdintlv.b8").getValue();
}

template <> inline StringRef buildPredicatePairReorderCallee<pto::PdintlvB16Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pdintlv.b16").getValue();
}

template <> inline StringRef buildPredicatePairReorderCallee<pto::PdintlvB32Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pdintlv.b32").getValue();
}

template <> inline StringRef buildPredicatePairReorderCallee<pto::PintlvB8Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pintlv.b8").getValue();
}

template <> inline StringRef buildPredicatePairReorderCallee<pto::PintlvB16Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pintlv.b16").getValue();
}

template <> inline StringRef buildPredicatePairReorderCallee<pto::PintlvB32Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pintlv.b32").getValue();
}

template <typename StoreOp> StringRef getPredicateStoreCallee(MLIRContext *context, bool post);

template <> inline StringRef getPredicateStoreCallee<pto::PstiOp>(MLIRContext *context, bool post) {
  return buildPstiCallee(context, post);
}

template <> inline StringRef getPredicateStoreCallee<pto::PstsOp>(MLIRContext *context, bool post) {
  return buildPstsCallee(context, post);
}

template <typename LoadOp> StringRef getPredicateLoadCallee(MLIRContext *context, bool post);

template <> inline StringRef getPredicateLoadCallee<pto::PldiOp>(MLIRContext *context, bool post) {
  return buildPldiCallee(context, post);
}

template <> inline StringRef getPredicateLoadCallee<pto::PldsOp>(MLIRContext *context, bool post) {
  return buildPldsCallee(context, post);
}

template <typename PredicateMaskOp> StringRef getPredicateMaskCallee(MLIRContext *context);

template <> inline StringRef getPredicateMaskCallee<pto::PnotOp>(MLIRContext *context) {
  return buildPnotCallee(context);
}

template <> inline StringRef getPredicateMaskCallee<pto::PselOp>(MLIRContext *context) {
  return buildPselCallee(context);
}

template <> inline StringRef getPredicateMaskCallee<pto::PandOp>(MLIRContext *context) {
  return buildPandCallee(context);
}

template <> inline StringRef getPredicateMaskCallee<pto::PorOp>(MLIRContext *context) {
  return buildPorCallee(context);
}

template <> inline StringRef getPredicateMaskCallee<pto::PxorOp>(MLIRContext *context) {
  return buildPxorCallee(context);
}

template <typename PackOp> StringRef getPredicatePackCallee(MLIRContext *context);

template <> inline StringRef getPredicatePackCallee<pto::PpackOp>(MLIRContext *context) {
  return buildPpackCallee(context);
}

template <> inline StringRef getPredicatePackCallee<pto::PunpackOp>(MLIRContext *context) {
  return buildPunpackCallee(context);
}

template <typename PltOp> StringRef buildPltCallee(MLIRContext *context);

template <> inline StringRef buildPltCallee<pto::PltB8Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.plt.b8.v300").getValue();
}

template <> inline StringRef buildPltCallee<pto::PltB16Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.plt.b16.v300").getValue();
}

template <> inline StringRef buildPltCallee<pto::PltB32Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.plt.b32.v300").getValue();
}

template <typename PltmOp> StringRef buildPltmCallee(MLIRContext *context);

template <> inline StringRef buildPltmCallee<pto::PltmB8Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pltm.b8.v300").getValue();
}

template <> inline StringRef buildPltmCallee<pto::PltmB16Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pltm.b16.v300").getValue();
}

template <> inline StringRef buildPltmCallee<pto::PltmB32Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pltm.b32.v300").getValue();
}

template <typename PsetOp> StringRef buildPsetCallee(MLIRContext *context);

template <> inline StringRef buildPsetCallee<pto::PsetB8Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pset.b8").getValue();
}

template <> inline StringRef buildPsetCallee<pto::PsetB16Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pset.b16").getValue();
}

template <> inline StringRef buildPsetCallee<pto::PsetB32Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pset.b32").getValue();
}

template <typename PgeOp> StringRef buildPgeCallee(MLIRContext *context);

template <> inline StringRef buildPgeCallee<pto::PgeB8Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pge.b8").getValue();
}

template <> inline StringRef buildPgeCallee<pto::PgeB16Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pge.b16").getValue();
}

template <> inline StringRef buildPgeCallee<pto::PgeB32Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pge.b32").getValue();
}

template <typename LoopOp> StringRef buildSetLoopCallee(MLIRContext *context);

template <typename ConfigOp> StringRef buildUnaryConfigCallee(MLIRContext *context);

template <typename ConfigOp> StringRef buildNullaryConfigCallee(MLIRContext *context);

template <> inline StringRef buildSetLoopCallee<pto::SetLoop2StrideOutToUbOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP2.STRIDE.OUTTOUB").getValue();
}

template <> inline StringRef buildSetLoopCallee<pto::SetLoop1StrideOutToUbOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP1.STRIDE.OUTTOUB").getValue();
}

template <> inline StringRef buildSetLoopCallee<pto::SetLoopSizeOutToUbOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP.SIZE.OUTTOUB").getValue();
}

template <> inline StringRef buildSetLoopCallee<pto::SetLoop2StrideUbToOutOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP2.STRIDE.UBTOOUT").getValue();
}

template <> inline StringRef buildSetLoopCallee<pto::SetLoop1StrideUbToOutOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP1.STRIDE.UBTOOUT").getValue();
}

template <> inline StringRef buildSetLoopCallee<pto::SetLoopSizeUbToOutOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP.SIZE.UBTOOUT").getValue();
}

template <> inline StringRef buildSetLoopCallee<pto::SetLoop3ParaOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP3.PARA").getValue();
}

template <> inline StringRef buildSetLoopCallee<pto::SetChannelParaOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.CHANNEL.PARA").getValue();
}

template <> inline StringRef buildUnaryConfigCallee<pto::SetMovPadValOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.MOV.PAD.VAL").getValue();
}

template <> inline StringRef buildUnaryConfigCallee<pto::SetQuantPreOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.QUANT.PRE.v300").getValue();
}

template <> inline StringRef buildUnaryConfigCallee<pto::SetReluAlphaOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.RELU.ALPHA").getValue();
}

template <> inline StringRef buildUnaryConfigCallee<pto::SetFixClipReluOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.FIX.CLIP.RELU").getValue();
}

template <> inline StringRef buildUnaryConfigCallee<pto::SetLoop2StrideOutToL1Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP2.STRIDE.OUTTOL1").getValue();
}

template <> inline StringRef buildUnaryConfigCallee<pto::SetLoop1StrideOutToL1Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP1.STRIDE.OUTTOL1").getValue();
}

template <> inline StringRef buildUnaryConfigCallee<pto::SetLoopSizeOutToL1Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP.SIZE.OUTTOL1").getValue();
}

template <> inline StringRef buildUnaryConfigCallee<pto::SetMte2NzParaOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.MTE2.NZ.PARA").getValue();
}

template <> inline StringRef buildUnaryConfigCallee<pto::SetPadValOutToL1Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.PAD.VAL.OUTTOL1").getValue();
}

template <> inline StringRef buildUnaryConfigCallee<pto::SetFpcOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.FPC").getValue();
}

template <> inline StringRef buildUnaryConfigCallee<pto::SetStoreAtomicCfgOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.ST.ATOMIC.CFG").getValue();
}

template <> inline StringRef buildNullaryConfigCallee<pto::SetAtomicS32Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.ATOMIC.S32").getValue();
}

template <> inline StringRef buildNullaryConfigCallee<pto::SetAtomicS8Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.ATOMIC.S8").getValue();
}

template <typename SyncOp> StringRef buildSyncCallee(MLIRContext *context);

template <> inline StringRef buildSyncCallee<pto::SetFlagOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.FLAG.IMM").getValue();
}

template <> inline StringRef buildSyncCallee<pto::WaitFlagOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.WAIT.FLAG.IMM").getValue();
}

template <> inline StringRef buildSyncCallee<pto::SetFlagDynOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.FLAG.REG").getValue();
}

template <> inline StringRef buildSyncCallee<pto::WaitFlagDynOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.WAIT.FLAG.REG").getValue();
}

template <> inline StringRef buildSyncCallee<pto::BarrierOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.BARRIER").getValue();
}

template <> inline StringRef buildSyncCallee<pto::SyncSetOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.CROSS.CORE").getValue();
}

template <> inline StringRef buildSyncCallee<pto::SyncWaitOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.WAIT.FLAG.DEV.REG").getValue();
}

template <> inline StringRef buildSyncCallee<pto::SetIntraBlockOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.INTRA.BLOCK.mode").getValue();
}

template <> inline StringRef buildSyncCallee<pto::WaitIntraBlockOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.WAIT.INTRA.BLOCK.mode").getValue();
}

template <> inline StringRef buildSyncCallee<pto::GetBufOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.GET.BUFI.mode").getValue();
}

template <> inline StringRef buildSyncCallee<pto::RlsBufOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.RLS.BUFI.mode").getValue();
}

template <typename QueryOp> StringRef buildRuntimeQueryCallee(MLIRContext *context);

template <> inline StringRef buildRuntimeQueryCallee<pto::GetBlockIdxOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.GET.BLOCK.IDX").getValue();
}

template <> inline StringRef buildRuntimeQueryCallee<pto::GetSubBlockIdxOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.GET.SUBBLOCKID").getValue();
}

template <> inline StringRef buildRuntimeQueryCallee<pto::GetBlockNumOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.GET.BLOCK.NUM").getValue();
}

template <> inline StringRef buildRuntimeQueryCallee<pto::GetSubBlockNumOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.GET.SUBBLOCKDIM").getValue();
}

template <typename QueryOp> StringRef buildSimtBlockQueryCallee(MLIRContext *context);

template <> inline StringRef buildSimtBlockQueryCallee<pto::GetBlockIdxOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.tpe.get.BLOCK.IDX").getValue();
}

template <> inline StringRef buildSimtBlockQueryCallee<pto::GetBlockNumOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.tpe.get.BLOCK.NUM").getValue();
}

template <typename AtomicOp>
FailureOr<StringRef> buildAtomicCallee(MLIRContext *context, Type ptrType, Type valueType, Attribute signednessAttr);

#define PTO_DECLARE_ATOMIC_CALLEE(OP, NAME)                                                                            \
  template <>                                                                                                          \
  inline FailureOr<StringRef> buildAtomicCallee<pto::OP>(MLIRContext * context, Type ptrType, Type valueType,          \
                                                         Attribute signednessAttr) {                                   \
    return buildAtomicCalleeName(context, ptrType, valueType, signednessAttr, NAME);                                   \
  }

PTO_DECLARE_ATOMIC_CALLEE(AtomicCasOp, "CAS")
PTO_DECLARE_ATOMIC_CALLEE(AtomicExchOp, "EXCH")
PTO_DECLARE_ATOMIC_CALLEE(AtomicAddOp, "ADD")
PTO_DECLARE_ATOMIC_CALLEE(AtomicSubOp, "SUB")
PTO_DECLARE_ATOMIC_CALLEE(AtomicMinOp, "MIN")
PTO_DECLARE_ATOMIC_CALLEE(AtomicMaxOp, "MAX")
PTO_DECLARE_ATOMIC_CALLEE(AtomicAndOp, "AND")
PTO_DECLARE_ATOMIC_CALLEE(AtomicOrOp, "OR")
PTO_DECLARE_ATOMIC_CALLEE(AtomicXorOp, "XOR")

#undef PTO_DECLARE_ATOMIC_CALLEE

} // namespace mlir::pto::detail

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

inline FailureOr<StringRef> buildReduxCalleeImpl(MLIRContext *context, Type valueType,
                                                 Attribute signednessAttr, StringRef kind) {
  std::string elem = getReduxIntrinsicTypeFragment(valueType, signednessAttr);
  if (elem.empty()) {
    return failure();
  }
  std::string name = "llvm.hivm.redux." + kind.str() + "." + elem;
  return StringAttr::get(context, name).getValue();
}

template <>
inline FailureOr<StringRef> buildReduxCallee<pto::ReduxAddIOp>(MLIRContext *context, Type valueType,
                                                               Attribute signednessAttr) {
  return buildReduxCalleeImpl(context, valueType, signednessAttr, "add");
}

template <>
inline FailureOr<StringRef> buildReduxCallee<pto::ReduxAddFOp>(MLIRContext *context, Type valueType,
                                                               Attribute signednessAttr) {
  return buildReduxCalleeImpl(context, valueType, signednessAttr, "add");
}

template <>
inline FailureOr<StringRef> buildReduxCallee<pto::ReduxMaxIOp>(MLIRContext *context, Type valueType,
                                                               Attribute signednessAttr) {
  return buildReduxCalleeImpl(context, valueType, signednessAttr, "max");
}

template <>
inline FailureOr<StringRef> buildReduxCallee<pto::ReduxMaxFOp>(MLIRContext *context, Type valueType,
                                                               Attribute signednessAttr) {
  return buildReduxCalleeImpl(context, valueType, signednessAttr, "max");
}

template <>
inline FailureOr<StringRef> buildReduxCallee<pto::ReduxMinIOp>(MLIRContext *context, Type valueType,
                                                               Attribute signednessAttr) {
  return buildReduxCalleeImpl(context, valueType, signednessAttr, "min");
}

template <>
inline FailureOr<StringRef> buildReduxCallee<pto::ReduxMinFOp>(MLIRContext *context, Type valueType,
                                                               Attribute signednessAttr) {
  return buildReduxCalleeImpl(context, valueType, signednessAttr, "min");
}

template <typename AtomicOp>
FailureOr<StringRef> buildAtomicCallee(MLIRContext *context, Type ptrType, Type valueType, Attribute signednessAttr);

// Written out explicitly rather than via a function-like macro: the body
// contains a return statement, which G.PRE.05 forbids in macro definitions.
template <> inline FailureOr<StringRef> buildAtomicCallee<pto::AtomicCasOp>(MLIRContext *context, Type ptrType,
                                                                            Type valueType, Attribute signednessAttr) {
  return buildAtomicCalleeName(context, ptrType, valueType, signednessAttr, "CAS");
}

template <> inline FailureOr<StringRef> buildAtomicCallee<pto::AtomicExchOp>(MLIRContext *context, Type ptrType,
                                                                             Type valueType, Attribute signednessAttr) {
  return buildAtomicCalleeName(context, ptrType, valueType, signednessAttr, "EXCH");
}

template <> inline FailureOr<StringRef> buildAtomicCallee<pto::AtomicAddOp>(MLIRContext *context, Type ptrType,
                                                                            Type valueType, Attribute signednessAttr) {
  return buildAtomicCalleeName(context, ptrType, valueType, signednessAttr, "ADD");
}

template <> inline FailureOr<StringRef> buildAtomicCallee<pto::AtomicSubOp>(MLIRContext *context, Type ptrType,
                                                                            Type valueType, Attribute signednessAttr) {
  return buildAtomicCalleeName(context, ptrType, valueType, signednessAttr, "SUB");
}

template <> inline FailureOr<StringRef> buildAtomicCallee<pto::AtomicMinOp>(MLIRContext *context, Type ptrType,
                                                                            Type valueType, Attribute signednessAttr) {
  return buildAtomicCalleeName(context, ptrType, valueType, signednessAttr, "MIN");
}

template <> inline FailureOr<StringRef> buildAtomicCallee<pto::AtomicMaxOp>(MLIRContext *context, Type ptrType,
                                                                            Type valueType, Attribute signednessAttr) {
  return buildAtomicCalleeName(context, ptrType, valueType, signednessAttr, "MAX");
}

template <> inline FailureOr<StringRef> buildAtomicCallee<pto::AtomicAndOp>(MLIRContext *context, Type ptrType,
                                                                            Type valueType, Attribute signednessAttr) {
  return buildAtomicCalleeName(context, ptrType, valueType, signednessAttr, "AND");
}

template <> inline FailureOr<StringRef> buildAtomicCallee<pto::AtomicOrOp>(MLIRContext *context, Type ptrType,
                                                                           Type valueType, Attribute signednessAttr) {
  return buildAtomicCalleeName(context, ptrType, valueType, signednessAttr, "OR");
}

template <> inline FailureOr<StringRef> buildAtomicCallee<pto::AtomicXorOp>(MLIRContext *context, Type ptrType,
                                                                            Type valueType, Attribute signednessAttr) {
  return buildAtomicCalleeName(context, ptrType, valueType, signednessAttr, "XOR");
}

} // namespace mlir::pto::detail

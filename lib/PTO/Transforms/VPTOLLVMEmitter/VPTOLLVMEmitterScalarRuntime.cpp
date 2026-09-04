// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
 // Please refer to the License for details. This software is provided on an "AS IS" BASIS.

#include "VPTOLLVMEmitterInternal.h"
#include "PTO/Transforms/VPTOLLVMEmitterHelper.h"

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/VPTOMemoryDist.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::pto {
namespace {

static std::string getAtomicElementTypeFragment(Type type,
                                                Attribute signednessAttr) {
  if (auto vecType = dyn_cast<VectorType>(type)) {
    if (vecType.getRank() != 1 || vecType.getDimSize(0) != 2) {
      return {};
    }
    if (vecType.getElementType().isF16())
    {
      return "f16x2";
    }
    if (vecType.getElementType().isBF16())
    {
      return "bf16x2";
    }
    return {};
  }
  if (type.isF16())
  {
    return "fp16";
  }
  if (type.isBF16())
  {
    return "bf16";
  }
  if (type.isF32())
  {
    return "fp32";
  }
  auto intType = dyn_cast<IntegerType>(type);
  if (!intType) {
    return {};
  }
  if (intType.getWidth() != 32 && intType.getWidth() != 64) {
    return {};
  }
  if (signednessAttr) {
    auto signedness = cast<pto::SignednessAttr>(signednessAttr).getValue();
    return std::string(signedness == pto::Signedness::Unsigned ? "u" : "s") +
           std::to_string(intType.getWidth());
  }
  return std::string(intType.isUnsigned() ? "u" : "s") +
         std::to_string(intType.getWidth());
}

static std::string getShuffleIntrinsicTypeFragment(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    switch (intType.getWidth()) {
    case 32:
      return "i32";
    case 64:
      return "i64";
    default:
      return {};
    }
  }
  if (type.isF16())
  {
    return "f16";
  }
  if (type.isF32())
  {
    return "f32";
  }
  if (auto vecType = dyn_cast<VectorType>(type)) {
    if (vecType.getRank() == 1 && vecType.getDimSize(0) == 2 &&
        vecType.getElementType().isF16()) {
      return "v2f16";
    }
  }
  return {};
}

static std::string getReduxIntrinsicTypeFragment(Type type,
                                                 Attribute signednessAttr) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    if (intType.getWidth() != 32) {
      return {};
    }
    bool isUnsigned = false;
    if (signednessAttr) {
      isUnsigned = cast<pto::SignednessAttr>(signednessAttr).getValue() ==
                   pto::Signedness::Unsigned;
    }
    return isUnsigned ? "u32" : "s32";
  }
  if (type.isF16())
  {
    return "f16";
  }
  if (type.isF32())
  {
    return "f32";
  }
  return {};
}



template <typename QueryOp>
static StringRef buildRuntimeQueryCallee(MLIRContext *context);

template <>
StringRef buildRuntimeQueryCallee<pto::GetCtrlOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.GET.CTRL").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetVms4SrOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.GET.VMS4.SR").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetTidXOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.TID.X").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetTidYOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.TID.Y").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetTidZOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.TID.Z").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetBlockDimXOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.BLOCK.DIM.X").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetBlockDimYOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.BLOCK.DIM.Y").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetBlockDimZOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.BLOCK.DIM.Z").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetGridDimXOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.GRID.DIM.X").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetGridDimYOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.GRID.DIM.Y").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetGridDimZOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.GRID.DIM.Z").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetBlockIdxXOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.BLOCK.IDX.X").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetBlockIdxYOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.BLOCK.IDX.Y").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetBlockIdxZOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.BLOCK.IDX.Z").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetVecCoreIdOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.tpe.get.VECCOREID").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetLaneIdOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.laneID").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetClock32Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.CLOCK32").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetClock64Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.CLOCK64").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetLaneMaskEqOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.LANEMASK.EQ").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetLaneMaskLeOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.LANEMASK.LE").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetLaneMaskLtOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.LANEMASK.LT").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetLaneMaskGeOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.LANEMASK.GE").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetLaneMaskGtOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.LANEMASK.GT").getValue();
}



template <typename VoteOp>
static StringRef buildVoteCallee(MLIRContext *context);

template <>
StringRef buildVoteCallee<pto::VoteAllOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.vote.all").getValue();
}

template <>
StringRef buildVoteCallee<pto::VoteAnyOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.vote.any").getValue();
}

template <>
StringRef buildVoteCallee<pto::VoteUniOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.vote.uni").getValue();
}

template <>
StringRef buildVoteCallee<pto::VoteBallotOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.vote.ballot").getValue();
}

template <typename BinaryOp>
static StringRef buildBinaryI64PureCallee(MLIRContext *context);

template <>
StringRef buildBinaryI64PureCallee<pto::Sbitset0Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SBITSET0").getValue();
}

template <>
StringRef buildBinaryI64PureCallee<pto::Sbitset1Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SBITSET1").getValue();
}

template <typename ShuffleOp>
static FailureOr<StringRef> buildShuffleCallee(MLIRContext *context,
                                               Type valueType);

template <>
FailureOr<StringRef> buildShuffleCallee<pto::ShuffleIdxOp>(MLIRContext *context,
                                                           Type valueType) {
  std::string elem = getShuffleIntrinsicTypeFragment(valueType);
  if (elem.empty())
  {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.shfl.idx." + elem).getValue();
}

template <>
FailureOr<StringRef> buildShuffleCallee<pto::ShuffleUpOp>(MLIRContext *context,
                                                          Type valueType) {
  std::string elem = getShuffleIntrinsicTypeFragment(valueType);
  if (elem.empty())
  {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.shfl.up." + elem).getValue();
}

template <>
FailureOr<StringRef> buildShuffleCallee<pto::ShuffleDownOp>(MLIRContext *context,
                                                            Type valueType) {
  std::string elem = getShuffleIntrinsicTypeFragment(valueType);
  if (elem.empty())
  {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.shfl.down." + elem).getValue();
}

template <>
FailureOr<StringRef> buildShuffleCallee<pto::ShuffleBflyOp>(MLIRContext *context,
                                                            Type valueType) {
  std::string elem = getShuffleIntrinsicTypeFragment(valueType);
  if (elem.empty())
  {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.shfl.bfly." + elem).getValue();
}

static Value buildShuffleControlValue(OpBuilder &builder, Location loc,
                                      Value controlValue, int64_t widthValue,
                                      unsigned controlMask) {
  Value lowBits = builder.create<arith::AndIOp>(
      loc, controlValue, getI32Constant(builder, loc, 0x1f));
  Value encodedWidth =
      getI32Constant(builder, loc, static_cast<uint32_t>(32 - widthValue) << 16);
  Value encodedMask =
      getI32Constant(builder, loc, static_cast<uint32_t>(controlMask) << 8);
  Value highBits = builder.create<arith::OrIOp>(loc, encodedWidth, encodedMask);
  return builder.create<arith::OrIOp>(loc, highBits, lowBits);
}

template <typename ReduxOp>
static FailureOr<StringRef> buildReduxCallee(MLIRContext *context,
                                             Type valueType,
                                             Attribute signednessAttr);

template <>
FailureOr<StringRef> buildReduxCallee<pto::ReduxAddOp>(MLIRContext *context,
                                                      Type valueType,
                                                      Attribute signednessAttr) {
  std::string elem = getReduxIntrinsicTypeFragment(valueType, signednessAttr);
  if (elem.empty())
  {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.redux.add." + elem).getValue();
}

template <>
FailureOr<StringRef> buildReduxCallee<pto::ReduxMaxOp>(MLIRContext *context,
                                                      Type valueType,
                                                      Attribute signednessAttr) {
  std::string elem = getReduxIntrinsicTypeFragment(valueType, signednessAttr);
  if (elem.empty())
  {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.redux.max." + elem).getValue();
}

template <>
FailureOr<StringRef> buildReduxCallee<pto::ReduxMinOp>(MLIRContext *context,
                                                      Type valueType,
                                                      Attribute signednessAttr) {
  std::string elem = getReduxIntrinsicTypeFragment(valueType, signednessAttr);
  if (elem.empty())
  {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.redux.min." + elem).getValue();
}


template <typename AtomicOp>
static FailureOr<StringRef> buildAtomicCallee(MLIRContext *context,
                                              Type ptrType, Type valueType,
                                              Attribute signednessAttr);

static FailureOr<StringRef> buildAtomicCalleeName(MLIRContext *context,
                                                  Type ptrType, Type valueType,
                                                  Attribute signednessAttr,
                                                  StringRef opName) {
  std::string elem = getAtomicElementTypeFragment(valueType, signednessAttr);
  if (elem.empty())
  {
    return failure();
  }
  auto ptrTy = dyn_cast<pto::PtrType>(ptrType);
  if (!ptrTy)
  {
    return failure();
  }

  StringRef space;
  switch (ptrTy.getMemorySpace().getAddressSpace()) {
  case pto::AddressSpace::GM:
    space = "G";
    break;
  case pto::AddressSpace::VEC:
    if (valueType.isInteger(64))
    {
      return failure();
    }
    space = "S";
    break;
  default:
    return failure();
  }

  return StringAttr::get(context, "llvm.hivm.atom." + opName.str() + "." +
                                      space.str() + "." + elem)
      .getValue();
}

#define PTO_BUILD_ATOMIC_CALLEE(OP, NAME)                                      \
  template <>                                                                  \
  [[maybe_unused]] FailureOr<StringRef> buildAtomicCallee<pto::OP>(            \
      MLIRContext *context, Type ptrType, Type valueType,                      \
      Attribute signednessAttr) {                                              \
    return buildAtomicCalleeName(context, ptrType, valueType, signednessAttr,  \
                                 NAME);                                        \
  }

PTO_BUILD_ATOMIC_CALLEE(AtomicCasOp, "CAS")
PTO_BUILD_ATOMIC_CALLEE(AtomicExchOp, "EXCH")
PTO_BUILD_ATOMIC_CALLEE(AtomicAddOp, "ADD")
PTO_BUILD_ATOMIC_CALLEE(AtomicSubOp, "SUB")
PTO_BUILD_ATOMIC_CALLEE(AtomicMinOp, "MIN")
PTO_BUILD_ATOMIC_CALLEE(AtomicMaxOp, "MAX")
PTO_BUILD_ATOMIC_CALLEE(AtomicAndOp, "AND")
PTO_BUILD_ATOMIC_CALLEE(AtomicOrOp, "OR")
PTO_BUILD_ATOMIC_CALLEE(AtomicXorOp, "XOR")

#undef PTO_BUILD_ATOMIC_CALLEE



template <typename ScalarOp>
static StringRef buildScalarIntrinsicCallee(MLIRContext *context);

template <>
StringRef buildScalarIntrinsicCallee<pto::PrmtOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.prmt").getValue();
}

static FailureOr<StringRef>
buildMulhiCallee(MLIRContext *context, Type resultType,
                 pto::Signedness signedness) {
  if (resultType.isInteger(32)) {
    return StringAttr::get(
               context, signedness == pto::Signedness::Unsigned
                            ? "llvm.hivm.mulhi.ui"
                            : "llvm.hivm.mulhi.i")
        .getValue();
  }
  if (resultType.isInteger(64) && signedness == pto::Signedness::Unsigned)
  {
    return StringAttr::get(context, "llvm.hivm.mul64hi.ui").getValue();
  }
  return failure();
}

static FailureOr<StringRef>
buildMulI32ToI64Callee(MLIRContext *context, pto::Signedness signedness) {
  return StringAttr::get(
             context, signedness == pto::Signedness::Unsigned
                          ? "llvm.hivm.mul.i32toi64.ui"
                          : "llvm.hivm.mul.i32toi64.i")
      .getValue();
}

static std::string getScalarFloatBuiltinFragment(Type type) {
  if (type.isF32())
  {
    return "f32";
  }
  if (type.isF16())
  {
    return "f16";
  }
  if (type.isBF16())
  {
    return "bf16";
  }
  return {};
}

static std::string getV2FloatBuiltinFragment(Type type, StringRef suffix) {
  auto vecType = dyn_cast<VectorType>(type);
  if (!vecType || vecType.getRank() != 1 || vecType.getDimSize(0) != 2) {
    return {};
  }
  Type elementType = vecType.getElementType();
  if (elementType.isF16()) {
    return suffix == "x2" ? "f16x2" : "v2f16";
  }
  if (elementType.isBF16()) {
    return suffix == "x2" ? "bf16x2" : "v2bf16";
  }
  return {};
}

static std::string getLLVMFloatBuiltinFragment(Type type) {
  std::string scalar = getScalarFloatBuiltinFragment(type);
  if (!scalar.empty())
  {
    return scalar;
  }

  return getV2FloatBuiltinFragment(type, "v2");
}

static std::string getHIVMFloatBuiltinFragment(Type type) {
  std::string scalar = getScalarFloatBuiltinFragment(type);
  if (!scalar.empty())
  {
    return scalar;
  }

  return getV2FloatBuiltinFragment(type, "x2");
}

static FailureOr<StringRef> buildSqrtCallee(MLIRContext *context, Type valueType) {
  std::string elem = getLLVMFloatBuiltinFragment(valueType);
  if (elem != "f32" && elem != "f16" && elem != "v2f16")
  {
    return failure();
  }
  return StringAttr::get(context, "llvm.sqrt." + elem).getValue();
}

static std::string getScalarHIVMFloatShortFragment(Type type) {
  if (type.isF32())
  {
    return "f";
  }
  if (type.isF16())
  {
    return "h";
  }
  if (type.isBF16())
  {
    return "y";
  }
  return {};
}

template <typename UnaryOp>
static FailureOr<StringRef> buildUnaryScalarMathCallee(MLIRContext *context,
                                                       Type valueType);

template <>
FailureOr<StringRef> buildUnaryScalarMathCallee<pto::AbsFOp>(MLIRContext *context,
                                                             Type valueType) {
  std::string elem = getLLVMFloatBuiltinFragment(valueType);
  if (elem != "f16" && elem != "f32" && elem != "v2f16" && elem != "v2bf16") {
    return failure();
  }
  return StringAttr::get(context, "llvm.fabs." + elem).getValue();
}

template <>
FailureOr<StringRef> buildUnaryScalarMathCallee<pto::ExpOp>(MLIRContext *context,
                                                            Type valueType) {
  std::string elem = getLLVMFloatBuiltinFragment(valueType);
  if (elem != "f32" && elem != "f16" && elem != "v2f16")
  {
    return failure();
  }
  return StringAttr::get(context, "llvm.exp." + elem).getValue();
}

template <>
FailureOr<StringRef> buildUnaryScalarMathCallee<pto::LogOp>(MLIRContext *context,
                                                            Type valueType) {
  std::string elem = getLLVMFloatBuiltinFragment(valueType);
  if (elem != "f32" && elem != "f16" && elem != "v2f16")
  {
    return failure();
  }
  return StringAttr::get(context, "llvm.log." + elem).getValue();
}

template <>
FailureOr<StringRef> buildUnaryScalarMathCallee<pto::CeilOp>(MLIRContext *context,
                                                             Type valueType) {
  std::string elem = getScalarHIVMFloatShortFragment(valueType);
  if (elem.empty())
  {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.ceil." + elem).getValue();
}

template <>
FailureOr<StringRef> buildUnaryScalarMathCallee<pto::FloorOp>(MLIRContext *context,
                                                              Type valueType) {
  std::string elem = getScalarHIVMFloatShortFragment(valueType);
  if (elem.empty())
  {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.floor." + elem).getValue();
}

template <>
FailureOr<StringRef> buildUnaryScalarMathCallee<pto::RintOp>(MLIRContext *context,
                                                             Type valueType) {
  std::string elem = getScalarHIVMFloatShortFragment(valueType);
  if (elem.empty())
  {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.rint." + elem).getValue();
}

template <>
FailureOr<StringRef> buildUnaryScalarMathCallee<pto::RoundOp>(MLIRContext *context,
                                                              Type valueType) {
  std::string elem = getScalarHIVMFloatShortFragment(valueType);
  if (elem.empty())
  {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.round." + elem).getValue();
}

template <typename BinaryOp>
static FailureOr<StringRef> buildBinaryScalarMathCallee(MLIRContext *context,
                                                        Type valueType);

template <>
FailureOr<StringRef> buildBinaryScalarMathCallee<pto::FMinOp>(MLIRContext *context,
                                                              Type valueType) {
  std::string elem = getLLVMFloatBuiltinFragment(valueType);
  if (elem != "f16" && elem != "f32" && elem != "bf16" &&
      elem != "v2f16" && elem != "v2bf16") {
    return failure();
  }
  return StringAttr::get(context, "llvm.minnum." + elem).getValue();
}

template <>
FailureOr<StringRef> buildBinaryScalarMathCallee<pto::FMaxOp>(MLIRContext *context,
                                                              Type valueType) {
  std::string elem = getLLVMFloatBuiltinFragment(valueType);
  if (elem != "f16" && elem != "f32" && elem != "bf16" &&
      elem != "v2f16" && elem != "v2bf16") {
    return failure();
  }
  return StringAttr::get(context, "llvm.maxnum." + elem).getValue();
}

template <>
FailureOr<StringRef> buildBinaryScalarMathCallee<pto::PowOp>(MLIRContext *context,
                                                             Type valueType) {
  std::string elem = getLLVMFloatBuiltinFragment(valueType);
  if (elem != "f32" && elem != "f16" && elem != "v2f16")
  {
    return failure();
  }
  return StringAttr::get(context, "llvm.pow." + elem).getValue();
}

static FailureOr<StringRef> buildFmaCallee(MLIRContext *context, Type valueType) {
  std::string elem = getHIVMFloatBuiltinFragment(valueType);
  if (elem.empty())
  {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.ffma." + elem + ".rrr").getValue();
}

static std::string getConvertScalarFragment(Type type,
                                            Attribute signednessAttr) {
  if (auto vecType = dyn_cast<VectorType>(type)) {
    if (vecType.getRank() != 1 || vecType.getDimSize(0) != 2) {
      return {};
    }
    Type elementType = vecType.getElementType();
    if (std::string elem = getLowPrecisionElementFragment(elementType);
        !elem.empty() && !pto::isPTOFloat4PackedType(elementType)) {
      return elem + "x2";
    }
    if (elementType.isF32())
    {
      return "f32x2";
    }
    if (elementType.isF16())
    {
      return "f16x2";
    }
    if (elementType.isBF16())
    {
      return "bf16x2";
    }
    return {};
  }
  if (type.isF32())
  {
    return "fp32";
  }
  if (type.isF16())
  {
    return "fp16";
  }
  if (type.isBF16())
  {
    return "bf16";
  }
  if (std::string elem = getLowPrecisionElementFragment(type); !elem.empty())
  {
    return elem;
  }
  auto intType = dyn_cast<IntegerType>(type);
  if (!intType || (intType.getWidth() != 32 && intType.getWidth() != 64) ||
      !signednessAttr) {
    return {};
  }
  auto signedness = cast<pto::SignednessAttr>(signednessAttr).getValue();
  return std::string(signedness == pto::Signedness::Unsigned ? "u" : "s") +
         std::to_string(intType.getWidth());
}

static FailureOr<StringRef> buildConvertCallee(MLIRContext *context,
                                               Type srcType, Type dstType,
                                               Attribute signednessAttr) {
  std::string src = getConvertScalarFragment(srcType, signednessAttr);
  std::string dst = getConvertScalarFragment(dstType, signednessAttr);
  if (src.empty() || dst.empty())
  {
    return failure();
  }
  return StringAttr::get(context,
                         "llvm.hivm." + src + ".to." + dst)
      .getValue();
}


template <typename QueryOp>
static StringRef buildRuntimeQueryCallee(MLIRContext *context);

template <>
StringRef buildRuntimeQueryCallee<pto::GetBlockIdxOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.GET.BLOCK.IDX").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetSubBlockIdxOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.GET.SUBBLOCKID").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetBlockNumOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.GET.BLOCK.NUM").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetSubBlockNumOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.GET.SUBBLOCKDIM").getValue();
}

template <typename QueryOp>
static StringRef buildSimtBlockQueryCallee(MLIRContext *context);

template <>
StringRef
buildSimtBlockQueryCallee<pto::GetBlockIdxOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.tpe.get.BLOCK.IDX").getValue();
}

template <>
StringRef
buildSimtBlockQueryCallee<pto::GetBlockNumOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.tpe.get.BLOCK.NUM").getValue();
}



template <typename QueryOp>
class LowerRuntimeQueryOpPattern final : public OpConversionPattern<QueryOp> {
public:
  explicit LowerRuntimeQueryOpPattern(TypeConverter &typeConverter,
                                      MLIRContext *context,
                                      LoweringState &state)
      : OpConversionPattern<QueryOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(QueryOp op, typename QueryOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert runtime-query result type");
    }

    StringRef calleeName = buildRuntimeQueryCallee<QueryOp>(op.getContext());
    auto funcType = rewriter.getFunctionType(TypeRange{}, TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName,
                                              TypeRange{resultType}, ValueRange{});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename QueryOp>
class LowerBlockRuntimeQueryOpPattern final
    : public OpConversionPattern<QueryOp> {
public:
  explicit LowerBlockRuntimeQueryOpPattern(TypeConverter &typeConverter,
                                           MLIRContext *context,
                                           LoweringState &state)
      : OpConversionPattern<QueryOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(QueryOp op, typename QueryOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert block runtime-query result type");
    }

    auto funcOp = op->template getParentOfType<func::FuncOp>();
    bool isSimtEntry =
        funcOp && funcOp->hasAttr(pto::kPTOSimtEntryAttrName);
    if (isSimtEntry && !resultType.isInteger(64)) {
      return rewriter.notifyMatchFailure(
          op, "SIMT block runtime-query expects an i64 PTO result");
    }

    StringRef calleeName =
        isSimtEntry ? buildSimtBlockQueryCallee<QueryOp>(op.getContext())
                    : buildRuntimeQueryCallee<QueryOp>(op.getContext());
    Type callResultType = isSimtEntry ? rewriter.getI32Type() : resultType;
    auto funcType =
        rewriter.getFunctionType(TypeRange{}, TypeRange{callResultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), calleeName, TypeRange{callResultType}, ValueRange{});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});

    Value result = call.getResult(0);
    if (isSimtEntry)
    {
      result = rewriter.create<arith::ExtUIOp>(op.getLoc(), resultType, result);
    }
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  LoweringState &state;
};

template <typename VoteOp>
class LowerVoteOpPattern final : public OpConversionPattern<VoteOp> {
public:
  explicit LowerVoteOpPattern(TypeConverter &typeConverter,
                              MLIRContext *context, LoweringState &state)
      : OpConversionPattern<VoteOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(VoteOp op, typename VoteOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
    {
      return rewriter.notifyMatchFailure(op, "failed to convert vote result type");
    }

    Type predType = this->getTypeConverter()->convertType(op.getPred().getType());
    if (!predType || predType != rewriter.getI1Type())
    {
      return rewriter.notifyMatchFailure(op, "failed to convert vote predicate type");
    }

    StringRef calleeName = buildVoteCallee<VoteOp>(op.getContext());
    auto funcType = rewriter.getFunctionType(TypeRange{predType}, TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName,
                                              TypeRange{resultType},
                                              ValueRange{adaptor.getPred()});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename ShuffleOp>
class LowerShuffleOpPattern final : public OpConversionPattern<ShuffleOp> {
public:
  explicit LowerShuffleOpPattern(TypeConverter &typeConverter,
                                 MLIRContext *context, LoweringState &state)
      : OpConversionPattern<ShuffleOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(ShuffleOp op, typename ShuffleOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
    {
      return rewriter.notifyMatchFailure(op, "failed to convert shuffle result type");
    }

    Type valueType = this->getTypeConverter()->convertType(op.getValue().getType());
    if (!valueType || valueType != resultType)
    {
      return rewriter.notifyMatchFailure(op, "unexpected converted shuffle operand type");
    }

    FailureOr<StringRef> calleeName =
        buildShuffleCallee<ShuffleOp>(op.getContext(), op.getValue().getType());
    if (failed(calleeName))
    {
      return rewriter.notifyMatchFailure(op, "unsupported shuffle VPTO signature");
    }

    IntegerAttr widthAttr = op.getWidthAttr();
    Value controlValue;
    unsigned controlMask = 0;
    if constexpr (std::is_same_v<ShuffleOp, pto::ShuffleIdxOp>) {
      controlValue = adaptor.getIndex();
      controlMask = 0x1f;
    } else if constexpr (std::is_same_v<ShuffleOp, pto::ShuffleUpOp>) {
      controlValue = adaptor.getOffset();
      controlMask = 0;
    } else if constexpr (std::is_same_v<ShuffleOp, pto::ShuffleDownOp>) {
      controlValue = adaptor.getOffset();
      controlMask = 0x1f;
    } else if constexpr (std::is_same_v<ShuffleOp, pto::ShuffleBflyOp>) {
      controlValue = adaptor.getMask();
      controlMask = 0x1f;
    }
    if (!controlValue)
    {
      return rewriter.notifyMatchFailure(op, "missing shuffle control operand");
    }

    Value control = buildShuffleControlValue(
        rewriter, op.getLoc(), controlValue, widthAttr.getInt(), controlMask);

    Type i32Type = rewriter.getI32Type();
    auto funcType = rewriter.getFunctionType(TypeRange{resultType, i32Type},
                                             TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{adaptor.getValue(), control});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename ReduxOp>
class LowerReduxOpPattern final : public OpConversionPattern<ReduxOp> {
public:
  explicit LowerReduxOpPattern(TypeConverter &typeConverter,
                               MLIRContext *context, LoweringState &state)
      : OpConversionPattern<ReduxOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(ReduxOp op, typename ReduxOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
    {
      return rewriter.notifyMatchFailure(op, "failed to convert redux result type");
    }

    Type valueType = this->getTypeConverter()->convertType(op.getValue().getType());
    if (!valueType || valueType != resultType)
    {
      return rewriter.notifyMatchFailure(op, "unexpected converted redux operand type");
    }

    FailureOr<StringRef> calleeName = buildReduxCallee<ReduxOp>(
        op.getContext(), op.getValue().getType(), op.getSignednessAttr());
    if (failed(calleeName))
    {
      return rewriter.notifyMatchFailure(op, "unsupported redux VPTO signature");
    }

    auto funcType = rewriter.getFunctionType(TypeRange{resultType},
                                             TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName,
                                              TypeRange{resultType},
                                              ValueRange{adaptor.getValue()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename AtomicOp>
class LowerAtomicBinaryOpPattern final : public OpConversionPattern<AtomicOp> {
public:
  explicit LowerAtomicBinaryOpPattern(TypeConverter &typeConverter,
                                      MLIRContext *context,
                                      LoweringState &state)
      : OpConversionPattern<AtomicOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(AtomicOp op, typename AtomicOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op.getOld().getType());
    Type valueType = this->getTypeConverter()->convertType(op.getValue().getType());
    if (!resultType || !valueType || resultType != valueType) {
      return rewriter.notifyMatchFailure(op,
                                         "unexpected atomic operand/result type");
    }

    Type ptrType = this->getTypeConverter()->convertType(op.getPtr().getType());
    if (!ptrType)
    {
      return rewriter.notifyMatchFailure(op, "failed to convert atomic pointer type");
    }

    FailureOr<StringRef> calleeName = buildAtomicCallee<AtomicOp>(
        op.getContext(), op.getPtr().getType(), op.getValue().getType(),
        op.getSignednessAttr());
    if (failed(calleeName))
    {
      return rewriter.notifyMatchFailure(op, "unsupported atomic VPTO signature");
    }

    auto funcType = rewriter.getFunctionType(
        TypeRange{ptrType, valueType, rewriter.getI32Type()},
        TypeRange{resultType});
    Value modeValue = getI32Constant(
        rewriter, op.getLoc(),
        static_cast<uint64_t>(op.getL2cacheAttr()
                                  ? op.getL2cacheAttr().getValue()
                                  : pto::StL2Cache::NMFV));
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{adaptor.getPtr(), adaptor.getValue(), modeValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerAtomicCasOpPattern final
    : public OpConversionPattern<pto::AtomicCasOp> {
public:
  explicit LowerAtomicCasOpPattern(TypeConverter &typeConverter,
                                   MLIRContext *context,
                                   LoweringState &state)
      : OpConversionPattern<pto::AtomicCasOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::AtomicCasOp op, pto::AtomicCasOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op.getOld().getType());
    Type compareType =
        this->getTypeConverter()->convertType(op.getCompare().getType());
    Type valueType = this->getTypeConverter()->convertType(op.getValue().getType());
    if (!resultType || !compareType || !valueType || resultType != compareType ||
        resultType != valueType) {
      return rewriter.notifyMatchFailure(op, "unexpected atomic CAS type");
    }

    Type ptrType = this->getTypeConverter()->convertType(op.getPtr().getType());
    if (!ptrType)
    {
      return rewriter.notifyMatchFailure(op, "failed to convert atomic pointer type");
    }

    FailureOr<StringRef> calleeName = buildAtomicCallee<pto::AtomicCasOp>(
        op.getContext(), op.getPtr().getType(), op.getValue().getType(),
        op.getSignednessAttr());
    if (failed(calleeName))
    {
      return rewriter.notifyMatchFailure(op, "unsupported atomic CAS signature");
    }

    auto funcType = rewriter.getFunctionType(
        TypeRange{ptrType, compareType, valueType, rewriter.getI32Type()},
        TypeRange{resultType});
    Value modeValue = getI32Constant(
        rewriter, op.getLoc(),
        static_cast<uint64_t>(op.getL2cacheAttr()
                                  ? op.getL2cacheAttr().getValue()
                                  : pto::StL2Cache::NMFV));
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{adaptor.getPtr(), adaptor.getCompare(), adaptor.getValue(),
                   modeValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename ScalarOp>
class LowerScalarIntrinsicOpPattern final : public OpConversionPattern<ScalarOp> {
public:
  explicit LowerScalarIntrinsicOpPattern(TypeConverter &typeConverter,
                                         MLIRContext *context,
                                         LoweringState &state)
      : OpConversionPattern<ScalarOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(ScalarOp op, typename ScalarOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      resultTypes))) {
      return rewriter.notifyMatchFailure(op, "failed to convert scalar result types");
    }

    SmallVector<Type> operandTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getOperandTypes(),
                                                      operandTypes))) {
      return rewriter.notifyMatchFailure(op, "failed to convert scalar operand types");
    }

    StringRef calleeName = buildScalarIntrinsicCallee<ScalarOp>(op.getContext());
    auto funcType = rewriter.getFunctionType(operandTypes, resultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName,
                                              resultTypes, adaptor.getOperands());
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerMulhiOpPattern final : public OpConversionPattern<pto::MulhiOp> {
public:
  explicit LowerMulhiOpPattern(TypeConverter &typeConverter,
                               MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::MulhiOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::MulhiOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    Type lhsType = getTypeConverter()->convertType(op.getLhs().getType());
    Type rhsType = getTypeConverter()->convertType(op.getRhs().getType());
    if (!resultType || !lhsType || !rhsType || lhsType != resultType ||
        rhsType != resultType) {
      return rewriter.notifyMatchFailure(op, "unexpected mulhi type");
    }

    pto::Signedness signedness = op.getSignednessAttr().getValue();
    FailureOr<StringRef> calleeName =
        buildMulhiCallee(op.getContext(), op.getResult().getType(), signedness);
    if (succeeded(calleeName)) {
      auto funcType =
          rewriter.getFunctionType(TypeRange{lhsType, rhsType}, TypeRange{resultType});
      auto call = rewriter.create<func::CallOp>(
          op.getLoc(), *calleeName, TypeRange{resultType},
          ValueRange{adaptor.getLhs(), adaptor.getRhs()});
      state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
      rewriter.replaceOp(op, call.getResults());
      return success();
    }

    if (!op.getResult().getType().isInteger(64) ||
        signedness != pto::Signedness::Signed) {
      return rewriter.notifyMatchFailure(op, "unsupported mulhi signature");
    }

    FailureOr<StringRef> unsignedCalleeName =
        buildMulhiCallee(op.getContext(), op.getResult().getType(),
                         pto::Signedness::Unsigned);
    if (failed(unsignedCalleeName))
    {
      return rewriter.notifyMatchFailure(op, "unsupported mul64hi signature");
    }

    auto funcType =
        rewriter.getFunctionType(TypeRange{lhsType, rhsType}, TypeRange{resultType});
    auto unsignedCall = rewriter.create<func::CallOp>(
        op.getLoc(), *unsignedCalleeName, TypeRange{resultType},
        ValueRange{adaptor.getLhs(), adaptor.getRhs()});
    state.plannedDecls.push_back(PlannedDecl{unsignedCalleeName->str(), funcType});

    Value corrected = buildSignedMulhiCorrection(
        op.getLoc(), rewriter, unsignedCall.getResult(0), adaptor.getLhs(),
        adaptor.getRhs(), resultType);
    rewriter.replaceOp(op, corrected);
    return success();
  }

private:
  static Value buildSignedMulhiCorrection(Location loc,
                                          ConversionPatternRewriter &rewriter,
                                          Value unsignedResult, Value lhs,
                                          Value rhs, Type resultType) {
    Value zero = getI64Constant(rewriter, loc, 0);
    Value lhsNeg = rewriter.create<LLVM::ICmpOp>(
        loc, LLVM::ICmpPredicate::slt, lhs, zero);
    Value rhsNeg = rewriter.create<LLVM::ICmpOp>(
        loc, LLVM::ICmpPredicate::slt, rhs, zero);
    Value subRhs = rewriter.create<LLVM::SubOp>(
        loc, unsignedResult, rhs);
    Value correctedLhs = rewriter.create<LLVM::SelectOp>(
        loc, resultType, lhsNeg, subRhs, unsignedResult);
    Value subLhs = rewriter.create<LLVM::SubOp>(
        loc, correctedLhs, lhs);
    Value corrected = rewriter.create<LLVM::SelectOp>(
        loc, resultType, rhsNeg, subLhs, correctedLhs);
    return corrected;
  }

  LoweringState &state;
};

class LowerMulI32ToI64OpPattern final
    : public OpConversionPattern<pto::MulI32ToI64Op> {
public:
  explicit LowerMulI32ToI64OpPattern(TypeConverter &typeConverter,
                                     MLIRContext *context,
                                     LoweringState &state)
      : OpConversionPattern<pto::MulI32ToI64Op>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::MulI32ToI64Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    Type lhsType = getTypeConverter()->convertType(op.getLhs().getType());
    Type rhsType = getTypeConverter()->convertType(op.getRhs().getType());
    if (!resultType || !lhsType || !rhsType)
    {
      return rewriter.notifyMatchFailure(op, "unexpected mul_i32toi64 type");
    }

    FailureOr<StringRef> calleeName =
        buildMulI32ToI64Callee(op.getContext(),
                               op.getSignednessAttr().getValue());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op,
                                         "unsupported mul_i32toi64 signature");
    }

    auto funcType =
        rewriter.getFunctionType(TypeRange{lhsType, rhsType}, TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{adaptor.getLhs(), adaptor.getRhs()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerSqrtOpPattern final : public OpConversionPattern<pto::SqrtOp> {
public:
  explicit LowerSqrtOpPattern(TypeConverter &typeConverter,
                              MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::SqrtOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::SqrtOp op, pto::SqrtOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    Type valueType = this->getTypeConverter()->convertType(op.getValue().getType());
    if (!resultType || !valueType || valueType != resultType)
    {
      return rewriter.notifyMatchFailure(op, "unexpected sqrt operand/result type");
    }

    FailureOr<StringRef> calleeName =
        buildSqrtCallee(op.getContext(), op.getValue().getType());
    if (failed(calleeName))
    {
      return rewriter.notifyMatchFailure(op, "unsupported sqrt VPTO signature");
    }

    auto funcType = rewriter.getFunctionType(TypeRange{valueType},
                                             TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName,
                                              TypeRange{resultType},
                                              ValueRange{adaptor.getValue()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename UnaryOp>
class LowerUnaryScalarMathOpPattern final : public OpConversionPattern<UnaryOp> {
public:
  explicit LowerUnaryScalarMathOpPattern(TypeConverter &typeConverter,
                                         MLIRContext *context,
                                         LoweringState &state)
      : OpConversionPattern<UnaryOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(UnaryOp op, typename UnaryOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    Type valueType = this->getTypeConverter()->convertType(op.getValue().getType());
    if (!resultType || !valueType || valueType != resultType)
    {
      return rewriter.notifyMatchFailure(op, "unexpected unary scalar math type");
    }

    FailureOr<StringRef> calleeName =
        buildUnaryScalarMathCallee<UnaryOp>(op.getContext(), op.getValue().getType());
    if (failed(calleeName))
    {
      return rewriter.notifyMatchFailure(op, "unsupported unary scalar math signature");
    }

    auto funcType = rewriter.getFunctionType(TypeRange{valueType},
                                             TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName,
                                              TypeRange{resultType},
                                              ValueRange{adaptor.getValue()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename BinaryOp>
class LowerBinaryScalarMathOpPattern final : public OpConversionPattern<BinaryOp> {
public:
  explicit LowerBinaryScalarMathOpPattern(TypeConverter &typeConverter,
                                          MLIRContext *context,
                                          LoweringState &state)
      : OpConversionPattern<BinaryOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(BinaryOp op, typename BinaryOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    Type lhsType = this->getTypeConverter()->convertType(op.getLhs().getType());
    Type rhsType = this->getTypeConverter()->convertType(op.getRhs().getType());
    if (!resultType || !lhsType || !rhsType ||
        lhsType != rhsType || lhsType != resultType) {
      return rewriter.notifyMatchFailure(op, "unexpected binary scalar math type");
    }

    FailureOr<StringRef> calleeName =
        buildBinaryScalarMathCallee<BinaryOp>(op.getContext(), op.getLhs().getType());
    if (failed(calleeName))
    {
      return rewriter.notifyMatchFailure(op, "unsupported binary scalar math signature");
    }

    auto funcType = rewriter.getFunctionType(TypeRange{lhsType, rhsType},
                                             TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{adaptor.getLhs(), adaptor.getRhs()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerFmaOpPattern final : public OpConversionPattern<pto::FmaOp> {
public:
  explicit LowerFmaOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                             LoweringState &state)
      : OpConversionPattern<pto::FmaOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::FmaOp op, pto::FmaOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    Type lhsType = this->getTypeConverter()->convertType(op.getLhs().getType());
    Type rhsType = this->getTypeConverter()->convertType(op.getRhs().getType());
    Type accType = this->getTypeConverter()->convertType(op.getAcc().getType());
    if (!resultType || !lhsType || !rhsType || !accType ||
        lhsType != rhsType || lhsType != accType || lhsType != resultType) {
      return rewriter.notifyMatchFailure(op, "unexpected fma scalar math type");
    }

    FailureOr<StringRef> calleeName = buildFmaCallee(op.getContext(),
                                                     op.getLhs().getType());
    if (failed(calleeName))
    {
      return rewriter.notifyMatchFailure(op, "unsupported fma scalar signature");
    }

    auto funcType = rewriter.getFunctionType(TypeRange{lhsType, rhsType, accType},
                                             TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{adaptor.getLhs(), adaptor.getRhs(), adaptor.getAcc()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerConvertOpPattern final : public OpConversionPattern<pto::ConvertOp> {
public:
  explicit LowerConvertOpPattern(TypeConverter &typeConverter,
                                 MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::ConvertOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::ConvertOp op, pto::ConvertOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getDst().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert result type");
    }

    FailureOr<StringRef> calleeName =
        buildConvertCallee(op.getContext(), op.getSrc().getType(),
                           op.getDst().getType(), op.getSignednessAttr());
    if (failed(calleeName))
    {
      return rewriter.notifyMatchFailure(op, "unsupported convert signature");
    }

    Value rounding = getI32Constant(
        rewriter, op.getLoc(), static_cast<uint64_t>(op.getRounding()));
    Value saturation = getI32Constant(
        rewriter, op.getLoc(), static_cast<uint64_t>(op.getSaturation()));

    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getSrc().getType(), rewriter.getI32Type(),
                  rewriter.getI32Type()},
        TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{adaptor.getSrc(), rounding, saturation});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerGetVms4SrOpPattern final
    : public OpConversionPattern<pto::GetVms4SrOp> {
public:
  explicit LowerGetVms4SrOpPattern(TypeConverter &typeConverter,
                                   MLIRContext *context,
                                   LoweringState &state)
      : OpConversionPattern<pto::GetVms4SrOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::GetVms4SrOp op, pto::GetVms4SrOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      resultTypes)) ||
        resultTypes.size() != 4) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert get_vms4_sr result types");
    }

    StringRef calleeName = buildRuntimeQueryCallee<pto::GetVms4SrOp>(
        op.getContext());
    auto funcType =
        rewriter.getFunctionType(TypeRange{}, TypeRange{rewriter.getI64Type()});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), calleeName, TypeRange{rewriter.getI64Type()},
        ValueRange{});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});

    SmallVector<Value> counts;
    counts.reserve(4);
    Value raw = call.getResult(0);
    for (unsigned i = 0; i < 4; ++i) {
      Value shifted = raw;
      if (i != 0) {
        shifted = rewriter.create<arith::ShRUIOp>(
            op.getLoc(), raw, getI64Constant(rewriter, op.getLoc(), i * 16));
      }
      counts.push_back(rewriter.create<arith::TruncIOp>(
          op.getLoc(), resultTypes[i], shifted));
    }
    rewriter.replaceOp(op, counts);
    return success();
  }

private:
  LoweringState &state;
};

template <typename BinaryOp>
class LowerBinaryI64PureOpPattern final : public OpConversionPattern<BinaryOp> {
public:
  explicit LowerBinaryI64PureOpPattern(TypeConverter &typeConverter,
                                       MLIRContext *context,
                                       LoweringState &state)
      : OpConversionPattern<BinaryOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(BinaryOp op, typename BinaryOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
    {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }

    StringRef calleeName = buildBinaryI64PureCallee<BinaryOp>(op.getContext());
    auto funcType =
        rewriter.getFunctionType(TypeRange{adaptor.getFirst().getType(),
                                           adaptor.getSecond().getType()},
                                 TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), calleeName, TypeRange{resultType},
        ValueRange{adaptor.getFirst(), adaptor.getSecond()});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};



} // namespace

static void addRuntimeQueryPatterns(TypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    LoweringState &state) {
  patterns.add<LowerRuntimeQueryOpPattern<pto::GetCtrlOp>,
               LowerGetVms4SrOpPattern,
               LowerRuntimeQueryOpPattern<pto::GetTidXOp>,
               LowerRuntimeQueryOpPattern<pto::GetTidYOp>,
               LowerRuntimeQueryOpPattern<pto::GetTidZOp>,
               LowerRuntimeQueryOpPattern<pto::GetBlockDimXOp>,
               LowerRuntimeQueryOpPattern<pto::GetBlockDimYOp>,
               LowerRuntimeQueryOpPattern<pto::GetBlockDimZOp>,
               LowerRuntimeQueryOpPattern<pto::GetGridDimXOp>,
               LowerRuntimeQueryOpPattern<pto::GetGridDimYOp>,
               LowerRuntimeQueryOpPattern<pto::GetGridDimZOp>,
               LowerRuntimeQueryOpPattern<pto::GetBlockIdxXOp>,
               LowerRuntimeQueryOpPattern<pto::GetBlockIdxYOp>,
               LowerRuntimeQueryOpPattern<pto::GetBlockIdxZOp>,
               LowerRuntimeQueryOpPattern<pto::GetVecCoreIdOp>,
               LowerRuntimeQueryOpPattern<pto::GetLaneIdOp>,
               LowerRuntimeQueryOpPattern<pto::GetClock32Op>,
               LowerRuntimeQueryOpPattern<pto::GetClock64Op>,
               LowerRuntimeQueryOpPattern<pto::GetLaneMaskEqOp>,
               LowerRuntimeQueryOpPattern<pto::GetLaneMaskLeOp>,
               LowerRuntimeQueryOpPattern<pto::GetLaneMaskLtOp>,
               LowerRuntimeQueryOpPattern<pto::GetLaneMaskGeOp>,
               LowerRuntimeQueryOpPattern<pto::GetLaneMaskGtOp>,
               LowerBlockRuntimeQueryOpPattern<pto::GetBlockIdxOp>,
               LowerRuntimeQueryOpPattern<pto::GetSubBlockIdxOp>,
               LowerBlockRuntimeQueryOpPattern<pto::GetBlockNumOp>,
               LowerRuntimeQueryOpPattern<pto::GetSubBlockNumOp>>(
      typeConverter, patterns.getContext(), state);
}

static void addScalarCollectivePatterns(TypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        LoweringState &state) {
  patterns.add<LowerVoteOpPattern<pto::VoteAllOp>,
               LowerVoteOpPattern<pto::VoteAnyOp>,
               LowerVoteOpPattern<pto::VoteUniOp>,
               LowerVoteOpPattern<pto::VoteBallotOp>,
               LowerShuffleOpPattern<pto::ShuffleIdxOp>,
               LowerShuffleOpPattern<pto::ShuffleUpOp>,
               LowerShuffleOpPattern<pto::ShuffleDownOp>,
               LowerShuffleOpPattern<pto::ShuffleBflyOp>,
               LowerReduxOpPattern<pto::ReduxAddOp>,
               LowerReduxOpPattern<pto::ReduxMaxOp>,
               LowerReduxOpPattern<pto::ReduxMinOp>,
               LowerAtomicCasOpPattern,
               LowerAtomicBinaryOpPattern<pto::AtomicExchOp>,
               LowerAtomicBinaryOpPattern<pto::AtomicAddOp>,
               LowerAtomicBinaryOpPattern<pto::AtomicSubOp>,
               LowerAtomicBinaryOpPattern<pto::AtomicMinOp>,
               LowerAtomicBinaryOpPattern<pto::AtomicMaxOp>,
               LowerAtomicBinaryOpPattern<pto::AtomicAndOp>,
               LowerAtomicBinaryOpPattern<pto::AtomicOrOp>,
               LowerAtomicBinaryOpPattern<pto::AtomicXorOp>>(
      typeConverter, patterns.getContext(), state);
}

static void addScalarMathPatterns(TypeConverter &typeConverter,
                                  RewritePatternSet &patterns,
                                  LoweringState &state) {
  patterns.add<LowerScalarIntrinsicOpPattern<pto::PrmtOp>,
               LowerMulhiOpPattern, LowerMulI32ToI64OpPattern,
               LowerSqrtOpPattern,
               LowerUnaryScalarMathOpPattern<pto::AbsFOp>,
               LowerUnaryScalarMathOpPattern<pto::ExpOp>,
               LowerUnaryScalarMathOpPattern<pto::LogOp>,
               LowerUnaryScalarMathOpPattern<pto::CeilOp>,
               LowerUnaryScalarMathOpPattern<pto::FloorOp>,
               LowerUnaryScalarMathOpPattern<pto::RintOp>,
               LowerUnaryScalarMathOpPattern<pto::RoundOp>,
               LowerBinaryScalarMathOpPattern<pto::FMinOp>,
               LowerBinaryScalarMathOpPattern<pto::FMaxOp>,
               LowerBinaryScalarMathOpPattern<pto::PowOp>, LowerFmaOpPattern,
               LowerConvertOpPattern,
               LowerBinaryI64PureOpPattern<pto::Sbitset0Op>,
               LowerBinaryI64PureOpPattern<pto::Sbitset1Op>>(
      typeConverter, patterns.getContext(), state);
}

void populateVPTOScalarAndRuntimePatterns(TypeConverter &typeConverter,
                                          RewritePatternSet &patterns,
                                          LoweringState &state) {
  addRuntimeQueryPatterns(typeConverter, patterns, state);
  addScalarCollectivePatterns(typeConverter, patterns, state);
  addScalarMathPatterns(typeConverter, patterns, state);
}
} // namespace mlir::pto

// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTO.cpp - C API for PTO dialect -----------------------------------===//
//
// This file provides the C API for the PTO dialect and its custom types.
//
// It must be built into an MLIR CAPI library (e.g. MLIRCAPIPTO) and linked
// by any consumers (e.g. Python extension).
//
//===----------------------------------------------------------------------===//

#include "pto-c/Dialect/PTOCAPI.h"

// unwrap/wrap + MLIR dialect registration C-API support.
#include "mlir/CAPI/IR.h"

#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"
#include "llvm/ADT/SmallVector.h"

// IMPORTANT: include the C++ dialect header that declares PtrType/TensorViewType.
// This header should itself include the generated PTOTypeDefs.h.inc.
#include "PTO/IR/PTO.h"

// C mirror enums consumed by the online-buildable Python bindings. This is the
// only TU that sees both the generated C++ `enum class mlir::pto::*` and the C
// mirror, so the static_asserts below break the build on any value drift.
#include "pto-c/Dialect/PTOEnums.h"

namespace {
#define PTO_ENUM_MIRROR_EQ(CppEnum, CppVal, CMirror)                            \
  static_assert(static_cast<uint32_t>(mlir::pto::CppEnum::CppVal) ==            \
                    static_cast<uint32_t>(CMirror),                             \
                "PTOEnums.h mirror drifted from mlir::pto::" #CppEnum)

PTO_ENUM_MIRROR_EQ(AddressSpace, Zero, MlirPTOAddressSpace::Zero);
PTO_ENUM_MIRROR_EQ(AddressSpace, GM, MlirPTOAddressSpace::GM);
PTO_ENUM_MIRROR_EQ(AddressSpace, MAT, MlirPTOAddressSpace::MAT);
PTO_ENUM_MIRROR_EQ(AddressSpace, LEFT, MlirPTOAddressSpace::LEFT);
PTO_ENUM_MIRROR_EQ(AddressSpace, RIGHT, MlirPTOAddressSpace::RIGHT);
PTO_ENUM_MIRROR_EQ(AddressSpace, ACC, MlirPTOAddressSpace::ACC);
PTO_ENUM_MIRROR_EQ(AddressSpace, VEC, MlirPTOAddressSpace::VEC);
PTO_ENUM_MIRROR_EQ(AddressSpace, BIAS, MlirPTOAddressSpace::BIAS);
PTO_ENUM_MIRROR_EQ(AddressSpace, SCALING, MlirPTOAddressSpace::SCALING);

PTO_ENUM_MIRROR_EQ(FenceScope, LocalMemory, MlirPTOFenceScope::LocalMemory);
PTO_ENUM_MIRROR_EQ(FenceScope, GM, MlirPTOFenceScope::GM);
PTO_ENUM_MIRROR_EQ(FenceScope, All, MlirPTOFenceScope::All);

PTO_ENUM_MIRROR_EQ(LoadCachePolicy, Default,
                   MlirPTOLoadCachePolicy::Default);
PTO_ENUM_MIRROR_EQ(LoadCachePolicy, L2Bypass,
                   MlirPTOLoadCachePolicy::L2Bypass);

PTO_ENUM_MIRROR_EQ(BLayout, RowMajor, MlirPTOBLayout::RowMajor);
PTO_ENUM_MIRROR_EQ(BLayout, ColMajor, MlirPTOBLayout::ColMajor);

PTO_ENUM_MIRROR_EQ(SLayout, NoneBox, MlirPTOSLayout::NoneBox);
PTO_ENUM_MIRROR_EQ(SLayout, RowMajor, MlirPTOSLayout::RowMajor);
PTO_ENUM_MIRROR_EQ(SLayout, ColMajor, MlirPTOSLayout::ColMajor);

PTO_ENUM_MIRROR_EQ(PadValue, Null, MlirPTOPadValue::Null);
PTO_ENUM_MIRROR_EQ(PadValue, Zero, MlirPTOPadValue::Zero);
PTO_ENUM_MIRROR_EQ(PadValue, Max, MlirPTOPadValue::Max);
PTO_ENUM_MIRROR_EQ(PadValue, Min, MlirPTOPadValue::Min);

PTO_ENUM_MIRROR_EQ(CompactMode, Null, MlirPTOCompactMode::Null);
PTO_ENUM_MIRROR_EQ(CompactMode, Normal, MlirPTOCompactMode::Normal);
PTO_ENUM_MIRROR_EQ(CompactMode, RowPlusOne, MlirPTOCompactMode::RowPlusOne);

PTO_ENUM_MIRROR_EQ(RoundMode, NONE, MlirPTORoundMode::NONE);
PTO_ENUM_MIRROR_EQ(RoundMode, RINT, MlirPTORoundMode::RINT);
PTO_ENUM_MIRROR_EQ(RoundMode, ROUND, MlirPTORoundMode::ROUND);
PTO_ENUM_MIRROR_EQ(RoundMode, FLOOR, MlirPTORoundMode::FLOOR);
PTO_ENUM_MIRROR_EQ(RoundMode, CEIL, MlirPTORoundMode::CEIL);
PTO_ENUM_MIRROR_EQ(RoundMode, TRUNC, MlirPTORoundMode::TRUNC);
PTO_ENUM_MIRROR_EQ(RoundMode, ODD, MlirPTORoundMode::ODD);
PTO_ENUM_MIRROR_EQ(RoundMode, CAST_RINT, MlirPTORoundMode::CAST_RINT);

PTO_ENUM_MIRROR_EQ(DivPrecision, Intrinsic, MlirPTODivPrecision::Intrinsic);
PTO_ENUM_MIRROR_EQ(DivPrecision, DiffCompensation, MlirPTODivPrecision::DiffCompensation);
PTO_ENUM_MIRROR_EQ(DivPrecision, Precision1UlpFtzTrue, MlirPTODivPrecision::Precision1UlpFtzTrue);
PTO_ENUM_MIRROR_EQ(DivPrecision, Precision0UlpFtzTrue, MlirPTODivPrecision::Precision0UlpFtzTrue);
PTO_ENUM_MIRROR_EQ(DivPrecision, Precision0UlpFtzFalse, MlirPTODivPrecision::Precision0UlpFtzFalse);
PTO_ENUM_MIRROR_EQ(DivPrecision, Precision1UlpFtzFalse, MlirPTODivPrecision::Precision1UlpFtzFalse);
PTO_ENUM_MIRROR_EQ(ExpPrecision, Default, MlirPTOExpPrecision::Default);
PTO_ENUM_MIRROR_EQ(ExpPrecision, HighPrecision, MlirPTOExpPrecision::HighPrecision);
PTO_ENUM_MIRROR_EQ(LogPrecision, Default, MlirPTOLogPrecision::Default);
PTO_ENUM_MIRROR_EQ(LogPrecision, HighPrecision, MlirPTOLogPrecision::HighPrecision);
PTO_ENUM_MIRROR_EQ(RecipPrecision, Default, MlirPTORecipPrecision::Default);
PTO_ENUM_MIRROR_EQ(RecipPrecision, HighPrecision, MlirPTORecipPrecision::HighPrecision);
PTO_ENUM_MIRROR_EQ(RemPrecision, Default, MlirPTORemPrecision::Default);
PTO_ENUM_MIRROR_EQ(RemPrecision, HighPrecision, MlirPTORemPrecision::HighPrecision);
PTO_ENUM_MIRROR_EQ(RsqrtPrecision, Default, MlirPTORsqrtPrecision::Default);
PTO_ENUM_MIRROR_EQ(RsqrtPrecision, HighPrecision, MlirPTORsqrtPrecision::HighPrecision);
PTO_ENUM_MIRROR_EQ(SqrtPrecision, Default, MlirPTOSqrtPrecision::Default);
PTO_ENUM_MIRROR_EQ(SqrtPrecision, HighPrecision, MlirPTOSqrtPrecision::HighPrecision);
PTO_ENUM_MIRROR_EQ(FmodPrecision, Default, MlirPTOFmodPrecision::Default);
PTO_ENUM_MIRROR_EQ(FmodPrecision, HighPrecision, MlirPTOFmodPrecision::HighPrecision);

PTO_ENUM_MIRROR_EQ(SaturationMode, ON, MlirPTOSaturationMode::ON);
PTO_ENUM_MIRROR_EQ(SaturationMode, OFF, MlirPTOSaturationMode::OFF);

PTO_ENUM_MIRROR_EQ(FmatrixMode, FMATRIX_A_AUTO, MlirPTOFmatrixMode::FMATRIX_A_AUTO);
PTO_ENUM_MIRROR_EQ(FmatrixMode, FMATRIX_B_AUTO, MlirPTOFmatrixMode::FMATRIX_B_AUTO);
PTO_ENUM_MIRROR_EQ(FmatrixMode, FMATRIX_A_MANUAL, MlirPTOFmatrixMode::FMATRIX_A_MANUAL);
PTO_ENUM_MIRROR_EQ(FmatrixMode, FMATRIX_B_MANUAL, MlirPTOFmatrixMode::FMATRIX_B_MANUAL);

PTO_ENUM_MIRROR_EQ(PIPE, PIPE_S, MlirPTOPIPE::PIPE_S);
PTO_ENUM_MIRROR_EQ(PIPE, PIPE_V, MlirPTOPIPE::PIPE_V);
PTO_ENUM_MIRROR_EQ(PIPE, PIPE_M, MlirPTOPIPE::PIPE_M);
PTO_ENUM_MIRROR_EQ(PIPE, PIPE_MTE1, MlirPTOPIPE::PIPE_MTE1);
PTO_ENUM_MIRROR_EQ(PIPE, PIPE_MTE2, MlirPTOPIPE::PIPE_MTE2);
PTO_ENUM_MIRROR_EQ(PIPE, PIPE_MTE3, MlirPTOPIPE::PIPE_MTE3);
PTO_ENUM_MIRROR_EQ(PIPE, PIPE_ALL, MlirPTOPIPE::PIPE_ALL);
PTO_ENUM_MIRROR_EQ(PIPE, PIPE_MTE4, MlirPTOPIPE::PIPE_MTE4);
PTO_ENUM_MIRROR_EQ(PIPE, PIPE_MTE5, MlirPTOPIPE::PIPE_MTE5);
PTO_ENUM_MIRROR_EQ(PIPE, PIPE_V2, MlirPTOPIPE::PIPE_V2);
PTO_ENUM_MIRROR_EQ(PIPE, PIPE_FIX, MlirPTOPIPE::PIPE_FIX);
PTO_ENUM_MIRROR_EQ(PIPE, VIRTUAL_PIPE_MTE2_L1A, MlirPTOPIPE::VIRTUAL_PIPE_MTE2_L1A);
PTO_ENUM_MIRROR_EQ(PIPE, VIRTUAL_PIPE_MTE2_L1B, MlirPTOPIPE::VIRTUAL_PIPE_MTE2_L1B);
PTO_ENUM_MIRROR_EQ(PIPE, PIPE_NUM, MlirPTOPIPE::PIPE_NUM);
PTO_ENUM_MIRROR_EQ(PIPE, PIPE_UNASSIGNED, MlirPTOPIPE::PIPE_UNASSIGNED);

PTO_ENUM_MIRROR_EQ(Layout, ND, MlirPTOLayout::ND);
PTO_ENUM_MIRROR_EQ(Layout, DN, MlirPTOLayout::DN);
PTO_ENUM_MIRROR_EQ(Layout, NZ, MlirPTOLayout::NZ);
PTO_ENUM_MIRROR_EQ(Layout, MX_A_ZZ, MlirPTOLayout::MX_A_ZZ);
PTO_ENUM_MIRROR_EQ(Layout, MX_B_NN, MlirPTOLayout::MX_B_NN);

PTO_ENUM_MIRROR_EQ(ConvLayout, NC1HWC0, MlirPTOConvLayout::NC1HWC0);
PTO_ENUM_MIRROR_EQ(ConvLayout, NDC1HWC0, MlirPTOConvLayout::NDC1HWC0);
PTO_ENUM_MIRROR_EQ(ConvLayout, FRACTAL_Z, MlirPTOConvLayout::FRACTAL_Z);
PTO_ENUM_MIRROR_EQ(ConvLayout, FRACTAL_Z_3D, MlirPTOConvLayout::FRACTAL_Z_3D);
PTO_ENUM_MIRROR_EQ(ConvLayout, NCHW, MlirPTOConvLayout::NCHW);
PTO_ENUM_MIRROR_EQ(ConvLayout, NHWC, MlirPTOConvLayout::NHWC);
PTO_ENUM_MIRROR_EQ(ConvLayout, GNCHW, MlirPTOConvLayout::GNCHW);
PTO_ENUM_MIRROR_EQ(ConvLayout, GNC1HWC0, MlirPTOConvLayout::GNC1HWC0);
PTO_ENUM_MIRROR_EQ(ConvLayout, NCDHW, MlirPTOConvLayout::NCDHW);

PTO_ENUM_MIRROR_EQ(AccToVecMode, SingleModeVec0, MlirPTOAccToVecMode::SingleModeVec0);
PTO_ENUM_MIRROR_EQ(AccToVecMode, SingleModeVec1, MlirPTOAccToVecMode::SingleModeVec1);
PTO_ENUM_MIRROR_EQ(AccToVecMode, DualModeSplitM, MlirPTOAccToVecMode::DualModeSplitM);
PTO_ENUM_MIRROR_EQ(AccToVecMode, DualModeSplitN, MlirPTOAccToVecMode::DualModeSplitN);

PTO_ENUM_MIRROR_EQ(TInsertMode, SPLIT2, MlirPTOTInsertMode::SPLIT2);
PTO_ENUM_MIRROR_EQ(TInsertMode, SPLIT4, MlirPTOTInsertMode::SPLIT4);

PTO_ENUM_MIRROR_EQ(ReluPreMode, NoRelu, MlirPTOReluPreMode::NoRelu);
PTO_ENUM_MIRROR_EQ(ReluPreMode, NormalRelu, MlirPTOReluPreMode::NormalRelu);
PTO_ENUM_MIRROR_EQ(ReluPreMode, ScalarRelu, MlirPTOReluPreMode::ScalarRelu);
PTO_ENUM_MIRROR_EQ(ReluPreMode, VectorRelu, MlirPTOReluPreMode::VectorRelu);
PTO_ENUM_MIRROR_EQ(ReluPreMode, Pwl, MlirPTOReluPreMode::Pwl);

PTO_ENUM_MIRROR_EQ(AtomicType, AtomicNone, MlirPTOAtomicType::AtomicNone);
PTO_ENUM_MIRROR_EQ(AtomicType, AtomicAdd, MlirPTOAtomicType::AtomicAdd);

PTO_ENUM_MIRROR_EQ(NotifyOp, AtomicAdd, MlirPTONotifyOp::AtomicAdd);
PTO_ENUM_MIRROR_EQ(NotifyOp, Set, MlirPTONotifyOp::Set);

PTO_ENUM_MIRROR_EQ(WaitCmp, EQ, MlirPTOWaitCmp::EQ);
PTO_ENUM_MIRROR_EQ(WaitCmp, NE, MlirPTOWaitCmp::NE);
PTO_ENUM_MIRROR_EQ(WaitCmp, GT, MlirPTOWaitCmp::GT);
PTO_ENUM_MIRROR_EQ(WaitCmp, GE, MlirPTOWaitCmp::GE);
PTO_ENUM_MIRROR_EQ(WaitCmp, LT, MlirPTOWaitCmp::LT);
PTO_ENUM_MIRROR_EQ(WaitCmp, LE, MlirPTOWaitCmp::LE);

PTO_ENUM_MIRROR_EQ(ReduceOp, Sum, MlirPTOReduceOp::Sum);
PTO_ENUM_MIRROR_EQ(ReduceOp, Max, MlirPTOReduceOp::Max);
PTO_ENUM_MIRROR_EQ(ReduceOp, Min, MlirPTOReduceOp::Min);

PTO_ENUM_MIRROR_EQ(SyncOpType, TLOAD, MlirPTOSyncOpType::TLOAD);
PTO_ENUM_MIRROR_EQ(SyncOpType, TSTORE_ACC, MlirPTOSyncOpType::TSTORE_ACC);
PTO_ENUM_MIRROR_EQ(SyncOpType, TSTORE_VEC, MlirPTOSyncOpType::TSTORE_VEC);
PTO_ENUM_MIRROR_EQ(SyncOpType, TMOV_M2L, MlirPTOSyncOpType::TMOV_M2L);
PTO_ENUM_MIRROR_EQ(SyncOpType, TMOV_M2S, MlirPTOSyncOpType::TMOV_M2S);
PTO_ENUM_MIRROR_EQ(SyncOpType, TMOV_M2B, MlirPTOSyncOpType::TMOV_M2B);
PTO_ENUM_MIRROR_EQ(SyncOpType, TMOV_M2V, MlirPTOSyncOpType::TMOV_M2V);
PTO_ENUM_MIRROR_EQ(SyncOpType, TMOV_V2M, MlirPTOSyncOpType::TMOV_V2M);
PTO_ENUM_MIRROR_EQ(SyncOpType, TMATMUL, MlirPTOSyncOpType::TMATMUL);
PTO_ENUM_MIRROR_EQ(SyncOpType, TVEC, MlirPTOSyncOpType::TVEC);
PTO_ENUM_MIRROR_EQ(SyncOpType, TVECWAIT_EVENT, MlirPTOSyncOpType::TVECWAIT_EVENT);

PTO_ENUM_MIRROR_EQ(EVENT, EVENT_ID0, MlirPTOEVENT::EVENT_ID0);
PTO_ENUM_MIRROR_EQ(EVENT, EVENT_ID1, MlirPTOEVENT::EVENT_ID1);
PTO_ENUM_MIRROR_EQ(EVENT, EVENT_ID2, MlirPTOEVENT::EVENT_ID2);
PTO_ENUM_MIRROR_EQ(EVENT, EVENT_ID3, MlirPTOEVENT::EVENT_ID3);
PTO_ENUM_MIRROR_EQ(EVENT, EVENT_ID4, MlirPTOEVENT::EVENT_ID4);
PTO_ENUM_MIRROR_EQ(EVENT, EVENT_ID5, MlirPTOEVENT::EVENT_ID5);
PTO_ENUM_MIRROR_EQ(EVENT, EVENT_ID6, MlirPTOEVENT::EVENT_ID6);
PTO_ENUM_MIRROR_EQ(EVENT, EVENT_ID7, MlirPTOEVENT::EVENT_ID7);

PTO_ENUM_MIRROR_EQ(QuantType, INT8_SYM, MlirPTOQuantType::INT8_SYM);
PTO_ENUM_MIRROR_EQ(QuantType, INT8_ASYM, MlirPTOQuantType::INT8_ASYM);
PTO_ENUM_MIRROR_EQ(QuantType, MXFP8, MlirPTOQuantType::MXFP8);
PTO_ENUM_MIRROR_EQ(QuantType, MXFP4_E2M1, MlirPTOQuantType::MXFP4_E2M1);

PTO_ENUM_MIRROR_EQ(QuantScaleAlg, OCP, MlirPTOQuantScaleAlg::OCP);
PTO_ENUM_MIRROR_EQ(QuantScaleAlg, NV, MlirPTOQuantScaleAlg::NV);

PTO_ENUM_MIRROR_EQ(MxGroupAxis, Axis0, MlirPTOMxGroupAxis::Axis0);
PTO_ENUM_MIRROR_EQ(MxGroupAxis, Axis1, MlirPTOMxGroupAxis::Axis1);

PTO_ENUM_MIRROR_EQ(VecStoreMode, ND, MlirPTOVecStoreMode::ND);
PTO_ENUM_MIRROR_EQ(VecStoreMode, NZ, MlirPTOVecStoreMode::NZ);

PTO_ENUM_MIRROR_EQ(LoadL2Hint, NormalFirstVictim, MlirPTOLoadL2Hint::NormalFirstVictim);
PTO_ENUM_MIRROR_EQ(LoadL2Hint, NormalLastVictim, MlirPTOLoadL2Hint::NormalLastVictim);
PTO_ENUM_MIRROR_EQ(LoadL2Hint, NormalPersistent, MlirPTOLoadL2Hint::NormalPersistent);
PTO_ENUM_MIRROR_EQ(LoadL2Hint, NotAllocKeep, MlirPTOLoadL2Hint::NotAllocKeep);
PTO_ENUM_MIRROR_EQ(LoadL2Hint, NotAllocClean, MlirPTOLoadL2Hint::NotAllocClean);
PTO_ENUM_MIRROR_EQ(LoadL2Hint, NotAllocDrop, MlirPTOLoadL2Hint::NotAllocDrop);
PTO_ENUM_MIRROR_EQ(StoreL2Hint, NormalFirstVictim, MlirPTOStoreL2Hint::NormalFirstVictim);
PTO_ENUM_MIRROR_EQ(StoreL2Hint, NormalLastVictim, MlirPTOStoreL2Hint::NormalLastVictim);
PTO_ENUM_MIRROR_EQ(StoreL2Hint, NormalPersistent, MlirPTOStoreL2Hint::NormalPersistent);
PTO_ENUM_MIRROR_EQ(StoreL2Hint, NotAllocClean, MlirPTOStoreL2Hint::NotAllocClean);

#undef PTO_ENUM_MIRROR_EQ
} // namespace

using namespace mlir;

namespace {

constexpr unsigned kCanonicalValidShapeInlineCapacity = 4;
constexpr unsigned kI32BitWidth = 32;
constexpr unsigned kGMTypeStrideInlineCapacity = 8;
constexpr int32_t kLegacyMaskPatternP0101Value = 0;
constexpr int32_t kLegacyMaskPatternP0001Value = 3;
constexpr int32_t kLegacyMaskPatternP1111Value = 4;
constexpr int32_t kLegacyMaskPatternP1010Value = 5;

using CanonicalValidShapeVector =
    SmallVector<int64_t, kCanonicalValidShapeInlineCapacity>;

} // namespace

static CanonicalValidShapeVector
canonicalizeTileBufValidShape(ArrayRef<int64_t> validShape) {
  CanonicalValidShapeVector canonical;
  canonical.reserve(validShape.size());
  for (int64_t dim : validShape) {
    canonical.push_back(dim < 0 ? ShapedType::kDynamic : dim);
  }
  return canonical;
}

// Dialect registration (provides mlirGetDialectHandle__pto__()).
// NOTE: adjust the third argument if your dialect class name/namespace differs.
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(PTO, pto, mlir::pto::PTODialect)

//===----------------------------------------------------------------------===//
// Type queries / constructors for !pto.ptr<elem>
//===----------------------------------------------------------------------===//

bool mlirPTOTypeIsAPtrType(MlirType type) {
  return isa<mlir::pto::PtrType>(unwrap(type));;
}

MlirType mlirPTOPtrTypeGet(MlirContext ctx, MlirType elementType) {
  auto c = unwrap(ctx);
  auto elem = unwrap(elementType);
  return wrap(mlir::pto::PtrType::get(c, elem));
}

MlirType mlirPTOPtrTypeGetWithMemorySpace(MlirContext ctx, MlirType elementType,
                                          MlirAttribute memorySpace) {
  auto c = unwrap(ctx);
  auto elem = unwrap(elementType);
  auto space = mlir::cast<mlir::pto::AddressSpaceAttr>(unwrap(memorySpace));
  return wrap(mlir::pto::PtrType::get(c, elem, space));
}

MlirType mlirPTOPtrTypeGetElementType(MlirType type) {
  auto t = cast<mlir::pto::PtrType>(unwrap(type));;
  return wrap(t.getElementType());
}

bool mlirPTOTypeIsAAsyncSessionType(MlirType type) {
  return isa<mlir::pto::AsyncSessionType>(unwrap(type));
}

MlirType mlirPTOAsyncSessionTypeGet(MlirContext ctx) {
  return wrap(mlir::pto::AsyncSessionType::get(unwrap(ctx)));
}

bool mlirPTOTypeIsADmaSessionType(MlirType type) { return isa<mlir::pto::DmaSessionType>(unwrap(type)); }

MlirType mlirPTODmaSessionTypeGet(MlirContext ctx, MlirPTODmaEngine engine)
{
    return wrap(mlir::pto::DmaSessionType::get(unwrap(ctx), static_cast<mlir::pto::DmaEngine>(engine)));
}

MlirPTODmaEngine mlirPTODmaSessionTypeGetEngine(MlirType type)
{
    return static_cast<MlirPTODmaEngine>(cast<mlir::pto::DmaSessionType>(unwrap(type)).getEngine());
}

bool mlirPTOTypeIsAAsyncEventType(MlirType type) { return isa<mlir::pto::AsyncEventType>(unwrap(type)); }

MlirType mlirPTOAsyncEventTypeGet(MlirContext ctx) {
  return wrap(mlir::pto::AsyncEventType::get(unwrap(ctx)));
}

bool mlirPTOTypeIsAPrefetchAsyncContextType(MlirType type) {
  return isa<mlir::pto::PrefetchAsyncContextType>(unwrap(type));
}

MlirType mlirPTOPrefetchAsyncContextTypeGet(MlirContext ctx) {
  return wrap(mlir::pto::PrefetchAsyncContextType::get(unwrap(ctx)));
}

bool mlirPTOTypeIsAHiF8Type(MlirType type) {
  return isa<mlir::pto::HiF8Type>(unwrap(type));
}

MlirType mlirPTOHiF8TypeGet(MlirContext ctx) {
  return wrap(mlir::pto::HiF8Type::get(unwrap(ctx)));
}

bool mlirPTOTypeIsAF8E8M0Type(MlirType type) {
  return isa<mlir::pto::F8E8M0Type>(unwrap(type));
}

MlirType mlirPTOF8E8M0TypeGet(MlirContext ctx) {
  return wrap(mlir::pto::F8E8M0Type::get(unwrap(ctx)));
}

bool mlirPTOTypeIsAHiF8x2Type(MlirType type) {
  return isa<mlir::pto::HiF8x2Type>(unwrap(type));
}

MlirType mlirPTOHiF8x2TypeGet(MlirContext ctx) {
  return wrap(mlir::pto::HiF8x2Type::get(unwrap(ctx)));
}

bool mlirPTOTypeIsAF4E1M2x2Type(MlirType type) {
  return isa<mlir::pto::F4E1M2x2Type>(unwrap(type));
}

MlirType mlirPTOF4E1M2x2TypeGet(MlirContext ctx) {
  return wrap(mlir::pto::F4E1M2x2Type::get(unwrap(ctx)));
}

bool mlirPTOTypeIsAF4E2M1x2Type(MlirType type) {
  return isa<mlir::pto::F4E2M1x2Type>(unwrap(type));
}

MlirType mlirPTOF4E2M1x2TypeGet(MlirContext ctx) {
  return wrap(mlir::pto::F4E2M1x2Type::get(unwrap(ctx)));
}

bool mlirPTOTypeIsABF16x2Type(MlirType type) {
  return isa<mlir::pto::BF16x2Type>(unwrap(type));
}

MlirType mlirPTOBF16x2TypeGet(MlirContext ctx) {
  return wrap(mlir::pto::BF16x2Type::get(unwrap(ctx)));
}

MlirAttribute mlirPTOPtrTypeGetMemorySpace(MlirType type) {
  auto t = cast<mlir::pto::PtrType>(unwrap(type));
  return wrap(t.getMemorySpace());
}

bool mlirPTOAttrIsAAddressSpaceAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::AddressSpaceAttr>(unwrap(attr));
}

MlirAttribute mlirPTOAddressSpaceAttrGet(MlirContext ctx, int32_t value) {
  auto c = unwrap(ctx);

  // 你的 ODS 里 AddressSpaceAttr 的参数是 EnumParameter<PTO_AddressSpaceEnum>
  // 通常对应 C++ 里是一个 enum class AddressSpace : int32_t
  auto v = static_cast<mlir::pto::AddressSpace>(value);

  return wrap(mlir::pto::AddressSpaceAttr::get(c, v));
}

int32_t mlirPTOAddressSpaceAttrGetValue(MlirAttribute attr) {
  auto a = mlir::cast<mlir::pto::AddressSpaceAttr>(unwrap(attr));
  return static_cast<int32_t>(a.getAddressSpace());
}


bool mlirPTOAttrIsALoadL2HintAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::LoadL2HintAttr>(unwrap(attr));
}

MlirAttribute mlirPTOLoadL2HintAttrGet(MlirContext ctx, int32_t value) {
  // Enum encodings are sparse and direction-specific; reject every hole.
  auto mode = mlir::pto::symbolizeLoadL2Hint(static_cast<uint32_t>(value));
  if (!mode) {
    return MlirAttribute{nullptr};
  }
  return wrap(mlir::pto::LoadL2HintAttr::get(unwrap(ctx), *mode));
}

int32_t mlirPTOLoadL2HintAttrGetValue(MlirAttribute attr) {
  return static_cast<int32_t>(mlir::cast<mlir::pto::LoadL2HintAttr>(unwrap(attr)).getValue());
}


bool mlirPTOAttrIsAStoreL2HintAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::StoreL2HintAttr>(unwrap(attr));
}

MlirAttribute mlirPTOStoreL2HintAttrGet(MlirContext ctx, int32_t value) {
  // Enum encodings are sparse and direction-specific; reject every hole.
  auto mode = mlir::pto::symbolizeStoreL2Hint(static_cast<uint32_t>(value));
  if (!mode) {
    return MlirAttribute{nullptr};
  }
  return wrap(mlir::pto::StoreL2HintAttr::get(unwrap(ctx), *mode));
}

int32_t mlirPTOStoreL2HintAttrGetValue(MlirAttribute attr) {
  return static_cast<int32_t>(mlir::cast<mlir::pto::StoreL2HintAttr>(unwrap(attr)).getValue());
}

bool mlirPTOAttrIsALoadCachePolicyAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::LoadCachePolicyAttr>(unwrap(attr));
}

MlirAttribute mlirPTOLoadCachePolicyAttrGet(MlirContext ctx, int32_t value) {
  const bool isValidPolicy =
      value >= static_cast<int32_t>(mlir::pto::LoadCachePolicy::Default) &&
      value <= static_cast<int32_t>(mlir::pto::LoadCachePolicy::L2Bypass);
  if (!isValidPolicy) {
    return MlirAttribute{nullptr};
  }
  auto policy = static_cast<mlir::pto::LoadCachePolicy>(value);
  return wrap(mlir::pto::LoadCachePolicyAttr::get(unwrap(ctx), policy));
}

int32_t mlirPTOLoadCachePolicyAttrGetValue(MlirAttribute attr) {
  auto policy = mlir::cast<mlir::pto::LoadCachePolicyAttr>(unwrap(attr));
  return static_cast<int32_t>(policy.getValue());
}

//===----------------------------------------------------------------------===//
// Type queries / constructors for !pto.tensor_view<shape x elem>
//===----------------------------------------------------------------------===//

bool mlirPTOTypeIsATensorViewType(MlirType type) {
  return isa<mlir::pto::TensorViewType>(unwrap(type));
}

MlirType mlirPTOTensorViewTypeGet(MlirContext ctx, intptr_t rank,
                                  const int64_t *shape, MlirType elementType) {
  auto c = unwrap(ctx);
  auto elem = unwrap(elementType);
  llvm::ArrayRef<int64_t> shp(shape, static_cast<size_t>(rank));
  return wrap(mlir::pto::TensorViewType::get(c, shp, elem));
}

intptr_t mlirPTOTensorViewTypeGetRank(MlirType type) {
  auto t = cast<mlir::pto::TensorViewType>(unwrap(type));
  return static_cast<intptr_t>(t.getShape().size());
}

MlirType mlirPTOTensorViewTypeGetElementType(MlirType type) {
  auto t = cast<mlir::pto::TensorViewType>(unwrap(type));
  return wrap(t.getElementType());
}

const int64_t *mlirPTOTensorViewTypeGetShape(MlirType type, intptr_t *numDimsOut) {
  auto t = cast<mlir::pto::TensorViewType>(unwrap(type));
  auto shape = t.getShape();
  *numDimsOut = static_cast<intptr_t>(shape.size());
  return shape.data();
}

//===----------------------------------------------------------------------===//
// !pto.tile_view<shape x elem>
//===----------------------------------------------------------------------===//

bool mlirPTOTypeIsAPartitionTensorViewType(MlirType type) {
  return isa<mlir::pto::PartitionTensorViewType>(unwrap(type));
}

MlirType mlirPTOPartitionTensorViewTypeGet(MlirContext ctx, intptr_t rank,
                                const int64_t *shape, MlirType elementType) {
  auto c = unwrap(ctx);
  auto elem = unwrap(elementType);
  llvm::ArrayRef<int64_t> shp(shape, static_cast<size_t>(rank));
  return wrap(mlir::pto::PartitionTensorViewType::get(c, shp, elem));
}

intptr_t mlirPTOPartitionTensorViewTypeGetRank(MlirType type) {
  auto t = cast<mlir::pto::PartitionTensorViewType>(unwrap(type));
  return static_cast<intptr_t>(t.getShape().size());
}

MlirType mlirPTOPartitionTensorViewTypeGetElementType(MlirType type) {
  auto t = mlir::cast<mlir::pto::PartitionTensorViewType>(unwrap(type));
  return wrap(t.getElementType());
}

const int64_t *mlirPTOPartitionTensorViewTypeGetShape(MlirType type, intptr_t *numDimsOut) {
  auto t = cast<mlir::pto::PartitionTensorViewType>(unwrap(type));
  auto shape = t.getShape();
  *numDimsOut = static_cast<intptr_t>(shape.size());
  return shape.data();
}

//===----------------------------------------------------------------------===//
// !pto.tile<shape x elem>
//===----------------------------------------------------------------------===//

bool mlirPTOTypeIsATileType(MlirType type) {
  return isa<mlir::pto::TileType>(unwrap(type));
}

MlirType mlirPTOTileTypeGet(MlirContext ctx, intptr_t rank,
                            const int64_t *shape, MlirType elementType) {
  auto c = unwrap(ctx);
  auto elem = unwrap(elementType);
  llvm::ArrayRef<int64_t> shp(shape, static_cast<size_t>(rank));
  return wrap(mlir::pto::TileType::get(c, shp, elem));
}

intptr_t mlirPTOTileTypeGetRank(MlirType type) {
  auto t = cast<mlir::pto::TileType>(unwrap(type));
  return static_cast<intptr_t>(t.getShape().size());
}

MlirType mlirPTOTileTypeGetElementType(MlirType type) {
  auto t = cast<mlir::pto::TileType>(unwrap(type));
  return wrap(t.getElementType());
}

const int64_t *mlirPTOTileTypeGetShape(MlirType type, intptr_t *numDimsOut) {
  auto t = cast<mlir::pto::TileType>(unwrap(type));
  auto shape = t.getShape();
  *numDimsOut = static_cast<intptr_t>(shape.size());
  return shape.data();
}

bool mlirPTOTypeIsATileBufType(MlirType type) {
  return mlir::isa<mlir::pto::TileBufType>(unwrap(type));
}

MlirType mlirPTOTileBufTypeGet(MlirContext ctx, intptr_t rank,
                               const int64_t *shape, MlirType elementType,
                               MlirAttribute memorySpace) {
  MLIRContext *c = unwrap(ctx);
  auto shp = llvm::ArrayRef<int64_t>(shape, rank);
  auto cfg = mlir::pto::TileBufConfigAttr::getDefault(c);
  auto canonicalValidShape = canonicalizeTileBufValidShape(llvm::ArrayRef<int64_t>{});
  auto ty = mlir::pto::TileBufType::get(c, shp, unwrap(elementType),
                                        unwrap(memorySpace), canonicalValidShape, cfg);
  return wrap(ty);
}

MlirType mlirPTOTileBufTypeGetWithConfig(MlirContext ctx, intptr_t rank,
                                         const int64_t *shape, MlirType elementType,
                                         MlirAttribute memorySpace, MlirAttribute config) {
  MLIRContext *c = unwrap(ctx);
  auto shp = llvm::ArrayRef<int64_t>(shape, rank);
  auto cfg = mlir::dyn_cast_or_null<mlir::pto::TileBufConfigAttr>(unwrap(config));
  if (!cfg) {
    cfg = mlir::pto::TileBufConfigAttr::getDefault(c);
  }
  auto ty = mlir::pto::TileBufType::get(c, shp, unwrap(elementType), unwrap(memorySpace), cfg);
  return wrap(ty);
}

MlirType mlirPTOTileBufTypeGetWithValidShape(MlirContext ctx,
                                             intptr_t rank,
                                             const int64_t *shape,
                                             MlirType elementType,
                                             MlirAttribute memorySpace,
                                             intptr_t validRank,
                                             const int64_t *validShape) {
  MLIRContext *c = unwrap(ctx);
  auto shp = llvm::ArrayRef<int64_t>(shape, rank);
  auto vs  = llvm::ArrayRef<int64_t>(validShape, validRank);
  auto cfg = mlir::pto::TileBufConfigAttr::getDefault(c);
  auto canonicalValidShape = canonicalizeTileBufValidShape(vs);

  auto ty = mlir::pto::TileBufType::get(c, shp, unwrap(elementType),
                                       unwrap(memorySpace), canonicalValidShape, cfg);
  return wrap(ty);
}

MlirType mlirPTOTileBufTypeGetWithValidShapeAndConfig(MlirContext ctx,
                                                      intptr_t rank,
                                                      const int64_t *shape,
                                                      MlirType elementType,
                                                      MlirAttribute memorySpace,
                                                      intptr_t validRank,
                                                      const int64_t *validShape,
                                                      MlirAttribute config) {
  MLIRContext *c = unwrap(ctx);
  auto shp = llvm::ArrayRef<int64_t>(shape, rank);
  auto vs  = llvm::ArrayRef<int64_t>(validShape, validRank);
  auto cfg = mlir::cast<mlir::pto::TileBufConfigAttr>(unwrap(config));
  auto canonicalValidShape = canonicalizeTileBufValidShape(vs);

  auto ty = mlir::pto::TileBufType::get(c, shp, unwrap(elementType),
                                       unwrap(memorySpace), canonicalValidShape, cfg);
  return wrap(ty);
}

bool mlirPTOTypeIsAConvTileBufType(MlirType type) {
  return mlir::isa<mlir::pto::ConvTileBufType>(unwrap(type));
}

MlirType mlirPTOConvTileBufTypeGet(MlirContext ctx, intptr_t rank,
                                const int64_t *shape, MlirType elementType,
                                MlirAttribute bufferSize,
                                MlirAttribute memorySpace,
                                MlirAttribute layout) {
    // Zero-length valid shape means "valid == declared".
    return mlirPTOConvTileBufTypeGetWithValid(
        ctx, rank, shape, elementType, bufferSize, memorySpace, layout, 0, nullptr);
}

MlirType mlirPTOConvTileBufTypeGetWithValid(
    MlirContext ctx, intptr_t rank, const int64_t* shape, MlirType elementType, MlirAttribute bufferSize,
    MlirAttribute memorySpace, MlirAttribute layout, intptr_t validRank, const int64_t* validShape)
{
    MLIRContext* c = unwrap(ctx);
    auto shp = llvm::ArrayRef<int64_t>(shape, rank);
    auto vshp = llvm::ArrayRef<int64_t>(validShape, validRank);
    auto bufferSizeAttr = mlir::dyn_cast<mlir::IntegerAttr>(unwrap(bufferSize));
    auto memorySpaceAttr = mlir::dyn_cast<mlir::pto::AddressSpaceAttr>(unwrap(memorySpace));
    auto layoutAttr = mlir::dyn_cast<mlir::pto::ConvLayoutAttr>(unwrap(layout));
    if (!bufferSizeAttr || !memorySpaceAttr || !layoutAttr) {
        return MlirType{nullptr};
    }
    // Canonicalize like the parser does: a valid shape equal to the declared
    // shape is stored as "unspecified", so the API and the textual path build
    // the identical type instead of the API one failing verification.
    if (vshp == shp) {
        vshp = llvm::ArrayRef<int64_t>();
    }

    // getChecked runs ConvTileBufType::verify, so rank/layout/memory-space
    // consistency holds for CAPI and Python-built types too; on failure the
    // diagnostics go to the UnknownLoc and the returned type is null. The
    // conv geometry config is not part of the type; it rides on the
    // allocating op.
    auto ty = mlir::pto::ConvTileBufType::getChecked(
        [c]() { return mlir::emitError(mlir::UnknownLoc::get(c)); }, c, shp, unwrap(elementType), bufferSizeAttr,
        memorySpaceAttr, layoutAttr, vshp);
    return wrap(ty);
}

intptr_t mlirPTOConvTileBufTypeGetRank(MlirType type) {
  return mlir::cast<mlir::pto::ConvTileBufType>(unwrap(type)).getRank();
}

MlirType mlirPTOConvTileBufTypeGetElementType(MlirType type)
{
    return wrap(mlir::cast<mlir::pto::ConvTileBufType>(unwrap(type)).getElementType());
}

const int64_t *mlirPTOConvTileBufTypeGetShape(MlirType type,
                                           intptr_t *numDimsOut) {
  static thread_local llvm::SmallVector<int64_t, 8> shape;
  auto dims = mlir::cast<mlir::pto::ConvTileBufType>(unwrap(type)).getShape();
  shape.assign(dims.begin(), dims.end());
  *numDimsOut = static_cast<intptr_t>(shape.size());
  return shape.data();
}

MlirAttribute mlirPTOConvTileBufTypeGetLayout(MlirType type) {
  return wrap(mlir::cast<mlir::pto::ConvTileBufType>(unwrap(type)).getLayout());
}

const int64_t* mlirPTOConvTileBufTypeGetValidShape(MlirType type, intptr_t* numDimsOut)
{
    static thread_local llvm::SmallVector<int64_t, 8> shape;
    auto dims = mlir::cast<mlir::pto::ConvTileBufType>(unwrap(type)).getValidShape();
    if (dims.empty()) {
        *numDimsOut = 0;
        return nullptr;
    }
    shape.assign(dims.begin(), dims.end());
    *numDimsOut = static_cast<intptr_t>(shape.size());
    return shape.data();
}

bool mlirPTOAttrIsABLayoutAttr(MlirAttribute attr) { return mlir::isa<mlir::pto::BLayoutAttr>(unwrap(attr)); }

MlirAttribute mlirPTOBLayoutAttrGet(MlirContext ctx, int32_t value)
{
    auto* c = unwrap(ctx);
    auto v = static_cast<mlir::pto::BLayout>(value);
    return wrap(mlir::pto::BLayoutAttr::get(c, v));
}

int32_t mlirPTOBLayoutAttrGetValue(MlirAttribute attr) {
  auto a = mlir::cast<mlir::pto::BLayoutAttr>(unwrap(attr));
  return static_cast<int32_t>(a.getValue());
}

bool mlirPTOAttrIsASLayoutAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::SLayoutAttr>(unwrap(attr));
}

MlirAttribute mlirPTOSLayoutAttrGet(MlirContext ctx, int32_t value) {
  auto *c = unwrap(ctx);
  auto v = static_cast<mlir::pto::SLayout>(value);
  return wrap(mlir::pto::SLayoutAttr::get(c, v));
}

int32_t mlirPTOSLayoutAttrGetValue(MlirAttribute attr) {
  auto a = mlir::cast<mlir::pto::SLayoutAttr>(unwrap(attr));
  return static_cast<int32_t>(a.getValue());
}

bool mlirPTOAttrIsAPadValueAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::PadValueAttr>(unwrap(attr));
}

MlirAttribute mlirPTOPadValueAttrGet(MlirContext ctx, int32_t value) {
  auto *c = unwrap(ctx);
  auto v = static_cast<mlir::pto::PadValue>(value);
  return wrap(mlir::pto::PadValueAttr::get(c, v));
}

int32_t mlirPTOPadValueAttrGetValue(MlirAttribute attr) {
  auto a = mlir::cast<mlir::pto::PadValueAttr>(unwrap(attr));
  return static_cast<int32_t>(a.getValue());
}

MlirAttribute mlirPTORoundModeAttrGet(MlirContext ctx, int32_t value) {
  auto *c = unwrap(ctx);
  auto mode = static_cast<mlir::pto::RoundMode>(value);
  return wrap(mlir::pto::RoundModeAttr::get(c, mode));
}

bool mlirPTOAttrIsARoundModeAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::RoundModeAttr>(unwrap(attr));
}

int32_t mlirPTORoundModeAttrGetValue(MlirAttribute attr) {
  auto a = mlir::cast<mlir::pto::RoundModeAttr>(unwrap(attr));
  return static_cast<int32_t>(a.getValue());
}

#define DEFINE_PTO_ENUM_ATTR_CAPI(NAME, ATTR, ENUM)                            \
  MlirAttribute mlirPTO##NAME##AttrGet(MlirContext ctx, int32_t value) {       \
    auto *c = unwrap(ctx);                                                     \
    auto mode = static_cast<mlir::pto::ENUM>(value);                           \
    return wrap(mlir::pto::ATTR::get(c, mode));                                \
  }                                                                            \
                                                                               \
  bool mlirPTOAttrIsA##NAME##Attr(MlirAttribute attr) {                        \
    return mlir::isa<mlir::pto::ATTR>(unwrap(attr));                           \
  }                                                                            \
                                                                               \
  int32_t mlirPTO##NAME##AttrGetValue(MlirAttribute attr) {                    \
    auto a = mlir::cast<mlir::pto::ATTR>(unwrap(attr));                        \
    return static_cast<int32_t>(a.getValue());                                 \
  }

DEFINE_PTO_ENUM_ATTR_CAPI(DivPrecision, DivPrecisionAttr, DivPrecision)
DEFINE_PTO_ENUM_ATTR_CAPI(ExpPrecision, ExpPrecisionAttr, ExpPrecision)
DEFINE_PTO_ENUM_ATTR_CAPI(LogPrecision, LogPrecisionAttr, LogPrecision)
DEFINE_PTO_ENUM_ATTR_CAPI(RecipPrecision, RecipPrecisionAttr, RecipPrecision)
DEFINE_PTO_ENUM_ATTR_CAPI(RemPrecision, RemPrecisionAttr, RemPrecision)
DEFINE_PTO_ENUM_ATTR_CAPI(RsqrtPrecision, RsqrtPrecisionAttr, RsqrtPrecision)
DEFINE_PTO_ENUM_ATTR_CAPI(SqrtPrecision, SqrtPrecisionAttr, SqrtPrecision)
DEFINE_PTO_ENUM_ATTR_CAPI(FmodPrecision, FmodPrecisionAttr, FmodPrecision)
DEFINE_PTO_ENUM_ATTR_CAPI(FmatrixMode, FmatrixModeAttr, FmatrixMode)

#undef DEFINE_PTO_ENUM_ATTR_CAPI

MlirAttribute mlirPTOSaturationModeAttrGet(MlirContext ctx, int32_t value) {
  auto *c = unwrap(ctx);
  auto mode = static_cast<mlir::pto::SaturationMode>(value);
  return wrap(mlir::pto::SaturationModeAttr::get(c, mode));
}

bool mlirPTOAttrIsASaturationModeAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::SaturationModeAttr>(unwrap(attr));
}

int32_t mlirPTOSaturationModeAttrGetValue(MlirAttribute attr) {
  auto a = mlir::cast<mlir::pto::SaturationModeAttr>(unwrap(attr));
  return static_cast<int32_t>(a.getValue());
}

MlirAttribute mlirPTOPipeAttrGet(MlirContext ctx, int32_t value) {
  auto *c = unwrap(ctx);
  auto v = static_cast<mlir::pto::PIPE>(value);
  return wrap(mlir::pto::PipeAttr::get(c, v));
}

bool mlirPTOAttrIsAPipeAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::PipeAttr>(unwrap(attr));
}

int32_t mlirPTOPipeAttrGetValue(MlirAttribute attr) {
  auto a = mlir::cast<mlir::pto::PipeAttr>(unwrap(attr));
  return static_cast<int32_t>(a.getPipe());
}

MlirAttribute mlirPTOLayoutAttrGet(MlirContext ctx, int32_t value) {
  auto *c = unwrap(ctx);
  auto v = static_cast<mlir::pto::Layout>(value);
  return wrap(mlir::pto::LayoutAttr::get(c, v));
}

bool mlirPTOAttrIsALayoutAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::LayoutAttr>(unwrap(attr));
}

int32_t mlirPTOLayoutAttrGetValue(MlirAttribute attr) {
  auto a = mlir::cast<mlir::pto::LayoutAttr>(unwrap(attr));
  return static_cast<int32_t>(a.getLayout());
}

MlirAttribute mlirPTOConvLayoutAttrGet(MlirContext ctx, int32_t value) {
  auto *c = unwrap(ctx);
  auto v = static_cast<mlir::pto::ConvLayout>(value);
  return wrap(mlir::pto::ConvLayoutAttr::get(c, v));
}

bool mlirPTOAttrIsAConvLayoutAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::ConvLayoutAttr>(unwrap(attr));
}

int32_t mlirPTOConvLayoutAttrGetValue(MlirAttribute attr) {
  auto a = mlir::cast<mlir::pto::ConvLayoutAttr>(unwrap(attr));
  return static_cast<int32_t>(a.getValue());
}

MlirAttribute mlirPTOSyncOpTypeAttrGet(MlirContext ctx, int32_t value) {
  auto *c = unwrap(ctx);
  auto mode = static_cast<mlir::pto::SyncOpType>(value);
  return wrap(mlir::pto::SyncOpTypeAttr::get(c, mode));
}

bool mlirPTOAttrIsASyncOpTypeAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::SyncOpTypeAttr>(unwrap(attr));
}

int32_t mlirPTOSyncOpTypeAttrGetValue(MlirAttribute attr) {
  auto a = mlir::cast<mlir::pto::SyncOpTypeAttr>(unwrap(attr));
  return static_cast<int32_t>(a.getOpType());
}

bool mlirPTOAttrIsAFenceScopeAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::FenceScopeAttr>(unwrap(attr));
}

MlirAttribute mlirPTOFenceScopeAttrGet(MlirContext ctx, int32_t value) {
  auto *c = unwrap(ctx);
  auto scope = static_cast<mlir::pto::FenceScope>(value);
  return wrap(mlir::pto::FenceScopeAttr::get(c, scope));
}

int32_t mlirPTOFenceScopeAttrGetValue(MlirAttribute attr) {
  auto a = mlir::cast<mlir::pto::FenceScopeAttr>(unwrap(attr));
  return static_cast<int32_t>(a.getScope());
}

MlirAttribute mlirPTOEventAttrGet(MlirContext ctx, int32_t value) {
  auto *c = unwrap(ctx);
  auto v = static_cast<mlir::pto::EVENT>(value);
  return wrap(mlir::pto::EventAttr::get(c, v));
}

MlirAttribute mlirPTOQuantTypeAttrGet(MlirContext ctx, int32_t value) {
  auto *c = unwrap(ctx);
  auto v = static_cast<mlir::pto::QuantType>(value);
  return wrap(mlir::pto::QuantTypeAttr::get(c, v));
}

bool mlirPTOAttrIsAQuantTypeAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::QuantTypeAttr>(unwrap(attr));
}

int32_t mlirPTOQuantTypeAttrGetValue(MlirAttribute attr) {
  auto a = mlir::cast<mlir::pto::QuantTypeAttr>(unwrap(attr));
  return static_cast<int32_t>(a.getValue());
}

MlirAttribute mlirPTOQuantScaleAlgAttrGet(MlirContext ctx, int32_t value) {
  auto *c = unwrap(ctx);
  auto v = static_cast<mlir::pto::QuantScaleAlg>(value);
  return wrap(mlir::pto::QuantScaleAlgAttr::get(c, v));
}

bool mlirPTOAttrIsAQuantScaleAlgAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::QuantScaleAlgAttr>(unwrap(attr));
}

int32_t mlirPTOQuantScaleAlgAttrGetValue(MlirAttribute attr) {
  auto a = mlir::cast<mlir::pto::QuantScaleAlgAttr>(unwrap(attr));
  return static_cast<int32_t>(a.getValue());
}

MlirAttribute mlirPTOMxGroupAxisAttrGet(MlirContext ctx, int32_t value) {
  auto *c = unwrap(ctx);
  auto v = static_cast<mlir::pto::MxGroupAxis>(value);
  return wrap(mlir::pto::MxGroupAxisAttr::get(c, v));
}

bool mlirPTOAttrIsAMxGroupAxisAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::MxGroupAxisAttr>(unwrap(attr));
}

int32_t mlirPTOMxGroupAxisAttrGetValue(MlirAttribute attr) {
  auto a = mlir::cast<mlir::pto::MxGroupAxisAttr>(unwrap(attr));
  return static_cast<int32_t>(a.getValue());
}

MlirAttribute mlirPTOVecStoreModeAttrGet(MlirContext ctx, int32_t value) {
  auto *c = unwrap(ctx);
  auto v = static_cast<mlir::pto::VecStoreMode>(value);
  return wrap(mlir::pto::VecStoreModeAttr::get(c, v));
}

bool mlirPTOAttrIsAVecStoreModeAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::VecStoreModeAttr>(unwrap(attr));
}

int32_t mlirPTOVecStoreModeAttrGetValue(MlirAttribute attr) {
  auto a = mlir::cast<mlir::pto::VecStoreModeAttr>(unwrap(attr));
  return static_cast<int32_t>(a.getValue());
}

bool mlirPTOAttrIsAEventAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::EventAttr>(unwrap(attr));
}

int32_t mlirPTOEventAttrGetValue(MlirAttribute attr) {
  auto a = mlir::cast<mlir::pto::EventAttr>(unwrap(attr));
  return static_cast<int32_t>(a.getEvent());
}

static std::optional<mlir::pto::MaskPattern>
maskPatternFromIsaValue(int32_t value) {
  switch (value) {
  case static_cast<int32_t>(mlir::pto::MaskPattern::P0101):
    return mlir::pto::MaskPattern::P0101;
  case static_cast<int32_t>(mlir::pto::MaskPattern::P1010):
    return mlir::pto::MaskPattern::P1010;
  case static_cast<int32_t>(mlir::pto::MaskPattern::P0001):
    return mlir::pto::MaskPattern::P0001;
  case static_cast<int32_t>(mlir::pto::MaskPattern::P0010):
    return mlir::pto::MaskPattern::P0010;
  case static_cast<int32_t>(mlir::pto::MaskPattern::P0100):
    return mlir::pto::MaskPattern::P0100;
  case static_cast<int32_t>(mlir::pto::MaskPattern::P1000):
    return mlir::pto::MaskPattern::P1000;
  case static_cast<int32_t>(mlir::pto::MaskPattern::P1111):
    return mlir::pto::MaskPattern::P1111;
  default:
    return std::nullopt;
  }
}

static std::optional<mlir::pto::MaskPattern>
maskPatternFromLegacyRaw(int32_t value) {
  switch (value) {
  case kLegacyMaskPatternP0101Value:
    return mlir::pto::MaskPattern::P0101;
  case kLegacyMaskPatternP0001Value:
    return mlir::pto::MaskPattern::P0001;
  case kLegacyMaskPatternP1111Value:
    return mlir::pto::MaskPattern::P1111;
  case kLegacyMaskPatternP1010Value:
    return mlir::pto::MaskPattern::P1010;
  default:
    return std::nullopt;
  }
}

MlirAttribute mlirPTOMaskPatternAttrGet(MlirContext ctx, int32_t value) {
  auto *c = unwrap(ctx);
  std::optional<mlir::pto::MaskPattern> v;
  switch (value) {
  case kLegacyMaskPatternP0101Value:
  case kLegacyMaskPatternP0001Value:
    v = maskPatternFromLegacyRaw(value);
    break;
  case static_cast<int32_t>(mlir::pto::MaskPattern::P1000):
  case static_cast<int32_t>(mlir::pto::MaskPattern::P1111):
    v = maskPatternFromIsaValue(value);
    break;
  default:
    break;
  }
  if (!v) {
    return MlirAttribute{nullptr};
  }
  return wrap(mlir::pto::MaskPatternAttr::get(c, *v));
}

MlirAttribute mlirPTOMaskPatternAttrGetLegacyRaw(MlirContext ctx, int32_t value) {
  auto *c = unwrap(ctx);
  std::optional<mlir::pto::MaskPattern> v = maskPatternFromLegacyRaw(value);
  if (!v) {
    return MlirAttribute{nullptr};
  }
  return wrap(mlir::pto::MaskPatternAttr::get(c, *v));
}

bool mlirPTOAttrIsAMaskPatternAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::MaskPatternAttr>(unwrap(attr));
}

int32_t mlirPTOMaskPatternAttrGetValue(MlirAttribute attr) {
  auto a = mlir::cast<mlir::pto::MaskPatternAttr>(unwrap(attr));
  return static_cast<int32_t>(a.getValue());
}

MlirAttribute mlirPTOMaskPatternAttrGetEnum(MlirContext ctx,
                                            MlirPTOMaskPattern value) {
  auto *c = unwrap(ctx);
  std::optional<mlir::pto::MaskPattern> v =
      maskPatternFromIsaValue(static_cast<int32_t>(value));
  if (!v) {
    return MlirAttribute{nullptr};
  }
  return wrap(mlir::pto::MaskPatternAttr::get(c, *v));
}

MlirPTOMaskPattern mlirPTOMaskPatternAttrGetEnumValue(MlirAttribute attr) {
  auto a = mlir::cast<mlir::pto::MaskPatternAttr>(unwrap(attr));
  return static_cast<MlirPTOMaskPattern>(static_cast<int32_t>(a.getValue()));
}

bool mlirAttributeIsAPTOCmpModeAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::CmpModeAttr>(unwrap(attr));
}

MlirAttribute mlirPTOCmpModeAttrGet(MlirContext ctx, MlirPTOCmpMode value) {
  auto *c = unwrap(ctx);
  auto mode = static_cast<mlir::pto::CmpMode>(value);
  return wrap(mlir::pto::CmpModeAttr::get(c, mode));
}

MlirPTOCmpMode mlirPTOCmpModeAttrGetValue(MlirAttribute attr) {
  auto a = mlir::cast<mlir::pto::CmpModeAttr>(unwrap(attr));
  return static_cast<MlirPTOCmpMode>(static_cast<uint32_t>(a.getValue()));
}

MlirAttribute mlirPTOCoalesceAttrGet(MlirContext ctx, MlirPTOCoalesce value) {
  auto *c = unwrap(ctx);
  auto coalesce = static_cast<mlir::pto::Coalesce>(value);
  return wrap(mlir::pto::CoalesceAttr::get(c, coalesce));
}

bool mlirPTOAttrIsACoalesceAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::CoalesceAttr>(unwrap(attr));
}

MlirPTOCoalesce mlirPTOCoalesceAttrGetValue(MlirAttribute attr) {
  auto a = mlir::cast<mlir::pto::CoalesceAttr>(unwrap(attr));
  return static_cast<MlirPTOCoalesce>(static_cast<uint32_t>(a.getValue()));
}

bool mlirPTOAttrIsATileBufConfigAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::TileBufConfigAttr>(unwrap(attr));
}

MlirAttribute mlirPTOTileBufConfigAttrGetDefault(MlirContext ctx) {
  auto *c = unwrap(ctx);
  return wrap(mlir::pto::TileBufConfigAttr::getDefault(c));
}

static mlir::pto::BLayoutAttr toBLayoutAttr(mlir::MLIRContext *c, mlir::Attribute a) {
  if (auto bl = mlir::dyn_cast<mlir::pto::BLayoutAttr>(a)) {
    return bl;
  }
  if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(a)) {
    return mlir::pto::BLayoutAttr::get(c, static_cast<mlir::pto::BLayout>(ia.getInt()));
  }
  return {};
}
static mlir::pto::SLayoutAttr toSLayoutAttr(mlir::MLIRContext *c, mlir::Attribute a) {
  if (auto sl = mlir::dyn_cast<mlir::pto::SLayoutAttr>(a)) {
    return sl;
  }
  if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(a)) {
    return mlir::pto::SLayoutAttr::get(c, static_cast<mlir::pto::SLayout>(ia.getInt()));
  }
  return {};
}
static mlir::pto::PadValueAttr toPadValueAttr(mlir::MLIRContext *c, mlir::Attribute a) {
  if (auto pv = mlir::dyn_cast<mlir::pto::PadValueAttr>(a)) {
    return pv;
  }
  if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(a)) {
    return mlir::pto::PadValueAttr::get(c, static_cast<mlir::pto::PadValue>(ia.getInt()));
  }
  return {};
}
static mlir::pto::CompactModeAttr toCompactModeAttr(mlir::MLIRContext *c,
                                                    mlir::Attribute a) {
  if (auto cm = mlir::dyn_cast<mlir::pto::CompactModeAttr>(a)) {
    return cm;
  }
  if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(a)) {
    return mlir::pto::CompactModeAttr::get(
        c, static_cast<mlir::pto::CompactMode>(ia.getInt()));
  }
  return {};
}

static mlir::IntegerAttr toI32IntegerAttr(mlir::Attribute a) {
  auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(a);
  if (!intAttr || !intAttr.getType().isSignlessInteger(kI32BitWidth)) {
    return {};
  }
  return intAttr;
}

bool mlirPTOAttrIsACompactModeAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::CompactModeAttr>(unwrap(attr));
}

MlirAttribute mlirPTOCompactModeAttrGet(MlirContext ctx, int32_t value) {
  auto *c = unwrap(ctx);
  return wrap(mlir::pto::CompactModeAttr::get(
      c, static_cast<mlir::pto::CompactMode>(value)));
}

int32_t mlirPTOCompactModeAttrGetValue(MlirAttribute attr) {
  auto a = mlir::cast<mlir::pto::CompactModeAttr>(unwrap(attr));
  return static_cast<int32_t>(a.getValue());
}

bool mlirPTOAttrIsAAccToVecModeAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::AccToVecModeAttr>(unwrap(attr));
}

MlirAttribute mlirPTOAccToVecModeAttrGet(MlirContext ctx, int32_t value) {
  auto *c = unwrap(ctx);
  return wrap(mlir::pto::AccToVecModeAttr::get(
      c, static_cast<mlir::pto::AccToVecMode>(value)));
}

int32_t mlirPTOAccToVecModeAttrGetValue(MlirAttribute attr) {
  auto a = mlir::cast<mlir::pto::AccToVecModeAttr>(unwrap(attr));
  return static_cast<int32_t>(a.getValue());
}

bool mlirPTOAttrIsATInsertModeAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::TInsertModeAttr>(unwrap(attr));
}

MlirAttribute mlirPTOTInsertModeAttrGet(MlirContext ctx, int32_t value) {
  auto *c = unwrap(ctx);
  return wrap(mlir::pto::TInsertModeAttr::get(
      c, static_cast<mlir::pto::TInsertMode>(value)));
}

int32_t mlirPTOTInsertModeAttrGetValue(MlirAttribute attr) {
  auto a = mlir::cast<mlir::pto::TInsertModeAttr>(unwrap(attr));
  return static_cast<int32_t>(a.getValue());
}

bool mlirPTOAttrIsAReluPreModeAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::ReluPreModeAttr>(unwrap(attr));
}

MlirAttribute mlirPTOReluPreModeAttrGet(MlirContext ctx, int32_t value) {
  auto *c = unwrap(ctx);
  return wrap(mlir::pto::ReluPreModeAttr::get(
      c, static_cast<mlir::pto::ReluPreMode>(value)));
}

int32_t mlirPTOReluPreModeAttrGetValue(MlirAttribute attr) {
  auto a = mlir::cast<mlir::pto::ReluPreModeAttr>(unwrap(attr));
  return static_cast<int32_t>(a.getValue());
}

bool mlirPTOAttrIsAAtomicTypeAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::AtomicTypeAttr>(unwrap(attr));
}

MlirAttribute mlirPTOAtomicTypeAttrGet(MlirContext ctx, int32_t value) {
  auto *c = unwrap(ctx);
  return wrap(mlir::pto::AtomicTypeAttr::get(
      c, static_cast<mlir::pto::AtomicType>(value)));
}

int32_t mlirPTOAtomicTypeAttrGetValue(MlirAttribute attr) {
  auto a = mlir::cast<mlir::pto::AtomicTypeAttr>(unwrap(attr));
  return static_cast<int32_t>(a.getValue());
}

bool mlirPTOAttrIsANotifyOpAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::NotifyOpAttr>(unwrap(attr));
}

MlirAttribute mlirPTONotifyOpAttrGet(MlirContext ctx, int32_t value) {
  auto *c = unwrap(ctx);
  return wrap(mlir::pto::NotifyOpAttr::get(
      c, static_cast<mlir::pto::NotifyOp>(value)));
}

int32_t mlirPTONotifyOpAttrGetValue(MlirAttribute attr) {
  auto a = mlir::cast<mlir::pto::NotifyOpAttr>(unwrap(attr));
  return static_cast<int32_t>(a.getValue());
}

bool mlirPTOAttrIsAWaitCmpAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::WaitCmpAttr>(unwrap(attr));
}

MlirAttribute mlirPTOWaitCmpAttrGet(MlirContext ctx, int32_t value) {
  auto *c = unwrap(ctx);
  return wrap(mlir::pto::WaitCmpAttr::get(
      c, static_cast<mlir::pto::WaitCmp>(value)));
}

int32_t mlirPTOWaitCmpAttrGetValue(MlirAttribute attr) {
  auto a = mlir::cast<mlir::pto::WaitCmpAttr>(unwrap(attr));
  return static_cast<int32_t>(a.getValue());
}

bool mlirPTOAttrIsAReduceOpAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::ReduceOpAttr>(unwrap(attr));
}

MlirAttribute mlirPTOReduceOpAttrGet(MlirContext ctx, int32_t value) {
  auto *c = unwrap(ctx);
  return wrap(mlir::pto::ReduceOpAttr::get(
      c, static_cast<mlir::pto::ReduceOp>(value)));
}

int32_t mlirPTOReduceOpAttrGetValue(MlirAttribute attr) {
  auto a = mlir::cast<mlir::pto::ReduceOpAttr>(unwrap(attr));
  return static_cast<int32_t>(a.getValue());
}

MlirAttribute mlirPTOTileBufConfigAttrGet(MlirContext ctx,
                                          MlirAttribute bLayout,
                                          MlirAttribute sLayout,
                                          MlirAttribute sFractalSize,
                                          MlirAttribute pad) {
  auto *c = unwrap(ctx);
  auto compactMode =
      wrap(mlir::pto::CompactModeAttr::get(c, mlir::pto::CompactMode::Null));
  return mlirPTOTileBufConfigAttrGetWithCompactMode(
      ctx, bLayout, sLayout, sFractalSize, pad, compactMode);
}

MlirAttribute mlirPTOTileBufConfigAttrGetWithCompactMode(
    MlirContext ctx, MlirAttribute bLayout, MlirAttribute sLayout,
    MlirAttribute sFractalSize, MlirAttribute pad, MlirAttribute compactMode) {
  auto *c = unwrap(ctx);
  auto blA = toBLayoutAttr(c, unwrap(bLayout));
  auto slA = toSLayoutAttr(c, unwrap(sLayout));
  auto pvA = toPadValueAttr(c, unwrap(pad));
  auto cmA = toCompactModeAttr(c, unwrap(compactMode));
  if (!blA || !slA || !pvA || !cmA) {
    return MlirAttribute{nullptr};
  }

  auto sz = mlir::dyn_cast<mlir::IntegerAttr>(unwrap(sFractalSize));
  if (!sz || !sz.getType().isInteger(kI32BitWidth)) {
    return MlirAttribute{nullptr};
  }

  return wrap(mlir::pto::TileBufConfigAttr::get(c, blA, slA, sz, pvA, cmA));
}

bool mlirPTOAttrIsAConvTileConfigAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::ConvTileConfigAttr>(unwrap(attr));
}

MlirAttribute mlirPTOConvTileConfigAttrGetDefault(MlirContext ctx) {
  auto *c = unwrap(ctx);
  return wrap(mlir::pto::ConvTileConfigAttr::getDefault(c));
}

MlirAttribute mlirPTOConvTileConfigAttrGet(
    MlirContext ctx, MlirAttribute fmapH, MlirAttribute fmapW,
    intptr_t padListSize, const int64_t *padList, MlirAttribute filterH,
    MlirAttribute filterW, MlirAttribute dilationH, MlirAttribute dilationW,
    MlirAttribute strideH, MlirAttribute strideW, MlirAttribute padValue,
    MlirAttribute channelSize, MlirAttribute repeatStride,
    MlirAttribute repeatTime, MlirAttribute repeatMode, MlirAttribute dstStride,
    MlirAttribute dstMposition, MlirAttribute transpose) {
  auto *c = unwrap(ctx);
  auto fmapHAttr = toI32IntegerAttr(unwrap(fmapH));
  auto fmapWAttr = toI32IntegerAttr(unwrap(fmapW));
  auto filterHAttr = toI32IntegerAttr(unwrap(filterH));
  auto filterWAttr = toI32IntegerAttr(unwrap(filterW));
  auto dilationHAttr = toI32IntegerAttr(unwrap(dilationH));
  auto dilationWAttr = toI32IntegerAttr(unwrap(dilationW));
  auto strideHAttr = toI32IntegerAttr(unwrap(strideH));
  auto strideWAttr = toI32IntegerAttr(unwrap(strideW));
  auto channelSizeAttr = toI32IntegerAttr(unwrap(channelSize));
  auto repeatStrideAttr = toI32IntegerAttr(unwrap(repeatStride));
  auto repeatTimeAttr = toI32IntegerAttr(unwrap(repeatTime));
  auto repeatModeAttr = toI32IntegerAttr(unwrap(repeatMode));
  auto dstStrideAttr = toI32IntegerAttr(unwrap(dstStride));
  auto dstMpositionAttr = toI32IntegerAttr(unwrap(dstMposition));
  auto transposeAttr = mlir::dyn_cast<mlir::BoolAttr>(unwrap(transpose));
  if (!fmapHAttr || !fmapWAttr || !filterHAttr || !filterWAttr ||
      !dilationHAttr || !dilationWAttr || !strideHAttr || !strideWAttr ||
      !channelSizeAttr || !repeatStrideAttr || !repeatTimeAttr ||
      !repeatModeAttr || !dstStrideAttr || !dstMpositionAttr ||
      !transposeAttr) {
    return MlirAttribute{nullptr};
  }

  auto pads =
      llvm::ArrayRef<int64_t>(padList, static_cast<size_t>(padListSize));
  // getChecked runs ConvTileConfigAttr::verify, so field ranges hold for
  // CAPI/Python-built configs too; a failure emits at the unknown location
  // and returns a null attribute.
  auto attr = mlir::pto::ConvTileConfigAttr::getChecked(
      [c]() { return mlir::emitError(mlir::UnknownLoc::get(c)); }, c, fmapHAttr,
      fmapWAttr, pads, filterHAttr, filterWAttr, dilationHAttr, dilationWAttr,
      strideHAttr, strideWAttr, unwrap(padValue), channelSizeAttr,
      repeatStrideAttr, repeatTimeAttr, repeatModeAttr, dstStrideAttr,
      dstMpositionAttr, transposeAttr);
  return wrap(attr);
}

MlirType mlirPTOGMTypeGet(MlirContext ctx, intptr_t rank, const int64_t *shape,
                          MlirType elementType) {
  auto *c = unwrap(ctx);
  auto elemTy = unwrap(elementType);
  llvm::ArrayRef<int64_t> shp(shape, static_cast<size_t>(rank));

  llvm::SmallVector<int64_t, kGMTypeStrideInlineCapacity> strides(
      static_cast<size_t>(rank), ShapedType::kDynamic);
  if (rank > 0) {
      strides[static_cast<size_t>(rank) - 1] = 1;
    }
  auto layout =
      StridedLayoutAttr::get(c, ShapedType::kDynamic, llvm::ArrayRef<int64_t>(strides));
  auto memSpace = mlir::pto::AddressSpaceAttr::get(c, mlir::pto::AddressSpace::GM);

  return wrap(MemRefType::get(shp, elemTy, layout, memSpace));
}

//===----------------------------------------------------------------------===//
// !pto.vreg<count x elem>
//===----------------------------------------------------------------------===//

bool mlirPTOTypeIsAVRegType(MlirType type) {
  return isa<mlir::pto::VRegType>(unwrap(type));
}

MlirType mlirPTOVRegTypeGet(MlirContext ctx, int64_t elementCount,
                            MlirType elementType) {
  return wrap(mlir::pto::VRegType::get(unwrap(ctx), elementCount,
                                       unwrap(elementType)));
}

int64_t mlirPTOVRegTypeGetElementCount(MlirType type) {
  return cast<mlir::pto::VRegType>(unwrap(type)).getElementCount();
}

MlirType mlirPTOVRegTypeGetElementType(MlirType type) {
  return wrap(cast<mlir::pto::VRegType>(unwrap(type)).getElementType());
}

//===----------------------------------------------------------------------===//
// !pto.mask<granularity>
//===----------------------------------------------------------------------===//

bool mlirPTOTypeIsAMaskType(MlirType type) {
  return isa<mlir::pto::MaskType>(unwrap(type));
}

MlirType mlirPTOMaskTypeGet(MlirContext ctx, MlirStringRef granularity) {
  return wrap(mlir::pto::MaskType::get(unwrap(ctx), unwrap(granularity)));
}

MlirStringRef mlirPTOMaskTypeGetGranularity(MlirType type) {
  return wrap(cast<mlir::pto::MaskType>(unwrap(type)).getGranularity());
}

//===----------------------------------------------------------------------===//
// !pto.vmivreg<count x elem, layout?>
//===----------------------------------------------------------------------===//

bool mlirPTOTypeIsAVMIVRegType(MlirType type) {
  return isa<mlir::pto::VMIVRegType>(unwrap(type));
}

MlirType mlirPTOVMIVRegTypeGet(MlirContext ctx, int64_t elementCount,
                               MlirType elementType, MlirAttribute layout) {
  return wrap(mlir::pto::VMIVRegType::get(unwrap(ctx), elementCount,
                                          unwrap(elementType), unwrap(layout)));
}

int64_t mlirPTOVMIVRegTypeGetElementCount(MlirType type) {
  return cast<mlir::pto::VMIVRegType>(unwrap(type)).getElementCount();
}

MlirType mlirPTOVMIVRegTypeGetElementType(MlirType type) {
  return wrap(cast<mlir::pto::VMIVRegType>(unwrap(type)).getElementType());
}

MlirAttribute mlirPTOVMIVRegTypeGetLayout(MlirType type) {
  return wrap(cast<mlir::pto::VMIVRegType>(unwrap(type)).getLayout());
}

//===----------------------------------------------------------------------===//
// !pto.vmimask<count x granularity, layout?>
//===----------------------------------------------------------------------===//

bool mlirPTOTypeIsAVMIMaskType(MlirType type) {
  return isa<mlir::pto::VMIMaskType>(unwrap(type));
}

MlirType mlirPTOVMIMaskTypeGet(MlirContext ctx, int64_t elementCount,
                               MlirStringRef granularity, MlirAttribute layout) {
  return wrap(mlir::pto::VMIMaskType::get(unwrap(ctx), elementCount,
                                          unwrap(granularity), unwrap(layout)));
}

int64_t mlirPTOVMIMaskTypeGetElementCount(MlirType type) {
  return cast<mlir::pto::VMIMaskType>(unwrap(type)).getElementCount();
}

MlirStringRef mlirPTOVMIMaskTypeGetGranularity(MlirType type) {
  return wrap(cast<mlir::pto::VMIMaskType>(unwrap(type)).getGranularity());
}

MlirAttribute mlirPTOVMIMaskTypeGetLayout(MlirType type) {
  return wrap(cast<mlir::pto::VMIMaskType>(unwrap(type)).getLayout());
}

//===----------------------------------------------------------------------===//
// !pto.align
//===----------------------------------------------------------------------===//

bool mlirPTOTypeIsAAlignType(MlirType type) {
  return isa<mlir::pto::AlignType>(unwrap(type));
}

MlirType mlirPTOAlignTypeGet(MlirContext ctx) {
  return wrap(mlir::pto::AlignType::get(unwrap(ctx)));
}

//===----------------------------------------------------------------------===//
// !pto.struct<fields...>
//===----------------------------------------------------------------------===//

bool mlirPTOTypeIsAStructType(MlirType type) {
  return isa<mlir::pto::StructType>(unwrap(type));
}

MlirType mlirPTOStructTypeGet(MlirContext ctx, intptr_t numFieldTypes,
                              MlirType const *fieldTypes) {
  MLIRContext *c = unwrap(ctx);
  llvm::SmallVector<mlir::Type> fields;
  fields.reserve(static_cast<size_t>(numFieldTypes));
  for (intptr_t i = 0; i < numFieldTypes; ++i) {
    fields.push_back(unwrap(fieldTypes[i]));
  }
  auto structType = mlir::pto::StructType::getChecked(
      [c]() { return mlir::emitError(mlir::UnknownLoc::get(c)); }, c,
      llvm::ArrayRef<mlir::Type>(fields));
  return wrap(structType);
}

intptr_t mlirPTOStructTypeGetNumFieldTypes(MlirType type) {
  return static_cast<intptr_t>(
      cast<mlir::pto::StructType>(unwrap(type)).getFieldTypes().size());
}

MlirType mlirPTOStructTypeGetFieldType(MlirType type, intptr_t index) {
  return wrap(cast<mlir::pto::StructType>(unwrap(type))
                  .getFieldTypes()[static_cast<size_t>(index)]);
}

//===----------------------------------------------------------------------===//
// TileBufType getters
//===----------------------------------------------------------------------===//

intptr_t mlirPTOTileBufTypeGetRank(MlirType type) {
  return static_cast<intptr_t>(
      cast<mlir::pto::TileBufType>(unwrap(type)).getRank());
}

MlirType mlirPTOTileBufTypeGetElementType(MlirType type) {
  return wrap(cast<mlir::pto::TileBufType>(unwrap(type)).getElementType());
}

MlirAttribute mlirPTOTileBufTypeGetMemorySpace(MlirType type) {
  return wrap(cast<mlir::pto::TileBufType>(unwrap(type)).getMemorySpace());
}

const int64_t *mlirPTOTileBufTypeGetShape(MlirType type, intptr_t *numDimsOut) {
  auto shape = cast<mlir::pto::TileBufType>(unwrap(type)).getShape();
  *numDimsOut = static_cast<intptr_t>(shape.size());
  return shape.data();
}

const int64_t *mlirPTOTileBufTypeGetValidShape(MlirType type,
                                               intptr_t *numDimsOut) {
  auto validShape = cast<mlir::pto::TileBufType>(unwrap(type)).getValidShape();
  *numDimsOut = static_cast<intptr_t>(validShape.size());
  return validShape.data();
}

MlirAttribute mlirPTOTileBufTypeGetBLayoutAttr(MlirType type) {
  return wrap(cast<mlir::pto::TileBufType>(unwrap(type)).getBLayoutAttr());
}

MlirAttribute mlirPTOTileBufTypeGetSLayoutAttr(MlirType type) {
  return wrap(cast<mlir::pto::TileBufType>(unwrap(type)).getSLayoutAttr());
}

int32_t mlirPTOTileBufTypeGetBLayoutValue(MlirType type) {
  return cast<mlir::pto::TileBufType>(unwrap(type)).getBLayoutValueI32();
}

int32_t mlirPTOTileBufTypeGetSLayoutValue(MlirType type) {
  return cast<mlir::pto::TileBufType>(unwrap(type)).getSLayoutValueI32();
}

int32_t mlirPTOTileBufTypeGetPadValue(MlirType type) {
  return cast<mlir::pto::TileBufType>(unwrap(type)).getPadValueI32();
}

int32_t mlirPTOTileBufTypeGetCompactMode(MlirType type) {
  return cast<mlir::pto::TileBufType>(unwrap(type)).getCompactModeI32();
}

int32_t mlirPTOTileBufTypeGetSFractalSize(MlirType type) {
  return cast<mlir::pto::TileBufType>(unwrap(type)).getSFractalSizeI32();
}

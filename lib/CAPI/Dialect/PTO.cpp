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

#include "pto-c/Dialect/PTO.h"

// unwrap/wrap + MLIR dialect registration C-API support.
#include "mlir/CAPI/IR.h"

#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"
#include "llvm/ADT/SmallVector.h"

// IMPORTANT: include the C++ dialect header that declares PtrType/TensorViewType.
// This header should itself include the generated PTOTypeDefs.h.inc.
#include "PTO/IR/PTO.h"

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

bool mlirPTOTypeIsAAsyncEventType(MlirType type) {
  return isa<mlir::pto::AsyncEventType>(unwrap(type));
}

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

bool mlirPTOAttrIsABLayoutAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::BLayoutAttr>(unwrap(attr));
}

MlirAttribute mlirPTOBLayoutAttrGet(MlirContext ctx, int32_t value) {
  auto *c = unwrap(ctx);
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
      [&]() { return mlir::emitError(mlir::UnknownLoc::get(c)); }, c,
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

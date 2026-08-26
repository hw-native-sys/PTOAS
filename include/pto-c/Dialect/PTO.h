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

#ifndef MLIR_C_DIALECT_PTO_H
#define MLIR_C_DIALECT_PTO_H

#include "mlir-c/IR.h" 

#ifdef __cplusplus
extern "C" {
#endif

// Provides: mlirGetDialectHandle__pto__()
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(PTO, pto);

// ---- !pto.ptr<elem> ----
MLIR_CAPI_EXPORTED bool mlirPTOTypeIsAPtrType(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirPTOPtrTypeGet(MlirContext ctx, MlirType elementType);
MLIR_CAPI_EXPORTED MlirType mlirPTOPtrTypeGetWithMemorySpace(MlirContext ctx, MlirType elementType,
                                          MlirAttribute memorySpace);
MLIR_CAPI_EXPORTED MlirType mlirPTOPtrTypeGetElementType(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOPtrTypeGetMemorySpace(MlirType type);

// ---- !pto.async_session / !pto.async_event ----
MLIR_CAPI_EXPORTED bool mlirPTOTypeIsAAsyncSessionType(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirPTOAsyncSessionTypeGet(MlirContext ctx);
MLIR_CAPI_EXPORTED bool mlirPTOTypeIsAAsyncEventType(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirPTOAsyncEventTypeGet(MlirContext ctx);
MLIR_CAPI_EXPORTED bool mlirPTOTypeIsAPrefetchAsyncContextType(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirPTOPrefetchAsyncContextTypeGet(MlirContext ctx);

// ---- !pto.hif8 / !pto.f8E8M0 / !pto.hif8x2 / !pto.f4E1M2x2 / !pto.f4E2M1x2 ----
MLIR_CAPI_EXPORTED bool mlirPTOTypeIsAHiF8Type(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirPTOHiF8TypeGet(MlirContext ctx);
MLIR_CAPI_EXPORTED bool mlirPTOTypeIsAF8E8M0Type(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirPTOF8E8M0TypeGet(MlirContext ctx);
MLIR_CAPI_EXPORTED bool mlirPTOTypeIsAHiF8x2Type(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirPTOHiF8x2TypeGet(MlirContext ctx);
MLIR_CAPI_EXPORTED bool mlirPTOTypeIsAF4E1M2x2Type(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirPTOF4E1M2x2TypeGet(MlirContext ctx);
MLIR_CAPI_EXPORTED bool mlirPTOTypeIsAF4E2M1x2Type(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirPTOF4E2M1x2TypeGet(MlirContext ctx);
MLIR_CAPI_EXPORTED bool mlirPTOTypeIsABF16x2Type(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirPTOBF16x2TypeGet(MlirContext ctx);

// ---- #pto.address_space<...> ----
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsAAddressSpaceAttr(MlirAttribute attr);

// Create: #pto.address_space<ub/gm/...>
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOAddressSpaceAttrGet(MlirContext ctx, int32_t value);

// Read back enum value (0..6)
MLIR_CAPI_EXPORTED int32_t mlirPTOAddressSpaceAttrGetValue(MlirAttribute attr);

// ---- #pto.load_cache_policy<...> ----
MLIR_CAPI_EXPORTED bool
mlirPTOAttrIsALoadCachePolicyAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirPTOLoadCachePolicyAttrGet(MlirContext ctx, int32_t value);
MLIR_CAPI_EXPORTED int32_t
mlirPTOLoadCachePolicyAttrGetValue(MlirAttribute attr);

// ---- !pto.tensor_view<shape x elem> ----
MLIR_CAPI_EXPORTED bool mlirPTOTypeIsATensorViewType(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirPTOTensorViewTypeGet(MlirContext ctx, intptr_t rank,
                                  const int64_t *shape, MlirType elementType);
MLIR_CAPI_EXPORTED intptr_t mlirPTOTensorViewTypeGetRank(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirPTOTensorViewTypeGetElementType(MlirType type);
// 返回内部 shape 数组指针（只读）；numDimsOut 返回维度数
MLIR_CAPI_EXPORTED const int64_t *mlirPTOTensorViewTypeGetShape(MlirType type, intptr_t *numDimsOut);

// ---- !pto.partition_tensor_view<shape x elem> ----
MLIR_CAPI_EXPORTED bool mlirPTOTypeIsAPartitionTensorViewType(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirPTOPartitionTensorViewTypeGet(MlirContext ctx, intptr_t rank,
                                           const int64_t *shape, MlirType elementType);
MLIR_CAPI_EXPORTED intptr_t mlirPTOPartitionTensorViewTypeGetRank(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirPTOPartitionTensorViewTypeGetElementType(MlirType type);
// 返回内部 shape 数组指针（只读）；numDimsOut 返回维度数
MLIR_CAPI_EXPORTED const int64_t *mlirPTOPartitionTensorViewTypeGetShape(MlirType type, intptr_t *numDimsOut);

// ---- !pto.tile<shape x elem> ----
MLIR_CAPI_EXPORTED bool mlirPTOTypeIsATileType(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirPTOTileTypeGet(MlirContext ctx, intptr_t rank,
                            const int64_t *shape, MlirType elementType);
MLIR_CAPI_EXPORTED intptr_t mlirPTOTileTypeGetRank(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirPTOTileTypeGetElementType(MlirType type);
MLIR_CAPI_EXPORTED const int64_t *mlirPTOTileTypeGetShape(MlirType type, intptr_t *numDimsOut);

// ---- !pto.vreg<count x elem> ----
MLIR_CAPI_EXPORTED bool     mlirPTOTypeIsAVRegType(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirPTOVRegTypeGet(MlirContext ctx, int64_t elementCount, MlirType elementType);
MLIR_CAPI_EXPORTED int64_t  mlirPTOVRegTypeGetElementCount(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirPTOVRegTypeGetElementType(MlirType type);

// ---- !pto.mask<granularity> ----
MLIR_CAPI_EXPORTED bool          mlirPTOTypeIsAMaskType(MlirType type);
MLIR_CAPI_EXPORTED MlirType      mlirPTOMaskTypeGet(MlirContext ctx, MlirStringRef granularity);
MLIR_CAPI_EXPORTED MlirStringRef mlirPTOMaskTypeGetGranularity(MlirType type);

// ---- !pto.vmivreg<count x elem, layout?> ----
MLIR_CAPI_EXPORTED bool          mlirPTOTypeIsAVMIVRegType(MlirType type);
MLIR_CAPI_EXPORTED MlirType      mlirPTOVMIVRegTypeGet(MlirContext ctx, int64_t elementCount,
                                    MlirType elementType, MlirAttribute layout /*may be null*/);
MLIR_CAPI_EXPORTED int64_t       mlirPTOVMIVRegTypeGetElementCount(MlirType type);
MLIR_CAPI_EXPORTED MlirType      mlirPTOVMIVRegTypeGetElementType(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOVMIVRegTypeGetLayout(MlirType type); // null attr if absent

// ---- !pto.vmimask<count x granularity, layout?> ----
MLIR_CAPI_EXPORTED bool          mlirPTOTypeIsAVMIMaskType(MlirType type);
MLIR_CAPI_EXPORTED MlirType      mlirPTOVMIMaskTypeGet(MlirContext ctx, int64_t elementCount,
                                    MlirStringRef granularity, MlirAttribute layout /*may be null*/);
MLIR_CAPI_EXPORTED int64_t       mlirPTOVMIMaskTypeGetElementCount(MlirType type);
MLIR_CAPI_EXPORTED MlirStringRef mlirPTOVMIMaskTypeGetGranularity(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOVMIMaskTypeGetLayout(MlirType type); // null attr if absent

// ---- !pto.align ----
MLIR_CAPI_EXPORTED bool     mlirPTOTypeIsAAlignType(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirPTOAlignTypeGet(MlirContext ctx);

// ---- !pto.struct<fields...> ----
MLIR_CAPI_EXPORTED bool     mlirPTOTypeIsAStructType(MlirType type);
// Null type (mlirTypeIsNull) if field types are invalid; emits a diagnostic
// on an UnknownLoc, mirroring StructType::getChecked.
MLIR_CAPI_EXPORTED MlirType mlirPTOStructTypeGet(MlirContext ctx, intptr_t numFieldTypes,
                              MlirType const *fieldTypes);
MLIR_CAPI_EXPORTED intptr_t mlirPTOStructTypeGetNumFieldTypes(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirPTOStructTypeGetFieldType(MlirType type, intptr_t index);

// ---- TileBufType getters（Get 已有 CAPI，仅补属性读取） ----
MLIR_CAPI_EXPORTED intptr_t       mlirPTOTileBufTypeGetRank(MlirType type);
MLIR_CAPI_EXPORTED MlirType       mlirPTOTileBufTypeGetElementType(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute  mlirPTOTileBufTypeGetMemorySpace(MlirType type);
MLIR_CAPI_EXPORTED const int64_t *mlirPTOTileBufTypeGetShape(MlirType type, intptr_t *numDimsOut);
MLIR_CAPI_EXPORTED const int64_t *mlirPTOTileBufTypeGetValidShape(MlirType type, intptr_t *numDimsOut);
MLIR_CAPI_EXPORTED MlirAttribute  mlirPTOTileBufTypeGetBLayoutAttr(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute  mlirPTOTileBufTypeGetSLayoutAttr(MlirType type);
MLIR_CAPI_EXPORTED int32_t        mlirPTOTileBufTypeGetBLayoutValue(MlirType type);
MLIR_CAPI_EXPORTED int32_t        mlirPTOTileBufTypeGetSLayoutValue(MlirType type);
MLIR_CAPI_EXPORTED int32_t        mlirPTOTileBufTypeGetPadValue(MlirType type);
MLIR_CAPI_EXPORTED int32_t        mlirPTOTileBufTypeGetCompactMode(MlirType type);
MLIR_CAPI_EXPORTED int32_t        mlirPTOTileBufTypeGetSFractalSize(MlirType type);

// ---- TileBufType ----
MLIR_CAPI_EXPORTED bool mlirPTOTypeIsATileBufType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirPTOTileBufTypeGet(
    MlirContext ctx, intptr_t rank, const int64_t *shape,
    MlirType elementType, MlirAttribute memorySpace);

MLIR_CAPI_EXPORTED MlirType mlirPTOTileBufTypeGetWithConfig(
    MlirContext ctx, intptr_t rank, const int64_t *shape,
    MlirType elementType, MlirAttribute memorySpace, MlirAttribute config);
// ---- Enum attrs helpers (BLayout/SLayout/PadValue in mlir::pto) ----
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsABLayoutAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOBLayoutAttrGet(MlirContext ctx, int32_t value);
MLIR_CAPI_EXPORTED int32_t mlirPTOBLayoutAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsASLayoutAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOSLayoutAttrGet(MlirContext ctx, int32_t value);
MLIR_CAPI_EXPORTED int32_t mlirPTOSLayoutAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsAPadValueAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOPadValueAttrGet(MlirContext ctx, int32_t value);
MLIR_CAPI_EXPORTED int32_t mlirPTOPadValueAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsACompactModeAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOCompactModeAttrGet(MlirContext ctx, int32_t value);
MLIR_CAPI_EXPORTED int32_t mlirPTOCompactModeAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsAAccToVecModeAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOAccToVecModeAttrGet(MlirContext ctx, int32_t value);
MLIR_CAPI_EXPORTED int32_t mlirPTOAccToVecModeAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsATInsertModeAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOTInsertModeAttrGet(MlirContext ctx, int32_t value);
MLIR_CAPI_EXPORTED int32_t mlirPTOTInsertModeAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsAReluPreModeAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOReluPreModeAttrGet(MlirContext ctx, int32_t value);
MLIR_CAPI_EXPORTED int32_t mlirPTOReluPreModeAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsAAtomicTypeAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOAtomicTypeAttrGet(MlirContext ctx, int32_t value);
MLIR_CAPI_EXPORTED int32_t mlirPTOAtomicTypeAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsANotifyOpAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirPTONotifyOpAttrGet(MlirContext ctx, int32_t value);
MLIR_CAPI_EXPORTED int32_t mlirPTONotifyOpAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsAWaitCmpAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOWaitCmpAttrGet(MlirContext ctx, int32_t value);
MLIR_CAPI_EXPORTED int32_t mlirPTOWaitCmpAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsAReduceOpAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOReduceOpAttrGet(MlirContext ctx, int32_t value);
MLIR_CAPI_EXPORTED int32_t mlirPTOReduceOpAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsAFenceScopeAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOFenceScopeAttrGet(MlirContext ctx, int32_t value);
MLIR_CAPI_EXPORTED int32_t mlirPTOFenceScopeAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirPTORoundModeAttrGet(MlirContext ctx, int32_t value);
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsARoundModeAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED int32_t mlirPTORoundModeAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirPTODivPrecisionAttrGet(MlirContext ctx, int32_t value);
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsADivPrecisionAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED int32_t mlirPTODivPrecisionAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOExpPrecisionAttrGet(MlirContext ctx, int32_t value);
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsAExpPrecisionAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED int32_t mlirPTOExpPrecisionAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOLogPrecisionAttrGet(MlirContext ctx, int32_t value);
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsALogPrecisionAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED int32_t mlirPTOLogPrecisionAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirPTORecipPrecisionAttrGet(MlirContext ctx, int32_t value);
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsARecipPrecisionAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED int32_t mlirPTORecipPrecisionAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirPTORemPrecisionAttrGet(MlirContext ctx, int32_t value);
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsARemPrecisionAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED int32_t mlirPTORemPrecisionAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirPTORsqrtPrecisionAttrGet(MlirContext ctx, int32_t value);
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsARsqrtPrecisionAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED int32_t mlirPTORsqrtPrecisionAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOSqrtPrecisionAttrGet(MlirContext ctx, int32_t value);
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsASqrtPrecisionAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED int32_t mlirPTOSqrtPrecisionAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOFmodPrecisionAttrGet(MlirContext ctx, int32_t value);
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsAFmodPrecisionAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED int32_t mlirPTOFmodPrecisionAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOSaturationModeAttrGet(MlirContext ctx, int32_t value);
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsASaturationModeAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED int32_t mlirPTOSaturationModeAttrGetValue(MlirAttribute attr);
// ---- Pipe attr ----
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOPipeAttrGet(MlirContext ctx, int32_t value);
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsAPipeAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED int32_t mlirPTOPipeAttrGetValue(MlirAttribute attr);
// ---- Layout attr ----
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOLayoutAttrGet(MlirContext ctx, int32_t value);
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsALayoutAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED int32_t mlirPTOLayoutAttrGetValue(MlirAttribute attr);
// ---- SyncOpType attr ----
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOSyncOpTypeAttrGet(MlirContext ctx, int32_t value);
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsASyncOpTypeAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED int32_t mlirPTOSyncOpTypeAttrGetValue(MlirAttribute attr);
// ---- Event attr ----
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOEventAttrGet(MlirContext ctx, int32_t value);
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsAEventAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED int32_t mlirPTOEventAttrGetValue(MlirAttribute attr);
// ---- MaskPattern attr ----
// Backward-compatible int entry point that accepts only unambiguous values
// {0, 3, 6, 7}. It rejects ambiguous raw ints {1, 2, 4, 5} so callers must
// choose either the ISA-aligned enum API below or the explicit legacy-raw
// compatibility API.
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOMaskPatternAttrGet(MlirContext ctx, int32_t value);
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsAMaskPatternAttr(MlirAttribute attr);
// Returns the ISA-aligned numeric value {1..7}.
MLIR_CAPI_EXPORTED int32_t mlirPTOMaskPatternAttrGetValue(MlirAttribute attr);
typedef enum MlirPTOMaskPattern {
  MlirPTOMaskPattern_P0101 = 1,
  MlirPTOMaskPattern_P1010 = 2,
  MlirPTOMaskPattern_P0001 = 3,
  MlirPTOMaskPattern_P0010 = 4,
  MlirPTOMaskPattern_P0100 = 5,
  MlirPTOMaskPattern_P1000 = 6,
  MlirPTOMaskPattern_P1111 = 7,
} MlirPTOMaskPattern;
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOMaskPatternAttrGetEnum(MlirContext ctx, MlirPTOMaskPattern value);
MLIR_CAPI_EXPORTED MlirPTOMaskPattern mlirPTOMaskPatternAttrGetEnumValue(MlirAttribute attr);
// Legacy raw-int compatibility path for historical PTOAS encodings:
//   0 -> P0101, 3 -> P0001, 4 -> P1111, 5 -> P1010.
// Removed legacy-only patterns 1/2 are rejected and return null.
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOMaskPatternAttrGetLegacyRaw(MlirContext ctx, int32_t value);

// ---- CmpMode (compare mode for cmp/cvt) ----
typedef enum MlirPTOCmpMode {
  MlirPTOCmpMode_EQ = 0,
  MlirPTOCmpMode_NE = 1,
  MlirPTOCmpMode_LT = 2,
  MlirPTOCmpMode_LE = 3,
  MlirPTOCmpMode_GT = 4,
  MlirPTOCmpMode_GE = 5,
} MlirPTOCmpMode;
MLIR_CAPI_EXPORTED bool mlirAttributeIsAPTOCmpModeAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOCmpModeAttrGet(MlirContext ctx, MlirPTOCmpMode value);
MLIR_CAPI_EXPORTED MlirPTOCmpMode mlirPTOCmpModeAttrGetValue(MlirAttribute attr);
typedef enum MlirPTOCoalesce {
  MlirPTOCoalesce_Row = 0,
  MlirPTOCoalesce_Elem = 1,
} MlirPTOCoalesce;
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsACoalesceAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOCoalesceAttrGet(MlirContext ctx, MlirPTOCoalesce value);
MLIR_CAPI_EXPORTED MlirPTOCoalesce mlirPTOCoalesceAttrGetValue(MlirAttribute attr);
// ---- TileBufConfigAttr ----
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsATileBufConfigAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute mlirPTOTileBufConfigAttrGetDefault(MlirContext ctx);

MLIR_CAPI_EXPORTED MlirAttribute mlirPTOTileBufConfigAttrGet(
    MlirContext ctx,
    MlirAttribute bLayout, MlirAttribute sLayout,
    MlirAttribute sFractalSize, MlirAttribute pad);
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOTileBufConfigAttrGetWithCompactMode(
    MlirContext ctx,
    MlirAttribute bLayout, MlirAttribute sLayout,
    MlirAttribute sFractalSize, MlirAttribute pad,
    MlirAttribute compactMode);
MLIR_CAPI_EXPORTED MlirType mlirPTOTileBufTypeGetWithValidShape(
    MlirContext ctx, intptr_t rank, const int64_t *shape, MlirType elementType,
    MlirAttribute memorySpace, intptr_t validRank, const int64_t *validShape);

MLIR_CAPI_EXPORTED MlirType mlirPTOTileBufTypeGetWithValidShapeAndConfig(
    MlirContext ctx, intptr_t rank, const int64_t *shape, MlirType elementType,
    MlirAttribute memorySpace, intptr_t validRank, const int64_t *validShape,
    MlirAttribute config);

// ---- QuantType attr ----
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOQuantTypeAttrGet(MlirContext ctx, int32_t value);
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsAQuantTypeAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED int32_t mlirPTOQuantTypeAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOQuantScaleAlgAttrGet(MlirContext ctx, int32_t value);
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsAQuantScaleAlgAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED int32_t mlirPTOQuantScaleAlgAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOMxGroupAxisAttrGet(MlirContext ctx, int32_t value);
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsAMxGroupAxisAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED int32_t mlirPTOMxGroupAxisAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOVecStoreModeAttrGet(MlirContext ctx, int32_t value);
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsAVecStoreModeAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED int32_t mlirPTOVecStoreModeAttrGetValue(MlirAttribute attr);

// ---- MemRef helpers ----
MLIR_CAPI_EXPORTED MlirType mlirPTOGMTypeGet(
    MlirContext ctx, intptr_t rank, const int64_t *shape, MlirType elementType);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_DIALECT_PTO_H

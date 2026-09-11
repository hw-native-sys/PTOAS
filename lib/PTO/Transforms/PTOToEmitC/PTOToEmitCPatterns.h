// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOToEmitCPatterns.h - pattern decls for PTO->EmitC ----------------===//
//===----------------------------------------------------------------------===//
//
// Pattern forward declarations and sync-token extraction templates shared by
// the PTOToEmitC*.cpp translation units. Split out of PTOToEmitCEmitters.h to
// keep each header under the 500-nbnc limit.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "PTOToEmitCEmitters.h"

namespace mlir {
namespace pto {

//---- conversion pattern forward declarations -----------------------------//
template <typename ArithOp, bool isMaximum>
struct ArithMinMaxFPropagateNaNToEmitC;
template <typename ArithOp, bool isUnsigned>
struct ArithMulExtendedToEmitC;
template <typename FenceOp>
struct PTOFenceToEmitC;
template <typename OpTy, bool IsStride>
struct PTOGetTensorViewMetadataToEmitC;
struct ArithFloatMinMaxToEmitCBase;

struct AffineApplyMulConstToEmitC;
struct ArithAddIToEmitC;
struct ArithAddUIExtendedToEmitC;
struct ArithBitcastToEmitC;
struct ArithCastOPToEmitC;
template <typename CastOp>
struct ArithCastToEmitC;
struct ArithCeilDivSIToEmitC;
struct ArithCeilDivUIToEmitC;
struct ArithCmpFToEmitC;
class ArithCmpIToEmitC;
struct ArithConstantToEmitC;
struct ArithDivSIToEmitC;
struct ArithDivUIToEmitC;
struct ArithExtSIToEmitC;
struct ArithExtUIToEmitC;
struct ArithFPToUIToEmitC;
struct ArithFloorDivSIToEmitC;
struct ArithIndexCastUIToEmitC;
struct ArithMaxNumFToEmitC;
struct ArithMaxSIToEmitC;
struct ArithMaxUIToEmitC;
using ArithMaximumFToEmitC = ArithMinMaxFPropagateNaNToEmitC<arith::MaximumFOp, /*isMaximum=*/true>;
struct ArithMinNumFToEmitC;
struct ArithMinSIToEmitC;
struct ArithMinUIToEmitC;
using ArithMinimumFToEmitC = ArithMinMaxFPropagateNaNToEmitC<arith::MinimumFOp, /*isMaximum=*/false>;
struct ArithMulIToEmitC;
using ArithMulSIExtendedToEmitC = ArithMulExtendedToEmitC<arith::MulSIExtendedOp, /*isUnsigned=*/false>;
using ArithMulUIExtendedToEmitC = ArithMulExtendedToEmitC<arith::MulUIExtendedOp, /*isUnsigned=*/true>;
struct ArithNegFToEmitC;
struct ArithRemFToEmitC;
struct ArithRemSIToEmitC;
struct ArithRemUIToEmitC;
struct ArithSelectToEmitC;
struct ArithShiftLeftToEmitC;
struct ArithShiftRightSIToEmitC;
struct ArithShiftRightUIToEmitC;
template <typename ArithOp, typename EmitCOp>
struct ArithSimpleBinaryToEmitC;
struct ArithSubIToEmitC;
struct ArithTruncIToEmitC;
struct ArithUIToFPToEmitC;
template <typename ArithOp, typename EmitCOp>
struct ArithUnsignedBitwiseBinaryToEmitC;
struct CallToEmitC;
struct CastPtrConversion;
struct FuncToEmitC;
struct MemRefCastToEmitC;
struct PTOAddSCToTADDSC;
struct PTOAddSToTADDS;
struct PTOAllocTileToEmitC;
struct PTOAndSToEmitC;
template <typename AsyncEventOp>
struct PTOAsyncEventToEmitC;
template <typename AsyncOp>
struct PTOAsyncTransferToEmitC;
struct PTOBitcastToEmitC;
struct PTOBuildAsyncSessionToEmitC;
struct PTOCmpSToEmitC;
struct PTOCmpToEmitC;
struct PTOColArgMaxToEmitC;
struct PTOColArgMinToEmitC;
struct PTOColExpandAddToEmitC;
struct PTOColExpandDivToEmitC;
struct PTOColExpandExpdifToEmitC;
struct PTOColExpandMaxToEmitC;
struct PTOColExpandMinToEmitC;
struct PTOColExpandMulToEmitC;
struct PTOColExpandSubToEmitC;
struct PTOColExpandToEmitC;
struct PTOColMaxToEmitC;
struct PTOColMinToEmitC;
struct PTOColProdToEmitC;
struct PTOColSumToEmitC;
template <typename CollectiveOp>
struct PTOCommCollectiveToEmitC;
struct PTOConcatToEmitC;
template <typename CrossOp, typename SyncOp>
struct PTOCrossSyncToSync;
struct PTOCvtToEmitC;
struct PTODeclareEventIdArrayToEmitC;
struct PTODeclareGlobalToEmitC;
struct PTODeclareLocalArrayToEmitC;
struct PTODeclareStructToEmitC;
struct PTODeclareTileToEmitC;
struct PTODequantToEmitC;
struct PTODivToTDIV;
struct PTOEventIdArrayGetToEmitC;
struct PTOEventIdArraySetToEmitC;
struct PTOExpToEmitC;
struct PTOExpandsToEmitC;
struct PTOExtractToEmitC;
struct PTOFModSToEmitC;
struct PTOFModToEmitC;
struct PTOFillPadToEmitC;
struct PTOGatherToEmitC;
struct PTOGatherbToEmitC;
struct PTOGetBlockIdxToEmitC;
struct PTOGetBlockNumToEmitC;
struct PTOGetBufDynToEmitC;
struct PTOGetBufToEmitC;
struct PTOGetPrefetchAsyncSessionToEmitC;
struct PTOGetSubBlockIdxToEmitC;
struct PTOGetSubBlockNumToEmitC;
struct PTOInitializeL2G2LPipeToEmitC;
struct PTOInitializeL2LPipeToEmitC;
struct PTOLReluToEmitC;
struct PTOLocalArrayGetToEmitC;
struct PTOLocalArraySetToEmitC;
struct PTOLogToEmitC;
struct PTOMGatherToMGATHER;
struct PTOMScatterToMSCATTER;
struct PTOMakePrefetchAsyncContextToEmitC;
struct PTOMakeTensorViewToEmitC;
struct PTOMaxSToEmitC;
struct PTOMaxToEmitC;
struct PTOMinToEmitC;
struct PTOMinsToEmitC;
struct PTOMovToEmitC;
struct PTOMrgSortToEmitC;
struct PTOMulToEmitC;
struct PTOMulsToEmitC;
template <typename SyncOp>
struct PTONamedIntraSyncToEmitC;
struct PTONegToEmitC;
struct PTONotToEmitC;
struct PTOOrToEmitC;
struct PTOOrsToEmitC;
template <typename OpTy>
struct PTOP2PCommToEmitC;
struct PTOPartAddToEmitC;
struct PTOPartArgMaxToEmitC;
struct PTOPartMaxToEmitC;
struct PTOPartMinToEmitC;
struct PTOPartMulToEmitC;
struct PTOPartitionViewStaticToEmitC;
struct PTOPowSToEmitC;
struct PTOPowToEmitC;
struct PTOPreluToEmitC;
struct PTOPrintOpToEmitC;
struct PTOPrintToTPRINT;
struct PTOQuantToEmitC;
struct PTORandomToEmitC;
struct PTORecipToEmitC;
struct PTOReluToEmitC;
struct PTORemSToEmitC;
struct PTORemToEmitC;
struct PTORlsBufDynToEmitC;
struct PTORlsBufToEmitC;
struct PTORowArgMaxToEmitC;
struct PTORowArgMinToEmitC;
struct PTORowExpandAddToEmitC;
struct PTORowExpandDivToEmitC;
struct PTORowExpandExpdifToEmitC;
struct PTORowExpandMaxToEmitC;
struct PTORowExpandMinToEmitC;
struct PTORowExpandMulToEmitC;
struct PTORowExpandSubToEmitC;
struct PTORowExpandToEmitC;
struct PTORowMaxToEmitC;
struct PTORowMinToEmitC;
struct PTORowProdToEmitC;
struct PTORowSumToEmitC;
struct PTORsqrtToEmitC;
struct PTOSORT32SToEmitC;
struct PTOScatterToEmitC;
struct PTOSelSToEmitC;
struct PTOSelToEmitC;
struct PTOSetFFTsToEmitC;
template <typename FlagOp>
struct PTOFlagToEmitC;
struct PTOSetQuantScalarToEmitC;
struct PTOSetValToSETVAL;
struct PTOShlSConstToEmitC;
struct PTOShlSToEmitC;
struct PTOShrSConstToEmitC;
struct PTOShrSToEmitC;
template <typename SignalOp>
struct PTOSignalCommToEmitC;
struct PTOSqrtSToEmitC;
struct PTOStructGetToEmitC;
struct PTOStructSetToEmitC;
struct PTOSubCSToEmitC;
struct PTOSubSCToEmitC;
struct PTOSubSSToEmitC;
struct PTOSubSToEmitC;
struct PTOSyncAllToEmitC;
struct PTOSyncFlagDynToEmitC;
struct PTOSyncSetToEmitC;
struct PTOSyncToEmitC;
struct PTOSyncWaitToEmitC;
struct PTOTAbsToTABS;
struct PTOTAddCToTADDC;
struct PTOTAddToTADD;
struct PTOTAllocToEmitC;
struct PTOTAndToEmitC;
struct PTOTAxpyToEmitC;
struct PTOTCIToEmitC;
struct PTOTDeInterleaveToEmitC;
struct PTOTDivSToEmitC;
struct PTOTFreeToEmitC;
struct PTOTGemvAccToTGEMVACC;
struct PTOTGemvToTGEMV;
struct PTOTInterleaveToEmitC;
struct PTOTLoadToTLOAD;
struct PTOTMatmulAccToTMATMULACC;
struct PTOTMatmulBiasToTMATMUL_BIAS;
struct PTOTMatmulToTMATMUL;
struct PTOTPopToEmitC;
struct PTOTPrefetchAsyncToEmitC;
struct PTOTPrefetchToTPREFETCH;
struct PTOTPushToEmitC;
struct PTOTReshapeToEmitC;
struct PTOTStoreToTSTORE;
struct PTOTTransToEmitC;
struct PTOTTriToEmitC;
struct PTOTileBufAddrToEmitC;
struct PTOTrapOpToEmitC;

struct PTOXORSToEmitC;
struct PTOXORToEmitC;
struct ReinterpretCastToEmitC;
template <typename SectionOpTy>
struct SectionToEmitC;
struct SubviewToEmitCPattern;

void populatePTOToEmitCPatterns(RewritePatternSet &patterns,
                               TypeConverter &typeConverter, MLIRContext *ctx,
                               PTOArch targetArch);

void populateArithPatterns(RewritePatternSet &patterns,
                           TypeConverter &typeConverter, MLIRContext *ctx,
                           PTOArch targetArch);
void populateMemrefPatterns(RewritePatternSet &patterns,
                            TypeConverter &typeConverter, MLIRContext *ctx,
                            PTOArch targetArch);
void populateLoadStorePatterns(RewritePatternSet &patterns,
                               TypeConverter &typeConverter, MLIRContext *ctx,
                               PTOArch targetArch);
void populateSyncCommPatterns(RewritePatternSet &patterns,
                              TypeConverter &typeConverter, MLIRContext *ctx,
                              PTOArch targetArch);
void populateScalarMiscPatterns(RewritePatternSet &patterns,
                                TypeConverter &typeConverter, MLIRContext *ctx,
                                PTOArch targetArch);
void populateTilePatterns(RewritePatternSet &patterns,
                          TypeConverter &typeConverter, MLIRContext *ctx,
                          PTOArch targetArch);
void populateTensorPatterns(RewritePatternSet &patterns,
                            TypeConverter &typeConverter, MLIRContext *ctx,
                            PTOArch targetArch);
void populateTensorReducePatterns(RewritePatternSet &patterns,
                                  TypeConverter &typeConverter, MLIRContext *ctx,
                                  PTOArch targetArch);



//---- sync-token extraction templates ------------------------------------//
template <typename T, typename = void>
struct HasGetSrcPipe : std::false_type {};
template <typename T>
struct HasGetSrcPipe<T, std::void_t<decltype(std::declval<T>().getSrcPipe())>> : std::true_type {};

template <typename T, typename = void>
struct HasGetDstPipe : std::false_type {};
template <typename T>
struct HasGetDstPipe<T, std::void_t<decltype(std::declval<T>().getDstPipe())>> : std::true_type {};

template <typename T, typename = void>
struct HasGetEventId : std::false_type {};
template <typename T>
struct HasGetEventId<T, std::void_t<decltype(std::declval<T>().getEventId())>> : std::true_type {};

template <typename T, typename = void>
struct HasGetSrcPipeAttr : std::false_type {};
template <typename T>
struct HasGetSrcPipeAttr<T, std::void_t<decltype(std::declval<T>().getSrcPipeAttr())>> : std::true_type {};

template <typename T, typename = void>
struct HasGetDstPipeAttr : std::false_type {};
template <typename T>
struct HasGetDstPipeAttr<T, std::void_t<decltype(std::declval<T>().getDstPipeAttr())>> : std::true_type {};

template <typename T, typename = void>
struct HasGetEventIdAttr : std::false_type {};
template <typename T>
struct HasGetEventIdAttr<T, std::void_t<decltype(std::declval<T>().getEventIdAttr())>> : std::true_type {};

template <typename SyncOpT>
static LogicalResult extractSyncTokens(SyncOpT op,
                                      std::string &srcTok,
                                      std::string &dstTok,
                                      std::string &evtTok,
                                      ConversionPatternRewriter &rewriter) {
  if constexpr (HasGetSrcPipe<SyncOpT>::value &&
                HasGetDstPipe<SyncOpT>::value &&
                HasGetEventId<SyncOpT>::value) {
    auto s = op.getSrcPipe();
    auto d = op.getDstPipe();
    auto e = op.getEventId();

    if constexpr (std::is_same<decltype(s), mlir::pto::PIPE>::value) srcTok = pipeTokFromPipeEnum(s);
    else srcTok = pipeTokFromPipeAttr(s);

    if constexpr (std::is_same<decltype(d), mlir::pto::PIPE>::value) dstTok = pipeTokFromPipeEnum(d);
    else dstTok = pipeTokFromPipeAttr(d);

    if constexpr (std::is_same<decltype(e), mlir::pto::EVENT>::value) evtTok = evtTokFromEventEnum(e);
    else evtTok = evtTokFromEventAttr(e);

    return success();
  }

  if constexpr (HasGetSrcPipeAttr<SyncOpT>::value &&
                HasGetDstPipeAttr<SyncOpT>::value &&
                HasGetEventIdAttr<SyncOpT>::value) {
    auto s = op.getSrcPipeAttr();
    auto d = op.getDstPipeAttr();
    auto e = op.getEventIdAttr();
    srcTok = pipeTokFromPipeAttr(s);
    dstTok = pipeTokFromPipeAttr(d);
    evtTok = evtTokFromEventAttr(e);
    return success();
  }

  return extractSyncTripletTokens(op.getOperation(), srcTok, dstTok, evtTok, rewriter);
}

} // namespace pto
} // namespace mlir

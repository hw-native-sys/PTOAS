//===- VPTOLowering.h - PTO to VPTO lowering contracts ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_VPTOLOWERING_H_
#define MLIR_DIALECT_PTO_TRANSFORMS_VPTOLOWERING_H_

#include "PTO/IR/PTO.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace pto {

enum class VPTOTileDomain {
  Vec,
  Acc,
  Mat,
};

enum class VPTOLoweringStrategy {
  PostUpdate,
  NoPostUpdate,
};

struct VPTOPartitionTrace {
  SmallVector<int64_t> offsets;
  SmallVector<int64_t> sizes;
  bool hasDynamicOffsets = false;
  bool hasDynamicSizes = false;
};

struct VPTOLoopProgramming {
  int64_t loop2 = 1;
  int64_t loop1 = 1;
  int64_t srcLoop2Stride = 1;
  int64_t srcLoop1Stride = 1;
  int64_t dstLoop2Stride = 1;
  int64_t dstLoop1Stride = 1;
};

enum class VPTOLoopScopeKind {
  None,
  AIVVectorScope,
};

struct VPTOLoopScopeContract {
  VPTOLoopScopeKind kind = VPTOLoopScopeKind::None;
  StringRef loweredAttr = "llvm.loop.aivector_scope";
  int64_t loopDepth = 0;
};

struct VPTOLoadContract {
  StringRef sourceLayout;
  SmallVector<int64_t> sourceShape;
  SmallVector<int64_t> sourceStrides;
  StringRef tileLayout;
  VPTOTileDomain tileDomain = VPTOTileDomain::Vec;
  Type elementType;
  Value validRowsValue;
  Value validColsValue;
  int64_t validRows = ShapedType::kDynamic;
  int64_t validCols = ShapedType::kDynamic;
  StringRef padMode;
  Value padValue;
  Value leftPaddingNum;
  Value rightPaddingNum;
  bool initOutBuffer = false;
  Value initCondition;
  VPTOPartitionTrace trace;
};

struct VPTOUnaryContract {
  StringRef family;
  VPTOTileDomain tileDomain = VPTOTileDomain::Vec;
  StringRef tileLayout;
  Value validRowsValue;
  Value validColsValue;
  int64_t validRows = ShapedType::kDynamic;
  int64_t validCols = ShapedType::kDynamic;
  Type elementType;
  VPTOLoopScopeContract loopScope;
};

struct VPTOBinaryContract {
  StringRef family;
  VPTOTileDomain tileDomain = VPTOTileDomain::Vec;
  StringRef tileLayout;
  Value validRowsValue;
  Value validColsValue;
  int64_t validRows = ShapedType::kDynamic;
  int64_t validCols = ShapedType::kDynamic;
  Type elementType;
  VPTOLoopScopeContract loopScope;
};

struct VPTOStoreContract {
  VPTOTileDomain srcDomain = VPTOTileDomain::Vec;
  StringRef destinationLayout;
  SmallVector<int64_t> destinationShape;
  SmallVector<int64_t> destinationStrides;
  Type elementType;
  Value validRowsValue;
  Value validColsValue;
  int64_t validRows = ShapedType::kDynamic;
  int64_t validCols = ShapedType::kDynamic;
  VPTOPartitionTrace trace;
};

void set_loop2_stride_outtoub(Operation *copyOp, int64_t dstStride,
                              int64_t srcStride, Builder &builder);
void set_loop1_stride_outtoub(Operation *copyOp, int64_t dstStride,
                              int64_t srcStride, Builder &builder);
void set_loop_size_outtoub(Operation *copyOp, int64_t loop2, int64_t loop1,
                           Builder &builder);
void set_loop2_stride_ubtoout(Operation *copyOp, int64_t srcStride,
                              int64_t dstStride, Builder &builder);
void set_loop1_stride_ubtoout(Operation *copyOp, int64_t srcStride,
                              int64_t dstStride, Builder &builder);
void set_loop_size_ubtoout(Operation *copyOp, int64_t loop2, int64_t loop1,
                           Builder &builder);
FailureOr<pto::VecScopeOp>
createLoopScopeRegion(Location loc, const VPTOLoopScopeContract &contract,
                      PatternRewriter &rewriter);
Value materializeBufferPointer(Value value, Type elementType,
                               Attribute memorySpace,
                               PatternRewriter &rewriter, Location loc);

LogicalResult lowerTLOAD(TLoadOp op, PatternRewriter &rewriter);
LogicalResult lowerTABS(TAbsOp op, PatternRewriter &rewriter,
                        VPTOLoweringStrategy strategy);
LogicalResult lowerTADD(TAddOp op, PatternRewriter &rewriter,
                        VPTOLoweringStrategy strategy);
LogicalResult lowerTSUB(TSubOp op, PatternRewriter &rewriter,
                        VPTOLoweringStrategy strategy);
LogicalResult lowerTMUL(TMulOp op, PatternRewriter &rewriter,
                        VPTOLoweringStrategy strategy);
LogicalResult lowerTDIV(TDivOp op, PatternRewriter &rewriter,
                        VPTOLoweringStrategy strategy);
LogicalResult lowerTMAX(TMaxOp op, PatternRewriter &rewriter,
                        VPTOLoweringStrategy strategy);
LogicalResult lowerTMIN(TMinOp op, PatternRewriter &rewriter,
                        VPTOLoweringStrategy strategy);
LogicalResult lowerTAND(TAndOp op, PatternRewriter &rewriter,
                        VPTOLoweringStrategy strategy);
LogicalResult lowerTANDS(TAndSOp op, PatternRewriter &rewriter);
LogicalResult lowerTOR(TOrOp op, PatternRewriter &rewriter,
                       VPTOLoweringStrategy strategy);
LogicalResult lowerTORS(TOrSOp op, PatternRewriter &rewriter);
LogicalResult lowerTXOR(TXorOp op, PatternRewriter &rewriter,
                        VPTOLoweringStrategy strategy);
LogicalResult lowerTXORS(TXorSOp op, PatternRewriter &rewriter);
LogicalResult lowerTEXP(TExpOp op, PatternRewriter &rewriter,
                        VPTOLoweringStrategy strategy);
LogicalResult lowerTLOG(TLogOp op, PatternRewriter &rewriter,
                        VPTOLoweringStrategy strategy);
LogicalResult lowerTSQRT(TSqrtOp op, PatternRewriter &rewriter,
                         VPTOLoweringStrategy strategy);
LogicalResult lowerTRSQRT(TRsqrtOp op, PatternRewriter &rewriter);
LogicalResult lowerTRECIP(TRecipOp op, PatternRewriter &rewriter,
                          VPTOLoweringStrategy strategy);
LogicalResult lowerTNEG(TNegOp op, PatternRewriter &rewriter,
                        VPTOLoweringStrategy strategy);
LogicalResult lowerTLRELU(TLReluOp op, PatternRewriter &rewriter,
                          VPTOLoweringStrategy strategy);
LogicalResult lowerTCI(TCIOp op, PatternRewriter &rewriter);
LogicalResult lowerTCVT(TCvtOp op, PatternRewriter &rewriter);
LogicalResult lowerTCmp(TCmpOp op, PatternRewriter &rewriter);
LogicalResult lowerTCmpS(TCmpSOp op, PatternRewriter &rewriter);
LogicalResult lowerTSel(TSelOp op, PatternRewriter &rewriter);
LogicalResult lowerTAddC(TAddCOp op, PatternRewriter &rewriter);
LogicalResult lowerTAddS(TAddSOp op, PatternRewriter &rewriter,
                         VPTOLoweringStrategy strategy);
LogicalResult lowerTAddSC(TAddSCOp op, PatternRewriter &rewriter);
LogicalResult lowerTMinS(TMinSOp op, PatternRewriter &rewriter,
                         VPTOLoweringStrategy strategy);
LogicalResult lowerTDivS(TDivSOp op, PatternRewriter &rewriter,
                         VPTOLoweringStrategy strategy);
LogicalResult lowerTMulS(TMulSOp op, PatternRewriter &rewriter,
                         VPTOLoweringStrategy strategy);
LogicalResult lowerTSubC(TSubCOp op, PatternRewriter &rewriter);
LogicalResult lowerTSubS(TSubSOp op, PatternRewriter &rewriter,
                         VPTOLoweringStrategy strategy);
LogicalResult lowerTSubSC(TSubSCOp op, PatternRewriter &rewriter);
LogicalResult lowerTMaxS(TMaxSOp op, PatternRewriter &rewriter,
                         VPTOLoweringStrategy strategy);
LogicalResult lowerTSelS(TSelSOp op, PatternRewriter &rewriter);
LogicalResult lowerTRELU(TReluOp op, PatternRewriter &rewriter,
                         VPTOLoweringStrategy strategy);
LogicalResult lowerTNOT(TNotOp op, PatternRewriter &rewriter,
                        VPTOLoweringStrategy strategy);
LogicalResult lowerTTRANS(TTransOp op, PatternRewriter &rewriter);
LogicalResult lowerTFILLPAD(TFillPadOp op, PatternRewriter &rewriter);
LogicalResult lowerTFILLPADExpand(TFillPadExpandOp op, PatternRewriter &rewriter);
LogicalResult lowerTRowMax(TRowMaxOp op, PatternRewriter &rewriter,
                           VPTOLoweringStrategy strategy);
LogicalResult lowerTRowMin(TRowMinOp op, PatternRewriter &rewriter,
                           VPTOLoweringStrategy strategy);
LogicalResult lowerTRowSum(TRowSumOp op, PatternRewriter &rewriter,
                           VPTOLoweringStrategy strategy);
LogicalResult lowerTColMax(TColMaxOp op, PatternRewriter &rewriter);
LogicalResult lowerTColMin(TColMinOp op, PatternRewriter &rewriter);
LogicalResult lowerTColSum(TColSumOp op, PatternRewriter &rewriter);
LogicalResult lowerTRowExpand(TRowExpandOp op, PatternRewriter &rewriter,
                              VPTOLoweringStrategy strategy);
LogicalResult lowerTColExpand(TColExpandOp op, PatternRewriter &rewriter);
LogicalResult lowerTRowExpandMul(TRowExpandMulOp op, PatternRewriter &rewriter,
                                 VPTOLoweringStrategy strategy);
LogicalResult lowerTRowExpandDiv(TRowExpandDivOp op, PatternRewriter &rewriter,
                                 VPTOLoweringStrategy strategy);
LogicalResult lowerTRowExpandSub(TRowExpandSubOp op, PatternRewriter &rewriter,
                                 VPTOLoweringStrategy strategy);
LogicalResult lowerTPartAdd(TPartAddOp op, PatternRewriter &rewriter);
LogicalResult lowerTPartMax(TPartMaxOp op, PatternRewriter &rewriter);
LogicalResult lowerTPartMin(TPartMinOp op, PatternRewriter &rewriter);
LogicalResult lowerTExpandS(TExpandsOp op, PatternRewriter &rewriter);
LogicalResult lowerTGather(TGatherOp op, PatternRewriter &rewriter);
LogicalResult lowerTGatherB(TGatherBOp op, PatternRewriter &rewriter);
LogicalResult lowerTScatter(TScatterOp op, PatternRewriter &rewriter);
LogicalResult lowerTMrgSort(TMrgSortOp op, PatternRewriter &rewriter);
LogicalResult lowerTSort32(TSort32Op op, PatternRewriter &rewriter);
LogicalResult lowerTSTORE(TStoreOp op, PatternRewriter &rewriter);
LogicalResult lowerSetFlag(SetFlagOp op, PatternRewriter &rewriter);
LogicalResult lowerWaitFlag(WaitFlagOp op, PatternRewriter &rewriter);
LogicalResult lowerBarrier(BarrierOp op, PatternRewriter &rewriter);
LogicalResult lowerGetBuf(GetBufOp op, PatternRewriter &rewriter);
LogicalResult lowerRlsBuf(RlsBufOp op, PatternRewriter &rewriter);
LogicalResult convertVPTOEmissionBoundaryToPtr(
    ModuleOp module, llvm::raw_ostream *diagOS = nullptr);

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_VPTOLOWERING_H_

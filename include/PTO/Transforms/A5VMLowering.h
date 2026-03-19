//===- A5VMLowering.h - PTO to A5VM lowering contracts ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_A5VMLOWERING_H_
#define MLIR_DIALECT_PTO_TRANSFORMS_A5VMLOWERING_H_

#include "PTO/IR/PTO.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace pto {

enum class A5VMTileDomain {
  Vec,
  Acc,
  Mat,
};

struct A5VMPartitionTrace {
  SmallVector<int64_t> offsets;
  SmallVector<int64_t> sizes;
  bool hasDynamicOffsets = false;
  bool hasDynamicSizes = false;
};

struct A5VMLoopProgramming {
  int64_t loop2 = 1;
  int64_t loop1 = 1;
  int64_t srcLoop2Stride = 1;
  int64_t srcLoop1Stride = 1;
  int64_t dstLoop2Stride = 1;
  int64_t dstLoop1Stride = 1;
};

enum class A5VMLoopScopeKind {
  None,
  AIVVectorScope,
};

struct A5VMLoopScopeContract {
  A5VMLoopScopeKind kind = A5VMLoopScopeKind::None;
  StringRef loweredAttr = "llvm.loop.aivector_scope";
  int64_t loopDepth = 0;
};

struct A5VMLoadContract {
  StringRef sourceLayout;
  SmallVector<int64_t> sourceShape;
  SmallVector<int64_t> sourceStrides;
  StringRef tileLayout;
  A5VMTileDomain tileDomain = A5VMTileDomain::Vec;
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
  A5VMPartitionTrace trace;
};

struct A5VMUnaryContract {
  StringRef family;
  A5VMTileDomain tileDomain = A5VMTileDomain::Vec;
  StringRef tileLayout;
  Value validRowsValue;
  Value validColsValue;
  int64_t validRows = ShapedType::kDynamic;
  int64_t validCols = ShapedType::kDynamic;
  Type elementType;
  A5VMLoopScopeContract loopScope;
};

struct A5VMBinaryContract {
  StringRef family;
  A5VMTileDomain tileDomain = A5VMTileDomain::Vec;
  StringRef tileLayout;
  Value validRowsValue;
  Value validColsValue;
  int64_t validRows = ShapedType::kDynamic;
  int64_t validCols = ShapedType::kDynamic;
  Type elementType;
  A5VMLoopScopeContract loopScope;
};

struct A5VMStoreContract {
  A5VMTileDomain srcDomain = A5VMTileDomain::Vec;
  StringRef destinationLayout;
  SmallVector<int64_t> destinationShape;
  SmallVector<int64_t> destinationStrides;
  Type elementType;
  Value validRowsValue;
  Value validColsValue;
  int64_t validRows = ShapedType::kDynamic;
  int64_t validCols = ShapedType::kDynamic;
  A5VMPartitionTrace trace;
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
LogicalResult attachLoopScopeMetadata(LoopLikeOpInterface loop,
                                      const A5VMLoopScopeContract &contract,
                                      PatternRewriter &rewriter);

LogicalResult programCopyGmToUbLoops(Operation *copyOp,
                                     const A5VMLoadContract &contract,
                                     Builder &builder);
LogicalResult programCopyUbToGmLoops(Operation *copyOp,
                                     const A5VMStoreContract &contract,
                                     Builder &builder);
LogicalResult buildUnaryVecScope(StringRef family,
                                 const A5VMUnaryContract &contract, Value src,
                                 Value dst, PatternRewriter &rewriter,
                                 Location loc);
LogicalResult buildBinaryVecScope(StringRef family,
                                  const A5VMBinaryContract &contract,
                                  Value src0, Value src1, Value dst,
                                  PatternRewriter &rewriter, Location loc);

LogicalResult lowerTLOAD(TLoadOp op, PatternRewriter &rewriter);
LogicalResult lowerTABS(TAbsOp op, PatternRewriter &rewriter);
LogicalResult lowerTADD(TAddOp op, PatternRewriter &rewriter);
LogicalResult lowerTSUB(TSubOp op, PatternRewriter &rewriter);
LogicalResult lowerTMUL(TMulOp op, PatternRewriter &rewriter);
LogicalResult lowerTDIV(TDivOp op, PatternRewriter &rewriter);
LogicalResult lowerTEXP(TExpOp op, PatternRewriter &rewriter);
LogicalResult lowerTLOG(TLogOp op, PatternRewriter &rewriter);
LogicalResult lowerTSQRT(TSqrtOp op, PatternRewriter &rewriter);
LogicalResult lowerTRECIP(TRecipOp op, PatternRewriter &rewriter);
LogicalResult lowerTRELU(TReluOp op, PatternRewriter &rewriter);
LogicalResult lowerTNOT(TNotOp op, PatternRewriter &rewriter);
LogicalResult lowerTSTORE(TStoreOp op, PatternRewriter &rewriter);
LogicalResult lowerSetFlag(SetFlagOp op, PatternRewriter &rewriter);
LogicalResult lowerWaitFlag(WaitFlagOp op, PatternRewriter &rewriter);
LogicalResult lowerBarrier(BarrierOp op, PatternRewriter &rewriter);

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_A5VMLOWERING_H_

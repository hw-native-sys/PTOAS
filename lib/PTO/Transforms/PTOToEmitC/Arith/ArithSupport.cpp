// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- ArithSupport.cpp - Arith lowering helpers --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "ArithInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

// CANN Open Software License Agreement Version 2.0 (the "License").
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.

//===- PTOToEmitCArith.cpp - arith/inter-core helpers to EmitC patterns ---------===//
//===----------------------------------------------------------------------===//






//===----------------------------------------------------------------------===//
// Arith -> EmitC (full dialect coverage for scalar ops)
//===----------------------------------------------------------------------===//

// Shared prologue for unsigned-interpretation arith lowering: resolves the
// unsigned-cast operands and their type for a binary integer op. Returns
// failure when the operand type is not a scalar integer/index.

FailureOr<UnsignedBinaryOperands>
getUnsignedBinaryOperands(Operation *op, Value lhs, Value rhs,
                          ConversionPatternRewriter &rewriter) {
  auto loc = op->getLoc();
  Type opTy = op->getResult(0).getType();
  auto intTy = dyn_cast<IntegerType>(opTy);
  if (!intTy && !isa<IndexType>(opTy)) {
    op->emitError("expected scalar integer or index type");
    return failure();
  }
  const unsigned bitWidth =
      intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);
  auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
  return UnsignedBinaryOperands{
      uTy, castSignlessIntToUnsignedSameWidth(rewriter, loc, lhs, bitWidth),
      castSignlessIntToUnsignedSameWidth(rewriter, loc, rhs, bitWidth)};
}

// Scalar integer/index operands are lowered at the 64-bit index width when the
// operand is an index; otherwise the integer's own width is kept.
unsigned getScalarIntOrIndexBitWidth(Type opTy) {
  if (auto intTy = dyn_cast<IntegerType>(opTy))
    return intTy.getWidth();
  return kPTOIndexBitWidth;
}

bool isScalarIntOrIndex(Type opTy) {
  return isa<IntegerType, IndexType>(opTy);
}

// Integer bitwise/div/rem ops (andi/ori/xori/divui/remui) on signless
// integers: perform in unsigned to avoid signedness pitfalls, then cast back.
// Shared prologue for scalar integer/index arith lowering: validates the
// operand type and resolves the converted result type.
// Shared pieces for signed ceil/floor division lowering: the truncating
// quotient/remainder plus the (signs differ?) predicate used by both
// adjustment directions.

SignedDivParts buildSignedDivParts(ConversionPatternRewriter &rewriter,
                                          Location loc, Type dstTy, Value lhs,
                                          Value rhs) {
  SignedDivParts parts;
  Value zero = makeEmitCIntConstant(rewriter, loc, dstTy, 0);
  parts.quotient = rewriter.create<emitc::DivOp>(loc, dstTy, lhs, rhs);
  parts.remainder = rewriter.create<emitc::RemOp>(loc, dstTy, lhs, rhs);
  parts.remainderNonZero = rewriter.create<emitc::CmpOp>(
      loc, rewriter.getI1Type(), emitc::CmpPredicate::ne, parts.remainder,
      zero);
  Value lhsLt0 = rewriter.create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                               emitc::CmpPredicate::lt, lhs,
                                               zero);
  Value rhsLt0 = rewriter.create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                               emitc::CmpPredicate::lt, rhs,
                                               zero);
  parts.signsDiffer = rewriter.create<emitc::CmpOp>(
      loc, rewriter.getI1Type(), emitc::CmpPredicate::ne, lhsLt0, rhsLt0);
  parts.signsSame = rewriter.create<emitc::CmpOp>(
      loc, rewriter.getI1Type(), emitc::CmpPredicate::eq, lhsLt0, rhsLt0);
  return parts;
}

// Shared tail for unsigned-interpretation binary lowering: resolve the
// unsigned-cast operands, apply the binary op, and cast back to dstTy.
// Full scalar-int binary lowering flow: shared prologue, i1 special case
// (EmitCI1Op), then the unsigned-interpretation tail. EmitCI1Op is used only
// when the operands are i1; pass EmitCOp itself when no distinct i1 op exists
// (callers with their own i1 handling should not use this driver).
// Shared signed ceil/floor division emission: compensate the truncating
// quotient by +/- 1 when the remainder is non-zero and the sign condition
// holds. CeilDiv adjusts when signs are the same; FloorDiv when they differ.
// Integer shifts on signless operands: compute in the unsigned C++ type of
// the same width, then cast back. i1 shifts are widened to u8 and truncated.
// Emit the sign-extension or zero-extension of a signless integer operand to
// `dstTy`. i1 sources are materialized as 0/-1 (signed) or passed through
// (unsigned) before widening.
LogicalResult emitWidenedInt(Operation *op, Value in, IntegerType srcIntTy,
                                    IntegerType dstIntTy, Type dstTy,
                                    bool isSigned,
                                    ConversionPatternRewriter &rewriter) {
  auto loc = op->getLoc();
  if (srcIntTy.getWidth() == 1) {
    if (isSigned) {
      Value zero = makeEmitCIntConstant(rewriter, loc, dstTy, 0);
      Value asInt = emitCCast(rewriter, loc, dstTy, in);
      Value neg = rewriter.create<emitc::SubOp>(loc, dstTy, zero, asInt).getResult();
      rewriter.replaceOp(op, neg);
    } else {
      rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, in);
    }
    return success();
  }
  if (isSigned) {
    // Signed widening relies on the C++ assignment conversion.
    rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, in);
    return success();
  }
  auto uDstTy = getUnsignedIntOpaqueType(rewriter.getContext(), dstIntTy.getWidth());
  Value srcU = castSignlessIntToUnsignedSameWidth(rewriter, loc, in,
                                                  srcIntTy.getWidth());
  Value extU = emitCCast(rewriter, loc, uDstTy, srcU);
  rewriter.replaceOp(op, emitCCast(rewriter, loc, dstTy, extU));
  return success();
}

// arith.cmpf lowering with ordered/unordered semantics.

// cmpf helpers: NaN tests, special always-true/false/ORD/UNO forms, and the
// ordered/unordered comparison composition.

Value cmpFIsNaN(ConversionPatternRewriter &rewriter, Location loc,
                       Value v) {
  return rewriter
      .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                            emitc::CmpPredicate::ne, v, v)
      .getResult();
}

Value cmpFIsNotNaN(ConversionPatternRewriter &rewriter, Location loc,
                          Value v) {
  return rewriter
      .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                            emitc::CmpPredicate::eq, v, v)
      .getResult();
}

std::optional<Value>
buildSpecialCmpFResult(arith::CmpFPredicate predicate,
                       ConversionPatternRewriter &rewriter, Location loc,
                       Type i1Ty, Value lhs, Value rhs) {
  switch (predicate) {
  case arith::CmpFPredicate::AlwaysFalse:
    return makeEmitCOpaqueConstant(rewriter, loc, i1Ty, "false");
  case arith::CmpFPredicate::AlwaysTrue:
    return makeEmitCOpaqueConstant(rewriter, loc, i1Ty, "true");
  case arith::CmpFPredicate::ORD:
    return rewriter
        .create<emitc::LogicalAndOp>(loc, i1Ty,
                                     cmpFIsNotNaN(rewriter, loc, lhs),
                                     cmpFIsNotNaN(rewriter, loc, rhs))
        .getResult();
  case arith::CmpFPredicate::UNO:
    return rewriter
        .create<emitc::LogicalOrOp>(loc, i1Ty, cmpFIsNaN(rewriter, loc, lhs),
                                    cmpFIsNaN(rewriter, loc, rhs))
        .getResult();
  default:
    return std::nullopt;
  }
}

std::optional<ArithCmpFConfig>
getCmpFConfig(arith::CmpFPredicate predicate) {
  switch (predicate) {
  case arith::CmpFPredicate::OEQ:
    return ArithCmpFConfig{false, emitc::CmpPredicate::eq};
  case arith::CmpFPredicate::OGT:
    return ArithCmpFConfig{false, emitc::CmpPredicate::gt};
  case arith::CmpFPredicate::OGE:
    return ArithCmpFConfig{false, emitc::CmpPredicate::ge};
  case arith::CmpFPredicate::OLT:
    return ArithCmpFConfig{false, emitc::CmpPredicate::lt};
  case arith::CmpFPredicate::OLE:
    return ArithCmpFConfig{false, emitc::CmpPredicate::le};
  case arith::CmpFPredicate::ONE:
    return ArithCmpFConfig{false, emitc::CmpPredicate::ne};
  case arith::CmpFPredicate::UEQ:
    return ArithCmpFConfig{true, emitc::CmpPredicate::eq};
  case arith::CmpFPredicate::UGT:
    return ArithCmpFConfig{true, emitc::CmpPredicate::gt};
  case arith::CmpFPredicate::UGE:
    return ArithCmpFConfig{true, emitc::CmpPredicate::ge};
  case arith::CmpFPredicate::ULT:
    return ArithCmpFConfig{true, emitc::CmpPredicate::lt};
  case arith::CmpFPredicate::ULE:
    return ArithCmpFConfig{true, emitc::CmpPredicate::le};
  case arith::CmpFPredicate::UNE:
    return ArithCmpFConfig{true, emitc::CmpPredicate::ne};
  default:
    return std::nullopt;
  }
}

Value buildCmpFResult(const ArithCmpFConfig &config,
                             ConversionPatternRewriter &rewriter, Location loc,
                             Type i1Ty, Value lhs, Value rhs) {
  Value cmp = rewriter
                  .create<emitc::CmpOp>(loc, i1Ty, config.predicate, lhs, rhs)
                  .getResult();
  Value unord = rewriter.create<emitc::LogicalOrOp>(
      loc, i1Ty, cmpFIsNaN(rewriter, loc, lhs), cmpFIsNaN(rewriter, loc, rhs));
  if (config.unordered)
    return rewriter.create<emitc::LogicalOrOp>(loc, i1Ty, unord, cmp)
        .getResult();
  Value ord = rewriter.create<emitc::LogicalAndOp>(
      loc, i1Ty, cmpFIsNotNaN(rewriter, loc, lhs),
      cmpFIsNotNaN(rewriter, loc, rhs));
  return rewriter.create<emitc::LogicalAndOp>(loc, i1Ty, ord, cmp).getResult();
}
// min/max integer lowering: `select(lhs < rhs, A, B)` where (A, B) picks the
// smaller operand for min and the larger for max. Unsigned variants compare
// through the unsigned C++ type of the same width so values with the sign bit
// set order correctly.
// Floating-point max/min variants.

// maxnum/minnum lowering: a plain lt-based min/max plus NaN-propagation
// selects on both operands.
//===----------------------------------------------------------------------===//
// Arith -> EmitC helpers
//===----------------------------------------------------------------------===//









// For signless iN integers lowered to signed C++ types, this creates a value
// representing the same N-bit pattern in an unsigned C++ type of the same
// width. This avoids incorrect sign-extension when later widening to a larger
// unsigned type.

// muli/addi/subi on signless integers: compute in the unsigned C++ type of
// the same width, then cast back. i1 arithmetic wraps to a single bit, so
// mul lowers to AND and add/sub to XOR.
//===----------------------------------------------------------------------===//
// pto.mgather lowering -> MGATHER(dst, src, indexes)  (pto-isa)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Kernel inference helpers
//===----------------------------------------------------------------------===//

enum class KernelKind { VecAdd, Matmul, Unknown };

[[maybe_unused]] static KernelKind inferKernelKind(func::FuncOp f) {
  bool hasAdd = false;
  bool hasMM  = false;
  f.walk([&](Operation *op) {
    if (isa<mlir::pto::TAddOp>(op)) {
      hasAdd = true;
    }
    if (isa<mlir::pto::TMatmulOp>(op)) {
      hasMM = true;
    }
    if (isa<mlir::pto::TMatmulAccOp>(op)) {
      hasMM = true;
    }
  });
  if (hasMM) {
    return KernelKind::Matmul;
  }
  if (hasAdd) {
    return KernelKind::VecAdd;
  }
  return KernelKind::Unknown;
}

[[maybe_unused]] static void inferTileMNK(func::FuncOp f, int &M, int &N, int &K) {
  M = 32; N = 32; K = 32;
  SmallVector<memref::SubViewOp, 4> subs;
  f.walk([&](memref::SubViewOp sv) { subs.push_back(sv); });

  auto readShape2D = [&](memref::SubViewOp sv, int &d0, int &d1) {
    auto resTy = mlir::cast<MemRefType>(sv.getResult().getType());
    if (resTy.getRank() == 2 && resTy.hasStaticShape()) {
      d0 = static_cast<int>(resTy.getDimSize(0));
      d1 = static_cast<int>(resTy.getDimSize(1));
    }
  };

  if (subs.empty()) {
    return;
  }

  int a0=32, a1=32;
  readShape2D(subs[0], a0, a1);
  M = a0; N = a1;

  if (subs.size() >= 2) {
    int b0=32, b1=32;
    readShape2D(subs[0], a0, a1);
    readShape2D(subs[1], b0, b1);
    M = a0; K = a1; N = b1;
  }
}




// Pick the C++ specifiers (extern "C"/static/__global__) for the emitted
// AICORE function based on its linkage and PTO entry attributes.
void applyFuncSpecifiers(func::FuncOp op,
                                ConversionPatternRewriter &rewriter,
                                emitc::FuncOp &emitcFunc) {
  if (pto::isPTOEntryFunction(op)) {
    emitcFunc.setSpecifiersAttr(
        rewriter.getStrArrayAttr({"extern \"C\"", "__global__ AICORE"}));
  } else if (op.isPrivate()) {
    emitcFunc.setSpecifiersAttr(rewriter.getStrArrayAttr({"static", "AICORE"}));
  } else if (pto::hasExternalArtifactVisibility(op)) {
    emitcFunc.setSpecifiersAttr(
        rewriter.getStrArrayAttr({"extern \"C\"", "AICORE"}));
  } else {
    emitcFunc.setSpecifiersAttr(rewriter.getStrArrayAttr({"AICORE"}));
  }
}

//===----------------------------------------------------------------------===//
// SubView lowering to GlobalTensor (keep your existing code)
//===----------------------------------------------------------------------===


InterCoreSyncCallDesc buildInterCoreSyncSetCallImpl(
    ConversionPatternRewriter &rewriter, Value msgVal, PTOArch targetArch,
    pto::PipeAttr pipeAttr) {
  auto *ctx = rewriter.getContext();
  std::string pipeTok = pipeTokFromPipeAttr(pipeAttr);

  (void)targetArch;
  InterCoreSyncCallDesc desc;
  desc.callee = "__builtin_cce_ffts_cross_core_sync";
  desc.args = rewriter.getArrayAttr({
      emitc::OpaqueAttr::get(ctx, pipeTok),
      IntegerAttr::get(IndexType::get(ctx), 0),
  });
  desc.operands.push_back(msgVal);
  return desc;
}

InterCoreSyncCallDesc buildInterCoreSyncSetCall(
    ConversionPatternRewriter &rewriter, Location loc, PTOArch targetArch,
    pto::PipeAttr pipeAttr, IntegerAttr eventIdAttr, int64_t fftsMode) {
  auto indexTy = emitc::OpaqueType::get(rewriter.getContext(), "int64_t");
  Value eventVal =
      makeEmitCIntConstant(rewriter, loc, indexTy,
                           getIntegerAttrSignedValue(eventIdAttr));
  Value msgVal = createFFTSMsg(rewriter, loc, eventVal, fftsMode);
  return buildInterCoreSyncSetCallImpl(rewriter, msgVal, targetArch, pipeAttr);
}

InterCoreSyncCallDesc buildInterCoreSyncSetCallDyn(
    ConversionPatternRewriter &rewriter, Location loc, PTOArch targetArch,
    pto::PipeAttr pipeAttr, Value eventIdVal, int64_t fftsMode) {
  Value msgVal = createFFTSMsg(rewriter, loc, eventIdVal, fftsMode);
  return buildInterCoreSyncSetCallImpl(rewriter, msgVal, targetArch, pipeAttr);
}

InterCoreSyncCallDesc buildInterCoreSyncWaitCall(
    ConversionPatternRewriter &rewriter, PTOArch targetArch,
    pto::PipeAttr pipeAttr, IntegerAttr eventIdAttr) {
  std::string pipeTok = pipeTokFromPipeAttr(pipeAttr);

  InterCoreSyncCallDesc desc;
  (void)targetArch;
  (void)pipeTok;
  desc.callee = "__builtin_cce_wait_flag_dev";
  desc.args = rewriter.getArrayAttr({eventIdAttr});
  return desc;
}

InterCoreSyncCallDesc buildInterCoreSyncWaitCallDyn(
    ConversionPatternRewriter &rewriter, Location loc, PTOArch targetArch,
    pto::PipeAttr pipeAttr, Value eventIdVal) {
  auto *ctx = rewriter.getContext();
  std::string pipeTok = pipeTokFromPipeAttr(pipeAttr);
  InterCoreSyncCallDesc desc;
  (void)targetArch;
  (void)pipeTok;
  desc.callee = "__builtin_cce_wait_flag_dev";
  desc.args = rewriter.getArrayAttr({IntegerAttr::get(IndexType::get(ctx), 0)});
  desc.operands.push_back(castInterCoreEventIdToI32(rewriter, loc, eventIdVal));
  return desc;
}

Value castInterCoreEventIdToI32(ConversionPatternRewriter &rewriter,
                                       Location loc, Value eventId) {
  auto i32Ty = emitc::OpaqueType::get(rewriter.getContext(), "int32_t");
  if (eventId.getType() == i32Ty)
    return eventId;
  return emitCCast(rewriter, loc, i32Ty, eventId);
}

Value createFFTSMsg(ConversionPatternRewriter &rewriter, Location loc,
                           Value eventId, int64_t fftsMode) {
  auto *ctx = rewriter.getContext();
  auto msgTy = emitc::OpaqueType::get(ctx, "uint16_t");
  auto msgArgs = rewriter.getArrayAttr({
      getFFTSModeCodegenArg(rewriter, fftsMode),
      IntegerAttr::get(IndexType::get(ctx), 0),
  });
  return rewriter
      .create<emitc::CallOpaqueOp>(loc, msgTy, "getFFTSMsg",
                                   /*args=*/msgArgs,
                                   /*templateArgs=*/ArrayAttr{},
                                   /*operands=*/ValueRange{eventId})
      .getResult(0);
}

Attribute getFFTSModeCodegenArg(ConversionPatternRewriter &rewriter,
                                       int64_t fftsMode) {
  auto *ctx = rewriter.getContext();
  if (fftsMode == 2)
    return emitc::OpaqueAttr::get(ctx, "FFTS_MODE_VAL");
  return emitc::OpaqueAttr::get(ctx, std::to_string(fftsMode));
}

bool hasInterCoreSyncOp(func::FuncOp func) {
  bool found = false;
  func.walk([&](Operation *op) {
    if (isa<pto::SyncSetOp, pto::SyncWaitOp, pto::SetCrossBlockOp,
            pto::WaitCrossBlockOp, pto::SetIntraBlockOp,
            pto::WaitIntraBlockOp>(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

bool hasSetFFTsOp(func::FuncOp func) {
  bool found = false;
  func.walk([&](Operation *op) {
    if (isa<pto::SetFFTsOp>(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}


std::optional<StringRef> getKernelKindMacro(func::FuncOp funcOp) {
  auto kernelKindAttr =
      funcOp->getAttrOfType<FunctionKernelKindAttr>(FunctionKernelKindAttr::name);
  if (!kernelKindAttr)
    return std::nullopt;

  switch (kernelKindAttr.getKernelKind()) {
  case FunctionKernelKind::Cube:
    return StringRef("__DAV_CUBE__");
  case FunctionKernelKind::Vector:
    return StringRef("__DAV_VEC__");
  }

  llvm_unreachable("unexpected kernel kind");
}



} // namespace pto
} // namespace mlir

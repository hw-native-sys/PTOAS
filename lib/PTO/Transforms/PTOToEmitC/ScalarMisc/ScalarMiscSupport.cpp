// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- ScalarMiscSupport.cpp - ScalarMisc lowering helpers --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "ScalarMiscInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

// CANN Open Software License Agreement Version 2.0 (the "License").
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.

//===- PTOToEmitCScalarMisc.cpp - sync/barrier/comm/async/declare lowering ---------===//
//===----------------------------------------------------------------------===//






StringRef scatterAtomicTok(pto::ScatterAtomicOp atomic) {
  switch (atomic) {
  case pto::ScatterAtomicOp::None:
    return "pto::ScatterAtomicOp::None";
  case pto::ScatterAtomicOp::Add:
    return "pto::ScatterAtomicOp::Add";
  case pto::ScatterAtomicOp::Max:
    return "pto::ScatterAtomicOp::Max";
  case pto::ScatterAtomicOp::Min:
    return "pto::ScatterAtomicOp::Min";
  }
  llvm_unreachable("unknown ScatterAtomicOp");
}

StringRef scatterOobTok(pto::ScatterOOB mode) {
  switch (mode) {
  case pto::ScatterOOB::Undefined:
    return "pto::ScatterOOB::Undefined";
  case pto::ScatterOOB::Skip:
    return "pto::ScatterOOB::Skip";
  case pto::ScatterOOB::Clamp:
    return "pto::ScatterOOB::Clamp";
  case pto::ScatterOOB::Wrap:
    return "pto::ScatterOOB::Wrap";
  }
  llvm_unreachable("unknown ScatterOOB");
}

StringRef scatterConflictTok(pto::ScatterConflict mode) {
  switch (mode) {
  case pto::ScatterConflict::Last:
    return "pto::ScatterConflict::Last";
  case pto::ScatterConflict::Default:
    return "pto::ScatterConflict::Default";
  }
  llvm_unreachable("unknown ScatterConflict");
}

StringRef coalesceTok(pto::Coalesce mode) {
  switch (mode) {
  case pto::Coalesce::Row:
    return "pto::Coalesce::Row";
  case pto::Coalesce::Elem:
    return "pto::Coalesce::Elem";
  }
  llvm_unreachable("unknown Coalesce");
}

// Strip conversion casts (unrealized + emitc) down to the producing value.
Value peelAllConversionCasts(Value v) {
  while (auto castOp = v.getDefiningOp<UnrealizedConversionCastOp>())
    v = castOp.getOperand(0);
  if (auto castOp = v.getDefiningOp<emitc::CastOp>())
    v = castOp.getOperand();
  return v;
}

bool isTileLikeValue(Value v) {
  auto ot = dyn_cast<emitc::OpaqueType>(v.getType());
  if (!ot)
    return false;
  StringRef s = ot.getValue();
  return s.contains("Tile<") || s.contains("ConvTile<");
}

//===----------------------------------------------------------------------===//
// pto.load_scalar / pto.store_scalar lowering -> ptr[offset]
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// pto.tabs lowering -> TABS(dst, src)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// pto.tadd lowering -> TADD(dst, src0, src1)
//===----------------------------------------------------------------------===//

// Historical hook for pre-annotated TNotify release drains. The automatic
// MemoryConsistency analysis pass that used to produce these attrs has been
// removed from the default pipeline; keeping the lowering hook is harmless for
// hand-authored or legacy IR that already carries the internal attrs.

// pto.declare_local_array -> emitc.variable of !emitc.array<...>.
// Renders as `T a[D1][D2]...;` in the emitted C++.

// pto.local_array_get %a[%i0, %i1, ...] -> scalar snapshot.
// Materialize the subscript read immediately so the MLIR SSA result keeps its
// value even if a later pto.local_array_set mutates the same backing array slot.

// pto.local_array_set %a[%i0, %i1, ...], %v -> emitc.assign to subscript slot.
// The C++ emitter prints this as `a[i0][i1]... = v;`. As above, adaptor values
// are already target-typed; pass them through directly.

// pto.declare_struct -> emitc.variable of !emitc.opaque<"PtoStruct_...">.
// Renders as `PtoStruct_X s;` in the emitted C++.

// The EmitC *value* type of a struct field, i.e. the type an lvalue to that
// field wraps. A nested struct is spelled directly here rather than going
// through the converter, which would hand back the pointer form used for
// passing whole structs around — a field lives inside its parent's storage and
// is reached with `.`, not through another pointer.
// The EmitC *value* type of a struct field, i.e. the type an lvalue to that
// field wraps. A nested struct is spelled directly here rather than going
// through the converter, which would hand back the pointer form used for
// passing whole structs around — a field lives inside its parent's storage and
// is reached with `.`, not through another pointer.
Type getStructFieldValueType(const TypeConverter *tc, Type fieldPtoTy) {
  if (auto st = dyn_cast<pto::StructType>(fieldPtoTy)) {
    return emitc::OpaqueType::get(st.getContext(), getStructTypeName(st));
  }
  return tc->convertType(fieldPtoTy);
}


FailureOr<Type> getStructMemberFieldType(mlir::pto::StructType structTy,
                                                int64_t index,
                                                const TypeConverter *tc) {
  if (index < 0 ||
      static_cast<unsigned>(index) >= structTy.getFieldTypes().size()) {
    return failure();
  }
  return getStructFieldValueType(
      tc, structTy.getFieldType(static_cast<unsigned>(index)));
}

FailureOr<Value> getStructAdaptorValue(ValueRange operands) {
  if (operands.empty()) {
    return failure();
  }
  return operands.front();
}

// Build the `s.fA.fB...` member-access chain for a constant struct path and
// return the final lvalue. `rootPtoTy` is the PTO struct type, walked in
// parallel to look up field types per step.
//
// Every step is an `emitc.member`, which requires an lvalue operand and yields
// an lvalue result, so the chain stays in lvalue form throughout — that is what
// makes a write land in the struct rather than in a copy of it.
//
// `root` is the converted struct, i.e. a pointer. Two shapes reach here:
//   - a local declared by pto.declare_struct, whose pointer is an address-of;
//     that is unwrapped back to the variable so the access prints as `s.f0`.
//   - any other pointer, notably a function argument. `emitc.member_of_ptr`
//     needs an lvalue *holding* the pointer rather than the raw pointer, so it
//     is parked in a variable first and the access prints as `p->f0`.
FailureOr<Value> buildStructMemberChain(
    ConversionPatternRewriter &rewriter, Location loc, const TypeConverter *tc,
    Value root, mlir::pto::StructType rootPtoTy, llvm::ArrayRef<int64_t> path) {
  Value ptr = peelUnrealized(root);

  // lvalue of the struct itself when we can name it; otherwise an lvalue
  // holding the pointer, consumed by the first member_of_ptr step.
  Value structLValue;
  Value ptrSlot;
  auto applyOp = ptr.getDefiningOp<emitc::ApplyOp>();
  if (applyOp && applyOp.getApplicableOperator() == "&") {
    structLValue = applyOp.getOperand();
  } else {
    if (!isa<emitc::PointerType>(ptr.getType())) {
      return failure();
    }
    ptrSlot = rewriter
                  .create<emitc::VariableOp>(
                      loc, ptr.getType(),
                      emitc::OpaqueAttr::get(rewriter.getContext(), ""))
                  .getResult();
    rewriter.create<emitc::AssignOp>(loc, ptrSlot, ptr);
  }

  Type curPtoTy = rootPtoTy;
  for (int64_t idx : path) {
    auto st = dyn_cast<mlir::pto::StructType>(curPtoTy);
    if (!st) {
      return failure();
    }
    FailureOr<Type> fieldTy = getStructMemberFieldType(st, idx, tc);
    if (failed(fieldTy)) {
      return failure();
    }
    Type resultTy = *fieldTy;
    auto name = rewriter.getStringAttr("f" + std::to_string(idx));
    // Only the first step off a bare pointer uses `->`; from there on the
    // chain is walking storage we can name, so it is all `.`.
    structLValue =
        structLValue
            ? rewriter.create<emitc::MemberOp>(loc, resultTy, name, structLValue)
                  .getResult()
            : rewriter
                  .create<emitc::MemberOfPtrOp>(loc, resultTy, name, ptrSlot)
                  .getResult();
    curPtoTy = st.getFieldType(static_cast<unsigned>(idx));
  }
  return structLValue;
}

/// Resolve the struct operand and member-access chain shared by struct_get
// and struct_set; returns the member lvalue or match failure.
FailureOr<Value>
resolveStructMember(Operation *op, ValueRange adaptorOperands, Type structPtoTy,
                    ArrayRef<int64_t> path, ConversionPatternRewriter &rewriter,
                    const TypeConverter *typeConverter) {
  FailureOr<Value> structValue = getStructAdaptorValue(adaptorOperands);
  const bool hasInvalidStructOperand =
      failed(structValue) || op->getNumOperands() == 0;
  if (hasInvalidStructOperand)
    return rewriter.notifyMatchFailure(op, "expected struct operand");
  auto structTy = dyn_cast<mlir::pto::StructType>(structPtoTy);
  if (!structTy)
    return rewriter.notifyMatchFailure(op, "expected !pto.struct operand");
  return buildStructMemberChain(rewriter, op->getLoc(), typeConverter,
                                *structValue, structTy, path);
}

// pto.struct_get %s[i, j, ...] -> `s.fi.fj...`. The verifier guarantees the path
// ends on a scalar, so the member lvalue is read with emitc.load. That load is
// materialized into its own C++ variable, which is what gives the SSA result
// value semantics: it keeps its value even if a later pto.struct_set writes the
// same field (mirrors pto.local_array_get).

// pto.struct_set %s[i, j, ...], %v -> `s.fi.fj... = v;`.

FailureOr<Value> buildCollectiveParallelGroup(
    ConversionPatternRewriter &rewriter, Location loc,
    ArrayRef<Value> groupGTs, int64_t root) {
  if (groupGTs.empty())
    return failure();

  auto firstTy = dyn_cast<emitc::OpaqueType>(groupGTs.front().getType());
  if (!firstTy)
    return failure();

  auto *ctx = rewriter.getContext();
  auto arrayTy = emitc::ArrayType::get({static_cast<int64_t>(groupGTs.size())},
                                       firstTy);
  auto groupArray = cast<TypedValue<emitc::ArrayType>>(
      rewriter
          .create<emitc::VariableOp>(loc, getEmitCVariableResultType(arrayTy),
                                     emitc::OpaqueAttr::get(ctx, "{}"))
          .getResult());

  auto indexTy = emitc::OpaqueType::get(ctx, "int");
  for (auto [idx, groupVal] : llvm::enumerate(groupGTs)) {
    Value idxVal =
        makeEmitCIntConstant(rewriter, loc, indexTy, static_cast<int64_t>(idx));
    Value slot =
        rewriter.create<emitc::SubscriptOp>(loc, groupArray, ValueRange{idxVal})
            .getResult();
    rewriter.create<emitc::AssignOp>(loc, slot, groupVal);
  }

  std::string pgTypeStr =
      (Twine("pto::comm::ParallelGroup<") + firstTy.getValue() + ">").str();
  auto pgTy = emitc::OpaqueType::get(ctx, pgTypeStr);
  Value sizeVal = makeEmitCIntConstant(rewriter, loc, indexTy,
                                       static_cast<int64_t>(groupGTs.size()));
  Value rootVal = makeEmitCIntConstant(rewriter, loc, indexTy, root);
  return rewriter
      .create<emitc::CallOpaqueOp>(
          loc, TypeRange{pgTy}, (Twine(pgTypeStr) + "::Create").str(),
          ArrayAttr{}, ArrayAttr{}, ValueRange{groupArray, sizeVal, rootVal})
      .getResult(0);
}

FailureOr<Value> buildCommGlobalTensorValue(
    ConversionPatternRewriter &rewriter, Location loc, Value originalValue,
    Value emittedValue, Operation *anchor) {
  Value value = peelUnrealized(emittedValue);
  if (isEmitCGlobalTensorLikeType(value.getType()))
    return value;
  return failure();
}

FailureOr<Value> buildCommTileValue(ConversionPatternRewriter &rewriter,
                                           Location loc, Value originalValue,
                                           Value emittedValue) {
  Value value = peelUnrealized(emittedValue);
  if (auto opaqueTy = dyn_cast<emitc::OpaqueType>(value.getType())) {
    StringRef typeStr = opaqueTy.getValue();
    if (typeStr.contains("Tile<") || typeStr.contains("ConvTile<"))
      return value;
  }
  return buildAsyncScratchTileValue(rewriter, loc, originalValue, emittedValue);
}

void emitTNotifyReleaseActions(ConversionPatternRewriter &rewriter,
                                      Location loc, bool drainMte2,
                                      bool drainMte3) {
  if (drainMte2)
    emitPipeBarrier(rewriter, loc, "PIPE_MTE2");
  if (drainMte3)
    emitPipeBarrier(rewriter, loc, "PIPE_MTE3");
}

std::string notifyOpTok(pto::NotifyOp op) {
  switch (op) {
  case pto::NotifyOp::AtomicAdd:
    return "pto::comm::NotifyOp::AtomicAdd";
  case pto::NotifyOp::Set:
    return "pto::comm::NotifyOp::Set";
  }
  return "pto::comm::NotifyOp::Set";
}

std::string reduceOpTok(pto::ReduceOp op) {
  switch (op) {
  case pto::ReduceOp::Sum:
    return "pto::comm::ReduceOp::Sum";
  case pto::ReduceOp::Max:
    return "pto::comm::ReduceOp::Max";
  case pto::ReduceOp::Min:
    return "pto::comm::ReduceOp::Min";
  }
  return "pto::comm::ReduceOp::Sum";
}

std::string waitCmpTok(pto::WaitCmp cmp) {
  switch (cmp) {
  case pto::WaitCmp::EQ:
    return "pto::comm::WaitCmp::EQ";
  case pto::WaitCmp::NE:
    return "pto::comm::WaitCmp::NE";
  case pto::WaitCmp::GT:
    return "pto::comm::WaitCmp::GT";
  case pto::WaitCmp::GE:
    return "pto::comm::WaitCmp::GE";
  case pto::WaitCmp::LT:
    return "pto::comm::WaitCmp::LT";
  case pto::WaitCmp::LE:
    return "pto::comm::WaitCmp::LE";
  }
  return "pto::comm::WaitCmp::EQ";
}


void emitConservativeGmFencePipeDrains(
    ConversionPatternRewriter &rewriter, Location loc) {
  emitPipeBarrier(rewriter, loc, "PIPE_MTE2");
  emitPipeBarrier(rewriter, loc, "PIPE_MTE3");
  emitPipeBarrier(rewriter, loc, "PIPE_FIX");
}

void emitDsbDdr(ConversionPatternRewriter &rewriter, Location loc) {
  auto *ctx = rewriter.getContext();
  auto args = rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, "DSB_DDR")});
  rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "dsb", args,
                                       ArrayAttr{}, ValueRange{});
}

void emitPipeBarrier(ConversionPatternRewriter &rewriter, Location loc,
                            StringRef pipeTok) {
  auto *ctx = rewriter.getContext();
  auto args = rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, pipeTok)});
  rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "pipe_barrier", args,
                                       ArrayAttr{}, ValueRange{});
}

std::string getAutoSyncTailModeToken(Operation *op) {
  if (op) {
    if (auto hintAttr = op->getAttrOfType<StringAttr>(kAutoSyncTailHintAttr)) {
      if (hintAttr.getValue() == kAutoSyncTailPolicyBarrierAll)
        return kAutoSyncTailModeBarrierAllToken.str();
      if (hintAttr.getValue() == kAutoSyncTailPolicyMte3ToSEvent0)
        return kAutoSyncTailModeMte3ToSEvent0Token.str();
    }
  }

  auto func = op ? op->getParentOfType<func::FuncOp>() : func::FuncOp();
  if (!func)
    return kAutoSyncTailModeBarrierAllToken.str();

  auto hintAttr = func->getAttrOfType<StringAttr>(kAutoSyncTailHintAttr);
  if (!hintAttr)
    return kAutoSyncTailModeBarrierAllToken.str();

  if (hintAttr.getValue() == kAutoSyncTailPolicyBarrierAll)
    return kAutoSyncTailModeBarrierAllToken.str();
  if (hintAttr.getValue() == kAutoSyncTailPolicyMte3ToSEvent0)
    return kAutoSyncTailModeMte3ToSEvent0Token.str();

  // Fallback to the conservative behavior when seeing unknown policies.
  return kAutoSyncTailModeBarrierAllToken.str();
}

bool isInVectorKernel(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isa<pto::SectionVectorOp>(parent))
      return true;

    auto kernelKindAttr = parent->getAttrOfType<FunctionKernelKindAttr>(
        FunctionKernelKindAttr::name);
    if (kernelKindAttr)
      return kernelKindAttr.getKernelKind() == FunctionKernelKind::Vector;
  }
  return false;
}


void emitInvalidateGmCacheAll(ConversionPatternRewriter &rewriter,
                                     Location loc) {
  auto *ctx = rewriter.getContext();
  auto args = rewriter.getArrayAttr({
      emitc::OpaqueAttr::get(ctx, "(__gm__ void*)0"),
      emitc::OpaqueAttr::get(ctx, "cache_line_t::ENTIRE_DATA_CACHE"),
  });
  rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "dcci", args,
                                       ArrayAttr{}, ValueRange{});
}

void emitInvalidateGmCacheSingleLine(ConversionPatternRewriter &rewriter,
                                            Location loc, Value addr) {
  rewriter.create<emitc::CallOpaqueOp>(
      loc, TypeRange{}, "PTOAS__DCCI_SINGLE_CACHE_LINE",
      ArrayAttr{}, ArrayAttr{}, ValueRange{addr});
}

Type getPointerLikeElementType(Type type) {
  if (auto ptrTy = dyn_cast<pto::PtrType>(type))
    return ptrTy.getElementType();
  if (auto memTy = dyn_cast<MemRefType>(type))
    return memTy.getElementType();
  return Type();
}

bool isGmCmoSpace(pto::AddressSpace space) {
  return space == pto::AddressSpace::GM || space == pto::AddressSpace::Zero;
}



} // namespace pto
} // namespace mlir

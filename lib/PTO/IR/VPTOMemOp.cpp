// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOMemOp.cpp - VPTO memory/atomic/misc ops -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VPTOInternal.h"

using namespace mlir;
using namespace mlir::pto;

static LogicalResult verifyVPTOScalarAccessTypes(Operation *op, Type ptrTy,
                                                 Type valueTy,
                                                 StringRef opNameForDiag) {
  Type elemTy;
  if (auto pty = dyn_cast<PtrType>(ptrTy)) {
    elemTy = pty.getElementType();
  } else if (auto memTy = dyn_cast<MemRefType>(ptrTy)) {
    elemTy = memTy.getElementType();
  } else {
    return op->emitOpError() << "expects " << opNameForDiag
                             << " pointer operand to be !pto.ptr or memref";
  }

  if (valueTy != elemTy) {
    return op->emitOpError() << "expects " << opNameForDiag
                             << " value type to match pointer element type";
  }
  return success();
}

LogicalResult SimtLaunchOp::verify() {
  if (auto parentFunc = (*this)->getParentOfType<func::FuncOp>()) {
    if (parentFunc->hasAttr(pto::kPTOSimtEntryAttrName)) {
      return emitOpError()
             << "must not appear inside a function marked with '"
             << pto::kPTOSimtEntryAttrName
             << "'; launch the SIMT entry from an outer non-simt function";
    }
  }

  func::FuncOp callee =
      SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(*this, getCalleeAttr());
  if (!callee) {
    return emitOpError() << "'" << getCalleeAttr().getValue()
                         << "' does not reference a valid function";
  }

  if (!callee->hasAttr(pto::kPTOSimtEntryAttrName)) {
    return emitOpError() << "callee '" << getCalleeAttr().getValue()
                         << "' must be marked with '"
                         << pto::kPTOSimtEntryAttrName << "'";
  }

  FunctionType calleeType = callee.getFunctionType();
  if (!calleeType.getResults().empty()) {
    return emitOpError("requires a callee with no results");
  }

  if (calleeType.getNumInputs() != getArgs().size()) {
    return emitOpError("incorrect number of operands for callee");
  }

  for (auto [index, argType, operand] :
       llvm::enumerate(calleeType.getInputs(), getArgs())) {
    if (argType != operand.getType()) {
      return emitOpError("operand type mismatch: expected operand type ")
             << argType << ", but provided " << operand.getType()
             << " for operand number " << index;
    }
  }
  return success();
}

static bool isVector2F16OrBF16Type(Type type) {
  return isVector2Of(type, [](Type elem) {
    return elem.isF16() || elem.isBF16();
  });
}

static bool isSupportedAtomicScalarType(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    return intType.getWidth() == mlir::pto::kValue32 || intType.getWidth() == 64;
  }
  return type.isF16() || type.isBF16() || type.isF32() ||
         isVector2F16OrBF16Type(type);
}

static LogicalResult verifyAtomicCommon(Operation *op, Value ptr, Type valueType,
                                        Type resultType, bool bitwise,
                                        Attribute signednessAttr) {
  if (!isSupportedAtomicScalarType(valueType)) {
    return op->emitOpError() << "requires i32, i64, f16, bf16, f32, "
                                "vector<2xf16> or vector<2xbf16> atomic value type";
  }
  if (resultType != valueType) {
    return op->emitOpError()
           << "requires atomic result type to match value type";
  }
  auto ptrTy = dyn_cast<PtrType>(ptr.getType());
  if (!ptrTy) {
    return op->emitOpError() << "requires !pto.ptr pointer operand";
  }
  if (ptrTy.getElementType() != valueType) {
    return op->emitOpError()
           << "requires atomic value type to match pointer element type";
  }
  AddressSpace addressSpace = ptrTy.getMemorySpace().getAddressSpace();
  if (addressSpace != AddressSpace::GM && addressSpace != AddressSpace::VEC) {
    return op->emitOpError() << "requires GM or UB pointer";
  }
  if (addressSpace == AddressSpace::VEC && valueType.isInteger(mlir::pto::kValue64)) {
    return op->emitOpError() << "does not support i64 UB-space atomics";
  }
  auto intType = dyn_cast<IntegerType>(valueType);
  if (bitwise) {
    if (!intType) {
      return op->emitOpError() << "requires integer type for bitwise atomics";
    }
    if (addressSpace == AddressSpace::VEC && intType.getWidth() == mlir::pto::kValue64) {
      return op->emitOpError() << "does not support i64 UB-space bitwise atomics";
    }
  }
  if (signednessAttr && !intType) {
    return op->emitOpError()
           << "does not accept signedness for floating-point atomics";
  }
  if (isVector2F16OrBF16Type(valueType)) {
    if (!isInsideSimtExecutionScope(op)) {
      return op->emitOpError() << "requires packed atomics to be inside a "
                                    "pto.simt_entry function or pto.section.simt on beta.1";
    }
    if (!op->getResult(0).use_empty()) {
      return op->emitOpError() << "does not support using the old value result for "
                                    "packed atomics on beta.1; leave the result unused";
    }
  }
  return success();
}

static LogicalResult verifyLdgStgAccess(Operation *op, Type ptrType,
                                        Type valueType) {
  auto ptrTy = dyn_cast<PtrType>(ptrType);
  if (!ptrTy) {
    return op->emitOpError() << "requires !pto.ptr operand";
  }
  if (ptrTy.getMemorySpace().getAddressSpace() != AddressSpace::GM) {
    return op->emitOpError() << "requires GM pointer";
  }

  if (auto intType = dyn_cast<IntegerType>(valueType)) {
    unsigned width = intType.getWidth();
    if (width == mlir::pto::kValue8 || width == 16 || width == 32 || width == 64) {
      return success();
    }
  }
  if (valueType.isF16() || valueType.isBF16() || valueType.isF32() ||
      valueType.isF64()) {
    return success();
  }
  if (pto::isPTOFloat8Type(valueType) || pto::isPTOHiFloat8Type(valueType)) {
    return success();
  }
  if (pto::isPTOPackedLdgStgVectorType(valueType)) {
    return success();
  }

  return op->emitOpError()
         << "currently supports 8/16/32/64-bit integer, "
            "f16/bf16/f32/f64/fp8/hif8, "
            "packed vector<2xT> (T = f16/bf16/f32/i8/i16/i32), "
            "packed vector<2/4/8xfp8>, and !pto.hif8x2 value type";
}

static LogicalResult verifyLdStDevAccess(Operation *op, Type ptrType,
                                         Type valueType) {
  if (op->hasAttr("l1cache") || op->hasAttr("l2cache")) {
    return op->emitOpError()
           << "does not accept l1cache or l2cache policy attributes";
  }

  auto ptrTy = dyn_cast<PtrType>(ptrType);
  if (!ptrTy) {
    return op->emitOpError() << "requires !pto.ptr operand";
  }
  if (ptrTy.getMemorySpace().getAddressSpace() != AddressSpace::GM) {
    return op->emitOpError() << "requires GM pointer";
  }

  auto intType = dyn_cast<IntegerType>(valueType);
  if (!intType || (intType.getWidth() != mlir::pto::kValue8 && intType.getWidth() != 16 &&
                   intType.getWidth() != mlir::pto::kValue32 && intType.getWidth() != 64)) {
    return op->emitOpError() << "supports only i8, i16, i32 or i64 values";
  }

  if (isInsideSimtExecutionScope(op)) {
    return op->emitOpError()
           << "must be outside pto.simt_entry functions and pto.section.simt";
  }
  auto funcOp = op->getParentOfType<func::FuncOp>();
  if (!funcOp || !pto::isPTOEntryFunction(funcOp)) {
    return op->emitOpError()
           << "requires an enclosing ordinary AICore entry function";
  }
  return success();
}

LogicalResult PTOLoadOp::verify() {
  if (failed(verifyVPTOScalarAccessTypes(getOperation(), getPtr().getType(),
                                         getValue().getType(), "load"))) {
    return failure();
  }
  return success();
}

LogicalResult PTOStoreOp::verify() {
  if (failed(verifyVPTOScalarAccessTypes(getOperation(), getPtr().getType(),
                                         getValue().getType(), "store"))) {
    return failure();
  }
  return success();
}

// Batch7: MemOp 双胞胎共用校验
static bool isSupportedPredicateLoadDist(llvm::StringRef dist);
static bool isSupportedPredicateStoreDist(llvm::StringRef dist);

template <typename OpTy>
static LogicalResult verifyPTOScalarAccessOp(OpTy op, llvm::StringRef tag) {
  if (failed(verifyVPTOScalarAccessTypes(op.getOperation(), op.getPtr().getType(),
                                         op.getValue().getType(), tag)) ||
      failed(verifyLdgStgAccess(op.getOperation(), op.getPtr().getType(),
                                op.getValue().getType()))) {
    return failure();
  }
  if (!isInsideSimtExecutionScope(op.getOperation())) {
    return op.emitOpError()
           << "must be inside a pto.simt_entry function or pto.section.simt";
  }
  return success();
}

template <typename OpTy>
static LogicalResult verifyGatherOffsetTypes(OpTy op, VRegType &offsetsType,
                                             VRegType &resultType,
                                             IntegerType &offsetsElemType) {
  offsetsType = dyn_cast<VRegType>(op.getOffsets().getType());
  resultType = dyn_cast<VRegType>(op.getResult().getType());
  if (!offsetsType || !resultType) {
    return op.emitOpError("offsets and result must be !pto.vreg<...>");
  }
  offsetsElemType = dyn_cast<IntegerType>(offsetsType.getElementType());
  if (!offsetsElemType) {
    return op.emitOpError("offset vector must use integer element type");
  }
  return success();
}

template <typename OpTy, typename OffsetCheck>
static LogicalResult verifyPredicateLoadOp(OpTy op, OffsetCheck offsetCheck) {
  if (!isBufferLike(op.getSource().getType())) {
    return op.emitOpError("requires a pointer-like source");
  }
  if (failed(verifyMaskTypeLike(op, op.getResult().getType(), "result type"))) {
    return failure();
  }
  if (classifyMemoryRole(op.getSource().getType()) == MemoryRole::GM) {
    return op.emitOpError("requires a UB-backed source");
  }
  if (failed(offsetCheck())) {
    return failure();
  }
  if (!isSupportedPredicateLoadDist(op.getDist())) {
    return op.emitOpError("requires predicate load dist to be NORM, US, or DS");
  }
  if (op.getUpdatedBase() &&
      op.getUpdatedBase().getType() != op.getSource().getType()) {
    return op.emitOpError("requires updated base result to match base type");
  }
  return success();
}

template <typename OpTy, typename OffsetCheck>
static LogicalResult verifyPredicateStoreOp(OpTy op, OffsetCheck offsetCheck) {
  if (failed(verifyMaskTypeLike(op, op.getValue().getType(), "value type"))) {
    return failure();
  }
  if (!isBufferLike(op.getDestination().getType())) {
    return op.emitOpError("requires a pointer-like destination");
  }
  if (classifyMemoryRole(op.getDestination().getType()) == MemoryRole::GM) {
    return op.emitOpError("requires a UB-backed destination");
  }
  if (failed(offsetCheck())) {
    return failure();
  }
  if (!isSupportedPredicateStoreDist(op.getDist())) {
    return op.emitOpError("requires predicate store dist to be NORM or PK");
  }
  if (op.getUpdatedBase() &&
      op.getUpdatedBase().getType() != op.getDestination().getType()) {
    return op.emitOpError("requires updated base result to match base type");
  }
  return success();
}

template <typename OpTy>
static LogicalResult verifyUbStoreAlignBase(OpTy op) {
  if (failed(verifyStoreAlignChain(op.getAlignIn(), op, "align_in type")) ||
      failed(verifyVRegTypeLike(op, op.getValue().getType(), "value type")) ||
      failed(verifyAlignTypeLike(op, op.getAlignOut().getType(),
                                 "align_out type"))) {
    return failure();
  }
  if (!isBufferLike(op.getBase().getType())) {
    return op.emitOpError("requires a pointer-like base");
  }
  if (classifyMemoryRole(op.getBase().getType()) == MemoryRole::GM) {
    return op.emitOpError("requires a UB-backed base");
  }
  return success();
}


LogicalResult PTOLdgOp::verify() {
  return verifyPTOScalarAccessOp(*this, "ldg");
}

LogicalResult PTOStgOp::verify() {
  return verifyPTOScalarAccessOp(*this, "stg");
}

LogicalResult PTOLdDevOp::verify() {
  if (failed(verifyVPTOScalarAccessTypes(getOperation(), getPtr().getType(),
                                         getValue().getType(), "ld_dev"))) {
    return failure();
  }
  return verifyLdStDevAccess(getOperation(), getPtr().getType(),
                             getValue().getType());
}

LogicalResult PTOStDevOp::verify() {
  if (failed(verifyVPTOScalarAccessTypes(getOperation(), getPtr().getType(),
                                         getValue().getType(), "st_dev"))) {
    return failure();
  }
  return verifyLdStDevAccess(getOperation(), getPtr().getType(),
                             getValue().getType());
}

LogicalResult AtomicCasOp::verify() {
  if (getCompare().getType() != getValue().getType()) {
    return emitOpError() << "requires compare and value types to match";
  }
  return verifyAtomicCommon(getOperation(), getPtr(), getValue().getType(),
                            getOld().getType(), /*bitwise=*/false,
                            getSignednessAttr());
}

LogicalResult AtomicExchOp::verify() {
  return verifyAtomicCommon(getOperation(), getPtr(), getValue().getType(),
                            getOld().getType(), /*bitwise=*/false,
                            getSignednessAttr());
}

LogicalResult AtomicAddOp::verify() {
  return verifyAtomicCommon(getOperation(), getPtr(), getValue().getType(),
                            getOld().getType(), /*bitwise=*/false,
                            getSignednessAttr());
}

LogicalResult AtomicSubOp::verify() {
  return verifyAtomicCommon(getOperation(), getPtr(), getValue().getType(),
                            getOld().getType(), /*bitwise=*/false,
                            getSignednessAttr());
}

LogicalResult AtomicMinOp::verify() {
  return verifyAtomicCommon(getOperation(), getPtr(), getValue().getType(),
                            getOld().getType(), /*bitwise=*/false,
                            getSignednessAttr());
}

LogicalResult AtomicMaxOp::verify() {
  return verifyAtomicCommon(getOperation(), getPtr(), getValue().getType(),
                            getOld().getType(), /*bitwise=*/false,
                            getSignednessAttr());
}

LogicalResult AtomicAndOp::verify() {
  return verifyAtomicCommon(getOperation(), getPtr(), getValue().getType(),
                            getOld().getType(), /*bitwise=*/true,
                            getSignednessAttr());
}

LogicalResult AtomicOrOp::verify() {
  return verifyAtomicCommon(getOperation(), getPtr(), getValue().getType(),
                            getOld().getType(), /*bitwise=*/true,
                            getSignednessAttr());
}

LogicalResult AtomicXorOp::verify() {
  return verifyAtomicCommon(getOperation(), getPtr(), getValue().getType(),
                            getOld().getType(), /*bitwise=*/true,
                            getSignednessAttr());
}

void PTOLoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getPtrMutable());
}

void PTOStoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getPtrMutable());
}

void PTOLdgOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getPtrMutable());
}

void PTOStgOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getPtrMutable());
}

void PTOLdDevOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getPtrMutable());
}

void PTOStDevOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getPtrMutable());
}

template <typename OpTy>
static void getAtomicEffects(
    OpTy op,
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &op.getPtrMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &op.getPtrMutable());
}

void AtomicCasOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getAtomicEffects(*this, effects);
}
void AtomicExchOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getAtomicEffects(*this, effects);
}
void AtomicAddOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getAtomicEffects(*this, effects);
}
void AtomicSubOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getAtomicEffects(*this, effects);
}
void AtomicMinOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getAtomicEffects(*this, effects);
}
void AtomicMaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getAtomicEffects(*this, effects);
}
void AtomicAndOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getAtomicEffects(*this, effects);
}
void AtomicOrOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getAtomicEffects(*this, effects);
}
void AtomicXorOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getAtomicEffects(*this, effects);
}

bool isSupportedPredicatePattern(StringRef pattern) {
  return pattern == "PAT_ALL" || pattern == "PAT_VL1" || pattern == "PAT_VL2" ||
         pattern == "PAT_VL3" || pattern == "PAT_VL4" || pattern == "PAT_VL8" ||
         pattern == "PAT_VL16" || pattern == "PAT_VL32" ||
         pattern == "PAT_VL64" || pattern == "PAT_VL128" ||
         pattern == "PAT_M3" || pattern == "PAT_M4" || pattern == "PAT_H" ||
         pattern == "PAT_Q" || pattern == "PAT_ALLF";
}

static bool isSupportedPredicateLoadDist(StringRef dist) {
  return dist == "NORM" || dist == "US" || dist == "DS";
}

static bool isSupportedPredicateStoreDist(StringRef dist) {
  return dist == "NORM" || dist == "PK";
}

static bool isSupportedSprToken(StringRef spr) { return spr == "AR"; }

void Vgather2Op::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

static bool isUnsignedOrSignlessIntegerOfWidth(Type type, unsigned width) {
  auto intType = dyn_cast<IntegerType>(type);
  return intType && intType.getWidth() == width && !intType.isSigned();
}

static bool isSameVgather2IntegerSemantics(IntegerType sourceType,
                                           IntegerType resultType) {
  if (!sourceType || !resultType ||
      sourceType.getWidth() != resultType.getWidth()) {
    return false;
  }
  if (sourceType.isUnsigned()) {
    return resultType.isUnsigned();
  }
  return !resultType.isUnsigned();
}

static bool isVgather2B8ResultType(IntegerType sourceType,
                                   IntegerType resultType) {
  if (!sourceType || !resultType || sourceType.getWidth() != mlir::pto::kValue8 ||
      resultType.getWidth() != mlir::pto::kValue16) {
    return false;
  }
  if (sourceType.isUnsigned()) {
    return resultType.isUnsigned();
  }
  return !resultType.isUnsigned();
}

static LogicalResult resolveVgather2WidthInfo(
    Operation *op, Type sourceElemType, Type resultElemType,
    unsigned &expectedOffsetWidth, StringRef &expectedMaskGranularity,
    int64_t &expectedLanes) {
  unsigned sourceElemWidth = getPTOStorageElemBitWidth(sourceElemType);
  if (sourceElemWidth == mlir::pto::kValue8 && isa<IntegerType>(sourceElemType)) {
    if (!isVgather2B8ResultType(cast<IntegerType>(sourceElemType),
                                dyn_cast<IntegerType>(resultElemType))) {
      return op->emitOpError(
          "8-bit gather requires i8/ui8 source and matching i16/ui16 result");
    }
    expectedOffsetWidth = mlir::pto::kValue16;
    expectedMaskGranularity = "b16";
    expectedLanes = mlir::pto::kValue128;
    return success();
  }
  if (sourceElemWidth == mlir::pto::kValue16) {
    if (auto sourceInt = dyn_cast<IntegerType>(sourceElemType)) {
      if (!isSameVgather2IntegerSemantics(
              sourceInt, dyn_cast<IntegerType>(resultElemType))) {
        return op->emitOpError(
            "16-bit integer gather requires matching i16/ui16 result");
      }
    } else if (!(sourceElemType.isF16() || sourceElemType.isBF16()) ||
               sourceElemType != resultElemType) {
      return op->emitOpError(
          "16-bit gather requires i16/ui16/f16/bf16 source and matching result");
    }
    expectedOffsetWidth = mlir::pto::kValue16;
    expectedMaskGranularity = "b16";
    expectedLanes = mlir::pto::kValue128;
    return success();
  }
  if (sourceElemWidth == mlir::pto::kValue32) {
    if (auto sourceInt = dyn_cast<IntegerType>(sourceElemType)) {
      if (!isSameVgather2IntegerSemantics(
              sourceInt, dyn_cast<IntegerType>(resultElemType))) {
        return op->emitOpError(
            "32-bit integer gather requires matching i32/ui32 result");
      }
    } else if (!sourceElemType.isF32() || sourceElemType != resultElemType) {
      return op->emitOpError(
          "32-bit gather requires i32/ui32/f32 source and matching result");
    }
    expectedOffsetWidth = mlir::pto::kValue32;
    expectedMaskGranularity = "b32";
    expectedLanes = mlir::pto::kValue64;
    return success();
  }
  return op->emitOpError(
      "requires source element type i8/ui8/i16/ui16/i32/ui32/f16/bf16/f32");
}

LogicalResult Vgather2Op::verify() {
  if (!isBufferLike(getSource().getType())) {
    return emitOpError("requires a pointer-like source");
  }
  MemoryRole sourceRole = classifyMemoryRole(getSource().getType());
  if (sourceRole == MemoryRole::GM) {
    return emitOpError("requires a UB-backed source");
  }
  VRegType offsetsType, resultType;
  IntegerType offsetsElemType;
  if (failed(verifyGatherOffsetTypes(*this, offsetsType, resultType,
                                     offsetsElemType))) {
    return failure();
  }
  if (offsetsType.getElementCount() != resultType.getElementCount()) {
    return emitOpError("offset and result vectors must have the same element count");
  }
  Type sourceElemType = getBufferElementType(getSource().getType());
  Type resultElemType = resultType.getElementType();
  unsigned resultElemWidth = getPTOStorageElemBitWidth(resultElemType);
  unsigned expectedOffsetWidth = 0;
  StringRef expectedMaskGranularity;
  int64_t expectedLanes = 0;
  if (failed(resolveVgather2WidthInfo(*this, sourceElemType, resultElemType,
                                      expectedOffsetWidth,
                                      expectedMaskGranularity, expectedLanes))) {
    return failure();
  }
  if (resultElemWidth != mlir::pto::kValue16 && resultElemWidth != 32) {
    return emitOpError("result element type must be 16-bit or 32-bit");
  }
  if (resultType.getElementCount() != expectedLanes) {
    return emitOpError() << "expects result type "
                         << formatVRegType(expectedLanes, resultElemType);
  }
  if (!isUnsignedOrSignlessIntegerOfWidth(offsetsElemType, expectedOffsetWidth)) {
    return emitOpError() << "requires ui" << expectedOffsetWidth << "/i"
                         << expectedOffsetWidth << " offset vector elements";
  }
  if (failed(verifyMaskTypeWithGranularityLike(
          getOperation(), getMask().getType(), "mask type",
          expectedMaskGranularity))) {
    return failure();
  }
  return success();
}

void VgatherbOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

// Shared gather offset-type verification plus the 32-bit offset width
// requirement. Vgather2 deliberately allows variable offset widths, so it
// does not go through this helper.
template <typename OpTy>
static LogicalResult verifyGatherOffsetTypesAndWidth(
    OpTy op, VRegType &offsetsType, VRegType &resultType,
    IntegerType &offsetsElemType) {
  if (failed(verifyGatherOffsetTypes(op, offsetsType, resultType,
                                     offsetsElemType))) {
    return failure();
  }
  if (offsetsElemType.getWidth() != mlir::pto::kValue32) {
    return op.emitOpError("currently requires 32-bit offset vector elements");
  }
  return success();
}

LogicalResult VgatherbOp::verify() {
  if (!isBufferLike(getSource().getType())) {
    return emitOpError("requires a pointer-like source");
  }
  MemoryRole sourceRole = classifyMemoryRole(getSource().getType());
  if (sourceRole == MemoryRole::GM) {
    return emitOpError("requires a UB-backed source");
  }

  if (failed(verifyMaskTypeWithGranularityLike(getOperation(), getMask().getType(),
                                               "mask type", "b32"))) {
    return failure();
  }

  VRegType offsetsType, resultType;
  IntegerType offsetsElemType;
  if (failed(verifyGatherOffsetTypesAndWidth(*this, offsetsType, resultType,
                                             offsetsElemType))) {
    return failure();
  }
  // vgatherb is a 32-byte block gather: each offset addresses one 32-byte block.
  // The offset vector holds VL/32 block addresses (always ui32), while the
  // result vector holds VL/sizeof(T) elements of the data type.  These counts
  // only coincide when sizeof(T)==4 (e.g. f32/i32/ui32).  For smaller types
  // the result has more elements than the offset, which is correct because the
  // hardware interprets the low VL/32 bytes of the offset register as block
  // addresses and gathers VL bytes of data per invocation.
  return success();
}

void Vgather2BcOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult Vgather2BcOp::verify() {
  if (!isBufferLike(getSource().getType())) {
    return emitOpError("requires a pointer-like source");
  }
  if (classifyMemoryRole(getSource().getType()) == MemoryRole::GM) {
    return emitOpError("requires a UB-backed source");
  }
  if (failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type"))) {
    return failure();
  }

  VRegType offsetsType, resultType;
  IntegerType offsetsElemType;
  if (failed(verifyGatherOffsetTypesAndWidth(*this, offsetsType, resultType,
                                             offsetsElemType))) {
    return failure();
  }
  if (offsetsType.getElementCount() != resultType.getElementCount()) {
    return emitOpError("offset and result vectors must have the same element count");
  }
  return success();
}
void VldasOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VldasOp::verify() {
  if (!isBufferLike(getSource().getType())) {
    return emitOpError("requires a pointer-like source");
  }
  if (failed(verifyAlignTypeLike(*this, getResult().getType(), "result type"))) {
    return failure();
  }
  if (classifyMemoryRole(getSource().getType()) == MemoryRole::GM) {
    return emitOpError("requires a UB-backed source");
  }
  return success();
}

LogicalResult SprclrOp::verify() {
  if (!isSupportedSprToken(getSpr())) {
    return emitOpError("requires spr to be \"AR\"");
  }
  if (failed(verifyNestedInVecScope(*this, "pto.sprclr"))) {
    return failure();
  }
  return success();
}

static LogicalResult verifySprStoreCommon(Operation *op, StringRef opName,
                                          StringRef spr, Value destination,
                                          Value offset,
                                          bool requireImmediateOffset) {
  if (!isSupportedSprToken(spr)) {
    return op->emitOpError("requires spr to be \"AR\"");
  }
  if (failed(verifyNestedInVecScope(op, opName))) {
    return failure();
  }
  auto ptrType = dyn_cast<pto::PtrType>(destination.getType());
  if (!ptrType) {
    return op->emitOpError("requires a pointer-like UB destination");
  }
  if (classifyMemoryRole(destination.getType()) != MemoryRole::UB) {
    return op->emitOpError("requires a UB-backed destination");
  }
  auto intType = dyn_cast<IntegerType>(ptrType.getElementType());
  if (!intType || intType.getWidth() != mlir::pto::kValue32 || intType.isSigned()) {
    return op->emitOpError("requires ui32/i32 UB destination element type");
  }
  if (!offset.getType().isInteger(mlir::pto::kValue32)) {
    return op->emitOpError("requires i32 offset");
  }
  if (requireImmediateOffset) {
    APInt offsetValue;
    if (!matchPattern(offset, m_ConstantInt(&offsetValue))) {
      return op->emitOpError("requires constant immediate offset");
    }
    int64_t signedOffset = offsetValue.getSExtValue();
    if (signedOffset < -mlir::pto::kValue128 || signedOffset > 127) {
      return op->emitOpError("requires signed 8-bit immediate offset");
    }
  }
  return success();
}

void SprstiOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult SprstiOp::verify() {
  if (failed(verifySprStoreCommon(getOperation(), "pto.sprsti", getSpr(),
                                  getDestination(), getOffset(),
                                  /*requireImmediateOffset=*/true))) {
    return failure();
  }
  if (getUpdatedBase() &&
      getUpdatedBase().getType() != getDestination().getType()) {
    return emitOpError("requires updated base result to match base type");
  }
  return success();
}

void SprstsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult SprstsOp::verify() {
  if (failed(verifySprStoreCommon(getOperation(), "pto.sprsts", getSpr(),
                                  getDestination(), getOffset(),
                                  /*requireImmediateOffset=*/false))) {
    return failure();
  }
  if (getUpdatedBase() &&
      getUpdatedBase().getType() != getDestination().getType()) {
    return emitOpError("requires updated base result to match base type");
  }
  return success();
}

void VldusOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VldusOp::verify() {
  if (failed(verifyLoadAlignChain(getAlign(), *this, "align type")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result type")) ||
      failed(verifyAlignTypeLike(*this, getUpdatedAlign().getType(),
                                 "updated align type"))) {
    return failure();
  }
  if (!isBufferLike(getSource().getType())) {
    return emitOpError("requires a pointer-like source");
  }
  if (classifyMemoryRole(getSource().getType()) == MemoryRole::GM) {
    return emitOpError("requires a UB-backed source");
  }
  if (static_cast<bool>(getIncrement()) != static_cast<bool>(getUpdatedBase())) {
    return emitOpError(
        "requires increment and updated base result to appear together");
  }
  if (getUpdatedBase() && getUpdatedBase().getType() != getSource().getType()) {
    return emitOpError("requires updated base result to match source type");
  }
  return success();
}

void UvldOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult UvldOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getResult().getType(), "result type"))) {
    return failure();
  }
  if (!isBufferLike(getSource().getType())) {
    return emitOpError("requires a buffer-like source");
  }
  if (classifyMemoryRole(getSource().getType()) == MemoryRole::GM) {
    return emitOpError("requires a UB-backed source");
  }

  auto sourceMemRef = dyn_cast<BaseMemRefType>(getSource().getType());
  if (!sourceMemRef) {
    return success();
  }

  Type sourceElementType = sourceMemRef.getElementType();
  Type vectorElementType = cast<VRegType>(getResult().getType()).getElementType();
  if (sourceElementType != vectorElementType) {
    return emitOpError(
        "requires source element type to match vector element type");
  }
  return success();
}

LogicalResult TensorViewAddrOp::verify() {
  Type srcType = getSrc().getType(); Type dstType = getDst().getType();
  Type elementType; int64_t expectedRank = -1;
  auto gmSpace = pto::AddressSpaceAttr::get(getContext(), pto::AddressSpace::GM);
  if (auto tvType = dyn_cast<pto::TensorViewType>(srcType)) {
    elementType = tvType.getElementType();
    expectedRank = tvType.getRank();
  } else if (auto partType = dyn_cast<pto::PartitionTensorViewType>(srcType)) {
    elementType = partType.getElementType();
    expectedRank = partType.getRank();
  } else if (auto memrefType = dyn_cast<BaseMemRefType>(srcType)) {
    elementType = memrefType.getElementType();
    expectedRank = memrefType.getRank();
    auto srcSpace =
        dyn_cast_or_null<pto::AddressSpaceAttr>(memrefType.getMemorySpace());
    if (srcSpace && srcSpace != gmSpace) {
      return emitOpError("memref source must stay in gm memory space");
    }
  } else {
    return emitOpError(
        "source must be a tensor_view, partition_tensor_view, or memref");
  }
  if (auto dstMemRefType = dyn_cast<BaseMemRefType>(dstType)) {
    if (dstMemRefType.getElementType() != elementType) {
      return emitOpError(
          "memref result element type must match source element type");
    }
    if (dstMemRefType.getRank() != expectedRank) {
      return emitOpError("memref result rank must match source rank");
    }
    auto dstSpace =
        dyn_cast_or_null<pto::AddressSpaceAttr>(dstMemRefType.getMemorySpace());
    if (dstSpace && dstSpace != gmSpace) {
      return emitOpError("memref result must stay in gm memory space");
    }
    return success();
  }
  auto dstPtrType = dyn_cast<pto::PtrType>(dstType);
  if (!dstPtrType) {
    return emitOpError("result must be a memref or !pto.ptr<...>");
  }
  if (dstPtrType.getElementType() != elementType) {
    return emitOpError(
        "pointer result element type must match source element type");
  }
  if (dstPtrType.getMemorySpace() != gmSpace) {
    return emitOpError("pointer result must stay in gm memory space");
  }
  return success();
}

LogicalResult TileBufAddrOp::verify() {
  Type dstType = getDst().getType();
  Type elementType;
  Attribute srcMemorySpace;
  int64_t srcRank = 0;

  if (auto srcTileType = dyn_cast<pto::TileBufType>(getSrc().getType())) {
    elementType = srcTileType.getElementType();
    srcMemorySpace = srcTileType.getMemorySpace();
    srcRank = static_cast<int64_t>(srcTileType.getShape().size());
  } else if (auto srcMemRefType = dyn_cast<BaseMemRefType>(getSrc().getType())) {
    elementType = srcMemRefType.getElementType();
    srcMemorySpace = srcMemRefType.getMemorySpace();
    srcRank = srcMemRefType.getRank();
  } else {
    return emitOpError("source must be a !pto.tile_buf<...> or memref");
  }

  auto srcSpace = dyn_cast_or_null<pto::AddressSpaceAttr>(srcMemorySpace);

  if (auto dstMemRefType = dyn_cast<BaseMemRefType>(dstType)) {
    if (dstMemRefType.getElementType() != elementType) {
      return emitOpError(
          "memref result element type must match tile element type");
    }
    if (dstMemRefType.getRank() != srcRank) {
      return emitOpError("memref result rank must match tile rank");
    }
    auto dstSpace =
        dyn_cast_or_null<pto::AddressSpaceAttr>(dstMemRefType.getMemorySpace());
    if (srcSpace && dstSpace && srcSpace != dstSpace) {
      return emitOpError("memref result must stay within the tile memory space");
    }
    return success();
  }

  auto dstPtrType = dyn_cast<pto::PtrType>(dstType);
  if (!dstPtrType) {
    return emitOpError("result must be a memref or !pto.ptr<...>");
  }
  if (dstPtrType.getElementType() != elementType) {
    return emitOpError(
        "pointer result element type must match tile element type");
  }
  if (srcSpace && dstPtrType.getMemorySpace() != srcSpace) {
    return emitOpError("pointer result must stay within the tile memory space");
  }
  return success();
}

void PldsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult PldsOp::verify() {
  return verifyPredicateLoadOp(
      *this, [&]() -> LogicalResult {
        if (!getOffset().getType().isIndex()) {
          return emitOpError("requires index offset");
        }
        return success();
      });
}

void PldiOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult PldiOp::verify() {
  return verifyPredicateLoadOp(
      *this, [&]() -> LogicalResult {
        if (!matchPattern(getOffset(), m_Constant())) {
          return emitOpError("requires offset to be a constant index immediate");
        }
        return success();
      });
}

template <typename HistOp>
static LogicalResult verifyHistogramOp(HistOp op) {
  if (failed(verifyVRegTypeLike(op, op.getAcc().getType(), "acc type")) ||
      failed(verifyVRegTypeLike(op, op.getSource().getType(), "source type")) ||
      failed(verifyMaskTypeWithGranularityLike(op, op.getMask().getType(),
                                               "mask type", "b8")) ||
      failed(verifyVRegTypeLike(op, op.getResult().getType(), "result type"))) {
    return failure();
  }
  auto accType = cast<VRegType>(op.getAcc().getType());
  auto sourceType = cast<VRegType>(op.getSource().getType());
  auto resultType = cast<VRegType>(op.getResult().getType());
  auto accElemType = dyn_cast<IntegerType>(accType.getElementType());
  auto sourceElemType = dyn_cast<IntegerType>(sourceType.getElementType());
  if (!accElemType || accElemType.getWidth() != mlir::pto::kValue16 ||
      accType.getElementCount() != mlir::pto::kValue128) {
    return op.emitOpError("requires acc type to be !pto.vreg<128xi16>");
  }
  if (!sourceElemType || sourceElemType.getWidth() != mlir::pto::kValue8 ||
      sourceType.getElementCount() != mlir::pto::kValue256) {
    return op.emitOpError("requires source type to be !pto.vreg<256xi8>");
  }
  if (resultType != accType) {
    return op.emitOpError("requires result type to match acc type");
  }
  if (!op.getBin().getType().isInteger(mlir::pto::kValue32)) {
    return op.emitOpError("requires bin operand to be i32");
  }
  return success();
}

LogicalResult Chistv2Op::verify() { return verifyHistogramOp(*this); }
LogicalResult Dhistv2Op::verify() { return verifyHistogramOp(*this); }

void VscatterOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VscatterOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getValue().getType(), "value type"))) {
    return failure();
  }
  if (!isBufferLike(getDestination().getType())) {
    return emitOpError("requires a pointer-like destination");
  }
  auto offsetsType = dyn_cast<VRegType>(getOffsets().getType());
  auto valueType = dyn_cast<VRegType>(getValue().getType());
  if (!offsetsType || !valueType) {
    return emitOpError("value and offsets must be !pto.vreg<...>");
  }
  auto offsetsElemType = dyn_cast<IntegerType>(offsetsType.getElementType());
  if (!offsetsElemType) {
    return emitOpError("offset vector must use integer element type");
  }
  unsigned valueElemWidth = getPTOStorageElemBitWidth(valueType.getElementType());
  if (valueElemWidth != mlir::pto::kValue8 && valueElemWidth != 16 && valueElemWidth != 32) {
    return emitOpError("requires 8-, 16-, or 32-bit value elements");
  }
  unsigned expectedOffsetWidth = valueElemWidth == 32 ? 32 : 16;
  if (offsetsElemType.getWidth() != expectedOffsetWidth) {
    return emitOpError() << "requires " << expectedOffsetWidth
                         << "-bit offset vector elements for "
                         << valueElemWidth << "-bit values";
  }
  int64_t expectedOffsetCount = valueElemWidth == 8
                                    ? valueType.getElementCount() / 2
                                    : valueType.getElementCount();
  if (offsetsType.getElementCount() != expectedOffsetCount) {
    return emitOpError() << "requires " << expectedOffsetCount
                         << " offsets for " << valueType.getElementCount()
                         << "x" << valueElemWidth << "-bit values";
  }
  if (failed(verifyMaskTypeWithGranularityLike(
          *this, getMask().getType(), "mask type",
          valueElemWidth == mlir::pto::kValue32 ? "b32" : "b16"))) {
    return failure();
  }
  auto destinationType = cast<PtrType>(getDestination().getType());
  if (destinationType.getElementType() != valueType.getElementType()) {
    return emitOpError(
        "requires destination element type to match value element type");
  }
  MemoryRole destinationRole = classifyMemoryRole(getDestination().getType());
  if (destinationRole == MemoryRole::GM) {
    return emitOpError("requires a UB-backed destination");
  }
  return success();
}

void PstsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

void PstiOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult PstiOp::verify() {
  return verifyPredicateStoreOp(
      *this, [&]() -> LogicalResult {
        if (!matchPattern(getOffset(), m_Constant())) {
          return emitOpError("requires offset to be a constant index immediate");
        }
        return success();
      });
}

LogicalResult PstsOp::verify() {
  return verifyPredicateStoreOp(
      *this, [&]() -> LogicalResult {
        if (!getOffset().getType().isIndex()) {
          return emitOpError("requires index offset");
        }
        return success();
      });
}

void VsstbOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VsstbOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getValue().getType(), "value type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type"))) {
    return failure();
  }
  if (!isBufferLike(getDestination().getType())) {
    return emitOpError("requires a pointer-like destination");
  }
  if (classifyMemoryRole(getDestination().getType()) == MemoryRole::GM) {
    return emitOpError("requires a UB-backed destination");
  }
  if (!getBlockStride().getType().isSignlessInteger(mlir::pto::kValue16)) {
    return emitOpError("requires block_stride to be i16");
  }
  if (!getRepeatStride().getType().isSignlessInteger(mlir::pto::kValue16)) {
    return emitOpError("requires repeat_stride to be i16");
  }
  if (getUpdatedBase() &&
      getUpdatedBase().getType() != getDestination().getType()) {
    return emitOpError("requires updated base result to match base type");
  }
  return success();
}

void VstasOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VstasOp::verify() {
  if (failed(verifyStoreAlignChain(getValue(), *this, "value type"))) {
    return failure();
  }
  if (!isBufferLike(getDestination().getType())) {
    return emitOpError("requires a pointer-like destination");
  }
  if (classifyMemoryRole(getDestination().getType()) == MemoryRole::GM) {
    return emitOpError("requires a UB-backed destination");
  }
  if (getUpdatedBase() &&
      getUpdatedBase().getType() != getDestination().getType()) {
    return emitOpError("requires updated base result to match base type");
  }
  return success();
}

void PstuOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getAlignInMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getBaseMutable());
}

LogicalResult PstuOp::verify() {
  if (failed(verifyStoreAlignChain(getAlignIn(), *this, "align_in type")) ||
      failed(verifyMaskTypeLike(*this, getValue().getType(), "value type")) ||
      failed(verifyAlignTypeLike(*this, getAlignOut().getType(), "align_out type"))) {
    return failure();
  }
  if (!isBufferLike(getBase().getType()) || !isBufferLike(getBaseOut().getType())) {
    return emitOpError("requires pointer-like base and base_out");
  }
  if (getBase().getType() != getBaseOut().getType()) {
    return emitOpError("requires base and base_out to have identical types");
  }
  if (classifyMemoryRole(getBase().getType()) == MemoryRole::GM) {
    return emitOpError("requires a UB-backed base");
  }
  auto baseType = cast<pto::PtrType>(getBase().getType());
  auto maskType = cast<pto::MaskType>(getValue().getType());
  auto elemType = dyn_cast<IntegerType>(baseType.getElementType());
  if (!elemType || elemType.isSigned() || (elemType.getWidth() != mlir::pto::kValue16 && elemType.getWidth() != 32)) {
    return emitOpError("requires ui16/ui32 UB base type");
  }
  if (maskType.isB16() && elemType.getWidth() != mlir::pto::kValue16) {
    return emitOpError("requires !pto.mask<b16> to pair with !pto.ptr<ui16, ub>");
  }
  if (maskType.isB32() && elemType.getWidth() != mlir::pto::kValue32) {
    return emitOpError("requires !pto.mask<b32> to pair with !pto.ptr<ui32, ub>");
  }
  return success();
}

void VstusOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getAlignInMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getBaseMutable());
}

LogicalResult VstusOp::verify() {
  if (failed(verifyUbStoreAlignBase(*this))) {
    return failure();
  }
  if (getBaseOut() && getBaseOut().getType() != getBase().getType()) {
    return emitOpError("requires updated base result to match base type");
  }
  return success();
}

void VsturOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getAlignInMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getBaseMutable());
}

LogicalResult VsturOp::verify() {
  if (failed(verifyUbStoreAlignBase(*this))) {
    return failure();
  }
  if (!isSupportedPostMode(getMode())) {
    return emitOpError("requires mode to be POST_UPDATE or NO_POST_UPDATE");
  }
  return success();
}

static LogicalResult verifyRawFillGeometry(Operation *op, Value byteOffset,
                                           Value repeatTimes,
                                           Value blockNum32b, Value dstGap32b) {
  const bool hasNonNegativeGeometry =
      succeeded(checkNonNegativeConst(op, byteOffset, "byte_offset")) &&
      succeeded(checkNonNegativeConst(op, repeatTimes, "repeat_times")) &&
      succeeded(checkNonNegativeConst(op, blockNum32b, "block_num_32b")) &&
      succeeded(checkNonNegativeConst(op, dstGap32b, "dst_gap_32b"));
  if (!hasNonNegativeGeometry) {
    return failure();
  }
  if (failed(checkConstAlignment(op, byteOffset, "byte_offset",
                                 kRawFillByteOffsetAlignment))) {
    return failure();
  }
  if (failed(checkConstMax(op, repeatTimes, "repeat_times",
                           kRawFillControlFieldMax)) ||
      failed(checkConstMax(op, blockNum32b, "block_num_32b",
                           kRawFillControlFieldMax)) ||
      failed(checkConstMax(op, dstGap32b, "dst_gap_32b",
                           kRawFillControlFieldMax))) {
    return failure();
  }
  return success();
}

static LogicalResult verifyRawFillWordBits(Operation *op, int64_t fillWordBits) {
  const bool validWordBits = fillWordBits == 16 || fillWordBits == 32;
  if (!validWordBits) {
    return op->emitOpError() << "fill_word_bits must be 16 or 32, got "
                             << fillWordBits;
  }
  return success();
}

static LogicalResult verifyRawFillDestination(Operation *op, Type dstType,
                                              StringRef dstName) {
  auto addressSpace = getBufferAddressSpace(dstType);
  if (!addressSpace) {
    return op->emitOpError()
           << "requires " << dstName
           << " with an explicit PTO address space for L1 raw fill";
  }
  if (*addressSpace != pto::AddressSpace::MAT) {
    return op->emitOpError()
           << "requires " << dstName << " in the mat/l1 address space, got "
           << getAddressSpaceDiagnosticName(*addressSpace);
  }
  return success();
}

LogicalResult RawFillL1Op::verify() {
  if (failed(
          verifyRawFillDestination(getOperation(), getDst().getType(), "dst"))) {
    return failure();
  }
  if (failed(verifyRawFillWordBits(getOperation(), getFillWordBits()))) {
    return failure();
  }
  return verifyRawFillGeometry(getOperation(), getByteOffset(),
                               getRepeatTimes(), getBlockNum_32b(),
                               getDstGap_32b());
}

LogicalResult CreateCbufMatrixOp::verify() {
  auto ptrType = dyn_cast<pto::PtrType>(getDst().getType());
  if (!ptrType) {
    return emitOpError("requires a typed !pto.ptr destination");
  }
  const bool matDestination =
      ptrType.getMemorySpace().getAddressSpace() == pto::AddressSpace::MAT;
  if (!matDestination) {
    return emitOpError()
           << "requires a mat/l1 destination, got "
           << getAddressSpaceDiagnosticName(
                  ptrType.getMemorySpace().getAddressSpace());
  }
  Type elementType = ptrType.getElementType();
  auto integerType = dyn_cast<IntegerType>(elementType);
  const bool canonicalView =
      integerType && integerType.isUnsigned() &&
      (integerType.getWidth() == 16 || integerType.getWidth() == 32);
  if (!canonicalView) {
    return emitOpError()
           << "requires a ui16 or ui32 destination view, got "
              "element type "
           << elementType;
  }
  if (failed(verifyRawFillWordBits(getOperation(), getFillWordBits()))) {
    return failure();
  }
  const bool wordWidthMatches =
      static_cast<unsigned>(getFillWordBits()) == integerType.getWidth();
  if (!wordWidthMatches) {
    return emitOpError()
           << "fill_word_bits " << getFillWordBits()
           << " does not match the " << integerType.getWidth()
           << "-bit destination view";
  }
  return verifyRawFillGeometry(getOperation(), /*byteOffset=*/{},
                               getRepeatTimes(), getBlockNum_32b(),
                               getDstGap_32b());
}

void RawFillL1Op::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

void CreateCbufMatrixOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

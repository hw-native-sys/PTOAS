// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOMemOpInternal.h - shared VPTO memop verify/misc helpers --------===//
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Shared memop helpers. They live in a detail namespace so the unqualified
// MLIR/LLVM names used by the original TU keep resolving without adding
// using-directives to the global scope of this header.
// Internal to lib/PTO/IR/VPTO/memop; not installed.
//===----------------------------------------------------------------------===//

#ifndef PTO_IR_VPTO_MEMOP_INTERNAL_H
#define PTO_IR_VPTO_MEMOP_INTERNAL_H

#include "VPTOInternal.h"

namespace mlir::pto::memop_detail {

using namespace mlir;
using namespace mlir::pto;

  [[maybe_unused]] static LogicalResult verifyVPTOScalarAccessTypes(Operation *op, Type ptrTy,
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

  [[maybe_unused]] static bool isVector2F16OrBF16Type(Type type) {
    return isVector2Of(type, [](Type elem) {
      return elem.isF16() || elem.isBF16();
    });
  }

  [[maybe_unused]] static bool isSupportedAtomicScalarType(Type type) {
    if (auto intType = dyn_cast<IntegerType>(type)) {
      return intType.getWidth() == mlir::pto::kValue32 ||
             intType.getWidth() == mlir::pto::kValue64;
    }
    return type.isF16() || type.isBF16() || type.isF32() ||
           isVector2F16OrBF16Type(type);
  }

  [[maybe_unused]] static LogicalResult verifyAtomicCommon(Operation *op, Value ptr, Type valueType,
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

  [[maybe_unused]] static LogicalResult verifyLdgStgAccess(Operation *op, Type ptrType,
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

  [[maybe_unused]] static LogicalResult verifyLdStDevAccess(Operation *op, Type ptrType,
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

  // Batch7: MemOp 双胞胎共用校验
  [[maybe_unused]] static bool isSupportedPredicateLoadDist(llvm::StringRef dist);
  [[maybe_unused]] static bool isSupportedPredicateStoreDist(llvm::StringRef dist);

  template <typename OpTy>
  [[maybe_unused]] static LogicalResult verifyPTOScalarAccessOp(OpTy op, llvm::StringRef tag) {
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
  [[maybe_unused]] static LogicalResult verifyGatherOffsetTypes(OpTy op, VRegType &offsetsType,
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
  [[maybe_unused]] static LogicalResult verifyPredicateLoadOp(OpTy op, OffsetCheck offsetCheck) {
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
  [[maybe_unused]] static LogicalResult verifyPredicateStoreOp(OpTy op, OffsetCheck offsetCheck) {
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
  [[maybe_unused]] static LogicalResult verifyUbStoreAlignBase(OpTy op) {
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

  template <typename OpTy>
  [[maybe_unused]] static void getAtomicEffects(
      OpTy op,
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
          &effects) {
    effects.emplace_back(MemoryEffects::Read::get(), &op.getPtrMutable());
    effects.emplace_back(MemoryEffects::Write::get(), &op.getPtrMutable());
  }

  [[maybe_unused]] static bool isSupportedPredicateLoadDist(StringRef dist) {
    return dist == "NORM" || dist == "US" || dist == "DS";
  }

  [[maybe_unused]] static bool isSupportedPredicateStoreDist(StringRef dist) {
    return dist == "NORM" || dist == "PK";
  }

  [[maybe_unused]] static bool isSupportedSprToken(StringRef spr) { return spr == "AR"; }

  [[maybe_unused]] static bool isUnsignedOrSignlessIntegerOfWidth(Type type, unsigned width) {
    auto intType = dyn_cast<IntegerType>(type);
    return intType && intType.getWidth() == width && !intType.isSigned();
  }

  [[maybe_unused]] static bool isSameVgather2IntegerSemantics(IntegerType sourceType,
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

  [[maybe_unused]] static bool isVgather2B8ResultType(IntegerType sourceType,
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

  [[maybe_unused]] static LogicalResult resolveVgather2WidthInfo(
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

  // Shared gather offset-type verification plus the 32-bit offset width
  // requirement. Vgather2 deliberately allows variable offset widths, so it
  // does not go through this helper.
  template <typename OpTy>
  [[maybe_unused]] static LogicalResult verifyGatherOffsetTypesAndWidth(
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

  [[maybe_unused]] static LogicalResult verifySprStoreCommon(Operation *op, StringRef opName,
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

  template <typename HistOp>
  [[maybe_unused]] static LogicalResult verifyHistogramOp(HistOp op) {
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

  [[maybe_unused]] static LogicalResult verifyRawFillGeometry(Operation *op, Value byteOffset,
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

  [[maybe_unused]] static LogicalResult verifyRawFillWordBits(Operation *op, int64_t fillWordBits) {
    const bool validWordBits = fillWordBits == 16 || fillWordBits == 32;
    if (!validWordBits) {
      return op->emitOpError() << "fill_word_bits must be 16 or 32, got "
                               << fillWordBits;
    }
    return success();
  }

  [[maybe_unused]] static LogicalResult verifyRawFillDestination(Operation *op, Type dstType,
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

} // namespace mlir::pto::memop_detail

#endif // PTO_IR_VPTO_MEMOP_INTERNAL_H

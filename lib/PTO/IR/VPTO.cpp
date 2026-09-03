// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTO.cpp - VPTO dialect -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/VPTOMemoryDist.h"
#include "PTO/Support/CodeConstants.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <optional>

using namespace mlir;
using namespace mlir::pto;

static llvm::cl::opt<bool> disableVPTOAlignChainVerification(
    "vpto-disable-align-chain-verification",
    llvm::cl::desc("Disable !pto.align linear-chain verifier checks"),
    llvm::cl::init(false), llvm::cl::Hidden);

static std::string formatVRegType(int64_t elementCount, Type elementType) {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  os << "!pto.vreg<" << elementCount << "x" << elementType << ">";
  return storage;
}

static std::string formatMaskType(StringRef granularity) {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  os << "!pto.mask<" << granularity << ">";
  return storage;
}

static LogicalResult verifyVRegTypeLike(Operation *op, Type type,
                                       StringRef roleDescription) {
  auto vecType = dyn_cast<VRegType>(type);
  if (!vecType) {
    return op->emitOpError() << roleDescription << " must be !pto.vreg<...>";
  }

  return VRegType::verify(
      [&]() { return op->emitOpError() << roleDescription << " "; },
      vecType.getElementCount(), vecType.getElementType());
}

static LogicalResult verifyMaskTypeLike(Operation *op, Type type,
                                        StringRef roleDescription) {
  if (!isa<MaskType>(type)) {
    return op->emitOpError() << roleDescription << " must be !pto.mask<...>";
  }
  return success();
}

static LogicalResult verifyNonLowPrecisionVRegElementTypeLike(
    Operation *op, Type type, StringRef roleDescription) {
  auto vecType = dyn_cast<VRegType>(type);
  if (!vecType) {
    return success();
  }
  if (pto::isPTOLowPrecisionType(vecType.getElementType())) {
    return op->emitOpError()
           << roleDescription
           << " must not use low-precision vector element type; "
              "low-precision vreg elements are currently only supported on "
              "explicit memory/conversion ops such as vlds/vsts/vcvt/vmulscvt/vpack";
  }
  return success();
}

static LogicalResult verifyMaskTypeWithGranularityLike(Operation *op, Type type,
                                                       StringRef roleDescription,
                                                       StringRef granularity) {
  auto maskType = dyn_cast<MaskType>(type);
  if (!maskType) {
    return op->emitOpError() << roleDescription << " must be !pto.mask<...>";
  }
  if (maskType.getGranularity() != granularity) {
    return op->emitOpError()
           << roleDescription << " must be " << formatMaskType(granularity);
  }
  return success();
}

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

static bool isMaskGranularityAdjacentWidening(StringRef inputGranularity,
                                              StringRef resultGranularity) {
  return (inputGranularity == "b8" && resultGranularity == "b16") ||
         (inputGranularity == "b16" && resultGranularity == "b32");
}

static bool isMaskGranularityAdjacentNarrowing(StringRef inputGranularity,
                                               StringRef resultGranularity) {
  return (inputGranularity == "b16" && resultGranularity == "b8") ||
         (inputGranularity == "b32" && resultGranularity == "b16");
}

static bool isSupportedShuffleValueType(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    return intType.getWidth() == mlir::pto::kValue32 || intType.getWidth() == 64;
  }
  if (auto vecType = dyn_cast<VectorType>(type)) {
    return vecType.getRank() == 1 && vecType.getDimSize(0) == mlir::pto::kValue2 &&
           vecType.getElementType().isF16();
  }
  return type.isF16() || type.isF32();
}

static bool isSupportedReduxValueType(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    return intType.getWidth() == mlir::pto::kValue32;
  }
  return type.isF16() || type.isF32();
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

static LogicalResult verifyShuffleSemanticControl(Operation *op,
                                                  Type controlType,
                                                  IntegerAttr widthAttr,
                                                  StringRef ctrlName) {
  if (!isSupportedShuffleValueType(op->getResultTypes().front())) {
    return op->emitOpError()
           << "requires i32, i64, f16, f32 or vector<2xf16> value/result type";
  }
  if (!controlType.isInteger(32)) {
    return op->emitOpError() << "requires " << ctrlName
                             << " operand to be i32";
  }

  int64_t width = widthAttr.getInt();
  if (width != mlir::pto::kValue16 && width != 32) {
    return op->emitOpError() << "requires width to be 16 or 32";
  }
  return success();
}

static LogicalResult verifyReduxSemanticType(Operation *op, Type valueType,
                                             Attribute signednessAttr,
                                             bool requireSignedness) {
  if (!isSupportedReduxValueType(valueType)) {
    return op->emitOpError()
           << "requires i32, f16 or f32 value/result type";
  }

  auto intType = dyn_cast<IntegerType>(valueType);
  if (!intType) {
    if (signednessAttr) {
      return op->emitOpError()
             << "does not accept signedness for floating-point redux";
    }
    return success();
  }

  if (!signednessAttr && requireSignedness) {
    return op->emitOpError()
           << "requires explicit signedness for integer redux";
  }

  if (!signednessAttr) {
    return success();
  }

  auto signedness = cast<pto::SignednessAttr>(signednessAttr).getValue();
  (void)signedness;
  return success();
}

static bool isStandardScalarConvertType(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    return intType.getWidth() == mlir::pto::kValue32 || intType.getWidth() == 64;
  }
  return type.isF16() || type.isBF16() || type.isF32();
}

static bool isIntegerLikeConvertType(Type type) {
  return isa<IntegerType>(type);
}

static bool isVector2Of(Type type, llvm::function_ref<bool(Type)> elementPred) {
  auto vecType = dyn_cast<VectorType>(type);
  return vecType && vecType.getRank() == 1 && vecType.getDimSize(0) == mlir::pto::kValue2 &&
         elementPred(vecType.getElementType());
}

static bool isSupportedPackedConvertType(Type type) {
  if (pto::isPTOHiFloat8x2Type(type)) {
    return true;
  }
  return isVector2Of(type, [](Type elem) {
    return elem.isF16() || elem.isBF16() || elem.isF32() ||
           pto::isPTOFloat8Type(elem) || pto::isPTOHiFloat8Type(elem);
  });
}

static bool isSupportedLowPrecisionConvertType(Type type) {
  return pto::isPTOFloat4PackedType(type);
}

static bool isVector2F16OrBF16Type(Type type) {
  return isVector2Of(type, [](Type elem) {
    return elem.isF16() || elem.isBF16();
  });
}

static bool isInsideSimtExecutionScope(Operation *op) {
  auto funcOp = op->getParentOfType<func::FuncOp>();
  return (funcOp && funcOp->hasAttr(pto::kPTOSimtEntryAttrName)) ||
         op->getParentOfType<pto::SectionSimtOp>();
}

static bool isSupportedConvertType(Type type) {
  return isStandardScalarConvertType(type) || isSupportedPackedConvertType(type) ||
         isSupportedLowPrecisionConvertType(type);
}

static LogicalResult verifyPackedConvertControls(Operation *op, Type srcType,
                                                 Type dstType,
                                                 pto::Rounding rounding) {
  auto isV2F16 = [](Type type) {
    return isVector2Of(type, [](Type elem) { return elem.isF16(); });
  };
  auto isV2BF16 = [](Type type) {
    return isVector2Of(type, [](Type elem) { return elem.isBF16(); });
  };
  auto isV2F32 = [](Type type) {
    return isVector2Of(type, [](Type elem) { return elem.isF32(); });
  };
  auto isV2F8 = [](Type type) {
    return isVector2Of(type, [](Type elem) { return pto::isPTOFloat8Type(elem); });
  };
  auto isV2HiF8 = [](Type type) {
    return pto::isPTOHiFloat8x2Type(type);
  };
  auto isF4 = [](Type type) { return pto::isPTOFloat4PackedType(type); };
  auto isRoundRAFZC = [](pto::Rounding rounding) {
    return rounding == pto::Rounding::R || rounding == pto::Rounding::A ||
           rounding == pto::Rounding::F || rounding == pto::Rounding::C ||
           rounding == pto::Rounding::Z;
  };

  if (isV2F32(srcType) && isV2F16(dstType)) {
    if (rounding == pto::Rounding::H) {
      return op->emitOpError()
             << "f32x2-to-f16x2 conversion supports rounding r/a/f/c/z/o";
    }
    return success();
  }
  if (isV2F32(srcType) && isV2BF16(dstType)) {
    if (rounding == pto::Rounding::O || rounding == pto::Rounding::H) {
      return op->emitOpError()
             << "f32x2-to-bf16x2 conversion supports rounding r/a/f/c/z";
    }
    return success();
  }
  if ((isV2F16(srcType) || isV2BF16(srcType)) && isV2F32(dstType)) {
    if (rounding == pto::Rounding::O || rounding == pto::Rounding::H) {
      return op->emitOpError()
             << "packed-to-f32x2 conversion supports rounding r/a/f/c/z";
    }
    return success();
  }
  if (isV2F32(srcType) && isV2F8(dstType)) {
    if (rounding != pto::Rounding::R) {
      return op->emitOpError()
             << "f32x2-to-f8x2 conversion supports rounding r";
    }
    return success();
  }
  if ((isV2F32(srcType) || isV2F16(srcType)) && isV2HiF8(dstType)) {
    if (rounding != pto::Rounding::A && rounding != pto::Rounding::H) {
      return op->emitOpError()
             << "f32x2/f16x2-to-hif8x2 conversion supports rounding a/h";
    }
    return success();
  }
  if ((isV2F8(srcType) || isV2HiF8(srcType)) &&
      (isV2F32(dstType) || isV2F16(dstType))) {
    if (!isRoundRAFZC(rounding)) {
      return op->emitOpError()
             << "f8x2/hif8x2-to-f32x2/f16x2 conversion supports rounding r/a/f/c/z";
    }
    return success();
  }
  if ((isV2BF16(srcType) && isF4(dstType)) ||
      (isF4(srcType) && isV2BF16(dstType))) {
    if (!isRoundRAFZC(rounding)) {
      return op->emitOpError()
             << "bf16x2-to-f4 and f4-to-bf16x2 conversion supports rounding r/a/f/c/z";
    }
    return success();
  }

  return op->emitOpError()
         << "unsupported packed conversion type pair; supported packed pairs are "
            "f32x2-to-f16x2, f16x2-to-f32x2, f32x2-to-bf16x2, and "
            "bf16x2-to-f32x2, f32x2-to-f8x2, f32x2/f16x2-to-hif8x2, "
            "f8x2/hif8x2-to-f32x2/f16x2, and bf16x2-to/from-f4";
}

static LogicalResult verifyConvertControls(Operation *op, Type srcType,
                                           Type dstType,
                                           pto::Rounding rounding,
                                           pto::Saturation saturation,
                                           Attribute signednessAttr) {
  if (!isSupportedConvertType(srcType) || !isSupportedConvertType(dstType)) {
    return op->emitOpError()
           << "requires i32, i64, f16, bf16, f32 or supported vector<2xT> "
              "conversion types";
  }

  bool srcInt = isIntegerLikeConvertType(srcType);
  bool dstInt = isIntegerLikeConvertType(dstType);
  bool srcPacked = isSupportedPackedConvertType(srcType);
  bool dstPacked = isSupportedPackedConvertType(dstType);
  bool srcLowPrecision = isSupportedLowPrecisionConvertType(srcType);
  bool dstLowPrecision = isSupportedLowPrecisionConvertType(dstType);
  if (srcPacked || dstPacked || srcLowPrecision || dstLowPrecision) {
    if (srcInt || dstInt) {
      return op->emitOpError()
             << "does not support mixed integer and packed conversion";
    }
    if (signednessAttr) {
      return op->emitOpError()
             << "does not accept signedness for packed floating conversion";
    }
    if (!((srcPacked || srcLowPrecision) && (dstPacked || dstLowPrecision))) {
      return op->emitOpError()
             << "does not support mixed scalar and packed conversion";
    }
    return verifyPackedConvertControls(op, srcType, dstType, rounding);
  }

  if (srcInt && dstInt) {
    return op->emitOpError()
           << "does not support integer-to-integer conversion";
  }

  if ((srcInt || dstInt) && !signednessAttr) {
    return op->emitOpError()
           << "requires signedness when converting to or from integer type";
  }
  if (!srcInt && !dstInt && signednessAttr) {
    return op->emitOpError()
           << "does not accept signedness for floating-to-floating conversion";
  }

  if (srcInt) {
    if (srcType.isInteger(mlir::pto::kValue64) && !dstType.isF32()) {
      return op->emitOpError()
             << "supports i64 conversion only to f32 in the confirmed slice";
    }
    if (srcType.isInteger(mlir::pto::kValue32) &&
        !(dstType.isF32() || dstType.isF16() || dstType.isBF16())) {
      return op->emitOpError()
             << "unsupported integer-to-floating conversion type pair";
    }
    if (rounding == pto::Rounding::O || rounding == pto::Rounding::H) {
      return op->emitOpError()
             << "integer-to-floating conversion supports rounding r/a/f/c/z";
    }
    (void)saturation;
    return success();
  }

  if (dstType.isInteger(mlir::pto::kValue64) && !srcType.isF32()) {
    return op->emitOpError()
           << "supports conversion to i64 only from f32 in the confirmed slice";
  }
  if (srcType.isF32()) {
    if (dstType.isInteger(mlir::pto::kValue32) || dstType.isInteger(64)) {
      if (saturation != pto::Saturation::Enable) {
        return op->emitOpError()
               << "fp32-to-integer conversion requires saturation enable";
      }
      if (rounding == pto::Rounding::O || rounding == pto::Rounding::H) {
        return op->emitOpError()
               << "fp32-to-integer conversion supports rounding r/a/f/c/z";
      }
      return success();
    }
    if (dstType.isF16() || dstType.isBF16() || dstType.isF32()) {
      if (dstType.isF16()) {
        if (rounding == pto::Rounding::H) {
          return op->emitOpError()
                 << "fp32-to-fp16 conversion supports rounding r/a/f/c/z/o";
        }
      } else if (rounding == pto::Rounding::O ||
                 rounding == pto::Rounding::H) {
        return op->emitOpError()
               << "fp32-to-floating conversion supports rounding r/a/f/c/z";
      }
      return success();
    }
  }

  if (srcType.isF16() || srcType.isBF16()) {
    if (dstType.isInteger(mlir::pto::kValue32)) {
      if (saturation != pto::Saturation::Enable) {
        return op->emitOpError()
               << "fp16/bf16-to-integer conversion requires saturation enable";
      }
      if (rounding == pto::Rounding::O || rounding == pto::Rounding::H) {
        return op->emitOpError()
               << "fp16/bf16-to-integer conversion supports rounding r/a/f/c/z";
      }
      return success();
    }
    if (dstType.isF32() || dstType.isF16() || dstType.isBF16()) {
      if (rounding == pto::Rounding::O || rounding == pto::Rounding::H) {
        return op->emitOpError()
               << "fp16/bf16-to-floating conversion supports rounding r/a/f/c/z";
      }
      return success();
    }
  }

  return op->emitOpError() << "unsupported conversion type pair";
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
    return op->emitOpError()
           << "requires i32, i64, f16, bf16, f32, vector<2xf16> or "
              "vector<2xbf16> atomic value type";
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
      return op->emitOpError()
             << "requires packed atomics to be inside a pto.simt_entry "
                "function or pto.section.simt on beta.1";
    }
    if (!op->getResult(0).use_empty()) {
      return op->emitOpError()
             << "does not support using the old value result for packed "
                "atomics on beta.1; leave the result unused";
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

LogicalResult PTOLdgOp::verify() {
  if (failed(verifyVPTOScalarAccessTypes(getOperation(), getPtr().getType(),
                                         getValue().getType(), "ldg"))) {
    return failure();
  }
  if (failed(verifyLdgStgAccess(getOperation(), getPtr().getType(),
                                getValue().getType()))) {
    return failure();
  }
  if (!isInsideSimtExecutionScope(getOperation())) {
    return emitOpError()
           << "must be inside a pto.simt_entry function or pto.section.simt";
  }
  return success();
}

LogicalResult PTOStgOp::verify() {
  if (failed(verifyVPTOScalarAccessTypes(getOperation(), getPtr().getType(),
                                         getValue().getType(), "stg"))) {
    return failure();
  }
  if (failed(verifyLdgStgAccess(getOperation(), getPtr().getType(),
                                getValue().getType()))) {
    return failure();
  }
  if (!isInsideSimtExecutionScope(getOperation())) {
    return emitOpError()
           << "must be inside a pto.simt_entry function or pto.section.simt";
  }
  return success();
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

LogicalResult ShuffleIdxOp::verify() {
  return verifyShuffleSemanticControl(getOperation(), getIndex().getType(),
                                      getWidthAttr(), "index");
}

LogicalResult ShuffleUpOp::verify() {
  return verifyShuffleSemanticControl(getOperation(), getOffset().getType(),
                                      getWidthAttr(), "offset");
}

LogicalResult ShuffleDownOp::verify() {
  return verifyShuffleSemanticControl(getOperation(), getOffset().getType(),
                                      getWidthAttr(), "offset");
}

LogicalResult ShuffleBflyOp::verify() {
  return verifyShuffleSemanticControl(getOperation(), getMask().getType(),
                                      getWidthAttr(), "mask");
}

LogicalResult ReduxAddOp::verify() {
  return verifyReduxSemanticType(getOperation(), getValue().getType(),
                                  getSignednessAttr(), /*requireSignedness=*/false);
}

LogicalResult ReduxMaxOp::verify() {
  return verifyReduxSemanticType(getOperation(), getValue().getType(),
                                  getSignednessAttr(), /*requireSignedness=*/true);
}

LogicalResult ReduxMinOp::verify() {
  return verifyReduxSemanticType(getOperation(), getValue().getType(),
                                  getSignednessAttr(), /*requireSignedness=*/true);
}

LogicalResult MulhiOp::verify() {
  if (!getResult().getType().isInteger(mlir::pto::kValue32) &&
      !getResult().getType().isInteger(mlir::pto::kValue64)) {
    return emitOpError() << "requires i32 or i64 result type";
  }
  return success();
}

LogicalResult MulI32ToI64Op::verify() { return success(); }

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

LogicalResult ConvertOp::verify() {
  return verifyConvertControls(getOperation(), getSrc().getType(),
                               getDst().getType(), getRounding(),
                               getSaturation(), getSignednessAttr());
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

static LogicalResult verifyNotNestedInVecScope(Operation *op,
                                               StringRef opNameForDiag) {
  if (op->getParentOfType<VecScopeOp>() ||
      op->getParentOfType<StrictVecScopeOp>()) {
    return op->emitOpError()
           << "must not be nested under pto.vecscope/pto.strict_vecscope; "
           << opNameForDiag << " is a UB helper op rather than a vecscope op";
  }
  return success();
}

static LogicalResult verifyNestedInVecScope(Operation *op,
                                            StringRef opNameForDiag) {
  if (op->getParentOfType<VecScopeOp>() || op->getParentOfType<StrictVecScopeOp>()) {
    return success();
  }
  return op->emitOpError()
         << "must be nested under pto.vecscope/pto.strict_vecscope; "
         << opNameForDiag << " is part of the vecscope control sequence";
}

static LogicalResult verifyAlignTypeLike(Operation *op, Type type,
                                         StringRef roleDescription) {
  if (!isa<AlignType>(type)) {
    return op->emitOpError() << roleDescription << " must be !pto.align";
  }
  return success();
}

static bool isSupportedVdupPosition(std::optional<StringRef> position) {
  return !position || *position == "LOWEST" || *position == "HIGHEST";
}

static bool isSupportedMovPadScalarType(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    return intType.isSignless() &&
           (intType.getWidth() == mlir::pto::kValue8 || intType.getWidth() == 16 ||
            intType.getWidth() == mlir::pto::kValue32);
  }
  if (auto floatType = dyn_cast<FloatType>(type)) {
    return floatType.isF16() || floatType.isBF16() || floatType.isF32();
  }
  return false;
}

static bool isMxElementType(Type type) {
  return isa<Float8E4M3FNType, Float8E5M2Type>(type) ||
         isa<pto::F4E1M2x2Type, pto::F4E2M1x2Type>(type);
}

static std::optional<StringRef> getVdupMaskGranularity(Type elementType) {
  if (auto intType = dyn_cast<IntegerType>(elementType)) {
    switch (intType.getWidth()) {
    case mlir::pto::kValue8:
      return StringRef("b8");
    case mlir::pto::kValue16:
      return StringRef("b16");
    case mlir::pto::kValue32:
      return StringRef("b32");
    default:
      return std::nullopt;
    }
  }
  if (elementType.isF16() || elementType.isBF16()) {
    return StringRef("b16");
  }
  if (elementType.isF32()) {
    return StringRef("b32");
  }
  return std::nullopt;
}

static bool isSupportedVtrcRoundMode(StringRef mode) {
  return mode == "R" || mode == "A" || mode == "F" || mode == "C" ||
         mode == "Z";
}

static bool isStoreAlignProducer(Operation *op) {
  return isa<InitAlignOp, PstuOp, VstusOp, VsturOp>(op);
}

static bool isStoreAlignSink(Operation *op) {
  return isa<VstasOp, VstarOp>(op);
}

static bool isLoadAlignProducer(Operation *op) {
  return isa<VldasOp, VldusOp>(op);
}

static scf::IfOp getEnclosingBranchIf(Operation *op) {
  for (Operation *cursor = op; cursor; cursor = cursor->getParentOp()) {
    auto ifOp = dyn_cast<scf::IfOp>(cursor);
    if (!ifOp) {
      continue;
    }
    Region *parentRegion = op->getParentRegion();
    if (parentRegion == &ifOp.getThenRegion() || parentRegion == &ifOp.getElseRegion()) {
      return ifOp;
    }
  }
  return nullptr;
}

static bool isValueOwnedByRegion(Value value, Region *region) {
  Region *valueRegion = nullptr;
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    valueRegion = blockArg.getParentRegion();
  } else if (Operation *def = value.getDefiningOp()) {
    valueRegion = def->getParentRegion();
  }

  while (valueRegion) {
    if (valueRegion == region) {
      return true;
    }
    Operation *parentOp = valueRegion->getParentOp();
    valueRegion = parentOp ? parentOp->getParentRegion() : nullptr;
  }
  return false;
}

static FailureOr<Value> resolveStoreAlignRoot(Value value, Operation *user);
static FailureOr<Value> resolveLoadAlignRoot(Value value, Operation *user);

static FailureOr<Value> resolveStoreAlignRootImpl(
    Value current, llvm::SmallPtrSet<void *, mlir::pto::kValue8> visited) {
  while (true) {
    if (!visited.insert(current.getAsOpaquePointer()).second) {
      return failure();
    }

    if (auto blockArg = dyn_cast<BlockArgument>(current)) {
      auto *owner = blockArg.getOwner();
      auto forOp = dyn_cast<scf::ForOp>(owner->getParentOp());
      if (!forOp) {
        return failure();
      }
      unsigned argNumber = blockArg.getArgNumber();
      unsigned ivCount = forOp.getNumInductionVars();
      if (argNumber < ivCount) {
        return failure();
      }
      unsigned iterIdx = argNumber - ivCount;
      if (iterIdx >= forOp.getInitArgs().size()) {
        return failure();
      }
      current = forOp.getInitArgs()[iterIdx];
      continue;
    }

    if (Operation *def = current.getDefiningOp()) {
      if (isa<InitAlignOp>(def)) {
        return current;
      }
      if (auto stateOp = dyn_cast<PstuOp>(def)) {
        current = stateOp.getAlignIn();
        continue;
      }
      if (auto stateOp = dyn_cast<VstusOp>(def)) {
        current = stateOp.getAlignIn();
        continue;
      }
      if (auto stateOp = dyn_cast<VsturOp>(def)) {
        current = stateOp.getAlignIn();
        continue;
      }
      if (auto forOp = dyn_cast<scf::ForOp>(def)) {
        auto result = dyn_cast<OpResult>(current);
        if (!result) {
          return failure();
        }
        unsigned resultIdx = result.getResultNumber();
        if (resultIdx >= forOp.getYieldedValues().size()) {
          return failure();
        }
        current = forOp.getYieldedValues()[resultIdx];
        continue;
      }
      if (auto ifOp = dyn_cast<scf::IfOp>(def)) {
        auto result = dyn_cast<OpResult>(current);
        if (!result || !ifOp.elseBlock()) {
          return failure();
        }
        unsigned resultIdx = result.getResultNumber();
        auto thenYield = dyn_cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
        auto elseYield = dyn_cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());
        if (!thenYield || !elseYield || resultIdx >= thenYield.getNumOperands() ||
            resultIdx >= elseYield.getNumOperands()) {
          return failure();
        }
        FailureOr<Value> thenRoot =
            resolveStoreAlignRootImpl(thenYield.getOperand(resultIdx), visited);
        FailureOr<Value> elseRoot =
            resolveStoreAlignRootImpl(elseYield.getOperand(resultIdx), visited);
        if (failed(thenRoot) || failed(elseRoot) || *thenRoot != *elseRoot) {
          return failure();
        }
        return *thenRoot;
      }
    }

    return failure();
  }
}

static FailureOr<Value> resolveStoreAlignRoot(Value value, Operation *user) {
  (void)user;
  return resolveStoreAlignRootImpl(value, {});
}

static LogicalResult verifyStoreAlignLoopThreading(Value align, Operation *user,
                                                   StringRef roleDescription) {
  Operation *cursor = user;
  while (auto forOp = cursor->getParentOfType<scf::ForOp>()) {
    Region *body = &forOp.getRegion();
    if (isValueOwnedByRegion(align, body)) {
      return success();
    }
    if (!isValueOwnedByRegion(align, body)) {
      return user->emitOpError()
             << roleDescription
             << " must be threaded through scf.for iter_args when used inside a "
                "loop";
    }
    cursor = forOp;
  }
  return success();
}

static FailureOr<Value> resolveSingleAlignIfResult(scf::IfOp ifOp) {
  SmallVector<unsigned> alignResultIndices;
  for (auto [index, type] : llvm::enumerate(ifOp.getResultTypes())) {
    if (isa<AlignType>(type)) {
      alignResultIndices.push_back(index);
    }
  }
  if (alignResultIndices.size() != 1) {
    return failure();
  }
  return ifOp.getResult(alignResultIndices.front());
}

static LogicalResult verifyStoreAlignLinearUses(Value value, Operation *user) {
  llvm::SmallPtrSet<void *, mlir::pto::kValue16> visited;
  Value current = value;

  while (visited.insert(current.getAsOpaquePointer()).second) {
    SmallVector<Value> nextValues;
    SmallVector<Operation *> terminalUsers;
    SmallVector<Operation *> branchUsers;

    for (OpOperand &use : current.getUses()) {
      Operation *owner = use.getOwner();
      if (isStoreAlignSink(owner)) {
        terminalUsers.push_back(owner);
        branchUsers.push_back(owner);
        continue;
      }
      if (auto stateOp = dyn_cast<PstuOp>(owner)) {
        nextValues.push_back(stateOp.getAlignOut());
        branchUsers.push_back(owner);
        continue;
      }
      if (auto stateOp = dyn_cast<VstusOp>(owner)) {
        nextValues.push_back(stateOp.getAlignOut());
        branchUsers.push_back(owner);
        continue;
      }
      if (auto stateOp = dyn_cast<VsturOp>(owner)) {
        nextValues.push_back(stateOp.getAlignOut());
        branchUsers.push_back(owner);
        continue;
      }
      if (auto forOp = dyn_cast<scf::ForOp>(owner)) {
        unsigned firstInitArg = forOp.getNumControlOperands();
        if (use.getOperandNumber() < firstInitArg) {
          return user->emitOpError()
                 << "found unexpected scf.for control operand use for !pto.align";
        }
        unsigned iterIdx = use.getOperandNumber() - firstInitArg;
        if (iterIdx >= forOp.getRegionIterArgs().size()) {
          return user->emitOpError()
                 << "found invalid scf.for iter_args use for !pto.align";
        }
        nextValues.push_back(forOp.getRegionIterArgs()[iterIdx]);
        continue;
      }
      if (auto yieldOp = dyn_cast<scf::YieldOp>(owner)) {
        auto forOp = dyn_cast<scf::ForOp>(yieldOp->getParentOp());
        if (!forOp) {
          return user->emitOpError()
                 << "found !pto.align yielded from non-scf.for loop";
        }
        unsigned resultIdx = use.getOperandNumber();
        if (resultIdx >= forOp.getNumResults()) {
          return user->emitOpError()
                 << "found invalid scf.yield result mapping for !pto.align";
        }
        nextValues.push_back(forOp.getResult(resultIdx));
        continue;
      }
      return user->emitOpError()
             << "found unsupported !pto.align consumer " << owner->getName();
    }

    if (nextValues.size() + terminalUsers.size() > 1) {
      scf::IfOp commonIf;
      for (Operation *branchUser : branchUsers) {
        scf::IfOp enclosingIf = getEnclosingBranchIf(branchUser);
        if (!enclosingIf) {
          commonIf = nullptr;
          break;
        }
        if (!commonIf) {
          commonIf = enclosingIf;
        }
        else if (commonIf != enclosingIf) {
          commonIf = nullptr;
          break;
        }
      }
      if (commonIf) {
        FailureOr<Value> mergedValue = resolveSingleAlignIfResult(commonIf);
        if (succeeded(mergedValue)) {
          current = *mergedValue;
          continue;
        }
      }
      return user->emitOpError()
             << "!pto.align value must form a single linear store-state chain";
    }
    if (nextValues.empty()) {
      return success();
    }
    current = nextValues.front();
  }

  return success();
}

static LogicalResult verifyStoreAlignChain(Value align, Operation *user,
                                           StringRef roleDescription) {
  if (disableVPTOAlignChainVerification) {
    return success();
  }

  if (failed(verifyAlignTypeLike(user, align.getType(), roleDescription))) {
    return failure();
  }

  if (failed(verifyStoreAlignLoopThreading(align, user, roleDescription))) {
    return failure();
  }

  FailureOr<Value> root = resolveStoreAlignRoot(align, user);
  if (failed(root)) {
    if (Operation *def = align.getDefiningOp()) {
      if (!isa<scf::ForOp>(def)) {
        return user->emitOpError()
               << roleDescription
               << " must be produced by pto.init_align or a prior store-state op, got "
               << def->getName();
      }
    }
    return user->emitOpError()
           << roleDescription
           << " must be produced by pto.init_align or a prior store-state op";
  }

  Operation *def = (*root).getDefiningOp();
  if (!isStoreAlignProducer(def)) {
    return user->emitOpError()
           << roleDescription
           << " must be produced by pto.init_align or a prior store-state op, got "
           << def->getName();
  }

  return verifyStoreAlignLinearUses(*root, user);
}

static FailureOr<Value> resolveLoadAlignRootImpl(
    Value current, llvm::SmallPtrSet<void *, mlir::pto::kValue8> visited) {
  while (true) {
    if (!visited.insert(current.getAsOpaquePointer()).second) {
      return failure();
    }

    if (auto blockArg = dyn_cast<BlockArgument>(current)) {
      auto *owner = blockArg.getOwner();
      auto forOp = dyn_cast<scf::ForOp>(owner->getParentOp());
      if (!forOp) {
        return failure();
      }
      unsigned argNumber = blockArg.getArgNumber();
      unsigned ivCount = forOp.getNumInductionVars();
      if (argNumber < ivCount) {
        return failure();
      }
      unsigned iterIdx = argNumber - ivCount;
      if (iterIdx >= forOp.getInitArgs().size()) {
        return failure();
      }
      current = forOp.getInitArgs()[iterIdx];
      continue;
    }

    if (Operation *def = current.getDefiningOp()) {
      if (isa<VldasOp>(def)) {
        return current;
      }
      if (auto stateOp = dyn_cast<VldusOp>(def)) {
        current = stateOp.getAlign();
        continue;
      }
      if (auto forOp = dyn_cast<scf::ForOp>(def)) {
        auto result = dyn_cast<OpResult>(current);
        if (!result) {
          return failure();
        }
        unsigned resultIdx = result.getResultNumber();
        if (resultIdx >= forOp.getYieldedValues().size()) {
          return failure();
        }
        current = forOp.getYieldedValues()[resultIdx];
        continue;
      }
      if (auto ifOp = dyn_cast<scf::IfOp>(def)) {
        auto result = dyn_cast<OpResult>(current);
        if (!result || !ifOp.elseBlock()) {
          return failure();
        }
        unsigned resultIdx = result.getResultNumber();
        auto thenYield = dyn_cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
        auto elseYield = dyn_cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());
        if (!thenYield || !elseYield || resultIdx >= thenYield.getNumOperands() ||
            resultIdx >= elseYield.getNumOperands()) {
          return failure();
        }
        FailureOr<Value> thenRoot =
            resolveLoadAlignRootImpl(thenYield.getOperand(resultIdx), visited);
        FailureOr<Value> elseRoot =
            resolveLoadAlignRootImpl(elseYield.getOperand(resultIdx), visited);
        if (failed(thenRoot) || failed(elseRoot) || *thenRoot != *elseRoot) {
          return failure();
        }
        return *thenRoot;
      }
    }

    return failure();
  }
}

static FailureOr<Value> resolveLoadAlignRoot(Value value, Operation *user) {
  (void)user;
  return resolveLoadAlignRootImpl(value, {});
}

static LogicalResult verifyLoadAlignLoopThreading(Value align, Operation *user,
                                                  StringRef roleDescription) {
  Operation *cursor = user;
  while (auto forOp = cursor->getParentOfType<scf::ForOp>()) {
    Region *body = &forOp.getRegion();
    if (isValueOwnedByRegion(align, body)) {
      return success();
    }
    if (!isValueOwnedByRegion(align, body)) {
      return user->emitOpError()
             << roleDescription
             << " must be threaded through scf.for iter_args when used inside a "
                "loop";
    }
    cursor = forOp;
  }
  return success();
}

static LogicalResult verifyLoadAlignLinearUses(Value value, Operation *user) {
  llvm::SmallPtrSet<void *, mlir::pto::kValue16> visited;
  Value current = value;

  while (visited.insert(current.getAsOpaquePointer()).second) {
    SmallVector<Value> nextValues;
    SmallVector<Operation *> branchUsers;

    for (OpOperand &use : current.getUses()) {
      Operation *owner = use.getOwner();
      if (auto stateOp = dyn_cast<VldusOp>(owner)) {
        nextValues.push_back(stateOp.getUpdatedAlign());
        branchUsers.push_back(owner);
        continue;
      }
      if (auto forOp = dyn_cast<scf::ForOp>(owner)) {
        unsigned firstInitArg = forOp.getNumControlOperands();
        if (use.getOperandNumber() < firstInitArg) {
          return user->emitOpError()
                 << "found unexpected scf.for control operand use for !pto.align";
        }
        unsigned iterIdx = use.getOperandNumber() - firstInitArg;
        if (iterIdx >= forOp.getRegionIterArgs().size()) {
          return user->emitOpError()
                 << "found invalid scf.for iter_args use for !pto.align";
        }
        nextValues.push_back(forOp.getRegionIterArgs()[iterIdx]);
        continue;
      }
      if (auto yieldOp = dyn_cast<scf::YieldOp>(owner)) {
        auto forOp = dyn_cast<scf::ForOp>(yieldOp->getParentOp());
        if (!forOp) {
          return user->emitOpError()
                 << "found !pto.align yielded from non-scf.for loop";
        }
        unsigned resultIdx = use.getOperandNumber();
        if (resultIdx >= forOp.getNumResults()) {
          return user->emitOpError()
                 << "found invalid scf.yield result mapping for !pto.align";
        }
        nextValues.push_back(forOp.getResult(resultIdx));
        continue;
      }
      return user->emitOpError()
             << "found unsupported !pto.align consumer " << owner->getName();
    }

    if (nextValues.size() > 1) {
      scf::IfOp commonIf;
      for (Operation *branchUser : branchUsers) {
        scf::IfOp enclosingIf = getEnclosingBranchIf(branchUser);
        if (!enclosingIf) {
          commonIf = nullptr;
          break;
        }
        if (!commonIf) {
          commonIf = enclosingIf;
        }
        else if (commonIf != enclosingIf) {
          commonIf = nullptr;
          break;
        }
      }
      if (commonIf) {
        FailureOr<Value> mergedValue = resolveSingleAlignIfResult(commonIf);
        if (succeeded(mergedValue)) {
          current = *mergedValue;
          continue;
        }
      }
      return user->emitOpError()
             << "!pto.align value must form a single linear load-state chain";
    }
    if (nextValues.empty()) {
      return success();
    }
    current = nextValues.front();
  }

  return success();
}

static LogicalResult verifyLoadAlignChain(Value align, Operation *user,
                                          StringRef roleDescription) {
  if (disableVPTOAlignChainVerification) {
    return success();
  }

  if (failed(verifyAlignTypeLike(user, align.getType(), roleDescription))) {
    return failure();
  }

  if (failed(verifyLoadAlignLoopThreading(align, user, roleDescription))) {
    return failure();
  }

  FailureOr<Value> root = resolveLoadAlignRoot(align, user);
  if (failed(root)) {
    if (Operation *def = align.getDefiningOp()) {
      if (!isa<scf::ForOp>(def)) {
        return user->emitOpError()
               << roleDescription
               << " must be produced by pto.vldas or a prior load-state op, got "
               << def->getName();
      }
    }
    return user->emitOpError()
           << roleDescription
           << " must be produced by pto.vldas or a prior load-state op";
  }

  Operation *def = (*root).getDefiningOp();
  if (!isLoadAlignProducer(def)) {
    return user->emitOpError()
           << roleDescription
           << " must be produced by pto.vldas or a prior load-state op, got "
           << def->getName();
  }

  return verifyLoadAlignLinearUses(*root, user);
}

static bool isSupportedPredicatePattern(StringRef pattern) {
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

static bool isSupportedPartToken(StringRef part) {
  return part == "LOWER" || part == "HIGHER";
}

static bool isSupportedSprToken(StringRef spr) { return spr == "AR"; }

static std::optional<StringRef> normalizeRoundModeToken(StringRef token) {
  if (token == "R" || token == "ROUND_R") {
    return StringRef("R");
  }
  if (token == "A" || token == "ROUND_A") {
    return StringRef("A");
  }
  if (token == "F" || token == "ROUND_F") {
    return StringRef("F");
  }
  if (token == "C" || token == "ROUND_C") {
    return StringRef("C");
  }
  if (token == "Z" || token == "ROUND_Z") {
    return StringRef("Z");
  }
  if (token == "O" || token == "ROUND_O") {
    return StringRef("O");
  }
  if (token == "H" || token == "ROUND_H") {
    return StringRef("H");
  }
  return std::nullopt;
}

static std::optional<StringRef> normalizeSaturationToken(StringRef token) {
  if (token == "SAT" || token == "RS_ENABLE") {
    return StringRef("SAT");
  }
  if (token == "NOSAT" || token == "RS_DISABLE") {
    return StringRef("NOSAT");
  }
  return std::nullopt;
}

static std::optional<StringRef> normalizeEvenOddPartToken(StringRef token) {
  if (token == "EVEN" || token == "PART_EVEN") {
    return StringRef("EVEN");
  }
  if (token == "ODD" || token == "PART_ODD") {
    return StringRef("ODD");
  }
  return std::nullopt;
}

static std::optional<StringRef> normalizePacked4PartToken(StringRef token) {
  if (token == "P0" || token == "PART_P0") {
    return StringRef("P0");
  }
  if (token == "P1" || token == "PART_P1") {
    return StringRef("P1");
  }
  if (token == "P2" || token == "PART_P2") {
    return StringRef("P2");
  }
  if (token == "P3" || token == "PART_P3") {
    return StringRef("P3");
  }
  return std::nullopt;
}

static std::optional<StringRef> normalizeVcvtPartToken(StringRef token) {
  if (auto normalized = normalizeEvenOddPartToken(token)) {
    return normalized;
  }
  return normalizePacked4PartToken(token);
}

namespace {

enum class VcvtElemKind {
  Invalid,
  F16,
  BF16,
  F32,
  F8E4M3,
  F8E5M2,
  HiF8,
  F4E1M2x2,
  F4E2M1x2,
  S8,
  U8,
  S16,
  U16,
  S32,
  U32,
  S64,
};

enum class VcvtPartFamily {
  EvenOdd,
  Packed4,
};

struct VcvtContract {
  bool requiresRnd;
  bool requiresSat;
  bool requiresPart;
  std::optional<VcvtPartFamily> partFamily = std::nullopt;
  const char *allowedRndModes = nullptr;
};

static VcvtElemKind classifyVcvtElemType(Type type) {
  if (type.isF16()) {
    return VcvtElemKind::F16;
  }
  if (type.isBF16()) {
    return VcvtElemKind::BF16;
  }
  if (type.isF32()) {
    return VcvtElemKind::F32;
  }
  if (pto::isPTOFloat8E4M3LikeType(type)) {
    return VcvtElemKind::F8E4M3;
  }
  if (pto::isPTOFloat8E5M2LikeType(type)) {
    return VcvtElemKind::F8E5M2;
  }
  if (pto::isPTOHiFloat8Type(type)) {
    return VcvtElemKind::HiF8;
  }
  if (isa<pto::F4E1M2x2Type>(type)) {
    return VcvtElemKind::F4E1M2x2;
  }
  if (isa<pto::F4E2M1x2Type>(type)) {
    return VcvtElemKind::F4E2M1x2;
  }
  if (auto intType = dyn_cast<IntegerType>(type)) {
    switch (intType.getWidth()) {
    case mlir::pto::kValue8:
      return intType.isUnsigned() ? VcvtElemKind::U8 : VcvtElemKind::S8;
    case mlir::pto::kValue16:
      return intType.isUnsigned() ? VcvtElemKind::U16 : VcvtElemKind::S16;
    case mlir::pto::kValue32:
      return intType.isUnsigned() ? VcvtElemKind::U32 : VcvtElemKind::S32;
    case mlir::pto::kValue64:
      return intType.isUnsigned() ? VcvtElemKind::Invalid : VcvtElemKind::S64;
    default:
      return VcvtElemKind::Invalid;
    }
  }
  return VcvtElemKind::Invalid;
}

static std::optional<unsigned> getVcvtElemBitWidth(VcvtElemKind kind) {
  switch (kind) {
  case VcvtElemKind::F16:
  case VcvtElemKind::BF16:
  case VcvtElemKind::S16:
  case VcvtElemKind::U16:
    return mlir::pto::kValue16;
  case VcvtElemKind::F32:
  case VcvtElemKind::S32:
  case VcvtElemKind::U32:
    return mlir::pto::kValue32;
  case VcvtElemKind::F8E4M3:
  case VcvtElemKind::F8E5M2:
  case VcvtElemKind::HiF8:
  case VcvtElemKind::F4E1M2x2:
  case VcvtElemKind::F4E2M1x2:
  case VcvtElemKind::S8:
  case VcvtElemKind::U8:
    return mlir::pto::kValue8;
  case VcvtElemKind::S64:
    return mlir::pto::kValue64;
  case VcvtElemKind::Invalid:
    return std::nullopt;
  }
  return std::nullopt;
}

static std::optional<VcvtPartFamily> classifyVcvtPartFamily(unsigned srcBits,
                                                            unsigned dstBits) {
  unsigned largerBits = std::max(srcBits, dstBits);
  unsigned smallerBits = std::min(srcBits, dstBits);
  if (largerBits == smallerBits * mlir::pto::kValue2) {
    return VcvtPartFamily::EvenOdd;
  }
  if (largerBits == smallerBits * mlir::pto::kValue4) {
    return VcvtPartFamily::Packed4;
  }
  return std::nullopt;
}

static bool isValidVcvtPartForFamily(StringRef part, VcvtPartFamily family) {
  switch (family) {
  case VcvtPartFamily::EvenOdd:
    return part == "EVEN" || part == "ODD";
  case VcvtPartFamily::Packed4:
    return part == "P0" || part == "P1" || part == "P2" || part == "P3";
  }
  return false;
}

static bool isValidVcvtRoundModeForContract(StringRef roundMode,
                                            const VcvtContract &contract) {
  if (!contract.allowedRndModes) {
    return true;
  }
  return StringRef(contract.allowedRndModes).contains(roundMode);
}

static std::optional<VcvtContract> lookupVcvtContract(VcvtElemKind src,
                                                      VcvtElemKind dst) {
  switch (src) {
  case VcvtElemKind::F32:
    switch (dst) {
    case VcvtElemKind::F8E4M3:
    case VcvtElemKind::F8E5M2:
      return VcvtContract{/*requiresRnd=*/true, /*requiresSat=*/true,
                          /*requiresPart=*/true, VcvtPartFamily::Packed4,
                          "RAHZ"};
    case VcvtElemKind::HiF8:
      return VcvtContract{/*requiresRnd=*/true, /*requiresSat=*/true,
                          /*requiresPart=*/true, VcvtPartFamily::Packed4,
                          "AH"};
    case VcvtElemKind::F16:
    case VcvtElemKind::BF16:
    case VcvtElemKind::S16:
    case VcvtElemKind::S64:
      return VcvtContract{/*requiresRnd=*/true, /*requiresSat=*/true,
                          /*requiresPart=*/true};
    case VcvtElemKind::S32:
      return VcvtContract{/*requiresRnd=*/true, /*requiresSat=*/true,
                          /*requiresPart=*/false};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::F16:
    switch (dst) {
    case VcvtElemKind::F8E4M3:
    case VcvtElemKind::F8E5M2:
      return VcvtContract{/*requiresRnd=*/true, /*requiresSat=*/true,
                          /*requiresPart=*/true, std::nullopt, "RAFZC"};
    case VcvtElemKind::HiF8:
      return VcvtContract{/*requiresRnd=*/true, /*requiresSat=*/true,
                          /*requiresPart=*/true, std::nullopt, "AH"};
    case VcvtElemKind::F32:
      return VcvtContract{/*requiresRnd=*/false, /*requiresSat=*/false,
                          /*requiresPart=*/true};
    case VcvtElemKind::S32:
      return VcvtContract{/*requiresRnd=*/true, /*requiresSat=*/false,
                          /*requiresPart=*/true};
    case VcvtElemKind::S16:
      return VcvtContract{/*requiresRnd=*/true, /*requiresSat=*/true,
                          /*requiresPart=*/false};
    case VcvtElemKind::S8:
    case VcvtElemKind::U8:
      return VcvtContract{/*requiresRnd=*/true, /*requiresSat=*/true,
                          /*requiresPart=*/true};
    case VcvtElemKind::BF16:
      return VcvtContract{/*requiresRnd=*/true, /*requiresSat=*/false,
                          /*requiresPart=*/false};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::BF16:
    switch (dst) {
    case VcvtElemKind::F8E4M3:
    case VcvtElemKind::F8E5M2:
      return VcvtContract{/*requiresRnd=*/true, /*requiresSat=*/true,
                          /*requiresPart=*/true, std::nullopt, "RAFZC"};
    case VcvtElemKind::F4E1M2x2:
    case VcvtElemKind::F4E2M1x2:
      return VcvtContract{/*requiresRnd=*/true, /*requiresSat=*/false,
                          /*requiresPart=*/true, VcvtPartFamily::Packed4,
                          "RAFZC"};
    case VcvtElemKind::F16:
      return VcvtContract{/*requiresRnd=*/true, /*requiresSat=*/true,
                          /*requiresPart=*/false};
    case VcvtElemKind::F32:
      return VcvtContract{/*requiresRnd=*/false, /*requiresSat=*/false,
                          /*requiresPart=*/true};
    case VcvtElemKind::S32:
      return VcvtContract{/*requiresRnd=*/true, /*requiresSat=*/true,
                          /*requiresPart=*/true};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::U8:
    switch (dst) {
    case VcvtElemKind::F16:
    case VcvtElemKind::U16:
    case VcvtElemKind::U32:
      return VcvtContract{/*requiresRnd=*/false, /*requiresSat=*/false,
                          /*requiresPart=*/true};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::S8:
    switch (dst) {
    case VcvtElemKind::F16:
    case VcvtElemKind::S16:
    case VcvtElemKind::S32:
      return VcvtContract{/*requiresRnd=*/false, /*requiresSat=*/false,
                          /*requiresPart=*/true};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::U16:
    switch (dst) {
    case VcvtElemKind::U8:
      return VcvtContract{/*requiresRnd=*/false, /*requiresSat=*/true,
                          /*requiresPart=*/true};
    case VcvtElemKind::U32:
      return VcvtContract{/*requiresRnd=*/false, /*requiresSat=*/false,
                          /*requiresPart=*/true};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::S16:
    switch (dst) {
    case VcvtElemKind::F16:
      return VcvtContract{/*requiresRnd=*/true, /*requiresSat=*/false,
                          /*requiresPart=*/false};
    case VcvtElemKind::F32:
    case VcvtElemKind::U32:
    case VcvtElemKind::S32:
      return VcvtContract{/*requiresRnd=*/false, /*requiresSat=*/false,
                          /*requiresPart=*/true};
    case VcvtElemKind::U8:
      return VcvtContract{/*requiresRnd=*/false, /*requiresSat=*/true,
                          /*requiresPart=*/true};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::U32:
    switch (dst) {
    case VcvtElemKind::U8:
    case VcvtElemKind::U16:
    case VcvtElemKind::S16:
      return VcvtContract{/*requiresRnd=*/false, /*requiresSat=*/true,
                          /*requiresPart=*/true};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::S32:
    switch (dst) {
    case VcvtElemKind::F32:
      return VcvtContract{/*requiresRnd=*/true, /*requiresSat=*/false,
                          /*requiresPart=*/false};
    case VcvtElemKind::U8:
    case VcvtElemKind::U16:
    case VcvtElemKind::S16:
      return VcvtContract{/*requiresRnd=*/false, /*requiresSat=*/true,
                          /*requiresPart=*/true};
    case VcvtElemKind::S64:
      return VcvtContract{/*requiresRnd=*/false, /*requiresSat=*/false,
                          /*requiresPart=*/true};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::S64:
    switch (dst) {
    case VcvtElemKind::F32:
      return VcvtContract{/*requiresRnd=*/true, /*requiresSat=*/false,
                          /*requiresPart=*/true};
    case VcvtElemKind::S32:
      return VcvtContract{/*requiresRnd=*/false, /*requiresSat=*/true,
                          /*requiresPart=*/true};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::F8E4M3:
  case VcvtElemKind::F8E5M2:
  case VcvtElemKind::HiF8:
    switch (dst) {
    case VcvtElemKind::F32:
      return VcvtContract{/*requiresRnd=*/false, /*requiresSat=*/false,
                          /*requiresPart=*/true, VcvtPartFamily::Packed4};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::F4E1M2x2:
  case VcvtElemKind::F4E2M1x2:
    switch (dst) {
    case VcvtElemKind::BF16:
      return VcvtContract{/*requiresRnd=*/false, /*requiresSat=*/false,
                          /*requiresPart=*/true, VcvtPartFamily::Packed4};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::Invalid:
    return std::nullopt;
  }
  return std::nullopt;
}

} // namespace

static std::optional<unsigned> getDistElementWidth(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    return intType.getWidth();
  }
  if (type.isF16() || type.isBF16()) {
    return mlir::pto::kValue16;
  }
  if (type.isF32()) {
    return mlir::pto::kValue32;
  }
  if (type.isF64()) {
    return mlir::pto::kValue64;
  }
  return std::nullopt;
}

static bool isSupportedVldx2DistToken(StringRef dist, Type elementType) {
  return lookupVPTOMemoryDist(VPTOMemoryOpFamily::LoadX2, dist,
                              getDistElementWidth(elementType));
}

static bool isSupportedVldsDistToken(StringRef dist, Type elementType) {
  return lookupVPTOMemoryDist(VPTOMemoryOpFamily::Load, dist,
                              getDistElementWidth(elementType));
}

static bool isSupportedVstsDistToken(StringRef dist) {
  return lookupVPTOMemoryDist(VPTOMemoryOpFamily::Store, dist);
}

static bool isSupportedVstsx2DistToken(StringRef dist) {
  return lookupVPTOMemoryDist(VPTOMemoryOpFamily::StoreX2, dist);
}

static std::optional<StringRef>
getVstsMaskGranularityOverride(StringRef dist) {
  const auto *contract =
      lookupVPTOMemoryDist(VPTOMemoryOpFamily::Store, dist);
  if (!contract || contract->maskGranularity.empty()) {
    return std::nullopt;
  }
  return contract->maskGranularity;
}

static bool isSupportedPostMode(StringRef mode) {
  return mode == "NO_POST_UPDATE" || mode == "POST_UPDATE";
}

static unsigned getIntOrFloatBitWidth(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    return intType.getWidth();
  }
  if (auto floatType = dyn_cast<FloatType>(type)) {
    return floatType.getWidth();
  }
  if (pto::isPTOFloat8Type(type) || pto::isPTOHiFloat8Type(type) ||
      pto::isPTOFloat4PackedType(type)) {
    return mlir::pto::kValue8;
  }
  if (pto::isPTOHiFloat8x2Type(type)) {
    return mlir::pto::kValue16;
  }
  return 0;
}

static bool isIntegerOrFloatLike(Type type) {
  return isa<IntegerType>(type) || isa<FloatType>(type);
}

static std::optional<int64_t> getVRegStorageBitWidth(Type type) {
  auto vecType = dyn_cast<VRegType>(type);
  if (!vecType) {
    return std::nullopt;
  }
  unsigned elemWidth = getIntOrFloatBitWidth(vecType.getElementType());
  if (!elemWidth) {
    return std::nullopt;
  }
  return vecType.getElementCount() * static_cast<int64_t>(elemWidth);
}

static LogicalResult verifyIntegerVRegTypeLike(Operation *op, Type type,
                                              StringRef roleDescription) {
  if (failed(verifyVRegTypeLike(op, type, roleDescription))) {
    return failure();
  }
  auto vecType = cast<VRegType>(type);
  if (!isa<IntegerType>(vecType.getElementType())) {
    return op->emitOpError()
           << roleDescription << " must use integer vector element type";
  }
  return success();
}

enum class MemoryRole {
  Unknown,
  GM,
  UB,
  Other,
};

static MemoryRole classifyMemoryRole(Type type) {
  auto memrefType = dyn_cast<BaseMemRefType>(type);
  if (!memrefType) {
    if (auto ptrType = dyn_cast<pto::PtrType>(type)) {
      switch (ptrType.getMemorySpace().getAddressSpace()) {
      case pto::AddressSpace::GM:
      case pto::AddressSpace::Zero:
        return MemoryRole::GM;
      case pto::AddressSpace::VEC:
        return MemoryRole::UB;
      default:
        return MemoryRole::Other;
      }
    }
    return MemoryRole::Other;
  }

  Attribute memorySpace = memrefType.getMemorySpace();
  if (!memorySpace) {
    return MemoryRole::Unknown;
  }

  if (auto addrSpace = dyn_cast<pto::AddressSpaceAttr>(memorySpace)) {
    switch (addrSpace.getAddressSpace()) {
    case pto::AddressSpace::GM:
    case pto::AddressSpace::Zero:
      return MemoryRole::GM;
    case pto::AddressSpace::VEC:
      return MemoryRole::UB;
    default:
      return MemoryRole::Other;
    }
  }

  if (auto intAttr = dyn_cast<IntegerAttr>(memorySpace)) {
    switch (intAttr.getInt()) {
    case static_cast<int64_t>(pto::AddressSpace::GM):
    case static_cast<int64_t>(pto::AddressSpace::Zero):
      return MemoryRole::GM;
    case static_cast<int64_t>(pto::AddressSpace::VEC):
      return MemoryRole::UB;
    default:
      return MemoryRole::Other;
    }
  }

  return MemoryRole::Other;
}

static bool isBufferLike(Type type) {
  return isa<BaseMemRefType, pto::PtrType>(type);
}

static int64_t getBufferElementByteSize(Type type) {
  Type elementType;
  if (auto ptrType = dyn_cast<pto::PtrType>(type)) {
    elementType = ptrType.getElementType();
  } else if (auto memrefType = dyn_cast<BaseMemRefType>(type)) {
    elementType = memrefType.getElementType();
  } else {
    return 0;
  }

  return getPTOStorageElemByteSize(elementType);
}

static Type getBufferElementType(Type type) {
  if (auto ptrType = dyn_cast<pto::PtrType>(type)) {
    return ptrType.getElementType();
  }
  if (auto memrefType = dyn_cast<BaseMemRefType>(type)) {
    return memrefType.getElementType();
  }
  return {};
}

static std::optional<AddressSpace> getBufferAddressSpace(Type type) {
  if (auto ptrType = dyn_cast<pto::PtrType>(type)) {
    return ptrType.getMemorySpace().getAddressSpace();
  }
  if (auto memrefType = dyn_cast<BaseMemRefType>(type)) {
    if (auto space =
            dyn_cast_or_null<pto::AddressSpaceAttr>(memrefType.getMemorySpace())) {
      return space.getAddressSpace();
    }
    if (auto intSpace = dyn_cast_or_null<IntegerAttr>(memrefType.getMemorySpace())) {
      return static_cast<AddressSpace>(intSpace.getInt());
    }
  }
  return std::nullopt;
}

template <typename BridgeLoadOp>
static LogicalResult verifyCubeBridgeLoadLikeOp(BridgeLoadOp op,
                                                AddressSpace expectedDstSpace,
                                                StringRef dstName) {
  if (!isBufferLike(op.getSource().getType()) ||
      !isBufferLike(op.getDestination().getType())) {
    return op.emitOpError("requires buffer-like source and destination");
  }

  if (getBufferAddressSpace(op.getSource().getType()) != AddressSpace::MAT) {
    return op.emitOpError("requires MAT source");
  }
  if (getBufferAddressSpace(op.getDestination().getType()) != expectedDstSpace) {
    return op.emitOpError()
           << "requires " << dstName << " destination";
  }

  int64_t sourceElemBytes = getBufferElementByteSize(op.getSource().getType());
  int64_t destinationElemBytes =
      getBufferElementByteSize(op.getDestination().getType());
  if (sourceElemBytes <= 0 || destinationElemBytes <= 0) {
    return op.emitOpError(
        "requires source and destination element types with known byte width");
  }
  if (sourceElemBytes != destinationElemBytes) {
    return op.emitOpError(
        "requires source and destination element byte widths to match");
  }

  return success();
}

static ParseResult parseRequiredOperandWithComma(
    OpAsmParser &parser, OpAsmParser::UnresolvedOperand &operand) {
  if (parser.parseOperand(operand)) {
    return failure();
  }
  (void)parser.parseOptionalComma();
  return success();
}

static ParseResult parseDmaTripleGroup(
    OpAsmParser &parser, StringRef keyword,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands) {
  if (parser.parseKeyword(keyword) || parser.parseLParen()) {
    return failure();
  }
  for (int i = 0; i < mlir::pto::kValue3; ++i) {
    OpAsmParser::UnresolvedOperand operand;
    if (parser.parseOperand(operand)) {
      return failure();
    }
    operands.push_back(operand);
    if (i != mlir::pto::kValue2 && parser.parseComma()) {
      return failure();
    }
  }
  return parser.parseRParen();
}

static ParseResult parseOptionalDmaTripleGroupAlias(
    OpAsmParser &parser, ArrayRef<StringRef> keywords,
    StringRef &parsedKeyword,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands) {
  parsedKeyword = {};
  for (StringRef keyword : keywords) {
    if (failed(parser.parseOptionalKeyword(keyword))) {
      continue;
    }
    parsedKeyword = keyword;
    if (parser.parseLParen()) {
      return failure();
    }
    for (int i = 0; i < mlir::pto::kValue3; ++i) {
      OpAsmParser::UnresolvedOperand operand;
      if (parser.parseOperand(operand)) {
        return failure();
      }
      operands.push_back(operand);
      if (i != mlir::pto::kValue2 && parser.parseComma()) {
        return failure();
      }
    }
    return parser.parseRParen();
  }
  return success();
}

static bool isDmaLoopKeyword(StringRef keyword) {
  if (keyword == "loop") {
    return true;
  }
  if (!keyword.consume_front("loop")) {
    return false;
  }
  if (keyword.empty()) {
    return false;
  }
  return llvm::all_of(keyword, llvm::isDigit);
}

static ParseResult parseDmaTripleTypes(OpAsmParser &parser,
                                       SmallVectorImpl<Type> &types) {
  for (int i = 0; i < mlir::pto::kValue3; ++i) {
    Type type;
    if (parser.parseType(type)) {
      return failure();
    }
    types.push_back(type);
    if (i != mlir::pto::kValue2 && parser.parseComma()) {
      return failure();
    }
  }
  return success();
}

static ParseResult parseDmaPadTypes(OpAsmParser &parser,
                                    SmallVectorImpl<Type> &types) {
  Type valueType;
  if (parser.parseType(valueType)) {
    return failure();
  }
  types.push_back(valueType);
  if (succeeded(parser.parseOptionalComma())) {
    Type leftType;
    Type rightType;
    if (parser.parseType(leftType) || parser.parseComma() ||
        parser.parseType(rightType)) {
      return failure();
    }
    types.push_back(leftType);
    types.push_back(rightType);
  }
  return success();
}

static void printDmaTripleGroup(OpAsmPrinter &printer, StringRef keyword,
                                Value first, Value second, Value third) {
  printer << " " << keyword << "(" << first << ", " << second << ", " << third
          << ")";
}

static void printDmaTripleTypes(OpAsmPrinter &printer, StringRef keyword,
                                Type first, Type second, Type third) {
  printer << ", " << keyword << " " << first << ", " << second << ", " << third;
}

static void printDmaPadGroup(OpAsmPrinter &printer, Value value, Value left,
                             Value right) {
  printer << " pad(" << value;
  if (left || right) {
    printer << ", " << left << ", " << right;
  }
  printer << ")";
}

static void printDmaPadTypes(OpAsmPrinter &printer, Type valueType,
                             Type leftType, Type rightType) {
  printer << ", pad " << valueType;
  if (leftType || rightType) {
    printer << ", " << leftType << ", " << rightType;
  }
}

static FailureOr<CubeLoadFracMode>
parseCubeLoadFracModeKeyword(StringRef keyword) {
  if (std::optional<CubeLoadFracMode> mode = symbolizeCubeLoadFracMode(keyword)) {
    return *mode;
  }
  return failure();
}

static ParseResult parseFixedKeywordOperandGroup(
    OpAsmParser &parser, StringRef keyword, int operandCount,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands) {
  if (parser.parseKeyword(keyword) || parser.parseLParen()) {
    return failure();
  }
  for (int i = 0; i < operandCount; ++i) {
    OpAsmParser::UnresolvedOperand operand;
    if (parser.parseOperand(operand)) {
      return failure();
    }
    operands.push_back(operand);
    if (i + 1 != operandCount && parser.parseComma()) {
      return failure();
    }
  }
  return parser.parseRParen();
}

static ParseResult parseFixedKeywordTypes(OpAsmParser &parser, StringRef keyword,
                                          int typeCount,
                                          SmallVectorImpl<Type> &types) {
  if (parser.parseKeyword(keyword)) {
    return failure();
  }
  for (int i = 0; i < typeCount; ++i) {
    Type type;
    if (parser.parseType(type)) {
      return failure();
    }
    types.push_back(type);
    if (i + 1 != typeCount && parser.parseComma()) {
      return failure();
    }
  }
  return success();
}

static ParseResult parseCubeLoadFracSrcLayoutGroup(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands) {
  if (parser.parseKeyword("src_layout") || parser.parseLParen()) {
    return failure();
  }
  OpAsmParser::UnresolvedOperand innerStride;
  if (parser.parseOperand(innerStride)) {
    return failure();
  }
  operands.push_back(innerStride);
  if (succeeded(parser.parseOptionalComma())) {
    OpAsmParser::UnresolvedOperand outerStride;
    if (parser.parseOperand(outerStride)) {
      return failure();
    }
    operands.push_back(outerStride);
  }
  return parser.parseRParen();
}

static ParseResult parseCubeLoadFracSrcLayoutTypes(OpAsmParser &parser,
                                                   SmallVectorImpl<Type> &types) {
  if (parser.parseKeyword("src_layout") || parser.parseLParen()) {
    return failure();
  }
  Type innerStrideType;
  if (parser.parseType(innerStrideType)) {
    return failure();
  }
  types.push_back(innerStrideType);
  if (succeeded(parser.parseOptionalComma())) {
    Type outerStrideType;
    if (parser.parseType(outerStrideType)) {
      return failure();
    }
    types.push_back(outerStrideType);
  }
  return parser.parseRParen();
}

static void printCubeLoadFracSrcLayoutGroup(OpAsmPrinter &printer,
                                            Value srcInnerStride,
                                            Value srcOuterStride) {
  printer << ", src_layout(" << srcInnerStride;
  if (srcOuterStride) {
    printer << ", " << srcOuterStride;
  }
  printer << ")";
}

static void printCubeLoadFracSrcLayoutTypes(OpAsmPrinter &printer,
                                            Type srcInnerStrideType,
                                            Type srcOuterStrideType) {
  printer << ", src_layout(" << srcInnerStrideType;
  if (srcOuterStrideType) {
    printer << ", " << srcOuterStrideType;
  }
  printer << ")";
}

static FailureOr<AccStoreMode> parseAccStoreModeKeyword(StringRef keyword) {
  if (std::optional<AccStoreMode> mode = symbolizeAccStoreMode(keyword)) {
    return *mode;
  }
  return failure();
}

[[maybe_unused]] static ParseResult parseAccStoreModeGroup(
    OpAsmParser &parser, StringRef &modeKeyword,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &modeOperands) {
  if (parser.parseKeyword(&modeKeyword)) {
    return failure();
  }
  if (failed(parseAccStoreModeKeyword(modeKeyword))) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected one of 'nz2nd', 'nz2dn', or 'nz2nz'");
  }
  auto parseModeOperandWithParens = [&]() -> ParseResult {
    OpAsmParser::UnresolvedOperand operand;
    if (parser.parseLParen() || parser.parseOperand(operand) || parser.parseRParen()) {
      return failure();
    }
    modeOperands.push_back(operand);
    return success();
  };
  auto parseModeOperandAfterLParen = [&]() -> ParseResult {
    OpAsmParser::UnresolvedOperand operand;
    if (parser.parseOperand(operand) || parser.parseRParen()) {
      return failure();
    }
    modeOperands.push_back(operand);
    return success();
  };

  switch (*parseAccStoreModeKeyword(modeKeyword)) {
  case AccStoreMode::Nz2nd:
    return success();
  case AccStoreMode::Nz2dn:
    (void)parser.parseOptionalComma();
    if (succeeded(parser.parseOptionalKeyword("loop0_src_stride"))) {
      return parseModeOperandWithParens();
    }
    if (failed(parser.parseOptionalLParen())) {
      return success();
    }
    return parseModeOperandAfterLParen();
  case AccStoreMode::Nz2nz:
    (void)parser.parseOptionalComma();
    if (succeeded(parser.parseOptionalKeyword("split"))) {
      return parseModeOperandWithParens();
    }
    if (failed(parser.parseOptionalLParen())) {
      return success();
    }
    return parseModeOperandAfterLParen();
  }
  return success();
}

[[maybe_unused]] static ParseResult
parseAccStoreModeTypes(OpAsmParser &parser, StringRef modeKeyword,
                       SmallVectorImpl<Type> &modeTypes) {
  if (parser.parseKeyword(modeKeyword)) {
    return failure();
  }
  auto parseModeTypeWithParens = [&]() -> ParseResult {
    Type modeType;
    if (parser.parseLParen() || parser.parseType(modeType) || parser.parseRParen()) {
      return failure();
    }
    modeTypes.push_back(modeType);
    return success();
  };
  auto parseModeTypeAfterLParen = [&]() -> ParseResult {
    Type modeType;
    if (parser.parseType(modeType) || parser.parseRParen()) {
      return failure();
    }
    modeTypes.push_back(modeType);
    return success();
  };

  switch (*parseAccStoreModeKeyword(modeKeyword)) {
  case AccStoreMode::Nz2nd:
    return success();
  case AccStoreMode::Nz2dn:
    (void)parser.parseOptionalComma();
    if (succeeded(parser.parseOptionalKeyword("loop0_src_stride"))) {
      return parseModeTypeWithParens();
    }
    if (failed(parser.parseOptionalLParen())) {
      return success();
    }
    return parseModeTypeAfterLParen();
  case AccStoreMode::Nz2nz:
    (void)parser.parseOptionalComma();
    if (succeeded(parser.parseOptionalKeyword("split"))) {
      return parseModeTypeWithParens();
    }
    if (failed(parser.parseOptionalLParen())) {
      return success();
    }
    return parseModeTypeAfterLParen();
  }
  return success();
}

[[maybe_unused]] static void printAccStoreModeGroup(OpAsmPrinter &printer,
                                                    AccStoreMode mode,
                                                    Value split,
                                                    Value loop0SrcStride) {
  printer << ", " << pto::stringifyAccStoreMode(mode);
  switch (mode) {
  case AccStoreMode::Nz2nd:
    return;
  case AccStoreMode::Nz2dn:
    if (loop0SrcStride) {
      printer << ", loop0_src_stride(" << loop0SrcStride << ")";
    }
    return;
  case AccStoreMode::Nz2nz:
    if (split) {
      printer << ", split(" << split << ")";
    }
    return;
  }
  llvm_unreachable("unexpected mte_l0c mode");
}

[[maybe_unused]] static void printAccStoreModeTypes(OpAsmPrinter &printer,
                                                    AccStoreMode mode,
                                                    Type splitType,
                                                    Type loop0SrcStrideType) {
  printer << ", " << pto::stringifyAccStoreMode(mode);
  switch (mode) {
  case AccStoreMode::Nz2nd:
    return;
  case AccStoreMode::Nz2dn:
    if (loop0SrcStrideType) {
      printer << ", loop0_src_stride(" << loop0SrcStrideType << ")";
    }
    return;
  case AccStoreMode::Nz2nz:
    if (splitType) {
      printer << ", split(" << splitType << ")";
    }
    return;
  }
  llvm_unreachable("unexpected mte_l0c mode");
}

[[maybe_unused]] static ParseResult parseMteL0cL1OptionalLoop3(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loop3CountOperands,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loop3SrcStrideOperands,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loop3DstStrideOperands) {
  StringRef parsedKeyword;
  SmallVector<OpAsmParser::UnresolvedOperand, mlir::pto::kValue3> loop3Operands;
  if (parseOptionalDmaTripleGroupAlias(parser, {"loop3"}, parsedKeyword,
                                       loop3Operands)) {
    return failure();
  }
  if (!parsedKeyword.empty()) {
    loop3CountOperands.push_back(loop3Operands[0]);
    loop3SrcStrideOperands.push_back(loop3Operands[1]);
    loop3DstStrideOperands.push_back(loop3Operands[mlir::pto::kValue2]);
  }
  return success();
}

[[maybe_unused]] static ParseResult parseMteL0cL1OptionalFpc(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &fpcOperands) {
  if (failed(parser.parseOptionalKeyword("fpc"))) {
    return success();
  }
  if (parser.parseLParen()) {
    return failure();
  }
  OpAsmParser::UnresolvedOperand operand;
  if (parser.parseOperand(operand) || parser.parseRParen()) {
    return failure();
  }
  fpcOperands.push_back(operand);
  return success();
}

[[maybe_unused]] static void printMteL0cL1OptionalFpc(OpAsmPrinter &printer,
                                                      Value fpc) {
  if (fpc) {
    printer << ", fpc(" << fpc << ")";
  }
}

[[maybe_unused]] static void
printMteL0cL1OptionalFpcType(OpAsmPrinter &printer, Type fpcType) {
  if (fpcType) {
    printer << ", fpc(" << fpcType << ")";
  }
}

[[maybe_unused]] static ParseResult parseMteL0cL1OptionalLoop3Types(
    OpAsmParser &parser, SmallVectorImpl<Type> &loop3CountTypes,
    SmallVectorImpl<Type> &loop3SrcStrideTypes,
    SmallVectorImpl<Type> &loop3DstStrideTypes, StringRef opName) {
  if (succeeded(parser.parseOptionalComma())) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword)) {
      return failure();
    }
    if (keyword != "loop3") {
      return parser.emitError(parser.getCurrentLocation(), "expected 'loop3'");
    }
    SmallVector<Type> loop3GroupTypes;
    if (parseDmaTripleTypes(parser, loop3GroupTypes)) {
      return failure();
    }
    loop3CountTypes.push_back(loop3GroupTypes[0]);
    loop3SrcStrideTypes.push_back(loop3GroupTypes[1]);
    loop3DstStrideTypes.push_back(loop3GroupTypes[mlir::pto::kValue2]);
    if (succeeded(parser.parseOptionalComma())) {
      return parser.emitError(parser.getCurrentLocation(),
                              (Twine(opName) +
                               " accepts at most one loop3 group")
                                  .str());
    }
  }
  return success();
}

[[maybe_unused]] static LogicalResult verifyAccStoreLikeModeOperands(
    Operation *op, AccStoreMode mode, Value split, Value loop0SrcStride,
    Value loop3Count, Value loop3SrcStride, Value loop3DstStride,
    StringRef nz2ndSplitError, StringRef nz2ndLoop0Error,
    StringRef nz2dnSplitError, StringRef nz2nzLoop0Error,
    StringRef nz2nzLoop3Error) {
  bool hasLoop3Count = static_cast<bool>(loop3Count);
  bool hasLoop3SrcStride = static_cast<bool>(loop3SrcStride);
  bool hasLoop3DstStride = static_cast<bool>(loop3DstStride);
  if ((hasLoop3Count != hasLoop3SrcStride) ||
      (hasLoop3Count != hasLoop3DstStride)) {
    return op->emitOpError(
        "requires loop3 count, src stride, and dst stride to appear together");
  }

  switch (mode) {
  case AccStoreMode::Nz2nd:
    if (split) {
      return op->emitOpError(nz2ndSplitError);
    }
    if (loop0SrcStride) {
      return op->emitOpError(nz2ndLoop0Error);
    }
    return success();
  case AccStoreMode::Nz2dn:
    if (split) {
      return op->emitOpError(nz2dnSplitError);
    }
    return success();
  case AccStoreMode::Nz2nz:
    if (loop0SrcStride) {
      return op->emitOpError(nz2nzLoop0Error);
    }
    if (loop3Count) {
      return op->emitOpError(nz2nzLoop3Error);
    }
    return success();
  }
  llvm_unreachable("unexpected mte_l0c mode");
}

struct StructuredAccStoreAsmState {
  std::optional<AccStoreUnitFlagCtrl> unitFlag;
  std::optional<AccStoreQuantPreMode> preQuantMode;
  std::optional<ReluPreMode> preReluMode;
  std::optional<AccStoreMode> mode;
  std::optional<AccStoreAtomicType> atomicType;
  std::optional<AccStoreAtomicOp> atomicOp;
  std::optional<AccStoreSatMode> satMode;

  SmallVector<OpAsmParser::UnresolvedOperand, 1> preQuantOperands;
  SmallVector<OpAsmParser::UnresolvedOperand, 1> preReluOperands;
  SmallVector<OpAsmParser::UnresolvedOperand, 1> clipValueOperands;
  SmallVector<OpAsmParser::UnresolvedOperand, 1> splitOperands;
  SmallVector<OpAsmParser::UnresolvedOperand, 1> loop0SrcStrideOperands;
  SmallVector<OpAsmParser::UnresolvedOperand, 1> loop3CountOperands;
  SmallVector<OpAsmParser::UnresolvedOperand, 1> loop3SrcStrideOperands;
  SmallVector<OpAsmParser::UnresolvedOperand, 1> loop3DstStrideOperands;

  SmallVector<Type, 1> preQuantTypes;
  SmallVector<Type, 1> preReluTypes;
  SmallVector<Type, 1> clipValueTypes;
  SmallVector<Type, 1> splitTypes;
  SmallVector<Type, 1> loop0SrcStrideTypes;
  SmallVector<Type, 1> loop3CountTypes;
  SmallVector<Type, 1> loop3SrcStrideTypes;
  SmallVector<Type, 1> loop3DstStrideTypes;
};

enum class StructuredAccStoreClauseKind {
  UnitFlag = 0,
  PreQuant = 1,
  PreRelu = 2,
  Layout = 3,
  Loop3 = 4,
  Sat = 5,
  Atomic = 6
};

static bool isStructuredAccStoreVectorQuantMode(AccStoreQuantPreMode mode) {
  switch (mode) {
  case AccStoreQuantPreMode::QF322HIF8PreVec:
  case AccStoreQuantPreMode::QF322HIF8PreHybridVec:
  case AccStoreQuantPreMode::DEQS32IntVec:
  case AccStoreQuantPreMode::REQ8Vec:
  case AccStoreQuantPreMode::DEQF16Vec:
  case AccStoreQuantPreMode::QF322FP8PreVec:
  case AccStoreQuantPreMode::QF322F32PreVec:
  case AccStoreQuantPreMode::QF162B8PreVec:
  case AccStoreQuantPreMode::QF162S4PreVec:
  case AccStoreQuantPreMode::REQ4Vec:
  case AccStoreQuantPreMode::QF322B8PreVec:
  case AccStoreQuantPreMode::QF322S4PreVec:
  case AccStoreQuantPreMode::DEQS16Vec:
  case AccStoreQuantPreMode::QF162S16PreVec:
  case AccStoreQuantPreMode::QF322F16PreVec:
  case AccStoreQuantPreMode::QF322BF16PreVec:
  case AccStoreQuantPreMode::QS322BF16PreVec:
    return true;
  default:
    return false;
  }
}

static bool isStructuredAccStoreScalingPayload(Value value) {
  auto ptrType = dyn_cast_or_null<PtrType>(value.getType());
  return ptrType &&
         ptrType.getMemorySpace().getAddressSpace() == AddressSpace::SCALING;
}

[[maybe_unused]] static bool isStructuredAccStoreScalingPayloadType(Type type) {
  auto ptrType = dyn_cast_or_null<PtrType>(type);
  return ptrType &&
         ptrType.getMemorySpace().getAddressSpace() == AddressSpace::SCALING;
}

static Type getStructuredAccStoreScalingElementType(Value value) {
  auto ptrType = dyn_cast_or_null<PtrType>(value.getType());
  if (!ptrType ||
      ptrType.getMemorySpace().getAddressSpace() != AddressSpace::SCALING) {
    return {};
  }
  return ptrType.getElementType();
}

[[maybe_unused]] static bool isStructuredAccStoreIntegerPayload(Value value) {
  return value.getType().isSignlessInteger();
}

static bool isStructuredAccStoreClipPayloadForUInt8(Type type) {
  auto intType = dyn_cast<IntegerType>(type);
  if (!intType || intType.getWidth() != mlir::pto::kValue16) {
    return false;
  }
  return intType.isUnsigned() || intType.isSignless();
}

static bool isStructuredAccStoreClipPayloadForSignedInt(Type type) {
  auto intType = dyn_cast<IntegerType>(type);
  if (!intType) {
    return false;
  }
  unsigned width = intType.getWidth();
  if (width != mlir::pto::kValue4 && width != 8 && width != 16) {
    return false;
  }
  return intType.isSigned() || intType.isSignless();
}

static bool isStructuredAccStoreFloatScalarPayloadType(Type type) {
  return type.isF16() || type.isF32() || type.isBF16();
}

static bool isStructuredAccStoreFloatScalarPayload(Value value) {
  return isStructuredAccStoreFloatScalarPayloadType(value.getType());
}

[[maybe_unused]] static bool isStructuredAccStoreIntegerPayloadType(Type type) {
  return type.isSignlessInteger();
}

static bool isStructuredAccStoreClipSupportedElementType(Type type) {
  if (auto floatType = dyn_cast<FloatType>(type)) {
    return floatType.isF16();
  }
  auto intType = dyn_cast<IntegerType>(type);
  if (!intType) {
    return false;
  }
  if (intType.isUnsignedInteger(mlir::pto::kValue8)) {
    return true;
  }
  if (intType.isSignlessInteger(mlir::pto::kValue4) || intType.isSignlessInteger(8) ||
      intType.isSignlessInteger(mlir::pto::kValue16)) {
    return true;
  }
  if (intType.isSignedInteger(mlir::pto::kValue4) || intType.isSignedInteger(8) ||
      intType.isSignedInteger(16)) {
    return true;
  }
  return false;
}

static LogicalResult verifyStructuredAccStoreClipPayload(Operation *op,
                                                        Type destinationElementType,
                                                        Value clipValue) {
  if (!clipValue) {
    return success();
  }

  Type clipType = clipValue.getType();
  if (destinationElementType.isF16()) {
    if (!clipType.isF16()) {
      return op->emitOpError("clip for f16 destination requires f16 payload");
    }
    return success();
  }

  auto intType = dyn_cast<IntegerType>(destinationElementType);
  if (!intType) {
    return op->emitOpError()
           << "clip requires destination element type to be f16, ui8, or signed 4/8/16-bit integer, got "
           << destinationElementType;
  }

  if (intType.isUnsignedInteger(mlir::pto::kValue8)) {
    if (!isStructuredAccStoreClipPayloadForUInt8(clipType)) {
      return op->emitOpError("clip for ui8 destination requires ui16/signless i16 payload");
    }
    return success();
  }

  if (intType.isSignlessInteger(mlir::pto::kValue4) ||
      intType.isSignlessInteger(mlir::pto::kValue8) ||
      intType.isSignlessInteger(mlir::pto::kValue16) ||
      intType.isSignedInteger(mlir::pto::kValue4) ||
      intType.isSignedInteger(mlir::pto::kValue8) ||
      intType.isSignedInteger(mlir::pto::kValue16)) {
    if (!isStructuredAccStoreClipPayloadForSignedInt(clipType)) {
      return op->emitOpError("clip for signed 4/8/16-bit destination requires signed/signless i4/i8/i16 payload");
    }
    return success();
  }

  return op->emitOpError()
         << "clip requires destination element type to be f16, ui8, or signed 4/8/16-bit integer, got "
         << destinationElementType;
}

static bool isStructuredAccStoreFloatPreQuantMode(AccStoreQuantPreMode mode) {
  switch (mode) {
  case AccStoreQuantPreMode::F32F16:
  case AccStoreQuantPreMode::QF322HIF8PreVec:
  case AccStoreQuantPreMode::QF322HIF8PreScalar:
  case AccStoreQuantPreMode::QF322HIF8PreHybridVec:
  case AccStoreQuantPreMode::QF322HIF8PreHybridScalar:
  case AccStoreQuantPreMode::QF322FP8PreVec:
  case AccStoreQuantPreMode::QF322FP8PreScalar:
  case AccStoreQuantPreMode::QF322F32PreVec:
  case AccStoreQuantPreMode::QF322F32PreScalar:
  case AccStoreQuantPreMode::F32BF16:
  case AccStoreQuantPreMode::QF162B8PreVec:
  case AccStoreQuantPreMode::QF162B8PreScalar:
  case AccStoreQuantPreMode::QF162S4PreVec:
  case AccStoreQuantPreMode::QF162S4PreScalar:
  case AccStoreQuantPreMode::QF322B8PreVec:
  case AccStoreQuantPreMode::QF322B8PreScalar:
  case AccStoreQuantPreMode::QF322S4PreVec:
  case AccStoreQuantPreMode::QF322S4PreScalar:
  case AccStoreQuantPreMode::QF322F16PreVec:
  case AccStoreQuantPreMode::QF322F16PreScalar:
  case AccStoreQuantPreMode::QF322BF16PreVec:
  case AccStoreQuantPreMode::QF322BF16PreScalar:
    return true;
  default:
    return false;
  }
}

static bool isStructuredAccStoreInt32PreQuantMode(AccStoreQuantPreMode mode) {
  switch (mode) {
  case AccStoreQuantPreMode::DEQS32IntVec:
  case AccStoreQuantPreMode::DEQS32IntScalar:
  case AccStoreQuantPreMode::REQ8Vec:
  case AccStoreQuantPreMode::REQ8Scalar:
  case AccStoreQuantPreMode::DEQF16Vec:
  case AccStoreQuantPreMode::DEQF16Scalar:
  case AccStoreQuantPreMode::DEQS16Vec:
  case AccStoreQuantPreMode::DEQS16Scalar:
  case AccStoreQuantPreMode::QF162S16PreVec:
  case AccStoreQuantPreMode::QF162S16PreScalar:
  case AccStoreQuantPreMode::QS322BF16PreVec:
  case AccStoreQuantPreMode::QS322BF16PreScalar:
    return true;
  default:
    return false;
  }
}

enum class StructuredAccStoreDestinationFamily {
  Any,
  F32,
  F16,
  BF16,
  I32,
  I16,
  I8,
  I4,
  FP8
};

static StructuredAccStoreDestinationFamily
getStructuredAccStorePreQuantDestinationFamily(AccStoreQuantPreMode mode) {
  switch (mode) {
  case AccStoreQuantPreMode::F32F16:
  case AccStoreQuantPreMode::QF322F16PreVec:
  case AccStoreQuantPreMode::QF322F16PreScalar:
  case AccStoreQuantPreMode::DEQF16Vec:
  case AccStoreQuantPreMode::DEQF16Scalar:
    return StructuredAccStoreDestinationFamily::F16;
  case AccStoreQuantPreMode::F32BF16:
  case AccStoreQuantPreMode::QF322BF16PreVec:
  case AccStoreQuantPreMode::QF322BF16PreScalar:
  case AccStoreQuantPreMode::QS322BF16PreVec:
  case AccStoreQuantPreMode::QS322BF16PreScalar:
    return StructuredAccStoreDestinationFamily::BF16;
  case AccStoreQuantPreMode::QF322F32PreVec:
  case AccStoreQuantPreMode::QF322F32PreScalar:
    return StructuredAccStoreDestinationFamily::F32;
  case AccStoreQuantPreMode::QF322HIF8PreVec:
  case AccStoreQuantPreMode::QF322HIF8PreScalar:
  case AccStoreQuantPreMode::QF322HIF8PreHybridVec:
  case AccStoreQuantPreMode::QF322HIF8PreHybridScalar:
  case AccStoreQuantPreMode::QF322FP8PreVec:
  case AccStoreQuantPreMode::QF322FP8PreScalar:
    return StructuredAccStoreDestinationFamily::FP8;
  case AccStoreQuantPreMode::DEQS32IntVec:
  case AccStoreQuantPreMode::DEQS32IntScalar:
    return StructuredAccStoreDestinationFamily::I32;
  case AccStoreQuantPreMode::QF162S16PreVec:
  case AccStoreQuantPreMode::QF162S16PreScalar:
  case AccStoreQuantPreMode::DEQS16Vec:
  case AccStoreQuantPreMode::DEQS16Scalar:
    return StructuredAccStoreDestinationFamily::I16;
  case AccStoreQuantPreMode::QF162B8PreVec:
  case AccStoreQuantPreMode::QF162B8PreScalar:
  case AccStoreQuantPreMode::REQ8Vec:
  case AccStoreQuantPreMode::REQ8Scalar:
  case AccStoreQuantPreMode::QF322B8PreVec:
  case AccStoreQuantPreMode::QF322B8PreScalar:
    return StructuredAccStoreDestinationFamily::I8;
  case AccStoreQuantPreMode::QF162S4PreVec:
  case AccStoreQuantPreMode::QF162S4PreScalar:
  case AccStoreQuantPreMode::REQ4Vec:
  case AccStoreQuantPreMode::REQ4Scalar:
  case AccStoreQuantPreMode::QF322S4PreVec:
  case AccStoreQuantPreMode::QF322S4PreScalar:
    return StructuredAccStoreDestinationFamily::I4;
  case AccStoreQuantPreMode::NoConvert:
    return StructuredAccStoreDestinationFamily::Any;
  }
  llvm_unreachable("unknown acc-store pre_quant mode");
}

static bool isStructuredAccStoreDestinationFamily(
    Type type, StructuredAccStoreDestinationFamily family) {
  switch (family) {
  case StructuredAccStoreDestinationFamily::Any:
    return true;
  case StructuredAccStoreDestinationFamily::F32:
    return type.isF32();
  case StructuredAccStoreDestinationFamily::F16:
    return type.isF16();
  case StructuredAccStoreDestinationFamily::BF16:
    return type.isBF16();
  case StructuredAccStoreDestinationFamily::I32:
    if (auto intType = dyn_cast<IntegerType>(type)) {
      return intType.getWidth() == mlir::pto::kValue32;
    }
    return false;
  case StructuredAccStoreDestinationFamily::I16:
    if (auto intType = dyn_cast<IntegerType>(type)) {
      return intType.getWidth() == mlir::pto::kValue16 && !intType.isUnsigned();
    }
    return false;
  case StructuredAccStoreDestinationFamily::I8:
    if (auto intType = dyn_cast<IntegerType>(type)) {
      return intType.getWidth() == mlir::pto::kValue8;
    }
    return false;
  case StructuredAccStoreDestinationFamily::I4:
    if (auto intType = dyn_cast<IntegerType>(type)) {
      return intType.getWidth() == mlir::pto::kValue4 && !intType.isUnsigned();
    }
    return false;
  case StructuredAccStoreDestinationFamily::FP8:
    return pto::isPTOFloat8Type(type) || pto::isPTOHiFloat8Type(type) ||
           pto::isPTOHiFloat8x2Type(type);
  }
  llvm_unreachable("unknown acc-store destination family");
}

static ParseResult parseStructuredAccStoreUnitFlag(OpAsmParser &parser,
                                                   StructuredAccStoreAsmState &state) {
  if (state.unitFlag) {
    return parser.emitError(parser.getCurrentLocation(), "duplicate unit_flag clause");
  }
  StringRef keyword;
  if (parser.parseLParen() || parser.parseKeyword(&keyword) || parser.parseRParen()) {
    return failure();
  }
  if (keyword == "check_only") {
    state.unitFlag = AccStoreUnitFlagCtrl::CheckOnly;
  }
  else if (keyword == "check_and_clear") {
    state.unitFlag = AccStoreUnitFlagCtrl::CheckAndClear;
  }
  else {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected 'check_only' or 'check_and_clear'");
}
  return success();
}

static ParseResult parseStructuredAccStorePreQuant(
    OpAsmParser &parser, StructuredAccStoreAsmState &state) {
  if (state.preQuantMode) {
    return parser.emitError(parser.getCurrentLocation(), "duplicate pre_quant clause");
  }
  OpAsmParser::UnresolvedOperand payload;
  StringRef modeKeyword;
  if (parser.parseLParen() || parser.parseOperand(payload) || parser.parseComma() ||
      parser.parseKeyword("mode") || parser.parseEqual() ||
      parser.parseKeyword(&modeKeyword) || parser.parseRParen()) {
    return failure();
  }
  auto mode = symbolizeAccStoreQuantPreMode(modeKeyword);
  if (!mode) {
    return parser.emitError(parser.getCurrentLocation(), "invalid pre_quant mode");
  }
  state.preQuantOperands.push_back(payload);
  state.preQuantMode = *mode;
  return success();
}

static ParseResult parseStructuredAccStorePreRelu(
    OpAsmParser &parser, StructuredAccStoreAsmState &state) {
  if (state.preReluMode) {
    return parser.emitError(parser.getCurrentLocation(), "duplicate pre_relu clause");
  }
  StringRef modeKeyword;
  bool hasPayload = false;
  OpAsmParser::UnresolvedOperand payload;
  if (parser.parseLParen()) {
    return failure();
  }
  if (failed(parser.parseOptionalKeyword("mode"))) {
    hasPayload = true;
    if (parser.parseOperand(payload) || parser.parseComma() ||
        parser.parseKeyword("mode")) {
      return failure();
    }
  }
  if (parser.parseEqual() || parser.parseKeyword(&modeKeyword)) {
    return failure();
  }
  auto mode = symbolizeReluPreMode(modeKeyword);
  if (!mode) {
    return parser.emitError(parser.getCurrentLocation(), "invalid pre_relu mode");
  }
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseKeyword("clip") || parser.parseEqual()) {
      return failure();
    }
    if (!state.clipValueOperands.empty()) {
      return parser.emitError(parser.getCurrentLocation(),
                              "duplicate clip payload in pre_relu clause");
    }
    OpAsmParser::UnresolvedOperand clipValue;
    if (parser.parseOperand(clipValue)) {
      return failure();
    }
    state.clipValueOperands.push_back(clipValue);
  }
  if (parser.parseRParen()) {
    return failure();
  }

  if (hasPayload) {
    state.preReluOperands.push_back(payload);
  }
  state.preReluMode = *mode;
  return success();
}

static ParseResult parseStructuredAccStoreLayout(
    OpAsmParser &parser, StructuredAccStoreAsmState &state, StringRef keyword) {
  auto mode = parseAccStoreModeKeyword(keyword);
  if (failed(mode)) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected one of 'nz2nd', 'nz2dn', or 'nz2nz'");
  }
  if (state.mode) {
    return parser.emitError(parser.getCurrentLocation(), "duplicate layout clause");
  }
  state.mode = *mode;
  if (*mode == AccStoreMode::Nz2dn) {
    if (succeeded(parser.parseOptionalLParen())) {
      OpAsmParser::UnresolvedOperand operand;
      if (parser.parseOperand(operand) || parser.parseRParen()) {
        return failure();
      }
      state.loop0SrcStrideOperands.push_back(operand);
    }
  } else if (*mode == AccStoreMode::Nz2nz) {
    if (succeeded(parser.parseOptionalLParen())) {
      OpAsmParser::UnresolvedOperand operand;
      if (parser.parseOperand(operand) || parser.parseRParen()) {
        return failure();
      }
      state.splitOperands.push_back(operand);
    }
  }
  return success();
}

static ParseResult parseStructuredAccStoreLoop3(
    OpAsmParser &parser, StructuredAccStoreAsmState &state) {
  if (!state.loop3CountOperands.empty()) {
    return parser.emitError(parser.getCurrentLocation(), "duplicate loop3 clause");
  }
  OpAsmParser::UnresolvedOperand count;
  OpAsmParser::UnresolvedOperand srcStride;
  OpAsmParser::UnresolvedOperand dstStride;
  if (parser.parseLParen() || parser.parseOperand(count) || parser.parseComma() ||
      parser.parseOperand(srcStride) || parser.parseComma() ||
      parser.parseOperand(dstStride) || parser.parseRParen()) {
    return failure();
  }
  state.loop3CountOperands.push_back(count);
  state.loop3SrcStrideOperands.push_back(srcStride);
  state.loop3DstStrideOperands.push_back(dstStride);
  return success();
}

static ParseResult parseStructuredAccStoreAtomic(
    OpAsmParser &parser, StructuredAccStoreAsmState &state) {
  if (state.atomicType || state.atomicOp) {
    return parser.emitError(parser.getCurrentLocation(), "duplicate atomic clause");
  }
  StringRef typeKeyword;
  StringRef opKeyword;
  if (parser.parseLParen() || parser.parseKeyword("type") || parser.parseEqual() ||
      parser.parseKeyword(&typeKeyword) || parser.parseComma() ||
      parser.parseKeyword("op") || parser.parseEqual() ||
      parser.parseKeyword(&opKeyword) || parser.parseRParen()) {
    return failure();
  }
  auto type = symbolizeAccStoreAtomicType(typeKeyword);
  auto op = symbolizeAccStoreAtomicOp(opKeyword);
  if (!type) {
    return parser.emitError(parser.getCurrentLocation(), "invalid atomic type");
  }
  if (!op) {
    return parser.emitError(parser.getCurrentLocation(), "invalid atomic op");
  }
  state.atomicType = *type;
  state.atomicOp = *op;
  return success();
}

static ParseResult parseStructuredAccStoreClauses(
    OpAsmParser &parser, StructuredAccStoreAsmState &state) {
  int lastClause = -1;
  bool seenClause = false;
  while (true) {
    if (seenClause) {
      if (failed(parser.parseOptionalComma())) {
        return success();
      }
    }
    StringRef keyword;
    OptionalParseResult optParseResult = parser.parseOptionalKeyword(&keyword);
    if (!optParseResult.has_value() || failed(*optParseResult)) {
      if (!seenClause) {
        return success();
      }
      return parser.emitError(parser.getCurrentLocation(), "expected valid keyword");
    }
    seenClause = true;

    StructuredAccStoreClauseKind kind;
    if (keyword == "unit_flag") {
      kind = StructuredAccStoreClauseKind::UnitFlag;
    }
    else if (keyword == "pre_quant") {
      kind = StructuredAccStoreClauseKind::PreQuant;
    }
    else if (keyword == "pre_relu") {
      kind = StructuredAccStoreClauseKind::PreRelu;
    }
    else if (keyword == "nz2nd" || keyword == "nz2dn" || keyword == "nz2nz") {
      kind = StructuredAccStoreClauseKind::Layout;
    }
    else if (keyword == "loop3") {
      kind = StructuredAccStoreClauseKind::Loop3;
    }
    else if (keyword == "sat" || keyword == "nosat") {
      kind = StructuredAccStoreClauseKind::Sat;
    }
    else if (keyword == "atomic") {
      kind = StructuredAccStoreClauseKind::Atomic;
    }
    else {
      return parser.emitError(parser.getCurrentLocation(), "unknown mte_l0c clause");
}

    if (static_cast<int>(kind) < lastClause) {
      return parser.emitError(parser.getCurrentLocation(),
                              "mte_l0c clauses must follow canonical order");
    }
    lastClause = static_cast<int>(kind);

    ParseResult parseResult = success();
    switch (kind) {
    case StructuredAccStoreClauseKind::UnitFlag:
      parseResult = parseStructuredAccStoreUnitFlag(parser, state);
      break;
    case StructuredAccStoreClauseKind::PreQuant:
      parseResult = parseStructuredAccStorePreQuant(parser, state);
      break;
    case StructuredAccStoreClauseKind::PreRelu:
      parseResult = parseStructuredAccStorePreRelu(parser, state);
      break;
    case StructuredAccStoreClauseKind::Layout:
      parseResult = parseStructuredAccStoreLayout(parser, state, keyword);
      break;
    case StructuredAccStoreClauseKind::Loop3:
      parseResult = parseStructuredAccStoreLoop3(parser, state);
      break;
    case StructuredAccStoreClauseKind::Sat:
      if (state.satMode) {
        return parser.emitError(parser.getCurrentLocation(), "duplicate sat/nosat clause");
      }
      if (keyword == "nosat") {
        state.satMode = AccStoreSatMode::NoSat;
        break;
      }
      if (succeeded(parser.parseOptionalLParen())) {
        StringRef satOption;
        if (parser.parseKeyword(&satOption) || satOption != "preserve_nan") {
          return parser.emitError(parser.getCurrentLocation(),
                                  "expected preserve_nan");
        }
        if (parser.parseRParen()) {
          return failure();
        }
        state.satMode = AccStoreSatMode::SatPreserveNan;
      } else {
        state.satMode = AccStoreSatMode::Sat;
      }
      break;
    case StructuredAccStoreClauseKind::Atomic:
      parseResult = parseStructuredAccStoreAtomic(parser, state);
      break;
    }
    if (failed(parseResult)) {
      return failure();
    }
  }
}

static ParseResult parseStructuredOptionalType(OpAsmParser &parser,
                                               SmallVectorImpl<Type> &types) {
  Type type;
  if (parser.parseType(type)) {
    return failure();
  }
  types.push_back(type);
  return success();
}

static LogicalResult verifyStructuredAccStoreLike(
    Operation *op, Type srcType, Type dstType, Value preQuant, Value preRelu,
    Value clipValue,
    Value split, Value loop0SrcStride, Value loop3Count, Value loop3SrcStride,
    Value loop3DstStride,
    std::optional<AccStoreUnitFlagCtrl> unitFlag,
    std::optional<AccStoreQuantPreMode> preQuantMode,
    std::optional<ReluPreMode> preReluMode, std::optional<AccStoreMode> mode,
    std::optional<AccStoreAtomicType> atomicType,
    std::optional<AccStoreAtomicOp> atomicOp, bool allowAtomic) {
  auto getBufferElementType = [](Type type) -> Type {
    if (auto ptrType = dyn_cast<pto::PtrType>(type)) {
      return ptrType.getElementType();
    }
    if (auto memrefType = dyn_cast<BaseMemRefType>(type)) {
      return memrefType.getElementType();
    }
    return {};
  };
  Type sourceElementType = getBufferElementType(srcType);
  Type destinationElementType = getBufferElementType(dstType);

  if (static_cast<bool>(preQuant) != static_cast<bool>(preQuantMode)) {
    return op->emitOpError("pre_quant requires payload and mode together");
  }
  if (preQuantMode) {
    if (*preQuantMode == AccStoreQuantPreMode::NoConvert) {
      // The no_convert keyword carries no quantization parameters; the
      // syntactic payload operand is ignored for compatibility with the
      // structured pre_quant clause form.
    } else if (isStructuredAccStoreVectorQuantMode(*preQuantMode)) {
      if (!isStructuredAccStoreScalingPayload(preQuant)) {
        return op->emitOpError("vector pre_quant mode requires scaling pointer payload");
      }
      if (!isStructuredAccStoreFloatScalarPayloadType(
              getStructuredAccStoreScalingElementType(preQuant))) {
        return op->emitOpError(
            "vector pre_quant mode requires scaling pointer element type to be f16, bf16, or f32");
      }
    } else if (!isStructuredAccStoreFloatScalarPayload(preQuant)) {
      return op->emitOpError(
          "scalar pre_quant mode requires f16/bf16/f32 payload");
    }

    auto emitIncompatibleQuantModeError = [&]() -> LogicalResult {
      return op->emitOpError()
             << "pre_quant mode " << stringifyAccStoreQuantPreMode(*preQuantMode)
             << " is incompatible with source element type " << sourceElementType
             << " and destination element type " << destinationElementType;
    };

    if (*preQuantMode != AccStoreQuantPreMode::NoConvert) {
      if (isa<Float32Type>(sourceElementType)) {
        if (!isStructuredAccStoreFloatPreQuantMode(*preQuantMode)) {
          return emitIncompatibleQuantModeError();
        }
      } else if (sourceElementType.isSignlessInteger(mlir::pto::kValue32)) {
        if (!isStructuredAccStoreInt32PreQuantMode(*preQuantMode)) {
          return emitIncompatibleQuantModeError();
        }
      } else {
        return op->emitOpError()
               << "pre_quant requires source element type to be f32 or i32, got "
               << sourceElementType;
      }

      StructuredAccStoreDestinationFamily destinationFamily =
          getStructuredAccStorePreQuantDestinationFamily(*preQuantMode);
      if (!isStructuredAccStoreDestinationFamily(destinationElementType,
                                                 destinationFamily)) {
        return emitIncompatibleQuantModeError();
      }
    }
  }

  if (clipValue && !isStructuredAccStoreClipSupportedElementType(destinationElementType)) {
    return op->emitOpError()
           << "clip requires destination element type to be f16, ui8, or signed 4/8/16-bit integer, got "
           << destinationElementType;
  }
  if (failed(verifyStructuredAccStoreClipPayload(op, destinationElementType,
                                                 clipValue))) {
    return failure();
  }

  if (!preReluMode) {
    if (preRelu) {
      return op->emitOpError("pre_relu payload requires pre_relu mode");
    }
    if (clipValue) {
      return op->emitOpError("clip requires pre_relu clause");
    }
  } else {
    switch (*preReluMode) {
    case ReluPreMode::NoRelu:
      if (preRelu) {
        return op->emitOpError("mode does not accept pre_relu payload");
      }
      break;
    case ReluPreMode::NormalRelu:
      if (preRelu) {
        return op->emitOpError("mode does not accept pre_relu payload");
      }
      break;
    case ReluPreMode::ScalarRelu:
      if (!preRelu) {
        return op->emitOpError("scalar_relu requires payload");
      }
      if (!isStructuredAccStoreFloatScalarPayload(preRelu)) {
        return op->emitOpError("scalar_relu requires f16/bf16/f32 payload");
      }
      break;
    case ReluPreMode::VectorRelu:
      if (!preRelu) {
        return op->emitOpError("vector_relu requires payload");
      }
      if (!isStructuredAccStoreScalingPayload(preRelu)) {
        return op->emitOpError("vector_relu requires scaling pointer payload");
      }
      if (!isStructuredAccStoreFloatScalarPayloadType(
              getStructuredAccStoreScalingElementType(preRelu))) {
        return op->emitOpError(
            "vector_relu requires scaling pointer element type to be f16, bf16, or f32");
      }
      break;
    case ReluPreMode::Pwl:
      return op->emitOpError("pwl is not supported for target_profile mte_l0c_l1");
    }
  }

  bool hasLoop3 = static_cast<bool>(loop3Count) || static_cast<bool>(loop3SrcStride) ||
                  static_cast<bool>(loop3DstStride);
  if (hasLoop3 && !(loop3Count && loop3SrcStride && loop3DstStride)) {
    return op->emitOpError("loop3 requires count, src stride, and dst stride together");
  }

  if (!mode) {
    if (split) {
      return op->emitOpError("split requires nz2nz");
    }
    if (loop0SrcStride) {
      return op->emitOpError("loop0_src_stride requires nz2dn");
    }
    if (loop3Count) {
      return op->emitOpError("loop3 requires nz2nd or nz2dn");
    }
  } else {
    switch (*mode) {
    case AccStoreMode::Nz2nd:
      if (split) {
        return op->emitOpError("nz2nd does not accept split");
      }
      if (loop0SrcStride) {
        return op->emitOpError("nz2nd does not accept loop0_src_stride");
      }
      break;
    case AccStoreMode::Nz2dn: {
      if (!loop0SrcStride) {
        return op->emitOpError("nz2dn requires loop0_src_stride");
      }
      if (split) {
        return op->emitOpError("nz2dn does not accept split");
      }
      APInt loop0Value;
      if (unitFlag && *unitFlag != AccStoreUnitFlagCtrl::Off &&
          (!matchPattern(loop0SrcStride, m_ConstantInt(&loop0Value)) ||
           !loop0Value.isOne())) {
        return op->emitOpError(
            "unit_flag must be off when nz2dn loop0_src_stride is not 1");
      }
      break;
    }
    case AccStoreMode::Nz2nz:
      if (loop0SrcStride) {
        return op->emitOpError("nz2nz does not accept loop0_src_stride");
      }
      if (loop3Count) {
        return op->emitOpError("loop3 requires nz2nd or nz2dn");
      }
      if (!isa<FloatType>(destinationElementType) ||
          !cast<FloatType>(destinationElementType).isF32()) {
        return op->emitOpError("nz2nz requires destination element type to be f32");
      }
      break;
    }
  }

  if (static_cast<bool>(atomicType) != static_cast<bool>(atomicOp)) {
    return op->emitOpError("atomic requires type and op together");
  }
  if ((atomicType || atomicOp) && !allowAtomic) {
    return op->emitOpError("atomic is only supported for mte_l0c_gm");
  }

  return success();
}

static void printStructuredAccStoreClauses(
    OpAsmPrinter &printer, std::optional<AccStoreUnitFlagCtrl> unitFlag,
    Value preQuant,
    std::optional<AccStoreQuantPreMode> preQuantMode, Value preRelu,
    std::optional<ReluPreMode> preReluMode, Value clipValue,
    std::optional<AccStoreMode> mode, Value split, Value loop0SrcStride,
    Value loop3Count, Value loop3SrcStride, Value loop3DstStride,
    std::optional<AccStoreSatMode> satMode,
    std::optional<AccStoreAtomicType> atomicType,
    std::optional<AccStoreAtomicOp> atomicOp) {
  if (unitFlag && *unitFlag != AccStoreUnitFlagCtrl::Off) {
    printer << ", unit_flag("
            << (*unitFlag == AccStoreUnitFlagCtrl::CheckOnly ? "check_only"
                                                             : "check_and_clear")
            << ")";
  }
  if (preQuantMode) {
    printer << ", pre_quant(" << preQuant << ", mode = "
            << stringifyAccStoreQuantPreMode(*preQuantMode) << ")";
  }
  if (preReluMode) {
    printer << ", pre_relu(";
    if (preRelu) {
      printer << preRelu << ", ";
    }
    printer << "mode = " << stringifyReluPreMode(*preReluMode);
    if (clipValue) {
      printer << ", clip = " << clipValue;
    }
    printer << ")";
  }
  if (mode) {
    switch (*mode) {
    case AccStoreMode::Nz2nd:
      printer << ", nz2nd";
      break;
    case AccStoreMode::Nz2dn:
      printer << ", nz2dn";
      if (loop0SrcStride) {
        printer << "(" << loop0SrcStride << ")";
      }
      break;
    case AccStoreMode::Nz2nz:
      printer << ", nz2nz";
      if (split) {
        printer << "(" << split << ")";
      }
      break;
    }
  }
  if (loop3Count) {
    printer << ", loop3(" << loop3Count << ", " << loop3SrcStride << ", "
            << loop3DstStride << ")";
  }
  if (satMode) {
    switch (*satMode) {
    case AccStoreSatMode::Sat:
      printer << ", sat";
      break;
    case AccStoreSatMode::NoSat:
      printer << ", nosat";
      break;
    case AccStoreSatMode::SatPreserveNan:
      printer << ", sat(preserve_nan)";
      break;
    }
  }
  if (atomicType && atomicOp) {
    printer << ", atomic(type = " << stringifyAccStoreAtomicType(*atomicType)
            << ", op = " << stringifyAccStoreAtomicOp(*atomicOp) << ")";
  }
}

static void printStructuredAccStoreOptionalTypes(
    OpAsmPrinter &printer, Value preQuant, Value preRelu, Value clipValue,
    Value split, Value loop0SrcStride, Value loop3Count, Value loop3SrcStride,
    Value loop3DstStride) {
  if (preQuant) {
    printer << ", " << preQuant.getType();
  }
  if (preRelu) {
    printer << ", " << preRelu.getType();
  }
  if (clipValue) {
    printer << ", " << clipValue.getType();
  }
  if (split) {
    printer << ", " << split.getType();
  }
  if (loop0SrcStride) {
    printer << ", " << loop0SrcStride.getType();
  }
  if (loop3Count) {
    printer << ", " << loop3Count.getType() << ", " << loop3SrcStride.getType()
            << ", " << loop3DstStride.getType();
  }
}

static ParseResult parseStructuredAccStoreTailTypes(
    OpAsmParser &parser, StructuredAccStoreAsmState &state) {
  if (!state.preQuantOperands.empty() &&
      (parser.parseComma() ||
       parseStructuredOptionalType(parser, state.preQuantTypes))) {
    return failure();
  }
  if (!state.preReluOperands.empty() &&
      (parser.parseComma() ||
       parseStructuredOptionalType(parser, state.preReluTypes))) {
    return failure();
  }
  if (!state.clipValueOperands.empty() &&
      (parser.parseComma() ||
       parseStructuredOptionalType(parser, state.clipValueTypes))) {
    return failure();
  }
  if (!state.splitOperands.empty() &&
      (parser.parseComma() ||
       parseStructuredOptionalType(parser, state.splitTypes))) {
    return failure();
  }
  if (!state.loop0SrcStrideOperands.empty() &&
      (parser.parseComma() ||
       parseStructuredOptionalType(parser, state.loop0SrcStrideTypes))) {
    return failure();
  }
  if (!state.loop3CountOperands.empty() &&
      (parser.parseComma() ||
       parseStructuredOptionalType(parser, state.loop3CountTypes) ||
       parser.parseComma() ||
       parseStructuredOptionalType(parser, state.loop3SrcStrideTypes) ||
       parser.parseComma() ||
       parseStructuredOptionalType(parser, state.loop3DstStrideTypes))) {
    return failure();
  }
  return success();
}

template <typename OpTy>
static void setStructuredAccStoreSegmentSizes(OperationState &result,
                                              ArrayRef<int32_t> segmentSizes) {
  auto &segments = result.getOrAddProperties<typename OpTy::Properties>()
                       .operandSegmentSizes;
  llvm::copy(segmentSizes, segments.begin());
}

template <typename OpTy>
static void addStructuredAccStoreAttrs(OperationState &result,
                                       Builder &builder,
                                       const StructuredAccStoreAsmState &state) {
  if (state.mode) {
    result.addAttribute("mode", AccStoreModeAttr::get(builder.getContext(),
                                                      *state.mode));
  }
  if (state.unitFlag) {
    result.addAttribute("unit_flag",
                        AccStoreUnitFlagCtrlAttr::get(builder.getContext(),
                                                      *state.unitFlag));
  }
  if (state.preQuantMode) {
    result.addAttribute("pre_quant_mode",
                        AccStoreQuantPreModeAttr::get(builder.getContext(),
                                                      *state.preQuantMode));
  }
  if (state.preReluMode) {
    result.addAttribute("pre_relu_mode",
                        ReluPreModeAttr::get(builder.getContext(),
                                             *state.preReluMode));
  }
  if (state.atomicType) {
    result.addAttribute("atomic_type",
                        AccStoreAtomicTypeAttr::get(builder.getContext(),
                                                    *state.atomicType));
  }
  if (state.atomicOp) {
    result.addAttribute("atomic_op",
                        AccStoreAtomicOpAttr::get(builder.getContext(),
                                                  *state.atomicOp));
  }
  if (state.satMode) {
    result.addAttribute("sat_mode",
                        AccStoreSatModeAttr::get(builder.getContext(),
                                                 *state.satMode));
  }
}

[[maybe_unused]] static ParseResult resolveStructuredMteL0cL1OptionalOperands(
    OpAsmParser &parser, StructuredAccStoreAsmState &state,
    SmallVectorImpl<Value> &resolvedOperands, OperationState &result) {
  auto location = parser.getCurrentLocation();
  if (parser.resolveOperands(state.preQuantOperands, state.preQuantTypes,
                             location, result.operands) ||
      parser.resolveOperands(state.preReluOperands, state.preReluTypes,
                             location, result.operands) ||
      parser.resolveOperands(state.clipValueOperands, state.clipValueTypes,
                             location, result.operands) ||
      parser.resolveOperands(state.splitOperands, state.splitTypes, location,
                             result.operands) ||
      parser.resolveOperands(state.loop0SrcStrideOperands,
                             state.loop0SrcStrideTypes, location,
                             result.operands) ||
      parser.resolveOperands(state.loop3CountOperands, state.loop3CountTypes,
                             location, result.operands) ||
      parser.resolveOperands(state.loop3SrcStrideOperands,
                             state.loop3SrcStrideTypes, location,
                             result.operands) ||
      parser.resolveOperands(state.loop3DstStrideOperands,
                             state.loop3DstStrideTypes, location,
                             result.operands)) {
    return failure();
  }

  auto extractResolved = [&](SmallVectorImpl<OpAsmParser::UnresolvedOperand> &ops,
                             SmallVectorImpl<Type> &types) -> Value {
    if (ops.empty()) {
      return {};
    }
    return result.operands[resolvedOperands.size()];
  };
  resolvedOperands.push_back(extractResolved(state.preQuantOperands,
                                             state.preQuantTypes));
  resolvedOperands.push_back(extractResolved(state.preReluOperands,
                                             state.preReluTypes));
  resolvedOperands.push_back(extractResolved(state.clipValueOperands,
                                             state.clipValueTypes));
  resolvedOperands.push_back(extractResolved(state.splitOperands,
                                             state.splitTypes));
  resolvedOperands.push_back(extractResolved(state.loop0SrcStrideOperands,
                                             state.loop0SrcStrideTypes));
  resolvedOperands.push_back(extractResolved(state.loop3CountOperands,
                                             state.loop3CountTypes));
  resolvedOperands.push_back(extractResolved(state.loop3SrcStrideOperands,
                                             state.loop3SrcStrideTypes));
  resolvedOperands.push_back(extractResolved(state.loop3DstStrideOperands,
                                             state.loop3DstStrideTypes));
  return success();
}

template <typename CopyOp>
static LogicalResult verifyCopyGmToUbufOp(CopyOp op, bool expectSourceGM) {
  if (!isBufferLike(op.getSource().getType()) ||
      !isBufferLike(op.getDestination().getType())) {
    return op.emitOpError(
        "requires typed !pto.ptr or memref source and destination");
  }

  MemoryRole sourceRole = classifyMemoryRole(op.getSource().getType());
  MemoryRole destinationRole = classifyMemoryRole(op.getDestination().getType());
  bool directionMatches = true;
  if (expectSourceGM) {
    directionMatches &= sourceRole != MemoryRole::UB;
    directionMatches &= destinationRole != MemoryRole::GM;
  } else {
    directionMatches &= sourceRole != MemoryRole::GM;
    directionMatches &= destinationRole != MemoryRole::UB;
  }

  if (!directionMatches) {
    return op.emitOpError()
           << "requires "
           << (expectSourceGM ? "GM source and UB destination"
                              : "UB source and GM destination");
  }

  int64_t sourceElemBytes = getBufferElementByteSize(op.getSource().getType());
  int64_t destinationElemBytes =
      getBufferElementByteSize(op.getDestination().getType());
  if (sourceElemBytes <= 0 || destinationElemBytes <= 0) {
    return op.emitOpError("requires copy source and destination element types with known byte width");
  }
  if (sourceElemBytes != destinationElemBytes) {
    return op.emitOpError("requires source and destination element byte widths to match");
  }

  return success();
}

template <typename DmaOp>
static LogicalResult verifyOptionalDmaLoopGroup(DmaOp op, Value count,
                                                Value srcStride,
                                                Value dstStride,
                                                StringRef name) {
  bool hasAny = static_cast<bool>(count) || static_cast<bool>(srcStride) ||
                static_cast<bool>(dstStride);
  bool hasAll = static_cast<bool>(count) && static_cast<bool>(srcStride) &&
                static_cast<bool>(dstStride);
  if (hasAny && !hasAll) {
    return op.emitOpError() << "requires " << name
                            << " group to provide count, src stride, and dst stride together";
  }
  return success();
}

static LogicalResult verifyDmaLoadStoreLoopGroups(Operation *op,
                                                  ValueRange loopCounts,
                                                  ValueRange loopSrcStrides,
                                                  ValueRange loopDstStrides) {
  if (loopCounts.size() != loopSrcStrides.size() ||
      loopCounts.size() != loopDstStrides.size()) {
    return op->emitOpError()
           << "requires each loop group to provide count, src stride, and dst stride together";
  }
  return success();
}

template <typename CopyOp>
static LogicalResult verifyCopyUbufToGmOp(CopyOp op, bool expectSourceGM) {
  if (!isBufferLike(op.getSource().getType()) ||
      !isBufferLike(op.getDestination().getType())) {
    return op.emitOpError(
        "requires typed !pto.ptr or memref source and destination");
  }

  MemoryRole sourceRole = classifyMemoryRole(op.getSource().getType());
  MemoryRole destinationRole = classifyMemoryRole(op.getDestination().getType());
  bool directionMatches = true;
  if (expectSourceGM) {
    directionMatches &= sourceRole != MemoryRole::UB;
    directionMatches &= destinationRole != MemoryRole::GM;
  } else {
    directionMatches &= sourceRole != MemoryRole::GM;
    directionMatches &= destinationRole != MemoryRole::UB;
  }

  if (!directionMatches) {
    return op.emitOpError()
           << "requires "
           << (expectSourceGM ? "GM source and UB destination"
                              : "UB source and GM destination");
  }

  int64_t sourceElemBytes = getBufferElementByteSize(op.getSource().getType());
  int64_t destinationElemBytes =
      getBufferElementByteSize(op.getDestination().getType());
  if (sourceElemBytes <= 0 || destinationElemBytes <= 0) {
    return op.emitOpError("requires copy source and destination element types with known byte width");
  }
  if (sourceElemBytes != destinationElemBytes) {
    return op.emitOpError("requires source and destination element byte widths to match");
  }

  return success();
}

template <typename CopyOp>
static LogicalResult verifyCopyCbufToUbufLikeOp(CopyOp op) {
  if (!isBufferLike(op.getSource().getType()) ||
      !isBufferLike(op.getDestination().getType())) {
    return op.emitOpError(
        "requires typed !pto.ptr or memref source and destination");
  }

  if (classifyMemoryRole(op.getSource().getType()) != MemoryRole::Other ||
      classifyMemoryRole(op.getDestination().getType()) != MemoryRole::UB) {
    return op.emitOpError("requires CBUF source and UB destination");
  }

  int64_t sourceElemBytes = getBufferElementByteSize(op.getSource().getType());
  int64_t destinationElemBytes =
      getBufferElementByteSize(op.getDestination().getType());
  if (sourceElemBytes <= 0 || destinationElemBytes <= 0) {
    return op.emitOpError("requires copy source and destination element types with known byte width");
  }
  if (sourceElemBytes != destinationElemBytes) {
    return op.emitOpError("requires source and destination element byte widths to match");
  }

  return success();
}

Type VRegType::parse(AsmParser &parser) {
  SmallVector<int64_t, 1> shape;
  Type elementType;
  SMLoc loc = parser.getCurrentLocation();

  if (failed(parser.parseLess()) ||
      failed(parser.parseDimensionList(shape, /*allowDynamic=*/false,
                                       /*withTrailingX=*/true)) ||
      shape.size() != 1 || failed(parser.parseType(elementType)) ||
      failed(parser.parseGreater())) {
    return {};
  }

  return parser.getChecked<VRegType>(loc, parser.getContext(), shape.front(),
                                    elementType);
}

void VRegType::print(AsmPrinter &printer) const {
  printer << "<" << getElementCount() << "x";
  printer.printType(getElementType());
  printer << ">";
}

LogicalResult VRegType::verify(function_ref<InFlightDiagnostic()> emitError,
                              int64_t elementCount, Type elementType) {
  if (elementCount <= 0) {
    return emitError() << "'" << formatVRegType(elementCount, elementType)
                       << "' expected a positive element count";
  }

  auto intOrFloat = mlir::dyn_cast<IntegerType>(elementType);
  unsigned elementBitWidth = 0;
  if (intOrFloat) {
    elementBitWidth = intOrFloat.getWidth();
  } else if (auto floatType = mlir::dyn_cast<FloatType>(elementType)) {
    elementBitWidth = floatType.getWidth();
  } else if (pto::isPTOLowPrecisionType(elementType)) {
    elementBitWidth = pto::getPTOStorageElemBitWidth(elementType);
  } else {
    return emitError() << "'" << formatVRegType(elementCount, elementType)
                       << "' expected an integer, floating-point, or PTO "
                          "low-precision element type";
  }

  if (elementCount * static_cast<int64_t>(elementBitWidth) != mlir::pto::kValue2048) {
    return emitError() << "'" << formatVRegType(elementCount, elementType)
                       << "' expected exactly 256 bytes";
  }

  return success();
}

static bool isForbiddenSynchronizationInsideVecScope(Operation *op) {
  // High-level synchronization is forbidden before and after lowering.
  if (isa<RecordEventOp, WaitEventOp, BarrierSyncOp>(op))
    return true;

  // Intra-core pipeline and buffer-id synchronization executes outside the
  // vector interval that it orders.
  if (isa<SetFlagOp, WaitFlagOp, SetFlagDynOp, WaitFlagDynOp, GetBufOp,
          GetBufDynOp, RlsBufOp, RlsBufDynOp, BarrierOp>(op))
    return true;

  // Intra-block, cross-core, system, cache, and SIMT synchronization likewise
  // delimit vector intervals. MemBarOp is intentionally not in this list.
  return isa<SyncSetOp, SyncWaitOp, CmoCacheInvalidOp, FenceBarrierAllOp,
             TSyncOp, SyncAllOp, DsbOp, DcciOp, SyncthreadsOp, ThreadfenceOp,
             ThreadfenceBlockOp>(op);
}

LogicalResult VecScopeOp::verify() {
  Region &bodyRegion = getBody();
  if (bodyRegion.empty()) {
    return emitOpError("expects a non-empty body region");
  }

  Block &body = bodyRegion.front();
  if (body.getNumArguments() != 0) {
    return emitOpError() << "expects body block to have no arguments, got "
                         << body.getNumArguments();
  }

  Operation *boundaryOp = nullptr;
  bodyRegion.walk([&](Operation *op) {
    if (!isForbiddenSynchronizationInsideVecScope(op))
      return WalkResult::advance();
    boundaryOp = op;
    return WalkResult::interrupt();
  });
  if (boundaryOp)
    return boundaryOp->emitOpError()
           << "must be outside 'pto.vecscope'; synchronization operations "
              "delimit vector scopes";

  return success();
}

LogicalResult StrictVecScopeOp::verify() {
  Region &bodyRegion = getBody();
  if (bodyRegion.empty()) {
    return emitOpError("expects a non-empty body region");
  }

  Block &body = bodyRegion.front();
  if (body.getNumArguments() != getCaptures().size()) {
    return emitOpError() << "expects body block to have "
                         << getCaptures().size()
                         << " arguments to match explicit captures, got "
                         << body.getNumArguments();
  }

  for (auto [idx, pair] :
       llvm::enumerate(llvm::zip(body.getArguments(), getCaptures()))) {
    BlockArgument blockArg = std::get<0>(pair);
    Value capture = std::get<1>(pair);
    if (blockArg.getType() != capture.getType()) {
      return emitOpError() << "expects body block argument #" << idx
                           << " to have type " << capture.getType()
                           << ", got " << blockArg.getType();
    }
  }

  Operation *boundaryOp = nullptr;
  bodyRegion.walk([&](Operation *op) {
    if (!isForbiddenSynchronizationInsideVecScope(op))
      return WalkResult::advance();
    boundaryOp = op;
    return WalkResult::interrupt();
  });
  if (boundaryOp)
    return boundaryOp->emitOpError()
           << "must be outside 'pto.strict_vecscope'; synchronization "
              "operations delimit vector scopes";
  return success();
}

bool MaskType::isSupportedGranularity(StringRef granularity) {
  return granularity == "b8" || granularity == "b16" ||
         granularity == "b32";
}

Type MaskType::parse(AsmParser &parser) {
  auto loc = parser.getCurrentLocation();
  StringRef granularity;
  if (failed(parser.parseLess()) || failed(parser.parseKeyword(&granularity)) ||
      failed(parser.parseGreater())) {
    return {};
  }

  return parser.getChecked<MaskType>(loc, parser.getContext(), granularity);
}

void MaskType::print(AsmPrinter &printer) const {
  printer << "<" << getGranularity() << ">";
}

LogicalResult
MaskType::verify(function_ref<InFlightDiagnostic()> emitError,
                 StringRef granularity) {
  if (!isSupportedGranularity(granularity)) {
    return emitError() << "'" << formatMaskType(granularity)
                       << "' expected granularity to be one of b8, b16, b32";
  }
  return success();
}

void CopyGmToUbufOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult CopyGmToUbufOp::verify() {
  return verifyCopyGmToUbufOp(*this, true);
}

void MteGmUbOp::build(OpBuilder &builder, OperationState &state, Value source,
                      Value destination, Value l2CacheCtl, Value lenBurst,
                      pto::DmaLoopConfig nburst,
                      llvm::ArrayRef<pto::DmaLoopConfig> loops,
                      std::optional<pto::DmaPadConfig> pad) {
  state.addOperands({source, destination, l2CacheCtl, lenBurst, nburst.count,
                     nburst.srcStride, nburst.dstStride});
  for (const pto::DmaLoopConfig &loop : loops) {
    state.addOperands(loop.count);
  }
  for (const pto::DmaLoopConfig &loop : loops) {
    state.addOperands(loop.srcStride);
  }
  for (const pto::DmaLoopConfig &loop : loops) {
    state.addOperands(loop.dstStride);
  }
  bool hasPadCounts = pad && pad->leftCount && pad->rightCount;
  assert((!pad || static_cast<bool>(pad->leftCount) ==
                       static_cast<bool>(pad->rightCount)) &&
         "mte_gm_ub pad config must provide both left and right counts, or omit both");
  if (pad) {
    state.addOperands(pad->value);
    if (hasPadCounts) {
      state.addOperands({pad->leftCount, pad->rightCount});
    }
  }

  state.addAttribute(
      getOperandSegmentSizeAttr(),
      builder.getDenseI32ArrayAttr(
          {1, 1, 1, 1, 1, 1, 1,
           static_cast<int32_t>(loops.size()),
           static_cast<int32_t>(loops.size()),
           static_cast<int32_t>(loops.size()),
           pad ? 1 : 0, hasPadCounts ? 1 : 0, hasPadCounts ? 1 : 0}));
}

void MteGmUbOp::build(OpBuilder &builder, OperationState &state, Value source,
                      Value destination, Value l2CacheCtl, Value lenBurst,
                      pto::DmaLoopConfig nburst,
                      std::optional<pto::DmaLoopConfig> loop1,
                      std::optional<pto::DmaLoopConfig> loop2,
                      std::optional<pto::DmaPadConfig> pad) {
  SmallVector<pto::DmaLoopConfig> loops;
  if (loop1) {
    loops.push_back(*loop1);
  }
  if (loop2) {
    loops.push_back(*loop2);
  }
  build(builder, state, source, destination, l2CacheCtl, lenBurst, nburst,
        loops, pad);
}

ParseResult MteGmUbOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand source, destination, l2CacheCtl, lenBurst;
  SmallVector<OpAsmParser::UnresolvedOperand> nburstOperands;
  SmallVector<OpAsmParser::UnresolvedOperand> loopCountOperands;
  SmallVector<OpAsmParser::UnresolvedOperand> loopSrcStrideOperands;
  SmallVector<OpAsmParser::UnresolvedOperand> loopDstStrideOperands;
  SmallVector<OpAsmParser::UnresolvedOperand> padOperands;
  if (parseRequiredOperandWithComma(parser, source) ||
      parseRequiredOperandWithComma(parser, destination) ||
      parseRequiredOperandWithComma(parser, l2CacheCtl) ||
      parser.parseOperand(lenBurst) ||
      parseDmaTripleGroup(parser, "nburst", nburstOperands)) {
    return failure();
  }
  while (true) {
    if (succeeded(parser.parseOptionalKeyword("pad"))) {
      if (parser.parseLParen()) {
        return failure();
      }
      OpAsmParser::UnresolvedOperand value;
      if (parser.parseOperand(value)) {
        return failure();
      }
      padOperands.push_back(value);
      if (succeeded(parser.parseOptionalComma())) {
        OpAsmParser::UnresolvedOperand left;
        OpAsmParser::UnresolvedOperand right;
        if (parser.parseOperand(left) || parser.parseComma() ||
            parser.parseOperand(right)) {
          return failure();
        }
        padOperands.push_back(left);
        padOperands.push_back(right);
      }
      if (parser.parseRParen()) {
        return failure();
      }
      break;
    }

    StringRef parsedKeyword;
    SmallVector<OpAsmParser::UnresolvedOperand, mlir::pto::kValue3> loopGroupOperands;
    if (parseOptionalDmaTripleGroupAlias(parser, {"loop", "loop1", "loop2"},
                                         parsedKeyword, loopGroupOperands)) {
      return failure();
    }
    if (parsedKeyword.empty()) {
      break;
    }
    loopCountOperands.push_back(loopGroupOperands[0]);
    loopSrcStrideOperands.push_back(loopGroupOperands[1]);
    loopDstStrideOperands.push_back(loopGroupOperands[mlir::pto::kValue2]);
  }

  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon()) {
    return failure();
  }

  Type sourceType, destinationType, l2CacheCtlType, lenBurstType;
  SmallVector<Type> nburstTypes, loopCountTypes, loopSrcStrideTypes,
      loopDstStrideTypes, padTypes;
  if (parser.parseType(sourceType) || parser.parseComma() ||
      parser.parseType(destinationType) || parser.parseComma() ||
      parser.parseType(l2CacheCtlType) || parser.parseComma() ||
      parser.parseType(lenBurstType) || parser.parseComma() ||
      parseDmaTripleTypes(parser, nburstTypes)) {
    return failure();
  }
  while (succeeded(parser.parseOptionalComma())) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword)) {
      return failure();
    }
    if (isDmaLoopKeyword(keyword)) {
      SmallVector<Type> loopGroupTypes;
      if (parseDmaTripleTypes(parser, loopGroupTypes)) {
        return failure();
      }
      loopCountTypes.push_back(loopGroupTypes[0]);
      loopSrcStrideTypes.push_back(loopGroupTypes[1]);
      loopDstStrideTypes.push_back(loopGroupTypes[mlir::pto::kValue2]);
      continue;
    }
    if (keyword == "pad") {
      if (!padTypes.empty() || parseDmaPadTypes(parser, padTypes)) {
        return failure();
      }
      continue;
    }
    return parser.emitError(parser.getCurrentLocation(),
                            "expected one of 'loop' or 'pad'");
  }

  int32_t loopGroupCount = static_cast<int32_t>(loopCountOperands.size());
  if (loopCountOperands.size() != loopSrcStrideOperands.size() ||
      loopCountOperands.size() != loopDstStrideOperands.size() ||
      loopCountTypes.size() != loopSrcStrideTypes.size() ||
      loopCountTypes.size() != loopDstStrideTypes.size()) {
    return parser.emitError(parser.getCurrentLocation(),
                            "requires each loop group to provide count, src stride, and dst stride");
  }
  if (loopCountOperands.size() != loopCountTypes.size()) {
    return parser.emitError(parser.getCurrentLocation(),
                            "requires loop operand and type groups to match");
  }

  auto &segments =
      result.getOrAddProperties<MteGmUbOp::Properties>().operandSegmentSizes;
  llvm::copy(ArrayRef<int32_t>{1, 1, 1, 1, 1, 1, 1,
                               loopGroupCount, loopGroupCount, loopGroupCount,
                               static_cast<int32_t>(padOperands.size() ? 1 : 0),
                               static_cast<int32_t>(padOperands.size() == 3 ? 1 : 0),
                               static_cast<int32_t>(padOperands.size() == 3 ? 1 : 0)},
             segments.begin());

  if (parser.resolveOperand(source, sourceType, result.operands) ||
      parser.resolveOperand(destination, destinationType, result.operands) ||
      parser.resolveOperand(l2CacheCtl, l2CacheCtlType, result.operands) ||
      parser.resolveOperand(lenBurst, lenBurstType, result.operands) ||
      parser.resolveOperands(nburstOperands, nburstTypes, parser.getCurrentLocation(),
                             result.operands) ||
      parser.resolveOperands(loopCountOperands, loopCountTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(loopSrcStrideOperands, loopSrcStrideTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(loopDstStrideOperands, loopDstStrideTypes,
                             parser.getCurrentLocation(),
                             result.operands) ||
      parser.resolveOperands(padOperands, padTypes, parser.getCurrentLocation(),
                             result.operands)) {
    return failure();
  }
  return success();
}

void MteGmUbOp::print(OpAsmPrinter &printer) {
  printer << " " << getSource() << ", " << getDestination() << ", "
          << getL2CacheCtl() << ", " << getLenBurst();
  printDmaTripleGroup(printer, "nburst", getNBurst(), getNburstSrcStride(),
                      getNburstDstStride());
  for (auto [count, srcStride, dstStride] :
       llvm::zip(getLoopCounts(), getLoopSrcStrides(), getLoopDstStrides())) {
    printDmaTripleGroup(printer, "loop", count, srcStride, dstStride);
  }
  if (getPadValue()) {
    printDmaPadGroup(printer, getPadValue(), getLeftPaddingCount(),
                     getRightPaddingCount());
  }
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : " << getSource().getType() << ", " << getDestination().getType()
          << ", " << getL2CacheCtl().getType() << ", " << getLenBurst().getType()
          << ", " << getNBurst().getType() << ", " << getNburstSrcStride().getType()
          << ", "
          << getNburstDstStride().getType();
  for (auto [count, srcStride, dstStride] :
       llvm::zip(getLoopCounts(), getLoopSrcStrides(), getLoopDstStrides())) {
    printDmaTripleTypes(printer, "loop", count.getType(), srcStride.getType(),
                        dstStride.getType());
  }
  if (getPadValue()) {
    printDmaPadTypes(printer, getPadValue().getType(),
                     getLeftPaddingCount() ? getLeftPaddingCount().getType() : Type{},
                     getRightPaddingCount() ? getRightPaddingCount().getType() : Type{});
  }
}

void MteGmUbOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult MteGmUbOp::verify() {
  if (failed(verifyCopyGmToUbufOp(*this, true))) {
    return failure();
  }
  if (failed(verifyDmaLoadStoreLoopGroups(
          getOperation(), getLoopCounts(), getLoopSrcStrides(),
          getLoopDstStrides()))) {
    return failure();
  }
  if (!getPadValue() && (getLeftPaddingCount() || getRightPaddingCount())) {
    return emitOpError() << "requires pad group to provide a pad value";
  }
  if (getPadValue() && static_cast<bool>(getLeftPaddingCount()) !=
                           static_cast<bool>(getRightPaddingCount())) {
    return emitOpError()
           << "requires pad group to provide both left and right counts, or omit both";
  }
  if (Value padValue = getPadValue()) {
    Type valueType = padValue.getType();
    if (!isSupportedMovPadScalarType(valueType)) {
      return emitOpError()
             << "expects pad value to be i8/i16/i32 or f16/bf16/f32 scalar, but got "
             << valueType;
    }
  }
  return success();
}

LogicalResult SetMovPadValOp::verify() {
  Type valueType = getValue().getType();
  if (isSupportedMovPadScalarType(valueType)) {
    return success();
  }
  return emitOpError()
         << "expects i8/i16/i32 or f16/bf16/f32 scalar operand, but got "
         << valueType;
}
void MadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getLhsMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getRhsMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

static LogicalResult verifyMadPointerKinds(Operation *op, Type lhsTy, Type rhsTy,
                                           Type dstTy,
                                           std::optional<Type> biasTy =
                                               std::nullopt) {
  auto lhsType = dyn_cast<pto::PtrType>(lhsTy);
  auto rhsType = dyn_cast<pto::PtrType>(rhsTy);
  auto dstType = dyn_cast<pto::PtrType>(dstTy);
  if (!lhsType || !rhsType || !dstType) {
    return op->emitOpError("requires typed !pto.ptr lhs/rhs/dst operands");
  }

  const auto lhsAS = lhsType.getMemorySpace().getAddressSpace();
  const auto rhsAS = rhsType.getMemorySpace().getAddressSpace();
  const auto dstAS = dstType.getMemorySpace().getAddressSpace();

  const bool isStrongCube =
      lhsAS == pto::AddressSpace::LEFT && rhsAS == pto::AddressSpace::RIGHT &&
      dstAS == pto::AddressSpace::ACC;
  if (!isStrongCube) {
    return op->emitOpError("requires l0a/l0b/l0c-typed lhs/rhs/dst pointers");
  }

  if (!biasTy) {
    return success();
  }

  auto biasType = dyn_cast<pto::PtrType>(*biasTy);
  if (!biasType) {
    return op->emitOpError("requires typed !pto.ptr bias operand");
  }
  if (biasType.getMemorySpace().getAddressSpace() != pto::AddressSpace::BIAS) {
    return op->emitOpError("requires bias pointer in !pto.ptr<..., bt>");
  }
  if (biasType.getElementType() != dstType.getElementType()) {
    return op->emitOpError("requires bias element type to match dst element type");
  }
  return success();
}

void MadAccOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getLhsMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getRhsMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getDstMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}


void MadBiasOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getLhsMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getRhsMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getBiasMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

static LogicalResult verifyMadMxCommon(Operation *op, Type lhsTy, Type rhsTy,
                                       Type dstTy,
                                       std::optional<Type> biasTy =
                                           std::nullopt) {
  if (failed(verifyMadPointerKinds(op, lhsTy, rhsTy, dstTy, biasTy))) {
    return failure();
  }

  auto lhsType = cast<pto::PtrType>(lhsTy);
  auto rhsType = cast<pto::PtrType>(rhsTy);
  auto dstType = cast<pto::PtrType>(dstTy);
  const auto lhsAS = lhsType.getMemorySpace().getAddressSpace();
  const auto rhsAS = rhsType.getMemorySpace().getAddressSpace();
  const auto dstAS = dstType.getMemorySpace().getAddressSpace();
  const bool isStrongCube =
      lhsAS == pto::AddressSpace::LEFT && rhsAS == pto::AddressSpace::RIGHT &&
      dstAS == pto::AddressSpace::ACC;
  if (!isStrongCube) {
    return op->emitOpError("requires l0a/l0b/l0c-typed lhs/rhs/dst pointers");
  }

  if (!isMxElementType(lhsType.getElementType()) ||
      !isMxElementType(rhsType.getElementType())) {
    return op->emitOpError(
        "requires MX lhs/rhs element types (f8E4M3FN, f8E5M2, f4E1M2x2, or "
        "f4E2M1x2)");
  }
  return success();
}

void MadMxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getLhsMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getRhsMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}


void MadMxAccOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getLhsMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getRhsMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getDstMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}


void MadMxBiasOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getLhsMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getRhsMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getBiasMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

static std::optional<pto::MadUnitFlagMode>
parseMadUnitFlagModeToken(StringRef token) {
  if (token == "check_only") {
    return pto::MadUnitFlagMode::CheckOnly;
  }
  if (token == "check_and_set") {
    return pto::MadUnitFlagMode::CheckAndSet;
  }
  return std::nullopt;
}

static StringRef stringifyMadUnitFlagModeToken(pto::MadUnitFlagMode mode) {
  switch (mode) {
  case pto::MadUnitFlagMode::CheckOnly:
    return "check_only";
  case pto::MadUnitFlagMode::CheckAndSet:
    return "check_and_set";
  }
  llvm_unreachable("unexpected mad unit flag mode");
}

static std::optional<pto::Tf32Mode> parseTf32ModeToken(StringRef token) {
  if (token == "round_even") {
    return pto::Tf32Mode::RoundEven;
  }
  if (token == "round_away") {
    return pto::Tf32Mode::RoundAway;
  }
  return std::nullopt;
}

static StringRef stringifyTf32ModeToken(pto::Tf32Mode mode) {
  switch (mode) {
  case pto::Tf32Mode::RoundEven:
    return "round_even";
  case pto::Tf32Mode::RoundAway:
    return "round_away";
  }
  llvm_unreachable("unexpected tf32 mode");
}

static StringRef stringifyMadSatModeToken(pto::MadSatMode mode) {
  switch (mode) {
  case pto::MadSatMode::Sat:
    return "sat";
  case pto::MadSatMode::NoSat:
    return "nosat";
  }
  llvm_unreachable("unexpected mad sat mode");
}

static LogicalResult verifyMadSemanticClauses(Operation *op, Type lhsTy,
                                              Type rhsTy, Type dstTy,
                                              std::optional<Type> biasTy,
                                              std::optional<pto::Tf32Mode> tf32Mode,
                                              std::optional<pto::MadSatMode> satMode,
                                              bool hasNDir) {
  if (failed(verifyMadPointerKinds(op, lhsTy, rhsTy, dstTy, biasTy))) {
    return failure();
  }

  auto lhsType = dyn_cast<pto::PtrType>(lhsTy);
  auto rhsType = dyn_cast<pto::PtrType>(rhsTy);
  auto dstType = dyn_cast<pto::PtrType>(dstTy);
  if (!lhsType || !rhsType || !dstType) {
    return op->emitOpError("requires typed !pto.ptr lhs/rhs/dst operands");
  }

  if (tf32Mode) {
    if (!(lhsType.getElementType().isF32() && rhsType.getElementType().isF32() &&
          dstType.getElementType().isF32())) {
      return op->emitOpError(
          "requires tf32_mode only for f32 lhs/rhs/dst element types");
    }
  }
  if (pto::isPTOHiFloat8Type(lhsType.getElementType()) !=
      pto::isPTOHiFloat8Type(rhsType.getElementType())) {
    return op->emitOpError(
        "requires lhs/rhs to both use hif8 or both use non-hif8 element types");
  }
  if (satMode) {
    auto isFloatLike = [](Type type) {
      if (isa<FloatType>(type)) {
        return true;
      }
      return pto::isPTOLowPrecisionType(type);
    };
    if (!(isFloatLike(lhsType.getElementType()) &&
          isFloatLike(rhsType.getElementType()) &&
          isFloatLike(dstType.getElementType()))) {
      return op->emitOpError(
          "requires sat/nosat only for floating lhs/rhs/dst element types");
    }
  }
  (void)hasNDir;
  return success();
}

template <typename OpT>
static ParseResult parseMadSemanticOpCommon(OpAsmParser &parser,
                                            OperationState &result,
                                            bool hasBias,
                                            bool parseTf32ModeClause) {
  OpAsmParser::UnresolvedOperand lhs, rhs, dst, bias;
  OpAsmParser::UnresolvedOperand m, n, k;
  StringRef unitFlagKeyword;
  StringRef tf32Keyword;
  NamedAttrList attrs;

  if (parseRequiredOperandWithComma(parser, lhs) ||
      parseRequiredOperandWithComma(parser, rhs) ||
      parseRequiredOperandWithComma(parser, dst) ||
      (hasBias && parseRequiredOperandWithComma(parser, bias)) ||
      parseRequiredOperandWithComma(parser, m) ||
      parseRequiredOperandWithComma(parser, n) ||
      parser.parseOperand(k)) {
    return failure();
  }

  auto parseUnitFlagClause = [&]() -> ParseResult {
    if (failed(parser.parseOptionalKeyword("unit_flag"))) {
      return success();
    }
    if (parser.parseLParen() || parser.parseKeyword(&unitFlagKeyword) ||
        parser.parseRParen()) {
      return failure();
    }
    auto mode = parseMadUnitFlagModeToken(unitFlagKeyword);
    if (!mode) {
      return parser.emitError(parser.getCurrentLocation())
             << "expected unit_flag(check_only|check_and_set)";
    }
    attrs.set("unit_flag_mode",
              pto::MadUnitFlagModeAttr::get(parser.getContext(), *mode));
    return success();
  };
  auto parseDisableGemvClause = [&]() -> ParseResult {
    if (succeeded(parser.parseOptionalKeyword("disable_gemv"))) {
      attrs.set("disable_gemv", UnitAttr::get(parser.getContext()));
    }
    return success();
  };
  auto parseSatClause = [&]() -> ParseResult {
    if (succeeded(parser.parseOptionalKeyword("sat"))) {
      attrs.set("sat_mode",
                pto::MadSatModeAttr::get(parser.getContext(),
                                         pto::MadSatMode::Sat));
      return success();
    }
    if (succeeded(parser.parseOptionalKeyword("nosat"))) {
      attrs.set("sat_mode",
                pto::MadSatModeAttr::get(parser.getContext(),
                                         pto::MadSatMode::NoSat));
    }
    return success();
  };
  auto parseTf32Clause = [&]() -> ParseResult {
    if (!parseTf32ModeClause) {
      return success();
    }
    if (failed(parser.parseOptionalKeyword("tf32_mode"))) {
      return success();
    }
    if (parser.parseLParen() || parser.parseKeyword(&tf32Keyword) ||
        parser.parseRParen()) {
      return failure();
    }
    auto mode = parseTf32ModeToken(tf32Keyword);
    if (!mode) {
      return parser.emitError(parser.getCurrentLocation())
             << "expected tf32_mode(round_even|round_away)";
    }
    attrs.set("tf32_mode", pto::Tf32ModeAttr::get(parser.getContext(), *mode));
    return success();
  };
  auto parseNDirClause = [&]() -> ParseResult {
    if (succeeded(parser.parseOptionalKeyword("n_dir"))) {
      attrs.set("n_dir", UnitAttr::get(parser.getContext()));
    }
    return success();
  };
  if (failed(parseUnitFlagClause()) || failed(parseDisableGemvClause()) ||
      failed(parseSatClause()) || failed(parseTf32Clause()) ||
      failed(parseNDirClause())) {
    return failure();
  }

  if (parser.parseOptionalAttrDict(attrs) || parser.parseColon()) {
    return failure();
  }

  Type lhsType, rhsType, dstType, mType, nType, kType, biasType;
  if (parser.parseType(lhsType) || parser.parseComma() ||
      parser.parseType(rhsType) || parser.parseComma() ||
      parser.parseType(dstType) || parser.parseComma()) {
    return failure();
  }
  if (hasBias) {
    if (parser.parseType(biasType) || parser.parseComma()) {
      return failure();
    }
  }
  if (parser.parseType(mType) || parser.parseComma() || parser.parseType(nType) ||
      parser.parseComma() || parser.parseType(kType)) {
    return failure();
  }

  result.addAttributes(attrs);
  if (hasBias) {
    if (parser.resolveOperand(lhs, lhsType, result.operands) ||
        parser.resolveOperand(rhs, rhsType, result.operands) ||
        parser.resolveOperand(dst, dstType, result.operands) ||
        parser.resolveOperand(bias, biasType, result.operands) ||
        parser.resolveOperand(m, mType, result.operands) ||
        parser.resolveOperand(n, nType, result.operands) ||
        parser.resolveOperand(k, kType, result.operands)) {
      return failure();
    }
  } else {
    if (parser.resolveOperand(lhs, lhsType, result.operands) ||
        parser.resolveOperand(rhs, rhsType, result.operands) ||
        parser.resolveOperand(dst, dstType, result.operands) ||
        parser.resolveOperand(m, mType, result.operands) ||
        parser.resolveOperand(n, nType, result.operands) ||
        parser.resolveOperand(k, kType, result.operands)) {
      return failure();
    }
  }
  return success();
}

static void printMadSemanticClauses(OpAsmPrinter &printer, Operation *op,
                                    bool allowTf32Mode) {
  if (auto unitFlagMode = op->getAttrOfType<pto::MadUnitFlagModeAttr>(
          "unit_flag_mode")) {
    printer << " unit_flag("
            << stringifyMadUnitFlagModeToken(unitFlagMode.getValue()) << ")";
  }
  if (op->hasAttr("disable_gemv")) {
    printer << " disable_gemv";
  }
  if (auto satMode = op->getAttrOfType<pto::MadSatModeAttr>("sat_mode")) {
    printer << ' ' << stringifyMadSatModeToken(satMode.getValue());
  }
  if (allowTf32Mode) {
    if (auto tf32Mode = op->getAttrOfType<pto::Tf32ModeAttr>("tf32_mode")) {
      printer << " tf32_mode(" << stringifyTf32ModeToken(tf32Mode.getValue())
              << ")";
    }
  }
  if (op->hasAttr("n_dir")) {
    printer << " n_dir";
  }
}

static ArrayRef<StringRef> getMadSemanticElidedAttrs(bool allowTf32Mode) {
  static constexpr StringRef kWithTf32[] = {"unit_flag_mode", "disable_gemv",
                                            "sat_mode", "tf32_mode", "n_dir"};
  static constexpr StringRef kWithoutTf32[] = {"unit_flag_mode",
                                               "disable_gemv", "sat_mode",
                                               "n_dir"};
  return allowTf32Mode ? ArrayRef<StringRef>(kWithTf32)
                       : ArrayRef<StringRef>(kWithoutTf32);
}

template <typename OpT>
static void printMadSemanticOpNoBias(OpAsmPrinter &printer, OpT op,
                                     bool allowTf32Mode) {
  printer << ' ' << op.getLhs() << ", " << op.getRhs() << ", " << op.getDst()
          << ", " << op.getM() << ", " << op.getN() << ", " << op.getK();
  printMadSemanticClauses(printer, op, allowTf32Mode);
  printer.printOptionalAttrDict(op->getAttrs(),
                                getMadSemanticElidedAttrs(allowTf32Mode));
  printer << " : " << op.getLhs().getType() << ", " << op.getRhs().getType()
          << ", " << op.getDst().getType() << ", " << op.getM().getType()
          << ", " << op.getN().getType() << ", " << op.getK().getType();
}

template <typename OpT>
static void printMadSemanticOpWithBias(OpAsmPrinter &printer, OpT op,
                                       bool allowTf32Mode) {
  printer << ' ' << op.getLhs() << ", " << op.getRhs() << ", " << op.getDst()
          << ", " << op.getBias() << ", " << op.getM() << ", " << op.getN()
          << ", " << op.getK();
  printMadSemanticClauses(printer, op, allowTf32Mode);
  printer.printOptionalAttrDict(op->getAttrs(),
                                getMadSemanticElidedAttrs(allowTf32Mode));
  printer << " : " << op.getLhs().getType() << ", " << op.getRhs().getType()
          << ", " << op.getDst().getType() << ", " << op.getBias().getType()
          << ", " << op.getM().getType() << ", " << op.getN().getType()
          << ", " << op.getK().getType();
}

LogicalResult MadOp::verify() {
  std::optional<pto::Tf32Mode> tf32Mode;
  if (auto tf32ModeAttr =
          (*this)->getAttrOfType<pto::Tf32ModeAttr>("tf32_mode")) {
    tf32Mode = tf32ModeAttr.getValue();
  }
  return verifyMadSemanticClauses(*this, getLhs().getType(), getRhs().getType(),
                                  getDst().getType(), std::nullopt, tf32Mode,
                                  getSatMode(),
                                  (*this)->hasAttr("n_dir"));
}

ParseResult MadOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseMadSemanticOpCommon<MadOp>(parser, result, /*hasBias=*/false,
                                         /*parseTf32ModeClause=*/true);
}

void MadOp::print(OpAsmPrinter &printer) {
  printMadSemanticOpNoBias(printer, *this, /*allowTf32Mode=*/true);
}

bool MadOp::isMadMxFamily() { return false; }
bool MadOp::hasBiasOperand() { return false; }
bool MadOp::readsAccumulator() { return false; }
bool MadOp::supportsTf32Mode() { return true; }
Value MadOp::getBiasOrNull() { return {}; }

LogicalResult MadAccOp::verify() {
  std::optional<pto::Tf32Mode> tf32Mode;
  if (auto tf32ModeAttr =
          (*this)->getAttrOfType<pto::Tf32ModeAttr>("tf32_mode")) {
    tf32Mode = tf32ModeAttr.getValue();
  }
  return verifyMadSemanticClauses(*this, getLhs().getType(), getRhs().getType(),
                                  getDst().getType(), std::nullopt, tf32Mode,
                                  getSatMode(),
                                  (*this)->hasAttr("n_dir"));
}

ParseResult MadAccOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseMadSemanticOpCommon<MadAccOp>(parser, result, /*hasBias=*/false,
                                            /*parseTf32ModeClause=*/true);
}

void MadAccOp::print(OpAsmPrinter &printer) {
  printMadSemanticOpNoBias(printer, *this, /*allowTf32Mode=*/true);
}

bool MadAccOp::isMadMxFamily() { return false; }
bool MadAccOp::hasBiasOperand() { return false; }
bool MadAccOp::readsAccumulator() { return true; }
bool MadAccOp::supportsTf32Mode() { return true; }
Value MadAccOp::getBiasOrNull() { return {}; }

LogicalResult MadBiasOp::verify() {
  std::optional<pto::Tf32Mode> tf32Mode;
  if (auto tf32ModeAttr =
          (*this)->getAttrOfType<pto::Tf32ModeAttr>("tf32_mode")) {
    tf32Mode = tf32ModeAttr.getValue();
  }
  return verifyMadSemanticClauses(*this, getLhs().getType(), getRhs().getType(),
                                  getDst().getType(), getBias().getType(),
                                  tf32Mode, getSatMode(),
                                  (*this)->hasAttr("n_dir"));
}

ParseResult MadBiasOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseMadSemanticOpCommon<MadBiasOp>(parser, result, /*hasBias=*/true,
                                             /*parseTf32ModeClause=*/true);
}

void MadBiasOp::print(OpAsmPrinter &printer) {
  printMadSemanticOpWithBias(printer, *this, /*allowTf32Mode=*/true);
}

bool MadBiasOp::isMadMxFamily() { return false; }
bool MadBiasOp::hasBiasOperand() { return true; }
bool MadBiasOp::readsAccumulator() { return false; }
bool MadBiasOp::supportsTf32Mode() { return true; }
Value MadBiasOp::getBiasOrNull() { return getBias(); }

LogicalResult MadMxOp::verify() {
  if (failed(verifyMadMxCommon(*this, getLhs().getType(), getRhs().getType(),
                               getDst().getType()))) {
    return failure();
  }
  return verifyMadSemanticClauses(*this, getLhs().getType(), getRhs().getType(),
                                  getDst().getType(), std::nullopt, std::nullopt,
                                  getSatMode(),
                                  (*this)->hasAttr("n_dir"));
}

ParseResult MadMxOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseMadSemanticOpCommon<MadMxOp>(parser, result, /*hasBias=*/false,
                                           /*parseTf32ModeClause=*/false);
}

void MadMxOp::print(OpAsmPrinter &printer) {
  printMadSemanticOpNoBias(printer, *this, /*allowTf32Mode=*/false);
}

bool MadMxOp::isMadMxFamily() { return true; }
bool MadMxOp::hasBiasOperand() { return false; }
bool MadMxOp::readsAccumulator() { return false; }
bool MadMxOp::supportsTf32Mode() { return false; }
Value MadMxOp::getBiasOrNull() { return {}; }
Attribute MadMxOp::getTf32ModeAttr() { return {}; }

LogicalResult MadMxAccOp::verify() {
  if (failed(verifyMadMxCommon(*this, getLhs().getType(), getRhs().getType(),
                               getDst().getType()))) {
    return failure();
  }
  return verifyMadSemanticClauses(*this, getLhs().getType(), getRhs().getType(),
                                  getDst().getType(), std::nullopt, std::nullopt,
                                  getSatMode(),
                                  (*this)->hasAttr("n_dir"));
}

ParseResult MadMxAccOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseMadSemanticOpCommon<MadMxAccOp>(parser, result, /*hasBias=*/false,
                                              /*parseTf32ModeClause=*/false);
}

void MadMxAccOp::print(OpAsmPrinter &printer) {
  printMadSemanticOpNoBias(printer, *this, /*allowTf32Mode=*/false);
}

bool MadMxAccOp::isMadMxFamily() { return true; }
bool MadMxAccOp::hasBiasOperand() { return false; }
bool MadMxAccOp::readsAccumulator() { return true; }
bool MadMxAccOp::supportsTf32Mode() { return false; }
Value MadMxAccOp::getBiasOrNull() { return {}; }
Attribute MadMxAccOp::getTf32ModeAttr() { return {}; }

LogicalResult MadMxBiasOp::verify() {
  if (failed(verifyMadMxCommon(*this, getLhs().getType(), getRhs().getType(),
                               getDst().getType(), getBias().getType()))) {
    return failure();
  }
  return verifyMadSemanticClauses(*this, getLhs().getType(), getRhs().getType(),
                                  getDst().getType(), getBias().getType(),
                                  std::nullopt, getSatMode(),
                                  (*this)->hasAttr("n_dir"));
}

ParseResult MadMxBiasOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseMadSemanticOpCommon<MadMxBiasOp>(parser, result, /*hasBias=*/true,
                                               /*parseTf32ModeClause=*/false);
}

void MadMxBiasOp::print(OpAsmPrinter &printer) {
  printMadSemanticOpWithBias(printer, *this, /*allowTf32Mode=*/false);
}

bool MadMxBiasOp::isMadMxFamily() { return true; }
bool MadMxBiasOp::hasBiasOperand() { return true; }
bool MadMxBiasOp::readsAccumulator() { return false; }
bool MadMxBiasOp::supportsTf32Mode() { return false; }
Value MadMxBiasOp::getBiasOrNull() { return getBias(); }
Attribute MadMxBiasOp::getTf32ModeAttr() { return {}; }

void MadRawOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getLhsMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getRhsMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult MadRawOp::verify() {
  return verifyMadPointerKinds(*this, getLhs().getType(), getRhs().getType(),
                               getDst().getType());
}

bool MadRawOp::isMadMxFamily() { return false; }
bool MadRawOp::hasBiasOperand() { return false; }
Value MadRawOp::getBiasOrNull() { return {}; }

void MadBiasRawOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getLhsMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getRhsMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getBiasMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult MadBiasRawOp::verify() {
  return verifyMadPointerKinds(*this, getLhs().getType(), getRhs().getType(),
                               getDst().getType(), getBias().getType());
}

bool MadBiasRawOp::isMadMxFamily() { return false; }
bool MadBiasRawOp::hasBiasOperand() { return true; }
Value MadBiasRawOp::getBiasOrNull() { return getBias(); }

void MadMxRawOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getLhsMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getRhsMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult MadMxRawOp::verify() {
  return verifyMadMxCommon(*this, getLhs().getType(), getRhs().getType(),
                           getDst().getType());
}

bool MadMxRawOp::isMadMxFamily() { return true; }
bool MadMxRawOp::hasBiasOperand() { return false; }
Value MadMxRawOp::getBiasOrNull() { return {}; }

void MadMxBiasRawOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getLhsMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getRhsMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getBiasMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult MadMxBiasRawOp::verify() {
  return verifyMadMxCommon(*this, getLhs().getType(), getRhs().getType(),
                           getDst().getType(), getBias().getType());
}

bool MadMxBiasRawOp::isMadMxFamily() { return true; }
bool MadMxBiasRawOp::hasBiasOperand() { return true; }
Value MadMxBiasRawOp::getBiasOrNull() { return getBias(); }

static bool isCompatibleScalarForSemanticType(Type semanticType,
                                              Type scalarType) {
  if (semanticType == scalarType) {
    return true;
  }

  auto semanticInt = dyn_cast<IntegerType>(semanticType);
  auto scalarInt = dyn_cast<IntegerType>(scalarType);
  if (!semanticInt || !scalarInt || semanticInt.getWidth() != scalarInt.getWidth()) {
    return false;
  }

  if (semanticInt.isSigned()) {
    return scalarInt.isSigned() || scalarInt.isSignless();
  }
  if (semanticInt.isUnsigned()) {
    return scalarInt.isUnsigned() || scalarInt.isSignless();
  }
  return scalarInt.isSignless();
}

LogicalResult VbrOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getResult().getType(), "result"))) {
    return failure();
  }

  auto resultVecType = cast<VRegType>(getResult().getType());
  Type elementType = getValue().getType();
  if (isa<ShapedType, VectorType>(elementType)) {
    return emitOpError("value must be a scalar matching the result element type");
  }
  Type resultElementType = resultVecType.getElementType();
  if (!isCompatibleScalarForSemanticType(resultElementType, elementType)) {
    return emitOpError("value type must match result element type");
  }
  return success();
}

template <typename ReductionOp>
static LogicalResult verifyWideningReductionVecOp(ReductionOp op,
                                                  StringRef opName) {
  if (failed(verifyVRegTypeLike(op, op.getInput().getType(), "input")) ||
      failed(verifyVRegTypeLike(op, op.getResult().getType(), "result"))) {
    return failure();
  }

  auto inputType = dyn_cast<VRegType>(op.getInput().getType());
  auto resultType = dyn_cast<VRegType>(op.getResult().getType());
  if (!inputType || !resultType) {
    return failure();
  }

  Type inputElemType = inputType.getElementType();
  Type expectedResultElemType = inputElemType;
  int64_t expectedResultLanes = inputType.getElementCount();
  if (auto inputInt = dyn_cast<IntegerType>(inputElemType)) {
    if (inputInt.getWidth() < mlir::pto::kValue8 || inputInt.getWidth() > 32) {
      return op.emitOpError(
          "requires 8-bit, 16-bit, or 32-bit integer vector element type");
    }
    if (inputInt.getWidth() == mlir::pto::kValue8) {
      expectedResultElemType =
          IntegerType::get(op.getContext(), mlir::pto::kValue16, inputInt.getSignedness());
      expectedResultLanes = inputType.getElementCount() / mlir::pto::kValue2;
    }
    if (inputInt.getWidth() == mlir::pto::kValue16) {
      expectedResultElemType =
          IntegerType::get(op.getContext(), mlir::pto::kValue32, inputInt.getSignedness());
      expectedResultLanes = inputType.getElementCount() / mlir::pto::kValue2;
    }
  } else if (!inputElemType.isF16() && !inputElemType.isF32()) {
    return op.emitOpError("requires i16/i32/f16/f32 vector element type");
  }

  if (resultType.getElementCount() == expectedResultLanes &&
      resultType.getElementType() == expectedResultElemType) {
    return success();
  }

  return op.emitOpError() << opName << " expects result type !pto.vreg<"
                          << expectedResultLanes << "x"
                          << expectedResultElemType
                          << " for input element type " << inputElemType;
}

LogicalResult VcaddOp::verify() {
  return verifyWideningReductionVecOp(*this, "vcadd");
}

LogicalResult VcmaxOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getInput().getType(), "input")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result"))) {
    return failure();
  }
  if (getInput().getType() != getResult().getType()) {
    return emitOpError("input and result must have the same vector type");
  }
  return success();
}

LogicalResult VcminOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getInput().getType(), "input")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result"))) {
    return failure();
  }
  if (getInput().getType() != getResult().getType()) {
    return emitOpError("input and result must have the same vector type");
  }
  return success();
}

LogicalResult VciOp::verify() {
  auto resultType = dyn_cast<VRegType>(getResult().getType());
  if (!resultType) {
    return emitOpError("result must be !pto.vreg<...>");
  }
  Type resultElemType = resultType.getElementType();
  bool supportedInteger = false;
  if (auto intType = dyn_cast<IntegerType>(resultElemType)) {
    supportedInteger = intType.getWidth() == mlir::pto::kValue8 || intType.getWidth() == 16 ||
                       intType.getWidth() == mlir::pto::kValue32;
  }
  bool supportedFloat = resultElemType.isF16() || resultElemType.isF32();
  if (!supportedInteger && !supportedFloat) {
    return emitOpError("result element type must be integer or f16/f32");
  }
  if (!isCompatibleScalarForSemanticType(resultElemType, getIndex().getType())) {
    return emitOpError("index type must match result element type");
  }
  return success();
}

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

LogicalResult Vgather2Op::verify() {
  if (!isBufferLike(getSource().getType())) {
    return emitOpError("requires a pointer-like source");
  }
  MemoryRole sourceRole = classifyMemoryRole(getSource().getType());
  if (sourceRole == MemoryRole::GM) {
    return emitOpError("requires a UB-backed source");
  }

  auto offsetsType = dyn_cast<VRegType>(getOffsets().getType());
  auto resultType = dyn_cast<VRegType>(getResult().getType());
  if (!offsetsType || !resultType) {
    return emitOpError("offsets and result must be !pto.vreg<...>");
  }

  auto offsetsElemType = dyn_cast<IntegerType>(offsetsType.getElementType());
  if (!offsetsElemType) {
    return emitOpError("offset vector must use integer element type");
  }
  if (offsetsType.getElementCount() != resultType.getElementCount()) {
    return emitOpError("offset and result vectors must have the same element count");
  }

  Type sourceElemType = getBufferElementType(getSource().getType());
  Type resultElemType = resultType.getElementType();
  unsigned sourceElemWidth = getPTOStorageElemBitWidth(sourceElemType);
  unsigned resultElemWidth = getPTOStorageElemBitWidth(resultElemType);
  unsigned expectedOffsetWidth = 0;
  StringRef expectedMaskGranularity;
  int64_t expectedLanes = 0;

  if (sourceElemWidth == mlir::pto::kValue8 && isa<IntegerType>(sourceElemType)) {
    if (!isVgather2B8ResultType(cast<IntegerType>(sourceElemType),
                                dyn_cast<IntegerType>(resultElemType))) {
      return emitOpError(
          "8-bit gather requires i8/ui8 source and matching i16/ui16 result");
    }
    expectedOffsetWidth = mlir::pto::kValue16;
    expectedMaskGranularity = "b16";
    expectedLanes = mlir::pto::kValue128;
  } else if (sourceElemWidth == mlir::pto::kValue16) {
    if (auto sourceInt = dyn_cast<IntegerType>(sourceElemType)) {
      if (!isSameVgather2IntegerSemantics(
              sourceInt, dyn_cast<IntegerType>(resultElemType))) {
        return emitOpError(
            "16-bit integer gather requires matching i16/ui16 result");
      }
    } else if (!(sourceElemType.isF16() || sourceElemType.isBF16()) ||
               sourceElemType != resultElemType) {
      return emitOpError(
          "16-bit gather requires i16/ui16/f16/bf16 source and matching result");
    }
    expectedOffsetWidth = mlir::pto::kValue16;
    expectedMaskGranularity = "b16";
    expectedLanes = mlir::pto::kValue128;
  } else if (sourceElemWidth == mlir::pto::kValue32) {
    if (auto sourceInt = dyn_cast<IntegerType>(sourceElemType)) {
      if (!isSameVgather2IntegerSemantics(
              sourceInt, dyn_cast<IntegerType>(resultElemType))) {
        return emitOpError(
            "32-bit integer gather requires matching i32/ui32 result");
      }
    } else if (!sourceElemType.isF32() || sourceElemType != resultElemType) {
      return emitOpError(
          "32-bit gather requires i32/ui32/f32 source and matching result");
    }
    expectedOffsetWidth = mlir::pto::kValue32;
    expectedMaskGranularity = "b32";
    expectedLanes = mlir::pto::kValue64;
  } else {
    return emitOpError(
        "requires source element type i8/ui8/i16/ui16/i32/ui32/f16/bf16/f32");
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

LogicalResult CopyUbufToUbufOp::verify() {
  if (!isBufferLike(getSource().getType()) || !isBufferLike(getDestination().getType())) {
    return emitOpError("requires pointer-like source and destination");
  }
  if (classifyMemoryRole(getSource().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getDestination().getType()) != MemoryRole::UB) {
    return emitOpError("requires UB-backed source and destination");
  }
  return success();
}

void CopyCbufToUbufOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult CopyCbufToUbufOp::verify() {
  return verifyCopyCbufToUbufLikeOp(*this);
}

void CopyUbufToCbufOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult CopyUbufToCbufOp::verify() {
  if (!isBufferLike(getSource().getType()) || !isBufferLike(getDestination().getType())) {
    return emitOpError("requires pointer-like source and destination");
  }
  if (classifyMemoryRole(getSource().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getDestination().getType()) != MemoryRole::Other) {
    return emitOpError("requires UB-backed source and CBUF-backed destination");
  }
  return success();
}

void MteUbUbOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult MteUbUbOp::verify() {
  if (!isBufferLike(getSource().getType()) || !isBufferLike(getDestination().getType())) {
    return emitOpError("requires pointer-like source and destination");
  }
  if (classifyMemoryRole(getSource().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getDestination().getType()) != MemoryRole::UB) {
    return emitOpError("requires UB-backed source and destination");
  }
  return success();
}

void MteUbL1Op::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult MteUbL1Op::verify() {
  if (!isBufferLike(getSource().getType()) || !isBufferLike(getDestination().getType())) {
    return emitOpError("requires pointer-like source and destination");
  }
  if (classifyMemoryRole(getSource().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getDestination().getType()) != MemoryRole::Other) {
    return emitOpError("requires UB-backed source and CBUF-backed destination");
  }
  return success();
}

void VgatherbOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
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

  auto offsetsType = dyn_cast<VRegType>(getOffsets().getType());
  auto resultType = dyn_cast<VRegType>(getResult().getType());
  if (!offsetsType || !resultType) {
    return emitOpError("offsets and result must be !pto.vreg<...>");
  }
  auto offsetsElemType = dyn_cast<IntegerType>(offsetsType.getElementType());
  if (!offsetsElemType) {
    return emitOpError("offset vector must use integer element type");
  }
  if (offsetsElemType.getWidth() != mlir::pto::kValue32) {
    return emitOpError("currently requires 32-bit offset vector elements");
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

  auto offsetsType = dyn_cast<VRegType>(getOffsets().getType());
  auto resultType = dyn_cast<VRegType>(getResult().getType());
  if (!offsetsType || !resultType) {
    return emitOpError("offsets and result must be !pto.vreg<...>");
  }
  auto offsetsElemType = dyn_cast<IntegerType>(offsetsType.getElementType());
  if (!offsetsElemType) {
    return emitOpError("offset vector must use integer element type");
  }
  if (offsetsElemType.getWidth() != mlir::pto::kValue32) {
    return emitOpError("currently requires 32-bit offset vector elements");
  }
  if (offsetsType.getElementCount() != resultType.getElementCount()) {
    return emitOpError("offset and result vectors must have the same element count");
  }
  return success();
}

LogicalResult VbitsortOp::verify() {
  if (!isBufferLike(getDestination().getType()) || !isBufferLike(getSource().getType()) ||
      !isBufferLike(getIndices().getType())) {
    return emitOpError("requires pointer-like destination/source/indices");
  }
  if (classifyMemoryRole(getDestination().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSource().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getIndices().getType()) != MemoryRole::UB) {
    return emitOpError("requires UB-backed destination/source/indices");
  }
  if (!getRepeatTimes().getType().isIndex()) {
    return emitOpError("repeat_times must be index");
  }
  if (failed(verifyNotNestedInVecScope(*this, "pto.vbitsort"))) {
    return failure();
  }
  return success();
}

void VbitsortOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getIndicesMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult Vmrgsort4Op::verify() {
  if (!isBufferLike(getDestination().getType()) || !isBufferLike(getSource0().getType()) ||
      !isBufferLike(getSource1().getType()) || !isBufferLike(getSource2().getType()) ||
      !isBufferLike(getSource3().getType())) {
    return emitOpError("requires pointer-like destination and sources");
  }
  if (classifyMemoryRole(getDestination().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSource0().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSource1().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSource2().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSource3().getType()) != MemoryRole::UB) {
    return emitOpError("requires UB-backed destination and sources");
  }
  auto dstPtrType = dyn_cast<pto::PtrType>(getDestination().getType());
  auto src0PtrType = dyn_cast<pto::PtrType>(getSource0().getType());
  auto src1PtrType = dyn_cast<pto::PtrType>(getSource1().getType());
  auto src2PtrType = dyn_cast<pto::PtrType>(getSource2().getType());
  auto src3PtrType = dyn_cast<pto::PtrType>(getSource3().getType());
  if (!dstPtrType || !src0PtrType || !src1PtrType || !src2PtrType ||
      !src3PtrType) {
    return emitOpError("requires ptr-backed destination and sources");
  }

  Type elemType = dstPtrType.getElementType();
  if (src0PtrType.getElementType() != elemType ||
      src1PtrType.getElementType() != elemType ||
      src2PtrType.getElementType() != elemType ||
      src3PtrType.getElementType() != elemType) {
    return emitOpError(
        "requires destination and all sources to have the same element type");
  }
  if (!elemType.isF16() && !elemType.isF32()) {
    return emitOpError("requires f16 or f32 element type");
  }
  if (failed(verifyNotNestedInVecScope(*this, "pto.vmrgsort4"))) {
    return failure();
  }
  return success();
}

LogicalResult VmaxOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getLhs().getType(), "lhs")) ||
      failed(verifyVRegTypeLike(*this, getRhs().getType(), "rhs")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result"))) {
    return failure();
  }
  if (getLhs().getType() != getRhs().getType() ||
      getLhs().getType() != getResult().getType()) {
    return emitOpError("lhs, rhs, and result must have the same vector type");
  }
  return success();
}

LogicalResult VminOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getLhs().getType(), "lhs")) ||
      failed(verifyVRegTypeLike(*this, getRhs().getType(), "rhs")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result"))) {
    return failure();
  }
  if (getLhs().getType() != getRhs().getType() ||
      getLhs().getType() != getResult().getType()) {
    return emitOpError("lhs, rhs, and result must have the same vector type");
  }
  return success();
}

void VldsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

template <typename LoadOp>
static LogicalResult verifyVldsCommon(LoadOp op) {
  if (!isBufferLike(op.getSource().getType())) {
    return op.emitOpError("requires a pointer-like source");
  }

  if (failed(verifyVRegTypeLike(op, op.getResult().getType(), "result type"))) {
    return failure();
  }

  MemoryRole sourceRole = classifyMemoryRole(op.getSource().getType());
  if (sourceRole == MemoryRole::GM) {
    return op.emitOpError("requires a UB-backed source");
  }

  if (op.getDistAttr()) {
    StringRef dist = *op.getDist();
    Type elementType =
        cast<VRegType>(op.getResult().getType()).getElementType();
    if (!isSupportedVldsDistToken(dist, elementType)) {
      return op.emitOpError(
          "supports only NORM, BRC_B8/B16/B32, US_B8/B16, DS_B8/B16, "
          "UNPK_B8/B16/B32, BRC_BLK, E2B_B16/B32, UNPK4, SPLT4CHN, and "
          "SPLT2CHN_B8/B16 load distributions");
    }
  }

  return success();
}

LogicalResult VldsOp::verify() {
  if (failed(verifyVldsCommon(*this))) {
    return failure();
  }
  if (Value updatedBase = getUpdatedBase()) {
    if (updatedBase.getType() != getSource().getType()) {
      return emitOpError("requires updated base result to match base type");
    }
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

LogicalResult InitAlignOp::verify() {
  return verifyAlignTypeLike(*this, getResult().getType(), "result type");
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

LogicalResult VdupOp::verify() {
  auto resultType = dyn_cast<VRegType>(getResult().getType());
  if (!resultType) {
    return emitOpError("result must be !pto.vreg<...>");
  }

  std::optional<StringRef> granularity =
      getVdupMaskGranularity(resultType.getElementType());
  if (!granularity) {
    return emitOpError("result element type must use b8, b16, or b32 mask granularity");
  }
  if (failed(verifyMaskTypeWithGranularityLike(
          getOperation(), getMask().getType(), "mask type", *granularity))) {
    return failure();
  }

  if (!isSupportedVdupPosition(getPosition())) {
    return emitOpError("position must be LOWEST or HIGHEST");
  }

  Type inputType = getInput().getType();
  if (auto inputVecType = dyn_cast<VRegType>(inputType)) {
    if (inputVecType != resultType) {
      return emitOpError("vector input must match result vector type");
    }
    return success();
  }

  if (getPosition()) {
    return emitOpError("position is only supported for vector input");
  }

  Type resultElementType = resultType.getElementType();
  if (!isCompatibleScalarForSemanticType(resultElementType, inputType)) {
    return emitOpError("scalar input must match result element type");
  }

  return success();
}

LogicalResult TensorViewAddrOp::verify() {
  Type srcType = getSrc().getType();
  Type dstType = getDst().getType();

  Type elementType;
  int64_t expectedRank = -1;
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

LogicalResult PsetB8Op::verify() {
  if (failed(verifyMaskTypeWithGranularityLike(*this, getResult().getType(),
                                               "result type", "b8"))) {
    return failure();
  }

  if (!isSupportedPredicatePattern(getPattern())) {
    return emitOpError("requires a supported PAT_* predicate pattern");
  }
  return success();
}

LogicalResult PsetB16Op::verify() {
  if (failed(verifyMaskTypeWithGranularityLike(*this, getResult().getType(),
                                               "result type", "b16"))) {
    return failure();
  }

  if (!isSupportedPredicatePattern(getPattern())) {
    return emitOpError("requires a supported PAT_* predicate pattern");
  }
  return success();
}

LogicalResult PsetB32Op::verify() {
  if (failed(verifyMaskTypeWithGranularityLike(*this, getResult().getType(),
                                               "result type", "b32"))) {
    return failure();
  }
  if (!isSupportedPredicatePattern(getPattern())) {
    return emitOpError("requires a supported PAT_* predicate pattern");
  }
  return success();
}

LogicalResult PgeB8Op::verify() {
  if (failed(verifyMaskTypeWithGranularityLike(*this, getResult().getType(),
                                               "result type", "b8"))) {
    return failure();
  }
  if (!isSupportedPredicatePattern(getPattern())) {
    return emitOpError("requires a supported PAT_* predicate pattern");
  }
  return success();
}

LogicalResult PgeB16Op::verify() {
  if (failed(verifyMaskTypeWithGranularityLike(*this, getResult().getType(),
                                               "result type", "b16"))) {
    return failure();
  }
  if (!isSupportedPredicatePattern(getPattern())) {
    return emitOpError("requires a supported PAT_* predicate pattern");
  }
  return success();
}

LogicalResult PgeB32Op::verify() {
  if (failed(verifyMaskTypeWithGranularityLike(*this, getResult().getType(),
                                               "result type", "b32"))) {
    return failure();
  }
  if (!isSupportedPredicatePattern(getPattern())) {
    return emitOpError("requires a supported PAT_* predicate pattern");
  }
  return success();
}

template <typename PltOp>
static LogicalResult verifyPredicateLaneCountOp(PltOp op,
                                                StringRef granularity) {
  if (failed(verifyMaskTypeWithGranularityLike(op, op.getMask().getType(),
                                               "mask type", granularity))) {
    return failure();
  }
  Type scalarType = op.getScalar().getType();
  auto scalarIntType = dyn_cast<IntegerType>(scalarType);
  if (!scalarIntType || scalarIntType.getWidth() != mlir::pto::kValue32) {
    return op.emitOpError("requires scalar to be i32");
  }
  if (op.getScalarOut().getType() != scalarType) {
    return op.emitOpError("requires scalar_out to match scalar type");
  }
  return success();
}

LogicalResult PltB8Op::verify() { return verifyPredicateLaneCountOp(*this, "b8"); }
LogicalResult PltB16Op::verify() {
  return verifyPredicateLaneCountOp(*this, "b16");
}
LogicalResult PltB32Op::verify() {
  return verifyPredicateLaneCountOp(*this, "b32");
}

template <typename PltmOp>
static LogicalResult verifyPredicateLoopBoundOp(PltmOp op,
                                                StringRef granularity) {
  if (failed(verifyMaskTypeWithGranularityLike(op, op.getMask().getType(),
                                               "mask type", granularity))) {
    return failure();
  }
  if (!op.getLoop().getType().isInteger(mlir::pto::kValue16)) {
    return op.emitOpError("requires loop operand to be i16");
  }
  if (!op.getBound().getType().isInteger(mlir::pto::kValue32)) {
    return op.emitOpError("requires bound operand to be i32");
  }
  return success();
}

LogicalResult PltmB8Op::verify() {
  return verifyPredicateLoopBoundOp(*this, "b8");
}
LogicalResult PltmB16Op::verify() {
  return verifyPredicateLoopBoundOp(*this, "b16");
}
LogicalResult PltmB32Op::verify() {
  return verifyPredicateLoopBoundOp(*this, "b32");
}

LogicalResult PpackOp::verify() {
  if (failed(verifyMaskTypeLike(*this, getInput().getType(), "input type")) ||
      failed(verifyMaskTypeLike(*this, getResult().getType(), "result type"))) {
    return failure();
  }
  if (!isSupportedPartToken(getPart())) {
    return emitOpError("requires part to be LOWER or HIGHER");
  }
  auto inputMaskType = cast<MaskType>(getInput().getType());
  auto resultMaskType = cast<MaskType>(getResult().getType());
  StringRef inputGranularity = inputMaskType.getGranularity();
  StringRef resultGranularity = resultMaskType.getGranularity();
  if (inputGranularity != resultGranularity &&
      !isMaskGranularityAdjacentNarrowing(inputGranularity, resultGranularity)) {
    return emitOpError(
        "requires result mask granularity to match the input or narrow by one step");
  }
  return success();
}

LogicalResult PunpackOp::verify() {
  if (failed(verifyMaskTypeLike(*this, getInput().getType(), "input type")) ||
      failed(verifyMaskTypeLike(*this, getResult().getType(), "result type"))) {
    return failure();
  }
  if (!isSupportedPartToken(getPart())) {
    return emitOpError("requires part to be LOWER or HIGHER");
  }
  auto inputMaskType = cast<MaskType>(getInput().getType());
  auto resultMaskType = cast<MaskType>(getResult().getType());
  StringRef inputGranularity = inputMaskType.getGranularity();
  StringRef resultGranularity = resultMaskType.getGranularity();
  if (inputGranularity != resultGranularity &&
      !isMaskGranularityAdjacentWidening(inputGranularity, resultGranularity)) {
    return emitOpError(
        "requires result mask granularity to match the input or widen by one step");
  }
  return success();
}

LogicalResult PbitcastOp::verify() {
  if (failed(verifyMaskTypeLike(*this, getInput().getType(), "input type")) ||
      failed(verifyMaskTypeLike(*this, getResult().getType(), "result type"))) {
    return failure();
  }
  return success();
}

LogicalResult PnotOp::verify() {
  if (failed(verifyMaskTypeLike(*this, getInput().getType(), "input type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyMaskTypeLike(*this, getResult().getType(), "result type"))) {
    return failure();
  }
  return success();
}

LogicalResult PselOp::verify() {
  if (failed(verifyMaskTypeLike(*this, getSrc0().getType(), "src0 type")) ||
      failed(verifyMaskTypeLike(*this, getSrc1().getType(), "src1 type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyMaskTypeLike(*this, getResult().getType(), "result type"))) {
    return failure();
  }
  return success();
}

template <typename BinaryMaskOp>
static LogicalResult verifyBinaryMaskOp(BinaryMaskOp op) {
  if (failed(verifyMaskTypeLike(op, op.getSrc0().getType(), "src0 type")) ||
      failed(verifyMaskTypeLike(op, op.getSrc1().getType(), "src1 type")) ||
      failed(verifyMaskTypeLike(op, op.getMask().getType(), "mask type")) ||
      failed(verifyMaskTypeLike(op, op.getResult().getType(), "result type"))) {
    return failure();
  }
  return success();
}

LogicalResult PandOp::verify() { return verifyBinaryMaskOp(*this); }
LogicalResult PorOp::verify() { return verifyBinaryMaskOp(*this); }
LogicalResult PxorOp::verify() { return verifyBinaryMaskOp(*this); }

void PldsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult PldsOp::verify() {
  if (!isBufferLike(getSource().getType())) {
    return emitOpError("requires a pointer-like source");
  }
  if (failed(verifyMaskTypeLike(*this, getResult().getType(), "result type"))) {
    return failure();
  }
  MemoryRole sourceRole = classifyMemoryRole(getSource().getType());
  if (sourceRole == MemoryRole::GM) {
    return emitOpError("requires a UB-backed source");
  }
  if (!getOffset().getType().isIndex()) {
    return emitOpError("requires index offset");
  }
  if (!isSupportedPredicateLoadDist(getDist())) {
    return emitOpError("requires predicate load dist to be NORM, US, or DS");
  }
  if (getUpdatedBase() &&
      getUpdatedBase().getType() != getSource().getType()) {
    return emitOpError("requires updated base result to match base type");
  }
  return success();
}

void PldiOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult PldiOp::verify() {
  if (!isBufferLike(getSource().getType())) {
    return emitOpError("requires a pointer-like source");
  }
  if (failed(verifyMaskTypeLike(*this, getResult().getType(), "result type"))) {
    return failure();
  }
  if (classifyMemoryRole(getSource().getType()) == MemoryRole::GM) {
    return emitOpError("requires a UB-backed source");
  }
  if (!matchPattern(getOffset(), m_Constant())) {
    return emitOpError("requires offset to be a constant index immediate");
  }
  if (!isSupportedPredicateLoadDist(getDist())) {
    return emitOpError("requires predicate load dist to be NORM, US, or DS");
  }
  if (getUpdatedBase() &&
      getUpdatedBase().getType() != getSource().getType()) {
    return emitOpError("requires updated base result to match base type");
  }
  return success();
}

template <typename OpTy>
static LogicalResult verifyElementwiseVecScalarOpLike(OpTy op) {
  auto inputType = dyn_cast<VRegType>(op.getInput().getType());
  auto resultType = dyn_cast<VRegType>(op.getResult().getType());
  if (!inputType || !resultType) {
    return op.emitOpError("input and result must be !pto.vreg<...>");
  }
  if (inputType != resultType) {
    return op.emitOpError("input and result vector types must match");
  }

  Type elemType = inputType.getElementType();
  Type scalarType = op.getScalar().getType();
  if (scalarType == elemType) {
    return success();
  }

  auto elemInt = dyn_cast<IntegerType>(elemType);
  auto scalarInt = dyn_cast<IntegerType>(scalarType);
  if (!elemInt || !scalarInt || elemInt.getWidth() != scalarInt.getWidth()) {
    return op.emitOpError("scalar type must match vector element type");
  }

  if (elemInt.isSigned() && (scalarInt.isSigned() || scalarInt.isSignless())) {
    return success();
  }
  if (elemInt.isUnsigned() &&
      (scalarInt.isUnsigned() || scalarInt.isSignless())) {
    return success();
  }
  if (elemInt.isSignless() && scalarInt.isSignless()) {
    return success();
  }

  return op.emitOpError(
      "integer scalar type must match vector element width and use matching signedness or signless i<width>");
}

template <typename OpTy>
static LogicalResult verifyVecScalarOpLike(OpTy op) {
  if (failed(verifyElementwiseVecScalarOpLike(op))) {
    return failure();
  }
  return success();
}

template <typename OpTy>
static LogicalResult verifyVecScalarMaskedOpLike(OpTy op) {
  if (failed(verifyElementwiseVecScalarOpLike(op))) {
    return failure();
  }
  if (failed(verifyMaskTypeLike(op, op.getMask().getType(), "mask type"))) {
    return failure();
  }
  if (failed(verifyNonLowPrecisionVRegElementTypeLike(
          op.getOperation(), op.getInput().getType(), "input type"))) {
    return failure();
  }
  return success();
}

template <typename CarryOp>
static LogicalResult verifyCarryVecOp(CarryOp op) {
  if (failed(verifyIntegerVRegTypeLike(op, op.getLhs().getType(), "lhs type")) ||
      failed(verifyIntegerVRegTypeLike(op, op.getRhs().getType(), "rhs type")) ||
      failed(verifyMaskTypeLike(op, op.getMask().getType(), "mask type")) ||
      failed(verifyIntegerVRegTypeLike(op, op.getResult().getType(),
                                       "result type")) ||
      failed(verifyMaskTypeLike(op, op.getCarry().getType(), "carry type"))) {
    return failure();
  }

  auto lhsType = cast<VRegType>(op.getLhs().getType());
  auto rhsType = cast<VRegType>(op.getRhs().getType());
  auto resultType = cast<VRegType>(op.getResult().getType());
  auto lhsElemType = cast<IntegerType>(lhsType.getElementType());
  if (lhsType != rhsType || lhsType != resultType) {
    return op.emitOpError("requires lhs, rhs, and result to have matching vector types");
  }
  if (lhsElemType.getWidth() != mlir::pto::kValue32) {
    return op.emitOpError("currently requires 32-bit integer vector elements");
  }
  return success();
}

template <typename CarryWithInputOp>
static LogicalResult verifyCarryVecOpWithInput(CarryWithInputOp op) {
  if (failed(verifyCarryVecOp(op)) ||
      failed(verifyMaskTypeLike(op, op.getCarryIn().getType(),
                                "carry_in type"))) {
    return failure();
  }
  return success();
}

LogicalResult VmulsOp::verify() { return verifyVecScalarMaskedOpLike(*this); }
LogicalResult VaddsOp::verify() { return verifyVecScalarMaskedOpLike(*this); }
LogicalResult VmaxsOp::verify() { return verifyVecScalarMaskedOpLike(*this); }
LogicalResult VminsOp::verify() { return verifyVecScalarMaskedOpLike(*this); }
LogicalResult VlreluOp::verify() { return verifyVecScalarMaskedOpLike(*this); }
LogicalResult VshlsOp::verify() {
  auto inputType = dyn_cast<VRegType>(getInput().getType());
  auto resultType = dyn_cast<VRegType>(getResult().getType());
  if (!inputType || !resultType) {
    return emitOpError("input and result must be !pto.vreg<...>");
  }
  if (failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type"))) {
    return failure();
  }
  if (inputType != resultType) {
    return emitOpError("input and result vector types must match");
  }
  if (!isa<IntegerType>(inputType.getElementType())) {
    return emitOpError("requires integer vector and integer scalar");
  }
  auto scalarType = dyn_cast<IntegerType>(getScalar().getType());
  if (!scalarType || !scalarType.isSignlessInteger(mlir::pto::kValue16)) {
    return emitOpError("requires signless i16 scalar");
  }
  return success();
}
LogicalResult VshrsOp::verify() {
  auto inputType = dyn_cast<VRegType>(getInput().getType());
  auto resultType = dyn_cast<VRegType>(getResult().getType());
  if (!inputType || !resultType) {
    return emitOpError("input and result must be !pto.vreg<...>");
  }
  if (failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type"))) {
    return failure();
  }
  if (inputType != resultType) {
    return emitOpError("input and result vector types must match");
  }
  if (!isa<IntegerType>(inputType.getElementType())) {
    return emitOpError("requires integer vector and integer scalar");
  }
  auto scalarType = dyn_cast<IntegerType>(getScalar().getType());
  if (!scalarType || !scalarType.isSignlessInteger(mlir::pto::kValue16)) {
    return emitOpError("requires signless i16 scalar");
  }
  return success();
}

LogicalResult VabsOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getInput().getType(), "operand type"))) {
    return failure();
  }
  if (failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type"))) {
    return failure();
  }
  if (failed(verifyVRegTypeLike(*this, getResult().getType(), "result type"))) {
    return failure();
  }
  if (getInput().getType() != getResult().getType()) {
    return emitOpError("requires matching register vector shape");
  }
  return success();
}

template <typename UnaryOp>
static LogicalResult verifyUnaryVecOp(UnaryOp op) {
  if (failed(verifyVRegTypeLike(op, op.getInput().getType(), "operand type"))) {
    return failure();
  }
  if (failed(verifyMaskTypeLike(op, op.getMask().getType(), "mask type"))) {
    return failure();
  }
  if (failed(verifyVRegTypeLike(op, op.getResult().getType(), "result type"))) {
    return failure();
  }
  if (failed(verifyNonLowPrecisionVRegElementTypeLike(
          op.getOperation(), op.getInput().getType(), "operand type"))) {
    return failure();
  }
  if (op.getInput().getType() != op.getResult().getType()) {
    return op.emitOpError("requires matching register vector shape");
  }
  return success();
}

LogicalResult VexpOp::verify() { return verifyUnaryVecOp(*this); }
LogicalResult VlnOp::verify() { return verifyUnaryVecOp(*this); }
LogicalResult VsqrtOp::verify() { return verifyUnaryVecOp(*this); }
LogicalResult VnegOp::verify() { return verifyUnaryVecOp(*this); }
LogicalResult VreluOp::verify() {
  if (failed(verifyUnaryVecOp(*this))) {
    return failure();
  }
  auto inputType = cast<VRegType>(getInput().getType());
  Type elemType = inputType.getElementType();
  if (auto intType = dyn_cast<IntegerType>(elemType)) {
    if (intType.getWidth() != mlir::pto::kValue32 || intType.isUnsigned()) {
      return emitOpError("requires si32/i32/f16/f32 vector element type");
    }
    return success();
  }
  if (!elemType.isF16() && !elemType.isF32()) {
    return emitOpError("requires si32/i32/f16/f32 vector element type");
  }
  return success();
}
LogicalResult VnotOp::verify() { return verifyUnaryVecOp(*this); }

template <typename BinaryOp>
static LogicalResult verifyBinaryVecOp(BinaryOp op,
                                       bool allowLowPrecision = false) {
  if (failed(verifyVRegTypeLike(op, op.getLhs().getType(), "lhs type"))) {
    return failure();
  }
  if (failed(verifyVRegTypeLike(op, op.getRhs().getType(), "rhs type"))) {
    return failure();
  }
  if (failed(verifyMaskTypeLike(op, op.getMask().getType(), "mask type"))) {
    return failure();
  }
  if (failed(verifyVRegTypeLike(op, op.getResult().getType(), "result type"))) {
    return failure();
  }
  if (!allowLowPrecision &&
      failed(verifyNonLowPrecisionVRegElementTypeLike(
          op.getOperation(), op.getLhs().getType(), "lhs type"))) {
    return failure();
}
  if (allowLowPrecision) {
    auto lhsType = cast<VRegType>(op.getLhs().getType());
    if (pto::isPTOBF16x2Type(lhsType.getElementType())) {
      return op.emitOpError(
          "does not support bf16x2 vector elements; low-precision bitwise "
          "operations require an 8-bit payload type");
}
  }
  if (op.getLhs().getType() != op.getRhs().getType() ||
      op.getLhs().getType() != op.getResult().getType()) {
    return op.emitOpError("requires matching register vector shapes");
}
  return success();
}

LogicalResult VaddOp::verify() { return verifyBinaryVecOp(*this); }
LogicalResult VsubOp::verify() { return verifyBinaryVecOp(*this); }
LogicalResult VmulOp::verify() { return verifyBinaryVecOp(*this); }
LogicalResult VdivOp::verify() { return verifyBinaryVecOp(*this); }
LogicalResult VandOp::verify() {
  return verifyBinaryVecOp(*this, /*allowLowPrecision=*/true);
}
LogicalResult VorOp::verify() {
  return verifyBinaryVecOp(*this, /*allowLowPrecision=*/true);
}
LogicalResult VxorOp::verify() {
  return verifyBinaryVecOp(*this, /*allowLowPrecision=*/true);
}

template <typename TernaryOp>
static LogicalResult verifyTernaryVecOp(TernaryOp op) {
  if (failed(verifyVRegTypeLike(op, op.getAcc().getType(), "acc type")) ||
      failed(verifyVRegTypeLike(op, op.getLhs().getType(), "lhs type")) ||
      failed(verifyVRegTypeLike(op, op.getRhs().getType(), "rhs type")) ||
      failed(verifyMaskTypeLike(op, op.getMask().getType(), "mask type")) ||
      failed(verifyVRegTypeLike(op, op.getResult().getType(), "result type"))) {
    return failure();
  }
  if (op.getAcc().getType() != op.getLhs().getType() ||
      op.getAcc().getType() != op.getRhs().getType() ||
      op.getAcc().getType() != op.getResult().getType()) {
    return op.emitOpError(
        "requires acc, lhs, rhs, and result to share one vector type");
  }
  return success();
}

LogicalResult VmaddOp::verify() {
  if (failed(verifyTernaryVecOp(*this))) {
    return failure();
  }
  Type elemType = cast<VRegType>(getAcc().getType()).getElementType();
  if (!elemType.isF16() && !elemType.isBF16() && !elemType.isF32()) {
    return emitOpError("requires f16/bf16/f32 vector element type");
  }
  return success();
}
// Shared verifier for vshl/vshr. Shift counts use a signed integer vreg.
// DSL frontends must bitcast signless or unsigned count vectors to this
// canonical form.
template <typename BinaryOp>
static LogicalResult verifyShiftVecOp(BinaryOp op) {
  const bool hasInvalidOperandType =
      failed(verifyVRegTypeLike(op, op.getLhs().getType(), "lhs type")) ||
      failed(verifyVRegTypeLike(op, op.getRhs().getType(), "rhs type")) ||
      failed(verifyMaskTypeLike(op, op.getMask().getType(), "mask type")) ||
      failed(verifyVRegTypeLike(op, op.getResult().getType(), "result type"));
  if (hasInvalidOperandType) {
    return failure();
  }
  if (failed(verifyNonLowPrecisionVRegElementTypeLike(
          op.getOperation(), op.getLhs().getType(), "lhs type"))) {
    return failure();
  }

  auto lhsType = cast<VRegType>(op.getLhs().getType());
  auto rhsType = cast<VRegType>(op.getRhs().getType());
  auto resultType = cast<VRegType>(op.getResult().getType());

  // Shifting is only meaningful for integer vectors.
  if (!isa<IntegerType>(lhsType.getElementType())) {
    return op.emitOpError("requires integer vector element type");
  }

  // Result type must match lhs exactly.
  if (lhsType != resultType) {
    return op.emitOpError("requires matching result register vector shape");
  }

  // Shift count must have the same lane count and element bitwidth as the
  // shifted data.
  const bool hasMismatchedLaneCount =
      lhsType.getElementCount() != rhsType.getElementCount();
  if (hasMismatchedLaneCount) {
    return op.emitOpError("requires matching lane count for shift count");
  }
  auto lhsElem = cast<IntegerType>(lhsType.getElementType());
  auto rhsElem = dyn_cast<IntegerType>(rhsType.getElementType());
  if (!rhsElem) {
    return op.emitOpError(
        "requires integer vector element type for shift count");
  }
  const bool hasMismatchedElementBitwidth =
      rhsElem.getWidth() != lhsElem.getWidth();
  if (hasMismatchedElementBitwidth) {
    return op.emitOpError(
        "requires shift count with matching element bitwidth");
  }
  if (!rhsElem.isSigned()) {
    return op.emitOpError(
        "requires shift count to use a signed integer element type");
  }
  return success();
}

LogicalResult VshlOp::verify() { return verifyShiftVecOp(*this); }
LogicalResult VshrOp::verify() { return verifyShiftVecOp(*this); }
LogicalResult VaddcOp::verify() { return verifyCarryVecOp(*this); }
LogicalResult VsubcOp::verify() { return verifyCarryVecOp(*this); }
LogicalResult VaddcsOp::verify() { return verifyCarryVecOpWithInput(*this); }
LogicalResult VsubcsOp::verify() { return verifyCarryVecOpWithInput(*this); }

template <typename ReductionOp>
static LogicalResult verifyReductionVecOp(ReductionOp op) {
  return verifyUnaryVecOp(op);
}

template <typename ReductionOp>
static LogicalResult verifyGroupReductionVecOp(ReductionOp op) {
  if (failed(verifyReductionVecOp(op))) {
    return failure();
  }
  auto inputType = cast<VRegType>(op.getInput().getType());
  Type elemType = inputType.getElementType();
  if (auto intType = dyn_cast<IntegerType>(elemType)) {
    if (intType.getWidth() != mlir::pto::kValue8 && intType.getWidth() != 16 &&
        intType.getWidth() != mlir::pto::kValue32) {
      return op.emitOpError(
          "requires 8-bit, 16-bit, or 32-bit integer vector element type");
    }
    return success();
  }
  if (!elemType.isF16() && !elemType.isF32()) {
    return op.emitOpError("requires i16/i32/f16/f32 vector element type");
  }
  return success();
}

LogicalResult VcgaddOp::verify() { return verifyGroupReductionVecOp(*this); }
LogicalResult VcgmaxOp::verify() { return verifyGroupReductionVecOp(*this); }
LogicalResult VcgminOp::verify() { return verifyGroupReductionVecOp(*this); }
LogicalResult VcpaddOp::verify() {
  if (failed(verifyReductionVecOp(*this))) {
    return failure();
  }
  auto inputType = cast<VRegType>(getInput().getType());
  Type elemType = inputType.getElementType();
  if (!elemType.isF16() && !elemType.isF32()) {
    return emitOpError("requires f16 or f32 vector element type");
  }
  return success();
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

template <typename ExtremaOp>
static LogicalResult verifyExtremaPredicateOp(ExtremaOp op) {
  if (failed(verifyVRegTypeLike(op, op.getInput().getType(), "input type")) ||
      failed(verifyMaskTypeLike(op, op.getMask().getType(), "mask type")) ||
      failed(verifyVRegTypeLike(op, op.getValue().getType(), "value type")) ||
      failed(verifyMaskTypeLike(op, op.getPredicate().getType(),
                                "predicate type"))) {
    return failure();
  }
  if (op.getInput().getType() != op.getValue().getType()) {
    return op.emitOpError(
        "requires input and value result to share one vector type");
  }
  if (op.getMask().getType() != op.getPredicate().getType()) {
    return op.emitOpError(
        "requires mask and predicate result to share one mask type");
  }

  Type elemType = cast<VRegType>(op.getInput().getType()).getElementType();
  if (elemType.isF16() || elemType.isF32()) {
    return success();
  }
  auto intType = dyn_cast<IntegerType>(elemType);
  if (!intType || (intType.getWidth() != mlir::pto::kValue8 && intType.getWidth() != 16 &&
                   intType.getWidth() != mlir::pto::kValue32)) {
    return op.emitOpError("requires i8/i16/i32/f16/f32 vector element type");
  }
  return success();
}

LogicalResult VcbmaxOp::verify() { return verifyExtremaPredicateOp(*this); }
LogicalResult VcbminOp::verify() { return verifyExtremaPredicateOp(*this); }

template <typename SelectOp>
static LogicalResult verifyLaneSelectOp(SelectOp op) {
  if (failed(verifyVRegTypeLike(op, op.getSrc0().getType(), "src0 type")) ||
      failed(verifyVRegTypeLike(op, op.getSrc1().getType(), "src1 type")) ||
      failed(verifyVRegTypeLike(op, op.getResult().getType(), "result type"))) {
    return failure();
  }

  auto src0Type = cast<VRegType>(op.getSrc0().getType());
  auto src1Type = cast<VRegType>(op.getSrc1().getType());
  auto resultType = cast<VRegType>(op.getResult().getType());
  if (src0Type != resultType) {
    return op.emitOpError("requires src0 and result to have identical vector types");
  }
  if (src1Type.getElementCount() != src0Type.getElementCount()) {
    return op.emitOpError("requires src0/src1 to have identical element counts");
  }
  auto src1ElemType = dyn_cast<IntegerType>(src1Type.getElementType());
  if (!src1ElemType) {
    return op.emitOpError("requires src1 to use integer vector elements");
  }
  if (src1ElemType.getWidth() != getIntOrFloatBitWidth(src0Type.getElementType())) {
    return op.emitOpError("requires src1 integer element width to match src0 element width");
  }
  return success();
}

template <typename PairOp>
static LogicalResult verifyPairVecResults(PairOp op) {
  if (failed(verifyVRegTypeLike(op, op.getLhs().getType(), "lhs type")) ||
      failed(verifyVRegTypeLike(op, op.getRhs().getType(), "rhs type")) ||
      failed(verifyVRegTypeLike(op, op.getLow().getType(), "low result type")) ||
      failed(verifyVRegTypeLike(op, op.getHigh().getType(), "high result type"))) {
    return failure();
  }
  if (op.getLhs().getType() != op.getRhs().getType() ||
      op.getLhs().getType() != op.getLow().getType() ||
      op.getLhs().getType() != op.getHigh().getType()) {
    return op.emitOpError("requires operands and results to share one vector type");
  }
  return success();
}

template <typename PartOp>
static LogicalResult verifyPartVecOp(PartOp op) {
  if (failed(verifyVRegTypeLike(op, op.getLhs().getType(), "lhs type")) ||
      failed(verifyVRegTypeLike(op, op.getRhs().getType(), "rhs type")) ||
      failed(verifyVRegTypeLike(op, op.getResult().getType(), "result type"))) {
    return failure();
  }
  if (op.getLhs().getType() != op.getRhs().getType() ||
      op.getLhs().getType() != op.getResult().getType()) {
    return op.emitOpError("requires operands and result to share one vector type");
  }
  if (!isSupportedPartToken(op.getPart())) {
    return op.emitOpError("requires part to be LOWER or HIGHER");
  }
  return success();
}

LogicalResult VselOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getSrc0().getType(), "src0 type")) ||
      failed(verifyVRegTypeLike(*this, getSrc1().getType(), "src1 type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result type"))) {
    return failure();
  }
  if (failed(verifyNonLowPrecisionVRegElementTypeLike(
          getOperation(), getSrc0().getType(), "src0 type"))) {
    return failure();
  }
  if (getSrc0().getType() != getSrc1().getType() ||
      getSrc0().getType() != getResult().getType()) {
    return emitOpError("requires src0, src1, and result to have identical vector types");
  }
  return success();
}

LogicalResult VselrOp::verify() { return verifyLaneSelectOp(*this); }
LogicalResult Vselrv2Op::verify() { return verifyLaneSelectOp(*this); }

LogicalResult VsqzOp::verify() { return verifyUnaryVecOp(*this); }

LogicalResult VusqzOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getSrc().getType(), "src type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result type"))) {
    return failure();
  }
  if (getSrc().getType() != getResult().getType()) {
    return emitOpError("requires src and result to share one vector type");
  }
  auto srcType = cast<VRegType>(getSrc().getType());
  auto elemType = dyn_cast<IntegerType>(srcType.getElementType());
  if (!elemType) {
    return emitOpError("requires signed integer vector element type");
  }
  if (elemType.isUnsigned()) {
    return emitOpError("requires signed integer vector element type");
  }
  unsigned width = elemType.getWidth();
  if (width != mlir::pto::kValue8 && width != 16 && width != mlir::pto::kValue32) {
    return emitOpError("requires s8/s16/s32 vector element type");
  }
  return success();
}

LogicalResult VpackOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getSrc().getType(), "src type")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result type"))) {
    return failure();
  }
  if (!isSupportedPartToken(getPart())) {
    return emitOpError("requires part to be LOWER or HIGHER");
  }
  auto srcType = cast<VRegType>(getSrc().getType());
  auto resultType = cast<VRegType>(getResult().getType());
  Type srcElemType = srcType.getElementType();
  Type resultElemType = resultType.getElementType();
  if (!isa<IntegerType>(srcElemType) || !isa<IntegerType>(resultElemType)) {
    return emitOpError("currently requires integer source and result element types");
  }
  if (resultType.getElementCount() != srcType.getElementCount() * mlir::pto::kValue2) {
    return emitOpError(
        "requires result element count to be twice the source element count");
  }
  unsigned srcWidth = getIntOrFloatBitWidth(srcElemType);
  unsigned resultWidth = getIntOrFloatBitWidth(resultElemType);
  if (!srcWidth || resultWidth * mlir::pto::kValue2 != srcWidth) {
    return emitOpError(
        "requires result element width to be half the source element width");
  }
  auto srcIntType = cast<IntegerType>(srcElemType);
  auto resultIntType = cast<IntegerType>(resultElemType);
  if (!resultIntType.isUnsigned()) {
    return emitOpError("requires unsigned result element type");
  }
  if (!((srcIntType.getWidth() == mlir::pto::kValue32 &&
         resultIntType.getWidth() == mlir::pto::kValue16) ||
        (srcIntType.getWidth() == mlir::pto::kValue16 &&
         resultIntType.getWidth() == mlir::pto::kValue8))) {
    return emitOpError(
        "currently supports only s32/u32 -> u16 and s16/u16 -> u8");
  }
  return success();
}

template <typename UnpackOp>
static LogicalResult verifyUnpackVecOp(UnpackOp op) {
  if (failed(verifyVRegTypeLike(op, op.getSrc().getType(), "src type")) ||
      failed(verifyVRegTypeLike(op, op.getResult().getType(), "result type"))) {
    return failure();
  }
  auto srcType = cast<VRegType>(op.getSrc().getType());
  auto resultType = cast<VRegType>(op.getResult().getType());
  Type srcElemType = srcType.getElementType();
  Type resultElemType = resultType.getElementType();
  if (!isa<IntegerType>(srcElemType) || !isa<IntegerType>(resultElemType)) {
    return op.emitOpError(
        "currently requires integer source and result element types");
  }
  if (srcType.getElementCount() != resultType.getElementCount() * 2) {
    return op.emitOpError(
        "requires source element count to be twice the result element count");
  }
  unsigned srcWidth = getIntOrFloatBitWidth(srcElemType);
  unsigned resultWidth = getIntOrFloatBitWidth(resultElemType);
  if (!srcWidth || srcWidth * mlir::pto::kValue2 != resultWidth) {
    return op.emitOpError(
        "requires result element width to be twice the source element width");
  }
  return success();
}

LogicalResult VsunpackOp::verify() { return verifyUnpackVecOp(*this); }
LogicalResult VzunpackOp::verify() { return verifyUnpackVecOp(*this); }

static bool isSupportedCmpMode(StringRef mode) {
  return mode == "eq" || mode == "ne" || mode == "lt" || mode == "le" ||
         mode == "gt" || mode == "ge";
}

LogicalResult VcmpOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getSrc0().getType(), "src0 type")) ||
      failed(verifyVRegTypeLike(*this, getSrc1().getType(), "src1 type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyMaskTypeLike(*this, getResult().getType(), "result type"))) {
    return failure();
  }
  if (getSrc0().getType() != getSrc1().getType()) {
    return emitOpError("requires src0 and src1 to have identical vector types");
  }
  if (!isSupportedCmpMode(getCmpMode())) {
    return emitOpError("requires cmp_mode to be one of eq/ne/lt/le/gt/ge");
  }
  return success();
}

LogicalResult VcmpsOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getSrc().getType(), "src type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyMaskTypeLike(*this, getResult().getType(), "result type"))) {
    return failure();
  }
  auto srcType = cast<VRegType>(getSrc().getType());
  Type srcElementType = srcType.getElementType();
  Type scalarType = getScalar().getType();
  if (!isCompatibleScalarForSemanticType(srcElementType, scalarType)) {
    return emitOpError("requires scalar type to match source element type");
  }
  if (!isSupportedCmpMode(getCmpMode())) {
    return emitOpError("requires cmp_mode to be one of eq/ne/lt/le/gt/ge");
  }
  return success();
}

ParseResult VtrcOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand input;
  OpAsmParser::UnresolvedOperand mask;
  std::string roundModeToken;
  NamedAttrList attrs;
  Type inputType, maskType, resultType;

  if (parser.parseOperand(input) || parser.parseComma() ||
      parser.parseOperand(mask) || parser.parseComma() ||
      parser.parseKeywordOrString(&roundModeToken) ||
      parser.parseOptionalAttrDict(attrs) ||
      parser.parseColonType(inputType) || parser.parseComma() ||
      parser.parseType(maskType) || parser.parseArrow() ||
      parser.parseType(resultType)) {
    return failure();
  }

  auto normalized = normalizeRoundModeToken(roundModeToken);
  if (!normalized || !isSupportedVtrcRoundMode(*normalized)) {
    return parser.emitError(parser.getCurrentLocation())
           << "round mode must be one of R/A/F/C/Z or "
              "ROUND_R/ROUND_A/ROUND_F/ROUND_C/ROUND_Z";
  }

  attrs.set("round_mode", parser.getBuilder().getStringAttr(*normalized));
  result.addAttributes(attrs);
  if (parser.resolveOperand(input, inputType, result.operands) ||
      parser.resolveOperand(mask, maskType, result.operands)) {
    return failure();
  }
  result.addTypes(resultType);
  return success();
}

void VtrcOp::print(OpAsmPrinter &printer) {
  printer << ' ' << getInput() << ", " << getMask() << ", ";
  Builder builder(getContext());
  auto normalized = normalizeRoundModeToken(getRoundMode());
  printer.printAttributeWithoutType(
      builder.getStringAttr(normalized.value_or(getRoundMode())));
  printer.printOptionalAttrDict((*this)->getAttrs(), {"round_mode"});
  printer << " : " << getInput().getType() << ", " << getMask().getType()
          << " -> " << getResult().getType();
}

LogicalResult VtrcOp::verify() {
  auto inputType = dyn_cast<VRegType>(getInput().getType());
  auto resultType = dyn_cast<VRegType>(getResult().getType());
  if (!inputType || !resultType) {
    return emitOpError("input and result must be !pto.vreg<...>");
  }
  if (failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type"))) {
    return failure();
  }
  if (inputType != resultType) {
    return emitOpError("requires input and result to have identical vreg type");
  }
  auto elemType = inputType.getElementType();
  if (!(elemType.isF16() || elemType.isF32() || elemType.isBF16())) {
    return emitOpError("requires f16/f32/bf16 vector element type");
  }
  auto expectedGranularity = getVdupMaskGranularity(elemType);
  if (!expectedGranularity) {
    return emitOpError("requires element type with supported predicate granularity");
  }
  if (failed(verifyMaskTypeWithGranularityLike(*this, getMask().getType(),
                                               "mask type",
                                               *expectedGranularity))) {
    return failure();
  }
  auto normalized = normalizeRoundModeToken(getRoundMode());
  if (!normalized || !isSupportedVtrcRoundMode(*normalized)) {
    return emitOpError("round mode must be one of R/A/F/C/Z");
  }
  return success();
}

LogicalResult VmulscvtOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getInput().getType(), "input type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result type"))) {
    return failure();
  }

  auto inputType = cast<VRegType>(getInput().getType());
  auto resultType = cast<VRegType>(getResult().getType());
  if (!inputType.getElementType().isF32()) {
    return emitOpError("requires f32 input vector element type");
  }
  if (!resultType.getElementType().isF16()) {
    return emitOpError("requires f16 result vector element type");
  }

  auto scalarType = getScalar().getType();
  if (!scalarType.isF32()) {
    return emitOpError("requires f32 scalar operand");
  }

  if (failed(verifyMaskTypeWithGranularityLike(*this, getMask().getType(),
                                               "mask type", "b32"))) {
    return failure();
  }

  auto inputBits = getVRegStorageBitWidth(inputType);
  auto resultBits = getVRegStorageBitWidth(resultType);
  if (!inputBits || !resultBits || *inputBits != *resultBits) {
    return emitOpError(
        "requires source and result to preserve total vector storage width");
  }

  auto normalizedRnd = normalizeRoundModeToken(getRnd());
  if (!normalizedRnd) {
    return emitOpError("rnd must be one of R/A/F/C/Z/O");
  }
  if (*normalizedRnd != "A") {
    return emitOpError("currently only supports rnd A");
  }

  auto normalizedPart = normalizeEvenOddPartToken(getPart());
  if (!normalizedPart) {
    return emitOpError("part must be EVEN or ODD");
  }
  return success();
}

ParseResult VcvtOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand input;
  OpAsmParser::UnresolvedOperand mask;
  NamedAttrList attrs;
  Type inputType, maskType, resultType;

  if (parser.parseOperand(input) || parser.parseComma() ||
      parser.parseOperand(mask) || parser.parseOptionalAttrDict(attrs) ||
      parser.parseColonType(inputType) || parser.parseComma() ||
      parser.parseType(maskType) || parser.parseArrow() ||
      parser.parseType(resultType)) {
    return failure();
  }

  Attribute legacyRndAttr = attrs.get("round_mode");
  Attribute rndAttr = attrs.get("rnd");
  if (legacyRndAttr && rndAttr) {
    return parser.emitError(parser.getCurrentLocation())
           << "rnd and round_mode cannot be specified together";
  }

  auto normalizeNamedStringAttr =
      [&](StringRef sourceName, StringRef canonicalName,
          auto normalizeFn) -> ParseResult {
    Attribute rawAttr = attrs.get(sourceName);
    if (!rawAttr) {
      return success();
    }
    auto strAttr = dyn_cast<StringAttr>(rawAttr);
    if (!strAttr) {
      return parser.emitError(parser.getCurrentLocation())
             << sourceName << " must be a string literal";
    }
    auto normalized = normalizeFn(strAttr.getValue());
    if (!normalized) {
      return parser.emitError(parser.getCurrentLocation())
             << sourceName << " has unsupported value '" << strAttr.getValue()
             << "'";
    }
    attrs.erase(sourceName);
    attrs.set(canonicalName, parser.getBuilder().getStringAttr(*normalized));
    return success();
  };
  if (failed(normalizeNamedStringAttr("round_mode", "rnd",
                                      normalizeRoundModeToken)) ||
      failed(normalizeNamedStringAttr("rnd", "rnd", normalizeRoundModeToken)) ||
      failed(normalizeNamedStringAttr("sat", "sat", normalizeSaturationToken)) ||
      failed(normalizeNamedStringAttr("part", "part", normalizeVcvtPartToken))) {
    return failure();
  }

  result.addAttributes(attrs);
  if (parser.resolveOperand(input, inputType, result.operands) ||
      parser.resolveOperand(mask, maskType, result.operands)) {
    return failure();
  }
  result.addTypes(resultType);
  return success();
}

void VcvtOp::print(OpAsmPrinter &printer) {
  printer << ' ' << getInput() << ", " << getMask();
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : " << getInput().getType() << ", " << getMask().getType()
          << " -> " << getResult().getType();
}

LogicalResult VcvtOp::verify() {
  auto inputType = dyn_cast<VRegType>(getInput().getType());
  auto resultType = dyn_cast<VRegType>(getResult().getType());
  if (!inputType || !resultType) {
    return emitOpError("input and result must be !pto.vreg<...>");
  }
  if (failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type"))) {
    return failure();
  }

  VcvtElemKind inputElemKind = classifyVcvtElemType(inputType.getElementType());
  VcvtElemKind resultElemKind = classifyVcvtElemType(resultType.getElementType());
  auto contract = lookupVcvtContract(inputElemKind, resultElemKind);
  if (!contract) {
    return emitOpError("unsupported vcvt source/result element type pair");
  }

  auto inputElemBits = getVcvtElemBitWidth(inputElemKind);
  auto resultElemBits = getVcvtElemBitWidth(resultElemKind);
  if (!inputElemBits || !resultElemBits) {
    return emitOpError("could not determine vcvt element bit width");
  }
  unsigned maskBitWidth = std::min(*inputElemBits, 32u);
  StringRef expectedMaskGranularity = maskBitWidth == 8    ? "b8"
                                      : maskBitWidth == 16 ? "b16"
                                      : maskBitWidth == 32 ? "b32"
                                                           : "";
  if (expectedMaskGranularity.empty()) {
    return emitOpError("could not determine vcvt mask granularity");
  }
  if (failed(verifyMaskTypeWithGranularityLike(
          *this, getMask().getType(), "mask type", expectedMaskGranularity))) {
    return failure();
  }
  if (inputType.getElementCount() * static_cast<int64_t>(*inputElemBits) !=
      resultType.getElementCount() * static_cast<int64_t>(*resultElemBits)) {
    return emitOpError("requires source and result vectors to carry the same "
                       "total number of bits");
  }

  if (getRndAttr()) {
    StringRef roundMode = *getRnd();
    auto normalizedRoundMode = normalizeRoundModeToken(roundMode);
    if (!normalizedRoundMode) {
      return emitOpError("rnd must be one of R/A/F/C/Z/O/H");
    }
    if (!isValidVcvtRoundModeForContract(*normalizedRoundMode, *contract)) {
      return emitOpError("rnd attr is not valid for this vcvt type pair");
    }
  }
  if (static_cast<bool>(getRndAttr()) != contract->requiresRnd) {
    return contract->requiresRnd ? emitOpError("requires rnd attr for this vcvt type pair")
                                 : emitOpError("rnd attr is not valid for this vcvt type pair");
  }

  if (getSatAttr()) {
    StringRef sat = *getSat();
    if (!normalizeSaturationToken(sat)) {
      return emitOpError("sat must be SAT or NOSAT");
    }
  }
  if (static_cast<bool>(getSatAttr()) != contract->requiresSat) {
    return contract->requiresSat ? emitOpError("requires sat attr for this vcvt type pair")
                                 : emitOpError("sat attr is not valid for this vcvt type pair");
  }

  if (getPartAttr()) {
    StringRef part = *getPart();
    auto normalizedPart = normalizeVcvtPartToken(part);
    if (!normalizedPart) {
      return emitOpError("part must be one of EVEN/ODD/P0/P1/P2/P3");
    }
    std::optional<VcvtPartFamily> partFamily = contract->partFamily;
    if (!partFamily) {
      partFamily = classifyVcvtPartFamily(*inputElemBits, *resultElemBits);
    }
    if (!partFamily) {
      return emitOpError("part attr is not supported for this vcvt width relation");
    }
    if (!isValidVcvtPartForFamily(*normalizedPart, *partFamily)) {
      switch (*partFamily) {
      case VcvtPartFamily::EvenOdd:
        return emitOpError("part must be EVEN or ODD for 8/16 and 16/32 vcvt forms");
      case VcvtPartFamily::Packed4:
        return emitOpError(
            "part must be P0, P1, P2, or P3 for packed vcvt forms");
      }
    }
  }
  if (static_cast<bool>(getPartAttr()) != contract->requiresPart) {
    return contract->requiresPart ? emitOpError("requires part attr for this vcvt type pair")
                                  : emitOpError("part attr is not valid for this vcvt type pair");
  }

  return success();
}

LogicalResult VbitcastOp::verify() {
  auto inputType = dyn_cast<VRegType>(getInput().getType());
  auto resultType = dyn_cast<VRegType>(getResult().getType());
  if (!inputType || !resultType) {
    return emitOpError("input and result must be !pto.vreg<...>");
  }

  auto getStorageBits = [](VRegType type) -> std::optional<int64_t> {
    Type elementType = type.getElementType();
    if (auto intType = dyn_cast<IntegerType>(elementType)) {
      return type.getElementCount() * static_cast<int64_t>(intType.getWidth());
    }
    if (auto floatType = dyn_cast<FloatType>(elementType)) {
      return type.getElementCount() *
             static_cast<int64_t>(floatType.getWidth());
}
    // Packed PTO element types (f8/hif8/f4x2/bf16x2/...) have a known storage
    // width even though they are not IntegerType/FloatType.
    unsigned packedBits = pto::getPTOStorageElemBitWidth(elementType);
    if (packedBits != 0) {
      return type.getElementCount() * static_cast<int64_t>(packedBits);
}
    return std::nullopt;
  };

  auto inputBits = getStorageBits(inputType);
  auto resultBits = getStorageBits(resultType);
  if (!inputBits || !resultBits) {
    return emitOpError("requires integer or floating-point vreg element type");
  }
  if (*inputBits != *resultBits) {
    return emitOpError("requires source and result vectors to carry the same "
                       "total number of bits");
  }

  return success();
}

LogicalResult PdintlvB8Op::verify() {
  if (failed(verifyMaskTypeWithGranularityLike(*this, getLhs().getType(),
                                               "lhs type", "b8")) ||
      failed(verifyMaskTypeWithGranularityLike(*this, getRhs().getType(),
                                               "rhs type", "b8")) ||
      failed(verifyMaskTypeWithGranularityLike(*this, getLow().getType(),
                                               "low type", "b8")) ||
      failed(verifyMaskTypeWithGranularityLike(*this, getHigh().getType(),
                                               "high type", "b8"))) {
    return failure();
  }
  return success();
}

LogicalResult PdintlvB16Op::verify() {
  if (failed(verifyMaskTypeWithGranularityLike(*this, getLhs().getType(),
                                               "lhs type", "b16")) ||
      failed(verifyMaskTypeWithGranularityLike(*this, getRhs().getType(),
                                               "rhs type", "b16")) ||
      failed(verifyMaskTypeWithGranularityLike(*this, getLow().getType(),
                                               "low type", "b16")) ||
      failed(verifyMaskTypeWithGranularityLike(*this, getHigh().getType(),
                                               "high type", "b16"))) {
    return failure();
  }
  return success();
}

LogicalResult PdintlvB32Op::verify() {
  if (failed(verifyMaskTypeWithGranularityLike(*this, getLhs().getType(),
                                               "lhs type", "b32")) ||
      failed(verifyMaskTypeWithGranularityLike(*this, getRhs().getType(),
                                               "rhs type", "b32")) ||
      failed(verifyMaskTypeWithGranularityLike(*this, getLow().getType(),
                                               "low type", "b32")) ||
      failed(verifyMaskTypeWithGranularityLike(*this, getHigh().getType(),
                                               "high type", "b32"))) {
    return failure();
  }
  return success();
}

LogicalResult PintlvB8Op::verify() {
  if (failed(verifyMaskTypeWithGranularityLike(*this, getLhs().getType(),
                                               "lhs type", "b8")) ||
      failed(verifyMaskTypeWithGranularityLike(*this, getRhs().getType(),
                                               "rhs type", "b8")) ||
      failed(verifyMaskTypeWithGranularityLike(*this, getLow().getType(),
                                               "low type", "b8")) ||
      failed(verifyMaskTypeWithGranularityLike(*this, getHigh().getType(),
                                               "high type", "b8"))) {
    return failure();
  }
  return success();
}

LogicalResult PintlvB16Op::verify() {
  if (failed(verifyMaskTypeWithGranularityLike(*this, getLhs().getType(),
                                               "lhs type", "b16")) ||
      failed(verifyMaskTypeWithGranularityLike(*this, getRhs().getType(),
                                               "rhs type", "b16")) ||
      failed(verifyMaskTypeWithGranularityLike(*this, getLow().getType(),
                                               "low type", "b16")) ||
      failed(verifyMaskTypeWithGranularityLike(*this, getHigh().getType(),
                                               "high type", "b16"))) {
    return failure();
  }
  return success();
}

LogicalResult PintlvB32Op::verify() {
  if (failed(verifyMaskTypeWithGranularityLike(*this, getLhs().getType(),
                                               "lhs type", "b32")) ||
      failed(verifyMaskTypeWithGranularityLike(*this, getRhs().getType(),
                                               "rhs type", "b32")) ||
      failed(verifyMaskTypeWithGranularityLike(*this, getLow().getType(),
                                               "low type", "b32")) ||
      failed(verifyMaskTypeWithGranularityLike(*this, getHigh().getType(),
                                               "high type", "b32"))) {
    return failure();
  }
  return success();
}

LogicalResult VintlvOp::verify() { return verifyPairVecResults(*this); }
LogicalResult VdintlvOp::verify() { return verifyPairVecResults(*this); }
LogicalResult Vintlvv2Op::verify() { return verifyPartVecOp(*this); }
LogicalResult Vdintlvv2Op::verify() { return verifyPartVecOp(*this); }

LogicalResult VmullOp::verify() {
  if (failed(verifyPairVecResults(*this)) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type"))) {
    return failure();
  }
  auto lhsType = cast<VRegType>(getLhs().getType());
  auto lhsElemType = dyn_cast<IntegerType>(lhsType.getElementType());
  if (!lhsElemType) {
    return emitOpError("requires integer vector element type");
  }
  if (lhsElemType.getWidth() != mlir::pto::kValue32) {
    return emitOpError("currently requires 32-bit integer vector elements");
  }
  return success();
}

LogicalResult VmulaOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getAcc().getType(), "acc type")) ||
      failed(verifyVRegTypeLike(*this, getLhs().getType(), "lhs type")) ||
      failed(verifyVRegTypeLike(*this, getRhs().getType(), "rhs type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result type"))) {
    return failure();
  }
  if (getAcc().getType() != getLhs().getType() ||
      getAcc().getType() != getRhs().getType() ||
      getAcc().getType() != getResult().getType()) {
    return emitOpError("requires acc, lhs, rhs, and result to share one vector type");
  }
  return success();
}

template <typename BinaryVecNoMaskOp>
static LogicalResult verifyBinaryVecNoMaskOp(BinaryVecNoMaskOp op) {
  if (failed(verifyVRegTypeLike(op, op.getLhs().getType(), "lhs type")) ||
      failed(verifyVRegTypeLike(op, op.getRhs().getType(), "rhs type")) ||
      failed(verifyVRegTypeLike(op, op.getResult().getType(), "result type"))) {
    return failure();
  }
  if (op.getLhs().getType() != op.getRhs().getType() ||
      op.getLhs().getType() != op.getResult().getType()) {
    return op.emitOpError("requires lhs, rhs, and result to share one vector type");
  }
  return success();
}

template <typename BinaryVecNoMaskOp>
static LogicalResult verifyFloatBinaryVecNoMaskOp(BinaryVecNoMaskOp op) {
  if (failed(verifyBinaryVecNoMaskOp(op))) {
    return failure();
  }
  auto lhsType = cast<VRegType>(op.getLhs().getType());
  Type elemType = lhsType.getElementType();
  if (!elemType.isF16() && !elemType.isF32()) {
    return op.emitOpError("requires f16 or f32 vector element type");
  }
  return success();
}

template <typename BinaryVecMaskOp>
static LogicalResult verifyFloatBinaryVecMaskOp(BinaryVecMaskOp op) {
  if (failed(verifyVRegTypeLike(op, op.getLhs().getType(), "lhs type")) ||
      failed(verifyVRegTypeLike(op, op.getRhs().getType(), "rhs type")) ||
      failed(verifyMaskTypeLike(op, op.getMask().getType(), "mask type")) ||
      failed(verifyVRegTypeLike(op, op.getResult().getType(), "result type"))) {
    return failure();
  }
  if (op.getLhs().getType() != op.getRhs().getType() ||
      op.getLhs().getType() != op.getResult().getType()) {
    return op.emitOpError("requires lhs, rhs, and result to share one vector type");
  }
  auto lhsType = cast<VRegType>(op.getLhs().getType());
  Type elemType = lhsType.getElementType();
  if (!elemType.isF16() && !elemType.isF32()) {
    return op.emitOpError("requires f16 or f32 vector element type");
  }
  return success();
}

LogicalResult VpreluOp::verify() { return verifyFloatBinaryVecMaskOp(*this); }
LogicalResult VexpdifOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getInput().getType(), "input type")) ||
      failed(verifyVRegTypeLike(*this, getMax().getType(), "max type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result type"))) {
    return failure();
  }

  auto inputType = cast<VRegType>(getInput().getType());
  auto maxType = cast<VRegType>(getMax().getType());
  auto resultType = cast<VRegType>(getResult().getType());
  if (inputType != maxType) {
    return emitOpError("requires input and max to share one vector type");
  }

  Type inputElemType = inputType.getElementType();
  if (!inputElemType.isF16() && !inputElemType.isF32()) {
    return emitOpError("requires f16 or f32 input vector element type");
  }
  auto expectedGranularity = getVdupMaskGranularity(inputElemType);
  if (!expectedGranularity) {
    return emitOpError("requires input element type with supported predicate granularity");
  }
  if (failed(verifyMaskTypeWithGranularityLike(*this, getMask().getType(),
                                               "mask type",
                                               *expectedGranularity))) {
    return failure();
  }
  if (!resultType.getElementType().isF32()) {
    return emitOpError("requires f32 result vector element type");
  }

  auto inputBits = getVRegStorageBitWidth(inputType);
  auto resultBits = getVRegStorageBitWidth(resultType);
  if (!inputBits || !resultBits || *inputBits != *resultBits) {
    return emitOpError(
        "requires source and result to preserve total vector storage width");
  }

  StringRef part = getPart();
  if (part != "EVEN" && part != "ODD") {
    return emitOpError("part must be EVEN or ODD");
  }
  return success();
}

LogicalResult VaxpyOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getSrc0().getType(), "src0 type")) ||
      failed(verifyVRegTypeLike(*this, getSrc1().getType(), "src1 type")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type"))) {
    return failure();
  }
  auto src0Type = cast<VRegType>(getSrc0().getType());
  auto src1Type = cast<VRegType>(getSrc1().getType());
  auto resultType = cast<VRegType>(getResult().getType());
  if (src0Type != src1Type || src0Type != resultType) {
    return emitOpError("requires src0, src1, and result to share one vector type");
  }
  Type elemType = src0Type.getElementType();
  if (!elemType.isF16() && !elemType.isF32()) {
    return emitOpError("requires f16 or f32 vector element type");
  }
  auto expectedGranularity = getVdupMaskGranularity(elemType);
  if (!expectedGranularity) {
    return emitOpError("requires element type with supported predicate granularity");
  }
  if (failed(verifyMaskTypeWithGranularityLike(*this, getMask().getType(),
                                               "mask type",
                                               *expectedGranularity))) {
    return failure();
  }
  if (getAlpha().getType() != elemType) {
    return emitOpError("requires alpha type to match vector element type");
  }
  return success();
}

template <typename ConvOp>
static LogicalResult verifyFusedConvVecOp(ConvOp op) {
  if (failed(verifyVRegTypeLike(op, op.getLhs().getType(), "lhs type")) ||
      failed(verifyVRegTypeLike(op, op.getRhs().getType(), "rhs type")) ||
      failed(verifyVRegTypeLike(op, op.getResult().getType(), "result type"))) {
    return failure();
  }
  auto lhsType = cast<VRegType>(op.getLhs().getType());
  auto rhsType = cast<VRegType>(op.getRhs().getType());
  auto resultType = cast<VRegType>(op.getResult().getType());
  if (lhsType != rhsType) {
    return op.emitOpError("requires lhs and rhs to share one vector type");
  }
  if (!isIntegerOrFloatLike(lhsType.getElementType()) ||
      !isIntegerOrFloatLike(resultType.getElementType())) {
    return op.emitOpError(
        "requires integer or floating-point vector element types");
  }
  auto lhsBits = getVRegStorageBitWidth(lhsType);
  auto resultBits = getVRegStorageBitWidth(resultType);
  if (!lhsBits || !resultBits || *lhsBits != *resultBits) {
    return op.emitOpError(
        "requires source and result to preserve total vector storage width");
  }
  return success();
}

LogicalResult VaddreluconvOp::verify() {
  return verifyFusedConvVecOp(*this);
}
LogicalResult VmulconvOp::verify() { return verifyFusedConvVecOp(*this); }

void Vldsx2Op::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult Vldsx2Op::verify() {
  if (!isBufferLike(getSource().getType())) {
    return emitOpError("requires a pointer-like source");
  }
  if (classifyMemoryRole(getSource().getType()) == MemoryRole::GM) {
    return emitOpError("requires a UB-backed source");
  }
  if (!getOffset().getType().isIndex()) {
    return emitOpError("requires index offset");
  }
  if (failed(verifyVRegTypeLike(*this, getLow().getType(), "low result type")) ||
      failed(verifyVRegTypeLike(*this, getHigh().getType(), "high result type"))) {
    return failure();
  }
  if (getLow().getType() != getHigh().getType()) {
    return emitOpError("requires low/high results to share one vector type");
  }
  Type elementType = cast<VRegType>(getLow().getType()).getElementType();
  if (!isSupportedVldx2DistToken(getDist(), elementType)) {
    return emitOpError("requires a supported x2 load distribution token");
  }
  if (getUpdatedBase() &&
      getUpdatedBase().getType() != getSource().getType()) {
    return emitOpError("requires updated base result to match base type");
  }
  return success();
}

void VstsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

template <typename StoreOp>
static LogicalResult verifyVstsCommon(StoreOp op) {
  if (failed(verifyVRegTypeLike(op, op.getValue().getType(), "value type"))) {
    return failure();
  }

  if (!isBufferLike(op.getDestination().getType())) {
    return op.emitOpError("requires a pointer-like destination");
  }

  MemoryRole destinationRole = classifyMemoryRole(op.getDestination().getType());
  if (destinationRole == MemoryRole::GM) {
    return op.emitOpError("requires a UB-backed destination");
  }

  if (std::optional<StringRef> dist = op.getDist();
      dist && !isSupportedVstsDistToken(*dist)) {
    return op.emitOpError("requires a supported store distribution token");
  }
  if (std::optional<StringRef> dist = op.getDist()) {
    if (std::optional<StringRef> granularity =
            getVstsMaskGranularityOverride(*dist)) {
      if (failed(verifyMaskTypeWithGranularityLike(op, op.getMask().getType(),
                                                   "mask type", *granularity))) {
        return failure();
      }
    } else if (failed(verifyMaskTypeLike(op, op.getMask().getType(),
                                         "mask type"))) {
      return failure();
    }
  } else if (failed(verifyMaskTypeLike(op, op.getMask().getType(),
                                       "mask type"))) {
    return failure();
  }

  return success();
}

LogicalResult VstsOp::verify() {
  if (failed(verifyVstsCommon(*this))) {
    return failure();
  }
  if (getUpdatedBase() &&
      getUpdatedBase().getType() != getDestination().getType()) {
    return emitOpError("requires updated base result to match base type");
  }
  return success();
}
void Vstsx2Op::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getLowMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getHighMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult Vstsx2Op::verify() {
  if (failed(verifyVRegTypeLike(*this, getLow().getType(), "low value type")) ||
      failed(verifyVRegTypeLike(*this, getHigh().getType(), "high value type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type"))) {
    return failure();
  }
  if (getLow().getType() != getHigh().getType()) {
    return emitOpError("requires low/high values to share one vector type");
  }
  if (!isBufferLike(getDestination().getType())) {
    return emitOpError("requires a pointer-like destination");
  }
  if (classifyMemoryRole(getDestination().getType()) == MemoryRole::GM) {
    return emitOpError("requires a UB-backed destination");
  }
  if (!getOffset().getType().isIndex()) {
    return emitOpError("requires index offset");
  }
  if (!isSupportedVstsx2DistToken(getDist())) {
    return emitOpError("requires a supported x2 store distribution token");
  }
  return success();
}

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

void VsldbOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VsldbOp::verify() {
  if (!isBufferLike(getSource().getType())) {
    return emitOpError("requires a pointer-like source");
  }
  if (classifyMemoryRole(getSource().getType()) == MemoryRole::GM) {
    return emitOpError("requires a UB-backed source");
  }
  if (failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result type"))) {
    return failure();
  }
  if (!getBlockStride().getType().isSignlessInteger(mlir::pto::kValue16)) {
    return emitOpError("requires block_stride to be i16");
  }
  if (!getRepeatStride().getType().isSignlessInteger(mlir::pto::kValue16)) {
    return emitOpError("requires repeat_stride to be i16");
  }
  if (getUpdatedBase() &&
      getUpdatedBase().getType() != getSource().getType()) {
    return emitOpError("requires updated base result to match base type");
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
  if (failed(verifyMaskTypeLike(*this, getValue().getType(), "value type"))) {
    return failure();
  }
  if (!isBufferLike(getDestination().getType())) {
    return emitOpError("requires a pointer-like destination");
  }
  if (classifyMemoryRole(getDestination().getType()) == MemoryRole::GM) {
    return emitOpError("requires a UB-backed destination");
  }
  if (!matchPattern(getOffset(), m_Constant())) {
    return emitOpError("requires offset to be a constant index immediate");
  }
  if (!isSupportedPredicateStoreDist(getDist())) {
    return emitOpError("requires predicate store dist to be NORM or PK");
  }
  if (getUpdatedBase() &&
      getUpdatedBase().getType() != getDestination().getType()) {
    return emitOpError("requires updated base result to match base type");
  }
  return success();
}

LogicalResult PstsOp::verify() {
  if (failed(verifyMaskTypeLike(*this, getValue().getType(), "value type"))) {
    return failure();
  }
  if (!isBufferLike(getDestination().getType())) {
    return emitOpError("requires a pointer-like destination");
  }
  MemoryRole destinationRole = classifyMemoryRole(getDestination().getType());
  if (destinationRole == MemoryRole::GM) {
    return emitOpError("requires a UB-backed destination");
  }
  if (!getOffset().getType().isIndex()) {
    return emitOpError("requires index offset");
  }
  if (!isSupportedPredicateStoreDist(getDist())) {
    return emitOpError("requires predicate store dist to be NORM or PK");
  }
  if (getUpdatedBase() &&
      getUpdatedBase().getType() != getDestination().getType()) {
    return emitOpError("requires updated base result to match base type");
  }
  return success();
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

void VstarOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VstarOp::verify() {
  if (failed(verifyStoreAlignChain(getValue(), *this, "value type"))) {
    return failure();
  }
  if (!isBufferLike(getDestination().getType())) {
    return emitOpError("requires a pointer-like destination");
  }
  if (classifyMemoryRole(getDestination().getType()) == MemoryRole::GM) {
    return emitOpError("requires a UB-backed destination");
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
  if (failed(verifyStoreAlignChain(getAlignIn(), *this, "align_in type")) ||
      failed(verifyVRegTypeLike(*this, getValue().getType(), "value type")) ||
      failed(verifyAlignTypeLike(*this, getAlignOut().getType(), "align_out type"))) {
    return failure();
  }
  if (!isBufferLike(getBase().getType())) {
    return emitOpError("requires a pointer-like base");
  }
  if (classifyMemoryRole(getBase().getType()) == MemoryRole::GM) {
    return emitOpError("requires a UB-backed base");
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
  if (failed(verifyStoreAlignChain(getAlignIn(), *this, "align_in type")) ||
      failed(verifyVRegTypeLike(*this, getValue().getType(), "value type")) ||
      failed(verifyAlignTypeLike(*this, getAlignOut().getType(), "align_out type"))) {
    return failure();
  }
  if (!isBufferLike(getBase().getType())) {
    return emitOpError("requires a pointer-like base");
  }
  if (classifyMemoryRole(getBase().getType()) == MemoryRole::GM) {
    return emitOpError("requires a UB-backed base");
  }
  if (!isSupportedPostMode(getMode())) {
    return emitOpError("requires mode to be POST_UPDATE or NO_POST_UPDATE");
  }
  return success();
}

void CopyUbufToGmOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult CopyUbufToGmOp::verify() {
  return verifyCopyUbufToGmOp(*this, false);
}

void MteUbGmOp::build(OpBuilder &builder, OperationState &state, Value source,
                       Value destination, Value lenBurst,
                       pto::DmaLoopConfig nburst, Value l2CacheCtl,
                       llvm::ArrayRef<pto::DmaLoopConfig> loops) {
  state.addOperands({source, destination, lenBurst, nburst.count,
                     nburst.srcStride, nburst.dstStride});
  if (l2CacheCtl) {
    state.addOperands(l2CacheCtl);
  }
  for (const pto::DmaLoopConfig &loop : loops) {
    state.addOperands(loop.count);
  }
  for (const pto::DmaLoopConfig &loop : loops) {
    state.addOperands(loop.srcStride);
  }
  for (const pto::DmaLoopConfig &loop : loops) {
    state.addOperands(loop.dstStride);
  }

  state.addAttribute(
      getOperandSegmentSizeAttr(),
      builder.getDenseI32ArrayAttr(
          {1, 1, 1, 1, 1, 1, l2CacheCtl ? 1 : 0,
           static_cast<int32_t>(loops.size()),
           static_cast<int32_t>(loops.size()),
           static_cast<int32_t>(loops.size())}));
}

void MteUbGmOp::build(OpBuilder &builder, OperationState &state, Value source,
                       Value destination, Value lenBurst,
                       pto::DmaLoopConfig nburst, Value l2CacheCtl,
                       std::optional<pto::DmaLoopConfig> loop1,
                       std::optional<pto::DmaLoopConfig> loop2) {
  SmallVector<pto::DmaLoopConfig> loops;
  if (loop1) {
    loops.push_back(*loop1);
  }
  if (loop2) {
    loops.push_back(*loop2);
  }
  build(builder, state, source, destination, lenBurst, nburst, l2CacheCtl,
        loops);
}

ParseResult MteUbGmOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand source, destination, lenBurst, l2CacheCtl;
  bool hasL2CacheCtl = false;
  SmallVector<OpAsmParser::UnresolvedOperand> nburstOperands;
  SmallVector<OpAsmParser::UnresolvedOperand> loopCountOperands;
  SmallVector<OpAsmParser::UnresolvedOperand> loopSrcStrideOperands;
  SmallVector<OpAsmParser::UnresolvedOperand> loopDstStrideOperands;
  if (parseRequiredOperandWithComma(parser, source) ||
      parseRequiredOperandWithComma(parser, destination) ||
      parser.parseOperand(lenBurst) ||
      parseDmaTripleGroup(parser, "nburst", nburstOperands)) {
    return failure();
  }
  if (succeeded(parser.parseOptionalKeyword("l2_cache_ctl"))) {
    hasL2CacheCtl = true;
    if (parser.parseLParen() || parser.parseOperand(l2CacheCtl) ||
        parser.parseRParen()) {
      return failure();
    }
  }
  while (true) {
    StringRef parsedKeyword;
    SmallVector<OpAsmParser::UnresolvedOperand, mlir::pto::kValue3> loopGroupOperands;
    if (parseOptionalDmaTripleGroupAlias(parser, {"loop", "loop1", "loop2"},
                                         parsedKeyword, loopGroupOperands)) {
      return failure();
    }
    if (parsedKeyword.empty()) {
      break;
    }
    loopCountOperands.push_back(loopGroupOperands[0]);
    loopSrcStrideOperands.push_back(loopGroupOperands[1]);
    loopDstStrideOperands.push_back(loopGroupOperands[mlir::pto::kValue2]);
  }

  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon()) {
    return failure();
  }

  Type sourceType, destinationType, lenBurstType, l2CacheCtlType;
  SmallVector<Type> nburstTypes, loopCountTypes, loopSrcStrideTypes,
      loopDstStrideTypes;
  if (parser.parseType(sourceType) || parser.parseComma() ||
      parser.parseType(destinationType) || parser.parseComma() ||
      parser.parseType(lenBurstType) || parser.parseComma() ||
      parseDmaTripleTypes(parser, nburstTypes)) {
    return failure();
  }
  if (hasL2CacheCtl) {
    if (parser.parseComma() || parser.parseType(l2CacheCtlType)) {
      return failure();
    }
  }
  while (succeeded(parser.parseOptionalComma())) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword)) {
      return failure();
    }
    if (isDmaLoopKeyword(keyword)) {
      SmallVector<Type> loopGroupTypes;
      if (parseDmaTripleTypes(parser, loopGroupTypes)) {
        return failure();
      }
      loopCountTypes.push_back(loopGroupTypes[0]);
      loopSrcStrideTypes.push_back(loopGroupTypes[1]);
      loopDstStrideTypes.push_back(loopGroupTypes[mlir::pto::kValue2]);
      continue;
    }
    return parser.emitError(parser.getCurrentLocation(),
                            "expected 'loop'");
  }

  int32_t loopGroupCount = static_cast<int32_t>(loopCountOperands.size());
  if (loopCountOperands.size() != loopSrcStrideOperands.size() ||
      loopCountOperands.size() != loopDstStrideOperands.size() ||
      loopCountTypes.size() != loopSrcStrideTypes.size() ||
      loopCountTypes.size() != loopDstStrideTypes.size()) {
    return parser.emitError(parser.getCurrentLocation(),
                            "requires each loop group to provide count, src stride, and dst stride");
  }
  if (loopCountOperands.size() != loopCountTypes.size()) {
    return parser.emitError(parser.getCurrentLocation(),
                            "requires loop operand and type groups to match");
  }

  auto &segments =
      result.getOrAddProperties<MteUbGmOp::Properties>().operandSegmentSizes;
  llvm::copy(ArrayRef<int32_t>{1, 1, 1, 1, 1, 1,
                               hasL2CacheCtl ? 1 : 0,
                               loopGroupCount, loopGroupCount, loopGroupCount},
             segments.begin());

  if (parser.resolveOperand(source, sourceType, result.operands) ||
      parser.resolveOperand(destination, destinationType, result.operands) ||
      parser.resolveOperand(lenBurst, lenBurstType, result.operands) ||
      parser.resolveOperands(nburstOperands, nburstTypes, parser.getCurrentLocation(),
                             result.operands)) {
    return failure();
  }
  if (hasL2CacheCtl &&
      parser.resolveOperand(l2CacheCtl, l2CacheCtlType, result.operands)) {
    return failure();
  }
  if (parser.resolveOperands(loopCountOperands, loopCountTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(loopSrcStrideOperands, loopSrcStrideTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(loopDstStrideOperands, loopDstStrideTypes,
                             parser.getCurrentLocation(),
                             result.operands)) {
    return failure();
  }
  return success();
}

void MteUbGmOp::print(OpAsmPrinter &printer) {
  printer << " " << getSource() << ", " << getDestination() << ", "
          << getLenBurst();
  printDmaTripleGroup(printer, "nburst", getNBurst(), getNburstSrcStride(),
                      getNburstDstStride());
  if (Value l2CacheCtl = getL2CacheCtl()) {
    printer << " l2_cache_ctl(" << l2CacheCtl << ")";
  }
  for (auto [count, srcStride, dstStride] :
       llvm::zip(getLoopCounts(), getLoopSrcStrides(), getLoopDstStrides())) {
    printDmaTripleGroup(printer, "loop", count, srcStride, dstStride);
  }
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : " << getSource().getType() << ", " << getDestination().getType()
          << ", " << getLenBurst().getType() << ", " << getNBurst().getType()
          << ", " << getNburstSrcStride().getType()
          << ", "
          << getNburstDstStride().getType();
  if (Value l2CacheCtl = getL2CacheCtl()) {
    printer << ", " << l2CacheCtl.getType();
  }
  for (auto [count, srcStride, dstStride] :
       llvm::zip(getLoopCounts(), getLoopSrcStrides(), getLoopDstStrides())) {
    printDmaTripleTypes(printer, "loop", count.getType(), srcStride.getType(),
                        dstStride.getType());
  }
}

void MteUbGmOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult MteUbGmOp::verify() {
  if (!isBufferLike(getSource().getType()) ||
      !isBufferLike(getDestination().getType())) {
    return emitOpError(
        "requires typed !pto.ptr or memref source and destination");
  }
  if (classifyMemoryRole(getSource().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getDestination().getType()) != MemoryRole::GM) {
    return emitOpError("requires UB source and GM destination");
  }
  int64_t sourceElemBytes = getBufferElementByteSize(getSource().getType());
  int64_t destinationElemBytes =
      getBufferElementByteSize(getDestination().getType());
  if (sourceElemBytes <= 0 || destinationElemBytes <= 0) {
    return emitOpError(
        "requires copy source and destination element types with known byte width");
  }
  if (sourceElemBytes != destinationElemBytes) {
    return emitOpError(
        "requires source and destination element byte widths to match");
  }
  if (Value l2CacheCtlValue = getL2CacheCtl()) {
    APInt l2CacheCtl;
    if (matchPattern(l2CacheCtlValue, m_ConstantInt(&l2CacheCtl)) &&
        (l2CacheCtl.isNegative() || l2CacheCtl.ugt(mlir::pto::kValue15))) {
      return emitOpError(
          "requires constant l2_cache_ctl to fit in range [0, 15]");
    }
  }
  return verifyDmaLoadStoreLoopGroups(
      getOperation(), getLoopCounts(), getLoopSrcStrides(),
      getLoopDstStrides());
}

void MteGmL1Op::build(OpBuilder &builder, OperationState &state, Value source,
                       Value destination, Value lenBurst,
                       pto::DmaLoopConfig nburst,
                       llvm::ArrayRef<pto::DmaLoopConfig> loops) {
  state.addOperands(
      {source, destination, lenBurst, nburst.count, nburst.srcStride,
       nburst.dstStride});
  for (const pto::DmaLoopConfig &loop : loops) {
    state.addOperands(loop.count);
  }
  for (const pto::DmaLoopConfig &loop : loops) {
    state.addOperands(loop.srcStride);
  }
  for (const pto::DmaLoopConfig &loop : loops) {
    state.addOperands(loop.dstStride);
  }

  state.addAttribute(
      getOperandSegmentSizeAttr(),
      builder.getDenseI32ArrayAttr(
          {1, 1, 1, 1, 1, 1,
           static_cast<int32_t>(loops.size()),
           static_cast<int32_t>(loops.size()),
           static_cast<int32_t>(loops.size())}));
}

void MteGmL1Op::build(OpBuilder &builder, OperationState &state, Value source,
                       Value destination, Value lenBurst,
                       pto::DmaLoopConfig nburst,
                       std::optional<pto::DmaLoopConfig> loop1,
                       std::optional<pto::DmaLoopConfig> loop2) {
  SmallVector<pto::DmaLoopConfig> loops;
  if (loop1) {
    loops.push_back(*loop1);
  }
  if (loop2) {
    loops.push_back(*loop2);
  }
  build(builder, state, source, destination, lenBurst, nburst, loops);
}

void MteL1UbOp::build(OpBuilder &builder, OperationState &state, Value source,
                        Value destination, Value lenBurst,
                        pto::DmaLoopConfig nburst,
                        llvm::ArrayRef<pto::DmaLoopConfig> loops) {
  state.addOperands(
      {source, destination, lenBurst, nburst.count, nburst.srcStride,
       nburst.dstStride});
  for (const pto::DmaLoopConfig &loop : loops) {
    state.addOperands(loop.count);
  }
  for (const pto::DmaLoopConfig &loop : loops) {
    state.addOperands(loop.srcStride);
  }
  for (const pto::DmaLoopConfig &loop : loops) {
    state.addOperands(loop.dstStride);
  }

  state.addAttribute(
      getOperandSegmentSizeAttr(),
      builder.getDenseI32ArrayAttr(
          {1, 1, 1, 1, 1, 1,
           static_cast<int32_t>(loops.size()),
           static_cast<int32_t>(loops.size()),
           static_cast<int32_t>(loops.size())}));
}

void MteL1UbOp::build(OpBuilder &builder, OperationState &state, Value source,
                        Value destination, Value lenBurst,
                        pto::DmaLoopConfig nburst,
                        std::optional<pto::DmaLoopConfig> loop1,
                        std::optional<pto::DmaLoopConfig> loop2) {
  SmallVector<pto::DmaLoopConfig> loops;
  if (loop1) {
    loops.push_back(*loop1);
  }
  if (loop2) {
    loops.push_back(*loop2);
  }
  build(builder, state, source, destination, lenBurst, nburst, loops);
}

void MteGmL1FracOp::build(OpBuilder &builder, OperationState &state,
                           Value source, Value destination,
                           pto::CubeLoadFracMode mode,
                           pto::CubeLoadFracShapeConfig shape,
                           pto::CubeLoadFracSrcLayoutConfig srcLayout,
                           pto::CubeLoadFracDstGroupConfig dstGroup,
                           pto::CubeLoadFracCtrlConfig ctrl) {
  state.addOperands({source, destination, shape.nValue, shape.dValue,
                     srcLayout.srcInnerStride});
  state.addOperands({dstGroup.groupCount, dstGroup.dstLoop2Stride,
                     dstGroup.dstLoop3Stride, dstGroup.dstLoop4Stride,
                     ctrl.l2CacheCtrl, ctrl.smallc0En});
  bool hasSrcOuterStride = srcLayout.srcOuterStride.has_value();
  if (hasSrcOuterStride) {
    state.addOperands(*srcLayout.srcOuterStride);
  }

  state.addAttribute(getModeAttrName(state.name),
                     CubeLoadFracModeAttr::get(builder.getContext(), mode));
}

ParseResult MteGmL1Op::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand source, destination, lenBurst;
  SmallVector<OpAsmParser::UnresolvedOperand> nburstOperands;
  SmallVector<OpAsmParser::UnresolvedOperand> loopCountOperands;
  SmallVector<OpAsmParser::UnresolvedOperand> loopSrcStrideOperands;
  SmallVector<OpAsmParser::UnresolvedOperand> loopDstStrideOperands;
  if (parseRequiredOperandWithComma(parser, source) ||
      parseRequiredOperandWithComma(parser, destination) ||
      parser.parseOperand(lenBurst) ||
      parseDmaTripleGroup(parser, "nburst", nburstOperands)) {
    return failure();
  }
  while (true) {
    StringRef parsedKeyword;
    SmallVector<OpAsmParser::UnresolvedOperand, mlir::pto::kValue3> loopGroupOperands;
    if (parseOptionalDmaTripleGroupAlias(parser, {"loop", "loop1", "loop2"},
                                         parsedKeyword, loopGroupOperands)) {
      return failure();
    }
    if (parsedKeyword.empty()) {
      break;
    }
    loopCountOperands.push_back(loopGroupOperands[0]);
    loopSrcStrideOperands.push_back(loopGroupOperands[1]);
    loopDstStrideOperands.push_back(loopGroupOperands[mlir::pto::kValue2]);
  }

  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon()) {
    return failure();
  }

  Type sourceType, destinationType, lenBurstType;
  SmallVector<Type> nburstTypes, loopCountTypes, loopSrcStrideTypes,
      loopDstStrideTypes;
  if (parser.parseType(sourceType) || parser.parseComma() ||
      parser.parseType(destinationType) || parser.parseComma() ||
      parser.parseType(lenBurstType) || parser.parseComma() ||
      parseDmaTripleTypes(parser, nburstTypes)) {
    return failure();
  }
  while (succeeded(parser.parseOptionalComma())) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword)) {
      return failure();
    }
    if (!isDmaLoopKeyword(keyword)) {
      return parser.emitError(parser.getCurrentLocation(), "expected 'loop'");
    }
    SmallVector<Type> loopGroupTypes;
    if (parseDmaTripleTypes(parser, loopGroupTypes)) {
      return failure();
    }
    loopCountTypes.push_back(loopGroupTypes[0]);
    loopSrcStrideTypes.push_back(loopGroupTypes[1]);
    loopDstStrideTypes.push_back(loopGroupTypes[mlir::pto::kValue2]);
  }

  int32_t loopGroupCount = static_cast<int32_t>(loopCountOperands.size());
  if (loopCountOperands.size() != loopSrcStrideOperands.size() ||
      loopCountOperands.size() != loopDstStrideOperands.size() ||
      loopCountTypes.size() != loopSrcStrideTypes.size() ||
      loopCountTypes.size() != loopDstStrideTypes.size()) {
    return parser.emitError(parser.getCurrentLocation(),
                            "requires each loop group to provide count, src stride, and dst stride");
  }
  if (loopCountOperands.size() != loopCountTypes.size()) {
    return parser.emitError(parser.getCurrentLocation(),
                            "requires loop operand and type groups to match");
  }

  auto &segments =
      result.getOrAddProperties<MteGmL1Op::Properties>().operandSegmentSizes;
  llvm::copy(ArrayRef<int32_t>{1, 1, 1, 1, 1, 1,
                               loopGroupCount, loopGroupCount, loopGroupCount},
             segments.begin());

  if (parser.resolveOperand(source, sourceType, result.operands) ||
      parser.resolveOperand(destination, destinationType, result.operands) ||
      parser.resolveOperand(lenBurst, lenBurstType, result.operands) ||
      parser.resolveOperands(nburstOperands, nburstTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(loopCountOperands, loopCountTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(loopSrcStrideOperands, loopSrcStrideTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(loopDstStrideOperands, loopDstStrideTypes,
                             parser.getCurrentLocation(), result.operands)) {
    return failure();
  }
  return success();
}

ParseResult MteL1UbOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand source, destination, lenBurst;
  SmallVector<OpAsmParser::UnresolvedOperand> nburstOperands;
  SmallVector<OpAsmParser::UnresolvedOperand> loopCountOperands;
  SmallVector<OpAsmParser::UnresolvedOperand> loopSrcStrideOperands;
  SmallVector<OpAsmParser::UnresolvedOperand> loopDstStrideOperands;
  if (parseRequiredOperandWithComma(parser, source) ||
      parseRequiredOperandWithComma(parser, destination) ||
      parser.parseOperand(lenBurst) ||
      parseDmaTripleGroup(parser, "nburst", nburstOperands)) {
    return failure();
  }
  while (true) {
    StringRef parsedKeyword;
    SmallVector<OpAsmParser::UnresolvedOperand, mlir::pto::kValue3> loopGroupOperands;
    if (parseOptionalDmaTripleGroupAlias(parser, {"loop", "loop1", "loop2"},
                                         parsedKeyword, loopGroupOperands)) {
      return failure();
    }
    if (parsedKeyword.empty()) {
      break;
    }
    loopCountOperands.push_back(loopGroupOperands[0]);
    loopSrcStrideOperands.push_back(loopGroupOperands[1]);
    loopDstStrideOperands.push_back(loopGroupOperands[mlir::pto::kValue2]);
  }

  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon()) {
    return failure();
  }

  Type sourceType, destinationType, lenBurstType;
  SmallVector<Type> nburstTypes, loopCountTypes, loopSrcStrideTypes,
      loopDstStrideTypes;
  if (parser.parseType(sourceType) || parser.parseComma() ||
      parser.parseType(destinationType) || parser.parseComma() ||
      parser.parseType(lenBurstType) || parser.parseComma() ||
      parseDmaTripleTypes(parser, nburstTypes)) {
    return failure();
  }
  while (succeeded(parser.parseOptionalComma())) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword)) {
      return failure();
    }
    if (!isDmaLoopKeyword(keyword)) {
      return parser.emitError(parser.getCurrentLocation(), "expected 'loop'");
    }
    SmallVector<Type> loopGroupTypes;
    if (parseDmaTripleTypes(parser, loopGroupTypes)) {
      return failure();
    }
    loopCountTypes.push_back(loopGroupTypes[0]);
    loopSrcStrideTypes.push_back(loopGroupTypes[1]);
    loopDstStrideTypes.push_back(loopGroupTypes[mlir::pto::kValue2]);
  }

  int32_t loopGroupCount = static_cast<int32_t>(loopCountOperands.size());
  if (loopCountOperands.size() != loopSrcStrideOperands.size() ||
      loopCountOperands.size() != loopDstStrideOperands.size() ||
      loopCountTypes.size() != loopSrcStrideTypes.size() ||
      loopCountTypes.size() != loopDstStrideTypes.size()) {
    return parser.emitError(parser.getCurrentLocation(),
                            "requires each loop group to provide count, src stride, and dst stride");
  }
  if (loopCountOperands.size() != loopCountTypes.size()) {
    return parser.emitError(parser.getCurrentLocation(),
                            "requires loop operand and type groups to match");
  }

  auto &segments =
      result.getOrAddProperties<MteL1UbOp::Properties>().operandSegmentSizes;
  llvm::copy(ArrayRef<int32_t>{1, 1, 1, 1, 1, 1,
                               loopGroupCount, loopGroupCount, loopGroupCount},
             segments.begin());

  if (parser.resolveOperand(source, sourceType, result.operands) ||
      parser.resolveOperand(destination, destinationType, result.operands) ||
      parser.resolveOperand(lenBurst, lenBurstType, result.operands) ||
      parser.resolveOperands(nburstOperands, nburstTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(loopCountOperands, loopCountTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(loopSrcStrideOperands, loopSrcStrideTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(loopDstStrideOperands, loopDstStrideTypes,
                             parser.getCurrentLocation(), result.operands)) {
    return failure();
  }
  return success();
}

ParseResult MteGmL1FracOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand source, destination;
  StringRef modeKeyword;
  SmallVector<OpAsmParser::UnresolvedOperand> shapeOperands;
  SmallVector<OpAsmParser::UnresolvedOperand> srcLayoutOperands;
  SmallVector<OpAsmParser::UnresolvedOperand> dstGroupOperands;
  SmallVector<OpAsmParser::UnresolvedOperand> ctrlOperands;

  if (parseRequiredOperandWithComma(parser, source) ||
      parseRequiredOperandWithComma(parser, destination) ||
      parser.parseKeyword(&modeKeyword) ||
      failed(parseCubeLoadFracModeKeyword(modeKeyword)) || parser.parseComma() ||
      parseFixedKeywordOperandGroup(parser, "shape", mlir::pto::kValue2, shapeOperands) ||
      parser.parseComma() ||
      parseCubeLoadFracSrcLayoutGroup(parser, srcLayoutOperands) ||
      parser.parseComma() ||
      parseFixedKeywordOperandGroup(parser, "dst_group", mlir::pto::kValue4, dstGroupOperands) ||
      parser.parseComma() ||
      parseFixedKeywordOperandGroup(parser, "ctrl", mlir::pto::kValue2, ctrlOperands)) {
    return failure();
  }

  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon()) {
    return failure();
  }

  Type sourceType, destinationType;
  SmallVector<Type> shapeTypes;
  SmallVector<Type> srcLayoutTypes;
  SmallVector<Type> dstGroupTypes;
  SmallVector<Type> ctrlTypes;

  if (parser.parseType(sourceType) || parser.parseComma() ||
      parser.parseType(destinationType) || parser.parseComma() ||
      parser.parseKeyword(modeKeyword) || parser.parseComma() ||
      parseFixedKeywordTypes(parser, "shape", mlir::pto::kValue2, shapeTypes) ||
      parser.parseComma() ||
      parseCubeLoadFracSrcLayoutTypes(parser, srcLayoutTypes) ||
      parser.parseComma() ||
      parseFixedKeywordTypes(parser, "dst_group", mlir::pto::kValue4, dstGroupTypes) ||
      parser.parseComma() ||
      parseFixedKeywordTypes(parser, "ctrl", mlir::pto::kValue2, ctrlTypes)) {
    return failure();
  }

  auto modeOr = parseCubeLoadFracModeKeyword(modeKeyword);
  if (failed(modeOr)) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected one of 'nd2nz' or 'dn2nz'");
  }
  if (shapeOperands.size() != 2 || shapeTypes.size() != 2) {
    return parser.emitError(parser.getCurrentLocation(),
                            "shape requires exactly two operands and types");
  }
  if (srcLayoutOperands.empty() || srcLayoutOperands.size() > mlir::pto::kValue2 ||
      srcLayoutTypes.empty() || srcLayoutTypes.size() > mlir::pto::kValue2) {
    return parser.emitError(parser.getCurrentLocation(),
                            "src_layout requires one or two operands and types");
  }
  if (dstGroupOperands.size() != 4 || dstGroupTypes.size() != 4) {
    return parser.emitError(parser.getCurrentLocation(),
                            "dst_group requires exactly four operands and types");
  }
  if (ctrlOperands.size() != 2 || ctrlTypes.size() != 2) {
    return parser.emitError(parser.getCurrentLocation(),
                            "ctrl requires exactly two operands and types");
  }
  if (srcLayoutOperands.size() != srcLayoutTypes.size()) {
    return parser.emitError(parser.getCurrentLocation(),
                            "src_layout operand and type groups must match");
  }

  bool hasSrcOuterStride = srcLayoutOperands.size() == 2;
  result.addAttribute(getModeAttrName(result.name),
                      CubeLoadFracModeAttr::get(parser.getContext(), *modeOr));

  SmallVector<Type> flatTypes;
  SmallVector<OpAsmParser::UnresolvedOperand> flatOperands;
  flatOperands.append({shapeOperands[0], shapeOperands[1], srcLayoutOperands[0]});
  flatTypes.append({shapeTypes[0], shapeTypes[1], srcLayoutTypes[0]});
  flatOperands.append(dstGroupOperands.begin(), dstGroupOperands.end());
  flatTypes.append(dstGroupTypes.begin(), dstGroupTypes.end());
  flatOperands.append(ctrlOperands.begin(), ctrlOperands.end());
  flatTypes.append(ctrlTypes.begin(), ctrlTypes.end());
  if (hasSrcOuterStride) {
    flatOperands.push_back(srcLayoutOperands[1]);
    flatTypes.push_back(srcLayoutTypes[1]);
  }

  if (parser.resolveOperand(source, sourceType, result.operands) ||
      parser.resolveOperand(destination, destinationType, result.operands) ||
      parser.resolveOperands(flatOperands, flatTypes, parser.getCurrentLocation(),
                             result.operands)) {
    return failure();
  }
  return success();
}

void MteGmL1Op::print(OpAsmPrinter &printer) {
  printer << " " << getSource() << ", " << getDestination() << ", "
          << getLenBurst();
  printDmaTripleGroup(printer, "nburst", getNBurst(), getNburstSrcStride(),
                      getNburstDstStride());
  for (auto [count, srcStride, dstStride] :
       llvm::zip(getLoopCounts(), getLoopSrcStrides(), getLoopDstStrides())) {
    printDmaTripleGroup(printer, "loop", count, srcStride, dstStride);
  }
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : " << getSource().getType() << ", " << getDestination().getType()
          << ", " << getLenBurst().getType() << ", " << getNBurst().getType()
          << ", " << getNburstSrcStride().getType() << ", "
          << getNburstDstStride().getType();
  for (auto [count, srcStride, dstStride] :
       llvm::zip(getLoopCounts(), getLoopSrcStrides(), getLoopDstStrides())) {
    printDmaTripleTypes(printer, "loop", count.getType(), srcStride.getType(),
                        dstStride.getType());
  }
}

void MteL1UbOp::print(OpAsmPrinter &printer) {
  printer << " " << getSource() << ", " << getDestination() << ", "
          << getLenBurst();
  printDmaTripleGroup(printer, "nburst", getNBurst(), getNburstSrcStride(),
                      getNburstDstStride());
  for (auto [count, srcStride, dstStride] :
       llvm::zip(getLoopCounts(), getLoopSrcStrides(), getLoopDstStrides())) {
    printDmaTripleGroup(printer, "loop", count, srcStride, dstStride);
  }
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : " << getSource().getType() << ", " << getDestination().getType()
          << ", " << getLenBurst().getType() << ", " << getNBurst().getType()
          << ", " << getNburstSrcStride().getType() << ", "
          << getNburstDstStride().getType();
  for (auto [count, srcStride, dstStride] :
       llvm::zip(getLoopCounts(), getLoopSrcStrides(), getLoopDstStrides())) {
    printDmaTripleTypes(printer, "loop", count.getType(), srcStride.getType(),
                        dstStride.getType());
  }
}

void MteL1BtOp::build(OpBuilder &builder, OperationState &state, Value source,
                       Value destination, Value lenBurst,
                       pto::DmaLoopConfig nburst) {
  state.addOperands({source, destination, lenBurst, nburst.count,
                     nburst.srcStride, nburst.dstStride});
}

void MteL1FbOp::build(OpBuilder &builder, OperationState &state, Value source,
                     Value destination, Value lenBurst,
                     pto::DmaLoopConfig nburst) {
  state.addOperands({source, destination, lenBurst, nburst.count,
                     nburst.srcStride, nburst.dstStride});
}

ParseResult MteL1BtOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand source, destination, lenBurst;
  SmallVector<OpAsmParser::UnresolvedOperand> nburstOperands;
  if (parseRequiredOperandWithComma(parser, source) ||
      parseRequiredOperandWithComma(parser, destination) ||
      parser.parseOperand(lenBurst) ||
      parseDmaTripleGroup(parser, "nburst", nburstOperands) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon()) {
    return failure();
  }

  Type sourceType, destinationType, lenBurstType;
  SmallVector<Type> nburstTypes;
  if (parser.parseType(sourceType) || parser.parseComma() ||
      parser.parseType(destinationType) || parser.parseComma() ||
      parser.parseType(lenBurstType) || parser.parseComma() ||
      parseDmaTripleTypes(parser, nburstTypes)) {
    return failure();
  }

  if (parser.resolveOperand(source, sourceType, result.operands) ||
      parser.resolveOperand(destination, destinationType, result.operands) ||
      parser.resolveOperand(lenBurst, lenBurstType, result.operands) ||
      parser.resolveOperands(nburstOperands, nburstTypes,
                             parser.getCurrentLocation(), result.operands)) {
    return failure();
  }
  return success();
}

ParseResult MteL1FbOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand source, destination, lenBurst;
  SmallVector<OpAsmParser::UnresolvedOperand> nburstOperands;
  if (parseRequiredOperandWithComma(parser, source) ||
      parseRequiredOperandWithComma(parser, destination) ||
      parser.parseOperand(lenBurst) ||
      parseDmaTripleGroup(parser, "nburst", nburstOperands) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon()) {
    return failure();
  }

  Type sourceType, destinationType, lenBurstType;
  SmallVector<Type> nburstTypes;
  if (parser.parseType(sourceType) || parser.parseComma() ||
      parser.parseType(destinationType) || parser.parseComma() ||
      parser.parseType(lenBurstType) || parser.parseComma() ||
      parseDmaTripleTypes(parser, nburstTypes)) {
    return failure();
  }

  if (parser.resolveOperand(source, sourceType, result.operands) ||
      parser.resolveOperand(destination, destinationType, result.operands) ||
      parser.resolveOperand(lenBurst, lenBurstType, result.operands) ||
      parser.resolveOperands(nburstOperands, nburstTypes,
                             parser.getCurrentLocation(), result.operands)) {
    return failure();
  }
  return success();
}

void MteL1BtOp::print(OpAsmPrinter &printer) {
  printer << " " << getSource() << ", " << getDestination() << ", "
          << getLenBurst();
  printDmaTripleGroup(printer, "nburst", getNBurst(), getNburstSrcGap(),
                      getNburstDstGap());
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : " << getSource().getType() << ", " << getDestination().getType()
          << ", " << getLenBurst().getType() << ", " << getNBurst().getType()
          << ", " << getNburstSrcGap().getType() << ", "
          << getNburstDstGap().getType();
}

void MteL1FbOp::print(OpAsmPrinter &printer) {
  printer << " " << getSource() << ", " << getDestination() << ", "
          << getLenBurst();
  printDmaTripleGroup(printer, "nburst", getNBurst(), getNburstSrcGap(),
                      getNburstDstGap());
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : " << getSource().getType() << ", " << getDestination().getType()
          << ", " << getLenBurst().getType() << ", " << getNBurst().getType()
          << ", " << getNburstSrcGap().getType() << ", "
          << getNburstDstGap().getType();
}

void MteGmL1FracOp::print(OpAsmPrinter &printer) {
  printer << " " << getSource() << ", " << getDestination() << ", "
          << pto::stringifyCubeLoadFracMode(getMode());
  printer << ", shape(" << getNValue() << ", " << getDValue() << ")";
  printCubeLoadFracSrcLayoutGroup(printer, getSrcInnerStride(),
                                  getSrcOuterStride());
  printer << ", dst_group(" << getGroupCount() << ", " << getDstLoop2Stride()
          << ", " << getDstLoop3Stride() << ", " << getDstLoop4Stride()
          << ")";
  printer << ", ctrl(" << getL2CacheCtrl() << ", " << getSmallc0En() << ")";
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                /*elidedAttrs=*/{"operandSegmentSizes",
                                                 "mode"});
  printer << " : " << getSource().getType() << ", " << getDestination().getType()
          << ", " << pto::stringifyCubeLoadFracMode(getMode())
          << ", shape " << getNValue().getType() << ", " << getDValue().getType();
  printCubeLoadFracSrcLayoutTypes(
      printer, getSrcInnerStride().getType(),
      getSrcOuterStride() ? getSrcOuterStride().getType() : Type());
  printer << ", dst_group " << getGroupCount().getType() << ", "
          << getDstLoop2Stride().getType() << ", "
          << getDstLoop3Stride().getType() << ", "
          << getDstLoop4Stride().getType() << ", ctrl "
          << getL2CacheCtrl().getType() << ", " << getSmallc0En().getType();
}

static constexpr uint64_t kRawFillControlFieldMax = 32767;
static constexpr uint64_t kRawFillByteOffsetAlignment = 32;

static StringRef getAddressSpaceDiagnosticName(pto::AddressSpace space) {
  switch (space) {
  case pto::AddressSpace::GM:
    return "gm";
  case pto::AddressSpace::MAT:
    return "mat/l1";
  case pto::AddressSpace::LEFT:
    return "left/l0a";
  case pto::AddressSpace::RIGHT:
    return "right/l0b";
  case pto::AddressSpace::ACC:
    return "acc/l0c";
  case pto::AddressSpace::VEC:
    return "vec/ub";
  case pto::AddressSpace::BIAS:
    return "bt/bias";
  case pto::AddressSpace::SCALING:
    return "fb/scaling";
  case pto::AddressSpace::Zero:
    return "zero";
  }
  return "unknown";
}

static LogicalResult verifyRawFillGeometry(Operation *op, Value byteOffset,
                                           Value repeatTimes,
                                           Value blockNum32b, Value dstGap32b) {
  auto checkNonNegativeConst = [&](Value value, StringRef name) -> LogicalResult {
    if (!value) {
      return success();
    }
    APInt intValue;
    if (matchPattern(value, m_ConstantInt(&intValue))) {
      if (intValue.isNegative()) {
        return op->emitOpError() << name << " must be non-negative";
      }
    }
    return success();
  };
  auto checkConstMax = [&](Value value, StringRef name,
                           uint64_t max) -> LogicalResult {
    if (!value) {
      return success();
    }
    APInt intValue;
    if (matchPattern(value, m_ConstantInt(&intValue))) {
      uint64_t fieldValue = intValue.getZExtValue();
      if (fieldValue > max) {
        return op->emitOpError() << name << " must be <= " << max;
      }
    }
    return success();
  };
  auto checkConstAlignment = [&](Value value, StringRef name,
                                 uint64_t alignment) -> LogicalResult {
    if (!value) {
      return success();
    }
    APInt intValue;
    const bool hasUnalignedConstant =
        matchPattern(value, m_ConstantInt(&intValue)) &&
        intValue.urem(alignment) != 0;
    if (hasUnalignedConstant) {
      return op->emitOpError()
             << name << " must be a multiple of " << alignment << " bytes";
    }
    return success();
  };
  const bool hasNonNegativeGeometry =
      succeeded(checkNonNegativeConst(byteOffset, "byte_offset")) &&
      succeeded(checkNonNegativeConst(repeatTimes, "repeat_times")) &&
      succeeded(checkNonNegativeConst(blockNum32b, "block_num_32b")) &&
      succeeded(checkNonNegativeConst(dstGap32b, "dst_gap_32b"));
  if (!hasNonNegativeGeometry) {
    return failure();
  }
  if (failed(checkConstAlignment(byteOffset, "byte_offset",
                                 kRawFillByteOffsetAlignment))) {
    return failure();
  }
  if (failed(checkConstMax(repeatTimes, "repeat_times",
                           kRawFillControlFieldMax)) ||
      failed(checkConstMax(blockNum32b, "block_num_32b",
                           kRawFillControlFieldMax)) ||
      failed(checkConstMax(dstGap32b, "dst_gap_32b",
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

LogicalResult MteGmL1Op::verify() {
  if (failed(verifyCopyGmToUbufOp(*this, true))) {
    return failure();
  }
  return verifyDmaLoadStoreLoopGroups(
      getOperation(), getLoopCounts(), getLoopSrcStrides(),
      getLoopDstStrides());
}

LogicalResult MteL1UbOp::verify() {
  if (failed(verifyCopyCbufToUbufLikeOp(*this))) {
    return failure();
  }
  return verifyDmaLoadStoreLoopGroups(
      getOperation(), getLoopCounts(), getLoopSrcStrides(),
      getLoopDstStrides());
}

LogicalResult MteL1BtOp::verify() {
  auto getBufferElementType = [](Type type) -> Type {
    if (auto ptrType = dyn_cast<pto::PtrType>(type)) {
      return ptrType.getElementType();
    }
    if (auto memrefType = dyn_cast<BaseMemRefType>(type)) {
      return memrefType.getElementType();
    }
    return {};
  };

  if (!isBufferLike(getSource().getType()) ||
      !isBufferLike(getDestination().getType())) {
    return emitOpError("requires buffer-like source and destination");
  }
  if (getBufferAddressSpace(getSource().getType()) != pto::AddressSpace::MAT) {
    return emitOpError("requires MAT source");
  }
  if (getBufferAddressSpace(getDestination().getType()) != pto::AddressSpace::BIAS) {
    return emitOpError("requires BIAS destination");
  }

  Type srcElem = getBufferElementType(getSource().getType());
  Type dstElem = getBufferElementType(getDestination().getType());
  const bool isF32 = srcElem.isF32() && dstElem.isF32();
  const bool isI32 = isa<IntegerType>(srcElem) && isa<IntegerType>(dstElem) &&
                     cast<IntegerType>(srcElem).getWidth() == 32 &&
                     cast<IntegerType>(dstElem).getWidth() == 32;
  const bool isF16ToF32 = srcElem.isF16() && dstElem.isF32();
  const bool isBF16ToF32 = srcElem.isBF16() && dstElem.isF32();
  if (!isF32 && !isI32 && !isF16ToF32 && !isBF16ToF32) {
    return emitOpError(
        "expects one of f32->f32, i32->i32, f16->f32, or bf16->f32");
  }
  return success();
}

LogicalResult MteL1FbOp::verify() {
  if (!isBufferLike(getSource().getType()) || !isBufferLike(getDestination().getType())) {
    return emitOpError(
        "requires typed !pto.ptr or memref source and destination");
  }

  auto getAddressSpace = [](Type type) -> std::optional<pto::AddressSpace> {
    if (auto ptrType = dyn_cast<pto::PtrType>(type)) {
      return ptrType.getMemorySpace().getAddressSpace();
    }
    if (auto memrefType = dyn_cast<BaseMemRefType>(type)) {
      Attribute memorySpace = memrefType.getMemorySpace();
      if (auto addrSpace = dyn_cast_or_null<pto::AddressSpaceAttr>(memorySpace)) {
        return addrSpace.getAddressSpace();
      }
      if (auto intAttr = dyn_cast_or_null<IntegerAttr>(memorySpace)) {
        return static_cast<pto::AddressSpace>(intAttr.getInt());
      }
    }
    return std::nullopt;
  };

  std::optional<pto::AddressSpace> sourceAS = getAddressSpace(getSource().getType());
  std::optional<pto::AddressSpace> destinationAS =
      getAddressSpace(getDestination().getType());
  if (!sourceAS || !destinationAS) {
    return emitOpError("requires source and destination with PTO address spaces");
  }
  if (*sourceAS != pto::AddressSpace::MAT) {
    return emitOpError("requires source in mat address space");
  }
  if (*destinationAS != pto::AddressSpace::SCALING) {
    return emitOpError("requires destination in scaling address space");
  }
  return success();
}

LogicalResult MteGmL1FracOp::verify() {
  if (failed(verifyCopyGmToUbufOp(*this, true))) {
    return failure();
  }

  auto checkNonNegativeConst = [&](Value value, StringRef name) -> LogicalResult {
    APInt intValue;
    if (matchPattern(value, m_ConstantInt(&intValue)) && intValue.isNegative()) {
      return emitOpError() << name << " must be non-negative";
    }
    return success();
  };
  if (failed(checkNonNegativeConst(getGroupCount(), "group_count")) ||
      failed(checkNonNegativeConst(getSrcInnerStride(), "src_inner_stride")) ||
      failed(checkNonNegativeConst(getDstLoop2Stride(), "dst_loop2_stride")) ||
      failed(checkNonNegativeConst(getDstLoop3Stride(), "dst_loop3_stride")) ||
      failed(checkNonNegativeConst(getDstLoop4Stride(), "dst_loop4_stride")) ||
      (getSrcOuterStride() &&
       failed(checkNonNegativeConst(getSrcOuterStride(), "src_outer_stride")))) {
    return failure();
  }

  APInt groupCount;
  if (matchPattern(getGroupCount(), m_ConstantInt(&groupCount)) &&
      groupCount.isZero()) {
    return emitOpError("group_count must be greater than zero");
  }

  APInt smallc0En;
  APInt dValue;
  if (matchPattern(getSmallc0En(), m_ConstantInt(&smallc0En)) &&
      smallc0En.getBoolValue() && matchPattern(getDValue(), m_ConstantInt(&dValue)) &&
      dValue.ugt(mlir::pto::kValue4)) {
    return emitOpError("smallc0_en requires d_value <= 4");
  }

  return success();
}

void MteGmL1Op::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

void MteL1UbOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

void MteL1BtOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

void MteL1FbOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

void MteGmL1FracOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

ParseResult MteL0cL1Op::parse(OpAsmParser &parser, OperationState &result) {
  Builder builder(parser.getContext());
  StructuredAccStoreAsmState state;
  OpAsmParser::UnresolvedOperand source, destination, m, n, srcStride,
      dstStride;
  if (parseRequiredOperandWithComma(parser, source) ||
      parseRequiredOperandWithComma(parser, destination) ||
      parseRequiredOperandWithComma(parser, m) ||
      parseRequiredOperandWithComma(parser, n) ||
      parseRequiredOperandWithComma(parser, srcStride) ||
      parseRequiredOperandWithComma(parser, dstStride) ||
      parseStructuredAccStoreClauses(parser, state) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon()) {
    return failure();
  }

  Type sourceType, destinationType, mType, nType, srcStrideType, dstStrideType;
  if (parser.parseType(sourceType) || parser.parseComma() ||
      parser.parseType(destinationType) || parser.parseComma() ||
      parser.parseType(mType) || parser.parseComma() || parser.parseType(nType) ||
      parser.parseComma() || parser.parseType(srcStrideType) ||
      parser.parseComma() || parser.parseType(dstStrideType) ||
      parseStructuredAccStoreTailTypes(parser, state)) {
    return failure();
  }

  setStructuredAccStoreSegmentSizes<MteL0cL1Op>(
      result, {1, 1, 1, 1, 1, 1, !state.preQuantOperands.empty() ? 1 : 0,
               !state.preReluOperands.empty() ? 1 : 0,
               !state.clipValueOperands.empty() ? 1 : 0,
               !state.splitOperands.empty() ? 1 : 0,
               !state.loop0SrcStrideOperands.empty() ? 1 : 0,
               !state.loop3CountOperands.empty() ? 1 : 0,
               !state.loop3SrcStrideOperands.empty() ? 1 : 0,
               !state.loop3DstStrideOperands.empty() ? 1 : 0});
  if (state.atomicType || state.atomicOp) {
    return parser.emitError(parser.getCurrentLocation(),
                            "atomic is only supported for mte_l0c_gm");
  }
  addStructuredAccStoreAttrs<MteL0cL1Op>(result, builder, state);

  if (parser.resolveOperand(source, sourceType, result.operands) ||
      parser.resolveOperand(destination, destinationType, result.operands) ||
      parser.resolveOperand(m, mType, result.operands) ||
      parser.resolveOperand(n, nType, result.operands) ||
      parser.resolveOperand(srcStride, srcStrideType, result.operands) ||
      parser.resolveOperand(dstStride, dstStrideType, result.operands) ||
      parser.resolveOperands(state.preQuantOperands, state.preQuantTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(state.preReluOperands, state.preReluTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(state.clipValueOperands, state.clipValueTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(state.splitOperands, state.splitTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(state.loop0SrcStrideOperands,
                             state.loop0SrcStrideTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(state.loop3CountOperands, state.loop3CountTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(state.loop3SrcStrideOperands,
                             state.loop3SrcStrideTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(state.loop3DstStrideOperands,
                             state.loop3DstStrideTypes,
                             parser.getCurrentLocation(), result.operands)) {
    return failure();
  }
  return success();
}

void MteL0cL1Op::print(OpAsmPrinter &printer) {
  printer << " " << getSource() << ", " << getDestination() << ", " << getM()
          << ", " << getN() << ", " << getSrcStride() << ", " << getDstStride();
  printStructuredAccStoreClauses(printer, getUnitFlag(), getPreQuant(),
                                 getPreQuantMode(), getPreRelu(),
                                 getPreReluMode(), getClipValue(), getMode(),
                                 getSplit(), getLoop0SrcStride(),
                                 getLoop3Count(), getLoop3SrcStride(),
                                 getLoop3DstStride(), getSatMode(),
                                 getAtomicType(), getAtomicOp());
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                /*elidedAttrs=*/{"operandSegmentSizes",
                                                 "mode",
                                                 "unit_flag",
                                                 "pre_quant_mode",
                                                 "pre_relu_mode",
                                                 "atomic_type",
                                                 "atomic_op",
                                                 "sat_mode"});
  printer << " : " << getSource().getType() << ", " << getDestination().getType()
          << ", " << getM().getType() << ", " << getN().getType() << ", "
          << getSrcStride().getType() << ", " << getDstStride().getType();
  printStructuredAccStoreOptionalTypes(
      printer, getPreQuant(), getPreRelu(), getClipValue(), getSplit(),
      getLoop0SrcStride(), getLoop3Count(), getLoop3SrcStride(),
      getLoop3DstStride());
}

static LogicalResult verifyCubeBridgeLoadStart(Operation *op, Value firstStart,
                                               StringRef firstName,
                                               Value secondStart,
                                               StringRef secondName) {
  auto checkNonNegativeConst = [&](Value value, StringRef name) -> LogicalResult {
    APInt intValue;
    if (matchPattern(value, m_ConstantInt(&intValue)) && intValue.isNegative()) {
      return op->emitOpError() << name << " must be non-negative";
    }
    return success();
  };
  if (failed(checkNonNegativeConst(firstStart, firstName)) ||
      failed(checkNonNegativeConst(secondStart, secondName))) {
    return failure();
  }
  return success();
}

template <typename OpTy>
static LogicalResult verifyCubeBridgeLoadStart(OpTy op) {
  return verifyCubeBridgeLoadStart(op.getOperation(), op.getStartRow(),
                                   "start_row", op.getStartCol(), "start_col");
}

template <typename OpTy>
static void setCubeBridgeLoadOperandSegmentSizes(
    OperationState &result, ArrayRef<int32_t> segmentSizes) {
  auto &segments = result.getOrAddProperties<typename OpTy::Properties>()
                       .operandSegmentSizes;
  llvm::copy(segmentSizes, segments.begin());
}

struct CubeBridgeLoadAsmOperand {
  OpAsmParser::UnresolvedOperand operand;
  Type type;
  bool present = false;
};

static std::optional<unsigned> getCubeBridgeLoadOperandIndex(
    StringRef keyword, ArrayRef<StringRef> shapeNames,
    ArrayRef<StringRef> fullNames) {
  for (auto [index, name] : llvm::enumerate(shapeNames)) {
    if (keyword == name) {
      return index;
    }
}

  for (auto [index, name] : llvm::enumerate(fullNames)) {
    if (keyword == name) {
      return shapeNames.size() + index;
}
}
  return std::nullopt;
}

template <typename OpTy>
static ParseResult parseMteL1L0OptionalOperandsOp(
    OpAsmParser &parser, OperationState &result, ArrayRef<StringRef> shapeNames,
    ArrayRef<StringRef> fullNames, StringRef operandDescription = "operands") {
  OpAsmParser::UnresolvedOperand source;
  OpAsmParser::UnresolvedOperand destination;
  if (parser.parseOperand(source) || parser.parseComma() ||
      parser.parseOperand(destination)) {
    return failure();
}

  SmallVector<OpAsmParser::UnresolvedOperand, 6> legacyOperands;
  SmallVector<CubeBridgeLoadAsmOperand, 10> namedOperands(10);
  SmallVector<unsigned, 10> namedOperandOrder;
  bool usesNamedOperands = false;

  auto parseNamedOperand = [&](StringRef keyword) -> ParseResult {
    std::optional<unsigned> index =
        getCubeBridgeLoadOperandIndex(keyword, shapeNames, fullNames);
    if (!index) {
      return parser.emitError(parser.getCurrentLocation(),
                              "unknown cube bridge load operand '")
             << keyword << "'";
}
    if (namedOperands[*index].present) {
      return parser.emitError(parser.getCurrentLocation(),
                              "duplicate cube bridge load operand '")
             << keyword << "'";
}
    if (parser.parseLParen() || parser.parseOperand(namedOperands[*index].operand) ||
        parser.parseRParen()) {
      return failure();
}
    namedOperands[*index].present = true;
    namedOperandOrder.push_back(*index);
    return success();
  };

  if (succeeded(parser.parseOptionalComma())) {
    StringRef keyword;
    if (succeeded(parser.parseOptionalKeyword(&keyword))) {
      usesNamedOperands = true;
      if (parseNamedOperand(keyword)) {
        return failure();
      }
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseKeyword(&keyword) || parseNamedOperand(keyword)) {
          return failure();
        }
      }
    } else {
      OpAsmParser::UnresolvedOperand operand;
      if (parser.parseOperand(operand)) {
        return failure();
      }
      legacyOperands.push_back(operand);
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseOperand(operand)) {
          return failure();
        }
        legacyOperands.push_back(operand);
      }
    }
  }

  if (!usesNamedOperands && legacyOperands.size() != 4 &&
      legacyOperands.size() != 6) {
    return parser.emitError(
               parser.getCurrentLocation(),
               "expects either four shape-derived or six full positional ")
           << operandDescription;
}

  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon()) {
    return failure();
  }

  Type sourceType;
  Type destinationType;
  if (parser.parseType(sourceType) || parser.parseComma() ||
      parser.parseType(destinationType)) {
    return failure();
  }

  SmallVector<Type, mlir::pto::kValue6> legacyTypes;
  if (usesNamedOperands) {
    for (unsigned index : namedOperandOrder) {
      Type type;
      if (parser.parseComma() || parser.parseType(type)) {
        return failure();
      }
      namedOperands[index].type = type;
    }
  } else {
    for (size_t index = 0; index < legacyOperands.size(); ++index) {
      Type type;
      if (parser.parseComma() || parser.parseType(type)) {
        return failure();
      }
      legacyTypes.push_back(type);
    }
  }

  SmallVector<int32_t, 12> segmentSizes(12, 0);
  segmentSizes[0] = 1;
  segmentSizes[1] = 1;

  if (parser.resolveOperand(source, sourceType, result.operands) ||
      parser.resolveOperand(destination, destinationType, result.operands)) {
    return failure();
}

  if (usesNamedOperands) {
    for (unsigned index = 0; index < namedOperands.size(); ++index) {
      if (!namedOperands[index].present) {
        continue;
      }
      segmentSizes[2 + index] = 1;
      if (parser.resolveOperand(namedOperands[index].operand,
                                namedOperands[index].type,
                                result.operands)) {
        return failure();
}
    }
  } else {
    const unsigned base = legacyOperands.size() <= 4 ? 0 : 4;
    for (unsigned index = 0; index < legacyOperands.size(); ++index) {
      segmentSizes[2 + base + index] = 1;
      if (parser.resolveOperand(legacyOperands[index], legacyTypes[index],
                                result.operands)) {
        return failure();
}
    }
  }
  setCubeBridgeLoadOperandSegmentSizes<OpTy>(result, segmentSizes);
  return success();
}

static void printMteL1L0OptionalOperandsOp(
    OpAsmPrinter &printer, Operation *operation, Value source, Value destination,
    ArrayRef<Value> shapeOperands, ArrayRef<StringRef> shapeNames,
    ArrayRef<Value> fullOperands, ArrayRef<StringRef> fullNames) {

  const bool hasShape = llvm::any_of(shapeOperands, [](Value value) {
    return static_cast<bool>(value);
  });
  const bool hasFull = llvm::any_of(fullOperands, [](Value value) {
    return static_cast<bool>(value);
  });
  const bool isShapeForm = hasShape && !hasFull &&
      llvm::all_of(shapeOperands, [](Value value) { return static_cast<bool>(value); });
  const bool isFullForm = hasFull && !hasShape &&
      llvm::all_of(fullOperands, [](Value value) { return static_cast<bool>(value); });

  printer << " " << source << ", " << destination;
  SmallVector<Value, mlir::pto::kValue10> printedOperands;
  if (isShapeForm) {
    for (Value value : shapeOperands) {
      printer << ", " << value;
      printedOperands.push_back(value);
    }
  } else if (isFullForm) {
    for (Value value : fullOperands) {
      printer << ", " << value;
      printedOperands.push_back(value);
    }
  } else {
    for (auto [index, value] : llvm::enumerate(shapeOperands)) {
      if (!value) {
        continue;
      }
      printer << ", " << shapeNames[index] << "(" << value << ")";
      printedOperands.push_back(value);
    }
    for (auto [index, value] : llvm::enumerate(fullOperands)) {
      if (!value) {
        continue;
      }
      printer << ", " << fullNames[index] << "(" << value << ")";
      printedOperands.push_back(value);
    }
  }

  printer.printOptionalAttrDict(operation->getAttrs(),
                                /*elidedAttrs=*/{"operandSegmentSizes"});
  printer << " : " << source.getType() << ", " << destination.getType();
  for (Value value : printedOperands) {
    printer << ", " << value.getType();
  }
}

static LogicalResult verifyMxLoadOperands(Operation *op,
                                          ArrayRef<Value> shapeOperands,
                                          ArrayRef<StringRef> shapeNames,
                                          ArrayRef<Value> fullOperands) {
  auto checkNonNegativeConst = [&](Value value,
                                   StringRef name) -> LogicalResult {
    APInt intValue;
    if (matchPattern(value, m_ConstantInt(&intValue)) && intValue.isNegative()) {
      return op->emitOpError() << name << " must be non-negative";
    }
    return success();
  };

  const bool hasShape = llvm::any_of(shapeOperands, [](Value value) {
    return static_cast<bool>(value);
  });
  const bool hasFull = llvm::any_of(fullOperands, [](Value value) {
    return static_cast<bool>(value);
  });
  if (hasShape && hasFull) {
    return op->emitOpError()
           << "cannot mix shape-derived MX operands with full MX operands";
  }
  if (!hasShape && !hasFull) {
    return op->emitOpError()
           << "requires either all shape-derived MX operands or all full MX operands";
  }

  if (hasShape) {
    for (auto [value, name] : llvm::zip(shapeOperands, shapeNames)) {
      if (!value) {
        return op->emitOpError()
               << "shape-derived MX form requires " << name;
      }
    }
    return verifyCubeBridgeLoadStart(op, shapeOperands[mlir::pto::kValue2], shapeNames[mlir::pto::kValue2],
                                     shapeOperands[mlir::pto::kValue3], shapeNames[mlir::pto::kValue3]);
  }

  static constexpr StringRef kFullNames[] = {
      "x_start", "y_start", "x_step", "y_step", "src_stride", "dst_stride"};
  for (auto [value, name] : llvm::zip(fullOperands, kFullNames)) {
    if (!value) {
      return op->emitOpError() << "full MX form requires " << name;
    }
  }
  if (failed(verifyCubeBridgeLoadStart(op, fullOperands[0], "x_start",
                                       fullOperands[1], "y_start"))) {
    return failure();
  }
  if (failed(checkNonNegativeConst(fullOperands[mlir::pto::kValue2], "x_step")) ||
      failed(checkNonNegativeConst(fullOperands[mlir::pto::kValue3], "y_step")) ||
      failed(checkNonNegativeConst(fullOperands[mlir::pto::kValue4], "src_stride")) ||
      failed(checkNonNegativeConst(fullOperands[mlir::pto::kValue5], "dst_stride"))) {
    return failure();
  }
  return success();
}

static LogicalResult verifyMxPointerAlignment(Operation *op, Value pointer,
                                              StringRef pointerName,
                                              int64_t alignmentBytes) {
  auto pointerCast = pointer.getDefiningOp<CastPtrOp>();
  if (!pointerCast || !isa<IntegerType>(pointerCast.getInput().getType())) {
    return success();
  }

  std::optional<int64_t> address =
      mlir::getConstantIntValue(pointerCast.getInput());
  if (!address || (*address % alignmentBytes) == 0) {
    return success();
  }

  return op->emitOpError()
         << "statically known LOAD.MX " << pointerName
         << " address must be aligned to " << alignmentBytes << " bytes, got "
         << *address;
}

static LogicalResult verifyMxLoadAlignment(Operation *op, Value source,
                                           Value destination) {
  constexpr int64_t kMxSourceAlignmentBytes = 32;
  constexpr int64_t kMxDestinationAddressUnitBytes = 16;
  if (failed(verifyMxPointerAlignment(op, source, "source",
                                      kMxSourceAlignmentBytes))) {
    return failure();
  }
  return verifyMxPointerAlignment(op, destination, "destination",
                                  kMxDestinationAddressUnitBytes);
}

static LogicalResult verifyStaticControlRange(Operation *op, Value value,
                                              StringRef name, int64_t min,
                                              int64_t max) {
  APInt intValue;
  if (!matchPattern(value, m_ConstantInt(&intValue))) {
    return success();
}
  int64_t signedValue = intValue.getSExtValue();
  if (signedValue < min) {
    return op->emitOpError() << name
                             << (min == 0 ? " must be non-negative"
                                          : " must be greater than zero");
}
  if (signedValue > max) {
    return op->emitOpError() << name << " must be <= " << max
                             << " to fit the hardware control field";
}
  return success();
}

template <typename OpTy>
static LogicalResult verifyExplicitCubeBridgeLoadControls(OpTy op) {
  constexpr int64_t kU16Max = 65535;
  constexpr int64_t kU8Max = 255;
  Operation *operation = op.getOperation();
  if (failed(verifyStaticControlRange(operation, op.getMStart(), "m_start", 0,
                                      kU16Max)) ||
      failed(verifyStaticControlRange(operation, op.getKStart(), "k_start", 0,
                                      kU16Max)) ||
      failed(verifyStaticControlRange(operation, op.getMStep(), "m_step", 1,
                                      kU8Max)) ||
      failed(verifyStaticControlRange(operation, op.getKStep(), "k_step", 1,
                                      kU8Max)) ||
      failed(verifyStaticControlRange(operation, op.getSrcStride(),
                                      "src_stride", 1, kU16Max)) ||
      failed(verifyStaticControlRange(operation, op.getDstStride(),
                                      "dst_stride", 1, kU16Max))) {
    return failure();
}
  return success();
}

template <typename OpTy>
static LogicalResult verifyS4CubeBridgeLoad(OpTy op,
                                            AddressSpace expectedDstSpace,
                                            StringRef dstName) {
  if (failed(verifyCubeBridgeLoadLikeOp(op, expectedDstSpace, dstName))) {
    return failure();
}

  Type sourceElem = getBufferElementType(op.getSource().getType());
  Type destinationElem = getBufferElementType(op.getDestination().getType());
  if (!pto::isPTOFloat4PackedType(sourceElem)) {
    return op.emitOpError(
        "requires packed FP4 source element type f4e1m2x2 or f4e2m1x2");
}
  if (!pto::isPTOFloat4PackedType(destinationElem)) {
    return op.emitOpError(
        "requires packed FP4 destination element type f4e1m2x2 or f4e2m1x2");
}
  if (sourceElem != destinationElem) {
    return op.emitOpError(
        "requires source and destination packed FP4 element types to match");
}
  if (failed(verifyExplicitCubeBridgeLoadControls(op))) {
    return failure();
}
  return verifyStaticControlRange(op.getOperation(), op.getTranspose(),
                                  "transpose", 0, 1);
}

template <typename OpTy>
static LogicalResult verifyRegularCubeBridgeLoad(OpTy op,
                                                 AddressSpace expectedDstSpace,
                                                 StringRef dstName) {
  if (failed(verifyCubeBridgeLoadLikeOp(op, expectedDstSpace, dstName))) {
    return failure();
}
  Type sourceElem = getBufferElementType(op.getSource().getType());
  Type destinationElem = getBufferElementType(op.getDestination().getType());
  if (pto::isPTOFloat4PackedType(sourceElem)) {
    return op.emitOpError("packed FP4 source requires the S4 load operation");
}
  if (pto::isPTOFloat4PackedType(destinationElem)) {
    return op.emitOpError(
        "packed FP4 destination requires the S4 load operation");
}
  return verifyExplicitCubeBridgeLoadControls(op);
}

static LogicalResult verifyMteL1L0LoadOperands(
    Operation *op, ArrayRef<Value> shapeOperands,
    ArrayRef<StringRef> shapeNames, ArrayRef<Value> fullOperands) {
  const bool hasShape = llvm::any_of(shapeOperands, [](Value value) {
    return static_cast<bool>(value);
  });
  const bool hasFull = llvm::any_of(fullOperands, [](Value value) {
    return static_cast<bool>(value);
  });
  if (hasShape && hasFull) {
    return op->emitOpError(
        "cannot mix shape-derived operands with full control operands");
}
  if (!hasShape && !hasFull) {
    return op->emitOpError(
        "requires either all shape-derived operands or all full control operands");
}

  if (hasShape) {
    for (auto [value, name] : llvm::zip(shapeOperands, shapeNames)) {
      if (!value) {
        return op->emitOpError()
               << "shape-derived form requires " << name;
}
}
    return verifyCubeBridgeLoadStart(op, shapeOperands[2], shapeNames[2],
                                     shapeOperands[3], shapeNames[3]);
  }

  static constexpr StringRef kFullNames[] = {
      "m_start", "k_start", "m_step", "k_step", "src_stride", "dst_stride"};
  for (auto [value, name] : llvm::zip(fullOperands, kFullNames)) {
    if (!value) {
      return op->emitOpError() << "full control form requires " << name;
}
}

  constexpr int64_t kU16Max = 65535;
  constexpr int64_t kU8Max = 255;
  if (failed(verifyStaticControlRange(op, fullOperands[0], "m_start", 0,
                                      kU16Max)) ||
      failed(verifyStaticControlRange(op, fullOperands[1], "k_start", 0,
                                      kU16Max)) ||
      failed(verifyStaticControlRange(op, fullOperands[2], "m_step", 1,
                                      kU8Max)) ||
      failed(verifyStaticControlRange(op, fullOperands[3], "k_step", 1,
                                      kU8Max)) ||
      failed(verifyStaticControlRange(op, fullOperands[4], "src_stride", 1,
                                      kU16Max)) ||
      failed(verifyStaticControlRange(op, fullOperands[5], "dst_stride", 1,
                                      kU16Max))) {
    return failure();
}
  return success();
}

LogicalResult MteL0cL1Op::verify() {
  if (!isBufferLike(getSource().getType()) ||
      !isBufferLike(getDestination().getType())) {
    return emitOpError("requires buffer-like source and destination");
}
  std::optional<AddressSpace> sourceSpace =
      getBufferAddressSpace(getSource().getType());
  std::optional<AddressSpace> destinationSpace =
      getBufferAddressSpace(getDestination().getType());
  if (sourceSpace != AddressSpace::ACC || destinationSpace != AddressSpace::MAT) {
    return emitOpError("requires ACC source and MAT destination");
  }
  return verifyStructuredAccStoreLike(
      *this, getSource().getType(), getDestination().getType(), getPreQuant(), getPreRelu(),
      getClipValue(), getSplit(), getLoop0SrcStride(), getLoop3Count(),
      getLoop3SrcStride(), getLoop3DstStride(), getUnitFlag(),
      getPreQuantMode(), getPreReluMode(), getMode(), std::nullopt,
      std::nullopt, /*allowAtomic=*/false);
}

LogicalResult MteL1L0aOp::verify() {
  if (failed(verifyCubeBridgeLoadLikeOp(*this, AddressSpace::LEFT, "LEFT"))) {
    return failure();
  }
  return verifyMteL1L0LoadOperands(
      getOperation(), {getM(), getK(), getStartRow(), getStartCol()},
      {"m", "k", "start_row", "start_col"},
      {getMStart(), getKStart(), getMStep(), getKStep(), getSrcStride(),
       getDstStride()});
}

LogicalResult MteL1L0bOp::verify() {
  if (failed(verifyCubeBridgeLoadLikeOp(*this, AddressSpace::RIGHT, "RIGHT"))) {
    return failure();
  }
  return verifyMteL1L0LoadOperands(
      getOperation(), {getK(), getN(), getStartRow(), getStartCol()},
      {"k", "n", "start_row", "start_col"},
      {getMStart(), getKStart(), getMStep(), getKStep(), getSrcStride(),
       getDstStride()});
}

ParseResult MteL1L0aOp::parse(OpAsmParser &parser, OperationState &result) {
  static constexpr StringRef kShapeNames[] = {
      "m", "k", "start_row", "start_col"};
  static constexpr StringRef kFullNames[] = {
      "m_start", "k_start", "m_step", "k_step", "src_stride", "dst_stride"};
  return parseMteL1L0OptionalOperandsOp<MteL1L0aOp>(
      parser, result, kShapeNames, kFullNames);
}

void MteL1L0aOp::print(OpAsmPrinter &printer) {
  static constexpr StringRef kShapeNames[] = {
      "m", "k", "start_row", "start_col"};
  static constexpr StringRef kFullNames[] = {
      "m_start", "k_start", "m_step", "k_step", "src_stride", "dst_stride"};
  printMteL1L0OptionalOperandsOp(
      printer, getOperation(), getSource(), getDestination(),
      {getM(), getK(), getStartRow(), getStartCol()}, kShapeNames,
      {getMStart(), getKStart(), getMStep(), getKStep(), getSrcStride(),
       getDstStride()},
      kFullNames);
}

ParseResult MteL1L0bOp::parse(OpAsmParser &parser, OperationState &result) {
  static constexpr StringRef kShapeNames[] = {
      "k", "n", "start_row", "start_col"};
  static constexpr StringRef kFullNames[] = {
      "m_start", "k_start", "m_step", "k_step", "src_stride", "dst_stride"};
  return parseMteL1L0OptionalOperandsOp<MteL1L0bOp>(
      parser, result, kShapeNames, kFullNames);
}

void MteL1L0bOp::print(OpAsmPrinter &printer) {
  static constexpr StringRef kShapeNames[] = {
      "k", "n", "start_row", "start_col"};
  static constexpr StringRef kFullNames[] = {
      "m_start", "k_start", "m_step", "k_step", "src_stride", "dst_stride"};
  printMteL1L0OptionalOperandsOp(
      printer, getOperation(), getSource(), getDestination(),
      {getK(), getN(), getStartRow(), getStartCol()}, kShapeNames,
      {getMStart(), getKStart(), getMStep(), getKStep(), getSrcStride(),
       getDstStride()},
      kFullNames);
}

ParseResult MteL1L0aMxOp::parse(OpAsmParser &parser, OperationState &result) {
  static constexpr StringRef kShapeNames[] = {
      "m", "k", "start_row", "start_col"};
  static constexpr StringRef kFullNames[] = {
      "x_start", "y_start", "x_step", "y_step", "src_stride", "dst_stride"};
  return parseMteL1L0OptionalOperandsOp<MteL1L0aMxOp>(
      parser, result, kShapeNames, kFullNames, "MX operands");
}

void MteL1L0aMxOp::print(OpAsmPrinter &printer) {
  static constexpr StringRef kShapeNames[] = {
      "m", "k", "start_row", "start_col"};
  static constexpr StringRef kFullNames[] = {
      "x_start", "y_start", "x_step", "y_step", "src_stride", "dst_stride"};
  printMteL1L0OptionalOperandsOp(
      printer, getOperation(), getSource(), getDestination(),
      {getM(), getK(), getStartRow(), getStartCol()}, kShapeNames,
      {getXStart(), getYStart(), getXStep(), getYStep(), getSrcStride(),
       getDstStride()},
      kFullNames);
}

LogicalResult MteL1L0aMxOp::verify() {
  if (failed(verifyCubeBridgeLoadLikeOp(*this, AddressSpace::LEFT, "LEFT"))) {
    return failure();
  }
  if (failed(verifyMxLoadOperands(
          getOperation(), {getM(), getK(), getStartRow(), getStartCol()},
          {"m", "k", "start_row", "start_col"},
          {getXStart(), getYStart(), getXStep(), getYStep(), getSrcStride(),
           getDstStride()}))) {
    return failure();
  }
  return verifyMxLoadAlignment(getOperation(), getSource(), getDestination());
}

ParseResult MteL1L0bMxOp::parse(OpAsmParser &parser, OperationState &result) {
  static constexpr StringRef kShapeNames[] = {
      "k", "n", "start_row", "start_col"};
  static constexpr StringRef kFullNames[] = {
      "x_start", "y_start", "x_step", "y_step", "src_stride", "dst_stride"};
  return parseMteL1L0OptionalOperandsOp<MteL1L0bMxOp>(
      parser, result, kShapeNames, kFullNames, "MX operands");
}

void MteL1L0bMxOp::print(OpAsmPrinter &printer) {
  static constexpr StringRef kShapeNames[] = {
      "k", "n", "start_row", "start_col"};
  static constexpr StringRef kFullNames[] = {
      "x_start", "y_start", "x_step", "y_step", "src_stride", "dst_stride"};
  printMteL1L0OptionalOperandsOp(
      printer, getOperation(), getSource(), getDestination(),
      {getK(), getN(), getStartRow(), getStartCol()}, kShapeNames,
      {getXStart(), getYStart(), getXStep(), getYStep(), getSrcStride(),
       getDstStride()},
      kFullNames);
}

LogicalResult MteL1L0bMxOp::verify() {
  if (failed(verifyCubeBridgeLoadLikeOp(*this, AddressSpace::RIGHT, "RIGHT"))) {
    return failure();
  }
  if (failed(verifyMxLoadOperands(
          getOperation(), {getK(), getN(), getStartRow(), getStartCol()},
          {"k", "n", "start_row", "start_col"},
          {getXStart(), getYStart(), getXStep(), getYStep(), getSrcStride(),
           getDstStride()}))) {
    return failure();
  }
  return verifyMxLoadAlignment(getOperation(), getSource(), getDestination());
}

LogicalResult LoadCbufToCaMxOp::verify() {
  if (failed(verifyCubeBridgeLoadLikeOp(*this, AddressSpace::LEFT, "LEFT"))) {
    return failure();
  }
  return verifyCubeBridgeLoadStart(getOperation(), getXStartPosition(),
                                   "x_start_position", getYStartPosition(),
                                   "y_start_position");
}

LogicalResult LoadCbufToCbMxOp::verify() {
  if (failed(verifyCubeBridgeLoadLikeOp(*this, AddressSpace::RIGHT, "RIGHT"))) {
    return failure();
  }
  return verifyCubeBridgeLoadStart(getOperation(), getXStartPosition(),
                                   "x_start_position", getYStartPosition(),
                                   "y_start_position");
}

LogicalResult LoadCbufToCaOp::verify() {
  return verifyRegularCubeBridgeLoad(*this, AddressSpace::LEFT, "LEFT");
}

LogicalResult LoadCbufToCbOp::verify() {
  return verifyRegularCubeBridgeLoad(*this, AddressSpace::RIGHT, "RIGHT");
}

LogicalResult LoadCbufToCaS4Op::verify() {
  return verifyS4CubeBridgeLoad(*this, AddressSpace::LEFT, "LEFT");
}

LogicalResult LoadCbufToCbS4Op::verify() {
  return verifyS4CubeBridgeLoad(*this, AddressSpace::RIGHT, "RIGHT");
}

void MteL1L0aOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

void MteL1L0bOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

void MteL1L0aMxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

void MteL1L0bMxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

void MteL0cL1Op::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

ParseResult MteL0cGmOp::parse(OpAsmParser &parser, OperationState &result) {
  Builder builder(parser.getContext());
  StructuredAccStoreAsmState state;
  OpAsmParser::UnresolvedOperand source, destination, m, n, srcStride,
      dstStride, sid, l2CacheCtrl;
  if (parseRequiredOperandWithComma(parser, source) ||
      parseRequiredOperandWithComma(parser, destination) ||
      parseRequiredOperandWithComma(parser, m) ||
      parseRequiredOperandWithComma(parser, n) ||
      parseRequiredOperandWithComma(parser, srcStride) ||
      parseRequiredOperandWithComma(parser, dstStride) ||
      parseRequiredOperandWithComma(parser, sid) ||
      parseRequiredOperandWithComma(parser, l2CacheCtrl) ||
      parseStructuredAccStoreClauses(parser, state) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon()) {
    return failure();
  }

  Type sourceType, destinationType, mType, nType, srcStrideType, dstStrideType,
      sidType, l2CacheCtrlType;
  if (parser.parseType(sourceType) || parser.parseComma() ||
      parser.parseType(destinationType) || parser.parseComma() ||
      parser.parseType(mType) || parser.parseComma() || parser.parseType(nType) ||
      parser.parseComma() || parser.parseType(srcStrideType) ||
      parser.parseComma() || parser.parseType(dstStrideType) ||
      parser.parseComma() || parser.parseType(sidType) ||
      parser.parseComma() || parser.parseType(l2CacheCtrlType) ||
      parseStructuredAccStoreTailTypes(parser, state)) {
    return failure();
  }

  setStructuredAccStoreSegmentSizes<MteL0cGmOp>(
      result, {1, 1, 1, 1, 1, 1, !state.preQuantOperands.empty() ? 1 : 0,
               !state.preReluOperands.empty() ? 1 : 0,
               !state.clipValueOperands.empty() ? 1 : 0, 1, 1,
               !state.splitOperands.empty() ? 1 : 0,
               !state.loop0SrcStrideOperands.empty() ? 1 : 0,
               !state.loop3CountOperands.empty() ? 1 : 0,
               !state.loop3SrcStrideOperands.empty() ? 1 : 0,
               !state.loop3DstStrideOperands.empty() ? 1 : 0});
  addStructuredAccStoreAttrs<MteL0cGmOp>(result, builder, state);

  if (parser.resolveOperand(source, sourceType, result.operands) ||
      parser.resolveOperand(destination, destinationType, result.operands) ||
      parser.resolveOperand(m, mType, result.operands) ||
      parser.resolveOperand(n, nType, result.operands) ||
      parser.resolveOperand(srcStride, srcStrideType, result.operands) ||
      parser.resolveOperand(dstStride, dstStrideType, result.operands) ||
      parser.resolveOperands(state.preQuantOperands, state.preQuantTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(state.preReluOperands, state.preReluTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(state.clipValueOperands, state.clipValueTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperand(sid, sidType, result.operands) ||
      parser.resolveOperand(l2CacheCtrl, l2CacheCtrlType, result.operands) ||
      parser.resolveOperands(state.splitOperands, state.splitTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(state.loop0SrcStrideOperands,
                             state.loop0SrcStrideTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(state.loop3CountOperands, state.loop3CountTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(state.loop3SrcStrideOperands,
                             state.loop3SrcStrideTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(state.loop3DstStrideOperands,
                             state.loop3DstStrideTypes,
                             parser.getCurrentLocation(), result.operands)) {
    return failure();
  }
  return success();
}

void MteL0cGmOp::print(OpAsmPrinter &printer) {
  printer << " " << getSource() << ", " << getDestination() << ", " << getM()
          << ", " << getN() << ", " << getSrcStride() << ", "
          << getDstStride() << ", " << getSid() << ", " << getL2CacheCtrl();
  printStructuredAccStoreClauses(printer, getUnitFlag(), getPreQuant(),
                                 getPreQuantMode(), getPreRelu(),
                                 getPreReluMode(), getClipValue(), getMode(),
                                 getSplit(), getLoop0SrcStride(),
                                 getLoop3Count(), getLoop3SrcStride(),
                                 getLoop3DstStride(), getSatMode(),
                                 getAtomicType(), getAtomicOp());
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                /*elidedAttrs=*/{"operandSegmentSizes",
                                                 "mode",
                                                 "unit_flag",
                                                 "pre_quant_mode",
                                                 "pre_relu_mode",
                                                 "atomic_type",
                                                 "atomic_op",
                                                 "sat_mode"});
  printer << " : " << getSource().getType() << ", " << getDestination().getType()
          << ", " << getM().getType() << ", " << getN().getType() << ", "
          << getSrcStride().getType() << ", " << getDstStride().getType()
          << ", " << getSid().getType() << ", " << getL2CacheCtrl().getType();
  printStructuredAccStoreOptionalTypes(
      printer, getPreQuant(), getPreRelu(), getClipValue(), getSplit(),
      getLoop0SrcStride(), getLoop3Count(), getLoop3SrcStride(),
      getLoop3DstStride());
}

LogicalResult MteL0cGmOp::verify() {
  if (!isBufferLike(getSource().getType()) ||
      !isBufferLike(getDestination().getType())) {
    return emitOpError("requires buffer-like source and destination");
  }
  std::optional<AddressSpace> sourceSpace =
      getBufferAddressSpace(getSource().getType());
  std::optional<AddressSpace> destinationSpace =
      getBufferAddressSpace(getDestination().getType());
  if (sourceSpace != AddressSpace::ACC || destinationSpace != AddressSpace::GM) {
    return emitOpError("requires ACC source and GM destination");
  }
  return verifyStructuredAccStoreLike(
      *this, getSource().getType(), getDestination().getType(), getPreQuant(), getPreRelu(),
      getClipValue(), getSplit(), getLoop0SrcStride(), getLoop3Count(),
      getLoop3SrcStride(), getLoop3DstStride(), getUnitFlag(),
      getPreQuantMode(), getPreReluMode(), getMode(), getAtomicType(),
      getAtomicOp(), /*allowAtomic=*/true);
}

void MteL0cGmOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

ParseResult MteL0cUbOp::parse(OpAsmParser &parser, OperationState &result) {
  Builder builder(parser.getContext());
  StructuredAccStoreAsmState state;
  OpAsmParser::UnresolvedOperand source, destination, m, n, srcStride,
      dstStride, subBlockId;
  bool hasSubBlockId = false;
  AccStoreUbDstMode dstMode = AccStoreUbDstMode::Single;
  if (parseRequiredOperandWithComma(parser, source) ||
      parseRequiredOperandWithComma(parser, destination) ||
      parseRequiredOperandWithComma(parser, m) ||
      parseRequiredOperandWithComma(parser, n) ||
      parseRequiredOperandWithComma(parser, srcStride) ||
      parseRequiredOperandWithComma(parser, dstStride)) {
    return failure();
  }
  if (parser.parseKeyword("dst_mode") || parser.parseLParen()) {
    return failure();
  }
  OptionalParseResult subBlockIdParse =
      parser.parseOptionalOperand(subBlockId);
  if (subBlockIdParse.has_value()) {
    if (failed(*subBlockIdParse)) {
      return failure();
    }
    hasSubBlockId = true;
  } else {
    StringRef dstModeKeyword;
    if (parser.parseKeyword(&dstModeKeyword)) {
      return failure();
    }
    if (dstModeKeyword == "split_m") {
      dstMode = AccStoreUbDstMode::SplitM;
    } else if (dstModeKeyword == "split_n") {
      dstMode = AccStoreUbDstMode::SplitN;
    } else {
      return parser.emitError(
          parser.getCurrentLocation(),
          "expected dst_mode(%sub_blockid), dst_mode(split_m), or "
          "dst_mode(split_n)");
    }
  }
  if (parser.parseRParen()) {
    return failure();
  }
  if (succeeded(parser.parseOptionalComma()) &&
      parseStructuredAccStoreClauses(parser, state)) {
    return failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon()) {
    return failure();
  }

  Type sourceType, destinationType, mType, nType, srcStrideType, dstStrideType,
      subBlockIdType;
  if (parser.parseType(sourceType) || parser.parseComma() ||
      parser.parseType(destinationType) || parser.parseComma() ||
      parser.parseType(mType) || parser.parseComma() || parser.parseType(nType) ||
      parser.parseComma() || parser.parseType(srcStrideType) ||
      parser.parseComma() || parser.parseType(dstStrideType)) {
    return failure();
  }
  if (hasSubBlockId &&
      (parser.parseComma() || parser.parseType(subBlockIdType))) {
    return failure();
  }
  if (parseStructuredAccStoreTailTypes(parser, state)) {
    return failure();
  }

  setStructuredAccStoreSegmentSizes<MteL0cUbOp>(
      result, {1, 1, 1, 1, 1, 1, !state.preQuantOperands.empty() ? 1 : 0,
               !state.preReluOperands.empty() ? 1 : 0,
               !state.clipValueOperands.empty() ? 1 : 0,
               hasSubBlockId ? 1 : 0,
               !state.splitOperands.empty() ? 1 : 0,
               !state.loop0SrcStrideOperands.empty() ? 1 : 0,
               !state.loop3CountOperands.empty() ? 1 : 0,
               !state.loop3SrcStrideOperands.empty() ? 1 : 0,
               !state.loop3DstStrideOperands.empty() ? 1 : 0});
  if (state.atomicType || state.atomicOp) {
    return parser.emitError(parser.getCurrentLocation(),
                            "atomic is only supported for mte_l0c_gm");
  }
  addStructuredAccStoreAttrs<MteL0cUbOp>(result, builder, state);
  result.addAttribute("dst_mode",
                      AccStoreUbDstModeAttr::get(builder.getContext(), dstMode));

  if (parser.resolveOperand(source, sourceType, result.operands) ||
      parser.resolveOperand(destination, destinationType, result.operands) ||
      parser.resolveOperand(m, mType, result.operands) ||
      parser.resolveOperand(n, nType, result.operands) ||
      parser.resolveOperand(srcStride, srcStrideType, result.operands) ||
      parser.resolveOperand(dstStride, dstStrideType, result.operands) ||
      parser.resolveOperands(state.preQuantOperands, state.preQuantTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(state.preReluOperands, state.preReluTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(state.clipValueOperands, state.clipValueTypes,
                             parser.getCurrentLocation(), result.operands) ||
      (hasSubBlockId &&
       parser.resolveOperand(subBlockId, subBlockIdType, result.operands)) ||
      parser.resolveOperands(state.splitOperands, state.splitTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(state.loop0SrcStrideOperands,
                             state.loop0SrcStrideTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(state.loop3CountOperands, state.loop3CountTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(state.loop3SrcStrideOperands,
                             state.loop3SrcStrideTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(state.loop3DstStrideOperands,
                             state.loop3DstStrideTypes,
                             parser.getCurrentLocation(), result.operands)) {
    return failure();
  }
  return success();
}

void MteL0cUbOp::print(OpAsmPrinter &printer) {
  printer << " " << getSource() << ", " << getDestination() << ", " << getM()
          << ", " << getN() << ", " << getSrcStride() << ", "
          << getDstStride() << ", dst_mode(";
  switch (getDstMode()) {
  case AccStoreUbDstMode::Single:
    printer << getSubBlockid();
    break;
  case AccStoreUbDstMode::SplitM:
    printer << "split_m";
    break;
  case AccStoreUbDstMode::SplitN:
    printer << "split_n";
    break;
  }
  printer << ")";
  printStructuredAccStoreClauses(printer, getUnitFlag(), getPreQuant(),
                                 getPreQuantMode(), getPreRelu(),
                                 getPreReluMode(), getClipValue(), getMode(),
                                 getSplit(), getLoop0SrcStride(),
                                 getLoop3Count(), getLoop3SrcStride(),
                                 getLoop3DstStride(), getSatMode(),
                                 std::nullopt, std::nullopt);
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                /*elidedAttrs=*/{"operandSegmentSizes",
                                                 "mode",
                                                 "unit_flag",
                                                 "pre_quant_mode",
                                                 "pre_relu_mode",
                                                 "dst_mode",
                                                 "sat_mode"});
  printer << " : " << getSource().getType() << ", " << getDestination().getType()
          << ", " << getM().getType() << ", " << getN().getType() << ", "
          << getSrcStride().getType() << ", " << getDstStride().getType();
  if (getSubBlockid()) {
    printer << ", " << getSubBlockid().getType();
  }
  printStructuredAccStoreOptionalTypes(
      printer, getPreQuant(), getPreRelu(), getClipValue(), getSplit(),
      getLoop0SrcStride(), getLoop3Count(), getLoop3SrcStride(),
      getLoop3DstStride());
}

LogicalResult MteL0cUbOp::verify() {
  if (!isBufferLike(getSource().getType()) ||
      !isBufferLike(getDestination().getType())) {
    return emitOpError("requires buffer-like source and destination");
  }
  std::optional<AddressSpace> sourceSpace =
      getBufferAddressSpace(getSource().getType());
  std::optional<AddressSpace> destinationSpace =
      getBufferAddressSpace(getDestination().getType());
  if (sourceSpace != AddressSpace::ACC || destinationSpace != AddressSpace::VEC) {
    return emitOpError("requires ACC source and UB destination");
  }
  if (failed(verifyStructuredAccStoreLike(
          *this, getSource().getType(), getDestination().getType(), getPreQuant(), getPreRelu(),
          getClipValue(), getSplit(), getLoop0SrcStride(), getLoop3Count(),
          getLoop3SrcStride(), getLoop3DstStride(), getUnitFlag(),
          getPreQuantMode(), getPreReluMode(), getMode(), std::nullopt,
          std::nullopt, /*allowAtomic=*/false))) {
    return failure();
  }

  if (getDstMode() == AccStoreUbDstMode::Single) {
    if (!getSubBlockid()) {
      return emitOpError("dst_mode(%sub_blockid) requires a sub_blockid operand");
    }
    APInt subBlockId;
    if (matchPattern(getSubBlockid(), m_ConstantInt(&subBlockId)) &&
        subBlockId.ugt(1)) {
      return emitOpError("sub_blockid must be 0 or 1");
    }
    return success();
  }
  if (getSubBlockid()) {
    return emitOpError("split destination modes do not accept sub_blockid");
  }

  if (getPreQuant() || getPreRelu() || getClipValue() || getPreQuantMode() ||
      getPreReluMode() || getSplit() || getLoop0SrcStride() ||
      getLoop3Count() || getLoop3SrcStride() || getLoop3DstStride()) {
    return emitOpError("dual destination mode cannot be combined with "
                       "pre_quant, pre_relu, clip, nz2dn, nz2nz, or loop3");
  }
  if (getMode() && *getMode() != AccStoreMode::Nz2nd) {
    return emitOpError("dual destination mode requires normal or nz2nd layout");
  }

  APInt mValue;
  APInt nValue;
  if (getDstMode() == AccStoreUbDstMode::SplitM &&
      matchPattern(getM(), m_ConstantInt(&mValue)) &&
      mValue.getZExtValue() % mlir::pto::kValue2 != 0) {
    return emitOpError("split-M dual destination requires m to be even");
  }
  if (getDstMode() == AccStoreUbDstMode::SplitN &&
      matchPattern(getN(), m_ConstantInt(&nValue)) &&
      nValue.getZExtValue() % mlir::pto::kValue32 != 0) {
    return emitOpError("split-N dual destination requires n to be a multiple of 32");
  }
  return success();
}

void MteL0cUbOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

//===----------------------------------------------------------------------===//
// UBVaddOp
//===----------------------------------------------------------------------===//

void UBVaddOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrc0Mutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getSrc1Mutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult UBVaddOp::verify() {
  if (!isBufferLike(getDst().getType()) || !isBufferLike(getSrc0().getType()) ||
      !isBufferLike(getSrc1().getType())) {
    return emitOpError("requires pointer-like operands");
  }
  if (classifyMemoryRole(getDst().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSrc0().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSrc1().getType()) != MemoryRole::UB) {
    return emitOpError("requires UB-backed operands");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// UBVsubOp
//===----------------------------------------------------------------------===//

void UBVsubOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrc0Mutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getSrc1Mutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult UBVsubOp::verify() {
  if (!isBufferLike(getDst().getType()) || !isBufferLike(getSrc0().getType()) ||
      !isBufferLike(getSrc1().getType())) {
    return emitOpError("requires pointer-like operands");
  }
  if (classifyMemoryRole(getDst().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSrc0().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSrc1().getType()) != MemoryRole::UB) {
    return emitOpError("requires UB-backed operands");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// UBVmulOp
//===----------------------------------------------------------------------===//

void UBVmulOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrc0Mutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getSrc1Mutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult UBVmulOp::verify() {
  if (!isBufferLike(getDst().getType()) || !isBufferLike(getSrc0().getType()) ||
      !isBufferLike(getSrc1().getType())) {
    return emitOpError("requires pointer-like operands");
  }
  if (classifyMemoryRole(getDst().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSrc0().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSrc1().getType()) != MemoryRole::UB) {
    return emitOpError("requires UB-backed operands");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// UBVdivOp
//===----------------------------------------------------------------------===//

void UBVdivOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrc0Mutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getSrc1Mutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult UBVdivOp::verify() {
  if (!isBufferLike(getDst().getType()) || !isBufferLike(getSrc0().getType()) ||
      !isBufferLike(getSrc1().getType())) {
    return emitOpError("requires pointer-like operands");
  }
  if (classifyMemoryRole(getDst().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSrc0().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSrc1().getType()) != MemoryRole::UB) {
    return emitOpError("requires UB-backed operands");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// UBVmaxOp
//===----------------------------------------------------------------------===//

void UBVmaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrc0Mutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getSrc1Mutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult UBVmaxOp::verify() {
  if (!isBufferLike(getDst().getType()) || !isBufferLike(getSrc0().getType()) ||
      !isBufferLike(getSrc1().getType())) {
    return emitOpError("requires pointer-like operands");
  }
  if (classifyMemoryRole(getDst().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSrc0().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSrc1().getType()) != MemoryRole::UB) {
    return emitOpError("requires UB-backed operands");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// UBVminOp
//===----------------------------------------------------------------------===//

void UBVminOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrc0Mutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getSrc1Mutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult UBVminOp::verify() {
  if (!isBufferLike(getDst().getType()) || !isBufferLike(getSrc0().getType()) ||
      !isBufferLike(getSrc1().getType())) {
    return emitOpError("requires pointer-like operands");
  }
  if (classifyMemoryRole(getDst().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSrc0().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSrc1().getType()) != MemoryRole::UB) {
    return emitOpError("requires UB-backed operands");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// UBVandOp
//===----------------------------------------------------------------------===//

void UBVandOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrc0Mutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getSrc1Mutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult UBVandOp::verify() {
  if (!isBufferLike(getDst().getType()) || !isBufferLike(getSrc0().getType()) ||
      !isBufferLike(getSrc1().getType())) {
    return emitOpError("requires pointer-like operands");
  }
  if (classifyMemoryRole(getDst().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSrc0().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSrc1().getType()) != MemoryRole::UB) {
    return emitOpError("requires UB-backed operands");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// UBVorOp
//===----------------------------------------------------------------------===//

void UBVorOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrc0Mutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getSrc1Mutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult UBVorOp::verify() {
  if (!isBufferLike(getDst().getType()) || !isBufferLike(getSrc0().getType()) ||
      !isBufferLike(getSrc1().getType())) {
    return emitOpError("requires pointer-like operands");
  }
  if (classifyMemoryRole(getDst().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSrc0().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSrc1().getType()) != MemoryRole::UB) {
    return emitOpError("requires UB-backed operands");
  }
  return success();
}

#define PTO_DEFINE_UB_BINARY_VERIFY_AND_EFFECTS(OpName)                   \
  void OpName::getEffects(                                                \
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> \
          &effects) {                                                     \
    effects.emplace_back(MemoryEffects::Read::get(), &getSrc0Mutable());  \
    effects.emplace_back(MemoryEffects::Read::get(), &getSrc1Mutable());  \
    effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());  \
  }                                                                       \
  LogicalResult OpName::verify() {                                        \
    if (!isBufferLike(getDst().getType()) ||                              \
        !isBufferLike(getSrc0().getType()) ||                             \
        !isBufferLike(getSrc1().getType())) {                             \
      return emitOpError("requires pointer-like operands");               \
    }                                                                     \
    if (classifyMemoryRole(getDst().getType()) != MemoryRole::UB ||       \
        classifyMemoryRole(getSrc0().getType()) != MemoryRole::UB ||      \
        classifyMemoryRole(getSrc1().getType()) != MemoryRole::UB) {      \
      return emitOpError("requires UB-backed operands");                  \
    }                                                                     \
    return success();                                                     \
  }

PTO_DEFINE_UB_BINARY_VERIFY_AND_EFFECTS(UBVaddReluOp)

//===----------------------------------------------------------------------===//
// UBVnotOp
//===----------------------------------------------------------------------===//

void UBVnotOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult UBVnotOp::verify() {
  if (!isBufferLike(getDst().getType()) || !isBufferLike(getSrc().getType())) {
    return emitOpError("requires pointer-like operands");
  }
  if (classifyMemoryRole(getDst().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSrc().getType()) != MemoryRole::UB) {
    return emitOpError("requires UB-backed operands");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// UBVabsOp
//===----------------------------------------------------------------------===//

void UBVabsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult UBVabsOp::verify() {
  if (!isBufferLike(getDst().getType()) || !isBufferLike(getSrc().getType())) {
    return emitOpError("requires pointer-like operands");
  }
  if (classifyMemoryRole(getDst().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSrc().getType()) != MemoryRole::UB) {
    return emitOpError("requires UB-backed operands");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// UBVreluOp
//===----------------------------------------------------------------------===//

void UBVreluOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult UBVreluOp::verify() {
  if (!isBufferLike(getDst().getType()) || !isBufferLike(getSrc().getType())) {
    return emitOpError("requires pointer-like operands");
  }
  if (classifyMemoryRole(getDst().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSrc().getType()) != MemoryRole::UB) {
    return emitOpError("requires UB-backed operands");
  }
  return success();
}

#define PTO_DEFINE_UB_UNARY_VERIFY_AND_EFFECTS(OpName)                    \
  void OpName::getEffects(                                                \
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> \
          &effects) {                                                     \
    effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable());   \
    effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());  \
  }                                                                       \
  LogicalResult OpName::verify() {                                        \
    if (!isBufferLike(getDst().getType()) ||                              \
        !isBufferLike(getSrc().getType())) {                              \
      return emitOpError("requires pointer-like operands");               \
    }                                                                     \
    if (classifyMemoryRole(getDst().getType()) != MemoryRole::UB ||       \
        classifyMemoryRole(getSrc().getType()) != MemoryRole::UB) {       \
      return emitOpError("requires UB-backed operands");                  \
    }                                                                     \
    return success();                                                     \
  }

PTO_DEFINE_UB_UNARY_VERIFY_AND_EFFECTS(UBVexpOp)
PTO_DEFINE_UB_UNARY_VERIFY_AND_EFFECTS(UBVlnOp)
PTO_DEFINE_UB_UNARY_VERIFY_AND_EFFECTS(UBVsqrtOp)
PTO_DEFINE_UB_UNARY_VERIFY_AND_EFFECTS(UBVrsqrtOp)

PTO_DEFINE_UB_UNARY_VERIFY_AND_EFFECTS(UBVaddSOp)
PTO_DEFINE_UB_UNARY_VERIFY_AND_EFFECTS(UBVmaxSOp)
PTO_DEFINE_UB_UNARY_VERIFY_AND_EFFECTS(UBVminSOp)

void UBVdupOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult UBVdupOp::verify() {
  if (!isBufferLike(getDst().getType())) {
    return emitOpError("requires pointer-like dst operand");
  }
  if (classifyMemoryRole(getDst().getType()) != MemoryRole::UB) {
    return emitOpError("requires UB-backed dst operand");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// UBVgatherbOp
//===----------------------------------------------------------------------===//

void UBVgatherbOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getOffsetMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult UBVgatherbOp::verify() {
  if (!isBufferLike(getDst().getType()) || !isBufferLike(getOffset().getType()) ||
      !isBufferLike(getSrc().getType())) {
    return emitOpError("requires pointer-like operands");
  }
  if (classifyMemoryRole(getDst().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getOffset().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSrc().getType()) != MemoryRole::UB) {
    return emitOpError("requires UB-backed operands");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// UBVgatherOp
//===----------------------------------------------------------------------===//

void UBVgatherOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult UBVgatherOp::verify() {
  if (!isBufferLike(getDst().getType()) || !isBufferLike(getSrc().getType())) {
    return emitOpError("requires pointer-like operands");
  }
  if (classifyMemoryRole(getDst().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSrc().getType()) != MemoryRole::UB) {
    return emitOpError("requires UB-backed operands");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// UBVshlOp
//===----------------------------------------------------------------------===//

void UBVshlOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult UBVshlOp::verify() {
  if (!isBufferLike(getDst().getType()) || !isBufferLike(getSrc().getType())) {
    return emitOpError("requires pointer-like operands");
  }
  if (classifyMemoryRole(getDst().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSrc().getType()) != MemoryRole::UB) {
    return emitOpError("requires UB-backed operands");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// UBVshrOp
//===----------------------------------------------------------------------===//

void UBVshrOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult UBVshrOp::verify() {
  if (!isBufferLike(getDst().getType()) || !isBufferLike(getSrc().getType())) {
    return emitOpError("requires pointer-like operands");
  }
  if (classifyMemoryRole(getDst().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSrc().getType()) != MemoryRole::UB) {
    return emitOpError("requires UB-backed operands");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// UBVmulSOp
//===----------------------------------------------------------------------===//

void UBVmulSOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult UBVmulSOp::verify() {
  if (!isBufferLike(getDst().getType()) || !isBufferLike(getSrc().getType())) {
    return emitOpError("requires pointer-like operands");
  }
  if (classifyMemoryRole(getDst().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSrc().getType()) != MemoryRole::UB) {
    return emitOpError("requires UB-backed operands");
  }
  return success();
}

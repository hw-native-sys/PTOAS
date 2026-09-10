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

#include "VPTOInternal.h"

using namespace mlir;
using namespace mlir::pto;

llvm::cl::opt<bool> disableVPTOAlignChainVerification(
    "vpto-disable-align-chain-verification",
    llvm::cl::desc("Disable !pto.align linear-chain verifier checks"),
    llvm::cl::init(false), llvm::cl::Hidden);

std::string formatVRegType(int64_t elementCount, Type elementType) {
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

LogicalResult verifyMaskTypeLike(Operation *op, Type type,
                                        StringRef roleDescription) {
  if (!isa<MaskType>(type)) {
    return op->emitOpError() << roleDescription << " must be !pto.mask<...>";
  }
  return success();
}

LogicalResult verifyNonLowPrecisionVRegElementTypeLike(
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

LogicalResult verifyMaskTypeWithGranularityLike(Operation *op, Type type,
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

static bool isStandardScalarConvertType(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    return intType.getWidth() == mlir::pto::kValue32 || intType.getWidth() == 64;
  }
  return type.isF16() || type.isBF16() || type.isF32();
}

static bool isIntegerLikeConvertType(Type type) {
  return isa<IntegerType>(type);
}

bool isVector2Of(Type type, llvm::function_ref<bool(Type)> elementPred) {
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

bool isInsideSimtExecutionScope(Operation *op) {
  auto funcOp = op->getParentOfType<func::FuncOp>();
  return (funcOp && funcOp->hasAttr(pto::kPTOSimtEntryAttrName)) ||
         op->getParentOfType<pto::SectionSimtOp>();
}

static bool isSupportedConvertType(Type type) {
  return isStandardScalarConvertType(type) || isSupportedPackedConvertType(type) ||
         isSupportedLowPrecisionConvertType(type);
}

static bool isV2F16(Type type) {
  return isVector2Of(type, [](Type elem) { return elem.isF16(); });
}
static bool isV2BF16(Type type) {
  return isVector2Of(type, [](Type elem) { return elem.isBF16(); });
}
static bool isV2F32(Type type) {
  return isVector2Of(type, [](Type elem) { return elem.isF32(); });
}
static bool isV2F8(Type type) {
  return isVector2Of(type, [](Type elem) { return pto::isPTOFloat8Type(elem); });
}
static bool isV2HiF8(Type type) { return pto::isPTOHiFloat8x2Type(type); }
static bool isF4(Type type) { return pto::isPTOFloat4PackedType(type); }
static bool isRoundRAFZC(pto::Rounding rounding) {
  return rounding == pto::Rounding::R || rounding == pto::Rounding::A ||
         rounding == pto::Rounding::F || rounding == pto::Rounding::C ||
         rounding == pto::Rounding::Z;
}

static LogicalResult verifyF32x2ToF16x2Pair(Operation *op,
                                                    pto::Rounding rounding) {
  if (rounding == pto::Rounding::H) {
    return op->emitOpError()
           << "f32x2-to-f16x2 conversion supports rounding r/a/f/c/z/o";
  }
  return success();
}

static LogicalResult verifyF32x2ToBF16x2Pair(Operation *op,
                                             pto::Rounding rounding) {
  if (rounding == pto::Rounding::O || rounding == pto::Rounding::H) {
    return op->emitOpError()
           << "f32x2-to-bf16x2 conversion supports rounding r/a/f/c/z";
  }
  return success();
}

static LogicalResult verifyPackedToF32x2Pair(Operation *op,
                                             pto::Rounding rounding) {
  if (rounding == pto::Rounding::O || rounding == pto::Rounding::H) {
    return op->emitOpError()
           << "packed-to-f32x2 conversion supports rounding r/a/f/c/z";
  }
  return success();
}

static LogicalResult verifyF32x2ToF8x2Pair(Operation *op,
                                           pto::Rounding rounding) {
  if (rounding != pto::Rounding::R) {
    return op->emitOpError()
           << "f32x2-to-f8x2 conversion supports rounding r";
  }
  return success();
}

static LogicalResult verifyToHiF8Pair(Operation *op, pto::Rounding rounding) {
  if (rounding != pto::Rounding::A && rounding != pto::Rounding::H) {
    return op->emitOpError()
           << "f32x2/f16x2-to-hif8x2 conversion supports rounding a/h";
  }
  return success();
}

static LogicalResult verifyFromF8HiF8Pair(Operation *op,
                                          pto::Rounding rounding) {
  if (!isRoundRAFZC(rounding)) {
    return op->emitOpError()
           << "f8x2/hif8x2-to-f32x2/f16x2 conversion supports rounding r/a/f/c/z";
  }
  return success();
}

static LogicalResult verifyBF16F4Pair(Operation *op, pto::Rounding rounding) {
  if (!isRoundRAFZC(rounding)) {
    return op->emitOpError()
           << "bf16x2-to-f4 and f4-to-bf16x2 conversion supports rounding r/a/f/c/z";
  }
  return success();
}

static LogicalResult verifyPackedConvertPair(Operation *op, Type srcType,
                                             Type dstType,
                                             pto::Rounding rounding) {
  if (isV2F32(srcType) && isV2F16(dstType)) {
    return verifyF32x2ToF16x2Pair(op, rounding);
  }
  if (isV2F32(srcType) && isV2BF16(dstType)) {
    return verifyF32x2ToBF16x2Pair(op, rounding);
  }
  if ((isV2F16(srcType) || isV2BF16(srcType)) && isV2F32(dstType)) {
    return verifyPackedToF32x2Pair(op, rounding);
  }
  if (isV2F32(srcType) && isV2F8(dstType)) {
    return verifyF32x2ToF8x2Pair(op, rounding);
  }
  if ((isV2F32(srcType) || isV2F16(srcType)) && isV2HiF8(dstType)) {
    return verifyToHiF8Pair(op, rounding);
  }
  if ((isV2F8(srcType) || isV2HiF8(srcType)) &&
      (isV2F32(dstType) || isV2F16(dstType))) {
    return verifyFromF8HiF8Pair(op, rounding);
  }
  bool isBF16F4 = (isV2BF16(srcType) && isF4(dstType)) ||
                  (isF4(srcType) && isV2BF16(dstType));
  if (isBF16F4) {
    return verifyBF16F4Pair(op, rounding);
  }
  return op->emitOpError()
         << "unsupported packed conversion type pair; supported packed pairs are "
            "f32x2-to-f16x2, f16x2-to-f32x2, f32x2-to-bf16x2, and "
            "bf16x2-to-f32x2, f32x2-to-f8x2, f32x2/f16x2-to-hif8x2, "
            "f8x2/hif8x2-to-f32x2/f16x2, and bf16x2-to/from-f4";
}

static LogicalResult verifyPackedConvertControls(Operation *op, Type srcType,
                                                 Type dstType,
                                                 pto::Rounding rounding) {
  return verifyPackedConvertPair(op, srcType, dstType, rounding);
}

static LogicalResult verifyPackedOrLowPrecisionConvert(
    Operation *op, Type srcType, Type dstType, pto::Rounding rounding,
    bool srcInt, bool dstInt, Attribute signednessAttr, bool srcPacked,
    bool dstPacked, bool srcLowPrecision, bool dstLowPrecision) {
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

static LogicalResult verifyConvertSignedness(Operation *op, bool srcInt,
                                             bool dstInt,
                                             Attribute signednessAttr) {
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
  return success();
}

static LogicalResult verifyIntToFloatConvert(Operation *op, Type srcType,
                                             Type dstType,
                                             pto::Rounding rounding,
                                             pto::Saturation saturation) {
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

static LogicalResult verifyF32ConvertTarget(Operation *op, Type dstType,
                                            pto::Rounding rounding,
                                            pto::Saturation saturation) {
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
  return op->emitOpError() << "unsupported conversion type pair";
}

static LogicalResult verifyF16OrBF16ConvertTarget(
    Operation *op, Type dstType, pto::Rounding rounding,
    pto::Saturation saturation) {
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
  return op->emitOpError() << "unsupported conversion type pair";
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
    return verifyPackedOrLowPrecisionConvert(
        op, srcType, dstType, rounding, srcInt, dstInt, signednessAttr,
        srcPacked, dstPacked, srcLowPrecision, dstLowPrecision);
  }
  if (failed(verifyConvertSignedness(op, srcInt, dstInt, signednessAttr))) {
    return failure();
  }
  if (srcInt) {
    return verifyIntToFloatConvert(op, srcType, dstType, rounding, saturation);
  }
  if (dstType.isInteger(mlir::pto::kValue64) && !srcType.isF32()) {
    return op->emitOpError()
           << "supports conversion to i64 only from f32 in the confirmed slice";
  }
  if (srcType.isF32()) {
    return verifyF32ConvertTarget(op, dstType, rounding, saturation);
  }
  if (srcType.isF16() || srcType.isBF16()) {
    return verifyF16OrBF16ConvertTarget(op, dstType, rounding, saturation);
  }
  return op->emitOpError() << "unsupported conversion type pair";
}

LogicalResult ConvertOp::verify() {
  return verifyConvertControls(getOperation(), getSrc().getType(),
                               getDst().getType(), getRounding(),
                               getSaturation(), getSignednessAttr());
}

LogicalResult verifyNotNestedInVecScope(Operation *op,
                                               StringRef opNameForDiag) {
  if (op->getParentOfType<VecScopeOp>() ||
      op->getParentOfType<StrictVecScopeOp>()) {
    return op->emitOpError()
           << "must not be nested under pto.vecscope/pto.strict_vecscope; "
           << opNameForDiag << " is a UB helper op rather than a vecscope op";
  }
  return success();
}

LogicalResult verifyNestedInVecScope(Operation *op,
                                            StringRef opNameForDiag) {
  if (op->getParentOfType<VecScopeOp>() || op->getParentOfType<StrictVecScopeOp>()) {
    return success();
  }
  return op->emitOpError()
         << "must be nested under pto.vecscope/pto.strict_vecscope; "
         << opNameForDiag << " is part of the vecscope control sequence";
}

std::optional<StringRef> normalizeRoundModeToken(StringRef token) {
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

std::optional<StringRef> normalizeSaturationToken(StringRef token) {
  if (token == "SAT" || token == "RS_ENABLE") {
    return StringRef("SAT");
  }
  if (token == "NOSAT" || token == "RS_DISABLE") {
    return StringRef("NOSAT");
  }
  return std::nullopt;
}

unsigned getIntOrFloatBitWidth(Type type) {
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

bool isIntegerOrFloatLike(Type type) {
  return isa<IntegerType>(type) || isa<FloatType>(type);
}

std::optional<int64_t> getVRegStorageBitWidth(Type type) {
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

LogicalResult verifyIntegerVRegTypeLike(Operation *op, Type type,
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

static LogicalResult verifyVmrgsort4BufferRoles(
    Operation *op, Value destination, Value source0, Value source1,
    Value source2, Value source3) {
  if (!isBufferLike(destination.getType()) || !isBufferLike(source0.getType()) ||
      !isBufferLike(source1.getType()) || !isBufferLike(source2.getType()) ||
      !isBufferLike(source3.getType())) {
    return op->emitOpError("requires pointer-like destination and sources");
  }
  if (classifyMemoryRole(destination.getType()) != MemoryRole::UB ||
      classifyMemoryRole(source0.getType()) != MemoryRole::UB ||
      classifyMemoryRole(source1.getType()) != MemoryRole::UB ||
      classifyMemoryRole(source2.getType()) != MemoryRole::UB ||
      classifyMemoryRole(source3.getType()) != MemoryRole::UB) {
    return op->emitOpError("requires UB-backed destination and sources");
  }
  return success();
}

static LogicalResult verifyVmrgsort4PtrAndElementTypes(
    Operation *op, Value destination, Value source0, Value source1,
    Value source2, Value source3) {
  auto dstPtrType = dyn_cast<pto::PtrType>(destination.getType());
  auto src0PtrType = dyn_cast<pto::PtrType>(source0.getType());
  auto src1PtrType = dyn_cast<pto::PtrType>(source1.getType());
  auto src2PtrType = dyn_cast<pto::PtrType>(source2.getType());
  auto src3PtrType = dyn_cast<pto::PtrType>(source3.getType());
  if (!dstPtrType || !src0PtrType || !src1PtrType || !src2PtrType ||
      !src3PtrType) {
    return op->emitOpError("requires ptr-backed destination and sources");
  }
  Type elemType = dstPtrType.getElementType();
  if (src0PtrType.getElementType() != elemType ||
      src1PtrType.getElementType() != elemType ||
      src2PtrType.getElementType() != elemType ||
      src3PtrType.getElementType() != elemType) {
    return op->emitOpError(
        "requires destination and all sources to have the same element type");
  }
  if (!elemType.isF16() && !elemType.isF32()) {
    return op->emitOpError("requires f16 or f32 element type");
  }
  return success();
}

LogicalResult Vmrgsort4Op::verify() {
  if (failed(verifyVmrgsort4BufferRoles(*this, getDestination(), getSource0(),
                                        getSource1(), getSource2(),
                                        getSource3())) ||
      failed(verifyVmrgsort4PtrAndElementTypes(
          *this, getDestination(), getSource0(), getSource1(), getSource2(),
          getSource3()))) {
    return failure();
  }
  if (failed(verifyNotNestedInVecScope(*this, "pto.vmrgsort4"))) {
    return failure();
  }
  return success();
}

LogicalResult InitAlignOp::verify() {
  return verifyAlignTypeLike(*this, getResult().getType(), "result type");
}

ParseResult normalizeNamedStringAttr(
    OpAsmParser &parser, NamedAttrList &attrs, StringRef sourceName,
    StringRef canonicalName,
    std::optional<StringRef> (*normalizeFn)(StringRef)) {
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
}

StringRef getAddressSpaceDiagnosticName(pto::AddressSpace space) {
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

LogicalResult checkConstMax(Operation *op, Value value, StringRef name,
                                   uint64_t max) {
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
}

LogicalResult checkConstAlignment(Operation *op, Value value,
                                         StringRef name,
                                         uint64_t alignment) {
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
}

LogicalResult verifyMxLoadOperands(Operation *op,
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
    return op->emitOpError() << "cannot mix shape-derived MX operands with full MX operands";
  }
  if (!hasShape && !hasFull) {
    return op->emitOpError() << "requires either all shape-derived MX operands or all full MX operands";
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

LogicalResult verifyMxLoadAlignment(Operation *op, Value source,
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

MemoryRole classifyMemoryRole(Type type) {
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

bool isForbiddenSynchronizationInsideVecScope(Operation *op) {
  // High-level synchronization is forbidden before and after lowering.
  if (isa<RecordEventOp, WaitEventOp, BarrierSyncOp>(op)) {
    return true;
  }

  // Intra-core pipeline and buffer-id synchronization executes outside the
  // vector interval that it orders.
  if (isa<SetFlagOp, WaitFlagOp, SetFlagDynOp, WaitFlagDynOp, GetBufOp,
          GetBufDynOp, RlsBufOp, RlsBufDynOp, BarrierOp>(op)) {
    return true;
  }

  // Intra-block, cross-core, system, cache, and SIMT synchronization likewise
  // delimit vector intervals. MemBarOp is intentionally not in this list.
  return isa<SyncSetOp, SyncWaitOp, CmoCacheInvalidOp, FenceBarrierAllOp,
             TSyncOp, SyncAllOp, DsbOp, DcciOp, SyncthreadsOp, ThreadfenceOp,
             ThreadfenceBlockOp>(op);
}

Operation *findForbiddenSyncInRegion(Region &body) {
  Operation *boundaryOp = nullptr;
  body.walk([&](Operation *op) {
    if (!isForbiddenSynchronizationInsideVecScope(op)) {
      return WalkResult::advance();
    }
    boundaryOp = op;
    return WalkResult::interrupt();
  });
  return boundaryOp;
}

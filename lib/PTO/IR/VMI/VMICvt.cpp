// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMI_ops_cvt.cpp - VMI convert ops -----------------------===//
//===----------------------------------------------------------------------===//

#include "VMIInternal.h"

using namespace mlir;
using namespace mlir::pto;

//===----------------------------------------------------------------------===//
// Group 7: SFU verifiers
//===----------------------------------------------------------------------===//

/// A5 vcvtff only widens a 16-bit source to f32 and an 8-bit source to f32
/// (plus the packed f4x2 -> bf16x2 pair).  An 8-bit source has no direct f16 or
/// bf16 form: the hardware reaches it through f32, so reject the pair here
/// instead of lowering it to a pto.vcvt that the LLVM emitter cannot legalize.
///
/// Note that pto.vmi.extf is not another name for pto.vmi.vcvt: the unified
/// vcvt is the authored op and vmi-lower-unified-to-legacy expands it (the
/// fp8 -> f16/bf16 pair becomes extf f8 -> f32 plus truncf f32 -> f16/bf16).
/// extf is the concrete post-expansion op with a one-to-one hardware mapping, so
/// a pair with no single instruction is out of its domain by construction.
static bool isSupportedVMIWideningPair(Type srcElem, Type dstElem) {
  if (pto::isPTOFloat4PackedType(srcElem) && pto::isPTOBF16x2Type(dstElem)) {
    return true;
  }
  unsigned srcBits = pto::getPTOStorageElemBitWidth(srcElem);
  unsigned dstBits = pto::getPTOStorageElemBitWidth(dstElem);
  if (srcBits == mlir::pto::kValue16 && dstBits == mlir::pto::kValue32) {
    return (srcElem.isF16() || srcElem.isBF16()) && dstElem.isF32();
  }
  if (srcBits == mlir::pto::kValue8 && dstBits == mlir::pto::kValue32) {
    return (pto::isPTOFloat8E4M3LikeType(srcElem) ||
            pto::isPTOFloat8E5M2LikeType(srcElem) ||
            pto::isPTOHiFloat8Type(srcElem)) &&
           dstElem.isF32();
  }
  return false;
}

/// Every VMI conversion carries the same optional conversion predicate: an
/// authored mask must describe the source data layout, because a predicate lane
/// and a source lane are one lane (verifyMaskMatchesData).  The operand is
/// optional, so an unmasked cast has nothing to verify.
template <typename OpTy>
static LogicalResult verifyConversionMask(OpTy op, VMIVRegType sourceType)
{
    Value mask = op.getMask();
    if (!mask) {
        return success();
    }
    return verifyMaskMatchesData(op.getOperation(), cast<VMIMaskType>(mask.getType()), sourceType);
}

/// The prologue every VMI cast verifier shares: type the operands, verify an
/// optional conversion mask against the source layout, and require matching
/// logical lane counts.
template <typename OpTy>
static LogicalResult verifyCastOperands(OpTy op, VMIVRegType& sourceType, VMIVRegType& resultType)
{
    sourceType = cast<VMIVRegType>(op.getSource().getType());
    if (failed(verifyConversionMask(op, sourceType))) {
        return failure();
    }
    resultType = cast<VMIVRegType>(op.getResult().getType());
    bool hasMatchingLaneCount = sourceType.getElementCount() == resultType.getElementCount();
    if (!hasMatchingLaneCount) {
        return op.emitOpError("requires source and result logical lane counts to match");
    }
    return success();
}

LogicalResult VMIExtFOp::verify() {
    VMIVRegType sourceType, resultType;
    if (failed(verifyCastOperands(*this, sourceType, resultType))) {
        return failure();
    }
  Type sourceElementType = sourceType.getElementType();
  Type resultElementType = resultType.getElementType();
  if (!isVMIFloatLikeType(sourceElementType) ||
      !isVMIFloatLikeType(resultElementType)) {
    return emitOpError(
        "requires floating-point-like source and result element types");
  }
  if (involvesBF16x2(sourceElementType, resultElementType) &&
      !lookupVMIFpToFpContract(sourceElementType, resultElementType)) {
    return emitOpError(
        "unsupported bf16x2 fp-to-fp conversion element type pair");
}
  if (getVMIElementBitWidth(sourceElementType) >=
      getVMIElementBitWidth(resultElementType)) {
    return emitOpError(
        "requires result element type to be wider than source element type");
}
  if (!isSupportedVMIWideningPair(sourceElementType, resultElementType)) {
    return emitOpError()
           << "unsupported fp widening type pair on A5: " << sourceElementType
           << " -> " << resultElementType
           << "; A5 vcvtff only widens f16/bf16 -> f32 and float8 -> f32, so an "
              "8-bit source reaches f16/bf16 through float8 -> f32 followed by "
              "f32 -> f16/bf16 rather than a single conversion (the unified "
              "pto.vmi.vcvt expands that pair for you)";
  }
  return success();
}

static LogicalResult verifyFPToFPRoundingAndSaturate(
    Operation *op, const std::optional<VMIFpToFpContract> &fpContract) {
  if (auto roundingAttr = op->getAttrOfType<StringAttr>("rounding")) {
    StringRef rounding = roundingAttr.getValue();
    if (rounding.size() != 1) {
      return op->emitOpError(
          "rounding attr must be a single-character mode token");
    }
    StringRef allowedRndModes =
        fpContract && !fpContract->allowedRndModes.empty()
            ? fpContract->allowedRndModes
            : StringRef("RAHZ");
    if (!allowedRndModes.contains(rounding)) {
      if (fpContract && !fpContract->allowedRndModes.empty()) {
        return op->emitOpError("rounding attr is not valid for this fp-to-fp "
                               "conversion type pair");
      }
      return op->emitOpError("rounding attr must be R, A, H, or Z");
    }
  }
  auto satAttr = op->getAttrOfType<StringAttr>("saturate");
  if (!fpContract || fpContract->requiresSat) {
    if (!satAttr) {
      return op->emitOpError("'saturate' attribute is required (SAT or NOSAT)");
    }
    StringRef satVal = satAttr.getValue();
    if (satVal != "SAT" && satVal != "NOSAT") {
      return op->emitOpError("saturate attr must be 'SAT' or 'NOSAT'");
    }
  } else if (satAttr) {
    return op->emitOpError("'saturate' attribute is not valid for this fp-to-fp "
                           "narrow conversion (no saturation)");
  }
  return success();
}

LogicalResult VMITruncFOp::verify() {
    VMIVRegType sourceType, resultType;
    if (failed(verifyCastOperands(*this, sourceType, resultType))) {
        return failure();
    }
  Type sourceElementType = sourceType.getElementType();
  Type resultElementType = resultType.getElementType();
  auto fpContract = lookupVMIFpToFpContract(sourceElementType, resultElementType);
  if (!isVMIFloatLikeType(sourceElementType) ||
      !isVMIFloatLikeType(resultElementType)) {
    return emitOpError(
        "requires floating-point-like source and result element types");
  }
  if (involvesBF16x2(sourceElementType, resultElementType) && !fpContract) {
    return emitOpError(
        "unsupported bf16x2 fp-to-fp conversion element type pair");
  }
  if (involvesVMIPackedFloatCarrier(sourceElementType, resultElementType) &&
      !fpContract) {
    return emitOpError(
        "unsupported packed fp-to-fp conversion element type pair");
  }
  unsigned srcBits = getVMIElementBitWidth(sourceElementType);
  unsigned dstBits = getVMIElementBitWidth(resultElementType);
  if (srcBits < dstBits) {
    return emitOpError(
        "requires result element type to be narrower than or same-width "
        "as source element type");
  }
  if (srcBits == dstBits && !fpContract) {
    return emitOpError("same-width fp-to-fp conversion is not supported "
                       "for this type pair; see lookupVMIFpToFpContract");
  }
  return verifyFPToFPRoundingAndSaturate(*this, fpContract);
}

// Batch12: cvt rounding/sat/else shared
template <typename OpTy>
static LogicalResult verifyCvtRoundingSat(OpTy op, bool requiresSat,
                                          llvm::StringRef directionName) {
  if (auto roundingAttr = op->template getAttrOfType<StringAttr>("rounding")) {
    StringRef rounding = roundingAttr.getValue();
    if (rounding != "R" && rounding != "A" && rounding != "F" &&
        rounding != "C" && rounding != "Z") {
      return op.emitOpError("rounding attr must be R, A, F, C, or Z");
    }
  }
  if (requiresSat) {
    auto satAttr = op->template getAttrOfType<StringAttr>("saturate");
    if (!satAttr) {
      return op.emitOpError("'saturate' attribute is required for this " +
                            directionName.str() + " conversion (SAT or NOSAT)");
    }
    StringRef satVal = satAttr.getValue();
    if (satVal != "SAT" && satVal != "NOSAT") {
      return op.emitOpError("saturate attr must be 'SAT' or 'NOSAT'");
    }
  } else {
    if (op->template getAttrOfType<StringAttr>("saturate")) {
      return op.emitOpError("'saturate' attribute is not valid for this " +
                            directionName.str() + " conversion (no overflow possible)");
    }
  }
  return success();
}

LogicalResult VMIFPToSIOp::verify() {
    VMIVRegType sourceType, resultType;
    if (failed(verifyCastOperands(*this, sourceType, resultType))) {
        return failure();
    }
  if (!isVMIFloatLikeType(sourceType.getElementType())) {
    return emitOpError("requires floating-point-like source element type");
  }
  if (!isVMISignedIntegerType(resultType.getElementType())) {
    return emitOpError("requires signed integer result element type");
  }
  auto contract =
      lookupVMIFpToSiContract(sourceType.getElementType(),
                              resultType.getElementType());
  if (!contract) {
    return emitOpError("unsupported fp-to-si conversion element type pair");
  }
  return verifyCvtRoundingSat(*this, contract->requiresSat, "fp-to-si");
}

LogicalResult VMIFPToUIOp::verify() {
    VMIVRegType sourceType, resultType;
    if (failed(verifyCastOperands(*this, sourceType, resultType))) {
        return failure();
    }
  if (!isVMIFloatLikeType(sourceType.getElementType())) {
    return emitOpError("requires floating-point-like source element type");
  }
  if (!isVMIUnsignedOrSignlessIntegerType(resultType.getElementType())) {
    return emitOpError(
        "requires unsigned or signless integer result element type");
  }
  auto contract =
      lookupVMIFpToUIContract(sourceType.getElementType(),
                              resultType.getElementType());
  if (!contract) {
    return emitOpError("unsupported fp-to-ui conversion element type pair");
  }
  return verifyCvtRoundingSat(*this, contract->requiresSat, "fp-to-ui");
}

LogicalResult VMISIToFPOp::verify() {
    VMIVRegType sourceType, resultType;
    if (failed(verifyCastOperands(*this, sourceType, resultType))) {
        return failure();
    }
  if (!isVMISignedIntegerType(sourceType.getElementType())) {
    return emitOpError("requires signed integer source element type");
  }
  if (!isVMIFloatLikeType(resultType.getElementType())) {
    return emitOpError("requires floating-point-like result element type");
  }
  unsigned srcBits = getVMIElementBitWidth(sourceType.getElementType());
  if (srcBits == mlir::pto::kValue32) {
    if (!resultType.getElementType().isF32()) {
      return emitOpError("requires f32 result element type for 32-bit "
                         "integer source");
    }
  } else if (srcBits == mlir::pto::kValue8) {
    if (!resultType.getElementType().isF16()) {
      return emitOpError("requires f16 result element type for 8-bit "
                         "integer source");
    }
  } else {
    return emitOpError("supports only si32 -> f32 or si8 -> f16");
  }
  return success();
}

LogicalResult VMIExtSIOp::verify() {
    VMIVRegType sourceType, resultType;
    if (failed(verifyCastOperands(*this, sourceType, resultType))) {
        return failure();
    }
  bool hasSignedTypes =
      isVMISignedIntegerType(sourceType.getElementType()) &&
      isVMISignedIntegerType(resultType.getElementType());
  if (!hasSignedTypes) {
    return emitOpError(
        "requires signed integer source and result element types");
  }
  if (getVMIElementBitWidth(sourceType.getElementType()) >=
      getVMIElementBitWidth(resultType.getElementType())) {
    return emitOpError(
        "requires result element type to be wider than source element type");
  }
  return success();
}

LogicalResult VMIExtUIOp::verify() {
    VMIVRegType sourceType, resultType;
    if (failed(verifyCastOperands(*this, sourceType, resultType))) {
        return failure();
    }
  bool hasUnsignedTypes =
      isVMIUnsignedOrSignlessIntegerType(sourceType.getElementType()) &&
      isVMIUnsignedOrSignlessIntegerType(resultType.getElementType());
  if (!hasUnsignedTypes) {
    return emitOpError(
        "requires unsigned or signless integer source and result element "
        "types");
  }
  if (getVMIElementBitWidth(sourceType.getElementType()) >=
      getVMIElementBitWidth(resultType.getElementType())) {
    return emitOpError(
        "requires result element type to be wider than source element type");
  }
  return success();
}

LogicalResult VMITruncIOp::verify() {
    VMIVRegType sourceType, resultType;
    if (failed(verifyCastOperands(*this, sourceType, resultType))) {
        return failure();
    }
  if (!isVMIIntegerLikeType(sourceType.getElementType()) ||
      !isVMIIntegerLikeType(resultType.getElementType())) {
    return emitOpError("requires integer source and result element types");
  }
  if (getVMIElementBitWidth(sourceType.getElementType()) <=
      getVMIElementBitWidth(resultType.getElementType())) {
    return emitOpError(
        "requires result element type to be narrower than source element type");
  }
  auto satAttr = (*this)->getAttrOfType<StringAttr>("saturate");
  if (!satAttr) {
    return emitOpError("'saturate' attribute is required (SAT or NOSAT)");
  }
  StringRef satVal = satAttr.getValue();
  if (satVal != "SAT" && satVal != "NOSAT") {
    return emitOpError("saturate attr must be 'SAT' or 'NOSAT'");
  }
  return success();
}

// Shared same-total-bits + layout verify core for Bitcast and VinterpretCast.
static LogicalResult verifyVMIInterpretCastCore(Operation *op, Type sourceTy,
                                                Type resultTy) {
  auto sourceType = cast<VMIVRegType>(sourceTy);
  auto resultType = cast<VMIVRegType>(resultTy);
  unsigned sourceBits =
      pto::getPTOStorageElemBitWidth(sourceType.getElementType());
  unsigned resultBits =
      pto::getPTOStorageElemBitWidth(resultType.getElementType());
  if (sourceBits == 0 || resultBits == 0) {
    return op->emitOpError(
        "requires integer or floating-point source and result element types");
  }
  if (sourceType.getElementCount() * static_cast<int64_t>(sourceBits) !=
      resultType.getElementCount() * static_cast<int64_t>(resultBits)) {
    return op->emitOpError(
        "requires source and result to carry the same total number of bits");
  }

  if (isLayoutAssigned(sourceType) || isLayoutAssigned(resultType)) {
    if (!isLayoutAssigned(sourceType) || !isLayoutAssigned(resultType)) {
      return op->emitOpError(
          "requires either both source and result to carry layout or neither "
          "to carry layout");
    }
    if (sourceType.getLayout() != resultType.getLayout()) {
      return op->emitOpError("requires source and result layouts to match");
    }
  }

  return success();
}

LogicalResult VMIBitcastOp::verify() {
  return verifyVMIInterpretCastCore(*this, getSource().getType(),
                                    getResult().getType());
}

static LogicalResult classifyCvtDirection(
    VMICvtOp op, Type srcElem, Type dstElem, unsigned srcBits, unsigned dstBits,
    bool srcFp, bool dstFp, bool srcInt, bool dstInt,
    const std::optional<VMIFpToFpContract> &fpContract, CvtDirection &dir) {
  if (srcFp && dstFp) {
    if (dstBits > srcBits) {
      dir = CvtDirection::FpWiden;
    } else if (dstBits < srcBits) {
      dir = CvtDirection::FpNarrow;
    } else {
      // Same-width fp→fp (e.g. bf16 → f16): only allowed for VMI fp-to-fp
      // contract pairs, routed through FpNarrow (1:1 TruncF).
      if (!fpContract) {
        return op.emitOpError(
            "same-width fp-to-fp conversion is not supported for this type "
            "pair; see lookupVMIFpToFpContract");
      }
      dir = CvtDirection::FpNarrow;
    }
  } else if (srcFp && dstInt) {
    if (isVMIUnsignedOrSignlessIntegerType(dstElem)) {
      dir = CvtDirection::FpToUi;
    } else {
      dir = CvtDirection::FpToSi;
    }
  } else if (srcInt && dstFp) {
    if (!isVMISignedIntegerType(srcElem)) {
      return op.emitOpError(
          "int-to-fp conversion requires explicitly signed integer source "
          "element type");
    }
    dir = CvtDirection::SiToFp;
  } else if (srcInt && dstInt) {
    if (dstBits > srcBits) {
      dir = CvtDirection::IntWiden;
    } else if (dstBits < srcBits) {
      dir = CvtDirection::IntNarrow;
    } else {
      return op.emitOpError(
          "int-to-int conversion must change element bit-width");
    }
  } else {
    return op.emitOpError(
        "unsupported element type combination for vcvt");
  }
  return success();
}

// Validate the rounding attribute for the given conversion direction.
static LogicalResult verifyCvtRounding(VMICvtOp op, CvtDirection dir,
                                       const std::optional<VMIFpToFpContract> &fpContract) {
  auto roundingAttr = op->getAttrOfType<StringAttr>("rounding");
  if (!roundingAttr) {
    return success();
  }
  if (dir != CvtDirection::FpNarrow && dir != CvtDirection::FpToSi &&
      dir != CvtDirection::FpToUi) {
    return op.emitOpError("'rounding' attribute is only valid for floating-point "
                          "narrowing or floating-point-to-integer conversions");
  }
  StringRef rnd = roundingAttr.getValue();
  if (rnd.size() != 1) {
    return op.emitOpError("rounding must be a single-character mode token");
  }
  if (dir == CvtDirection::FpNarrow) {
    StringRef allowedRndModes =
        fpContract && !fpContract->allowedRndModes.empty()
            ? fpContract->allowedRndModes
            : StringRef("RAHZ");
    if (!allowedRndModes.contains(rnd)) {
      if (fpContract && !fpContract->allowedRndModes.empty()) {
        return op.emitOpError(
            "rounding is not valid for this fp-to-fp conversion type pair");
      }
      return op.emitOpError("rounding must be 'R' (nearest-even), "
                            "'A' (away-from-zero), 'H' (half-up), "
                            "or 'Z' (toward-zero)");
    }
  } else if (rnd != "R" && rnd != "A" && rnd != "F" && rnd != "C" &&
             rnd != "Z") {
    return op.emitOpError("rounding must be 'R', 'A', 'F', 'C', or 'Z' for "
                          "floating-point-to-integer conversions");
  }
  return success();
}

// Validate the saturate attribute for FpToSi conversions.
static LogicalResult verifyCvtSaturateFpToSi(VMICvtOp op, Type srcElem,
                                             Type dstElem,
                                             StringAttr satAttr) {
  auto contract = lookupVMIFpToSiContract(srcElem, dstElem);
  if (!contract) {
    return op.emitOpError("unsupported fp-to-si conversion element type pair");
  }
  if (contract->requiresSat) {
    if (!satAttr) {
      return op.emitOpError("'saturate' attribute is required for this "
                            "fp-to-si conversion; write 'SAT' or 'NOSAT'");
    }
    StringRef satVal = satAttr.getValue();
    if (satVal != "SAT" && satVal != "NOSAT") {
      return op.emitOpError("saturate must be 'SAT' or 'NOSAT'");
    }
  } else if (satAttr) {
    return op.emitOpError("'saturate' attribute is not valid for this "
                          "fp-to-si conversion (no overflow possible)");
  }
  return success();
}

// Validate the saturate attribute for FpToUi conversions.
static LogicalResult verifyCvtSaturateFpToUi(VMICvtOp op, Type srcElem,
                                             Type dstElem,
                                             StringAttr satAttr) {
  auto contract = lookupVMIFpToUIContract(srcElem, dstElem);
  if (!contract) {
    return op.emitOpError("unsupported fp-to-ui conversion element type pair");
  }
  if (contract->requiresSat) {
    if (!satAttr) {
      return op.emitOpError("'saturate' attribute is required for this "
                            "fp-to-ui conversion; write 'SAT' or 'NOSAT'");
    }
    StringRef satVal = satAttr.getValue();
    if (satVal != "SAT" && satVal != "NOSAT") {
      return op.emitOpError("saturate must be 'SAT' or 'NOSAT'");
    }
  } else if (satAttr) {
    return op.emitOpError("'saturate' attribute is not valid for this "
                          "fp-to-ui conversion (no overflow possible)");
  }
  return success();
}

// Validate the saturate attribute for narrow (FpNarrow / IntNarrow) conversions.
static LogicalResult verifyCvtSaturateNarrow(
    VMICvtOp op, CvtDirection dir, unsigned srcBits, unsigned dstBits,
    Type srcElem, Type dstElem, StringAttr satAttr,
    const std::optional<VMIFpToFpContract> &fpContract) {
  // Fp-narrow: default to requiring a saturate attribute, but consult the
  // fp-to-fp contract when one exists (e.g. bf16x2->f4x2 narrows with
  // requiresSat=false and must NOT carry saturate).
  bool needSat = (dir == CvtDirection::IntNarrow);
  if (dir == CvtDirection::FpNarrow) {
    needSat = !fpContract || fpContract->requiresSat;
  }
  if (needSat) {
    if (!satAttr) {
      return op.emitOpError("'saturate' attribute is required for fp-narrow / "
                            "int-narrow conversions; write 'SAT' or 'NOSAT'");
    }
    StringRef satVal = satAttr.getValue();
    if (satVal != "SAT" && satVal != "NOSAT") {
      return op.emitOpError("saturate must be 'SAT' or 'NOSAT'");
    }
    bool signedSource = isa<IntegerType>(srcElem) &&
                        cast<IntegerType>(srcElem).isSigned();
    bool signedResult = isa<IntegerType>(dstElem) &&
                        cast<IntegerType>(dstElem).isSigned();
    // Only si32 -> si16 has a native signed int-narrow conversion.  The
    // narrower signed destinations have no native hardware form, so the
    // lowering aliases them through the unsigned conversion, which is
    // bit-pattern equal ONLY under NOSAT (SAT would clamp to
    // [0, 2^dstBits - 1] instead of the signed range).
    bool nativeSignedNarrow =
        srcBits == mlir::pto::kValue32 && dstBits == mlir::pto::kValue16;
    if (dir == CvtDirection::IntNarrow && satVal == "SAT" &&
        signedSource && signedResult && !nativeSignedNarrow) {
      return op.emitOpError(
          "signed int-narrow to a signed destination does not support "
          "saturate=\"SAT\" (no native hardware form; only "
          "saturate=\"NOSAT\" is allowed)");
    }
  } else if (satAttr && dir == CvtDirection::FpNarrow) {
    return op.emitOpError("'saturate' attribute is not valid for this fp-to-fp "
                          "narrow conversion (no saturation)");
  } else if (satAttr) {
    return op.emitOpError("'saturate' attribute is only valid for fp-narrow / "
                          "int-narrow conversions");
  }
  return success();
}

// Dispatch saturate validation to the correct sub-verifier.
static LogicalResult verifyCvtSaturate(VMICvtOp op, CvtDirection dir,
                                       unsigned srcBits, unsigned dstBits,
                                       Type srcElem, Type dstElem,
                                       StringAttr satAttr,
                                       const std::optional<VMIFpToFpContract> &fpContract) {
  if (dir == CvtDirection::FpToSi) {
    return verifyCvtSaturateFpToSi(op, srcElem, dstElem, satAttr);
  }
  if (dir == CvtDirection::FpToUi) {
    return verifyCvtSaturateFpToUi(op, srcElem, dstElem, satAttr);
  }
  return verifyCvtSaturateNarrow(op, dir, srcBits, dstBits, srcElem, dstElem,
                                 satAttr, fpContract);
}

LogicalResult VMICvtOp::verify() {
    VMIVRegType sourceType, resultType;
    if (failed(verifyCastOperands(*this, sourceType, resultType))) {
        return failure();
    }
  Type srcElem = sourceType.getElementType();
  Type dstElem = resultType.getElementType();
unsigned srcBits = getVMIElementBitWidth(srcElem), dstBits = getVMIElementBitWidth(dstElem);
bool srcFp = isVMIFloatLikeType(srcElem), dstFp = isVMIFloatLikeType(dstElem);
bool srcInt = isVMIIntegerLikeType(srcElem), dstInt = isVMIIntegerLikeType(dstElem);
  auto fpContract = srcFp && dstFp ? lookupVMIFpToFpContract(srcElem, dstElem)
                                   : std::nullopt;
  if (involvesBF16x2(srcElem, dstElem) && !fpContract) {
    return emitOpError("unsupported conversion involving bf16x2 element type");
  }
  if (srcFp && dstFp && involvesVMIPackedFloatCarrier(srcElem, dstElem) &&
      !fpContract) {
    return emitOpError(
        "unsupported packed fp-to-fp conversion element type pair");
  }
  CvtDirection dir = CvtDirection::FpNarrow;
  if (failed(classifyCvtDirection(*this, srcElem, dstElem, srcBits, dstBits,
                                  srcFp, dstFp, srcInt, dstInt, fpContract,
                                  dir))) {
    return failure();
  }
  if (failed(verifyCvtRounding(*this, dir, fpContract))) {
    return failure();
  }
  auto satAttr = (*this)->getAttrOfType<StringAttr>("saturate");
  if (failed(verifyCvtSaturate(*this, dir, srcBits, dstBits, srcElem, dstElem,
                               satAttr, fpContract))) {
    return failure();
  }
  if (auto pmodeAttr = (*this)->getAttrOfType<StringAttr>("pmode")) {
    if (failed(verifyVMIPMode(getOperation(), pmodeAttr.getValue()))) {
      return failure();
    }
  }
  return success();
}

LogicalResult VMIVinterpretCastOp::verify() {
  return verifyVMIInterpretCastCore(*this, getSource().getType(),
                                    getResult().getType());
}

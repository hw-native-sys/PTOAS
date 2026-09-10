// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMI_helpers.cpp - VMI internal helper implementations ---------------===//
//===----------------------------------------------------------------------===//

#include "VMIInternal.h"

using namespace mlir;
using namespace mlir::pto;

// ---------------------------------------------------------------------------
// FpToSi hardware contract (mirrors VPTO lookupVcvtContract fp→int rows)
// ---------------------------------------------------------------------------

namespace mlir::pto {
std::optional<VMIFpToSiContract>
lookupVMIFpToSiContract(Type srcElem, Type dstElem) {
  // Must be float → explicitly signed integer.
  if (!isVMIFloatLikeType(srcElem)) {
    return std::nullopt;
  }
  auto dstInt = dyn_cast<IntegerType>(dstElem);
  if (!dstInt || !dstInt.isSigned()) {
    return std::nullopt;
  }

  bool srcF32 = srcElem.isF32();
  bool srcF16 = srcElem.isF16();
  bool srcBF16 = srcElem.isBF16();
  unsigned dstBits = dstInt.getWidth();
  // f32 → s32: same width, rnd, sat, no part
  if (srcF32 && dstBits == 32) {
    return VMIFpToSiContract{/*requiresSat=*/true, /*requiresPart=*/false};
  }
  // f32 → s16: narrow 2×, rnd, sat, part (EvenOdd)
  if (srcF32 && dstBits == 16) {
    return VMIFpToSiContract{/*requiresSat=*/true, /*requiresPart=*/true};
  }
  // f16 → s32: widen 2×, rnd, NO sat, part (EvenOdd)
  if (srcF16 && dstBits == 32) {
    return VMIFpToSiContract{/*requiresSat=*/false, /*requiresPart=*/true};
  }
  // f16 → s16: same width, rnd, sat, no part
  if (srcF16 && dstBits == 16) {
    return VMIFpToSiContract{/*requiresSat=*/true, /*requiresPart=*/false};
  }
  // f16 → s8: narrow 2×, rnd, sat, part (EvenOdd)
  if (srcF16 && dstBits == 8) {
    return VMIFpToSiContract{/*requiresSat=*/true, /*requiresPart=*/true};
  }
  // bf16 → s32: widen 2×, rnd, sat, part (EvenOdd)
  if (srcBF16 && dstBits == 32) {
    return VMIFpToSiContract{/*requiresSat=*/true, /*requiresPart=*/true};
  }

  return std::nullopt;
}

// ---------------------------------------------------------------------------
// FpToUi hardware contract (mirrors VPTO lookupVcvtContract fp→uint rows)
// ---------------------------------------------------------------------------

std::optional<VMIFpToUiContract>
lookupVMIFpToUIContract(Type srcElem, Type dstElem) {
  // Must be float → unsigned integer. VMI signless integers carry
  // unsigned semantics.
  if (!isVMIFloatLikeType(srcElem)) {
    return std::nullopt;
  }
  auto dstInt = dyn_cast<IntegerType>(dstElem);
  if (!dstInt || dstInt.isSigned()) {
    return std::nullopt;
  }

  bool srcF16 = srcElem.isF16();
  unsigned dstBits = dstInt.getWidth();
  // f16 → u8: narrow 2×, rnd, sat, part (EvenOdd)
  if (srcF16 && dstBits == 8) {
    return VMIFpToUiContract{/*requiresSat=*/true, /*requiresPart=*/true};
  }

  return std::nullopt;
}

// ---------------------------------------------------------------------------
// FpToFp hardware contract (VMI-owned; may diverge from VPTO).
// Enumerates same-width fp->fp whitelist entries plus the fp->fp narrow
// paths whose sat/rounding semantics differ from the generic truncf default
// (e.g. bf16x2->f4x2 narrows with NO saturation).
// ---------------------------------------------------------------------------

std::optional<VMIFpToFpContract>
lookupVMIFpToFpContract(Type srcElem, Type dstElem) {
  if (!isVMIFloatLikeType(srcElem) || !isVMIFloatLikeType(dstElem)) {
    return std::nullopt;
  }
  unsigned srcBits = pto::getPTOStorageElemBitWidth(srcElem);
  unsigned dstBits = pto::getPTOStorageElemBitWidth(dstElem);
  // bf16x2 -> f4x2 (32->8 narrow): Packed4, rnd, NO sat. Mirrors the VPTO
  // bf16->f4 contract row (requiresSat=false) so the VMI verifier does not
  // force a saturate attribute that the physical pto.vcvt would reject.
  if (pto::isPTOBF16x2Type(srcElem) && pto::isPTOFloat4PackedType(dstElem)) {
    return VMIFpToFpContract{/*requiresRnd=*/true, /*requiresSat=*/false,
                            /*requiresPart=*/true,
                            /*allowedRndModes=*/"RAFZC"};
}
  // f4x2 -> bf16x2 (8->32 widen): Packed4, no rnd, no sat. Mirrors the VPTO
  // f4->bf16 contract row (requiresSat=false, requiresRnd=false). Widen has
  // no rounding/saturate semantics by construction; the contract exists so
  // the involvesBF16x2 / packed-carrier gates in the VMI verifiers pass.
  if (pto::isPTOFloat4PackedType(srcElem) && pto::isPTOBF16x2Type(dstElem)) {
    return VMIFpToFpContract{/*requiresRnd=*/false, /*requiresSat=*/false,
                            /*requiresPart=*/true,
                            /*allowedRndModes=*/StringRef()};
  }
  if (srcBits != dstBits) {
    return std::nullopt;
  }
  // bf16 -> f16: same-width, rnd, sat, no part.
  if (srcElem.isBF16() && dstElem.isF16()) {
    return VMIFpToFpContract{/*requiresRnd=*/true, /*requiresSat=*/true,
                            /*requiresPart=*/false,
                            /*allowedRndModes=*/StringRef()};
  }
  // f16 -> bf16: same-width, rnd, NO sat, no part.
  bool srcIsF16 = srcElem.isF16();
  bool dstIsBF16 = dstElem.isBF16();
  if (srcIsF16 && dstIsBF16) {
    return VMIFpToFpContract{/*requiresRnd=*/true, /*requiresSat=*/false,
                            /*requiresPart=*/false,
                            /*allowedRndModes=*/StringRef()};
  }
  return std::nullopt;
}

} // namespace mlir::pto

VMILayoutAttr VMILayoutAttr::getContiguous(MLIRContext *context,
                                           int64_t laneStride) {
  return VMILayoutAttr::get(context, "contiguous", 1, 1, 0, laneStride);
}

VMILayoutAttr VMILayoutAttr::getDeinterleaved(MLIRContext *context,
                                              int64_t factor,
                                              int64_t laneStride) {
  return VMILayoutAttr::get(context, "deinterleaved", factor, 1, 0,
                            laneStride);
}

VMILayoutAttr VMILayoutAttr::getBlockDeinterleaved(MLIRContext *context,
                                                   int64_t factor) {
  return VMILayoutAttr::get(context, "block_deinterleaved", factor, 1, 0, 1);
}

VMILayoutAttr VMILayoutAttr::getGroupSlots(MLIRContext *context,
                                           int64_t numGroups, int64_t slots,
                                           int64_t laneStride) {
  return VMILayoutAttr::get(context, "num_groups", numGroups, 1, slots,
                            laneStride);
}

static ParseResult parseContiguousLayoutFields(AsmParser &parser,
                                               int64_t &laneStride) {
  while (succeeded(parser.parseOptionalComma())) {
    StringRef field;
    if (failed(parser.parseKeyword(&field)) || failed(parser.parseEqual()) ||
        field != "lane_stride" || failed(parser.parseInteger(laneStride))) {
      parser.emitError(parser.getCurrentLocation(),
                       "expected 'lane_stride = <integer>'");
      return failure();
    }
  }
  return success();
}

static ParseResult parseDeinterleavedLayoutFields(AsmParser &parser,
                                                  int64_t &laneStride) {
  while (succeeded(parser.parseOptionalComma())) {
    StringRef field;
    if (failed(parser.parseKeyword(&field)) || failed(parser.parseEqual())) {
      return failure();
    }
    if (field == "lane_stride") {
      if (failed(parser.parseInteger(laneStride))) {
        return failure();
      }
    } else {
      parser.emitError(parser.getCurrentLocation(),
                       "expected 'lane_stride = <integer>'");
      return failure();
    }
  }
  return success();
}

static ParseResult parseNumGroupsLayoutFields(AsmParser &parser,
                                              int64_t &slots,
                                              int64_t &laneStride) {
  while (succeeded(parser.parseOptionalComma())) {
    StringRef field;
    if (failed(parser.parseKeyword(&field)) || failed(parser.parseEqual())) {
      return failure();
    }
    if (field == "slots") {
      if (failed(parser.parseInteger(slots))) {
        return failure();
      }
    } else if (field == "lane_stride") {
      if (failed(parser.parseInteger(laneStride))) {
        return failure();
      }
    } else {
      parser.emitError(parser.getCurrentLocation(),
                       "expected 'slots = <integer>' or "
                       "'lane_stride = <integer>'");
      return failure();
    }
  }
  return success();
}

Attribute VMILayoutAttr::parse(AsmParser &parser, Type) {
  SMLoc loc = parser.getCurrentLocation();
  StringRef kind;
  int64_t factor = 1;
  int64_t blockElems = 1;
  int64_t slots = 0;
  int64_t laneStride = 1;

  if (failed(parser.parseLess()) || failed(parser.parseKeyword(&kind))) {
    return {};
  }

  if (kind == "contiguous") {
    factor = 1;
    if (failed(parseContiguousLayoutFields(parser, laneStride))) {
      return {};
    }
  } else if (kind == "deinterleaved") {
    if (failed(parser.parseEqual()) || failed(parser.parseInteger(factor)) ||
        failed(parseDeinterleavedLayoutFields(parser, laneStride))) {
      return {};
    }
  } else if (kind == "block_deinterleaved") {
    if (failed(parser.parseEqual()) || failed(parser.parseInteger(factor))) {
      return {};
    }
  } else if (kind == "num_groups") {
    if (failed(parser.parseEqual()) || failed(parser.parseInteger(factor)) ||
        failed(parseNumGroupsLayoutFields(parser, slots, laneStride))) {
      return {};
    }
  } else {
    parser.emitError(parser.getCurrentLocation(),
                     "expected VMI layout kind 'contiguous' or "
                     "'deinterleaved' or 'block_deinterleaved' or "
                     "'num_groups'");
    return {};
  }

  if (failed(parser.parseGreater())) {
    return {};
  }
  return parser.getChecked<VMILayoutAttr>(loc, parser.getContext(), kind,
                                          factor, blockElems, slots,
                                          laneStride);
}

void VMILayoutAttr::print(AsmPrinter &printer) const {
  printer << "<" << getKind();
  if (isContiguous()) {
    if (getLaneStride() != 1) {
      printer << ", lane_stride = " << getLaneStride();
    }
  } else if (isDeinterleaved()) {
    printer << " = " << getFactor();
    if (getLaneStride() != 1) {
      printer << ", lane_stride = " << getLaneStride();
    }
  } else if (isBlockDeinterleaved()) {
    printer << " = " << getFactor();
  } else if (isGroupSlots()) {
    printer << " = " << getFactor();
    if (getSlots() != 0) {
      printer << ", slots = " << getSlots();
    }
    if (getLaneStride() != 1) {
      printer << ", lane_stride = " << getLaneStride();
    }
  }
  printer << ">";
}

static LogicalResult verifyContiguousLayout(
    function_ref<InFlightDiagnostic()> emitError, int64_t factor,
    int64_t blockElems, int64_t slots) {
  if (factor != 1 || blockElems != 1 || slots != 0) {
    return emitError()
           << "#pto.vmi.layout<contiguous> requires factor, block_elems, "
              "and slots to be their defaults";
  }
  return success();
}

static LogicalResult verifyDeinterleavedLayout(
    function_ref<InFlightDiagnostic()> emitError, int64_t factor,
    int64_t blockElems, int64_t slots) {
  if (factor != mlir::pto::kValue2 && factor != mlir::pto::kValue4) {
    return emitError() << "#pto.vmi.layout<deinterleaved = " << factor
                       << "> expected factor to be 2 or 4";
  }
  if (blockElems != 1) {
    return emitError() << "#pto.vmi.layout<deinterleaved = " << factor
                       << ", block_elems = " << blockElems
                       << "> requires block_elems to be omitted";
  }
  if (slots != 0) {
    return emitError() << "#pto.vmi.layout<deinterleaved = " << factor
                       << "> requires slots to be omitted";
  }
  return success();
}

static LogicalResult verifyBlockDeinterleavedLayout(
    function_ref<InFlightDiagnostic()> emitError, int64_t factor,
    int64_t blockElems, int64_t slots, int64_t laneStride) {
  if (factor != mlir::pto::kValue2 && factor != mlir::pto::kValue4) {
    return emitError() << "#pto.vmi.layout<block_deinterleaved = " << factor
                       << "> expected factor to be 2 or 4";
  }
  if (blockElems != 1 || slots != 0 || laneStride != 1) {
    return emitError()
           << "#pto.vmi.layout<block_deinterleaved = " << factor
           << "> does not accept block_elems, slots, or lane_stride";
  }
  return success();
}

static LogicalResult verifyNumGroupsLayout(
    function_ref<InFlightDiagnostic()> emitError, int64_t factor,
    int64_t blockElems, int64_t slots) {
  if (factor <= 0) {
    return emitError() << "#pto.vmi.layout<num_groups = " << factor
                       << "> requires num_groups to be positive";
  }
  if (blockElems != 1) {
    return emitError() << "#pto.vmi.layout<num_groups = " << factor
                       << "> requires block_elems to be omitted";
  }
  if (slots < 0) {
    return emitError() << "#pto.vmi.layout<num_groups = " << factor
                       << ", slots = " << slots
                       << "> requires slots to be omitted or positive";
  }
  return success();
}

LogicalResult
VMILayoutAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                      StringRef kind, int64_t factor, int64_t blockElems,
                      int64_t slots, int64_t laneStride) {
  if (laneStride <= 0) {
    return emitError() << "#pto.vmi.layout<" << kind
                       << "> requires lane_stride to be positive";
  }
  if (kind == "contiguous") {
    return verifyContiguousLayout(emitError, factor, blockElems, slots);
  }
  if (kind == "deinterleaved") {
    return verifyDeinterleavedLayout(emitError, factor, blockElems, slots);
  }
  if (kind == "block_deinterleaved") {
    return verifyBlockDeinterleavedLayout(emitError, factor, blockElems, slots,
                                          laneStride);
  }
  if (kind == "num_groups") {
    return verifyNumGroupsLayout(emitError, factor, blockElems, slots);
  }
  return emitError() << "expected VMI layout kind to be 'contiguous' or "
                        "'deinterleaved' or 'block_deinterleaved' or "
                        "'num_groups'";
}

Type VMIVRegType::parse(AsmParser &parser) {
  SmallVector<int64_t, 1> shape;
  Type elementType;
  Attribute layout;
  SMLoc loc = parser.getCurrentLocation();

  if (failed(parser.parseLess()) ||
      failed(parser.parseDimensionList(shape, /*allowDynamic=*/false,
                                       /*withTrailingX=*/true)) ||
      shape.size() != 1 || failed(parser.parseType(elementType)) ||
      failed(parseOptionalVMILayout(parser, layout)) ||
      failed(parser.parseGreater())) {
    return {};
  }

  return parser.getChecked<VMIVRegType>(loc, parser.getContext(), shape.front(),
                                        elementType, layout);
}

void VMIVRegType::print(AsmPrinter &printer) const {
  printer << "<" << getElementCount() << "x";
  printer.printType(getElementType());
  if (getLayout()) {
    printer << ", " << getLayout();
  }
  printer << ">";
}

LogicalResult VMIVRegType::verify(function_ref<InFlightDiagnostic()> emitError,
                                  int64_t elementCount, Type elementType,
                                  Attribute layout) {
  if (elementCount <= 0) {
    return emitError() << "'"
                       << formatVMIVRegType(elementCount, elementType, layout)
                       << "' expected a positive element count";
}

  if (!isSupportedVMIElementType(elementType)) {
    return emitError() << "'"
                       << formatVMIVRegType(elementCount, elementType, layout)
                       << "' expected an integer, index, floating-point, or "
                          "PTO low-precision element type";
}
  if (!isVMIPredicateMaskableElementType(elementType)) {
    return emitError() << "'"
                       << formatVMIVRegType(elementCount, elementType, layout)
                       << "' expected an 8-bit, 16-bit, or 32-bit logical "
                          "element type";
}
  if (layout && !mlir::isa<VMILayoutAttr>(layout)) {
    return emitError() << "'"
                       << formatVMIVRegType(elementCount, elementType, layout)
                       << "' expected layout to be #pto.vmi.layout";
}
  if (auto layoutAttr = llvm::dyn_cast_or_null<VMILayoutAttr>(layout)) {
    if (layoutAttr.isGroupSlots() &&
        elementCount != layoutAttr.getNumGroups()) {
      return emitError() << "'"
                         << formatVMIVRegType(elementCount, elementType, layout)
                         << "' expected num_groups layout to describe exactly "
                            "one logical result lane per group";
}
  }

  return success();
}

bool VMIMaskType::isSupportedGranularity(StringRef granularity) {
  return granularity == "pred" || isConcreteGranularity(granularity);
}

bool VMIMaskType::isConcreteGranularity(StringRef granularity) {
  return granularity == "b8" || granularity == "b16" || granularity == "b32";
}

Type VMIMaskType::parse(AsmParser &parser) {
  SmallVector<int64_t, 1> shape;
  StringRef granularity;
  Attribute layout;
  SMLoc loc = parser.getCurrentLocation();

  if (failed(parser.parseLess()) ||
      failed(parser.parseDimensionList(shape, /*allowDynamic=*/false,
                                       /*withTrailingX=*/true)) ||
      shape.size() != 1 || failed(parser.parseKeyword(&granularity)) ||
      failed(parseOptionalVMILayout(parser, layout)) ||
      failed(parser.parseGreater())) {
    return {};
  }

  return parser.getChecked<VMIMaskType>(loc, parser.getContext(), shape.front(),
                                        granularity, layout);
}

void VMIMaskType::print(AsmPrinter &printer) const {
  printer << "<" << getElementCount() << "x" << getGranularity();
  if (getLayout()) {
    printer << ", " << getLayout();
  }
  printer << ">";
}

LogicalResult VMIMaskType::verify(function_ref<InFlightDiagnostic()> emitError,
                                  int64_t elementCount, StringRef granularity,
                                  Attribute layout) {
  if (elementCount <= 0) {
    return emitError() << "'"
                       << formatVMIMaskType(elementCount, granularity, layout)
                       << "' expected a positive element count";
  }

  if (!isSupportedGranularity(granularity)) {
    return emitError() << "'"
                       << formatVMIMaskType(elementCount, granularity, layout)
                       << "' expected granularity to be one of pred, b8, b16, "
                          "b32";
  }

  if (layout && !mlir::isa<VMILayoutAttr>(layout)) {
    return emitError() << "'"
                       << formatVMIMaskType(elementCount, granularity, layout)
                       << "' expected layout to be #pto.vmi.layout";
  }

  if (granularity == "pred" && layout) {
    return emitError() << "'"
                       << formatVMIMaskType(elementCount, granularity, layout)
                       << "' pred mask must not carry layout";
  }

  return success();
}

//===----------------------------------------------------------------------===//

std::string formatVMIVRegType(int64_t elementCount, Type elementType,
                                     Attribute layout) {
  std::string result;
  llvm::raw_string_ostream os(result);
  os << "!pto.vmi.vreg<" << elementCount << "x" << elementType;
  if (layout) {
    os << ", " << layout;
  }
  os << ">";
  return result;
}

std::string formatVMIMaskType(int64_t elementCount,
                                     StringRef granularity, Attribute layout) {
  std::string result;
  llvm::raw_string_ostream os(result);
  os << "!pto.vmi.mask<" << elementCount << "x" << granularity;
  if (layout) {
    os << ", " << layout;
  }
  os << ">";
  return result;
}

bool matchesVMIIntSemantics(IntegerType intType,
                                   VMIIntSignSemantics semantics) {
  switch (semantics) {
  case VMIIntSignSemantics::Unsigned:
    return intType.isUnsigned() || intType.isSignless();
  case VMIIntSignSemantics::Signed:
    return intType.isSigned();
  case VMIIntSignSemantics::Any:
    return true;
  }
  llvm_unreachable("bad VMIIntSignSemantics");
}

bool isCompatibleVMIScalarForSemanticType(Type semanticType,
                                              Type scalarType) {
  if (semanticType == scalarType) {
    return true;
  }

  auto semanticInt = dyn_cast<IntegerType>(semanticType);
  auto scalarInt = dyn_cast<IntegerType>(scalarType);
  if (!semanticInt || !scalarInt ||
      semanticInt.getWidth() != scalarInt.getWidth()) {
    return false;
  }

  return isSemanticSignCompatible(semanticInt, scalarInt);
}

LogicalResult parseOptionalVMILayout(AsmParser &parser,
                                            Attribute &layout) {
  if (failed(parser.parseOptionalComma())) {
    return success();
  }

  if (failed(parser.parseAttribute(layout))) {
    return failure();
  }
  if (!mlir::isa<VMILayoutAttr>(layout)) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected #pto.vmi.layout attribute");
  }
  return success();
}

FailureOr<VMILayoutAttr> getAssignedVMILayout(Type type) {
  Attribute layout;
  if (auto vregType = dyn_cast<VMIVRegType>(type)) {
    layout = vregType.getLayout();
  }
  else if (auto maskType = dyn_cast<VMIMaskType>(type)) {
    layout = maskType.getLayout();
  }
  else {
    return failure();
  }

  auto layoutAttr = dyn_cast_or_null<VMILayoutAttr>(layout);
  if (!layoutAttr) {
    return failure();
  }
  return layoutAttr;
}

int64_t getMaskGranularityBitWidth(StringRef granularity) {
  if (granularity == "b8") {
    return mlir::pto::kValue8;
  }
  if (granularity == "b16") {
    return mlir::pto::kValue16;
  }
  if (granularity == "b32") {
    return mlir::pto::kValue32;
  }
  return 0;
}

StringRef getMaskGranularityForBitWidth(int64_t bits) {
  switch (bits) {
  case mlir::pto::kValue8:
    return "b8";
  case mlir::pto::kValue16:
    return "b16";
  case mlir::pto::kValue32:
    return "b32";
  default:
    return "";
  }
}

FailureOr<StringRef> getVMIMaskPhysicalGranularity(VMIMaskType type) {
  int64_t bits = getMaskGranularityBitWidth(type.getGranularity());
  if (bits == 0) {
    return failure();
  }

  VMILayoutAttr layout = type.getLayoutAttr();
  int64_t laneStride = layout && layout.hasLaneStride() ? layout.getLaneStride()
                                                        : 1;
  StringRef physicalGranularity =
      getMaskGranularityForBitWidth(bits * laneStride);
  if (physicalGranularity.empty()) {
    return failure();
  }
  return physicalGranularity;
}

FailureOr<int64_t> getPhysicalLanesPerPart(Type type) {
  if (auto vregType = dyn_cast<VMIVRegType>(type)) {
    return getDataLanesPerPart(getVMIPhysicalDataElementType(vregType));
  }
  if (auto maskType = dyn_cast<VMIMaskType>(type)) {
    FailureOr<StringRef> physicalGranularity =
        getVMIMaskPhysicalGranularity(maskType);
    if (failed(physicalGranularity)) {
      return failure();
    }
    return getMaskLanesPerPart(*physicalGranularity);
  }
  return failure();
}

FailureOr<int64_t> getDenseLaneStride(Type type) {
  FailureOr<VMILayoutAttr> layout = getAssignedVMILayout(type);
  if (failed(layout)) {
    return failure();
  }
  if (isa<VMIMaskType>(type)) {
    return 1;
  }
  return (*layout).isDense() ? (*layout).getLaneStride() : 1;
}

static LogicalResult
verifyAllSameVRegShapeCore(Operation *op, ArrayRef<VMIVRegType> types,
                           bool requireSameElement, bool requireSameLayout) {
  if (types.empty()) {
    return success();
  }

  VMIVRegType first = types.front();
  bool anyLayout = llvm::any_of(
      types, [](VMIVRegType type) { return isLayoutAssigned(type); });

  for (VMIVRegType type : types) {
    if (type.getElementCount() != first.getElementCount()) {
      return op->emitOpError(
          "requires all VMI data values to have the same logical lane count");
    }
    if (requireSameElement && type.getElementType() != first.getElementType()) {
      return op->emitOpError(
          "requires all VMI data values to have the same element type");
    }
    if (anyLayout && !isLayoutAssigned(type)) {
      return op->emitOpError(
          "requires either all or no VMI data values to carry layout");
    }
    if (requireSameLayout && anyLayout &&
        type.getLayout() != first.getLayout()) {
      return op->emitOpError("requires all layout-assigned VMI data values to "
                             "have the same layout");
    }
  }
  return success();
}

LogicalResult
verifyAllSameVRegShapeAndLayout(Operation *op, ArrayRef<VMIVRegType> types,
                                bool requireSameElement) {
  return verifyAllSameVRegShapeCore(op, types, requireSameElement,
                                    /*requireSameLayout=*/true);
}

LogicalResult verifyAllSameVRegShapeAndLayoutPresence(
    Operation *op, ArrayRef<VMIVRegType> types, bool requireSameElement) {
  return verifyAllSameVRegShapeCore(op, types, requireSameElement,
                                    /*requireSameLayout=*/false);
}

LogicalResult verifyFloatUnaryVRegOp(Operation *op, VMIVRegType source,
                                            VMIVRegType result) {
  if (failed(
          verifyBF16x2ComputeElementType(op, source.getElementType()))) {
    return failure();
  }
  if (!isVMIFloatLikeType(source.getElementType())) {
    return op->emitOpError("requires floating-point-like VMI element type");
  }
  return verifyAllSameVRegShapeAndLayout(op, {source, result},
                                         /*requireSameElement=*/true);
}

LogicalResult verifyFloatTernaryVRegOp(Operation *op, VMIVRegType lhs,
                                              VMIVRegType rhs, VMIVRegType acc,
                                              VMIVRegType result) {
  if (failed(verifyBF16x2ComputeElementType(op, lhs.getElementType()))) {
    return failure();
  }
  if (!isVMIFloatLikeType(lhs.getElementType())) {
    return op->emitOpError("requires floating-point-like VMI element type");
  }
  return verifyAllSameVRegShapeAndLayout(op, {lhs, rhs, acc, result},
                                         /*requireSameElement=*/true);
}

LogicalResult
verifyAllSameMaskShapeLayoutAndGranularity(Operation *op,
                                           ArrayRef<VMIMaskType> types) {
  if (types.empty()) {
    return success();
  }

  VMIMaskType first = types.front();
  bool anyLayout = llvm::any_of(
      types, [](VMIMaskType type) { return isLayoutAssigned(type); });

  for (VMIMaskType type : types) {
    if (type.getElementCount() != first.getElementCount()) {
      return op->emitOpError(
          "requires all VMI mask values to have the same logical lane count");
    }
    if (type.getGranularity() != first.getGranularity()) {
      return op->emitOpError(
          "requires all VMI mask values to have the same granularity");
    }
    if (anyLayout && !isLayoutAssigned(type)) {
      return op->emitOpError(
          "requires either all or no VMI mask values to carry layout");
    }
    if (anyLayout && type.getLayout() != first.getLayout()) {
      return op->emitOpError(
          "requires all layout-assigned VMI mask values to have the same "
          "layout");
    }
  }
  return success();
}

LogicalResult verifyMaskMatchesData(Operation *op, VMIMaskType maskType,
                                           VMIVRegType dataType) {
  if (maskType.getElementCount() != dataType.getElementCount()) {
    return op->emitOpError(
        "requires mask logical lane count to match data lane count");
  }

  if (isLayoutAssigned(maskType) || isLayoutAssigned(dataType)) {
    if (!isLayoutAssigned(maskType) || !isLayoutAssigned(dataType)) {
      return op->emitOpError("requires either both mask and data to carry "
                             "layout or neither to carry layout");
    }
    if (maskType.getLayout() != dataType.getLayout()) {
      return op->emitOpError("requires mask layout to match data layout");
    }
  }

  if (maskType.isPred()) {
    return success();
  }

  unsigned elementBitWidth = getVMIElementBitWidth(dataType.getElementType());
  int64_t maskBitWidth = getMaskGranularityBitWidth(maskType.getGranularity());
  if (elementBitWidth != 0 && maskBitWidth != 0 &&
      elementBitWidth != static_cast<unsigned>(maskBitWidth)) {
    return op->emitOpError(
        "requires mask granularity to match data element width");
  }

  return success();
}

bool isUBBackedMemoryType(Type type) {
  if (auto ptrType = dyn_cast<PtrType>(type)) {
    return ptrType.getMemorySpace().getAddressSpace() == AddressSpace::VEC;
  }

  auto memrefType = dyn_cast<BaseMemRefType>(type);
  if (!memrefType) {
    return false;
  }

  Attribute memorySpace = memrefType.getMemorySpace();
  if (auto addressSpace = dyn_cast_or_null<AddressSpaceAttr>(memorySpace)) {
    return addressSpace.getAddressSpace() == AddressSpace::VEC;
  }
  if (auto integerSpace = dyn_cast_or_null<IntegerAttr>(memorySpace)) {
    return integerSpace.getInt() == static_cast<int64_t>(AddressSpace::VEC);
  }
  return false;
}

LogicalResult verifyMemoryElementMatches(Operation *op, Type memoryType,
                                                VMIVRegType dataType,
                                                StringRef role) {
  Type memoryElementType = getMemoryElementType(memoryType);
  if (!memoryElementType) {
    return success();
  }
  if (memoryElementType != dataType.getElementType()) {
    return op->emitOpError() << "requires memory " << role
                             << " element type to match VMI data element type";
  }
  return success();
}

bool isVMI8To16GatherPair(Type sourceElemType, Type resultElemType) {
  auto srcInt = dyn_cast<IntegerType>(sourceElemType);
  auto resInt = dyn_cast<IntegerType>(resultElemType);
  if (!srcInt || !resInt ||
      srcInt.getWidth() != mlir::pto::kValue8 ||
      resInt.getWidth() != mlir::pto::kValue16) {
    return false;
  }
  if (srcInt.isUnsigned()) {
    return resInt.isUnsigned();
  }
  return !resInt.isUnsigned();
}

LogicalResult verifyGatherMemoryElementMatches(
    Operation *op, Type memoryType, VMIVRegType dataType, StringRef role) {
  Type memoryElementType = getMemoryElementType(memoryType);
  if (!memoryElementType) {
    return success();
  }
  if (memoryElementType == dataType.getElementType()) {
    return success();
  }
  if (isVMI8To16GatherPair(memoryElementType, dataType.getElementType())) {
    return success();
  }
  return op->emitOpError()
         << "requires memory " << role
         << " element type to match VMI data element type"
            " or be an 8-bit integer promoted to a matching 16-bit integer";
}

bool isSameWidth16BitGatherPair(Type sourceElemType,
                                       Type resultElemType) {
  // Existing VMI f16/bf16 path: same-width 16-bit float gather.
  if (sourceElemType == resultElemType &&
      (sourceElemType.isF16() || sourceElemType.isBF16())) {
    return true;
  }
  auto srcInt = dyn_cast<IntegerType>(sourceElemType);
  auto resInt = dyn_cast<IntegerType>(resultElemType);
  // Existing VMI ui16/i16 path: same-width 16-bit integer gather with matching
  // integer semantics (signless i16 / i16 is accepted as the non-unsigned side).
  if (!srcInt || !resInt ||
      srcInt.getWidth() != mlir::pto::kValue16 ||
      resInt.getWidth() != mlir::pto::kValue16) {
    return false;
  }
  if (srcInt.isUnsigned()) {
    return resInt.isUnsigned();
  }
  return !resInt.isUnsigned();
}

LogicalResult verifyContiguousIfLayoutAssigned(Operation *op,
                                                      VMIVRegType type,
                                                      StringRef role) {
  VMILayoutAttr layout = type.getLayoutAttr();
  if (layout && !layout.isContiguous()) {
    return op->emitOpError()
           << "requires layout-assigned " << role
           << " to use #pto.vmi.layout<contiguous>";
  }
  return success();
}

bool isPackedByteGroupStore(Type memoryType, VMIVRegType dataType) {
  Type memoryElementType = getMemoryElementType(memoryType);
  if (!memoryElementType) {
    return false;
  }
  auto memoryIntegerType = dyn_cast<IntegerType>(memoryElementType);
  auto dataIntegerType = dyn_cast<IntegerType>(dataType.getElementType());
  return memoryIntegerType && dataIntegerType &&
         memoryIntegerType.getWidth() == mlir::pto::kValue8 && dataIntegerType.getWidth() == mlir::pto::kValue32;
}

LogicalResult verifyNumGroups(Operation *op, VMIVRegType type,
                                     int64_t numGroups) {
  if (numGroups <= 0) {
    return op->emitOpError("requires num_groups to be positive");
  }
  if (type.getElementCount() % numGroups != 0) {
    return op->emitOpError()
           << "requires num_groups to evenly divide VMI logical lane count "
           << type.getElementCount();
  }
  return success();
}

LogicalResult verifyPhysicalVRegParts(Operation *op,
                                             VMIVRegType vregType,
                                             TypeRange physicalTypes) {
  FailureOr<int64_t> lanesPerPart = getPhysicalLanesPerPart(vregType);
  Type physicalElementType = getVMIPhysicalDataElementType(vregType);
  if (failed(lanesPerPart)) {
    return op->emitOpError(
        "requires data element type with known physical lane count");
  }
  for (Type physicalType : physicalTypes) {
    auto partType = dyn_cast<VRegType>(physicalType);
    if (!partType) {
      return op->emitOpError("requires physical data parts to be !pto.vreg");
    }
    if (partType.getElementCount() != *lanesPerPart ||
        partType.getElementType() != physicalElementType) {
      return op->emitOpError(
          "requires physical data part type to match VMI lane-map helper");
    }
  }
  return success();
}

LogicalResult verifyPhysicalMaskParts(Operation *op,
                                             VMIMaskType maskType,
                                             TypeRange physicalTypes) {
  if (maskType.isPred()) {
    return op->emitOpError(
        "requires layout-assigned mask with concrete granularity");
  }
  FailureOr<StringRef> physicalGranularity =
      getVMIMaskPhysicalGranularity(maskType);
  if (failed(physicalGranularity)) {
    return op->emitOpError(
        "requires mask type with supported physical carrier granularity");
  }
  for (Type physicalType : physicalTypes) {
    auto partType = dyn_cast<MaskType>(physicalType);
    if (!partType) {
      return op->emitOpError("requires physical mask parts to be !pto.mask");
    }
    if (partType.getGranularity() != *physicalGranularity) {
      return op->emitOpError(
          "requires physical mask part granularity to match VMI mask carrier");
    }
  }
  return success();
}

LogicalResult verifyPhysicalParts(Operation *op, Type vmiType,
                                         TypeRange physicalTypes) {
  FailureOr<int64_t> expectedArity = getVMIPhysicalArity(vmiType);
  if (failed(expectedArity)) {
    return op->emitOpError(
        "requires a layout-assigned VMI type with computable physical arity");
  }
  if (static_cast<int64_t>(physicalTypes.size()) != *expectedArity) {
    return op->emitOpError() << "requires " << *expectedArity
                             << " physical parts, got " << physicalTypes.size();
  }
  if (auto vregType = dyn_cast<VMIVRegType>(vmiType)) {
    return verifyPhysicalVRegParts(op, vregType, physicalTypes);
  }
  auto maskType = dyn_cast<VMIMaskType>(vmiType);
  if (!maskType) {
    return op->emitOpError("requires VMI data or mask type");
  }
  return verifyPhysicalMaskParts(op, maskType, physicalTypes);
}

std::optional<int64_t>
mapDenseLogicalLaneToPartIndex(int64_t elementCount, int64_t factor,
                               int64_t blockElems, int64_t logicalLane,
                               int64_t &part) {
  if (logicalLane < 0 || logicalLane >= elementCount || factor <= 0 ||
      blockElems <= 0) {
    return std::nullopt;
  }
  int64_t block = logicalLane / blockElems;
  int64_t inBlockLane = logicalLane % blockElems;
  part = block % factor;
  int64_t partBlock = block / factor;
  return partBlock * blockElems + inBlockLane;
}

std::optional<int64_t>
mapDensePartIndexToLogicalLane(int64_t elementCount, int64_t factor,
                               int64_t blockElems, int64_t part,
                               int64_t indexInPart) {
  if (part < 0 || part >= factor || indexInPart < 0 || factor <= 0 ||
      blockElems <= 0) {
    return std::nullopt;
  }
  int64_t partBlock = indexInPart / blockElems;
  int64_t inBlockLane = indexInPart % blockElems;
  int64_t logicalBlock = partBlock * factor + part;
  int64_t logicalLane = logicalBlock * blockElems + inBlockLane;
  if (logicalLane >= elementCount) {
    return std::nullopt;
  }
  return logicalLane;
}

int64_t getDenseLogicalLanesInPart(int64_t elementCount, int64_t factor,
                                          int64_t blockElems, int64_t part) {
  int64_t maxIndex = -1;
  for (int64_t lane = 0; lane < elementCount; ++lane) {
    int64_t lanePart = 0;
    std::optional<int64_t> index = mapDenseLogicalLaneToPartIndex(
        elementCount, factor, blockElems, lane, lanePart);
    if (index && lanePart == part) {
      maxIndex = std::max(maxIndex, *index);
    }
  }
  return maxIndex + 1;
}

LogicalResult verifyReductionGroupAndPmode(
    Operation *op, VMIVRegType sourceType, VMIVRegType resultType,
    IntegerAttr groupAttr, std::optional<StringRef> pmode) {
  if (groupAttr) {
    int64_t C = groupAttr.getInt();
    if (C <= 0) {
      return op->emitOpError("group count must be positive");
    }
    if (sourceType.getElementCount() % C != 0) {
      return op->emitOpError("group count ") << C
                                            << " must divide source lane count "
                                            << sourceType.getElementCount();
    }
    if (resultType.getElementCount() != C) {
      return op->emitOpError("result lane count must equal group count ")
             << C << ", got " << resultType.getElementCount();
    }
    if (auto resultLayout = resultType.getLayoutAttr()) {
      if (!resultLayout.isGroupSlots() ||
          resultLayout.getNumGroups() != C) {
        return op->emitOpError()
               << "layout-assigned result must use "
                  "#pto.vmi.layout<num_groups = "
               << C << ">";
      }
    }
  } else if (resultType.getElementCount() != 1) {
    return op->emitOpError("full reduction (no group) requires 1-lane result, got ")
           << resultType.getElementCount();
  }
  if (sourceType.getElementType() != resultType.getElementType()) {
    return op->emitOpError("source and result element types must match");
  }
  if (pmode) {
    StringRef val = *pmode;
    if (val != "zero" && val != "merge") {
      return op->emitOpError("pmode must be \"zero\" or \"merge\", got \"")
             << val << "\"";
    }
  }
  return success();
}

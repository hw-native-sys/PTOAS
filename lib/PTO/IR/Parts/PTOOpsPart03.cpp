// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This implementation fragment is included by PTO.cpp and intentionally is
// not listed as a separate CMake translation unit.

static LogicalResult verifyMGatherMScatterTileShape(Operation *op, Type dataTy,
                                                    Type idxTy,
                                                    StringRef dataName,
                                                    std::optional<pto::Coalesce> coalesce) {
  auto dataValid = getValidShapeVec(dataTy);
  auto idxValid = getValidShapeVec(idxTy);
  if (dataValid.size() != 2 || idxValid.size() != 2)
    return op->emitOpError() << "expects " << dataName
                             << " and idx to have rank-2 valid_shape";

  auto idxTile = dyn_cast<pto::TileBufType>(idxTy);
  if (!idxTile)
    return op->emitOpError("expects idx to be a tile_buf type");

  const bool idxRowMajor =
      idxTile.getBLayoutValueI32() ==
      static_cast<int32_t>(pto::BLayout::RowMajor);
  const bool idxColMajor =
      idxTile.getBLayoutValueI32() ==
      static_cast<int32_t>(pto::BLayout::ColMajor);

  const bool rowCoalesce1xR =
      idxRowMajor && isKnownZeroOrUnitExtent(idxValid[0]) &&
      hasCompatibleKnownExtent(idxValid[1], dataValid[0]);
  const bool rowCoalesceRx1 =
      idxColMajor && hasCompatibleKnownExtent(idxValid[0], dataValid[0]) &&
      isKnownZeroOrUnitExtent(idxValid[1]);
  const bool baseRowCoalesce =
      idxRowMajor && hasCompatibleKnownExtent(idxValid[0], dataValid[0]) &&
      isKnownZeroOrUnitExtent(idxValid[1]);
  const bool elemCoalesce =
      hasCompatibleKnownExtent(idxValid[0], dataValid[0]) &&
      hasCompatibleKnownExtent(idxValid[1], dataValid[1]);

  if (!coalesce) {
    if (baseRowCoalesce || elemCoalesce)
      return success();
    return op->emitOpError()
           << "expects idx valid_shape to be [" << dataName
           << ".valid_row, 0|1] or match " << dataName
           << " valid_shape when coalesce is omitted";
  }

  if (*coalesce == pto::Coalesce::Row && (rowCoalesce1xR || rowCoalesceRx1))
    return success();

  if (*coalesce == pto::Coalesce::Elem && elemCoalesce)
    return success();

  if (*coalesce == pto::Coalesce::Row)
    return op->emitOpError()
           << "expects row-coalesce idx valid_shape to be [0|1, " << dataName
           << ".valid_row] or [" << dataName << ".valid_row, 0|1]";

  return op->emitOpError()
         << "expects elem-coalesce idx valid_shape to match " << dataName
         << " valid_shape";
}

template <typename AttrT>
static AttrT getPTOOpAttr(Operation *op, StringRef name) {
  if (Attribute propsAttr = op->getPropertiesAsAttribute()) {
    if (auto props = dyn_cast<DictionaryAttr>(propsAttr)) {
      if (auto attr = dyn_cast_or_null<AttrT>(props.get(name)))
        return attr;
    }
  }
  return dyn_cast_or_null<AttrT>(op->getRawDictionaryAttrs().get(name));
}

template <typename OpTy>
static ParseResult parsePTOInherentAttrs(OpAsmParser &parser,
                                         OperationState &result,
                                         NamedAttrList &parsedAttrs,
                                         ArrayRef<StringRef> inherentAttrNames) {
  auto attrLoc = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDict(parsedAttrs))
    return failure();

  auto &properties = result.getOrAddProperties<typename OpTy::Properties>();
  OpTy::populateDefaultProperties(result.name, properties);
  if (failed(OpTy::setPropertiesFromAttr(
          properties, parsedAttrs.getDictionary(parser.getContext()), [&] {
            return parser.emitError(attrLoc)
                   << "'" << result.name.getStringRef() << "' op ";
          })))
    return failure();

  for (StringRef attrName : inherentAttrNames)
    parsedAttrs.erase(attrName);
  result.attributes = parsedAttrs;
  return success();
}

static NamedAttrList getNonInherentAttrs(Operation *op,
                                         ArrayRef<StringRef> inherentAttrNames) {
  NamedAttrList attrs;
  for (NamedAttribute attr : op->getRawDictionaryAttrs()) {
    if (llvm::is_contained(inherentAttrNames, attr.getName().getValue()))
      continue;
    attrs.append(attr);
  }
  return attrs;
}

static pto::CoalesceAttr getMGatherCoalesceAttrIfPresent(pto::MGatherOp op) {
  return dyn_cast_or_null<pto::CoalesceAttr>(op.getProperties().coalesce);
}

static pto::GatherOOBAttr getMGatherGatherOobAttrIfPresent(pto::MGatherOp op) {
  return dyn_cast_or_null<pto::GatherOOBAttr>(op.getProperties().gatherOob);
}

static pto::GatherOOB getGatherOobOrDefault(pto::MGatherOp op) {
  if (auto attr = getMGatherGatherOobAttrIfPresent(op))
    return attr.getValue();
  return pto::GatherOOB::Undefined;
}

static pto::CoalesceAttr getMScatterCoalesceAttrIfPresent(pto::MScatterOp op) {
  return dyn_cast_or_null<pto::CoalesceAttr>(op.getProperties().coalesce);
}

static pto::ScatterAtomicOpAttr
getMScatterScatterAtomicOpAttrIfPresent(pto::MScatterOp op) {
  return dyn_cast_or_null<pto::ScatterAtomicOpAttr>(
      op.getProperties().scatterAtomicOp);
}

static pto::ScatterOOBAttr getMScatterScatterOobAttrIfPresent(
    pto::MScatterOp op) {
  return dyn_cast_or_null<pto::ScatterOOBAttr>(op.getProperties().scatterOob);
}

static pto::ScatterConflictAttr getMScatterScatterConflictAttrIfPresent(
    pto::MScatterOp op) {
  return dyn_cast_or_null<pto::ScatterConflictAttr>(
      op.getProperties().scatterConflict);
}

static std::optional<pto::Coalesce> getCoalesceIfPresent(pto::MGatherOp op) {
  if (auto attr = getMGatherCoalesceAttrIfPresent(op))
    return attr.getValue();
  return std::nullopt;
}

static std::optional<pto::Coalesce> getCoalesceIfPresent(pto::MScatterOp op) {
  if (auto attr = getMScatterCoalesceAttrIfPresent(op))
    return attr.getValue();
  return std::nullopt;
}

static pto::ScatterAtomicOp getScatterAtomicOpOrDefault(pto::MScatterOp op) {
  if (auto attr = getMScatterScatterAtomicOpAttrIfPresent(op))
    return attr.getValue();
  return pto::ScatterAtomicOp::None;
}

static pto::ScatterOOB getScatterOobOrDefault(pto::MScatterOp op) {
  if (auto attr = getMScatterScatterOobAttrIfPresent(op))
    return attr.getValue();
  return pto::ScatterOOB::Undefined;
}

static pto::ScatterConflictAttr getScatterConflictAttrIfPresent(
    pto::MScatterOp op) {
  return getMScatterScatterConflictAttrIfPresent(op);
}

static Value getTPrintTmpIfPresent(pto::TPrintOp op) {
  return op->getNumOperands() > 1 ? op->getOperand(1) : Value();
}

static LogicalResult verifyMGatherMScatterIdxTile(Operation *op, Type ty,
                                                  StringRef name) {
  if (failed(verifyTileBufCommon(op, ty, name)))
    return failure();
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::VEC)
    return op->emitOpError() << "expects " << name
                             << " to be in the vec address space";
  auto tb = dyn_cast<pto::TileBufType>(ty);
  if (!tb)
    return op->emitOpError() << "expects " << name << " to be a tile_buf type";
  int32_t blayout = tb.getBLayoutValueI32();
  if (blayout != static_cast<int32_t>(pto::BLayout::RowMajor) &&
      blayout != static_cast<int32_t>(pto::BLayout::ColMajor))
    return op->emitOpError() << "expects " << name
                             << " to use row_major or col_major blayout";
  if (tb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox))
    return op->emitOpError() << "expects " << name
                             << " to use the none_box slayout";
  return success();
}

static bool isA5TLoadStoreTransferElemType(Type ty) {
  return ty.isInteger(8) || ty.isInteger(16) || ty.isInteger(32) ||
         ty.isInteger(64) || ty.isF16() || ty.isBF16() || ty.isF32() ||
         isPTOLowPrecisionType(ty);
}

static bool isA5AccStorePreQuantDstType(Type srcElem, Type dstElem) {
  if (srcElem.isInteger(32))
    return dstElem.isInteger(8) || dstElem.isF16() || dstElem.isBF16();
  if (!srcElem.isF32())
    return false;
  return dstElem.isInteger(8) || dstElem.isF16() || dstElem.isBF16() ||
         dstElem.isF32() || isPTOHiFloat8Type(dstElem) ||
         dstElem.isFloat8E4M3() || dstElem.isFloat8E4M3FN() ||
         dstElem.isFloat8E4M3FNUZ() || dstElem.isFloat8E4M3B11FNUZ();
}

static bool isA5LowPrecisionTCvtPair(Type srcElem, Type dstElem) {
  if (srcElem.isF32())
    return isPTOFloat8Type(dstElem) || isPTOHiFloat8Type(dstElem);
  if (srcElem.isF16())
    return isPTOHiFloat8Type(dstElem);
  if (srcElem.isBF16())
    return isPTOFloat4PackedType(dstElem);
  if (isPTOFloat4PackedType(srcElem))
    return dstElem.isBF16();
  if (isPTOFloat8Type(srcElem) || isPTOHiFloat8Type(srcElem))
    return dstElem.isF32();
  return false;
}

static bool isA5SupportedTCvtPair(Type srcElem, Type dstElem) {
  if (isPTOLowPrecisionType(srcElem) || isPTOLowPrecisionType(dstElem))
    return isA5LowPrecisionTCvtPair(srcElem, dstElem);
  return true;
}

static LogicalResult verifyTileBufCommon(Operation *op, Type ty, StringRef name,
                                         bool allowLowPrecision) {
  auto tb = dyn_cast<pto::TileBufType>(ty);
  if (tb) {
    if (tb.getRank() != 2)
      return op->emitOpError() << "expects " << name << " to be a rank-2 tile_buf";
    Type elemTy = tb.getElementType();
    if (!allowLowPrecision && isPTOLowPrecisionType(elemTy))
      return op->emitOpError() << name << ": dtype " << elemTy
                               << " is not supported by this op yet";
  } else if (auto mr = dyn_cast<MemRefType>(ty)) {
    if (mr.getRank() != 2)
      return op->emitOpError() << "expects " << name << " to be a rank-2 memref";
    if (!allowLowPrecision && isPTOLowPrecisionType(mr.getElementType()))
      return op->emitOpError() << name << ": dtype " << mr.getElementType()
                               << " is not supported by this op yet";
  } else {
    return op->emitOpError() << "expects " << name << " to be a !pto.tile_buf or rank-2 memref";
  }

  auto validShape = getValidShapeVec(ty);
  if (validShape.size() != 2)
    return op->emitOpError() << "expects " << name << " to have a rank-2 valid_shape";
  auto shape = getShapeVec(ty);
  for (unsigned i = 0; i < 2; ++i) {
    if (shape[i] != ShapedType::kDynamic && validShape[i] != ShapedType::kDynamic &&
        validShape[i] > shape[i])
      return op->emitOpError() << "expects " << name << " to satisfy valid_shape[" << i
                               << "] <= shape[" << i << "]";
  }
  return success();
}

static LogicalResult verifyTileBufSameElemType(Operation *op, Type lhs, Type rhs,
                                               StringRef lhsName,
                                               StringRef rhsName) {
  if (!isTileLikeType(lhs) || !isTileLikeType(rhs))
    return op->emitOpError() << "expects " << lhsName << " and " << rhsName
                             << " to be !pto.tile_buf or memref";
  if (getElemTy(lhs) != getElemTy(rhs))
    return op->emitOpError() << "expects " << lhsName << " and " << rhsName
                             << " to have the same element type";
  return success();
}

static LogicalResult verifyTileBufSameValidShape(Operation *op, Type lhs, Type rhs,
                                                 StringRef lhsName, StringRef rhsName) {
  if (!isTileLikeType(lhs) || !isTileLikeType(rhs))
    return success();
  auto lhsValid = getValidShapeVec(lhs);
  auto rhsValid = getValidShapeVec(rhs);
  for (size_t i = 0; i < lhsValid.size() && i < rhsValid.size(); ++i) {
    if (lhsValid[i] != ShapedType::kDynamic && rhsValid[i] != ShapedType::kDynamic &&
        lhsValid[i] != rhsValid[i])
      return op->emitOpError() << "expects " << lhsName << " and " << rhsName
                               << " to have the same valid_shape";
  }
  if (lhsValid.size() != rhsValid.size())
    return op->emitOpError() << "expects " << lhsName << " and " << rhsName
                             << " to have the same valid_shape";
  return success();
}

static LogicalResult verifyTileBufSameLogicalExtent(Operation *op, Type lhs,
                                                    Type rhs, StringRef lhsName,
                                                    StringRef rhsName,
                                                    bool compareValidShape) {
  if (!isTileLikeType(lhs) || !isTileLikeType(rhs))
    return success();

  auto lhsExtent = getLogicalTileExtentVec(lhs, compareValidShape);
  auto rhsExtent = getLogicalTileExtentVec(rhs, compareValidShape);
  auto emitMismatch = [&]() -> LogicalResult {
    if (compareValidShape)
      return op->emitOpError() << "expects " << lhsName << " and " << rhsName
                               << " to have the same valid_shape";
    return op->emitOpError() << "expects " << lhsName << " and " << rhsName
                             << " to have compatible shapes";
  };
  if (lhsExtent.size() != rhsExtent.size())
    return emitMismatch();

  for (size_t i = 0, e = lhsExtent.size(); i < e; ++i) {
    if (lhsExtent[i] != ShapedType::kDynamic &&
        rhsExtent[i] != ShapedType::kDynamic && lhsExtent[i] != rhsExtent[i])
      return emitMismatch();
  }
  return success();
}

static LogicalResult verifyPartialValidPattern(Operation *op, Type src0Ty,
                                               Type src1Ty, Type dstTy) {
  auto src0Valid = getValidShapeVec(src0Ty);
  auto src1Valid = getValidShapeVec(src1Ty);
  auto dstValid = getValidShapeVec(dstTy);
  if (src0Valid.size() != 2 || src1Valid.size() != 2 || dstValid.size() != 2)
    return op->emitOpError("expects src0, src1, and dst to have rank-2 valid_shape");

  auto lessEqualKnown = [](int64_t lhs, int64_t rhs) {
    return lhs == ShapedType::kDynamic || rhs == ShapedType::kDynamic || lhs <= rhs;
  };
  auto equalsKnown = [](ArrayRef<int64_t> lhs, ArrayRef<int64_t> rhs) {
    for (auto [a, b] : llvm::zip(lhs, rhs)) {
      if (a != ShapedType::kDynamic && b != ShapedType::kDynamic && a != b)
        return false;
    }
    return true;
  };

  for (unsigned i = 0; i < 2; ++i) {
    if (!lessEqualKnown(src0Valid[i], dstValid[i]) ||
        !lessEqualKnown(src1Valid[i], dstValid[i]))
      return op->emitOpError(
          "expects src0/src1 valid_shape to be less than or equal to dst valid_shape");
  }
  if (!equalsKnown(src0Valid, dstValid) && !equalsKnown(src1Valid, dstValid))
    return op->emitOpError(
        "expects at least one of src0/src1 valid_shape to match dst valid_shape");
  return success();
}

static LogicalResult verifyPartialValidPatternLoose(Operation *op, Type src0Ty,
                                                    Type src1Ty, Type dstTy) {
  auto src0Valid = getValidShapeVec(src0Ty);
  auto src1Valid = getValidShapeVec(src1Ty);
  auto dstValid = getValidShapeVec(dstTy);
  if (src0Valid.size() != 2 || src1Valid.size() != 2 || dstValid.size() != 2)
    return op->emitOpError("expects src0, src1, and dst to have rank-2 valid_shape");

  auto lessEqualKnown = [](int64_t lhs, int64_t rhs) {
    return lhs == ShapedType::kDynamic || rhs == ShapedType::kDynamic || lhs <= rhs;
  };

  for (unsigned i = 0; i < 2; ++i) {
    if (!lessEqualKnown(src0Valid[i], dstValid[i]) ||
        !lessEqualKnown(src1Valid[i], dstValid[i]))
      return op->emitOpError(
          "expects src0/src1 valid_shape to be less than or equal to dst valid_shape");
  }
  return success();
}

[[maybe_unused]] static bool hasKnownZeroValidRegion(Type ty) {
  auto valid = getValidShapeVec(ty);
  if (valid.size() != 2)
    return false;
  return valid[0] == 0 || valid[1] == 0;
}

static LogicalResult verifyScalarTileOp(Operation *op, Type srcTy, Type dstTy,
                                        StringRef srcName, StringRef dstName,
                                        bool requireValidRowsEqual,
                                        bool requireValidColsEqual) {
  if (failed(verifyTileBufCommon(op, srcTy, srcName)) ||
      failed(verifyTileBufCommon(op, dstTy, dstName)))
    return failure();
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  if (!srcSpace || *srcSpace != pto::AddressSpace::VEC)
    return op->emitOpError() << "expects " << srcName
                             << " to be in the vec address space";
  if (!dstSpace || *dstSpace != pto::AddressSpace::VEC)
    return op->emitOpError() << "expects " << dstName
                             << " to be in the vec address space";
  if (failed(verifyTileBufSameElemType(op, srcTy, dstTy, srcName, dstName)))
    return failure();

  auto srcValid = getValidShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (srcValid.size() != 2 || dstValid.size() != 2)
    return op->emitOpError()
           << "expects " << srcName << " and " << dstName
           << " to have rank-2 valid_shape";
  if (requireValidRowsEqual &&
      srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
      srcValid[0] != dstValid[0])
    return op->emitOpError()
           << "expects " << srcName << " and " << dstName
           << " to have the same valid_shape[0]";
  if (requireValidColsEqual &&
      srcValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
      srcValid[1] != dstValid[1])
    return op->emitOpError()
           << "expects " << srcName << " and " << dstName
           << " to have the same valid_shape[1]";
  return success();
}

static FailureOr<Type>
verifyMatchingRowMajorBinaryTileOpCommon(Operation *op, Type src0Ty, Type src1Ty,
                                         Type dstTy) {
  if (failed(verifyTileBufCommon(op, src0Ty, "src0")) ||
      failed(verifyTileBufCommon(op, src1Ty, "src1")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst")))
    return failure();
  if (failed(verifyTileBufSameElemType(op, src0Ty, src1Ty, "src0", "src1")) ||
      failed(verifyTileBufSameElemType(op, src0Ty, dstTy, "src0", "dst")) ||
      failed(verifyTileBufSameValidShape(op, src0Ty, src1Ty, "src0", "src1")) ||
      failed(verifyTileBufSameValidShape(op, src0Ty, dstTy, "src0", "dst")))
    return failure();
  if (!isRowMajorTileBuf(src0Ty) || !isRowMajorTileBuf(src1Ty) ||
      !isRowMajorTileBuf(dstTy)) {
    op->emitOpError("expects src0, src1, and dst to use row-major layout");
    return failure();
  }
  return getElemTy(src0Ty);
}

static FailureOr<Type>
verifyNumericScalarTileOpCommon(Operation *op, Type srcTy, Type dstTy,
                                Type scalarTy, bool requireValidRowsEqual) {
  if (failed(verifyScalarTileOp(op, srcTy, dstTy, "src", "dst",
                                requireValidRowsEqual,
                                /*requireValidColsEqual=*/true)))
    return failure();
  if (!mlir::isa<IntegerType, FloatType>(scalarTy)) {
    op->emitOpError("scalar must be a scalar type (integer/float)");
    return failure();
  }
  return getElemTy(srcTy);
}

static FailureOr<Type>
verifyShiftLikeBinaryTileOpCommon(Operation *op, Type src0Ty, Type src1Ty,
                                   Type dstTy) {
  if (failed(verifyTileBufCommon(op, src0Ty, "src0")) ||
      failed(verifyTileBufCommon(op, src1Ty, "src1")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst")))
    return failure();
  Type e0 = getElemTy(src0Ty);
  Type e1 = getElemTy(src1Ty);
  Type ed = getElemTy(dstTy);
  if (!e0 || !e1 || !ed) {
    op->emitOpError("failed to get element type for operands");
    return failure();
  }
  if (e0 != e1 || e0 != ed) {
    op->emitOpError("expects src0, src1, and dst to have the same element type");
    return failure();
  }
  if (!isRowMajorTileBuf(src0Ty) || !isRowMajorTileBuf(src1Ty) ||
      !isRowMajorTileBuf(dstTy)) {
    op->emitOpError("expects src0, src1, and dst to use row-major layout");
    return failure();
  }
  if (failed(verifyTileBufSameValidShape(op, src0Ty, dstTy, "src0", "dst")) ||
      failed(verifyTileBufSameValidShape(op, src1Ty, dstTy, "src1", "dst")))
    return failure();
  return e0;
}

static FailureOr<Type> verifyDistinctRowMajorUnaryTileOpCommon(
    Operation *op, Value src, Value dst, StringRef srcName = "src",
    StringRef dstName = "dst") {
  if (src == dst) {
    op->emitOpError("expects src and dst to use different storage");
    return failure();
  }
  Type srcTy = src.getType();
  Type dstTy = dst.getType();
  if (failed(verifyTileBufCommon(op, srcTy, srcName)) ||
      failed(verifyTileBufCommon(op, dstTy, dstName)))
    return failure();

  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  if (!srcElem || !dstElem) {
    op->emitOpError("failed to get element type for src/dst");
    return failure();
  }
  if (srcElem != dstElem) {
    op->emitOpError("expects src and dst to have the same element type");
    return failure();
  }
  if (!isRowMajorTileBuf(srcTy) || !isRowMajorTileBuf(dstTy)) {
    op->emitOpError("expects src and dst to use row-major layout");
    return failure();
  }
  if (failed(verifyTileBufSameValidShape(op, srcTy, dstTy, srcName, dstName)))
    return failure();
  return srcElem;
}

static LogicalResult verifyArithmeticElemTypeForArch(
    Operation *op, Type elemTy, PTOArch targetArch, bool allowInt8OnA5,
    bool allowBf16OnA5, StringRef a2a3Error, StringRef a5Error) {
  bool supported = elemTy.isInteger(32) || elemTy.isInteger(16) ||
                   elemTy.isF16() || elemTy.isF32();
  if (targetArch == PTOArch::A5)
    supported = supported || (allowInt8OnA5 && elemTy.isInteger(8)) ||
                (allowBf16OnA5 && elemTy.isBF16());
  if (supported)
    return success();
  return op->emitOpError(targetArch == PTOArch::A5 ? a5Error : a2a3Error);
}

static LogicalResult verifyArithmeticBinaryTileOpWithArchDispatch(
    Operation *op, Type src0Ty, Type src1Ty, Type dstTy, bool allowInt8OnA5,
    bool allowBf16OnA5, StringRef a2a3Error, StringRef a5Error) {
  auto verifyByArch = [&](PTOArch targetArch) -> LogicalResult {
    FailureOr<Type> elemOr =
        verifyMatchingRowMajorBinaryTileOpCommon(op, src0Ty, src1Ty, dstTy);
    if (failed(elemOr))
      return failure();
    return verifyArithmeticElemTypeForArch(op, *elemOr, targetArch,
                                           allowInt8OnA5, allowBf16OnA5,
                                           a2a3Error, a5Error);
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(op, verifyA2A3, verifyA5);
}

static LogicalResult verifyArithmeticScalarTileOpWithArchDispatch(
    Operation *op, Type srcTy, Type dstTy, Type scalarTy, bool allowInt8OnA5,
    bool allowBf16OnA5, StringRef a2a3Error, StringRef a5Error,
    bool requireValidRowsEqualOnA2A3 = true,
    bool requireValidRowsEqualOnA5 = false) {
  auto verifyByArch = [&](PTOArch targetArch,
                          bool requireValidRowsEqual) -> LogicalResult {
    FailureOr<Type> elemOr = verifyNumericScalarTileOpCommon(
        op, srcTy, dstTy, scalarTy, requireValidRowsEqual);
    if (failed(elemOr))
      return failure();
    return verifyArithmeticElemTypeForArch(op, *elemOr, targetArch,
                                           allowInt8OnA5, allowBf16OnA5,
                                           a2a3Error, a5Error);
  };
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyByArch(PTOArch::A3, requireValidRowsEqualOnA2A3);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyByArch(PTOArch::A5, requireValidRowsEqualOnA5);
  };
  return dispatchVerifierByArch(op, verifyA2A3, verifyA5);
}

static LogicalResult verifyTColReductionElemTypeForArch(
    Operation *op, Type elemTy, PTOArch targetArch, bool allowInt8OnA5,
    bool allowBf16OnA5, StringRef a2a3Error, StringRef a5Error) {
  bool ok = elemTy.isF16() || elemTy.isF32() || elemTy.isInteger(16) ||
            elemTy.isInteger(32);
  if (targetArch == PTOArch::A5)
    ok = ok || (allowInt8OnA5 && elemTy.isInteger(8)) ||
         (allowBf16OnA5 && elemTy.isBF16());
  if (ok)
    return success();
  return op->emitOpError(targetArch == PTOArch::A5 ? a5Error : a2a3Error);
}

static LogicalResult verifyTColReductionOpWithArchDispatch(
    Operation *op, Type srcTy, Type dstTy, bool requireNonZeroSrcOnA2A3,
    bool requireNonZeroSrcOnA5, bool allowInt8OnA5, bool allowBf16OnA5,
    StringRef a2a3Error, StringRef a5Error) {
  auto verifyByArch = [&](PTOArch targetArch,
                          bool requireNonZeroSrc) -> LogicalResult {
    if (failed(verifyNDStyleVecTile(op, srcTy, "src")) ||
        failed(verifyNDStyleVecTile(op, dstTy, "dst")))
      return failure();
    if (getElemTy(srcTy) != getElemTy(dstTy))
      return op->emitOpError("expects src and dst to have the same element type");
    if (failed(verifyColReductionValidRegion(op, srcTy, dstTy, requireNonZeroSrc)))
      return failure();
    Type elem = getElemTy(srcTy);
    return verifyTColReductionElemTypeForArch(op, elem, targetArch, allowInt8OnA5,
                                              allowBf16OnA5, a2a3Error, a5Error);
  };
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyByArch(PTOArch::A3, requireNonZeroSrcOnA2A3);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyByArch(PTOArch::A5, requireNonZeroSrcOnA5);
  };
  return dispatchVerifierByArch(op, verifyA2A3, verifyA5);
}

static bool hasCompatibleKnownExtent(int64_t lhs, int64_t rhs) {
  return lhs == ShapedType::kDynamic || rhs == ShapedType::kDynamic || lhs == rhs;
}

static bool isKnownUnitExtent(int64_t value) {
  return value == ShapedType::kDynamic || value == 1;
}

static bool isKnownZeroOrUnitExtent(int64_t value) {
  return value == ShapedType::kDynamic || value == 0 || value == 1;
}

static bool hasCompatibleKnownExtentOrZero(int64_t lhs, int64_t rhs) {
  return lhs == ShapedType::kDynamic || rhs == ShapedType::kDynamic ||
         lhs == 0 || lhs == rhs;
}

static LogicalResult verifyVecTileStorage(Operation *op, Type ty, StringRef name) {
  if (failed(verifyTileBufCommon(op, ty, name)))
    return failure();
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::VEC)
    return op->emitOpError() << "expects " << name << " to be in the vec address space";
  return success();
}
static LogicalResult verifyVecTileCommonA2A3(Operation *op, Type ty,
                                             StringRef name) {
  if (failed(verifyTileBufCommon(op, ty, name)))
    return failure();
  auto tb = dyn_cast<pto::TileBufType>(ty);
  auto as = getPTOMemorySpaceEnum(ty);
  if (as && *as != pto::AddressSpace::VEC)
    return op->emitOpError() << "expects " << name << " to be in the vec address space";
  if (tb && tb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor))
    return op->emitOpError() << "expects " << name << " to use the row_major blayout";
  return success();
}

static LogicalResult verifyVecTileCommonA5(Operation *op, Type ty,
                                           StringRef name) {
  return verifyVecTileCommonA2A3(op, ty, name);
}

static LogicalResult verifyVecTileCommon(Operation *op, Type ty, StringRef name) {
  switch (getVerifierTargetArch(op)) {
  case VerifierTargetArch::A2A3:
    return verifyVecTileCommonA2A3(op, ty, name);
  case VerifierTargetArch::A5:
    return verifyVecTileCommonA5(op, ty, name);
  }
  return failure();
}

static LogicalResult verifyVecTileUnaryOp(Operation *op, Type srcTy, Type dstTy,
                                          StringRef srcName,
                                          StringRef dstName,
                                          bool allowBf16,
                                          bool allowInt8) {
  if (failed(verifyVecTileCommon(op, srcTy, srcName)) ||
      failed(verifyVecTileCommon(op, dstTy, dstName)))
    return failure();
  if (failed(verifyTileBufSameElemType(op, srcTy, dstTy, srcName, dstName)))
    return failure();
  if (!isSupportedVecElemType(getElemTy(srcTy), allowBf16, allowInt8))
    return op->emitOpError() << "expects vec tile element types to be supported";
  return success();
}

static LogicalResult verifyAccTileCommonA2A3(Operation *op, Type ty,
                                             StringRef name) {
  if (failed(verifyTileBufCommon(op, ty, name)))
    return failure();
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::ACC)
    return op->emitOpError() << "expects " << name << " to be in the acc address space";
  return success();
}

static LogicalResult verifyAccTileCommonA5(Operation *op, Type ty,
                                           StringRef name) {
  return verifyAccTileCommonA2A3(op, ty, name);
}

static LogicalResult verifyAccTileCommon(Operation *op, Type ty, StringRef name) {
  switch (getVerifierTargetArch(op)) {
  case VerifierTargetArch::A2A3:
    return verifyAccTileCommonA2A3(op, ty, name);
  case VerifierTargetArch::A5:
    return verifyAccTileCommonA5(op, ty, name);
  }
  return failure();
}

static LogicalResult verifyMatTileOperandsA2A3(Operation *op, Type lhsTy,
                                               Type rhsTy, Type dstTy,
                                               bool allowLowPrecision) {
  if (failed(verifyTileBufCommon(op, lhsTy, "lhs", allowLowPrecision)) ||
      failed(verifyTileBufCommon(op, rhsTy, "rhs", allowLowPrecision)) ||
      failed(verifyAccTileCommon(op, dstTy, "dst")))
    return failure();
  auto lhsSpace = getPTOMemorySpaceEnum(lhsTy);
  auto rhsSpace = getPTOMemorySpaceEnum(rhsTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  if (!lhsSpace || !rhsSpace || !dstSpace)
    return op->emitOpError("expects lhs, rhs, and dst to have explicit address spaces");
  if (*lhsSpace != pto::AddressSpace::LEFT || *rhsSpace != pto::AddressSpace::RIGHT ||
      *dstSpace != pto::AddressSpace::ACC)
    return op->emitOpError(
        "expects lhs, rhs, and dst to use the left, right, and acc address spaces");
  auto lhsShape = getMatmulLogicalShapeVec(lhsTy);
  auto rhsShape = getMatmulLogicalShapeVec(rhsTy);
  auto dstShape = getMatmulLogicalShapeVec(dstTy);
  if ((lhsShape[0] != dstShape[0] || rhsShape[1] != dstShape[1] || lhsShape[1] != rhsShape[0]))
    return op->emitOpError(
        "expects static matmul tile shapes lhs[M,K], rhs[K,N], and dst[M,N]");
  auto lhsValid = getValidShapeVec(lhsTy);
  auto rhsValid = getValidShapeVec(rhsTy);
  if (lhsValid.size() == 2 && rhsValid.size() == 2) {
    int64_t m = lhsValid[0];
    int64_t k = lhsValid[1];
    int64_t n = rhsValid[1];
    if ((m != ShapedType::kDynamic && (m < 0 || m > 4095)) ||
        (k != ShapedType::kDynamic && (k < 0 || k > 4095)) ||
        (n != ShapedType::kDynamic && (n < 0 || n > 4095)))
      return op->emitOpError("expects m, k, and n valid sizes to be in [0, 4095]");
  }
  return success();
}

static LogicalResult verifyMatTileOperandsA5(Operation *op, Type lhsTy,
                                             Type rhsTy, Type dstTy,
                                             bool allowLowPrecision) {
  if (failed(verifyMatTileOperandsA2A3(op, lhsTy, rhsTy, dstTy,
                                       allowLowPrecision)))
    return failure();

  auto lhsTb = mlir::dyn_cast<pto::TileBufType>(lhsTy);
  auto rhsTb = mlir::dyn_cast<pto::TileBufType>(rhsTy);
  auto dstTb = mlir::dyn_cast<pto::TileBufType>(dstTy);
  if (!lhsTb || !rhsTb || !dstTb)
    return success();

  if (lhsTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor))
    return op->emitOpError("expects lhs to use the col_major blayout on A5");
  if (rhsTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor))
    return op->emitOpError("expects rhs to use the row_major blayout on A5");
  if (dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor))
    return op->emitOpError("expects dst to use the col_major blayout on A5");

  if (lhsTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor))
    return op->emitOpError("expects lhs to use the row_major slayout on A5");
  if (rhsTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::ColMajor))
    return op->emitOpError("expects rhs to use the col_major slayout on A5");
  if (dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor))
    return op->emitOpError("expects dst to use the row_major slayout on A5");
  return success();
}

static LogicalResult verifyMatTileOperands(Operation *op, Type lhsTy, Type rhsTy,
                                           Type dstTy,
                                           bool allowLowPrecision) {
  switch (getVerifierTargetArch(op)) {
  case VerifierTargetArch::A2A3:
    return verifyMatTileOperandsA2A3(op, lhsTy, rhsTy, dstTy,
                                     allowLowPrecision);
  case VerifierTargetArch::A5:
    return verifyMatTileOperandsA5(op, lhsTy, rhsTy, dstTy,
                                   allowLowPrecision);
  }
  return failure();
}

static LogicalResult verifyGemvTileOperandsA2A3(Operation *op, Type lhsTy,
                                                Type rhsTy, Type dstTy) {
  if (failed(verifyTileBufCommon(op, lhsTy, "lhs")) ||
      failed(verifyTileBufCommon(op, rhsTy, "rhs")) ||
      failed(verifyAccTileCommon(op, dstTy, "dst")))
    return failure();

  auto lhsSpace = getPTOMemorySpaceEnum(lhsTy);
  auto rhsSpace = getPTOMemorySpaceEnum(rhsTy);
  if (!lhsSpace || !rhsSpace)
    return op->emitOpError("expects lhs and rhs to have explicit address spaces");
  if (*lhsSpace != pto::AddressSpace::LEFT || *rhsSpace != pto::AddressSpace::RIGHT)
    return op->emitOpError(
        "expects lhs and rhs to use the left and right address spaces");

  auto lhsValid = getValidShapeVec(lhsTy);
  auto rhsValid = getValidShapeVec(rhsTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (lhsValid[0] != ShapedType::kDynamic && lhsValid[0] != 1)
    return op->emitOpError("expects lhs valid_shape[0] to be 1 for tgemv");
  if (isa<pto::TileBufType>(dstTy) && dstValid[0] != ShapedType::kDynamic &&
      dstValid[0] != 1)
    return op->emitOpError("expects dst valid_shape[0] to be 1 for tgemv");
  if (lhsValid[1] != ShapedType::kDynamic && rhsValid[0] != ShapedType::kDynamic &&
      lhsValid[1] != rhsValid[0])
    return op->emitOpError()
           << "expects lhs valid_shape[1] to equal rhs valid_shape[0], but got "
           << lhsValid[1] << " vs " << rhsValid[0];
  if (rhsValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
      rhsValid[1] != dstValid[1])
    return op->emitOpError()
           << "expects rhs valid_shape[1] to equal dst valid_shape[1], but got "
           << rhsValid[1] << " vs " << dstValid[1];
  return success();
}

static LogicalResult verifyGemvTileOperandsA5(Operation *op, Type lhsTy,
                                              Type rhsTy, Type dstTy) {
  if (failed(verifyGemvTileOperandsA2A3(op, lhsTy, rhsTy, dstTy)))
    return failure();
  return verifyMatTileOperandsA5(op, lhsTy, rhsTy, dstTy);
}

static LogicalResult verifyGemvTileOperands(Operation *op, Type lhsTy, Type rhsTy,
                                            Type dstTy) {
  switch (getVerifierTargetArch(op)) {
  case VerifierTargetArch::A2A3:
    return verifyGemvTileOperandsA2A3(op, lhsTy, rhsTy, dstTy);
  case VerifierTargetArch::A5:
    return verifyGemvTileOperandsA5(op, lhsTy, rhsTy, dstTy);
  }
  return failure();
}

static LogicalResult verifyA5MxMatTileOperands(Operation *op, Type lhsTy,
                                               Type rhsTy, Type dstTy) {
  if (failed(verifyMatTileOperandsA5(op, lhsTy, rhsTy, dstTy,
                                     /*allowLowPrecision=*/true)))
    return failure();

  auto lhsShape = getShapeVec(lhsTy);
  auto rhsShape = getShapeVec(rhsTy);
  if (lhsShape.size() == 2 && rhsShape.size() == 2) {
    int64_t lhsK = lhsShape[1];
    int64_t rhsK = rhsShape[0];
    auto checkPhysicalK = [&](int64_t value, StringRef name) -> LogicalResult {
      if (value != ShapedType::kDynamic && (value < 1 || (value % 64) != 0))
        return op->emitOpError() << "expects " << name
                                 << " physical K shape to be a positive multiple of 64 on A5";
      return success();
    };
    if (failed(checkPhysicalK(lhsK, "lhs")))
      return failure();
    if (failed(checkPhysicalK(rhsK, "rhs")))
      return failure();
  }

  auto lhsValid = getValidShapeVec(lhsTy);
  auto rhsValid = getValidShapeVec(rhsTy);
  if (lhsValid.size() == 2 && rhsValid.size() == 2) {
    int64_t m = lhsValid[0];
    int64_t k = lhsValid[1];
    int64_t n = rhsValid[1];
    if ((m != ShapedType::kDynamic && (m < 1 || m > 4095)) ||
        (k != ShapedType::kDynamic && (k < 1 || k > 4095)) ||
        (n != ShapedType::kDynamic && (n < 1 || n > 4095)))
      return op->emitOpError("expects m, k, and n valid sizes to be in [1, 4095]");
  }
  return success();
}

static int64_t ceilDivKnown(int64_t value, int64_t divisor) {
  if (value == ShapedType::kDynamic)
    return ShapedType::kDynamic;
  return (value + divisor - 1) / divisor;
}

static LogicalResult verifyA5MxMatScaleTile(Operation *op, Type scaleTy,
                                            Type lhsTy, Type rhsTy,
                                            StringRef scaleName,
                                            bool isLeftScale) {
  if (failed(verifyTileBufCommon(op, scaleTy, scaleName,
                                 /*allowLowPrecision=*/true)))
    return failure();
  auto scaleSpace = getPTOMemorySpaceEnum(scaleTy);
  if (!scaleSpace || *scaleSpace != pto::AddressSpace::SCALING)
    return op->emitOpError() << "expects " << scaleName
                             << " to be in the scaling address space";

  auto checkDims = [&](ArrayRef<int64_t> scaleDims, ArrayRef<int64_t> lhsDims,
                       ArrayRef<int64_t> rhsDims, StringRef dimsName) -> LogicalResult {
    if (scaleDims.size() != 2 || lhsDims.size() != 2 || rhsDims.size() != 2)
      return op->emitOpError() << "expects " << scaleName << ", lhs, and rhs to have rank-2 "
                               << dimsName;

    int64_t m = lhsDims[0];
    int64_t k = lhsDims[1];
    int64_t n = rhsDims[1];
    int64_t scaleK = ceilDivKnown(k, 32);
    int64_t expectedRows = isLeftScale ? m : scaleK;
    int64_t expectedCols = isLeftScale ? scaleK : n;
    if (!hasCompatibleKnownExtent(scaleDims[0], expectedRows) ||
        !hasCompatibleKnownExtent(scaleDims[1], expectedCols)) {
      return op->emitOpError()
             << "expects " << scaleName << " " << dimsName << " to be "
             << (isLeftScale ? "[M, ceil(K/32)]" : "[ceil(K/32), N]");
    }
    return success();
  };

  if (failed(checkDims(getShapeVec(scaleTy), getShapeVec(lhsTy), getShapeVec(rhsTy),
                       "shape")))
    return failure();
  if (failed(checkDims(getValidShapeVec(scaleTy), getValidShapeVec(lhsTy),
                       getValidShapeVec(rhsTy), "valid_shape")))
    return failure();

  auto scaleTb = dyn_cast<pto::TileBufType>(scaleTy);
  if (!scaleTb)
    return success();
  if (scaleTb.getBLayoutValueI32() !=
      static_cast<int32_t>(isLeftScale ? pto::BLayout::RowMajor
                                       : pto::BLayout::ColMajor)) {
    return op->emitOpError()
           << "expects " << scaleName << " to use the "
           << (isLeftScale ? "row_major" : "col_major")
           << " blayout on A5";
  }
  if (scaleTb.getSLayoutValueI32() !=
      static_cast<int32_t>(isLeftScale ? pto::SLayout::RowMajor
                                       : pto::SLayout::ColMajor)) {
    return op->emitOpError()
           << "expects " << scaleName << " to use the "
           << (isLeftScale ? "row_major" : "col_major")
           << " slayout on A5";
  }
  if (scaleTb.getSFractalSizeI32() != 32)
    return op->emitOpError() << "expects " << scaleName
                             << " to use fractal=32 on A5";
  return success();
}

static LogicalResult verifyA5MxMatScaleTiles(Operation *op, Type lhsScaleTy,
                                             Type rhsScaleTy, Type lhsTy,
                                             Type rhsTy) {
  if (failed(verifyA5MxMatScaleTile(op, lhsScaleTy, lhsTy, rhsTy, "a_scale",
                                    /*isLeftScale=*/true)))
    return failure();
  return verifyA5MxMatScaleTile(op, rhsScaleTy, lhsTy, rhsTy, "b_scale",
                                /*isLeftScale=*/false);
}

static LogicalResult verifyA5MxGemvTileOperands(Operation *op, Type lhsTy,
                                                Type rhsTy, Type dstTy) {
  if (failed(verifyTileBufCommon(op, lhsTy, "lhs", /*allowLowPrecision=*/true)) ||
      failed(verifyTileBufCommon(op, rhsTy, "rhs", /*allowLowPrecision=*/true)) ||
      failed(verifyAccTileCommon(op, dstTy, "dst")))
    return failure();

  auto lhsSpace = getPTOMemorySpaceEnum(lhsTy);
  auto rhsSpace = getPTOMemorySpaceEnum(rhsTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  if (!lhsSpace || !rhsSpace || !dstSpace)
    return op->emitOpError("expects lhs, rhs, and dst to have explicit address spaces");
  if (*lhsSpace != pto::AddressSpace::LEFT || *rhsSpace != pto::AddressSpace::RIGHT ||
      *dstSpace != pto::AddressSpace::ACC)
    return op->emitOpError(
        "expects lhs, rhs, and dst to use the left, right, and acc address spaces");

  auto lhsShape = getMatmulLogicalShapeVec(lhsTy);
  auto rhsShape = getMatmulLogicalShapeVec(rhsTy);
  auto dstShape = getMatmulLogicalShapeVec(dstTy);
  if ((lhsShape[0] != dstShape[0] || rhsShape[1] != dstShape[1] ||
       lhsShape[1] != rhsShape[0]))
    return op->emitOpError(
        "expects static matmul tile shapes lhs[M,K], rhs[K,N], and dst[M,N]");

  auto lhsValid = getValidShapeVec(lhsTy);
  auto rhsValid = getValidShapeVec(rhsTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (lhsValid.size() == 2 && rhsValid.size() == 2) {
    int64_t m = lhsValid[0];
    int64_t k = lhsValid[1];
    int64_t n = rhsValid[1];
    if ((m != ShapedType::kDynamic && (m < 1 || m > 4095)) ||
        (k != ShapedType::kDynamic && (k < 1 || k > 4095)) ||
        (n != ShapedType::kDynamic && (n < 1 || n > 4095)))
      return op->emitOpError("expects m, k, and n valid sizes to be in [1, 4095]");
  }

  if (lhsValid[0] != ShapedType::kDynamic && lhsValid[0] != 1)
    return op->emitOpError("expects lhs valid_shape[0] to be 1 for tgemv");
  if (dstValid[0] != ShapedType::kDynamic && dstValid[0] != 1)
    return op->emitOpError("expects dst valid_shape[0] to be 1 for tgemv");
  if (lhsValid[1] != ShapedType::kDynamic && rhsValid[0] != ShapedType::kDynamic &&
      lhsValid[1] != rhsValid[0])
    return op->emitOpError()
           << "expects lhs valid_shape[1] to equal rhs valid_shape[0], but got "
           << lhsValid[1] << " vs " << rhsValid[0];
  if (rhsValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
      rhsValid[1] != dstValid[1])
    return op->emitOpError()
           << "expects rhs valid_shape[1] to equal dst valid_shape[1], but got "
           << rhsValid[1] << " vs " << dstValid[1];

  auto lhsTb = dyn_cast<pto::TileBufType>(lhsTy);
  auto rhsTb = dyn_cast<pto::TileBufType>(rhsTy);
  auto dstTb = dyn_cast<pto::TileBufType>(dstTy);
  if (!lhsTb || !rhsTb || !dstTb)
    return success();

  if (lhsTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor))
    return op->emitOpError("expects lhs to use the col_major blayout on A5");
  if (rhsTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor))
    return op->emitOpError("expects rhs to use the row_major blayout on A5");
  if (dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor))
    return op->emitOpError("expects dst to use the col_major blayout on A5");

  if (lhsTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor))
    return op->emitOpError("expects lhs to use the row_major slayout on A5");
  if (rhsTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::ColMajor))
    return op->emitOpError("expects rhs to use the col_major slayout on A5");
  if (dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor))
    return op->emitOpError("expects dst to use the row_major slayout on A5");
  return success();
}

static LogicalResult verifyA5MxGemvScaleTile(Operation *op, Type scaleTy,
                                             Type lhsTy, Type rhsTy,
                                             StringRef scaleName,
                                             bool isLeftScale) {
  if (failed(verifyTileBufCommon(op, scaleTy, scaleName,
                                 /*allowLowPrecision=*/true)))
    return failure();
  auto scaleSpace = getPTOMemorySpaceEnum(scaleTy);
  if (!scaleSpace || *scaleSpace != pto::AddressSpace::SCALING)
    return op->emitOpError() << "expects " << scaleName
                             << " to be in the scaling address space";

  auto scaleShape = getShapeVec(scaleTy);
  auto scaleValid = getValidShapeVec(scaleTy);
  auto lhsShape = getShapeVec(lhsTy);
  auto rhsShape = getShapeVec(rhsTy);
  auto lhsValid = getValidShapeVec(lhsTy);
  auto rhsValid = getValidShapeVec(rhsTy);
  if (scaleShape.size() != 2 || scaleValid.size() != 2 ||
      lhsShape.size() != 2 || rhsShape.size() != 2 || lhsValid.size() != 2 ||
      rhsValid.size() != 2)
    return op->emitOpError() << "expects " << scaleName
                             << ", lhs, and rhs to have rank-2 shape/valid_shape";

  int64_t logicalM = lhsValid[0];
  int64_t logicalK = lhsValid[1];
  int64_t logicalN = rhsValid[1];
  int64_t scaleK = ceilDivKnown(logicalK, 32);

  int64_t expectedShapeRows = isLeftScale ? logicalM : scaleK;
  int64_t expectedShapeCols = isLeftScale ? scaleK : rhsShape[1];
  int64_t expectedValidRows = isLeftScale ? logicalM : scaleK;
  int64_t expectedValidCols = isLeftScale ? scaleK : logicalN;

  if (!hasCompatibleKnownExtent(scaleShape[0], expectedShapeRows) ||
      !hasCompatibleKnownExtent(scaleShape[1], expectedShapeCols) ||
      !hasCompatibleKnownExtent(scaleValid[0], expectedValidRows) ||
      !hasCompatibleKnownExtent(scaleValid[1], expectedValidCols)) {
    if (isLeftScale)
      return op->emitOpError()
             << "expects " << scaleName
             << " shape/valid_shape to be [M, ceil(K/32)]";
    return op->emitOpError()
           << "expects " << scaleName
           << " shape/valid_shape to be [ceil(K/32), aligned_N]/[ceil(K/32), N]";
  }
  return success();
}

static LogicalResult verifyMatBiasTileA2A3(Operation *op, Type biasTy, Type dstTy,
                                           bool requireFloatBias) {
  if (failed(verifyTileBufCommon(op, biasTy, "bias")))
    return failure();
  auto biasSpace = getPTOMemorySpaceEnum(biasTy);
  if (!biasSpace || *biasSpace != pto::AddressSpace::BIAS)
    return op->emitOpError("expects bias to be in the bias address space");
  auto biasShape = getShapeVec(biasTy);
  if (biasShape[0] != ShapedType::kDynamic && biasShape[0] != 1)
    return op->emitOpError("expects bias to have 1 row");
  if (requireFloatBias) {
    if (!getElemTy(biasTy).isF32())
      return op->emitOpError("expects bias to have element type f32");
  } else if (getElemTy(biasTy) != getElemTy(dstTy)) {
    return op->emitOpError("expects bias and dst to have the same element type");
  }
  return success();
}

static LogicalResult verifyMatBiasTileA5(Operation *op, Type biasTy, Type dstTy,
                                         bool requireFloatBias) {
  if (failed(verifyMatBiasTileA2A3(op, biasTy, dstTy, requireFloatBias)))
    return failure();
  if (auto biasTb = dyn_cast<pto::TileBufType>(biasTy)) {
    if (biasTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor))
      return op->emitOpError("expects bias to use the row_major blayout on A5");
  }
  return success();
}

static LogicalResult verifyMatBiasTile(Operation *op, Type biasTy, Type dstTy,
                                       bool requireFloatBias) {
  switch (getVerifierTargetArch(op)) {
  case VerifierTargetArch::A2A3:
    return verifyMatBiasTileA2A3(op, biasTy, dstTy, requireFloatBias);
  case VerifierTargetArch::A5:
    return verifyMatBiasTileA5(op, biasTy, dstTy, requireFloatBias);
  }
  return failure();
}

static LogicalResult verifyMatmulTypeTriple(Operation *op, Type lhsElemTy,
                                            Type rhsElemTy, Type dstElemTy) {
  bool isA5 = getVerifierTargetArch(op) == VerifierTargetArch::A5;
  auto isInt8 = [](Type ty) {
    return ty.isInteger(8);
  };
  if (dstElemTy.isInteger(32) && isInt8(lhsElemTy) && isInt8(rhsElemTy))
    return success();

  auto isSupportedFpInput = [](Type ty) {
    return ty.isF16() || ty.isBF16() || ty.isF32();
  };
  if (dstElemTy.isF32() && lhsElemTy == rhsElemTy && isSupportedFpInput(lhsElemTy))
    return success();

  auto isA5TMatmulFp8Type = [](Type ty) {
    if (auto ft = mlir::dyn_cast<FloatType>(ty))
      return ft.isFloat8E4M3() || ft.isFloat8E4M3FN() ||
             ft.isFloat8E4M3FNUZ() || ft.isFloat8E4M3B11FNUZ() ||
             ft.isFloat8E5M2() || ft.isFloat8E5M2FNUZ();
    return false;
  };
  if (isA5 && dstElemTy.isF32()) {
    if (isA5TMatmulFp8Type(lhsElemTy) && isA5TMatmulFp8Type(rhsElemTy))
      return success();
    if (isPTOHiFloat8Type(lhsElemTy) && lhsElemTy == rhsElemTy)
      return success();
  }

  return op->emitOpError()
         << "expects (dst, lhs, rhs) element types to match one of "
            "(i32, i8, i8), (f32, f16, f16), (f32, bf16, bf16), (f32, f32, f32)"
            << (isA5 ? ", (f32, fp8, fp8), or (f32, hif8, hif8)" : "");
}

LogicalResult pto::TAddOp::verify() {
  return verifyArithmeticBinaryTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tadd element type to be i32/i16/f16/f32",
      "expects A5 tadd element type to be i32/i16/i8/f16/bf16/f32");
}

LogicalResult pto::TAddCOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type t0 = getSrc0().getType();
  Type t1 = getSrc1().getType();
  Type t2 = getSrc2().getType();
  Type td = getDst().getType();

  if (!isPTOShapedLike(t0) || !isPTOShapedLike(t1) ||
      !isPTOShapedLike(t2) || !isPTOShapedLike(td))
    return emitOpError("expects src0/src1/src2/dst to be memref/tile_buf types");

  auto s0 = getShapeVec(t0);
  auto s1 = getShapeVec(t1);
  auto s2 = getShapeVec(t2);
  auto sd = getShapeVec(td);
  if (s0 != s1 || s0 != s2 || s0 != sd)
    return emitOpError("expects src0/src1/src2/dst to have the same shape");
  return success();
}
LogicalResult pto::TAddSOp::verify() {
  return verifyArithmeticScalarTileOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(), getScalar().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tadds element type to be i32/i16/f16/f32",
      "expects A5 tadds element type to be i32/i16/i8/f16/bf16/f32",
      /*requireValidRowsEqualOnA2A3=*/true,
      /*requireValidRowsEqualOnA5=*/true);
}

LogicalResult pto::TAxpyOp::verify() {
  auto verifyCommon = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
        failed(verifyVecTileCommon(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
      return failure();

    Type scalarTy = getScalar().getType();
    Type srcElem = getElemTy(srcTy);
    if (scalarTy != srcElem)
      return emitOpError("expects scalar type to match src element type");
    if (getShapeVec(srcTy) != getShapeVec(dstTy))
      return emitOpError("expects src and dst to have the same shape");
    return success();
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyCommon()))
      return failure();
    Type srcElem = getElemTy(getSrc().getType());
    Type dstElem = getElemTy(getDst().getType());
    bool sameType = srcElem == dstElem;
    bool widenF16ToF32 = srcElem.isF16() && dstElem.isF32();
    if (!(sameType || widenF16ToF32))
      return emitOpError(
          "expects dst/src element types to match, or dst=f32 and src=f16");
    if (!(dstElem.isF16() || dstElem.isF32()))
      return emitOpError("expects A2/A3 taxpy dst element type to be f16/f32");
    if (!(srcElem.isF16() || srcElem.isF32()))
      return emitOpError("expects A2/A3 taxpy src element type to be f16/f32");
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyCommon()))
      return failure();
    Type srcElem = getElemTy(getSrc().getType());
    Type dstElem = getElemTy(getDst().getType());
    bool sameType = srcElem == dstElem;
    bool widenF16ToF32 = srcElem.isF16() && dstElem.isF32();
    if (!(sameType || widenF16ToF32))
      return emitOpError(
          "expects dst/src element types to match, or dst=f32 and src=f16");
    if (!(dstElem.isF16() || dstElem.isF32() || dstElem.isBF16()))
      return emitOpError("expects A5 taxpy dst element type to be f16/bf16/f32");
    if (!(srcElem.isF16() || srcElem.isF32() || srcElem.isBF16()))
      return emitOpError("expects A5 taxpy src element type to be f16/bf16/f32");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult pto::TAddSCOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type ts0 = getSrc0().getType();
  Type ts1 = getSrc1().getType();
  Type td = getDst().getType();
  if (!isPTOShapedLike(ts0) || !isPTOShapedLike(ts1) || !isPTOShapedLike(td))
    return emitOpError("expects src0/src1/dst to be PTO shaped-like types");

  auto s0 = getShapeVec(ts0);
  auto s1 = getShapeVec(ts1);
  auto sd = getShapeVec(td);
  if (s0 != s1 || s0 != sd)
    return emitOpError("expects src0/src1/dst to have the same shape");
  return success();
}

LogicalResult pto::TAndOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<Type> {
    return verifyMatchingRowMajorBinaryTileOpCommon(
        getOperation(), getSrc0().getType(), getSrc1().getType(),
        getDst().getType());
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16))
      return emitOpError(
          "expects A2/A3 tand src0, src1, and dst element type to be i8/i16");
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
                it.getWidth() != 32))
      return emitOpError(
          "expects A5 tand src0, src1, and dst element type to be i8/i16/i32");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TConcatOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<Type> {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type td = getDst().getType();
    if (failed(verifyTileBufCommon(*this, t0, "src0")) ||
        failed(verifyTileBufCommon(*this, t1, "src1")) ||
        failed(verifyTileBufCommon(*this, td, "dst")))
      return failure();

    Type e0 = getElemTy(t0);
    Type e1 = getElemTy(t1);
    Type ed = getElemTy(td);
    if (!e0 || !e1 || !ed) {
      emitOpError("failed to get element type for operands");
      return failure();
    }
    if (e0 != e1 || e0 != ed) {
      emitOpError("expects src0, src1, and dst to have the same element type");
      return failure();
    }

    auto v0 = getValidShapeVec(getSrc0());
    auto v1 = getValidShapeVec(getSrc1());
    auto vd = getValidShapeVec(getDst());
    if (v0.size() != 2 || v1.size() != 2 || vd.size() != 2)
      return emitOpError("expects src0, src1, and dst to have rank-2 valid_shape");

    // validRow must match dst (when known).
    if (v0[0] != ShapedType::kDynamic && vd[0] != ShapedType::kDynamic && v0[0] != vd[0])
      return emitOpError("expects src0 valid row to match dst valid row");
    if (v1[0] != ShapedType::kDynamic && vd[0] != ShapedType::kDynamic && v1[0] != vd[0])
      return emitOpError("expects src1 valid row to match dst valid row");

    // Total valid columns must fit within dst static cols (when known).
    auto sd = getShapeVec(td);
    if (sd.size() == 2 && sd[1] != ShapedType::kDynamic &&
        v0[1] != ShapedType::kDynamic && v1[1] != ShapedType::kDynamic) {
      if (v0[1] + v1[1] > sd[1])
        return emitOpError("expects src0.valid_col + src1.valid_col <= dst.cols");
    }

    return e0;
  };

  auto verifyElemType = [&](Type elem) -> LogicalResult {
    if (elem.isF16() || elem.isF32() || elem.isBF16())
      return success();
    auto it = mlir::dyn_cast<IntegerType>(elem);
    if (!it ||
        (it.getWidth() != 8 && it.getWidth() != 16 && it.getWidth() != 32))
      return emitOpError("expects element type to be i8, i16, i32, f16, f32, or bf16");
    return success();
  };

  auto verifyLocVec = [&](Type ty, StringRef name) -> LogicalResult {
    auto as = getPTOMemorySpaceEnum(ty);
    if (!as || *as != pto::AddressSpace::VEC)
      return emitOpError() << "expects " << name << " to use loc=vec";
    return success();
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr))
      return failure();
    if (failed(verifyLocVec(getSrc0().getType(), "src0")) ||
        failed(verifyLocVec(getSrc1().getType(), "src1")) ||
        failed(verifyLocVec(getDst().getType(), "dst")))
      return failure();
    return verifyElemType(*elemOr);
  };

  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr))
      return failure();
    if (failed(verifyLocVec(getSrc0().getType(), "src0")) ||
        failed(verifyLocVec(getSrc1().getType(), "src1")) ||
        failed(verifyLocVec(getDst().getType(), "dst")))
      return failure();
    if (!isRowMajorTileBuf(getSrc0().getType()) || !isRowMajorTileBuf(getSrc1().getType()) ||
        !isRowMajorTileBuf(getDst().getType()))
      return emitOpError("expects src0, src1, and dst to use row-major layout");
    return verifyElemType(*elemOr);
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult pto::TConcatidxOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<std::pair<Type, Type>> {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type ti0 = getSrc0Idx().getType();
    Type ti1 = getSrc1Idx().getType();
    Type td = getDst().getType();
    if (failed(verifyTileBufCommon(*this, t0, "src0")) ||
        failed(verifyTileBufCommon(*this, t1, "src1")) ||
        failed(verifyTileBufCommon(*this, ti0, "src0Idx")) ||
        failed(verifyTileBufCommon(*this, ti1, "src1Idx")) ||
        failed(verifyTileBufCommon(*this, td, "dst")))
      return failure();

    // Check data element type consistency.
    Type e0 = getElemTy(t0);
    Type e1 = getElemTy(t1);
    Type ed = getElemTy(td);
    if (!e0 || !e1 || !ed) {
      emitOpError("failed to get element type for data operands");
      return failure();
    }
    if (e0 != e1 || e0 != ed) {
      emitOpError("expects src0, src1, and dst to have the same element type");
      return failure();
    }

    // Check index element type consistency.
    Type ei0 = getElemTy(ti0);
    Type ei1 = getElemTy(ti1);
    if (!ei0 || !ei1) {
      emitOpError("failed to get element type for index operands");
      return failure();
    }
    if (ei0 != ei1) {
      emitOpError("expects src0Idx and src1Idx to have the same element type");
      return failure();
    }

    // All five tiles must be rank-2.
    auto v0  = getValidShapeVec(getSrc0());
    auto v1  = getValidShapeVec(getSrc1());
    auto vi0 = getValidShapeVec(getSrc0Idx());
    auto vi1 = getValidShapeVec(getSrc1Idx());
    auto vd  = getValidShapeVec(getDst());
    if (v0.size() != 2 || v1.size() != 2 || vi0.size() != 2 ||
        vi1.size() != 2 || vd.size() != 2)
      return emitOpError("expects all operands to have rank-2 valid_shape");

    // validRow must match dst (when known).
    auto checkValidRow = [&](const auto &v, StringRef name) -> LogicalResult {
      if (v[0] != ShapedType::kDynamic && vd[0] != ShapedType::kDynamic &&
          v[0] != vd[0])
        return emitOpError("expects ") << name << " valid row to match dst valid row";
      return success();
    };
    if (failed(checkValidRow(v0, "src0")) ||
        failed(checkValidRow(v1, "src1")) ||
        failed(checkValidRow(vi0, "src0Idx")) ||
        failed(checkValidRow(vi1, "src1Idx")))
      return failure();

    // Index tile must have cols >= 1 (when known).
    if (vi0[1] != ShapedType::kDynamic && vi0[1] < 1)
      return emitOpError("expects src0Idx valid_col >= 1");
    if (vi1[1] != ShapedType::kDynamic && vi1[1] < 1)
      return emitOpError("expects src1Idx valid_col >= 1");

    return std::make_pair(e0, ei0);
  };

  auto verifyElementTypes = [&](Type dataElem, Type idxElem) -> LogicalResult {
    // Data element type: f16, f32, bf16, i8, i16, i32 (signless).
    if (!dataElem.isF16() && !dataElem.isF32() && !dataElem.isBF16()) {
      auto it = mlir::dyn_cast<IntegerType>(dataElem);
      if (!it || !it.isSignless() ||
          (it.getWidth() != 8 && it.getWidth() != 16 && it.getWidth() != 32))
        return emitOpError()
               << "expects data element type to be i8, i16, i32, f16, f32, or bf16";
    }

    // Index element type: i8, i16, i32 (signless).
    auto it = mlir::dyn_cast<IntegerType>(idxElem);
    if (!it || !it.isSignless() ||
        (it.getWidth() != 8 && it.getWidth() != 16 && it.getWidth() != 32))
      return emitOpError()
             << "expects index element type to be i8, i16, or i32";
    return success();
  };

  auto verifyLocVec = [&](Type ty, StringRef name) -> LogicalResult {
    auto as = getPTOMemorySpaceEnum(ty);
    if (!as || *as != pto::AddressSpace::VEC)
      return emitOpError() << "expects " << name << " to use loc=vec";
    return success();
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    auto elemOr = verifyCommon();
    if (failed(elemOr))
      return failure();
    if (failed(verifyLocVec(getSrc0().getType(), "src0")) ||
        failed(verifyLocVec(getSrc1().getType(), "src1")) ||
        failed(verifyLocVec(getSrc0Idx().getType(), "src0Idx")) ||
        failed(verifyLocVec(getSrc1Idx().getType(), "src1Idx")) ||
        failed(verifyLocVec(getDst().getType(), "dst")))
      return failure();
    return verifyElementTypes(elemOr->first, elemOr->second);
  };

  auto verifyA5 = [&]() -> LogicalResult {
    auto elemOr = verifyCommon();
    if (failed(elemOr))
      return failure();
    if (failed(verifyLocVec(getSrc0().getType(), "src0")) ||
        failed(verifyLocVec(getSrc1().getType(), "src1")) ||
        failed(verifyLocVec(getSrc0Idx().getType(), "src0Idx")) ||
        failed(verifyLocVec(getSrc1Idx().getType(), "src1Idx")) ||
        failed(verifyLocVec(getDst().getType(), "dst")))
      return failure();
    if (!isRowMajorTileBuf(getSrc0().getType()) ||
        !isRowMajorTileBuf(getSrc1().getType()) ||
        !isRowMajorTileBuf(getSrc0Idx().getType()) ||
        !isRowMajorTileBuf(getSrc1Idx().getType()) ||
        !isRowMajorTileBuf(getDst().getType()))
      return emitOpError(
          "expects all operands to use row-major layout");
    return verifyElementTypes(elemOr->first, elemOr->second);
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult pto::TAndSOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<Type> {
    return verifyDistinctRowMajorUnaryTileOpCommon(getOperation(), getSrc(),
                                                   getDst(), "src", "dst");
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16))
      return emitOpError(
          "expects A2/A3 tands src, scalar, and dst element type to be i8/i16");
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
                it.getWidth() != 32))
      return emitOpError(
          "expects A5 tands src, scalar, and dst element type to be i8/i16/i32");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static ParseResult parseTCILikeOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand s, tmp, dst;
  Type sTy, tmpTy, dstTy;

  if (parser.parseKeyword("ins") || parser.parseLParen() || parser.parseOperand(s))
    return failure();

  bool hasTmp = succeeded(parser.parseOptionalComma());
  if (hasTmp && parser.parseOperand(tmp))
    return failure();

  if (parser.parseColonType(sTy))
    return failure();
  if (hasTmp) {
    if (parser.parseComma() || parser.parseType(tmpTy))
      return failure();
  }
  if (parser.parseRParen() || parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) || parser.parseRParen() ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (parser.resolveOperand(s, sTy, result.operands))
    return failure();
  if (hasTmp && parser.resolveOperand(tmp, tmpTy, result.operands))
    return failure();
  if (parser.resolveOperand(dst, dstTy, result.operands))
    return failure();

  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr({1, hasTmp ? 1 : 0, 1}));
  return success();
}

static void printTCILikeOp(OpAsmPrinter &p, Operation *op, Value s, Value tmp,
                           Value dst) {
  p << " ins(" << s;
  if (tmp)
    p << ", " << tmp;
  p << " : " << s.getType();
  if (tmp)
    p << ", " << tmp.getType();
  p << ") outs(" << dst << " : " << dst.getType() << ")";
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"operandSegmentSizes"});
}

ParseResult mlir::pto::TCIOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTCILikeOp(parser, result);
}

void mlir::pto::TCIOp::print(OpAsmPrinter &p) {
  printTCILikeOp(p, getOperation(), getOperand(0), getTmp(), getDst());
}

LogicalResult pto::TCIOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type dstTy = getDst().getType();
  if (failed(verifyTileBufCommon(*this, dstTy, "dst")))
    return failure();
  if (getTmp() && failed(verifyTileBufCommon(*this, getTmp().getType(), "tmp")))
    return failure();

  auto elemTy = mlir::dyn_cast<IntegerType>(getElemTy(dstTy));
  if (!elemTy)
    return emitOpError("expects dst element type to be integer");

  unsigned bw = elemTy.getWidth();
  if (bw != 16 && bw != 32)
    return emitOpError("expects dst element type to be i16/i32");

  auto sTy = mlir::dyn_cast<IntegerType>(getOperand(0).getType());
  if (!sTy)
    return emitOpError("expects S to be integer");

  if (sTy != elemTy)
    return emitOpError("expects S and dst element type to be exactly the same type");
  auto shape = getShapeVec(dstTy);
  if (shape.size() != 2)
    return emitOpError("expects dst to be rank-2");
  if (shape[1] != ShapedType::kDynamic && shape[1] == 1)
    return emitOpError("expects dst cols to be different from 1");

  return success();
}

LogicalResult pto::TTriOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();

  Type dstTy = getDst().getType();
  if (failed(verifyVecTileCommon(*this, dstTy, "dst")))
    return failure();

  auto diagonalTy = mlir::dyn_cast<IntegerType>(getDiagonal().getType());
  if (!diagonalTy)
    return emitOpError("expects diagonal to be an integer operand");

  int32_t upperOrLower = getUpperOrLower();
  if (upperOrLower != 0 && upperOrLower != 1)
    return emitOpError("expects upperOrLower to be 0 (lower) or 1 (upper)");

  Type elemTy = getElemTy(dstTy);
  return dispatchVerifierByArch(
      getOperation(),
      [&]() -> LogicalResult {
        if (!isSupportedVecElemType(elemTy, /*allowBf16=*/false,
                                    /*allowInt8=*/false))
          return emitOpError()
                 << "expects A2/A3 dst element type to be f16/f32/i16/i32/u16/u32";
        return success();
      },
      [&]() -> LogicalResult {
        if (!isSupportedVecElemType(elemTy, /*allowBf16=*/true,
                                    /*allowInt8=*/true))
          return emitOpError()
                 << "expects A5 dst element type to be f16/f32/bf16/i8/i16/i32/u8/u16/u32";
        return success();
      });
}

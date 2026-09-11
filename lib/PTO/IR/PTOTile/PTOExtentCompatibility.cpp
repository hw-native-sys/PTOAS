// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.


static LogicalResult verifyMGatherMScatterCoalesce(
    Operation *op, StringRef dataName, std::optional<pto::Coalesce> coalesce,
    bool baseRow, bool row1xR, bool rowRx1, bool elem) {
  if (!coalesce) {
    if (baseRow || elem)
      return success();
    return op->emitOpError()
           << "expects idx valid_shape to be [" << dataName
           << ".valid_row, 0|1] or match " << dataName
           << " valid_shape when coalesce is omitted";
  }
  if (*coalesce == pto::Coalesce::Row && (row1xR || rowRx1))
    return success();
  if (*coalesce == pto::Coalesce::Elem && elem)
    return success();
  if (*coalesce == pto::Coalesce::Row)
    return op->emitOpError()
           << "expects row-coalesce idx valid_shape to be [0|1, " << dataName
           << ".valid_row] or [" << dataName << ".valid_row, 0|1]";
  return op->emitOpError()
         << "expects elem-coalesce idx valid_shape to match " << dataName
         << " valid_shape";
}

static LogicalResult verifyMGatherMScatterTileShape(Operation *op, Type dataTy,
                                                    Type idxTy,
                                                    StringRef dataName,
                                                    std::optional<pto::Coalesce> coalesce) {
  auto dataValid = getValidShapeVec(dataTy);
  auto idxValid = getValidShapeVec(idxTy);
  if (dataValid.size() != 2 || idxValid.size() != 2) {
    return op->emitOpError() << "expects " << dataName
                             << " and idx to have rank-2 valid_shape";
  }

  auto idxTile = dyn_cast<pto::TileBufType>(idxTy);
  if (!idxTile) {
    return op->emitOpError("expects idx to be a tile_buf type");
  }

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

  return verifyMGatherMScatterCoalesce(
      op, dataName, coalesce, baseRowCoalesce, rowCoalesce1xR,
      rowCoalesceRx1, elemCoalesce);
}

template <typename AttrT>
static AttrT getPTOOpAttr(Operation *op, StringRef name) {
  if (Attribute propsAttr = op->getPropertiesAsAttribute()) {
    if (auto props = dyn_cast<DictionaryAttr>(propsAttr)) {
      if (auto attr = dyn_cast_or_null<AttrT>(props.get(name))) {
        return attr;
      }
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
  if (parser.parseOptionalAttrDict(parsedAttrs)) {
    return failure();
  }

  auto &properties = result.getOrAddProperties<typename OpTy::Properties>();
  OpTy::populateDefaultProperties(result.name, properties);
  if (failed(OpTy::setPropertiesFromAttr(
          properties, parsedAttrs.getDictionary(parser.getContext()), [&] {
            return parser.emitError(attrLoc)
                   << "'" << result.name.getStringRef() << "' op ";
          }))) {
    return failure();
  }

  for (StringRef attrName : inherentAttrNames) {
    parsedAttrs.erase(attrName);
  }
  result.attributes = parsedAttrs;
  return success();
}

static NamedAttrList getNonInherentAttrs(Operation *op,
                                         ArrayRef<StringRef> inherentAttrNames) {
  NamedAttrList attrs;
  for (NamedAttribute attr : op->getRawDictionaryAttrs()) {
    if (llvm::is_contained(inherentAttrNames, attr.getName().getValue())) {
      continue;
    }
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
  if (auto attr = getMGatherGatherOobAttrIfPresent(op)) {
    return attr.getValue();
  }
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
  if (auto attr = getMGatherCoalesceAttrIfPresent(op)) {
    return attr.getValue();
  }
  return std::nullopt;
}

static std::optional<pto::Coalesce> getCoalesceIfPresent(pto::MScatterOp op) {
  if (auto attr = getMScatterCoalesceAttrIfPresent(op)) {
    return attr.getValue();
  }
  return std::nullopt;
}

static pto::ScatterAtomicOp getScatterAtomicOpOrDefault(pto::MScatterOp op) {
  if (auto attr = getMScatterScatterAtomicOpAttrIfPresent(op)) {
    return attr.getValue();
  }
  return pto::ScatterAtomicOp::None;
}

static pto::ScatterOOB getScatterOobOrDefault(pto::MScatterOp op) {
  if (auto attr = getMScatterScatterOobAttrIfPresent(op)) {
    return attr.getValue();
  }
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
  if (failed(verifyTileBufInVec(op, ty, name)))
    return failure();
  auto tb = dyn_cast<pto::TileBufType>(ty);
  if (!tb) {
    return op->emitOpError() << "expects " << name << " to be a tile_buf type";
  }
  int32_t blayout = tb.getBLayoutValueI32();
  if (blayout != static_cast<int32_t>(pto::BLayout::RowMajor) &&
      blayout != static_cast<int32_t>(pto::BLayout::ColMajor)) {
    return op->emitOpError() << "expects " << name
                             << " to use row_major or col_major blayout";
  }
  if (tb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox)) {
    return op->emitOpError() << "expects " << name
                             << " to use the none_box slayout";
  }
  return success();
}

static bool isA5TLoadStoreTransferElemType(Type ty) {
  return ty.isInteger(8) || ty.isInteger(16) || ty.isInteger(32) ||
         ty.isInteger(64) || ty.isF16() || ty.isBF16() || ty.isF32() ||
         isPTOLowPrecisionType(ty);
}

static bool isA5AccStorePreQuantDstType(Type srcElem, Type dstElem) {
  if (srcElem.isInteger(32)) {
    return dstElem.isInteger(8) || dstElem.isF16() || dstElem.isBF16();
  }
  if (!srcElem.isF32()) {
    return false;
  }
  return dstElem.isInteger(8) || dstElem.isF16() || dstElem.isBF16() ||
         dstElem.isF32() || isPTOHiFloat8Type(dstElem) ||
         isPTOFloat8E4M3LikeType(dstElem);
}

static bool isA5LowPrecisionTCvtPair(Type srcElem, Type dstElem) {
  if (srcElem.isF32()) {
    return isPTOFloat8Type(dstElem) || isPTOHiFloat8Type(dstElem);
  }
  if (srcElem.isF16()) {
    return isPTOHiFloat8Type(dstElem);
  }
  if (srcElem.isBF16()) {
    return isPTOFloat4PackedType(dstElem);
  }
  if (isPTOFloat4PackedType(srcElem)) {
    return dstElem.isBF16();
  }
  if (isPTOFloat8Type(srcElem) || isPTOHiFloat8Type(srcElem)) {
    return dstElem.isF32();
  }
  return false;
}

static bool isA5SupportedTCvtPair(Type srcElem, Type dstElem) {
  if (isPTOLowPrecisionType(srcElem) || isPTOLowPrecisionType(dstElem)) {
    return isA5LowPrecisionTCvtPair(srcElem, dstElem);
  }
  return true;
}

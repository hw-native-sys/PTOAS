// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This implementation fragment is included by PTO.cpp and intentionally is
// not listed as a separate CMake translation unit.

static ParseResult parseSyncEventOpCommon(OpAsmParser &parser,
                                          OperationState &result,
                                          StringAttr pipeAttrName,
                                          StringAttr eventIdAttrName) {
  PipeAttr pipeAttr;
  if (succeeded(parser.parseOptionalLess())) {
    StringRef pipeTok;
    if (parser.parseKeyword(&pipeTok) || parser.parseGreater())
      return failure();
    auto pipeOr = symbolizePIPE(pipeTok);
    if (!pipeOr)
      return parser.emitError(parser.getCurrentLocation())
             << "unknown pipe token: " << pipeTok;
    pipeAttr = PipeAttr::get(parser.getContext(), *pipeOr);
    result.addAttribute(pipeAttrName, pipeAttr);
  } else if (parser.parseAttribute(pipeAttr, pipeAttrName,
                                   result.attributes)) {
    return failure();
  }
  if (parser.parseComma())
    return failure();

  OpAsmParser::UnresolvedOperand eventOperand;
  OptionalParseResult parseEventOperand =
      parser.parseOptionalOperand(eventOperand);
  if (parseEventOperand.has_value()) {
    if (failed(*parseEventOperand))
      return failure();
    if (parser.resolveOperand(eventOperand, parser.getBuilder().getIndexType(),
                              result.operands))
      return failure();
  } else {
    IntegerAttr eventAttr;
    if (parser.parseAttribute(eventAttr, parser.getBuilder().getI32Type(),
                              eventIdAttrName, result.attributes))
      return failure();
  }

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

static void printSyncEventOpCommon(OpAsmPrinter &p, Operation *op,
                                   PipeAttr pipeAttr, IntegerAttr eventAttr,
                                   Value eventDyn, StringRef pipeAttrName,
                                   StringRef eventIdAttrName) {
  p << " <" << stringifyPIPE(pipeAttr.getPipe()) << ">, ";
  if (eventAttr)
    p << eventAttr.getInt();
  else
    p << eventDyn;
  p.printOptionalAttrDict(op->getAttrs(), {pipeAttrName, eventIdAttrName});
}

[[maybe_unused]] static mlir::Type parsePTOTypeAllowNoBang(mlir::OpAsmParser &parser) {
  mlir::Type ty;

  mlir::OptionalParseResult opt = parser.parseOptionalType(ty);

  if (opt.has_value()) {
    if (failed(*opt))
      return mlir::Type();
    return ty;
  }


  llvm::StringRef head;
  if (failed(parser.parseKeyword(&head)))
    return mlir::Type();

  mlir::MLIRContext *ctx = parser.getContext();

  auto parseShapeElemForOpParser =
      [&](llvm::SmallVectorImpl<int64_t> &shape, mlir::Type &elem) -> mlir::LogicalResult {
        if (failed(parser.parseLess()))
          return failure();
        if (failed(parser.parseDimensionList(shape, /*allowDynamic=*/true)))
          return failure();
        if (failed(parser.parseType(elem)))
          return failure();
        if (failed(parser.parseGreater()))
          return failure();
        return success();
      };

  if (head == "pto.tile_view") {
    llvm::SmallVector<int64_t, 4> shape;
    mlir::Type elem;
    if (failed(parseShapeElemForOpParser(shape, elem)))
      return mlir::Type();
    return mlir::pto::PartitionTensorViewType::get(ctx, shape, elem);
  }

  if (head == "pto.tile") {
    llvm::SmallVector<int64_t, 4> shape;
    mlir::Type elem;
    if (failed(parseShapeElemForOpParser(shape, elem)))
      return mlir::Type();
    return mlir::pto::TileType::get(ctx, shape, elem);
  }

  if (head == "pto.ptr") {
    if (failed(parser.parseLess()))
      return mlir::Type();
    mlir::Type elem;
    if (failed(parser.parseType(elem)))
      return mlir::Type();
    auto memorySpace = pto::AddressSpaceAttr::get(ctx, pto::AddressSpace::GM);
    if (succeeded(parser.parseOptionalComma())) {
      StringRef memorySpaceKeyword;
      if (failed(parser.parseKeyword(&memorySpaceKeyword)))
        return mlir::Type();
      auto parsed = parsePtrAddressSpaceKeyword(memorySpaceKeyword);
      if (!parsed) {
        parser.emitError(parser.getCurrentLocation(),
                         "!pto.ptr address space must be one of "
                         "`gm|ub|mat|l1|left|l0a|right|l0b|acc|l0c|vec|bias|bt|scaling|fb`");
        return mlir::Type();
      }
      memorySpace = pto::AddressSpaceAttr::get(ctx, *parsed);
    }
    if (failed(parser.parseGreater()))
      return mlir::Type();
    return mlir::pto::PtrType::get(ctx, elem, memorySpace);
  }

  if (head == "pto.tensor_view") {
    llvm::SmallVector<int64_t, 4> shape;
    mlir::Type elem;
    if (failed(parseShapeElemForOpParser(shape, elem)))
      return mlir::Type();
    return mlir::pto::TensorViewType::get(ctx, shape, elem);
  }

  return mlir::Type();
}

mlir::Type TensorViewType::parse(::mlir::AsmParser &parser) {
  SmallVector<int64_t, 4> shape;
  Type elementType;
  if (failed(parseShapeAndElem(parser, shape, elementType, /*allowDynamic=*/true)))
    return Type();
  return TensorViewType::get(parser.getContext(), shape, elementType);
}

void TensorViewType::print(::mlir::AsmPrinter &printer) const {
  printShapeAndElem(printer, getShape(), getElementType());
}

mlir::Type PtrType::parse(::mlir::AsmParser &parser) {
  Type elementType;
  if (failed(parser.parseLess()) || failed(parser.parseType(elementType)))
    return {};

  auto memorySpace =
      pto::AddressSpaceAttr::get(parser.getContext(), pto::AddressSpace::GM);
  if (succeeded(parser.parseOptionalComma())) {
    StringRef memorySpaceKeyword;
    if (failed(parser.parseKeyword(&memorySpaceKeyword)))
      return {};
    auto parsed = parsePtrAddressSpaceKeyword(memorySpaceKeyword);
    if (!parsed) {
      parser.emitError(parser.getCurrentLocation(),
                       "!pto.ptr address space must be one of "
                       "`gm|ub|mat|l1|left|l0a|right|l0b|acc|l0c|vec|bias|bt|scaling|fb`");
      return {};
    }
    memorySpace = pto::AddressSpaceAttr::get(parser.getContext(), *parsed);
  }

  if (failed(parser.parseGreater()))
    return {};
  return PtrType::get(parser.getContext(), elementType, memorySpace);
}

void PtrType::print(::mlir::AsmPrinter &printer) const {
  printer << "<" << getElementType();
  StringRef memorySpaceKeyword =
      printPtrAddressSpaceKeyword(getMemorySpace().getAddressSpace());
  if (!memorySpaceKeyword.empty())
    printer << ", " << memorySpaceKeyword;
  printer << ">";
}

//===----------------------------------------------------------------------===//
// pto.tdivs custom asm to support both:
//   pto.tdivs ins(%src, %scalar : !pto.tile_buf<...>, f32) outs(%dst : !pto.tile_buf<...>)
//   pto.tdivs ins(%scalar, %src : f32, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
// The operand order in the op follows textual input order.
//===----------------------------------------------------------------------===//

ParseResult mlir::pto::TDivSOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand op0, op1, dst;
  Type ty0, ty1, dstTy;

  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(op0) || parser.parseComma() ||
      parser.parseOperand(op1) || parser.parseColonType(ty0) ||
      parser.parseComma() || parser.parseType(ty1) || parser.parseRParen())
    return failure();

  if (parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) ||
      parser.parseRParen())
    return failure();

  NamedAttrList attrs;
  if (parser.parseOptionalAttrDict(attrs))
    return failure();

  auto tile0 = dyn_cast<mlir::pto::TileBufType>(ty0);
  auto tile1 = dyn_cast<mlir::pto::TileBufType>(ty1);
  if ((tile0 && tile1) || (!tile0 && !tile1))
    return parser.emitError(parser.getCurrentLocation(),
                            "expected exactly one tile_buf operand and one scalar operand");

  if (!dyn_cast<mlir::pto::TileBufType>(dstTy))
    return parser.emitError(parser.getCurrentLocation(),
                            "expected outs type to be !pto.tile_buf<...>");

  // Keep textual order so later lowering can distinguish the two APIs by the
  // first ins operand type.
  if (parser.resolveOperand(op0, ty0, result.operands) ||
      parser.resolveOperand(op1, ty1, result.operands))
    return failure();

  if (parser.resolveOperand(dst, dstTy, result.operands))
    return failure();

  result.addAttributes(attrs);
  return success();
}

void mlir::pto::TDivSOp::print(OpAsmPrinter &p) {
  p << " ins(";
  p << getSrc() << ", " << getScalar() << " : "
    << getSrc().getType() << ", " << getScalar().getType();
  p << ") outs(" << getDst() << " : " << getDst().getType() << ")";

  p.printOptionalAttrDict((*this)->getAttrs());
}


//===----------------------------------------------------------------------===//
// pto.tgather custom asm supports three PTO-ISA forms:
//   1) index+tmp   : ins(%src, %indices, %tmp : srcTy, indicesTy, tmpTy) outs(%dst : dstTy)
//   2) compare+tmp : ins(%src, %kValue, %tmp : srcTy, scalarTy, tmpTy)
//                    outs(%dst, %cdst : dstTy, cdstTy) {cmpMode = #pto.cmp<gt>, offset = 7}
//   3) mask        : ins(%src, {maskPattern = #pto.mask_pattern<P0101>} : srcTy) outs(%dst : dstTy)
//===----------------------------------------------------------------------===//

ParseResult mlir::pto::TGatherOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand src, dst, cdst;
  SmallVector<OpAsmParser::UnresolvedOperand, 3> insOps;
  SmallVector<Type, 3> insTypes;
  Type srcTy, dstTy, cdstTy;
  bool hasCdst = false;
  bool hasMask = false;
  bool hasIndices = false;
  bool hasTmp = false;
  bool hasKValue = false;

  if (parser.parseKeyword("ins") || parser.parseLParen() || parser.parseOperand(src))
    return failure();

  if (!succeeded(parser.parseOptionalComma())) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected ',' after src operand in ins(...)");
  }

  if (succeeded(parser.parseOptionalLBrace())) {
    if (parser.parseKeyword("maskPattern") || parser.parseEqual())
      return failure();

    Attribute rawMaskAttr;
    if (parser.parseAttribute(rawMaskAttr) || parser.parseRBrace())
      return failure();

    auto mp = llvm::dyn_cast<mlir::pto::MaskPatternAttr>(rawMaskAttr);
    if (!mp) {
      return parser.emitError(parser.getCurrentLocation(),
                              "expected #pto.mask_pattern<Pxxxx> for maskPattern");
    }

    result.addAttribute("maskPattern", mp);
    hasMask = true;

    if (parser.parseColonType(srcTy) || parser.parseRParen())
      return failure();
  } else {
    OpAsmParser::UnresolvedOperand extra;
    if (parser.parseOperand(extra))
      return failure();
    insOps.push_back(extra);
    while (succeeded(parser.parseOptionalComma())) {
      if (insOps.size() == 3) {
        return parser.emitError(parser.getCurrentLocation(),
                                "expected at most 3 extra operands in tgather ins(...)");
      }
      if (parser.parseOperand(extra))
        return failure();
      insOps.push_back(extra);
    }

    if (parser.parseColon() || parser.parseType(srcTy))
      return failure();
    for (size_t i = 0; i < insOps.size(); ++i) {
      Type ty;
      if (parser.parseComma() || parser.parseType(ty))
        return failure();
      insTypes.push_back(ty);
    }
    if (parser.parseRParen())
      return failure();
  }

  if (parser.parseKeyword("outs") || parser.parseLParen() || parser.parseOperand(dst))
    return failure();
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(cdst))
      return failure();
    hasCdst = true;
  }
  if (parser.parseColonType(dstTy))
    return failure();
  if (hasCdst && (parser.parseComma() || parser.parseType(cdstTy)))
    return failure();
  if (parser.parseRParen())
    return failure();

  if (succeeded(parser.parseOptionalKeyword("maskPattern"))) {
    if (hasMask)
      return parser.emitError(parser.getCurrentLocation(),
                              "maskPattern may only be specified once");
    if (parser.parseEqual())
      return failure();
    Attribute rawMaskAttr;
    if (parser.parseAttribute(rawMaskAttr))
      return failure();
    auto mp = llvm::dyn_cast<mlir::pto::MaskPatternAttr>(rawMaskAttr);
    if (!mp) {
      return parser.emitError(parser.getCurrentLocation(),
                              "expected #pto.mask_pattern<Pxxxx> for maskPattern");
    }
    result.addAttribute("maskPattern", mp);
    hasMask = true;
  }

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (hasMask) {
    if (!insOps.empty())
      return parser.emitError(parser.getCurrentLocation(),
                              "mask-pattern tgather does not take extra ins operands");
    if (hasCdst)
      return parser.emitError(parser.getCurrentLocation(),
                              "mask-pattern tgather expects a single outs operand");
  } else if (hasCdst) {
    if (insOps.empty() ||
        !(mlir::isa<IntegerType>(insTypes.front()) ||
          mlir::isa<FloatType>(insTypes.front())))
      return parser.emitError(parser.getCurrentLocation(),
                              "compare-form tgather expects a scalar kValue operand");
    hasKValue = true;
    if (insOps.size() >= 2) {
      if (!isTileLikeType(insTypes[1]))
        return parser.emitError(parser.getCurrentLocation(),
                                "compare-form tgather tmp must be tile-like");
      hasTmp = true;
    }
    if (insOps.size() == 3) {
      return parser.emitError(parser.getCurrentLocation(),
                              "compare-form tgather expects at most src, kValue, tmp in ins(...)");
    }
  } else {
    if (!insOps.empty() && !isTileLikeType(insTypes.front())) {
      return parser.emitError(parser.getCurrentLocation(),
                              "index-form tgather expects tile-like indices; "
                              "compare-form must use outs(dst, cdst)");
    }
    if (!insOps.empty()) {
      hasIndices = true;
      if (insOps.size() >= 2) {
        if (!isTileLikeType(insTypes[1]))
          return parser.emitError(parser.getCurrentLocation(),
                                  "index-form tgather tmp must be tile-like");
        hasTmp = true;
      }
    }
    if (insOps.size() == 3) {
      return parser.emitError(parser.getCurrentLocation(),
                              "index-form tgather expects at most src, indices, tmp in ins(...)");
    }
  }

  if (parser.resolveOperand(src, srcTy, result.operands) ||
      parser.resolveOperand(dst, dstTy, result.operands))
    return failure();
  if (hasCdst && parser.resolveOperand(cdst, cdstTy, result.operands))
    return failure();
  if (hasIndices && parser.resolveOperand(insOps[0], insTypes[0], result.operands))
    return failure();
  if (hasTmp && parser.resolveOperand(insOps[hasIndices ? 1 : 1], insTypes[1], result.operands))
    return failure();
  if (hasKValue && parser.resolveOperand(insOps[0], insTypes[0], result.operands))
    return failure();

  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {1, 1, hasCdst ? 1 : 0, hasIndices ? 1 : 0,
                           hasTmp ? 1 : 0, hasKValue ? 1 : 0}));
  return success();
}

void mlir::pto::TGatherOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc() << ", ";
  if (auto mp = getMaskPatternAttr()) {
    p << "{maskPattern = " << mp << "} : " << getSrc().getType();
  } else if (getCdst()) {
    p << getKValue();
    if (getTmp()) {
      p << ", " << getTmp();
      p << " : " << getSrc().getType() << ", " << getKValue().getType()
        << ", " << getTmp().getType();
    } else {
      p << " : " << getSrc().getType() << ", " << getKValue().getType();
    }
  } else {
    p << getIndices();
    if (getTmp()) {
      p << ", " << getTmp();
      p << " : " << getSrc().getType() << ", " << getIndices().getType()
        << ", " << getTmp().getType();
    } else {
      p << " : " << getSrc().getType() << ", " << getIndices().getType();
    }
  }
  p << ") outs(" << getDst();
  if (getCdst())
    p << ", " << getCdst();
  p << " : " << getDst().getType();
  if (getCdst())
    p << ", " << getCdst().getType();
  p << ")";

  if (getMaskPatternAttr()) {
    p.printOptionalAttrDict((*this)->getAttrs(),
                            /*elidedAttrs=*/{"maskPattern", "operandSegmentSizes"});
  } else {
    p.printOptionalAttrDict((*this)->getAttrs(),
                            /*elidedAttrs=*/{"operandSegmentSizes"});
  }
}

ParseResult mlir::pto::TScatterOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  OpAsmParser::UnresolvedOperand src, indexes, dst;
  Type srcTy, idxTy, dstTy;
  bool hasMask = false;
  bool hasIndexes = false;

  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(src))
    return failure();

  if (!succeeded(parser.parseOptionalComma()))
    return parser.emitError(parser.getCurrentLocation(),
                            "expected ',' after src operand in ins(...)");

  if (succeeded(parser.parseOptionalLBrace())) {
    if (parser.parseKeyword("maskPattern") || parser.parseEqual())
      return failure();
    Attribute rawMaskAttr;
    if (parser.parseAttribute(rawMaskAttr) || parser.parseRBrace())
      return failure();
    auto mp = llvm::dyn_cast<mlir::pto::MaskPatternAttr>(rawMaskAttr);
    if (!mp)
      return parser.emitError(parser.getCurrentLocation(),
                              "expected #pto.mask_pattern<Pxxxx> for maskPattern");
    result.addAttribute("maskPattern", mp);
    hasMask = true;
    if (parser.parseColonType(srcTy) || parser.parseRParen())
      return failure();
  } else {
    if (parser.parseOperand(indexes))
      return failure();
    hasIndexes = true;
    if (parser.parseColon() || parser.parseType(srcTy) || parser.parseComma() ||
        parser.parseType(idxTy) || parser.parseRParen())
      return failure();
  }

  if (parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) ||
      parser.parseRParen())
    return failure();

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (result.attributes.get("maskPattern"))
    hasMask = true;

  if (hasMask && hasIndexes)
    return parser.emitError(parser.getCurrentLocation(),
                            "mask-pattern tscatter does not take indexes");
  if (!hasMask && !hasIndexes)
    return parser.emitError(parser.getCurrentLocation(),
                            "expected indexes operand or maskPattern for tscatter");

  if (parser.resolveOperand(src, srcTy, result.operands) ||
      parser.resolveOperand(dst, dstTy, result.operands) ||
      (hasIndexes && parser.resolveOperand(indexes, idxTy, result.operands)))
    return failure();
  return success();
}

void mlir::pto::TScatterOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc() << ", ";
  if (getMaskPatternAttr()) {
    p << "{maskPattern = " << getMaskPatternAttr() << "} : "
      << getSrc().getType();
  } else {
    p << getIndexes() << " : " << getSrc().getType() << ", "
      << getIndexes().getType();
  }
  p << ") outs(" << getDst() << " : " << getDst().getType() << ")";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"maskPattern"});
}

namespace {

struct CommRecvClause {
  OpAsmParser::UnresolvedOperand ping;
  std::optional<OpAsmParser::UnresolvedOperand> pong;
  Type pingTy;
  Type pongTy;
};

static ParseResult parseCommRecvClause(OpAsmParser &parser,
                                       CommRecvClause &recvClause) {
  if (parser.parseKeyword("recv") || parser.parseLParen() ||
      parser.parseOperand(recvClause.ping))
    return failure();
  if (succeeded(parser.parseOptionalComma())) {
    OpAsmParser::UnresolvedOperand pong;
    if (parser.parseOperand(pong))
      return failure();
    recvClause.pong = pong;
  }
  return parser.parseRParen();
}

static ParseResult parseCommCollectiveTail(
    OpAsmParser &parser, OperationState &result,
    ArrayRef<OpAsmParser::UnresolvedOperand> fixedOperands,
    SmallVectorImpl<Type> &fixedTypes, CommRecvClause &recvClause,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &groupOps,
    SmallVectorImpl<Type> &groupTypes, ArrayRef<int32_t> operandSegmentsPrefix,
    ArrayRef<StringRef> requiredAttrs) {
  if (parser.parseComma() || parser.parseKeyword("group") || parser.parseLParen())
    return failure();

  OpAsmParser::UnresolvedOperand group;
  if (parser.parseOperand(group))
    return failure();
  groupOps.push_back(group);
  while (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(group))
      return failure();
    groupOps.push_back(group);
  }

  if (parser.parseRParen())
    return failure();

  if (parser.parseColon())
    return failure();

  for (size_t i = 0; i < fixedTypes.size(); ++i) {
    if (i != 0 && parser.parseComma())
      return failure();
    if (parser.parseType(fixedTypes[i]))
      return failure();
  }
  if (parser.parseComma() || parser.parseType(recvClause.pingTy))
    return failure();
  if (recvClause.pong) {
    if (parser.parseComma() || parser.parseType(recvClause.pongTy))
      return failure();
  }
  for (size_t i = 0; i < groupOps.size(); ++i) {
    Type groupTy;
    if (parser.parseComma() || parser.parseType(groupTy))
      return failure();
    groupTypes.push_back(groupTy);
  }
  if (parser.parseRParen())
    return failure();

  NamedAttrList attrs;
  if (parser.parseOptionalAttrDict(attrs))
    return failure();
  for (StringRef attrName : requiredAttrs) {
    if (!attrs.get(attrName)) {
      return parser.emitError(parser.getCurrentLocation())
             << "expected '" << attrName << "' attribute";
    }
  }
  result.addAttributes(attrs);

  for (auto [operand, type] : llvm::zip_equal(fixedOperands, fixedTypes)) {
    if (parser.resolveOperand(operand, type, result.operands))
      return failure();
  }
  if (parser.resolveOperand(recvClause.ping, recvClause.pingTy, result.operands))
    return failure();
  if (recvClause.pong &&
      parser.resolveOperand(*recvClause.pong, recvClause.pongTy, result.operands))
    return failure();
  if (parser.resolveOperands(groupOps, groupTypes, parser.getCurrentLocation(),
                             result.operands))
    return failure();

  SmallVector<int32_t, 5> segmentSizes(operandSegmentsPrefix.begin(),
                                       operandSegmentsPrefix.end());
  segmentSizes.push_back(static_cast<int32_t>(groupOps.size()));
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(segmentSizes));
  return success();
}

static void printCommRecvClause(OpAsmPrinter &p, Value ping, Value pong) {
  p << "recv(" << ping;
  if (pong)
    p << ", " << pong;
  p << ")";
}

static void printCommGroupTypes(OpAsmPrinter &p, ValueRange group) {
  for (Value groupValue : group)
    p << ", " << groupValue.getType();
}

static void printCommGroupClause(OpAsmPrinter &p, ValueRange group) {
  p << "group(";
  p.printOperands(group);
  p << ")";
}

} // namespace

ParseResult mlir::pto::TBroadcastOp::parse(OpAsmParser &parser,
                                           OperationState &result) {
  OpAsmParser::UnresolvedOperand src;
  CommRecvClause recvClause;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> groupOps;
  SmallVector<Type, 4> groupTypes;

  if (parser.parseLParen() || parser.parseOperand(src) || parser.parseComma())
    return failure();
  if (failed(parseCommRecvClause(parser, recvClause)))
    return failure();

  SmallVector<OpAsmParser::UnresolvedOperand, 1> fixedOperands{src};
  SmallVector<Type, 1> fixedTypes(1);
  if (failed(parseCommCollectiveTail(parser, result, fixedOperands, fixedTypes,
                                     recvClause, groupOps, groupTypes,
                                     {1, 1, recvClause.pong ? 1 : 0}, {"root"})))
    return failure();
  return success();
}

void mlir::pto::TBroadcastOp::print(OpAsmPrinter &p) {
  p << "(" << getSrc() << ", ";
  printCommRecvClause(p, getPing(), getPong());
  p << ", ";
  printCommGroupClause(p, getGroup());
  p << " : " << getSrc().getType() << ", " << getPing().getType();
  if (getPong())
    p << ", " << getPong().getType();
  printCommGroupTypes(p, getGroup());
  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes"});
}

ParseResult mlir::pto::CommTGatherOp::parse(OpAsmParser &parser,
                                            OperationState &result) {
  OpAsmParser::UnresolvedOperand dst;
  CommRecvClause recvClause;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> groupOps;
  SmallVector<Type, 4> groupTypes;

  if (parser.parseLParen() || parser.parseOperand(dst) || parser.parseComma())
    return failure();
  if (failed(parseCommRecvClause(parser, recvClause)))
    return failure();

  SmallVector<OpAsmParser::UnresolvedOperand, 1> fixedOperands{dst};
  SmallVector<Type, 1> fixedTypes(1);
  if (failed(parseCommCollectiveTail(
          parser, result, fixedOperands, fixedTypes, recvClause, groupOps,
          groupTypes, {1, 1, recvClause.pong ? 1 : 0},
          {"root"})))
    return failure();
  return success();
}

void mlir::pto::CommTGatherOp::print(OpAsmPrinter &p) {
  p << "(" << getDst() << ", ";
  printCommRecvClause(p, getPing(), getPong());
  p << ", ";
  printCommGroupClause(p, getGroup());
  p << " : " << getDst().getType() << ", " << getPing().getType();
  if (getPong())
    p << ", " << getPong().getType();
  printCommGroupTypes(p, getGroup());
  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes"});
}

ParseResult mlir::pto::CommTScatterOp::parse(OpAsmParser &parser,
                                             OperationState &result) {
  OpAsmParser::UnresolvedOperand src;
  CommRecvClause recvClause;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> groupOps;
  SmallVector<Type, 4> groupTypes;

  if (parser.parseLParen() || parser.parseOperand(src) || parser.parseComma())
    return failure();
  if (failed(parseCommRecvClause(parser, recvClause)))
    return failure();

  SmallVector<OpAsmParser::UnresolvedOperand, 1> fixedOperands{src};
  SmallVector<Type, 1> fixedTypes(1);
  if (failed(parseCommCollectiveTail(
          parser, result, fixedOperands, fixedTypes, recvClause, groupOps,
          groupTypes, {1, 1, recvClause.pong ? 1 : 0},
          {"root"})))
    return failure();
  return success();
}

void mlir::pto::CommTScatterOp::print(OpAsmPrinter &p) {
  p << "(" << getSrc() << ", ";
  printCommRecvClause(p, getPing(), getPong());
  p << ", ";
  printCommGroupClause(p, getGroup());
  p << " : " << getSrc().getType() << ", " << getPing().getType();
  if (getPong())
    p << ", " << getPong().getType();
  printCommGroupTypes(p, getGroup());
  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes"});
}

ParseResult mlir::pto::TReduceOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  OpAsmParser::UnresolvedOperand dst, acc;
  CommRecvClause recvClause;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> groupOps;
  SmallVector<Type, 4> groupTypes;

  if (parser.parseLParen() || parser.parseOperand(dst) || parser.parseComma() ||
      parser.parseOperand(acc) || parser.parseComma())
    return failure();
  if (failed(parseCommRecvClause(parser, recvClause)))
    return failure();

  SmallVector<OpAsmParser::UnresolvedOperand, 2> fixedOperands{dst, acc};
  SmallVector<Type, 2> fixedTypes(2);
  if (failed(parseCommCollectiveTail(
          parser, result, fixedOperands, fixedTypes, recvClause, groupOps,
          groupTypes, {1, 1, 1, recvClause.pong ? 1 : 0},
          {"reduceOp", "root"})))
    return failure();
  return success();
}

void mlir::pto::TReduceOp::print(OpAsmPrinter &p) {
  p << "(" << getDst() << ", " << getAcc() << ", ";
  printCommRecvClause(p, getRecvPing(), getRecvPong());
  p << ", ";
  printCommGroupClause(p, getGroup());
  p << " : " << getDst().getType() << ", " << getAcc().getType() << ", "
    << getRecvPing().getType();
  if (getRecvPong())
    p << ", " << getRecvPong().getType();
  printCommGroupTypes(p, getGroup());
  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes"});
}

ParseResult mlir::pto::MakeTensorViewOp::parse(OpAsmParser &parser,
                                               OperationState &result) {
  OpAsmParser::UnresolvedOperand ptr;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> shapeOps;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> strideOps;

  Type resultTy;

  // %ptr
  if (parser.parseOperand(ptr))
    return failure();

  // , shape = [ ... ]
  if (parser.parseComma() || parser.parseKeyword("shape") || parser.parseEqual() ||
      parser.parseLSquare() ||
      parser.parseOperandList(shapeOps) ||
      parser.parseRSquare())
    return failure();

  // strides = [ ... ]
  if (parser.parseComma() || parser.parseKeyword("strides") || parser.parseEqual() ||
      parser.parseLSquare() ||
      parser.parseOperandList(strideOps) ||
      parser.parseRSquare())
    return failure();

  // attr-dict
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // : result-type
  if (parser.parseColonType(resultTy))
    return failure();
  result.addTypes(resultTy);

  auto tvTy = llvm::dyn_cast<mlir::pto::TensorViewType>(resultTy);
  if (!tvTy)
    return parser.emitError(parser.getCurrentLocation(),
                            "expected result type pto.tensor_view<...>");

  Type elemTy = tvTy.getElementType();

  Type ptrTy = mlir::pto::PtrType::get(parser.getContext(), elemTy);

  // resolve %ptr
  if (parser.resolveOperand(ptr, ptrTy, result.operands))
    return failure();

  // resolve shape/strides 为 index
  Type indexTy = parser.getBuilder().getIndexType();
  if (parser.resolveOperands(shapeOps, indexTy, result.operands))
    return failure();
  if (parser.resolveOperands(strideOps, indexTy, result.operands))
    return failure();

  auto segAttr = parser.getBuilder().getDenseI32ArrayAttr(
      {1, (int32_t)shapeOps.size(), (int32_t)strideOps.size()});
  result.addAttribute("operandSegmentSizes", segAttr);

  return success();
}

void mlir::pto::MakeTensorViewOp::print(OpAsmPrinter &p) {
  p << " " << getPtr();

  p << ", shape = [";
  p.printOperands(getShape());
  p << "]";

  p << ", strides = [";
  p.printOperands(getStrides());
  p << "]";

  p.printOptionalAttrDict((*this)->getAttrs(),
                        /*elidedAttrs=*/{"operandSegmentSizes"});

  p << " : " << getResult().getType();
}

// Layout inference helpers for make_tensor_view
static std::optional<int64_t> getConstIndexValue(Value v) {
  if (auto c = v.getDefiningOp<arith::ConstantIndexOp>())
    return c.value();
  if (auto c = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
      return ia.getInt();
  }
  return std::nullopt;
}

static FailureOr<mlir::pto::PartitionTensorViewType>
inferPartitionViewResultTypeFromSizes(mlir::pto::TensorViewType sourceType,
                                      ValueRange sizes) {
  if (!sourceType)
    return failure();

  if ((int64_t)sizes.size() != sourceType.getRank())
    return failure();

  SmallVector<int64_t, 4> shape;
  shape.reserve(sizes.size());
  for (Value size : sizes) {
    auto constSize = getConstIndexValue(size);
    if (constSize && *constSize >= 0)
      shape.push_back(*constSize);
    else
      shape.push_back(ShapedType::kDynamic);
  }

  return mlir::pto::PartitionTensorViewType::get(
      sourceType.getContext(), shape, sourceType.getElementType());
}

ParseResult mlir::pto::PartitionViewOp::parse(OpAsmParser &parser,
                                              OperationState &result) {
  OpAsmParser::UnresolvedOperand source;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> offsets;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> sizes;
  Type sourceTy;
  Type resultTy;
  bool hasExplicitResultTy = false;

  if (parser.parseOperand(source) || parser.parseComma() ||
      parser.parseKeyword("offsets") || parser.parseEqual() ||
      parser.parseLSquare() || parser.parseOperandList(offsets) ||
      parser.parseRSquare() || parser.parseComma() ||
      parser.parseKeyword("sizes") || parser.parseEqual() ||
      parser.parseLSquare() || parser.parseOperandList(sizes) ||
      parser.parseRSquare() || parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(sourceTy))
    return failure();

  if (succeeded(parser.parseOptionalArrow())) {
    if (parser.parseType(resultTy))
      return failure();
    hasExplicitResultTy = true;
  }

  if (parser.resolveOperand(source, sourceTy, result.operands))
    return failure();

  Type indexTy = parser.getBuilder().getIndexType();
  if (parser.resolveOperands(offsets, indexTy, result.operands) ||
      parser.resolveOperands(sizes, indexTy, result.operands))
    return failure();

  auto &properties = result.getOrAddProperties<PartitionViewOp::Properties>();
  llvm::copy(ArrayRef<int32_t>(
                 {1, static_cast<int32_t>(offsets.size()),
                  static_cast<int32_t>(sizes.size())}),
             properties.operandSegmentSizes.begin());

  if (hasExplicitResultTy) {
    result.addTypes(resultTy);
    return success();
  }

  ValueRange allOperands(result.operands);
  ValueRange sizeOperands =
      allOperands.slice(1 + offsets.size(), sizes.size());
  auto inferredResultType = inferPartitionViewResultTypeFromSizes(
      dyn_cast<mlir::pto::TensorViewType>(sourceTy), sizeOperands);
  if (failed(inferredResultType)) {
    return parser.emitError(parser.getCurrentLocation(),
                            "failed to infer pto.partition_view result type");
  }

  result.addTypes(*inferredResultType);
  return success();
}

void mlir::pto::PartitionViewOp::print(OpAsmPrinter &printer) {
  printer << " " << getSource() << ", offsets = [";
  printer.printOperands(getOffsets());
  printer << "], sizes = [";
  printer.printOperands(getSizes());
  printer << "]";
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                /*elidedAttrs=*/{"operandSegmentSizes"});
  printer << " : " << getSource().getType();

  auto inferredResultType = inferPartitionViewResultTypeFromSizes(
      dyn_cast<mlir::pto::TensorViewType>(getSource().getType()), getSizes());
  if (succeeded(inferredResultType) && *inferredResultType == getResult().getType())
    return;

  printer << " -> " << getResult().getType();
}

static std::optional<int64_t> getConstantIntegerValueEx(
    Value v, bool includeIndexAndIntOpsInConstFold) {
  if (includeIndexAndIntOpsInConstFold) {
    if (auto c = v.getDefiningOp<arith::ConstantIndexOp>())
      return c.value();
    if (auto c = v.getDefiningOp<arith::ConstantIntOp>())
      return c.value();
  }
  if (auto c = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
      return ia.getInt();
  }
  return std::nullopt;
}

static LogicalResult verifyNonNegativeIndexRowCol(
    Operation &op, Value indexRow, Value indexCol,
    bool includeIndexAndIntOpsInConstFold) {
  if (!indexRow.getType().isIndex() || !indexCol.getType().isIndex())
    return op.emitOpError("expects indexRow and indexCol to be index type");
  auto row =
      getConstantIntegerValueEx(indexRow, includeIndexAndIntOpsInConstFold);
  auto col =
      getConstantIntegerValueEx(indexCol, includeIndexAndIntOpsInConstFold);
  if (row && *row < 0)
    return op.emitOpError("expects indexRow to be non-negative");
  if (col && *col < 0)
    return op.emitOpError("expects indexCol to be non-negative");
  return success();
}

static LogicalResult verifyExtractStaticBoundsCommon(
    Operation &op, Value indexRow, Value indexCol, Type srcTy, Type dstTy,
    bool includeIndexAndIntOpsInConstFold) {
  auto row =
      getConstantIntegerValueEx(indexRow, includeIndexAndIntOpsInConstFold);
  auto col =
      getConstantIntegerValueEx(indexCol, includeIndexAndIntOpsInConstFold);
  auto srcShape = getShapeVec(srcTy);
  auto dstShape = getShapeVec(dstTy);
  if (srcShape.size() != 2 || dstShape.size() != 2)
    return op.emitOpError("expects src and dst to be rank-2 tile_buf");
  if (row && srcShape[0] != ShapedType::kDynamic &&
      dstShape[0] != ShapedType::kDynamic &&
      *row + dstShape[0] > srcShape[0])
    return op.emitOpError("expects indexRow + dst.rows <= src.rows");
  if (col && srcShape[1] != ShapedType::kDynamic &&
      dstShape[1] != ShapedType::kDynamic &&
      *col + dstShape[1] > srcShape[1])
    return op.emitOpError("expects indexCol + dst.cols <= src.cols");
  return success();
}

static LogicalResult verifyInsertStaticBoundsCommon(
    Operation &op, Value indexRow, Value indexCol, Type srcTy, Type dstTy,
    bool includeIndexAndIntOpsInConstFold) {
  auto row =
      getConstantIntegerValueEx(indexRow, includeIndexAndIntOpsInConstFold);
  auto col =
      getConstantIntegerValueEx(indexCol, includeIndexAndIntOpsInConstFold);
  auto srcShape = getValidShapeVec(srcTy);
  auto dstShape = getShapeVec(dstTy);
  if (srcShape.size() != 2 || dstShape.size() != 2)
    return op.emitOpError("expects src and dst to be rank-2 tile_buf");
  if (row && srcShape[0] != ShapedType::kDynamic &&
      dstShape[0] != ShapedType::kDynamic &&
      *row + srcShape[0] > dstShape[0])
    return op.emitOpError("expects indexRow + src.rows <= dst.rows");
  if (col && srcShape[1] != ShapedType::kDynamic &&
      dstShape[1] != ShapedType::kDynamic &&
      *col + srcShape[1] > dstShape[1])
    return op.emitOpError("expects indexCol + src.cols <= dst.cols");
  return success();
}

static unsigned getElemByteSize(Type ty) {
  return getPTOStorageElemByteSize(ty);
}

static LogicalResult verifyTileBufLayoutConstraints(Operation *op,
                                                    pto::TileBufType tb,
                                                    StringRef name) {
  auto shape = tb.getShape();
  if (shape.size() != 2)
    return op->emitOpError() << "expects " << name << " to be rank-2";

  int64_t rows = shape[0];
  int64_t cols = shape[1];
  if (rows != ShapedType::kDynamic && rows <= 0)
    return op->emitOpError() << "expects " << name << " rows to be positive";
  if (cols != ShapedType::kDynamic && cols <= 0)
    return op->emitOpError() << "expects " << name << " cols to be positive";

  unsigned elemBytes = getElemByteSize(tb.getElementType());
  if (elemBytes == 0)
    return op->emitOpError() << "expects " << name
                             << " element type to have a byte size";

  auto cfg = tb.getConfigAttr();
  if (!cfg)
    cfg = TileBufConfigAttr::getDefault(tb.getContext());
  auto readBLayout = [](Attribute attr, int32_t &out) -> bool {
    if (auto layout = dyn_cast_or_null<BLayoutAttr>(attr)) {
      out = static_cast<int32_t>(layout.getValue());
      return true;
    }
    if (auto value = dyn_cast_or_null<IntegerAttr>(attr)) {
      out = static_cast<int32_t>(value.getInt());
      return true;
    }
    return false;
  };
  auto readSLayout = [](Attribute attr, int32_t &out) -> bool {
    if (auto layout = dyn_cast_or_null<SLayoutAttr>(attr)) {
      out = static_cast<int32_t>(layout.getValue());
      return true;
    }
    if (auto value = dyn_cast_or_null<IntegerAttr>(attr)) {
      out = static_cast<int32_t>(value.getInt());
      return true;
    }
    return false;
  };
  int32_t blayout = 0;
  int32_t slayout = 0;
  if (!readBLayout(cfg.getBLayout(), blayout) ||
      !readSLayout(cfg.getSLayout(), slayout))
    return op->emitOpError() << "expects " << name
                             << " to have concrete tile layout attributes";
  constexpr int64_t kAlignedBytes = 32;

  auto checkByteAlignment = [&](int64_t dim, StringRef layoutName,
                                StringRef byteExpr) -> LogicalResult {
    if (dim == ShapedType::kDynamic)
      return success();
    int64_t bytes = dim * static_cast<int64_t>(elemBytes);
    if (bytes % kAlignedBytes == 0)
      return success();
    return op->emitOpError()
           << "expects " << name << " " << layoutName
           << " none_box tile " << byteExpr
           << " to be 32-byte aligned, but got " << bytes << " bytes";
  };

  if (slayout == static_cast<int32_t>(SLayout::NoneBox)) {
    if (blayout == static_cast<int32_t>(BLayout::RowMajor))
      return checkByteAlignment(cols, "row-major",
                                "row byte size (cols * sizeof(dtype))");
    return checkByteAlignment(rows, "col-major",
                              "column byte size (rows * sizeof(dtype))");
  }

  int64_t innerRows = 0;
  int64_t innerCols = 0;
  int32_t fractal = static_cast<int32_t>(cfg.getSFractalSize().getInt());
  switch (fractal) {
  case 1024:
    innerRows = 16;
    innerCols = 16;
    break;
  case 32:
    innerRows = 16;
    innerCols = 2;
    break;
  case 512:
    if (kAlignedBytes % elemBytes != 0)
      return op->emitOpError() << "expects " << name
                               << " element byte size to divide 32 for boxed "
                                  "fractal-512 tile layout";
    if (slayout == static_cast<int32_t>(SLayout::RowMajor)) {
      innerRows = 16;
      innerCols = kAlignedBytes / static_cast<int64_t>(elemBytes);
    } else if (slayout == static_cast<int32_t>(SLayout::ColMajor)) {
      innerRows = kAlignedBytes / static_cast<int64_t>(elemBytes);
      innerCols = 16;
    }
    break;
  default:
    break;
  }
  if (innerRows <= 0 || innerCols <= 0)
    return op->emitOpError() << "expects " << name
                             << " to use a supported boxed tile layout";

  auto loc = getPTOMemorySpaceEnum(tb);
  bool allowUnalignedRows =
      (loc && *loc == pto::AddressSpace::VEC) || fractal == 32 || rows == 1;
  if (!allowUnalignedRows && rows != ShapedType::kDynamic &&
      rows % innerRows != 0)
    return op->emitOpError()
           << "expects " << name
           << " boxed tile rows to be a multiple of innerRows (" << innerRows
           << "), but got " << rows;
  if (cols != ShapedType::kDynamic && cols % innerCols != 0)
    return op->emitOpError()
           << "expects " << name
           << " boxed tile cols to be a multiple of innerCols (" << innerCols
           << "), but got " << cols;

  return success();
}

[[maybe_unused]] static bool isSupportedLoadStoreElemTypeA2A3(Type ty) {
  if (ty.isF16() || ty.isBF16() || ty.isF32())
    return true;
  if (auto it = dyn_cast<IntegerType>(ty)) {
    unsigned width = it.getWidth();
    return width == 8 || width == 16 || width == 32 || width == 64;
  }
  return false;
}

static bool isSupportedGatherElemTypeA2A3(Type ty) {
  if (ty.isF16() || ty.isF32())
    return true;
  if (auto it = dyn_cast<IntegerType>(ty)) {
    unsigned width = it.getWidth();
    return width == 16 || width == 32;
  }
  return false;
}

static bool isSupportedGatherElemTypeA5(Type ty) {
  if (isSupportedGatherElemTypeA2A3(ty) || ty.isBF16())
    return true;
  if (isPTOHiFloat8Type(ty))
    return true;
  if (auto ft = dyn_cast<FloatType>(ty)) {
    unsigned width = ft.getWidth();
    return width == 8;
  }
  if (auto it = dyn_cast<IntegerType>(ty))
    return it.getWidth() == 8 || it.getWidth() == 16 || it.getWidth() == 32;
  return false;
}

static bool isStaticLayoutInt(int64_t value) {
  return value != ShapedType::kDynamic && value >= 0;
}

static std::optional<int64_t> multiplyLayoutInts(int64_t lhs, int64_t rhs) {
  int64_t product = 0;
  if (llvm::MulOverflow(lhs, rhs, product))
    return std::nullopt;
  return product;
}

static std::optional<mlir::pto::Layout>
inferLayout(ArrayRef<int64_t> shape, ArrayRef<int64_t> strides,
            unsigned elemBytes) {
  if (shape.size() != strides.size() || elemBytes == 0)
    return std::nullopt;
  if (llvm::any_of(shape, [](int64_t dim) { return !isStaticLayoutInt(dim); }) ||
      llvm::any_of(strides,
                   [](int64_t stride) { return !isStaticLayoutInt(stride); }))
    return std::nullopt;

  // NZ / fractal: rank>=5, check middle dims (sh3/sh4/sh5 per spec)
  if (shape.size() >= 5) {
    int64_t sh3 = shape[2], sh4 = shape[3], sh5 = shape[4];
    int64_t st4 = strides[3], st5 = strides[4];
    auto sh3TimesSh4 = multiplyLayoutInts(sh3, sh4);
    auto fractalBytes =
        sh3TimesSh4
            ? multiplyLayoutInts(*sh3TimesSh4, static_cast<int64_t>(elemBytes))
            : std::nullopt;
    bool alignMatch = (sh3 == 16) && fractalBytes && (*fractalBytes == 512);
    bool strideMatch = (st5 == 1) && (st4 == sh5);
    if (alignMatch && strideMatch)
      return mlir::pto::Layout::NZ;
  }

  // ND: row-major contiguous
  bool isRowMajor = true;
  for (int i = 0, e = (int)shape.size() - 1; i < e; ++i) {
    auto expectedStride = multiplyLayoutInts(strides[i + 1], shape[i + 1]);
    if (!expectedStride || strides[i] != *expectedStride) {
      isRowMajor = false;
      break;
    }
  }
  if (isRowMajor && strides.back() == 1)
    return mlir::pto::Layout::ND;

  // DN: col-major
  bool isColMajor = true;
  for (int i = 0, e = (int)shape.size() - 1; i < e; ++i) {
    auto expectedStride = multiplyLayoutInts(strides[i], shape[i]);
    if (!expectedStride || strides[i + 1] != *expectedStride) {
      isColMajor = false;
      break;
    }
  }
  if (isColMajor && strides.front() == 1)
    return mlir::pto::Layout::DN;

  return mlir::pto::Layout::ND; // fallback
}

static std::optional<pto::Layout> getLogicalViewLayout(Value value) {
  if (!value)
    return std::nullopt;
  if (auto part = value.getDefiningOp<pto::PartitionViewOp>())
    return getLogicalViewLayout(part.getSource());
  if (auto make = value.getDefiningOp<pto::MakeTensorViewOp>()) {
    // Prefer the explicit layout attribute when available.  After rank-2 →
    // rank-5 canonicalization, the padded leading strides satisfy the ND
    // (row-major) recurrence even for DN (col-major) data, so inferLayout
    // alone would misclassify DN as ND (the col-major recurrence breaks at
    // the boundary between padded unit-extent dims and real dims).  The
    // layout attribute carries the *intended* memory layout and is the
    // authoritative source — inferLayout is only a fallback for views that
    // lack an explicit layout.
    if (auto layoutAttr = make.getLayoutAttr())
      return layoutAttr.getLayout();
    auto tvTy = dyn_cast<pto::TensorViewType>(make.getResult().getType());
    if (!tvTy)
      return std::nullopt;
    SmallVector<int64_t> shape(tvTy.getShape().begin(), tvTy.getShape().end());
    SmallVector<int64_t> strides;
    strides.reserve(make.getStrides().size());
    for (Value stride : make.getStrides()) {
      auto cst = getConstIndexValue(stride);
      if (!cst)
        return std::nullopt;
      strides.push_back(*cst);
    }
    return inferLayout(shape, strides, getElemByteSize(tvTy.getElementType()));
  }
  return std::nullopt;
}

static std::optional<pto::Layout> getTileBufLogicalLayout(pto::TileBufType type) {
  if (!type)
    return std::nullopt;
  int32_t sl = type.getSLayoutValueI32();
  int32_t bl = type.getBLayoutValueI32();
  if (sl != static_cast<int32_t>(pto::SLayout::NoneBox))
    return pto::Layout::NZ;
  if (bl == static_cast<int32_t>(pto::BLayout::RowMajor))
    return pto::Layout::ND;
  if (bl == static_cast<int32_t>(pto::BLayout::ColMajor))
    return pto::Layout::DN;
  return std::nullopt;
}

static bool isRowMajorTileBuf(Type ty) {
  auto tb = mlir::dyn_cast<pto::TileBufType>(ty);
  return tb && tb.getBLayoutValueI32() == static_cast<int32_t>(pto::BLayout::RowMajor);
}

static LogicalResult verifyRowReductionSrcLayout(Operation *op, Type ty,
                                                 StringRef name) {
  if (failed(verifyTileBufCommon(op, ty, name)))
    return failure();
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::VEC)
    return op->emitOpError() << "expects " << name << " to be in the vec address space";
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    if (tb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor))
      return op->emitOpError() << "expects " << name << " to use the row_major blayout";
  }
  if (auto mr = dyn_cast<MemRefType>(ty))
    (void)mr;
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    if (tb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox))
      return op->emitOpError() << "expects " << name
                               << " to use the none_box slayout";
  }
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    auto layout = getTileBufLogicalLayout(tb);
    if (layout && *layout != pto::Layout::ND)
      return op->emitOpError() << "expects " << name
                               << " to use an ND-style tile layout";
  }
  return success();
}

static LogicalResult verifyRowReductionDstLayout(Operation *op, Type ty,
                                                 StringRef name) {
  if (failed(verifyTileBufCommon(op, ty, name)))
    return failure();
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::VEC)
    return op->emitOpError() << "expects " << name << " to be in the vec address space";
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    if (tb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox))
      return op->emitOpError() << "expects " << name
                               << " to use the none_box slayout";
  }
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    if (tb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) &&
        tb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor))
      return op->emitOpError() << "expects " << name
                               << " to use the row_major or col_major blayout";
  }
  if (auto mr = dyn_cast<MemRefType>(ty))
    (void)mr;
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    auto layout = getTileBufLogicalLayout(tb);
    if (layout && *layout == pto::Layout::DN) {
      auto shape = getShapeVec(ty);
      if (shape.size() == 2 && shape[1] != ShapedType::kDynamic && shape[1] != 1)
        return op->emitOpError() << "expects DN-style " << name
                                 << " to have shape[1] == 1";
      return success();
    }
    if (layout && *layout == pto::Layout::ND)
      return success();
    if (layout)
      return op->emitOpError() << "expects " << name
                               << " to use a DN-style column vector tile or legacy ND-style tile";
  }
  // The dst valid_shape[1] == 1 constraint for row reductions is enforced in
  // verifyRowReductionValidRegion (it must be conditional on the no-op-marker
  // path), so it is intentionally not duplicated here. A previous unreachable
  // copy of that check lived after this return and has been removed.
  return success();
}

static LogicalResult verifyRowReductionValidRegion(Operation *op, Type srcTy,
                                                   Type dstTy,
                                                   bool allowEmptyMarker) {
  auto srcValid = getValidShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (srcValid.size() != 2 || dstValid.size() != 2)
    return op->emitOpError("expects src and dst to have rank-2 valid_shape");
  // A fully-empty dst valid region (0x0) is PyPTO's dual-AIV no-op replay
  // marker: the op writes no elements, so accept it and skip the non-empty
  // structural constraints. Only plain reductions opt in (allowEmptyMarker);
  // arg reductions (trowargmax/trowargmin) still produce a real per-row index,
  // so they stay strict. One-sided empties (only one dim 0) still fall through
  // and are rejected below. Hardware Rv=0 no-op is tracked in pto-isa#143;
  // PTOAS only guarantees the IR is legal here.
  if (allowEmptyMarker && dstValid[0] == 0 && dstValid[1] == 0)
    return success();
  if (srcValid[0] != ShapedType::kDynamic && srcValid[0] == 0)
    return op->emitOpError("expects src valid_shape[0] to be non-zero");
  if (srcValid[1] != ShapedType::kDynamic && srcValid[1] == 0)
    return op->emitOpError("expects src valid_shape[1] to be non-zero");
  if (srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
      srcValid[0] != dstValid[0])
    return op->emitOpError("expects src and dst to have the same valid_shape[0]");
  if (dstValid[1] != ShapedType::kDynamic && dstValid[1] != 1)
    return op->emitOpError("expects dst valid_shape[1] to be 1");
  return success();
}

static bool isSupportedRowReductionElemType(Type elem) {
  return elem.isInteger(16) || elem.isInteger(32) || elem.isF16() ||
         elem.isF32();
}

[[maybe_unused]] static LogicalResult
verifyTRowReductionNoTmpCommon(Operation *op, Type srcTy, Type dstTy,
                               StringRef elemTypeError) {
  if (failed(verifyRowReductionSrcLayout(op, srcTy, "src")) ||
      failed(verifyRowReductionDstLayout(op, dstTy, "dst")))
    return failure();
  if (getElemTy(srcTy) != getElemTy(dstTy))
    return op->emitOpError("expects src and dst to have the same element type");
  if (failed(verifyRowReductionValidRegion(op, srcTy, dstTy,
                                           /*allowEmptyMarker=*/true)))
    return failure();
  if (!isSupportedRowReductionElemType(getElemTy(srcTy)))
    return op->emitOpError(elemTypeError);
  return success();
}

static LogicalResult verifyTRowReductionWithTmpCommon(Operation *op, Type srcTy,
                                                      Type tmpTy, Type dstTy,
                                                      StringRef elemTypeError) {
  if (failed(verifyRowReductionSrcLayout(op, srcTy, "src")) ||
      failed(verifyVecTileStorage(op, tmpTy, "tmp")) ||
      failed(verifyRowReductionDstLayout(op, dstTy, "dst")))
    return failure();
  if (getElemTy(srcTy) != getElemTy(dstTy))
    return op->emitOpError("expects src and dst to have the same element type");
  if (failed(verifyRowReductionValidRegion(op, srcTy, dstTy,
                                           /*allowEmptyMarker=*/true)))
    return failure();
  if (!isSupportedRowReductionElemType(getElemTy(srcTy)))
    return op->emitOpError(elemTypeError);
  return success();
}

static std::optional<int64_t> getVectorRepeatElements(Type elemTy) {
  unsigned elemBits = elemTy ? getPTOStorageElemBitWidth(elemTy) : 0;
  if (elemBits == 0 || 2048 % elemBits != 0)
    return std::nullopt;
  return static_cast<int64_t>(2048 / elemBits);
}

static std::optional<int64_t> getVectorBlockElements(Type elemTy) {
  unsigned elemBits = elemTy ? getPTOStorageElemBitWidth(elemTy) : 0;
  if (elemBits == 0 || 256 % elemBits != 0)
    return std::nullopt;
  return static_cast<int64_t>(256 / elemBits);
}

static int64_t ceilDivInt64(int64_t numerator, int64_t denominator) {
  assert(denominator > 0 && "denominator must be positive");
  assert(numerator >= 0 && "numerator must be non-negative");
  return (numerator + denominator - 1) / denominator;
}

static std::optional<int64_t> getArgReductionTmpMinStride(Type elemTy,
                                                          int64_t srcValidCols) {
  if (srcValidCols == ShapedType::kDynamic || srcValidCols < 0)
    return std::nullopt;
  auto repeatElems = getVectorRepeatElements(elemTy);
  auto blockElems = getVectorBlockElements(elemTy);
  if (!repeatElems || !blockElems)
    return std::nullopt;
  int64_t repeats = ceilDivInt64(srcValidCols, *repeatElems);
  return (ceilDivInt64(repeats * 2, *blockElems) +
          ceilDivInt64(repeats, *blockElems)) *
         *blockElems;
}

static bool hasExactKnownValidShape(Type lhsTy, Type rhsTy) {
  return getValidShapeVec(lhsTy) == getValidShapeVec(rhsTy);
}

static LogicalResult verifyTColArgTmpA2A3(Operation *op, Type srcTy,
                                          Type tmpTy) {
  if (failed(verifyVecTileCommon(op, tmpTy, "tmp")) ||
      failed(verifyTileBufSameElemType(op, srcTy, tmpTy, "src", "tmp")))
    return failure();

  if (hasExactKnownValidShape(srcTy, tmpTy))
    return success();

  auto srcValid = getValidShapeVec(srcTy);
  auto tmpValid = getValidShapeVec(tmpTy);
  if (srcValid.size() != 2 || tmpValid.size() != 2)
    return op->emitOpError("expects src and tmp to have rank-2 valid_shape");
  if (tmpValid[0] != ShapedType::kDynamic && tmpValid[0] < 1)
    return op->emitOpError("expects A2/A3 tmp valid_shape[0] to be at least 1");
  if (srcValid[1] != ShapedType::kDynamic) {
    auto minStride = getArgReductionTmpMinStride(getElemTy(srcTy), srcValid[1]);
    if (!minStride)
      return op->emitOpError("failed to infer A2/A3 tmp stride from src element type");
    if (tmpValid[1] != ShapedType::kDynamic && tmpValid[1] < *minStride)
      return op->emitOpError()
             << "expects A2/A3 tmp valid_shape[1] to be at least "
             << *minStride << " for src valid_shape[1] = " << srcValid[1];
  }
  return success();
}

static LogicalResult verifyTColArgReductionOpA2A3(Operation *op, Type srcTy,
                                                  Type tmpTy, Type dstTy) {
  if (failed(verifyNDStyleVecTile(op, srcTy, "src")) ||
      failed(verifyTColArgTmpA2A3(op, srcTy, tmpTy)) ||
      failed(verifyColArgReductionDstLayout(op, dstTy, "dst")))
    return failure();
  if (failed(verifyColReductionValidRegion(op, srcTy, dstTy,
                                           /*requireNonZeroSrc=*/true)))
    return failure();
  Type srcElemTy = getElemTy(srcTy);
  unsigned srcElemBits = srcElemTy ? getPTOStorageElemBitWidth(srcElemTy) : 0;
  if (!(mlir::isa<IntegerType, FloatType>(srcElemTy) &&
        (srcElemBits == 8 || srcElemBits == 16 || srcElemBits == 32)))
    return op->emitOpError(
        "expects src/tmp element type to be 1, 2, or 4 bytes wide");
  auto dstInt = dyn_cast<IntegerType>(getElemTy(dstTy));
  if (!dstInt || dstInt.getWidth() != 32)
    return op->emitOpError("expects dst element type to be i32 or ui32");
  return success();
}

static LogicalResult verifyTColArgReductionOpA5(Operation *op, Type srcTy,
                                                Type tmpTy, Type dstTy) {
  if (failed(verifyNDStyleVecTile(op, srcTy, "src")) ||
      failed(verifyVecTileCommon(op, tmpTy, "tmp")) ||
      failed(verifyColArgReductionDstLayout(op, dstTy, "dst")))
    return failure();
  if (failed(verifyColReductionValidRegion(op, srcTy, dstTy,
                                           /*requireNonZeroSrc=*/true)))
    return failure();
  Type srcElemTy = getElemTy(srcTy);
  unsigned srcElemBits = srcElemTy ? getPTOStorageElemBitWidth(srcElemTy) : 0;
  if (!(mlir::isa<IntegerType, FloatType>(srcElemTy) &&
        (srcElemBits == 8 || srcElemBits == 16 || srcElemBits == 32)))
    return op->emitOpError(
        "expects src element type to be 1, 2, or 4 bytes wide");
  auto dstInt = dyn_cast<IntegerType>(getElemTy(dstTy));
  if (!dstInt || dstInt.getWidth() != 32)
    return op->emitOpError("expects dst element type to be i32 or ui32");
  return success();
}

static LogicalResult verifyTColSumTmpStride(Operation *op, Type srcTy,
                                            Type tmpTy, bool isBinary) {
  if (!isBinary)
    return success();

  auto srcValid = getValidShapeVec(srcTy);
  auto tmpShape = getShapeVec(tmpTy);
  if (srcValid.size() != 2 || tmpShape.size() != 2)
    return op->emitOpError("expects src and tmp to be rank-2 tiles");

  int64_t srcValidCols = srcValid[1];
  int64_t tmpStride = tmpShape[1];
  if (srcValidCols != ShapedType::kDynamic && tmpStride != ShapedType::kDynamic &&
      tmpStride < srcValidCols) {
    return op->emitOpError()
           << "expects tmp shape[1] to be at least src valid_shape[1] when "
              "isBinary is true; got "
           << tmpStride << " vs " << srcValidCols;
  }
  return success();
}

static LogicalResult verifyTRowArgTmpA2A3(Operation *op, Type srcTy,
                                          Type tmpTy) {
  if (failed(verifyVecTileStorage(op, tmpTy, "tmp")) ||
      failed(verifyTileBufSameElemType(op, srcTy, tmpTy, "src", "tmp")))
    return failure();

  if (hasExactKnownValidShape(srcTy, tmpTy))
    return success();

  auto srcShape = getShapeVec(srcTy);
  auto tmpShape = getShapeVec(tmpTy);
  auto srcValid = getValidShapeVec(srcTy);
  auto tmpValid = getValidShapeVec(tmpTy);
  if (srcShape.size() != 2 || tmpShape.size() != 2 || srcValid.size() != 2 ||
      tmpValid.size() != 2)
    return op->emitOpError("expects src and tmp to be rank-2 tiles");

  auto repeatElems = getVectorRepeatElements(getElemTy(srcTy));
  if (!repeatElems)
    return op->emitOpError("failed to infer A2/A3 tmp contract from src element type");

  if (srcValid[1] != ShapedType::kDynamic && srcValid[1] <= *repeatElems) {
    auto tmpTile = dyn_cast<pto::TileBufType>(tmpTy);
    auto layout = tmpTile ? getTileBufLogicalLayout(tmpTile) : std::nullopt;
    if (layout && *layout == pto::Layout::DN) {
      if (tmpShape[1] != ShapedType::kDynamic && tmpShape[1] != 1)
        return op->emitOpError("expects A2/A3 tmp DN layout to have shape[1] == 1");
      if (tmpValid[1] != ShapedType::kDynamic && tmpValid[1] != 1)
        return op->emitOpError(
            "expects A2/A3 tmp DN layout to have valid_shape[1] == 1");
      if (srcValid[0] != ShapedType::kDynamic && tmpValid[0] != ShapedType::kDynamic &&
          tmpValid[0] < srcValid[0] * 2)
        return op->emitOpError()
               << "expects A2/A3 tmp DN layout to have valid_shape[0] >= "
               << (srcValid[0] * 2);
      return success();
    }

    if (!layout || *layout != pto::Layout::ND)
      return op->emitOpError(
          "expects A2/A3 tmp to use DN 1-col or ND 2-col layout when src valid_shape[1] fits in one repeat");
    if (failed(verifyVecTileCommon(op, tmpTy, "tmp")))
      return failure();
    if (srcValid[0] != ShapedType::kDynamic && tmpValid[0] != ShapedType::kDynamic &&
        tmpValid[0] < srcValid[0])
      return op->emitOpError("expects A2/A3 tmp valid_shape[0] to cover src valid rows");
    if (tmpValid[1] != ShapedType::kDynamic && tmpValid[1] < 2)
      return op->emitOpError(
          "expects A2/A3 tmp valid_shape[1] to be at least 2 in the small-col ND path");
    return success();
  }

  if (failed(verifyVecTileCommon(op, tmpTy, "tmp")))
    return failure();
  if (srcShape[0] != ShapedType::kDynamic && tmpShape[0] != ShapedType::kDynamic &&
      tmpShape[0] != srcShape[0])
    return op->emitOpError("expects A2/A3 tmp shape[0] to match src shape[0]");
  if (srcValid[0] != ShapedType::kDynamic && tmpValid[0] != ShapedType::kDynamic &&
      tmpValid[0] < srcValid[0])
    return op->emitOpError("expects A2/A3 tmp valid_shape[0] to cover src valid rows");
  if (srcValid[1] != ShapedType::kDynamic) {
    auto minStride = getArgReductionTmpMinStride(getElemTy(srcTy), srcValid[1]);
    if (!minStride)
      return op->emitOpError("failed to infer A2/A3 tmp stride from src element type");
    if (tmpValid[1] != ShapedType::kDynamic && tmpValid[1] < *minStride)
      return op->emitOpError()
             << "expects A2/A3 tmp valid_shape[1] to be at least "
             << *minStride << " for src valid_shape[1] = " << srcValid[1];
  }
  return success();
}

static LogicalResult verifyTRowArgReductionOpA2A3(Operation *op, Type srcTy,
                                                  Type tmpTy, Type dstTy) {
  if (failed(verifyRowReductionSrcLayout(op, srcTy, "src")) ||
      failed(verifyTRowArgTmpA2A3(op, srcTy, tmpTy)) ||
      failed(verifyRowReductionDstLayout(op, dstTy, "dst")))
    return failure();
  if (failed(verifyRowReductionValidRegion(op, srcTy, dstTy,
                                           /*allowEmptyMarker=*/false)))
    return failure();
  Type srcElem = getElemTy(srcTy);
  if (!isSupportedRowReductionElemType(srcElem))
    return op->emitOpError("expects src element type to be i16/i32/f16/f32");
  auto dstInt = dyn_cast<IntegerType>(getElemTy(dstTy));
  if (!dstInt || dstInt.getWidth() != 32)
    return op->emitOpError("expects dst element type to be i32 or ui32");
  return success();
}

static LogicalResult verifyTRowArgReductionOpA5(Operation *op, Type srcTy,
                                                Type tmpTy, Type dstTy) {
  if (failed(verifyRowReductionSrcLayout(op, srcTy, "src")) ||
      failed(verifyVecTileCommon(op, tmpTy, "tmp")) ||
      failed(verifyRowReductionDstLayout(op, dstTy, "dst")))
    return failure();
  if (failed(verifyRowReductionValidRegion(op, srcTy, dstTy,
                                           /*allowEmptyMarker=*/false)))
    return failure();
  Type srcElem = getElemTy(srcTy);
  if (!isSupportedRowReductionElemType(srcElem))
    return op->emitOpError("expects src element type to be i16/i32/f16/f32");
  auto dstInt = dyn_cast<IntegerType>(getElemTy(dstTy));
  if (!dstInt || dstInt.getWidth() != 32)
    return op->emitOpError("expects dst element type to be i32 or ui32");
  return success();
}

static LogicalResult verifyNDStyleVecTile(Operation *op, Type ty, StringRef name,
                                          bool allowLowPrecision) {
  if (failed(verifyTileBufCommon(op, ty, name, allowLowPrecision)))
    return failure();
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::VEC)
    return op->emitOpError() << "expects " << name << " to be in the vec address space";
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    if (tb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor))
      return op->emitOpError() << "expects " << name << " to use the row_major blayout";
    if (tb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox))
      return op->emitOpError() << "expects " << name << " to use the none_box slayout";
  }
  return success();
}

// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

static void printLegacyOrAttrMemBar(OpAsmPrinter &p, MemBarAttr kind,
                                    ArrayRef<NamedAttribute> attrs) {
  p << ' ' << '"' << stringifyMemBarKind(kind.getKind()) << '"';
  p.printOptionalAttrDict(attrs, {"kind"});
}

static ParseResult parseLegacyOrAttrDsbMem(OpAsmParser &parser,
                                           DsbMemAttr &attr) {
  auto loc = parser.getCurrentLocation();
  std::string token;
  if (succeeded(parser.parseOptionalString(&token))) {
    auto kind = symbolizeDsbMem(token);
    if (!kind) {
      return parser.emitError(loc) << "invalid dsb memory token: " << token;
    }
    attr = DsbMemAttr::get(parser.getContext(), *kind);
    return success();
  }

  Attribute parsed;
  if (failed(parser.parseAttribute(parsed))) {
    return failure();
  }
  auto dsbMemAttr = dyn_cast<DsbMemAttr>(parsed);
  if (!dsbMemAttr) {
    return parser.emitError(loc, "expected dsb_mem attribute");
  }
  attr = dsbMemAttr;
  return success();
}

static void printLegacyOrAttrDsbMem(OpAsmPrinter &printer, Operation *op,
                                    DsbMemAttr mem) {
  (void)op;
  printer << ' ' << '"' << stringifyDsbMem(mem.getKind()) << '"';
}

static ParseResult parseLegacyOrAttrDcciCacheLine(OpAsmParser &parser,
                                                  DcciCacheLineAttr &attr) {
  auto loc = parser.getCurrentLocation();
  std::string token;
  if (succeeded(parser.parseOptionalString(&token))) {
    auto kind = symbolizeDcciCacheLine(token);
    if (!kind) {
      return parser.emitError(loc) << "invalid dcci cache token: " << token;
    }
    attr = DcciCacheLineAttr::get(parser.getContext(), *kind);
    return success();
  }

  Attribute parsed;
  if (failed(parser.parseAttribute(parsed))) {
    return failure();
  }
  auto cacheAttr = dyn_cast<DcciCacheLineAttr>(parsed);
  if (!cacheAttr) {
    return parser.emitError(loc, "expected dcci_cache_line attribute");
  }
  attr = cacheAttr;
  return success();
}

static void printLegacyOrAttrDcciCacheLine(OpAsmPrinter &printer, Operation *op,
                                           DcciCacheLineAttr cache) {
  (void)op;
  printer << ' ' << '"' << stringifyDcciCacheLine(cache.getKind()) << '"';
}

static ParseResult parseOptionalDcciDst(OpAsmParser &parser,
                                        DcciDstAttr &attr) {
  if (failed(parser.parseOptionalComma())) {
    return success();
  }

  auto loc = parser.getCurrentLocation();
  std::string token;
  if (succeeded(parser.parseOptionalString(&token))) {
    auto kind = symbolizeDcciDst(token);
    if (!kind) {
      return parser.emitError(loc) << "invalid dcci dst token: " << token;
    }
    attr = DcciDstAttr::get(parser.getContext(), *kind);
    return success();
  }

  Attribute parsed;
  if (failed(parser.parseAttribute(parsed))) {
    return failure();
  }
  auto dstAttr = dyn_cast<DcciDstAttr>(parsed);
  if (!dstAttr) {
    return parser.emitError(loc, "expected dcci_dst attribute");
  }
  attr = dstAttr;
  return success();
}

static void printOptionalDcciDst(OpAsmPrinter &printer, Operation *op,
                                 DcciDstAttr dst) {
  (void)op;
  if (!dst) {
    return;
  }
  printer << ", \"" << stringifyDcciDst(dst.getKind()) << '"';
}

LogicalResult DcciOp::verify() {
  auto space = getPTOMemorySpaceEnum(getPtr().getType());
  if (!space) {
    return emitOpError("expects ptr to have a PTO memory space");
  }
  if (*space != pto::AddressSpace::GM && *space != pto::AddressSpace::VEC) {
    return emitOpError("expects ptr memory space to be gm or ub/vec");
  }

  return success();
}

static ParseResult parseLegacyOrAttrPipe(OpAsmParser &parser, PipeAttr &attr) {
  auto loc = parser.getCurrentLocation();
  std::string token;
  if (succeeded(parser.parseOptionalString(&token))) {
    auto pipe = symbolizePIPE(token);
    if (!pipe) {
      return parser.emitError(loc) << "invalid pipe token: " << token;
    }
    attr = PipeAttr::get(parser.getContext(), *pipe);
    return success();
  }

  if (succeeded(parser.parseOptionalLess())) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword) || parser.parseGreater()) {
      return failure();
    }
    auto pipe = symbolizePIPE(keyword);
    if (!pipe) {
      return parser.emitError(loc) << "invalid pipe token: " << keyword;
    }
    attr = PipeAttr::get(parser.getContext(), *pipe);
    return success();
  }

  Attribute parsed;
  if (failed(parser.parseAttribute(parsed))) {
    return failure();
  }
  auto pipeAttr = dyn_cast<PipeAttr>(parsed);
  if (!pipeAttr) {
    return parser.emitError(loc, "expected pipe attribute");
  }
  attr = pipeAttr;
  return success();
}

static ParseResult parseLegacyOrAttrEvent(OpAsmParser &parser, EventAttr &attr) {
  auto loc = parser.getCurrentLocation();
  std::string token;
  if (succeeded(parser.parseOptionalString(&token))) {
    auto event = symbolizeEVENT(token);
    if (!event) {
      return parser.emitError(loc) << "invalid event token: " << token;
    }
    attr = EventAttr::get(parser.getContext(), *event);
    return success();
  }

  if (succeeded(parser.parseOptionalLess())) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword) || parser.parseGreater()) {
      return failure();
    }
    auto event = symbolizeEVENT(keyword);
    if (!event) {
      return parser.emitError(loc) << "invalid event token: " << keyword;
    }
    attr = EventAttr::get(parser.getContext(), *event);
    return success();
  }

  Attribute parsed;
  if (failed(parser.parseAttribute(parsed))) {
    return failure();
  }
  auto eventAttr = dyn_cast<EventAttr>(parsed);
  if (!eventAttr) {
    return parser.emitError(loc, "expected event attribute");
  }
  attr = eventAttr;
  return success();
}

static ParseResult parseI32LiteralAttr(OpAsmParser &parser, IntegerAttr &attr) {
  auto loc = parser.getCurrentLocation();
  int64_t value = 0;
  if (failed(parser.parseInteger(value))) {
    return failure();
  }
  if (value < std::numeric_limits<int32_t>::min() ||
      value > std::numeric_limits<int32_t>::max()) {
    return parser.emitError(loc, "expected 32-bit integer literal");
  }
  attr = IntegerAttr::get(IntegerType::get(parser.getContext(), 32), value);
  return success();
}

static ParseResult parseOptionalSyncMode(OpAsmParser &parser,
                                         IntegerAttr &modeAttr) {
  if (succeeded(parser.parseOptionalComma()))
    return parseI32LiteralAttr(parser, modeAttr);
  modeAttr = IntegerAttr::get(IntegerType::get(parser.getContext(), 32), 0);
  return success();
}

static void printLegacySyncTriplet(OpAsmPrinter &p, PipeAttr srcPipe,
                                   PipeAttr dstPipe, EventAttr eventId,
                                   ArrayRef<NamedAttribute> attrs) {
  p << "[<" << stringifyPIPE(srcPipe.getPipe()) << ">, <"
    << stringifyPIPE(dstPipe.getPipe()) << ">, <"
    << stringifyEVENT(eventId.getEvent()) << ">]";
  p.printOptionalAttrDict(attrs, {"src_pipe", "dst_pipe", "event_id"});
}

static ParseResult parseLegacySyncTriplet(OpAsmParser &parser,
                                          OperationState &result) {
  PipeAttr srcPipe;
  PipeAttr dstPipe;
  EventAttr eventId;
  if (parser.parseLSquare() || parseLegacyOrAttrPipe(parser, srcPipe) ||
      parser.parseComma() || parseLegacyOrAttrPipe(parser, dstPipe) ||
      parser.parseComma() || parseLegacyOrAttrEvent(parser, eventId) ||
      parser.parseRSquare() || parser.parseOptionalAttrDict(result.attributes))
    return failure();
  result.addAttribute("src_pipe", srcPipe);
  result.addAttribute("dst_pipe", dstPipe);
  result.addAttribute("event_id", eventId);
  return success();
}

ParseResult SetFlagOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseLegacySyncTriplet(parser, result);
}

void SetFlagOp::print(OpAsmPrinter &p) {
  printLegacySyncTriplet(p, getSrcPipe(), getDstPipe(), getEventId(),
                         (*this)->getAttrs());
}

ParseResult WaitFlagOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseLegacySyncTriplet(parser, result);
}

void WaitFlagOp::print(OpAsmPrinter &p) {
  printLegacySyncTriplet(p, getSrcPipe(), getDstPipe(), getEventId(),
                         (*this)->getAttrs());
}

ParseResult MemBarOp::parse(OpAsmParser &parser, OperationState &result) {
  MemBarAttr kind;
  if (parseLegacyOrAttrMemBar(parser, kind)) {
    return failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }
  result.addAttribute("kind", kind);
  return success();
}

void MemBarOp::print(OpAsmPrinter &p) {
  printLegacyOrAttrMemBar(p, getKind(), (*this)->getAttrs());
}

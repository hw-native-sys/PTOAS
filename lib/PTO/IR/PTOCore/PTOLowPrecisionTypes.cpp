// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

llvm::TypeSize mlir::pto::HiF8Type::getTypeSizeInBits(
    const DataLayout &, DataLayoutEntryListRef) const {
  return getOneByteTypeSize();
}

llvm::TypeSize mlir::pto::F8E8M0Type::getTypeSizeInBits(
    const DataLayout &, DataLayoutEntryListRef) const {
  return getOneByteTypeSize();
}

uint64_t mlir::pto::HiF8Type::getABIAlignment(const DataLayout &,
                                              DataLayoutEntryListRef) const {
  return 1;
}

uint64_t mlir::pto::F8E8M0Type::getABIAlignment(const DataLayout &,
                                                DataLayoutEntryListRef) const {
  return 1;
}

uint64_t mlir::pto::HiF8Type::getPreferredAlignment(
    const DataLayout &, DataLayoutEntryListRef) const {
  return 1;
}

uint64_t mlir::pto::F8E8M0Type::getPreferredAlignment(
    const DataLayout &, DataLayoutEntryListRef) const {
  return 1;
}

static llvm::TypeSize getTwoByteTypeSize() { return llvm::TypeSize::getFixed(mlir::pto::kValue16); }

llvm::TypeSize mlir::pto::HiF8x2Type::getTypeSizeInBits(
    const DataLayout &, DataLayoutEntryListRef) const {
  return getTwoByteTypeSize();
}

uint64_t mlir::pto::HiF8x2Type::getABIAlignment(
    const DataLayout &, DataLayoutEntryListRef) const {
    return mlir::pto::kValue2;
}

uint64_t mlir::pto::HiF8x2Type::getPreferredAlignment(
    const DataLayout &, DataLayoutEntryListRef) const {
    return mlir::pto::kValue2;
}

llvm::TypeSize mlir::pto::F4E1M2x2Type::getTypeSizeInBits(
    const DataLayout &, DataLayoutEntryListRef) const {
  return getOneByteTypeSize();
}

uint64_t mlir::pto::F4E1M2x2Type::getABIAlignment(
    const DataLayout &, DataLayoutEntryListRef) const {
  return 1;
}

uint64_t mlir::pto::F4E1M2x2Type::getPreferredAlignment(
    const DataLayout &, DataLayoutEntryListRef) const {
  return 1;
}

llvm::TypeSize mlir::pto::F4E2M1x2Type::getTypeSizeInBits(
    const DataLayout &, DataLayoutEntryListRef) const {
  return getOneByteTypeSize();
}

uint64_t mlir::pto::F4E2M1x2Type::getABIAlignment(
    const DataLayout &, DataLayoutEntryListRef) const {
  return 1;
}

uint64_t mlir::pto::F4E2M1x2Type::getPreferredAlignment(
    const DataLayout &, DataLayoutEntryListRef) const {
  return 1;
}

static llvm::TypeSize getFourByteTypeSize() { return llvm::TypeSize::getFixed(mlir::pto::kValue32); }

llvm::TypeSize mlir::pto::BF16x2Type::getTypeSizeInBits(
    const DataLayout &, DataLayoutEntryListRef) const {
  return getFourByteTypeSize();
}

uint64_t mlir::pto::BF16x2Type::getABIAlignment(
    const DataLayout &, DataLayoutEntryListRef) const {
    return mlir::pto::kValue4;
}

uint64_t mlir::pto::BF16x2Type::getPreferredAlignment(
    const DataLayout &, DataLayoutEntryListRef) const {
    return mlir::pto::kValue4;
}

static VerifierTargetArch getVerifierTargetArch(Operation *op) {
  auto module = op ? op->getParentOfType<ModuleOp>() : ModuleOp();
  if (isA5ModuleTarget(module)) {
    return VerifierTargetArch::A5;
  }

  if (auto archName = getVerifierArchName(op)) {
    return archName->equals_insensitive("a5") ? VerifierTargetArch::A5
                            : VerifierTargetArch::A2A3;
  }

  switch (getPTOParserTargetArch(op ? op->getContext() : nullptr)) {
  case PTOParserTargetArch::A5:
    return VerifierTargetArch::A5;
  case PTOParserTargetArch::A3:
  case PTOParserTargetArch::Unspecified:
    return VerifierTargetArch::A2A3;
  }

  return VerifierTargetArch::A2A3;
}

static std::optional<StringRef> getVerifierArchName(Operation *op) {
  auto module = op ? op->getParentOfType<ModuleOp>() : ModuleOp();
  if (!module) {
    return std::nullopt;
  }
  if (auto arch = module->getAttrOfType<StringAttr>(kPTOTargetArchAttrName)) {
    return arch.getValue();
  }
  return std::nullopt;
}

static SmallVector<int64_t, mlir::pto::kValue4> canonicalizeTileBufValidShape(ArrayRef<int64_t> validShape)
{
    SmallVector<int64_t, mlir::pto::kValue4> canonical;
    canonical.reserve(validShape.size());
    for (int64_t dim : validShape) {
        canonical.push_back(dim < 0 ? ShapedType::kDynamic : dim);
    }
    return canonical;
}

template <typename FnA2A3, typename FnA5>
static LogicalResult dispatchVerifierByArch(Operation *op, FnA2A3 &&verifyA2A3,
                                            FnA5 &&verifyA5) {
  switch (getVerifierTargetArch(op)) {
  case VerifierTargetArch::A2A3:
    return verifyA2A3();
  case VerifierTargetArch::A5:
    return verifyA5();
  }
  return failure();
}
static std::optional<pto::AddressSpace> parsePtrAddressSpaceKeyword(StringRef keyword) {
  return llvm::StringSwitch<std::optional<pto::AddressSpace>>(keyword)
      .Case("gm", pto::AddressSpace::GM)
      .Case("mat", pto::AddressSpace::MAT)
      .Case("l1", pto::AddressSpace::MAT)
      .Case("left", pto::AddressSpace::LEFT)
      .Case("l0a", pto::AddressSpace::LEFT)
      .Case("right", pto::AddressSpace::RIGHT)
      .Case("l0b", pto::AddressSpace::RIGHT)
      .Case("acc", pto::AddressSpace::ACC)
      .Case("l0c", pto::AddressSpace::ACC)
      .Case("vec", pto::AddressSpace::VEC)
      .Case("ub", pto::AddressSpace::VEC)
      .Case("bias", pto::AddressSpace::BIAS)
      .Case("bt", pto::AddressSpace::BIAS)
      .Case("scaling", pto::AddressSpace::SCALING)
      .Case("fb", pto::AddressSpace::SCALING)
      .Default(std::nullopt);
}

static StringRef printPtrAddressSpaceKeyword(pto::AddressSpace space) {
  switch (space) {
  case pto::AddressSpace::GM:
  case pto::AddressSpace::Zero:
    return "gm";
  case pto::AddressSpace::MAT:
    return "l1";
  case pto::AddressSpace::LEFT:
    return "l0a";
  case pto::AddressSpace::RIGHT:
    return "l0b";
  case pto::AddressSpace::ACC:
    return "l0c";
  case pto::AddressSpace::VEC:
    return "ub";
  case pto::AddressSpace::BIAS:
    return "bt";
  case pto::AddressSpace::SCALING:
    return "fb";
  }
  llvm_unreachable("unhandled pointer address space");
}

static ParseResult parseSyncEventOpCommon(OpAsmParser &parser,
                                          OperationState &result,
                                          StringAttr pipeAttrName,
                                          StringAttr eventIdAttrName) {
  PipeAttr pipeAttr;
  if (succeeded(parser.parseOptionalLess())) {
    StringRef pipeTok;
    if (parser.parseKeyword(&pipeTok) || parser.parseGreater()) {
      return failure();
    }
    auto pipeOr = symbolizePIPE(pipeTok);
    if (!pipeOr) {
      return parser.emitError(parser.getCurrentLocation())
             << "unknown pipe token: " << pipeTok;
    }
    pipeAttr = PipeAttr::get(parser.getContext(), *pipeOr);
    result.addAttribute(pipeAttrName, pipeAttr);
  } else if (parser.parseAttribute(pipeAttr, pipeAttrName,
                                   result.attributes)) {
    return failure();
  }
  if (parser.parseComma()) {
    return failure();
  }

  OpAsmParser::UnresolvedOperand eventOperand;
  OptionalParseResult parseEventOperand =
      parser.parseOptionalOperand(eventOperand);
  if (parseEventOperand.has_value()) {
    if (failed(*parseEventOperand)) {
      return failure();
    }
    if (parser.resolveOperand(eventOperand, parser.getBuilder().getIndexType(),
                              result.operands)) {
      return failure();
    }
  } else {
    IntegerAttr eventAttr;
    if (parser.parseAttribute(eventAttr, parser.getBuilder().getI32Type(),
                              eventIdAttrName, result.attributes)) {
      return failure();
    }
  }

  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }
  return success();
}

static void printSyncEventOpCommon(OpAsmPrinter &p, Operation *op,
                                   PipeAttr pipeAttr, IntegerAttr eventAttr,
                                   Value eventDyn, StringRef pipeAttrName,
                                   StringRef eventIdAttrName) {
  p << " <" << stringifyPIPE(pipeAttr.getPipe()) << ">, ";
  if (eventAttr) {
    p << eventAttr.getInt();
  } else {
    p << eventDyn;
  }
  p.printOptionalAttrDict(op->getAttrs(), {pipeAttrName, eventIdAttrName});
}

static LogicalResult parsePTOShapeAndElement(OpAsmParser& parser, SmallVectorImpl<int64_t>& shape, Type& elementType)
{
    if (parser.parseLess() || parser.parseDimensionList(shape, /*allowDynamic=*/true) ||
        parser.parseType(elementType) || parser.parseGreater()) {
        return failure();
    }
    return success();
}
